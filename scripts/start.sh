#!/usr/bin/env bash
# Supervisor for the ming-tts uvicorn process (a single process; it spawns
# the model worker processes itself). Modeled on the Kokoro pod's
# start_kokoro.sh, without its memory watchdog and restart windows:
#
#   - sources /etc/ming.env if present (container env works just as well)
#   - downloads the model before uvicorn starts, with retries, so the
#     workers load from disk instead of racing one download; the same for
#     the forced-aligner weights when ALIGNER=mms_fa (into TORCH_HOME)
#   - restarts uvicorn when it exits, with exponential backoff (2 s doubling
#     to 60 s; reset once a run lasted longer than 60 s). Uvicorn exits with
#     143 after a graceful SIGTERM: it re-raises the signal once drained.
#   - optional proactive restart after MAX_UPTIME seconds (0 = off), with a
#     graceful drain: SIGTERM, then DRAIN_TIMEOUT s for in-flight requests
#   - forwards SIGTERM/SIGINT so `docker stop` drains the same way
#
# Works in the Docker image (/app, system Python) and in a /workspace
# checkout with a .venv (MING_PROJ, see boot_runpod.sh).
set -u

ENV_FILE=${MING_ENV_FILE:-/etc/ming.env}
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

PROJ=${MING_PROJ:-/app}
PORT=${PORT:-8080}
MAX_UPTIME=${MAX_UPTIME:-0}
DRAIN_TIMEOUT=${DRAIN_TIMEOUT:-60}
cd "$PROJ" || exit 1

if [ -x "$PROJ/.venv/bin/python" ]; then
    PY="$PROJ/.venv/bin/python"
else
    PY=$(command -v python3 || command -v python)
fi

log() { echo "[supervisor] $(date -u +%FT%TZ) $*"; }

# Best-effort Axiom event, so supervisor-side restarts are visible next to
# the app's own logs. Silent no-op without AXIOM_TOKEN/AXIOM_DATASET.
ship_axiom() {
    [ -n "${AXIOM_TOKEN:-}" ] && [ -n "${AXIOM_DATASET:-}" ] || return 0
    curl -sS --max-time 5 -X POST \
        -H "Authorization: Bearer $AXIOM_TOKEN" -H 'Content-Type: application/json' \
        --data "[{\"_time\":$(date +%s)000,\"level\":\"$2\",\"service\":\"ming-tts-supervisor\",\"msg\":\"$1\"${3:+,$3}}]" \
        "https://api.axiom.co/v1/datasets/$AXIOM_DATASET/ingest" >/dev/null 2>&1 || true
}

# Resolve MODEL_ID to a local directory: a baked /app/model, a directory
# given explicitly, or a Hub snapshot downloaded into HF_HOME. Exporting the
# local path means the workers make no network calls at load time.
resolve_model() {
    if [ -z "${MODEL_ID:-}" ] && [ -f /app/model/config.json ]; then
        MODEL_ID=/app/model
    fi
    MODEL_ID=${MODEL_ID:-inclusionAI/Ming-3.0-tts-0.5B}
    local delay=10 path
    while [ ! -d "$MODEL_ID" ]; do
        log "downloading $MODEL_ID into ${HF_HOME:-the default HF cache}"
        if path=$(MODEL_ID="$MODEL_ID" "$PY" -c \
                'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["MODEL_ID"]))') \
                && [ -d "$path" ]; then
            MODEL_ID=$path
        else
            log "model download failed, retrying in ${delay}s"
            ship_axiom "supervisor.model_download_failed" "WARN"
            sleep "$delay"
            delay=$((delay * 2 > 300 ? 300 : delay * 2))
        fi
    done
    export MODEL_ID
    log "model ready at $MODEL_ID"
}

# ALIGNER=mms_fa: fetch torchaudio's MMS_FA checkpoint into TORCH_HOME (the
# volume) before the aligner process needs it, so a flaky download is
# retried here instead of crash-looping the aligner. Failure is not fatal:
# the aligner keeps retrying from inside its process, and the API falls
# back to estimated timestamps until it is ready.
resolve_aligner() {
    [ "${ALIGNER:-off}" = "mms_fa" ] || return 0
    local attempt=1 delay=10
    while [ "$attempt" -le 5 ]; do
        log "fetching the MMS_FA aligner weights into ${TORCH_HOME:-the default torch cache}"
        if "$PY" -c 'import torchaudio; torchaudio.pipelines.MMS_FA.get_model()' >/dev/null 2>&1; then
            log "aligner weights ready"
            return 0
        fi
        log "aligner download failed (attempt $attempt), retrying in ${delay}s"
        ship_axiom "supervisor.aligner_download_failed" "WARN"
        sleep "$delay"
        delay=$((delay * 2 > 300 ? 300 : delay * 2))
        attempt=$((attempt + 1))
    done
    log "aligner weights not available yet; the aligner process will retry"
}

# SIGTERM, then wait for uvicorn to drain (--timeout-graceful-shutdown) and
# stop its workers; SIGKILL if it is still around after that.
terminate_child() {
    local pid=$1 waited=0
    kill -TERM "$pid" 2>/dev/null || return 0
    while [ "$waited" -lt $((DRAIN_TIMEOUT + 20)) ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        log "pid=$pid still alive after ${waited}s, SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
    fi
}

uptime_watchdog() {
    local pid=$1
    sleep "$MAX_UPTIME"
    if kill -0 "$pid" 2>/dev/null; then
        log "max uptime ${MAX_UPTIME}s reached, restarting pid=$pid"
        ship_axiom "supervisor.max_uptime_restart" "INFO" "\"max_uptime_s\":$MAX_UPTIME"
        terminate_child "$pid"
    fi
}

on_term() {
    log "received signal, draining"
    [ -n "${watchdog:-}" ] && kill -TERM "$watchdog" 2>/dev/null
    [ -n "${child:-}" ] && terminate_child "$child"
    exit 0
}
trap on_term TERM INT

resolve_model
resolve_aligner

backoff=2
while true; do
    started=$(date +%s)
    log "starting uvicorn on :$PORT (workers=${NUM_WORKERS:-3} aligner=${ALIGNER:-off} backoff=${backoff}s)"
    "$PY" -m uvicorn app.main:app \
        --host 0.0.0.0 --port "$PORT" --workers 1 \
        --timeout-keep-alive 30 --timeout-graceful-shutdown "$DRAIN_TIMEOUT" &
    child=$!
    watchdog=
    if [ "$MAX_UPTIME" -gt 0 ]; then
        uptime_watchdog "$child" &
        watchdog=$!
    fi

    wait "$child"
    code=$?
    child=
    if [ -n "$watchdog" ]; then
        kill -TERM "$watchdog" 2>/dev/null
        wait "$watchdog" 2>/dev/null
    fi

    ran=$(( $(date +%s) - started ))
    log "uvicorn exited code=$code after ${ran}s"
    if [ "$MAX_UPTIME" -gt 0 ] && [ "$ran" -ge "$MAX_UPTIME" ]; then
        # Planned restart from uptime_watchdog: come straight back.
        backoff=2
        continue
    fi
    ship_axiom "supervisor.uvicorn_exited" "WARN" "\"exit_code\":$code,\"uptime_s\":$ran"
    if [ "$ran" -lt 60 ]; then
        backoff=$((backoff * 2 > 60 ? 60 : backoff * 2))
    else
        backoff=2
    fi
    log "restarting in ${backoff}s"
    sleep "$backoff"
done
