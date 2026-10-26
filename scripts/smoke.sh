#!/usr/bin/env bash
# Smoke test against a running server:
#   URL=https://<pod-id>-8080.proxy.runpod.net API_TOKEN=... bash scripts/smoke.sh
# Checks /health and /v1/voices, synthesizes one sentence into $OUT (wav),
# then uploads a clone voice, speaks with it into $CLONE_OUT and deletes it.
set -euo pipefail

URL=${URL:-http://localhost:8080}
: "${API_TOKEN:?set API_TOKEN}"
OUT=${OUT:-out.wav}
CLONE_OUT=${CLONE_OUT:-clone_out.wav}
REF=${REF:-$(dirname "$0")/../voices/DB30.wav}
AUTH="Authorization: Bearer $API_TOKEN"
TMP=$(mktemp -d)
CREATED=
cleanup() {
    rm -rf "$TMP"
    # Remove the smoke clone if a later step failed before deleting it.
    [ -z "$CREATED" ] || curl -fsS -X DELETE -H "$AUTH" "$URL/v1/voices/$CREATED" >/dev/null || true
}
trap cleanup EXIT

echo "== GET /health"
curl -fsS "$URL/health"
echo

echo "== GET /v1/voices"
curl -fsS -H "$AUTH" "$URL/v1/voices" | python3 -c '
import json, sys
voices = json.load(sys.stdin)["voices"]
kinds = {}
for voice in voices:
    kinds[voice["kind"]] = kinds.get(voice["kind"], 0) + 1
print(len(voices), "voices", kinds)
print("english builtins:", [v["id"] for v in voices if v["kind"] == "builtin" and v["language"] == "en"])
print("clones:", [v["id"] for v in voices if v["kind"] == "clone"])'

# Decode a /v1/tts response: write the PCM as a wav, print timings.
decode() {
    python3 -c '
import base64, json, sys, wave
body = json.load(open(sys.argv[1]))
pcm = base64.b64decode(body["audio"])
with wave.open(sys.argv[2], "wb") as out:
    out.setnchannels(1); out.setsampwidth(2); out.setframerate(body["sample_rate"])
    out.writeframes(pcm)
print(sys.argv[2], body["duration_seconds"], "s,", len(body["words"]), "words")
print("inference_status:", body["inference_status"])' "$1" "$2"
}

echo "== POST /v1/tts -> $OUT"
curl -fsS -H "$AUTH" -H "Content-Type: application/json" \
    -d '{"text": "The garden will remain open until sunset, provided the weather stays clear.", "voice": "US_James", "sample_rate": 24000, "output_format": "pcm"}' \
    "$URL/v1/tts" -o "$TMP/tts.json"
decode "$TMP/tts.json" "$OUT"

VOICE_ID="smoke_$(date +%s)"
echo "== POST /v1/voices/clone ($VOICE_ID from $REF)"
curl -fsS -H "$AUTH" \
    -F "audio=@$REF;type=audio/wav" -F "prompt_text=感谢你的认可。" -F "language=zh" -F "id=$VOICE_ID" \
    "$URL/v1/voices/clone"
CREATED=$VOICE_ID
echo

echo "== POST /v1/tts with $VOICE_ID -> $CLONE_OUT"
curl -fsS -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"text\": \"雨后的山谷很安静，远处传来几声鸟鸣。\", \"voice\": \"$VOICE_ID\"}" \
    "$URL/v1/tts" -o "$TMP/clone.json"
decode "$TMP/clone.json" "$CLONE_OUT"

echo "== DELETE /v1/voices/$VOICE_ID"
curl -fsS -X DELETE -H "$AUTH" "$URL/v1/voices/$VOICE_ID"
CREATED=
echo
