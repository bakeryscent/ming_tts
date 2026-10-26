# bench: capacity measurement for the Ming TTS pod

Tools to answer one question: **how many people can listen at the same time,
without gaps, from one RTX A5000 pod, and what does an audio hour cost?**

| file | what |
|---|---|
| `loadtest.py` | asyncio + httpx concurrency sweep / sustained run, writes JSON + Markdown |
| `locustfile.py` | the same load in Locust, for its live UI and charts |
| `corpus.py` | 63 English sentences (42–249 chars) + 5 each for es/fr/de/zh |
| `tests/test_loadtest.py` | stats math and early-stop against an in-process fake server |

## Setup

```bash
cd ming_tts
python3.12 -m venv bench/.venv
bench/.venv/bin/pip install -r bench/requirements.txt          # loadtest.py
bench/.venv/bin/pip install -r bench/requirements-locust.txt   # + Locust
bench/.venv/bin/pip install -r bench/requirements-test.txt     # + pytest, fastapi
bench/.venv/bin/python -m pytest bench/tests -q

export MING_URL=https://<podid>-8080.proxy.runpod.net
export MING_TOKEN=<API_TOKEN of the pod>
```

Run the client from a machine close to where the Bun API runs; the numbers
include the network path. Each audio second is ~64 KB of base64 PCM at
24 kHz, so 10 audio s/s of throughput is ~5 Mbit/s down.

## loadtest.py

Sweep (quick look, 24 requests per level, stops at the first failing level):

```bash
bench/.venv/bin/python bench/loadtest.py --concurrency 1,2,3,4,6,8,12,16 --requests 48 \
  --out bench/sweep_w3.json --md bench/sweep_w3.md
```

Sustained (hold one level for 10 minutes):

```bash
bench/.venv/bin/python bench/loadtest.py --concurrency 6 --duration 600 --warmup 0 \
  --out bench/sustained_c6_w3.json --md bench/sustained_c6_w3.md
```

Other flags: `--voice US_James`, `--language en|es|fr|de|zh`,
`--sample-rate 24000|44100`, `--timeout 120`, `--warmup 6`, `--rtf-target 0.8`,
`--max-error-rate 0.05`, `--no-early-stop`, `--gpu-hourly 0.16`. `--help` lists all.

Each level runs `c` closed-loop clients: every client sends the next corpus
sentence as soon as its previous one returns. All clients share one keep-alive
`httpx.AsyncClient`. A `429` is counted, then retried once after `Retry-After`;
the retry wait is part of that request's latency.

With a fixed `--requests N`, the last requests of a level run with fewer than
`c` in flight (drain tail), which understates throughput when `N` is close to
`c`. Use `N >= 4c` or `--duration` for the numbers you act on.

### Reading the table

| column | meaning |
|---|---|
| `c` | concurrent clients (requests in flight from this tool) |
| `n` / `ok` | requests sent / answered 200 with valid audio |
| `429 raw / retried-ok` | requests whose first attempt got 429 / of those, how many the single retry saved |
| `other err` | final failures other than 429: 5xx, 504, `transport` (timeouts, resets), `invalid-body` |
| `p50/p95/p99 s` | wall latency of successful requests, client side, including queue and retry wait |
| `RTF mean / p95` | per request `latency / audio duration`. **The gate.** < 1 means the sentence was ready before an equally long one finished playing |
| `audio s/s` | audio seconds produced per wall second at this level (throughput) |
| `chars/s` | input characters synthesized per wall second |
| `queue ms` | mean server-reported `queue_ms` (time waiting for a free worker) |
| `srv RTF` | mean server `runtime_ms / audio`: pure synthesis speed, without queue or network |
| `in-flight max` | peak concurrent requests this tool had open (sanity check that `c` was reached) |
| `workers` | successful requests per server worker id (should be balanced) |

The JSON adds per-level `littles_in_flight`, `error_rate`, `failures`,
`duration_mismatches` (server `duration_seconds` vs decoded PCM length), the
`/health` snapshot, `/stats` after the run, and every request record.

How it typically looks as `c` grows: while `c <= NUM_WORKERS`, latency stays
flat and throughput grows linearly. Past that, requests queue: `queue ms`
and RTF climb while `audio s/s` flattens. Past `NUM_WORKERS + MAX_QUEUE`,
429s appear. The capacity is the knee: the last level that still adds
throughput while p95 RTF stays under target.

The sweep stops after a level whose p95 RTF exceeds `--rtf-target` or whose
error rate (requests not served on the first attempt) exceeds
`--max-error-rate`. `--no-early-stop` runs all levels anyway, e.g. to see
where admission control starts returning 429.

## Capacity model

**Why RTF < 1 is the gate.** The Bun API asks for one sentence at a time and
plays them back to back at 1x. If it requests sentence *n+1* when sentence *n*
starts playing, *n+1* must arrive within the playback time of *n*. Sentence
lengths in a passage are similar on average, so the condition per request
is `latency < its own audio duration`, i.e. `RTF < 1`. The target is applied
to **p95** with margin (default 0.8) because a short sentence followed by a
long one, random arrivals and new listeners starting all eat into it.
Time-to-first-audio for a new listener is the plain latency, read the p95
column for that.

**Listeners from throughput.** A listener consumes exactly 1 audio second
per wall second. A pod that produces `A` audio seconds per second can
therefore feed `A` continuous listeners:

```
listeners ≈ A = throughput_audio_s_per_s
```

**Little's law (consistency check).** With `X` successful requests/s and
mean latency `R`, requests in flight are `N = X × R`. Throughput in audio
is `A = X × mean_audio`, so

```
N = A × R / mean_audio          (littles_in_flight in the JSON, ≈ c)
```

A real listener only issues `1 / mean_audio` requests per second, so it keeps
`R / mean_audio` (the mean RTF) requests in flight. `A` listeners keep
`A × R / mean_audio = c` requests in flight: the same depth the benchmark
held at that level, so they see the same latency distribution. That is why
the greedy closed-loop clients of this tool translate directly into
listeners via throughput, not via `c`.

**Best level.** Among levels with p95 RTF <= target and zero 429/5xx, walk up
in `c` and keep the level that raises throughput by more than 5% over the
best so far. Extra concurrency that only adds queueing is not capacity.

**Headroom.** `1 - p95 RTF` is the share of a sentence's playback time left
over at p95. At 0.6 the next sentence is ready with 40% of the current one
still to play. That slack, not extra listeners, is what absorbs bursts:
the estimate assumes steady load, so the Bun router should still treat 429
as "fall back to another provider".

**Cost.** At GPU price `P` USD/h:

```
USD per 1000 audio s = P / 3600 / A × 1000
USD per 1M chars     = P / 3600 / chars_per_s × 1e6
```

Both assume the pod runs at that throughput all the time; idle hours cost
the same.

## Measurement protocol

1. **Warm up.** Wait for `GET /health` to return `ok: true` with all workers
   loaded. `loadtest.py` sends `--warmup` unmeasured requests (default 6) so
   every worker has compiled kernels and caches before the sweep.
2. **Sweep.** Run the sweep above with `--requests 48` (or `--duration 60`).
   Note the best level and its throughput from the capacity block.
3. **Sustained.** Hold the chosen `c` for 10 minutes (`--duration 600
   --warmup 0`). While it runs, on the pod:
   ```bash
   nvidia-smi dmon -s um -d 5        # sm% (GPU busy), mem% and fb (VRAM MB)
   watch -n 10 'curl -s -H "Authorization: Bearer $API_TOKEN" localhost:8080/stats'
   ```
   Pass if p95 RTF stays under target with zero 429/5xx, `/stats` p95
   does not drift upward, and VRAM is flat (growth means a leak or
   fragmentation that will eventually OOM a worker).
4. **Pick the replica count.** Restart the pod with `NUM_WORKERS=2`, then
   `3`, then `4`, repeating steps 1–3 for each (name the outputs `_w2`,
   `_w3`, `_w4`). Choose the count with the highest sustained throughput that
   passes. If `sm%` is already ~100 at fewer workers, more replicas only
   time-slice the GPU and raise per-request latency; if VRAM gets close to
   24 GB, the extra replica is not worth the OOM risk.

Run step 2 once with `--language es` (or the language mix you serve) as well:
non-English routes through a different Ming task and may be slower.

## RunPod notes

- The HTTP proxy (`https://<podid>-8080.proxy.runpod.net`) sits behind
  Cloudflare, which cuts requests at ~100 s (HTTP 524, reported here as a
  5xx). It also adds some latency per request. For pure server numbers,
  expose 8080 as a TCP port and use `http://<public-ip>:<mapped-port>`.
- Measure through whichever path the Bun API will use in production; that is
  the latency listeners get.

## Locust

Same load, with Locust's live charts:

```bash
MING_TOKEN=... bench/.venv/bin/locust -f bench/locustfile.py \
  --host https://<podid>-8080.proxy.runpod.net
# headless: add --headless -u 6 -r 1 -t 10m --csv bench/locust_c6
```

`-u` is the concurrency `c`. Env `MING_VOICE`, `MING_LANGUAGE`,
`MING_SAMPLE_RATE` change the request. Besides the normal `POST /v1/tts`
row, every success records `[SERVER_RUNTIME]` and `[QUEUE]` (ms, from
`inference_status`) and `[RTF x1000]` (1000 = real time, so the p95 column
should stay under 800). Locust's "Aggregated" row mixes these together;
read the named rows. Throughput in audio s/s and the capacity estimate come
only from `loadtest.py`.
