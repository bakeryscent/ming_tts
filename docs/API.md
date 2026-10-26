# ming-tts-api — HTTP contract (v1)

Self-hosted Ming-3.0-tts-0.5B server for RunPod. Mirrors the shape of the
self-hosted Kokoro pod (`tts_kokoro_api`) so the Bun `tts_api` router can
add it as a provider with the same plumbing: bearer token, JSON in, base64
PCM out, `inference_status`, `words`.

All endpoints except `/health` require `Authorization: Bearer <API_TOKEN>`.
Errors are JSON `{"detail": "<message>"}` with the HTTP status below.

## GET /health
No auth. `200 {"ok": true, "workers": {"total": 3, "busy": 1}, "queue": 0, "model_loaded": true}`.
Returns `503 {"ok": false, ...}` while workers are still loading the model.

## GET /stats  (auth)
Counters for observability: requests, errors, timeouts, rejected (429),
queue depth, busy workers, p50/p95 latency over the last N requests,
audio seconds generated, prompt-cache size, uptime, GPU name/VRAM used,
and `aligner` (`{"backend": "off"}`, or the aligner process's state, pid,
restarts, jobs done, fallbacks, queue and VRAM when `ALIGNER` is enabled).

## GET /v1/voices  (auth)
```json
{"voices": [
  {"id": "US_James",   "kind": "builtin", "language": "en", "gender": "male",   "description": "narration, explainer and documentary reads"},
  {"id": "lingguang",  "kind": "clone",   "language": "zh", "prompt_seconds": 3.3},
  {"id": "clone_7f3a", "kind": "clone",   "language": "en", "prompt_seconds": 6.1, "created_at": "..."}
]}
```
`builtin` voices are Ming "IP" voices (task `IP`, `use_zero_spk_emb`):
US_Emma, US_James, US_Stella, UK_George, UK_Charlotte, plus the Chinese
catalog in the Ming cookbook. `clone` voices are a reference wav + its
transcript stored under `VOICES_DIR` (`<id>.wav` + `<id>.json`).

## POST /v1/voices/clone  (auth, multipart/form-data)
Fields: `audio` (wav/mp3/flac, 1–20 s, mono or stereo), `prompt_text`
(exact transcript of the audio, required), `language` (`en`|`zh`|...),
optional `id` (slug; server generates `clone_<8hex>` otherwise).
Stores the reference under `VOICES_DIR`, returns the voice object above.
`400` on bad audio/length, `409` if `id` exists.

## DELETE /v1/voices/{id}  (auth)
Removes a `clone` voice. `404` if missing, `405` for `builtin`.

## POST /v1/tts  (auth, application/json)
```json
{
  "text": "The garden will remain open until sunset.",
  "voice": "US_James",
  "language": "en",
  "sample_rate": 24000,
  "output_format": "pcm",
  "return_timestamps": true,
  "instruction": null,
  "controls": null,
  "seed": null,
  "cfg": 2.0,
  "sigma": 0.25,
  "temperature": 0.0,
  "max_length": 24
}
```
- `text`: required, 1..`MAX_TEXT_CHARS` (env, default 1200). Longer → `413`.
  Callers (the Bun API) already split text into sentences; the server
  additionally splits into Ming fragments (`max_length`) internally.
- `voice`: a voice id from `/v1/voices`. Mutually exclusive with `style`.
- `style`: voice-design description (task `VOICE_DESIGN`); no `voice`.
- `instruction`: free-form delivery instruction (task `FREEFORM`), no `voice`
  required (the model is better without a reference here).
- `language`: `en`, `zh`, `de`, `es`, `fr`, `ko`, `ja`; `auto` = detect
  zh vs en from the text. Non-zh/en languages route to task `MULTILINGUAL`.
- `controls`: optional, one of the Ming structured control sets:
  `{"emotion": "高兴"}` (task `EMOTION`), `{"speed": "快速", "pitch": "中", "volume": "中"}`
  (task `BASIC`), `{"dialect": "cantonese"}` (task `DIALECT`). Only valid
  with a `clone` voice (they need a reference). `400` on invalid combos
  (the server reuses `tts_tasks.build_task_generation_kwargs` validation).
- `sample_rate`: 24000 (default, halves payload vs native) or 44100 (native).
- `output_format`: `pcm` (s16le, base64) or `wav` (base64 RIFF).
- `seed`: int or null (null → fixed default 1895, deterministic like `infer.py`).

Response `200`:
```json
{
  "audio": "<base64>",
  "output_format": "pcm",
  "sample_rate": 24000,
  "duration_seconds": 2.84,
  "words": [{"id": 0, "start": 0.00, "end": 0.21, "text": "The"}, ...],
  "words_source": "aligned",
  "inference_status": {"status": "succeeded", "cost": 0.0, "runtime_ms": 1310,
                       "queue_ms": 40, "tokens_input": 42, "fragments": 1, "worker": 2,
                       "align_ms": 35}
}
```
- `words`: same `Word` shape as Kokoro/DeepInfra, one entry per
  whitespace-delimited word (one per character for Chinese/Japanese), with
  attached punctuation, in the caller's spelling (`"42"`, not `"forty two"`).
  Times are seconds, rounded to ms, monotonic and non-overlapping.
  `null` and `words: []` when `return_timestamps` is false.
- `words_source` tells the Bun API how much to trust them:
  - `"aligned"`: forced alignment (`ALIGNER=mms_fa`) placed every word on the
    audio. Words the aligner cannot hear as letters (digits, symbols) take the
    gap between their aligned neighbours. Expect Kokoro-grade accuracy
    (~20–50 ms on word boundaries).
  - `"estimated"`: Ming has no alignment head; without an aligner (the
    default, `ALIGNER=off`) the audio duration (minus trailing silence) is
    spread over the words proportionally to a per-word weight (letters +
    punctuation pause bonus), per fragment so multi-fragment utterances stay
    anchored. Also returned when the aligner is enabled but did not answer
    for this request: still loading, crashed, timed out
    (`ALIGNER_TIMEOUT_S`) or failed on at least one fragment (the other
    fragments keep their aligned times, the failed one is estimated).
- `inference_status.align_ms`: wall time of the alignment step, queue
  included (`null` when `ALIGNER=off` or timestamps were not requested;
  near 0 when the aligner was skipped because it is not ready). It is added
  to the request's latency after synthesis; see "Forced alignment" below.
- Task routing (server-side):
  | input                      | Ming task       | conditioning                       |
  |----------------------------|-----------------|------------------------------------|
  | `voice` = builtin          | `IP`            | `controls={"ip": id}`, zero spk emb |
  | `voice` = clone            | `TTS`           | prompt wav + prompt text, spk emb  |
  | `voice` = clone, lang≠zh/en| `MULTILINGUAL`  | + `language_id`                    |
  | `style`                    | `VOICE_DESIGN`  | zero spk emb                       |
  | `instruction`              | `FREEFORM`      | zero spk emb                       |
  | `controls.emotion`         | `EMOTION`       | clone voice required               |
  | `controls.speed/pitch/vol` | `BASIC`         | clone voice required               |
  | `controls.dialect`         | `DIALECT`       | clone voice required               |

Status codes: `400` validation, `401` token, `404` voice, `413` text too
long, `429` queue full (`Retry-After: 2`), `503` not ready, `504` request
timed out (`REQUEST_TIMEOUT_S`, default 90), `500` synthesis error.

## Concurrency model
A pool of `NUM_WORKERS` processes (default 3 on a 24 GB A5000), each
owning one model copy and CUDA context. The FastAPI process holds a
bounded FIFO queue (`MAX_QUEUE`, default 16); requests over the cap get
`429` immediately so the Bun router can fall back to another provider
instead of piling up. One request occupies one worker end-to-end.

## Forced alignment (`ALIGNER=mms_fa`)
Optional, off by default until measured on the pod. One extra process
(not one per worker) loads torchaudio's `MMS_FA` CTC aligner, a wav2vec2
model trained on 1,100+ languages over romanized text, and serves the whole
pool one job at a time. For every request with `return_timestamps`, after
synthesis, the API sends it the PCM and the words; the aligner romanizes
the words with `uroman` (pinyin, romaji, Latin for hangul, accents
stripped), resamples the audio to 16 kHz, aligns each Ming fragment on its
own slice of audio and returns word start/end in seconds.

Cost: ~1.2 GB of VRAM (fp32 weights + activations) on top of the replicas,
roughly one quarter of a Ming replica; plan `NUM_WORKERS=3` + aligner ≈
16–17 GB. Latency: tens of ms of GPU time for a 3 s sentence (wav2vec2
forward + Viterbi), plus the pipe round trip, serialised across the pool;
`align_ms` in every response reports the real figure, and the load test
(`bench/loadtest.py`, default on; `--no-timestamps` to compare) shows it
per level. Alignment never fails a request: past `ALIGNER_TIMEOUT_S`
(default 5 s), or when the process is loading, dead or errors, the
response carries the estimate and `words_source: "estimated"`. The aligner
process is restarted with backoff when it dies, like a worker. Weights
(~1.2 GB) are downloaded by `scripts/start.sh` into `TORCH_HOME`
(`/workspace/hf/torch`, the volume) on first start, never baked into the
image. Languages: all seven Ming languages are supported.

## Env
| var | default | meaning |
|---|---|---|
| `API_TOKEN` | required | bearer token |
| `MODEL_ID` | `inclusionAI/Ming-3.0-tts-0.5B` | HF repo or local dir |
| `HF_HOME` | `/workspace/hf` | HF cache (put on the volume disk) |
| `VOICES_DIR` | `/workspace/voices` | clone store (volume disk) |
| `NUM_WORKERS` | `3` | model replicas |
| `MAX_QUEUE` | `16` | queued requests before 429 |
| `REQUEST_TIMEOUT_S` | `90` | per request, queue + synth |
| `MAX_TEXT_CHARS` | `1200` | 413 above |
| `TEXT_NORMALIZATION` | `1` | pynini FST TN on/off |
| `ALIGNER` | `off` | `mms_fa` enables forced word alignment (`words_source: "aligned"`); `fake` is the test stand-in |
| `ALIGNER_TIMEOUT_S` | `5` | alignment deadline per request (queue + GPU); past it the estimate is returned |
| `TORCH_HOME` | `/workspace/hf/torch` | torchaudio cache for the aligner weights (volume disk) |
| `PORT` | `8080` | |
| `AXIOM_TOKEN` / `AXIOM_DATASET` | unset | structured log shipping (same as kokoro pod) |
