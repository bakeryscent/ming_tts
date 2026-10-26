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
audio seconds generated, prompt-cache size, uptime, GPU name/VRAM used.

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
  "words_source": "estimated",
  "inference_status": {"status": "succeeded", "cost": 0.0, "runtime_ms": 1310,
                       "queue_ms": 40, "tokens_input": 42, "fragments": 1, "worker": 2}
}
```
- `words`: Ming has no alignment head. Phase 1 returns **estimated**
  timestamps: audio duration (minus trailing silence) spread over the words
  proportionally to a per-word weight (letters + punctuation pause bonus),
  per fragment so multi-fragment utterances stay anchored. Same `Word`
  shape as Kokoro/DeepInfra. `words_source` is `"estimated"` so the Bun API
  can decide how to use them; a forced aligner can upgrade this later.
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
| `PORT` | `8080` | |
| `AXIOM_TOKEN` / `AXIOM_DATASET` | unset | structured log shipping (same as kokoro pod) |
