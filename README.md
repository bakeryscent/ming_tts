# ming_tts — Ming-3.0-tts-0.5B self-hosted su RunPod

Server HTTP (FastAPI + pool di processi GPU) che espone [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) alla tua `tts_api`, con lo stesso stile del pod Kokoro (`tts_kokoro_api`): bearer token, JSON in, PCM base64 out, log JSON → Axiom, supervisor con restart, immagine Docker pubblicata da GitHub Actions su ghcr.io.

| Documento | Contenuto |
|---|---|
| [docs/RUNPOD.md](docs/RUNPOD.md) | Scelte (GPU, template, dischi, costi) e procedura di deploy passo-passo |
| [docs/API.md](docs/API.md) | Contratto HTTP: `/v1/tts`, `/v1/voices`, clone, `/health`, `/stats`, env |
| [bench/README.md](bench/README.md) | Load test, modello di capacità (quanti ascoltatori regge il pod), protocollo di misura |

## Layout
```
app/        server: main.py (routes) · pool.py (coda + worker) · worker.py (processo GPU)
            engine.py (request → MingTTS) · voices.py · timestamps.py (stima) · audio.py
            aligner.py (forced alignment MMS_FA) · aligner_service.py (il suo processo)
scripts/    start.sh (supervisor, CMD dell'immagine) · boot_runpod.sh · smoke.sh
bench/      loadtest.py (sweep di concorrenza) · locustfile.py · corpus.py
voices/     voci clone di esempio (lingguang, DB30) + catalogo voci cinesi builtin
tests/      98 test; il pool gira con FAKE_ENGINE=1 e l'aligner con ALIGNER=fake, senza GPU né torch
Dockerfile  pytorch 2.6 / CUDA 12.4 + Ming @ commit pinnato + flash-attn + pynini + uroman
```

## Sviluppo locale (Mac, senza GPU)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
MING_DIR=/path/to/Ming-3.0-tts .venv/bin/python -m pytest -q     # senza MING_DIR i test engine sono skip
API_TOKEN=dev FAKE_ENGINE=1 NUM_WORKERS=2 .venv/bin/uvicorn app.main:app --port 8080
```

## Deploy
1. Push su GitHub → la workflow pubblica `ghcr.io/<owner>/ming_tts:latest`.
2. RunPod: template con quell'immagine, A5000, container 25 GB, volume 15 GB su `/workspace`, porta HTTP 8080, env `API_TOKEN` (+ `NUM_WORKERS=3`). Dettagli in `docs/RUNPOD.md`.
   Timestamp parola: stimati di default; `ALIGNER=mms_fa` accende il forced aligner (un processo in più, ~1.2 GB VRAM) e le risposte passano a `words_source: "aligned"`.
3. `URL=https://<podid>-8080.proxy.runpod.net API_TOKEN=... bash scripts/smoke.sh`
4. `MING_URL=... MING_TOKEN=... bench/.venv/bin/python bench/loadtest.py --concurrency 1,2,3,4,6,8,12,16 --requests 48`

## Cosa non è ancora fatto
- Il modello non è mai stato eseguito su GPU da questa macchina: la prima verifica reale è lo smoke test sul pod.
- L'aligner (`ALIGNER=mms_fa`) non è ancora stato misurato su GPU: VRAM e `align_ms` reali vanno letti dal pod (`docs/RUNPOD.md`, passo 6) prima di accenderlo in produzione.
- Il provider `ming.ts` nel repo `tts_api` (router, fallback, billing) va scritto a parte.
- Ottimizzazioni di fase 2 (CUDA graphs sul DiT, MPS, batching) solo dopo le misure.
