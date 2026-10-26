# Deploy su RunPod — Ming-3.0-tts-0.5B su RTX A5000

## Decisioni

| Scelta | Valore | Perché |
|---|---|---|
| GPU | RTX A5000 24 GB, **Community Cloud** | $0.16/h (~$117/mese) contro $0.27/h Secure. Il modello pesa 3 GB in bf16: 24 GB bastano per 3–4 repliche. Ampere sm_86 è supportato da flash-attn 2.7.4. |
| Template | **Immagine Docker nostra** (`ghcr.io/<owner>/ming_tts`), non "PyTorch 2.4.0" | Ming richiede torch 2.6.0 + transformers 4.46.3 + pynini + flash-attn: sul template stock dovresti reinstallare torch a ogni ricreazione del pod. L'immagine è riproducibile, parte in 1–2 minuti e la CI la ricostruisce a ogni push (stesso flusso del pod Kokoro). |
| Container disk | **25 GB** | Immagine ~12 GB decompressa + cache pip/temp. $0.10/GB/mese → $2.5/mese. 20 GB è al limite. |
| Volume disk (pod) | **15 GB** montato su `/workspace` | Tiene la cache HuggingFace (3 GB di pesi) e le voci clonate: sopravvive a stop/start e ricreazione dell'immagine. $1.5/mese. |
| Network volume | **No** | Vincola il pod a un datacenter e riduce la disponibilità di A5000. Serve solo se vuoi più pod che condividono le voci clonate. |
| Pod vs Serverless | **Pod on-demand** | Serverless A5000 costa $0.69/h attivo + cold start (~40 s per caricare 3 repliche): conviene solo sotto il 25% di utilizzo. Con traffico continuo il pod flat vince, e hai già il router con fallback su DeepInfra per i buchi. |
| Concorrenza | `NUM_WORKERS=3` processi, 1 modello ciascuno | Il decode di Ming è batch-size-1 e limitato dai lanci kernel: più processi riempiono la GPU in time-slicing. Ogni replica occupa ~4–5 GB (3 GB pesi + attivazioni + contesto CUDA): 3 repliche ≈ 15 GB, 4 ≈ 20 GB sui 24 disponibili. 2/3/4 va scelto con il load test. |

## Costo atteso
A $0.16/h: **$117/mese** flat. Il load test stampa $/1000 secondi audio e $/1M caratteri alla capacità misurata.

## Procedura

### 1. Repo e immagine
1. Crea il repo GitHub (es. `bakeryscent/ming_tts`), push di questa cartella.
2. La workflow `.github/workflows/docker.yml` pubblica `ghcr.io/<owner>/ming_tts:latest` (+ tag sha). Rendi il package **public** in GitHub → Packages, oppure crea un *Registry credential* su RunPod (Settings → Container Registry Auth) con un PAT `read:packages`.
3. Opzionale: `workflow_dispatch` con `bake_model=true` produce un'immagine con i pesi dentro (~15 GB): avvio più veloce ma pull più lento. Default: i pesi si scaricano al primo avvio nel volume.

### 2. Template RunPod
Pods → Templates → New Template:
- Container image: `ghcr.io/<owner>/ming_tts:latest`
- Container disk: 25 GB · Volume disk: 15 GB · Volume mount path: `/workspace`
- Expose HTTP ports: `8080` · Expose TCP: `22` (opzionale, SSH)
- Container start command: vuoto (usa il `CMD` dell'immagine)
- Environment variables:
  ```
  API_TOKEN=<token lungo e casuale>
  NUM_WORKERS=3
  MAX_QUEUE=16
  REQUEST_TIMEOUT_S=90
  HF_HOME=/workspace/hf
  VOICES_DIR=/workspace/voices
  AXIOM_TOKEN=...        (opzionale, come il pod Kokoro)
  AXIOM_DATASET=...
  ```

### 3. Pod
Deploy → GPU **RTX A5000** → Community Cloud → scegli il template → Deploy.
Primo avvio: pull immagine (~2 min) + download pesi (3 GB, ~1 min) + load di 3 repliche e warmup (~1 min). `GET https://<podid>-8080.proxy.runpod.net/health` risponde 503 finché non è pronto, poi 200.

### 4. Verifica
```bash
URL=https://<podid>-8080.proxy.runpod.net API_TOKEN=<token> bash scripts/smoke.sh
```

### 5. Misura il carico
Vedi `bench/README.md`. In breve: sweep di concorrenza 1→16, poi 10 minuti sostenuti al livello scelto; ripeti con `NUM_WORKERS=2`, `3`, `4` e tieni quello con più secondi-audio/secondo con p95 RTF < 0.8. Guarda `nvidia-smi dmon -s um` via SSH durante il test: se SM% resta sotto ~70% con 4 worker, è il momento delle ottimizzazioni di fase 2.

## Fase 2 (dopo le misure, in ordine di resa/costo)
1. **CUDA graphs / torch.compile sul passo DiT** (10 step di flow matching × 8 layer per ogni 80 ms di audio: è il collo di bottiglia dei lanci kernel). Stimato 1.5–2× sul throughput per replica.
2. **CUDA MPS** nel container (`nvidia-cuda-mps-control -d` prima di uvicorn): fa girare davvero in parallelo i kernel dei processi worker invece del time-slicing. Funziona su molti host RunPod ma non su tutti: va provato.
3. **Decode batched** (batch > 1 nello stesso processo): il guadagno maggiore ma richiede modificare `modeling_bailingmm.sample` (padding, position ids, latent history per riga). Solo se 1+2 non bastano.
4. **Forced alignment** (torchaudio MMS_FA) per timestamp parola reali, se la sottolineatura stimata non basta.

## Note operative
- Il proxy RunPod aggiunge ~100–400 ms al primo handshake TLS: il client Bun usa keep-alive (come per Kokoro). Una frase da 3 s con RTF 0.3 risponde in ~1 s netto.
- Il proxy RunPod chiude le richieste HTTP oltre ~100 s: `REQUEST_TIMEOUT_S=90` sta sotto. Il Bun API spezza già i testi in frasi, `MAX_TEXT_CHARS=1200` è una rete di sicurezza.
- Stop del pod = container disk perso, volume conservato: al riavvio niente nuovo download dei pesi.
- Se la A5000 Community non è disponibile nella regione: RTX 3090 (24 GB, $0.22/h) o A4000/A4500 (16/20 GB, 2 worker) sono i sostituti drop-in; RTX 4090 ($0.34/h) ha ~2× la velocità per replica.
