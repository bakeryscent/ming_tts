# ming-tts-api: Ming-3.0-tts-0.5B behind a FastAPI worker pool (docs/API.md).
#
#   docker build -t ming-tts-api .
#   docker build --build-arg BAKE_MODEL=1 -t ming-tts-api:baked .
#
# Model weights: by default they are NOT in the image. scripts/start.sh
# downloads them on first start into HF_HOME (/workspace/hf, the RunPod
# volume), so the image stays ~10 GB and every later start reuses the volume
# copy. BAKE_MODEL=1 snapshots them into /app/model at build time instead
# (bigger image, no download at start); start.sh uses /app/model when present.
# The forced-aligner weights (ALIGNER=mms_fa, torchaudio MMS_FA, ~1.2 GB)
# are never baked: start.sh fetches them into TORCH_HOME under the same
# volume when the aligner is enabled.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# build-essential: torch.compile (MING_COMPILE) builds kernels with a C compiler on first use.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      libsndfile1 \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ming is not a pip package: vendor the repo at a pinned commit.
ARG MING_REPO=https://github.com/inclusionAI/Ming-3.0-tts
ARG MING_COMMIT=028c1eace193e7acdb256453da89b6e9207e052b
RUN git init -q /app/ming \
    && git -C /app/ming remote add origin "$MING_REPO" \
    && git -C /app/ming fetch -q --depth 1 origin "$MING_COMMIT" \
    && git -C /app/ming checkout -q FETCH_HEAD \
    && rm -rf /app/ming/.git

# Ming-3.0-tts-0.5b runtime, pinned as in its requirements.txt. torch and
# torchaudio already match the base image, so pip leaves them in place.
RUN pip install --no-cache-dir \
      torch==2.6.0 \
      torchaudio==2.6.0 \
      transformers==4.46.3 \
      tokenizers \
      accelerate==1.3.0 \
      huggingface_hub \
      numpy \
      soundfile==0.12.1 \
      tqdm \
      loguru \
      x_transformers==2.9.2 \
      diffusers==0.33.0 \
      onnxruntime \
      inflect \
      einops \
      packaging

# FST text normalization (TEXT_NORMALIZATION=1); PyPI has a cp311 manylinux wheel.
RUN pip install --no-cache-dir pynini==2.1.6

# Romanizer for the forced aligner (ALIGNER=mms_fa): MMS_FA aligns Latin
# text only, uroman turns Chinese/Japanese/Korean (and accents) into it.
# Pure Python, data files included; torchaudio already ships the pipeline.
RUN pip install --no-cache-dir uroman==1.3.1.1

# Optional: Ming falls back to PyTorch SDPA without flash-attn, so a missing
# or incompatible wheel must not fail the build.
ARG FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir "$FLASH_ATTN_URL" \
    || echo "WARN: flash-attn not installed; Ming will use PyTorch SDPA"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG BAKE_MODEL=0
RUN if [ "$BAKE_MODEL" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('inclusionAI/Ming-3.0-tts-0.5B', local_dir='/app/model')"; \
    fi

COPY app/ ./app/
COPY voices/ ./voices/
COPY scripts/ ./scripts/

# glibc allocator tuning, carried over from the Kokoro pod. Every synthesis
# allocates transient buffers of a few hundred kB to a few MB (float tensor ->
# numpy -> int16 -> pickled pipe message -> base64). glibc mmaps blocks over
# 128 KiB and returns them on free, but a dynamic threshold ratchets up after
# the first large free; from then on such buffers come from arena heaps whose
# freed space is never returned to the OS (measured on the Kokoro pod: ~1.5 MiB
# of RSS retained per request). Setting MMAP/TRIM_THRESHOLD_ explicitly
# disables the ratchet; MALLOC_ARENA_MAX caps the number of per-thread arenas.
# expandable_segments lets each replica's CUDA allocator grow and shrink
# without fragmenting the card the replicas share.
ENV HF_HOME=/workspace/hf \
    TORCH_HOME=/workspace/hf/torch \
    VOICES_DIR=/workspace/voices \
    MING_DIR=/app/ming \
    ALIGNER=off \
    PORT=8080 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=131072 \
    MALLOC_TOP_PAD_=131072

EXPOSE 8080

# /health is 503 until every replica has loaded and warmed up; the first
# start may also download the model, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["bash", "/app/scripts/start.sh"]
