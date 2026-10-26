#!/bin/bash
# RunPod "Container Start Command" for the stock RunPod PyTorch image, as an
# alternative to our Docker image (whose CMD already runs start.sh).
# Layout on the volume:
#   /workspace/ming_tts         this repo, with a .venv holding the Dockerfile's pip set
#   /workspace/Ming-3.0-tts     the Ming repo at the commit pinned in the Dockerfile
# Start command: bash /workspace/ming_tts/scripts/boot_runpod.sh
/start.sh &  # RunPod template services (sshd, nginx)

export MING_PROJ=/workspace/ming_tts
export MING_DIR=${MING_DIR:-/workspace/Ming-3.0-tts}
export HF_HOME=${HF_HOME:-/workspace/hf}
export VOICES_DIR=${VOICES_DIR:-/workspace/voices}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# glibc allocator tuning; the rationale is in the Dockerfile.
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072

exec bash /workspace/ming_tts/scripts/start.sh
