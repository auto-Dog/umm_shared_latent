#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${UNIQUERY_PYTHON:-python}"
CONFIG="${CONFIG:-configs/local_pt_smoke.yaml}"
NPROC="${NPROC:-1}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"

cd "$ROOT"
exec "$PYTHON" -m umm_uniquery.resilient_launch \
  --nproc-per-node="$NPROC" \
  --max-restarts="$MAX_RESTARTS" \
  --config "$CONFIG" \
  "$@"
