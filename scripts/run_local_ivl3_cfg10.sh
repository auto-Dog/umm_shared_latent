#!/usr/bin/env bash
# Local single-card run of configs/local_pt_ivl3_cfg10.yaml (OpenUni-aligned
# convergence check, 10% CC12M). All pretrained weights come from local copies
# under /home/mingjun/models (downloaded per README "预训练模型" via hf-mirror);
# only streaming data is pulled from the Hub.
#
# The 24GB 3090 cannot hold the config's batch 32, so the launch overrides it to
# batch 16 x grad-accum 2 (same effective batch 32 as the config). Override with
# BATCH/GRAD_ACCUM env vars if needed.
#
# Usage:
#   scripts/run_local_ivl3_cfg10.sh          # defaults: GPU 2, timestamped out dir
#   CUDA_VISIBLE_DEVICES=0 scripts/run_local_ivl3_cfg10.sh
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# GPU 2 is the idle 3090 on this host (0/1/3 carry other users' jobs).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# Reduce allocator fragmentation on the 24GB card (batch 32 OOMs otherwise).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${UNIQUERY_PYTHON:-/home/mingjun/.conda/envs/uniquery/bin/python}"
CONFIG="${CONFIG:-configs/local_pt_ivl3_cfg10.yaml}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
# Fresh timestamped output dir so auto_resume never picks up the abnormal old
# checkpoint under outputs/pt_ivl3_cfg10. Defaults to the main checkout's outputs/
# (absolute path) so training artifacts survive worktree cleanup. Override with
# RUN_DIR for a fixed path.
RUN_DIR="${RUN_DIR:-/home/mingjun/umm_uniquery/outputs/pt_ivl3_cfg10_$(date +%Y%m%d-%H%M%S)}"

# Local model copies (README "预训练模型" table):
MODEL_ROOT="${MODEL_ROOT:-/home/mingjun/models}"

cd "$ROOT"
mkdir -p "$RUN_DIR"
exec "$PYTHON" -m umm_uniquery.resilient_launch \
  --nproc-per-node=1 \
  --max-restarts="$MAX_RESTARTS" \
  --config "$CONFIG" \
  --set training.output_dir="$RUN_DIR" \
  --set training.run_name="$(basename "$RUN_DIR")" \
  --set training.per_device_train_batch_size="${BATCH:-16}" \
  --set training.gradient_accumulation_steps="${GRAD_ACCUM:-2}" \
  --set model.ivl3_id="$MODEL_ROOT/internvl3-1b" \
  --set model.sana_id="$MODEL_ROOT/sana-600m-512px" \
  --set model.vae_id="$MODEL_ROOT/sana-vae" \
  >> "$RUN_DIR/train.log" 2>&1
