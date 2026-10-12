#!/usr/bin/env bash
# Run all probe models on the current prompt registry.
# Usage: bash run_all.sh [prompts.jsonl] [gpu_sana] [gpu_ivl3enh] [gpu_openuni]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/mingjun/.conda/envs/bagel/bin/python
PROMPTS="${1:-$ROOT/prompts/umbrella_baseball.jsonl}"
GPU_SANA="${2:-1}"
GPU_ENH="${3:-2}"
GPU_OPENUNI="${4:-2}"

echo "[$(date '+%F %T')] probes on $PROMPTS"

CUDA_VISIBLE_DEVICES=$GPU_SANA $PY "$ROOT/scripts/gen_sana512.py" \
    --prompts "$PROMPTS" --out "$ROOT/outputs/sana512" --gpu "$GPU_SANA" \
    2>&1 | tee "$ROOT/logs/sana512.log"

CUDA_VISIBLE_DEVICES=$GPU_ENH $PY "$ROOT/scripts/gen_sana512_ivl3enh.py" \
    --prompts "$PROMPTS" --out "$ROOT/outputs/sana512_ivl3enh" --gpu "$GPU_ENH" \
    2>&1 | tee "$ROOT/logs/sana512_ivl3enh.log"

CUDA_VISIBLE_DEVICES=$GPU_OPENUNI $PY "$ROOT/scripts/gen_openuni_b512.py" \
    --prompts "$PROMPTS" --out "$ROOT/outputs/openuni_b512" --gpu "$GPU_OPENUNI" \
    2>&1 | tee "$ROOT/logs/openuni_b512.log"

echo "[$(date '+%F %T')] all probes done"
