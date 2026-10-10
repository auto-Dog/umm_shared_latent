#!/usr/bin/env bash
set -euo pipefail

# Dual-card PT with trainable Sana on CC12M 12M + RedCaps5M 5M (17M samples).
# Launcher wrapper around run_local.sh; run with: bash scripts/run_flow17m.sh
cd "$(dirname "$0")/.."

export CONFIG=configs/local_pt_ivl3_cfg10_flow17m.yaml
export CUDA_VISIBLE_DEVICES=1,2
export NPROC=2
export UNIQUERY_PYTHON=/home/mingjun/.conda/envs/uniquery/bin/python
export RUN_DIR=/home/mingjun/umm_uniquery/outputs/pt_ivl3_cfg10_flow17m_20261009-2130
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The `umm_uniquery` package is an editable install pointing at the MAIN checkout;
# this run must use the worktree copy (redcaps5m + warmup_steps), so shadow it.
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

exec bash scripts/run_local.sh
