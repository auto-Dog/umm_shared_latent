#!/usr/bin/env bash
# run_remote.sh — fully-automatic remote training submission for the 3-stage
# UniQuery pipeline (InternVL3-1B backbone + Sana-0.6B + DC-AE VAE).
#
# What it does, in order:
#   1. Downloads the frozen backbone weights to ./cache/{modelname}
#      (internvl3-1b, sana-600m-512px, dc-ae-f32c32-sana-1.1-diffusers),
#      resumable + idempotent, through HF_ENDPOINT (hf-mirror by default).
#   2. Stage 1 (configs/remote_pt_cc12m_1_2m.yaml): 1.2M CC12M caption->image,
#      trains learnable query + connector only (Sana frozen).
#   3. Stage 2 (configs/remote_pt_edit_1_2m.yaml): 1.2M OmniEdit, continues
#      query + connector and makes Sana trainable (init = stage-1 adapter).
#   4. Stage 3 (configs/remote_pt_finetune_blip3o_metaquery.yaml): 600K BLIP3o
#      long-caption + 600K MetaQuery-Instruct, fine-tunes query + connector +
#      Sana (init = stage-2 adapter).
#
# Each dataset trains under its own config + output_dir, so every stage keeps its
# own checkpoints under outputs/remote_pt_*/checkpoint-*. Finished stages are
# skipped on re-run; a stage that crashed mid-way auto-resumes from its own output
# dir (TorchElastic + recovery.auto_resume). Use fresh output_dir paths to retrain
# a stage from scratch.
#
# Edit-stage caveat: Sana has no native image-to-image conditioning and the
# InternVL3 MetaQuery path is text-only, so OmniEdit samples are consumed as
# "edit instruction text -> edited image" (reference image not part of the
# conditioning). The stage-2/3 configs lower the micro batch (8x4 = effective 32)
# and the LR (5e-5) so the extra trainable-Sana activations fit a 24GB card.
#
# Usage:
#   bash scripts/run_remote.sh                                   # GPU 0, 1 proc
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash scripts/run_remote.sh  # multi-GPU DDP
#   CACHE_ROOT=/mnt/models bash scripts/run_remote.sh            # other cache dir
#   RERUN_STAGES=1 bash scripts/run_remote.sh                    # launch even if complete
#
# Env knobs: HF_ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER, HF_HUB_DOWNLOAD_TIMEOUT,
#   HF_HUB_ETAG_TIMEOUT, CUDA_VISIBLE_DEVICES, UNIQUERY_PYTHON, CACHE_ROOT,
#   NPROC, MAX_RESTARTS, RERUN_STAGES.

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Large streaming shards (multi-GB parquet) fail on short read timeouts; 900s covers
# an entire ~6GB shard at ~7MB/s. Override if the remote link is faster/slower.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Reduce allocator fragmentation on 24GB cards (esp. stages 2/3 with trainable Sana).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${UNIQUERY_PYTHON:-python}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
NPROC="${NPROC:-1}"
MAX_RESTARTS="${MAX_RESTARTS:-30}"
RERUN_STAGES="${RERUN_STAGES:-0}"
# Make the src-layout package importable without a pip install -e.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

die() {
    echo "[run_remote] $*" >&2
    exit 1
}

# ---------------------------------------------------------------- preflight

"$PYTHON" -c "import umm_uniquery, torch, transformers, diffusers, datasets" \
    || die "dependencies missing; install the project first (e.g. pip install -e '.[tracking]')"

# ---------------------------------------------------------------- model download

download_repo() {
    local repo="$1" dir="$2"
    echo "[dl] $repo -> $dir"
    "$PYTHON" - "$repo" "$dir" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PYEOF
}

ensure_model() {
    local repo="$1" dir="$2"
    shift 2
    mkdir -p "$dir"
    local file ok=1
    for file in "$@"; do
        [ -s "$dir/$file" ] || ok=0
    done
    if [ "$ok" = 1 ]; then
        echo "[dl] $repo already complete at $dir"
        return 0
    fi
    download_repo "$repo" "$dir"
    for file in "$@"; do
        [ -s "$dir/$file" ] || die "download of $repo incomplete: $dir/$file missing"
    done
    echo "[dl] $repo complete -> $dir"
}

echo "[dl] downloading frozen backbone weights into $CACHE_ROOT"
ensure_model "OpenGVLab/InternVL3-1B" \
    "$CACHE_ROOT/internvl3-1b" \
    config.json model.safetensors
ensure_model "Efficient-Large-Model/Sana_600M_512px_diffusers" \
    "$CACHE_ROOT/sana-600m-512px" \
    model_index.json transformer/config.json
ensure_model "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers" \
    "$CACHE_ROOT/dc-ae-f32c32-sana-1.1-diffusers" \
    config.json diffusion_pytorch_model.safetensors

# ---------------------------------------------------------------- stage runs

stage_complete() {
    local out="$1"
    [ -f "$out/adapter_model.safetensors" ] \
        && [ -f "$out/training_args.bin" ] \
        && [ -f "$out/trainable_parameters.json" ]
}

run_stage() {
    local name="$1" cfg="$2" out="$3" init_from="$4"
    local outdir="$ROOT/$out"
    if [ "$RERUN_STAGES" != 1 ] && stage_complete "$outdir"; then
        echo "[skip] $name already complete -> $outdir"
        return 0
    fi
    mkdir -p "$outdir"

    local overrides=(
        --set "training.output_dir=$outdir"
        --set "training.run_name=$name"
        --set "model.ivl3_id=$CACHE_ROOT/internvl3-1b"
        --set "model.sana_id=$CACHE_ROOT/sana-600m-512px"
        --set "model.vae_id=$CACHE_ROOT/dc-ae-f32c32-sana-1.1-diffusers"
    )
    if [ -n "$init_from" ]; then
        local prev="$ROOT/$init_from"
        [ -f "$prev/adapter_model.safetensors" ] \
            || die "stage $name needs init_checkpoint from $prev, but $prev/adapter_model.safetensors is missing"
        overrides+=(--set "model.init_checkpoint=$prev")
    fi

    echo "[run] $name: config=$cfg out=$outdir (log: $outdir/train.log)"
    if "$PYTHON" -m umm_uniquery.resilient_launch \
        --config "$ROOT/$cfg" \
        --nproc-per-node "$NPROC" \
        --max-restarts "$MAX_RESTARTS" \
        "${overrides[@]}" > "$outdir/train.log" 2>&1; then
        :
    else
        local rc=$?
        echo "[fail] $name exit=$rc; tail of $outdir/train.log:" >&2
        tail -n 40 "$outdir/train.log" >&2 || true
        die "stage $name failed (exit=$rc)"
    fi
    stage_complete "$outdir" \
        || die "stage $name finished but no final adapter at $outdir"
    echo "[ok] $name finished -> $outdir"
}

run_stage "uniquery-remote-pt-cc12m-1_2m" \
    "configs/remote_pt_cc12m_1_2m.yaml" \
    "outputs/remote_pt_cc12m_1_2m" \
    ""

run_stage "uniquery-remote-pt-edit-1_2m" \
    "configs/remote_pt_edit_1_2m.yaml" \
    "outputs/remote_pt_edit_1_2m" \
    "outputs/remote_pt_cc12m_1_2m"

run_stage "uniquery-remote-pt-finetune-blip3o-metaquery" \
    "configs/remote_pt_finetune_blip3o_metaquery.yaml" \
    "outputs/remote_pt_finetune_blip3o_metaquery" \
    "outputs/remote_pt_edit_1_2m"

echo "[run_remote] all stages complete."
echo "  Stage 1 (CC12M)        -> outputs/remote_pt_cc12m_1_2m"
echo "  Stage 2 (OmniEdit)     -> outputs/remote_pt_edit_1_2m"
echo "  Stage 3 (BLIP3o+MQuery)-> outputs/remote_pt_finetune_blip3o_metaquery"
