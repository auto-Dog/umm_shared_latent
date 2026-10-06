#!/usr/bin/env bash

# This launcher uses bash-only features (arrays). When invoked as `sh run_autodl.sh`
# the shebang is ignored and dash chokes on the array syntax, so re-exec under bash.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Large streaming shards (multi-GB parquet) fail on short read timeouts; 900s covers
# an entire ~6GB shard at ~7MB/s. Override with HF_HUB_DOWNLOAD_TIMEOUT if needed.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# AutoDL routes all container egress through a shared HTTP proxy
# (http_proxy/https_proxy = http://172.29.51.4:12798). That proxy throttles
# HuggingFace streaming to ~1.5MB/s vs ~5-10MB/s direct, which starves the GPUs
# on this data-bound run (CC12M WebDataset shards are read straight off the Hub).
# Bypass the proxy for HF hosts only; everything else (wandb, github, ...) still
# uses it. datasets' fsspec HTTP backend sets trust_env=True, and huggingface_hub
# uses requests, so both honour no_proxy.
export no_proxy="hf-mirror.com,huggingface.co,hf.co,xethub.hf.co,${no_proxy:-}"
export NO_PROXY="$no_proxy"


ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${UNIQUERY_PYTHON:-python}"
NPROC="${NPROC:-2}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
# Local pretrained weight copies on this autodl host (the configs inherit the
# /home/mingjun/models paths from local_pt_ivl3_cfg10.yaml, which don't exist
# here). Override with MODEL_ROOT if the weights live elsewhere.
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
# Reduce allocator fragmentation on the 24GB card (trainable Sana otherwise OOMs).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Pick ONE stage per invocation (this launcher no longer chains both):
#   bash scripts/run_autodl.sh blip3o      # stage A: BLIP3o long-caption 600K
#   bash scripts/run_autodl.sh metaquery   # stage B: MetaQuery-Instruct 600K
# The stage name may also come from the STAGE env var. Any extra args after the
# stage are forwarded to resilient_launch as `--set k=v` overrides.
#
# Each stage has its own default config / output dir / init checkpoint, all
# overridable via env (CONFIG, RUN_DIR, INIT_CHECKPOINT):
#   blip3o   -> configs/local_finetune_blip3o.yaml   outputs/local_finetune_blip3o
#               init = outputs/pt_ivl3_cfg10/checkpoint-16667 (trained PT adapter)
#   metaquery-> configs/local_finetune_metaquery.yaml outputs/local_finetune_metaquery
#               init = outputs/local_finetune_blip3o (stage A's final adapter)
#
# Run blip3o first, then metaquery (it consumes blip3o's adapter). Fixed output
# dirs mean re-running a stage auto-resumes it from its own checkpoints.
# ---------------------------------------------------------------------------

STAGE="${STAGE:-}"
case "${1:-}" in
    blip3o | metaquery)
        STAGE="$1"
        shift
        ;;
esac

case "$STAGE" in
    blip3o)
        CONFIG="${CONFIG:-configs/local_finetune_blip3o.yaml}"
        RUN_DIR="${RUN_DIR:-outputs/local_finetune_blip3o}"
        INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pt_ivl3_cfg10/checkpoint-16667}"
        ;;
    metaquery)
        CONFIG="${CONFIG:-configs/local_finetune_metaquery.yaml}"
        RUN_DIR="${RUN_DIR:-outputs/local_finetune_metaquery}"
        INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/local_finetune_blip3o}"
        ;;
    *)
        echo "usage: bash scripts/run_autodl.sh <blip3o|metaquery> [extra --set ...]" >&2
        echo "  blip3o    stage A: configs/local_finetune_blip3o.yaml (init: PT adapter)" >&2
        echo "  metaquery stage B: configs/local_finetune_metaquery.yaml (init: stage A output)" >&2
        exit 2
        ;;
esac

cd "$ROOT"
[ -f "$CONFIG" ] || {
    echo "[run_autodl] config not found: $CONFIG" >&2
    exit 1
}
mkdir -p "$RUN_DIR"

overrides=(
    --set "training.output_dir=$RUN_DIR"
    --set "training.run_name=$(basename "$RUN_DIR")"
    --set "model.ivl3_id=$MODEL_ROOT/internvl3-1b"
    --set "model.sana_id=$MODEL_ROOT/sana-600m-512px"
    --set "model.vae_id=$MODEL_ROOT/sana-vae"
)
if [ -n "${INIT_CHECKPOINT:-}" ]; then
    overrides+=(--set "model.init_checkpoint=$INIT_CHECKPOINT")
fi

echo "[run_autodl] stage=$STAGE config=$CONFIG out=$RUN_DIR init=${INIT_CHECKPOINT:-<none>}"
echo "[run_autodl] log: $RUN_DIR/train.log"

exec "$PYTHON" -m umm_uniquery.resilient_launch \
    --nproc-per-node="$NPROC" \
    --max-restarts="$MAX_RESTARTS" \
    --config "$CONFIG" \
    "${overrides[@]}" \
    "$@" > "$RUN_DIR/train.log" 2>&1
