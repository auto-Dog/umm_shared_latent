"""Probe generation with OpenUni-B-512 (InternVL3-1B + Sana-0.6B-512px).

Loads the official OpenUni-B-512 model (256 meta-queries, 6-layer connector)
from the OpenUni repo and the official SFT checkpoint (blip3o60k), then
generates 512x512 images per probe prompt.

The prompt goes through InternVL3-1B as an image-generation instruction
("Generate an image: {input}" + CFG prompt) — no Gemma2 text encoder, unlike
gen_sana512.py / gen_sana512_ivl3enh.py.

Code is taken from the OpenUni geneval worktree (has the diffusers 0.32.0
tokenizer fix: SanaPipeline is built with tokenizer=self.tokenizer).

Usage:
    CUDA_VISIBLE_DEVICES=2 python gen_openuni_b512.py --out ../outputs/openuni_b512
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_GPU = "2"
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _GPU = sys.argv[_i + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# OpenUni repo root must be importable (src.* modules). Use the geneval
# worktree which contains the diffusers 0.32.0 SanaPipeline tokenizer fix.
OPENUNI_ROOT = "/home/mingjun/OpenUni/.claude/worktrees/openuni-geneval-b512"
sys.path.insert(0, OPENUNI_ROOT)
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

import torch  # noqa: E402
from xtuner.registry import BUILDER  # noqa: E402
from xtuner.model.utils import guess_load_checkpoint  # noqa: E402
from mmengine.config import Config  # noqa: E402

from common import add_common_args, load_prompts, write_meta  # noqa: E402

LOCAL_CONFIG = OPENUNI_ROOT + "/geneval_b512/run/openuni_b_internvl3_1b_sana_0_6b_512_hf_local.py"
CHECKPOINT = "/data/mingjun/openuni_model/openuni_b_internvl3_1b_sana_0_6b_512_hf_blip3o60k.pth"
HEIGHT = WIDTH = 512
STEPS = 20
GUIDANCE = 4.5


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    print(f"building OpenUni-B-512 from {LOCAL_CONFIG} ...", flush=True)
    t0 = time.time()
    config = Config.fromfile(LOCAL_CONFIG)
    model = BUILDER.build(config.model).cuda().bfloat16().eval()

    print(f"loading checkpoint {CHECKPOINT} ...", flush=True)
    state_dict = guess_load_checkpoint(CHECKPOINT)
    unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"model built+loaded in {time.time()-t0:.1f}s; unexpected={unexpected.unexpected_keys[:5]}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts)
    for p in prompts:
        pid, text = p["id"], p["prompt"]
        rows = []
        for seed in args.seeds:
            class_info = model.prepare_text_conditions(text, cfg_prompt=None)
            input_ids = class_info["input_ids"]
            attention_mask = class_info["attention_mask"]
            generator = torch.Generator(device=model.device).manual_seed(seed)
            t1 = time.time()
            samples = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                cfg_scale=GUIDANCE, num_steps=STEPS, generator=generator,
                height=HEIGHT, width=WIDTH, progress_bar=False,
            )
            print(f"[{pid}] seed={seed} samples shape={tuple(samples.shape)}", flush=True)
            img = torch.clamp(127.5 * samples + 128.0, 0, 255)
            img = img.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)[0].numpy()
            from PIL import Image
            d = args.out / pid
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(d / f"{seed}.png")
            rows.append({"prompt": text, "prompt_id": pid, "seed": seed,
                         "model": "openuni_b512", "checkpoint": "blip3o60k",
                         "steps": STEPS, "guidance": GUIDANCE,
                         "size": f"{WIDTH}x{HEIGHT}", "seconds": round(time.time()-t1, 2)})
            print(f"[{pid}] seed={seed} {time.time()-t1:.1f}s -> {d / f'{seed}.png'}", flush=True)
        write_meta(args.out, pid, rows)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
