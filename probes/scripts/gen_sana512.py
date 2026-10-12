"""Probe generation with vanilla Sana_600M_512px_diffusers.

Loads the complete Sana pipeline (official Gemma2-2B text encoder) and
generates 512x512 images straight from each probe prompt. No InternVL3 /
no OpenUni — this is the plain text-to-image backbone baseline.

Recipe matches sana_geneval/generate_geneval_sana.py: fp16 pipeline,
vae+text_encoder in bf16, 20 DPM++ steps, guidance 4.5.

Usage:
    CUDA_VISIBLE_DEVICES=1 python gen_sana512.py --out ../outputs/sana512
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_GPU = "1"
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _GPU = sys.argv[_i + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import SanaPipeline

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
from common import add_common_args, load_prompts, write_meta  # noqa: E402

MODEL_DIR = "/home/mingjun/models/sana-600m-512px"
HEIGHT = WIDTH = 512
STEPS = 20
GUIDANCE = 4.5


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    print(f"loading Sana 512px from {MODEL_DIR} (fp16, vae+text_encoder bf16)...", flush=True)
    t0 = time.time()
    pipe = SanaPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.to("cuda")
    print(f"loaded in {time.time()-t0:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts)
    for p in prompts:
        pid, text = p["id"], p["prompt"]
        rows = []
        for seed in args.seeds:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            t1 = time.time()
            img = pipe(
                prompt=text,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            ).images[0]
            d = args.out / pid
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{seed}.png")
            rows.append({"prompt": text, "prompt_id": pid, "seed": seed,
                         "model": "sana512", "steps": STEPS, "guidance": GUIDANCE,
                         "size": f"{WIDTH}x{HEIGHT}", "seconds": round(time.time()-t1, 2)})
            print(f"[{pid}] seed={seed} {time.time()-t1:.1f}s -> {d / f'{seed}.png'}", flush=True)
        write_meta(args.out, pid, rows)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
