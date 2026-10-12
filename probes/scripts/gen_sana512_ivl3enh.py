"""Probe generation: Sana_600M_512px + InternVL3-1B prompt enhancement.

The probe prompt is first rewritten by InternVL3-1B (3 stochastic candidates,
temp 0.8 / top_p 0.9, keeping umbrella+baseball+relation via a light semantic
check, falling back to the original prompt), then Sana 512px generates from
the chosen enhanced prompt with the same recipe as gen_sana512.py.

Usage:
    CUDA_VISIBLE_DEVICES=1 python gen_sana512_ivl3enh.py --out ../outputs/sana512_ivl3enh
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

_GPU = "2"
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _GPU = sys.argv[_i + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModel, AutoTokenizer
from diffusers import SanaPipeline

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
from common import add_common_args, load_prompts, write_meta  # noqa: E402

IVL3_DIR = "/home/mingjun/models/internvl3-1b"
SANA_DIR = "/home/mingjun/models/sana-600m-512px"
MAX_NEW = 256
TEMP = 0.8
TOP_P = 0.9
ATTEMPTS = 3
HEIGHT = WIDTH = 512
STEPS = 20
GUIDANCE = 4.5

SYSTEM = (
    "You are an expert prompt writer for a text-to-image diffusion model. "
    "Rewrite the given image description into ONE detailed English sentence, "
    "adding vivid scene detail (setting, lighting, materials, style, background). "
    "CRITICAL: keep every object, every count, every color and every spatial "
    "relation exactly the same as in the original. Do not add new objects, "
    "do not change colors or counts, do not move objects. Output only the "
    "rewritten sentence, nothing else."
)

# Light semantic check for the umbrella+baseball probe: both objects must stay
# present and the vertical relation word must survive. (Reuse the GenEval
# checker wholesale if a metadata registry is ever added.)
_OBJ = re.compile(r"umbrella", re.I)
_OBJ2 = re.compile(r"baseball", re.I)
_REL = re.compile(r"\b(below|under|beneath|underneath)\b", re.I)


def required_check(text: str) -> bool:
    return bool(_OBJ.search(text) and _OBJ2.search(text) and _REL.search(text))


def enhance(model, tokenizer, text: str) -> tuple[str, list[str]]:
    old_system = model.system_message
    model.system_message = SYSTEM
    candidates = []
    try:
        for _ in range(ATTEMPTS):
            resp, _ = model.chat(
                tokenizer, None, text,
                dict(max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=TOP_P),
                history=None, return_history=True,
            )
            resp = resp.strip().strip('"')
            candidates.append(resp)
            if required_check(resp):
                return resp, candidates
    finally:
        model.system_message = old_system
    return text, candidates  # fallback to original prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    print(f"loading InternVL3-1B from {IVL3_DIR} ...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        IVL3_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(IVL3_DIR, trust_remote_code=True, use_fast=False)
    print(f"internvl3 loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"loading Sana 512px from {SANA_DIR} (fp16, vae+text_encoder bf16)...", flush=True)
    pipe = SanaPipeline.from_pretrained(SANA_DIR, torch_dtype=torch.float16)
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.to("cuda")
    print(f"loaded in {time.time()-t0:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts)
    for p in prompts:
        pid, text = p["id"], p["prompt"]
        enhanced, candidates = enhance(model, tokenizer, text)
        print(f"[{pid}] enhanced: {enhanced!r}  (candidates: {len(candidates)})", flush=True)

        rows = []
        for seed in args.seeds:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            t1 = time.time()
            img = pipe(
                prompt=enhanced,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            ).images[0]
            d = args.out / pid
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{seed}.png")
            rows.append({"prompt": text, "enhanced_prompt": enhanced,
                         "candidates": candidates, "prompt_id": pid, "seed": seed,
                         "model": "sana512_ivl3enh", "enhancer": "internvl3-1b",
                         "steps": STEPS, "guidance": GUIDANCE,
                         "size": f"{WIDTH}x{HEIGHT}", "seconds": round(time.time()-t1, 2)})
            print(f"[{pid}] seed={seed} {time.time()-t1:.1f}s -> {d / f'{seed}.png'}", flush=True)
        write_meta(args.out, pid, rows)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
