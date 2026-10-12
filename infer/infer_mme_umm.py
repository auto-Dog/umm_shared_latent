#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MME-P (Perception, 10 yes/no subtasks) eval for UMM(MetaQuery) adapters.

Answers each MME question with the verified learnable-query multimodal
understanding path (user-end injection):
    user turn = <image tokens> + question + <begin_of_img><img0..255><end_of_img>
               (query_suffix right before <|im_end|>)
    assistant turn = <|im_start|>assistant\n  ->  text answer ("Yes, ..." / "No, ...")

Same harness as infer_mme_internvl3.py: official 4-column tab result files
(image \\t question \\t gt \\t response), resumable per image via done.json.
Score the result dir with the official eval_tool/calculation.py (yes/no only).

Usage:
  CUDA_VISIBLE_DEVICES=0 <uniquery>/bin/python infer/infer_mme_umm.py \
      --checkpoint /home/mingjun/models/t2i_blip3o_sft_1007_9000 --tag sft
  --task NAME  run one task | --limit N  smoke test | --gpu 0  (pre-torch)
"""
import argparse
import json
import os
import time
from pathlib import Path

# ---------- pre-scan argv for --gpu BEFORE importing torch (device-list cache gotcha) ----------
_args = argparse.ArgumentParser(add_help=False)
_args.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
_args.add_argument("--results_dir", default=None)
_args.add_argument("--task", default=None)
_args.add_argument("--limit", type=int, default=None)
_tmp, _ = _args.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _tmp.gpu
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchvision.transforms as T
from PIL import Image
from umm_uniquery.config import load_config
from umm_uniquery.modeling import UniQueryInternVL3Model
from umm_uniquery.modeling.internvl3_model import _INTERNVL3_SYSTEM, _INTERNVL3_USER

print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | visible "
      f"{torch.cuda.device_count()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")

PROJECT = Path("/home/mingjun/umm_uniquery")
ROOT = Path("/home/mingjun/Awesome-Multimodal-Large-Language-Models/tools/eval_tool")
DATASET_DIR = ROOT / "MME_Benchmark_release_version" / "MME_Benchmark"
DEFAULT_RESULTS_DIR = PROJECT / "outputs" / "mme_umm"
CHECKPOINT_DEFAULT = "/home/mingjun/models/t2i_blip3o_sft_1007_9000"
CHECKPOINT_CC12M = "/home/mingjun/models/t2i_cc12m_pt_1007_16667"

UNDERSTAND_SYSTEM = (
    "You are a helpful, harmless multimodal assistant. Answer the user's "
    "question about the provided image concisely and accurately."
)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SEP_TOKEN = "<|im_end|>"

PERCEPTION_TASKS = ["existence", "count", "position", "color", "posters", "celebrity",
                    "scene", "landmark", "artwork", "OCR"]

# ---------------- dataset loading (same as infer_mme_internvl3.py) ----------------
def list_qa_files(task_dir: Path):
    qa_dir = task_dir / "questions_answers_YN"
    if qa_dir.is_dir():
        return sorted(qa_dir.glob("*.txt"))
    return sorted(task_dir.glob("*.txt"))


def load_task(task_dir: Path):
    items = []
    img_dir = task_dir / "images" if (task_dir / "images").is_dir() else task_dir
    for qa_path in list_qa_files(task_dir):
        base = qa_path.stem
        img_path = None
        for ext in (".jpg", ".png"):
            cand = img_dir / (base + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            raise FileNotFoundError(f"cannot find image for {base}")
        qas = []
        for line in qa_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            q, gt = line.split("\t")
            qas.append((q.strip(), gt.strip()))
        items.append({"image_path": img_path, "image_name": img_path.name, "qas": qas})
    return items


# ---------------- dynamic-resolution preprocessing (same as infer_mme_internvl3.py) ----------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if abs(ratio[0] * ratio[1] - area) < abs(best_ratio[0] * best_ratio[1] - area):
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios,
                                                    orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_aspect_ratio[0])) * image_size,
            (i // (target_aspect_ratio[0])) * image_size,
            ((i % (target_aspect_ratio[0])) + 1) * image_size,
            ((i // (target_aspect_ratio[0])) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ---------------- model + user-end query-injection answering ----------------
class UMMMME:
    """MME adapter: cache per-image pixel_values, answer via learnable-query
    understanding (query_suffix at the end of the user turn, before <|im_end|>)."""

    def __init__(self, checkpoint, system=UNDERSTAND_SYSTEM, max_num=12, max_new_tokens=48):
        config = load_config(PROJECT / "configs" / "local_pt_ivl3_cfg10.yaml")
        model = UniQueryInternVL3Model(config["model"])
        model.load_adapter(checkpoint)
        model.to("cuda").eval()
        model.ivl3.img_context_token_id = model.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model = model
        self.system = system
        self.max_num = max_num
        self.max_new_tokens = max_new_tokens
        self._pv_cache = {}
        self._device = next(model.connector.parameters()).device
        self._dtype = next(model.connector.parameters()).dtype
        self.eos_id = model.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        gen_cfg = model.ivl3.language_model.generation_config
        gen_cfg.pad_token_id = self.eos_id
        gen_cfg.eos_token_id = self.eos_id
        print(f"UMM loaded: {checkpoint} | queries: {model.num_queries} | "
              f"device {self._device}", flush=True)

    @torch.inference_mode()
    def answer(self, image_path, question):
        model = self.model
        key = str(image_path)
        if key not in self._pv_cache:
            pv = load_image(key, max_num=self.max_num).to(self._dtype).to(self._device)
            self._pv_cache[key] = pv
        pv = self._pv_cache[key]
        num_patches = pv.shape[0]

        # user turn: <image tokens> + question + query_suffix (before <|im_end|>)
        user_input = f"<image>\n{question}" + model.query_suffix
        base = _INTERNVL3_SYSTEM.format(system=self.system) + _INTERNVL3_USER.format(input=user_input)
        image_tokens = (
            "<img>" + IMG_CONTEXT_TOKEN * model.ivl3.num_image_token * num_patches + "</img>"
        )
        query = base.replace("<image>", image_tokens, 1)

        encoded = model.tokenizer(query, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        outputs = model.ivl3.generate(
            pixel_values=pv,
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_id,
            pad_token_id=self.eos_id,
        )
        text = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return text.split(SEP_TOKEN)[0].strip()


# ---------------- run + resume (same as infer_mme_internvl3.py) ----------------
def _load_done(results_dir: Path):
    fp = results_dir / "done.json"
    if fp.exists():
        return json.loads(fp.read_text())
    return {}


def _save_done(results_dir: Path, done):
    (results_dir / "done.json").write_text(json.dumps(done, indent=1, sort_keys=True))


def run_eval(model_inst, results_dir, tasks, limit=None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    done = _load_done(results_dir)
    t_start = time.time()
    n_total, n_skip = 0, 0
    for task_name in tasks:
        items = load_task(DATASET_DIR / task_name)
        if limit is not None:
            items = items[:limit]
        done_set = set(done.get(task_name, []))
        # recover previously written lines so skipped images keep their rows
        tfp = results_dir / f"{task_name}.txt"
        prev_rows = {}
        if tfp.exists():
            for line in tfp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) == 4:
                    prev_rows.setdefault(parts[0], []).append(line + "\n")
        # drop images that are marked done but have no full rows on disk: either a
        # crashed mid-write (partial rows in file) or done.json saved during a run
        # killed before this task's file was written (name in done.json but not in
        # prev_rows at all). Those answers are unrecoverable -> re-answer them.
        bad = [name for name in done_set
               if name not in prev_rows or len(prev_rows.get(name, [])) != 2]
        done_set -= set(bad)
        todo = [it for it in items if it["image_name"] not in done_set]

        t_task = time.time()
        rows = []
        for i, it in enumerate(todo, 1):
            for question, gt in it["qas"]:
                resp = str(model_inst.answer(it["image_path"], question))
                resp = resp.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
                rows.append(f"{it['image_name']}\t{question}\t{gt}\t{resp}\n")
            done_set.add(it["image_name"])
            done[task_name] = sorted(done_set)
            if i % 20 == 0 or i == len(todo):
                elapsed = time.time() - t_task
                print(f"  [{task_name}] {i}/{len(todo)}  {elapsed:.0f}s "
                      f"({i/elapsed:.2f} img/s)", flush=True)
                _save_done(results_dir, done)
        # merge with rows of previously-completed images and write full file
        # (rows already end with "\n"; keep exactly one trailing newline per row)
        all_rows = []
        for it in items:
            name = it["image_name"]
            if name in prev_rows and len(prev_rows[name]) == 2:
                all_rows.extend(prev_rows[name])
            elif name in done_set:
                all_rows.extend(r for r in rows if r.startswith(name + "\t"))
        tfp.write_text("".join(r if r.endswith("\n") else r + "\n" for r in all_rows),
                       encoding="utf-8")
        _save_done(results_dir, done)
        n_total += len(todo)
        n_skip += len(items) - len(todo)
        print(f"  [ok] {task_name}: {len(items)} img, processed {len(todo)}, "
              f"skipped {len(items)-len(todo)}, took {time.time()-t_task:.0f}s", flush=True)

    print(f"\ndone: {len(tasks)} tasks, processed {n_total} images, skipped {n_skip}, "
          f"total {time.time()-t_start:.0f}s -> {results_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description="MME-P eval for UMM (user-end learnable-query understanding)")
    p.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT,
                   help=f"adapter dir (default SFT: {CHECKPOINT_DEFAULT}; cc12m: {CHECKPOINT_CC12M})")
    p.add_argument("--tag", default=None, help="result-dir suffix, e.g. sft / cc12m")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--results_dir", default=None, help="override full results dir")
    p.add_argument("--task", default=None, help="single task name; default all 10 perception tasks")
    p.add_argument("--limit", type=int, default=None, help="cap images per task (smoke test)")
    args = p.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tasks = [args.task] if args.task else PERCEPTION_TASKS
    if args.results_dir:
        results_dir = args.results_dir
    elif args.tag:
        results_dir = DEFAULT_RESULTS_DIR / f"results_{args.tag}"
    else:
        tag = Path(args.checkpoint).name
        results_dir = DEFAULT_RESULTS_DIR / f"results_{tag}"

    print("tasks:", tasks)
    print("results_dir:", results_dir)
    model_inst = UMMMME(args.checkpoint)
    run_eval(model_inst, results_dir, tasks, limit=args.limit)


if __name__ == "__main__":
    main()
