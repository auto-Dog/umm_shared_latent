from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed

from umm_uniquery.config import load_config
from umm_uniquery.hf_mirror import install_hf_mirror_rewrite
from umm_uniquery.modeling import UniQueryInternVL3Model, UniQueryModel
from umm_uniquery.registry import STAGES
from umm_uniquery.training.stages import StageComponents  # noqa: F401 - registers stages
from umm_uniquery.training.trainer import UniQueryTrainer
from umm_uniquery.training.recovery import (
    CheckpointCompletionCallback,
    find_latest_resumable_checkpoint,
    is_resumable_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UMM UniQuery baseline")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="Dotted config override, for example training.output_dir=/mnt/run",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Trainer checkpoint to resume, including optimizer and stream skip state",
    )
    return parser.parse_args()


def _training_arguments(config: dict, total_samples: int) -> TrainingArguments:
    train = config["training"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = (
        int(train["per_device_train_batch_size"])
        * int(train.get("gradient_accumulation_steps", 1))
        * world_size
    )
    derived_steps = math.ceil(total_samples / effective_batch)
    max_steps = int(train.get("max_steps") or derived_steps)
    return TrainingArguments(
        output_dir=train["output_dir"],
        run_name=train.get("run_name", Path(train["output_dir"]).name),
        max_steps=max_steps,
        per_device_train_batch_size=int(train["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train.get("learning_rate", 1e-4)),
        weight_decay=float(train.get("weight_decay", 0.1)),
        warmup_ratio=float(train.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(train.get("max_grad_norm", 1.0)),
        bf16=bool(train.get("bf16", True)),
        tf32=bool(train.get("tf32", True)),
        gradient_checkpointing=False,
        logging_steps=int(train.get("logging_steps", 10)),
        save_steps=int(train.get("save_steps", 500)),
        save_total_limit=int(train.get("save_total_limit", 2)),
        report_to=train.get("report_to", "wandb"),
        dataloader_num_workers=int(train.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(train.get("dataloader_pin_memory", True)),
        dataloader_drop_last=True,
        # Multimodal batches have variable token/patch shapes and cannot be safely
        # concatenated by DataLoaderDispatcher before rank distribution.
        accelerator_config={"dispatch_batches": False, "split_batches": False},
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ignore_data_skip=False,
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        optim=train.get("optim", "adamw_torch_fused"),
        deepspeed=train.get("deepspeed"),
    )


def _print_data_sample(model: Any, components: StageComponents) -> None:
    """Print one collated sample exactly as the model will receive it (RANK 0).

    Each ExactStreamingMixture.__iter__() builds fresh, seed-deterministic iterators,
    so consuming one sample here does not disturb the trainer's own iterator.
    """
    example = next(iter(components.dataset))
    batch = components.collator([example])
    print("=" * 72)
    print("[data-sample] task:", example["task"])
    print("[data-sample] prompt:", example["prompt"])
    text_encoder = getattr(model, "processor", None) or getattr(model, "tokenizer", None)
    if text_encoder is not None:
        decoded = text_encoder.decode(
            batch["input_ids"][0], skip_special_tokens=False
        )
        print("[data-sample] decoded input_ids:", decoded[:2000])
    for name, value in batch.items():
        if name == "input_ids":
            continue
        print(
            f"[data-sample] {name}: shape={tuple(value.shape)} dtype={value.dtype}"
            + (
                f" range=[{value.min():.3f}, {value.max():.3f}]"
                if value.dtype.is_floating_point
                else ""
            )
        )
    print("[data-sample] input_ids: shape=" + str(tuple(batch["input_ids"].shape)))
    print("=" * 72)


def main() -> None:
    install_hf_mirror_rewrite()
    args = parse_args()
    config = load_config(args.config, args.overrides)
    deepspeed = config["training"].get("deepspeed")
    if deepspeed and not Path(deepspeed).is_absolute():
        config_path = Path(args.config).resolve()
        project_root = next(
            (parent for parent in config_path.parents if (parent / "pyproject.toml").exists()),
            Path.cwd(),
        )
        config["training"]["deepspeed"] = str((project_root / deepspeed).resolve())
    set_seed(int(config.get("seed", 42)))

    if config["model"].get("backbone") == "internvl3":
        model = UniQueryInternVL3Model(config["model"])
    else:
        model = UniQueryModel(config["model"])
    init_checkpoint = config["model"].get("init_checkpoint")
    if init_checkpoint:
        model.load_adapter(init_checkpoint, strict=False)

    stage_builder = STAGES.get(config["stage"])
    components = stage_builder(config, model)
    if int(os.environ.get("RANK", "0")) == 0:
        _print_data_sample(model, components)
    training_args = _training_arguments(config, components.dataset.total_samples)

    resume_checkpoint = args.resume_from_checkpoint
    recovery = config["training"].get("failure_recovery", {})
    elastic_restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    if elastic_restart_count > 0 and recovery.get("enabled", True):
        resume_checkpoint = (
            find_latest_resumable_checkpoint(training_args.output_dir)
            or (None if resume_checkpoint == "auto" else resume_checkpoint)
        )
    elif resume_checkpoint == "auto":
        resume_checkpoint = find_latest_resumable_checkpoint(training_args.output_dir)
    elif (
        resume_checkpoint is None
        and recovery.get("enabled", True)
        and recovery.get("auto_resume", True)
    ):
        resume_checkpoint = find_latest_resumable_checkpoint(training_args.output_dir)
    if resume_checkpoint is not None and not is_resumable_checkpoint(resume_checkpoint):
        raise ValueError(f"Checkpoint is incomplete or not resumable: {resume_checkpoint}")
    if int(os.environ.get("RANK", "0")) == 0:
        if resume_checkpoint:
            print(f"[recovery] Resuming from complete checkpoint: {resume_checkpoint}")
        elif elastic_restart_count > 0:
            print("[recovery] No complete checkpoint found; restarting from the beginning")

    trainable_millions = model.trainable_parameter_count / 1_000_000
    guard = config["model"].get("trainable_parameter_guard_m")
    if guard is not None:
        lower, upper = guard
        if not (float(lower) <= trainable_millions <= float(upper)):
            raise ValueError(
                f"Trainable parameter count {trainable_millions:.2f}M is outside guard "
                f"[{lower}, {upper}]M; check the freeze policy and connector config."
            )

    trainer = UniQueryTrainer(
        model=model,
        args=training_args,
        train_dataset=components.dataset,
        data_collator=components.collator,
        callbacks=[CheckpointCompletionCallback()],
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
