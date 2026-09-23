"""Pre-flight checks for model loading and mirrored streaming data.

This utility is opt-in and is not invoked by the training entry point.
"""

from __future__ import annotations

import argparse
import os

from umm_uniquery.config import load_config
from umm_uniquery.data import ExactStreamingMixture
from umm_uniquery.hf_mirror import install_hf_mirror_rewrite
from umm_uniquery.modeling import UniQueryModel


def main() -> None:
    install_hf_mirror_rewrite()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_pt_smoke.yaml")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    print("stage       :", config["stage"])
    print("mllm_id     :", config["model"]["mllm_id"])
    print("sana_id     :", config["model"]["sana_id"])
    print("vae_id      :", config["model"]["vae_id"])
    print("HF_ENDPOINT :", os.environ.get("HF_ENDPOINT"))

    if args.model:
        model = UniQueryModel(config["model"])
        print(f"effective trainable params: {model.trainable_parameter_count / 1e6:.2f}M")
        print(
            f"MetaQuery tokens: {model.num_queries} queries "
            f"(embedding rows {model.metaquery_token_start}:{model.metaquery_token_end + 1})"
        )

    if args.data:
        dataset = ExactStreamingMixture(
            config["data"]["sources"],
            seed=int(config.get("seed", 42)),
            shuffle_buffer=int(config["data"].get("shuffle_buffer", 500)),
        )
        iterator = iter(dataset)
        for index in range(args.samples):
            row = next(iterator)
            prompt = (row.get("prompt") or "")[:60].replace("\n", " ")
            print(
                f"row {index}: task={row['task']} src={len(row['source_images'])} "
                f"img={row['target_image'].size} prompt={prompt!r}"
            )


if __name__ == "__main__":
    main()
