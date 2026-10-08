from __future__ import annotations

import argparse
import subprocess
import sys

from umm_uniquery.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch UniQuery under TorchElastic with automatic checkpoint recovery"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--max-restarts", type=int, default=None)
    parser.add_argument("--monitor-interval", type=float, default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Continue from this checkpoint. Omitted, the trainer falls back to the "
        "config's failure_recovery.auto_resume / TorchElastic restart handling.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    recovery = config["training"].get("failure_recovery", {})
    if not recovery.get("enabled", True):
        max_restarts = 0
    else:
        max_restarts = int(
            args.max_restarts
            if args.max_restarts is not None
            else recovery.get("max_restarts", 5)
        )
    monitor_interval = float(
        args.monitor_interval
        if args.monitor_interval is not None
        else recovery.get("monitor_interval_seconds", 5)
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.nproc_per_node}",
        f"--max-restarts={max_restarts}",
        f"--monitor-interval={monitor_interval}",
        "-m",
        "umm_uniquery.train",
        "--config",
        args.config,
    ]
    for override in args.overrides:
        command.extend(["--set", override])
    if args.resume_from_checkpoint:
        command.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])

    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
