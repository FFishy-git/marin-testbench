# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive explicit A100 configurations from the pinned Marin recipes."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .factors import OutputSettings
from .reference import REVISION, TASK, UPSTREAM

ARMS = (
    "muon",
    "muont",
    "outer_left_down",
    "inner_outer_down",
    "down_hybrid",
    "inner_outer_all",
    "null_all",
    "left_all",
    "inner_outer_all_diag",
    "inner_outer_hidden",
    "inner_outer_mlpdiag",
    "inner_outer_leftdiag",
    "outer_left_all",
    "outer_left_all",
    "inner_outer_all_lag",
    "null_mlpdiag",
    "inner_outer_mlpdiag_lag",
    "outer_leftdiag",
    "outer_leftdiag_fused",
    "outer_leftdiag_combined_fused",
    "outer_leftdiag_upstream_fused",
    "null_leftdiag_upstream_fused",
    "inner_outer_mlpdiag_fused",
    "inner_outer_leftdiag_fused",
)
BATCHES = ("0p5m", "1m", "2m", "4m")


def configuration(arm, batch, seed, microbatch):
    base = "muon" if arm == "muon" else "muont"
    path = UPSTREAM / "reproduce" / "marin" / "configs" / "130m" / f"{base}_{batch}.json"
    config = json.loads(path.read_text())
    config["runtime"]["attention"] = "FA2"
    config["runtime"]["deterministic"] = True
    config["runtime"]["compile_options"] = {
        "max_autotune": True,
        "triton.cudagraphs": False,
        "max_autotune_gemm_backends": "ATEN",
        "emulate_precision_casts": True,
    }
    config["training"]["seed"] = seed
    config["training"]["micro_batch_size"] = microbatch
    config["experiment"] = {
        "arm": arm,
        "upstream_revision": REVISION,
        "output": asdict(OutputSettings()),
        "checkpoint_every": 250,
        "diagnostic_every": 250,
        "refresh_install_age": 1,
        "expected_model_parameters": 265585152,
        "hardware_adaptation": f"A100 FA2; world size 8; microbatch {microbatch}, global batch unchanged",
    }
    config["experiment"]["diagonal_backend"] = "triton" if arm.endswith("_fused") else "torch"
    total = microbatch * config["training"]["world_size"]
    if config["training"]["global_batch_sequences"] % total:
        raise ValueError("Microbatch must divide the pinned global batch")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=TASK / "experiment" / "configs")
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for batch in BATCHES:
        for arm in args.arms:
            path = args.out / f"{arm}_{batch}_seed{args.seed}.json"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(configuration(arm, batch, args.seed, args.microbatch), indent=2) + "\n")
    print(f"Wrote {len(args.arms) * len(BATCHES)} explicit configurations to {args.out}")


if __name__ == "__main__":
    main()
