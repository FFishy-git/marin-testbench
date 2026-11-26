# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-generate 130M 1x Chinchilla JSON configs across weight-decay modes, LRs, and seeds 0-3."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from experiment.factors import OutputSettings
from experiment.reference import REVISION, UPSTREAM

EXPECTED_PARAMS_130M = 265_585_152

LR_GRID: list[tuple[str, float]] = [
    ("0p0113", 0.016 / math.sqrt(2.0)),  # 0.01131370849898476
    ("0p016", 0.016),                    # 0.01600000000000000
    ("0p0226", 0.02262741696),           # 0.02262741696000000
    ("0p032", 0.032),                    # 0.03200000000000000
]
LR_BY_TAG: dict[str, float] = dict(LR_GRID)

# Map logical arm_key -> (BlockToner arm, optional decay_mode override)
# Note: "wt_mlp_lag" omits decay_mode so its JSON config is byte-identical to wave1a/lr-bracket!
ARM_SPECS: dict[str, tuple[str, str | None]] = {
    "wt_mlp_lag": ("inner_outer_mlpdiag_lag", None),
    "wt_mlp_lag_wd_iso": ("inner_outer_mlpdiag_lag", "isotropic"),
    "wt_mlp_lag_wd_lr2": ("inner_outer_mlpdiag_lag", "two_sided"),
}


def run_name(arm_key: str, seed: int, lr_tag: str) -> str:
    return f"{arm_key}_130m_1x_seed{seed}_lr{lr_tag}"


def build_130m_config(*, arm_key: str, seed: int, lr: float) -> dict:
    arm, decay_mode = ARM_SPECS[arm_key]
    base_path = UPSTREAM / "reproduce" / "marin" / "configs" / "130m" / "muont_0p5m.json"
    config = json.loads(base_path.read_text())

    config["runtime"]["attention"] = "FA2"
    config["runtime"]["deterministic"] = True
    config["runtime"]["compile_options"] = {
        "max_autotune": True,
        "triton.cudagraphs": False,
        "max_autotune_gemm_backends": "ATEN",
        "emulate_precision_casts": True,
    }
    config["runtime"]["eval_every"] = 250

    config["training"]["seed"] = seed
    config["training"]["micro_batch_size"] = 4

    config["optimizer"]["lr"] = lr
    config["optimizer"]["auxiliary_lr"] = 0.0048

    if "root" in config:
        config["root"]["chunk"] = 0

    exp_cfg: dict = {
        "arm": arm,
        "arm_label": arm_key,
        "upstream_revision": REVISION,
        "output": asdict(OutputSettings()),
        "checkpoint_every": 2560,
        "diagnostic_every": 250,
        "expected_model_parameters": EXPECTED_PARAMS_130M,
        "hardware_adaptation": "A100 FA2; world size 8; microbatch 4; root.chunk=0; model=130m, chinchilla=1x",
    }
    if decay_mode is not None:
        exp_cfg["decay_mode"] = decay_mode
    config["experiment"] = exp_cfg
    return config


def generate_all(out_dir: Path | None = None) -> dict[str, dict]:
    target = out_dir or (TASK / "configs")
    target.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    for seed in (0, 1, 2, 3):
        for lr_tag, lr_val in LR_GRID:
            for arm_key, (arm, decay_mode) in ARM_SPECS.items():
                name = run_name(arm_key, seed, lr_tag)
                cfg = build_130m_config(arm_key=arm_key, seed=seed, lr=lr_val)
                cfg_path = target / f"seed{seed}" / f"{name}.json"
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
                index[name] = {
                    "name": name,
                    "arm_key": arm_key,
                    "arm": arm,
                    "decay_mode": decay_mode or "default",
                    "seed": seed,
                    "lr_tag": lr_tag,
                    "lr": lr_val,
                    "config": str(cfg_path),
                }
    manifest_path = target / "all_configs_index.json"
    manifest_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Generated {len(index)} configurations in {target}")
    return index


if __name__ == "__main__":
    generate_all()
