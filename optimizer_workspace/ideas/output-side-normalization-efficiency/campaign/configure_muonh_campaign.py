# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-generate 130M 1x Chinchilla JSON configs across Hyperball arms (Toner-matched & Marin-exact), LR grid, and seeds 0-3."""

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
    ("0p008", 0.008),                     # 0.00800000000000000  (lower adaptive boundary guard if 0.01131 wins)
    ("0p0113", 0.016 / math.sqrt(2.0)),   # 0.01131370849898476  (Marin core lower bracket)
    ("0p016", 0.016),                     # 0.01600000000000000  (Marin May Recipe 0.01531 / Aug Refit 0.01744)
    ("0p0226", 0.02262741696),            # 0.02262741696000000  (Marin core upper bracket: sqrt(2) * 0.016)
    ("0p032", 0.032),                     # 0.03200000000000000  (upper adaptive boundary guard if 0.02263 wins)
    ("0p0453", 0.032 * math.sqrt(2.0)),   # 0.04525483399593904  (retained for index compatibility)
    ("0p064", 0.064),                     # 0.06400000000000000  (retained for index compatibility)
]
LR_BY_TAG: dict[str, float] = dict(LR_GRID)
CORE_STAGE1_LRS: tuple[str, ...] = ("0p0226", "0p032", "0p016", "0p0453")
MARIN_STAGE1_LRS: tuple[str, ...] = ("0p016", "0p0226", "0p0113")

# Map logical arm_key -> (BlockToner arm, optional decay_mode, optional head_mode, optional recipe)
# Note: existing arm_keys keep recipe=None so their JSON configs stay byte-for-byte identical!
ARM_SPECS: dict[str, tuple[str, str | None, str | None, str | None]] = {
    "wt_mlp_lag": ("inner_outer_mlpdiag_lag", None, None, None),
    # Controlled Matrix-Hyperball arms under Toner WSD + AdamW head
    "muonh": ("none_all", "hyperball", None, None),
    "muonth": ("input_only", "hyperball", None, None),
    "wt_mlp_lag_muonh": ("inner_outer_mlpdiag_lag", "hyperball", None, None),
    # Full-Hyperball triad under Toner WSD
    "muonh_adamh": ("none_all", "hyperball", "adamh", None),
    "muonth_adamh": ("input_only", "hyperball", "adamh", None),
    "wt_mlp_lag_muonh_adamh": ("inner_outer_mlpdiag_lag", "hyperball", "adamh", None),
    # Canonical Marin-Exact MuonH Recipe (1pct-noclip schedule + AdamH head at lr + 3/13 Adam LR + quintic-5 NS)
    "muonh_marin": ("none_all", "hyperball", "adamh", "marin"),
    "muonth_marin": ("input_only", "hyperball", "adamh", "marin"),
    "wt_mlp_lag_muonh_marin": ("inner_outer_mlpdiag_lag", "hyperball", "adamh", "marin"),
}


def run_name(arm_key: str, seed: int, lr_tag: str) -> str:
    return f"{arm_key}_130m_1x_seed{seed}_lr{lr_tag}"


def build_130m_config(*, arm_key: str, seed: int, lr: float) -> dict:
    arm, decay_mode, head_mode, recipe = ARM_SPECS[arm_key]
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
    if decay_mode == "hyperball":
        config["optimizer"]["weight_decay"] = 0.0
    if head_mode == "adamh":
        config["optimizer"]["head_weight_decay"] = 0.0
        config["optimizer"]["head_lr"] = lr

    if recipe == "marin":
        tokens_per_batch = int(config["training"]["batch_tokens"])   # 524,288
        tokens = float(config["training"]["tokens"])                 # 2,684,354,560
        steps = int(config["training"]["steps"])                     # 5,120
        warmup_steps = int(0.01 * steps)                             # 51 (1% warmup)
        config["optimizer"]["auxiliary_lr"] = lr * 3.0 / 13.0        # muonh_ratio = 13 / 3
        config["optimizer"]["betas"] = [0.9, 0.999 ** (tokens_per_batch / 131_072.0)]
        config["optimizer"]["eps"] = 9.676e-18 * math.sqrt(tokens / tokens_per_batch)
        config["optimizer"]["clip_norm"] = 0.0                       # 1pct-noclip: max_grad_norm = None
        config["optimizer"]["momentum"] = 0.95
        config["optimizer"]["nesterov"] = True
        config["optimizer"]["ns_steps"] = 5
        config["optimizer"]["ns_eps"] = 1e-8
        config["optimizer"]["ns_mode"] = "tpu"
        config["optimizer"]["ns_coeffs"] = "quintic"
        config["schedule"] = {
            "warmup_steps": warmup_steps,
            "stable_steps": 0,
            "decay_steps": steps - warmup_steps,
            "kind": "linear",
            "min_lr_ratio": 0.05,
        }

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
    if head_mode is not None:
        exp_cfg["head_mode"] = head_mode
    if recipe == "marin":
        exp_cfg["recipe"] = "marin_1pct_noclip"
    config["experiment"] = exp_cfg
    return config


def generate_all(out_dir: Path | None = None) -> dict[str, dict]:
    target = out_dir or (TASK / "configs")
    target.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    for seed in (0, 1, 2, 3):
        for lr_tag, lr_val in LR_GRID:
            for arm_key, (arm, decay_mode, head_mode, recipe) in ARM_SPECS.items():
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
                    "head_mode": head_mode or "adamw",
                    "recipe": recipe or "toner_wsd",
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
