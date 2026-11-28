# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration builder for 130M 1x Chinchilla efficiency & loss-Pareto research."""

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

LR_BY_TAG: dict[str, float] = {
    "0p008": 0.008,
    "0p0113": 0.016 / math.sqrt(2.0),
    "0p0135": 0.016 / (2.0**0.25),
    "0p016": 0.016,
    "0p019": 0.016 * (2.0**0.25),
    "0p0226": 0.02262741696,
    "0p032": 0.032,
}


def build_eff_130m_config(
    *,
    label: str,
    arm: str,
    seed: int = 0,
    lr: float = 0.016,
    steps: int = 5120,
    decay_mode: str | None = None,
    head_mode: str | None = None,
    weight_decay: float | None = None,
    lazy: dict | None = None,
    output_overrides: dict | None = None,
    compile_ns: bool = False,
    ns_steps: int = 10,
    ns_coeffs: str = "simple",
    ns_eps: float = 1e-5,
) -> dict:
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
    if steps != 5120:
        batch_tokens = int(config["training"]["batch_tokens"])
        config["training"]["steps"] = int(steps)
        config["training"]["tokens"] = int(steps) * batch_tokens
        stable = int(round(0.20 * steps))
        config["schedule"] = {
            "warmup_steps": 0,
            "stable_steps": stable,
            "decay_steps": int(steps) - stable,
            "kind": "linear",
            "min_lr_ratio": 0.0,
        }

    config["optimizer"]["lr"] = float(lr)
    config["optimizer"]["auxiliary_lr"] = 0.0048
    config["optimizer"]["ns_steps"] = int(ns_steps)
    config["optimizer"]["ns_coeffs"] = str(ns_coeffs)
    config["optimizer"]["ns_eps"] = float(ns_eps)
    if compile_ns:
        config["optimizer"]["compile_ns"] = True

    if decay_mode == "hyperball":
        config["optimizer"]["weight_decay"] = 0.0
    elif weight_decay is not None:
        config["optimizer"]["weight_decay"] = float(weight_decay)

    if head_mode == "adamh":
        config["optimizer"]["head_weight_decay"] = 0.0
        config["optimizer"]["head_lr"] = float(lr)

    if "root" in config:
        config["root"]["chunk"] = 0

    out_settings = asdict(OutputSettings())
    allow_custom_output = False
    if output_overrides:
        out_settings.update(output_overrides)
        allow_custom_output = True

    exp_cfg: dict = {
        "arm": arm,
        "arm_label": label,
        "upstream_revision": REVISION,
        "output": out_settings,
        "checkpoint_every": max(1, steps // 2),
        "diagnostic_every": 250,
        "expected_model_parameters": EXPECTED_PARAMS_130M,
        "hardware_adaptation": "A100 FA2; world size 8; microbatch 4; root.chunk=0; model=130m, chinchilla=1x",
    }
    if allow_custom_output:
        exp_cfg["allow_custom_output"] = True
    if decay_mode is not None:
        exp_cfg["decay_mode"] = decay_mode
    if head_mode is not None:
        exp_cfg["head_mode"] = head_mode
    if lazy is not None:
        exp_cfg["lazy"] = lazy
    config["experiment"] = exp_cfg
    return config
