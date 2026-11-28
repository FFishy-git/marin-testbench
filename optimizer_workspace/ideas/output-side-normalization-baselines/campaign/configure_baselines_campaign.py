# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate 130M 1x Chinchilla configs for the five Muon-family baselines."""

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
    ("0p008", 0.008),
    ("0p0113", 0.016 / math.sqrt(2.0)),
    ("0p016", 0.016),
    ("0p0226", 0.02262741696),
    ("0p032", 0.032),
]
LR_BY_TAG: dict[str, float] = dict(LR_GRID)
STAGE1_LRS: tuple[str, ...] = ("0p0113", "0p016", "0p0226")
BOUNDARY_LRS: dict[str, str] = {"0p0113": "0p008", "0p0226": "0p032"}

# Core 5 baselines + optional secondary variants (muon2_f, muoneq_rc)
PRIMARY_ARMS: tuple[str, ...] = (
    "muown",
    "normuon",
    "muon2",
    "muoneq_r",
    "newton_muon",
)
SECONDARY_ARMS: tuple[str, ...] = (
    "muon2_f",
    "muoneq_rc",
)

BASELINE_SPECS: dict[str, dict] = {
    "muown": {
        "label": "Muown (Row-Norm Control, arXiv:2605.10797)",
        "source": "arXiv:2605.10797 / kcc-lion/muown (also MiMo-V2.6 arXiv:2610.11959v1)",
        "optimizer_name": "muon",
        "baseline": {
            "g_betas": [0.9, 0.95],
            "g_eps": 1e-8,
            # Matches Marin auxiliary_lr / lr (0.0048 / 0.016 = 0.30)
            "g_lr_ratio": 0.30,
        },
    },
    "normuon": {
        "label": "NorMuon (Neuron-wise Normalized Muon, arXiv:2510.05491)",
        "source": "arXiv:2510.05491 / zichongli5/NorMuon",
        "optimizer_name": "muon",
        "baseline": {
            "beta2": 0.95,
            "normuon_eps": 1e-10,
        },
    },
    "muon2": {
        "label": "Muon^2 (Adaptive 2nd-Moment Pre-NS, arXiv:2604.09967)",
        "source": "arXiv:2604.09967 Algorithm 1",
        "optimizer_name": "muon",
        "baseline": {
            "factorized": False,
            "beta2": 0.95,
            "muon2_eps": 1e-8,
            "use_nesterov": True,
        },
    },
    "muon2_f": {
        "label": "Muon^2-F (Factorized 2nd-Moment Pre-NS, arXiv:2604.09967)",
        "source": "arXiv:2604.09967 Algorithm 2",
        "optimizer_name": "muon",
        "baseline": {
            "factorized": True,
            "beta2": 0.95,
            "muon2_eps": 1e-8,
            "use_nesterov": True,
        },
    },
    "muoneq_r": {
        "label": "MuonEq-R (Row Equilibration Pre-NS, arXiv:2603.28254)",
        "source": "arXiv:2603.28254 / MaeChd/muon-eq (default LLM hidden-weight mode)",
        "optimizer_name": "muon",
        "baseline": {
            "normalize_mode": "row",
            "rowcol_scale_exponent": -0.5,
            "rowcol_eps": 1e-8,
            "rowcol_clip": None,
        },
    },
    "muoneq_rc": {
        "label": "MuonEq-RC (Row+Col Equilibration Pre-NS, arXiv:2603.28254)",
        "source": "arXiv:2603.28254 / MaeChd/muon-eq (two-sided RC mode)",
        "optimizer_name": "muon",
        "baseline": {
            "normalize_mode": "rowcol",
            "rowcol_scale_exponent": -0.5,
            "rowcol_eps": 1e-8,
            "rowcol_clip": None,
        },
    },
    "newton_muon": {
        "label": "Newton-Muon (Input-Cov Right Inverse Pre-NS, arXiv:2604.01472)",
        "source": "arXiv:2604.01472 / zhehangdu/Newton-Muon",
        "optimizer_name": "muont",
        "baseline": {
            "refresh_every": 32,
            "precond_ewma": 0.95,
            "precond_ridge_mult": 0.2,
            "precond_eps": 1e-8,
            "precond_init_diag": 0.001,
            "down_block_diag": True,
        },
    },
}


def run_name(arm_key: str, seed: int, lr_tag: str) -> str:
    return f"{arm_key}_130m_1x_seed{seed}_lr{lr_tag}"


def build_baseline_130m_config(
    *,
    arm_key: str,
    seed: int = 0,
    lr: float = 0.016,
    steps: int = 5120,
    weight_decay: float = 0.1,
    ns_steps: int = 10,
    ns_coeffs: str = "simple",
    ns_eps: float = 1e-5,
) -> dict:
    spec = BASELINE_SPECS[arm_key]
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

    config["training"]["seed"] = int(seed)
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

    config["optimizer"]["name"] = spec["optimizer_name"]
    config["optimizer"]["lr"] = float(lr)
    config["optimizer"]["auxiliary_lr"] = 0.0048
    config["optimizer"]["weight_decay"] = float(weight_decay)
    config["optimizer"]["ns_steps"] = int(ns_steps)
    config["optimizer"]["ns_coeffs"] = str(ns_coeffs)
    config["optimizer"]["ns_eps"] = float(ns_eps)

    if "root" in config:
        config["root"]["chunk"] = 0

    config["experiment"] = {
        "arm": arm_key,
        "arm_label": spec["label"],
        "source": spec["source"],
        "baseline": dict(spec["baseline"]),
        "upstream_revision": REVISION,
        "output": asdict(OutputSettings()),
        "checkpoint_every": max(1, steps // 2),
        "diagnostic_every": 250,
        "expected_model_parameters": EXPECTED_PARAMS_130M,
        "hardware_adaptation": "A100 FA2; world size 8; microbatch 4; root.chunk=0; model=130m, chinchilla=1x",
    }
    return config


def generate_all(out_dir: Path | None = None) -> dict[str, dict]:
    target = out_dir or (TASK / "configs")
    target.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    for seed in (0, 1, 2, 3):
        for lr_tag, lr_val in LR_GRID:
            for arm_key in (*PRIMARY_ARMS, *SECONDARY_ARMS):
                name = run_name(arm_key, seed, lr_tag)
                cfg = build_baseline_130m_config(arm_key=arm_key, seed=seed, lr=lr_val)
                cfg_path = target / f"seed{seed}" / f"{name}.json"
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
                index[name] = {
                    "name": name,
                    "arm_key": arm_key,
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
