# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate explicit JSON configurations for all 4 phases of the confirmation campaign."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from experiment.factors import OutputSettings
from experiment.reference import REVISION, UPSTREAM

EXPECTED_PARAMS = {
    "130m": 265_585_152,
    "520m": 799_605_760,
}

PINNED_130M_LR = {
    "muon": 0.016,
    "muont": 0.02262741696,
    "od_all": 0.02262741696,
    "wt_mlp_lag": 0.02262741696,
}

ARM_MAP = {
    "muon": "muon",
    "muont": "muont",
    "od_all": "inner_outer_leftdiag",
    "wt_mlp_lag": "inner_outer_mlpdiag_lag",
    "null_wt": "null_mlpdiag",
}


def build_config(
    *,
    model_size: str,
    batch: str,
    arm_key: str,
    seed: int,
    lr: float,
    auxiliary_lr: float | None = None,
    microbatch: int,
    world_size: int | None = None,
    chunk: int = 0,
    begin: str | None = None,
    host_workers: int | None = None,
    host_threads: int | None = None,
    chinchilla_multiplier: int = 1,
    eval_every: int | None = None,
    checkpoint_every: int | None = None,
) -> dict:
    arm = ARM_MAP[arm_key]
    base = "muon" if arm == "muon" else "muont"
    base_path = UPSTREAM / "reproduce" / "marin" / "configs" / model_size / f"{base}_{batch}.json"
    config = json.loads(base_path.read_text())
    if config.get("engine") != "pytorch":
        raise ValueError(f"Expected PyTorch base config, got {config.get('engine')} at {base_path}")

    config["runtime"]["attention"] = "FA2"
    config["runtime"]["deterministic"] = True
    config["runtime"]["compile_options"] = {
        "max_autotune": True,
        "triton.cudagraphs": False,
        "max_autotune_gemm_backends": "ATEN",
        "emulate_precision_casts": True,
    }
    if eval_every is not None:
        config["runtime"]["eval_every"] = eval_every

    config["training"]["seed"] = seed
    config["training"]["micro_batch_size"] = microbatch
    if world_size is not None:
        config["training"]["world_size"] = world_size
    if chinchilla_multiplier != 1:
        config["training"]["steps"] *= chinchilla_multiplier
        config["training"]["tokens"] *= chinchilla_multiplier
        config["training"]["chinchilla_multiplier"] = chinchilla_multiplier
        config["schedule"]["stable_steps"] *= chinchilla_multiplier
        config["schedule"]["decay_steps"] *= chinchilla_multiplier

    config["optimizer"]["lr"] = lr
    if auxiliary_lr is not None:
        config["optimizer"]["auxiliary_lr"] = auxiliary_lr

    if "root" in config:
        config["root"]["chunk"] = chunk
        if begin is not None:
            config["root"]["begin"] = begin
        if host_workers is not None:
            config["root"]["host_workers"] = host_workers
        if host_threads is not None:
            config["root"]["host_threads"] = host_threads

    total = microbatch * config["training"]["world_size"]
    if config["training"]["global_batch_sequences"] % total != 0:
        raise ValueError(
            f"Microbatch {microbatch} * world {config['training']['world_size']} does not divide "
            f"global_batch_sequences {config['training']['global_batch_sequences']}"
        )
    accumulation = config["training"]["global_batch_sequences"] // total
    if chunk and accumulation % chunk != 0:
        raise ValueError(f"root.chunk={chunk} must divide accumulation={accumulation}")

    steps = config["training"]["steps"]
    if config["schedule"]["warmup_steps"] + config["schedule"]["stable_steps"] + config["schedule"]["decay_steps"] != steps:
        raise ValueError("Schedule steps do not sum to training.steps")

    config["experiment"] = {
        "arm": arm,
        "arm_label": arm_key,
        "upstream_revision": REVISION,
        "output": asdict(OutputSettings()),
        "checkpoint_every": checkpoint_every if checkpoint_every is not None else steps,
        "diagnostic_every": eval_every if eval_every is not None else config["runtime"]["eval_every"],
        "expected_model_parameters": EXPECTED_PARAMS[model_size],
        "hardware_adaptation": (
            f"A100 FA2; world size {config['training']['world_size']}; microbatch {microbatch}; root.chunk={chunk}; "
            f"model={model_size}, chinchilla={chinchilla_multiplier}x"
        ),
    }
    return config


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def generate_phase1(out_dir: Path) -> dict[str, list[dict]]:
    specs_1a = [
        ("muont_130m_1x_seed0_lr0p016", "muont", 0.016),
        ("muont_130m_1x_seed0_lr0p032", "muont", 0.032),
        ("wt_mlp_lag_130m_1x_seed0_lr0p016", "wt_mlp_lag", 0.016),
        ("wt_mlp_lag_130m_1x_seed0_lr0p032", "wt_mlp_lag", 0.032),
    ]
    specs_1b = [
        ("od_all_130m_1x_seed0_lr0p0226", "od_all", 0.02262741696),
        ("od_all_130m_1x_seed0_lr0p032", "od_all", 0.032),
        ("muon_130m_1x_seed0_lr0p0226", "muon", 0.02262741696),
        ("null_wt_130m_1x_seed0_lr0p0226", "null_wt", 0.02262741696),
    ]
    waves = {}
    for wave_name, specs in (("wave1a", specs_1a), ("wave1b", specs_1b)):
        entries = []
        for name, arm_key, lr in specs:
            cfg = build_config(
                model_size="130m",
                batch="0p5m",
                arm_key=arm_key,
                seed=0,
                lr=lr,
                auxiliary_lr=0.0048,
                microbatch=4,
                chinchilla_multiplier=1,
                eval_every=250,
                checkpoint_every=2560,
            )
            path = out_dir / "phase1" / f"{name}.json"
            write_json(path, cfg)
            entries.append({"name": name, "arm_key": arm_key, "arm": ARM_MAP[arm_key], "seed": 0, "lr": lr, "config": str(path)})
        waves[wave_name] = entries
    return waves


def generate_phase2(out_dir: Path, lrs: dict[str, float]) -> dict[str, list[dict]]:
    waves = {}
    for seed, wave_name in ((1, "wave2a"), (2, "wave2b"), (3, "wave2c")):
        entries = []
        for arm_key in ("muon", "muont", "od_all", "wt_mlp_lag"):
            name = f"{arm_key}_130m_1x_seed{seed}"
            cfg = build_config(
                model_size="130m",
                batch="0p5m",
                arm_key=arm_key,
                seed=seed,
                lr=lrs[arm_key],
                auxiliary_lr=0.0048,
                microbatch=4,
                chinchilla_multiplier=1,
                eval_every=250,
                checkpoint_every=2560,
            )
            path = out_dir / "phase2" / f"{name}.json"
            write_json(path, cfg)
            entries.append({"name": name, "arm_key": arm_key, "arm": ARM_MAP[arm_key], "seed": seed, "lr": lrs[arm_key], "config": str(path)})
        waves[wave_name] = entries
    return waves


def generate_phase3(out_dir: Path, lrs: dict[str, float]) -> dict[str, list[dict]]:
    scale = 4.0 ** (-0.3)
    aux_lr = 0.0048 * scale
    entries = []
    for arm_key in ("muon", "muont", "od_all", "wt_mlp_lag"):
        name = f"{arm_key}_130m_4x_seed0"
        lr = lrs[arm_key] * scale
        cfg = build_config(
            model_size="130m",
            batch="0p5m",
            arm_key=arm_key,
            seed=0,
            lr=lr,
            auxiliary_lr=aux_lr,
            microbatch=4,
            chinchilla_multiplier=4,
            eval_every=1000,
            checkpoint_every=5120,
        )
        path = out_dir / "phase3" / f"{name}.json"
        write_json(path, cfg)
        entries.append({"name": name, "arm_key": arm_key, "arm": ARM_MAP[arm_key], "seed": 0, "lr": lr, "config": str(path)})
    return {"wave3a": entries}


def generate_phase4(
    out_dir: Path,
    microbatch_520m: int,
    *,
    chunk_520m: int = 0,
    host_workers_520m: int | None = None,
    host_threads_520m: int | None = None,
) -> dict[str, list[dict]]:
    lrs_520m = {
        "muon": 0.016,
        "muont": 0.02262741696,
        "od_all": 0.02262741696,
        "wt_mlp_lag": 0.02262741696,
    }
    entries = []
    for arm_key in ("muon", "muont", "od_all", "wt_mlp_lag"):
        name = f"{arm_key}_520m_1x_seed0"
        cfg = build_config(
            model_size="520m",
            batch="1m",
            arm_key=arm_key,
            seed=0,
            lr=lrs_520m[arm_key],
            auxiliary_lr=0.0024,
            microbatch=microbatch_520m,
            chunk=chunk_520m,
            host_workers=host_workers_520m,
            host_threads=host_threads_520m,
            chinchilla_multiplier=1,
            eval_every=500,
            checkpoint_every=2560,
        )
        path = out_dir / "phase4" / f"{name}.json"
        write_json(path, cfg)
        entries.append({"name": name, "arm_key": arm_key, "arm": ARM_MAP[arm_key], "seed": 0, "lr": lrs_520m[arm_key], "config": str(path)})
    return {"wave4a": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=TASK / "configs")
    parser.add_argument("--lr-muon", type=float, default=PINNED_130M_LR["muon"])
    parser.add_argument("--lr-muont", type=float, default=PINNED_130M_LR["muont"])
    parser.add_argument("--lr-od-all", type=float, default=PINNED_130M_LR["od_all"])
    parser.add_argument("--lr-wt-mlp-lag", type=float, default=PINNED_130M_LR["wt_mlp_lag"])
    parser.add_argument("--microbatch-520m", type=int, default=2)
    parser.add_argument("--chunk-520m", type=int, default=2)
    parser.add_argument("--host-workers-520m", type=int, default=4)
    parser.add_argument("--host-threads-520m", type=int, default=6)
    parser.add_argument("--phases", nargs="+", choices=("1", "2", "3", "4"), default=["1", "2", "3", "4"])
    args = parser.parse_args()

    lrs = {
        "muon": args.lr_muon,
        "muont": args.lr_muont,
        "od_all": args.lr_od_all,
        "wt_mlp_lag": args.lr_wt_mlp_lag,
    }
    manifest_path = args.out / "waves_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if "1" in args.phases:
        manifest.update(generate_phase1(args.out))
    if "2" in args.phases:
        manifest.update(generate_phase2(args.out, lrs))
    if "3" in args.phases:
        manifest.update(generate_phase3(args.out, lrs))
    if "4" in args.phases:
        manifest.update(
            generate_phase4(
                args.out,
                args.microbatch_520m,
                chunk_520m=args.chunk_520m,
                host_workers_520m=args.host_workers_520m,
                host_threads_520m=args.host_threads_520m,
            )
        )
    write_json(manifest_path, manifest)
    total_runs = sum(len(v) for v in manifest.values())
    print(f"Wrote {total_runs} run configs across {len(manifest)} waves to {args.out}")


if __name__ == "__main__":
    main()
