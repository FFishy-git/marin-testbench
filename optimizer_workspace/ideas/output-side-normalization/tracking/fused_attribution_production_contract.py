# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library execution contract for three full-budget attribution waves."""

from pathlib import Path

FAMILIES = ("muont", "outer_leftdiag_fused", "null_leftdiag_fused")
WAVES = ("wave13", "wave14", "wave15")
LR_GRID = (("lower", 0.04525483392), ("center", 0.06399999989260598), ("upper", 0.09050966784))
ACTIVE_NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
WORLD, STEPS, BATCH_TOKENS, TOKENS = 8, 5120, 524288, 2684354560
ENVIRONMENT = {
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
}
PREPARATION = "fused-attribution-production-preparation.json"
PREPARED_STATUS = "fused_attribution_production_prepared_gpu_denied"
CPU_STATUS = "fused_attribution_production_cpu_verified_gpu_denied"
FROZEN_STATUS = "frozen_fused_attribution_full_budget_production"
CONTROLLER_CPU_STATUS = "fused_attribution_production_controller_cpu_verified_gpu_denied"
ACCEPTANCE_STATUS = "fused_attribution_validation_reconstructed_for_production"
WORKER_MARKER = "OUTPUT_MARIN_FUSED_ATTRIBUTION_PRODUCTION_WORKER"
CHILD_MARKER = "OUTPUT_MARIN_FUSED_ATTRIBUTION_PRODUCTION_CHILD"
MONITOR_INTERVAL = 900
TRAINING_TIMEOUT = 21600
SCHEDULE = {
    "warmup_steps": 0,
    "stable_steps": 1024,
    "decay_steps": 4096,
    "kind": "linear",
    "min_lr_ratio": 0.0,
}


def wave_spec(wave: str) -> dict:
    if wave not in WAVES:
        raise ValueError("Unreleased attribution wave")
    index = WAVES.index(wave)
    label, lr = LR_GRID[index]
    return {
        "wave": wave,
        "lr_label": label,
        "matrix_lr": lr,
        "families": list(FAMILIES),
        "nodes": list(ACTIVE_NODES[index:] + ACTIVE_NODES[:index]),
        "predecessor_wave": WAVES[index - 1] if index else None,
    }


def paths(root: Path, wave: str) -> tuple[Path, Path, Path, Path]:
    wave_spec(wave)
    return (
        root / f"run-preparation/{wave}-launch-binding.json",
        root / f"controllers/{wave}-fused-attribution-20261118/tracking",
        root / f"run-preparation/{wave}-launch-state.json",
        root / f"run-preparation/{wave}-workers",
    )


def node_for(wave: str, family: str) -> str:
    if family not in FAMILIES:
        raise ValueError("Unreleased attribution family")
    return wave_spec(wave)["nodes"][FAMILIES.index(family)]


def config_name(family: str, label: str) -> str:
    if family not in FAMILIES or label not in {item[0] for item in LR_GRID}:
        raise ValueError("Unknown attribution configuration")
    return f"experiment/configs/{family}_full_lr_{label}_0p5m_seed0.json"


def output_path(root: Path, wave: str, family: str) -> Path:
    node_for(wave, family)
    return root / "runs" / wave / family


def worker_path(root: Path, wave: str, family: str) -> Path:
    node_for(wave, family)
    return paths(root, wave)[3] / f"{family}.json"


def worker_command(root: Path, wave: str, family: str) -> list[str]:
    node_for(wave, family)
    return [
        PYTHON,
        "-P",
        str(paths(root, wave)[1] / "fused_attribution_production.py"),
        "--root",
        str(root),
        "--wave",
        wave,
        "--phase",
        "worker",
        "--family",
        family,
    ]


def commands(root: Path, candidate: Path, wave: str) -> dict[str, list[str]]:
    spec = wave_spec(wave)
    return {
        node_for(wave, family): [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "experiment.train",
            "--config",
            str(candidate / config_name(family, spec["lr_label"])),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--out",
            str(output_path(root, wave, family)),
            "--launch-release",
            str(paths(root, wave)[0]),
        ]
        for family in FAMILIES
    }


def require_config(config: dict, wave: str, family: str) -> None:
    spec = wave_spec(wave)
    training, runtime, optimizer = config["training"], config["runtime"], config["optimizer"]
    if (
        family not in FAMILIES
        or config["experiment"]["arm"] != family
        or any(type(training[key]) is not int for key in ("steps", "tokens", "batch_tokens", "world_size", "seed"))
        or training["steps"] != STEPS
        or training["tokens"] != TOKENS
        or training["batch_tokens"] != BATCH_TOKENS
        or training["world_size"] != WORLD
        or training["seed"] != 0
        or training["global_batch_sequences"] * config["model"]["sequence_length"] != BATCH_TOKENS
        or config["schedule"] != SCHEDULE
        or optimizer["lr"] != spec["matrix_lr"]
        or optimizer["ns_mode"] != "tpu"
        or optimizer["ns_steps"] != 10
        or config["experiment"]["refresh_install_age"] != 1
        or runtime["compile"] == "eager"
        or runtime["attention"] != "FA2"
        or runtime["head"] != "compiled"
        or config["root"]["graphs"] is not True
        or (family != "muont" and config["experiment"]["diagonal_backend"] != "triton")
    ):
        raise ValueError("Production configuration differs from the validated full-schedule LR grid")
