# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist scaled CPU state trees without changing the original training sources."""

import hashlib
import json
import random
import shutil
from pathlib import Path

import ns_training_path_contract as contract
import ns_training_path_validation as validation
import torch
from analyze_tuning import executed_sources
from fused_attribution_validation import model_shapes, save_interrupted_manifest
from selected_lr_release_owner import candidate_files, read, sha256

PRESERVED = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/"
    "candidates/ns-full-model-owner-single-pass-20261119T0257"
)


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def candidate(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Copy actual configuration/source inputs, scaling only model dimensions for CPU states."""
    root, path = tmp_path / "root", tmp_path / "candidate"
    archived = Path(__file__).resolve().parents[1]
    source = archived if (archived / contract.specs()[0]["config_name"]).is_file() else PRESERVED
    for directory in ("experiment", "artifacts/pinned-sources-materialized"):
        shutil.copytree(source / directory, path / directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (path / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    names = {
        *(spec["config_name"] for spec in contract.specs()),
        *(f"experiment/configs/{spec['family']}_full_lr_lower_0p5m_seed0.json" for spec in contract.specs()),
    }
    for name in names:
        config = read(path / name)
        config["model"].update(layers=8, hidden=4, mlp=8, vocab_size=16)
        record(path / name, config)
    record(root / "data/experiment_receipt.json", {"data": "training-path scaled CPU fixture"})
    frozen = {
        "status": contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(path),
        "files_sha256": candidate_files(path),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "plan": validation.plan(root, path),
    }
    frozen["commands"] = frozen["plan"]["commands"]
    record(contract.locations(root)[0], frozen)
    return root, path, frozen


def state(config: dict, step: int, rank: int, identity: dict) -> dict:
    """Build complete small rank states accepted by the unchanged numerical readers."""
    hidden, mlp, layers = (config["model"][key] for key in ("hidden", "mlp", "layers"))
    groups = {}
    for name, count, width in (("d", 3, hidden), ("f", 1, mlp)):
        groups[name] = {
            "C": torch.eye(width).expand(count, width, width).clone(),
            "tol": torch.full((count, 3), 1e-4),
            "basis_step": [20] * count,
            "warm": True,
            "cached_root_graph_active": True,
            "SB": torch.ones(count, 2 * width, width, dtype=torch.float16),
            "S2": torch.ones(count, width, 2 * width, dtype=torch.float16),
            "pending": [
                {"step": step - 1, "basis": torch.eye(width), "tol": torch.full((3,), 1e-4)},
                *([None] * (count - 1)),
            ],
        }
    momentum = {
        (hidden, hidden): torch.full((4, hidden, hidden), float(rank)),
        (mlp, hidden): torch.zeros(2, mlp, hidden),
        (hidden, mlp): torch.zeros(1, hidden, mlp),
    }
    weight = 1 - config["statistics"]["momentum"] ** step
    engine = {
        "groups": groups,
        "momentum": {},
        "weight": weight,
        "steps": step,
        "t": step,
        "fallbacks": 0,
        "extra": 0,
        "refresh_install_age": 1,
    }
    if config["experiment"]["arm"] == "muont":
        engine["momentum"] = {shape: {"m": value} for shape, value in momentum.items()}
        fast = engine
    else:
        fast = {
            "lagged": {},
            "input": engine,
            "output": None,
            "input_diagonal": None,
            "output_diagonal": {
                "moment": {hidden: torch.ones(5 * layers, hidden), mlp: torch.ones(2 * layers, mlp)},
                "weight": weight,
            },
            "momentum": momentum,
            "steps": step,
        }
    shapes = model_shapes(config)
    names = [
        ["head.weight"],
        [
            "body.emb.weight",
            *(f"body.blocks.{layer}.{kind}.weight" for layer in range(layers) for kind in ("ln1", "ln2")),
            "body.norm.weight",
        ],
    ]
    auxiliary, parameter_groups, index = {}, [], 0
    settings = config["optimizer"]
    for group_index, parameters in enumerate(names):
        indices = []
        for name in parameters:
            indices.append(index)
            auxiliary[index] = {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros(shapes[name]),
                "exp_avg_sq": torch.ones(shapes[name]),
            }
            index += 1
        parameter_groups.append(
            {
                "params": indices,
                "lr": settings["auxiliary_lr"],
                "base_lr": settings["auxiliary_lr"],
                "betas": tuple(settings["betas"]),
                "eps": settings["eps"],
                "weight_decay": settings["head_weight_decay"] if group_index == 0 else 0.0,
                "fused": True,
                "amsgrad": False,
            }
        )
    saved = {
        "step": step,
        "next_global_sequence": step * config["training"]["global_batch_sequences"],
        "config": config,
        "rank": rank,
        "world": contract.WORLD,
        "identity": identity,
        "optimizers": [None, {"state": auxiliary, "param_groups": parameter_groups}],
        "fast": fast,
        "output": None,
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.zeros(16, dtype=torch.uint8),
        "python_rng": random.Random(rank + step).getstate(),
    }
    if rank == 0:
        saved["model"] = {name: torch.zeros(shape) for name, shape in shapes.items()}
    return saved


def saved_trial(candidate: Path, frozen: dict, item: dict) -> None:
    """Persist independent interrupted history and all three complete checkpoint steps."""
    spec = {key: item[key] for key in contract.specs()[0]}
    config = contract.require_config(candidate, spec)
    identity = {
        "schema": 1,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_sha256": executed_sources(candidate, frozen),
        "data_receipt_sha256": frozen["data_receipt_sha256"],
    }
    rows = []
    for step in range(1, 25):
        row = {
            "step": step,
            "tokens": step * contract.BATCH_TOKENS,
            "train_loss": 4.0 - step / 1000,
            "input_refresh_installations": (
                [{"installation_step": step - 1, "source_step": step - 2}] if step >= 2 else []
            ),
            "diagonal_capture_parity_passed": True,
            "outer_diagonal_update_parity_passed": True,
            "output_work_retained_passed": True,
            "null_identity_outer_passed": True,
        }
        if step in (1, 22, 24):
            row.update(
                c4_loss=3.5,
                selection_loss=3.4,
                heldout_losses={f"heldout/{name}": 3.4 for name in config["data"]["mixture_weights"]},
            )
        rows.append(row)
    continuous, resumed = Path(item["continuous"]), Path(item["resumed"])
    for directory, endpoint in ((continuous, 24), (resumed, 22)):
        record(directory / "config.json", config)
        record(
            directory / "run.json",
            {
                "status": "execution_check_complete",
                "execution_check": True,
                "config": config,
                "source_sha256": identity["source_sha256"],
                "data_receipt_sha256": frozen["data_receipt_sha256"],
                "completed_steps": endpoint,
                "resume": None,
            },
        )
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:endpoint]))
        for step in contract.CHECKPOINTS:
            if step > endpoint:
                continue
            folder = directory / f"checkpoints/step_{step:06d}"
            record(folder / "complete.json", {"step": step, "world": contract.WORLD})
            for rank in range(contract.WORLD):
                torch.save(state(config, step, rank, identity), folder / f"rank_{rank}.pt")
    save_interrupted_manifest(resumed)
    manifest = read(resumed / "run.json")
    manifest.update(completed_steps=24, resume=str(resumed / "checkpoints/step_000022"))
    record(resumed / "run.json", manifest)
    (resumed / "metrics_from_000022.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[22:]))
    for step in (23, 24):
        shutil.copytree(continuous / f"checkpoints/step_{step:06d}", resumed / f"checkpoints/step_{step:06d}")


def complete_receipt(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Persist all 54 command receipts, including one actual negative continuation result."""
    root, path, frozen = candidate(tmp_path)
    planned = validation.plan(root, path)
    owner = {
        "controller_pid": 301,
        "controller_start_ticks": "controller-fixture",
        "validator_pid": 302,
        "validator_start_ticks": "validator-fixture",
    }
    entries, measured = [], []
    work = contract.locations(root)[3]
    for index, item in enumerate(planned["trials"]):
        saved_trial(path, frozen, item)
        if index == 0:
            checkpoint_path = Path(item["resumed"]) / "checkpoints/step_000024/rank_7.pt"
            value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            value["fast"]["momentum"][(4, 4)]["m"][0, 0, 0] += 0.1
            torch.save(value, checkpoint_path)
        decision = validation.trial(item, path, frozen)
        for offset, (mode, command) in enumerate(zip(contract.MODES, item["commands"], strict=True)):
            log = work / f"{item['name']}_{mode}.log"
            log.write_text("CPU external command receipt fixture; no training executed\n")
            pid = 1000 + index * 3 + offset
            entry = {
                "command": command,
                "marker": f"training-path-command-{pid}",
                "spawn_status": "acknowledged",
                "generation": {
                    "pid": pid,
                    "command": command,
                    "parent": owner["validator_pid"],
                    "start_ticks": str(pid + 900),
                    "group": pid,
                    "session": pid,
                },
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "generations": {},
                "log": str(log),
                "log_sha256": sha256(log),
            }
            record(log.with_suffix(".execution.json"), entry)
            entries.append(entry | {"trial": item["name"], "mode": mode, "trajectory": decision["trajectories"][mode]})
        measured.append(decision)
    record(
        contract.locations(root)[2],
        {
            "controller_pid": owner["controller_pid"],
            "controller_start_ticks": owner["controller_start_ticks"],
            "child_pid": owner["validator_pid"],
            "child_start_ticks": owner["validator_start_ticks"],
            "binding_sha256": sha256(contract.locations(root)[0]),
        },
    )
    result = planned | {
        "status": validation.STATUS,
        "owner": owner,
        "binding_sha256": sha256(contract.locations(root)[0]),
        "gpu_commands": entries,
        "results": measured,
        "accepted_cases": [item["name"] for item in measured if item["accepted"]],
        "rejected_cases": [item["name"] for item in measured if not item["accepted"]],
        "all_cases_accepted": all(item["accepted"] for item in measured),
    }
    record(contract.locations(root)[4], result)
    return root, path, result
