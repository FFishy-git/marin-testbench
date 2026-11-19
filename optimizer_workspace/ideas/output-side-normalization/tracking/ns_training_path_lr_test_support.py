# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist compact actual-LR CPU trajectories while retaining the pinned training files."""

import shutil
from pathlib import Path

import ns_training_path_lr_contract as contract
import ns_training_path_lr_validation as validation
import torch
from ns_lr_full_model_test_support import saved_trial as saved_trial
from ns_training_path_test_support import record as record
from selected_lr_release_owner import candidate_files, read, sha256

PRESERVED = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/"
    "candidates/ns-training-path-owner-closure-corrected-20261119T0736"
)


def candidate(tmp_path: Path, *, full: bool = False) -> tuple[Path, Path, dict]:
    """Scale checkpoint model dimensions only; original training files remain byte-identical."""
    root, path = tmp_path / "root", tmp_path / "candidate"
    archived = Path(__file__).resolve().parents[1]
    original = archived if (archived / contract.specs()[0]["config_name"]).is_file() else PRESERVED
    if full:
        for directory in ("experiment", "artifacts/pinned-sources-materialized"):
            shutil.copytree(
                original / directory, path / directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
    else:
        for name in contract.TRAINING_FILES:
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / name, target)
        record(path / "experiment/upstream_manifest.json", {"revision": "CPU-protocol-fixture", "files": {}})
    (path / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    names = {
        *(spec["config_name"] for spec in contract.specs()),
        *(f"experiment/configs/{family}_full_lr_lower_0p5m_seed0.json" for family in contract.study.contract.FAMILIES),
    }
    for name in names:
        config = read(original / name)
        config["model"].update(layers=8, hidden=4, mlp=8, vocab_size=16)
        record(path / name, config)
    record(root / "data/experiment_receipt.json", {"data": "training-path actual-LR CPU fixture"})
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


def complete_receipt(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Create all 162 persisted CPU command records, including an independent rank-7 failure."""
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
        saved_trial(root, path, frozen, item)
        if index == 0:
            checkpoint = Path(item["resumed"]) / "checkpoints/step_000024/rank_7.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state["fast"]["momentum"][(4, 4)]["m"][0, 0, 0] += 0.1
            torch.save(state, checkpoint)
        decision = validation.trial(item, path, frozen)
        for offset, (mode, command) in enumerate(zip(contract.MODES, item["commands"], strict=True)):
            log = work / f"{item['name']}_{mode}.log"
            log.write_text("CPU external command receipt fixture; no training executed\n")
            pid = 1000 + index * 3 + offset
            entry = {
                "command": command,
                "marker": f"training-path-actual-lr-command-{pid}",
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
