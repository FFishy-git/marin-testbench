# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist original terminal/command evidence at the accepted numerical-reader boundary."""

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ns_training_path_lr_predecessor as predecessor
from fused_attribution_validation_test_support import keeper, record
from ns_training_path_predecessor_test_support import generation
from selected_lr_release_owner import read, sha256


def completed(root: Path, monkeypatch) -> tuple[Path, dict]:
    """Use a real isolated report process and files; no numerical GPU acceptance is simulated."""
    record(root / "data/experiment_receipt.json", {"data": "CPU external evidence fixture"})
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "train.py").write_text("# preserved external numerical source\n")
    binding, helpers, state_path, work, result_path = predecessor.contract.locations(root)
    helpers.mkdir(parents=True)
    controller = helpers / "ns_training_path_gpu_controller.py"
    shutil.copy2(predecessor.controller.__file__, controller)
    helper_hashes = {controller.name: sha256(controller)}
    reporter = root / "report-reader/ns_training_path_reporting.py"
    reporter.parent.mkdir()
    reporter.write_text(
        "import json\n" "def evidence(root):\n" " return json.loads((root/'accepted-report.json').read_text())\n"
    )
    monkeypatch.setattr(predecessor.reporting, "__file__", str(reporter))
    monkeypatch.setattr(predecessor, "REPORTER_SHA256", sha256(reporter))
    archive = root / "run-preparation/issued-source.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"preserved original archived fixture")
    for relative in (predecessor.reporting.OWNER_CPU, predecessor.reporting.FOLLOWUP_CPU):
        record(root / relative, {"status": "accepted external CPU evidence fixture"})
    monkeypatch.setattr(predecessor.reporting, "OWNER_CPU_SHA256", sha256(root / predecessor.reporting.OWNER_CPU))
    monkeypatch.setattr(predecessor.reporting, "FOLLOWUP_CPU_SHA256", sha256(root / predecessor.reporting.FOLLOWUP_CPU))
    restored = keeper()
    controller_command = [predecessor.contract.PYTHON, "-c", "import signal; signal.pause()"]
    validator_command = [predecessor.contract.PYTHON, "-c", "import signal; signal.pause()", "validator"]
    frozen = {
        "status": predecessor.contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": helper_hashes,
        "files_sha256": {"train.py": sha256(candidate / "train.py")},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "controller_command": controller_command,
        "validator_command": validator_command,
        "environment": predecessor.contract.ENVIRONMENT,
        "controller_environment": predecessor.contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""},
        "authenticated_files": {},
    }
    record(binding, frozen)
    binding_digest = sha256(binding)
    monkeypatch.setattr(predecessor.reporting, "BINDING_SHA256", binding_digest)
    intent = {
        "status": "acknowledged",
        "binding_sha256": binding_digest,
        "pid": 701001,
        "start_ticks": "701101",
        "command": controller_command,
        "generation": generation(701001, 1, controller_command),
        "marker": "original-controller-fixture",
    }
    record(helpers.parent / "controller.json", intent)
    monkeypatch.setattr(predecessor.reporting, "INTENT_SHA256", sha256(helpers.parent / "controller.json"))
    entries, cases = [], []
    for index in range(18):
        name = f"original-case-{index}"
        directories = {mode: work / f"{name}_{mode}" for mode in ("continuous", "resumed")}
        trajectories = {}
        for mode in predecessor.contract.MODES:
            directory = directories["continuous" if mode == "continuous" else "resumed"]
            record(directory / "config.json", {"fixture": name})
            manifest = predecessor.INTERRUPTED_MANIFEST if mode == "interrupted" else "run.json"
            record(directory / manifest, {"status": "retained original external trajectory"})
            history = directory / f"metrics_{mode}.jsonl"
            history.write_text('{"fixture":"retained scalar series"}\n')
            checkpoints = {}
            for step in (22,) if mode == "interrupted" else (22, 23, 24):
                folder = directory / f"checkpoints/step_{step:06d}"
                record(folder / "complete.json", {"step": step, "world": 8})
                ranks = {}
                for rank in range(8):
                    path = folder / f"rank_{rank}.pt"
                    path.write_bytes(f"original retained rank {rank}; not loaded by handoff".encode())
                    ranks[path.name] = sha256(path)
                checkpoints[str(step)] = {
                    "receipt_sha256": sha256(folder / "complete.json"),
                    "rank_bytes": [(folder / name).stat().st_size for name in ranks],
                    "rank_sha256": ranks,
                }
            trajectories[mode] = {
                "manifest_sha256": sha256(directory / manifest),
                "history_sha256": {history.name: sha256(history)},
                "checkpoints": checkpoints,
            }
            command = [predecessor.contract.PYTHON, "-m", "experiment.train", name, mode]
            log = work / f"{name}_{mode}.log"
            log.write_text("retained original external command evidence\n")
            pid = 710000 + index * 3 + predecessor.contract.MODES.index(mode)
            entry = {
                "trial": name,
                "mode": mode,
                "generation": generation(pid, 701002, command),
                "marker": f"original-command-{pid}",
                "log": str(log),
                "log_sha256": sha256(log),
                "exit_code": 0,
                "cleanup_exit_code": 0,
            }
            record(log.with_suffix(".execution.json"), entry)
            entries.append(entry)
        cases.append(
            {
                "name": name,
                "accepted": index != 0,
                "continuous": str(directories["continuous"]),
                "resumed": str(directories["resumed"]),
                "trajectories": trajectories,
            }
        )
    summary = {
        "accepted_cases": [case["name"] for case in cases if case["accepted"]],
        "rejected_cases": [cases[0]["name"]],
        "all_cases_accepted": False,
    }
    result = {"gpu_commands": entries, "results": cases, **summary}
    record(result_path, result)
    state = {
        "status": "validation_complete",
        "binding_sha256": binding_digest,
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "command": validator_command,
        "child_pid": 701002,
        "child_start_ticks": "701102",
        "child_generation": generation(701002, 701001, validator_command),
        "child_marker": "original-validator-fixture",
        "restored_keeper": restored,
        **summary,
    }
    record(state_path, state)
    for name in ("controller.log", "validator.log"):
        (helpers.parent / name).write_text("retained original controller evidence\n")
    waiter_work = root / predecessor.reporting.FOLLOWUP_WORK
    waiter_work.mkdir(parents=True)
    waiter_command = predecessor.followup.command(root, root / predecessor.reporting.FOLLOWUP_CPU)
    waiter_intent = {
        "status": "acknowledged",
        "binding_sha256": binding_digest,
        "cpu_receipt_sha256": predecessor.reporting.FOLLOWUP_CPU_SHA256,
        "runner_sha256": predecessor.FOLLOWUP_SHA256,
        "command": waiter_command,
        "pid": 701003,
        "start_ticks": "701103",
        "generation": generation(701003, 1, waiter_command),
        "marker": "original-waiter-fixture",
    }
    record(waiter_work / "intent.json", waiter_intent)
    monkeypatch.setattr(predecessor.reporting, "FOLLOWUP_INTENT_SHA256", sha256(waiter_work / "intent.json"))
    record(
        waiter_work / "state.json",
        {
            "status": "terminal_reconstructed",
            "pid": waiter_intent["pid"],
            "start_ticks": waiter_intent["start_ticks"],
            "binding_sha256": binding_digest,
            "terminal_generation": generation(701004, 701003, ["python", "original-terminal"]),
            "terminal_marker": "original-terminal-fixture",
        },
    )
    for name in ("followup.log", "terminal.log"):
        (waiter_work / name).write_text("retained original waiter evidence\n")
    terminal_path, attempt, outcome, _ = predecessor.controller.terminal_paths(root, intent)
    record(attempt, {"time": (datetime.now(UTC) + timedelta(seconds=1)).isoformat()})
    evidence_paths = (
        binding,
        state_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        helpers.parent / "validator.log",
        result_path,
    )
    terminal = {
        **summary,
        "keeper": restored,
        "reconstruction_intent_sha256": sha256(attempt),
        "evidence_sha256": {str(path.relative_to(root)): sha256(path) for path in evidence_paths},
    }
    record(terminal_path, terminal)
    record(outcome, {"terminal_sha256": sha256(terminal_path)})
    report = {
        "execution_owner": {"accepted": True},
        "gpu_progress": {
            "status": "validation_complete",
            "result_status": predecessor.reporting.RESULT_STATUS,
            "completed_commands": 54,
            "decisions": {"cases": cases},
            "terminal": {
                "status": "complete",
                "saved_exact_restoration": True,
                "receipt": str(terminal_path),
                "receipt_sha256": sha256(terminal_path),
            },
        },
        "terminal_followup": {"status": "terminal_reconstructed"},
        "numerical_gpu_acceptance": {"status": "saved_independent_terminal_receipt"},
    }
    record(root / "accepted-report.json", report)
    return root, restored


def rebind_terminal(root: Path) -> None:
    """Refresh fixture hashes so negative tests cannot pass on a stale outer digest alone."""
    _, helpers, _, _, result = predecessor.contract.locations(root)
    intent = read(helpers.parent / "controller.json")
    terminal_path, _, outcome, _ = predecessor.controller.terminal_paths(root, intent)
    terminal = read(terminal_path)
    terminal["evidence_sha256"][str(result.relative_to(root))] = sha256(result)
    record(terminal_path, terminal)
    record(outcome, {"terminal_sha256": sha256(terminal_path)})
    report = read(root / "accepted-report.json")
    report["gpu_progress"]["terminal"]["receipt_sha256"] = sha256(terminal_path)
    record(root / "accepted-report.json", report)
