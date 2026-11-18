# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject altered validation receipts and preserve all-rank checkpoint requirements."""

import hashlib
import json
from pathlib import Path

import pytest
import run_upstream_full_model as controller
import torch
from benchmark_diagonal import source_definitions
from experiment import records
from prepare_upstream_validation import require_cpu_source
from wave7_release import candidate_inventory


def test_upstream_controller_wrong_node_refuses_before_allocation_transfer(tmp_path):
    definitions = source_definitions(
        Path(controller.__file__),
        ("validate",),
        {"binding": lambda root: {}, "local_node": lambda: "28.216.17.9", "NODE": controller.NODE, "Path": Path},
    )
    with pytest.raises(RuntimeError, match=r"original \.139"):
        definitions["validate"](tmp_path)
    assert not (tmp_path / "run-preparation/upstream-full-model-corrected-state.json").exists()


@pytest.mark.parametrize("corruption", ["duplicate_command", "changed_log", "missing_scalar_gate"])
def test_upstream_receipt_rejects_success_flags_with_changed_commands_or_logs(tmp_path, corruption):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    source = tmp_path / "candidate"
    configurations = {}
    for arm in controller.ARMS:
        path = source / f"experiment/configs/{arm}_0p5m_seed0.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"experiment": {"arm": arm}}))
        configurations[arm] = controller.sha256(path)
    owner = {"controller_pid": 1234, "controller_start_ticks": "5678"}
    (preparation / "upstream-full-model-corrected-state.json").write_text(json.dumps(owner))
    work = tmp_path / "validation/upstream_full_model_corrected"
    work.mkdir(parents=True)
    commands = controller.validation_commands(controller.PYTHON, source, tmp_path, work)
    executions = []
    for index, command in enumerate(commands):
        log = work / f"{index}.log"
        log.write_text("passed fixture\n")
        executions.append({"command": command, "exit_code": 0, "log": str(log), "log_sha256": controller.sha256(log)})
    receipt = {
        "owner": owner,
        "source_sha256": {},
        "data_receipt_sha256": "data",
        "environment": controller.ENVIRONMENT,
        "config_sha256": configurations,
        "checks": dict.fromkeys(controller.GATES, True),
        "evidence": {"real_data_smokes": {arm: {} for arm in controller.ARMS}, "gpu_commands": executions},
    }
    if corruption == "duplicate_command":
        executions[-1]["command"] = executions[-2]["command"]
    elif corruption == "changed_log":
        Path(executions[-1]["log"]).write_text("different outcome\n")
    else:
        del receipt["checks"]["gpu_upstream_scalar_step"]
    (preparation / "validation-upstream-full-model-corrected-gpu.json").write_text(json.dumps(receipt))
    frozen = {"candidate": str(source), "source_sha256": {}, "data_receipt_sha256": "data", "commands": commands}
    with pytest.raises(ValueError, match="coverage or log identity"):
        controller.require_validation(tmp_path, frozen)
    assert not (preparation / "upstream-full-model-corrected-binding.json").exists()


def test_upstream_frozen_checkpoint_comparison_covers_rank_seven(tmp_path):
    source = Path(records.__file__)
    comparisons = source_definitions(source, ("compare_tree", "compare_continuation"), {"torch": torch})
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    for rank in range(8):
        saved = {
            "step": 24,
            "next_global_sequence": 3072,
            "identity": {"source": "bound"},
            "model": {"weight": torch.tensor([float(rank)])},
        }
        torch.save(saved, full / f"rank_{rank}.pt")
        torch.save(saved, resumed / f"rank_{rank}.pt")
    assert comparisons["compare_continuation"](full, resumed, 8) == {
        "tensor_comparisons": 8,
        "max_absolute_error": 0.0,
    }
    changed = torch.load(resumed / "rank_7.pt", weights_only=False)
    changed["model"]["weight"].add_(0.01)
    torch.save(changed, resumed / "rank_7.pt")
    with pytest.raises(AssertionError):
        comparisons["compare_continuation"](full, resumed, 8)


@pytest.mark.parametrize("corruption", ["rank", "world", "step", "config", "source"])
def test_checkpoint_identity_rejects_shared_wrong_metadata(tmp_path, corruption):
    source = tmp_path / "candidate"
    (source / "experiment").mkdir(parents=True)
    module = source / "experiment/train.py"
    module.write_text("bound source fixture\n")
    manifest = source / "experiment/upstream_manifest.json"
    manifest.write_text(json.dumps({"revision": "fixture", "files": {}}))
    frozen = {
        "data_receipt_sha256": "data",
        "files_sha256": {
            "experiment/train.py": controller.sha256(module),
            "experiment/upstream_manifest.json": controller.sha256(manifest),
        },
    }
    config = {"training": {"global_batch_sequences": 256}}
    identity = {
        "schema": 1,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "data_receipt_sha256": "data",
        "source_sha256": {"experiment/train.py": controller.sha256(module)},
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "complete.json").write_text(json.dumps({"step": 24, "world": 8}))
    for rank in range(8):
        torch.save(
            {"rank": rank, "world": 8, "step": 24, "config": config, "next_global_sequence": 6144, "identity": identity},
            checkpoint / f"rank_{rank}.pt",
        )
    controller.require_checkpoint_identity(checkpoint, config, source, frozen, 24)
    saved = torch.load(checkpoint / "rank_7.pt", weights_only=False)
    if corruption == "source":
        saved["identity"]["source_sha256"]["experiment/train.py"] = "changed"
    elif corruption == "config":
        saved["config"]["training"]["global_batch_sequences"] = 128
    else:
        saved[corruption] = 0
    torch.save(saved, checkpoint / "rank_7.pt")
    with pytest.raises(ValueError, match="rank/config/source identity"):
        controller.require_checkpoint_identity(checkpoint, config, source, frozen, 24)


def test_validator_predecessor_cpu_refuses_duplicate_candidate_commands(tmp_path):
    candidate = tmp_path / "candidate"
    (candidate / "experiment").mkdir(parents=True)
    (candidate / "experiment/train.py").write_text("source fixture\n")
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    initial = tmp_path / "initial.json"
    initial.write_text("{}")
    archive = tmp_path / "source.tar"
    archive.write_bytes(b"archive")
    log = tmp_path / "candidate.log"
    log.write_text("passed fixture\n")
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    helper = helpers / "fixture.py"
    helper.write_text("helper fixture\n")
    record = {
        "status": "full_model_observer_cpu_verified_gpu_execution_denied",
        "cuda_visible_devices": "",
        "candidate": str(candidate),
        "files_sha256": candidate_inventory(candidate),
        "archive": str(archive),
        "archive_sha256": controller.sha256(archive),
        "initial_preparation": str(initial),
        "initial_preparation_sha256": controller.sha256(initial),
        "helpers": str(helpers),
        "helpers_sha256": {"fixture.py": controller.sha256(helper)},
        "commands": (
            [
                {
                    "cwd": str(candidate),
                    "command": ["candidate"],
                    "exit_code": 0,
                    "log": str(log),
                    "log_sha256": controller.sha256(log),
                }
            ]
            * 2
        ),
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="predecessor source"):
        require_cpu_source(receipt)


def test_validator_spawn_failure_retains_marker_command_intent(tmp_path, monkeypatch):
    def failed_spawn(*args, **kwargs):
        raise OSError("spawn unavailable")

    monkeypatch.setattr(controller.subprocess, "Popen", failed_spawn)
    state_path = tmp_path / "state.json"
    with pytest.raises(OSError, match="spawn unavailable"):
        controller.bounded_validator(["validator"], tmp_path, tmp_path / "validator.log", {}, state_path)
    record = json.loads(state_path.read_text())
    assert record["spawn_status"] == "intent"
    assert record["command"] == ["validator"]
    assert len(record["child_marker"]) == 32
    assert "child_pid" not in record


def test_corrected_validator_keeps_original_execution_evidence_separate(tmp_path):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    originals = (
        preparation / "upstream-full-model-binding.json",
        preparation / "upstream-full-model-state.json",
        preparation / "validation-upstream-full-model-gpu.json",
    )
    for path in originals:
        path.write_text("preserved original evidence\n")
    binding, helpers, state = controller.locations(tmp_path)
    assert all(path.read_text() == "preserved original evidence\n" for path in originals)
    assert not binding.exists() and not state.exists()
    assert helpers != tmp_path / "controllers/upstream-full-model-20261117/tracking"
