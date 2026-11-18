# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject partial captured histories and misbound validation processes without invoking CUDA."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if Path(__file__).parent.name == "tests":
    from experiment import execution_owner as owner
    from experiment.upstream_validation import ARMS, require_upstream_capture_history, validation_commands
else:
    from upstream_validation import ARMS, require_upstream_capture_history, validation_commands

    package_name = "_upstream_validation_cpu"
    if package_name not in sys.modules:
        path = Path(
            "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
            "/candidates/upstream-full-model-lifecycle-20261117/experiment"
        )
        spec = importlib.util.spec_from_file_location(
            package_name, path / "__init__.py", submodule_search_locations=[str(path)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    name = f"{package_name}.upstream_validation"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("upstream_validation.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    name = f"{package_name}.execution_owner"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("upstream_execution_owner.py"))
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)


def captured_rows(arm):
    return [
        {
            "step": step,
            "diagonal_capture_parity_passed": True,
            "upstream_product_parity_passed": True,
            "upstream_scheduled_scalars_passed": True,
            "upstream_pinned_input_exact": True,
            "upstream_output_equation_passed": True,
            "upstream_null_exact": arm == ARMS[3],
            "upstream_owned_stacks_checked": 3,
            "upstream_output_equation_max_error": 1e-7,
        }
        for step in range(1, 25)
    ]


def write_history(directory, rows):
    (directory / "metrics_from_000000.jsonl").write_text("\n".join(json.dumps(row) for row in rows[:22]))
    (directory / "metrics_from_000022.jsonl").write_text("\n".join(json.dumps(row) for row in rows[22:]))


@pytest.mark.parametrize("arm", ARMS[2:])
@pytest.mark.parametrize("corruption", ["flag", "null", "shape", "error", "missing", "duplicate"])
def test_pinned_capture_history_rejects_incomplete_or_corrupted_resume_tail(tmp_path, arm, corruption):
    rows = captured_rows(arm)
    write_history(tmp_path, rows)
    result = require_upstream_capture_history(tmp_path, arm)
    assert result == {
        "updates": 24,
        "owned_shapes_per_update": 3,
        "null_exact": arm == ARMS[3],
        "max_equation_error": 1e-7,
    }
    if corruption == "flag":
        rows[-1]["upstream_pinned_input_exact"] = False
    elif corruption == "null":
        rows[-1]["upstream_null_exact"] = not rows[-1]["upstream_null_exact"]
    elif corruption == "shape":
        rows[-1]["upstream_owned_stacks_checked"] = 2
    elif corruption == "error":
        rows[-1]["upstream_output_equation_max_error"] = float("nan")
    elif corruption == "missing":
        rows.pop()
    else:
        rows[-1]["step"] = 23
    write_history(tmp_path, rows)
    with pytest.raises(ValueError):
        require_upstream_capture_history(tmp_path, arm)


@pytest.fixture
def owned_processes(tmp_path, monkeypatch):
    """Fake the /proc, hostname and environment I/O boundaries; use real source/data hashes."""
    root = tmp_path / "root"
    task = tmp_path / "candidate"
    task.mkdir()
    helpers = root / "controllers/upstream-full-model-corrected-20261117/tracking"
    helpers.mkdir(parents=True)
    helper = helpers / "run_upstream_full_model.py"
    helper.write_text("bound controller fixture\n")
    source = task / "bound.py"
    source.write_text("bound candidate fixture\n")
    receipt = root / "data/experiment_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    archive = root / "source.tar"
    archive.write_bytes(b"archive fixture")
    preparation = root / "run-preparation"
    preparation.mkdir()
    marker = "upstream-validation-test"
    monkeypatch.setenv(owner.MARKER, marker)
    monkeypatch.setattr(owner, "TASK", task)
    monkeypatch.setattr(owner, "OUTPUT_ROOT", root)
    monkeypatch.setattr(owner, "ENVIRONMENT", {})
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: owner.NODE)
    python = owner.PYTHON
    command = [
        python,
        "-m",
        "experiment.validate",
        "--gpu",
        "--data",
        str(root / "data"),
        "--data-receipt",
        str(receipt),
        "--work",
        str(root / "validation/upstream_full_model_corrected"),
        "--receipt",
        str(preparation / "validation-upstream-full-model-corrected-gpu.json"),
    ]
    binding = {
        "status": "frozen_pending_upper_lr_and_synthetic_probe",
        "root": str(root),
        "candidate": str(task),
        "node": owner.NODE,
        "helpers": str(helpers),
        "files_sha256": {"bound.py": owner.sha256(source)},
        "helpers_sha256": {helper.name: owner.sha256(helper)},
        "archive": str(archive),
        "archive_sha256": owner.sha256(archive),
        "data_receipt_sha256": owner.sha256(receipt),
        "environment": {},
    }
    binding_path = preparation / "upstream-full-model-corrected-binding.json"
    binding_path.write_text(json.dumps(binding))
    current = os.getpid()
    state = {
        "status": "validating",
        "controller_pid": 20001,
        "controller_start_ticks": "71",
        "child_pid": 20002,
        "child_start_ticks": "72",
        "child_marker": marker,
        "node": owner.NODE,
        "candidate": str(task),
        "binding_sha256": owner.sha256(binding_path),
        "command": command,
    }
    state_path = preparation / "upstream-full-model-corrected-state.json"
    state_path.write_text(json.dumps(state))

    def fields(parent, ticks):
        values = ["0"] * 20
        values[0], values[1], values[19] = "S", str(parent), ticks
        return values

    processes = {
        20001: (fields(1, "71"), [python, str(helper), "--root", str(root), "--phase", "validate"]),
        20002: (fields(20001, "72"), command),
        20003: (fields(20002, "73"), []),
        current: (fields(20003, "74"), [python, "-m", "experiment.train"]),
    }
    monkeypatch.setattr(owner, "observed", lambda pid: processes[pid])
    return root, task, state_path, state, processes


@pytest.mark.parametrize("corruption", ["controller", "child", "marker", "source"])
def test_validation_ownership_rejects_reused_generations_or_changed_source(owned_processes, corruption, monkeypatch):
    root, task, _, state, processes = owned_processes
    assert owner.require_controller()["controller_pid"] == state["controller_pid"]
    if corruption == "controller":
        processes[state["controller_pid"]][0][19] = "wrong generation"
    elif corruption == "child":
        processes[state["child_pid"]][0][19] = "wrong generation"
    elif corruption == "marker":
        monkeypatch.setenv(owner.MARKER, "unowned")
    else:
        (task / "bound.py").write_text("different source\n")
    with pytest.raises(RuntimeError):
        owner.require_controller()
    assert root.joinpath("data/experiment_receipt.json").read_text() == "{}"


def test_corrected_validator_refuses_original_release_paths(owned_processes):
    root, _, state_path, _, _ = owned_processes
    preparation = root / "run-preparation"
    binding = preparation / "upstream-full-model-corrected-binding.json"
    assert owner.require_controller()["status"] == "validating"
    binding.rename(preparation / "upstream-full-model-binding.json")
    state_path.rename(preparation / "upstream-full-model-state.json")
    with pytest.raises(FileNotFoundError):
        owner.require_controller()


@pytest.mark.parametrize("arm", ARMS)
def test_training_ownership_requires_canonical_eight_rank_trajectory(owned_processes, arm, monkeypatch):
    root, task, _, _, processes = owned_processes
    config_path = task / f"experiment/configs/{arm}_0p5m_seed0.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"experiment": {"arm": arm}}
    config_path.write_text(json.dumps(config))
    commands = validation_commands(owner.PYTHON, task, root, root / "validation/upstream_full_model_corrected")
    command = commands[1 + 3 * ARMS.index(arm)]
    processes[20003] = (processes[20003][0], command)
    monkeypatch.setattr(owner.sys, "argv", [str(task / "experiment/train.py"), *command[7:]])
    args = SimpleNamespace(launch_release=None, config=config_path)
    owner.require_training_owner(args, config)
    assert command[4:6] == ["--nproc_per_node=8", "-m"]
    processes[20003] = (processes[20003][0], [*command[:4], "--nproc_per_node=2", *command[5:]])
    with pytest.raises(RuntimeError, match="eight-rank launcher"):
        owner.require_training_owner(args, config)
    args.launch_release = root / "run-preparation/wave8-launch-binding.json"
    with pytest.raises(RuntimeError, match="production release has not been issued"):
        owner.require_training_owner(args, config)


def test_synthetic_ownership_rejects_wrong_rank_count_or_receipt(owned_processes):
    root, task, _, _, processes = owned_processes
    command = validation_commands(owner.PYTHON, task, root, root / "validation/upstream_full_model_corrected")[0]
    processes[20003] = (processes[20003][0], command)
    assert owner.require_synthetic_owner()["status"] == "validating"
    processes[20003] = (processes[20003][0], [*command[:-1], str(root / "unbound.json")])
    with pytest.raises(RuntimeError, match="exact two-rank"):
        owner.require_synthetic_owner()
