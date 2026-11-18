# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Wave-10 source preservation and failed-launch evidence without GPUs."""

import json
import os
import shlex
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import analyze_owned_output_run as analysis
import owned_output_production as production
import pytest
from launch_wave5 import read, sha256
from wave7_release import candidate_inventory


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def predecessor(tmp_path, monkeypatch):
    root, candidate = tmp_path / "root", tmp_path / "validated"
    (root / "run-preparation").mkdir(parents=True)
    experiment = candidate / "experiment"
    (experiment / "tests").mkdir(parents=True)
    (experiment / "execution_owner.py").write_text(
        "from .owned_output_validation import VALIDATION_ENVIRONMENT, validation_commands\n"
        "from .reference import TASK\n\n"
        "def require_training_owner(args, config):\n"
        "    if args.launch_release is not None:\n"
        '        raise RuntimeError("owned output full-budget production release has not been issued")\n'
    )
    (experiment / "train.py").write_text("def train():\n    return 42\n")
    (experiment / "production_owner.py").write_text("def old_owner():\n    return False\n")
    record(
        experiment / "cluster.json",
        {"wave_name": "validated", "environment": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}},
    )
    for arm in production.ARMS:
        record(experiment / f"configs/{arm}_0p5m_seed0.json", {"arm": arm, "steps": 5120, "lr": 0.02262741696})
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    frozen = {
        "candidate": str(candidate),
        "files_sha256": candidate_inventory(candidate),
        "helpers": str(Path(production.__file__).parent),
        "helpers_sha256": {},
    }
    record(root / "run-preparation/owned-output-trace-binding.json", frozen)
    monkeypatch.setattr(production, "original", lambda root: frozen)
    return root, candidate, frozen


def test_preparation_retains_running_validator_and_all_numerical_bytes(predecessor, tmp_path):
    root, previous, frozen = predecessor
    before = candidate_inventory(previous)
    candidate = tmp_path / "production"
    result = production.prepare(root, candidate)
    assert set(result["changed_files"]) == production.CHANGES
    assert candidate_inventory(previous) == before == frozen["files_sha256"]
    assert (candidate / "experiment/train.py").read_bytes() == (previous / "experiment/train.py").read_bytes()
    assert (candidate / "experiment/production_owner.py").read_bytes() == (
        previous / "experiment/production_owner.py"
    ).read_bytes()
    for arm in production.ARMS:
        relative = f"experiment/configs/{arm}_0p5m_seed0.json"
        assert (candidate / relative).read_bytes() == (previous / relative).read_bytes()
    assert production.require_preparation(root, candidate)["files_sha256"] == result["files_sha256"]
    with pytest.raises(FileExistsError):
        production.prepare(root, candidate)
    (candidate / "experiment/train.py").write_text("def train():\n    return 99\n")
    with pytest.raises(ValueError, match="scope"):
        production.require_preparation(root, candidate)
    assert not production.paths(root)[0].exists()


@pytest.mark.parametrize("failure", ["exit", "timeout", "source_drift"])
def test_cpu_failure_is_archived_and_cannot_authorize_release(predecessor, tmp_path, monkeypatch, failure):
    root, _, _ = predecessor
    candidate = tmp_path / "production"
    production.prepare(root, candidate)

    def failed(command, **kwargs):
        kwargs["stdout"].write("CPU command observed\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 600)
        if failure == "source_drift":
            (candidate / "experiment/train.py").write_text("def train():\n    return 99\n")
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(production.subprocess, "run", failed)
    out = tmp_path / "cpu"
    with pytest.raises(RuntimeError, match="preserved"):
        production.cpu(root, candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["status"] == "production_cpu_failed"
    assert receipt["exit_code"] == {"exit": 1, "timeout": 124, "source_drift": 0}[failure]
    assert (out / "source.tar").is_file()
    assert receipt["log_sha256"] == sha256(out / "candidate.log")
    with pytest.raises(ValueError):
        production.require_cpu(root, out / "receipt.json")
    assert not production.paths(root)[0].exists()


def test_incomplete_original_validation_prevents_production_release(predecessor, tmp_path, monkeypatch):
    root, _, _ = predecessor
    candidate = tmp_path / "production"
    production.prepare(root, candidate)

    def passed(command, **kwargs):
        kwargs["stdout"].write("CPU command passed\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(production.subprocess, "run", passed)
    production.cpu(root, candidate, tmp_path / "cpu")

    def incomplete(root):
        raise RuntimeError("Original full-trace validator has not passed")

    monkeypatch.setattr(production, "require_completed_validation", incomplete)
    with pytest.raises(RuntimeError, match="has not passed"):
        production.freeze(root, tmp_path / "cpu/receipt.json")
    assert not production.paths(root)[0].exists()
    assert not production.paths(root)[1].exists()
    assert not (root / "runs/wave10").exists()


@pytest.mark.parametrize("fault", (None, "intent", "state", "keeper", "marker_generation", "final_keeper"))
def test_validation_reconstruction_rejects_changed_acknowledgment_or_allocation(tmp_path, monkeypatch, fault):
    root = tmp_path / "root"
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    preparation = root / "run-preparation"
    preparation.mkdir(parents=True)
    binding_path = preparation / "owned-output-trace-binding.json"
    state_path = preparation / "owned-output-trace-state.json"
    receipt_path = preparation / "validation-owned-output-trace-gpu.json"
    intent_path = helpers.parent / "controller.json"
    record(binding_path, {"node": production.NODES[-1], "candidate": "validated", "helpers": str(helpers)})
    record(receipt_path, {"evidence": {"gpu_commands": [0] * 14}})
    command = [sys.executable, str(helpers / "run_owned_output_trace.py"), "--root", str(root), "--phase", "validate"]
    record(
        state_path,
        {
            "status": "passed",
            "controller_pid": 123,
            "controller_start_ticks": "456",
            "child_pid": 234,
            "child_marker": "child",
            "binding_sha256": sha256(binding_path),
            "validation_sha256": sha256(receipt_path),
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "keeper_restore": 0,
        },
    )
    record(
        intent_path,
        {
            "status": "acknowledged",
            "pid": 123,
            "start_ticks": "456",
            "command": command,
            "generation": {"pid": 123, "start_ticks": "456", "command": command, "status": "S"},
            "marker": "controller",
            "binding_sha256": sha256(binding_path),
            "candidate": "validated",
            "cuda_visible_devices": production.MASK,
        },
    )
    (helpers / "fault.txt").write_text(fault or "")
    (helpers / "owned_output_handoff.py").write_text(
        "import hashlib,json\nfrom pathlib import Path\n"
        "BASE=Path(__file__).parent\nCHILD_MARKER='child'\n"
        "def read(path): return json.loads(Path(path).read_text())\n"
        "def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n"
        "def observed(pid): return None\n"
        "def same_generation(a,b): return a==b\n"
        "def marker_processes(name,value):\n"
        "    if name=='child': return {}\n"
        "    ticks='reused' if (BASE/'fault.txt').read_text()=='marker_generation' else 'original'\n"
        "    return {789:{'pid':789,'start_ticks':ticks,'status':'S'}}\n"
        "def keeper_observation():\n"
        "    ticks='reused' if (BASE/'changed').exists() else 'original'\n"
        "    return {'pid':789,'generation':{789:{'pid':789,'start_ticks':ticks,'status':'S'}}}\n"
    )
    (helpers / "wave7_release.py").write_text(
        "from contextlib import contextmanager\n"
        "@contextmanager\n"
        "def node_guard(root,node):\n"
        "    with (root/'locked').open('w'):\n"
        "        yield\n"
    )
    (helpers / "run_owned_output_trace.py").write_text(
        "import json\nfrom pathlib import Path\nimport owned_output_handoff as h\n"
        "CONTROLLER_MARKER='controller'\n"
        "def binding(root):\n"
        "    return h.read(root/'run-preparation/owned-output-trace-binding.json')|{'required_gates':[0]*26}\n"
        "def require_validation(root,frozen):\n"
        "    assert (root/'locked').exists()\n"
        "    fault=(Path(__file__).parent/'fault.txt').read_text()\n"
        "    if fault in ('intent','state'):\n"
        "        path=Path(__file__).parent.parent/'controller.json' if fault=='intent' else "
        "root/'run-preparation/owned-output-trace-state.json'\n"
        "        value=h.read(path); value['controller_start_ticks']='changed'; path.write_text(json.dumps(value))\n"
        "    elif fault=='keeper': (Path(__file__).parent/'changed').touch()\n"
        "    return h.read(root/'run-preparation/validation-owned-output-trace-gpu.json')\n"
    )
    monkeypatch.setattr(production, "original", lambda root: read(binding_path))
    real_run = subprocess.run

    def receiver(command, **kwargs):
        remote = shlex.split(command[-1])
        script = remote[remote.index("-c") + 1]
        return real_run(
            [sys.executable, "-c", script, *remote[remote.index("-c") + 2 :]],
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)},
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )

    monkeypatch.setattr(production.subprocess, "run", receiver)
    if fault in ("intent", "state", "keeper", "marker_generation"):
        with pytest.raises(subprocess.CalledProcessError):
            production.require_completed_validation(root)
        return
    evidence = production.require_completed_validation(root)
    assert evidence["commands"] == 14 and evidence["checks"] == 26
    if fault == "final_keeper":
        (helpers / "changed").touch()
        with pytest.raises(subprocess.CalledProcessError):
            production.require_completed_validation(root, accepted=evidence)
    else:
        assert production.require_completed_validation(root, accepted=evidence) == evidence


@pytest.mark.parametrize("fault", ("state_sha256", "receipt_sha256", "intent_sha256"))
def test_changed_accepted_validation_prevents_release_publication(predecessor, tmp_path, monkeypatch, fault):
    root, _, _ = predecessor
    candidate = tmp_path / "production"
    production.prepare(root, candidate)

    def passed(command, **kwargs):
        kwargs["stdout"].write("CPU command passed\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(production.subprocess, "run", passed)
    production.cpu(root, candidate, tmp_path / "cpu")
    paths = {
        "binding_sha256": root / "run-preparation/owned-output-trace-binding.json",
        "state_sha256": root / "run-preparation/owned-output-trace-state.json",
        "receipt_sha256": root / "run-preparation/validation-owned-output-trace-gpu.json",
        "intent_sha256": root / "controllers/owned-output-trace-validation-20261117/controller.json",
    }
    for key, path in paths.items():
        if key != "binding_sha256":
            record(path, {"status": "accepted"})
    accepted = {key: sha256(path) for key, path in paths.items()}

    def changed(root):
        record(paths[fault], {"status": "changed after acceptance"})
        return accepted

    monkeypatch.setattr(production, "require_completed_validation", changed)
    with pytest.raises(ValueError, match="Accepted original validation"):
        production.freeze(root, tmp_path / "cpu/receipt.json")
    assert not production.paths(root)[0].exists()
    assert not production.paths(root)[1].exists()
    assert not (root / "runs/wave10").exists()


def test_allocation_refusal_leaves_no_launch_intent(tmp_path, monkeypatch):
    (tmp_path / "run-preparation").mkdir()
    monkeypatch.setattr(production, "binding", lambda root: {})
    monkeypatch.setattr(production, "require_completed_validation", lambda root: {})

    def occupied(*args):
        raise RuntimeError("allocation occupied")

    monkeypatch.setattr(production, "remote", occupied)
    with pytest.raises(RuntimeError, match="occupied"):
        production.launch(tmp_path)
    assert not (tmp_path / "run-preparation/wave10-launch-state.json").exists()
    assert not (tmp_path / "runs/wave10").exists()


def test_worker_cleanup_failure_preserves_error_and_refuses_keeper_start(tmp_path, monkeypatch):
    (tmp_path / "run-preparation").mkdir()
    release_path, _ = production.paths(tmp_path)
    record(release_path, {"status": "bound"})
    arm = production.ARMS[0]
    release = {"helpers": str(tmp_path)}
    command = production.worker_command(tmp_path, release, arm)
    generation = {
        "pid": os.getpid(),
        "status": "S",
        "parent": os.getppid(),
        "group": os.getpgrp(),
        "session": os.getsid(0),
        "start_ticks": "123",
        "command": command,
    }
    generation["parent"] = 999
    path = production.worker_path(tmp_path, arm)
    record(
        path.with_suffix(".start.json"),
        {
            "generation": generation,
            "worker_marker": "owned",
            "release_binding_sha256": sha256(release_path),
        },
    )
    record(
        tmp_path / "run-preparation/wave10-launch-state.json",
        {
            "status": "launching",
            "binding_sha256": sha256(release_path),
            "worker_markers": {arm: "owned"},
        },
    )
    monkeypatch.setenv(production.WORKER_MARKER, "owned")
    monkeypatch.setattr(production, "binding", lambda root: release)
    monkeypatch.setattr(production, "observed", lambda pid: generation | {"parent": 1})
    monkeypatch.setattr(production, "node_guard", lambda *args: nullcontext([]))
    keeper = {"generation": {}}
    monkeypatch.setattr(production, "allocation", lambda *args: keeper)
    monkeypatch.setattr(production, "keeper_observation", lambda: keeper)
    monkeypatch.setattr(production, "release_keeper", lambda state, path: state.update(keeper_release_attempted=True))
    restored = []
    monkeypatch.setattr(production, "node_action", lambda root, action: restored.append(action))

    def failed(root, release, state, path):
        state["cleanup_error"] = "unverified live child"
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(production, "run_child", failed)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        production.run_arm(tmp_path, arm)
    state = read(path)
    assert state["status"] == "needs_attention"
    assert "refused" in state["keeper_restore"]
    assert restored == []


def test_analysis_retains_five_complete_windows_and_matched_null(tmp_path, monkeypatch):
    root = tmp_path
    source, previous = root / "source", root / "previous"
    record(root / "run-preparation/wave9-launch-binding.json", {"candidate": str(previous)})
    record(
        root / "run-preparation/wave10-launch-binding.json",
        {
            "candidate": str(source),
            "predecessor_release_sha256": sha256(root / "run-preparation/wave9-launch-binding.json"),
        },
    )
    for index, arm in enumerate(production.ARMS):
        config = {
            "experiment": {"arm": arm},
            "model": {"sequence_length": 4096},
            "data": {"mixture_weights": {"dclm": 1.0}},
            "training": {
                "steps": 5120,
                "tokens": 2684354560,
                "batch_tokens": 524288,
                "seed": 0,
                "heldout_windows": 64,
            },
            "runtime": {"eval_max_windows": 0},
        }
        for candidate in (source, previous):
            record(candidate / f"experiment/configs/{arm}_0p5m_seed0.json", config)
        for wave in ("wave9", "wave10"):
            directory = root / "runs" / wave / arm
            record(
                directory / "run.json",
                {
                    "status": "complete",
                    "completed_steps": 5120,
                    "config": config,
                    "data_receipt_sha256": "paired-data",
                    "execution_check": False,
                    "elapsed_seconds": 10240.0,
                },
            )
            rows = []
            for step in range(1, 5121):
                row = {
                    "step": step,
                    "tokens": step * 524288,
                    "train_loss": 5.0 - index * 0.01,
                    "step_seconds": 2.0,
                    "elapsed_seconds": step * 2.0,
                    "end_to_end_seconds": step * 2.0,
                }
                if step % 250 == 1 or step == 5120:
                    row.update(c4_loss=3.5 - index * 0.01, selection_loss=3.0 - index * 0.01)
                rows.append(row)
            (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(analysis, "verify_histories", lambda root: {})
    result = analysis.analyze(root)
    candidate = result["analyses"][0]["comparisons"]["wave10/outer_leftdiag_upstream_fused"]
    assert [window["end_tokens"] for window in candidate["training_windows"]] == [
        index * 536870912 for index in range(1, 6)
    ]
    assert [window["delta"] for window in candidate["training_windows"]] == pytest.approx([-0.02] * 5)
    assert candidate["final_comparison_eligible"] is True
    null = result["analyses"][2]
    assert null["baseline"] == "wave10/null_leftdiag_upstream_fused"
    assert null["comparisons"]["wave10/outer_leftdiag_upstream_fused"]["evaluations"]["c4_loss"][
        "final_delta"
    ] == pytest.approx(0.01)
    assert len(result["analyses"]) == 7
