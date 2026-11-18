# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real archive and process-start boundaries for selected-LR validation ownership."""

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import owned_output_handoff as ownership
import pytest
import run_selected_lr_validation as controller
import selected_lr_wave10_handoff as handoff
from prepare_selected_lr_validation import require_archive
from selected_lr_validation import locations


def test_selected_lr_archive_accepts_exact_bytes_and_rejects_changed_inventory(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"candidate and controller bytes\n")
    expected = {"tracking/source.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        bundle.add(source, arcname="tracking/source.py")
    require_archive(archive, expected, alias=False)
    source.write_bytes(b"other bytes\n")
    with pytest.raises(ValueError):
        require_archive(archive, {"tracking/source.py": hashlib.sha256(source.read_bytes()).hexdigest()}, alias=False)
    duplicate = tmp_path / "duplicate.tar"
    with tarfile.open(duplicate, "x") as bundle:
        bundle.add(source, arcname="tracking/source.py")
        bundle.add(source, arcname="tracking/source.py")
    with pytest.raises(ValueError):
        require_archive(duplicate, expected, alias=False)


def test_selected_lr_helper_closure_includes_imported_numerical_and_cleanup_helpers():
    names = controller.helper_names(Path(controller.__file__).parent)
    assert {
        "validate_selected_lr_full_budget.py",
        "validate_fixed_age_tuning.py",
        "run_diagonal_diagnostic.py",
        "wave7_release.py",
        "prepare_selected_lr_full_budget.py",
        "selected_lr_wave10_handoff.py",
    } <= set(names)


def test_selected_lr_start_withholds_intent_when_predecessor_is_running(tmp_path, monkeypatch):
    root = tmp_path / "output"
    _, helpers, state_path, work, receipt = locations(root)
    helpers.mkdir(parents=True)
    frozen = {"status": controller.STATUS, "candidate": str(tmp_path / "candidate")}
    monkeypatch.setattr(controller, "binding", lambda root: frozen)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", controller.MASK)

    def running(root):
        raise RuntimeError("original Wave10 workers still running")

    monkeypatch.setattr(controller, "require_completed_wave10", running)
    with pytest.raises(RuntimeError):
        controller.start(root)
    assert not any(path.exists() for path in (helpers.parent / "controller.json", state_path, work, receipt))


def test_selected_lr_detached_start_records_exact_handle_and_refuses_duplicate(tmp_path, monkeypatch):
    root = tmp_path / "output"
    binding_path, helpers, _, _, _ = locations(root)
    helpers.mkdir(parents=True)
    binding_path.parent.mkdir(parents=True)
    binding_path.write_text("{}\n")
    frozen = {"status": controller.STATUS}
    monkeypatch.setattr(controller, "binding", lambda root: frozen)
    monkeypatch.setattr(controller, "require_completed_wave10", lambda root: {"status": "complete"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", controller.MASK)
    marker_file = helpers.parent / "observed.json"
    script = (
        "import json,os,time,sys\nfrom pathlib import Path\n"
        f"Path({str(marker_file)!r}).write_text(json.dumps({{'pid':os.getpid(),'argv':sys.argv,"
        f"'mask':os.environ['CUDA_VISIBLE_DEVICES'],'marker':os.environ[{controller.CONTROLLER_MARKER!r}]}}))\n"
        "time.sleep(60)\n"
    )
    command = [controller.PYTHON, "-c", script]
    monkeypatch.setattr(controller, "controller_command", lambda root: command)
    intent = controller.start(root)
    try:
        process = controller.observed(intent["pid"])
        assert process["start_ticks"] == intent["start_ticks"]
        assert process["command"] == command
        assert process["group"] == intent["pid"]
        assert process["session"] == intent["pid"]
        with pytest.raises(FileExistsError):
            controller.start(root)
        assert json.loads((helpers.parent / "controller.json").read_text())["pid"] == intent["pid"]
    finally:
        process = controller.observed(intent["pid"])
        if process is not None and process["start_ticks"] == intent["start_ticks"]:
            os.kill(intent["pid"], 15)
        os.waitpid(intent["pid"], 0)


def test_selected_lr_bounded_child_failure_preserves_cleanup_and_log(tmp_path, monkeypatch):
    root = tmp_path / "output"
    _, helpers, state_path, _, _ = locations(root)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    frozen = {"candidate": str(candidate)}
    command = [controller.PYTHON, "-c", "import sys; print('original failure'); sys.exit(7)"]
    monkeypatch.setattr(controller, "validator_command", lambda *args: command)
    state = {}
    code = controller.bounded_validator(root, frozen, state)
    assert code == 7
    assert state["cleanup_exit_code"] == 7
    assert not controller.tagged_processes(state["child_marker"])
    saved = json.loads(state_path.read_text())
    assert saved["command"] == command
    assert saved["log_sha256"] == controller.sha256(Path(saved["log"]))
    assert "original failure" in Path(saved["log"]).read_text()
    assert controller.observed(saved["child_pid"]) is None


def record(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def proc_record(directory, pid, parent, ticks, command, environment):
    path = directory / str(pid)
    path.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[:4] = ["S", str(parent), str(pid), str(pid)]
    fields[19] = ticks
    path.joinpath("stat").write_text(f"{pid} (wave11 CPU fixture) {' '.join(fields)}\n")
    path.joinpath("cmdline").write_bytes(("\0".join(command) + "\0").encode())
    path.joinpath("environ").write_bytes(
        ("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode()
    )


@pytest.fixture
def wave10_handoff(tmp_path, monkeypatch):
    root, proc = tmp_path / "output", tmp_path / "proc"
    helpers = root / "controllers/wave10-launch-20261117/tracking"
    arms = ("muont", "outer_leftdiag_fused", "outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")
    nodes = (controller.NODE, "28.216.17.80", "28.216.17.144", "28.216.17.139")
    release_path = root / "run-preparation/wave10-launch-binding.json"
    release = {
        "arms": list(arms),
        "nodes": list(nodes),
        "helpers": str(helpers),
        "commands": {
            node: [controller.PYTHON, "-m", "experiment.train", "--config", f"{arm}.json"]
            for arm, node in zip(arms, nodes, strict=True)
        },
    }
    record(release_path, release)
    keeper_file = tmp_path / "keeper.pid"
    keeper_file.write_text("51000\n")
    monkeypatch.setattr(ownership, "PROC_ROOT", proc)
    monkeypatch.setattr(ownership, "KEEPER_PID", keeper_file)
    for rank in range(9):
        pid = 51000 + rank
        command = (
            [
                ownership.PYTHON,
                ownership.TORCHRUN,
                "--standalone",
                "--nproc-per-node=8",
                ownership.WORKLOAD,
                *ownership.WORKLOAD_ARGS,
            ]
            if rank == 0
            else [ownership.PYTHON, "-u", ownership.WORKLOAD, *ownership.WORKLOAD_ARGS]
        )
        proc_record(
            proc,
            pid,
            1 if rank == 0 else 51000,
            str(1000 + rank),
            command,
            {handoff.WORKER_MARKER: "worker-muont"},
        )
    generation = {pid: ownership.observed(pid) for pid in range(51000, 51009)}
    histories = {
        arm: {
            "manifest_sha256": f"manifest-{arm}",
            "metrics_sha256": f"metrics-{arm}",
            "checkpoint_receipt_sha256": f"checkpoint-{arm}",
            "checkpoint_rank_bytes": [100] * 8,
        }
        for arm in arms
    }
    columns = {
        arm: {"point_counts": {"tokens": 5120, "train/loss": 5120}, "final_metrics": {"tokens": 2684354560}}
        for arm in arms
    }
    for index, (arm, node) in enumerate(zip(arms, nodes, strict=True)):
        worker_pid, child_pid = 52000 + index, 53000 + index
        command = [
            controller.PYTHON,
            str(helpers / "owned_output_production.py"),
            "--root",
            str(root),
            "--phase",
            "worker",
            "--arm",
            arm,
        ]
        state_path = root / f"run-preparation/wave10-worker-{node}.json"
        record(
            state_path,
            {
                "status": "complete",
                "arm": arm,
                "node": node,
                "command": release["commands"][node],
                "worker_command": command,
                "worker_pid": worker_pid,
                "worker_start_ticks": str(2000 + index),
                "worker_marker": f"worker-{arm}",
                "child_pid": child_pid,
                "child_marker": f"child-{arm}",
                "release_binding_sha256": controller.sha256(release_path),
                "completed_run": histories[arm],
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "keeper_restore": 0,
                "restored_keeper": {
                    "pid": 51000,
                    "generation": {str(pid): process | {"status": "R"} for pid, process in generation.items()},
                },
            },
        )
        record(
            state_path.with_suffix(".start.json"),
            {
                "status": "acknowledged",
                "arm": arm,
                "node": node,
                "pid": worker_pid,
                "start_ticks": str(2000 + index),
                "command": command,
                "release_binding_sha256": controller.sha256(release_path),
                "worker_marker": f"worker-{arm}",
                "generation": {
                    "pid": worker_pid,
                    "status": "S",
                    "parent": 1,
                    "start_ticks": str(2000 + index),
                    "command": command,
                    "group": worker_pid,
                    "session": worker_pid,
                },
            },
        )
    record(
        root / "run-preparation/wave10-launch-state.json",
        {
            "binding_sha256": controller.sha256(release_path),
            "worker_markers": {arm: f"worker-{arm}" for arm in arms},
        },
    )
    analysis = {
        "analyses": [{"comparisons": {"muont": {"final_comparison_eligible": True, "training_windows": [{}] * 5}}}]
    }
    record(root / "run-preparation/analysis-wave10.json", analysis)
    record(
        root / "run-preparation/swanlab-upload-wave10.json",
        {
            "status": "verified",
            "visibility": "PRIVATE",
            "project": "siyuc3141/output-side-normalization-marin",
            "runs": {
                arm: {
                    "status": "verified",
                    **histories[arm],
                    "verification": {
                        "state": "FINISHED",
                        "configuration_verified": True,
                        "token_axis_verified": True,
                        "integer_values_exact": True,
                        "csv_export_max_ulp": 2,
                        "all_scalar_values_exact": True,
                        **columns[arm],
                        "serialization_errors": {
                            name: {"max_absolute": 0.0, "max_ulp": 0.0} for name in columns[arm]["point_counts"]
                        },
                    },
                }
                for arm in arms
            },
        },
    )
    evidence = SimpleNamespace(
        root=root,
        proc=proc,
        state_path=root / f"run-preparation/wave10-worker-{nodes[0]}.json",
        keeper_file=keeper_file,
        generation=generation,
        failure=None,
        effects=[],
    )

    def run(command, **kwargs):
        if "experiment.bootstrap" in command:
            if evidence.failure == "bootstrap":
                raise RuntimeError("CPU-injected bootstrap failure")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "ssh":
            arm = next(arm for arm in arms if f"--arm {arm}" in command[-1])
            payload = {
                "status": "wave10_original_training_absent_exact_keeper_verified",
                "arm": arm,
                "node": nodes[arms.index(arm)],
                "original_handles_absent": True,
                "training_child_markers_absent": True,
            }
        elif "-c" in command:
            script = command[command.index("-c") + 1]
            if "snapshot(root)" in script:
                if evidence.failure == "prerequisite":
                    raise RuntimeError("CPU-injected prerequisite failure")
                payload = {
                    "snapshot": {
                        "status": "complete",
                        "all_workers_terminal": True,
                        "binding_sha256": controller.sha256(release_path),
                        "workers": [
                            {
                                "arm": arm,
                                "status": "complete",
                                "receipt": json.loads(
                                    root.joinpath(f"run-preparation/wave10-worker-{node}.json").read_text()
                                ),
                            }
                            for arm, node in zip(arms, nodes, strict=True)
                        ],
                    },
                    "analysis": analysis,
                    "histories": histories,
                }
            elif "expected_columns" in script:
                payload = columns
            elif "completed_run" in script:
                payload = histories[command[-1]]
            else:
                raise AssertionError(f"Unexpected CPU subprocess: {command}")
        else:
            evidence.effects.append(command)
            raise AssertionError(f"Unexpected execution or restoration: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    def check_output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return f"{nodes[0]}\n"
        if command == ["ps", "-eo", "args="]:
            return "-m torch.distributed.run\n" if evidence.failure == "idle" else ""
        if command == ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"]:
            if evidence.failure == "keeper":
                raise RuntimeError("CPU-injected keeper failure")
            return "\n".join(f"{rank}, GPU-{rank}, NVIDIA A100-SXM4-40GB" for rank in range(8))
        if command == ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]:
            return "\n".join(f"{51001 + rank}, GPU-{rank}" for rank in range(8))
        if command == ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]:
            return "\n".join(str(pid) for pid in range(51001, 51009))
        raise AssertionError(f"Unexpected CPU device/process query: {command}")

    def forbidden(*args, **kwargs):
        evidence.effects.append(args)
        raise AssertionError("CPU handoff verification must not launch or signal a process")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden)
    return evidence


def test_selected_lr_handoff_accepts_original_acknowledgment_and_json_restoration(wave10_handoff):
    evidence = wave10_handoff
    before = evidence.state_path.read_bytes()
    result = handoff.probe(evidence.root, "muont")
    assert result["status"] == "wave10_original_training_absent_exact_keeper_verified"
    assert result["original_handles_absent"] is True
    assert result["training_child_markers_absent"] is True
    assert result["keeper"]["pid"] == 51000
    assert result["keeper"]["generation"] == evidence.generation
    assert result["inherited_worker_marker_pids"] == list(range(51000, 51009))
    assert evidence.state_path.read_bytes() == before
    assert evidence.effects == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (("status",), "starting"),
        (("pid",), 59000),
        (("start_ticks",), "reused"),
        (("command",), ["unissued-worker"]),
        (("arm",), "outer_leftdiag_fused"),
        (("node",), "28.216.17.80"),
        (("release_binding_sha256",), "different-release"),
        (("worker_marker",), "different-worker"),
        (("generation", "pid"), 59000),
        (("generation", "start_ticks"), "reused"),
        (("generation", "command"), ["unissued-worker"]),
        (("generation", "group"), 59000),
        (("generation", "session"), 59000),
    ),
)
def test_selected_lr_handoff_rejects_changed_original_acknowledgment(wave10_handoff, field, replacement):
    path = wave10_handoff.state_path.with_suffix(".start.json")
    acknowledged = json.loads(path.read_text())
    target = acknowledged if len(field) == 1 else acknowledged[field[0]]
    target[field[-1]] = replacement
    record(path, acknowledged)
    with pytest.raises(RuntimeError):
        handoff.probe(wave10_handoff.root, "muont")
    assert wave10_handoff.effects == []


@pytest.mark.parametrize("fault", ("pid", "rank_pid", "start_ticks", "group", "session", "command", "missing", "extra"))
def test_selected_lr_handoff_rejects_replaced_restored_keeper(wave10_handoff, fault):
    state = json.loads(wave10_handoff.state_path.read_text())
    keeper = state["restored_keeper"]
    rank = keeper["generation"]["51001"]
    if fault == "pid":
        keeper["pid"] = 59000
    elif fault == "rank_pid":
        rank["pid"] = 59000
    elif fault == "start_ticks":
        rank["start_ticks"] = "reused"
    elif fault in ("group", "session"):
        rank[fault] = 59000
    elif fault == "command":
        rank["command"] = ["replacement-keeper"]
    elif fault == "missing":
        del keeper["generation"]["51001"]
    else:
        keeper["generation"]["59000"] = rank | {"pid": 59000}
    record(wave10_handoff.state_path, state)
    with pytest.raises(RuntimeError):
        handoff.probe(wave10_handoff.root, "muont")
    assert wave10_handoff.effects == []


@pytest.mark.parametrize("handle", ("worker_pid", "child_pid"))
def test_selected_lr_handoff_refuses_live_original_handle(wave10_handoff, handle):
    state = json.loads(wave10_handoff.state_path.read_text())
    proc_record(wave10_handoff.proc, state[handle], 1, "2000", state["worker_command"], {})
    with pytest.raises(RuntimeError):
        handoff.probe(wave10_handoff.root, "muont")
    assert wave10_handoff.effects == []


def test_selected_lr_handoff_refuses_inherited_marker_outside_restored_keeper(wave10_handoff):
    proc_record(wave10_handoff.proc, 59000, 1, "3000", ["unrelated-process"], {handoff.WORKER_MARKER: "worker-muont"})
    with pytest.raises(RuntimeError):
        handoff.probe(wave10_handoff.root, "muont")
    assert wave10_handoff.effects == []


@pytest.mark.parametrize("failure", ("guard", "prerequisite", "idle", "keeper", "bootstrap"))
def test_selected_lr_preflight_failure_is_terminal_without_keeper_mutation(wave10_handoff, monkeypatch, failure):
    evidence = wave10_handoff
    binding_path, helpers, state_path, work, receipt = locations(evidence.root)
    candidate = evidence.root / "candidate"
    candidate.mkdir()
    record(binding_path, {"candidate": str(candidate)})
    command = controller.controller_command(evidence.root)
    proc_record(
        evidence.proc,
        os.getpid(),
        1,
        "controller-generation",
        command,
        controller.ENVIRONMENT | {controller.CONTROLLER_MARKER: "controller-marker"},
    )
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": os.getpid(),
            "start_ticks": "controller-generation",
            "command": command,
            "marker": "controller-marker",
            "binding_sha256": controller.sha256(binding_path),
        },
    )
    monkeypatch.setenv(controller.CONTROLLER_MARKER, "controller-marker")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", controller.MASK)
    # The candidate's CPU/archive acceptance is outside this preflight failure contract.
    monkeypatch.setattr(controller, "binding", lambda root: json.loads(binding_path.read_text()))
    evidence.failure = failure
    keeper_before = evidence.keeper_file.read_bytes()
    flock = controller.fcntl.flock
    preflight = []

    def lock(fd, operation):
        preflight.append(json.loads(state_path.read_text()))
        if failure == "guard":
            raise BlockingIOError("CPU-injected guard failure")
        return flock(fd, operation)

    monkeypatch.setattr(controller.fcntl, "flock", lock)
    with pytest.raises((RuntimeError, BlockingIOError)) as error:
        controller.validate(evidence.root)
    assert preflight[0]["status"] == "preflight"
    saved = json.loads(state_path.read_text())
    assert saved["status"] == "needs_attention"
    assert saved["error"] == f"{type(error.value).__name__}: {error.value}"
    for state in (preflight[0], saved):
        assert state["binding_sha256"] == controller.sha256(binding_path)
        assert state["controller_pid"] == os.getpid()
        assert state["controller_start_ticks"] == "controller-generation"
        assert state["node"] == controller.NODE
        assert state["candidate"] == str(candidate)
    assert not saved.get("keeper_release_attempted")
    assert "keeper_restore" not in saved
    assert evidence.keeper_file.read_bytes() == keeper_before
    assert all(
        ownership.same_generation(process, ownership.observed(pid)) for pid, process in evidence.generation.items()
    )
    assert evidence.effects == []
    assert not work.exists()
    assert not receipt.exists()
