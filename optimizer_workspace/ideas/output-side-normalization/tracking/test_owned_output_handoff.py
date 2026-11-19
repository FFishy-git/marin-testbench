# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise exact keeper ownership and experiment absence against a fake process tree."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import owned_output_handoff as handoff
import pytest


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def process(proc, pid, parent, command, *, marker=None, session=None, ticks=None):
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[0:4] = ["S", str(parent), str(pid), str(pid if session is None else session)]
    fields[19] = str(pid) if ticks is None else ticks
    (directory / "stat").write_text(f"{pid} (fixture) {' '.join(fields)}")
    (directory / "cmdline").write_bytes(("\0".join(command) + "\0").encode())
    (directory / "environ").write_bytes(b"" if marker is None else (marker + "\0").encode())


@pytest.fixture
def allocation(tmp_path, monkeypatch):
    root, proc = tmp_path / "root", tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(handoff, "PROC_ROOT", proc)
    pidfile = tmp_path / "keeper.pid"
    pidfile.write_text("500")
    monkeypatch.setattr(handoff, "KEEPER_PID", pidfile)
    marker = "family-marker"
    for pid in range(500, 509):
        command = (
            [handoff.PYTHON, handoff.TORCHRUN, "--standalone", "--nproc-per-node=8", handoff.WORKLOAD]
            if pid == 500
            else [handoff.PYTHON, "-u", handoff.WORKLOAD]
        )
        process(
            proc,
            pid,
            1 if pid == 500 else 500,
            [*command, *handoff.WORKLOAD_ARGS],
            marker=f"{handoff.WORKER_MARKER}={marker}",
        )
    directory = root / handoff.CEILING
    campaign = directory / "campaign.json"
    record(campaign, {"root": str(root)})
    planned, trials = [], []
    for index in range(2):
        name = f"muont_lr{index}"
        command = ["bound-train", name]
        path = directory / "workers" / f"{name}.execution.json"
        record(path, {"child_pid": 100 + index, "child_marker": name, "command": command, "cleanup_exit_code": 0})
        planned.append({"name": name, "command": command})
        trials.append(
            {
                "name": name,
                "command": command,
                "status": "complete",
                "exit_code": 0,
                "execution_sha256": handoff.sha256(path),
            }
        )
    binding = directory / "execution-binding.json"
    record(
        binding,
        {
            "campaign": str(campaign),
            "campaign_sha256": handoff.sha256(campaign),
            "plan": [{"family": "muont", "node": "28.216.17.9", "trials": planned}],
        },
    )
    record(directory / "launch-state.json", {"worker_markers": {"muont": marker}})
    state = directory / "workers/muont.json"
    record(
        state,
        {
            "family": "muont",
            "node": "28.216.17.9",
            "status": "complete",
            "keeper_restore": 0,
            "binding_sha256": handoff.sha256(binding),
            "pid": 99,
            "worker_marker": marker,
            "trials": trials,
        },
    )
    output = {
        "devices": [f"{index}, GPU-{index}, NVIDIA A100-SXM4-40GB" for index in range(8)],
        "occupied": [f"{501 + index}, GPU-{index}" for index in range(8)],
    }

    def check_output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return "28.216.17.9\n"
        key = "devices" if "--query-gpu=index,uuid,name" in command else "occupied"
        return "\n".join(output[key]) + "\n"

    monkeypatch.setattr(handoff.subprocess, "check_output", check_output)
    return root, proc, state, output


def test_inherited_marker_is_allowed_only_on_the_exact_healthy_keeper(allocation):
    root, _, _, _ = allocation
    result = handoff.probe(root, "muont")
    assert result["original_handles_absent"] and result["training_child_markers_absent"]
    assert result["inherited_worker_marker_pids"] == list(range(500, 509))
    assert len(result["keeper"]["occupied"]) == 8


def test_sanitized_keeper_also_passes_without_silently_requiring_a_restart(allocation):
    root, proc, _, _ = allocation
    for pid in range(500, 509):
        (proc / str(pid) / "environ").write_bytes(b"")
    assert handoff.probe(root, "muont")["inherited_worker_marker_pids"] == []


@pytest.mark.parametrize(
    "fault",
    (
        "live_worker",
        "reused_child",
        "child_marker_on_keeper",
        "orphan_worker_marker",
        "extra_keeper_child",
        "changed_keeper_command",
        "wrong_session",
        "unrelated_occupancy",
        "duplicate_gpu",
        "missing_gpu",
        "wrong_gpu",
        "false_restore",
    ),
)
def test_unrelated_or_live_experiment_processes_never_authorize_handoff(allocation, fault):
    root, proc, state, output = allocation
    if fault in ("live_worker", "reused_child", "orphan_worker_marker", "extra_keeper_child"):
        pid = {"live_worker": 99, "reused_child": 100, "orphan_worker_marker": 900, "extra_keeper_child": 900}[fault]
        process(
            proc,
            pid,
            500 if fault == "extra_keeper_child" else 1,
            ["unrelated"],
            marker=f"{handoff.WORKER_MARKER}=family-marker" if fault == "orphan_worker_marker" else None,
        )
    elif fault == "child_marker_on_keeper":
        (proc / "501/environ").write_bytes(f"{handoff.CHILD_MARKER}=muont_lr0\0".encode())
    elif fault == "changed_keeper_command":
        (proc / "501/cmdline").write_bytes(b"unrelated\0")
    elif fault == "wrong_session":
        process(proc, 501, 500, [handoff.PYTHON, "-u", handoff.WORKLOAD, *handoff.WORKLOAD_ARGS], session=500)
    elif fault == "unrelated_occupancy":
        output["occupied"][-1] = "900, GPU-7"
    elif fault == "duplicate_gpu":
        output["occupied"][-1] = "508, GPU-0"
    elif fault == "missing_gpu":
        output["devices"].pop()
    elif fault == "wrong_gpu":
        output["devices"][-1] = "7, GPU-7, NVIDIA H100"
    else:
        value = handoff.read(state)
        value["keeper_restore"] = False
        record(state, value)
    with pytest.raises(RuntimeError):
        handoff.probe(root, "muont")


def test_keeper_generation_drift_during_device_inspection_is_refused(allocation, monkeypatch):
    root, proc, _, _ = allocation
    check_output = handoff.subprocess.check_output

    def drifting(command, **kwargs):
        value = check_output(command, **kwargs)
        if "--query-compute-apps=pid,gpu_uuid" in command:
            path = proc / "501/stat"
            path.write_text(path.read_text().replace("501", "999", 1).rsplit(" ", 1)[0] + " 999")
        return value

    monkeypatch.setattr(handoff.subprocess, "check_output", drifting)
    with pytest.raises(RuntimeError, match="generation changed"):
        handoff.probe(root, "muont")


@pytest.mark.parametrize("transition", ("R", "Z", "parent", "group", "session", "command"))
def test_handoff_accepts_scheduler_transition_but_refuses_owner_changes(allocation, monkeypatch, transition):
    root, proc, _, _ = allocation
    check_output = handoff.subprocess.check_output

    def transitioning(command, **kwargs):
        value = check_output(command, **kwargs)
        if "--query-compute-apps=pid,gpu_uuid" in command:
            directory = proc / "501"
            if transition == "command":
                (directory / "cmdline").write_bytes(b"unrelated\0")
            else:
                path = directory / "stat"
                prefix, fields = path.read_text().rsplit(")", 1)
                fields = fields.split()
                index = {"R": 0, "Z": 0, "parent": 1, "group": 2, "session": 3}[transition]
                fields[index] = transition if index == 0 else "999"
                path.write_text(prefix + ") " + " ".join(fields))
        return value

    monkeypatch.setattr(handoff.subprocess, "check_output", transitioning)
    if transition == "R":
        result = handoff.probe(root, "muont")
        assert result["status"] == "original_training_absent_exact_keeper_verified"
        assert result["inherited_worker_marker_pids"] == list(range(500, 509))
    else:
        with pytest.raises(RuntimeError, match="generation changed"):
            handoff.probe(root, "muont")


@pytest.mark.parametrize("exit_state", ("removed", "zombie", "environ_removed"))
def test_marker_scan_rescans_exiting_helpers_and_retains_new_live_children(allocation, monkeypatch, exit_state):
    _, proc, _, _ = allocation
    marker = f"{handoff.CHILD_MARKER}=compiler-helper"
    process(proc, 900, 1, ["exiting-helper"], marker=marker)
    process(proc, 901, 1, ["surviving-helper"], marker=marker)
    read_bytes = Path.read_bytes
    changed = False

    def exiting(path):
        nonlocal changed
        if path != proc / "900/environ" or changed:
            return read_bytes(path)
        changed = True
        tokens = read_bytes(path)
        process(proc, 902, 1, ["new-helper"], marker=marker)
        if exit_state == "zombie":
            stat = proc / "900/stat"
            prefix, fields = stat.read_text().rsplit(")", 1)
            fields = fields.split()
            fields[0] = "Z"
            stat.write_text(prefix + ") " + " ".join(fields))
            (proc / "900/cmdline").write_bytes(b"")
        else:
            shutil.rmtree(proc / "900")
            if exit_state == "environ_removed":
                raise FileNotFoundError(path)
        return tokens

    monkeypatch.setattr(Path, "read_bytes", exiting)
    result = handoff.marker_processes(handoff.CHILD_MARKER, "compiler-helper")
    assert set(result) == {901, 902}
    assert result[901]["command"] == ["surviving-helper"]
    assert result[902]["command"] == ["new-helper"]


def test_exiting_training_marker_can_complete_exact_keeper_handoff(allocation, monkeypatch):
    root, proc, _, _ = allocation
    process(proc, 900, 1, ["exiting-child"], marker=f"{handoff.CHILD_MARKER}=muont_lr0")
    read_bytes = Path.read_bytes

    def exiting(path):
        tokens = read_bytes(path)
        if path == proc / "900/environ":
            shutil.rmtree(proc / "900")
        return tokens

    monkeypatch.setattr(Path, "read_bytes", exiting)
    result = handoff.probe(root, "muont")
    assert result["original_handles_absent"] and result["training_child_markers_absent"]
    assert result["inherited_worker_marker_pids"] == list(range(500, 509))
    assert len(result["keeper"]["occupied"]) == 8


@pytest.mark.parametrize("change", ("start_ticks", "parent", "group", "session", "command"))
def test_marker_scan_refuses_live_identity_changes(allocation, monkeypatch, change):
    _, proc, _, _ = allocation
    process(proc, 900, 1, ["marked-helper"], marker=f"{handoff.CHILD_MARKER}=compiler-helper")
    read_bytes = Path.read_bytes
    changed = False

    def substituted(path):
        nonlocal changed
        tokens = read_bytes(path)
        if path == proc / "900/environ" and not changed:
            changed = True
            if change == "command":
                (proc / "900/cmdline").write_bytes(b"replacement-helper\0")
            else:
                stat = proc / "900/stat"
                prefix, fields = stat.read_text().rsplit(")", 1)
                fields = fields.split()
                fields[{"start_ticks": 19, "parent": 1, "group": 2, "session": 3}[change]] = "999"
                stat.write_text(prefix + ") " + " ".join(fields))
        return tokens

    monkeypatch.setattr(Path, "read_bytes", substituted)
    with pytest.raises(RuntimeError):
        handoff.marker_processes(handoff.CHILD_MARKER, "compiler-helper")


def test_marker_scan_refuses_reused_pid_after_transient_disappearance(allocation, monkeypatch):
    _, proc, _, _ = allocation
    marker = f"{handoff.CHILD_MARKER}=compiler-helper"
    process(proc, 900, 1, ["original-helper"], marker=marker)
    process(proc, 901, 1, ["surviving-helper"], marker=marker)
    read_bytes = Path.read_bytes
    vanished = False
    replaced = False

    def substituted(path):
        nonlocal vanished, replaced
        tokens = read_bytes(path)
        if path == proc / "900/environ" and not vanished:
            vanished = True
            shutil.rmtree(proc / "900")
        elif path == proc / "901/environ" and vanished and not replaced:
            replaced = True
            process(proc, 900, 1, ["replacement-helper"], marker=marker, ticks="999")
        return tokens

    monkeypatch.setattr(Path, "read_bytes", substituted)
    with pytest.raises(RuntimeError):
        handoff.marker_processes(handoff.CHILD_MARKER, "compiler-helper")


def test_marker_scan_never_returns_absence_on_persistent_unreadable_environment(allocation, monkeypatch):
    _, proc, _, _ = allocation
    process(proc, 900, 1, ["marked-helper"], marker=f"{handoff.CHILD_MARKER}=compiler-helper")
    read_bytes = Path.read_bytes

    def unreadable(path):
        if path == proc / "900/environ":
            raise FileNotFoundError(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(RuntimeError):
        handoff.marker_processes(handoff.CHILD_MARKER, "compiler-helper")


def test_marker_scan_refuses_continuous_descendant_exit_churn(allocation, monkeypatch):
    _, proc, _, _ = allocation
    marker = f"{handoff.CHILD_MARKER}=compiler-helper"
    process(proc, 900, 1, ["exiting-helper"], marker=marker)
    read_bytes = Path.read_bytes

    def exiting(path):
        tokens = read_bytes(path)
        if path.name == "environ" and marker.encode() in tokens.split(b"\0"):
            pid = int(path.parent.name)
            process(proc, pid + 1, 1, ["replacement-helper"], marker=marker)
            shutil.rmtree(path.parent)
        return tokens

    monkeypatch.setattr(Path, "read_bytes", exiting)
    with pytest.raises(RuntimeError):
        handoff.marker_processes(handoff.CHILD_MARKER, "compiler-helper")
    assert any(path.is_dir() and int(path.name) >= 900 for path in proc.iterdir())


def test_marker_scan_refuses_live_marker_removal_during_rescan(allocation, monkeypatch):
    _, proc, _, _ = allocation
    marker = f"{handoff.CHILD_MARKER}=compiler-helper"
    for pid in (900, 901):
        process(proc, pid, 1, ["marked-helper"], marker=marker)
    read_bytes = Path.read_bytes
    changed = False

    def changed_environment(path):
        nonlocal changed
        tokens = read_bytes(path)
        if path == proc / "901/environ" and not changed:
            changed = True
            shutil.rmtree(proc / "900")
            path.write_bytes(b"")
        return tokens

    monkeypatch.setattr(Path, "read_bytes", changed_environment)
    with pytest.raises(RuntimeError):
        handoff.marker_processes(handoff.CHILD_MARKER, "compiler-helper")


def test_unrelated_worker_discovered_after_exit_never_authorizes_handoff(allocation, monkeypatch):
    root, proc, _, _ = allocation
    marker = f"{handoff.WORKER_MARKER}=family-marker"
    process(proc, 900, 1, ["exiting-helper"], marker=marker)
    read_bytes = Path.read_bytes

    def exiting(path):
        tokens = read_bytes(path)
        if path == proc / "900/environ":
            process(proc, 901, 1, ["unrelated-worker"], marker=marker)
            shutil.rmtree(proc / "900")
        return tokens

    monkeypatch.setattr(Path, "read_bytes", exiting)
    with pytest.raises(RuntimeError):
        handoff.probe(root, "muont")


def test_marker_scan_handles_real_linux_child_exit_after_environment_read(monkeypatch):
    monkeypatch.setattr(handoff, "PROC_ROOT", Path("/proc"))
    marker = uuid4().hex
    read_bytes = Path.read_bytes
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        env=os.environ | {handoff.CHILD_MARKER: marker},
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as child:

        def exiting(path):
            tokens = read_bytes(path)
            if path == Path("/proc") / str(child.pid) / "environ":
                child.stdin.close()
                child.wait(timeout=10)
            return tokens

        monkeypatch.setattr(Path, "read_bytes", exiting)
        assert handoff.marker_processes(handoff.CHILD_MARKER, marker) == {}
        assert child.returncode == 0
