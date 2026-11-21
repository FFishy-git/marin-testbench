# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline restoration acceptance with private process files and fake OS commands."""

import json
import os
import shutil
import subprocess

import owned_output_handoff as handoff
import pytest
import related_restore as recovery


def process_files(proc, pid, command, *, parent=1, marker=None):
    directory = proc / str(pid)
    directory.mkdir()
    fields = ["0"] * 21
    fields[:4] = ["S", str(parent), str(pid), str(pid)]
    fields[19:21] = [str(pid * 10), "4096"]
    (directory / "stat").write_text(f"{pid} (offline fixture) {' '.join(fields)}")
    (directory / "cmdline").write_bytes(("\0".join(command) + "\0").encode())
    environment = f"{recovery.MARKER}={marker}\0".encode() if marker is not None else b""
    (directory / "environ").write_bytes(environment)


class RestoreBoundary:
    def __init__(self, proc, manager, keeper_pointer):
        self.proc = proc
        self.manager = manager
        self.keeper_pointer = keeper_pointer
        self.devices = [[str(index), f"GPU-original-{index}", "NVIDIA A100-SXM4-40GB"] for index in range(8)]
        self.restored_devices = [row.copy() for row in self.devices]
        self.occupied = ""
        self.manager_starts = []
        self.keeper_pid = 43000

    def check_output(self, command, **kwargs):
        assert kwargs.get("text") is True
        if command == ["hostname", "-I"]:
            return "28.216.17.9\n"
        if command == ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]:
            return self.occupied
        if command == ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"]:
            return "".join(", ".join(row) + "\n" for row in self.restored_devices)
        if command == ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]:
            return "".join(
                f"{self.keeper_pid + index + 1}, {device[1]}\n" for index, device in enumerate(self.restored_devices)
            )
        raise AssertionError(f"Unexpected offline command: {command}")

    def run(self, command, **kwargs):
        assert command == ["bash", str(self.manager), "start"]
        environment = kwargs["env"]
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert recovery.MARKER not in environment
        assert environment["GPU_RESOURCE_MANAGER_LOCAL_ONLY"] == "1"
        lock_fd = int(environment["GPU_RESOURCE_MANAGER_LOCK_FD"])
        assert kwargs["pass_fds"] == (lock_fd,)
        assert os.readlink(f"/proc/self/fd/{lock_fd}") == str(self.manager.with_suffix(".lock"))
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 120
        self.manager_starts.append({"command": command, "environment": environment})
        launcher = [
            handoff.PYTHON,
            handoff.TORCHRUN,
            "--standalone",
            "--nproc-per-node=8",
            handoff.WORKLOAD,
            *handoff.WORKLOAD_ARGS,
        ]
        process_files(self.proc, self.keeper_pid, launcher)
        for rank in range(8):
            process_files(
                self.proc,
                self.keeper_pid + rank + 1,
                [handoff.PYTHON, "-u", handoff.WORKLOAD, *handoff.WORKLOAD_ARGS],
                parent=self.keeper_pid,
            )
        self.keeper_pointer.write_text(f"{self.keeper_pid}\n")
        return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def refused_owner(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    root = tmp_path / "output"
    (root / "run-preparation").mkdir(parents=True)
    manager = tmp_path / "offline_manager.sh"
    manager.write_text("# offline manager identity; never executed\n")
    keeper_pointer = tmp_path / "keeper.pid"
    boundary = RestoreBoundary(proc, manager, keeper_pointer)
    monkeypatch.setattr(handoff, "PROC_ROOT", proc)
    monkeypatch.setattr(handoff, "KEEPER_PID", keeper_pointer)
    monkeypatch.setattr(recovery, "KEEPER_PID", keeper_pointer)
    monkeypatch.setattr(recovery, "ROOT", root)
    monkeypatch.setattr(recovery, "LOCK", manager.with_suffix(".lock"))
    monkeypatch.setattr(recovery, "MANAGER", manager)
    monkeypatch.setattr(subprocess, "check_output", boundary.check_output)
    monkeypatch.setattr(subprocess, "run", boundary.run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv(recovery.MARKER, "inherited-offline-marker")
    original_keeper = 42000
    for pid in range(original_keeper, original_keeper + 9):
        process_files(proc, pid, ["/cpu/original_keeper.py"], parent=1 if pid == original_keeper else original_keeper)
    generation = {str(pid): handoff.observed(pid) for pid in range(original_keeper, original_keeper + 9)}
    for directory in proc.iterdir():
        shutil.rmtree(directory)
    state = {
        "status": "needs_attention",
        "pid": 41000,
        "child_pid": 41001,
        "child_marker": "original-offline-child",
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_release_complete": True,
        "restoration_error": "RuntimeError: CUDA occupancy prevents keeper restoration",
        "manager_sha256": recovery.digest(manager),
        "keeper": {
            "pid": original_keeper,
            "generation": generation,
            "devices": boundary.devices,
            "occupied": [[str(original_keeper + index + 1), device[1]] for index, device in enumerate(boundary.devices)],
        },
    }
    original = tmp_path / "original.owner.json"
    original.write_text(json.dumps(state, indent=2) + "\n")
    receipt = tmp_path / "recovery.json"
    return original, receipt, state, boundary


@pytest.mark.parametrize("handle", ["owner", "child", "keeper"])
def test_live_original_handle_refuses_without_manager_start_or_evidence_changes(refused_owner, handle):
    original, receipt, state, boundary = refused_owner
    pid = state["keeper"]["pid"] if handle == "keeper" else state["pid" if handle == "owner" else "child_pid"]
    process_files(boundary.proc, pid, ["/cpu/still_live.py"])
    before = original.read_bytes()
    with pytest.raises(RuntimeError, match="still live"):
        recovery.restore(original, receipt)
    assert boundary.manager_starts == []
    assert original.read_bytes() == before
    assert not receipt.exists()
    assert not boundary.keeper_pointer.exists()


def test_live_marked_descendant_refuses_without_manager_start_or_evidence_changes(refused_owner):
    original, receipt, state, boundary = refused_owner
    process_files(boundary.proc, 44000, ["/cpu/marked_descendant.py"], marker=state["child_marker"])
    before = original.read_bytes()
    with pytest.raises(RuntimeError, match="marked descendants"):
        recovery.restore(original, receipt)
    assert boundary.manager_starts == []
    assert original.read_bytes() == before
    assert not receipt.exists()
    assert not boundary.keeper_pointer.exists()


def test_occupied_gpus_refuse_without_manager_start_or_evidence_changes(refused_owner):
    original, receipt, _, boundary = refused_owner
    boundary.occupied = "45000\n"
    before = original.read_bytes()
    with pytest.raises(RuntimeError, match="CUDA occupancy remains"):
        recovery.restore(original, receipt)
    assert boundary.manager_starts == []
    assert original.read_bytes() == before
    assert not receipt.exists()
    assert not boundary.keeper_pointer.exists()


def test_changed_original_manager_refuses_without_start_or_evidence_changes(refused_owner):
    original, receipt, _, boundary = refused_owner
    boundary.manager.write_text("# changed offline manager identity\n")
    before = original.read_bytes()
    with pytest.raises(ValueError, match="manager changed"):
        recovery.restore(original, receipt)
    assert boundary.manager_starts == []
    assert original.read_bytes() == before
    assert not receipt.exists()
    assert not boundary.keeper_pointer.exists()


def test_valid_recovery_verifies_exact_physical_gpus_and_preserves_original_owner(refused_owner):
    original, receipt, state, boundary = refused_owner
    before = original.read_bytes()
    result = recovery.restore(original, receipt)
    assert len(boundary.manager_starts) == 1
    assert json.loads(json.dumps(result)) == json.loads(receipt.read_text())
    assert result["status"] == "complete"
    assert result["keeper_restore"] == 0
    assert result["original"] == str(original)
    assert result["original_sha256"] == recovery.digest(original)
    assert original.read_bytes() == before
    assert json.loads(original.read_text())["status"] == "needs_attention"
    keeper = result["restored_keeper"]
    assert keeper["pid"] == boundary.keeper_pid
    assert keeper["devices"] == state["keeper"]["devices"]
    assert set(map(int, keeper["generation"])) == set(range(boundary.keeper_pid, boundary.keeper_pid + 9))
    assert keeper["occupied"] == [
        [str(boundary.keeper_pid + index + 1), device[1]] for index, device in enumerate(boundary.devices)
    ]
    receipt_before = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        recovery.restore(original, receipt)
    assert len(boundary.manager_starts) == 1
    assert receipt.read_bytes() == receipt_before
    assert original.read_bytes() == before


def test_changed_physical_gpu_set_cannot_be_reported_as_completed_recovery(refused_owner):
    original, receipt, _, boundary = refused_owner
    boundary.restored_devices[0][1] = "GPU-different-physical-device"
    before = original.read_bytes()
    with pytest.raises(RuntimeError, match="original physical GPUs"):
        recovery.restore(original, receipt)
    assert len(boundary.manager_starts) == 1
    assert original.read_bytes() == before
    assert not receipt.exists()
