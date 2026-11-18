# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise watcher receipts and guarded restoration through fake OS/process I/O."""

import fcntl
import json
import os
import subprocess
from pathlib import Path

import pytest
import watch_wave7_release
import wave7_release
from launch_wave5 import KEEPER


class WatcherIO:
    """Keep real state files and controller guards; fake only /proc, time and subprocesses."""

    def __init__(self, root, monkeypatch):
        self.root = root
        self.node = wave7_release.NODES[0]
        self.out = root / "runs/wave7" / wave7_release.ARMS[0]
        self.path = self.out.parent / f"{self.out.name}.watcher.json"
        self.out.parent.mkdir(parents=True)
        preparation = root / "run-preparation"
        preparation.mkdir()
        self.helpers = root / "helpers"
        self.helpers.mkdir()
        tracking = Path(wave7_release.__file__).resolve().parent
        for name in wave7_release.HELPERS:
            (self.helpers / name).write_bytes((tracking / name).read_bytes())
        archive = root / "source.tar"
        archive.write_bytes(b"immutable archive")
        self.candidate = root / "candidate"
        record = {
            "helpers": str(self.helpers),
            "candidate": str(self.candidate),
            "arms": list(wave7_release.ARMS),
            "archive": str(archive),
            "archive_sha256": wave7_release.sha256(archive),
            "helpers_sha256": {name: wave7_release.sha256(self.helpers / name) for name in wave7_release.HELPERS},
        }
        (preparation / "wave7-release-binding.json").write_text(json.dumps(record))
        monkeypatch.setattr(wave7_release, "__file__", str(self.helpers / "wave7_release.py"))
        monkeypatch.setattr(watch_wave7_release, "__file__", str(self.helpers / "watch_wave7_release.py"))
        self.pid = 91001
        self.processes = {}
        self.starts = []
        self.spawns = []
        self.restore_states = []
        self.cuda = ""
        self.restore_reply = None
        self.restore_effect = None
        self.poll_effect = None
        self.lock_effect = None
        self.clock = 0.0
        self.spawn_status = "watching"
        self.restore_timeout = False
        self.pending_lock = None
        self.add_process(os.getpid(), ["python", str(self.helpers / "watch_wave7_release.py")], "801")
        original_text, original_bytes, original_iterdir, original_unlink = (
            Path.read_text,
            Path.read_bytes,
            Path.iterdir,
            Path.unlink,
        )

        def text(path, *args, **kwargs):
            if path.parent.parent == Path("/proc") and path.name == "stat":
                pid = int(path.parent.name)
                if pid not in self.processes:
                    raise FileNotFoundError(path)
                status, ticks, _ = self.processes[pid]
                return f"{pid} (python) " + " ".join([status, *["0"] * 18, ticks])
            return original_text(path, *args, **kwargs)

        def data(path, *args, **kwargs):
            if path.parent.parent == Path("/proc") and path.name == "cmdline":
                pid = int(path.parent.name)
                if pid not in self.processes:
                    raise FileNotFoundError(path)
                return b"\0".join(arg.encode() for arg in self.processes[pid][2]) + b"\0"
            return original_bytes(path, *args, **kwargs)

        def directories(path):
            if path == Path("/proc"):
                return iter(path / str(pid) for pid in self.processes)
            return original_iterdir(path)

        def unlink(path, **kwargs):
            if path != Path("/root/dl/runtime/gpu_matmul.pid"):
                return original_unlink(path, **kwargs)
            return None

        monkeypatch.setattr(Path, "read_text", text)
        monkeypatch.setattr(Path, "read_bytes", data)
        monkeypatch.setattr(Path, "iterdir", directories)
        monkeypatch.setattr(Path, "unlink", unlink)
        monkeypatch.setattr(subprocess, "run", self.run)
        monkeypatch.setattr(subprocess, "check_output", self.check_output)
        monkeypatch.setattr(subprocess, "Popen", self.popen)
        monkeypatch.setattr(watch_wave7_release.time, "sleep", self.sleep)
        monkeypatch.setattr(watch_wave7_release.time, "monotonic", lambda: self.clock)

    def add_process(self, pid, arguments, ticks="123", status="S"):
        self.processes[pid] = (status, ticks, arguments)

    def add_training(self, pid=None, ticks="123", out=None):
        out = self.out if out is None else out
        self.add_process(
            self.pid if pid is None else pid,
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                "-m",
                "experiment.train",
                "--config",
                str(self.candidate / "experiment/configs" / f"{self.out.name}_0p5m_seed0.json"),
                "--out",
                str(out),
            ],
            ticks,
        )

    def saved(self):
        return json.loads(self.path.read_text())

    def sleep(self, seconds):
        self.clock += seconds
        if self.poll_effect is not None and seconds == 15:
            action, self.poll_effect = self.poll_effect, None
            action()
        if self.lock_effect is not None and seconds == 0.1:
            action, self.lock_effect = self.lock_effect, None
            action()

    def check_output(self, command, **kwargs):
        if command == ["hostname", "-I"]:
            return self.node
        if command == ["ps", "-eo", "args="]:
            return "\n".join(" ".join(fields[2]) for fields in self.processes.values() if fields[0] != "Z")
        if command[0] == "nvidia-smi":
            return self.cuda
        raise AssertionError(f"Unexpected external command: {command}")

    def run(self, command, **kwargs):
        if command[:2] == ["bash", KEEPER]:
            with (self.root / "run-preparation" / f".wave7-node-{self.node}.lock").open("a") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.starts.append(self.node)
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(f"Unexpected external command: {command}")

    def popen(self, command, **kwargs):
        self.spawns.append(command)
        if Path(command[1]).name == "wave7_release.py":
            controller_pid = 94001
            self.add_process(controller_pid, command, "1001")
            return RestoreChild(self, controller_pid, command, kwargs["stdout"])
        watcher_pid = 92001
        self.add_process(watcher_pid, command, "901")
        ticks = command[command.index("--start-ticks") + 1] if "--start-ticks" in command else None
        state = {
            "pid": watcher_pid,
            "watcher_start_ticks": "901",
            "training_pid": self.pid,
            "training_start_ticks": ticks,
            "root": str(self.root),
            "out": str(self.out),
            "status": self.spawn_status,
        }
        self.path.write_text(json.dumps(state))
        return subprocess.CompletedProcess(command, 0)


class RestoreChild:
    """Execute guarded restoration on join, or retain a controller/manager and node lock."""

    def __init__(self, boundary, pid, command, log):
        self.boundary = boundary
        self.pid = pid
        self.command = command
        self.log_path = Path(log.name)

    def wait(self, *, timeout):
        boundary = self.boundary
        boundary.restore_states.append(boundary.saved())
        if boundary.restore_timeout:
            boundary.pending_lock = (boundary.root / "run-preparation" / f".wave7-node-{boundary.node}.lock").open("a")
            fcntl.flock(boundary.pending_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            boundary.add_process(95001, ["bash", KEEPER, "start"], "1101")
            raise subprocess.TimeoutExpired(self.command, timeout)
        if boundary.restore_effect is not None:
            boundary.restore_effect()
        if boundary.restore_reply is not None:
            reply = boundary.restore_reply
        else:
            try:
                result = wave7_release.node_operation(
                    Path(self.command[self.command.index("--root") + 1]),
                    self.command[self.command.index("--phase") + 1],
                    lock_timeout=float(self.command[self.command.index("--node-lock-timeout") + 1]),
                )
                reply = subprocess.CompletedProcess(self.command, 0, json.dumps(result), "")
            except Exception as error:
                reply = subprocess.CompletedProcess(self.command, 1, "", str(error))
        self.log_path.write_text((reply.stdout or "") + (reply.stderr or ""))
        del boundary.processes[self.pid]
        return reply.returncode


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    boundary = WatcherIO(tmp_path, monkeypatch)
    try:
        yield boundary
    finally:
        if boundary.pending_lock is not None:
            boundary.pending_lock.close()


def test_live_watcher_persists_generation_and_restores_only_after_parent_exit(boundary):
    boundary.add_training()
    intermediate = []

    def parent_exit():
        intermediate.append(boundary.saved())
        del boundary.processes[boundary.pid]

    boundary.poll_effect = parent_exit
    watch_wave7_release.watch(boundary.pid, boundary.out, boundary.root, boundary.path, "123")
    assert intermediate[0]["status"] == "watching"
    assert intermediate[0]["training_start_ticks"] == "123"
    assert boundary.restore_states[0]["status"] == "restoring"
    assert boundary.saved()["status"] == "restored"
    assert type(boundary.saved()["keeper_restore"]) is int
    assert boundary.starts == [boundary.node]
    assert watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")["status"] == "restored"
    assert len(boundary.spawns) == 1


def test_exited_parent_restore_waits_for_discovery_node_lock_and_preserves_ticks(boundary):
    lock_path = boundary.root / "run-preparation" / f".wave7-node-{boundary.node}.lock"
    with lock_path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        delayed = []

        def release_discovery():
            delayed.append(boundary.saved())
            assert boundary.starts == []
            fcntl.flock(held, fcntl.LOCK_UN)

        boundary.lock_effect = release_discovery
        watch_wave7_release.watch(boundary.pid, boundary.out, boundary.root, boundary.path, "123")
    assert delayed[0]["status"] == "restoring"
    assert delayed[0]["training_start_ticks"] == "123"
    assert boundary.saved()["status"] == "restored"
    assert boundary.starts == [boundary.node]


@pytest.mark.parametrize("phase", ("watching", "restoring"))
def test_ensure_acknowledges_live_generation_and_is_idempotent_under_discovery_lock(boundary, phase):
    if phase == "watching":
        boundary.add_training()
    boundary.spawn_status = phase
    lock_path = boundary.root / "run-preparation" / f".wave7-node-{boundary.node}.lock"
    with lock_path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first = watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
        second = watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
    assert first == second
    assert first["training_start_ticks"] == "123"
    assert first["status"] == phase
    assert len(boundary.spawns) == 1
    assert boundary.starts == []


@pytest.mark.parametrize("foreign", ("training", "cuda"))
def test_exited_parent_foreign_workload_refuses_guarded_restore_and_keeps_failure(boundary, foreign):
    if foreign == "training":
        boundary.add_training(pid=93001, out=boundary.root / "other-run")
    else:
        boundary.cuda = "93001\n"
    with pytest.raises(RuntimeError):
        watch_wave7_release.watch(boundary.pid, boundary.out, boundary.root, boundary.path, "123")
    assert boundary.saved()["status"] == "needs_attention"
    assert boundary.saved()["training_start_ticks"] == "123"
    assert boundary.saved()["controller_exit_code"] != 0
    assert boundary.starts == []
    with pytest.raises(RuntimeError):
        watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
    assert len(boundary.spawns) == 1


@pytest.mark.parametrize("change", ("reused_pid", "changed_output"))
def test_parent_generation_change_refuses_restore_without_interrupting_process(boundary, change):
    boundary.add_training()

    def change_parent():
        boundary.add_training(
            ticks="124" if change == "reused_pid" else "123",
            out=boundary.root / "foreign-run" if change == "changed_output" else None,
        )

    boundary.poll_effect = change_parent
    with pytest.raises(RuntimeError):
        watch_wave7_release.watch(boundary.pid, boundary.out, boundary.root, boundary.path, "123")
    assert boundary.saved()["status"] == "needs_attention"
    assert boundary.pid in boundary.processes
    assert boundary.restore_states == []
    assert boundary.starts == []


@pytest.mark.parametrize(
    "reply", ({"keeper_restore": False}, {"keeper_restore": 1}, {"keeper_restore": 0, "node": "other"})
)
def test_invalid_controller_acknowledgment_is_not_promoted_to_restored(boundary, reply):
    payload = {"node": boundary.node, **reply}
    boundary.restore_reply = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
    with pytest.raises(RuntimeError):
        watch_wave7_release.watch(boundary.pid, boundary.out, boundary.root, boundary.path, "123")
    state = boundary.saved()
    assert state["status"] == "needs_attention"
    assert state["restore_acknowledgment"] == payload
    with pytest.raises(RuntimeError):
        watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
    assert len(boundary.spawns) == 1


def test_stopped_or_reused_watcher_is_not_reinstalled_over_saved_identity(boundary):
    boundary.add_training()
    receipt = watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
    boundary.add_process(receipt["pid"], boundary.spawns[0], ticks="902")
    with pytest.raises(RuntimeError):
        watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
    assert boundary.saved() == receipt
    assert len(boundary.spawns) == 1
    assert boundary.starts == []


def test_restored_receipt_rejects_reused_training_pid_and_changed_success_type(boundary):
    watch_wave7_release.watch(boundary.pid, boundary.out, boundary.root, boundary.path, "123")
    original = boundary.saved()
    boundary.add_training(ticks="124")
    with pytest.raises(RuntimeError):
        watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
    del boundary.processes[boundary.pid]
    invalid = {**original, "keeper_restore": False}
    boundary.path.write_text(json.dumps(invalid))
    with pytest.raises(RuntimeError):
        watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
    assert boundary.saved() == invalid
    assert len(boundary.spawns) == 1
    assert boundary.starts == [boundary.node]


def test_restoration_integrity_failure_preserves_candidate_independent_safety(boundary):
    (boundary.helpers / "launch_wave5.py").write_text("changed helper")
    with pytest.raises(RuntimeError):
        watch_wave7_release.watch(boundary.pid, boundary.out, boundary.root, boundary.path, "123")
    assert boundary.saved()["status"] == "needs_attention"
    assert boundary.starts == []


def test_restore_timeout_preserves_controller_manager_and_node_lock_without_retry(boundary):
    boundary.restore_timeout = True
    with pytest.raises(subprocess.TimeoutExpired):
        watch_wave7_release.watch(boundary.pid, boundary.out, boundary.root, boundary.path, "123")
    state = boundary.saved()
    pending = state["pending_restore_controller"]
    assert state["status"] == "needs_attention"
    assert pending["pid"] == state["controller_pid"]
    assert pending["start_ticks"] == boundary.processes[pending["pid"]][1]
    assert pending["command"] == boundary.processes[pending["pid"]][2]
    assert 95001 in boundary.processes
    assert "controller_exit_code" not in state
    assert "keeper_restore" not in state
    with (boundary.root / "run-preparation" / f".wave7-node-{boundary.node}.lock").open("a") as lock:
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(RuntimeError):
        watch_wave7_release.ensure(boundary.pid, boundary.out, boundary.root, "123")
    assert len(boundary.spawns) == 1
    assert boundary.saved() == state
    assert pending["pid"] in boundary.processes and 95001 in boundary.processes
