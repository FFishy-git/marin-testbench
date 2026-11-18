# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require actual failed-worker receipt shapes and exact generation-bound recovery allocation."""

import json
import os
import select
import signal
import subprocess
from contextlib import nullcontext
from pathlib import Path

import owned_output_handoff
import pytest
import selected_lr_failure_handoff as handoff


@pytest.fixture(autouse=True)
def isolated_manager(tmp_path, monkeypatch):
    source = tmp_path / "serialized-manager.sh"
    source.write_bytes(Path(handoff.KEEPER).read_bytes())
    monkeypatch.setattr(handoff, "KEEPER", str(source))
    monkeypatch.setattr(handoff, "MANAGER_SOURCE", source)
    monkeypatch.setattr(handoff, "MANAGER_LOCK", tmp_path / "gpu_resource_manager.lock")
    return source


def record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def keeper(pid: int) -> dict:
    return {
        "pid": pid,
        "generation": {
            str(worker): {
                "pid": worker,
                "status": "S",
                "parent": 1 if worker == pid else pid,
                "group": worker,
                "session": worker,
                "start_ticks": str(worker * 100),
                "command": ["keeper", str(worker)],
            }
            for worker in range(pid, pid + 9)
        },
        "devices": [[str(rank), f"GPU-{rank}", "A100-SXM4-40GB"] for rank in range(8)],
        "occupied": [[str(pid + rank + 1), f"GPU-{rank}"] for rank in range(8)],
    }


@pytest.fixture
def failed(tmp_path, monkeypatch):
    root = tmp_path / "root"
    release_path = root / "run-preparation/wave11-launch-binding.json"
    archive = root / "source.tar"
    archive.parent.mkdir()
    archive.write_bytes(b"preserved-original-archive")
    allocations = {node: {"keeper": keeper(1000 + index * 100)} for index, node in enumerate(handoff.NODES)}
    commands = {node: ["original-training", arm] for arm, node in zip(handoff.ARMS, handoff.NODES, strict=True)}
    record(
        release_path,
        {
            "root": str(root),
            "candidate": str(root / "failed-candidate"),
            "files_sha256": {},
            "helpers": str(root / "failed-helpers"),
            "helpers_sha256": {},
            "arms": handoff.ARMS,
            "nodes": handoff.NODES,
            "archive": str(archive),
            "archive_sha256": handoff.sha256(archive),
            "allocation_by_node": allocations,
            "commands": commands,
        },
    )
    digest = handoff.sha256(release_path)
    launch = {"status": "launched", "binding_sha256": digest, "workers": {}, "worker_markers": {}, "monitor_pid": 800}
    workers = []
    for index, (arm, node) in enumerate(zip(handoff.ARMS, handoff.NODES, strict=True)):
        accepted = allocations[node]["keeper"]
        path = root / f"run-preparation/wave11-workers/{arm}.json"
        child_log = root / f"runs/wave11/{arm}.torchrun.log"
        child_log.parent.mkdir(parents=True, exist_ok=True)
        child_log.write_text(f"RuntimeError: {handoff.FAILURE}\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".log").write_text("original worker failed and restored\n")
        state = {
            "status": "needs_attention",
            "arm": arm,
            "node": node,
            "out": str(root / f"runs/wave11/{arm}"),
            "release_binding_sha256": digest,
            "command": commands[node],
            "log": str(child_log),
            "log_sha256": handoff.sha256(child_log),
            "exit_code": 1,
            "cleanup_exit_code": 1,
            "keeper_restore": 0,
            "keeper_release_attempted": True,
            "child_marker": f"child-{index}",
            "worker_pid": 900 + index,
            "child_pid": 950 + index,
            "worker_start_ticks": str(index),
            "worker_marker": f"worker-{index}",
            "worker_command": ["original-worker", arm],
            "keeper": {
                "pid": accepted["pid"],
                "start_ticks": accepted["generation"][str(accepted["pid"])]["start_ticks"],
                "generation": [
                    [process["pid"], process["start_ticks"], process["group"]]
                    for process in accepted["generation"].values()
                ],
            },
            "restored_keeper": keeper(2000 + index * 100),
        }
        record(path, state)
        record(
            path.with_suffix(".start.json"),
            {
                "status": "acknowledged",
                "arm": arm,
                "node": node,
                "release_binding_sha256": digest,
                "pid": state["worker_pid"],
                "start_ticks": state["worker_start_ticks"],
                "worker_marker": state["worker_marker"],
                "command": state["worker_command"],
                "starter": {"pid": 850 + index},
            },
        )
        launch["workers"][arm] = {"original": True}
        launch["worker_markers"][arm] = state["worker_marker"]
        workers.append({"arm": arm, "status": "needs_attention", "receipt": state})
    record(root / "run-preparation/wave11-launch-state.json", launch)
    monitor = root / "run-preparation/monitor-wave11"
    record(
        monitor / "latest.json",
        {"status": "needs_attention", "all_workers_terminal": True, "binding_sha256": digest, "workers": workers},
    )
    (monitor / "stopped_at").write_text("2026-11-17T21:51:38+08:00\n")
    monkeypatch.setattr(handoff, "local_node", lambda: handoff.NODES[0])
    monkeypatch.setattr(handoff, "candidate_inventory", lambda candidate: {})
    states = {arm: handoff.read(root / f"run-preparation/wave11-workers/{arm}.json") for arm in handoff.ARMS}
    saved = states[handoff.ARMS[0]]["restored_keeper"]
    current = saved | {"generation": {int(pid): process for pid, process in saved["generation"].items()}}
    monkeypatch.setattr(handoff, "keeper_observation", lambda: current)
    monkeypatch.setattr(handoff, "observed", lambda pid: current["generation"].get(pid))
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(owned_output_handoff, "PROC_ROOT", proc)
    return root, states, current


@pytest.mark.parametrize("fault", ("none", "live_worker", "live_child", "changed_keeper", "escaped_marker"))
def test_handoff_preserves_failure_and_rejects_live_or_changed_allocation(failed, monkeypatch, fault):
    root, states, current = failed
    state = states[handoff.ARMS[0]]
    before = handoff.failure_evidence(root)
    marked = {int(pid): process for pid, process in current["generation"].items()}
    if fault in ("live_worker", "live_child"):
        pid = state["worker_pid"] if fault == "live_worker" else state["child_pid"]
        process = owned_output_handoff.PROC_ROOT / str(pid)
        process.mkdir()
        fields = ["0"] * 20
        fields[0:4], fields[19] = ["S", "1", str(pid), str(pid)], "original"
        (process / "stat").write_text(f"{pid} (worker) " + " ".join(fields))
        (process / "cmdline").write_bytes(b"original-worker\0")
    elif fault == "changed_keeper":
        monkeypatch.setattr(handoff, "keeper_observation", lambda: keeper(5000))
    elif fault == "escaped_marker":
        marked[7000] = keeper(7000)["generation"]["7000"]
    monkeypatch.setattr(
        handoff,
        "marker_processes",
        lambda name, value: marked if name == handoff.WORKER_MARKER else {},
    )
    if fault == "none":
        result = handoff.probe(root, handoff.ARMS[0])
        assert result["keeper"] == current
        assert result["failure_evidence"] == before
    else:
        with pytest.raises(RuntimeError):
            handoff.probe(root, handoff.ARMS[0])
    assert handoff.failure_evidence(root) == before
    assert not (root / "runs/wave12").exists()


@pytest.mark.parametrize("fault", ("output_created", "cleanup_changed", "rewritten_child_log", "snapshot_changed"))
def test_recovery_rejects_rewritten_or_nonzero_update_failure(failed, fault):
    root, states, _ = failed
    arm, state = handoff.ARMS[0], states[handoff.ARMS[0]]
    path = root / f"run-preparation/wave11-workers/{arm}.json"
    if fault == "output_created":
        Path(state["out"]).mkdir()
    elif fault == "cleanup_changed":
        record(path, state | {"cleanup_exit_code": 0})
    elif fault == "rewritten_child_log":
        Path(state["log"]).write_text("different original failure\n")
    else:
        snapshot = root / "run-preparation/monitor-wave11/latest.json"
        record(snapshot, handoff.read(snapshot) | {"all_workers_terminal": False})
    with pytest.raises(RuntimeError):
        handoff.failure_evidence(root)


def test_compact_keeper_requires_exact_released_generation():
    accepted = keeper(1000)
    compact = {
        "pid": 1000,
        "start_ticks": "100000",
        "generation": [
            [process["pid"], process["start_ticks"], process["group"]] for process in accepted["generation"].values()
        ],
    }
    assert handoff.released_keeper(compact, accepted)
    assert not handoff.released_keeper(compact | {"start_ticks": "reused"}, accepted)


@pytest.mark.parametrize("changed", (False, True))
def test_validation_retains_historical_keeper_and_requires_declared_succession(failed, monkeypatch, changed):
    root, _, current = failed
    binding_path, helpers, state_path, _, receipt_path = handoff.locations(root)
    record(binding_path, {"original": True})
    record(receipt_path, {"passed": True})
    original = handoff.read(root / "run-preparation/wave11-launch-binding.json")["allocation_by_node"][handoff.NODE][
        "keeper"
    ]
    if changed:
        original = keeper(8000)
    record(
        state_path,
        {
            "status": "passed",
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "keeper_restore": 0,
            "binding_sha256": handoff.sha256(binding_path),
            "validation_sha256": handoff.sha256(receipt_path),
            "controller_pid": 810,
            "controller_start_ticks": "original",
            "child_pid": 811,
            "child_marker": "validation-child",
            "restored_keeper": original,
        },
    )
    record(
        helpers.parent / "controller.json",
        {"status": "acknowledged", "pid": 810, "start_ticks": "original", "marker": "validation-controller"},
    )
    monkeypatch.setattr(
        handoff,
        "marker_processes",
        lambda name, value: current["generation"] if name in (handoff.WORKER_MARKER, handoff.CONTROLLER_MARKER) else {},
    )
    if changed:
        with pytest.raises(RuntimeError):
            handoff.validation_terminal(root)
        return
    result = handoff.validation_terminal(root)
    assert result["keeper"] == original
    assert result["current_keeper"] == current
    assert result["keeper"]["pid"] != result["current_keeper"]["pid"]


def test_installed_torchrun_creates_independent_rank_sessions(tmp_path):
    script = (
        "import json,os,sys\nfrom pathlib import Path\n"
        "r={'pid':os.getpid(),'parent':os.getppid(),'group':os.getpgrp(),"
        "'session':os.getsid(0),'rank':int(os.environ['RANK'])}\n"
        "Path(sys.argv[1],os.environ['RANK']+'.json').write_text(json.dumps(r))\n"
    )
    child = subprocess.Popen(
        [
            handoff.PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "--no-python",
            handoff.PYTHON,
            "-c",
            script,
            str(tmp_path),
        ],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, errors = child.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise
    assert child.returncode == 0, errors
    ranks = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    assert {rank["rank"] for rank in ranks} == {0, 1}
    assert all(rank["parent"] == child.pid for rank in ranks)
    assert all(rank["pid"] == rank["group"] == rank["session"] for rank in ranks)


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "before_release",
        "before_signal",
        "during_release",
        "between_check_and_signal",
        "reparented",
        "pointer",
        "occupancy",
    ),
)
def test_accepted_keeper_release_never_signals_a_replacement_or_claims_unverified_release(tmp_path, monkeypatch, fault):
    accepted = keeper(1000)
    current = {int(pid): value for pid, value in accepted["generation"].items()}
    keeper_file, state_path = tmp_path / "keeper.pid", tmp_path / "worker.json"
    keeper_file.write_text(str(accepted["pid"]) + "\n")
    monkeypatch.setattr(handoff, "KEEPER_PID", keeper_file)
    replacement = current[1001] | {"start_ticks": "replacement"}
    if fault == "before_release":
        changed = accepted | {"generation": accepted["generation"] | {"1001": replacement}}
        monkeypatch.setattr(handoff, "keeper_observation", lambda: changed)
    else:
        monkeypatch.setattr(handoff, "keeper_observation", lambda: accepted)
    if fault == "before_signal":
        current[1001] = replacement
    monkeypatch.setattr(handoff, "observed", current.get)
    handles, writes, signaled = {}, [], []

    def open_handle(pid):
        handle, writer = os.pipe()
        handles[handle] = (pid, dict(current[pid]), writer)
        writes.append(writer)
        return handle

    def stop(handle, signum):
        pid, identity, writer = handles[handle]
        if fault == "between_check_and_signal" and pid == 1001:
            current[pid] = replacement
        assert identity["start_ticks"] != "replacement"
        signaled.append(pid)
        if current.get(pid, {}).get("start_ticks") == identity["start_ticks"]:
            current.pop(pid)
        if fault == "during_release" and pid == 1001:
            current[pid] = replacement
        if fault == "reparented" and pid == 1001:
            current.pop(1000)
            for worker in current:
                current[worker] = current[worker] | {"parent": 1}
        if not current and fault == "pointer":
            keeper_file.write_text("different-owner\n")
        os.write(writer, b"exited")

    monkeypatch.setattr(handoff, "pidfd_open", open_handle)
    monkeypatch.setattr(handoff.signal, "pidfd_send_signal", stop)
    monkeypatch.setattr(
        handoff.subprocess, "check_output", lambda *args, **kwargs: "99999\n" if fault == "occupancy" else ""
    )
    state = {}
    with handoff.keeper_guard() as manager_lock:
        try:
            if fault not in ("none", "reparented"):
                with pytest.raises(RuntimeError):
                    handoff.release_accepted_keeper(accepted, state, state_path, manager_lock)
                assert state.get("keeper_release_complete") is not True
                assert keeper_file.exists()
                if fault in ("before_release", "before_signal"):
                    assert signaled == []
                    assert not state_path.exists()
                else:
                    assert handoff.read(state_path)["keeper_release_complete"] is False
                return
            result = handoff.release_accepted_keeper(accepted, state, state_path, manager_lock)
            assert signaled == [*range(1001, 1009), 1000]
            assert result["pid"] == 1000
            assert result["start_ticks"] == "100000"
            assert handoff.read(state_path)["keeper_release_complete"] is True
            assert not keeper_file.exists()
        finally:
            for writer in writes:
                os.close(writer)


def test_real_manager_lock_blocks_competing_start_and_supports_inherited_restore(tmp_path, isolated_manager):
    pointer = tmp_path / "gpu_matmul.pid"
    pointer.write_text("original-owner\n")
    environment = os.environ | {
        "BASH_ENV": "/dev/null",
        "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1",
        "GPU_KEEPER_ROOT_DIR": str(tmp_path),
        "GPU_RESOURCE_MANAGER_CONFIG": "/dev/null",
    }
    with handoff.keeper_guard() as handle:
        blocked = subprocess.run(
            ["bash", str(isolated_manager), "start"], env=environment, capture_output=True, text=True, timeout=10
        )
        assert blocked.returncode == 75
        assert pointer.read_text() == "original-owner\n"
        inherited = subprocess.run(
            ["bash", str(isolated_manager), "config"],
            env=environment | {"GPU_RESOURCE_MANAGER_LOCK_FD": str(handle)},
            pass_fds=(handle,),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert inherited.returncode == 0, inherited.stderr
    released = subprocess.run(
        ["bash", str(isolated_manager), "config"], env=environment, capture_output=True, text=True, timeout=10
    )
    assert released.returncode == 0, released.stderr
    assert pointer.read_text() == "original-owner\n"


def test_manager_rejects_descriptor_for_a_different_lock(tmp_path, isolated_manager):
    environment = os.environ | {
        "BASH_ENV": "/dev/null",
        "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1",
        "GPU_KEEPER_ROOT_DIR": str(tmp_path),
        "GPU_RESOURCE_MANAGER_CONFIG": "/dev/null",
    }
    (tmp_path / "gpu_resource_manager.lock").touch()
    with (tmp_path / "unrelated.lock").open("a") as unrelated:
        result = subprocess.run(
            ["bash", str(isolated_manager), "config"],
            env=environment | {"GPU_RESOURCE_MANAGER_LOCK_FD": str(unrelated.fileno())},
            pass_fds=(unrelated.fileno(),),
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert result.returncode == 2
    assert not (tmp_path / "gpu_matmul.pid").exists()


@pytest.mark.parametrize("inherited", (False, True))
def test_manager_started_keeper_preserves_session_and_releases_lock(tmp_path, isolated_manager, inherited):
    workload = tmp_path / "script/hy_infuse_moe/gpu_matmul.py"
    workload.parent.mkdir(parents=True)
    workload.write_text("# CPU-only fake workload\n")
    ready = tmp_path / "keeper-ready.json"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    torchrun = fakebin / "torchrun"
    torchrun.write_text(
        f"#!{handoff.PYTHON}\n"
        "import ctypes,json,os,signal\nfrom pathlib import Path\n"
        "assert ctypes.CDLL(None).prctl(15,b'python-fake',0,0,0)==0\n"
        "lock=Path(os.environ['FAKE_LOCK']).stat()\n"
        "held=[]\n"
        "for entry in Path('/proc/self/fd').iterdir():\n"
        "    try:\n"
        "        actual=entry.stat()\n"
        "    except FileNotFoundError:\n"
        "        continue\n"
        "    if (actual.st_dev,actual.st_ino)==(lock.st_dev,lock.st_ino):\n"
        "        held.append(entry.name)\n"
        "Path(os.environ['FAKE_READY']).write_text(json.dumps({\n"
        "    'pid':os.getpid(),'group':os.getpgrp(),'session':os.getsid(0),\n"
        "    'lock_descriptors':held,'lock_environment':os.environ.get('GPU_RESOURCE_MANAGER_LOCK_FD')\n"
        "}))\n"
        "signal.pause()\n"
    )
    torchrun.chmod(0o755)
    nvidia = fakebin / "nvidia-smi"
    nvidia.write_text('#!/bin/bash\nif [[ -f "$FAKE_READY" ]]; then printf "123\\n"; fi\n')
    nvidia.chmod(0o755)
    environment = os.environ | {
        "BASH_ENV": "/dev/null",
        "CUDA_VISIBLE_DEVICES": "",
        "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1",
        "GPU_KEEPER_ROOT_DIR": str(tmp_path),
        "GPU_RESOURCE_MANAGER_CONFIG": "/dev/null",
        "GPU_MATMUL_DEVICES": "0",
        "GPU_MATMUL_PROCESSES": "1",
        "FAKE_LOCK": str(handoff.MANAGER_LOCK),
        "FAKE_READY": str(ready),
        "PATH": str(fakebin) + ":" + os.environ["PATH"],
    }
    pid = None
    try:
        with handoff.keeper_guard() if inherited else nullcontext() as handle:
            startup = subprocess.run(
                ["bash", str(isolated_manager), "start"],
                env=environment | ({"GPU_RESOURCE_MANAGER_LOCK_FD": str(handle)} if inherited else {}),
                pass_fds=(handle,) if inherited else (),
                capture_output=True,
                text=True,
                timeout=30,
            )
            pointer = tmp_path / "gpu_matmul.pid"
            if pointer.exists():
                pid = int(pointer.read_text())
            assert startup.returncode == 0, startup.stderr
            state = json.loads(ready.read_text())
            assert state["pid"] == state["group"] == state["session"] == pid
            assert state["lock_descriptors"] == []
            assert state["lock_environment"] is None
            if inherited:
                blocked = subprocess.run(
                    ["bash", str(isolated_manager), "config"],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                assert blocked.returncode == 75
        available = subprocess.run(
            ["bash", str(isolated_manager), "config"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert available.returncode == 0, available.stderr
        os.kill(pid, 0)
    finally:
        pointer = tmp_path / "gpu_matmul.pid"
        if pid is None and pointer.exists():
            pid = int(pointer.read_text())
        if pid is not None:
            handle = handoff.pidfd_open(pid)
            try:
                signal.pidfd_send_signal(handle, signal.SIGTERM)
                poll = select.poll()
                poll.register(handle, select.POLLIN)
                assert poll.poll(10000) == [(handle, select.POLLIN)]
            finally:
                os.close(handle)


def test_real_pidfd_signals_only_its_original_process_and_reports_exit():
    child = subprocess.Popen([handoff.PYTHON, "-c", "import signal; signal.pause()"], start_new_session=True)
    handle = handoff.pidfd_open(child.pid)
    try:
        signal.pidfd_send_signal(handle, signal.SIGTERM)
        assert child.wait(timeout=10) == -signal.SIGTERM
        poll = select.poll()
        poll.register(handle, select.POLLIN)
        assert poll.poll(0) == [(handle, select.POLLIN)]
        with pytest.raises(ProcessLookupError):
            signal.pidfd_send_signal(handle, signal.SIGTERM)
    finally:
        os.close(handle)
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
