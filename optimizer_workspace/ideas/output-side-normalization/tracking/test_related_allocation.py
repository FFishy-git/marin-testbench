# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only ownership refusal, dynamic descendant cleanup, and signal deferral."""

import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import owned_output_handoff as handoff
import pytest
import related_allocation as allocation

CPU_CHILD = """
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

mode, proc = sys.argv[1], Path(sys.argv[2])
worker = None
stopping = False

def event(name, **fields):
    print(json.dumps({"event": name, "pid": os.getpid(), **fields}), flush=True)

def finish(signum, frame):
    event("terminated", signal=signum)
    raise SystemExit(0)

def spawn_after_term(signum, frame):
    global worker, stopping
    if stopping:
        return
    stopping = True
    worker = subprocess.Popen(
        [sys.executable, __file__, "worker", str(proc)],
        start_new_session=True,
    )
    event("descendant_started", child_pid=worker.pid)
    try:
        status = worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait(timeout=5)
        event("descendant_forced_exit")
        raise SystemExit(7)
    event("descendant_reaped", child_pid=worker.pid, returncode=status)
    raise SystemExit(status)

signal.signal(signal.SIGTERM, spawn_after_term if mode == "launcher" else finish)
signal.signal(signal.SIGALRM, lambda *_: sys.exit(8))
signal.alarm(20)
(proc / str(os.getpid())).symlink_to(Path("/proc") / str(os.getpid()))
event("ready", mode=mode)
while True:
    signal.pause()
"""

DEFERRED_CHILD = """
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
import related_allocation as allocation

acknowledgment = Path(sys.argv[2])
spawn = subprocess.Popen
child = None

def interrupted_spawn(*args, **kwargs):
    process = spawn(*args, **kwargs)
    os.kill(os.getpid(), signal.SIGTERM)
    os.kill(os.getpid(), signal.SIGINT)
    return process

signal.signal(signal.SIGTERM, allocation.interrupt)
signal.signal(signal.SIGINT, allocation.interrupt)
try:
    with allocation.defer_signals() as pending:
        with patch.object(subprocess, "Popen", interrupted_spawn):
            child = subprocess.Popen(
                [sys.executable, "-c", "import signal; signal.pause()"],
                start_new_session=True,
            )
        acknowledgment.write_text(json.dumps({"child_pid": child.pid}))
    child.terminate()
    child.wait(timeout=5)
    restored = False
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except KeyboardInterrupt:
        restored = True
    print(json.dumps({
        "pending": pending,
        "restored_handler_interrupted": restored,
        "child_pid": child.pid,
        "child_returncode": child.returncode,
    }), flush=True)
finally:
    if child is not None and child.poll() is None:
        child.kill()
        child.wait(timeout=5)
"""

ORPHAN_CHILD = """
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

mode, proc, evidence, marker_key = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]

def finish(signum, frame):
    (evidence / f"{mode}_exit.json").write_text(json.dumps({
        "pid": os.getpid(),
        "parent": os.getppid(),
        "marker": os.environ.get(marker_key),
        "signal": signum,
    }))
    os._exit(0)

signal.signal(signal.SIGTERM, finish)
signal.signal(signal.SIGALRM, lambda *_: os._exit(8))
signal.alarm(20)
(proc / str(os.getpid())).symlink_to(Path("/proc") / str(os.getpid()))
child_pid = None
if mode == "parent":
    child = subprocess.Popen(
        [sys.executable, __file__, "grandchild", str(proc), str(evidence), marker_key],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_ready = json.loads(child.stdout.readline())
    assert child_ready["pid"] == child.pid
    child.stdout.close()
    child_pid = child.pid
print(json.dumps({"event": "ready", "pid": os.getpid(), "child_pid": child_pid}), flush=True)
while True:
    signal.pause()
"""

ORPHAN_DRIVER = """
import ctypes
import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, sys.argv[1])
import owned_output_handoff as handoff
import related_allocation as allocation

program, proc, evidence = map(Path, sys.argv[2:])
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(36, 1, 0, 0, 0) == 0
marker = uuid4().hex
environment = {key: value for key, value in os.environ.items() if key != allocation.MARKER}
environment["CUDA_VISIBLE_DEVICES"] = ""
sentinel = parent = None
grandchild_pid = None
signal_events = []
actual_send = signal.pidfd_send_signal

def spawn(mode, env):
    return subprocess.Popen(
        [sys.executable, str(program), mode, str(proc), str(evidence), allocation.MARKER],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

def acknowledgment(process):
    readable, _, _ = select.select([process.stdout], [], [], 5)
    assert readable, "CPU child did not acknowledge startup"
    record = json.loads(process.stdout.readline())
    assert record["event"] == "ready" and record["pid"] == process.pid
    return record

def send_after_parent_exit(handle, signum, *args, **kwargs):
    fields = dict(
        line.split(":", 1) for line in (Path("/proc/self/fdinfo") / str(handle)).read_text().splitlines()
    )
    pid = int(fields["Pid"])
    actual_send(handle, signum, *args, **kwargs)
    signal_events.append({"pid": pid, "signal": signum})
    if pid == parent.pid:
        assert parent.wait(timeout=5) == 0
        reparented = handoff.observed(grandchild_pid)
        assert reparented["parent"] == os.getpid()
        assert reparented["status"] not in handoff.EXIT_STATES
        (evidence / "reparented.json").write_text(json.dumps(reparented))

def finish(process):
    if process is not None and process.poll() is None:
        process.kill()
        process.wait(timeout=5)
    if process is not None:
        process.communicate(timeout=5)

try:
    sentinel = spawn("sentinel", environment)
    acknowledgment(sentinel)
    parent = spawn("parent", environment | {allocation.MARKER: marker})
    grandchild_pid = acknowledgment(parent)["child_pid"]
    with patch.object(handoff, "PROC_ROOT", proc):
        initial = handoff.marker_processes(allocation.MARKER, marker)
        assert list(initial) == [parent.pid, grandchild_pid]
        assert initial[grandchild_pid]["parent"] == parent.pid
        with patch.object(signal, "pidfd_send_signal", send_after_parent_exit):
            allocation.cleanup(parent, marker)
        assert handoff.marker_processes(allocation.MARKER, marker) == {}
        assert sentinel.poll() is None
    waited_pid, status = os.waitpid(grandchild_pid, os.WNOHANG)
    assert waited_pid == grandchild_pid and os.waitstatus_to_exitcode(status) == 0
    assert not (Path("/proc") / str(grandchild_pid)).exists()
    assert not (Path("/proc") / str(parent.pid)).exists()
    initial_grandchild = initial[grandchild_pid]
    reparented = json.loads((evidence / "reparented.json").read_text())
    assert reparented["parent"] != initial_grandchild["parent"]
    for key in ("pid", "start_ticks", "session", "group", "command"):
        assert reparented[key] == initial_grandchild[key]
    print(json.dumps({
        "parent_pid": parent.pid,
        "grandchild_pid": grandchild_pid,
        "sentinel_pid": sentinel.pid,
        "marker": marker,
        "signals": signal_events,
        "sentinel_alive": sentinel.poll() is None,
    }), flush=True)
finally:
    finish(parent)
    if grandchild_pid is not None:
        try:
            handle = allocation.pidfd(grandchild_pid)
        except ProcessLookupError:
            pass
        else:
            try:
                actual_send(handle, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(handle)
        try:
            os.waitpid(grandchild_pid, 0)
        except ChildProcessError:
            pass
    finish(sentinel)
"""


def fake_process(proc, pid, command):
    directory = proc / str(pid)
    directory.mkdir()
    fields = ["0"] * 21
    fields[:4] = ["S", "1", str(pid), str(pid)]
    fields[19:21] = [str(pid), "4096"]
    (directory / "stat").write_text(f"{pid} (cpu fixture) {' '.join(fields)}")
    (directory / "cmdline").write_bytes(("\0".join(command) + "\0").encode())
    (directory / "environ").write_bytes(b"")


def ready(process):
    readable, _, _ = select.select([process.stdout], [], [], 5)
    assert readable, "CPU child did not acknowledge startup"
    event = json.loads(process.stdout.readline())
    assert event["event"] == "ready"
    assert event["pid"] == process.pid


def finish_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_unknown_keeper_refuses_restoration_without_touching_processes(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    keeper_pid, unrelated_pid = os.getpid() + 100000, os.getpid() + 100001
    fake_process(proc, keeper_pid, [sys.executable, handoff.WORKLOAD])
    fake_process(proc, unrelated_pid, [sys.executable, "/cpu/unrelated.py"])
    before = {path.relative_to(proc): path.read_bytes() for path in proc.rglob("*") if path.is_file()}
    with patch.object(handoff, "PROC_ROOT", proc):
        allocation.require_no_training()
        with pytest.raises(RuntimeError, match="Unknown keeper process prevents restoration"):
            allocation.require_no_training(keeper_absent=True)
    after = {path.relative_to(proc): path.read_bytes() for path in proc.rglob("*") if path.is_file()}
    assert after == before


def test_cleanup_reaps_worker_born_after_term_and_preserves_other_marker(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    program = tmp_path / "cpu_child.py"
    program.write_text(CPU_CHILD)
    marker = uuid4().hex
    sentinel = subprocess.Popen(
        [sys.executable, str(program), "sentinel", str(proc)],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", allocation.MARKER: uuid4().hex},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    launcher = None
    try:
        ready(sentinel)
        launcher = subprocess.Popen(
            [sys.executable, str(program), "launcher", str(proc)],
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", allocation.MARKER: marker},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        ready(launcher)
        with patch.object(handoff, "PROC_ROOT", proc):
            initial = handoff.marker_processes(allocation.MARKER, marker)
            assert set(initial) == {launcher.pid}
            allocation.cleanup(launcher, marker)
            assert handoff.marker_processes(allocation.MARKER, marker) == {}
        stdout, stderr = launcher.communicate(timeout=5)
        events = [json.loads(line) for line in stdout.splitlines()]
        started = next(event for event in events if event["event"] == "descendant_started")
        ended = next(event for event in events if event["event"] == "descendant_reaped")
        worker_pid = started["child_pid"]
        assert worker_pid not in initial
        assert ended["child_pid"] == worker_pid
        assert ended["returncode"] == 0
        assert any(event["event"] == "terminated" and event["pid"] == worker_pid for event in events)
        assert launcher.returncode == 0, stderr
        assert not (Path("/proc") / str(worker_pid)).exists()
        assert not (Path("/proc") / str(launcher.pid)).exists()
        assert sentinel.poll() is None
    finally:
        if launcher is not None:
            finish_owned(launcher)
            launcher.communicate(timeout=5)
        finish_owned(sentinel)
        sentinel.communicate(timeout=5)


def test_cleanup_drains_reparented_grandchild_after_parent_term_and_preserves_unmarked_sentinel(tmp_path):
    proc, evidence = tmp_path / "proc", tmp_path / "evidence"
    proc.mkdir()
    evidence.mkdir()
    child_program, driver_program = tmp_path / "orphan_child.py", tmp_path / "orphan_driver.py"
    child_program.write_text(ORPHAN_CHILD)
    driver_program.write_text(ORPHAN_DRIVER)
    result = subprocess.run(
        [
            sys.executable,
            str(driver_program),
            str(Path(allocation.__file__).parent),
            str(child_program),
            str(proc),
            str(evidence),
        ],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["sentinel_alive"]
    assert record["signals"] == [
        {"pid": record["parent_pid"], "signal": signal.SIGTERM},
        {"pid": record["grandchild_pid"], "signal": signal.SIGTERM},
    ]
    parent_exit = json.loads((evidence / "parent_exit.json").read_text())
    grandchild_exit = json.loads((evidence / "grandchild_exit.json").read_text())
    assert parent_exit["marker"] == grandchild_exit["marker"] == record["marker"]
    assert parent_exit["pid"] == record["parent_pid"]
    assert grandchild_exit["pid"] == record["grandchild_pid"]
    assert grandchild_exit["parent"] != record["parent_pid"]
    assert not (Path("/proc") / str(record["parent_pid"])).exists()
    assert not (Path("/proc") / str(record["grandchild_pid"])).exists()
    assert not (Path("/proc") / str(record["sentinel_pid"])).exists()


def test_startup_signals_defer_until_child_acknowledgment_and_restore_handler(tmp_path):
    program = tmp_path / "deferred_child.py"
    program.write_text(DEFERRED_CHILD)
    acknowledgment = tmp_path / "acknowledgment.json"
    result = subprocess.run(
        [sys.executable, str(program), str(Path(allocation.__file__).parent), str(acknowledgment)],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["pending"] == [signal.SIGTERM, signal.SIGINT]
    assert evidence["restored_handler_interrupted"]
    assert json.loads(acknowledgment.read_text())["child_pid"] == evidence["child_pid"]
    assert evidence["child_returncode"] == -signal.SIGTERM
    assert not (Path("/proc") / str(evidence["child_pid"])).exists()
