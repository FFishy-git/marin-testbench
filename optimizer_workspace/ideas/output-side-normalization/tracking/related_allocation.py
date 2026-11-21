# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own a related-baseline command and restore the exact eight-GPU allocation."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import select
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from owned_output_handoff import (
    KEEPER_PID,
    WORKLOAD,
    keeper_observation,
    live_processes,
    marker_processes,
    observed,
    same_generation,
)

TASK = Path(__file__).resolve().parents[1]
ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
MANAGER = Path("/root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh")
LOCK = KEEPER_PID.with_name("gpu_resource_manager.lock")
MARKER = "OUTPUT_MARIN_RELATED_CHILD"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")


def local_node():
    addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
    return next(node for node in NODES if node in addresses)


@contextmanager
def defer_signals():
    pending = []

    def deferred(signum, _frame):
        pending.append(signum)

    handlers = {signum: signal.signal(signum, deferred) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield pending
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def write(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pidfd(pid):
    if os.uname().sysname != "Linux" or os.uname().machine != "x86_64":
        raise RuntimeError("Allocation release requires the validated Linux x86-64 pidfd ABI")
    handle = LIBC.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if handle < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return handle


def release(accepted, state, path):
    """Bind pidfds before signaling only the observed keeper launcher and ranks."""
    generation = {int(pid): item for pid, item in accepted["generation"].items()}
    with ExitStack() as stack:
        handles = {}
        for pid, expected in generation.items():
            handle = pidfd(pid)
            stack.callback(os.close, handle)
            if not same_generation(expected, observed(pid)):
                raise RuntimeError("Keeper generation changed before release")
            handles[pid] = handle
        current = keeper_observation()
        if current["pid"] != accepted["pid"] or current["devices"] != accepted["devices"]:
            raise RuntimeError("Keeper allocation changed before release")
        if any(not same_generation(item, current["generation"].get(pid)) for pid, item in generation.items()):
            raise RuntimeError("Keeper ranks changed before release")
        state.update(keeper_release_attempted=True)
        write(path, state)
        for pid in [p for p in generation if p != accepted["pid"]] + [accepted["pid"]]:
            try:
                signal.pidfd_send_signal(handles[pid], signal.SIGTERM)
            except ProcessLookupError:
                pass
        poll = select.poll()
        for handle in handles.values():
            poll.register(handle, select.POLLIN)
        pending = set(handles.values())
        deadline = time.monotonic() + 45
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Accepted keeper did not exit")
            for handle, events in poll.poll(max(1, int(remaining * 1000))):
                if events & select.POLLIN:
                    pending.discard(handle)
                    poll.unregister(handle)
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("CUDA occupancy remains after keeper release")
    if KEEPER_PID.read_text().strip() != str(accepted["pid"]):
        raise RuntimeError("Keeper pointer changed during release")
    KEEPER_PID.unlink()
    state.update(keeper_release_complete=True)
    write(path, state)


def cleanup(child, marker):
    """Reap the owned launcher and any marked ranks without signaling other jobs."""
    for stop_signal, duration in ((signal.SIGTERM, 45), (signal.SIGKILL, 15)):
        deadline = time.monotonic() + duration
        while True:
            child.poll()
            processes = marker_processes(MARKER, marker, allow_reparent=True)
            if not processes and child.poll() is not None:
                return
            for pid, expected in processes.items():
                try:
                    handle = pidfd(pid)
                except ProcessLookupError:
                    continue
                try:
                    current = observed(pid, allow_reparent=True)
                    if current is None or current["status"] in ("Z", "X", "x"):
                        continue
                    if not same_generation(expected, current, allow_reparent=True):
                        raise RuntimeError("Owned child changed during cleanup")
                    try:
                        signal.pidfd_send_signal(handle, stop_signal)
                    except ProcessLookupError:
                        pass
                finally:
                    os.close(handle)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    raise RuntimeError("Owned descendants remain after bounded cleanup")


def require_no_training(*, keeper_absent=False):
    for pid, process in live_processes().items():
        if pid == os.getpid():
            continue
        command = process["command"]
        if "torch.distributed.run" in command or "experiment.train" in command:
            raise RuntimeError("Existing training or unacknowledged launcher prevents allocation transfer")
        if keeper_absent and WORKLOAD in command:
            raise RuntimeError("Unknown keeper process prevents restoration")


def interrupt(signum, _frame):
    raise KeyboardInterrupt(f"Allocation owner received signal {signum}")


def run_owned(command, state_path, log_path):
    """Execute under the node lock, preserving failure evidence and restoration."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists() or log_path.exists():
        raise FileExistsError("Allocation attempt already has evidence")
    state = {
        "time": now(),
        "status": "preflight",
        "pid": os.getpid(),
        "generation": observed(os.getpid()),
        "command": command,
        "log": str(log_path),
        "task": str(TASK),
        "source_sha256": {str(p.relative_to(TASK)): digest(p) for p in sorted((TASK / "experiment").rglob("*.py"))},
        "manager_sha256": digest(MANAGER),
    }
    write(state_path, state)
    marker, child = uuid4().hex, None
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    node = local_node()
    node_lock_path = ROOT / "run-preparation" / f".wave7-node-{node}.lock"
    with node_lock_path.open("a") as node_lock, LOCK.open("a") as lock:
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_no_training()
        accepted = keeper_observation()
        state["keeper"] = accepted
        write(state_path, state)
        try:
            with defer_signals() as pending:
                release(accepted, state, state_path)
            if pending:
                raise KeyboardInterrupt("Interrupted during allocation release")
            environment = os.environ | ENVIRONMENT | {MARKER: marker}
            with log_path.open("x") as stream:
                with defer_signals() as pending:
                    child = subprocess.Popen(
                        command,
                        cwd=TASK,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    state.update(
                        status="running",
                        child_pid=child.pid,
                        child_marker=marker,
                        child_generation=observed(child.pid),
                    )
                    write(state_path, state)
                if pending:
                    raise KeyboardInterrupt("Interrupted during owned child startup")
                state["exit_code"] = child.wait()
                if state["exit_code"]:
                    raise RuntimeError(f"Command exited {state['exit_code']}; inspect {log_path}")
                state["status"] = "command_complete"
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            with defer_signals():
                try:
                    if child is not None:
                        cleanup(child, marker)
                        state["cleanup_exit_code"] = child.returncode
                    if state.get("keeper_release_attempted"):
                        if digest(MANAGER) != state["manager_sha256"]:
                            raise RuntimeError("Keeper manager changed; restoration requires review")
                        for pid in accepted["generation"]:
                            process = observed(int(pid))
                            if process is not None and process["status"] not in ("Z", "X", "x"):
                                raise RuntimeError("Accepted keeper release is incomplete; restoration refused")
                        require_no_training(keeper_absent=True)
                        if subprocess.check_output(
                            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                        ).strip():
                            raise RuntimeError("CUDA occupancy prevents keeper restoration")
                        if KEEPER_PID.exists():
                            if KEEPER_PID.read_text().strip() != str(accepted["pid"]):
                                raise RuntimeError("Keeper pointer changed; restoration refused")
                            KEEPER_PID.unlink()
                        environment = {key: value for key, value in os.environ.items() if key != MARKER}
                        environment.update(
                            GPU_RESOURCE_MANAGER_LOCAL_ONLY="1", GPU_RESOURCE_MANAGER_LOCK_FD=str(lock.fileno())
                        )
                        subprocess.run(
                            ["bash", str(MANAGER), "start"],
                            env=environment,
                            pass_fds=(lock.fileno(),),
                            check=True,
                            timeout=120,
                        )
                        restored = keeper_observation()
                        if restored["devices"] != accepted["devices"]:
                            raise RuntimeError("Restored keeper occupies a different allocation")
                        state.update(keeper_restore=0, restored_keeper=restored)
                        if state["status"] == "command_complete":
                            state["status"] = "complete"
                except BaseException as error:
                    state.update(status="needs_attention", restoration_error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    state.update(time=now())
                    if log_path.exists():
                        state["log_sha256"] = digest(log_path)
                    write(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    run_owned(command, args.state.resolve(), args.log.resolve())


if __name__ == "__main__":
    main()
