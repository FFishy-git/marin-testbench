# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained node-level GPU keeper release, owned execution, and guaranteed keeper restoration."""

from __future__ import annotations

import argparse
import csv
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

TASK = Path(__file__).resolve().parents[1]
PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
KEEPER_PYTHON = "/opt/conda/envs/torch-base/bin/python3.13"
KEEPER_PID = Path("/root/dl/runtime/gpu_matmul.pid")
MANAGER = Path("/root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh")
WORKLOAD = "/root/dl/runtime/script/hy_infuse_moe/gpu_matmul.py"
TORCHRUN = "/opt/conda/envs/torch-base/bin/torchrun"
WORKLOAD_ARGS = ["--size", "8192", "--dtype", "fp16", "--pause", "0"]
LOCK = KEEPER_PID.with_name("gpu_resource_manager.lock")
PROC_ROOT = Path("/proc")
MARKER = "OUTPUT_MARIN_CONFIRM_CHILD"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
EXIT_STATES = ("Z", "X", "x")
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


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def local_node() -> str:
    addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
    return next(node for node in NODES if node in addresses)


def observed(pid: int, *, allow_reparent: bool = False) -> dict | None:
    directory = PROC_ROOT / str(pid)
    try:
        before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        command = [arg.decode() for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
        after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    ownership = slice(2, 4) if allow_reparent else slice(1, 4)
    if before[19] != after[19] or before[ownership] != after[ownership]:
        return None
    return {
        "pid": pid,
        "status": after[0],
        "parent": int(after[1]),
        "group": int(after[2]),
        "session": int(after[3]),
        "start_ticks": after[19],
        "command": command,
    }


def live_processes(*, allow_reparent: bool = False) -> dict[int, dict]:
    result = {}
    for path in PROC_ROOT.iterdir():
        if path.name.isdigit():
            proc = observed(int(path.name), allow_reparent=allow_reparent)
            if proc is not None and proc["status"] != "Z":
                result[proc["pid"]] = proc
    return result


def marker_processes(name: str, value: str) -> dict[int, dict]:
    token = f"{name}={value}".encode()
    result = {}
    for pid, proc in live_processes(allow_reparent=True).items():
        if proc["status"] in EXIT_STATES:
            continue
        try:
            tokens = (PROC_ROOT / str(pid) / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if token in tokens:
            result[pid] = proc
    return result


def pidfd(pid: int) -> int:
    handle = LIBC.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if handle < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return handle


@contextmanager
def defer_signals():
    pending = []

    def handler(signum, _frame):
        pending.append(signum)

    old = {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield pending
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def require_no_training() -> None:
    for pid, proc in live_processes().items():
        if pid == os.getpid():
            continue
        argv = proc["command"]
        if not argv:
            continue
        exe = Path(argv[0]).name
        if not (exe.startswith("python") or exe == "torchrun"):
            continue
        cmd = " ".join(argv)
        if "allocation.py" in cmd or "run_campaign.py" in cmd or "run_lr_campaign.py" in cmd:
            continue
        if "experiment.train" in cmd or "torch.distributed.run" in cmd:
            raise RuntimeError(f"Active training process found (pid={pid}): {cmd[:120]}")


def release_keeper_if_present() -> bool:
    """Release the GPU keeper if running; return True if a keeper was released."""
    if not KEEPER_PID.exists():
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if occupied:
            raise RuntimeError(f"GPUs occupied without keeper pid file: {occupied}")
        return False

    pid = int(KEEPER_PID.read_text().strip())
    procs = live_processes()
    generation = {p: info for p, info in procs.items() if p == pid or info["parent"] == pid}
    if not generation:
        KEEPER_PID.unlink(missing_ok=True)
        return False

    with ExitStack() as stack:
        handles = {}
        for p in generation:
            try:
                h = pidfd(p)
                stack.callback(os.close, h)
                handles[p] = h
            except ProcessLookupError:
                pass
        for p in [k for k in handles if k != pid] + ([pid] if pid in handles else []):
            try:
                signal.pidfd_send_signal(handles[p], signal.SIGTERM)
            except ProcessLookupError:
                pass
        poll = select.poll()
        for h in handles.values():
            poll.register(h, select.POLLIN)
        pending = set(handles.values())
        deadline = time.monotonic() + 45
        while pending:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise RuntimeError("Keeper did not exit within 45s")
            for h, ev in poll.poll(max(1, int(rem * 1000))):
                if ev & select.POLLIN:
                    pending.discard(h)
                    poll.unregister(h)

    time.sleep(1.0)
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if occupied:
        raise RuntimeError(f"CUDA occupancy remains after keeper release: {occupied}")
    KEEPER_PID.unlink(missing_ok=True)
    return True


def restore_keeper(lock_fd: int) -> None:
    deadline = time.monotonic() + 20.0
    occupied = ""
    while True:
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if not occupied or time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    if occupied:
        raise RuntimeError(f"CUDA occupancy prevents keeper restoration: {occupied}")
    if KEEPER_PID.exists():
        KEEPER_PID.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k != MARKER}
    env.update(GPU_RESOURCE_MANAGER_LOCAL_ONLY="1", GPU_RESOURCE_MANAGER_LOCK_FD=str(lock_fd))
    subprocess.run(["bash", str(MANAGER), "start"], env=env, pass_fds=(lock_fd,), check=True, timeout=120)


def cleanup_marked(child: subprocess.Popen, marker: str) -> None:
    for sig, dur in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        deadline = time.monotonic() + dur
        while True:
            child.poll()
            procs = marker_processes(MARKER, marker)
            if not procs and child.poll() is not None:
                return
            for pid in list(procs):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.3)


def run_owned(command: list[str], state_path: Path, log_path: Path) -> None:
    state = {"time": now(), "status": "preflight", "pid": os.getpid(), "command": command, "log": str(log_path)}
    write_json(state_path, state)
    marker, child = uuid4().hex, None
    node = local_node()
    node_lock_path = PVC_ROOT / "run-preparation" / f".wave7-node-{node}.lock"
    node_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with node_lock_path.open("a") as node_lock, LOCK.open("a") as lock:
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_no_training()
        keeper_released = False
        try:
            with defer_signals():
                keeper_released = release_keeper_if_present()
            state.update(status="running", keeper_released=keeper_released, child_marker=marker, start_time=now())
            write_json(state_path, state)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ | ENVIRONMENT | {MARKER: marker}
            with log_path.open("w") as stream:
                child = subprocess.Popen(
                    command,
                    cwd=TASK,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                state["child_pid"] = child.pid
                write_json(state_path, state)
                rc = child.wait()
            state["exit_code"] = rc
            if rc != 0:
                raise RuntimeError(f"Command exited {rc}; see {log_path}")
            state["status"] = "command_complete"
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with defer_signals():
                try:
                    if child is not None:
                        cleanup_marked(child, marker)
                    restore_keeper(lock.fileno())
                    state["keeper_restored"] = True
                    if state["status"] == "command_complete":
                        state["status"] = "complete"
                except BaseException as exc:
                    state.update(status="needs_attention", restoration_error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    state["end_time"] = now()
                    write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        parser.error("Provide a command after --")
    run_owned(cmd, args.state.resolve(), args.log.resolve())


if __name__ == "__main__":
    main()
