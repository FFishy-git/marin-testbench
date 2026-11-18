# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Idempotently watch one exact wave-5 training process and restore its keeper."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

KEEPER = "/root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh"
WORKLOAD = b"/root/dl/runtime/script/hy_infuse_moe/gpu_matmul.py"


def process(pid: int) -> tuple[str, str, str] | None:
    directory = Path("/proc") / str(pid)
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        arguments = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return None
    return fields[0], fields[19], arguments


def write(path: Path, state: dict) -> None:
    state["time"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def watch(pid: int, out: Path, state_path: Path) -> None:
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = process(pid)
        state = {"status": "starting", "pid": os.getpid(), "training_pid": pid, "out": str(out)}
        if observed and observed[0] != "Z":
            if "-m torch.distributed.run" not in observed[2] or str(out) not in observed[2]:
                state["status"] = "needs_attention"
                write(state_path, state)
                raise RuntimeError("Training process identity changed")
            state["status"] = "watching"
            state["training_start_ticks"] = observed[1]
            write(state_path, state)
            while observed and observed[0] != "Z":
                if observed[1] != state["training_start_ticks"]:
                    state["status"] = "needs_attention"
                    write(state_path, state)
                    raise RuntimeError("Training PID reused; keeper restoration refused")
                time.sleep(15)
                observed = process(pid)
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        )
        if occupied.strip():
            state["status"] = "needs_attention"
            write(state_path, state)
            raise RuntimeError("CUDA occupancy remains after training; keeper restoration refused")
        for directory in Path("/proc").iterdir():
            if directory.name.isdigit():
                try:
                    if WORKLOAD in (directory / "cmdline").read_bytes().split(b"\0"):
                        raise RuntimeError("Existing keeper-path process prevents broad startup cleanup")
                except FileNotFoundError:
                    continue
        Path("/root/dl/runtime/gpu_matmul.pid").unlink(missing_ok=True)
        restored = subprocess.run(["bash", KEEPER, "start"], env={**os.environ, "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1"})
        state.update(
            status="restored" if restored.returncode == 0 else "needs_attention", keeper_restore=restored.returncode
        )
        write(state_path, state)


def ensure(pid: int, out: Path) -> dict:
    path = out.parent / f"{out.name}.watcher.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["training_pid"] != pid or state["out"] != str(out):
            raise RuntimeError("Existing watcher belongs to a different training identity")
        observed = process(state["pid"])
        if state["status"] == "restored":
            return state
        if state["status"] != "watching":
            raise RuntimeError("Existing watcher failed its recorded identity/restoration gate")
        if observed and observed[0] != "Z" and str(path) in observed[2] and str(Path(__file__)) in observed[2]:
            return state
        raise RuntimeError("Existing watcher stopped; preserve evidence and reconcile before restarting")
    with (out.parent / f"{out.name}.watcher.log").open("a") as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--pid", str(pid), "--out", str(out), "--run", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if path.exists():
            state = json.loads(path.read_text())
            if state["status"] in ("watching", "restored"):
                return state
            if state["status"] == "needs_attention":
                raise RuntimeError("Watcher failed its startup identity/restoration gate")
        if child.poll() is not None:
            raise RuntimeError("Watcher exited before its startup acknowledgment")
        time.sleep(0.1)
    raise TimeoutError("Watcher startup acknowledgment timed out; reconcile its saved process before retrying")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.run:
        watch(args.pid, args.out, args.run)
    else:
        print(json.dumps(ensure(args.pid, args.out)))


if __name__ == "__main__":
    main()
