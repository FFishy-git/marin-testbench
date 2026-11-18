# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Watch an exact Wave-7 training generation; restore only through its bound controller."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from launch_wave5 import NODES
from watch_wave5 import write

STARTUP_TIMEOUT = 30
NODE_LOCK_TIMEOUT = 120
RESTORE_TIMEOUT = 240


def observed_process(pid: int) -> tuple[str, str, list[str]] | None:
    directory = Path("/proc") / str(pid)
    try:
        before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        arguments = [arg.decode() for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
        after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if before[19] != after[19]:
        raise RuntimeError("Process generation changed during observation")
    return after[0], after[19], arguments


def option(arguments: list[str], name: str, value: str) -> bool:
    return (
        arguments.count(name) == 1
        and arguments.index(name) + 1 < len(arguments)
        and (arguments[arguments.index(name) + 1] == value)
    )


def bound_controller(root: Path, out: Path) -> Path:
    record = json.loads((root / "run-preparation/wave7-release-binding.json").read_text())
    directory = Path(record["helpers"])
    if Path(__file__).resolve() != (directory / "watch_wave7_release.py").resolve():
        raise RuntimeError("Execute the bound Wave-7 watcher, not the mutable checkout")
    if out.name not in record["arms"] or out != root / "runs/wave7" / out.name:
        raise RuntimeError("Watcher output is outside the bound Wave-7 allocation")
    return directory / "wave7_release.py"


def training_generation(pid: int, out: Path, root: Path, expected: str | None) -> str | None:
    observed = observed_process(pid)
    if observed is None:
        return expected
    status, ticks, arguments = observed
    if expected is not None and ticks != expected:
        raise RuntimeError("Training PID reused; keeper restoration refused")
    if status == "Z":
        return ticks
    record = json.loads((root / "run-preparation/wave7-release-binding.json").read_text())
    config = Path(record["candidate"]) / "experiment/configs" / f"{out.name}_0p5m_seed0.json"
    if (
        arguments.count("-m") != 2
        or not any(arguments[i : i + 2] == ["-m", "torch.distributed.run"] for i in range(len(arguments) - 1))
        or not any(arguments[i : i + 2] == ["-m", "experiment.train"] for i in range(len(arguments) - 1))
        or not option(arguments, "--config", str(config))
        or not option(arguments, "--out", str(out))
    ):
        raise RuntimeError("Training process does not match the bound config/output identity")
    return ticks


def checked_state(path: Path, pid: int, out: Path, root: Path, expected: str | None) -> dict:
    state = json.loads(path.read_text())
    if (
        type(state.get("training_pid")) is not int
        or state["training_pid"] != pid
        or state.get("out") != str(out)
        or state.get("root") != str(root)
        or type(state.get("pid")) is not int
        or state["pid"] <= 0
        or (expected is not None and state.get("training_start_ticks") != expected)
    ):
        raise RuntimeError("Existing watcher belongs to a different training generation")
    ticks = training_generation(pid, out, root, state.get("training_start_ticks"))
    if ticks != state.get("training_start_ticks"):
        raise RuntimeError("Existing watcher did not acknowledge the training generation")
    if state.get("status") == "restored":
        training = observed_process(pid)
        if (
            type(state.get("keeper_restore")) is not int
            or state["keeper_restore"] != 0
            or (training is not None and training[0] != "Z")
        ):
            raise RuntimeError("Existing watcher has a contradictory restoration receipt")
        return state
    if state.get("status") not in ("watching", "restoring"):
        raise RuntimeError("Existing watcher failed; preserve its restoration evidence")
    observed = observed_process(state["pid"])
    if (
        observed is None
        or observed[0] == "Z"
        or observed[1] != state.get("watcher_start_ticks")
        or str(Path(__file__).resolve()) not in observed[2]
        or not option(observed[2], "--run", str(path))
        or not option(observed[2], "--pid", str(pid))
        or not option(observed[2], "--out", str(out))
        or not option(observed[2], "--root", str(root))
    ):
        raise RuntimeError("Existing watcher stopped or changed generation; reconcile without restarting")
    return state


def watch(pid: int, out: Path, root: Path, state_path: Path, expected_start_ticks: str | None = None) -> None:
    """Persist monitoring and guarded restoration outcomes for one acknowledged training PID."""
    if state_path != out.parent / f"{out.name}.watcher.json":
        raise ValueError("Watcher receipt path is not canonical")
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            raise FileExistsError("Watcher evidence already exists; use ensure instead of overwriting")
        state = {
            "pid": os.getpid(),
            "training_pid": pid,
            "training_start_ticks": expected_start_ticks,
            "out": str(out),
            "root": str(root),
        }
        try:
            controller = bound_controller(root, out)
            own = observed_process(os.getpid())
            if own is None:
                raise RuntimeError("Cannot establish watcher process generation")
            state["watcher_start_ticks"] = own[1]
            state["training_start_ticks"] = training_generation(pid, out, root, expected_start_ticks)
            observed = observed_process(pid)
            if observed is not None and observed[0] != "Z":
                state["status"] = "watching"
                write(state_path, state)
                while observed is not None and observed[0] != "Z":
                    training_generation(pid, out, root, state["training_start_ticks"])
                    time.sleep(15)
                    observed = observed_process(pid)
            training_generation(pid, out, root, state["training_start_ticks"])
            # Acknowledge before waiting on discovery's node lock. Only the controller may restore.
            state["status"] = "restoring"
            write(state_path, state)
            command = [
                sys.executable,
                str(controller),
                "--root",
                str(root),
                "--phase",
                "restore",
                "--node-lock-timeout",
                str(NODE_LOCK_TIMEOUT),
            ]
            log_path = state_path.with_suffix(".restore.log")
            with log_path.open("x") as log:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            state.update(controller_pid=child.pid, controller_command=command, controller_log=str(log_path))
            controller_process = observed_process(child.pid)
            if controller_process is None:
                raise RuntimeError("Cannot establish restore controller process generation; preserve without retry")
            state["controller_start_ticks"] = controller_process[1]
            write(state_path, state)
            try:
                code = child.wait(timeout=RESTORE_TIMEOUT)
            except subprocess.TimeoutExpired:
                state["pending_restore_controller"] = {
                    "pid": child.pid,
                    "start_ticks": state["controller_start_ticks"],
                    "command": command,
                    "log": str(log_path),
                }
                raise
            state["controller_exit_code"] = code
            output = log_path.read_text()
            if code:
                raise RuntimeError(f"Bound keeper restoration refused: {output[-4000:]}")
            acknowledgment = json.loads(output.splitlines()[-1])
            state["restore_acknowledgment"] = acknowledgment
            state["keeper_restore"] = acknowledgment.get("keeper_restore")
            record = json.loads((root / "run-preparation/wave7-release-binding.json").read_text())
            if (
                acknowledgment.get("node") != NODES[record["arms"].index(out.name)]
                or type(state["keeper_restore"]) is not int
                or state["keeper_restore"] != 0
            ):
                raise RuntimeError("Bound keeper restoration did not acknowledge integer zero")
            state["status"] = "restored"
            write(state_path, state)
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            write(state_path, state)
            raise


def ensure(pid: int, out: Path, root: Path, expected_start_ticks: str | None = None) -> dict:
    """Install once or return a generation-checked live/restored watcher acknowledgment."""
    root, out = root.resolve(), out.resolve()
    bound_controller(root, out)
    path = out.parent / f"{out.name}.watcher.json"
    with path.with_suffix(".ensure.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            return checked_state(path, pid, out, root, expected_start_ticks)
        ticks = training_generation(pid, out, root, expected_start_ticks)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--pid",
            str(pid),
            "--out",
            str(out),
            "--root",
            str(root),
            "--run",
            str(path),
        ]
        if ticks is not None:
            command.extend(("--start-ticks", ticks))
        with (out.parent / f"{out.name}.watcher.log").open("a") as log:
            child = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if path.exists():
                return checked_state(path, pid, out, root, ticks)
            if child.poll() is not None:
                raise RuntimeError("Watcher exited before its startup acknowledgment")
            time.sleep(0.1)
        raise TimeoutError("Watcher startup timed out; preserve its process and reconcile before retrying")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-ticks")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.run:
        watch(args.pid, args.out, args.root, args.run, args.start_ticks)
    else:
        print(json.dumps(ensure(args.pid, args.out, args.root, args.start_ticks)))


if __name__ == "__main__":
    main()
