# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one bounded, identity-checked diagnostic on the released wave-4 lagged node."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

from launch_wave5 import PYTHON, WORKLOAD, node_action, read, sha256
from validate_wave5 import ENVIRONMENT
from watch_wave5 import process, write

NODE = "28.216.17.9"
TASK = Path(__file__).resolve().parents[1]
PROCESS_MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"


def tagged_processes(marker: str) -> dict[int, str]:
    """Identify diagnostic descendants even when torchrun creates independent rank sessions."""
    token = f"{PROCESS_MARKER}={marker}".encode()
    generation = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if token not in (directory / "environ").read_bytes().split(b"\0"):
                continue
            observed = process(int(directory.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if observed is not None and observed[0] != "Z":
            generation[int(directory.name)] = observed[1]
    return generation


def finish_child(
    child: subprocess.Popen,
    marker: str,
    generation: dict[int, str],
    terminate_timeout: float = 60,
    kill_timeout: float = 15,
) -> None:
    """Stop only marker-bound process generations, including reparented rank workers."""
    for signum, timeout in ((signal.SIGTERM, terminate_timeout), (signal.SIGKILL, kill_timeout)):
        deadline = time.monotonic() + timeout
        signaled = set()
        while True:
            current = tagged_processes(marker)
            generation.update(current)
            for pid, started in current.items():
                identity = pid, started
                if identity in signaled:
                    continue
                observed = process(pid)
                if observed is not None and observed[1] == started:
                    try:
                        os.kill(pid, signum)
                    except ProcessLookupError:
                        pass
                    signaled.add(identity)
            if child.poll() is not None and not tagged_processes(marker):
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise TimeoutError("Diagnostic launcher or marker-bound rank workers remain live after cleanup")


def keeper_generation(pid: int) -> list[tuple[int, str, int]]:
    """Record only the keeper launcher and its independently sessioned rank workers."""
    generation = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        worker = int(path.name)
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            command = (path / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if worker != pid and int(fields[1]) != pid:
            continue
        if fields[0] == "Z":
            continue
        if WORKLOAD not in command or int(fields[2]) != worker or int(fields[3]) != worker:
            raise RuntimeError("Keeper has an unrecognized child command or session")
        generation.append((worker, fields[19], int(fields[2])))
    if not any(worker == pid for worker, _, _ in generation):
        raise RuntimeError("Keeper launcher identity disappeared")
    return generation


def release_keeper(state: dict, state_path: Path) -> dict:
    """Stop only the currently identified keeper session; refuse unrelated CUDA occupancy."""
    keeper_file = Path("/root/dl/runtime/gpu_matmul.pid")
    pid = int(keeper_file.read_text())
    observed = process(pid)
    if observed is None or observed[0] == "Z" or WORKLOAD.decode() not in observed[2].split():
        raise RuntimeError("Diagnostic allocation is not held by the expected keeper")
    parent = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    if int(parent[2]) != pid or int(parent[3]) != pid:
        raise RuntimeError("Keeper does not own its expected session and process group")
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).split()
    if not occupied:
        raise RuntimeError("Keeper health/occupancy is missing; do not silently replace it")
    generation = keeper_generation(pid)
    owned = {worker for worker, _, _ in generation if worker != pid}
    for worker in occupied:
        if int(worker) not in owned:
            raise RuntimeError("Unrelated CUDA occupancy prevents diagnostic")
    groups = {group for _, _, group in generation}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != "Z" and int(fields[2]) in groups and int(path.name) not in owned | {pid}:
            raise RuntimeError("Keeper rank group contains an unidentified process")
    if process(pid) != observed:
        raise RuntimeError("Keeper identity changed before diagnostic release")
    state.update(
        keeper_release_attempted=True,
        keeper={"pid": pid, "start_ticks": observed[1], "generation": generation},
    )
    write(state_path, state)
    for worker, started, group in generation:
        current = process(worker)
        if current is not None and current[0] != "Z" and current[1] == started:
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if all(
            (current := process(worker)) is None or current[0] == "Z" or current[1] != started
            for worker, started, _ in generation
        ):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Identified keeper generation did not exit")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("CUDA occupancy remains after keeper release")
    if keeper_file.read_text().strip() == str(pid):
        keeper_file.unlink()
    return {"pid": pid, "start_ticks": observed[1], "generation": generation}


def run(root: Path, out: Path, profile: str = "parity") -> dict:
    """Preserve each diagnostic's own source/log/result identity and restoration outcome."""
    if NODE not in subprocess.check_output(["hostname", "-I"], text=True).split():
        raise RuntimeError("Diagnostic must run on its designated .9 node")
    with (root / "run-preparation/.diagonal-diagnostic.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists():
            raise FileExistsError("Diagnostic already has evidence; use a new identity after inspection")
        binding = root / "run-preparation/wave5-candidate.json"
        frozen = read(binding)
        source = Path(frozen["candidate"])
        if any(sha256(source / name) != digest for name, digest in frozen["files_sha256"].items()):
            raise ValueError("Frozen candidate identity changed")
        queued = read(root / "run-preparation/wave5-launch-queue.json")
        if any(sha256(TASK / name) != digest for name, digest in queued["helpers_sha256"].items()):
            raise ValueError("Bound restoration helpers changed")
        entry = next(item for item in read(root / "run-preparation/wave4.json")["launches"] if item["node"] == NODE)
        predecessor = process(entry["pid"])
        if predecessor is not None and predecessor[0] != "Z":
            raise RuntimeError("Predecessor is still live or its PID was reused")
        manifest = read(Path(entry["out"]) / "run.json")
        training = manifest["config"]["training"]
        if manifest["status"] != "complete" or manifest["completed_steps"] != training["steps"]:
            raise RuntimeError("Predecessor did not complete")
        if "keeper start exit 0" not in (root / "runs/wave4/keeper-restore-28.216.17.9.log").read_text():
            raise RuntimeError("Predecessor restoration is incomplete")
        processes = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
        if any("-m torch.distributed.run" in item or "-m experiment.train" in item for item in processes):
            raise RuntimeError("Existing training/torchrun process prevents diagnostic")
        out.mkdir()
        diagnostic = TASK / "tracking/diagnose_fused_diagonal.py"
        archive = out / "diagnostic-source.tar"
        with tarfile.open(archive, "x") as stream:
            for path in (Path(__file__).resolve(), diagnostic):
                stream.add(path, arcname=f"tracking/{path.name}", recursive=False)
        command = [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(diagnostic),
            "--binding",
            str(binding),
            "--out",
            str(out / "result.json"),
            "--profile",
            profile,
        ]
        state = {
            "status": "preflight",
            "profile": profile,
            "node": NODE,
            "controller_pid": os.getpid(),
            "binding_sha256": sha256(binding),
            "diagnostic_sha256": sha256(diagnostic),
            "controller_sha256": sha256(Path(__file__)),
            "source_archive_sha256": sha256(archive),
            "command": command,
            "environment": ENVIRONMENT,
            "process_marker": uuid4().hex,
        }
        state_path = out / "state.json"
        write(state_path, state)
        child = None
        generation = {}
        interrupted_signals = []

        def interrupted(signum, frame):
            interrupted_signals.append(signum)

        handlers = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGINT)}
        try:
            state["keeper"] = release_keeper(state, state_path)
            with (out / "diagnostic.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=TASK,
                    env={**os.environ, **ENVIRONMENT, PROCESS_MARKER: state["process_marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                state.update(status="running", diagnostic_pid=child.pid)
                write(state_path, state)
                deadline = time.monotonic() + 1800
                while True:
                    generation.update(tagged_processes(state["process_marker"]))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        state["timed_out"] = True
                        raise subprocess.TimeoutExpired(command, 1800)
                    try:
                        state["exit_code"] = child.wait(timeout=min(1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            if state["exit_code"]:
                raise RuntimeError("Diagnostic failed; preserved evidence must be inspected")
            state.update(status="measured_diagnostic_not_launch_gate", result_sha256=sha256(out / "result.json"))
        except Exception as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if child is not None:
                    try:
                        finish_child(child, state["process_marker"], generation)
                        state["cleanup_exit_code"] = child.returncode
                    except Exception as error:
                        state.update(status="needs_attention", cleanup_error=f"{type(error).__name__}: {error}")
                if state.get("keeper_release_attempted"):
                    if child is not None and (
                        child.poll() is None or tagged_processes(state["process_marker"]) or "cleanup_error" in state
                    ):
                        state["keeper_restore"] = "refused_live_diagnostic"
                    else:
                        try:
                            node_action(root, "restore")
                            state["keeper_restore"] = 0
                        except Exception as error:
                            state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            finally:
                state["diagnostic_generation"] = generation
                if interrupted_signals:
                    state["interrupted_signals"] = interrupted_signals
                try:
                    write(state_path, state)
                finally:
                    for signum, handler in handlers.items():
                        signal.signal(signum, handler)
        if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
            raise RuntimeError("Diagnostic keeper restoration failed; inspect the saved state")
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("parity", "sensitivity"), default="parity")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.out, args.profile), indent=2))


if __name__ == "__main__":
    main()
