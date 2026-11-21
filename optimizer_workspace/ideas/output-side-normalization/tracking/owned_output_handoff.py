# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only LR-worker absence and exact restored-keeper ownership checks."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

PROC_ROOT = Path("/proc")
KEEPER_PID = Path("/root/dl/runtime/gpu_matmul.pid")
PYTHON = "/opt/conda/envs/torch-base/bin/python3.13"
WORKLOAD = "/root/dl/runtime/script/hy_infuse_moe/gpu_matmul.py"
TORCHRUN = "/opt/conda/envs/torch-base/bin/torchrun"
WORKER_MARKER = "OUTPUT_MARIN_TUNING_WORKER_ID"
CHILD_MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"
CEILING = "campaigns/lr-fixed-age-ceiling-visible8-recovery-20261117"
WORKLOAD_ARGS = ["--size", "8192", "--dtype", "fp16", "--pause", "0"]
MARKER_SCAN_ATTEMPTS = 3
EXIT_STATES = ("Z", "X", "x")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(pid: int, *, allow_reparent: bool = False) -> dict | None:
    """Reject generation drift instead of accepting a mixed /proc observation."""
    directory = PROC_ROOT / str(pid)
    try:
        before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        command = [arg.decode() for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
        after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    ownership = slice(2, 4) if allow_reparent else slice(1, 4)
    if before[19] != after[19] or before[ownership] != after[ownership]:
        raise RuntimeError("Handoff process generation changed during observation")
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
            process = observed(int(path.name), allow_reparent=allow_reparent)
            if process is not None and process["status"] != "Z":
                result[process["pid"]] = process
    return result


def same_generation(before: dict, after: dict | None, *, allow_reparent: bool = False) -> bool:
    """Allow scheduler state transitions while retaining every ownership field."""
    ignored = ("status", "parent") if allow_reparent else ("status",)
    return (
        after is not None
        and before["status"] not in ("Z", "X", "x")
        and after["status"] not in ("Z", "X", "x")
        and {key: value for key, value in before.items() if key not in ignored}
        == {key: value for key, value in after.items() if key not in ignored}
    )


def marker_processes(name: str, value: str, *, allow_reparent: bool = False) -> dict[int, dict]:
    """Rescan transient exits without accepting mixed generations or partial absence."""
    token = f"{name}={value}".encode()
    identities, marked = {}, set()
    for _ in range(MARKER_SCAN_ATTEMPTS):
        result, unstable = {}, set()
        for pid, process in live_processes(allow_reparent=allow_reparent).items():
            if process["status"] in EXIT_STATES:
                continue
            if pid in identities and not same_generation(identities[pid], process, allow_reparent=allow_reparent):
                raise RuntimeError(f"Marked process generation changed during handoff: pid={pid}")
            try:
                tokens = (PROC_ROOT / str(pid) / "environ").read_bytes().split(b"\0")
            except (FileNotFoundError, ProcessLookupError):
                identities.setdefault(pid, process)
                unstable.add(pid)
                continue
            if token not in tokens:
                if pid in marked:
                    raise RuntimeError(f"Marked process environment changed during handoff: pid={pid}")
                continue
            identities.setdefault(pid, process)
            marked.add(pid)
            result[pid] = process
        for pid, process in result.items():
            current = observed(pid, allow_reparent=allow_reparent)
            if current is None:
                unstable.add(pid)
            elif current["status"] in EXIT_STATES:
                # Exiting processes lose cmdline, but their owner fields must remain exact.
                fields = ("pid", "start_ticks", "group", "session") + (() if allow_reparent else ("parent",))
                if any(current[key] != process[key] for key in fields):
                    raise RuntimeError(f"Marked process generation changed during handoff: pid={pid}")
                unstable.add(pid)
            elif not same_generation(process, current, allow_reparent=allow_reparent):
                raise RuntimeError(f"Marked process generation changed during handoff: pid={pid}")
        if not unstable:
            return result
    raise RuntimeError(
        f"Marked process scan remained unstable during handoff after {MARKER_SCAN_ATTEMPTS} attempts: "
        f"pids={sorted(unstable)}"
    )


def require_absent(pid: int) -> None:
    process = observed(pid)
    if process is not None and process["status"] != "Z":
        raise RuntimeError("Original LR worker or training handle is not absent")


def keeper_observation() -> dict:
    """Require exact commands, independent sessions and one owner per physical A100."""
    pid_text = KEEPER_PID.read_text()
    pid = int(pid_text)
    processes = live_processes()
    generation = {worker: process for worker, process in processes.items() if worker == pid or process["parent"] == pid}
    launcher = [PYTHON, TORCHRUN, "--standalone", "--nproc-per-node=8", WORKLOAD, *WORKLOAD_ARGS]
    rank_command = [PYTHON, "-u", WORKLOAD, *WORKLOAD_ARGS]
    if len(generation) != 9 or pid not in generation:
        raise RuntimeError("Restored keeper lacks its exact launcher and eight ranks")
    for worker, process in generation.items():
        if (
            process["command"] != (launcher if worker == pid else rank_command)
            or process["group"] != worker
            or process["session"] != worker
        ):
            raise RuntimeError("Restored keeper command or session differs")
    groups = set(generation)
    if any(process["group"] in groups and worker not in generation for worker, process in processes.items()):
        raise RuntimeError("Restored keeper session contains an unrelated process")
    devices = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True
            ).splitlines(),
            skipinitialspace=True,
        )
    )
    if (
        len(devices) != 8
        or any(len(row) != 3 for row in devices)
        or sorted(int(row[0]) for row in devices) != list(range(8))
        or len({row[1] for row in devices}) != 8
        or any("A100-SXM4-40GB" not in row[2] for row in devices)
    ):
        raise RuntimeError("Restored allocation is not the original eight physical A100s")
    occupied = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
            ).splitlines(),
            skipinitialspace=True,
        )
    )
    if (
        len(occupied) != 8
        or any(len(row) != 2 for row in occupied)
        or {int(row[0]) for row in occupied} != set(generation) - {pid}
        or {row[1] for row in occupied} != {row[1] for row in devices}
    ):
        raise RuntimeError("Restored keeper does not exclusively occupy all eight GPUs")
    if KEEPER_PID.read_text() != pid_text or any(
        not same_generation(process, observed(worker)) for worker, process in generation.items()
    ):
        raise RuntimeError("Restored keeper generation changed during device inspection")
    return {"pid": pid, "generation": generation, "devices": devices, "occupied": occupied}


def probe(root: Path, family: str) -> dict:
    """Distinguish restored keepers from live experiment descendants without mutating either."""
    directory = root / CEILING
    campaign_path = directory / "campaign.json"
    binding_path = directory / "execution-binding.json"
    frozen = read(binding_path)
    assignment = next(item for item in frozen["plan"] if item["family"] == family)
    state = read(directory / "workers" / f"{family}.json")
    if (
        frozen["campaign"] != str(campaign_path)
        or frozen["campaign_sha256"] != sha256(campaign_path)
        or state["binding_sha256"] != sha256(binding_path)
        or state["family"] != family
        or state["node"] != assignment["node"]
        or state["status"] != "complete"
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or [trial["name"] for trial in state["trials"]] != [trial["name"] for trial in assignment["trials"]]
        or state["worker_marker"] != read(directory / "launch-state.json")["worker_markers"][family]
        or assignment["node"] not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Handoff does not identify the original completed LR family/allocation")
    require_absent(state["pid"])
    for trial, planned in zip(state["trials"], assignment["trials"], strict=True):
        child_path = directory / "workers" / f"{trial['name']}.execution.json"
        child = read(child_path)
        if (
            trial["command"] != planned["command"]
            or trial["status"] != "complete"
            or type(trial.get("exit_code")) is not int
            or trial["exit_code"] != 0
            or trial["execution_sha256"] != sha256(child_path)
            or child["command"] != planned["command"]
            or type(child.get("cleanup_exit_code")) is not int
            or child["cleanup_exit_code"] != 0
            or not child["child_marker"]
        ):
            raise RuntimeError("Handoff original training command or cleanup identity differs")
        require_absent(child["child_pid"])
        if marker_processes(CHILD_MARKER, child["child_marker"]):
            raise RuntimeError("Original LR training-child marker remains")
    keeper = keeper_observation()
    marked = marker_processes(WORKER_MARKER, state["worker_marker"])
    if any(
        pid not in keeper["generation"] or not same_generation(keeper["generation"][pid], process)
        for pid, process in marked.items()
    ):
        raise RuntimeError("LR worker marker belongs to a process outside the exact restored keeper")
    if any(not same_generation(process, observed(pid)) for pid, process in keeper["generation"].items()):
        raise RuntimeError("Keeper generation changed after inherited-marker inspection")
    if sha256(binding_path) != state["binding_sha256"]:
        raise RuntimeError("Original LR binding changed during handoff")
    return {
        "status": "original_training_absent_exact_keeper_verified",
        "family": family,
        "node": assignment["node"],
        "original_handles_absent": True,
        "training_child_markers_absent": True,
        "keeper": keeper,
        "inherited_worker_marker_pids": sorted(marked),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.root, args.family), allow_nan=False))


if __name__ == "__main__":
    main()
