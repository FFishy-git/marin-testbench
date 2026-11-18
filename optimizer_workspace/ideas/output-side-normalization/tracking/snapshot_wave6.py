# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind wave-6 observations to its exact inventory and retain failure precedence."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import snapshot
from launch_wave5 import NODES, PYTHON, read

ARMS = ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused")


def verify_inventory(root: Path, *, require_full: bool = False) -> None:
    report = read(root / "run-preparation/wave6.json")
    entries = report["launches"]
    if report.get("expected_launches") != len(ARMS):
        raise ValueError("Wave-6 inventory requires the four-run budget")
    if require_full and len(entries) != len(ARMS):
        raise ValueError("Wave-6 inventory is partial; no completed-wave analysis")
    seen = set()
    for entry in entries:
        node = entry["node"]
        if node not in NODES or node in seen:
            raise ValueError("Wave-6 inventory contains an unexpected or duplicate node")
        seen.add(node)
        arm = ARMS[NODES.index(node)]
        if (
            (entry["arm"], entry["seed"], entry["batch"]) != (arm, 0, "0p5m")
            or entry["out"] != str(root / "runs/wave6" / arm)
            or type(entry["pid"]) is not int
            or entry["pid"] <= 0
        ):
            raise ValueError("Wave-6 inventory differs from its canonical node/config/output identity")


def observe(root: Path) -> dict:
    verify_inventory(root)
    result = snapshot.snapshot(root, "wave6")
    entries = read(root / "run-preparation/wave6.json")["launches"]
    for run, entry in zip(result["runs"], entries, strict=True):
        run["keeper_watcher"] = observe_watcher(entry, run["status"])
    if any(
        run["status"] == "stopped_incomplete" or run["keeper_watcher"]["status"] == "needs_attention"
        for run in result["runs"]
    ):
        result["status"] = "needs_attention"
    elif result["status"] == "complete" and any(run["keeper_watcher"]["status"] != "restored" for run in result["runs"]):
        result["status"] = "awaiting_keeper_restoration"
    return result


def observe_watcher(entry: dict, training_status: str) -> dict:
    path = Path(entry["out"]).parent / f"{Path(entry['out']).name}.watcher.json"
    if not path.exists():
        return {"status": "needs_attention", "error": "No keeper watcher receipt"}
    state = read(path)
    if (
        state.get("training_pid") != entry["pid"]
        or state.get("out") != entry["out"]
        or type(state.get("pid")) is not int
        or state["pid"] <= 0
        or ("start_ticks" in entry and state.get("training_start_ticks") != entry["start_ticks"])
    ):
        return {"status": "needs_attention", "error": "Keeper watcher identity differs", "receipt": state}
    if state.get("status") == "restored":
        if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0 or training_status == "running":
            return {"status": "needs_attention", "error": "Contradictory keeper restoration receipt", "receipt": state}
        return {"status": "restored", "receipt": state}
    if state.get("status") not in ("watching", "restoring"):
        return {
            "status": "needs_attention",
            "error": "Keeper watcher failed or has no startup acknowledgment",
            "receipt": state,
        }
    if not isinstance(state.get("watcher_start_ticks"), str):
        return {"status": "needs_attention", "error": "Watcher generation is not recorded", "receipt": state}
    probe = """
import json, pathlib, sys
path = pathlib.Path("/proc") / sys.argv[1]
try:
    before = (path / "stat").read_text().rsplit(")", 1)[1].split()
    arguments = [arg.decode() for arg in (path / "cmdline").read_bytes().split(b"\\0") if arg]
    after = (path / "stat").read_text().rsplit(")", 1)[1].split()
except (FileNotFoundError, ProcessLookupError):
    print(json.dumps({"status": "missing"}))
else:
    print(json.dumps({"status": after[0], "start_ticks": after[19], "same_generation": before[19] == after[19],
                      "arguments": arguments}))
"""
    try:
        observed = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                f"root@{entry['node']}",
                shlex.join([PYTHON, "-c", probe, str(state["pid"])]),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"status": "unknown", "error": "Watcher SSH observation timed out", "receipt": state}
    if observed.returncode == 255:
        return {"status": "unknown", "error": observed.stderr.strip(), "receipt": state}
    if observed.returncode:
        return {"status": "unknown", "error": observed.stderr.strip(), "receipt": state}
    try:
        process = json.loads(observed.stdout)
    except json.JSONDecodeError:
        return {"status": "unknown", "error": "Invalid watcher process observation", "receipt": state}
    arguments = process.get("arguments", [])
    expected_options = {
        "--run": str(path),
        "--pid": str(entry["pid"]),
        "--out": entry["out"],
        "--root": str(path.parents[2]),
    }
    valid_options = all(
        arguments.count(option) == 1
        and arguments.index(option) + 1 < len(arguments)
        and arguments[arguments.index(option) + 1] == value
        for option, value in expected_options.items()
    )
    if (
        process.get("status") not in ("missing", "Z")
        and process.get("same_generation") is True
        and process.get("start_ticks") == state["watcher_start_ticks"]
        and any(Path(argument).name == "watch_wave6.py" for argument in arguments)
        and valid_options
    ):
        return {"status": state["status"], "receipt": state}
    return {"status": "needs_attention", "error": "Keeper watcher stopped or PID was reused", "receipt": state}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = observe(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(record["status"])


if __name__ == "__main__":
    main()
