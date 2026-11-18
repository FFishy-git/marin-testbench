# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only original Wave-10 process absence and exact restored-keeper checks."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from owned_output_handoff import keeper_observation, marker_processes, observed, require_absent, same_generation
from selected_lr_validation import MASK
from watch_wave5 import write

WORKER_MARKER = "OUTPUT_MARIN_WAVE10_WORKER_ID"
CHILD_MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"


def probe(root: Path, arm: str) -> dict:
    release_path = root / "run-preparation/wave10-launch-binding.json"
    release = json.loads(release_path.read_text())
    index = release["arms"].index(arm)
    node = release["nodes"][index]
    helpers = Path(release["helpers"])
    script = (
        "import json,sys\nfrom pathlib import Path\nfrom owned_output_run_history import binding,completed_run\n"
        "root=Path(sys.argv[1]); release=binding(root)\n"
        "print(json.dumps(completed_run(root,sys.argv[2],release),allow_nan=False))"
    )
    measured = json.loads(
        subprocess.run(
            [release["commands"][node][0], "-P", "-c", script, str(root), arm],
            cwd=helpers,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)},
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        ).stdout
    )
    state_path = root / f"run-preparation/wave10-worker-{node}.json"
    state = json.loads(state_path.read_text())
    acknowledged = json.loads(state_path.with_suffix(".start.json").read_text())
    launch = json.loads(root.joinpath("run-preparation/wave10-launch-state.json").read_text())
    command = [
        release["commands"][node][0],
        str(helpers / "owned_output_production.py"),
        "--root",
        str(root),
        "--phase",
        "worker",
        "--arm",
        arm,
    ]
    if (
        node not in subprocess.check_output(["hostname", "-I"], text=True).split()
        or state["status"] != "complete"
        or state["arm"] != arm
        or state["node"] != node
        or state["command"] != release["commands"][node]
        or state["release_binding_sha256"] != launch["binding_sha256"]
        or state["worker_marker"] != launch["worker_markers"][arm]
        or state["completed_run"] != measured
        or acknowledged["status"] != "acknowledged"
        or acknowledged["arm"] != arm
        or acknowledged["node"] != node
        or acknowledged["release_binding_sha256"] != state["release_binding_sha256"]
        or acknowledged["worker_marker"] != state["worker_marker"]
        or acknowledged["pid"] != state["worker_pid"]
        or acknowledged["start_ticks"] != state["worker_start_ticks"]
        or acknowledged["command"] != command
        or state["worker_command"] != command
        or acknowledged["generation"]["pid"] != state["worker_pid"]
        or acknowledged["generation"]["start_ticks"] != state["worker_start_ticks"]
        or acknowledged["generation"]["command"] != command
        or acknowledged["generation"]["group"] != state["worker_pid"]
        or acknowledged["generation"]["session"] != state["worker_pid"]
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or "error" in state
        or "cleanup_error" in state
    ):
        raise RuntimeError("Wave10 original completed worker or restoration receipt differs")
    require_absent(state["worker_pid"])
    require_absent(state["child_pid"])
    if marker_processes(CHILD_MARKER, state["child_marker"]):
        raise RuntimeError("Wave10 marked training descendants remain")
    keeper = keeper_observation()
    restored = state["restored_keeper"]
    generation = {int(pid): process for pid, process in restored["generation"].items()}
    if (
        restored["pid"] != keeper["pid"]
        or set(generation) != set(keeper["generation"])
        or any(not same_generation(process, keeper["generation"][pid]) for pid, process in generation.items())
    ):
        raise RuntimeError("Wave10 current keeper differs from its original restoration receipt")
    marked = marker_processes(WORKER_MARKER, state["worker_marker"])
    if any(
        pid not in keeper["generation"] or not same_generation(process, keeper["generation"][pid])
        for pid, process in marked.items()
    ):
        raise RuntimeError("Wave10 inherited worker marker belongs outside its exact restored keeper")
    if any(not same_generation(process, observed(pid)) for pid, process in keeper["generation"].items()):
        raise RuntimeError("Wave10 restored keeper changed during marker inspection")
    return {
        "status": "wave10_original_training_absent_exact_keeper_verified",
        "arm": arm,
        "node": node,
        "original_handles_absent": True,
        "training_child_markers_absent": True,
        "keeper": keeper,
        "inherited_worker_marker_pids": sorted(marked),
        "cuda_visible_devices": MASK,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = probe(args.root, args.arm)
    if args.out:
        write(args.out, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
