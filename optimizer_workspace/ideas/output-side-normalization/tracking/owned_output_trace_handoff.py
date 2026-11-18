# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only failed-validator absence and exact restored-keeper ownership probe."""

import argparse
import json
import subprocess
from pathlib import Path

from owned_output_handoff import keeper_observation, marker_processes, require_absent, same_generation, sha256

NODE = "28.216.17.139"
CONTROLLER_MARKER = "OUTPUT_MARIN_OWNED_HANDOFF_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"


def probe(root: Path) -> dict:
    binding_path = root / "run-preparation/owned-output-handoff-binding.json"
    frozen = json.loads(binding_path.read_text())
    state_path = root / "run-preparation/owned-output-handoff-state.json"
    state = json.loads(state_path.read_text())
    intent_path = Path(frozen["helpers"]).parent / "controller.json"
    intent = json.loads(intent_path.read_text())
    if (
        frozen["root"] != str(root)
        or frozen["node"] != NODE
        or state["binding_sha256"] != sha256(binding_path)
        or intent["binding_sha256"] != sha256(binding_path)
        or intent["status"] != "acknowledged"
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["status"] != "needs_attention"
        or type(state["exit_code"]) is not int
        or state["exit_code"] != 1
        or type(state["cleanup_exit_code"]) is not int
        or state["cleanup_exit_code"] != 1
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or "cleanup_error" in state
        or not state["child_marker"]
        or not intent["marker"]
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Failed-validator terminal ownership or restored allocation differs")
    require_absent(state["controller_pid"])
    require_absent(state["child_pid"])
    if marker_processes(CHILD_MARKER, state["child_marker"]):
        raise RuntimeError("Failed-validator diagnostic children remain live")
    keeper = keeper_observation()
    marked = marker_processes(CONTROLLER_MARKER, intent["marker"])
    if any(
        pid not in keeper["generation"] or not same_generation(keeper["generation"][pid], process)
        for pid, process in marked.items()
    ):
        raise RuntimeError("Failed-controller marker belongs outside the exact restored keeper")
    return {
        "status": "failed_validator_absent_exact_restored_keeper_verified",
        "node": NODE,
        "binding_sha256": sha256(binding_path),
        "state_sha256": sha256(state_path),
        "intent_sha256": sha256(intent_path),
        "original_handles_absent": True,
        "training_child_markers_absent": True,
        "keeper": keeper,
        "inherited_controller_marker_pids": sorted(marked),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.root), allow_nan=False))


if __name__ == "__main__":
    main()
