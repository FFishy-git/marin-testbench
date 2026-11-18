# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and run one bounded combined-product probe on the restored Wave-7 .139 node."""

import argparse
import json
import os
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import PYTHON, node_action, read, sha256
from run_fixed_age_validation import VALIDATION_HELPERS, require_completed_wave
from validate_wave5 import ENVIRONMENT
from watch_wave5 import process, write
from wave7_release import bounded_execution, idle_training, local_node, node_guard, release_keeper

NODE = "28.216.17.139"
EXTRA_HELPERS = (
    "run_combined_outer_gpu.py",
    "diagnose_combined_outer_gpu.py",
    "diagnose_combined_outer.py",
    "diagnose_fused_diagonal.py",
    "benchmark_diagonal.py",
)


def freeze(root: Path, out: Path) -> dict:
    """Preserve the probe and byte-identical release guards separately from training sources."""
    predecessor = require_completed_wave(root)
    if out.exists():
        raise FileExistsError("Combined-product probe already has evidence; inspect it without overwriting")
    original = root / "campaigns/lr-fixed-age-0p5m-20261116/validation-helpers"
    original_binding = read(original.parent / "validation-binding.json")
    if set(original_binding["helpers_sha256"]) != set(VALIDATION_HELPERS):
        raise ValueError("Original allocation guard inventory differs")
    out.mkdir()
    helpers = out / "helpers"
    helpers.mkdir()
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as stream:
        for name in (*VALIDATION_HELPERS, *EXTRA_HELPERS):
            source = original / name if name in VALIDATION_HELPERS else Path(__file__).parent / name
            if name in VALIDATION_HELPERS and sha256(source) != original_binding["helpers_sha256"][name]:
                raise ValueError("Original allocation guard changed")
            destination = helpers / name
            destination.write_bytes(source.read_bytes())
            stream.add(destination, arcname=f"tracking/{name}", recursive=False)
    result = {
        "status": "frozen_production_probe_not_launch_gate",
        "root": str(root.resolve()),
        "out": str(out.resolve()),
        "node": NODE,
        "helpers_sha256": {name: sha256(helpers / name) for name in (*VALIDATION_HELPERS, *EXTRA_HELPERS)},
        "archive_sha256": sha256(archive),
        "original_guard_binding_sha256": sha256(original.parent / "validation-binding.json"),
        "source_release_sha256": sha256(root / "run-preparation/wave7-release-binding.json"),
        "predecessor": predecessor,
        "environment": ENVIRONMENT,
    }
    write(out / "binding.json", result)
    return result


def binding(root: Path, out: Path) -> dict:
    frozen = read(out / "binding.json")
    if (
        Path(__file__).resolve() != (out / "helpers/run_combined_outer_gpu.py").resolve()
        or frozen["root"] != str(root.resolve())
        or frozen["out"] != str(out.resolve())
        or frozen["node"] != NODE
        or set(frozen["helpers_sha256"]) != set((*VALIDATION_HELPERS, *EXTRA_HELPERS))
        or any(sha256(out / "helpers" / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or sha256(out / "source.tar") != frozen["archive_sha256"]
        or sha256(root / "campaigns/lr-fixed-age-0p5m-20261116/validation-binding.json")
        != frozen["original_guard_binding_sha256"]
        or sha256(root / "run-preparation/wave7-release-binding.json") != frozen["source_release_sha256"]
        or frozen["environment"] != ENVIRONMENT
    ):
        raise ValueError("Combined-product probe source/controller binding changed")
    return frozen


def run(root: Path, out: Path) -> dict:
    frozen = binding(root, out)
    if local_node() != NODE:
        raise RuntimeError("Combined-product probe must run on its original .139 allocation")
    state_path = out / "state.json"
    if state_path.exists() or (out / "result.json").exists():
        raise FileExistsError("Combined-product execution evidence exists; preserve it without retry")
    with node_guard(root, NODE) as signals:
        predecessor = require_completed_wave(root)
        snapshot = out / "predecessor-snapshot.json"
        release = read(root / "run-preparation/wave7-release-binding.json")
        subprocess.run(
            [
                PYTHON,
                str(Path(release["helpers"]) / "snapshot_wave7_release.py"),
                "--root",
                str(root),
                "--out",
                str(snapshot),
            ],
            check=True,
            timeout=240,
        )
        if read(snapshot)["status"] != "complete":
            raise RuntimeError("Wave-7 parent/restoration state remains live or unknown")
        idle_training()
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(devices) != 8 or any("A100-SXM4-40GB" not in name for name in devices):
            raise ValueError("Combined-product probe requires the original eight-A100 allocation")
        state = {
            "status": "preflight",
            "root": str(root),
            "python": PYTHON,
            "node": NODE,
            "controller_pid": os.getpid(),
            "controller_start_ticks": process(os.getpid())[1],
            "helpers_sha256": frozen["helpers_sha256"],
            "binding_sha256": sha256(out / "binding.json"),
            "predecessor": predecessor,
            "predecessor_snapshot_sha256": sha256(snapshot),
        }
        write(state_path, state)
        try:
            state["keeper"] = release_keeper(state, state_path)
            state["status"] = "running"
            command = [
                PYTHON,
                str(out / "helpers/diagnose_combined_outer_gpu.py"),
                "--root",
                str(root),
                "--out",
                str(out),
            ]
            code = bounded_execution(command, out / "helpers", out / "diagnostic.log", state, state_path, timeout=1800)
            state["exit_code"] = code
            result = read(out / "result.json")
            state["result_sha256"] = sha256(out / "result.json")
            if code or result["status"] != "passed_production_combined_outer_diagnostic_not_launch_gate":
                raise RuntimeError("Production combined-product probe failed; inspect preserved evidence")
            binding(root, out)
            state.update(status="measured_production_probe_not_launch_gate", result_sha256=sha256(out / "result.json"))
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state:
                        raise RuntimeError("Unverified child cleanup prevents keeper restoration")
                    node_action(root, "restore")
                    state["keeper_restore"] = 0
                except Exception as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(state_path, state)
        if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
            raise RuntimeError("Combined-product probe keeper restoration failed")
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "check", "run"), required=True)
    args = parser.parse_args()
    result = (
        freeze(args.root, args.out)
        if args.phase == "freeze"
        else (binding(args.root, args.out) if args.phase == "check" else run(args.root, args.out))
    )
    print(json.dumps({"status": result["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
