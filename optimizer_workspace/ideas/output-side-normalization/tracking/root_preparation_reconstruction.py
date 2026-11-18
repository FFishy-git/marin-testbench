# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct a preserved root diagnostic whose before-call graph expectation was premature."""

import argparse
import copy
import os
from pathlib import Path

import root_preparation_controller as controller
import root_preparation_validation as validation
import torch
from selected_lr_failure_handoff import same_keeper
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

BINDING_SHA256 = "a98ae6fc80841f9fe0c6c9bbf5e4cad01857c6fef690bf5db31ff31cfa03ecc5"
FAILED_STATE_SHA256 = "da27f7f677d4612401d9d84a9ff584680a62b4c30109934e0ce3771ac150a1a5"
FAILED_RECEIPT_SHA256 = "0f64ced3dbd46675275181d49f856eb909702415403c17ed2afd764cad68e303"
FAILURE = "ValueError: Root trace first-capture/next-replay lifecycle differs"
STATUS = "failed_root_diagnostic_independently_reconstructed_before_call_lifecycle_corrected_not_production"


def require_trace_lifecycle(lifecycle: dict, regime: str, mode: str) -> None:
    """Check before-call flags, including the unprepared update-24 initial capture."""
    if regime not in validation.REGIMES or mode not in ("continuous", "resumed"):
        raise ValueError("Root trace lifecycle regime or mode differs")
    if set(lifecycle) != {f"{rank}:{step}" for rank in range(validation.WORLD) for step in validation.STEPS}:
        raise ValueError("Root trace lifecycle lacks all ranks or update coordinates")
    for coordinate, groups in lifecycle.items():
        step = int(coordinate.split(":")[1])
        eager = regime == "unprepared" and mode == "resumed" and step == validation.FIRST_RESUMED
        first_capture = step == validation.BOUNDARY or (
            mode == "resumed"
            and (step == validation.FIRST_RESUMED or (regime == "unprepared" and step == validation.ENDPOINT))
        )
        expected = {"warm": not eager, "graph_active": not first_capture}
        if set(groups) != set(validation.GROUPS) or any(
            set(group) != set(expected)
            or any(type(group[key]) is not bool or group[key] is not value for key, value in expected.items())
            for group in groups.values()
        ):
            raise ValueError(f"Root before-call lifecycle differs at {regime}/{mode}/{coordinate}")


def analysis(root: Path, candidate: Path, prepared: dict, probes: list) -> dict:
    """Keep every original comparison while checking the corrected before-call lifecycle."""
    original = validation.original_evidence(root, prepared)
    outputs = {
        f"{regime}_{protocol}_{mode}": validation.directory(root, regime, protocol, mode)
        for regime in validation.REGIMES
        for protocol in validation.PROTOCOLS
        for mode in ("continuous", "resumed")
    }
    continuation = {
        regime: {
            protocol: validation.pair_set(
                outputs[f"{regime}_{protocol}_resumed"], outputs[f"{regime}_{protocol}_continuous"]
            )
            for protocol in validation.PROTOCOLS
        }
        for regime in validation.REGIMES
    }
    transparency = {
        regime: {
            mode: validation.pair_set(outputs[f"{regime}_trace_{mode}"], outputs[f"{regime}_control_{mode}"])
            for mode in ("continuous", "resumed")
        }
        for regime in validation.REGIMES
    }
    controls = {
        protocol: validation.pair_set(
            outputs[f"prepared_{protocol}_continuous"], outputs[f"unprepared_{protocol}_continuous"]
        )
        for protocol in validation.PROTOCOLS
    }
    prior_work = root / "validation/fused_continuation_20261118"
    reproduction = {
        f"{protocol}_{mode}": validation.pair_set(
            outputs[f"unprepared_{protocol}_{mode}"], prior_work / f"{protocol}_{mode}"
        )
        for protocol in validation.PROTOCOLS
        for mode in ("continuous", "resumed")
    }
    trace_evidence, lifecycle = {}, {}
    for regime in validation.REGIMES:
        for mode in ("continuous", "resumed"):
            out = outputs[f"{regime}_trace_{mode}"]
            trace_evidence[f"{regime}_{mode}"] = validation.traces(out, candidate, prepared)
            lifecycle[f"{regime}_{mode}"] = validation.root_trace_lifecycle(out)
            require_trace_lifecycle(lifecycle[f"{regime}_{mode}"], regime, mode)
    cold = validation.cold_probe_evidence(root, candidate, prepared, probes)
    if validation.original_evidence(root, prepared) != original or validation.source(candidate, root) != prepared:
        raise ValueError("Original root evidence changed during corrected reconstruction")
    gates = {
        "cold_capture_repeat_safety": cold["within_tolerance"],
        "trace_transparency": all(
            validation.pair_set_pass(value) for modes in transparency.values() for value in modes.values()
        ),
        "continuous_control_unchanged": all(validation.pair_set_pass(value) for value in controls.values()),
        "unprepared_original_reproduced": all(validation.pair_set_pass(value) for value in reproduction.values()),
        "prepared_all_rank_continuation": all(
            validation.pair_set_pass(value) for value in continuation["prepared"].values()
        ),
    }
    return {
        "continuation": continuation,
        "transparency": transparency,
        "continuous_controls": controls,
        "unprepared_original_reproduction": reproduction,
        "traces": trace_evidence,
        "trace_root_lifecycle": lifecycle,
        "cold_root": cold,
        "gates": gates,
        "diagnostic_intervention_accepted": all(gates.values()),
        "scope": "Read-only failed-diagnostic reconstruction; unchanged original comparisons and numerical tolerances.",
    }


def failed_owner(root: Path) -> tuple[dict, dict, dict, dict, dict[str, str]]:
    binding_path, helpers, state_path, _, receipt_path = controller.locations(root)
    if (
        sha256(binding_path) != BINDING_SHA256
        or sha256(state_path) != FAILED_STATE_SHA256
        or sha256(receipt_path) != FAILED_RECEIPT_SHA256
    ):
        raise ValueError("Preserved failed root binding, state or receipt differs")
    frozen, state, receipt = read(binding_path), read(state_path), read(receipt_path)
    intent_path = helpers.parent / "controller.json"
    intent = read(intent_path)
    if (
        state["status"] != "needs_attention"
        or receipt["status"] != "needs_attention"
        or receipt["error"] != FAILURE
        or state["error"] != "RuntimeError: Root diagnostic validator failed"
        or state["binding_sha256"] != BINDING_SHA256
        or state["keeper_release_complete"] is not True
        or state["marked_descendants_absent"] is not True
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or any(type(state[key]) is not int or state[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
        or "cleanup_error" in state
        or intent["status"] != "acknowledged"
        or intent["binding_sha256"] != BINDING_SHA256
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or intent["command"] != controller.controller_command(root)
        or receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or any(controller.handoff.observed(state[key]) is not None for key in ("controller_pid", "child_pid"))
        or controller.handoff.marker_processes(controller.CHILD_MARKER, state["child_marker"])
        or state["log_sha256"] != sha256(Path(state["log"]))
    ):
        raise RuntimeError("Original failed root owner, cleanup or restoration is incomplete")
    paths = (binding_path, state_path, receipt_path, intent_path, Path(state["log"]), helpers.parent / "controller.log")
    hashes = {str(path.relative_to(root)): sha256(path) for path in paths}
    return frozen, state, receipt, intent, hashes


def reconstruct(root: Path, out: Path) -> dict:
    """Reconstruct all saved trajectories after failure, with no CUDA or issued-file mutation."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or controller.local_node() != controller.NODE:
        raise RuntimeError("Root failure reconstruction requires hidden CUDA on the original allocation")
    if out.exists() or not out.parent.is_dir() or not out.is_relative_to(root / "run-preparation"):
        raise FileExistsError("Root failure reconstruction requires a new external receipt path")
    frozen, state, receipt, intent, evidence = failed_owner(root)
    if controller.binding(root) != frozen:
        raise ValueError("Original failed root CPU/source/archive binding differs")
    candidate = Path(frozen["candidate"])
    prepared = validation.source(candidate, root)
    if read(candidate / validation.CONFIG)["root"]["inline_steps"] != validation.BOUNDARY - 2:
        raise ValueError("Root reconstruction boundary is not the first continuous capture")
    expected = validation.plan(candidate, root)
    if (
        any(receipt[key] != value for key, value in expected.items() if key != "status")
        or receipt["execution_binding_sha256"] != BINDING_SHA256
        or receipt["candidate_preparation_sha256"] != frozen["preparation_sha256"]
        or len(receipt["gpu_commands"]) != len(expected["commands"])
    ):
        raise ValueError("Original failed root command plan or preparation differs")
    for index, (execution, command) in enumerate(zip(receipt["gpu_commands"], expected["commands"], strict=True)):
        regime, protocol, mode = validation.coordinates(index)
        log = controller.locations(root)[3] / f"{regime}_{protocol}_{mode}.log"
        if (
            (execution["regime"], execution["protocol"], execution["mode"]) != (regime, protocol, mode)
            or execution["command"] != command
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or execution["log"] != str(log)
            or execution["log_sha256"] != sha256(log)
            or execution["trajectory"]
            != validation.trajectory(
                validation.directory(root, regime, protocol, mode), candidate, prepared, regime, mode
            )
        ):
            raise ValueError("Original root command/log/trajectory reconstruction differs")
    corrected = analysis(root, candidate, prepared, receipt["cold_probes"])
    if corrected["cold_root"] != receipt["cold_probe_evidence"] or failed_owner(root)[4] != evidence:
        raise ValueError("Original cold-root or failed-owner evidence changed during reconstruction")
    with controller.keeper_guard():
        keeper = controller.handoff.keeper_observation()
        controller.require_keeper_shape(keeper)
        if (
            not same_keeper(keeper, state["restored_keeper"])
            or keeper["devices"] != frozen["allocation_predecessor"]["restored_keeper"]["devices"]
            or not set(controller.handoff.marker_processes(controller.CONTROLLER_MARKER, intent["marker"])).issubset(
                {int(pid) for pid in keeper["generation"]}
            )
        ):
            raise RuntimeError("Original failed root exact restored keeper or inherited marker differs")
    result = {
        "status": STATUS,
        "original_status": state["status"],
        "original_failure": FAILURE,
        "terminal_time": state["time"],
        "evidence_sha256": evidence,
        "binding_sha256": BINDING_SHA256,
        "verifier_sha256": sha256(Path(__file__)),
        "gpu_commands": copy.deepcopy(receipt["gpu_commands"]),
        "analysis": corrected,
        "keeper": keeper,
        "controller_pid_absent": state["controller_pid"],
        "validator_pid_absent": state["child_pid"],
        "marked_descendants_absent": True,
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "scope": (
            "Preserved failed diagnostic only; no corrected training numerics, GPU execution or production release."
        ),
    }
    write(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = reconstruct(args.root.resolve(strict=True), args.out.resolve())
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
