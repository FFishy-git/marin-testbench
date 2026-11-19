# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the preserved failed root diagnostic and its separate completed reconstruction."""

import os
import stat
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import owned_output_handoff as handoff
import root_preparation_contract as previous
import root_preparation_reconstruction as reconstruction
import root_preparation_validation as validation
from fused_attribution_validation import require_keeper_shape
from prepare_selected_lr_validation import require_archive
from root_preparation_controller import inventory
from selected_lr_failure_handoff import keeper_guard, same_keeper
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

CPU = "run-preparation/root-stage-replay-cpu/failed-owner-reconstruction-20261118T1615/receipt.json"
CPU_SHA256 = "aa4dd55db15ade6610dfdaac076974b085729c9dd115d986f319702fedc273a5"
RUNNER = "run-preparation/probes/run-root-failure-reconstruction-20261118T1618.py"
RUNNER_SHA256 = "aaf2ae178b982e4288cce445ce9bc9a6941fdd1828fc4f86e5fda658a83dae9f"
WORK = "run-preparation/root-failure-reconstruction-20261118T1618"
RESULT = "run-preparation/root-failed-diagnostic-reconstruction-20261118T1618.json"
VERIFIER_SHA256 = "88103539f8759c51ec9caadf2808c4f0331b6357e255a3d072dd071e49be8089"
RUNNER_MARKER = "OUTPUT_MARIN_ROOT_FAILED_READONLY_RECONSTRUCTION"
SEAL = "run-preparation/root-failed-reconstruction-terminal-seal-20261118.json"
SEAL_STATUS = "failed_root_independent_reconstruction_sealed_owners_absent_exact_restoration_diagnostic_only"
GATES = {
    "cold_capture_repeat_safety",
    "trace_transparency",
    "continuous_control_unchanged",
    "unprepared_original_reproduced",
    "prepared_all_rank_continuation",
}


def require_failed_terminal(state: dict, intent: dict) -> None:
    if (
        state.get("status") != "needs_attention"
        or state.get("binding_sha256") != reconstruction.BINDING_SHA256
        or state.get("error") != "RuntimeError: Root diagnostic validator failed"
        or any(type(state.get(key)) is not int or state[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or state.get("keeper_release_complete") is not True
        or state.get("marked_descendants_absent") is not True
        or "cleanup_error" in state
        or intent.get("status") != "acknowledged"
        or intent.get("binding_sha256") != reconstruction.BINDING_SHA256
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
    ):
        raise RuntimeError("Original failed root diagnostic cleanup or restoration is incomplete")


def require_reconstruction_cpu(root: Path, frozen: dict) -> dict:
    path = root / CPU
    cpu = read(path)
    if (
        sha256(path) != CPU_SHA256
        or cpu["status"] != "root_stage_replay_cpu_archive_verified_gpu_unissued"
        or cpu["root"] != str(root)
        or cpu["helpers"] != str(path.parent / "tracking")
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or "error" in cpu
        or cpu["helpers_sha256"]["root_preparation_reconstruction.py"] != VERIFIER_SHA256
        or any(cpu["helpers_sha256"].get(name) != value for name, value in frozen["helpers_sha256"].items())
        or any(sha256(Path(cpu["helpers"]) / name) != value for name, value in cpu["helpers_sha256"].items())
        or any(cpu[f"{name}_sha256"] != sha256(Path(cpu[name])) for name in ("archive", "log", "tests"))
        or any(sha256(path.parent / name) != value for name, value in cpu["fixtures_sha256"].items())
    ):
        raise ValueError("Failed-root reconstruction exact CPU/source acceptance differs")
    require_archive(
        Path(cpu["archive"]),
        {f"tracking/{name}": value for name, value in cpu["helpers_sha256"].items()} | cpu["fixtures_sha256"],
        alias=False,
    )
    suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        not cases
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("errors", "failures", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("error", "failure", "skipped"))
    ):
        raise ValueError("Failed-root reconstruction archived tests are incomplete")
    return cpu


def analysis_pairs(root: Path, analysis: dict) -> list[tuple[dict, Path, Path]]:
    outputs = {
        f"{regime}_{protocol}_{mode}": validation.directory(root, regime, protocol, mode)
        for regime in validation.REGIMES
        for protocol in validation.PROTOCOLS
        for mode in ("continuous", "resumed")
    }
    if (
        set(analysis["continuation"]) != set(validation.REGIMES)
        or any(set(value) != set(validation.PROTOCOLS) for value in analysis["continuation"].values())
        or set(analysis["transparency"]) != set(validation.REGIMES)
        or any(set(value) != {"continuous", "resumed"} for value in analysis["transparency"].values())
        or set(analysis["continuous_controls"]) != set(validation.PROTOCOLS)
        or set(analysis["unprepared_original_reproduction"])
        != {f"{protocol}_{mode}" for protocol in validation.PROTOCOLS for mode in ("continuous", "resumed")}
        or set(analysis["traces"])
        != {f"{regime}_{mode}" for regime in validation.REGIMES for mode in ("continuous", "resumed")}
        or set(analysis["trace_root_lifecycle"]) != set(analysis["traces"])
        or set(analysis["gates"]) != GATES
    ):
        raise ValueError("Failed-root reconstruction lacks complete original comparison coverage")
    pairs = [
        (
            analysis["continuation"][regime][protocol],
            outputs[f"{regime}_{protocol}_resumed"],
            outputs[f"{regime}_{protocol}_continuous"],
        )
        for regime in validation.REGIMES
        for protocol in validation.PROTOCOLS
    ]
    pairs.extend(
        (
            analysis["transparency"][regime][mode],
            outputs[f"{regime}_trace_{mode}"],
            outputs[f"{regime}_control_{mode}"],
        )
        for regime in validation.REGIMES
        for mode in ("continuous", "resumed")
    )
    pairs.extend(
        (
            analysis["continuous_controls"][protocol],
            outputs[f"prepared_{protocol}_continuous"],
            outputs[f"unprepared_{protocol}_continuous"],
        )
        for protocol in validation.PROTOCOLS
    )
    pairs.extend(
        (
            analysis["unprepared_original_reproduction"][f"{protocol}_{mode}"],
            outputs[f"unprepared_{protocol}_{mode}"],
            root / "validation/fused_continuation_20261118" / f"{protocol}_{mode}",
        )
        for protocol in validation.PROTOCOLS
        for mode in ("continuous", "resumed")
    )
    return pairs


def require_analysis(root: Path, analysis: dict, gpu: dict) -> dict[str, str]:
    """Check full comparison coverage and deduplicate authentication of retained tensor files."""
    tensor_hashes = {}
    for pair, actual, expected in analysis_pairs(root, analysis):
        if set(pair["checkpoints"]) != {str(step) for step in validation.STEPS}:
            raise ValueError("Failed-root checkpoint comparison lacks all updates")
        for step in validation.STEPS:
            checkpoint = pair["checkpoints"][str(step)]
            if checkpoint["step"] != step or [rank["rank"] for rank in checkpoint["ranks"]] != list(
                range(validation.WORLD)
            ):
                raise ValueError("Failed-root checkpoint comparison lacks all ranks")
            for rank in checkpoint["ranks"]:
                paths = {
                    str(directory / f"checkpoints/step_{step:06d}/rank_{rank['rank']}.pt")
                    for directory in (actual, expected)
                }
                if (
                    set(rank["checkpoint_sha256"]) != paths
                    or type(rank["tensor_comparisons"]) is not int
                    or rank["tensor_comparisons"] <= 0
                    or rank["failed_tensor_count"]
                    != sum(not item["within_tolerance"] for item in rank["changed_tensors"])
                ):
                    raise ValueError("Failed-root comparison tensor identity or failure counts differ")
                for name, value in rank["checkpoint_sha256"].items():
                    if name in tensor_hashes and tensor_hashes[name] != value:
                        raise ValueError("Failed-root comparison reuses changed checkpoint identities")
                    tensor_hashes[name] = value
    for regime in validation.REGIMES:
        for mode in ("continuous", "resumed"):
            key = f"{regime}_{mode}"
            reconstruction.require_trace_lifecycle(analysis["trace_root_lifecycle"][key], regime, mode)
            directory = validation.directory(root, regime, "trace", mode)
            expected_paths = {
                str(directory / f"continuation-trace/rank_{rank}/update_{step:06d}{suffix}")
                for rank in range(validation.WORLD)
                for step in validation.STEPS
                for suffix in (".pt", ".json")
            }
            hashes = analysis["traces"][key]["files_sha256"]
            if set(hashes) != expected_paths:
                raise ValueError("Failed-root trace comparison lacks complete source-bound coverage")
            tensor_hashes.update(hashes)
    gates = {
        "cold_capture_repeat_safety": analysis["cold_root"]["within_tolerance"],
        "trace_transparency": all(
            validation.pair_set_pass(value) for modes in analysis["transparency"].values() for value in modes.values()
        ),
        "continuous_control_unchanged": all(
            validation.pair_set_pass(value) for value in analysis["continuous_controls"].values()
        ),
        "unprepared_original_reproduced": all(
            validation.pair_set_pass(value) for value in analysis["unprepared_original_reproduction"].values()
        ),
        "prepared_all_rank_continuation": all(
            validation.pair_set_pass(value) for value in analysis["continuation"]["prepared"].values()
        ),
    }
    if (
        analysis["cold_root"] != gpu["cold_probe_evidence"]
        or analysis["gates"] != gates
        or any(type(value) is not bool for value in gates.values())
        or analysis["diagnostic_intervention_accepted"] is not all(gates.values())
    ):
        raise ValueError("Failed-root reconstruction predicates differ from retained comparisons")
    return tensor_hashes


def reconstructed_predecessor(root: Path) -> dict:
    """Require a completed separate reconstruction and absent owners before GPU handoff."""
    binding_path, helpers, state_path, _, gpu_path = previous.locations(root)
    failed, intent = read(state_path), read(helpers.parent / "controller.json")
    require_failed_terminal(failed, intent)
    work = root / WORK
    state = read(work / "state.json")
    if state.get("status") != "reconstructed":
        raise RuntimeError("Separate failed-root independent reconstruction is incomplete")
    acknowledgment = read(work / "intent.json")
    paths = (
        binding_path,
        state_path,
        gpu_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        Path(failed["log"]),
        root / CPU,
        root / RUNNER,
        work / "intent.json",
        work / "state.json",
        work / "runner.log",
        root / RESULT,
    )
    hashes = {str(path.relative_to(root)): sha256(path) for path in paths}
    frozen, gpu, result = read(binding_path), read(gpu_path), read(root / RESULT)
    command = [previous.PYTHON, "-P", str(root / RUNNER), "--phase", "execute"]
    if (
        hashes[str(binding_path.relative_to(root))] != reconstruction.BINDING_SHA256
        or hashes[str(state_path.relative_to(root))] != reconstruction.FAILED_STATE_SHA256
        or hashes[str(gpu_path.relative_to(root))] != reconstruction.FAILED_RECEIPT_SHA256
        or hashes[RUNNER] != RUNNER_SHA256
        or acknowledgment["status"] != "acknowledged"
        or acknowledgment["command"] != command
        or not acknowledgment["marker"]
        or acknowledgment["generation"]["command"] != command
        or acknowledgment["generation"]["pid"] != acknowledgment["pid"]
        or acknowledgment["generation"]["start_ticks"] != acknowledgment["start_ticks"]
        or acknowledgment["generation"]["group"] != acknowledgment["pid"]
        or acknowledgment["generation"]["session"] != acknowledgment["pid"]
        or acknowledgment["generation"]["status"] in ("Z", "X", "x")
        or state["pid"] != acknowledgment["pid"]
        or state["start_ticks"] != acknowledgment["start_ticks"]
        or state["command"] != command
        or any(
            entry["runner_sha256"] != RUNNER_SHA256
            or entry["binding_sha256"] != reconstruction.BINDING_SHA256
            or entry["cpu_receipt_sha256"] != CPU_SHA256
            or entry["out"] != str(root / RESULT)
            for entry in (state, acknowledgment)
        )
        or type(state["exit_code"]) is not int
        or state["exit_code"] != 0
        or "error" in state
        or state["result_status"] != reconstruction.STATUS
        or state["result_sha256"] != hashes[RESULT]
        or result["status"] != reconstruction.STATUS
        or result["original_status"] != failed["status"]
        or result["original_failure"] != reconstruction.FAILURE
        or result["terminal_time"] != failed["time"]
        or result["binding_sha256"] != reconstruction.BINDING_SHA256
        or result["verifier_sha256"] != VERIFIER_SHA256
        or result["controller_pid_absent"] != failed["controller_pid"]
        or result["validator_pid_absent"] != failed["child_pid"]
        or result["marked_descendants_absent"] is not True
        or not same_keeper(result["keeper"], failed["restored_keeper"])
        or result["tolerance"] != {"rtol": 1e-5, "atol": 1e-6}
        or result["gpu_commands"] != gpu["gpu_commands"]
        or gpu["status"] != "needs_attention"
        or gpu["error"] != reconstruction.FAILURE
        or gpu["execution_binding_sha256"] != reconstruction.BINDING_SHA256
        or gpu["owner"]
        != {"controller_pid": failed["controller_pid"], "controller_start_ticks": failed["controller_start_ticks"]}
        or intent["command"] != previous.controller_command(root)
        or failed["log_sha256"] != sha256(Path(failed["log"]))
        or set(result["evidence_sha256"]) != {str(path.relative_to(root)) for path in paths[:6]}
        or any(hashes[name] != value for name, value in result["evidence_sha256"].items())
        or len(gpu["gpu_commands"]) != 12
        or any(
            entry["command"] != command
            or type(entry["exit_code"]) is not int
            or entry["exit_code"] != 0
            or entry["log_sha256"] != sha256(Path(entry["log"]))
            for entry, command in zip(gpu["gpu_commands"], frozen["commands"], strict=True)
        )
        or frozen["files_sha256"] != candidate_files(Path(frozen["candidate"]))
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or any(sha256(helpers / name) != value for name, value in frozen["helpers_sha256"].items())
    ):
        raise ValueError("Failed-root predecessor source, original failure or independent reconstruction differs")
    if (
        any(handoff.observed(pid) is not None for pid in (failed["controller_pid"], failed["child_pid"], state["pid"]))
        or handoff.marker_processes(previous.CHILD_MARKER, failed["child_marker"])
        or handoff.marker_processes(RUNNER_MARKER, acknowledgment["marker"])
    ):
        raise RuntimeError("Original failed-root or reconstruction owners or descendants remain present")
    cpu = require_reconstruction_cpu(root, frozen)
    times = [datetime.fromisoformat(entry["time"]) for entry in (failed, cpu, acknowledgment, result, state)]
    if any(time.tzinfo is None for time in times) or times != sorted(times):
        raise ValueError("Failed-root predecessor terminal/reconstruction chronology differs")
    require_archive(Path(frozen["archive"]), inventory(frozen), alias=True)
    tensor_hashes = require_analysis(root, result["analysis"], gpu)
    require_keeper_shape(failed["restored_keeper"])
    if hashes != {str(path.relative_to(root)): sha256(path) for path in paths}:
        raise ValueError("Failed-root predecessor evidence changed during authentication")
    return {
        "status": "independent_failed_root_reconstructed_exact_restoration_pending_diagnostic_only_handoff",
        "evidence_sha256": hashes,
        "retained_tensor_sha256": tensor_hashes,
        "controller_pid": failed["controller_pid"],
        "child_pid": failed["child_pid"],
        "controller_marker": intent["marker"],
        "child_marker": failed["child_marker"],
        "restored_keeper": failed["restored_keeper"],
        "gates": result["analysis"]["gates"],
        "diagnostic_intervention_accepted": result["analysis"]["diagnostic_intervention_accepted"],
        "scope": (
            "Failed diagnostic reconstruction only. Production continuation and optimizer-quality gates are unchanged."
        ),
    }


def file_identity(path: Path) -> dict[str, int]:
    info = path.stat(follow_symlinks=False)
    if path != path.resolve(strict=True) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Failed-root seal requires canonical singly linked regular evidence")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "links": info.st_nlink,
    }


def seal(root: Path) -> dict:
    """Hash all retained evidence once after the independent reconstructor exits."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Failed-root terminal sealing requires hidden CUDA")
    out = root / SEAL
    if out.exists():
        raise FileExistsError("Failed-root terminal sealing already has immutable evidence")
    accepted = reconstructed_predecessor(root)
    identities = {}
    for name, expected in accepted["retained_tensor_sha256"].items():
        path = Path(name)
        identity = file_identity(path)
        if sha256(path) != expected or file_identity(path) != identity:
            raise ValueError("Failed-root seal retained checkpoint or trace bytes changed")
        identities[name] = identity
    with keeper_guard():
        keeper = handoff.keeper_observation()
        require_keeper_shape(keeper)
        if not same_keeper(keeper, accepted["restored_keeper"]) or not set(
            handoff.marker_processes(previous.CONTROLLER_MARKER, accepted["controller_marker"])
        ).issubset({int(pid) for pid in keeper["generation"]}):
            raise RuntimeError("Failed-root terminal seal exact restoration differs")
    if accepted != reconstructed_predecessor(root):
        raise ValueError("Failed-root reconstruction evidence changed during sealing")
    result = {
        "status": SEAL_STATUS,
        "verifier_sha256": sha256(Path(__file__)),
        "predecessor": accepted,
        "retained_file_identity": identities,
        "keeper": keeper,
        "scope": "Full retained-file digest sealing and exact restoration only. Production gates remain unchanged.",
    }
    write(out, result)
    return result


def predecessor(root: Path) -> dict:
    accepted = reconstructed_predecessor(root)
    path = root / SEAL
    raw, sealed = path.read_bytes(), read(path)
    if (
        sealed["status"] != SEAL_STATUS
        or sealed["verifier_sha256"] != sha256(Path(__file__))
        or sealed["predecessor"] != accepted
        or not same_keeper(sealed["keeper"], accepted["restored_keeper"])
        or set(sealed["retained_file_identity"]) != set(accepted["retained_tensor_sha256"])
        or datetime.fromisoformat(sealed["time"]).tzinfo is None
        or datetime.fromisoformat(sealed["time"]) < datetime.fromisoformat(read(root / WORK / "state.json")["time"])
        or any(file_identity(Path(name)) != identity for name, identity in sealed["retained_file_identity"].items())
    ):
        raise ValueError("Failed-root independent terminal seal or retained-file identity differs")
    if path.read_bytes() != raw:
        raise ValueError("Failed-root terminal seal changed during authentication")
    return accepted | {"terminal_seal_sha256": sha256(path)}
