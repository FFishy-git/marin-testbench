# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report authenticated saved full-model diagnostic status without CUDA execution."""

import csv
import subprocess
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import owned_output_handoff as handoff
import rational_continuation_contract as contract
from fused_attribution_validation_controller import process_context, require_helper_bytes
from prepare_rational_continuation import inventory, require_cpu
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

CPU = "run-preparation/rational-continuation-cpu/common-root-20261118T2116/receipt.json"
CPU_SHA256 = "37104b5d9afc53f536d1dc74568493d6fdfea91f3f9c5368ff801f4eb4ecbf63"
BINDING_SHA256 = "a1a7d4ebd88b563b5f80d36ba7e0033752315043b6b799714e08e3cb25e27ba3"
OBSERVATIONS = "run-preparation/monitor-rational-continuation"
FOLLOWUP_CPU = "run-preparation/rational-terminal-followup-cpu/20261118T2222/receipt.json"
FOLLOWUP_CPU_SHA256 = "ad5197a53a05b225c2ebc26193aa4008bbea55b5f4145fb288173f28a3c9830a"
FOLLOWUP_WORK = "run-preparation/rational-continuation-terminal-followup-20261118"
FOLLOWUP_INTENT_SHA256 = "6d79465415fe352bd559d498cc1340e01c99503a66bb494c54d2d4d92f2c06d7"
TERMINAL = "run-preparation/rational-continuation-terminal-independent-20261118.json"


def handle(expected: dict, cwd: Path, marker_name: str, marker: str) -> dict:
    """Distinguish a live original generation from an absent or exited recorded handle."""
    actual = handoff.observed(expected["pid"])
    if actual is None:
        return {"status": "absent", "expected": expected}
    if any(actual[key] != expected[key] for key in ("pid", "start_ticks", "command", "group", "session")):
        raise RuntimeError("Rational continuation observed a different recorded process generation")
    if actual["status"] in ("Z", "X", "x"):
        return {"status": "exited", "expected": expected, "observed": actual}
    verified = process_context(expected["pid"], expected["command"], cwd, marker_name, marker, contract.ENVIRONMENT)
    if verified["start_ticks"] != expected["start_ticks"]:
        raise RuntimeError("Rational continuation generation changed during context inspection")
    return {"status": "live_exact_generation", "expected": expected, "observed": verified}


def live_observation(root: Path) -> dict:
    """Record current processes and physical occupancy; never interpret tensor or production gates."""
    binding, helpers, state_path, _, result_path = contract.locations(root)
    frozen = read(binding)
    if sha256(binding) != BINDING_SHA256:
        raise ValueError("Rational continuation observation frozen binding changed")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    intent_path = helpers.parent / "controller.json"
    paths = [path for path in (binding, intent_path, state_path, result_path) if path.is_file()]
    identities = {str(path.relative_to(root)): sha256(path) for path in paths}
    intent, state, result = execution_evidence(root, frozen)
    owners = {}
    if intent.get("status") == "acknowledged":
        owners["controller"] = handle(intent["generation"], helpers, contract.CONTROLLER_MARKER, intent["marker"])
    if state.get("spawn_status") == "acknowledged":
        owners["validator"] = handle(
            state["child_generation"],
            Path(frozen["candidate"]),
            contract.CHILD_MARKER,
            state["child_marker"],
        )
    devices = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True, timeout=20
            ).splitlines(),
            skipinitialspace=True,
        )
    )
    if devices != frozen["allocation_predecessor"]["restored_keeper"]["devices"]:
        raise RuntimeError("Rational continuation observed different physical devices")
    occupied = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True, timeout=20
            ).splitlines(),
            skipinitialspace=True,
        )
    )
    if any(len(row) != 2 or row[1] not in {device[1] for device in devices} for row in occupied):
        raise ValueError("Rational continuation physical occupancy response differs")
    marked = handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"]) if state.get("child_marker") else {}
    active = state.get("active_command")
    if active and active not in frozen["commands"]:
        raise ValueError("Rational continuation observed an unplanned active command")
    launchers = [
        process
        for process in marked.values()
        if process["command"] == active and process["parent"] == state["child_pid"]
    ]
    ranks = [
        process
        for process in marked.values()
        if any(process["parent"] == launcher["pid"] for launcher in launchers)
        and process["command"][:5] == [contract.PYTHON, "-u", "-m", "experiment.train", "--config"]
    ]
    rank_pids = {process["pid"] for process in ranks}
    occupancy = [
        {"pid": int(pid), "gpu_uuid": uuid, "active_training_rank": int(pid) in rank_pids} for pid, uuid in occupied
    ]
    for name, owner in owners.items():
        if owner["status"] == "live_exact_generation" and not handoff.same_generation(
            owner["observed"], handoff.observed(owner["observed"]["pid"])
        ):
            raise RuntimeError(f"Rational continuation {name} changed during physical observation")
    if any(sha256(root / name) != digest for name, digest in identities.items()):
        raise RuntimeError("Rational continuation receipts changed during live observation")
    return {
        "status": "rational_continuation_read_only_live_observation",
        "binding_sha256": BINDING_SHA256,
        "source_sha256": sha256(Path(__file__)),
        "evidence_sha256": identities,
        "saved_state": state.get("status"),
        "saved_result_status": result.get("status"),
        "retained_commands": len(result.get("gpu_commands", [])),
        "owners": owners,
        "active_command": active,
        "active_launchers": launchers,
        "active_ranks": ranks,
        "marked_process_count": len(marked),
        "devices": devices,
        "occupied": occupancy,
        "scope": (
            "One read-only process/device observation; no numerical, restoration, quality or production acceptance."
        ),
    }


def save_live_observation(root: Path, now: datetime) -> dict:
    if not contract.locations(root)[0].is_file():
        return {"status": "rational_continuation_not_issued"}
    observation = live_observation(root)
    directory = root / OBSERVATIONS
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now.strftime('%Y%m%dT%H%M%S%f')}.json"
    if path.exists():
        raise FileExistsError("Rational continuation observation already exists")
    write(path, observation)
    return {"status": observation["status"], "path": str(path), "retained_commands": observation["retained_commands"]}


def optional(path: Path) -> dict:
    return read(path) if path.is_file() else {}


def followup_evidence(root: Path) -> tuple[dict, dict, dict, dict]:
    """Authenticate the separate read-only owner and retain failed or pending states."""
    work, cpu_path = root / FOLLOWUP_WORK, root / FOLLOWUP_CPU
    if not (work / "intent.json").is_file():
        return {}, {}, {}, {}
    paths = [cpu_path, work / "intent.json", work / "state.json"]
    hashes = {path: sha256(path) for path in paths}
    cpu, intent, state = (read(path) for path in paths)
    files = cpu["files_sha256"]
    suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    runner = cpu_path.parent / "rational_continuation_followup.py"
    command = [
        contract.PYTHON,
        "-P",
        str(runner),
        "--root",
        str(root),
        "--cpu",
        str(cpu_path),
        "--phase",
        "execute",
    ]
    if (
        hashes[cpu_path] != FOLLOWUP_CPU_SHA256
        or hashes[work / "intent.json"] != FOLLOWUP_INTENT_SHA256
        or cpu["status"] != "rational_continuation_read_only_followup_cpu_archive_verified"
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or cpu["test_count"] != len(cases)
        or len(cases) != 8
        or sum(int(suite.attrib["tests"]) for suite in suites) != 8
        or any(int(suite.attrib[name]) for suite in suites for name in ("failures", "errors", "skipped"))
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
        or any(sha256(cpu_path.parent / name) != digest for name, digest in files.items())
        or any(sha256(Path(cpu[name])) != cpu[f"{name}_sha256"] for name in ("archive", "tests", "log"))
        or intent["status"] != "acknowledged"
        or intent["command"] != command
        or intent["generation"]["command"] != command
        or intent["generation"]["pid"] != intent["pid"]
        or intent["generation"]["start_ticks"] != intent["start_ticks"]
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or state["status"]
        not in (
            "waiting_for_original_owners",
            "terminal_ready",
            "reconstructing_terminal",
            "terminal_reconstructed",
            "needs_attention",
        )
        or any(
            item["binding_sha256"] != BINDING_SHA256
            or item["cpu_receipt_sha256"] != FOLLOWUP_CPU_SHA256
            or item["runner_sha256"] != files[runner.name]
            or type(item["interval"]) is not int
            or item["interval"] != 900
            for item in (intent, state)
        )
    ):
        raise ValueError("Read-only rational follow-up saved owner/source/CPU evidence changed")
    require_archive(Path(cpu["archive"]), files, alias=False)
    terminal = {}
    if state.get("terminal_command"):
        helpers = contract.locations(root)[1]
        expected = [
            contract.PYTHON,
            "-P",
            str(helpers / "rational_continuation_controller.py"),
            "--root",
            str(root),
            "--phase",
            "terminal",
            "--out",
            str(root / TERMINAL),
        ]
        if state["terminal_command"] != expected:
            raise ValueError("Read-only rational follow-up retargeted the issued terminal verifier")
    if state["status"] == "terminal_reconstructed":
        path = root / TERMINAL
        terminal = read(path)
        paths.extend((path, Path(state["terminal_log"])))
        hashes.update({path: sha256(path) for path in paths if path not in hashes})
        _, helpers, original_state, _, result = contract.locations(root)
        original = read(original_state)
        if (
            not state.get("terminal_command")
            or type(state["terminal_exit_code"]) is not int
            or state["terminal_exit_code"] != 0
            or type(state["terminal_cleanup_exit_code"]) is not int
            or state["terminal_cleanup_exit_code"] != 0
            or state["terminal_descendants_absent"] is not True
            or state["terminal_sha256"] != hashes[path]
            or state["terminal_log"] != str(work / "terminal.log")
            or state["terminal_log_sha256"] != hashes[Path(state["terminal_log"])]
            or terminal["status"] != "rational_continuation_terminal_independently_reconstructed_exact_restoration"
            or terminal["status"] != state["terminal_status"]
            or terminal["binding_sha256"] != BINDING_SHA256
            or terminal["result_sha256"] != sha256(result)
            or terminal["terminal_time"] != original["time"]
            or terminal["controller_pid_absent"] != original["controller_pid"]
            or terminal["validator_pid_absent"] != original["child_pid"]
            or terminal["analysis"] != read(result)["analysis"]
            or state["intervention_accepted"] is not terminal["analysis"]["diagnostic_intervention_accepted"]
            or terminal["verifier_sha256"] != sha256(helpers / "rational_continuation_controller.py")
            or any(sha256(root / name) != digest for name, digest in terminal["evidence_sha256"].items())
            or "error" in state
        ):
            raise ValueError("Read-only rational terminal result or exact cleanup receipt changed")
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Read-only rational follow-up evidence changed during reporting")
    return cpu, intent, state, terminal


def evidence(root: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Authenticate saved source and command receipts without interpreting tensor arithmetic."""
    cpu_path = root / CPU
    cpu = require_cpu(root, cpu_path)
    binding, helpers, _, _, _ = contract.locations(root)
    frozen = read(binding)
    candidate = Path(cpu["candidate"])
    prepared = read(candidate / contract.PREPARATION)
    if (
        sha256(cpu_path) != CPU_SHA256
        or sha256(binding) != BINDING_SHA256
        or frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["cpu_receipt"] != str(cpu_path)
        or frozen["cpu_receipt_sha256"] != CPU_SHA256
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != cpu["helpers_sha256"]
        or frozen["files_sha256"] != cpu["files_sha256"]
        or frozen["preparation_sha256"] != cpu["preparation_sha256"]
        or frozen["allocation_predecessor"] != prepared["allocation_predecessor"]
        or frozen["environment"] != contract.ENVIRONMENT
        or frozen["plan"] != contract.plan(candidate, root)
        or frozen["commands"] != frozen["plan"]["commands"]
        or frozen["controller_command"] != contract.controller_command(root)
        or frozen["validator_command"] != contract.validator_command(root, candidate)
        or frozen["archive"] != str(root / "run-preparation/source-rational-continuation-20261118.tar")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("Rational continuation reporting frozen source or CPU evidence differs")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_archive(Path(frozen["archive"]), inventory(prepared), alias=True)
    intent, state, result = execution_evidence(root, frozen)
    return cpu, frozen, intent, state, result


def execution_evidence(root: Path, frozen: dict) -> tuple[dict, dict, dict]:
    """Validate ordered saved command outcomes and their original execution owner."""
    _, helpers, state_path, work, result_path = contract.locations(root)
    paths = [path for path in (helpers.parent / "controller.json", state_path, result_path) if path.is_file()]
    hashes = {path: sha256(path) for path in paths}
    intent, state, result = (optional(path) for path in (helpers.parent / "controller.json", state_path, result_path))
    if intent and (
        intent["binding_sha256"] != BINDING_SHA256
        or intent["command"] != frozen["controller_command"]
        or intent["status"] not in ("starting", "acknowledged", "needs_attention")
        or (
            intent["status"] == "acknowledged"
            and (
                intent["generation"]["pid"] != intent["pid"]
                or intent["generation"]["start_ticks"] != intent["start_ticks"]
                or intent["generation"]["command"] != intent["command"]
            )
        )
    ):
        raise ValueError("Rational continuation saved controller intent differs")
    if state and (
        intent.get("status") != "acknowledged"
        or state["status"] not in ("preflight", "validating", "diagnostic_complete", "needs_attention")
        or state["binding_sha256"] != BINDING_SHA256
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["allocation_predecessor"] != frozen["allocation_predecessor"]
        or (
            state.get("spawn_status") == "acknowledged"
            and (
                state["command"] != frozen["validator_command"]
                or state["child_generation"]["parent"] != state["controller_pid"]
                or state["child_generation"]["pid"] != state["child_pid"]
                or state["child_generation"]["start_ticks"] != state["child_start_ticks"]
                or state["child_generation"]["command"] != state["command"]
            )
        )
    ):
        raise ValueError("Rational continuation saved controller/validator ancestry differs")
    if result:
        expected = frozen["plan"]
        commands = result["gpu_commands"]
        if (
            not state
            or result["status"]
            not in ("diagnosing", "needs_attention", "completed_rational_start_full_model_diagnostic_not_production")
            or result["binding_sha256"] != BINDING_SHA256
            or result["owner"]
            != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
            or any(result.get(key) != value for key, value in expected.items() if key != "status")
            or len(commands) > len(expected["commands"])
        ):
            raise ValueError("Rational continuation saved diagnostic plan or owner differs")
        for entry, coordinate, command in zip(commands, contract.coordinates(), expected["commands"], strict=False):
            variant, protocol, mode = coordinate
            log = work / f"{variant}_{protocol}_{mode}.log"
            if (
                (entry["variant"], entry["protocol"], entry["mode"]) != coordinate
                or entry["command"] != command
                or type(entry["exit_code"]) is not int
                or entry["log"] != str(log)
                or log.is_symlink()
                or log.stat().st_nlink != 1
                or entry["log_sha256"] != sha256(log)
            ):
                raise ValueError("Rational continuation saved command outcome or retained log differs")
    if state.get("status") == "diagnostic_complete" and (
        result.get("status") != "completed_rational_start_full_model_diagnostic_not_production"
        or len(result["gpu_commands"]) != len(contract.coordinates())
        or any(entry["exit_code"] != 0 for entry in result["gpu_commands"])
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state.get("marked_descendants_absent") is not True
        or state.get("keeper_release_complete") is not True
        or state["result_sha256"] != sha256(result_path)
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
    ):
        raise ValueError("Rational continuation saved completion lacks commands, cleanup or restoration")
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("Rational continuation saved state changed during reporting")
    return intent, state, result


def lines(root: Path, local_time) -> list[str]:
    if not contract.locations(root)[0].is_file():
        return []
    cpu, frozen, intent, state, result = evidence(root)
    current = [
        "### Rational-Start Full-Model Continuation Diagnostic",
        "",
        f"Accepted CPU/source archive: `PVC/{CPU}` at {local_time(cpu['time'])}; "
        "215 passing tests, 112 helpers and six fixtures.",
        f"Saved freeze at {local_time(frozen['time'])}; pinned/rational x control/trace, twelve eight-rank commands.",
        "Seed 0, LR 0.04525483392, 524,288 tokens/update and original 5,120-update schedule; "
        "bounded 22/23/24 trajectories.",
    ]
    if intent:
        current.append(f"Saved controller intent `{intent['status']}` at {local_time(intent['time'])}.")
    if state:
        current.append(f"Saved controller state `{state['status']}` at {local_time(state['time'])}.")
        if "active_command" in state:
            command = state["active_command"]
            target = command[command.index("--out") + 1]
            current.append(f"Saved active training command targets `{target}`.")
        if "keeper_restore" in state:
            current.append(f"Recorded keeper restoration: `{state['keeper_restore']}`.")
        if "error" in state:
            current.append(f"Preserved controller failure: `{state['error']}`.")
    if result:
        commands = result["gpu_commands"]
        successes = sum(entry["exit_code"] == 0 for entry in commands)
        current.append(f"Retained command outcomes: {len(commands)}/12; {successes} integer-zero exits.")
        if "error" in result:
            current.append(f"Preserved diagnostic failure: `{result['error']}`.")
        if "analysis" in result:
            for name, value in result["analysis"]["gates"].items():
                current.append(f"Saved diagnostic predicate `{name}`: `{value}`.")
    _, _, followup, terminal = followup_evidence(root)
    if followup:
        current.append(
            f"Saved separate read-only follow-up `{followup['status']}` at {local_time(followup['time'])}; "
            "original-owner checks every 900 seconds."
        )
        if terminal:
            current.append(
                f"Saved independently reconstructed terminal: `PVC/{TERMINAL}`; "
                f"intervention accepted `{terminal['analysis']['diagnostic_intervention_accepted']}`."
            )
        if "error" in followup:
            current.append(f"Preserved read-only follow-up failure: `{followup['error']}`.")
    current.extend(
        [
            "Receipt-backed state does not establish current liveness or tensor arithmetic. Independent retained-result "
            "and exact-restoration reconstruction remain required. Compilation, CUDA graphs "
            "and original tolerances stay "
            "enabled. This bounded diagnostic supplies no production authorization, full-budget quality, "
            "efficiency or SwanLab import.",
            "",
        ]
    )
    return current
