# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render accepted CPU and saved read-only follow-up evidence without numerical or live claims."""

from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import read, sha256

CPU = "run-preparation/root-stage-precision-cpu/20261118T1714/receipt.json"
CPU_SHA256 = "9e81ed1dbf4b8a26c61b4b8c46b662468f65514e0e26de97b186adf663b44584"
FOLLOW_CPU = "run-preparation/root-stage-followup-cpu/20261118T1723/receipt.json"
FOLLOW_CPU_SHA256 = "b4aae671b09ad778fe4a721024deb6beec6df2a1b943c6416a331e538e96004f"
FOLLOW_WORK = "run-preparation/root-stage-terminal-followup-20261118T1718"
QUEUE_INTENT = "run-preparation/root-stage-replay-queue-20261118T1642/intent.json"
QUEUE_INTENT_SHA256 = "8cc051299c5fbf4611f4396fc375a873062be984a53e389d3c4082c6cdf517be"
TERMINAL = "run-preparation/root-stage-terminal-independently-reconstructed-20261118T1718.json"
PRECISION = "run-preparation/root-stage-precision-independent-20261118T1718.json"
STATUSES = (
    "waiting_for_queue",
    "waiting_for_diagnostic",
    "terminal_reconstruction_ready",
    "reconstructing_terminal_outputs",
    "inspecting_saved_precision",
    "terminal_reconstructed_precision_denied",
    "terminal_and_precision_reconstructed",
    "needs_attention",
)


def cpu_evidence(root: Path, path: Path, expected_hash: str, status: str, count: int) -> dict:
    receipt = read(path)
    if (
        sha256(path) != expected_hash
        or receipt["status"] != status
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or any(receipt[f"{name}_sha256"] != sha256(Path(receipt[name])) for name in ("archive", "log", "tests"))
    ):
        raise ValueError("Root precision saved CPU execution or receipt bytes differ")
    if "helpers_sha256" in receipt:
        helpers = Path(receipt["helpers"])
        if helpers != path.parent / "tracking" or receipt["root"] != str(root):
            raise ValueError("Root precision CPU helper path differs")
        inventory = {f"tracking/{name}": value for name, value in receipt["helpers_sha256"].items()}
        inventory |= receipt["fixtures_sha256"]
    else:
        inventory = receipt["files_sha256"]
        if receipt["runner"] != str(path.parent / "follow-root-stage-terminal-20261118T1718.py"):
            raise ValueError("Root precision follow-up runner path differs")
    if any(sha256(path.parent / name) != value for name, value in inventory.items()):
        raise ValueError("Root precision archived source or fixture bytes differ")
    require_archive(Path(receipt["archive"]), inventory, alias=False)
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        len(cases) != count
        or sum(int(suite.attrib["tests"]) for suite in suites) != count
        or any(int(suite.attrib[name]) for suite in suites for name in ("errors", "failures", "skipped"))
        or any(case.find(name) is not None for case in cases for name in ("error", "failure", "skipped"))
    ):
        raise ValueError("Root precision archived CPU tests lack complete passing coverage")
    return receipt


def evidence(root: Path) -> tuple[dict, dict, dict, dict, dict, dict]:
    cpu_path, follow_cpu_path = root / CPU, root / FOLLOW_CPU
    work = root / FOLLOW_WORK
    paths = [cpu_path, follow_cpu_path, work / "intent.json", work / "state.json", root / QUEUE_INTENT]
    hashes = {path: sha256(path) for path in paths}
    cpu = cpu_evidence(root, cpu_path, CPU_SHA256, "root_stage_precision_cpu_archive_verified_cuda_analysis_pending", 76)
    follow_cpu = cpu_evidence(
        root, follow_cpu_path, FOLLOW_CPU_SHA256, "root_stage_terminal_followup_cpu_archive_verified", 6
    )
    intent, state = read(work / "intent.json"), read(work / "state.json")
    command = ["/opt/conda/envs/torch-base/bin/python", "-P", follow_cpu["runner"], "--phase", "execute"]
    if (
        hashes[root / QUEUE_INTENT] != QUEUE_INTENT_SHA256
        or follow_cpu["precision_cpu_receipt_sha256"] != CPU_SHA256
        or intent["status"] != "acknowledged"
        or intent["command"] != command
        or intent["pid"] != intent["generation"]["pid"]
        or intent["start_ticks"] != intent["generation"]["start_ticks"]
        or intent["generation"]["command"] != command
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or intent["generation"]["status"] in ("Z", "X", "x")
        or not intent["marker"]
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or state["status"] not in STATUSES
        or any(
            record["runner_sha256"] != follow_cpu["runner_sha256"]
            or record["cpu_receipt_sha256"] != CPU_SHA256
            or record["queue_intent_sha256"] != QUEUE_INTENT_SHA256
            or type(record["interval_seconds"]) is not int
            or record["interval_seconds"] != 900
            for record in (intent, state)
        )
    ):
        raise ValueError("Root precision saved follow-up acknowledgment or source identity differs")
    commands = state["commands"]
    if (
        [entry["phase"] for entry in commands] != ["terminal", "precision"][: len(commands)]
        or len(commands) > 2
        or (state["status"] in STATUSES[:3] and commands)
        or ("error" in state and state["status"] != "needs_attention")
    ):
        raise ValueError("Root precision follow-up phase order or failure state differs")
    terminal, precision = {}, {}
    if state.get("terminal_sha256"):
        path = root / TERMINAL
        paths.append(path)
        hashes[path] = sha256(path)
        terminal = read(path)
        if (
            hashes[path] != state["terminal_sha256"]
            or terminal["status"] != state["terminal_status"]
            or terminal["status"] != "root_stage_replay_terminal_independently_reconstructed_exact_restoration_verified"
        ):
            raise ValueError("Root precision saved terminal result identity differs")
    for index, entry in enumerate(commands):
        issued = root / "controllers/root-stage-replay-20261118/tracking"
        if entry["phase"] == "terminal":
            argv = [
                command[0],
                "-P",
                str(issued / "root_stage_gpu_controller.py"),
                "--root",
                str(root),
                "--phase",
                "terminal",
                "--out",
                str(root / TERMINAL),
            ]
        else:
            argv = [
                command[0],
                "-P",
                str(Path(cpu["helpers"]) / "analyze_root_stage_precision.py"),
                "--root",
                str(root),
                "--terminal",
                str(root / TERMINAL),
                "--terminal-sha256",
                state["terminal_sha256"],
                "--out",
                str(root / PRECISION),
            ]
        if (
            entry["command"] != argv
            or entry["status"] not in ("starting", "running", "complete", "needs_attention")
            or (index == 1 and commands[0]["status"] != "complete")
            or (
                entry["status"] == "complete"
                and (
                    any(type(entry[name]) is not int or entry[name] != 0 for name in ("exit_code", "cleanup_exit_code"))
                    or entry["marked_descendants_absent"] is not True
                )
            )
        ):
            raise ValueError("Root precision follow-up phase source or cleanup differs")
        if "log_sha256" in entry:
            path = work / f"{entry['phase']}.log"
            paths.append(path)
            hashes[path] = sha256(path)
            if entry["log"] != str(path) or entry["log_sha256"] != hashes[path]:
                raise ValueError("Root precision read-only phase log differs")
        elif entry["status"] == "complete":
            raise ValueError("Root precision completed phase lacks retained log")
    if state["status"] == "terminal_and_precision_reconstructed":
        path = root / PRECISION
        paths.append(path)
        hashes[path] = sha256(path)
        precision = read(path)
        if (
            len(commands) != 2
            or any(entry["status"] != "complete" for entry in commands)
            or hashes[path] != state["precision_sha256"]
            or precision["terminal_sha256"] != state["terminal_sha256"]
            or precision["status"] != state["precision_status"]
            or precision["analyzer_sha256"] != cpu["helpers_sha256"]["analyze_root_stage_precision.py"]
            or terminal["intermediates_interpretable"] is not True
            or precision["historical_interpretation_allowed"]
            is not precision["historical_outputs_reproduced_within_original_tolerance"]
        ):
            raise ValueError("Root precision saved inspection result/source or terminal predicate differs")
    elif state["status"] == "terminal_reconstructed_precision_denied":
        if len(commands) != 1 or not terminal or terminal["intermediates_interpretable"] is not False:
            raise ValueError("Root precision denial lacks the failed transparency predicate")
    times = [datetime.fromisoformat(record["time"]) for record in (cpu, follow_cpu, intent, state)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("Root precision saved CPU/owner chronology differs")
    if hashes != {path: sha256(path) for path in paths}:
        raise ValueError("Root precision evidence changed during reporting")
    return cpu, follow_cpu, intent, state, terminal, precision


def lines(root: Path, local_time) -> list[str]:
    if not (root / FOLLOW_WORK / "intent.json").is_file():
        return []
    current = ["### Root-Stage Terminal And Precision Follow-Up", "", f"`PVC/{FOLLOW_WORK}`."]
    cpu, follow_cpu, intent, state, terminal, precision = evidence(root)
    current.append(
        f"Saved precision CPU acceptance at {local_time(cpu['time'])}: 76 tests, "
        f"{len(cpu['helpers_sha256'])} helpers and {len(cpu['fixtures_sha256'])} fixtures."
    )
    current.append(f"Saved read-only follow-up CPU acceptance at {local_time(follow_cpu['time'])}: six tests.")
    current.append(
        f"Saved follow-up acknowledgment at {local_time(intent['time'])}: PID `{intent['pid']}/{intent['start_ticks']}`."
    )
    current.append(f"Saved follow-up state `{state['status']}` at {local_time(state['time'])}; 900-second observations.")
    for entry in state["commands"]:
        current.append(
            f"Read-only phase `{entry['phase']}`: `{entry['status']}`, "
            f"recorded exit `{entry.get('exit_code', 'pending')}`."
        )
    if terminal:
        current.append(f"Saved terminal reconstruction: `PVC/{TERMINAL}`; retained-output/restoration receipt bound.")
    if precision:
        current.append(
            f"Saved precision inspection: `PVC/{PRECISION}`; historical-output reproduction predicate "
            f"`{precision['historical_outputs_reproduced_within_original_tolerance']}`."
        )
    if "error" in state:
        current.append(f"Preserved read-only follow-up failure: `{state['error']}`.")
    current.append(
        "Saved acknowledgment is not current liveness. This renderer does not independently rederive numerical "
        "comparison, cast semantics or restoration. No optimizer result, efficiency "
        "or production acceptance is inferred."
    )
    return [*current, ""]
