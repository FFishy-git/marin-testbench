# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report saved actual-LR queue evidence without launching or reconstructing tensors."""

import subprocess
from pathlib import Path

import ns_lr_full_model_queue as queue
from selected_lr_release_owner import read, sha256

CPU = "run-preparation/ns-actual-lr-queue-cpu/20261119T0534/receipt.json"
CPU_SHA256 = "3b42f4992490e39e191f60ea94a1bc5b9131bc87683c5c1bb6cf68713ff06335"
INTENT_SHA256 = "eb656f9a934583d5ca6a8fbbb974a89b04a2f7096b5d206b0caaadd779ed20b1"
STATUSES = {
    "waiting_for_original_full_model_queue",
    "original_lower_lr_terminal_ready",
    "issuing_actual_lr_validation",
    "actual_lr_controller_acknowledged",
    "waiting_for_issued_actual_lr_owner",
    "actual_lr_terminal_ready",
    "actual_lr_terminal_reconstructed",
    "needs_attention",
}
DECISIONS = ("accepted_cases", "rejected_cases", "all_cases_accepted")


def evidence(root: Path) -> tuple[dict, dict, dict, dict]:
    """Authenticate saved source, acknowledgment, phase logs and terminal decisions."""
    cpu_path, work = root / CPU, root / queue.WORK
    if sha256(cpu_path) != CPU_SHA256:
        raise ValueError("NS actual-LR accepted queue receipt changed")
    cpu = queue.require_cpu(root, cpu_path)
    paths = [cpu_path, *(path for path in (work / "intent.json", work / "state.json") if path.is_file())]
    hashes = {path: sha256(path) for path in paths}
    intent, state = (read(path) if path.is_file() else {} for path in (work / "intent.json", work / "state.json"))
    if hashes[cpu_path] != CPU_SHA256:
        raise ValueError("NS actual-LR accepted receipt changed during reporting")
    if intent and (
        intent["status"] not in ("starting", "acknowledged", "needs_attention")
        or intent["command"] != queue.command(root, cpu_path)
        or intent["cpu_receipt"] != str(cpu_path)
        or intent["cpu_receipt_sha256"] != CPU_SHA256
        or intent["parent_cpu_receipt_sha256"] != queue.PARENT_SHA256
        or intent["runner_sha256"] != cpu["helpers_sha256"][queue.NEW_HELPERS[0]]
        or intent["interval"] != queue.INTERVAL
        or not intent["marker"]
    ):
        raise ValueError("NS actual-LR saved acknowledgment or source differs")
    if intent.get("status") == "acknowledged":
        generation = intent["generation"]
        if (
            hashes[work / "intent.json"] != INTENT_SHA256
            or generation["pid"] != intent["pid"]
            or generation["start_ticks"] != intent["start_ticks"]
            or generation["command"] != intent["command"]
            or generation["group"] != intent["pid"]
            or generation["session"] != intent["pid"]
        ):
            raise ValueError("NS actual-LR original queue generation differs")
    if state and (
        intent.get("status") != "acknowledged"
        or state["status"] not in STATUSES
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or any(
            state[name] != intent[name]
            for name in ("cpu_receipt_sha256", "parent_cpu_receipt_sha256", "runner_sha256", "interval")
        )
        or ("error" in state and state["status"] != "needs_attention")
        or len(state["phases"]) > 4
        or [entry["phase"] for entry in state["phases"]]
        != ["freeze", "check", "start", "terminal"][: len(state["phases"])]
    ):
        raise ValueError("NS actual-LR saved state or phase order differs")
    commands = {phase: (argv, cwd) for phase, argv, cwd in queue.phases(root)}
    helpers = queue.contract.locations(root)[1]
    commands["terminal"] = (
        [
            queue.contract.PYTHON,
            "-P",
            str(helpers / "ns_lr_full_model_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            "terminal",
            "--out",
            str(root / queue.TERMINAL),
        ],
        helpers,
    )
    for entry in state.get("phases", []):
        argv, cwd = commands[entry["phase"]]
        generation, log = entry["generation"], work / f"{entry['phase']}.log"
        if (
            entry["command"] != argv
            or entry["cwd"] != str(cwd)
            or entry["spawn_status"] != "acknowledged"
            or generation["pid"] != entry["pid"]
            or generation["start_ticks"] != entry["start_ticks"]
            or generation["parent"] != state["pid"]
            or generation["command"] != argv
            or generation["group"] != entry["pid"]
            or generation["session"] != entry["pid"]
            or any(type(entry[name]) is not int or entry[name] != 0 for name in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["log"] != str(log)
            or entry["log_sha256"] != sha256(log)
        ):
            raise ValueError("NS actual-LR completed phase command, ancestry or cleanup differs")
    terminal = {}
    if state.get("status") == "actual_lr_terminal_reconstructed":
        path = root / queue.TERMINAL
        paths.append(path)
        hashes[path] = sha256(path)
        terminal = read(path)
        if (
            len(state["phases"]) != 4
            or hashes[path] != state["terminal_sha256"]
            or state["terminal"] != str(path)
            or terminal["status"] != queue.controller.TERMINAL_STATUS
            or terminal["binding_sha256"] != state["issued_binding_sha256"]
            or any(state[name] != terminal[name] for name in DECISIONS)
            or terminal["controller_pid_absent"] != state["issued_controller"]["pid"]
            or terminal["verifier_sha256"] != sha256(helpers / "ns_lr_full_model_gpu_controller.py")
            or any(sha256(root / name) != digest for name, digest in terminal["evidence_sha256"].items())
        ):
            raise ValueError("NS actual-LR terminal origin or retained decisions differ")
    if hashes != {path: sha256(path) for path in paths}:
        raise RuntimeError("NS actual-LR queue evidence changed during reporting")
    return cpu, intent, state, terminal


def lines(root: Path, local_time) -> list[str]:
    if not (root / CPU).is_file():
        return []
    title = "### NS Actual-LR Validation Queue"
    try:
        cpu, intent, state, terminal = evidence(root)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as failure:
        return [title, "", f"Saved actual-LR queue evidence refused: `{type(failure).__name__}: {failure}`.", ""]
    result = [
        title,
        "",
        f"Saved separate queue CPU/source-archive acceptance at {local_time(cpu.get('time'))}; "
        f"{queue.TEST_COUNT} passing cases and {len(cpu['helpers_sha256'])} authenticated helpers.",
        f"Accepted owner: `PVC/{queue.PARENT}`.",
        "Original lower-LR queue/owner absence, independent terminal and exact keeper handoff remain required.",
        "Fifty-four actual-LR cases and 162 bounded eight-rank commands retain the original 5,120-update schedule.",
    ]
    if intent:
        result.append(
            f"Saved queue intent `{intent['status']}` at {local_time(intent.get('time'))}; "
            f"original-owner checks every {queue.INTERVAL} seconds."
        )
    if state:
        result.append(
            f"Saved queue `{state['status']}` at {local_time(state.get('time'))}; "
            f"{len(state['phases'])}/4 completed phases."
        )
        if state.get("error"):
            result.append(f"Preserved queue failure: `{state['error']}`.")
    if terminal:
        result.extend(
            [
                f"Saved terminal reconstruction at {local_time(terminal.get('time'))}; "
                f"{len(terminal['accepted_cases'])} accepted and {len(terminal['rejected_cases'])} rejected cases.",
                f"Saved all-case predicate: `{terminal['all_cases_accepted']}`.",
            ]
        )
    result.extend(
        [
            "Saved receipts do not establish current liveness or rederive tensors. Numerical rejection stays rejected.",
            "Bounded loss is continuation evidence. Full-budget waves, large shared-token comparisons and "
            "verified private SwanLab histories remain required. No production or efficiency claim.",
            "",
        ]
    )
    return result
