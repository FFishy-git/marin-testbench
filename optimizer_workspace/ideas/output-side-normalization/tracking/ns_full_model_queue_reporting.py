# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report authenticated full-model queue evidence without launching or interpreting tensors."""

import subprocess
from pathlib import Path

import ns_full_model_contract as contract
import ns_full_model_queue as queue
from selected_lr_release_owner import read, sha256

CPU = "run-preparation/ns-full-model-queue-cpu/20261119T0318/receipt.json"
CPU_SHA256 = "5c3722a9d740154d4b34ad9bc5aa7b40eda276354af4d0b9b9a1ced24f72123c"
INTENT_SHA256 = "d4679f0fdfe04a939a7d95f1ff2c4e57cdff282275d7660564ec50b95cd197d9"
STATUSES = {
    "waiting_for_original_ns_queue",
    "original_ns_terminal_ready",
    "issuing_full_model_validation",
    "full_model_controller_acknowledged",
    "waiting_for_issued_full_model_owner",
    "full_model_terminal_ready",
    "full_model_terminal_reconstructed",
    "needs_attention",
}


def evidence(root: Path) -> tuple[dict, dict, dict, dict]:
    """Authenticate saved commands, sources, logs and case decisions without process claims."""
    cpu_path, work = root / CPU, root / queue.WORK
    if sha256(cpu_path) != CPU_SHA256:
        raise ValueError("NS full-model accepted queue receipt changed")
    cpu = queue.require_cpu(root, cpu_path)
    paths = [cpu_path, *(path for path in (work / "intent.json", work / "state.json") if path.is_file())]
    hashes = {path: sha256(path) for path in paths}
    if hashes[cpu_path] != CPU_SHA256:
        raise ValueError("NS full-model accepted queue receipt changed during checking")
    intent, state = (read(path) if path.is_file() else {} for path in (work / "intent.json", work / "state.json"))
    if intent.get("status") == "acknowledged" and hashes[work / "intent.json"] != INTENT_SHA256:
        raise ValueError("NS full-model original queue acknowledgment changed")
    if intent and (
        intent["status"] not in ("starting", "acknowledged", "needs_attention")
        or intent["command"] != queue.command(root, cpu_path)
        or intent["cpu_receipt"] != str(cpu_path)
        or intent["cpu_receipt_sha256"] != hashes[cpu_path]
        or intent["parent_cpu_receipt_sha256"] != queue.PARENT_SHA256
        or intent["runner_sha256"] != cpu["helpers_sha256"][queue.NEW_HELPERS[0]]
        or intent["interval"] != queue.INTERVAL
        or not intent["marker"]
    ):
        raise ValueError("NS full-model saved queue acknowledgment or accepted source differs")
    if intent.get("status") == "acknowledged":
        generation = intent["generation"]
        if (
            generation["pid"] != intent["pid"]
            or generation["start_ticks"] != intent["start_ticks"]
            or generation["command"] != intent["command"]
            or generation["group"] != intent["pid"]
            or generation["session"] != intent["pid"]
        ):
            raise ValueError("NS full-model saved queue original generation differs")
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
        raise ValueError("NS full-model saved queue state or completed phase order differs")
    commands = {phase: (argv, cwd) for phase, argv, cwd in queue.phases(root)}
    helpers = contract.locations(root)[1]
    commands["terminal"] = (
        [
            contract.PYTHON,
            "-P",
            str(helpers / "ns_full_model_gpu_controller.py"),
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
            raise ValueError("NS full-model saved completed queue command/ancestry/cleanup differs")
    terminal = {}
    if state.get("status") == "full_model_terminal_reconstructed":
        path = root / queue.TERMINAL
        paths.append(path)
        hashes[path] = sha256(path)
        terminal = read(path)
        if (
            len(state["phases"]) != 4
            or hashes[path] != state["terminal_sha256"]
            or state["terminal"] != str(path)
            or terminal["status"] != "ns_full_model_terminal_independently_reconstructed_exact_restoration"
            or terminal["binding_sha256"] != state["issued_binding_sha256"]
            or any(state[name] != terminal[name] for name in ("accepted_cases", "rejected_cases", "all_cases_accepted"))
            or terminal["controller_pid_absent"] != state["issued_controller"]["pid"]
            or terminal["verifier_sha256"] != sha256(helpers / "ns_full_model_gpu_controller.py")
            or any(sha256(root / name) != digest for name, digest in terminal["evidence_sha256"].items())
        ):
            raise ValueError("NS full-model saved terminal origin or retained decisions differ")
    if hashes != {path: sha256(path) for path in paths}:
        raise RuntimeError("NS full-model queue evidence changed during reporting")
    return cpu, intent, state, terminal


def lines(root: Path, local_time) -> list[str]:
    if not (root / CPU).is_file():
        return []
    title = "### NS Full-Model Validation Queue"
    try:
        cpu, intent, state, terminal = evidence(root)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as failure:
        return [title, "", f"Saved full-model queue evidence refused: `{type(failure).__name__}: {failure}`.", ""]
    result = [
        title,
        "",
        f"Saved separate queue CPU/source-archive acceptance at {local_time(cpu.get('time'))}; "
        f"{queue.TEST_COUNT} passing cases and {len(cpu['helpers_sha256'])} authenticated helpers.",
        f"Accepted owner: `PVC/{queue.PARENT}`.",
        "Original NS queue/owner/worker absence, independent terminal evidence "
        "and exact keeper handoff remain required.",
        "Eighteen lower-LR cases and 54 eight-rank commands retain the original 5,120-update schedule.",
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
            "Bounded loss is continuation evidence, not optimizer quality. "
            "Fair full-budget LR waves, large shared-token "
            "comparisons and verified private SwanLab histories remain required. No production or efficiency claim.",
            "",
        ]
    )
    return result
