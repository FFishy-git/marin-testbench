# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report archived actual-LR training-path queue receipts without launch or tensor replay."""

import subprocess
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_queue as queue
from ns_training_path_predecessor import accepted_archive
from selected_lr_release_owner import read, sha256

CPU = "run-preparation/ns-training-path-actual-lr-queue-cpu/20261119T1116/receipt.json"
CPU_SHA256 = "2830b671656c1ccf77e3923b220fb753e5d8b69a833d67fafc26ea14f1ecce1d"
INTENT_SHA256 = "f0c270d123da8fd8f53e2b59efe7bdbd9b95003fa17c69509fa44f6b01359256"
PHASES = ("seal", "check-seal", "freeze", "check", "start", "terminal")
STATUSES = {
    "waiting_for_original_training_path_owners",
    "original_training_path_terminal_ready",
    "sealing_original_training_path_terminal",
    "actual_lr_training_path_controller_acknowledged",
    "waiting_for_issued_actual_lr_training_path_owners",
    "actual_lr_training_path_terminal_ready",
    "actual_lr_training_path_terminal_reconstructed",
    "needs_attention",
}
DECISIONS = ("accepted_cases", "rejected_cases", "all_cases_accepted")


def phase_command(root: Path, phase: str, state: dict) -> tuple[list[str], Path]:
    """Describe the archived command contract without invoking a readiness checker."""
    inherited, issued = (root / queue.PARENT).parent / "tracking", queue.contract.locations(root)[1]
    if phase in ("seal", "check-seal"):
        helpers = (root / queue.owner.PREDECESSOR_RECEIPT).parent / "tracking"
        argv = [
            queue.contract.PYTHON,
            "-P",
            str(helpers / "prepare_ns_training_path_lr_predecessor.py"),
            "--root",
            str(root),
            "--phase",
            phase,
            "--cpu-receipt",
            str(root / queue.owner.PREDECESSOR_RECEIPT),
        ]
        if phase == "check-seal":
            argv.extend(("--seal-sha256", state["predecessor_seal_sha256"]))
        return argv, helpers
    helpers = inherited if phase == "freeze" else issued
    argv = [
        queue.contract.PYTHON,
        "-P",
        str(helpers / "ns_training_path_lr_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        phase,
    ]
    if phase == "freeze":
        argv.extend(("--cpu-receipt", str(root / queue.PARENT), "--seal-sha256", state["predecessor_seal_sha256"]))
    if phase == "terminal":
        output = queue.controller.terminal_paths(root, state["issued_controller"])[0]
        argv.extend(("--out", str(output)))
    return argv, helpers


def require_phase(root: Path, state: dict, entry: dict) -> None:
    argv, cwd = phase_command(root, entry["phase"], state)
    generation, log = entry["generation"], root / queue.WORK / f"{entry['phase']}.log"
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
        or "cleanup_error" in entry
        or entry["log"] != str(log)
        or entry["log_sha256"] != sha256(log)
    ):
        raise ValueError("Training-path actual-LR saved phase command, ancestry or cleanup differs")


def evidence(root: Path) -> tuple[dict, dict, dict, dict]:
    """Authenticate saved archive, original acknowledgement, phase logs and terminal results."""
    cpu_path, work = root / CPU, root / queue.WORK
    cpu = accepted_archive(root, (CPU, CPU_SHA256, queue.CPU_STATUS, queue.TEST_COUNT))
    if (
        cpu["parent_cpu_receipt_sha256"] != queue.PARENT_SHA256
        or cpu["command"] != queue.cpu_command(cpu_path.parent)
        or cpu["environment"] != queue.ENVIRONMENT
        or cpu["scope"] != queue.SCOPE
        or cpu["helpers_sha256"][queue.NEW_HELPERS[0]] != sha256(Path(queue.__file__))
    ):
        raise ValueError("Training-path actual-LR reporter queue source or archive origin differs")
    paths = [cpu_path, *(path for path in (work / "intent.json", work / "state.json") if path.is_file())]
    hashes = {path: sha256(path) for path in paths}
    intent, state = (read(path) if path.is_file() else {} for path in (work / "intent.json", work / "state.json"))
    if hashes[cpu_path] != CPU_SHA256:
        raise ValueError("Training-path actual-LR CPU receipt changed during reporting")
    if intent and (
        intent["status"] not in ("starting", "acknowledged", "needs_attention")
        or intent["command"] != queue.command(root, cpu_path)
        or intent["cpu_receipt"] != str(cpu_path)
        or intent["cpu_receipt_sha256"] != CPU_SHA256
        or intent["parent_cpu_receipt_sha256"] != queue.PARENT_SHA256
        or intent["runner_sha256"] != cpu["helpers_sha256"][queue.NEW_HELPERS[0]]
        or intent["interval"] != queue.INTERVAL
        or intent["scope"] != queue.SCOPE
        or not intent["marker"]
    ):
        raise ValueError("Training-path actual-LR saved queue acknowledgement differs")
    if intent.get("status") == "acknowledged":
        own = intent["generation"]
        if (
            hashes[work / "intent.json"] != INTENT_SHA256
            or own["pid"] != intent["pid"]
            or own["start_ticks"] != intent["start_ticks"]
            or own["command"] != intent["command"]
            or own["group"] != intent["pid"]
            or own["session"] != intent["pid"]
        ):
            raise ValueError("Training-path actual-LR original queue generation changed")
    if state and (
        intent.get("status") != "acknowledged"
        or state["status"] not in STATUSES
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or any(
            state[name] != intent[name]
            for name in ("cpu_receipt_sha256", "parent_cpu_receipt_sha256", "runner_sha256", "interval", "scope")
        )
        or ("error" in state and state["status"] != "needs_attention")
        or len(state["phases"]) > len(PHASES)
        or [entry["phase"] for entry in state["phases"]] != list(PHASES[: len(state["phases"])])
    ):
        raise ValueError("Training-path actual-LR saved state or phase order differs")
    for entry in state.get("phases", []):
        require_phase(root, state, entry)
    if state.get("original_observations"):
        path = work / f"original-observation-{state['original_observations'] - 1:06d}.json"
        paths.append(path)
        hashes[path] = sha256(path)
        observed = read(path)
        if any(state["original_observation"].get(name) != value for name, value in observed.items() if name != "time"):
            raise ValueError("Training-path actual-LR saved original-owner observation changed")
    if state.get("predecessor_seal_sha256"):
        path = root / queue.predecessor.SEAL
        paths.append(path)
        hashes[path] = sha256(path)
        sealed = read(path)
        if (
            hashes[path] != state["predecessor_seal_sha256"]
            or sealed["original_decisions"] != state["original_decisions"]
        ):
            raise ValueError("Training-path actual-LR predecessor seal or original rejections changed")
    terminal = {}
    if state.get("status") == "actual_lr_training_path_terminal_reconstructed":
        path, attempt, outcome, _ = queue.controller.terminal_paths(root, state["issued_controller"])
        paths.extend((path, attempt, outcome))
        hashes.update({item: sha256(item) for item in (path, attempt, outcome)})
        terminal, finished = read(path), read(outcome)
        if (
            len(state["phases"]) != len(PHASES)
            or hashes[path] != state["terminal_sha256"]
            or hashes[outcome] != state["terminal_outcome_sha256"]
            or state["terminal"] != str(path)
            or terminal["status"] != queue.controller.TERMINAL_STATUS
            or terminal["binding_sha256"] != state["issued_binding_sha256"]
            or any(state[name] != terminal[name] for name in DECISIONS)
            or terminal["controller_pid_absent"] != state["issued_controller"]["pid"]
            or terminal["controller_start_ticks"] != state["issued_controller"]["start_ticks"]
            or terminal["verifier_sha256"]
            != sha256(queue.contract.locations(root)[1] / "ns_training_path_lr_gpu_controller.py")
            or terminal["predecessor_acceptance"]["seal_sha256"] != state["predecessor_seal_sha256"]
            or terminal["production_authorized"] is not False
            or finished["status"] != "complete"
            or finished["output"] != str(path)
            or finished["terminal_sha256"] != hashes[path]
            or finished["reconstruction_intent_sha256"] != hashes[attempt]
            or terminal["reconstruction_intent_sha256"] != hashes[attempt]
            or any(sha256(root / name) != digest for name, digest in terminal["evidence_sha256"].items())
        ):
            raise ValueError("Training-path actual-LR canonical terminal outcome or decisions differ")
    if hashes != {path: sha256(path) for path in paths}:
        raise RuntimeError("Training-path actual-LR saved evidence changed during reporting")
    return cpu, intent, state, terminal


def lines(root: Path, local_time) -> list[str]:
    if not (root / CPU).is_file():
        return []
    title = "### NS Actual-LR Training-Path Queue"
    try:
        cpu, intent, state, terminal = evidence(root)
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        ET.ParseError,
        tarfile.TarError,
    ) as failure:
        return [
            title,
            "",
            f"Saved training-path actual-LR queue evidence refused: `{type(failure).__name__}: {failure}`.",
            "",
        ]
    result = [
        title,
        "",
        f"Saved queue CPU/source-archive acceptance at {local_time(cpu.get('time'))}; "
        f"{queue.TEST_COUNT} passing cases and {len(cpu['helpers_sha256'])} authenticated helpers.",
        f"Accepted execution-owner parent: `PVC/{queue.PARENT}`.",
        "54 actual-LR configurations and 162 bounded eight-rank commands retain the original 5,120-update schedule.",
        "The existing original waiter owns its sole canonical terminal; this queue invokes no original terminal.",
    ]
    if intent:
        result.append(
            f"Saved queue intent `{intent['status']}` at {local_time(intent.get('time'))}; "
            f"original-owner checks every {queue.INTERVAL} seconds."
        )
    if state:
        result.append(
            f"Saved queue `{state['status']}` at {local_time(state.get('time'))}; "
            f"{len(state['phases'])}/{len(PHASES)} completed phases."
        )
        if state.get("original_observation", {}).get("owners"):
            result.append(
                "Saved original-owner states: "
                + ", ".join(
                    f"{name} `{item['status']}`" for name, item in state["original_observation"]["owners"].items()
                )
                + "."
            )
        if state.get("error"):
            result.append(f"Preserved queue failure: `{state['error']}`.")
        if state.get("original_decisions"):
            original = state["original_decisions"]
            result.append(
                f"Retained original decisions: {len(original['accepted_cases'])} accepted, "
                f"{len(original['rejected_cases'])} rejected; all-case predicate `{original['all_cases_accepted']}`."
            )
    if terminal:
        result.append(
            f"Saved canonical actual-LR terminal at {local_time(terminal.get('time'))}; "
            f"{len(terminal['accepted_cases'])} accepted, {len(terminal['rejected_cases'])} rejected; "
            f"all-case predicate `{terminal['all_cases_accepted']}`."
        )
    result.extend(
        [
            "Saved receipts do not establish current liveness or rederive tensor arithmetic. "
            "Failed phases and consumed claims remain retained.",
            "Bounded continuation gates supply no production, full-budget quality, efficiency or SwanLab acceptance. "
            "Large shared-token comparisons and independently verified private full histories remain required.",
            "",
        ]
    )
    return result
