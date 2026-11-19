# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal the original training-path terminal and exact restoration without numerical replay."""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import ns_training_path_contract as contract
import ns_training_path_followup as followup
import ns_training_path_gpu_controller as controller
import ns_training_path_reporting as reporting
import owned_output_handoff as handoff
from fused_attribution_validation import INTERRUPTED_MANIFEST, require_keeper_shape
from fused_attribution_validation_controller import require_helper_bytes
from root_stage_failed_predecessor import file_identity
from selected_lr_failure_handoff import keeper_guard, same_keeper
from selected_lr_release_owner import read, sha256

STATUS = "original_training_path_terminal_sealed_actual_lr_owner_unissued"
SEAL = "run-preparation/ns-training-path-actual-lr-predecessor-seal-20261119.json"
REPORTER_SHA256 = "89ece70437d29bc03b85c05fb60b870b2933b213295dfc156dbffb5d81ccc1db"
FOLLOWUP_SHA256 = "7b051e8d4d2ccf5aeac5226fb151aa739a7eb7a360bb4d405b63b9a656d232bb"
CONTROLLER_SHA256 = "4a6823fc21e67a72c89e7b33cba9162429cd7a7e0b8180dcc22149691fa000a7"
SCOPE = (
    "Read-only original-training-path canonical terminal, complete 54-command/18-case results, "
    "retained rejections, original owner/waiter/terminal-child absence and exact restored keeper. "
    "Authenticate the accepted original report and frozen one-shot reconstruction, never replay "
    "tensors or invoke another terminal. Bind checkpoint payload digests to private file identities "
    "whose change times precede the original reconstruction claim. No actual-LR acceptance, GPU "
    "transfer, execution owner, production release, quality, elapsed-efficiency or SwanLab authority."
)


def require_hidden_cuda() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Training-path actual-LR predecessor inspection requires hidden CUDA")


def original(root: Path) -> tuple[dict, dict, dict, dict]:
    """Read only the fixed original owner and waiter identities before expensive reporting."""
    require_hidden_cuda()
    binding, helpers, state, _, _ = contract.locations(root)
    work = root / reporting.FOLLOWUP_WORK
    paths = (binding, helpers.parent / "controller.json", state, work / "intent.json", work / "state.json")
    identities = {path: file_identity(path) for path in paths}
    if (
        root != root.resolve(strict=True)
        or sha256(binding) != reporting.BINDING_SHA256
        or sha256(paths[1]) != reporting.INTENT_SHA256
        or sha256(paths[3]) != reporting.FOLLOWUP_INTENT_SHA256
    ):
        raise ValueError("Training-path actual-LR predecessor original acknowledgment changed")
    frozen, intent, saved, waiter_intent, waiter_state = (read(path) for path in paths)
    if (
        frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or intent["status"] != "acknowledged"
        or intent["binding_sha256"] != reporting.BINDING_SHA256
        or intent["command"] != frozen["controller_command"]
        or saved["binding_sha256"] != reporting.BINDING_SHA256
        or saved["controller_pid"] != intent["pid"]
        or saved["controller_start_ticks"] != intent["start_ticks"]
        or saved["command"] != frozen["validator_command"]
        or saved["child_generation"]["parent"] != intent["pid"]
        or waiter_intent["status"] != "acknowledged"
        or waiter_intent["binding_sha256"] != reporting.BINDING_SHA256
        or waiter_intent["cpu_receipt_sha256"] != reporting.FOLLOWUP_CPU_SHA256
        or waiter_intent["command"] != followup.command(root, root / reporting.FOLLOWUP_CPU)
        or waiter_intent["runner_sha256"] != FOLLOWUP_SHA256
        or waiter_state["pid"] != waiter_intent["pid"]
        or waiter_state["start_ticks"] != waiter_intent["start_ticks"]
        or waiter_state["binding_sha256"] != reporting.BINDING_SHA256
        or any(file_identity(path) != identity for path, identity in identities.items())
    ):
        raise ValueError("Training-path actual-LR predecessor original owner/waiter attribution differs")
    return frozen, intent, saved, waiter_intent


def readiness(root: Path) -> dict:
    """Poll exact original handles; saved completion alone cannot release allocation."""
    frozen, intent, state, waiter = original(root)
    helpers, work = Path(frozen["helpers"]), root / reporting.FOLLOWUP_WORK
    owners = {
        "controller": followup.handle(
            intent["generation"], helpers, contract.CONTROLLER_MARKER, intent["marker"], frozen["controller_environment"]
        ),
        "validator": followup.handle(
            state["child_generation"],
            Path(frozen["candidate"]),
            contract.CHILD_MARKER,
            state["child_marker"],
            frozen["environment"],
        ),
        "terminal_waiter": followup.handle(
            waiter["generation"], work, followup.MARKER, waiter["marker"], followup.ENVIRONMENT
        ),
    }
    marked = handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"])
    marked.update(handoff.marker_processes(followup.MARKER, waiter["marker"]))
    waiter_state = read(work / "state.json")
    if waiter_state.get("terminal_generation"):
        owners["terminal_child"] = followup.handle(
            waiter_state["terminal_generation"],
            helpers,
            followup.CHILD_MARKER,
            waiter_state["terminal_marker"],
            followup.ENVIRONMENT,
        )
        marked.update(handoff.marker_processes(followup.CHILD_MARKER, waiter_state["terminal_marker"]))
    result_path = contract.locations(root)[4]
    if result_path.is_file():
        for entry in read(result_path).get("gpu_commands", []):
            if handoff.observed(entry["generation"]["pid"]) is not None:
                owners[f"command_{entry['generation']['pid']}"] = followup.handle(
                    entry["generation"],
                    Path(frozen["candidate"]),
                    contract.COMMAND_MARKER,
                    entry["marker"],
                    frozen["environment"],
                )
            marked.update(handoff.marker_processes(contract.COMMAND_MARKER, entry["marker"]))
    controller_descendants = handoff.marker_processes(contract.CONTROLLER_MARKER, intent["marker"])
    permitted = {int(pid) for pid in state.get("restored_keeper", {}).get("generation", {})}
    other_descendants = set(controller_descendants) - permitted - {intent["pid"]}
    if other_descendants:
        marked.update({pid: controller_descendants[pid] for pid in other_descendants})
    if marked:
        owners["marked_descendants"] = {"status": "marked_descendants_present", "pids": sorted(marked)}
    ready = all(item["status"] == "absent" for item in owners.values())
    return {
        "status": "original_training_path_owners_absent" if ready else "waiting_for_original_training_path_owners",
        "ready": ready,
        "owners": owners,
    }


def require_reader_sources(frozen: dict) -> None:
    modules = (
        (reporting, REPORTER_SHA256),
        (followup, FOLLOWUP_SHA256),
        (controller, CONTROLLER_SHA256),
    )
    if any(Path(module.__file__).is_symlink() or sha256(Path(module.__file__)) != digest for module, digest in modules):
        raise ValueError("Training-path actual-LR predecessor accepted reporting/terminal reader changed")
    if frozen["helpers_sha256"]["ns_training_path_gpu_controller.py"] != CONTROLLER_SHA256:
        raise ValueError("Training-path actual-LR predecessor is not the original issued terminal")
    require_helper_bytes(Path(frozen["helpers"]), frozen["helpers_sha256"])


def original_report(root: Path) -> dict:
    """Read the original report in an isolated CPU subprocess using archived source origins."""
    helpers = Path(reporting.__file__).resolve().parent
    command = [
        contract.PYTHON,
        "-B",
        "-P",
        "-c",
        "import json,sys; from pathlib import Path; import ns_training_path_reporting as report; "
        "assert Path(report.__file__).resolve()==Path(sys.argv[2]); "
        "print(json.dumps(report.evidence(Path(sys.argv[1])),allow_nan=False))",
        str(root),
        str(helpers / "ns_training_path_reporting.py"),
    ]
    result = subprocess.run(
        command,
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training-path original archived reporting failed: {result.stderr[-4000:]}")
    return json.loads(result.stdout)


def evidence(root: Path) -> dict:
    """Authenticate accepted terminal receipts and checkpoint identities, without tensor reads."""
    if not readiness(root)["ready"]:
        raise RuntimeError("Training-path actual-LR predecessor original owner or waiter remains present")
    frozen, intent, state, _ = original(root)
    require_reader_sources(frozen)
    report = original_report(root)
    progress = report.get("gpu_progress", {})
    terminal_report = progress.get("terminal", {})
    followup_report = report.get("terminal_followup", {})
    if (
        report.get("execution_owner", {}).get("accepted") is not True
        or progress.get("status") != "validation_complete"
        or progress.get("result_status") != reporting.RESULT_STATUS
        or progress.get("completed_commands") != 54
        or terminal_report.get("status") != "complete"
        or terminal_report.get("saved_exact_restoration") is not True
        or followup_report.get("status") != "terminal_reconstructed"
        or report.get("numerical_gpu_acceptance", {}).get("status") != "saved_independent_terminal_receipt"
    ):
        raise RuntimeError("Training-path actual-LR predecessor complete original terminal/report is unavailable")
    output, attempt, outcome, _ = controller.terminal_paths(root, intent)
    terminal, claim = read(output), read(attempt)
    result = read(contract.locations(root)[4])
    summary = {key: result[key] for key in ("accepted_cases", "rejected_cases", "all_cases_accepted")}
    if (
        terminal_report["receipt"] != str(output)
        or terminal_report["receipt_sha256"] != sha256(output)
        or any(terminal[key] != value for key, value in summary.items())
        or any(state[key] != value for key, value in summary.items())
        or len(result["results"]) != 18
        or len(result["gpu_commands"]) != 54
        or len(progress["decisions"]["cases"]) != 18
        or progress["decisions"]["cases"] != result["results"]
        or terminal["reconstruction_intent_sha256"] != sha256(attempt)
        or read(outcome)["terminal_sha256"] != sha256(output)
    ):
        raise ValueError("Training-path actual-LR predecessor terminal coverage or rejection summaries differ")
    paths = {root / name: digest for name, digest in terminal["evidence_sha256"].items()} | {
        output: sha256(output),
        attempt: sha256(attempt),
        outcome: sha256(outcome),
        Path(frozen["archive"]): frozen["archive_sha256"],
        root / reporting.OWNER_CPU: reporting.OWNER_CPU_SHA256,
        root / reporting.FOLLOWUP_CPU: reporting.FOLLOWUP_CPU_SHA256,
        root / reporting.FOLLOWUP_WORK / "intent.json": reporting.FOLLOWUP_INTENT_SHA256,
    }
    paths.update(
        {
            root / reporting.FOLLOWUP_WORK / name: sha256(root / reporting.FOLLOWUP_WORK / name)
            for name in ("state.json", "followup.log", "terminal.log")
        }
    )
    paths.update({Path(frozen["candidate"]) / name: digest for name, digest in frozen["files_sha256"].items()})
    paths.update({Path(frozen["helpers"]) / name: digest for name, digest in frozen["helpers_sha256"].items()})
    paths[root / "data/experiment_receipt.json"] = frozen["data_receipt_sha256"]
    payloads = {}
    claim_time = datetime.fromisoformat(claim["time"])
    if claim_time.tzinfo is None:
        raise ValueError("Training-path original reconstruction claim has no absolute time")
    claim_ns = int(claim_time.timestamp() * 1_000_000_000)
    for entry in result["gpu_commands"]:
        log = Path(entry["log"])
        paths.update(
            {log: entry["log_sha256"], log.with_suffix(".execution.json"): sha256(log.with_suffix(".execution.json"))}
        )
    for case in result["results"]:
        for mode, trajectory in case["trajectories"].items():
            directory = Path(case["continuous"] if mode == "continuous" else case["resumed"])
            manifest = INTERRUPTED_MANIFEST if mode == "interrupted" else "run.json"
            paths[directory / manifest] = trajectory["manifest_sha256"]
            paths[directory / "config.json"] = sha256(directory / "config.json")
            paths.update({directory / name: digest for name, digest in trajectory["history_sha256"].items()})
            for step, checkpoint in trajectory["checkpoints"].items():
                folder = directory / f"checkpoints/step_{int(step):06d}"
                paths[folder / "complete.json"] = checkpoint["receipt_sha256"]
                for rank, size in enumerate(checkpoint["rank_bytes"]):
                    path = folder / f"rank_{rank}.pt"
                    identity = file_identity(path)
                    digest = checkpoint["rank_sha256"][path.name]
                    if identity["bytes"] != size or identity["ctime_ns"] > claim_ns:
                        raise ValueError("Training-path payload changed after the original reconstruction claim")
                    saved = {"sha256": digest, "identity": identity}
                    if path in payloads and payloads[path] != saved:
                        raise ValueError("Training-path original payload identities or retained digests conflict")
                    payloads[path] = saved
    records = {}
    for name, item in frozen["authenticated_files"].items():
        path = Path(name)
        if not path.is_relative_to(root) or file_identity(path) != item["identity"]:
            raise ValueError("Training-path original accepted source/dependency identity changed")
        records[str(path.relative_to(root))] = item
    for path, digest in paths.items():
        identity = file_identity(path)
        if sha256(path) != digest or file_identity(path) != identity:
            raise ValueError("Training-path original terminal evidence changed during authentication")
        records[str(path.relative_to(root))] = {"sha256": digest, "identity": identity}
    if (
        any(file_identity(root / name) != item["identity"] for name, item in records.items())
        or any(file_identity(path) != item["identity"] for path, item in payloads.items())
        or not readiness(root)["ready"]
    ):
        raise RuntimeError("Training-path original evidence or absent owners changed during authentication")
    return {
        "status": STATUS,
        "root": str(root),
        "evidence": records,
        "checkpoint_payloads": {str(path.relative_to(root)): item for path, item in payloads.items()},
        "original_terminal": str(output),
        "original_terminal_sha256": sha256(output),
        "original_decisions": summary,
        "restored_keeper": terminal["keeper"],
        "controller_marker": intent["marker"],
        "actual_lr_execution_authorized": False,
        "production_authorized": False,
        "scope": SCOPE,
    }


def require_identities(root: Path, saved: dict) -> None:
    for group in ("evidence", "checkpoint_payloads"):
        if any(file_identity(root / name) != item["identity"] for name, item in saved[group].items()):
            raise ValueError("Training-path actual-LR predecessor sealed original evidence changed")


def seal(root: Path) -> dict:
    """Publish one distinct predecessor seal only after exact original terminal restoration."""
    require_hidden_cuda()
    out = root / SEAL
    if out.exists() or out.is_symlink():
        raise FileExistsError("Training-path actual-LR predecessor seal is one-shot")
    saved = evidence(root)
    with keeper_guard():
        keeper = handoff.keeper_observation()
        require_keeper_shape(keeper)
        if not same_keeper(keeper, saved["restored_keeper"]) or not readiness(root)["ready"]:
            raise RuntimeError("Training-path original restored keeper or owner absence changed")
        require_identities(root, saved)
        controller.publish_terminal_record(out, saved)
    return read(out)


def require_seal(root: Path, digest: str) -> dict:
    """Authenticate the caller-bound seal and unchanged original identities after transfer."""
    require_hidden_cuda()
    path = root / SEAL
    identity, raw = file_identity(path), path.read_bytes()
    saved = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest() != digest
        or file_identity(path) != identity
        or saved["status"] != STATUS
        or saved["root"] != str(root)
        or saved["scope"] != SCOPE
        or saved["actual_lr_execution_authorized"] is not False
        or saved["production_authorized"] is not False
        or not saved["checkpoint_payloads"]
        or type(saved["original_decisions"]["all_cases_accepted"]) is not bool
        or saved["original_decisions"]["all_cases_accepted"] != (not saved["original_decisions"]["rejected_cases"])
        or len(saved["original_decisions"]["accepted_cases"]) + len(saved["original_decisions"]["rejected_cases"]) != 18
        or saved["evidence"].get(str(Path(saved["original_terminal"]).relative_to(root)), {}).get("sha256")
        != saved["original_terminal_sha256"]
    ):
        raise ValueError("Training-path actual-LR predecessor caller-bound seal differs")
    frozen, _, _, _ = original(root)
    require_reader_sources(frozen)
    require_identities(root, saved)
    if not readiness(root)["ready"] or file_identity(path) != identity:
        raise RuntimeError("Training-path original owners or sealed identities changed")
    return saved
