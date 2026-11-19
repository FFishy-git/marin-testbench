# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain the incomplete original training-path attempt before a distinct recovery."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import ns_training_path_contract as original
import owned_output_handoff as handoff
from fused_attribution_validation import INTERRUPTED_MANIFEST, require_keeper_shape
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive as accepted_archive
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_failure_handoff import keeper_guard, same_keeper
from selected_lr_release_owner import candidate_files, read, sha256
from wave7_release import local_node

PARENT_SPEC = (
    "run-preparation/ns-training-path-owner-cpu/closure-corrected-20261119T0738/receipt.json",
    "b5bc010df3c86aa849afb014acf2c9815e913ebcb50843f38808d78cad739fbd",
    "ns_training_path_execution_owner_cpu_archive_verified_gpu_unissued",
    55,
)
NODE = "28.216.17.9"
SEAL = "run-preparation/ns-training-path-failed-attempt-recovery-seal-20261119.json"
STATUS = "incomplete_training_path_failure_sealed_for_distinct_recovery_not_numerical_acceptance"
WORKS = (
    (
        "run-preparation/ns-training-path-terminal-followup-20261119",
        "OUTPUT_MARIN_NS_TRAINING_PATH_FOLLOWUP",
        "followup.log",
        "RuntimeError: Training-path original owners exited without complete cleanup/restoration",
    ),
    (
        "run-preparation/ns-training-path-actual-lr-queue-20261119",
        "OUTPUT_MARIN_NS_TRAINING_PATH_LR_QUEUE",
        "queue.log",
        "RuntimeError: Training-path actual-LR predecessor complete original terminal/report is unavailable",
    ),
)
FAILED_NAME = "null_leftdiag_fused_ns_tpu_q10_full_lr_lower"
FAILED_LOG = f"validation/ns_training_path_20261119/{FAILED_NAME}_resumed.log"
RECORDS = {
    "run-preparation/ns-training-path-binding-20261119.json": (
        "20133ef8358e51cc97e83b866063843ccc10a314ac2b72aef5b7a8b4354f0064"
    ),
    "run-preparation/ns-training-path-state-20261119.json": (
        "cfa6e79ab0b670dc48a45cb63688b9c74db14d77f8b9b22dae7bc00311188eee"
    ),
    "run-preparation/validation-ns-training-path-gpu-20261119.json": (
        "67e1008d7f3abdbc36a3ed21e26d8558cffebe8269f9e3a1ec4dfc413c2db362"
    ),
    "controllers/ns-training-path-20261119/controller.json": (
        "7b24631110346132bc10b93e539faf07ba071422582715f12f5f060d496213aa"
    ),
    "controllers/ns-training-path-20261119/controller.log": (
        "315488ffbb967dfd02b67ae723f26552fa150abe9fa9f2ac0a2cccb2a72d41ce"
    ),
    "controllers/ns-training-path-20261119/validator.log": (
        "9d869bc0fd348146eacf65d2d4e7f504b2f444db8743230f5bf0d3046775a070"
    ),
    f"validation/ns_training_path_20261119/{FAILED_NAME}_resumed.execution.json": (
        "fa4b0c0785ce467463ea0a6470de5b9d9438387c5aa1b98e26bbff38970e9e5b"
    ),
    FAILED_LOG: "3f2d743024a831f750af017b29398b7aa975e3ea4a2c2c671a1d691ddc1efade",
    f"{WORKS[0][0]}/intent.json": "8e7f56916fec0bf0ce65adb7d2740bec85be5bf2ea4afd584f3ec0431715f64d",
    f"{WORKS[0][0]}/state.json": "e8dae79366a18b7639dba92d0ef8e15138a3707bef3cf2de882bd00619256519",
    f"{WORKS[0][0]}/followup.log": "f95c715606ff9c52cb477abd31c274f089d905c26ddb612bd342ac7f8b915863",
    f"{WORKS[1][0]}/intent.json": "f0c270d123da8fd8f53e2b59efe7bdbd9b95003fa17c69509fa44f6b01359256",
    f"{WORKS[1][0]}/state.json": "19a138faedc5289a7cc304b114418515f86d6470fa5f63fd32e9c10b546c7c5c",
    f"{WORKS[1][0]}/queue.log": "618aa256353f7932d37e288f6f37bdaf3c005b2d851a5f162bd9bd6f8dba939e",
}
SCOPE = (
    "Authenticate the exact incomplete original44/54-command14/18-decision training-path "
    "failure, six accepted and eight rejected retained decisions, failed resumed command, "
    "both zero-launch failed follow-ups, original owners/descendants absent and exact restored "
    "keeper. Retain payload digests from the byte-pinned aggregate plus current file identities; "
    "do not claim fresh payload hashing or numerical reconstruction. No original terminal, "
    "completion, numerical waiver, corrected-source acceptance, allocation transfer, production, "
    "quality/efficiency or SwanLab authority. A distinct accepted owner may consume this failed "
    "handoff solely for a new complete diagnostic; original failures/rejections remain unchanged."
)


def checked_file(path: Path, digest: str) -> dict:
    """Hash singly linked canonical evidence and reject changes during inspection."""
    identity = file_identity(path)
    if sha256(path) != digest or file_identity(path) != identity:
        raise ValueError(f"Training-path recovery evidence changed: {path}")
    return {"sha256": digest, "identity": identity}


def retained_evidence(root: Path, frozen: dict, result: dict) -> tuple[dict, dict]:
    """Preserve every issued command and payload identity without rederiving tensors."""
    evidence = {name: checked_file(root / name, digest) for name, digest in RECORDS.items()}
    payloads = {}
    for index, entry in enumerate(result["gpu_commands"]):
        item, mode = result["trials"][index // 3], original.MODES[index % 3]
        command = frozen["commands"][index]
        log = original.locations(root)[3] / f"{item['name']}_{mode}.log"
        sidecar = log.with_suffix(".execution.json")
        execution = read(sidecar)
        if (
            entry["command"] != command
            or entry["trial"] != item["name"]
            or entry["mode"] != mode
            or entry["log"] != str(log)
            or entry["spawn_status"] != "acknowledged"
            or entry["generation"]["command"] != command
            or entry["generation"]["parent"] != result["owner"]["validator_pid"]
            or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or any(
                execution.get(key) != value for key, value in entry.items() if key not in ("trial", "mode", "trajectory")
            )
            or any(key in execution for key in ("error", "cleanup_error"))
        ):
            raise ValueError("Training-path recovery retained command evidence differs")
        for path, digest in ((log, entry["log_sha256"]), (sidecar, sha256(sidecar))):
            evidence[str(path.relative_to(root))] = checked_file(path, digest)
        directory = Path(item["continuous"] if mode == "continuous" else item["resumed"])
        trajectory = entry["trajectory"]
        manifest = directory / (INTERRUPTED_MANIFEST if mode == "interrupted" else "run.json")
        evidence[str(manifest.relative_to(root))] = checked_file(manifest, trajectory["manifest_sha256"])
        for name, digest in trajectory["history_sha256"].items():
            path = directory / name
            evidence[str(path.relative_to(root))] = checked_file(path, digest)
        for step, checkpoint in trajectory["checkpoints"].items():
            checkpoint_dir = directory / f"checkpoints/step_{int(step):06d}"
            receipt = checkpoint_dir / "complete.json"
            evidence[str(receipt.relative_to(root))] = checked_file(receipt, checkpoint["receipt_sha256"])
            for name, digest in checkpoint["rank_sha256"].items():
                path = checkpoint_dir / name
                relative = str(path.relative_to(root))
                identity = file_identity(path)
                item = {"sha256": digest, "identity": identity}
                if relative in payloads and payloads[relative] != item:
                    raise ValueError("Training-path recovery retained payload identities conflict")
                if identity["bytes"] != checkpoint["rank_bytes"][int(name.removeprefix("rank_").removesuffix(".pt"))]:
                    raise ValueError("Training-path recovery retained payload size changed")
                payloads[relative] = item
    return evidence, payloads


def failed_structure(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> dict:
    """Keep incomplete execution separate from keeper restoration and retained decisions."""
    failed = read(root / FAILED_LOG.replace(".log", ".execution.json"))
    decisions = result["results"]
    names = [item["name"] for item in frozen["plan"]["trials"]]
    if (
        frozen["root"] != str(root)
        or frozen["commands"] != frozen["plan"]["commands"]
        or len(frozen["commands"]) != 54
        or len(names) != 18
        or result["status"] != "needs_attention"
        or result["binding_sha256"] != sha256(original.locations(root)[0])
        or result["trials"] != frozen["plan"]["trials"]
        or len(result["gpu_commands"]) != 44
        or len(decisions) != 14
        or [item["name"] for item in decisions] != names[:14]
        or [item["accepted"] for item in decisions] != [True] * 6 + [False] * 8
        or state["status"] != "needs_attention"
        or state["error"] != "RuntimeError: NS training-path bounded validator failed"
        or any(type(state[key]) is not int or state[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or state["keeper_release_attempted"] is not True
        or state["keeper_release_complete"] is not True
        or state["marked_descendants_absent"] is not True
        or any(key in state for key in ("cleanup_error", "deferred_signals"))
        or intent["status"] != "acknowledged"
        or intent["binding_sha256"] != result["binding_sha256"]
        or state["binding_sha256"] != result["binding_sha256"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["generation"]["command"] != frozen["controller_command"]
        or state["child_generation"]["command"] != frozen["validator_command"]
        or state["child_generation"]["parent"] != intent["pid"]
        or result["owner"]
        != {
            "controller_pid": intent["pid"],
            "controller_start_ticks": intent["start_ticks"],
            "validator_pid": state["child_pid"],
            "validator_start_ticks": state["child_start_ticks"],
        }
        or failed["command"] != frozen["commands"][44]
        or failed["generation"]["parent"] != state["child_pid"]
        or failed["spawn_status"] != "acknowledged"
        or any(type(failed[key]) is not int or failed[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
        or failed["marked_descendants_absent"] is not True
        or "cleanup_error" in failed
        or failed["log_sha256"] != RECORDS[FAILED_LOG]
        or "Marked process generation changed during handoff" not in (root / FAILED_LOG).read_text()
    ):
        raise ValueError("Training-path recovery original incomplete failure or decisions differ")
    return {
        "planned_commands": 54,
        "completed_commands": 44,
        "planned_cases": 18,
        "case_decisions": 14,
        "accepted_cases": names[:6],
        "rejected_cases": names[6:14],
        "missing_case_decisions": names[14:],
        "failed_command": frozen["commands"][44],
        "failed_execution": failed,
        "coverage_complete": False,
        "numerically_accepted": False,
        "decisions": decisions,
    }


def require_absent_owners(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> list[dict]:
    """Reject substituted PIDs and all original marked descendants except the exact keeper."""
    expected = [intent["generation"], state["child_generation"]]
    marks = [(original.CONTROLLER_MARKER, intent["marker"]), (original.CHILD_MARKER, state["child_marker"])]
    for entry in result["gpu_commands"]:
        expected.append(entry["generation"])
        marks.append((original.COMMAND_MARKER, entry["marker"]))
    failed = read(root / FAILED_LOG.replace(".log", ".execution.json"))
    expected.append(failed["generation"])
    marks.append((original.COMMAND_MARKER, failed["marker"]))
    followups = []
    for relative, marker, log, error in WORKS:
        work = root / relative
        acknowledgment, saved = read(work / "intent.json"), read(work / "state.json")
        generation = acknowledgment["generation"]
        if (
            acknowledgment["status"] != "acknowledged"
            or generation["pid"] != acknowledgment["pid"]
            or generation["start_ticks"] != acknowledgment["start_ticks"]
            or generation["command"] != acknowledgment["command"]
            or generation["group"] != generation["pid"]
            or generation["session"] != generation["pid"]
            or saved["pid"] != generation["pid"]
            or saved["start_ticks"] != generation["start_ticks"]
            or saved["status"] != "needs_attention"
            or saved["error"] != error
            or saved["interval"] != 900
            or saved.get("phases", []) != []
            or any(key in saved for key in ("terminal_command", "active_phase", "issued_controller", "cleanup_error"))
            or datetime.fromisoformat(saved["time"]) <= datetime.fromisoformat(state["time"])
        ):
            raise ValueError("Training-path recovery failed follow-up identity or zero-launch state differs")
        expected.append(generation)
        marks.append((marker, acknowledgment["marker"]))
        for observation in work.glob("*observation-*.json"):
            file_identity(observation)
        followups.append({"intent": acknowledgment, "state": saved, "log": str(work / log)})
    if any(handoff.observed(item["pid"]) is not None for item in expected):
        raise RuntimeError("Training-path recovery original or follow-up process handle remains present")
    permitted = {int(pid) for pid in state["restored_keeper"]["generation"]}
    for marker, value in marks:
        marked = handoff.marker_processes(marker, value)
        if marker == original.CONTROLLER_MARKER:
            if not set(marked).issubset(permitted):
                raise RuntimeError("Training-path recovery original controller marker exists outside its keeper")
        elif marked:
            raise RuntimeError("Training-path recovery original training or follow-up descendants remain live")
    if frozen["helpers"] != str(original.locations(root)[1]) or any(
        (original.locations(root)[1].parent / name).exists()
        for name in ("terminal-intent.json", "terminal-outcome.json")
    ):
        raise ValueError("Training-path recovery original has unexpected canonical terminal claims")
    return followups


def evidence(root: Path) -> dict:
    """Authenticate the failed attempt and current restoration without releasing allocation."""
    if root != root.resolve(strict=True) or local_node() != NODE or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Training-path recovery predecessor requires canonical original node and hidden CUDA")
    parent = accepted_archive(root, PARENT_SPEC)
    binding, helpers, state_path, _, result_path = original.locations(root)
    frozen, intent, state, result = (
        read(binding),
        read(helpers.parent / "controller.json"),
        read(state_path),
        read(result_path),
    )
    summary = failed_structure(root, frozen, intent, state, result)
    if (
        frozen["cpu_receipt"] != str(root / PARENT_SPEC[0])
        or frozen["cpu_receipt_sha256"] != PARENT_SPEC[1]
        or frozen["files_sha256"] != parent["files_sha256"]
        or frozen["candidate"] != parent["candidate"]
        or candidate_files(Path(parent["candidate"])) != parent["files_sha256"]
        or frozen["helpers_sha256"] != parent["helpers_sha256"]
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("Training-path recovery original accepted archive/source identity differs")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_archive(Path(frozen["archive"]), parent["inventory_sha256"], alias=True)
    followups = require_absent_owners(root, frozen, intent, state, result)
    records, payloads = retained_evidence(root, frozen, result)
    for relative, _, _, _ in WORKS:
        for path in (root / relative).glob("*observation-*.json"):
            records[str(path.relative_to(root))] = checked_file(path, sha256(path))
    records[PARENT_SPEC[0]] = checked_file(root / PARENT_SPEC[0], PARENT_SPEC[1])
    require_keeper_shape(state["restored_keeper"])
    with keeper_guard():
        keeper = handoff.keeper_observation()
        if not same_keeper(keeper, state["restored_keeper"]) or keeper["devices"] != frozen["devices"]:
            raise RuntimeError("Training-path recovery exact failed-attempt keeper restoration differs")
    if any(file_identity(root / name) != entry["identity"] for name, entry in (records | payloads).items()):
        raise ValueError("Training-path recovery evidence changed during authentication")
    require_absent_owners(root, frozen, intent, state, result)
    return {
        "status": STATUS,
        "root": str(root),
        "node": NODE,
        "evidence": records,
        "checkpoint_payloads": payloads,
        "retained_summary": summary,
        "failed_followups": followups,
        "restored_keeper": state["restored_keeper"],
        "original_binding_sha256": RECORDS[str(binding.relative_to(root))],
        "original_source_archive_sha256": frozen["archive_sha256"],
        "gpu_execution_authorized": False,
        "production_authorized": False,
        "scope": SCOPE,
    }


def seal(root: Path) -> dict:
    out = root / SEAL
    if out.exists() or out.is_symlink():
        raise FileExistsError("Training-path recovery failed handoff seal already exists")
    saved = evidence(root)
    saved["verifier_sha256"] = sha256(Path(__file__))
    saved["time"] = datetime.now(UTC).isoformat()
    with tempfile.TemporaryDirectory(prefix=".ns-training-path-failed-seal-", dir=out.parent) as directory:
        staged = Path(directory) / "record.json"
        with staged.open("x") as stream:
            stream.write(json.dumps(saved, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, out)
        staged.unlink()
        descriptor = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return read(out)


def require_seal(root: Path, digest: str) -> dict:
    path = root / SEAL
    identity, raw, saved = file_identity(path), path.read_bytes(), read(path)
    if (
        hashlib.sha256(raw).hexdigest() != digest
        or saved["status"] != STATUS
        or saved["root"] != str(root)
        or saved["scope"] != SCOPE
        or saved["verifier_sha256"] != sha256(Path(__file__))
        or saved["gpu_execution_authorized"] is not False
        or saved["production_authorized"] is not False
        or saved["retained_summary"]["coverage_complete"] is not False
        or saved["retained_summary"]["numerically_accepted"] is not False
        or any(saved["evidence"][name]["sha256"] != value for name, value in RECORDS.items())
        or any(
            file_identity(root / name) != entry["identity"]
            for name, entry in (saved["evidence"] | saved["checkpoint_payloads"]).items()
        )
        or any(checked_file(root / name, entry["sha256"]) != entry for name, entry in saved["evidence"].items())
    ):
        raise ValueError("Training-path recovery sealed incomplete evidence changed")
    binding, helpers, state_path, _, result_path = original.locations(root)
    frozen, intent, state, result = (
        read(binding),
        read(helpers.parent / "controller.json"),
        read(state_path),
        read(result_path),
    )
    if failed_structure(root, frozen, intent, state, result) != saved["retained_summary"]:
        raise ValueError("Training-path recovery sealed retained decisions changed")
    require_absent_owners(root, frozen, intent, state, result)
    if path.read_bytes() != raw or file_identity(path) != identity:
        raise ValueError("Training-path recovery failed handoff seal changed during checking")
    return saved
