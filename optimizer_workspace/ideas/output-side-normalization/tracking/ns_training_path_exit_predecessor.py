# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve both failed training-path attempts before a fresh exit-drain diagnostic."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import ns_training_path_recovery_predecessor as original_failure
import owned_output_handoff as handoff
from fused_attribution_validation import INTERRUPTED_MANIFEST, require_keeper_shape
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_failure_handoff import keeper_guard, same_keeper
from selected_lr_release_owner import candidate_files, read, sha256
from wave7_release import local_node

PARENT_SPEC = original_failure.PARENT_SPEC
original = original_failure.original
NODE = original_failure.NODE
RECOVERY_SPEC = (
    "run-preparation/ns-training-path-marker-exit-recovery-20261119-cpu/20261119T1415/receipt.json",
    "6c5b8af36ba473ea042fbb061153a4e94a819960a51255651198b41c9c345638",
    "ns_training_path_marker_exit_recovery_cpu_archive_verified_gpu_unissued",
    103,
)
ORIGINAL_SEAL = original_failure.SEAL
ORIGINAL_SEAL_SHA256 = "19f4bdeb9b3b520413fdeb542bb33d22588afe3483835603fd298179eb4cc3bb"
SEAL = "run-preparation/ns-training-path-exit-drain-predecessor-seal-20261119.json"
STATUS = "both_failed_training_path_attempts_sealed_for_distinct_exit_drain_not_numerical_acceptance"
RECOVERY_NAMESPACE = "ns-training-path-marker-exit-recovery-20261119"
WORK = "validation/ns_training_path_marker_exit_recovery_20261119"
TRIAL = "muont_ns_tpu_q5_full_lr_lower"
FAILED_LOG = f"{WORK}/{TRIAL}_resumed.log"
RECORDS = {
    ORIGINAL_SEAL: ORIGINAL_SEAL_SHA256,
    f"run-preparation/{RECOVERY_NAMESPACE}-binding.json": (
        "8c7afc88680630425b0662ce19a063e0d433413ea215360e9b77c2535f479332"
    ),
    f"run-preparation/{RECOVERY_NAMESPACE}-state.json": (
        "60718e668366a97640ec3b77292ca3b79c14baabfb9098ad28b56a305a4f7e6d"
    ),
    f"run-preparation/validation-{RECOVERY_NAMESPACE}-gpu.json": (
        "df44e83cb4c6856fd39898225d9e4158c0807b2dce4328e6ca4beb27f1736260"
    ),
    f"controllers/{RECOVERY_NAMESPACE}/controller.json": (
        "85013b6805e40957bfe98e610002cbbf233c1cc705ddafc3011e21a881f742d9"
    ),
    f"controllers/{RECOVERY_NAMESPACE}/controller.log": (
        "c85c1fa695e9a0f9c36e20f703e6569c863f7db6fd1ef5b4e66c8d3608db213b"
    ),
    f"controllers/{RECOVERY_NAMESPACE}/validator.log": (
        "9d36ad8abc1281a00a59152633b6899dc7f14c5d985b406cde5ee6d87190934e"
    ),
    FAILED_LOG: "dc2393a3f8571f0082548a487c74899087f0280daf87e3a1594dbd4efac2339a",
    FAILED_LOG.replace(".log", ".execution.json"): "f2492ab758dbc35d4179153931bfb78f963f70cf0624f0cf893c95f2e707e5ae",
}
SCOPE = (
    "Retain the original44/54-command14/18-decision failure, both zero-launch failed follow-ups "
    "and distinct recovery2/54-command0/18-decision failure under their original source identities. "
    "Authenticate absent owners/markers, exact restored keeper, byte-pinned scalar/log evidence "
    "and retained checkpoint digests/current identities without fresh large-payload hashing. "
    "No completion, terminal, numerical waiver, production or GPU authority. Only a separately "
    "accepted fresh exit-drain diagnostic may use this failed handoff; old identities remain consumed."
)


def checked(path: Path, digest: str) -> dict:
    identity = file_identity(path)
    if sha256(path) != digest or file_identity(path) != identity:
        raise ValueError(f"Exit-drain predecessor evidence changed: {path}")
    return {"sha256": digest, "identity": identity}


def recorded(root: Path) -> tuple[dict, dict, dict, dict]:
    return tuple(
        read(root / name)
        for name in (
            f"run-preparation/{RECOVERY_NAMESPACE}-binding.json",
            f"controllers/{RECOVERY_NAMESPACE}/controller.json",
            f"run-preparation/{RECOVERY_NAMESPACE}-state.json",
            f"run-preparation/validation-{RECOVERY_NAMESPACE}-gpu.json",
        )
    )


def recovery_structure(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> dict:
    """Validate the consumed recovery's incomplete coverage and failed third command."""
    failed = read(root / FAILED_LOG.replace(".log", ".execution.json"))
    if (
        frozen["root"] != str(root)
        or len(frozen["commands"]) != 54
        or len(frozen["plan"]["trials"]) != 18
        or frozen["commands"] != frozen["plan"]["commands"]
        or state["status"] != "needs_attention"
        or state["error"] != "RuntimeError: NS training-path bounded validator failed"
        or any(type(state[key]) is not int or state[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or state["marked_descendants_absent"] is not True
        or state["keeper_release_attempted"] is not True
        or state["keeper_release_complete"] is not True
        or any(key in state for key in ("cleanup_error", "deferred_signals"))
        or intent["status"] != "acknowledged"
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["generation"]["command"] != frozen["controller_command"]
        or state["child_generation"]["command"] != frozen["validator_command"]
        or state["child_generation"]["parent"] != intent["pid"]
        or result["status"] != "needs_attention"
        or result["results"] != []
        or len(result["gpu_commands"]) != 2
        or result["trials"] != frozen["plan"]["trials"]
        or result["owner"]
        != {
            "controller_pid": intent["pid"],
            "controller_start_ticks": intent["start_ticks"],
            "validator_pid": state["child_pid"],
            "validator_start_ticks": state["child_start_ticks"],
        }
        or any(
            value != RECORDS[f"run-preparation/{RECOVERY_NAMESPACE}-binding.json"]
            for value in (state["binding_sha256"], intent["binding_sha256"], result["binding_sha256"])
        )
        or failed["command"] != frozen["commands"][2]
        or failed["generation"]["parent"] != state["child_pid"]
        or failed["spawn_status"] != "acknowledged"
        or any(type(failed[key]) is not int or failed[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
        or failed["marked_descendants_absent"] is not True
        or failed["log_sha256"] != RECORDS[FAILED_LOG]
        or "cleanup_error" in failed
    ):
        raise ValueError("Exit-drain predecessor incomplete recovery structure differs")
    for index, entry in enumerate(result["gpu_commands"]):
        if (
            entry["command"] != frozen["commands"][index]
            or entry["trial"] != TRIAL
            or entry["mode"] != ("continuous", "interrupted")[index]
            or entry["generation"]["parent"] != state["child_pid"]
            or entry["generation"]["command"] != entry["command"]
            or entry["spawn_status"] != "acknowledged"
            or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or any(key in entry for key in ("error", "cleanup_error"))
        ):
            raise ValueError("Exit-drain predecessor successful command ancestry or coverage differs")
    return {
        "completed_commands": 2,
        "planned_commands": 54,
        "case_decisions": 0,
        "planned_cases": 18,
        "coverage_complete": False,
        "numerically_accepted": False,
        "failed_command": failed,
    }


def require_absent(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> None:
    handles = [intent["generation"], state["child_generation"]]
    handles += [entry["generation"] for entry in result["gpu_commands"]]
    failed = read(root / FAILED_LOG.replace(".log", ".execution.json"))
    handles.append(failed["generation"])
    _, old_helpers, old_state_path, _, old_result_path = original.locations(root)
    old_intent, old_state, old_result = (
        read(old_helpers.parent / "controller.json"),
        read(old_state_path),
        read(old_result_path),
    )
    handles += [old_intent["generation"], old_state["child_generation"]]
    handles += [entry["generation"] for entry in old_result["gpu_commands"]]
    handles.append(read(root / original_failure.FAILED_LOG.replace(".log", ".execution.json"))["generation"])
    handles += [read(root / relative / "intent.json")["generation"] for relative, _, _, _ in original_failure.WORKS]
    if any((handoff.PROC_ROOT / str(entry["pid"])).exists() for entry in handles):
        raise RuntimeError("Exit-drain predecessor consumed recovery handles remain present")
    permitted = {int(pid) for pid in state["restored_keeper"]["generation"]}
    marks = [("OUTPUT_MARIN_NS_TRAINING_PATH_RECOVERY_CONTROLLER", intent["marker"], permitted)]
    marks.append(("OUTPUT_MARIN_NS_TRAINING_PATH_RECOVERY_VALIDATOR", state["child_marker"], set()))
    marks += [
        ("OUTPUT_MARIN_NS_TRAINING_PATH_RECOVERY_COMMAND", entry["marker"], set())
        for entry in [*result["gpu_commands"], failed]
    ]
    for name, value, allowed in marks:
        if not set(handoff.marker_processes(name, value)).issubset(allowed):
            raise RuntimeError("Exit-drain predecessor recovery marker remains outside its restored keeper")
    directory = Path(frozen["helpers"]).parent
    if any((directory / name).exists() for name in ("terminal-intent.json", "terminal-outcome.json")):
        raise ValueError("Exit-drain predecessor has unexpected recovery terminal claims")


def retained_recovery(root: Path, frozen: dict, state: dict, result: dict) -> tuple[dict, dict]:
    records = {name: checked(root / name, digest) for name, digest in RECORDS.items()}
    payloads = {}
    for entry in result["gpu_commands"]:
        log = Path(entry["log"])
        sidecar = read(log.with_suffix(".execution.json"))
        if {key: value for key, value in sidecar.items() if key != "time"} != {
            key: value for key, value in entry.items() if key not in ("time", "trial", "mode", "trajectory")
        }:
            raise ValueError("Exit-drain predecessor command sidecar changed")
        for path, digest in (
            (log, entry["log_sha256"]),
            (log.with_suffix(".execution.json"), sha256(log.with_suffix(".execution.json"))),
        ):
            records[str(path.relative_to(root))] = checked(path, digest)
        trial = frozen["plan"]["trials"][0]
        directory = Path(trial["continuous"] if entry["mode"] == "continuous" else trial["resumed"])
        trajectory = entry["trajectory"]
        manifest = directory / ("run.json" if entry["mode"] == "continuous" else INTERRUPTED_MANIFEST)
        records[str(manifest.relative_to(root))] = checked(manifest, trajectory["manifest_sha256"])
        for name, digest in trajectory["history_sha256"].items():
            path = directory / name
            records[str(path.relative_to(root))] = checked(path, digest)
        for step, checkpoint in trajectory["checkpoints"].items():
            work = directory / f"checkpoints/step_{int(step):06d}"
            path = work / "complete.json"
            records[str(path.relative_to(root))] = checked(path, checkpoint["receipt_sha256"])
            for rank, digest in checkpoint["rank_sha256"].items():
                path = work / rank
                identity = file_identity(path)
                if identity["bytes"] != checkpoint["rank_bytes"][int(rank.removeprefix("rank_").removesuffix(".pt"))]:
                    raise ValueError("Exit-drain predecessor checkpoint payload size changed")
                payloads[str(path.relative_to(root))] = {"sha256": digest, "identity": identity}
    return records, payloads


def evidence(root: Path) -> dict:
    if root != root.resolve(strict=True) or local_node() != NODE or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Exit-drain predecessor requires the original node and hidden CUDA")
    parent = accepted_archive(root, RECOVERY_SPEC)
    previous = original_failure.require_seal(root, ORIGINAL_SEAL_SHA256)
    frozen, intent, state, result = recorded(root)
    summary = recovery_structure(root, frozen, intent, state, result)
    if (
        frozen["cpu_receipt_sha256"] != RECOVERY_SPEC[1]
        or frozen["cpu_receipt"] != str(root / RECOVERY_SPEC[0])
        or frozen["files_sha256"] != parent["files_sha256"]
        or frozen["helpers_sha256"] != parent["helpers_sha256"]
        or candidate_files(Path(parent["candidate"])) != parent["files_sha256"]
        or frozen["allocation_predecessor"] != previous
        or sha256(Path(frozen["archive"])) != frozen["archive_sha256"]
    ):
        raise ValueError("Exit-drain predecessor recovery accepted source differs")
    require_helper_bytes(Path(frozen["helpers"]), frozen["helpers_sha256"])
    require_archive(Path(frozen["archive"]), parent["inventory_sha256"], alias=True)
    require_absent(root, frozen, intent, state, result)
    records, payloads = retained_recovery(root, frozen, state, result)
    for name, item in previous["evidence"].items():
        records[name] = checked(root / name, item["sha256"])
    payloads |= previous["checkpoint_payloads"]
    for path, digest in (
        (root / RECOVERY_SPEC[0], RECOVERY_SPEC[1]),
        (Path(frozen["archive"]), frozen["archive_sha256"]),
    ):
        records[str(path.relative_to(root))] = checked(path, digest)
    require_keeper_shape(state["restored_keeper"])
    with keeper_guard():
        if not same_keeper(handoff.keeper_observation(), state["restored_keeper"]):
            raise RuntimeError("Exit-drain predecessor exact restored keeper differs")
    if state["restored_keeper"]["devices"] != frozen["devices"]:
        raise ValueError("Exit-drain predecessor physical devices differ")
    require_absent(root, frozen, intent, state, result)
    if any(file_identity(root / name) != item["identity"] for name, item in (records | payloads).items()):
        raise ValueError("Exit-drain predecessor retained evidence changed")
    return {
        "status": STATUS,
        "root": str(root),
        "node": NODE,
        "evidence": records,
        "checkpoint_payloads": payloads,
        "retained_summary": summary,
        "original_retained_summary": previous["retained_summary"],
        "restored_keeper": state["restored_keeper"],
        "gpu_execution_authorized": False,
        "production_authorized": False,
        "scope": SCOPE,
    }


def seal(root: Path) -> dict:
    path = root / SEAL
    if path.exists() or path.is_symlink():
        raise FileExistsError("Exit-drain predecessor seal already exists")
    saved = evidence(root) | {"verifier_sha256": sha256(Path(__file__)), "time": datetime.now(UTC).isoformat()}
    with tempfile.TemporaryDirectory(prefix=".exit-drain-seal-", dir=path.parent) as directory:
        staged = Path(directory) / "record.json"
        with staged.open("x") as stream:
            json.dump(saved, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
        staged.unlink()
    return read(path)


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
        or any(checked(root / name, item["sha256"]) != item for name, item in saved["evidence"].items())
        or any(file_identity(root / name) != item["identity"] for name, item in saved["checkpoint_payloads"].items())
    ):
        raise ValueError("Exit-drain predecessor sealed evidence differs")
    frozen, intent, state, result = recorded(root)
    if recovery_structure(root, frozen, intent, state, result) != saved["retained_summary"]:
        raise ValueError("Exit-drain predecessor sealed recovery coverage differs")
    original_failure.require_seal(root, ORIGINAL_SEAL_SHA256)
    require_absent(root, frozen, intent, state, result)
    if file_identity(path) != identity or path.read_bytes() != raw:
        raise ValueError("Exit-drain predecessor seal changed during checking")
    return saved


def failed_structure(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> dict:
    """Retain original-failure assertions in inherited acceptance regressions."""
    return original_failure.failed_structure(root, frozen, intent, state, result)


def require_absent_owners(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> list[dict]:
    return original_failure.require_absent_owners(root, frozen, intent, state, result)


WORKS = original_failure.WORKS
