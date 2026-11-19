# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the failed exit-drain diagnostic and waiter before a distinct attempt."""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import ns_training_path_exit_predecessor as previous
import ns_training_path_reporting as reporting
from fused_attribution_validation import INTERRUPTED_MANIFEST, require_keeper_shape
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_failure_handoff import same_keeper
from selected_lr_release_owner import candidate_files, read, sha256
from wave7_release import local_node

PARENT_SPEC = previous.PARENT_SPEC
RECOVERY_SPEC = previous.RECOVERY_SPEC
original = previous.original
NODE = previous.NODE
WORKS = previous.WORKS
SEAL = "run-preparation/ns-training-path-exit-transition-predecessor-seal-20261119.json"
STATUS = "failed_exit_drain_and_waiter_sealed_for_distinct_transition_not_numerical_acceptance"
OLD_NAMESPACE = "ns-training-path-exit-drain-20261119"
WORK = "validation/ns_training_path_exit_drain_20261119"
WAITER = "run-preparation/ns-training-path-exit-drain-terminal-followup-20261119"
FAILED_LOG = f"{WORK}/muont_ns_tpu_q10_full_lr_lower_continuous.log"
CPU_SPEC = (
    "run-preparation/ns-training-path-exit-drain-20261119-cpu/pidfd-bound-20261119T1542/receipt.json",
    "941990541351584ed2d6b2a4e0302f7e3cb5159e851442f8832ce60ed9c50f69",
    "ns_training_path_exit_drain_cpu_archive_verified_gpu_unissued",
    125,
)
WAITER_CPU_SPEC = (
    "run-preparation/ns-training-path-exit-drain-followup-cpu/20261119T1620/receipt.json",
    "5d94403384b5f6bf9d79dc1438cfec8ae025de267ee76b971345c1cc37146462",
    "ns_training_path_exit_drain_read_only_followup_cpu_archive_verified",
    24,
)
PREVIOUS_SEAL_SHA256 = "07980e69f353192141e52d7dba5e98ea9fda35841f5d1c7be683408fb16237e3"
RECORDS = {
    f"run-preparation/{OLD_NAMESPACE}-binding.json": "80032984ac27c3cd79aa72a579e0b213190146e57c8a87c6b5f9fd1e216095c7",
    f"run-preparation/{OLD_NAMESPACE}-state.json": "cbba16af513319bfa0cb02856dfb9912ba1cdaaefbb87f0e00d87cd6c4d935c6",
    f"run-preparation/validation-{OLD_NAMESPACE}-gpu.json": (
        "ff9b3bddf016149388cb90f02bacf628a0bf281767bb0027de0fb8d5f06744fd"
    ),
    f"controllers/{OLD_NAMESPACE}/controller.json": "64ec305891cdc9cfa1626db79364bf6f3d1a272e6a251ab9485a71e95114f610",
    f"controllers/{OLD_NAMESPACE}/controller.log": "0efd430b2c9b0a76b78608d1169d331d9747fc4c92f14d05216ef485dcf72c5d",
    f"controllers/{OLD_NAMESPACE}/validator.log": "bfc96d64edeb80f45e9b138a7471efc3e4659ae8b27b4dcb862bfb9a289965ee",
    FAILED_LOG: "001708efe35289ad1e74d745bb4ac4fcbf28eae9c862eea78bf7ae41120a4d94",
    FAILED_LOG.replace(".log", ".execution.json"): "9862d278deac9f1c08af390b38632782ed06666503343773944e930a7c4d1862",
    f"{WAITER}/intent.json": "457a7e76095697d1c11a26e5b870b0c1888549a46da4cd255e3ee2aaae4d2c88",
    f"{WAITER}/state.json": "edaf4d1965b6ca4d8fd8b4a7e1bb5c650d38fb69916abd30076b6725ae683d3c",
    f"{WAITER}/followup.log": "e8aba5e6a40b6e9ba5c87f714592ba3155e7c85e0ce0c5ea26fbace00ac3cc9b",
    f"{WAITER}/observation-000000.json": "fdfbd18b5bc0665ee705b3a3734324a831cb911d711b8f60c14b6a492a1af860",
}
SCOPE = (
    "Preserve original44/54commands14/18decisions, recovery2/54commands0decisions, "
    "exit-drain6/54commands2/18accepted-baseline decisions and failed waiter. Check exact "
    "absent owners/markers and restored eight-GPU keeper. Retain saved checkpoint digests "
    "and current identities without new large-payload hashing. No complete terminal, numerical "
    "waiver, production or GPU authority; only separately accepted transition diagnostics."
)
accepted_archive = previous.accepted_archive
checked = previous.checked


def recorded(root: Path) -> tuple[dict, dict, dict, dict]:
    return tuple(
        read(root / name)
        for name in (
            f"run-preparation/{OLD_NAMESPACE}-binding.json",
            f"controllers/{OLD_NAMESPACE}/controller.json",
            f"run-preparation/{OLD_NAMESPACE}-state.json",
            f"run-preparation/validation-{OLD_NAMESPACE}-gpu.json",
        )
    )


def transition_structure(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> dict:
    """Require the actual failed command, two decisions and separate waiter failure."""
    failed = read(root / FAILED_LOG.replace(".log", ".execution.json"))
    waiter = read(root / WAITER / "intent.json")
    waiter_state = read(root / WAITER / "state.json")
    digest = RECORDS[f"run-preparation/{OLD_NAMESPACE}-binding.json"]
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
        or state["command"] != frozen["validator_command"]
        or result["status"] != "needs_attention"
        or result["error"]
        != "ValueError: NS full-model command, exact process generation, cleanup or log identity differs"
        or len(result["gpu_commands"]) != 6
        or len(result["results"]) != 2
        or result["trials"] != frozen["plan"]["trials"]
        or result["owner"]
        != {
            "controller_pid": intent["pid"],
            "controller_start_ticks": intent["start_ticks"],
            "validator_pid": state["child_pid"],
            "validator_start_ticks": state["child_start_ticks"],
        }
        or any(item["binding_sha256"] != digest for item in (state, intent, result, waiter, waiter_state))
        or waiter_state["status"] != "needs_attention"
        or waiter_state["error"]
        != "RuntimeError: Training-path original owners exited without complete cleanup/restoration"
        or waiter_state["observations"] != 1
        or waiter_state["pid"] != waiter["pid"]
        or waiter_state["start_ticks"] != waiter["start_ticks"]
        or waiter["status"] != "acknowledged"
        or waiter["cpu_receipt_sha256"] != WAITER_CPU_SPEC[1]
        or waiter_state["cpu_receipt_sha256"] != WAITER_CPU_SPEC[1]
        or waiter_state["runner_sha256"] != waiter["runner_sha256"]
        or any(key in waiter_state for key in ("terminal_generation", "terminal_sha256"))
        or failed["command"] != frozen["commands"][6]
        or failed["spawn_status"] != "acknowledged"
        or any(type(failed[key]) is not int or failed[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
        or failed["marked_descendants_absent"] is not True
        or failed["log_sha256"] != RECORDS[FAILED_LOG]
        or "cleanup_error" in failed
    ):
        raise ValueError("Exit-transition predecessor incomplete diagnostic or waiter differs")
    reporting._generation(intent["generation"], frozen["controller_command"], intent["pid"], intent["start_ticks"])
    reporting._generation(
        state["child_generation"],
        frozen["validator_command"],
        state["child_pid"],
        state["child_start_ticks"],
        intent["pid"],
    )
    reporting._generation(
        failed["generation"],
        frozen["commands"][6],
        failed["generation"]["pid"],
        failed["generation"]["start_ticks"],
        state["child_pid"],
    )
    reporting._generation(waiter["generation"], waiter["command"], waiter["pid"], waiter["start_ticks"])
    records = {}
    for index, entry in enumerate(result["gpu_commands"]):
        trial, mode = frozen["plan"]["trials"][index // 3], original.MODES[index % 3]
        if entry["trial"] != trial["name"] or entry["mode"] != mode:
            raise ValueError("Exit-transition predecessor command order differs")
        reporting._command(
            entry, frozen["commands"][index], root / WORK / f"{trial['name']}_{mode}.log", state["child_pid"], records
        )
    decisions = reporting._decisions(result, frozen["plan"], 6)
    if decisions["rejected_cases"] or decisions["accepted_cases"] != [
        item["name"] for item in frozen["plan"]["trials"][:2]
    ]:
        raise ValueError("Exit-transition predecessor retained baseline decisions differ")
    return {
        "completed_commands": 6,
        "planned_commands": 54,
        "case_decisions": 2,
        "planned_cases": 18,
        "accepted_cases": decisions["accepted_cases"],
        "decisions": result["results"],
        "coverage_complete": False,
        "numerically_accepted": False,
        "failed_command": failed,
        "waiter": waiter_state,
    }


def require_absent(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> None:
    failed = read(root / FAILED_LOG.replace(".log", ".execution.json"))
    waiter = read(root / WAITER / "intent.json")
    handles = [intent["generation"], state["child_generation"], waiter["generation"], failed["generation"]]
    handles += [entry["generation"] for entry in result["gpu_commands"]]
    if any((previous.handoff.PROC_ROOT / str(item["pid"])).exists() for item in handles):
        raise RuntimeError("Exit-transition predecessor consumed owner handles remain")
    permitted = {int(pid) for pid in state["restored_keeper"]["generation"]}
    markers = [
        ("OUTPUT_MARIN_NS_TRAINING_PATH_EXIT_CONTROLLER", intent["marker"], permitted),
        ("OUTPUT_MARIN_NS_TRAINING_PATH_EXIT_VALIDATOR", state["child_marker"], set()),
        ("OUTPUT_MARIN_NS_TRAINING_PATH_EXIT_FOLLOWUP", waiter["marker"], set()),
    ]
    markers += [
        ("OUTPUT_MARIN_NS_TRAINING_PATH_EXIT_COMMAND", entry["marker"], set())
        for entry in [*result["gpu_commands"], failed]
    ]
    for name, value, allowed in markers:
        if not set(previous.handoff.marker_processes(name, value)).issubset(allowed):
            raise RuntimeError("Exit-transition predecessor marked descendant remains")
    directory = Path(frozen["helpers"]).parent
    if any((directory / name).exists() for name in ("terminal-intent.json", "terminal-outcome.json")):
        raise ValueError("Exit-transition predecessor unexpectedly claimed a complete terminal")


def retained(root: Path, frozen: dict, result: dict) -> tuple[dict, dict]:
    records = {name: checked(root / name, digest) for name, digest in RECORDS.items()}
    payloads = {}
    for index, entry in enumerate(result["gpu_commands"]):
        trial = frozen["plan"]["trials"][index // 3]
        directory = Path(trial["continuous"] if entry["mode"] == "continuous" else trial["resumed"])
        trajectory = entry["trajectory"]
        manifest = directory / (INTERRUPTED_MANIFEST if entry["mode"] == "interrupted" else "run.json")
        log = Path(entry["log"])
        for path, digest in (
            (manifest, trajectory["manifest_sha256"]),
            (log, entry["log_sha256"]),
            (log.with_suffix(".execution.json"), sha256(log.with_suffix(".execution.json"))),
            *((directory / name, digest) for name, digest in trajectory["history_sha256"].items()),
        ):
            records[str(path.relative_to(root))] = checked(path, digest)
        for step, checkpoint in trajectory["checkpoints"].items():
            work = directory / f"checkpoints/step_{int(step):06d}"
            path = work / "complete.json"
            records[str(path.relative_to(root))] = checked(path, checkpoint["receipt_sha256"])
            for rank, digest in checkpoint["rank_sha256"].items():
                path = work / rank
                identity = file_identity(path)
                if identity["bytes"] != checkpoint["rank_bytes"][int(rank.removeprefix("rank_").removesuffix(".pt"))]:
                    raise ValueError("Exit-transition retained checkpoint size differs")
                name = str(path.relative_to(root))
                value = {"sha256": digest, "identity": identity}
                if name in payloads and payloads[name] != value:
                    raise ValueError("Exit-transition retained checkpoint records conflict")
                payloads[name] = value
    return records, payloads


def accepted_waiter(root: Path) -> dict:
    """Verify the helper-only waiter archive under its separate acceptance schema."""
    path = root / WAITER_CPU_SPEC[0]
    cpu = reporting._receipt(path, WAITER_CPU_SPEC[1])
    helpers = Path(cpu["helpers"])
    inventory = {f"tracking/{name}": digest for name, digest in cpu["helpers_sha256"].items()}
    if (
        cpu["status"] != WAITER_CPU_SPEC[2]
        or cpu["root"] != str(root)
        or helpers != path.parent / "tracking"
        or cpu["environment"]["CUDA_VISIBLE_DEVICES"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["binding_sha256"] != RECORDS[f"run-preparation/{OLD_NAMESPACE}-binding.json"]
        or cpu["original_intent_sha256"] != RECORDS[f"controllers/{OLD_NAMESPACE}/controller.json"]
    ):
        raise ValueError("Exit-transition preserved waiter CPU acceptance differs")
    require_helper_bytes(helpers, cpu["helpers_sha256"])
    require_archive(Path(cpu["archive"]), inventory, alias=False)
    reporting._tests(Path(cpu["tests"]), {"test_ns_training_path_exit_followup": WAITER_CPU_SPEC[3]})
    for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log")):
        if cpu[name] != str(path.parent / filename):
            raise ValueError("Exit-transition preserved waiter artifact path differs")
        reporting._receipt(path, WAITER_CPU_SPEC[1])
        checked(Path(cpu[name]), cpu[f"{name}_sha256"])
    return cpu | {"inventory_sha256": inventory}


@contextmanager
def recorded_keeper_guard(frozen: dict):
    """Serialize a read-only keeper check against the issued manager snapshot."""
    manager = Path("/root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh")
    archived = Path(frozen["helpers"]) / "recovery_keeper_manager.sh"
    if sha256(archived) != frozen["helpers_sha256"][archived.name] or sha256(manager) != sha256(archived):
        raise RuntimeError("Exit-transition active keeper manager differs from issued archive")
    lock_path = Path("/root/dl/runtime/gpu_resource_manager.lock")
    with lock_path.open("a") as lock:
        before = os.fstat(lock.fileno())
        current = lock_path.stat()
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("Exit-transition keeper lock changed")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sha256(manager) != frozen["helpers_sha256"][archived.name]:
            raise RuntimeError("Exit-transition keeper manager changed under lock")
        yield


def evidence(root: Path) -> dict:
    if root != root.resolve(strict=True) or local_node() != NODE or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Exit-transition predecessor requires the original node and hidden CUDA")
    accepted = accepted_archive(root, CPU_SPEC)
    waiter_cpu = accepted_waiter(root)
    older = previous.require_seal(root, PREVIOUS_SEAL_SHA256)
    frozen, intent, state, result = recorded(root)
    summary = transition_structure(root, frozen, intent, state, result)
    if (
        frozen["cpu_receipt_sha256"] != CPU_SPEC[1]
        or frozen["cpu_receipt"] != str(root / CPU_SPEC[0])
        or frozen["helpers_sha256"] != accepted["helpers_sha256"]
        or frozen["files_sha256"] != accepted["files_sha256"]
        or frozen["allocation_predecessor"] != older
        or candidate_files(Path(frozen["candidate"])) != accepted["files_sha256"]
        or read(root / WAITER / "intent.json")["runner_sha256"]
        != waiter_cpu["helpers_sha256"]["ns_training_path_exit_followup.py"]
    ):
        raise ValueError("Exit-transition preserved accepted source differs")
    require_helper_bytes(Path(frozen["helpers"]), frozen["helpers_sha256"])
    require_archive(Path(frozen["archive"]), accepted["inventory_sha256"], alias=True)
    require_absent(root, frozen, intent, state, result)
    records, payloads = retained(root, frozen, result)
    records |= {name: checked(root / name, value["sha256"]) for name, value in older["evidence"].items()}
    payloads |= older["checkpoint_payloads"]
    for path, digest in (
        (root / CPU_SPEC[0], CPU_SPEC[1]),
        (root / previous.SEAL, PREVIOUS_SEAL_SHA256),
        (Path(frozen["archive"]), frozen["archive_sha256"]),
        (root / WAITER_CPU_SPEC[0], WAITER_CPU_SPEC[1]),
        *((Path(waiter_cpu[name]), waiter_cpu[f"{name}_sha256"]) for name in ("archive", "tests", "log")),
    ):
        records[str(path.relative_to(root))] = checked(path, digest)
    for name, digest in waiter_cpu["inventory_sha256"].items():
        path = Path(waiter_cpu["helpers"]).parent / name
        records[str(path.relative_to(root))] = checked(path, digest)
    require_keeper_shape(state["restored_keeper"])
    with recorded_keeper_guard(frozen):
        if not same_keeper(previous.handoff.keeper_observation(), state["restored_keeper"]):
            raise RuntimeError("Exit-transition predecessor exact restored keeper differs")
    if state["restored_keeper"]["devices"] != frozen["devices"]:
        raise ValueError("Exit-transition predecessor physical GPUs differ")
    require_absent(root, frozen, intent, state, result)
    if any(file_identity(root / name) != item["identity"] for name, item in (records | payloads).items()):
        raise ValueError("Exit-transition predecessor evidence changed during checking")
    return {
        "status": STATUS,
        "root": str(root),
        "node": NODE,
        "evidence": records,
        "checkpoint_payloads": payloads,
        "retained_summary": summary,
        "original_retained_summary": older["original_retained_summary"],
        "recovery_retained_summary": older["retained_summary"],
        "restored_keeper": state["restored_keeper"],
        "gpu_execution_authorized": False,
        "production_authorized": False,
        "scope": SCOPE,
    }


def seal(root: Path) -> dict:
    path = root / SEAL
    if path.exists() or path.is_symlink():
        raise FileExistsError("Exit-transition predecessor seal is already consumed")
    saved = evidence(root) | {"verifier_sha256": sha256(Path(__file__)), "time": datetime.now(UTC).isoformat()}
    with tempfile.TemporaryDirectory(prefix=".exit-transition-seal-", dir=path.parent) as directory:
        staged = Path(directory) / "record.json"
        with staged.open("x") as stream:
            json.dump(saved, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
    return read(path)


def require_seal(root: Path, digest: str) -> dict:
    path = root / SEAL
    identity, raw = file_identity(path), path.read_bytes()
    saved = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest() != digest
        or saved["status"] != STATUS
        or saved["root"] != str(root)
        or saved["scope"] != SCOPE
        or saved["verifier_sha256"] != sha256(Path(__file__))
        or saved["gpu_execution_authorized"] is not False
        or saved["production_authorized"] is not False
        or any(checked(root / name, item["sha256"]) != item for name, item in saved["evidence"].items())
        or any(file_identity(root / name) != item["identity"] for name, item in saved["checkpoint_payloads"].items())
    ):
        raise ValueError("Exit-transition predecessor seal differs")
    frozen, intent, state, result = recorded(root)
    if transition_structure(root, frozen, intent, state, result) != saved["retained_summary"]:
        raise ValueError("Exit-transition predecessor saved coverage differs")
    previous.require_seal(root, PREVIOUS_SEAL_SHA256)
    require_absent(root, frozen, intent, state, result)
    if path.read_bytes() != raw or file_identity(path) != identity:
        raise ValueError("Exit-transition predecessor seal changed")
    return saved


failed_structure = previous.failed_structure
require_absent_owners = previous.require_absent_owners
handoff = previous.handoff
