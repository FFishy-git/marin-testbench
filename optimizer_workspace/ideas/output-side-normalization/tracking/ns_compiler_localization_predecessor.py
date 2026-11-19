# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal completed original and actual-LR diagnostics before separate compiler localization."""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath

import ns_training_path_lr_queue as queue
import ns_training_path_lr_queue_reporting as reporting
import owned_output_handoff as handoff
from fused_attribution_validation import INTERRUPTED_MANIFEST, require_keeper_shape
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_failure_handoff import keeper_guard, same_keeper
from selected_lr_release_owner import read, sha256

QUEUE_SPEC = (reporting.CPU, reporting.CPU_SHA256, queue.CPU_STATUS, queue.TEST_COUNT)
EXECUTION_SPEC = (
    "run-preparation/ns-compiler-localization-execution-cpu/20261119T1225/receipt.json",
    "577b13e053e5b823a58f0e1ed6176c4f40141d9fee399327c59ec292951c3936",
    "ns_compiler_localization_execution_cpu_archive_verified_gpu_unissued",
    23,
)
OWNER_SPEC = (
    queue.PARENT,
    queue.PARENT_SHA256,
    "ns_training_path_actual_lr_execution_owner_cpu_archive_verified_gpu_unissued",
    53,
)
REPORTER_SHA256 = "c5264e454fcec70e70fe60f342f6af7093a356e02770191d6cce347204e17a54"
GATES = {
    "actual_lr_complete_trajectories",
    "boundary_all_rank_continuation",
    "post_cached_root_all_rank_continuation",
    "scalar_loss_continuation",
    "captured_row_update_work",
}
STATUS = "original_and_actual_lr_terminals_sealed_for_compiler_localization_owner_unissued"
SEAL = "run-preparation/ns-compiler-localization-predecessor-seal-20261119.json"
SCOPE = (
    "Read-only predecessor handoff for separate compiler localization. Require the original "
    "waiter's canonical terminal and actual-LR queue's sole canonical terminal, all 162 "
    "commands/54 decisions, unchanged rejected cases, original/actual owners and queue/phase "
    "descendants absent, and exact restored keeper. Authenticate completed reconstruction "
    "receipts and checkpoint file identities; never deserialize tensor payloads or invoke "
    "another terminal. No GPU release, compiler execution, continuation correction, production, "
    "quality, efficiency or SwanLab authority. Failed or consumed predecessor outcomes stay failed."
)


def hidden_cuda() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Localization predecessor inspection requires hidden CUDA")


def original_queue(root: Path) -> tuple[dict, dict]:
    """Bind the single accepted queue generation before observing any live handle."""
    hidden_cuda()
    work = root / queue.WORK
    path = work / "intent.json"
    state_path = work / "state.json"
    identity, raw = file_identity(path), path.read_bytes()
    state_identity = file_identity(state_path)
    intent, state = json.loads(raw), read(work / "state.json")
    if (
        root != root.resolve(strict=True)
        or hashlib.sha256(raw).hexdigest() != reporting.INTENT_SHA256
        or intent["status"] != "acknowledged"
        or intent["cpu_receipt"] != str(root / reporting.CPU)
        or intent["command"] != queue.command(root, root / reporting.CPU)
        or intent["generation"]["command"] != intent["command"]
        or intent["generation"]["pid"] != intent["pid"]
        or intent["generation"]["start_ticks"] != intent["start_ticks"]
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or not intent["marker"]
        or intent["cpu_receipt_sha256"] != reporting.CPU_SHA256
        or intent["parent_cpu_receipt_sha256"] != queue.PARENT_SHA256
        or intent["runner_sha256"] != sha256(Path(queue.__file__))
        or intent["interval"] != queue.INTERVAL
        or intent["scope"] != queue.SCOPE
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or state["status"] not in reporting.STATUSES
        or ("error" in state and state["status"] != "needs_attention")
        or [entry["phase"] for entry in state["phases"]] != list(reporting.PHASES[: len(state["phases"])])
        or len(state["phases"]) > len(reporting.PHASES)
        or any(
            state[name] != intent[name]
            for name in ("cpu_receipt_sha256", "runner_sha256", "parent_cpu_receipt_sha256", "interval", "scope")
        )
        or file_identity(path) != identity
        or file_identity(state_path) != state_identity
    ):
        raise ValueError("Localization predecessor original actual-LR queue identity differs")
    return intent, state


def readiness(root: Path) -> dict:
    """Observe exact original/actual owners; a saved terminal or timeout cannot imply absence."""
    intent, state = original_queue(root)
    observer = queue.predecessor.followup
    original = queue.predecessor.readiness(root)
    owners = {f"original_{name}": value for name, value in original["owners"].items()}
    owners["actual_lr_queue"] = observer.handle(
        intent["generation"], root / queue.WORK, queue.MARKER, intent["marker"], queue.ENVIRONMENT
    )
    marked = handoff.marker_processes(queue.MARKER, intent["marker"])
    entries = list(state.get("phases", []))
    if state.get("active_phase") and state["active_phase"] not in entries:
        entries.append(state["active_phase"])
    for entry in entries:
        if entry.get("spawn_status") == "acknowledged":
            owners[f"queue_phase_{entry['pid']}"] = observer.handle(
                entry["generation"], Path(entry["cwd"]), queue.PHASE_MARKER, entry["marker"], queue.ENVIRONMENT
            )
        marked.update(handoff.marker_processes(queue.PHASE_MARKER, entry["marker"]))
    binding, helpers, state_path, _, result_path = queue.contract.locations(root)
    if state.get("issued_controller"):
        frozen, issued = read(binding), read(helpers.parent / "controller.json")
        if (
            sha256(binding) != state["issued_binding_sha256"]
            or issued != state["issued_controller"]
            or issued["status"] != "acknowledged"
            or issued["binding_sha256"] != sha256(binding)
            or frozen["controller_command"] != queue.owner.controller_command(root)
            or issued["command"] != frozen["controller_command"]
            or frozen["cpu_receipt_sha256"] != queue.PARENT_SHA256
            or frozen["predecessor_acceptance"]["seal_sha256"] != state["predecessor_seal_sha256"]
        ):
            raise ValueError("Localization predecessor issued actual-LR generation differs")
        saved = read(state_path) if state_path.is_file() else {}
        owners["actual_lr_controller"] = observer.handle(
            issued["generation"],
            helpers,
            queue.contract.CONTROLLER_MARKER,
            issued["marker"],
            queue.owner.CONTROLLER_ENVIRONMENT | {queue.owner.PARENT_PROOF_MARKER: issued["parent_proof"]},
        )
        if saved.get("spawn_status") == "acknowledged":
            owners["actual_lr_validator"] = observer.handle(
                saved["child_generation"],
                Path(frozen["candidate"]),
                queue.contract.CHILD_MARKER,
                saved["child_marker"],
                queue.contract.ENVIRONMENT,
            )
            marked.update(handoff.marker_processes(queue.contract.CHILD_MARKER, saved["child_marker"]))
        for entry in read(result_path).get("gpu_commands", []) if result_path.is_file() else []:
            owners[f"actual_lr_command_{entry['generation']['pid']}"] = observer.handle(
                entry["generation"],
                Path(frozen["candidate"]),
                queue.contract.COMMAND_MARKER,
                entry["marker"],
                queue.contract.ENVIRONMENT,
            )
            marked.update(handoff.marker_processes(queue.contract.COMMAND_MARKER, entry["marker"]))
        controller_descendants = handoff.marker_processes(queue.contract.CONTROLLER_MARKER, issued["marker"])
        permitted = {int(pid) for pid in saved.get("restored_keeper", {}).get("generation", {})}
        marked.update(
            {
                pid: value
                for pid, value in controller_descendants.items()
                if pid not in permitted and pid != issued["pid"]
            }
        )
    if marked:
        owners["marked_descendants"] = {"status": "marked_descendants_present", "pids": sorted(marked)}
    ready = all(value["status"] == "absent" for value in owners.values())
    if ready and (
        not binding.is_file()
        or not state.get("issued_controller")
        or state["status"] != "actual_lr_training_path_terminal_reconstructed"
    ):
        raise RuntimeError("Localization predecessor queue exited without its complete original actual-LR terminal")
    return {
        "status": "original_and_actual_lr_owners_absent" if ready else "waiting_for_original_and_actual_lr_owners",
        "ready": ready,
        "owners": owners,
        "queue_status": state["status"],
    }


def queue_report(root: Path) -> tuple[dict, dict, dict, dict]:
    """Authenticate completed queue evidence in a fresh archived reporting subprocess."""
    helpers = Path(reporting.__file__).resolve().parent
    reporter = helpers / "ns_training_path_lr_queue_reporting.py"
    file_identity(reporter)
    if sha256(reporter) != REPORTER_SHA256:
        raise ValueError("Localization predecessor accepted queue reporter source changed")
    command = [
        queue.contract.PYTHON,
        "-B",
        "-P",
        "-c",
        "import json,sys; from pathlib import Path; import ns_training_path_lr_queue_reporting as report; "
        "assert Path(report.__file__).resolve()==Path(sys.argv[2]); "
        "print(json.dumps(report.evidence(Path(sys.argv[1])),allow_nan=False))",
        str(root),
        str(helpers / "ns_training_path_lr_queue_reporting.py"),
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
        raise RuntimeError(f"Localization predecessor archived queue report failed: {result.stderr[-4000:]}")
    saved = json.loads(result.stdout)
    if not isinstance(saved, list) or len(saved) != 4 or any(not isinstance(item, dict) for item in saved):
        raise ValueError("Localization predecessor archived queue report schema differs")
    return tuple(saved)


def decision_summary(cases: list[dict], names: list[str]) -> dict:
    """Require complete unique case coverage and retain every rejected predicate."""
    if (
        len(set(names)) != len(names)
        or [case["name"] for case in cases] != names
        or any(
            set(case["gates"]) != GATES
            or any(type(value) is not bool for value in case["gates"].values())
            or type(case["accepted"]) is not bool
            or case["accepted"] != all(case["gates"].values())
            for case in cases
        )
    ):
        raise ValueError("Localization predecessor actual-LR decision coverage or gate semantics differ")
    return {
        "accepted_cases": [case["name"] for case in cases if case["accepted"]],
        "rejected_cases": [case["name"] for case in cases if not case["accepted"]],
        "all_cases_accepted": all(case["accepted"] for case in cases),
    }


def retain_record(root: Path, records: dict, path: Path, item: dict) -> None:
    """Reject duplicate evidence with conflicting digests or original file identities."""
    name = str(path.relative_to(root))
    relative = PurePosixPath(name)
    if ".." in relative.parts or str(relative) != name or relative.is_absolute():
        raise ValueError("Localization predecessor evidence path escapes its canonical root")
    if name in records and records[name] != item:
        raise ValueError("Localization predecessor retained evidence identities conflict")
    records[name] = item


def evidence(root: Path) -> dict:
    """Bind completed numerical receipts and unchanged checkpoint identities without tensor reads."""
    if not readiness(root)["ready"]:
        raise RuntimeError("Localization predecessor original or actual-LR owner remains present")
    accepted_execution = accepted_archive(root, EXECUTION_SPEC)
    accepted_queue = accepted_archive(root, QUEUE_SPEC)
    accepted_owner = accepted_archive(root, OWNER_SPEC)
    cpu, queue_intent, queue_state, terminal = queue_report(root)
    intent, saved_queue = original_queue(root)
    if (
        cpu != accepted_queue
        or queue_intent != intent
        or queue_state != saved_queue
        or queue_state["status"] != "actual_lr_training_path_terminal_reconstructed"
        or len(queue_state["phases"]) != 6
    ):
        raise ValueError("Localization predecessor complete queue/archive attribution differs")
    binding, helpers, state_path, _, result_path = queue.contract.locations(root)
    frozen, state, result = read(binding), read(state_path), read(result_path)
    issued = read(helpers.parent / "controller.json")
    terminal_path, attempt, outcome, _ = queue.controller.terminal_paths(root, issued)
    previous = queue.predecessor.require_seal(root, queue_state["predecessor_seal_sha256"])
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_archive(Path(frozen["archive"]), accepted_owner["inventory_sha256"], alias=True)
    expected_plan = queue.controller.validation.plan(root, Path(frozen["candidate"]))
    decisions = decision_summary(result["results"], [case["name"] for case in expected_plan["trials"]])
    if (
        frozen["status"] != queue.contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["candidate"] != accepted_owner["candidate"]
        or frozen["cpu_receipt"] != str(root / queue.PARENT)
        or frozen["cpu_receipt_sha256"] != OWNER_SPEC[1]
        or frozen["helpers_sha256"] != accepted_owner["helpers_sha256"]
        or frozen["files_sha256"] != accepted_owner["files_sha256"]
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["allocation_predecessor"] != previous
        or frozen["devices"] != previous["restored_keeper"]["devices"]
        or frozen["devices"] != terminal["keeper"]["devices"]
        or frozen["plan"] != expected_plan
        or frozen["commands"] != expected_plan["commands"]
        or any(result[name] != value for name, value in expected_plan.items())
        or frozen["plan"]["trials"] != result["trials"]
        or len(frozen["plan"]["trials"]) != 54
        or len(frozen["commands"]) != 162
        or len(result["results"]) != 54
        or len(result["gpu_commands"]) != 162
        or [case["name"] for case in result["results"]] != [case["name"] for case in frozen["plan"]["trials"]]
        or result["commands"] != frozen["commands"]
        or result["status"] != queue.controller.validation.STATUS
        or result["owner"]
        != {
            "controller_pid": state["controller_pid"],
            "controller_start_ticks": state["controller_start_ticks"],
            "validator_pid": state["child_pid"],
            "validator_start_ticks": state["child_start_ticks"],
        }
        or result["binding_sha256"] != sha256(binding)
        or state["status"] != "validation_complete"
        or state["binding_sha256"] != sha256(binding)
        or state["controller_pid"] != issued["pid"]
        or state["controller_start_ticks"] != issued["start_ticks"]
        or state["child_generation"]["pid"] != state["child_pid"]
        or state["child_generation"]["start_ticks"] != state["child_start_ticks"]
        or state["child_generation"]["parent"] != issued["pid"]
        or state["child_generation"]["command"] != frozen["validator_command"]
        or state["allocation_predecessor"] != previous
        or state["result_sha256"] != sha256(result_path)
        or any(
            type(state[name]) is not int or state[name] != 0
            for name in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state["marked_descendants_absent"] is not True
        or state["keeper_release_complete"] is not True
        or any(name in state for name in ("error", "cleanup_error", "deferred_signals"))
        or "error" in result
        or "error" in queue_state
        or type(queue_state["all_cases_accepted"]) is not bool
        or any(
            value[name] != expected
            for value in (result, state, terminal, queue_state)
            for name, expected in decisions.items()
        )
        or queue_state["original_decisions"] != previous["original_decisions"]
        or terminal_path != Path(queue_state["terminal"])
        or sha256(terminal_path) != queue_state["terminal_sha256"]
        or terminal["status"] != queue.controller.TERMINAL_STATUS
        or terminal["terminal_time"] != state["time"]
        or terminal["result_sha256"] != sha256(result_path)
        or terminal["binding_sha256"] != sha256(binding)
        or terminal["predecessor_acceptance"] != frozen["predecessor_acceptance"]
        or terminal["reconstruction_intent_sha256"] != sha256(attempt)
        or read(outcome)["status"] != "complete"
        or read(outcome)["terminal_sha256"] != sha256(terminal_path)
        or read(outcome)["output"] != str(terminal_path)
        or read(outcome)["reconstruction_intent_sha256"] != sha256(attempt)
        or terminal["production_authorized"] is not False
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
    ):
        raise ValueError("Localization predecessor actual-LR coverage, rejection or restoration differs")
    for phase in queue_state["phases"]:
        reporting.require_phase(root, queue_state, phase)
    original_claim = read(attempt)
    if (
        original_claim["status"] != "reconstruction_claimed"
        or original_claim["output"] != str(terminal_path)
        or original_claim["binding_sha256"] != sha256(binding)
        or original_claim["controller_pid"] != issued["pid"]
        or original_claim["controller_start_ticks"] != issued["start_ticks"]
        or original_claim["controller_marker"] != issued["marker"]
        or original_claim["verifier_sha256"] != frozen["helpers_sha256"]["ns_training_path_lr_gpu_controller.py"]
        or terminal["verifier_sha256"] != original_claim["verifier_sha256"]
        or terminal["controller_pid_absent"] != issued["pid"]
        or terminal["validator_pid_absent"] != state["child_pid"]
    ):
        raise ValueError("Localization predecessor original canonical reconstruction attribution differs")
    times = [datetime.fromisoformat(value["time"]) for value in (result, state, terminal, queue_state)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("Localization predecessor terminal chronology differs")
    claim = datetime.fromisoformat(original_claim["time"])
    if claim.tzinfo is None or not times[1] <= claim <= times[2]:
        raise ValueError("Localization predecessor original reconstruction claim chronology differs")
    paths = {root / name: digest for name, digest in terminal["evidence_sha256"].items()}
    paths |= {terminal_path: sha256(terminal_path), attempt: sha256(attempt), outcome: sha256(outcome)}
    paths |= {
        root / queue.WORK / name: sha256(root / queue.WORK / name) for name in ("intent.json", "state.json", "queue.log")
    }
    paths[root / queue.predecessor.SEAL] = queue_state["predecessor_seal_sha256"]
    for entry in queue_state["phases"]:
        paths[Path(entry["log"])] = entry["log_sha256"]
    paths |= {Path(frozen["candidate"]) / name: digest for name, digest in frozen["files_sha256"].items()}
    paths |= {helpers / name: digest for name, digest in frozen["helpers_sha256"].items()}
    for spec, receipt in ((EXECUTION_SPEC, accepted_execution), (QUEUE_SPEC, cpu), (OWNER_SPEC, accepted_owner)):
        paths[root / spec[0]] = spec[1]
        for name in ("archive", "tests", "log"):
            paths[Path(receipt[name])] = receipt[f"{name}_sha256"]
        paths |= {Path(receipt["helpers"]).parent / name: digest for name, digest in receipt["inventory_sha256"].items()}
    for index, (entry, command) in enumerate(zip(result["gpu_commands"], frozen["commands"], strict=True)):
        planned, mode = expected_plan["trials"][index // 3], queue.contract.MODES[index % 3]
        log = queue.contract.locations(root)[3] / f"{planned['name']}_{mode}.log"
        if (
            entry["command"] != command
            or entry["trial"] != planned["name"]
            or entry["mode"] != mode
            or entry["trajectory"] != result["results"][index // 3]["trajectories"][mode]
            or entry["generation"]["parent"] != state["child_pid"]
            or any(type(entry[name]) is not int or entry[name] != 0 for name in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
        ):
            raise ValueError("Localization predecessor actual-LR command execution or cleanup differs")
        queue.controller.validation.require_entry(entry, command, log)
        paths[log] = entry["log_sha256"]
        paths[log.with_suffix(".execution.json")] = sha256(log.with_suffix(".execution.json"))
    claim_ns = int(claim.timestamp() * 1_000_000_000)
    payloads = checkpoint_payloads(root, result["results"], claim_ns)
    for case in result["results"]:
        planned = next(item for item in expected_plan["trials"] if item["name"] == case["name"])
        if any(case[name] != value for name, value in planned.items()) or set(case["trajectories"]) != set(
            queue.contract.MODES
        ):
            raise ValueError("Localization predecessor actual-LR case configuration or trajectories differ")
        for mode, trajectory in case["trajectories"].items():
            directory = Path(case["continuous"] if mode == "continuous" else case["resumed"])
            expected_steps = (
                {str(queue.contract.BOUNDARY)}
                if mode == "interrupted"
                else {str(step) for step in queue.contract.CHECKPOINTS}
            )
            if (
                set(trajectory["checkpoints"]) != expected_steps
                or trajectory["updates"]
                != (queue.contract.BOUNDARY if mode == "interrupted" else queue.contract.ENDPOINT)
                or trajectory["tokens"] != trajectory["updates"] * queue.contract.BATCH_TOKENS
            ):
                raise ValueError("Localization predecessor actual-LR trajectory endpoint coverage differs")
            paths[directory / ("run.json" if mode != "interrupted" else INTERRUPTED_MANIFEST)] = trajectory[
                "manifest_sha256"
            ]
            paths[directory / "config.json"] = sha256(directory / "config.json")
            paths |= {directory / name: digest for name, digest in trajectory["history_sha256"].items()}
            for step, checkpoint in trajectory["checkpoints"].items():
                folder = directory / f"checkpoints/step_{int(step):06d}"
                paths[folder / "complete.json"] = checkpoint["receipt_sha256"]
                if len(checkpoint["rank_bytes"]) != queue.contract.WORLD or set(checkpoint["rank_sha256"]) != {
                    f"rank_{rank}.pt" for rank in range(queue.contract.WORLD)
                }:
                    raise ValueError("Localization predecessor actual-LR checkpoint rank coverage differs")
    retained = dict(previous["evidence"])
    for name, item in frozen["authenticated_files"].items():
        path = Path(name)
        if not path.is_relative_to(root) or file_identity(path) != item["identity"]:
            raise ValueError("Localization predecessor operation-authenticated input changed")
        retain_record(root, retained, path, item)
    for path, digest in paths.items():
        identity = file_identity(path)
        if sha256(path) != digest or file_identity(path) != identity:
            raise ValueError("Localization predecessor retained evidence changed during authentication")
        retain_record(root, retained, path, {"sha256": digest, "identity": identity})
    result = {
        "status": STATUS,
        "root": str(root),
        "scope": SCOPE,
        "evidence": retained,
        "original_checkpoint_payloads": previous["checkpoint_payloads"],
        "actual_lr_checkpoint_payloads": payloads,
        "original_terminal": previous["original_terminal"],
        "original_terminal_sha256": previous["original_terminal_sha256"],
        "actual_lr_terminal": str(terminal_path),
        "actual_lr_terminal_sha256": sha256(terminal_path),
        "original_decisions": previous["original_decisions"],
        "actual_lr_decisions": decisions,
        "restored_keeper": terminal["keeper"],
        "controller_marker": issued["marker"],
        "gpu_execution_authorized": False,
        "production_authorized": False,
    }
    require_identities(root, result)
    if not readiness(root)["ready"]:
        raise RuntimeError("Localization predecessor owners changed during authentication")
    return result


def checkpoint_payloads(root: Path, cases: list[dict], claim_ns: int) -> dict:
    """Bind complete rank payload identities to hashes retained by the original reconstructor."""
    payloads = {}
    for case in cases:
        for mode, trajectory in case["trajectories"].items():
            directory = Path(case["continuous"] if mode == "continuous" else case["resumed"])
            for step, checkpoint in trajectory["checkpoints"].items():
                folder = directory / f"checkpoints/step_{int(step):06d}"
                if len(checkpoint["rank_bytes"]) != queue.contract.WORLD or set(checkpoint["rank_sha256"]) != {
                    f"rank_{rank}.pt" for rank in range(queue.contract.WORLD)
                }:
                    raise ValueError("Localization predecessor checkpoint payload rank coverage differs")
                for rank, size in enumerate(checkpoint["rank_bytes"]):
                    path = folder / f"rank_{rank}.pt"
                    identity = file_identity(path)
                    if identity["bytes"] != size or identity["ctime_ns"] > claim_ns:
                        raise ValueError("Localization predecessor checkpoint changed after original reconstruction")
                    retain_record(
                        root,
                        payloads,
                        path,
                        {
                            "sha256": checkpoint["rank_sha256"][path.name],
                            "identity": identity,
                        },
                    )
    return payloads


def require_identities(root: Path, saved: dict) -> None:
    for group in ("evidence", "original_checkpoint_payloads", "actual_lr_checkpoint_payloads"):
        if not saved[group]:
            raise ValueError("Localization predecessor sealed evidence is empty")
        for name, item in saved[group].items():
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != name
                or file_identity(root / name) != item["identity"]
            ):
                raise ValueError("Localization predecessor sealed evidence changed")


def seal(root: Path) -> dict:
    """Publish one immutable handoff after exact original and actual-LR restoration."""
    hidden_cuda()
    path = root / SEAL
    if path.exists() or path.is_symlink():
        raise FileExistsError("Localization predecessor seal is one-shot")
    saved = evidence(root)
    with keeper_guard():
        keeper = handoff.keeper_observation()
        require_keeper_shape(keeper)
        require_identities(root, saved)
        if not same_keeper(keeper, saved["restored_keeper"]) or not readiness(root)["ready"]:
            raise RuntimeError("Localization predecessor exact restoration or owner absence changed")
        queue.controller.publish_terminal_record(path, saved)
    return read(path)


def require_seal(root: Path, digest: str) -> dict:
    """Check the caller-bound handoff and original file identities after resource transfer."""
    hidden_cuda()
    path = root / SEAL
    identity, raw = file_identity(path), path.read_bytes()
    saved = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest() != digest
        or file_identity(path) != identity
        or saved["status"] != STATUS
        or saved["root"] != str(root)
        or saved["scope"] != SCOPE
        or saved["gpu_execution_authorized"] is not False
        or saved["production_authorized"] is not False
        or not saved["actual_lr_checkpoint_payloads"]
        or not saved["original_checkpoint_payloads"]
        or any(
            len(saved[group]["accepted_cases"]) + len(saved[group]["rejected_cases"]) != count
            or len(set(saved[group]["accepted_cases"]) | set(saved[group]["rejected_cases"])) != count
            or type(saved[group]["all_cases_accepted"]) is not bool
            or saved[group]["all_cases_accepted"] != (not saved[group]["rejected_cases"])
            for group, count in (("original_decisions", 18), ("actual_lr_decisions", 54))
        )
        or any(
            saved["evidence"].get(str(Path(saved[name]).relative_to(root)), {}).get("sha256") != saved[f"{name}_sha256"]
            for name in ("original_terminal", "actual_lr_terminal")
        )
    ):
        raise ValueError("Localization predecessor caller-bound seal differs")
    require_identities(root, saved)
    _, state = original_queue(root)
    previous = queue.predecessor.require_seal(root, state["predecessor_seal_sha256"])
    original_result = read(queue.contract.locations(root)[4])
    expected = decision_summary(original_result["results"], [case["name"] for case in original_result["trials"]])
    terminal_path, attempt, _, _ = queue.controller.terminal_paths(root, state["issued_controller"])
    claim = datetime.fromisoformat(read(attempt)["time"])
    if claim.tzinfo is None:
        raise ValueError("Localization predecessor original reconstruction claim lacks absolute time")
    payloads = checkpoint_payloads(root, original_result["results"], int(claim.timestamp() * 1_000_000_000))
    if (
        saved["original_terminal"] != previous["original_terminal"]
        or saved["original_terminal_sha256"] != previous["original_terminal_sha256"]
        or saved["original_decisions"] != previous["original_decisions"]
        or saved["original_checkpoint_payloads"] != previous["checkpoint_payloads"]
        or saved["actual_lr_decisions"] != expected
        or saved["actual_lr_terminal"] != state["terminal"]
        or saved["actual_lr_terminal_sha256"] != state["terminal_sha256"]
        or saved["actual_lr_checkpoint_payloads"] != payloads
        or not same_keeper(saved["restored_keeper"], read(terminal_path)["keeper"])
        or saved["controller_marker"] != state["issued_controller"]["marker"]
    ):
        raise ValueError("Localization predecessor seal rebound completed original decisions")
    if not readiness(root)["ready"] or file_identity(path) != identity:
        raise RuntimeError("Localization predecessor original owners or seal changed")
    return saved
