# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize only the separate attribution release's acknowledged full-budget ranks."""

import hashlib
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .fused_attribution_production_contract import (
    ACCEPTANCE_STATUS,
    CHILD_MARKER,
    CONTROLLER_CPU_STATUS,
    CPU_STATUS,
    ENVIRONMENT,
    FAMILIES,
    FROZEN_STATUS,
    MONITOR_INTERVAL,
    PREPARATION,
    PREPARED_STATUS,
    PYTHON,
    TRAINING_TIMEOUT,
    WAVES,
    WORKER_MARKER,
    WORLD,
    commands,
    config_name,
    node_for,
    output_path,
    paths,
    require_config,
    wave_spec,
    worker_command,
    worker_path,
)
from .fused_attribution_validation_contract import (
    FROZEN_STATUS as VALIDATION_STATUS,
)
from .fused_attribution_validation_contract import (
    MODES,
    controller_command,
    locations,
    trial_specs,
    validation_commands,
    validator_command,
)
from .fused_attribution_validation_contract import (
    PREPARATION as VALIDATION_PREPARATION,
)
from .selected_lr_release_owner import candidate_files, observed, read, require_archive, sha256

TASK = Path(__file__).resolve().parents[1]
CHANGES = {
    "experiment/train.py",
    "experiment/fused_attribution_production_owner.py",
    "experiment/fused_attribution_production_contract.py",
    "experiment/tests/test_fused_attribution_production_owner.py",
}
NUMERICAL_STATUS = "passed_fused_attribution_actual_config_continuation_and_work_gates"
ACK_TIMEOUT = 30
CONTROLLER_TESTS = (
    "test_fused_attribution_production.py",
    "test_fused_attribution_production_handoff.py",
    "test_prepare_fused_attribution_production.py",
    "test_fused_attribution_production_owner.py",
    "test_fused_attribution_run_history.py",
)
BASE_CHECKS = {
    "actual_full_budget_config",
    "compiled_model",
    "cached_root_checkpoint",
    "boundary_continuation",
    "post_cached_root_continuation",
    "fixed_age_history",
    "all_rank_checkpoint_identity",
    "loss_continuation",
}
FUSED_CHECKS = {"captured_diagonal_rows", "outer_diagonal_update_parity", "output_work_retained"}


def cpu_command() -> list[str]:
    return [
        PYTHON,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-q",
        "-m",
        "not slow and not integration and not data_integration and not cluster "
        "and not requires_cluster and not docker and not manual",
        "experiment/tests",
    ]


def controller_cpu_command() -> list[str]:
    command = cpu_command()
    return [command[0], "-P", "-B", *command[1:-1], *(f"tracking/{name}" for name in CONTROLLER_TESTS)]


def same_generation(actual: dict, expected: dict) -> bool:
    return (
        actual.get("status", "S") not in ("Z", "X", "x")
        and expected.get("status", "S") not in ("Z", "X", "x")
        and all(actual[key] == expected[key] for key in ("pid", "start_ticks", "group", "session", "command"))
    )


def require_helpers(helpers: Path, hashes: dict) -> None:
    if (
        helpers.is_symlink()
        or not helpers.is_dir()
        or {path.name for path in helpers.iterdir() if path.name != "__pycache__"} != set(hashes)
        or any(
            Path(name).name != name
            or (helpers / name).is_symlink()
            or not (helpers / name).is_file()
            or sha256(helpers / name) != digest
            for name, digest in hashes.items()
        )
    ):
        raise RuntimeError("Attribution production helper closure differs from its frozen inventory")


def require_cpu_evidence(root: Path, path: Path, status: str, candidate: Path) -> dict:
    receipt = read(path)
    if (
        receipt["status"] != status
        or receipt["root"] != str(root)
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or any(key in receipt for key in ("error", "errors"))
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
    ):
        raise RuntimeError("Attribution production requires intact successful hidden-CUDA CPU evidence")
    if status == CPU_STATUS and (
        receipt["candidate"] != str(candidate) or receipt["cwd"] != str(candidate) or receipt["command"] != cpu_command()
    ):
        raise RuntimeError("Attribution candidate CPU invocation differs")
    return receipt


def require_controller_archive(path: Path, expected: dict) -> None:
    """Controller CPU archives contain only regular helpers and source fixtures."""
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise RuntimeError("Attribution controller archive inventory differs")
        for member in members:
            if not member.isfile():
                raise RuntimeError("Attribution controller archived source is aliased")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected[member.name]:
                    raise RuntimeError("Attribution controller archived bytes differ")


def require_evidence(root: Path, hashes: dict, required: set[Path]) -> None:
    """Check rank-boundary evidence; full checkpoint-map verification belongs to locked preflight."""
    for name in hashes:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Attribution accepted evidence has a nonlocal path")
    for path in required:
        name = str(path.relative_to(root))
        if name not in hashes or path.is_symlink() or sha256(path) != hashes[name]:
            raise RuntimeError("Attribution accepted source or ownership evidence changed")


def require_validation(root: Path, prepared: dict, release: dict) -> dict:
    """Bind accepted original reconstruction metadata, never execute a tensor reader here."""
    binding, helpers, state_path, work, receipt_path = locations(root)
    intent_path = helpers.parent / "controller.json"
    acceptance_path = root / "run-preparation/fused-attribution-production-validation-acceptance-20261118.json"
    if (
        release["validation_binding"] != str(binding)
        or release["validation_binding_sha256"] != sha256(binding)
        or release["validation_binding_sha256"] != prepared["validation_binding_sha256"]
        or release["validation_acceptance"] != str(acceptance_path)
        or release["validation_acceptance_sha256"] != sha256(acceptance_path)
    ):
        raise RuntimeError("Attribution production original validation or acceptance identity differs")
    original, accepted, state, receipt, intent = (
        read(path) for path in (binding, acceptance_path, state_path, receipt_path, intent_path)
    )
    previous = Path(original["candidate"])
    owner = {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
    if (
        original["status"] != VALIDATION_STATUS
        or original["root"] != str(root)
        or original["helpers"] != str(helpers)
        or previous != previous.resolve()
        or original["files_sha256"] != candidate_files(previous)
        or any(
            prepared[key] != original[source]
            for key, source in (
                ("parent_candidate", "candidate"),
                ("parent_cpu_receipt", "cpu_receipt"),
                ("parent_cpu_receipt_sha256", "cpu_receipt_sha256"),
                ("parent_controller_cpu_receipt", "controller_cpu_receipt"),
                ("parent_controller_cpu_receipt_sha256", "controller_cpu_receipt_sha256"),
                ("parent_archive", "archive"),
                ("parent_archive_sha256", "archive_sha256"),
                ("parent_preparation_sha256", "preparation_sha256"),
                ("data_receipt_sha256", "data_receipt_sha256"),
            )
        )
        or original["archive_sha256"] != sha256(Path(original["archive"]))
        or original["cpu_receipt_sha256"] != sha256(Path(original["cpu_receipt"]))
        or original["controller_cpu_receipt_sha256"] != sha256(Path(original["controller_cpu_receipt"]))
        or original["preparation_sha256"] != sha256(previous / VALIDATION_PREPARATION)
        or accepted["status"] != ACCEPTANCE_STATUS
        or accepted["root"] != str(root)
        or accepted["binding_sha256"] != sha256(binding)
        or accepted["state_sha256"] != sha256(state_path)
        or accepted["receipt_sha256"] != sha256(receipt_path)
        or accepted["intent_sha256"] != sha256(intent_path)
        or accepted["owner"] != owner
        or type(accepted["commands"]) is not int
        or accepted["commands"] != 27
        or type(accepted["checks"]) is not int
        or accepted["checks"] != 93
        or accepted["numerical_status"] != NUMERICAL_STATUS
        or state["status"] != "passed"
        or any(
            type(state[key]) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(key in state or key in accepted or key in receipt for key in ("error", "errors", "cleanup_error"))
        or state["binding_sha256"] != sha256(binding)
        or state["validation_sha256"] != sha256(receipt_path)
        or state.get("keeper_release_complete") is not True
        or state["command"] != validator_command(root, previous)
        or accepted["keeper"] != state["restored_keeper"]
        or intent["status"] != "acknowledged"
        or intent["pid"] != owner["controller_pid"]
        or intent["start_ticks"] != owner["controller_start_ticks"]
        or intent["binding_sha256"] != sha256(binding)
        or intent["command"] != controller_command(root)
        or not same_generation(
            intent["generation"],
            {
                "pid": intent["pid"],
                "start_ticks": intent["start_ticks"],
                "group": intent["pid"],
                "session": intent["pid"],
                "command": intent["command"],
            },
        )
        or receipt["status"] != NUMERICAL_STATUS
        or receipt["owner"] != owner
        or receipt["execution_binding_sha256"] != sha256(binding)
        or receipt["candidate_preparation_sha256"] != original["preparation_sha256"]
        or original["commands"] != validation_commands(previous, root, work)
        or len(receipt["gpu_commands"]) != 27
        or len(receipt["trials"]) != 9
        or sum(len(trial["checks"]) for trial in receipt["trials"]) != 93
        or any(value is not True for trial in receipt["trials"] for value in trial["checks"].values())
    ):
        raise RuntimeError("Attribution production lacks exact complete original numerical acceptance")
    require_helpers(helpers, original["helpers_sha256"])
    required = {
        binding,
        state_path,
        receipt_path,
        intent_path,
        Path(original["archive"]),
        Path(original["cpu_receipt"]),
        Path(original["controller_cpu_receipt"]),
        previous / VALIDATION_PREPARATION,
        root / "data/experiment_receipt.json",
        *(helpers / name for name in original["helpers_sha256"]),
    }
    required.update(Path(execution["log"]) for execution in receipt["gpu_commands"])
    evidence = accepted["evidence_sha256"]
    require_evidence(root, evidence, required)
    specs = trial_specs()
    for trial, spec in zip(receipt["trials"], specs, strict=True):
        checks = BASE_CHECKS | (FUSED_CHECKS if spec["family"] != "muont" else set())
        if spec["family"] == "null_leftdiag_fused":
            checks = checks | {"null_identity_outer"}
        if (
            any(trial[key] != value for key, value in spec.items())
            or trial["config_sha256"] != sha256(previous / spec["config_name"])
            or trial["config"] != str(previous / spec["config_name"])
            or set(trial["checks"]) != checks
            or trial["commands"] != original["commands"][specs.index(spec) * 3 : (specs.index(spec) + 1) * 3]
        ):
            raise RuntimeError("Attribution accepted trial identity or complete named gates differ")
    for index, (execution, command) in enumerate(zip(receipt["gpu_commands"], original["commands"], strict=True)):
        if (
            execution["command"] != command
            or execution["trial"] != specs[index // 3]["name"]
            or execution["mode"] != MODES[index % 3]
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or execution["log_sha256"] != sha256(Path(execution["log"]))
        ):
            raise RuntimeError("Attribution accepted command outcome or original log differs")
    return original


def require_allocation(root: Path, release: dict) -> None:
    wave = release["wave"]
    spec = wave_spec(wave)
    for family, node in zip(FAMILIES, spec["nodes"], strict=True):
        allocation = release["allocation_by_node"][node]
        predecessor = spec["predecessor_wave"] or ("attribution_validation" if node.endswith(".9") else "wave12")
        keeper = allocation["keeper"]
        generation = {int(pid): process for pid, process in keeper["generation"].items()}
        if (
            allocation["status"] != "fused_attribution_exact_predecessor_keeper_verified"
            or allocation["wave"] != wave
            or allocation["family"] != family
            or allocation["node"] != node
            or allocation["predecessor_wave"] != predecessor
            or len(generation) != WORLD + 1
            or keeper["pid"] not in generation
            or any(
                process["pid"] != pid
                or process["group"] != pid
                or process["session"] != pid
                or process.get("status", "S") in ("Z", "X", "x")
                for pid, process in generation.items()
            )
        ):
            raise RuntimeError("Attribution accepted keeper or immediate physical-node predecessor differs")
        if predecessor == "attribution_validation":
            required = {locations(root)[0], locations(root)[2], locations(root)[4]}
        else:
            binding = root / f"run-preparation/{predecessor}-launch-binding.json"
            previous = read(binding)
            families = previous["arms"] if predecessor == "wave12" else previous["families"]
            previous_family = families[previous["nodes"].index(node)]
            if allocation["predecessor_family"] != previous_family:
                raise RuntimeError("Attribution allocation predecessor family differs on the rotated node")
            state_path = root / f"run-preparation/{predecessor}-workers/{previous_family}.json"
            state = read(state_path)
            expected = {int(pid): process for pid, process in state["restored_keeper"]["generation"].items()}
            if (
                state["status"] != "complete"
                or any(
                    type(state[key]) is not int or state[key] != 0
                    for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
                )
                or state["node"] != node
                or state["release_binding_sha256"] != sha256(binding)
                or state["restored_keeper"]["pid"] != keeper["pid"]
                or set(expected) != set(generation)
                or any(not same_generation(process, generation[pid]) for pid, process in expected.items())
                or any(key in state for key in ("error", "errors", "cleanup_error"))
            ):
                raise RuntimeError("Attribution preceding worker did not restore the exact accepted keeper")
            required = {binding, state_path, state_path.with_suffix(".start.json"), Path(state["log"])}
        require_evidence(root, allocation["evidence_sha256"], required)


def require_release(release_path: Path) -> dict:
    """Authenticate separate production inventories, CPU archives and original acceptance."""
    prepared_path = TASK / PREPARATION
    if not prepared_path.is_file() or not release_path.is_file():
        raise RuntimeError("Attribution full-budget training requires a separate accepted production release")
    prepared = read(prepared_path)
    root = Path(prepared["root"])
    release = read(release_path)
    wave = release["wave"]
    binding, helpers, _, _ = paths(root, wave)
    spec = wave_spec(wave)
    files = candidate_files(TASK)
    if (
        root != root.resolve()
        or release_path != binding
        or release["status"] != FROZEN_STATUS
        or release["root"] != str(root)
        or release["candidate"] != str(TASK)
        or prepared["status"] != PREPARED_STATUS
        or prepared["candidate"] != str(TASK)
        or any(key in prepared or key in release for key in ("error", "errors"))
        or release["files_sha256"] != files
        or prepared["files_sha256"] != files
        or prepared["changed_files"] != sorted(CHANGES)
        or release["preparation_sha256"] != sha256(prepared_path)
        or release["helpers"] != str(helpers)
        or release["families"] != spec["families"]
        or release["nodes"] != spec["nodes"]
        or release["environment"] != ENVIRONMENT
        or release["commands"] != commands(root, TASK, wave)
        or type(release["monitor_interval_seconds"]) is not int
        or release["monitor_interval_seconds"] != MONITOR_INTERVAL
        or type(release["training_timeout_seconds"]) is not int
        or release["training_timeout_seconds"] != TRAINING_TIMEOUT
        or set(release["allocation_by_node"]) != set(spec["nodes"])
        or release["archive_sha256"] != sha256(Path(release["archive"]))
        or prepared["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
    ):
        raise RuntimeError("Attribution production candidate, release command or allocation scope differs")
    require_helpers(helpers, release["helpers_sha256"])
    if any(release["helpers_sha256"].get(name) != digest for name, digest in prepared["helpers_sha256"].items()):
        raise RuntimeError("Attribution production does not preserve the CPU-tested preparation closure")
    if any(
        files[f"experiment/{name}"] != release["helpers_sha256"].get(name)
        for name in ("fused_attribution_production_owner.py", "fused_attribution_production_contract.py")
    ):
        raise RuntimeError("Attribution copied production owner or contract differs from the tested helper")
    previous = Path(require_validation(root, prepared, release)["candidate"])
    before = candidate_files(previous)
    changed = {name for name in before.keys() | files.keys() if before.get(name) != files.get(name)}
    old_train = (previous / "experiment/train.py").read_text()
    replacements = (
        (
            "from .fused_attribution_owner import require_fused_attribution_owner",
            "from .fused_attribution_production_owner import require_fused_attribution_production_owner",
        ),
        ("    require_fused_attribution_owner(args)\n", "    require_fused_attribution_production_owner(args)\n"),
    )
    for old, new in replacements:
        if old_train.count(old) != 1:
            raise RuntimeError("Attribution original early ownership dispatch differs")
        old_train = old_train.replace(old, new, 1)
    metadata = prepared["parent_metadata_sha256"]
    if (
        changed != CHANGES
        or (TASK / "experiment/train.py").read_bytes() != old_train.encode()
        or metadata != {path.name: sha256(path) for path in previous.glob("*.json")}
        or any(sha256(TASK / name) != digest for name, digest in metadata.items())
    ):
        raise RuntimeError("Attribution production changed original numerical, configuration or metadata bytes")
    for item in WAVES:
        for family in FAMILIES:
            require_config(read(TASK / config_name(family, wave_spec(item)["lr_label"])), item, family)
    require_allocation(root, release)
    cpu_path = Path(release["cpu_receipt"])
    controller_path = Path(release["controller_cpu_receipt"])
    if release["cpu_receipt_sha256"] != sha256(cpu_path) or release["controller_cpu_receipt_sha256"] != sha256(
        controller_path
    ):
        raise RuntimeError("Attribution production CPU receipt identities differ")
    cpu = require_cpu_evidence(root, cpu_path, CPU_STATUS, TASK)
    controller = require_cpu_evidence(root, controller_path, CONTROLLER_CPU_STATUS, TASK)
    if (
        cpu["files_sha256"] != files
        or cpu["preparation_sha256"] != release["preparation_sha256"]
        or cpu["validation_binding_sha256"] != release["validation_binding_sha256"]
        or cpu["helpers_sha256"] != prepared["helpers_sha256"]
        or controller["candidate_cpu_receipt"] != str(cpu_path)
        or controller["candidate_cpu_receipt_sha256"] != sha256(cpu_path)
        or controller["helpers_sha256"] != release["helpers_sha256"]
        or controller["command"] != controller_cpu_command()
        or controller["cwd"] != str(Path(controller["helpers"]).parent)
    ):
        raise RuntimeError("Attribution production CPU acceptance differs from its frozen closure")
    require_helpers(Path(cpu["helpers"]), cpu["helpers_sha256"])
    require_helpers(Path(controller["helpers"]), controller["helpers_sha256"])
    require_archive(
        Path(cpu["archive"]),
        files
        | metadata
        | {PREPARATION: release["preparation_sha256"]}
        | {f"tracking/{name}": digest for name, digest in cpu["helpers_sha256"].items()},
    )
    fixtures = controller["fixture_files_sha256"]
    if any(files.get(name) != digest for name, digest in fixtures.items()):
        raise RuntimeError("Attribution controller CPU fixtures differ from the production candidate")
    require_controller_archive(
        Path(controller["archive"]),
        {f"tracking/{name}": digest for name, digest in controller["helpers_sha256"].items()} | fixtures,
    )
    require_archive(
        Path(release["archive"]),
        files
        | metadata
        | {
            PREPARATION: release["preparation_sha256"],
            "candidate-cpu-receipt.json": release["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": release["controller_cpu_receipt_sha256"],
            "validation-acceptance.json": release["validation_acceptance_sha256"],
        }
        | {f"tracking/{name}": digest for name, digest in release["helpers_sha256"].items()},
    )
    return release


def require_process(pid: int, worker_marker: str, child_marker: str | None = None) -> dict:
    process = observed(pid)
    if (
        not worker_marker
        or process["cwd"] != str(TASK)
        or process["environment"].get(WORKER_MARKER) != worker_marker
        or (child_marker is not None and process["environment"].get(CHILD_MARKER) != child_marker)
        or any(process["environment"].get(name) != value for name, value in ENVIRONMENT.items())
    ):
        raise RuntimeError("Attribution process cwd, markers or explicit eight-GPU environment differs")
    return process


def require_fused_attribution_production_owner(args: object) -> None:
    """Fail before config/output/CUDA unless this is the acknowledged released training rank."""
    if args.launch_release is None:
        raise RuntimeError("Attribution full-budget training requires a separate accepted production release")
    release = require_release(args.launch_release)
    root, wave = Path(release["root"]), release["wave"]
    config = read(args.config)
    family = config["experiment"]["arm"]
    node = node_for(wave, family)
    command = release["commands"][node]
    rank_command = [PYTHON, *command[5:]]
    require_config(config, wave, family)
    if (
        args.config != TASK / config_name(family, wave_spec(wave)["lr_label"])
        or args.data != root / "data"
        or args.data_receipt != root / "data/experiment_receipt.json"
        or args.out != output_path(root, wave, family)
        or args.max_steps is not None
        or args.resume is not None
        or args.execution_boundary is not None
        or args.verify_diagonal_capture
        or args.verify_outer_diagonal_updates
        or [PYTHON, "-m", "experiment.train", *sys.argv[1:]] != rank_command
        or any(os.environ.get(key) != value for key, value in ENVIRONMENT.items())
        or os.environ.get("WORLD_SIZE") != str(WORLD)
        or os.environ.get("LOCAL_WORLD_SIZE") != str(WORLD)
        or os.environ.get("RANK") not in tuple(str(rank) for rank in range(WORLD))
        or os.environ.get("LOCAL_RANK") != os.environ.get("RANK")
        or Path.cwd().resolve() != TASK
        or node not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Attribution full-budget invocation differs from the exact released trial")
    state_path = worker_path(root, wave, family)
    intent_path = state_path.with_suffix(".start.json")
    launch_path = paths(root, wave)[2]
    digest = sha256(args.launch_release)
    expected = worker_command(root, wave, family)
    deadline, identity = time.monotonic() + ACK_TIMEOUT, None
    while True:
        state, intent, launch = (read(path) for path in (state_path, intent_path, launch_path))
        current = (state["worker_pid"], state["worker_start_ticks"], state["worker_marker"])
        if identity is None:
            identity = current
        worker = require_process(state["worker_pid"], state["worker_marker"])
        if (
            current != identity
            or intent["status"] != "acknowledged"
            or any(key in item for item in (state, intent, launch) for key in ("error", "errors", "cleanup_error"))
            or intent["pid"] != worker["pid"]
            or intent["start_ticks"] != worker["start_ticks"]
            or not same_generation(worker, intent["generation"])
            or worker["parent"] not in (intent["generation"]["parent"], 1)
            or worker["start_ticks"] != state["worker_start_ticks"]
            or worker["command"] != expected
            or worker["group"] != worker["pid"]
            or worker["session"] != worker["pid"]
            or state["worker_command"] != expected
            or intent["command"] != expected
            or state["wave"] != wave
            or intent["wave"] != wave
            or state["family"] != family
            or intent["family"] != family
            or state["node"] != node
            or intent["node"] != node
            or state["release_binding_sha256"] != digest
            or intent["release_binding_sha256"] != digest
            or intent["worker_marker"] != state["worker_marker"]
            or set(launch["worker_markers"]) != set(FAMILIES)
            or len(set(launch["worker_markers"].values())) != len(FAMILIES)
            or launch["worker_markers"][family] != state["worker_marker"]
            or launch["binding_sha256"] != digest
            or launch["status"] not in ("launching", "launched")
            or os.environ.get(WORKER_MARKER) != state["worker_marker"]
        ):
            raise RuntimeError("Attribution one-shot worker acknowledgment or original generation differs")
        if state["status"] == "running" and "child_start_ticks" in state:
            break
        if state["status"] not in ("preflight", "starting", "running") or time.monotonic() >= deadline:
            raise RuntimeError("Attribution training has no acknowledged launcher")
        time.sleep(0.05)
    child = require_process(state["child_pid"], state["worker_marker"], state["child_marker"])
    rank = require_process(os.getpid(), state["worker_marker"], state["child_marker"])
    keeper = release["allocation_by_node"][node]["keeper"]
    generation = {int(pid): process for pid, process in keeper["generation"].items()}
    compact = state["keeper"]
    if (
        state.get("keeper_release_complete") is not True
        or compact["pid"] != keeper["pid"]
        or compact["start_ticks"] != generation[keeper["pid"]]["start_ticks"]
        or sorted(compact["generation"])
        != sorted([pid, process["start_ticks"], process["group"]] for pid, process in generation.items())
        or not state["child_marker"]
        or state["command"] != command
        or child["command"] != command
        or child["start_ticks"] != state["child_start_ticks"]
        or child["parent"] != worker["pid"]
        or child["group"] != child["pid"]
        or child["session"] != child["pid"]
        or rank["command"] not in (rank_command, [PYTHON, "-u", *command[5:]])
        or rank["parent"] != child["pid"]
        or rank["group"] != rank["pid"]
        or rank["session"] != rank["pid"]
        or os.environ.get(CHILD_MARKER) != state["child_marker"]
        or any(
            rank["environment"].get(key) != os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        )
    ):
        raise RuntimeError("Attribution training rank or original torchrun child generation differs")
    stable = (
        "status",
        "wave",
        "family",
        "node",
        "worker_pid",
        "worker_start_ticks",
        "worker_marker",
        "worker_command",
        "child_pid",
        "child_start_ticks",
        "child_marker",
        "command",
        "release_binding_sha256",
        "keeper_release_complete",
        "keeper",
    )
    after, current_launch = read(state_path), read(launch_path)
    if (
        observed(worker["pid"]) != worker
        or observed(child["pid"]) != child
        or observed(rank["pid"]) != rank
        or sha256(args.launch_release) != digest
        or read(intent_path) != intent
        or any(after[key] != state[key] for key in stable)
        or current_launch["worker_markers"] != launch["worker_markers"]
        or current_launch["binding_sha256"] != digest
        or current_launch["status"] not in ("launching", "launched")
        or any(key in after or key in current_launch for key in ("error", "errors", "cleanup_error"))
    ):
        raise RuntimeError("Attribution production ownership changed during inspection")
