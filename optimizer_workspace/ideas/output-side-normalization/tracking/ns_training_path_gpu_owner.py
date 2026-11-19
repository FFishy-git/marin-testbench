# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the distinct training-path validator and its bounded eight-rank commands."""

import csv
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import ns_training_path_contract as contract
import ns_training_path_execution_source as source
import ns_training_path_predecessor as predecessor
import ns_training_path_validation as validation
import owned_output_handoff as handoff
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import (
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from saved_ns_gpu_owner import require_held_lock
from selected_lr_failure_handoff import MANAGER_LOCK, released_keeper, same_keeper
from selected_lr_release_owner import candidate_files, read, sha256
from selected_lr_release_owner import observed as process_details

NODE = "28.216.17.9"
TIMEOUT = 54 * validation.COMMAND_TIMEOUT + 14400
CONTROLLER_ENVIRONMENT = contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
PREDECESSOR_RECEIPT = "run-preparation/ns-training-path-predecessor-cpu/20261119T0649/receipt.json"
PREDECESSOR_RECEIPT_SHA256 = "398dcf9cb1e63b44092c3aa5a6b49075294fff5c882e16789240d458d3b834a8"
PREDECESSOR_VERIFIER_SHA256 = "bda4a06eb58540afe60dfd682f4b3083525fb7c75582d13324f6ff53413ec050"
PREDECESSOR_SEALER_SHA256 = "6bb32b63bf3ac49429c18a99b1affd40869b5b176443f4074db6b56307be9055"
PREDECESSOR_SEAL_SHA256 = "107bbcbebf3d9b97b1d1e874623a6e106ff1c0c50e31a4c078fa8aa4d210750f"
PREDECESSOR_KEEPER = 1947488
ARCHIVE = "run-preparation/source-ns-training-path-20261119.tar"
PARENT_PROOF_MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_PARENT_PROOF"


def controller_command(root: Path) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(contract.locations(root)[1] / "ns_training_path_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "execute",
    ]


def validator_command(root: Path, candidate: Path) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(contract.locations(root)[1] / "ns_training_path_validation.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--phase",
        "execute",
    ]


def require_hidden_cuda() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("NS training-path controller acceptance requires hidden CUDA")


def predecessor_binding(root: Path) -> dict:
    """Authenticate the accepted sealer receipt and both verifier bytes together."""
    path = root / PREDECESSOR_RECEIPT
    identity = file_identity(path)
    raw = path.read_bytes()
    cpu = json.loads(raw)
    helpers = path.parent / "tracking"
    if (
        hashlib.sha256(raw).hexdigest() != PREDECESSOR_RECEIPT_SHA256
        or cpu["root"] != str(root)
        or cpu["status"] != "ns_training_path_predecessor_cpu_archive_verified_gpu_unissued"
        or cpu["cuda_visible_devices"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["helpers"] != str(helpers)
        or cpu["helpers_sha256"]["ns_training_path_predecessor.py"] != PREDECESSOR_VERIFIER_SHA256
        or cpu["helpers_sha256"]["prepare_ns_training_path_predecessor.py"] != PREDECESSOR_SEALER_SHA256
        or sha256(helpers / "ns_training_path_predecessor.py") != PREDECESSOR_VERIFIER_SHA256
        or sha256(helpers / "prepare_ns_training_path_predecessor.py") != PREDECESSOR_SEALER_SHA256
        or sha256(Path(predecessor.__file__)) != PREDECESSOR_VERIFIER_SHA256
        or file_identity(path) != identity
    ):
        raise ValueError("NS training-path accepted predecessor receipt/verifier identity differs")
    return {
        "cpu_receipt": str(path),
        "cpu_receipt_sha256": PREDECESSOR_RECEIPT_SHA256,
        "verifier_sha256": PREDECESSOR_VERIFIER_SHA256,
        "sealer_sha256": PREDECESSOR_SEALER_SHA256,
        "seal": str(root / predecessor.SEAL),
        "seal_sha256": PREDECESSOR_SEAL_SHA256,
    }


def require_sealed_predecessor(root: Path) -> dict:
    """Perform the hidden-CUDA public predecessor check without reconstructing tensors."""
    require_hidden_cuda()
    predecessor_binding(root)
    saved = predecessor.require_seal(root, PREDECESSOR_SEAL_SHA256)
    if saved["restored_keeper"]["pid"] != PREDECESSOR_KEEPER:
        raise ValueError("NS training-path predecessor is not the accepted exact keeper")
    return saved


def require_predecessor_keeper(root: Path, expected: dict) -> dict:
    if require_sealed_predecessor(root) != expected:
        raise ValueError("NS training-path original sealed predecessor changed before transfer")
    keeper = handoff.keeper_observation()
    require_keeper_shape(keeper)
    if not same_keeper(keeper, expected["restored_keeper"]):
        raise RuntimeError("NS training-path exact accepted predecessor keeper is not healthy")
    return keeper


def authenticated_inventory(paths: dict[str, str]) -> dict:
    """Capture identities only after the operation authenticates every supplied digest."""
    result = {}
    for name, digest in paths.items():
        path = Path(name).resolve(strict=True)
        identity = file_identity(path)
        if sha256(path) != digest or file_identity(path) != identity:
            raise ValueError("NS training-path authenticated input changed during inventory capture")
        item = {"sha256": digest, "identity": identity}
        if str(path) in result and result[str(path)] != item:
            raise ValueError("NS training-path canonical authenticated input digests conflict")
        result[str(path)] = item
    return result


def inventory_record(inventory: dict, path: Path) -> dict:
    return inventory.get(str(path.resolve(strict=True)), {})


def require_operation_inventory(inventory: dict) -> None:
    """Check every operation-authenticated inode on every call; no persistent cache."""
    if not inventory or any(file_identity(Path(name)) != item["identity"] for name, item in inventory.items()):
        raise ValueError("NS training-path authenticated operation inventory changed")


def candidate_file_names(candidate: Path) -> set[str]:
    """Enumerate the complete source/config namespace without rehashing its contents."""
    pinned, alias = candidate / "artifacts/pinned-sources-materialized", candidate / "artifacts/external-sources"
    if (
        not alias.is_symlink()
        or alias.readlink() != Path("pinned-sources-materialized")
        or alias.resolve() != pinned.resolve()
    ):
        raise ValueError("NS training-path source inventory lacks its exact pinned-source alias")
    result = set()
    for directory in (candidate / "experiment", pinned):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("NS training-path source/config directory is absent or aliased")
        for path in directory.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError("NS training-path source/config inventory contains an alias")
            if path.is_file() and (directory == pinned or path.suffix in (".py", ".json")):
                result.add(str(path.relative_to(candidate)))
    return result


def require_owner_only_candidate(root: Path, candidate: Path) -> tuple[dict, dict]:
    """Require independent accepted-parent authentication and exactly one adapter change."""
    prepared = source.require_preparation(root, candidate)
    original = Path(prepared["parent_candidate"])
    original_files, files = candidate_files(original), candidate_files(candidate)
    if (
        original == candidate
        or original.is_relative_to(candidate)
        or candidate.is_relative_to(original)
        or prepared["changed_files"] != [source.OWNER]
        or files != prepared["files_sha256"]
        or {name for name in files.keys() | original_files.keys() if files.get(name) != original_files.get(name)}
        != {source.OWNER}
    ):
        raise ValueError("NS training-path source/config change escapes its owner-only accepted parent")
    return prepared, original_files


def parent_proof(raw: bytes, frozen: dict) -> str:
    """Bind full CPU acceptance and inventory to the immutable controller environment."""
    proof = {
        "binding_sha256": hashlib.sha256(raw).hexdigest(),
        "cpu_receipt_sha256": frozen["cpu_receipt_sha256"],
        "inventory_sha256": (
            hashlib.sha256(
                json.dumps(frozen["authenticated_files"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        ),
    }
    return hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_parent_proof(root: Path, raw: bytes, frozen: dict) -> None:
    """Visible callers need the live original parent that independently accepted this CPU receipt."""
    _, helpers, state_path, _, _ = contract.locations(root)
    intent, state = read(helpers.parent / "controller.json"), read(state_path)
    proof = parent_proof(raw, frozen)
    if (
        intent["status"] != "acknowledged"
        or intent["binding_sha256"] != hashlib.sha256(raw).hexdigest()
        or intent["parent_proof"] != proof
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["binding_sha256"] != intent["binding_sha256"]
        or intent["command"] != controller_command(root)
        or os.environ.get(PARENT_PROOF_MARKER) != proof
    ):
        raise RuntimeError("NS training-path visible caller lacks an independently accepted parent proof")
    process = process_context(
        intent["pid"],
        intent["command"],
        helpers,
        contract.CONTROLLER_MARKER,
        intent["marker"],
        CONTROLLER_ENVIRONMENT | {PARENT_PROOF_MARKER: proof},
    )
    require_acknowledged_controller(intent, process)


def require_binding(root: Path) -> dict:
    """Require full hidden acceptance or the separately archived visible identity checker."""
    path, helpers, _, work, _ = contract.locations(root)
    file_identity(path)
    raw = path.read_bytes()
    frozen = json.loads(raw)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        require_parent_proof(root, raw, frozen)
    candidate = Path(frozen["candidate"])
    cpu_path = Path(frozen["cpu_receipt"])
    require_operation_inventory(frozen["authenticated_files"])
    cpu = read(cpu_path)
    sealed_path = root / predecessor.SEAL
    sealed_raw = sealed_path.read_bytes()
    sealed = json.loads(sealed_raw)
    if (
        root != root.resolve(strict=True)
        or candidate != candidate.resolve(strict=True)
        or frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != cpu["candidate"]
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != cpu["helpers_sha256"]
        or frozen["files_sha256"] != cpu["files_sha256"]
        or cpu["status"] != source.CPU_STATUS
        or cpu["root"] != str(root)
        or cpu["cuda_visible_devices"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["parent_cpu_receipt_sha256"] != source.PARENT_SHA256
        or frozen["preparation_sha256"] != cpu["preparation_sha256"]
        or frozen["cpu_receipt_sha256"] != sha256(cpu_path)
        or frozen["predecessor_acceptance"] != predecessor_binding(root)
        or hashlib.sha256(sealed_raw).hexdigest() != PREDECESSOR_SEAL_SHA256
        or frozen["allocation_predecessor"] != sealed
        or sealed["restored_keeper"]["pid"] != PREDECESSOR_KEEPER
        or frozen["devices"] != sealed["restored_keeper"]["devices"]
        or frozen["node"] != NODE
        or frozen["environment"] != contract.ENVIRONMENT
        or frozen["controller_environment"] != CONTROLLER_ENVIRONMENT
        or frozen["controller_command"] != controller_command(root)
        or frozen["validator_command"] != validator_command(root, candidate)
        or frozen["validation_timeout"] != TIMEOUT
        or frozen["archive"] != str(root / ARCHIVE)
        or frozen["scope"] != source.SCOPE
        or frozen["predecessor_acceptance"]
        != {
            "cpu_receipt": cpu["predecessor_binding"]["cpu_receipt"],
            "cpu_receipt_sha256": cpu["predecessor_binding"]["cpu_receipt_sha256"],
            "verifier_sha256": cpu["predecessor_binding"]["verifier_sha256"],
            "sealer_sha256": cpu["predecessor_binding"]["checker_sha256"],
            "seal": cpu["predecessor_binding"]["seal"],
            "seal_sha256": cpu["predecessor_binding"]["seal_sha256"],
        }
        or frozen["allowed_transform"] != source.OWNER
        or candidate_file_names(candidate) != set(frozen["files_sha256"])
        or candidate_file_names(Path(frozen["parent_candidate"])) != set(frozen["parent_files_sha256"])
        or {
            name
            for name in frozen["files_sha256"].keys() | frozen["parent_files_sha256"].keys()
            if frozen["files_sha256"].get(name) != frozen["parent_files_sha256"].get(name)
        }
        != {source.OWNER}
        or frozen["commands"] != frozen["plan"]["commands"]
        or len(frozen["commands"]) != 54
        or len(frozen["plan"]["trials"]) != 18
        or frozen["plan"]["root"] != str(root)
        or frozen["plan"]["candidate"] != str(candidate)
        or frozen["plan"]["tolerance"] != contract.TOLERANCE
        or frozen["plan"]["environment"] != contract.ENVIRONMENT
        or frozen["commands"] != contract.commands(candidate, root, work)
        or frozen["plan"] != validation.plan(root, candidate)
        or inventory_record(frozen["authenticated_files"], cpu_path).get("sha256") != frozen["cpu_receipt_sha256"]
        or inventory_record(frozen["authenticated_files"], root / ARCHIVE).get("sha256") != frozen["archive_sha256"]
        or inventory_record(frozen["authenticated_files"], candidate / source.PREPARATION).get("sha256")
        != frozen["preparation_sha256"]
        or inventory_record(frozen["authenticated_files"], root / "data/experiment_receipt.json").get("sha256")
        != frozen["data_receipt_sha256"]
        or any(
            inventory_record(frozen["authenticated_files"], candidate / name).get("sha256") != digest
            for name, digest in frozen["files_sha256"].items()
        )
        or any(
            inventory_record(frozen["authenticated_files"], Path(frozen["parent_candidate"]) / name).get("sha256")
            != digest
            for name, digest in frozen["parent_files_sha256"].items()
        )
        or any(
            inventory_record(frozen["authenticated_files"], helpers / name).get("sha256") != digest
            for name, digest in frozen["helpers_sha256"].items()
        )
        or any(
            inventory_record(frozen["authenticated_files"], cpu_path.parent / name).get("sha256") != digest
            for name, digest in cpu["inventory_sha256"].items()
        )
        or any(
            inventory_record(frozen["authenticated_files"], root / name).get("identity") != identity
            or inventory_record(frozen["authenticated_files"], root / name).get("sha256")
            != sealed["evidence_sha256"][name]
            for name, identity in sealed["file_identity"].items()
        )
    ):
        raise ValueError("NS training-path frozen source/protocol/sealer identity differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        prepared, original_files = require_owner_only_candidate(root, candidate)
        if (
            source.require_cpu(root, cpu_path) != cpu
            or frozen["files_sha256"] != prepared["files_sha256"]
            or frozen["parent_candidate"] != prepared["parent_candidate"]
            or frozen["parent_files_sha256"] != original_files
            or frozen["allocation_predecessor"] != require_sealed_predecessor(root)
            or frozen["commands"] != contract.commands(candidate, root, work)
            or frozen["plan"] != validation.plan(root, candidate)
        ):
            raise ValueError("NS training-path full hidden acceptance differs")
        require_helper_bytes(helpers, frozen["helpers_sha256"])
        if sha256(root / ARCHIVE) != frozen["archive_sha256"]:
            raise ValueError("NS training-path issued archive changed")
        require_archive(root / ARCHIVE, cpu["inventory_sha256"], alias=True)
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"NS training-path imported mutable execution helper: {name}")
    if Path(__file__).resolve() != helpers / Path(__file__).name or path.read_bytes() != raw:
        raise ValueError("NS training-path issued owner or binding changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        require_parent_proof(root, raw, frozen)
    require_operation_inventory(frozen["authenticated_files"])
    return frozen


def process(pid: int, command: list[str], cwd: Path, marker_name: str, marker: str) -> dict:
    return process_context(pid, command, cwd, marker_name, marker, contract.ENVIRONMENT)


class CUDAUUID(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type]]] = [("bytes", ctypes.c_ubyte * 16)]


def cuda_ordinal_uuids() -> list[str]:
    """Read this process's driver ordinals without creating a context or launching kernels."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != contract.ENVIRONMENT["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("NS training-path ordinal inspection requires the accepted visible mask")
    driver = ctypes.CDLL("libcuda.so.1")
    signatures = (
        ("cuInit", [ctypes.c_uint]),
        ("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)]),
        ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
        ("cuDeviceGetUuid_v2", [ctypes.POINTER(CUDAUUID), ctypes.c_int]),
    )
    for name, arguments in signatures:
        function = getattr(driver, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
    if driver.cuInit(0) != 0:
        raise RuntimeError("NS training-path identity-only CUDA driver initialization failed")
    count = ctypes.c_int()
    if driver.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value != contract.WORLD:
        raise RuntimeError("NS training-path driver ordinal coverage differs from eight physical devices")
    result = []
    for ordinal in range(count.value):
        device, identifier = ctypes.c_int(), CUDAUUID()
        if (
            driver.cuDeviceGet(ctypes.byref(device), ordinal) != 0
            or driver.cuDeviceGetUuid_v2(ctypes.byref(identifier), device.value) != 0
        ):
            raise RuntimeError("NS training-path driver UUID inspection failed before training")
        result.append(f"GPU-{UUID(bytes=bytes(identifier.bytes))}")
    return result


def require_controller(root: Path, candidate: Path) -> tuple[dict, dict]:
    frozen = require_binding(root)
    binding, helpers, state_path, _, _ = contract.locations(root)
    intent, deadline = read(helpers.parent / "controller.json"), time.monotonic() + 30
    while True:
        state = read(state_path)
        if state.get("spawn_status") == "acknowledged" and intent["status"] == "acknowledged":
            break
        if state.get("status") not in ("preflight", "validating") or time.monotonic() >= deadline:
            raise RuntimeError("NS training-path lacks original controller/validator acknowledgment")
        time.sleep(0.01)
        intent = read(helpers.parent / "controller.json")
    if (
        frozen["candidate"] != str(candidate)
        or state["status"] != "validating"
        or state["binding_sha256"] != sha256(binding)
        or intent["binding_sha256"] != sha256(binding)
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["allocation_predecessor"] != frozen["allocation_predecessor"]
        or state["keeper_release_complete"] is not True
        or not released_keeper(state["keeper"], frozen["allocation_predecessor"]["restored_keeper"])
        or intent["command"] != controller_command(root)
        or state["command"] != validator_command(root, candidate)
        or state["child_generation"]["pid"] != state["child_pid"]
        or state["child_generation"]["start_ticks"] != state["child_start_ticks"]
        or state["child_generation"]["parent"] != state["controller_pid"]
        or state["child_generation"]["command"] != state["command"]
        or Path.cwd().resolve() != candidate
        or any(os.environ.get(name) != value for name, value in contract.ENVIRONMENT.items())
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True, timeout=20).split()
    ):
        raise RuntimeError("NS training-path original owner, transfer or environment differs")
    controller = process_context(
        state["controller_pid"],
        intent["command"],
        helpers,
        contract.CONTROLLER_MARKER,
        intent["marker"],
        CONTROLLER_ENVIRONMENT,
    )
    require_acknowledged_controller(intent, controller)
    validator = process(state["child_pid"], state["command"], candidate, contract.CHILD_MARKER, state["child_marker"])
    if (
        validator["parent"] != controller["pid"]
        or validator["start_ticks"] != state["child_start_ticks"]
        or not handoff.same_generation(validator, state["child_generation"])
    ):
        raise RuntimeError("NS training-path validator original direct-child ancestry differs")
    for name, lock in (("node", root / "run-preparation" / f".wave7-node-{NODE}.lock"), ("manager", MANAGER_LOCK)):
        require_held_lock(controller["pid"], state["lock_descriptors"][name], lock)
    devices, occupied = (
        list(csv.reader(subprocess.check_output(argv, text=True, timeout=20).splitlines(), skipinitialspace=True))
        for argv in (
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        )
    )
    if devices != frozen["devices"] or handoff.KEEPER_PID.exists():
        raise RuntimeError("NS training-path physical allocation or keeper release differs")
    if cuda_ordinal_uuids() != [row[1] for row in sorted(devices, key=lambda row: int(row[0]))]:
        raise RuntimeError("NS training-path CUDA ordinal-to-UUID mapping differs before training")
    marked = handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"])
    launchers = {
        entry["pid"]: entry
        for entry in marked.values()
        if entry["command"] == state.get("active_command") and entry["parent"] == validator["pid"]
    }
    ranks = {
        entry["pid"]: entry
        for entry in marked.values()
        if entry["parent"] in launchers and entry["command"] == [contract.PYTHON, "-u", *state["active_command"][5:]]
    }
    if len(launchers) > 1:
        raise RuntimeError("NS training-path occupancy has competing active launchers")
    for pid, uuid in occupied:
        if int(pid) not in ranks:
            raise RuntimeError("NS training-path CUDA occupancy includes an unowned process")
        environment = process_details(int(pid))["environment"]
        local = environment.get("LOCAL_RANK")
        if (
            local not in tuple(str(index) for index in range(contract.WORLD))
            or environment.get("RANK") != local
            or uuid != devices[int(local)][1]
        ):
            raise RuntimeError("NS training-path rank occupies a different physical GPU")
    current = process_context(
        controller["pid"],
        intent["command"],
        helpers,
        contract.CONTROLLER_MARKER,
        intent["marker"],
        CONTROLLER_ENVIRONMENT,
    )
    if not handoff.same_generation(controller, current) or not handoff.same_generation(
        validator, process(validator["pid"], state["command"], candidate, contract.CHILD_MARKER, state["child_marker"])
    ):
        raise RuntimeError("NS training-path owner changed during allocation inspection")
    return state, frozen


def require_gpu_owner(root: Path, candidate: Path) -> dict:
    state, _ = require_controller(root, candidate)
    if os.getpid() != state["child_pid"] or os.getppid() != state["controller_pid"]:
        raise RuntimeError("NS training-path validator is not the original direct child")
    return {
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "validator_pid": state["child_pid"],
        "validator_start_ticks": state["child_start_ticks"],
    }


def require_rank(root: Path, candidate: Path, args: object) -> None:
    state, frozen = require_controller(root, candidate)
    invocation = [contract.PYTHON, "-m", "experiment.train", *sys.argv[1:]]
    matches = [argv for argv in frozen["commands"] if invocation == [argv[0], *argv[5:]]]
    if len(matches) != 1:
        raise RuntimeError("NS training-path rank is outside its exact bounded command protocol")
    command = matches[0]
    if (
        state.get("active_command") != command
        or args.launch_release != contract.locations(root)[0]
        or str(args.config) != command[command.index("--config") + 1]
        or os.environ.get("WORLD_SIZE") != str(contract.WORLD)
        or os.environ.get("LOCAL_WORLD_SIZE") != str(contract.WORLD)
        or os.environ.get("RANK") not in tuple(str(index) for index in range(contract.WORLD))
        or os.environ.get("LOCAL_RANK") != os.environ.get("RANK")
    ):
        raise RuntimeError("NS training-path active command or rank topology differs")
    marker = os.environ.get(contract.COMMAND_MARKER, "")
    if not marker:
        raise RuntimeError("NS training-path rank has no per-command ownership marker")
    rank = process(os.getpid(), [command[0], "-u", *command[5:]], candidate, contract.COMMAND_MARKER, marker)
    launcher = process(rank["parent"], command, candidate, contract.COMMAND_MARKER, marker)
    if (
        launcher["parent"] != state["child_pid"]
        or process_details(rank["pid"])["environment"].get(contract.CHILD_MARKER) != state["child_marker"]
        or process_details(launcher["pid"])["environment"].get(contract.CHILD_MARKER) != state["child_marker"]
    ):
        raise RuntimeError("NS training-path rank lacks original independently sessioned launcher ancestry")
