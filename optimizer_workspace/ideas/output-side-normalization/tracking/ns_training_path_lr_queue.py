# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wait for the original canonical terminal, then issue one accepted actual-LR diagnostic."""

import argparse
import ast
import fcntl
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

import ns_training_path_lr_contract as contract
import ns_training_path_lr_gpu_controller as controller
import ns_training_path_lr_gpu_owner as owner
import ns_training_path_lr_gpu_source as source
import ns_training_path_lr_predecessor as predecessor
import owned_output_handoff as handoff
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write
from wave7_release import local_node

PARENT = "run-preparation/ns-training-path-actual-lr-execution-owner-cpu/20261119T1051/receipt.json"
PARENT_SHA256 = "0f31cbbad4a378042841cdf918bf9d8c0320cffa12668b26718a6816549ad2ca"
WORK = "run-preparation/ns-training-path-actual-lr-queue-20261119"
MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_LR_QUEUE"
PHASE_MARKER = contract.CONTROLLER_MARKER
ENVIRONMENT = source.ENVIRONMENT
INTERVAL = 900
PHASE_TIMEOUT = 1800
TERMINAL_TIMEOUT = 14400
CPU_STATUS = "ns_training_path_actual_lr_queue_cpu_archive_verified_gpu_unissued"
NEW_HELPERS = (
    "ns_training_path_lr_queue.py",
    "ns_training_path_lr_queue_test_support.py",
    "test_ns_training_path_lr_queue.py",
)
TEST_COUNT = 33
SCOPE = (
    "One hidden-CUDA original-training-path successor queue. Preserve the accepted 177-helper "
    "actual-LR execution-owner archive and candidate unchanged. Wait for exact original "
    "controller/validator/waiter/terminal-child absence and the existing waiter's complete "
    "canonical terminal/restoration. Invoke no original terminal. Separately archive-bound "
    "predecessor seal/check-seal, caller-digest-bound owner freeze/check/start, then the issued "
    "canonical actual-LR terminal after exact owner/descendant absence and clean restoration. "
    "Retain all original and actual-LR rejections, failed phases and consumed claims; no retry, "
    "keeper operation, tensor reader, production release, quality/efficiency or SwanLab authority. " + contract.SCOPE
)


def parent(root: Path) -> dict:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("Training-path actual-LR queue accepted owner receipt changed")
    return source.require_cpu(root, path)


def helper_sources(inherited: dict) -> dict[str, Path]:
    sources = {name: Path(inherited["helpers"]) / name for name in inherited["helpers_sha256"]}
    if set(sources).intersection(NEW_HELPERS):
        raise ValueError("Training-path actual-LR queue would overwrite accepted helpers")
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or path != path.resolve(strict=True):
            raise ValueError("Training-path actual-LR queue source is aliased")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Training-path actual-LR queue requires a complete flat closure")
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Training-path actual-LR queue import escapes its accepted closure")
    return sources


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        contract.PYTHON,
        "--with",
        "pytest-timeout",
        "python",
        "-P",
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(out),
        "-p",
        "no:cacheprovider",
        "-p",
        "root_diagnostic_pytest_guard",
        "--timeout=60",
        "--session-timeout=600",
        "--import-mode=prepend",
        "-m",
        SAFE_MARKERS,
        "-q",
        str(out / "tracking" / NEW_HELPERS[-1]),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    """Authenticate the full queue archive, inherited fixtures and passing CPU suite."""
    identity, raw, receipt = file_identity(path), path.read_bytes(), read(path)
    inherited = parent(root)
    hashes = {name: sha256(item) for name, item in helper_sources(inherited).items()}
    inventory = inherited["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or receipt["candidate"] != inherited["candidate"]
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != inventory
        or receipt["command"] != cpu_command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != SCOPE
        or len(cases) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(NEW_HELPERS[-1]).stem}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != TEST_COUNT
        or sum(int(suite.attrib["tests"]) for suite in suites) != TEST_COUNT
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in inventory.items())
    ):
        raise ValueError("Training-path actual-LR queue CPU/source/archive acceptance differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), inventory, alias=True)
    if path.read_bytes() != raw or file_identity(path) != identity:
        raise ValueError("Training-path actual-LR queue receipt changed during verification")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    owner.require_hidden_cuda()
    if (
        out.exists()
        or out.is_symlink()
        or out != out.resolve()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise FileExistsError("Training-path actual-LR queue CPU output must be unused and external")
    inherited = parent(root)
    sources = helper_sources(inherited)
    hashes = {name: sha256(item) for name, item in sources.items()}
    inventory = inherited["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    out.mkdir(parents=True)
    for name, digest in inventory.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        original = sources[Path(name).name] if name.startswith("tracking/") else Path(inherited["helpers"]).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha256(target) != digest:
            raise ValueError("Training-path actual-LR queue inherited bytes changed during staging")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    log, error = out / "cpu.log", None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(out),
                cwd=out,
                env=os.environ
                | ENVIRONMENT
                | {
                    "UV_OFFLINE": "1",
                    "PYTHONPATH": f"{out / 'tracking'}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
                },
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Training-path actual-LR queue CPU tests timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(out / name, arcname=name, recursive=False)
        bundle.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": inherited["candidate"],
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "helpers": str(out / "tracking"),
        "helpers_sha256": hashes,
        "inventory_sha256": inventory,
        "command": cpu_command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": SCOPE,
        **{name: str(item) for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))},
        **{
            f"{name}_sha256": sha256(item)
            for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
            if item.is_file()
        },
    }
    if error:
        receipt["error"] = error
    write(out / "receipt.json", receipt)
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        receipt.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", receipt)
        raise RuntimeError(f"Training-path actual-LR queue CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / NEW_HELPERS[0]:
        raise RuntimeError("Training-path actual-LR queue must execute its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Training-path actual-LR queue imported mutable helper: {name}")
    return receipt


def readiness(root: Path) -> dict:
    """Use the accepted original handle observer; never invoke the original terminal."""
    observation = predecessor.readiness(root)
    if not observation["ready"]:
        return observation
    previous = predecessor.evidence(root)
    return observation | {
        "status": "original_training_path_terminal_ready",
        "original_terminal": previous["original_terminal"],
        "original_terminal_sha256": previous["original_terminal_sha256"],
        "original_decisions": previous["original_decisions"],
    }


def empty_launch(root: Path) -> None:
    if any(
        path.exists() or path.is_symlink()
        for path in (*contract.locations(root), root / owner.ARCHIVE, root / predecessor.SEAL)
    ):
        raise FileExistsError("Training-path actual-LR queue found existing one-shot execution evidence")


def predecessor_command(root: Path, phase: str, seal_sha256: str | None = None) -> tuple[list[str], Path]:
    path = root / owner.PREDECESSOR_RECEIPT
    if sha256(path) != owner.PREDECESSOR_RECEIPT_SHA256:
        raise ValueError("Training-path actual-LR queue accepted predecessor receipt changed")
    receipt = source.accepted_archive(
        root,
        (
            owner.PREDECESSOR_RECEIPT,
            owner.PREDECESSOR_RECEIPT_SHA256,
            "ns_training_path_actual_lr_predecessor_cpu_archive_verified_owner_unissued",
            12,
        ),
    )
    helpers = Path(receipt["helpers"])
    if (
        receipt["helpers_sha256"]["ns_training_path_lr_predecessor.py"] != owner.PREDECESSOR_VERIFIER_SHA256
        or receipt["helpers_sha256"]["prepare_ns_training_path_lr_predecessor.py"] != owner.PREDECESSOR_SEALER_SHA256
    ):
        raise ValueError("Training-path actual-LR queue predecessor archive origin differs")
    argv = [
        contract.PYTHON,
        "-P",
        str(helpers / "prepare_ns_training_path_lr_predecessor.py"),
        "--root",
        str(root),
        "--phase",
        phase,
        "--cpu-receipt",
        str(path),
    ]
    if seal_sha256 is not None:
        argv.extend(("--seal-sha256", seal_sha256))
    return argv, helpers


def run_phase(root: Path, phase: str, argv: list[str], cwd: Path, state: dict) -> dict:
    """Retain execution and descendant cleanup even when a one-shot phase fails."""
    work, marker, generations = root / WORK, uuid4().hex, {}
    log = work / f"{phase}.log"
    if log.exists() or any(entry["phase"] == phase for entry in state["phases"]):
        raise FileExistsError("Training-path actual-LR queue phase already has one-shot evidence")
    entry = {"phase": phase, "command": argv, "cwd": str(cwd), "marker": marker, "spawn_status": "starting"}
    state["active_phase"] = entry
    write(work / "state.json", state)
    environment = os.environ | ENVIRONMENT | {"PYTHONPATH": str(cwd), PHASE_MARKER: marker}
    environment.pop(MARKER, None)
    # The issued owner replaces this marker, preserving its independent lifetime.
    child = None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            own = acknowledged_process(child, argv, cwd, PHASE_MARKER, marker, ENVIRONMENT)
            entry.update(spawn_status="acknowledged", pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
            write(work / "state.json", state)
            entry["exit_code"] = child.wait(timeout=TERMINAL_TIMEOUT if phase == "terminal" else PHASE_TIMEOUT)
            if type(entry["exit_code"]) is not int or entry["exit_code"] != 0:
                raise RuntimeError(f"Training-path actual-LR queue {phase} failed; evidence remains consumed")
    finally:
        try:
            if child is not None:
                finish_marked_child(child, marker, generations, PHASE_MARKER)
                entry.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
        except BaseException as failure:
            entry["cleanup_error"] = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            entry.update(log=str(log), log_sha256=sha256(log) if log.is_file() else None, descendants=generations)
            write(work / "state.json", state)
    return entry


def retain_phase(root: Path, phase: str, argv: list[str], cwd: Path, state: dict) -> None:
    state["phases"].append(run_phase(root, phase, argv, cwd, state))
    write(root / WORK / "state.json", state)


def launch(root: Path, state: dict) -> dict:
    empty_launch(root)
    observation = readiness(root)
    if not observation["ready"] or state["phases"] or state.get("active_phase"):
        raise RuntimeError("Training-path actual-LR queue cannot repeat or bypass original terminal gates")
    state.update(status="sealing_original_training_path_terminal", original_observation=observation)
    write(root / WORK / "state.json", state)
    retain_phase(root, "seal", *predecessor_command(root, "seal"), state)
    seal_sha256 = sha256(root / predecessor.SEAL)
    sealed = predecessor.require_seal(root, seal_sha256)
    if any(
        sealed[name] != observation[name]
        for name in ("original_terminal", "original_terminal_sha256", "original_decisions")
    ):
        raise ValueError("Training-path actual-LR queue original terminal changed during seal")
    state.update(predecessor_seal_sha256=seal_sha256, original_decisions=sealed["original_decisions"])
    write(root / WORK / "state.json", state)
    retain_phase(root, "check-seal", *predecessor_command(root, "check-seal", seal_sha256), state)
    inherited, issued = (root / PARENT).parent / "tracking", contract.locations(root)[1]
    freeze = [
        contract.PYTHON,
        "-P",
        str(inherited / "ns_training_path_lr_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "freeze",
        "--cpu-receipt",
        str(root / PARENT),
        "--seal-sha256",
        seal_sha256,
    ]
    retain_phase(root, "freeze", freeze, inherited, state)
    for phase in ("check", "start"):
        argv = [
            contract.PYTHON,
            "-P",
            str(issued / "ns_training_path_lr_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            phase,
        ]
        retain_phase(root, phase, argv, issued, state)
    binding = contract.locations(root)[0]
    frozen, intent = read(binding), read(issued.parent / "controller.json")
    own = process_context(
        intent["pid"],
        owner.controller_command(root),
        issued,
        contract.CONTROLLER_MARKER,
        intent["marker"],
        owner.CONTROLLER_ENVIRONMENT | {owner.PARENT_PROOF_MARKER: intent["parent_proof"]},
    )
    require_acknowledged_controller(intent, own)
    if (
        intent["binding_sha256"] != sha256(binding)
        or frozen["cpu_receipt_sha256"] != PARENT_SHA256
        or frozen["predecessor_acceptance"]["seal_sha256"] != seal_sha256
        or frozen["allocation_predecessor"]["original_decisions"] != state["original_decisions"]
    ):
        raise RuntimeError("Training-path actual-LR queue issued a different owner or predecessor")
    state.update(
        status="actual_lr_training_path_controller_acknowledged",
        issued_binding_sha256=sha256(binding),
        issued_controller=intent,
    )
    write(root / WORK / "state.json", state)
    return state


def issued_readiness(root: Path, state: dict) -> dict:
    """Require exact issued owner absence and complete cleanup before consuming its terminal."""
    binding, helpers, state_path, _, result_path = contract.locations(root)
    frozen, intent = read(binding), read(helpers.parent / "controller.json")
    if (
        sha256(binding) != state["issued_binding_sha256"]
        or intent != state["issued_controller"]
        or intent["binding_sha256"] != sha256(binding)
        or frozen["cpu_receipt_sha256"] != PARENT_SHA256
        or frozen["controller_command"] != owner.controller_command(root)
        or frozen["environment"] != contract.ENVIRONMENT
        or frozen["predecessor_acceptance"]["seal_sha256"] != state["predecessor_seal_sha256"]
        or frozen["allocation_predecessor"]["original_decisions"] != state["original_decisions"]
    ):
        raise ValueError("Training-path actual-LR issued source, predecessor or controller changed")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    owners = {
        "controller": predecessor.followup.handle(
            intent["generation"],
            helpers,
            contract.CONTROLLER_MARKER,
            intent["marker"],
            owner.CONTROLLER_ENVIRONMENT | {owner.PARENT_PROOF_MARKER: intent["parent_proof"]},
        )
    }
    saved = read(state_path) if state_path.is_file() else {}
    if saved.get("spawn_status") == "acknowledged":
        owners["validator"] = predecessor.followup.handle(
            saved["child_generation"],
            Path(frozen["candidate"]),
            contract.CHILD_MARKER,
            saved["child_marker"],
            contract.ENVIRONMENT,
        )
    marked = (
        handoff.marker_processes(contract.CHILD_MARKER, saved.get("child_marker", ""))
        if saved.get("child_marker")
        else {}
    )
    if result_path.is_file():
        for entry in read(result_path).get("gpu_commands", []):
            marked.update(handoff.marker_processes(contract.COMMAND_MARKER, entry["marker"]))
            if handoff.observed(entry["generation"]["pid"]) is not None:
                owners[f"command_{entry['generation']['pid']}"] = predecessor.followup.handle(
                    entry["generation"],
                    Path(frozen["candidate"]),
                    contract.COMMAND_MARKER,
                    entry["marker"],
                    contract.ENVIRONMENT,
                )
    if any(item["status"] != "absent" for item in owners.values()) or marked:
        return {
            "status": "waiting_for_issued_actual_lr_training_path_owners",
            "ready": False,
            "owners": owners,
            "marked_pids": sorted(marked),
        }
    if (
        saved.get("status") != "validation_complete"
        or saved.get("spawn_status") != "acknowledged"
        or saved["binding_sha256"] != state["issued_binding_sha256"]
        or saved["controller_pid"] != intent["pid"]
        or saved["controller_start_ticks"] != intent["start_ticks"]
        or any(
            type(saved.get(key)) is not int or saved[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or saved.get("marked_descendants_absent") is not True
        or saved.get("keeper_release_complete") is not True
        or not saved.get("restored_keeper")
        or any(key in saved for key in ("error", "cleanup_error", "deferred_signals"))
    ):
        raise RuntimeError("Training-path actual-LR original owner exited without complete cleanup/restoration")
    terminal, attempt, outcome, _ = controller.terminal_paths(root, intent)
    if any(path.exists() or path.is_symlink() for path in (terminal, attempt, outcome)):
        raise FileExistsError("Training-path actual-LR terminal already has a consumed claim or outcome")
    return {
        "status": "actual_lr_training_path_terminal_ready",
        "ready": True,
        "owners": owners,
        "terminal": str(terminal),
    }


def terminal_phase(root: Path, state: dict) -> dict:
    observation = issued_readiness(root, state)
    if not observation["ready"]:
        raise RuntimeError("Training-path actual-LR original owner or descendant remains")
    helpers, output = contract.locations(root)[1], Path(observation["terminal"])
    argv = [
        contract.PYTHON,
        "-P",
        str(helpers / "ns_training_path_lr_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(output),
    ]
    retain_phase(root, "terminal", argv, helpers, state)
    result = read(output)
    _, attempt, outcome, _ = controller.terminal_paths(root, state["issued_controller"])
    finished = read(outcome)
    if (
        result["status"] != controller.TERMINAL_STATUS
        or result["binding_sha256"] != state["issued_binding_sha256"]
        or result["controller_pid_absent"] != state["issued_controller"]["pid"]
        or result["controller_start_ticks"] != state["issued_controller"]["start_ticks"]
        or result["verifier_sha256"]
        != read(contract.locations(root)[0])["helpers_sha256"]["ns_training_path_lr_gpu_controller.py"]
        or result["predecessor_acceptance"]["seal_sha256"] != state["predecessor_seal_sha256"]
        or result["production_authorized"] is not False
        or finished["status"] != "complete"
        or finished["output"] != str(output)
        or finished["terminal_sha256"] != sha256(output)
        or finished["reconstruction_intent_sha256"] != sha256(attempt)
        or result["reconstruction_intent_sha256"] != sha256(attempt)
        or any(sha256(root / name) != digest for name, digest in result["evidence_sha256"].items())
    ):
        raise ValueError("Training-path actual-LR canonical terminal outcome, origin or evidence differs")
    state.update(
        status="actual_lr_training_path_terminal_reconstructed",
        terminal=str(output),
        terminal_sha256=sha256(output),
        terminal_outcome_sha256=sha256(outcome),
        **{name: result[name] for name in ("accepted_cases", "rejected_cases", "all_cases_accepted")},
    )
    write(root / WORK / "state.json", state)
    return state


def command(root: Path, cpu_receipt: Path) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(cpu_receipt.parent / "tracking" / NEW_HELPERS[0]),
        "--root",
        str(root),
        "--cpu-receipt",
        str(cpu_receipt),
        "--phase",
        "execute",
    ]


def start(root: Path, cpu_receipt: Path) -> dict:
    receipt = accepted(root, cpu_receipt)
    work = root / WORK
    if work.exists() or work.is_symlink() or local_node() != owner.NODE:
        raise FileExistsError("Training-path actual-LR queue already installed or allocation differs")
    readiness(root)
    empty_launch(root)
    work.mkdir()
    intent = {
        "status": "starting",
        "command": command(root, cpu_receipt),
        "marker": uuid4().hex,
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "runner_sha256": receipt["helpers_sha256"][NEW_HELPERS[0]],
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "interval": INTERVAL,
        "scope": SCOPE,
    }
    write(work / "intent.json", intent)
    child = None
    try:
        with (work / "queue.log").open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=work,
                env=os.environ | ENVIRONMENT | {"PYTHONPATH": receipt["helpers"], MARKER: intent["marker"]},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        own = acknowledged_process(child, intent["command"], work, MARKER, intent["marker"], ENVIRONMENT)
        intent.update(status="acknowledged", pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
    except BaseException as failure:
        intent.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        if child is not None:
            finish_marked_child(child, intent["marker"], {}, MARKER)
        raise
    finally:
        write(work / "intent.json", intent)
    return intent


def execute(root: Path, cpu_receipt: Path) -> dict:
    accepted(root, cpu_receipt)
    work = root / WORK
    intent, deadline = read(work / "intent.json"), time.monotonic() + 30
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
        intent = read(work / "intent.json")
    own = process_context(os.getpid(), command(root, cpu_receipt), work, MARKER, intent["marker"], ENVIRONMENT)
    require_acknowledged_controller(intent, own)
    if (
        intent["cpu_receipt_sha256"] != sha256(cpu_receipt)
        or intent["runner_sha256"] != sha256(Path(__file__))
        or intent["command"] != command(root, cpu_receipt)
        or intent["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or intent["interval"] != INTERVAL
        or intent["scope"] != SCOPE
        or local_node() != owner.NODE
    ):
        raise RuntimeError("Training-path actual-LR queue original identity or allocation differs")
    state = {
        "status": "waiting_for_original_training_path_owners",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        **{
            name: intent[name]
            for name in ("cpu_receipt_sha256", "runner_sha256", "parent_cpu_receipt_sha256", "interval")
        },
        "phases": [],
        "scope": SCOPE,
    }
    with (work / "execution.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            index = 0
            while True:
                observation = readiness(root)
                write(work / f"original-observation-{index:06d}.json", observation)
                state.update(
                    status=observation["status"], original_observation=observation, original_observations=index + 1
                )
                write(work / "state.json", state)
                if observation["ready"]:
                    launch(root, state)
                    break
                index += 1
                time.sleep(INTERVAL)
            while True:
                observation = issued_readiness(root, state)
                state.update(status=observation["status"], issued_observation=observation)
                write(work / "state.json", state)
                if observation["ready"]:
                    terminal_phase(root, state)
                    break
                time.sleep(INTERVAL)
        except BaseException as failure:
            state.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
            raise
        finally:
            write(work / "state.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check", "start", "execute"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Training-path actual-LR queue requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Accepted queue phases require --cpu-receipt")
        cpu_receipt = args.cpu_receipt.resolve(strict=True)
        result = (
            accepted(root, cpu_receipt)
            if args.phase == "check"
            else (start if args.phase == "start" else execute)(root, cpu_receipt)
        )
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
