# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wait for original lower-LR terminal handoff, then issue one accepted actual-LR diagnostic."""

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

import ns_lr_full_model_contract as contract
import ns_lr_full_model_execution_source as source
import ns_lr_full_model_gpu_controller as controller
import ns_lr_full_model_gpu_owner as owner
import ns_lr_full_model_predecessor as predecessor
import owned_output_handoff as handoff
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from ns_budget_queue import handle
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write
from wave7_release import local_node

PARENT = "run-preparation/ns-actual-lr-owner-cpu/20261119T0505/receipt.json"
PARENT_SHA256 = "6e043746bb948fc10d9f4b791105ed62b0c70c013ca75026505fe459098631d4"
WORK = "run-preparation/ns-actual-lr-queue-20261119"
TERMINAL = "run-preparation/ns-actual-lr-terminal-independent-20261119.json"
MARKER = "OUTPUT_MARIN_NS_ACTUAL_LR_QUEUE"
PHASE_MARKER = contract.CONTROLLER_MARKER
ENVIRONMENT = source.ENVIRONMENT
INTERVAL = 900
PHASE_TIMEOUT = 900
TERMINAL_TIMEOUT = owner.TIMEOUT
CPU_STATUS = "ns_actual_lr_queue_cpu_archive_verified_gpu_unissued"
NEW_HELPERS = (
    "ns_lr_full_model_queue.py",
    "ns_lr_full_model_queue_test_support.py",
    "test_ns_lr_full_model_queue.py",
)
TEST_COUNT = 27
SCOPE = (
    "One original lower-LR queue/terminal gated actual-LR freeze/check/start/terminal sequence. "
    "Queue executes no tensors and performs no keeper operation. Preserve all accepted/rejected "
    "lower-LR and actual-LR decisions and original tolerances. No production or SwanLab authority. " + contract.SCOPE
)


def parent(root: Path) -> dict:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("NS actual-LR queue accepted owner receipt changed")
    return source.require_cpu(root, path)


def helper_sources(inherited: dict) -> dict[str, Path]:
    sources = {name: Path(inherited["helpers"]) / name for name in inherited["helpers_sha256"]}
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("NS actual-LR queue helper is absent or aliased")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("NS actual-LR queue helper escapes its accepted closure")
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
    """Authenticate the complete queue archive and unchanged parent source inventory."""
    raw, receipt = path.read_bytes(), read(path)
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
        raise ValueError("NS actual-LR queue complete CPU/source/fixture acceptance differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), inventory, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("NS actual-LR queue CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("NS actual-LR queue CPU staging needs hidden CUDA and unused external output")
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
            raise ValueError("NS actual-LR queue staged source or inherited fixture changed")
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
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "NS actual-LR queue CPU tests timed out"
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
        raise RuntimeError(f"NS actual-LR queue CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / NEW_HELPERS[0]:
        raise RuntimeError("NS actual-LR queue must execute its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("NS actual-LR queue imported a mutable inherited helper")
    return receipt


def readiness(root: Path) -> dict:
    """Observe original generations before authenticating their completed terminal handoff."""
    observed = predecessor.readiness(root)
    if not observed["ready"]:
        return observed
    return observed | {"status": "original_lower_lr_terminal_ready", "predecessor": predecessor.evidence(root)}


def empty_launch(root: Path) -> None:
    if any(path.exists() for path in (*contract.locations(root), root / controller.ARCHIVE, root / TERMINAL)):
        raise FileExistsError("NS actual-LR queue found existing one-shot execution evidence")


def phases(root: Path) -> list[tuple[str, list[str], Path]]:
    inherited = (root / PARENT).parent / "tracking"
    issued = contract.locations(root)[1]
    return [
        (
            "freeze",
            [
                contract.PYTHON,
                "-P",
                str(inherited / "ns_lr_full_model_gpu_controller.py"),
                "--root",
                str(root),
                "--phase",
                "freeze",
                "--cpu-receipt",
                str(root / PARENT),
            ],
            inherited,
        ),
        *(
            (
                phase,
                [
                    contract.PYTHON,
                    "-P",
                    str(issued / "ns_lr_full_model_gpu_controller.py"),
                    "--root",
                    str(root),
                    "--phase",
                    phase,
                ],
                issued,
            )
            for phase in ("check", "start")
        ),
    ]


def run_phase(root: Path, phase: str, argv: list[str], cwd: Path, state: dict) -> dict:
    work, marker, generations = root / WORK, uuid4().hex, {}
    log = work / f"{phase}.log"
    if log.exists() or any(entry["phase"] == phase for entry in state["phases"]):
        raise FileExistsError("NS actual-LR queue phase already has one-shot evidence")
    entry = {"phase": phase, "command": argv, "cwd": str(cwd), "marker": marker, "spawn_status": "starting"}
    state["active_phase"] = entry
    write(work / "state.json", state)
    environment = os.environ | ENVIRONMENT | {"PYTHONPATH": str(cwd), PHASE_MARKER: marker}
    environment.pop(MARKER, None)
    # A detached GPU controller replaces this marker, so phase cleanup cannot kill its new owner.
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
                raise RuntimeError(f"NS actual-LR queue {phase} failed")
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


def launch(root: Path, state: dict) -> dict:
    empty_launch(root)
    observation = readiness(root)
    if not observation["ready"] or state["phases"]:
        raise RuntimeError("NS actual-LR queue cannot repeat or bypass the original terminal gate")
    state.update(status="issuing_actual_lr_validation", original_observation=observation)
    write(root / WORK / "state.json", state)
    for phase, argv, cwd in phases(root):
        state["phases"].append(run_phase(root, phase, argv, cwd, state))
        write(root / WORK / "state.json", state)
    binding, helpers, _, _, _ = contract.locations(root)
    frozen, intent = read(binding), read(helpers.parent / "controller.json")
    own = process_context(
        intent["pid"],
        owner.controller_command(root),
        helpers,
        contract.CONTROLLER_MARKER,
        intent["marker"],
        contract.ENVIRONMENT,
    )
    require_acknowledged_controller(intent, own)
    if intent["binding_sha256"] != sha256(binding) or frozen["cpu_receipt_sha256"] != PARENT_SHA256:
        raise RuntimeError("NS actual-LR queue issued a different accepted owner")
    state.update(
        status="actual_lr_controller_acknowledged",
        issued_binding_sha256=sha256(binding),
        issued_controller=intent,
    )
    write(root / WORK / "state.json", state)
    return state


def issued_readiness(root: Path, state: dict) -> dict:
    binding, helpers, state_path, _, _ = contract.locations(root)
    frozen, intent = read(binding), read(helpers.parent / "controller.json")
    if (
        sha256(binding) != state["issued_binding_sha256"]
        or intent != state["issued_controller"]
        or intent["binding_sha256"] != sha256(binding)
        or frozen["cpu_receipt_sha256"] != PARENT_SHA256
        or frozen["controller_command"] != owner.controller_command(root)
        or frozen["environment"] != contract.ENVIRONMENT
    ):
        raise ValueError("NS actual-LR queue issued source or controller identity changed")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    owners = {
        "controller": handle(
            intent["generation"], helpers, contract.CONTROLLER_MARKER, intent["marker"], contract.ENVIRONMENT
        )
    }
    saved = read(state_path) if state_path.is_file() else {}
    if saved.get("spawn_status") == "acknowledged":
        owners["validator"] = handle(
            saved["child_generation"],
            Path(frozen["candidate"]),
            contract.CHILD_MARKER,
            saved["child_marker"],
            contract.ENVIRONMENT,
        )
    if any(item["status"] != "absent" for item in owners.values()):
        return {"status": "waiting_for_issued_actual_lr_owner", "ready": False, "owners": owners}
    if (
        saved.get("status") != "validation_complete"
        or saved["binding_sha256"] != state["issued_binding_sha256"]
        or saved["controller_pid"] != intent["pid"]
        or saved["controller_start_ticks"] != intent["start_ticks"]
        or any(key in saved for key in ("error", "cleanup_error", "deferred_signals"))
        or handoff.marker_processes(contract.CHILD_MARKER, saved["child_marker"])
    ):
        raise RuntimeError("NS actual-LR original owner exited without complete clean validation evidence")
    return {"status": "actual_lr_terminal_ready", "ready": True, "owners": owners}


def terminal_phase(root: Path, state: dict) -> dict:
    if (root / TERMINAL).exists() or not issued_readiness(root, state)["ready"]:
        raise RuntimeError("NS actual-LR terminal already exists or original owner remains")
    helpers = contract.locations(root)[1]
    argv = [
        contract.PYTHON,
        "-P",
        str(helpers / "ns_lr_full_model_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(root / TERMINAL),
    ]
    state["phases"].append(run_phase(root, "terminal", argv, helpers, state))
    result = read(root / TERMINAL)
    if (
        result["status"] != controller.TERMINAL_STATUS
        or result["binding_sha256"] != state["issued_binding_sha256"]
        or result["controller_pid_absent"] != state["issued_controller"]["pid"]
        or result["verifier_sha256"]
        != read(contract.locations(root)[0])["helpers_sha256"]["ns_lr_full_model_gpu_controller.py"]
        or any(sha256(root / name) != digest for name, digest in result["evidence_sha256"].items())
    ):
        raise ValueError("NS actual-LR terminal origin or retained evidence differs")
    state.update(
        status="actual_lr_terminal_reconstructed",
        terminal=str(root / TERMINAL),
        terminal_sha256=sha256(root / TERMINAL),
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
    readiness(root)
    empty_launch(root)
    work = root / WORK
    if work.exists() or local_node() != owner.NODE:
        raise RuntimeError("NS actual-LR queue already installed or allocation differs")
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
        or local_node() != owner.NODE
    ):
        raise RuntimeError("NS actual-LR queue original identity or allocation differs")
    state = {
        "status": "waiting_for_original_full_model_queue",
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
            while True:
                observation = readiness(root)
                state.update(status=observation["status"], original_observation=observation)
                write(work / "state.json", state)
                if observation["ready"]:
                    launch(root, state)
                    break
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
        raise RuntimeError("NS actual-LR queue requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Accepted queue phases require --cpu-receipt")
        cpu_receipt = args.cpu_receipt.resolve(strict=True)
        if args.phase == "check":
            result = accepted(root, cpu_receipt)
        elif args.phase == "start":
            result = start(root, cpu_receipt)
        else:
            result = execute(root, cpu_receipt)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
