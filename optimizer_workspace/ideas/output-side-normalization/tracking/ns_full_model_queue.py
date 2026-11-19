# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue one accepted full-model NS validation after the original NS queue exits."""

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

import ns_full_model_contract as contract
import ns_full_model_execution_source as source
import ns_full_model_gpu_owner as owner
import ns_full_model_predecessor as predecessor
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
from selected_lr_release_owner import read, sha256
from watch_wave5 import write
from wave7_release import local_node

PARENT = "run-preparation/ns-full-model-owner-cpu/single-pass-20261119T0300/receipt.json"
PARENT_SHA256 = "cbbd6514d9c6431bfd616ed73f52de13a5eb72011b96a24986e64e9b8e06eaff"
ORIGINAL_QUEUE_SOURCE = "run-preparation/ns-budget-queue-cpu/20261119T0024/tracking/ns_budget_queue.py"
ORIGINAL_QUEUE_SHA256 = "a1077fb77d1bebaab4889ed16637b1465fd7e702054d5264c48a80b20cbfc29a"
WORK = "run-preparation/ns-full-model-queue-20261119"
TERMINAL = "run-preparation/ns-full-model-terminal-independent-20261119.json"
MARKER = "OUTPUT_MARIN_NS_FULL_MODEL_QUEUE"
PHASE_MARKER = contract.CONTROLLER_MARKER
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONFAULTHANDLER": "1",
}
INTERVAL = 900
PHASE_TIMEOUT = 900
TERMINAL_TIMEOUT = owner.TIMEOUT
CPU_STATUS = "ns_full_model_queue_cpu_archive_verified_gpu_unissued"
NEW_HELPERS = ("ns_full_model_queue.py", "test_ns_full_model_queue.py")
TEST_COUNT = 25
SCOPE = (
    "One original-NS-owner-gated accepted full-model freeze/check/start/terminal sequence. "
    "Queue executes no tensors and performs no keeper operation. Preserve all accepted/rejected "
    "actual-config decisions and original tolerances. No production or SwanLab authority. " + contract.SCOPE
)


def parent(root: Path) -> dict:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("NS full-model queue accepted owner receipt changed")
    return source.require_cpu(root, path)


def helper_sources(inherited: dict) -> dict[str, Path]:
    sources = {name: Path(inherited["helpers"]) / name for name in inherited["helpers_sha256"]}
    original_queue = Path(inherited["root"]) / ORIGINAL_QUEUE_SOURCE
    if sha256(original_queue) != ORIGINAL_QUEUE_SHA256:
        raise ValueError("NS full-model queue original accepted process observer changed")
    sources[original_queue.name] = original_queue
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("NS full-model queue source is absent or aliased")
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
                raise ValueError("NS full-model queue source escapes its accepted closure")
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
        source.previous.parent.original.SAFE_MARKERS,
        "-q",
        str(out / "tracking" / NEW_HELPERS[-1]),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    raw, receipt = path.read_bytes(), read(path)
    inherited = parent(root)
    sources = helper_sources(inherited)
    hashes = {name: sha256(item) for name, item in sources.items()}
    inventory = inherited["inventory_sha256"] | {f"tracking/{name}": value for name, value in hashes.items()}
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != inventory
        or receipt["candidate"] != inherited["candidate"]
        or receipt["command"] != cpu_command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or len(cases) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(NEW_HELPERS[-1]).stem}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in inventory.items())
    ):
        raise ValueError("NS full-model queue complete CPU/source/fixture acceptance differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), inventory, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("NS full-model queue CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("NS full-model queue CPU staging needs hidden CUDA and unused external output")
    inherited = parent(root)
    sources = helper_sources(inherited)
    hashes = {name: sha256(item) for name, item in sources.items()}
    inventory = inherited["inventory_sha256"] | {f"tracking/{name}": value for name, value in hashes.items()}
    out.mkdir(parents=True)
    for name, digest in inventory.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        original = sources[Path(name).name] if name.startswith("tracking/") else (root / PARENT).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha256(target) != digest:
            raise ValueError("NS full-model queue staged source or inherited fixture changed")
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
            code, error = 124, "NS full-model queue CPU tests timed out"
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
        **{name: str(item) for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))},
        **{
            f"{name}_sha256": sha256(item)
            for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
            if item.is_file()
        },
        "scope": SCOPE,
    }
    if error:
        receipt["error"] = error
    write(out / "receipt.json", receipt)
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        receipt.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", receipt)
        raise RuntimeError(f"NS full-model queue CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / NEW_HELPERS[0]:
        raise RuntimeError("NS full-model queue must execute its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("NS full-model queue imported a mutable inherited helper")
    return receipt


def readiness(root: Path) -> dict:
    """Observe the original queue before requiring completed terminal/restoration evidence."""
    work = root / predecessor.QUEUE
    paths = [work / "intent.json", work / "state.json"]
    identities = {path: sha256(path) for path in paths}
    if identities[paths[0]] != predecessor.QUEUE_INTENT_SHA256:
        raise ValueError("NS full-model queue original NS acknowledgment changed")
    intent, state = (read(path) for path in paths)
    owners = {
        "original_queue": handle(intent["generation"], work, predecessor.QUEUE_MARKER, intent["marker"], ENVIRONMENT)
    }
    phases = [*state.get("phases", [])]
    active = state.get("active_phase", {})
    if active.get("spawn_status") == "acknowledged":
        phases.append(active)
    for entry in phases:
        owners[f"phase_{entry['phase']}"] = handle(
            entry["generation"],
            Path(entry["cwd"]),
            predecessor.previous.CONTROLLER_MARKER,
            entry["marker"],
            ENVIRONMENT,
        )
    if identities != {path: sha256(path) for path in paths}:
        raise RuntimeError("NS full-model original queue changed during observation")
    if any(item["status"] != "absent" for item in owners.values()):
        return {"status": "waiting_for_original_ns_queue", "ready": False, "owners": owners}
    previous = predecessor.evidence(root)
    return {"status": "original_ns_terminal_ready", "ready": True, "owners": owners, "predecessor": previous}


def empty_launch(root: Path) -> None:
    if any(
        path.exists()
        for path in (
            *contract.locations(root),
            root / "run-preparation/source-ns-full-model-20261119.tar",
            root / TERMINAL,
        )
    ):
        raise FileExistsError("NS full-model queue found existing one-shot execution evidence")


def phases(root: Path) -> list[tuple[str, list[str], Path]]:
    inherited_helpers = (root / PARENT).parent / "tracking"
    issued = contract.locations(root)[1]
    return [
        (
            "freeze",
            [
                contract.PYTHON,
                "-P",
                str(inherited_helpers / "ns_full_model_gpu_controller.py"),
                "--root",
                str(root),
                "--phase",
                "freeze",
                "--cpu-receipt",
                str(root / PARENT),
            ],
            inherited_helpers,
        ),
        *(
            (
                phase,
                [
                    contract.PYTHON,
                    "-P",
                    str(issued / "ns_full_model_gpu_controller.py"),
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
        raise FileExistsError("NS full-model queue phase already has one-shot evidence")
    entry = {"phase": phase, "command": argv, "cwd": str(cwd), "marker": marker, "spawn_status": "starting"}
    state["active_phase"] = entry
    write(work / "state.json", state)
    environment = os.environ | ENVIRONMENT | {"PYTHONPATH": str(cwd), PHASE_MARKER: marker}
    environment.pop(MARKER, None)
    # The separately acknowledged controller replaces this phase marker.
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
                raise RuntimeError(f"NS full-model queue {phase} failed")
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
        raise RuntimeError("NS full-model queue cannot repeat or bypass the original terminal gate")
    state.update(status="issuing_full_model_validation", original_observation=observation)
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
        raise RuntimeError("NS full-model queue issued a different accepted owner")
    state.update(
        status="full_model_controller_acknowledged",
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
        raise ValueError("NS full-model queue issued source or controller identity changed")
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
        return {"status": "waiting_for_issued_full_model_owner", "ready": False, "owners": owners}
    if (
        saved.get("status") != "validation_complete"
        or saved["binding_sha256"] != state["issued_binding_sha256"]
        or saved["controller_pid"] != intent["pid"]
        or saved["controller_start_ticks"] != intent["start_ticks"]
        or any(key in saved for key in ("error", "cleanup_error", "deferred_signals"))
        or handoff.marker_processes(contract.CHILD_MARKER, saved["child_marker"])
    ):
        raise RuntimeError("NS full-model original owner exited without complete clean validation evidence")
    return {"status": "full_model_terminal_ready", "ready": True, "owners": owners}


def terminal_phase(root: Path, state: dict) -> dict:
    if (root / TERMINAL).exists() or not issued_readiness(root, state)["ready"]:
        raise RuntimeError("NS full-model terminal already exists or original owner remains")
    helpers = contract.locations(root)[1]
    argv = [
        contract.PYTHON,
        "-P",
        str(helpers / "ns_full_model_gpu_controller.py"),
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
        result["status"] != "ns_full_model_terminal_independently_reconstructed_exact_restoration"
        or result["binding_sha256"] != state["issued_binding_sha256"]
        or result["controller_pid_absent"] != state["issued_controller"]["pid"]
        or result["verifier_sha256"]
        != read(contract.locations(root)[0])["helpers_sha256"]["ns_full_model_gpu_controller.py"]
        or any(sha256(root / name) != digest for name, digest in result["evidence_sha256"].items())
    ):
        raise ValueError("NS full-model terminal origin or retained evidence differs")
    state.update(
        status="full_model_terminal_reconstructed",
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
        raise RuntimeError("NS full-model queue already installed or allocation differs")
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
        raise RuntimeError("NS full-model queue original identity or allocation differs")
    state = {
        "status": "waiting_for_original_ns_queue",
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
        raise RuntimeError("NS full-model queue requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Accepted queue phase requires --cpu-receipt")
        path = args.cpu_receipt.resolve(strict=True)
        if args.phase == "check":
            result = accepted(root, path)
        else:
            result = (start if args.phase == "start" else execute)(root, path)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
