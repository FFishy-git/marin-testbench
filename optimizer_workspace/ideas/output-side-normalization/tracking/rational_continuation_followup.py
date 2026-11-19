# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own one read-only issued terminal reconstruction after the original diagnostic exits."""

import argparse
import fcntl
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

import owned_output_handoff as handoff
import rational_continuation_contract as contract
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

BINDING_SHA256 = "a1a7d4ebd88b563b5f80d36ba7e0033752315043b6b799714e08e3cb25e27ba3"
WORK = "run-preparation/rational-continuation-terminal-followup-20261118"
TERMINAL = "run-preparation/rational-continuation-terminal-independent-20261118.json"
MARKER = "OUTPUT_MARIN_RATIONAL_CONTINUATION_FOLLOWUP"
CHILD_MARKER = "OUTPUT_MARIN_RATIONAL_CONTINUATION_READONLY"
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1"}
INTERVAL = 900
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
CPU_STATUS = "rational_continuation_read_only_followup_cpu_archive_verified"


def frozen(root: Path) -> dict:
    path, helpers, _, _, _ = contract.locations(root)
    if sha256(path) != BINDING_SHA256:
        raise ValueError("Read-only follow-up frozen binding changed")
    binding = read(path)
    require_helper_bytes(helpers, binding["helpers_sha256"])
    if binding["helpers"] != str(helpers) or binding["root"] != str(root):
        raise ValueError("Read-only follow-up source paths changed")
    return binding


def readiness(root: Path, binding: dict) -> dict:
    """Wait on exact original handles; only actual absence permits terminal verification."""
    _, helpers, state_path, _, _ = contract.locations(root)
    intent, state = read(helpers.parent / "controller.json"), read(state_path)
    if (
        intent["status"] != "acknowledged"
        or intent["binding_sha256"] != BINDING_SHA256
        or state["binding_sha256"] != BINDING_SHA256
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or intent["generation"]["pid"] != intent["pid"]
        or intent["generation"]["start_ticks"] != intent["start_ticks"]
        or intent["generation"]["command"] != intent["command"]
        or intent["command"] != binding["controller_command"]
        or state["command"] != binding["validator_command"]
        or state["child_generation"]["parent"] != intent["pid"]
    ):
        raise ValueError("Read-only follow-up original owner binding changed")
    owners = {}
    for name, expected, cwd, marker_name, marker in (
        ("controller", intent["generation"], helpers, contract.CONTROLLER_MARKER, intent["marker"]),
        (
            "validator",
            state["child_generation"],
            Path(binding["candidate"]),
            contract.CHILD_MARKER,
            state["child_marker"],
        ),
    ):
        observed = handoff.observed(expected["pid"])
        if observed is None:
            owners[name] = {"status": "absent", "expected": expected}
            continue
        if any(observed[key] != expected[key] for key in ("pid", "start_ticks", "group", "session", "command")):
            raise RuntimeError("Read-only follow-up original process generation was substituted")
        if observed["status"] in ("Z", "X", "x"):
            owners[name] = {"status": "exited_handle_present", "observed": observed}
            continue
        verified = process_context(expected["pid"], expected["command"], cwd, marker_name, marker, contract.ENVIRONMENT)
        if verified["start_ticks"] != expected["start_ticks"]:
            raise RuntimeError("Read-only follow-up owner changed during observation")
        owners[name] = {"status": "live_exact_generation", "observed": verified}
    if any(owner["status"] != "absent" for owner in owners.values()):
        return {
            "status": "waiting_for_original_owners",
            "ready": False,
            "owners": owners,
            "saved_state": state["status"],
        }
    if state["status"] != "diagnostic_complete":
        raise RuntimeError("Original rational diagnostic exited without completed evidence; failure preserved")
    return {"status": "terminal_ready", "ready": True, "owners": owners, "saved_state": state["status"]}


def accepted(path: Path) -> dict:
    receipt = read(path)
    files = receipt["files_sha256"]
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != CPU_STATUS
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["binding_sha256"] != BINDING_SHA256
        or receipt["cuda_visible_devices"] != ""
        or len(cases) != receipt["test_count"]
        or receipt["test_count"] != 8
        or sum(int(suite.attrib["tests"]) for suite in suites) != 8
        or any(int(suite.attrib[name]) for suite in suites for name in ("failures", "errors", "skipped"))
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
        or any(sha256(path.parent / name) != digest for name, digest in files.items())
        or any(sha256(Path(receipt[name])) != receipt[f"{name}_sha256"] for name in ("archive", "log", "tests"))
        or Path(__file__).resolve() != path.parent / Path(__file__).name
    ):
        raise ValueError("Read-only follow-up CPU/source acceptance changed")
    require_archive(Path(receipt["archive"]), files, alias=False)
    return receipt


def command(root: Path, cpu: Path, phase: str) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--cpu",
        str(cpu),
        "--phase",
        phase,
    ]


def terminal_phase(root: Path, helpers: Path, work: Path, state: dict) -> None:
    """Execute only the issued read-only verifier and retain child/cleanup evidence."""
    if state.get("terminal_command") or (root / TERMINAL).exists():
        raise FileExistsError("Read-only terminal phase already has one-shot evidence")
    argv = [
        contract.PYTHON,
        "-P",
        str(helpers / "rational_continuation_controller.py"),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(root / TERMINAL),
    ]
    marker, generations = uuid4().hex, {}
    state.update(status="reconstructing_terminal", terminal_command=argv, terminal_marker=marker)
    write(work / "state.json", state)
    log, child = work / "terminal.log", None
    environment = os.environ | ENVIRONMENT | {"PYTHONPATH": str(helpers), CHILD_MARKER: marker}
    environment.pop(MARKER, None)
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                argv,
                cwd=helpers,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process = acknowledged_process(child, argv, helpers, CHILD_MARKER, marker, ENVIRONMENT)
            state.update(terminal_pid=child.pid, terminal_generation=process)
            write(work / "state.json", state)
            code = child.wait(timeout=14400)
            state["terminal_exit_code"] = code
            if type(code) is not int or code != 0:
                raise RuntimeError("Issued read-only rational terminal reconstruction failed")
        result = read(root / TERMINAL)
        if result["status"] != "rational_continuation_terminal_independently_reconstructed_exact_restoration":
            raise ValueError("Issued terminal result has an unexpected scope")
        state.update(
            status="terminal_reconstructed",
            terminal_sha256=sha256(root / TERMINAL),
            terminal_status=result["status"],
            intervention_accepted=result["analysis"]["diagnostic_intervention_accepted"],
        )
    finally:
        if child is not None:
            finish_marked_child(child, marker, generations, CHILD_MARKER)
            state.update(terminal_cleanup_exit_code=child.returncode, terminal_descendants_absent=True)
        state.update(terminal_generations=generations, terminal_log=str(log), terminal_log_sha256=sha256(log))
        write(work / "state.json", state)


def start(root: Path, cpu: Path) -> dict:
    accepted(cpu)
    binding = frozen(root)
    readiness(root, binding)
    work = root / WORK
    if work.exists() or (root / TERMINAL).exists():
        raise FileExistsError("Read-only rational follow-up already has one-shot evidence")
    work.mkdir()
    intent = {
        "status": "starting",
        "command": command(root, cpu, "execute"),
        "marker": uuid4().hex,
        "runner_sha256": sha256(Path(__file__)),
        "cpu_receipt_sha256": sha256(cpu),
        "binding_sha256": BINDING_SHA256,
        "interval": INTERVAL,
        "scope": "Read-only issued terminal verification; no GPU allocation, training or production authorization.",
    }
    write(work / "intent.json", intent)
    child = None
    try:
        with (work / "followup.log").open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=work,
                env=os.environ
                | ENVIRONMENT
                | {"PYTHONPATH": f"{cpu.parent}:{binding['helpers']}", MARKER: intent["marker"]},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        own = acknowledged_process(child, intent["command"], work, MARKER, intent["marker"], ENVIRONMENT)
        intent.update(status="acknowledged", pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
    except BaseException as error:
        intent.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        if child is not None:
            finish_marked_child(child, intent["marker"], {}, MARKER)
        raise
    finally:
        write(work / "intent.json", intent)
    return intent


def execute(root: Path, cpu: Path) -> dict:
    accepted(cpu)
    binding, work = frozen(root), root / WORK
    intent, deadline = read(work / "intent.json"), time.monotonic() + 30
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
        intent = read(work / "intent.json")
    own = process_context(os.getpid(), command(root, cpu, "execute"), work, MARKER, intent["marker"], ENVIRONMENT)
    require_acknowledged_controller(intent, own)
    if intent["runner_sha256"] != sha256(Path(__file__)) or intent["cpu_receipt_sha256"] != sha256(cpu):
        raise ValueError("Read-only follow-up acknowledged source changed")
    state = {
        "status": "waiting_for_original_owners",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "runner_sha256": intent["runner_sha256"],
        "cpu_receipt_sha256": intent["cpu_receipt_sha256"],
        "binding_sha256": BINDING_SHA256,
        "interval": INTERVAL,
    }
    with (work / "execution.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                observation = readiness(root, frozen(root))
                state.update(status=observation["status"], original_observation=observation)
                write(work / "state.json", state)
                if observation["ready"]:
                    terminal_phase(root, Path(binding["helpers"]), work, state)
                    break
                time.sleep(INTERVAL)
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write(work / "state.json", state)
    return state


def cpu_acceptance(root: Path, out: Path) -> dict:
    if out.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Read-only CPU acceptance requires hidden CUDA and unused output")
    binding = frozen(root)
    source = Path(__file__).parent
    out.mkdir(parents=True)
    names = (Path(__file__).name, "test_rational_continuation_followup.py")
    for name in names:
        shutil.copy2(source / name, out / name)
    files = {name: sha256(out / name) for name in names}
    argv = [
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
        str(out / names[1]),
        "--junitxml",
        str(out / "tests.xml"),
    ]
    log = out / "cpu.log"
    with log.open("x") as stream:
        code = subprocess.run(
            argv,
            cwd=out,
            env=os.environ
            | ENVIRONMENT
            | {
                "UV_OFFLINE": "1",
                "PYTHONPATH": f"{out}:{binding['helpers']}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=900,
        ).returncode
    suites = list(ET.parse(out / "tests.xml").getroot().iter("testsuite")) if (out / "tests.xml").exists() else []
    cases = [case for suite in suites for case in suite.findall("testcase")]
    passed = (
        type(code) is int
        and code == 0
        and len(cases) == 8
        and sum(int(suite.attrib["tests"]) for suite in suites) == 8
        and not any(int(suite.attrib[name]) for suite in suites for name in ("failures", "errors", "skipped"))
        and not any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
        and files == {name: sha256(out / name) for name in names}
        and files == {name: sha256(source / name) for name in names}
    )
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in files:
            bundle.add(out / name, arcname=name, recursive=False)
    require_archive(archive, files, alias=False)
    receipt = {
        "status": CPU_STATUS if passed else "needs_attention",
        "exit_code": code,
        "test_count": len(cases),
        "cuda_visible_devices": "",
        "binding_sha256": BINDING_SHA256,
        "files_sha256": files,
        "command": argv,
        **{
            key: value
            for name, path in (("archive", archive), ("log", log), ("tests", out / "tests.xml"))
            for key, value in ((name, str(path)), (f"{name}_sha256", sha256(path) if path.exists() else None))
        },
    }
    write(out / "receipt.json", receipt)
    if not passed:
        raise RuntimeError("Read-only follow-up CPU failure preserved")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "start", "execute"))
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError("Read-only follow-up requires hidden CUDA and fixed environment")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("--out is required for CPU acceptance")
        result = cpu_acceptance(root, args.out.resolve())
    else:
        if args.cpu is None:
            parser.error("--cpu is required for issued follow-up")
        result = (start if args.phase == "start" else execute)(root, args.cpu.resolve(strict=True))
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
