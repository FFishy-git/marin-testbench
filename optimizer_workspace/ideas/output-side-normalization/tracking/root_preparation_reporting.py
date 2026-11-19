# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render saved root diagnostic evidence without executing CUDA or inferring live ownership."""

import tarfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from prepare_root_preparation_diagnostic import archive_inventory as candidate_archive_inventory
from prepare_selected_lr_validation import require_archive
from root_preparation_contract import (
    CPU_STATUS as CANDIDATE_CPU_STATUS,
)
from root_preparation_contract import (
    FROZEN_STATUS,
    MODES,
    PREPARATION,
    PROTOCOLS,
    REGIMES,
    controller_command,
    locations,
    plan,
    validator_command,
)
from root_preparation_controller import CPU_STATUS, TESTS, cpu_command, inventory
from root_preparation_probe import command as probe_command
from root_preparation_validation import COMPLETE_STATUS
from root_stage_precision_reporting import lines as root_precision_lines
from root_start_gpu_reporting import lines as root_start_gpu_lines
from root_start_rational_reporting import lines as root_rational_lines
from selected_lr_release_owner import candidate_files, read, sha256

FAILED_RECONSTRUCTION_DIRECTORY = "run-preparation/root-failure-reconstruction-20261118T1618"
FAILED_RECONSTRUCTION_RESULT = "run-preparation/root-failed-diagnostic-reconstruction-20261118T1618.json"
FAILED_RECONSTRUCTION_RUNNER = "run-preparation/probes/run-root-failure-reconstruction-20261118T1618.py"
FAILED_RECONSTRUCTION_RUNNER_SHA256 = "aaf2ae178b982e4288cce445ce9bc9a6941fdd1828fc4f86e5fda658a83dae9f"
FAILED_RECONSTRUCTION_CPU = (
    "run-preparation/root-stage-replay-cpu/failed-owner-reconstruction-20261118T1615/receipt.json"
)
FAILED_RECONSTRUCTION_CPU_SHA256 = "aa4dd55db15ade6610dfdaac076974b085729c9dd115d986f319702fedc273a5"
FAILED_RECONSTRUCTION_STATUS = (
    "failed_root_diagnostic_independently_reconstructed_before_call_lifecycle_corrected_not_production"
)
ROOT_STAGE_QUEUE = "run-preparation/root-stage-replay-queue-20261118T1642"
ROOT_STAGE_QUEUE_CPU = "run-preparation/root-stage-queue-cpu/20261118T1656/receipt.json"
ROOT_STAGE_QUEUE_CPU_SHA256 = "5ee8f6fcdc73f05075eb557ad22120254c17c28b61ea7254329b7b3013cb5d7b"
ROOT_STAGE_QUEUE_RUNNER_SHA256 = "dc8102b025d7ab8c2579192864b5a2c322b3983deaa01f090ac4aa6f8b415b2e"
ROOT_STAGE_CPU = "run-preparation/root-stage-replay-cpu/sealed-failed-handoff-20261118T1638/receipt.json"
ROOT_STAGE_CPU_SHA256 = "2cebe11da69d328c4d1ddb2fa282566e0cf1938d842da008cdbff4008c0180d0"
ROOT_STAGE_PREDECESSOR_INTENT_SHA256 = "327ca1e1d26c20ab53b57ac5927de013d827bdcdde866c61ad2d93c6d8ec8851"
ROOT_STAGE_PHASES = ("seal-predecessor", "freeze", "check", "start")


def optional(path: Path) -> dict:
    return read(path) if path.is_file() else {}


def cpu_evidence(root: Path, path: Path) -> tuple[dict, dict]:
    receipt = read(path)
    parent_path = Path(receipt["candidate_cpu_receipt"])
    parent = read(parent_path)
    candidate, helpers = Path(parent["candidate"]), Path(receipt["helpers"])
    fixtures = {
        str(source.relative_to(candidate)): sha256(source)
        for source in (candidate / "experiment/configs").glob("*full_lr*.json")
    }
    if (
        path.is_symlink()
        or receipt["root"] != str(root)
        or receipt["candidate_cpu_receipt_sha256"] != sha256(parent_path)
        or parent["root"] != str(root)
        or parent["status"] != CANDIDATE_CPU_STATUS
        or "error" in parent
        or parent["exit_code"] != 0
        or type(parent["exit_code"]) is not int
        or parent["cuda_visible_devices"] != ""
        or parent["files_sha256"] != candidate_files(candidate)
        or parent["preparation_sha256"] != sha256(candidate / PREPARATION)
        or helpers != path.parent / "tracking"
        or helpers.is_symlink()
        or helpers != helpers.resolve()
        or receipt["command"] != cpu_command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["fixtures_sha256"] != fixtures
        or any(
            parent["files_sha256"][name] != value or sha256(path.parent / name) != value
            for name, value in fixtures.items()
        )
        or any(receipt["helpers_sha256"].get(name) != value for name, value in parent["helpers_sha256"].items())
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != value
            for name, value in receipt["helpers_sha256"].items()
        )
        or any(
            Path(receipt[name]) != path.parent / filename or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("log", "controller.log"), ("tests", "tests.xml"), ("archive", "source.tar"))
        )
    ):
        raise ValueError("Saved root executor CPU source, command, log or fixture bytes differ")
    prepared = read(candidate / PREPARATION)
    if (
        parent["files_sha256"] != prepared["files_sha256"]
        or parent["helpers_sha256"] != prepared["helpers_sha256"]
        or any(parent[f"{name}_sha256"] != sha256(Path(parent[name])) for name in ("log", "tests", "archive"))
        or any(sha256(Path(parent["helpers"]) / name) != value for name, value in parent["helpers_sha256"].items())
    ):
        raise ValueError("Saved root executor accepted parent evidence differs")
    require_archive(
        Path(parent["archive"]), candidate_archive_inventory(prepared, parent["preparation_sha256"]), alias=True
    )
    require_archive(
        Path(receipt["archive"]),
        {f"tracking/{name}": value for name, value in receipt["helpers_sha256"].items()} | fixtures,
        alias=False,
    )
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    counts = {
        name: sum(int(suite.attrib[name]) for suite in suites) for name in ("tests", "failures", "errors", "skipped")
    }
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        counts["tests"] != len(cases)
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(name).stem for name in TESTS}
        or (
            receipt["status"] == CPU_STATUS
            and (
                receipt["exit_code"] != 0
                or "error" in receipt
                or not cases
                or any(counts[name] for name in ("failures", "errors", "skipped"))
                or any(case.find(name) is not None for case in cases for name in ("failure", "error", "skipped"))
            )
        )
    ):
        raise ValueError("Saved root executor CPU acceptance contradicts complete XML coverage")
    return receipt, counts


def execution_evidence(root: Path) -> tuple[dict, dict, dict, dict]:
    binding_path, helpers, state_path, work, gpu_path = locations(root)
    frozen = read(binding_path)
    candidate = Path(frozen["candidate"])
    prepared = read(candidate / PREPARATION)
    digest = sha256(binding_path)
    if (
        frozen["status"] != FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["preparation_sha256"] != sha256(candidate / PREPARATION)
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["controller_cpu_receipt_sha256"] != sha256(Path(frozen["controller_cpu_receipt"]))
        or frozen["allocation_predecessor"] != prepared["allocation_predecessor"]
        or frozen["plan"] != plan(candidate, root)
        or frozen["commands"] != frozen["plan"]["commands"]
        or any(sha256(helpers / name) != value for name, value in frozen["helpers_sha256"].items())
        or any(
            sha256(root / name) != value for name, value in frozen["allocation_predecessor"]["evidence_sha256"].items()
        )
    ):
        raise ValueError("Saved root binding, source archive, plan or predecessor differs")
    require_archive(Path(frozen["archive"]), inventory(frozen), alias=True)
    intent, state, gpu = (optional(path) for path in (helpers.parent / "controller.json", state_path, gpu_path))
    if intent and (intent["binding_sha256"] != digest or intent["command"] != controller_command(root)):
        raise ValueError("Saved root controller intent differs from frozen command")
    if state and (
        state["binding_sha256"] != digest
        or state["candidate"] != str(candidate)
        or state["allocation_predecessor"] != frozen["allocation_predecessor"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or ("command" in state and state["command"] != validator_command(root, candidate))
    ):
        raise ValueError("Saved root executor state differs from acknowledged ownership")
    if gpu:
        if (
            not state
            or gpu["execution_binding_sha256"] != digest
            or gpu["candidate_preparation_sha256"] != frozen["preparation_sha256"]
            or gpu["commands"] != frozen["commands"]
            or gpu["owner"]
            != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
            or len(gpu["gpu_commands"]) > len(frozen["commands"])
            or len(gpu["cold_probes"]) > 2
        ):
            raise ValueError("Saved root GPU command plan, source or owner differs")
        for index, entry in enumerate(gpu["cold_probes"]):
            log = work / f"cold-root-{index}.log"
            if (
                type(entry["index"]) is not int
                or entry["index"] != index
                or entry["command"] != probe_command(root, candidate, index)
                or type(entry["exit_code"]) is not int
                or entry["log"] != str(log)
                or entry["log_sha256"] != sha256(log)
            ):
                raise ValueError("Saved root cold command/log provenance differs")
        for index, entry in enumerate(gpu["gpu_commands"]):
            command = frozen["commands"][index]
            regime, protocol, mode = REGIMES[index // 6], PROTOCOLS[index % 6 // 3], MODES[index % 3]
            log = work / f"{regime}_{protocol}_{mode}.log"
            if (
                entry["command"] != command
                or (entry["regime"], entry["protocol"], entry["mode"]) != (regime, protocol, mode)
                or type(entry["exit_code"]) is not int
                or entry["log"] != str(log)
                or entry["log_sha256"] != sha256(log)
            ):
                raise ValueError("Saved root full-model command/log provenance differs")
    return frozen, intent, state, gpu


def failed_reconstruction_evidence(root: Path) -> tuple[dict, dict, dict]:
    """Authenticate a separate saved reconstruction without changing the failed GPU receipt."""
    work = root / FAILED_RECONSTRUCTION_DIRECTORY
    paths = [
        work / "intent.json",
        work / "state.json",
        root / FAILED_RECONSTRUCTION_CPU,
        root / FAILED_RECONSTRUCTION_RUNNER,
    ]
    hashes = {path: sha256(path) for path in paths}
    intent, state, cpu = (read(path) for path in paths[:3])
    frozen, _, failed, gpu = execution_evidence(root)
    command = [frozen["python"], "-P", str(paths[3]), "--phase", "execute"]
    if (
        hashes[paths[2]] != FAILED_RECONSTRUCTION_CPU_SHA256
        or hashes[paths[3]] != FAILED_RECONSTRUCTION_RUNNER_SHA256
        or intent["status"] != "acknowledged"
        or intent["command"] != command
        or not intent["marker"]
        or intent["pid"] != intent["generation"]["pid"]
        or intent["start_ticks"] != intent["generation"]["start_ticks"]
        or intent["generation"]["command"] != command
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or intent["generation"]["status"] in ("Z", "X", "x")
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or state["command"] != command
        or state["status"] not in ("reconstructing", "reconstructed", "needs_attention")
        or failed["status"] != "needs_attention"
        or gpu["status"] != "needs_attention"
        or cpu["status"] != "root_stage_replay_cpu_archive_verified_gpu_unissued"
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or "error" in cpu
        or cpu["helpers"] != str(paths[2].parent / "tracking")
        or any(
            record["runner_sha256"] != FAILED_RECONSTRUCTION_RUNNER_SHA256
            or record["cpu_receipt_sha256"] != FAILED_RECONSTRUCTION_CPU_SHA256
            or record["binding_sha256"] != sha256(locations(root)[0])
            or record["out"] != str(root / FAILED_RECONSTRUCTION_RESULT)
            for record in (intent, state)
        )
        or any(sha256(Path(cpu["helpers"]) / name) != value for name, value in cpu["helpers_sha256"].items())
        or any(cpu[f"{name}_sha256"] != sha256(Path(cpu[name])) for name in ("log", "tests", "archive"))
        or any(sha256(paths[2].parent / name) != value for name, value in cpu["fixtures_sha256"].items())
    ):
        raise ValueError("Separate root reconstruction source, acknowledgment or original failure differs")
    require_archive(
        Path(cpu["archive"]),
        {f"tracking/{name}": value for name, value in cpu["helpers_sha256"].items()} | cpu["fixtures_sha256"],
        alias=False,
    )
    suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or not cases
        or any(case.find(name) is not None for case in cases for name in ("failure", "error", "skipped"))
        or any(int(suite.attrib[name]) for suite in suites for name in ("failures", "errors", "skipped"))
    ):
        raise ValueError("Separate root reconstruction archived CPU tests are incomplete")
    times = [datetime.fromisoformat(record["time"]) for record in (failed, cpu, intent, state)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("Separate root reconstruction terminal, CPU, acknowledgment or state chronology differs")
    result_path, result = root / FAILED_RECONSTRUCTION_RESULT, {}
    if state["status"] == "reconstructed":
        paths.append(result_path)
        hashes[result_path] = sha256(result_path)
        result = read(result_path)
        required = {
            locations(root)[0],
            locations(root)[2],
            locations(root)[4],
            locations(root)[1].parent / "controller.json",
            locations(root)[1].parent / "controller.log",
            Path(failed["log"]),
        }
        if (
            type(state["exit_code"]) is not int
            or state["exit_code"] != 0
            or "error" in state
            or state["result_status"] != FAILED_RECONSTRUCTION_STATUS
            or state["result_sha256"] != hashes[result_path]
            or result["status"] != FAILED_RECONSTRUCTION_STATUS
            or result["original_status"] != failed["status"]
            or result["original_failure"] != gpu["error"]
            or result["terminal_time"] != failed["time"]
            or result["binding_sha256"] != state["binding_sha256"]
            or result["verifier_sha256"] != cpu["helpers_sha256"]["root_preparation_reconstruction.py"]
            or result["gpu_commands"] != gpu["gpu_commands"]
            or result["tolerance"] != gpu["tolerance"]
            or result["controller_pid_absent"] != failed["controller_pid"]
            or result["validator_pid_absent"] != failed["child_pid"]
            or result["marked_descendants_absent"] is not True
            or set(result["evidence_sha256"]) != {str(path.relative_to(root)) for path in required}
            or any(sha256(root / name) != value for name, value in result["evidence_sha256"].items())
            or result["analysis"]["diagnostic_intervention_accepted"] is not all(result["analysis"]["gates"].values())
            or datetime.fromisoformat(result["time"]).tzinfo is None
            or not times[-2] <= datetime.fromisoformat(result["time"]) <= times[-1]
        ):
            raise ValueError("Separate root reconstruction result is not bound to the failed diagnostic")
    elif result_path.exists() and state["status"] != "needs_attention":
        raise ValueError("Separate root reconstruction output exists without a terminal acknowledgment")
    if state["status"] == "needs_attention" and (
        type(state["exit_code"]) is not int or state["exit_code"] != 1 or not state["error"]
    ):
        raise ValueError("Separate root reconstruction failure is not recorded")
    if any(sha256(path) != hashes[path] for path in paths):
        raise ValueError("Separate saved root reconstruction evidence changed during reporting")
    return intent, state, result


def root_stage_queue_evidence(root: Path) -> tuple[dict, dict, dict]:
    """Authenticate saved queue progress without inspecting or allocating a GPU."""
    work, acceptance = root / ROOT_STAGE_QUEUE, root / ROOT_STAGE_QUEUE_CPU
    paths = [acceptance, root / ROOT_STAGE_CPU, work / "intent.json", work / "state.json"]
    hashes = {path: sha256(path) for path in paths}
    cpu, parent, intent, state = map(read, paths)
    archived = acceptance.parent
    runner = archived / "queue-root-stage-after-reconstruction-20261118T1642.py"
    helpers = Path(parent["helpers"])
    issued = root / "controllers/root-stage-replay-20261118/tracking"
    command = ["/opt/conda/envs/torch-base/bin/python", "-P", str(runner), "--phase", "execute"]
    predecessor_intent = root / FAILED_RECONSTRUCTION_DIRECTORY / "intent.json"
    if (
        hashes[acceptance] != ROOT_STAGE_QUEUE_CPU_SHA256
        or hashes[root / ROOT_STAGE_CPU] != ROOT_STAGE_CPU_SHA256
        or sha256(predecessor_intent) != ROOT_STAGE_PREDECESSOR_INTENT_SHA256
        or cpu["status"] != "root_stage_queue_cpu_archive_verified_gpu_unissued"
        or cpu["runner"] != str(runner)
        or cpu["runner_sha256"] != ROOT_STAGE_QUEUE_RUNNER_SHA256
        or sha256(runner) != ROOT_STAGE_QUEUE_RUNNER_SHA256
        or cpu["cpu_receipt_sha256"] != ROOT_STAGE_CPU_SHA256
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or any(cpu[f"{name}_sha256"] != sha256(Path(cpu[name])) for name in ("log", "tests", "archive"))
        or any(sha256(archived / name) != value for name, value in cpu["files_sha256"].items())
        or parent["status"] != "root_stage_replay_cpu_archive_verified_gpu_unissued"
        or parent["root"] != str(root)
        or any(sha256(helpers / name) != value for name, value in parent["helpers_sha256"].items())
        or any(sha256(helpers.parent / name) != value for name, value in parent["fixtures_sha256"].items())
        or any(parent[f"{name}_sha256"] != sha256(Path(parent[name])) for name in ("log", "tests", "archive"))
        or intent["status"] != "acknowledged"
        or intent["command"] != command
        or intent["generation"]["command"] != command
        or intent["pid"] != intent["generation"]["pid"]
        or intent["start_ticks"] != intent["generation"]["start_ticks"]
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or intent["generation"]["status"] in ("Z", "X", "x")
        or not intent["marker"]
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or state["status"]
        not in (
            "waiting_for_independent_reconstruction",
            "sealing_terminal_predecessor",
            "preparing_diagnostic",
            "diagnostic_controller_acknowledged",
            "needs_attention",
        )
        or any(
            record["runner_sha256"] != ROOT_STAGE_QUEUE_RUNNER_SHA256
            or record["queue_cpu_receipt_sha256"] != ROOT_STAGE_QUEUE_CPU_SHA256
            or record["cpu_receipt_sha256"] != ROOT_STAGE_CPU_SHA256
            or record["predecessor_intent_sha256"] != ROOT_STAGE_PREDECESSOR_INTENT_SHA256
            or record["predecessor_pid"] != read(predecessor_intent)["pid"]
            or record["predecessor_start_ticks"] != read(predecessor_intent)["start_ticks"]
            or type(record["interval_seconds"]) is not int
            or record["interval_seconds"] != 900
            for record in (intent, state)
        )
    ):
        raise ValueError("Saved root-stage queue archive, acknowledgment or predecessor identity differs")
    require_archive(Path(cpu["archive"]), cpu["files_sha256"], alias=False)
    require_archive(
        Path(parent["archive"]),
        {f"tracking/{name}": value for name, value in parent["helpers_sha256"].items()} | parent["fixtures_sha256"],
        alias=False,
    )
    for receipt, expected_tests in ((cpu, 6), (parent, 61)):
        suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
        cases = [case for suite in suites for case in suite.findall("testcase")]
        if (
            len(cases) != expected_tests
            or sum(int(suite.attrib["tests"]) for suite in suites) != expected_tests
            or any(int(suite.attrib[name]) for suite in suites for name in ("errors", "failures", "skipped"))
            or any(case.find(name) is not None for case in cases for name in ("error", "failure", "skipped"))
        ):
            raise ValueError("Saved root-stage queue archived tests are incomplete")
    commands = state["commands"]
    if (
        len(commands) > len(ROOT_STAGE_PHASES)
        or [entry["phase"] for entry in commands] != list(ROOT_STAGE_PHASES[: len(commands)])
        or (state["status"] == "waiting_for_independent_reconstruction" and commands)
        or ("error" in state and state["status"] != "needs_attention")
    ):
        raise ValueError("Saved root-stage queue phase order or terminal failure differs")
    for index, entry in enumerate(commands):
        directory = helpers if entry["phase"] in ("seal-predecessor", "freeze") else issued
        expected = [
            command[0],
            "-P",
            str(directory / "root_stage_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            entry["phase"],
        ]
        if entry["phase"] == "freeze":
            expected.extend(("--cpu-receipt", str(root / ROOT_STAGE_CPU)))
        if (
            entry["command"] != expected
            or entry["status"] not in ("starting", "running", "complete", "needs_attention")
            or (index < len(commands) - 1 and entry["status"] != "complete")
            or (entry["status"] == "complete" and (type(entry["exit_code"]) is not int or entry["exit_code"] != 0))
        ):
            raise ValueError("Saved root-stage phase source, order or execution differs")
        if "log_sha256" in entry:
            log = work / f"{entry['phase']}.log"
            paths.append(log)
            hashes[log] = sha256(log)
            if hashes[log] != entry["log_sha256"]:
                raise ValueError("Saved root-stage phase retained log differs")
        elif entry["status"] == "complete":
            raise ValueError("Saved root-stage completed phase lacks retained log")
    diagnostic = {}
    if state["status"] == "diagnostic_controller_acknowledged":
        binding_path = root / "run-preparation/root-stage-replay-binding-20261118.json"
        controller_path = issued.parent / "controller.json"
        paths.extend((binding_path, controller_path))
        hashes.update({path: sha256(path) for path in (binding_path, controller_path)})
        frozen, controller = read(binding_path), read(controller_path)
        if (
            len(commands) != 4
            or any(entry["status"] != "complete" for entry in commands)
            or state["controller"] != controller
            or controller["status"] != "acknowledged"
            or controller["binding_sha256"] != hashes[binding_path]
            or controller["command"] != frozen["controller_command"]
            or controller["generation"]["command"] != controller["command"]
            or controller["pid"] != controller["generation"]["pid"]
            or controller["start_ticks"] != controller["generation"]["start_ticks"]
            or controller["pid"] != state["controller_observation"]["pid"]
            or controller["start_ticks"] != state["controller_observation"]["start_ticks"]
            or frozen["status"] != "frozen_root_stage_replay_diagnostic_only"
            or frozen["cpu_receipt_sha256"] != ROOT_STAGE_CPU_SHA256
            or frozen["helpers_sha256"] != parent["helpers_sha256"]
            or any(sha256(issued / name) != value for name, value in frozen["helpers_sha256"].items())
        ):
            raise ValueError("Saved root-stage queue successor acknowledgment differs from issued binding")
        state_path = root / "run-preparation/root-stage-replay-state-20261118.json"
        if state_path.exists():
            paths.append(state_path)
            hashes[state_path] = sha256(state_path)
            diagnostic = read(state_path)
            if (
                diagnostic["binding_sha256"] != hashes[binding_path]
                or diagnostic["controller_pid"] != controller["pid"]
                or diagnostic["controller_start_ticks"] != controller["start_ticks"]
                or diagnostic["predecessor"] != frozen["predecessor"]
            ):
                raise ValueError("Saved root-stage diagnostic state differs from its acknowledged owner")
    times = [datetime.fromisoformat(record["time"]) for record in (parent, cpu, intent, state)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("Saved root-stage queue acceptance or execution chronology differs")
    if any(sha256(path) != hashes[path] for path in paths):
        raise ValueError("Saved root-stage queue evidence changed during reporting")
    return intent, state, diagnostic


def executor_lines(root: Path, local_time) -> list[str]:
    lines = []
    for path in sorted(root.glob("run-preparation/root-executor-cpu/**/receipt.json")):
        current = ["### Restored-Root Diagnostic Executor", "", f"`PVC/{path.relative_to(root)}`."]
        try:
            receipt, counts = cpu_evidence(root, path)
            current.append(
                f"Saved CPU archive at {local_time(receipt.get('time'))}: `{receipt['status']}`, "
                f"integer exit `{receipt['exit_code']}`; {counts['tests']} tests, "
                f"{counts['failures']} failures and {counts['errors']} errors."
            )
            if receipt["status"] == CPU_STATUS:
                current.append(
                    f"Source/archive-bound executor acceptance: {len(receipt['helpers_sha256'])} helpers and "
                    f"{len(receipt['fixtures_sha256'])} fixtures; complete XML verified, CUDA hidden."
                )
            else:
                current.append("Preserved executor denial; passing test exit alone does not establish acceptance.")
                if "error" in receipt:
                    current.append(f"Preserved error: `{receipt['error']}`.")
            current.append("CPU evidence establishes no GPU result, production gate or optimizer-quality claim.")
        except (ValueError, KeyError, TypeError, OSError, ET.ParseError, RuntimeError, tarfile.TarError) as error:
            current.append(f"Saved executor evidence differs: `{type(error).__name__}: {error}`; acceptance withheld.")
        lines.extend([*current, ""])
    if locations(root)[0].is_file():
        current = ["### Restored-Root GPU Diagnostic", "", f"`PVC/{locations(root)[0].relative_to(root)}`."]
        try:
            frozen, intent, state, gpu = execution_evidence(root)
            current.append(f"Saved diagnostic freeze at {local_time(frozen.get('time'))}; twelve bounded commands.")
            if intent:
                current.append(f"Saved controller intent `{intent['status']}` at {local_time(intent.get('time'))}.")
            if state:
                current.append(f"Saved executor state `{state['status']}` at {local_time(state.get('time'))}.")
                if "keeper_restore" in state:
                    current.append(f"Recorded keeper restoration: `{state['keeper_restore']}`.")
                if "error" in state:
                    current.append(f"Preserved executor error: `{state['error']}`.")
            if gpu:
                current.append(
                    f"Saved GPU receipt `{gpu['status']}` at {local_time(gpu.get('time'))}: "
                    f"{sum(entry['exit_code'] == 0 for entry in gpu['cold_probes'])}/2 cold-probe exits and "
                    f"{sum(entry['exit_code'] == 0 for entry in gpu['gpu_commands'])}/12 full-model exits "
                    "are integer zero."
                )
                if "error" in gpu:
                    current.append(f"Preserved GPU error: `{gpu['error']}`.")
                if gpu["status"] == COMPLETE_STATUS:
                    current.append(
                        "Saved diagnostic intervention predicate: "
                        f"`{gpu['analysis']['diagnostic_intervention_accepted']}`. "
                        "Independent retained-output/checkpoint reconstruction is still required."
                    )
            current.append(
                "Saved receipts do not establish live occupancy. No production acceptance, full-budget result "
                "or SwanLab import is inferred; all original gates remain required."
            )
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, tarfile.TarError) as error:
            current.append(f"Saved GPU diagnostic evidence differs: `{type(error).__name__}: {error}`; claims withheld.")
        lines.extend([*current, ""])
    work = root / FAILED_RECONSTRUCTION_DIRECTORY
    if (work / "intent.json").is_file():
        current = ["### Independent Failed-Root Reconstruction", "", f"`PVC/{FAILED_RECONSTRUCTION_DIRECTORY}`."]
        try:
            intent, state, result = failed_reconstruction_evidence(root)
            current.append(
                f"Saved one-shot acknowledgment at {local_time(intent['time'])}: "
                f"PID `{intent['pid']}/{intent['start_ticks']}`."
            )
            current.append(f"Saved reconstruction state `{state['status']}` at {local_time(state['time'])}.")
            if result:
                current.append(f"Separate reconstructed receipt: `PVC/{FAILED_RECONSTRUCTION_RESULT}`.")
                for name, accepted in result["analysis"]["gates"].items():
                    current.append(f"Saved diagnostic predicate `{name}`: `{accepted}`.")
            if "error" in state:
                current.append(f"Preserved reconstruction error: `{state['error']}`.")
            current.append(
                "The original GPU receipt remains `needs_attention`. Saved acknowledgment does not establish "
                "process liveness. No CUDA launch, continuation acceptance, optimizer result "
                "or SwanLab import is inferred."
            )
        except (ValueError, KeyError, TypeError, OSError, ET.ParseError, RuntimeError, tarfile.TarError) as error:
            current.append(
                f"Separate reconstruction evidence differs: `{type(error).__name__}: {error}`; claims withheld."
            )
        lines.extend([*current, ""])
    if (root / ROOT_STAGE_QUEUE / "intent.json").is_file():
        current = ["### Queued Root-Stage Diagnostic", "", f"`PVC/{ROOT_STAGE_QUEUE}`."]
        try:
            intent, state, diagnostic = root_stage_queue_evidence(root)
            current.append(
                f"Saved archived queue acknowledgment at {local_time(intent['time'])}: "
                f"PID `{intent['pid']}/{intent['start_ticks']}`; 900-second observations."
            )
            current.append(f"Saved queue state `{state['status']}` at {local_time(state['time'])}.")
            for entry in state["commands"]:
                current.append(
                    f"One-shot phase `{entry['phase']}`: `{entry['status']}`, "
                    f"recorded exit `{entry.get('exit_code', 'pending')}`."
                )
            if diagnostic:
                current.append(
                    f"Saved separately owned diagnostic state `{diagnostic['status']}` at "
                    f"{local_time(diagnostic['time'])}; {len(diagnostic['executions'])}/2 retained workers."
                )
                if "keeper_restore" in diagnostic:
                    current.append(f"Recorded diagnostic keeper restoration: `{diagnostic['keeper_restore']}`.")
                if "error" in diagnostic:
                    current.append(f"Preserved diagnostic failure: `{diagnostic['error']}`.")
            if "error" in state:
                current.append(f"Preserved queue failure: `{state['error']}`.")
            current.append(
                "Saved queue records establish no current liveness or numerical result. Full retained-output "
                "reconstruction and exact restoration remain separate; all production gates stay required."
            )
        except (ValueError, KeyError, TypeError, OSError, ET.ParseError, RuntimeError, tarfile.TarError) as error:
            current.append(
                f"Saved root-stage queue evidence differs: `{type(error).__name__}: {error}`; claims withheld."
            )
        lines.extend([*current, ""])
    try:
        lines.extend(root_precision_lines(root, local_time))
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError, RuntimeError, tarfile.TarError) as error:
        lines.extend(
            [
                "### Root-Stage Terminal And Precision Follow-Up",
                "",
                f"Saved precision follow-up evidence differs: `{type(error).__name__}: {error}`; claims withheld.",
                "",
            ]
        )
    try:
        lines.extend(root_rational_lines(root, local_time))
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError, RuntimeError, tarfile.TarError) as error:
        lines.extend(
            [
                "### Cancellation-Free Root-Start CPU Hypothesis",
                "",
                f"Saved rational-start CPU evidence differs: `{type(error).__name__}: {error}`; claims withheld.",
                "",
            ]
        )
    try:
        lines.extend(root_start_gpu_lines(root, local_time))
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError, RuntimeError, tarfile.TarError) as error:
        lines.extend(
            [
                "### Separately Owned CUDA Start-Accuracy Trial",
                "",
                f"Saved start-accuracy owner evidence differs: `{type(error).__name__}: {error}`; claims withheld.",
                "",
            ]
        )
    return lines
