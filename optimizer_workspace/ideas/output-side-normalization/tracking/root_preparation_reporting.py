# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render saved root diagnostic evidence without executing CUDA or inferring live ownership."""

import tarfile
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
from selected_lr_release_owner import candidate_files, read, sha256


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
    return lines
