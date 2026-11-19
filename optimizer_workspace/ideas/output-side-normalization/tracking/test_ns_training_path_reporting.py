# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""File-only numerical CPU reporting refuses altered evidence and unaccepted execution."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_reporting as reporting
import pytest
import update_endeavor as journal


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def source_archive(directory: Path, inventory: dict) -> Path:
    path = directory / "source.tar"
    with tarfile.open(path, "w") as bundle:
        for name in inventory:
            bundle.add(directory / name, arcname=name, recursive=False)
        bundle.add(directory / reporting.ALIAS, arcname=reporting.ALIAS, recursive=False)
    return path


def cpu_command(directory: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        "/opt/conda/envs/torch-base/bin/python",
        "--with",
        "pytest-timeout",
        "python",
        "-P",
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(directory),
        "-p",
        "no:cacheprovider",
        "-p",
        "root_diagnostic_pytest_guard",
        "--timeout=60",
        "--session-timeout=600",
        "--import-mode=prepend",
        "-m",
        "not slow and not integration and not data_integration and not cluster "
        "and not requires_cluster and not docker and not manual",
        "-q",
        str(directory / "tracking/test_ns_training_path_validation.py"),
        "--junitxml",
        str(directory / "tests.xml"),
    ]


@pytest.fixture
def issued(tmp_path, monkeypatch):
    root = tmp_path / "root"
    directory = (root / reporting.CPU).parent
    directory.mkdir(parents=True)
    candidate = root / "candidates/inherited-numerical-not-execution-owner"
    candidate.mkdir(parents=True)
    helpers = {
        name: f"# Issued source {name}; must never be imported by the reporter.\n"
        for name in (
            reporting.NEW_HELPERS
            | set(reporting.contract.NUMERICAL_READERS)
            | {"ns_training_path_predecessor.py", "prepare_ns_training_path_predecessor.py"}
        )
    }
    helpers["recovery_keeper_manager.sh"] = "#!/bin/sh\nexit 0\n"
    for index in range(reporting.HELPER_COUNT - len(helpers)):
        helpers[f"accepted_helper_{index}.py"] = f"# predecessor helper {index}\n"
    for name, content in helpers.items():
        path = directory / "tracking" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
    hashes = {name: digest(directory / "tracking" / name) for name in helpers}
    reader_hashes = {name: hashes[name] for name in reporting.contract.NUMERICAL_READERS}
    monkeypatch.setattr(reporting.contract, "NUMERICAL_READERS", reader_hashes)
    training = directory / "experiment/train.py"
    training.parent.mkdir()
    training.write_text("# CPU fixture for byte-pinned original trainer\n")
    (directory / "experiment/fused_attribution_owner.py").write_text("# accepted parent owner\n")
    monkeypatch.setattr(reporting.contract, "TRAINING_FILES", {"experiment/train.py": digest(training)})
    pinned = directory / reporting.PINNED
    pinned.mkdir(parents=True)
    (pinned / "muon.py").write_text("# issued NS source\n")
    (directory / reporting.ALIAS).symlink_to("pinned-sources-materialized", target_is_directory=True)
    for spec in reporting.contract.specs():
        reference = {
            "experiment": {"arm": spec["family"], "refresh_install_age": 1, "diagonal_backend": "triton"},
            "training": {
                "steps": 5120,
                "tokens": 2684354560,
                "batch_tokens": 524288,
                "world_size": 8,
                "seed": 0,
                "global_batch_sequences": 256,
            },
            "model": {"sequence_length": 2048},
            "schedule": {
                "warmup_steps": 0,
                "stable_steps": 1024,
                "decay_steps": 4096,
                "kind": "linear",
                "min_lr_ratio": 0.0,
            },
            "optimizer": {"lr": 0.04525483392, "ns_mode": "tpu", "ns_steps": 10},
            "runtime": {"compile": "default", "attention": "FA2", "head": "compiled"},
            "root": {"graphs": True},
        }
        record(directory / f"experiment/configs/{spec['family']}_full_lr_lower_0p5m_seed0.json", reference)
        actual = json.loads(json.dumps(reference))
        actual["optimizer"].update(ns_steps=spec["budget"], ns_mode=spec["operand"])
        record(directory / spec["config_name"], actual)
    inventory = {
        str(path.relative_to(directory)): digest(path)
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    inventory[f"{reporting.ALIAS}/muon.py"] = digest(pinned / "muon.py")
    for index in range(reporting.INVENTORY_COUNT - len(inventory)):
        path = directory / f"fixtures/input_{index}.json"
        record(path, {"accepted_fixture": index})
        inventory[str(path.relative_to(directory))] = digest(path)
    inherited_helpers = {name: value for name, value in hashes.items() if name not in reporting.NEW_HELPERS}
    inherited_inventory = {
        name: value
        for name, value in inventory.items()
        if name not in {f"tracking/{helper}" for helper in reporting.NEW_HELPERS}
    }
    parent_path = root / reporting.PARENT_CPU
    record(
        parent_path,
        {
            "status": "ns_training_path_predecessor_cpu_archive_verified_gpu_unissued",
            "root": str(root),
            "candidate": str(candidate),
            "helpers_sha256": inherited_helpers,
            "inventory_sha256": inherited_inventory,
        },
    )
    monkeypatch.setattr(reporting, "PARENT_SHA256", digest(parent_path))
    tests = ET.Element("testsuites")
    suite = ET.SubElement(
        tests,
        "testsuite",
        {
            "tests": "34",
            "failures": "0",
            "errors": "0",
            "skipped": "0",
            "time": "148.833",
        },
    )
    for index in range(34):
        ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "test_ns_training_path_validation",
                "name": f"test_numerical_case_{index}",
            },
        )
    ET.ElementTree(tests).write(directory / "tests.xml")
    (directory / "cpu.log").write_text("34 passed in 148.833s; no GPU commands executed\n")
    source_archive(directory, inventory)
    for name, filename in (("ARCHIVE", "source.tar"), ("TESTS", "tests.xml"), ("LOG", "cpu.log")):
        monkeypatch.setattr(reporting, f"{name}_SHA256", digest(directory / filename))
    receipt = {
        "status": reporting.CPU_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_cpu_receipt_sha256": digest(parent_path),
        "helpers": str(directory / "tracking"),
        "helpers_sha256": hashes,
        "inventory_sha256": inventory,
        "command": cpu_command(directory),
        "cwd": str(directory),
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "NCCL_INSPECTOR_ENABLE": "0",
            "PYTHONFAULTHANDLER": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "cuda_visible_devices": "",
        "exit_code": 0,
        "scope": (
            "Separate CPU/source-archive numerical component; execution owner and launch unissued. "
            + reporting.contract.SCOPE
        ),
        "time": "2026-11-18T23:10:02.160788+00:00",
    }
    for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log")):
        receipt[name] = str(directory / filename)
        receipt[f"{name}_sha256"] = digest(directory / filename)
    record(root / reporting.CPU, receipt)
    monkeypatch.setattr(reporting, "CPU_SHA256", digest(root / reporting.CPU))
    return root, directory, receipt


def test_issued_numerical_cpu_reports_prelaunch_without_execution_or_mutable_helpers(issued, monkeypatch):
    root, _, _ = issued
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "visible-rank-must-not-affect-reporting")
    for name in ("Popen", "run", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden_operation)
    for name in ("system", "kill"):
        monkeypatch.setattr(os, name, forbidden_operation)
    saved = reporting.evidence(root)
    assert saved["numerical_cpu"]["accepted"] is True
    assert saved["numerical_cpu"]["passed_cases"] == 34
    assert saved["numerical_cpu"]["seconds"] == 148.833
    assert saved["numerical_cpu"]["helpers"] == 143
    assert saved["numerical_cpu"]["inventory"] == 509
    assert saved["execution_owner"]["accepted"] is False
    assert saved["gpu_progress"] is None and saved["numerical_gpu_acceptance"] is None
    text = "\n".join(reporting.lines(root, lambda value: value))
    assert "54 bounded eight-rank commands" in text and "planned commands, not GPU executions" in text
    assert not reporting.contract.locations(root)[0].exists()


def forbidden_operation(*args, **kwargs):
    raise AssertionError("A read-only reporter attempted process, signal or execution work")


@pytest.mark.parametrize(
    "target",
    (
        "receipt.json",
        "source.tar",
        "tests.xml",
        "cpu.log",
        "tracking/ns_training_path_validation.py",
        "tracking/test_ns_training_path_validation.py",
        "experiment/train.py",
        "experiment/configs/muont_ns_tpu_q5_full_lr_lower_0p5m_seed0.json",
    ),
)
@pytest.mark.parametrize("change", ("missing", "tampered"))
def test_numerical_cpu_refuses_missing_or_tampered_issued_evidence(issued, target, change):
    root, directory, _ = issued
    path = directory / target
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\nchanged evidence\n")
    with pytest.raises((ValueError, OSError)):
        reporting.evidence(root)
    text = "\n".join(reporting.lines(root, lambda value: value))
    assert "evidence refused" in text and "34 passing CPU cases" not in text


def test_numerical_cpu_rejects_rehashed_different_source_under_original_receipt_pin(issued):
    root, directory, receipt = issued
    path = directory / "experiment/train.py"
    path.write_text("# substituted original training path\n")
    receipt["inventory_sha256"]["experiment/train.py"] = digest(path)
    source_archive(directory, receipt["inventory_sha256"])
    receipt["archive_sha256"] = digest(directory / "source.tar")
    record(root / reporting.CPU, receipt)
    with pytest.raises(ValueError):
        reporting.evidence(root)


@pytest.mark.parametrize("failure", ("exit", "xml"))
def test_numerical_cpu_failure_never_becomes_acceptance_even_with_zero_success_text(issued, monkeypatch, failure):
    root, directory, receipt = issued
    if failure == "exit":
        receipt.update(status="needs_attention", exit_code=1, error="Preserved numerical CPU failure")
    else:
        path = directory / "tests.xml"
        tests = ET.parse(path)
        suite = next(tests.getroot().iter("testsuite"))
        suite.set("failures", "1")
        ET.SubElement(suite.find("testcase"), "failure", {"message": "Numerical continuation rejected"})
        tests.write(path)
        receipt["tests_sha256"] = digest(path)
        monkeypatch.setattr(reporting, "TESTS_SHA256", digest(path))
    record(root / reporting.CPU, receipt)
    monkeypatch.setattr(reporting, "CPU_SHA256", digest(root / reporting.CPU))
    with pytest.raises(ValueError):
        reporting.evidence(root)
    text = "\n".join(reporting.lines(root, lambda value: value))
    assert "evidence refused" in text and "34 passing CPU cases" not in text
    if failure == "exit":
        assert receipt["error"] in text


def test_unaccepted_owner_and_completed_saved_files_never_establish_gpu_results(issued):
    root, _, _ = issued
    binding, helpers, state, _, result = reporting.contract.locations(root)
    for path, payload in (
        (binding, {"status": reporting.contract.FROZEN_STATUS, "commands": [["unbound"]] * 54}),
        (helpers.parent / "controller.json", {"status": "acknowledged", "pid": 123, "start_ticks": "456"}),
        (state, {"status": "validation_complete", "exit_code": 0}),
        (
            result,
            {
                "status": "ns_training_path_actual_config_gates_reconstructed_not_production",
                "accepted_cases": ["unbound"],
                "rejected_cases": ["retained_negative_control"],
                "all_cases_accepted": False,
                "gpu_commands": [{"exit_code": 0}] * 54,
            },
        ),
    ):
        record(path, payload)
    saved = reporting.evidence(root)
    assert saved["numerical_cpu"]["accepted"] is True
    assert saved["execution_owner"]["accepted"] is False
    assert len(saved["execution_owner"]["unaccepted_saved_paths"]) == 4
    assert saved["gpu_progress"] is None and saved["numerical_gpu_acceptance"] is None
    assert json.loads(result.read_bytes())["rejected_cases"] == ["retained_negative_control"]


def test_updater_in_memory_summary_retains_cpu_scope_and_manual_milestones(issued):
    root, _, _ = issued
    now = datetime(2026, 11, 19, tzinfo=UTC)
    original = f"# Research\n\nWave12 accepted cloud milestone.\n\n{journal.ANCHOR}\n\nOriginal rejection remains.\n"
    payload = journal.render(root, now)
    detailed = journal.replace_block(original, payload, journal.START, journal.END, journal.ANCHOR)
    summary = journal.research_summary(payload, now)
    assert "Wave12 accepted cloud milestone." in detailed and "Original rejection remains." in detailed
    assert f"### {reporting.TITLE}" in detailed and f"### {reporting.TITLE}" in summary
    assert "34 passing CPU cases" in summary
    assert "54 bounded eight-rank commands" in summary
    assert "No separately accepted training-path execution owner" in summary


def stat_identity(path: Path) -> dict:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
        "links": value.st_nlink,
    }


@pytest.fixture
def owned(issued, monkeypatch):
    root, numerical_directory, numerical = issued
    original = Path(numerical["candidate"])
    candidate = root / "candidates/separate-accepted-owner"
    candidate.mkdir()
    directory = (root / reporting.OWNER_CPU).parent
    directory.mkdir(parents=True)
    for target in (original, candidate, directory):
        for name in ("experiment", reporting.PINNED):
            shutil.copytree(numerical_directory / name, target / name)
        (target / reporting.ALIAS).symlink_to("pinned-sources-materialized", target_is_directory=True)
    shutil.copytree(numerical_directory / "fixtures", directory / "fixtures")
    shutil.copytree(numerical_directory / "tracking", directory / "tracking")
    (candidate / reporting.OWNER_FILE).write_text("# separately accepted owner adapter\n")
    shutil.copy2(candidate / reporting.OWNER_FILE, directory / reporting.OWNER_FILE)
    for name in reporting.OWNER_NEW_HELPERS:
        (directory / "tracking" / name).write_text(f"# issued owner source {name}\n")
    helpers = {path.name: digest(path) for path in (directory / "tracking").iterdir()}
    parent_helpers = (root / reporting.PARENT_CPU).parent / "tracking"
    parent_helpers.mkdir()
    for name in ("ns_training_path_predecessor.py", "prepare_ns_training_path_predecessor.py"):
        shutil.copy2(numerical_directory / "tracking" / name, parent_helpers / name)
    payload = root / "historical/process.pt"
    payload.parent.mkdir()
    payload.write_bytes(b"Historical opaque payload: reporter must not open this file\n")
    sealed = {
        "status": "original_ns_and_failed_queues_sealed_for_distinct_training_path_diagnostic",
        "root": str(root),
        "gpu_execution_authorized": False,
        "production_authorized": False,
        "restored_keeper": {"pid": 1947488, "devices": [[str(index), f"GPU-{index}", "fixture"] for index in range(8)]},
        "file_identity": {"historical/process.pt": stat_identity(payload)},
        "evidence_sha256": {"historical/process.pt": digest(payload)},
    }
    record(root / reporting.SEAL, sealed)
    monkeypatch.setattr(reporting, "SEAL_SHA256", digest(root / reporting.SEAL))
    predecessor = {
        "cpu_receipt": str(root / reporting.PARENT_CPU),
        "cpu_receipt_sha256": reporting.PARENT_SHA256,
        "verifier": str(parent_helpers / "ns_training_path_predecessor.py"),
        "verifier_sha256": helpers["ns_training_path_predecessor.py"],
        "checker": str(parent_helpers / "prepare_ns_training_path_predecessor.py"),
        "checker_sha256": helpers["prepare_ns_training_path_predecessor.py"],
        "seal": str(root / reporting.SEAL),
        "seal_sha256": reporting.SEAL_SHA256,
    }
    data = root / "data/experiment_receipt.json"
    record(data, {"fixture": "accepted data receipt"})
    files = {
        name: value
        for name, value in numerical["inventory_sha256"].items()
        if name.startswith(("experiment/", f"{reporting.PINNED}/"))
    } | {reporting.OWNER_FILE: digest(candidate / reporting.OWNER_FILE)}
    new_helpers = {name: helpers[name] for name in reporting.OWNER_NEW_HELPERS}
    commands = reporting.contract.commands(candidate, root, reporting.contract.locations(root)[3])
    prepared = {
        "status": "ns_training_path_execution_owner_prepared_gpu_unissued",
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(original),
        "parent_cpu_receipt_sha256": reporting.CPU_SHA256,
        "parent_metadata_sha256": {},
        "files_sha256": files,
        "changed_files": [reporting.OWNER_FILE],
        "helpers_sha256": helpers,
        "new_helpers_sha256": new_helpers,
        "data_receipt_sha256": digest(data),
        "predecessor_binding": predecessor,
        "predecessor_seal_identity": stat_identity(root / reporting.SEAL),
        "commands": commands,
        "scope": reporting.OWNER_SCOPE,
    }
    record(candidate / reporting.OWNER_PREPARATION, prepared)
    shutil.copy2(candidate / reporting.OWNER_PREPARATION, directory / reporting.OWNER_PREPARATION)
    shutil.copy2(original / reporting.OWNER_FILE, directory / reporting.PARENT_OWNER_FILE)
    inventory = (
        numerical["inventory_sha256"]
        | files
        | {
            reporting.OWNER_PREPARATION: digest(candidate / reporting.OWNER_PREPARATION),
            reporting.PARENT_OWNER_FILE: digest(original / reporting.OWNER_FILE),
        }
        | {f"tracking/{name}": value for name, value in helpers.items()}
    )
    source_archive(directory, inventory)
    tests = ET.Element("testsuites")
    for name, count in (("test_ns_training_path_gpu", 45), ("test_ns_training_path_execution_source", 10)):
        suite = ET.SubElement(
            tests,
            "testsuite",
            {
                "tests": str(count),
                "failures": "0",
                "errors": "0",
                "skipped": "0",
                "time": "23.735" if count == 45 else "0",
            },
        )
        for index in range(count):
            ET.SubElement(suite, "testcase", {"classname": name, "name": f"test_case_{index}"})
    ET.ElementTree(tests).write(directory / "tests.xml")
    (directory / "cpu.log").write_text("55 owner CPU cases passed; GPU unissued\n")
    cpu = {
        "status": reporting.OWNER_CPU_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_cpu_receipt_sha256": reporting.CPU_SHA256,
        "predecessor_binding": predecessor,
        "predecessor_seal_identity": prepared["predecessor_seal_identity"],
        "preparation_sha256": digest(candidate / reporting.OWNER_PREPARATION),
        "files_sha256": files,
        "helpers": str(directory / "tracking"),
        "helpers_sha256": helpers,
        "new_helpers_sha256": new_helpers,
        "inventory_sha256": inventory,
        "cwd": str(directory),
        "environment": numerical["environment"],
        "cuda_visible_devices": "",
        "exit_code": 0,
        "scope": reporting.OWNER_SCOPE,
        "time": "2026-11-18T23:38:31+00:00",
    }
    command = cpu_command(directory)
    command[command.index("-q") + 1 : command.index("--junitxml")] = [
        str(directory / f"tracking/{name}.py")
        for name in ("test_ns_training_path_gpu", "test_ns_training_path_execution_source")
    ]
    cpu["command"] = command
    for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log")):
        cpu[name] = str(directory / filename)
        cpu[f"{name}_sha256"] = digest(directory / filename)
        monkeypatch.setattr(reporting, f"OWNER_{name.upper()}_SHA256", digest(directory / filename))
    record(root / reporting.OWNER_CPU, cpu)
    monkeypatch.setattr(reporting, "OWNER_CPU_SHA256", digest(root / reporting.OWNER_CPU))
    return root, directory, cpu, prepared, sealed


def test_owner_cpu_acceptance_keeps_numerical_acceptance_separate_and_gpu_unissued(owned, monkeypatch):
    root, _, _, _, _ = owned
    for name in ("Popen", "run", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden_operation)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "visible-rank")
    saved = reporting.evidence(root)
    assert saved["numerical_cpu"]["passed_cases"] == 34
    assert saved["execution_owner"]["accepted"] is True
    assert saved["execution_owner"]["passed_cases"] == 55
    assert saved["execution_owner"]["seconds"] == 23.735
    assert saved["execution_owner"]["helpers"] == 149 and saved["execution_owner"]["inventory"] == 517
    assert saved["gpu_progress"] == {"status": "gpu_unissued", "completed_commands": 0}
    assert saved["numerical_gpu_acceptance"] is None


@pytest.mark.parametrize(
    "target",
    (
        "receipt.json",
        "source.tar",
        "tests.xml",
        "cpu.log",
        "tracking/ns_training_path_gpu_owner.py",
        "tracking/recovery_keeper_manager.sh",
        "experiment/fused_attribution_owner.py",
    ),
)
def test_owner_cpu_changed_evidence_refuses_owner_but_preserves_numerical_cpu(owned, target):
    root, directory, _, _, _ = owned
    path = directory / target
    path.write_bytes(path.read_bytes() + b"\nchanged accepted owner evidence\n")
    saved = reporting.evidence(root)
    assert saved["numerical_cpu"]["accepted"] is True
    assert saved["execution_owner"]["accepted"] is False
    assert saved["execution_owner"]["status"] == "evidence_refused"
    assert saved["gpu_progress"] is None and saved["numerical_gpu_acceptance"] is None


@pytest.mark.parametrize("fault", ("cpu_failure", "changed_candidate", "changed_seal", "rehashed_receipt"))
def test_owner_cpu_binding_faults_never_authorize_progress(owned, monkeypatch, fault):
    root, _, cpu, prepared, _ = owned
    if fault == "cpu_failure":
        cpu.update(status="needs_attention", exit_code=1, error="Preserved owner CPU failure")
        record(root / reporting.OWNER_CPU, cpu)
        monkeypatch.setattr(reporting, "OWNER_CPU_SHA256", digest(root / reporting.OWNER_CPU))
    elif fault == "changed_candidate":
        (Path(prepared["candidate"]) / reporting.OWNER_FILE).write_text("# changed candidate\n")
    elif fault == "changed_seal":
        (root / reporting.SEAL).write_text('{"status":"substituted"}\n')
    else:
        cpu["helpers_sha256"]["ns_training_path_gpu_owner.py"] = "0" * 64
        record(root / reporting.OWNER_CPU, cpu)
    saved = reporting.evidence(root)
    assert saved["numerical_cpu"]["accepted"] is True
    assert saved["execution_owner"]["accepted"] is False and saved["gpu_progress"] is None


@pytest.fixture
def frozen(owned, monkeypatch):
    root, directory, cpu, prepared, sealed = owned
    binding, helpers, _, work, _ = reporting.contract.locations(root)
    shutil.copytree(directory / "tracking", helpers)
    issued_archive = root / reporting.ISSUED_ARCHIVE
    shutil.copy2(directory / "source.tar", issued_archive)
    monkeypatch.setattr(reporting, "ISSUED_ARCHIVE_SHA256", digest(issued_archive))
    candidate = Path(cpu["candidate"])
    numerical = json.loads((root / reporting.CPU).read_bytes())
    parent_files = {
        name: value
        for name, value in numerical["inventory_sha256"].items()
        if name.startswith(("experiment/", f"{reporting.PINNED}/"))
    }
    commands = prepared["commands"]
    plan = {
        "root": str(root),
        "candidate": str(candidate),
        "world": 8,
        "environment": reporting.contract.ENVIRONMENT,
        "commands": commands,
        "trials": [
            spec
            | {
                "config_sha256": digest(candidate / spec["config_name"]),
                "continuous": str(work / f"{spec['name']}_continuous"),
                "resumed": str(work / f"{spec['name']}_resumed"),
                "commands": commands[index * 3 : index * 3 + 3],
            }
            for index, spec in enumerate(reporting.contract.specs())
        ],
        "planned_tokens": 18 * 2 * 24 * 524288,
        "checkpoint_steps": [22, 23, 24],
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "training_sources_sha256": reporting.contract.TRAINING_FILES,
        "numerical_readers_sha256": reporting.contract.NUMERICAL_READERS,
        "scope": reporting.contract.SCOPE,
    }
    paths = {
        root / reporting.OWNER_CPU: reporting.OWNER_CPU_SHA256,
        issued_archive: digest(issued_archive),
        candidate / reporting.OWNER_PREPARATION: cpu["preparation_sha256"],
        root / "data/experiment_receipt.json": prepared["data_receipt_sha256"],
        root / reporting.PARENT_CPU: reporting.PARENT_SHA256,
        root / reporting.SEAL: reporting.SEAL_SHA256,
    }
    for folder, inventory in (
        (candidate, cpu["files_sha256"]),
        (Path(prepared["parent_candidate"]), parent_files),
        (helpers, cpu["helpers_sha256"]),
        (directory, cpu["inventory_sha256"]),
    ):
        paths |= {(folder / name).resolve(): value for name, value in inventory.items()}
    authenticated = {str(path): {"sha256": value, "identity": stat_identity(path)} for path, value in paths.items()}
    authenticated[str(root / "historical/process.pt")] = {
        "sha256": sealed["evidence_sha256"]["historical/process.pt"],
        "identity": sealed["file_identity"]["historical/process.pt"],
    }
    predecessor = cpu["predecessor_binding"]
    frozen = {
        "status": reporting.contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "node": "28.216.17.9",
        "environment": reporting.contract.ENVIRONMENT,
        "controller_environment": cpu["environment"],
        "helpers": str(helpers),
        "helpers_sha256": cpu["helpers_sha256"],
        "files_sha256": cpu["files_sha256"],
        "parent_candidate": prepared["parent_candidate"],
        "parent_files_sha256": parent_files,
        "allowed_transform": reporting.OWNER_FILE,
        "preparation_sha256": cpu["preparation_sha256"],
        "cpu_receipt": str(root / reporting.OWNER_CPU),
        "cpu_receipt_sha256": reporting.OWNER_CPU_SHA256,
        "archive": str(issued_archive),
        "archive_sha256": digest(issued_archive),
        "data_receipt_sha256": prepared["data_receipt_sha256"],
        "allocation_predecessor": sealed,
        "predecessor_acceptance": {
            "cpu_receipt": predecessor["cpu_receipt"],
            "cpu_receipt_sha256": predecessor["cpu_receipt_sha256"],
            "verifier_sha256": predecessor["verifier_sha256"],
            "sealer_sha256": predecessor["checker_sha256"],
            "seal": predecessor["seal"],
            "seal_sha256": predecessor["seal_sha256"],
        },
        "authenticated_files": authenticated,
        "devices": sealed["restored_keeper"]["devices"],
        "plan": plan,
        "commands": commands,
        "controller_command": [
            reporting.contract.PYTHON,
            "-P",
            str(helpers / "ns_training_path_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            "execute",
        ],
        "validator_command": [
            reporting.contract.PYTHON,
            "-P",
            str(helpers / "ns_training_path_validation.py"),
            "--root",
            str(root),
            "--candidate",
            str(candidate),
            "--phase",
            "execute",
        ],
        "validation_timeout": 63000,
        "scope": reporting.OWNER_SCOPE,
    }
    record(binding, frozen)
    monkeypatch.setattr(reporting, "BINDING_SHA256", digest(binding))
    return root, frozen


def test_bound_frozen_owner_is_unlaunched_without_opening_historical_payloads(frozen, monkeypatch):
    root, _ = frozen
    original_open = Path.open

    def bounded_open(path, *args, **kwargs):
        if path.suffix == ".pt":
            raise AssertionError("Reporter tried to read a historical tensor payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", bounded_open)
    saved = reporting.evidence(root)
    assert saved["execution_owner"]["accepted"] is True
    assert saved["gpu_progress"]["status"] == "frozen_unlaunched"
    assert saved["gpu_progress"]["completed_commands"] == 0
    assert saved["numerical_gpu_acceptance"] is None


@pytest.mark.parametrize("fault", ("binding", "helper", "extra_helper", "payload_identity", "data"))
def test_changed_frozen_evidence_withholds_progress_not_prior_cpu_acceptance(frozen, fault):
    root, bound = frozen
    if fault == "binding":
        bound["commands"][0].append("--unbound")
        record(reporting.contract.locations(root)[0], bound)
    elif fault == "helper":
        (Path(bound["helpers"]) / "ns_training_path_gpu_controller.py").write_text("# different source\n")
    elif fault == "extra_helper":
        (Path(bound["helpers"]) / "unbound_import.py").write_text("raise RuntimeError('unbound executable source')\n")
    elif fault == "payload_identity":
        path = root / "historical/process.pt"
        content = path.read_bytes()
        path.unlink()
        path.write_bytes(content)
    else:
        record(root / "data/experiment_receipt.json", {"data": "substituted"})
    saved = reporting.evidence(root)
    assert saved["numerical_cpu"]["accepted"] is True
    assert saved["numerical_gpu_acceptance"] is None
    if fault == "data":
        assert saved["execution_owner"]["accepted"] is False
    else:
        assert saved["execution_owner"]["accepted"] is True
        assert saved["gpu_progress"]["status"] == "evidence_refused"


@pytest.fixture
def running(frozen, monkeypatch):
    root, bound = frozen
    _, helpers, state_path, _, result_path = reporting.contract.locations(root)
    inventory_hash = hashlib.sha256(
        json.dumps(bound["authenticated_files"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    proof = hashlib.sha256(
        json.dumps(
            {
                "binding_sha256": reporting.BINDING_SHA256,
                "cpu_receipt_sha256": reporting.OWNER_CPU_SHA256,
                "inventory_sha256": inventory_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    intent = {
        "status": "acknowledged",
        "binding_sha256": reporting.BINDING_SHA256,
        "command": bound["controller_command"],
        "marker": "original-controller",
        "parent_proof": proof,
        "pid": 101,
        "start_ticks": "501",
        "generation": {
            "pid": 101,
            "start_ticks": "501",
            "parent": 1,
            "group": 101,
            "session": 101,
            "command": bound["controller_command"],
        },
        "time": "2026-11-18T23:42:00+00:00",
    }
    record(helpers.parent / "controller.json", intent)
    monkeypatch.setattr(reporting, "INTENT_SHA256", digest(helpers.parent / "controller.json"))
    (helpers.parent / "controller.log").write_text("saved controller output\n")
    (helpers.parent / "validator.log").write_text("saved validator output\n")
    state = {
        "status": "validating",
        "binding_sha256": reporting.BINDING_SHA256,
        "controller_pid": 101,
        "controller_start_ticks": "501",
        "allocation_predecessor": bound["allocation_predecessor"],
        "spawn_status": "acknowledged",
        "command": bound["validator_command"],
        "child_marker": "original-validator",
        "log": str(helpers.parent / "validator.log"),
        "child_pid": 102,
        "child_start_ticks": "502",
        "child_generation": {
            "pid": 102,
            "start_ticks": "502",
            "parent": 101,
            "group": 102,
            "session": 102,
            "command": bound["validator_command"],
        },
        "keeper_release_attempted": True,
        "keeper_release_complete": True,
        "active_command": bound["commands"][0],
        "time": "2026-11-18T23:42:01+00:00",
    }
    record(state_path, state)
    result = bound["plan"] | {
        "status": "validating",
        "binding_sha256": reporting.BINDING_SHA256,
        "owner": {
            "controller_pid": 101,
            "controller_start_ticks": "501",
            "validator_pid": 102,
            "validator_start_ticks": "502",
        },
        "gpu_commands": [],
        "results": [],
        "time": "2026-11-18T23:42:02+00:00",
    }
    record(result_path, result)
    return root, bound, intent, state, result


def complete_commands(root: Path, bound: dict, result: dict, count: int) -> None:
    work = reporting.contract.locations(root)[3]
    work.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        trial = bound["plan"]["trials"][index // 3]
        mode = ("continuous", "interrupted", "resumed")[index % 3]
        command = bound["commands"][index]
        log = work / f"{trial['name']}_{mode}.log"
        log.write_text("synthetic saved command exited zero; no training executed by tests\n")
        entry = {
            "command": command,
            "marker": f"original-command-{index}",
            "spawn_status": "acknowledged",
            "generation": {
                "pid": 200 + index,
                "start_ticks": str(700 + index),
                "parent": 102,
                "group": 200 + index,
                "session": 200 + index,
                "command": command,
            },
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "marked_descendants_absent": True,
            "generations": {},
            "log": str(log),
            "log_sha256": digest(log),
        }
        record(log.with_suffix(".execution.json"), entry)
        result["gpu_commands"].append(entry | {"trial": trial["name"], "mode": mode, "trajectory": {}})
    record(reporting.contract.locations(root)[4], result)


@pytest.mark.parametrize("count", (0, 2, 54))
def test_saved_running_command_counts_do_not_infer_numerical_acceptance(running, count):
    root, bound, _, _, result = running
    complete_commands(root, bound, result, count)
    saved = reporting.evidence(root)
    assert saved["gpu_progress"]["status"] == "validating"
    assert saved["gpu_progress"]["completed_commands"] == count
    assert saved["gpu_progress"]["last_saved_command_index"] == 1
    assert saved["numerical_gpu_acceptance"] is None


@pytest.mark.parametrize("fault", ("unbound_count", "replaced_log", "changed_sidecar", "changed_owner", "failed_exit"))
def test_saved_command_progress_refuses_altered_receipts_and_log_binding(running, fault):
    root, bound, _, _, result = running
    complete_commands(root, bound, result, 1)
    entry = result["gpu_commands"][0]
    if fault == "unbound_count":
        result["gpu_commands"].append(entry)
    elif fault == "replaced_log":
        Path(entry["log"]).write_text("substituted command log\n")
    elif fault == "changed_sidecar":
        record(Path(entry["log"]).with_suffix(".execution.json"), {"exit_code": 0})
    elif fault == "changed_owner":
        result["owner"]["validator_pid"] = 999
    else:
        entry["exit_code"] = 1
        record(
            Path(entry["log"]).with_suffix(".execution.json"),
            {key: value for key, value in entry.items() if key not in ("trial", "mode", "trajectory")},
        )
    record(reporting.contract.locations(root)[4], result)
    saved = reporting.evidence(root)
    assert saved["execution_owner"]["accepted"] is True
    assert saved["gpu_progress"]["status"] == "evidence_refused"
    assert saved["gpu_progress"]["completed_commands"] is None and saved["numerical_gpu_acceptance"] is None


@pytest.fixture
def completed(running):
    root, bound, intent, state, result = running
    complete_commands(root, bound, result, 54)
    for index, item in enumerate(bound["plan"]["trials"]):
        gates = {name: True for name in reporting.GATES}
        gates["boundary_all_rank_continuation"] = index != 0
        trajectories = {}
        for mode, endpoint in (("continuous", 24), ("interrupted", 22), ("resumed", 24)):
            history = {"metrics_from_000000.jsonl": "a" * 64}
            if mode == "resumed":
                history["metrics_from_000022.jsonl"] = "b" * 64
            trajectories[mode] = {
                "manifest_sha256": "c" * 64,
                "history_sha256": history,
                "updates": endpoint,
                "tokens": endpoint * 524288,
                "fixed_age_installations": 1,
                "checkpoints": {
                    str(step): {
                        "receipt_sha256": "d" * 64,
                        "rank_bytes": [128] * 8,
                        "rank_sha256": {f"rank_{rank}.pt": "e" * 64 for rank in range(8)},
                    }
                    for step in (22, 23, 24)
                    if step <= endpoint
                },
            }
        checks = [
            "diagonal_capture_parity_passed",
            "outer_diagonal_update_parity_passed",
            "output_work_retained_passed",
        ]
        if item["family"] == "null_leftdiag_fused":
            checks.append("null_identity_outer_passed")
        result["results"].append(
            item
            | {
                "trajectories": trajectories,
                "continuation": {
                    "22": {"step": 22, "within_tolerance": index != 0, "ranks": []},
                    "23": {"step": 23, "within_tolerance": False, "ranks": []},
                    "24": {"step": 24, "within_tolerance": True, "ranks": []},
                },
                "loss": {"within_tolerance": True, "comparisons": 24, "failures": []},
                "capture": (
                    {mode: {"updates": 24, "checks": checks} for mode in ("continuous", "resumed")}
                    if item["family"] != "muont"
                    else {}
                ),
                "gates": gates,
                "accepted": index != 0,
                "first_resumed_diagnostic_only": True,
            }
        )
    result.update(
        status=reporting.RESULT_STATUS,
        accepted_cases=[item["name"] for item in result["results"] if item["accepted"]],
        rejected_cases=[item["name"] for item in result["results"] if not item["accepted"]],
        all_cases_accepted=False,
        time="2026-11-18T23:44:00+00:00",
    )
    record(reporting.contract.locations(root)[4], result)
    state.update(
        status="validation_complete",
        result_sha256=digest(reporting.contract.locations(root)[4]),
        accepted_cases=result["accepted_cases"],
        rejected_cases=result["rejected_cases"],
        all_cases_accepted=False,
        exit_code=0,
        cleanup_exit_code=0,
        marked_descendants_absent=True,
        keeper_restore=0,
        restored_keeper={
            "devices": bound["devices"],
            "pid": 999,
            "generation": {
                str(pid): {
                    "pid": pid,
                    "start_ticks": str(9000 + pid),
                    "parent": 1 if pid == 999 else 999,
                    "group": pid,
                    "session": pid,
                    "status": "S",
                    "command": ["keeper"] if pid == 999 else ["keeper-rank", str(pid - 1000)],
                }
                for pid in range(999, 1008)
            },
            "occupied": [[str(1000 + rank), bound["devices"][rank][1]] for rank in range(8)],
        },
        log_sha256=digest(Path(state["log"])),
        time="2026-11-18T23:44:01+00:00",
    )
    record(reporting.contract.locations(root)[2], state)
    return root, bound, intent, state, result


def test_saved_complete_reconstruction_retains_rejections_but_terminal_stays_pending(completed):
    root, _, _, _, result = completed
    saved = reporting.evidence(root)
    assert saved["gpu_progress"]["completed_commands"] == 54
    assert saved["gpu_progress"]["decisions"]["rejected_cases"] == result["rejected_cases"]
    assert saved["gpu_progress"]["decisions"]["all_cases_accepted"] is False
    assert saved["gpu_progress"]["terminal"]["status"] == "not_recorded"
    assert saved["numerical_gpu_acceptance"] is None


@pytest.mark.parametrize("predicate", ("22", "24", "loss"))
def test_contradictory_saved_numerical_acceptance_withholds_decisions(completed, predicate):
    root, _, _, state, result = completed
    case = result["results"][1]
    if predicate == "loss":
        case["loss"]["within_tolerance"] = False
    else:
        case["continuation"][predicate]["within_tolerance"] = False
    record(reporting.contract.locations(root)[4], result)
    state["result_sha256"] = digest(reporting.contract.locations(root)[4])
    record(reporting.contract.locations(root)[2], state)
    saved = reporting.evidence(root)
    assert saved["execution_owner"]["accepted"] is True
    assert saved["gpu_progress"]["status"] == "evidence_refused"
    assert saved["numerical_gpu_acceptance"] is None


@pytest.mark.parametrize("fault", ("trajectory", "checkpoint_rank", "resumed_history", "capture", "null_check"))
def test_incomplete_retained_trial_coverage_withholds_decisions(completed, fault):
    root, _, _, state, result = completed
    case = next(case for case in result["results"] if case["family"] == "null_leftdiag_fused")
    if fault == "trajectory":
        del case["trajectories"]["interrupted"]
    elif fault == "checkpoint_rank":
        del case["trajectories"]["resumed"]["checkpoints"]["24"]["rank_sha256"]["rank_7.pt"]
    elif fault == "resumed_history":
        del case["trajectories"]["resumed"]["history_sha256"]["metrics_from_000022.jsonl"]
    elif fault == "capture":
        del case["capture"]["resumed"]
    else:
        case["capture"]["continuous"]["checks"].remove("null_identity_outer_passed")
    record(reporting.contract.locations(root)[4], result)
    state["result_sha256"] = digest(reporting.contract.locations(root)[4])
    record(reporting.contract.locations(root)[2], state)
    saved = reporting.evidence(root)
    assert saved["execution_owner"]["accepted"] is True
    assert saved["gpu_progress"]["status"] == "evidence_refused"
    assert saved["numerical_gpu_acceptance"] is None


def terminal_records(root, bound, intent, state, result, outcome_status):
    directory = reporting.contract.locations(root)[1].parent
    token = hashlib.sha256(
        json.dumps(
            {
                "binding_sha256": reporting.BINDING_SHA256,
                "controller_pid": 101,
                "controller_start_ticks": "501",
                "controller_marker": intent["marker"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    output = root / "run-preparation" / f"ns-training-path-terminal-{token}.json"
    attempt_path = directory / "terminal-intent.json"
    attempt = {
        "status": "reconstruction_claimed",
        "binding_sha256": reporting.BINDING_SHA256,
        "controller_pid": 101,
        "controller_start_ticks": "501",
        "controller_marker": intent["marker"],
        "output": str(output),
        "verifier_sha256": bound["helpers_sha256"]["ns_training_path_gpu_controller.py"],
        "reconstructor_pid": 888,
        "time": "2026-11-18T23:44:02+00:00",
    }
    record(attempt_path, attempt)
    if outcome_status != "claimed":
        outcome = {
            "status": outcome_status,
            "output": str(output),
            "reconstruction_intent_sha256": digest(attempt_path),
            "time": "2026-11-18T23:44:04+00:00",
        }
        if outcome_status == "needs_attention":
            outcome["error"] = "Preserved one-shot terminal reconstruction failure"
        else:
            paths = (
                reporting.contract.locations(root)[0],
                reporting.contract.locations(root)[2],
                directory / "controller.json",
                directory / "controller.log",
                directory / "validator.log",
                reporting.contract.locations(root)[4],
            )
            terminal = {
                "status": reporting.TERMINAL_STATUS,
                "binding_sha256": reporting.BINDING_SHA256,
                "result_sha256": state["result_sha256"],
                "terminal_time": state["time"],
                "controller_pid_absent": 101,
                "controller_start_ticks": "501",
                "controller_marker": intent["marker"],
                "validator_pid_absent": 102,
                "reconstruction_intent_sha256": digest(attempt_path),
                "keeper": state["restored_keeper"],
                "accepted_cases": result["accepted_cases"],
                "rejected_cases": result["rejected_cases"],
                "all_cases_accepted": False,
                "evidence_sha256": {str(path.relative_to(root)): digest(path) for path in paths},
                "verifier_sha256": bound["helpers_sha256"]["ns_training_path_gpu_controller.py"],
                "predecessor_acceptance": bound["predecessor_acceptance"],
                "production_authorized": False,
                "scope": reporting.OWNER_SCOPE,
                "time": "2026-11-18T23:44:03+00:00",
            }
            record(output, terminal)
            outcome["terminal_sha256"] = digest(output)
        record(directory / "terminal-outcome.json", outcome)
    return output


@pytest.mark.parametrize("outcome_status", ("claimed", "needs_attention", "complete"))
def test_one_shot_terminal_retains_failed_outcome_or_receipt_bound_rejections(completed, outcome_status):
    root, bound, intent, state, result = completed
    terminal_records(root, bound, intent, state, result, outcome_status)
    saved = reporting.evidence(root)
    terminal = saved["gpu_progress"]["terminal"]
    assert terminal["status"] == ("reconstruction_claimed" if outcome_status == "claimed" else outcome_status)
    if outcome_status == "complete":
        assert saved["numerical_gpu_acceptance"]["rejected_cases"] == result["rejected_cases"]
        assert saved["numerical_gpu_acceptance"]["all_cases_accepted"] is False
    else:
        assert saved["numerical_gpu_acceptance"] is None
        if outcome_status == "needs_attention":
            assert terminal["error"] == "Preserved one-shot terminal reconstruction failure"


@pytest.mark.parametrize("missing", ("state", "result"))
@pytest.mark.parametrize("outcome_status", ("claimed", "needs_attention", "complete"))
def test_terminal_attempt_remains_visible_without_state_or_result(running, missing, outcome_status):
    root, bound, intent, state, result = running
    terminal_records(root, bound, intent, state, result, "claimed")
    if outcome_status != "claimed":
        directory = reporting.contract.locations(root)[1].parent
        attempt_path = directory / "terminal-intent.json"
        attempt = json.loads(attempt_path.read_bytes())
        outcome = {
            "status": outcome_status,
            "output": attempt["output"],
            "reconstruction_intent_sha256": digest(attempt_path),
            "time": "2026-11-18T23:44:04+00:00",
        }
        if outcome_status == "needs_attention":
            outcome["error"] = "Original state/result was not ready for terminal reconstruction"
        else:
            record(Path(attempt["output"]), {"status": reporting.TERMINAL_STATUS})
            outcome["terminal_sha256"] = digest(Path(attempt["output"]))
        record(directory / "terminal-outcome.json", outcome)
    reporting.contract.locations(root)[2 if missing == "state" else 4].unlink()
    saved = reporting.evidence(root)
    terminal = saved["gpu_progress"]["terminal"]
    expected_status = {
        "claimed": "reconstruction_claimed",
        "needs_attention": "needs_attention",
        "complete": "evidence_refused",
    }[outcome_status]
    assert terminal["status"] == expected_status
    assert terminal["accepted"] is False
    assert saved["numerical_gpu_acceptance"] is None
    rendered = "\n".join(reporting.lines(root, lambda value: value or "not recorded"))
    assert f"One-shot terminal outcome `{expected_status}`" in rendered
    if outcome_status == "needs_attention":
        assert terminal["error"] == outcome["error"] and outcome["error"] in rendered


def test_terminal_attempt_remains_visible_when_completion_loses_result(completed):
    root, bound, intent, state, result = completed
    terminal_records(root, bound, intent, state, result, "needs_attention")
    reporting.contract.locations(root)[4].unlink()
    saved = reporting.evidence(root)
    assert saved["gpu_progress"]["status"] == "evidence_refused"
    assert saved["gpu_progress"]["completed_commands"] is None
    assert saved["gpu_progress"]["terminal"]["status"] == "needs_attention"
    assert saved["numerical_gpu_acceptance"] is None


@pytest.mark.parametrize("change", ("scheduler_snapshot", "generation", "ownership", "dead", "devices"))
def test_terminal_keeper_uses_issuer_identity_not_whole_observation_equality(completed, change):
    root, bound, intent, state, result = completed
    output = terminal_records(root, bound, intent, state, result, "complete")
    terminal = json.loads(output.read_bytes())
    keeper = terminal["keeper"]
    if change == "scheduler_snapshot":
        keeper["generation"]["1000"]["status"] = "R"
        keeper["occupied"].reverse()
        keeper["time"] = "2026-11-18T23:44:03+00:00"
    elif change == "generation":
        keeper["generation"]["1000"]["start_ticks"] = "substituted-generation"
    elif change == "ownership":
        keeper["generation"]["1000"]["parent"] = 222
    elif change == "dead":
        keeper["generation"]["1000"]["status"] = "Z"
    else:
        keeper["devices"][0][1] = "GPU-substituted"
    record(output, terminal)
    outcome_path = reporting.contract.locations(root)[1].parent / "terminal-outcome.json"
    outcome = json.loads(outcome_path.read_bytes())
    outcome["terminal_sha256"] = digest(output)
    record(outcome_path, outcome)
    saved = reporting.evidence(root)
    assert saved["execution_owner"]["accepted"] is True
    if change == "scheduler_snapshot":
        assert saved["gpu_progress"]["terminal"]["status"] == "complete"
        assert saved["numerical_gpu_acceptance"]["rejected_cases"] == result["rejected_cases"]
    else:
        assert saved["gpu_progress"]["status"] == "evidence_refused"
        assert saved["numerical_gpu_acceptance"] is None


@pytest.fixture
def followed(completed, monkeypatch):
    root, bound, original, owner_state, _ = completed
    cpu_path = root / reporting.FOLLOWUP_CPU
    directory, helpers = cpu_path.parent, cpu_path.parent / "tracking"
    shutil.copytree(Path(bound["helpers"]), helpers)
    for name in ("ns_training_path_followup.py", "test_ns_training_path_followup.py"):
        (helpers / name).write_text(f"# separately accepted waiter fixture {name}\n")
    files = {path.name: digest(path) for path in helpers.iterdir()}
    tests = ET.Element("testsuites")
    suite = ET.SubElement(tests, "testsuite", tests="19", failures="0", errors="0", skipped="0", time="3.4")
    for index in range(19):
        ET.SubElement(suite, "testcase", classname="test_ns_training_path_followup", name=f"case_{index}")
    ET.ElementTree(tests).write(directory / "tests.xml")
    (directory / "cpu.log").write_text("retained passing CPU tests\n")
    archive = directory / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        for name in files:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    cpu = {
        "status": "ns_training_path_read_only_followup_cpu_archive_verified",
        "root": str(root),
        "binding_sha256": reporting.BINDING_SHA256,
        "original_intent_sha256": reporting.INTENT_SHA256,
        "original_archive_sha256": reporting.ISSUED_ARCHIVE_SHA256,
        "environment": reporting.FOLLOWUP_ENVIRONMENT,
        "exit_code": 0,
        "scope": reporting.FOLLOWUP_SCOPE,
        "helpers": str(helpers),
        "helpers_sha256": files,
        "command": reporting._cpu_command(directory, ("test_ns_training_path_followup",)),
        "time": "2026-11-19T00:46:00+00:00",
    }
    for name, path in (("archive", archive), ("tests", directory / "tests.xml"), ("log", directory / "cpu.log")):
        cpu[name], cpu[f"{name}_sha256"] = str(path), digest(path)
    record(cpu_path, cpu)
    monkeypatch.setattr(reporting, "FOLLOWUP_CPU_SHA256", digest(cpu_path))
    work = root / reporting.FOLLOWUP_WORK
    command = [
        reporting.contract.PYTHON,
        "-P",
        str(helpers / "ns_training_path_followup.py"),
        "--root",
        str(root),
        "--cpu",
        str(cpu_path),
        "--phase",
        "execute",
    ]
    intent = {
        "status": "acknowledged",
        "binding_sha256": reporting.BINDING_SHA256,
        "cpu_receipt_sha256": digest(cpu_path),
        "runner_sha256": files["ns_training_path_followup.py"],
        "command": command,
        "pid": 444,
        "start_ticks": "1444",
        "generation": {"pid": 444, "start_ticks": "1444", "group": 444, "session": 444, "command": command},
        "scope": reporting.FOLLOWUP_SCOPE,
        "interval": 900,
    }
    record(work / "intent.json", intent)
    monkeypatch.setattr(reporting, "FOLLOWUP_INTENT_SHA256", digest(work / "intent.json"))
    observation = {
        "status": "waiting_for_original_owners",
        "ready": False,
        "owners": {
            "controller": {"status": "live_exact_generation", "observed": original["generation"]},
            "validator": {"status": "live_exact_generation", "observed": owner_state["child_generation"]},
        },
        "time": "2026-11-19T00:47:38+00:00",
    }
    record(work / "observation-000000.json", observation)
    state = {
        "status": "waiting_for_original_owners",
        "pid": intent["pid"],
        "start_ticks": intent["start_ticks"],
        "binding_sha256": reporting.BINDING_SHA256,
        "cpu_receipt_sha256": digest(cpu_path),
        "runner_sha256": files["ns_training_path_followup.py"],
        "scope": reporting.FOLLOWUP_SCOPE,
        "interval": 900,
        "observations": 1,
        "original_observation": observation,
        "time": "2026-11-19T00:47:39+00:00",
    }
    record(work / "state.json", state)
    return root, bound, state


@pytest.mark.parametrize(
    "fault", ("none", "waiter_failure", "changed_runner", "observation", "wrong_owner", "premature_complete")
)
def test_saved_terminal_waiter_reporting_preserves_scope_and_refuses_changed_evidence(followed, fault):
    root, _, state = followed
    work = root / reporting.FOLLOWUP_WORK
    if fault == "waiter_failure":
        state.update(status="needs_attention", error="retained terminal failure")
        record(work / "state.json", state)
    elif fault == "changed_runner":
        runner = (root / reporting.FOLLOWUP_CPU).parent / "tracking/ns_training_path_followup.py"
        runner.write_text("# changed\n")
    elif fault == "observation":
        record(work / "observation-000000.json", {"status": "substituted observation"})
    elif fault == "wrong_owner":
        state["original_observation"]["owners"]["validator"]["observed"]["start_ticks"] = "substituted"
        record(work / "state.json", state)
        record(work / "observation-000000.json", state["original_observation"])
    elif fault == "premature_complete":
        state.update(status="terminal_reconstructed", terminal_sha256="no accepted terminal")
        record(work / "state.json", state)
    saved = reporting.evidence(root)
    followup = saved["terminal_followup"]
    if fault in ("none", "waiter_failure"):
        assert followup["status"] == state["status"]
        assert followup["saved_original_owners"] == {
            "controller": "live_exact_generation",
            "validator": "live_exact_generation",
        }
        assert followup["passed_cases"] == 19
        if fault == "waiter_failure":
            assert followup["error"] == state["error"]
    else:
        assert followup["status"] == "evidence_refused"
    assert saved["gpu_progress"]["completed_commands"] == 54
    assert saved["gpu_progress"]["terminal"]["status"] == "not_recorded"
    assert saved["execution_owner"]["accepted"] is True
    assert saved["numerical_gpu_acceptance"] is None


@pytest.mark.parametrize("fault", ("none", "child_ancestry", "rejection", "cleanup_bool", "log"))
def test_terminal_waiter_completion_requires_original_terminal_and_exact_child(followed, fault):
    root, bound, state = followed
    work = root / reporting.FOLLOWUP_WORK
    _, helpers, owner_state_path, _, result_path = reporting.contract.locations(root)
    original, owner_state, result = (
        json.loads(path.read_bytes()) for path in (helpers.parent / "controller.json", owner_state_path, result_path)
    )
    output = terminal_records(root, bound, original, owner_state, result, "complete")
    command = [
        reporting.contract.PYTHON,
        "-P",
        str(helpers / "ns_training_path_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(output),
    ]
    log = work / "terminal.log"
    log.write_text("original issued terminal reconstruction passed; rejected cases retained\n")
    state.update(
        status="terminal_reconstructed",
        terminal=str(output),
        terminal_sha256=digest(output),
        terminal_command=command,
        terminal_pid=555,
        terminal_generation={
            "pid": 555,
            "start_ticks": "1555",
            "parent": state["pid"],
            "group": 555,
            "session": 555,
            "command": command,
        },
        terminal_exit_code=0,
        terminal_cleanup_exit_code=0,
        terminal_descendants_absent=True,
        terminal_log=str(log),
        terminal_log_sha256=digest(log),
        terminal_outcome_sha256=digest(helpers.parent / "terminal-outcome.json"),
        **{key: result[key] for key in ("accepted_cases", "rejected_cases", "all_cases_accepted")},
    )
    if fault == "child_ancestry":
        state["terminal_generation"]["parent"] = 999
    elif fault == "rejection":
        state["rejected_cases"] = []
    elif fault == "cleanup_bool":
        state["terminal_cleanup_exit_code"] = False
    elif fault == "log":
        log.write_text("substituted terminal log\n")
    record(work / "state.json", state)
    saved = reporting.evidence(root)
    assert saved["gpu_progress"]["terminal"]["status"] == "complete"
    assert saved["numerical_gpu_acceptance"]["rejected_cases"] == result["rejected_cases"]
    if fault == "none":
        assert saved["terminal_followup"]["status"] == "terminal_reconstructed"
    else:
        assert saved["terminal_followup"]["status"] == "evidence_refused"
