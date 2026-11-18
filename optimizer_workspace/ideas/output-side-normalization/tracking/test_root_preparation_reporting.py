# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate saved diagnostic reporting with CUDA hidden and immutable execution untouched."""

import json
import shutil
from pathlib import Path

import pytest
import root_preparation_reporting as reporting
from root_preparation_controller import CPU_STATUS, cpu_command
from selected_lr_release_owner import read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
ACCEPTED = ROOT / "run-preparation/root-executor-cpu/stable-source-20261118T1313"


@pytest.mark.parametrize(
    "fault", ("none", "boolean", "log", "source", "xml", "xml_coverage", "fixture", "archive", "parent")
)
def test_saved_executor_reporting_denies_substituted_evidence_without_mutating_original(tmp_path, fault):
    out = tmp_path / "executor"
    shutil.copytree(ACCEPTED, out)
    path = out / "receipt.json"
    receipt = read(path)
    original_hash = sha256(ACCEPTED / "receipt.json")
    receipt.update(helpers=str(out / "tracking"), cwd=str(out), command=cpu_command(out))
    for name, filename in (("log", "controller.log"), ("tests", "tests.xml"), ("archive", "source.tar")):
        receipt[name] = str(out / filename)
    if fault == "boolean":
        receipt["exit_code"] = False
    elif fault == "log":
        (out / "controller.log").write_text("changed\n")
    elif fault == "source":
        (out / "tracking/root_preparation_probe.py").write_text("# changed\n")
    elif fault == "xml":
        (out / "tests.xml").write_text("<invalid")
    elif fault == "xml_coverage":
        tests = out / "tests.xml"
        tests.write_text(tests.read_text().replace("test_root_preparation_controller", "missing_controller"))
        receipt["tests_sha256"] = sha256(tests)
    elif fault == "fixture":
        receipt["fixtures_sha256"].pop(next(iter(receipt["fixtures_sha256"])))
    elif fault == "archive":
        (out / "source.tar").write_bytes(b"changed")
    elif fault == "parent":
        receipt["candidate_cpu_receipt_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt))
    if fault == "none":
        value, counts = reporting.cpu_evidence(ROOT, path)
        assert value["status"] == CPU_STATUS and counts == {"tests": 133, "failures": 0, "errors": 0, "skipped": 0}
    else:
        with pytest.raises((ValueError, KeyError)):
            reporting.cpu_evidence(ROOT, path)
    assert sha256(ACCEPTED / "receipt.json") == original_hash


@pytest.mark.parametrize("name", ("hardened-20261118T1305", "fixture-closure-20261118T1306"))
def test_reporting_preserves_failed_archive_receipts_without_claiming_acceptance(name):
    receipt, counts = reporting.cpu_evidence(ROOT, ROOT / f"run-preparation/root-executor-cpu/{name}/receipt.json")
    assert receipt["status"] == "needs_attention"
    if name.startswith("hardened"):
        assert counts["errors"] == 15 and receipt["exit_code"] == 1
    else:
        assert counts["errors"] == 0 and receipt["exit_code"] == 0
        assert receipt["error"] == "Executor source changed during CPU verification"


def test_no_saved_receipts_produce_no_live_claims(tmp_path):
    assert reporting.executor_lines(tmp_path, str) == []


@pytest.mark.parametrize("fault", ("none", "coordinates", "command", "log_alias", "boolean", "cold_log_alias", "owner"))
def test_saved_gpu_reporting_requires_bound_command_coordinates_logs_and_owner(tmp_path, monkeypatch, fault):
    path = reporting.locations(ROOT)[4]
    receipt = read(path)
    receipt["gpu_commands"] = receipt["gpu_commands"][:1]
    assert len(receipt["gpu_commands"]) == 1 and len(receipt["cold_probes"]) == 2
    entry = receipt["gpu_commands"][0]
    if fault == "coordinates":
        entry["protocol"] = "trace"
    elif fault == "command":
        entry["command"] = receipt["commands"][1]
    elif fault == "boolean":
        entry["exit_code"] = False
    elif fault == "owner":
        receipt["owner"]["controller_start_ticks"] = "different-generation"
    elif fault in ("log_alias", "cold_log_alias"):
        selected = entry if fault == "log_alias" else receipt["cold_probes"][0]
        alias = tmp_path / "same-bytes.log"
        shutil.copy2(selected["log"], alias)
        selected["log"] = str(alias)
        assert sha256(alias) == selected["log_sha256"]
    substituted = tmp_path / "gpu.json"
    substituted.write_text(json.dumps(receipt))
    snapshot_hash = sha256(substituted)

    def read_snapshot(source):
        return read(substituted if source == path else source)

    monkeypatch.setattr(reporting, "read", read_snapshot)
    if fault == "none":
        frozen, intent, state, gpu = reporting.execution_evidence(ROOT)
        assert gpu["gpu_commands"][0]["command"] == frozen["commands"][0]
        assert gpu["owner"]["controller_pid"] == state["controller_pid"] == intent["pid"]
    else:
        with pytest.raises(ValueError):
            reporting.execution_evidence(ROOT)
    assert sha256(substituted) == snapshot_hash
