# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete execution archives and the distinct original-terminal handoff."""

import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_gpu_controller as controller
import ns_training_path_lr_gpu_owner as owner
import ns_training_path_lr_gpu_source as source
import ns_training_path_lr_gpu_test_support as support
import pytest
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


@pytest.fixture
def accepted_adapter(tmp_path, monkeypatch):
    previous = source.inherited(ROOT)
    retained = Path(previous["helpers"]).parent
    archived, candidate, root = (tmp_path / name for name in ("parent-archive", "candidate", "root"))
    shutil.copytree(retained, archived, symlinks=True)
    shutil.copytree(retained / "experiment", candidate / "experiment")
    shutil.copytree(retained / "artifacts", candidate / "artifacts", symlinks=True)
    for path in Path(previous["candidate"]).glob("*.json"):
        shutil.copy2(retained / path.name, candidate / path.name)
    (root / "data").mkdir(parents=True)
    write(root / "data/experiment_receipt.json", {"fixture": "independently accepted external data"})
    prepared = read(candidate / source.PREPARATION)
    prepared.update(
        root=str(root),
        candidate=str(candidate),
        data_receipt_sha256=sha256(root / "data/experiment_receipt.json"),
        plan=source.adapter.validation.plan(root, candidate),
    )
    write(candidate / source.PREPARATION, prepared)
    shutil.copy2(candidate / source.PREPARATION, archived / source.PREPARATION)
    previous = previous | {
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(archived / "tracking"),
        "inventory_sha256": previous["inventory_sha256"] | {source.PREPARATION: sha256(candidate / source.PREPARATION)},
    }
    monkeypatch.setattr(source, "inherited", lambda root: previous)
    for name, value in source.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    return root, candidate, archived, previous


def complete_cpu(command, cwd, **kwargs):
    suite = ET.Element("testsuite", tests=str(sum(source.TEST_COUNTS.values())), failures="0", errors="0", skipped="0")
    for module, count in source.TEST_COUNTS.items():
        for index in range(count):
            ET.SubElement(suite, "testcase", classname=module, name=f"external_cpu_case_{index}")
    tree = ET.Element("testsuites")
    tree.append(suite)
    ET.ElementTree(tree).write(Path(cwd) / "tests.xml")
    return subprocess.CompletedProcess(command, 0)


def test_execution_archive_preserves_adapter_and_parent_bytes_without_issuing_gpu(accepted_adapter, monkeypatch):
    root, candidate, _, previous = accepted_adapter
    original = candidate_files(candidate)
    prepared = (candidate / source.PREPARATION).read_bytes()
    monkeypatch.setattr(source.subprocess, "run", complete_cpu)
    cpu = source.cpu(root, root / "execution-cpu")
    assert source.require_cpu(root, root / "execution-cpu/receipt.json") == cpu
    assert cpu["files_sha256"] == original and (candidate / source.PREPARATION).read_bytes() == prepared
    assert all(cpu["helpers_sha256"][name] == digest for name, digest in previous["helpers_sha256"].items())
    assert len(cpu["helpers_sha256"]) == len(previous["helpers_sha256"]) + len(source.NEW_HELPERS)
    assert len(source.adapter.validation.plan(root, candidate)["commands"]) == 162
    assert not any(path.exists() for path in source.adapter.contract.locations(root))
    assert not (root / owner.predecessor.SEAL).exists()


@pytest.mark.parametrize("fault", ("candidate", "parent", "xml", "helper"))
def test_rehashed_source_or_incomplete_execution_cpu_evidence_is_refused(accepted_adapter, monkeypatch, fault):
    root, candidate, archived, _ = accepted_adapter
    monkeypatch.setattr(source.subprocess, "run", complete_cpu)
    cpu = source.cpu(root, root / "execution-cpu")
    receipt = root / "execution-cpu/receipt.json"
    if fault in ("candidate", "parent"):
        path = (candidate if fault == "candidate" else archived) / "experiment/train.py"
        with path.open("ab") as stream:
            stream.write(b"\n# unaccepted numerical source\n")
        cpu["files_sha256"] = candidate_files(candidate)
    elif fault == "xml":
        tree = ET.parse(cpu["tests"])
        suite = tree.getroot().find("testsuite")
        suite.remove(suite.findall("testcase")[-1])
        suite.set("tests", str(sum(source.TEST_COUNTS.values()) - 1))
        tree.write(cpu["tests"])
        cpu["tests_sha256"] = sha256(Path(cpu["tests"]))
    else:
        path = Path(cpu["helpers"]) / "ns_training_path_lr_gpu_owner.py"
        with path.open("ab") as stream:
            stream.write(b"\n# unaccepted owner\n")
        cpu["helpers_sha256"][path.name] = sha256(path)
        cpu["inventory_sha256"][f"tracking/{path.name}"] = sha256(path)
    write(receipt, cpu)
    with pytest.raises(ValueError):
        source.require_cpu(root, receipt)
    assert not source.adapter.contract.locations(root)[0].exists()


def test_execution_import_closure_refuses_added_unaccepted_dependency(accepted_adapter, monkeypatch, tmp_path):
    _, _, _, previous = accepted_adapter
    local = tmp_path / "new-execution"
    local.mkdir()
    for name, path in source.execution_sources(previous).items():
        if name in source.NEW_HELPERS:
            shutil.copy2(path, local / name)
    with (local / source.NEW_HELPERS[0]).open("a") as stream:
        stream.write("\nimport unaccepted_mutable_gpu_reader\n")
    monkeypatch.setattr(source, "__file__", str(local / source.NEW_HELPERS[0]))
    with pytest.raises(ValueError):
        source.execution_sources(previous)


def test_freeze_refuses_live_original_predecessor_before_creating_execution_paths(accepted_adapter, monkeypatch):
    root, candidate, _, _ = accepted_adapter
    monkeypatch.setattr(source.subprocess, "run", complete_cpu)
    cpu = source.cpu(root, root / "execution-cpu")

    def live_original(root, digest):
        raise RuntimeError("Original terminal waiter and validator are still live")

    monkeypatch.setattr(owner, "require_sealed_predecessor", live_original)
    with pytest.raises(RuntimeError):
        controller.freeze(root, root / "execution-cpu/receipt.json", "a" * 64)
    assert cpu["files_sha256"] == candidate_files(candidate)
    assert not any(path.exists() for path in source.adapter.contract.locations(root))
    assert not (root / owner.ARCHIVE).exists()


def test_freeze_and_visible_identity_check_bind_terminal_rejections_without_hidden_calls(accepted_adapter, monkeypatch):
    root, candidate, archived, adapter_cpu = accepted_adapter
    adapter_receipt = root / source.PARENT
    adapter_receipt.parent.mkdir(parents=True)
    write(adapter_receipt, adapter_cpu)
    monkeypatch.setattr(source, "PARENT_SHA256", sha256(adapter_receipt))
    monkeypatch.setattr(source.subprocess, "run", complete_cpu)
    cpu = source.cpu(root, root / "execution-cpu")
    receipt = root / "execution-cpu/receipt.json"
    predecessor_receipt = root / owner.PREDECESSOR_RECEIPT
    predecessor_receipt.parent.mkdir(parents=True)
    write(predecessor_receipt, {"fixture": "accepted predecessor CPU archive"})
    monkeypatch.setattr(owner, "PREDECESSOR_RECEIPT_SHA256", sha256(predecessor_receipt))
    evidence_path = root / "original-terminal.json"
    write(evidence_path, {"accepted_cases": [], "rejected_cases": ["original-fused"], "production_authorized": False})
    payload = root / "original-checkpoint.pt"
    payload.write_bytes(b"retained external checkpoint; never reconstructed by execution freeze")
    original_terminal = owner.authenticated_inventory({str(evidence_path): sha256(evidence_path)})
    original_payload = owner.authenticated_inventory({str(payload): sha256(payload)})
    previous = {
        "status": owner.predecessor.STATUS,
        "root": str(root),
        "restored_keeper": support.keeper(),
        "evidence": {"original-terminal.json": original_terminal[str(evidence_path)]},
        "checkpoint_payloads": {"original-checkpoint.pt": original_payload[str(payload)]},
        "original_decisions": {
            "accepted_cases": [],
            "rejected_cases": ["original-fused"],
            "all_cases_accepted": False,
        },
        "actual_lr_execution_authorized": False,
        "production_authorized": False,
    }
    seal = root / owner.predecessor.SEAL
    write(seal, previous)
    previous = read(seal)
    digest = sha256(seal)
    acceptance = {
        "cpu_receipt": str(predecessor_receipt),
        "cpu_receipt_sha256": sha256(predecessor_receipt),
        "verifier_sha256": owner.PREDECESSOR_VERIFIER_SHA256,
        "sealer_sha256": owner.PREDECESSOR_SEALER_SHA256,
        "seal": str(seal),
        "seal_sha256": digest,
    }
    monkeypatch.setattr(owner, "require_sealed_predecessor", lambda root, seal_sha256: previous)
    monkeypatch.setattr(owner, "predecessor_binding", lambda root, seal_sha256: acceptance)
    monkeypatch.setattr(owner, "require_predecessor_keeper", lambda *args: previous["restored_keeper"])
    monkeypatch.setattr(controller, "keeper_guard", lambda: support.locked(root / "manager.lock"))
    monkeypatch.setattr(
        source,
        "accepted_archive",
        lambda root, spec: {
            "helpers": str(archived / "tracking"),
            "inventory_sha256": {},
            **{name: cpu[name] for name in ("archive", "archive_sha256", "tests", "tests_sha256", "log", "log_sha256")},
        },
    )
    frozen = controller.freeze(root, receipt, digest)
    binding, helpers, _, _, _ = source.adapter.contract.locations(root)
    assert frozen["allocation_predecessor"]["original_decisions"]["rejected_cases"] == ["original-fused"]
    assert frozen["allowed_transform"] == [] and frozen["files_sha256"] == candidate_files(candidate)
    assert frozen["predecessor_acceptance"]["seal_sha256"] == digest
    assert owner.inventory_record(frozen["authenticated_files"], payload) == original_payload[str(payload)]
    monkeypatch.setattr(owner, "__file__", str(helpers / "ns_training_path_lr_gpu_owner.py"))
    # Imported inherited helpers are a subprocess/import boundary; this fixture retains
    # real issued bytes and checks identity acceptance without loading them into this test.
    monkeypatch.setattr(owner, "sys", type(sys)("fixture-sys"))
    owner.sys.modules = {}
    assert owner.require_binding(root) == frozen

    def hidden_check(*args):
        raise AssertionError("Visible identity acceptance cannot call hidden-only verification")

    monkeypatch.setattr(source, "require_cpu", hidden_check)
    monkeypatch.setattr(owner, "require_accepted_adapter", hidden_check)
    monkeypatch.setattr(owner, "require_sealed_predecessor", hidden_check)
    monkeypatch.setattr(owner, "require_parent_proof", lambda *args: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", owner.contract.ENVIRONMENT["CUDA_VISIBLE_DEVICES"])
    assert owner.require_binding(root) == frozen
    with payload.open("ab") as stream:
        stream.write(b"changed after accepted original terminal")
    with pytest.raises(ValueError):
        owner.require_binding(root)
    assert binding.is_file() and not owner.contract.locations(root)[3].exists()
