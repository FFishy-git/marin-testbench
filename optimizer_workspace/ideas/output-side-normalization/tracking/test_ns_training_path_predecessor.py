# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise retained failure evidence, private archive checks and exact resource handoff."""

import copy
import os
import signal
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_predecessor as predecessor
import pytest
from fused_attribution_validation_test_support import keeper, record
from ns_training_path_predecessor_test_support import completed, generation
from selected_lr_release_owner import read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


@pytest.fixture(scope="module")
def original_comparison():
    fixture = Path(__file__).resolve().parents[1] / "fixtures/ns-budget-comparison.json"
    if not fixture.is_file():
        fixture = ROOT / "diagnostics/ns-budget-20261118/comparison.json"
    return read(fixture)


@pytest.mark.parametrize(
    "fault", ("none", "missing_process", "missing_phase", "missing_cross", "eager_changed", "waived")
)
def test_retained_full_comparison_preserves_eager_reproduction_and_every_failed_pair(original_comparison, fault):
    comparison = copy.deepcopy(original_comparison)
    arm = comparison["processes"][0]["analysis"]["arms"][0]
    if fault == "missing_process":
        comparison["processes"].pop()
    elif fault == "missing_phase":
        arm["calls"][0]["phases"].pop()
    elif fault == "missing_cross":
        comparison["cross_process"]["comparisons"].pop()
    elif fault == "eager_changed":
        arm["calls"][0]["phases"][0]["vs_saved_gpu_output"]["exact"] = False
    elif fault == "waived":
        comparison["historical_controls_interpretable"] = True
    if fault == "none":
        projected = predecessor.retained_numerics(comparison)
        assert projected["historical_eager_comparisons"] == 8
        assert projected["historical_compiled_failed_comparisons"] == 40
        assert projected["historical_controls_interpretable"] is False
        pairs = projected["source_pairs"]
        assert len(pairs) == 12
        assert sum(len(item["comparisons"]) for item in pairs) == 144
        assert all(not pair["within_tolerance"] for item in pairs for pair in item["comparisons"])
        assert projected["original_comparison"] == original_comparison
    else:
        with pytest.raises(ValueError):
            predecessor.retained_numerics(comparison)


@pytest.fixture
def original_archives(tmp_path, monkeypatch):
    specs, receipts = [], []
    for index, (_, _, status, count) in enumerate(predecessor.ARCHIVES):
        directory = tmp_path / f"cpu-{index}"
        helper = directory / "tracking" / predecessor.QUEUES[index][2]
        helper.parent.mkdir(parents=True)
        helper.write_text("# Original immutable archive fixture\n")
        (directory / "cpu.log").write_text("Original completed CPU runner\n")
        suites = ET.Element("testsuites")
        suite = ET.SubElement(suites, "testsuite", tests=str(count), failures="0", errors="0", skipped="0")
        for item in range(count):
            ET.SubElement(suite, "testcase", classname="original_archive_tests", name=f"original_case_{item}")
        ET.ElementTree(suites).write(directory / "tests.xml")
        inventory = {f"tracking/{helper.name}": sha256(helper)}
        archive = directory / "source.tar"
        with tarfile.open(archive, "x") as stream:
            stream.add(helper, arcname=f"tracking/{helper.name}", recursive=False)
            if index:
                alias = tarfile.TarInfo("artifacts/external-sources")
                alias.type = tarfile.SYMTYPE
                alias.linkname = "pinned-sources-materialized"
                stream.addfile(alias)
        receipt = {
            "status": status,
            "root": str(tmp_path),
            "helpers": str(helper.parent),
            "helpers_sha256": {helper.name: sha256(helper)},
            "parent_cpu_receipt_sha256": f"original-parent-{index}",
            "exit_code": 0,
            "cuda_visible_devices": "",
            **{
                name: str(path)
                for name, path in (
                    ("archive", archive),
                    ("log", directory / "cpu.log"),
                    ("tests", directory / "tests.xml"),
                )
            },
            **{
                f"{name}_sha256": sha256(path)
                for name, path in (
                    ("archive", archive),
                    ("log", directory / "cpu.log"),
                    ("tests", directory / "tests.xml"),
                )
            },
        }
        if index:
            receipt["inventory_sha256"] = inventory
        else:
            receipt["fixtures_sha256"] = {}
        path = directory / "receipt.json"
        record(path, receipt)
        specs.append((str(path.relative_to(tmp_path)), sha256(path), status, count))
        receipts.append(receipt)
    monkeypatch.setattr(predecessor, "ARCHIVES", tuple(specs))
    return tmp_path, receipts


@pytest.mark.parametrize("fault", ("none", "changed_helper", "truncated_tests", "extra_source"))
def test_original_archive_verifier_rejects_changed_source_and_incomplete_cpu_evidence(original_archives, fault):
    root, _ = original_archives
    spec = predecessor.ARCHIVES[0]
    directory = (root / spec[0]).parent
    if fault == "changed_helper":
        next((directory / "tracking").iterdir()).write_text("# changed source\n")
    elif fault == "truncated_tests":
        tree = ET.parse(directory / "tests.xml")
        suite = tree.getroot().find("testsuite")
        suite.remove(suite.findall("testcase")[-1])
        tree.write(directory / "tests.xml")
    elif fault == "extra_source":
        with tarfile.open(directory / "source.tar", "a") as stream:
            stream.add(directory / "cpu.log", arcname="unaccepted.py", recursive=False)
    if fault == "none":
        original = read(root / spec[0])
        assert predecessor.accepted_archive(root, spec) == original
        assert sha256(root / spec[0]) == spec[1]
    else:
        with pytest.raises(ValueError):
            predecessor.accepted_archive(root, spec)


def test_dependency_inventory_retains_archives_and_canonical_training_source(original_archives, monkeypatch):
    root, receipts = original_archives
    candidate = root / "candidate"
    source = candidate / "materialized/train.py"
    source.parent.mkdir(parents=True)
    source.write_text("# accepted original training source\n")
    (candidate / "experiment").symlink_to("materialized", target_is_directory=True)
    training = {"candidate": str(candidate)}
    cpu_path = predecessor.ARCHIVES[0][0]
    monkeypatch.setattr(predecessor, "TRAINING_CPU", cpu_path)
    monkeypatch.setattr(predecessor, "RECORDS", {cpu_path: predecessor.ARCHIVES[0][1]})
    monkeypatch.setattr(predecessor, "TRAINING_FILES", {"experiment/train.py": sha256(source)})
    frozen = {name: receipts[0][name] for name in ("archive", "archive_sha256", "helpers", "helpers_sha256")} | {
        "cpu_receipt": str(root / cpu_path),
        "cpu_receipt_sha256": predecessor.ARCHIVES[0][1],
    }
    retained = predecessor.dependencies(root, receipts, frozen, training)
    expected = {str(Path(cpu[name]).relative_to(root)) for cpu in receipts for name in ("archive", "tests", "log")}
    expected |= {spec[0] for spec in predecessor.ARCHIVES}
    expected |= {
        str((Path(cpu["helpers"]) / name).relative_to(root)) for cpu in receipts for name in cpu["helpers_sha256"]
    }
    expected.add(str(source.relative_to(root)))
    assert set(retained) == expected
    assert all(retained[name] == sha256(root / name) for name in retained)
    assert "candidate/experiment/train.py" not in retained


@pytest.fixture
def failed_queues(original_archives):
    root, receipts = original_archives
    for index, ((relative, _, filename), cpu) in enumerate(zip(predecessor.QUEUES, receipts, strict=True)):
        argv = [
            predecessor.diagnostic.PYTHON,
            "-P",
            str(Path(cpu["helpers"]) / filename),
            "--root",
            str(root),
            "--cpu-receipt",
            str(root / predecessor.ARCHIVES[index][0]),
            "--phase",
            "execute",
        ]
        own = generation(800000 + index, 1, argv)
        intent = {
            "status": "acknowledged",
            "pid": own["pid"],
            "start_ticks": own["start_ticks"],
            "command": argv,
            "generation": own,
            "marker": f"original-{index}",
            "cpu_receipt": str(root / predecessor.ARCHIVES[index][0]),
            "cpu_receipt_sha256": predecessor.ARCHIVES[index][1],
            "parent_cpu_receipt_sha256": cpu["parent_cpu_receipt_sha256"],
            "runner_sha256": cpu["helpers_sha256"][filename],
            "interval": 900,
        }
        state = {
            "pid": own["pid"],
            "start_ticks": own["start_ticks"],
            **{
                name: intent[name]
                for name in ("cpu_receipt_sha256", "parent_cpu_receipt_sha256", "runner_sha256", "interval")
            },
        }
        if index:
            state.update(status="needs_attention", error=predecessor.ERRORS[relative], phases=[])
        else:
            state.update(
                status="ns_diagnostic_terminal_reconstructed",
                historical_controls_interpretable=False,
                phases=[{"phase": name} for name in ("freeze", "check", "start", "terminal")],
            )
        record(root / relative / "intent.json", intent)
        record(root / relative / "state.json", state)
    return root, receipts


@pytest.mark.parametrize("fault", ("none", "waived_failure", "issued_phase", "changed_ancestry"))
def test_failed_original_queues_preserve_zero_issuance_and_exact_saved_ancestry(failed_queues, fault):
    root, receipts = failed_queues
    path = root / predecessor.LOWER_QUEUE / "state.json"
    state = read(path)
    if fault == "waived_failure":
        state["status"] = "full_model_terminal_reconstructed"
    elif fault == "issued_phase":
        state["phases"] = [{"phase": "freeze"}]
    elif fault == "changed_ancestry":
        state["start_ticks"] = "replacement"
    record(path, state)
    if fault == "none":
        result = predecessor.queues(root, receipts)
        assert len(result) == 3
        assert result[1]["state"]["error"] == predecessor.ERRORS[predecessor.LOWER_QUEUE]
        assert result[2]["state"]["error"] == predecessor.ERRORS[predecessor.ACTUAL_QUEUE]
        assert result[1]["state"]["phases"] == result[2]["state"]["phases"] == []
    else:
        with pytest.raises(ValueError):
            predecessor.queues(root, receipts)


def test_live_original_queue_process_denies_handoff_without_touching_resources(failed_queues):
    root, receipts = failed_queues
    child = subprocess.Popen(
        [sys.executable, "-c", "import signal; signal.pause()"],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        start_new_session=True,
    )
    work = root / predecessor.LOWER_QUEUE
    intent, state = read(work / "intent.json"), read(work / "state.json")
    intent.update(pid=child.pid)
    intent["generation"].update(pid=child.pid, group=child.pid, session=child.pid)
    state["pid"] = child.pid
    record(work / "intent.json", intent)
    record(work / "state.json", state)
    try:
        with pytest.raises(RuntimeError, match="remain present"):
            predecessor.queues(root, receipts)
        assert not (root / predecessor.SEAL).exists()
    finally:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=10)


@pytest.fixture
def completed_chain(failed_queues, original_comparison, monkeypatch):
    root, receipts = failed_queues
    accepted = completed(root, receipts, original_comparison)
    monkeypatch.setattr(predecessor, "OWNER_ARCHIVE", accepted["owner_spec"])
    candidate = root / "candidate"
    hashes = {}
    for name in predecessor.TRAINING_FILES:
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "def newton_schulz_batched():\n    pass\n" if name.endswith("/muon.py") else "# accepted source\n"
        )
        hashes[name] = sha256(target)
    monkeypatch.setattr(predecessor, "TRAINING_FILES", hashes)
    record(root / predecessor.TRAINING_CPU, {"candidate": str(candidate), "files_sha256": hashes})
    monkeypatch.setattr(predecessor, "RECORDS", {name: sha256(root / name) for name in predecessor.RECORDS})
    return root, accepted


@pytest.mark.parametrize("fault", ("none", "terminal_coverage", "worker_cleanup", "unissued_phase", "source_changed"))
def test_complete_evidence_path_authenticates_retained_terminal_and_rejects_invalid_chain(
    completed_chain, monkeypatch, fault
):
    root, original = completed_chain
    terminal_path = root / predecessor.TERMINAL
    state_path = predecessor.diagnostic.locations(root)[2]
    if fault == "terminal_coverage":
        terminal = read(terminal_path)
        del terminal["evidence_sha256"]["diagnostics/ns-budget-20261118/process-0.pt"]
        record(terminal_path, terminal)
        queue_path = root / predecessor.NS_QUEUE / "state.json"
        queue_state = read(queue_path)
        queue_state["terminal_sha256"] = sha256(terminal_path)
        record(queue_path, queue_state)
    elif fault == "worker_cleanup":
        state = read(state_path)
        state["executions"][0]["cleanup_exit_code"] = 1
        record(state_path, state)
    elif fault == "unissued_phase":
        queue_path = root / predecessor.NS_QUEUE / "state.json"
        state = read(queue_path)
        state["phases"][0]["command"][-1] = "unaccepted-receipt.json"
        record(queue_path, state)
    elif fault == "source_changed":
        cpu = read(root / predecessor.TRAINING_CPU)
        (Path(cpu["candidate"]) / "experiment/train.py").write_text("# changed original trainer\n")
    monkeypatch.setattr(predecessor, "RECORDS", {name: sha256(root / name) for name in predecessor.RECORDS})
    if fault != "none":
        with pytest.raises(ValueError):
            predecessor.evidence(root)
        assert not (root / predecessor.SEAL).exists()
        return
    saved = predecessor.evidence(root)
    assert saved["restored_keeper"] == original["restored_keeper"]
    assert saved["numerical_evidence"]["historical_controls_interpretable"] is False
    assert saved["numerical_evidence"]["all_source_pairs_within_tolerance"] is False
    assert set(saved["file_identity"]) == set(saved["evidence_sha256"])
    assert set(original["terminal"]["evidence_sha256"]).issubset(saved["evidence_sha256"])
    assert saved["dependency_sha256"]
    assert not (root / predecessor.SEAL).exists()


def test_full_evidence_seal_roundtrip_preserves_negative_decisions_and_original_files(completed_chain, monkeypatch):
    root, original = completed_chain
    monkeypatch.setattr(predecessor.handoff, "keeper_observation", lambda: original["restored_keeper"])

    @contextmanager
    def guard():
        yield

    monkeypatch.setattr(predecessor, "keeper_guard", guard)
    sealed = predecessor.seal(root)
    checked = predecessor.require_seal(root, sha256(root / predecessor.SEAL))
    assert checked == sealed
    assert all(sha256(root / name) == digest for name, digest in checked["evidence_sha256"].items())
    assert checked["queues"][1]["state"]["phases"] == checked["queues"][2]["state"]["phases"] == []
    assert checked["gpu_execution_authorized"] is checked["production_authorized"] is False
    assert predecessor.live(root, sha256(root / predecessor.SEAL)) == original["restored_keeper"]


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "retained_output_changed",
        "dependency_changed",
        "missing_identity",
        "keeper_replaced",
        "decision_changed",
        "concurrent_seal",
        "changed_during_keeper_check",
        "changed_during_owner_check",
        "failed_write",
    ),
)
def test_one_shot_seal_preserves_original_outputs_and_denies_changed_resource_or_decision(
    tmp_path, monkeypatch, original_comparison, fault
):
    (tmp_path / "run-preparation").mkdir()
    original = tmp_path / "retained-output.pt"
    original.write_bytes(b"immutable retained tensor payload")
    pinned = tmp_path / "original-terminal.json"
    record(pinned, {"status": "original independent terminal"})
    dependency = tmp_path / "accepted-helper.py"
    dependency.write_text("# accepted helper dependency\n")
    monkeypatch.setattr(predecessor, "RECORDS", {pinned.name: sha256(pinned)})
    saved = {
        "status": predecessor.STATUS,
        "root": str(tmp_path),
        "scope": predecessor.SCOPE,
        "evidence_sha256": {name: sha256(tmp_path / name) for name in (pinned.name, original.name, dependency.name)},
        "file_identity": {
            name: predecessor.file_identity(tmp_path / name) for name in (pinned.name, original.name, dependency.name)
        },
        "dependency_sha256": {dependency.name: sha256(dependency)},
        "numerical_evidence": predecessor.retained_numerics(original_comparison),
        "restored_keeper": keeper(),
        "gpu_execution_authorized": False,
        "production_authorized": False,
    }
    monkeypatch.setattr(predecessor, "evidence", lambda root: saved)
    monkeypatch.setattr(predecessor, "absent_owners", lambda *args: None)
    observed = copy.deepcopy(saved["restored_keeper"])
    if fault == "keeper_replaced":
        observed["generation"]["100"]["start_ticks"] = "replacement-keeper"

    def observation():
        if fault == "changed_during_keeper_check":
            dependency.write_text("# changed during keeper check\n")
        return observed

    monkeypatch.setattr(predecessor.handoff, "keeper_observation", observation)

    @contextmanager
    def guard():
        if fault == "concurrent_seal":
            record(tmp_path / predecessor.SEAL, {"status": "first caller's immutable seal"})
        yield

    monkeypatch.setattr(predecessor, "keeper_guard", guard)
    if fault == "concurrent_seal":
        with pytest.raises(FileExistsError):
            predecessor.seal(tmp_path)
        assert read(tmp_path / predecessor.SEAL)["status"] == "first caller's immutable seal"
        return
    if fault == "changed_during_keeper_check":
        with pytest.raises(ValueError):
            predecessor.seal(tmp_path)
        assert not (tmp_path / predecessor.SEAL).exists()
        return
    if fault == "failed_write":

        def failed_fsync(descriptor):
            raise OSError("injected disk write failure")

        monkeypatch.setattr(predecessor.os, "fsync", failed_fsync)
        with pytest.raises(OSError):
            predecessor.seal(tmp_path)
        assert not (tmp_path / predecessor.SEAL).exists()
        assert not list((tmp_path / "run-preparation").glob(".ns-training-path-seal-*"))
        return
    if fault == "keeper_replaced":
        with pytest.raises(RuntimeError):
            predecessor.seal(tmp_path)
        assert not (tmp_path / predecessor.SEAL).exists()
        return
    sealed = predecessor.seal(tmp_path)
    seal_path = tmp_path / predecessor.SEAL
    digest = sha256(seal_path)
    if fault == "retained_output_changed":
        original.write_bytes(b"modified retained tensor payload")
    elif fault == "dependency_changed":
        dependency.write_text("# changed accepted helper\n")
    elif fault == "missing_identity":
        del sealed["file_identity"][dependency.name]
        record(seal_path, sealed)
        digest = sha256(seal_path)
    elif fault == "decision_changed":
        sealed["numerical_evidence"]["historical_controls_interpretable"] = True
        record(seal_path, sealed)
        digest = sha256(seal_path)
    elif fault == "changed_during_owner_check":
        monkeypatch.setattr(
            predecessor,
            "absent_owners",
            lambda *args: dependency.write_text("# changed during owner check\n"),
        )
    if fault == "none":
        before = original.read_bytes()
        checked = predecessor.require_seal(tmp_path, digest)
        assert checked["numerical_evidence"]["historical_controls_interpretable"] is False
        assert checked["gpu_execution_authorized"] is checked["production_authorized"] is False
        assert predecessor.live(tmp_path, digest) == observed
        assert original.read_bytes() == before
        with pytest.raises(FileExistsError):
            predecessor.seal(tmp_path)
    else:
        with pytest.raises(ValueError):
            predecessor.require_seal(tmp_path, digest)
