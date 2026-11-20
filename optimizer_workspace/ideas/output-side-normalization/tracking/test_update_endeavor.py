# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise durable ledger updates without changing manual research prose."""

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import ns_training_path_exit_followup_reporting as exit_followup_report
import ns_training_path_exit_reporting as exit_report
import ns_training_path_query_serialized_followup_reporting as query_serialized_followup_report
import ns_training_path_query_serialized_reporting as query_serialized_report
import ns_training_path_recheck_exit_followup_reporting as recheck_exit_followup_report
import ns_training_path_recheck_exit_reporting as recheck_exit_report
import ns_training_path_recovery_reporting as recovery_report
import ns_training_path_scan_exit_followup_reporting as scan_exit_followup_report
import ns_training_path_scan_exit_reporting as scan_exit_report
import ns_training_path_serialized_followup_reporting as serialized_followup_report
import ns_training_path_serialized_reporting as serialized_report
import ns_training_path_transition_followup_reporting as transition_followup_report
import ns_training_path_transition_reporting as transition_report
import prepare_fused_attribution_validation as attribution_validation_preparation
import prepare_root_preparation_diagnostic as root_preparation
import pytest
import root_preparation_contract as root_contract
import saved_ns_gpu_owner as saved_ns_owner
from analyze_wave7_snapshot import analyze_capture
from fused_attribution_validation import failed_profiler_evidence
from fused_attribution_validation import plan as attribution_validation_plan
from fused_attribution_validation_contract import PREPARATION as ATTRIBUTION_VALIDATION_PREPARATION
from fused_attribution_validation_contract import SCHEDULE as ATTRIBUTION_VALIDATION_SCHEDULE
from fused_attribution_validation_contract import controller_command as attribution_validation_controller_command
from fused_attribution_validation_contract import locations as attribution_validation_locations
from fused_attribution_validation_contract import trial_specs as attribution_validation_trial_specs
from fused_attribution_validation_contract import validator_command as attribution_validation_validator_command
from fused_attribution_validation_test_support import record, write_failed_profiler
from fused_continuation_contract import CPU_STATUS as CONTINUATION_CPU_STATUS
from fused_continuation_contract import FROZEN_STATUS as CONTINUATION_FROZEN_STATUS
from fused_continuation_contract import PREPARATION as CONTINUATION_PREPARATION
from fused_continuation_contract import PREPARED_STATUS as CONTINUATION_PREPARED_STATUS
from fused_continuation_contract import controller_command as continuation_controller_command
from fused_continuation_contract import locations as continuation_locations
from fused_continuation_contract import plan as continuation_plan
from fused_continuation_contract import validator_command as continuation_validator_command
from fused_continuation_controller import CONTROLLER_CPU_STATUS as CONTINUATION_CONTROLLER_CPU_STATUS
from fused_continuation_controller import cpu_command as continuation_controller_cpu_command
from fused_continuation_controller import fixture_names as continuation_controller_fixture_names
from fused_continuation_controller import inventory as continuation_inventory
from fused_continuation_test_support import ISSUED_CANDIDATE
from prepare_fused_attribution import PREPARATION as ATTRIBUTION_PREPARATION
from prepare_fused_attribution_validation import CHANGES as ATTRIBUTION_VALIDATION_CHANGES
from prepare_fused_attribution_validation import COPIES as ATTRIBUTION_VALIDATION_COPIES
from prepare_fused_attribution_validation import EXPANSION_POLICY as ATTRIBUTION_VALIDATION_EXPANSION_POLICY
from prepare_fused_attribution_validation import transformed_train as attribution_validation_transformed_train
from prepare_owned_output_handoff import CHANGES as HANDOFF_CHANGES
from prepare_owned_output_handoff import PREPARATION as HANDOFF_PREPARATION
from prepare_owned_output_validation import cpu_command as owned_output_cpu_command
from selected_lr_stage_profile_contract import FROZEN_STATUS as PROFILE_FROZEN_STATUS
from selected_lr_stage_profile_contract import PREPARATION as PROFILE_PREPARATION
from selected_lr_stage_profile_contract import controller_command as profile_controller_command
from selected_lr_stage_profile_contract import locations as profile_locations
from selected_lr_stage_profile_contract import validator_command as profile_validator_command
from selected_lr_stage_profile_validation import plan as profile_validation_plan
from selected_lr_validation import ARMS as SELECTED_LR_ARMS
from selected_lr_validation import PREPARATION as SELECTED_LR_VALIDATION_PREPARATION
from selected_lr_validation import controller_command as selected_lr_controller_command
from selected_lr_validation import locations as selected_lr_locations
from selected_lr_validation import plan as selected_lr_validation_plan
from selected_lr_validation import validator_command as selected_lr_validator_command
from update_endeavor import (
    ANCHOR,
    END,
    START,
    SUMMARY_ANCHOR,
    SUMMARY_END,
    SUMMARY_START,
    boundary_queue_lines,
    fused_attribution_lines,
    fused_attribution_validation_lines,
    fused_continuation_controller_lines,
    fused_continuation_lines,
    owned_output_handoff_cpu_lines,
    production_lines,
    research_summary,
    root_preparation_cpu_lines,
    saved_ns_replay_cpu_lines,
    selected_lr_production_cpu_lines,
    selected_lr_release_lines,
    selected_lr_stage_profile_lines,
    selected_lr_stage_profile_reader_lines,
    selected_lr_validation_lines,
    tuning_result_lines,
    update,
    upstream_full_model_validation_lines,
    upstream_gpu_lines,
    without_digests,
)
from upstream_production import cpu_command
from wave7_release import ARMS, candidate_inventory


def continuation_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "source",
        "log",
        "archive",
        "rehashed_archive",
        "tests",
        "bool_exit",
        "failed",
        "legacy_failed",
        "legacy_zero",
        "schedule",
        "predecessor",
    ),
)
def test_root_diagnostic_ledger_preserves_attempts_and_denies_changed_evidence(tmp_path, fault):
    root = tmp_path / "root"
    previous, candidate = root / "candidates/parent", root / "candidates/root-preparation-fixture"
    (previous / "experiment").mkdir(parents=True)
    (previous / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    config = previous / root_contract.CONFIG
    config.parent.mkdir(parents=True)
    shutil.copy2(ISSUED_CANDIDATE / root_contract.CONFIG, config)
    (previous / "experiment/optimizer.py").write_text("# unchanged numerical source\n")
    shutil.copytree(previous, candidate, symlinks=True)
    for name in root_preparation.CHANGES:
        source = candidate / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# diagnostic fixture {name}\n")
    binding, data = continuation_locations(root)[0], root / "data/experiment_receipt.json"
    record(binding, {"fixture": "terminal parent"})
    record(data, {"fixture": "unchanged data"})
    observation = root / root_preparation.REPLAY_OBSERVATION
    shutil.copy2(
        ISSUED_CANDIDATE.parents[1] / root_preparation.REPLAY_OBSERVATION,
        observation,
    )
    output = root / "run-preparation/root-preparation-cpu/fixture"
    helpers = output / "tracking"
    helpers.mkdir(parents=True)
    for name in (
        "prepare_root_preparation_diagnostic.py",
        "root_preparation_contract.py",
        "root_preparation_checked.py",
        "prepare_restored_roots.py",
        "root_diagnostic_pytest_guard.py",
    ):
        (helpers / name).write_text(f"# archived fixture {name}\n")
    digest = continuation_digest
    prepared = {
        "status": root_contract.PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_binding_sha256": digest(binding),
        "data_receipt_sha256": digest(data),
        "parent_files_sha256": candidate_inventory(previous),
        "files_sha256": candidate_inventory(candidate),
        "parent_metadata_sha256": {},
        "changed_files": sorted(root_preparation.CHANGES),
        "allocation_predecessor": {"evidence_sha256": {root_preparation.REPLAY_OBSERVATION: digest(observation)}},
        "helpers_sha256": {source.name: digest(source) for source in helpers.iterdir()},
        "plan": root_contract.plan(candidate, root),
    }
    if fault == "schedule":
        prepared["plan"]["commands"][0].append("--prepare-restored-roots")
    record(candidate / root_contract.PREPARATION, prepared)
    log, archive, tests = output / "candidate.log", output / "source.tar", output / "tests.xml"
    log.write_text("7 passed\n")
    failed = fault in ("failed", "legacy_failed")
    tests.write_text(
        f'<testsuites><testsuite tests="7" failures="{int(failed)}" errors="0" skipped="0"/>' "</testsuites>\n"
    )
    sources = (
        {name: candidate / name for name in prepared["files_sha256"]}
        | {root_contract.PREPARATION: candidate / root_contract.PREPARATION}
        | {f"tracking/{name}": helpers / name for name in prepared["helpers_sha256"]}
    )
    with tarfile.open(archive, "x") as bundle:
        for name, source in sources.items():
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    command = root_preparation.cpu_command(candidate)
    if fault in ("legacy_failed", "legacy_zero"):
        position = command.index("--rootdir")
        del command[position : position + 2]
    receipt = {
        "status": "needs_attention" if failed else root_contract.CPU_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "cwd": str(candidate),
        "command": [*command, "--junitxml", str(tests)],
        "exit_code": 1 if failed else 0,
        "cuda_visible_devices": "",
        "files_sha256": prepared["files_sha256"],
        "preparation_sha256": digest(candidate / root_contract.PREPARATION),
        "helpers": str(helpers),
        "helpers_sha256": prepared["helpers_sha256"],
        **{name: str(source) for name, source in (("log", log), ("tests", tests), ("archive", archive))},
        **{f"{name}_sha256": digest(source) for name, source in (("log", log), ("tests", tests), ("archive", archive))},
    }
    if fault == "source":
        (candidate / "experiment/optimizer.py").write_text("# changed optimizer\n")
    elif fault == "log":
        log.write_text("substituted log\n")
    elif fault == "archive":
        archive.write_bytes(b"substituted archive")
    elif fault == "rehashed_archive":
        replacement = output / "replacement.tar"
        with tarfile.open(archive) as original, tarfile.open(replacement, "x") as bundle:
            for member in original.getmembers():
                if member.isfile():
                    if member.name == "tracking/root_preparation_checked.py":
                        bundle.add(helpers / "prepare_restored_roots.py", arcname=member.name, recursive=False)
                    else:
                        with original.extractfile(member) as stream:
                            bundle.addfile(member, stream)
                else:
                    bundle.addfile(member)
        replacement.replace(archive)
        receipt["archive_sha256"] = digest(archive)
    elif fault == "tests":
        tests.write_text("<testsuites/>\n")
    elif fault == "bool_exit":
        receipt["exit_code"] = False
    elif fault == "predecessor":
        observation.write_text("{}")
    record(output / "receipt.json", receipt)
    document = tmp_path / "ENDEAVOR.md"
    manual = "# Endeavor\n\nWave 12 verified; root diagnostic GPU-unissued.\n\n"
    document.write_text(manual + ANCHOR + "\n")
    update(document, root, datetime(2026, 11, 18, tzinfo=UTC))
    text = document.read_text()
    assert text.startswith(manual) and text.endswith(ANCHOR + "\n")
    current = "\n".join(root_preparation_cpu_lines(root))
    accepted = "Source/archive-bound root diagnostic CPU suite exited zero"
    if fault == "none":
        assert accepted in current and accepted in text
        assert "7 tests passed" in current
    elif failed:
        assert accepted not in current and accepted not in text
        assert "Preserved CPU failure" in current and "Original source/log/XML/archive verified" in current
    else:
        assert accepted not in current and accepted not in text
        assert "Root CPU acceptance and GPU/result claims withheld" in current
    assert not root_contract.locations(root)[0].exists()
    assert not root_contract.locations(root)[1].exists()
    assert not root_contract.locations(root)[2].exists()
    assert not root_contract.locations(root)[3].exists()


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "source",
        "log",
        "archive",
        "fixture",
        "inherited",
        "command",
        "guard",
        "bool_exit",
        "failed",
        "contradiction",
        "alias",
        "unrelated",
    ),
)
def test_saved_ns_cpu_ledger_requires_archive_and_preserves_manual_results(tmp_path, fault):
    root = tmp_path / "root"
    output = root / "run-preparation/saved-ns-gpu-adapter-cpu/fixture"
    helpers = output / "tracking"
    helpers.mkdir(parents=True)
    names = {
        "saved_ns_gpu_worker.py",
        "saved_ns_gpu_owner.py",
        "replay_saved_ns.py",
        "analyze_fused_continuation_stages.py",
        "diagnose_continuation.py",
        "diagnose_fused_diagonal.py",
        "benchmark_diagonal.py",
        "saved_ns_gpu_adapter.py",
        "replay_pytest_guard.py",
        "recovery_keeper_manager.sh",
        "test_saved_ns_gpu_owner.py",
        "test_saved_ns_gpu_adapter.py",
        "test_saved_ns_gpu_worker.py",
        "test_replay_saved_ns.py",
        "inherited_owner.py",
    }
    for name in names:
        (helpers / name).write_text(f"# archived fixture {name}\n")
    digest = continuation_digest
    muon_name = "artifacts/external-sources/Toner-fixture/src/toner/fast/muon.py"
    muon = output / muon_name
    muon.parent.mkdir(parents=True)
    muon.write_text("# executed pinned NS fixture\n")
    manifest_name = "experiment/upstream_manifest.json"
    record(
        output / manifest_name,
        {"revision": "fixture", "files": {"src/toner/fast/muon.py": digest(muon)}},
    )
    hashes = {name: digest(helpers / name) for name in names}
    fixtures = {name: digest(output / name) for name in (manifest_name, muon_name)}
    record(
        continuation_locations(root)[0],
        {
            "helpers_sha256": {"inherited_owner.py": hashes["inherited_owner.py"]},
            "files_sha256": {
                manifest_name: fixtures[manifest_name],
                muon_name.replace("external-sources", "pinned-sources-materialized"): fixtures[muon_name],
            },
        },
    )
    log, archive = output / "cpu.log", output / "source.tar"
    log.write_text("7 passed\n")
    with tarfile.open(archive, "w") as bundle:
        for name in hashes:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in fixtures:
            bundle.add(output / name, arcname=name, recursive=False)
    receipt = {
        "status": saved_ns_owner.CPU_STATUS,
        "root": str(root),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixtures_sha256": fixtures,
        "command": saved_ns_owner.cpu_command(helpers),
        "cwd": str(output),
        "exit_code": 0,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": digest(log),
        "archive": str(archive),
        "archive_sha256": digest(archive),
    }
    if fault == "source":
        (helpers / "saved_ns_gpu_worker.py").write_text("# substituted worker\n")
    elif fault == "log":
        log.write_text("1 failed\n")
    elif fault == "archive":
        archive.write_bytes(b"substituted archive")
    elif fault == "fixture":
        muon.write_text("# substituted NS\n")
    elif fault == "inherited":
        hashes["inherited_owner.py"] = "0" * 64
    elif fault == "command":
        receipt["command"] = receipt["command"][:-1]
    elif fault == "guard":
        hashes.pop("replay_pytest_guard.py")
    elif fault == "bool_exit":
        receipt["exit_code"] = False
    elif fault == "failed":
        receipt.update(status="needs_attention", exit_code=1, error="preserved fixture failure")
    elif fault == "contradiction":
        receipt["exit_code"] = 1
    elif fault == "alias":
        alias = output / "helpers-alias"
        alias.symlink_to(helpers, target_is_directory=True)
        receipt["helpers"] = str(alias)
    elif fault == "unrelated":
        receipt["root"] = str(tmp_path / "other-root")
    record(output / "receipt.json", receipt)
    document = tmp_path / "ENDEAVOR.md"
    manual = "# Endeavor\n\nWave 12 verified; saved-NS replay unlaunched.\n\n"
    document.write_text(manual + ANCHOR + "\n")
    update(document, root, datetime(2026, 11, 18, tzinfo=UTC))
    text = document.read_text()
    assert text.startswith(manual) and text.endswith(ANCHOR + "\n")
    current = "\n".join(saved_ns_replay_cpu_lines(root))
    accepted = "Source/archive-bound replay CPU suite exited zero"
    if fault == "none":
        assert accepted in current and accepted in text
        assert "not a fresh process/device observation" in current
        assert "separately issued replay owner" in current
    elif fault == "failed":
        assert accepted not in current and accepted not in text
        assert "Preserved CPU failure" in current
        assert "preserved fixture failure" in current
    else:
        assert accepted not in current and accepted not in text
        assert "Replay CPU acceptance and GPU/result claims withheld" in current
    assert not (root / "run-preparation/saved-ns-replay-binding-20261118.json").exists()
    assert not (root / "controllers/saved-ns-replay-20261118").exists()
    assert not (root / "run-preparation/saved-ns-replay-state-20261118.json").exists()
    assert not (root / "diagnostics/saved-ns-replay-20261118").exists()


@pytest.mark.parametrize("fault", ("none", "source", "log", "archive", "failed", "unrelated"))
def test_continuation_refresh_preserves_manual_results_and_requires_bound_cpu_evidence(tmp_path, fault):
    root = tmp_path / "root"
    previous, candidate = root / "candidates/parent", root / "candidates/fused-continuation-fixture"
    (previous / "experiment").mkdir(parents=True)
    (previous / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    (previous / "experiment/optimizer.py").write_text("# unchanged numerical source\n")
    shutil.copytree(previous, candidate, symlinks=True)
    (candidate / "experiment/trace.py").write_text("# diagnostic instrumentation\n")
    binding = root / "run-preparation/fused-attribution-validation-binding-20261118.json"
    record(binding, {"fixture": "preserved failed parent"})
    data = root / "data/experiment_receipt.json"
    record(data, {"fixture": "unchanged data"})
    digest = continuation_digest
    prepared = {
        "status": CONTINUATION_PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_binding_sha256": digest(binding),
        "data_receipt_sha256": digest(data),
        "parent_files_sha256": candidate_inventory(previous),
        "files_sha256": candidate_inventory(candidate),
        "parent_metadata_sha256": {},
        "changed_files": ["experiment/trace.py"],
        "allocation_predecessor": {"evidence_sha256": {}},
    }
    output = root / "run-preparation/fused-continuation-cpu/fixture"
    helpers = output / "tracking"
    helpers.mkdir(parents=True)
    helper = helpers / "prepare_fused_continuation.py"
    helper.write_text("# issued archived diagnostic helper\n")
    prepared["helpers_sha256"] = {helper.name: digest(helper)}
    record(candidate / CONTINUATION_PREPARATION, prepared)
    log = output / "candidate.log"
    log.write_text("7 passed\n")
    archive = output / "source.tar"
    expected = {
        **{name: candidate / name for name in prepared["files_sha256"]},
        CONTINUATION_PREPARATION: candidate / CONTINUATION_PREPARATION,
        f"tracking/{helper.name}": helper,
    }
    with tarfile.open(archive, "w") as bundle:
        for name, source in expected.items():
            bundle.add(source, arcname=name)
        link = tarfile.TarInfo("artifacts/external-sources")
        link.type, link.linkname = tarfile.SYMTYPE, "pinned-sources-materialized"
        bundle.addfile(link)
    cpu = {
        "status": CONTINUATION_CPU_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": digest(candidate / CONTINUATION_PREPARATION),
        "files_sha256": prepared["files_sha256"],
        "helpers_sha256": prepared["helpers_sha256"],
        "helpers": str(helpers),
        "command": cpu_command(),
        "exit_code": 0,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": digest(log),
        "archive": str(archive),
        "archive_sha256": digest(archive),
    }
    if fault == "source":
        (candidate / "experiment/optimizer.py").write_text("# changed numerical source\n")
    elif fault == "log":
        log.write_text("1 failed\n")
    elif fault == "archive":
        archive.write_bytes(b"replacement")
    elif fault == "failed":
        cpu.update(status="needs_attention", exit_code=1)
    elif fault == "unrelated":
        cpu["candidate"] = str(previous)
    record(output / "receipt.json", cpu)
    document = tmp_path / "ENDEAVOR.md"
    manual = "# Endeavor\n\nWave 12 verified; Wave 13 unlaunched.\n\n"
    document.write_text(manual + ANCHOR + "\n")
    update(document, root, datetime(2026, 11, 18, tzinfo=UTC))
    text = document.read_text()
    assert text.startswith(manual) and text.endswith(ANCHOR + "\n")
    current = "\n".join(fused_continuation_lines(root))
    accepted = "Source/archive-bound diagnostic CPU suite exited zero"
    if fault == "none":
        assert accepted in current and accepted in text
    elif fault in ("failed", "unrelated"):
        assert accepted not in current and accepted not in text
        assert ("Preserved CPU failure" if fault == "failed" else "No saved candidate CPU acceptance") in current
    else:
        assert accepted not in current and accepted not in text
        assert "CPU acceptance and result claims withheld" in current


@pytest.mark.parametrize("fault", ("none", "log", "archive", "fixture", "candidate", "failed"))
def test_executor_ledger_preserves_failures_and_withholds_changed_archived_evidence(tmp_path, fault):
    root = tmp_path / "root"
    candidate = root / "candidates/fused-continuation-fixture"
    for name in continuation_controller_fixture_names():
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fixture": name}) + "\n")
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    files = candidate_inventory(candidate)
    prepared = {"files_sha256": files, "helpers_sha256": {}, "parent_metadata_sha256": {}}
    record(candidate / CONTINUATION_PREPARATION, prepared)
    cpu_path = root / "run-preparation/fused-continuation-cpu/fixture/receipt.json"
    cpu_helpers = cpu_path.parent / "tracking"
    cpu_helpers.mkdir(parents=True)
    cpu_log = cpu_path.parent / "candidate.log"
    cpu_log.write_text("9 passed\n")
    cpu_archive = cpu_path.parent / "source.tar"
    with tarfile.open(cpu_archive, "x") as bundle:
        for name in (*files, CONTINUATION_PREPARATION):
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    cpu = {
        "root": str(root),
        "status": CONTINUATION_CPU_STATUS,
        "candidate": str(candidate),
        "files_sha256": files,
        "preparation_sha256": continuation_digest(candidate / CONTINUATION_PREPARATION),
        "helpers_sha256": {},
        "helpers": str(cpu_helpers),
        "log": str(cpu_log),
        "log_sha256": continuation_digest(cpu_log),
        "archive": str(cpu_archive),
        "archive_sha256": continuation_digest(cpu_archive),
        "exit_code": 0,
        "cuda_visible_devices": "",
    }
    record(cpu_path, cpu)
    output = root / "run-preparation/fused-continuation-controller-cpu/fixture"
    helpers = output / "tracking"
    helpers.mkdir(parents=True)
    helper = helpers / "executor.py"
    helper.write_text("# archived executor\n")
    hashes = {helper.name: continuation_digest(helper)}
    fixtures = {name: files[name] for name in continuation_controller_fixture_names()}
    for name in fixtures:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((candidate / name).read_bytes())
    archive = output / "controller-source.tar"
    with tarfile.open(archive, "x") as bundle:
        bundle.add(helper, arcname=f"tracking/{helper.name}", recursive=False)
        for name in fixtures:
            bundle.add(output / name, arcname=name, recursive=False)
    log = output / "controller.log"
    log.write_text("9 passed\n")
    receipt = {
        "root": str(root),
        "status": CONTINUATION_CONTROLLER_CPU_STATUS,
        "candidate_cpu_receipt": str(cpu_path),
        "candidate_cpu_receipt_sha256": continuation_digest(cpu_path),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixture_files_sha256": fixtures,
        "command": continuation_controller_cpu_command(),
        "exit_code": 0,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": continuation_digest(log),
        "archive": str(archive),
        "archive_sha256": continuation_digest(archive),
    }
    if fault == "log":
        log.write_text("1 failed\n")
    elif fault == "archive":
        archive.write_bytes(b"replaced archive")
    elif fault == "fixture":
        (output / next(iter(fixtures))).write_text("{}\n")
    elif fault == "candidate":
        (candidate / next(iter(fixtures))).write_text("{}\n")
    elif fault == "failed":
        receipt.update(status="needs_attention", exit_code=1)
    record(output / "receipt.json", receipt)
    document = tmp_path / "ENDEAVOR.md"
    manual = "# Endeavor\n\nWave 12 verified; no Wave 13 launch.\n\n"
    document.write_text(manual + ANCHOR + "\n")
    update(document, root, datetime(2026, 11, 18, tzinfo=UTC))
    text = document.read_text()
    assert text.startswith(manual) and text.endswith(ANCHOR + "\n")
    current = "\n".join(fused_continuation_controller_lines(root))
    accepted = "Source/archive-bound executor CPU suite exited zero"
    if fault == "none":
        assert accepted in current and accepted in text
        assert "9 configuration fixtures verified" in current
    elif fault == "failed":
        assert accepted not in current and "Preserved executor CPU failure" in current
    else:
        assert accepted not in current and "Executor acceptance and GPU/result claims withheld" in current


@pytest.mark.parametrize("fault", ("none", "log", "binding"))
def test_diagnostic_ledger_reports_only_its_bound_six_command_progress(tmp_path, fault):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    for name in continuation_controller_fixture_names():
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ISSUED_CANDIDATE / name, path)
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    binding, helpers, state, work, gpu = continuation_locations(root)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True)
    helper = helpers / "fused_continuation_controller.py"
    helper.write_text("# issued diagnostic controller\n")
    prepared = {"parent_metadata_sha256": {}}
    record(candidate / CONTINUATION_PREPARATION, prepared)
    cpu_path, controller_path = root / "candidate-cpu.json", root / "executor-cpu.json"
    record(cpu_path, {"fixture": "accepted candidate"})
    record(controller_path, {"fixture": "accepted controller"})
    archive = root / "source.tar"
    frozen = {
        "status": CONTINUATION_FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": candidate_inventory(candidate),
        "helpers": str(helpers),
        "helpers_sha256": {helper.name: continuation_digest(helper)},
        "preparation_sha256": continuation_digest(candidate / CONTINUATION_PREPARATION),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": continuation_digest(cpu_path),
        "controller_cpu_receipt": str(controller_path),
        "controller_cpu_receipt_sha256": continuation_digest(controller_path),
        "archive": str(archive),
        "plan": continuation_plan(candidate, root),
        "allocation_predecessor": {"evidence_sha256": {}},
    }
    frozen["commands"] = frozen["plan"]["commands"]
    with tarfile.open(archive, "x") as bundle:
        for name in continuation_inventory(frozen):
            if name.startswith("tracking/"):
                source = helpers / Path(name).name
            elif name == "candidate-cpu-receipt.json":
                source = cpu_path
            elif name == "controller-cpu-receipt.json":
                source = controller_path
            else:
                source = candidate / name
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    frozen["archive_sha256"] = continuation_digest(archive)
    record(binding, frozen)
    owner = {"controller_pid": 25001, "controller_start_ticks": "cpu-controller"}
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": 25001,
            "start_ticks": "cpu-controller",
            "binding_sha256": continuation_digest(binding),
            "command": continuation_controller_command(root),
        },
    )
    record(
        state,
        owner
        | {
            "status": "validating",
            "binding_sha256": continuation_digest(binding),
            "candidate": str(candidate),
            "allocation_predecessor": frozen["allocation_predecessor"],
            "command": continuation_validator_command(root, candidate),
        },
    )
    work.mkdir(parents=True)
    log = work / "control_continuous.log"
    log.write_text("original control command completed\n")
    record(
        gpu,
        {
            "status": "diagnosing",
            "execution_binding_sha256": continuation_digest(binding),
            "candidate_preparation_sha256": frozen["preparation_sha256"],
            "owner": owner,
            "commands": frozen["commands"],
            "gpu_commands": [
                {
                    "command": frozen["commands"][0],
                    "exit_code": 0,
                    "log": str(log),
                    "log_sha256": continuation_digest(log),
                }
            ],
        },
    )
    if fault == "log":
        log.write_text("replaced command log\n")
    elif fault == "binding":
        value = frozen | {"commands": frozen["commands"][1:]}
        record(binding, value)
    text = "\n".join(fused_continuation_controller_lines(root))
    if fault == "none":
        assert "1/6 integer-zero command exits" in text
        assert "Saved executor state: `validating`" in text
    else:
        assert "Diagnostic progress and acceptance claims withheld" in text
        assert "1/6 integer-zero command exits" not in text


@pytest.fixture
def attribution_validation_ledger(tmp_path):
    root = tmp_path / "root"
    previous, candidate = tmp_path / "parent", root / "candidates/fused-attribution-validation-ledger"
    write_failed_profiler(root)
    binding_path, helpers, state_path, work, gpu_path = attribution_validation_locations(root)
    for directory in (previous, candidate):
        (directory / "experiment/configs").mkdir(parents=True)
        (directory / "artifacts/pinned-sources-materialized").mkdir(parents=True)
        (directory / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    train = (
        "from pathlib import Path\n"
        "from .outer_diagonal_checks import OuterDiagonalVerifier\n"
        "    require_fused_attribution_owner()\n"
        '    if args.verify_outer_diagonal_updates and config["experiment"]["arm"] != "outer_leftdiag_fused":\n'
        '        raise ValueError("Outer update verification covers only outer_leftdiag_fused")\n'
        "        outer_verifier = OuterDiagonalVerifier(fast)\n"
        "            for optimizer in optimizers:\n                optimizer.step()\n"
    )
    (previous / "experiment/train.py").write_text(train)
    (previous / "experiment/block_toner.py").write_text("# unchanged numerical fixture\n")
    parent_helpers = tmp_path / "parent-helpers"
    parent_helpers.mkdir()
    (parent_helpers / "prepare_fused_attribution.py").write_text("# archived parent verifier\n")
    cpu_helpers = tmp_path / "cpu/tracking"
    cpu_helpers.mkdir(parents=True)
    for source, destination in ATTRIBUTION_VALIDATION_COPIES.items():
        raw = f"# issued validation fixture {source}\n"
        (cpu_helpers / source).write_text(raw)
        target = candidate / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(raw)
        if source != "fused_attribution_validation_contract.py":
            inherited = previous / destination
            inherited.parent.mkdir(parents=True, exist_ok=True)
            inherited.write_text("# inherited owner/check fixture\n")
    (cpu_helpers / "prepare_fused_attribution_validation.py").write_text("# own archived preparer\n")
    (candidate / "experiment/train.py").write_bytes(attribution_validation_transformed_train(previous))
    shutil.copy2(previous / "experiment/block_toner.py", candidate / "experiment/block_toner.py")
    for spec in attribution_validation_trial_specs():
        config = {
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
            "runtime": {"compile": "inductor", "attention": "FA2", "head": "compiled"},
            "root": {"graphs": True},
            "optimizer": {"lr": spec["matrix_lr"], "ns_mode": "tpu", "ns_steps": 10},
            "schedule": ATTRIBUTION_VALIDATION_SCHEDULE,
        }
        for directory in (previous, candidate):
            record(directory / spec["config_name"], config)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def archive(path, entries, alias=None):
        with tarfile.open(path, "w") as bundle:
            for name, source in entries.items():
                bundle.add(source, arcname=name, recursive=False)
            if alias is not None:
                bundle.add(alias / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)

    release = root / "run-preparation/wave12-launch-binding.json"
    record(release, {"files_sha256": {}})
    data = root / "data/experiment_receipt.json"
    record(data, {"data": "issued data identity"})
    for name in ("analysis-wave12.json", "swanlab-upload-wave12.json"):
        record(release.parent / name, {"status": "complete"})
    parent_path = previous / ATTRIBUTION_PREPARATION
    parent = {
        "status": "fused_attribution_full_schedule_prepared_gpu_denied",
        "root": str(root),
        "candidate": str(previous),
        "files_sha256": candidate_inventory(previous),
        "parent_metadata_sha256": {},
        "helpers_sha256": {path.name: digest(path) for path in parent_helpers.iterdir()},
        "parent_release_sha256": digest(release),
        "data_receipt_sha256": digest(data),
        "predecessor_results": {
            "analysis_sha256": digest(release.parent / "analysis-wave12.json"),
            "cloud_sha256": digest(release.parent / "swanlab-upload-wave12.json"),
        },
    }
    record(parent_path, parent)
    parent_out = tmp_path / "parent-cpu"
    parent_out.mkdir()
    parent_log = parent_out / "candidate.log"
    parent_log.write_text("original parent CPU execution\n")
    parent_archive = parent_out / "source.tar"
    archive(
        parent_archive,
        {name: previous / name for name in parent["files_sha256"]}
        | {ATTRIBUTION_PREPARATION: parent_path}
        | {f"tracking/{name}": parent_helpers / name for name in parent["helpers_sha256"]},
        previous,
    )
    parent_cpu_path = parent_out / "receipt.json"
    record(
        parent_cpu_path,
        {
            "status": "fused_attribution_cpu_verified_gpu_denied",
            "root": str(root),
            "candidate": str(previous),
            "files_sha256": parent["files_sha256"],
            "helpers_sha256": parent["helpers_sha256"],
            "preparation_sha256": digest(parent_path),
            "command": cpu_command(),
            "cwd": str(previous),
            "exit_code": 0,
            "cuda_visible_devices": "",
            "log": str(parent_log),
            "log_sha256": digest(parent_log),
            "archive": str(parent_archive),
            "archive_sha256": digest(parent_archive),
        },
    )
    shutil.copy2(parent_path, candidate / ATTRIBUTION_PREPARATION)
    planned = attribution_validation_plan(candidate, root, work)
    predecessor = failed_profiler_evidence(root)
    prepared = {
        "status": "fused_attribution_validation_prepared_no_gpu_binding",
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_cpu_receipt": str(parent_cpu_path),
        "parent_cpu_receipt_sha256": digest(parent_cpu_path),
        "parent_preparation_sha256": digest(parent_path),
        "parent_metadata_sha256": {ATTRIBUTION_PREPARATION: digest(parent_path)},
        "scientific_parent_release_sha256": digest(release),
        "data_receipt_sha256": digest(data),
        "files_sha256": candidate_inventory(candidate),
        "changed_files": sorted(ATTRIBUTION_VALIDATION_CHANGES),
        "helpers_sha256": {path.name: digest(path) for path in cpu_helpers.iterdir()},
        "allocation_predecessor": predecessor,
        "plan": planned,
        "expansion_policy": ATTRIBUTION_VALIDATION_EXPANSION_POLICY,
    }
    preparation_path = candidate / ATTRIBUTION_VALIDATION_PREPARATION
    record(preparation_path, prepared)
    cpu_log, cpu_archive = cpu_helpers.parent / "candidate.log", cpu_helpers.parent / "source.tar"
    cpu_log.write_text("original validation CPU execution\n")
    candidate_entries = {
        name: candidate / name for name in prepared["files_sha256"] | prepared["parent_metadata_sha256"]
    }
    candidate_entries[ATTRIBUTION_VALIDATION_PREPARATION] = preparation_path
    archive(
        cpu_archive,
        candidate_entries | {f"tracking/{name}": cpu_helpers / name for name in prepared["helpers_sha256"]},
        candidate,
    )
    cpu_path = root / "run-preparation/fused-attribution-validation-cpu/fixture/receipt.json"
    record(
        cpu_path,
        {
            "status": "fused_attribution_validation_cpu_verified_gpu_denied",
            "root": str(root),
            "candidate": str(candidate),
            "files_sha256": prepared["files_sha256"],
            "helpers_sha256": prepared["helpers_sha256"],
            "helpers": str(cpu_helpers),
            "preparation_sha256": digest(preparation_path),
            "command": cpu_command(),
            "cwd": str(candidate),
            "exit_code": 0,
            "cuda_visible_devices": "",
            "log": str(cpu_log),
            "log_sha256": digest(cpu_log),
            "archive": str(cpu_archive),
            "archive_sha256": digest(cpu_archive),
        },
    )
    helpers.mkdir(parents=True)
    controller_helpers = tmp_path / "controller-cpu/tracking"
    shutil.copytree(cpu_helpers, controller_helpers)
    (controller_helpers / "fused_attribution_validation_controller.py").write_text("# archived controller fixture\n")
    for path in controller_helpers.iterdir():
        shutil.copy2(path, helpers / path.name)
    hashes = {path.name: digest(path) for path in helpers.iterdir()}
    fixtures = {
        spec["config_name"]: prepared["files_sha256"][spec["config_name"]]
        for spec in attribution_validation_trial_specs()
    }
    for name in fixtures:
        target = controller_helpers.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate / name, target)
    controller_log = controller_helpers.parent / "controller.log"
    controller_log.write_text("original controller CPU execution\n")
    controller_archive = controller_helpers.parent / "controller-source.tar"
    archive(
        controller_archive,
        {f"tracking/{name}": controller_helpers / name for name in hashes}
        | {name: controller_helpers.parent / name for name in fixtures},
    )
    controller_path = controller_helpers.parent / "receipt.json"
    record(
        controller_path,
        {
            "status": "fused_attribution_validation_controller_cpu_verified_no_gpu_binding",
            "root": str(root),
            "candidate_cpu_receipt": str(cpu_path),
            "candidate_cpu_receipt_sha256": digest(cpu_path),
            "helpers": str(controller_helpers),
            "helpers_sha256": hashes,
            "fixture_files_sha256": fixtures,
            "command": [
                "/opt/conda/envs/torch-base/bin/python",
                "-P",
                "-B",
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                "-p",
                "no:cacheprovider",
                "-q",
                "-m",
                "not slow and not integration and not data_integration and not cluster "
                "and not requires_cluster and not docker and not manual",
                "tracking/test_fused_attribution_validation_controller.py",
                "tracking/test_fused_attribution_validation_owner.py",
                "tracking/test_selected_lr_failure_handoff.py",
            ],
            "cwd": str(controller_helpers.parent),
            "exit_code": 0,
            "cuda_visible_devices": "",
            "log": str(controller_log),
            "log_sha256": digest(controller_log),
            "archive": str(controller_archive),
            "archive_sha256": digest(controller_archive),
        },
    )
    frozen_archive = root / "run-preparation" / f"source-{helpers.parent.name}.tar"
    archive(
        frozen_archive,
        candidate_entries
        | {
            "candidate-cpu-receipt.json": cpu_path,
            "controller-cpu-receipt.json": controller_path,
        }
        | {f"tracking/{name}": helpers / name for name in hashes},
        candidate,
    )
    record(
        binding_path,
        {
            "status": "frozen_fused_attribution_validation_pending_gpu_gates",
            "root": str(root),
            "candidate": str(candidate),
            "node": "28.216.17.9",
            "python": "/opt/conda/envs/torch-base/bin/python",
            "environment": planned["environment"],
            "files_sha256": prepared["files_sha256"],
            "preparation_sha256": digest(preparation_path),
            "cpu_receipt": str(cpu_path),
            "cpu_receipt_sha256": digest(cpu_path),
            "controller_cpu_receipt": str(controller_path),
            "controller_cpu_receipt_sha256": digest(controller_path),
            "helpers": str(helpers),
            "helpers_sha256": hashes,
            "archive": str(frozen_archive),
            "archive_sha256": digest(frozen_archive),
            "data_receipt_sha256": prepared["data_receipt_sha256"],
            "allocation_predecessor": predecessor,
            "plan": planned,
            "commands": [command for trial in planned["trials"] for command in trial["commands"]],
            "work": str(work),
            "receipt": str(gpu_path),
            "validation_timeout": 32400,
        },
    )
    owner = {"controller_pid": 1001, "controller_start_ticks": "2001"}
    command = attribution_validation_controller_command(root)
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "binding_sha256": digest(binding_path),
            "command": command,
            "allocation_predecessor": predecessor,
            "pid": 1001,
            "start_ticks": "2001",
            "marker": "owner",
            "log": str(helpers.parent / "controller.log"),
            "generation": {"pid": 1001, "start_ticks": "2001", "group": 1001, "session": 1001, "command": command},
        },
    )
    record(
        state_path,
        owner
        | {
            "status": "validating",
            "binding_sha256": digest(binding_path),
            "candidate": str(candidate),
            "node": "28.216.17.9",
            "allocation_predecessor": predecessor,
            "spawn_status": "acknowledged",
            "command": attribution_validation_validator_command(root, candidate),
            "child_pid": 1002,
            "child_start_ticks": "2002",
            "child_marker": "validator",
            "log": str(helpers.parent / "validator.log"),
            "active_command": planned["trials"][0]["commands"][0],
        },
    )
    work.mkdir(parents=True)
    log = work / f"{planned['trials'][0]['name']}_continuous.log"
    log.write_text("original bounded command output\n")
    record(
        gpu_path,
        planned
        | {
            "status": "validating",
            "owner": owner,
            "execution_binding_sha256": digest(binding_path),
            "candidate_preparation_sha256": digest(preparation_path),
            "time": "2026-11-18T00:00:00+00:00",
            "gpu_commands": [
                {
                    "trial": planned["trials"][0]["name"],
                    "mode": "continuous",
                    "command": planned["trials"][0]["commands"][0],
                    "exit_code": 0,
                    "log": str(log),
                    "log_sha256": digest(log),
                }
            ],
        },
    )
    return root, candidate, cpu_path, log, archive


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "rehashed_source",
        "unbound_count",
        "replaced_log",
        "cpu_failure",
        "helper",
        "data",
        "predecessor",
        "owner",
        "trial_checks",
        "active_command",
        "cpu_log",
        "cpu_archive",
        "controller_log",
        "controller_cpu_failure",
        "cpu_helper",
        "gpu_binding",
        "gpu_status",
    ),
)
def test_attribution_validation_refresh_preserves_milestones_and_withholds_unbound_progress(
    attribution_validation_ledger,
    fault,
):
    root, candidate, cpu_path, log, _ = attribution_validation_ledger
    _, helpers, state_path, _, gpu_path = attribution_validation_locations(root)
    document = root / "ENDEAVOR.md"
    prose = f"# Research\n\nWave12 final result/cloud milestone.\n\n{ANCHOR}\n\nOriginal profiler failure preserved.\n"
    document.write_text(prose)
    failed_path = profile_locations(root)[4]
    failed_bytes = failed_path.read_bytes()
    if fault == "rehashed_source":
        path = candidate / "experiment/block_toner.py"
        path.write_text("# different optimizer\n")
        prepared_path = candidate / ATTRIBUTION_VALIDATION_PREPARATION
        prepared = json.loads(prepared_path.read_bytes())
        prepared["files_sha256"] = candidate_inventory(candidate)
        record(prepared_path, prepared)
        cpu = json.loads(cpu_path.read_bytes())
        cpu["files_sha256"] = prepared["files_sha256"]
        cpu["preparation_sha256"] = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
        record(cpu_path, cpu)
    elif fault == "unbound_count":
        receipt = json.loads(gpu_path.read_bytes())
        receipt["gpu_commands"].append(dict(receipt["gpu_commands"][0]))
        record(gpu_path, receipt)
    elif fault == "replaced_log":
        log.write_text("different execution\n")
    elif fault == "cpu_failure":
        receipt = json.loads(cpu_path.read_bytes())
        receipt.update(status="needs_attention", exit_code=7, error="CPU execution failed")
        record(cpu_path, receipt)
    elif fault == "helper":
        (helpers / "fused_attribution_validation_controller.py").write_text("# replacement controller\n")
    elif fault == "data":
        record(root / "data/experiment_receipt.json", {"data": "different data"})
    elif fault == "predecessor":
        record(failed_path, {"status": "passed"})
    elif fault == "owner":
        receipt = json.loads(gpu_path.read_bytes())
        receipt["owner"]["controller_start_ticks"] = "different"
        record(gpu_path, receipt)
    elif fault == "trial_checks":
        receipt = json.loads(gpu_path.read_bytes())
        receipt["trials"][0]["checks"] = {"invented_pass": True}
        record(gpu_path, receipt)
    elif fault == "active_command":
        state = json.loads(state_path.read_bytes())
        state["active_command"] = json.loads(gpu_path.read_bytes())["trials"][-1]["commands"][0]
        record(state_path, state)
    elif fault in ("cpu_log", "cpu_archive"):
        cpu = json.loads(cpu_path.read_bytes())
        path = Path(cpu["log"] if fault == "cpu_log" else cpu["archive"])
        path.write_bytes(b"different CPU evidence")
    elif fault == "controller_log":
        state = json.loads(state_path.read_bytes())
        Path(state["log"]).write_text("original validator output\n")
        state["log_sha256"] = hashlib.sha256(Path(state["log"]).read_bytes()).hexdigest()
        record(state_path, state)
        Path(state["log"]).write_text("different validator output\n")
    elif fault == "gpu_binding":
        receipt = json.loads(gpu_path.read_bytes())
        receipt["execution_binding_sha256"] = "different"
        record(gpu_path, receipt)
    elif fault == "gpu_status":
        receipt = json.loads(gpu_path.read_bytes())
        receipt["status"] = "passed_selected_lr_stage_profile_continuation_and_trace_gates"
        record(gpu_path, receipt)
    elif fault == "controller_cpu_failure":
        binding = json.loads(attribution_validation_locations(root)[0].read_bytes())
        controller_path = Path(binding["controller_cpu_receipt"])
        controller = json.loads(controller_path.read_bytes())
        controller.update(status="needs_attention", exit_code=7, error="controller CPU execution failed")
        record(controller_path, controller)
    elif fault == "cpu_helper":
        cpu = json.loads(cpu_path.read_bytes())
        (Path(cpu["helpers"]) / "prepare_fused_attribution_validation.py").write_text("# different archived helper\n")
    update(document, root, datetime(2026, 11, 18, tzinfo=UTC))
    text = document.read_text()
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    current = "\n".join(fused_attribution_validation_lines(root))
    if fault == "none":
        assert "Source-bound saved attribution GPU commands: 1 / 27 exited zero" in current
        assert "| `null_leftdiag_fused_full_lr_upper` |" in current
        assert "not independent numerical acceptance" in current
        assert "Saved validator owner `1002/2002`" in current
        assert "passed_fused_attribution_actual_config_continuation_and_work_gates" not in current
        assert "Source-bound saved attribution GPU commands: 1 / 27" in text
    else:
        assert "attribution progress counts withheld" in current
        assert "Source-bound saved attribution GPU commands:" not in current
        assert "Source/archive-bound validation CPU suite exited zero" not in current
    assert failed_path.read_bytes() == (failed_bytes if fault != "predecessor" else b'{"status": "passed"}\n')


def test_attribution_validation_distinct_preparation_does_not_inherit_bound_candidate_progress(
    attribution_validation_ledger,
):
    root, candidate, _, _, _ = attribution_validation_ledger
    separate = candidate.with_name("fused-attribution-validation-separate")
    shutil.copytree(candidate, separate, symlinks=True)
    path = separate / ATTRIBUTION_VALIDATION_PREPARATION
    prepared = json.loads(path.read_bytes())
    prepared["candidate"] = str(separate)
    prepared["plan"] = attribution_validation_plan(separate, root, attribution_validation_locations(root)[3])
    record(path, prepared)
    current = "\n".join(fused_attribution_validation_lines(root))
    bound, unbound = current.split(f"Saved preparation: `{path}`.")
    assert "Source-bound saved attribution GPU commands: 1 / 27 exited zero" in bound
    assert "No saved candidate CPU acceptance" in unbound
    assert "No canonical execution binding for this candidate" in unbound
    assert "Source-bound saved attribution GPU commands:" not in unbound


def test_attribution_validation_copied_parent_metadata_does_not_emit_parent_study_mismatch(
    attribution_validation_ledger,
):
    root, candidate, _, _, _ = attribution_validation_ledger
    copied_parent = candidate / ATTRIBUTION_PREPARATION
    parent_bytes = copied_parent.read_bytes()
    assert fused_attribution_lines(root) == []
    validation = "\n".join(fused_attribution_validation_lines(root))
    assert "Source-bound saved attribution GPU commands: 1 / 27 exited zero" in validation
    assert copied_parent.read_bytes() == parent_bytes
    (candidate / ATTRIBUTION_VALIDATION_PREPARATION).unlink()
    assert fused_attribution_lines(root) == []


def test_attribution_validation_cpu_acceptance_uses_issued_archive_without_live_binding(
    attribution_validation_ledger,
    tmp_path,
    monkeypatch,
):
    root, _, cpu_path, _, _ = attribution_validation_ledger
    binding_path, helpers, state_path, _, gpu_path = attribution_validation_locations(root)
    for path in (binding_path, state_path, gpu_path, helpers.parent / "controller.json"):
        path.unlink()
    mutable = tmp_path / "mutable-working-helpers"
    mutable.mkdir()
    (mutable / "prepare_fused_attribution_validation.py").write_text("# later different working helper\n")
    monkeypatch.setattr(
        attribution_validation_preparation,
        "__file__",
        str(mutable / "prepare_fused_attribution_validation.py"),
    )
    current = "\n".join(fused_attribution_validation_lines(root))
    assert "Source/archive-bound validation CPU suite exited zero with CUDA hidden" in current
    assert str(cpu_path) in current
    assert "No canonical execution binding" in current
    assert "Source-bound saved attribution GPU commands:" not in current


def test_attribution_validation_rehashed_replacement_archive_does_not_bind_changed_source(
    attribution_validation_ledger,
):
    root, candidate, cpu_path, _, archive = attribution_validation_ledger
    binding_path, helpers, state_path, _, gpu_path = attribution_validation_locations(root)
    for path in (binding_path, state_path, gpu_path, helpers.parent / "controller.json"):
        path.unlink()
    cpu = json.loads(cpu_path.read_bytes())
    replaced = Path(cpu["archive"])
    entries = {}
    with tarfile.open(replaced) as bundle:
        for member in bundle.getmembers():
            if member.isfile():
                entries[member.name] = (
                    Path(cpu["helpers"]) / Path(member.name).name
                    if member.name.startswith("tracking/")
                    else candidate / member.name
                )
    changed = Path(cpu["helpers"]).parent / "different-train.py"
    changed.write_bytes((candidate / "experiment/train.py").read_bytes() + b"\n# replaced execution\n")
    entries["experiment/train.py"] = changed
    archive(replaced, entries, candidate)
    cpu["archive_sha256"] = hashlib.sha256(replaced.read_bytes()).hexdigest()
    record(cpu_path, cpu)
    current = "\n".join(fused_attribution_validation_lines(root))
    assert "CPU acceptance and attribution progress counts withheld" in current
    assert "Source/archive-bound validation CPU suite exited zero" not in current


@pytest.mark.parametrize("fault", ("none", "status", "cleanup", "receipt_hash"))
def test_attribution_validation_terminal_saved_acceptance_requires_complete_bound_receipt(
    attribution_validation_ledger,
    fault,
):
    root, _, _, _, _ = attribution_validation_ledger
    _, helpers, state_path, work, gpu_path = attribution_validation_locations(root)
    receipt = json.loads(gpu_path.read_bytes())
    commands = []
    for trial in receipt["trials"]:
        trial["checks"] = dict.fromkeys(trial["checks"], True)
        trial["evidence"] = {"scope": "saved fixture evidence; no tensors loaded"}
        for mode, command in zip(("continuous", "interrupted", "resumed"), trial["commands"], strict=True):
            log = work / f"{trial['name']}_{mode}.log"
            log.write_text(f"completed {trial['name']}/{mode}\n")
            commands.append(
                {
                    "trial": trial["name"],
                    "mode": mode,
                    "command": command,
                    "exit_code": 0,
                    "log": str(log),
                    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            )
    receipt.update(status="passed_fused_attribution_actual_config_continuation_and_work_gates", gpu_commands=commands)
    if fault == "status":
        receipt["status"] = "passed_selected_lr_stage_profile_continuation_and_trace_gates"
    record(gpu_path, receipt)
    state = json.loads(state_path.read_bytes())
    log = helpers.parent / "validator.log"
    log.write_text("original validator completed and reconstructed\n")
    state.update(
        status="passed",
        active_command=commands[-1]["command"],
        exit_code=0,
        cleanup_exit_code=0,
        keeper_restore=0,
        marked_descendants_absent=True,
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(gpu_path.read_bytes()).hexdigest(),
    )
    if fault == "cleanup":
        state["cleanup_exit_code"] = 1
    elif fault == "receipt_hash":
        state["validation_sha256"] = "different"
    record(state_path, state)
    failed_path = profile_locations(root)[4]
    failed_bytes = failed_path.read_bytes()
    current = "\n".join(fused_attribution_validation_lines(root))
    if fault == "none":
        assert "Source-bound saved attribution GPU commands: 27 / 27 exited zero" in current
        assert "Saved controller records reconstructed attribution validation gates" in current
        assert "this renderer does not repeat tensor checks" in current
    else:
        assert "attribution progress counts withheld" in current
        assert "Source-bound saved attribution GPU commands:" not in current
        assert "Saved controller records reconstructed attribution validation gates" not in current
    assert failed_path.read_bytes() == failed_bytes


def test_attribution_validation_failed_cpu_receipt_is_preserved_without_execution_binding(
    attribution_validation_ledger,
):
    root, _, cpu_path, _, _ = attribution_validation_ledger
    binding_path, helpers, state_path, _, gpu_path = attribution_validation_locations(root)
    for path in (binding_path, state_path, gpu_path, helpers.parent / "controller.json"):
        path.unlink()
    cpu = json.loads(cpu_path.read_bytes())
    cpu.update(status="needs_attention", exit_code=7, error="CPU validation failed")
    record(cpu_path, cpu)
    current = "\n".join(fused_attribution_validation_lines(root))
    assert "Preserved attribution validation CPU failure: `needs_attention`, exit `7`" in current
    assert "Preserved attribution validation CPU error: `CPU validation failed`" in current
    assert "Source/archive-bound validation CPU suite exited zero" not in current
    assert "Source-bound saved attribution GPU commands:" not in current


@pytest.fixture
def attribution_ledger(tmp_path):
    issued_root = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    issued_cpu = json.loads(
        (issued_root / "run-preparation/fused-attribution-cpu/full-schedule-20261118T0257/receipt.json").read_text()
    )
    prepared = json.loads((Path(issued_cpu["candidate"]) / ATTRIBUTION_PREPARATION).read_text())
    root = tmp_path / "root"
    candidate, previous = root / "candidates/fused-attribution-ledger", root / "parent"
    for source, destination in (
        (Path(prepared["candidate"]), candidate),
        (Path(prepared["parent_candidate"]), previous),
    ):
        shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    release = root / "run-preparation/wave12-launch-binding.json"
    release.parent.mkdir(parents=True)
    shutil.copy2(prepared["parent_release"], release)
    parent_archive = root / "parent.tar"
    parent_archive.write_bytes(b"parent fixture")
    data = root / "data/experiment_receipt.json"
    data.parent.mkdir(parents=True)
    data.write_text("{}\n")
    for name in ("analysis-wave12.json", "swanlab-upload-wave12.json"):
        (release.parent / name).write_text("{}\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    prepared.update(
        root=str(root),
        candidate=str(candidate),
        parent_candidate=str(previous),
        parent_release=str(release),
        parent_release_sha256=digest(release),
        parent_archive=str(parent_archive),
        parent_archive_sha256=digest(parent_archive),
        data_receipt_sha256=digest(data),
    )
    prepared["predecessor_results"]["analysis_sha256"] = digest(release.parent / "analysis-wave12.json")
    prepared["predecessor_results"]["cloud_sha256"] = digest(release.parent / "swanlab-upload-wave12.json")
    for trial in prepared["trials"]:
        trial["config"] = str(candidate / Path(trial["config"]).relative_to(issued_cpu["candidate"]))
    preparation = candidate / ATTRIBUTION_PREPARATION
    preparation.write_text(json.dumps(prepared) + "\n")
    out = root / "run-preparation/fused-attribution-cpu/ledger"
    out.mkdir(parents=True)
    log, archive = out / "candidate.log", out / "source.tar"
    shutil.copy2(issued_cpu["log"], log)
    with tarfile.open(issued_cpu["archive"]) as original, tarfile.open(archive, "x") as bundle:
        for name in (
            prepared["files_sha256"]
            | prepared["parent_metadata_sha256"]
            | {ATTRIBUTION_PREPARATION: digest(preparation)}
        ):
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        for name in prepared["helpers_sha256"]:
            member = original.getmember(f"tracking/{name}")
            bundle.addfile(member, original.extractfile(member))
    cpu = issued_cpu | {
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": digest(preparation),
        "cwd": str(candidate),
        "log": str(log),
        "log_sha256": digest(log),
        "archive": str(archive),
        "archive_sha256": digest(archive),
    }
    receipt = out / "receipt.json"
    receipt.write_text(json.dumps(cpu) + "\n")
    return root, candidate, log, archive, receipt


@pytest.mark.parametrize("fault", ("none", "source", "log", "archive", "cpu_failure", "validation_child"))
def test_attribution_refresh_preserves_milestones_and_rejects_unbound_cpu_acceptance(attribution_ledger, fault):
    root, candidate, log, archive, receipt_path = attribution_ledger
    document = root / "ENDEAVOR.md"
    prose = f"# Research\n\nWave12 five-window/cloud milestone.\n\n{ANCHOR}\n\nProfiler failure preserved.\n"
    document.write_text(prose)
    if fault == "source":
        with (candidate / "experiment/block_toner.py").open("a") as stream:
            stream.write("\n# changed source\n")
    elif fault == "log":
        log.write_text("different CPU execution\n")
    elif fault == "archive":
        archive.write_bytes(b"replaced source archive")
    elif fault == "cpu_failure":
        receipt = json.loads(receipt_path.read_text())
        receipt.update(status="fused_attribution_cpu_failed", exit_code=7, error="CPU execution failed")
        receipt_path.write_text(json.dumps(receipt) + "\n")
    elif fault == "validation_child":
        child = candidate.with_name("fused-attribution-validation-child")
        child.mkdir()
        shutil.copy2(candidate / ATTRIBUTION_PREPARATION, child / ATTRIBUTION_PREPARATION)
    update(document, root, datetime(2026, 11, 18, tzinfo=UTC))
    text = document.read_text()
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    current = "\n".join(fused_attribution_lines(root))
    assert "no new wave results or cloud imports" in current or "launch/result claims withheld" in current
    if fault in ("none", "validation_child"):
        assert "Source/archive-bound CPU suite exited zero" in current
        assert "Nine exploratory seed-0 full-budget runs" in current
        assert "Saved attribution evidence differs or is incomplete" not in current
        assert "fused-attribution-validation-child" not in current
    elif fault == "cpu_failure":
        assert "Preserved attribution CPU error: `CPU execution failed`" in current
        assert "Source/archive-bound CPU suite exited zero" not in current
    else:
        assert "CPU acceptance and launch/result claims withheld" in current
        assert "Source/archive-bound CPU suite exited zero" not in current


@pytest.mark.parametrize("fault", ("none", "protected_input", "archive", "cpu_failure"))
def test_saved_reader_ledger_preserves_failed_gpu_status_and_withholds_changed_evidence(tmp_path, fault):
    root = tmp_path / "root"
    out = root / "run-preparation/selected-lr-stage-profile-reader-cpu/fixture"
    out.mkdir(parents=True)
    failed = profile_locations(root)[4]
    failed.write_text('{"status":"needs_attention"}\n')
    scope = "MuonT saved rank traces only; no tensor/trial reconstruction, global acceptance, or GPU grant."
    helper = out / "tracking/reader.py"
    helper.parent.mkdir()
    helper.write_text("# archived CPU reader fixture\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    helpers = {"reader.py": digest(helper)}
    inputs = {str(failed): digest(failed)}
    prepared_path = out / "preparation.json"
    prepared_path.write_text(json.dumps({"scope": scope, "helpers_sha256": helpers, "inputs_sha256": inputs}))
    evidence = {
        str(rank): {
            "training_update_events": 20,
            "optimizer_step_events": 20,
            "cuda_training_update_events": 20,
            "cuda_optimizer_step_events": 20,
            "device_kernel_events": 100,
        }
        for rank in range(8)
    }
    evidence_path = out / "saved-trace-evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    log = out / "cpu.log"
    log.write_text("original CPU execution\n")
    command = {"exit_code": 0, "cuda_visible_devices": "", "log": str(log), "log_sha256": digest(log)}
    inventory = {
        "preparation.json": digest(prepared_path),
        "saved-trace-evidence.json": digest(evidence_path),
        "tracking/reader.py": digest(helper),
        "cpu.log": digest(log),
    }
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(out / name, arcname=name, recursive=False)
    receipt = {
        "status": "selected_lr_stage_profile_muont_reader_cpu_verified_gpu_denied",
        "root": str(root),
        "out": str(out),
        "scope": scope,
        "preparation_sha256": digest(prepared_path),
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "archive_inventory_sha256": inventory,
        "helpers_sha256": helpers,
        "inputs_sha256": inputs,
        "failed_receipt_status_preserved": "needs_attention",
        "rank_traces": evidence,
        **{key: dict(command) for key in ("frozen_binding_check", "cpu", "lint", "saved_trace_reconstruction")},
    }
    if fault == "protected_input":
        failed.write_text('{"status":"passed"}\n')
    elif fault == "archive":
        archive.write_bytes(b"replaced archive")
    elif fault == "cpu_failure":
        receipt["cpu"]["exit_code"] = 7
    (out / "receipt.json").write_text(json.dumps(receipt))
    document = root / "ENDEAVOR.md"
    prose = f"# Research\n\nOriginal profiling failure preserved.\n\n{ANCHOR}\n\nWave12 cloud milestone.\n"
    document.write_text(prose)
    update(document, root, datetime(2026, 11, 18, tzinfo=UTC))
    text = document.read_text()
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    current = "\n".join(selected_lr_stage_profile_reader_lines(root))
    if fault == "none":
        assert "Eight saved MuonT ranks each reconstruct 20 CPU updates" in current
        assert "The original profiling receipt remains `needs_attention`" in current
    else:
        assert "Reader acceptance/counts withheld" in current
        assert "Eight saved MuonT ranks each reconstruct 20 CPU updates" not in current


@pytest.fixture
def profile_ledger(tmp_path):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    binding_path, helpers, state_path, work, receipt_path = profile_locations(root)
    helpers.mkdir(parents=True)
    binding_path.parent.mkdir(parents=True)
    configs = candidate / "experiment/configs"
    configs.mkdir(parents=True)
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for arm in SELECTED_LR_ARMS:
        config = {
            "experiment": {"arm": arm, "refresh_install_age": 1},
            "training": {"steps": 5120, "tokens": 2684354560, "batch_tokens": 524288, "world_size": 8, "seed": 0},
            "runtime": {"compile": "inductor", "attention": "FA2", "head": "compiled"},
        }
        (configs / f"{arm}_0p5m_seed0.json").write_text(json.dumps(config))
    artifacts = {
        "archive": root / "source.tar",
        "cpu_receipt": root / "cpu.json",
        "controller_cpu_receipt": root / "controller-cpu.json",
        "preparation": candidate / PROFILE_PREPARATION,
        "data_receipt": root / "data/experiment_receipt.json",
        "predecessor_release": root / "run-preparation/wave12-launch-binding.json",
    }
    for key, path in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if key != "archive":
            path.write_text("{}\n")
    for name in ("run_selected_lr_stage_profile.py", "validate_selected_lr_stage_profile.py"):
        (helpers / name).write_text("# source-bound reporter fixture\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    hashes = {path.name: digest(path) for path in helpers.iterdir()}
    files = candidate_inventory(candidate)
    artifacts["preparation"].write_text(json.dumps({"parent_metadata_sha256": {}}))
    artifacts["cpu_receipt"].write_text(
        json.dumps(
            {
                "status": "selected_lr_stage_profile_validation_cpu_verified_no_gpu_binding",
                "exit_code": 0,
                "cuda_visible_devices": "",
                "files_sha256": files,
                "helpers_sha256": hashes,
            }
        )
    )
    artifacts["controller_cpu_receipt"].write_text(
        json.dumps(
            {
                "status": "selected_lr_stage_profile_controller_cpu_verified_no_gpu_binding",
                "exit_code": 0,
                "cuda_visible_devices": "",
                "helpers_sha256": hashes,
            }
        )
    )
    planned = profile_validation_plan(candidate, root, work)
    frozen = {
        "status": PROFILE_FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "node": "28.216.17.9",
        "work": str(work),
        "receipt": str(receipt_path),
        "environment": planned["environment"],
        "files_sha256": files,
        "helpers_sha256": hashes,
        "plan": planned,
        "commands": [command for trial in planned["trials"] for command in trial["commands"]],
        **{key: str(artifacts[key]) for key in ("archive", "cpu_receipt", "controller_cpu_receipt")},
        **{f"{key}_sha256": digest(path) for key, path in artifacts.items() if key != "archive"},
    }
    expected = (
        files
        | {
            PROFILE_PREPARATION: frozen["preparation_sha256"],
            "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": value for name, value in hashes.items()}
    )
    with tarfile.open(artifacts["archive"], "x") as bundle:
        for name in expected:
            source = (
                helpers / Path(name).name
                if name.startswith("tracking/")
                else (
                    artifacts["cpu_receipt"]
                    if name == "candidate-cpu-receipt.json"
                    else (
                        artifacts["controller_cpu_receipt"]
                        if name == "controller-cpu-receipt.json"
                        else candidate / name
                    )
                )
            )
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    frozen["archive_sha256"] = digest(artifacts["archive"])
    binding_path.write_text(json.dumps(frozen))
    owner = {"controller_pid": 123, "controller_start_ticks": "456"}
    (helpers.parent / "controller.json").write_text(
        json.dumps(
            {
                "status": "acknowledged",
                "binding_sha256": digest(binding_path),
                "command": profile_controller_command(root),
                "pid": 123,
                "start_ticks": "456",
            }
        )
    )
    state = {
        "status": "validating",
        "binding_sha256": digest(binding_path),
        **owner,
        "candidate": str(candidate),
        "node": frozen["node"],
        "command": profile_validator_command(root, candidate),
    }
    state_path.write_text(json.dumps(state))
    work.mkdir(parents=True)
    log = work / "muont_continuous.log"
    log.write_text("original bounded command\n")
    receipt = planned | {
        "status": "validating",
        "owner": owner,
        "gpu_commands": [
            {
                "arm": "muont",
                "mode": "continuous",
                "command": frozen["commands"][0],
                "exit_code": 0,
                "log": str(log),
                "log_sha256": digest(log),
            }
        ],
    }
    receipt_path.write_text(json.dumps(receipt))
    return root, log, state_path, receipt_path


@pytest.mark.parametrize("fault", ("none", "log", "owner", "failed_terminal"))
def test_profile_ledger_preserves_manual_prose_and_withholds_unbound_counts(profile_ledger, fault):
    root, log, state_path, receipt_path = profile_ledger
    document = root / "ENDEAVOR.md"
    prose = f"# Research\n\nWave12 cloud milestone remains preserved.\n\n{ANCHOR}\n\nNext experiment.\n"
    document.write_text(prose)
    if fault == "log":
        log.write_text("replaced command log\n")
    elif fault == "owner":
        receipt = json.loads(receipt_path.read_text())
        receipt["owner"]["controller_start_ticks"] = "different"
        receipt_path.write_text(json.dumps(receipt))
    elif fault == "failed_terminal":
        state = json.loads(state_path.read_text())
        state.update(
            status="passed",
            exit_code=0,
            cleanup_exit_code=1,
            keeper_restore=0,
            validation_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        )
        state_path.write_text(json.dumps(state))
    update(document, root, datetime(2026, 11, 18, tzinfo=UTC))
    text = document.read_text()
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    if fault == "none":
        assert "Source-bound saved GPU commands: 1 / 12 exited zero" in text
        assert "passed_selected_lr_stage_profile_continuation_and_trace_gates" not in text
    else:
        assert "profiler progress counts are withheld" in text
        # A rejection must not leave previously appended progress as an accepted current count.
        current = "\n".join(selected_lr_stage_profile_lines(root))
        assert "Source-bound saved GPU commands: 1 / 12 exited zero" not in current


@pytest.fixture
def selected_lr_ledger(tmp_path):
    candidate = tmp_path / "candidate"
    configs = candidate / "experiment/configs"
    configs.mkdir(parents=True)
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for arm in SELECTED_LR_ARMS:
        config = {
            "experiment": {"arm": arm, "refresh_install_age": 1},
            "training": {
                "steps": 5120,
                "tokens": 2684354560,
                "batch_tokens": 524288,
                "world_size": 8,
                "seed": 0,
            },
            "runtime": {"compile": "inductor", "attention": "FA2", "head": "compiled"},
        }
        (configs / f"{arm}_0p5m_seed0.json").write_text(json.dumps(config))
    root = tmp_path / "output"
    binding_path, helpers, state_path, work, receipt_path = selected_lr_locations(root)
    helpers.mkdir(parents=True)
    binding_path.parent.mkdir(parents=True)
    (root / "data").mkdir()
    artifacts = {
        "archive": root / "source.tar",
        "cpu_receipt": root / "cpu.json",
        "controller_cpu_receipt": root / "controller-cpu.json",
        "preparation": candidate / SELECTED_LR_VALIDATION_PREPARATION,
        "data_receipt": root / "data/experiment_receipt.json",
        "predecessor_release": root / "run-preparation/wave10-launch-binding.json",
    }
    for path in artifacts.values():
        path.write_text("{}\n")
    helper = helpers / "run_selected_lr_validation.py"
    helper.write_text("original_source = True\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    planned = selected_lr_validation_plan(candidate, root, work)
    frozen = {
        "status": "frozen_pending_completed_wave10_and_selected_lr_gpu_validation",
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "node": "28.216.17.9",
        "work": str(work),
        "receipt": str(receipt_path),
        "environment": planned["environment"],
        "files_sha256": candidate_inventory(candidate),
        "helpers_sha256": {helper.name: digest(helper)},
        "plan": planned,
        "commands": [command for trial in planned["trials"] for command in trial["commands"]],
        **{key: str(artifacts[key]) for key in ("archive", "cpu_receipt", "controller_cpu_receipt")},
        **{f"{key}_sha256": digest(path) for key, path in artifacts.items()},
    }
    binding_path.write_text(json.dumps(frozen))
    return root, candidate, helper, state_path, receipt_path, planned


def test_selected_lr_ledger_preserves_prose_and_reports_frozen_unlaunched_binding(selected_lr_ledger):
    root, _, _, _, _, _ = selected_lr_ledger
    document = root / "ENDEAVOR.md"
    prose = f"# Research\n\nPrior numerical failure remains preserved.\n\n{ANCHOR}\n\nNext wave hypothesis.\n"
    document.write_text(prose)
    update(document, root, datetime(2026, 11, 17, tzinfo=UTC))
    text = document.read_text()
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    assert "Planned 12 eight-rank commands, 30 check flags and 100,663,296 validation tokens" in text
    assert "No saved controller intent or GPU execution" in text
    assert "exited zero" not in text


@pytest.mark.parametrize(
    "drift",
    ("none", "source", "helper", "log", "owner", "command", "failure", "trials_null", "checks_null", "commands_null"),
)
def test_selected_lr_ledger_reports_bound_command_prefix_and_withholds_changed_evidence(selected_lr_ledger, drift):
    root, candidate, helper, state_path, receipt_path, planned = selected_lr_ledger
    binding_path, helpers, _, work, _ = selected_lr_locations(root)
    digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    intent = {
        "status": "acknowledged",
        "binding_sha256": digest,
        "command": selected_lr_controller_command(root),
        "pid": 123,
        "start_ticks": "456",
    }
    (helpers.parent / "controller.json").write_text(json.dumps(intent))
    state = {
        "status": "validating",
        "binding_sha256": digest,
        "controller_pid": 123,
        "controller_start_ticks": "456",
        "candidate": str(candidate),
        "node": "28.216.17.9",
        "command": selected_lr_validator_command(root, candidate),
    }
    state_path.write_text(json.dumps(state))
    work.mkdir(parents=True)
    log = work / "muont_continuous.log"
    log.write_text("original GPU outcome\n")
    receipt = planned | {
        "status": "validating",
        "owner": {"controller_pid": 123, "controller_start_ticks": "456"},
        "gpu_commands": [
            {
                "trial": "muont",
                "command": planned["trials"][0]["commands"][0],
                "exit_code": 0,
                "log": str(log),
                "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            }
        ],
    }
    if drift == "source":
        (candidate / "experiment/configs/muont_0p5m_seed0.json").write_text("{}")
    elif drift == "helper":
        helper.write_text("changed_source = True\n")
    elif drift == "log":
        log.write_text("replaced GPU outcome\n")
    elif drift == "owner":
        receipt["owner"]["controller_start_ticks"] = "another-generation"
    elif drift == "command":
        receipt["gpu_commands"][0]["command"] = planned["trials"][0]["commands"][1]
    elif drift == "failure":
        receipt.update(status="needs_attention", error="RuntimeError: cached-root numerical reconstruction failed")
    elif drift == "trials_null":
        receipt["trials"] = None
    elif drift == "checks_null":
        receipt["trials"][0]["checks"] = None
    elif drift == "commands_null":
        receipt["gpu_commands"] = None
    receipt_path.write_text(json.dumps(receipt))
    text = "\n".join(selected_lr_validation_lines(root))
    if drift in ("none", "failure"):
        assert "1 / 12 exited zero; 0 failed" in text
        assert "0 / 30 true" in text
        assert "This is not fresh liveness" in text
        assert "Full-budget quality remains unmeasured" in text
        if drift == "failure":
            assert "Saved GPU receipt status `needs_attention`" in text
            assert "Saved GPU validation failure: `RuntimeError: cached-root numerical reconstruction failed`" in text
    else:
        assert "validation progress counts are withheld" in text
        assert "exited zero" not in text


@pytest.mark.parametrize("drift", ("none", "source", "log", "archive", "failed", "scope"))
@pytest.mark.parametrize("kind", ("denied", "release", "recovery"))
def test_selected_lr_production_ledger_verifies_saved_archive_and_preserves_failure(selected_lr_ledger, drift, kind):
    root, previous, _, _, _, _ = selected_lr_ledger
    candidate = root / f"candidates/{kind}"
    shutil.copytree(previous, candidate, symlinks=True)
    owner = "selected_lr_production_owner" if kind == "denied" else "selected_lr_release_owner"
    wave = "wave12" if kind == "recovery" else "wave11"
    additions = {
        "experiment/train.py": "def train(): pass\n",
        f"experiment/{owner}.py": "def owner(): raise RuntimeError('denied')\n",
        f"experiment/tests/test_{owner}.py": "def test_denial(): pass\n",
        "experiment/cluster.json": json.dumps({"wave_name": wave}) + "\n",
    }
    for name, content in additions.items():
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    if drift == "scope":
        (candidate / "experiment/unrelated.py").write_text("unrelated = True\n")
        additions["experiment/unrelated.py"] = "unrelated = True\n"
    files = candidate_inventory(candidate)
    binding_path = selected_lr_locations(root)[0]
    prepared = {
        "status": (
            "selected_lr_production_prepared_gpu_denied" if kind == "denied" else "prepared_release_owner_gpu_pending"
        ),
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "files_sha256": files,
        "changed_files": sorted(additions),
        "validation_binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "helpers_sha256": {},
    }
    preparation_name = (
        "selected-lr-production-preparation.json" if kind == "denied" else "selected-lr-release-preparation.json"
    )
    prepared_path = candidate / preparation_name
    prepared_path.write_text(json.dumps(prepared))
    folder = {
        "denied": "selected-lr-production-cpu",
        "release": "selected-lr-release-cpu",
        "recovery": "selected-lr-recovery-cpu",
    }[kind]
    directory = root / f"run-preparation/{folder}/{kind}"
    directory.mkdir(parents=True)
    log, archive = directory / "candidate.log", directory / "source.tar"
    log.write_text("preserved CPU outcome\n")
    with tarfile.open(archive, "x") as bundle:
        for name in files:
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(prepared_path, arcname=prepared_path.name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    success = (
        "selected_lr_production_cpu_verified_gpu_denied"
        if kind == "denied"
        else "selected_lr_release_cpu_verified_gpu_pending"
    )
    failure = "selected_lr_production_cpu_failed" if kind == "denied" else "selected_lr_release_cpu_failed"
    receipt = prepared | {
        "status": failure if drift == "failed" else success,
        "exit_code": 1 if drift == "failed" else 0,
        "preparation_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        "command": cpu_command(),
        "cwd": str(candidate),
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))
    if drift == "source":
        (candidate / "experiment/train.py").write_text("modified = True\n")
    elif drift == "log":
        log.write_text("replaced outcome\n")
    elif drift == "archive":
        archive.write_bytes(b"replaced archive")
    text = "\n".join(selected_lr_production_cpu_lines(root))
    if drift in ("none", "failed"):
        assert f"### Wave-{wave.removeprefix('wave')} Selected-LR Production CPU" in text
        assert receipt["status"] in text
        assert f"integer exit `{receipt['exit_code']}`" in text
        assert "original log/archive verified" in text
        assert "CPU evidence grants no GPU allocation" in text
    else:
        assert "Outcome counts withheld" in text
        assert "integer exit" not in text


@pytest.mark.parametrize("wave", ("wave11", "wave12"))
@pytest.mark.parametrize("drift", ("none", "launch_binding"))
def test_selected_lr_release_ledger_reads_each_frozen_wave_and_withholds_other_launches(tmp_path, wave, drift):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    helpers = tmp_path / f"controllers/{wave}-launch-20261117/tracking"
    helpers.mkdir(parents=True)
    helper = helpers / "selected_lr_run_history.py"
    helper.write_text(
        "import json\n"
        "def binding(root):\n"
        f"    return json.loads((root / 'run-preparation/{wave}-launch-binding.json').read_text())\n"
    )
    release_path = preparation / f"{wave}-launch-binding.json"
    archive = preparation / "source.tar"
    archive.write_bytes(b"fixture archive")
    release_path.write_text(
        json.dumps(
            {
                "helpers": str(helpers),
                "helpers_sha256": {helper.name: hashlib.sha256(helper.read_bytes()).hexdigest()},
                "status": "frozen_test_release",
                "files_sha256": {"experiment/train.py": "test-source"},
                "archive": str(archive),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    )
    launch = {
        "binding_sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
        "status": "launched",
        "workers": {"candidate": {"pid": 123}},
    }
    if drift == "launch_binding":
        launch["binding_sha256"] = "other-release"
    (preparation / f"{wave}-launch-state.json").write_text(json.dumps(launch))
    other_wave = "wave12" if wave == "wave11" else "wave11"
    (preparation / f"{other_wave}-launch-state.json").write_text("{}\n")
    lines, accepted = selected_lr_release_lines(tmp_path, wave)
    text = "\n".join(lines)
    assert f"### Wave-{wave.removeprefix('wave')} Selected-LR Production Release" in text
    if drift == "none":
        assert accepted
        assert "Saved original launch `launched`; 1 worker acknowledgments." in text
    else:
        assert not accepted
        assert "Release/launch counts withheld." in text
        assert "worker acknowledgments" not in text


def test_refresh_preserves_prose_and_distinguishes_interim_from_completed(tmp_path):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Research\n\nManual hypothesis.\n\n{ANCHOR}\n\nManual next decision.\n"
    document.write_text(prose)
    (preparation / "wave4.json").write_text(
        json.dumps(
            {
                "time": "2026-11-16T09:40:00+00:00",
                "launches": [{"out": str(tmp_path / "runs/wave4/candidate")}],
            }
        )
    )
    monitor = preparation / "monitor-wave4"
    monitor.mkdir()
    snapshot = {
        "time": "2026-11-16T10:12:14+00:00",
        "status": "active",
        "runs": [
            {
                "name": "candidate",
                "node": "node",
                "status": "running",
                "latest_metrics": {"step": 700},
                "planned_steps": 2560,
                "keeper_watcher": {"status": "watching"},
            }
        ],
    }
    (monitor / "latest.json").write_text(json.dumps(snapshot))
    evaluation = {"mean_late_delta": -0.01, "late_tokens": [524288000], "final_delta": None}
    comparison = {
        "final_comparison_eligible": False,
        "training_windows": [{"delta": -0.02}],
        "evaluations": {"c4_loss": evaluation, "selection_loss": evaluation},
        "runtime": {
            "arm": {"steady_step_seconds_per_million_tokens": 2.2},
            "baseline": {"steady_step_seconds_per_million_tokens": 2.0},
            "delta": {"final_total_seconds": None},
        },
    }
    analysis = {
        "analyses": [
            {
                "baseline": "wave4/baseline",
                "window_tokens": 536870912,
                "comparisons": {"wave4/candidate": comparison},
            }
        ],
    }
    (preparation / "analysis-wave4.json").write_text(json.dumps(analysis))
    now = datetime(2026, 11, 16, 10, 20, tzinfo=UTC)
    update(document, tmp_path, now)
    interim = document.read_text()
    assert interim.split(START)[0] + interim.split(END)[1].lstrip("\n") == prose
    assert "interim; no final outcome" in interim
    assert "Final C4 delta" not in interim
    assert "No complete-wave verified SwanLab receipt" in interim
    assert "2026-11-16 18:12:14" in interim
    assert "Keeper watcher for `candidate`: `watching`." in interim
    snapshot["status"] = "complete"
    snapshot["runs"][0].update(status="complete", latest_metrics={"step": 2560})
    snapshot["runs"][0]["keeper_watcher"] = {"status": "restored"}
    (monitor / "latest.json").write_text(json.dumps(snapshot))
    comparison["final_comparison_eligible"] = True
    evaluation["final_delta"] = -0.005
    comparison["runtime"]["delta"]["final_total_seconds"] = 120
    (preparation / "analysis-wave4.json").write_text(json.dumps(analysis))
    (preparation / "swanlab-upload-wave4.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "verified_at": "2026-11-16T12:00:00+00:00",
                "runs": {"candidate": {"status": "verified"}},
            }
        )
    )
    update(document, tmp_path, now)
    completed = document.read_text()
    assert completed.count(START) == completed.count(END) == 1
    assert "2560 / 2560" in completed
    assert "Final C4 delta -0.005000" in completed
    assert "all 1 histories verified" in completed
    assert "Keeper watcher for `candidate`: `restored`." in completed
    assert completed.split(START)[0] + completed.split(END)[1].lstrip("\n") == prose


@pytest.mark.parametrize("entrypoint", ("ENDEAVOR.md", "DETAILS.md"))
def test_split_refresh_updates_research_and_details_without_changing_manual_decisions(tmp_path, entrypoint):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    research, details = tmp_path / "ENDEAVOR.md", tmp_path / "DETAILS.md"
    research_prose = (
        "# Research\n\nExecution ledger: [`DETAILS.md`](DETAILS.md).\n\n"
        "Rational continuation remains proposed and unlaunched.\n\n"
        f"{SUMMARY_ANCHOR}\n\nReserved seeds and original tolerances stay unchanged.\n"
    )
    details_prose = f"# Preserved receipts\n\nEarlier failure remains evidence.\n\n{ANCHOR}\n"
    research.write_text(research_prose)
    details.write_text(details_prose)
    for wave in (8, 9, 10):
        record(preparation / f"wave{wave}.json", {"time": "2026-11-18T01:00:00+00:00", "launches": []})
    now = datetime(2026, 11, 18, 12, 35, tzinfo=UTC)
    update(tmp_path / entrypoint, tmp_path, now)
    summary = research.read_text()
    ledger = details.read_text()
    assert summary.split(SUMMARY_START)[0] + summary.split(SUMMARY_END)[1].lstrip("\n") == research_prose
    assert ledger.split(START)[0] + ledger.split(END)[1].lstrip("\n") == details_prose
    assert START not in summary
    assert SUMMARY_START not in ledger
    assert "### wave8\n" in ledger and "### wave8\n" not in summary
    assert "### wave9\n" in summary and "### wave10\n" in summary
    assert "No complete-wave verified SwanLab receipt" in summary
    record(
        preparation / "wave12.json",
        {"time": "2026-11-18T02:00:00+00:00", "launches": [{"out": str(tmp_path / "runs/wave12/candidate")}]},
    )
    record(
        preparation / "swanlab-upload-wave12.json",
        {
            "status": "verified",
            "verified_at": "2026-11-18T03:00:00+00:00",
            "runs": {"candidate": {"status": "verified"}},
        },
    )
    update(tmp_path / entrypoint, tmp_path, now)
    summary = research.read_text()
    assert summary.count(SUMMARY_START) == summary.count(SUMMARY_END) == 1
    assert "### wave9\n" not in summary and "### wave12\n" in summary
    assert "No complete-wave verified SwanLab receipt" in summary
    assert "histories verified at 2026-11-18 11:00:00" not in summary
    assert summary.split(SUMMARY_START)[0] + summary.split(SUMMARY_END)[1].lstrip("\n") == research_prose


@pytest.mark.parametrize("fault", ("none", "changed_receipt"))
def test_recovery_refresh_retains_distinct_launch_and_withholds_changed_acceptance(tmp_path, fault):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    cpu_path = live / recovery_report.preparation.CPU_PATH
    record_value = json.loads(cpu_path.read_bytes())
    target = tmp_path / recovery_report.preparation.CPU_PATH
    target.parent.mkdir(parents=True)
    shutil.copy2(cpu_path, target)
    if fault == "changed_receipt":
        record(target, record_value | {"status": "forged accepted recovery"})
    if fault == "changed_receipt":
        text = "\n".join(recovery_report.lines(tmp_path, lambda value: value))
        assert "Recovery evidence withheld:" in text
        assert "103 passing cases" not in text
        return
    assert hashlib.sha256(target.read_bytes()).hexdigest() == recovery_report.CPU_SHA256
    text = "\n".join(recovery_report.lines(live, lambda value: value))
    assert "Recovery evidence withheld:" not in text
    assert "Recovery saved status:" in text
    assert "44/54 commands, 14/18 decisions" in text
    assert "does not establish current liveness" in text
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {recovery_report.TITLE}" in summary
    assert "103 passing cases" in summary


def test_exit_drain_report_retains_fresh_acceptance_and_withholds_changed_binding(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    text = "\n".join(exit_report.lines(live, lambda value: value))
    assert "evidence withheld:" not in text.lower()
    assert "125 passing cases" in text
    assert "/54 completed commands" in text
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {exit_report.TITLE}" in summary
    cpu = live / exit_report.CPU_PATH
    target = tmp_path / exit_report.CPU_PATH
    target.parent.mkdir(parents=True)
    shutil.copy2(cpu, target)
    record(target, json.loads(target.read_text()) | {"status": "substituted"})
    refused = "\n".join(exit_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()


def test_transition_report_preserves_accepted_launch_and_denies_changed_cpu_receipt(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    text = "\n".join(transition_report.lines(live, lambda value: value))
    assert "evidence withheld:" not in text.lower()
    assert "129 passing cases" in text
    assert "Transition saved status:" in text
    assert "exit-drain6/54commands2/18decisions" in text
    assert "does not establish current liveness" in text
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {transition_report.TITLE}" in summary
    target = tmp_path / transition_report.CPU_PATH
    target.parent.mkdir(parents=True)
    shutil.copy2(live / transition_report.CPU_PATH, target)
    record(target, json.loads(target.read_text()) | {"status": "substituted"})
    refused = "\n".join(transition_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "129 passing cases" not in refused


def test_scan_exit_report_authenticates_distinct_progress_and_withholds_changed_acceptance(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    saved = scan_exit_report.report_module(live).evidence(live)
    assert len(saved["cpu"]["helpers_sha256"]) == 178
    assert len(saved["cpu"]["inventory_sha256"]) == 547
    assert saved["sealed"]["retained_summary"]["completed_commands"] == 23
    assert saved["sealed"]["retained_summary"]["case_decisions"] == 7
    assert saved["progress"]["completed_commands"] <= 54
    text = "\n".join(scan_exit_report.lines(live, lambda value: value))
    assert "evidence withheld:" not in text.lower()
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {scan_exit_report.TITLE}" in summary
    assert "transition23/54commands7/18decisions" in summary
    target = tmp_path / scan_exit_report.CPU_PATH
    target.parent.mkdir(parents=True)
    shutil.copy2(live / scan_exit_report.CPU_PATH, target)
    record(target, json.loads(target.read_text()) | {"status": "substituted"})
    refused = "\n".join(scan_exit_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "129 passing cases" not in refused


def test_recheck_exit_report_preserves_failed_predecessor_and_refuses_changed_acceptance(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    saved = recheck_exit_report.report_module(live).evidence(live)
    assert len(saved["cpu"]["helpers_sha256"]) == 183
    assert len(saved["cpu"]["inventory_sha256"]) == 552
    assert saved["sealed"]["retained_summary"]["completed_commands"] == 8
    assert saved["sealed"]["retained_summary"]["case_decisions"] == 2
    assert saved["progress"]["completed_commands"] <= 54
    text = "\n".join(recheck_exit_report.lines(live, lambda value: value))
    assert "evidence withheld:" not in text.lower()
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {recheck_exit_report.TITLE}" in summary
    assert "scan-exit8/54commands2/18decisions" in summary
    target = tmp_path / recheck_exit_report.CPU_PATH
    target.parent.mkdir(parents=True)
    shutil.copy2(live / recheck_exit_report.CPU_PATH, target)
    record(target, json.loads(target.read_text()) | {"status": "substituted"})
    refused = "\n".join(recheck_exit_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "136 passing cases" not in refused


def test_recheck_exit_waiter_report_preserves_observation_and_refuses_substituted_intent(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    saved = recheck_exit_followup_report.evidence(live)
    assert saved["pid"] == 3559219 and saved["start_ticks"] == "943421918"
    assert saved["status"] in ("waiting_for_original_owners", "terminal_reconstructed", "needs_attention")
    text = "\n".join(recheck_exit_followup_report.lines(live, lambda value: value))
    assert "evidence withheld:" not in text.lower()
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {recheck_exit_followup_report.TITLE}" in summary
    target = tmp_path / recheck_exit_followup_report.preparation.WORK / "intent.json"
    target.parent.mkdir(parents=True)
    shutil.copy2(live / recheck_exit_followup_report.preparation.WORK / "intent.json", target)
    cpu = tmp_path / recheck_exit_followup_report.CPU
    cpu.parent.mkdir(parents=True)
    shutil.copy2(live / recheck_exit_followup_report.CPU, cpu)
    record(target, json.loads(target.read_text()) | {"pid": 999999})
    refused = "\n".join(recheck_exit_followup_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "waiting_for_original_owners" not in refused


def test_serialized_report_retains_cpu_acceptance_and_withholds_changed_receipt(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    lines = serialized_report.lines(live, lambda value: value)
    assert any("141" in line and "passing cases" in line for line in lines)
    summary = research_summary("\n".join(lines) + "\n" + END, datetime.now(UTC))
    assert f"### {serialized_report.TITLE}" in summary
    path = tmp_path / serialized_report.CPU_PATH
    path.parent.mkdir(parents=True)
    changed = json.loads((live / serialized_report.CPU_PATH).read_text())
    changed["exit_code"] = 1
    path.write_text(json.dumps(changed))
    assert any("withheld" in line for line in serialized_report.lines(tmp_path, lambda value: value))


def test_query_serialized_report_retains_acceptance_and_withholds_changed_receipt(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    lines = query_serialized_report.lines(live, lambda value: value)
    assert any("142" in line and "passing cases" in line for line in lines)
    summary = research_summary("\n".join(lines) + "\n" + END, datetime.now(UTC))
    assert f"### {query_serialized_report.TITLE}" in summary
    path = tmp_path / query_serialized_report.CPU_PATH
    path.parent.mkdir(parents=True)
    changed = json.loads((live / query_serialized_report.CPU_PATH).read_text())
    changed["exit_code"] = 1
    path.write_text(json.dumps(changed))
    assert any("withheld" in line for line in query_serialized_report.lines(tmp_path, lambda value: value))


def test_serialized_waiter_report_retains_observation_and_refuses_substituted_receipt(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    saved = serialized_followup_report.evidence(live)
    intent_path = live / serialized_followup_report.preparation.WORK / "intent.json"
    intent = json.loads(intent_path.read_text())
    assert (saved["pid"], saved["start_ticks"]) == (intent["pid"], intent["start_ticks"])
    text = "\n".join(serialized_followup_report.lines(live, lambda value: value))
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {serialized_followup_report.TITLE}" in summary
    target = tmp_path / serialized_followup_report.preparation.WORK / "intent.json"
    target.parent.mkdir(parents=True)
    shutil.copy2(intent_path, target)
    cpu = tmp_path / serialized_followup_report.CPU
    cpu.parent.mkdir(parents=True)
    shutil.copy2(live / serialized_followup_report.CPU, cpu)
    record(target, intent | {"pid": intent["pid"] + 1})
    refused = "\n".join(serialized_followup_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "waiting_for_original_owners" not in refused


def test_query_serialized_waiter_report_retains_observation_and_refuses_substituted_receipt(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    saved = query_serialized_followup_report.evidence(live)
    intent_path = live / query_serialized_followup_report.preparation.WORK / "intent.json"
    intent = json.loads(intent_path.read_text())
    assert (saved["pid"], saved["start_ticks"]) == (intent["pid"], intent["start_ticks"])
    assert saved["observations"] >= 1
    text = "\n".join(query_serialized_followup_report.lines(live, lambda value: value))
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {query_serialized_followup_report.TITLE}" in summary
    target = tmp_path / query_serialized_followup_report.preparation.WORK / "intent.json"
    target.parent.mkdir(parents=True)
    shutil.copy2(intent_path, target)
    cpu = tmp_path / query_serialized_followup_report.CPU
    cpu.parent.mkdir(parents=True)
    shutil.copy2(live / query_serialized_followup_report.CPU, cpu)
    record(target, intent | {"pid": intent["pid"] + 1})
    refused = "\n".join(query_serialized_followup_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "waiting_for_original_owners" not in refused


def test_scan_exit_waiter_report_retains_owner_observation_and_denies_changed_receipt(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    saved = scan_exit_followup_report.evidence(live)
    assert saved["pid"] == 3412036 and saved["start_ticks"] == "942807401"
    assert saved["observations"] >= 1
    assert saved["saved_original_owners"] == {
        "controller": "live_exact_generation",
        "validator": "live_exact_generation",
    }
    text = "\n".join(scan_exit_followup_report.lines(live, lambda value: value))
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {scan_exit_followup_report.TITLE}" in summary
    intent = tmp_path / scan_exit_followup_report.preparation.WORK / "intent.json"
    intent.parent.mkdir(parents=True)
    shutil.copy2(live / scan_exit_followup_report.preparation.WORK / "intent.json", intent)
    cpu = tmp_path / scan_exit_followup_report.CPU
    cpu.parent.mkdir(parents=True)
    shutil.copy2(live / scan_exit_followup_report.CPU, cpu)
    record(cpu, json.loads(cpu.read_text()) | {"status": "substituted"})
    refused = "\n".join(scan_exit_followup_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "Saved waiter status" not in refused


@pytest.mark.parametrize("fault", ("none", "changed_command", "unacknowledged"))
def test_transition_report_failed_command_retains_partial_coverage(tmp_path, fault):
    command = ["python", "training.py"]
    log = tmp_path / "validation/ns_training_path_transition_20261119/case_interrupted.log"
    log.parent.mkdir(parents=True)
    log.write_text("failed before training\n")
    entry = {
        "command": command,
        "log": str(log),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "spawn_status": "needs_attention" if fault == "unacknowledged" else "acknowledged",
        "exit_code": 1,
        "cleanup_exit_code": 1,
        "marked_descendants_absent": True,
    }
    if fault != "unacknowledged":
        entry["generation"] = {
            "pid": 200,
            "start_ticks": "201",
            "command": command,
            "parent": 100,
            "group": 200,
            "session": 200,
        }
    if fault == "changed_command":
        entry["command"] = ["replacement"]
    record(log.with_suffix(".execution.json"), entry)
    frozen = {"commands": [["completed"], command], "plan": {"trials": [{"name": "case"}]}}
    result = {"gpu_commands": [{"retained": True}]}
    if fault == "changed_command":
        with pytest.raises(ValueError):
            transition_report.failed_command_evidence(tmp_path, frozen, {"child_pid": 100}, result, {})
        return
    details = transition_report.failed_command_evidence(tmp_path, frozen, {"child_pid": 100}, result, {})
    assert len(result["gpu_commands"]) == 1
    assert details["mode"] == "interrupted"
    assert details["exit_code"] == 1
    assert details["marked_descendants_absent"] is True
    assert ("generation" in details) == (fault != "unacknowledged")


def test_exit_waiter_saved_report_authenticates_owner_and_withholds_changed_receipt(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    saved = exit_followup_report.evidence(live)
    assert saved["pid"] == 3071163 and saved["start_ticks"] == "941088367"
    assert saved["observations"] >= 1
    text = "\n".join(exit_followup_report.lines(live, lambda value: value))
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {exit_followup_report.TITLE}" in summary
    intent = tmp_path / exit_followup_report.preparation.WORK / "intent.json"
    intent.parent.mkdir(parents=True)
    shutil.copy2(live / exit_followup_report.preparation.WORK / "intent.json", intent)
    cpu = tmp_path / exit_followup_report.CPU
    cpu.parent.mkdir(parents=True)
    shutil.copy2(live / exit_followup_report.CPU, cpu)
    record(cpu, json.loads(cpu.read_text()) | {"status": "substituted"})
    refused = "\n".join(exit_followup_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "Saved waiter status" not in refused


def test_transition_waiter_report_preserves_observations_and_denies_changed_receipt(tmp_path):
    live = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    saved = transition_followup_report.evidence(live)
    assert saved["pid"] == 3127027 and saved["start_ticks"] == "941414079"
    assert saved["observations"] >= 1
    text = "\n".join(transition_followup_report.lines(live, lambda value: value))
    assert "evidence withheld:" not in text.lower()
    assert "175 helpers" in text
    summary = research_summary(text + "\n" + END, datetime.now(UTC))
    assert f"### {transition_followup_report.TITLE}" in summary
    intent = tmp_path / transition_followup_report.preparation.WORK / "intent.json"
    intent.parent.mkdir(parents=True)
    shutil.copy2(live / transition_followup_report.preparation.WORK / "intent.json", intent)
    cpu = tmp_path / transition_followup_report.CPU
    cpu.parent.mkdir(parents=True)
    shutil.copy2(live / transition_followup_report.CPU, cpu)
    record(cpu, json.loads(cpu.read_text()) | {"status": "substituted"})
    refused = "\n".join(transition_followup_report.lines(tmp_path, lambda value: value))
    assert "evidence withheld:" in refused.lower()
    assert "Saved waiter status" not in refused


def test_exit_drain_preparer_matching_tampered_copies_cannot_execute(tmp_path, monkeypatch):
    cpu = tmp_path / "receipt.json"
    helper = tmp_path / "tracking/prepare_ns_training_path_exit.py"
    local = tmp_path / "local/prepare_ns_training_path_exit.py"
    helper.parent.mkdir()
    local.parent.mkdir()
    accepted = b"VALUE = 7\n"
    helper.write_bytes(accepted)
    local.write_bytes(accepted)
    record(cpu, {"helpers_sha256": {local.name: hashlib.sha256(accepted).hexdigest()}})
    monkeypatch.setattr(exit_report, "CPU_SHA256", hashlib.sha256(cpu.read_bytes()).hexdigest())
    side_effect = tmp_path / "unexpected-execution"
    tampered = accepted + f"from pathlib import Path\nPath({str(side_effect)!r}).write_text('bad')\n".encode()
    helper.write_bytes(tampered)
    local.write_bytes(tampered)
    with pytest.raises(ValueError):
        exit_report.preparation_module(cpu, local)
    assert not side_effect.exists()


def test_exit_drain_preparer_executes_verified_buffer_after_source_replacement(tmp_path, monkeypatch):
    cpu = tmp_path / "receipt.json"
    helper = tmp_path / "tracking/prepare_ns_training_path_exit.py"
    local = tmp_path / "local/prepare_ns_training_path_exit.py"
    helper.parent.mkdir()
    local.parent.mkdir()
    accepted = b"VALUE = 7\n"
    helper.write_bytes(accepted)
    local.write_bytes(accepted)
    record(cpu, {"helpers_sha256": {local.name: hashlib.sha256(accepted).hexdigest()}})
    monkeypatch.setattr(exit_report, "CPU_SHA256", hashlib.sha256(cpu.read_bytes()).hexdigest())
    side_effect = tmp_path / "unexpected-execution"
    tampered = accepted + f"from pathlib import Path\nPath({str(side_effect)!r}).write_text('bad')\n".encode()
    original_open = Path.open

    def replaced_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == helper and args == ("rb",):
            local.write_bytes(tampered)
        return stream

    monkeypatch.setattr(Path, "open", replaced_open)
    implementation = exit_report.preparation_module(cpu, local)
    assert implementation.VALUE == 7
    assert local.read_bytes() == tampered
    assert not side_effect.exists()


@pytest.mark.parametrize(
    "fault", ("none", "controller_start", "validator_start", "validator_parent", "validator_command")
)
def test_exit_drain_saved_generation_substitution_refuses_projection(fault):
    command = ["python", "validator.py"]
    intent = {"pid": 100, "start_ticks": "101"}
    state = {
        "spawn_status": "acknowledged",
        "command": command,
        "child_pid": 200,
        "child_start_ticks": "201",
        "child_generation": {
            "pid": 200,
            "start_ticks": "201",
            "command": command,
            "parent": 100,
            "group": 200,
            "session": 200,
        },
    }
    result = {
        "owner": {
            "controller_pid": 100,
            "controller_start_ticks": "101",
            "validator_pid": 200,
            "validator_start_ticks": "201",
        }
    }
    if fault == "controller_start":
        result["owner"]["controller_start_ticks"] = "substituted"
    elif fault == "validator_start":
        state["child_generation"]["start_ticks"] = "substituted"
    elif fault == "validator_parent":
        state["child_generation"]["parent"] = 99
    elif fault == "validator_command":
        state["child_generation"]["command"] = ["replacement"]
    frozen = {"validator_command": command}
    if fault == "none":
        assert exit_report.checked_result(result, frozen, intent, state) == result
    else:
        with pytest.raises(ValueError):
            exit_report.checked_result(result, frozen, intent, state)


@pytest.mark.parametrize("fault", ("prelaunch", "reader_failure", "other_exit", "changed_command"))
def test_exit_drain_failure_details_do_not_claim_successful_cleanup(tmp_path, fault):
    log = tmp_path / "validation/ns_training_path_exit_drain_20261119/trial_interrupted.log"
    log.parent.mkdir(parents=True)
    command = ["python", "train.py"]
    frozen = {"commands": [["completed"], command], "plan": {"trials": [{"name": "trial"}]}}
    state = {"child_pid": 100}
    result = {"gpu_commands": [{"retained": True}]}
    if fault != "prelaunch":
        log.write_text("retained failure log\n")
        entry = {
            "command": command,
            "log": str(log),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "spawn_status": "acknowledged",
            "generation": {
                "pid": 200,
                "start_ticks": "201",
                "command": command,
                "parent": 100,
                "group": 200,
                "session": 200,
            },
            "time": "2026-11-19T08:00:00+00:00",
            "exit_code": 0 if fault == "reader_failure" else 2,
            "cleanup_exit_code": 0 if fault == "reader_failure" else 2,
            "marked_descendants_absent": fault == "reader_failure",
        }
        if fault == "changed_command":
            entry["command"] = ["replacement"]
        record(log.with_suffix(".execution.json"), entry)
    if fault == "changed_command":
        with pytest.raises(ValueError):
            exit_report.failed_command_evidence(tmp_path, frozen, state, result, {})
        return
    details = exit_report.failed_command_evidence(tmp_path, frozen, state, result, {})
    assert len(result["gpu_commands"]) == 1
    assert details["mode"] == "interrupted"
    if fault == "prelaunch":
        assert "exit_code" not in details and "marked_descendants_absent" not in details
    elif fault == "other_exit":
        assert details["exit_code"] == 2 and details["marked_descendants_absent"] is False
    else:
        assert details["exit_code"] == 0 and details["marked_descendants_absent"] is True


@pytest.mark.parametrize("fault", ("none", "changed_decisions", "missing_commands"))
def test_recovery_terminal_reporting_requires_complete_canonical_decisions(tmp_path, fault):
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    intent = {
        "binding_sha256": recovery_report.BINDING_SHA256,
        "pid": 910001,
        "start_ticks": "910101",
        "marker": "CPU-recovery-controller",
    }
    identity = {
        "binding_sha256": intent["binding_sha256"],
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "controller_marker": intent["marker"],
    }
    token = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = tmp_path / "run-preparation" / f"{recovery_report.source.NAMESPACE}-terminal-{token}.json"
    frozen = {
        "helpers": str(helpers / "tracking"),
        "helpers_sha256": {"ns_training_path_recovery_gpu_controller.py": "CPU-issued-controller"},
        "predecessor_acceptance": {"fixture": "retained incomplete original"},
    }
    attempt = {
        "status": "reconstruction_claimed",
        "output": str(path),
        **identity,
        "verifier_sha256": "CPU-issued-controller",
        "time": "2026-11-19T06:00:00+00:00",
    }
    record(helpers / "terminal-intent.json", attempt)
    attempt_digest = hashlib.sha256((helpers / "terminal-intent.json").read_bytes()).hexdigest()
    keeper = {
        "pid": 100,
        "world": 1,
        "generation": {"100": {"pid": 100, "status": "R"}},
        "devices": [["0", "CPU-UUID", "CPU"]],
        "occupied": [["100", "CPU-UUID"]],
    }
    state = {
        "status": "validation_complete",
        "result_sha256": "CPU-retained-results",
        "child_pid": 910002,
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "restored_keeper": keeper,
    }
    progress = {
        "completed_commands": 54,
        "decisions": list(range(18)),
        "accepted_cases": ["CPU-MuonT"],
        "rejected_cases": ["CPU-fused"],
    }
    terminal = {
        "status": recovery_report.TERMINAL_STATUS,
        "binding_sha256": intent["binding_sha256"],
        "result_sha256": state["result_sha256"],
        "reconstruction_intent_sha256": attempt_digest,
        "controller_pid_absent": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "controller_marker": intent["marker"],
        "validator_pid_absent": state["child_pid"],
        "verifier_sha256": "CPU-issued-controller",
        "predecessor_acceptance": frozen["predecessor_acceptance"],
        "production_authorized": False,
        "scope": recovery_report.source.SCOPE,
        "keeper": keeper,
        "accepted_cases": progress["accepted_cases"],
        "rejected_cases": progress["rejected_cases"],
        "all_cases_accepted": False,
        "evidence_sha256": {},
        "time": "2026-11-19T06:01:00+00:00",
    }
    if fault == "changed_decisions":
        terminal["rejected_cases"] = []
        terminal["all_cases_accepted"] = True
    elif fault == "missing_commands":
        progress["completed_commands"] = 44
    record(path, terminal)
    record(
        helpers / "terminal-outcome.json",
        {
            "status": "complete",
            "output": str(path),
            "reconstruction_intent_sha256": attempt_digest,
            "terminal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    if fault == "none":
        saved = recovery_report.terminal_evidence(tmp_path, frozen, intent, state, progress, {})
        assert saved["rejected_cases"] == ["CPU-fused"] and saved["production_authorized"] is False
    else:
        with pytest.raises(ValueError):
            recovery_report.terminal_evidence(tmp_path, frozen, intent, state, progress, {})


def test_split_refresh_refuses_both_replacements_when_research_boundary_is_ambiguous(tmp_path):
    (tmp_path / "run-preparation").mkdir()
    research, details = tmp_path / "ENDEAVOR.md", tmp_path / "DETAILS.md"
    research.write_text(f"[`DETAILS.md`](DETAILS.md)\n{SUMMARY_START}\n{SUMMARY_START}\n{SUMMARY_END}\n")
    details.write_text(f"# Historical evidence\n\n{ANCHOR}\n")
    before = {path: path.read_bytes() for path in (research, details)}
    with pytest.raises(ValueError, match="Ambiguous automated ledger markers"):
        update(details, tmp_path, datetime.now(UTC))
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "exit_code,evidence_error", [(0, None), (1, None), (124, "timeout sentinel"), (0, "archive drift")]
)
def test_refresh_records_owned_validation_cpu_outcomes_and_withholds_changed_source(tmp_path, exit_code, evidence_error):
    candidate = tmp_path / "candidate"
    experiment = candidate / "experiment"
    experiment.mkdir(parents=True)
    module = experiment / "owned_output_validation.py"
    module.write_text("observed = True\n")
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    output = tmp_path / "run-preparation/owned-output-validation-cpu/observation"
    output.mkdir(parents=True)
    predecessor = output / "predecessor.json"
    predecessor.write_text("{}")
    log, archive = output / "candidate.log", output / "source.tar"
    log.write_text("original CPU outcome\n")
    archive.write_bytes(b"preserved fixture")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    files = candidate_inventory(candidate)
    initial = {
        "status": "owned_output_validation_prepared_gpu_denied",
        "candidate": str(candidate),
        "predecessor_cpu_receipt": str(predecessor),
        "predecessor_cpu_receipt_sha256": digest(predecessor),
        "files_sha256": files,
        "predecessor_files_sha256": {},
        "changed_files": list(files),
    }
    preparation = candidate / "owned-output-validation-preparation.json"
    preparation.write_text(json.dumps(initial))
    record = {
        **initial,
        "time": "2026-11-17T05:43:00+00:00",
        "status": (
            "owned_output_validation_cpu_verified_gpu_denied"
            if exit_code == 0 and evidence_error is None
            else "owned_output_validation_cpu_failed"
        ),
        "preparation": str(preparation),
        "preparation_sha256": digest(preparation),
        "command": owned_output_cpu_command(),
        "cwd": str(candidate),
        "cuda_visible_devices": "",
        "exit_code": exit_code,
        "log": str(log),
        "log_sha256": digest(log),
        "archive": str(archive),
        "archive_sha256": digest(archive),
    }
    if evidence_error is not None:
        record["error"] = evidence_error
    receipt = output / "receipt.json"
    receipt.write_text(json.dumps(record))
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Owned follow-up\n\nOriginal hypothesis and failed evidence.\n\n{ANCHOR}\n"
    document.write_text(prose)
    now = datetime(2026, 11, 17, 5, 45, tzinfo=UTC)
    update(document, tmp_path, now)
    text = document.read_text()
    assert record["status"] in text
    assert f"integer exit `{exit_code}`" in text
    if evidence_error is not None:
        assert f"Recorded CPU evidence failure: `{evidence_error}`" in text
    assert "No GPU gate" in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    log.write_text("replaced result\n")
    update(document, tmp_path, now)
    assert "CPU outcome differs; summaries withheld" in document.read_text()
    log.write_text("original CPU outcome\n")
    module.write_text("changed observer\n")
    update(document, tmp_path, now)
    assert "CPU outcome differs; summaries withheld" in document.read_text()
    assert f"integer exit `{exit_code}`" not in document.read_text()
    assert document.read_text().split(START)[0] + document.read_text().split(END)[1].lstrip("\n") == prose


def test_refresh_records_ownership_only_cpu_evidence_and_withholds_archive_or_predecessor_drift(tmp_path):
    previous, candidate = tmp_path / "previous", tmp_path / "candidate"
    for source in (previous, candidate):
        (source / "artifacts/pinned-sources-materialized").mkdir(parents=True)
        (source / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
        for name in HANDOFF_CHANGES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if source == previous else '{"ownership":"successor"}')
    original_path = tmp_path / "run-preparation/owned-output-full-model-binding.json"
    original_path.parent.mkdir()
    original_path.write_text(json.dumps({"candidate": str(previous), "files_sha256": candidate_inventory(previous)}))
    out = tmp_path / "run-preparation/owned-output-handoff-cpu/observation"
    out.mkdir(parents=True)
    helper = out / "prepare_owned_output_handoff.py"
    helper.write_text("preserved preparer\n")
    log = out / "candidate.log"
    log.write_text("original CPU success\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    initial = {
        "status": "owned_output_handoff_prepared_gpu_denied",
        "root": str(tmp_path),
        "candidate": str(candidate),
        "original_binding_sha256": digest(original_path),
        "original_candidate": str(previous),
        "files_sha256": candidate_inventory(candidate),
        "changed_files": sorted(HANDOFF_CHANGES),
        "preparation_helper_sha256": digest(helper),
    }
    preparation = candidate / HANDOFF_PREPARATION
    preparation.write_text(json.dumps(initial))
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in initial["files_sha256"]:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(preparation, arcname=f"candidate/{HANDOFF_PREPARATION}", recursive=False)
        bundle.add(helper, arcname="tracking/prepare_owned_output_handoff.py", recursive=False)
    saved_archive = archive.read_bytes()
    record = {
        **initial,
        "status": "owned_output_handoff_cpu_verified_gpu_denied",
        "exit_code": 0,
        "cuda_visible_devices": "",
        "preparation": str(preparation),
        "preparation_sha256": digest(preparation),
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "log": str(log),
        "log_sha256": digest(log),
        "command": owned_output_cpu_command(),
        "cwd": str(candidate),
    }
    receipt = out / "receipt.json"
    receipt.write_text(json.dumps(record))
    text = "\n".join(owned_output_handoff_cpu_lines(tmp_path))
    assert record["status"] in text
    assert "exactly three ownership changes bound" in text
    archive.write_bytes(saved_archive + b"drift")
    text = "\n".join(owned_output_handoff_cpu_lines(tmp_path))
    assert "predecessor or CPU outcome differs; summaries withheld" in text
    archive.write_bytes(saved_archive)
    (previous / "experiment/execution_owner.py").write_text("changed original ownership\n")
    assert record["status"] not in "\n".join(owned_output_handoff_cpu_lines(tmp_path))


@pytest.mark.parametrize("identity", ("owned-output-full-model", "owned-output-handoff", "owned-output-trace"))
def test_refresh_tracks_unlaunched_owned_binding_then_only_bound_saved_command_progress(tmp_path, identity):
    preparation = tmp_path / "run-preparation"
    helpers = tmp_path / "controllers/owned/tracking"
    helpers.mkdir(parents=True)
    controller = {
        "owned-output-full-model": "run_owned_output_validation.py",
        "owned-output-handoff": "run_owned_output_handoff.py",
        "owned-output-trace": "run_owned_output_trace.py",
    }[identity]
    helper = helpers / controller
    helper.write_text("controller = True\n")
    candidate = tmp_path / "candidate"
    experiment = candidate / "experiment"
    (experiment / "configs").mkdir(parents=True)
    (experiment / "owned_output_validation.py").write_text("observed = True\n")
    (experiment / "configs/outer_0p5m_seed0.json").write_text("{}")
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    preparation.mkdir()
    cpu = preparation / "cpu.json"
    cpu.write_text("{}")
    archive = preparation / "source.tar"
    archive.write_bytes(b"fixture source")
    data = tmp_path / "data/experiment_receipt.json"
    data.parent.mkdir()
    data.write_text("{}")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    files = candidate_inventory(candidate)
    frozen = {
        "status": "frozen_pending_completed_ceiling_and_owned_gpu_validation",
        "root": str(tmp_path),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": {helper.name: digest(helper)},
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "cpu_receipt": str(cpu),
        "cpu_receipt_sha256": digest(cpu),
        "data_receipt_sha256": digest(data),
        "files_sha256": files,
        "source_sha256": {"experiment/owned_output_validation.py": files["experiment/owned_output_validation.py"]},
        "commands": [["python", f"trajectory-{index}"] for index in range(14)],
        "required_gates": ["cpu_numerics", "gpu_owned_output_factors"],
        "arms": ["outer"],
        "environment": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
    }
    binding = preparation / f"{identity}-binding.json"
    binding.write_text(json.dumps(frozen))
    other = "owned-output-handoff" if identity == "owned-output-full-model" else "owned-output-full-model"
    foreign_state = preparation / f"{other}-state.json"
    foreign_state.write_text(json.dumps({"status": "passed", "binding_sha256": digest(binding), "controller_pid": 999}))
    foreign_receipt = preparation / f"validation-{other}-gpu.json"
    foreign_receipt.write_text(json.dumps({"checks": {"gpu_owned_output_factors": True}}))
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Original research\n\n{ANCHOR}\n"
    document.write_text(prose)
    now = datetime(2026, 11, 17, 5, 45, tzinfo=UTC)
    update(document, tmp_path, now)
    assert "remains unlaunched" in document.read_text()
    assert "14 planned commands" in document.read_text()
    assert "PID `999`" not in document.read_text()
    assert "exited zero" not in document.read_text()
    if identity != "owned-output-full-model":
        intent = {
            "status": "acknowledged",
            "command": [
                "/opt/conda/envs/torch-base/bin/python",
                str(helper),
                "--root",
                str(tmp_path),
                "--phase",
                "validate",
            ],
            "candidate": str(candidate),
            "binding_sha256": digest(binding),
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
            "pid": 123,
        }
        (helpers.parent / "controller.json").write_text(json.dumps(intent))
        update(document, tmp_path, now)
        assert "one-shot controller intent `acknowledged`" in document.read_text()
        assert "detached startup remains unverified" in document.read_text()
        assert "remains unlaunched" not in document.read_text()
    state = {
        "status": "validating",
        "controller_pid": 123,
        "controller_start_ticks": "456",
        "binding_sha256": digest(binding),
    }
    (preparation / f"{identity}-state.json").write_text(json.dumps(state))
    log = preparation / "synthetic.log"
    log.write_text("original synthetic result\n")
    receipt = {
        "owner": {"controller_pid": 123, "controller_start_ticks": "456"},
        "candidate_path": str(candidate),
        "source_sha256": frozen["source_sha256"],
        "data_receipt_sha256": digest(data),
        "config_sha256": {"outer": files["experiment/configs/outer_0p5m_seed0.json"]},
        "environment": frozen["environment"],
        "checks": {"cpu_numerics": True, "gpu_owned_output_factors": False},
        "evidence": {
            "gpu_commands": [
                {"command": frozen["commands"][0], "exit_code": 0, "log": str(log), "log_sha256": digest(log)}
            ]
        },
    }
    receipt_path = preparation / f"validation-{identity}-gpu.json"
    receipt_path.write_text(json.dumps(receipt))
    update(document, tmp_path, now)
    assert "1 / 14 exited zero; 0 failed" in document.read_text()
    assert "not fresh liveness" in document.read_text().lower()
    state.update(status="needs_attention", error="owned factor exact equality failed", keeper_restore=0)
    (preparation / f"{identity}-state.json").write_text(json.dumps(state))
    update(document, tmp_path, now)
    assert "Recorded validation failure: `owned factor exact equality failed`" in document.read_text()
    assert "Recorded `keeper_restore`: `0`" in document.read_text()
    assert "needs_attention" in document.read_text()
    receipt["owner"]["controller_start_ticks"] = "reused"
    receipt_path.write_text(json.dumps(receipt))
    update(document, tmp_path, now)
    assert "progress counts withheld" in document.read_text()
    assert "1 / 14 exited zero" not in document.read_text()
    helper.write_text("changed controller\n")
    update(document, tmp_path, now)
    assert "binding identity differs" in document.read_text()
    assert "PID `123`" not in document.read_text()
    assert document.read_text().split(START)[0] + document.read_text().split(END)[1].lstrip("\n") == prose


def test_split_refresh_strips_error_line_whitespace_without_changing_manual_prose(tmp_path, monkeypatch):
    research, details = tmp_path / "ENDEAVOR.md", tmp_path / "DETAILS.md"
    research_prose = f"# Research  \n\n[`DETAILS.md`](DETAILS.md)\n\n{SUMMARY_ANCHOR}\n"
    details_prose = f"# Receipts  \n\n{ANCHOR}\n"
    research.write_text(research_prose)
    details.write_text(details_prose)
    payload = (
        f"{START}\n### NS Transition Actual-LR Queue\n"
        "Preserved queue failure: `RuntimeError: missing terminal\n"
        "    \n  File original_reader.py, line 46  \nFileNotFoundError: terminal.json`\n"
        f"{END}"
    )
    monkeypatch.setattr("update_endeavor.render", lambda root, now: payload)

    update(details, tmp_path, datetime.now(UTC))

    for document, start, end, manual in (
        (research, SUMMARY_START, SUMMARY_END, research_prose),
        (details, START, END, details_prose),
    ):
        refreshed = document.read_text()
        block = refreshed.split(start)[1].split(end)[0]
        assert "RuntimeError: missing terminal" in block
        assert "  File original_reader.py, line 46\nFileNotFoundError: terminal.json" in block
        assert all(line == line.rstrip() for line in block.splitlines())
        assert refreshed.split(start)[0] + refreshed.split(end)[1].lstrip("\n") == manual


def test_refresh_refuses_ambiguous_ledger_without_modifying_document(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    original = f"Manual prose\n{START}\n{START}\n{END}\n{ANCHOR}\n"
    document.write_text(original)
    with pytest.raises(ValueError, match="Ambiguous"):
        update(document, tmp_path, datetime.now(UTC))
    assert document.read_text() == original


def test_document_monitor_retries_failed_refresh_on_next_900_second_cycle(tmp_path):
    task = tmp_path / "task"
    tracking = task / "tracking"
    tracking.mkdir(parents=True)
    monitor = tracking / "monitor_endeavor.sh"
    monitor.write_text((Path(__file__).parent / monitor.name).read_text())
    document = task / "DETAILS.md"
    document.write_text("Manual prose\n")
    updater = tracking / "update_endeavor.py"
    updater.write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--root', type=Path)\n"
        "parser.add_argument('--document', type=Path)\n"
        "args = parser.parse_args()\n"
        "attempt = args.root / 'attempt'\n"
        "if not attempt.exists():\n"
        "    attempt.write_text(args.document.read_text())\n"
        "    raise SystemExit(7)\n"
        "args.document.write_text(args.document.read_text() + 'Refreshed ledger\\n')\n"
    )
    root = tmp_path / "output"
    root.mkdir()
    sleep = tmp_path / "sleep"
    sleep.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$1\" >> {tmp_path / 'delays'}\n"
        f"[[ -f {tmp_path / 'slept'} ]] && exit 73\n"
        f"touch {tmp_path / 'slept'}\n"
    )
    sleep.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", 'PATH="$1:$PATH" exec bash "$2" "$3"', "monitor-test", str(tmp_path), str(monitor), str(root)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 73
    assert (root / "attempt").read_text() == "Manual prose\n"
    assert document.read_text() == "Manual prose\nRefreshed ledger\n"
    assert (tmp_path / "delays").read_text().splitlines() == ["900", "900"]
    assert (root / "run-preparation/monitor-endeavor/stopped_at").is_file()


@pytest.mark.parametrize("wave", ("wave9", "wave10"))
@pytest.mark.parametrize("failed", (False, True))
def test_production_preparation_refresh_preserves_prose_and_withholds_changed_source(tmp_path, wave, failed):
    preparation = tmp_path / "run-preparation"
    owned = wave == "wave10"
    directory = preparation / (
        "owned-output-production-cpu/observation" if owned else "upstream-production-cpu/observation"
    )
    directory.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    experiment = candidate / "experiment"
    (experiment / "tests").mkdir(parents=True)
    owner_source = "owned_production_owner.py" if owned else "production_owner.py"
    for name in ("execution_owner.py", owner_source, "train.py"):
        (experiment / name).write_text("pass\n")
    (experiment / "cluster.json").write_text("{}\n")
    owner_test = "test_owned_output_release.py" if owned else "test_upstream_release.py"
    (experiment / f"tests/{owner_test}").write_text("pass\n")
    materialized = candidate / "artifacts/pinned-sources-materialized"
    materialized.mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    files = candidate_inventory(candidate)
    changed = {
        "experiment/execution_owner.py",
        f"experiment/{owner_source}",
        "experiment/cluster.json",
        f"experiment/tests/{owner_test}",
    }
    previous = {name: "previous" if name in changed else digest for name, digest in files.items()}
    previous.pop(f"experiment/{owner_source}")
    numerical_path = preparation / (
        "owned-output-trace-binding.json" if owned else "upstream-full-model-corrected-binding.json"
    )
    numerical_path.write_text(
        json.dumps(
            {
                "status": "unissued_validation_fixture",
                "root": str(tmp_path),
                "candidate": str(candidate),
                "files_sha256": previous,
                "archive": str(directory / "unissued.tar"),
                "archive_sha256": "unissued",
                "cpu_receipt": str(directory / "unissued.json"),
                "cpu_receipt_sha256": "unissued",
                "data_receipt_sha256": "unissued",
                "helpers": str(directory),
                "helpers_sha256": {},
                "commands": [],
            }
        )
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    prepared_path = candidate / "production-preparation.json"
    prepared_path.write_text(
        json.dumps(
            {
                "status": "prepared_owner_only_no_production_release",
                "candidate": str(candidate),
                "validation_binding_sha256": digest(numerical_path),
                "files_sha256": files,
                "changed_files": sorted(changed),
                "owner_sha256": files[f"experiment/{owner_source}"],
                "test_sha256": files[f"experiment/tests/{owner_test}"],
            }
        )
    )
    log, archive = directory / "candidate.log", directory / "source.tar"
    log.write_text("118 passed\n")
    if owned:
        with tarfile.open(archive, "w") as bundle:
            for name in files:
                bundle.add(candidate / name, arcname=name, recursive=False)
            bundle.add(prepared_path, arcname="production-preparation.json", recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    else:
        archive.write_bytes(b"source archive fixture")
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "production_cpu_failed" if failed else "production_cpu_verified_gpu_release_pending",
                "candidate": str(candidate),
                "preparation_sha256": digest(prepared_path),
                "files_sha256": files,
                "command": cpu_command(),
                "cwd": str(candidate),
                "cuda_visible_devices": "",
                "exit_code": 1 if failed else 0,
                "log": str(log),
                "log_sha256": digest(log),
                "archive": str(archive),
                "archive_sha256": digest(archive),
            }
        )
    )
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Research\n\nPrepared is not launched.\n\n{ANCHOR}\n\nOpen work.\n"
    document.write_text(prose)
    update(document, tmp_path, datetime.now(UTC))
    refreshed = document.read_text()
    if failed:
        assert "Saved CPU failure: integer exit `1`" in refreshed
        assert "CPU command exited zero" not in refreshed
    else:
        assert "CPU command exited zero with CUDA hidden" in refreshed
    assert f"No saved {'Wave-10' if owned else 'Wave-9'} production release or launch" in refreshed
    assert refreshed.split(START)[0] + refreshed.split(END)[1].lstrip("\n") == prose
    (experiment / "train.py").write_text("raise RuntimeError('changed numerical source')\n")
    update(document, tmp_path, datetime.now(UTC))
    changed_document = document.read_text()
    assert "preparation summaries withheld" in changed_document
    assert "CPU command exited zero" not in changed_document
    assert changed_document.split(START)[0] + changed_document.split(END)[1].lstrip("\n") == prose


@pytest.fixture(params=("wave9", "wave10", "wave12"))
def production_release_record(tmp_path, request):
    wave = request.param
    identity = (
        "wave11-selected-lr-validation"
        if wave == "wave12"
        else "owned-output-trace" if wave == "wave10" else "upstream-full-model-corrected"
    )
    gpu = "validation-wave11-selected-lr-gpu.json" if wave == "wave12" else f"validation-{identity}-gpu.json"
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    artifacts = {}
    for name in (
        f"{identity}-binding.json",
        gpu,
        f"{identity}-state.json",
        "prepared.json",
        "cpu.json",
        "source.tar",
    ):
        path = preparation / name
        path.write_text("{}\n")
        artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    helpers = tmp_path / (f"controllers/{wave}-launch-20261117/tracking" if wave == "wave12" else "helpers")
    helpers.mkdir(parents=True)
    helper = helpers / "controller.py"
    helper.write_text("pass\n")
    release_path = preparation / f"{wave}-launch-binding.json"
    helper_hashes = {"controller.py": hashlib.sha256(helper.read_bytes()).hexdigest()}
    if wave == "wave12":
        reader = helpers / "selected_lr_run_history.py"
        reader.write_text(
            "import json\n"
            "def binding(root):\n"
            "    return json.loads((root/'run-preparation/wave12-launch-binding.json').read_text())\n"
        )
        helper_hashes[reader.name] = hashlib.sha256(reader.read_bytes()).hexdigest()
    release = {
        "status": (
            "frozen_selected_lr_production_after_original_gpu_validation"
            if wave == "wave12"
            else "frozen_validated_launch_release"
        ),
        "helpers": str(helpers),
        "helpers_sha256": helper_hashes,
        "files_sha256": {"experiment/train.py": "test-source"},
        "archive": str(preparation / "source.tar"),
        "archive_sha256": artifacts["source.tar"],
        "validation_binding_sha256": artifacts[f"{identity}-binding.json"],
        "validation_sha256": artifacts[gpu],
        "state_sha256": artifacts[f"{identity}-state.json"],
        "preparation": str(preparation / "prepared.json"),
        "preparation_sha256": artifacts["prepared.json"],
        "cpu_receipt": str(preparation / "cpu.json"),
        "cpu_receipt_sha256": artifacts["cpu.json"],
        "monitor_interval_seconds": 900,
        "arms": ["muont", "legacy", "upstream", "null"],
    }
    if wave == "wave12":
        failed = preparation / "wave11-launch-binding.json"
        failed.write_text("{}\n")
        release["recovery_of"] = str(failed)
        release["failure_evidence"] = {
            "run-preparation/wave11-launch-binding.json": hashlib.sha256(failed.read_bytes()).hexdigest()
        }
    release_path.write_text(json.dumps(release))
    return wave, release_path, helper


@pytest.mark.parametrize("production_release_record", ("wave9", "wave10"), indirect=True)
def test_partial_production_launch_reports_preserved_failure_not_completed_training(tmp_path, production_release_record):
    wave, release_path, _ = production_release_record
    preparation = tmp_path / "run-preparation"
    launch = {
        "status": "needs_attention",
        "binding_sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
        "workers": {"muont": {"status": "running"}},
        "error": "SSH acknowledgment timed out",
    }
    launch_path = preparation / f"{wave}-launch-state.json"
    launch_path.write_text(json.dumps(launch))
    lines = "\n".join(production_lines(tmp_path, wave))
    assert "1 / 4 worker acknowledgments" in lines
    assert "Original intents are preserved without retry" in lines
    launch["binding_sha256"] = "changed"
    launch_path.write_text(json.dumps(launch))
    withheld = "\n".join(production_lines(tmp_path, wave))
    assert "worker acknowledgments withheld" in withheld
    assert "1 / 4 worker acknowledgments" not in withheld


@pytest.mark.parametrize("changed", ("snapshot", "analysis", "release_helper", "release_absent"))
def test_production_refresh_withholds_misbound_progress_results_and_uploads(
    tmp_path, production_release_record, changed
):
    wave, release_path, helper = production_release_record
    release_digest = hashlib.sha256(release_path.read_bytes()).hexdigest()
    preparation = tmp_path / "run-preparation"
    (preparation / f"{wave}.json").write_text(
        json.dumps({"launches": [{"out": str(tmp_path / f"runs/{wave}/upstream")}]})
    )
    monitor = preparation / f"monitor-{wave}"
    monitor.mkdir()
    snapshot = {
        "binding_sha256": release_digest,
        "status": "complete",
        "runs": [
            {
                "name": "upstream",
                "node": "node",
                "status": "complete",
                "latest_metrics": {"step": 5120},
                "planned_steps": 5120,
                "keeper_watcher": {"status": "restored"},
            }
        ],
    }
    snapshot_path = monitor / "latest.json"
    snapshot_path.write_text(json.dumps(snapshot))
    evaluation = {"mean_late_delta": -0.123456, "late_tokens": [2684354560], "final_delta": -0.123456}
    analysis = {
        "source_release_sha256": release_digest,
        "analyses": [
            {
                "baseline": f"{wave}/muont",
                "window_tokens": 536870912,
                "comparisons": {
                    f"{wave}/upstream": {
                        "final_comparison_eligible": True,
                        "training_windows": [{"delta": -0.123456}],
                        "evaluations": {"c4_loss": evaluation, "selection_loss": evaluation},
                        "runtime": {
                            "arm": {"steady_step_seconds_per_million_tokens": 2.2},
                            "baseline": {"steady_step_seconds_per_million_tokens": 2.0},
                            "delta": {"final_total_seconds": 120},
                        },
                    }
                },
            }
        ],
    }
    analysis_path = preparation / f"analysis-{wave}.json"
    analysis_path.write_text(json.dumps(analysis))
    (preparation / f"swanlab-upload-{wave}.json").write_text(
        json.dumps({"status": "verified", "runs": {"upstream": {"status": "verified"}}})
    )
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Research\n\nManual wave interpretation.\n\n{ANCHOR}\n"
    document.write_text(prose)
    update(document, tmp_path, datetime.now(UTC))
    current = document.read_text()
    assert "5120 / 5120" in current
    assert "Keeper watcher for `upstream`: `restored`." in current
    assert "full-budget eligible" in current
    assert "536,870,912-token training-window deltas: -0.123456" in current
    assert "Final C4 delta -0.123456" in current
    assert "all 1 histories verified" in current
    if changed == "snapshot":
        snapshot["binding_sha256"] = "different release"
        snapshot_path.write_text(json.dumps(snapshot))
    elif changed == "analysis":
        analysis["source_release_sha256"] = "different release"
        analysis_path.write_text(json.dumps(analysis))
    elif changed == "release_helper":
        helper.write_text("raise RuntimeError('changed release helper')\n")
    else:
        release_path.unlink()
    update(document, tmp_path, datetime.now(UTC))
    withheld = document.read_text()
    if changed != "analysis":
        assert "monitor identity differs; progress and keeper summaries withheld" in withheld
        assert "5120 / 5120" not in withheld
        assert "Keeper watcher for `upstream`" not in withheld
    else:
        assert "5120 / 5120" in withheld
    if changed != "snapshot":
        assert "analysis identity differs; loss and runtime summaries withheld" in withheld
        assert "full-budget eligible" not in withheld
        assert "-0.123456" not in withheld
        assert "Steady seconds/MTok" not in withheld
    else:
        assert "Final C4 delta -0.123456" in withheld
    if changed in ("release_helper", "release_absent"):
        assert "all 1 histories verified" not in withheld
    assert withheld.split(START)[0] + withheld.split(END)[1].lstrip("\n") == prose


def test_upstream_cpu_refresh_withholds_changed_candidate_without_altering_manual_prose(tmp_path):
    preparation = tmp_path / "run-preparation"
    directory = preparation / "upstream-outer-cpu/observation"
    directory.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    experiment = candidate / "experiment"
    experiment.mkdir(parents=True)
    module = experiment / "prototype.py"
    module.write_text("def update():\n    return 1\n")
    materialized = candidate / "artifacts/pinned-sources-materialized"
    materialized.mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    initial = candidate / "preparation.json"
    initial.write_text("{}")
    archive = directory / "source.tar"
    archive.write_bytes(b"fixture archive")
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    helper = helpers / "prototype.py"
    helper.write_text(module.read_text())
    log = directory / "cpu.log"
    log.write_text("CPU result fixture\n")
    for name in ("wave8-launch-binding.json", "wave8-candidate.json"):
        (preparation / name).write_text("{}")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    record = {
        "status": "cpu_verified_gpu_execution_denied",
        "candidate": str(candidate),
        "initial_preparation": str(initial),
        "initial_preparation_sha256": digest(initial),
        "predecessor_release_sha256": digest(preparation / "wave8-launch-binding.json"),
        "predecessor_candidate_sha256": digest(preparation / "wave8-candidate.json"),
        "files_sha256": {"experiment/prototype.py": digest(module)},
        "helpers": str(helpers),
        "helpers_sha256": {"prototype.py": digest(helper)},
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "commands": [{"exit_code": 0, "log": str(log), "log_sha256": digest(log)}] * 2,
        "arms": ["outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused"],
        "cuda_visible_devices": "",
    }
    (directory / "receipt.json").write_text(json.dumps(record))
    document = tmp_path / "ENDEAVOR.md"
    original = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(original)
    now = datetime.now(UTC)
    update(document, tmp_path, now)
    text = document.read_text()
    assert "cpu_verified_gpu_execution_denied" in text
    assert "null_leftdiag_upstream_fused" in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == original
    module.write_text("def update():\n    return 2\n")
    update(document, tmp_path, now)
    text = document.read_text()
    assert "CPU source or log identity differs" in text
    assert "cpu_verified_gpu_execution_denied" not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == original
    log.unlink()
    update(document, tmp_path, now)
    assert "CPU source or log identity differs" in document.read_text()


@pytest.fixture
def full_model_observer_record(tmp_path):
    root = tmp_path
    output = root / "run-preparation/upstream-full-model-cpu/observation"
    output.mkdir(parents=True)
    candidates = []
    for name in ("prototype", "successor"):
        candidate = root / name
        (candidate / "experiment").mkdir(parents=True)
        (candidate / "experiment/update.py").write_text("def update():\n    return 1\n")
        (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
        (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
        candidates.append(candidate)
    prototype, successor = candidates
    receipt = output / "prototype.json"
    receipt.write_text("{}")
    initial = successor / "full-model-preparation.json"
    initial.write_text("{}")
    archive = output / "source.tar"
    archive.write_bytes(b"cpu fixture")
    helpers = output / "helpers"
    helpers.mkdir()
    helper = helpers / "observer.py"
    helper.write_text("def check():\n    return 1\n")
    candidate_log = output / "candidate.log"
    candidate_log.write_text("candidate CPU fixture\n")
    observer_log = output / "observer.log"
    observer_log.write_text("observer CPU fixture\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    record = {
        "status": "full_model_observer_cpu_verified_gpu_execution_denied",
        "candidate": str(successor),
        "prototype_candidate": str(prototype),
        "prototype_cpu_receipt": str(receipt),
        "prototype_cpu_receipt_sha256": digest(receipt),
        "initial_preparation": str(initial),
        "initial_preparation_sha256": digest(initial),
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "helpers": str(helpers),
        "helpers_sha256": {"observer.py": digest(helper)},
        "files_sha256": {"experiment/update.py": digest(successor / "experiment/update.py")},
        "prototype_files_sha256": {"experiment/update.py": digest(prototype / "experiment/update.py")},
        "commands": [
            {
                "command": [
                    "/opt/conda/envs/torch-base/bin/python",
                    "-m",
                    "pytest",
                    "-c",
                    "/dev/null",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "-m",
                    "not slow and not integration and not data_integration and not cluster "
                    "and not requires_cluster and not docker and not manual",
                    test,
                ],
                "cwd": str(cwd),
                "exit_code": 0,
                "log": str(log),
                "log_sha256": digest(log),
            }
            for cwd, test, log in (
                (successor, "experiment/tests", candidate_log),
                (helpers.parent, "tracking/test_upstream_full_model_checks.py", observer_log),
            )
        ],
        "arms": ["muont", "outer_leftdiag_fused", "outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused"],
        "cuda_visible_devices": "",
    }
    (output / "receipt.json").write_text(json.dumps(record))
    document = root / "ENDEAVOR.md"
    prose = f"# Research\n\nManual next experiment.\n\n{ANCHOR}\n"
    document.write_text(prose)
    return root, output / "receipt.json", record, document, prose


def test_full_model_observer_ledger_withholds_changed_prototype_and_preserves_manual_prose(full_model_observer_record):
    root, _, record, document, prose = full_model_observer_record
    update(document, root, datetime.now(UTC))
    text = document.read_text()
    assert record["status"] in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    (Path(record["prototype_candidate"]) / "experiment/update.py").write_text("def update():\n    return 2\n")
    update(document, root, datetime.now(UTC))
    text = document.read_text()
    assert "observer CPU source or log identity differs" in text
    assert record["status"] not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose


@pytest.mark.parametrize("corruption", ["duplicate", "cwd", "argv", "log"])
def test_full_model_observer_ledger_withholds_duplicate_or_misbound_commands(full_model_observer_record, corruption):
    root, receipt, record, document, prose = full_model_observer_record
    if corruption == "duplicate":
        record["commands"][1] = record["commands"][0].copy()
    elif corruption == "cwd":
        record["commands"][1]["cwd"] = record["candidate"]
    elif corruption == "argv":
        record["commands"][1]["command"][-1] = "experiment/tests"
    else:
        Path(record["commands"][1]["log"]).write_text("observer evidence changed\n")
    receipt.write_text(json.dumps(record))
    update(document, root, datetime.now(UTC))
    text = document.read_text()
    assert "observer CPU source or log identity differs" in text
    assert record["status"] not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose


@pytest.mark.parametrize(
    "fault",
    [None, "owner", "command", "log", "candidate", "missing_log", "missing_hash", "initial", "failed", "all_commands"],
)
def test_full_model_ledger_reports_bound_progress_and_withholds_changed_evidence(tmp_path, fault):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    candidate = tmp_path / "candidate"
    (candidate / "experiment/configs").mkdir(parents=True)
    source = candidate / "experiment/validate.py"
    source.write_text("def validate():\n    return True\n")
    config = candidate / "experiment/configs/muont_0p5m_seed0.json"
    config.write_text("{}")
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    data = tmp_path / "data/experiment_receipt.json"
    data.parent.mkdir()
    data.write_text("{}")
    archive = preparation / "source.tar"
    archive.write_bytes(b"fixture")
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    helper = helpers / "controller.py"
    helper.write_text("def control():\n    return True\n")
    log = preparation / "synthetic.log"
    log.write_text("successful command\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    frozen = {
        "candidate": str(candidate),
        "commands": [["synthetic"], ["continuous"], ["interrupted"], ["resumed"]],
        "required_gates": ["cpu_numerics", "gpu_collector", "checkpoint_continuation"],
        "data_receipt_sha256": digest(data),
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "helpers": str(helpers),
        "helpers_sha256": {helper.name: digest(helper)},
        "environment": {"NCCL_INSPECTOR_ENABLE": "0"},
        "source_sha256": {"experiment/validate.py": digest(source)},
        "files_sha256": {
            "experiment/validate.py": digest(source),
            "experiment/configs/muont_0p5m_seed0.json": digest(config),
        },
        "arms": ["muont"],
    }
    binding = preparation / "upstream-full-model-corrected-binding.json"
    binding.write_text(json.dumps(frozen))
    state = {"controller_pid": 1234, "controller_start_ticks": "5678", "binding_sha256": digest(binding)}
    receipt = {
        "time": "2026-11-17T00:35:00+00:00",
        "owner": {"controller_pid": 1234, "controller_start_ticks": "5678"},
        "candidate_path": str(candidate),
        "environment": frozen["environment"],
        "source_sha256": frozen["source_sha256"],
        "data_receipt_sha256": digest(data),
        "config_sha256": {"muont": digest(config)},
        "checks": {"cpu_numerics": True, "gpu_collector": True, "checkpoint_continuation": False},
        "evidence": {
            "gpu_commands": [{"command": ["synthetic"], "log": str(log), "log_sha256": digest(log), "exit_code": 0}]
        },
    }
    receipt_path = preparation / "validation-upstream-full-model-corrected-gpu.json"
    receipt_path.write_text(json.dumps(receipt))
    if fault == "owner":
        receipt["owner"]["controller_start_ticks"] = "reused"
    elif fault == "command":
        receipt["evidence"]["gpu_commands"][0]["command"] = ["unplanned"]
    elif fault == "log":
        log.write_text("changed log\n")
    elif fault == "candidate":
        source.write_text("def validate():\n    return False\n")
    elif fault == "missing_log":
        del receipt["evidence"]["gpu_commands"][0]["log"]
    elif fault == "missing_hash":
        del receipt["evidence"]["gpu_commands"][0]["log_sha256"]
    elif fault == "initial":
        del receipt["evidence"]["gpu_commands"]
    elif fault == "failed":
        receipt["evidence"]["gpu_commands"][0]["exit_code"] = 1
    elif fault == "all_commands":
        receipt["evidence"]["gpu_commands"] = [
            {"command": command, "log": str(log), "log_sha256": digest(log), "exit_code": 0}
            for command in frozen["commands"]
        ]
    receipt_path.write_text(json.dumps(receipt))
    text = "\n".join(upstream_full_model_validation_lines(tmp_path, binding, frozen, state))
    if fault in (None, "initial", "failed", "all_commands"):
        assert {
            None: "1 / 4 exited zero; 0 failed",
            "initial": "0 / 4 exited zero; 0 failed",
            "failed": "0 / 4 exited zero; 1 failed",
            "all_commands": "4 / 4 exited zero; 0 failed",
        }[fault] in text
        assert "2 / 3 true (including CPU checks)" in text
        assert "independent tensor reconstruction" in text
        assert "production-launch approval" in text
    else:
        assert "progress counts withheld" in text
        assert "exited zero" not in text


def test_upstream_gpu_ledger_distinguishes_frozen_owned_and_unbound_evidence(tmp_path):
    preparation = tmp_path / "run-preparation"
    directory = preparation / "upstream-outer-gpu-observation"
    helpers = directory / "helpers"
    helpers.mkdir(parents=True)
    helper = helpers / "probe.py"
    helper.write_text("def probe():\n    return 1\n")
    candidate = tmp_path / "candidate"
    (candidate / "experiment").mkdir(parents=True)
    module = candidate / "experiment/prototype.py"
    module.write_text(helper.read_text())
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    campaign = tmp_path / "campaigns/upper/campaign.json"
    campaign.parent.mkdir(parents=True)
    campaign.write_text("{}")
    release = preparation / "wave8-launch-binding.json"
    release.write_text("{}")
    archive = directory / "source.tar"
    archive.write_bytes(b"fixture archive")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    files = {"experiment/prototype.py": digest(module)}
    cpu_path = preparation / "cpu.json"
    cpu_path.write_text(json.dumps({"candidate": str(candidate), "files_sha256": files}))
    execution = campaign.parent / "execution-binding.json"
    execution.write_text(
        json.dumps(
            {
                "campaign": str(campaign),
                "campaign_sha256": digest(campaign),
                "helpers_sha256": {"probe.py": digest(helper)},
            }
        )
    )
    frozen = {
        "time": "2026-11-16T21:25:45+00:00",
        "root": str(tmp_path),
        "out": str(directory),
        "status": "frozen_waiting_upper_lr_completion_not_launch_gate",
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": digest(cpu_path),
        "candidate_files_sha256": files,
        "campaign": str(campaign),
        "campaign_sha256": digest(campaign),
        "execution_binding_sha256": digest(execution),
        "predecessor_release_sha256": digest(release),
        "helpers": str(helpers),
        "helpers_sha256": {"probe.py": digest(helper)},
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "plan": {"cpu_receipt_sha256": digest(cpu_path), "cases": 54, "shapes": [[16, 512, 512]] * 3},
    }
    binding = directory / "binding.json"
    binding.write_text(json.dumps(frozen))
    lines = "\n".join(upstream_gpu_lines(tmp_path, binding))
    assert "54 production-shape cases; 6 eager/CUDA-graph workloads" in lines
    assert "remains unlaunched" in lines
    state = {
        "status": "measured_upstream_probe_not_launch_gate",
        "binding_sha256": digest(binding),
        "controller_pid": 123,
        "controller_start_ticks": "456",
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
    }
    result = {
        "status": "passed_upstream_production_diagnostic_not_launch_gate",
        "owner": {"controller_pid": 123, "controller_start_ticks": "456"},
        "binding_sha256": digest(binding),
        "helpers_sha256": frozen["helpers_sha256"],
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result))
    state["result_sha256"] = digest(result_path)
    (directory / "state.json").write_text(json.dumps(state))
    lines = "\n".join(upstream_gpu_lines(tmp_path, binding))
    assert "Saved controller-bound result" in lines
    assert "Recorded `keeper_restore`: `0`" in lines
    assert "does not independently rederive numerical parity" in lines
    result["owner"]["controller_pid"] = 789
    result_path.write_text(json.dumps(result))
    state["result_sha256"] = digest(result_path)
    (directory / "state.json").write_text(json.dumps(state))
    lines = "\n".join(upstream_gpu_lines(tmp_path, binding))
    assert "result ownership or hash differs" in lines
    assert "Saved controller-bound result" not in lines
    helper.write_text("def probe():\n    return 2\n")
    lines = "\n".join(upstream_gpu_lines(tmp_path, binding))
    assert "source or binding identity differs" in lines
    assert "measured_upstream_probe_not_launch_gate" not in lines
    assert "Recorded `keeper_restore`" not in lines


def test_boundary_queue_ledger_tracks_waiting_and_exit_but_withholds_changed_binding(tmp_path):
    directory = tmp_path / "campaigns/boundary"
    queue = directory / "queue"
    helpers = queue / "helpers"
    helpers.mkdir(parents=True)
    campaign = directory / "campaign.json"
    campaign.write_text("{}")
    validation = directory / "validation-binding.json"
    validation.write_text("{}")
    executor = helpers / "executor.py"
    executor.write_text("fixture = True\n")
    helper_hash = hashlib.sha256(executor.read_bytes()).hexdigest()
    execution = directory / "execution-binding.json"
    execution.write_text(json.dumps({"helpers_sha256": {"executor.py": helper_hash}}))
    predecessor = tmp_path / "run-preparation/wave8-launch-binding.json"
    predecessor.parent.mkdir()
    predecessor.write_text("{}")
    for name in ("queue_lr_boundary.py", "queue_lr_boundary.sh"):
        (helpers / name).write_text("fixture = True\n")
    archive = queue / "source.tar"
    archive.write_bytes(b"fixture archive")
    frozen = {
        "status": "frozen_waiting_for_wave8",
        "campaign": str(campaign),
        "campaign_sha256": hashlib.sha256(campaign.read_bytes()).hexdigest(),
        "validation_binding_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "execution_binding_sha256": hashlib.sha256(execution.read_bytes()).hexdigest(),
        "predecessor_release_sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
        "helpers": str(helpers),
        "helpers_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in helpers.iterdir()},
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "interval_seconds": 900,
    }
    binding = queue / "binding.json"
    binding.write_text(json.dumps(frozen))
    digest = hashlib.sha256(binding.read_bytes()).hexdigest()
    (queue / "installation.json").write_text(
        json.dumps({"status": "acknowledged", "pid": 12345, "binding_sha256": digest})
    )
    (queue / "state.json").write_text(json.dumps({"status": "waiting_for_wave8", "binding_sha256": digest}))
    (queue / "validate.json").write_text(json.dumps({"status": "starting", "pid": 23456, "start_ticks": "4567"}))
    text = "\n".join(boundary_queue_lines(tmp_path, campaign))
    assert "waiting_for_wave8" in text
    assert "12345" in text and "23456" in text
    assert "not a fresh liveness check" in text
    (queue / "stopped_at").write_text("2026-11-17T04:30:00+08:00\n")
    assert "no live queue inferred" in "\n".join(boundary_queue_lines(tmp_path, campaign))
    executor.write_text("changed = True\n")
    withheld = "\n".join(boundary_queue_lines(tmp_path, campaign))
    assert "summaries withheld" in withheld
    assert "12345" not in withheld and "23456" not in withheld


def test_refresh_tracks_unlaunched_tuning_then_withholds_stale_result_identity(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    directory = tmp_path / "campaigns/lr"
    directory.mkdir(parents=True)
    path = directory / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "status": "prepared_unvalidated_not_queued",
                "time": "2026-11-16T11:22:00+00:00",
                "families": ["muont"],
            }
        )
    )
    analysis = {
        "status": "incomplete",
        "time": "2026-11-16T11:25:00+00:00",
        "campaign_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trials": [{"family": "muont", "complete": False}] * 3,
        "completed_family_lr_choices": {},
        "observed_tokens_by_family": {"muont": 0},
    }
    (directory / "analysis.json").write_text(json.dumps(analysis))
    now = datetime(2026, 11, 16, 11, 30, tzinfo=UTC)
    update(document, tmp_path, now)
    text = document.read_text()
    assert "prepared_unvalidated_not_queued" in text
    assert "| `muont` | 0 / 3 | 0 | none; incomplete family sweep |" in text
    assert "2026-11-16 19:25:00" in text
    analysis["status"] = "complete"
    analysis["trials"] = [{"family": "muont", "complete": True}] * 3
    analysis["completed_family_lr_choices"] = {"muont": {"trial": "muont_lr_base"}}
    analysis["observed_tokens_by_family"] = {"muont": 503316480}
    (directory / "analysis.json").write_text(json.dumps(analysis))
    update(document, tmp_path, now)
    assert "| `muont` | 3 / 3 | 503,316,480 | `muont_lr_base` |" in document.read_text()
    path.write_text(path.read_text() + "\n")
    update(document, tmp_path, now)
    stale = document.read_text()
    assert "Saved analysis does not match the campaign identity" in stale
    assert "muont_lr_base" not in stale


def tuning_result_fixture():
    trials = []
    for family in ("muont", "candidate"):
        for label, multiplier, lr in (("down", 0.5, 0.01), ("base", 1.0, 0.02), ("up", 2.0, 0.04)):
            complete = label == "down"
            candidate = family == "candidate"
            trials.append(
                {
                    "name": f"{family}_lr_{label}",
                    "family": family,
                    "lr_multiplier": multiplier,
                    "matrix_lr": lr,
                    "status": "complete" if complete else "running",
                    "complete": complete,
                    "observed_tokens": 100 if complete else (45 if candidate else 60),
                    "final_selection_loss": 4.1 if candidate else 4.0,
                    "mean_late_selection_loss": 4.2 if candidate else 4.0,
                    "mean_late_c4_loss_reporting_only": 4.3 if candidate else 4.0,
                    "training_windows": [
                        {"end_tokens": endpoint, "mean": 5.2 if candidate else 5.0}
                        for endpoint in (range(20, 101, 20) if complete else (20, 40, 60))
                    ],
                    "runtime": {"total_seconds": 110 if candidate else 100, "total_is_final": complete},
                }
            )
    campaign = {
        "selection": {"training_window_tokens": 20},
        "trials": [
            {
                **{key: trial[key] for key in ("name", "family", "matrix_lr", "lr_multiplier")},
                "tokens": 100,
            }
            for trial in trials
        ],
    }
    return campaign, {"trials": trials}


def test_tuning_results_pair_same_lr_complete_endpoints_and_shared_whole_windows():
    campaign, analysis = tuning_result_fixture()
    text = "\n".join(tuning_result_lines(campaign, analysis))
    assert "| `candidate_lr_down` | `complete` | 4.100000 | +0.200000 | +0.300000 | 110.00 | +10.00% |" in text
    assert "| `candidate_lr_base` | `running` | n/a | n/a | n/a | n/a | n/a |" in text
    assert "| `candidate_lr_down` | W1: +0.200000, W2: +0.200000, W3: +0.200000, W4: +0.200000, W5: +0.200000 |" in text
    assert "| `candidate_lr_base` | W1: +0.200000, W2: +0.200000 |" in text
    assert "C4 is reporting only" in text
    assert "not a fresh history read" in text
    assert "matching coverage is unverified" in text
    assert "not verified paired held-out estimates" in text


def test_tuning_results_require_matching_recorded_evaluation_tokens_when_available():
    campaign, analysis = tuning_result_fixture()
    for trial in analysis["trials"]:
        trial["late_evaluation_tokens"] = {"selection_loss": [60, 80, 100], "c4_loss": [60, 80, 100]}
    text = "\n".join(tuning_result_lines(campaign, analysis))
    assert "identical nonempty recorded evaluation-token lists" in text
    assert "matching coverage is unverified" not in text
    candidate = next(trial for trial in analysis["trials"] if trial["name"] == "candidate_lr_down")
    candidate["late_evaluation_tokens"]["selection_loss"] = [80, 100]
    candidate["late_evaluation_tokens"]["c4_loss"] = []
    text = "\n".join(tuning_result_lines(campaign, analysis))
    assert "| `candidate_lr_down` | `complete` | 4.100000 | n/a | n/a | 110.00 | +10.00% |" in text


def test_boundary_campaign_ledger_separates_cumulative_choices_from_new_grid(tmp_path):
    campaign, analysis = tuning_result_fixture()
    campaign.update(
        study="wave7_fixed_age_matrix_lr_upper_boundary",
        status="prepared_unvalidated_not_queued",
        families=["muont", "candidate"],
        initial_trials_per_family=2,
        initial_tokens_per_family=200,
        cumulative_tokens_per_family=500,
        parent_campaign=str(tmp_path / "campaigns/parent/campaign.json"),
    )
    directory = tmp_path / "campaigns/boundary"
    directory.mkdir(parents=True)
    path = directory / "campaign.json"
    path.write_text(json.dumps(campaign))
    analysis.update(
        status="incomplete",
        campaign_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        completed_family_lr_choices={},
        observed_tokens_by_family={"muont": 220, "candidate": 190},
        completed_cumulative_family_lr_choices={},
        cumulative_observed_tokens_by_family={"muont": 520, "candidate": 490},
    )
    (directory / "analysis.json").write_text(json.dumps(analysis))
    document = tmp_path / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    update(document, tmp_path, datetime.now(UTC))
    text = document.read_text()
    assert "| `muont` | 520 | none; incomplete five-point sweep | n/a |" in text
    assert "New-grid choices are not cumulative choices" in text
    assert "all 2 completed trials per family" in text
    assert "GPU validation and launch wait for complete Wave-8 analysis" in text


@pytest.mark.parametrize("mismatch", ["matrix_lr", "lr_multiplier"])
def test_tuning_results_withhold_unmatched_lr_deltas(mismatch):
    campaign, analysis = tuning_result_fixture()
    candidate = next(trial for trial in analysis["trials"] if trial["name"] == "candidate_lr_down")
    candidate[mismatch] *= 1.1
    next(trial for trial in campaign["trials"] if trial["name"] == candidate["name"])[mismatch] = candidate[mismatch]
    text = "\n".join(tuning_result_lines(campaign, analysis))
    assert "| `candidate_lr_down` | `complete` | 4.100000 | n/a | n/a | 110.00 | n/a |" in text
    assert "| `candidate_lr_down` | none shared; no value inferred |" in text


@pytest.mark.parametrize("mismatch", ["valid_lr_tuple", "family", "duplicate"])
def test_tuning_results_reject_trial_identity_differences_from_plan(mismatch):
    campaign, analysis = tuning_result_fixture()
    candidate = next(trial for trial in analysis["trials"] if trial["name"] == "candidate_lr_down")
    if mismatch == "valid_lr_tuple":
        candidate.update(matrix_lr=0.02, lr_multiplier=1.0)
    elif mismatch == "family":
        candidate["family"] = "muont"
    else:
        analysis["trials"][-1] = candidate.copy()
    text = "\n".join(tuning_result_lines(campaign, analysis))
    assert "trial identities differ from the campaign plan" in text
    assert "Final Selection |" not in text
    assert "W1:" not in text


@pytest.mark.parametrize("incomplete", ["candidate", "reference", "timer"])
def test_tuning_results_do_not_promote_failed_or_incomplete_endpoints(incomplete):
    campaign, analysis = tuning_result_fixture()
    candidate = next(trial for trial in analysis["trials"] if trial["name"] == "candidate_lr_down")
    reference = next(trial for trial in analysis["trials"] if trial["name"] == "muont_lr_down")
    if incomplete == "candidate":
        candidate["status"] = "failed"
        expected = "| `candidate_lr_down` | `failed` | n/a | n/a | n/a | n/a | n/a |"
    elif incomplete == "reference":
        reference["complete"] = False
        expected = "| `candidate_lr_down` | `complete` | 4.100000 | n/a | n/a | 110.00 | n/a |"
    else:
        candidate["runtime"]["total_is_final"] = False
        expected = "| `candidate_lr_down` | `complete` | 4.100000 | +0.200000 | +0.300000 | n/a | n/a |"
    assert expected in "\n".join(tuning_result_lines(campaign, analysis))


@pytest.mark.parametrize("mismatch", ["campaign_hash", "valid_lr_tuple", "family", "duplicate"])
def test_tuning_results_are_recurring_and_withheld_for_changed_campaign(tmp_path, mismatch):
    campaign, analysis = tuning_result_fixture()
    campaign.update(study="wave7_fixed_age_matrix_lr", status="prepared", families=["muont", "candidate"])
    directory = tmp_path / "campaigns/fixed-age"
    directory.mkdir(parents=True)
    path = directory / "campaign.json"
    path.write_text(json.dumps(campaign))
    analysis.update(
        status="incomplete",
        campaign_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        completed_family_lr_choices={},
        observed_tokens_by_family={"muont": 220, "candidate": 190},
    )
    (directory / "analysis.json").write_text(json.dumps(analysis))
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    update(document, tmp_path, datetime.now(UTC))
    assert "Final Selection | Late Selection Mean Delta" in document.read_text()
    assert document.read_text().split(START)[0] + document.read_text().split(END)[1].lstrip("\n") == prose
    if mismatch == "campaign_hash":
        path.write_text(path.read_text() + "\n")
    else:
        candidate = next(trial for trial in analysis["trials"] if trial["name"] == "candidate_lr_down")
        if mismatch == "valid_lr_tuple":
            candidate.update(matrix_lr=0.02, lr_multiplier=1.0)
        elif mismatch == "family":
            candidate["family"] = "muont"
        else:
            analysis["trials"][-1] = candidate.copy()
        analysis["completed_family_lr_choices"] = {"candidate": {"trial": "invalid_choice"}}
        (directory / "analysis.json").write_text(json.dumps(analysis))
    update(document, tmp_path, datetime.now(UTC))
    assert "Saved analysis does not match the campaign identity" in document.read_text()
    assert "Final Selection | Late Selection Mean Delta" not in document.read_text()
    assert "Complete / Prepared Trials" not in document.read_text()
    assert "invalid_choice" not in document.read_text()


def test_refresh_records_diagnostic_completion_without_promoting_launch_gate(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    directory = tmp_path / "validation/diagonal_diagnostic_20261116"
    directory.mkdir(parents=True)
    result_path = directory / "result.json"
    result_path.write_text(json.dumps({"status": "measured_diagnostic_not_launch_gate", "ranks": []}))
    (directory / "state.json").write_text(
        json.dumps(
            {
                "status": "measured_diagnostic_not_launch_gate",
                "time": "2026-11-16T12:20:00+00:00",
                "keeper_restore": 0,
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
        )
    )
    now = datetime(2026, 11, 16, 12, 25, tzinfo=UTC)
    update(document, tmp_path, now)
    text = document.read_text()
    assert "measured_diagnostic_not_launch_gate" in text
    assert "Recorded keeper restoration: `0`" in text
    assert "Result receipt: `PVC/validation/diagonal_diagnostic_20261116/result.json`." in text
    result_path.write_text(result_path.read_text() + "\n")
    update(document, tmp_path, now)
    assert "Result identity does not match" in document.read_text()
    assert "Result receipt:" not in document.read_text()


def test_refresh_keeps_failed_combined_probe_and_restoration_without_promoting_gate(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    release = preparation / "wave7-release-binding.json"
    release.write_text(json.dumps({"status": "frozen"}))
    guard = tmp_path / "campaigns/lr-fixed-age-0p5m-20261116/validation-binding.json"
    guard.parent.mkdir(parents=True)
    guard.write_text(json.dumps({"status": "frozen"}))
    directory = tmp_path / "validation/combined_outer_gpu_test"
    helpers = directory / "helpers"
    helpers.mkdir(parents=True)
    helper = helpers / "probe.py"
    helper.write_text("# Frozen probe\n")
    archive = directory / "source.tar"
    archive.write_bytes(b"frozen archive")
    digests = {helper.name: hashlib.sha256(helper.read_bytes()).hexdigest()}
    binding = directory / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "status": "frozen_production_probe_not_launch_gate",
                "root": str(tmp_path.resolve()),
                "out": str(directory.resolve()),
                "source_release_sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
                "original_guard_binding_sha256": hashlib.sha256(guard.read_bytes()).hexdigest(),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "helpers_sha256": digests,
            }
        )
    )
    owner = {"controller_pid": 1234, "controller_start_ticks": "5678"}
    result_path = directory / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "failed_production_parity_not_launch_gate",
                "owner": owner,
                "helpers_sha256": digests,
                "cases": [{"exact": {"ns_source": True}, "parameter_delta": {"within_tolerance": False}}],
                "timing": [],
            }
        )
    )
    (directory / "state.json").write_text(
        json.dumps(
            {
                "status": "needs_attention",
                "keeper_restore": 0,
                "binding_sha256": hashlib.sha256(binding.read_bytes()).hexdigest(),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                **owner,
            }
        )
    )
    update(document, tmp_path, datetime.now(UTC))
    text = document.read_text()
    assert "failed_production_parity_not_launch_gate" in text
    assert "0 / 1 recorded cases pass" in text
    assert "Recorded keeper restoration: `0`." in text
    assert "never full-model validation or an optimizer efficiency win" in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    result_path.write_text(result_path.read_text() + "\n")
    update(document, tmp_path, datetime.now(UTC))
    assert "Numerical result is not controller hash-bound" in document.read_text()
    assert "0 / 1 recorded cases pass" not in document.read_text()


def test_refresh_distinguishes_stopped_queue_execution_from_original_binding(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    preparation = tmp_path / "run-preparation"
    directory = preparation / "queue-wave5-launch"
    directory.mkdir(parents=True)
    (preparation / "wave5-launch-queue.json").write_text(
        json.dumps({"status": "queued_pending_validation_and_predecessors", "time": "2026-11-16T11:03:00+00:00"})
    )
    (preparation / "queue-wave5-launch.log").write_text(
        "2026-11-16T19:03:38+08:00 launch_status=waiting\n"
        "complete\nactive\n2026-11-16T19:48:59+08:00 launch_status=needs_attention\n"
    )
    (directory / "stopped_at").write_text("2026-11-16T19:48:59+08:00\n")
    update(document, tmp_path, datetime(2026, 11, 16, 12, 25, tzinfo=UTC))
    text = document.read_text()
    assert "Binding state: `queued_pending_validation_and_predecessors`" in text
    assert "Last execution event: `needs_attention` at 2026-11-16 19:48:59" in text
    assert "Queue exited at 2026-11-16 19:48:59" in text


def test_refresh_tracks_new_geometry_gates_without_promoting_failed_predecessor(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Manual research\n\nSeparate outer-only hypothesis.\n\n{ANCHOR}\n"
    document.write_text(prose)
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    for filename, state in (
        ("wave5-validation-state.json", {"status": "needs_attention", "keeper_restore": 0}),
        (
            "wave6-candidate.json",
            {
                "status": "frozen_pending_gpu_validation",
                "required_gates": ["cpu_numerics", "outer_diagonal_pipeline"],
            },
        ),
        ("wave6-validation-state.json", {"status": "validating"}),
        (
            "wave6-validation-preflight-missing-keeper-20261116T133349Z.json",
            {"status": "needs_attention", "error": "Missing keeper metadata"},
        ),
    ):
        (preparation / filename).write_text(json.dumps(state))
    receipt = preparation / "validation-wave6-gpu.json"
    receipt.write_text(
        json.dumps(
            {
                "checks": {"cpu_numerics": True, "outer_diagonal_pipeline": False},
                "evidence": {"gpu_commands": []},
            }
        )
    )
    now = datetime(2026, 11, 16, 12, 55, tzinfo=UTC)
    update(document, tmp_path, now)
    text = document.read_text()
    assert "### Wave-5 Validation\n\nReceipt state: `needs_attention`" in text
    assert "### Wave-6 Validation\n\nReceipt state: `validating`" in text
    assert "not evidence of GPU validation or training launch" in text
    assert "1 / 2 required gates passed" in text
    assert "Pending or failed gates: `outer_diagonal_pipeline`." in text
    assert "Recorded failure: `Missing keeper metadata`." in text
    assert "### wave6\n" not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    (preparation / "wave6-validation-state.json").write_text(json.dumps({"status": "passed", "keeper_restore": 0}))
    receipt.write_text(
        json.dumps(
            {
                "checks": {"cpu_numerics": True, "outer_diagonal_pipeline": True},
                "evidence": {"gpu_commands": [{"exit_code": 0}]},
            }
        )
    )
    update(document, tmp_path, now)
    completed = document.read_text()
    assert "### Wave-5 Validation\n\nReceipt state: `needs_attention`" in completed
    assert "### Wave-6 Validation\n\nReceipt state: `passed`" in completed
    assert "Recorded keeper restoration: `0`" in completed
    assert "2 / 2 required gates passed" in completed
    assert "Pending or failed gates: none." in completed
    assert "Completed GPU command receipts: 1; failed commands: 0." in completed
    assert "Recorded failure: `Missing keeper metadata`." in completed
    assert "### wave6\n" not in completed


def test_refresh_reports_saved_checkpoint_diagnostic_only_for_unchanged_validation(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    receipt = preparation / "validation-wave6-gpu.json"
    receipt.write_text(json.dumps({"checks": {"checkpoint_continuation": False}}))
    diagnostic = {
        "status": "cpu_checkpoint_diagnostic_not_launch_gate",
        "validation_receipts_sha256": {receipt.name: hashlib.sha256(receipt.read_bytes()).hexdigest()},
        "pairs": [
            {
                "step": step,
                "tensor_comparisons": 3,
                "changed_tensor_count": 1 if step == 24 else 0,
                "failed_tensor_count": 1 if step == 24 else 0,
                "scalar_or_structure_differences": [],
            }
            for step in (22, 24)
        ],
    }
    (preparation / "diagnostic-wave6-muont-continuation-20261116.json").write_text(json.dumps(diagnostic))
    update(document, tmp_path, datetime.now(UTC))
    text = document.read_text()
    assert "| 22 | 3 | 0 | 0 | 0 |" in text
    assert "| 24 | 3 | 1 | 1 | 0 |" in text
    receipt.write_text(receipt.read_text() + "\n")
    update(document, tmp_path, datetime.now(UTC))
    stale = document.read_text()
    assert "Validation receipt identities differ" in stale
    assert "| 24 |" not in stale


def test_refresh_separates_fixed_age_candidate_from_preserved_failed_validation(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    for wave, status in (("wave6", "needs_attention"), ("wave7", "validating")):
        (preparation / f"{wave}-candidate.json").write_text(
            json.dumps(
                {
                    "status": "frozen_pending_gpu_validation",
                    "required_gates": ["checkpoint_continuation"],
                    "refresh_install_age": 1,
                }
            )
        )
        (preparation / f"{wave}-validation-state.json").write_text(json.dumps({"status": status}))
        (preparation / f"validation-{wave}-gpu.json").write_text(
            json.dumps({"checks": {"checkpoint_continuation": False}, "evidence": {"gpu_commands": []}})
        )
    update(document, tmp_path, datetime.now(UTC))
    text = document.read_text()
    assert "### Wave-6 Validation\n\nReceipt state: `needs_attention`" in text
    assert "### Wave-7 Validation\n\nReceipt state: `validating`" in text
    assert "Bound installation age: `1` optimizer update." in text
    assert "`PVC/run-preparation/validation-wave7-gpu.json`." in text
    assert "### wave7\n" not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose


def test_refresh_records_unequal_refresh_prefixes_and_withholds_changed_source(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    release = preparation / "wave7-release-binding.json"
    release.write_text(json.dumps({"status": "frozen_pending_launch"}))
    diagnostic = {
        "status": "recorded_refresh_diagnostic",
        "time": "2026-11-16T15:13:10+00:00",
        "source_release_sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
        "common_observed_tokens": 24,
        "window_tokens": 32,
        "refresh_install_age": 1,
        "runs": {
            arm: {
                "all_observed": {"updates": updates, "events": events, "installed": events, "discarded": 0},
                "complete_shared_windows": [],
            }
            for arm, updates, events in (("muont", 4, 2), ("outer_leftdiag", 3, 1))
        },
    }
    (preparation / "analysis-wave7-refresh.json").write_text(json.dumps(diagnostic))
    update(document, tmp_path, datetime.now(UTC))
    text = document.read_text()
    assert "2026-11-16 23:13:10" in text
    assert "complete shared 32-token windows: 0" in text
    assert "| `muont` | 4 | 2 | 2 | 0 |" in text
    assert "| `outer_leftdiag` | 3 | 1 | 1 | 0 |" in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    release.write_text(release.read_text() + "\n")
    update(document, tmp_path, datetime.now(UTC))
    stale = document.read_text()
    assert "Saved diagnostic does not match the source-release identity" in stale
    assert "| `muont` | 4 | 2 | 2 | 0 |" not in stale


def test_refresh_distinguishes_frozen_tuning_validation_from_executed_and_stale_gates(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    directory = tmp_path / "campaigns/fixed-age"
    directory.mkdir(parents=True)
    (directory / "campaign.json").write_text(
        json.dumps({"status": "prepared_unvalidated_not_queued", "families": ["muont"]})
    )
    binding = directory / "validation-binding.json"
    binding.write_text(json.dumps({"status": "frozen_pending_completed_wave7_and_actual_gpu_validation"}))
    update(document, tmp_path, datetime.now(UTC))
    assert "No actual-config GPU validation execution is recorded." in document.read_text()
    receipt = directory / "validation-gpu.json"
    receipt.write_text(json.dumps({"trials": [{"checks": {"post_cached_root_continuation": True}}]}))
    (directory / "validation-state.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "keeper_restore": 0,
                "binding_sha256": hashlib.sha256(binding.read_bytes()).hexdigest(),
                "validation_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            }
        )
    )
    update(document, tmp_path, datetime.now(UTC))
    text = document.read_text()
    assert "Saved actual-config GPU gates: 1 / 1 trials passed." in text
    assert "Recorded validation keeper restoration: `0`." in text
    receipt.write_text(receipt.read_text() + "\n")
    update(document, tmp_path, datetime.now(UTC))
    stale = document.read_text()
    assert "GPU receipt is not fully hash-bound" in stale
    assert "1 / 1 trials passed" not in stale


@pytest.mark.parametrize("changed", ("owner", "command", "log", "helper"))
@pytest.mark.parametrize("status", ("validating", "needs_attention"))
@pytest.mark.parametrize("complete", (True, False))
def test_refresh_reports_provisional_validation_then_withholds_changed_evidence(tmp_path, changed, status, complete):
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    directory = tmp_path / "campaigns/lr"
    helpers = directory / "validation-helpers"
    helpers.mkdir(parents=True)
    helper = helpers / "validator.py"
    helper.write_text("# Frozen validator\n")
    archive = directory / "validation-source.tar"
    archive.write_bytes(b"frozen archive")
    campaign = directory / "campaign.json"
    campaign.write_text(json.dumps({"status": "prepared_unvalidated_not_queued", "families": ["muont"]}))
    trial = {
        "name": "muont_lr_down",
        "family": "muont",
        "config": "actual-config.json",
        "config_sha256": "actual-config-hash",
        "commands": [["validator", phase] for phase in ("continuous", "interrupted", "resumed")],
        "continuous": "continuous",
        "resumed": "resumed",
        "checks": {"boundary_continuation": False, "post_cached_root_continuation": False},
    }
    plan = {
        "status": "planned_not_executed",
        "campaign_sha256": hashlib.sha256(campaign.read_bytes()).hexdigest(),
        "trials": [trial],
        "gpu_commands": [],
    }
    binding = directory / "validation-binding.json"
    binding.write_text(
        json.dumps(
            {
                "status": "frozen_pending_validation",
                "campaign": str(campaign),
                "campaign_sha256": plan["campaign_sha256"],
                "helpers": str(helpers),
                "helpers_sha256": {helper.name: hashlib.sha256(helper.read_bytes()).hexdigest()},
                "archive": str(archive),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "plan": plan,
            }
        )
    )
    owner = {"controller_pid": 1234, "controller_start_ticks": "5678"}
    (directory / "validation-state.json").write_text(
        json.dumps({"status": status, "binding_sha256": hashlib.sha256(binding.read_bytes()).hexdigest(), **owner})
    )
    log = directory / "continuous.log"
    log.write_text("Successful actual-config command\n")
    receipt = {
        **plan,
        "status": status,
        "time": "2026-11-16T17:07:00+00:00",
        "owner": owner,
        "trials": [
            {
                **trial,
                "checks": dict.fromkeys(trial["checks"], complete),
                "boundary_continuation": {"tensor_comparisons": 2123, "max_absolute_error": 0.0},
                "post_cached_root_continuation": {"tensor_comparisons": 2135, "max_absolute_error": 0.0},
            }
        ],
        "gpu_commands": [
            {
                "trial": trial["name"],
                "command": command,
                "exit_code": 0,
                "log": str(log),
                "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            }
            for command in trial["commands"]
        ],
    }
    receipt_path = directory / "validation-gpu.json"
    receipt_path.write_text(json.dumps(receipt))
    update(document, tmp_path, datetime.now(UTC))
    text = document.read_text()
    assert f"3 / 3 commands exited zero; {int(complete)} / 1 trial check sets report complete" in text
    expected_status = "complete; provisional" if complete else "pending or incomplete"
    assert f"| `muont_lr_down` | {expected_status} |" in text
    assert "not a fresh liveness check or a final hash-bound validation receipt" in text
    assert ("| `muont_lr_down` | Update-24 post-resume | 2,135 | 0 |" in text) is complete
    assert ("| `muont_lr_down` | Update-22 boundary | 2,123 | 0 |" in text) is complete
    assert ("not an independent checkpoint reconstruction or final launch approval" in text) is complete
    if status == "needs_attention":
        assert "validator failed or requires attention" in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    if changed == "owner":
        receipt["owner"] = {**owner, "controller_start_ticks": "reused"}
    elif changed == "command":
        receipt["gpu_commands"][0]["command"] = ["different", "configuration"]
    elif changed == "log":
        log.write_text("Altered command evidence\n")
    else:
        helper.write_text("# Changed validator\n")
    receipt_path.write_text(json.dumps(receipt))
    update(document, tmp_path, datetime.now(UTC))
    withheld = document.read_text()
    assert "Saved provisional validation identity differs" in withheld
    assert "commands exited zero" not in withheld
    assert "| `muont_lr_down` |" not in withheld
    assert "Saved numerical summaries" not in withheld
    assert withheld.split(START)[0] + withheld.split(END)[1].lstrip("\n") == prose


@pytest.mark.parametrize("changed", ("source", "comparison", "history", "manifest"))
def test_refresh_publishes_strict_captured_windows_then_withholds_changed_evidence(tmp_path, changed):
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    release = preparation / "wave7-release-binding.json"
    release.write_text(json.dumps({"status": "frozen_pending_launch"}))
    observation = preparation / "wave7-observations/20261116T1545Z"
    identities = {}
    for index, arm in enumerate(ARMS):
        directory = observation / arm
        directory.mkdir(parents=True)
        manifest = directory / "run.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "running",
                    "data_receipt_sha256": "same-data",
                    "config": {
                        "model": {"sequence_length": 4096},
                        "data": {"mixture": "same-mixture"},
                        "training": {
                            "steps": 5120,
                            "batch_tokens": 524288,
                            "tokens": 2684354560,
                            "seed": 0,
                            "heldout_windows": 64,
                        },
                    },
                }
            )
        )
        metrics = directory / "metrics_from_000000.jsonl"
        metrics.write_text(
            "".join(
                json.dumps({"step": step, "tokens": step * 524288, "train_loss": 4.0 - 0.01 * index}) + "\n"
                for step in range(1, 1026)
            )
        )
        identities[arm] = {
            "observed_updates": 1025,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "history_sha256": {metrics.name: hashlib.sha256(metrics.read_bytes()).hexdigest()},
        }
    record = {
        "status": "captured_strict_wave7_observation",
        "time": "2026-11-16T15:45:23+00:00",
        "source_release_sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
        "runs": identities,
    }
    receipt = observation / "observation.json"
    receipt.write_text(json.dumps(record))
    analysis_path = observation / "analysis.json"
    analysis = analyze_capture(observation)
    analysis_path.write_text(json.dumps(analysis))
    update(document, tmp_path, datetime.now(UTC))
    text = document.read_text()
    assert "### Wave-7 Strict Captured Comparison" in text
    assert "2026-11-16 23:45:23" in text
    assert "| `outer_leftdiag` | 1025 |" in text
    assert "536,870,912-token training-window deltas: -0.020000" in text
    assert "Final C4 delta" not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    if changed == "source":
        release.write_text(release.read_text() + "\n")
    elif changed == "comparison":
        analysis["analyses"][0]["comparisons"]["wave7/outer_leftdiag"]["training_windows"][0]["delta"] = -1
        analysis_path.write_text(json.dumps(analysis))
    elif changed == "history":
        metrics.write_text(metrics.read_text() + "\n")
    else:
        state = json.loads(manifest.read_text())
        state["status"] = "complete"
        manifest.write_text(json.dumps(state))
        record["runs"][ARMS[-1]]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        receipt.write_text(json.dumps(record))
    update(document, tmp_path, datetime.now(UTC))
    withheld = document.read_text()
    assert "captured result summaries withheld" in withheld
    assert "Reconstructed comparison:" not in withheld
    assert "| `outer_leftdiag` | 1025 |" not in withheld
    assert withheld.split(START)[0] + withheld.split(END)[1].lstrip("\n") == prose


def test_refresh_tracks_tuning_execution_partial_cloud_and_unknown_charges_without_changing_prose(tmp_path):
    document = tmp_path / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    directory = tmp_path / "campaigns/fixed-age"
    helpers = directory / "execution-helpers"
    helpers.mkdir(parents=True)
    helper = helpers / "runner.py"
    helper.write_text("# Frozen runner fixture\n")
    archive = directory / "execution-source.tar"
    archive.write_bytes(b"archive fixture")
    campaign_path = directory / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "study": "wave7_fixed_age_matrix_lr",
                "status": "prepared_unvalidated_not_queued",
                "families": ["muont"],
                "trials": [{"name": f"muont_lr_{label}"} for label in ("down", "base", "up")],
            }
        )
    )
    validation_path = directory / "validation-binding.json"
    validation_path.write_text(json.dumps({"status": "frozen_pending_completed_wave7_and_actual_gpu_validation"}))
    execution_path = directory / "execution-binding.json"
    campaign_digest = hashlib.sha256(campaign_path.read_bytes()).hexdigest()
    execution_path.write_text(
        json.dumps(
            {
                "status": "frozen_pending_completed_wave7_and_actual_gpu_validation",
                "campaign": str(campaign_path),
                "campaign_sha256": campaign_digest,
                "validation_binding_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                "helpers": str(helpers),
                "helpers_sha256": {helper.name: hashlib.sha256(helper.read_bytes()).hexdigest()},
                "archive": str(archive),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    )
    now = datetime(2026, 11, 16, 16, 20, tzinfo=UTC)
    update(document, tmp_path, now)
    assert "No tuning worker launch is recorded." in document.read_text()
    digest = hashlib.sha256(execution_path.read_bytes()).hexdigest()
    (directory / "launch-state.json").write_text(
        json.dumps({"status": "needs_attention", "binding_sha256": digest, "monitor_pid": 1234})
    )
    monitor = directory / "monitor"
    monitor.mkdir()
    snapshot_path = monitor / "latest.json"
    snapshot = {
        "status": "active",
        "campaign_sha256": campaign_digest,
        "binding_sha256": digest,
        "workers": [{"family": "muont", "status": "unknown"}],
        "trials": [
            {"name": "muont_lr_down", "status": "failed", "observed_tokens": 33554432},
            {"name": "muont_lr_base", "status": "running", "observed_tokens": None},
        ],
    }
    snapshot_path.write_text(json.dumps(snapshot))
    cloud_path = directory / "swanlab-upload.json"
    entry = {"status": "verified", "campaign_sha256": campaign_digest, "execution_binding_sha256": digest}
    cloud = {
        "status": "partial_verified",
        "campaign_sha256": campaign_digest,
        "execution_binding_sha256": digest,
        "visibility": "PRIVATE",
        "project": "test/output-side-normalization-marin",
        "runs": {"muont_lr_down": entry},
    }
    cloud_path.write_text(json.dumps(cloud))
    update(document, tmp_path, now)
    text = document.read_text()
    assert "| `muont` | `unknown` | `not recorded` |" in text
    assert "| `muont_lr_down` | `failed` | 33,554,432 |" in text
    assert "| `muont_lr_base` | `running` | unknown; not zero |" in text
    assert "Saved tuning cloud verification: 1 / 3 trial histories" in text
    assert "no complete-study import claimed" in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    cloud.update(
        status="verified",
        runs={f"muont_lr_{label}": entry for label in ("down", "base", "up")},
    )
    cloud_path.write_text(json.dumps(cloud))
    update(document, tmp_path, now)
    assert "All tuning histories have a saved verified receipt" in document.read_text()
    cloud["execution_binding_sha256"] = "different binding"
    cloud_path.write_text(json.dumps(cloud))
    snapshot["binding_sha256"] = "different binding"
    snapshot_path.write_text(json.dumps(snapshot))
    update(document, tmp_path, now)
    text = document.read_text()
    assert "Saved tuning cloud identity differs" in text
    assert "Saved tuning monitor identity differs" in text
    assert "| `muont_lr_down` |" not in text
    helper.write_text("# Changed runner fixture\n")
    update(document, tmp_path, now)
    assert "Execution binding identity differs" in document.read_text()
    assert document.read_text().split(START)[0] + document.read_text().split(END)[1].lstrip("\n") == prose


def test_without_digests_keeps_readings_and_drops_receipt_citations() -> None:
    digest = "a" * 64
    kept = without_digests(
        [
            f"Saved validation-only binding: `run-preparation/x.json`; SHA256 `{digest}`.",
            f"Saved controller log SHA256 `{digest}`.",
            f"Resume diverged by update 24: 460 tensors exceed tolerance. Evidence SHA256 `{digest}`.",
            f"| wave12 | fused | -0.0025 | `{digest}` |",
            "Plain reading with no citation.",
        ]
    )
    assert not any(re.search(r"[0-9a-f]{32,}", line) for line in kept)
    assert "Saved validation-only binding: `run-preparation/x.json`." in kept
    assert "Resume diverged by update 24: 460 tensors exceed tolerance." in kept
    assert "Plain reading with no citation." in kept
    assert "| wave12 | fused | -0.0025 |  |" in kept
    assert not any("Saved controller log" in line for line in kept)
