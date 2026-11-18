# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check explicit new-grid execution plans and fail-closed frozen predecessor evidence."""

import ast
import copy
import json
import subprocess
from pathlib import Path

import boundary_handoff
import pytest
import run_fixed_age_tuning
import run_fixed_age_validation
from fixed_age_protocol import analyzer_name, multipliers
from launch_wave5 import NODES
from prepare_lr_boundary import MULTIPLIERS, prepare
from prepare_tuning import sha256
from test_lr_boundary import completed_parent as parent_fixture
from test_lr_boundary import fixed_source as source_fixture
from upstream_validation import ARMS as UPSTREAM_ARMS
from validate_fixed_age_tuning import plan
from wave7_release import read


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def boundary_campaign(tmp_path):
    source = source_fixture.__wrapped__(tmp_path)
    root, candidate, parent = parent_fixture.__wrapped__(source)
    out = root / "campaigns/boundary"
    prepare(root, parent, out)
    return root, candidate, out / "campaign.json"


def test_boundary_validation_and_execution_plans_keep_all_eight_actual_configs(boundary_campaign):
    _, _, path = boundary_campaign
    campaign = read(path)
    validation = plan(path, path.parent / "validation")
    assignments = run_fixed_age_tuning.plan(path)
    assert multipliers(path) == MULTIPLIERS
    assert analyzer_name(path) == "analyze_lr_boundary.py"
    assert validation["planned_validation_tokens"] == 201326592
    assert len(validation["trials"]) == 8
    assert [item["node"] for item in assignments] == list(NODES)
    assert sum(item["tokens"] for assignment in assignments for item in assignment["trials"]) == 1342177280
    for actual, trial in zip(validation["trials"], campaign["trials"], strict=True):
        assert actual["config"] == trial["config"]
        assert len(actual["commands"]) == 3
        assert not any(actual["checks"].values())
        for command in actual["commands"]:
            assert command[command.index("--config") + 1] == trial["config"]
            assert command[command.index("--execution-boundary") + 1] == "22"
            assert "--nproc_per_node=8" in command
        assert [command[command.index("--max-steps") + 1] for command in actual["commands"]] == ["24", "22", "24"]
    for assignment in assignments:
        assert len(assignment["trials"]) == 2
        for trial in assignment["trials"]:
            assert "--max-steps" not in trial["command"]
            assert "--resume" not in trial["command"]
    assert not (path.parent / "validation").exists()
    assert not (path.parent / "workers").exists()


def test_protocol_dispatch_refuses_arbitrary_campaign_grid(boundary_campaign):
    _, _, path = boundary_campaign
    campaign = read(path)
    campaign["study"] = "arbitrary_lr_grid"
    record(path, campaign)
    with pytest.raises(ValueError, match="Unrecognized"):
        multipliers(path)
    with pytest.raises(ValueError, match="Unrecognized"):
        run_fixed_age_tuning.plan(path)


@pytest.fixture
def predecessor(tmp_path, monkeypatch):
    helpers = tmp_path / "controllers/wave8-launch-20261117/tracking"
    helpers.mkdir(parents=True)
    (helpers / "wave8_release.py").write_text("# Bound predecessor fixture\n")
    archive = tmp_path / "wave8.tar"
    archive.write_bytes(b"frozen predecessor fixture")
    release = {
        "status": "frozen_validated_launch_release",
        "root": str(tmp_path),
        "helpers": str(helpers),
        "helpers_sha256": {"wave8_release.py": sha256(helpers / "wave8_release.py")},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "arms": boundary_handoff.ARMS,
        "nodes": NODES,
    }
    binding_path = tmp_path / "run-preparation/wave8-launch-binding.json"
    record(binding_path, release)
    analyses = [
        {
            "comparisons": {
                arm: {"final_comparison_eligible": True, "training_windows": [1] * 5} for arm in boundary_handoff.ARMS
            }
        }
    ]
    analysis = {"source_release_sha256": sha256(binding_path), "analyses": analyses}
    measured = {
        arm: {
            "complete": True,
            "manifest_sha256": f"{arm}-manifest",
            "metrics_sha256": f"{arm}-metrics",
            "checkpoint_receipt_sha256": f"{arm}-checkpoint",
            "checkpoint_rank_bytes": [10] * 8,
        }
        for arm in boundary_handoff.ARMS
    }
    record(tmp_path / "run-preparation/analysis-wave8.json", analysis)
    cloud = {
        "status": "verified",
        "visibility": "PRIVATE",
        "project": "owner/output-side-normalization-marin",
        "runs": {
            arm: {
                **measured[arm],
                "status": "verified",
                "verification": {
                    "state": "FINISHED",
                    "configuration_verified": True,
                    "token_axis_verified": True,
                    "integer_values_exact": True,
                    "point_counts": {"tokens": 5120, "train/loss": 5120, "eval/selection_loss": 22, "eval/c4_loss": 22},
                },
            }
            for arm in boundary_handoff.ARMS
        },
    }
    record(tmp_path / "run-preparation/swanlab-upload-wave8.json", cloud)
    observed = {
        "status": "complete",
        "all_workers_terminal": True,
        "binding_sha256": sha256(binding_path),
        "workers": [
            {"arm": arm, "status": "complete", "receipt": {"keeper_restore": 0}} for arm in boundary_handoff.ARMS
        ],
    }
    payload = {"analysis": analysis, "runs": measured}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--phase" in command:
            record(Path(command[command.index("--out") + 1]), observed)
            output = "{}"
        else:
            output = json.dumps(payload)
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(boundary_handoff.subprocess, "run", run)
    return tmp_path, observed, payload, calls


def test_predecessor_requires_rederived_full_results_and_fresh_snapshot(predecessor):
    root, _, _, calls = predecessor
    assert boundary_handoff.require_completed(root)["cloud_sha256"] == sha256(
        root / "run-preparation/swanlab-upload-wave8.json"
    )
    assert boundary_handoff.snapshot(root, root / "fresh.json")["status"] == "complete"
    assert len(calls) == 2
    assert "--phase" in calls[-1] and "snapshot" in calls[-1]


@pytest.mark.parametrize(
    "fault", ["active", "unknown", "live_child", "restore", "boolean_restore", "binding", "missing"]
)
def test_fresh_predecessor_snapshot_refuses_nonterminal_or_incomplete_evidence(predecessor, fault):
    root, observed, _, _ = predecessor
    if fault == "active":
        observed["status"] = "active"
    elif fault == "unknown":
        observed["workers"][0]["status"] = "unknown"
    elif fault == "live_child":
        observed["workers"][0]["status"] = "live_orphaned_training"
    elif fault == "restore":
        observed["workers"][0]["receipt"]["keeper_restore"] = 1
    elif fault == "boolean_restore":
        observed["workers"][0]["receipt"]["keeper_restore"] = False
    elif fault == "binding":
        observed["binding_sha256"] = "changed"
    else:
        observed["workers"].pop()
    with pytest.raises(RuntimeError, match="active, unknown or incomplete"):
        boundary_handoff.snapshot(root, root / "fresh.json")


@pytest.mark.parametrize("fault", ["window", "analysis", "partial", "metrics", "counts", "config", "integer", "private"])
def test_predecessor_refuses_partial_or_stale_loss_and_cloud_evidence(predecessor, fault):
    root, _, payload, _ = predecessor
    cloud_path = root / "run-preparation/swanlab-upload-wave8.json"
    cloud = read(cloud_path)
    arm = boundary_handoff.ARMS[0]
    if fault == "window":
        payload["analysis"]["analyses"][0]["comparisons"][arm]["training_windows"].pop()
    elif fault == "analysis":
        payload["analysis"]["extra"] = "changed"
    elif fault == "partial":
        cloud["status"] = "partial_verified"
    elif fault == "metrics":
        cloud["runs"][arm]["metrics_sha256"] = "changed"
    elif fault == "counts":
        cloud["runs"][arm]["verification"]["point_counts"]["train/loss"] = 5119
    elif fault == "config":
        cloud["runs"][arm]["verification"]["configuration_verified"] = False
    elif fault == "integer":
        cloud["runs"][arm]["verification"]["integer_values_exact"] = False
    else:
        cloud["visibility"] = "PUBLIC"
    record(cloud_path, cloud)
    with pytest.raises(ValueError):
        boundary_handoff.require_completed(root)


@pytest.fixture
def wave9_predecessor(tmp_path, monkeypatch):
    helpers = tmp_path / "controllers/wave9-launch-20261117/tracking"
    helpers.mkdir(parents=True)
    names = ("upstream_production.py", "analyze_upstream_run.py", "upstream_run_history.py", "upstream_validation.py")
    for name in names:
        (helpers / name).write_text("# Frozen Wave-9 observation fixture\n")
    (helpers / "upstream_validation.py").write_text(f"ARMS = {UPSTREAM_ARMS!r}\n")
    archive = tmp_path / "run-preparation/source-wave9-launch-20261117.tar"
    archive.parent.mkdir()
    archive.write_bytes(b"original Wave-9 archive")
    binding_path = tmp_path / "run-preparation/wave9-launch-binding.json"
    record(
        binding_path,
        {
            "status": "frozen_validated_launch_release",
            "root": str(tmp_path),
            "helpers": str(helpers),
            "helpers_sha256": {name: sha256(helpers / name) for name in names},
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "arms": UPSTREAM_ARMS,
            "nodes": NODES,
        },
    )
    evaluation_tokens = [step * 524288 for step in (*range(1, 5120, 250), 5120)]
    evaluation = {
        "shared_tokens": evaluation_tokens,
        "late_start_tokens_exclusive": 1342177280,
        "late_end_tokens": 2684354560,
        "late_tokens": evaluation_tokens[11:],
        "mean_late_delta": -0.01,
        "last_shared_tokens": 2684354560,
        "final_tokens": 2684354560,
        "final_value": 3.5,
        "final_delta": -0.02,
    }
    comparison = {
        "final_comparison_eligible": True,
        "common_tokens": 2684354560,
        "training_windows": [
            {"start_tokens": start, "end_tokens": end}
            for start, end in (
                (0, 536870912),
                (536870912, 1073741824),
                (1073741824, 1610612736),
                (1610612736, 2147483648),
                (2147483648, 2684354560),
            )
        ],
        "evaluations": {"selection_loss": copy.deepcopy(evaluation), "c4_loss": copy.deepcopy(evaluation)},
    }
    analysis = {
        "wave": "wave9",
        "window_tokens": 536870912,
        "source_release_sha256": sha256(binding_path),
        "analyses": [
            {
                "window_tokens": 536870912,
                "comparisons": {f"wave9/{arm}": copy.deepcopy(comparison) for arm in UPSTREAM_ARMS},
            }
        ],
    }
    measured = {
        arm: {
            "complete": True,
            "observed_updates": 5120,
            "observed_tokens": 2684354560,
            "manifest_sha256": f"{arm}-manifest",
            "metrics_sha256": f"{arm}-metrics",
            "checkpoint_receipt_sha256": f"{arm}-checkpoint",
            "checkpoint_rank_bytes": [100] * 8,
        }
        for arm in UPSTREAM_ARMS
    }
    cloud = {
        "status": "verified",
        "visibility": "PRIVATE",
        "project": "owner/output-side-normalization-marin",
        "runs": {
            arm: {
                **copy.deepcopy(measured[arm]),
                "status": "verified",
                "verification": {
                    "state": "FINISHED",
                    "configuration_verified": True,
                    "token_axis_verified": True,
                    "integer_values_exact": True,
                    "point_counts": {"tokens": 5120, "train/loss": 5120, "eval/selection_loss": 22, "eval/c4_loss": 22},
                },
            }
            for arm in UPSTREAM_ARMS
        },
    }
    record(tmp_path / "run-preparation/analysis-wave9.json", analysis)
    record(tmp_path / "run-preparation/swanlab-upload-wave9.json", cloud)
    observed = {
        "status": "complete",
        "all_workers_terminal": True,
        "binding_sha256": sha256(binding_path),
        "workers": [
            {
                "arm": arm,
                "status": "complete",
                "receipt": {
                    "arm": arm,
                    "node": node,
                    "release_binding_sha256": sha256(binding_path),
                    "status": "complete",
                    "exit_code": 0,
                    "cleanup_exit_code": 0,
                    "keeper_restore": 0,
                },
            }
            for node, arm in zip(NODES, UPSTREAM_ARMS, strict=True)
        ],
    }
    payload = {"analysis": analysis, "runs": measured}

    def run(command, **kwargs):
        if command[:2] == [boundary_handoff.PYTHON, str(helpers / "upstream_production.py")]:
            options = dict(zip(command[2::2], command[3::2], strict=True))
            if options["--root"] != str(tmp_path) or options["--phase"] != "snapshot":
                raise AssertionError("Only original-bound observation is permitted")
            record(Path(options["--out"]), observed)
            output = "{}"
        elif command[:2] == [boundary_handoff.PYTHON, "-c"] and command[3:] == [str(helpers), str(tmp_path)]:
            modules = {node.module for node in ast.walk(ast.parse(command[2])) if isinstance(node, ast.ImportFrom)}
            if modules != {"pathlib", "analyze_upstream_run", "upstream_run_history"}:
                raise AssertionError("Rederivation must use the frozen Torch-free upstream readers")
            output = json.dumps(payload)
        else:
            raise AssertionError("Unexpected subprocess outside the frozen Wave-9 observation boundary")
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(boundary_handoff.subprocess, "run", run)
    return tmp_path, observed, payload


def test_wave9_handoff_returns_bound_final_evidence_and_reobserves_saved_terminal_state(wave9_predecessor):
    root, observed, _ = wave9_predecessor
    assert boundary_handoff.require_completed(root, wave="wave9") == {
        "analysis_sha256": sha256(root / "run-preparation/analysis-wave9.json"),
        "cloud_sha256": sha256(root / "run-preparation/swanlab-upload-wave9.json"),
        "release_sha256": sha256(root / "run-preparation/wave9-launch-binding.json"),
    }
    out = root / "fresh.json"
    record(out, {"status": "active"})
    assert boundary_handoff.snapshot(root, out, wave="wave9") == observed
    observed["workers"][-1]["status"] = "running"
    with pytest.raises(RuntimeError):
        boundary_handoff.snapshot(root, out, wave="wave9")
    assert read(out)["workers"][-1]["status"] == "running"
    assert not (root / "runs").exists()


@pytest.mark.parametrize("evidence", ["helper", "archive", "arms", "helpers_path"])
def test_wave9_handoff_refuses_changed_original_release_before_observation(wave9_predecessor, evidence):
    root, _, _ = wave9_predecessor
    binding_path = root / "run-preparation/wave9-launch-binding.json"
    frozen = read(binding_path)
    if evidence == "helper":
        (Path(frozen["helpers"]) / "upstream_production.py").write_text("# changed\n")
    elif evidence == "archive":
        Path(frozen["archive"]).write_bytes(b"changed")
    elif evidence == "arms":
        frozen["arms"] = boundary_handoff.ARMS
    else:
        frozen["helpers"] = str(root / "mutable/tracking")
    record(binding_path, frozen)
    out = root / "fresh.json"
    with pytest.raises(ValueError):
        boundary_handoff.snapshot(root, out, wave="wave9")
    with pytest.raises(ValueError):
        boundary_handoff.require_completed(root, wave="wave9")
    assert not out.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "live_child",
        "generation_changed",
        "duplicate_arm",
        "missing_worker",
        "binding",
        "receipt_binding",
        "receipt_node",
        "receipt_status",
        "worker_error",
        "cleanup_error",
        "receipt_error",
        "descendants",
        "snapshot_error",
        "accounting_error",
    ],
)
def test_wave9_fresh_snapshot_refuses_unverified_original_worker_evidence(wave9_predecessor, fault):
    root, observed, _ = wave9_predecessor
    worker = observed["workers"][-1]
    if fault in ("unknown", "live_child", "generation_changed"):
        worker["status"] = {
            "unknown": "unknown",
            "live_child": "live_orphaned_training",
            "generation_changed": "worker_generation_changed",
        }[fault]
    elif fault == "duplicate_arm":
        worker["arm"] = observed["workers"][0]["arm"]
    elif fault == "missing_worker":
        observed["workers"].pop()
    elif fault == "binding":
        observed["binding_sha256"] = "stale"
    elif fault == "receipt_binding":
        worker["receipt"]["release_binding_sha256"] = "stale"
    elif fault == "receipt_node":
        worker["receipt"]["node"] = NODES[0]
    elif fault == "receipt_status":
        worker["receipt"]["status"] = "needs_attention"
    elif fault == "worker_error":
        worker["error"] = "SSH failed"
    elif fault in ("cleanup_error", "receipt_error"):
        worker["receipt"]["cleanup_error" if fault == "cleanup_error" else "error"] = "failed"
    elif fault == "descendants":
        worker["descendants"] = {"123": "original-child"}
    elif fault == "snapshot_error":
        observed["error"] = "SSH observation failed"
    else:
        observed["runs"] = [{"arm": worker["arm"], "accounting_error": "final history changed"}]
    with pytest.raises(RuntimeError):
        boundary_handoff.snapshot(root, root / "fresh.json", wave="wave9")


def test_wave9_handoff_does_not_accept_saved_terminal_receipt_when_fresh_subprocess_fails(
    wave9_predecessor, monkeypatch
):
    root, observed, _ = wave9_predecessor
    out = root / "fresh.json"
    record(out, observed)

    def unavailable(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="SSH unavailable")

    monkeypatch.setattr(boundary_handoff.subprocess, "run", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        boundary_handoff.snapshot(root, out, wave="wave9")
    assert read(out) == observed


@pytest.mark.parametrize("key", ["exit_code", "cleanup_exit_code", "keeper_restore"])
@pytest.mark.parametrize("value", [1, False, "0", None])
def test_wave9_fresh_snapshot_requires_successful_integer_exit_and_restore_receipts(wave9_predecessor, key, value):
    root, observed, _ = wave9_predecessor
    observed["workers"][-1]["receipt"][key] = value
    with pytest.raises(RuntimeError):
        boundary_handoff.snapshot(root, root / "fresh.json", wave="wave9")


@pytest.mark.parametrize(
    "fault",
    ["empty", "missing_arm", "window_count", "window_start", "window_end", "common_tokens", "late", "shared", "final"],
)
def test_wave9_handoff_refuses_matching_but_incomplete_window_or_evaluation_analysis(wave9_predecessor, fault):
    root, _, payload = wave9_predecessor
    analysis = payload["analysis"]
    comparisons = analysis["analyses"][0]["comparisons"]
    comparison = comparisons[f"wave9/{UPSTREAM_ARMS[-1]}"]
    if fault == "empty":
        analysis["analyses"].clear()
    elif fault == "missing_arm":
        comparisons.pop(f"wave9/{UPSTREAM_ARMS[-1]}")
    elif fault == "window_count":
        comparison["training_windows"].pop()
    elif fault in ("window_start", "window_end"):
        key = "start_tokens" if fault == "window_start" else "end_tokens"
        comparison["training_windows"][2][key] += 524288
    elif fault == "common_tokens":
        comparison["common_tokens"] -= 524288
    elif fault == "late":
        comparison["evaluations"]["selection_loss"]["late_tokens"].pop(0)
    elif fault == "shared":
        comparison["evaluations"]["c4_loss"]["shared_tokens"].pop(0)
    else:
        comparison["evaluations"]["c4_loss"]["final_tokens"] = None
    record(root / "run-preparation/analysis-wave9.json", analysis)
    with pytest.raises(ValueError):
        boundary_handoff.require_completed(root, wave="wave9")


@pytest.mark.parametrize(
    "fault",
    [
        "saved_analysis",
        "partial_history",
        "partial_cloud",
        "public",
        "missing_cloud_arm",
        "manifest",
        "metrics",
        "checkpoint",
        "rank_sizes",
        "train_count",
        "token_count",
        "selection_count",
        "c4_count",
        "config",
        "axis",
        "integer",
        "cloud_state",
        "updates",
        "tokens",
        "missing_rank",
        "empty_rank",
    ],
)
def test_wave9_handoff_refuses_partial_or_stale_final_history_and_private_cloud_provenance(wave9_predecessor, fault):
    root, _, payload = wave9_predecessor
    cloud_path = root / "run-preparation/swanlab-upload-wave9.json"
    cloud = read(cloud_path)
    arm = UPSTREAM_ARMS[-1]
    measured = payload["runs"][arm]
    imported = cloud["runs"][arm]
    if fault == "saved_analysis":
        payload["analysis"]["extra"] = "not the saved result"
    elif fault == "partial_history":
        measured["complete"] = False
    elif fault == "partial_cloud":
        cloud["status"] = "partial_verified"
    elif fault == "public":
        cloud["visibility"] = "PUBLIC"
    elif fault == "missing_cloud_arm":
        cloud["runs"].pop(arm)
    elif fault in ("manifest", "metrics", "checkpoint"):
        key = {
            "manifest": "manifest_sha256",
            "metrics": "metrics_sha256",
            "checkpoint": "checkpoint_receipt_sha256",
        }[fault]
        imported[key] = "stale"
    elif fault == "rank_sizes":
        imported["checkpoint_rank_bytes"][-1] += 1
    elif fault in ("train_count", "token_count", "selection_count", "c4_count"):
        key = {
            "train_count": "train/loss",
            "token_count": "tokens",
            "selection_count": "eval/selection_loss",
            "c4_count": "eval/c4_loss",
        }[fault]
        imported["verification"]["point_counts"][key] -= 1
    elif fault in ("config", "axis", "integer"):
        key = {
            "config": "configuration_verified",
            "axis": "token_axis_verified",
            "integer": "integer_values_exact",
        }[fault]
        imported["verification"][key] = False
    elif fault == "cloud_state":
        imported["verification"]["state"] = "RUNNING"
    elif fault in ("updates", "tokens"):
        measured["observed_updates" if fault == "updates" else "observed_tokens"] -= 1
    elif fault == "missing_rank":
        measured["checkpoint_rank_bytes"].pop()
        imported["checkpoint_rank_bytes"].pop()
    else:
        measured["checkpoint_rank_bytes"][-1] = imported["checkpoint_rank_bytes"][-1] = 0
    record(cloud_path, cloud)
    with pytest.raises(ValueError):
        boundary_handoff.require_completed(root, wave="wave9")


@pytest.mark.parametrize("wave", ["wave7", "Wave9", "wave9-launch-20261117"])
def test_handoff_rejects_other_wave_spellings_without_observation(tmp_path, wave):
    with pytest.raises(ValueError):
        boundary_handoff.release(tmp_path, wave=wave)
    with pytest.raises(ValueError):
        boundary_handoff.require_completed(tmp_path, wave=wave)
    with pytest.raises(ValueError):
        boundary_handoff.snapshot(tmp_path, tmp_path / "fresh.json", wave=wave)
    assert not (tmp_path / "fresh.json").exists()


def test_boundary_freeze_keeps_new_configs_and_binds_wave8_without_gpu_execution(boundary_campaign, monkeypatch):
    root, _, path = boundary_campaign
    wave8_path = root / "run-preparation/wave8-launch-binding.json"
    record(wave8_path, {"status": "predecessor_fixture"})
    monkeypatch.setattr(boundary_handoff, "release", lambda root: {"status": "bound"})
    frozen = run_fixed_age_validation.freeze(path)
    assert frozen["status"] == "frozen_pending_completed_wave8_and_actual_gpu_validation"
    assert frozen["predecessor_release_sha256"] == sha256(wave8_path)
    assert len(frozen["plan"]["trials"]) == 8
    monkeypatch.setattr(
        run_fixed_age_validation, "__file__", str(Path(frozen["helpers"]) / "run_fixed_age_validation.py")
    )
    assert run_fixed_age_validation.binding(path)["plan"] == frozen["plan"]
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "bound check", "")
    )
    execution = run_fixed_age_tuning.freeze(path)
    assert execution["status"] == "frozen_pending_completed_wave8_and_actual_gpu_validation"
    assert len(execution["plan"]) == 4
    assert all(len(assignment["trials"]) == 2 for assignment in execution["plan"])
    monkeypatch.setattr(run_fixed_age_tuning, "__file__", str(Path(execution["helpers"]) / "run_fixed_age_tuning.py"))
    assert run_fixed_age_tuning.binding(path)["plan"] == execution["plan"]
    assert not (path.parent / "validation-state.json").exists()
    assert not (path.parent / "validation").exists()
    assert not (path.parent / "launch-state.json").exists()
    assert not (path.parent / "monitor").exists()
    record(wave8_path, {"status": "changed"})
    with pytest.raises(ValueError, match="different Wave-8 predecessor"):
        run_fixed_age_validation.binding(path)


def test_boundary_ready_refuses_missing_actual_validation_before_gpu_launch(boundary_campaign, monkeypatch):
    _, _, path = boundary_campaign
    monkeypatch.setattr(run_fixed_age_tuning, "binding", lambda path: {"validation_binding_sha256": "expected"})
    record(path.parent / "validation-gpu.json", {"status": "not_executed"})
    state = {"status": "passed", "keeper_restore": 1}
    record(path.parent / "validation-state.json", state)
    with pytest.raises(ValueError, match="keeper restoration have not passed"):
        run_fixed_age_tuning.require_ready(path)
    assert not (path.parent / "launch-state.json").exists()


def test_boundary_handoff_dispatch_does_not_accept_old_wave7_results(boundary_campaign, monkeypatch):
    root, _, path = boundary_campaign
    calls = []
    monkeypatch.setattr(boundary_handoff, "require_completed", lambda root: calls.append(root) or {"wave": "wave8"})
    assert run_fixed_age_validation.require_completed_wave(root, campaign_path=path) == {"wave": "wave8"}
    assert calls == [root]
