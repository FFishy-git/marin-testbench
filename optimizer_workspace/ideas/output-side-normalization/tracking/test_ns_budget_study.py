# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise fair NS LR selection and large-window comparisons on real history files."""

import copy
import json
import shutil
from pathlib import Path

import ns_budget_study as study
import prepare_ns_budget_study as preparation
import pytest
from compare_runs import RunHistory
from selected_lr_release_owner import candidate_files, read, sha256

PRESERVED = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/ns-full-model-owner-single-pass-20261119T0257"
)


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


@pytest.fixture
def protocol(tmp_path):
    candidate = tmp_path / "candidate"
    archived = Path(__file__).resolve().parents[1]
    source = archived if (archived / study.trial_specs()[0]["config_name"]).is_file() else PRESERVED
    names = {
        *(spec["config_name"] for spec in study.trial_specs()),
        *(f"experiment/configs/{family}_full_lr_lower_0p5m_seed0.json" for family in study.contract.FAMILIES),
    }
    for name in names:
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    record(candidate / "experiment/upstream_manifest.json", {"revision": "fixture", "files": {}})
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    record(tmp_path / "data/experiment_receipt.json", {"data": "identical fixture"})
    frozen = {
        "candidate": str(candidate),
        "files_sha256": candidate_files(candidate),
        "data_receipt_sha256": sha256(tmp_path / "data/experiment_receipt.json"),
    }
    return study.plan(tmp_path, candidate, frozen)


def fixture_history(protocol, name, *, steps=study.STEPS, late_loss=3.0, endpoint_loss=3.0, c4_loss=3.5):
    trial = next(item for item in protocol["trials"] if item["name"] == name)
    config = copy.deepcopy(trial["config"])
    manifest = {
        "config": config,
        "status": "complete" if steps == study.STEPS else "running",
        "completed_steps": steps,
        "data_receipt_sha256": protocol["data_receipt_sha256"],
        "source_sha256": protocol["executed_source_sha256"],
        "elapsed_seconds": steps * 1.5,
    }
    rows = []
    weights = config["data"]["mixture_weights"]
    for step in range(1, steps + 1):
        row = {
            "step": step,
            "tokens": step * study.contract.BATCH_TOKENS,
            "train_loss": 4.0 + (0.25 if step > 4096 else 0),
            "step_seconds": 1.5,
            "elapsed_seconds": step * 1.5,
            "end_to_end_seconds": step * 1.5,
        }
        if (step - 1) % config["runtime"]["eval_every"] == 0 or step == study.STEPS:
            target = endpoint_loss if step == study.STEPS else late_loss if step > 2560 else 4.0
            heldout = {f"heldout/{source}": target for source in weights}
            row.update(
                heldout_losses=heldout,
                selection_loss=sum(weight * heldout[f"heldout/{source}"] for source, weight in weights.items())
                / sum(weights.values()),
                c4_loss=c4_loss,
            )
        rows.append(row)
    return RunHistory(manifest, rows, [rows])


def save_history(protocol, name, run):
    trial = next(item for item in protocol["trials"] if item["name"] == name)
    directory = Path(trial["out"])
    record(directory / "run.json", run.manifest)
    record(directory / "config.json", trial["config"])
    path = directory / "metrics_from_000000.jsonl"
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in run.rows))
    if run.manifest["status"] == "complete":
        folder = directory / "checkpoints/step_005120"
        record(folder / "complete.json", {"step": study.STEPS, "world": study.contract.WORLD})
        for rank in range(study.contract.WORLD):
            (folder / f"rank_{rank}.pt").write_bytes(f"rank {rank}".encode())
    return directory


def test_ns_study_full_protocol_keeps_every_actual_lr_and_requires_new_source_gates(protocol):
    study.require_plan(protocol)
    assert len(protocol["trials"]) == 54
    assert protocol["planned_grid_tokens"] == 144955146240
    assert protocol["execution_authorized"] is False
    for item in protocol["trials"]:
        gate = item["required_actual_config_gate"]
        assert gate["configuration_sha256"] == item["config_sha256"]
        assert gate["production_source_revalidation_required"] is True
        assert item["config"]["schedule"]["decay_steps"] == 4096
    with pytest.raises(ValueError):
        changed = copy.deepcopy(protocol)
        changed["trials"][0]["required_actual_config_gate"]["production_source_revalidation_required"] = False
        study.require_plan(changed)


def test_ns_study_parent_data_identity_comes_from_authenticated_preparation(protocol):
    path = Path(protocol["candidate"]) / preparation.parent.PREPARATION
    prepared = {key: protocol[key] for key in ("candidate", "files_sha256", "data_receipt_sha256")}
    record(path, prepared)
    parent = {key: protocol[key] for key in ("candidate", "files_sha256")} | {"preparation_sha256": sha256(path)}
    assert preparation.source_record(parent) == prepared
    record(path, prepared | {"data_receipt_sha256": "substituted-data"})
    with pytest.raises(ValueError):
        preparation.source_record(parent)


def test_ns_study_independently_selected_muont_fused_and_null_keep_exact_full_budget_comparisons(protocol):
    cases = [study.BASELINE, "outer_leftdiag_fused_ns_f32_q5", "null_leftdiag_fused_ns_f32_q5"]
    runs = {}
    losses = ((3.5, 3.0, 3.3), (2.9, 3.1, 3.4), (3.4, 3.2, 3.0))
    for case, choices in zip(cases, losses, strict=True):
        for (label, _), loss in zip(study.contract.LR_GRID, choices, strict=True):
            name = f"{case}_full_lr_{label}"
            runs[name] = fixture_history(protocol, name, late_loss=loss, endpoint_loss=loss)
    result = study.analyze_histories(protocol, runs)
    assert [result["provisional_history_lr_choices"][case]["lr_label"] for case in cases] == ["center", "lower", "upper"]
    tuned = result["provisional_tuned_comparisons"]
    assert {item["role"] for item in tuned} == {
        "provisional_selected_case_vs_selected_canonical_muont",
        "provisional_independently_tuned_fused_vs_matched_null",
    }
    for report in tuned:
        for comparison in report["comparisons"].values():
            assert comparison["final_comparison_eligible"] is True
            assert len(comparison["training_windows"]) == 5
            assert len(comparison["evaluations"]["selection_loss"]["late_tokens"]) == 11
    assert result["symmetric_expansion_required"] is True
    assert result["superiority_claim_permitted"] is False


@pytest.mark.parametrize("fault", ("schedule", "auxiliary_lr", "source"))
def test_ns_study_refuses_source_change_outside_actual_finite_map(protocol, fault):
    candidate = Path(protocol["candidate"])
    if fault == "source":
        record(candidate / "experiment/changed.json", {"source": "new"})
    else:
        trial = protocol["trials"][0]
        path = candidate / trial["config_name"]
        config = read(path)
        if fault == "schedule":
            config["schedule"]["stable_steps"] += 1
        else:
            config["optimizer"]["auxiliary_lr"] *= 2
        record(path, config)
    with pytest.raises(ValueError):
        study.require_plan(protocol)


def test_ns_study_selects_late_mixture_not_endpoint_or_reporting_c4(protocol):
    case = "outer_leftdiag_fused_ns_f32_q5"
    runs = {
        f"{case}_full_lr_lower": fixture_history(
            protocol, f"{case}_full_lr_lower", late_loss=3.5, endpoint_loss=2.0, c4_loss=1.0
        ),
        f"{case}_full_lr_center": fixture_history(
            protocol, f"{case}_full_lr_center", late_loss=3.0, endpoint_loss=3.2, c4_loss=5.0
        ),
        f"{case}_full_lr_upper": fixture_history(
            protocol, f"{case}_full_lr_upper", late_loss=3.2, endpoint_loss=3.0, c4_loss=2.0
        ),
    }
    result = study.analyze_histories(protocol, runs)
    winner = result["provisional_history_lr_choices"][case]
    assert winner["lr_label"] == "center"
    assert winner["mean_late_selection_loss"] == pytest.approx((10 * 3.0 + 3.2) / 11)
    assert not result["symmetric_expansion_required"]
    assert result["superiority_claim_permitted"] is False
    incomplete = study.analyze_histories(protocol, {name: run for name, run in runs.items() if "upper" not in name})
    assert case not in incomplete["provisional_history_lr_choices"]
    assert incomplete["incomplete_case_grids"][case] == [f"{case}_full_lr_upper"]


def test_ns_study_boundary_winner_requires_expansion_for_every_compared_case(protocol):
    case = study.BASELINE
    runs = {
        f"{case}_full_lr_{label}": fixture_history(
            protocol, f"{case}_full_lr_{label}", late_loss=loss, endpoint_loss=loss
        )
        for label, loss in (("lower", 3.5), ("center", 3.2), ("upper", 3.0))
    }
    result = study.analyze_histories(protocol, runs)
    assert result["boundary_winners"] == [case]
    assert result["symmetric_expansion_required"] is True
    assert result["expansion"]["apply_equally_to_all_compared_cases"] is True
    assert result["expansion"]["additional_matrix_lrs"] == pytest.approx([0.032, 0.128])
    assert result["confirmation_seeds"] == [3, 4, 5]
    assert result["production_acceptance"] is False


@pytest.mark.parametrize("steps,windows,eligible", ((1536, 1, False), (study.STEPS, 5, True)))
def test_ns_study_compares_canonical_finite_map_and_matched_null_in_large_shared_windows(
    protocol, steps, windows, eligible
):
    names = [
        f"{study.BASELINE}_full_lr_center",
        *(f"{family}_ns_f32_q5_full_lr_center" for family in study.contract.FAMILIES),
    ]
    runs = {name: fixture_history(protocol, name, steps=steps) for name in names}
    fused = "outer_leftdiag_fused_ns_f32_q5_full_lr_center"
    for row in runs[fused].rows:
        row["train_loss"] -= 0.01
    result = study.analyze_histories(protocol, runs)
    cohort = [
        item
        for item in result["same_lr_comparisons"]
        if item["operand"] == "f32" and item["budget"] == 5 and item["lr_label"] == "center"
    ]
    assert {item["role"] for item in cohort} == {
        "canonical_muont_same_lr",
        "same_finite_map_muont",
        "matched_null_same_lr",
    }
    for report in cohort:
        item = report["comparisons"][fused]
        assert len(item["training_windows"]) == windows
        assert [window["delta"] for window in item["training_windows"]] == pytest.approx([-0.01] * windows)
        assert item["final_comparison_eligible"] is eligible
        assert len(item["final_two_complete_training_windows"]) == min(windows, 2)
        assert item["evaluations"]["selection_loss"]["final_tokens"] == (study.TOKENS if eligible else None)
    assert not result["provisional_history_lr_choices"]


@pytest.mark.parametrize("fault", ("mixture", "missing_eval", "source", "execution_check", "wrong_seed"))
def test_ns_study_refuses_inconsistent_loss_or_unpaired_history(protocol, fault):
    name = f"{study.BASELINE}_full_lr_lower"
    run = fixture_history(protocol, name)
    if fault == "mixture":
        run.rows[250]["selection_loss"] += 0.001
    elif fault == "missing_eval":
        run.rows[250].pop("selection_loss")
    elif fault == "source":
        run.manifest["source_sha256"] = {"experiment/not_executed.py": "changed"}
    elif fault == "execution_check":
        run.manifest["execution_check"] = True
    else:
        run.manifest["config"]["training"]["seed"] = 1
    with pytest.raises(ValueError):
        study.analyze_histories(protocol, {name: run})


@pytest.mark.parametrize("fault", ("none", "missing_rank", "truncated_manifest", "shared_metrics"))
def test_ns_study_actual_file_analysis_keeps_history_only_scope_and_rejects_incomplete_final_evidence(protocol, fault):
    name = f"{study.BASELINE}_full_lr_lower"
    run = fixture_history(protocol, name)
    directory = save_history(protocol, name, run)
    if fault == "missing_rank":
        (directory / "checkpoints/step_005120/rank_7.pt").unlink()
    elif fault == "truncated_manifest":
        run.manifest["completed_steps"] -= 1
        record(directory / "run.json", run.manifest)
    elif fault == "shared_metrics":
        (directory / "shared.jsonl").hardlink_to(directory / "metrics_from_000000.jsonl")
    if fault == "none":
        result = study.analyze(protocol)
        assert result["complete_histories"] == 1
        assert not result["provisional_history_lr_choices"]
        assert len(result["runs"][name]["training_windows"]) == 5
        assert result["private_cloud_verification"] is False
        assert result["history_evidence_sha256"][str(directory / "run.json")] == sha256(directory / "run.json")
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            study.analyze(protocol)
