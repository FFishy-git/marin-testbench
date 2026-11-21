# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline acceptance for unique related trials and source-bound LR selection."""

import json
import shutil
from pathlib import Path

import pytest
from related_campaign import (
    ARMS,
    BATCH_TOKENS,
    MOUSSE_REFERENCE,
    PROFILES,
    STEPS,
    TASK,
    TOKENS,
    WORLD,
    analyze,
    campaign_path,
    prepare,
    sha256,
    status,
)


@pytest.fixture
def campaign_fixture(tmp_path):
    source = tmp_path / "source"
    for folder, _ in PROFILES.values():
        destination = source / "experiment/comparisons" / folder
        shutil.copytree(TASK / "experiment/comparisons" / folder, destination)
    (source / "experiment/train.py").write_text("# Offline trainer identity\n")
    revision = json.loads((TASK / "experiment/upstream_manifest.json").read_text())["revision"]
    upstream = source / "artifacts/external-sources" / f"Toner-{revision}"
    paths = {
        "reproduce/marin/train.py": "# Offline upstream trainer identity\n",
        "src/toner/fast/muon.py": "# Offline common NS identity\n",
        "reproduce/marin/data_plan.json": '{"files":[]}\n',
    }
    for name, text in paths.items():
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (source / "experiment/upstream_manifest.json").write_text(
        json.dumps({"revision": revision, "files": {name: sha256(upstream / name) for name in paths}})
    )
    oracle = source / MOUSSE_REFERENCE
    oracle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TASK / MOUSSE_REFERENCE, oracle)
    root = tmp_path / "output"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "experiment_receipt.json").write_text(
        json.dumps({"root": str(data), "data_plan_sha256": sha256(upstream / "reproduce/marin/data_plan.json")})
    )
    return root, source


@pytest.fixture
def prepared(campaign_fixture):
    root, source = campaign_fixture
    return root, source, prepare(root, source=source)


def write_trial(campaign, trial, *, steps=STEPS, manifest_status="complete", loss=3.0, split=None):
    directory = Path(trial["out"])
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(trial["config"]).read_text())
    manifest = {
        "status": manifest_status,
        "completed_steps": steps,
        "execution_check": False,
        "config": config,
        "data": campaign["data"],
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "source_revision": campaign["source_revision"],
        "source_sha256": campaign["source_sha256"],
        "world_size": WORLD,
    }
    (directory / "run.json").write_text(json.dumps(manifest))
    rows = [
        {
            "step": step,
            "tokens": step * BATCH_TOKENS,
            "train_loss": 4.0,
            "step_seconds": 1.0,
            "elapsed_seconds": step + 10,
            "end_to_end_seconds": step + 11,
        }
        for step in range(1, steps + 1)
    ]
    if rows and steps == STEPS:
        weights = config["data"]["mixture_weights"]
        heldout = {f"heldout/{name}": loss for name in weights}
        selection = sum(weight * heldout[f"heldout/{name}"] for name, weight in weights.items()) / sum(weights.values())
        # C4 and training loss prefer lr0; selection_loss deliberately prefers lr2.
        rows[-1].update(
            heldout_losses=heldout,
            selection_loss=selection,
            c4_loss=float(2 + trial["lr_index"]),
            train_loss=float(2 + trial["lr_index"]),
        )
    parts = [(0, rows)] if split is None else [(0, rows[:split]), (split, rows[split:])]
    for start, segment in parts:
        (directory / f"metrics_from_{start:06d}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in segment))
    if manifest_status == "complete" and steps == STEPS:
        checkpoint = directory / f"checkpoints/step_{STEPS:06d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "complete.json").write_text(json.dumps({"step": STEPS, "world": WORLD}))
        for rank in range(WORLD):
            (checkpoint / f"rank_{rank}.pt").write_bytes(b"offline checkpoint receipt fixture")
    return directory


def rewrite_json(path, change):
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))


def test_prepare_deduplicates_shared_profile_runs_and_preserves_source(campaign_fixture):
    root, source = campaign_fixture
    before = {str(path.relative_to(source)): sha256(path) for path in source.rglob("*") if path.is_file()}
    campaign = prepare(root, source=source)
    assert json.loads(campaign_path(root).read_text()) == campaign
    assert len(campaign["trials"]) == 18
    assert len({trial["out"] for trial in campaign["trials"]}) == 18
    assert len(campaign["waves"]) == 6
    assert sum(len(wave["trials"]) for wave in campaign["waves"]) == 24
    assert all(sum(trial["arm"] == arm for trial in campaign["trials"]) == 3 for arm in ARMS)
    shared = [trial for trial in campaign["trials"] if trial["arm"] in ARMS[:2]]
    assert len(shared) == 6
    assert all(len(trial["aliases"]) == 2 and len(trial["profiles"]) == 2 for trial in shared)
    assert {str(path.relative_to(source)): sha256(path) for path in source.rglob("*") if path.is_file()} == before
    assert not (campaign_path(root).parent / "runs").exists()
    with pytest.raises(FileExistsError):
        prepare(root, source=source)
    assert json.loads(campaign_path(root).read_text()) == campaign


def test_preparation_refuses_divergent_shared_config_without_creating_campaign(campaign_fixture):
    root, source = campaign_fixture
    path = source / "experiment/comparisons/related_paper_short/muon_control_0p5m_seed0_lr0.json"
    rewrite_json(path, lambda config: config.update(reference_validation_loss=9.0))
    with pytest.raises(ValueError):
        prepare(root, source=source)
    assert not campaign_path(root).parent.exists()


def test_status_distinguishes_absent_pending_running_and_completed_evidence(prepared):
    root, _, campaign = prepared
    initial = status(root)
    assert initial["status"] == "not_started"
    assert initial["completed_trials"] == 0
    trial = campaign["trials"][0]
    Path(trial["out"]).mkdir(parents=True)
    assert status(root)["trials"][0]["status"] == "awaiting_manifest"
    directory = write_trial(campaign, trial, steps=7, manifest_status="running")
    observed = status(root)
    assert observed["status"] == "running"
    assert observed["trials"][0]["observed_steps"] == 7
    assert observed["trials"][0]["process_liveness"] == "not_observed"
    rewrite_json(directory / "run.json", lambda manifest: manifest.update(status="complete", completed_steps=STEPS))
    truncated = status(root)["trials"][0]
    assert truncated["status"] == "invalid_evidence"
    assert not truncated["complete"]
    assert "history_truncated" in truncated["reasons"]


def test_analysis_withholds_lr_until_all_three_and_selects_only_final_mixture(prepared):
    root, _, campaign = prepared
    trials = [trial for trial in campaign["trials"] if trial["arm"] == "muon_control"]
    for index, trial in enumerate(trials):
        write_trial(campaign, trial, loss=float(5 - index), split=160)
        result = analyze(root)
        if index < 2:
            assert result["completed_arm_lr_choices"] == {}
    assert result["status"] == "incomplete"
    assert result["completed_arm_lr_choices"]["muon_control"]["trial"] == trials[2]["name"]
    assert result["observed_tokens_by_arm"]["muon_control"] == 3 * TOKENS
    assert all(len(item["training_windows"]) == 5 for item in result["trials"] if item["complete"])
    directory = Path(trials[2]["out"])
    rewrite_json(directory / "run.json", lambda manifest: manifest.update(status="failed"))
    assert analyze(root)["completed_arm_lr_choices"] == {}


@pytest.mark.parametrize(
    "damage",
    [
        "missing_rank",
        "empty_rank",
        "checkpoint_world",
        "checkpoint_step",
        "missing_selection",
        "wrong_selection",
        "nonfinite_heldout",
        "execution_check",
        "gap",
        "overlap",
        "trailing_partial",
    ],
)
def test_complete_manifest_never_overrides_invalid_final_evidence(prepared, damage):
    root, _, campaign = prepared
    trial = campaign["trials"][0]
    directory = write_trial(campaign, trial)
    metrics = directory / "metrics_from_000000.jsonl"
    checkpoint = directory / f"checkpoints/step_{STEPS:06d}"
    if damage == "missing_rank":
        (checkpoint / "rank_7.pt").unlink()
    elif damage == "empty_rank":
        (checkpoint / "rank_7.pt").write_bytes(b"")
    elif damage == "checkpoint_world":
        rewrite_json(checkpoint / "complete.json", lambda receipt: receipt.update(world=1))
    elif damage == "checkpoint_step":
        rewrite_json(checkpoint / "complete.json", lambda receipt: receipt.update(step=319))
    elif damage == "execution_check":
        rewrite_json(directory / "run.json", lambda manifest: manifest.update(execution_check=True))
    else:
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if damage == "missing_selection":
            del rows[-1]["selection_loss"]
        elif damage == "wrong_selection":
            rows[-1]["selection_loss"] += 0.1
        elif damage == "nonfinite_heldout":
            rows[-1]["heldout_losses"]["heldout/dclm"] = float("nan")
        elif damage == "gap":
            del rows[100]
        elif damage == "overlap":
            (directory / "metrics_from_000319.jsonl").write_text(json.dumps(rows[-1]) + "\n")
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if damage == "trailing_partial":
            with metrics.open("a") as stream:
                stream.write('{"step":321')
    result = status(root)
    json.dumps(result, allow_nan=False)
    assert result["status"] == "needs_attention"
    assert result["completed_trials"] == 0
    assert result["trials"][0]["status"] == "invalid_evidence"
    assert not result["trials"][0]["complete"]
    assert analyze(root)["completed_arm_lr_choices"] == {}


@pytest.mark.parametrize("identity", ["config", "data_hash", "data_root", "source", "missing_source", "world"])
def test_analysis_rejects_wrong_executed_identity(prepared, identity):
    root, _, campaign = prepared
    directory = write_trial(campaign, campaign["trials"][0])

    def change(manifest):
        if identity == "config":
            manifest["config"]["optimizer"]["lr"] *= 2
        elif identity == "data_hash":
            manifest["data_receipt_sha256"] = "0" * 64
        elif identity == "data_root":
            manifest["data"] = str(root / "other-data")
        elif identity == "source":
            manifest["source_sha256"]["experiment/train.py"] = "0" * 64
        elif identity == "missing_source":
            del manifest["source_sha256"]["experiment/train.py"]
        else:
            manifest["world_size"] = 1

    rewrite_json(directory / "run.json", change)
    result = analyze(root)
    assert result["status"] == "needs_attention"
    assert not result["trials"][0]["complete"]
    assert result["completed_arm_lr_choices"] == {}


@pytest.mark.parametrize("binding", ["source", "config", "wave", "data_receipt", "campaign_trials", "selection"])
def test_status_refuses_changed_preparation_binding(prepared, binding):
    root, source, campaign = prepared
    if binding == "source":
        (source / "experiment/train.py").write_text("# Changed source\n")
    elif binding == "config":
        rewrite_json(Path(campaign["trials"][0]["config"]), lambda config: config["training"].update(steps=160))
    elif binding == "wave":
        rewrite_json(Path(campaign["waves"][0]["manifest"]), lambda wave: wave["wave"].pop())
    elif binding == "data_receipt":
        rewrite_json(root / "data/experiment_receipt.json", lambda receipt: receipt.update(root="/different"))
    elif binding == "campaign_trials":
        rewrite_json(campaign_path(root), lambda record: record["trials"].pop())
    else:
        rewrite_json(campaign_path(root), lambda record: record["selection"].update(metric="c4_loss"))
    with pytest.raises(ValueError):
        status(root)


def test_running_partial_row_is_pending_not_completion_and_status_is_read_only(prepared):
    root, _, campaign = prepared
    directory = write_trial(campaign, campaign["trials"][0], steps=7, manifest_status="running")
    with (directory / "metrics_from_000000.jsonl").open("a") as stream:
        stream.write('{"step":8')
    before = {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}
    result = status(root)
    assert result["status"] == "running"
    assert result["trials"][0]["status"] == "running"
    assert not result["trials"][0]["complete"]
    assert {str(path): sha256(path) for path in root.rglob("*") if path.is_file()} == before


def test_all_eighteen_verified_trials_complete_and_report_six_lr_choices(prepared):
    root, _, campaign = prepared
    for trial in campaign["trials"]:
        write_trial(campaign, trial, loss=float(5 - trial["lr_index"]))
    result = analyze(root)
    assert result["status"] == "complete"
    assert result["completed_trials"] == 18
    assert set(result["completed_arm_lr_choices"]) == set(ARMS)
    assert all(choice["lr_index"] == 2 for choice in result["completed_arm_lr_choices"].values())
    assert all(trial["checkpoint_receipt_sha256"] and trial["history_sha256"] for trial in result["trials"])
