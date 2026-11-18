# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Equal-budget trials preserve controls and bind their actual written configs."""

import json
import math
from pathlib import Path

import pytest
from analyze_tuning import analyze
from prepare_tuning import FAMILIES, prepare, sha256, trial_config

TASK = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("batch, steps, stable", [("0p5m", 320, 64), ("1m", 160, 32), ("2m", 80, 16)])
def test_short_trial_completes_schedule_without_changing_auxiliary_or_holdout(batch, steps, stable):
    reference = json.loads((TASK / f"experiment/configs/muont_{batch}_seed0.json").read_text())
    before = json.loads(json.dumps(reference))
    actual = trial_config(reference, "muont", batch, "lr_down", 1 / math.sqrt(2))
    assert reference == before
    assert actual["training"]["steps"] == steps
    assert actual["training"]["tokens"] == 167772160
    assert actual["training"]["heldout_windows"] == 64
    assert actual["schedule"]["stable_steps"] == stable
    assert actual["schedule"]["decay_steps"] == steps - stable
    assert actual["schedule"]["min_lr_ratio"] == 0
    assert actual["optimizer"]["lr"] == pytest.approx(reference["optimizer"]["lr"] / math.sqrt(2))
    assert {key: value for key, value in actual["optimizer"].items() if key != "lr"} == {
        key: value for key, value in reference["optimizer"].items() if key != "lr"
    }
    assert actual["data"] == reference["data"]
    assert actual["statistics"] == reference["statistics"]
    assert actual["root"] == reference["root"]
    assert actual["runtime"]["eval_every"] * actual["training"]["batch_tokens"] == 20971520
    assert actual["experiment"]["checkpoint_every"] * actual["training"]["batch_tokens"] == 41943040
    assert "reference_validation_loss" not in actual


@pytest.fixture
def prepared_campaign(tmp_path):
    source = tmp_path / "frozen"
    configs = source / "experiment/configs"
    configs.mkdir(parents=True)
    original = {}
    for family in FAMILIES:
        relative = f"experiment/configs/{family}_0p5m_seed0.json"
        text = (TASK / relative).read_text()
        (source / relative).write_text(text)
        original[relative] = text
    (source / "experiment/train.py").write_text("# Bound trainer fixture\n")
    original["experiment/train.py"] = "# Bound trainer fixture\n"
    upstream = source / "artifacts/pinned-sources-materialized/Toner-fixture"
    upstream_files = {
        "reproduce/marin/model.py": "# Frozen model fixture\n",
        "src/toner/fast/muon.py": "# Frozen optimizer fixture\n",
    }
    for name, text in upstream_files.items():
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        original[str(path.relative_to(source))] = text
    (source / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    manifest = source / "experiment/upstream_manifest.json"
    manifest.write_text(
        json.dumps({"revision": "fixture", "files": {name: sha256(upstream / name) for name in upstream_files}})
    )
    original[str(manifest.relative_to(source))] = manifest.read_text()
    root = tmp_path / "output"
    (root / "data").mkdir(parents=True)
    data = root / "data/experiment_receipt.json"
    data.write_text('{"status":"verified"}\n')
    receipt = tmp_path / "frozen.json"
    receipt.write_text(
        json.dumps(
            {
                "candidate": str(source),
                "files_sha256": {name: sha256(source / name) for name in original},
                "data_receipt_sha256": sha256(data),
            }
        )
    )
    return source, original, root, receipt


def test_prepare_binds_equal_families_outside_source_without_mutating_it(prepared_campaign):
    source, original, root, receipt = prepared_campaign
    out = root / "campaigns/lr"
    result = prepare(root, receipt, out, "0p5m")
    assert json.loads((out / "campaign.json").read_text()) == result
    assert len(result["trials"]) == 12
    assert result["status"] == "prepared_unvalidated_not_queued"
    assert result["initial_tokens_per_family"] == 503316480
    for family in FAMILIES:
        trials = [trial for trial in result["trials"] if trial["family"] == family]
        assert len(trials) == 3
        assert sum(trial["tokens"] for trial in trials) == 503316480
        for trial in trials:
            path = Path(trial["config"])
            assert sha256(path) == trial["config_sha256"]
            assert json.loads(path.read_text())["experiment"]["arm"] == family
    assert {name: (source / name).read_text() for name in original} == original
    assert not (root / "runs").exists()
    with pytest.raises(FileExistsError):
        prepare(root, receipt, out, "0p5m")
    assert json.loads((out / "campaign.json").read_text()) == result


def write_completed_trial(source, out, campaign, trial, index):
    config = json.loads(Path(trial["config"]).read_text())
    directory = out / "runs" / trial["name"]
    directory.mkdir(parents=True)
    sources = {
        "experiment/train.py": sha256(source / "experiment/train.py"),
        "artifacts/external-sources/Toner-fixture/reproduce/marin/model.py": sha256(
            source / "artifacts/external-sources/Toner-fixture/reproduce/marin/model.py"
        ),
        "artifacts/external-sources/Toner-fixture/src/toner/fast/muon.py": sha256(
            source / "artifacts/external-sources/Toner-fixture/src/toner/fast/muon.py"
        ),
    }
    manifest = {
        "status": "complete",
        "completed_steps": 320,
        "config": config,
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "source_sha256": sources,
    }
    (directory / "run.json").write_text(json.dumps(manifest))
    weights = config["data"]["mixture_weights"]
    heldout = {f"heldout/{name}": float(5 - index) for name in weights}
    selection = sum(weight * heldout[f"heldout/{name}"] for name, weight in weights.items())
    selection /= sum(weights.values())
    rows = [
        {"step": step, "tokens": step * 524288, "train_loss": 6.0, "end_to_end_seconds": step + 10}
        for step in range(1, 321)
    ]
    # C4 prefers lr_down, while the held-out mixture prefers lr_up.
    rows[-1].update(c4_loss=float(2 + index), heldout_losses=heldout, selection_loss=selection)
    (directory / "metrics_from_000000.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    checkpoint = directory / "checkpoints/step_000320"
    checkpoint.mkdir(parents=True)
    (checkpoint / "complete.json").write_text(json.dumps({"step": 320, "world": 8}))
    for rank in range(8):
        (checkpoint / f"rank_{rank}.pt").write_bytes(b"rank checkpoint fixture")
    return directory


def test_analyzer_withholds_incomplete_choices_then_ranks_selection_not_c4(prepared_campaign):
    source, _, root, receipt = prepared_campaign
    out = root / "campaigns/lr"
    campaign = prepare(root, receipt, out, "0p5m")
    initial = analyze(out / "campaign.json")
    assert initial["status"] == "incomplete"
    assert initial["completed_family_lr_choices"] == {}
    assert initial["observed_tokens_by_family"] == dict.fromkeys(FAMILIES, 0)
    family_trials = [trial for trial in campaign["trials"] if trial["family"] == "muont"]
    for index, trial in enumerate(family_trials):
        directory = write_completed_trial(source, out, campaign, trial, index)
        result = analyze(out / "campaign.json")
        if index < 2:
            assert result["completed_family_lr_choices"] == {}
    assert result["status"] == "incomplete"
    assert result["completed_family_lr_choices"]["muont"]["trial"] == "muont_lr_up"
    assert result["observed_tokens_by_family"]["muont"] == 503316480
    assert all(len(item["training_windows"]) == 5 for item in result["trials"] if item["family"] == "muont")
    failed = json.loads((directory / "run.json").read_text())
    failed["status"] = "failed"
    (directory / "run.json").write_text(json.dumps(failed))
    after_failure = analyze(out / "campaign.json")
    assert after_failure["completed_family_lr_choices"] == {}
    assert after_failure["observed_tokens_by_family"]["muont"] == 503316480


@pytest.fixture
def completed_muont_campaign(prepared_campaign):
    source, _, root, receipt = prepared_campaign
    out = root / "campaigns/lr"
    campaign = prepare(root, receipt, out, "0p5m")
    trials = [trial for trial in campaign["trials"] if trial["family"] == "muont"]
    for index, trial in enumerate(trials):
        write_completed_trial(source, out, campaign, trial, index)
    return source, out, campaign


def test_analyzer_accepts_frozen_materialized_source_aliases(completed_muont_campaign):
    _, out, _ = completed_muont_campaign
    for path in (out / "runs").glob("*/run.json"):
        manifest = json.loads(path.read_text())
        manifest["source_sha256"] = {
            (
                str(Path("artifacts/pinned-sources-materialized") / Path(name).relative_to("artifacts/external-sources"))
                if Path(name).is_relative_to("artifacts/external-sources")
                else name
            ): digest
            for name, digest in manifest["source_sha256"].items()
        }
        path.write_text(json.dumps(manifest))
    result = analyze(out / "campaign.json")
    assert result["completed_family_lr_choices"]["muont"]["trial"] == "muont_lr_up"


@pytest.mark.parametrize("change", ["upstream", "missing_upstream", "source_root"])
def test_analyzer_rejects_consistent_executed_source_changes(completed_muont_campaign, change):
    source, out, campaign = completed_muont_campaign
    relative = "artifacts/external-sources/Toner-fixture/src/toner/fast/muon.py"
    if change == "upstream":
        (source / relative).write_text("# Changed optimizer in every completed trial\n")
        for path in (out / "runs").glob("*/run.json"):
            manifest = json.loads(path.read_text())
            manifest["source_sha256"][relative] = sha256(source / relative)
            path.write_text(json.dumps(manifest))
    elif change == "missing_upstream":
        for path in (out / "runs").glob("*/run.json"):
            manifest = json.loads(path.read_text())
            del manifest["source_sha256"][relative]
            path.write_text(json.dumps(manifest))
    else:
        campaign["source"] = str(out)
        (out / "campaign.json").write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match=r"[Ss]ource"):
        analyze(out / "campaign.json")


@pytest.mark.parametrize("change", ["half_budget", "seed", "heldout", "matrix_lr", "model"])
def test_analyzer_rejects_self_consistent_trial_protocol_changes(completed_muont_campaign, change):
    _, out, campaign = completed_muont_campaign
    trial = next(trial for trial in campaign["trials"] if trial["name"] == "muont_lr_up")
    path = Path(trial["config"])
    config = json.loads(path.read_text())
    if change == "half_budget":
        config["training"]["steps"] //= 2
        config["training"]["tokens"] //= 2
        trial.update(steps=config["training"]["steps"], tokens=config["training"]["tokens"])
    elif change == "seed":
        config["training"]["seed"] = 1
        trial["seed"] = 1
    elif change == "heldout":
        config["training"]["heldout_windows"] = 32
    elif change == "matrix_lr":
        config["optimizer"]["lr"] *= 2
        trial["matrix_lr"] = config["optimizer"]["lr"]
    else:
        config["model"]["hidden"] *= 2
    path.write_text(json.dumps(config))
    trial["config_sha256"] = sha256(path)
    (out / "campaign.json").write_text(json.dumps(campaign))
    manifest_path = out / "runs" / trial["name"] / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = config
    manifest["completed_steps"] = config["training"]["steps"]
    manifest_path.write_text(json.dumps(manifest))
    if change == "half_budget":
        history_path = manifest_path.parent / "metrics_from_000000.jsonl"
        rows = [json.loads(line) for line in history_path.read_text().splitlines()][:160]
        rows[-1].update(
            selection_loss=1.0,
            c4_loss=2.0,
            heldout_losses={f"heldout/{name}": 1.0 for name in config["data"]["mixture_weights"]},
        )
        history_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        checkpoint = manifest_path.parent / "checkpoints/step_000160"
        checkpoint.mkdir()
        (checkpoint / "complete.json").write_text(json.dumps({"step": 160, "world": 8}))
        for rank in range(8):
            (checkpoint / f"rank_{rank}.pt").write_bytes(b"rank checkpoint fixture")
    with pytest.raises(ValueError, match=r"protocol|permitted LR transformation"):
        analyze(out / "campaign.json")
