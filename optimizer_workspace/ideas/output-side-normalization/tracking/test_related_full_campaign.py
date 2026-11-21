# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline selected-LR confirmation with real recipes and temporary run evidence."""

import json
import shutil
from pathlib import Path

import pytest
import related_campaign as short
import related_full_campaign as full
from experiment import launch
from experiment.related_configure import comparison_configuration


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def rewrite(path, change):
    record = json.loads(path.read_text())
    change(record)
    save(path, record)


@pytest.fixture
def study(tmp_path):
    source = tmp_path / "source"
    for path in (short.TASK / "experiment").rglob("*.py"):
        target = source / path.relative_to(short.TASK)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for folder, _ in short.PROFILES.values():
        shutil.copytree(short.TASK / "experiment/comparisons" / folder, source / "experiment/comparisons" / folder)
    manifest_path = short.TASK / "experiment/upstream_manifest.json"
    manifest = short.read(manifest_path)
    shutil.copyfile(manifest_path, source / "experiment/upstream_manifest.json")
    prefix = f"artifacts/external-sources/Toner-{manifest['revision']}"
    inputs = [
        name
        for name in manifest["files"]
        if Path(name).suffix == ".py"
        and (Path(name).parent == Path("reproduce/marin") or Path(name).is_relative_to("src/toner"))
    ]
    inputs += [
        "reproduce/marin/data_plan.json",
        "reproduce/marin/configs/130m/muon_0p5m.json",
        "reproduce/marin/configs/130m/muont_0p5m.json",
    ]
    for name in [*(f"{prefix}/{name}" for name in inputs), short.MOUSSE_REFERENCE]:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(short.TASK / name, target)
    root = tmp_path / "output"
    shard = root / "data/dclm/tiny.bin"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"\x00\x00\x00\x00")
    save(
        root / "data/experiment_receipt.json",
        {
            "root": str(root / "data"),
            "data_plan_sha256": short.sha256(source / prefix / "reproduce/marin/data_plan.json"),
            "files": [
                {
                    "file": "dclm/tiny.bin",
                    "bytes": 4,
                    "mtime_ns": shard.stat().st_mtime_ns,
                    "sha256": short.sha256(shard),
                }
            ],
        },
    )
    campaign = short.prepare(root, source=source)
    return root, source, campaign


def write_trial(campaign, trial, *, completed=None, state="complete", loss=3.0, split=None):
    directory = Path(trial["out"])
    config = short.read(Path(trial["config"]))
    steps = config["training"]["steps"] if completed is None else completed
    save(
        directory / "run.json",
        {
            "status": state,
            "completed_steps": steps,
            "execution_check": False,
            "resume": None,
            "config": config,
            "data": campaign["data"],
            "data_receipt_sha256": campaign["data_receipt_sha256"],
            "source_revision": campaign["source_revision"],
            "source_sha256": campaign["source_sha256"],
            "world_size": short.WORLD,
        },
    )
    rows = [
        {
            "step": step,
            "tokens": step * short.BATCH_TOKENS,
            "train_loss": 4.0,
            "step_seconds": 1.0,
            "elapsed_seconds": step + 10,
            "end_to_end_seconds": step + 11,
        }
        for step in range(1, steps + 1)
    ]
    if rows and steps == config["training"]["steps"]:
        weights = config["data"]["mixture_weights"]
        heldout = {f"heldout/{name}": loss for name in weights}
        mixture = sum(weight * heldout[f"heldout/{name}"] for name, weight in weights.items()) / sum(weights.values())
        rows[-1].update(selection_loss=mixture, heldout_losses=heldout, c4_loss=2.0 + trial["lr_index"])
    segments = [(0, rows)] if split is None else [(0, rows[:split]), (split, rows[split:])]
    for start, segment in segments:
        (directory / f"metrics_from_{start:06d}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in segment))
    if state == "complete" and steps == config["training"]["steps"]:
        checkpoint = directory / f"checkpoints/step_{steps:06d}"
        save(checkpoint / "complete.json", {"step": steps, "world": short.WORLD})
        for rank in range(short.WORLD):
            (checkpoint / f"rank_{rank}.pt").write_bytes(b"offline nonempty checkpoint fixture")
    return directory


@pytest.fixture
def completed_study(study):
    root, source, campaign = study
    for trial in campaign["trials"]:
        # Different arms deliberately choose different LRs, independently of C4.
        preferred = short.ARMS.index(trial["arm"]) % 3
        write_trial(campaign, trial, loss=3.0 + abs(trial["lr_index"] - preferred))
    return root, source, campaign


@pytest.fixture
def prepared(completed_study):
    root, source, short_campaign = completed_study
    return root, source, short_campaign, full.prepare(root)


def test_prepare_requires_all_eighteen_short_trials_before_any_full_files(study):
    root, source, campaign = study
    for trial in campaign["trials"][:-1]:
        write_trial(campaign, trial)
    with pytest.raises(ValueError, match="eighteen"):
        full.prepare(root)
    assert not (source / full.CONFIG_DIRECTORY).exists()
    assert not full.campaign_path(root).parent.exists()
    assert full.status(root)["status"] == "not_prepared"


def test_prepare_writes_actual_selected_recipes_and_deduplicated_validator_manifests(completed_study, monkeypatch):
    root, source, short_campaign = completed_study
    before = {str(path): short.sha256(path) for path in source.rglob("*") if path.is_file()}
    selected = short.analyze(root)["completed_arm_lr_choices"]
    campaign = full.prepare(root)
    monkeypatch.setattr(launch, "TASK", source)
    assert short.read(full.campaign_path(root)) == campaign
    assert len(campaign["trials"]) == 12
    assert len({trial["out"] for trial in campaign["trials"]}) == 12
    assert len(campaign["waves"]) == 4
    assert sum(len(wave["trials"]) for wave in campaign["waves"]) == 16
    assert len(campaign["short_confirmation"]["evidence"]) == 18
    assert campaign["short_confirmation"]["choices"] == selected
    lookup = {trial["name"]: trial for trial in short_campaign["trials"]}
    for trial in campaign["trials"]:
        choice = selected[trial["arm"]]
        chosen = lookup[choice["trial"]]
        expected = comparison_configuration(
            trial["arm"],
            batch="0p5m",
            seed=trial["seed"],
            microbatch=4,
            lr_multiplier=chosen["lr_multiplier"],
            stage="full",
        )
        actual = short.read(Path(trial["config"]))
        assert actual == expected
        assert actual["training"]["steps"] == 5120
        assert actual["training"]["tokens"] == 2684354560
        assert actual["schedule"]["stable_steps"] == 1024
        assert actual["schedule"]["decay_steps"] == 4096
        assert actual["optimizer"]["lr"] == choice["matrix_lr"]
        assert actual["experiment"]["comparison"]["quality_claim"] == "none"
        assert trial["seed"] in (1, 2)
        assert len(trial["aliases"]) == (2 if trial["arm"] in short.ARMS[:2] else 1)
    for wave in campaign["waves"]:
        manifest = short.read(Path(wave["manifest"]))
        name, entries = launch.wave_entries(Path(wave["manifest"]))
        assert name == wave["wave_name"]
        assert [entry["name"] for entry in entries] == wave["trials"]
        assert all(entry["path"].is_file() for entry in entries)
        assert len(manifest["wave"]) == 4
        assert len({entry["name"] for entry in manifest["wave"]}) == 4
        assert [entry["name"] for entry in manifest["wave"]] == wave["trials"]
        for entry in manifest["wave"]:
            assert short.read(source / entry["config"])["training"]["seed"] == wave["seed"]
    assert {name: short.sha256(Path(name)) for name in before} == before
    assert not (full.campaign_path(root).parent / "runs").exists()
    assert full.status(root)["status"] == "not_started"
    with pytest.raises(FileExistsError):
        full.prepare(root)


@pytest.mark.parametrize("damage", ["history", "checkpoint", "selection"])
def test_failed_short_evidence_blocks_preparation(completed_study, damage):
    root, source, campaign = completed_study
    trial = campaign["trials"][-1]
    directory = Path(trial["out"])
    if damage == "checkpoint":
        (directory / "checkpoints/step_000320/rank_7.pt").unlink()
    else:
        path = directory / "metrics_from_000000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "history":
            del rows[100]
        else:
            rows[-1]["selection_loss"] += 1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        full.prepare(root)
    assert not (source / full.CONFIG_DIRECTORY).exists()


def test_status_tracks_full_progress_without_using_manifest_only_completion(prepared):
    root, _, _, campaign = prepared
    trial = campaign["trials"][0]
    directory = write_trial(campaign, trial, completed=320, state="running")
    progress = full.status(root)
    assert progress["status"] == "running"
    assert progress["trials"][0]["observed_steps"] == 320
    assert progress["trials"][0]["planned_steps"] == 5120
    rewrite(directory / "run.json", lambda manifest: manifest.update(status="complete", completed_steps=5120))
    result = full.status(root)
    assert result["status"] == "needs_attention"
    assert not result["trials"][0]["complete"]
    assert full.analyze(root)["paired_comparisons"] == {}


@pytest.mark.parametrize(
    "damage", ["missing_rank", "wrong_receipt", "selection", "gap", "source", "seed", "short_resume"]
)
def test_invalid_full_endpoint_withholds_pairing(prepared, damage):
    root, _, short_campaign, campaign = prepared
    trial = campaign["trials"][0]
    directory = write_trial(campaign, trial)
    if damage == "missing_rank":
        (directory / "checkpoints/step_005120/rank_7.pt").unlink()
    elif damage == "wrong_receipt":
        rewrite(directory / "checkpoints/step_005120/complete.json", lambda record: record.update(step=320))
    elif damage in ("selection", "gap"):
        path = directory / "metrics_from_000000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "selection":
            rows[-1]["selection_loss"] += 0.1
        else:
            del rows[500]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif damage == "source":
        rewrite(directory / "run.json", lambda record: record["source_sha256"].update({"experiment/train.py": "0" * 64}))
    elif damage == "seed":
        rewrite(directory / "run.json", lambda record: record["config"]["training"].update(seed=0))
    else:
        rewrite(
            directory / "run.json",
            lambda record: record.update(
                resume=str(Path(short_campaign["trials"][0]["out"]) / "checkpoints/step_000320")
            ),
        )
    result = full.analyze(root)
    assert result["status"] == "needs_attention"
    assert result["trials"][0]["status"] == "invalid_evidence"
    assert result["paired_comparisons"] == {}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "damage", ["short_history", "short_rank", "choice", "full_config", "full_wave", "data", "data_file", "source"]
)
def test_frozen_selection_and_full_binding_cannot_change(prepared, damage):
    root, source, short_campaign, campaign = prepared
    if damage == "short_history":
        path = Path(short_campaign["trials"][0]["out"]) / "metrics_from_000000.jsonl"
        path.write_text(path.read_text().replace('"train_loss": 4.0', '"train_loss": 4.1'))
    elif damage == "short_rank":
        path = Path(short_campaign["trials"][0]["out"]) / "checkpoints/step_000320/rank_7.pt"
        path.write_bytes(b"changed rank evidence")
    elif damage == "choice":
        rewrite(
            full.campaign_path(root),
            lambda record: record["short_confirmation"]["choices"]["muon_control"].update(lr_index=2),
        )
    elif damage == "full_config":
        rewrite(Path(campaign["trials"][0]["config"]), lambda config: config["optimizer"].update(lr=0.5))
    elif damage == "full_wave":
        rewrite(Path(campaign["waves"][0]["manifest"]), lambda record: record["wave"].pop())
    elif damage == "data":
        rewrite(root / "data/experiment_receipt.json", lambda record: record.update(root="/other"))
    elif damage == "data_file":
        (root / "data/dclm/tiny.bin").write_bytes(b"changed")
    else:
        (source / "experiment/train.py").write_text("# Changed source\n")
    with pytest.raises(ValueError):
        full.status(root)


def test_paired_comparisons_require_both_seeds_and_never_reselect_full_lr(prepared):
    root, _, _, campaign = prepared
    for trial in campaign["trials"]:
        if trial["seed"] == 1:
            write_trial(campaign, trial, loss=3.0 + 0.1 * short.ARMS.index(trial["arm"]), split=2560)
    assert full.analyze(root)["paired_comparisons"] == {}
    for trial in campaign["trials"]:
        if trial["seed"] == 2 and trial["arm"] in ("muon_control", "go_muon_control"):
            write_trial(campaign, trial, loss=3.0 + 0.1 * short.ARMS.index(trial["arm"]))
    partial = full.analyze(root)
    paired = partial["paired_comparisons"]["go_muon_control_minus_muon_control"]
    assert paired["seeds"] == [1, 2]
    assert paired["mean_final_selection_loss_delta"] == pytest.approx(0.2)
    assert set(partial["paired_comparisons"]) == {"go_muon_control_minus_muon_control"}
    for trial in campaign["trials"]:
        if not (Path(trial["out"]) / "run.json").exists():
            write_trial(campaign, trial, loss=3.0 + 0.1 * short.ARMS.index(trial["arm"]))
    before = {str(path): short.sha256(path) for path in root.rglob("*") if path.is_file()}
    result = full.analyze(root)
    assert result["status"] == "complete"
    assert result["completed_trials"] == 12
    assert len(result["paired_comparisons"]) == 10
    assert result["quality_claim"] == "none"
    assert "completed_arm_lr_choices" not in result
    for comparison in result["paired_comparisons"].values():
        assert comparison["quality_claim"] == "none"
        assert all(item["pairing"]["per_update_paired"] for item in comparison["per_seed"].values())
        assert all(len(item["training_windows"]) == 5 for item in comparison["per_seed"].values())
    assert {str(path): short.sha256(path) for path in root.rglob("*") if path.is_file()} == before
