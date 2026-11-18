# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise new-grid identity, parent provenance, token accounting and five-point choices."""

import copy
import os
import subprocess
from pathlib import Path

import pytest
from analyze_fixed_age_tuning import analyze as analyze_initial
from analyze_lr_boundary import analyze
from analyze_lr_ceiling import analyze as analyze_ceiling
from fixed_age_protocol import analyzer_name, environment, multipliers, predecessor_wave, require_gpu_environment
from launch_wave5 import PYTHON
from prepare_fixed_age_tuning import prepare as prepare_initial
from prepare_lr_boundary import MULTIPLIERS, prepare
from prepare_lr_ceiling import GPU_VISIBILITY
from prepare_lr_ceiling import MULTIPLIERS as CEILING_MULTIPLIERS
from prepare_lr_ceiling import prepare as prepare_ceiling
from prepare_tuning import TRIAL_TOKENS, sha256
from test_fixed_age_tuning import complete_trial, write_record
from test_fixed_age_tuning import fixed_source as source_fixture
from wave7_release import ARMS, candidate_inventory, read


@pytest.fixture
def failed_ceiling(completed_expansion):
    root, candidate, parent = completed_expansion
    directory = root / "campaigns/failed-ceiling"
    campaign = prepare_ceiling(root, parent, directory)
    log = directory / "validation/muont_lr_ceiling_4x_continuous.log"
    log.parent.mkdir()
    log.write_text("RuntimeError: No CUDA GPUs are available\n" * 8)
    command = ["fixture-launcher", "--config", campaign["trials"][0]["config"], "--max-steps", "24"]
    write_record(directory / "validation-binding.json", {"plan": {"trials": [{"commands": [command]}]}})
    write_record(directory / "execution-binding.json", {"status": "fixture"})
    write_record(
        directory / "validation-state.json",
        {
            "status": "needs_attention",
            "exit_code": 1,
            "keeper_restore": 0,
            "binding_sha256": sha256(directory / "validation-binding.json"),
        },
    )
    write_record(
        directory / "validation-gpu.json",
        {
            "status": "needs_attention",
            "campaign_sha256": sha256(directory / "campaign.json"),
            "trials": [{"checks": {"continuation": False}} for _ in campaign["trials"]],
            "gpu_commands": [
                {
                    "trial": "muont_lr_ceiling_4x",
                    "command": command,
                    "exit_code": 1,
                    "log": str(log),
                    "log_sha256": sha256(log),
                }
            ],
        },
    )
    for name in ("binding", "installation", "validate", "state"):
        write_record(directory / f"queue/{name}.json", {"status": "needs_attention"})
    (directory / "queue/stopped_at").write_text("2026-11-17T12:42:07+08:00\n")
    files = [
        "campaign.json",
        "validation-binding.json",
        "execution-binding.json",
        "validation-state.json",
        "validation-gpu.json",
        "queue/binding.json",
        "queue/installation.json",
        "queue/validate.json",
        "queue/state.json",
        "queue/stopped_at",
        "validation/muont_lr_ceiling_4x_continuous.log",
    ]
    observation = root / "run-preparation/failed-ceiling-terminal.json"
    write_record(
        observation,
        {
            "status": "terminal_zero_update_cuda_mask_failure_keeper_verified",
            "campaign": str(directory / "campaign.json"),
            "queue_process": "absent",
            "controller_process": "exited",
            "marked_children": {},
            "files_sha256": {name: sha256(directory / name) for name in files},
        },
    )
    return root, candidate, parent, directory / "campaign.json", observation


def test_ceiling_recovery_preserves_original_evidence_trial_bytes_and_search_allowances(failed_ceiling, monkeypatch):
    root, candidate, parent, failed, observation = failed_ceiling
    original = read(failed)
    inventory = candidate_inventory(candidate)
    evidence = {path: sha256(path) for path in failed.parent.rglob("*") if path.is_file()}
    out = root / "campaigns/recovery"
    campaign = prepare_ceiling(root, parent, out, failed_campaign=failed, terminal_observation=observation)
    result = analyze_ceiling(out / "campaign.json")
    assert result["status"] == "incomplete"
    assert result["completed_cumulative_family_lr_choices"] == {}
    assert result["observed_tokens_by_family"] == dict.fromkeys(ARMS, 0)
    assert campaign["max_attempts_per_family"] == original["max_attempts_per_family"]
    assert campaign["remaining_attempts_per_family"] == original["remaining_attempts_per_family"]
    assert campaign["recovery"]["failed_tuning_tokens_by_family"] == dict.fromkeys(ARMS, 0)
    assert campaign["recovery"]["failed_validation_commands"] == 1
    assert candidate_inventory(candidate) == inventory
    assert {path: sha256(path) for path in evidence} == evidence
    for actual, previous in zip(campaign["trials"], original["trials"], strict=True):
        assert actual["config"] != previous["config"]
        assert Path(actual["config"]).read_bytes() == Path(previous["config"]).read_bytes()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="visibility differs"):
        require_gpu_environment(out / "campaign.json")
    child = subprocess.run(
        [PYTHON, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
        env=os.environ | environment(out / "campaign.json"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip() == GPU_VISIBILITY
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_VISIBILITY)
    require_gpu_environment(out / "campaign.json")
    assert not (out / "validation").exists()
    assert not (out / "queue").exists()


@pytest.mark.parametrize("fault", ["history", "checkpoint", "trial", "log_drift", "running_queue", "allowance", "mask"])
def test_ceiling_recovery_rejects_nonzero_or_changed_failure_and_protocol(failed_ceiling, fault):
    root, _, parent, failed, observation = failed_ceiling
    out = root / "campaigns/recovery"
    if fault in ("allowance", "mask"):
        campaign = prepare_ceiling(root, parent, out, failed_campaign=failed, terminal_observation=observation)
        if fault == "allowance":
            campaign["remaining_attempts_per_family"] += 1
        else:
            campaign["recovery"]["cuda_visible_devices"] = "0"
        write_record(out / "campaign.json", campaign)
        with pytest.raises(ValueError):
            analyze_ceiling(out / "campaign.json")
        return
    if fault == "history":
        write_record(failed.parent / "validation/trajectory/metrics_from_000000.jsonl", {"step": 1})
    elif fault == "checkpoint":
        path = failed.parent / "validation/trajectory/checkpoints/step_000022/rank_0.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"nonzero checkpoint")
    elif fault == "trial":
        (failed.parent / "runs").mkdir()
    elif fault == "log_drift":
        (failed.parent / "validation/muont_lr_ceiling_4x_continuous.log").write_text("new failure")
    else:
        record = read(observation)
        record["queue_process"] = "running"
        write_record(observation, record)
    with pytest.raises(ValueError):
        prepare_ceiling(root, parent, out, failed_campaign=failed, terminal_observation=observation)
    assert not out.exists()


@pytest.fixture
def fixed_source(tmp_path):
    return source_fixture.__wrapped__(tmp_path)


@pytest.fixture
def completed_parent(fixed_source):
    root, candidate, source = fixed_source
    directory = root / "campaigns/initial"
    campaign = prepare_initial(root, source, directory)
    for family in ARMS:
        trials = [trial for trial in campaign["trials"] if trial["family"] == family]
        for index, trial in enumerate(trials):
            complete_trial(campaign, trial, directory / "runs" / trial["name"], candidate, 5.0 - index, 2.0 + index)
    write_record(directory / "analysis.json", analyze_initial(directory / "campaign.json"))
    write_record(directory / "execution-binding.json", {"status": "fixture"})
    upload = {
        "status": "verified",
        "verified_at": "2026-11-17T03:00:00+08:00",
        "campaign_sha256": sha256(directory / "campaign.json"),
        "execution_binding_sha256": sha256(directory / "execution-binding.json"),
        "runs": {},
    }
    for trial in campaign["trials"]:
        run = directory / "runs" / trial["name"]
        checkpoint = run / "checkpoints/step_000320"
        upload["runs"][trial["name"]] = {
            "status": "verified",
            "campaign_sha256": upload["campaign_sha256"],
            "execution_binding_sha256": upload["execution_binding_sha256"],
            "trial_config_sha256": trial["config_sha256"],
            "manifest_sha256": sha256(run / "run.json"),
            "metrics_sha256": sha256(run / "metrics_from_000000.jsonl"),
            "checkpoint_receipt_sha256": sha256(checkpoint / "complete.json"),
            "checkpoint_rank_bytes": [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)],
            "verification": {
                "state": "FINISHED",
                "point_counts": {"tokens": 320, "train/loss": 320, "eval/selection_loss": 9, "eval/c4_loss": 9},
            },
        }
    write_record(directory / "swanlab-upload.json", upload)
    return root, candidate, directory / "campaign.json"


def test_boundary_preserves_frozen_parent_and_source_and_prepares_equal_allowances(completed_parent):
    root, candidate, parent = completed_parent
    inventory, parent_hash = candidate_inventory(candidate), sha256(parent)
    out = root / "campaigns/boundary"
    campaign = prepare(root, parent, out)
    result = analyze(out / "campaign.json")
    assert len(campaign["trials"]) == 8
    assert campaign["initial_trials_per_family"] == 2
    assert campaign["initial_tokens_per_family"] == 2 * TRIAL_TOKENS
    assert campaign["cumulative_tokens_per_family"] == 5 * TRIAL_TOKENS
    assert campaign["max_attempts_per_family"] == 6
    assert campaign["max_tokens_per_family"] == 6 * TRIAL_TOKENS
    assert campaign["remaining_attempts_per_family"] == 4
    assert result["status"] == "incomplete"
    assert result["completed_cumulative_family_lr_choices"] == {}
    assert result["observed_tokens_by_family"] == dict.fromkeys(ARMS, 0)
    assert result["cumulative_observed_tokens_by_family"] == dict.fromkeys(ARMS, 3 * TRIAL_TOKENS)
    assert candidate_inventory(candidate) == inventory
    assert sha256(parent) == parent_hash
    assert not (out / "runs").exists()
    assert not (out / "validation-binding.json").exists()
    for trial in campaign["trials"]:
        reference = read(Path(trial["reference_config"]))
        config = read(Path(trial["config"]))
        assert config["optimizer"]["lr"] == reference["optimizer"]["lr"] * trial["lr_multiplier"]
        assert {key: value for key, value in config["optimizer"].items() if key != "lr"} == {
            key: value for key, value in reference["optimizer"].items() if key != "lr"
        }
        assert config["root"] == reference["root"]
        assert config["statistics"] == reference["statistics"]
        assert config["data"] == reference["data"]
        assert config["experiment"]["refresh_install_age"] == 1
        assert config["training"]["steps"] == 320
        assert config["schedule"]["stable_steps"] == 64
        assert config["schedule"]["decay_steps"] == 256
    assert [trial["lr_multiplier"] for trial in campaign["trials"]] == [
        multiplier for _, multiplier in MULTIPLIERS for _ in ARMS
    ]
    with pytest.raises(FileExistsError):
        prepare(root, parent, out)
    assert read(out / "campaign.json") == campaign


@pytest.mark.parametrize("fault", ["partial", "analysis", "cloud_count", "cloud_identity", "source", "mixture"])
def test_boundary_rejects_changed_parent_without_writing_new_campaign(completed_parent, fault):
    root, candidate, parent = completed_parent
    if fault in ("partial", "cloud_count", "cloud_identity"):
        path = parent.parent / "swanlab-upload.json"
        record = read(path)
        if fault == "partial":
            record["status"] = "partial_verified"
        elif fault == "cloud_count":
            record["runs"]["muont_lr_up"]["verification"]["point_counts"]["train/loss"] = 319
        else:
            record["runs"]["muont_lr_up"]["metrics_sha256"] = "changed"
    elif fault == "analysis":
        path = parent.parent / "analysis.json"
        record = read(path)
        record["completed_family_lr_choices"]["muont"]["matrix_lr"] *= 2
    elif fault == "source":
        (candidate / "experiment/train.py").write_text("# Changed fixture trainer\n")
    else:
        path = parent
        record = read(path)
        record["selection"]["weights"]["dclm"] *= 2
    if fault != "source":
        write_record(path, record)
    output = root / "campaigns/boundary"
    with pytest.raises(ValueError):
        prepare(root, parent, output)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["lr", "budget", "grid", "parent_upload", "refresh_age", "identity"])
def test_boundary_rejects_self_consistent_new_protocol_changes(completed_parent, fault):
    root, _, parent = completed_parent
    out = root / "campaigns/boundary"
    campaign = prepare(root, parent, out)
    if fault in ("lr", "budget"):
        trial = campaign["trials"][0]
        path = Path(trial["config"])
        config = read(path)
        if fault == "lr":
            config["optimizer"]["lr"] *= 2
            trial["matrix_lr"] = config["optimizer"]["lr"]
        else:
            config["training"]["tokens"] //= 2
            config["training"]["steps"] //= 2
            trial.update(tokens=config["training"]["tokens"], steps=config["training"]["steps"])
        write_record(path, config)
        trial["config_sha256"] = sha256(path)
    elif fault == "grid":
        campaign["new_grid"][0]["multiplier"] = 4.0
    elif fault == "parent_upload":
        campaign["parent_upload_sha256"] = "changed"
    elif fault == "refresh_age":
        campaign["refresh_install_age"] = True
    else:
        campaign["trials"][-1] = copy.deepcopy(campaign["trials"][0])
    write_record(out / "campaign.json", campaign)
    with pytest.raises(ValueError):
        analyze(out / "campaign.json")


def test_completed_expansion_ranks_five_points_not_c4_and_failed_tokens_stay_charged(completed_parent):
    root, candidate, parent = completed_parent
    out = root / "campaigns/boundary"
    campaign = prepare(root, parent, out)
    for family in ARMS:
        trials = [trial for trial in campaign["trials"] if trial["family"] == family]
        for index, trial in enumerate(trials):
            complete_trial(campaign, trial, out / "runs" / trial["name"], candidate, 2.9 - index / 10, 6.0 + index)
    result = analyze(out / "campaign.json")
    assert result["status"] == "complete"
    assert result["cumulative_observed_tokens_by_family"] == dict.fromkeys(ARMS, 5 * TRIAL_TOKENS)
    for family, choice in result["completed_cumulative_family_lr_choices"].items():
        assert choice["trial"] == f"{family}_{MULTIPLIERS[-1][0]}"
        assert choice["upper_boundary_selected"]
    for trial in result["trials"]:
        assert trial["late_evaluation_tokens"] == {key: [TRIAL_TOKENS] for key in ("selection_loss", "c4_loss")}
    path = out / "runs" / campaign["trials"][-1]["name"] / "run.json"
    record = read(path)
    record["status"] = "failed"
    write_record(path, record)
    result = analyze(out / "campaign.json")
    assert result["status"] == "incomplete"
    assert ARMS[-1] not in result["completed_cumulative_family_lr_choices"]
    assert result["cumulative_observed_tokens_by_family"][ARMS[-1]] == 5 * TRIAL_TOKENS


def test_cumulative_choice_can_remain_at_parent_grid(completed_parent):
    root, candidate, parent = completed_parent
    out = root / "campaigns/boundary"
    campaign = prepare(root, parent, out)
    for trial in campaign["trials"]:
        complete_trial(campaign, trial, out / "runs" / trial["name"], candidate, 6.0, 1.0)
    result = analyze(out / "campaign.json")
    for family, choice in result["completed_cumulative_family_lr_choices"].items():
        assert choice["trial"] == f"{family}_lr_up"
        assert choice["campaign"] == str(parent)
        assert not choice["upper_boundary_selected"]


@pytest.fixture
def completed_expansion(completed_parent, request):
    root, candidate, initial = completed_parent
    out = root / "campaigns/boundary"
    campaign = prepare(root, initial, out)
    for family in ARMS:
        trials = [trial for trial in campaign["trials"] if trial["family"] == family]
        for index, trial in enumerate(trials):
            selection = 4.2 - index / 10 if getattr(request, "param", None) == "resolved" else 2.9 - index / 10
            complete_trial(campaign, trial, out / "runs" / trial["name"], candidate, selection, 6.0 + index)
    write_record(out / "analysis.json", analyze(out / "campaign.json"))
    write_record(out / "execution-binding.json", {"status": "completed_expansion_fixture"})
    upload = {
        "status": "verified",
        "verified_at": "2026-11-17T07:42:00+08:00",
        "campaign_sha256": sha256(out / "campaign.json"),
        "execution_binding_sha256": sha256(out / "execution-binding.json"),
        "runs": {},
    }
    for trial in campaign["trials"]:
        directory = out / "runs" / trial["name"]
        checkpoint = directory / "checkpoints/step_000320"
        upload["runs"][trial["name"]] = {
            "status": "verified",
            "campaign_sha256": upload["campaign_sha256"],
            "execution_binding_sha256": upload["execution_binding_sha256"],
            "trial_config_sha256": trial["config_sha256"],
            "manifest_sha256": sha256(directory / "run.json"),
            "metrics_sha256": sha256(directory / "metrics_from_000000.jsonl"),
            "checkpoint_receipt_sha256": sha256(checkpoint / "complete.json"),
            "checkpoint_rank_bytes": [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)],
            "verification": {
                "state": "FINISHED",
                "point_counts": {"tokens": 320, "train/loss": 320, "eval/selection_loss": 9, "eval/c4_loss": 9},
            },
        }
    write_record(out / "swanlab-upload.json", upload)
    write_record(
        root / "run-preparation/wave9-launch-binding.json",
        {"status": "frozen_validated_launch_release", "root": str(root)},
    )
    return root, candidate, out / "campaign.json"


def test_ceiling_prepares_equal_seventh_point_allowances_and_preserves_executed_sources(completed_expansion):
    root, candidate, parent = completed_expansion
    inventory, parent_hash = candidate_inventory(candidate), sha256(parent)
    out = root / "campaigns/ceiling"
    campaign = prepare_ceiling(root, parent, out)
    result = analyze_ceiling(out / "campaign.json")
    assert result["status"] == "incomplete"
    assert result["completed_cumulative_family_lr_choices"] == {}
    assert result["cumulative_observed_tokens_by_family"] == dict.fromkeys(ARMS, 5 * TRIAL_TOKENS)
    assert campaign["max_attempts_per_family"] == 4
    assert campaign["remaining_attempts_per_family"] == 2
    assert campaign["cumulative_tokens_per_family"] == 7 * TRIAL_TOKENS
    assert multipliers(out / "campaign.json") == CEILING_MULTIPLIERS
    assert analyzer_name(out / "campaign.json") == "analyze_lr_ceiling.py"
    assert predecessor_wave(out / "campaign.json") == "wave9"
    assert candidate_inventory(candidate) == inventory
    assert sha256(parent) == parent_hash
    assert not (out / "validation").exists()
    assert not (out / "runs").exists()
    for trial in campaign["trials"]:
        reference = read(Path(trial["reference_config"]))
        config = read(Path(trial["config"]))
        assert config["optimizer"]["lr"] == reference["optimizer"]["lr"] * trial["lr_multiplier"]
        assert config["statistics"] == reference["statistics"]
        assert config["training"]["steps"] == 320
        assert config["schedule"]["stable_steps"] == 64
        assert config["schedule"]["decay_steps"] == 256
        assert config["training"]["heldout_windows"] == 64
    with pytest.raises(FileExistsError):
        prepare_ceiling(root, parent, out)


@pytest.mark.parametrize("best", ["expansion", "ceiling"])
def test_ceiling_ranks_all_seven_endpoints_not_c4_and_withholds_failed_family(completed_expansion, best):
    root, candidate, parent = completed_expansion
    out = root / "campaigns/ceiling"
    campaign = prepare_ceiling(root, parent, out)
    for trial in campaign["trials"]:
        value = 2.0 if best == "ceiling" else 5.0
        complete_trial(campaign, trial, out / "runs" / trial["name"], candidate, value, 0.1)
    result = analyze_ceiling(out / "campaign.json")
    assert result["status"] == "complete"
    assert result["cumulative_observed_tokens_by_family"] == dict.fromkeys(ARMS, 7 * TRIAL_TOKENS)
    for family, choice in result["completed_cumulative_family_lr_choices"].items():
        assert choice["campaign"] == str(out / "campaign.json" if best == "ceiling" else parent)
        assert choice["trial"].startswith(family + "_")
    path = out / "runs" / campaign["trials"][-1]["name"] / "run.json"
    record = read(path)
    record["status"] = "failed"
    write_record(path, record)
    result = analyze_ceiling(out / "campaign.json")
    assert result["status"] == "incomplete"
    assert ARMS[-1] not in result["completed_cumulative_family_lr_choices"]
    assert result["cumulative_observed_tokens_by_family"][ARMS[-1]] == 7 * TRIAL_TOKENS


@pytest.mark.parametrize("completed_expansion", ["resolved"], indirect=True)
def test_ceiling_does_not_spend_more_trials_when_parent_already_resolves_boundary(completed_expansion):
    root, _, parent = completed_expansion
    out = root / "campaigns/ceiling"
    with pytest.raises(ValueError, match="unresolved upper-edge"):
        prepare_ceiling(root, parent, out)
    assert not out.exists()


@pytest.mark.parametrize("fault", ["parent_partial", "parent_counts", "lr", "budget", "predecessor", "quota"])
def test_ceiling_rejects_changed_parent_or_self_consistent_protocol(completed_expansion, fault):
    root, _, parent = completed_expansion
    out = root / "campaigns/ceiling"
    if fault.startswith("parent"):
        path = parent.parent / "swanlab-upload.json"
        record = read(path)
        if fault == "parent_partial":
            record["status"] = "partial_verified"
        else:
            record["runs"][next(iter(record["runs"]))]["verification"]["point_counts"]["train/loss"] = 319
        write_record(path, record)
        with pytest.raises(ValueError):
            prepare_ceiling(root, parent, out)
        assert not out.exists()
        return
    campaign = prepare_ceiling(root, parent, out)
    if fault in ("lr", "budget"):
        trial = campaign["trials"][0]
        path = Path(trial["config"])
        config = read(path)
        if fault == "lr":
            config["optimizer"]["lr"] *= 2
            trial["matrix_lr"] = config["optimizer"]["lr"]
        else:
            config["training"]["steps"] //= 2
            config["training"]["tokens"] //= 2
            trial.update(steps=config["training"]["steps"], tokens=config["training"]["tokens"])
        write_record(path, config)
        trial["config_sha256"] = sha256(path)
    elif fault == "predecessor":
        campaign["predecessor_release_sha256"] = "different"
    else:
        campaign["remaining_attempts_per_family"] = 4
    write_record(out / "campaign.json", campaign)
    with pytest.raises(ValueError):
        analyze_ceiling(out / "campaign.json")
