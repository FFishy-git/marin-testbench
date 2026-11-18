# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise fixed-age LR preparation and selection with real temporary evidence files."""

import copy
import json
import os
import subprocess
from pathlib import Path
from subprocess import run as run_process

import pytest
import run_fixed_age_tuning
import run_fixed_age_validation
import validate_fixed_age_tuning
from analyze_fixed_age_tuning import analyze
from analyze_wave7_snapshot import capture
from compare_runs import WINDOW, compare, history
from launch_wave5 import WORKLOAD
from prepare_fixed_age_tuning import prepare
from prepare_tuning import sha256
from run_fixed_age_validation import COMPARISONS, require_actual_receipt, require_completed_wave
from validate_fixed_age_tuning import executed_trajectory, plan
from validate_wave5 import ENVIRONMENT
from wave7_release import ARMS, GATES, candidate_inventory, read, sources

TASK = Path(__file__).resolve().parents[1]


def test_cpu_hidden_actual_validation_refuses_before_creating_execution_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    campaign = tmp_path / "campaign.json"
    with pytest.raises(RuntimeError, match="CPU-only CUDA visibility"):
        run_fixed_age_validation.execute(campaign)
    assert list(tmp_path.iterdir()) == []


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


@pytest.fixture
def fixed_source(tmp_path):
    candidate = tmp_path / "candidate"
    reference = read(TASK / "experiment/configs/inner_outer_leftdiag_0p5m_seed0.json")
    for arm in ARMS:
        config = copy.deepcopy(reference)
        config["experiment"].update(
            arm=arm,
            refresh_install_age=1,
            diagonal_backend="triton" if arm.endswith("_fused") else "torch",
        )
        write_record(candidate / f"experiment/configs/{arm}_0p5m_seed0.json", config)
    (candidate / "experiment/train.py").write_text("# Bound training fixture\n")
    pinned = candidate / "artifacts/pinned-sources-materialized/Toner-fixture"
    for name in ("reproduce/marin/model.py", "src/toner/fast/engine.py"):
        path = pinned / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Bound {name} fixture\n")
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    write_record(
        candidate / "experiment/upstream_manifest.json",
        {
            "revision": "fixture",
            "files": {str(path.relative_to(pinned)): sha256(path) for path in pinned.rglob("*.py")},
        },
    )
    original_helpers = tmp_path / "original-helpers"
    release_helpers = tmp_path / "release-helpers"
    original_helpers.mkdir()
    release_helpers.mkdir()
    (original_helpers / "controller.py").write_text("# Frozen original fixture\n")
    (release_helpers / "controller.py").write_text("# Frozen release fixture\n")
    original_archive = tmp_path / "original.tar"
    release_archive = tmp_path / "release.tar"
    original_archive.write_bytes(b"frozen original archive fixture")
    release_archive.write_bytes(b"frozen release archive fixture")
    preparation = tmp_path / "run-preparation"
    write_record(tmp_path / "data/experiment_receipt.json", {"status": "verified"})
    numerical = {
        "candidate": str(candidate),
        "files_sha256": candidate_inventory(candidate),
        "source_sha256": sources(candidate),
        "helpers": str(original_helpers),
        "helpers_sha256": {"controller.py": sha256(original_helpers / "controller.py")},
        "archive": str(original_archive),
        "archive_sha256": sha256(original_archive),
        "data_receipt_sha256": sha256(tmp_path / "data/experiment_receipt.json"),
    }
    original_path = preparation / "wave7-candidate.json"
    gpu_path = preparation / "validation-wave7-gpu.json"
    state_path = preparation / "wave7-validation-state.json"
    write_record(original_path, numerical)
    gpu_log = preparation / "original-validation.log"
    gpu_log.write_text("original successful validation fixture\n")
    write_record(
        gpu_path,
        {
            "checks": dict.fromkeys(GATES, True),
            "source_sha256": numerical["source_sha256"],
            "data_receipt_sha256": numerical["data_receipt_sha256"],
            "environment": ENVIRONMENT,
            "config_sha256": {arm: sha256(candidate / f"experiment/configs/{arm}_0p5m_seed0.json") for arm in ARMS},
            "evidence": {
                "gpu_commands": [{"exit_code": 0, "log": str(gpu_log), "log_sha256": sha256(gpu_log)} for _ in range(13)]
            },
        },
    )
    write_record(state_path, {"status": "passed", "keeper_restore": 0, "validation_sha256": sha256(gpu_path)})
    release = {
        **numerical,
        "status": "frozen_validated_logging_release",
        "helpers": str(release_helpers),
        "helpers_sha256": {"controller.py": sha256(release_helpers / "controller.py")},
        "archive": str(release_archive),
        "archive_sha256": sha256(release_archive),
        "arms": ARMS,
        "required_gates": GATES,
        "refresh_install_age": 1,
        "original_validation_binding_sha256": sha256(original_path),
        "original_gpu_validation_sha256": sha256(gpu_path),
        "original_validation_state_sha256": sha256(state_path),
    }
    release_path = preparation / "wave7-release-binding.json"
    write_record(release_path, release)
    return tmp_path, candidate, release_path


def test_fixed_age_preparation_preserves_source_and_exposes_no_winners(fixed_source):
    root, candidate, release = fixed_source
    inventory = candidate_inventory(candidate)
    output = root / "campaigns/fixed-age"
    campaign = prepare(root, release, output)
    analysis = analyze(output / "campaign.json")
    assert candidate_inventory(candidate) == inventory
    assert campaign["status"] == "prepared_unvalidated_not_queued"
    assert campaign["families"] == list(ARMS)
    assert campaign["initial_tokens_per_family"] == 503316480
    assert campaign["refresh_install_age"] == 1
    assert not (output / "runs").exists()
    assert analysis["status"] == "incomplete"
    assert analysis["completed_family_lr_choices"] == {}
    assert analysis["observed_tokens_by_family"] == dict.fromkeys(ARMS, 0)
    for trial in campaign["trials"]:
        config = read(Path(trial["config"]))
        reference = read(Path(trial["reference_config"]))
        assert config["training"]["steps"] == 320
        assert config["training"]["tokens"] == 167772160
        assert config["schedule"]["stable_steps"] + config["schedule"]["decay_steps"] == 320
        assert config["experiment"]["refresh_install_age"] == 1
        assert config["experiment"]["diagonal_backend"] == reference["experiment"]["diagonal_backend"]
        assert config["training"]["heldout_windows"] == reference["training"]["heldout_windows"]
        assert config["root"] == reference["root"]
        assert config["statistics"] == reference["statistics"]
    with pytest.raises(FileExistsError):
        prepare(root, release, output)
    assert read(output / "campaign.json") == campaign


@pytest.mark.parametrize("fault", ("gpu", "restoration", "helper", "archive", "extra_source"))
def test_changed_numerical_evidence_refuses_preparation_without_output(fixed_source, fault):
    root, candidate, release = fixed_source
    output = root / "campaigns/fixed-age"
    binding = read(release)
    if fault == "gpu":
        path = root / "run-preparation/validation-wave7-gpu.json"
        record = read(path)
        record["checks"]["real_data_cached_root_continuation"] = False
        write_record(path, record)
    elif fault == "restoration":
        path = root / "run-preparation/wave7-validation-state.json"
        record = read(path)
        record["keeper_restore"] = 1
        write_record(path, record)
    elif fault == "helper":
        (Path(binding["helpers"]) / "controller.py").write_text("# Changed controller\n")
    elif fault == "archive":
        Path(binding["archive"]).write_bytes(b"changed archive")
    else:
        (candidate / "experiment/extra.py").write_text("# Unbound executed source\n")
    with pytest.raises(ValueError):
        prepare(root, release, output)
    assert not output.exists()


def complete_trial(campaign: dict, trial: dict, directory: Path, candidate: Path, selection: float, c4: float) -> None:
    config = read(Path(trial["config"]))
    source = {
        str(path.relative_to(candidate)): sha256(path)
        for path in [
            candidate / "experiment/train.py",
            candidate / "artifacts/external-sources/Toner-fixture/reproduce/marin/model.py",
            candidate / "artifacts/external-sources/Toner-fixture/src/toner/fast/engine.py",
        ]
    }
    write_record(
        directory / "run.json",
        {
            "status": "complete",
            "completed_steps": 320,
            "config": config,
            "source_sha256": source,
            "data_receipt_sha256": campaign["data_receipt_sha256"],
        },
    )
    rows = [{"step": step, "tokens": step * 524288, "train_loss": 6.0} for step in range(1, 321)]
    weights = config["data"]["mixture_weights"]
    heldout = {f"heldout/{name}": selection for name in weights}
    final_selection = sum(weight * heldout[f"heldout/{name}"] for name, weight in weights.items())
    final_selection /= sum(weights.values())
    rows[-1].update(selection_loss=final_selection, c4_loss=c4, heldout_losses=heldout)
    (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = directory / "checkpoints/step_000320"
    write_record(checkpoint / "complete.json", {"step": 320, "world": 8})
    for rank in range(8):
        (checkpoint / f"rank_{rank}.pt").write_bytes(b"offline checkpoint fixture")


def test_completed_fixed_age_sweeps_rank_mixture_and_charge_failed_tokens(fixed_source):
    root, candidate, release = fixed_source
    output = root / "campaigns/fixed-age"
    campaign = prepare(root, release, output)
    for family in ARMS:
        trials = [trial for trial in campaign["trials"] if trial["family"] == family]
        for index, trial in enumerate(trials):
            complete_trial(campaign, trial, output / "runs" / trial["name"], candidate, 4.0 - index, 2.0 + index)
    result = analyze(output / "campaign.json")
    assert result["status"] == "complete"
    assert result["observed_tokens_by_family"] == dict.fromkeys(ARMS, 503316480)
    assert {family: item["trial"] for family, item in result["completed_family_lr_choices"].items()} == {
        family: f"{family}_lr_up" for family in ARMS
    }
    assert all(len(trial["training_windows"]) == 5 for trial in result["trials"])
    failed = output / "runs/outer_leftdiag_fused_lr_up/run.json"
    record = read(failed)
    record["status"] = "failed"
    write_record(failed, record)
    incomplete = analyze(output / "campaign.json")
    assert incomplete["status"] == "incomplete"
    assert "outer_leftdiag_fused" not in incomplete["completed_family_lr_choices"]
    assert incomplete["observed_tokens_by_family"]["outer_leftdiag_fused"] == 503316480


def test_changed_trial_lr_with_self_consistent_hash_cannot_be_ranked(fixed_source):
    root, _, release = fixed_source
    output = root / "campaigns/fixed-age"
    campaign = prepare(root, release, output)
    trial = campaign["trials"][0]
    config_path = Path(trial["config"])
    config = read(config_path)
    config["optimizer"]["lr"] *= 2
    write_record(config_path, config)
    trial.update(config_sha256=sha256(config_path), matrix_lr=config["optimizer"]["lr"])
    write_record(output / "campaign.json", campaign)
    with pytest.raises(ValueError, match="permitted LR transformation"):
        analyze(output / "campaign.json")


def test_actual_validation_plan_keeps_all_lr_configs_and_original_continuation_scope(fixed_source):
    root, candidate, release = fixed_source
    directory = root / "campaigns/fixed-age"
    campaign = prepare(root, release, directory)
    result = plan(directory / "campaign.json", directory / "validation")
    assert result["status"] == "planned_not_executed"
    assert result["gpu_commands"] == []
    assert result["planned_validation_tokens"] == 301989888
    assert [item["name"] for item in result["trials"]] == [item["name"] for item in campaign["trials"]]
    assert not (directory / "validation").exists()
    for item, trial in zip(result["trials"], campaign["trials"], strict=True):
        assert item["config_sha256"] == sha256(Path(trial["config"]))
        assert not any(item["checks"].values())
        parsed = []
        for command in item["commands"]:
            options = {
                name: command[command.index(name) + 1]
                for name in ("--config", "--out", "--max-steps", "--execution-boundary")
            }
            parsed.append(options)
            assert options["--config"] == trial["config"]
            assert int(options["--execution-boundary"]) == 22
            assert "--nproc_per_node=8" in command
            if trial["family"].endswith("_fused"):
                assert "--verify-diagonal-capture" in command
                assert "--verify-outer-diagonal-updates" in command
        assert [int(options["--max-steps"]) for options in parsed] == [24, 22, 24]
        assert [options["--out"] for options in parsed] == [item["continuous"], item["resumed"], item["resumed"]]
        resume = item["commands"][2]
        assert resume[resume.index("--resume") + 1] == str(Path(item["resumed"]) / "checkpoints/step_000022")
        config = read(Path(item["config"]))
        assert config["training"]["steps"] == 320
        assert config["schedule"]["stable_steps"] + config["schedule"]["decay_steps"] == 320
    assert candidate_inventory(candidate) == read(release)["files_sha256"]


def validation_trajectory(directory: Path, config: dict, candidate: Path, numerical: dict) -> dict:
    source = {
        str(path.relative_to(candidate)): sha256(path)
        for path in [
            candidate / "experiment/train.py",
            candidate / "artifacts/external-sources/Toner-fixture/reproduce/marin/model.py",
            candidate / "artifacts/external-sources/Toner-fixture/src/toner/fast/engine.py",
        ]
    }
    write_record(
        directory / "run.json",
        {
            "config": config,
            "source_sha256": source,
            "data_receipt_sha256": numerical["data_receipt_sha256"],
            "status": "execution_check_complete",
            "completed_steps": 24,
        },
    )
    rows = [{"step": step, "tokens": step * 524288, "train_loss": 4.0} for step in range(1, 25)]
    (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for step in (22, 24):
        checkpoint = directory / f"checkpoints/step_{step:06d}"
        write_record(checkpoint / "complete.json", {"step": step, "world": 8})
        for rank in range(8):
            (checkpoint / f"rank_{rank}.pt").write_bytes(b"offline rank fixture")
    return executed_trajectory(directory, config, candidate, numerical)


@pytest.fixture
def actual_short_receipt(fixed_source):
    root, candidate, release = fixed_source
    directory = root / "campaigns/fixed-age"
    prepare(root, release, directory)
    planned = plan(directory / "campaign.json", directory / "validation")
    actual = copy.deepcopy(planned)
    actual["status"] = "passed_actual_short_config_gates"
    numerical = read(release)
    for item in actual["trials"]:
        item["checks"] = dict.fromkeys(item["checks"], True)
        item["cached_root_checkpoint"] = {"ranks": 8, "owned_graph_groups": 2}
        for key in ("boundary_continuation", "post_cached_root_continuation"):
            item[key] = {"tensor_comparisons": 16, "max_absolute_error": 0.0}
        item["trajectory_identity"] = {}
        item["fixed_age_history"] = {}
        if item["family"].endswith("_fused"):
            item["capture_history"] = {}
        for trajectory in ("continuous", "resumed"):
            item["trajectory_identity"][trajectory] = validation_trajectory(
                Path(item[trajectory]), read(Path(item["config"])), candidate, numerical
            )
            item["fixed_age_history"][trajectory] = {
                "rank": 0,
                "updates": 24,
                "installation_age": 1,
                "observed_installations": 5,
            }
            if item["family"].endswith("_fused"):
                item["capture_history"][trajectory] = {"updates": 24, "captured_rows_and_outer_updates": True}
        for index, command in enumerate(item["commands"]):
            log = directory / "validation" / f"{item['name']}_{('continuous', 'interrupted', 'resumed')[index]}.log"
            log.write_text("offline successful execution log\n")
            actual["gpu_commands"].append(
                {
                    "trial": item["name"],
                    "command": command,
                    "exit_code": 0,
                    "log": str(log),
                    "log_sha256": sha256(log),
                }
            )
    receipt = directory / "validation-gpu.json"
    write_record(receipt, actual)
    return {
        "campaign": str(directory / "campaign.json"),
        "receipt": str(receipt),
        "work": str(directory / "validation"),
        "plan": planned,
    }, actual


def test_actual_short_receipt_requires_full_execution_then_refuses_changed_history(actual_short_receipt):
    binding, actual = actual_short_receipt
    assert require_actual_receipt(binding)["status"] == "passed_actual_short_config_gates"
    path = Path(actual["trials"][-1]["resumed"]) / "metrics_from_000000.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["train_loss"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="trajectory or scheduling evidence changed"):
        require_actual_receipt(binding)


@pytest.mark.parametrize(
    "fault",
    ("missing_trial", "shortened_command", "failed_command", "log", "empty_tensors", "missing_capture", "missing_rank"),
)
def test_actual_short_receipt_refuses_partial_or_changed_execution(actual_short_receipt, fault):
    binding, actual = actual_short_receipt
    if fault == "missing_trial":
        actual["trials"].pop()
    elif fault == "shortened_command":
        command = actual["gpu_commands"][0]["command"]
        command[command.index("--max-steps") + 1] = "3"
    elif fault == "failed_command":
        actual["gpu_commands"][0]["exit_code"] = 1
    elif fault == "log":
        Path(actual["gpu_commands"][0]["log"]).write_text("changed after successful validation")
    elif fault == "empty_tensors":
        actual["trials"][0]["post_cached_root_continuation"]["tensor_comparisons"] = 0
    elif fault == "missing_capture":
        actual["trials"][-1]["capture_history"]["resumed"]["captured_rows_and_outer_updates"] = False
    else:
        (Path(actual["trials"][0]["resumed"]) / "checkpoints/step_000024/rank_7.pt").unlink()
    write_record(Path(binding["receipt"]), actual)
    with pytest.raises((ValueError, FileNotFoundError)):
        require_actual_receipt(binding)


@pytest.fixture
def completed_wave7(fixed_source):
    root, candidate, release = fixed_source
    numerical = read(release)
    cloud = {
        "status": "verified",
        "visibility": "PRIVATE",
        "project": "fixture/output-side-normalization-marin",
        "runs": {},
    }
    for arm in ARMS:
        trial = {"config": str(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")}
        directory = root / f"runs/wave7/{arm}"
        complete_trial({"data_receipt_sha256": numerical["data_receipt_sha256"]}, trial, directory, candidate, 3.0, 3.4)
        manifest = read(directory / "run.json")
        manifest["completed_steps"] = 5120
        write_record(directory / "run.json", manifest)
        rows = [
            {"step": step, "tokens": step * 524288, "train_loss": 4.0, "c4_loss": 3.4, "selection_loss": 3.0}
            for step in range(1, 5121)
        ]
        metrics = directory / "metrics_from_000000.jsonl"
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
        checkpoint = directory / "checkpoints/step_005120"
        write_record(checkpoint / "complete.json", {"step": 5120, "world": 8})
        for rank in range(8):
            (checkpoint / f"rank_{rank}.pt").write_bytes(b"offline predecessor rank")
        cloud["runs"][arm] = {
            "status": "verified",
            "manifest_sha256": sha256(directory / "run.json"),
            "metrics_sha256": sha256(metrics),
            "checkpoint_receipt_sha256": sha256(checkpoint / "complete.json"),
            "checkpoint_rank_bytes": [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)],
            "verification": {
                "state": "FINISHED",
                "configuration_verified": True,
                "token_axis_verified": True,
                "integer_values_exact": True,
                "point_counts": {"train/loss": 5120, "tokens": 5120},
            },
        }
    from_paths = {f"wave7/{arm}": root / f"runs/wave7/{arm}" for arm in ARMS}
    runs = {name: history(path) for name, path in from_paths.items()}
    analysis = {
        "wave": "wave7",
        "window_tokens": WINDOW,
        "analyses": [
            compare({name: runs[name] for name in (baseline, *names)}, baseline, WINDOW)
            for baseline, names in COMPARISONS
        ],
        "unpaired_runs": {},
    }
    write_record(root / "run-preparation/analysis-wave7.json", analysis)
    write_record(root / "run-preparation/swanlab-upload-wave7.json", cloud)
    return root


def test_completed_predecessor_needs_five_windows_cloud_history_and_all_rank_files(completed_wave7):
    root = completed_wave7
    assert require_completed_wave(root)["analysis_sha256"] == sha256(root / "run-preparation/analysis-wave7.json")
    metrics = root / "runs/wave7/outer_leftdiag/metrics_from_000000.jsonl"
    metrics.write_text(metrics.read_text() + "\n")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        require_completed_wave(root)


def test_validation_freeze_preserves_campaign_and_requires_bound_controller(fixed_source, monkeypatch):
    root, candidate, release = fixed_source
    directory = root / "campaigns/fixed-age"
    prepare(root, release, directory)
    campaign_path = directory / "campaign.json"
    identity = sha256(campaign_path), candidate_inventory(candidate)
    frozen = run_fixed_age_validation.freeze(campaign_path)
    assert frozen["status"] == "frozen_pending_completed_wave7_and_actual_gpu_validation"
    assert not (directory / "validation-state.json").exists()
    assert not (directory / "validation-gpu.json").exists()
    assert not (directory / "validation").exists()
    assert (sha256(campaign_path), candidate_inventory(candidate)) == identity
    monkeypatch.setattr(
        run_fixed_age_validation, "__file__", str(Path(frozen["helpers"]) / "run_fixed_age_validation.py")
    )
    assert run_fixed_age_validation.binding(campaign_path)["campaign_sha256"] == identity[0]
    with pytest.raises(FileExistsError):
        run_fixed_age_validation.freeze(campaign_path)
    (Path(frozen["helpers"]) / "validate_fixed_age_tuning.py").write_text("# changed validator\n")
    with pytest.raises(ValueError, match="validation source, helper, archive or plan identity changed"):
        run_fixed_age_validation.binding(campaign_path)


@pytest.mark.parametrize("fault", ("archive", "campaign", "original_log", "source"))
def test_validation_freeze_refuses_changed_bound_evidence(fixed_source, monkeypatch, fault):
    root, candidate, release = fixed_source
    directory = root / "campaigns/fixed-age"
    prepare(root, release, directory)
    campaign_path = directory / "campaign.json"
    frozen = run_fixed_age_validation.freeze(campaign_path)
    monkeypatch.setattr(
        run_fixed_age_validation, "__file__", str(Path(frozen["helpers"]) / "run_fixed_age_validation.py")
    )
    if fault == "archive":
        Path(frozen["archive"]).write_bytes(b"changed archive")
    elif fault == "campaign":
        campaign_path.write_text(campaign_path.read_text() + "\n")
    elif fault == "original_log":
        (root / "run-preparation/original-validation.log").write_text("changed original GPU execution log")
    else:
        (candidate / "experiment/train.py").write_text("# changed training source\n")
    with pytest.raises(ValueError):
        run_fixed_age_validation.binding(campaign_path)


def test_gpu_worker_requires_guarded_controller_generation_and_unique_child_marker(fixed_source, monkeypatch):
    root, _, release = fixed_source
    directory = root / "campaigns/fixed-age"
    prepare(root, release, directory)
    campaign_path = directory / "campaign.json"
    frozen = run_fixed_age_validation.freeze(campaign_path)
    helpers = Path(frozen["helpers"])
    monkeypatch.setattr(validate_fixed_age_tuning, "__file__", str(helpers / "validate_fixed_age_tuning.py"))
    state = {
        "status": "validating",
        "binding_sha256": sha256(directory / "validation-binding.json"),
        "controller_pid": 90001,
        "controller_start_ticks": "3456",
        "child_pid": os.getpid(),
        "child_marker": "fixture-unique-child-marker",
    }
    write_record(directory / "validation-state.json", state)
    monkeypatch.setenv("OUTPUT_MARIN_DIAGNOSTIC_ID", state["child_marker"])
    arguments = [
        "/opt/conda/envs/torch-base/bin/python",
        str(helpers / "run_fixed_age_validation.py"),
        "--campaign",
        str(campaign_path),
        "--phase",
        "validate",
    ]
    monkeypatch.setattr(validate_fixed_age_tuning, "observed_process", lambda pid: ("S", "3456", arguments))
    assert validate_fixed_age_tuning.require_owned_execution(campaign_path, frozen["plan"]) == {
        "controller_pid": 90001,
        "controller_start_ticks": "3456",
    }
    monkeypatch.setenv("OUTPUT_MARIN_DIAGNOSTIC_ID", "different-child")
    with pytest.raises(RuntimeError, match="guarded controller generation"):
        validate_fixed_age_tuning.require_owned_execution(campaign_path, frozen["plan"])
    monkeypatch.setenv("OUTPUT_MARIN_DIAGNOSTIC_ID", state["child_marker"])
    monkeypatch.setattr(validate_fixed_age_tuning, "observed_process", lambda pid: ("S", "9876", arguments))
    with pytest.raises(RuntimeError, match="guarded controller generation"):
        validate_fixed_age_tuning.require_owned_execution(campaign_path, frozen["plan"])


def test_live_capture_rejects_unfinished_jsonl_without_creating_observation(fixed_source):
    root, candidate, release = fixed_source
    numerical = read(release)
    for arm in ARMS:
        directory = root / f"runs/wave7/{arm}"
        validation_trajectory(
            directory, read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json"), candidate, numerical
        )
        manifest = read(directory / "run.json")
        manifest["status"] = "running"
        write_record(directory / "run.json", manifest)
    broken = root / "runs/wave7/outer_leftdiag/metrics_from_000000.jsonl"
    broken.write_bytes(broken.read_bytes() + b'{"step":25')
    output = root / "observation"
    with pytest.raises(ValueError, match="unfinished JSONL row"):
        capture(root, output)
    assert not output.exists()
    broken.write_bytes(broken.read_bytes().removesuffix(b'{"step":25'))
    record = capture(root, output)
    assert {arm: item["observed_updates"] for arm, item in record["runs"].items()} == dict.fromkeys(ARMS, 24)
    assert sha256(output / "outer_leftdiag/metrics_from_000000.jsonl") == sha256(broken)


@pytest.fixture
def tuning_execution(actual_short_receipt, completed_wave7, monkeypatch):
    actual_binding, _ = actual_short_receipt
    campaign_path = Path(actual_binding["campaign"])
    frozen_validation = run_fixed_age_validation.freeze(campaign_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "bound check passed\n", ""),
    )
    frozen = run_fixed_age_tuning.freeze(campaign_path)
    monkeypatch.setattr(run_fixed_age_tuning, "__file__", str(Path(frozen["helpers"]) / "run_fixed_age_tuning.py"))
    write_record(
        campaign_path.parent / "validation-state.json",
        {
            "status": "passed",
            "keeper_restore": 0,
            "cleanup_exit_code": 0,
            "binding_sha256": sha256(campaign_path.parent / "validation-binding.json"),
            "validation_sha256": sha256(Path(frozen_validation["receipt"])),
        },
    )
    return campaign_path, frozen


def test_execution_freeze_keeps_three_full_short_trials_and_requires_actual_validation(tuning_execution):
    campaign_path, frozen = tuning_execution
    assert run_fixed_age_tuning.binding(campaign_path)["campaign_sha256"] == sha256(campaign_path)
    assert sum(trial["tokens"] for family in frozen["plan"] for trial in family["trials"]) == 2013265920
    assert [family["node"] for family in frozen["plan"]] == list(run_fixed_age_tuning.NODES)
    for family in frozen["plan"]:
        assert [trial["name"] for trial in family["trials"]] == [
            f"{family['family']}_{label}" for label in ("lr_down", "lr_base", "lr_up")
        ]
        assert all("--max-steps" not in trial["command"] for trial in family["trials"])
    state_path = campaign_path.parent / "validation-state.json"
    state = read(state_path)
    state["keeper_restore"] = "unverified"
    write_record(state_path, state)
    with pytest.raises(ValueError, match="keeper restoration have not passed"):
        run_fixed_age_tuning.require_ready(campaign_path)
    assert not (campaign_path.parent / "launch-state.json").exists()
    assert not (campaign_path.parent / "workers").exists()
    (Path(frozen["helpers"]) / "monitor_fixed_age_tuning.sh").write_text("changed monitor\n")
    with pytest.raises(ValueError, match="execution helper"):
        run_fixed_age_tuning.binding(campaign_path)


def test_execution_completion_ignores_live_sibling_and_refuses_missing_rank(tuning_execution):
    campaign_path, _ = tuning_execution
    campaign = read(campaign_path)
    first, second = campaign["trials"][:2]
    complete_trial(campaign, first, campaign_path.parent / "runs" / first["name"], Path(campaign["source"]), 3.0, 3.4)
    sibling = campaign_path.parent / "runs" / second["name"]
    sibling.mkdir()
    (sibling / "run.json").write_text('{"status":"running"}\n')
    (sibling / "metrics_from_000000.jsonl").write_text('{"unfinished')
    assert run_fixed_age_tuning.binding(campaign_path)["campaign_sha256"] == sha256(campaign_path)
    assert run_fixed_age_tuning.completed_trial(campaign_path, first["name"])["observed_tokens"] == 167772160
    (campaign_path.parent / "runs" / first["name"] / "checkpoints/step_000320/rank_7.pt").unlink()
    with pytest.raises(FileNotFoundError):
        run_fixed_age_tuning.completed_trial(campaign_path, first["name"])


def test_execution_snapshot_retains_failed_tokens_and_unknown_remote_state(tuning_execution, monkeypatch):
    campaign_path, frozen = tuning_execution
    campaign = read(campaign_path)
    trial = campaign["trials"][0]
    directory = campaign_path.parent / "runs" / trial["name"]
    complete_trial(campaign, trial, directory, Path(campaign["source"]), 3.0, 3.4)
    manifest = read(directory / "run.json")
    manifest["status"] = "failed"
    write_record(directory / "run.json", manifest)
    write_record(campaign_path.parent / "launch-state.json", {"status": "needs_attention"})

    def probe(command, **kwargs):
        node = command[-2].removeprefix("root@")
        if node == frozen["plan"][0]["node"]:
            raise subprocess.TimeoutExpired(command, 300)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"status": "not_launched", "family": ARMS[run_fixed_age_tuning.NODES.index(node)]}),
            "",
        )

    monkeypatch.setattr(subprocess, "run", probe)
    record = run_fixed_age_tuning.snapshot(campaign_path)
    assert record["status"] == "active"
    assert record["all_workers_terminal"] is False
    assert record["workers"][0]["status"] == "unknown"
    assert record["trials"][0]["observed_tokens"] == 167772160
    metrics = directory / "metrics_from_000000.jsonl"
    metrics.write_bytes(metrics.read_bytes() + b'{"step":321')
    unreadable = run_fixed_age_tuning.snapshot(campaign_path)
    assert unreadable["trials"][0]["observed_tokens"] is None
    assert "accounting_error" in unreadable["trials"][0]


def test_execution_partial_launch_keeps_acknowledged_worker_and_monitor_on_remote_timeout(tuning_execution, monkeypatch):
    campaign_path, frozen = tuning_execution
    original_text, original_bytes = Path.read_text, Path.read_bytes
    monitor_pid = 987100
    monitor_command = ["bash", str(Path(frozen["helpers"]) / "monitor_fixed_age_tuning.sh"), str(campaign_path)]

    def path_text(path, *args, **kwargs):
        if path == Path(f"/proc/{monitor_pid}/stat"):
            return f"{monitor_pid} (bash) " + " ".join(["S", "1", *["0"] * 17, "12345"])
        return original_text(path, *args, **kwargs)

    def path_bytes(path, *args, **kwargs):
        if path == Path(f"/proc/{monitor_pid}/cmdline"):
            return b"\0".join(argument.encode() for argument in monitor_command) + b"\0"
        return original_bytes(path, *args, **kwargs)

    class MonitorChild:
        pid = monitor_pid

        def poll(self):
            return None

    def spawn(command, **kwargs):
        assert command == monitor_command
        (campaign_path.parent / "monitor/pid").write_text(str(monitor_pid))
        return MonitorChild()

    def run(command, **kwargs):
        family = ARMS[run_fixed_age_tuning.NODES.index(command[-2].removeprefix("root@"))]
        if family != ARMS[0]:
            raise subprocess.TimeoutExpired(command, 300)
        launch = read(campaign_path.parent / "launch-state.json")
        acknowledgment = {
            "family": family,
            "node": run_fixed_age_tuning.NODES[0],
            "status": "running",
            "binding_sha256": launch["binding_sha256"],
            "worker_marker": launch["worker_markers"][family],
            "pid": 987101,
            "start_ticks": "23456",
            "command": run_fixed_age_tuning.worker_command(campaign_path, frozen, family),
        }
        write_record(campaign_path.parent / "workers" / f"{family}.json", acknowledgment)
        return subprocess.CompletedProcess(command, 0, json.dumps(acknowledgment), "")

    monkeypatch.setattr(Path, "read_text", path_text)
    monkeypatch.setattr(Path, "read_bytes", path_bytes)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        run_fixed_age_tuning.launch(campaign_path)
    state = read(campaign_path.parent / "launch-state.json")
    assert state["status"] == "needs_attention"
    assert state["monitor_pid"] == monitor_pid
    assert state["workers"][ARMS[0]]["pid"] == 987101
    assert read(campaign_path.parent / "workers" / f"{ARMS[0]}.json")["status"] == "running"
    assert "keeper_restore" not in state
    with pytest.raises(FileExistsError):
        run_fixed_age_tuning.launch(campaign_path)


@pytest.fixture
def family_process_io(tuning_execution, monkeypatch):
    campaign_path, frozen = tuning_execution
    campaign = read(campaign_path)
    family = ARMS[0]
    assignment = frozen["plan"][0]
    workers = campaign_path.parent / "workers"
    workers.mkdir()
    marker = "unique-worker-launch-fixture"
    write_record(
        campaign_path.parent / "launch-state.json",
        {
            "status": "launching",
            "binding_sha256": sha256(campaign_path.parent / "execution-binding.json"),
            "worker_markers": {family: marker},
        },
    )
    monkeypatch.setenv(run_fixed_age_tuning.WORKER_MARKER, marker)
    write_record(
        workers / f"{family}.start.json",
        {
            "pid": os.getpid(),
            "start_ticks": "12345",
            "family": family,
            "node": assignment["node"],
            "command": run_fixed_age_tuning.worker_command(campaign_path, frozen, family),
            "worker_marker": marker,
            "binding_sha256": sha256(campaign_path.parent / "execution-binding.json"),
        },
    )
    keeper = 987001
    processes = {
        os.getpid(): {
            "parent": 1,
            "command": run_fixed_age_tuning.worker_command(campaign_path, frozen, family),
        },
        keeper: {
            "parent": 1,
            "command": ["python", WORKLOAD.decode()],
        },
        keeper
        + 1: {
            "parent": keeper,
            "command": ["python", WORKLOAD.decode()],
        },
    }
    boundary = {"trial_commands": [], "restored": [], "fail_trial": None, "visible_devices": 8}
    original_text, original_bytes, original_iterdir = Path.read_text, Path.read_bytes, Path.iterdir
    original_unlink = Path.unlink

    def path_text(path, *args, **kwargs):
        if path == Path("/root/dl/runtime/gpu_matmul.pid"):
            return str(keeper)
        if path.parent.parent == Path("/proc") and path.name == "stat":
            pid = int(path.parent.name)
            if pid not in processes:
                raise FileNotFoundError(path)
            return f"{pid} (python) " + " ".join(
                ["S", str(processes[pid]["parent"]), str(pid), str(pid), *["0"] * 15, "12345"]
            )
        return original_text(path, *args, **kwargs)

    def path_bytes(path, *args, **kwargs):
        if path.parent.parent == Path("/proc"):
            pid = int(path.parent.name)
            if pid not in processes:
                raise FileNotFoundError(path)
            if path.name == "cmdline":
                return b"\0".join(argument.encode() for argument in processes[pid]["command"]) + b"\0"
            return b""
        return original_bytes(path, *args, **kwargs)

    def output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return assignment["node"]
        if command == ["ps", "-eo", "args="]:
            return "\n".join(" ".join(value["command"]) for value in processes.values())
        if command[0] == "nvidia-smi":
            if "compute-apps" in command[1]:
                return str(keeper + 1) if keeper + 1 in processes else ""
            return "A100-SXM4-40GB\n" * 8
        raise AssertionError(command)

    def run(command, **kwargs):
        if Path(command[1]).name == "snapshot_wave7_release.py":
            write_record(Path(command[command.index("--out") + 1]), {"status": "complete"})
        elif command[0] == "bash":
            boundary["restored"].append(command)
        elif command[1] == "-c":
            if boundary["visible_devices"] != 8:
                raise subprocess.CalledProcessError(1, command)
        elif command[1:3] != ["-m", "experiment.bootstrap"]:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    class TrialChild:
        pid = 987010
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            return self.returncode

    def spawn(command, **kwargs):
        child = TrialChild()
        boundary["trial_commands"].append(command)
        name = Path(command[command.index("--out") + 1]).name
        trial = next(item for item in campaign["trials"] if item["name"] == name)
        log = workers / f"{name}.torchrun.log"
        log.write_text("fake successful training command\n")
        complete_trial(campaign, trial, Path(command[command.index("--out") + 1]), Path(campaign["source"]), 3.0, 3.4)
        if name == boundary["fail_trial"]:
            manifest = read(Path(command[command.index("--out") + 1]) / "run.json")
            manifest["status"] = "failed"
            write_record(Path(command[command.index("--out") + 1]) / "run.json", manifest)
            child.returncode = 1
        else:
            child.returncode = 0
        return child

    monkeypatch.setattr(Path, "read_text", path_text)
    monkeypatch.setattr(Path, "read_bytes", path_bytes)
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda path: (
            iter(Path("/proc") / str(pid) for pid in processes) if path == Path("/proc") else original_iterdir(path)
        ),
    )
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda path, **kwargs: (
            None if path == Path("/root/dl/runtime/gpu_matmul.pid") else original_unlink(path, **kwargs)
        ),
    )
    monkeypatch.setattr(os, "killpg", lambda group, number: processes.pop(group))
    monkeypatch.setattr(subprocess, "check_output", output)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    return campaign_path, family, assignment, boundary


@pytest.mark.parametrize("fail", (False, True))
def test_family_worker_runs_complete_schedules_or_preserves_failure_and_restores(family_process_io, fail):
    campaign_path, family, assignment, boundary = family_process_io
    if fail:
        boundary["fail_trial"] = assignment["trials"][1]["name"]
        with pytest.raises(RuntimeError, match="Short-LR trial failed"):
            run_fixed_age_tuning.run_family(campaign_path, family)
    else:
        assert run_fixed_age_tuning.run_family(campaign_path, family)["status"] == "complete"
    state = read(campaign_path.parent / "workers" / f"{family}.json")
    assert state["keeper_restore"] == 0
    assert len(boundary["restored"]) == 1
    assert len(boundary["trial_commands"]) == (2 if fail else 3)
    assert boundary["trial_commands"] == [
        trial["command"] for trial in assignment["trials"][: len(boundary["trial_commands"])]
    ]
    assert state["status"] == ("needs_attention" if fail else "complete")
    assert all(trial["observed_tokens"] == 167772160 for trial in state["trials"] if trial["status"] == "complete")
    with pytest.raises(FileExistsError):
        run_fixed_age_tuning.run_family(campaign_path, family)


def test_family_worker_refuses_wrong_spawn_generation_before_keeper_release(family_process_io):
    campaign_path, family, _, boundary = family_process_io
    path = campaign_path.parent / "workers" / f"{family}.start.json"
    state = read(path)
    state["start_ticks"] = "different-generation"
    write_record(path, state)
    with pytest.raises(RuntimeError, match="persisted bound spawn"):
        run_fixed_age_tuning.run_family(campaign_path, family)
    assert boundary["restored"] == []
    assert boundary["trial_commands"] == []
    assert not (campaign_path.parent / "workers" / f"{family}.json").exists()


def test_family_worker_cuda_visibility_failure_preserves_keeper_before_training(family_process_io):
    campaign_path, family, _, boundary = family_process_io
    boundary["visible_devices"] = 0
    with pytest.raises(subprocess.CalledProcessError):
        run_fixed_age_tuning.run_family(campaign_path, family)
    assert boundary["restored"] == []
    assert boundary["trial_commands"] == []
    assert not (campaign_path.parent / "workers" / f"{family}.json").exists()


def test_dead_worker_with_marked_live_child_is_not_terminal_or_restored(family_process_io, monkeypatch):
    campaign_path, family, assignment, boundary = family_process_io
    directory = campaign_path.parent / "workers"
    state = read(directory / f"{family}.start.json")
    state.update(
        status="needs_attention",
        pid=987200,
        trials=[{"name": assignment["trials"][0]["name"], "status": "running"}],
    )
    write_record(directory / f"{family}.json", state)
    write_record(directory / f"{assignment['trials'][0]['name']}.execution.json", {"child_marker": "orphan-marker"})
    original_iterdir, original_bytes, original_text = Path.iterdir, Path.read_bytes, Path.read_text
    child = 987201
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda path: iter([Path(f"/proc/{child}")]) if path == Path("/proc") else original_iterdir(path),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path, *args, **kwargs: (
            b"OUTPUT_MARIN_DIAGNOSTIC_ID=orphan-marker\0"
            if path == Path(f"/proc/{child}/environ")
            else (
                b"python\0-m\0experiment.train\0"
                if path == Path(f"/proc/{child}/cmdline")
                else original_bytes(path, *args, **kwargs)
            )
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            f"{child} (python) " + " ".join(["S", "1", *["0"] * 17, "34567"])
            if path == Path(f"/proc/{child}/stat")
            else original_text(path, *args, **kwargs)
        ),
    )
    observed = run_fixed_age_tuning.probe_worker(campaign_path, family)
    assert observed["status"] == "live_orphaned_trial"
    assert child in observed["descendants"]
    assert "keeper_restore" not in read(directory / f"{family}.json")
    assert boundary["restored"] == []


def test_short_lr_logging_loader_retains_unique_trials_and_ignores_live_sibling(tuning_execution):
    campaign_path, frozen = tuning_execution
    campaign = read(campaign_path)
    family = ARMS[0]
    selected = [trial for trial in campaign["trials"] if trial["family"] == family]
    for trial in selected:
        directory = campaign_path.parent / "runs" / trial["name"]
        complete_trial(campaign, trial, directory, Path(campaign["source"]), 3.0, 3.4)
        manifest = read(directory / "run.json")
        manifest.update(
            start_time="2026-11-17T00:00:00+08:00",
            end_time="2026-11-17T00:10:00+08:00",
            elapsed_seconds=600.0,
            source_revision="fixture",
            experiment_git_revision="fixture",
            python="fixture",
            torch="fixture",
            cuda="fixture",
            driver="fixture",
            gpu="A100 fixture",
            world_size=8,
        )
        write_record(directory / "run.json", manifest)
    sibling = campaign_path.parent / "runs" / campaign["trials"][1]["name"]
    sibling.mkdir()
    (sibling / "run.json").write_text('{"status":"running"}')
    (sibling / "metrics_from_000000.jsonl").write_text('{"unfinished')
    logging_python = (
        "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/" "output-side-normalization/.venv-logging/bin/python"
    )
    result = run_process(
        [
            logging_python,
            "-c",
            "import json,sys; from pathlib import Path; from swanlab_fixed_age_tuning import histories; "
            "items=histories(Path(sys.argv[1])); "
            "print(json.dumps([{'name':h.arm,'id':h.run_id,'rows':len(h.metrics),"
            "'config':h.manifest['config'],'final':h.metrics[-1]} for h in items]))",
            str(campaign_path),
        ],
        env={**os.environ, "PYTHONPATH": frozen["helpers"]},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    items = json.loads(result.stdout)
    assert {item["name"] for item in items} == {trial["name"] for trial in selected}
    assert len({item["id"] for item in items}) == 3
    assert all(item["rows"] == 320 and item["final"]["tokens"] == 167772160 for item in items)
    assert [item["config"] for item in items] == [read(Path(trial["config"])) for trial in selected]
