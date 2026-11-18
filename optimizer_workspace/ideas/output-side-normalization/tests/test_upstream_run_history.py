# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Wave-9 provenance, full-budget comparisons and terminal import gates offline."""

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import upstream_run_history as history
from analyze_tuning import executed_sources
from analyze_upstream_run import analyze
from compare_runs import WINDOW
from launch_wave5 import NODES, PYTHON, sha256
from upstream_validation import GATES, SYNTHETIC_GATES, validation_commands
from validate_wave5 import ENVIRONMENT
from wave7_release import candidate_inventory

TRACKING = Path(history.__file__).parent
LOGGING_PYTHON = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/.venv-logging/bin/python"
)
EVALUATION_STEPS = {1, *range(251, 5121, 250), 5120}


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def source_hashes(files: dict) -> dict:
    return {name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")}


def archive(path: Path, candidate: Path, helpers: Path, names: tuple[str, ...]) -> None:
    with tarfile.open(path, "w") as bundle:
        for name in names:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in candidate_inventory(candidate):
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)


@pytest.fixture
def release_fixture(tmp_path, monkeypatch):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    previous = tmp_path / "candidates/upstream-full-model-corrected-validator-20261117"
    modules = {
        "experiment/__init__.py": "",
        "experiment/train.py": "def train():\n    return 'original full-budget trainer'\n",
        "experiment/execution_owner.py": "def require_training_owner():\n    raise RuntimeError('unreleased')\n",
        "experiment/block_toner.py": "def step(gradient, lr):\n    return -lr * gradient\n",
        "experiment/tests/test_upstream_release.py": "def test_unreleased():\n    assert True\n",
        "experiment/tests/test_upstream_validation.py": "def test_continuation_contract():\n    assert True\n",
    }
    for name, content in modules.items():
        path = previous / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    pinned = previous / "artifacts/pinned-sources-materialized/Toner-fixture"
    pinned_files = {
        "reproduce/marin/model.py": "def forward(x):\n    return x\n",
        "reproduce/marin/optimizer.py": "def update(x, gradient):\n    return x - gradient\n",
        "src/toner/fast/roots.py": "def root(x):\n    return x ** 0.5\n",
    }
    for name, content in pinned_files.items():
        path = pinned / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    record(
        previous / "experiment/upstream_manifest.json",
        {"revision": "fixture", "files": {name: sha256(pinned / name) for name in pinned_files}},
    )
    base_config = {
        "engine": "pytorch",
        "model": {"hidden": 512, "sequence_length": 4096, "layers": 32},
        "training": {
            "steps": 5120,
            "tokens": 2684354560,
            "batch_tokens": 524288,
            "world_size": 8,
            "seed": 0,
            "heldout_windows": 64,
            "global_batch_sequences": 128,
        },
        "data": {"mixture_weights": {"dclm": 3.8, "starcoder": 0.25, "proofpile": 0.055}},
        "runtime": {"eval_every": 250, "eval_max_windows": 0},
        "schedule": {"stable_steps": 1024, "decay_steps": 4096, "kind": "linear"},
        "optimizer": {"lr": 0.02262741696, "name": "muont", "weight_decay": 0.1},
    }
    for arm in history.ARMS:
        record(previous / f"experiment/configs/{arm}_0p5m_seed0.json", base_config | {"experiment": {"arm": arm}})
    cluster = {
        "wave_name": "upstream-full-model-corrected-validation-unreleased",
        "nodes": list(NODES),
        "python": PYTHON,
        "output_root": str(tmp_path),
        "gpus_per_run": 8,
        "environment": ENVIRONMENT,
        "wave": [{"arm": arm, "seed": 0, "batch": "0p5m"} for arm in history.ARMS],
    }
    record(previous / "experiment/cluster.json", cluster)
    record(tmp_path / "data/experiment_receipt.json", {"status": "verified", "shards": ["pinned-train", "pinned-c4"]})
    original_helpers = tmp_path / "controllers/upstream-full-model-corrected-20261117/tracking"
    original_helpers.mkdir(parents=True)
    original_names = (
        "upstream_validation.py",
        "swanlab_refresh_upload.py",
        "refresh_history.py",
        "compare_runs.py",
        "analyze_tuning.py",
        "prepare_tuning.py",
        "launch_wave5.py",
        "validate_wave5.py",
        "wave7_release.py",
    )
    for name in original_names:
        shutil.copy2(TRACKING / name, original_helpers / name)
    original_archive = preparation / "source-upstream-full-model-corrected-20261117.tar"
    archive(original_archive, previous, original_helpers, original_names)
    work = tmp_path / "validation/upstream_full_model_corrected"
    work.mkdir(parents=True)
    numerical_path = preparation / "upstream-full-model-corrected-binding.json"
    numerical = {
        "status": "frozen_pending_upper_lr_and_synthetic_probe",
        "root": str(tmp_path),
        "candidate": str(previous),
        "helpers": str(original_helpers),
        "helpers_sha256": {name: sha256(original_helpers / name) for name in original_names},
        "archive": str(original_archive),
        "archive_sha256": sha256(original_archive),
        "files_sha256": candidate_inventory(previous),
        "environment": ENVIRONMENT,
        "arms": list(history.ARMS),
        "required_gates": list(GATES),
        "data_receipt_sha256": sha256(tmp_path / "data/experiment_receipt.json"),
        "commands": validation_commands(PYTHON, previous, tmp_path, work),
    }
    numerical["source_sha256"] = source_hashes(numerical["files_sha256"])
    record(numerical_path, numerical)
    receipt_path = preparation / "validation-upstream-full-model-corrected-gpu.json"
    validation_command = [
        PYTHON,
        "-m",
        "experiment.validate",
        "--gpu",
        "--data",
        str(tmp_path / "data"),
        "--data-receipt",
        str(tmp_path / "data/experiment_receipt.json"),
        "--work",
        str(work),
        "--receipt",
        str(receipt_path),
    ]
    executions = []
    for index, command in enumerate(numerical["commands"]):
        log = work / f"command_{index}.log"
        log.write_text(f"zero-exit owned command {index}\n")
        executions.append({"command": command, "log": str(log), "log_sha256": sha256(log), "exit_code": 0})
    owner = {"controller_pid": 301264, "controller_start_ticks": "1178276991"}
    receipt = {
        "owner": owner,
        "candidate_path": str(previous),
        "source_sha256": numerical["source_sha256"],
        "config_sha256": {arm: sha256(previous / f"experiment/configs/{arm}_0p5m_seed0.json") for arm in history.ARMS},
        "environment": ENVIRONMENT,
        "data_receipt_sha256": numerical["data_receipt_sha256"],
        "checks": dict.fromkeys(GATES, True),
        "evidence": {
            "validation_command": validation_command,
            "gpu_commands": executions,
            "gpu_receipt": str(work / "gpu_checks.json"),
            "real_data_smokes": {
                arm: {"out": str(work / f"{arm}_continuous"), "continuation": {"tensor_comparisons": 8}}
                for arm in history.ARMS
            },
        },
    }
    record(receipt_path, receipt)
    record(
        work / "gpu_checks.json",
        {"world": 2, "source_sha256": numerical["source_sha256"], "checks": dict.fromkeys(SYNTHETIC_GATES, True)},
    )
    state_path = preparation / "upstream-full-model-corrected-state.json"
    record(
        state_path,
        {
            **owner,
            "status": "passed",
            "command": validation_command,
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "keeper_restore": 0,
            "validation_sha256": sha256(receipt_path),
            "binding_sha256": sha256(numerical_path),
        },
    )
    source = tmp_path / "candidates/upstream-full-model-production-20261117"
    shutil.copytree(previous, source, symlinks=True)
    (source / "experiment/execution_owner.py").write_text(
        "from .production_owner import require_production_owner\n"
        "def require_training_owner():\n    return require_production_owner()\n"
    )
    production_owner = TRACKING / "upstream_production_owner.py"
    (source / "experiment/production_owner.py").write_text(
        production_owner.read_text() if production_owner.exists() else "def require_production_owner():\n    return\n"
    )
    (source / "experiment/tests/test_upstream_release.py").write_text(
        "def test_exact_production_contract():\n    assert True\n"
    )
    record(source / "experiment/cluster.json", cluster | {"wave_name": "wave9"})
    release_path, helpers = history.paths(tmp_path)
    helpers.mkdir(parents=True)
    for name in original_names:
        shutil.copy2(original_helpers / name, helpers / name)
    for name in history.EXTRA_HELPERS:
        current = TRACKING / name
        (helpers / name).write_text(current.read_text() if current.exists() else "# CPU deployment fixture\n")
    shutil.copy2(source / "experiment/production_owner.py", helpers / "upstream_production_owner.py")
    shutil.copy2(source / "experiment/tests/test_upstream_release.py", helpers / "test_upstream_production_owner.py")
    files = candidate_inventory(source)
    production_preparation = source / "production-preparation.json"
    record(
        production_preparation,
        {
            "status": "prepared_owner_only_no_production_release",
            "candidate": str(source),
            "validation_binding_sha256": sha256(numerical_path),
            "files_sha256": files,
            "changed_files": sorted(history.CHANGES),
            "owner_sha256": files["experiment/production_owner.py"],
            "test_sha256": files["experiment/tests/test_upstream_release.py"],
        },
    )
    cpu_directory = preparation / "production-cpu/20261117T0900"
    cpu_directory.mkdir(parents=True)
    cpu_archive = cpu_directory / "source.tar"
    archive(cpu_archive, source, helpers, ())
    cpu_log = cpu_directory / "candidate.log"
    cpu_log.write_text("108 CPU tests passed; CUDA hidden\n")
    cpu_path = cpu_directory / "receipt.json"
    record(
        cpu_path,
        {
            "status": "production_cpu_verified_gpu_release_pending",
            "candidate": str(source),
            "preparation_sha256": sha256(production_preparation),
            "files_sha256": files,
            "command": [
                PYTHON,
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
                "experiment/tests",
            ],
            "cwd": str(source),
            "exit_code": 0,
            "cuda_visible_devices": "",
            "log": str(cpu_log),
            "log_sha256": sha256(cpu_log),
            "archive": str(cpu_archive),
            "archive_sha256": sha256(cpu_archive),
        },
    )
    launch_archive = preparation / "source-wave9-launch-20261117.tar"
    names = (*original_names, *history.EXTRA_HELPERS)
    archive(launch_archive, source, helpers, names)
    release = {
        "status": "frozen_validated_launch_release",
        "root": str(tmp_path),
        "candidate": str(source),
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in names},
        "archive": str(launch_archive),
        "archive_sha256": sha256(launch_archive),
        "validation_binding_sha256": sha256(numerical_path),
        "validation_sha256": sha256(receipt_path),
        "state_sha256": sha256(state_path),
        "preparation": str(production_preparation),
        "preparation_sha256": sha256(production_preparation),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": sha256(cpu_path),
        "files_sha256": files,
        "source_sha256": source_hashes(files),
        "data_receipt_sha256": numerical["data_receipt_sha256"],
        "environment": ENVIRONMENT,
        "arms": list(history.ARMS),
        "nodes": list(NODES),
        "commands": history.commands(tmp_path, source),
        "monitor_interval_seconds": 900,
        "training_timeout_seconds": 21600,
    }
    record(release_path, release)
    monkeypatch.setattr(history, "__file__", str(helpers / "upstream_run_history.py"))
    return tmp_path, source, release


def run_history(root: Path, source: Path, release: dict, arm: str, *, steps: int = 5120, delta: float = 0.0) -> Path:
    directory = root / "runs/wave9" / arm
    config = json.loads((source / f"experiment/configs/{arm}_0p5m_seed0.json").read_text())
    manifest_sources = executed_sources(source, release)
    manifest_sources = {
        name.replace("artifacts/pinned-sources-materialized/", "artifacts/external-sources/"): digest
        for name, digest in manifest_sources.items()
    }
    record(
        directory / "run.json",
        {
            "config": config,
            "data_receipt_sha256": release["data_receipt_sha256"],
            "source_sha256": manifest_sources,
            "status": "complete" if steps == 5120 else "running",
            "completed_steps": steps,
            "elapsed_seconds": 6000.0 + 1000 * delta,
            "start_time": "2026-11-17T01:00:00+00:00",
            "end_time": "2026-11-17T03:00:00+00:00",
            "source_revision": "fixture",
            "experiment_git_revision": None,
            "python": "3.13",
            "torch": "recorded",
            "cuda": "recorded",
            "driver": "recorded",
            "gpu": "NVIDIA A100-SXM4-40GB",
            "world_size": 8,
        },
    )
    rows = []
    for step in range(1, steps + 1):
        row = {
            "step": step,
            "tokens": step * 524288,
            "train_loss": 4.0 - 0.0001 * step + delta,
            "step_seconds": 1.0 + delta,
            "elapsed_seconds": 10.0 + step * (1.0 + delta),
            "end_to_end_seconds": 20.0 + step * (1.0 + delta),
            "peak_allocated_bytes": 20 * 2**30,
        }
        if step in EVALUATION_STEPS:
            selection = 3.8 - 0.0001 * step + delta
            row.update(
                c4_loss=4.0 - 0.0001 * step + delta,
                selection_loss=selection,
                heldout_losses={f"heldout/{name}": selection for name in config["data"]["mixture_weights"]},
            )
        rows.append(row)
    (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if steps == 5120:
        checkpoint = directory / "checkpoints/step_005120"
        record(checkpoint / "complete.json", {"step": 5120, "world": 8})
        for rank in range(8):
            (checkpoint / f"rank_{rank}.pt").write_bytes(f"rank-{rank} nonempty endpoint".encode())
    return directory


def test_bound_histories_use_full_production_and_pinned_source_identity(release_fixture):
    root, source, release = release_fixture
    run_history(root, source, release, "muont")
    accepted = history.binding(root)
    measured = history.completed_run(root, "muont", accepted)
    assert measured["observed_tokens"] == 2684354560
    assert measured["observed_updates"] == 5120
    assert measured["complete"] is True
    assert len(measured["checkpoint_rank_bytes"]) == 8
    original_source = root / "candidates/upstream-full-model-corrected-validator-20261117"
    assert sha256(source / "experiment/train.py") == sha256(original_source / "experiment/train.py")
    assert (
        sha256(source / "experiment/production_owner.py") == release["source_sha256"]["experiment/production_owner.py"]
    )


def test_full_budget_commands_match_explicit_node_arm_release_contract(release_fixture):
    root, source, _ = release_fixture
    planned = history.binding(root)["commands"]
    for node, arm in zip(NODES, history.ARMS, strict=True):
        assert planned[node] == [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "experiment.train",
            "--config",
            str(source / f"experiment/configs/{arm}_0p5m_seed0.json"),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--out",
            str(root / "runs/wave9" / arm),
            "--launch-release",
            str(root / "run-preparation/wave9-launch-binding.json"),
        ]


@pytest.mark.parametrize("fault", ["source", "data", "gpu_log", "state"])
def test_helper_observation_survives_numerical_changes_but_analysis_refuses(release_fixture, fault):
    root, source, _ = release_fixture
    if fault == "source":
        (source / "experiment/train.py").write_text("changed numerical implementation\n")
    elif fault == "data":
        record(root / "data/experiment_receipt.json", {"replacement": True})
    elif fault == "gpu_log":
        (root / "validation/upstream_full_model_corrected/command_12.log").write_text("failed command\n")
    else:
        path = root / "run-preparation/upstream-full-model-corrected-state.json"
        state = json.loads(path.read_text())
        state["keeper_restore"] = 1
        record(path, state)
    assert history.helper_binding(root)["status"] == "frozen_validated_launch_release"
    with pytest.raises(ValueError):
        history.binding(root)


@pytest.mark.parametrize("fault", ["helper", "archive", "node_arm", "bounded_command"])
def test_immutable_helper_or_protocol_changes_refuse_even_observation(release_fixture, fault):
    root, _, release = release_fixture
    release_path, helpers = history.paths(root)
    if fault == "helper":
        (helpers / "upstream_run_history.py").write_text("changed observer\n")
    elif fault == "archive":
        Path(release["archive"]).write_bytes(b"replacement archive")
    elif fault == "node_arm":
        release["commands"][NODES[0]], release["commands"][NODES[1]] = (
            release["commands"][NODES[1]],
            release["commands"][NODES[0]],
        )
        record(release_path, release)
    else:
        release["commands"][NODES[0]].extend(["--max-steps", "24"])
        record(release_path, release)
    with pytest.raises(ValueError):
        history.helper_binding(root)


@pytest.mark.parametrize("fault", ["command", "gate", "restoration", "synthetic"])
def test_passed_state_cannot_replace_command_gate_or_restoration_evidence(release_fixture, fault):
    root, _, release = release_fixture
    preparation = root / "run-preparation"
    receipt_path = preparation / "validation-upstream-full-model-corrected-gpu.json"
    state_path = preparation / "upstream-full-model-corrected-state.json"
    receipt, state = json.loads(receipt_path.read_text()), json.loads(state_path.read_text())
    if fault == "command":
        receipt["evidence"]["gpu_commands"][-1]["command"] = receipt["evidence"]["gpu_commands"][-2]["command"]
    elif fault == "gate":
        receipt["checks"]["gpu_upstream_scalar_step"] = False
    elif fault == "restoration":
        state["keeper_restore"] = False
    else:
        path = root / "validation/upstream_full_model_corrected/gpu_checks.json"
        synthetic = json.loads(path.read_text())
        synthetic["checks"]["gpu_upstream_scalar_step"] = False
        record(path, synthetic)
    record(receipt_path, receipt)
    state["validation_sha256"] = sha256(receipt_path)
    record(state_path, state)
    release.update(validation_sha256=sha256(receipt_path), state_sha256=sha256(state_path))
    record(history.paths(root)[0], release)
    with pytest.raises(ValueError):
        history.binding(root)


def test_rehashed_extra_numerical_change_still_exceeds_production_migration(release_fixture):
    root, source, release = release_fixture
    (source / "experiment/block_toner.py").write_text("def step(gradient, lr):\n    return -2 * lr * gradient\n")
    release["files_sha256"] = candidate_inventory(source)
    release["source_sha256"] = source_hashes(release["files_sha256"])
    record(history.paths(root)[0], release)
    with pytest.raises(ValueError):
        history.binding(root)


@pytest.mark.parametrize("fault", ["rank7", "evaluation", "source_alias", "duplicate", "complete_truncated"])
def test_completed_history_rejects_missing_endpoint_or_unrecoverable_provenance(release_fixture, fault):
    root, source, release = release_fixture
    directory = run_history(root, source, release, "muont")
    if fault == "rank7":
        (directory / "checkpoints/step_005120/rank_7.pt").write_bytes(b"")
    elif fault == "source_alias":
        manifest = json.loads((directory / "run.json").read_text())
        name = next(name for name in manifest["source_sha256"] if name.startswith("artifacts/external-sources/"))
        manifest["source_sha256"][name.replace("external-sources", "pinned-sources-materialized")] = manifest[
            "source_sha256"
        ][name]
        record(directory / "run.json", manifest)
    else:
        path = directory / "metrics_from_000000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "evaluation":
            for key in ("c4_loss", "selection_loss", "heldout_losses"):
                del rows[-1][key]
        elif fault == "duplicate":
            rows[250] = rows[249]
        else:
            rows.pop()
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        history.completed_run(root, "muont", release)


def test_analysis_preserves_five_windows_and_each_control_direction(release_fixture):
    root, source, release = release_fixture
    deltas = {
        "muont": 0.0,
        "outer_leftdiag_fused": -0.01,
        "outer_leftdiag_upstream_fused": -0.02,
        "null_leftdiag_upstream_fused": 0.005,
    }
    for arm, delta in deltas.items():
        run_history(root, source, release, arm, delta=delta)
    result = analyze(root)
    assert result["wave"] == "wave9"
    assert result["source_release_sha256"] == sha256(history.paths(root)[0])
    assert len(result["analyses"]) == 4
    expected_pairs = (
        ("muont", history.ARMS[1:]),
        ("outer_leftdiag_fused", ("outer_leftdiag_upstream_fused",)),
        ("null_leftdiag_upstream_fused", ("outer_leftdiag_upstream_fused",)),
        ("muont", ("null_leftdiag_upstream_fused",)),
    )
    for analysis, (reference, candidates) in zip(result["analyses"], expected_pairs, strict=True):
        assert analysis["baseline"] == f"wave9/{reference}"
        assert set(analysis["comparisons"]) == {f"wave9/{arm}" for arm in (reference, *candidates)}
        for arm in candidates:
            pair = analysis["comparisons"][f"wave9/{arm}"]
            delta = deltas[arm] - deltas[reference]
            assert pair["final_comparison_eligible"] is True
            assert pair["pairing"]["kind"] == "paired_updates"
            assert [window["end_tokens"] for window in pair["training_windows"]] == [
                WINDOW * index for index in range(1, 6)
            ]
            assert [window["delta"] for window in pair["training_windows"]] == pytest.approx([delta] * 5)
            for key in ("c4_loss", "selection_loss"):
                assert len(pair["evaluations"][key]["shared_tokens"]) == 22
                assert len(pair["evaluations"][key]["late_tokens"]) == 11
                assert pair["evaluations"][key]["final_delta"] == pytest.approx(delta)
    assert result["elapsed_time_diagnostic"]["runs"]["wave9/muont"][-1]["censored"] is True


def test_interim_analysis_uses_shared_prefix_and_never_publishes_final_losses(release_fixture):
    root, source, release = release_fixture
    for index, arm in enumerate(history.ARMS):
        run_history(root, source, release, arm, steps=1024 + 256 * index, delta=-0.001 * index)
    measured = history.measured_run(root, history.ARMS[0], release)
    assert measured["observed_tokens"] == WINDOW
    assert measured["complete"] is False
    with pytest.raises(ValueError):
        history.completed_run(root, history.ARMS[0], release)
    result = analyze(root)
    first = result["analyses"][0]
    for arm in history.ARMS[1:]:
        pair = first["comparisons"][f"wave9/{arm}"]
        assert pair["common_tokens"] == WINDOW
        assert len(pair["training_windows"]) == 1
        assert pair["final_comparison_eligible"] is False
        assert pair["evaluations"]["selection_loss"]["final_delta"] is None
        assert pair["runtime"]["delta"]["final_total_seconds"] is None


def test_history_binding_does_not_import_torch_or_execute_process_operations(release_fixture):
    root, _, release = release_fixture
    script = """
import runpy, subprocess, sys
def refused(*args, **kwargs):
    raise AssertionError("history binding attempted a process operation")
subprocess.run = subprocess.Popen = subprocess.check_output = refused
module = runpy.run_path(sys.argv[1])
result = module["binding"](__import__("pathlib").Path(sys.argv[2]))
assert result["status"] == "frozen_validated_launch_release"
assert "torch" not in sys.modules
assert not any(name.startswith("experiment.") for name in sys.modules)
"""
    subprocess.run(
        [PYTHON, "-c", script, str(Path(release["helpers"]) / "upstream_run_history.py"), str(root)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(TRACKING)},
        timeout=30,
    )
    assert history.helper_binding(root)["commands"] == release["commands"]


def terminal_snapshot(root: Path, source: Path, release: dict) -> dict:
    workers = []
    for index, arm in enumerate(history.ARMS):
        run_history(root, source, release, arm, delta=-0.001 * index)
        log = root / "runs/wave9" / f"{arm}.log"
        log.write_text(f"completed uninterrupted full-budget {arm}\n")
        workers.append(
            {
                "arm": arm,
                "status": "complete",
                "receipt": {
                    "arm": arm,
                    "node": NODES[index],
                    "status": "complete",
                    "release_binding_sha256": sha256(history.paths(root)[0]),
                    "exit_code": 0,
                    "cleanup_exit_code": 0,
                    "keeper_restore": 0,
                    "command": release["commands"][NODES[index]],
                    "log": str(log),
                    "log_sha256": sha256(log),
                    "completed_run": history.completed_run(root, arm, release),
                },
            }
        )
    snapshot = {
        "status": "complete",
        "all_workers_terminal": True,
        "binding_sha256": sha256(history.paths(root)[0]),
        "workers": workers,
    }
    record(root / "run-preparation/monitor-wave9/latest.json", snapshot)
    return snapshot


@pytest.mark.parametrize("fault", ["not_terminal", "duplicate_arm", "restoration", "changed_log", "changed_history"])
def test_cloud_import_refuses_changed_or_nonterminal_saved_worker_evidence(release_fixture, fault):
    root, source, release = release_fixture
    snapshot = terminal_snapshot(root, source, release)
    if fault == "not_terminal":
        snapshot["all_workers_terminal"] = False
    elif fault == "duplicate_arm":
        snapshot["workers"][-1] = snapshot["workers"][-2]
    elif fault == "restoration":
        snapshot["workers"][-1]["receipt"]["keeper_restore"] = False
    elif fault == "changed_log":
        Path(snapshot["workers"][-1]["receipt"]["log"]).write_text("unverified cleanup\n")
    else:
        manifest_path = root / "runs/wave9/muont/run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["elapsed_seconds"] += 1
        record(manifest_path, manifest)
    record(root / "run-preparation/monitor-wave9/latest.json", snapshot)
    script = """
import json, sys
from pathlib import Path
from swanlab_upstream_run import require_completed_snapshot
root=Path(sys.argv[1])
release=json.loads((root/"run-preparation/wave9-launch-binding.json").read_text())
try:
    require_completed_snapshot(root, release)
except RuntimeError:
    print("refused")
else:
    raise AssertionError("unverified terminal evidence reached cloud import")
"""
    result = subprocess.run(
        [str(LOGGING_PYTHON), "-c", script, str(root)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(TRACKING), "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=30,
    )
    assert result.stdout.strip() == "refused"
    assert not (root / "run-preparation/swanlab-upload-wave9.json").exists()


def test_private_cloud_import_roundtrip_verifies_every_scalar_and_full_history(release_fixture):
    root, source, release = release_fixture
    terminal_snapshot(root, source, release)
    credential = root / "fake-cloud.env"
    credential.write_text("SWANLAB_API_KEY=offline-fixture\n")
    script = """
import json, runpy, sys
from pathlib import Path
from types import SimpleNamespace
import swanlab
root=Path(sys.argv[1]); helpers=Path(sys.argv[2]); credential=Path(sys.argv[3])
sys.path.insert(0, str(helpers))
stored={}
class CloudRun:
    def __init__(self, options):
        assert options["public"] is False
        assert options["group"] == "wave9"
        assert options["job_type"] == "historical-import"
        self.id=options["id"]; self.path="offline/output-side-normalization-marin/"+self.id
        self.url="offline:"+self.id; self.state="RUNNING"; self.points={}
        self.profile={"config": {key: {"value": value} for key,value in options["config"].items()}}
    def log(self, row, step):
        assert step not in self.points
        self.points[step]=row
    def finish(self, **kwargs):
        self.state="FINISHED"
    def wrapper(self):
        return SimpleNamespace(ok=True)
    def metrics(self, *, keys, x_axis="step", **kwargs):
        columns=[]
        for key in keys:
            points=[]
            for step,row in sorted(self.points.items()):
                if key not in row:
                    continue
                point={"step":step, "value":row[key]}
                if x_axis=="tokens":
                    point["index"]=row["tokens"]
                points.append(point)
            columns.append({"key":key, "metrics":points})
        return {"list":columns}
class Api:
    username="offline"
    def __init__(self, **kwargs):
        pass
    def projects(self, *args, **kwargs):
        return [SimpleNamespace(name="output-side-normalization-marin")]
    def project(self, path):
        return SimpleNamespace(project_id="private", visibility="PRIVATE", url="offline:project",
                               wrapper=lambda:SimpleNamespace(ok=True))
    def run(self, path):
        return stored[path]
def initialize(**kwargs):
    run=CloudRun(kwargs); stored[run.path]=run; return run
swanlab.Api=Api
swanlab.login=lambda **kwargs:None
swanlab.init=initialize
swanlab.define_metric=lambda *args,**kwargs:None
sys.argv=[str(helpers/"swanlab_upstream_run.py"), "--root", str(root), "--credential-file", str(credential)]
runpy.run_path(str(helpers/"swanlab_upstream_run.py"), run_name="__main__")
assert len(stored)==4
for run in stored.values():
    assert len(run.points)==5120 and run.points[5120]["tokens"]==2684354560
assert "torch" not in sys.modules
"""
    subprocess.run(
        [str(LOGGING_PYTHON), "-c", script, str(root), release["helpers"], str(credential)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(TRACKING), "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=60,
    )
    uploaded = json.loads((root / "run-preparation/swanlab-upload-wave9.json").read_text())
    assert uploaded["status"] == "verified"
    assert uploaded["visibility"] == "PRIVATE"
    assert set(uploaded["runs"]) == set(history.ARMS)
    for entry in uploaded["runs"].values():
        assert entry["status"] == "verified"
        assert entry["verification"]["point_counts"]["train/loss"] == 5120
        assert entry["verification"]["point_counts"]["eval/c4_loss"] == 22
        assert entry["verification"]["point_counts"]["eval/selection_loss"] == 22
        assert entry["verification"]["configuration_verified"] is True
        assert entry["verification"]["token_axis_verified"] is True
