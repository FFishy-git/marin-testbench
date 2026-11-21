# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU full-worker acceptance with real campaign bindings and subprocess fakes."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import related_campaign as short
import related_execution as shared
import related_full_campaign as full
import related_full_execution as execution
import torch
from experiment import data_access


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def rewrite(path, change):
    record = short.read(path)
    change(record)
    save(path, record)


def checkpoint(directory, step, campaign, config):
    path = directory / "checkpoints" / f"step_{step:06d}"
    save(path / "complete.json", {"step": step, "world": short.WORLD})
    for rank in range(short.WORLD):
        torch.save(
            {
                "step": step,
                "next_global_sequence": step * config["training"]["global_batch_sequences"],
                "rank": rank,
                "world": short.WORLD,
                "config": config,
                "identity": {
                    "schema": 1,
                    "source_sha256": campaign["source_sha256"],
                    "config_sha256": short.config_identity(config),
                    "data_receipt_sha256": campaign["data_receipt_sha256"],
                },
                "model": {"weight": torch.tensor([[float(step), float(rank)]])},
            },
            path / f"rank_{rank}.pt",
        )


def manifest(directory, step, campaign, config, *, smoke):
    save(
        directory / "run.json",
        {
            "status": "execution_check_complete" if smoke else "complete",
            "completed_steps": step,
            "execution_check": smoke,
            "resume": None,
            "config": config,
            "data": campaign["data"],
            "data_receipt_sha256": campaign["data_receipt_sha256"],
            "source_revision": campaign["source_revision"],
            "source_sha256": campaign["source_sha256"],
            "world_size": short.WORLD,
        },
    )
    checkpoint(directory, step, campaign, config)


def endpoint(campaign, trial, *, steps=None, loss=3.0):
    directory = Path(trial["out"])
    config = short.read(Path(trial["config"]))
    steps = config["training"]["steps"] if steps is None else steps
    manifest(directory, steps, campaign, config, smoke=False)
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
    weights = config["data"]["mixture_weights"]
    losses = {f"heldout/{name}": loss for name in weights}
    mixture = sum(weight * losses[f"heldout/{name}"] for name, weight in weights.items()) / sum(weights.values())
    rows[-1].update(selection_loss=mixture, heldout_losses=losses, c4_loss=2.0)
    (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "output"
    for folder, _ in short.PROFILES.values():
        shutil.copytree(short.TASK / "experiment/comparisons" / folder, source / "experiment/comparisons" / folder)
    for name in (*full.RECIPE_MODULES, "experiment/train.py", "experiment/cluster.json"):
        shutil.copyfile(short.TASK / name, source / name)
    revision = short.read(short.TASK / "experiment/upstream_manifest.json")["revision"]
    upstream = source / "artifacts/external-sources" / f"Toner-{revision}"
    inputs = (
        "reproduce/marin/train.py",
        "src/toner/fast/muon.py",
        "reproduce/marin/configs/130m/muon_0p5m.json",
        "reproduce/marin/configs/130m/muont_0p5m.json",
    )
    for name in inputs:
        target = upstream / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(short.TASK / upstream.relative_to(source) / name, target)
    components = ("dclm", "starcoder", "proofpile", "paloma/c4_en")
    plan_path = upstream / "reproduce/marin/data_plan.json"
    save(plan_path, {"files": [{"component": name, "file": "tiny.bin", "tokens": 1, "docs": 0} for name in components]})
    save(
        source / "experiment/upstream_manifest.json",
        {
            "revision": revision,
            "files": {name: short.sha256(upstream / name) for name in (*inputs, "reproduce/marin/data_plan.json")},
        },
    )
    oracle = source / short.MOUSSE_REFERENCE
    oracle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(short.TASK / short.MOUSSE_REFERENCE, oracle)
    records = []
    for component in components:
        path = root / "data" / component / "tiny.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00\x00\x00\x00")
        records.append(
            {
                "file": f"{component}/tiny.bin",
                "tokens": 1,
                "docs": 0,
                "bytes": 4,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": short.sha256(path),
            }
        )
    save(
        root / "data/experiment_receipt.json",
        {"root": str(root / "data"), "data_plan_sha256": short.sha256(plan_path), "files": records},
    )
    monkeypatch.setattr(data_access, "PLAN", plan_path)
    monkeypatch.setenv(shared.MARKER, "full-worker-inherited-token")
    for key, value in shared.ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    campaign = short.prepare(root, source=source)
    for trial in campaign["trials"]:
        preferred = short.ARMS.index(trial["arm"]) % 3
        endpoint(campaign, trial, loss=3.0 + abs(trial["lr_index"] - preferred))
    return root, source, campaign, full.prepare(root)


def validation_receipt(campaign, wave):
    work = Path(campaign["root"]) / "validation" / wave["wave_name"]
    work.mkdir(parents=True)
    commands, probes, smokes = [], {}, {}
    sources = {name: digest for name, digest in campaign["source_sha256"].items() if name.startswith("experiment/")}

    def command_record(command):
        log = work / f"command_{len(commands)}.log"
        log.write_text("offline successful command\n")
        commands.append({"command": command, "exit_code": 0, "log": str(log), "log_sha256": short.sha256(log)})

    for key, module, checks, world_key in (
        ("gpu_receipt", "experiment.gpu_checks", shared.SYNTHETIC_CHECKS, "world"),
        ("related_gpu_receipt", "experiment.related_gpu_checks", shared.RELATED_CHECKS, "world_size"),
    ):
        path = work / f"{key}.json"
        save(path, {"checks": dict.fromkeys(checks, True), world_key: 2, "source_sha256": sources})
        probes[key] = str(path)
        command_record([*shared.launcher(shared.PYTHON, 2), module, "--receipt", str(path)])
    trials = {trial["name"]: trial for trial in campaign["trials"]}
    hashes = {}
    for name in wave["trials"]:
        trial = trials[name]
        alias = next(alias for alias in trial["aliases"] if alias["wave"] == wave["wave_name"])
        hashes[name] = alias["config_sha256"]
        config = short.read(Path(alias["config"]))
        complete, split = shared.continuation_steps(config)
        continuous, resumed = work / f"{name}_continuous", work / f"{name}_resumed"
        common = shared.trial_command(
            Path(alias["config"]), Path(campaign["data"]), Path(campaign["data_receipt"]), continuous
        )[:-2]
        command_record([*common, "--out", str(continuous), "--max-steps", str(complete)])
        command_record([*common, "--out", str(resumed), "--max-steps", str(split)])
        manifest(continuous, complete, campaign, config, smoke=True)
        checkpoint(resumed, split, campaign, config)
        resume_step, pre_refresh = split, None
        if trial["arm"] in shared.RELATED_ARMS and trial["arm"] != "muon_control":
            resume_step = split + 1
            prefix = work / f"{name}_pre_refresh_continuous"
            command_record(
                [
                    *common,
                    "--out",
                    str(resumed),
                    "--max-steps",
                    str(resume_step),
                    "--resume",
                    str(resumed / "checkpoints" / f"step_{split:06d}"),
                ]
            )
            command_record([*common, "--out", str(prefix), "--max-steps", str(resume_step)])
            manifest(prefix, resume_step, campaign, config, smoke=True)
            checkpoint(resumed, resume_step, campaign, config)
            pre_refresh = {"tensor_comparisons": short.WORLD, "max_absolute_error": 0.0}
        command_record(
            [
                *common,
                "--out",
                str(resumed),
                "--max-steps",
                str(complete),
                "--resume",
                str(resumed / "checkpoints" / f"step_{resume_step:06d}"),
            ]
        )
        manifest(resumed, complete, campaign, config, smoke=True)
        smokes[name] = {
            "out": str(continuous),
            "completed_steps": complete,
            "split_steps": split,
            "continuation": {"tensor_comparisons": short.WORLD, "max_absolute_error": 0.0},
            "pre_refresh_continuation": pre_refresh,
        }
    path = work / "receipt.json"
    save(
        path,
        {
            "wave_name": wave["wave_name"],
            "config_sha256": hashes,
            "source_sha256": sources,
            "data_receipt_sha256": campaign["data_receipt_sha256"],
            "environment": shared.ENVIRONMENT,
            "checks": dict.fromkeys(shared.REQUIRED_CHECKS, True),
            "evidence": {"cpu_test_exit_code": 0, "gpu_commands": commands, "real_data_smokes": smokes, **probes},
        },
    )
    return path


@pytest.fixture
def validated(prepared):
    root, source, short_campaign, campaign = prepared
    wave = next(wave for wave in campaign["waves"] if wave["profile"] == "controlled" and wave["seed"] == 1)
    return root, source, short_campaign, campaign, wave, validation_receipt(campaign, wave)


def process_boundary(monkeypatch, campaign, state_path, *, fault=None):
    events = []
    trials = {trial["out"]: trial for trial in campaign["trials"]}

    class Child:
        def __init__(self, command, **kwargs):
            assert not events or events[-1]["joined"]
            trial = trials[command[command.index("--out") + 1]]
            assert command == [
                shared.PYTHON,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                "-m",
                "experiment.train",
                "--config",
                trial["config"],
                "--data",
                campaign["data"],
                "--data-receipt",
                campaign["data_receipt"],
                "--out",
                trial["out"],
            ]
            assert kwargs["cwd"] == campaign["source"]
            assert kwargs["env"][shared.MARKER] == "full-worker-inherited-token"
            assert all(kwargs["env"][key] == value for key, value in shared.ENVIRONMENT.items())
            assert kwargs["stdin"] == subprocess.DEVNULL
            assert kwargs["stderr"] == subprocess.STDOUT
            assert kwargs["start_new_session"] is True
            assert not Path(trial["out"]).exists()
            self.pid = 1000 + len(events)
            self.trial, self.stream = trial, kwargs["stdout"]
            self.event = {"trial": trial["name"], "joined": False}
            events.append(self.event)
            self.stream.write("offline child diagnostic\n")

        def wait(self):
            persisted = short.read(state_path)
            assert persisted["children"][-1]["pid"] == self.pid
            assert persisted["children"][-1]["status"] == "running"
            self.event["joined"] = True
            if fault == "exit":
                directory = Path(self.trial["out"])
                directory.mkdir()
                (directory / "failure.txt").write_text("preserved child failure\n")
                return 17
            if fault == "short":
                endpoint(campaign, self.trial, steps=short.STEPS)
                rewrite(
                    Path(self.trial["out"]) / "run.json",
                    lambda record: record.update(completed_steps=full.STEPS),
                )
            else:
                endpoint(campaign, self.trial)
            if fault == "missing_rank":
                (Path(self.trial["out"]) / "checkpoints/step_005120/rank_7.pt").unlink()
            if fault == "binding":
                path = Path(campaign["source"]) / "experiment/train.py"
                path.write_text(path.read_text() + "\n# changed after child startup\n")
            return 0

    def popen(command, **kwargs):
        if fault == "spawn":
            kwargs["stdout"].write("offline spawn failure\n")
            raise OSError("offline subprocess start failed")
        return Child(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    return events


def test_preflight_binds_each_full_wave_and_is_read_only(prepared):
    root, _, _, campaign = prepared
    receipts = [validation_receipt(campaign, wave) for wave in campaign["waves"]]
    before = {str(path): short.sha256(path) for path in root.rglob("*") if path.is_file()}
    names = [trial["name"] for trial in campaign["trials"]]
    result = execution.validate_trials(root, names, receipts)
    assert len(result["trials"]) == 12
    assert len(result["validations"]) == 4
    assert result["campaign_sha256"] == short.sha256(full.campaign_path(root))
    assert all(trial["steps"] == 5120 and trial["seed"] in (1, 2) for trial in result["trials"])
    assert all(validation["smoke_artifacts_sha256"] for validation in result["validations"])
    assert {str(path): short.sha256(path) for path in root.rglob("*") if path.is_file()} == before
    assert not (full.campaign_path(root).parent / "runs").exists()


def test_shared_trials_accept_paper_alias_but_uncovered_seed_or_profile_does_not(prepared):
    root, _, _, campaign = prepared
    wave = next(wave for wave in campaign["waves"] if wave["profile"] == "paper_adapters" and wave["seed"] == 1)
    receipt = validation_receipt(campaign, wave)
    shared_names = wave["trials"][:2]
    result = execution.validate_trials(root, shared_names, [receipt])
    assert [trial["validation_wave"] for trial in result["trials"]] == [wave["wave_name"]] * 2
    for trial in campaign["trials"]:
        if trial["name"] not in wave["trials"]:
            with pytest.raises(ValueError):
                execution.validate_trials(root, [trial["name"]], [receipt])
    assert not (full.campaign_path(root).parent / "runs").exists()


def test_short_validation_cannot_authorize_fresh_full_seed(prepared):
    root, _, short_campaign, campaign = prepared
    receipt = validation_receipt(short_campaign, short_campaign["waves"][0])
    with pytest.raises(ValueError):
        execution.validate_trials(root, [campaign["trials"][0]["name"]], [receipt])
    assert not (full.campaign_path(root).parent / "runs").exists()


@pytest.mark.parametrize("damage", ["source", "config", "short_history", "data", "gpu_log", "compiled_gate", "wave"])
def test_changed_bound_evidence_blocks_compute_without_creating_state(validated, monkeypatch, damage):
    root, source, short_campaign, campaign, wave, receipt = validated
    record = short.read(receipt)
    if damage == "source":
        path = source / "experiment/train.py"
        path.write_text(path.read_text() + "\n# changed source identity\n")
    elif damage == "config":
        rewrite(Path(campaign["trials"][0]["config"]), lambda config: config["training"].update(seed=0))
    elif damage == "short_history":
        path = Path(short_campaign["trials"][0]["out"]) / "metrics_from_000000.jsonl"
        path.write_text(path.read_text().replace('"train_loss": 4.0', '"train_loss": 4.2'))
    elif damage == "data":
        (root / "data/dclm/tiny.bin").write_bytes(b"\x01\x00\x00\x00")
    elif damage == "gpu_log":
        Path(record["evidence"]["gpu_commands"][0]["log"]).write_text("changed GPU evidence\n")
    elif damage == "compiled_gate":
        rewrite(
            Path(record["evidence"]["gpu_receipt"]),
            lambda probe: probe["checks"].update(synthetic_compiled_body=False),
        )
    else:
        rewrite(receipt, lambda record: record["config_sha256"].pop(wave["trials"][-1]))
    state = root / "worker.json"
    events = process_boundary(monkeypatch, campaign, state)
    with pytest.raises(ValueError):
        execution.run_trials(root, wave["trials"][:2], [receipt], state)
    assert events == []
    assert not state.exists()
    assert not (full.campaign_path(root).parent / "runs").exists()


@pytest.mark.parametrize("phase", ["final", "pre_refresh", "summary"])
def test_actual_all_rank_comparisons_block_forged_continuation(validated, monkeypatch, phase):
    root, _, _, campaign, wave, receipt = validated
    name = wave["trials"][2]
    smoke = short.read(receipt)["evidence"]["real_data_smokes"][name]
    if phase == "summary":
        rewrite(
            receipt,
            lambda record: record["evidence"]["real_data_smokes"][name]["continuation"].update(tensor_comparisons=1),
        )
    else:
        step = smoke["completed_steps"] if phase == "final" else smoke["split_steps"] + 1
        path = Path(smoke["out"]).with_name(f"{name}_resumed") / "checkpoints" / f"step_{step:06d}" / "rank_7.pt"
        state = torch.load(path, weights_only=False)
        state["model"]["weight"][0, 0] += 0.25
        torch.save(state, path)
    state_path = root / "worker.json"
    events = process_boundary(monkeypatch, campaign, state_path)
    with pytest.raises((ValueError, AssertionError)):
        execution.run_trials(root, wave["trials"][:2], [receipt], state_path)
    assert events == []
    assert not state_path.exists()


def test_worker_joins_sequential_full_budget_children_and_records_completion(validated, monkeypatch):
    root, _, _, campaign, wave, receipt = validated
    names, state_path = wave["trials"][:2], root / "worker.json"
    events = process_boundary(monkeypatch, campaign, state_path)
    result = execution.run_trials(root, names, [receipt], state_path)
    assert short.read(state_path) == result
    assert result["status"] == "complete" and result["completed_trials"] == 2
    assert [event["trial"] for event in events] == names
    assert all(event["joined"] for event in events)
    assert result["allocation_restoration"] == "owned_by_parent_wrapper"
    for child in result["children"]:
        assert child["exit_code"] == 0 and child["status"] == "complete"
        assert child["completion"]["complete"] is True
        assert child["completion"]["observed_steps"] == 5120
        assert child["completion"]["observed_tokens"] == 2684354560
        assert child["log_sha256"] == short.sha256(Path(child["log"]))
        assert child["completion"]["history_sha256"]


@pytest.mark.parametrize("fault", ["exit", "short", "missing_rank", "binding", "spawn"])
def test_failure_preserves_evidence_and_halts_remaining_trials(validated, monkeypatch, fault):
    root, _, _, campaign, wave, receipt = validated
    names, state_path = wave["trials"][:2], root / "worker.json"
    events = process_boundary(monkeypatch, campaign, state_path, fault=fault)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        execution.run_trials(root, names, [receipt], state_path)
    result = short.read(state_path)
    assert result["status"] == "needs_attention" and result["completed_trials"] == 0
    assert len(result["children"]) == 1
    child = result["children"][0]
    assert child["status"] == "needs_attention"
    assert child["log_sha256"] == short.sha256(Path(child["log"]))
    assert len(events) == (0 if fault == "spawn" else 1)
    if fault == "exit":
        assert child["exit_code"] == 17
        assert (Path(child["out"]) / "failure.txt").is_file()
    elif fault in ("short", "missing_rank"):
        assert child["exit_code"] == 0
        assert child["completion"]["complete"] is False
    later = next(trial for trial in campaign["trials"] if trial["name"] == names[1])
    assert not Path(later["out"]).exists()
    assert not Path(later["out"]).with_name(f"{names[1]}.torchrun.log").exists()
    with pytest.raises((FileExistsError, ValueError)):
        execution.run_trials(root, names, [receipt], state_path)


@pytest.mark.parametrize("existing", ["state", "temporary_state", "later_out", "later_log", "symlink", "state_symlink"])
def test_existing_artifacts_prevent_any_child_and_are_preserved(validated, monkeypatch, existing):
    root, _, _, campaign, wave, receipt = validated
    names, state_path = wave["trials"][:2], root / "worker.json"
    later_out = Path(next(trial for trial in campaign["trials"] if trial["name"] == names[1])["out"])
    path = {
        "state": state_path,
        "temporary_state": state_path.with_suffix(".json.tmp"),
        "later_out": later_out,
        "later_log": later_out.with_name(f"{names[1]}.torchrun.log"),
        "symlink": later_out,
        "state_symlink": state_path,
    }[existing]
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing in ("symlink", "state_symlink"):
        path.symlink_to(root / "absent")
    else:
        path.write_text("preserve this evidence\n")
    events = process_boundary(monkeypatch, campaign, state_path)
    with pytest.raises(FileExistsError):
        execution.run_trials(root, names, [receipt], state_path)
    assert events == []
    if existing in ("symlink", "state_symlink"):
        assert path.is_symlink()
        assert not (root / "absent").exists()
    else:
        assert path.read_text() == "preserve this evidence\n"
    if existing != "state":
        assert not state_path.exists()


@pytest.mark.parametrize("overlap", ["log", "inside_out", "out_parent"])
def test_overlapping_state_and_trial_artifacts_are_rejected_without_writes(validated, monkeypatch, overlap):
    root, _, _, campaign, wave, receipt = validated
    trial = campaign["trials"][0]
    out = Path(trial["out"])
    log = out.with_name(f"{out.name}.torchrun.log")
    state = {
        "log": log,
        "inside_out": out / "worker.json",
        "out_parent": out.parent,
    }[overlap]
    events = process_boundary(monkeypatch, campaign, state)
    with pytest.raises(ValueError):
        execution.run_trials(root, wave["trials"][:2], [receipt], state)
    assert events == []
    assert not state.exists()
    assert not out.parent.exists()


@pytest.mark.parametrize("missing", ["marker", "environment"])
def test_unowned_execution_cannot_start_compute(validated, monkeypatch, missing):
    root, _, _, campaign, wave, receipt = validated
    monkeypatch.delenv(shared.MARKER if missing == "marker" else next(iter(shared.ENVIRONMENT)))
    state_path = root / "worker.json"
    events = process_boundary(monkeypatch, campaign, state_path)
    with pytest.raises(RuntimeError):
        execution.run_trials(root, wave["trials"][:2], [receipt], state_path)
    assert events == []
    assert not state_path.exists()


def test_cli_check_only_returns_bound_json_without_allocation_or_state(validated, monkeypatch, capsys):
    root, _, _, _, wave, receipt = validated
    monkeypatch.delenv(shared.MARKER)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "related_full_execution.py",
            "--root",
            str(root),
            "--trial",
            wave["trials"][0],
            "--validation-receipt",
            str(receipt),
            "--check-only",
        ],
    )
    execution.main()
    result = json.loads(capsys.readouterr().out)
    assert result["campaign"] == str(full.campaign_path(root))
    assert result["trials"][0]["name"] == wave["trials"][0]
    assert not (root / "worker.json").exists()
