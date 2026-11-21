# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full queues with genuine temporary evidence and fake OS process boundaries."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import related_campaign as short
import related_execution as shared
import related_full_campaign as full
import related_full_execution as execution
import related_full_sequence as sequence
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
    directory, config = Path(trial["out"]), short.read(Path(trial["config"]))
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
    monkeypatch.setattr(sequence.sequence.allocation, "TASK", source)
    monkeypatch.delenv(shared.MARKER, raising=False)
    campaign = short.prepare(root, source=source)
    for trial in campaign["trials"]:
        preferred = short.ARMS.index(trial["arm"]) % 3
        endpoint(campaign, trial, loss=3.0 + abs(trial["lr_index"] - preferred))
    return root, source, campaign, full.prepare(root)


def specifications(root, wave):
    name = wave["wave_name"]
    directory = root / "preparation" / name
    return [
        {
            "kind": "validation",
            "wave_name": name,
            "work": str(root / "validation" / name),
            "receipt": str(directory / "validation.json"),
            "allocation_state": str(directory / "validation-owned.json"),
            "allocation_log": str(directory / "validation-owned.log"),
        },
        {
            "kind": "full-trials",
            "trials": wave["trials"] if wave["profile"] == "controlled" else wave["trials"][2:],
            "validation_receipts": [str(directory / "validation.json")],
            "worker_state": str(directory / "worker.json"),
            "allocation_state": str(directory / "trials-owned.json"),
            "allocation_log": str(directory / "trials-owned.log"),
        },
    ]


@pytest.fixture
def planned(prepared):
    root, source, short_campaign, campaign = prepared
    wave = campaign["waves"][0]
    path = root / "preparation" / "queue.json"
    plan = sequence.prepare_plan(root, specifications(root, wave), path, node="28.216.17.9")
    return root, source, short_campaign, campaign, path, plan


def validation_receipt(campaign, stage):
    work = Path(stage["work"])
    work.mkdir(parents=True)
    wave = next(wave for wave in campaign["waves"] if wave["wave_name"] == stage["wave_name"])
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
    for name in wave["trials"]:
        trial = trials[name]
        alias = next(alias for alias in trial["aliases"] if alias["wave"] == wave["wave_name"])
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
    save(
        Path(stage["receipt"]),
        {
            "wave_name": wave["wave_name"],
            "config_sha256": stage["config_sha256"],
            "source_sha256": sources,
            "data_receipt_sha256": campaign["data_receipt_sha256"],
            "environment": shared.ENVIRONMENT,
            "checks": dict.fromkeys(shared.REQUIRED_CHECKS, True),
            "evidence": {"cpu_test_exit_code": 0, "gpu_commands": commands, "real_data_smokes": smokes, **probes},
        },
    )


def process_boundary(monkeypatch, campaign, plan, *, fault=None, node=None):
    events = []
    stages = plan["binding"]["stages"]
    trials = {trial["out"]: trial for trial in campaign["trials"]}
    actual_output = subprocess.check_output

    def check_output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return (node or plan["binding"]["node"]) + "\n"
        return actual_output(command, **kwargs)

    class Child:
        def __init__(self, command, **kwargs):
            self.command, self.kwargs, self.returncode = command, kwargs, None
            self.outer = command[1] == str(Path(sequence.sequence.allocation.__file__).resolve())
            self.pid = 10000 + len(events)
            if self.outer:
                self.index = sum(event["kind"] == "wrapper" for event in events)
                self.stage = stages[self.index]
                assert command == sequence.sequence.wrapper_command(self.stage)
                assert kwargs["cwd"] == Path(campaign["source"])
                assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["start_new_session"] is True
                if self.index:
                    previous = short.read(Path(stages[0]["allocation_state"]))
                    assert previous["status"] == "complete" and previous["keeper_restore"] == 0
            else:
                self.trial = trials[command[command.index("--out") + 1]]
                assert command == shared.trial_command(
                    Path(self.trial["config"]),
                    Path(campaign["data"]),
                    Path(campaign["data_receipt"]),
                    Path(self.trial["out"]),
                )
                assert kwargs["env"][shared.MARKER] == "offline-owned-marker-1"
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
                assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["stderr"] == subprocess.STDOUT
                assert not Path(self.trial["out"]).exists()
                kwargs["stdout"].write("offline full trial joined\n")
            self.event = {"kind": "wrapper" if self.outer else "trial", "joined": False}
            events.append(self.event)

        def wait(self):
            if self.returncode is not None:
                return self.returncode
            if not self.outer:
                endpoint(campaign, self.trial, steps=short.STEPS if fault == "truncated" else None)
                self.returncode = 0
            else:
                stage = self.stage
                if self.index == 0:
                    validation_receipt(campaign, stage)
                    if fault == "gate":
                        rewrite(Path(stage["receipt"]), lambda r: r["checks"].update(go_muon_gpu_numerics=False))
                    if fault == "pre_refresh":
                        name = next(name for name in stage["config_sha256"] if name.startswith("go_muon"))
                        smoke = short.read(Path(stage["receipt"]))["evidence"]["real_data_smokes"][name]
                        path = (
                            Path(smoke["out"]).with_name(f"{name}_resumed")
                            / "checkpoints"
                            / f"step_{smoke['split_steps'] + 1:06d}"
                            / "rank_7.pt"
                        )
                        record = torch.load(path, weights_only=False)
                        record["model"]["weight"] += 0.25
                        torch.save(record, path)
                else:
                    with monkeypatch.context() as environment:
                        for key, value in sequence.sequence.allocation.ENVIRONMENT.items():
                            environment.setenv(key, value)
                        environment.setenv(shared.MARKER, "offline-owned-marker-1")
                        try:
                            execution.run_trials(
                                Path(campaign["root"]),
                                stage["trials"],
                                [Path(path) for path in stage["validation_receipts"]],
                                Path(stage["worker_state"]),
                            )
                        except ValueError:
                            if fault != "truncated":
                                raise
                            self.returncode = 9
                    if fault == "worker":
                        rewrite(Path(stage["worker_state"]), lambda r: r["children"][0].update(exit_code=1))
                    elif fault == "history":
                        directory = Path(
                            next(out for out, trial in trials.items() if trial["name"] == stage["trials"][0])
                        )
                        history = directory / "metrics_from_000000.jsonl"
                        rows = history.read_text().splitlines()
                        del rows[500]
                        history.write_text("\n".join(rows) + "\n")
                    elif fault == "receipt":
                        rewrite(
                            Path(stage["validation_receipts"][0]),
                            lambda receipt: receipt["checks"].update(go_muon_gpu_numerics=False),
                        )
                    elif fault == "prior_restore":
                        rewrite(
                            Path(stages[0]["allocation_state"]),
                            lambda owner: owner.update(keeper_restore=1),
                        )
                self.returncode = 0 if self.returncode is None else self.returncode
                log = Path(stage["allocation_log"])
                log.write_text("offline allocation joined and restored\n")
                owner = {
                    "status": "complete",
                    "command": stage["command"],
                    "log": str(log),
                    "log_sha256": short.sha256(log),
                    "source_sha256": {
                        name: digest
                        for name, digest in campaign["source_sha256"].items()
                        if name.startswith("experiment/")
                    },
                    "exit_code": self.returncode,
                    "cleanup_exit_code": self.returncode,
                    "keeper_restore": 0,
                    "keeper_release_complete": True,
                    "child_marker": f"offline-owned-marker-{self.index}",
                    "keeper": {"pid": 1000 + self.index, "devices": list(range(8))},
                    "restored_keeper": {"pid": 1001 + self.index, "devices": list(range(8))},
                }
                if fault == "nonzero" and self.index == 0:
                    self.returncode = 7
                    owner.update(status="needs_attention", exit_code=7, cleanup_exit_code=7)
                if (fault == "first_restore" and self.index == 0) or (fault == "second_restore" and self.index == 1):
                    owner["keeper_restore"] = False
                if fault == "log" and self.index == 0:
                    owner["log_sha256"] = "0" * 64
                save(Path(stage["allocation_state"]), owner)
                if fault == "source" and self.index == 0:
                    path = Path(campaign["source"]) / "experiment/train.py"
                    path.write_text(path.read_text() + "\n# changed during stage\n")
                if fault == "runs_alias" and self.index == 0:
                    runs = full.campaign_path(Path(campaign["root"])).parent / "runs"
                    runs.symlink_to(Path(campaign["source"]), target_is_directory=True)
            self.event["joined"] = True
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(subprocess, "Popen", Child)
    return events


def test_prepare_four_queues_covers_twelve_unique_full_trials_without_launch(prepared):
    root, source, _, campaign = prepared
    before = {str(path): short.sha256(path) for path in source.rglob("*") if path.is_file()}
    names, plans = [], []
    nodes = short.read(source / "experiment/cluster.json")["nodes"]
    for wave, node in zip(campaign["waves"], nodes, strict=True):
        plan = sequence.prepare_plan(
            root, specifications(root, wave), root / "preparation" / f"{wave['wave_name']}.plan.json", node=node
        )
        plans.append(plan)
        names += plan["binding"]["stages"][1]["trials"]
        assert sequence.require_plan(Path(plan["path"])) == plan
        assert plan["binding"]["campaign_sha256"] == short.sha256(full.campaign_path(root))
        assert plan["binding"]["data_files"] == campaign["data_files"]
    assert len(names) == len(set(names)) == 12
    assert set(names) == {trial["name"] for trial in campaign["trials"]}
    assert [len(plan["binding"]["stages"][1]["trials"]) for plan in plans] == [4, 2, 4, 2]
    assert {str(path): short.sha256(path) for path in source.rglob("*") if path.is_file()} == before
    assert not (full.campaign_path(root).parent / "runs").exists()
    assert not (root / "validation").exists()


@pytest.mark.parametrize("profile", ["controlled", "paper_adapters"])
def test_sequence_joins_validation_then_real_worker_under_fresh_restored_owner(prepared, monkeypatch, profile):
    root, _, _, campaign = prepared
    wave = next(wave for wave in campaign["waves"] if wave["profile"] == profile and wave["seed"] == 1)
    path = root / "preparation" / "queue.json"
    plan = sequence.prepare_plan(root, specifications(root, wave), path, node="28.216.17.9")
    events = process_boundary(monkeypatch, campaign, plan)
    result = sequence.run_sequence(path)
    assert short.read(Path(plan["state"])) == result
    assert result["status"] == "complete" and result["completed_stages"] == 2
    assert all(event["joined"] for event in events)
    assert [event["kind"] for event in events] == (
        ["wrapper", "wrapper"] + ["trial"] * (4 if profile == "controlled" else 2)
    )
    assert all(stage["allocation"]["keeper_restore"] == 0 for stage in result["stages"])
    assert result["stages"][0]["validation"]["wave_name"] == wave["wave_name"]
    worker = short.read(Path(plan["binding"]["stages"][1]["worker_state"]))
    assert worker["completed_trials"] == (4 if profile == "controlled" else 2)
    assert all(child["completion"]["observed_steps"] == 5120 for child in worker["children"])
    assert result["stages"][1]["worker_state_sha256"] == short.sha256(Path(plan["binding"]["stages"][1]["worker_state"]))


@pytest.mark.parametrize("fault", ["nonzero", "first_restore", "log", "gate", "pre_refresh", "source"])
def test_invalid_validation_or_first_restore_halts_before_full_owner(planned, monkeypatch, fault):
    _, _, _, campaign, path, plan = planned
    events = process_boundary(monkeypatch, campaign, plan, fault=fault)
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        sequence.run_sequence(path)
    result = short.read(Path(plan["state"]))
    assert result["status"] == "needs_attention" and result["completed_stages"] == 0
    assert events == [{"kind": "wrapper", "joined": True}]
    assert result["stages"][0]["status"] == "needs_attention"
    assert Path(plan["binding"]["stages"][0]["allocation_log"]).is_file()
    assert not Path(plan["binding"]["stages"][1]["allocation_state"]).exists()
    assert not Path(plan["binding"]["stages"][1]["worker_state"]).exists()
    assert not (full.campaign_path(Path(campaign["root"])).parent / "runs").exists()
    with pytest.raises((FileExistsError, ValueError)):
        sequence.run_sequence(path)


@pytest.mark.parametrize("fault", ["second_restore", "worker", "history", "truncated", "receipt", "prior_restore"])
def test_full_worker_or_second_restore_failure_withholds_queue_completion(planned, monkeypatch, fault):
    _, _, _, campaign, path, plan = planned
    events = process_boundary(monkeypatch, campaign, plan, fault=fault)
    with pytest.raises((RuntimeError, ValueError)):
        sequence.run_sequence(path)
    result = short.read(Path(plan["state"]))
    assert result["status"] == "needs_attention" and result["completed_stages"] == 1
    assert result["stages"][0]["status"] == "complete"
    assert result["stages"][1]["status"] == "needs_attention"
    assert all(event["joined"] for event in events)
    assert Path(plan["binding"]["stages"][1]["worker_state"]).is_file()
    assert Path(plan["binding"]["stages"][1]["allocation_log"]).is_file()
    with pytest.raises(FileExistsError):
        sequence.run_sequence(path)


@pytest.mark.parametrize("damage", ["full_config", "source", "data", "short_history", "plan", "helper"])
def test_changed_binding_refuses_before_any_owner(planned, monkeypatch, damage):
    root, source, short_campaign, campaign, path, plan = planned
    if damage == "full_config":
        rewrite(Path(campaign["trials"][0]["config"]), lambda c: c["training"].update(seed=0))
    elif damage == "source":
        source.joinpath("experiment/train.py").write_text("# replaced trainer\n")
    elif damage == "data":
        (root / "data/dclm/tiny.bin").write_bytes(b"\x01\x00\x00\x00")
    elif damage == "short_history":
        history = Path(short_campaign["trials"][0]["out"]) / "metrics_from_000000.jsonl"
        history.write_text(history.read_text().replace('"train_loss": 4.0', '"train_loss": 4.1'))
    elif damage == "plan":
        rewrite(path, lambda p: p["binding"].update(campaign_sha256="0" * 64))
    else:
        rewrite(path, lambda p: p["binding"]["helpers_sha256"].update({"related_full_execution.py": "0" * 64}))
    events = process_boundary(monkeypatch, campaign, plan)
    with pytest.raises(ValueError):
        sequence.run_sequence(path)
    assert events == []
    assert not Path(plan["state"]).exists()


@pytest.mark.parametrize("existing", ["worker", "later_out", "later_log", "owner", "temporary_state"])
def test_existing_later_evidence_refuses_before_validation_and_preserves_content(planned, monkeypatch, existing):
    _, _, _, campaign, path, plan = planned
    stage = plan["binding"]["stages"][1]
    target = {
        "worker": Path(stage["worker_state"]),
        "later_out": Path(stage["outputs"][-2]),
        "later_log": Path(stage["outputs"][-1]),
        "owner": Path(stage["allocation_state"]),
        "temporary_state": Path(plan["state"]).with_suffix(".json.tmp"),
    }[existing]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("preserved evidence\n")
    events = process_boundary(monkeypatch, campaign, plan)
    with pytest.raises(FileExistsError):
        sequence.run_sequence(path)
    assert target.read_text() == "preserved evidence\n"
    assert events == []
    assert not Path(plan["state"]).exists()


@pytest.mark.parametrize("overlap", ["validation_work", "short_run", "state_alias", "temporary_alias", "source"])
def test_prepare_rejects_overlapping_outputs_and_protected_artifacts_without_plan(prepared, overlap):
    root, source, short_campaign, campaign = prepared
    specs = specifications(root, campaign["waves"][0])
    path = root / "preparation" / "queue.json"
    if overlap == "validation_work":
        specs[1]["worker_state"] = str(Path(specs[0]["work"]) / "worker.json")
    elif overlap == "short_run":
        specs[0]["allocation_log"] = str(Path(short_campaign["trials"][0]["out"]) / "owner.log")
    elif overlap == "state_alias":
        specs[1]["allocation_state"] = str(path.with_suffix(".state.json"))
    elif overlap == "temporary_alias":
        specs[1]["allocation_log"] = str(Path(specs[0]["allocation_log"]).with_suffix(".log.tmp"))
    else:
        specs[0]["allocation_log"] = str(source / "owner.log")
    with pytest.raises(ValueError):
        sequence.prepare_plan(root, specs, path, node="28.216.17.9")
    assert not path.exists()
    assert not (source / "owner.log").exists()


def test_paper_queue_cannot_duplicate_shared_full_trials(prepared):
    root, _, _, campaign = prepared
    wave = next(wave for wave in campaign["waves"] if wave["profile"] == "paper_adapters")
    specs = specifications(root, wave)
    specs[1]["trials"] = wave["trials"]
    path = root / "preparation" / "queue.json"
    with pytest.raises(ValueError):
        sequence.prepare_plan(root, specs, path, node="28.216.17.9")
    assert not path.exists()


def test_prepare_preserves_dangling_output_symlink_instead_of_writing_its_target(prepared):
    root, _, _, campaign = prepared
    specs = specifications(root, campaign["waves"][0])
    output = Path(specs[0]["allocation_log"])
    output.parent.mkdir(parents=True)
    target = root / "missing-output.log"
    output.symlink_to(target)
    path = root / "preparation" / "queue.json"
    with pytest.raises(FileExistsError):
        sequence.prepare_plan(root, specs, path, node="28.216.17.9")
    assert output.is_symlink()
    assert not target.exists()
    assert not path.exists()


@pytest.mark.parametrize("destination", ["source", "data", "short", "cache", "benign"])
def test_runs_parent_symlink_refuses_preparation_without_touching_target(prepared, destination):
    root, source, short_campaign, campaign = prepared
    targets = {
        "source": source,
        "data": root / "data",
        "short": Path(short_campaign["trials"][0]["out"]).parent,
        "cache": root / "run-preparation/related-verified-cache",
        "benign": root / "otherwise-unused-output",
    }
    target = targets[destination]
    target.mkdir(parents=True, exist_ok=True)
    before = {str(path): short.sha256(path) for path in target.rglob("*") if path.is_file()}
    runs = full.campaign_path(root).parent / "runs"
    runs.symlink_to(target, target_is_directory=True)
    plan_path = root / "preparation" / "queue.json"
    with pytest.raises(ValueError):
        sequence.prepare_plan(root, specifications(root, campaign["waves"][0]), plan_path, node="28.216.17.9")
    assert runs.is_symlink()
    assert not plan_path.exists()
    assert {str(path): short.sha256(path) for path in target.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("destination", ["source", "data", "short"])
def test_runs_parent_redirected_after_preparation_refuses_before_validation(planned, monkeypatch, destination):
    root, source, short_campaign, campaign, path, plan = planned
    target = {
        "source": source,
        "data": root / "data",
        "short": Path(short_campaign["trials"][0]["out"]).parent,
    }[destination]
    before = {str(item): short.sha256(item) for item in target.rglob("*") if item.is_file()}
    runs = full.campaign_path(root).parent / "runs"
    runs.symlink_to(target, target_is_directory=True)
    events = process_boundary(monkeypatch, campaign, plan)
    with pytest.raises(ValueError):
        sequence.run_sequence(path)
    assert events == []
    assert not Path(plan["state"]).exists()
    assert {str(item): short.sha256(item) for item in target.rglob("*") if item.is_file()} == before


def test_runs_parent_redirected_during_validation_halts_before_full_owner(planned, monkeypatch):
    _, source, _, campaign, path, plan = planned
    before = {str(item): short.sha256(item) for item in source.rglob("*") if item.is_file()}
    events = process_boundary(monkeypatch, campaign, plan, fault="runs_alias")
    with pytest.raises(ValueError):
        sequence.run_sequence(path)
    assert events == [{"kind": "wrapper", "joined": True}]
    assert short.read(Path(plan["state"]))["status"] == "needs_attention"
    assert not Path(plan["binding"]["stages"][1]["allocation_state"]).exists()
    assert not Path(plan["binding"]["stages"][1]["worker_state"]).exists()
    assert {str(item): short.sha256(item) for item in source.rglob("*") if item.is_file()} == before


def test_existing_parent_alias_retargeting_refuses_bound_validation_outputs(prepared, monkeypatch):
    root, _, _, campaign = prepared
    target = root / "safe-validation-parent"
    target.mkdir()
    alias = root / "validation-alias"
    alias.symlink_to(target, target_is_directory=True)
    specs = specifications(root, campaign["waves"][0])
    specs[0]["work"] = str(alias / "work")
    path = root / "preparation" / "queue.json"
    plan = sequence.prepare_plan(root, specs, path, node="28.216.17.9")
    assert sequence.require_plan(path) == plan
    target.rmdir()
    target.symlink_to(root / "data", target_is_directory=True)
    before = {str(item): short.sha256(item) for item in (root / "data").rglob("*") if item.is_file()}
    events = process_boundary(monkeypatch, campaign, plan)
    with pytest.raises(ValueError):
        sequence.run_sequence(path)
    assert events == []
    assert not (root / "data/work").exists()
    assert not Path(plan["state"]).exists()
    assert {str(item): short.sha256(item) for item in (root / "data").rglob("*") if item.is_file()} == before


@pytest.mark.parametrize("fault", ["node", "marker"])
def test_wrong_node_or_inherited_owner_refuses_before_wrappers(planned, monkeypatch, fault):
    _, _, _, campaign, path, plan = planned
    events = process_boundary(monkeypatch, campaign, plan, node="28.216.17.80" if fault == "node" else None)
    if fault == "marker":
        monkeypatch.setenv(shared.MARKER, "already-owned")
    with pytest.raises(RuntimeError):
        sequence.run_sequence(path)
    assert events == []
    assert not Path(plan["state"]).exists()


def test_cli_prepare_and_verify_emit_launch_commands_without_compute(prepared, monkeypatch, capsys):
    root, _, _, campaign = prepared
    specs_path, path = root / "specifications.json", root / "preparation" / "queue.json"
    save(specs_path, specifications(root, campaign["waves"][0]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "related_full_sequence.py",
            "--prepare",
            str(specs_path),
            "--root",
            str(root),
            "--node",
            "28.216.17.9",
            "--plan",
            str(path),
        ],
    )
    sequence.main()
    prepared_result = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(sys, "argv", ["related_full_sequence.py", "--plan", str(path)])
    sequence.main()
    verified = json.loads(capsys.readouterr().out)
    assert prepared_result["status"] == "prepared" and verified["status"] == "verified"
    assert prepared_result["launch_command"] == verified["launch_command"]
    assert verified["launch_requested"] is False
    assert verified["launch_command"][-1] == "--launch"
    assert verified["controller_environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert verified["allocation_environment"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
    assert not Path(short.read(path)["state"]).exists()
    assert not (root / "validation").exists()
