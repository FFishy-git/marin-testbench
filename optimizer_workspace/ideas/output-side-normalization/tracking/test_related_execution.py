# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only launch acceptance with real temporary receipts and an OS-boundary fake."""

import json
import os
import pickle
import shutil
from pathlib import Path

import pytest
import related_execution as execution
import torch
from experiment import data_access
from experiment.records import compare_continuation
from related_campaign import MOUSSE_REFERENCE, PROFILES, TASK, WORLD, config_identity, prepare, read, sha256


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def checkpoint(directory, step, config, campaign):
    path = directory / "checkpoints" / f"step_{step:06d}"
    save(path / "complete.json", {"step": step, "world": WORLD})
    for rank in range(WORLD):
        state = {
            "step": step,
            "next_global_sequence": step * config["training"]["global_batch_sequences"],
            "config": config,
            "rank": rank,
            "world": WORLD,
            "identity": {
                "schema": 1,
                "source_sha256": campaign["source_sha256"],
                "config_sha256": config_identity(config),
                "data_receipt_sha256": campaign["data_receipt_sha256"],
            },
            "optimizers": [{"momentum": torch.tensor([step, rank], dtype=torch.float32)}],
            "fast": {"cached_root": torch.eye(2), "root_step": step},
            "output": None,
            "torch_rng": torch.tensor([rank, 1, 2], dtype=torch.uint8),
            "cuda_rng": None,
            "python_rng": (3, (rank, step), None),
        }
        if rank == 0:
            state["model"] = {"weight": torch.tensor([[step, 0], [1, 2]], dtype=torch.float32)}
        torch.save(state, path / f"rank_{rank}.pt")


def run_manifest(campaign, config, *, steps, status, smoke):
    return {
        "status": status,
        "completed_steps": steps,
        "execution_check": smoke,
        "config": config,
        "data": campaign["data"],
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "source_revision": campaign["source_revision"],
        "source_sha256": campaign["source_sha256"],
        "world_size": WORLD,
    }


def write_smoke(campaign, config, directory, step):
    save(
        directory / "run.json",
        run_manifest(campaign, config, steps=step, status="execution_check_complete", smoke=True),
    )
    checkpoint(directory, step, config, campaign)


def write_completed_trial(campaign, config, directory):
    steps, batch = config["training"]["steps"], config["training"]["batch_tokens"]
    save(directory / "run.json", run_manifest(campaign, config, steps=steps, status="complete", smoke=False))
    weights = config["data"]["mixture_weights"]
    losses = {f"heldout/{name}": 3.0 for name in weights}
    selection = sum(weight * losses[f"heldout/{name}"] for name, weight in weights.items()) / sum(weights.values())
    rows = [
        {
            "step": step,
            "tokens": step * batch,
            "train_loss": 4.0,
            "step_seconds": 1.0,
            "elapsed_seconds": step + 10,
            "end_to_end_seconds": step + 11,
        }
        for step in range(1, steps + 1)
    ]
    rows[-1].update(heldout_losses=losses, selection_loss=selection, c4_loss=3.1)
    (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint(directory, steps, config, campaign)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "output"
    for folder, _ in PROFILES.values():
        shutil.copytree(TASK / "experiment/comparisons" / folder, source / "experiment/comparisons" / folder)
    shutil.copyfile(TASK / "experiment/cluster.json", source / "experiment/cluster.json")
    (source / "experiment/train.py").write_text("# temporary offline source identity\n")
    revision = read(TASK / "experiment/upstream_manifest.json")["revision"]
    upstream = source / "artifacts/external-sources" / f"Toner-{revision}"
    for name in ("reproduce/marin/train.py", "src/toner/fast/muon.py"):
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# temporary executed source identity\n")
    components = ("dclm", "starcoder", "proofpile", "paloma/c4_en")
    plan_path = upstream / "reproduce/marin/data_plan.json"
    save(plan_path, {"files": [{"component": name, "file": "tiny.bin", "tokens": 1, "docs": 0} for name in components]})
    upstream_files = ("reproduce/marin/train.py", "src/toner/fast/muon.py", "reproduce/marin/data_plan.json")
    save(
        source / "experiment/upstream_manifest.json",
        {"revision": revision, "files": {name: sha256(upstream / name) for name in upstream_files}},
    )
    oracle = source / MOUSSE_REFERENCE
    oracle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TASK / MOUSSE_REFERENCE, oracle)
    data = root / "data"
    records = []
    for component in components:
        path = data / component / "tiny.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00\x00\x00\x00")
        records.append(
            {
                "file": f"{component}/tiny.bin",
                "tokens": 1,
                "docs": 0,
                "bytes": 4,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    save(data / "experiment_receipt.json", {"root": str(data), "data_plan_sha256": sha256(plan_path), "files": records})
    monkeypatch.setattr(data_access, "PLAN", plan_path)
    campaign = prepare(root, source=source)
    monkeypatch.setenv(execution.MARKER, "inherited-allocation-token")
    for key, value in execution.ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    return root, source, campaign


def make_validation(campaign, wave):
    work = Path(campaign["root"]) / "validation" / wave["wave_name"]
    work.mkdir(parents=True)
    executions, smokes = [], {}

    def command_record(command):
        log = work / f"command_{len(executions)}.log"
        log.write_text("temporary successful command log\n")
        executions.append({"command": command, "exit_code": 0, "log": str(log), "log_sha256": sha256(log)})

    sources = {name: digest for name, digest in campaign["source_sha256"].items() if name.startswith("experiment/")}
    probes = {}
    for key, module, checks, world_key in (
        ("gpu_receipt", "experiment.gpu_checks", execution.SYNTHETIC_CHECKS, "world"),
        ("related_gpu_receipt", "experiment.related_gpu_checks", execution.RELATED_CHECKS, "world_size"),
    ):
        path = work / f"{module.rsplit('.', 1)[-1]}.json"
        probe_checks = dict.fromkeys(checks, True)
        if key == "gpu_receipt":
            probe_checks["synthetic_compiled_body"] = True
        save(path, {"source_sha256": sources, "checks": probe_checks, world_key: 2})
        command_record([*execution.launcher(execution.PYTHON, 2), module, "--receipt", str(path)])
        probes[key] = str(path)
    trials = {trial["name"]: trial for trial in campaign["trials"]}
    config_hashes = {}
    for name in wave["trials"]:
        trial = trials[name]
        alias = next(item for item in trial["aliases"] if item["wave"] == wave["wave_name"])
        config_path, config = Path(alias["config"]), read(Path(alias["config"]))
        complete, split = execution.continuation_steps(config)
        full, resumed = work / f"{name}_continuous", work / f"{name}_resumed"
        common = execution.trial_command(config_path, Path(campaign["data"]), Path(campaign["data_receipt"]), full)[:-2]
        command_record([*common, "--out", str(full), "--max-steps", str(complete)])
        command_record([*common, "--out", str(resumed), "--max-steps", str(split)])
        write_smoke(campaign, config, full, complete)
        checkpoint(resumed, split, config, campaign)
        resume_step, pre_refresh = split, None
        if trial["arm"] in execution.RELATED_ARMS and trial["arm"] != "muon_control":
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
            write_smoke(campaign, config, prefix, resume_step)
            checkpoint(resumed, resume_step, config, campaign)
            pre_refresh = compare_continuation(
                prefix / "checkpoints" / f"step_{resume_step:06d}",
                resumed / "checkpoints" / f"step_{resume_step:06d}",
                WORLD,
            )
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
        write_smoke(campaign, config, resumed, complete)
        smokes[name] = {
            "out": str(full),
            "completed_steps": complete,
            "split_steps": split,
            "continuation": compare_continuation(
                full / "checkpoints" / f"step_{complete:06d}",
                resumed / "checkpoints" / f"step_{complete:06d}",
                WORLD,
            ),
            "pre_refresh_continuation": pre_refresh,
        }
        config_hashes[name] = alias["config_sha256"]
    path = work / "validation.json"
    save(
        path,
        {
            "wave_name": wave["wave_name"],
            "source_sha256": sources,
            "config_sha256": config_hashes,
            "environment": execution.ENVIRONMENT,
            "data_receipt_sha256": campaign["data_receipt_sha256"],
            "checks": dict.fromkeys(execution.REQUIRED_CHECKS, True),
            "evidence": {"cpu_test_exit_code": 0, "gpu_commands": executions, "real_data_smokes": smokes, **probes},
        },
    )
    return path


@pytest.fixture
def validated(prepared):
    root, source, campaign = prepared
    wave = next(item for item in campaign["waves"] if item["profile"] == "controlled" and item["lr_index"] == 1)
    return root, source, campaign, make_validation(campaign, wave), wave["trials"]


def install_process_boundary(monkeypatch, campaign, *, mode="complete", mutate=None):
    events = []

    class Child:
        def __init__(self, command, **kwargs):
            assert not events or events[-1][0] == "joined"
            self.command, self.kwargs = command, kwargs
            self.pid = 40000 + len(events)
            events.append(("started", command))
            assert kwargs["cwd"] == campaign["source"]
            assert kwargs["env"][execution.MARKER] == "inherited-allocation-token"
            assert kwargs["stdin"] == execution.subprocess.DEVNULL
            assert kwargs["stderr"] == execution.subprocess.STDOUT
            kwargs["stdout"].write("temporary child output\n")

        def wait(self):
            out = Path(self.command[self.command.index("--out") + 1])
            config = read(Path(self.command[self.command.index("--config") + 1]))
            if mode == "complete":
                write_completed_trial(campaign, config, out)
            elif mode == "manifest_only":
                save(
                    out / "run.json",
                    run_manifest(campaign, config, steps=config["training"]["steps"], status="complete", smoke=False),
                )
            if mutate:
                mutate()
            events.append(("joined", self.command))
            return 9 if mode == "failed" else 0

    monkeypatch.setattr(execution.subprocess, "Popen", Child)
    return events


def test_preflight_is_read_only_and_shared_trial_accepts_paper_wave_alias(prepared):
    root, source, campaign = prepared
    wave = next(item for item in campaign["waves"] if item["profile"] == "paper_adapters" and item["lr_index"] == 1)
    receipt = make_validation(campaign, wave)
    before = {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}
    name = wave["trials"][0]
    plan = execution.validate_trials(root, [name], [receipt])
    assert plan["trials"][0]["validation_wave"] == wave["wave_name"]
    assert plan["trials"][0]["config"].startswith(str(source / "experiment/comparisons/related_paper_short"))
    assert {str(path): sha256(path) for path in root.rglob("*") if path.is_file()} == before
    assert not Path(plan["trials"][0]["out"]).exists()


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "extra_source",
        "config_hash",
        "wave",
        "environment",
        "gate",
        "gpu_exit",
        "gpu_log",
        "gpu_commands_missing",
        "probe_source",
        "data_hash",
        "actual_data",
        "smoke_steps",
        "continuation",
        "pre_refresh",
        "smoke_manifest",
        "pre_refresh_checkpoint",
        "smoke_command",
    ],
)
def test_invalid_validation_refuses_before_any_subprocess_or_execution_evidence(validated, monkeypatch, damage):
    root, source, campaign, path, names = validated
    receipt = read(path)
    name = names[2]
    smoke = receipt["evidence"]["real_data_smokes"][name]
    if damage == "source":
        (source / "experiment/train.py").write_text("# changed source\n")
    elif damage == "extra_source":
        (source / "experiment/uncounted.py").write_text("# added source\n")
    elif damage == "config_hash":
        receipt["config_sha256"][name] = "0" * 64
    elif damage == "wave":
        receipt["wave_name"] = "unbound-wave"
    elif damage == "environment":
        receipt["environment"] = {}
    elif damage == "gate":
        receipt["checks"]["go_muon_gpu_numerics"] = 1
    elif damage == "gpu_exit":
        receipt["evidence"]["gpu_commands"][0]["exit_code"] = 7
    elif damage == "gpu_log":
        Path(receipt["evidence"]["gpu_commands"][0]["log"]).write_text("changed log\n")
    elif damage == "gpu_commands_missing":
        receipt["evidence"]["gpu_commands"] = []
    elif damage == "probe_source":
        probe = Path(receipt["evidence"]["related_gpu_receipt"])
        record = read(probe)
        record["source_sha256"] = {}
        save(probe, record)
    elif damage == "data_hash":
        receipt["data_receipt_sha256"] = "0" * 64
    elif damage == "actual_data":
        (Path(campaign["data"]) / "dclm/tiny.bin").write_bytes(b"changed data")
    elif damage == "smoke_steps":
        smoke["completed_steps"] -= 1
    elif damage == "continuation":
        smoke["continuation"]["tensor_comparisons"] = 0
    elif damage == "pre_refresh":
        smoke["pre_refresh_continuation"] = None
    elif damage == "smoke_manifest":
        manifest = Path(smoke["out"]) / "run.json"
        record = read(manifest)
        record["completed_steps"] -= 1
        save(manifest, record)
    elif damage == "pre_refresh_checkpoint":
        (Path(smoke["out"]).with_name(f"{name}_resumed") / "checkpoints/step_000006/rank_7.pt").unlink()
    else:
        receipt["evidence"]["gpu_commands"] = [
            item
            for item in receipt["evidence"]["gpu_commands"]
            if not (name in " ".join(item["command"]) and "--resume" in item["command"])
        ]
    save(path, receipt)
    events = install_process_boundary(monkeypatch, campaign)
    state = root / "worker.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        execution.run_trials(root, [name], [path], state)
    assert events == []
    assert not state.exists()
    assert not Path(next(trial["out"] for trial in campaign["trials"] if trial["name"] == name)).exists()


@pytest.mark.parametrize("pair", ["final", "pre_refresh"])
@pytest.mark.parametrize("damage", ["unreadable", "tensor_mismatch"])
def test_damaged_nonempty_smoke_checkpoint_invalidates_cached_acceptance_before_start(
    validated, monkeypatch, pair, damage
):
    root, _, campaign, path, names = validated
    name = names[2]
    execution.validate_trials(root, [name], [path])
    smoke = read(path)["evidence"]["real_data_smokes"][name]
    directory = Path(smoke["out"]).with_name(f"{name}_resumed")
    step = smoke["completed_steps"] if pair == "final" else smoke["split_steps"] + 1
    rank_path = directory / "checkpoints" / f"step_{step:06d}" / "rank_7.pt"
    before = rank_path.stat()
    if damage == "unreadable":
        rank_path.write_bytes(b"corrupt nonempty checkpoint")
    else:
        state = torch.load(rank_path, map_location="cpu", weights_only=False)
        state["optimizers"][0]["momentum"].add_(1)
        torch.save(state, rank_path)
        os.utime(rank_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert rank_path.stat().st_mtime_ns == before.st_mtime_ns
    assert rank_path.stat().st_size > 0
    events = install_process_boundary(monkeypatch, campaign)
    state_path = root / "worker.json"
    with pytest.raises((ValueError, AssertionError, RuntimeError, pickle.UnpicklingError, EOFError)):
        execution.run_trials(root, [name], [path], state_path)
    assert events == []
    assert not state_path.exists()
    assert not Path(next(trial["out"] for trial in campaign["trials"] if trial["name"] == name)).exists()


def test_false_secondary_synthetic_compiled_body_refuses_before_start(validated, monkeypatch):
    root, _, campaign, path, names = validated
    probe_path = Path(read(path)["evidence"]["gpu_receipt"])
    probe = read(probe_path)
    probe["checks"]["synthetic_compiled_body"] = False
    save(probe_path, probe)
    events = install_process_boundary(monkeypatch, campaign)
    state_path = root / "worker.json"
    with pytest.raises(ValueError):
        execution.run_trials(root, names[:1], [path], state_path)
    assert events == []
    assert not state_path.exists()
    assert not Path(next(trial["out"] for trial in campaign["trials"] if trial["name"] == names[0])).exists()


@pytest.mark.parametrize("summary", ["continuation", "pre_refresh_continuation"])
@pytest.mark.parametrize("field", ["tensor_comparisons", "max_absolute_error"])
def test_changed_continuation_summary_cannot_reuse_cached_tensor_acceptance(validated, monkeypatch, summary, field):
    root, _, campaign, path, names = validated
    name = names[2]
    execution.validate_trials(root, [name], [path])
    receipt = read(path)
    comparison = receipt["evidence"]["real_data_smokes"][name][summary]
    comparison[field] = comparison[field] + 1 if field == "tensor_comparisons" else 1e100
    save(path, receipt)
    events = install_process_boundary(monkeypatch, campaign)
    state_path = root / "worker.json"
    with pytest.raises(ValueError):
        execution.run_trials(root, [name], [path], state_path)
    assert events == []
    assert not state_path.exists()


@pytest.mark.parametrize("artifact", ["split_rank", "checkpoint_receipt", "smoke_manifest", "matched_pair"])
def test_changed_smoke_artifact_during_child_blocks_next_trial(validated, monkeypatch, artifact):
    root, _, campaign, path, names = validated
    name = names[2]
    smoke = read(path)["evidence"]["real_data_smokes"][name]
    full = Path(smoke["out"])
    resumed = full.with_name(f"{name}_resumed")

    def mutate():
        if artifact == "smoke_manifest":
            manifest_path = full / "run.json"
            manifest = read(manifest_path)
            manifest["hostname"] = "changed-offline-fixture"
            save(manifest_path, manifest)
        elif artifact == "checkpoint_receipt":
            receipt_path = resumed / "checkpoints" / f"step_{smoke['split_steps']:06d}" / "complete.json"
            receipt = read(receipt_path)
            receipt["time"] = "changed-offline-fixture"
            save(receipt_path, receipt)
        else:
            directories = [resumed] if artifact == "split_rank" else [full, resumed]
            step = smoke["split_steps"] if artifact == "split_rank" else smoke["completed_steps"]
            for directory in directories:
                rank_path = directory / "checkpoints" / f"step_{step:06d}" / "rank_7.pt"
                state = torch.load(rank_path, map_location="cpu", weights_only=False)
                state["optimizers"][0]["momentum"].add_(1)
                torch.save(state, rank_path)

    events = install_process_boundary(monkeypatch, campaign, mutate=mutate)
    state_path = root / "worker.json"
    with pytest.raises(ValueError):
        execution.run_trials(root, names[:2], [path], state_path)
    assert [event[0] for event in events] == ["started", "joined"]
    result = read(state_path)
    assert result["status"] == "needs_attention"
    assert result["completed_trials"] == 0
    second = next(trial for trial in campaign["trials"] if trial["name"] == names[1])
    assert not Path(second["out"]).exists()


def test_uncovered_actual_lr_trial_never_inherits_another_trials_gates(validated):
    root, _, _, path, names = validated
    with pytest.raises(ValueError, match="No passing exact-config validation"):
        execution.validate_trials(root, [names[2].replace("lr1", "lr2")], [path])


def test_worker_requires_inherited_allocation_marker_before_compute(validated, monkeypatch):
    root, _, campaign, path, names = validated
    monkeypatch.delenv(execution.MARKER)
    events = install_process_boundary(monkeypatch, campaign)
    state = root / "worker.json"
    with pytest.raises(RuntimeError, match="allocation wrapper"):
        execution.run_trials(root, names[:1], [path], state)
    assert events == []
    assert not state.exists()


def test_worker_joins_trials_sequentially_and_records_verified_completion(validated, monkeypatch):
    root, _, campaign, path, names = validated
    events = install_process_boundary(monkeypatch, campaign)
    state = root / "worker.json"
    result = execution.run_trials(root, names[:2], [path], state)
    assert [event[0] for event in events] == ["started", "joined", "started", "joined"]
    assert result["status"] == "complete"
    assert result["completed_trials"] == 2
    assert read(state) == result
    for child in result["children"]:
        assert child["status"] == "complete"
        assert child["exit_code"] == 0
        assert child["log_sha256"] == sha256(Path(child["log"]))
        assert child["completion"]["complete"]
        assert "--resume" not in child["command"]
        assert "--max-steps" not in child["command"]
    with pytest.raises(FileExistsError):
        execution.run_trials(root, names[:2], [path], root / "retry.json")
    assert len(events) == 4


@pytest.mark.parametrize("mode", ["failed", "manifest_only"])
def test_nonzero_child_or_zero_exit_incomplete_manifest_stops_without_next_trial(validated, monkeypatch, mode):
    root, _, campaign, path, names = validated
    events = install_process_boundary(monkeypatch, campaign, mode=mode)
    state = root / "worker.json"
    with pytest.raises((RuntimeError, ValueError)):
        execution.run_trials(root, names[:2], [path], state)
    result = read(state)
    assert [event[0] for event in events] == ["started", "joined"]
    assert result["status"] == "needs_attention"
    assert result["completed_trials"] == 0
    assert result["children"][0]["exit_code"] == (9 if mode == "failed" else 0)
    assert result["children"][0]["log_sha256"] == sha256(Path(result["children"][0]["log"]))
    second = next(trial for trial in campaign["trials"] if trial["name"] == names[1])
    assert not Path(second["out"]).exists()


def test_validation_changed_during_child_is_preserved_and_next_trial_is_not_started(validated, monkeypatch):
    root, _, campaign, path, names = validated

    def mutate():
        receipt = read(path)
        receipt["checks"]["go_muon_gpu_statistics"] = False
        save(path, receipt)

    events = install_process_boundary(monkeypatch, campaign, mutate=mutate)
    state = root / "worker.json"
    with pytest.raises(ValueError):
        execution.run_trials(root, names[:2], [path], state)
    assert [event[0] for event in events] == ["started", "joined"]
    assert read(state)["status"] == "needs_attention"
    assert read(path)["checks"]["go_muon_gpu_statistics"] is False


def test_preexisting_later_trial_log_refuses_the_entire_worker_without_overwriting(validated, monkeypatch):
    root, _, campaign, path, names = validated
    later = next(trial for trial in campaign["trials"] if trial["name"] == names[1])
    out = Path(later["out"])
    out.parent.mkdir(parents=True)
    log = out.with_name(f"{out.name}.torchrun.log")
    log.write_text("preserve previous evidence\n")
    events = install_process_boundary(monkeypatch, campaign)
    with pytest.raises(FileExistsError):
        execution.run_trials(root, names[:2], [path], root / "worker.json")
    assert events == []
    assert log.read_text() == "preserve previous evidence\n"
    assert not (root / "worker.json").exists()
