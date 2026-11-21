# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU monitor acceptance with real campaign files and fake node-probe I/O."""

import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import monitor_related_full as monitor
import pytest
import related_campaign as short
import related_full_campaign as full

NODE = "28.216.17.9"


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def identity(pid):
    return {
        "pid": pid,
        "status": "S",
        "parent": 1,
        "group": pid,
        "session": pid,
        "start_ticks": "1234",
        "command": ["/cpu/owned.py"],
    }


def owner(tmp_path, *, state=None, receipt=None):
    path = tmp_path / "owner.json"
    save(path, state or {"status": "running", "pid": 123, "generation": identity(123)})
    registration = {"node": NODE, "state": str(path)}
    if receipt:
        registration["receipt"] = str(receipt)
    return [registration]


def node_probe(_node, _pids):
    return {"processes": {"123": identity(123)}, "gpus": "", "compute": ""}


def file_hashes(root):
    return {str(path): short.sha256(path) for path in root.rglob("*") if path.is_file()}


def history(campaign, trial, *, steps=None, state="complete", loss=3.0):
    config = short.read(Path(trial["config"]))
    directory = Path(trial["out"])
    steps = config["training"]["steps"] if steps is None else steps
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
            "train_loss": 4.0 - step / 10000,
            "step_seconds": 1.0,
            "elapsed_seconds": step + 10,
            "end_to_end_seconds": step + 11,
        }
        for step in range(1, steps + 1)
    ]
    if rows and steps == config["training"]["steps"]:
        weights = config["data"]["mixture_weights"]
        losses = {f"heldout/{name}": loss for name in weights}
        selection = sum(weight * losses[f"heldout/{name}"] for name, weight in weights.items()) / sum(weights.values())
        rows[-1].update(selection_loss=selection, heldout_losses=losses, c4_loss=2.0)
    path = directory / "metrics_from_000000.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if state == "complete" and steps == config["training"]["steps"]:
        checkpoint = directory / f"checkpoints/step_{steps:06d}"
        save(checkpoint / "complete.json", {"step": steps, "world": short.WORLD})
        for rank in range(short.WORLD):
            (checkpoint / f"rank_{rank}.pt").write_bytes(b"offline checkpoint receipt fixture")
    return path


@pytest.fixture
def prepared(tmp_path):
    source, root = tmp_path / "source", tmp_path / "output"
    for folder, _ in short.PROFILES.values():
        shutil.copytree(short.TASK / "experiment/comparisons" / folder, source / "experiment/comparisons" / folder)
    for name in full.RECIPE_MODULES:
        shutil.copyfile(short.TASK / name, source / name)
    (source / "experiment/train.py").write_text("# Offline trainer identity\n")
    revision = short.read(short.TASK / "experiment/upstream_manifest.json")["revision"]
    upstream = source / "artifacts/external-sources" / f"Toner-{revision}"
    for name in ("reproduce/marin/train.py", "src/toner/fast/muon.py"):
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Offline upstream identity\n")
    save(upstream / "reproduce/marin/data_plan.json", {"files": []})
    for family in ("muon", "muont"):
        name = f"reproduce/marin/configs/130m/{family}_0p5m.json"
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(short.TASK / upstream.relative_to(source) / name, path)
    save(
        source / "experiment/upstream_manifest.json",
        {
            "revision": revision,
            "files": {
                str(path.relative_to(upstream)): short.sha256(path) for path in upstream.rglob("*") if path.is_file()
            },
        },
    )
    oracle = source / short.MOUSSE_REFERENCE
    oracle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(short.TASK / short.MOUSSE_REFERENCE, oracle)
    shard = root / "data/dclm/tiny.bin"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"\x00\x00\x00\x00")
    save(
        root / "data/experiment_receipt.json",
        {
            "root": str(shard.parents[1]),
            "data_plan_sha256": short.sha256(upstream / "reproduce/marin/data_plan.json"),
            "files": [
                {
                    "file": "dclm/tiny.bin",
                    "bytes": shard.stat().st_size,
                    "mtime_ns": shard.stat().st_mtime_ns,
                    "sha256": short.sha256(shard),
                }
            ],
        },
    )
    campaign = short.prepare(root, source=source)
    for trial in campaign["trials"]:
        preferred = short.ARMS.index(trial["arm"]) % 3
        history(campaign, trial, loss=3.0 + abs(trial["lr_index"] - preferred))
    return root, full.prepare(root)


def test_not_prepared_and_future_owner_remain_observable_without_mutations(tmp_path):
    registrations = [{"node": NODE, "state": str(tmp_path / "future-owner.json")}]
    result = monitor.observe(tmp_path, registrations, inspect_node=node_probe)
    assert result["training"]["status"] == "not_prepared"
    assert result["training"]["planned_trials"] == 12
    assert result["owners"][0]["status"] == "unavailable"
    assert not result["training_complete"]
    assert not result["needs_attention"]
    assert list(tmp_path.iterdir()) == []


def test_full_progress_retains_losses_tokens_and_generation_bound_owner_receipt(prepared, tmp_path):
    root, campaign = prepared
    trial = campaign["trials"][0]
    path = history(campaign, trial, steps=3, state="running")
    receipt = tmp_path / "validation.json"
    save(receipt, {"checks": {"mousse_gpu_numerics": True}, "evidence": {"gpu_commands": [["cpu-fixture"]]}})
    registrations = owner(tmp_path, receipt=receipt)
    before = file_hashes(tmp_path)
    first = monitor.observe(root, registrations, inspect_node=node_probe)
    assert file_hashes(tmp_path) == before
    assert first["training"]["status"] == "running"
    assert first["short_training"]["status"] == "complete"
    assert first["owners"][0]["owner_liveness"] == "live"
    assert first["owners"][0]["checks"]["mousse_gpu_numerics"]
    assert first["owners"][0]["completed_commands"] == 1
    assert first["training"]["planned_trials"] == 12
    assert first["training"]["trials"][0]["observed_steps"] == 3
    assert first["training"]["trials"][0]["observed_tokens"] == 3 * short.BATCH_TOKENS
    assert first["training"]["trials"][0]["latest_train_loss"] == 3.9997
    row = {"step": 4, "tokens": 4 * short.BATCH_TOKENS, "train_loss": 3.8, "selection_loss": 3.2}
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    later = monitor.observe(root, registrations, inspect_node=node_probe)
    progress = later["training"]["trials"][0]
    assert progress["observed_steps"] == 4
    assert progress["planned_steps"] == 5120
    assert progress["latest_train_loss"] == 3.8
    assert progress["latest_selection_loss"] == 3.2
    assert not progress["complete"]
    assert not later["training_complete"]
    assert not later["needs_attention"]


@pytest.mark.parametrize("damage", ["nonfinite", "gap", "identity"])
def test_invalid_full_training_evidence_requires_attention_and_withholds_metrics(prepared, damage):
    root, campaign = prepared
    path = history(campaign, campaign["trials"][0], steps=3, state="running")
    if damage == "identity":
        manifest_path = path.parent / "run.json"
        manifest = short.read(manifest_path)
        manifest["config"]["training"]["seed"] = 0
        save(manifest_path, manifest)
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "gap":
            del rows[1]
        else:
            rows[1]["train_loss"] = float("nan")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = monitor.observe(root, [], inspect_node=node_probe)
    assert result["needs_attention"]
    assert result["training"]["status"] == "needs_attention"
    assert result["training"]["trials"][0]["status"] == "invalid_evidence"
    assert result["training"]["trials"][0]["latest_train_loss"] is None
    assert result["training"]["trials"][0]["reasons"]
    assert not result["training_complete"]
    json.dumps(result, allow_nan=False)


def test_live_partial_jsonl_is_not_completion_and_next_snapshot_recovers(prepared):
    root, campaign = prepared
    path = history(campaign, campaign["trials"][0], steps=3, state="running")
    complete = path.read_text()
    with path.open("a") as stream:
        stream.write('{"step":4')
    partial = monitor.observe(root, [], inspect_node=node_probe)
    assert partial["training"]["trials"][0]["status"] == "running"
    assert partial["training"]["trials"][0]["latest_train_loss"] is None
    assert not partial["training_complete"]
    assert not partial["needs_attention"]
    path.write_text(complete)
    recovered = monitor.observe(root, [], inspect_node=node_probe)
    assert recovered["training"]["trials"][0]["observed_steps"] == 3
    assert recovered["training"]["trials"][0]["latest_train_loss"] == 3.9997


def test_full_completion_is_independent_of_historical_keeper_snapshot(prepared, tmp_path):
    root, campaign = prepared
    for trial in campaign["trials"]:
        history(campaign, trial)
    state = {
        "status": "complete",
        "pid": 123,
        "generation": identity(123),
        "keeper_restore": 0,
        "restored_keeper": {"generation": {"200": identity(200)}, "occupied": [["200", "GPU-physical"]]},
    }
    registrations = owner(tmp_path, state=state)
    result = monitor.observe(root, registrations, inspect_node=node_probe)
    assert result["training"]["status"] == "complete"
    assert result["training"]["completed_trials"] == 12
    assert all(trial["observed_steps"] == 5120 and trial["complete"] for trial in result["training"]["trials"])
    assert result["training_complete"]
    assert not result["owners"][0]["restored_keeper_live"]
    assert not result["historical_owner_snapshot"]["all_registered_owners_complete_and_restored"]
    assert result["historical_owner_snapshot"]["scope"] == "diagnostic_only_not_full_campaign_completion"
    assert "all_registered_owners_complete_and_restored" not in result
    assert result["training"]["quality_claim"] == "none"


def test_probe_timeout_retains_error_and_unknown_liveness_with_attention(tmp_path):
    registrations = owner(tmp_path)

    def timeout(_node, _pids):
        raise subprocess.TimeoutExpired("cpu-probe", 45)

    result = monitor.observe(tmp_path, registrations, inspect_node=timeout)
    assert result["needs_attention"]
    assert result["nodes"][NODE]["observation_status"] == "unavailable"
    assert "TimeoutExpired" in result["nodes"][NODE]["error"]
    assert result["owners"][0]["owner_liveness"] == "not_observed"
    assert result["training"]["status"] == "not_prepared"


def test_reused_owner_pid_requires_attention_without_claiming_training_completion(tmp_path):
    registrations = owner(tmp_path)
    result = monitor.observe(
        tmp_path,
        registrations,
        inspect_node=lambda *_: {
            "processes": {"123": identity(123) | {"start_ticks": "5678"}},
            "gpus": "",
            "compute": "",
        },
    )
    assert result["needs_attention"]
    assert result["owners"][0]["owner_liveness"] == "exited_or_changed"
    assert not result["training_complete"]


def test_malformed_owner_schema_preserves_full_progress_and_exception(prepared, tmp_path):
    root, campaign = prepared
    history(campaign, campaign["trials"][0], steps=3, state="running")
    registrations = owner(tmp_path, state={"status": "running", "pid": 123})
    result = monitor.observe(root, registrations, inspect_node=node_probe)
    assert result["needs_attention"]
    assert result["allocation_observation"]["status"] == "invalid_evidence"
    assert "KeyError" in result["allocation_observation"]["error"]
    assert result["training"]["trials"][0]["observed_steps"] == 3


def test_changed_bound_source_retains_reporting_exception_and_attention(prepared):
    root, campaign = prepared
    path = Path(campaign["source"]) / "experiment/train.py"
    path.write_text("# Changed offline source identity\n")
    result = monitor.observe(root, [], inspect_node=node_probe)
    assert result["needs_attention"]
    assert result["training"]["status"] == "invalid_evidence"
    assert "ValueError" in result["training"]["error"]
    assert not result["training_complete"]


def test_unapproved_node_is_refused_before_any_process_probe(tmp_path):
    def forbidden_probe(*_):
        pytest.fail("An unapproved registration must never reach the probe")

    registrations = [{"node": "192.0.2.1", "state": str(tmp_path / "owner.json")}]
    with pytest.raises(ValueError, match="unapproved node"):
        monitor.observe(tmp_path, registrations, inspect_node=forbidden_probe)


def test_once_persists_atomic_snapshot_and_appends_history_under_real_lock(tmp_path):
    registrations = tmp_path / "registrations.json"
    save(registrations, owner(tmp_path))
    out = tmp_path / "monitor"
    before = (tmp_path / "owner.json").read_bytes()
    first = monitor.run_monitor(tmp_path, registrations, out, once=True, inspect_node=node_probe)
    second = monitor.run_monitor(tmp_path, registrations, out, once=True, inspect_node=node_probe)
    assert short.read(out / "latest.json") == second
    assert [json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()] == [first, second]
    assert second["owners"][0]["owner_liveness"] == "live"
    assert not (out / "latest.json.tmp").exists()
    assert (tmp_path / "owner.json").read_bytes() == before
    persisted = file_hashes(out)
    with (out / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            monitor.run_monitor(tmp_path, registrations, out, once=True, inspect_node=node_probe)
    assert file_hashes(out) == persisted


def test_invalid_registration_json_is_persisted_with_attention_then_recovers(tmp_path):
    registrations = tmp_path / "registrations.json"
    registrations.write_text("[")
    out = tmp_path / "monitor"
    invalid = monitor.run_monitor(tmp_path, registrations, out, once=True, inspect_node=node_probe)
    assert invalid["needs_attention"]
    assert "JSONDecodeError" in invalid["registrations"]["error"]
    assert invalid["training"]["status"] == "not_prepared"
    save(registrations, [])
    recovered = monitor.run_monitor(tmp_path, registrations, out, once=True, inspect_node=node_probe)
    assert not recovered["needs_attention"]
    assert recovered["registrations"]["status"] == "ok"
    assert len((out / "observations.jsonl").read_text().splitlines()) == 2


def test_cli_once_reports_not_prepared_without_any_node_probe(tmp_path):
    registrations = tmp_path / "registrations.json"
    save(registrations, [])
    out = tmp_path / "monitor"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(monitor.__file__)),
            "--root",
            str(tmp_path),
            "--registrations",
            str(registrations),
            "--out",
            str(out),
            "--interval",
            "60",
            "--once",
        ],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["training"] == "not_prepared"
    assert not summary["training_complete"]
    assert not summary["needs_attention"]
    assert short.read(out / "latest.json")["training"]["planned_trials"] == 12
