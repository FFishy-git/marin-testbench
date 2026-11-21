# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline cache acceptance from actual all-rank continuation verification."""

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import related_execution as execution
import related_verified_cache as cache
import torch
from experiment.records import compare_continuation
from related_campaign import TASK, WORLD, config_identity, read, sha256


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def checkpoint(directory, step, config, campaign):
    path = directory / "checkpoints" / f"step_{step:06d}"
    save(path / "complete.json", {"step": step, "world": WORLD})
    for rank in range(WORLD):
        torch.save(
            {
                "step": step,
                "next_global_sequence": step * config["training"]["global_batch_sequences"],
                "identity": {
                    "source_sha256": campaign["source_sha256"],
                    "config_sha256": config_identity(config),
                    "data_receipt_sha256": campaign["data_receipt_sha256"],
                },
                "model": {"weight": torch.tensor([step, rank], dtype=torch.float32)},
                "fast": {"root": torch.eye(2), "root_step": step},
            },
            path / f"rank_{rank}.pt",
        )


def smoke(directory, step, config, campaign):
    checkpoint(directory, step, config, campaign)
    save(
        directory / "run.json",
        {
            "status": "execution_check_complete",
            "completed_steps": step,
            "execution_check": True,
            "config": config,
            "source_revision": campaign["source_revision"],
            "source_sha256": campaign["source_sha256"],
            "data": campaign["data"],
            "data_receipt_sha256": campaign["data_receipt_sha256"],
            "world_size": WORLD,
        },
    )


@pytest.fixture
def verified(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "output"
    for name in cache.INTERPRETATION_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TASK / name, path)
    monkeypatch.setattr(cache, "TASK", source)
    monkeypatch.setattr(execution, "ARTIFACT_HASHES", {})
    monkeypatch.setattr(execution, "VERIFIED_COMPARISONS", {})
    data_receipt = root / "data/receipt.json"
    save(data_receipt, {"offline_data": True})
    sources = {"experiment/records.py": sha256(source / "experiment/records.py")}
    upstream = source / "artifacts/pinned/upstream.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# tiny offline upstream\n")
    full_sources = {**sources, "artifacts/pinned/upstream.py": sha256(upstream)}
    campaign = {
        "root": str(root),
        "source": str(source),
        "source_sha256": full_sources,
        "source_revision": "offline",
        "data": str(data_receipt.parent),
        "data_receipt": str(data_receipt),
        "data_receipt_sha256": sha256(data_receipt),
        "waves": [],
        "trials": [],
    }
    work, wave_name = root / "validation", "controlled_seed0_lr1"
    executions, smokes, probes, config_hashes = [], {}, {}, {}

    def command_record(command):
        log = work / f"command_{len(executions)}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("offline successful boundary\n")
        executions.append({"command": command, "exit_code": 0, "log": str(log), "log_sha256": sha256(log)})

    for key, module, checks, world_key in (
        ("gpu_receipt", "experiment.gpu_checks", execution.SYNTHETIC_CHECKS, "world"),
        ("related_gpu_receipt", "experiment.related_gpu_checks", execution.RELATED_CHECKS, "world_size"),
    ):
        path = work / f"{key}.json"
        save(path, {"source_sha256": sources, "checks": dict.fromkeys(checks, True), world_key: 2})
        probes[key] = str(path)
        command_record([*execution.launcher(execution.PYTHON, 2), module, "--receipt", str(path)])
    arms = ("muon_control", "inner_outer_mlpdiag", "go_muon_control", "mousse_control")
    for arm in arms:
        name = f"{arm}_0p5m_seed0_lr1"
        config = read(TASK / "experiment/comparisons/related_controlled_short" / f"{name}.json")
        config_path = source / "configs" / f"{name}.json"
        save(config_path, config)
        alias = {"config": str(config_path), "config_sha256": sha256(config_path), "wave": wave_name}
        campaign["trials"].append({"name": name, "arm": arm, "aliases": [alias]})
        config_hashes[name] = alias["config_sha256"]
        complete, split = execution.continuation_steps(config)
        full, resumed = work / f"{name}_continuous", work / f"{name}_resumed"
        common = execution.trial_command(config_path, data_receipt.parent, data_receipt, full)[:-2]
        command_record([*common, "--out", str(full), "--max-steps", str(complete)])
        command_record([*common, "--out", str(resumed), "--max-steps", str(split)])
        smoke(full, complete, config, campaign)
        checkpoint(resumed, split, config, campaign)
        resume_step, pre_refresh = split, None
        if arm in execution.RELATED_ARMS and arm != "muon_control":
            resume_step = split + 1
            prefix = work / f"{name}_pre_refresh_continuous"
            smoke(prefix, resume_step, config, campaign)
            checkpoint(resumed, resume_step, config, campaign)
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
        smoke(resumed, complete, config, campaign)
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
    campaign["waves"] = [{"wave_name": wave_name, "trials": [trial["name"] for trial in campaign["trials"]]}]
    receipt_path = work / "validation.json"
    receipt = {
        "wave_name": wave_name,
        "source_sha256": sources,
        "config_sha256": config_hashes,
        "environment": execution.ENVIRONMENT,
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "checks": dict.fromkeys(execution.REQUIRED_CHECKS, True),
        "evidence": {"cpu_test_exit_code": 0, "gpu_commands": executions, "real_data_smokes": smokes, **probes},
    }
    save(receipt_path, receipt)
    validation = execution.require_validation(campaign, receipt_path)
    generated = cache.cache_path(campaign, receipt_path)
    if generated.exists():
        generated.unlink()
        generated.parent.rmdir()
    return (
        campaign,
        receipt_path,
        receipt,
        validation,
        dict(execution.ARTIFACT_HASHES),
        dict(execution.VERIFIED_COMPARISONS),
    )


def deny_checkpoint_reads(path, *args, **kwargs):
    if path.suffix == ".pt":
        raise AssertionError("Verified cache reread a checkpoint")
    return ORIGINAL_OPEN(path, *args, **kwargs)


ORIGINAL_OPEN = Path.open


def test_store_and_lookup_reuse_actual_verified_pairs_without_checkpoint_reads(verified, monkeypatch):
    campaign, path, receipt, validation, identities, comparisons = verified
    output = cache.cache_path(campaign, path)
    monkeypatch.setattr(Path, "open", deny_checkpoint_reads)
    record = cache.store(output, validation, receipt, identities)
    assert read(output) == record
    hit = cache.validation_cache(campaign, path)
    assert hit is not None
    assert hit["validation"] == validation
    assert hit["artifact_hashes"] == identities
    assert hit["verified_comparisons"] == comparisons
    assert not output.with_suffix(".json.tmp").exists()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        cache.store(output, validation, receipt, identities)
    assert output.read_bytes() == original
    execution.ARTIFACT_HASHES.clear()
    execution.VERIFIED_COMPARISONS.clear()
    assert execution.require_validation(campaign, path) == validation


def test_changed_artifact_with_restored_mtime_invalidates_lookup_and_store(verified):
    campaign, path, receipt, validation, identities, _ = verified
    output = cache.cache_path(campaign, path)
    cache.store(output, validation, receipt, identities)
    artifact = Path(next(name for name in identities if name.endswith("rank_0.pt")))
    original = artifact.stat()
    contents = artifact.read_bytes()
    artifact.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    os.utime(artifact, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert artifact.stat().st_size == original.st_size
    assert artifact.stat().st_mtime_ns == original.st_mtime_ns
    assert artifact.stat().st_ctime_ns != original.st_ctime_ns
    assert cache.load(output, path) is None
    other = output.with_name("changed-artifact.json")
    with pytest.raises(ValueError):
        cache.store(other, validation, receipt, identities)
    assert not other.exists()


@pytest.mark.parametrize("change", ["cache", "receipt", "log", "probe", "config", "source", "upstream", "manifest"])
def test_changed_control_or_cache_cannot_reuse_verified_comparisons(verified, change):
    campaign, path, receipt, validation, identities, _ = verified
    output = cache.cache_path(campaign, path)
    cache.store(output, validation, receipt, identities)
    if change == "cache":
        record = read(output)
        first = next(iter(record["files"]))
        record["files"][first]["stat"][2] += 1
        output.chmod(0o644)
        save(output, record)
    elif change == "receipt":
        receipt["evidence"]["real_data_smokes"][next(iter(validation["aliases"]))]["continuation"][
            "tensor_comparisons"
        ] += 1
        save(path, receipt)
    else:
        target = {
            "log": next(iter(validation["logs_sha256"])),
            "probe": next(iter(validation["gpu_receipts_sha256"])),
            "config": next(iter(validation["aliases"].values()))["config"],
            "source": str(cache.TASK / "tracking/related_execution.py"),
            "upstream": str(cache.TASK / "artifacts/pinned/upstream.py"),
            "manifest": next(name for name in identities if name.endswith("run.json")),
        }[change]
        with Path(target).open("ab") as stream:
            stream.write(b"\nchanged\n")
    assert cache.load(output, path) is None
    assert cache.validation_cache(campaign, path) is None


def test_missing_cache_and_changed_campaign_do_not_create_or_accept_evidence(verified):
    campaign, path, receipt, validation, identities, _ = verified
    output = cache.cache_path(campaign, path)
    assert cache.validation_cache(campaign, path) is None
    assert not output.parent.exists()
    cache.store(output, validation, receipt, identities)
    changed = copy.deepcopy(campaign)
    changed["trials"][0]["aliases"][0]["config_sha256"] = "0" * 64
    assert cache.validation_cache(changed, path) is None
    changed = copy.deepcopy(campaign)
    changed["data_receipt_sha256"] = "0" * 64
    assert cache.validation_cache(changed, path) is None


def test_valid_checksum_with_missing_artifact_inventory_is_not_accepted(verified):
    campaign, path, receipt, validation, identities, _ = verified
    output = cache.cache_path(campaign, path)
    cache.store(output, validation, receipt, identities)
    record = read(output)
    record.pop("integrity_sha256")
    record["files"].pop(next(iter(record["files"])))
    record["integrity_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output.chmod(0o644)
    save(output, record)
    assert cache.load(output, path) is None
