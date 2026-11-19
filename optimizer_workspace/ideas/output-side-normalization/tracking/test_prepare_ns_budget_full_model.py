# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual NS-budget candidate transformations, finite updates and GPU denial."""

import copy
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import prepare_ns_budget_full_model as preparation
import pytest
import torch
from benchmark_diagonal import source_definitions
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

PRESERVED = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/fused-continuation-owner-fixture-20261118T0634"
)
REVISION = "Toner-dab7f3777c75e740536df822f50ffa1d929a340c"


def fixture_source() -> Path:
    archived = Path(__file__).resolve().parents[1]
    return archived if (archived / preparation.config_name("muont", "tpu", 5, "lower")).is_file() else PRESERVED


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root, previous = tmp_path / "root", tmp_path / "previous"
    root.mkdir()
    source = fixture_source()
    for directory in ("experiment", "artifacts/pinned-sources-materialized"):
        shutil.copytree(
            source / directory,
            previous / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    if source != PRESERVED:
        for name in preparation.configurations(source):
            (previous / name).unlink()
        shutil.copy2(source / preparation.PARENT_OWNER_FIXTURE, previous / preparation.OWNER_PATH)
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    write(previous / "preserved-evidence.json", {"status": "preserved failed predecessor"})
    frozen = {
        "candidate": str(previous),
        "files_sha256": candidate_files(previous),
        "data_receipt_sha256": "fixture",
    }
    binding = root / preparation.original.PARENT_BINDING
    binding.parent.mkdir(parents=True)
    write(binding, {"status": "preserved CPU fixture"})
    monkeypatch.setattr(preparation, "parent", lambda observed: frozen)
    monkeypatch.setattr(
        preparation,
        "helper_sources",
        lambda observed: {name: Path(preparation.__file__).parent / name for name in preparation.NEW_HELPERS},
    )
    return root, previous, frozen


def test_full_model_candidate_keeps_equal_grids_and_denies_unowned_training_before_outputs(parent, tmp_path):
    root, previous, frozen = parent
    candidate = tmp_path / "candidate"
    prepared = preparation.prepare(root, candidate)
    assert candidate_files(previous) == frozen["files_sha256"]
    assert (candidate / "preserved-evidence.json").read_bytes() == (previous / "preserved-evidence.json").read_bytes()
    assert len(prepared["trials"]) == 18 and prepared["full_budget_config_count"] == 54
    assert prepared["planned_diagnostic_tokens"] == 452984832
    assert prepared["planned_full_grid_tokens"] == 144955146240
    references = {
        family: read(previous / f"experiment/configs/{family}_full_lr_lower_0p5m_seed0.json")
        for family in preparation.FAMILIES
    }
    for name, actual in preparation.configurations(candidate).items():
        expected = copy.deepcopy(references[actual["experiment"]["arm"]])
        expected["optimizer"].update(
            lr=actual["optimizer"]["lr"],
            ns_steps=actual["optimizer"]["ns_steps"],
            ns_mode=actual["optimizer"]["ns_mode"],
        )
        assert actual == expected and read(candidate / name) == expected
    out = tmp_path / "unowned-training"
    code = subprocess.run(
        [
            preparation.owner.PYTHON,
            "-P",
            "-m",
            "experiment.train",
            "--config",
            str(candidate / preparation.config_name("outer_leftdiag_fused", "f32", 5, "lower")),
            "--data",
            str(root / "missing"),
            "--data-receipt",
            str(root / "missing-receipt"),
            "--out",
            str(out),
            "--launch-release",
            str(root / "missing-release"),
        ],
        cwd=candidate,
        env=os.environ | preparation.owner.ENVIRONMENT | {"PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert code.returncode != 0 and "separately accepted owner" in code.stderr
    assert not out.exists() and not (root / "runs").exists()
    with pytest.raises(FileExistsError):
        preparation.prepare(root, candidate)


@pytest.mark.parametrize("mutation", ("schedule", "auxiliary_lr", "numerics", "owner"))
def test_ns_rehashed_candidate_cannot_change_unrelated_settings_or_remove_execution_denial(parent, tmp_path, mutation):
    root, _, _ = parent
    candidate = tmp_path / "candidate"
    prepared = preparation.prepare(root, candidate)
    if mutation in ("schedule", "auxiliary_lr"):
        path = candidate / preparation.config_name("muont", "f32", 5, "center")
        config = read(path)
        if mutation == "schedule":
            config["schedule"]["stable_steps"] = 2048
        else:
            config["optimizer"]["auxiliary_lr"] *= 2
        write(path, config)
    else:
        path = candidate / (preparation.OWNER_PATH if mutation == "owner" else "experiment/block_toner.py")
        path.write_text("def require_fused_attribution_owner(args): pass\n" if mutation == "owner" else "# changed\n")
    prepared["files_sha256"] = candidate_files(candidate)
    write(candidate / preparation.PREPARATION, prepared)
    with pytest.raises(ValueError):
        preparation.require_preparation(root, candidate)
    assert not (root / "runs").exists()


@pytest.mark.parametrize("shape", ((4, 4), (3, 7), (7, 3)))
@pytest.mark.parametrize("steps", (5, 7, 10))
def test_actual_config_budget_changes_preserved_batched_update_against_independent_recurrence(shape, steps):
    source = fixture_source()
    config = preparation.configurations(source)[preparation.config_name("outer_leftdiag_fused", "f32", steps, "lower")]
    pinned = source / "artifacts/pinned-sources-materialized" / REVISION
    arguments = source_definitions(
        pinned / "reproduce/marin/optimizer.py", ("fast_arguments",), {"SimpleNamespace": SimpleNamespace}
    )["fast_arguments"](config)
    numerical = source_definitions(
        pinned / "src/toner/fast/muon.py", ("ABC", "_bmm", "newton_schulz_batched"), {"torch": torch}
    )
    block = source_definitions(
        source / "experiment/block_toner.py",
        ("stack_unit_rms", "block_update_stack"),
        {"torch": torch, "MU": SimpleNamespace(**numerical)},
    )
    generator = torch.Generator().manual_seed(17)
    weight, gradient, momentum = (torch.randn(2, *shape, generator=generator) for _ in range(3))
    right_inner = torch.eye(shape[1]).repeat(2, 1, 1) * 1.25
    right_outer = torch.eye(shape[1]).repeat(2, 1, 1)
    outer = torch.linspace(0.8, 1.2, shape[0]).repeat(2, 1)
    expected_momentum = 0.95 * momentum + gradient
    x = torch.bmm(gradient + 0.95 * expected_momentum, right_inner)
    x = x / (torch.linalg.matrix_norm(x, dim=(-2, -1), keepdim=True) + 1e-5)
    transposed = shape[0] > shape[1]
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    if transposed:
        x = x.transpose(-2, -1)
    left = outer * shape[0] ** 0.5 / outer.norm(dim=-1, keepdim=True)
    expected_delta = -config["optimizer"]["lr"] * (
        max(1.0, shape[0] / shape[1]) ** 0.5 * left[:, :, None] * x + 0.1 * weight
    )
    actual_momentum, delta = block["block_update_stack"](
        weight,
        gradient,
        momentum,
        (None, outer),
        (right_inner, right_outer),
        args=arguments,
        lr=config["optimizer"]["lr"],
        weight_decay=0.1,
    )
    torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
    torch.testing.assert_close(delta, expected_delta, rtol=1e-5, atol=1e-6)
    alternate = copy.deepcopy(arguments)
    alternate.muon_ns_steps = 7 if steps == 5 else 5
    other = block["block_update_stack"](
        weight,
        gradient,
        momentum,
        (None, outer),
        (right_inner, right_outer),
        args=alternate,
        lr=config["optimizer"]["lr"],
        weight_decay=0.1,
    )[1]
    assert not torch.equal(delta, other)


def test_ns_full_model_cpu_failure_preserves_source_and_denied_candidate(parent, tmp_path, monkeypatch):
    root, previous, _ = parent
    candidate, out = tmp_path / "candidate", tmp_path / "failed-cpu"
    preparation.prepare(root, candidate)
    fixtures = {
        name: sha256(previous / name)
        for name in (
            "experiment/upstream_manifest.json",
            f"artifacts/external-sources/{REVISION}/src/toner/fast/muon.py",
        )
    }
    monkeypatch.setattr(preparation, "accepted_parent", lambda observed: {"fixtures_sha256": fixtures})
    monkeypatch.setattr(
        preparation, "cpu_command", lambda observed: [preparation.owner.PYTHON, "-c", "raise SystemExit(7)"]
    )
    with pytest.raises(RuntimeError):
        preparation.cpu(root, candidate, out)
    saved = read(out / "receipt.json")
    assert saved["status"] == "needs_attention" and saved["exit_code"] == 7
    assert sha256(Path(saved["archive"])) == saved["archive_sha256"]
    assert saved["cuda_visible_devices"] == "" and not (root / "runs").exists()
    assert (out / "artifacts/external-sources").is_symlink()
    assert (out / "artifacts/external-sources").readlink() == Path("pinned-sources-materialized")
    assert all(sha256(out / name) == digest for name, digest in fixtures.items())
    preparation.require_archive(Path(saved["archive"]), saved["inventory_sha256"], alias=True)


def test_ns_full_model_rejects_inherited_fixture_conflicting_with_preserved_source(parent, tmp_path, monkeypatch):
    root, _, _ = parent
    candidate, out = tmp_path / "candidate", tmp_path / "conflicting-fixture"
    preparation.prepare(root, candidate)
    name = f"artifacts/external-sources/{REVISION}/src/toner/fast/muon.py"
    monkeypatch.setattr(preparation, "accepted_parent", lambda observed: {"fixtures_sha256": {name: "0" * 64}})
    with pytest.raises(ValueError, match="conflicts with preserved source"):
        preparation.cpu(root, candidate, out)
    assert not (out / "cpu.log").exists()
    assert (out / name).read_bytes() == (candidate / name).read_bytes()
    preparation.require_preparation(root, candidate)
