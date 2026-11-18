# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the separate production diagnostic's oracle and result release."""

import json
import subprocess
from pathlib import Path

import pytest
import run_upstream_outer_gpu as controller
import torch
from diagnose_upstream_outer_gpu import independent_outer_delta, require_owner
from launch_wave5 import NODES
from run_upstream_outer_gpu import require_result
from wave7_release import candidate_inventory


@pytest.mark.parametrize("shape", [(2, 4, 4), (2, 8, 4), (2, 4, 8)])
@pytest.mark.parametrize("apply_left", [False, True])
def test_gpu_diagnostic_fp64_oracle_matches_explicit_matrix_equation(shape, apply_left):
    generator = torch.Generator().manual_seed(79)
    weights, direction = (torch.randn(shape, generator=generator, dtype=torch.float64) for _ in range(2))
    count, rows, columns = shape
    matrix = torch.randn(count, columns, columns, generator=generator, dtype=torch.float64)
    right = matrix @ matrix.mT + torch.eye(columns, dtype=torch.float64)
    left = torch.rand(count, rows, generator=generator, dtype=torch.float64) + 0.25
    expected = []
    for weight, value, output, factor in zip(weights, direction, left, right, strict=True):
        normalized = factor * (factor.shape[-1] ** 0.5 / factor.norm())
        output_matrix = torch.diag(output / output.square().mean().sqrt()) if apply_left else torch.eye(rows)
        scale = max(1.0, rows / columns) ** 0.5
        expected.append(
            -0.023 * (scale * output_matrix.double() @ value @ normalized + 0.1 * weight @ normalized @ normalized)
        )
    actual = independent_outer_delta(weights, direction, left if apply_left else None, right, lr=0.023, decay=0.1)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)


def test_gpu_result_release_rejects_missing_cases_and_failed_parity_despite_passed_label():
    plan = {
        "shapes": [[2, 4, 8]],
        "patterns": ["gaussian", "zero"],
        "nesterov": [True],
        "learning_rates": [0.0, 0.02],
        "cases": 4,
    }
    state = {"controller_pid": 123, "controller_start_ticks": "456", "binding_sha256": "binding"}
    frozen = {"plan": plan, "helpers_sha256": {"probe.py": "digest"}}
    exact = dict.fromkeys(
        (
            "momentum",
            "ns_source",
            "ns_output",
            "parameter_delta",
            "outer_momentum",
            "outer_ns_source",
            "outer_ns_output",
        ),
        True,
    )
    cases = [
        {
            "shape": [2, 4, 8],
            "pattern": pattern,
            "nesterov": True,
            "lr": lr,
            "exact_null_and_unchanged_input": exact.copy(),
            "same_ns_outer_equation": {
                "within_tolerance": True,
                "finite": True,
                "mismatched_elements": 0,
                "elements": 64,
            },
        }
        for pattern in plan["patterns"]
        for lr in plan["learning_rates"]
    ]
    result = {
        **plan,
        "status": "passed_upstream_production_diagnostic_not_launch_gate",
        "owner": {"controller_pid": 123, "controller_start_ticks": "456"},
        "binding_sha256": "binding",
        "helpers_sha256": frozen["helpers_sha256"],
        "cases": cases,
        "timing": [
            {
                "shape": [2, 4, 8],
                "graphs": graphs,
                "paired_median_ms": {"legacy_direct": 1.0, "upstream_outer": 0.8, "upstream_null": 0.7},
                "upstream_outer_over_legacy": 0.8,
                "upstream_outer_over_null": 0.8 / 0.7,
            }
            for graphs in (False, True)
        ],
    }
    require_result(frozen, result, state)
    result["cases"] = cases[:-1]
    with pytest.raises(ValueError, match="coverage"):
        require_result(frozen, result, state)
    result["cases"] = cases
    cases[-1]["exact_null_and_unchanged_input"]["parameter_delta"] = False
    with pytest.raises(ValueError, match="parity"):
        require_result(frozen, result, state)
    cases[-1]["exact_null_and_unchanged_input"]["parameter_delta"] = True
    result["timing"][0]["paired_median_ms"]["upstream_outer"] = float("nan")
    with pytest.raises(ValueError, match="timing"):
        require_result(frozen, result, state)


@pytest.fixture
def bound_probe(tmp_path, monkeypatch):
    root = tmp_path / "root"
    preparation = root / "run-preparation"
    preparation.mkdir(parents=True)
    wave_helpers = root / "controllers/wave8-launch-20261117/tracking"
    wave_helpers.mkdir(parents=True)
    wave_archive = root / "wave-source.tar"
    wave_archive.write_bytes(b"wave archive fixture")
    wave_release = preparation / "wave8-launch-binding.json"
    wave_release.write_text(
        json.dumps(
            {
                "root": str(root),
                "status": "frozen_validated_launch_release",
                "helpers": str(wave_helpers),
                "arms": ["muont", "outer_leftdiag", "outer_leftdiag_fused", "outer_leftdiag_combined_fused"],
                "nodes": list(NODES),
                "helpers_sha256": {},
                "archive": str(wave_archive),
                "archive_sha256": controller.sha256(wave_archive),
            }
        )
    )
    previous = preparation / "wave8-candidate.json"
    previous.write_text("{}")
    candidate = root / "candidates/candidate"
    configs = candidate / "experiment/configs"
    configs.mkdir(parents=True)
    pinned = candidate / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for arm in ("outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused"):
        (configs / f"{arm}_0p5m_seed0.json").write_text(
            json.dumps(
                {
                    "model": {"layers": 32, "hidden": 512, "mlp": 2048},
                    "training": {"world_size": 8},
                    "optimizer": {"lr": 0.02, "ns_mode": "tpu"},
                    "experiment": {"arm": arm},
                }
            )
        )
    initial = candidate / "preparation.json"
    initial.write_text("{}")
    cpu = preparation / "cpu"
    cpu.mkdir()
    cpu_archive = cpu / "source.tar"
    cpu_archive.write_bytes(b"cpu archive fixture")
    log = cpu / "cpu.log"
    log.write_text("CPU fixture\n")
    receipt = cpu / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "cpu_verified_gpu_execution_denied",
                "candidate": str(candidate),
                "cuda_visible_devices": "",
                "arms": ["outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused"],
                "initial_preparation": str(initial),
                "initial_preparation_sha256": controller.sha256(initial),
                "files_sha256": candidate_inventory(candidate),
                "archive": str(cpu_archive),
                "archive_sha256": controller.sha256(cpu_archive),
                "commands": [{"exit_code": 0, "log": str(log), "log_sha256": controller.sha256(log)}] * 2,
                "predecessor_release_sha256": controller.sha256(wave_release),
                "predecessor_candidate_sha256": controller.sha256(previous),
            }
        )
    )
    campaign_directory = root / "campaigns" / controller.CAMPAIGN
    campaign_directory.mkdir(parents=True)
    campaign = campaign_directory / "campaign.json"
    campaign.write_text("{}")
    helpers = campaign_directory / "execution-helpers"
    helpers.mkdir()
    (helpers / "run_fixed_age_tuning.py").write_text("# execution fixture\n")
    execution_archive = campaign_directory / "execution-source.tar"
    execution_archive.write_bytes(b"execution archive fixture")
    (campaign_directory / "execution-binding.json").write_text(
        json.dumps(
            {
                "campaign": str(campaign),
                "campaign_sha256": controller.sha256(campaign),
                "archive": str(execution_archive),
                "archive_sha256": controller.sha256(execution_archive),
                "helpers": str(helpers),
                "helpers_sha256": {"run_fixed_age_tuning.py": controller.sha256(helpers / "run_fixed_age_tuning.py")},
            }
        )
    )
    out = preparation / "probe"
    controller.freeze(root, receipt, out)
    monkeypatch.setattr(controller, "__file__", str(out / "helpers/run_upstream_outer_gpu.py"))
    return root, receipt, out


def test_probe_completed_result_prerequisites_missing_refuses_without_gpu_or_keeper_changes(bound_probe, monkeypatch):
    root, receipt, out = bound_probe

    def boundary(command, **kwargs):
        if command == ["hostname", "-I"]:
            return controller.NODE
        raise AssertionError(f"Unexpected GPU/process side effect: {command}")

    monkeypatch.setattr(controller.subprocess, "check_output", boundary)
    with pytest.raises(RuntimeError, match="cloud verification are pending"):
        controller.run(root, receipt, out)
    assert not (out / "state.json").exists()
    assert not (out / "diagnostic.log").exists()


@pytest.mark.parametrize("changed", ("helper", "candidate", "cpu_log", "execution"))
def test_probe_changed_bound_evidence_refuses_before_allocation_transfer(bound_probe, monkeypatch, changed):
    root, receipt, out = bound_probe
    cpu = json.loads(receipt.read_text())
    paths = {
        "helper": out / "helpers/diagnose_upstream_outer_gpu.py",
        "candidate": Path(cpu["candidate"]) / "experiment/configs/null_leftdiag_upstream_fused_0p5m_seed0.json",
        "cpu_log": Path(cpu["commands"][0]["log"]),
        "execution": root / "campaigns" / controller.CAMPAIGN / "execution-helpers/run_fixed_age_tuning.py",
    }
    paths[changed].write_text("changed fixture\n")

    def boundary(command, **kwargs):
        raise AssertionError("GPU or process command before bound-source refusal")

    monkeypatch.setattr(controller.subprocess, "check_output", boundary)
    with pytest.raises(ValueError):
        controller.run(root, receipt, out)
    assert not (out / "state.json").exists()


def test_gpu_child_unowned_parent_refuses_before_cuda(tmp_path):
    with subprocess.Popen(["sleep", "60"]) as parent:
        started = (Path("/proc") / str(parent.pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
        (tmp_path / "state.json").write_text(
            json.dumps(
                {
                    "controller_pid": parent.pid,
                    "controller_start_ticks": str(int(started) + 1),
                    "child_pid": 1,
                    "python": controller.PYTHON,
                    "root": str(tmp_path),
                    "child_marker": "unowned",
                    "status": "running",
                }
            )
        )
        try:
            with pytest.raises(RuntimeError, match="ownership"):
                require_owner(tmp_path, tmp_path / "receipt.json")
        finally:
            parent.terminate()
            parent.wait(timeout=5)
