# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check combined post-NS algebra against an independent float64 matrix expression."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_combined_outer_gpu as controller
import torch
from diagnose_combined_outer import combined_outer_delta, combined_outer_update
from diagnose_combined_outer_gpu import require_owner


@pytest.mark.parametrize("shape", ((3, 4, 4), (2, 8, 4), (2, 4, 8)))
@pytest.mark.parametrize("zero_lr", (False, True))
def test_combined_outer_matches_independent_normalized_direction_and_decay(shape, zero_lr):
    generator = torch.Generator().manual_seed(31)
    count, rows, columns = shape
    weights = torch.randn(shape, generator=generator, dtype=torch.float64)
    direction = torch.randn(shape, generator=generator, dtype=torch.float64)
    right = torch.randn(count, columns, columns, generator=generator, dtype=torch.float64)
    right = right @ right.mT + torch.eye(columns, dtype=torch.float64)
    left = torch.rand(count, rows, generator=generator, dtype=torch.float64) + 0.1
    lr, decay = (0.0 if zero_lr else 0.023), 0.1
    expected = []
    for weight, value, output_factor, input_factor in zip(weights, direction, left, right, strict=True):
        normalized_left = torch.diag(output_factor / output_factor.square().mean().sqrt())
        normalized_right = input_factor / (input_factor.square().sum() / columns).sqrt()
        scale = max(1.0, rows / columns) ** 0.5
        expected.append(
            -lr
            * (scale * normalized_left @ value @ normalized_right + decay * weight @ normalized_right @ normalized_right)
        )

    def normalize(factor):
        dimensions = (-2, -1) if factor.ndim == 3 else (-1,)
        return factor * (factor.shape[-1] ** 0.5 / factor.norm(dim=dimensions, keepdim=True))

    actual = combined_outer_delta(weights, direction, left, right, lr=lr, weight_decay=decay, unit_rms=normalize)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)
    if zero_lr:
        assert torch.count_nonzero(actual).item() == 0


@pytest.mark.parametrize("inner", (False, True))
@pytest.mark.parametrize("nesterov", (False, True))
def test_combined_update_retains_exact_momentum_and_pre_ns_source(inner, nesterov):
    generator = torch.Generator().manual_seed(47)
    weights = torch.randn(2, 4, 8, generator=generator, dtype=torch.float64)
    gradients = torch.randn(weights.shape, generator=generator, dtype=torch.float64)
    momenta = torch.randn(weights.shape, generator=generator, dtype=torch.float64)
    left = torch.rand(2, 4, generator=generator, dtype=torch.float64) + 0.1
    right = torch.randn(2, 8, 8, generator=generator, dtype=torch.float64)
    right = right @ right.mT + torch.eye(8, dtype=torch.float64)
    expected_momentum = 0.95 * momenta + gradients
    expected_source = torch.bmm(gradients + 0.95 * expected_momentum if nesterov else expected_momentum, right)
    if inner:
        expected_source = expected_source * left[:, :, None]

    def ns(source, steps, eps, mode):
        assert steps == 10 and eps == 1e-5 and mode == "tpu"
        torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
        return source

    def normalize(factor):
        dims = (-2, -1) if factor.ndim == 3 else (-1,)
        return factor * (factor.shape[-1] ** 0.5 / factor.norm(dim=dims, keepdim=True))

    trace = {}
    actual_momentum, delta = combined_outer_update(
        weights,
        gradients,
        momenta,
        (left if inner else None, left),
        (right, right),
        args=SimpleNamespace(
            muon_momentum=0.95, muon_nesterov=nesterov, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="tpu"
        ),
        lr=0.02,
        weight_decay=0.1,
        muon=SimpleNamespace(newton_schulz_batched=ns),
        unit_rms=normalize,
        trace=trace,
    )
    normalized_right, normalized_left = normalize(right), normalize(left)
    expected_delta = -0.02 * (
        normalized_left[:, :, None] * torch.bmm(expected_source, normalized_right)
        + 0.1 * torch.bmm(torch.bmm(weights, normalized_right), normalized_right)
    )
    torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
    torch.testing.assert_close(trace["ns_source"], expected_source, rtol=0, atol=0)
    torch.testing.assert_close(trace["ns_output"], expected_source, rtol=0, atol=0)
    torch.testing.assert_close(delta, expected_delta, rtol=1e-12, atol=1e-12)


@pytest.fixture
def bound_probe(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "run-preparation").mkdir(parents=True)
    release = root / "run-preparation/wave7-release-binding.json"
    release.write_text("{}\n")
    guard = root / "campaigns/lr-fixed-age-0p5m-20261116/validation-binding.json"
    guard.parent.mkdir(parents=True)
    guard.write_text("{}\n")
    out = root / "probe"
    helpers = out / "helpers"
    helpers.mkdir(parents=True)
    names = (*controller.VALIDATION_HELPERS, *controller.EXTRA_HELPERS)
    for name in names:
        (helpers / name).write_text("# frozen fixture\n")
    (out / "source.tar").write_bytes(b"frozen probe archive")
    (out / "binding.json").write_text(
        json.dumps(
            {
                "root": str(root.resolve()),
                "out": str(out.resolve()),
                "node": controller.NODE,
                "helpers_sha256": {name: controller.sha256(helpers / name) for name in names},
                "archive_sha256": controller.sha256(out / "source.tar"),
                "original_guard_binding_sha256": controller.sha256(guard),
                "source_release_sha256": controller.sha256(release),
                "environment": controller.ENVIRONMENT,
            }
        )
    )
    monkeypatch.setattr(controller, "__file__", str(helpers / "run_combined_outer_gpu.py"))
    return root, out


def test_combined_probe_wrong_node_refuses_before_any_gpu_transfer(bound_probe, monkeypatch):
    root, out = bound_probe

    def observed(command, **kwargs):
        if command == ["hostname", "-I"]:
            return "28.216.17.9"
        raise AssertionError("GPU command before wrong-node refusal")

    monkeypatch.setattr(controller.subprocess, "check_output", observed)
    with pytest.raises(RuntimeError, match=r"original \.139"):
        controller.run(root, out)
    assert not (out / "state.json").exists()
    assert not (out / "diagnostic.log").exists()


@pytest.mark.parametrize("changed", ("helper", "allocation_guard"))
def test_combined_probe_changed_binding_blocks_gpu_transfer(bound_probe, monkeypatch, changed):
    root, out = bound_probe
    path = (
        out / "helpers/diagnose_combined_outer_gpu.py"
        if changed == "helper"
        else root / "campaigns/lr-fixed-age-0p5m-20261116/validation-binding.json"
    )
    path.write_text("# altered source\n")

    def observed(command, **kwargs):
        raise AssertionError("Process or GPU command before source refusal")

    monkeypatch.setattr(controller.subprocess, "check_output", observed)
    with pytest.raises(ValueError, match="binding changed"):
        controller.run(root, out)
    assert not (out / "state.json").exists()


def test_combined_worker_stale_controller_refuses_before_cuda(tmp_path):
    with subprocess.Popen(["sleep", "60"]) as parent:
        started = (Path("/proc") / str(parent.pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
        state = {
            "controller_pid": parent.pid,
            "controller_start_ticks": str(int(started) + 1),
            "child_pid": 1,
            "python": controller.PYTHON,
            "root": str(tmp_path),
            "child_marker": "not-owned",
            "status": "running",
        }
        (tmp_path / "state.json").write_text(json.dumps(state))
        try:
            with pytest.raises(RuntimeError, match="bound controller ownership"):
                require_owner(tmp_path)
        finally:
            parent.terminate()
            parent.wait(timeout=5)
