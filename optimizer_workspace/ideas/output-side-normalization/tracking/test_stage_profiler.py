# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real bounded CPU traces and numerical pass-through before CUDA profiling."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from stage_profiler import BLOCK_FUNCTIONS, StageProfiler, digest
from torch.profiler import ProfilerActivity


def block_module():
    if Path(__file__).parent.name == "tests":
        return importlib.import_module("experiment.block_toner")
    candidate = Path(
        "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
        "/candidates/upstream-full-model-production-20261117"
    )
    package_name = "_stage_profile_numerical_reference"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            candidate / "experiment/__init__.py",
            submodule_search_locations=[str(candidate / "experiment")],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        importlib.import_module(f"{package_name}.reference")
    return importlib.import_module(f"{package_name}.block_toner")


def test_stage_profile_exports_only_requested_interval_and_preserves_results(tmp_path):
    output = tmp_path / "rank0"
    profiler = StageProfiler(
        output, first=2, last=3, rank=0, provenance={"arm": "fixture"}, activities=[ProfilerActivity.CPU]
    )
    multiply = profiler.wrap(torch.mul, "fixture_multiply")
    for update in range(1, 5):
        profiler.begin_update(update)
        left = torch.tensor([update, update + 1], dtype=torch.float64, requires_grad=True)
        actual = multiply(left, left)
        assert torch.equal(actual, left.square())
        actual.sum().backward()
        assert torch.equal(left.grad, 2 * left.detach())
        profiler.end_update(update)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "captured"
    assert summary["observed_updates"] == [2, 3]
    assert summary["trace_sha256"] == digest(output / "trace.json")
    assert summary["provenance"] == {"arm": "fixture"}
    regions = {region["name"]: region for region in summary["regions"]}
    assert regions["osn/training_update"]["calls"] == 2
    assert regions["osn/fixture_multiply"]["calls"] == 2
    assert regions["osn/fixture_multiply"]["cpu_total_microseconds"] > 0
    trace = json.loads((output / "trace.json").read_text())
    assert sum(event.get("name") == "osn/fixture_multiply" for event in trace["traceEvents"]) == 2
    original = (output / "summary.json").read_bytes()
    profiler.finish()
    assert (output / "summary.json").read_bytes() == original


def test_stage_profile_partial_interval_remains_incomplete_and_cannot_overwrite(tmp_path):
    output = tmp_path / "rank3"
    profiler = StageProfiler(output, first=2, last=4, rank=3, provenance={}, activities=[ProfilerActivity.CPU])
    profiler.begin_update(2)
    with profiler.scope("fixture_add"):
        assert torch.equal(torch.ones(4) + 1, torch.full((4,), 2.0))
    profiler.end_update(2)
    profiler.finish()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert summary["observed_updates"] == [2]
    assert summary["rank"] == 3
    original = (output / "summary.json").read_bytes()
    second = StageProfiler(output, first=2, last=4, rank=3, provenance={}, activities=[ProfilerActivity.CPU])
    with pytest.raises(FileExistsError):
        second.begin_update(2)
    assert (output / "summary.json").read_bytes() == original


def test_stage_profile_real_block_step_preserves_momentum_weights_and_finite_guard(tmp_path, monkeypatch):
    module = block_module()
    generator = torch.Generator().manual_seed(73)
    weights = [torch.randn(4, 4, generator=generator), torch.randn(8, 4, generator=generator)]
    gradients = [torch.randn(4, 4, generator=generator), torch.randn(8, 4, generator=generator)]
    args = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32")

    def engine():
        fast = module.BlockToner.__new__(module.BlockToner)
        fast.arm = module.BLOCK_ARMS["outer_leftdiag_fused"]
        fast.input = fast.output = fast.input_diagonal = fast.output_diagonal = None
        fast.world, fast.rank, fast.device, fast.pg = 1, 0, torch.device("cpu"), None
        fast.args = args
        fast.lagged, fast.last_metrics, fast.steps = {}, {}, 0
        fast.update_verifier = None
        fast.params = [torch.nn.Parameter(weight.clone()) for weight in weights]
        for parameter, gradient in zip(fast.params, gradients, strict=True):
            parameter.grad = gradient.clone()
        fast.shape_groups = [
            (tuple(weight.shape), [[(index, "q", parameter)]])
            for index, (weight, parameter) in enumerate(zip(weights, fast.params, strict=True))
        ]
        fast.momentum = {shape: torch.zeros(1, *shape) for shape, _ in fast.shape_groups}
        fast.param_groups = [{"params": fast.params, "lr": 0.02, "weight_decay": 0.1}]
        return fast

    reference, observed = engine(), engine()
    reference.step()
    for name, _ in BLOCK_FUNCTIONS:
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(module, "profile_scope", lambda name: None, raising=False)
    profiler = StageProfiler(
        tmp_path / "rank0", first=1, last=1, rank=0, provenance={}, activities=[ProfilerActivity.CPU]
    )
    profiler.install(observed, module)
    profiler.begin_update(1)
    observed.step()
    profiler.end_update(1)
    for actual, expected in zip(observed.params, reference.params, strict=True):
        assert torch.equal(actual, expected)
    assert all(torch.equal(observed.momentum[shape], reference.momentum[shape]) for shape in observed.momentum)
    assert observed.steps == reference.steps == 1
    summary = json.loads((tmp_path / "rank0/summary.json").read_text())
    assert summary["status"] == "captured"
    assert any(region["name"] == "osn/optimizer_step" for region in summary["regions"])
    observed.params[-1].grad[0, 0] = float("inf")
    before = [parameter.detach().clone() for parameter in observed.params]
    with pytest.raises(FloatingPointError):
        observed.step()
    assert all(torch.equal(actual, expected) for actual, expected in zip(observed.params, before, strict=True))
