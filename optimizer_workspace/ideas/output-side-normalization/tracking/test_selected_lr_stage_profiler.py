# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real CPU traces, gradients and optimizer state without CUDA."""

import json

import pytest
import torch
from selected_lr_stage_profiler import StageProfiler, digest
from torch.profiler import ProfilerActivity


def test_profile_interval_preserves_gradients_weights_momentum_and_immutable_trace(tmp_path):
    actual = torch.nn.Parameter(torch.tensor([0.5, -1.0], dtype=torch.float64))
    expected = torch.nn.Parameter(actual.detach().clone())
    profiled = torch.optim.SGD([actual], lr=0.02, momentum=0.95)
    reference = torch.optim.SGD([expected], lr=0.02, momentum=0.95)
    output = tmp_path / "rank0"
    profiler = StageProfiler(
        output, first=2, last=3, rank=0, provenance={"arm": "fixture"}, activities=[ProfilerActivity.CPU]
    )
    step = profiler.wrap(profiled.step, "optimizer_step")
    for update in range(1, 5):
        profiler.begin_update(update)
        profiled.zero_grad()
        reference.zero_grad()
        with profiler.scope("forward_and_data"):
            loss = (actual * update).square().sum()
        baseline = (expected * update).square().sum()
        assert torch.equal(loss, baseline)
        with profiler.scope("backward_and_output_collection"):
            loss.backward()
        baseline.backward()
        assert torch.equal(actual.grad, expected.grad)
        step()
        reference.step()
        assert torch.equal(actual, expected)
        assert torch.equal(profiled.state[actual]["momentum_buffer"], reference.state[expected]["momentum_buffer"])
        profiler.end_update(update)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "captured"
    assert summary["observed_updates"] == [2, 3]
    assert summary["activities"] == ["CPU"]
    assert summary["trace_sha256"] == digest(output / "trace.json")
    regions = {region["name"]: region for region in summary["regions"]}
    for name in ("training_update", "forward_and_data", "backward_and_output_collection", "optimizer_step"):
        assert regions[f"osn/{name}"]["calls"] == 2
        assert regions[f"osn/{name}"]["cpu_total_microseconds"] > 0
    trace = json.loads((output / "trace.json").read_text())
    assert sum(event.get("name") == "osn/training_update" for event in trace["traceEvents"]) == 2
    original = (output / "summary.json").read_bytes(), (output / "trace.json").read_bytes()
    profiler.finish()
    assert original == ((output / "summary.json").read_bytes(), (output / "trace.json").read_bytes())


def test_partial_trace_is_incomplete_and_cannot_replace_existing_rank_evidence(tmp_path):
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
