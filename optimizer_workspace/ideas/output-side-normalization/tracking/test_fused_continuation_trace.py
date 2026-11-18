# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check actual CPU arithmetic, copied storage and immutable diagnostic evidence."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from experiment import block_toner

if Path(__file__).parent.name == "tests":
    from experiment import fused_attribution_validation_checks as checks
    from experiment.fused_continuation_trace import ContinuationTrace
else:
    from fused_continuation_trace import ContinuationTrace

    spec = importlib.util.spec_from_file_location(
        "experiment.fused_attribution_validation_checks",
        Path(__file__).with_name("fused_attribution_validation_checks.py"),
    )
    checks = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = checks
    spec.loader.exec_module(checks)


@pytest.mark.parametrize("trace_on", (False, True))
def test_trace_preserves_update_gradients_momentum_and_saved_storage(tmp_path, trace_on):
    parameter = torch.nn.Parameter(torch.tensor([0.5, -1.0], dtype=torch.float64))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.SGD([parameter], lr=0.02, momentum=0.95)
    baseline = torch.optim.SGD([reference], lr=0.02, momentum=0.95)
    trace = ContinuationTrace(tmp_path / "rank0", rank=0, provenance={"candidate": "paired-source"})
    step = trace.wrap(optimizer.step, "optimizer") if trace_on else optimizer.step
    saved = {}
    for update in range(21, 25):
        trace.begin_update(update) if trace_on else None
        optimizer.zero_grad()
        baseline.zero_grad()
        loss, expected_loss = (parameter * update).square().sum(), (reference * update).square().sum()
        loss.backward()
        expected_loss.backward()
        assert torch.equal(parameter.grad, reference.grad)
        if trace_on:
            trace.record("gradient", parameter.grad)
        saved[update] = parameter.grad.detach().clone()
        step()
        baseline.step()
        assert torch.equal(parameter, reference)
        assert torch.equal(optimizer.state[parameter]["momentum_buffer"], baseline.state[reference]["momentum_buffer"])
        trace.end_update(update) if trace_on else None
    if not trace_on:
        assert not (tmp_path / "rank0").exists()
        return
    for update in (22, 23, 24):
        path = tmp_path / f"rank0/update_{update:06d}.pt"
        value = torch.load(path, weights_only=False)
        assert torch.equal(value["records"][0]["values"], saved[update])
        receipt = json.loads(path.with_suffix(".json").read_text())
        assert receipt["trace_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert receipt["rank"] == 0 and receipt["update"] == update
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        trace.begin_update(24)
    assert path.read_bytes() == original


def test_trace_copies_do_not_pollute_strict_applied_outer_observation(tmp_path):
    observer = checks.AppliedOuterTrace()
    trace = ContinuationTrace(tmp_path / "rank3", rank=3, provenance={})
    trace.fast = SimpleNamespace(update_verifier=SimpleNamespace(observation=observer))
    trace.begin_update(23)
    vector = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    direction = torch.arange(16.0).reshape(1, 4, 4)
    with observer:
        applied = torch.stack(list(vector.unbind(0)))
        trace.record("applied", applied)
        normalized = block_toner.stack_unit_rms(applied)
        result = direction * normalized[:, :, None]
        trace.record("result", result)
    assert observer.enabled
    observer.enabled = False
    observer.require_applied(applied, result)
    trace.end_update(23)
    saved = torch.load(tmp_path / "rank3/update_000023.pt", weights_only=False)
    assert torch.equal(saved["records"][0]["values"], applied)
    assert torch.equal(saved["records"][1]["values"], result)
