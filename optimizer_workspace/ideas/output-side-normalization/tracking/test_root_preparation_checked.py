# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for exact preservation around a separate restored-root intervention."""

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if Path(__file__).parent.name == "tests":
    from experiment import root_preparation_checked as checked
else:
    import root_preparation_checked as checked


@pytest.fixture
def fast():
    group = SimpleNamespace(
        per=1,
        C=torch.eye(4).unsqueeze(0),
        tol=torch.ones(1),
        basis=SimpleNamespace(SB=torch.ones(1, 8, 4).half(), S2=torch.ones(1, 4, 8).half()),
        basis_step=[20],
        pending=[{"step": 21, "basis": torch.eye(4), "tol": torch.ones(1)}],
        graph=None,
        warm=False,
    )
    engine = SimpleNamespace(
        args=SimpleNamespace(toner_graphs=True, toner_inline=20),
        groups={"d": group},
        begun=False,
        steps=22,
        device="cpu",
    )
    engine._roots = lambda scratch: (scratch.C + 1, None, torch.zeros(1, dtype=torch.bool), None)
    item = SimpleNamespace(input=engine, output=None, momentum=torch.arange(4), steps=22)

    def state():
        return {
            "input": {
                "groups": {
                    "d": {
                        "C": group.C,
                        "tol": group.tol,
                        "SB": group.basis.SB,
                        "S2": group.basis.S2,
                        "basis_step": group.basis_step,
                        "pending": group.pending,
                        "warm": group.warm,
                        "cached_root_graph_active": group.graph is not None,
                    }
                },
                "steps": engine.steps,
            },
            "output": None,
            "momentum": item.momentum,
            "steps": item.steps,
        }

    item.runtime_state = state
    return item


def test_checked_preparation_preserves_numerics_and_rng_and_publishes_only_descriptors(fast):
    before = checked.without_readiness(fast.runtime_state())
    cpu_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    result = checked.checked_preparation(fast, rank=3, step=22)
    checked.exact_tree(checked.without_readiness(fast.runtime_state()), before)
    assert torch.equal(torch.get_rng_state(), cpu_rng) and random.getstate() == python_rng
    assert result["rank"] == 3 and result["checkpoint_step"] == 22 and result["groups"] == ["d"]
    assert result["numerical_state_exact"] and result["torch_rng_exact"] and result["python_rng_exact"]
    assert result["cuda_rng_exact"] is None and not result["scratch_outputs_consumed"]
    assert fast.input.groups["d"].warm and fast.input.groups["d"].graph is None


@pytest.mark.parametrize("fault", ("covariance", "pending", "momentum", "counter", "cpu_rng", "python_rng", "graph"))
def test_checked_preparation_rejects_mutations_and_keeps_readiness_false(fast, fault):
    cpu_rng, python_rng = torch.get_rng_state().clone(), random.getstate()

    def mutate(scratch):
        if fault == "covariance":
            fast.input.groups["d"].C.add_(1)
        elif fault == "pending":
            fast.input.groups["d"].pending[0]["step"] += 1
        elif fault == "momentum":
            fast.momentum.add_(1)
        elif fault == "counter":
            fast.steps += 1
        elif fault == "cpu_rng":
            torch.rand(1)
        elif fault == "python_rng":
            random.random()
        else:
            fast.input.groups["d"].graph = object()
        return scratch.C

    fast.input._roots = mutate
    try:
        with pytest.raises(ValueError):
            checked.checked_preparation(fast, rank=0, step=22)
        assert not fast.input.groups["d"].warm
    finally:
        torch.set_rng_state(cpu_rng)
        random.setstate(python_rng)


@pytest.mark.parametrize("fault", ("step", "engine_step", "output", "missing_input", "disabled"))
def test_wrong_restored_regime_is_refused(fast, fault):
    step = 21 if fault == "step" else 22
    if fault == "engine_step":
        fast.input.steps = 21
    elif fault == "output":
        fast.output = object()
    elif fault == "missing_input":
        fast.input = None
    elif fault == "disabled":
        fast.input.args.toner_graphs = False
    with pytest.raises(ValueError):
        checked.checked_preparation(fast, rank=0, step=step)


def test_only_process_local_readiness_fields_are_excluded(fast):
    state = copy.deepcopy(fast.runtime_state())
    normalized = checked.without_readiness(state)
    assert "warm" in state["input"]["groups"]["d"]
    assert "warm" not in normalized["input"]["groups"]["d"]
    normalized["input"]["groups"]["d"]["pending"][0]["step"] += 1
    assert state["input"]["groups"]["d"]["pending"][0]["step"] == 21
    with pytest.raises(ValueError):
        checked.exact_tree(normalized, checked.without_readiness(state))
