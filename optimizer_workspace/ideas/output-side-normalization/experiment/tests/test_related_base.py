# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from experiment.related_base import MATRIX_KINDS, MatrixBaseline, MuonControl
from toner.fast import muon as MU
from torch import nn


def model_and_config():
    generator = torch.Generator().manual_seed(51)
    blocks = [
        SimpleNamespace(
            **{
                kind: SimpleNamespace(weight=nn.Parameter(torch.randn(3, 4, generator=generator)))
                for kind in MATRIX_KINDS
            }
        )
        for _ in range(2)
    ]
    return SimpleNamespace(body=SimpleNamespace(blocks=blocks)), {
        "optimizer": {
            "lr": 0.02,
            "weight_decay": 0.1,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 10,
            "ns_eps": 1e-5,
            "ns_mode": "f32",
        }
    }


def test_muon_control_updates_and_resume_match_independent_reference():
    model, config = model_and_config()
    control = MuonControl(
        model, config, positions=None, rows_per_microbatch=0, capacity=0, rows_total=0, world=1, rank=0, device="cpu"
    )
    reference_weights = {key: parameter.detach().clone() for key, parameter in control.matrices.items()}
    reference_momentum = {key: torch.zeros_like(parameter) for key, parameter in control.matrices.items()}
    generator = torch.Generator().manual_seed(17)
    for step in range(3):
        control.begin_step(step)
        for key, parameter in control.matrices.items():
            gradient = torch.randn(parameter.shape, generator=generator)
            parameter.grad = gradient
            reference_momentum[key] = 0.95 * reference_momentum[key] + gradient
            source = gradient + 0.95 * reference_momentum[key]
            direction = MU.newton_schulz_batched(source[None], 10, 1e-5, "f32")[0]
            reference_weights[key] -= 0.02 * (direction + 0.1 * reference_weights[key])
        control.step()
        for key, parameter in control.matrices.items():
            torch.testing.assert_close(parameter, reference_weights[key], rtol=1e-5, atol=1e-6)
        if step == 0:
            state = copy.deepcopy(control.runtime_state())
            restored_model = copy.deepcopy(model)
            control = MuonControl(
                restored_model,
                config,
                positions=None,
                rows_per_microbatch=0,
                capacity=0,
                rows_total=0,
                world=1,
                rank=0,
                device="cpu",
            )
            control.restore_runtime_state(state)
    assert control.steps == 3


def test_nonfinite_update_leaves_all_matrix_shapes_unchanged():
    model, config = model_and_config()
    control = MatrixBaseline(model, config, world=1, rank=0, device="cpu")
    before = {key: parameter.detach().clone() for key, parameter in control.matrices.items()}
    changes = {key: torch.ones_like(parameter) for key, parameter in control.owned.items()}
    changes[1, "down"][0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        control.apply_updates(changes)
    for key, parameter in control.matrices.items():
        torch.testing.assert_close(parameter, before[key], rtol=0, atol=0)


def distributed_owner_worker(rank, rendezvous, directory):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    model, config = model_and_config()
    control = MatrixBaseline(model, config, world=2, rank=rank, device="cpu")
    control.apply_updates({key: torch.full_like(parameter, rank + 1.0) for key, parameter in control.owned.items()})
    torch.save({key: parameter.detach() for key, parameter in control.matrices.items()}, directory / f"owner_{rank}.pt")
    dist.destroy_process_group()


def test_owner_updates_are_applied_to_all_replicas(tmp_path):
    model, _ = model_and_config()
    mp.spawn(distributed_owner_worker, args=(f"file://{tmp_path / 'rendezvous'}", tmp_path), nprocs=2, join=True)
    for rank in range(2):
        state = torch.load(tmp_path / f"owner_{rank}.pt", weights_only=True)
        for (layer, kind), parameter in state.items():
            expected = getattr(model.body.blocks[layer], kind).weight + (layer + 1)
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
