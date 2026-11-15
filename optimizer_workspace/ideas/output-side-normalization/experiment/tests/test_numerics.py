# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from experiment.capture import OutputLinear, OutputMoments
from experiment.factors import OutputSettings, direction_and_decay, inverse_factors
from experiment.optimizer import down_update
from experiment.records import checkpoint_identity, restore_checkpoint, save_checkpoint
from toner.fast.muon import newton_schulz_batched
from torch import nn


def polar(matrix):
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


@pytest.mark.parametrize("shape", [(4, 4), (3, 7), (7, 3)])
def test_identity_factors_preserve_muon_direction_and_decay(shape):
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(*shape, generator=generator, dtype=torch.float64)
    momentum = torch.randn(*shape, generator=generator, dtype=torch.float64)
    left = torch.eye(shape[0], dtype=torch.float64)
    right = torch.eye(shape[1], dtype=torch.float64)
    for arm in ("muont", "outer_left_down", "inner_outer_down", "down_hybrid"):
        direction, decay = direction_and_decay(
            weight,
            momentum,
            arm=arm,
            polar=polar,
            right_inner=right,
            right_outer=right,
            left_inner=left,
            left_outer=left,
        )
        torch.testing.assert_close(direction, polar(momentum), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(decay, weight, rtol=1e-12, atol=1e-12)


def test_output_roots_match_known_eigenvalues_and_scale_invariance():
    generator = torch.Generator().manual_seed(1)
    q, _ = torch.linalg.qr(torch.randn(5, 5, generator=generator, dtype=torch.float64))
    values = torch.tensor([0.0, 0.1, 0.5, 1.0, 8.0], dtype=torch.float64)
    moment = (q * values) @ q.T
    settings = OutputSettings()
    inner, outer = inverse_factors(moment, settings)
    expected_values = 5 * values / values.sum() + settings.damping
    expected_inner = (q * expected_values.pow(-0.5)) @ q.T
    expected_outer = (q * expected_values.pow(-0.125)) @ q.T
    expected_outer *= 5**0.5 / expected_outer.norm()
    torch.testing.assert_close(inner, expected_inner, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(outer, expected_outer, rtol=1e-10, atol=1e-10)
    scaled_inner, scaled_outer = inverse_factors(moment * 17, settings)
    torch.testing.assert_close(scaled_inner, inner, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(scaled_outer, outer, rtol=1e-10, atol=1e-10)
    empty_inner, empty_outer = inverse_factors(torch.zeros_like(moment), settings)
    torch.testing.assert_close(empty_inner, torch.eye(5, dtype=torch.float64))
    torch.testing.assert_close(empty_outer, torch.eye(5, dtype=torch.float64))


def test_hybrid_transpose_orientation_and_left_shaped_decay():
    generator = torch.Generator().manual_seed(23)
    gradient = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    weight = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    moment = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    inner, outer = inverse_factors(moment @ moment.T, OutputSettings())
    direction, decay = direction_and_decay(
        weight,
        gradient,
        arm="down_hybrid",
        polar=polar,
        left_inner=inner,
        left_outer=outer,
    )
    u, _, vh = torch.linalg.svd(gradient.T @ inner, full_matrices=False)
    expected_direction = (u @ vh @ outer).T
    torch.testing.assert_close(direction, expected_direction, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(decay, outer @ outer @ weight, rtol=1e-12, atol=1e-12)
    # Exact polar gives a wide full-row-rank direction equal row energy before outer shaping.
    torch.testing.assert_close(direction @ direction.T, outer @ outer, rtol=1e-12, atol=1e-12)


def test_output_capture_corrects_loss_scaling_without_changing_parameter_gradients():
    generator = torch.Generator().manual_seed(11)
    linear = nn.Linear(4, 3, bias=False)
    reference = copy.deepcopy(linear)
    sums = torch.zeros(3, 3)
    counts = torch.zeros((), dtype=torch.int64)
    positions = torch.tensor([0, 2, 4])
    scaling = torch.tensor(12.0)
    observed = OutputLinear(linear, sums, counts, positions, scaling)
    x = torch.randn(2, 5, 4, generator=generator)
    target = torch.randn(2, 5, 3, generator=generator)
    loss = ((observed(x) - target).square().sum()) / scaling
    reference_loss = ((reference(x) - target).square().sum()) / scaling
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(linear.weight.grad, reference.weight.grad, rtol=1e-6, atol=1e-6)
    delta = 2 * (reference(x).detach() - target)
    rows = delta[:, positions].reshape(-1, 3)
    torch.testing.assert_close(sums, rows.T @ rows, rtol=1e-6, atol=1e-6)
    assert counts.item() == 6
    # Evaluation forward must not collect moments.
    observed(x).detach()
    assert counts.item() == 6


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_compiled_capture_pools_microbatches_and_keeps_gradients(backend):
    generator = torch.Generator().manual_seed(33)
    linear = nn.Linear(4, 3, bias=False)
    reference = copy.deepcopy(linear)
    sums, counts = torch.zeros(3, 3), torch.zeros((), dtype=torch.int64)
    positions, scaling = torch.tensor([0, 2]), torch.tensor(5.0)
    observed = OutputLinear(linear, sums, counts, positions, scaling)
    compiled = torch.compile(observed, backend=backend, fullgraph=True)
    expected_sum = torch.zeros_like(sums)
    for _ in range(2):
        x = torch.randn(1, 4, 4, generator=generator)
        target = torch.randn(1, 4, 3, generator=generator)
        (compiled(x).sub(target).square().sum() / 5).backward()
        (reference(x).sub(target).square().sum() / 5).backward()
        delta = 2 * (reference(x).detach() - target)
        rows = delta[:, positions].reshape(-1, 3)
        expected_sum += rows.T @ rows
    torch.testing.assert_close(linear.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sums, expected_sum, rtol=1e-5, atol=1e-6)
    assert counts.item() == 4


def test_moment_ema_uses_pooled_rows_before_trace_normalization_and_caches_roots():
    model = nn.Module()
    model.body = nn.Module()
    block = nn.Module()
    block.down = nn.Linear(4, 3, bias=False)
    model.body.blocks = nn.ModuleList([block])
    moments = OutputMoments(
        model,
        [0, 2],
        OutputSettings(momentum=0.5, refresh=2),
        world=1,
        rank=0,
        device="cpu",
    )
    moments.begin(4)
    x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(72))
    model.body.blocks[0].down(x).square().sum().div(4).backward()
    raw = moments.sums.clone() / 4
    moments.update(4)
    torch.testing.assert_close(moments.moment, raw * 0.5)
    roots = moments.inner.clone()
    moments.begin(4)
    model.body.blocks[0].down(x * 2).square().sum().div(4).backward()
    second = moments.sums.clone() / 4
    moments.update(4)
    torch.testing.assert_close(moments.moment, raw * 0.25 + second * 0.5)
    torch.testing.assert_close(moments.inner, roots)


def test_checkpoint_continuation_preserves_optimizer_and_rng(tmp_path):
    torch.manual_seed(9)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    config = {"training": {"global_batch_sequences": 8}}
    data_receipt = tmp_path / "data.json"
    data_receipt.write_text('{"dataset": "synthetic_cpu_check"}\n')
    identity = checkpoint_identity(config, data_receipt)

    def step(network, opt):
        opt.zero_grad()
        network(torch.randn(2, 4)).square().mean().backward()
        opt.step()

    step(model, optimizer)
    save_checkpoint(tmp_path, model, [optimizer], None, None, step=1, config=config, rank=0, world=1, identity=identity)
    step(model, optimizer)
    expected = copy.deepcopy(model.state_dict())
    restored = nn.Linear(4, 3)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    resumed = restore_checkpoint(
        tmp_path / "checkpoints" / "step_000001",
        restored,
        [restored_optimizer],
        None,
        None,
        config=config,
        rank=0,
        world=1,
        identity=identity,
    )
    step(restored, restored_optimizer)
    assert resumed == 1
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_checkpoint_changed_data_receipt_refuses_resume_before_loading_weights(tmp_path):
    config = {"training": {"global_batch_sequences": 8}}
    data = tmp_path / "data.json"
    data.write_text('{"tokens_sha256": "original"}\n')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    original_identity = checkpoint_identity(config, data)
    save_checkpoint(
        tmp_path,
        model,
        [optimizer],
        None,
        None,
        step=1,
        config=config,
        rank=0,
        world=1,
        identity=original_identity,
    )
    restored = nn.Linear(4, 3)
    before = copy.deepcopy(restored.state_dict())
    data.write_text('{"tokens_sha256": "changed"}\n')
    with pytest.raises(ValueError, match="prepared-data identity"):
        restore_checkpoint(
            tmp_path / "checkpoints/step_000001",
            restored,
            [optimizer],
            None,
            None,
            config=config,
            rank=0,
            world=1,
            identity=checkpoint_identity(config, data),
        )
    for name, parameter in restored.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_checkpoint_wrong_rank_ownership_refuses_resume_before_loading_weights(tmp_path):
    config = {"training": {"global_batch_sequences": 8}}
    data = tmp_path / "data.json"
    data.write_text('{"dataset": "synthetic_ownership_check"}\n')
    identity = checkpoint_identity(config, data)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    save_checkpoint(
        tmp_path,
        model,
        [optimizer],
        None,
        None,
        step=1,
        config=config,
        rank=0,
        world=1,
        identity=identity,
    )
    directory = tmp_path / "checkpoints/step_000001"
    state = torch.load(directory / "rank_0.pt", weights_only=False)
    state["rank"] = 1
    torch.save(state, directory / "rank_0.pt")
    restored = nn.Linear(4, 3)
    before = copy.deepcopy(restored.state_dict())
    with pytest.raises(ValueError, match="ownership"):
        restore_checkpoint(
            directory,
            restored,
            [optimizer],
            None,
            None,
            config=config,
            rank=0,
            world=1,
            identity=identity,
        )
    for name, parameter in restored.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_checkpoint_output_moment_roots_and_accumulated_backward_continue(tmp_path):
    config = {"training": {"global_batch_sequences": 8}}
    data = tmp_path / "data.json"
    data.write_text('{"dataset": "synthetic_output_checkpoint"}\n')
    identity = checkpoint_identity(config, data)

    def construct():
        model = nn.Module()
        model.body = nn.Module()
        block = nn.Module()
        block.down = nn.Linear(4, 3, bias=False)
        model.body.blocks = nn.ModuleList([block])
        output = OutputMoments(model, [0, 2], OutputSettings(refresh=2), world=1, rank=0, device="cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        return model, output, optimizer

    def update(model, output, optimizer):
        optimizer.zero_grad()
        output.begin(8)
        for _ in range(2):
            x = torch.randn(1, 4, 4)
            model.body.blocks[0].down(x).square().sum().div(8).backward()
        output.update(4)
        optimizer.step()

    torch.manual_seed(34)
    model, output, optimizer = construct()
    for _ in range(2):
        update(model, output, optimizer)
    save_checkpoint(
        tmp_path,
        model,
        [optimizer],
        None,
        output,
        step=2,
        config=config,
        rank=0,
        world=1,
        identity=identity,
    )
    for _ in range(2):
        update(model, output, optimizer)
    expected_parameters = copy.deepcopy(model.state_dict())
    expected_output = copy.deepcopy(output.state_dict())
    restored, restored_output, restored_optimizer = construct()
    restore_checkpoint(
        tmp_path / "checkpoints/step_000002",
        restored,
        [restored_optimizer],
        None,
        restored_output,
        config=config,
        rank=0,
        world=1,
        identity=identity,
    )
    for _ in range(2):
        update(restored, restored_output, restored_optimizer)
    for name, parameter in restored.state_dict().items():
        torch.testing.assert_close(parameter, expected_parameters[name], rtol=0, atol=0)
    for name in ("moment", "inner", "outer"):
        torch.testing.assert_close(restored_output.state_dict()[name], expected_output[name], rtol=0, atol=0)
    assert restored_output.steps == expected_output["steps"]
    assert restored_output.root_step == expected_output["root_step"]


def distributed_moment_worker(rank, rendezvous, directory):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    model = nn.Module()
    model.body = nn.Module()
    model.body.blocks = nn.ModuleList([nn.Module(), nn.Module()])
    for block in model.body.blocks:
        block.down = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            block.down.weight.fill_(0.25)
    moments = OutputMoments(
        model,
        [0, 2],
        OutputSettings(momentum=0.5),
        world=2,
        rank=rank,
        device="cpu",
    )
    moments.begin(7)
    x = torch.ones(rank + 1, 3, 3) * (rank + 1)
    for block in model.body.blocks:
        (block.down(x).square().sum() / 7).backward()
    moments.update(2 * (rank + 1))
    torch.save(moments.state_dict(), directory / f"output_{rank}.pt")
    dist.destroy_process_group()


def test_distributed_output_statistics_pool_unequal_rank_counts_before_ema(tmp_path):
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    mp.spawn(distributed_moment_worker, args=(rendezvous, tmp_path), nprocs=2, join=True)
    # Rank 0: two sampled rows with delta=(1.5,1.5).
    # Rank 1: four sampled rows with delta=(3,3).
    expected = torch.full((1, 2, 2), 0.5 * (2 * 1.5**2 + 4 * 3**2) / 6)
    for rank in range(2):
        state = torch.load(tmp_path / f"output_{rank}.pt", weights_only=True)
        torch.testing.assert_close(state["moment"], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("arm", ["outer_left_down", "inner_outer_down", "down_hybrid"])
def test_actual_down_update_matches_independent_finite_ns_equations(arm):
    generator = torch.Generator().manual_seed(41)
    weight = torch.randn(2, 3, 7, generator=generator)
    gradient = torch.randn(2, 3, 7, generator=generator)
    momentum = torch.randn(2, 3, 7, generator=generator)
    left = torch.randn(2, 3, 9, generator=generator)
    right = torch.randn(2, 7, 9, generator=generator)
    li, lo = inverse_factors(left @ left.mT, OutputSettings())
    ri, ro = inverse_factors(right @ right.mT, OutputSettings())
    args = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32")
    actual_momentum, actual = down_update(
        weight,
        gradient,
        momentum,
        arm=arm,
        left_inner=li,
        left_outer=lo,
        right_inner=ri,
        right_outer=ro,
        args=args,
        lr=0.02,
        weight_decay=0.1,
        right_product=torch.matmul,
        left_product=torch.matmul,
    )
    expected_momentum = 0.95 * momentum + gradient
    source = gradient + 0.95 * expected_momentum
    if arm == "down_hybrid":
        source = li @ source
        direction = lo @ newton_schulz_batched(source, 10, 1e-5, "f32")
        decay = lo @ lo @ weight
    else:
        source = source @ ri
        if arm == "inner_outer_down":
            source = li @ source
        direction = lo @ newton_schulz_batched(source, 10, 1e-5, "f32") @ ro
        decay = weight @ ro @ ro
    torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
    torch.testing.assert_close(actual, -0.02 * (direction + 0.1 * decay), rtol=1e-5, atol=1e-6)
