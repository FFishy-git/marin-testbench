# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import io
from dataclasses import asdict
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from experiment.block_toner import INPUT_KIND, INPUT_KINDS, ROW_SCALE
from experiment.go_muon import (
    GoMuon,
    GoMuonSettings,
    GoMuonStatistics,
    compact_support_polar,
    inverse_quarter_factor,
    matched_direction,
    oas_factor,
)
from experiment.related_base import MATRIX_KINDS
from toner.fast.muon import newton_schulz_batched
from torch import nn
from torch.nn import functional as F


def reference_oas(moment, count):
    width = moment.shape[0]
    trace = torch.trace(moment)
    squared_trace = torch.trace(moment @ moment)
    denominator = (count + 1 - 2 / width) * (squared_trace - trace**2 / width)
    coefficient = (
        min(1.0, max(0.0, ((1 - 2 / width) * squared_trace + trace**2).item() / denominator.item()))
        if denominator.item() > 0
        else 1.0
    )
    return (1 - coefficient) * moment + coefficient * trace / width * torch.eye(width, dtype=moment.dtype)


def reference_quarter(moment, *, normalize=False, damping=0.0):
    width = moment.shape[0]
    effective = moment * width / torch.trace(moment) if normalize else moment
    effective = effective + damping * torch.eye(width, dtype=moment.dtype)
    values, vectors = torch.linalg.eigh(effective)
    return vectors @ torch.diag(values.pow(-0.25)) @ vectors.mT


@pytest.mark.parametrize("width,count", [(2, 1), (2, 50), (5, 8), (5, 1000)])
@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_oas_factor_matches_n_aware_paper_equation(width, count, scale):
    generator = torch.Generator().manual_seed(311)
    rows = torch.randn(9, width, dtype=torch.float64, generator=generator)
    rows[:, 0] *= 4
    moment = rows.mT @ rows / 9
    expected = reference_oas(moment, count) * scale
    actual = oas_factor(moment * scale, count)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=scale * 1e-12)


def test_oas_isotropic_and_zero_observations_have_finite_quarter_roots():
    moment = torch.eye(4, dtype=torch.float64) * 3
    torch.testing.assert_close(oas_factor(moment, 2), moment, rtol=0, atol=0)
    zero = torch.zeros_like(moment)
    torch.testing.assert_close(oas_factor(zero, 2), zero, rtol=0, atol=0)
    root = inverse_quarter_factor(zero, relative_floor=1e-7, absolute_floor=1e-12)
    torch.testing.assert_close(root, torch.eye(4, dtype=torch.float64) * 1000, rtol=1e-12, atol=1e-12)


def test_quarter_root_floors_roundoff_negative_and_null_eigenvalues():
    moment = torch.diag(torch.tensor([-1e-12, 0.0, 2.0, 8.0], dtype=torch.float64))
    root = inverse_quarter_factor(moment, relative_floor=1e-7, absolute_floor=1e-15)
    expected = torch.diag(torch.tensor([8e-7, 8e-7, 2.0, 8.0], dtype=torch.float64).pow(-0.25))
    torch.testing.assert_close(root, expected, rtol=1e-12, atol=1e-12)


def test_fp32_moment_quarter_root_preserves_ridge_on_rank_deficient_geometry():
    generator = torch.Generator().manual_seed(116)
    rows = torch.randn(2, 5, generator=generator)
    rows[:, 0] *= 8
    moment = rows.mT @ rows
    actual = inverse_quarter_factor(
        moment, relative_floor=1e-7, absolute_floor=1e-30, normalize_trace=True, damping=0.001
    )
    expected = reference_quarter(moment.double(), normalize=True, damping=0.001).float()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(4, 4), (3, 7), (7, 3)])
def test_matched_direction_matches_exact_svd_quarter_sandwich_and_graft(shape):
    generator = torch.Generator().manual_seed(711)
    source = torch.randn(*shape, dtype=torch.float64, generator=generator)
    samples = [torch.randn(width + 3, width, dtype=torch.float64, generator=generator) for width in shape]
    output, activation = (rows.mT @ rows for rows in samples)
    left = inverse_quarter_factor(output, relative_floor=1e-7, absolute_floor=1e-30)
    right = inverse_quarter_factor(activation, relative_floor=1e-7, absolute_floor=1e-30)
    expected_left, expected_right = reference_quarter(output), reference_quarter(activation)
    transformed = expected_left @ source @ expected_right
    u, _, vh = torch.linalg.svd(transformed, full_matrices=False)
    raw = expected_left @ (u @ vh) @ expected_right
    expected = raw / raw.norm() * min(shape) ** 0.5
    actual = matched_direction(source, left, right, polar_method="exact_svd")
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.norm(), torch.tensor(min(shape) ** 0.5, dtype=torch.float64))


@pytest.mark.parametrize("shape", [(1024, 256), (256, 1024)])
def test_fp32_rectangular_graft_preserves_norm_measured_in_fp64(shape):
    generator = torch.Generator().manual_seed(953)
    source = torch.randn(shape, generator=generator)
    left = torch.diag(torch.rand(shape[0], generator=generator) + 0.5)
    right = torch.diag(torch.rand(shape[1], generator=generator) + 0.5)
    actual = matched_direction(source, left, right, polar_method="exact_svd")
    u, _, vh = torch.linalg.svd(left.double() @ source.double() @ right.double(), full_matrices=False)
    raw = left.double() @ (u @ vh) @ right.double()
    expected = (raw * (min(shape) ** 0.5 / raw.norm())).float()
    torch.testing.assert_close(actual, expected, rtol=0.003, atol=5e-5)
    torch.testing.assert_close(
        actual.double().norm(), torch.tensor(min(shape) ** 0.5, dtype=torch.float64), rtol=1e-6, atol=1e-6
    )


def test_finite_polar_control_keeps_matching_factors_and_final_graft():
    generator = torch.Generator().manual_seed(89)
    source = torch.randn(3, 5, generator=generator)
    left = torch.diag(torch.tensor([0.2, 1.0, 3.0]))
    right = torch.diag(torch.tensor([2.0, 0.5, 0.8, 1.2, 4.0]))
    central = newton_schulz_batched((left @ source @ right)[None], 10, 1e-5, "f32")[0]
    raw = left @ central @ right
    expected = raw * (3**0.5 / raw.norm())
    actual = matched_direction(source, left, right, polar_method="newton_schulz")
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.norm(), torch.tensor(3**0.5), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("polar_method", ["exact_svd", "newton_schulz"])
def test_zero_source_skips_completion_and_graft(polar_method):
    source = torch.zeros(5, 3)
    actual = matched_direction(source, torch.eye(5) * 2, torch.eye(3) * 3, polar_method=polar_method)
    torch.testing.assert_close(actual, source, rtol=0, atol=0)
    torch.testing.assert_close(compact_support_polar(source), source, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 3)])
def test_rank_deficient_polar_keeps_semi_orthogonal_null_completion(shape):
    source = torch.zeros(*shape, dtype=torch.float64)
    source[0, 0] = 2.0
    polar = compact_support_polar(source)
    expected_gram = torch.eye(min(shape), dtype=torch.float64)
    gram = polar @ polar.mT if shape[0] <= shape[1] else polar.mT @ polar
    torch.testing.assert_close(gram, expected_gram, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close((source * polar).sum(), torch.tensor(2.0, dtype=torch.float64))
    actual = matched_direction(
        source,
        torch.eye(shape[0], dtype=torch.float64),
        torch.eye(shape[1], dtype=torch.float64),
        polar_method="exact_svd",
    )
    torch.testing.assert_close(actual, polar, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.norm(), torch.tensor(min(shape) ** 0.5, dtype=torch.float64))


def test_sparse_statistics_sample_and_hold_instead_of_per_step_ema():
    sparse = GoMuonStatistics(
        GoMuonSettings(statistic_mode="paper_sparse", polar_method="exact_svd", backward_decay=0.8),
        decay=0.5,
        damping=0.001,
        refresh=3,
    )
    control = GoMuonStatistics(GoMuonSettings(), decay=0.5, damping=0.001, refresh=3)
    input_key, output_key = (0, "attn"), (0, "q")
    inputs = [torch.diag(torch.tensor(value, dtype=torch.float64)) for value in ((1, 6), (8, 1), (2, 9), (7, 2))]
    outputs = [torch.diag(torch.tensor(value, dtype=torch.float64)) for value in ((3, 1), (2, 4), (9, 2), (1, 8))]
    control_input = torch.zeros(2, 2, dtype=torch.float64)
    control_output = control_input.clone()
    first_roots = None
    for step, (activation, backward) in enumerate(zip(inputs, outputs, strict=True)):
        sparse.update(step, {input_key: activation}, {output_key: backward}, 20)
        control.update(step, {input_key: activation}, {output_key: backward}, 20)
        control_input = 0.5 * control_input + 0.5 * activation
        control_output = 0.5 * control_output + 0.5 * backward
        if step == 0:
            first_roots = control.state_dict()["input_factors"][input_key]
        if step in (1, 2):
            torch.testing.assert_close(sparse.state_dict()["input_moments"][input_key], reference_oas(inputs[0], 20))
            torch.testing.assert_close(sparse.state_dict()["output_moments"][output_key], reference_oas(outputs[0], 20))
            torch.testing.assert_close(control.state_dict()["input_factors"][input_key], first_roots)
    sparse_state, control_state = sparse.state_dict(), control.state_dict()
    expected_backward = 0.8**3 * reference_oas(outputs[0], 20) + (1 - 0.8**3) * reference_oas(outputs[3], 20)
    torch.testing.assert_close(sparse_state["input_moments"][input_key], reference_oas(inputs[3], 20))
    torch.testing.assert_close(sparse_state["output_moments"][output_key], expected_backward)
    torch.testing.assert_close(control_state["input_moments"][input_key], control_input)
    torch.testing.assert_close(control_state["output_moments"][output_key], control_output)
    torch.testing.assert_close(
        sparse_state["output_factors"][output_key], reference_quarter(expected_backward), rtol=1e-12, atol=1e-12
    )
    assert sparse_state["observations"] == 2
    assert control_state["observations"] == 4
    assert sparse_state["last_root_step"] == control_state["last_root_step"] == 3


class CpuBlock(nn.Module):
    """Small real linear/SwiGLU graph with the production input-buffer contract."""

    def __init__(self, capacity):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(3, 3, bias=False) for _ in range(4))
        self.gate, self.up = (nn.Linear(3, 5, bias=False) for _ in range(2))
        self.down = nn.Linear(5, 3, bias=False)
        for kind in INPUT_KINDS:
            self.register_buffer(
                f"input_rows_{kind}",
                torch.zeros(capacity, 5 if kind == "down" else 3, dtype=torch.bfloat16),
                persistent=False,
            )
        self.last_inputs, self.last_outputs = {}, {}

    def forward(self, x, at, positions):
        outputs = {kind: getattr(self, kind)(x) for kind in ("q", "k", "v")}
        attention = outputs["q"] + outputs["k"] + outputs["v"]
        outputs["o"] = self.o(attention)
        middle = x + outputs["o"]
        outputs["gate"], outputs["up"] = self.gate(middle), self.up(middle)
        down_input = F.silu(outputs["gate"]) * outputs["up"]
        outputs["down"] = self.down(down_input)
        inputs = {"attn": x, "o": attention, "mlp": middle, "down": down_input}
        with torch.no_grad():
            for kind, rows in inputs.items():
                getattr(self, f"input_rows_{kind}").index_copy_(
                    0, at, rows[:, positions].reshape(-1, rows.shape[-1]).to(torch.bfloat16)
                )
        for output in outputs.values():
            if output.requires_grad:
                output.retain_grad()
        self.last_inputs, self.last_outputs = inputs, outputs
        return middle + outputs["down"]


class CpuBody(nn.Module):
    def __init__(self, *, layers=1, capacity=4):
        super().__init__()
        self.blocks = nn.ModuleList(CpuBlock(capacity) for _ in range(layers))
        self.register_buffer("stat_at", torch.arange(2), persistent=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x, self.stat_at, [0, 2])
        return x


class CpuModel(nn.Module):
    def __init__(self, *, layers=1):
        super().__init__()
        self.body = CpuBody(layers=layers)

    def forward(self, x):
        return self.body(x)


def baseline_config(settings, *, refresh=2):
    return {
        "optimizer": {
            "lr": 0.02,
            "weight_decay": 0.1,
            "momentum": 0.8,
            "nesterov": True,
            "ns_steps": 10,
            "ns_eps": 1e-5,
            "ns_mode": "f32",
        },
        "statistics": {"momentum": 0.5, "damping": 0.001},
        "root": {"refresh": refresh},
        "experiment": {"related_baseline": {"settings": asdict(settings)}},
    }


def baseline(model, settings, *, world=1, rank=0):
    return GoMuon(
        model,
        baseline_config(settings),
        positions=[0, 2],
        rows_per_microbatch=2,
        capacity=4,
        rows_total=4 * world,
        world=world,
        rank=rank,
        device="cpu",
    )


def backward_step(model, optimizer, inputs, *, step):
    model.zero_grad(set_to_none=True)
    optimizer.begin_step(step)
    for index, x in enumerate(inputs):
        optimizer.select_microbatch(index)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(x).float().square().sum().div(2).backward()
    optimizer.finish_backward()
    optimizer.step()


def test_all_projection_capture_is_gradient_transparent_and_pools_microbatches():
    torch.manual_seed(51)
    model = CpuModel()
    reference = copy.deepcopy(model)
    optimizer = baseline(model, GoMuonSettings(polar_method="exact_svd"))
    optimizer.begin_step(0)
    generator = torch.Generator().manual_seed(120)
    inputs = [torch.randn(1, 3, 3, generator=generator).to(torch.bfloat16) for _ in range(2)]
    reference_inputs = {kind: [] for kind in INPUT_KINDS}
    reference_outputs = {kind: [] for kind in MATRIX_KINDS}
    for index, x in enumerate(inputs):
        optimizer.select_microbatch(index)
        reference.body.stat_at.copy_(torch.arange(index * 2, (index + 1) * 2))
        cotangent = torch.randn(1, 3, 3, generator=generator).to(torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual, expected = model(x), reference(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward(cotangent)
        expected.backward(cotangent)
        block = reference.body.blocks[0]
        for kind, value in block.last_inputs.items():
            reference_inputs[kind].append(value[:, [0, 2]].reshape(2, -1).float())
        for kind, value in block.last_outputs.items():
            reference_outputs[kind].append(value.grad[:, [0, 2]].reshape(2, -1).float())
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
    optimizer.finish_backward()
    state = optimizer.runtime_state()["statistics"]
    for kind, rows in reference_inputs.items():
        pooled = torch.cat(rows)
        torch.testing.assert_close(state["input_moments"][0, kind], 0.5 * pooled.mT @ pooled / 4)
    for kind, rows in reference_outputs.items():
        pooled = torch.cat(rows)
        torch.testing.assert_close(optimizer.output_rows[0, kind].float() / ROW_SCALE, pooled, rtol=0, atol=0)
        torch.testing.assert_close(state["output_moments"][0, kind], 0.5 * pooled.mT @ pooled / 4)
    before = copy.deepcopy(optimizer.runtime_state())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(torch.zeros(1, 3, 3, dtype=torch.bfloat16))
    after = optimizer.runtime_state()
    for name in ("input_moments", "output_moments", "input_factors", "output_factors"):
        for key, tensor in before["statistics"][name].items():
            torch.testing.assert_close(after["statistics"][name][key], tensor, rtol=0, atol=0)
    for kind, rows in reference_outputs.items():
        torch.testing.assert_close(optimizer.output_rows[0, kind].float() / ROW_SCALE, torch.cat(rows), rtol=0, atol=0)


@pytest.mark.parametrize("statistic_mode", ["ema_per_step", "paper_sparse"])
def test_optimizer_applies_expected_momentum_graft_shape_scale_and_isotropic_adapter(statistic_mode):
    torch.manual_seed(80)
    model = CpuModel()
    settings = GoMuonSettings(statistic_mode=statistic_mode, polar_method="exact_svd")
    optimizer = baseline(model, settings)
    optimizer.begin_step(0)
    generator = torch.Generator().manual_seed(97)
    for index in range(2):
        optimizer.select_microbatch(index)
        x = torch.randn(1, 3, 3, generator=generator).to(torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(x).float().square().sum().backward()
    optimizer.finish_backward()
    state = optimizer.runtime_state()["statistics"]
    expected_weights, expected_momentum = {}, {}
    for key, parameter in optimizer.matrices.items():
        gradient = parameter.grad.float()
        expected_momentum[key] = (1 - 0.8) * gradient if statistic_mode == "paper_sparse" else gradient
        source = expected_momentum[key] + 0.8 * expected_momentum[key]
        common = statistic_mode == "ema_per_step"
        left = reference_quarter(state["output_moments"][key].double(), normalize=common, damping=0.001 if common else 0)
        right = reference_quarter(
            state["input_moments"][key[0], INPUT_KIND[key[1]]].double(), normalize=common, damping=0.001 if common else 0
        )
        u, _, vh = torch.linalg.svd(left @ source.double() @ right, full_matrices=False)
        raw = left @ (u @ vh) @ right
        direction = (raw * (min(parameter.shape) ** 0.5 / raw.norm())).float()
        scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
        expected_weights[key] = parameter.detach().clone() - 0.02 * (scale * direction + 0.1 * parameter.detach())
    optimizer.step()
    for key, parameter in optimizer.matrices.items():
        torch.testing.assert_close(parameter.detach(), expected_weights[key], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(optimizer.runtime_state()["base"]["momentum"][key], expected_momentum[key])
    assert optimizer.steps == 1
    assert optimizer.last_metrics["go_muon_root_age"] == 0


@pytest.mark.parametrize("statistic_mode", ["ema_per_step", "paper_sparse"])
def test_checkpoint_resume_matches_continuous_updates_across_refreshes(statistic_mode):
    torch.manual_seed(13)
    model = CpuModel()
    settings = GoMuonSettings(statistic_mode=statistic_mode, polar_method="exact_svd")
    optimizer = baseline(model, settings)
    generator = torch.Generator().manual_seed(140)
    batches = [[torch.randn(1, 3, 3, generator=generator).to(torch.bfloat16) for _ in range(2)] for _ in range(5)]
    for step in range(2):
        backward_step(model, optimizer, batches[step], step=step)
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.runtime_state()}
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    resumed_model = CpuModel()
    resumed_model.load_state_dict(saved["model"])
    resumed_optimizer = baseline(resumed_model, settings)
    resumed_optimizer.restore_runtime_state(saved["optimizer"])
    for step in range(2, 5):
        backward_step(model, optimizer, batches[step], step=step)
        backward_step(resumed_model, resumed_optimizer, batches[step], step=step)
    for parameter, restored in zip(model.parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(parameter.detach(), restored.detach(), rtol=0, atol=0)
    continuous, resumed = optimizer.runtime_state(), resumed_optimizer.runtime_state()
    for key, value in continuous["base"]["momentum"].items():
        torch.testing.assert_close(resumed["base"]["momentum"][key], value, rtol=0, atol=0)
    for name in ("input_moments", "output_moments", "input_factors", "output_factors"):
        for key, value in continuous["statistics"][name].items():
            torch.testing.assert_close(resumed["statistics"][name][key], value, rtol=0, atol=0)
    assert continuous["base"]["steps"] == resumed["base"]["steps"] == 5
    assert continuous["statistics"]["last_root_step"] == resumed["statistics"]["last_root_step"] == 4
    assert continuous["statistics"]["observations"] == resumed["statistics"]["observations"]


def distributed_pooling_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    torch.manual_seed(27)
    model = CpuModel(layers=2)
    reference = copy.deepcopy(model)
    optimizer = baseline(model, GoMuonSettings(), world=2, rank=rank)
    optimizer.begin_step(0)
    generator = torch.Generator().manual_seed(32 + rank)
    input_rows = {(layer, kind): [] for layer in range(2) for kind in INPUT_KINDS}
    output_rows = {(layer, kind): [] for layer in range(2) for kind in MATRIX_KINDS}
    for index in range(2):
        optimizer.select_microbatch(index)
        reference.body.stat_at.copy_(torch.arange(index * 2, (index + 1) * 2))
        x = torch.randn(1, 3, 3, generator=generator).to(torch.bfloat16) * (rank + 1)
        cotangent = torch.randn(1, 3, 3, generator=generator).to(torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual, expected = model(x), reference(x)
        actual.backward(cotangent)
        expected.backward(cotangent)
        for layer, block in enumerate(reference.body.blocks):
            for kind, value in block.last_inputs.items():
                input_rows[layer, kind].append(value[:, [0, 2]].reshape(2, -1).float())
            for kind, value in block.last_outputs.items():
                output_rows[layer, kind].append(value.grad[:, [0, 2]].reshape(2, -1).float())
    optimizer.finish_backward()
    torch.save(
        {
            "state": optimizer.runtime_state(),
            "input_rows": {key: torch.cat(rows) for key, rows in input_rows.items()},
            "output_rows": {key: torch.cat(rows) for key, rows in output_rows.items()},
        },
        directory / f"rank_{rank}.pt",
    )
    dist.destroy_process_group()


def test_distributed_moments_pool_all_rank_rows_before_local_ema(tmp_path):
    mp.spawn(distributed_pooling_worker, args=(f"file://{tmp_path / 'rendezvous'}", tmp_path), nprocs=2, join=True)
    records = [torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True) for rank in range(2)]
    for rank, record in enumerate(records):
        state = record["state"]["statistics"]
        assert state["sample_count"] == 8
        for moments, rows_name in (("input_moments", "input_rows"), ("output_moments", "output_rows")):
            for key, moment in state[moments].items():
                assert key[0] == rank
                rows = torch.cat([part[rows_name][key] for part in records])
                torch.testing.assert_close(moment, 0.5 * rows.mT @ rows / 8, rtol=1e-5, atol=1e-6)


def distributed_abort_worker(rank, rendezvous, directory, statistic_mode, fault):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        torch.manual_seed(78)
        model = CpuModel(layers=2)
        optimizer = baseline(
            model, GoMuonSettings(statistic_mode=statistic_mode, polar_method="exact_svd"), world=2, rank=rank
        )
        generator = torch.Generator().manual_seed(200 + rank)
        for step in range(2):
            inputs = [torch.randn(1, 3, 3, generator=generator).to(torch.bfloat16) for _ in range(2)]
            backward_step(model, optimizer, inputs, step=step)
        optimizer.begin_step(2)
        model.zero_grad(set_to_none=True)
        for index in range(2):
            optimizer.select_microbatch(index)
            x = torch.randn(1, 3, 3, generator=generator).to(torch.bfloat16)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model(x).float().sum().backward()
        if fault == "gradient_before_step":
            optimizer.finish_backward()
        before = {
            "weights": {key: parameter.detach().clone() for key, parameter in optimizer.matrices.items()},
            "runtime": optimizer.runtime_state(),
        }
        if rank == 1:
            if fault.startswith("gradient"):
                # The last owned matrix exposes mutation of earlier momentum buffers.
                optimizer.owned[1, "down"].grad[0, 0] = float("nan")
            elif fault == "input_nan":
                optimizer.input_rows[1, "down"][0, 0] = float("nan")
            elif fault == "output_nan":
                optimizer.output_rows[1, "down"][0, 0] = float("nan")
            else:
                optimizer.input_rows[1, "down"].fill_(torch.finfo(torch.bfloat16).max)
        aborted = False
        try:
            if fault == "gradient_before_step":
                optimizer.step()
            else:
                optimizer.finish_backward()
        except FloatingPointError:
            aborted = True
        # A live collective after the exception proves both ranks exit together;
        # process-group destruction is not the mechanism that releases a peer.
        peers = torch.tensor(int(aborted))
        dist.all_reduce(peers)
        assert peers.item() == 2
        torch.save(
            {
                "before": before,
                "after": {
                    "weights": {key: parameter.detach().clone() for key, parameter in optimizer.matrices.items()},
                    "runtime": optimizer.runtime_state(),
                },
                "aborted": aborted,
            },
            directory / f"abort_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("statistic_mode", ["ema_per_step", "paper_sparse"])
@pytest.mark.parametrize(
    "fault", ["gradient_before_finish", "gradient_before_step", "input_nan", "output_nan", "raw_overflow"]
)
def test_distributed_invalid_observation_aborts_both_owners_without_state_mutation(tmp_path, statistic_mode, fault):
    mp.spawn(
        distributed_abort_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", tmp_path, statistic_mode, fault),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        record = torch.load(tmp_path / f"abort_{rank}.pt", weights_only=True)
        assert record["aborted"]
        before, after = record["before"], record["after"]
        for key, parameter in before["weights"].items():
            torch.testing.assert_close(after["weights"][key], parameter, rtol=0, atol=0)
        for key, momentum in before["runtime"]["base"]["momentum"].items():
            torch.testing.assert_close(after["runtime"]["base"]["momentum"][key], momentum, rtol=0, atol=0)
        assert after["runtime"]["base"]["steps"] == before["runtime"]["base"]["steps"] == 2
        for name, value in before["runtime"]["statistics"].items():
            if isinstance(value, dict):
                for key, tensor in value.items():
                    torch.testing.assert_close(after["runtime"]["statistics"][name][key], tensor, rtol=0, atol=0)
            else:
                assert after["runtime"]["statistics"][name] == value
