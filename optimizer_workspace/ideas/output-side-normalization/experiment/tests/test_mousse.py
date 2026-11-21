# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import io
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from experiment import reference  # noqa: F401
from experiment.mousse import Mousse, MousseSettings, eigenpairs, mousse_direction, paper_newton_schulz
from experiment.related_base import MATRIX_KINDS
from toner.fast import muon as MU
from torch import nn

PINNED_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "artifacts/external-sources/Mousse-d00c1bf17790fbe56424ee5567cce80d8e75f4b2/dion/dion/mousse.py"
)


@pytest.fixture(scope="module")
def author():
    """Execute pinned geometry and NS, without broken single-matrix transport or compilation."""
    parsed = ast.parse(PINNED_SOURCE.read_text())
    functions = {
        "clean_eigenvalues",
        "muon_update_newton_schulz",
        "zeropower_via_newtonschulz5",
    }
    selected = []
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            node.decorator_list = []
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "mousse_update_batch_async":
            geometry = next(
                child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "apply_soap_and_ns"
            )
            arguments = (
                "shampoo_epsilon",
                "shampoo_update_freq",
                "shampoo_alpha",
                "LR_correction",
                "newton_schulz_func",
                "flatten",
                "epsilon",
                "apply_norm",
                "use_LorR",
                "optimizer_debug",
                "debug_logs",
                "param_name",
            )
            factory = ast.FunctionDef(
                name="make_geometry",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=name) for name in arguments],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[geometry, ast.Return(value=ast.Name(id="apply_soap_and_ns", ctx=ast.Load()))],
                decorator_list=[],
            )
            selected.append(factory)
    module = ast.Module(body=selected, type_ignores=[])
    module.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(PINNED_SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


def make_model(rows=3, columns=5, layers=1):
    blocks = []
    for _ in range(layers):
        block = nn.Module()
        for kind in MATRIX_KINDS:
            setattr(block, kind, nn.Linear(columns, rows, bias=False))
        blocks.append(block)
    body = nn.Module()
    body.blocks = nn.ModuleList(blocks)
    model = nn.Module()
    model.body = body
    return model


def make_optimizer(model, settings=None, *, momentum=0.8, nesterov=True, ns_mode="f32", world=1, rank=0):
    settings = settings or MousseSettings()
    config = {
        "optimizer": {
            "lr": 0.03,
            "weight_decay": 0.1,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": 6,
            "ns_eps": 1e-5,
            "ns_mode": ns_mode,
        },
        "experiment": {"related_baseline": {"settings": asdict(settings)}},
    }
    return Mousse(
        model,
        config,
        positions=None,
        rows_per_microbatch=0,
        capacity=0,
        rows_total=0,
        world=world,
        rank=rank,
        device="cpu",
    )


def assign_gradients(model, step):
    generator = torch.Generator().manual_seed(100 + step)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.randn(parameter.shape, generator=generator) * (index + 1)


def polynomial_reference(matrix, steps, epsilon):
    """FP64 polynomial oracle; singular values evolve independently under NS."""
    left, singular, right = torch.linalg.svd(matrix.double(), full_matrices=False)
    singular = singular / (singular.square().sum().sqrt() + epsilon)
    for _ in range(steps):
        singular = 3.4445 * singular - 4.7750 * singular**3 + 2.0315 * singular**5
    return (left * singular[None, :]) @ right


@pytest.mark.parametrize("shape", [(4, 7), (7, 4), (5, 5)])
@pytest.mark.parametrize("graft", [False, True])
def test_control_direction_matches_singular_polynomial_in_eigen_coordinates(shape, graft):
    generator = torch.Generator().manual_seed(41)
    source = torch.randn(shape, generator=generator)
    left_vectors = torch.linalg.qr(torch.randn(shape[0], shape[0], generator=generator)).Q
    right_vectors = torch.linalg.qr(torch.randn(shape[1], shape[1], generator=generator)).Q
    left_values = torch.linspace(0.2, 4.0, shape[0])
    right_values = torch.linspace(0.1, 3.0, shape[1])
    settings = MousseSettings(norm_graft=graft)
    actual = mousse_direction(
        source,
        (left_values, left_vectors),
        (right_values, right_vectors),
        settings,
        ns_steps=6,
        ns_epsilon=1e-5,
        ns_mode="f32",
    )
    left_scale = left_values.double().pow(-settings.alpha)
    right_scale = right_values.double().pow(-settings.alpha)
    rotated = left_vectors.double().T @ source.double() @ right_vectors.double()
    orthogonal = polynomial_reference(left_scale[:, None] * rotated * right_scale[None, :], 6, 1e-5)
    expected = (
        left_vectors.double() @ (left_scale[:, None] * orthogonal * right_scale[None, :]) @ right_vectors.double().T
    )
    if graft:
        expected = expected * (orthogonal.norm() / expected.norm())
        torch.testing.assert_close(actual.norm().double(), orthogonal.norm(), rtol=2e-5, atol=2e-6)
        assert abs(orthogonal.norm().item() - min(shape) ** 0.5) > 0.01
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    assert (actual.double() - expected).abs().max().item() < 1e-5
    assert (actual.double() - expected).abs().mean().item() < 2e-6


@pytest.mark.parametrize("shape", [(4, 7), (7, 4)])
def test_paper_ns_matches_executable_pinned_helper_exactly(author, shape):
    matrix = torch.randn(shape, generator=torch.Generator().manual_seed(8))
    actual = paper_newton_schulz(matrix, 1e-8)
    expected = author.zeropower_via_newtonschulz5(matrix, epsilon=1e-8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("profile,ns_mode", [("controlled", "f32"), ("controlled", "bf16"), ("paper", "f32")])
@pytest.mark.parametrize("graft", [False, True])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("shape", [(3, 5), (5, 3)])
def test_step_matches_executable_author_geometry_and_direction(author, profile, ns_mode, graft, nesterov, shape):
    settings = MousseSettings(profile=profile, norm_graft=graft, refresh=3)
    model = make_model(*shape)
    optimizer = make_optimizer(model, settings, nesterov=nesterov, ns_mode=ns_mode)
    key = (0, "q")
    parameter = model.body.blocks[0].q.weight
    ns = (
        author.zeropower_via_newtonschulz5
        if profile == "paper"
        else lambda source, epsilon: MU.newton_schulz(source, 6, epsilon, ns_mode)
    )
    author_state = {
        "L": torch.zeros(shape[0], shape[0]),
        "R": torch.zeros(shape[1], shape[1]),
        "eig_L": (torch.ones(shape[0]), torch.eye(shape[0])),
        "eig_R": (torch.ones(shape[1]), torch.eye(shape[1])),
    }
    debug = {}
    geometry = author.make_geometry(
        epsilon=settings.paper_ns_epsilon if profile == "paper" else 1e-5,
        shampoo_epsilon=settings.damping,
        shampoo_update_freq=settings.refresh,
        shampoo_alpha=settings.alpha,
        LR_correction=True,
        flatten=False,
        newton_schulz_func=ns,
        apply_norm=graft,
        use_LorR=0,
        optimizer_debug=True,
        param_name="q",
        debug_logs=debug,
    )
    momentum = torch.zeros(shape)
    ratio = (shape[0] / shape[1]) ** 0.5 if profile == "paper" else max(1, shape[0] / shape[1]) ** 0.5
    for step in range(5):
        assign_gradients(model, step)
        start = parameter.detach().clone()
        gradient = parameter.grad.clone()
        momentum = 0.8 * momentum + gradient
        source = gradient + 0.8 * momentum if nesterov else momentum
        source = source.to(torch.bfloat16) if profile == "paper" else source
        author_direction = geometry(source, gradient, author_state, step=step + 1, shampoo_beta=settings.beta)
        optimizer.begin_step(step)
        optimizer.step()
        state = optimizer.runtime_state()["grams"][key]
        torch.testing.assert_close(state["left"], author_state["L"], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(state["right"], author_state["R"], rtol=1e-6, atol=1e-7)
        for ours, theirs in (
            (state["eigen_left"], author_state["eig_L"]),
            (state["eigen_right"], author_state["eig_R"]),
        ):
            torch.testing.assert_close(ours[0], theirs[0], rtol=2e-5, atol=1e-6)
            torch.testing.assert_close(ours[1].abs(), theirs[1].abs(), rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(optimizer.runtime_state()["base"]["momentum"][key], momentum, rtol=0, atol=0)
        expected_delta = -0.03 * (ratio * author_direction.float() + 0.1 * start)
        torch.testing.assert_close(parameter.detach(), start + expected_delta, rtol=2e-5, atol=2e-6)
        if profile == "controlled":
            actual_direction = ((start - parameter.detach()) / 0.03 - 0.1 * start) / ratio
            expected_norm = debug["norm_U_ns/q"] if graft else debug["rms_U_after_rotation/q"] * gradient.numel() ** 0.5
            torch.testing.assert_close(actual_direction.norm(), torch.tensor(expected_norm), rtol=2e-5, atol=2e-6)


def test_gram_ema_is_not_the_direct_output_moment_or_mean_of_local_grams():
    model = make_model(rows=2, columns=2)
    settings = MousseSettings(beta=0.5)
    optimizer = make_optimizer(model, settings)
    # Two ranks/examples have identical inputs but opposing cotangents.
    inputs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    cotangents = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
    local_gradients = cotangents[:, :, None] * inputs[:, None, :]
    pooled_gradient = local_gradients.mean(0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = pooled_gradient.clone()
    optimizer.step()
    state = optimizer.runtime_state()["grams"][(0, "q")]
    direct_output = cotangents.T @ cotangents / 2
    local_grams_mean = (local_gradients @ local_gradients.mT).mean(0)
    torch.testing.assert_close(state["left"], torch.zeros(2, 2), rtol=0, atol=0)
    assert not torch.allclose(state["left"], (1 - settings.beta) * direct_output)
    assert not torch.allclose(state["left"], (1 - settings.beta) * local_grams_mean)
    for parameter, start in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(parameter.detach(), start * (1 - 0.03 * 0.1), rtol=1e-6, atol=1e-7)


def test_cached_eigen_refresh_and_serialized_resume_preserve_updates():
    settings = MousseSettings(beta=0.7, refresh=3)
    model = make_model()
    optimizer = make_optimizer(model, settings)
    previous = None
    left_moment = torch.zeros(3, 3)
    for step in range(2):
        assign_gradients(model, step)
        gradient = model.body.blocks[0].q.weight.grad
        left_moment = settings.beta * left_moment + (1 - settings.beta) * (gradient @ gradient.T)
        optimizer.begin_step(step)
        optimizer.select_microbatch(0)
        optimizer.finish_backward()
        optimizer.begin()
        optimizer.step()
        state = copy.deepcopy(optimizer.runtime_state()["grams"][(0, "q")])
        torch.testing.assert_close(state["left"], left_moment, rtol=1e-6, atol=1e-7)
        if previous:
            torch.testing.assert_close(state["eigen_left"][0], previous["eigen_left"][0], rtol=0, atol=0)
            torch.testing.assert_close(state["eigen_left"][1], previous["eigen_left"][1], rtol=0, atol=0)
            assert not torch.allclose(state["left"], previous["left"])
        previous = state
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.runtime_state()}, stream)
    stream.seek(0)
    snapshot = torch.load(stream, weights_only=True)
    resumed_model = make_model()
    resumed_model.load_state_dict(snapshot["model"])
    resumed = make_optimizer(resumed_model, settings)
    resumed.restore_runtime_state(snapshot["optimizer"])
    with pytest.raises(ValueError, match="counters"):
        resumed.begin_step(0)
    for step in range(2, 6):
        for current_model, current_optimizer in ((model, optimizer), (resumed_model, resumed)):
            assign_gradients(current_model, step)
            current_optimizer.begin_step(step)
            current_optimizer.step()
        for actual, expected in zip(resumed_model.parameters(), model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        state = optimizer.runtime_state()["grams"][(0, "q")]
        if step == 2:
            assert state["refresh_step"] == 1
        if step == 3:
            assert state["refresh_step"] == 4
            assert not torch.allclose(state["eigen_left"][0], previous["eigen_left"][0])
        assert resumed.last_metrics == optimizer.last_metrics


def test_disabling_bias_correction_does_not_rescale_the_stored_ema():
    model = make_model()
    settings = MousseSettings(beta=0.5, refresh=1, bias_correction=False)
    optimizer = make_optimizer(model, settings)
    expected_left, expected_right = torch.zeros(3, 3), torch.zeros(5, 5)
    for step in range(3):
        assign_gradients(model, step)
        gradient = model.body.blocks[0].q.weight.grad
        expected_left = 0.5 * expected_left + 0.5 * gradient @ gradient.T
        expected_right = 0.5 * expected_right + 0.5 * gradient.T @ gradient
        optimizer.begin_step(step)
        optimizer.step()
    state = optimizer.runtime_state()["grams"][(0, "q")]
    torch.testing.assert_close(state["left"], expected_left, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(state["right"], expected_right, rtol=1e-6, atol=1e-7)


def nonfinite_gradient_worker(rank, rendezvous, directory):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    model = make_model(layers=2)
    optimizer = make_optimizer(model, world=2, rank=rank)
    assign_gradients(model, 0)
    if rank == 1:
        model.body.blocks[1].down.weight.grad[0, 0] = float("nan")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    failed = False
    try:
        optimizer.step()
    except FloatingPointError:
        failed = True
    torch.save(
        {
            "failed": failed,
            "unchanged": all(
                torch.equal(start, parameter) for start, parameter in zip(before, model.parameters(), strict=True)
            ),
            "steps": optimizer.steps,
            "left_trace": sum(state["left"].trace() for state in optimizer.runtime_state()["grams"].values()),
        },
        directory / f"nonfinite_{rank}.pt",
    )
    dist.destroy_process_group()


def test_nonfinite_owner_gradient_aborts_every_rank_before_moments_or_weights_change(tmp_path):
    mp.spawn(nonfinite_gradient_worker, args=(f"file://{tmp_path / 'rendezvous'}", tmp_path), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path / f"nonfinite_{rank}.pt", weights_only=True)
        assert result["failed"]
        assert result["unchanged"]
        assert result["steps"] == 0
        torch.testing.assert_close(result["left_trace"], torch.tensor(0.0), rtol=0, atol=0)


def test_eigen_cleanup_shifts_the_entire_spectrum_like_the_pinned_author(author):
    settings = MousseSettings(damping=0.02, bias_correction=False)
    moment = torch.diag(torch.tensor([-0.2, 1.0, 3.0]))
    actual, _ = eigenpairs(moment, settings, 1)
    unclean = torch.linalg.eigvalsh(moment * (3 / moment.trace()) + settings.damping * torch.eye(3))
    expected = author.clean_eigenvalues(unclean, settings.damping)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[-1] > unclean[-1] + settings.damping


@pytest.mark.parametrize("decay", ["isotropic", "input_shaped", "none"])
def test_explicit_decay_choice_changes_only_the_decay_delta(decay):
    model = make_model()
    no_decay_model = copy.deepcopy(model)
    settings = MousseSettings(decay=decay)
    optimizer = make_optimizer(model, settings)
    no_decay = make_optimizer(no_decay_model, replace(settings, decay="none"))
    assign_gradients(model, 0)
    assign_gradients(no_decay_model, 0)
    start = model.body.blocks[0].q.weight.detach().clone()
    optimizer.step()
    no_decay.step()
    expected_decay = start if decay == "isotropic" else torch.zeros_like(start)
    if decay == "input_shaped":
        values, vectors = optimizer.runtime_state()["grams"][(0, "q")]["eigen_right"]
        factor = (vectors * values.pow(-settings.alpha)[None, :]) @ vectors.T
        factor = factor * (5**0.5 / factor.norm())
        expected_decay = start @ factor @ factor
    difference = model.body.blocks[0].q.weight.detach() - no_decay_model.body.blocks[0].q.weight.detach()
    torch.testing.assert_close(difference, -0.03 * 0.1 * expected_decay, rtol=2e-5, atol=1e-7)
