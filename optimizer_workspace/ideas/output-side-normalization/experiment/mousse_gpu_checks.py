# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-rank A100 acceptance probes for Mousse, called by related_gpu_checks.

The independent common-NS oracle preserves staged BF16 operands with FP32
products, not exact-polar identities. Paper parity executes the pinned MIT
source's inner geometry and eager NS bodies, injecting closure arguments and
omitting compilation/optional imports and its broken single-matrix transport.
These small synthetic probes are not quality, performance, or memory-fit evidence.
"""

import ast
import copy
import hashlib
import io
from dataclasses import asdict
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from .mousse import Mousse, MousseSettings, eigenpairs, mousse_direction, paper_newton_schulz
from .records import compare_tree
from .reference import TASK
from .related_base import MATRIX_KINDS

RTOL = 0.003
ATOL = 5e-5
SHAPES = ((128, 128), (256, 128), (128, 256))
PINNED_SOURCE = TASK / "artifacts/external-sources/Mousse-d00c1bf17790fbe56424ee5567cce80d8e75f4b2/dion/dion/mousse.py"
PINNED_SOURCE_SHA256 = "874f02c33531b26cfa0b5c6c1c798afe72ec9c63ce07f804ee6caea222ccd201"


def pinned_reference():
    """Load actual author function bodies without importing distributed or Triton dependencies."""
    source = PINNED_SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != PINNED_SOURCE_SHA256:
        raise ValueError(f"Pinned Mousse reference SHA256 mismatch: {digest}")
    parsed = ast.parse(source.decode("utf-8"))
    names = {"clean_eigenvalues", "muon_update_newton_schulz", "zeropower_via_newtonschulz5", "apply_soap_and_ns"}
    functions = [node for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef) and node.name in names]
    geometry = next(node for node in functions if node.name == "apply_soap_and_ns")
    closure = (
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
    geometry.args.kwonlyargs.extend(ast.arg(arg=name) for name in closure)
    geometry.args.kw_defaults.extend([None] * len(closure))
    for function in functions:
        function.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": torch, "source_sha256": digest}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])), str(PINNED_SOURCE), "exec"
        ),
        namespace,
    )
    return SimpleNamespace(**namespace)


def _assert_close(actual, expected, label, errors):
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL, msg=label)
    difference = (actual.float() - expected.float()).abs()
    errors.append(
        {
            "label": label,
            "max_absolute_error": difference.max().item(),
            "mean_absolute_error": difference.mean().item(),
        }
    )


def _product(left, right, mode):
    if mode == "tpu":
        return torch.mm(left.to(torch.bfloat16), right.to(torch.bfloat16), out_dtype=torch.float32)
    return torch.mm(left, right)


def _staged_ns(source, mode):
    matrix = source.float() / (torch.linalg.matrix_norm(source.float()) + 1e-5)
    transpose = matrix.shape[0] > matrix.shape[1]
    if transpose:
        matrix = matrix.T
    for _ in range(10):
        gram = _product(matrix, matrix.T, mode)
        squared_gram = _product(gram, gram, mode)
        polynomial = -4.7750 * gram + 2.0315 * squared_gram
        matrix = 3.4445 * matrix + _product(polynomial, matrix, mode)
    return matrix.T if transpose else matrix


def _staged_direction(source, left, right, settings, mode):
    left_values, left_vectors = left
    right_values, right_vectors = right
    matrix = torch.mm(left_vectors.T, source.float())
    matrix = torch.mm(matrix, right_vectors)
    matrix = matrix / left_values.abs().pow(settings.alpha)[:, None]
    matrix = matrix / right_values.abs().pow(settings.alpha)[None, :]
    orthogonal = _staged_ns(matrix, mode)
    matrix = orthogonal / left_values.abs().pow(settings.alpha)[:, None]
    matrix = matrix / right_values.abs().pow(settings.alpha)[None, :]
    matrix = torch.mm(left_vectors, matrix)
    matrix = torch.mm(matrix, right_vectors.T)
    if settings.norm_graft:
        matrix = matrix * (orthogonal.norm() / matrix.norm())
    return matrix, orthogonal.norm()


def _common_direction_checks(device, modes=("f32", "tpu")):
    errors = []
    generator = torch.Generator(device=device).manual_seed(743)
    for shape in SHAPES:
        source = torch.randn(shape, generator=generator, device=device)
        left = (
            torch.logspace(-4, 1, shape[0], device=device),
            torch.linalg.qr(torch.randn(shape[0], shape[0], generator=generator, device=device)).Q,
        )
        right = (
            torch.logspace(-4, 1, shape[1], device=device),
            torch.linalg.qr(torch.randn(shape[1], shape[1], generator=generator, device=device)).Q,
        )
        for mode in modes:
            for graft in (False, True):
                settings = MousseSettings(norm_graft=graft)
                actual = mousse_direction(source, left, right, settings, ns_steps=10, ns_epsilon=1e-5, ns_mode=mode)
                expected, target_norm = _staged_direction(source, left, right, settings, mode)
                label = f"common.{shape}.{mode}.graft={graft}"
                _assert_close(actual, expected, label, errors)
                if graft:
                    _assert_close(actual.norm(), target_norm, f"{label}.actual_ns_norm", errors)
    return errors


def _paper_checks(device, author):
    errors = []
    generator = torch.Generator(device=device).manual_seed(911)
    for shape in SHAPES:
        gradient = torch.randn(shape, generator=generator, device=device)
        _assert_close(
            paper_newton_schulz(gradient, 1e-8),
            author.zeropower_via_newtonschulz5(gradient, epsilon=1e-8),
            f"paper_helper.{shape}",
            errors,
        )
        for graft in (False, True):
            settings = MousseSettings(profile="paper", norm_graft=graft, refresh=3)
            left_moment = torch.zeros(shape[0], shape[0], device=device)
            right_moment = torch.zeros(shape[1], shape[1], device=device)
            for _ in range(3):
                history = torch.randn(shape, generator=generator, device=device)
                left_moment.add_(history @ history.T)
                right_moment.add_(history.T @ history)
            state = {"L": torch.zeros_like(left_moment), "R": torch.zeros_like(right_moment)}
            state["L"].copy_(left_moment)
            state["R"].copy_(right_moment)
            left = right = None
            for step in range(1, 6):
                gradient = torch.randn(shape, generator=generator, device=device)
                source = torch.randn(shape, generator=generator, device=device).to(torch.bfloat16)
                arguments = dict(
                    shampoo_epsilon=settings.damping,
                    shampoo_update_freq=settings.refresh,
                    shampoo_alpha=settings.alpha,
                    LR_correction=True,
                    newton_schulz_func=author.zeropower_via_newtonschulz5,
                    flatten=False,
                    epsilon=settings.paper_ns_epsilon,
                    apply_norm=graft,
                    use_LorR=0,
                    optimizer_debug=False,
                    debug_logs=None,
                    param_name="probe",
                )
                author.apply_soap_and_ns(
                    source,
                    gradient,
                    state,
                    step,
                    settings.beta,
                    **arguments,
                )
                left_moment.mul_(settings.beta).add_(gradient @ gradient.T, alpha=1 - settings.beta)
                right_moment.mul_(settings.beta).add_(gradient.T @ gradient, alpha=1 - settings.beta)
                if step % settings.refresh == 1:
                    left = eigenpairs(left_moment, settings, step)
                    right = eigenpairs(right_moment, settings, step)
                    for name, ours, theirs in (("left", left, state["eig_L"]), ("right", right, state["eig_R"])):
                        our_factor = (ours[1] * ours[0].pow(-settings.alpha)[None, :]) @ ours[1].T
                        their_factor = (theirs[1] * theirs[0].pow(-settings.alpha)[None, :]) @ theirs[1].T
                        _assert_close(our_factor, their_factor, f"paper_roots.{shape}.{step}.{name}", errors)
                # Frozen bases isolate finite-NS arithmetic from eigensolver
                # subspace choices, which are not BF16-rotation invariant.
                frozen = {
                    "L": torch.zeros_like(left_moment),
                    "R": torch.zeros_like(right_moment),
                    "eig_L": left,
                    "eig_R": right,
                }
                expected = author.apply_soap_and_ns(source, gradient, frozen, 2, settings.beta, **arguments)
                actual = mousse_direction(source, left, right, settings, ns_steps=10, ns_epsilon=1e-5, ns_mode="tpu")
                label = f"paper_geometry.{shape}.{step}.graft={graft}"
                _assert_close(actual, expected.float(), label, errors)
                _assert_close(left_moment, state["L"], f"{label}.left_ema", errors)
                _assert_close(right_moment, state["R"], f"{label}.right_ema", errors)
    return errors


class ProjectionProbe(nn.Module):
    """Uncaptured synthetic network using all seven matrix kinds and both rectangular orientations."""

    def __init__(self, layers, device):
        super().__init__()
        self.body = nn.Module()
        self.body.blocks = nn.ModuleList([nn.Module() for _ in range(layers)])
        for block in self.body.blocks:
            for kind in MATRIX_KINDS:
                rows, columns = (64, 32) if kind in ("gate", "up") else (32, 64) if kind == "down" else (32, 32)
                setattr(block, kind, nn.Linear(columns, rows, bias=False))
        self.to(device)
        generator = torch.Generator(device=device).manual_seed(342)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, device=device) * 0.05)

    def forward(self, inputs):
        for block in self.body.blocks:
            attention = block.o(torch.tanh(block.q(inputs) + block.k(inputs) + block.v(inputs)))
            mlp = block.down(torch.tanh(block.gate(inputs)) * block.up(inputs))
            inputs = inputs + attention + mlp
        return inputs


def _optimizer(model, settings, device, world, rank, ns_mode):
    config = {
        "optimizer": {
            "lr": 0.02,
            "weight_decay": 0.1,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 10,
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
        device=device,
    )


def _snapshot(model, optimizer):
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.runtime_state()}, stream)
    stream.seek(0)
    return torch.load(stream, map_location="cpu", weights_only=True)


def _cached_resume_checks(device, world, rank, ns_mode="tpu"):
    comparisons = []
    for settings in (
        MousseSettings(beta=0.938, damping=0.001, refresh=4),
        MousseSettings(profile="paper"),
    ):
        model = ProjectionProbe(world, device)
        optimizer = _optimizer(model, settings, device, world, rank, ns_mode)

        def advance(network, current, step):
            generator = torch.Generator(device=device).manual_seed(414 + step)
            for parameter in network.parameters():
                parameter.grad = torch.randn(parameter.shape, generator=generator, device=device) * 0.01
            current.begin_step(step)
            current.step()

        for step in range(settings.refresh - 1):
            advance(model, optimizer, step)
        checkpoint = _snapshot(model, optimizer)
        cached = checkpoint["optimizer"]["grams"]
        for state in cached.values():
            if state["refresh_step"] != 1:
                raise AssertionError("Mousse refreshed before its configured phase")
        for step in range(settings.refresh - 1, settings.refresh + 2):
            advance(model, optimizer, step)
            if step == settings.refresh - 1:
                for key, state in optimizer.runtime_state()["grams"].items():
                    compare_tree(
                        tuple(value.cpu() for value in state["eigen_left"]),
                        cached[key]["eigen_left"],
                        "cached_left",
                        comparisons,
                    )
                    compare_tree(
                        tuple(value.cpu() for value in state["eigen_right"]),
                        cached[key]["eigen_right"],
                        "cached_right",
                        comparisons,
                    )
        expected = _snapshot(model, optimizer)
        restored = ProjectionProbe(world, device)
        resumed = _optimizer(restored, settings, device, world, rank, ns_mode)
        restored.load_state_dict(checkpoint["model"])
        resumed.restore_runtime_state(checkpoint["optimizer"])
        for step in range(settings.refresh - 1, settings.refresh + 2):
            advance(restored, resumed, step)
        compare_tree(_snapshot(restored, resumed), expected, settings.profile, comparisons)
        for state in resumed.runtime_state()["grams"].values():
            if state["refresh_step"] != settings.refresh + 1:
                raise AssertionError("Mousse continuation missed the scheduled refresh")
    return {"tensor_comparisons": len(comparisons), "max_absolute_error": max(comparisons, default=0.0)}


def _distributed_gradient_checks(device, world, rank, ns_mode="tpu"):
    errors = []
    model = ProjectionProbe(world, device)
    local_reference = copy.deepcopy(model)
    single_reference = copy.deepcopy(model)
    settings = MousseSettings(beta=0.938, damping=0.001, refresh=4)
    optimizer = _optimizer(model, settings, device, world, rank, ns_mode)
    reference_optimizer = _optimizer(single_reference, settings, device, 1, 0, ns_mode)
    distributed = nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None
    )
    generator = torch.Generator(device=device).manual_seed(994 + rank)
    inputs = torch.randn(8, 32, generator=generator, device=device)
    target = torch.randn(8, 32, generator=generator, device=device) * (rank + 1)
    for network in (distributed, local_reference):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            prediction = network(inputs)
        (prediction.float() * target).mean().backward()
    for parameter, local, reference in zip(
        model.parameters(), local_reference.parameters(), single_reference.parameters(), strict=True
    ):
        pooled = local.grad.clone()
        dist.all_reduce(pooled)
        pooled /= world
        _assert_close(parameter.grad, pooled, "ddp_pooled_gradient", errors)
        reference.grad = pooled
    optimizer.begin_step(0)
    reference_optimizer.begin_step(0)
    optimizer.step()
    reference_optimizer.step()
    for key, parameter in optimizer.matrices.items():
        _assert_close(parameter, reference_optimizer.matrices[key], f"ddp_update.{key}", errors)
    for key, state in optimizer.runtime_state()["grams"].items():
        reference = reference_optimizer.runtime_state()["grams"][key]
        _assert_close(state["left"], reference["left"], f"ddp_left_gram.{key}", errors)
        _assert_close(state["right"], reference["right"], f"ddp_right_gram.{key}", errors)
    return errors


@torch.no_grad()
def run_checks(device: torch.device) -> dict:
    """Return acceptance flags and numerical evidence only after all two-rank A100 checks pass."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("Mousse GPU acceptance requires A100 SM80")
    if not dist.is_initialized() or dist.get_world_size() != 2 or dist.get_backend() != "nccl":
        raise ValueError("Mousse GPU acceptance requires an initialized two-rank NCCL group")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Mousse GPU acceptance requires TF32 disabled")
    world, rank = dist.get_world_size(), dist.get_rank()
    common = _common_direction_checks(device)
    author = pinned_reference()
    paper = _paper_checks(device, author)
    dist.barrier()
    continuation = _cached_resume_checks(device, world, rank)
    with torch.enable_grad():
        distributed = _distributed_gradient_checks(device, world, rank)
    dist.barrier()
    return {
        "mousse_gpu_numerics": True,
        "mousse_gpu_common_direction": True,
        "mousse_gpu_paper_source": True,
        "mousse_gpu_cached_resume": True,
        "mousse_gpu_distributed_gradients": True,
        "mousse_gpu_evidence": {
            "rtol": RTOL,
            "atol": ATOL,
            "rank": rank,
            "world_size": world,
            "common_ns_modes": ["f32", "tpu"],
            "common_ns_steps": 10,
            "shapes": list(SHAPES),
            "pinned_source_sha256": author.source_sha256,
            "common_direction": common,
            "paper_source": paper,
            "cached_resume": continuation,
            "distributed_gradients": distributed,
            "paper_reference_adaptation": "Injected closure arguments; omitted compile decorators and transport.",
            "paper_geometry_basis": "Common frozen bases; independently refreshed inverse roots checked separately.",
        },
    }
