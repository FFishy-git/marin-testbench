# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from experiment.block_toner import (
    BLOCK_ARMS,
    BLOCK_KINDS,
    HYPERBALL_NORM_FLOOR,
    INPUT_KINDS,
    ROW_SCALE,
    BlockToner,
    DecayMode,
    DiagonalMoments,
    MatrixFactors,
    RowObservedLinear,
    Side,
    block_update,
    block_update_stack,
    diagonal_roots,
    hyperball_delta,
    stack_factors,
)
from experiment.capture import captures_input
from experiment.factors import OutputSettings, inverse_factors
from experiment.optimizer import AdamH
from toner.fast.muon import newton_schulz_batched
from torch import nn

ARGS = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32")


def unit(factor):
    return factor * (factor.shape[-1] ** 0.5 / factor.norm())


def random_factors(rows, columns, generator):
    left = torch.randn(rows, rows + 2, generator=generator)
    right = torch.randn(columns, columns + 2, generator=generator)
    left_inner, left_outer = inverse_factors(left @ left.T, OutputSettings())
    right_inner, right_outer = inverse_factors(right @ right.T, OutputSettings())
    return left_inner, left_outer, right_inner, right_outer


def levanter_norm_preserving_reference(param: torch.Tensor, update: torch.Tensor, lr: float) -> torch.Tensor:
    """Direct port of levanter.optim.util.norm_preserving_update / grug.fast_track.optimizer."""
    p = param.double()
    u = update.double()
    dims = (-2, -1)
    p_norm = torch.sqrt(torch.sum(p * p, dim=dims, keepdim=True))
    u_norm = torch.sqrt(torch.sum(u * u, dim=dims, keepdim=True)).clamp_min(HYPERBALL_NORM_FLOOR)
    new_p = p - lr * u * p_norm / u_norm
    new_p_norm = torch.sqrt(torch.sum(new_p * new_p, dim=dims, keepdim=True)).clamp_min(HYPERBALL_NORM_FLOOR)
    return (new_p / new_p_norm * p_norm - p).to(param.dtype)


def test_hyperball_arms_input_and_output_partitioning():
    assert BLOCK_ARMS["none_all"].input_kinds(Side.FULL) == ()
    assert BLOCK_ARMS["none_all"].input_kinds(Side.DIAGONAL) == ()
    assert BLOCK_ARMS["none_all"].output_kinds(Side.FULL) == ()
    assert BLOCK_ARMS["none_all"].output_kinds(Side.DIAGONAL) == ()
    assert not BLOCK_ARMS["none_all"].apply_left
    assert not captures_input("none_all")

    assert BLOCK_ARMS["input_only"].input_kinds(Side.FULL) == INPUT_KINDS
    assert BLOCK_ARMS["input_only"].input_kinds(Side.DIAGONAL) == ()
    assert BLOCK_ARMS["input_only"].output_kinds(Side.FULL) == ()
    assert BLOCK_ARMS["input_only"].output_kinds(Side.DIAGONAL) == ()
    assert not BLOCK_ARMS["input_only"].apply_left
    assert captures_input("input_only")


def test_hyperball_matches_levanter_and_conserves_frobenius_norm_and_scale_invariance():
    generator = torch.Generator().manual_seed(2026)
    for left_kind, right_kind in [
        ("none", "none"),        # MuonH
        ("none", "dense"),       # MuonTH
        ("dense", "dense"),      # WT-MLP-Lag-H hidden blocks
        ("diagonal", "dense"),   # WT-MLP-Lag-H expansion blocks
    ]:
        count, shape = 4, (8, 5)
        weights = torch.randn(count, *shape, generator=generator) * 0.05
        init_norms = weights.norm(dim=(-2, -1)).clone()
        momenta = torch.zeros(count, *shape)

        for step_idx in range(12):
            grads = torch.randn(count, *shape, generator=generator) * 0.01
            ri_list, ro_list, li_list, lo_list = [], [], [], []
            for _ in range(count):
                li, lo, ri, ro = random_factors(*shape, generator)
                ri_list.append(ri if right_kind == "dense" else None)
                ro_list.append(ro if right_kind == "dense" else None)
                if left_kind == "dense":
                    li_list.append(li)
                    lo_list.append(lo)
                elif left_kind == "diagonal":
                    li_list.append(torch.rand(shape[0], generator=generator) + 0.2)
                    lo_list.append(torch.rand(shape[0], generator=generator) + 0.2)
                else:
                    li_list.append(None)
                    lo_list.append(None)

            lr = 0.032 * (1.0 - 0.05 * step_idx)
            per_deltas, per_moms = [], []
            for idx in range(count):
                fac = MatrixFactors(li_list[idx], lo_list[idx], ri_list[idx], ro_list[idx])
                m_new, d_single = block_update(
                    weights[idx],
                    grads[idx],
                    momenta[idx],
                    fac,
                    args=ARGS,
                    lr=lr,
                    weight_decay=0.1,  # ignored under HYPERBALL
                    decay_mode=DecayMode.HYPERBALL,
                )
                # Independent explicit direction + Levanter reference
                exp_m = 0.95 * momenta[idx] + grads[idx]
                src = grads[idx] + 0.95 * exp_m
                if ri_list[idx] is not None:
                    src = src @ ri_list[idx]
                if li_list[idx] is not None:
                    src = (li_list[idx][:, None] * src) if left_kind == "diagonal" else (li_list[idx] @ src)
                d0 = newton_schulz_batched(src[None], 10, 1e-5, "f32")[0]
                if ro_list[idx] is not None:
                    d0 = d0 @ unit(ro_list[idx])
                if lo_list[idx] is not None:
                    l_hat = unit(lo_list[idx])
                    d0 = (l_hat[:, None] * d0) if left_kind == "diagonal" else (l_hat @ d0)
                expected = levanter_norm_preserving_reference(weights[idx], d0, lr)
                torch.testing.assert_close(d_single, expected, rtol=1e-5, atol=1e-6)

                # Scale-invariance check: scaling direction by any c > 0 yields identical update
                scaled_d = hyperball_delta(weights[idx], 7.31 * d0, lr)
                torch.testing.assert_close(d_single, scaled_d, rtol=1e-6, atol=1e-7)

                per_deltas.append(d_single)
                per_moms.append(m_new)

            batched_m, batched_delta = block_update_stack(
                weights,
                grads,
                momenta,
                (
                    stack_factors(li_list, leading=count, width=shape[0]),
                    stack_factors(lo_list, leading=count, width=shape[0]),
                ),
                (
                    stack_factors(ri_list, leading=count, width=shape[1]),
                    stack_factors(ro_list, leading=count, width=shape[1]),
                ),
                args=ARGS,
                lr=lr,
                weight_decay=0.0,
                decay_mode=DecayMode.HYPERBALL,
            )
            torch.testing.assert_close(batched_m, torch.stack(per_moms), rtol=0, atol=0)
            torch.testing.assert_close(batched_delta, torch.stack(per_deltas), rtol=1e-4, atol=1e-6)
            weights = weights + batched_delta
            momenta = batched_m

        final_norms = weights.norm(dim=(-2, -1))
        torch.testing.assert_close(final_norms, init_norms, rtol=1e-5, atol=1e-6)


def test_hyperball_zero_gradient_is_safe_noop():
    w = torch.randn(2, 6, 4)
    z = torch.zeros_like(w)
    d = hyperball_delta(w, z, 0.032)
    assert torch.isfinite(d).all()
    torch.testing.assert_close(w + d, w, rtol=1e-6, atol=1e-7)


def test_adamh_matches_levanter_and_checkpoint_roundtrip():
    gen = torch.Generator().manual_seed(77)
    w_init = torch.randn(32, 16, generator=gen) * 0.04
    init_norm = w_init.norm().item()

    p_uninterrupted = nn.Parameter(w_init.clone())
    opt_uninterrupted = AdamH([p_uninterrupted], lr=0.028, betas=(0.8, 0.98), eps=1e-15)

    p_ref = w_init.double().clone()
    mu_ref = torch.zeros_like(p_ref)
    nu_ref = torch.zeros_like(p_ref)

    saved_state = None
    saved_param = None
    grads = [torch.randn(32, 16, generator=gen) * 0.01 for _ in range(8)]

    for idx, g in enumerate(grads, start=1):
        lr_step = 0.028 * (1.0 - 0.05 * idx)
        opt_uninterrupted.param_groups[0]["lr"] = lr_step
        p_uninterrupted.grad = g.clone()
        opt_uninterrupted.step()

        g64 = g.double()
        mu_ref = 0.8 * mu_ref + 0.2 * g64
        nu_ref = 0.98 * nu_ref + 0.02 * (g64 * g64)
        u64 = (mu_ref / (1.0 - 0.8**idx)) / (torch.sqrt(nu_ref / (1.0 - 0.98**idx)) + 1e-15)
        p_ref = p_ref + levanter_norm_preserving_reference(p_ref, u64, lr_step)

        if idx == 4:
            saved_state = copy.deepcopy(opt_uninterrupted.state_dict())
            saved_param = p_uninterrupted.detach().clone()

    torch.testing.assert_close(p_uninterrupted.detach(), p_ref.float(), rtol=1e-5, atol=1e-6)
    assert abs(p_uninterrupted.detach().norm().item() - init_norm) / init_norm < 1e-6

    # Resume from step 4 and run steps 5..8: must be bitwise identical to uninterrupted
    p_resumed = nn.Parameter(saved_param)
    opt_resumed = AdamH([p_resumed], lr=0.028, betas=(0.8, 0.98), eps=1e-15)
    opt_resumed.load_state_dict(saved_state)
    for idx in range(5, 9):
        opt_resumed.param_groups[0]["lr"] = 0.028 * (1.0 - 0.05 * idx)
        p_resumed.grad = grads[idx - 1].clone()
        opt_resumed.step()

    torch.testing.assert_close(p_resumed.detach(), p_uninterrupted.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(6, 6), (3, 7), (8, 4)])
def test_two_sided_update_matches_independent_finite_ns_equations(shape):
    generator = torch.Generator().manual_seed(5)
    weight, gradient, momentum = (torch.randn(*shape, generator=generator) for _ in range(3))
    li, lo, ri, ro = random_factors(*shape, generator)
    actual_momentum, actual = block_update(
        weight, gradient, momentum, MatrixFactors(li, lo, ri, ro), args=ARGS, lr=0.02, weight_decay=0.1
    )
    expected_momentum = 0.95 * momentum + gradient
    source = li @ (gradient + 0.95 * expected_momentum) @ ri
    direction = unit(lo) @ newton_schulz_batched(source[None], 10, 1e-5, "f32")[0] @ unit(ro)
    decay = weight @ unit(ro) @ unit(ro)
    scale = max(1.0, shape[0] / shape[1]) ** 0.5
    torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
    torch.testing.assert_close(actual, -0.02 * (scale * direction + 0.1 * decay), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "sides",
    [("dense", "dense"), ("dense", "diagonal"), ("none", "dense"), ("none", "diagonal"), ("diagonal", "none")],
)
def test_batched_update_matches_per_matrix_update(sides):
    left_kind, right_kind = sides
    count, shape = 5, (6, 3)
    generator = torch.Generator().manual_seed(31)
    weight, gradient, momentum = (torch.randn(count, *shape, generator=generator) for _ in range(3))
    common = {"args": ARGS, "lr": 0.02, "weight_decay": 0.1}

    def side(kind, width):
        if kind == "none":
            return [None] * count, [None] * count
        if kind == "diagonal":
            vectors = [torch.rand(width, generator=generator) + 0.2 for _ in range(count)]
            return vectors, [torch.rand(width, generator=generator) + 0.2 for _ in range(count)]
        pairs = []
        for _ in range(count):
            sample = torch.randn(width, width + 2, generator=generator)
            pairs.append(inverse_factors(sample @ sample.T, OutputSettings()))
        return [inner for inner, _ in pairs], [outer for _, outer in pairs]

    left_inner, left_outer = side(left_kind, shape[0])
    right_inner, right_outer = side(right_kind, shape[1])
    expected_momenta, expected = [], []
    for index in range(count):
        factors = MatrixFactors(left_inner[index], left_outer[index], right_inner[index], right_outer[index])
        moment, delta = block_update(weight[index], gradient[index], momentum[index], factors, **common)
        expected_momenta.append(moment)
        expected.append(delta)
    actual_momenta, actual = block_update_stack(
        weight,
        gradient,
        momentum,
        (
            stack_factors(left_inner, leading=count, width=shape[0]),
            stack_factors(left_outer, leading=count, width=shape[0]),
        ),
        (
            stack_factors(right_inner, leading=count, width=shape[1]),
            stack_factors(right_outer, leading=count, width=shape[1]),
        ),
        **common,
    )
    torch.testing.assert_close(actual_momenta, torch.stack(expected_momenta), rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("decay_mode", ["default", "isotropic", "two_sided"])
@pytest.mark.parametrize("left_kind", ["dense", "diagonal", "none"])
def test_decay_modes_match_explicit_equations_per_matrix_and_batched(decay_mode, left_kind):
    generator = torch.Generator().manual_seed(99)
    count, shape = 3, (6, 4)
    weights = torch.randn(count, *shape, generator=generator)
    gradients = torch.randn(count, *shape, generator=generator)
    momenta = torch.randn(count, *shape, generator=generator)

    ri_list, ro_list, li_list, lo_list = [], [], [], []
    for _ in range(count):
        li, lo, ri, ro = random_factors(*shape, generator)
        ri_list.append(ri)
        ro_list.append(ro)
        if left_kind == "dense":
            li_list.append(li)
            lo_list.append(lo)
        elif left_kind == "diagonal":
            li_list.append(torch.rand(shape[0], generator=generator) + 0.2)
            lo_list.append(torch.rand(shape[0], generator=generator) + 0.2)
        else:
            li_list.append(None)
            lo_list.append(None)

    per_deltas = []
    for idx in range(count):
        factors = MatrixFactors(li_list[idx], lo_list[idx], ri_list[idx], ro_list[idx])
        _, delta = block_update(
            weights[idx],
            gradients[idx],
            momenta[idx],
            factors,
            args=ARGS,
            lr=0.016,
            weight_decay=0.1,
            decay_mode=decay_mode,
        )
        exp_mom = 0.95 * momenta[idx] + gradients[idx]
        src = (gradients[idx] + 0.95 * exp_mom) @ ri_list[idx]
        if li_list[idx] is not None:
            src = (li_list[idx][:, None] * src) if left_kind == "diagonal" else (li_list[idx] @ src)
        d0 = newton_schulz_batched(src[None], 10, 1e-5, "f32")[0]
        r_hat = unit(ro_list[idx])
        d_lr = d0 @ r_hat
        if lo_list[idx] is not None:
            l_hat = unit(lo_list[idx])
            d_lr = (l_hat[:, None] * d_lr) if left_kind == "diagonal" else (l_hat @ d_lr)
        else:
            l_hat = None

        if decay_mode == "isotropic":
            q = weights[idx]
        elif decay_mode == "default":
            q = weights[idx] @ r_hat @ r_hat
        else:
            q = weights[idx] @ r_hat @ r_hat
            if l_hat is not None:
                q = ((l_hat * l_hat)[:, None] * q) if left_kind == "diagonal" else (l_hat @ l_hat @ q)
        scale = max(1.0, shape[0] / shape[1]) ** 0.5
        expected = -0.016 * (scale * d_lr + 0.1 * q)
        torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)
        per_deltas.append(delta)

    _, batched_delta = block_update_stack(
        weights,
        gradients,
        momenta,
        (stack_factors(li_list, leading=count, width=shape[0]), stack_factors(lo_list, leading=count, width=shape[0])),
        (stack_factors(ri_list, leading=count, width=shape[1]), stack_factors(ro_list, leading=count, width=shape[1])),
        args=ARGS,
        lr=0.016,
        weight_decay=0.1,
        decay_mode=decay_mode,
    )
    torch.testing.assert_close(batched_delta, torch.stack(per_deltas), rtol=1e-4, atol=1e-6)


def test_quintic_newton_schulz_and_marin_exact_config():
    from reproduce.marin.train import schedule_factor

    from campaign.configure_muonh_campaign import build_130m_config
    from experiment.block_toner import QUINTIC_NS_COEFFS, _newton_schulz, _newton_schulz_batched, hyperball_delta
    from experiment.optimizer import AdamH

    gen = torch.Generator().manual_seed(1234)
    mats = torch.randn(4, 16, 8, generator=gen)

    # Default coeffs='simple' must be bitwise identical to MU.newton_schulz_batched
    simple_out = _newton_schulz_batched(mats, 10, 1e-5, "f32", "simple")
    ref_out = newton_schulz_batched(mats, 10, 1e-5, "f32")
    assert torch.equal(simple_out, ref_out)

    # Quintic 5-step matches per-matrix _newton_schulz and explicit Levanter recurrence
    quintic_batched = _newton_schulz_batched(mats, 5, 1e-8, "f32", "quintic")
    for idx in range(4):
        per = _newton_schulz(mats[idx], 5, 1e-8, "f32", "quintic")
        x = mats[idx].float()
        x = x / (torch.linalg.matrix_norm(x) + 1e-8)
        x = x.T  # 16 > 8
        for step_i in range(5):
            a, b, c = QUINTIC_NS_COEFFS[step_i]
            A = x @ x.T
            x = a * x + (b * A + c * (A @ A)) @ x
        expected = x.T
        torch.testing.assert_close(per, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(quintic_batched[idx], expected, rtol=1e-5, atol=1e-6)
        svals = torch.linalg.svdvals(per)
        assert float(svals.min()) > 0.65 and float(svals.max()) < 1.35

    # Marin-exact 130M config check (1pct-noclip + AdamH + quintic5 + 3/13 Adam LR)
    cfg = build_130m_config(arm_key="wt_mlp_lag_muonh_marin", seed=0, lr=0.016)
    assert cfg["experiment"]["decay_mode"] == "hyperball"
    assert cfg["experiment"]["head_mode"] == "adamh"
    assert cfg["experiment"]["recipe"] == "marin_1pct_noclip"
    assert cfg["optimizer"]["lr"] == 0.016
    assert cfg["optimizer"]["head_lr"] == 0.016
    assert cfg["optimizer"]["weight_decay"] == 0.0
    assert cfg["optimizer"]["head_weight_decay"] == 0.0
    assert abs(cfg["optimizer"]["auxiliary_lr"] - 0.016 * 3.0 / 13.0) < 1e-15
    assert abs(cfg["optimizer"]["betas"][0] - 0.9) < 1e-15
    assert abs(cfg["optimizer"]["betas"][1] - 0.999**4) < 1e-15
    assert cfg["optimizer"]["clip_norm"] == 0.0
    assert cfg["optimizer"]["ns_steps"] == 5
    assert cfg["optimizer"]["ns_eps"] == 1e-8
    assert cfg["optimizer"]["ns_coeffs"] == "quintic"
    assert cfg["schedule"] == {
        "warmup_steps": 51,
        "stable_steps": 0,
        "decay_steps": 5069,
        "kind": "linear",
        "min_lr_ratio": 0.05,
    }

    # Verify exact Optax 1pct-noclip schedule match and lr=0 safety at step 0
    assert schedule_factor(0, cfg["schedule"]) == 0.0
    assert abs(schedule_factor(51, cfg["schedule"]) - 1.0) < 1e-15
    assert abs(schedule_factor(5119, cfg["schedule"]) - (1.0 - 0.95 * (5068.0 / 5069.0))) < 1e-15

    w0 = torch.randn(8, 8, generator=gen)
    u0 = torch.randn(8, 8, generator=gen)
    d0 = hyperball_delta(w0, u0, 0.0)
    assert torch.isfinite(d0).all() and float(d0.abs().max()) == 0.0

    p = torch.nn.Parameter(w0.clone())
    p.grad = u0.clone()
    opt = AdamH([p], lr=0.0, betas=(0.9, 0.996), eps=6.9e-16)
    opt.step()
    assert torch.equal(p.data, w0)

