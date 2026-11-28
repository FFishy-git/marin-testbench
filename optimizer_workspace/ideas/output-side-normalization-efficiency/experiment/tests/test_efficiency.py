# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LazySchedule, BatchExactDenseEngine, HeadBlockMoments, role-aware arms, and MetricHyperball."""

import copy
from types import SimpleNamespace

import pytest
import torch
from experiment.block_toner import (
    BLOCK_ARMS,
    BlockToner,
    DecayMode,
    MatrixFactors,
    Side,
    block_update,
    block_update_stack,
    hyperball_delta,
    metric_hyperball_delta,
    stack_factors,
    unit_rms,
)
from experiment.capture import build_model
from experiment.factors import OutputSettings, inverse_factors
from experiment.lazy_roots import LazySchedule, exact_dense_powers, exact_head_block_powers
from reproduce.marin.model import initialize_model, sample_positions

ARGS = SimpleNamespace(
    muon_momentum=0.95,
    muon_nesterov=True,
    muon_ns_steps=10,
    muon_eps=1e-5,
    muon_ns_mode="f32",
    muon_ns_coeffs="simple",
    muon_compile_ns=False,
)


def test_lazy_schedule_cadence_and_microbatch_gating():
    sched = LazySchedule(
        enabled=True,
        input_every=8,
        output_every=8,
        warmup_until=8,
        warmup_every=2,
        capture_microbatches=1,
    )
    assert [sched.refresh_input(s) for s in range(8)] == [True, False, True, False, True, False, True, False]
    assert sched.refresh_input(8) is True
    assert any(sched.refresh_input(s) for s in range(9, 16)) is False
    assert sched.refresh_input(16) is True
    assert [sched.capture_microbatch(mb, 4) for mb in range(4)] == [False, False, False, True]


def test_exact_dense_and_headblock_powers_match_reference():
    gen = torch.Generator().manual_seed(42)
    X = torch.randn(3, 24, 16, generator=gen)
    C = X.mT @ X
    r_in, r_out = exact_dense_powers(C, damping=1e-3, inner_power=0.5, outer_power=0.125)
    for i in range(3):
        exp_in, exp_out_unit = inverse_factors(C[i], OutputSettings(damping=1e-3))
        torch.testing.assert_close(r_in[i], exp_in, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(unit_rms(r_out[i]), exp_out_unit, rtol=1e-5, atol=1e-6)

    # Head-block powers: 4 heads of dim 4 (total width 16)
    C_hb = torch.stack([C[:, h * 4 : (h + 1) * 4, h * 4 : (h + 1) * 4] for h in range(4)], dim=1)
    hb_in, hb_out = exact_head_block_powers(C_hb, damping=1e-3, inner_power=0.5, outer_power=0.125)
    assert hb_in.shape == (3, 4, 4, 4)
    for i in range(3):
        full_trace = C_hb[i].diagonal(dim1=-2, dim2=-1).sum()
        normed = C_hb[i] * (16.0 / full_trace)
        for h in range(4):
            lam, V = torch.linalg.eigh(0.5 * (normed[h] + normed[h].T))
            ref_in = (V * (lam.clamp_min(0) + 1e-3).pow(-0.5)) @ V.T
            torch.testing.assert_close(hb_in[i, h], ref_in, rtol=1e-5, atol=1e-6)


def test_headblock_and_metric_hyperball_per_matrix_vs_batched():
    gen = torch.Generator().manual_seed(2027)
    count, H, hd, cols = 4, 2, 4, 8
    rows = H * hd
    weights = torch.randn(count, rows, cols, generator=gen) * 0.05
    init_norms = weights.norm(dim=(-2, -1)).clone()
    grads = torch.randn(count, rows, cols, generator=gen) * 0.01
    momenta = torch.randn(count, rows, cols, generator=gen) * 0.01

    li_list, lo_list, ri_list, ro_list = [], [], [], []
    for _ in range(count):
        xh = torch.randn(H, 10, hd, generator=gen)
        ch = xh.mT @ xh
        li, lo = exact_head_block_powers(ch[None], damping=1e-3, inner_power=0.5, outer_power=0.125)
        li_list.append(li[0])
        lo_list.append(lo[0])
        xr = torch.randn(12, cols, generator=gen)
        ri, ro = exact_dense_powers((xr.T @ xr)[None], damping=1e-3, inner_power=0.5, outer_power=0.125)
        ri_list.append(ri[0])
        ro_list.append(ro[0])

    for mode in (DecayMode.DEFAULT, DecayMode.METRIC_HYPERBALL):
        exp_m, exp_d = [], []
        for idx in range(count):
            fac = MatrixFactors(li_list[idx], lo_list[idx], ri_list[idx], ro_list[idx])
            m_i, d_i = block_update(
                weights[idx],
                grads[idx],
                momenta[idx],
                fac,
                args=ARGS,
                lr=0.016,
                weight_decay=0.1,
                decay_mode=mode,
            )
            exp_m.append(m_i)
            exp_d.append(d_i)

        got_m, got_d = block_update_stack(
            weights,
            grads,
            momenta.clone(),
            (stack_factors(li_list, leading=count, width=rows), stack_factors(lo_list, leading=count, width=rows)),
            (stack_factors(ri_list, leading=count, width=cols), stack_factors(ro_list, leading=count, width=cols)),
            args=ARGS,
            lr=0.016,
            weight_decay=0.1,
            decay_mode=mode,
        )
        torch.testing.assert_close(got_m, torch.stack(exp_m), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(got_d, torch.stack(exp_d), rtol=1e-4, atol=1e-6)
        if mode == DecayMode.METRIC_HYPERBALL:
            updated_norms = (weights + got_d).norm(dim=(-2, -1))
            torch.testing.assert_close(updated_norms, init_norms, rtol=1e-5, atol=1e-6)
            # Zero weight_decay reduces identically to standard Hyperball
            zero_wd = metric_hyperball_delta(weights[0], grads[0], weights[0] @ ri_list[0], 0.016, 0.0)
            pure_hb = hyperball_delta(weights[0], grads[0], 0.016)
            torch.testing.assert_close(zero_wd, pure_hb, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "arm_name",
    [
        "inner_outer_mlpdiag_lag",
        "resdense_alldiag_lag",
        "headdiag_qk_mlpdiag_lag",
        "headdiag_qkv_mlpdiag_lag",
        "headdiag_qk_vdiag_mlpdiag_lag",
    ],
)
def test_lazy_role_arms_end_to_end_and_checkpoint_continuation(arm_name):
    assert arm_name in BLOCK_ARMS
    cfg = {
        "model": {
            "hidden": 16,
            "mlp": 32,
            "heads": 2,
            "layers": 2,
            "sequence_length": 16,
            "vocab_size": 64,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
            "residual_dtype": "f32",
            "embedding_init_scale": "sqrt_hidden",
        },
        "optimizer": {
            "name": "muont",
            "lr": 0.016,
            "auxiliary_lr": 0.0048,
            "weight_decay": 0.1,
            "head_weight_decay": 0.1,
            "betas": [0.8, 0.98],
            "eps": 1e-15,
            "clip_norm": 1.0,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "ns_eps": 1e-5,
            "ns_mode": "f32",
        },
        "statistics": {
            "momentum": 0.938,
            "head": 4,
            "stride": 4,
            "damping": 0.001,
            "inner_strength": 0.5,
            "outer_strength": 0.125,
        },
        "root": {
            "mode": "fast",
            "iterations": 2,
            "refresh": 4,
            "inline_steps": 4,
            "tolerance_kind": "floor",
            "tolerance_fraction": 0.06666666666666667,
            "min_tolerance": 5e-6,
            "epsilon": 1e-4,
            "rho_max": 0.2,
            "repair_rounds": 2,
            "graphs": False,
            "host_workers": 1,
            "host_threads": 1,
            "refresh_flagged": True,
            "chunk": 0,
            "begin": "head",
            "overlap": 1,
        },
        "experiment": {
            "arm": arm_name,
            "output": {
                "momentum": 0.938,
                "damping": 0.001,
                "inner_strength": 0.5,
                "outer_strength": 0.125,
                "refresh": 4,
            },
            "lazy": {
                "enabled": True,
                "input_every": 3,
                "output_every": 3,
                "warmup_until": 2,
                "warmup_every": 1,
                "capture_microbatches": 1,
                "input_ema_decay": 0.75,
                "output_ema_decay": 0.75,
                "exact_gpu_roots": True,
            },
        },
    }
    positions = sample_positions(16, 4, 4)
    rows_per_mb = 2 * len(positions)
    lazy = LazySchedule.from_config(cfg)

    def build_pair():
        m = build_model(cfg, positions=positions, row_capacity=rows_per_mb, rows_per_microbatch=rows_per_mb, device="cpu")
        initialize_model(m, 123, "sqrt_hidden")
        bt = BlockToner(
            m,
            cfg,
            positions,
            rows_per_microbatch=rows_per_mb,
            capacity=rows_per_mb,
            rows_total=rows_per_mb,
            world=1,
            rank=0,
            device="cpu",
        )
        bt.param_groups[0]["lr"] = 0.016
        return m, bt

    def run_step(m, bt, step_idx, batches):
        bt.zero_grad()
        do_in = lazy.refresh_input(step_idx)
        do_out = lazy.refresh_output(step_idx)
        bt.set_step_capture(capture_in=do_in, capture_out=do_out)
        for mb_idx, tok in enumerate(batches):
            cap = mb_idx == len(batches) - 1
            if do_in and cap:
                m.body.stat_at.copy_(torch.arange(rows_per_mb))
            bt.select_microbatch(0 if (do_out and cap) else mb_idx, capturing=(do_out and cap))
            # Synthetic f32 body check without FA2 doc_attn: directly exercise linear/observer forward + backward
            h = m.body.emb(tok)
            for blk in m.body.blocks:
                q, k, v = blk.q(h), blk.k(h), blk.v(h)
                attn_proxy = torch.tanh(q + k + v)
                x_mid = h + blk.o(attn_proxy)
                g, u = blk.gate(x_mid), blk.up(x_mid)
                h = x_mid + blk.down(torch.nn.functional.silu(g) * u)
                if do_in and cap:
                    blk.input_rows_attn.copy_(h[:, positions].reshape(-1, 16).bfloat16())
                    blk.input_rows_o.copy_(attn_proxy[:, positions].reshape(-1, 16).bfloat16())
                    blk.input_rows_mlp.copy_(x_mid[:, positions].reshape(-1, 16).bfloat16())
                    blk.input_rows_down.copy_((torch.nn.functional.silu(g) * u)[:, positions].reshape(-1, 32).bfloat16())
                if do_out and cap:
                    for kind in ("q", "k", "v", "o", "gate", "up", "down"):
                        side = bt.arm.output_side(kind)
                        w_out = 32 if kind in ("gate", "up") else 16
                        fake_cot = (h[..., :1].expand(-1, -1, w_out))[:, positions].reshape(-1, w_out)
                        if side in (Side.FULL, Side.HEAD_BLOCK):
                            bt.output_rows[0, kind].copy_(fake_cot.bfloat16())
                            bt.output_rows[1, kind].copy_((fake_cot * 1.1).bfloat16())
                        elif side == Side.DIAGONAL:
                            bt.output_rows[0, kind].copy_(fake_cot.bfloat16())
                            bt.output_rows[1, kind].copy_((fake_cot * 1.1).bfloat16())
            loss = (m.head(h).square().mean()) * 0.5
            if mb_idx == len(batches) - 1 and do_in:
                bt.begin()
            loss.backward()
        bt.finish_backward()
        bt.step()

    gen = torch.Generator().manual_seed(999)
    all_batches = [[torch.randint(0, 64, (2, 16), generator=gen) for _ in range(2)] for _ in range(6)]

    m_full, bt_full = build_pair()
    saved_model, saved_bt = None, None
    for s in range(6):
        run_step(m_full, bt_full, s, all_batches[s])
        if s == 2:
            saved_model = copy.deepcopy(m_full.state_dict())
            saved_bt = copy.deepcopy(bt_full.runtime_state())

    # Verify lazy refresh count: steps 0, 1 (warmup) and step 2 ((2-2)%3==0) and step 5 ((5-2)%3==0) = 4 refreshes out of 6 steps
    assert bt_full.input.steps == 4
    assert bt_full.output.steps == 4

    # Resume at step 3 (index 3..5) and verify exact parameter match
    m_res, bt_res = build_pair()
    m_res.load_state_dict(saved_model)
    bt_res.restore_runtime_state(saved_bt)
    for s in range(3, 6):
        run_step(m_res, bt_res, s, all_batches[s])

    for k in m_full.state_dict():
        torch.testing.assert_close(m_res.state_dict()[k], m_full.state_dict()[k], rtol=1e-6, atol=1e-7)


def test_async_host_both_lag_input_and_continuation():
    cfg = {
        "model": {
            "hidden": 16,
            "mlp": 32,
            "heads": 2,
            "layers": 2,
            "sequence_length": 16,
            "vocab_size": 64,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
            "residual_dtype": "f32",
            "embedding_init_scale": "sqrt_hidden",
        },
        "optimizer": {
            "name": "muont",
            "lr": 0.016,
            "auxiliary_lr": 0.0048,
            "weight_decay": 0.1,
            "head_weight_decay": 0.1,
            "betas": [0.8, 0.98],
            "eps": 1e-15,
            "clip_norm": 1.0,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "ns_eps": 1e-5,
            "ns_mode": "f32",
        },
        "statistics": {
            "momentum": 0.938,
            "head": 4,
            "stride": 4,
            "damping": 0.001,
            "inner_strength": 0.5,
            "outer_strength": 0.125,
        },
        "root": {
            "mode": "fast",
            "iterations": 2,
            "refresh": 4,
            "inline_steps": 4,
            "tolerance_kind": "floor",
            "tolerance_fraction": 0.06666666666666667,
            "min_tolerance": 5e-6,
            "epsilon": 1e-4,
            "rho_max": 0.2,
            "repair_rounds": 2,
            "graphs": False,
            "host_workers": 1,
            "host_threads": 1,
            "refresh_flagged": True,
            "chunk": 0,
            "begin": "head",
            "overlap": 1,
        },
        "experiment": {
            "arm": "headdiag_qk_mlpdiag_lag",
            "output": {
                "momentum": 0.938,
                "damping": 0.001,
                "inner_strength": 0.5,
                "outer_strength": 0.125,
                "refresh": 4,
            },
            "lazy": {
                "enabled": True,
                "input_every": 3,
                "output_every": 3,
                "warmup_until": 2,
                "warmup_every": 1,
                "capture_microbatches": 1,
                "input_ema_decay": 0.75,
                "output_ema_decay": 0.75,
                "root_backend": "async_host_both",
            },
        },
    }
    lazy = LazySchedule.from_config(cfg)
    assert lazy.lag_input is True
    assert lazy.root_backend == "async_host_both"
    positions = sample_positions(16, 4, 4)
    rows_per_mb = 2 * len(positions)

    def build_pair():
        m = build_model(cfg, positions=positions, row_capacity=rows_per_mb, rows_per_microbatch=rows_per_mb, device="cpu")
        initialize_model(m, 321, "sqrt_hidden")
        bt = BlockToner(
            m,
            cfg,
            positions,
            rows_per_microbatch=rows_per_mb,
            capacity=rows_per_mb,
            rows_total=rows_per_mb,
            world=1,
            rank=0,
            device="cpu",
        )
        bt.param_groups[0]["lr"] = 0.016
        return m, bt

    def run_step(m, bt, step_idx, batches):
        bt.zero_grad()
        do_in = lazy.refresh_input(step_idx)
        do_out = lazy.refresh_output(step_idx)
        bt.set_step_capture(capture_in=do_in, capture_out=do_out)
        for mb_idx, tok in enumerate(batches):
            cap = mb_idx == len(batches) - 1
            if do_in and cap:
                m.body.stat_at.copy_(torch.arange(rows_per_mb))
            bt.select_microbatch(0 if (do_out and cap) else mb_idx, capturing=(do_out and cap))
            h = m.body.emb(tok)
            for blk in m.body.blocks:
                q, k, v = blk.q(h), blk.k(h), blk.v(h)
                attn_proxy = torch.tanh(q + k + v)
                x_mid = h + blk.o(attn_proxy)
                g, u = blk.gate(x_mid), blk.up(x_mid)
                h = x_mid + blk.down(torch.nn.functional.silu(g) * u)
                if do_in and cap:
                    blk.input_rows_attn.copy_(h[:, positions].reshape(-1, 16).bfloat16())
                    blk.input_rows_o.copy_(attn_proxy[:, positions].reshape(-1, 16).bfloat16())
                    blk.input_rows_mlp.copy_(x_mid[:, positions].reshape(-1, 16).bfloat16())
                    blk.input_rows_down.copy_((torch.nn.functional.silu(g) * u)[:, positions].reshape(-1, 32).bfloat16())
                if do_out and cap:
                    for kind in ("q", "k", "v", "o", "gate", "up", "down"):
                        w_out = 32 if kind in ("gate", "up") else 16
                        fake_cot = (h[..., :1].expand(-1, -1, w_out))[:, positions].reshape(-1, w_out)
                        bt.output_rows[0, kind].copy_(fake_cot.bfloat16())
                        bt.output_rows[1, kind].copy_((fake_cot * 1.1).bfloat16())
            loss = (m.head(h).square().mean()) * 0.5
            if mb_idx == len(batches) - 1 and do_in:
                bt.begin()
            loss.backward()
        bt.finish_backward()
        bt.step()

    gen = torch.Generator().manual_seed(777)
    all_batches = [[torch.randint(0, 64, (2, 16), generator=gen) for _ in range(2)] for _ in range(6)]

    # Compare truly uninterrupted run (m_unint, no runtime_state call at s=2) vs checkpointed+resumed run
    m_unint, bt_unint = build_pair()
    for s in range(6):
        run_step(m_unint, bt_unint, s, all_batches[s])

    m_part1, bt_part1 = build_pair()
    for s in range(3):
        run_step(m_part1, bt_part1, s, all_batches[s])
    saved_model = copy.deepcopy(m_part1.state_dict())
    saved_bt = copy.deepcopy(bt_part1.runtime_state())

    m_res, bt_res = build_pair()
    m_res.load_state_dict(saved_model)
    bt_res.restore_runtime_state(saved_bt)
    for s in range(3, 6):
        run_step(m_res, bt_res, s, all_batches[s])

    assert bt_unint.input.steps == 4
    assert bt_res.input.steps == 4
    for k in m_unint.state_dict():
        torch.testing.assert_close(m_res.state_dict()[k], m_unint.state_dict()[k], rtol=1e-6, atol=1e-7)
