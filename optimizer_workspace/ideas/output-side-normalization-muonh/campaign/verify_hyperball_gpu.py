# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A100 GPU verification of default bitwise parity, Hyperball Ozaki FP64 accuracy, and Frobenius sphere conservation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from experiment import reference as _ref  # noqa: F401
from experiment.block_toner import (
    HYPERBALL_NORM_FLOOR,
    DecayMode,
    _compiled_block_in,
    _compiled_block_out,
    _prepare_side_factor,
    block_update_stack,
    stack_unit_rms,
)
from experiment.factors import OutputSettings, inverse_factors
from experiment.optimizer import AdamH
from toner.fast import muon as MU


def make_dense_factors(batch: int, dim: int, gen: torch.Generator, device: torch.device):
    inners, outers = [], []
    settings = OutputSettings()
    for _ in range(batch):
        x = torch.randn(dim, dim + 16, generator=gen, dtype=torch.float32)
        gram = (x @ x.T) / (dim + 16)
        inner, outer = inverse_factors(gram, settings)
        inners.append(inner)
        outers.append(outer)
    return torch.stack(inners).to(device), torch.stack(outers).to(device)


def make_diag_factors(batch: int, dim: int, gen: torch.Generator, device: torch.device):
    inner = (torch.rand(batch, dim, generator=gen, dtype=torch.float32) * 3.0 + 0.1).to(device)
    outer = (torch.rand(batch, dim, generator=gen, dtype=torch.float32) * 2.0 + 0.2).to(device)
    return inner, outer


def fp64_hyperball_reference_delta(
    weights: torch.Tensor,
    direction: torch.Tensor,
    left_outer: torch.Tensor | None,
    right_outer: torch.Tensor | None,
    lr: float,
) -> torch.Tensor:
    w64 = weights.double()
    d64 = direction.double()
    if right_outer is not None:
        r_hat = stack_unit_rms(right_outer).double()
        d64 = torch.bmm(d64, r_hat) if r_hat.ndim == 3 else d64 * r_hat[:, None, :]
    if left_outer is not None:
        l_hat = stack_unit_rms(left_outer).double()
        d64 = torch.bmm(l_hat, d64) if l_hat.ndim == 3 else d64 * l_hat[:, :, None]
    dims = (-2, -1)
    p_norm = w64.square().sum(dim=dims, keepdim=True).sqrt()
    u_norm = d64.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    p_mid = w64 - lr * d64 * (p_norm / u_norm)
    p_mid_norm = p_mid.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    return (p_mid * (p_norm / p_mid_norm) - w64).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", 0)
    gen = torch.Generator().manual_seed(20261127)
    opt_args = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="tpu")
    lr = 0.032

    cases = [
        ("muonh_512x512", 4, 512, 512, "none", "none"),
        ("muonh_2048x512", 4, 2048, 512, "none", "none"),
        ("muonth_512x512", 4, 512, 512, "none", "dense"),
        ("muonth_512x2048", 4, 512, 2048, "none", "dense"),
        ("wt_mlp_lag_h_qkv", 4, 512, 512, "dense", "dense"),
        ("wt_mlp_lag_h_down", 4, 512, 2048, "dense", "dense"),
        ("wt_mlp_lag_h_gate_up", 4, 2048, 512, "diag", "dense"),
    ]

    records = []
    for label, bsz, rows, cols, left_kind, right_kind in cases:
        weights = (torch.randn(bsz, rows, cols, generator=gen) * 0.02).to(device)
        grads = (torch.randn(bsz, rows, cols, generator=gen) * 0.005).to(device)
        momenta = (torch.randn(bsz, rows, cols, generator=gen) * 0.01).to(device)
        ri, ro = make_dense_factors(bsz, cols, gen, device) if right_kind == "dense" else (None, None)
        if left_kind == "dense":
            li, lo = make_dense_factors(bsz, rows, gen, device)
        elif left_kind == "diag":
            li, lo = make_diag_factors(bsz, rows, gen, device)
        else:
            li, lo = None, None

        # 1. Default bitwise parity check when right factors exist
        bitwise_diff = 0.0
        if right_kind == "dense":
            li_prep = _prepare_side_factor(li, side="left", normalize_rms=False, compiled=True)
            lo_prep = _prepare_side_factor(lo, side="left", normalize_rms=True, compiled=True)
            ri_prep = _prepare_side_factor(ri, side="right", normalize_rms=False, compiled=True)
            ro_prep = _prepare_side_factor(ro, side="right", normalize_rms=True, compiled=True)
            _, src = _compiled_block_in(
                grads,
                momenta,
                li_prep,
                ri_prep,
                float(opt_args.muon_momentum),
                bool(opt_args.muon_nesterov),
                True,
                li is not None and li.ndim == 3,
            )
            direction = MU.newton_schulz_batched(src, opt_args.muon_ns_steps, opt_args.muon_eps, opt_args.muon_ns_mode)
            lr_t = torch.tensor(0.016, dtype=torch.float32, device=device)
            wd_t = torch.tensor(0.1, dtype=torch.float32, device=device)
            scale = max(1.0, rows / cols) ** 0.5
            orig_delta = _compiled_block_out(
                weights, direction, lo_prep, ro_prep, lr_t, wd_t, scale, True, lo is not None and lo.ndim == 3
            )
            _, def_delta = block_update_stack(
                weights, grads, momenta, (li, lo), (ri, ro), args=opt_args, lr=0.016, weight_decay=0.1, compiled=True, decay_mode=DecayMode.DEFAULT
            )
            bitwise_diff = (orig_delta - def_delta).abs().max().item()
            assert bitwise_diff == 0.0, f"{label}: default bitwise_diff={bitwise_diff}"

        # 2. Hyperball FP64 Levanter accuracy check
        li_prep = _prepare_side_factor(li, side="left", normalize_rms=False, compiled=True)
        ri_prep = _prepare_side_factor(ri, side="right", normalize_rms=False, compiled=True)
        _, src = _compiled_block_in(
            grads,
            momenta,
            li_prep,
            ri_prep,
            float(opt_args.muon_momentum),
            bool(opt_args.muon_nesterov),
            ri is not None and ri.ndim == 3,
            li is not None and li.ndim == 3,
        )
        direction = MU.newton_schulz_batched(src, opt_args.muon_ns_steps, opt_args.muon_eps, opt_args.muon_ns_mode)
        _, hb_delta = block_update_stack(
            weights, grads, momenta, (li, lo), (ri, ro), args=opt_args, lr=lr, weight_decay=0.0, compiled=True, decay_mode=DecayMode.HYPERBALL
        )
        ref_delta = fp64_hyperball_reference_delta(weights, direction, lo, ro, lr)
        rel_err = ((hb_delta - ref_delta).double().norm() / ref_delta.double().norm()).item()
        assert rel_err < 1e-5, f"{label} (hyperball): rel_err={rel_err}"

        # 3. 50-step Frobenius norm conservation check on GPU
        w_run = weights.clone()
        m_run = momenta.clone()
        init_norms = w_run.norm(dim=(-2, -1))
        for _ in range(50):
            g_step = torch.randn_like(w_run) * 0.005
            m_run, d_step = block_update_stack(
                w_run, g_step, m_run, (li, lo), (ri, ro), args=opt_args, lr=lr, weight_decay=0.0, compiled=True, decay_mode=DecayMode.HYPERBALL
            )
            w_run = w_run + d_step
        max_rel_drift = ((w_run.norm(dim=(-2, -1)) - init_norms).abs() / init_norms).max().item()
        assert max_rel_drift < 1e-5, f"{label}: 50-step norm drift={max_rel_drift}"

        records.append(
            {
                "case": label,
                "default_bitwise_diff": bitwise_diff,
                "hyperball_rel_l2_vs_fp64": rel_err,
                "max_50step_rel_norm_drift": max_rel_drift,
            }
        )

    # 4. AdamH 50-step Frobenius norm conservation on [128256, 512] head shape
    head_p = nn.Parameter((torch.randn(4096, 512, generator=gen) * (512**-0.5)).to(device))
    head_init_norm = head_p.detach().norm().item()
    opt_head = AdamH([head_p], lr=lr, betas=(0.8, 0.98), eps=1e-15)
    for _ in range(50):
        head_p.grad = torch.randn_like(head_p) * 0.002
        opt_head.step()
    head_rel_drift = abs(head_p.detach().norm().item() - head_init_norm) / head_init_norm
    assert head_rel_drift < 1e-5, f"AdamH 50-step norm drift={head_rel_drift}"

    out_payload = {
        "time": datetime.now(UTC).isoformat(),
        "passed": True,
        "adamh_50step_rel_norm_drift": head_rel_drift,
        "cases": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2) + "\n")
    print(json.dumps(out_payload))


if __name__ == "__main__":
    main()
