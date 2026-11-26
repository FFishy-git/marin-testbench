# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A100 GPU verification of default bitwise parity and isotropic / two-sided Ozaki accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from experiment import reference as _ref  # noqa: F401
from experiment.block_toner import (
    DecayMode,
    _compiled_block_in,
    _compiled_block_out,
    _prepare_side_factor,
    block_update_stack,
    stack_unit_rms,
)
from experiment.factors import OutputSettings, inverse_factors
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


def fp64_reference_delta(
    weights: torch.Tensor,
    direction: torch.Tensor,
    left_outer: torch.Tensor | None,
    right_outer: torch.Tensor | None,
    lr: float,
    wd: float,
    decay_mode: str,
) -> torch.Tensor:
    w64 = weights.double()
    d64 = direction.double()
    scale = max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5

    if right_outer is not None:
        r_hat = stack_unit_rms(right_outer).double()
        d64 = torch.bmm(d64, r_hat) if r_hat.ndim == 3 else d64 * r_hat[:, None, :]
    else:
        r_hat = None

    if left_outer is not None:
        l_hat = stack_unit_rms(left_outer).double()
        d64 = torch.bmm(l_hat, d64) if l_hat.ndim == 3 else d64 * l_hat[:, :, None]
    else:
        l_hat = None

    if decay_mode == "isotropic":
        q64 = w64
    elif decay_mode == "default":
        if r_hat is not None:
            wr = torch.bmm(w64, r_hat) if r_hat.ndim == 3 else w64 * r_hat[:, None, :]
            q64 = torch.bmm(wr, r_hat) if r_hat.ndim == 3 else wr * r_hat[:, None, :]
        elif l_hat is not None:
            lw = torch.bmm(l_hat, w64) if l_hat.ndim == 3 else l_hat[:, :, None] * w64
            q64 = torch.bmm(l_hat, lw) if l_hat.ndim == 3 else l_hat[:, :, None] * lw
        else:
            q64 = w64
    elif decay_mode == "two_sided":
        q64 = w64
        if r_hat is not None:
            q64 = torch.bmm(q64, r_hat) if r_hat.ndim == 3 else q64 * r_hat[:, None, :]
            q64 = torch.bmm(q64, r_hat) if r_hat.ndim == 3 else q64 * r_hat[:, None, :]
        if l_hat is not None:
            q64 = torch.bmm(l_hat, q64) if l_hat.ndim == 3 else l_hat[:, :, None] * q64
            q64 = torch.bmm(l_hat, q64) if l_hat.ndim == 3 else l_hat[:, :, None] * q64
    else:
        raise ValueError(decay_mode)

    return (-lr * (scale * d64 + wd * q64)).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", 0)
    gen = torch.Generator().manual_seed(20261127)
    opt_args = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="tpu")
    lr, wd = 0.016, 0.1

    cases = [
        ("qkv_dense_left", 4, 512, 512, "dense"),
        ("down_dense_left", 4, 512, 2048, "dense"),
        ("gate_up_diag_left", 4, 2048, 512, "diag"),
        ("step1_no_left_512", 4, 512, 512, "none"),
        ("step1_no_left_2048", 4, 2048, 512, "none"),
    ]

    records = []
    for label, bsz, rows, cols, left_kind in cases:
        weights = (torch.randn(bsz, rows, cols, generator=gen) * 0.02).to(device)
        grads = (torch.randn(bsz, rows, cols, generator=gen) * 0.005).to(device)
        momenta = (torch.randn(bsz, rows, cols, generator=gen) * 0.01).to(device)
        ri, ro = make_dense_factors(bsz, cols, gen, device)
        if left_kind == "dense":
            li, lo = make_dense_factors(bsz, rows, gen, device)
        elif left_kind == "diag":
            li, lo = make_diag_factors(bsz, rows, gen, device)
        else:
            li, lo = None, None

        # Original path bitwise parity check for default mode
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
            ri.ndim == 3,
            li is not None and li.ndim == 3,
        )
        direction = MU.newton_schulz_batched(src, opt_args.muon_ns_steps, opt_args.muon_eps, opt_args.muon_ns_mode)
        lr_t = torch.tensor(lr, dtype=torch.float32, device=device)
        wd_t = torch.tensor(wd, dtype=torch.float32, device=device)
        scale = max(1.0, rows / cols) ** 0.5
        orig_delta = _compiled_block_out(
            weights, direction, lo_prep, ro_prep, lr_t, wd_t, scale, ro.ndim == 3, lo is not None and lo.ndim == 3
        )

        _, def_delta = block_update_stack(
            weights, grads, momenta, (li, lo), (ri, ro), args=opt_args, lr=lr, weight_decay=wd, compiled=True, decay_mode=DecayMode.DEFAULT
        )
        bitwise_diff = (orig_delta - def_delta).abs().max().item()
        assert bitwise_diff == 0.0, f"{label}: default bitwise_diff={bitwise_diff}"

        mode_errors = {}
        for mode in ("default", "isotropic", "two_sided"):
            _, actual_delta = block_update_stack(
                weights,
                grads,
                momenta,
                (li, lo),
                (ri, ro),
                args=opt_args,
                lr=lr,
                weight_decay=wd,
                compiled=True,
                decay_mode=mode,
            )
            ref_delta = fp64_reference_delta(weights, direction, lo, ro, lr, wd, mode)
            rel_err = ((actual_delta - ref_delta).double().norm() / ref_delta.double().norm()).item()
            max_abs = (actual_delta - ref_delta).abs().max().item()
            assert rel_err < 1e-5, f"{label} ({mode}): rel_err={rel_err}"
            mode_errors[mode] = {"rel_l2_error": rel_err, "max_abs_error": max_abs}

        records.append({"case": label, "default_bitwise_diff": bitwise_diff, "modes": mode_errors})

    out_payload = {"time": datetime.now(UTC).isoformat(), "passed": True, "cases": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2) + "\n")
    print(json.dumps(out_payload))


if __name__ == "__main__":
    main()
