# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit & mathematical parity tests for all five Muon-family baselines."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from campaign.configure_baselines_campaign import (
    PRIMARY_ARMS,
    SECONDARY_ARMS,
    build_baseline_130m_config,
)
from experiment.baselines import BASELINE_ARMS, _ns_orthogonalize, _shape_scale, build_baseline


class TinyBlock(nn.Module):
    def __init__(self, hidden: int = 16, mlp: int = 64):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(hidden, mlp, bias=False)
        self.up = nn.Linear(hidden, mlp, bias=False)
        self.down = nn.Linear(mlp, hidden, bias=False)
        self.register_buffer("input_rows_attn", torch.randn(32, hidden), persistent=False)
        self.register_buffer("input_rows_o", torch.randn(32, hidden), persistent=False)
        self.register_buffer("input_rows_mlp", torch.randn(32, hidden), persistent=False)
        self.register_buffer("input_rows_down", torch.randn(32, mlp), persistent=False)


class TinyModel(nn.Module):
    def __init__(self, layers: int = 2, hidden: int = 16, mlp: int = 64):
        super().__init__()
        self.body = nn.Module()
        self.body.blocks = nn.ModuleList([TinyBlock(hidden, mlp) for _ in range(layers)])
        self.head = nn.Linear(hidden, 32, bias=False)


def _tiny_config(arm_key: str) -> dict:
    cfg = build_baseline_130m_config(arm_key=arm_key, seed=0, lr=0.016)
    cfg["model"]["hidden"] = 16
    cfg["model"]["mlp"] = 64
    cfg["model"]["layers"] = 2
    cfg["optimizer"]["ns_steps"] = 5
    cfg["optimizer"]["ns_mode"] = "fp32"
    return cfg


def _assign_grads(model: TinyModel, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.1


def test_normuon_parity() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    cfg = _tiny_config("normuon")
    opt = build_baseline(model, cfg, world=1, rank=0, device=torch.device("cpu"))
    w0 = model.body.blocks[0].down.weight.detach().clone()
    _assign_grads(model, 42)
    grad = model.body.blocks[0].down.weight.grad.detach().clone().float()

    opt.step()
    w1 = model.body.blocks[0].down.weight.detach().clone()

    # Exact reference from zichongli5/NorMuon/normuon.py lines 48-56
    beta1 = cfg["optimizer"]["momentum"]
    beta2 = cfg["experiment"]["baseline"]["beta2"]
    eps = cfg["experiment"]["baseline"]["normuon_eps"]
    buf = grad.clone()
    src = grad + beta1 * buf
    ortho = _ns_orthogonalize(src, cfg["optimizer"])
    vnorm = ortho.norm(dim=(-2, -1), keepdim=True)
    v_mean = (ortho * ortho).mean(dim=-1, keepdim=True)
    v2 = (1.0 - beta2) * v_mean
    scaled = ortho / (v2.sqrt() + eps)
    vnorm_new = scaled.norm(dim=(-2, -1), keepdim=True)
    direction = scaled * (vnorm / (vnorm_new + eps))
    scale = _shape_scale(w0.shape)
    expected = w0 - cfg["optimizer"]["lr"] * (scale * direction + cfg["optimizer"]["weight_decay"] * w0)
    err = (w1 - expected).abs().max().item()
    assert err < 1e-6, f"NorMuon parity error {err}"


def test_muown_parity() -> None:
    torch.manual_seed(1)
    model = TinyModel()
    cfg = _tiny_config("muown")
    opt = build_baseline(model, cfg, world=1, rank=0, device=torch.device("cpu"))
    w0 = model.body.blocks[0].q.weight.detach().clone().float()
    _assign_grads(model, 43)
    grad = model.body.blocks[0].q.weight.grad.detach().clone().float()

    opt.step()
    w1 = model.body.blocks[0].q.weight.detach().clone().float()

    # Exact reference from kcc-lion/muown/optim/muown.py
    lr = cfg["optimizer"]["lr"]
    wd = cfg["optimizer"]["weight_decay"]
    bcfg = cfg["experiment"]["baseline"]
    g = w0.norm(dim=1, keepdim=True)
    vn = g.clone()
    u = w0 / g
    v = u * vn
    grad_g = (grad * u).sum(dim=1, keepdim=True)
    grad_v = (g / vn) * (grad - u * grad_g)
    src = grad_v + cfg["optimizer"]["momentum"] * grad_v
    ortho = _ns_orthogonalize(src, cfg["optimizer"])
    v_new = v - lr * _shape_scale(w0.shape) * ortho
    beta1, beta2 = bcfg["g_betas"]
    m_g = (1 - beta1) * grad_g
    v_g = (1 - beta2) * (grad_g * grad_g)
    g_new = g - (lr * bcfg["g_lr_ratio"]) * (m_g / (1 - beta1)) / ((v_g / (1 - beta2)).sqrt() + bcfg["g_eps"])
    w_exp = g_new * (v_new / v_new.norm(dim=1, keepdim=True)) - (lr * wd) * w0
    err = (w1 - w_exp).abs().max().item()
    assert err < 1e-6, f"Muown parity error {err}"


def test_muoneq_parity() -> None:
    for arm in ("muoneq_r", "muoneq_rc"):
        torch.manual_seed(2)
        model = TinyModel()
        cfg = _tiny_config(arm)
        opt = build_baseline(model, cfg, world=1, rank=0, device=torch.device("cpu"))
        w0 = model.body.blocks[0].gate.weight.detach().clone().float()
        _assign_grads(model, 44)
        grad = model.body.blocks[0].gate.weight.grad.detach().clone().float()

        opt.step()
        w1 = model.body.blocks[0].gate.weight.detach().clone().float()

        src = grad + cfg["optimizer"]["momentum"] * grad
        eps = cfg["experiment"]["baseline"]["rowcol_eps"]
        r = (src.square().sum(dim=1) + eps).rsqrt()
        if arm == "muoneq_r":
            eq = src * r[:, None]
        else:
            c = (src.square().sum(dim=0) + eps).rsqrt()
            eq = (src * r[:, None]) * c[None, :]
        ortho = _ns_orthogonalize(eq, cfg["optimizer"])
        expected = w0 - cfg["optimizer"]["lr"] * (_shape_scale(w0.shape) * ortho + cfg["optimizer"]["weight_decay"] * w0)
        err = (w1 - expected).abs().max().item()
        assert err < 1e-6, f"{arm} parity error {err}"


def test_muon2_parity() -> None:
    for arm in ("muon2", "muon2_f"):
        torch.manual_seed(3)
        model = TinyModel()
        cfg = _tiny_config(arm)
        opt = build_baseline(model, cfg, world=1, rank=0, device=torch.device("cpu"))
        w0 = model.body.blocks[0].up.weight.detach().clone().float()
        _assign_grads(model, 45)
        grad = model.body.blocks[0].up.weight.grad.detach().clone().float()

        opt.step()
        w1 = model.body.blocks[0].up.weight.detach().clone().float()

        b1 = cfg["optimizer"]["momentum"]
        b2 = cfg["experiment"]["baseline"]["beta2"]
        eps = cfg["experiment"]["baseline"]["muon2_eps"]
        buf = (1 - b1) * grad
        m_src = grad.lerp(buf, b1)
        g_sq = grad * grad
        if arm == "muon2":
            v = (1 - b2) * g_sq
        else:
            r = (1 - b2) * g_sq.sum(dim=1)
            c = (1 - b2) * g_sq.sum(dim=0)
            v = (r[:, None] * c[None, :]) / r.sum()
        ortho = _ns_orthogonalize(m_src / (v.sqrt() + eps), cfg["optimizer"])
        expected = w0 - cfg["optimizer"]["lr"] * (_shape_scale(w0.shape) * ortho + cfg["optimizer"]["weight_decay"] * w0)
        err = (w1 - expected).abs().max().item()
        assert err < 1e-6, f"{arm} parity error {err}"


def test_newton_muon_and_checkpoint_resume() -> None:
    for arm in (*PRIMARY_ARMS, *SECONDARY_ARMS):
        torch.manual_seed(10)
        m_full = TinyModel()
        m_res = copy.deepcopy(m_full)
        cfg = _tiny_config(arm)
        opt_full = build_baseline(m_full, cfg, rows_total=32, world=1, rank=0, device=torch.device("cpu"))
        opt_res = build_baseline(m_res, cfg, rows_total=32, world=1, rank=0, device=torch.device("cpu"))

        for s in range(3):
            _assign_grads(m_full, 100 + s)
            _assign_grads(m_res, 100 + s)
            opt_full.step()
            opt_res.step()

        # Save & restore into a brand new optimizer instance
        saved_state = copy.deepcopy(opt_res.runtime_state())
        opt_res2 = build_baseline(m_res, cfg, rows_total=32, world=1, rank=0, device=torch.device("cpu"))
        opt_res2.restore_runtime_state(saved_state)

        for s in range(3, 6):
            _assign_grads(m_full, 100 + s)
            _assign_grads(m_res, 100 + s)
            opt_full.step()
            opt_res2.step()

        for p1, p2 in zip(m_full.parameters(), m_res.parameters(), strict=True):
            diff = (p1 - p2).abs().max().item()
            assert diff == 0.0, f"{arm} checkpoint resume mismatch: {diff}"
            assert torch.isfinite(p1).all().item(), f"{arm} produced nonfinite weights"


def main() -> None:
    test_normuon_parity()
    test_muown_parity()
    test_muoneq_parity()
    test_muon2_parity()
    test_newton_muon_and_checkpoint_resume()
    print(f"ALL TESTS PASSED ({len(BASELINE_ARMS)} registered arms verified)")


if __name__ == "__main__":
    main()
