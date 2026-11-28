# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controlled distributed implementations of five recent Muon-family baselines.

Baselines implemented (all inheriting MatrixBaseline for identical 224-matrix
ownership, FP32 additive updates, DDP synchronization, and checkpoint resume):

1. ``muown``      -- Muown: Row-Norm Control for Muon Optimization
                     (arXiv:2605.10797; official repo kcc-lion/muown; also used
                     in MiMo-V2.6 arXiv:2610.11959v1).
2. ``normuon``    -- NorMuon: Making Muon more efficient and scalable
                     (arXiv:2510.05491; official repo zichongli5/NorMuon).
3. ``muon2``      -- Muon^2: Boosting Muon via Adaptive Second-Moment
   ``muon2_f``       Preconditioning (arXiv:2604.09967, Algorithms 1 & 2).
4. ``muoneq_r``   -- MuonEq: Balancing Before Orthogonalization with Lightweight
   ``muoneq_rc``     Equilibration (arXiv:2603.28254; official repo MaeChd/muon-eq).
5. ``newton_muon``-- The Newton-Muon Optimizer
                     (arXiv:2604.01472; official repo zhehangdu/Newton-Muon).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .block_toner import INPUT_KIND, _newton_schulz
from .related_base import MatrixBaseline, MuonControl


def _ns_orthogonalize(source: torch.Tensor, settings: dict) -> torch.Tensor:
    return _newton_schulz(
        source,
        int(settings["ns_steps"]),
        float(settings["ns_eps"]),
        str(settings.get("ns_mode", "tpu")),
        str(settings.get("ns_coeffs", "simple")),
    )


def _shape_scale(shape: tuple[int, int]) -> float:
    return max(1.0, shape[0] / shape[1]) ** 0.5


class NorMuonBaseline(MatrixBaseline):
    """NorMuon (arXiv:2510.05491; zichongli5/NorMuon).

    Applies post-orthogonalization neuron-wise (row-wise) second-moment
    normalization with exact Frobenius norm preservation:
        vnorm     = ||O||_F
        v_mean    = mean(O^2, dim=-1, keepdim=True)
        v_t       = beta2 * v_{t-1} + (1 - beta2) * v_mean
        O_scaled  = O / (sqrt(v_t) + eps)
        O_final   = O_scaled * (vnorm / (||O_scaled||_F + eps))
    """

    def __init__(self, model, config, *, positions=None, rows_per_microbatch=0, capacity=0, rows_total=0, world, rank, device):
        super().__init__(model, config, world=world, rank=rank, device=device)
        bcfg = config.get("experiment", {}).get("baseline", {})
        self.beta2 = float(bcfg.get("beta2", 0.95))
        self.normuon_eps = float(bcfg.get("normuon_eps", 1e-10))
        self.second_momentum = {
            key: torch.zeros((param.shape[0], 1), device=device, dtype=torch.float32)
            for key, param in self.owned.items()
        }

    @torch.no_grad()
    def step(self):
        settings = self.config["optimizer"]
        group = self.param_groups[0]
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        deltas = {}
        for key, param in self.owned.items():
            source = self.momentum_source(key, param.grad)
            ortho = _ns_orthogonalize(source, settings)
            vnorm = ortho.norm(dim=(-2, -1), keepdim=True)
            v_mean = (ortho * ortho).mean(dim=-1, keepdim=True)
            v_buf = self.second_momentum[key]
            v_buf.lerp_(v_mean, 1.0 - self.beta2)
            step_size = 1.0 / v_buf.sqrt().add_(self.normuon_eps)
            scaled = ortho * step_size
            vnorm_new = scaled.norm(dim=(-2, -1), keepdim=True)
            direction = scaled * (vnorm / (vnorm_new + self.normuon_eps))
            scale = _shape_scale((param.shape[0], param.shape[1]))
            deltas[key] = -lr * (scale * direction + wd * param.float())
        self.apply_updates(deltas)
        self.last_metrics = {"baseline_steps": self.steps}

    def runtime_state(self):
        state = self.base_runtime_state()
        state["second_momentum"] = self.second_momentum
        return state

    def restore_runtime_state(self, state):
        self.restore_base_runtime_state(state)
        for key, val in state["second_momentum"].items():
            self.second_momentum[key].copy_(val)


class MuownBaseline(MatrixBaseline):
    """Muown: Row-Norm Control for Muon Optimization (arXiv:2605.10797; kcc-lion/muown).

    Maintains an implicit weight-normalization parameterization W = g * (v / ||v||_row)
    purely inside the optimizer:
      - Direction v updated via Muon (momentum + Nesterov + Newton-Schulz).
      - Row-magnitude vector g (shape [out, 1]) updated via bias-corrected Adam.
      - Recomposes W_new = g * (v_new / ||v_new||_row), applies decoupled WD
        (-lr * wd * W_old), and resyncs g = ||W_new||_row.

    LR scaling note:
      In kcc-lion/muown, v uses `lr_base * 0.2 * sqrt(max(m, n))` and g uses
      `lr_base`. In Marin/Toner convention, `optimizer.lr` is already the Muon
      matrix step size (scaled by `max(1, m/n)^0.5`), while `auxiliary_lr` is
      the AdamW step size (`0.0048 / 0.016 = 0.30`). We expose `g_lr_ratio`
      (default `auxiliary_lr / lr` = 0.30) so g updates at the matched AdamW
      step scale while respecting the shared WSD schedule.
    """

    def __init__(self, model, config, *, positions=None, rows_per_microbatch=0, capacity=0, rows_total=0, world, rank, device):
        super().__init__(model, config, world=world, rank=rank, device=device)
        bcfg = config.get("experiment", {}).get("baseline", {})
        opt = config["optimizer"]
        self.g_betas = tuple(float(x) for x in bcfg.get("g_betas", (0.9, 0.95)))
        self.g_eps = float(bcfg.get("g_eps", 1e-8))
        default_ratio = float(opt["auxiliary_lr"]) / max(float(opt["lr"]), 1e-12)
        self.g_lr_ratio = float(bcfg.get("g_lr_ratio", default_ratio))
        self.norm_floor = float(bcfg.get("norm_floor", 1e-12))

        self.g_mag: dict[tuple[int, str], torch.Tensor] = {}
        self.v_norm: dict[tuple[int, str], torch.Tensor] = {}
        self.m_g: dict[tuple[int, str], torch.Tensor] = {}
        self.v_g: dict[tuple[int, str], torch.Tensor] = {}
        for key, param in self.owned.items():
            w_norm = param.detach().float().norm(dim=1, keepdim=True).clamp_min(self.norm_floor)
            self.g_mag[key] = w_norm.clone()
            self.v_norm[key] = w_norm.clone()
            self.m_g[key] = torch.zeros_like(w_norm)
            self.v_g[key] = torch.zeros_like(w_norm)

    @torch.no_grad()
    def step(self):
        settings = self.config["optimizer"]
        group = self.param_groups[0]
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        beta1, beta2 = self.g_betas
        step_idx = self.steps + 1
        bc1 = 1.0 - beta1**step_idx
        bc2 = 1.0 - beta2**step_idx
        g_lr = lr * self.g_lr_ratio

        deltas = {}
        for key, param in self.owned.items():
            w_old = param.float()
            grad_w = param.grad.float()
            g = self.g_mag[key]
            vn = self.v_norm[key]

            # Exact _wn_pre_ns Jacobian from kcc-lion/muown/optim/muown.py
            g_safe = g.clamp_min(self.norm_floor)
            vn_safe = vn.clamp_min(self.norm_floor)
            u = w_old / g_safe
            v = u * vn_safe
            grad_g = (grad_w * u).sum(dim=1, keepdim=True)
            grad_v = (g_safe / vn_safe) * (grad_w - u * grad_g)

            # Muon update on direction v
            source = self.momentum_source(key, grad_v)
            ortho = _ns_orthogonalize(source, settings)
            scale = _shape_scale((param.shape[0], param.shape[1]))
            v_new = v - (lr * scale) * ortho

            # Bias-corrected Adam update on row magnitudes g
            mg = self.m_g[key]
            vg = self.v_g[key]
            mg.mul_(beta1).add_(grad_g, alpha=1.0 - beta1)
            vg.mul_(beta2).addcmul_(grad_g, grad_g, value=1.0 - beta2)
            g.addcdiv_(mg / bc1, (vg / bc2).sqrt().add_(self.g_eps), value=-g_lr)

            # Recompose W_new = g * v_new / ||v_new||_row
            vn_new = v_new.norm(dim=1, keepdim=True).clamp_min(self.norm_floor)
            w_new = g * (v_new / vn_new)
            self.v_norm[key].copy_(vn_new)

            if wd != 0.0:
                w_new = w_new - (lr * wd) * w_old
                g.copy_(w_new.norm(dim=1, keepdim=True).clamp_min(self.norm_floor))

            deltas[key] = w_new - w_old

        self.apply_updates(deltas)
        self.last_metrics = {"baseline_steps": self.steps}

    def runtime_state(self):
        state = self.base_runtime_state()
        state.update({"g_mag": self.g_mag, "v_norm": self.v_norm, "m_g": self.m_g, "v_g": self.v_g})
        return state

    def restore_runtime_state(self, state):
        self.restore_base_runtime_state(state)
        for name in ("g_mag", "v_norm", "m_g", "v_g"):
            dst = getattr(self, name)
            for key, val in state[name].items():
                dst[key].copy_(val)


class Muon2Baseline(MatrixBaseline):
    """Muon^2 & Muon^2-F (arXiv:2604.09967, Algorithms 1 & 2).

    Applies Adam-style element-wise (or rank-1 factorized for Muon^2-F)
    second-moment preconditioning to the momentum matrix *before* Newton-Schulz:
        M_t      = beta1 * M_{t-1} + (1 - beta1) * G_t   (or Nesterov)
        V_t      = beta2 * V_{t-1} + (1 - beta2) * (G_t^2)
        M_tilde  = M_t / (sqrt(V_t) + eps)
        O_t      = NS(M_tilde, K)
    """

    def __init__(self, model, config, *, positions=None, rows_per_microbatch=0, capacity=0, rows_total=0, world, rank, device):
        super().__init__(model, config, world=world, rank=rank, device=device)
        bcfg = config.get("experiment", {}).get("baseline", {})
        arm = config["experiment"]["arm"]
        self.factorized = bool(bcfg.get("factorized", arm == "muon2_f"))
        self.beta2 = float(bcfg.get("beta2", 0.95))
        self.muon2_eps = float(bcfg.get("muon2_eps", 1e-8))
        self.use_nesterov = bool(bcfg.get("use_nesterov", config["optimizer"].get("nesterov", True)))

        if self.factorized:
            self.row_v = {
                key: torch.zeros(param.shape[0], device=device, dtype=torch.float32)
                for key, param in self.owned.items()
            }
            self.col_v = {
                key: torch.zeros(param.shape[1], device=device, dtype=torch.float32)
                for key, param in self.owned.items()
            }
            self.second_moment = {}
        else:
            self.second_moment = {
                key: torch.zeros_like(param, dtype=torch.float32)
                for key, param in self.owned.items()
            }
            self.row_v = {}
            self.col_v = {}

    @torch.no_grad()
    def step(self):
        settings = self.config["optimizer"]
        group = self.param_groups[0]
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        beta1 = float(settings["momentum"])

        deltas = {}
        for key, param in self.owned.items():
            g = param.grad.float()
            buf = self.momentum[key]
            buf.lerp_(g, 1.0 - beta1)
            m_src = g.lerp(buf, beta1) if self.use_nesterov else buf

            g_sq = g * g
            if self.factorized:
                r = self.row_v[key]
                c = self.col_v[key]
                r.lerp_(g_sq.sum(dim=1), 1.0 - self.beta2)
                c.lerp_(g_sq.sum(dim=0), 1.0 - self.beta2)
                v_est = (r[:, None] * c[None, :]) / r.sum().clamp_min(1e-30)
            else:
                v_buf = self.second_moment[key]
                v_buf.lerp_(g_sq, 1.0 - self.beta2)
                v_est = v_buf

            m_tilde = m_src / (v_est.sqrt() + self.muon2_eps)
            direction = _ns_orthogonalize(m_tilde, settings)
            scale = _shape_scale((param.shape[0], param.shape[1]))
            deltas[key] = -lr * (scale * direction + wd * param.float())

        self.apply_updates(deltas)
        self.last_metrics = {"baseline_steps": self.steps}

    def runtime_state(self):
        state = self.base_runtime_state()
        if self.factorized:
            state["row_v"] = self.row_v
            state["col_v"] = self.col_v
        else:
            state["second_moment"] = self.second_moment
        return state

    def restore_runtime_state(self, state):
        self.restore_base_runtime_state(state)
        if self.factorized:
            for key, val in state["row_v"].items():
                self.row_v[key].copy_(val)
            for key, val in state["col_v"].items():
                self.col_v[key].copy_(val)
        else:
            for key, val in state["second_moment"].items():
                self.second_moment[key].copy_(val)


class MuonEqBaseline(MatrixBaseline):
    """MuonEq: Balancing Before Orthogonalization with Lightweight Equilibration
    (arXiv:2603.28254; MaeChd/muon-eq).

    Stateless pre-NS equilibration of the Nesterov momentum matrix U:
      - ``muoneq_r``  (default for LLM hidden weights):
            D_r = (sum_j U_{ij}^2 + eps)^{scale_exp}   (scale_exp = -0.5)
            U_tilde = D_r[:, None] * U
      - ``muoneq_rc`` (two-sided equilibration):
            D_r = (sum_j U_{ij}^2 + eps)^{scale_exp}   (scale_exp = -0.25 or -0.5)
            D_c = (sum_i U_{ij}^2 + eps)^{scale_exp}
            U_tilde = D_r[:, None] * U * D_c[None, :]
    """

    def __init__(self, model, config, *, positions=None, rows_per_microbatch=0, capacity=0, rows_total=0, world, rank, device):
        super().__init__(model, config, world=world, rank=rank, device=device)
        bcfg = config.get("experiment", {}).get("baseline", {})
        arm = config["experiment"]["arm"]
        default_mode = "rowcol" if arm == "muoneq_rc" else "row"
        self.normalize_mode = str(bcfg.get("normalize_mode", default_mode))
        self.scale_exponent = float(bcfg.get("rowcol_scale_exponent", -0.5))
        self.rowcol_eps = float(bcfg.get("rowcol_eps", 1e-8))
        self.rowcol_clip = bcfg.get("rowcol_clip", None)

    def _equilibrate(self, mat: torch.Tensor) -> torch.Tensor:
        exp = self.scale_exponent
        eps = self.rowcol_eps
        clip = self.rowcol_clip

        def _pow_scale(s: torch.Tensor) -> torch.Tensor:
            s = s + eps
            if exp == -0.5:
                s = s.rsqrt()
            elif exp == -0.25:
                s = s.rsqrt().sqrt()
            else:
                s = s.pow(exp)
            if clip is not None:
                c = float(clip)
                s = s.clamp(min=1.0 / c, max=c)
            return s

        sq = mat.square()
        if self.normalize_mode == "row":
            r = _pow_scale(sq.sum(dim=1))
            return mat * r[:, None]
        if self.normalize_mode == "col":
            c = _pow_scale(sq.sum(dim=0))
            return mat * c[None, :]
        if self.normalize_mode == "rowcol":
            r = _pow_scale(sq.sum(dim=1))
            c = _pow_scale(sq.sum(dim=0))
            return (mat * r[:, None]) * c[None, :]
        raise ValueError(f"Unsupported MuonEq normalize_mode: {self.normalize_mode!r}")

    @torch.no_grad()
    def step(self):
        settings = self.config["optimizer"]
        group = self.param_groups[0]
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        deltas = {}
        for key, param in self.owned.items():
            source = self.momentum_source(key, param.grad)
            equilibrated = self._equilibrate(source)
            direction = _ns_orthogonalize(equilibrated, settings)
            scale = _shape_scale((param.shape[0], param.shape[1]))
            deltas[key] = -lr * (scale * direction + wd * param.float())
        self.apply_updates(deltas)
        self.last_metrics = {"baseline_steps": self.steps}


class NewtonMuonBaseline(MatrixBaseline):
    """The Newton-Muon Optimizer (arXiv:2604.01472; zhehangdu/Newton-Muon).

    Right-preconditions gradients with the inverse EWMA input-activation
    second moment (ZZ^T + ridge * I)^{-1} *before* momentum and Newton-Schulz:
        G_pre = G @ (ZZ^T + ridge * I)^{-1}
        M_t   = beta * M_{t-1} + G_pre    (+ Nesterov)
        O_t   = NS(M_t, K)

    Reuses the harness's sampled input-row buffers (`block.input_rows_attn`,
    `input_rows_o`, `input_rows_mlp`, `input_rows_down`). For the 4d-wide
    `down` projection, matches the official repo's 4 x (d x d) block-diagonal
    covariance (`down_block_diag=True`, default) or optional dense 4d x 4d.
    """

    def __init__(self, model, config, *, positions=None, rows_per_microbatch=0, capacity=0, rows_total=0, world, rank, device):
        super().__init__(model, config, world=world, rank=rank, device=device)
        bcfg = config.get("experiment", {}).get("baseline", {})
        self.blocks = list(model.body.blocks)
        self.hidden = int(config["model"]["hidden"])
        self.mlp = int(config["model"]["mlp"])
        self.rows_total = int(rows_total)
        self.refresh_every = int(bcfg.get("refresh_every", 32))
        self.precond_ewma = float(bcfg.get("precond_ewma", 0.95))
        self.ridge_mult = float(bcfg.get("precond_ridge_mult", 0.2))
        self.precond_eps = float(bcfg.get("precond_eps", 1e-8))
        self.init_diag = float(bcfg.get("precond_init_diag", 0.001))
        self.down_block_diag = bool(bcfg.get("down_block_diag", True))
        if self.down_block_diag and self.mlp % self.hidden != 0:
            raise ValueError(f"down_block_diag requires mlp ({self.mlp}) divisible by hidden ({self.hidden})")
        self.down_blocks = self.mlp // self.hidden if self.down_block_diag else 1

        self.owned_layers = sorted({layer for layer, _ in self.owned})
        self.cov: dict[tuple[int, str], torch.Tensor] = {}
        self.inv: dict[tuple[int, str], torch.Tensor] = {}
        for layer in self.owned_layers:
            for ikind in ("attn", "o", "mlp"):
                cov = torch.zeros((self.hidden, self.hidden), device=device, dtype=torch.float32)
                cov.diagonal().fill_(self.init_diag)
                self.cov[(layer, ikind)] = cov
                self.inv[(layer, ikind)] = torch.eye(self.hidden, device=device, dtype=torch.float32)
            if self.down_block_diag:
                cov = torch.zeros((self.down_blocks, self.hidden, self.hidden), device=device, dtype=torch.float32)
                cov.diagonal(dim1=-2, dim2=-1).fill_(self.init_diag)
                self.cov[(layer, "down")] = cov
                self.inv[(layer, "down")] = (
                    torch.eye(self.hidden, device=device, dtype=torch.float32)
                    .unsqueeze(0)
                    .repeat(self.down_blocks, 1, 1)
                )
            else:
                cov = torch.zeros((self.mlp, self.mlp), device=device, dtype=torch.float32)
                cov.diagonal().fill_(self.init_diag)
                self.cov[(layer, "down")] = cov
                self.inv[(layer, "down")] = torch.eye(self.mlp, device=device, dtype=torch.float32)
        self.fallbacks = 0
        self.gpu_ms = 0.0

    def _should_refresh(self, step_idx: int) -> bool:
        return step_idx == 0 or ((step_idx + 1) % self.refresh_every == 0)

    def _batch_cov(self, x: torch.Tensor, *, block_diag: bool) -> torch.Tensor:
        # x has shape [rows_local, width] in bf16/fp32
        x_f = x.float()
        if not block_diag:
            local = x_f.mT @ x_f
        else:
            n = x_f.shape[0]
            xb = x_f.view(n, self.down_blocks, self.hidden).permute(1, 0, 2)  # [B, N, d]
            local = torch.bmm(xb.transpose(1, 2), xb)  # [B, d, d]
        if self.world > 1:
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        denom = max(float(self.rows_total if self.rows_total > 0 else x_f.shape[0] * self.world), 1.0)
        return local / denom

    def _invert_spd(self, cov: torch.Tensor) -> torch.Tensor:
        d = cov.shape[-1]
        k = cov.clone()
        diag = k.diagonal(dim1=-2, dim2=-1)
        ridge = (diag.sum(dim=-1) / float(d)) * self.ridge_mult + self.precond_eps
        diag.add_(ridge.unsqueeze(-1))
        L, info = torch.linalg.cholesky_ex(k, upper=False, check_errors=False)
        inv = torch.cholesky_inverse(L, upper=False)
        if cov.ndim == 2:
            if int(info.item()) != 0:
                self.fallbacks += 1
                return torch.eye(d, device=cov.device, dtype=torch.float32)
            return inv
        bad = info != 0
        if bad.any():
            self.fallbacks += int(bad.sum().item())
            inv[bad] = torch.eye(d, device=cov.device, dtype=torch.float32)
        return inv

    @torch.no_grad()
    def _refresh_preconditioners(self):
        # All ranks participate in all_reduce for every layer's captured input rows
        # so owned layers receive the exact global-batch second moment.
        owned_set = set(self.owned_layers)
        for layer, block in enumerate(self.blocks):
            for ikind, attr in (
                ("attn", "input_rows_attn"),
                ("o", "input_rows_o"),
                ("mlp", "input_rows_mlp"),
                ("down", "input_rows_down"),
            ):
                rows = getattr(block, attr, None)
                if rows is None or rows.numel() == 0:
                    continue
                is_bd = ikind == "down" and self.down_block_diag
                batch_cov = self._batch_cov(rows, block_diag=is_bd)
                if layer in owned_set:
                    cov = self.cov[(layer, ikind)]
                    cov.lerp_(batch_cov, 1.0 - self.precond_ewma)
                    self.inv[(layer, ikind)].copy_(self._invert_spd(cov))

    def _precondition_grad(self, layer: int, kind: str, grad: torch.Tensor) -> torch.Tensor:
        ikind = INPUT_KIND[kind]
        inv = self.inv[(layer, ikind)]
        g = grad.float()
        if ikind == "down" and self.down_block_diag:
            out_d = g.shape[0]
            gb = g.view(out_d, self.down_blocks, self.hidden).permute(1, 0, 2)  # [B, out_d, d]
            pre = torch.bmm(gb, inv)  # [B, out_d, d]
            return pre.permute(1, 0, 2).reshape(out_d, self.mlp)
        return g @ inv

    @torch.no_grad()
    def step(self):
        if self._should_refresh(self.steps):
            self._refresh_preconditioners()

        settings = self.config["optimizer"]
        group = self.param_groups[0]
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        deltas = {}
        for key, param in self.owned.items():
            layer, kind = key
            g_pre = self._precondition_grad(layer, kind, param.grad)
            source = self.momentum_source(key, g_pre)
            direction = _ns_orthogonalize(source, settings)
            scale = _shape_scale((param.shape[0], param.shape[1]))
            deltas[key] = -lr * (scale * direction + wd * param.float())

        self.apply_updates(deltas)
        self.last_metrics = {
            "baseline_steps": self.steps,
            "newton_muon_fallbacks": self.fallbacks,
        }

    def runtime_state(self):
        state = self.base_runtime_state()
        state.update({"cov": self.cov, "inv": self.inv, "fallbacks": self.fallbacks})
        return state

    def restore_runtime_state(self, state):
        self.restore_base_runtime_state(state)
        for key, val in state["cov"].items():
            self.cov[key].copy_(val)
        for key, val in state["inv"].items():
            self.inv[key].copy_(val)
        self.fallbacks = int(state.get("fallbacks", 0))


BASELINE_ARMS: dict[str, type[MatrixBaseline]] = {
    "muon_control": MuonControl,
    "normuon": NorMuonBaseline,
    "muown": MuownBaseline,
    "muon2": Muon2Baseline,
    "muon2_f": Muon2Baseline,
    "muoneq_r": MuonEqBaseline,
    "muoneq_rc": MuonEqBaseline,
    "newton_muon": NewtonMuonBaseline,
}


def build_baseline(model, config, **runtime) -> MatrixBaseline:
    arm = config["experiment"]["arm"]
    return BASELINE_ARMS[arm](model, config, **runtime)
