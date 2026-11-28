# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep Toner's fast input updates; specialize the down group or every block matrix, including AdamH."""

from types import SimpleNamespace

import torch
from reproduce.marin.optimizer import build_optimizers, fast_arguments
from toner.fast import muon as MU
from toner.fast.engine import FastTonerCore, gather_uneven

from .baselines import BASELINE_ARMS, build_baseline
from .block_toner import BLOCK_ARMS, BlockToner, hyperball_delta
from .related_base import MatrixBaseline
from .toner_state import engine_state, restore_engine


class AdamH(torch.optim.Optimizer):
    """PyTorch implementation of Marin/Levanter's AdamH (bias-corrected Adam + Frobenius Hyperball).

    Matches ``levanter.optim.adamh.scale_by_adamh`` and ``experiments.grug.fast_track.adamh``:
    updates ``mu = b1 * mu + (1 - b1) * g``, ``nu = b2 * nu + (1 - b2) * g^2``, computes
    ``u = (mu / (1 - b1^t)) / (sqrt(nu / (1 - b2^t)) + eps)``, and applies ``hyperball_delta(p, u, lr)``.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        betas: tuple[float, float] = (0.8, 0.98),
        eps: float = 1e-15,
    ):
        defaults = {"lr": lr, "betas": tuple(betas), "eps": float(eps)}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                work_dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                g = p.grad.to(work_dtype)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=work_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=work_dtype)
                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                bc1 = 1.0 - beta1**t
                bc2 = 1.0 - beta2**t
                u = (exp_avg / bc1) / ((exp_avg_sq / bc2).sqrt().add_(eps))
                p.add_(hyperball_delta(p, u, lr))
        return loss


def accurate_right(matrix, root):
    return matrix @ root


def accurate_left(root, matrix):
    return root @ matrix


def down_update(
    weight,
    gradient,
    momentum,
    *,
    arm,
    left_inner,
    left_outer,
    right_inner,
    right_outer,
    args,
    lr,
    weight_decay,
    right_product,
    left_product,
):
    """Return updated momentum and the actual finite-NS down parameter change."""
    momentum = args.muon_momentum * momentum + gradient
    source = gradient + args.muon_momentum * momentum if args.muon_nesterov else momentum
    if arm == "down_hybrid":
        source = left_product(left_inner, source)
    else:
        source = right_product(source, right_inner)
        if arm == "inner_outer_down":
            source = left_product(left_inner, source)
    direction = torch.stack(
        [MU.newton_schulz(matrix, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode) for matrix in source]
    )
    if arm == "down_hybrid":
        decay = left_product(left_outer, left_product(left_outer, weight))
    else:
        dimension = right_outer.shape[-1]
        right_outer = right_outer * (dimension**0.5 / right_outer.norm(dim=(-2, -1), keepdim=True))
        direction = right_product(direction, right_outer)
        decay = right_product(right_product(weight, right_outer), right_outer)
    direction = left_product(left_outer, direction)
    scale = max(1.0, weight.shape[-2] / weight.shape[-1]) ** 0.5
    return momentum, -lr * (scale * direction + weight_decay * decay)


class DownToner(FastTonerCore):
    """A narrow extension retaining upstream roots and non-down updates."""

    def __init__(self, model, config, output, *, rows_total, world, rank, device):
        self.arm, self.output = config["experiment"]["arm"], output
        hidden, mlp = config["model"]["hidden"], config["model"]["mlp"]
        blocks = list(model.body.blocks)
        if self.arm == "down_hybrid":
            placeholder = torch.empty(0, hidden, dtype=torch.bfloat16, device=device)
            blocks = [
                SimpleNamespace(
                    input_rows_attn=b.input_rows_attn,
                    input_rows_o=b.input_rows_o,
                    input_rows_mlp=b.input_rows_mlp,
                    input_rows_down=placeholder,
                )
                for b in blocks
            ]
            mlp = hidden
        super().__init__(
            blocks,
            dict(model.named_parameters()),
            "muont",
            hid=hidden,
            mlp=mlp,
            rows_total=rows_total,
            args=fast_arguments(config),
            world=world,
            rank=rank,
            device=device,
        )
        if self.arm == "down_hybrid":
            del self.groups["f"]
            self.flags_host = torch.zeros(sum(g.per for g in self.groups.values()), dtype=torch.uint8, pin_memory=True)
        down_shape = (hidden, config["model"]["mlp"])
        self.down_group = next(group for group in self.shape_groups if group[0] == down_shape)
        self.shape_groups = [group for group in self.shape_groups if group[0] != down_shape]
        self.down_momentum = torch.zeros((len(self.down_group[2][rank]), *down_shape), device=device)
        self.finished_roots = None

    def _finish_roots(self):
        self.finished_roots = super()._finish_roots()
        return self.finished_roots

    @torch.no_grad()
    def step(self):
        super().step()
        _, _, per_rank = self.down_group
        mine = per_rank[self.rank]
        gradient = torch.stack([p.grad for p in mine]).float()
        group = self.param_groups[0]
        right_inner, right_outer = (None, None) if self.arm == "down_hybrid" else self.finished_roots["f"]
        self.down_momentum, delta = down_update(
            torch.stack(mine),
            gradient,
            self.down_momentum,
            arm=self.arm,
            left_inner=self.output.inner,
            left_outer=self.output.outer,
            right_inner=right_inner,
            right_outer=right_outer,
            args=self.args,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            right_product=accurate_right,
            left_product=accurate_left,
        )
        if not torch.isfinite(delta).all().item():
            raise FloatingPointError("Nonfinite down update")
        gathered = gather_uneven(delta, [len(ps) for ps in per_rank], group=self.pg)[1] if self.world > 1 else [delta]
        for parameters, updates in zip(per_rank, gathered, strict=True):
            torch._foreach_add_(parameters, list(updates.unbind(0)))
        self.finished_roots = None


def build(model, config, output, *, positions, rows_per_microbatch, capacity, rows_total, world, rank, device):
    arm = config["experiment"]["arm"]
    if arm in BASELINE_ARMS:
        fast = build_baseline(
            model,
            config,
            positions=positions,
            rows_per_microbatch=rows_per_microbatch,
            capacity=capacity,
            rows_total=rows_total,
            world=world,
            rank=rank,
            device=device,
        )
        return _with_auxiliary(model, config, fast)
    if arm in BLOCK_ARMS:
        fast = BlockToner(
            model,
            config,
            positions,
            rows_per_microbatch=rows_per_microbatch,
            capacity=capacity,
            rows_total=rows_total,
            world=world,
            rank=rank,
            device=device,
        )
        return _with_auxiliary(model, config, fast)
    if output is None:
        return build_optimizers(model, config, rows_total=rows_total, world=world, rank=rank, device=device)
    fast = DownToner(model, config, output, rows_total=rows_total, world=world, rank=rank, device=device)
    return _with_auxiliary(model, config, fast)


def _with_auxiliary(model, config, fast):
    """Auxiliary routing: default pinned AdamW, or canonical Marin AdamH on lm_head + Adam on emb/norms."""
    matrix_ids = {id(p) for p in fast.params}
    rest = [p for p in model.parameters() if id(p) not in matrix_ids]
    settings = config["optimizer"]
    head_mode = config.get("experiment", {}).get("head_mode", "adamw")
    use_fused = torch.device(model.head.weight.device).type == "cuda"

    if head_mode == "adamw":
        auxiliary = torch.optim.AdamW(
            [
                {"params": [model.head.weight], "weight_decay": settings["head_weight_decay"]},
                {"params": [p for p in rest if p is not model.head.weight], "weight_decay": 0.0},
            ],
            lr=settings["auxiliary_lr"],
            betas=tuple(settings["betas"]),
            eps=settings["eps"],
            fused=use_fused,
        )
        for optimizer, rate in ((fast, settings["lr"]), (auxiliary, settings["auxiliary_lr"])):
            for group in optimizer.param_groups:
                group["base_lr"] = rate
        return [fast, auxiliary], fast, rest

    if head_mode == "adamh":
        head_lr = float(settings.get("head_lr", settings["lr"]))
        head_opt = AdamH(
            [model.head.weight],
            lr=head_lr,
            betas=tuple(settings["betas"]),
            eps=settings["eps"],
        )
        auxiliary = torch.optim.AdamW(
            [{"params": [p for p in rest if p is not model.head.weight], "weight_decay": 0.0}],
            lr=settings["auxiliary_lr"],
            betas=tuple(settings["betas"]),
            eps=settings["eps"],
            fused=use_fused,
        )
        for optimizer, rate in (
            (fast, settings["lr"]),
            (head_opt, head_lr),
            (auxiliary, settings["auxiliary_lr"]),
        ):
            for group in optimizer.param_groups:
                group["base_lr"] = rate
        return [fast, head_opt, auxiliary], fast, rest

    raise ValueError(f"Unsupported head_mode: {head_mode!r}")


def toner_engines(fast):
    """Every Toner fast engine owned by a matrix optimizer."""
    if fast is None or isinstance(fast, MatrixBaseline):
        return []
    if isinstance(fast, BlockToner):
        return fast.engines
    return [fast]


def fast_runtime_state(fast):
    """Save numerical root state, including cached bases and pending refreshes."""
    if isinstance(fast, (BlockToner, MatrixBaseline)):
        return fast.runtime_state()
    state = engine_state(fast)
    if isinstance(fast, DownToner):
        state["down_momentum"] = fast.down_momentum
    return state


def restore_fast(fast, state):
    if isinstance(fast, (BlockToner, MatrixBaseline)):
        fast.restore_runtime_state(state)
        return
    restore_engine(fast, state)
    if isinstance(fast, DownToner):
        fast.down_momentum.copy_(state["down_momentum"])
