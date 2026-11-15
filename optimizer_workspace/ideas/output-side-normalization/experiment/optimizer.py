# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep Toner's fast input updates; specialize only the down-projection group."""

import concurrent.futures
from types import SimpleNamespace

import torch
from reproduce.marin.optimizer import build_optimizers, fast_arguments
from toner.fast import muon as MU
from toner.fast import roots as PR
from toner.fast.engine import FastTonerCore, gather_uneven


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
    # BF16 NS amplifies small GEMM differences; preserve the per-matrix reference.
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
            # The upstream constructor requires this group; use a tiny placeholder and
            # discard it before any statistics, roots, host work, or update can run.
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


def build(model, config, output, *, rows_total, world, rank, device):
    optimizers, fast, clipping = (
        build_optimizers(model, config, rows_total=rows_total, world=world, rank=rank, device=device)
        if output is None
        else (None, None, None)
    )
    if output is None:
        return optimizers, fast, clipping
    fast = DownToner(model, config, output, rows_total=rows_total, world=world, rank=rank, device=device)
    matrix_ids = {id(p) for p in fast.params}
    rest = [p for p in model.parameters() if id(p) not in matrix_ids]
    settings = config["optimizer"]
    auxiliary = torch.optim.AdamW(
        [
            {"params": [model.head.weight], "weight_decay": settings["head_weight_decay"]},
            {"params": [p for p in rest if p is not model.head.weight], "weight_decay": 0.0},
        ],
        lr=settings["auxiliary_lr"],
        betas=tuple(settings["betas"]),
        eps=settings["eps"],
        fused=True,
    )
    for optimizer, rate in ((fast, settings["lr"]), (auxiliary, settings["auxiliary_lr"])):
        for group in optimizer.param_groups:
            group["base_lr"] = rate
    return [fast, auxiliary], fast, rest


def fast_runtime_state(fast):
    """Save numerical root state, including cached bases and pending refreshes."""
    fast.side.synchronize()
    for group in fast.groups.values():
        for pending in group.pending:
            if pending is not None:
                pending[0].result()
    groups = {}
    for name, group in fast.groups.items():
        groups[name] = {
            "C": group.C,
            "tol": group.tol,
            "basis_step": group.basis_step,
            "warm": group.warm,
            "SB": None if group.basis is None else group.basis.SB,
            "S2": None if group.basis is None else group.basis.S2,
            "pending": [
                (
                    None
                    if pending is None
                    else {
                        "step": pending[1],
                        "basis": group.host_out[index].clone(),
                        "tol": group.host_tol[index].clone(),
                    }
                )
                for index, pending in enumerate(group.pending)
            ],
        }
    state = {
        "groups": groups,
        "momentum": fast.state,
        "weight": fast.weight,
        "steps": fast.steps,
        "t": fast.t,
        "fallbacks": fast.fallbacks,
        "extra": fast.extra,
    }
    if isinstance(fast, DownToner):
        state["down_momentum"] = fast.down_momentum
    return state


def restore_fast(fast, state):
    for name, saved in state["groups"].items():
        group = fast.groups[name]
        group.C.copy_(saved["C"])
        group.tol.copy_(saved["tol"])
        group.basis_step = saved["basis_step"]
        # CUDA graphs and process-local kernel compilation are not serializable.
        # Rewarm against the restored numerical state before capturing fresh graphs.
        group.warm = False
        if saved["S2"] is not None:
            group.basis = PR.Basis.of_splits(saved["S2"].to(fast.device))
            group.basis.SB = saved["SB"].to(fast.device)
        for index, pending in enumerate(saved["pending"]):
            if pending is not None:
                group.host_out[index].copy_(pending["basis"])
                group.host_tol[index].copy_(pending["tol"])
                future = concurrent.futures.Future()
                future.set_result(None)
                group.pending[index] = (future, pending["step"])
    fast.state = {
        shape: {name: tensor.to(fast.device) for name, tensor in values.items()}
        for shape, values in state["momentum"].items()
    }
    fast.weight, fast.steps, fast.t = state["weight"], state["steps"], state["t"]
    fast.fallbacks, fast.extra = state["fallbacks"], state["extra"]
    if isinstance(fast, DownToner):
        fast.down_momentum.copy_(state["down_momentum"])
