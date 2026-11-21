# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backward-only output moments and down-free input row capture."""

import torch
import torch.distributed as dist
import triton
from reproduce.marin import ops
from reproduce.marin.model import Body, Model, norm
from toner.fast.engine import gather_uneven, layer_counts
from torch import nn
from torch.nn import functional as F

from .factors import inverse_factors


@torch.library.custom_op("output_marin::cotangent_sum", mutates_args=("sums", "counts"))
def cotangent_sum(
    gradient: torch.Tensor,
    sums: torch.Tensor,
    counts: torch.Tensor,
    positions: torch.Tensor,
    scaling: torch.Tensor,
) -> None:
    rows = gradient.index_select(1, positions).reshape(-1, gradient.shape[-1]).float()
    rows = rows * scaling
    sums.add_(rows.mT @ rows)
    counts.add_(rows.shape[0])


@cotangent_sum.register_fake
def _(gradient, sums, counts, positions, scaling):
    return None


class ObserveOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, sums, counts, positions, scaling):
        # These buffers are intentionally mutated by different backward nodes.
        # Saving them with version checks would reject later layers' observations.
        ctx.sums, ctx.counts = sums, counts
        ctx.save_for_backward(positions, scaling)
        return output.view_as(output)

    @staticmethod
    def backward(ctx, gradient):
        positions, scaling = ctx.saved_tensors
        cotangent_sum(gradient, ctx.sums, ctx.counts, positions, scaling)
        return gradient, None, None, None, None


class OutputLinear(nn.Module):
    """The original linear map followed by a gradient-transparent observation."""

    def __init__(self, linear, sums, counts, positions, scaling):
        super().__init__()
        self.weight = linear.weight
        self.register_buffer("output_sum", sums, persistent=False)
        self.register_buffer("output_count", counts, persistent=False)
        self.register_buffer("output_positions", positions, persistent=False)
        self.register_buffer("output_scaling", scaling, persistent=False)

    def forward(self, x):
        output = F.linear(x, self.weight)
        return ObserveOutput.apply(
            output, self.output_sum, self.output_count, self.output_positions, self.output_scaling
        )


_TABLES = {}


@torch.library.custom_op("output_marin::capture_without_down", mutates_args=("bufs",))
def capture_without_down(
    xs: list[torch.Tensor],
    attn: list[torch.Tensor],
    xmid: list[torch.Tensor],
    w1: list[torch.Tensor],
    w2: list[torch.Tensor],
    bufs: list[torch.Tensor],
    at: torch.Tensor,
    idx: torch.Tensor,
    eps: float,
) -> None:
    # Reuse Toner's sampled RMSNorm and attention-output kernels, omitting SwiGLU rows.
    layers, positions = len(xs), idx.numel()
    x = xs[0]
    columns = (xs, w1, bufs[0::3], xmid, w2, bufs[2::3], attn, bufs[1::3])
    key = tuple(t.data_ptr() for column in columns for t in column)
    table = _TABLES.get(key)
    if table is None:
        if len(_TABLES) >= 64:
            _TABLES.clear()
        table = torch.tensor(key, dtype=torch.int64).pin_memory().to(at.device, non_blocking=True)
        _TABLES[key] = table
    grid = (layers, x.size(0) * positions)
    width = x.shape[-1]
    block = triton.next_power_of_2(width)
    ops._norm_rows[grid](table, layers, at, idx, positions, width, x.stride(0), x.stride(1), eps, BLOCK=block)
    ops._norm_rows[grid](
        table[3 * layers :],
        layers,
        at,
        idx,
        positions,
        width,
        x.stride(0),
        x.stride(1),
        eps,
        BLOCK=block,
    )
    ops._gather_rows[grid](
        table[6 * layers :],
        layers,
        at,
        idx,
        positions,
        width,
        x.stride(0),
        x.stride(1),
        BLOCK=512,
    )


@capture_without_down.register_fake
def _(xs, attn, xmid, w1, w2, bufs, at, idx, eps):
    return None


class DownFreeBody(Body):
    """Pinned block computation with three input buffers instead of four."""

    def forward(self, tokens, cu, meta):
        x = self.emb(tokens)
        if self.config["residual_dtype"] == "bf16":
            x = x.to(torch.bfloat16)
        captures = []
        for block in self.blocks:
            x = block(x, self.cos, self.sin, cu, meta, captures)
        xi, attn, xm, _, _ = (list(c) for c in zip(*captures, strict=True))
        capture_without_down(
            xi,
            attn,
            xm,
            [b.ln1.weight.detach() for b in self.blocks],
            [b.ln2.weight.detach() for b in self.blocks],
            [getattr(b, f"input_rows_{kind}") for b in self.blocks for kind in ("attn", "o", "mlp")],
            self.stat_at,
            self.blocks[0].stat_idx,
            self.config["norm_eps"],
        )
        return norm(self.norm, x, self.config["residual_dtype"])


class HybridModel(nn.Module):
    def __init__(self, config, **capture):
        super().__init__()
        self.body = DownFreeBody(config, **capture)
        self.head = nn.Linear(config["hidden"], config["vocab_size"], bias=False)
        for block in self.body.blocks:
            del block.input_rows_down


# Arms whose matrix updates use no input statistic at all.
NO_INPUT_CAPTURE = ("muon", "left_all", "muon_control", "mousse_control", "mousse_paper")


def captures_input(arm):
    return arm not in NO_INPUT_CAPTURE


def build_model(config, *, positions, row_capacity, rows_per_microbatch, device):
    arm = config["experiment"]["arm"]
    if not captures_input(arm):
        return Model(config["model"]).to(device)
    model_type = HybridModel if arm == "down_hybrid" else Model
    model = model_type(
        config["model"],
        positions=positions,
        row_capacity=row_capacity,
        rows_per_microbatch=rows_per_microbatch,
    ).to(device)
    return model


class OutputMoments:
    """Pool FP32 raw sums across microbatches/ranks before EMA and inversion."""

    def __init__(self, model, positions, settings, *, world, rank, device):
        self.settings, self.world, self.rank = settings, world, rank
        self.layers = len(model.body.blocks)
        self.counts_per_rank = layer_counts(self.layers, world)
        start = sum(self.counts_per_rank[:rank])
        self.owned_layers = list(range(start, start + self.counts_per_rank[rank]))
        width = model.body.blocks[0].down.weight.shape[0]
        # Independent storage prevents one layer's backward mutation from changing
        # another layer's saved-buffer version counter in AOTAutograd/DDP.
        self.layer_sums = [torch.zeros(width, width, device=device) for _ in range(self.layers)]
        self.layer_counts = [torch.zeros((), dtype=torch.int64, device=device) for _ in range(self.layers)]
        self.scaling = torch.ones((), device=device)
        self.positions = torch.tensor(positions, dtype=torch.int64, device=device)
        self.moment = torch.zeros(len(self.owned_layers), width, width, device=device)
        self.inner = torch.eye(width, device=device).expand_as(self.moment).clone()
        self.outer = self.inner.clone()
        self.steps, self.weight, self.root_step = 0, 0.0, -1
        self.pg = dist.new_group(backend=dist.get_backend()) if world > 1 else None
        for layer, block in enumerate(model.body.blocks):
            block.down = OutputLinear(
                block.down, self.layer_sums[layer], self.layer_counts[layer], self.positions, self.scaling
            )

    @property
    def sums(self):
        return torch.stack(self.layer_sums)

    @property
    def counts(self):
        return torch.stack(self.layer_counts)

    def begin(self, inverse_loss_scale):
        for value in self.layer_sums + self.layer_counts:
            value.zero_()
        self.scaling.fill_(inverse_loss_scale)

    @torch.no_grad()
    def update(self, expected_local_count):
        sums, counts = self.sums, self.counts
        if not torch.all(counts == expected_local_count).item():
            raise RuntimeError(f"Output capture count mismatch: {counts.tolist()}")
        if not torch.isfinite(sums).all().item():
            raise FloatingPointError("Nonfinite output moment sum")
        pooled = torch.empty_like(self.moment)
        if self.world > 1:
            dist.all_reduce(counts, group=self.pg)
            dist.reduce_scatter_tensor(pooled, sums, group=self.pg)
        else:
            pooled.copy_(sums)
        start = self.owned_layers[0]
        means = pooled / counts[start : start + len(self.owned_layers), None, None]
        beta = self.settings.momentum
        self.moment.mul_(beta).add_(means, alpha=1 - beta)
        self.weight = beta * self.weight + 1 - beta
        if self.steps % self.settings.refresh == 0:
            self.inner, self.outer = inverse_factors(self.moment, self.settings)
            self.root_step = self.steps
        self.steps += 1

    def state_dict(self):
        return {
            "moment": self.moment,
            "inner": self.inner,
            "outer": self.outer,
            "steps": self.steps,
            "weight": self.weight,
            "root_step": self.root_step,
            "owned_layers": self.owned_layers,
        }

    def load_state_dict(self, state):
        if state["owned_layers"] != self.owned_layers:
            raise ValueError("Resume requires the same output-factor ownership")
        for name in ("moment", "inner", "outer"):
            getattr(self, name).copy_(state[name])
        self.steps, self.weight, self.root_step = state["steps"], state["weight"], state["root_step"]

    @torch.no_grad()
    def snapshot(self):
        parts = (
            gather_uneven(self.moment / self.weight, self.counts_per_rank, group=self.pg)[1]
            if self.world > 1
            else [self.moment / self.weight]
        )
        return torch.cat(parts).cpu() if self.rank == 0 else None
