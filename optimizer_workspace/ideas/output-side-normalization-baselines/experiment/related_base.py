# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common matrix ownership and checkpoint state for related-method controls."""

import torch
import torch.distributed as dist
from toner.fast import muon as MU
from toner.fast.engine import layer_counts

MATRIX_KINDS = ("q", "k", "v", "o", "gate", "up", "down")


class MatrixBaseline:
    """Route the same block matrices as MuonT, leaving auxiliary routing unchanged."""

    def __init__(self, model, config, *, world, rank, device):
        self.config = config
        self.world, self.rank, self.device = world, rank, device
        blocks = list(model.body.blocks)
        counts = layer_counts(len(blocks), world)
        owner = {layer: node for node in range(world) for layer in range(sum(counts[:node]), sum(counts[: node + 1]))}
        self.matrices = {
            (layer, kind): getattr(block, kind).weight for layer, block in enumerate(blocks) for kind in MATRIX_KINDS
        }
        self.owned = {key: parameter for key, parameter in self.matrices.items() if owner[key[0]] == rank}
        self.params = list(self.matrices.values())
        settings = config["optimizer"]
        self.param_groups = [{"params": self.params, "lr": settings["lr"], "weight_decay": settings["weight_decay"]}]
        self.momentum = {key: torch.zeros_like(parameter, dtype=torch.float32) for key, parameter in self.owned.items()}
        self.steps = 0
        self.last_metrics = {}
        shapes = {}
        for key, parameter in self.matrices.items():
            shapes.setdefault(tuple(parameter.shape), []).append(key)
        self.shape_groups = [
            (shape, [[key for key in keys if owner[key[0]] == node] for node in range(world)])
            for shape, keys in shapes.items()
        ]

    def momentum_source(self, key, gradient, gradient_scale=1.0):
        settings = self.config["optimizer"]
        gradient = gradient.float() * gradient_scale
        buffer = self.momentum[key]
        buffer.mul_(settings["momentum"]).add_(gradient)
        return gradient + settings["momentum"] * buffer if settings["nesterov"] else buffer

    @torch.no_grad()
    def apply_updates(self, deltas):
        """Communicate owner-computed FP32 changes in the same fixed matrix order."""
        if deltas.keys() != self.owned.keys():
            raise ValueError("Baseline updates must cover exactly the rank-owned matrices")
        finite = torch.tensor(
            all(torch.isfinite(delta).all().item() for delta in deltas.values()),
            device=self.device,
            dtype=torch.int32,
        )
        if self.world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Nonfinite related-baseline update")
        for shape, per_rank in self.shape_groups:
            keys = per_rank[self.rank]
            local = (
                torch.stack([deltas[key].float() for key in keys])
                if keys
                else torch.empty((0, *shape), device=self.device, dtype=torch.float32)
            )
            if self.world > 1:
                gathered = [
                    torch.empty((len(keys), *shape), device=self.device, dtype=torch.float32) for keys in per_rank
                ]
                dist.all_gather(gathered, local)
            else:
                gathered = [local]
            for keys, values in zip(per_rank, gathered, strict=True):
                if keys:
                    torch._foreach_add_([self.matrices[key] for key in keys], list(values.unbind(0)))
        self.steps += 1

    def base_runtime_state(self):
        return {"steps": self.steps, "momentum": self.momentum}

    def restore_base_runtime_state(self, state):
        if state["momentum"].keys() != self.momentum.keys():
            raise ValueError("Resume requires unchanged baseline matrix ownership")
        for key, value in state["momentum"].items():
            self.momentum[key].copy_(value)
        self.steps = state["steps"]

    def begin(self):
        pass

    def begin_step(self, step):
        if step != self.steps:
            raise ValueError("Training and baseline update counters differ")

    def select_microbatch(self, index):
        pass

    def finish_backward(self):
        pass

    def runtime_state(self):
        return self.base_runtime_state()

    def restore_runtime_state(self, state):
        self.restore_base_runtime_state(state)


class MuonControl(MatrixBaseline):
    """Same-runner isotropic-decay control for related optimizers' system overhead."""

    def __init__(self, model, config, *, positions, rows_per_microbatch, capacity, rows_total, world, rank, device):
        super().__init__(model, config, world=world, rank=rank, device=device)

    @torch.no_grad()
    def step(self):
        settings = self.config["optimizer"]
        group = self.param_groups[0]
        deltas = {}
        for key, parameter in self.owned.items():
            source = self.momentum_source(key, parameter.grad)
            direction = MU.newton_schulz(source, settings["ns_steps"], settings["ns_eps"], settings["ns_mode"])
            scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
            deltas[key] = -group["lr"] * (scale * direction + group["weight_decay"] * parameter.float())
        self.apply_updates(deltas)
        self.last_metrics = {"related_baseline_steps": self.steps}
