# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy schedule, zero-cuSOLVER async-host output root solvers, and per-head block-diagonal roots."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

import torch
import torch.distributed as dist
from toner.fast import roots as PR
from toner.fast.engine import layer_counts
from toner.fast.syrk import list_offsets, syrk_list_packed, unpack_ema_


@dataclass(frozen=True)
class LazySchedule:
    enabled: bool = False
    input_every: int = 8
    output_every: int = 8
    warmup_until: int = 16
    warmup_every: int = 2
    capture_microbatches: int = 1
    input_ema_decay: float = 0.75
    output_ema_decay: float = 0.75
    lag_input: bool = False
    # root_backend: "async_host" (FastTonerCore cached-basis on inputs + async CPU eigh on outputs; 0 GPU cuSOLVER stalls)
    #               or "gpu_eigh" (batched torch.linalg.eigh on GPU)
    root_backend: str = "async_host"
    input_inline_steps: int = 1
    input_refresh_every: int = 2

    @classmethod
    def from_config(cls, config: dict) -> LazySchedule:
        raw = config.get("experiment", {}).get("lazy")
        if not raw:
            return cls(enabled=False)
        default_backend = "gpu_eigh" if raw.get("exact_gpu_roots", False) and "root_backend" not in raw else "async_host"
        backend = str(raw.get("root_backend", default_backend))
        return cls(
            enabled=bool(raw.get("enabled", True)),
            input_every=int(raw.get("input_every", 8)),
            output_every=int(raw.get("output_every", 8)),
            warmup_until=int(raw.get("warmup_until", 16)),
            warmup_every=int(raw.get("warmup_every", 2)),
            capture_microbatches=int(raw.get("capture_microbatches", 1)),
            input_ema_decay=float(raw.get("input_ema_decay", 0.75)),
            output_ema_decay=float(raw.get("output_ema_decay", 0.75)),
            lag_input=bool(raw.get("lag_input", backend == "async_host_both")),
            root_backend=backend,
            input_inline_steps=int(raw.get("input_inline_steps", 1)),
            input_refresh_every=int(raw.get("input_refresh_every", 2)),
        )

    def refresh_input(self, step: int) -> bool:
        if not self.enabled:
            return True
        if step < self.warmup_until:
            return step % self.warmup_every == 0
        return (step - self.warmup_until) % self.input_every == 0

    def refresh_output(self, step: int) -> bool:
        if not self.enabled:
            return True
        if step < self.warmup_until:
            return step % self.warmup_every == 0
        return (step - self.warmup_until) % self.output_every == 0

    def capture_microbatch(self, mb_index: int, accumulation: int) -> bool:
        if not self.enabled:
            return True
        return mb_index >= max(0, accumulation - self.capture_microbatches)


def _reconstruct_dense_from_eig(
    V: torch.Tensor,
    lam: torch.Tensor,
    pos: torch.Tensor,
    *,
    damping: float,
    inner_power: float,
    outer_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = V.size(-1)
    eye = torch.eye(d, dtype=V.dtype, device=V.device).expand_as(V)
    x = lam.clamp_min(0.0)
    f_in = (x + damping).pow(-inner_power)
    f_out = (x + damping).pow(-outer_power)
    if V.is_cuda and V.dim() == 3:
        Vt_lo = PR.rows_lo(V.mT.contiguous())
        if PR._sym_ok(d):
            r_in = PR.mm16_sym(PR.cols_hi(V * f_in[..., None, :]), Vt_lo, False)
            r_out = PR.mm16_sym(PR.cols_hi(V * f_out[..., None, :]), Vt_lo, False)
        else:
            r_in = PR.mm16(PR.cols_hi(V * f_in[..., None, :]), Vt_lo, False)
            r_in = 0.5 * (r_in + r_in.mT)
            r_out = PR.mm16(PR.cols_hi(V * f_out[..., None, :]), Vt_lo, False)
            r_out = 0.5 * (r_out + r_out.mT)
    else:
        r_in = (V * f_in[..., None, :]) @ V.mT
        r_in = 0.5 * (r_in + r_in.mT)
        r_out = (V * f_out[..., None, :]) @ V.mT
        r_out = 0.5 * (r_out + r_out.mT)
    return torch.where(pos, r_in, eye), torch.where(pos, r_out, eye)


def exact_dense_powers(
    C: torch.Tensor,
    *,
    damping: float,
    inner_power: float,
    outer_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched exact inverse powers (A_d(C)^(-inner), A_d(C)^(-outer)) on GPU or CPU."""
    if C.numel() == 0:
        return C.clone(), C.clone()
    sym = 0.5 * (C + C.mT)
    d = sym.size(-1)
    trace = sym.diagonal(dim1=-2, dim2=-1).sum(-1)
    pos = (trace > 0)[..., None, None]
    eye = torch.eye(d, dtype=sym.dtype, device=sym.device).expand_as(sym)
    normed = torch.where(pos, sym * (d / trace.clamp_min(1e-30))[..., None, None], eye)
    lam, V = torch.linalg.eigh(normed)
    return _reconstruct_dense_from_eig(
        V, lam, pos, damping=damping, inner_power=inner_power, outer_power=outer_power
    )


def exact_head_block_powers(
    C: torch.Tensor,
    *,
    damping: float,
    inner_power: float,
    outer_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched exact inverse powers on head-block moments C of shape [N, H, hd, hd]."""
    if C.numel() == 0:
        return C.clone(), C.clone()
    N, H, hd, _ = C.shape
    full_d = H * hd
    sym = 0.5 * (C + C.mT)
    trace = sym.diagonal(dim1=-2, dim2=-1).sum((-2, -1))
    pos = (trace > 0)[:, None, None, None]
    eye = torch.eye(hd, dtype=sym.dtype, device=sym.device).view(1, 1, hd, hd).expand(N, H, hd, hd)
    normed = torch.where(pos, sym * (full_d / trace.clamp_min(1e-30))[:, None, None, None], eye)
    flat = normed.reshape(N * H, hd, hd)
    lam, V = torch.linalg.eigh(flat)
    x = lam.clamp_min(0.0)
    f_in = (x + damping).pow(-inner_power)
    f_out = (x + damping).pow(-outer_power)
    Vt = V.mT
    r_in = (V * f_in[..., None, :]) @ Vt
    r_in = (0.5 * (r_in + r_in.mT)).view(N, H, hd, hd)
    r_out = (V * f_out[..., None, :]) @ Vt
    r_out = (0.5 * (r_out + r_out.mT)).view(N, H, hd, hd)
    return torch.where(pos, r_in, eye), torch.where(pos, r_out, eye)


class _DenseGroup:
    def __init__(
        self,
        keys: list[tuple[int, str]],
        width: int,
        bufs: list[torch.Tensor],
        counts: list[int],
        rank: int,
        device,
        *,
        pin_host: bool = False,
    ):
        self.keys = keys
        self.n = width
        self.bufs = bufs
        self.counts = counts
        self.per = counts[rank]
        lo = sum(counts[:rank])
        self.own = keys[lo : lo + self.per]
        self.offs = None
        packed_dim = width * (width + 1) // 2
        self.partial = torch.empty(len(keys), packed_dim, dtype=torch.float32, device=device)
        self.summed = torch.empty(self.per, packed_dim, dtype=torch.float32, device=device)
        self.C = torch.zeros(self.per, width, width, dtype=torch.float32, device=device)
        if pin_host and self.per > 0:
            self.host_C = torch.empty(self.per, width, width, dtype=torch.float32, pin_memory=True)
            self.host_V = torch.empty(self.per, width, width, dtype=torch.float32, pin_memory=True)
            self.host_lam = torch.empty(self.per, width, dtype=torch.float32, pin_memory=True)
            self.host_pos = torch.empty(self.per, dtype=torch.bool, pin_memory=True)
            self.gpu_V = torch.empty(self.per, width, width, dtype=torch.float32, device=device)
            self.gpu_lam = torch.empty(self.per, width, dtype=torch.float32, device=device)
            self.gpu_pos = torch.empty(self.per, dtype=torch.bool, device=device)
        else:
            self.host_C = self.host_V = self.host_lam = self.host_pos = None
            self.gpu_V = self.gpu_lam = self.gpu_pos = None


def _host_eigh_slice(host_c: torch.Tensor, host_v: torch.Tensor, host_lam: torch.Tensor, host_pos: torch.Tensor, j: int, copied_ev):
    d = host_c.size(-1)
    torch.set_num_threads(4 if d >= 2048 else 2)
    copied_ev.synchronize()
    mat = host_c[j]
    sym = 0.5 * (mat + mat.mT)
    tr = float(sym.diagonal().sum().item())
    if tr > 0.0:
        normed = sym * (d / max(tr, 1e-30))
        lam, V = torch.linalg.eigh(normed)
        host_v[j].copy_(V)
        host_lam[j].copy_(lam)
        host_pos[j] = True
    else:
        host_v[j].zero_()
        host_v[j].diagonal().fill_(1.0)
        host_lam[j].zero_()
        host_pos[j] = False


class BatchExactDenseEngine:
    """Sharded dense second-moment EMA with either async host parallel eigh (0 GPU cuSOLVER) or GPU eigh."""

    def __init__(
        self,
        group_specs: list[tuple[str, list[tuple[int, str]], int, list[torch.Tensor], list[int]]],
        *,
        decay: float,
        damping: float,
        inner_power: float,
        outer_power: float,
        rows_total: int,
        world: int,
        rank: int,
        device,
        stream_priority: int = 0,
        async_host: bool = False,
    ):
        self.decay = float(decay)
        self.damping = float(damping)
        self.inner_power = float(inner_power)
        self.outer_power = float(outer_power)
        self.rows_total = int(rows_total)
        self.world = world
        self.rank = rank
        self.device = device
        self.is_cuda = torch.device(device).type == "cuda"
        self.async_host = bool(async_host) and self.is_cuda
        self.main_stream = torch.cuda.current_stream(device) if self.is_cuda else None
        self.side = torch.cuda.Stream(device=device, priority=stream_priority) if self.is_cuda else None
        self.ev_rows_consumed = torch.cuda.Event() if self.is_cuda else None
        self.ev_done = torch.cuda.Event() if self.is_cuda else None
        self.pg_stats = dist.new_group(backend=dist.get_backend()) if world > 1 else None
        self.groups = {
            name: _DenseGroup(keys, width, bufs, counts, rank, device, pin_host=self.async_host)
            for name, keys, width, bufs, counts in group_specs
            if keys
        }
        total_per = sum(g.per for g in self.groups.values())
        self.host_pool = (
            concurrent.futures.ThreadPoolExecutor(max_workers=min(4, max(1, total_per)))
            if self.async_host
            else None
        )
        self._pending_futs: list[concurrent.futures.Future] = []
        self.weight = 0.0
        self.steps = 0
        self.t = 0
        self.fallbacks = 0
        self.extra = 0
        self.begun = False
        self._latest_roots: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _update_ema_on_current_stream(self) -> None:
        b = self.decay
        for g in self.groups.values():
            if self.is_cuda and g.bufs[0].dtype == torch.bfloat16:
                if g.offs is None:
                    g.offs = list_offsets(g.bufs)
                syrk_list_packed(g.bufs, g.offs, g.partial, acc=False)
            else:
                idx_i, idx_j = torch.triu_indices(g.n, g.n, device=self.device)
                grams = torch.stack([buf.float().mT @ buf.float() for buf in g.bufs])
                g.partial.copy_(grams[:, idx_i, idx_j])
        if self.is_cuda:
            self.ev_rows_consumed.record(self.side)
        works = []
        for g in self.groups.values():
            if self.world == 1:
                g.summed.copy_(g.partial)
            elif len(set(g.counts)) == 1:
                works.append(dist.reduce_scatter_tensor(g.summed, g.partial, group=self.pg_stats, async_op=True))
            else:
                works.append(dist.reduce_scatter(g.summed, list(g.partial.split(g.counts)), group=self.pg_stats, async_op=True))
        for w in works:
            w.wait()
        alpha = (1.0 - b) / self.rows_total
        for g in self.groups.values():
            if g.per:
                if self.is_cuda:
                    unpack_ema_(g.summed, g.C, b, alpha)
                else:
                    idx_i, idx_j = torch.triu_indices(g.n, g.n, device=self.device)
                    mat = torch.zeros(g.per, g.n, g.n, dtype=torch.float32, device=self.device)
                    mat[:, idx_i, idx_j] = g.summed
                    mat[:, idx_j, idx_i] = g.summed
                    g.C.mul_(b).add_(mat, alpha=alpha)
        self.weight = b * self.weight + (1.0 - b)

    @torch.no_grad()
    def begin(self) -> None:
        if not self.is_cuda:
            self._update_ema_on_current_stream()
            self._latest_roots = {
                name: exact_dense_powers(
                    g.C, damping=self.damping, inner_power=self.inner_power, outer_power=self.outer_power
                )
                for name, g in self.groups.items()
            }
            self.begun = True
            return

        ev_src = torch.cuda.Event()
        ev_src.record(self.main_stream)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev_src)
            self._update_ema_on_current_stream()
            if self.async_host:
                self._pending_futs = []
                for g in sorted(self.groups.values(), key=lambda grp: grp.n, reverse=True):
                    if not g.per:
                        continue
                    g.host_C.copy_(g.C, non_blocking=True)
                    ev_copied = torch.cuda.Event(blocking=True)
                    ev_copied.record(self.side)
                    for j in range(g.per):
                        self._pending_futs.append(
                            self.host_pool.submit(_host_eigh_slice, g.host_C, g.host_V, g.host_lam, g.host_pos, j, ev_copied)
                        )
            else:
                self._latest_roots = {
                    name: exact_dense_powers(
                        g.C, damping=self.damping, inner_power=self.inner_power, outer_power=self.outer_power
                    )
                    for name, g in self.groups.items()
                }
            self.ev_done.record(self.side)
        if not self.async_host:
            for ri, ro in self._latest_roots.values():
                ri.record_stream(self.main_stream)
                ro.record_stream(self.main_stream)
        self.begun = True

    def hosts_ready(self) -> bool:
        if not self.async_host:
            return self.ev_done.query() if self.is_cuda else True
        return all(f.done() for f in self._pending_futs)

    def _finalize_async_host_roots(self, stream) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        for f in self._pending_futs:
            f.result()
        self._pending_futs = []
        roots = {}
        with torch.cuda.stream(stream):
            for name, g in self.groups.items():
                if not g.per:
                    empty = torch.empty(0, g.n, g.n, dtype=torch.float32, device=self.device)
                    roots[name] = (empty, empty)
                    continue
                g.gpu_V.copy_(g.host_V, non_blocking=True)
                g.gpu_lam.copy_(g.host_lam, non_blocking=True)
                g.gpu_pos.copy_(g.host_pos, non_blocking=True)
                ri, ro = _reconstruct_dense_from_eig(
                    g.gpu_V,
                    g.gpu_lam,
                    g.gpu_pos[:, None, None],
                    damping=self.damping,
                    inner_power=self.inner_power,
                    outer_power=self.outer_power,
                )
                ri.record_stream(self.main_stream)
                ro.record_stream(self.main_stream)
                roots[name] = (ri, ro)
        self._latest_roots = roots
        return roots

    def finish_roots_on_main(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if not self.begun:
            self.begin()
        self.begun = False
        if self.async_host:
            roots = self._finalize_async_host_roots(self.side)
            self.main_stream.wait_stream(self.side)
            return roots
        if self.is_cuda:
            self.main_stream.wait_event(self.ev_done)
        return self._latest_roots

    def finish_roots_on_side(self, target_stream) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if not self.begun:
            self.begin()
        self.begun = False
        if self.async_host:
            return self._finalize_async_host_roots(target_stream)
        if self.is_cuda and target_stream is not self.side:
            target_stream.wait_event(self.ev_done)
        return self._latest_roots

    def shutdown(self) -> None:
        if self.host_pool is not None:
            self.host_pool.shutdown(wait=True)

    def state_dict(self) -> dict:
        if self.is_cuda:
            self.side.synchronize()
        for f in self._pending_futs:
            f.result()
        return {
            "groups": {name: {"C": g.C.clone()} for name, g in self.groups.items()},
            "weight": self.weight,
            "steps": self.steps,
            "t": self.t,
        }

    def load_state_dict(self, state: dict) -> None:
        for name, saved in state["groups"].items():
            self.groups[name].C.copy_(saved["C"])
        self.weight = state["weight"]
        self.steps = state["steps"]
        self.t = state["t"]


def _host_headblock_eigh(host_c: torch.Tensor, damping: float, inner_power: float, outer_power: float, copied_ev):
    torch.set_num_threads(2)
    copied_ev.synchronize()
    return exact_head_block_powers(
        host_c, damping=damping, inner_power=inner_power, outer_power=outer_power
    )


class HeadBlockMoments:
    """Sharded per-head block-diagonal [per, H, hd, hd] second-moment EMAs and async-host exact roots."""

    def __init__(
        self,
        rows: dict[tuple[int, str], torch.Tensor],
        kinds: tuple[str, ...],
        *,
        layers: int,
        hidden: int,
        heads: int,
        decay: float,
        damping: float,
        inner_power: float,
        outer_power: float,
        rows_total: int,
        world: int,
        rank: int,
        device,
        async_host: bool = False,
    ):
        if hidden % heads != 0:
            raise ValueError(f"Hidden width {hidden} is not divisible by heads {heads}")
        self.heads = heads
        self.hd = hidden // heads
        self.hidden = hidden
        self.decay = float(decay)
        self.damping = float(damping)
        self.inner_power = float(inner_power)
        self.outer_power = float(outer_power)
        self.rows_total = int(rows_total)
        self.world = world
        self.rank = rank
        self.device = device
        self.is_cuda = torch.device(device).type == "cuda"
        self.async_host = bool(async_host) and self.is_cuda

        l_counts = layer_counts(layers, world)
        self.counts = [len(kinds) * c for c in l_counts]
        self.keys = [(layer, kind) for layer in range(layers) for kind in kinds]
        self.bufs = [rows[key] for key in self.keys]
        self.per = self.counts[rank]
        lo = sum(self.counts[:rank])
        self.own = self.keys[lo : lo + self.per]
        flat_block = heads * self.hd * self.hd
        self.partial = torch.empty(len(self.keys), flat_block, dtype=torch.float32, device=device)
        self.summed = torch.empty(self.per, flat_block, dtype=torch.float32, device=device)
        self.C = torch.zeros(self.per, heads, self.hd, self.hd, dtype=torch.float32, device=device)
        self.host_C = (
            torch.empty(self.per, heads, self.hd, self.hd, dtype=torch.float32, pin_memory=True)
            if self.async_host and self.per > 0
            else None
        )
        self.host_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if self.async_host else None
        self._pending_fut: concurrent.futures.Future | None = None
        self.weight = 0.0
        self.pg = dist.new_group(backend=dist.get_backend()) if world > 1 else None

    @torch.no_grad()
    def compute_local_grams(self) -> None:
        K = len(self.bufs)
        R = self.bufs[0].shape[0]
        H, hd = self.heads, self.hd
        stacked = torch.stack(self.bufs)
        xh = stacked.view(K, R, H, hd).permute(0, 2, 3, 1).contiguous().view(K * H, hd, R)
        if self.is_cuda and xh.dtype == torch.bfloat16:
            grams = torch.bmm(xh, xh.mT, out_dtype=torch.float32)
        else:
            xf = xh.float()
            grams = torch.bmm(xf, xf.mT)
        self.partial.copy_(grams.view(K, H * hd * hd))

    @torch.no_grad()
    def start_ema_and_async_roots(self, stream) -> None:
        if self.world == 1:
            self.summed.copy_(self.partial)
        elif len(set(self.counts)) == 1:
            dist.reduce_scatter_tensor(self.summed, self.partial, group=self.pg)
        else:
            dist.reduce_scatter(self.summed, list(self.partial.split(self.counts)), group=self.pg)
        b = self.decay
        if self.per:
            self.C.mul_(b).add_(self.summed.view_as(self.C), alpha=(1.0 - b) / self.rows_total)
        self.weight = b * self.weight + (1.0 - b)
        if self.async_host and self.per > 0:
            self.host_C.copy_(self.C, non_blocking=True)
            ev_copied = torch.cuda.Event(blocking=True)
            ev_copied.record(stream)
            self._pending_fut = self.host_pool.submit(
                _host_headblock_eigh, self.host_C, self.damping, self.inner_power, self.outer_power, ev_copied
            )

    def hosts_ready(self) -> bool:
        return self._pending_fut is None or self._pending_fut.done()

    @torch.no_grad()
    def collect_async_roots(self, stream) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
        if self._pending_fut is not None:
            ri_cpu, ro_cpu = self._pending_fut.result()
            self._pending_fut = None
            with torch.cuda.stream(stream):
                ri = ri_cpu.pin_memory().to(self.device, non_blocking=True)
                ro = ro_cpu.pin_memory().to(self.device, non_blocking=True)
            return {key: (ri[i], ro[i]) for i, key in enumerate(self.own)}
        ri, ro = exact_head_block_powers(
            self.C,
            damping=self.damping,
            inner_power=self.inner_power,
            outer_power=self.outer_power,
        )
        return {key: (ri[i], ro[i]) for i, key in enumerate(self.own)}

    @torch.no_grad()
    def finish_update_and_roots(self) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
        if self.world == 1:
            self.summed.copy_(self.partial)
        elif len(set(self.counts)) == 1:
            dist.reduce_scatter_tensor(self.summed, self.partial, group=self.pg)
        else:
            dist.reduce_scatter(self.summed, list(self.partial.split(self.counts)), group=self.pg)
        b = self.decay
        if self.per:
            self.C.mul_(b).add_(self.summed.view_as(self.C), alpha=(1.0 - b) / self.rows_total)
        self.weight = b * self.weight + (1.0 - b)
        ri, ro = exact_head_block_powers(
            self.C,
            damping=self.damping,
            inner_power=self.inner_power,
            outer_power=self.outer_power,
        )
        return {key: (ri[i], ro[i]) for i, key in enumerate(self.own)}

    @torch.no_grad()
    def update_and_factors(self) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
        self.compute_local_grams()
        return self.finish_update_and_roots()

    def shutdown(self) -> None:
        if self.host_pool is not None:
            self.host_pool.shutdown(wait=True)

    def state_dict(self) -> dict:
        if self._pending_fut is not None:
            self._pending_fut.result()
        return {"C": self.C.clone(), "weight": self.weight}

    def load_state_dict(self, state: dict) -> None:
        self.C.copy_(state["C"])
        self.weight = state["weight"]
