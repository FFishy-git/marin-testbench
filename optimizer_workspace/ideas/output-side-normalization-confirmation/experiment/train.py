# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run controlled A100 Marin experiments using the pinned model and data."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from reproduce.marin import head_ce
from reproduce.marin.data import MarinMixture, eval_sets, pack
from reproduce.marin.model import initialize_model, sample_positions
from reproduce.marin.optimizer import clip_gradients
from reproduce.marin.train import schedule_factor
from torch import nn
from torch.nn import functional as F

from .block_toner import BLOCK_ARMS, BlockToner
from .bootstrap import verify_snapshot
from .capture import OutputMoments, build_model, captures_input
from .data_access import require_receipt
from .factors import OutputSettings
from .optimizer import build, restore_fast, toner_engines
from .records import atomic_json, checkpoint_identity, manifest, restore_checkpoint, save_checkpoint, utc_now


def _slice_engine_state(saved_eng: dict, sub: int, ratio: int) -> dict:
    groups = {}
    for gname, g in saved_eng["groups"].items():
        per_old = g["C"].shape[0]
        per_new = per_old // ratio
        lo, hi = sub * per_new, (sub + 1) * per_new
        groups[gname] = {
            "C": g["C"][lo:hi],
            "tol": g["tol"][lo:hi],
            "basis_step": g["basis_step"][lo:hi],
            "warm": False,
            "SB": None if g["SB"] is None else g["SB"][lo:hi],
            "S2": None if g["S2"] is None else g["S2"][lo:hi],
            "pending": g["pending"][lo:hi],
        }
    mom = {}
    for shape, vals in saved_eng["momentum"].items():
        mom[shape] = {}
        for k, t in vals.items():
            m_new = t.shape[0] // ratio
            mom[shape][k] = t[sub * m_new : (sub + 1) * m_new]
    return {**saved_eng, "groups": groups, "momentum": mom}


def _restore_bench_midtrain_state(ckpt_dir: Path, optimizers, fast, *, rank: int, world: int) -> None:
    comp_path = ckpt_dir / "complete.json"
    if not comp_path.exists():
        return
    saved_world = int(json.loads(comp_path.read_text()).get("world", 0))
    if saved_world == world:
        st = torch.load(ckpt_dir / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
        for opt, saved in zip(optimizers, st["optimizers"], strict=True):
            if opt is not fast and saved is not None:
                blrs = [g["base_lr"] for g in opt.param_groups]
                opt.load_state_dict(saved)
                for g, blr in zip(opt.param_groups, blrs, strict=True):
                    g["base_lr"] = blr
        if fast is not None and st.get("fast") is not None:
            restore_fast(fast, st["fast"])
        return
    if world > saved_world and world % saved_world == 0 and fast is not None:
        ratio = world // saved_world
        q, sub = divmod(rank, ratio)
        st = torch.load(ckpt_dir / f"rank_{q}.pt", map_location="cpu", weights_only=False)
        fs = st.get("fast")
        if fs is None:
            return
        if isinstance(fast, BlockToner):
            owned_layers = set()
            lo_l = sum(fast.layer_counts[:rank])
            owned_layers.update(range(lo_l, lo_l + fast.layer_counts[rank]))
            sliced = {
                "lagged": {k: v for k, v in fs.get("lagged", {}).items() if k[0] in owned_layers},
                "input": None if fs["input"] is None else _slice_engine_state(fs["input"], sub, ratio),
                "output": None if fs["output"] is None else _slice_engine_state(fs["output"], sub, ratio),
                "input_diagonal": fs["input_diagonal"],
                "output_diagonal": fs["output_diagonal"],
                "momentum": {
                    sh: t[sub * (t.shape[0] // ratio) : (sub + 1) * (t.shape[0] // ratio)]
                    for sh, t in fs["momentum"].items()
                },
                "steps": fs["steps"],
            }
            restore_fast(fast, sliced)
        else:
            restore_fast(fast, _slice_engine_state(fs, sub, ratio))


def main():
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, help="Execution check; keeps full reference schedule.")
    parser.add_argument("--init-model-from", type=Path, help="Load initial model weights from a saved checkpoint directory.")
    parser.add_argument("--init-state", action="store_true", help="Also restore optimizer and Toner statistics/basis state from --init-model-from.")
    parser.add_argument("--bench-midtrain", action="store_true", help="Benchmark mid-training steps without eval or checkpoint IO.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verify_snapshot()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    model_config, training, runtime = config["model"], config["training"], config["runtime"]
    if world != training["world_size"]:
        raise ValueError(f"Expected {training['world_size']} ranks, received {world}")
    if runtime["attention"] != "FA2" or runtime["head"] != "compiled":
        raise ValueError("This A100 experiment requires FA2 and the compiled head")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    require_receipt(args.data, args.data_receipt)
    identity = checkpoint_identity(config, args.data_receipt)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("These launch configurations target A100 SM80")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    seed = training["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(runtime["deterministic"])
    micro, seq = training["micro_batch_size"], model_config["sequence_length"]
    batch = training["global_batch_sequences"]
    accumulation = batch // (world * micro)
    if batch != accumulation * world * micro:
        raise ValueError("Global batch does not divide into full microbatches")
    arm = config["experiment"]["arm"]
    capture = config["optimizer"]["name"] == "muont"
    positions, rows, chunk = None, 0, accumulation
    if capture:
        statistics = config["statistics"]
        positions = sample_positions(seq, statistics["head"], statistics["stride"])
        rows = micro * len(positions)
        chunk = config["root"]["chunk"] or accumulation
        if accumulation % chunk != 0:
            raise ValueError("Statistic chunks must divide the accumulated microbatches")
        if chunk != accumulation and arm not in ("muont", *BLOCK_ARMS):
            raise ValueError("Non-block output runner currently requires full accumulation buffers")
    model = build_model(
        config,
        positions=positions,
        row_capacity=chunk * rows,
        rows_per_microbatch=rows,
        device=device,
    )
    initialize_model(model, seed, model_config["embedding_init_scale"])
    data_step_offset = 0
    if args.init_model_from is not None:
        ckpt = torch.load(args.init_model_from / "rank_0.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        data_step_offset = int(ckpt["step"])
        del ckpt
    count = sum(p.numel() for p in model.parameters())
    if count != config["experiment"]["expected_model_parameters"]:
        raise ValueError(f"Unexpected parameter count {count}")
    output = None
    if arm not in ("muon", "muont") and arm not in BLOCK_ARMS:
        output = OutputMoments(
            model,
            positions,
            OutputSettings(**config["experiment"]["output"]),
            world=world,
            rank=rank,
            device=device,
        )
    optimizers, fast, clipping = build(
        model,
        config,
        output,
        positions=positions,
        rows_per_microbatch=rows,
        capacity=chunk * rows,
        rows_total=accumulation * rows * world,
        world=world,
        rank=rank,
        device=device,
    )
    start_step = 0
    if args.resume:
        start_step = restore_checkpoint(
            args.resume, model, optimizers, fast, output, config=config, rank=rank, world=world, identity=identity
        )
    elif args.init_model_from is not None and args.init_state:
        _restore_bench_midtrain_state(args.init_model_from, optimizers, fast, rank=rank, world=world)
    compile_mode = runtime["compile"]
    body = (
        model.body
        if compile_mode == "eager"
        else torch.compile(model.body, options=runtime["compile_options"], dynamic=False)
    )

    def compiled_loss(h, weight, target):
        return F.cross_entropy(F.linear(h, weight.to(torch.bfloat16)).float(), target, ignore_index=-100)

    loss_fn = (
        compiled_loss
        if compile_mode == "eager"
        else torch.compile(compiled_loss, options=runtime["compile_options"], dynamic=False)
    )

    def forward_loss(tokens, target, cu, meta):
        hidden = body(tokens, cu, meta).to(torch.bfloat16).reshape(-1, model_config["hidden"])
        return loss_fn(hidden, model.head.weight, target.reshape(-1))

    model.forward = forward_loss
    run = (
        model
        if world == 1
        else nn.parallel.DistributedDataParallel(
            model,
            device_ids=None,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            # Keep bucket layouts fixed across fresh and resumed processes.
            find_unused_parameters=True,
        )
    )
    armed = {"value": False}

    def begin_hook(_):
        if armed["value"]:
            armed["value"] = False
            fast.begin()

    begin_mode = config.get("root", {}).get("begin", "head") if fast else "head"
    if fast:
        if begin_mode == "head":
            model.head.weight.register_post_accumulate_grad_hook(begin_hook)
        if isinstance(fast, BlockToner):
            fast.defer_finite_check = True
    mixture = MarinMixture(
        args.data,
        weights=config["data"]["mixture_weights"],
        seq=seq,
        seed=seed,
        heldout_windows=training["heldout_windows"],
    )
    evaluation = None if args.bench_midtrain else eval_sets(args.data, mixture=mixture, include=["c4_en"], seq=seq)
    capacity = micro * seq // 2 + 1

    def to_device(tokens):
        tokens, target, cu, meta = pack(tokens, capacity, config["data"]["eos_token_id"])
        return (
            tokens.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
            cu.to(device, non_blocking=True),
            meta,
        )

    def evaluate(dataset):
        size = len(dataset)
        per_rank = math.ceil(size / world)
        total = torch.zeros(2, dtype=torch.float64, device=device)
        for offset in range(0, per_rank, micro):
            start = rank * per_rank + offset
            real = max(0, min(micro, per_rank - offset, size - start))
            tokens = dataset.batch(start, real) if real else np.empty((0, seq), dtype=np.uint32)
            if real < micro:
                tokens = np.concatenate((tokens, np.repeat(dataset.batch(0, 1), micro - real, axis=0)))
            tokens, target, cu, meta = to_device(tokens)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = body(tokens, cu, meta).to(torch.bfloat16).detach().reshape(-1, model_config["hidden"])
                _, _, losses = head_ce.head_ce_forward(hidden, model.head.weight.to(torch.bfloat16), target.reshape(-1))
            total[0] += losses.reshape(micro, seq)[:real].double().sum()
            total[1] += (target[:real] != -100).sum()
        if world > 1:
            dist.all_reduce(total)
        return (total[0] / total[1]).item()

    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=args.resume is not None)
        if args.resume:
            existing = json.loads((args.out / "run.json").read_text())
            if existing["config"] != config:
                raise ValueError("Output directory contains a different experiment")
            # A resume writes a separate metric segment, preserving previous evidence.
        atomic_json(args.out / "config.json", config)
        record = manifest(config, args.data, args.data_receipt)
        record["resume"] = None if args.resume is None else str(args.resume.resolve())
        record["init_model_from"] = None if args.init_model_from is None else str(args.init_model_from.resolve())
        record["bench_midtrain"] = bool(args.bench_midtrain)
        record["data_step_offset"] = data_step_offset
        record["execution_check"] = args.max_steps is not None
        atomic_json(args.out / "run.json", record)
    if world > 1:
        dist.barrier()
    limit = min(training["steps"], args.max_steps) if args.max_steps else training["steps"]
    if start_step >= limit:
        raise ValueError("Checkpoint has already reached the requested step limit")
    metric_path = args.out / f"metrics_from_{start_step:06d}.jsonl"
    if rank == 0 and metric_path.exists():
        raise FileExistsError(metric_path)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    completed = start_step
    try:
        for step in range(start_step, limit):
            step_start = time.perf_counter()
            model.zero_grad(set_to_none=True)
            if output:
                output.begin(micro * (seq - 1) * accumulation)
            eff_step = step + data_step_offset
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * schedule_factor(eff_step, config["schedule"])
            loss_sum = torch.zeros((), device=device)
            for index in range(accumulation):
                if world > 1:
                    run.require_backward_grad_sync = index == accumulation - 1
                if fast and captures_input(arm):
                    if chunk < accumulation and index > 0 and index % chunk == 0:
                        fast.chunk_wait()
                    slot = index % chunk
                    model.body.stat_at.copy_(torch.arange(slot * rows, (slot + 1) * rows, device=device))
                if isinstance(fast, BlockToner):
                    fast.select_microbatch(index, chunk=chunk)
                start = eff_step * batch + (rank * accumulation + index) * micro
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = run(*to_device(mixture.batch(start, micro)))
                if (
                    fast
                    and captures_input(arm)
                    and chunk < accumulation
                    and (index + 1) % chunk == 0
                    and index < accumulation - 1
                ):
                    fast.chunk_stats()
                if fast and index == accumulation - 1:
                    if begin_mode == "head":
                        armed["value"] = True
                    else:
                        fast.begin()
                if isinstance(fast, BlockToner):
                    fast.before_microbatch_backward(index, chunk=chunk)
                (loss / accumulation).backward()
                if (
                    isinstance(fast, BlockToner)
                    and chunk < accumulation
                    and (index + 1) % chunk == 0
                    and index < accumulation - 1
                ):
                    fast.chunk_output_stats()
                loss_sum += loss.detach() / accumulation
            if isinstance(fast, BlockToner):
                fast.finish_backward()
            output_start = time.perf_counter()
            if output:
                output.update(accumulation * rows)
            output_elapsed = time.perf_counter() - output_start
            clip_gradients(clipping, config["optimizer"]["clip_norm"])
            for optimizer in optimizers:
                optimizer.step()
            if world > 1:
                dist.all_reduce(loss_sum)
                loss_sum /= world
            if not torch.isfinite(loss_sum).item():
                raise FloatingPointError("Nonfinite training loss")
            torch.cuda.current_stream(device).synchronize()
            if isinstance(fast, BlockToner):
                fast.check_last_step_finite()
            completed = step + 1
            metrics = {
                "step": completed,
                "tokens": completed * batch * seq,
                "train_loss": loss_sum.item(),
                "step_seconds": time.perf_counter() - step_start,
                "output_host_seconds": output_elapsed,
                "elapsed_seconds": time.perf_counter() - wall_start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            if isinstance(fast, BlockToner):
                metrics.update(fast.last_metrics)
            elif fast:
                metrics["input_fallbacks"] = fast.fallbacks
                metrics["input_gpu_ms_previous"] = fast.gpu_ms
            if output:
                metrics["output_root_age"] = step - output.root_step
            if not args.bench_midtrain and (step % runtime["eval_every"] == 0 or completed == limit):
                eval_start = time.perf_counter()
                metrics["c4_loss"] = evaluate(evaluation["paloma/c4_en"])
                metrics["heldout_losses"] = {
                    name: evaluate(dataset) for name, dataset in evaluation.items() if name.startswith("heldout/")
                }
                weights = config["data"]["mixture_weights"]
                metrics["selection_loss"] = sum(
                    weight * metrics["heldout_losses"][f"heldout/{name}"] for name, weight in weights.items()
                ) / sum(weights.values())
                metrics["eval_seconds"] = time.perf_counter() - eval_start
            if not args.bench_midtrain and (
                completed % config["experiment"]["checkpoint_every"] == 0 or completed == limit
            ):
                save_start = time.perf_counter()
                save_checkpoint(
                    args.out,
                    model,
                    optimizers,
                    fast,
                    output,
                    step=completed,
                    config=config,
                    rank=rank,
                    world=world,
                    identity=identity,
                )
                metrics["checkpoint_seconds"] = time.perf_counter() - save_start
            metrics["end_to_end_seconds"] = time.perf_counter() - wall_start
            if rank == 0:
                print(json.dumps(metrics), flush=True)
                with metric_path.open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
        if rank == 0:
            record.update(
                status="execution_check_complete" if args.max_steps else "complete",
                end_time=utc_now(),
                completed_steps=completed,
                elapsed_seconds=time.perf_counter() - wall_start,
            )
            atomic_json(args.out / "run.json", record)
    except BaseException as error:
        if rank == 0:
            record.update(status="failed", end_time=utc_now(), completed_steps=completed, error=repr(error))
            atomic_json(args.out / "run.json", record)
        raise
    finally:
        for engine in toner_engines(fast):
            engine.side.synchronize()
            engine.pool.shutdown(wait=True)
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
