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

from .block_toner import BLOCK_ARMS, BlockToner, DiagonalMoments, OwnedOutputMoments
from .bootstrap import verify_snapshot
from .capture import OutputMoments, build_model, captures_input
from .data_access import require_receipt
from .execution_owner import require_training_owner
from .factors import OutputSettings
from .optimizer import build, toner_engines
from .outer_diagonal_checks import OuterDiagonalVerifier
from .owned_output_observer import OwnedOutputVerifier
from .records import (
    atomic_json,
    checkpoint_identity,
    manifest,
    restore_checkpoint,
    save_checkpoint,
    utc_now,
)
from .upstream_full_model_checks import UpstreamProductVerifier


def main():
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--launch-release", type=Path, help="Explicit separately frozen Wave-8 production authorization."
    )
    parser.add_argument("--max-steps", type=int, help="Execution check; keeps full reference schedule.")
    parser.add_argument(
        "--execution-boundary",
        type=int,
        help="Matched evaluation/checkpoint boundary for execution checks.",
    )
    parser.add_argument(
        "--verify-diagonal-capture",
        action="store_true",
        help="Compare fused captured rows with Torch during an execution check.",
    )
    parser.add_argument(
        "--verify-outer-diagonal-updates",
        action="store_true",
        help="Compare actual outer-only stacks before parameter application during an execution check.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    require_training_owner(args, config)
    verify_snapshot()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    model_config, training, runtime = (
        config["model"],
        config["training"],
        config["runtime"],
    )
    if world != training["world_size"]:
        raise ValueError(f"Expected {training['world_size']} ranks, received {world}")
    if runtime["attention"] != "FA2" or runtime["head"] != "compiled":
        raise ValueError("This A100 experiment requires FA2 and the compiled head")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if (
        args.execution_boundary is not None or args.verify_diagonal_capture or args.verify_outer_diagonal_updates
    ) and args.max_steps is None:
        raise ValueError("Additional execution gates require a bounded execution check")
    if args.verify_outer_diagonal_updates and config["experiment"]["arm"] not in (
        "outer_leftdiag_fused",
        "outer_leftdiag_upstream_fused",
        "null_leftdiag_upstream_fused",
    ):
        raise ValueError("Outer update verification covers only the fused outer-diagonal arms")
    if args.execution_boundary is not None and not 0 < args.execution_boundary <= args.max_steps:
        raise ValueError("Execution boundary must fall inside the bounded execution check")
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
    positions, rows = None, 0
    if capture:
        statistics = config["statistics"]
        positions = sample_positions(seq, statistics["head"], statistics["stride"])
        rows = micro * len(positions)
        if config["root"]["chunk"] != 0:
            raise ValueError("Output runner currently requires full accumulation buffers")
    model = build_model(
        config,
        positions=positions,
        row_capacity=accumulation * rows,
        rows_per_microbatch=rows,
        device=device,
    )
    initialize_model(model, seed, model_config["embedding_init_scale"])
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
        capacity=accumulation * rows,
        rows_total=accumulation * rows * world,
        world=world,
        rank=rank,
        device=device,
    )
    start_step = 0
    if args.resume:
        start_step = restore_checkpoint(
            args.resume,
            model,
            optimizers,
            fast,
            output,
            config=config,
            rank=rank,
            world=world,
            identity=identity,
        )
    diagonal_reference = None
    outer_verifier = None
    if args.verify_outer_diagonal_updates:
        verifier = (
            OwnedOutputVerifier if isinstance(fast.output_diagonal, OwnedOutputMoments) else UpstreamProductVerifier
        )
        outer_verifier = verifier(fast) if fast.arm.upstream_product_kernels else OuterDiagonalVerifier(fast)
        fast.update_verifier = outer_verifier
    if args.verify_diagonal_capture and outer_verifier is None:
        if not isinstance(fast, BlockToner) or fast.output_diagonal is None or fast.output_diagonal.backend != "triton":
            raise ValueError("Captured-row verification requires a fused diagonal output arm")
        actual = fast.output_diagonal
        diagonal_reference = DiagonalMoments(
            actual.bufs,
            decay=actual.decay,
            damping=actual.damping,
            inner_power=actual.inner_power,
            outer_power=actual.outer_power,
            rows_total=actual.rows_total,
            world=world,
            device=device,
            backend="torch",
        )
        diagonal_reference.load_state_dict(actual.state_dict())
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

    if fast:
        model.head.weight.register_post_accumulate_grad_hook(begin_hook)
    mixture = MarinMixture(
        args.data,
        weights=config["data"]["mixture_weights"],
        seq=seq,
        seed=seed,
        heldout_windows=training["heldout_windows"],
    )
    evaluation = eval_sets(args.data, mixture=mixture, include=["c4_en"], seq=seq)
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
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * schedule_factor(step, config["schedule"])
            loss_sum = torch.zeros((), device=device)
            for index in range(accumulation):
                if world > 1:
                    run.require_backward_grad_sync = index == accumulation - 1
                if fast and captures_input(arm):
                    model.body.stat_at.copy_(torch.arange(index * rows, (index + 1) * rows, device=device))
                if isinstance(fast, BlockToner):
                    fast.select_microbatch(index)
                start = step * batch + (rank * accumulation + index) * micro
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = run(*to_device(mixture.batch(start, micro)))
                if fast and index == accumulation - 1:
                    armed["value"] = True
                (loss / accumulation).backward()
                loss_sum += loss.detach() / accumulation
            if isinstance(fast, BlockToner):
                fast.finish_backward()
            if outer_verifier is not None:
                outer_verifier.begin()
            if diagonal_reference is not None:
                for width, reducer in fast.output_diagonal.reducers.items():
                    sums = reducer()
                    for index, key in enumerate(fast.output_diagonal.keys[width]):
                        buffer = fast.output_diagonal.bufs[key]
                        torch.testing.assert_close(
                            sums[index],
                            buffer.float().square().sum(0),
                            rtol=1e-5,
                            atol=1e-6,
                        )
                        torch.testing.assert_close(
                            sums[index].double(),
                            buffer.double().square().sum(0),
                            rtol=1e-5,
                            atol=1e-6,
                        )
                diagonal_reference.update()
            output_start = time.perf_counter()
            if output:
                output.update(accumulation * rows)
            output_elapsed = time.perf_counter() - output_start
            clip_gradients(clipping, config["optimizer"]["clip_norm"])
            for optimizer in optimizers:
                optimizer.step()
            outer_parity = {} if outer_verifier is None else outer_verifier.finish()
            if diagonal_reference is not None:
                for width, moment in fast.output_diagonal.moment.items():
                    torch.testing.assert_close(moment, diagonal_reference.moment[width], rtol=1e-5, atol=1e-6)
                reference_factors = diagonal_reference.factors()
                for key, factors in fast.output_diagonal.factors().items():
                    for actual, reference in zip(factors, reference_factors[key], strict=True):
                        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            if world > 1:
                dist.all_reduce(loss_sum)
                loss_sum /= world
            if not torch.isfinite(loss_sum).item():
                raise FloatingPointError("Nonfinite training loss")
            torch.cuda.synchronize()
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
                metrics["input_refresh_installations"] = fast.refresh_installations
            if diagonal_reference is not None or outer_verifier is not None:
                metrics["diagonal_capture_parity_passed"] = True
            metrics.update(outer_parity)
            if output:
                metrics["output_root_age"] = step - output.root_step
            if step % runtime["eval_every"] == 0 or completed == limit or completed == args.execution_boundary:
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
            if (
                completed % config["experiment"]["checkpoint_every"] == 0
                or completed == limit
                or completed == args.execution_boundary
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
            record.update(
                status="failed",
                end_time=utc_now(),
                completed_steps=completed,
                error=repr(error),
            )
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
