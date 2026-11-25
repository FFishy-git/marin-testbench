# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate completed final checkpoints across all 16 tokenized Paloma domains on 8xA100."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from experiment.capture import build_model
from reproduce.marin import head_ce
from reproduce.marin.data import MarinMixture, eval_sets, pack

ALL_PALOMA = [
    "4chan_meta_sep",
    "c4_100_domains",
    "c4_en",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "dolma-v1_5",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)

    config = json.loads((args.run_dir / "config.json").read_text())
    latest = json.loads((args.run_dir / "latest_checkpoint.json").read_text())
    ckpt_dir = Path(latest["path"])
    model_cfg = config["model"]
    seq = model_cfg["sequence_length"]
    micro = args.micro_batch_size or config["training"]["micro_batch_size"]

    model = build_model(config, positions=None, row_capacity=0, rows_per_microbatch=0, device=device)
    ckpt = torch.load(ckpt_dir / "rank_0.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    step = int(ckpt["step"])
    del ckpt
    model.eval()

    mixture = MarinMixture(
        args.data,
        weights=config["data"]["mixture_weights"],
        seq=seq,
        seed=config["training"]["seed"],
        heldout_windows=config["training"]["heldout_windows"],
    )
    evaluation = eval_sets(args.data, mixture=mixture, include=ALL_PALOMA, seq=seq)
    capacity = micro * seq // 2 + 1
    eos = config["data"]["eos_token_id"]

    def evaluate(dataset) -> float:
        size = len(dataset)
        per_rank = math.ceil(size / world)
        total = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad():
            for offset in range(0, per_rank, micro):
                start = rank * per_rank + offset
                real = max(0, min(micro, per_rank - offset, size - start))
                tokens = dataset.batch(start, real) if real else np.empty((0, seq), dtype=np.uint32)
                if real < micro:
                    tokens = np.concatenate((tokens, np.repeat(dataset.batch(0, 1), micro - real, axis=0)))
                toks, target, cu, meta = pack(tokens, capacity, eos)
                toks = toks.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                cu = cu.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.body(toks, cu, meta).to(torch.bfloat16).reshape(-1, model_cfg["hidden"])
                    _, _, losses = head_ce.head_ce_forward(hidden, model.head.weight.to(torch.bfloat16), target.reshape(-1))
                total[0] += losses.reshape(micro, seq)[:real].double().sum()
                total[1] += (target[:real] != -100).sum()
        if world > 1:
            dist.all_reduce(total)
        return (total[0] / total[1]).item()

    scores = {}
    for name, dataset in sorted(evaluation.items()):
        scores[name] = evaluate(dataset)
    paloma_vals = [v for k, v in scores.items() if k.startswith("paloma/")]
    heldout_weights = config["data"]["mixture_weights"]
    selection = sum(w * scores[f"heldout/{k}"] for k, w in heldout_weights.items()) / sum(heldout_weights.values())
    result = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_dir),
        "step": step,
        "arm": config["experiment"]["arm"],
        "model_size": model_cfg["size"],
        "selection_loss": selection,
        "paloma_macro_mean": sum(paloma_vals) / len(paloma_vals),
        "scores": scores,
    }
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"run": args.run_dir.name, "selection_loss": selection, "paloma_macro_mean": result["paloma_macro_mean"]}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
