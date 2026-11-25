#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect fixed-data output moments at saved models without optimizer updates.

All checkpoints use the same token windows and the same compiled forward/backward.
This is a checkpoint probe, not a reconstruction of a training EMA trajectory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits

PAPER = Path(__file__).resolve().parents[1]
IDEA = PAPER.parent
CANDIDATE = (
    Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    / "candidates/sym-mlpdiag-fastlag-20261120"
)
sys.path.insert(0, str(CANDIDATE))

import experiment.reference  # noqa: E402,F401
from experiment.fast_cotangent import attach_fast_block_observers  # noqa: E402
from reproduce.marin.data import MarinMixture, pack  # noqa: E402
from reproduce.marin.model import Model, sample_positions  # noqa: E402
from torch.nn import functional as F  # noqa: E402

KINDS = ("q", "k", "v", "o", "down", "gate", "up")
KS = (8, 16, 32, 64)


def top_basis(item):
    key, tensor = item
    m = tensor.numpy().astype(np.float64)
    m = (m + m.T) * 0.5
    d = m.shape[0]
    scale = np.trace(m) / d
    assert scale > 0 and np.isfinite(m).all()
    m /= scale
    w, v = scipy.linalg.eigh(m, subset_by_index=(d - 65, d - 1), driver="evr")
    residual = np.linalg.norm(m @ v - v * w) / np.linalg.norm(m)
    gaps = [(w[-k] - w[-k - 1]) / max(abs(w[-k]), 1e-300) for k in KS]
    return key, v[:, -64:].astype(np.float32), np.array(gaps), float(residual)


def prepare_tokens(root: Path, out: Path, windows: int):
    mixture = MarinMixture(str(root / "data"), seed=0, heldout_windows=64)
    # A held-out mixture with deterministic proportions; same windows at every checkpoint.
    indices = []
    for replicate in range(2):
        comp_ids = [0] * (windows - 2) + [1, 2]
        indices.append([(c, int(mixture.heldout[c][replicate * windows + j])) for j, c in enumerate(comp_ids)])
    tokens = np.stack([[mixture.streams[c].window(w) for c, w in group] for group in indices])
    np.save(out / "probe_tokens.npy", tokens)
    meta = {
        "windows_per_replicate": windows,
        "replicates": 2,
        "tokens_per_window": 4096,
        "sources": indices,
        "selection": "held-out seed-0 mixture windows, disjoint replicates",
        "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
    }
    (out / "probe_data.json").write_text(json.dumps(meta, indent=2) + "\n")


def run_worker(args):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda", args.device)
    jobs = json.loads(args.manifest.read_text())[args.worker :: args.workers]
    tokens = np.load(args.out / "probe_tokens.npy")
    model, body, loss_fn, rows = None, None, None, None
    current_width = None
    options = {
        "max_autotune": True,
        "triton.cudagraphs": False,
        "max_autotune_gemm_backends": "ATEN",
        "emulate_precision_casts": True,
    }
    start = time.monotonic()
    for job in jobs:
        result_path = args.out / f"probe_{job['id']}.npz"
        if result_path.exists():
            print("cached", job["id"], flush=True)
            continue
        ckpt_path = Path(job["checkpoint"])
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
        config = ck["config"]
        width = config["model"]["hidden"]
        if width != current_width:
            if model is not None:
                del model, body, loss_fn, rows
                torch.cuda.empty_cache()
            model = Model(config["model"]).to(device)
            # Only activation gradients are required; the checkpoint stays fixed.
            model.requires_grad_(False)
            model.body.emb.weight.requires_grad_(True)
            positions = sample_positions(4096, 32, 16)
            rows, row_off, _, _ = attach_fast_block_observers(
                model.body.blocks,
                hidden_mode="full",
                mlp_mode="full",
                positions=positions,
                capacity=len(positions),
                device=device,
            )
            assert row_off.item() == 0
            body = torch.compile(model.body, options=options, dynamic=False)

            def loss_impl(h, weight, target):
                return F.cross_entropy(F.linear(h, weight.to(torch.bfloat16)).float(), target, ignore_index=-100)

            loss_fn = torch.compile(loss_impl, options=options, dynamic=False)
            current_width = width
        model.load_state_dict(ck["model"], strict=True)
        metadata = {
            **job,
            "config": config,
            "checkpoint_bytes": ckpt_path.stat().st_size,
            "checkpoint_mtime_ns": ckpt_path.stat().st_mtime_ns,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "probe": json.loads((args.out / "probe_data.json").read_text()),
            "sample_positions": positions,
            "loss_scale": 65536,
            "statistics": "Uncentered, equally weighted fixed-checkpoint moments; no EMA or updates",
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "capture_source_sha256": (
                hashlib.sha256((CANDIDATE / "experiment/fast_cotangent.py").read_bytes()).hexdigest()
            ),
            "model_source_sha256": hashlib.sha256(Path(sys.modules[Model.__module__].__file__).read_bytes()).hexdigest(),
        }
        del ck
        sums = {}
        losses = []
        basis_output = {}
        for rep in range(tokens.shape[0]):
            sums.clear()
            rep_losses = []
            for n in range(tokens.shape[1]):
                tok, target, cu, meta = pack(tokens[rep, n : n + 1], 2049)
                tok, target, cu = tok.to(device), target.to(device), cu.to(device)
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = body(tok, cu, meta).reshape(-1, width)
                    loss = loss_fn(hidden, model.head.weight, target.reshape(-1))
                loss.backward()
                rep_losses.append(float(loss.detach()))
                with torch.no_grad():
                    for key, buf in rows.items():
                        x = buf.float()
                        gram = x.T @ x
                        if key in sums:
                            sums[key].add_(gram)
                        else:
                            sums[key] = gram
                if n == 0 or n + 1 == tokens.shape[1]:
                    print(
                        job["id"],
                        "rep",
                        rep,
                        "window",
                        n + 1,
                        "loss",
                        rep_losses[-1],
                        "elapsed",
                        round(time.monotonic() - start, 1),
                        flush=True,
                    )
            cpu_moments = [(key, value.cpu()) for key, value in sums.items()]
            sums.clear()
            with threadpool_limits(limits=1):
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    for key, vectors, gaps, residual in pool.map(top_basis, cpu_moments):
                        prefix = f"r{rep}_l{key[0]:02d}_{key[1]}"
                        basis_output[prefix] = vectors
                        basis_output[prefix + "_gaps"] = gaps
                        basis_output[prefix + "_residual"] = residual
            del cpu_moments
            losses.append(rep_losses)
        metadata["losses"] = losses
        metadata["elapsed_seconds"] = time.monotonic() - start
        basis_output["metadata"] = json.dumps(metadata)
        np.savez_compressed(result_path, **basis_output)
        print("DONE", job["id"], "elapsed", round(time.monotonic() - start, 1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare_tokens(args.root, args.out, args.windows)
    else:
        run_worker(args)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
