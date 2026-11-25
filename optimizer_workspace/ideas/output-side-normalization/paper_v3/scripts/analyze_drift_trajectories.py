#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure eigenspace drift directly from uninterrupted training checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits

KINDS = ("q", "k", "v", "o", "down")
KS = (8, 16, 32, 64)


def checkpoint_bases(task):
    run, step, rank, cache = task
    cache = Path(cache)
    path = Path(run) / "checkpoints" / f"step_{step:06d}" / f"rank_{rank}.pt"
    receipt = {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()[:24]
    dest = cache / f"{key}.npz"
    if dest.exists():
        return str(dest)
    torch.set_num_threads(1)
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    groups = ck["fast"]["output"]["groups"]
    mats = groups["out_hidden"]["C"]
    assert mats.shape[0] == 20 and ck["world"] == 8 and ck["rank"] == rank
    vs, gaps, residuals = [], [], []
    with threadpool_limits(limits=1):
        for raw in mats:
            m = raw.double().numpy()
            m = (m + m.T) * 0.5
            d = m.shape[0]
            w, v = scipy.linalg.eigh(m, subset_by_index=(d - 65, d - 1), driver="evr")
            vs.append(v[:, -64:].astype(np.float32))
            gaps.append([(w[-k] - w[-k - 1]) / max(abs(w[-k]), 1e-300) for k in KS])
            residuals.append(float(np.linalg.norm(m @ v - v * w) / np.linalg.norm(m)))
    np.savez_compressed(
        dest,
        vectors=np.stack(vs),
        gaps=np.array(gaps),
        residuals=np.array(residuals),
        step=step,
        rank=rank,
        receipt=json.dumps(receipt),
    )
    return str(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "ema_basis_cache"
    cache.mkdir(exist_ok=True)
    runs = {
        "MuonT-null": args.root / "runs/wave1/null_all",
        "WT": args.root / "runs/wave_sym_fastlag/inner_outer_mlpdiag_lag",
    }
    steps = [*range(250, 5001, 250), 5120]
    records, receipts = [], []
    start = time.monotonic()
    for arm, run in runs.items():
        tasks = [(str(run), s, r, str(cache)) for s in steps for r in range(8)]
        paths = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, filename in enumerate(pool.map(checkpoint_bases, tasks), 1):
                with np.load(filename) as obj:
                    paths[int(obj["step"]), int(obj["rank"])] = filename
                if n % 16 == 0 or n == len(tasks):
                    print(f"{arm}: {n}/{len(tasks)} shards; {time.monotonic() - start:.1f}s", flush=True)
        for s1, s2 in pairwise(steps):
            for rank in range(8):
                with np.load(paths[s1, rank]) as a, np.load(paths[s2, rank]) as b:
                    av, bv = a["vectors"], b["vectors"]
                    ag, bg = a["gaps"], b["gaps"]
                    for j, kind in enumerate(KINDS * 4):
                        row = dict(
                            arm=arm,
                            scale="130M",
                            step1=s1,
                            step2=s2,
                            interval=s2 - s1,
                            layer=4 * rank + j // 5,
                            kind=kind,
                        )
                        for q, k in enumerate(KS):
                            u, v = av[j, :, -k:].astype(float), bv[j, :, -k:].astype(float)
                            row[f"drift{k}"] = float(1 - np.sum((u.T @ v) ** 2) / k)
                            row[f"gap{k}_min"] = float(min(ag[j, q], bg[j, q]))
                        records.append(row)
        for path in paths.values():
            with np.load(path) as a:
                receipts.append({**json.loads(str(a["receipt"])), "max_residual": float(a["residuals"].max())})
    result = {
        "metric": "1 - ||U.T V||_F^2/k",
        "ks": KS,
        "records": records,
        "sources": receipts,
        "scope": (
            "Uninterrupted 130M seed-0 checkpoints; dense q/k/v/o/down moments only. "
            "MuonT-null is the full-output passive control, with a different numerical path from WT."
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "ema_drift.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for kind in KINDS:
        values = {}
        for arm in runs:
            values[arm] = np.mean(
                [r["drift16"] for r in records if r["arm"] == arm and r["kind"] == kind and r["interval"] == 250]
            )
        print(kind, values, "reduction", 1 - values["WT"] / values["MuonT-null"], flush=True)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
