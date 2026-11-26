#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize training-EMA and fixed-data checkpoint subspace diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

KINDS = ("q", "k", "v", "o", "down", "gate", "up")
KS = (8, 16, 32, 64)


def read_probe(path):
    with np.load(path) as x:
        metadata = json.loads(str(x["metadata"]))
        vectors = {
            key: x[key].astype(np.float64)
            for key in x.files
            if key.startswith("r") and not key.endswith(("gaps", "residual"))
        }
        gaps = {key: x[key] for key in x.files if key.endswith("gaps")}
        residual = max(float(x[key]) for key in x.files if key.endswith("residual"))
    return metadata, vectors, gaps, residual


def drift(u, v, k):
    value = float(1 - np.sum((u[:, -k:].T @ v[:, -k:]) ** 2) / k)
    assert -1e-6 <= value <= 1 + 1e-6
    return value


def compare(records, baseline, arm, k=16, allow_partial=False, min_gap=0.0):
    dimensions = sorted(set((r["scale"], r["seed"], r["kind"], r["interval"]) for r in records))
    comparisons = []
    for scale, seed, kind, interval in dimensions:
        selected = [
            r for r in records if (r["scale"], r["seed"], r["kind"], r["interval"]) == (scale, seed, kind, interval)
        ]
        a = {(r["step1"], r["layer"], r.get("replicate", 0)): r for r in selected if r["arm"] == baseline}
        b = {(r["step1"], r["layer"], r.get("replicate", 0)): r for r in selected if r["arm"] == arm}
        assert len(a) == sum(r["arm"] == baseline for r in selected)
        assert len(b) == sum(r["arm"] == arm for r in selected)
        if not a or not b:
            assert allow_partial, f"Missing {baseline} or {arm} for {(scale, seed, kind, interval)}"
            continue
        if not allow_partial:
            assert a.keys() == b.keys()
        keys = sorted(a.keys() & b.keys())
        paired_count = len(keys)
        if min_gap:
            keys = [key for key in keys if min(a[key][f"gap{k}_min"], b[key][f"gap{k}_min"]) >= min_gap]
        if not keys:
            continue
        av = np.array([a[key][f"drift{k}"] for key in keys])
        bv = np.array([b[key][f"drift{k}"] for key in keys])
        intervals = sorted(set(key[0] for key in keys))
        repmeans = []
        for rep in sorted(set(key[2] for key in keys)):
            aa = np.mean([a[key][f"drift{k}"] for key in keys if key[2] == rep])
            bb = np.mean([b[key][f"drift{k}"] for key in keys if key[2] == rep])
            repmeans.append(
                {
                    "replicate": rep,
                    "baseline": float(aa),
                    "arm_mean": float(bb),
                    "reduction_pct": float(100 * (1 - bb / aa)),
                }
            )
        wins = sum(
            np.mean([b[key][f"drift{k}"] - a[key][f"drift{k}"] for key in keys if key[0] == step]) < 0
            for step in intervals
        )
        comparisons.append(
            {
                "scale": scale,
                "seed": seed,
                "kind": kind,
                "interval": interval,
                "k": k,
                "baseline": baseline,
                "baseline_mean": float(av.mean()),
                "arm": arm,
                "arm_mean": float(bv.mean()),
                "reduction_pct": float(100 * (1 - bv.mean() / av.mean())),
                "paired_difference": float((bv - av).mean()),
                "intervals": len(intervals),
                "intervals_lower": int(wins),
                "layer_interval_replicates": len(keys),
                "eligible_pairs": paired_count,
                "min_boundary_gap": min_gap,
                "fraction_lower": float(np.mean(bv < av)),
                "replicate_means": repmeans,
            }
        )
    return comparisons


def interval_means(records):
    groups = {}
    for row in records:
        key = tuple(row[k] for k in ("scale", "seed", "arm", "step1", "step2", "interval", "kind"))
        key = (*key, row.get("replicate", 0))
        groups.setdefault(key, []).append(row)
    output = []
    fields = ("scale", "seed", "arm", "step1", "step2", "interval", "kind", "replicate")
    for key, rows in sorted(groups.items()):
        assert len(rows) == 32 and len({row["layer"] for row in rows}) == 32
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                "layers": 32,
                **{f"drift{k}": float(np.mean([r[f"drift{k}"] for r in rows])) for k in KS},
            }
        )
    return output


def sampling_means(records):
    """Summarize between-set variation at fixed checkpoints, not temporal drift."""
    groups = {}
    for row in records:
        key = tuple(row[k] for k in ("scale", "seed", "arm", "kind"))
        groups.setdefault(key, []).append(row)
    return [
        {
            **dict(zip(("scale", "seed", "arm", "kind"), key, strict=True)),
            "checkpoint_layer_count": len(rows),
            **{f"drift{k}": float(np.mean([r[f"drift{k}"] for r in rows])) for k in KS},
        }
        for key, rows in sorted(groups.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--compact", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert len({job["id"] for job in manifest}) == len(manifest)
    design = json.loads((args.diagnostics / "probes/probe_data.json").read_text())
    missing = [job["id"] for job in manifest if not (args.diagnostics / "probes" / f"probe_{job['id']}.npz").exists()]
    if missing and not args.allow_partial:
        raise RuntimeError(f"Missing {len(missing)} checkpoint probes")
    records, sampling, sources = [], [], []
    for scale, seed, arm in sorted(set((j["scale"], j["seed"], j["arm"]) for j in manifest)):
        jobs = sorted(
            [j for j in manifest if (j["scale"], j["seed"], j["arm"]) == (scale, seed, arm)], key=lambda j: j["step"]
        )
        previous = None
        for job in jobs:
            if job["id"] in missing:
                previous = None
                continue
            meta, vec, gaps, residual = read_probe(args.diagnostics / "probes" / f"probe_{job['id']}.npz")
            assert (meta["scale"], meta["seed"], meta["arm"], meta["step"]) == (scale, seed, arm, job["step"])
            assert meta["probe"]["tokens_sha256"] == design["tokens_sha256"]
            assert residual < 1e-10
            expected = {f"r{rep}_l{layer:02d}_{kind}" for rep in range(2) for layer in range(32) for kind in KINDS}
            assert set(vec) == expected
            hidden = meta["config"]["model"]["hidden"]
            for key, basis in vec.items():
                width = hidden * (4 if key.rsplit("_", 1)[-1] in ("gate", "up") else 1)
                assert basis.shape == (width, 64) and np.isfinite(basis).all()
                assert np.linalg.norm(basis.T @ basis - np.eye(64)) < 1e-5
            sources.append(
                {
                    "id": job["id"],
                    "scale": scale,
                    "seed": seed,
                    "arm": arm,
                    "step": job["step"],
                    "checkpoint": meta["checkpoint"],
                    "lr": meta["config"]["optimizer"]["lr"],
                    "max_residual": residual,
                    "tokens_sha256": meta["probe"]["tokens_sha256"],
                    "script_sha256": meta["script_sha256"],
                    "checkpoint_mtime_ns": meta["checkpoint_mtime_ns"],
                }
            )
            for layer in range(32):
                for kind in KINDS:
                    u, v = vec[f"r0_l{layer:02d}_{kind}"], vec[f"r1_l{layer:02d}_{kind}"]
                    sampling.append(
                        {
                            "scale": scale,
                            "seed": seed,
                            "arm": arm,
                            "step": job["step"],
                            "layer": layer,
                            "kind": kind,
                            **{f"drift{k}": drift(u, v, k) for k in KS},
                        }
                    )
            if previous is not None:
                oldjob, oldvec, oldgaps = previous
                for rep in range(2):
                    for layer in range(32):
                        for kind in KINDS:
                            key = f"r{rep}_l{layer:02d}_{kind}"
                            row = {
                                "scale": scale,
                                "seed": seed,
                                "arm": arm,
                                "step1": oldjob["step"],
                                "step2": job["step"],
                                "interval": job["step"] - oldjob["step"],
                                "replicate": rep,
                                "layer": layer,
                                "kind": kind,
                            }
                            for q, k in enumerate(KS):
                                row[f"drift{k}"] = drift(oldvec[key], vec[key], k)
                                row[f"gap{k}_min"] = float(min(oldgaps[key + "_gaps"][q], gaps[key + "_gaps"][q]))
                            records.append(row)
            previous = job, vec, gaps
    ema = json.loads((args.diagnostics / "ema_drift.json").read_text())
    for row in ema["records"]:
        row["seed"] = 0
    result = {
        "missing": missing,
        "probe_records": records,
        "sampling_records": sampling,
        "ema_records": ema["records"],
        "sources": sources,
        "baseline": "Muon",
        "manifest": str(args.manifest.resolve()),
        "probe_comparisons": [
            c for arm in ("MuonT", "WT") for k in KS for c in compare(records, "Muon", arm, k, args.allow_partial)
        ],
        "incremental_comparisons": [c for k in KS for c in compare(records, "MuonT", "WT", k, args.allow_partial)],
        "ema_comparisons": [c for k in KS for c in compare(ema["records"], "MuonT-null", "WT", k)],
        "probe_gap_comparisons": [
            c for arm in ("MuonT", "WT") for c in compare(records, "Muon", arm, 16, args.allow_partial, min_gap=0.01)
        ],
        "ema_gap_comparisons": compare(ema["records"], "MuonT-null", "WT", 16, min_gap=0.01),
        "sampling_summary": sampling_means(sampling),
        "design": design,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.compact:
        assert not missing
        compact = {
            key: value
            for key, value in result.items()
            if key not in ("probe_records", "ema_records", "sampling_records")
        }
        compact.update(
            ema_records=interval_means(ema["records"]),
            probe_records=interval_means(records),
            raw_results=str(args.out.resolve()),
            checkpoint_counts={
                f"{scale}_seed{seed}": sum(j["scale"] == scale and j["seed"] == seed for j in manifest)
                for scale, seed in sorted({(j["scale"], j["seed"]) for j in manifest})
            },
            ema_source_file=str((args.diagnostics / "ema_drift.json").resolve()),
        )
        args.compact.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(compact, separators=(",", ":"), allow_nan=False) + "\n").encode()
        args.compact.write_bytes(gzip.compress(payload, mtime=0))
    print("missing", len(missing), "probe records", len(records))
    for c in result["probe_comparisons"]:
        if c["k"] == 16:
            print(
                c["scale"],
                c["seed"],
                c["kind"],
                c["arm"],
                c["interval"],
                f"{c['baseline_mean']:.4f} -> {c['arm_mean']:.4f}",
                f"{c['reduction_pct']:+.1f}%",
                f"{c['intervals_lower']}/{c['intervals']} intervals",
                flush=True,
            )


if __name__ == "__main__":
    main()
