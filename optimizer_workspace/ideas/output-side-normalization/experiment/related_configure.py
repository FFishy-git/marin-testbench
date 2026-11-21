# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare paired related-method comparisons without launching or replacing waves."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from .configure import BATCHES, configuration
from .go_muon import GoMuonSettings
from .mousse import MousseSettings
from .reference import TASK

CONTROL_ARMS = ("muon_control", "inner_outer_mlpdiag", "go_muon_control", "mousse_control")
PAPER_ARMS = ("muon_control", "inner_outer_mlpdiag", "go_muon_paper", "mousse_paper")
PROFILES = {"controlled": CONTROL_ARMS, "paper_adapters": PAPER_ARMS}
LR_MULTIPLIERS = (1 / math.sqrt(2), 1.0, math.sqrt(2))
SHORT_TOKENS = 167772160


def comparison_configuration(arm, *, batch, seed, microbatch, lr_multiplier, stage):
    config = configuration(arm, batch, seed, microbatch)
    if arm in ("muon_control", "mousse_control", "mousse_paper"):
        config["optimizer"]["name"] = "muon"
    config["optimizer"]["lr"] *= lr_multiplier
    config["experiment"]["matrix_decay"] = "isotropic"
    common = {
        "comparison": "related_baselines",
        "stage": stage,
        "matrix_lr_multiplier": lr_multiplier,
        "matrix_decay": "isotropic",
        "status": "unvalidated",
        "quality_claim": "none",
    }
    if arm.startswith("go_muon"):
        paper = arm == "go_muon_paper"
        settings = GoMuonSettings(
            statistic_mode="paper_sparse" if paper else "ema_per_step",
            polar_method="exact_svd" if paper else "newton_schulz",
        )
        config["experiment"]["related_baseline"] = {
            "source": "arxiv:2608.09763v1",
            "label": "GO-MUON source-recipe adapter" if paper else "GO-MUON-form common-factor control",
            "settings": asdict(settings),
            "adaptations": [
                "Common Marin model, data, sampled BF16 observations, auxiliary optimizer and schedule.",
                "Isotropic matrix decay is chosen here; the paper's matrix decay is unspecified.",
                "Capture remains attached on reuse steps; source hook-gated cost is not reproduced.",
                "Quarter-root eigendecomposition uses FP64 and returns FP32 factors.",
                "Cached quarter factors are held between exact root refreshes; ours refines current moments.",
                *(
                    []
                    if paper
                    else ["Common per-step EMA, trace ridge and finite NS replace OAS/sparse refresh/exact polar."]
                ),
            ],
        }
    elif arm.startswith("mousse"):
        paper = arm == "mousse_paper"
        settings = (
            MousseSettings(profile="paper")
            if paper
            else MousseSettings(
                beta=config["statistics"]["momentum"],
                damping=config["statistics"]["damping"],
                refresh=config["root"]["refresh"],
            )
        )
        config["experiment"]["related_baseline"] = {
            "source": "Mousse:d00c1bf17790fbe56424ee5567cce80d8e75f4b2",
            "label": "Mousse author-numerics adapter" if paper else "Mousse-form common-setting control",
            "settings": asdict(settings),
            "adaptations": [
                "Common Marin model, data, momentum, auxiliary optimizer and schedule.",
                "Owner routing and FP32 additive parameter changes replace the source distributed implementation.",
                "Zero-trace and zero-source behavior is defined explicitly.",
                "Finite NS runs in the eigenbasis, not the original coordinates used by ours.",
                *(
                    []
                    if paper
                    else ["Common EMA/ridge/refresh, finite NS and shape scaling replace source numerical choices."]
                ),
            ],
        }
    config["experiment"]["comparison"] = common
    if stage == "short":
        steps, remainder = divmod(SHORT_TOKENS, config["training"]["batch_tokens"])
        if remainder:
            raise ValueError("Short token budget must divide into full updates")
        config["training"]["steps"] = steps
        config["training"]["tokens"] = SHORT_TOKENS
        config["schedule"]["warmup_steps"] = 0
        config["schedule"]["stable_steps"] = steps // 5
        config["schedule"]["decay_steps"] = steps - steps // 5
        config["runtime"]["eval_every"] = max(1, steps // 20)
        config["experiment"]["checkpoint_every"] = steps
    elif stage != "full":
        raise ValueError(f"Unknown comparison stage: {stage}")
    return config


def prepare_comparison(out, *, profile, stage, batch, seeds, microbatch, lr_multipliers):
    """Write immutable configs and four-entry manifests usable by validate/launch."""
    out = out.resolve()
    out.relative_to(TASK / "experiment")
    out.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for seed in seeds:
        for index, multiplier in enumerate(lr_multipliers):
            entries = []
            for arm in PROFILES[profile]:
                name = f"{arm}_{batch}_seed{seed}_lr{index}"
                config = comparison_configuration(
                    arm,
                    batch=batch,
                    seed=seed,
                    microbatch=microbatch,
                    lr_multiplier=multiplier,
                    stage=stage,
                )
                path = out / f"{name}.json"
                text = json.dumps(config, indent=2) + "\n"
                path.write_text(text)
                entries.append(
                    {"arm": arm, "seed": seed, "batch": batch, "name": name, "config": str(path.relative_to(TASK))}
                )
                artifacts.append(
                    {"config": str(path.relative_to(TASK)), "sha256": hashlib.sha256(text.encode()).hexdigest()}
                )
            manifest = out / f"wave_seed{seed}_lr{index}.json"
            manifest.write_text(
                json.dumps({"wave_name": f"related_{profile}_{stage}_seed{seed}_lr{index}", "wave": entries}, indent=2)
                + "\n"
            )
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory under experiment/.")
    parser.add_argument("--profile", choices=PROFILES, default="controlled")
    parser.add_argument("--stage", choices=("short", "full"), default="short")
    parser.add_argument("--batch", choices=BATCHES, default="0p5m")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--lr-multipliers", nargs="+", type=float, default=LR_MULTIPLIERS)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in args.lr_multipliers):
        parser.error("LR multipliers must be positive and finite")
    artifacts = prepare_comparison(
        args.out,
        profile=args.profile,
        stage=args.stage,
        batch=args.batch,
        seeds=args.seeds,
        microbatch=args.microbatch,
        lr_multipliers=args.lr_multipliers,
    )
    print(json.dumps({"status": "prepared_not_validated", "configs": artifacts}, indent=2))


if __name__ == "__main__":
    main()
