# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare equal-budget LR trials outside any frozen experiment source tree."""

import argparse
import copy
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

FAMILIES = ("muon", "muont", "inner_outer_mlpdiag", "inner_outer_leftdiag")
BATCH_TOKENS = {"0p5m": 524288, "1m": 1048576, "2m": 2097152}
TRIAL_TOKENS = 167772160
MULTIPLIERS = (("lr_down", 1 / math.sqrt(2)), ("lr_base", 1.0), ("lr_up", math.sqrt(2)))
ATTEMPTS_PER_FAMILY = 9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trial_config(reference: dict, family: str, batch: str, label: str, multiplier: float) -> dict:
    """Complete a 20%-stable/80%-decay schedule while changing only matrix LR."""
    config = copy.deepcopy(reference)
    training = config["training"]
    if config["experiment"]["arm"] != family or training["batch_tokens"] != BATCH_TOKENS[batch]:
        raise ValueError("Reference family or batch differs from the proposed trial")
    if training["seed"] != 0 or training["heldout_windows"] != 64:
        raise ValueError("The LR study requires seed 0 and its 64 reserved windows")
    full_tokens = training["tokens"]
    steps, remainder = divmod(TRIAL_TOKENS, training["batch_tokens"])
    stable, phase_remainder = divmod(steps, 5)
    if remainder or phase_remainder:
        raise ValueError("Trial budget must divide into the selected batch and schedule phases")
    training.update(steps=steps, tokens=TRIAL_TOKENS, chinchilla_multiplier=TRIAL_TOKENS / full_tokens)
    config["optimizer"]["lr"] *= multiplier
    config["schedule"].update(
        warmup_steps=0, stable_steps=stable, decay_steps=4 * stable, kind="linear", min_lr_ratio=0.0
    )
    config["runtime"]["eval_every"] = steps // 8
    config["experiment"].update(checkpoint_every=steps // 4, diagnostic_every=steps // 8)
    config["experiment"]["tuning"] = {
        "family": family,
        "batch": batch,
        "stage": "matrix_lr",
        "trial": label,
        "lr_multiplier": multiplier,
        "reference_matrix_lr": reference["optimizer"]["lr"],
        "trial_tokens": TRIAL_TOKENS,
        "selection_metric": "final heldout-mixture selection_loss",
        "reporting_only_metric": "c4_loss",
        "decay_policy": "Matrix and auxiliary decay coefficients unchanged; LR times decay is not held fixed.",
    }
    config.pop("reference_validation_loss", None)
    return config


def prepare(
    root: Path,
    source_receipt: Path,
    out: Path,
    batch: str,
    *,
    families: tuple[str, ...] = FAMILIES,
    multipliers: tuple[tuple[str, float], ...] = MULTIPLIERS,
) -> dict:
    """Bind prepared trial configs to existing source/data without launching."""
    frozen = json.loads(source_receipt.read_text())
    source = Path(frozen["candidate"])
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError("Tuning preparation must not write inside the frozen source")
    if any(sha256(source / name) != expected for name, expected in frozen["files_sha256"].items()):
        raise ValueError("Frozen candidate files changed; do not prepare from mixed identities")
    data_receipt = root / "data/experiment_receipt.json"
    if sha256(data_receipt) != frozen["data_receipt_sha256"]:
        raise ValueError("Prepared-data receipt differs from the source binding")
    payloads = {}
    trials = []
    for label, multiplier in multipliers:
        for family in families:
            relative = f"experiment/configs/{family}_{batch}_seed0.json"
            if relative not in frozen["files_sha256"]:
                raise ValueError(f"Reference configuration is not bound: {relative}")
            reference = json.loads((source / relative).read_text())
            name = f"{family}_{label}"
            config = trial_config(reference, family, batch, label, multiplier)
            payload = json.dumps(config, indent=2, allow_nan=False) + "\n"
            path = out / "configs" / f"{name}.json"
            payloads[path] = payload
            trials.append(
                {
                    "name": name,
                    "family": family,
                    "batch": batch,
                    "seed": 0,
                    "lr_multiplier": multiplier,
                    "matrix_lr": config["optimizer"]["lr"],
                    "config": str(path.resolve()),
                    "config_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                    "reference_config": str(source / relative),
                    "reference_config_sha256": frozen["files_sha256"][relative],
                    "steps": config["training"]["steps"],
                    "tokens": config["training"]["tokens"],
                }
            )
    out.mkdir(parents=True, exist_ok=False)
    (out / "configs").mkdir()
    for path, payload in payloads.items():
        path.write_text(payload)
    result = {
        "time": datetime.now(UTC).isoformat(),
        "status": "prepared_unvalidated_not_queued",
        "source": str(source),
        "source_receipt": str(source_receipt.resolve()),
        "source_receipt_sha256": sha256(source_receipt),
        "data_receipt_sha256": sha256(data_receipt),
        "preparation_helper_sha256": sha256(Path(__file__)),
        "families": list(families),
        "batch": batch,
        "seed": 0,
        "trials": trials,
        "initial_trials_per_family": len(multipliers),
        "initial_tokens_per_family": len(multipliers) * TRIAL_TOKENS,
        "max_attempts_per_family": ATTEMPTS_PER_FAMILY,
        "max_tokens_per_family": ATTEMPTS_PER_FAMILY * TRIAL_TOKENS,
        "failed_attempts": "Retain actual identities and charge observed tokens; never silently retry.",
        "selection": {
            "metric": "selection_loss",
            "point": "completed short-budget endpoint",
            "holdout": "Seed-0 64 reserved windows per source, excluded by the pinned MarinMixture.",
            "weights": {"dclm": 3.8, "starcoder": 0.25, "proofpile": 0.055},
            "c4": "Reporting only; never used to rank trials or select LR.",
            "training_window_tokens": TRIAL_TOKENS // 5,
            "late_selection": "Secondary exact-token final-half mean; not the selection criterion.",
            "confirmation": "Freeze selected settings before independent full-budget paired fresh seeds.",
        },
        "launch_condition": (
            "Review wave5 numerical/loss/systems outcomes and choose the candidate backend; "
            "validate the actual short configs and source, then acquire a fresh verified allocation. "
            "No GPU job or queue is created by this preparation."
        ),
    }
    (out / "campaign.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", choices=BATCH_TOKENS, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.source_receipt, args.out, args.batch)
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt": str(args.out / "campaign.json"),
                "trials": len(result["trials"]),
                "initial_tokens_per_family": result["initial_tokens_per_family"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
