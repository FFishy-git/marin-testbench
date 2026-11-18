# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank complete equal-budget LR sweeps using reserved mixture loss, never C4."""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from compare_runs import completion, history, runtime, window_means
from prepare_tuning import FAMILIES, MULTIPLIERS, TRIAL_TOKENS, sha256, trial_config


def normalized_sources(source: Path, inventory: dict) -> dict:
    """Resolve manifest symlink aliases without accepting sources outside the frozen root."""
    result = {}
    for name, digest in inventory.items():
        relative = str((source / name).resolve().relative_to(source.resolve()))
        if relative in result:
            raise ValueError(f"Duplicate executed source identity: {relative}")
        result[relative] = digest
    return result


def executed_sources(source: Path, frozen: dict) -> dict:
    """Reconstruct the complete trainer inventory from the frozen Toner manifest."""
    paths = normalized_sources(source, frozen["files_sha256"])
    upstream = json.loads((source / "experiment/upstream_manifest.json").read_text())
    prefix = source / "artifacts/external-sources" / f"Toner-{upstream['revision']}"
    names = [
        name for name in frozen["files_sha256"] if Path(name).parts[0] == "experiment" and Path(name).suffix == ".py"
    ]
    for name, digest in upstream["files"].items():
        path = Path(name)
        trainer_module = path.parent == Path("reproduce/marin")
        optimizer_module = path.is_relative_to("src/toner")
        if path.suffix != ".py" or not (trainer_module or optimizer_module):
            continue
        relative = str((prefix / path).resolve().relative_to(source.resolve()))
        if paths.get(relative) != digest:
            raise ValueError(f"Executed upstream source is not frozen: {relative}")
        names.append(str((prefix / path).relative_to(source)))
    return normalized_sources(source, {name: sha256(source / name) for name in names})


def analyze(
    campaign_path: Path,
    *,
    families: tuple[str, ...] = FAMILIES,
    multipliers: tuple[tuple[str, float], ...] = MULTIPLIERS,
) -> dict:
    """Expose incomplete trials and withhold family winners until the sweep completes."""
    campaign = json.loads(campaign_path.read_text())
    receipt = Path(campaign["source_receipt"])
    if sha256(receipt) != campaign["source_receipt_sha256"]:
        raise ValueError("Campaign source binding changed")
    frozen = json.loads(receipt.read_text())
    source = Path(campaign["source"])
    if source.resolve() != Path(frozen["candidate"]).resolve():
        raise ValueError("Campaign source root differs from the frozen candidate")
    if campaign["data_receipt_sha256"] != frozen["data_receipt_sha256"]:
        raise ValueError("Campaign data receipt differs from the frozen binding")
    if any(sha256(source / name) != digest for name, digest in frozen["files_sha256"].items()):
        raise ValueError("Frozen source or configuration changed")
    expected_sources = executed_sources(source, frozen)
    rows = []
    source_identity = None
    window = campaign["selection"]["training_window_tokens"]
    names = [trial["name"] for trial in campaign["trials"]]
    if len(names) != len(set(names)):
        raise ValueError("Campaign contains duplicate trial identities")
    protocol = {
        f"{family}_{label}": (family, label, multiplier) for label, multiplier in multipliers for family in families
    }
    if (
        set(names) != set(protocol)
        or tuple(campaign["families"]) != families
        or campaign["seed"] != 0
        or campaign["initial_trials_per_family"] != len(multipliers)
        or campaign["initial_tokens_per_family"] != len(multipliers) * TRIAL_TOKENS
        or window != TRIAL_TOKENS // 5
    ):
        raise ValueError("Campaign differs from the equal-budget fixed-holdout LR protocol")
    for trial in campaign["trials"]:
        family, label, multiplier = protocol[trial["name"]]
        reference_path = source / f"experiment/configs/{family}_{campaign['batch']}_seed0.json"
        if (
            trial["family"] != family
            or trial["batch"] != campaign["batch"]
            or trial["seed"] != 0
            or trial["lr_multiplier"] != multiplier
            or Path(trial["reference_config"]).resolve() != reference_path.resolve()
            or trial["reference_config_sha256"] != frozen["files_sha256"][str(reference_path.relative_to(source))]
        ):
            raise ValueError(f"Trial differs from its frozen LR protocol: {trial['name']}")
        reference = json.loads(reference_path.read_text())
        expected_config = trial_config(reference, family, campaign["batch"], label, multiplier)
        path = Path(trial["config"])
        if sha256(path) != trial["config_sha256"]:
            raise ValueError(f"Prepared trial config changed: {trial['name']}")
        config = json.loads(path.read_text())
        if (
            config != expected_config
            or trial["tokens"] != TRIAL_TOKENS
            or trial["steps"] != expected_config["training"]["steps"]
            or trial["matrix_lr"] != expected_config["optimizer"]["lr"]
            or campaign["selection"]["weights"] != reference["data"]["mixture_weights"]
        ):
            raise ValueError(f"Trial configuration differs from the permitted LR transformation: {trial['name']}")
        directory = campaign_path.parent / "runs" / trial["name"]
        item = {
            "name": trial["name"],
            "family": trial["family"],
            "lr_multiplier": trial["lr_multiplier"],
            "matrix_lr": trial["matrix_lr"],
            "status": "not_launched",
            "observed_tokens": 0,
            "complete": False,
            "final_selection_loss": None,
            "mean_late_selection_loss": None,
            "mean_late_c4_loss_reporting_only": None,
            "training_windows": [],
            "runtime": None,
        }
        if not (directory / "run.json").exists():
            rows.append(item)
            continue
        manifest = json.loads((directory / "run.json").read_text())
        if manifest["config"] != config or manifest["data_receipt_sha256"] != campaign["data_receipt_sha256"]:
            raise ValueError(f"Trial configuration or data identity mismatch: {trial['name']}")
        actual_sources = normalized_sources(source, manifest["source_sha256"])
        if actual_sources != expected_sources:
            raise ValueError(f"Trial executed source differs from the frozen source: {trial['name']}")
        if source_identity is not None and actual_sources != source_identity:
            raise ValueError("Compared trials did not use the same source identity")
        source_identity = actual_sources
        item["status"] = manifest["status"]
        segments = sorted(directory.glob("metrics_from_*.jsonl"))
        if not segments or all(segment.stat().st_size == 0 for segment in segments):
            if manifest["status"] == "complete":
                raise ValueError(f"Completed trial has no history: {trial['name']}")
            rows.append(item)
            continue
        run = history(directory)
        state = completion(run)
        item.update(
            complete=state["complete"],
            observed_tokens=state["observed_tokens"],
            training_windows=[
                {"end_tokens": endpoint, "mean": mean} for endpoint, mean in window_means(run.rows, window).items()
            ],
            runtime=runtime(run, state["observed_tokens"]),
        )
        if item["complete"]:
            final = run.rows[-1]
            weights = campaign["selection"]["weights"]
            if config["data"]["mixture_weights"] != weights:
                raise ValueError("Selection weights differ from the planned mixture")
            recomputed = sum(weight * final["heldout_losses"][f"heldout/{name}"] for name, weight in weights.items())
            recomputed /= sum(weights.values())
            if recomputed != final["selection_loss"]:
                raise ValueError(f"Final selection loss is inconsistent with held-out sources: {trial['name']}")
            checkpoint = directory / "checkpoints" / f"step_{trial['steps']:06d}"
            complete = json.loads((checkpoint / "complete.json").read_text())
            world = config["training"]["world_size"]
            if complete["step"] != trial["steps"] or complete["world"] != world:
                raise ValueError("Trial checkpoint receipt differs from the budget or rank ownership")
            if any((checkpoint / f"rank_{rank}.pt").stat().st_size == 0 for rank in range(world)):
                raise ValueError("Trial has an empty final rank checkpoint")
            item["final_selection_loss"] = final["selection_loss"]
            for metric, target in (
                ("selection_loss", "mean_late_selection_loss"),
                ("c4_loss", "mean_late_c4_loss_reporting_only"),
            ):
                late = [row[metric] for row in run.rows if row["tokens"] > trial["tokens"] // 2 and metric in row]
                item[target] = math.fsum(late) / len(late) if late else None
        rows.append(item)
    winners = {}
    for family in campaign["families"]:
        trials = [item for item in rows if item["family"] == family]
        if len(trials) != campaign["initial_trials_per_family"]:
            raise ValueError(f"Family {family} has unequal search allowance")
        if all(item["complete"] for item in trials):
            best = min(trials, key=lambda item: (item["final_selection_loss"], item["name"]))
            winners[family] = {
                "trial": best["name"],
                "matrix_lr": best["matrix_lr"],
                "final_selection_loss": best["final_selection_loss"],
            }
    return {
        "time": datetime.now(UTC).isoformat(),
        "status": "complete" if rows and all(item["complete"] for item in rows) else "incomplete",
        "campaign": str(campaign_path),
        "campaign_sha256": sha256(campaign_path),
        "trials": rows,
        "completed_family_lr_choices": winners,
        "observed_tokens_by_family": {
            family: sum(item["observed_tokens"] for item in rows if item["family"] == family)
            for family in campaign["families"]
        },
        "interpretation": (
            "Exploratory short-budget LR choices from completed family sweeps; no C4-based selection, "
            "optimizer superiority, or fresh-seed confirmation. Failed attempts are not winners."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.campaign)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
