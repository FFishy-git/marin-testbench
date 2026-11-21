# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze selected short-study LRs for paired fresh-seed full-budget confirmation.

Only prepare writes configs. Reporting verifies histories and checkpoint receipts,
not checkpoint tensors, process liveness, accelerator acceptance or restoration.
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import related_campaign as short_campaign
from analyze_tuning import normalized_sources
from compare_runs import compare, completion, history, runtime, window_means
from related_campaign import ARMS, BATCH_TOKENS, TASK, WORLD, config_identity, read, sha256

sys.path.insert(0, str(TASK))

from experiment.related_configure import comparison_configuration

CAMPAIGN_NAME = "related-selected-full-seed12-20261121"
CONFIG_DIRECTORY = "experiment/comparisons/related_selected_full_seed12"
SEEDS = (1, 2)
STEPS = 5120
TOKENS = STEPS * BATCH_TOKENS
WINDOW_TOKENS = TOKENS // 5
PROFILES = {profile: arms for profile, (_, arms) in short_campaign.PROFILES.items()}
HELPERS = (*short_campaign.HELPERS, "related_full_campaign.py")
RECIPE_MODULES = (
    "experiment/related_configure.py",
    "experiment/configure.py",
    "experiment/reference.py",
    "experiment/factors.py",
    "experiment/go_muon.py",
    "experiment/mousse.py",
)
IMPORTED_RECIPE_SHA256 = {name: sha256(TASK / name) for name in RECIPE_MODULES}


def campaign_path(root: Path) -> Path:
    return root.resolve() / "campaigns" / CAMPAIGN_NAME / "campaign.json"


def _short_binding(root: Path) -> tuple[dict, dict]:
    analysis = short_campaign.analyze(root)
    if (
        analysis["status"] != "complete"
        or analysis["completed_trials"] != 18
        or set(analysis["completed_arm_lr_choices"]) != set(ARMS)
    ):
        raise ValueError("All eighteen verified short trials must complete before full confirmation")
    campaign = read(short_campaign.campaign_path(root))
    evidence = {}
    for trial in analysis["trials"]:
        directory = Path(trial["out"])
        checkpoint = Path(trial["final_checkpoint"])
        evidence[trial["name"]] = {
            "manifest": str(directory / "run.json"),
            "manifest_sha256": trial["run_manifest_sha256"],
            "history_sha256": {str(directory / name): digest for name, digest in trial["history_sha256"].items()},
            "checkpoint_receipt": str(checkpoint / "complete.json"),
            "checkpoint_receipt_sha256": trial["checkpoint_receipt_sha256"],
            "checkpoint_ranks": {
                str(checkpoint / f"rank_{rank}.pt"): {
                    "bytes": (checkpoint / f"rank_{rank}.pt").stat().st_size,
                    "mtime_ns": (checkpoint / f"rank_{rank}.pt").stat().st_mtime_ns,
                }
                for rank in range(WORLD)
            },
            "final_selection_loss": trial["final_selection_loss"],
        }
    return campaign, {
        "campaign": analysis["campaign"],
        "campaign_sha256": analysis["campaign_sha256"],
        "choices": analysis["completed_arm_lr_choices"],
        "evidence": evidence,
        "checkpoint_binding": "receipt hash and rank file size/mtime; no tensor reconstruction or content hash",
    }


def _recipe_binding(source: Path) -> dict:
    bound = {}
    for name in RECIPE_MODULES:
        digest = sha256(source / name)
        if digest != IMPORTED_RECIPE_SHA256[name] or digest != sha256(TASK / name):
            raise ValueError(f"Imported recipe code differs from the selected source: {name}")
        bound[name] = digest
    upstream_manifest = read(source / "experiment/upstream_manifest.json")
    prefix = f"artifacts/external-sources/Toner-{upstream_manifest['revision']}"
    for family in ("muon", "muont"):
        relative = f"reproduce/marin/configs/130m/{family}_0p5m.json"
        name = f"{prefix}/{relative}"
        digest = sha256(source / name)
        if digest != upstream_manifest["files"][relative] or digest != sha256(TASK / name):
            raise ValueError(f"Pinned full-budget recipe differs: {name}")
        bound[name] = digest
    return bound


def _data_binding(short: dict) -> dict:
    """Check the content receipt's file metadata without rereading the full dataset."""
    root = Path(short["data"]).resolve()
    receipt = read(Path(short["data_receipt"]))
    files = {}
    for record in receipt["files"]:
        path = (root / record["file"]).resolve()
        path.relative_to(root)
        stat = path.stat()
        if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"Prepared data changed since its content receipt: {path}")
        files[str(path)] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "receipt_sha256": record["sha256"],
        }
    if not files:
        raise ValueError("Prepared-data receipt has no file identities")
    return files


def _plan(root: Path, short: dict, selection: dict) -> tuple[list[dict], list[dict], dict[Path, str]]:
    source = Path(short["source"])
    configs = {}
    short_trials = {trial["name"]: trial for trial in short["trials"]}
    for seed in SEEDS:
        for arm in ARMS:
            choice = selection["choices"][arm]
            selected = short_trials[choice["trial"]]
            multiplier = selected["lr_multiplier"]
            microbatch = read(Path(selected["config"]))["training"]["micro_batch_size"]
            recipe_short = comparison_configuration(
                arm, batch="0p5m", seed=0, microbatch=microbatch, lr_multiplier=multiplier, stage="short"
            )
            if recipe_short != read(Path(selected["config"])):
                raise ValueError(f"Selected short configuration no longer matches the actual recipe: {arm}")
            config = comparison_configuration(
                arm, batch="0p5m", seed=seed, microbatch=microbatch, lr_multiplier=multiplier, stage="full"
            )
            if (
                config["training"]["steps"] != STEPS
                or config["training"]["tokens"] != TOKENS
                or config["training"]["batch_tokens"] != BATCH_TOKENS
                or config["training"]["world_size"] != WORLD
                or config["training"]["seed"] != seed
                or config["optimizer"]["lr"] != choice["matrix_lr"]
                or config["schedule"]
                != {
                    "warmup_steps": 0,
                    "stable_steps": 1024,
                    "decay_steps": 4096,
                    "kind": "linear",
                    "min_lr_ratio": 0.0,
                }
            ):
                raise ValueError(f"Selected full-budget recipe differs from the confirmation protocol: {arm}")
            configs[(seed, arm)] = config
    trials, waves, payloads = {}, [], {}
    for seed in SEEDS:
        for profile, arms in PROFILES.items():
            wave_name = f"related_{profile}_selected_full_seed{seed}"
            entries = []
            for arm in arms:
                choice = selection["choices"][arm]
                name = f"{arm}_0p5m_seed{seed}_lr{choice['lr_index']}"
                relative = f"{CONFIG_DIRECTORY}/{name}.json"
                path = source / relative
                payload = json.dumps(configs[(seed, arm)], indent=2, allow_nan=False) + "\n"
                digest = hashlib.sha256(payload.encode()).hexdigest()
                payloads[path] = payload
                alias = {"config": str(path), "config_sha256": digest, "wave": wave_name}
                if name in trials:
                    trials[name]["profiles"].append(profile)
                    trials[name]["aliases"].append(alias)
                else:
                    trials[name] = {
                        "name": name,
                        "arm": arm,
                        "family": arm,
                        "seed": seed,
                        "batch": "0p5m",
                        "lr_index": choice["lr_index"],
                        "lr_multiplier": short_trials[choice["trial"]]["lr_multiplier"],
                        "matrix_lr": choice["matrix_lr"],
                        "selected_short_trial": choice["trial"],
                        "config": str(path),
                        "config_sha256": digest,
                        "config_identity_sha256": config_identity(configs[(seed, arm)]),
                        "profiles": [profile],
                        "aliases": [alias],
                        "out": str(campaign_path(root).parent / "runs" / name),
                        "steps": STEPS,
                        "tokens": TOKENS,
                        "world_size": WORLD,
                    }
                entries.append({"arm": arm, "seed": seed, "batch": "0p5m", "name": name, "config": relative})
            manifest = source / CONFIG_DIRECTORY / f"wave_{profile}_seed{seed}.json"
            payload = json.dumps({"wave_name": wave_name, "wave": entries}, indent=2, allow_nan=False) + "\n"
            payloads[manifest] = payload
            waves.append(
                {
                    "wave_name": wave_name,
                    "profile": profile,
                    "seed": seed,
                    "manifest": str(manifest),
                    "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                    "trials": [entry["name"] for entry in entries],
                }
            )
    return list(trials.values()), waves, payloads


def _protocol() -> dict:
    return {
        "metric": "selection_loss",
        "lr_selection": "frozen_seed0_short_endpoint_choices",
        "seeds": list(SEEDS),
        "comparators": list(ARMS[:2]),
        "require_both_seeds_complete": True,
        "training_window_tokens": WINDOW_TOKENS,
        "c4": "reporting_only",
        "quality_claim": "none",
    }


def _unused(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Preserve existing campaign/config artifacts: {path}")


def prepare(root: Path) -> dict:
    """Write twelve full configs only after the entire bound short study passes."""
    root = root.resolve()
    _unused(campaign_path(root).parent)
    short, selection = _short_binding(root)
    source = Path(short["source"])
    directory = source / CONFIG_DIRECTORY
    _unused(directory)
    recipes = _recipe_binding(source)
    data_files = _data_binding(short)
    binding, executed = short_campaign._source_inventory(source)
    trials, waves, payloads = _plan(root, short, selection)
    result = {
        "schema_version": 1,
        "name": CAMPAIGN_NAME,
        "time": datetime.now(UTC).isoformat(),
        "status": "prepared_unvalidated_not_launched",
        "root": str(root),
        "source": str(source),
        "source_revision": short["source_revision"],
        "source_sha256": executed,
        "files_sha256": {
            **binding,
            **recipes,
            **{
                str(path.relative_to(source)): hashlib.sha256(payload.encode()).hexdigest()
                for path, payload in payloads.items()
            },
        },
        "helpers_sha256": {name: sha256(Path(__file__).parent / name) for name in HELPERS},
        "data": short["data"],
        "data_receipt": short["data_receipt"],
        "data_receipt_sha256": short["data_receipt_sha256"],
        "data_files": data_files,
        "short_confirmation": selection,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "trials": trials,
        "waves": waves,
        "unique_trials": 12,
        "trials_per_arm": 2,
        "tokens_per_arm": 2 * TOKENS,
        "selection": _protocol(),
        "limitations": (
            "Frozen short-study LRs; fresh seeds 1 and 2, not short-checkpoint resumes. Shared Muon/WT aliases "
            "run once per seed. Exact-config GPU validation and allocation remain parent-owned gates. "
            "Paired two-seed descriptive comparisons are not statistical inference or optimizer superiority. "
            "Checkpoint receipt/nonempty files do not prove tensors, cleanup or keeper restoration."
        ),
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=False)
    for path, payload in payloads.items():
        with path.open("x") as stream:
            stream.write(payload)
    campaign_path(root).parent.mkdir(parents=True, exist_ok=False)
    with campaign_path(root).open("x") as stream:
        stream.write(serialized)
    return result


def _bound_campaign(root: Path) -> dict:
    campaign = read(campaign_path(root))
    short, selection = _short_binding(root.resolve())
    source = Path(short["source"])
    recipes = _recipe_binding(source)
    data_files = _data_binding(short)
    binding, executed = short_campaign._source_inventory(source)
    trials, waves, payloads = _plan(root.resolve(), short, selection)
    expected_files = {
        **binding,
        **recipes,
        **{
            str(path.relative_to(source)): hashlib.sha256(payload.encode()).hexdigest()
            for path, payload in payloads.items()
        },
    }
    if (
        campaign["root"] != str(root.resolve())
        or campaign["name"] != CAMPAIGN_NAME
        or campaign["source"] != str(source)
        or campaign["source_revision"] != short["source_revision"]
        or campaign["source_sha256"] != executed
        or campaign["files_sha256"] != expected_files
        or campaign["short_confirmation"] != selection
        or campaign["trials"] != trials
        or campaign["waves"] != waves
        or campaign["arms"] != list(ARMS)
        or campaign["seeds"] != list(SEEDS)
        or campaign["unique_trials"] != 12
        or campaign["trials_per_arm"] != 2
        or campaign["tokens_per_arm"] != 2 * TOKENS
        or campaign["selection"] != _protocol()
        or campaign["helpers_sha256"] != {name: sha256(Path(__file__).parent / name) for name in HELPERS}
        or any(campaign[key] != short[key] for key in ("data", "data_receipt", "data_receipt_sha256"))
        or campaign["data_files"] != data_files
    ):
        raise ValueError("Full campaign differs from its frozen short choices/source/data/protocol")
    for name, digest in expected_files.items():
        if sha256(source / name) != digest:
            raise ValueError(f"Bound full source/config/manifest changed: {name}")
    return campaign


def _endpoint(directory: Path, run, trial: dict) -> dict:
    final = run.rows[-1]
    weights = read(Path(trial["config"]))["data"]["mixture_weights"]
    losses = [final["heldout_losses"][f"heldout/{name}"] for name in weights]
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in losses):
        raise ValueError("Invalid full-budget held-out evaluation")
    mixture = sum(weight * loss for weight, loss in zip(weights.values(), losses, strict=True)) / sum(weights.values())
    if mixture != final["selection_loss"]:
        raise ValueError("Final selection_loss differs from the reserved mixture")
    checkpoint = directory / "checkpoints" / f"step_{trial['steps']:06d}"
    receipt_path = checkpoint / "complete.json"
    receipt = read(receipt_path)
    if (
        type(receipt["step"]) is not int
        or type(receipt["world"]) is not int
        or receipt["step"] != trial["steps"]
        or receipt["world"] != trial["world_size"]
    ):
        raise ValueError("Full-budget checkpoint step or rank ownership differs")
    for rank in range(trial["world_size"]):
        path = checkpoint / f"rank_{rank}.pt"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing/empty full-budget rank checkpoint: {rank}")
    return {
        "final_selection_loss": final["selection_loss"],
        "final_c4_loss_reporting_only": final.get("c4_loss"),
        "final_checkpoint": str(checkpoint),
        "checkpoint_receipt_sha256": sha256(receipt_path),
    }


def _trial_status(campaign: dict, trial: dict) -> dict:
    directory = Path(trial["out"])
    item = {
        **{key: trial[key] for key in ("name", "arm", "family", "seed", "lr_index", "matrix_lr", "out")},
        "status": "not_started",
        "manifest_status": None,
        "process_liveness": "not_observed",
        "complete": False,
        "observed_steps": 0,
        "observed_tokens": 0,
        "planned_steps": trial["steps"],
        "final_selection_loss": None,
        "final_c4_loss_reporting_only": None,
        "latest_metrics": None,
        "training_windows": [],
        "runtime": None,
        "reasons": [],
    }
    if not (directory / "run.json").exists():
        if directory.exists():
            item.update(status="awaiting_manifest", reasons=["run_manifest_missing"])
        return item
    try:
        manifest = read(directory / "run.json")
        item["manifest_status"] = manifest["status"]
        source = Path(campaign["source"])
        if (
            manifest["config"] != read(Path(trial["config"]))
            or config_identity(manifest["config"]) != trial["config_identity_sha256"]
            or manifest["data_receipt_sha256"] != campaign["data_receipt_sha256"]
            or str(Path(manifest["data"]).resolve()) != campaign["data"]
            or manifest["world_size"] != trial["world_size"]
            or manifest["source_revision"] != campaign["source_revision"]
            or normalized_sources(source, manifest["source_sha256"])
            != normalized_sources(source, campaign["source_sha256"])
        ):
            raise ValueError("Full trial config/data/source/seed identity differs")
        if manifest.get("resume") and not Path(manifest["resume"]).resolve().is_relative_to(
            (directory / "checkpoints").resolve()
        ):
            raise ValueError("Fresh-seed confirmation cannot resume a short or other trial checkpoint")
        item["identity_match"] = True
        item["status"] = "running" if manifest["status"] == "running" else "incomplete"
        if manifest["status"] == "failed":
            item["status"] = "failed"
        segments = sorted(directory.glob("metrics_from_*.jsonl"))
        if not segments or all(segment.stat().st_size == 0 for segment in segments):
            item["reasons"] = ["history_missing"]
            if manifest["status"] == "complete":
                item["status"] = "invalid_evidence"
            return item
        run = history(directory)
        if run.manifest != manifest:
            raise ValueError("Full trial manifest changed during observation")
        json.dumps(run.rows, allow_nan=False)
        progress = completion(run)
        item.update(
            observed_steps=progress["observed_steps"],
            observed_tokens=progress["observed_tokens"],
            latest_metrics=run.rows[-1],
            reasons=progress["reasons"],
            training_windows=[
                {"end_tokens": end, "mean": mean}
                for end, mean in window_means(run.rows, campaign["selection"]["training_window_tokens"]).items()
            ],
            runtime=runtime(run, progress["observed_tokens"]),
        )
        if progress["complete"]:
            item.update(_endpoint(directory, run, trial))
            item.update(
                complete=True,
                status="complete",
                run_manifest_sha256=sha256(directory / "run.json"),
                history_sha256={segment.name: sha256(segment) for segment in segments},
            )
        elif manifest["status"] == "complete" or manifest.get("execution_check"):
            item["status"] = "invalid_evidence"
    except (OSError, ValueError, KeyError, TypeError) as error:
        item.update(complete=False, latest_metrics=None, reasons=[f"{type(error).__name__}: {error}"])
        item["status"] = (
            "running"
            if item["manifest_status"] == "running"
            and item.get("identity_match")
            and isinstance(error, json.JSONDecodeError)
            else "invalid_evidence"
        )
    return item


def status(root: Path) -> dict:
    """Report full-budget progress without altering files or observing processes."""
    if not campaign_path(root).exists():
        return {"status": "not_prepared", "campaign": str(campaign_path(root)), "planned_trials": 12, "trials": []}
    campaign = _bound_campaign(root)
    trials = [_trial_status(campaign, trial) for trial in campaign["trials"]]
    state = "incomplete"
    if all(trial["complete"] for trial in trials):
        state = "complete"
    elif any(trial["status"] in ("failed", "invalid_evidence") for trial in trials):
        state = "needs_attention"
    elif any(trial["status"] == "running" for trial in trials):
        state = "running"
    elif all(trial["status"] == "not_started" for trial in trials):
        state = "not_started"
    return {
        "time": datetime.now(UTC).isoformat(),
        "status": state,
        "campaign": str(campaign_path(root)),
        "campaign_sha256": sha256(campaign_path(root)),
        "quality_claim": "none",
        "process_liveness": "not_observed",
        "completed_trials": sum(trial["complete"] for trial in trials),
        "planned_trials": 12,
        "trials": trials,
        "observed_tokens_by_arm": {
            arm: sum(trial["observed_tokens"] for trial in trials if trial["arm"] == arm) for arm in ARMS
        },
    }


def analyze(root: Path) -> dict:
    """Expose paired descriptive deltas only after both seeds pass for each pair."""
    result = status(root)
    paired = {}
    histories = {}
    trials = {(trial["arm"], trial["seed"]): trial for trial in result["trials"]}
    for arm in ARMS:
        for baseline in ARMS[:2]:
            if arm == baseline or not all(
                trials.get((method, seed), {}).get("complete") for method in (arm, baseline) for seed in SEEDS
            ):
                continue
            per_seed = {}
            for seed in SEEDS:
                base = trials[(baseline, seed)]
                actual = trials[(arm, seed)]
                for item in (base, actual):
                    if item["name"] in histories:
                        continue
                    directory = Path(item["out"])
                    run = history(directory)
                    if (
                        sha256(directory / "run.json") != item["run_manifest_sha256"]
                        or {segment.name: sha256(segment) for segment in sorted(directory.glob("metrics_from_*.jsonl"))}
                        != item["history_sha256"]
                        or not completion(run)["complete"]
                    ):
                        raise ValueError("Verified full history changed before paired comparison")
                    histories[item["name"]] = run
                comparison = compare(
                    {baseline: histories[base["name"]], arm: histories[actual["name"]]},
                    baseline,
                    WINDOW_TOKENS,
                )
                per_seed[str(seed)] = comparison["comparisons"][arm]
            paired[f"{arm}_minus_{baseline}"] = {
                "arm": arm,
                "baseline": baseline,
                "seeds": list(SEEDS),
                "per_seed": per_seed,
                "mean_final_selection_loss_delta": (
                    math.fsum(per_seed[str(seed)]["evaluations"]["selection_loss"]["final_delta"] for seed in SEEDS)
                    / len(SEEDS)
                ),
                "quality_claim": "none",
            }
    result["paired_comparisons"] = paired
    result["interpretation"] = (
        "Frozen seed-0 selected LRs, paired fresh seeds 1 and 2. Both seeds must complete for each reported pair. "
        "Descriptive final selection_loss deltas, not optimizer superiority or replicated statistical inference. "
        "C4 is reporting only; no full-run LR reselection. Cleanup, GPU correctness and restoration are separate."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "status", "analyze"), required=True)
    args = parser.parse_args()
    try:
        function = {"prepare": prepare, "status": status, "analyze": analyze}[args.phase]
        print(json.dumps(function(args.root), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "blocked", "error": f"{type(error).__name__}: {error}"}))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
