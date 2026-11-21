# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and report the related-method LR campaign without executing jobs.

Completion checks JSONL coverage, final held-out evaluation and an eight-rank
checkpoint receipt. It does not observe process liveness, reconstruct checkpoint
tensors, validate GPU numerics, or authorize execution or allocation changes.
"""

import argparse
import copy
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from analyze_tuning import normalized_sources
from compare_runs import completion, history, runtime, window_means

TASK = Path(__file__).resolve().parents[1]
CAMPAIGN_NAME = "related-baselines-20261121"
ARMS = (
    "muon_control",
    "inner_outer_mlpdiag",
    "go_muon_control",
    "mousse_control",
    "go_muon_paper",
    "mousse_paper",
)
PROFILES = {
    "controlled": ("related_controlled_short", ARMS[:4]),
    "paper_adapters": ("related_paper_short", (*ARMS[:2], *ARMS[4:])),
}
LR_MULTIPLIERS = (1 / math.sqrt(2), 1.0, math.sqrt(2))
STEPS = 320
BATCH_TOKENS = 524288
TOKENS = STEPS * BATCH_TOKENS
WORLD = 8
MOUSSE_REFERENCE = "artifacts/external-sources/Mousse-d00c1bf17790fbe56424ee5567cce80d8e75f4b2/dion/dion/mousse.py"
MOUSSE_REFERENCE_SHA256 = "874f02c33531b26cfa0b5c6c1c798afe72ec9c63ce07f804ee6caea222ccd201"
HELPERS = ("related_campaign.py", "compare_runs.py", "analyze_tuning.py", "prepare_tuning.py")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def config_identity(config: dict) -> str:
    """Match experiment.records.checkpoint_identity without importing Torch."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def campaign_path(root: Path) -> Path:
    return root.resolve() / "campaigns" / CAMPAIGN_NAME / "campaign.json"


def _source_inventory(source: Path) -> tuple[dict, dict]:
    upstream_manifest = read(source / "experiment/upstream_manifest.json")
    upstream = source / "artifacts/external-sources" / f"Toner-{upstream_manifest['revision']}"
    paths = sorted((source / "experiment").rglob("*.py"))
    if not (source / "experiment/train.py").is_file():
        raise ValueError("Missing experiment trainer source")
    upstream_paths = sorted((upstream / "reproduce/marin").glob("*.py"))
    upstream_paths += sorted((upstream / "src/toner").rglob("*.py"))
    expected_upstream = {
        name
        for name in upstream_manifest["files"]
        if Path(name).suffix == ".py"
        and (Path(name).parent == Path("reproduce/marin") or Path(name).is_relative_to("src/toner"))
    }
    if {str(path.relative_to(upstream)) for path in upstream_paths} != expected_upstream or not expected_upstream:
        raise ValueError("Executed upstream source inventory differs from the pinned manifest")
    for path in [*upstream_paths, upstream / "reproduce/marin/data_plan.json"]:
        if sha256(path) != upstream_manifest["files"][str(path.relative_to(upstream))]:
            raise ValueError(f"Pinned upstream source changed: {path}")
    paths += upstream_paths
    executed = {str(path.relative_to(source)): sha256(path) for path in paths}
    binding = {
        **executed,
        "experiment/upstream_manifest.json": sha256(source / "experiment/upstream_manifest.json"),
        str((upstream / "reproduce/marin/data_plan.json").relative_to(source)): sha256(
            upstream / "reproduce/marin/data_plan.json"
        ),
        MOUSSE_REFERENCE: sha256(source / MOUSSE_REFERENCE),
    }
    if binding[MOUSSE_REFERENCE] != MOUSSE_REFERENCE_SHA256:
        raise ValueError("Pinned Mousse reference SHA256 mismatch")
    return binding, executed


def _trial_protocol(config: dict) -> dict:
    result = copy.deepcopy(config)
    del result["optimizer"]["lr"]
    del result["experiment"]["comparison"]["matrix_lr_multiplier"]
    return result


def _plan(root: Path, source: Path) -> tuple[list[dict], list[dict], dict]:
    trials = {}
    waves = []
    bound = {}
    references = {}
    common = None
    for profile, (folder, arms) in PROFILES.items():
        for lr_index, multiplier in enumerate(LR_MULTIPLIERS):
            relative = f"experiment/comparisons/{folder}/wave_seed0_lr{lr_index}.json"
            manifest = read(source / relative)
            wave_name = f"related_{profile}_short_seed0_lr{lr_index}"
            entries = manifest["wave"]
            if manifest["wave_name"] != wave_name or tuple(entry["arm"] for entry in entries) != arms:
                raise ValueError(f"Related wave differs from the six-method LR protocol: {relative}")
            bound[relative] = sha256(source / relative)
            names = []
            for entry in entries:
                arm = entry["arm"]
                name = f"{arm}_0p5m_seed0_lr{lr_index}"
                config_relative = f"experiment/comparisons/{folder}/{name}.json"
                if entry != {"arm": arm, "seed": 0, "batch": "0p5m", "name": name, "config": config_relative}:
                    raise ValueError(f"Related wave entry identity changed: {name}")
                path = source / config_relative
                config = read(path)
                digest = sha256(path)
                bound[config_relative] = digest
                training = config["training"]
                comparison = config["experiment"]["comparison"]
                if (
                    config["experiment"]["arm"] != arm
                    or training["seed"] != 0
                    or training["steps"] != STEPS
                    or training["tokens"] != TOKENS
                    or training["batch_tokens"] != BATCH_TOKENS
                    or training["world_size"] != WORLD
                    or training["heldout_windows"] != 64
                    or comparison["comparison"] != "related_baselines"
                    or comparison["stage"] != "short"
                    or comparison["matrix_lr_multiplier"] != multiplier
                    or config["experiment"]["matrix_decay"] != "isotropic"
                    or comparison["matrix_decay"] != "isotropic"
                    or config["experiment"]["checkpoint_every"] != STEPS
                    or config["schedule"]
                    != {
                        "warmup_steps": 0,
                        "stable_steps": 64,
                        "decay_steps": 256,
                        "kind": "linear",
                        "min_lr_ratio": 0.0,
                    }
                ):
                    raise ValueError(f"Related short-trial budget or protocol changed: {name}")
                protocol = _trial_protocol(config)
                if arm in references and protocol != references[arm]:
                    raise ValueError(f"LR trials change more than matrix LR: {arm}")
                references[arm] = protocol
                shared = {key: config[key] for key in ("model", "data", "training", "runtime", "schedule")}
                shared["optimizer"] = {
                    key: value for key, value in config["optimizer"].items() if key not in ("lr", "name")
                }
                if common is not None and shared != common:
                    raise ValueError(f"Related trials do not share architecture/data/auxiliary settings: {name}")
                common = shared
                matrix_lr = config["optimizer"]["lr"]
                if isinstance(matrix_lr, bool) or not isinstance(matrix_lr, (int, float)):
                    raise ValueError(f"Invalid matrix LR: {name}")
                if not math.isfinite(matrix_lr) or matrix_lr <= 0:
                    raise ValueError(f"Invalid matrix LR: {name}")
                alias = {"config": str(path), "config_sha256": digest, "wave": wave_name}
                if name in trials:
                    if arm not in ARMS[:2] or digest != trials[name]["config_sha256"]:
                        raise ValueError(f"Shared trial configs are not byte-identical: {name}")
                    trials[name]["profiles"].append(profile)
                    trials[name]["aliases"].append(alias)
                else:
                    trials[name] = {
                        "name": name,
                        "arm": arm,
                        "family": arm,
                        "seed": 0,
                        "batch": "0p5m",
                        "lr_index": lr_index,
                        "lr_multiplier": multiplier,
                        "matrix_lr": matrix_lr,
                        "config": str(path),
                        "config_sha256": digest,
                        "config_identity_sha256": config_identity(config),
                        "profiles": [profile],
                        "aliases": [alias],
                        "out": str(campaign_path(root).parent / "runs" / name),
                        "steps": STEPS,
                        "tokens": TOKENS,
                        "world_size": WORLD,
                    }
                names.append(name)
            waves.append(
                {
                    "wave_name": wave_name,
                    "profile": profile,
                    "lr_index": lr_index,
                    "manifest": str(source / relative),
                    "manifest_sha256": bound[relative],
                    "trials": names,
                }
            )
    ordered = [trials[f"{arm}_0p5m_seed0_lr{index}"] for index in range(3) for arm in ARMS]
    for arm in ARMS:
        base = trials[f"{arm}_0p5m_seed0_lr1"]["matrix_lr"]
        if any(
            not math.isclose(trials[f"{arm}_0p5m_seed0_lr{i}"]["matrix_lr"], base * scale, rel_tol=1e-12)
            for i, scale in enumerate(LR_MULTIPLIERS)
        ):
            raise ValueError(f"Matrix LR grid changed: {arm}")
    return ordered, waves, bound


def prepare(root: Path, *, source: Path = TASK) -> dict:
    """Bind eighteen existing trials in a new campaign without copying or launching."""
    root, source = root.resolve(), source.resolve()
    path = campaign_path(root)
    if path.parent.exists():
        raise FileExistsError(path.parent)
    trials, waves, configs = _plan(root, source)
    binding, executed = _source_inventory(source)
    data_receipt = root / "data/experiment_receipt.json"
    data = read(data_receipt)
    upstream = read(source / "experiment/upstream_manifest.json")
    plan = f"artifacts/external-sources/Toner-{upstream['revision']}/reproduce/marin/data_plan.json"
    if data["data_plan_sha256"] != binding[plan]:
        raise ValueError("Prepared-data receipt belongs to a different pinned plan")
    result = {
        "schema_version": 1,
        "name": CAMPAIGN_NAME,
        "time": datetime.now(UTC).isoformat(),
        "status": "prepared_unvalidated_not_launched",
        "root": str(root),
        "source": str(source),
        "source_revision": upstream["revision"],
        "files_sha256": {**binding, **configs},
        "source_sha256": executed,
        "helpers_sha256": {name: sha256(Path(__file__).parent / name) for name in HELPERS},
        "data": str(Path(data["root"]).resolve()),
        "data_receipt": str(data_receipt),
        "data_receipt_sha256": sha256(data_receipt),
        "arms": list(ARMS),
        "trials": trials,
        "waves": waves,
        "unique_trials": 18,
        "trials_per_arm": 3,
        "tokens_per_arm": 3 * TOKENS,
        "selection": {
            "metric": "selection_loss",
            "point": "completed_update_320",
            "require_all_three_complete": True,
            "weights": read(Path(trials[0]["config"]))["data"]["mixture_weights"],
            "c4": "reporting_only",
            "tie_break": "trial_name",
            "training_window_tokens": TOKENS // 5,
        },
        "limitations": (
            "Preparation is not GPU validation or launch approval. Shared Muon/WT trials run once, not once per "
            "manifest. Status reads local evidence without process liveness. Checkpoints are checked by receipt "
            "and nonempty rank files, not tensor reconstruction. No silent retries or fresh-seed claims."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _bound_campaign(root: Path) -> dict:
    campaign = read(campaign_path(root))
    source = Path(campaign["source"])
    if campaign["root"] != str(root.resolve()) or campaign["name"] != CAMPAIGN_NAME:
        raise ValueError("Campaign root or identity changed")
    for relative, digest in campaign["files_sha256"].items():
        if sha256(source / relative) != digest:
            raise ValueError(f"Bound source/config/manifest changed: {relative}")
    if campaign["helpers_sha256"] != {name: sha256(Path(__file__).parent / name) for name in HELPERS}:
        raise ValueError("Campaign reporting helpers changed")
    binding, executed = _source_inventory(source)
    trials, waves, configs = _plan(root.resolve(), source)
    upstream = read(source / "experiment/upstream_manifest.json")
    if (
        campaign["files_sha256"] != {**binding, **configs}
        or campaign["source_sha256"] != executed
        or campaign["source_revision"] != upstream["revision"]
        or campaign["trials"] != trials
        or campaign["waves"] != waves
        or campaign["arms"] != list(ARMS)
        or campaign["unique_trials"] != 18
        or campaign["trials_per_arm"] != 3
        or campaign["tokens_per_arm"] != 3 * TOKENS
        or campaign["selection"]
        != {
            "metric": "selection_loss",
            "point": "completed_update_320",
            "require_all_three_complete": True,
            "weights": read(Path(trials[0]["config"]))["data"]["mixture_weights"],
            "c4": "reporting_only",
            "tie_break": "trial_name",
            "training_window_tokens": TOKENS // 5,
        }
    ):
        raise ValueError("Campaign differs from the bound eighteen-trial protocol")
    receipt = root.resolve() / "data/experiment_receipt.json"
    if (
        campaign["data_receipt"] != str(receipt)
        or sha256(receipt) != campaign["data_receipt_sha256"]
        or str(Path(read(receipt)["root"]).resolve()) != campaign["data"]
    ):
        raise ValueError("Campaign prepared-data receipt changed")
    return campaign


def _final_evidence(directory: Path, final: dict, campaign: dict) -> dict:
    weights = campaign["selection"]["weights"]
    losses = [final["heldout_losses"][f"heldout/{name}"] for name in weights]
    for value in [final["selection_loss"], *losses]:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Invalid final held-out evaluation")
    expected = sum(weight * loss for weight, loss in zip(weights.values(), losses, strict=True)) / sum(weights.values())
    if expected != final["selection_loss"]:
        raise ValueError("Final selection_loss differs from the held-out mixture")
    checkpoint = directory / "checkpoints" / f"step_{STEPS:06d}"
    receipt_path = checkpoint / "complete.json"
    receipt = read(receipt_path)
    if type(receipt["step"]) is not int or type(receipt["world"]) is not int:
        raise ValueError("Invalid final checkpoint receipt")
    if receipt["step"] != STEPS or receipt["world"] != WORLD:
        raise ValueError("Final checkpoint budget or rank ownership differs")
    if any(not (checkpoint / f"rank_{rank}.pt").is_file() for rank in range(WORLD)):
        raise ValueError("Missing final rank checkpoint")
    if any((checkpoint / f"rank_{rank}.pt").stat().st_size == 0 for rank in range(WORLD)):
        raise ValueError("Empty final rank checkpoint")
    return {
        "final_selection_loss": final["selection_loss"],
        "final_c4_loss_reporting_only": final.get("c4_loss"),
        "final_checkpoint": str(checkpoint),
        "checkpoint_receipt_sha256": sha256(receipt_path),
    }


def _trial_status(campaign: dict, trial: dict) -> dict:
    directory = Path(trial["out"])
    item = {
        "name": trial["name"],
        "arm": trial["arm"],
        "family": trial["arm"],
        "lr_index": trial["lr_index"],
        "matrix_lr": trial["matrix_lr"],
        "out": str(directory),
        "status": "not_started",
        "manifest_status": None,
        "process_liveness": "not_observed",
        "complete": False,
        "observed_steps": 0,
        "observed_tokens": 0,
        "planned_steps": STEPS,
        "final_selection_loss": None,
        "final_c4_loss_reporting_only": None,
        "latest_metrics": None,
        "training_windows": [],
        "runtime": None,
        "reasons": [],
    }
    manifest_path = directory / "run.json"
    if not manifest_path.exists():
        if directory.exists():
            item.update(status="awaiting_manifest", reasons=["run_manifest_missing"])
        return item
    try:
        manifest = read(manifest_path)
        item["manifest_status"] = manifest["status"]
        item["run_manifest_sha256"] = sha256(manifest_path)
        source = Path(campaign["source"])
        config = read(Path(trial["config"]))
        if (
            manifest["config"] != config
            or config_identity(manifest["config"]) != trial["config_identity_sha256"]
            or manifest["data_receipt_sha256"] != campaign["data_receipt_sha256"]
            or str(Path(manifest["data"]).resolve()) != campaign["data"]
            or manifest["world_size"] != WORLD
            or manifest["source_revision"] != campaign["source_revision"]
            or normalized_sources(source, manifest["source_sha256"])
            != normalized_sources(source, campaign["source_sha256"])
        ):
            raise ValueError("Executed config/data/source/rank identity differs from the campaign")
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
            raise ValueError("Run manifest changed during the history observation")
        json.dumps(run.rows, allow_nan=False)
        state = completion(run)
        item.update(
            observed_steps=state["observed_steps"],
            observed_tokens=state["observed_tokens"],
            latest_metrics=run.rows[-1],
            reasons=state["reasons"],
            training_windows=[
                {"end_tokens": end, "mean": mean} for end, mean in window_means(run.rows, TOKENS // 5).items()
            ],
            runtime=runtime(run, state["observed_tokens"]),
        )
        if state["complete"]:
            item.update(_final_evidence(directory, run.rows[-1], campaign))
            item.update(
                status="complete",
                complete=True,
                history_sha256={segment.name: sha256(segment) for segment in segments},
            )
        elif manifest["status"] == "complete" or manifest.get("execution_check"):
            item["status"] = "invalid_evidence"
    except (OSError, ValueError, KeyError, TypeError) as error:
        item["complete"] = False
        item["latest_metrics"] = None
        item["reasons"] = [f"{type(error).__name__}: {error}"]
        # A running writer can expose an unfinished JSONL line. Never turn that
        # read into completion or silently accept a truncated completed history.
        item["status"] = (
            "running"
            if item["manifest_status"] == "running"
            and item.get("identity_match")
            and isinstance(error, json.JSONDecodeError)
            else "invalid_evidence"
        )
    return item


def status(root: Path) -> dict:
    """Report observed progress and verified completion without SSH or writes."""
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
        "process_liveness": "not_observed",
        "completed_trials": sum(trial["complete"] for trial in trials),
        "planned_trials": 18,
        "trials": trials,
        "observed_tokens_by_arm": {
            arm: sum(trial["observed_tokens"] for trial in trials if trial["arm"] == arm) for arm in ARMS
        },
    }


def analyze(root: Path) -> dict:
    """Select each arm's LR only after all three fully verified endpoints exist."""
    result = status(root)
    choices = {}
    for arm in ARMS:
        trials = [trial for trial in result["trials"] if trial["arm"] == arm]
        if all(trial["complete"] for trial in trials):
            best = min(trials, key=lambda trial: (trial["final_selection_loss"], trial["name"]))
            choices[arm] = {
                "trial": best["name"],
                "lr_index": best["lr_index"],
                "matrix_lr": best["matrix_lr"],
                "final_selection_loss": best["final_selection_loss"],
            }
    result["completed_arm_lr_choices"] = choices
    result["interpretation"] = (
        "Exploratory seed-0 short-budget LR choices, not optimizer superiority or fresh-seed confirmation. "
        "Only final selection_loss selects LR; C4, training loss and runtime are reporting only. "
        "Completion does not prove process cleanup, GPU acceptance or keeper restoration."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "status", "analyze"), required=True)
    parser.add_argument("--source", type=Path, default=TASK, help="Existing source tree, used only for preparation.")
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            result = prepare(args.root, source=args.source)
        else:
            result = status(args.root) if args.phase == "status" else analyze(args.root)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "blocked", "error": f"{type(error).__name__}: {error}"}))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
