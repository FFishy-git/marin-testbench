# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist full related-campaign progress and read-only allocation observations."""

import argparse
import fcntl
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import monitor_related as shared
import related_full_campaign as full_campaign

ERRORS = (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError)
TRIAL_FIELDS = (
    "name",
    "arm",
    "seed",
    "lr_index",
    "status",
    "observed_steps",
    "observed_tokens",
    "planned_steps",
    "complete",
    "reasons",
)
SCOPE = {
    "training": full_campaign.CAMPAIGN_NAME,
    "training_completion": "full_update_history_final_selection_eval_and_eight_rank_checkpoint_receipt_only",
    "allocation": "registered_generation_bound_owner_snapshots",
    "historical_owner_restoration": "diagnostic_only_not_full_campaign_completion",
    "needs_attention": "registered_allocations_retained_short_evidence_full_training_and_observation_errors",
    "quality_claim": "none",
}


def failure(error: Exception) -> dict:
    return {"status": "invalid_evidence", "error": f"{type(error).__name__}: {error}"}


def observe(root: Path, registrations: list[dict], *, inspect_node=shared.probe) -> dict:
    """Report full training separately from generation-bound owner diagnostics."""
    if any(item["node"] not in shared.NODES for item in registrations):
        raise ValueError("Monitor registration includes an unapproved node")
    result = {
        "time": datetime.now(UTC).isoformat(),
        "scope": dict(SCOPE),
        "nodes": {},
        "owners": [],
        "allocation_observation": {"status": "not_observed"},
        "short_training": {"status": "not_observed"},
        "training": {"status": "not_observed"},
        "training_complete": False,
        "historical_owner_snapshot": {
            "all_registered_owners_complete_and_restored": False,
            "scope": SCOPE["historical_owner_restoration"],
        },
        "needs_attention": False,
    }
    try:
        allocations = shared.observe(root, registrations, inspect_node=inspect_node)
        result.update(
            time=allocations["time"],
            nodes=allocations["nodes"],
            owners=allocations["owners"],
            allocation_observation={"status": "ok"},
            short_training=allocations["training"],
            needs_attention=allocations["needs_attention"],
        )
        result["historical_owner_snapshot"]["all_registered_owners_complete_and_restored"] = allocations[
            "all_registered_owners_complete_and_restored"
        ]
        if any(node["observation_status"] != "ok" for node in result["nodes"].values()):
            result["allocation_observation"] = {"status": "unavailable", "errors": "see_nodes"}
            result["needs_attention"] = True
    except ERRORS as error:
        result["allocation_observation"] = failure(error)
        result["needs_attention"] = True
    try:
        progress = full_campaign.status(root)
        json.dumps(progress, allow_nan=False)
        result["training"] = {key: value for key, value in progress.items() if key not in ("trials", "process_liveness")}
        result["training"]["trials"] = [
            {key: trial[key] for key in TRIAL_FIELDS}
            | {
                "latest_train_loss": (trial["latest_metrics"] or {}).get("train_loss"),
                "latest_selection_loss": (trial["latest_metrics"] or {}).get("selection_loss"),
                "final_selection_loss": trial["final_selection_loss"],
                "final_c4_loss_reporting_only": trial["final_c4_loss_reporting_only"],
            }
            for trial in progress["trials"]
        ]
        result["training_complete"] = progress["status"] == "complete"
        if progress["status"] in ("needs_attention", "invalid_evidence", "blocked"):
            result["needs_attention"] = True
    except ERRORS as error:
        result["training"] = failure(error)
        result["needs_attention"] = True
    return result


def run_monitor(
    root: Path,
    registrations: Path,
    out: Path,
    *,
    interval: float = 60,
    once: bool = False,
    inspect_node=shared.probe,
) -> dict:
    """Write locked snapshots and an append-only observation history."""
    if interval < 5:
        raise ValueError("Monitor interval must be at least five seconds")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                entries = shared.read(registrations)
            except (OSError, ValueError) as error:
                result = observe(root, [], inspect_node=inspect_node)
                result["registrations"] = {"path": str(registrations), **failure(error)}
                result["needs_attention"] = True
            else:
                result = observe(root, entries, inspect_node=inspect_node)
                result["registrations"] = {"path": str(registrations), "status": "ok"}
            shared.write(out / "latest.json", result)
            with (out / "observations.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "time": result["time"],
                        "training": result["training"]["status"],
                        "training_complete": result["training_complete"],
                        "needs_attention": result["needs_attention"],
                        "owners": [
                            {"node": item["node"], "status": item["status"], "live": item.get("owner_liveness")}
                            for item in result["owners"]
                        ],
                    }
                ),
                flush=True,
            )
            if once:
                return result
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registrations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least five seconds")
    run_monitor(args.root, args.registrations, args.out, interval=args.interval, once=args.once)


if __name__ == "__main__":
    main()
