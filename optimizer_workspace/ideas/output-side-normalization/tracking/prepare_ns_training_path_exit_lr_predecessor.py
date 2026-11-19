# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect a disjoint exit-drain handoff; archive issuance awaits an explicit closure resolution."""

import argparse
import json
from pathlib import Path

import ns_training_path_exit_lr_predecessor as predecessor

PARENT = "run-preparation/ns-training-path-actual-lr-component-cpu/20261119T0926/receipt.json"
PARENT_SHA256 = "aca115258fca1cfb4adf4bbfe58b6606f1b63b022bf0d10525efcb3d9d7a4251"
NEW_HELPERS = (
    "ns_training_path_exit_lr_predecessor.py",
    "test_ns_training_path_exit_lr_predecessor.py",
    "prepare_ns_training_path_exit_lr_predecessor.py",
)
CPU_RECEIPT: str | None = None
CPU_RECEIPT_SHA256: str | None = None


def source_plan(root: Path) -> dict:
    """Describe separate source parents and conflicts without copying or replacing helpers."""
    predecessor.require_root(root)
    previous = predecessor.saved(root / PARENT, PARENT_SHA256)
    waiter = predecessor.waiter_cpu(root)
    old, fresh = previous["helpers_sha256"], waiter["helpers_sha256"]
    conflicts = {
        name: {"actual_lr_sha256": old[name], "exit_waiter_sha256": fresh[name]}
        for name in old.keys() & fresh.keys()
        if old[name] != fresh[name]
    }
    local = Path(__file__).resolve().parent
    return {
        "status": "exit_drain_actual_lr_predecessor_sources_distinct_archive_merge_deferred",
        "namespace": predecessor.NAMESPACE,
        "new_helpers_sha256": {name: predecessor.checked(local / name)[1]["sha256"] for name in NEW_HELPERS},
        "actual_lr_cpu_receipt": str(root / PARENT),
        "actual_lr_cpu_receipt_sha256": PARENT_SHA256,
        "waiter_cpu_receipt": str(root / predecessor.FOLLOWUP_CPU),
        "waiter_cpu_receipt_sha256": predecessor.FOLLOWUP_CPU_SHA256,
        "conflicting_helpers": conflicts,
        "execution_closure_merged": False,
        "gpu_execution_authorized": False,
        "scope": predecessor.SCOPE,
    }


def require_cpu(root: Path, path: Path) -> dict:
    """Refuse issuance until a separately reviewed complete source closure has acceptance."""
    predecessor.require_root(root)
    if CPU_RECEIPT is None or CPU_RECEIPT_SHA256 is None:
        raise RuntimeError("Exit-drain actual-LR predecessor CPU archive merge/acceptance is deferred")
    # A new acceptance verifier is required when resolving the closure, not merely new constants.
    raise RuntimeError("Exit-drain actual-LR predecessor resolved-closure verifier has not been supplied")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--phase", required=True, choices=("sources", "cpu", "observe", "evidence", "seal", "check-seal")
    )
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    predecessor.require_root(root)
    if args.phase == "sources":
        result = source_plan(root)
    elif args.phase == "cpu":
        raise RuntimeError("Exit-drain predecessor archive merging is deferred; no output was issued")
    elif args.phase == "observe":
        observed = predecessor.readiness(root)
        result = {key: observed[key] for key in ("status", "ready", "marked_pids")}
        result["owners"] = {name: item["status"] for name, item in observed["owners"].items()}
    elif args.phase == "evidence":
        record = predecessor.evidence(root)
        result = {key: record[key] for key in ("status", "original_terminal_sha256", "original_decisions")}
    else:
        if args.cpu_receipt is None:
            parser.error("Seal phases require a distinct accepted --cpu-receipt")
        require_cpu(root, args.cpu_receipt)
        if args.phase == "seal":
            result = predecessor.seal(root)
        else:
            if args.seal_sha256 is None:
                parser.error("Seal checking requires --seal-sha256")
            result = predecessor.require_seal(root, args.seal_sha256)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
