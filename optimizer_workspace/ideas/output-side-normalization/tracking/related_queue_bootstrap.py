# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebind a trusted accepted preflight, then join its local training sequence."""

import argparse
import json
import os
from pathlib import Path

import related_execution as execution
import related_sequence as sequence
import related_verified_cache as cache
from related_campaign import read, sha256


def prepare(accepted_path: Path, accepted_sha256: str, plan_path: Path) -> dict:
    if sha256(accepted_path) != accepted_sha256:
        raise ValueError("Accepted preflight changed")
    accepted = read(accepted_path)
    binding = accepted["binding"]
    source = Path(binding["source"])
    for name, digest in binding["files_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Accepted experiment source or configuration changed: {name}")
    if sha256(Path(binding["campaign"])) != binding["campaign_sha256"]:
        raise ValueError("Accepted campaign changed")
    for validation in binding["external_validations"].values():
        receipt_path = Path(validation["path"])
        if sha256(receipt_path) != validation["sha256"]:
            raise ValueError("Accepted validation receipt changed")
        receipt = read(receipt_path)
        artifacts = validation["smoke_artifacts_sha256"]
        for name, expected in artifacts.items():
            if execution.smoke_hash(Path(name)) != expected:
                raise ValueError(f"Accepted checkpoint content changed: {name}")
        # These summaries came from the accepted preflight's actual tensor
        # comparisons; freshly verified content preserves that evidence.
        execution.VERIFIED_COMPARISONS.update(cache._comparison_seeds(receipt, artifacts))
        campaign = sequence.campaign_module._bound_campaign(Path(binding["root"]))
        current = execution.require_validation(campaign, receipt_path)
        if current != validation:
            raise ValueError("Current launch gates differ from the accepted preflight")
    if sha256(accepted_path) != accepted_sha256:
        raise ValueError("Accepted preflight changed during rebinding")
    return sequence.prepare_plan(
        Path(binding["root"]),
        accepted["specifications"],
        plan_path,
        node=binding["node"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--accepted-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    plan = prepare(args.accepted.resolve(), args.accepted_sha256, args.plan.resolve())
    print(json.dumps({"status": "rebound", "plan": plan["path"]}), flush=True)
    if args.launch:
        sequence.run_stage(
            [execution.PYTHON, str(Path(sequence.__file__).resolve()), "--plan", str(args.plan.resolve())]
        ).check_returncode()


if __name__ == "__main__":
    main()
