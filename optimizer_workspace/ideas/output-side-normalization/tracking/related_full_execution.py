# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run gated full-budget trials synchronously inside related_allocation."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import related_execution as shared
import related_full_campaign as full_campaign
from related_campaign import WORLD, read, sha256


def validate_trials(root: Path, trial_names: list[str], validation_receipts: list[Path]) -> dict:
    """Bind fresh-seed full trials to passing exact four-config validation waves."""
    root = root.resolve()
    campaign = full_campaign._bound_campaign(root)
    source = Path(campaign["source"])
    cluster = read(source / "experiment/cluster.json")
    if (
        cluster["environment"] != shared.ENVIRONMENT
        or cluster["python"] != shared.PYTHON
        or cluster["gpus_per_run"] != WORLD
    ):
        raise ValueError("Campaign cluster runtime differs from start_local")
    if not trial_names or len(set(trial_names)) != len(trial_names):
        raise ValueError("Select distinct full campaign trials")
    trials = {trial["name"]: trial for trial in campaign["trials"]}
    if any(name not in trials for name in trial_names):
        raise ValueError("Unknown full campaign trial")
    shared.require_receipt(Path(campaign["data"]), Path(campaign["data_receipt"]))
    validations = [shared.require_validation(campaign, path.resolve()) for path in validation_receipts]
    selected = []
    for name in trial_names:
        coverage = next((validation for validation in validations if name in validation["aliases"]), None)
        if coverage is None:
            raise ValueError(f"No passing exact-config validation covers actual full trial: {name}")
        selected.append(
            {
                **trials[name],
                "config": coverage["aliases"][name]["config"],
                "validation_receipt": coverage["path"],
                "validation_receipt_sha256": coverage["sha256"],
                "validation_wave": coverage["wave_name"],
            }
        )
    return {
        "campaign": str(full_campaign.campaign_path(root)),
        "campaign_sha256": sha256(full_campaign.campaign_path(root)),
        "source": campaign["source"],
        "source_sha256": campaign["source_sha256"],
        "cluster_sha256": sha256(source / "experiment/cluster.json"),
        "data": campaign["data"],
        "data_receipt": campaign["data_receipt"],
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "environment": shared.ENVIRONMENT,
        "validations": validations,
        "trials": selected,
    }


def run_trials(root: Path, trial_names: list[str], validation_receipts: list[Path], state_path: Path) -> dict:
    """Join each fresh full-budget child; the parent wrapper owns restoration."""
    marker = os.environ.get(shared.MARKER)
    if not marker or any(os.environ.get(key) != value for key, value in shared.ENVIRONMENT.items()):
        raise RuntimeError("Execution requires the allocation wrapper's inherited marker and environment")
    plan = validate_trials(root, trial_names, validation_receipts)
    shared.unused(state_path)
    state_path = state_path.resolve()
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    shared.unused(state_path)
    shared.unused(temporary)
    for trial in plan["trials"]:
        out = Path(trial["out"])
        for artifact in (out, out.with_name(f"{out.name}.torchrun.log")):
            if any(path.is_relative_to(artifact) or artifact.is_relative_to(path) for path in (state_path, temporary)):
                raise ValueError("Worker state and trial output/log paths must be disjoint")
            shared.unused(artifact)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "time": shared.now(),
        "status": "starting",
        "pid": os.getpid(),
        "allocation_marker": marker,
        "worker_sha256": sha256(Path(__file__)),
        "plan": plan,
        "children": [],
        "completed_trials": 0,
        "allocation_restoration": "owned_by_parent_wrapper",
    }
    with state_path.open("x") as stream:
        stream.write(json.dumps(state, indent=2, allow_nan=False) + "\n")
    try:
        for trial in plan["trials"]:
            if validate_trials(root, trial_names, validation_receipts) != plan:
                raise ValueError("Full campaign source or validation changed before child startup")
            out = Path(trial["out"])
            shared.unused(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            log = out.with_name(f"{out.name}.torchrun.log")
            child_record = {
                "name": trial["name"],
                "status": "starting",
                "time": shared.now(),
                "command": shared.trial_command(
                    Path(trial["config"]), Path(plan["data"]), Path(plan["data_receipt"]), out
                ),
                "cwd": plan["source"],
                "environment": shared.ENVIRONMENT,
                "allocation_marker": marker,
                "out": str(out),
                "log": str(log),
            }
            state["children"].append(child_record)
            state["status"] = "running"
            shared.write(state_path, state)
            with log.open("x") as stream:
                child = subprocess.Popen(
                    child_record["command"],
                    cwd=plan["source"],
                    env=os.environ | shared.ENVIRONMENT,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                child_record.update(status="running", pid=child.pid)
                shared.write(state_path, state)
                child_record["exit_code"] = child.wait()
            child_record.update(time=shared.now(), status="joined", log_sha256=sha256(log))
            shared.write(state_path, state)
            if child_record["exit_code"] != 0:
                raise RuntimeError(f"Trial exited {child_record['exit_code']}; inspect {log}")
            if validate_trials(root, trial_names, validation_receipts) != plan:
                raise ValueError("Full campaign source or validation changed during the trial")
            campaign = full_campaign._bound_campaign(root.resolve())
            original = next(item for item in campaign["trials"] if item["name"] == trial["name"])
            completion = full_campaign._trial_status(campaign, original)
            child_record["completion"] = completion
            if (
                not completion["complete"]
                or completion["planned_steps"] != full_campaign.STEPS
                or completion["observed_steps"] != full_campaign.STEPS
            ):
                raise ValueError(f"Child exited zero without verified full-budget completion: {trial['name']}")
            child_record["status"] = "complete"
            state["completed_trials"] += 1
            shared.write(state_path, state)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        if state["children"]:
            state["children"][-1]["status"] = "needs_attention"
            log = Path(state["children"][-1]["log"])
            if log.is_file():
                state["children"][-1]["log_sha256"] = sha256(log)
        raise
    finally:
        state["time"] = shared.now()
        shared.write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trial", action="append", required=True)
    parser.add_argument("--validation-receipt", type=Path, action="append", required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.state is None:
        parser.error("Synchronous execution requires an exclusive --state")
    result = (
        validate_trials(args.root, args.trial, args.validation_receipt)
        if args.check_only
        else run_trials(args.root, args.trial, args.validation_receipt, args.state)
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
