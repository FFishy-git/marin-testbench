# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind and join one node's exact full-wave validation and fresh full trials."""

import argparse
import json
import os
from pathlib import Path

import related_execution as shared
import related_full_campaign as campaign_module
import related_full_execution as execution
import related_sequence as sequence
from related_campaign import WORLD, read, sha256
from related_execution import now, unused, write

HELPERS = tuple(
    dict.fromkeys(
        (
            "related_full_sequence.py",
            "related_full_execution.py",
            *campaign_module.HELPERS,
            *sequence.HELPERS,
        )
    )
)
CONTROLLER_ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def stages_binding(campaign: dict, specifications: list[dict]) -> list[dict]:
    if len(specifications) != 2 or [spec["kind"] for spec in specifications] != ["validation", "full-trials"]:
        raise ValueError("A full queue requires one validation followed by its unique full trials")
    validation, trials = specifications
    wave = next((wave for wave in campaign["waves"] if wave["wave_name"] == validation["wave_name"]), None)
    if wave is None:
        raise ValueError("Validation wave is outside the full campaign")
    selected = wave["trials"] if wave["profile"] == "controlled" else wave["trials"][2:]
    receipt = sequence.absolute(validation["receipt"])
    if trials["trials"] != selected or [sequence.absolute(path) for path in trials["validation_receipts"]] != [receipt]:
        raise ValueError("Full trials must use the preceding exact wave: controlled four or paper last two")
    lookup = {trial["name"]: trial for trial in campaign["trials"]}
    aliases = {
        name: next(alias for alias in lookup[name]["aliases"] if alias["wave"] == wave["wave_name"])
        for name in wave["trials"]
    }
    work, worker = sequence.absolute(validation["work"]), sequence.absolute(trials["worker_state"])
    validation_state, validation_log = (
        sequence.absolute(validation[key]) for key in ("allocation_state", "allocation_log")
    )
    trials_state, trials_log = (sequence.absolute(trials[key]) for key in ("allocation_state", "allocation_log"))
    command = [
        shared.PYTHON,
        str(Path(execution.__file__).resolve()),
        "--root",
        campaign["root"],
        "--state",
        worker,
    ]
    for name in selected:
        command += ["--trial", name]
    command += ["--validation-receipt", receipt]
    outputs = [trials_state, trials_log, worker]
    for name in selected:
        out = lookup[name]["out"]
        outputs += [out, str(Path(out).with_name(f"{name}.torchrun.log"))]
    return [
        {
            "kind": "validation",
            "wave_name": wave["wave_name"],
            "work": work,
            "receipt": receipt,
            "config_sha256": {name: alias["config_sha256"] for name, alias in aliases.items()},
            "allocation_state": validation_state,
            "allocation_log": validation_log,
            "outputs": [validation_state, validation_log, work, receipt],
            "command": [
                shared.PYTHON,
                "-m",
                "experiment.validate",
                "--gpu",
                "--data",
                campaign["data"],
                "--data-receipt",
                campaign["data_receipt"],
                "--work",
                work,
                "--receipt",
                receipt,
                "--wave-manifest",
                wave["manifest"],
            ],
        },
        {
            "kind": "full-trials",
            "trials": list(selected),
            "validation_receipts": [receipt],
            "config_sha256": {name: aliases[name]["config_sha256"] for name in selected},
            "worker_state": worker,
            "allocation_state": trials_state,
            "allocation_log": trials_log,
            "outputs": outputs,
            "command": command,
        },
    ]


def binding(root: Path, specifications: list[dict], node: str) -> dict:
    campaign = campaign_module._bound_campaign(root.resolve())
    source = Path(campaign["source"])
    cluster = read(source / "experiment/cluster.json")
    if (
        source.resolve() != sequence.allocation.TASK.resolve()
        or shared.PYTHON != sequence.allocation.PYTHON
        or cluster["python"] != shared.PYTHON
        or cluster["environment"] != shared.ENVIRONMENT
        or cluster["gpus_per_run"] != WORLD
        or node not in cluster["nodes"]
    ):
        raise ValueError("Full sequence must use its bound local wrapper source, node and cluster runtime")
    shared.require_receipt(Path(campaign["data"]), Path(campaign["data_receipt"]))
    trial_paths = []
    for trial in campaign["trials"]:
        out = Path(trial["out"])
        log = out.with_name(f"{trial['name']}.torchrun.log")
        trial_paths += [out, out.with_suffix(out.suffix + ".tmp"), log, log.with_suffix(log.suffix + ".tmp")]
    return {
        "schema_version": 1,
        "root": campaign["root"],
        "source": campaign["source"],
        "node": node,
        "campaign": str(campaign_module.campaign_path(root)),
        "campaign_sha256": sha256(campaign_module.campaign_path(root)),
        "files_sha256": {
            **campaign["files_sha256"],
            "experiment/cluster.json": sha256(source / "experiment/cluster.json"),
        },
        "helpers_sha256": {name: sha256(Path(__file__).parent / name) for name in HELPERS},
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "data_files": campaign["data_files"],
        "runs_directory": str(campaign_module.campaign_path(root).parent / "runs"),
        "trial_output_paths": {str(path): str(path.resolve()) for path in trial_paths},
        "controller_environment": CONTROLLER_ENVIRONMENT,
        "allocation_environment": sequence.allocation.ENVIRONMENT,
        "stages": stages_binding(campaign, specifications),
    }


def require_path_layout(plan: dict) -> None:
    sequence.require_path_layout(plan)
    paths = sequence.output_paths(plan)
    expanded = [item for path in paths for item in (path, path.with_suffix(path.suffix + ".tmp"))]
    if len(set(expanded)) != len(expanded):
        raise ValueError("Full sequence outputs overlap temporary evidence")
    protected = [Path(plan["binding"]["root"]) / "campaigns" / campaign_module.short_campaign.CAMPAIGN_NAME]
    protected += [Path(plan["binding"]["root"]) / "run-preparation" / "related-verified-cache"]
    for index, path in enumerate(expanded):
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in (*protected, *expanded[:index])):
            raise ValueError("Full sequence evidence overlaps short campaign, cache or another output")
    runs = Path(plan["binding"]["runs_directory"])
    if runs.is_symlink() or runs.resolve() != runs:
        raise ValueError("Full campaign runs parent must remain canonical without symlink redirection")
    canonical = {str(path): str(path.resolve()) for path in expanded}
    if canonical != plan["resolved_outputs"]:
        raise ValueError("Full sequence output targets changed after preparation")
    trial_paths = plan["binding"]["trial_output_paths"]
    if any(Path(name).resolve() != Path(target) or Path(name) != Path(target) for name, target in trial_paths.items()):
        raise ValueError("Full campaign trial output, log or temporary target is redirected")
    protected += [
        Path(plan["binding"]["source"]),
        Path(plan["binding"]["campaign"]),
        Path(read(Path(plan["binding"]["campaign"]))["data"]),
        Path(plan["binding"]["root"]) / "data/experiment_receipt.json",
    ]
    protected = [path.resolve() for path in protected]
    candidates = list(dict.fromkeys((*canonical.values(), *trial_paths.values())))
    for index, target in enumerate(candidates):
        path = Path(target)
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in protected):
            raise ValueError("Canonical full sequence evidence overlaps protected artifacts")
        if any(path.is_relative_to(Path(other)) or Path(other).is_relative_to(path) for other in candidates[:index]):
            raise ValueError("Canonical full sequence evidence overlaps another output")


def prepare_plan(root: Path, stages: list[dict], plan_path: Path, *, node: str) -> dict:
    """Write a full queue only after the full campaign and all short endpoints bind."""
    unused(plan_path)
    for stage in stages:
        for key in ("allocation_state", "allocation_log", "work", "receipt", "worker_state"):
            if key in stage:
                unused(Path(stage[key]))
    plan_path = plan_path.resolve()
    plan = {
        "path": str(plan_path),
        "state": str(plan_path.with_suffix(".state.json")),
        "specifications": stages,
        "binding": binding(root, stages, node),
    }
    plan["resolved_outputs"] = {
        str(target): str(target.resolve())
        for path in sequence.output_paths(plan)
        for target in (path, path.with_suffix(path.suffix + ".tmp"))
    }
    require_path_layout(plan)
    for path in sequence.output_paths(plan):
        unused(path)
        unused(path.with_suffix(path.suffix + ".tmp"))
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("x") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    return plan


def require_plan(path: Path) -> dict:
    plan = read(path)
    bound = plan["binding"]
    if (
        plan["path"] != str(path.resolve())
        or plan["state"] != str(path.resolve().with_suffix(".state.json"))
        or binding(Path(bound["root"]), plan["specifications"], bound["node"]) != bound
    ):
        raise ValueError("Full sequence campaign, source, configs, helpers, data or plan changed")
    require_path_layout(plan)
    return plan


def require_worker(campaign: dict, stage: dict, trial_plan: dict) -> dict:
    worker = read(Path(stage["worker_state"]))
    if (
        worker.get("status") != "complete"
        or type(worker.get("completed_trials")) is not int
        or worker["completed_trials"] != len(stage["trials"])
        or worker.get("worker_sha256") != sha256(Path(execution.__file__))
        or worker.get("allocation_restoration") != "owned_by_parent_wrapper"
        or worker.get("plan") != trial_plan
        or [child["name"] for child in worker["children"]] != stage["trials"]
    ):
        raise ValueError("Full worker did not complete the exact selected trial plan")
    allocation = read(Path(stage["allocation_state"]))
    marker = allocation.get("child_marker")
    if not marker or worker.get("allocation_marker") != marker:
        raise ValueError("Full worker does not belong to the completed allocation owner")
    for child, trial in zip(worker["children"], trial_plan["trials"], strict=True):
        out = Path(trial["out"])
        log = out.with_name(f"{trial['name']}.torchrun.log")
        original = next(item for item in campaign["trials"] if item["name"] == trial["name"])
        completion = campaign_module._trial_status(campaign, original)
        if (
            child.get("status") != "complete"
            or type(child.get("exit_code")) is not int
            or child["exit_code"] != 0
            or child.get("allocation_marker") != marker
            or child.get("out") != str(out)
            or child.get("log") != str(log)
            or child.get("log_sha256") != sha256(log)
            or child.get("command")
            != shared.trial_command(Path(trial["config"]), Path(campaign["data"]), Path(campaign["data_receipt"]), out)
            or child.get("cwd") != campaign["source"]
            or child.get("environment") != shared.ENVIRONMENT
            or completion["complete"] is not True
            or completion["observed_steps"] != campaign_module.STEPS
            or child.get("completion") != completion
        ):
            raise ValueError("Full worker child or its full-budget completion artifacts differ")
    return worker


def run_sequence(plan_path: Path) -> dict:
    """Join validation and full trials through fresh wrappers, stopping on failure."""
    plan_path = plan_path.resolve()
    plan = require_plan(plan_path)
    if os.environ.get(sequence.allocation.MARKER) or sequence.allocation.local_node() != plan["binding"]["node"]:
        raise RuntimeError("Full sequence must start unowned on its exact planned local node")
    for path in sequence.output_paths(plan)[1:]:
        unused(path)
        unused(path.with_suffix(path.suffix + ".tmp"))
    state_path = Path(plan["state"])
    state = {"time": now(), "status": "starting", "plan_sha256": sha256(plan_path), "stages": [], "completed_stages": 0}
    write(state_path, state)
    try:
        for index, stage in enumerate(plan["binding"]["stages"]):
            if sha256(plan_path) != state["plan_sha256"] or require_plan(plan_path) != plan:
                raise ValueError("Full sequence binding changed before the next stage")
            for pending in plan["binding"]["stages"][index:]:
                for value in pending["outputs"]:
                    unused(Path(value))
                    unused(Path(value).with_suffix(Path(value).suffix + ".tmp"))
            campaign = campaign_module._bound_campaign(Path(plan["binding"]["root"]))
            trial_plan = None
            if index:
                previous = plan["binding"]["stages"][index - 1]
                sources = {
                    name: digest for name, digest in campaign["source_sha256"].items() if name.startswith("experiment/")
                }
                if sequence.require_restored(previous, sources) != state["stages"][-1]["allocation"]:
                    raise ValueError("Previous allocation restoration evidence changed")
                trial_plan = execution.validate_trials(
                    Path(campaign["root"]), stage["trials"], [Path(path) for path in stage["validation_receipts"]]
                )
                if trial_plan["validations"] != [state["stages"][-1]["validation"]]:
                    raise ValueError("Full-trial validation changed after restoration")
            record = {"kind": stage["kind"], "status": "running", "command": sequence.wrapper_command(stage)}
            state["stages"].append(record)
            state["status"] = "running"
            write(state_path, state)
            for key in ("allocation_state", "allocation_log"):
                Path(stage[key]).parent.mkdir(parents=True, exist_ok=True)
            result = sequence.run_stage(record["command"])
            record["exit_code"] = result.returncode
            for key in ("allocation_state", "allocation_log"):
                path = Path(stage[key])
                if path.is_file():
                    record[f"{key}_sha256"] = sha256(path)
            write(state_path, state)
            if result.returncode != 0:
                raise RuntimeError("Full owned stage exited nonzero; preserve evidence without retry")
            sources = {
                name: digest for name, digest in campaign["source_sha256"].items() if name.startswith("experiment/")
            }
            record["allocation"] = sequence.require_restored(stage, sources)
            if stage["kind"] == "validation":
                record["validation"] = shared.require_validation(campaign, Path(stage["receipt"]))
                if record["validation"]["wave_name"] != stage["wave_name"]:
                    raise ValueError("Executed validation belongs to a different full wave")
            else:
                if (
                    execution.validate_trials(
                        Path(campaign["root"]), stage["trials"], [Path(path) for path in stage["validation_receipts"]]
                    )
                    != trial_plan
                ):
                    raise ValueError("Full-trial validation changed during the worker")
                if sequence.require_restored(previous, sources) != state["stages"][-2]["allocation"]:
                    raise ValueError("First allocation restoration evidence changed during full trials")
                require_worker(campaign, stage, trial_plan)
                record["worker_state_sha256"] = sha256(Path(stage["worker_state"]))
            if sha256(plan_path) != state["plan_sha256"] or require_plan(plan_path) != plan:
                raise ValueError("Full sequence binding changed during the stage")
            record["status"] = "complete"
            state["completed_stages"] += 1
            write(state_path, state)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        if state["stages"]:
            state["stages"][-1]["status"] = "needs_attention"
        raise
    finally:
        state["time"] = now()
        write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, help="JSON list containing validation and full-trials specifications.")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--node")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.prepare is not None and (args.root is None or args.node is None):
        parser.error("--prepare requires --root and --node")
    if args.prepare is None and (args.root is not None or args.node is not None):
        parser.error("--root and --node belong to --prepare; existing plans retain their binding")
    os.environ.update(CONTROLLER_ENVIRONMENT)
    plan = (
        prepare_plan(args.root, read(args.prepare), args.plan, node=args.node)
        if args.prepare is not None
        else require_plan(args.plan)
    )
    print(
        json.dumps(
            {
                "status": "prepared" if args.prepare is not None else "verified",
                "plan": plan["path"],
                "plan_sha256": sha256(Path(plan["path"])),
                "node": plan["binding"]["node"],
                "launch_requested": args.launch,
                "launch_command": [shared.PYTHON, str(Path(__file__).resolve()), "--plan", plan["path"], "--launch"],
                "stage_commands": [sequence.wrapper_command(stage) for stage in plan["binding"]["stages"]],
                "controller_environment": CONTROLLER_ENVIRONMENT,
                "allocation_environment": sequence.allocation.ENVIRONMENT,
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    if args.launch:
        print(json.dumps(run_sequence(args.plan), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
