# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and execute local short-campaign stages through fresh owned allocations."""

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

import related_allocation as allocation
import related_campaign as campaign_module
import related_execution as execution
from related_campaign import campaign_path, read, sha256
from related_execution import now, unused, write

HELPERS = (
    "related_sequence.py",
    "related_allocation.py",
    "owned_output_handoff.py",
    "related_execution.py",
    "related_verified_cache.py",
    *campaign_module.HELPERS,
)


def absolute(value: str) -> str:
    return str(Path(value).resolve())


def stages_binding(campaign: dict, specifications: list[dict]) -> tuple[list[dict], dict]:
    stages, produced, external, selected = [], {}, {}, set()
    trials = {trial["name"]: trial for trial in campaign["trials"]}
    waves = {wave["wave_name"]: wave for wave in campaign["waves"]}
    for spec in specifications:
        state, log = absolute(spec["allocation_state"]), absolute(spec["allocation_log"])
        if spec["kind"] == "validation":
            wave = waves[spec["wave_name"]]
            work, receipt = absolute(spec["work"]), absolute(spec["receipt"])
            if receipt in produced or receipt in external:
                raise ValueError("Validation output reuses an earlier receipt")
            command = [
                execution.PYTHON,
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
            ]
            stage = {
                "kind": "validation",
                "wave_name": wave["wave_name"],
                "work": work,
                "receipt": receipt,
                "config_sha256": {
                    name: next(
                        alias["config_sha256"] for alias in trials[name]["aliases"] if alias["wave"] == wave["wave_name"]
                    )
                    for name in wave["trials"]
                },
                "outputs": [state, log, work, receipt],
            }
            produced[receipt] = wave
        elif spec["kind"] == "short-trials":
            names = spec["trials"]
            if not names or len(set(names)) != len(names) or selected.intersection(names):
                raise ValueError("Trials must be unique across this short sequence")
            if any(name not in trials for name in names):
                raise ValueError("Sequence contains a trial outside the short campaign")
            selected.update(names)
            receipts = [absolute(path) for path in spec["validation_receipts"]]
            covered = set()
            for receipt in receipts:
                if receipt in produced:
                    covered.update(produced[receipt]["trials"])
                else:
                    validation = execution.require_validation(campaign, Path(receipt))
                    external[receipt] = validation
                    covered.update(validation["aliases"])
            if not set(names).issubset(covered):
                raise ValueError("Short trials lack exact-config validation coverage")
            worker_state = absolute(spec["worker_state"])
            command = [
                execution.PYTHON,
                str(Path(execution.__file__).resolve()),
                "--root",
                campaign["root"],
                "--state",
                worker_state,
            ]
            for name in names:
                command += ["--trial", name]
            for receipt in receipts:
                command += ["--validation-receipt", receipt]
            outputs = [state, log, worker_state]
            for name in names:
                out = trials[name]["out"]
                outputs += [out, str(Path(out).with_name(f"{name}.torchrun.log"))]
            stage = {
                "kind": "short-trials",
                "trials": names,
                "config_sha256": {name: trials[name]["config_sha256"] for name in names},
                "validation_receipts": receipts,
                "worker_state": worker_state,
                "outputs": outputs,
            }
        else:
            raise ValueError("Only validation and short-trials stages are released")
        stages.append({**stage, "allocation_state": state, "allocation_log": log, "command": command})
    if not stages:
        raise ValueError("A sequence requires at least one stage")
    return stages, external


def binding(root: Path, specifications: list[dict], node: str) -> dict:
    campaign = campaign_module._bound_campaign(root.resolve())
    source = Path(campaign["source"])
    cluster = read(source / "experiment/cluster.json")
    if (
        source.resolve() != allocation.TASK.resolve()
        or execution.PYTHON != allocation.PYTHON
        or cluster["python"] != execution.PYTHON
        or cluster["environment"] != execution.ENVIRONMENT
        or cluster["gpus_per_run"] != campaign_module.WORLD
        or node not in cluster["nodes"]
    ):
        raise ValueError("Sequence must use the local wrapper source and cluster runtime")
    execution.require_receipt(Path(campaign["data"]), Path(campaign["data_receipt"]))
    stages, external = stages_binding(campaign, specifications)
    return {
        "schema_version": 1,
        "root": campaign["root"],
        "source": campaign["source"],
        "node": node,
        "campaign": str(campaign_path(root)),
        "campaign_sha256": sha256(campaign_path(root)),
        "files_sha256": {
            **campaign["files_sha256"],
            "experiment/cluster.json": sha256(source / "experiment/cluster.json"),
        },
        "helpers_sha256": {name: sha256(Path(__file__).parent / name) for name in HELPERS},
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "external_validations": external,
        "stages": stages,
    }


def output_paths(plan: dict) -> list[Path]:
    paths = [Path(plan["path"]), Path(plan["state"])]
    for stage in plan["binding"]["stages"]:
        paths += [Path(value) for value in stage["outputs"]]
    return paths


def require_path_layout(plan: dict) -> None:
    paths = output_paths(plan)
    source = Path(plan["binding"]["source"]).resolve()
    protected = [source, Path(plan["binding"]["campaign"]), Path(read(Path(plan["binding"]["campaign"]))["data"])]
    for index, path in enumerate(paths):
        if any(path.is_relative_to(item) or item.is_relative_to(path) for item in protected):
            raise ValueError("Sequence outputs overlap source, campaign or prepared data")
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths[:index]):
            raise ValueError("Sequence output paths overlap")


def prepare_plan(root: Path, stages: list[dict], plan_path: Path, *, node: str) -> dict:
    """Bind parent-supplied stages without launching or releasing any allocation."""
    plan_path = plan_path.resolve()
    plan = {
        "path": str(plan_path),
        "state": str(plan_path.with_suffix(".state.json")),
        "specifications": stages,
        "binding": binding(root, stages, node),
    }
    require_path_layout(plan)
    for path in output_paths(plan):
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
        raise ValueError("Sequence source, configs, helpers, data or plan binding changed")
    require_path_layout(plan)
    return plan


def wrapper_command(stage: dict) -> list[str]:
    return [
        allocation.PYTHON,
        str(Path(allocation.__file__).resolve()),
        "--state",
        stage["allocation_state"],
        "--log",
        stage["allocation_log"],
        "--",
        *stage["command"],
    ]


def require_restored(stage: dict, sources: dict) -> dict:
    state = read(Path(stage["allocation_state"]))
    if (
        state.get("status") != "complete"
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state.get("command") != stage["command"]
        or state.get("log") != stage["allocation_log"]
        or state.get("source_sha256") != sources
        or state.get("keeper_release_complete") is not True
        or not state.get("keeper", {}).get("devices")
        or state.get("restored_keeper", {}).get("devices") != state["keeper"]["devices"]
        or sha256(Path(stage["allocation_log"])) != state.get("log_sha256")
    ):
        raise ValueError("Stage did not complete cleanup and exact allocation restoration")
    return state


def run_stage(command: list[str]) -> subprocess.CompletedProcess:
    """Forward cancellation to the restoration owner and join it without killing it."""
    previous = {sig: signal.signal(sig, allocation.interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    child = None
    try:
        with allocation.defer_signals() as pending:
            child = subprocess.Popen(command, cwd=allocation.TASK, stdin=subprocess.DEVNULL, start_new_session=True)
        if pending:
            raise KeyboardInterrupt("Sequence interrupted during allocation owner startup")
        return subprocess.CompletedProcess(command, child.wait())
    except BaseException:
        if child is not None:
            with allocation.defer_signals():
                if child.poll() is None:
                    handle = allocation.pidfd(child.pid)
                    try:
                        try:
                            signal.pidfd_send_signal(handle, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    finally:
                        os.close(handle)
                child.wait()
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_sequence(plan_path: Path) -> dict:
    """Run local wrappers synchronously, halting without retries on any failure."""
    plan_path = plan_path.resolve()
    plan = require_plan(plan_path)
    if os.environ.get(allocation.MARKER) or allocation.local_node() != plan["binding"]["node"]:
        raise RuntimeError("Sequence must start outside an owned child on its planned local node")
    for path in output_paths(plan)[1:]:
        unused(path)
        unused(path.with_suffix(path.suffix + ".tmp"))
    state_path = Path(plan["state"])
    state = {"time": now(), "status": "starting", "plan_sha256": sha256(plan_path), "stages": [], "completed_stages": 0}
    write(state_path, state)
    try:
        for index, stage in enumerate(plan["binding"]["stages"]):
            if sha256(plan_path) != state["plan_sha256"] or require_plan(plan_path) != plan:
                raise ValueError("Sequence binding changed before the next stage")
            for pending in plan["binding"]["stages"][index:]:
                for value in pending["outputs"]:
                    unused(Path(value))
            campaign = campaign_module._bound_campaign(Path(plan["binding"]["root"]))
            trial_plan = None
            if stage["kind"] == "short-trials":
                trial_plan = execution.validate_trials(
                    Path(campaign["root"]), stage["trials"], [Path(path) for path in stage["validation_receipts"]]
                )
            record = {"kind": stage["kind"], "status": "running", "command": wrapper_command(stage)}
            state["stages"].append(record)
            state["status"] = "running"
            write(state_path, state)
            for key in ("allocation_state", "allocation_log"):
                Path(stage[key]).parent.mkdir(parents=True, exist_ok=True)
            result = run_stage(record["command"])
            record["exit_code"] = result.returncode
            for key in ("allocation_state", "allocation_log"):
                path = Path(stage[key])
                if path.is_file():
                    record[f"{key}_sha256"] = sha256(path)
            write(state_path, state)
            if result.returncode != 0:
                raise RuntimeError("Owned stage exited nonzero; preserve evidence without retry")
            sources = {
                name: digest
                for name, digest in plan["binding"]["files_sha256"].items()
                if name.startswith("experiment/") and name.endswith(".py")
            }
            record["allocation"] = require_restored(stage, sources)
            if stage["kind"] == "validation":
                record["validation"] = execution.require_validation(campaign, Path(stage["receipt"]))
                if record["validation"]["wave_name"] != stage["wave_name"]:
                    raise ValueError("Executed validation belongs to a different planned wave")
            else:
                worker = read(Path(stage["worker_state"]))
                if (
                    worker.get("status") != "complete"
                    or worker.get("completed_trials") != len(stage["trials"])
                    or worker.get("plan") != trial_plan
                    or [child["name"] for child in worker["children"]] != stage["trials"]
                    or any(
                        child.get("status") != "complete" or child.get("completion", {}).get("complete") is not True
                        for child in worker["children"]
                    )
                ):
                    raise ValueError("Short worker did not verify every selected trial")
                for name in stage["trials"]:
                    trial = next(item for item in campaign["trials"] if item["name"] == name)
                    if not campaign_module._trial_status(campaign, trial)["complete"]:
                        raise ValueError("Short-trial completion artifacts are incomplete")
                record["worker_state_sha256"] = sha256(Path(stage["worker_state"]))
            if sha256(plan_path) != state["plan_sha256"] or require_plan(plan_path) != plan:
                raise ValueError("Sequence binding changed during the stage")
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
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = require_plan(args.plan) if args.check_only else run_sequence(args.plan)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
