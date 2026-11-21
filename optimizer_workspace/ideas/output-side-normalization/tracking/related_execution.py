# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run gated related-campaign trials synchronously inside related_allocation."""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import related_campaign as campaign_module
from related_allocation import MARKER
from related_campaign import TASK, WORLD, campaign_path, read, sha256
from related_verified_cache import cache_path, store, validation_cache

sys.path.insert(0, str(TASK))

from experiment.data_access import require_receipt
from experiment.launch import BLOCK_CHECKS, ENVIRONMENT, PYTHON, RELATED_CHECKS, continuation_steps
from experiment.records import compare_continuation
from experiment.related_baselines import RELATED_ARMS

BASE_CHECKS = (
    "cpu_numerics",
    "cpu_distributed_statistics",
    "model_parameter_count",
    "gpu_collector",
    "distributed_statistics",
    "fa2",
    "compiled_model",
    "real_data_four_arm_smoke",
    "checkpoint_continuation",
    "gpu_down_updates",
    "cached_root_continuation",
    "microbatch_partition",
)
REQUIRED_CHECKS = (*BASE_CHECKS, *BLOCK_CHECKS, *RELATED_CHECKS)
SYNTHETIC_CHECKS = (
    "gpu_collector",
    "distributed_statistics",
    "fa2",
    "gpu_down_updates",
    "cached_root_continuation",
    "microbatch_partition",
    "synthetic_compiled_body",
    *BLOCK_CHECKS,
)
VERIFIED_COMPARISONS = {}
ARTIFACT_HASHES = {}


def now() -> str:
    return datetime.now(UTC).isoformat()


def write(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def launcher(python: str, ranks: int) -> list[str]:
    return [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}", "-m"]


def trial_command(config: Path, data: Path, receipt: Path, out: Path) -> list[str]:
    """Use start_local's exact eight-rank command, without detaching the worker."""
    return [
        *launcher(PYTHON, WORLD),
        "experiment.train",
        "--config",
        str(config),
        "--data",
        str(data),
        "--data-receipt",
        str(receipt),
        "--out",
        str(out),
    ]


def comparison_passed(record: dict) -> bool:
    count, error = record.get("tensor_comparisons"), record.get("max_absolute_error")
    return type(count) is int and count > 0 and type(error) in (float, int) and math.isfinite(error) and error >= 0


def require_checkpoint(directory: Path, step: int) -> None:
    checkpoint = directory / "checkpoints" / f"step_{step:06d}"
    receipt = read(checkpoint / "complete.json")
    if type(receipt["step"]) is not int or type(receipt["world"]) is not int:
        raise ValueError("Invalid smoke checkpoint counters")
    if receipt["step"] != step or receipt["world"] != WORLD:
        raise ValueError("Smoke checkpoint step or ownership differs")
    for rank in range(WORLD):
        path = checkpoint / f"rank_{rank}.pt"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty smoke checkpoint: {path}")


def require_smoke_run(directory: Path, step: int, config: dict, campaign: dict) -> None:
    manifest = read(directory / "run.json")
    if (
        manifest["status"] != "execution_check_complete"
        or type(manifest["completed_steps"]) is not int
        or manifest["completed_steps"] != step
        or manifest.get("execution_check") is not True
        or manifest["config"] != config
        or manifest["world_size"] != WORLD
        or manifest["source_revision"] != campaign["source_revision"]
        or manifest["source_sha256"] != campaign["source_sha256"]
        or manifest["data_receipt_sha256"] != campaign["data_receipt_sha256"]
        or str(Path(manifest["data"]).resolve()) != campaign["data"]
    ):
        raise ValueError(f"Actual smoke manifest differs from validation: {directory}")
    require_checkpoint(directory, step)


def smoke_hash(path: Path) -> str:
    """Reuse a content hash only while the local file generation is unchanged."""
    before = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    cached = ARTIFACT_HASHES.get(str(path))
    if cached and cached[0] == identity:
        return cached[1]
    digest = sha256(path)
    after = path.stat()
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"Smoke artifact changed during content verification: {path}")
    ARTIFACT_HASHES[str(path)] = (identity, digest)
    return digest


def checkpoint_artifacts(directory: Path, step: int) -> dict:
    checkpoint = directory / "checkpoints" / f"step_{step:06d}"
    return {
        str(path): smoke_hash(path)
        for path in (checkpoint / "complete.json", *(checkpoint / f"rank_{rank}.pt" for rank in range(WORLD)))
    }


def verified_comparison(full: Path, resumed: Path, step: int, artifacts: dict, reported: dict) -> None:
    identity = (str(full), str(resumed), step, tuple(sorted(artifacts.items())))
    comparison = VERIFIED_COMPARISONS.get(identity)
    if comparison is None:
        comparison = compare_continuation(
            full / "checkpoints" / f"step_{step:06d}",
            resumed / "checkpoints" / f"step_{step:06d}",
            WORLD,
        )
        VERIFIED_COMPARISONS[identity] = comparison
    if comparison != reported:
        raise ValueError("Reported continuation differs from the actual all-rank checkpoint comparison")


def require_smoke(campaign: dict, trial: dict, alias: dict, smoke: dict, commands: list[list[str]]) -> dict:
    config_path = Path(alias["config"])
    config = read(config_path)
    complete, split = continuation_steps(config)
    if (
        type(smoke.get("completed_steps")) is not int
        or type(smoke.get("split_steps")) is not int
        or smoke["completed_steps"] != complete
        or smoke["split_steps"] != split
        or not comparison_passed(smoke.get("continuation", {}))
    ):
        raise ValueError(f"Refresh-crossing continuation missing: {trial['name']}")
    full = Path(smoke["out"]).resolve()
    if full.name != f"{trial['name']}_continuous":
        raise ValueError("Smoke output does not identify the selected trial")
    resumed = full.with_name(f"{trial['name']}_resumed")
    common = trial_command(config_path, Path(campaign["data"]), Path(campaign["data_receipt"]), full)[:-2]
    expected = [
        [*common, "--out", str(full), "--max-steps", str(complete)],
        [*common, "--out", str(resumed), "--max-steps", str(split)],
    ]
    artifacts = {}
    checkpoint_step = split
    if trial["arm"] in RELATED_ARMS and trial["arm"] != "muon_control":
        if not comparison_passed(smoke.get("pre_refresh_continuation") or {}):
            raise ValueError(f"Pre-refresh factor continuation missing: {trial['name']}")
        checkpoint_step = split + 1
        prefix = full.with_name(f"{trial['name']}_pre_refresh_continuous")
        expected += [
            [
                *common,
                "--out",
                str(resumed),
                "--max-steps",
                str(checkpoint_step),
                "--resume",
                str(resumed / "checkpoints" / f"step_{split:06d}"),
            ],
            [*common, "--out", str(prefix), "--max-steps", str(checkpoint_step)],
        ]
        require_smoke_run(prefix, checkpoint_step, config, campaign)
        require_checkpoint(resumed, checkpoint_step)
        artifacts.update(checkpoint_artifacts(prefix, checkpoint_step))
        artifacts.update(checkpoint_artifacts(resumed, checkpoint_step))
        artifacts[str(prefix / "run.json")] = smoke_hash(prefix / "run.json")
    expected.append(
        [
            *common,
            "--out",
            str(resumed),
            "--max-steps",
            str(complete),
            "--resume",
            str(resumed / "checkpoints" / f"step_{checkpoint_step:06d}"),
        ]
    )
    if any(command not in commands for command in expected):
        raise ValueError(f"Actual smoke commands do not cover the trial: {trial['name']}")
    require_smoke_run(full, complete, config, campaign)
    require_smoke_run(resumed, complete, config, campaign)
    require_checkpoint(resumed, split)
    artifacts.update(checkpoint_artifacts(full, complete))
    artifacts.update(checkpoint_artifacts(resumed, complete))
    artifacts.update(checkpoint_artifacts(resumed, split))
    for directory in (full, resumed):
        artifacts[str(directory / "run.json")] = smoke_hash(directory / "run.json")
    verified_comparison(full, resumed, complete, artifacts, smoke["continuation"])
    if trial["arm"] in RELATED_ARMS and trial["arm"] != "muon_control":
        verified_comparison(prefix, resumed, checkpoint_step, artifacts, smoke["pre_refresh_continuation"])
    if any(smoke_hash(Path(path)) != digest for path, digest in artifacts.items()):
        raise ValueError("Smoke artifacts changed during continuation verification")
    return artifacts


def require_validation(campaign: dict, path: Path) -> dict:
    cached = validation_cache(campaign, path)
    if cached is not None:
        ARTIFACT_HASHES.update(cached["artifact_hashes"])
        VERIFIED_COMPARISONS.update(cached["verified_comparisons"])
    receipt = read(path)
    wave = next((item for item in campaign["waves"] if item["wave_name"] == receipt.get("wave_name")), None)
    if wave is None:
        raise ValueError("Validation wave is not bound to this campaign")
    trials = {trial["name"]: trial for trial in campaign["trials"]}
    aliases = {
        name: next(alias for alias in trials[name]["aliases"] if alias["wave"] == wave["wave_name"])
        for name in wave["trials"]
    }
    expected = {name: alias["config_sha256"] for name, alias in aliases.items()}
    sources = {name: digest for name, digest in campaign["source_sha256"].items() if name.startswith("experiment/")}
    if (
        len(expected) != 4
        or receipt.get("config_sha256") != expected
        or receipt.get("source_sha256") != sources
        or receipt.get("environment") != ENVIRONMENT
        or receipt.get("data_receipt_sha256") != campaign["data_receipt_sha256"]
        or any(receipt.get("checks", {}).get(name) is not True for name in REQUIRED_CHECKS)
    ):
        raise ValueError("Validation source, selected-four configs, environment, data or gates differ")
    evidence = receipt.get("evidence", {})
    if type(evidence.get("cpu_test_exit_code")) is not int or evidence["cpu_test_exit_code"] != 0:
        raise ValueError("Validation CPU command did not pass")
    executions = evidence.get("gpu_commands")
    if not executions:
        raise ValueError("Validation has no actual GPU command evidence")
    commands, logs = [], {}
    for execution in executions:
        command = execution["command"]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(argument, str) for argument in command)
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or sha256(Path(execution["log"])) != execution["log_sha256"]
        ):
            raise ValueError("Validation GPU command failed or its log changed")
        commands.append(command)
        logs[execution["log"]] = execution["log_sha256"]
    probes = {}
    for key, module, checks, world_key in (
        ("gpu_receipt", "experiment.gpu_checks", SYNTHETIC_CHECKS, "world"),
        ("related_gpu_receipt", "experiment.related_gpu_checks", RELATED_CHECKS, "world_size"),
    ):
        probe_path = Path(evidence[key])
        probe = read(probe_path)
        if (
            [*launcher(PYTHON, 2), module, "--receipt", str(probe_path)] not in commands
            or probe.get("source_sha256") != sources
            or probe.get(world_key) != 2
            or any(probe.get("checks", {}).get(name) is not True for name in checks)
        ):
            raise ValueError(f"Source-bound accelerator receipt differs: {probe_path}")
        probes[str(probe_path)] = sha256(probe_path)
    smokes = evidence.get("real_data_smokes", {})
    if set(smokes) != set(expected):
        raise ValueError("Actual real-data smoke coverage is not the selected four")
    artifacts = {}
    for name, alias in aliases.items():
        artifacts.update(require_smoke(campaign, trials[name], alias, smokes[name], commands))
    result = {
        "path": str(path),
        "sha256": sha256(path),
        "wave_name": wave["wave_name"],
        "aliases": aliases,
        "logs_sha256": logs,
        "gpu_receipts_sha256": probes,
        "smoke_artifacts_sha256": artifacts,
    }
    if cached is None and Path(campaign["source"]).resolve() == TASK and not cache_path(campaign, path).exists():
        store(cache_path(campaign, path), result, receipt, ARTIFACT_HASHES)
    return result


def validate_trials(root: Path, trial_names: list[str], validation_receipts: list[Path]) -> dict:
    """Read-only preflight; each trial must have a passing exact bound wave alias."""
    campaign = campaign_module._bound_campaign(root.resolve())
    cluster = read(Path(campaign["source"]) / "experiment/cluster.json")
    if cluster["environment"] != ENVIRONMENT or cluster["python"] != PYTHON or cluster["gpus_per_run"] != WORLD:
        raise ValueError("Campaign cluster runtime differs from start_local")
    if not trial_names or len(set(trial_names)) != len(trial_names):
        raise ValueError("Select distinct campaign trials")
    trials = {trial["name"]: trial for trial in campaign["trials"]}
    if any(name not in trials for name in trial_names):
        raise ValueError("Unknown campaign trial")
    require_receipt(Path(campaign["data"]), Path(campaign["data_receipt"]))
    validations = [require_validation(campaign, path.resolve()) for path in validation_receipts]
    selected = []
    for name in trial_names:
        coverage = next((validation for validation in validations if name in validation["aliases"]), None)
        if coverage is None:
            raise ValueError(f"No passing exact-config validation covers actual trial: {name}")
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
        "campaign": str(campaign_path(root)),
        "campaign_sha256": sha256(campaign_path(root)),
        "source": campaign["source"],
        "source_sha256": campaign["source_sha256"],
        "cluster_sha256": sha256(Path(campaign["source"]) / "experiment/cluster.json"),
        "data": campaign["data"],
        "data_receipt": campaign["data_receipt"],
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "environment": ENVIRONMENT,
        "validations": validations,
        "trials": selected,
    }


def unused(path: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"Preserve existing execution evidence: {path}")


def run_trials(root: Path, trial_names: list[str], validation_receipts: list[Path], state_path: Path) -> dict:
    """Join each child locally; allocation cleanup/restoration belongs to the caller."""
    marker = os.environ.get(MARKER)
    if not marker or any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError("Execution requires the allocation wrapper's inherited marker and environment")
    plan = validate_trials(root, trial_names, validation_receipts)
    state_path = state_path.resolve()
    unused(state_path)
    unused(state_path.with_suffix(state_path.suffix + ".tmp"))
    for trial in plan["trials"]:
        out = Path(trial["out"])
        unused(out)
        unused(out.with_name(f"{out.name}.torchrun.log"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "time": now(),
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
                raise ValueError("Campaign source or validation changed before child startup")
            out = Path(trial["out"])
            unused(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            log = out.with_name(f"{out.name}.torchrun.log")
            child_record = {
                "name": trial["name"],
                "status": "starting",
                "time": now(),
                "command": trial_command(Path(trial["config"]), Path(plan["data"]), Path(plan["data_receipt"]), out),
                "cwd": plan["source"],
                "environment": ENVIRONMENT,
                "allocation_marker": marker,
                "out": str(out),
                "log": str(log),
            }
            state["children"].append(child_record)
            state["status"] = "running"
            write(state_path, state)
            with log.open("x") as stream:
                child = subprocess.Popen(
                    child_record["command"],
                    cwd=plan["source"],
                    env=os.environ | ENVIRONMENT,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                child_record.update(status="running", pid=child.pid)
                write(state_path, state)
                child_record["exit_code"] = child.wait()
            child_record.update(time=now(), status="joined", log_sha256=sha256(log))
            write(state_path, state)
            if child_record["exit_code"] != 0:
                raise RuntimeError(f"Trial exited {child_record['exit_code']}; inspect {log}")
            if validate_trials(root, trial_names, validation_receipts) != plan:
                raise ValueError("Campaign source or validation changed during the trial")
            campaign = campaign_module._bound_campaign(root.resolve())
            original = next(item for item in campaign["trials"] if item["name"] == trial["name"])
            completion = campaign_module._trial_status(campaign, original)
            child_record["completion"] = completion
            if not completion["complete"]:
                raise ValueError(f"Child exited zero without verified campaign completion: {trial['name']}")
            child_record["status"] = "complete"
            state["completed_trials"] += 1
            write(state_path, state)
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
        state["time"] = now()
        write(state_path, state)
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
