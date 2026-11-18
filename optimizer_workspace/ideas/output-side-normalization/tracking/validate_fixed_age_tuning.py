# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the twelve actual short-LR configurations without editing their frozen trainer."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from compare_runs import history
from fixed_age_protocol import analyze, environment, require_gpu_environment
from launch_wave5 import PYTHON, read, sha256
from prepare_fixed_age_tuning import require_source
from run_diagonal_diagnostic import PROCESS_MARKER
from watch_wave5 import write
from watch_wave7_release import observed_process, option

WORLD = 8
BOUNDARY = 22
ENDPOINT = 24
TRIAL_GATES = (
    "actual_short_config",
    "compiled_model",
    "cached_root_checkpoint",
    "boundary_continuation",
    "post_cached_root_continuation",
    "fixed_age_history",
)
FUSED_GATES = ("captured_diagonal_rows", "outer_diagonal_update_parity")


def plan(campaign_path: Path, work: Path) -> dict:
    """Describe all actual-config trajectories; a command plan is not execution evidence."""
    analyze(campaign_path)
    campaign = read(campaign_path)
    root = Path(campaign["root"])
    frozen = require_source(root, Path(campaign["source_receipt"]))
    source = Path(campaign["source"])
    if work.resolve().is_relative_to(source.resolve()):
        raise ValueError("Validation output must stay outside the frozen trainer")
    if work.resolve() != (campaign_path.parent / "validation").resolve():
        raise ValueError("Actual validation must use its separate canonical campaign directory")
    launcher = [PYTHON, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", "-m", "experiment.train"]
    trials = []
    for trial in campaign["trials"]:
        full = work / f"{trial['name']}_continuous"
        interrupted = work / f"{trial['name']}_resumed"
        common = [
            *launcher,
            "--config",
            trial["config"],
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--execution-boundary",
            str(BOUNDARY),
        ]
        if trial["family"].endswith("_fused"):
            common.extend(("--verify-diagonal-capture", "--verify-outer-diagonal-updates"))
        commands = [
            [*common, "--out", str(full), "--max-steps", str(ENDPOINT)],
            [*common, "--out", str(interrupted), "--max-steps", str(BOUNDARY)],
            [
                *common,
                "--out",
                str(interrupted),
                "--max-steps",
                str(ENDPOINT),
                "--resume",
                str(interrupted / f"checkpoints/step_{BOUNDARY:06d}"),
            ],
        ]
        gates = (*TRIAL_GATES, *FUSED_GATES) if trial["family"].endswith("_fused") else TRIAL_GATES
        trials.append(
            {
                "name": trial["name"],
                "family": trial["family"],
                "config": trial["config"],
                "config_sha256": trial["config_sha256"],
                "checks": dict.fromkeys(gates, False),
                "continuous": str(full),
                "resumed": str(interrupted),
                "commands": commands,
            }
        )
    return {
        "status": "planned_not_executed",
        "campaign": str(campaign_path.resolve()),
        "campaign_sha256": sha256(campaign_path),
        "validator_sha256": sha256(Path(__file__)),
        "source_release_sha256": campaign["source_receipt_sha256"],
        "source_sha256": frozen["source_sha256"],
        "data_receipt_sha256": campaign["data_receipt_sha256"],
        "environment": environment(campaign_path),
        "world": WORLD,
        "continuous_steps": ENDPOINT,
        "interrupted_steps": BOUNDARY,
        "resumed_steps": ENDPOINT,
        "continuation_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "trials": trials,
        "gpu_commands": [],
        "planned_validation_tokens": sum(
            2 * ENDPOINT * read(Path(trial["config"]))["training"]["batch_tokens"] for trial in campaign["trials"]
        ),
        "scope": (
            "Every actual 320-update short configuration, checked through update 24 after cached-root capture; "
            "full short schedule retained, execution-only limits do not become tuning results. "
            "Original unchanged-source CPU/synthetic gates are separately hash-bound; not reexecuted here. "
            "No long-horizon Torch/fused equivalence, full-budget loss or efficiency claim."
        ),
    }


def require_owned_execution(campaign_path: Path, expected_plan: dict) -> dict:
    """Accept GPU work only from the frozen, generation-bound guarded controller."""
    directory = campaign_path.parent
    frozen = read(directory / "validation-binding.json")
    state = read(directory / "validation-state.json")
    deadline = time.monotonic() + 30
    while state.get("child_pid") != os.getpid() and time.monotonic() < deadline:
        time.sleep(0.1)
        state = read(directory / "validation-state.json")
    helpers = Path(frozen["helpers"])
    parent = observed_process(state["controller_pid"])
    if (
        Path(__file__).resolve() != (helpers / "validate_fixed_age_tuning.py").resolve()
        or frozen["plan"] != expected_plan
        or sha256(Path(frozen["archive"])) != frozen["archive_sha256"]
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or state["status"] != "validating"
        or state["binding_sha256"] != sha256(directory / "validation-binding.json")
        or state["child_pid"] != os.getpid()
        or os.environ.get(PROCESS_MARKER) != state["child_marker"]
        or parent is None
        or parent[0] == "Z"
        or parent[1] != state["controller_start_ticks"]
        or str(helpers / "run_fixed_age_validation.py") not in parent[2]
        or not option(parent[2], "--phase", "validate")
        or not option(parent[2], "--campaign", str(campaign_path))
    ):
        raise RuntimeError("Actual GPU validation is not owned by its bound guarded controller generation")
    return {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}


def executed_trajectory(directory: Path, config: dict, source: Path, frozen: dict) -> dict:
    """Require the exact 24-update execution-check identity rather than a completed trial."""
    run = history(directory)
    if (
        run.manifest["config"] != config
        or run.manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or normalized_sources(source, run.manifest["source_sha256"]) != executed_sources(source, frozen)
        or run.manifest["status"] != "execution_check_complete"
        or run.manifest["completed_steps"] != ENDPOINT
        or len(run.rows) != ENDPOINT
    ):
        raise ValueError("Actual short-configuration validation trajectory differs from its bound identity")
    checkpoints = {}
    for step in (BOUNDARY, ENDPOINT):
        checkpoint = directory / f"checkpoints/step_{step:06d}"
        receipt = read(checkpoint / "complete.json")
        sizes = [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(WORLD)]
        if receipt["step"] != step or receipt["world"] != WORLD or not all(size > 0 for size in sizes):
            raise ValueError("Actual short-configuration checkpoint receipt or rank coverage is incomplete")
        checkpoints[str(step)] = {"receipt_sha256": sha256(checkpoint / "complete.json"), "rank_bytes": sizes}
    return {
        "manifest_sha256": sha256(directory / "run.json"),
        "observed_updates": len(run.rows),
        "history_sha256": {path.name: sha256(path) for path in sorted(directory.glob("metrics_from_*.jsonl"))},
        "checkpoints": checkpoints,
    }


def run(campaign_path: Path, work: Path, receipt_path: Path) -> dict:
    """Execute all thirty-six commands and record all-rank comparisons at the original tolerances."""
    result = plan(campaign_path, work)
    if receipt_path.resolve() != (campaign_path.parent / "validation-gpu.json").resolve():
        raise ValueError("Actual validation receipt must use its canonical campaign path")
    if work.exists() or receipt_path.exists():
        raise FileExistsError("Actual short-config validation already has evidence; preserve it without retry")
    result["owner"] = require_owned_execution(campaign_path, result)
    require_gpu_environment(campaign_path)
    campaign = read(campaign_path)
    root, source = Path(campaign["root"]), Path(campaign["source"])
    frozen = require_source(root, Path(campaign["source_receipt"]))
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if occupied:
        raise RuntimeError("Short-config validation refused while CUDA workloads remain")
    sys.path.insert(0, str(source))
    validation = importlib.import_module("experiment.validate")
    records = importlib.import_module("experiment.records")
    for module in (validation, records):
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise ValueError("Validation numerical helpers were imported from a different trainer")
    validation.require_receipt(root / "data", root / "data/experiment_receipt.json")
    work.mkdir(parents=True, exist_ok=False)
    result["status"] = "validating"
    write(receipt_path, result)
    try:
        for item in result["trials"]:
            config = read(Path(item["config"]))
            full, interrupted = Path(item["continuous"]), Path(item["resumed"])
            for index, command in enumerate(item["commands"]):
                log = work / f"{item['name']}_{('continuous', 'interrupted', 'resumed')[index]}.log"
                with log.open("x") as stream:
                    code = subprocess.run(
                        command,
                        cwd=source,
                        env={**os.environ, **result["environment"]},
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    ).returncode
                execution = {
                    "trial": item["name"],
                    "command": command,
                    "exit_code": code,
                    "log": str(log),
                    "log_sha256": sha256(log),
                }
                result["gpu_commands"].append(execution)
                write(receipt_path, result)
                if code:
                    raise RuntimeError(f"Actual short-config validation failed; inspect {log}")
                if index == 1:
                    cached = interrupted / f"checkpoints/step_{BOUNDARY:06d}"
                    item["cached_root_checkpoint"] = validation.require_cached_root_checkpoint(cached, WORLD)
                    item["boundary_continuation"] = records.compare_continuation(
                        full / f"checkpoints/step_{BOUNDARY:06d}", cached, WORLD
                    )
            item["post_cached_root_continuation"] = records.compare_continuation(
                full / f"checkpoints/step_{ENDPOINT:06d}",
                interrupted / f"checkpoints/step_{ENDPOINT:06d}",
                WORLD,
            )
            item["fixed_age_history"] = {
                "continuous": validation.require_fixed_age_history(full),
                "resumed": validation.require_fixed_age_history(interrupted),
            }
            item["trajectory_identity"] = {
                "continuous": executed_trajectory(full, config, source, frozen),
                "resumed": executed_trajectory(interrupted, config, source, frozen),
            }
            if item["family"].endswith("_fused"):
                item["capture_history"] = {
                    "continuous": validation.require_outer_capture_history(full),
                    "resumed": validation.require_outer_capture_history(interrupted),
                }
            item["checks"] = dict.fromkeys(item["checks"], True)
            write(receipt_path, result)
        if sha256(campaign_path) != result["campaign_sha256"]:
            raise ValueError("Campaign changed during actual short-configuration validation")
        require_source(root, Path(campaign["source_receipt"]))
        analyze(campaign_path)
        result["status"] = "passed_actual_short_config_gates"
        write(receipt_path, result)
        return result
    except BaseException as error:
        result.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(receipt_path, result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt is None:
        result = plan(args.campaign, args.work)
        print(json.dumps(result, indent=2))
        return
    result = run(args.campaign, args.work, args.receipt)
    print(json.dumps({"status": result["status"], "validated_trials": len(result["trials"])}, indent=2))


if __name__ == "__main__":
    main()
