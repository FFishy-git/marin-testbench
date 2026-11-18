# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a bounded upstream-product probe; GPU use waits for completed upper-LR work."""

import argparse
import json
import math
import os
import subprocess
import tarfile
from pathlib import Path

from boundary_handoff import release
from diagnose_upstream_outer_gpu import cpu_source, describe
from launch_wave5 import PYTHON, node_action, read, sha256
from validate_wave5 import ENVIRONMENT
from watch_wave5 import process, write
from wave7_release import bounded_execution, idle_training, local_node, node_guard, release_keeper

NODE = "28.216.17.139"
CAMPAIGN = "lr-fixed-age-upper-boundary-0p5m-20261117"
EXTRA_HELPERS = (
    "run_upstream_outer_gpu.py",
    "diagnose_upstream_outer_gpu.py",
    "diagnose_combined_outer_gpu.py",
    "diagnose_combined_outer.py",
    "diagnose_fused_diagonal.py",
    "benchmark_diagonal.py",
)


def predecessor_release(root: Path) -> tuple[Path, dict]:
    release(root)
    campaign = root / "campaigns" / CAMPAIGN / "campaign.json"
    execution = read(campaign.parent / "execution-binding.json")
    if (
        execution["campaign"] != str(campaign)
        or execution["campaign_sha256"] != sha256(campaign)
        or sha256(Path(execution["archive"])) != execution["archive_sha256"]
        or any(
            sha256(Path(execution["helpers"]) / name) != digest for name, digest in execution["helpers_sha256"].items()
        )
    ):
        raise ValueError("Predecessor upper-LR execution source or campaign changed")
    return campaign, execution


def freeze(root: Path, receipt_path: Path, out: Path) -> dict:
    """Archive diagnostic helpers separately without stopping a keeper or starting a GPU job."""
    candidate, cpu = cpu_source(receipt_path)
    plan = describe(receipt_path)
    campaign, execution = predecessor_release(root)
    if candidate.parent.parent != root or out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise ValueError("Upstream probe needs a new evidence directory outside its root-bound candidate")
    out.mkdir(parents=True, exist_ok=False)
    helpers = out / "helpers"
    helpers.mkdir()
    archive = out / "source.tar"
    names = tuple(execution["helpers_sha256"]) + EXTRA_HELPERS
    if len(set(names)) != len(names):
        raise ValueError("Upstream probe helper inventory has duplicate names")
    with tarfile.open(archive, "x") as bundle:
        for name in names:
            original = (
                Path(execution["helpers"]) / name
                if name in execution["helpers_sha256"]
                else Path(__file__).parent / name
            )
            destination = helpers / name
            destination.write_bytes(original.read_bytes())
            bundle.add(destination, arcname=f"tracking/{name}", recursive=False)
        bundle.add(receipt_path, arcname="cpu-receipt.json", recursive=False)
    result = {
        "status": "frozen_waiting_upper_lr_completion_not_launch_gate",
        "root": str(root.resolve()),
        "out": str(out.resolve()),
        "node": NODE,
        "cpu_receipt": str(receipt_path.resolve()),
        "cpu_receipt_sha256": sha256(receipt_path),
        "candidate_files_sha256": cpu["files_sha256"],
        "campaign": str(campaign),
        "campaign_sha256": sha256(campaign),
        "execution_binding_sha256": sha256(campaign.parent / "execution-binding.json"),
        "predecessor_release_sha256": sha256(root / "run-preparation/wave8-launch-binding.json"),
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in names},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "environment": ENVIRONMENT,
        "plan": plan,
        "policy": (
            "No GPU launch or queue at freeze. Run only after upper-LR trials complete, full-resolution cloud "
            "verification, fresh terminal/restoration checks and exclusive node lock. One bounded marked child, "
            "cleanup and keeper restoration. No training authorization or retries after execution evidence exists."
        ),
    }
    write(out / "binding.json", result)
    return result


def binding(root: Path, receipt_path: Path, out: Path) -> dict:
    frozen = read(out / "binding.json")
    campaign, predecessor = predecessor_release(root)
    candidate, cpu = cpu_source(receipt_path)
    helpers = out / "helpers"
    if (
        Path(__file__).resolve() != (helpers / "run_upstream_outer_gpu.py").resolve()
        or frozen["status"] != "frozen_waiting_upper_lr_completion_not_launch_gate"
        or frozen["root"] != str(root.resolve())
        or frozen["out"] != str(out.resolve())
        or frozen["node"] != NODE
        or frozen["cpu_receipt"] != str(receipt_path.resolve())
        or frozen["cpu_receipt_sha256"] != sha256(receipt_path)
        or frozen["candidate_files_sha256"] != cpu["files_sha256"]
        or candidate.parent.parent != root
        or frozen["campaign"] != str(campaign)
        or frozen["campaign_sha256"] != sha256(campaign)
        or frozen["execution_binding_sha256"] != sha256(campaign.parent / "execution-binding.json")
        or frozen["predecessor_release_sha256"] != sha256(root / "run-preparation/wave8-launch-binding.json")
        or set(frozen["helpers_sha256"]) != set(predecessor["helpers_sha256"]) | set(EXTRA_HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or any(frozen["helpers_sha256"][name] != digest for name, digest in predecessor["helpers_sha256"].items())
        or sha256(Path(frozen["archive"])) != frozen["archive_sha256"]
        or frozen["environment"] != ENVIRONMENT
        or frozen["plan"] != describe(receipt_path)
    ):
        raise ValueError("Upstream GPU probe binding changed or is not executing its frozen copy")
    return frozen


def completed_predecessor(frozen: dict) -> dict:
    """Rederive the completed study and compare original full-resolution cloud provenance."""
    campaign = Path(frozen["campaign"])
    execution = read(campaign.parent / "execution-binding.json")
    cloud_path = campaign.parent / "swanlab-upload.json"
    analysis_path = campaign.parent / "analysis.json"
    if not cloud_path.exists() or not analysis_path.exists():
        raise RuntimeError("Upper-LR completed results and cloud verification are pending")
    script = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_fixed_age_tuning import require_ready,snapshot,completed_trial
from analyze_lr_boundary import analyze
path=Path(sys.argv[2])
require_ready(path)
observed=snapshot(path)
if observed["status"]!="complete" or observed["all_workers_terminal"] is not True:
    raise RuntimeError("Upper-LR workers or restoration remain live, unknown or incomplete")
analysis=analyze(path)
trials={trial["name"]:completed_trial(path,trial["name"]) for trial in json.load(open(path))["trials"]}
print(json.dumps({"snapshot":observed,"analysis":analysis,"trials":trials},allow_nan=False))
"""
    completed = subprocess.run(
        [PYTHON, "-c", script, execution["helpers"], str(campaign)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    evidence = json.loads(completed.stdout)
    saved = read(analysis_path)
    rederived = evidence["analysis"]
    rederived["time"] = saved["time"]
    cloud = read(cloud_path)
    planned = read(campaign)["trials"]
    if (
        rederived != saved
        or saved["status"] != "complete"
        or set(saved["completed_cumulative_family_lr_choices"])
        != {"muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused"}
        or cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or not cloud["project"].endswith("/output-side-normalization-marin")
        or cloud["campaign_sha256"] != sha256(campaign)
        or cloud["execution_binding_sha256"] != frozen["execution_binding_sha256"]
        or set(cloud["runs"]) != {trial["name"] for trial in planned}
    ):
        raise ValueError("Upper-LR complete analysis or verified cloud campaign differs")
    for trial in planned:
        run = campaign.parent / "runs" / trial["name"]
        checkpoint = run / "checkpoints/step_000320"
        imported = cloud["runs"][trial["name"]]
        verification = imported["verification"]
        provenance = {
            "manifest_sha256": sha256(run / "run.json"),
            "metrics_sha256": sha256(run / "metrics_from_000000.jsonl"),
            "checkpoint_receipt_sha256": sha256(checkpoint / "complete.json"),
            "checkpoint_rank_bytes": [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)],
            "campaign_sha256": sha256(campaign),
            "execution_binding_sha256": frozen["execution_binding_sha256"],
            "trial_config_sha256": trial["config_sha256"],
        }
        if (
            imported["status"] != "verified"
            or any(imported[key] != value for key, value in provenance.items())
            or verification["state"] != "FINISHED"
            or any(
                verification[key] is not True
                for key in ("configuration_verified", "token_axis_verified", "integer_values_exact")
            )
            or verification["point_counts"]["train/loss"] != 320
            or verification["point_counts"]["tokens"] != 320
            or verification["point_counts"]["eval/selection_loss"] != 9
            or verification["point_counts"]["eval/c4_loss"] != 9
        ):
            raise ValueError(f"Upper-LR cloud history differs from its original completed trial: {trial['name']}")
    return {
        "snapshot": evidence["snapshot"],
        "analysis_sha256": sha256(analysis_path),
        "cloud_sha256": sha256(cloud_path),
    }


def require_result(frozen: dict, actual: dict, state: dict) -> None:
    """Reject a passed label with missing shapes, parity cases, owners or timing evidence."""
    plan = frozen["plan"]
    expected_cases = {
        (tuple(shape), pattern, nesterov, lr)
        for shape in plan["shapes"]
        for pattern in plan["patterns"]
        for nesterov in plan["nesterov"]
        for lr in plan["learning_rates"]
    }
    expected_timing = {(tuple(shape), graphs) for shape in plan["shapes"] for graphs in (False, True)}
    if (
        actual["status"] != "passed_upstream_production_diagnostic_not_launch_gate"
        or any(actual.get(key) != value for key, value in plan.items() if key not in ("status", "cases"))
        or actual["owner"]
        != {
            "controller_pid": state["controller_pid"],
            "controller_start_ticks": state["controller_start_ticks"],
        }
        or actual["binding_sha256"] != state["binding_sha256"]
        or actual["helpers_sha256"] != frozen["helpers_sha256"]
        or len(actual["cases"]) != len(expected_cases)
        or {(tuple(item["shape"]), item["pattern"], item["nesterov"], item["lr"]) for item in actual["cases"]}
        != expected_cases
        or len(actual["timing"]) != len(expected_timing)
        or {(tuple(item["shape"]), item["graphs"]) for item in actual["timing"]} != expected_timing
    ):
        raise ValueError("Upstream diagnostic has incomplete or changed coverage/ownership")
    exact_fields = {
        "momentum",
        "ns_source",
        "ns_output",
        "parameter_delta",
        "outer_momentum",
        "outer_ns_source",
        "outer_ns_output",
    }
    for item in actual["cases"]:
        equation = item["same_ns_outer_equation"]
        if (
            item["exact_null_and_unchanged_input"] != dict.fromkeys(exact_fields, True)
            or equation["within_tolerance"] is not True
            or equation["finite"] is not True
            or equation["mismatched_elements"] != 0
            or equation["elements"] != math.prod(item["shape"])
        ):
            raise ValueError("Upstream diagnostic reports failed or missing numerical parity")
    for item in actual["timing"]:
        values = item["paired_median_ms"]
        if (
            set(values) != {"legacy_direct", "upstream_outer", "upstream_null"}
            or any(not math.isfinite(value) or value <= 0 for value in values.values())
            or item["upstream_outer_over_legacy"] != values["upstream_outer"] / values["legacy_direct"]
            or item["upstream_outer_over_null"] != values["upstream_outer"] / values["upstream_null"]
        ):
            raise ValueError("Upstream diagnostic timing has missing or nonfinite workloads")


def run(root: Path, receipt_path: Path, out: Path) -> dict:
    frozen = binding(root, receipt_path, out)
    if local_node() != NODE:
        raise RuntimeError("Upstream probe must use its designated original .139 allocation")
    state_path = out / "state.json"
    if state_path.exists() or (out / "result.json").exists():
        raise FileExistsError("Upstream diagnostic execution evidence already exists; do not retry")
    with node_guard(root, NODE) as signals:
        predecessor = completed_predecessor(frozen)
        idle_training()
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(devices) != 8 or any("A100-SXM4-40GB" not in name for name in devices):
            raise ValueError("Upstream probe requires the original eight-A100 allocation")
        write(out / "predecessor.json", predecessor)
        state = {
            "status": "preflight",
            "root": str(root),
            "python": PYTHON,
            "node": NODE,
            "controller_pid": os.getpid(),
            "controller_start_ticks": process(os.getpid())[1],
            "helpers_sha256": frozen["helpers_sha256"],
            "binding_sha256": sha256(out / "binding.json"),
            "cpu_receipt_sha256": sha256(receipt_path),
            "predecessor_sha256": sha256(out / "predecessor.json"),
        }
        write(state_path, state)
        try:
            state["keeper"] = release_keeper(state, state_path)
            state["status"] = "running"
            command = [
                PYTHON,
                str(out / "helpers/diagnose_upstream_outer_gpu.py"),
                "--cpu-receipt",
                str(receipt_path),
                "--out",
                str(out),
            ]
            code = bounded_execution(command, out / "helpers", out / "diagnostic.log", state, state_path, timeout=1800)
            state["exit_code"] = code
            actual = read(out / "result.json")
            state["result_sha256"] = sha256(out / "result.json")
            if code or actual["status"] != "passed_upstream_production_diagnostic_not_launch_gate":
                raise RuntimeError("Upstream probe failed; preserve results without changing tolerances")
            require_result(frozen, actual, state)
            binding(root, receipt_path, out)
            state["status"] = "measured_upstream_probe_not_launch_gate"
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state:
                        raise RuntimeError("Unverified marked-child cleanup prevents keeper restoration")
                    node_action(root, "restore")
                    state["keeper_restore"] = 0
                except Exception as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(state_path, state)
        if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
            raise RuntimeError("Upstream probe keeper restoration failed")
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "check", "run"), required=True)
    args = parser.parse_args()
    result = (
        freeze(args.root, args.cpu_receipt, args.out)
        if args.phase == "freeze"
        else (
            binding(args.root, args.cpu_receipt, args.out)
            if args.phase == "check"
            else run(args.root, args.cpu_receipt, args.out)
        )
    )
    print(json.dumps({"status": result["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
