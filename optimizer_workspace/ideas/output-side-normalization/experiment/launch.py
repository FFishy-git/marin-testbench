# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect four A100 nodes and launch four independent eight-rank experiments."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from .records import atomic_json, sha256, utc_now
from .reference import TASK
from .related_baselines import RELATED_ARMS

CLUSTER = json.loads((TASK / "experiment" / "cluster.json").read_text())
NODES = tuple(CLUSTER["nodes"])
# Wave entries name an arm and its seed, so one wave can mix a candidate with a
# matched control and a second seed of the same arm.
WAVE = tuple((entry["arm"], entry["seed"], entry["batch"]) for entry in CLUSTER["wave"])


def run_name(arm, seed, batch="0p5m"):
    batch_suffix = "" if batch == "0p5m" else f"_{batch}"
    seed_suffix = "" if seed == 0 else f"_seed{seed}"
    return f"{arm}{batch_suffix}{seed_suffix}"


def config_path(arm, seed, batch="0p5m"):
    return TASK / "experiment" / "configs" / f"{arm}_{batch}_seed{seed}.json"


WAVE_NAME = CLUSTER["wave_name"]
PYTHON = CLUSTER["python"]
OUTPUT_ROOT = Path(CLUSTER["output_root"])
ENVIRONMENT = CLUSTER["environment"]
# Synthetic GPU gates for the all-block arms, in addition to the first-wave gates.
BLOCK_CHECKS = ("gpu_output_rows", "gpu_block_updates", "block_distributed_statistics", "block_compiled_body")
RELATED_CHECKS = (
    "go_muon_gpu_numerics",
    "go_muon_gpu_statistics",
    "go_muon_gpu_compiled_capture",
    "mousse_gpu_numerics",
)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]


def wave_entries(path=None):
    """Resolve a separately prepared wave without changing the active cluster wave."""
    payload = CLUSTER if path is None else json.loads(path.read_text())
    entries = []
    for entry in payload["wave"]:
        config = (
            TASK / entry["config"] if "config" in entry else config_path(entry["arm"], entry["seed"], entry["batch"])
        )
        config.resolve().relative_to(TASK / "experiment")
        loaded = json.loads(config.read_text())
        if loaded["experiment"]["arm"] != entry["arm"] or loaded["training"]["seed"] != entry["seed"]:
            raise ValueError("Wave entry differs from its configuration")
        entries.append(
            {**entry, "name": entry.get("name", run_name(entry["arm"], entry["seed"], entry["batch"])), "path": config}
        )
    if len(entries) != len(NODES) or len({entry["name"] for entry in entries}) != len(entries):
        raise ValueError("A wave must have one uniquely named entry per node")
    return payload["wave_name"], entries


def continuation_steps(config):
    """Cross each related method's first periodic refresh before accepting resume."""
    related = config["experiment"].get("related_baseline")
    if related is None:
        return 3, 2
    if config["experiment"]["arm"].startswith("go_muon"):
        refresh = config["root"]["refresh"]
    else:
        refresh = related["settings"]["refresh"]
    return 2 * refresh + 2, refresh + 1


def remote(node, arguments):
    command = shlex.join(arguments)
    result = subprocess.run([*SSH, f"root@{node}", command], text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{node}: {result.stderr.strip()}")
    return result.stdout.strip()


def inspect(node, data, receipt, wave):
    gpu_output = remote(
        node,
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
    )
    process_output = remote(
        node,
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    gpus = []
    blockers = []
    for row in gpu_output.splitlines():
        index, name, utilization, memory = (value.strip() for value in row.split(","))
        gpu = {"index": int(index), "name": name, "utilization": int(utilization), "memory_mib": int(memory)}
        gpus.append(gpu)
        if "A100-SXM4-40GB" not in name:
            blockers.append(f"GPU {index} is not the planned A100 40GB")
        if gpu["utilization"] > 5 or gpu["memory_mib"] > 512:
            blockers.append(f"GPU {index} is occupied")
    if len(gpus) != 8:
        blockers.append(f"Expected 8 GPUs, found {len(gpus)}")
    if process_output:
        blockers.append("Existing CUDA compute processes must be released by their owner")
    code_check = remote(
        node,
        [
            PYTHON,
            "-c",
            "import pathlib; p=pathlib.Path(" + repr(str(TASK / "experiment" / "train.py")) + "); "
            "print('present' if p.is_file() else 'missing')",
        ],
    )
    if code_check != "present":
        blockers.append("Experiment source is missing on this node")
    else:
        paths = sorted((TASK / "experiment").rglob("*.py"))
        paths += sorted((TASK / "experiment" / "configs").glob("*.json"))
        paths += [entry["path"] for entry in wave]
        expected = {str(path.relative_to(TASK)): sha256(path) for path in paths}
        hashes = remote(
            node,
            [
                PYTHON,
                "-c",
                "import pathlib,hashlib,json;"
                f"root=pathlib.Path({str(TASK)!r});names={list(expected)!r};"
                "print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}))",
            ],
        )
        if json.loads(hashes) != expected:
            blockers.append("Remote experiment code/configuration hashes differ")
    if data is None or receipt is None:
        blockers.append("Prepared data and full-plan SHA256 receipt have not been supplied")
    else:
        result = remote(
            node,
            [
                PYTHON,
                "-c",
                "import sys;sys.path.insert(0," + repr(str(TASK)) + ");"
                "from pathlib import Path;from experiment.data_access import require_receipt;"
                f"require_receipt(Path({str(data)!r}),Path({str(receipt)!r}));print('verified')",
            ],
        )
        if result != "verified":
            blockers.append("Data receipt check failed")
    return {"node": node, "gpus": gpus, "compute_processes": process_output, "blockers": blockers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-receipt", type=Path)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--report", type=Path, default=OUTPUT_ROOT / "run-preparation" / "preflight.json")
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT / "runs" / WAVE_NAME)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--wave-manifest", type=Path, help="Prepared four-entry comparison wave.")
    args = parser.parse_args()
    wave_name, wave = wave_entries(args.wave_manifest)
    if args.wave_manifest is not None and args.out == OUTPUT_ROOT / "runs" / WAVE_NAME:
        args.out = OUTPUT_ROOT / "runs" / wave_name
    reports = []
    for node in NODES:
        try:
            reports.append(inspect(node, args.data, args.data_receipt, wave))
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            reports.append({"node": node, "blockers": [str(error)]})
    blockers = [f"{report['node']}: {blocker}" for report in reports for blocker in report["blockers"]]
    if args.validation_receipt is None:
        blockers.append("GPU/distributed/full-model smoke/continuation validation receipt is missing")
    else:
        receipt = json.loads(args.validation_receipt.read_text())
        if receipt.get("environment") != ENVIRONMENT:
            blockers.append("Validation used different cluster runtime settings")
        expected = {str(p.relative_to(TASK)): sha256(p) for p in sorted((TASK / "experiment").rglob("*.py"))}
        if receipt["source_sha256"] != expected:
            blockers.append("Validation receipt is stale for the current experiment code")
        configurations = {entry["name"]: sha256(entry["path"]) for entry in wave}
        if receipt.get("config_sha256") != configurations:
            blockers.append("Full-model validation has not covered the current arm configurations")
        if not all(
            receipt["checks"].get(check, False)
            for check in (
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
                *BLOCK_CHECKS,
            )
        ):
            blockers.append("Required numerical and real-data checks have not all passed")
        if not receipt.get("evidence"):
            blockers.append("Validation receipt has no supporting evidence")
        smokes = receipt.get("evidence", {}).get("real_data_smokes", {})
        if any(entry["arm"] in RELATED_ARMS for entry in wave):
            for check in RELATED_CHECKS:
                if not receipt["checks"].get(check, False):
                    blockers.append(f"Required related-baseline accelerator check is pending: {check}")
        for entry in wave:
            if entry["arm"] in RELATED_ARMS:
                complete, split = continuation_steps(json.loads(entry["path"].read_text()))
                evidence = smokes.get(entry["name"], {})
                if evidence.get("completed_steps") != complete or evidence.get("split_steps") != split:
                    blockers.append(f"{entry['name']}: baseline refresh-crossing continuation is missing")
                if entry["arm"] != "muon_control" and not evidence.get("pre_refresh_continuation"):
                    blockers.append(f"{entry['name']}: restored factors have not been consumed before the next refresh")
        if args.data_receipt is not None and receipt.get("data_receipt_sha256") != sha256(args.data_receipt):
            blockers.append("Validation receipt belongs to different prepared data")
        for execution in receipt.get("evidence", {}).get("gpu_commands", []):
            if execution["exit_code"] != 0 or sha256(Path(execution["log"])) != execution["log_sha256"]:
                blockers.append("Validation command evidence failed or changed")
    report = {
        "time": utc_now(),
        "wave_name": wave_name,
        "environment": ENVIRONMENT,
        "nodes": reports,
        "blockers": blockers,
        "launches": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)
    if not args.launch:
        return
    if blockers:
        raise SystemExit("Launch refused: resolve recorded blockers first.")
    if args.out is None:
        parser.error("--launch requires --out")
    for node, entry in zip(NODES, wave, strict=True):
        arm, seed, batch = entry["arm"], entry["seed"], entry["batch"]
        config = entry["path"]
        out = args.out / entry["name"]
        # The remote launch uses a dedicated helper; it never stops other processes.
        result = remote(
            node,
            [
                PYTHON,
                "-c",
                "import sys;sys.path.insert(0," + repr(str(TASK)) + ");"
                "from experiment.launch import start_local;from pathlib import Path;"
                f"start_local(Path({str(config)!r}),Path({str(args.data)!r}),"
                f"Path({str(args.data_receipt)!r}),Path({str(out)!r}))",
            ],
        )
        report["launches"].append({"node": node, "arm": arm, "seed": seed, "batch": batch, **json.loads(result)})
        atomic_json(args.report, report)
        print(f"Launched {entry['name']} on {node}: {result}", flush=True)


def start_local(config, data, receipt, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    log = out.parent / f"{out.name}.torchrun.log"
    with log.open("x") as stream:
        command = [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
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
        process = subprocess.Popen(
            command,
            cwd=TASK,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, **ENVIRONMENT},
        )
    print(
        json.dumps(
            {"pid": process.pid, "log": str(log), "out": str(out), "command": command, "environment": ENVIRONMENT}
        )
    )


if __name__ == "__main__":
    main()
