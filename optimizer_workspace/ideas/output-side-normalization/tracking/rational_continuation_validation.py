# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run full-model start interventions and preserve complete continuation evidence."""

import argparse
import os
import subprocess
from pathlib import Path

import rational_continuation_contract as contract
import torch
from analyze_tuning import executed_sources, normalized_sources
from diagnose_continuation import compare_saved
from fused_attribution_validation import INTERRUPTED_MANIFEST, numerical_modules, save_interrupted_manifest
from fused_continuation_contract import validation_commands as original_commands
from fused_continuation_validation import COMPLETE_STATUS as ORIGINAL_STATUS
from fused_continuation_validation import checkpoint_pair, loss_pair, traces, trajectory
from prepare_rational_continuation import require_preparation
from rational_root_start import RATIONAL_SHA256, ROOTS_SHA256
from root_preparation_reconstruction import require_trace_lifecycle
from root_preparation_validation import pair_set_pass, root_trace_lifecycle
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

STATUS = "completed_rational_start_full_model_diagnostic_not_production"
COMMAND_TIMEOUT = 900
STEPS = (contract.BOUNDARY, contract.FIRST_RESUMED, contract.ENDPOINT)


def pair(actual: Path, expected: Path) -> dict:
    return {
        "checkpoints": {str(step): checkpoint_pair(actual, expected, step) for step in STEPS},
        "loss": loss_pair(actual, expected),
    }


def start_receipts(out: Path, candidate: Path, variant: str, mode: str) -> dict:
    _, records, _ = numerical_modules(candidate)
    identity = records.checkpoint_identity(
        read(candidate / contract.CONFIG),
        Path(read(candidate / contract.PREPARATION)["root"]) / "data/experiment_receipt.json",
    )
    starts = (0, contract.BOUNDARY) if mode == "resumed" else (0,)
    paths = {f"root_start_rank_{rank}_from_{step:06d}.json" for rank in range(contract.WORLD) for step in starts}
    actual = {path.name for path in out.glob("root_start_rank_*.json")}
    if mode == "interrupted":
        actual = {name for name in actual if name.endswith("_from_000000.json")}
    if actual != paths:
        raise ValueError("Rational continuation lacks complete per-rank start identity")
    hashes = {name: sha256(out / name) for name in sorted(paths)}
    for step in starts:
        for rank in range(contract.WORLD):
            path = out / f"root_start_rank_{rank}_from_{step:06d}.json"
            if (
                path.is_symlink()
                or path.stat().st_nlink != 1
                or read(path)
                != {
                    "status": "source_bound_start_installed_before_model_and_root_warmup",
                    "variant": variant,
                    "pinned_roots_sha256": ROOTS_SHA256,
                    "rational_start_sha256": RATIONAL_SHA256,
                    "installer_sha256": sha256(candidate / "experiment/rational_root_start.py"),
                    "compilation_preserved": True,
                    "compile_options": {"dynamic": False},
                    "root_graph_lifecycle_unchanged": True,
                    "rank": rank,
                    "start_step": step,
                    "identity": identity,
                }
            ):
                raise ValueError("Rational continuation executed start variant/source differs")
    if any(sha256(out / name) != value for name, value in hashes.items()):
        raise ValueError("Rational continuation start receipts changed during reconstruction")
    return hashes


def execution_evidence(root: Path, candidate: Path, prepared: dict, variant: str, mode: str, out: Path) -> dict:
    evidence = trajectory(out, candidate, prepared, mode)
    evidence["root_start"] = start_receipts(out, candidate, variant, mode)
    manifest_path = out / (INTERRUPTED_MANIFEST if mode == "interrupted" else "run.json")
    if read(manifest_path)["root_start_variant"] != variant:
        raise ValueError("Rational continuation manifest changed its numerical start")
    return evidence


def original(root: Path, candidate: Path) -> dict:
    receipt = root / "run-preparation/diagnostic-fused-continuation-gpu-20261118.json"
    binding = root / "run-preparation/fused-continuation-binding-20261118.json"
    raw, gpu, frozen = receipt.read_bytes(), read(receipt), read(binding)
    prior_candidate = Path(frozen["candidate"])
    work = root / "validation/fused_continuation_20261118"
    commands = original_commands(prior_candidate, root, work)
    coordinates = [(protocol, mode) for protocol in contract.PROTOCOLS for mode in contract.MODES]
    if (
        gpu["status"] != ORIGINAL_STATUS
        or gpu["execution_binding_sha256"] != sha256(binding)
        or gpu["candidate_preparation_sha256"] != frozen["preparation_sha256"]
        or gpu["commands"] != commands
        or len(gpu["gpu_commands"]) != len(coordinates)
    ):
        raise ValueError("Original full-model continuation execution binding or command coverage changed")
    sources = executed_sources(prior_candidate, frozen)
    for entry, coordinate, command in zip(gpu["gpu_commands"], coordinates, commands, strict=True):
        protocol, mode = coordinate
        out = work / f"{protocol}_{'continuous' if mode == 'continuous' else 'resumed'}"
        manifest_path = out / (INTERRUPTED_MANIFEST if mode == "interrupted" else "run.json")
        retained, manifest = entry["trajectory"], read(manifest_path)
        log = work / f"{protocol}_{mode}.log"
        if (
            (entry["protocol"], entry["mode"]) != coordinate
            or entry["command"] != command
            or type(entry["exit_code"]) is not int
            or entry["exit_code"] != 0
            or entry["log"] != str(log)
            or log.is_symlink()
            or log.stat().st_nlink != 1
            or entry["log_sha256"] != sha256(log)
            or manifest_path.is_symlink()
            or manifest_path.stat().st_nlink != 1
            or retained["manifest_sha256"] != sha256(manifest_path)
            or manifest["config"] != read(candidate / contract.CONFIG)
            or read(out / "config.json") != manifest["config"]
            or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(prior_candidate, manifest["source_sha256"]) != sources
            or set(retained["history_sha256"])
            != (
                {"metrics_from_000000.jsonl", f"metrics_from_{contract.BOUNDARY:06d}.jsonl"}
                if mode == "resumed"
                else {"metrics_from_000000.jsonl"}
            )
            or any(
                (out / name).is_symlink() or (out / name).stat().st_nlink != 1 or sha256(out / name) != value
                for name, value in retained["history_sha256"].items()
            )
        ):
            raise ValueError("Original full-model continuation history/source changed")
        steps = (contract.BOUNDARY,) if mode == "interrupted" else STEPS
        if set(retained["checkpoints"]) != {str(step) for step in steps}:
            raise ValueError("Original full-model continuation checkpoint coverage changed")
        for step in steps:
            checkpoint = retained["checkpoints"][str(step)]
            folder = out / f"checkpoints/step_{step:06d}"
            if (
                (folder / "complete.json").is_symlink()
                or (folder / "complete.json").stat().st_nlink != 1
                or checkpoint["receipt_sha256"] != sha256(folder / "complete.json")
                or set(checkpoint["rank_sha256"]) != {f"rank_{rank}.pt" for rank in range(contract.WORLD)}
                or any(
                    (folder / name).is_symlink()
                    or (folder / name).stat().st_nlink != 1
                    or sha256(folder / name) != value
                    for name, value in checkpoint["rank_sha256"].items()
                )
            ):
                raise ValueError("Original full-model continuation checkpoint changed")
    if receipt.read_bytes() != raw:
        raise ValueError("Original full-model continuation receipt changed during reconstruction")
    return gpu


def analyze(root: Path, candidate: Path, prepared: dict) -> dict:
    original_evidence = original(root, candidate)
    continuation, transparency, reproduction, lifecycle, trace_evidence = {}, {}, {}, {}, {}
    for variant in contract.VARIANTS:
        continuation[variant] = {
            protocol: pair(
                contract.directory(root, variant, protocol, "resumed"),
                contract.directory(root, variant, protocol, "continuous"),
            )
            for protocol in contract.PROTOCOLS
        }
        transparency[variant] = {
            mode: pair(
                contract.directory(root, variant, "trace", mode), contract.directory(root, variant, "control", mode)
            )
            for mode in ("continuous", "resumed")
        }
        for mode in ("continuous", "resumed"):
            out = contract.directory(root, variant, "trace", mode)
            key = f"{variant}_{mode}"
            trace_evidence[key] = traces(out, candidate, prepared)
            lifecycle[key] = root_trace_lifecycle(out)
            require_trace_lifecycle(lifecycle[key], "unprepared", mode)
    for protocol in contract.PROTOCOLS:
        for mode in ("continuous", "resumed"):
            reproduction[f"{protocol}_{mode}"] = pair(
                contract.directory(root, "pinned", protocol, mode),
                root / "validation/fused_continuation_20261118" / f"{protocol}_{mode}",
            )
    gates = {
        "trace_transparency": all(pair_set_pass(value) for modes in transparency.values() for value in modes.values()),
        "pinned_original_reproduced": all(pair_set_pass(value) for value in reproduction.values()),
        "rational_all_rank_continuation": all(pair_set_pass(value) for value in continuation["rational"].values()),
    }
    result = {
        "continuation": continuation,
        "transparency": transparency,
        "pinned_original_reproduction": reproduction,
        "trace_root_lifecycle": lifecycle,
        "traces": trace_evidence,
        "gates": gates,
        "diagnostic_intervention_accepted": all(gates.values()),
        "scope": "Full-model bounded diagnostic; no quality, runtime, tuned-LR or production acceptance.",
    }
    if gates["trace_transparency"] and gates["pinned_original_reproduced"]:
        stage_pairs = []
        for variant in contract.VARIANTS:
            for rank in range(contract.WORLD):
                for step in STEPS:
                    paths = [
                        contract.directory(root, variant, "trace", mode)
                        / f"continuation-trace/rank_{rank}/update_{step:06d}.pt"
                        for mode in ("resumed", "continuous")
                    ]
                    saved = [torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in paths]
                    changed, scalars = [], []
                    count = compare_saved(
                        saved[0]["records"],
                        saved[1]["records"],
                        f"{variant}.rank_{rank}.update_{step}",
                        changed,
                        scalars,
                    )
                    stage_pairs.append(
                        {
                            "variant": variant,
                            "rank": rank,
                            "step": step,
                            "tensor_comparisons": count,
                            "changed_tensors": changed,
                            "scalar_or_structure_differences": scalars,
                        }
                    )
        result["intermediates"] = stage_pairs
    else:
        result["intermediates_withheld"] = "Trace transparency or pinned historical reproduction failed."
    if original(root, candidate) != original_evidence:
        raise ValueError("Original full-model evidence changed during intervention analysis")
    return result


def execute(root: Path, candidate: Path) -> dict:
    prepared = require_preparation(root, candidate)
    _, _, state_path, work, receipt = contract.locations(root)
    _, _, owner = numerical_modules(candidate)
    owned = owner.require_gpu_owner(root)
    if work.exists() or receipt.exists():
        raise FileExistsError("Rational full-model execution already has one-shot evidence")
    work.mkdir(parents=True)
    result = contract.plan(candidate, root) | {
        "status": "diagnosing",
        "owner": owned,
        "binding_sha256": sha256(contract.locations(root)[0]),
        "gpu_commands": [],
    }
    write(receipt, result)
    try:
        for coordinate, command in zip(contract.coordinates(), result["commands"], strict=True):
            variant, protocol, mode = coordinate
            if require_preparation(root, candidate) != prepared or owner.require_gpu_owner(root) != owned:
                raise RuntimeError("Rational full-model source or owner changed")
            state = read(state_path) | {"active_command": command}
            write(state_path, state)
            log = work / f"{variant}_{protocol}_{mode}.log"
            with log.open("x") as stream:
                code = subprocess.run(
                    command,
                    cwd=candidate,
                    env=os.environ | contract.ENVIRONMENT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    timeout=COMMAND_TIMEOUT,
                ).returncode
            entry = {
                "variant": variant,
                "protocol": protocol,
                "mode": mode,
                "command": command,
                "exit_code": code,
                "log": str(log),
                "log_sha256": sha256(log),
            }
            result["gpu_commands"].append(entry)
            write(receipt, result)
            if type(code) is not int or code != 0:
                raise RuntimeError(f"Rational full-model command failed; retained at {log}")
            out = contract.directory(root, variant, protocol, mode)
            if mode == "interrupted":
                save_interrupted_manifest(out)
            entry["trajectory"] = execution_evidence(root, candidate, prepared, variant, mode, out)
            write(receipt, result)
        result.update(status=STATUS, analysis=analyze(root, candidate, prepared))
        write(receipt, result)
        return result
    except BaseException as failure:
        result.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(receipt, result)
        raise


def reconstruct(root: Path, candidate: Path) -> dict:
    receipt = contract.locations(root)[4]
    raw, result = receipt.read_bytes(), read(receipt)
    prepared = require_preparation(root, candidate)
    state = read(contract.locations(root)[2])
    planned = contract.plan(candidate, root)
    if (
        result["status"] != STATUS
        or "error" in result
        or any(result.get(key) != value for key, value in planned.items() if key != "status")
        or result["owner"]
        != {
            "controller_pid": state["controller_pid"],
            "controller_start_ticks": state["controller_start_ticks"],
        }
        or result["binding_sha256"] != sha256(contract.locations(root)[0])
        or result["commands"] != contract.validation_commands(candidate, root, contract.locations(root)[3])
        or len(result["gpu_commands"]) != len(contract.coordinates())
    ):
        raise ValueError("Rational full-model result lacks complete original execution")
    for entry, coordinate, command in zip(
        result["gpu_commands"], contract.coordinates(), result["commands"], strict=True
    ):
        variant, protocol, mode = coordinate
        if (
            (entry["variant"], entry["protocol"], entry["mode"]) != coordinate
            or entry["command"] != command
            or type(entry["exit_code"]) is not int
            or entry["exit_code"] != 0
            or entry["log"] != str(contract.locations(root)[3] / f"{variant}_{protocol}_{mode}.log")
            or Path(entry["log"]).is_symlink()
            or Path(entry["log"]).stat().st_nlink != 1
            or entry["log_sha256"] != sha256(Path(entry["log"]))
            or entry["trajectory"]
            != execution_evidence(
                root, candidate, prepared, variant, mode, contract.directory(root, variant, protocol, mode)
            )
        ):
            raise ValueError("Rational full-model retained trajectory differs")
    if analyze(root, candidate, prepared) != result["analysis"] or receipt.read_bytes() != raw:
        raise ValueError("Rational full-model result differs from independent reconstruction")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--phase", choices=("execute", "reconstruct"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    function = execute if args.phase == "execute" else reconstruct
    result = function(args.root.resolve(strict=True), args.candidate.resolve(strict=True))
    print(result["status"])


if __name__ == "__main__":
    main()
