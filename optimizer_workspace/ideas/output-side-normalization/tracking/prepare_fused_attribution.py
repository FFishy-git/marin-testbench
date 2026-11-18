# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a full-schedule fused/null LR neighborhood from immutable Wave12 source."""

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import PYTHON, read, sha256
from prepare_selected_lr_stage_profile import HELPERS as PARENT_RUNTIME_HELPERS
from prepare_selected_lr_stage_profile import original, paths
from prepare_selected_lr_stage_profile_validation import helper_names
from prepare_selected_lr_validation import require_archive
from prepare_upstream_outer import replace_once
from selected_lr_stage_profile_handoff import frozen_result
from upstream_production import cpu_command
from watch_wave5 import write
from wave7_release import candidate_inventory

PREPARATION = "fused-attribution-preparation.json"
STATUS = "fused_attribution_full_schedule_prepared_gpu_denied"
CPU_STATUS = "fused_attribution_cpu_verified_gpu_denied"
FAMILIES = ("muont", "outer_leftdiag_fused", "null_leftdiag_fused")
LR_GRID = (
    ("lower", 0.04525483392),
    ("center", 0.06399999989260598),
    ("upper", 0.09050966784),
)
STEPS = 5120
TOKENS = 2684354560
WINDOW_TOKENS = 536870912
COPIES = {
    "fused_attribution_checks.py": "experiment/fused_attribution_checks.py",
    "test_fused_attribution_checks.py": "experiment/tests/test_fused_attribution_checks.py",
}
OWNER = '''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deny GPU work until a separate source-bound attribution execution owner is issued."""


def require_fused_attribution_owner():
    raise RuntimeError("Fused attribution requires separate validated execution ownership")
'''
POLICY = (
    "Nine exploratory seed0 full-schedule runs; equal three-point matrix-LR grids for MuonT, "
    "outer fused output normalization and its matched null. Retain the original 5120-update "
    "schedule, 2684354560 tokens, fixed input refresh age, compiled model and cached-root graphs. "
    "The null observes, communicates and roots the same output moments, then replaces only the "
    "applied outer vectors by ones. Output-factor packing, unit-RMS normalization and application "
    "remain active; the identity replacement adds work, so null timing is diagnostic. "
    "Compare five complete 536870912-token windows, "
    "all exact shared evaluations and all eleven final-half evaluations; select each family "
    "by final-half mixture selection loss, with endpoint and final two training windows reported. "
    "C4 is reporting-only. A boundary winner requires symmetric full-budget expansion. "
    "Reserve seeds3,4,5 for confirmation. This preparation cannot authorize any GPU command, "
    "queue, launch, checkpoint migration or production release."
)


def config_name(family: str, label: str) -> str:
    return f"experiment/configs/{family}_full_lr_{label}_0p5m_seed0.json"


def configurations(previous: Path) -> dict[str, dict]:
    """Vary only the arm and matrix LR, keeping every full-schedule setting."""
    references = {
        "muont": read(previous / "experiment/configs/muont_0p5m_seed0.json"),
        "outer_leftdiag_fused": read(previous / "experiment/configs/outer_leftdiag_fused_0p5m_seed0.json"),
    }
    for family, config in references.items():
        if (
            config["experiment"]["arm"] != family
            or config["optimizer"]["lr"] != LR_GRID[1][1]
            or config["training"]["steps"] != STEPS
            or config["training"]["tokens"] != TOKENS
            or config["training"]["seed"] != 0
            or config["training"]["world_size"] != 8
            or config["training"]["batch_tokens"] != 524288
            or config["schedule"] != references["muont"]["schedule"]
            or config["experiment"]["refresh_install_age"] != 1
            or config["runtime"]["compile"] == "eager"
            or not config["root"]["graphs"]
        ):
            raise ValueError("Attribution requires the actual Wave12 center-LR full schedules")
    configs = {}
    for label, lr in LR_GRID:
        for family in FAMILIES:
            base = family if family != "null_leftdiag_fused" else "outer_leftdiag_fused"
            config = copy.deepcopy(references[base])
            config["experiment"]["arm"] = family
            config["optimizer"]["lr"] = lr
            configs[config_name(family, label)] = config
    return configs


def result_evidence(root: Path, frozen: dict) -> dict:
    """Reconstruct completed original histories without querying the active profiling allocation."""
    analysis_path = root / "run-preparation/analysis-wave12.json"
    cloud_path = root / "run-preparation/swanlab-upload-wave12.json"
    analysis, cloud = read(analysis_path), read(cloud_path)
    helpers = Path(frozen["helpers"])
    rebuilt = frozen_result(
        PYTHON,
        helpers,
        "import json,sys;from pathlib import Path;"
        "from selected_lr_run_history import binding,completed_run;"
        "from analyze_selected_lr_run import analyze;"
        "root=Path(sys.argv[1]);release=binding(root);"
        "print(json.dumps({'analysis':analyze(root),"
        "'histories':{arm:completed_run(root,arm,release) for arm in release['arms']}},allow_nan=False))",
        [str(root)],
    )
    if (
        rebuilt["analysis"] != {key: value for key, value in analysis.items() if key != "time"}
        or analysis["source_release_sha256"] != sha256(paths(root)[0])
        or cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or cloud["project"] != "siyuc3141/output-side-normalization-marin"
        or set(cloud["runs"]) != set(frozen["arms"])
    ):
        raise ValueError("Attribution requires unchanged completed Wave12 results and verified cloud evidence")
    columns = frozen_result(
        str(root / ".venv-logging/bin/python"),
        helpers,
        "import json,sys;from pathlib import Path;"
        "from swanlab_refresh_upload import load_history,expected_columns;"
        "root=Path(sys.argv[1]);arms=json.loads(sys.argv[2]);"
        "print(json.dumps({arm:{'point_counts':{k:len(v) for k,v in expected_columns("
        "load_history(root,'wave12',arm)).items()},'final_metrics':"
        "load_history(root,'wave12',arm).metrics[-1]} for arm in arms},allow_nan=False))",
        [str(root), json.dumps(frozen["arms"])],
    )
    for arm in frozen["arms"]:
        history, imported = rebuilt["histories"][arm], cloud["runs"][arm]
        verification = imported["verification"]
        if (
            imported["status"] != "verified"
            or any(
                imported[key] != history[key]
                for key in ("manifest_sha256", "metrics_sha256", "checkpoint_receipt_sha256", "checkpoint_rank_bytes")
            )
            or verification["state"] != "FINISHED"
            or verification["configuration_verified"] is not True
            or verification["token_axis_verified"] is not True
            or verification["integer_values_exact"] is not True
            or verification["csv_export_max_ulp"] != 2
            or verification["point_counts"] != columns[arm]["point_counts"]
            or verification["final_metrics"] != columns[arm]["final_metrics"]
            or set(verification["serialization_errors"]) != set(verification["point_counts"])
            or any(
                not math.isfinite(error[key]) or error[key] < 0 or (key == "max_ulp" and error[key] > 2)
                for error in verification["serialization_errors"].values()
                for key in ("max_absolute", "max_ulp")
            )
            or verification["serialization_errors"]["tokens"] != {"max_absolute": 0.0, "max_ulp": 0.0}
            or verification["all_scalar_values_exact"]
            != all(error["max_absolute"] == 0 for error in verification["serialization_errors"].values())
            or verification["point_counts"]["tokens"] != STEPS
            or verification["point_counts"]["train/loss"] != STEPS
            or verification["point_counts"]["eval/selection_loss"] != 22
            or verification["point_counts"]["eval/c4_loss"] != 22
        ):
            raise ValueError("Attribution predecessor cloud identity differs from the original completed history")
    for item in analysis["analyses"]:
        for comparison in item["comparisons"].values():
            if (
                comparison["final_comparison_eligible"] is not True
                or len(comparison["training_windows"]) != 5
                or len(comparison["evaluations"]["selection_loss"]["shared_tokens"]) != 22
                or len(comparison["evaluations"]["selection_loss"]["late_tokens"]) != 11
            ):
                raise ValueError("Attribution requires all five windows and exact full-budget evaluations")
    return {
        "analysis_sha256": sha256(analysis_path),
        "cloud_sha256": sha256(cloud_path),
        "completed_histories": rebuilt["histories"],
    }


def transformed_files(previous: Path) -> dict[str, bytes]:
    """Add one matched-null arm and deny the new trainer before configuration or CUDA setup."""
    block = replace_once(
        (previous / "experiment/block_toner.py").read_text(),
        "    lag_left: bool = False",
        "    lag_left: bool = False\n"
        "    # Matched outer-diagonal attribution retains output work and applies identity vectors.\n"
        "    identity_left_outer: bool = False",
    )
    block = replace_once(
        block,
        '    "outer_leftdiag_fused": BlockArm(Side.FULL, Side.DIAGONAL, Side.DIAGONAL, '
        "apply_left=True, apply_left_inner=False),",
        '    "outer_leftdiag_fused": BlockArm(Side.FULL, Side.DIAGONAL, Side.DIAGONAL, '
        "apply_left=True, apply_left_inner=False),\n"
        '    "null_leftdiag_fused": BlockArm(Side.FULL, Side.DIAGONAL, Side.DIAGONAL, '
        "apply_left=True, apply_left_inner=False, identity_left_outer=True),",
    )
    block = replace_once(
        block,
        "        elif not self.arm.apply_left_inner:\n"
        "            left = {key: (None, outer) for key, (_, outer) in left.items()}",
        "        elif not self.arm.apply_left_inner:\n"
        "            left = {key: (None, outer) for key, (_, outer) in left.items()}\n"
        "        if self.arm.identity_left_outer:\n"
        "            left = {key: (inner, torch.ones_like(outer)) for key, (inner, outer) in left.items()}",
    )
    configure = replace_once(
        (previous / "experiment/configure.py").read_text(),
        '    "outer_leftdiag_fused",',
        '    "outer_leftdiag_fused",\n    "null_leftdiag_fused",',
    )
    train = replace_once(
        (previous / "experiment/train.py").read_text(),
        "from .selected_lr_release_owner import require_selected_lr_release_owner",
        "from .fused_attribution_owner import require_fused_attribution_owner",
    )
    train = replace_once(train, "    require_selected_lr_release_owner(args)", "    require_fused_attribution_owner()")
    tests = replace_once(
        (previous / "experiment/tests/test_block_toner.py").read_text(),
        '        "outer_leftdiag_fused": ((), ("q", "k", "v", "o", "gate", "up", "down")),',
        '        "outer_leftdiag_fused": ((), ("q", "k", "v", "o", "gate", "up", "down")),\n'
        '        "null_leftdiag_fused": ((), ("q", "k", "v", "o", "gate", "up", "down")),',
    )
    tests = replace_once(
        tests,
        '        "outer_leftdiag_fused",\n    ]',
        '        "outer_leftdiag_fused",\n        "null_leftdiag_fused",\n    ]',
    )
    cluster = read(previous / "experiment/cluster.json")
    cluster["wave_name"] = "fused-attribution-full-schedule-unreleased"
    directory = Path(__file__).parent
    checks = {destination: (directory / source).read_bytes() for source, destination in COPIES.items()}
    test_name = COPIES["test_fused_attribution_checks.py"]
    checks[test_name] = replace_once(
        checks[test_name].decode(),
        "from fused_attribution_checks import",
        "from experiment.fused_attribution_checks import",
    ).encode()
    return {
        "experiment/block_toner.py": block.encode(),
        "experiment/configure.py": configure.encode(),
        "experiment/train.py": train.encode(),
        "experiment/tests/test_block_toner.py": tests.encode(),
        "experiment/fused_attribution_owner.py": OWNER.encode(),
        "experiment/cluster.json": (json.dumps(cluster, indent=2) + "\n").encode(),
        **checks,
        **{
            name: (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
            for name, config in configurations(previous).items()
        },
    }


def helper_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    names = helper_names(
        directory,
        (Path(__file__).name, *COPIES, *PARENT_RUNTIME_HELPERS, "test_prepare_fused_attribution.py"),
    )
    return {name: sha256(directory / name) for name in names}


def preparation_record(root: Path, candidate: Path, frozen: dict, files: dict, evidence: dict) -> dict:
    previous = Path(frozen["candidate"])
    trials = []
    for label, lr in LR_GRID:
        for family in FAMILIES:
            trials.append(
                {
                    "name": f"{family}_full_lr_{label}",
                    "family": family,
                    "lr_label": label,
                    "matrix_lr": lr,
                    "config": str(candidate / config_name(family, label)),
                    "steps": STEPS,
                    "tokens": TOKENS,
                    "paired_seed": 0,
                }
            )
    return {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_release": str(paths(root)[0]),
        "parent_release_sha256": sha256(paths(root)[0]),
        "parent_archive": frozen["archive"],
        "parent_archive_sha256": frozen["archive_sha256"],
        "parent_metadata_sha256": {path.name: sha256(path) for path in sorted(previous.glob("*.json"))},
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "files_sha256": files,
        "changed_files": sorted(transformed_files(previous)),
        "helpers_sha256": helper_hashes(),
        "predecessor_results": evidence,
        "trials": trials,
        "total_training_tokens": len(trials) * TOKENS,
        "training_window_tokens": WINDOW_TOKENS,
        "selection": {
            "metric": "selection_loss",
            "rule": "minimum mean of the eleven exact shared final-half evaluations within each family",
            "endpoint_role": "reported separately, never an independent replication",
            "boundary_action": "symmetric full-budget expansion before tuned-superiority claims",
        },
        "reserved_confirmation_seeds": [3, 4, 5],
        "policy": POLICY,
    }


def prepare(root: Path, candidate: Path) -> dict:
    """Preserve the issued Wave12 source and create a separate unconditionally GPU-denied candidate."""
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or candidate.is_relative_to(Path(frozen["helpers"])):
        raise FileExistsError("Attribution requires an unused separate candidate")
    evidence, contents = result_evidence(root, frozen), transformed_files(previous)
    helpers = helper_hashes()
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, content in contents.items():
        (candidate / name).write_bytes(content)
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != set(contents) or helper_hashes() != helpers:
        raise ValueError("Attribution candidate exceeds its exact null/config/denial/test scope")
    result = preparation_record(root, candidate, frozen, files, evidence)
    write(candidate / PREPARATION, result)
    require_preparation(root, candidate)
    return result


def require_preparation(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    record, files = read(candidate / PREPARATION), candidate_inventory(candidate)
    expected = preparation_record(root, candidate, frozen, files, result_evidence(root, frozen))
    contents = transformed_files(previous)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if (
        record.keys() != expected.keys() | {"time"}
        or any(record[key] != value for key, value in expected.items())
        or changed != set(contents)
        or any((candidate / name).read_bytes() != content for name, content in contents.items())
        or any(sha256(candidate / name) != digest for name, digest in expected["parent_metadata_sha256"].items())
    ):
        raise ValueError("Attribution source, full-schedule configuration, evidence or exact transformation differs")
    return record


def archive_inventory(prepared: dict, digest: str) -> dict:
    return (
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: digest}
        | {f"tracking/{name}": value for name, value in prepared["helpers_sha256"].items()}
    )


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    """Run all candidate CPU tests, retaining failures and exact tested source archives."""
    root, candidate, out = root.resolve(), candidate.resolve(), out.resolve()
    prepared = require_preparation(root, candidate)
    if out.exists() or any(out.is_relative_to(path) for path in (candidate, Path(prepared["parent_candidate"]))):
        raise FileExistsError("Attribution CPU evidence requires an unused external directory")
    digest = sha256(candidate / PREPARATION)
    out.mkdir(parents=True, exist_ok=False)
    command, log, archive = cpu_command(), out / "candidate.log", out / "source.tar"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command,
                cwd=candidate,
                env=os.environ
                | {
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(candidate),
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Attribution candidate CPU timeout at 600 seconds"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in prepared["files_sha256"] | prepared["parent_metadata_sha256"] | {PREPARATION: digest}:
                bundle.add(candidate / name, arcname=name, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
            for name in prepared["helpers_sha256"]:
                bundle.add(Path(__file__).parent / name, arcname=f"tracking/{name}", recursive=False)
        require_archive(archive, archive_inventory(prepared, digest), alias=True)
        if require_preparation(root, candidate) != prepared or sha256(candidate / PREPARATION) != digest:
            raise ValueError("Attribution source or predecessor evidence changed during CPU verification")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": CPU_STATUS if code == 0 and error is None else "fused_attribution_cpu_failed",
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": digest,
        "files_sha256": prepared["files_sha256"],
        "helpers_sha256": prepared["helpers_sha256"],
        "command": command,
        "cwd": str(candidate),
        "cuda_visible_devices": "",
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
        "policy": POLICY,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError("Attribution candidate CPU/archive verification failed; preserve the original receipt")
    return require_cpu(root, out / "receipt.json")


def require_cpu(root: Path, path: Path) -> dict:
    receipt = read(path)
    candidate = Path(receipt["candidate"])
    prepared = require_preparation(root, candidate)
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root.resolve())
        or receipt["candidate"] != str(candidate.resolve())
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(candidate)
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
    ):
        raise ValueError("Attribution CPU outcome or original tested identity differs")
    require_archive(Path(receipt["archive"]), archive_inventory(prepared, receipt["preparation_sha256"]), alias=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--phase", choices=("prepare", "check", "cpu", "verify"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "verify":
        if args.receipt is None:
            parser.error("--receipt is required for verify")
        result = require_cpu(args.root, args.receipt)
    else:
        if args.candidate is None:
            parser.error("--candidate is required")
        if args.phase == "prepare":
            result = prepare(args.root, args.candidate)
        elif args.phase == "check":
            result = require_preparation(args.root, args.candidate)
        else:
            if args.out is None:
                parser.error("--out is required for cpu")
            result = cpu(args.root, args.candidate, args.out)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
