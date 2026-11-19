# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare distinct NS-budget full-model configurations without issuing GPU ownership."""

import argparse
import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_budget_gpu_owner as owner
import prepare_rational_continuation as original
from prepare_fused_attribution import FAMILIES, LR_GRID, TOKENS, WINDOW_TOKENS
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

PARENT_CPU = "run-preparation/ns-budget-gpu-owner-cpu/20261119T0003/receipt.json"
PARENT_CPU_SHA256 = "18e0b21734ebdebca8934319753fc6dbed37e5e565c5e28df5d0d1f453f81c69"
PREPARATION = "ns-budget-full-model-preparation.json"
STATUS = "ns_budget_full_model_prepared_gpu_denied"
CPU_STATUS = "ns_budget_full_model_cpu_archive_verified_gpu_unissued"
STEPS = (5, 7, 10)
MODES = ("tpu", "f32")
NEW_HELPERS = ("prepare_ns_budget_full_model.py", "test_prepare_ns_budget_full_model.py")
OWNER_PATH = "experiment/fused_attribution_owner.py"
PARENT_OWNER_FIXTURE = "fixtures/ns-full-model-parent/fused_attribution_owner.py"
DENIED_OWNER = '''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deny NS-budget training until a distinct full-model execution owner is accepted."""


def require_fused_attribution_owner(args: object) -> None:
    raise RuntimeError("NS-budget full-model execution requires a separately accepted owner")
'''
POLICY = (
    "Preparation only, no GPU owner, queue, launch or production release. Eighteen distinct "
    "family/operand-mode/NS-budget diagnostic cases use the lower LR and unchanged 5120-update "
    "schedule, continuous24/interrupted22/resumed24. Require original NS CUDA diagnostic and "
    "independently reconstructed terminal/restoration first, then exact source-bound ownership, "
    "actual-config compiled/cached-root all-rank checkpoints, loss continuation and fused/null "
    "captured-row/update/work gates. A changed finite map is not an equivalent speedup or a "
    "continuation correction. Fifty-four seed0 full-budget configs reserve equal three-point LR "
    "grids for all eighteen cases; failed cases stay recorded and cannot launch production. "
    "Use final-half shared mixture-selection evaluations for LR selection, five complete "
    "536870912-token training windows, all exact shared evaluations, endpoint and final two "
    "windows for reporting; C4 is reporting-only. Symmetric LR expansion is required for a "
    "boundary winner. Confirm selected arms on fresh paired seeds3/4/5. Kernel timing excludes "
    "compilation/capture/transfers/training and is not whole-training efficiency. Full production "
    "histories, matched timing and verified private SwanLab imports remain required."
)


def accepted_parent(root: Path) -> dict:
    path = root / PARENT_CPU
    if sha256(path) != PARENT_CPU_SHA256:
        raise ValueError("NS full-model accepted diagnostic-owner archive changed")
    return owner.require_cpu(root, path)


def parent(root: Path) -> dict:
    accepted_parent(root)
    return original.parent(root)


def config_name(family: str, mode: str, steps: int, label: str) -> str:
    return f"experiment/configs/{family}_ns_{mode}_q{steps}_full_lr_{label}_0p5m_seed0.json"


def configurations(previous: Path) -> dict[str, dict]:
    """Derive equal LR grids while changing only matrix LR, step budget and operand mode."""
    configs = {}
    for family in FAMILIES:
        reference = read(previous / f"experiment/configs/{family}_full_lr_lower_0p5m_seed0.json")
        if (
            reference["experiment"]["arm"] != family
            or reference["optimizer"]["lr"] != LR_GRID[0][1]
            or reference["optimizer"]["ns_steps"] != 10
            or reference["optimizer"]["ns_mode"] != "tpu"
            or reference["training"]["steps"] != 5120
            or reference["training"]["tokens"] != TOKENS
            or reference["training"]["batch_tokens"] != 524288
            or reference["training"]["world_size"] != 8
            or reference["training"]["seed"] != 0
            or reference["experiment"]["refresh_install_age"] != 1
            or reference["runtime"]["compile"] != "max-autotune-no-cudagraphs"
            or reference["runtime"]["compile_options"]["emulate_precision_casts"] is not True
            or reference["root"]["graphs"] is not True
        ):
            raise ValueError("NS full-model requires the preserved lower-LR full-schedule families")
        for mode in MODES:
            for steps in STEPS:
                for label, lr in LR_GRID:
                    config = copy.deepcopy(reference)
                    config["optimizer"].update(lr=lr, ns_steps=steps, ns_mode=mode)
                    configs[config_name(family, mode, steps, label)] = config
    return configs


def trial_specs() -> list[dict]:
    return [
        {
            "family": family,
            "mode": mode,
            "steps": steps,
            "config_name": config_name(family, mode, steps, "lower"),
            "lr": LR_GRID[0][1],
            "trajectories": {"continuous": 24, "interrupted": 22, "resumed": [22, 24]},
            "planned_updates": 48,
            "planned_tokens": 48 * 524288,
            "fused_capture_update_work_gates": family != "muont",
        }
        for family in FAMILIES
        for mode in MODES
        for steps in STEPS
    ]


def contents(previous: Path) -> dict[str, bytes]:
    return {
        OWNER_PATH: DENIED_OWNER.encode(),
        **{
            name: (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
            for name, config in configurations(previous).items()
        },
    }


def helper_sources(root: Path) -> dict[str, Path]:
    inherited = accepted_parent(root)
    sources = {name: Path(inherited["helpers"]) / name for name in inherited["helpers_sha256"]}
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("NS full-model helper is absent or aliased")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("NS full-model imports outside its accepted source closure")
    return sources


def record(root: Path, candidate: Path, frozen: dict) -> dict:
    previous = Path(frozen["candidate"])
    return {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_binding_sha256": sha256(root / original.PARENT_BINDING),
        "parent_cpu_receipt_sha256": PARENT_CPU_SHA256,
        "parent_metadata_sha256": {path.name: sha256(path) for path in previous.glob("*.json")},
        "files_sha256": candidate_files(candidate),
        "changed_files": sorted(contents(previous)),
        "helpers_sha256": {name: sha256(path) for name, path in helper_sources(root).items()},
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "trials": trial_specs(),
        "planned_diagnostic_tokens": sum(item["planned_tokens"] for item in trial_specs()),
        "full_budget_config_count": len(configurations(previous)),
        "planned_full_grid_tokens": len(configurations(previous)) * TOKENS,
        "training_window_tokens": WINDOW_TOKENS,
        "reserved_confirmation_seeds": [3, 4, 5],
        "required_predecessor_terminal": "run-preparation/ns-budget-terminal-independent-20261119.json",
        "scope": POLICY,
    }


def prepare(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(strict=True), candidate.resolve()
    frozen = parent(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or previous.is_relative_to(candidate):
        raise FileExistsError("NS full-model preparation requires an unused distinct source identity")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, payload in contents(previous).items():
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    prepared = record(root, candidate, frozen)
    changed = {
        name
        for name in prepared["files_sha256"].keys() | frozen["files_sha256"].keys()
        if prepared["files_sha256"].get(name) != frozen["files_sha256"].get(name)
    }
    if changed != set(contents(previous)):
        raise ValueError("NS full-model source changed outside new configs and denied owner")
    write(candidate / PREPARATION, prepared)
    return require_preparation(root, candidate)


def require_preparation(root: Path, candidate: Path) -> dict:
    frozen, prepared = parent(root), read(candidate / PREPARATION)
    previous = Path(frozen["candidate"])
    if (
        candidate != candidate.resolve()
        or candidate.is_symlink()
        or {key: value for key, value in prepared.items() if key != "time"} != record(root, candidate, frozen)
        or any((candidate / name).read_bytes() != payload for name, payload in contents(previous).items())
        or any(
            digest != frozen["files_sha256"].get(name)
            for name, digest in prepared["files_sha256"].items()
            if name not in contents(previous)
        )
        or any(sha256(candidate / name) != digest for name, digest in prepared["parent_metadata_sha256"].items())
    ):
        raise ValueError("NS full-model exact transformation or preserved parent differs")
    return prepared


def inventory(prepared: dict) -> dict:
    return (
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: sha256(Path(prepared["candidate"]) / PREPARATION)}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()}
    )


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        owner.PYTHON,
        "--with",
        "pytest-timeout",
        "python",
        "-P",
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(out),
        "-p",
        "no:cacheprovider",
        "-p",
        "root_diagnostic_pytest_guard",
        "--timeout=60",
        "--session-timeout=600",
        "--import-mode=prepend",
        "-m",
        original.SAFE_MARKERS,
        "-q",
        str(out / "tracking" / NEW_HELPERS[-1]),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or out.exists() or out.is_relative_to(candidate):
        raise RuntimeError("NS full-model CPU acceptance requires hidden CUDA and unused external evidence")
    prepared, sources = require_preparation(root, candidate), helper_sources(root)
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    for name in inventory(prepared):
        if name.startswith("tracking/"):
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if (candidate / name).is_symlink():
            target.symlink_to((candidate / name).readlink())
        else:
            shutil.copy2(candidate / name, target)
    inherited = accepted_parent(root)
    parent_owner = Path(prepared["parent_candidate"]) / OWNER_PATH
    target_owner = out / PARENT_OWNER_FIXTURE
    target_owner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_owner, target_owner)
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for name, digest in inherited["fixtures_sha256"].items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != digest:
                raise ValueError("NS full-model inherited fixture conflicts with preserved source")
        else:
            shutil.copy2((root / PARENT_CPU).parent / name, target)
    log, error = out / "cpu.log", None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(out),
                cwd=out,
                env=os.environ
                | owner.ENVIRONMENT
                | {
                    "UV_OFFLINE": "1",
                    "PYTHONPATH": f"{helpers}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "NS full-model CPU tests timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    fixtures = inherited["fixtures_sha256"] | {PARENT_OWNER_FIXTURE: sha256(parent_owner)}
    expected = inventory(prepared) | fixtures
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            bundle.add(out / name, arcname=name, recursive=False)
        bundle.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    require_archive(archive, expected, alias=True)
    unchanged = require_preparation(root, candidate) == prepared
    result = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None and unchanged else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": prepared["helpers_sha256"],
        "files_sha256": prepared["files_sha256"],
        "fixtures_sha256": fixtures,
        "inventory_sha256": expected,
        "preparation_sha256": sha256(candidate / PREPARATION),
        "parent_cpu_receipt_sha256": PARENT_CPU_SHA256,
        "command": cpu_command(out),
        "cwd": str(out),
        "environment": owner.ENVIRONMENT,
        "exit_code": code,
        "cuda_visible_devices": "",
        "scope": POLICY,
        **{name: str(path) for name, path in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))},
        **{
            f"{name}_sha256": sha256(path)
            for name, path in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
            if path.is_file()
        },
    }
    if error:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"NS full-model CPU failure preserved at {out}")
    return result


def require_cpu(root: Path, path: Path) -> dict:
    receipt = read(path)
    prepared = require_preparation(root, Path(receipt["candidate"]))
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != CPU_STATUS
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or receipt["root"] != str(root)
        or receipt["preparation_sha256"] != sha256(Path(receipt["candidate"]) / PREPARATION)
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["fixtures_sha256"]
        != accepted_parent(root)["fixtures_sha256"]
        | {PARENT_OWNER_FIXTURE: sha256(Path(prepared["parent_candidate"]) / OWNER_PATH)}
        or receipt["inventory_sha256"] != inventory(prepared) | receipt["fixtures_sha256"]
        or receipt["command"] != cpu_command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != owner.ENVIRONMENT
        or receipt["helpers"] != str(path.parent / "tracking")
        or not cases
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(NEW_HELPERS[-1]).stem}
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(sha256(Path(receipt[name])) != receipt[f"{name}_sha256"] for name in ("archive", "tests", "log"))
        or any(sha256(path.parent / name) != digest for name, digest in receipt["inventory_sha256"].items())
    ):
        raise ValueError("NS full-model CPU/archive coverage or source identity differs")
    require_archive(
        Path(receipt["archive"]),
        receipt["inventory_sha256"],
        alias=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("prepare", "cpu", "check"))
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in owner.ENVIRONMENT.items()):
        raise RuntimeError("NS full-model preparation requires its fixed CUDA-hidden environment")
    root = args.root.resolve(strict=True)
    if args.phase == "check":
        if args.cpu_receipt is None:
            parser.error("Checking requires --cpu-receipt")
        result = require_cpu(root, args.cpu_receipt.resolve(strict=True))
    else:
        if args.candidate is None or (args.phase == "cpu" and args.out is None):
            parser.error("Preparation requires --candidate; CPU acceptance also requires --out")
        candidate = args.candidate.resolve()
        result = prepare(root, candidate) if args.phase == "prepare" else cpu(root, candidate, args.out.resolve())
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
