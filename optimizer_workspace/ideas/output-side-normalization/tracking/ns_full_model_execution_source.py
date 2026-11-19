# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and archive a distinct execution owner while preserving the accepted validator."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_full_model_contract as contract
import prepare_ns_full_model_validation as previous
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

PARENT = "run-preparation/ns-full-model-validation-component-cpu/single-pass-20261119T0132/receipt.json"
PARENT_SHA256 = "8cc1fb7f84a13165af61acef29325c25aba1010224ce9cafff68366cecf6a359"
PREPARATION = "ns-full-model-owner-preparation.json"
PREPARED_STATUS = "ns_full_model_execution_owner_prepared_gpu_unissued"
CPU_STATUS = "ns_full_model_execution_owner_cpu_archive_verified_gpu_unissued"
OWNER = "experiment/fused_attribution_owner.py"
NEW_HELPERS = (
    "ns_full_model_execution_source.py",
    "ns_full_model_predecessor.py",
    "ns_full_model_gpu_owner.py",
    "ns_full_model_gpu_controller.py",
    "test_ns_full_model_gpu.py",
)
TESTS = ("test_ns_full_model_validation.py", NEW_HELPERS[-1])
TEST_CASE_COUNTS = {"test_ns_full_model_validation": 28, "test_ns_full_model_gpu": 41}
ADAPTER = '''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load only the separately frozen NS full-model execution owner before CUDA."""

import importlib
import sys
from pathlib import Path

from .selected_lr_release_owner import read, sha256

TASK = Path(__file__).resolve().parents[1]


def execution_owner():
    path = TASK / "ns-full-model-owner-preparation.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("NS full-model execution owner is not separately prepared")
    prepared = read(path)
    root = Path(prepared["root"])
    binding = root / "run-preparation/ns-full-model-binding-20261119.json"
    helpers = root / "controllers/ns-full-model-20261119/tracking"
    frozen = read(binding)
    helper = helpers / "ns_full_model_gpu_owner.py"
    if (
        root != root.resolve(strict=True)
        or prepared["candidate"] != str(TASK)
        or frozen["candidate"] != str(TASK)
        or frozen["helpers"] != str(helpers)
        or helpers.is_symlink()
        or helpers != helpers.resolve(strict=True)
        or frozen["preparation_sha256"] != sha256(path)
        or sha256(helper) != frozen["helpers_sha256"][helper.name]
        or helper.is_symlink()
        or sha256(Path(__file__)) != frozen["files_sha256"]["experiment/fused_attribution_owner.py"]
    ):
        raise RuntimeError("NS full-model candidate adapter or issued helper identity differs")
    sys.path.insert(0, str(helpers))
    owner = importlib.import_module("ns_full_model_gpu_owner")
    if Path(owner.__file__).resolve(strict=True) != helper:
        raise RuntimeError("NS full-model candidate imported a mutable execution owner")
    return root, owner


def require_gpu_owner(root):
    expected, owner = execution_owner()
    if root != expected:
        raise RuntimeError("NS full-model validator root differs")
    return owner.require_gpu_owner(root, TASK)


def require_fused_attribution_owner(args):
    root, owner = execution_owner()
    owner.require_rank(root, TASK, args)
'''
SCOPE = (
    "Separate actual-config execution owner; preserve pinned trainer/root arithmetic and "
    "accepted validator bytes. Require original NS diagnostic terminal/restoration and "
    "queue/worker absence before exact keeper transfer. No production authority. " + contract.SCOPE
)


def parent(root: Path) -> dict:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("NS full-model execution parent CPU receipt changed")
    return previous.require_cpu(root, path)


def execution_sources(inherited: dict) -> dict[str, Path]:
    sources = {name: Path(inherited["helpers"]) / name for name in inherited["helpers_sha256"]}
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("NS full-model execution helper is missing or aliased")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("NS full-model execution helper escapes its accepted import closure")
    return sources


def helper_sources(root: Path) -> dict[str, Path]:
    return execution_sources(parent(root))


def preparation_record(root: Path, candidate: Path, inherited: dict, sources: dict[str, Path]) -> dict:
    original = Path(inherited["candidate"])
    return {
        "status": PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(original),
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "parent_metadata_sha256": {path.name: sha256(path) for path in original.glob("*.json")},
        "files_sha256": candidate_files(candidate),
        "changed_files": [OWNER],
        "helpers_sha256": {name: sha256(path) for name, path in sources.items()},
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "required_predecessor_terminal": "run-preparation/ns-budget-terminal-independent-20261119.json",
        "plan": contract.commands(candidate, root, contract.locations(root)[3]),
        "scope": SCOPE,
    }


def prepare(root: Path, candidate: Path) -> dict:
    inherited = parent(root)
    original, sources = Path(inherited["candidate"]), execution_sources(inherited)
    if (
        candidate != candidate.resolve()
        or candidate.exists()
        or candidate.is_relative_to(original)
        or original.is_relative_to(candidate)
    ):
        raise FileExistsError("NS full-model execution preparation needs an unused separate candidate")
    shutil.copytree(original, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (candidate / OWNER).write_bytes(ADAPTER.encode())
    write(candidate / PREPARATION, preparation_record(root, candidate, inherited, sources))
    return require_candidate(root, candidate, inherited, sources)


def require_candidate(root: Path, candidate: Path, inherited: dict, sources: dict[str, Path]) -> dict:
    raw = (candidate / PREPARATION).read_bytes()
    prepared = read(candidate / PREPARATION)
    original = Path(inherited["candidate"])
    files, original_files = candidate_files(candidate), candidate_files(original)
    if (
        candidate != candidate.resolve()
        or candidate.is_symlink()
        or {key: value for key, value in prepared.items() if key != "time"}
        != preparation_record(root, candidate, inherited, sources)
        or (candidate / OWNER).read_bytes() != ADAPTER.encode()
        or {name for name in files.keys() | original_files.keys() if files.get(name) != original_files.get(name)}
        != {OWNER}
        or any(sha256(candidate / name) != digest for name, digest in prepared["parent_metadata_sha256"].items())
        or (candidate / PREPARATION).read_bytes() != raw
    ):
        raise ValueError("NS full-model exact owner-only transformation or preserved parent changed")
    return prepared


def require_preparation(root: Path, candidate: Path) -> dict:
    inherited = parent(root)
    return require_candidate(root, candidate, inherited, execution_sources(inherited))


def candidate_inventory(inherited: dict, prepared: dict, candidate: Path) -> dict:
    return (
        inherited["inventory_sha256"]
        | prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: sha256(candidate / PREPARATION)}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()}
    )


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        contract.PYTHON,
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
        previous.parent.original.SAFE_MARKERS,
        "-q",
        *(str(out / "tracking" / name) for name in TESTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    raw, cpu = path.read_bytes(), read(path)
    candidate = Path(cpu["candidate"])
    inherited = parent(root)
    prepared = require_candidate(root, candidate, inherited, execution_sources(inherited))
    expected = candidate_inventory(inherited, prepared, candidate)
    suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        cpu["status"] != CPU_STATUS
        or cpu["root"] != str(root)
        or cpu["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or cpu["helpers"] != str(path.parent / "tracking")
        or cpu["helpers_sha256"] != prepared["helpers_sha256"]
        or cpu["inventory_sha256"] != expected
        or cpu["preparation_sha256"] != sha256(candidate / PREPARATION)
        or cpu["files_sha256"] != prepared["files_sha256"]
        or cpu["command"] != cpu_command(path.parent)
        or cpu["cwd"] != str(path.parent)
        or cpu["environment"] != previous.parent.owner.ENVIRONMENT
        or cpu["cuda_visible_devices"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or not cases
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(name).stem for name in TESTS}
        or {name: sum(case.attrib["classname"].split(".")[-1] == name for case in cases) for name in TEST_CASE_COUNTS}
        != TEST_CASE_COUNTS
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            cpu[name] != str(path.parent / filename) or cpu[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in expected.items())
    ):
        raise ValueError("NS full-model execution CPU source, command, inventory or complete XML differs")
    require_helper_bytes(path.parent / "tracking", cpu["helpers_sha256"])
    require_archive(Path(cpu["archive"]), expected, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("NS full-model execution CPU receipt changed during checking")
    return cpu


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    environment = previous.parent.owner.ENVIRONMENT
    if (
        any(os.environ.get(name) != value for name, value in environment.items())
        or out.exists()
        or out.is_relative_to(candidate)
        or candidate.is_relative_to(out)
    ):
        raise RuntimeError("NS full-model owner CPU staging needs fixed hidden CUDA and unused output")
    inherited = parent(root)
    sources = execution_sources(inherited)
    prepared = require_candidate(root, candidate, inherited, sources)
    expected = candidate_inventory(inherited, prepared, candidate)
    out.mkdir(parents=True)
    (out / "artifacts").mkdir()
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for name, digest in expected.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        if name.startswith("tracking/"):
            original = sources[Path(name).name]
        elif name in prepared["files_sha256"] or name in prepared["parent_metadata_sha256"] or name == PREPARATION:
            original = candidate / name
        else:
            original = (root / PARENT).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha256(target) != digest:
            raise ValueError("NS full-model execution staged source or fixture changed")
    log, error = out / "cpu.log", None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(out),
                cwd=out,
                env=os.environ
                | environment
                | {
                    "UV_OFFLINE": "1",
                    "PYTHONPATH": f"{out / 'tracking'}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "NS full-model execution CPU suite timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            bundle.add(out / name, arcname=name, recursive=False)
        bundle.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "preparation_sha256": sha256(candidate / PREPARATION),
        "files_sha256": prepared["files_sha256"],
        "helpers": str(out / "tracking"),
        "helpers_sha256": prepared["helpers_sha256"],
        "inventory_sha256": expected,
        "command": cpu_command(out),
        "cwd": str(out),
        "environment": environment,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": SCOPE,
        **{name: str(path) for name, path in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))},
        **{
            f"{name}_sha256": sha256(path)
            for name, path in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
            if path.is_file()
        },
    }
    if error:
        receipt["error"] = error
    write(out / "receipt.json", receipt)
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        receipt.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", receipt)
        raise RuntimeError(f"NS full-model owner CPU failure preserved at {out}") from failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("prepare", "cpu", "check"))
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.phase == "check":
        if args.cpu_receipt is None or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            parser.error("Checking requires hidden CUDA and --cpu-receipt")
        result = require_cpu(root, args.cpu_receipt.resolve(strict=True))
    else:
        if args.candidate is None:
            parser.error("Preparation/CPU phases require --candidate")
        candidate = args.candidate.resolve()
        if args.phase == "prepare":
            result = prepare(root, candidate)
        else:
            if args.out is None:
                parser.error("CPU acceptance requires --out")
            result = cpu(root, candidate, args.out.resolve())
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
