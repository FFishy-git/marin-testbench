# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a distinct actual-LR adapter without issuing an execution owner or GPU binding."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_contract as contract
import ns_training_path_lr_predecessor as predecessor
import ns_training_path_lr_validation as validation
import prepare_ns_training_path_lr_predecessor as parent
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_ns_training_path_lr_validation import ENVIRONMENT
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_release_owner import SAFE_MARKERS, candidate_files, read, sha256
from watch_wave5 import write

PARENT = "run-preparation/ns-training-path-actual-lr-predecessor-cpu/20261119T0953/receipt.json"
PARENT_SHA256 = "2ad5cee992e97594b0a3b9d79680e6fa273435dbb6885bafbb2fa82d19b36230"
VERIFIER_SHA256 = "8d7a21a8c77f1906496e6436371af1c8ae81238c7afa683f4e048fda30077d75"
CHECKER_SHA256 = "0ecbecb44647b99d690b5437c34981fcdb5a6e1c60a8b093a008b647a47ae170"
PREPARATION = "ns-training-path-actual-lr-owner-preparation.json"
PREPARED_STATUS = "ns_training_path_actual_lr_adapter_prepared_execution_owner_unissued"
CPU_STATUS = "ns_training_path_actual_lr_adapter_cpu_archive_verified_execution_owner_unissued"
OWNER = "experiment/fused_attribution_owner.py"
PARENT_OWNER = "fixtures/ns-training-path-actual-lr-parent-owner.py"
NEW_HELPERS = ("ns_training_path_lr_execution_source.py", "test_ns_training_path_lr_execution_source.py")
TEST_MODULE = "test_ns_training_path_lr_execution_source"
TEST_COUNT = 14
SCOPE = (
    "Owner-only source preparation for the distinct training-path actual-LR diagnostic. "
    "Preserve all original training, NS compiler-boundary, configuration, pinned-source, "
    "data and accepted numerical/predecessor helper bytes. Change only the candidate owner "
    "adapter, retain its parent bytes, and reserve exactly54 configurations/162 commands. "
    "No execution owner/controller, original terminal acceptance, predecessor seal, GPU binding, "
    "allocation transfer, launch, production, optimizer quality, elapsed-efficiency or SwanLab "
    "authority is issued by this component. A subsequent separately accepted execution owner "
    "must bind the original completed canonical terminal, retained rejections, owner/waiter/"
    "descendant absence and exact restored keeper before freeze or transfer. " + contract.SCOPE
)
ADAPTER = '''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load only a separately issued actual-LR training-path owner before CUDA."""

import importlib
import sys
from pathlib import Path

from .selected_lr_release_owner import read, sha256

TASK = Path(__file__).resolve().parents[1]


def execution_owner():
    path = TASK / "ns-training-path-actual-lr-owner-preparation.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Training-path actual-LR adapter is not separately prepared")
    prepared = read(path)
    root = Path(prepared["root"])
    binding = root / "run-preparation/ns-training-path-actual-lr-binding-20261119.json"
    if not binding.is_file() or binding.is_symlink() or binding.stat().st_nlink != 1:
        raise RuntimeError("Training-path actual-LR execution owner is unissued")
    helpers = root / "controllers/ns-training-path-actual-lr-20261119/tracking"
    frozen = read(binding)
    helper = helpers / "ns_training_path_lr_gpu_owner.py"
    if (
        root != root.resolve(strict=True)
        or prepared["candidate"] != str(TASK)
        or prepared["status"] != "ns_training_path_actual_lr_adapter_prepared_execution_owner_unissued"
        or frozen["status"] != "frozen_ns_training_path_actual_lr_validation_not_production"
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(TASK)
        or frozen["helpers"] != str(helpers)
        or helpers.is_symlink()
        or helpers != helpers.resolve(strict=True)
        or frozen["preparation_sha256"] != sha256(path)
        or helper.is_symlink()
        or sha256(helper) != frozen["helpers_sha256"][helper.name]
        or sha256(Path(__file__)) != frozen["files_sha256"]["experiment/fused_attribution_owner.py"]
    ):
        raise RuntimeError("Training-path actual-LR adapter or issued owner identity differs")
    sys.path.insert(0, str(helpers))
    owner = importlib.import_module("ns_training_path_lr_gpu_owner")
    if Path(owner.__file__).resolve(strict=True) != helper:
        raise RuntimeError("Training-path actual-LR adapter imported a mutable owner")
    return root, owner


def require_ns_training_path_lr_gpu_owner(root):
    expected, owner = execution_owner()
    if root != expected:
        raise RuntimeError("Training-path actual-LR validator root differs")
    return owner.require_gpu_owner(root, TASK)


def require_fused_attribution_owner(args):
    root, owner = execution_owner()
    owner.require_rank(root, TASK, args)
'''


def require_hidden_cuda() -> None:
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Training-path actual-LR adapter preparation requires fixed hidden CUDA")


def inherited(root: Path) -> dict:
    previous = accepted_archive(root, (PARENT, PARENT_SHA256, parent.STATUS, 12))
    if (
        previous["environment"] != ENVIRONMENT
        or previous["scope"] != predecessor.SCOPE
        or previous["helpers_sha256"]["ns_training_path_lr_predecessor.py"] != VERIFIER_SHA256
        or previous["helpers_sha256"]["prepare_ns_training_path_lr_predecessor.py"] != CHECKER_SHA256
        or previous["parent_cpu_receipt_sha256"] != parent.PARENT_SHA256
    ):
        raise ValueError("Training-path actual-LR adapter parent or predecessor provenance differs")
    return previous


def execution_sources(previous: dict) -> dict[str, Path]:
    """Preserve inherited helpers and reject new imports outside the complete flat archive."""
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    if set(sources).intersection(NEW_HELPERS):
        raise ValueError("Training-path actual-LR adapter would replace an accepted helper")
    local = Path(__file__).resolve().parent
    sources.update({name: local / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file() or path != path.resolve(strict=True):
            raise ValueError("Training-path actual-LR adapter helper is absent or aliased")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Training-path actual-LR adapter requires a complete flat import closure")
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Training-path actual-LR adapter import escapes its accepted closure")
    if any(sha256(sources[name]) != digest for name, digest in previous["helpers_sha256"].items()):
        raise ValueError("Training-path actual-LR adapter inherited helper changed")
    return sources


def parent_files(previous: dict) -> dict:
    """Anchor the live parent to the complete accepted source/config archive, not its metadata."""
    archived, original = Path(previous["helpers"]).parent, Path(previous["candidate"])
    expected = {
        name: digest
        for name, digest in previous["inventory_sha256"].items()
        if name.startswith(("experiment/", "artifacts/pinned-sources-materialized/"))
    }
    if candidate_files(archived) != expected or candidate_files(original) != expected:
        raise ValueError("Training-path actual-LR adapter accepted parent source/config inventory changed")
    metadata = {path.name: previous["inventory_sha256"].get(path.name) for path in original.glob("*.json")}
    if any(digest is None or sha256(original / name) != digest for name, digest in metadata.items()):
        raise ValueError("Training-path actual-LR adapter accepted parent metadata changed")
    return expected


def preparation_record(root: Path, candidate: Path, previous: dict, sources: dict[str, Path]) -> dict:
    parent_files(previous)
    validation.require_numerical_sources(candidate)
    for module in (contract, validation, predecessor, parent):
        path = Path(module.__file__)
        if path.is_symlink() or sha256(path) != sha256(sources[path.name]):
            raise ValueError("Training-path actual-LR adapter imported a changed accepted component")
    original = Path(previous["candidate"])
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
        "predecessor_cpu_receipt_sha256": PARENT_SHA256,
        "predecessor_verifier_sha256": VERIFIER_SHA256,
        "predecessor_checker_sha256": CHECKER_SHA256,
        "required_predecessor_seal": str(root / predecessor.SEAL),
        "plan": validation.plan(root, candidate),
        "scope": SCOPE,
    }


def require_candidate(root: Path, candidate: Path, previous: dict, sources: dict[str, Path]) -> dict:
    path = candidate / PREPARATION
    identity, raw, prepared = file_identity(path), path.read_bytes(), read(path)
    files, original_files = candidate_files(candidate), parent_files(previous)
    if (
        root != root.resolve(strict=True)
        or candidate != candidate.resolve(strict=True)
        or candidate.is_symlink()
        or candidate == Path(previous["candidate"])
        or candidate.is_relative_to(Path(previous["candidate"]))
        or Path(previous["candidate"]).is_relative_to(candidate)
        or {key: value for key, value in prepared.items() if key != "time"}
        != preparation_record(root, candidate, previous, sources)
        or (candidate / OWNER).read_bytes() != ADAPTER.encode()
        or {name for name in files.keys() | original_files.keys() if files.get(name) != original_files.get(name)}
        != {OWNER}
        or (candidate / PREPARATION).read_bytes() != raw
        or file_identity(path) != identity
        or any(sha256(candidate / name) != digest for name, digest in prepared["parent_metadata_sha256"].items())
    ):
        raise ValueError("Training-path actual-LR adapter escapes its exact accepted owner-only transformation")
    return prepared


def prepare(root: Path, candidate: Path) -> dict:
    require_hidden_cuda()
    previous = inherited(root)
    original, sources = Path(previous["candidate"]), execution_sources(previous)
    parent_files(previous)
    if (
        candidate != candidate.resolve()
        or candidate.exists()
        or candidate.is_symlink()
        or candidate.is_relative_to(original)
        or original.is_relative_to(candidate)
    ):
        raise FileExistsError("Training-path actual-LR adapter preparation requires an unused separate candidate")
    shutil.copytree(original, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (candidate / OWNER).write_bytes(ADAPTER.encode())
    write(candidate / PREPARATION, preparation_record(root, candidate, previous, sources))
    return require_candidate(root, candidate, previous, sources)


def require_preparation(root: Path, candidate: Path) -> dict:
    previous = inherited(root)
    return require_candidate(root, candidate, previous, execution_sources(previous))


def inventory(previous: dict, prepared: dict, candidate: Path) -> dict:
    return (
        previous["inventory_sha256"]
        | prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: sha256(candidate / PREPARATION)}
        | {PARENT_OWNER: previous["inventory_sha256"][OWNER]}
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
        SAFE_MARKERS,
        "-q",
        str(out / "tracking" / f"{TEST_MODULE}.py"),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    if TEST_COUNT < 1:
        raise RuntimeError("Training-path actual-LR adapter CPU coverage is not yet bound")
    identity, raw, receipt = file_identity(path), path.read_bytes(), read(path)
    previous, candidate = inherited(root), Path(receipt["candidate"])
    prepared = require_candidate(root, candidate, previous, execution_sources(previous))
    expected = inventory(previous, prepared, candidate)
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["inventory_sha256"] != expected
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["command"] != cpu_command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != SCOPE
        or len(cases) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {TEST_MODULE}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != TEST_COUNT
        or sum(int(suite.attrib["tests"]) for suite in suites) != TEST_COUNT
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in expected.items())
    ):
        raise ValueError("Training-path actual-LR adapter CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", receipt["helpers_sha256"])
    require_archive(Path(receipt["archive"]), expected, alias=True)
    if path.read_bytes() != raw or file_identity(path) != identity:
        raise ValueError("Training-path actual-LR adapter CPU receipt changed during checking")
    return receipt


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    require_hidden_cuda()
    if (
        out != out.resolve()
        or out.exists()
        or out.is_symlink()
        or out.is_relative_to(candidate)
        or candidate.is_relative_to(out)
    ):
        raise RuntimeError("Training-path actual-LR adapter CPU archive requires unused external output")
    previous = inherited(root)
    sources = execution_sources(previous)
    prepared = require_candidate(root, candidate, previous, sources)
    expected = inventory(previous, prepared, candidate)
    out.mkdir(parents=True)
    for name, digest in expected.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        if name.startswith("tracking/"):
            original = sources[Path(name).name]
        elif name == PARENT_OWNER:
            original = Path(previous["helpers"]).parent / OWNER
        elif name in prepared["files_sha256"] or name in prepared["parent_metadata_sha256"] or name == PREPARATION:
            original = candidate / name
        else:
            original = Path(previous["helpers"]).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha256(target) != digest:
            raise ValueError("Training-path actual-LR adapter bytes changed during staging")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    log, error = out / "cpu.log", None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(out),
                cwd=out,
                env=os.environ
                | ENVIRONMENT
                | {
                    "UV_OFFLINE": "1",
                    "PYTHONPATH": f"{out / 'tracking'}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
                },
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Training-path actual-LR adapter CPU suite timed out"
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
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": SCOPE,
        **{name: str(item) for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))},
        **{
            f"{name}_sha256": sha256(item)
            for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
            if item.is_file()
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
        raise RuntimeError(f"Training-path actual-LR adapter CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Training-path actual-LR adapter requires its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Training-path actual-LR adapter imported a mutable helper: {name}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("prepare", "cpu", "check"))
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    require_hidden_cuda()
    root = args.root.resolve(strict=True)
    if args.phase == "check":
        if args.cpu_receipt is None:
            parser.error("Archived checking requires --cpu-receipt")
        result = accepted(root, args.cpu_receipt.resolve(strict=True))
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
