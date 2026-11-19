# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept a distinct actual-LR execution closure while retaining the accepted adapter."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_execution_source as adapter
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_release_owner import SAFE_MARKERS, candidate_files, read, sha256
from watch_wave5 import write

PARENT = "run-preparation/ns-training-path-actual-lr-adapter-cpu/20261119T1020/receipt.json"
PARENT_SHA256 = "82c1ad5b154041485d80333954260e7b8b2b67aca7feedaf3868ca3e87dee5f1"
PREPARATION = adapter.PREPARATION
OWNER = adapter.OWNER
CPU_STATUS = "ns_training_path_actual_lr_execution_owner_cpu_archive_verified_gpu_unissued"
ENVIRONMENT = adapter.ENVIRONMENT
NEW_HELPERS = (
    "ns_training_path_lr_gpu_source.py",
    "ns_training_path_lr_gpu_owner.py",
    "ns_training_path_lr_gpu_controller.py",
    "ns_training_path_lr_gpu_test_support.py",
    "test_ns_training_path_lr_gpu.py",
    "test_ns_training_path_lr_gpu_source.py",
)
TEST_COUNTS = {"test_ns_training_path_lr_gpu": 45, "test_ns_training_path_lr_gpu_source": 8}
SCOPE = (
    "Distinct actual-LR training-path execution owner. Preserve the accepted 171-helper "
    "adapter archive and candidate byte-for-byte; add only the separate execution closure. "
    "Freeze requires a caller-bound original canonical terminal seal with all18 decisions/"
    "54 commands, retained rejections, original controller/validator/waiter/descendant absence "
    "and exact restored keeper. Visible ranks authenticate an independently accepted live "
    "hidden-CPU parent proof and operation inode inventory; they never invoke hidden-only "
    "predecessor checks. Physical CUDA UUIDs, node/manager locks, exact bounded launcher/rank "
    "ancestry, cleanup/restoration and a canonical one-shot terminal remain required. "
    "CPU acceptance grants no GPU transfer, production, quality/efficiency or SwanLab result. " + adapter.contract.SCOPE
)


def inherited(root: Path) -> dict:
    """Authenticate the unchanged adapter archive and its accepted parent chain."""
    previous = accepted_archive(root, (PARENT, PARENT_SHA256, adapter.CPU_STATUS, adapter.TEST_COUNT))
    if adapter.require_cpu(root, root / PARENT) != previous:
        raise ValueError("Actual-LR execution owner adapter parent authentication differs")
    return previous


def execution_sources(previous: dict) -> dict[str, Path]:
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    if set(sources).intersection(NEW_HELPERS):
        raise ValueError("Actual-LR execution closure would overwrite accepted parent helpers")
    local = Path(__file__).resolve().parent
    sources.update({name: local / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or path != path.resolve(strict=True):
            raise ValueError("Actual-LR execution helper is aliased")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Actual-LR execution requires a complete flat import closure")
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Actual-LR execution import escapes accepted closure")
    if any(sha256(sources[name]) != digest for name, digest in previous["helpers_sha256"].items()):
        raise ValueError("Actual-LR execution inherited helper bytes changed")
    return sources


def require_candidate(root: Path, candidate: Path, previous: dict) -> dict:
    """Retain every accepted candidate and preparation byte, including the adapter."""
    archived = Path(previous["helpers"]).parent
    expected = {
        name: digest
        for name, digest in previous["inventory_sha256"].items()
        if name.startswith(("experiment/", "artifacts/pinned-sources-materialized/"))
    }
    prepared = read(candidate / PREPARATION)
    metadata = prepared["parent_metadata_sha256"] | {PREPARATION: previous["inventory_sha256"][PREPARATION]}
    if (
        root != root.resolve(strict=True)
        or candidate != candidate.resolve(strict=True)
        or str(candidate) != previous["candidate"]
        or candidate_files(candidate) != expected
        or candidate_files(archived) != expected
        or prepared["status"] != adapter.PREPARED_STATUS
        or prepared["root"] != str(root)
        or prepared["candidate"] != str(candidate)
        or prepared["files_sha256"] != expected
        or prepared["plan"] != adapter.validation.plan(root, candidate)
        or prepared["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or (candidate / OWNER).read_bytes() != adapter.ADAPTER.encode()
        or {path.name for path in candidate.glob("*.json")} != set(metadata)
        or any(sha256(candidate / name) != digest for name, digest in metadata.items())
    ):
        raise ValueError("Actual-LR execution candidate differs from independently accepted adapter archive")
    adapter.validation.require_numerical_sources(candidate)
    return prepared


def require_preparation(root: Path, candidate: Path) -> dict:
    return require_candidate(root, candidate, inherited(root))


def inventory(previous: dict, sources: dict[str, Path]) -> dict:
    return previous["inventory_sha256"] | {f"tracking/{name}": sha256(path) for name, path in sources.items()}


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        adapter.contract.PYTHON,
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
        *(str(out / "tracking" / f"{name}.py") for name in TEST_COUNTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    identity, raw, cpu = file_identity(path), path.read_bytes(), read(path)
    previous = inherited(root)
    candidate = Path(cpu["candidate"])
    prepared = require_candidate(root, candidate, previous)
    sources = execution_sources(previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    expected = inventory(previous, sources)
    suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        cpu["status"] != CPU_STATUS
        or cpu["root"] != str(root)
        or cpu["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or cpu["adapter_cpu_receipt"] != str(root / PARENT)
        or cpu["preparation_sha256"] != sha256(candidate / PREPARATION)
        or cpu["files_sha256"] != prepared["files_sha256"]
        or cpu["helpers"] != str(path.parent / "tracking")
        or cpu["helpers_sha256"] != hashes
        or cpu["inventory_sha256"] != expected
        or cpu["command"] != cpu_command(path.parent)
        or cpu["cwd"] != str(path.parent)
        or cpu["environment"] != ENVIRONMENT
        or cpu["cuda_visible_devices"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["scope"] != SCOPE
        or len(cases) != sum(TEST_COUNTS.values())
        or {case.attrib["classname"].split(".")[-1] for case in cases} != set(TEST_COUNTS)
        or {name: sum(case.attrib["classname"].split(".")[-1] == name for case in cases) for name in TEST_COUNTS}
        != TEST_COUNTS
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
        raise ValueError("Actual-LR execution complete CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(cpu["archive"]), expected, alias=True)
    if path.read_bytes() != raw or file_identity(path) != identity:
        raise ValueError("Actual-LR execution CPU receipt changed during checking")
    return cpu


def cpu(root: Path, out: Path) -> dict:
    adapter.require_hidden_cuda()
    previous = inherited(root)
    candidate = Path(previous["candidate"])
    prepared = require_candidate(root, candidate, previous)
    sources = execution_sources(previous)
    expected = inventory(previous, sources)
    if out != out.resolve() or out.exists() or out.is_symlink() or out.is_relative_to(candidate):
        raise FileExistsError("Actual-LR execution CPU acceptance requires unused external output")
    out.mkdir(parents=True)
    for name, digest in expected.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        original = sources[Path(name).name] if name.startswith("tracking/") else Path(previous["helpers"]).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha256(target) != digest:
            raise ValueError("Actual-LR execution bytes changed during staging")
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
            code, error = 124, "Actual-LR execution CPU acceptance timed out"
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
        "adapter_cpu_receipt": str(root / PARENT),
        "preparation_sha256": sha256(candidate / PREPARATION),
        "files_sha256": prepared["files_sha256"],
        "helpers": str(out / "tracking"),
        "helpers_sha256": {name: sha256(path) for name, path in sources.items()},
        "inventory_sha256": expected,
        "command": cpu_command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
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
        raise RuntimeError(f"Actual-LR execution CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Actual-LR execution acceptance requires its archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Actual-LR execution imported mutable helper: {name}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "check"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    adapter.require_hidden_cuda()
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Archived checking requires --cpu-receipt")
        result = accepted(root, args.cpu_receipt.resolve(strict=True))
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
