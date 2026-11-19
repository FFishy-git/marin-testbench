# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept the distinct training-path numerical component without issuing a GPU owner."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_contract as contract
import ns_training_path_predecessor as predecessor
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write

PARENT = (
    "run-preparation/ns-training-path-predecessor-cpu/20261119T0649/receipt.json",
    "398dcf9cb1e63b44092c3aa5a6b49075294fff5c882e16789240d458d3b834a8",
    "ns_training_path_predecessor_cpu_archive_verified_gpu_unissued",
    37,
)
STATUS = "ns_training_path_validation_component_cpu_archive_verified_gpu_unissued"
NEW_HELPERS = (
    "ns_training_path_contract.py",
    "ns_training_path_validation.py",
    "ns_training_path_test_support.py",
    "test_ns_training_path_validation.py",
    "prepare_ns_training_path_validation.py",
)
TESTS = ("test_ns_training_path_validation.py",)
SCOPE = "Separate CPU/source-archive numerical component; execution owner and launch unissued. " + contract.SCOPE


def inherited(root: Path) -> dict:
    return predecessor.accepted_archive(root, PARENT)


def helper_sources(previous: dict) -> dict[str, Path]:
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Training-path validation source is absent or aliased")
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
                raise ValueError("Training-path validation import escapes the accepted source closure")
    for name, digest in contract.NUMERICAL_READERS.items():
        if sha256(sources[name]) != digest:
            raise ValueError("Training-path validation accepted numerical reader changed")
    return sources


def command(out: Path) -> list[str]:
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
        *(str(out / "tracking" / name) for name in TESTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    raw, receipt = path.read_bytes(), read(path)
    previous = inherited(root)
    sources = helper_sources(previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    expected = previous["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != previous["candidate"]
        or receipt["parent_cpu_receipt_sha256"] != PARENT[1]
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != expected
        or receipt["command"] != command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != predecessor.diagnostic.ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != SCOPE
        or len(cases) != 34
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(name).stem for name in TESTS}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in expected.items())
    ):
        raise ValueError("Training-path validation CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), expected, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("Training-path validation CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    environment = predecessor.diagnostic.ENVIRONMENT
    if (
        any(os.environ.get(name) != value for name, value in environment.items())
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("Training-path validation staging requires hidden CUDA and unused external output")
    previous = inherited(root)
    sources = helper_sources(previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    expected = previous["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    out.mkdir(parents=True)
    for name, digest in expected.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        source = sources[Path(name).name] if name.startswith("tracking/") else (root / PARENT[0]).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise ValueError("Training-path validation source changed during staging")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    log, error = out / "cpu.log", None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command(out),
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
            code, error = 124, "Training-path validation CPU suite timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            bundle.add(out / name, arcname=name, recursive=False)
        bundle.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    result = {
        "status": STATUS if code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": previous["candidate"],
        "parent_cpu_receipt_sha256": PARENT[1],
        "helpers": str(out / "tracking"),
        "helpers_sha256": hashes,
        "inventory_sha256": expected,
        "command": command(out),
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
        result["error"] = error
    write(out / "receipt.json", result)
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        result.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", result)
        raise RuntimeError(f"Training-path validation CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Training-path validation requires its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("Training-path validation imported a mutable helper")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "check"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in predecessor.diagnostic.ENVIRONMENT.items()):
        raise RuntimeError("Training-path validation entry requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU staging requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Archived check requires --cpu-receipt")
        result = accepted(root, args.cpu_receipt.resolve(strict=True))
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
