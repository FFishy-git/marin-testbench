# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept a distinct archived training-path evidence helper before any live seal."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_predecessor as predecessor
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write

PARENT_SPEC = (
    predecessor.TRAINING_CPU,
    predecessor.RECORDS[predecessor.TRAINING_CPU],
    "ns_full_model_execution_owner_cpu_archive_verified_gpu_unissued",
    69,
)
STATUS = "ns_training_path_predecessor_cpu_archive_verified_gpu_unissued"
NEW_HELPERS = (
    "ns_training_path_predecessor.py",
    "ns_training_path_predecessor_test_support.py",
    "prepare_ns_training_path_predecessor.py",
    "test_ns_training_path_predecessor.py",
    "test_prepare_ns_training_path_predecessor.py",
)
TESTS = NEW_HELPERS[-2:]
TEST_COUNTS = {"test_ns_training_path_predecessor": 32, "test_prepare_ns_training_path_predecessor": 5}
COMPARISON = "fixtures/ns-budget-comparison.json"
COMPARISON_SOURCE = "diagnostics/ns-budget-20261118/comparison.json"
COMPARISON_SHA256 = predecessor.RECORDS[COMPARISON_SOURCE]
ENVIRONMENT = predecessor.diagnostic.ENVIRONMENT


def parent(root: Path) -> dict:
    return predecessor.accepted_archive(root, PARENT_SPEC)


def helper_sources(previous: dict) -> dict[str, Path]:
    """Preserve every accepted parent helper and bind new imports to that closure."""
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    local = Path(__file__).resolve().parent
    sources.update({name: local / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Training-path predecessor archive helper is absent or aliased")
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
                raise ValueError("Training-path predecessor import escapes the accepted archive closure")
    return sources


def inventory(root: Path, previous: dict, sources: dict[str, Path]) -> dict:
    path = root / COMPARISON_SOURCE
    if sha256(path) != COMPARISON_SHA256:
        raise ValueError("Training-path predecessor original comparison fixture changed")
    return (
        previous["inventory_sha256"]
        | {f"tracking/{name}": sha256(source) for name, source in sources.items()}
        | {COMPARISON: COMPARISON_SHA256}
    )


def command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        predecessor.diagnostic.PYTHON,
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
    previous = parent(root)
    sources = helper_sources(previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    expected = inventory(root, previous, sources)
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != previous["candidate"]
        or receipt["parent_cpu_receipt_sha256"] != PARENT_SPEC[1]
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != expected
        or receipt["command"] != command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != predecessor.SCOPE
        or len(cases) != sum(TEST_COUNTS.values())
        or {case.attrib["classname"].split(".")[-1] for case in cases} != set(TEST_COUNTS)
        or {name: sum(case.attrib["classname"].split(".")[-1] == name for case in cases) for name in TEST_COUNTS}
        != TEST_COUNTS
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
        raise ValueError("Training-path predecessor CPU/source/archive/test evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), expected, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("Training-path predecessor CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("Training-path predecessor CPU staging needs hidden CUDA and unused external output")
    previous = parent(root)
    sources = helper_sources(previous)
    expected = inventory(root, previous, sources)
    out.mkdir(parents=True)
    for name, digest in expected.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        source = (
            sources[Path(name).name]
            if name.startswith("tracking/")
            else root / COMPARISON_SOURCE if name == COMPARISON else (root / PARENT_SPEC[0]).parent / name
        )
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise ValueError("Training-path predecessor source changed during staging")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    log, error = out / "cpu.log", None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command(out),
                cwd=out,
                env=os.environ
                | ENVIRONMENT
                | {
                    "UV_OFFLINE": "1",
                    "PYTHONPATH": f"{out / 'tracking'}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Training-path predecessor CPU tests timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            bundle.add(out / name, arcname=name, recursive=False)
        bundle.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "status": STATUS if code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": previous["candidate"],
        "parent_cpu_receipt_sha256": PARENT_SPEC[1],
        "helpers": str(out / "tracking"),
        "helpers_sha256": {name: sha256(source) for name, source in sources.items()},
        "inventory_sha256": expected,
        "command": command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": predecessor.SCOPE,
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
        raise RuntimeError(f"Training-path predecessor CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Training-path predecessor requires its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("Training-path predecessor imported a mutable helper")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "check", "seal", "check-seal"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Training-path predecessor entry requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Archived acceptance requires --cpu-receipt")
        result = accepted(root, args.cpu_receipt.resolve(strict=True))
        if args.phase == "seal":
            result = predecessor.seal(root)
        elif args.phase == "check-seal":
            if args.seal_sha256 is None:
                parser.error("Seal checking requires caller-bound --seal-sha256")
            result = predecessor.require_seal(root, args.seal_sha256)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
