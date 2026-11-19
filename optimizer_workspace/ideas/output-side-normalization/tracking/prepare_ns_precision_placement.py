# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive the CPU precision screen with a distinct single-read localization correction."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_precision_placement as placement
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write

PARENT = (
    "run-preparation/ns-compiler-localization-execution-cpu/20261119T1225/receipt.json",
    "577b13e053e5b823a58f0e1ed6176c4f40141d9fee399327c59ec292951c3936",
    "ns_compiler_localization_execution_cpu_archive_verified_gpu_unissued",
    23,
)
CHANGED = ("ns_compiler_localization.py", "test_ns_compiler_localization.py")
NEW = ("ns_precision_placement.py", "test_ns_precision_placement.py", "prepare_ns_precision_placement.py")
TEST_FILES = (
    "test_ns_compiler_localization.py",
    "test_ns_compiler_localization_execution.py",
    "test_ns_precision_placement.py",
)
TEST_COUNTS = {
    "test_ns_compiler_localization": 24,
    "test_ns_compiler_localization_execution": 23,
    "test_ns_precision_placement": 17,
}
STATUS = "cpu_precision_placement_sources_tests_verified_no_gpu_authority"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONFAULTHANDLER": "1",
}


def source_inventory(root: Path) -> tuple[dict, dict, dict]:
    """Retain every parent file, changing only the reviewed numerical loader and its tests."""
    parent = accepted_archive(root, PARENT)
    sources = {name: Path(parent["helpers"]) / name for name in parent["helpers_sha256"]}
    if any(name in sources for name in NEW) or any(name not in sources for name in CHANGED):
        raise ValueError("Precision placement helper changes differ from the declared successor")
    sources.update({name: Path(__file__).resolve().parent / name for name in (*CHANGED, *NEW)})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in (*CHANGED, *NEW):
        if sources[name].is_symlink() or not sources[name].is_file():
            raise ValueError("Precision placement source is absent or aliased")
        for node in ast.walk(ast.parse(sources[name].read_bytes(), filename=name)):
            modules = (
                [entry.name for entry in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Precision placement helper escapes the declared source closure")
    hashes = {name: sha256(path) for name, path in sources.items()}
    inventory = parent["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    return parent, sources, inventory


def command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        PYTHON,
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
        *[str(out / "tracking" / name) for name in TEST_FILES],
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, out: Path) -> dict:
    path = out / "receipt.json"
    raw, receipt = path.read_bytes(), read(path)
    parent, sources, inventory = source_inventory(root)
    hashes = {name: sha256(source) for name, source in sources.items()}
    suites = list(ET.parse(out / "tests.xml").getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != parent["candidate"]
        or receipt["parent_cpu_receipt_sha256"] != PARENT[1]
        or receipt["helpers"] != str(out / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != inventory
        or receipt["changed_parent_helpers"] != list(CHANGED)
        or receipt["new_helpers"] != list(NEW)
        or receipt["command"] != command(out)
        or receipt["cwd"] != str(out)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != placement.SCOPE
        or receipt["ranking"] != placement.RANKING
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(name).stem for name in TEST_FILES}
        or {name: sum(case.attrib["classname"].split(".")[-1] == name for case in cases) for name in TEST_COUNTS}
        != TEST_COUNTS
        or len(cases) != sum(TEST_COUNTS.values())
        or len(cases) != receipt["test_count"]
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(out / filename) or receipt[f"{name}_sha256"] != sha256(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(out / name) != digest for name, digest in inventory.items())
    ):
        raise ValueError("Precision placement complete source/test/archive evidence differs")
    require_helper_bytes(out / "tracking", hashes)
    require_archive(out / "source.tar", inventory, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("Precision placement CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if out.exists() or out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError("Precision placement archive needs an unused external directory")
    parent, sources, inventory = source_inventory(root)
    out.mkdir(parents=True)
    for name, digest in inventory.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        source = sources[Path(name).name] if name.startswith("tracking/") else (root / PARENT[0]).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise ValueError("Precision placement source changed while staging")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    error = None
    with (out / "cpu.log").open("x") as stream:
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
            code, error = 124, "Precision placement tests timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    with tarfile.open(out / "source.tar", "x") as archive:
        for name in inventory:
            archive.add(out / name, arcname=name, recursive=False)
        archive.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    try:
        cases = list(ET.parse(out / "tests.xml").getroot().iter("testcase"))
    except (OSError, ET.ParseError) as failure:
        cases = []
        error = f"{type(failure).__name__}: {failure}"
    receipt = {
        "status": STATUS if code == 0 and error is None else "needs_attention",
        "time": datetime.now(UTC).isoformat(),
        "root": str(root),
        "candidate": parent["candidate"],
        "parent_cpu_receipt_sha256": PARENT[1],
        "helpers": str(out / "tracking"),
        "helpers_sha256": {name: sha256(source) for name, source in sources.items()},
        "inventory_sha256": inventory,
        "changed_parent_helpers": list(CHANGED),
        "new_helpers": list(NEW),
        "command": command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "test_count": len(cases),
        "scope": placement.SCOPE,
        "ranking": placement.RANKING,
        **{
            name: str(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
        **{
            f"{name}_sha256": sha256(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
            if (out / filename).is_file()
        },
    }
    if error is not None:
        receipt["error"] = error
    write(out / "receipt.json", receipt)
    try:
        return require_cpu(root, out)
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as error:
        receipt.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(out / "receipt.json", receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check"))
    args = parser.parse_args()
    if any(os.environ.get(name) != expected for name, expected in ENVIRONMENT.items()):
        raise RuntimeError("Precision placement preparation requires the fixed hidden-CUDA environment")
    root, out = args.root.resolve(strict=True), args.out.resolve()
    if args.phase == "check" and Path(__file__).resolve() != out / "tracking" / Path(__file__).name:
        raise RuntimeError("Precision placement check requires the archived entry")
    if args.phase == "check":
        for name in read(out / "receipt.json")["helpers_sha256"]:
            module = sys.modules.get(Path(name).stem)
            if module is not None and Path(module.__file__).resolve() != out / "tracking" / name:
                raise RuntimeError("Precision placement check imported mutable helper bytes")
    result = cpu(root, out) if args.phase == "cpu" else require_cpu(root, out)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
