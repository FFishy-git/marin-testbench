# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive the streaming localization executor without issuing CUDA ownership."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_compiler_localization_execution as execution
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write

PARENT = (
    "run-preparation/ns-compiler-localization-component-cpu/20261119T1200/receipt.json",
    "bd732674ace715fce23c1995f458131846c018634b39b64488926bb6342b5091",
    "ns_compiler_localization_component_cpu_archive_verified_gpu_unissued",
    22,
)
STATUS = "ns_compiler_localization_execution_cpu_archive_verified_gpu_unissued"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
NEW_HELPERS = (
    "ns_compiler_localization_execution.py",
    "prepare_ns_compiler_localization_execution.py",
    "test_ns_compiler_localization_execution.py",
)
TEST_COUNT = 23
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONFAULTHANDLER": "1",
}


def inherited(root: Path) -> dict:
    return accepted_archive(root, PARENT)


def helper_sources(previous: dict) -> dict[str, Path]:
    """Preserve the entire accepted localization parent and bind all new local imports."""
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    if any(name in sources for name in NEW_HELPERS):
        raise ValueError("Localization execution cannot replace an accepted parent helper")
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Localization execution helper is absent or aliased")
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
                raise ValueError("Localization execution helper escapes its accepted closure")
    return sources


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
        str(out / "tracking" / NEW_HELPERS[-1]),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    """Recheck complete staged tests, parent hashes, helpers and archive before acceptance."""
    raw, receipt = path.read_bytes(), read(path)
    previous = inherited(root)
    sources = helper_sources(previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    inventory = previous["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != previous["candidate"]
        or receipt["parent_cpu_receipt_sha256"] != PARENT[1]
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != inventory
        or receipt["command"] != command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != execution.SCOPE
        or len(cases) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(NEW_HELPERS[-1]).stem}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != TEST_COUNT
        or sum(int(suite.attrib["tests"]) for suite in suites) != TEST_COUNT
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in inventory.items())
    ):
        raise ValueError("Localization execution complete CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), inventory, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("Localization execution CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("Localization execution CPU archive requires hidden CUDA and unused external output")
    previous = inherited(root)
    sources = helper_sources(previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    inventory = previous["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    out.mkdir(parents=True)
    for name, digest in inventory.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        source = sources[Path(name).name] if name.startswith("tracking/") else (root / PARENT[0]).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise ValueError("Localization execution source changed during staging")
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
            code, error = 124, "Localization execution CPU tests timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(out / name, arcname=name, recursive=False)
        bundle.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "status": STATUS if code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": previous["candidate"],
        "parent_cpu_receipt_sha256": PARENT[1],
        "helpers": str(out / "tracking"),
        "helpers_sha256": hashes,
        "inventory_sha256": inventory,
        "command": command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": execution.SCOPE,
        **{
            name: str(item)
            for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", out / "cpu.log"))
        },
        **{
            f"{name}_sha256": sha256(item)
            for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", out / "cpu.log"))
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
        raise RuntimeError(f"Localization execution CPU failure retained at {out}") from failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Localization execution entry requires fixed hidden CUDA")
    root, out = args.root.resolve(strict=True), args.out.resolve()
    if args.phase == "check" and Path(__file__).resolve() != out / "tracking" / Path(__file__).name:
        raise RuntimeError("Localization execution verification requires its archived entry")
    if args.phase == "check":
        for name in read(out / "receipt.json")["helpers_sha256"]:
            module = sys.modules.get(Path(name).stem)
            if module is not None and Path(module.__file__).resolve() != out / "tracking" / name:
                raise RuntimeError("Localization execution imported a mutable parent helper")
    result = cpu(root, out) if args.phase == "cpu" else require_cpu(root, out / "receipt.json")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
