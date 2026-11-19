# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive the original-terminal actual-LR handoff component without issuing ownership."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_predecessor as predecessor
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_ns_training_path_lr_validation import ENVIRONMENT
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write

PARENT = "run-preparation/ns-training-path-actual-lr-component-cpu/20261119T0926/receipt.json"
PARENT_SHA256 = "aca115258fca1cfb4adf4bbfe58b6606f1b63b022bf0d10525efcb3d9d7a4251"
STATUS = "ns_training_path_actual_lr_predecessor_cpu_archive_verified_owner_unissued"
NEW_HELPERS = (
    "ns_training_path_lr_predecessor.py",
    "ns_training_path_lr_predecessor_test_support.py",
    "test_ns_training_path_lr_predecessor.py",
    "prepare_ns_training_path_lr_predecessor.py",
)
TEST_COUNT = 12
TEST_MODULE = "test_ns_training_path_lr_predecessor"


def inherited(root: Path) -> dict:
    return accepted_archive(
        root,
        (
            PARENT,
            PARENT_SHA256,
            "ns_training_path_actual_lr_component_cpu_archive_verified_owner_unissued",
            29,
        ),
    )


def helper_sources(root: Path, previous: dict) -> dict[str, Path]:
    """Preserve both original closures and pin the file-only report before staging."""
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    path = root / predecessor.reporting.FOLLOWUP_CPU
    original = read(path)
    helpers = path.parent / "tracking"
    cases = list(ET.parse(original["tests"]).getroot().iter("testcase"))
    suites = list(ET.parse(original["tests"]).getroot().iter("testsuite"))
    if (
        sha256(path) != predecessor.reporting.FOLLOWUP_CPU_SHA256
        or original["status"] != predecessor.followup.CPU_STATUS
        or original["root"] != str(root)
        or original["helpers"] != str(helpers)
        or original["environment"] != predecessor.followup.ENVIRONMENT
        or type(original["exit_code"]) is not int
        or original["exit_code"] != 0
        or "error" in original
        or len(cases) != 19
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != 19
        or sum(int(suite.attrib["tests"]) for suite in suites) != 19
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
        or any(
            original[name] != str(path.parent / filename) or original[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
    ):
        raise ValueError("Training-path actual-LR handoff original waiter CPU/archive evidence differs")
    require_helper_bytes(helpers, original["helpers_sha256"])
    require_archive(
        Path(original["archive"]),
        {f"tracking/{name}": digest for name, digest in original["helpers_sha256"].items()},
        alias=False,
    )
    for name, digest in original["helpers_sha256"].items():
        if name in sources and sha256(sources[name]) != digest:
            raise ValueError("Training-path actual-LR handoff inherited waiter helpers conflict")
        sources.setdefault(name, helpers / name)
    local = Path(__file__).resolve().parent
    reporter = local / "ns_training_path_reporting.py"
    if sha256(reporter) != predecessor.REPORTER_SHA256:
        raise ValueError("Training-path actual-LR handoff original report source changed")
    sources[reporter.name] = reporter
    if set(sources).intersection(NEW_HELPERS):
        raise ValueError("Training-path actual-LR handoff would replace an inherited helper")
    sources.update({name: local / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in (*NEW_HELPERS, reporter.name):
        source = sources[name]
        if source.is_symlink() or not source.is_file():
            raise ValueError("Training-path actual-LR handoff source is absent or aliased")
        for node in ast.walk(ast.parse(source.read_bytes(), filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Training-path actual-LR handoff requires a complete flat import closure")
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Training-path actual-LR handoff import escapes its accepted source closure")
    return sources


def command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        predecessor.contract.PYTHON,
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
    raw, receipt = path.read_bytes(), read(path)
    previous = inherited(root)
    hashes = {name: sha256(source) for name, source in helper_sources(root, previous).items()}
    inventory = previous["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != previous["candidate"]
        or receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or receipt["original_followup_cpu_sha256"] != predecessor.reporting.FOLLOWUP_CPU_SHA256
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
        or receipt["scope"] != predecessor.SCOPE
        or len(cases) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {TEST_MODULE}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != TEST_COUNT
        or sum(int(suite.attrib["tests"]) for suite in suites) != TEST_COUNT
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in inventory.items())
    ):
        raise ValueError("Training-path actual-LR handoff complete CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), inventory, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("Training-path actual-LR handoff CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or out != out.resolve()
        or out.exists()
        or out.is_symlink()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("Training-path actual-LR handoff staging needs hidden CUDA and unused external output")
    previous = inherited(root)
    sources = helper_sources(root, previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    inventory = previous["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    out.mkdir(parents=True)
    for name, digest in inventory.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        source = sources[Path(name).name] if name.startswith("tracking/") else (root / PARENT).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise ValueError("Training-path actual-LR handoff parent bytes changed during staging")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    log, error = out / "cpu.log", None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command(out),
                cwd=out,
                env=os.environ | ENVIRONMENT | {"UV_OFFLINE": "1", "PYTHONPATH": str(out / "tracking")},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Training-path actual-LR handoff CPU tests timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(out / name, arcname=name, recursive=False)
        bundle.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "status": STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": previous["candidate"],
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "original_followup_cpu_sha256": predecessor.reporting.FOLLOWUP_CPU_SHA256,
        "helpers": str(out / "tracking"),
        "helpers_sha256": hashes,
        "inventory_sha256": inventory,
        "command": command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": predecessor.SCOPE,
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
        raise RuntimeError(f"Training-path actual-LR handoff CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Training-path actual-LR handoff requires its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Training-path actual-LR handoff imported a mutable helper: {name}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check", "observe", "seal", "check-seal"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Training-path actual-LR handoff entry requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Archived checking requires --cpu-receipt")
        result = accepted(root, args.cpu_receipt.resolve(strict=True))
        if args.phase == "observe":
            if args.out is None or args.out.exists() or args.out.resolve().parent != root / "run-preparation":
                parser.error("Observation requires unused root run-preparation output")
            result = predecessor.readiness(root) | {
                "cpu_receipt_sha256": sha256(args.cpu_receipt),
                "scope": predecessor.SCOPE,
            }
            write(args.out, result)
        elif args.phase == "seal":
            result = predecessor.seal(root)
        elif args.phase == "check-seal":
            if args.seal_sha256 is None:
                parser.error("Seal checking requires --seal-sha256")
            result = predecessor.require_seal(root, args.seal_sha256)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
