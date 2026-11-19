# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept a separate transition terminal handoff closure before actual-LR ownership."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_transition_lr_predecessor as predecessor
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive

NUMERICAL_CPU = "run-preparation/ns-training-path-actual-lr-component-cpu/20261119T0926/receipt.json"
NUMERICAL_CPU_SHA256 = "aca115258fca1cfb4adf4bbfe58b6606f1b63b022bf0d10525efcb3d9d7a4251"
STATUS = "transition_actual_lr_predecessor_cpu_archive_verified_execution_archives_separate"
PENDING_STATUS = "transition_actual_lr_predecessor_staged_acceptance_pending"
TEST_MODULE = "test_ns_training_path_transition_lr_predecessor"
TEST_COUNT = 29
NEW_HELPERS = (
    "ns_training_path_transition_lr_predecessor.py",
    "prepare_ns_training_path_transition_lr_predecessor.py",
    f"{TEST_MODULE}.py",
)
PARENTS = {
    "ns_training_path_exit_lr_predecessor.py": predecessor.PARENT_SHA256,
    "test_ns_training_path_exit_lr_predecessor.py": "0a59d82243719252e9dcb31cb1d4261f2afcacb08da14ef7d72de937a0410bea",
    "prepare_ns_training_path_exit_lr_predecessor.py": (
        "624500764687f1a08f09f6a704907a1f1e06d3b9753a3069964f2e27a9c613bc"
    ),
}
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)


def helper_sources(root: Path) -> dict[str, Path]:
    """Keep the issued waiter closure intact; do not import the numerical archive."""
    waiter = predecessor.waiter_cpu(root)
    helpers = Path(waiter["helpers"])
    require_helper_bytes(helpers, waiter["helpers_sha256"])
    require_archive(
        Path(waiter["archive"]),
        {f"tracking/{name}": digest for name, digest in waiter["helpers_sha256"].items()},
        alias=False,
    )
    sources = {name: helpers / name for name in waiter["helpers_sha256"]}
    local = Path(__file__).resolve().parent
    pins = predecessor.REPORT_SOURCES | {
        "ns_training_path_transition_reporting.py": predecessor.REPORTER_SHA256,
        **PARENTS,
    }
    for name, digest in pins.items():
        path = local / name
        predecessor.checked(path, digest)
        if name in sources and predecessor.checked(sources[name], digest)[1]["sha256"] != digest:
            raise ValueError("Transition actual-LR handoff source conflicts with its issued waiter")
        sources[name] = path
    if set(sources) & set(NEW_HELPERS):
        raise ValueError("Transition actual-LR handoff would overwrite an issued helper")
    sources |= {name: local / name for name in NEW_HELPERS}
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in (*NEW_HELPERS, *pins):
        payload, _ = predecessor.checked(sources[name])
        for node in ast.walk(ast.parse(payload)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Transition actual-LR handoff requires a flat source closure")
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Transition actual-LR handoff import escapes its source closure")
    return sources


def source_plan(root: Path) -> dict:
    """Retain the conflicting archive identities without replacing either handoff."""
    predecessor.require_root(root)
    numerical = predecessor.saved(root / NUMERICAL_CPU, NUMERICAL_CPU_SHA256)
    waiter = predecessor.waiter_cpu(root)
    conflicts = {
        name: {"numerical_sha256": digest, "execution_sha256": waiter["helpers_sha256"][name]}
        for name, digest in numerical["helpers_sha256"].items()
        if name in waiter["helpers_sha256"] and digest != waiter["helpers_sha256"][name]
    }
    return {
        "namespace": predecessor.NAMESPACE,
        "numerical_cpu_receipt_sha256": NUMERICAL_CPU_SHA256,
        "waiter_cpu_receipt_sha256": predecessor.FOLLOWUP_CPU_SHA256,
        "conflicting_helpers": conflicts,
        "execution_closure_merged": False,
        "gpu_execution_authorized": False,
        "scope": predecessor.SCOPE,
    }


def command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        predecessor.PYTHON,
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


def validate_cpu(root: Path, path: Path, saved: dict) -> None:
    """Verify unique complete tests, exact sources and the entire helper-only archive."""
    predecessor.require_root(root)
    sources = helper_sources(root)
    hashes = {name: predecessor.checked(source)[1]["sha256"] for name, source in sources.items()}
    directory = path.parent
    suites = list(ET.parse(saved["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        saved["status"] != STATUS
        or saved["root"] != str(root)
        or saved["helpers"] != str(directory / "tracking")
        or saved["helpers_sha256"] != hashes
        or saved["source_plan"] != source_plan(root)
        or saved["command"] != command(directory)
        or saved["cwd"] != str(directory)
        or saved["environment"] != predecessor.ENVIRONMENT
        or type(saved["exit_code"]) is not int
        or saved["exit_code"] != 0
        or "error" in saved
        or saved["scope"] != predecessor.SCOPE
        or len(cases) != TEST_COUNT
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {TEST_MODULE}
        or sum(int(suite.attrib["tests"]) for suite in suites) != TEST_COUNT
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
    ):
        raise ValueError("Transition actual-LR handoff complete CPU/source acceptance differs")
    for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log")):
        if saved[name] != str(directory / filename):
            raise ValueError("Transition actual-LR handoff acceptance artifact path differs")
        predecessor.checked(directory / filename, saved[f"{name}_sha256"])
    require_helper_bytes(directory / "tracking", hashes)
    require_archive(Path(saved["archive"]), {f"tracking/{name}": digest for name, digest in hashes.items()}, alias=False)


def require_cpu(root: Path, path: Path) -> dict:
    raw, receipt = predecessor.checked(path)
    saved = json.loads(raw)
    validate_cpu(root, path, saved)
    if predecessor.checked(path)[1] != receipt or path.read_bytes() != raw:
        raise ValueError("Transition actual-LR handoff CPU receipt changed during checking")
    return saved


def accepted(root: Path, path: Path) -> dict:
    """Require the accepted archived entry and imported helper origins."""
    saved = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Transition actual-LR predecessor requires its archived entry")
    for name in saved["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Transition actual-LR predecessor imported an unarchived helper: {name}")
    return saved


def publish(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value | {"time": datetime.now(UTC).isoformat()}, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def cpu(root: Path, out: Path) -> dict:
    """Stage a distinct helper-only archive without granting acceptance."""
    predecessor.require_root(root)
    if (
        any(os.environ.get(name) != value for name, value in predecessor.ENVIRONMENT.items())
        or out != out.resolve()
        or out.exists()
        or out.is_symlink()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("Transition actual-LR handoff acceptance requires unused hidden-CUDA output")
    sources, plan = helper_sources(root), source_plan(root)
    hashes = {name: predecessor.checked(path)[1]["sha256"] for name, path in sources.items()}
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, path in sources.items():
        shutil.copy2(path, helpers / name)
    code, error = 127, None
    with (out / "cpu.log").open("x") as stream:
        try:
            code = subprocess.run(
                command(out),
                cwd=out,
                env=os.environ | predecessor.ENVIRONMENT | {"UV_OFFLINE": "1", "PYTHONPATH": str(helpers)},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Transition actual-LR handoff CPU acceptance timed out"
        except OSError as failure:
            error = f"{type(failure).__name__}: {failure}"
    with tarfile.open(out / "source.tar", "x") as archive:
        for name in hashes:
            archive.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    for filename in ("source.tar", "tests.xml", "cpu.log"):
        path = out / filename
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    receipt = {
        "status": PENDING_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "source_plan": plan,
        "command": command(out),
        "cwd": str(out),
        "environment": predecessor.ENVIRONMENT,
        "exit_code": code,
        "scope": predecessor.SCOPE,
    }
    for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log")):
        path = out / filename
        receipt[name] = str(path)
        receipt[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if error:
        receipt["error"] = error
    publish(out / "staged.json", receipt)
    return receipt


def accept_cpu(root: Path, directory: Path) -> dict:
    """Independently authenticate staged artifacts before publishing one acceptance."""
    predecessor.require_root(root)
    path = directory / "receipt.json"
    if path.exists() or (directory / "acceptance-failure.json").exists():
        raise FileExistsError("Transition actual-LR CPU acceptance attempt is one-shot")
    staged_path = directory / "staged.json"
    raw, identity = predecessor.checked(staged_path)
    staged = json.loads(raw)
    receipt = staged | {"status": STATUS}
    try:
        if staged["status"] != PENDING_STATUS:
            raise ValueError("Transition actual-LR CPU staging did not complete successfully")
        validate_cpu(root, path, receipt)
        if predecessor.checked(staged_path)[1] != identity:
            raise ValueError("Transition actual-LR CPU staging changed during acceptance")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        publish(
            directory / "acceptance-failure.json",
            {
                "status": "needs_attention",
                "error": f"{type(failure).__name__}: {failure}",
                "staged_sha256": identity["sha256"],
                "gpu_execution_authorized": False,
            },
        )
        raise
    publish(path, receipt)
    return require_cpu(root, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("sources", "cpu", "accept", "check", "observe", "seal", "check-seal"), required=True
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    predecessor.require_root(root)
    if args.phase == "sources":
        print(json.dumps(source_plan(root), allow_nan=False), flush=True)
        return
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    elif args.phase == "accept":
        if args.out is None:
            parser.error("CPU acceptance requires the staged --out directory")
        directory = args.out.resolve(strict=True)
        if Path(__file__).resolve() != directory / "tracking" / Path(__file__).name:
            raise RuntimeError("Transition actual-LR CPU acceptance requires its staged archived entry")
        result = accept_cpu(root, directory)
    else:
        if args.cpu_receipt is None:
            parser.error("Archived operations require --cpu-receipt")
        result = accepted(root, args.cpu_receipt.resolve(strict=True))
        if args.phase == "observe":
            if args.out is None or args.out.resolve().parent != root / "run-preparation":
                parser.error("Observation requires an unused root run-preparation output")
            result = predecessor.readiness(root) | {
                "cpu_receipt_sha256": hashlib.sha256(args.cpu_receipt.read_bytes()).hexdigest()
            }
            publish(args.out, result)
        elif args.phase == "seal":
            result = predecessor.seal(root)
        elif args.phase == "check-seal":
            if args.seal_sha256 is None:
                parser.error("Seal checking requires --seal-sha256")
            result = predecessor.require_seal(root, args.seal_sha256)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
