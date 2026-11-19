# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept a separate file-only localization handoff without issuing GPU ownership."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_compiler_localization_predecessor as predecessor
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_ns_compiler_localization_execution import ENVIRONMENT
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write

STATUS = "ns_compiler_localization_predecessor_cpu_archive_verified_owner_unissued"
NEW_HELPERS = (
    "ns_compiler_localization_predecessor.py",
    "ns_compiler_localization_predecessor_test_support.py",
    "test_ns_compiler_localization_predecessor.py",
    "prepare_ns_compiler_localization_predecessor.py",
)
TEST_COUNT = 25
REPORTER = "ns_training_path_lr_queue_reporting.py"


def inherited(root: Path) -> tuple[dict, dict]:
    """Authenticate both original archives separately; their trainer adapters differ."""
    return (
        accepted_archive(root, predecessor.EXECUTION_SPEC),
        accepted_archive(root, predecessor.QUEUE_SPEC),
    )


def helper_sources(previous: dict, actual_lr: dict) -> dict[str, Path]:
    """Merge only byte-compatible helper closures, keeping localization's source inventory."""
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    for name, digest in actual_lr["helpers_sha256"].items():
        if name in sources and sha256(sources[name]) != digest:
            raise ValueError("Localization handoff inherited helper closures conflict")
        sources.setdefault(name, Path(actual_lr["helpers"]) / name)
    local = Path(__file__).resolve().parent
    reporter = local / REPORTER
    file_identity(reporter)
    if REPORTER in sources or sha256(reporter) != predecessor.REPORTER_SHA256:
        raise ValueError("Localization handoff file-only reporter differs from its measured source")
    if set(sources).intersection(NEW_HELPERS):
        raise ValueError("Localization handoff would replace an accepted helper")
    sources |= {REPORTER: reporter, **{name: local / name for name in NEW_HELPERS}}
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name, path in sources.items():
        file_identity(path)
        if name not in (*NEW_HELPERS, REPORTER):
            continue
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Localization handoff requires a complete flat import closure")
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError(f"Localization handoff helper import escapes its accepted closure: {name}")
    return sources


def command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        predecessor.queue.contract.PYTHON,
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
        str(out / "tracking" / NEW_HELPERS[2]),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    """Verify staged passing tests, original archives, closure and unchanged source bytes."""
    identity, raw = file_identity(path), path.read_bytes()
    receipt = read(path)
    previous, actual_lr = inherited(root)
    hashes = {name: sha256(source) for name, source in helper_sources(previous, actual_lr).items()}
    inventory = previous["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != previous["candidate"]
        or receipt["parent_cpu_receipt_sha256"] != predecessor.EXECUTION_SPEC[1]
        or receipt["actual_lr_queue_cpu_receipt_sha256"] != predecessor.QUEUE_SPEC[1]
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != inventory
        or receipt["reporter_sha256"] != predecessor.REPORTER_SHA256
        or receipt["command"] != command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != predecessor.SCOPE
        or len(cases) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(NEW_HELPERS[2]).stem}
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
        raise ValueError("Localization handoff complete CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), inventory, alias=True)
    if path.read_bytes() != raw or file_identity(path) != identity:
        raise ValueError("Localization handoff CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or out.exists()
        or out.is_symlink()
        or out != out.resolve()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("Localization handoff CPU archive needs hidden CUDA and unused external output")
    previous, actual_lr = inherited(root)
    sources = helper_sources(previous, actual_lr)
    hashes = {name: sha256(source) for name, source in sources.items()}
    inventory = previous["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in hashes.items()}
    out.mkdir(parents=True)
    for name, digest in inventory.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        source = sources[Path(name).name] if name.startswith("tracking/") else Path(previous["helpers"]).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise ValueError("Localization handoff inherited source changed during staging")
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
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Localization handoff CPU tests timed out"
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
        "parent_cpu_receipt_sha256": predecessor.EXECUTION_SPEC[1],
        "actual_lr_queue_cpu_receipt_sha256": predecessor.QUEUE_SPEC[1],
        "reporter_sha256": predecessor.REPORTER_SHA256,
        "helpers": str(out / "tracking"),
        "helpers_sha256": hashes,
        "inventory_sha256": inventory,
        "command": command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": predecessor.SCOPE,
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
        raise RuntimeError(f"Localization handoff CPU failure retained at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Localization handoff requires its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Localization handoff imported a mutable helper: {name}")
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
        raise RuntimeError("Localization handoff entry requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Archived phases require --cpu-receipt")
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
