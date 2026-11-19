# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive a source-bound NS full-budget study protocol; never authorize training."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_budget_study as study
import ns_full_model_execution_source as parent
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

PARENT = "run-preparation/ns-full-model-owner-cpu/single-pass-20261119T0300/receipt.json"
PARENT_SHA256 = "cbbd6514d9c6431bfd616ed73f52de13a5eb72011b96a24986e64e9b8e06eaff"
STATUS = "ns_budget_study_protocol_cpu_archive_verified_execution_unissued"
PLAN = "study-plan.json"
NEW_HELPERS = ("ns_budget_study.py", "prepare_ns_budget_study.py", "test_ns_budget_study.py")
TEST_COUNT = 19
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONFAULTHANDLER": "1",
}


def inherited(root: Path) -> dict:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("NS study accepted execution-owner parent changed")
    return parent.require_cpu(root, path)


def source_record(previous: dict) -> dict:
    candidate = Path(previous["candidate"])
    path = candidate / parent.PREPARATION
    prepared = read(path)
    if (
        sha256(path) != previous["preparation_sha256"]
        or prepared["candidate"] != previous["candidate"]
        or prepared["files_sha256"] != previous["files_sha256"]
    ):
        raise ValueError("NS study authenticated parent preparation identity differs")
    return {
        "candidate": previous["candidate"],
        "files_sha256": previous["files_sha256"],
        "data_receipt_sha256": prepared["data_receipt_sha256"],
    }


def helper_sources(previous: dict) -> dict[str, Path]:
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("NS study helper is absent or aliased")
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
                raise ValueError("NS study helper escapes the authenticated archive closure")
    return sources


def command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        study.contract.PYTHON,
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
        parent.previous.parent.original.SAFE_MARKERS,
        "-q",
        str(out / "tracking" / NEW_HELPERS[-1]),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    raw, receipt = path.read_bytes(), read(path)
    previous = inherited(root)
    sources = helper_sources(previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    planned = study.plan(root, Path(previous["candidate"]), source_record(previous))
    inventory = (
        previous["inventory_sha256"]
        | {f"tracking/{name}": digest for name, digest in hashes.items()}
        | {PLAN: sha256(path.parent / PLAN)}
    )
    study.require_plan(read(path.parent / PLAN))
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != previous["candidate"]
        or receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != inventory
        or receipt["plan"] != str(path.parent / PLAN)
        or receipt["plan_sha256"] != sha256(path.parent / PLAN)
        or {key: value for key, value in read(path.parent / PLAN).items() if key != "time"} != planned
        or receipt["command"] != command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
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
        raise ValueError("NS study complete CPU/source/protocol/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), inventory, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("NS study CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("NS study CPU acceptance needs fixed hidden CUDA and unused external output")
    previous = inherited(root)
    sources = helper_sources(previous)
    hashes = {name: sha256(source) for name, source in sources.items()}
    out.mkdir(parents=True)
    for name, digest in previous["inventory_sha256"].items():
        if name.startswith("artifacts/external-sources/"):
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((root / PARENT).parent / name, target)
        if sha256(target) != digest:
            raise ValueError("NS study parent source changed during staging")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for name, source in sources.items():
        shutil.copy2(source, out / "tracking" / name)
    write(out / PLAN, study.plan(root, Path(previous["candidate"]), source_record(previous)))
    inventory = (
        previous["inventory_sha256"]
        | {f"tracking/{name}": digest for name, digest in hashes.items()}
        | {PLAN: sha256(out / PLAN)}
    )
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
            code, error = 124, "NS study CPU tests timed out"
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
        "helpers": str(out / "tracking"),
        "helpers_sha256": hashes,
        "inventory_sha256": inventory,
        "plan": str(out / PLAN),
        "plan_sha256": sha256(out / PLAN),
        "command": command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        **{name: str(item) for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))},
        **{
            f"{name}_sha256": sha256(item)
            for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
        },
        "scope": study.SCOPE,
    }
    if error:
        receipt["error"] = error
    write(out / "receipt.json", receipt)
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        receipt.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", receipt)
        raise RuntimeError(f"NS study CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("NS study must execute its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("NS study imported a mutable inherited helper")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check", "analyze"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("NS study entry requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Accepted study entry requires --cpu-receipt")
        path = args.cpu_receipt.resolve(strict=True)
        result = accepted(root, path)
        if args.phase == "analyze":
            if args.out is None or args.out.exists():
                parser.error("Analysis requires an unused --out")
            result = study.analyze(read(path.parent / PLAN))
            write(args.out, result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
