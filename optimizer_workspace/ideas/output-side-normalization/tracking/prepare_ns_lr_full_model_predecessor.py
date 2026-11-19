# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive the original-terminal handoff component without issuing actual-LR ownership."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_lr_full_model_predecessor as predecessor
import prepare_ns_lr_full_model_validation as parent
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write

PARENT = "run-preparation/ns-actual-lr-validator-cpu/single-pass-20261119T0419/receipt.json"
PARENT_SHA256 = "51e59291b1d5939201352ba2ab30cbb8d8298d3c46a19a20a004cbedc36dee3c"
STATUS = "ns_actual_lr_predecessor_component_cpu_archive_verified_owner_unissued"
NEW_HELPERS = (
    "ns_lr_full_model_predecessor.py",
    "ns_lr_full_model_predecessor_test_support.py",
    "prepare_ns_lr_full_model_predecessor.py",
    "test_ns_lr_full_model_predecessor.py",
)
TEST_COUNT = 13
EXTRA_SOURCES = {
    "ns_budget_queue.py": (
        "run-preparation/ns-budget-queue-cpu/20261119T0024/tracking/ns_budget_queue.py",
        "a1077fb77d1bebaab4889ed16637b1465fd7e702054d5264c48a80b20cbfc29a",
    ),
    "ns_full_model_queue.py": (
        "run-preparation/ns-full-model-queue-cpu/20261119T0318/tracking/ns_full_model_queue.py",
        "f98a3174bd227f5b3376281b2677559e4d450e6ddc641d66f98f13184c7d4a65",
    ),
    "ns_full_model_queue_reporting.py": (
        None,
        "2cc1357af583c24d218f09e10cca8650aea854dd174a6d47fafcbc41730b9a37",
    ),
}


def inherited(root: Path) -> dict:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("NS actual-LR predecessor accepted validator component changed")
    return parent.require_cpu(root, path)


def helper_sources(root: Path, previous: dict) -> dict[str, Path]:
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    local = Path(__file__).resolve().parent
    for name, (relative, digest) in EXTRA_SOURCES.items():
        path = root / relative if relative is not None else local / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError("NS actual-LR predecessor original queue/reporting source changed")
        sources[name] = path
    sources.update({name: local / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in (*EXTRA_SOURCES, *NEW_HELPERS):
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("NS actual-LR predecessor helper is absent or aliased")
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
                raise ValueError("NS actual-LR predecessor helper escapes its authenticated closure")
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
        str(out / "tracking" / NEW_HELPERS[-1]),
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
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != inventory
        or receipt["command"] != command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != parent.parent.ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != predecessor.SCOPE
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
        raise ValueError("NS actual-LR predecessor complete CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), inventory, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("NS actual-LR predecessor CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    environment = parent.parent.ENVIRONMENT
    if (
        any(os.environ.get(name) != value for name, value in environment.items())
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("NS actual-LR predecessor CPU staging needs fixed hidden CUDA and unused external output")
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
            raise ValueError("NS actual-LR predecessor parent source changed during staging")
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
            code, error = 124, "NS actual-LR predecessor CPU tests timed out"
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
        "command": command(out),
        "cwd": str(out),
        "environment": environment,
        "cuda_visible_devices": "",
        "exit_code": code,
        **{name: str(item) for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))},
        **{
            f"{name}_sha256": sha256(item)
            for name, item in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
        },
        "scope": predecessor.SCOPE,
    }
    if error:
        receipt["error"] = error
    write(out / "receipt.json", receipt)
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        receipt.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", receipt)
        raise RuntimeError(f"NS actual-LR predecessor CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("NS actual-LR predecessor must execute its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("NS actual-LR predecessor imported a mutable inherited helper")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check", "observe"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in parent.parent.ENVIRONMENT.items()):
        raise RuntimeError("NS actual-LR predecessor entry requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Accepted predecessor entry requires --cpu-receipt")
        accepted_receipt = accepted(root, args.cpu_receipt.resolve(strict=True))
        result = accepted_receipt
        if args.phase == "observe":
            if args.out is None or args.out.exists() or args.out.resolve().parent != root / "run-preparation":
                parser.error("Read-only observation needs an unused root run-preparation output")
            result = predecessor.readiness(root) | {
                "cpu_receipt_sha256": sha256(args.cpu_receipt),
                "scope": predecessor.SCOPE,
            }
            write(args.out, result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
