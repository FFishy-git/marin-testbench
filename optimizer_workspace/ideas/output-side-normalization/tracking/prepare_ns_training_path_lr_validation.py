# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive the original-training-path actual-LR component without issuing ownership."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_contract as contract
import ns_training_path_lr_validation as validation
from fused_attribution_validation_controller import require_helper_bytes
from ns_training_path_predecessor import accepted_archive
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, candidate_files, read, sha256
from watch_wave5 import write

PARENTS = (
    (
        "run-preparation/ns-training-path-owner-cpu/closure-corrected-20261119T0738/receipt.json",
        "b5bc010df3c86aa849afb014acf2c9815e913ebcb50843f38808d78cad739fbd",
        "ns_training_path_execution_owner_cpu_archive_verified_gpu_unissued",
        55,
    ),
    (
        "run-preparation/ns-actual-lr-validator-cpu/single-pass-20261119T0419/receipt.json",
        "51e59291b1d5939201352ba2ab30cbb8d8298d3c46a19a20a004cbedc36dee3c",
        "ns_actual_lr_validator_component_cpu_archive_verified_owner_unissued",
        21,
    ),
)
NEW_HELPERS = (
    "ns_training_path_lr_contract.py",
    "ns_training_path_lr_validation.py",
    "ns_training_path_lr_test_support.py",
    "test_ns_training_path_lr_validation.py",
    "prepare_ns_training_path_lr_validation.py",
)
TEST_MODULE = "test_ns_training_path_lr_validation"
TEST_COUNT = 29
PLAN = "training-path-actual-lr-plan.json"
STATUS = "ns_training_path_actual_lr_component_cpu_archive_verified_owner_unissued"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONFAULTHANDLER": "1",
}
SCOPE = "CPU/source-archive numerical component only; predecessor/owner/binding/queue/launch unissued. " + contract.SCOPE


def inherited(root: Path) -> tuple[dict, dict]:
    """Authenticate both original archives directly, preserving failed-queue dependencies."""
    training, numerical = (accepted_archive(root, spec) for spec in PARENTS)
    if (
        len(training["helpers_sha256"]) != 149
        or len(numerical["helpers_sha256"]) != 141
        or any(
            training["helpers_sha256"][name] != digest
            for name, digest in numerical["helpers_sha256"].items()
            if name in training["helpers_sha256"]
        )
        or any(numerical["helpers_sha256"].get(name) != digest for name, digest in contract.NUMERICAL_READERS.items())
    ):
        raise ValueError("Training-path actual-LR parent archives conflict or numerical readers changed")
    return training, numerical


def helper_sources(training: dict, numerical: dict) -> dict[str, Path]:
    """Keep every inherited helper byte; require complete flat imports for new helpers."""
    sources = {name: Path(training["helpers"]) / name for name in training["helpers_sha256"]}
    sources.update(
        {name: Path(numerical["helpers"]) / name for name in numerical["helpers_sha256"] if name not in sources}
    )
    if set(sources).intersection(NEW_HELPERS):
        raise ValueError("Training-path actual-LR component would replace an inherited helper")
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Training-path actual-LR helper is absent or aliased")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Training-path actual-LR flat archive cannot import relative mutable helpers")
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Training-path actual-LR helper escapes the complete accepted import closure")
    for previous in (training, numerical):
        if any(sha256(sources[name]) != digest for name, digest in previous["helpers_sha256"].items()):
            raise ValueError("Training-path actual-LR inherited helper bytes changed")
    return sources


def source_plan(root: Path, training: dict) -> dict:
    candidate = Path(training["candidate"])
    if candidate_files(candidate) != training["files_sha256"]:
        raise ValueError("Training-path actual-LR original candidate inventory changed")
    validation.require_numerical_sources(candidate)
    return validation.plan(root, candidate)


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
        str(out / "tracking" / f"{TEST_MODULE}.py"),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    raw, receipt = path.read_bytes(), read(path)
    training, numerical = inherited(root)
    hashes = {name: sha256(source) for name, source in helper_sources(training, numerical).items()}
    expected = (
        training["inventory_sha256"]
        | {f"tracking/{name}": digest for name, digest in hashes.items()}
        | {PLAN: sha256(path.parent / PLAN)}
    )
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != training["candidate"]
        or receipt["parent_cpu_receipts_sha256"] != {name: digest for name, digest, _, _ in PARENTS}
        or receipt["files_sha256"] != training["files_sha256"]
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != hashes
        or receipt["inventory_sha256"] != expected
        or receipt["plan"] != str(path.parent / PLAN)
        or receipt["plan_sha256"] != sha256(path.parent / PLAN)
        or {key: value for key, value in read(path.parent / PLAN).items() if key != "time"}
        != source_plan(root, training)
        or receipt["command"] != command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != SCOPE
        or len(cases) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {TEST_MODULE}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != TEST_COUNT
        or sum(int(suite.attrib["tests"]) for suite in suites) != TEST_COUNT
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in expected.items())
    ):
        raise ValueError("Training-path actual-LR complete CPU/source/protocol/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", hashes)
    require_archive(Path(receipt["archive"]), expected, alias=True)
    if path.read_bytes() != raw:
        raise ValueError("Training-path actual-LR CPU receipt changed during checking")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    """Copy accepted sources, run the complete CPU suite and retain every failed attempt."""
    if (
        any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or out != out.resolve()
        or out.exists()
        or out.is_symlink()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("Training-path actual-LR staging requires fixed hidden CUDA and unused external output")
    training, numerical = inherited(root)
    sources = helper_sources(training, numerical)
    hashes = {name: sha256(source) for name, source in sources.items()}
    planned = source_plan(root, training)
    out.mkdir(parents=True)
    for name, digest in training["inventory_sha256"].items():
        if name.startswith("artifacts/external-sources/"):
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((root / PARENTS[0][0]).parent / name, target)
        if sha256(target) != digest:
            raise ValueError("Training-path actual-LR parent bytes changed during staging")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for name, source in sources.items():
        shutil.copy2(source, out / "tracking" / name)
    write(out / PLAN, planned)
    inventory = (
        training["inventory_sha256"]
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
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Training-path actual-LR CPU suite timed out"
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
        "candidate": training["candidate"],
        "parent_cpu_receipts_sha256": {name: digest for name, digest, _, _ in PARENTS},
        "files_sha256": training["files_sha256"],
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
        "scope": SCOPE,
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
        raise RuntimeError(f"Training-path actual-LR CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    """Use the accepted archived entry and exact imported helper origins."""
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Training-path actual-LR component requires its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Training-path actual-LR imported a mutable helper: {name}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Training-path actual-LR component entry requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("--out is required for CPU acceptance")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("--cpu-receipt is required for archived checking")
        result = accepted(root, args.cpu_receipt.resolve(strict=True))
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
