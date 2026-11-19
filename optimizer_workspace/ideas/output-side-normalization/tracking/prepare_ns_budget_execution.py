# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and CPU-test the separate NS execution component without issuing a CUDA owner."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

PARENT = "run-preparation/ns-budget-trial-cpu/20261118T2310/receipt.json"
PARENT_SHA256 = "44191e83c1b8c4bdc666aa94710edb25d60e80de302a07ccdba85fc36b80b2fa"
PARENT_STATUS = "ns_budget_graph_component_cpu_sources_and_tests_verified_cuda_owner_absent"
DEPENDENCY = "controllers/rational-continuation-20261118/tracking"
BINDING = "run-preparation/rational-continuation-binding-20261118.json"
BINDING_SHA256 = "a1a7d4ebd88b563b5f80d36ba7e0033752315043b6b799714e08e3cb25e27ba3"
STATUS = "ns_budget_execution_cpu_archive_verified_cuda_owner_unissued"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
NEW_HELPERS = ("ns_budget_execution.py", "prepare_ns_budget_execution.py", "test_replay_saved_ns.py")
TESTS = ("test_replay_saved_ns.py",)
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1", "UV_OFFLINE": "1"}


def parent(root: Path) -> dict:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("NS execution parent graph-component receipt changed")
    receipt = read(path)
    suites = list(ET.parse(path.parent / "tests.xml").getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != PARENT_STATUS
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or receipt["dependency_archive"] != DEPENDENCY
        or receipt["tests"] != len(cases)
        or len(cases) != 29
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("errors", "failures", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("error", "failure", "skipped"))
        or any(sha256(path.parent / name) != value for name, value in receipt["files_sha256"].items())
        or any(
            receipt[key] != sha256(path.parent / filename)
            for key, filename in (
                ("archive_sha256", "source.tar"),
                ("test_xml_sha256", "tests.xml"),
                ("cpu_log_sha256", "cpu.log"),
            )
        )
    ):
        raise ValueError("NS execution parent graph-component source or complete CPU evidence differs")
    require_archive(path.parent / "source.tar", receipt["files_sha256"], alias=False)
    if sha256(root / BINDING) != BINDING_SHA256:
        raise ValueError("NS execution inherited helper binding changed")
    binding = read(root / BINDING)
    if binding["helpers"] != str(root / DEPENDENCY):
        raise ValueError("NS execution inherited helper directory differs")
    require_helper_bytes(root / DEPENDENCY, binding["helpers_sha256"])
    return receipt


def helper_sources(root: Path) -> dict[str, Path]:
    accepted = parent(root)
    inherited = read(root / BINDING)
    sources = {name: root / DEPENDENCY / name for name in inherited["helpers_sha256"]}
    for name in accepted["files_sha256"]:
        if name.startswith("tracking/"):
            sources[Path(name).name] = (root / PARENT).parent / name
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {
        "torch",
        "triton",
        "numpy",
        "pytest",
        "pytest_timeout",
        "swanlab",
        "dotenv",
        "experiment",
    }
    for name, path in sources.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"NS execution helper is missing or aliased: {name}")
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError(f"NS execution helper imports outside its accepted closure: {name}")
    return sources


def fixtures(root: Path) -> dict[str, Path]:
    accepted = parent(root)
    return {name: (root / PARENT).parent / name for name in accepted["files_sha256"] if not name.startswith("tracking/")}


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
        *(str(out / "tracking" / name) for name in TESTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    receipt_bytes = path.read_bytes()
    receipt = read(path)
    sources = helper_sources(root)
    fixture_sources = fixtures(root)
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        receipt["status"] != STATUS
        or receipt["root"] != str(root)
        or receipt["parent_receipt_sha256"] != PARENT_SHA256
        or receipt["dependency_binding_sha256"] != BINDING_SHA256
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != {name: sha256(source) for name, source in sources.items()}
        or receipt["fixtures_sha256"] != {name: sha256(source) for name, source in fixture_sources.items()}
        or receipt["command"] != command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or not cases
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(name).stem for name in TESTS}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("errors", "failures", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("error", "failure", "skipped"))
    ):
        raise ValueError("NS execution complete CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", receipt["helpers_sha256"])
    if any(sha256(path.parent / name) != digest for name, digest in receipt["fixtures_sha256"].items()):
        raise ValueError("NS execution archived numerical fixture differs")
    require_archive(
        Path(receipt["archive"]),
        {f"tracking/{name}": digest for name, digest in receipt["helpers_sha256"].items()} | receipt["fixtures_sha256"],
        alias=False,
    )
    if path.read_bytes() != receipt_bytes:
        raise ValueError("NS execution CPU receipt changed during reconstruction")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("NS execution CPU archive requires hidden CUDA and an unused external evidence directory")
    sources, fixture_sources = helper_sources(root), fixtures(root)
    hashes, fixture_hashes = (
        {name: sha256(path) for name, path in collection.items()} for collection in (sources, fixture_sources)
    )
    (out / "tracking").mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, out / "tracking" / name)
    for name, source in fixture_sources.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    error = None
    with (out / "cpu.log").open("x") as stream:
        try:
            code = subprocess.run(
                command(out),
                cwd=out,
                env=os.environ
                | ENVIRONMENT
                | {"PYTHONPATH": f"{out / 'tracking'}:/opt/conda/envs/torch-base/lib/python3.13/site-packages"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "NS execution CPU acceptance timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    with tarfile.open(out / "source.tar", "x") as archive:
        for name in hashes:
            archive.add(out / "tracking" / name, arcname=f"tracking/{name}", recursive=False)
        for name in fixture_hashes:
            archive.add(out / name, arcname=name, recursive=False)
    receipt = {
        "status": STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "helpers": str(out / "tracking"),
        "helpers_sha256": hashes,
        "fixtures_sha256": fixture_hashes,
        "parent_receipt_sha256": PARENT_SHA256,
        "dependency_binding_sha256": BINDING_SHA256,
        "command": command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "exit_code": code,
        "cuda_visible_devices": "",
        **{
            name: str(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
        **{
            f"{name}_sha256": sha256(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
            if (out / filename).is_file()
        },
        "scope": (
            "NS numerical execution/reconstruction CPU archive only; no CUDA owner, binding, allocation or production."
        ),
    }
    if error:
        receipt["error"] = error
    write(out / "receipt.json", receipt)
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        receipt.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", receipt)
        raise RuntimeError(f"NS execution CPU failure preserved at {out}") from failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check"))
    args = parser.parse_args()
    root, out = args.root.resolve(strict=True), args.out.resolve()
    result = cpu(root, out) if args.phase == "cpu" else require_cpu(root, out / "receipt.json")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
