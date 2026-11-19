# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a distinct full-model candidate from the preserved failed continuation source."""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import rational_continuation_contract as contract
import rational_continuation_predecessor as predecessor
from fused_attribution_validation_controller import require_helper_bytes
from prepare_fused_attribution_production import replace_once
from prepare_selected_lr_validation import require_archive
from root_start_gpu_owner import require_cpu_archive
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

PARENT_BINDING = "run-preparation/fused-continuation-binding-20261118.json"
START_CPU = "run-preparation/root-start-gpu-owner-cpu/20261118T1952/receipt.json"
START_CPU_SHA256 = "5c5172704bc7f1d3c4e8e2e97c4276013f63bef824238f853243c79e4de985b3"
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
COPIES = {
    "rational_continuation_contract.py": "experiment/rational_continuation_contract.py",
    "rational_root_start.py": "experiment/rational_root_start.py",
    "root_start_rational.py": "experiment/root_start_rational.py",
    "test_rational_root_start.py": "experiment/tests/test_rational_root_start.py",
}
CHANGES = {"experiment/train.py", "experiment/fused_attribution_owner.py", *COPIES.values()}
NEW_HELPERS = (
    "rational_continuation_contract.py",
    "rational_continuation_predecessor.py",
    "rational_root_start.py",
    "prepare_rational_continuation.py",
    "rational_continuation_validation.py",
    "rational_continuation_controller.py",
    "test_rational_root_start.py",
    "test_rational_continuation.py",
)
CPU_TESTS = ("test_rational_continuation.py", "test_selected_lr_failure_handoff.py")


def parent(root: Path) -> dict:
    frozen = read(root / PARENT_BINDING)
    previous = Path(frozen["candidate"])
    if (
        candidate_files(previous) != frozen["files_sha256"]
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or sha256(root / START_CPU) != START_CPU_SHA256
    ):
        raise ValueError("Rational full-model parent source or accepted accuracy archive changed")
    require_archive(
        Path(frozen["archive"]),
        {
            **frozen["files_sha256"],
            **read(previous / "fused-continuation-preparation.json")["parent_metadata_sha256"],
            "fused-continuation-preparation.json": frozen["preparation_sha256"],
            "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
            **{f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()},
        },
        alias=True,
    )
    return frozen


def helper_sources(root: Path) -> dict[str, Path]:
    cpu = require_cpu_archive(root, root / START_CPU, "root_start_gpu_owner_cpu_archive_verified_gpu_unissued", 121)
    sources = {name: Path(cpu["helpers"]) / name for name in cpu["helpers_sha256"]}
    sources.update({name: Path(__file__).parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout", "swanlab", "dotenv"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or aliased rational continuation helper: {name}")
        for node in ast.walk(ast.parse(path.read_bytes())):
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError(f"Rational continuation helper escapes accepted closure: {name}")
    return sources


def fixture_sources(root: Path) -> dict[str, Path]:
    accepted = require_cpu_archive(root, root / START_CPU, "root_start_gpu_owner_cpu_archive_verified_gpu_unissued", 121)
    return {name: (root / START_CPU).parent / name for name in accepted["fixtures_sha256"]}


def contents(previous: Path) -> dict[str, bytes]:
    directory = Path(__file__).parent
    train = (previous / "experiment/train.py").read_text()
    train = replace_once(
        train,
        "from .fused_attribution_owner import require_fused_attribution_owner\n",
        "from .fused_attribution_owner import require_fused_attribution_owner\n"
        "from .rational_root_start import install_start, require_resume_start\n",
    )
    train = replace_once(
        train,
        '    parser.add_argument("--trace-continuation", action="store_true")\n',
        '    parser.add_argument("--trace-continuation", action="store_true")\n'
        '    parser.add_argument("--root-start-variant", choices=("pinned", "rational"), required=True)\n',
    )
    train = replace_once(
        train,
        "    verify_snapshot()\n",
        "    verify_snapshot()\n" "    root_start_receipt = install_start(root_module, args.root_start_variant)\n",
    )
    train = replace_once(
        train,
        "    if args.resume:\n        start_step = restore_checkpoint(\n",
        "    if args.resume:\n"
        "        require_resume_start(args.out, rank, args.root_start_variant, identity)\n"
        "        start_step = restore_checkpoint(\n",
    )
    train = replace_once(
        train,
        '        record["resume"] = None if args.resume is None else str(args.resume.resolve())\n',
        '        record["resume"] = None if args.resume is None else str(args.resume.resolve())\n'
        '        record["root_start_variant"] = args.root_start_variant\n',
    )
    train = replace_once(
        train,
        '    metric_path = args.out / f"metrics_from_{start_step:06d}.jsonl"\n',
        '    start_receipt_path = args.out / f"root_start_rank_{rank}_from_{start_step:06d}.json"\n'
        "    if start_receipt_path.exists():\n"
        "        raise FileExistsError(start_receipt_path)\n"
        "    root_start_receipt.update(rank=rank, start_step=start_step, identity=identity)\n"
        "    atomic_json(start_receipt_path, root_start_receipt)\n"
        '    metric_path = args.out / f"metrics_from_{start_step:06d}.jsonl"\n',
    )
    owner = replace_once(
        (previous / "experiment/fused_attribution_owner.py").read_text(),
        "from .fused_continuation_contract import (",
        "from .rational_continuation_contract import (",
    )
    test = replace_once(
        (directory / "test_rational_root_start.py").read_text(),
        "import rational_root_start as installation\n",
        "from experiment import rational_root_start as installation\n",
    )
    return {
        "experiment/train.py": train.encode(),
        "experiment/fused_attribution_owner.py": owner.encode(),
        **{
            target: (test.encode()) if name == "test_rational_root_start.py" else (directory / name).read_bytes()
            for name, target in COPIES.items()
        },
    }


def prepare(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(strict=True), candidate.resolve()
    frozen, previous_evidence = parent(root), predecessor.evidence(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or previous.is_relative_to(candidate):
        raise FileExistsError("Rational continuation requires an unused distinct candidate")
    sources = helper_sources(root)
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, payload in contents(previous).items():
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    files = candidate_files(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Rational full-model candidate exceeded start/trainer/owner scope")
    record = {
        "status": contract.PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_binding_sha256": sha256(root / PARENT_BINDING),
        "parent_metadata_sha256": {path.name: sha256(path) for path in previous.glob("*.json")},
        "files_sha256": files,
        "changed_files": sorted(changed),
        "helpers_sha256": {name: sha256(path) for name, path in sources.items()},
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "allocation_predecessor": previous_evidence,
        "plan": contract.plan(candidate, root),
        "scope": "Start-only numerical intervention before original warm-up; no production authorization.",
    }
    write(candidate / contract.PREPARATION, record)
    return require_preparation(root, candidate)


def require_preparation(root: Path, candidate: Path) -> dict:
    record, frozen = read(candidate / contract.PREPARATION), parent(root)
    previous = Path(frozen["candidate"])
    if (
        root != root.resolve()
        or candidate != candidate.resolve()
        or candidate.is_symlink()
        or record["status"] != contract.PREPARED_STATUS
        or record["root"] != str(root)
        or record["candidate"] != str(candidate)
        or record["parent_candidate"] != str(previous)
        or record["parent_binding_sha256"] != sha256(root / PARENT_BINDING)
        or record["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or record["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or record["parent_metadata_sha256"] != {path.name: sha256(path) for path in previous.glob("*.json")}
        or record["files_sha256"] != candidate_files(candidate)
        or record["changed_files"] != sorted(CHANGES)
        or record["helpers_sha256"] != {name: sha256(path) for name, path in helper_sources(root).items()}
        or record["allocation_predecessor"] != predecessor.evidence(root)
        or record["plan"] != contract.plan(candidate, root)
        or any((candidate / name).read_bytes() != payload for name, payload in contents(previous).items())
        or any(sha256(candidate / name) != value for name, value in record["parent_metadata_sha256"].items())
        or any(
            value != frozen["files_sha256"].get(name)
            for name, value in record["files_sha256"].items()
            if name not in CHANGES
        )
    ):
        raise ValueError("Rational full-model exact candidate transformation or predecessor differs")
    return record


def inventory(record: dict) -> dict:
    candidate = Path(record["candidate"])
    return (
        record["files_sha256"]
        | record["parent_metadata_sha256"]
        | {contract.PREPARATION: sha256(candidate / contract.PREPARATION)}
        | {f"tracking/{name}": value for name, value in record["helpers_sha256"].items()}
    )


def cpu_command(root: Path, out: Path, candidate: Path) -> list[str]:
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
        str(root),
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
        str(candidate / "experiment/tests"),
        *(str(out / "tracking" / name) for name in CPU_TESTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    prepared = require_preparation(root, candidate)
    if (
        out.exists()
        or out != out.resolve()
        or out.is_relative_to(candidate)
        or candidate.is_relative_to(out)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise FileExistsError("Rational full-model CPU acceptance needs hidden CUDA and unused external evidence")
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, path in helper_sources(root).items():
        shutil.copy2(path, helpers / name)
    fixtures = {}
    for name, source in fixture_sources(root).items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        fixtures[name] = sha256(target)
    log, archive = out / "cpu.log", out / "source.tar"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(root, out, candidate),
                cwd=candidate,
                env=os.environ
                | {
                    "CUDA_VISIBLE_DEVICES": "",
                    "UV_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": f"{candidate}:{helpers}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Rational continuation CPU acceptance timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    with tarfile.open(archive, "x") as bundle:
        for name in inventory(prepared):
            bundle.add(
                helpers / Path(name).name if name.startswith("tracking/") else candidate / name,
                arcname=name,
                recursive=False,
            )
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        for name in fixtures:
            bundle.add(out / name, arcname=f"cpu-fixtures/{name}", recursive=False)
    try:
        require_archive(
            archive,
            inventory(prepared) | {f"cpu-fixtures/{name}": value for name, value in fixtures.items()},
            alias=True,
        )
        if require_preparation(root, candidate) != prepared:
            raise ValueError("Rational continuation source changed during CPU acceptance")
        require_helper_bytes(helpers, prepared["helpers_sha256"])
    except (OSError, ValueError, RuntimeError, KeyError, tarfile.TarError) as failure:
        error = f"{error}; {failure}" if error else f"{type(failure).__name__}: {failure}"
    result = {
        "status": contract.CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "files_sha256": prepared["files_sha256"],
        "helpers_sha256": prepared["helpers_sha256"],
        "preparation_sha256": sha256(candidate / contract.PREPARATION),
        "command": cpu_command(root, out, candidate),
        "exit_code": code,
        "cuda_visible_devices": "",
        "cwd": str(candidate),
        "parent_start_cpu_sha256": START_CPU_SHA256,
        "fixtures_sha256": fixtures,
        **{name: str(path) for name, path in (("log", log), ("archive", archive), ("tests", out / "tests.xml"))},
        **{
            f"{name}_sha256": sha256(path)
            for name, path in (
                ("log", log),
                ("archive", archive),
                ("tests", out / "tests.xml"),
            )
            if path.is_file()
        },
        "scope": "Full candidate and owner CPU/source/archive acceptance only; no CUDA or production result.",
    }
    if error:
        result["error"] = error
    write(out / "receipt.json", result)
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, tarfile.TarError) as failure:
        result.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", result)
        raise RuntimeError(f"Rational continuation CPU failure preserved at {out}") from failure


def require_cpu(root: Path, path: Path) -> dict:
    raw = path.read_bytes()
    result = read(path)
    candidate = Path(result["candidate"])
    prepared = require_preparation(root, candidate)
    suites = list(ET.parse(result["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    expected_modules = {source.stem for source in (candidate / "experiment/tests").glob("test_*.py")}
    expected_modules |= {Path(name).stem for name in CPU_TESTS}
    require_helper_bytes(path.parent / "tracking", result["helpers_sha256"])
    fixtures = {name: sha256(source) for name, source in fixture_sources(root).items()}
    if (
        path != path.resolve()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or result["status"] != contract.CPU_STATUS
        or type(result["exit_code"]) is not int
        or result["exit_code"] != 0
        or result["root"] != str(root)
        or result["cuda_visible_devices"] != ""
        or result["cwd"] != str(candidate)
        or "error" in result
        or result["parent_start_cpu_sha256"] != START_CPU_SHA256
        or result["helpers"] != str(path.parent / "tracking")
        or result["fixtures_sha256"] != fixtures
        or any(sha256(path.parent / name) != value for name, value in fixtures.items())
        or result["files_sha256"] != prepared["files_sha256"]
        or result["helpers_sha256"] != prepared["helpers_sha256"]
        or result["preparation_sha256"] != sha256(Path(result["candidate"]) / contract.PREPARATION)
        or result["command"] != cpu_command(root, path.parent, Path(result["candidate"]))
        or any(
            result[name] != str(path.parent / filename)
            or Path(result[name]).is_symlink()
            or Path(result[name]).stat().st_nlink != 1
            or result[f"{name}_sha256"] != sha256(Path(result[name]))
            for name, filename in (("log", "cpu.log"), ("archive", "source.tar"), ("tests", "tests.xml"))
        )
        or not cases
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or {case.attrib["classname"].split(".")[-1] for case in cases} != expected_modules
        or any(int(suite.attrib[key]) for suite in suites for key in ("errors", "failures", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("error", "failure", "skipped"))
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or any(sha256(Path(result["helpers"]) / name) != value for name, value in result["helpers_sha256"].items())
    ):
        raise ValueError("Rational continuation complete CPU/source evidence differs")
    require_archive(
        Path(result["archive"]),
        inventory(prepared) | {f"cpu-fixtures/{name}": value for name, value in fixtures.items()},
        alias=True,
    )
    if path.read_bytes() != raw:
        raise ValueError("Rational continuation CPU receipt changed during reconstruction")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("prepare", "check", "cpu"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root, candidate = args.root.resolve(strict=True), args.candidate.resolve()
    if args.phase == "prepare":
        result = prepare(root, candidate)
    elif args.phase == "check":
        result = require_preparation(root, candidate)
    else:
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, candidate, args.out.resolve())
    print(result["status"])


if __name__ == "__main__":
    main()
