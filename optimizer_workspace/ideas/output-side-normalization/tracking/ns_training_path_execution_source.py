# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive an owner-only successor to the accepted training-path numerical component."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_contract as contract
import ns_training_path_predecessor as predecessor
import ns_training_path_validation as validation
import prepare_ns_training_path_validation as parent
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_release_owner import SAFE_MARKERS, candidate_files, read, sha256
from watch_wave5 import write

ENVIRONMENT = predecessor.diagnostic.ENVIRONMENT
PARENT = "run-preparation/ns-training-path-validation-component-cpu/20261119T0707/receipt.json"
PARENT_SHA256 = "d44e758c6cd8d851b5299ab9a191a31b69da6f0aa23cdf21ca7310ca03eb0ce3"
PREDECESSOR_CPU = "run-preparation/ns-training-path-predecessor-cpu/20261119T0649/receipt.json"
PREDECESSOR_CPU_SHA256 = "398dcf9cb1e63b44092c3aa5a6b49075294fff5c882e16789240d458d3b834a8"
VERIFIER_SHA256 = "bda4a06eb58540afe60dfd682f4b3083525fb7c75582d13324f6ff53413ec050"
CHECKER_SHA256 = "6bb32b63bf3ac49429c18a99b1affd40869b5b176443f4074db6b56307be9055"
SEAL_SHA256 = "107bbcbebf3d9b97b1d1e874623a6e106ff1c0c50e31a4c078fa8aa4d210750f"
PREPARATION = "ns-training-path-owner-preparation.json"
PREPARED_STATUS = "ns_training_path_execution_owner_prepared_gpu_unissued"
CPU_STATUS = "ns_training_path_execution_owner_cpu_archive_verified_gpu_unissued"
OWNER = "experiment/fused_attribution_owner.py"
PARENT_OWNER = "fixtures/ns-training-path-parent-owner.py"
NEW_HELPERS = (
    "ns_training_path_execution_source.py",
    "ns_training_path_gpu_owner.py",
    "ns_training_path_gpu_controller.py",
    "ns_training_path_gpu_test_support.py",
    "test_ns_training_path_gpu.py",
    "test_ns_training_path_execution_source.py",
)
TESTS = NEW_HELPERS[-2:]
TEST_CASE_COUNTS: dict[str, int] = {
    "test_ns_training_path_gpu": 45,
    "test_ns_training_path_execution_source": 10,
}
ADAPTER = '''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load only the separately frozen original-training-path diagnostic owner before CUDA."""

import importlib
import sys
from pathlib import Path

from .selected_lr_release_owner import read, sha256

TASK = Path(__file__).resolve().parents[1]


def execution_owner():
    path = TASK / "ns-training-path-owner-preparation.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("NS training-path execution owner is not separately prepared")
    prepared = read(path)
    root = Path(prepared["root"])
    binding = root / "run-preparation/ns-training-path-binding-20261119.json"
    helpers = root / "controllers/ns-training-path-20261119/tracking"
    frozen = read(binding)
    helper = helpers / "ns_training_path_gpu_owner.py"
    if (
        root != root.resolve(strict=True)
        or prepared["candidate"] != str(TASK)
        or frozen["candidate"] != str(TASK)
        or frozen["helpers"] != str(helpers)
        or helpers.is_symlink()
        or helpers != helpers.resolve(strict=True)
        or frozen["preparation_sha256"] != sha256(path)
        or sha256(helper) != frozen["helpers_sha256"][helper.name]
        or helper.is_symlink()
        or sha256(Path(__file__)) != frozen["files_sha256"]["experiment/fused_attribution_owner.py"]
    ):
        raise RuntimeError("NS training-path candidate adapter or issued helper identity differs")
    sys.path.insert(0, str(helpers))
    owner = importlib.import_module("ns_training_path_gpu_owner")
    if Path(owner.__file__).resolve(strict=True) != helper:
        raise RuntimeError("NS training-path candidate imported a mutable execution owner")
    return root, owner


def require_ns_training_path_gpu_owner(root):
    expected, owner = execution_owner()
    if root != expected:
        raise RuntimeError("NS training-path validator root differs")
    return owner.require_gpu_owner(root, TASK)


def require_fused_attribution_owner(args):
    root, owner = execution_owner()
    owner.require_rank(root, TASK, args)
'''
SCOPE = (
    "Separate training-path diagnostic owner; only the candidate owner adapter changes. "
    "Inherit accepted predecessor and numerical-component helpers unchanged. Jointly bind "
    "predecessor CPU receipt, archived verifier and independent seal. Preserve rejected "
    "original NS controls and both zero-phase failed queues. No production authority. " + contract.SCOPE
)


def require_hidden_cuda() -> None:
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("NS training-path execution entry requires fixed hidden CUDA")


def predecessor_binding(root: Path) -> dict:
    """Bind the accepted original checker and its already-published independent seal."""
    helpers = (root / PREDECESSOR_CPU).parent / "tracking"
    return {
        "cpu_receipt": str(root / PREDECESSOR_CPU),
        "cpu_receipt_sha256": PREDECESSOR_CPU_SHA256,
        "verifier": str(helpers / "ns_training_path_predecessor.py"),
        "verifier_sha256": VERIFIER_SHA256,
        "checker": str(helpers / "prepare_ns_training_path_predecessor.py"),
        "checker_sha256": CHECKER_SHA256,
        "seal": str(root / predecessor.SEAL),
        "seal_sha256": SEAL_SHA256,
    }


def require_predecessor(root: Path) -> dict:
    """Recheck sealed identities without hidden-only APIs or historical tensor reads.

    The archived hidden checker establishes the seal before preparation/acceptance.
    This bounded identity check also works after keeper transfer in CUDA-visible ranks.
    Owner checks separately establish live ownership and exact keeper restoration.
    """
    if root != root.resolve(strict=True):
        raise ValueError("NS training-path predecessor root is not canonical")
    binding = predecessor_binding(root)
    receipt_path, seal_path = Path(binding["cpu_receipt"]), Path(binding["seal"])
    receipt_identity, seal_identity = file_identity(receipt_path), file_identity(seal_path)
    receipt_raw, seal_raw = receipt_path.read_bytes(), seal_path.read_bytes()
    if (
        hashlib.sha256(receipt_raw).hexdigest() != PREDECESSOR_CPU_SHA256
        or hashlib.sha256(seal_raw).hexdigest() != SEAL_SHA256
    ):
        raise ValueError("NS training-path accepted predecessor receipt or bound seal changed")
    cpu, saved = json.loads(receipt_raw), json.loads(seal_raw)
    for name, digest in (
        ("ns_training_path_predecessor.py", VERIFIER_SHA256),
        ("prepare_ns_training_path_predecessor.py", CHECKER_SHA256),
    ):
        path = receipt_path.parent / "tracking" / name
        identity = file_identity(path)
        if cpu["helpers_sha256"].get(name) != digest or sha256(path) != digest or file_identity(path) != identity:
            raise ValueError("NS training-path accepted predecessor verifier or checker changed")
    loaded = Path(predecessor.require_retained_identities.__code__.co_filename)
    if (
        loaded.is_symlink()
        or sha256(loaded) != VERIFIER_SHA256
        or cpu["status"] != "ns_training_path_predecessor_cpu_archive_verified_gpu_unissued"
        or cpu["root"] != str(root)
        or cpu["helpers"] != str(receipt_path.parent / "tracking")
        or saved["status"] != predecessor.STATUS
        or saved["root"] != str(root)
        or saved["scope"] != predecessor.SCOPE
        or {name: saved["evidence_sha256"].get(name) for name in predecessor.RECORDS} != predecessor.RECORDS
        or saved["numerical_evidence"]
        != predecessor.retained_numerics(saved["numerical_evidence"]["original_comparison"])
        or saved["gpu_execution_authorized"] is not False
        or saved["production_authorized"] is not False
        or any(saved["evidence_sha256"].get(name) != digest for name, digest in saved["dependency_sha256"].items())
        or any(sha256(root / name) != digest for name, digest in predecessor.RECORDS.items())
    ):
        raise ValueError("NS training-path sealed predecessor decisions or accepted verifier differ")
    predecessor.require_retained_identities(root, saved)
    predecessor.absent_owners(root, saved)
    if file_identity(receipt_path) != receipt_identity or file_identity(seal_path) != seal_identity:
        raise ValueError("NS training-path predecessor receipt or seal changed during checking")
    return saved


def check_predecessor(root: Path) -> dict:
    """Run only the bound archived checker under fixed hidden CUDA, without resealing."""
    require_hidden_cuda()
    saved = require_predecessor(root)
    binding = predecessor_binding(root)
    helpers = Path(binding["checker"]).parent
    command = [
        contract.PYTHON,
        "-P",
        binding["checker"],
        "--root",
        str(root),
        "--phase",
        "check-seal",
        "--cpu-receipt",
        binding["cpu_receipt"],
        "--seal-sha256",
        binding["seal_sha256"],
    ]
    result = subprocess.run(
        command,
        cwd=helpers,
        env=os.environ
        | ENVIRONMENT
        | {"PYTHONPATH": f"{helpers}:/opt/conda/envs/torch-base/lib/python3.13/site-packages"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0 or result.stdout.strip() != predecessor.STATUS:
        raise RuntimeError(f"NS training-path archived predecessor check failed: {result.stderr[-4000:]}")
    if require_predecessor(root) != saved:
        raise ValueError("NS training-path original sealed predecessor changed during archived checking")
    return saved


def inherited(root: Path) -> dict:
    previous = predecessor.accepted_archive(root, (PARENT, PARENT_SHA256, parent.STATUS, 34))
    original_path = root / PREDECESSOR_CPU
    original = read(original_path)
    if (
        sha256(original_path) != PREDECESSOR_CPU_SHA256
        or previous["parent_cpu_receipt_sha256"] != PREDECESSOR_CPU_SHA256
        or previous["candidate"] != original["candidate"]
        or len(original["helpers_sha256"]) != 138
        or len(previous["helpers_sha256"]) != 143
        or any(previous["helpers_sha256"].get(name) != digest for name, digest in original["helpers_sha256"].items())
        or set(previous["helpers_sha256"]) - set(original["helpers_sha256"]) != set(parent.NEW_HELPERS)
    ):
        raise ValueError("NS training-path numerical parent does not preserve the accepted predecessor closure")
    return previous


def execution_sources(previous: dict) -> dict[str, Path]:
    """Preserve inherited bytes and resolve explicit new executable imports before tests."""
    local = Path(__file__).resolve().parent
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    if set(sources).intersection(NEW_HELPERS):
        raise ValueError("NS training-path new execution closure would replace an accepted helper")
    pending, inspected = list(NEW_HELPERS), set()
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    while pending:
        name = pending.pop()
        if name in inspected:
            continue
        inspected.add(name)
        path = local / name
        if path.is_symlink() or not path.is_file() or path != path.resolve(strict=True):
            raise ValueError(f"NS training-path new execution helper is absent or aliased: {name}")
        sources[name] = path
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("NS training-path flat execution closure cannot import relative mutable sources")
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in modules:
                dependency = module.split(".")[0]
                filename = f"{dependency}.py"
                if dependency in external or filename in sources or filename in inspected:
                    continue
                if not dependency.startswith("ns_training_path_"):
                    raise ValueError(f"NS training-path import escapes the accepted execution closure: {module}")
                pending.append(filename)
    for name, digest in previous["helpers_sha256"].items():
        if sha256(sources[name]) != digest:
            raise ValueError("NS training-path inherited helper bytes changed")
    for name, digest in contract.NUMERICAL_READERS.items():
        if sha256(sources[name]) != digest:
            raise ValueError("NS training-path accepted numerical reader changed")
    return sources


def require_numerical_sources(candidate: Path, sources: dict[str, Path]) -> None:
    reader_paths = {
        "ns_full_model_contract.py": Path(contract.specs.__code__.co_filename),
        "ns_full_model_validation.py": Path(validation.trial.__code__.co_filename),
    }
    if any(sha256(candidate / name) != digest for name, digest in contract.TRAINING_FILES.items()) or any(
        path.is_symlink() or sha256(path) != contract.NUMERICAL_READERS[name] for name, path in reader_paths.items()
    ):
        raise ValueError("NS training-path original training boundary or imported numerical reader changed")
    for module in (contract, validation, parent, predecessor):
        path = Path(module.__file__)
        if path.is_symlink() or sha256(path) != sha256(sources[path.name]):
            raise ValueError("NS training-path imported inherited numerical or predecessor helper changed")


def parent_candidate_inventory(previous: dict) -> dict:
    """Anchor the entire original source/config inventory to the accepted archive bytes."""
    archived = Path(previous["helpers"]).parent
    expected = {
        name: digest
        for name, digest in previous["inventory_sha256"].items()
        if name.startswith(("experiment/", "artifacts/pinned-sources-materialized/"))
    }
    if (
        candidate_files(archived) != expected
        or candidate_files(Path(previous["candidate"])) != expected
        or any(
            sha256(Path(previous["candidate"]) / name) != digest
            for name, digest in parent_metadata_inventory(previous).items()
        )
    ):
        raise ValueError("NS training-path entire accepted parent source/config inventory changed")
    return expected


def parent_metadata_inventory(previous: dict) -> dict:
    return {
        name: digest
        for name, digest in previous["inventory_sha256"].items()
        if "/" not in name and name.endswith(".json")
    }


def preparation_record(root: Path, candidate: Path, previous: dict, sources: dict[str, Path]) -> dict:
    require_numerical_sources(candidate, sources)
    parent_candidate_inventory(previous)
    original = Path(previous["candidate"])
    hashes = {name: sha256(path) for name, path in sources.items()}
    return {
        "status": PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(original),
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "parent_metadata_sha256": parent_metadata_inventory(previous),
        "files_sha256": candidate_files(candidate),
        "changed_files": [OWNER],
        "helpers_sha256": hashes,
        "new_helpers_sha256": {
            name: digest for name, digest in hashes.items() if name not in previous["helpers_sha256"]
        },
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "predecessor_binding": predecessor_binding(root),
        "predecessor_seal_identity": file_identity(root / predecessor.SEAL),
        "commands": contract.commands(candidate, root, contract.locations(root)[3]),
        "scope": SCOPE,
    }


def require_candidate(root: Path, candidate: Path, previous: dict, sources: dict[str, Path]) -> dict:
    path = candidate / PREPARATION
    identity, raw, prepared = file_identity(path), path.read_bytes(), read(path)
    files, original_files = candidate_files(candidate), parent_candidate_inventory(previous)
    if (
        candidate != candidate.resolve(strict=True)
        or candidate.is_symlink()
        or {key: value for key, value in prepared.items() if key != "time"}
        != preparation_record(root, candidate, previous, sources)
        or (candidate / OWNER).read_bytes() != ADAPTER.encode()
        or {name for name in files.keys() | original_files.keys() if files.get(name) != original_files.get(name)}
        != {OWNER}
        or any(sha256(candidate / name) != digest for name, digest in prepared["parent_metadata_sha256"].items())
        or path.read_bytes() != raw
        or file_identity(path) != identity
    ):
        raise ValueError("NS training-path exact owner-only transformation or preserved parent changed")
    return prepared


def prepare(root: Path, candidate: Path) -> dict:
    require_hidden_cuda()
    check_predecessor(root)
    previous = inherited(root)
    original, sources = Path(previous["candidate"]), execution_sources(previous)
    if (
        candidate != candidate.resolve()
        or candidate.exists()
        or candidate.is_symlink()
        or candidate.is_relative_to(original)
        or original.is_relative_to(candidate)
    ):
        raise FileExistsError("NS training-path preparation requires an unused separate candidate")
    shutil.copytree(original, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (candidate / OWNER).write_bytes(ADAPTER.encode())
    write(candidate / PREPARATION, preparation_record(root, candidate, previous, sources))
    return require_preparation(root, candidate)


def require_preparation(root: Path, candidate: Path) -> dict:
    require_predecessor(root)
    previous = inherited(root)
    return require_candidate(root, candidate, previous, execution_sources(previous))


def inventory(previous: dict, prepared: dict, candidate: Path) -> dict:
    return (
        previous["inventory_sha256"]
        | prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: sha256(candidate / PREPARATION)}
        | {PARENT_OWNER: previous["inventory_sha256"][OWNER]}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()}
    )


def cpu_command(out: Path) -> list[str]:
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
        *(str(out / "tracking" / name) for name in TESTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    """Authenticate owner evidence without hidden-CUDA-only calls in visible rank checks."""
    if any(type(count) is not int or count < 1 for count in TEST_CASE_COUNTS.values()):
        raise RuntimeError("NS training-path execution CPU test coverage is not yet bound")
    identity, raw, receipt = file_identity(path), path.read_bytes(), read(path)
    previous, candidate = inherited(root), Path(receipt["candidate"])
    prepared = require_preparation(root, candidate)
    expected = inventory(previous, prepared, candidate)
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    counts = {
        Path(name).stem: sum(case.attrib["classname"].split(".")[-1] == Path(name).stem for case in cases)
        for name in TESTS
    }
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or receipt["predecessor_binding"] != predecessor_binding(root)
        or receipt["predecessor_seal_identity"] != prepared["predecessor_seal_identity"]
        or receipt["helpers"] != str(path.parent / "tracking")
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["new_helpers_sha256"] != prepared["new_helpers_sha256"]
        or receipt["inventory_sha256"] != expected
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["command"] != cpu_command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["scope"] != SCOPE
        or not cases
        or set(counts) != set(TEST_CASE_COUNTS)
        or counts != TEST_CASE_COUNTS
        or {case.attrib["classname"].split(".")[-1] for case in cases} != set(counts)
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(path.parent / filename) or receipt[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != digest for name, digest in expected.items())
    ):
        raise ValueError("NS training-path execution complete CPU/source/archive evidence differs")
    require_helper_bytes(path.parent / "tracking", receipt["helpers_sha256"])
    require_archive(Path(receipt["archive"]), expected, alias=True)
    if file_identity(path) != identity or path.read_bytes() != raw:
        raise ValueError("NS training-path execution CPU receipt changed during checking")
    return receipt


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    require_hidden_cuda()
    if any(type(count) is not int or count < 1 for count in TEST_CASE_COUNTS.values()):
        raise RuntimeError("NS training-path execution CPU test coverage is not yet bound")
    if (
        out != out.resolve()
        or out.exists()
        or out.is_symlink()
        or out.is_relative_to(candidate)
        or candidate.is_relative_to(out)
    ):
        raise RuntimeError("NS training-path owner CPU staging requires unused external output")
    check_predecessor(root)
    previous, prepared = inherited(root), require_preparation(root, candidate)
    sources = execution_sources(previous)
    expected = inventory(previous, prepared, candidate)
    out.mkdir(parents=True)
    (out / "artifacts").mkdir()
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for name, digest in expected.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        if name.startswith("tracking/"):
            original = sources[Path(name).name]
        elif name == PARENT_OWNER:
            original = Path(previous["helpers"]).parent / OWNER
        elif name in prepared["files_sha256"] or name in prepared["parent_metadata_sha256"] or name == PREPARATION:
            original = candidate / name
        else:
            original = Path(previous["helpers"]).parent / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha256(target) != digest:
            raise ValueError("NS training-path staged source or inherited fixture changed")
    log, error = out / "cpu.log", None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(out),
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
            code, error = 124, "NS training-path execution CPU suite timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            bundle.add(out / name, arcname=name, recursive=False)
        bundle.add(out / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "predecessor_binding": prepared["predecessor_binding"],
        "predecessor_seal_identity": prepared["predecessor_seal_identity"],
        "preparation_sha256": sha256(candidate / PREPARATION),
        "files_sha256": prepared["files_sha256"],
        "helpers": str(out / "tracking"),
        "helpers_sha256": prepared["helpers_sha256"],
        "new_helpers_sha256": prepared["new_helpers_sha256"],
        "inventory_sha256": expected,
        "command": cpu_command(out),
        "cwd": str(out),
        "environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": SCOPE,
        **{name: str(path) for name, path in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))},
        **{
            f"{name}_sha256": sha256(path)
            for name, path in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
            if path.is_file()
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
        raise RuntimeError(f"NS training-path owner CPU failure preserved at {out}") from failure


def accepted(root: Path, path: Path) -> dict:
    """Require a fresh archived entry, inherited import origins and the archived seal check."""
    require_hidden_cuda()
    receipt = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("NS training-path execution source must check its accepted archived entry")
    for name in receipt["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"NS training-path execution imported a mutable helper: {name}")
    check_predecessor(root)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("prepare", "cpu", "check"))
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    require_hidden_cuda()
    root = args.root.resolve(strict=True)
    if args.phase == "check":
        if args.cpu_receipt is None:
            parser.error("Checking requires --cpu-receipt")
        result = accepted(root, args.cpu_receipt.resolve(strict=True))
    else:
        if args.candidate is None:
            parser.error("Preparation/CPU phases require --candidate")
        candidate = args.candidate.resolve()
        if args.phase == "prepare":
            result = prepare(root, candidate)
        else:
            if args.out is None:
                parser.error("CPU acceptance requires --out")
            result = cpu(root, candidate, args.out.resolve())
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
