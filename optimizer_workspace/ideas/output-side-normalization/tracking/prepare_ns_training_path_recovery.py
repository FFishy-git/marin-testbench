# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and issue a separately owned marker-exit recovery diagnostic."""

import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_recovery_predecessor as predecessor
import ns_training_path_recovery_source as source
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_failure_handoff import keeper_guard, same_keeper
from selected_lr_release_owner import SAFE_MARKERS, candidate_files, read, sha256
from watch_wave5 import write

PREPARATION = source.PREPARATION
OWNER = source.OWNER
CPU_STATUS = source.CPU_STATUS
SCOPE = source.SCOPE
CPU_PATH = f"run-preparation/{source.NAMESPACE}-cpu/20261119T1415/receipt.json"
PARENT_SHA256 = predecessor.PARENT_SPEC[1]
LOCAL_HELPERS = (
    "ns_training_path_recovery_predecessor.py",
    "ns_training_path_recovery_source.py",
    "prepare_ns_training_path_recovery.py",
    "test_ns_training_path_recovery.py",
)
TEST_COUNTS = {"test_ns_training_path_recovery_gpu": 45, "test_owned_output_handoff": 36}
RECOVERY_TEST_COUNT = 22


def require_hidden_cuda() -> None:
    if any(os.environ.get(name) != value for name, value in source.ENVIRONMENT.items()):
        raise RuntimeError("Training-path recovery preparation requires fixed hidden CUDA")


def inherited(root: Path) -> dict:
    return predecessor.accepted_archive(root, predecessor.PARENT_SPEC)


def helper_bytes(parent: dict) -> dict[str, bytes]:
    """Inherit accepted helpers and replace only the measured handoff implementation."""
    directory = Path(parent["helpers"])
    sources = {}
    for name, digest in parent["helpers_sha256"].items():
        path = directory / name
        if sha256(path) != digest:
            raise ValueError("Training-path recovery inherited helper changed")
        sources[name] = path.read_bytes()
    local = Path(__file__).resolve().parent
    for name, digest in (
        ("owned_output_handoff.py", source.HANDOFF_SHA256),
        ("test_owned_output_handoff.py", source.HANDOFF_TEST_SHA256),
    ):
        path = local / name
        if sha256(path) != digest:
            raise ValueError("Training-path recovery local measured marker-exit correction changed")
        sources[name] = path.read_bytes()
    for name in LOCAL_HELPERS:
        if name in sources:
            raise ValueError("Training-path recovery new helper collides with the original closure")
        sources[name] = (local / name).read_bytes()
    sources |= source.derived_sources(parent)
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    pending = [
        *LOCAL_HELPERS,
        *source.derived_sources(parent),
        "owned_output_handoff.py",
        "test_owned_output_handoff.py",
    ]
    inspected = set()
    while pending:
        name = pending.pop()
        if name in inspected:
            continue
        inspected.add(name)
        content = sources[name]
        for node in ast.walk(ast.parse(content, filename=name)):
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            for module in modules:
                dependency = module.split(".")[0]
                if dependency in external:
                    continue
                filename = f"{dependency}.py"
                if filename not in sources:
                    raise ValueError(f"Training-path recovery source closure is incomplete: {name}: {module}")
                pending.append(filename)
    return sources


def digest_bytes(contents: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}


def predecessor_binding(root: Path) -> dict:
    seal_path = root / predecessor.SEAL
    receipt_path = root / CPU_PATH
    cpu_receipt = read(receipt_path)
    if (
        cpu_receipt["status"] != CPU_STATUS
        or cpu_receipt["root"] != str(root)
        or cpu_receipt["helpers_sha256"]["ns_training_path_recovery_predecessor.py"]
        != sha256(Path(predecessor.__file__))
        or cpu_receipt["helpers_sha256"]["prepare_ns_training_path_recovery.py"] != sha256(Path(__file__))
    ):
        raise ValueError("Training-path recovery predecessor checker is not CPU/source accepted")
    return {
        "cpu_receipt": str(receipt_path),
        "cpu_receipt_sha256": sha256(receipt_path),
        "verifier_sha256": sha256(Path(predecessor.__file__)),
        "sealer_sha256": sha256(Path(__file__)),
        "checker_sha256": sha256(Path(__file__)),
        "seal": str(seal_path),
        "seal_sha256": sha256(seal_path),
        "seal_identity": file_identity(seal_path),
    }


def require_predecessor(root: Path) -> dict:
    return predecessor.require_seal(root, sha256(root / predecessor.SEAL))


def prepare(root: Path, candidate: Path) -> dict:
    require_hidden_cuda()
    parent = inherited(root)
    if candidate.exists() or candidate.is_symlink() or candidate != candidate.resolve():
        raise FileExistsError("Training-path recovery needs an unused canonical candidate")
    previous = Path(parent["candidate"])
    if candidate.is_relative_to(previous) or previous.is_relative_to(candidate):
        raise ValueError("Training-path recovery candidate overlaps its accepted parent")
    contents = helper_bytes(parent)
    shutil.copytree(
        previous,
        candidate,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ruff_cache", "tracking", "ENDEAVOR.md"),
    )
    (candidate / OWNER).write_bytes(source.ADAPTER.encode())
    record = {
        "status": source.PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "files_sha256": candidate_files(candidate),
        "parent_files_sha256": parent["files_sha256"],
        "parent_metadata_sha256": {
            name: digest
            for name, digest in parent["inventory_sha256"].items()
            if "/" not in name and name.endswith(".json")
        },
        "helpers_sha256": digest_bytes(contents),
        "allowed_helper_replacement": {
            "owned_output_handoff.py": {
                "before": parent["helpers_sha256"]["owned_output_handoff.py"],
                "after": source.HANDOFF_SHA256,
            },
        },
        "changed_files": [OWNER],
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "scope": SCOPE,
    }
    write(candidate / PREPARATION, record)
    return require_preparation(root, candidate)


def require_preparation(root: Path, candidate: Path) -> dict:
    parent = inherited(root)
    path = candidate / PREPARATION
    identity, raw, prepared = file_identity(path), path.read_bytes(), read(path)
    files = candidate_files(candidate)
    hashes = digest_bytes(helper_bytes(parent))
    if (
        prepared["status"] != source.PREPARED_STATUS
        or prepared["root"] != str(root)
        or prepared["candidate"] != str(candidate)
        or prepared["parent_candidate"] != parent["candidate"]
        or prepared["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or prepared["files_sha256"] != files
        or prepared["parent_files_sha256"] != parent["files_sha256"]
        or prepared["parent_metadata_sha256"]
        != {
            name: digest
            for name, digest in parent["inventory_sha256"].items()
            if "/" not in name and name.endswith(".json")
        }
        or prepared["allowed_helper_replacement"]
        != {
            "owned_output_handoff.py": {
                "before": parent["helpers_sha256"]["owned_output_handoff.py"],
                "after": source.HANDOFF_SHA256,
            }
        }
        or prepared["helpers_sha256"] != hashes
        or prepared["changed_files"] != [OWNER]
        or prepared["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or prepared["scope"] != SCOPE
        or (candidate / OWNER).read_bytes() != source.ADAPTER.encode()
        or {
            name
            for name in files.keys() | parent["files_sha256"].keys()
            if files.get(name) != parent["files_sha256"].get(name)
        }
        != {OWNER}
        or any(sha256(candidate / name) != value for name, value in prepared["parent_metadata_sha256"].items())
        or any(sha256(candidate / name) != value for name, value in predecessor.original.TRAINING_FILES.items())
        or file_identity(path) != identity
        or path.read_bytes() != raw
    ):
        raise ValueError("Training-path recovery accepted owner-only candidate or helper derivation changed")
    return prepared


def inventory(parent: dict, candidate: Path, prepared: dict) -> dict[str, str]:
    return (
        parent["inventory_sha256"]
        | prepared["files_sha256"]
        | {PREPARATION: sha256(candidate / PREPARATION)}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()}
    )


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        predecessor.original.PYTHON,
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
        *(
            str(out / "tracking" / f"{name}.py")
            for name in TEST_COUNTS | {"test_ns_training_path_recovery": RECOVERY_TEST_COUNT}
        ),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    require_hidden_cuda()
    parent, prepared = inherited(root), require_preparation(root, candidate)
    contents = helper_bytes(parent)
    expected = inventory(parent, candidate, prepared)
    if out.exists() or out.is_symlink() or out != out.resolve() or out.is_relative_to(candidate):
        raise FileExistsError("Training-path recovery CPU staging requires unused external output")
    out.mkdir(parents=True)
    for name, digest in expected.items():
        if name.startswith("artifacts/external-sources/"):
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("tracking/"):
            target.write_bytes(contents[Path(name).name])
        else:
            original = candidate / name if (candidate / name).is_file() else Path(parent["helpers"]).parent / name
            shutil.copy2(original, target)
        if sha256(target) != digest:
            raise ValueError("Training-path recovery staging source changed")
    (out / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    error = None
    with (out / "cpu.log").open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(out),
                cwd=out,
                env=os.environ
                | source.ENVIRONMENT
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
            code, error = 124, "Training-path recovery CPU acceptance timed out"
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
        "files_sha256": prepared["files_sha256"],
        "preparation_sha256": sha256(candidate / PREPARATION),
        "helpers": str(out / "tracking"),
        "helpers_sha256": prepared["helpers_sha256"],
        "inventory_sha256": expected,
        "command": cpu_command(out),
        "cwd": str(out),
        "environment": source.ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        "scope": SCOPE,
        **{
            name: str(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
        **{
            f"{name}_sha256": sha256(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
            if (out / filename).is_file()
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
        raise RuntimeError(f"Training-path recovery CPU failure retained at {out}") from failure


def require_cpu(root: Path, path: Path) -> dict:
    identity, raw, saved = file_identity(path), path.read_bytes(), read(path)
    parent = inherited(root)
    candidate = Path(saved["candidate"])
    prepared = require_preparation(root, candidate)
    expected = inventory(parent, candidate, prepared)
    counts = TEST_COUNTS | {"test_ns_training_path_recovery": RECOVERY_TEST_COUNT}
    suites = list(ET.parse(saved["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        RECOVERY_TEST_COUNT < 1
        or saved["status"] != CPU_STATUS
        or saved["root"] != str(root)
        or saved["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or saved["files_sha256"] != prepared["files_sha256"]
        or saved["preparation_sha256"] != sha256(candidate / PREPARATION)
        or saved["helpers"] != str(path.parent / "tracking")
        or saved["helpers_sha256"] != prepared["helpers_sha256"]
        or saved["inventory_sha256"] != expected
        or saved["command"] != cpu_command(path.parent)
        or saved["cwd"] != str(path.parent)
        or saved["environment"] != source.ENVIRONMENT
        or saved["cuda_visible_devices"] != ""
        or type(saved["exit_code"]) is not int
        or saved["exit_code"] != 0
        or "error" in saved
        or saved["scope"] != SCOPE
        or len(cases) != sum(counts.values())
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or {name: sum(case.attrib["classname"].split(".")[-1] == name for case in cases) for name in counts} != counts
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(sha256(path.parent / name) != digest for name, digest in expected.items())
        or any(
            saved[name] != str(path.parent / filename) or saved[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
    ):
        raise ValueError("Training-path recovery complete CPU/source/archive acceptance differs")
    require_helper_bytes(path.parent / "tracking", saved["helpers_sha256"])
    require_archive(Path(saved["archive"]), expected, alias=True)
    if file_identity(path) != identity or path.read_bytes() != raw:
        raise ValueError("Training-path recovery CPU receipt changed during checking")
    return saved


def require_binding(root: Path, owner_namespace: dict) -> dict:
    """Retain full hidden acceptance and the existing live-parent proof for visible ranks."""
    contract, validation = owner_namespace["contract"], owner_namespace["validation"]
    binding, helpers, _, work, _ = contract.locations(root)
    identity, raw, frozen = file_identity(binding), binding.read_bytes(), read(binding)
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    if not hidden:
        owner_namespace["require_parent_proof"](root, raw, frozen)
    owner_namespace["require_operation_inventory"](frozen["authenticated_files"])
    cpu_path = Path(frozen["cpu_receipt"])
    cpu_receipt = require_cpu(root, cpu_path) if hidden else read(cpu_path)
    candidate = Path(frozen["candidate"])
    parent_candidate = Path(frozen["parent_candidate"])
    sealed = read(root / predecessor.SEAL)
    if (
        root != root.resolve(strict=True)
        or candidate != candidate.resolve(strict=True)
        or frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != cpu_receipt["candidate"]
        or cpu_path != root / CPU_PATH
        or frozen["cpu_receipt_sha256"] != sha256(cpu_path)
        or cpu_receipt["status"] != CPU_STATUS
        or cpu_receipt["root"] != str(root)
        or cpu_receipt["cuda_visible_devices"] != ""
        or type(cpu_receipt["exit_code"]) is not int
        or cpu_receipt["exit_code"] != 0
        or "error" in cpu_receipt
        or cpu_receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or cpu_receipt["helpers_sha256"] != frozen["helpers_sha256"]
        or cpu_receipt["files_sha256"] != frozen["files_sha256"]
        or frozen["helpers"] != str(helpers)
        or frozen["preparation_sha256"] != sha256(candidate / PREPARATION)
        or owner_namespace["candidate_file_names"](candidate) != set(frozen["files_sha256"])
        or owner_namespace["candidate_file_names"](parent_candidate) != set(frozen["parent_files_sha256"])
        or {
            name
            for name in frozen["files_sha256"].keys() | frozen["parent_files_sha256"].keys()
            if frozen["files_sha256"].get(name) != frozen["parent_files_sha256"].get(name)
        }
        != {OWNER}
        or candidate == parent_candidate
        or candidate.is_relative_to(parent_candidate)
        or parent_candidate.is_relative_to(candidate)
        or frozen["allocation_predecessor"] != sealed
        or frozen["predecessor_acceptance"] != predecessor_binding(root)
        or frozen["devices"] != sealed["restored_keeper"]["devices"]
        or frozen["node"] != predecessor.NODE
        or frozen["environment"] != contract.ENVIRONMENT
        or frozen["controller_environment"] != owner_namespace["CONTROLLER_ENVIRONMENT"]
        or frozen["scope"] != SCOPE
        or frozen["allowed_transform"] != OWNER
        or frozen["controller_command"] != owner_namespace["controller_command"](root)
        or frozen["validator_command"] != owner_namespace["validator_command"](root, candidate)
        or frozen["validation_timeout"] != owner_namespace["TIMEOUT"]
        or frozen["archive"] != str(root / owner_namespace["ARCHIVE"])
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["commands"] != contract.commands(candidate, root, work)
        or frozen["plan"] != validation.plan(root, candidate)
        or len(frozen["commands"]) != 54
        or len(frozen["plan"]["trials"]) != 18
        or frozen["plan"]["tolerance"] != contract.TOLERANCE
        or frozen["commands"] != frozen["plan"]["commands"]
    ):
        raise ValueError("Training-path recovery frozen source, predecessor or protocol changed")
    expected = {
        str(cpu_path): frozen["cpu_receipt_sha256"],
        frozen["archive"]: frozen["archive_sha256"],
        str(candidate / PREPARATION): frozen["preparation_sha256"],
        str(root / predecessor.SEAL): frozen["predecessor_acceptance"]["seal_sha256"],
        str(root / "data/experiment_receipt.json"): frozen["data_receipt_sha256"],
    }
    expected |= {str(candidate / name): digest for name, digest in frozen["files_sha256"].items()}
    expected |= {str(parent_candidate / name): digest for name, digest in frozen["parent_files_sha256"].items()}
    expected |= {str(helpers / name): digest for name, digest in frozen["helpers_sha256"].items()}
    expected |= {str(cpu_path.parent / name): digest for name, digest in cpu_receipt["inventory_sha256"].items()}
    expected |= {cpu_receipt[name]: cpu_receipt[f"{name}_sha256"] for name in ("archive", "tests", "log")}
    if (
        any(
            owner_namespace["inventory_record"](frozen["authenticated_files"], Path(name)).get("sha256") != digest
            for name, digest in expected.items()
        )
        or owner_namespace["inventory_record"](frozen["authenticated_files"], root / predecessor.SEAL).get("identity")
        != frozen["predecessor_acceptance"]["seal_identity"]
        or any(
            owner_namespace["inventory_record"](frozen["authenticated_files"], root / name) != entry
            for name, entry in (sealed["evidence"] | sealed["checkpoint_payloads"]).items()
        )
    ):
        raise ValueError("Training-path recovery frozen fields escape their authenticated inventory")
    if hidden:
        prepared = require_preparation(root, candidate)
        if (
            require_predecessor(root) != sealed
            or frozen["files_sha256"] != prepared["files_sha256"]
            or frozen["parent_candidate"] != prepared["parent_candidate"]
            or frozen["parent_files_sha256"] != prepared["parent_files_sha256"]
        ):
            raise ValueError("Training-path recovery original failed handoff or owner-only source changed")
        require_helper_bytes(helpers, frozen["helpers_sha256"])
        if sha256(Path(frozen["archive"])) != frozen["archive_sha256"]:
            raise ValueError("Training-path recovery issued archive changed")
        require_archive(Path(frozen["archive"]), cpu_receipt["inventory_sha256"], alias=True)
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Training-path recovery imported an unarchived helper: {name}")
    if Path(owner_namespace["__file__"]).resolve() != helpers / "ns_training_path_recovery_gpu_owner.py":
        raise RuntimeError("Training-path recovery owner is outside the issued archive")
    if file_identity(binding) != identity or binding.read_bytes() != raw:
        raise ValueError("Training-path recovery binding changed during checking")
    if not hidden:
        owner_namespace["require_parent_proof"](root, raw, frozen)
    owner_namespace["require_operation_inventory"](frozen["authenticated_files"])
    return frozen


def freeze(root: Path, cpu_receipt: Path, contract, owner, validation) -> dict:
    """Bind the fresh source archive after separate CPU acceptance and failed handoff sealing."""
    require_hidden_cuda()
    cpu_receipt_data = accepted(root, cpu_receipt)

    if cpu_receipt != root / CPU_PATH:
        raise ValueError("Training-path recovery CPU receipt is outside its issued namespace")
    candidate = Path(cpu_receipt_data["candidate"])
    prepared = require_preparation(root, candidate)
    binding, helpers, state, work, result = contract.locations(root)
    archive = root / owner.ARCHIVE
    if any(path.exists() or path.is_symlink() for path in (binding, helpers.parent, state, work, result, archive)):
        raise FileExistsError("Training-path recovery execution namespace already has evidence")
    previous = require_predecessor(root)
    with keeper_guard():
        if not same_keeper(predecessor.handoff.keeper_observation(), previous["restored_keeper"]):
            raise RuntimeError("Training-path recovery exact failed-attempt keeper changed before freeze")
    helpers.mkdir(parents=True)
    for name in cpu_receipt_data["helpers_sha256"]:
        shutil.copy2(cpu_receipt.parent / "tracking" / name, helpers / name)
    expected = cpu_receipt_data["inventory_sha256"]
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            original = helpers / Path(name).name if name.startswith("tracking/") else cpu_receipt.parent / name
            bundle.add(original, arcname=name, recursive=False)
        bundle.add(
            cpu_receipt.parent / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False
        )
    require_archive(archive, expected, alias=True)
    paths = {
        str(cpu_receipt): sha256(cpu_receipt),
        str(archive): sha256(archive),
        str(candidate / PREPARATION): sha256(candidate / PREPARATION),
        str(root / predecessor.SEAL): sha256(root / predecessor.SEAL),
        str(root / "data/experiment_receipt.json"): prepared["data_receipt_sha256"],
    }
    paths |= {str(candidate / name): digest for name, digest in prepared["files_sha256"].items()}
    paths |= {
        str(Path(prepared["parent_candidate"]) / name): digest
        for name, digest in prepared["parent_files_sha256"].items()
    }
    paths |= {str(helpers / name): digest for name, digest in cpu_receipt_data["helpers_sha256"].items()}
    paths |= {str(cpu_receipt.parent / name): digest for name, digest in expected.items()}
    paths |= {cpu_receipt_data[name]: cpu_receipt_data[f"{name}_sha256"] for name in ("archive", "tests", "log")}
    authenticated = owner.authenticated_inventory(paths)
    for name, entry in (previous["evidence"] | previous["checkpoint_payloads"]).items():
        if str(root / name) in authenticated and authenticated[str(root / name)] != entry:
            raise ValueError("Training-path recovery failed evidence conflicts with accepted inventory")
        authenticated[str(root / name)] = entry
    frozen = {
        "status": contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "node": owner.NODE,
        "environment": contract.ENVIRONMENT,
        "controller_environment": owner.CONTROLLER_ENVIRONMENT,
        "helpers": str(helpers),
        "helpers_sha256": cpu_receipt_data["helpers_sha256"],
        "files_sha256": prepared["files_sha256"],
        "parent_candidate": prepared["parent_candidate"],
        "parent_files_sha256": prepared["parent_files_sha256"],
        "allowed_transform": OWNER,
        "preparation_sha256": sha256(candidate / PREPARATION),
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "data_receipt_sha256": prepared["data_receipt_sha256"],
        "allocation_predecessor": previous,
        "predecessor_acceptance": predecessor_binding(root),
        "authenticated_files": authenticated,
        "devices": previous["restored_keeper"]["devices"],
        "plan": validation.plan(root, candidate),
        "commands": contract.commands(candidate, root, work),
        "controller_command": owner.controller_command(root),
        "validator_command": owner.validator_command(root, candidate),
        "validation_timeout": owner.TIMEOUT,
        "scope": SCOPE,
    }
    if require_predecessor(root) != previous or require_preparation(root, candidate) != prepared:
        raise ValueError("Training-path recovery accepted source or failed handoff changed before publication")
    owner.require_operation_inventory(authenticated)
    write(binding, frozen)
    return frozen


def accepted(root: Path, path: Path) -> dict:
    require_hidden_cuda()
    saved = require_cpu(root, path)
    helpers = path.parent / "tracking"
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Training-path recovery acceptance requires its archived entry")
    for name in saved["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Training-path recovery imported an unarchived CPU helper: {name}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "cpu", "check", "seal", "check-seal"), required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    require_hidden_cuda()
    root = args.root.resolve(strict=True)
    if args.phase == "prepare":
        if args.candidate is None:
            parser.error("Preparation requires --candidate")
        result = prepare(root, args.candidate.resolve())
    elif args.phase == "cpu":
        if args.candidate is None or args.out is None:
            parser.error("CPU acceptance requires --candidate and --out")
        result = cpu(root, args.candidate.resolve(strict=True), args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Archived phases require --cpu-receipt")
        path = args.cpu_receipt.resolve(strict=True)
        result = accepted(root, path)
        if args.phase == "seal":
            result = predecessor.seal(root)
        elif args.phase == "check-seal":
            if args.seal_sha256 is None:
                parser.error("Seal checking requires --seal-sha256")
            result = predecessor.require_seal(root, args.seal_sha256)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
