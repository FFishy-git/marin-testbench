# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report issued training-path numerical CPU evidence without execution authority."""

import hashlib
import json
import stat
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

import ns_training_path_contract as contract

TITLE = "NS Original-Training-Path Diagnostic"
CPU = "run-preparation/ns-training-path-validation-component-cpu/20261119T0707/receipt.json"
CPU_SHA256 = "d44e758c6cd8d851b5299ab9a191a31b69da6f0aa23cdf21ca7310ca03eb0ce3"
ARCHIVE_SHA256 = "c68597acaf94a170dbe36051a4d15f099ccea2837822982628c3c21bc76546de"
TESTS_SHA256 = "d0dfea9cdf7487635cbfd7958cd14c0bf68befe61fb99278b17de2b29cb94a8d"
LOG_SHA256 = "60c19a2d70fb06225c4b35996201f3daf8612153b2585a1b2dd13e6445368141"
PARENT_CPU = "run-preparation/ns-training-path-predecessor-cpu/20261119T0649/receipt.json"
PARENT_SHA256 = "398dcf9cb1e63b44092c3aa5a6b49075294fff5c882e16789240d458d3b834a8"
CPU_STATUS = "ns_training_path_validation_component_cpu_archive_verified_gpu_unissued"
TEST_MODULE = "test_ns_training_path_validation"
TEST_COUNT = 34
HELPER_COUNT = 143
INVENTORY_COUNT = 509
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
ALIAS = "artifacts/external-sources"
PINNED = "artifacts/pinned-sources-materialized"
NEW_HELPERS = {
    "ns_training_path_contract.py",
    "ns_training_path_validation.py",
    "ns_training_path_test_support.py",
    "test_ns_training_path_validation.py",
    "prepare_ns_training_path_validation.py",
}
CPU_ENVIRONMENT = contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
OWNER_CPU = "run-preparation/ns-training-path-owner-cpu/closure-corrected-20261119T0738/receipt.json"
OWNER_CPU_SHA256 = "b5bc010df3c86aa849afb014acf2c9815e913ebcb50843f38808d78cad739fbd"
OWNER_ARCHIVE_SHA256 = "2bfa3676e9475e81e274147f199c4a44eb0d16896e7a756577141ea85fba5708"
OWNER_TESTS_SHA256 = "557c96eeda595105f483182b969a202ead39e4ffd2572f8048f6a1036de2a9a1"
OWNER_LOG_SHA256 = "45335dfa123253dde31a2a7a1f8f968512cc23b3145793e4e15d4ee4c07009d4"
OWNER_CPU_STATUS = "ns_training_path_execution_owner_cpu_archive_verified_gpu_unissued"
OWNER_PREPARATION = "ns-training-path-owner-preparation.json"
OWNER_FILE = "experiment/fused_attribution_owner.py"
PARENT_OWNER_FILE = "fixtures/ns-training-path-parent-owner.py"
OWNER_TEST_COUNTS = {"test_ns_training_path_gpu": 45, "test_ns_training_path_execution_source": 10}
OWNER_NEW_HELPERS = {
    "ns_training_path_execution_source.py",
    "ns_training_path_gpu_owner.py",
    "ns_training_path_gpu_controller.py",
    "ns_training_path_gpu_test_support.py",
    *(f"{name}.py" for name in OWNER_TEST_COUNTS),
}
OWNER_SCOPE = (
    "Separate training-path diagnostic owner; only the candidate owner adapter changes. "
    "Inherit accepted predecessor and numerical-component helpers unchanged. Jointly bind "
    "predecessor CPU receipt, archived verifier and independent seal. Preserve rejected "
    "original NS controls and both zero-phase failed queues. No production authority. " + contract.SCOPE
)
SEAL = "run-preparation/ns-training-path-predecessor-seal-20261119.json"
SEAL_SHA256 = "107bbcbebf3d9b97b1d1e874623a6e106ff1c0c50e31a4c078fa8aa4d210750f"
ISSUED_ARCHIVE = "run-preparation/source-ns-training-path-20261119.tar"
RESULT_STATUS = "ns_training_path_actual_config_gates_reconstructed_not_production"
TERMINAL_STATUS = "ns_training_path_terminal_independently_reconstructed_exact_restoration_not_production"
BINDING_SHA256: str | None = "20133ef8358e51cc97e83b866063843ccc10a314ac2b72aef5b7a8b4354f0064"
ISSUED_ARCHIVE_SHA256 = "6fc0ec29d5075f654210b54b61d4c2e1b2440eb35555cb1c5fa636388bccd5ad"
INTENT_SHA256: str | None = "7b24631110346132bc10b93e539faf07ba071422582715f12f5f060d496213aa"
GATES = {
    "actual_config_complete_trajectories",
    "boundary_all_rank_continuation",
    "post_cached_root_all_rank_continuation",
    "scalar_loss_continuation",
    "captured_row_update_work",
}
EVIDENCE_ERRORS = (ValueError, KeyError, TypeError, OSError, RuntimeError, ET.ParseError, tarfile.TarError)
FOLLOWUP_CPU = "run-preparation/ns-training-path-followup-cpu/closure-corrected-20261119T0847/receipt.json"
FOLLOWUP_CPU_SHA256 = "8e462e95601a843d703fae2dcc92235bed1e00d0a8432d382fb84bbae778259b"
FOLLOWUP_INTENT_SHA256 = "8e7f56916fec0bf0ce65adb7d2740bec85be5bf2ea4afd584f3ec0431715f64d"
FOLLOWUP_WORK = "run-preparation/ns-training-path-terminal-followup-20261119"
FOLLOWUP_ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1"}
FOLLOWUP_SCOPE = (
    "One read-only original-generation waiter and original issued canonical terminal verifier. "
    "No keeper transfer, GPU execution, new tensor reader, numerical waiver, production release, "
    "quality/efficiency claim or SwanLab authority. Rejected cases and consumed terminal claims "
    "remain retained. Check exact owners every 900 seconds; do not restart on observation timeout."
)


def _identity(path: Path) -> tuple:
    info = path.lstat()
    if path != path.resolve(strict=True) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Evidence file is shared, aliased or noncanonical: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def _sha256(path: Path) -> str:
    before = _identity(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if _identity(path) != before:
        raise ValueError(f"Evidence file changed during hashing: {path}")
    return digest


def _receipt(path: Path, digest: str) -> dict:
    before = _identity(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest or _identity(path) != before:
        raise ValueError(f"Accepted CPU receipt changed: {path}")
    return json.loads(raw)


def _relative(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or name == ".":
        raise ValueError("Source inventory has a noncanonical relative path")
    return name


def _source_path(directory: Path, name: str) -> Path:
    _relative(name)
    if name.startswith(f"{ALIAS}/"):
        return directory / PINNED / name.removeprefix(f"{ALIAS}/")
    return directory / name


def _cpu_command(directory: Path, modules: tuple[str, ...] = (TEST_MODULE,)) -> list[str]:
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
        str(directory),
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
        *(str(directory / "tracking" / f"{module}.py") for module in modules),
        "--junitxml",
        str(directory / "tests.xml"),
    ]


def _archive(path: Path, inventory: dict) -> None:
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        expected = set(inventory) | {ALIAS}
        if len(members) != len(expected) or {member.name for member in members} != expected:
            raise ValueError("Numerical CPU source archive inventory differs")
        alias = bundle.getmember(ALIAS)
        if not alias.issym() or alias.linkname != "pinned-sources-materialized":
            raise ValueError("Numerical CPU source archive alias differs")
        for name, digest in inventory.items():
            member = bundle.getmember(_relative(name))
            if not member.isfile():
                raise ValueError("Numerical CPU archived source is not a regular file")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Numerical CPU archived source bytes differ: {name}")


def _tests(path: Path, expected: dict[str, int] | None = None) -> dict:
    expected = {TEST_MODULE: TEST_COUNT} if expected is None else expected
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        {name: sum(case.attrib["classname"].split(".")[-1] == name for case in cases) for name in expected} != expected
        or {case.attrib["classname"].split(".")[-1] for case in cases} != set(expected)
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
    ):
        raise ValueError("Numerical CPU XML does not establish the complete passing CPU suite")
    return {"passed_cases": len(cases), "seconds": sum(float(suite.attrib["time"]) for suite in suites)}


def _numerical_evidence(root: Path) -> dict:
    """Authenticate only the issued CPU component; leave execution evidence unaccepted.

    Hashes are checked against its own staged sources and archive, never mutable
    preparers. This does not rerun acceptance or open historical checkpoint tensors.
    """
    if not root.is_absolute() or root != root.resolve(strict=True):
        raise ValueError("Training-path reporting requires a canonical root")
    path = root / CPU
    receipt = _receipt(path, CPU_SHA256)
    if (
        receipt.get("status") != CPU_STATUS
        or type(receipt.get("exit_code")) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
    ):
        raise ValueError(
            f"Numerical CPU acceptance withheld: status={receipt.get('status')}, "
            f"exit={receipt.get('exit_code')}, error={receipt.get('error')}"
        )
    directory = path.parent
    candidate = Path(receipt["candidate"])
    helpers, inventory = receipt["helpers_sha256"], receipt["inventory_sha256"]
    if (
        receipt["root"] != str(root)
        or not candidate.is_absolute()
        or candidate != candidate.resolve(strict=True)
        or not candidate.is_relative_to(root / "candidates")
        or receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or receipt["helpers"] != str(directory / "tracking")
        or len(helpers) != HELPER_COUNT
        or len(inventory) != INVENTORY_COUNT
        or {name.removeprefix("tracking/"): digest for name, digest in inventory.items() if name.startswith("tracking/")}
        != helpers
        or any(Path(name).name != name for name in helpers)
        or receipt["command"] != _cpu_command(directory)
        or receipt["cwd"] != str(directory)
        or receipt["environment"] != CPU_ENVIRONMENT
        or receipt["cuda_visible_devices"] != ""
        or receipt["scope"]
        != "Separate CPU/source-archive numerical component; execution owner and launch unissued. " + contract.SCOPE
        or any(inventory.get(name) != digest for name, digest in contract.TRAINING_FILES.items())
        or any(helpers.get(name) != digest for name, digest in contract.NUMERICAL_READERS.items())
    ):
        raise ValueError("Numerical CPU source, command, environment or predecessor binding differs")
    sources = {_source_path(directory, name): digest for name, digest in inventory.items()}
    alias = directory / ALIAS
    if (
        not alias.is_symlink()
        or alias.readlink() != Path("pinned-sources-materialized")
        or alias.resolve(strict=True) != directory / PINNED
        or {item.name for item in (directory / "tracking").iterdir()} != set(helpers)
    ):
        raise ValueError("Numerical CPU helper inventory or pinned-source alias differs")
    parent_path = root / PARENT_CPU
    parent = _receipt(parent_path, PARENT_SHA256)
    if (
        parent["status"] != "ns_training_path_predecessor_cpu_archive_verified_gpu_unissued"
        or parent["root"] != str(root)
        or parent["candidate"] != receipt["candidate"]
        or any(helpers.get(name) != digest for name, digest in parent["helpers_sha256"].items())
        or set(helpers) - set(parent["helpers_sha256"]) != NEW_HELPERS
        or inventory != parent["inventory_sha256"] | {f"tracking/{name}": digest for name, digest in helpers.items()}
    ):
        raise ValueError("Numerical CPU component replaced its accepted predecessor source closure")
    records = {path: CPU_SHA256, parent_path: PARENT_SHA256} | sources
    for name, filename, digest in (
        ("archive", "source.tar", ARCHIVE_SHA256),
        ("tests", "tests.xml", TESTS_SHA256),
        ("log", "cpu.log", LOG_SHA256),
    ):
        target = directory / filename
        if receipt[name] != str(target) or receipt[f"{name}_sha256"] != digest:
            raise ValueError(f"Numerical CPU {name} binding differs")
        records[target] = digest
    identities = {target: _identity(target) for target in records}
    if any(_sha256(target) != digest for target, digest in records.items()):
        raise ValueError("Numerical CPU issued source, helper, test, log or archive changed")
    _archive(directory / "source.tar", inventory)
    tests = _tests(directory / "tests.xml")
    for spec in contract.specs():
        contract.require_config(directory, spec)
    if any(_identity(target) != identities[target] for target in records):
        raise ValueError("Numerical CPU evidence changed during reporting")
    binding, issued_helpers, state, _, result = contract.locations(root)
    unaccepted = [
        str(target.relative_to(root))
        for target in (binding, issued_helpers.parent / "controller.json", state, result)
        if target.exists() or target.is_symlink()
    ]
    return {
        "numerical_cpu": {
            "status": receipt["status"],
            "accepted": True,
            "receipt": str(path),
            "receipt_sha256": CPU_SHA256,
            "archive_sha256": ARCHIVE_SHA256,
            "tests_sha256": TESTS_SHA256,
            "log_sha256": LOG_SHA256,
            "test_source_sha256": helpers[f"{TEST_MODULE}.py"],
            "helpers": len(helpers),
            "inventory": len(inventory),
            "time": receipt.get("time"),
            **tests,
        },
        "execution_owner": {"status": "acceptance_unbound", "accepted": False, "unaccepted_saved_paths": unaccepted},
        "gpu_progress": None,
        "numerical_gpu_acceptance": None,
    }


def _stat_identity(path: Path) -> dict:
    device, inode, size, mtime, ctime, links = _identity(path)
    return {"device": device, "inode": inode, "bytes": size, "mtime_ns": mtime, "ctime_ns": ctime, "links": links}


def _saved(path: Path, records: dict) -> dict:
    identity = _identity(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        _sha256(path) != digest
        or _identity(path) != identity
        or (path in records and records[path] != (digest, identity))
    ):
        raise ValueError(f"Saved evidence changed during reading: {path}")
    records[path] = (digest, identity)
    return json.loads(raw)


def _checked(path: Path, digest: str, records: dict) -> None:
    identity = _identity(path)
    if _sha256(path) != digest or (path in records and records[path] != (digest, identity)):
        raise ValueError(f"Bound evidence bytes changed: {path}")
    records[path] = (digest, identity)


def _source_inventory(directory: Path) -> set[str]:
    alias, pinned = directory / ALIAS, directory / PINNED
    if not alias.is_symlink() or alias.readlink() != Path("pinned-sources-materialized") or alias.resolve() != pinned:
        raise ValueError("Owner source alias is not the accepted pinned-source alias")
    names = set()
    for folder in (directory / "experiment", pinned):
        if folder != folder.resolve(strict=True) or not folder.is_dir():
            raise ValueError("Owner source directory is absent or aliased")
        for path in folder.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError("Owner source inventory contains an unexpected alias")
            if path.is_file() and (folder == pinned or path.suffix in (".py", ".json")):
                names.add(str(path.relative_to(directory)))
    return names


def _owner_cpu(root: Path, records: dict) -> tuple[dict, dict, dict]:
    path, parent_path = root / OWNER_CPU, root / CPU
    _checked(path, OWNER_CPU_SHA256, records)
    cpu, parent = _saved(path, records), _receipt(parent_path, CPU_SHA256)
    candidate, directory = Path(cpu["candidate"]), path.parent
    inventory, helpers = cpu["inventory_sha256"], cpu["helpers_sha256"]
    prepared_path = candidate / OWNER_PREPARATION
    _checked(prepared_path, cpu["preparation_sha256"], records)
    prepared = _saved(prepared_path, records)
    original = Path(parent["candidate"])
    original_files = {
        name: digest
        for name, digest in parent["inventory_sha256"].items()
        if name.startswith(("experiment/", f"{PINNED}/"))
    }
    metadata = {
        name: digest for name, digest in parent["inventory_sha256"].items() if "/" not in name and name.endswith(".json")
    }
    new_helpers = {name: digest for name, digest in helpers.items() if name not in parent["helpers_sha256"]}
    expected_inventory = (
        parent["inventory_sha256"]
        | cpu["files_sha256"]
        | metadata
        | {OWNER_PREPARATION: cpu["preparation_sha256"], PARENT_OWNER_FILE: original_files[OWNER_FILE]}
        | {f"tracking/{name}": digest for name, digest in helpers.items()}
    )
    if (
        cpu["status"] != OWNER_CPU_STATUS
        or cpu["root"] != str(root)
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["parent_cpu_receipt_sha256"] != CPU_SHA256
        or candidate != candidate.resolve(strict=True)
        or not candidate.is_relative_to(root / "candidates")
        or candidate == original
        or candidate.is_relative_to(original)
        or original.is_relative_to(candidate)
        or cpu["helpers"] != str(directory / "tracking")
        or len(helpers) != 149
        or len(inventory) != 517
        or set(new_helpers) != OWNER_NEW_HELPERS
        or cpu["new_helpers_sha256"] != new_helpers
        or any(helpers.get(name) != digest for name, digest in parent["helpers_sha256"].items())
        or any(Path(name).name != name for name in helpers)
        or cpu["command"] != _cpu_command(directory, tuple(OWNER_TEST_COUNTS))
        or cpu["cwd"] != str(directory)
        or cpu["environment"] != CPU_ENVIRONMENT
        or cpu["cuda_visible_devices"] != ""
        or cpu["scope"] != OWNER_SCOPE
        or inventory != expected_inventory
        or prepared["status"] != "ns_training_path_execution_owner_prepared_gpu_unissued"
        or prepared["root"] != str(root)
        or prepared["candidate"] != str(candidate)
        or prepared["parent_candidate"] != str(original)
        or prepared["parent_cpu_receipt_sha256"] != CPU_SHA256
        or prepared["parent_metadata_sha256"] != metadata
        or prepared["changed_files"] != [OWNER_FILE]
        or prepared["files_sha256"] != cpu["files_sha256"]
        or prepared["helpers_sha256"] != helpers
        or prepared["new_helpers_sha256"] != new_helpers
        or prepared["scope"] != OWNER_SCOPE
        or prepared["predecessor_binding"] != cpu["predecessor_binding"]
        or prepared["predecessor_seal_identity"] != cpu["predecessor_seal_identity"]
        or {
            name
            for name in original_files.keys() | cpu["files_sha256"].keys()
            if original_files.get(name) != cpu["files_sha256"].get(name)
        }
        != {OWNER_FILE}
    ):
        raise ValueError("Accepted owner CPU source, predecessor, owner-only transformation or command differs")
    predecessor = cpu["predecessor_binding"]
    previous_helpers = (root / PARENT_CPU).parent / "tracking"
    if (
        predecessor["cpu_receipt"] != str(root / PARENT_CPU)
        or predecessor["cpu_receipt_sha256"] != PARENT_SHA256
        or predecessor["seal"] != str(root / SEAL)
        or predecessor["seal_sha256"] != SEAL_SHA256
        or predecessor["verifier"] != str(previous_helpers / "ns_training_path_predecessor.py")
        or predecessor["checker"] != str(previous_helpers / "prepare_ns_training_path_predecessor.py")
        or any(
            predecessor[f"{key}_sha256"] != helpers[name]
            for key, name in (
                ("verifier", "ns_training_path_predecessor.py"),
                ("checker", "prepare_ns_training_path_predecessor.py"),
            )
        )
        or _stat_identity(root / SEAL) != cpu["predecessor_seal_identity"]
    ):
        raise ValueError("Accepted owner predecessor receipt, verifier or seal identity differs")
    _checked(root / SEAL, SEAL_SHA256, records)
    sealed = _saved(root / SEAL, records)
    if (
        sealed["status"] != "original_ns_and_failed_queues_sealed_for_distinct_training_path_diagnostic"
        or sealed["root"] != str(root)
        or sealed["gpu_execution_authorized"] is not False
        or sealed["production_authorized"] is not False
    ):
        raise ValueError("Accepted owner predecessor seal status or authority differs")
    for key in ("verifier", "checker"):
        _checked(Path(predecessor[key]), predecessor[f"{key}_sha256"], records)
    for name, filename, digest in (
        ("archive", "source.tar", OWNER_ARCHIVE_SHA256),
        ("tests", "tests.xml", OWNER_TESTS_SHA256),
        ("log", "cpu.log", OWNER_LOG_SHA256),
    ):
        target = directory / filename
        if cpu[name] != str(target) or cpu[f"{name}_sha256"] != digest:
            raise ValueError(f"Accepted owner CPU {name} binding differs")
        _checked(target, digest, records)
    if {path.name for path in (directory / "tracking").iterdir()} != set(helpers):
        raise ValueError("Accepted owner archived helper inventory differs")
    for name, digest in inventory.items():
        _checked(_source_path(directory, name), digest, records)
    _archive(directory / "source.tar", inventory)
    for folder, files in ((candidate, cpu["files_sha256"]), (original, original_files)):
        if _source_inventory(folder) != set(files):
            raise ValueError("Accepted owner complete source/config namespace differs")
        for name, digest in files.items():
            _checked(folder / name, digest, records)
        for name, digest in metadata.items():
            _checked(folder / name, digest, records)
    _checked(root / "data/experiment_receipt.json", prepared["data_receipt_sha256"], records)
    if prepared["commands"] != contract.commands(candidate, root, contract.locations(root)[3]):
        raise ValueError("Accepted owner exact diagnostic plan differs")
    tests = _tests(directory / "tests.xml", OWNER_TEST_COUNTS)
    return cpu, prepared, sealed | {"reporting_cpu_tests": tests}


def _plan(root: Path, candidate: Path) -> dict:
    work = contract.locations(root)[3]
    commands = contract.commands(candidate, root, work)
    return {
        "root": str(root),
        "candidate": str(candidate),
        "world": contract.WORLD,
        "environment": contract.ENVIRONMENT,
        "commands": commands,
        "trials": [
            spec
            | {
                "config_sha256": _sha256(candidate / spec["config_name"]),
                "continuous": str(work / f"{spec['name']}_continuous"),
                "resumed": str(work / f"{spec['name']}_resumed"),
                "commands": commands[index * 3 : index * 3 + 3],
            }
            for index, spec in enumerate(contract.specs())
        ],
        "planned_tokens": len(contract.specs()) * 2 * contract.ENDPOINT * contract.BATCH_TOKENS,
        "checkpoint_steps": list(contract.CHECKPOINTS),
        "tolerance": contract.TOLERANCE,
        "training_sources_sha256": contract.TRAINING_FILES,
        "numerical_readers_sha256": contract.NUMERICAL_READERS,
        "scope": contract.SCOPE,
    }


def _controller_command(root: Path) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(contract.locations(root)[1] / "ns_training_path_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "execute",
    ]


def _validator_command(root: Path, candidate: Path) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(contract.locations(root)[1] / "ns_training_path_validation.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--phase",
        "execute",
    ]


def _binding(root: Path, cpu: dict, prepared: dict, sealed: dict, records: dict) -> dict:
    path, helpers, _, _, _ = contract.locations(root)
    _checked(path, BINDING_SHA256, records)
    frozen = _saved(path, records)
    candidate, parent = Path(cpu["candidate"]), Path(prepared["parent_candidate"])
    seal = {key: value for key, value in sealed.items() if key != "reporting_cpu_tests"}
    original = _receipt(root / CPU, CPU_SHA256)
    parent_files = {
        name: digest
        for name, digest in original["inventory_sha256"].items()
        if name.startswith(("experiment/", f"{PINNED}/"))
    }
    predecessor = cpu["predecessor_binding"]
    acceptance = {
        "cpu_receipt": predecessor["cpu_receipt"],
        "cpu_receipt_sha256": predecessor["cpu_receipt_sha256"],
        "verifier_sha256": predecessor["verifier_sha256"],
        "sealer_sha256": predecessor["checker_sha256"],
        "seal": predecessor["seal"],
        "seal_sha256": predecessor["seal_sha256"],
    }
    if (
        frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != cpu["helpers_sha256"]
        or frozen["files_sha256"] != cpu["files_sha256"]
        or frozen["parent_candidate"] != str(parent)
        or frozen["parent_files_sha256"] != parent_files
        or frozen["allowed_transform"] != OWNER_FILE
        or frozen["preparation_sha256"] != cpu["preparation_sha256"]
        or frozen["cpu_receipt"] != str(root / OWNER_CPU)
        or frozen["cpu_receipt_sha256"] != OWNER_CPU_SHA256
        or frozen["archive"] != str(root / ISSUED_ARCHIVE)
        or frozen["archive_sha256"] != ISSUED_ARCHIVE_SHA256
        or frozen["data_receipt_sha256"] != prepared["data_receipt_sha256"]
        or frozen["allocation_predecessor"] != seal
        or frozen["predecessor_acceptance"] != acceptance
        or frozen["devices"] != seal["restored_keeper"]["devices"]
        or frozen["node"] != "28.216.17.9"
        or frozen["environment"] != contract.ENVIRONMENT
        or frozen["controller_environment"] != CPU_ENVIRONMENT
        or frozen["controller_command"] != _controller_command(root)
        or frozen["validator_command"] != _validator_command(root, candidate)
        or frozen["validation_timeout"] != 63000
        or frozen["scope"] != OWNER_SCOPE
        or frozen["plan"] != _plan(root, candidate)
        or frozen["commands"] != frozen["plan"]["commands"]
    ):
        raise ValueError("Issued binding differs from accepted owner/source/data/predecessor protocol")
    if {path.name for path in helpers.iterdir()} != set(frozen["helpers_sha256"]):
        raise ValueError("Issued helper directory has an unbound or missing inventory entry")
    _checked(root / ISSUED_ARCHIVE, frozen["archive_sha256"], records)
    _archive(root / ISSUED_ARCHIVE, cpu["inventory_sha256"])
    required = {
        root / OWNER_CPU: OWNER_CPU_SHA256,
        root / ISSUED_ARCHIVE: frozen["archive_sha256"],
        root / PARENT_CPU: PARENT_SHA256,
        root / SEAL: SEAL_SHA256,
        candidate / OWNER_PREPARATION: cpu["preparation_sha256"],
        root / "data/experiment_receipt.json": prepared["data_receipt_sha256"],
    }
    for folder, inventory in (
        (candidate, cpu["files_sha256"]),
        (parent, parent_files),
        (helpers, cpu["helpers_sha256"]),
        ((root / OWNER_CPU).parent, cpu["inventory_sha256"]),
    ):
        for name, digest in inventory.items():
            target = _source_path(folder, name)
            required[target] = digest
            _checked(target, digest, records)
    inventory = frozen["authenticated_files"]
    for target, digest in required.items():
        item = inventory.get(str(target.resolve(strict=True)), {})
        if item.get("sha256") != digest or item.get("identity") != _stat_identity(target):
            raise ValueError("Issued authenticated source/data/helper/receipt inventory differs")
    for name, identity in seal["file_identity"].items():
        target = root / _relative(name)
        item = inventory.get(str(target), {})
        if item != {"identity": identity, "sha256": seal["evidence_sha256"][name]}:
            raise ValueError("Issued predecessor payload identity or retained digest differs")
    for name, item in inventory.items():
        target = Path(name)
        if not target.is_relative_to(root) or item["identity"] != _stat_identity(target):
            raise ValueError("Issued operation inventory changed; payloads are not reread")
    return frozen


def _generation(value: dict, command: list[str], pid: int, start: str, parent: int | None = None) -> None:
    if (
        type(pid) is not int
        or pid <= 0
        or not start
        or value["pid"] != pid
        or value["start_ticks"] != start
        or value["command"] != command
        or value["group"] != pid
        or value["session"] != pid
        or (parent is not None and value["parent"] != parent)
    ):
        raise ValueError("Saved original process generation or direct ancestry differs")


def _command(entry: dict, command: list[str], log: Path, validator_pid: int, records: dict) -> None:
    stored = _saved(log.with_suffix(".execution.json"), records)
    if {key: value for key, value in stored.items() if key != "time"} != {
        key: value for key, value in entry.items() if key not in ("trial", "mode", "trajectory", "time")
    }:
        raise ValueError("Saved command differs from its canonical execution sidecar")
    generation = entry["generation"]
    _generation(generation, command, generation["pid"], generation["start_ticks"], validator_pid)
    if (
        entry["command"] != command
        or entry["spawn_status"] != "acknowledged"
        or not entry["marker"]
        or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
        or entry["marked_descendants_absent"] is not True
        or entry["log"] != str(log)
        or any(key in entry for key in ("error", "cleanup_error"))
    ):
        raise ValueError("Saved command exit, spawn acknowledgment or cleanup is incomplete")
    _checked(log, entry["log_sha256"], records)


def _is_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(letter in "0123456789abcdef" for letter in value)


def _trial_predicates(case: dict) -> None:
    """Check retained reader coverage and predicates without reopening trajectories or tensors."""
    trajectories = case["trajectories"]
    if set(trajectories) != set(contract.MODES) or case["gates"]["actual_config_complete_trajectories"] is not True:
        raise ValueError("Saved case lacks complete retained trajectory coverage")
    rank_names = {f"rank_{rank}.pt" for rank in range(contract.WORLD)}
    for mode, trajectory in trajectories.items():
        endpoint = contract.BOUNDARY if mode == "interrupted" else contract.ENDPOINT
        steps = {str(step) for step in contract.CHECKPOINTS if step <= endpoint}
        history_names = {"metrics_from_000000.jsonl"}
        if mode == "resumed":
            history_names.add(f"metrics_from_{contract.BOUNDARY:06d}.jsonl")
        if (
            type(trajectory["updates"]) is not int
            or trajectory["updates"] != endpoint
            or type(trajectory["tokens"]) is not int
            or trajectory["tokens"] != endpoint * contract.BATCH_TOKENS
            or type(trajectory["fixed_age_installations"]) is not int
            or trajectory["fixed_age_installations"] <= 0
            or not _is_digest(trajectory["manifest_sha256"])
            or set(trajectory["history_sha256"]) != history_names
            or not all(_is_digest(value) for value in trajectory["history_sha256"].values())
            or set(trajectory["checkpoints"]) != steps
        ):
            raise ValueError("Saved trajectory endpoint, history or checkpoint coverage differs")
        for checkpoint in trajectory["checkpoints"].values():
            if (
                not _is_digest(checkpoint["receipt_sha256"])
                or len(checkpoint["rank_bytes"]) != contract.WORLD
                or any(type(value) is not int or value <= 0 for value in checkpoint["rank_bytes"])
                or set(checkpoint["rank_sha256"]) != rank_names
                or not all(_is_digest(value) for value in checkpoint["rank_sha256"].values())
            ):
                raise ValueError("Saved trajectory lacks complete retained all-rank checkpoint evidence")
    continuation = case["continuation"]
    if set(continuation) != {str(step) for step in contract.CHECKPOINTS} or any(
        type(continuation[str(step)]["step"]) is not int
        or continuation[str(step)]["step"] != step
        or type(continuation[str(step)]["within_tolerance"]) is not bool
        for step in contract.CHECKPOINTS
    ):
        raise ValueError("Saved continuation checkpoint predicates or diagnostic coverage differ")
    if (
        type(case["loss"]["within_tolerance"]) is not bool
        or case["gates"]["boundary_all_rank_continuation"] != continuation["22"]["within_tolerance"]
        or case["gates"]["post_cached_root_all_rank_continuation"] != continuation["24"]["within_tolerance"]
        or case["gates"]["scalar_loss_continuation"] != case["loss"]["within_tolerance"]
    ):
        raise ValueError("Saved acceptance gates contradict retained checkpoint or loss predicates")
    capture = case["capture"]
    if case["gates"]["captured_row_update_work"] is not True:
        raise ValueError("Saved captured-row/update/work gate contradicts the reader schema")
    if case["family"] == "muont":
        if capture != {}:
            raise ValueError("Saved MuonT case has unexpected fused capture evidence")
        return
    checks = ["diagonal_capture_parity_passed", "outer_diagonal_update_parity_passed", "output_work_retained_passed"]
    if case["family"] == "null_leftdiag_fused":
        checks.append("null_identity_outer_passed")
    if set(capture) != {"continuous", "resumed"} or any(
        type(value["updates"]) is not int or value["updates"] != contract.ENDPOINT or value["checks"] != checks
        for value in capture.values()
    ):
        raise ValueError("Saved fused/null captured-row/update/work coverage differs")


def _decisions(result: dict, planned: dict, completed: int) -> dict:
    cases = result["results"]
    if len(cases) > completed // 3:
        raise ValueError("Saved case decision lacks three completed commands")
    for index, case in enumerate(cases):
        if (
            any(case[key] != value for key, value in planned["trials"][index].items())
            or set(case["gates"]) != GATES
            or any(type(value) is not bool for value in case["gates"].values())
            or type(case["accepted"]) is not bool
            or case["accepted"] != all(case["gates"].values())
            or case["first_resumed_diagnostic_only"] is not True
        ):
            raise ValueError("Saved case protocol, gate predicate or retained decision differs")
        _trial_predicates(case)
    accepted = [case["name"] for case in cases if case["accepted"]]
    rejected = [case["name"] for case in cases if not case["accepted"]]
    if result["status"] == RESULT_STATUS and (
        len(cases) != 18
        or completed != 54
        or result["accepted_cases"] != accepted
        or result["rejected_cases"] != rejected
        or type(result["all_cases_accepted"]) is not bool
        or result["all_cases_accepted"] != (not rejected)
        or "error" in result
    ):
        raise ValueError("Saved reconstructed result has incomplete coverage or altered rejection lists")
    return {
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "all_cases_accepted": not rejected if len(cases) == 18 else None,
        "cases": cases,
    }


def _same_keeper(actual: dict, expected: dict) -> bool:
    """Mirror issuer keeper identity, allowing only nonterminal scheduler-state transitions."""
    before = {int(pid): process for pid, process in expected["generation"].items()}
    after = {int(pid): process for pid, process in actual["generation"].items()}
    return (
        actual["pid"] == expected["pid"]
        and actual["devices"] == expected["devices"]
        and set(before) == set(after)
        and all(
            process["status"] not in ("Z", "X", "x")
            and after[pid]["status"] not in ("Z", "X", "x")
            and {key: value for key, value in process.items() if key != "status"}
            == {key: value for key, value in after[pid].items() if key != "status"}
            for pid, process in before.items()
        )
    )


def _terminal(root: Path, frozen: dict, intent: dict, state: dict | None, result: dict | None, records: dict) -> dict:
    directory = contract.locations(root)[1].parent
    token = hashlib.sha256(
        json.dumps(
            {
                "binding_sha256": BINDING_SHA256,
                "controller_pid": intent["pid"],
                "controller_start_ticks": intent["start_ticks"],
                "controller_marker": intent["marker"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    output = root / "run-preparation" / f"ns-training-path-terminal-{token}.json"
    attempt_path, outcome_path = directory / "terminal-intent.json", directory / "terminal-outcome.json"
    if not any(path.exists() or path.is_symlink() for path in (output, attempt_path, outcome_path)):
        return {"status": "not_recorded"}
    attempt = _saved(attempt_path, records)
    verifier = frozen["helpers_sha256"]["ns_training_path_gpu_controller.py"]
    if (
        attempt["status"] != "reconstruction_claimed"
        or attempt["binding_sha256"] != BINDING_SHA256
        or attempt["controller_pid"] != intent["pid"]
        or attempt["controller_start_ticks"] != intent["start_ticks"]
        or attempt["controller_marker"] != intent["marker"]
        or attempt["output"] != str(output)
        or attempt["verifier_sha256"] != verifier
    ):
        raise ValueError("One-shot terminal intent is not bound to the original controller/verifier")
    if not outcome_path.exists():
        if output.exists():
            raise ValueError("Terminal receipt exists without a completed one-shot outcome")
        return {"status": "reconstruction_claimed", "time": attempt.get("time"), "accepted": False}
    outcome = _saved(outcome_path, records)
    if (
        outcome["output"] != str(output)
        or outcome["reconstruction_intent_sha256"] != records[attempt_path][0]
        or outcome["status"] not in ("complete", "needs_attention")
    ):
        raise ValueError("One-shot terminal outcome differs from its bound intent")
    if outcome["status"] == "needs_attention":
        return {
            "status": "needs_attention",
            "error": outcome.get("error"),
            "time": outcome.get("time"),
            "receipt_present": output.exists(),
            "accepted": False,
        }
    if state is None or result is None:
        return {
            "status": "evidence_refused",
            "claimed_status": outcome["status"],
            "error": "One-shot terminal completion claim lacks the original state or result receipt",
            "time": outcome.get("time"),
            "receipt_present": output.exists(),
            "accepted": False,
        }
    terminal = _saved(output, records)
    if (
        outcome["terminal_sha256"] != records[output][0]
        or "error" in outcome
        or terminal["status"] != TERMINAL_STATUS
        or terminal["binding_sha256"] != BINDING_SHA256
        or terminal["result_sha256"] != state["result_sha256"]
        or terminal["controller_pid_absent"] != intent["pid"]
        or terminal["controller_start_ticks"] != intent["start_ticks"]
        or terminal["controller_marker"] != intent["marker"]
        or terminal["validator_pid_absent"] != state["child_pid"]
        or terminal["reconstruction_intent_sha256"] != records[attempt_path][0]
        or terminal["verifier_sha256"] != verifier
        or terminal["production_authorized"] is not False
        or terminal["predecessor_acceptance"] != frozen["predecessor_acceptance"]
        or terminal["scope"] != OWNER_SCOPE
        or terminal["terminal_time"] != state["time"]
        or state["status"] != "validation_complete"
        or any(terminal[name] != result[name] for name in ("accepted_cases", "rejected_cases", "all_cases_accepted"))
        or not _same_keeper(terminal["keeper"], state["restored_keeper"])
        or terminal["keeper"]["devices"] != frozen["devices"]
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state["keeper_release_attempted"] is not True
        or state["keeper_release_complete"] is not True
        or state["marked_descendants_absent"] is not True
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
    ):
        raise ValueError("One-shot terminal decisions, generation, cleanup or recorded restoration differs")
    paths = (
        contract.locations(root)[0],
        contract.locations(root)[2],
        directory / "controller.json",
        directory / "controller.log",
        directory / "validator.log",
        contract.locations(root)[4],
    )
    if set(terminal["evidence_sha256"]) != {str(path.relative_to(root)) for path in paths}:
        raise ValueError("One-shot terminal evidence inventory is incomplete")
    for name, digest in terminal["evidence_sha256"].items():
        _checked(root / _relative(name), digest, records)
    times = [datetime.fromisoformat(value["time"]) for value in (intent, result, state, attempt, terminal, outcome)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("One-shot terminal chronology differs")
    return {
        "status": "complete",
        "receipt": str(output),
        "receipt_sha256": records[output][0],
        "time": terminal["time"],
        "accepted_cases": terminal["accepted_cases"],
        "rejected_cases": terminal["rejected_cases"],
        "all_cases_accepted": terminal["all_cases_accepted"],
        "saved_exact_restoration": True,
    }


def _execution(root: Path, frozen: dict, records: dict) -> dict:
    _, helpers, state_path, work, result_path = contract.locations(root)
    path = helpers.parent / "controller.json"
    if not path.exists():
        if state_path.exists() or result_path.exists():
            raise ValueError("Saved execution evidence has no original controller intent")
        return {"status": "frozen_unlaunched", "completed_commands": 0, "terminal": {"status": "not_recorded"}}
    if INTENT_SHA256 is None:
        return {"status": "acknowledgment_unbound", "completed_commands": None}
    _checked(path, INTENT_SHA256, records)
    intent = _saved(path, records)
    proof = hashlib.sha256(
        json.dumps(
            {
                "binding_sha256": BINDING_SHA256,
                "cpu_receipt_sha256": OWNER_CPU_SHA256,
                "inventory_sha256": (
                    hashlib.sha256(
                        json.dumps(frozen["authenticated_files"], sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if (
        intent["binding_sha256"] != BINDING_SHA256
        or intent["command"] != _controller_command(root)
        or intent["parent_proof"] != proof
        or not intent["marker"]
        or intent["status"] not in ("starting", "acknowledged", "needs_attention")
    ):
        raise ValueError("Saved controller intent or source-bound parent proof differs")
    if intent.get("log_sha256") is not None:
        _checked(helpers.parent / "controller.log", intent["log_sha256"], records)
    if intent["status"] != "acknowledged":
        if state_path.exists() or result_path.exists():
            raise ValueError("Saved execution lacks an acknowledged original controller")
        return {
            "status": intent["status"],
            "time": intent.get("time"),
            "error": intent.get("error"),
            "completed_commands": 0,
            "terminal": {"status": "not_recorded"},
        }
    _generation(intent["generation"], intent["command"], intent["pid"], intent["start_ticks"])
    progress = {
        "status": "acknowledged",
        "time": intent.get("time"),
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "completed_commands": 0,
        "terminal": {"status": "not_recorded"},
    }
    if not state_path.exists():
        progress["terminal"] = _terminal(root, frozen, intent, None, None, records)
        return progress
    state = _saved(state_path, records)
    if (
        state["binding_sha256"] != BINDING_SHA256
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["allocation_predecessor"] != frozen["allocation_predecessor"]
        or state["status"] not in ("preflight", "validating", "validation_complete", "needs_attention")
        or ("error" in state and state["status"] != "needs_attention")
    ):
        raise ValueError("Saved controller state has changed original attribution or phase")
    progress.update(
        status=state["status"],
        time=state.get("time"),
        spawn_status=state.get("spawn_status"),
        error=state.get("error"),
        cleanup_error=state.get("cleanup_error"),
        keeper_restore=state.get("keeper_restore"),
    )
    candidate = Path(frozen["candidate"])
    if "command" in state and (
        state["command"] != _validator_command(root, candidate)
        or state["log"] != str(helpers.parent / "validator.log")
        or not state["child_marker"]
        or state["spawn_status"] not in ("starting", "acknowledged")
    ):
        raise ValueError("Saved validator command, spawn attribution or log path differs")
    if state.get("spawn_status") == "acknowledged":
        _generation(
            state["child_generation"], state["command"], state["child_pid"], state["child_start_ticks"], intent["pid"]
        )
        progress.update(validator_pid=state["child_pid"], validator_start_ticks=state["child_start_ticks"])
    if state.get("log_sha256") is not None:
        _checked(Path(state["log"]), state["log_sha256"], records)
        progress["validator_log_sha256"] = state["log_sha256"]
    if state.get("active_command") is not None:
        if state["active_command"] not in frozen["commands"]:
            raise ValueError("Saved active command is outside the exact frozen plan")
        progress["last_saved_command"] = state["active_command"]
        progress["last_saved_command_index"] = frozen["commands"].index(state["active_command"]) + 1
    if not result_path.exists():
        progress["terminal"] = _terminal(root, frozen, intent, state, None, records)
        if state["status"] == "validation_complete":
            progress.update(
                status="evidence_refused", completed_commands=None, error="Saved completion lacks a result receipt"
            )
        return progress
    result = _saved(result_path, records)
    planned = frozen["plan"]
    if (
        state.get("spawn_status") != "acknowledged"
        or any(result[key] != value for key, value in planned.items())
        or result["binding_sha256"] != BINDING_SHA256
        or result["owner"]
        != {
            "controller_pid": intent["pid"],
            "controller_start_ticks": intent["start_ticks"],
            "validator_pid": state["child_pid"],
            "validator_start_ticks": state["child_start_ticks"],
        }
        or result["status"] not in ("validating", "needs_attention", RESULT_STATUS)
        or ("error" in result and result["status"] != "needs_attention")
        or len(result["gpu_commands"]) > 54
    ):
        raise ValueError("Saved GPU receipt owner, status or original protocol differs")
    command_rows = []
    for index, entry in enumerate(result["gpu_commands"]):
        trial, mode = planned["trials"][index // 3], contract.MODES[index % 3]
        log = work / f"{trial['name']}_{mode}.log"
        if entry["trial"] != trial["name"] or entry["mode"] != mode:
            raise ValueError("Saved command history trial order differs")
        _command(entry, frozen["commands"][index], log, state["child_pid"], records)
        command_rows.append(
            {
                "trial": entry["trial"],
                "mode": mode,
                "pid": entry["generation"]["pid"],
                "start_ticks": entry["generation"]["start_ticks"],
                "log_sha256": entry["log_sha256"],
                "receipt_sha256": records[log.with_suffix(".execution.json")][0],
            }
        )
    decisions = _decisions(result, planned, len(command_rows))
    if state["status"] == "validation_complete" and (
        result["status"] != RESULT_STATUS
        or state["result_sha256"] != records[result_path][0]
        or any(state[name] != decisions[name] for name in ("accepted_cases", "rejected_cases", "all_cases_accepted"))
        or any(type(state.get(key)) is not int or state[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
    ):
        raise ValueError("Saved controller reconstruction disagrees with result or command coverage")
    progress.update(
        completed_commands=len(command_rows),
        commands=command_rows,
        result_status=result["status"],
        result_sha256=records[result_path][0],
        decisions=decisions,
        result_error=result.get("error"),
    )
    progress["terminal"] = _terminal(root, frozen, intent, state, result, records)
    return progress


def _followup(root: Path, frozen: dict, progress: dict, records: dict) -> dict:
    """Authenticate saved waiter observations without invoking its process or tensor checks."""
    work, cpu_path = root / FOLLOWUP_WORK, root / FOLLOWUP_CPU
    intent_path, state_path = work / "intent.json", work / "state.json"
    if not intent_path.exists() and not intent_path.is_symlink():
        return {"status": "not_recorded"}
    _checked(cpu_path, FOLLOWUP_CPU_SHA256, records)
    _checked(intent_path, FOLLOWUP_INTENT_SHA256, records)
    cpu, intent, state = (_saved(path, records) for path in (cpu_path, intent_path, state_path))
    helpers, directory = cpu_path.parent / "tracking", cpu_path.parent
    files = cpu["helpers_sha256"]
    runner = helpers / "ns_training_path_followup.py"
    command = [
        contract.PYTHON,
        "-P",
        str(runner),
        "--root",
        str(root),
        "--cpu",
        str(cpu_path),
        "--phase",
        "execute",
    ]
    if (
        cpu["status"] != "ns_training_path_read_only_followup_cpu_archive_verified"
        or cpu["root"] != str(root)
        or cpu["binding_sha256"] != BINDING_SHA256
        or cpu["original_intent_sha256"] != INTENT_SHA256
        or cpu["original_archive_sha256"] != ISSUED_ARCHIVE_SHA256
        or cpu["environment"] != FOLLOWUP_ENVIRONMENT
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["scope"] != FOLLOWUP_SCOPE
        or cpu["helpers"] != str(helpers)
        or {name: files.get(name) for name in frozen["helpers_sha256"]} != frozen["helpers_sha256"]
        or set(files) - set(frozen["helpers_sha256"])
        != {"ns_training_path_followup.py", "test_ns_training_path_followup.py"}
        or {path.name for path in helpers.iterdir()} != set(files)
        or cpu["command"] != _cpu_command(directory, ("test_ns_training_path_followup",))
        or intent["status"] != "acknowledged"
        or intent["binding_sha256"] != BINDING_SHA256
        or intent["cpu_receipt_sha256"] != FOLLOWUP_CPU_SHA256
        or intent["runner_sha256"] != files[runner.name]
        or intent["command"] != command
        or intent["generation"]["command"] != command
        or intent["generation"]["pid"] != intent["pid"]
        or intent["generation"]["start_ticks"] != intent["start_ticks"]
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or state["binding_sha256"] != BINDING_SHA256
        or state["cpu_receipt_sha256"] != FOLLOWUP_CPU_SHA256
        or state["runner_sha256"] != files[runner.name]
        or any(value["scope"] != FOLLOWUP_SCOPE or value["interval"] != 900 for value in (intent, state))
        or type(state["observations"]) is not int
        or state["observations"] < 1
    ):
        raise ValueError("Training-path saved terminal waiter source, owner or receipt differs")
    for name, digest in files.items():
        _checked(helpers / name, digest, records)
    for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log")):
        path = directory / filename
        if cpu[name] != str(path):
            raise ValueError("Training-path terminal waiter archive/test/log path differs")
        _checked(path, cpu[f"{name}_sha256"], records)
    tests = _tests(Path(cpu["tests"]), {"test_ns_training_path_followup": 19})
    with tarfile.open(cpu["archive"]) as bundle:
        members = bundle.getmembers()
        expected = {f"tracking/{name}": digest for name, digest in files.items()}
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise ValueError("Training-path terminal waiter archive membership differs")
        for member in members:
            if not member.isfile():
                raise ValueError("Training-path terminal waiter archive contains an alias")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected[member.name]:
                    raise ValueError("Training-path terminal waiter archived helper differs")
    observation = _saved(work / f"observation-{state['observations'] - 1:06d}.json", records)
    if observation != state["original_observation"]:
        raise ValueError("Training-path terminal waiter latest observation differs")
    original = _saved(contract.locations(root)[1].parent / "controller.json", records)
    owner_state = _saved(contract.locations(root)[2], records)
    for name, generation in (("controller", original["generation"]), ("validator", owner_state["child_generation"])):
        reported = observation["owners"][name]
        expected = reported.get("expected", reported.get("observed"))
        if any(expected[key] != generation[key] for key in ("pid", "start_ticks", "command", "group", "session")):
            raise ValueError("Training-path terminal waiter observed another original generation")
    if state["status"] == "terminal_reconstructed":
        terminal = progress.get("terminal", {})
        generation = state.get("terminal_generation", {})
        terminal_command = [
            contract.PYTHON,
            "-P",
            str(Path(frozen["helpers"]) / "ns_training_path_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            "terminal",
            "--out",
            terminal.get("receipt", ""),
        ]
        if (
            terminal.get("status") != "complete"
            or state["terminal_sha256"] != terminal["receipt_sha256"]
            or state["terminal"] != terminal["receipt"]
            or state["terminal_command"] != terminal_command
            or generation.get("command") != terminal_command
            or generation.get("pid") != state["terminal_pid"]
            or generation.get("parent") != state["pid"]
            or generation.get("group") != state["terminal_pid"]
            or generation.get("session") != state["terminal_pid"]
            or any(state[key] != terminal[key] for key in ("accepted_cases", "rejected_cases", "all_cases_accepted"))
            or any(
                type(state.get(key)) is not int or state[key] != 0
                for key in ("terminal_exit_code", "terminal_cleanup_exit_code")
            )
            or state["terminal_descendants_absent"] is not True
            or "error" in state
        ):
            raise ValueError("Training-path terminal waiter claims completion without accepted original terminal")
        log = work / "terminal.log"
        if state["terminal_log"] != str(log):
            raise ValueError("Training-path terminal waiter completed log path differs")
        _checked(log, state["terminal_log_sha256"], records)
        _checked(contract.locations(root)[1].parent / "terminal-outcome.json", state["terminal_outcome_sha256"], records)
    elif state["status"] not in (
        "waiting_for_original_owners",
        "waiting_for_original_descendants",
        "terminal_ready",
        "reconstructing_terminal",
        "needs_attention",
    ):
        raise ValueError("Training-path terminal waiter state is unknown")
    return {
        "status": state["status"],
        "pid": state["pid"],
        "start_ticks": state["start_ticks"],
        "time": state.get("time"),
        "observations": state["observations"],
        "saved_original_owners": {name: item["status"] for name, item in observation["owners"].items()},
        "error": state.get("error"),
        "cpu_time": cpu.get("time"),
        **tests,
    }


def evidence(root: Path) -> dict:
    """Report independent CPU, owner and saved execution evidence without live verifiers."""
    saved = _numerical_evidence(root)
    path = root / OWNER_CPU
    if not path.exists() and not path.is_symlink():
        saved["execution_owner"]["status"] = "accepted_receipt_not_present"
        return saved
    records = {}
    try:
        cpu, prepared, sealed = _owner_cpu(root, records)
        if any(_identity(path) != identity for path, (_, identity) in records.items()):
            raise ValueError("Owner CPU evidence changed during reporting")
    except EVIDENCE_ERRORS as failure:
        saved["execution_owner"] = {
            "status": "evidence_refused",
            "accepted": False,
            "error": f"{type(failure).__name__}: {failure}",
        }
        return saved
    saved["execution_owner"] = {
        "status": cpu["status"],
        "accepted": True,
        "receipt": str(root / OWNER_CPU),
        "receipt_sha256": OWNER_CPU_SHA256,
        "candidate": cpu["candidate"],
        "archive_sha256": OWNER_ARCHIVE_SHA256,
        "tests_sha256": OWNER_TESTS_SHA256,
        "log_sha256": OWNER_LOG_SHA256,
        "helpers": len(cpu["helpers_sha256"]),
        "inventory": len(cpu["inventory_sha256"]),
        "time": cpu.get("time"),
        **sealed["reporting_cpu_tests"],
    }
    binding_path = contract.locations(root)[0]
    if not binding_path.exists() and not binding_path.is_symlink():
        saved["gpu_progress"] = {"status": "gpu_unissued", "completed_commands": 0}
        return saved
    if BINDING_SHA256 is None:
        saved["gpu_progress"] = {"status": "binding_unbound", "completed_commands": None}
        return saved
    try:
        frozen = _binding(root, cpu, prepared, sealed, records)
        progress = _execution(root, frozen, records)
        if any(_identity(path) != identity for path, (_, identity) in records.items()):
            raise ValueError("Saved execution evidence changed during reporting")
        saved["gpu_progress"] = progress
        try:
            saved["terminal_followup"] = _followup(root, frozen, progress, records)
        except EVIDENCE_ERRORS as failure:
            saved["terminal_followup"] = {
                "status": "evidence_refused",
                "error": f"{type(failure).__name__}: {failure}",
            }
        if progress.get("terminal", {}).get("status") == "complete":
            saved["numerical_gpu_acceptance"] = {
                "status": "saved_independent_terminal_receipt",
                **{
                    name: progress["terminal"][name]
                    for name in ("accepted_cases", "rejected_cases", "all_cases_accepted")
                },
            }
    except EVIDENCE_ERRORS as failure:
        saved["gpu_progress"] = {
            "status": "evidence_refused",
            "completed_commands": None,
            "error": f"{type(failure).__name__}: {failure}",
        }
    return saved


def lines(root: Path, local_time) -> list[str]:
    """Render CPU/prelaunch evidence without claims from unaccepted owner files."""
    path = root / CPU
    if not path.exists() and not path.is_symlink() and not (root / PARENT_CPU).exists():
        return []
    heading = [f"### {TITLE}", ""]
    try:
        saved = evidence(root)
    except EVIDENCE_ERRORS as failure:
        return [
            *heading,
            f"Saved numerical CPU evidence refused: `{type(failure).__name__}: {failure}`.",
            "CPU acceptance and GPU progress/results withheld; original receipts remain preserved.",
            "",
        ]
    cpu = saved["numerical_cpu"]
    result = [
        *heading,
        f"Saved separate numerical CPU/source-archive acceptance at {local_time(cpu['time'])}: "
        f"{cpu['passed_cases']} passing CPU cases in {cpu['seconds']:.3f} seconds; "
        f"{cpu['helpers']} authenticated helpers and {cpu['inventory']} inventory entries.",
        f"Accepted numerical receipt: `PVC/{CPU}`; inherited numerical candidate, not an accepted execution owner.",
        f"Receipt SHA256 `{cpu['receipt_sha256']}`; archive SHA256 `{cpu['archive_sha256']}`.",
        f"CPU XML SHA256 `{cpu['tests_sha256']}`; log SHA256 `{cpu['log_sha256']}`; "
        f"test source SHA256 `{cpu['test_source_sha256']}`.",
        "The distinct plan has eighteen lower-LR cases and 54 bounded eight-rank commands "
        "(continuous24/interrupted22/resumed24); these are planned commands, not GPU executions or accepted GPU cases.",
    ]
    owner, progress = saved["execution_owner"], saved["gpu_progress"]
    if owner["accepted"]:
        result.extend(
            [
                f"Saved separately accepted execution-owner CPU archive at {local_time(owner['time'])}: "
                f"{owner['passed_cases']} passing CPU cases in {owner['seconds']:.3f} seconds; "
                f"{owner['helpers']} authenticated helpers and {owner['inventory']} inventory entries.",
                f"Owner candidate: `{owner['candidate']}`; receipt SHA256 `{owner['receipt_sha256']}`; "
                f"archive SHA256 `{owner['archive_sha256']}`.",
                f"Owner CPU XML SHA256 `{owner['tests_sha256']}`; log SHA256 `{owner['log_sha256']}`.",
            ]
        )
    else:
        result.append(
            "No separately accepted training-path execution owner is established by this report. "
            "Acknowledgment, running progress, completion and numerical GPU acceptance are withheld."
        )
        if owner.get("error"):
            result.append(f"Owner CPU evidence refused: `{owner['error']}`.")
    if owner.get("unaccepted_saved_paths"):
        result.append(
            "Unaccepted saved execution evidence is present; its counts, decisions and status "
            "are withheld until separately bound acceptance. Original evidence remains preserved."
        )
    if progress:
        result.append(f"Saved execution phase `{progress['status']}` at {local_time(progress.get('time'))}.")
        if progress.get("completed_commands") is not None:
            result.append(
                f"{progress['completed_commands']}/54 authenticated zero-exit commands; "
                "command exits alone are not numerical acceptance."
            )
        if progress.get("controller_pid"):
            result.append(
                f"Saved controller PID/start `{progress['controller_pid']}/{progress['controller_start_ticks']}`; "
                f"validator spawn `{progress.get('spawn_status', 'not recorded')}`."
            )
        if progress.get("validator_pid"):
            result.append(
                f"Saved validator PID/start `{progress['validator_pid']}/{progress['validator_start_ticks']}`."
            )
        if progress.get("last_saved_command"):
            result.append(
                f"Last saved command {progress['last_saved_command_index']}/54: "
                f"`{json.dumps(progress['last_saved_command'])}`; not a fresh liveness observation."
            )
        for key in ("error", "cleanup_error", "result_error"):
            if progress.get(key):
                result.append(f"Preserved {key}: `{progress[key]}`.")
        decisions = progress.get("decisions")
        if decisions:
            result.append(
                f"Saved case predicates: {len(decisions['accepted_cases'])} accepted, "
                f"{len(decisions['rejected_cases'])} rejected; result status `{progress['result_status']}`. "
                "These are receipt predicates, not freshly reconstructed tensors."
            )
            for case in decisions["cases"]:
                false_gates = ",".join(name for name, value in case["gates"].items() if not value) or "none"
                result.append(
                    f"Case `{case['name']}`: saved accepted `{case['accepted']}`; " f"false gates `{false_gates}`."
                )
        terminal = progress.get("terminal", {})
        if terminal:
            result.append(
                f"One-shot terminal outcome `{terminal['status']}`; "
                "only a bound complete outcome records independent reconstruction and exact restoration."
            )
            if terminal.get("error"):
                result.append(f"Preserved terminal failure: `{terminal['error']}`.")
        if progress["status"] in ("binding_unbound", "acknowledgment_unbound", "evidence_refused"):
            result.append("Execution progress, numerical acceptance and terminal completion claims are withheld.")
    result.extend(
        [
            "Reporting checks issued file/archive evidence only; it does not reconstruct tensors, "
            "inspect current liveness or grant launch authority. Rejected historical controls remain rejected.",
            "CPU tests are not GPU results, optimizer superiority, causal proof, full-budget quality, "
            "whole-training efficiency, production authority or SwanLab acceptance.",
            "",
        ]
    )
    followup = saved.get("terminal_followup", {})
    if followup.get("status") not in (None, "not_recorded"):
        result[-1:-1] = [
            f"Saved read-only terminal waiter `{followup['status']}` at {local_time(followup.get('time'))}; "
            "not a fresh process check.",
        ]
        if followup.get("pid"):
            result[-1:-1] = [
                f"Waiter PID/start `{followup['pid']}/{followup['start_ticks']}`; "
                f"{followup['observations']} retained original-owner observations at 900-second intervals.",
                f"Saved original owner states: `{json.dumps(followup['saved_original_owners'], sort_keys=True)}`.",
            ]
        if followup.get("error"):
            result[-1:-1] = [f"Preserved read-only terminal waiter error: `{followup['error']}`."]
    return result
