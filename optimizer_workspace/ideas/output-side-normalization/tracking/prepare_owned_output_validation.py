# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare an observation and validation successor without issuing GPU authorization."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import PYTHON, read, sha256
from prepare_upstream_outer import replace_once
from watch_wave5 import write
from wave7_release import candidate_inventory

COPIES = {
    "owned_output_validation_observer.py": "experiment/owned_output_observer.py",
    "owned_output_validation_observer_checks.py": "experiment/tests/test_owned_output_observer.py",
    "owned_output_validation.py": "experiment/owned_output_validation.py",
    "test_owned_output_validation.py": "experiment/tests/test_owned_output_validation.py",
    "owned_output_gpu_checks.py": "experiment/owned_output_gpu_checks.py",
    "owned_output_execution_owner.py": "experiment/execution_owner.py",
    "test_owned_output_execution_owner.py": "experiment/tests/test_owned_output_execution_owner.py",
}
HELPERS = ("prepare_owned_output_validation.py", *COPIES)
CHANGES = {
    *COPIES.values(),
    "experiment/cluster.json",
    "experiment/validate.py",
    "experiment/tests/test_upstream_validation.py",
}
MASK = "0,1,2,3,4,5,6,7"
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)


def cpu_command() -> list[str]:
    return [
        PYTHON,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-m",
        SAFE_MARKERS,
        "experiment/tests",
        "-q",
    ]


def require_predecessor(receipt: Path) -> tuple[Path, dict]:
    """Require the immutable observation candidate and its real successful CPU result."""
    record = read(receipt)
    previous = Path(record["candidate"])
    if (
        record["status"] != "owned_output_observer_cpu_verified_gpu_denied"
        or type(record["exit_code"]) is not int
        or record["exit_code"] != 0
        or record["cuda_visible_devices"] != ""
        or record["command"] != cpu_command()
        or record["cwd"] != str(previous)
        or record["preparation"] != str(previous / "owned-output-observer-preparation.json")
        or record["preparation_sha256"] != sha256(Path(record["preparation"]))
        or record["files_sha256"] != candidate_inventory(previous)
        or record["archive_sha256"] != sha256(Path(record["archive"]))
        or record["log_sha256"] != sha256(Path(record["log"]))
        or record["predecessor_cpu_receipt_sha256"] != sha256(Path(record["predecessor_cpu_receipt"]))
    ):
        raise ValueError("Owned-output predecessor source or successful CPU evidence differs")
    initial = read(Path(record["preparation"]))
    if (
        initial["files_sha256"] != record["files_sha256"]
        or initial["helpers_sha256"] != record["helpers_sha256"]
        or any(sha256(Path(__file__).parent / name) != digest for name, digest in record["helpers_sha256"].items())
    ):
        raise ValueError("Owned-output predecessor preparation helpers changed")
    return previous, record


def prepare(receipt: Path, candidate: Path) -> dict:
    """Retain numerical source while adding explicit owned observations and bounded commands."""
    previous, record = require_predecessor(receipt)
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Owned-output validation needs a distinct unused candidate")
    helpers = Path(__file__).parent
    for name in HELPERS:
        if not (helpers / name).is_file():
            raise FileNotFoundError(f"Owned-output successor helper is missing: {name}")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, destination in COPIES.items():
        shutil.copy2(helpers / name, candidate / destination)
    path = candidate / "experiment/validate.py"
    text = replace_once(
        path.read_text(),
        "from .upstream_validation import ARMS, SYNTHETIC_GATES, require_upstream_capture_history, validation_commands",
        "from .upstream_validation import ARMS, SYNTHETIC_GATES, require_upstream_capture_history",
    )
    text = replace_once(
        text,
        "from .records import atomic_json, compare_continuation, sha256, utc_now",
        "from .owned_output_validation import (\n"
        "    require_owned_capture_history,\n"
        "    require_owned_synthetic_receipt,\n"
        "    validation_commands,\n"
        ")\n"
        "from .records import atomic_json, compare_continuation, sha256, utc_now",
    )
    text = replace_once(
        text,
        '            "real_data_upstream_product_parity": False,',
        '            "real_data_upstream_product_parity": False,\n'
        '            "gpu_owned_output_factors": False,\n'
        '            "real_data_owned_output_factor_parity": False,',
    )
    text = replace_once(
        text,
        '    receipt["evidence"]["gpu_receipt"] = str(gpu_receipt)\n    smoke_records = {}',
        '    receipt["evidence"]["gpu_receipt"] = str(gpu_receipt)\n'
        "    owned_command = validation_commands(sys.executable, TASK, OUTPUT_ROOT, args.work)[1]\n"
        '    execute(owned_command, args.work / "owned_output_gpu_checks.log")\n'
        '    owned_receipt = args.work / "owned_output_gpu_checks.json"\n'
        "    owned_checked = require_owned_synthetic_receipt(\n"
        "        json.loads(owned_receipt.read_text()),\n"
        '        receipt["source_sha256"],\n'
        '        json.loads(config_path(ARMS[2], 0, "0p5m").read_text()),\n'
        "    )\n"
        '    receipt["checks"]["gpu_owned_output_factors"] = True\n'
        '    receipt["evidence"]["owned_gpu_receipt"] = str(owned_receipt)\n'
        '    receipt["evidence"]["owned_gpu_receipt_sha256"] = sha256(owned_receipt)\n'
        '    receipt["evidence"]["owned_gpu_summary"] = owned_checked\n'
        "    atomic_json(args.receipt, receipt)\n"
        "    smoke_records = {}",
    )
    text = replace_once(
        text,
        '        smoke_records[name] = {\n            "out": str(full),',
        "        owned_observations = (\n"
        "            {\n"
        "                label: require_owned_capture_history(directory, arm)\n"
        '                for label, directory in (("continuous", full), ("resumed", interrupted))\n'
        "            }\n"
        "            if arm in ARMS[2:]\n"
        "            else None\n"
        "        )\n"
        "        smoke_records[name] = {\n"
        '            "owned_output_observations": owned_observations,\n'
        '            "out": str(full),',
    )
    text = replace_once(
        text,
        "        real_data_upstream_product_parity=True,",
        "        real_data_upstream_product_parity=True,\n" "        real_data_owned_output_factor_parity=True,",
    )
    path.write_text(text)
    # Preserve the generic captured-history tests; old ownership tests belong to
    # their issued source and are replaced by the distinct successor owner tests.
    path = candidate / "experiment/tests/test_upstream_validation.py"
    old = path.read_text()
    begin = old.index("def captured_rows(")
    stop = old.index("\n\n@pytest.fixture\ndef owned_processes(")
    path.write_text(
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
        '"""Preserved generic upstream captured-history regressions."""\n\n'
        "import json\n\nimport pytest\n"
        "from experiment.upstream_validation import ARMS, require_upstream_capture_history\n\n\n"
        + old[begin:stop].rstrip()
        + "\n"
    )
    cluster_path = candidate / "experiment/cluster.json"
    cluster = read(cluster_path)
    cluster["wave_name"] = "owned-output-full-model-validation-unreleased"
    cluster["environment"]["CUDA_VISIBLE_DEVICES"] = MASK
    write(cluster_path, cluster)
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | record["files_sha256"] if files.get(name) != record["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError(f"Owned-output validation exceeds its allowed files: {sorted(changed ^ CHANGES)}")
    initial = {
        "status": "owned_output_validation_prepared_gpu_denied",
        "candidate": str(candidate.resolve()),
        "predecessor_cpu_receipt": str(receipt.resolve()),
        "predecessor_cpu_receipt_sha256": sha256(receipt),
        "predecessor_files_sha256": record["files_sha256"],
        "files_sha256": files,
        "changed_files": sorted(changed),
        "helpers_sha256": {name: sha256(helpers / name) for name in HELPERS},
        "policy": (
            "Numerical block, kernel, training and config bytes preserved. Distinct validation identity only. "
            "No GPU binding/controller/queue or production authorization; ceiling campaign retains allocation priority."
        ),
    }
    write(candidate / "owned-output-validation-preparation.json", initial)
    return initial


def require_preparation(candidate: Path) -> dict:
    initial = read(candidate / "owned-output-validation-preparation.json")
    previous, record = require_predecessor(Path(initial["predecessor_cpu_receipt"]))
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | record["files_sha256"] if files.get(name) != record["files_sha256"].get(name)
    }
    helpers = Path(__file__).parent
    if (
        initial["status"] != "owned_output_validation_prepared_gpu_denied"
        or initial["candidate"] != str(candidate.resolve())
        or initial["predecessor_cpu_receipt_sha256"] != sha256(Path(initial["predecessor_cpu_receipt"]))
        or initial["predecessor_files_sha256"] != record["files_sha256"]
        or files != initial["files_sha256"]
        or changed != CHANGES
        or set(initial["helpers_sha256"]) != set(HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in initial["helpers_sha256"].items())
        or any(sha256(candidate / destination) != sha256(helpers / name) for name, destination in COPIES.items())
    ):
        raise ValueError("Owned-output validation preparation or source changed")
    if (candidate / "experiment/block_toner.py").read_bytes() != (previous / "experiment/block_toner.py").read_bytes():
        raise ValueError("Owned-output validation changed numerical block arithmetic")
    return initial


def require_cpu_archive(archive: Path, initial: dict, preparation_sha256: str) -> None:
    """Bind every archived source byte to the tested inventory and preparation."""
    expected = {f"candidate/{name}": digest for name, digest in initial["files_sha256"].items()}
    expected.update({f"tracking/{name}": digest for name, digest in initial["helpers_sha256"].items()})
    expected["candidate/owned-output-validation-preparation.json"] = preparation_sha256
    alias = "candidate/artifacts/external-sources"
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) + 1 or {member.name for member in members} != set(expected) | {alias}:
            raise ValueError("Owned-output CPU archive inventory differs from tested source")
        link = bundle.getmember(alias)
        if not link.issym() or link.linkname != "pinned-sources-materialized":
            raise ValueError("Owned-output CPU archive pinned-source alias differs")
        for name, digest in expected.items():
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError("Owned-output CPU archive source is not a regular file")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Owned-output CPU archive bytes differ from tested source")


def cpu(candidate: Path, output: Path) -> dict:
    """Preserve every CPU command outcome and source bytes without issuing GPU permission."""
    initial = require_preparation(candidate)
    if output.exists() or output.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Owned-output validation CPU evidence needs a new external directory")
    output.mkdir(parents=True, exist_ok=False)
    log = output / "candidate.log"
    preparation_path = candidate / "owned-output-validation-preparation.json"
    preparation_hash = sha256(preparation_path)
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(),
                cwd=candidate,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
            error = "CPU command exceeded 600 seconds; exit 124 is the timeout sentinel, not a process return code"
    archive = output / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in initial["files_sha256"]:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(
            preparation_path,
            arcname="candidate/owned-output-validation-preparation.json",
            recursive=False,
        )
        for name in HELPERS:
            bundle.add(Path(__file__).parent / name, arcname=f"tracking/{name}", recursive=False)
    try:
        if require_preparation(candidate) != initial or sha256(preparation_path) != preparation_hash:
            raise ValueError("Owned-output validation source changed during CPU testing or archival")
        require_cpu_archive(archive, initial, preparation_hash)
    except ValueError as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    record = {
        **initial,
        "status": (
            "owned_output_validation_cpu_verified_gpu_denied"
            if code == 0 and error is None
            else "owned_output_validation_cpu_failed"
        ),
        "preparation": str(preparation_path),
        "preparation_sha256": preparation_hash,
        "command": cpu_command(),
        "cwd": str(candidate),
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "cuda_visible_devices": "",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    if error is not None:
        record["error"] = error
    write(output / "receipt.json", record)
    if code or error is not None:
        raise RuntimeError(f"Owned-output validation CPU failure preserved at {output}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--phase", choices=("prepare", "cpu"), default="prepare")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare" and args.cpu_receipt is None:
        parser.error("Preparation requires --cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU verification requires --out")
    result = prepare(args.cpu_receipt, args.candidate) if args.phase == "prepare" else cpu(args.candidate, args.out)
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
