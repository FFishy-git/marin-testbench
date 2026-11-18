# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a distinct upstream validator without releasing execution or changing prior evidence."""

import argparse
import json
import shutil
from pathlib import Path

from launch_wave5 import read, sha256
from prepare_upstream_full_model import cpu_test_commands
from prepare_upstream_outer import replace_once
from watch_wave5 import write
from wave7_release import candidate_inventory

CHANGES = {
    "experiment/validate.py",
    "experiment/launch.py",
    "experiment/execution_owner.py",
    "experiment/gpu_checks.py",
    "experiment/upstream_validation.py",
    "experiment/cluster.json",
    "experiment/tests/test_upstream_release.py",
    "experiment/tests/test_upstream_validation.py",
}


def require_cpu_source(receipt_path: Path) -> tuple[Path, dict]:
    """Require current actual-path source and recorded successful CPU evidence."""
    record = read(receipt_path)
    candidate = Path(record["candidate"])
    if (
        record["status"] != "full_model_observer_cpu_verified_gpu_execution_denied"
        or record["cuda_visible_devices"] != ""
        or candidate_inventory(candidate) != record["files_sha256"]
        or sha256(Path(record["archive"])) != record["archive_sha256"]
        or sha256(Path(record["initial_preparation"])) != record["initial_preparation_sha256"]
        or len(record["commands"]) != 2
        or [(item["cwd"], item["command"]) for item in record["commands"]]
        != [(str(cwd), command) for _, cwd, command in cpu_test_commands(candidate, Path(record["helpers"]))]
        or any(
            type(item["exit_code"]) is not int
            or item["exit_code"] != 0
            or sha256(Path(item["log"])) != item["log_sha256"]
            for item in record["commands"]
        )
        or any(sha256(Path(record["helpers"]) / name) != digest for name, digest in record["helpers_sha256"].items())
    ):
        raise ValueError("Actual-path CPU predecessor source or recorded evidence differs")
    return candidate, record


def prepare(receipt_path: Path, candidate: Path) -> dict:
    previous, record = require_cpu_source(receipt_path)
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Validator preparation needs a distinct new candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helpers = Path(__file__).parent
    for source, destination in (
        ("upstream_validation.py", "experiment/upstream_validation.py"),
        ("upstream_execution_owner.py", "experiment/execution_owner.py"),
        ("test_upstream_validation.py", "experiment/tests/test_upstream_validation.py"),
    ):
        shutil.copy2(helpers / source, candidate / destination)
    path = candidate / "experiment/validate.py"
    text = replace_once(
        path.read_text(),
        "from .reference import TASK, UPSTREAM",
        "from .reference import TASK, UPSTREAM\n"
        "from .upstream_validation import ARMS, SYNTHETIC_GATES, require_upstream_capture_history, validation_commands",
    )
    text = replace_once(
        text,
        '    if WAVE != (\n        ("muont", 0, "0p5m"),\n        ("outer_leftdiag", 0, "0p5m"),\n'
        '        ("outer_leftdiag_fused", 0, "0p5m"),\n'
        '        ("outer_leftdiag_combined_fused", 0, "0p5m"),\n    ):\n'
        '        raise ValueError("This candidate validation covers only the four Wave-8 configurations")',
        '    if WAVE != tuple((arm, 0, "0p5m") for arm in ARMS):\n'
        '        raise ValueError("Upstream validation requires the exact four full-budget configurations")',
    )
    text = text.replace("real_data_combined_outer_update_parity", "real_data_upstream_product_parity")
    text = replace_once(
        text,
        '    for name in (\n        "gpu_collector",\n        "distributed_statistics",\n        "fa2",\n'
        '        "gpu_down_updates",\n        "cached_root_continuation",\n        "microbatch_partition",\n'
        "        *BLOCK_CHECKS,\n    ):",
        "    for name in SYNTHETIC_GATES:",
    )
    text = replace_once(
        text,
        "    checked = json.loads(gpu_receipt.read_text())\n    for name in SYNTHETIC_GATES:",
        "    checked = json.loads(gpu_receipt.read_text())\n"
        '    if checked["world"] != 2 or checked["source_sha256"] != receipt["source_sha256"]:\n'
        '        raise ValueError("Synthetic GPU source or rank coverage differs")\n'
        "    for name in SYNTHETIC_GATES:",
    )
    text = text.replace(
        'if arm in ("outer_leftdiag_fused", "outer_leftdiag_combined_fused"):',
        'if arm.endswith("_fused"):',
    )
    text = replace_once(
        text,
        '            "experiment/tests",\n            "-q",',
        '            "experiment/tests",\n            "-q",\n            "-m",\n'
        '            "not slow and not integration and not data_integration and not cluster "\n'
        '            "and not requires_cluster and not docker and not manual",',
    )
    text = replace_once(
        text,
        '                if arm.endswith("_fused"):\n                    require_outer_capture_history(directory)\n'
        '                if arm == "outer_leftdiag_combined_fused":\n'
        '                    if not all(row.get("combined_outer_update_parity_passed") is True for row in rows):\n'
        '                        raise ValueError("Actual combined arithmetic parity did not cover every update")',
        '                if arm == "outer_leftdiag_fused":\n'
        "                    require_outer_capture_history(directory)\n"
        "                else:\n                    require_upstream_capture_history(directory, arm)",
    )
    text = replace_once(
        text,
        '    receipt["evidence"]["real_data_smokes"] = smoke_records',
        '    if [item["command"] for item in receipt["evidence"]["gpu_commands"]] != validation_commands(\n'
        "        sys.executable, TASK, OUTPUT_ROOT, args.work\n"
        "    ):\n"
        '        raise ValueError("Executed validation commands differ from the canonical plan")\n'
        '    receipt["evidence"]["real_data_smokes"] = smoke_records',
    )
    start = text.index('            "combined_gate_scope": {')
    stop = text.index("\n            },", start) + len("\n            },")
    text = (
        text[:start]
        + (
            '            "upstream_gate_scope": {\n'
            '                "arms": list(ARMS[2:]),\n'
            '                "exact": ["momentum", "ns_source", "ns_output", "null_delta"],\n'
            '                "output_equation_tolerance": {"rtol": 1e-5, "atol": 1e-6},\n'
            '                "changing_scalar_gate": "gpu_upstream_scalar_step",\n'
            '                "coverage": "Every owned shape, every update, both actual trajectories",\n'
            "            },"
        )
        + text[stop:]
    )
    path.write_text(text)
    path = candidate / "experiment/launch.py"
    path.write_text(
        replace_once(
            path.read_text(),
            '    "gpu_combined_outer_updates",',
            '    "gpu_upstream_product_updates",\n    "gpu_upstream_scalar_step",',
        )
    )
    path = candidate / "experiment/gpu_checks.py"
    text = replace_once(
        path.read_text(),
        "from .execution_owner import require_controller",
        "from .execution_owner import require_synthetic_owner",
    )
    path.write_text(replace_once(text, "    require_controller()", "    require_synthetic_owner()"))
    path = candidate / "experiment/tests/test_upstream_release.py"
    path.write_text(
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
        '"""A validation candidate cannot inherit a production launch release."""\n\n'
        "from types import SimpleNamespace\n\nimport pytest\n\n"
        "from experiment import execution_owner\n\n\n"
        "def test_unreleased_candidate_refuses_gpu_validation(tmp_path, monkeypatch):\n"
        '    monkeypatch.setattr(execution_owner, "OUTPUT_ROOT", tmp_path)\n'
        "    with pytest.raises(FileNotFoundError):\n        execution_owner.require_gpu_owner()\n\n\n"
        "def test_unreleased_candidate_cannot_reuse_wave8_production_authorization():\n"
        '    with pytest.raises(RuntimeError, match="production release has not been issued"):\n'
        '        execution_owner.require_training_owner(SimpleNamespace(launch_release="wave8"), {})\n'
    )
    cluster_path = candidate / "experiment/cluster.json"
    cluster = read(cluster_path)
    cluster["wave_name"] = "upstream-full-model-corrected-validation-unreleased"
    write(cluster_path, cluster)
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | record["files_sha256"] if files.get(name) != record["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError(f"Validator preparation changed outside its exact scope: {sorted(changed ^ CHANGES)}")
    result = {
        "status": "validator_prepared_not_frozen_no_execution_release",
        "candidate": str(candidate.resolve()),
        "predecessor_cpu_receipt": str(receipt_path.resolve()),
        "predecessor_cpu_receipt_sha256": sha256(receipt_path),
        "predecessor_files_sha256": record["files_sha256"],
        "files_sha256": files,
        "changed_files": sorted(changed),
        "helpers_sha256": {
            name: sha256(helpers / name)
            for name in (
                "prepare_upstream_validation.py",
                "upstream_validation.py",
                "upstream_execution_owner.py",
                "test_upstream_validation.py",
            )
        },
        "policy": "No binding/controller/queue or CUDA approval; preserve predecessor and synthetic probe.",
    }
    write(candidate / "validation-preparation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.cpu_receipt, args.candidate)
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
