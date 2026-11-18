# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the short-study LR choices at full budget without authorizing GPU work."""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import NODES, PYTHON, read, sha256
from prepare_fixed_age_tuning import require_source
from prepare_upstream_outer import replace_once
from upstream_production import cpu_command
from watch_wave5 import write
from wave7_release import ARMS, candidate_inventory

STUDY = "selected_lr_full_budget_seed0"
PARENT = "campaigns/lr-fixed-age-ceiling-visible8-recovery-20261117/campaign.json"
PREPARATION = "selected-lr-preparation.json"
OWNER = '''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep the selected-LR preparation GPU-denied until separate ownership is issued."""


def require_selected_lr_owner():
    raise RuntimeError("Selected-LR full-budget candidate has no GPU execution binding")
'''
CHANGES = {
    *(f"experiment/configs/{arm}_0p5m_seed0.json" for arm in ARMS),
    "experiment/train.py",
    "experiment/validate.py",
    "experiment/gpu_checks.py",
    "experiment/selected_lr_owner.py",
    "experiment/cluster.json",
}
TRAIN_IMPORT = (
    "from .records import atomic_json, checkpoint_identity, manifest, restore_checkpoint, save_checkpoint, utc_now"
)
VALIDATE_IMPORT = "from .reference import TASK, UPSTREAM"
GPU_IMPORT = "from .reference import TASK"
LAUNCH_CONDITION = (
    "Complete Wave10 with all-rank checkpoints, original worker/child absence, exact keeper restoration, "
    "five shared 536870912-token windows and full-resolution private SwanLab verification. Review its systems "
    "result. Independently validate all four selected full-budget configurations through continuous24 versus "
    "interrupted22/resumed24, compiled capture, fixed-age roots and fused update checks at unchanged tolerances. "
    "Issue distinct generation-bound validation/production ownership, then acquire original eight-GPU nodes. "
    "This preparation starts no process, queue or GPU job."
)


def evidence_paths(root: Path) -> list[Path]:
    parent = root / PARENT
    boundary = Path(read(parent)["parent_campaign"])
    initial = Path(read(boundary)["parent_campaign"])
    return [
        *(
            path.parent / name
            for path in (parent, boundary, initial)
            for name in ("campaign.json", "analysis.json", "swanlab-upload.json", "execution-binding.json")
        ),
        root / "run-preparation/wave10-launch-binding.json",
        root / "run-preparation/wave7-release-binding.json",
    ]


def require_execution(campaign_path: Path) -> dict:
    """Bind the complete archived helper closure for each contributing campaign."""
    record = read(campaign_path.parent / "execution-binding.json")
    helpers = Path(record["helpers"])
    expected = {
        "campaign.json": sha256(campaign_path),
        **{f"tracking/{name}": digest for name, digest in record["helpers_sha256"].items()},
        **{
            f"configs/{Path(trial['config']).name}": sha256(Path(trial["config"]))
            for trial in read(campaign_path)["trials"]
        },
    }
    if (
        record["campaign_sha256"] != expected["campaign.json"]
        or record["archive_sha256"] != sha256(Path(record["archive"]))
        or not {"analyze_tuning.py", "swanlab_fixed_age_tuning.py", "swanlab_refresh_upload.py"}
        <= record["helpers_sha256"].keys()
        or any(sha256(helpers / name) != digest for name, digest in record["helpers_sha256"].items())
    ):
        raise ValueError("Selected-LR contributing execution helpers or archive changed")
    with tarfile.open(record["archive"]) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise ValueError("Selected-LR contributing execution archive inventory differs")
        for name, digest in expected.items():
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError("Selected-LR contributing execution archive contains an alias")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Selected-LR contributing execution archive bytes differ")
    return record


def frozen_result(python: str, helpers: Path, script: str, argument: Path) -> dict:
    """Resolve imports only from the verified helper directory, regardless of the caller's cwd."""
    result = subprocess.run(
        [python, "-P", "-c", script, str(argument)],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    return json.loads(result.stdout)


def require_choices(root: Path) -> tuple[dict, dict]:
    """Rederive the completed seven-point selection and bind every contributing cloud history."""
    source = require_source(root, root / "run-preparation/wave7-release-binding.json")
    parent_path = root / PARENT
    parent = read(parent_path)
    execution = require_execution(parent_path)
    helpers = Path(execution["helpers"])
    if "analyze_lr_ceiling.py" not in execution["helpers_sha256"]:
        raise ValueError("Selected-LR parent lacks its frozen ceiling analyzer")
    script = (
        "import json,sys\nfrom pathlib import Path\nimport analyze_lr_ceiling as module\n"
        "assert Path(module.__file__).resolve()==Path.cwd()/'analyze_lr_ceiling.py'\n"
        "print(json.dumps(module.analyze(Path(sys.argv[1])),allow_nan=False))"
    )
    actual = frozen_result(PYTHON, helpers, script, parent_path)
    saved = read(parent_path.parent / "analysis.json")
    if (
        actual["status"] != "complete"
        or {key: value for key, value in actual.items() if key != "time"}
        != {key: value for key, value in saved.items() if key != "time"}
        or set(actual["completed_cumulative_family_lr_choices"]) != set(ARMS)
        or any(choice["upper_boundary_selected"] for choice in actual["completed_cumulative_family_lr_choices"].values())
    ):
        raise ValueError("Selected-LR choices require the completed, unchanged interior seven-point result")
    campaigns = [parent_path, Path(parent["parent_campaign"])]
    campaigns.append(Path(read(campaigns[-1])["parent_campaign"]))
    evidence = {}
    for campaign_path in campaigns:
        campaign = read(campaign_path)
        cloud_path = campaign_path.parent / "swanlab-upload.json"
        cloud = read(cloud_path)
        binding_path = campaign_path.parent / "execution-binding.json"
        contributing = require_execution(campaign_path)
        local_histories = frozen_result(
            str(root / ".venv-logging/bin/python"),
            Path(contributing["helpers"]),
            (
                "import json,sys\nfrom pathlib import Path\n"
                "from swanlab_fixed_age_tuning import histories\n"
                "from swanlab_refresh_upload import expected_columns\n"
                "items=histories(Path(sys.argv[1]))\n"
                "print(json.dumps({h.arm:{'point_counts':{k:len(v) for k,v in expected_columns(h).items()},"
                "'final_metrics':h.metrics[-1]} for h in items},allow_nan=False))"
            ),
            campaign_path,
        )
        if (
            campaign["root"] != str(root)
            or campaign["source_receipt_sha256"] != sha256(root / "run-preparation/wave7-release-binding.json")
            or campaign["data_receipt_sha256"] != source["data_receipt_sha256"]
            or cloud["status"] != "verified"
            or cloud["visibility"] != "PRIVATE"
            or cloud["project"] != "siyuc3141/output-side-normalization-marin"
            or cloud["campaign_sha256"] != sha256(campaign_path)
            or cloud["execution_binding_sha256"] != sha256(binding_path)
            or set(cloud["runs"]) != {trial["name"] for trial in campaign["trials"]}
        ):
            raise ValueError("Selected-LR search histories lack full-resolution verified provenance")
        for trial in campaign["trials"]:
            directory = campaign_path.parent / "runs" / trial["name"]
            checkpoint = directory / "checkpoints/step_000320"
            entry = cloud["runs"][trial["name"]]
            verification = entry["verification"]
            if (
                entry["status"] != "verified"
                or entry["campaign_sha256"] != sha256(campaign_path)
                or entry["execution_binding_sha256"] != sha256(binding_path)
                or entry["trial_config_sha256"] != sha256(Path(trial["config"]))
                or entry["manifest_sha256"] != sha256(directory / "run.json")
                or entry["metrics_sha256"] != sha256(directory / "metrics_from_000000.jsonl")
                or entry["checkpoint_receipt_sha256"] != sha256(checkpoint / "complete.json")
                or entry["checkpoint_rank_bytes"]
                != [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)]
                or verification["state"] != "FINISHED"
                or verification.get("configuration_verified") is not True
                or verification.get("token_axis_verified") is not True
                or verification.get("integer_values_exact") is not True
                or verification.get("csv_export_max_ulp") != 2
                or verification["point_counts"] != local_histories[trial["name"]]["point_counts"]
                or verification["final_metrics"] != local_histories[trial["name"]]["final_metrics"]
                or set(verification["serialization_errors"]) != set(verification["point_counts"])
                or any(
                    not math.isfinite(error[field]) or error[field] < 0 or (field == "max_ulp" and error[field] > 2)
                    for error in verification["serialization_errors"].values()
                    for field in ("max_absolute", "max_ulp")
                )
                or verification["serialization_errors"]["tokens"] != {"max_absolute": 0.0, "max_ulp": 0.0}
                or verification["all_scalar_values_exact"]
                != all(error["max_absolute"] == 0 for error in verification["serialization_errors"].values())
                or any(
                    verification["point_counts"].get(key) != count
                    for key, count in (
                        ("tokens", 320),
                        ("train/loss", 320),
                        ("eval/selection_loss", 9),
                        ("eval/c4_loss", 9),
                    )
                )
            ):
                raise ValueError("Selected-LR cloud verification differs from an original local search history")
        for path in (campaign_path, campaign_path.parent / "analysis.json", cloud_path, binding_path):
            evidence[str(path)] = sha256(path)
    wave10_path = root / "run-preparation/wave10-launch-binding.json"
    wave10 = read(wave10_path)
    if wave10["status"] != "frozen_validated_launch_release" or wave10["root"] != str(root):
        raise ValueError("Selected-LR preparation requires the separately bound Wave10 predecessor")
    evidence[str(wave10_path)] = sha256(wave10_path)
    evidence[str(root / "run-preparation/wave7-release-binding.json")] = sha256(
        root / "run-preparation/wave7-release-binding.json"
    )
    return source, {"choices": actual["completed_cumulative_family_lr_choices"], "files_sha256": evidence}


def full_config(reference: dict, choice: dict) -> dict:
    """Retain the full schedule and every optimizer setting except the selected matrix LR."""
    config = copy.deepcopy(reference)
    if (
        config["training"]["steps"] != 5120
        or config["training"]["tokens"] != 2684354560
        or config["training"]["batch_tokens"] != 524288
        or config["training"]["seed"] != 0
        or config["training"]["world_size"] != 8
        or config["experiment"]["refresh_install_age"] != 1
    ):
        raise ValueError("Selected-LR reference must retain the complete 0.5M-token batch budget")
    config["optimizer"]["lr"] = choice["matrix_lr"]
    config.pop("reference_validation_loss", None)
    return config


def guarded_source(previous: Path, name: str) -> str:
    text = (previous / name).read_text()
    anchor = {
        "train.py": TRAIN_IMPORT,
        "validate.py": VALIDATE_IMPORT,
        "gpu_checks.py": GPU_IMPORT,
    }[Path(name).name]
    text = replace_once(text, anchor, anchor + "\nfrom .selected_lr_owner import require_selected_lr_owner")
    call = "    require_selected_lr_owner()\n"
    if name.endswith("validate.py"):
        call = "    if args.gpu:\n        require_selected_lr_owner()\n"
    return replace_once(text, "    args = parser.parse_args()\n", "    args = parser.parse_args()\n" + call)


def prepare(root: Path, candidate: Path) -> dict:
    """Create an unused isolated candidate with selected full-budget configs and denied GPU entry points."""
    root, candidate = root.resolve(), candidate.resolve()
    source, selection = require_choices(root)
    previous = Path(source["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous.resolve()):
        raise FileExistsError("Selected-LR preparation requires a distinct unused candidate identity")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for arm in ARMS:
        name = f"experiment/configs/{arm}_0p5m_seed0.json"
        config = full_config(read(previous / name), selection["choices"][arm])
        (candidate / name).write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    for name in ("experiment/train.py", "experiment/validate.py", "experiment/gpu_checks.py"):
        (candidate / name).write_text(guarded_source(previous, name))
    (candidate / "experiment/selected_lr_owner.py").write_text(OWNER)
    cluster = read(previous / "experiment/cluster.json")
    cluster["wave_name"] = "wave11-selected-lr-full-budget-unreleased"
    (candidate / "experiment/cluster.json").write_text(json.dumps(cluster, indent=2, allow_nan=False) + "\n")
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | source["files_sha256"] if files.get(name) != source["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Selected-LR candidate exceeds its configuration/ownership-only scope")
    result = {
        "status": "selected_lr_full_budget_prepared_gpu_denied",
        "study": STUDY,
        "root": str(root),
        "candidate": str(candidate),
        "source": str(previous),
        "source_receipt": str(root / "run-preparation/wave7-release-binding.json"),
        "data_receipt_sha256": source["data_receipt_sha256"],
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "selection": selection,
        "preparation_helper_sha256": sha256(Path(__file__)),
        "arms": list(ARMS),
        "nodes": list(NODES),
        "seed": 0,
        "steps_per_arm": 5120,
        "tokens_per_arm": 2684354560,
        "total_training_tokens": 4 * 2684354560,
        "training_window_tokens": 536870912,
        "validation_boundary": 22,
        "validation_endpoint": 24,
        "planned_validation_tokens": 4 * 2 * 24 * 524288,
        "launch_condition": LAUNCH_CONDITION,
        "interpretation": (
            "Full-budget transfer test of short-schedule selections, not independently tuned full-budget optima. "
            "Reused seed0 is exploratory. C4 is reporting only. Fresh paired-seed confirmation follows only "
            "if surviving loss/runtime evidence warrants it; no tuning attempt is consumed by preparation."
        ),
    }
    write(candidate / PREPARATION, result)
    require_preparation(candidate)
    return result


def require_preparation(candidate: Path) -> dict:
    """Recheck immutable selection evidence and the exact permitted candidate transformation."""
    record = read(candidate / PREPARATION)
    root = Path(record["root"])
    source = require_source(root, Path(record["source_receipt"]))
    previous = Path(source["candidate"])
    expected_cluster = read(previous / "experiment/cluster.json")
    expected_cluster["wave_name"] = "wave11-selected-lr-full-budget-unreleased"
    parent_root = read(root / PARENT).get("root")
    parent_analysis = read((root / PARENT).parent / "analysis.json")
    expected_evidence = {str(path): sha256(path) for path in evidence_paths(root)}
    if (
        record["status"] != "selected_lr_full_budget_prepared_gpu_denied"
        or record["study"] != STUDY
        or record["candidate"] != str(candidate.resolve())
        or record["source"] != str(previous)
        or record["source_receipt"] != str(root / "run-preparation/wave7-release-binding.json")
        or record["data_receipt_sha256"] != source["data_receipt_sha256"]
        or record["files_sha256"] != candidate_inventory(candidate)
        or record["preparation_helper_sha256"] != sha256(Path(__file__))
        or set(record["changed_files"]) != CHANGES
        or parent_root != str(root)
        or parent_analysis["status"] != "complete"
        or record["selection"]["choices"] != parent_analysis["completed_cumulative_family_lr_choices"]
        or set(record["selection"]["choices"]) != set(ARMS)
        or record["selection"]["files_sha256"] != expected_evidence
        or tuple(record["arms"]) != ARMS
        or tuple(record["nodes"]) != NODES
        or record["seed"] != 0
        or record["steps_per_arm"] != 5120
        or record["tokens_per_arm"] != 2684354560
        or record["total_training_tokens"] != 4 * 2684354560
        or record["training_window_tokens"] != 536870912
        or record["validation_boundary"] != 22
        or record["validation_endpoint"] != 24
        or record["planned_validation_tokens"] != 4 * 2 * 24 * 524288
        or record["launch_condition"] != LAUNCH_CONDITION
        or (candidate / "experiment/selected_lr_owner.py").read_text() != OWNER
        or any(
            (candidate / name).read_text() != guarded_source(previous, name)
            for name in ("experiment/train.py", "experiment/validate.py", "experiment/gpu_checks.py")
        )
        or read(candidate / "experiment/cluster.json") != expected_cluster
        or any(
            read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
            != full_config(
                read(previous / f"experiment/configs/{arm}_0p5m_seed0.json"), record["selection"]["choices"][arm]
            )
            for arm in ARMS
        )
    ):
        raise ValueError("Selected-LR preparation, source or selection evidence changed")
    changed = {
        name
        for name in record["files_sha256"].keys() | source["files_sha256"]
        if record["files_sha256"].get(name) != source["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Selected-LR preparation no longer has its exact ownership/configuration scope")
    _, selection = require_choices(root)
    if record["selection"] != selection:
        raise ValueError("Selected-LR contributing search histories changed")
    return record


def cpu(candidate: Path, out: Path) -> dict:
    """Preserve candidate CPU outcomes and archive the tested source without granting execution."""
    record = require_preparation(candidate)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Selected-LR CPU evidence requires an unused external directory")
    out.mkdir(parents=True)
    command = cpu_command()
    error = None
    log = out / "candidate.log"
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command,
                cwd=candidate,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "CPU command exceeded 600 seconds; exit124 is the timeout sentinel"
    preparation_hash = sha256(candidate / PREPARATION)
    archive = out / "source.tar"
    names = record["files_sha256"] | {PREPARATION: preparation_hash}
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in names:
                bundle.add(candidate / name, arcname=name, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
            bundle.add(Path(__file__), arcname="tracking/prepare_selected_lr_full_budget.py", recursive=False)
        if require_preparation(candidate) != record or sha256(candidate / PREPARATION) != preparation_hash:
            raise ValueError("Selected-LR source changed during CPU verification")
        with tarfile.open(archive) as bundle:
            expected = names | {"tracking/prepare_selected_lr_full_budget.py": sha256(Path(__file__))}
            members = bundle.getmembers()
            alias = bundle.getmember("artifacts/external-sources")
            if (
                len(members) != len(expected) + 1
                or {member.name for member in members} != set(expected) | {"artifacts/external-sources"}
                or not alias.issym()
                or alias.linkname != "pinned-sources-materialized"
            ):
                raise ValueError("Selected-LR CPU archive inventory differs")
            for name, digest in expected.items():
                with bundle.extractfile(name) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        raise ValueError("Selected-LR CPU archive differs from tested bytes")
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": (
            "selected_lr_full_budget_cpu_verified_gpu_denied"
            if code == 0 and error is None
            else "selected_lr_cpu_failed"
        ),
        "candidate": str(candidate.resolve()),
        "preparation_sha256": preparation_hash,
        "files_sha256": record["files_sha256"],
        "command": command,
        "cwd": str(candidate.resolve()),
        "cuda_visible_devices": "",
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] == "selected_lr_cpu_failed":
        raise RuntimeError(f"Selected-LR CPU failure preserved at {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "check", "cpu"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU verification requires --out")
    result = (
        prepare(args.root, args.candidate)
        if args.phase == "prepare"
        else cpu(args.candidate, args.out) if args.phase == "cpu" else require_preparation(args.candidate)
    )
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}, allow_nan=False))


if __name__ == "__main__":
    main()
