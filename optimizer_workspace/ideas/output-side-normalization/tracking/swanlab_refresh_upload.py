# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import completed wave JSONL histories without changing the trainer."""

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import swanlab
from dotenv import dotenv_values
from refresh_history import scalar_history_row

WAVES = {
    "wave0": ("muon", "muont", "outer_left_down", "down_hybrid"),
    "wave1": ("inner_outer_all", "left_all", "inner_outer_all_diag", "null_all"),
    "wave2": ("inner_outer_hidden", "inner_outer_mlpdiag", "inner_outer_leftdiag", "outer_left_all"),
    "wave3": ("inner_outer_mlpdiag_seed1", "muont_seed1", "null_mlpdiag", "inner_outer_all_lag"),
    "wave4": ("inner_outer_mlpdiag_lag", "inner_outer_mlpdiag_1m", "muont_1m", "inner_outer_mlpdiag_seed2"),
    "wave4-baseline": ("muont_seed2",),
    "wave5": ("inner_outer_mlpdiag", "inner_outer_mlpdiag_fused", "inner_outer_leftdiag_fused", "muont"),
}
PROJECT = "output-side-normalization-marin"
PVC = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
TRAINING_METADATA = (
    "start_time",
    "end_time",
    "status",
    "completed_steps",
    "elapsed_seconds",
    "source_revision",
    "experiment_git_revision",
    "source_sha256",
    "data_receipt_sha256",
    "python",
    "torch",
    "cuda",
    "driver",
    "gpu",
    "world_size",
)
LOSS_KEYS = {
    "train_loss": "train/loss",
    "c4_loss": "eval/c4_loss",
    "selection_loss": "eval/selection_loss",
}
CSV_EXPORT_MAX_ULP = 2
BATCH_TOKENS = {"0p5m": 524288, "1m": 1048576, "2m": 2097152, "4m": 4194304}


def run_name(arm: str, seed: int, batch: str = "0p5m") -> str:
    batch_suffix = "" if batch == "0p5m" else f"_{batch}"
    seed_suffix = "" if seed == 0 else f"_seed{seed}"
    return f"{arm}{batch_suffix}{seed_suffix}"


@dataclass
class History:
    wave: str
    arm: str
    manifest: dict
    metrics: list[dict]
    provenance: dict
    run_id: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_receipt(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metric_row(row: dict) -> dict:
    """Flatten training metrics while preserving sparse evaluation records."""
    result = {"tokens": row["tokens"]}
    for key, value in scalar_history_row(row).items():
        if key in ("step", "tokens"):
            continue
        if isinstance(value, dict):
            for nested, scalar in value.items():
                target = f"eval/{nested}" if key == "heldout_losses" else f"runtime/{key}/{nested}"
                if target in result:
                    raise ValueError(f"Duplicate metric {target}")
                result[target] = scalar
        else:
            result[LOSS_KEYS.get(key, f"runtime/{key}")] = value
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid scalar at update {row['step']}: {key}")
    return result


def load_history(root: Path, wave: str, arm: str) -> History:
    """Validate complete histories and checkpoint receipts before authentication."""
    directory = root / "runs" / wave / arm
    manifest_path = directory / "run.json"
    metrics_path = directory / "metrics_from_000000.jsonl"
    manifest = json.loads(manifest_path.read_text())
    training = manifest["config"]["training"]
    steps = training["steps"]
    batch_tokens = training["batch_tokens"]
    if manifest["status"] != "complete" or manifest["completed_steps"] != steps:
        raise ValueError(f"{arm}: training has not completed")
    # A wave may hold a second seed of an arm, whose directory is ARM_seedN rather than
    # ARM, so validate the directory against both the arm and the seed it claims.
    seed = training["seed"]
    named = manifest["config"]["experiment"]["arm"]
    batch = next((name for name, tokens in BATCH_TOKENS.items() if tokens == batch_tokens), None)
    if batch is None:
        raise ValueError(f"{arm}: unrecognized batch token count {batch_tokens}")
    if run_name(named, seed, batch) != arm:
        raise ValueError(f"{arm}: directory does not match manifest arm {named} seed {seed}")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if [row["step"] for row in rows] != list(range(1, steps + 1)):
        raise ValueError(f"{arm}: missing, duplicate, or out-of-order training updates")
    if any(row["tokens"] != row["step"] * batch_tokens for row in rows):
        raise ValueError(f"{arm}: token axis mismatch")
    if rows[-1]["tokens"] != training["tokens"]:
        raise ValueError(f"{arm}: incomplete token budget")
    checkpoint = directory / "checkpoints" / f"step_{steps:06d}"
    complete_path = checkpoint / "complete.json"
    complete = json.loads(complete_path.read_text())
    world = training["world_size"]
    if complete["step"] != steps or complete["world"] != world:
        raise ValueError(f"{arm}: final checkpoint receipt mismatch")
    rank_sizes = [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(world)]
    if not all(size > 0 for size in rank_sizes):
        raise ValueError(f"{arm}: empty final checkpoint rank file")
    metrics = [metric_row(row) for row in rows]
    for key in LOSS_KEYS.values():
        if key not in metrics[-1]:
            raise ValueError(f"{arm}: final metric missing: {key}")
    provenance = {
        "metrics_sha256": sha256(metrics_path),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint_receipt_sha256": sha256(complete_path),
        "checkpoint_rank_bytes": rank_sizes,
    }
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:12]
    return History(wave, arm, manifest, metrics, provenance, f"{wave}-{arm}-{identity}")


def expected_columns(history: History) -> dict:
    columns: dict[str, list[tuple[int, float]]] = {}
    for step, row in enumerate(history.metrics, start=1):
        for key, value in row.items():
            columns.setdefault(key, []).append((step, value))
    return columns


def verify_config_value(cloud: object, local: object, path: str) -> None:
    """Check metadata identities exactly, allowing only float export rounding."""
    if isinstance(local, dict):
        if not isinstance(cloud, dict) or cloud.keys() != local.keys():
            raise RuntimeError(f"{path}: cloud configuration keys differ")
        for key, value in local.items():
            verify_config_value(cloud[key], value, f"{path}/{key}")
        return
    if isinstance(local, list):
        if not isinstance(cloud, list) or len(cloud) != len(local):
            raise RuntimeError(f"{path}: cloud configuration list differs")
        for index, (actual, expected) in enumerate(zip(cloud, local, strict=True)):
            verify_config_value(actual, expected, f"{path}/{index}")
        return
    if isinstance(local, float):
        if not isinstance(cloud, (int, float)) or not math.isfinite(cloud):
            raise RuntimeError(f"{path}: invalid cloud configuration float")
        if abs(cloud - local) <= CSV_EXPORT_MAX_ULP * math.ulp(local):
            return
    elif cloud == local:
        return
    raise RuntimeError(f"{path}: cloud configuration value differs")


def verify_history(api: swanlab.Api, path: str, history: History) -> dict:
    """Read full-resolution cloud exports and compare every scalar and update."""
    remote = api.run(path.lstrip("/"))
    response = remote.wrapper()
    if not response.ok:
        raise RuntimeError(response.errmsg)
    if remote.state != "FINISHED":
        raise RuntimeError(f"{history.arm}: unexpected cloud state {remote.state}")
    config = remote.profile["config"]
    for key, value in history.manifest["config"].items():
        verify_config_value(config[key]["value"], value, f"{history.arm}/config/{key}")
    training_record = {key: history.manifest[key] for key in TRAINING_METADATA}
    verify_config_value(config["training_record"]["value"], training_record, f"{history.arm}/training_record")
    expected = expected_columns(history)
    exported = remote.metrics(keys=list(expected), all=True, ignore_timestamp=True)
    actual = {column["key"]: column["metrics"] for column in exported["list"]}
    counts = {}
    errors = {}
    for key, local in expected.items():
        cloud = actual.get(key, [])
        if len(cloud) != len(local):
            raise RuntimeError(f"{history.arm}/{key}: cloud points {len(cloud)}, expected {len(local)}")
        max_absolute = 0.0
        max_ulp = 0.0
        for point, (step, value) in zip(cloud, local, strict=True):
            error = abs(point["value"] - value)
            ulp_error = error / math.ulp(float(value))
            # House's CSV export can change the float64 value by two ULPs.
            valid = error == 0 if isinstance(value, int) else ulp_error <= CSV_EXPORT_MAX_ULP
            if point["step"] != step or not math.isfinite(point["value"]) or not valid:
                raise RuntimeError(f"{history.arm}/{key}: cloud value mismatch at update {step}")
            max_absolute = max(max_absolute, error)
            max_ulp = max(max_ulp, ulp_error)
        counts[key] = len(cloud)
        errors[key] = {"max_absolute": max_absolute, "max_ulp": max_ulp}
    axis_export = remote.metrics(keys=["train/loss", "eval/c4_loss"], all=True, ignore_timestamp=True, x_axis="tokens")
    axis_columns = {column["key"]: column["metrics"] for column in axis_export["list"]}
    if set(axis_columns) != {"train/loss", "eval/c4_loss"}:
        raise RuntimeError(f"{history.arm}: missing token-axis export columns")
    for key, points in axis_columns.items():
        if sum(math.isfinite(point["value"]) for point in points) != len(expected[key]):
            raise RuntimeError(f"{history.arm}/{key}: incomplete token-axis export")
    for column in axis_export["list"]:
        for point in column["metrics"]:
            if point["index"] != history.metrics[point["step"] - 1]["tokens"]:
                raise RuntimeError(f"{history.arm}: custom token axis mismatch at update {point['step']}")
    return {
        "verified_at": utc_now(),
        "state": remote.state,
        "url": remote.url,
        "point_counts": counts,
        "final_metrics": history.metrics[-1],
        "all_scalar_values_exact": all(error["max_absolute"] == 0 for error in errors.values()),
        "integer_values_exact": True,
        "csv_export_max_ulp": CSV_EXPORT_MAX_ULP,
        "serialization_errors": errors,
        "token_axis_verified": True,
        "configuration_verified": True,
    }


def upload_history(history: History, key: str, root: Path, project: str, receipt: dict, receipt_path: Path) -> str:
    settings = swanlab.Settings(
        _env_file=None,
        api_key=key,
        interactive=False,
        terminal=swanlab.Settings.Terminal(proxy_type="none"),
        probe=swanlab.Settings.Probe(
            hardware=False, runtime=False, requirements=False, conda=False, git=False, swanlab=False, monitor=False
        ),
    )
    metadata = {key: history.manifest[key] for key in TRAINING_METADATA}
    training = history.manifest["config"]["training"]
    seed = training["seed"]
    batch = next(name for name, tokens in BATCH_TOKENS.items() if tokens == training["batch_tokens"])
    arm = history.manifest["config"]["experiment"]["arm"]
    run = swanlab.init(
        project=project,
        public=False,
        name=f"{history.wave}-{arm}-{batch}-seed{seed}",
        description=(
            "Historical JSONL import of completed Toner/Marin A100 training. "
            "Original timestamps and hardware are in training_record. Cloud timestamps refer to this import. "
            f"Seed {seed}; exploratory comparison, no independently tuned confirmation claim."
        ),
        group=history.wave,
        job_type="historical-import",
        tags=[f"seed{seed}", batch, "A100", "historical-import", arm],
        id=history.run_id,
        resume="allow",
        mode="online",
        on_online_error="raise",
        log_dir=str(root / "swanlab"),
        config={
            **history.manifest["config"],
            "training_record": metadata,
            "historical_import": {
                **history.provenance,
                "format": "metrics_from_000000.jsonl",
                "rows": len(history.metrics),
                "telemetry": (
                    "Original scalars unchanged; host-refresh lists projected to count/age/source-step summaries; "
                    "raw events retained in hash-bound local JSONL; uploader monitoring disabled"
                ),
            },
        },
        settings=settings,
    )
    entry = {"id": run.id, "path": run.path, "url": run.url, "status": "initialized", **history.provenance}
    receipt["runs"][history.arm] = entry
    write_receipt(receipt_path, receipt)
    swanlab.define_metric("tokens", section_name="Progress", hidden=True)
    swanlab.define_metric("train/*", x_axis="tokens", section_name="Training")
    swanlab.define_metric("eval/*", x_axis="tokens", section_name="Evaluation")
    swanlab.define_metric("runtime/*", x_axis="tokens", section_name="Recorded Training Runtime")
    try:
        for step, row in enumerate(history.metrics, start=1):
            run.log(row, step=step)
        run.finish()
    except BaseException:
        run.finish(state="crashed", error="Historical import failed; local JSONL remains authoritative.")
        raise
    entry["status"] = "uploaded"
    entry["uploaded_at"] = utc_now()
    write_receipt(receipt_path, receipt)
    return entry["path"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PVC)
    parser.add_argument("--credential-file", type=Path, default=Path("/root/dl/runtime/script/.env"))
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--wave", choices=WAVES, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    histories = [load_history(args.root, args.wave, arm) for arm in WAVES[args.wave]]
    receipt_path = args.root / f"run-preparation/swanlab-upload-{args.wave}.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"runs": {}}
    key = dotenv_values(args.credential_file).get("SWANLAB_API_KEY")
    if not key:
        raise ValueError("SWANLAB_API_KEY is missing from the credential file")
    swanlab.login(api_key=key, save=False)
    api = swanlab.Api(api_key=key)
    project_path = f"{api.username}/{args.project}"
    if receipt.get("project", project_path) != project_path:
        raise RuntimeError("Upload receipt belongs to a different SwanLab project")
    existing = list(api.projects(api.username, search=args.project, all=True))
    if not any(project.name == args.project for project in existing):
        if args.verify_only:
            raise RuntimeError("Project does not exist")
        created = api.create_project(name=args.project, visibility="PRIVATE", description="Output-side normalization")
        if created is None:
            raise RuntimeError("Could not create private SwanLab project")
    project = api.project(project_path)
    response = project.wrapper()
    if not response.ok or not project.project_id or project.visibility != "PRIVATE":
        raise RuntimeError("Cannot verify that the SwanLab project is private")
    receipt.update(project=project_path, project_url=project.url, visibility=project.visibility, sdk=swanlab.__version__)
    write_receipt(receipt_path, receipt)
    for history in histories:
        entry = receipt["runs"].get(history.arm)
        if entry and any(entry[key] != value for key, value in history.provenance.items()):
            raise RuntimeError(f"{history.arm}: source history changed after receipt creation")
        if entry is None:
            if args.verify_only:
                raise RuntimeError(f"{history.arm}: no upload receipt")
            path = upload_history(history, key, args.root, args.project, receipt, receipt_path)
        else:
            if entry["status"] == "initialized" and not args.verify_only:
                raise RuntimeError(f"{history.arm}: interrupted import; inspect cloud data using --verify-only")
            path = entry["path"]
        for attempt in range(3):
            try:
                verified = verify_history(swanlab.Api(api_key=key), path, history)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(5)
        receipt["runs"][history.arm].update(status="verified", verification=verified)
        write_receipt(receipt_path, receipt)
        print(f"VERIFIED {history.arm}: {len(history.metrics)} updates; {verified['url']}", flush=True)
    receipt["status"] = "verified"
    receipt["verified_at"] = utc_now()
    write_receipt(receipt_path, receipt)
    print(f"Private project: {project.url}\nReceipt: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
