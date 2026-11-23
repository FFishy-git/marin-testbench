# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Idempotent SwanLab uploader for the 34 remaining completed runs:
  - runs/wave_sym_fastlag (4 runs)
  - campaigns/related-baselines-20261121 (18 runs)
  - campaigns/related-selected-full-seed12-20261121 (12 runs)
Uploads into private project siyuc3141/output-side-normalization-marin and writes verified receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.setdefault(_var, "http://star-proxy.oa.com:3128")

import swanlab
from dotenv import dotenv_values

PVC = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
PROJECT = "output-side-normalization-marin"
CRED_FILE = Path("/root/dl/runtime/script/.env")

GROUPS = (
    (
        "wave_sym_fastlag",
        PVC / "runs" / "wave_sym_fastlag",
        PVC / "run-preparation" / "swanlab-upload-wave_sym_fastlag.json",
    ),
    (
        "related-baselines-20261121",
        PVC / "campaigns" / "related-baselines-20261121" / "runs",
        PVC / "campaigns" / "related-baselines-20261121" / "swanlab_receipt.json",
    ),
    (
        "related-selected-full-seed12-20261121",
        PVC / "campaigns" / "related-selected-full-seed12-20261121" / "runs",
        PVC / "campaigns" / "related-selected-full-seed12-20261121" / "swanlab_receipt.json",
    ),
)

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


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def metric_row(row: dict) -> dict:
    result = {"tokens": row["tokens"]}
    for key, value in row.items():
        if key in ("step", "tokens"):
            continue
        if isinstance(value, dict):
            for nested, scalar in value.items():
                if isinstance(scalar, (int, float)) and not isinstance(scalar, bool) and math.isfinite(scalar):
                    target = f"eval/{nested}" if key == "heldout_losses" else f"runtime/{key}/{nested}"
                    result[target] = scalar
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[LOSS_KEYS.get(key, f"runtime/{key}")] = value
    return result


def load_completed_run(group: str, run_dir: Path) -> dict | None:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    steps = manifest.get("config", {}).get("training", {}).get("steps")
    if manifest.get("status") != "complete" or manifest.get("completed_steps") != steps:
        return None
    segments = sorted(run_dir.glob("metrics_from_*.jsonl"))
    if not segments:
        return None
    by_step: dict[int, dict] = {}
    for seg in segments:
        for line in seg.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_step[int(row["step"])] = row
    if len(by_step) != steps or min(by_step) != 1 or max(by_step) != steps:
        return None
    ordered_rows = [by_step[s] for s in range(1, steps + 1)]
    metrics = [metric_row(r) for r in ordered_rows]
    provenance = {
        "manifest_sha256": sha256(manifest_path),
        "segments_sha256": {seg.name: sha256(seg) for seg in segments},
    }
    ident = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:12]
    safe_grp = group.replace("-", "_")
    run_id = f"{safe_grp[:16]}-{run_dir.name[:24]}-{ident}".replace("_", "-")
    return {
        "group": group,
        "name": run_dir.name,
        "dir": run_dir,
        "manifest": manifest,
        "metrics": metrics,
        "provenance": provenance,
        "run_id": run_id,
    }


def upload_one(item: dict, *, key: str, project: str, receipt: dict, receipt_path: Path) -> str:
    manifest = item["manifest"]
    cfg = manifest["config"]
    training = cfg["training"]
    opt = cfg["optimizer"]
    arm = cfg["experiment"]["arm"]
    seed = training["seed"]
    lr = opt["lr"]
    group = item["group"]
    run_key = item["name"]

    settings = swanlab.Settings(
        _env_file=None,
        api_key=key,
        interactive=False,
        terminal=swanlab.Settings.Terminal(proxy_type="none"),
        probe=swanlab.Settings.Probe(
            hardware=False, runtime=False, requirements=False, conda=False, git=False, swanlab=False, monitor=False
        ),
    )
    metadata = {k: manifest.get(k) for k in TRAINING_METADATA if k in manifest}
    run = swanlab.init(
        project=project,
        public=False,
        name=f"{group}-{item['name']}",
        description=f"{group}: arm={arm}, seed={seed}, steps={training['steps']}, lr={lr:.6g}.",
        group=group,
        job_type="historical-import",
        tags=[group, f"seed{seed}", arm, f"lr={lr:.5g}", "A100"],
        id=item["run_id"],
        resume="allow",
        mode="online",
        on_online_error="raise",
        log_dir=str(PVC / "swanlab"),
        config={
            **cfg,
            "training_record": metadata,
            "historical_import": {**item["provenance"], "rows": len(item["metrics"])},
        },
        settings=settings,
    )
    entry = {"id": run.id, "path": run.path, "url": run.url, "status": "initialized", **item["provenance"]}
    receipt["runs"][run_key] = entry
    write_json(receipt_path, receipt)

    swanlab.define_metric("tokens", section_name="Progress", hidden=True)
    swanlab.define_metric("train/*", x_axis="tokens", section_name="Training")
    swanlab.define_metric("eval/*", x_axis="tokens", section_name="Evaluation")
    swanlab.define_metric("runtime/*", x_axis="tokens", section_name="Recorded Training Runtime")

    try:
        for step, row in enumerate(item["metrics"], start=1):
            run.log(row, step=step)
        run.finish()
    except BaseException:
        run.finish(state="crashed", error="Historical import failed; local JSONL remains authoritative.")
        raise

    entry["status"] = "uploaded"
    entry["uploaded_at"] = utc_now()
    write_json(receipt_path, receipt)
    return entry["path"]


def verify_finished(key: str, path: str, expected_steps: int) -> dict:
    for attempt in range(4):
        try:
            api = swanlab.Api(api_key=key)
            remote = api.run(path.lstrip("/"))
            resp = remote.wrapper()
            if not resp.ok:
                raise RuntimeError(resp.errmsg)
            if remote.state != "FINISHED":
                raise RuntimeError(f"Unexpected state {remote.state}")
            return {"verified_at": utc_now(), "state": remote.state, "url": remote.url, "steps": expected_steps}
        except RuntimeError:
            if attempt == 3:
                raise
            time.sleep(5)
    raise RuntimeError("Unreachable")


def main() -> None:
    key = dotenv_values(CRED_FILE).get("SWANLAB_API_KEY")
    if not key:
        raise ValueError(f"SWANLAB_API_KEY missing from {CRED_FILE}")

    swanlab.login(api_key=key, save=False)
    api = swanlab.Api(api_key=key)
    project_path = f"{api.username}/{PROJECT}"
    proj = api.project(project_path)
    resp = proj.wrapper()
    if not resp.ok or not proj.project_id:
        raise RuntimeError(f"Cannot access SwanLab project {project_path}")
    print(f"SwanLab Project: {proj.url} (visibility={proj.visibility})", flush=True)

    total_up = 0
    total_skip = 0
    for group, runs_dir, receipt_path in GROUPS:
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {"runs": {}}
        receipt.update(project=project_path, project_url=proj.url, visibility=proj.visibility, sdk=swanlab.__version__)
        write_json(receipt_path, receipt)

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            item = load_completed_run(group, run_dir)
            if item is None:
                continue
            run_key = item["name"]
            prev = receipt["runs"].get(run_key)
            if prev and prev.get("status") == "verified" and prev.get("manifest_sha256") == item["provenance"]["manifest_sha256"]:
                total_skip += 1
                print(f"SKIP {group}/{run_key} (already verified)", flush=True)
                continue
            print(f"UPLOADING {group}/{run_key} ({len(item['metrics'])} steps)...", flush=True)
            path = upload_one(item, key=key, project=PROJECT, receipt=receipt, receipt_path=receipt_path)
            ver = verify_finished(key, path, len(item["metrics"]))
            receipt["runs"][run_key].update(status="verified", verification=ver)
            write_json(receipt_path, receipt)
            total_up += 1
            print(f"VERIFIED {group}/{run_key}: {ver['url']}", flush=True)

        receipt["status"] = "verified"
        receipt["updated_at"] = utc_now()
        write_json(receipt_path, receipt)
        print(f"GROUP DONE {group}: verified={sum(1 for r in receipt['runs'].values() if r.get('status')=='verified')}/{len(receipt['runs'])} -> {receipt_path}", flush=True)

    print(f"ALL DONE: uploaded={total_up}, skipped={total_skip}", flush=True)


if __name__ == "__main__":
    main()
