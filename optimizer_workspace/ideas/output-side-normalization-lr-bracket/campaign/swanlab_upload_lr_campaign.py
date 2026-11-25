# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Idempotent SwanLab uploader for lr-bracket-20261125 runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.setdefault(_var, "http://star-proxy.oa.com:3128")

import swanlab
from dotenv import dotenv_values

PVC = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC / "campaigns" / "lr-bracket-20261125"
DEFAULT_PROJECT = "output-side-normalization-confirmation"
DEFAULT_CRED = Path("/root/dl/runtime/script/.env")

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


def load_completed_run(run_dir: Path) -> dict | None:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    training = manifest.get("config", {}).get("training", {})
    steps = training.get("steps")
    if manifest.get("status") != "complete" or manifest.get("completed_steps") != steps:
        return None
    segments = sorted(run_dir.glob("metrics_from_*.jsonl"))
    if not segments:
        return None
    by_step: dict[int, dict] = {}
    for seg in segments:
        for line in seg.read_text().splitlines():
            if line.strip():
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
    return {
        "name": run_dir.name,
        "dir": run_dir,
        "manifest": manifest,
        "metrics": metrics,
        "provenance": provenance,
        "run_id": f"lrbracket-{run_dir.name}-{ident}",
    }


def main() -> None:
    creds = dotenv_values(DEFAULT_CRED)
    key = creds.get("SWANLAB_API_KEY") or os.environ.get("SWANLAB_API_KEY")
    if not key:
        print("No SWANLAB_API_KEY found; skipping SwanLab upload.")
        return

    receipt_path = CAMPAIGN_ROOT / "swanlab_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {"project": DEFAULT_PROJECT, "runs": {}}

    runs_root = CAMPAIGN_ROOT / "runs"
    if not runs_root.is_dir():
        return

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        item = load_completed_run(run_dir)
        if item is None:
            continue
        existing = receipt["runs"].get(item["name"])
        if existing and existing.get("status") == "verified" and existing.get("manifest_sha256") == item["provenance"]["manifest_sha256"]:
            print(f"SKIP SwanLab {item['name']} (already verified)")
            continue

        manifest = item["manifest"]
        cfg = manifest["config"]
        training = cfg["training"]
        model = cfg["model"]
        opt = cfg["optimizer"]
        arm = cfg["experiment"]["arm"]
        arm_label = cfg["experiment"].get("arm_label", arm)
        seed = training["seed"]
        lr = opt["lr"]

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
            project=DEFAULT_PROJECT,
            public=False,
            name=f"lrbracket-{item['name']}",
            description=f"LR-bracket campaign: model={model['size']}, 1x Chinchilla, arm={arm} ({arm_label}), seed={seed}, lr={lr:.6g}.",
            group="lr-bracket-20261125",
            job_type="historical-import",
            tags=["lr-bracket-20261125", model["size"], "1x", f"seed{seed}", arm_label, arm, f"lr={lr:.5g}", "A100"],
            id=item["run_id"],
            resume="allow",
            mode="online",
            on_online_error="raise",
            log_dir=str(CAMPAIGN_ROOT / "swanlab"),
            config={**cfg, "training_record": metadata, "historical_import": {**item["provenance"], "rows": len(item["metrics"])}},
            settings=settings,
        )
        swanlab.define_metric("tokens", section_name="Progress", hidden=True)
        swanlab.define_metric("train/*", x_axis="tokens", section_name="Training")
        swanlab.define_metric("eval/*", x_axis="tokens", section_name="Evaluation")
        swanlab.define_metric("runtime/*", x_axis="tokens", section_name="Recorded Training Runtime")

        paloma_file = CAMPAIGN_ROOT / "paloma_eval" / f"{item['name']}.json"
        paloma_extra = {}
        if paloma_file.is_file():
            pdata = json.loads(paloma_file.read_text())
            paloma_extra["eval/paloma_macro_mean"] = pdata["paloma_macro_mean"]
            for k, v in pdata.get("scores", {}).items():
                if k.startswith("paloma/"):
                    paloma_extra[f"eval/{k}"] = v

        total = len(item["metrics"])
        for step, row in enumerate(item["metrics"], start=1):
            if step == total and paloma_extra:
                row = {**row, **paloma_extra}
            run.log(row, step=step)
        run.finish()
        receipt["runs"][item["name"]] = {
            "id": run.id,
            "path": run.path,
            "url": run.url,
            "status": "verified",
            "uploaded_at": utc_now(),
            **item["provenance"],
        }
        write_json(receipt_path, receipt)
        print(f"Uploaded {item['name']} -> {run.url}")


if __name__ == "__main__":
    main()
