# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import completed short-LR histories with unique trial identities and full scalar verification."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import swanlab
from dotenv import dotenv_values
from run_fixed_age_tuning import binding, completed_trial
from swanlab_refresh_upload import (
    PROJECT,
    TRAINING_METADATA,
    History,
    metric_row,
    sha256,
    upload_history,
    utc_now,
    verify_history,
    write_receipt,
)


def histories(campaign_path: Path) -> list[History]:
    """Only fully completed, source-bound trials can become cloud histories."""
    binding(campaign_path)
    campaign = json.loads(campaign_path.read_text())
    result = []
    for trial in campaign["trials"]:
        directory = campaign_path.parent / "runs" / trial["name"]
        if not (directory / "run.json").exists():
            continue
        manifest = json.loads((directory / "run.json").read_text())
        if manifest["status"] != "complete":
            continue
        completed_trial(campaign_path, trial["name"])
        for key in TRAINING_METADATA:
            if key not in manifest:
                raise ValueError(f"{trial['name']}: original training metadata is missing: {key}")
        segments = sorted(directory.glob("metrics_from_*.jsonl"))
        if [path.name for path in segments] != ["metrics_from_000000.jsonl"]:
            raise ValueError("Short-LR import requires the original single non-resumed history")
        rows = [json.loads(line) for line in segments[0].read_text().splitlines()]
        metrics = [metric_row(row) for row in rows]
        if any(key not in metrics[-1] for key in ("train/loss", "eval/c4_loss", "eval/selection_loss")):
            raise ValueError("Short-LR final loss fields are missing")
        checkpoint = directory / f"checkpoints/step_{trial['steps']:06d}"
        provenance = {
            "manifest_sha256": sha256(directory / "run.json"),
            "metrics_sha256": sha256(segments[0]),
            "checkpoint_receipt_sha256": sha256(checkpoint / "complete.json"),
            "checkpoint_rank_bytes": [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)],
            "campaign_sha256": sha256(campaign_path),
            "execution_binding_sha256": sha256(campaign_path.parent / "execution-binding.json"),
            "trial_config_sha256": trial["config_sha256"],
        }
        identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:12]
        label = manifest["config"]["experiment"]["tuning"]["trial"]
        result.append(
            History(
                f"{campaign_path.parent.name}-{label}",
                trial["name"],
                manifest,
                metrics,
                provenance,
                f"{campaign_path.parent.name}-{trial['name']}-{identity}",
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    campaign_path = args.campaign.resolve()
    complete = histories(campaign_path)
    if not complete:
        print("No complete short-LR history; authentication and import not performed.", flush=True)
        return
    receipt_path = campaign_path.parent / "swanlab-upload.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"runs": {}}
    if (
        receipt.get("campaign_sha256", sha256(campaign_path)) != sha256(campaign_path)
        or receipt.get("execution_binding_sha256", sha256(campaign_path.parent / "execution-binding.json"))
        != sha256(campaign_path.parent / "execution-binding.json")
        or not set(receipt["runs"]) <= {trial["name"] for trial in json.loads(campaign_path.read_text())["trials"]}
    ):
        raise ValueError("Short-LR upload receipt differs from the bound campaign or trial inventory")
    key = dotenv_values(args.credential_file).get("SWANLAB_API_KEY")
    if not key:
        raise ValueError("SWANLAB_API_KEY is missing from the credential file")
    swanlab.login(api_key=key, save=False)
    api = swanlab.Api(api_key=key)
    project_path = f"{api.username}/{PROJECT}"
    if receipt.get("project", project_path) != project_path:
        raise ValueError("Tuning upload receipt belongs to another project")
    project = api.project(project_path)
    response = project.wrapper()
    if not response.ok or not project.project_id or project.visibility != "PRIVATE":
        raise RuntimeError("Cannot verify the existing SwanLab research project is private")
    receipt.update(
        project=project_path,
        visibility="PRIVATE",
        sdk=swanlab.__version__,
        campaign_sha256=sha256(campaign_path),
        execution_binding_sha256=sha256(campaign_path.parent / "execution-binding.json"),
    )
    write_receipt(receipt_path, receipt)
    for history in complete:
        entry = receipt["runs"].get(history.arm)
        if entry and any(entry.get(name) != value for name, value in history.provenance.items()):
            raise RuntimeError("Short-LR source identity changed after upload receipt creation")
        if entry is None:
            if args.verify_only:
                raise RuntimeError(f"{history.arm}: no original upload receipt")
            path = upload_history(history, key, campaign_path.parent, PROJECT, receipt, receipt_path)
        else:
            if entry["status"] == "initialized" and not args.verify_only:
                raise RuntimeError("Interrupted short-LR import; inspect with --verify-only rather than reimporting")
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
        print(f"VERIFIED {history.arm}: {len(history.metrics)} original updates", flush=True)
    expected = {trial["name"] for trial in json.loads(campaign_path.read_text())["trials"]}
    receipt["status"] = (
        "verified"
        if set(receipt["runs"]) == expected and all(run["status"] == "verified" for run in receipt["runs"].values())
        else "partial_verified"
    )
    receipt["verified_at"] = utc_now()
    write_receipt(receipt_path, receipt)
    print(f"Short-LR receipt: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
