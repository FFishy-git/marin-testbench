# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist related-baseline observations without launching or changing workloads."""

import argparse
import csv
import fcntl
import io
import json
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from owned_output_handoff import same_generation
from related_campaign import campaign_path, sha256, status

HELPERS = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/torch-base/bin/python"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PROBE = """
import csv, io, json, subprocess, sys
sys.path.insert(0, sys.argv[1])
from owned_output_handoff import observed
pids = json.loads(sys.argv[2])
processes = {str(pid): observed(pid) for pid in pids}
gpus = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
     "--format=csv,noheader,nounits"], text=True, timeout=15)
compute = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
     "--format=csv,noheader"], text=True, timeout=15)
for row in csv.reader(io.StringIO(compute)):
    if row:
        pid = int(row[0].strip())
        processes[str(pid)] = observed(pid)
print(json.dumps({"processes": processes, "gpus": gpus, "compute": compute}))
"""


def read(path):
    return json.loads(path.read_text())


def write(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def probe(node, pids):
    command = [PYTHON, "-c", PROBE, str(HELPERS), json.dumps(sorted(pids))]
    addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
    if node not in addresses:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            f"root@{node}",
            shlex.join(command),
        ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=45, check=True)
    return json.loads(result.stdout)


def observe(root, registrations, *, inspect_node=probe):
    """Join persisted progress with fresh, generation-bound node observations."""
    owners = []
    unavailable = []
    grouped = {}
    for registration in registrations:
        path = Path(registration["state"])
        node = registration["node"]
        try:
            state = read(path)
        except (OSError, ValueError) as error:
            unavailable.append(
                {"node": node, "state": str(path), "status": "unavailable", "error": f"{type(error).__name__}: {error}"}
            )
            grouped.setdefault(node, set())
            continue
        owners.append((registration, path, state))
        identities = [state["generation"]]
        if state.get("child_generation"):
            identities.append(state["child_generation"])
        if state.get("restored_keeper"):
            identities.extend(state["restored_keeper"]["generation"].values())
        grouped.setdefault(node, set()).update(item["pid"] for item in identities)
    nodes = {}
    for node, pids in grouped.items():
        try:
            nodes[node] = {"observation_status": "ok", **inspect_node(node, pids)}
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            nodes[node] = {"observation_status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    result = {
        "time": datetime.now(UTC).isoformat(),
        "nodes": nodes,
        "owners": unavailable,
        "training": {"status": "not_prepared"},
        "needs_attention": False,
        "all_registered_owners_complete_and_restored": bool(owners) and not unavailable,
    }
    for registration, path, state in owners:
        node = registration["node"]
        observation = nodes[node]
        item = {
            "node": node,
            "state": str(path),
            "status": state["status"],
            "pid": state["pid"],
            "child_pid": state.get("child_pid"),
            "exit_code": state.get("exit_code"),
            "keeper_restore": state.get("keeper_restore"),
            "error": state.get("error"),
            "restoration_error": state.get("restoration_error"),
            "owner_liveness": "not_observed",
            "child_liveness": "not_observed",
        }
        if observation["observation_status"] == "ok":
            processes = observation["processes"]
            for field, identity in (
                ("owner_liveness", state["generation"]),
                ("child_liveness", state.get("child_generation")),
            ):
                if identity:
                    current = processes.get(str(identity["pid"]))
                    item[field] = "live" if same_generation(identity, current) else "exited_or_changed"
            if state["status"] == "running" and item["owner_liveness"] != "live":
                result["needs_attention"] = True
            restored = state.get("restored_keeper")
            if restored:
                occupied = {
                    (int(row[0].strip()), row[1].strip())
                    for row in csv.reader(io.StringIO(observation["compute"]))
                    if row
                }
                expected = {(int(pid), gpu) for pid, gpu in restored["occupied"]}
                item["restored_keeper_live"] = occupied == expected and all(
                    same_generation(identity, processes.get(str(identity["pid"])))
                    for identity in restored["generation"].values()
                )
                if not item["restored_keeper_live"]:
                    result["all_registered_owners_complete_and_restored"] = False
            else:
                result["all_registered_owners_complete_and_restored"] = False
        else:
            result["all_registered_owners_complete_and_restored"] = False
        recovered = False
        if registration.get("recovery"):
            try:
                recovery = read(Path(registration["recovery"]))
                recovered = (
                    recovery.get("status") == "complete"
                    and recovery.get("keeper_restore") == 0
                    and recovery.get("original") == str(path)
                    and recovery.get("original_sha256") == sha256(path)
                )
                item["restoration_recovery"] = "recorded" if recovered else "invalid_evidence"
            except (OSError, ValueError) as error:
                item["restoration_recovery"] = "unavailable"
                item["recovery_error"] = f"{type(error).__name__}: {error}"
        if state["status"] == "needs_attention" and not recovered:
            result["needs_attention"] = True
        if state["status"] != "complete" or state.get("keeper_restore") != 0:
            result["all_registered_owners_complete_and_restored"] = False
        receipt_path = registration.get("receipt")
        if receipt_path:
            try:
                receipt = read(Path(receipt_path))
                item["receipt_observation"] = "ok"
                item["checks"] = receipt.get("checks", {})
                commands = receipt.get("evidence", {}).get("gpu_commands", [])
                item["completed_commands"] = len(commands)
                item["last_command"] = commands[-1] if commands else None
            except (OSError, ValueError) as error:
                item["receipt_observation"] = "unavailable"
                item["receipt_error"] = f"{type(error).__name__}: {error}"
        result["owners"].append(item)
    if campaign_path(root).exists():
        try:
            progress = status(root)
            result["training"] = {
                key: value for key, value in progress.items() if key not in ("trials", "process_liveness")
            }
            result["training"]["trials"] = [
                {key: trial[key] for key in ("name", "status", "observed_steps", "planned_steps", "complete", "reasons")}
                | {
                    "latest_train_loss": (trial["latest_metrics"] or {}).get("train_loss"),
                    "latest_selection_loss": (trial["latest_metrics"] or {}).get("selection_loss"),
                }
                for trial in progress["trials"]
            ]
            if progress["status"] == "needs_attention":
                result["needs_attention"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["training"] = {"status": "invalid_evidence", "error": f"{type(error).__name__}: {error}"}
            result["needs_attention"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registrations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least five seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            registrations = read(args.registrations)
            if any(item["node"] not in NODES for item in registrations):
                raise ValueError("Monitor registration includes an unapproved node")
            result = observe(args.root, registrations)
            write(args.out / "latest.json", result)
            with (args.out / "observations.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "time": result["time"],
                        "training": result["training"]["status"],
                        "needs_attention": result["needs_attention"],
                        "owners": [
                            {"node": item["node"], "status": item["status"], "live": item.get("owner_liveness")}
                            for item in result["owners"]
                        ],
                    }
                ),
                flush=True,
            )
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
