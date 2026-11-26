"""Spawn run_wd_campaign.py detached in a new session and verify all 4 nodes enter active allocations."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "wd-ablation-20261127"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")


def main() -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    pid_file = CAMPAIGN_ROOT / "orchestrator.pid"
    stdout_log = CAMPAIGN_ROOT / "orchestrator.stdout.log"

    if pid_file.is_file():
        old_pid = int(pid_file.read_text().strip())
        if Path(f"/proc/{old_pid}").exists():
            raise RuntimeError(f"Orchestrator already running with PID {old_pid}")

    with stdout_log.open("a") as stream:
        proc = subprocess.Popen(
            [PYTHON, "-u", str(TASK / "campaign" / "run_wd_campaign.py")],
            cwd=TASK,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f"{proc.pid}\n")
    print(f"Spawned detached orchestrator PID={proc.pid}, log={stdout_log}", flush=True)

    # Wait up to 45s for all 4 nodes to write running allocation state files
    alloc_dir = CAMPAIGN_ROOT / "allocations"
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Orchestrator exited prematurely with code {proc.returncode}!")
        states = {}
        if alloc_dir.is_dir():
            for p in sorted(alloc_dir.glob("train_*.state.json")):
                data = json.loads(p.read_text())
                if data.get("status") in ("preflight", "running"):
                    for n in NODES:
                        if p.name.endswith(f"_{n.replace('.', '_')}.state.json"):
                            states[n] = {"file": p.name, "status": data.get("status"), "child_pid": data.get("child_pid")}
        if len(states) == 4 and all(v["status"] == "running" for v in states.values()):
            print(json.dumps({"orchestrator_pid": proc.pid, "active_node_allocations": states}, indent=2))
            return
        time.sleep(2.0)

    print("Warning: timed out waiting for all 4 running states; inspect orchestrator.log", flush=True)


if __name__ == "__main__":
    main()
