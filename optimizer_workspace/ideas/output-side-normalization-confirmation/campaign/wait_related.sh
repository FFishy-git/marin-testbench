#!/usr/bin/env bash
set -euo pipefail

# Wait for the live "related-selected-full-seed12-20261121" campaign to complete
# cleanly across all 4 A100 nodes without modifying any files in the predecessor folder.

PYTHON="/opt/conda/envs/torch-base/bin/python"
STATUS_SCRIPT="/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization/tracking/related_full_campaign.py"
PVC_ROOT="/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
POLL_SECONDS="${POLL_SECONDS:-120}"
NODES=("28.216.17.9" "28.216.17.80" "28.216.17.144" "28.216.17.139")
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

echo "[$(date -Is)] Waiting for related-selected-full-seed12-20261121 to finish (poll every ${POLL_SECONDS}s)..."
CONSECUTIVE_HARD_FAILS=0

while true; do
    out="$("$PYTHON" "$STATUS_SCRIPT" --root "$PVC_ROOT" --phase status 2>/dev/null)" || {
        echo "[$(date -Is)] WARNING: status script returned non-zero; retrying in ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
        continue
    }

    summary="$("$PYTHON" - << 'EOF'
import json, sys
d = json.load(sys.stdin)
status = d.get("status")
done = d.get("completed_trials", 0)
total = d.get("planned_trials", 12)
running = [f"{t['name']}({t['observed_steps']}/{t['planned_steps']})" for t in d.get("trials", []) if t.get("status"] == "running"]
hard_failed = [f"{t['name']}({t['status']})" for t in d.get("trials", []) if t.get("status"] == "failed"]
transient = [f"{t['name']}({t['status']})" for t in d.get("trials", []) if t.get("status"] == "invalid_evidence"]
not_started = [t["name"] for t in d.get("trials", []) if t.get("status"] == "not_started"]
print(f"STATUS={status}|DONE={done}|TOTAL={total}|RUNNING={','.join(running)}|NOT_STARTED={len(not_started)}|TRANSIENT={','.join(transient)}|HARD_FAILED={','.join(hard_failed)}")
EOF
<<< "$out")"

    echo "[$(date -Is)] $summary"

    hard_failed_val="${summary##*HARD_FAILED=}"
    if [[ -n "$hard_failed_val" ]]; then
        CONSECUTIVE_HARD_FAILS=$((CONSECUTIVE_HARD_FAILS + 1))
        echo "[$(date -Is)] WARNING: Hard failure detected (${CONSECUTIVE_HARD_FAILS}/3): $hard_failed_val"
        if [[ "$CONSECUTIVE_HARD_FAILS" -ge 3 ]]; then
            echo "[$(date -Is)] ERROR: Persistent hard failure in related-selected-full-seed12-20261121: $hard_failed_val"
            exit 2
        fi
    else
        CONSECUTIVE_HARD_FAILS=0
    fi

    if [[ "$summary" == STATUS=complete\|DONE=12\|TOTAL=12\|* ]]; then
        echo "[$(date -Is)] All 12/12 trials report status=complete. Checking 4 nodes for leftover training processes..."
        busy=0
        for node in "${NODES[@]}"; do
            procs="$(ssh "${SSH_OPTS[@]}" "root@${node}" "ps -ef | grep -E 'experiment\.train|torch\.distributed\.run|related_full_sequence|related_full_execution' | grep -v grep || true")"
            if [[ -n "$procs" ]]; then
                echo "[$(date -Is)] Node ${node} still has active campaign processes:"
                echo "$procs"
                busy=1
            fi
        done
        if [[ "$busy" -eq 0 ]]; then
            echo "[$(date -Is)] Verified: related-selected-full-seed12-20261121 is 12/12 complete and all 4 nodes are clear of campaign processes!"
            exit 0
        fi
        echo "[$(date -Is)] Waiting 30s for final cleanup/keeper restoration on nodes..."
        sleep 30
        continue
    fi

    sleep "$POLL_SECONDS"
done
