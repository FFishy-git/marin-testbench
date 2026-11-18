#!/usr/bin/env bash
set -euo pipefail

root="${1:?usage: monitor_validation.sh OUTPUT_ROOT NODE PID WAVE}"
node="${2:?missing node}"
pid="${3:?missing validation PID}"
wave="${4:?missing wave}"
[[ "$pid" =~ ^[1-9][0-9]*$ ]]
directory="$root/run-preparation/monitor-validation-$wave"
mkdir -p "$directory"
exec 9>"$directory/monitor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    rc=0
    ssh -o BatchMode=yes -o ConnectTimeout=10 "root@$node" \
        "ps -p $pid -o pid=,stat=,etime=,args=" > "$directory/$stamp.process.txt" 2>&1 || rc=$?
    printf '%s observation_exit=%s\n' "$(date -Is)" "$rc"
    if [[ -f "$root/run-preparation/validation-$wave.json" ]]; then
        cp "$root/run-preparation/validation-$wave.json" "$directory/$stamp.receipt.json"
    fi
    if [[ "$rc" == 1 ]] || awk 'NR == 1 && $2 ~ /^Z/ { found=1 } END { exit !found }' \
        "$directory/$stamp.process.txt"; then
        printf '%s\n' "$(date -Is)" > "$directory/process_exited_at"
        break
    fi
    sleep 900
done
