#!/usr/bin/env bash
set -euo pipefail

task="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?usage: monitor_endeavor.sh OUTPUT_ROOT}"
directory="$root/run-preparation/monitor-endeavor"
mkdir -p "$directory"
exec 9>"$directory/monitor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    if /opt/conda/envs/torch-base/bin/python "$task/tracking/update_endeavor.py" \
        --root "$root" --document "$task/ENDEAVOR.md"; then
        printf '%s ledger_refreshed\n' "$(date -Is)"
    else
        printf '%s ledger_refresh_refused_retry_next_cycle\n' "$(date -Is)"
    fi
    sleep 900
done
