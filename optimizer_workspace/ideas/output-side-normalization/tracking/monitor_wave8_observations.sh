#!/usr/bin/env bash
set -euo pipefail

helpers="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${1:?usage: monitor_wave8_observations.sh OUTPUT_ROOT}"
directory="$root/run-preparation/monitor-wave8-observations"
mkdir -p "$directory"
exec 9>"$directory/monitor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    if /opt/conda/envs/torch-base/bin/python \
        "$root/controllers/wave8-launch-20261117/tracking/wave8_release.py" \
        --root "$root" --phase snapshot --out "$directory/$stamp-process.json"; then
        status="$(/opt/conda/envs/torch-base/bin/python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
            "$directory/$stamp-process.json")"
        printf '%s process=%s\n' "$(date -Is)" "$status"
        if [[ "$status" == complete || "$status" == needs_attention ]]; then
            printf '%s terminal_process=%s\n' "$(date -Is)" "$status"
            break
        fi
        if /opt/conda/envs/torch-base/bin/python "$helpers/analyze_wave8_snapshot.py" \
            --root "$root" --observation "$root/run-preparation/wave8-observations/$stamp"; then
            printf '%s strict_comparison_saved\n' "$(date -Is)"
        else
            printf '%s strict_observation_refused_evidence_preserved\n' "$(date -Is)"
        fi
    else
        printf '%s process_observation_unknown_no_restart\n' "$(date -Is)"
    fi
    sleep 900
done
