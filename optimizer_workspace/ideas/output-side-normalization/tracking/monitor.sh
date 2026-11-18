#!/usr/bin/env bash
set -euo pipefail

task="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?usage: monitor.sh OUTPUT_ROOT WAVE}"
wave="${2:?usage: monitor.sh OUTPUT_ROOT WAVE}"
directory="$root/run-preparation/monitor-$wave"
mkdir -p "$directory"
exec 9>"$directory/monitor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    status="$(/opt/conda/envs/torch-base/bin/python "$task/tracking/snapshot.py" \
        --root "$root" --wave "$wave" --out "$directory/$stamp.json")"
    cp "$directory/$stamp.json" "$directory/latest.json"
    printf '%s %s\n' "$(date -Is)" "$status"
    if [[ "$status" == complete ]]; then
        /opt/conda/envs/torch-base/bin/python "$task/tracking/analyze_wave.py" --root "$root" --wave "$wave"
        export HTTP_PROXY=http://star-proxy.oa.com:3128
        export HTTPS_PROXY="$HTTP_PROXY"
        export ALL_PROXY="$HTTP_PROXY"
        "$root/.venv-logging/bin/python" "$task/tracking/swanlab_upload.py" \
            --root "$root" --wave "$wave" \
            --credential-file /root/dl/runtime/script/.env
        printf '%s\n' "$(date -Is)" > "$directory/analysis_and_upload_verified_at"
        break
    fi
    [[ "$status" != needs_attention ]] || exit 1
    sleep 900
done
