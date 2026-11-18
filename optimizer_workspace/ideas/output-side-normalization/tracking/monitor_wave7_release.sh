#!/usr/bin/env bash
set -euo pipefail

helpers="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${1:?usage: monitor_wave7_release.sh OUTPUT_ROOT}"
directory="$root/run-preparation/monitor-wave7"
mkdir -p "$directory"
exec 9>"$directory/monitor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    status="$(/opt/conda/envs/torch-base/bin/python "$helpers/snapshot_wave7_release.py" \
        --root "$root" --out "$directory/$stamp.json")"
    cp "$directory/$stamp.json" "$directory/latest.json"
    printf '%s %s\n' "$(date -Is)" "$status"
    if [[ "$status" == complete ]]; then
        /opt/conda/envs/torch-base/bin/python "$helpers/analyze_wave7_release.py" --root "$root"
        HTTP_PROXY=http://star-proxy.oa.com:3128 \
        HTTPS_PROXY=http://star-proxy.oa.com:3128 \
        ALL_PROXY=http://star-proxy.oa.com:3128 \
        "$root/.venv-logging/bin/python" "$helpers/swanlab_wave7_release.py" \
            --root "$root" --credential-file /root/dl/runtime/script/.env
        printf '%s\n' "$(date -Is)" > "$directory/analysis_and_upload_verified_at"
        break
    fi
    [[ "$status" != needs_attention ]] || exit 1
    sleep 900
done
