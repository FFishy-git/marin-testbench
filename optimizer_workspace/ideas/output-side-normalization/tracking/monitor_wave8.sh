#!/usr/bin/env bash
set -euo pipefail

helpers="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${1:?usage: monitor_wave8.sh OUTPUT_ROOT}"
directory="$root/run-preparation/monitor-wave8"
mkdir -p "$directory"
exec 9>"$directory/monitor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    /opt/conda/envs/torch-base/bin/python "$helpers/wave8_release.py" \
        --root "$root" --phase snapshot --out "$directory/$stamp.json"
    cp "$directory/$stamp.json" "$directory/latest.json"
    status="$(/opt/conda/envs/torch-base/bin/python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$directory/latest.json")"
    printf '%s %s\n' "$(date -Is)" "$status"
    if /opt/conda/envs/torch-base/bin/python "$helpers/analyze_wave8.py" --root "$root"; then
        analysis_ok=true
    else
        printf '%s analysis_incomplete_or_refused\n' "$(date -Is)"
        analysis_ok=false
    fi
    if [[ "$status" == complete && "$analysis_ok" == true ]]; then
        if HTTP_PROXY=http://star-proxy.oa.com:3128 \
            HTTPS_PROXY=http://star-proxy.oa.com:3128 \
            ALL_PROXY=http://star-proxy.oa.com:3128 \
            "$root/.venv-logging/bin/python" "$helpers/swanlab_wave8.py" \
                --root "$root" --credential-file /root/dl/runtime/script/.env; then
            /opt/conda/envs/torch-base/bin/python -c \
                'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "verified"' \
                "$root/run-preparation/swanlab-upload-wave8.json"
            printf '%s\n' "$(date -Is)" > "$directory/analysis_and_upload_verified_at"
            break
        fi
        printf '%s upload_or_verification_refused\n' "$(date -Is)"
    fi
    [[ "$status" != needs_attention ]] || exit 1
    sleep 900
done
