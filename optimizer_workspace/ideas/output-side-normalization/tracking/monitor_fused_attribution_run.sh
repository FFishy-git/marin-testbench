#!/usr/bin/env bash
set -euo pipefail

helpers="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${1:?usage: monitor_fused_attribution_run.sh OUTPUT_ROOT WAVE}"
wave="${2:?usage: monitor_fused_attribution_run.sh OUTPUT_ROOT WAVE}"
directory="$root/run-preparation/monitor-$wave"
mkdir -p "$directory"
exec 9>"$directory/monitor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    if /opt/conda/envs/torch-base/bin/python -P "$helpers/fused_attribution_production.py" \
        --root "$root" --wave "$wave" --phase snapshot --out "$directory/$stamp.json"; then
        cp "$directory/$stamp.json" "$directory/latest.json.tmp"
        mv "$directory/latest.json.tmp" "$directory/latest.json"
        status="$(/opt/conda/envs/torch-base/bin/python -c \
            'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$directory/latest.json")"
        printf '%s %s\n' "$(date -Is)" "$status"
        analysis_ok=false
        if /opt/conda/envs/torch-base/bin/python -P "$helpers/analyze_fused_attribution_run.py" \
            --root "$root" --wave "$wave"; then
            analysis_ok=true
        else
            printf '%s analysis_incomplete_or_refused\n' "$(date -Is)"
        fi
        if [[ "$status" == complete && "$analysis_ok" == true ]]; then
            if HTTP_PROXY=http://star-proxy.oa.com:3128 \
                HTTPS_PROXY=http://star-proxy.oa.com:3128 \
                ALL_PROXY=http://star-proxy.oa.com:3128 \
                "$root/.venv-logging/bin/python" -P "$helpers/swanlab_fused_attribution_run.py" \
                    --root "$root" --wave "$wave" --credential-file /root/dl/runtime/script/.env; then
                /opt/conda/envs/torch-base/bin/python -P "$helpers/fused_attribution_production.py" \
                    --root "$root" --wave "$wave" --phase completion
                printf '%s\n' "$(date -Is)" > "$directory/analysis_and_upload_verified_at"
                break
            fi
            printf '%s upload_or_verification_refused\n' "$(date -Is)"
        fi
        [[ "$status" != needs_attention ]] || exit 1
    else
        printf '%s snapshot_refused_retry_original_handles_next_cycle\n' "$(date -Is)"
    fi
    sleep 900
done
