#!/usr/bin/env bash
set -euo pipefail

helpers="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
campaign="${1:?usage: monitor_fixed_age_tuning.sh CAMPAIGN_JSON}"
directory="$(dirname -- "$campaign")/monitor"
mkdir -p "$directory"
exec 9>"$directory/monitor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    /opt/conda/envs/torch-base/bin/python "$helpers/run_fixed_age_tuning.py" \
        --campaign "$campaign" --phase snapshot --out "$directory/$stamp.json"
    cp "$directory/$stamp.json" "$directory/latest.json"
    status="$(/opt/conda/envs/torch-base/bin/python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$directory/latest.json")"
    printf '%s %s\n' "$(date -Is)" "$status"
    analyzer="$(/opt/conda/envs/torch-base/bin/python -c \
        'import sys; sys.path.insert(0,sys.argv[1]); from pathlib import Path; from fixed_age_protocol import analyzer_name; print(analyzer_name(Path(sys.argv[2])))' \
        "$helpers" "$campaign")"
    if /opt/conda/envs/torch-base/bin/python "$helpers/$analyzer" \
        --campaign "$campaign" --out "$(dirname -- "$campaign")/analysis.json"; then
        analysis_ok=true
    else
        printf '%s analysis_refused\n' "$(date -Is)"
        analysis_ok=false
    fi
    root="$(/opt/conda/envs/torch-base/bin/python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$campaign")"
    if HTTP_PROXY=http://star-proxy.oa.com:3128 \
        HTTPS_PROXY=http://star-proxy.oa.com:3128 \
        ALL_PROXY=http://star-proxy.oa.com:3128 \
        "$root/.venv-logging/bin/python" "$helpers/swanlab_fixed_age_tuning.py" \
            --campaign "$campaign" --credential-file /root/dl/runtime/script/.env; then
        upload_ok=true
    else
        printf '%s upload_or_verification_refused\n' "$(date -Is)"
        upload_ok=false
    fi
    if [[ "$status" == complete && "$analysis_ok" == true && "$upload_ok" == true ]]; then
        /opt/conda/envs/torch-base/bin/python -c \
            'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "verified"' \
            "$(dirname -- "$campaign")/swanlab-upload.json"
        printf '%s\n' "$(date -Is)" > "$directory/analysis_and_upload_verified_at"
        break
    fi
    [[ "$status" != needs_attention ]] || exit 1
    sleep 900
done
