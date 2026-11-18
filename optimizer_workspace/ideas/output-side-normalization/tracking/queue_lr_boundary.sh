#!/usr/bin/env bash
set -euo pipefail

helpers="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
campaign="${1:?usage: queue_lr_boundary.sh CAMPAIGN_JSON}"
directory="$(dirname -- "$campaign")/queue"
exec 9>"$directory/queue.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    result="$(/opt/conda/envs/torch-base/bin/python "$helpers/queue_lr_boundary.py" \
        --campaign "$campaign" --phase step)"
    status="$(/opt/conda/envs/torch-base/bin/python -c \
        'import json,sys; print(json.loads(sys.argv[1])["status"])' "$result")"
    printf '%s boundary_handoff=%s\n' "$(date -Is)" "$status"
    [[ "$status" != needs_attention ]] || exit 1
    [[ "$status" != launched ]] || break
    sleep 900
done
