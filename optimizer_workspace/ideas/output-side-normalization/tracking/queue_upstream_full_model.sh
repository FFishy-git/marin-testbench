#!/usr/bin/env bash
set -euo pipefail

helpers="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${1:?usage: queue_upstream_full_model.sh OUTPUT_ROOT}"
directory="$root/run-preparation/upstream-full-model-corrected-queue"
exec 9>"$directory/queue.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    result="$(/opt/conda/envs/torch-base/bin/python "$helpers/queue_upstream_full_model.py" \
        --root "$root" --phase step)"
    status="$(/opt/conda/envs/torch-base/bin/python -c \
        'import json,sys; print(json.loads(sys.argv[1])["status"])' "$result")"
    printf '%s upstream_full_model_handoff=%s\n' "$(date -Is)" "$status"
    [[ "$status" != needs_attention ]] || exit 1
    [[ "$status" != validated ]] || break
    sleep 900
done
