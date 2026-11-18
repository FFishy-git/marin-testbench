#!/usr/bin/env bash
set -euo pipefail

helpers="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${1:?usage: queue_upstream_outer_gpu.sh OUTPUT_ROOT CPU_RECEIPT PROBE_DIRECTORY}"
receipt="${2:?CPU receipt required}"
out="${3:?Probe directory required}"
directory="$out/queue"
exec 9>"$directory/queue.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    result="$(/opt/conda/envs/torch-base/bin/python "$helpers/queue_upstream_outer_gpu.py" \
        --root "$root" --cpu-receipt "$receipt" --out "$out" --phase step)"
    status="$(/opt/conda/envs/torch-base/bin/python -c \
        'import json,sys; print(json.loads(sys.argv[1])["status"])' "$result")"
    printf '%s upstream_probe_handoff=%s\n' "$(date -Is)" "$status"
    [[ "$status" != needs_attention ]] || exit 1
    [[ "$status" != measured ]] || break
    sleep 900
done
