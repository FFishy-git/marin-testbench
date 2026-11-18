#!/usr/bin/env bash
set -euo pipefail

task="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?usage: queue_wave5_launch.sh OUTPUT_ROOT}"
directory="$root/run-preparation/queue-wave5-launch"
mkdir -p "$directory"
exec 9>"$directory/queue.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

/opt/conda/envs/torch-base/bin/python -c \
    'import hashlib,json,pathlib,sys; root,task=map(pathlib.Path,sys.argv[1:]); q=json.load(open(root/"run-preparation/wave5-launch-queue.json")); assert all(hashlib.sha256((task/n).read_bytes()).hexdigest()==s for n,s in q["helpers_sha256"].items()), "Queued launch helpers changed"' \
    "$root" "$task"

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    /opt/conda/envs/torch-base/bin/python "$task/tracking/snapshot.py" \
        --root "$root" --wave wave4 --out "$directory/$stamp-wave4.json"
    cp "$directory/$stamp-wave4.json" "$directory/latest-wave4.json"
    if [[ -f "$root/run-preparation/wave4-baseline.json" ]]; then
        /opt/conda/envs/torch-base/bin/python "$task/tracking/snapshot.py" \
            --root "$root" --wave wave4-baseline --out "$directory/$stamp-baseline.json"
        cp "$directory/$stamp-baseline.json" "$directory/latest-baseline.json"
    fi
    status="$(/opt/conda/envs/torch-base/bin/python "$task/tracking/launch_wave5.py" \
        --root "$root" --queue-status "$directory")"
    printf '%s launch_status=%s\n' "$(date -Is)" "$status"
    if [[ "$status" == ready ]]; then
        /opt/conda/envs/torch-base/bin/python "$task/tracking/launch_wave5.py" --root "$root"
        printf '%s\n' "$(date -Is)" > "$directory/launched_at"
        break
    fi
    [[ "$status" != needs_attention ]] || exit 1
    sleep 900
done
