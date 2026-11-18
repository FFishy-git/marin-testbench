#!/usr/bin/env bash
set -euo pipefail

task="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?usage: queue_wave5_validation.sh OUTPUT_ROOT CANDIDATE}"
candidate="${2:?missing isolated candidate source root}"
directory="$root/run-preparation/queue-wave5-validation"
mkdir -p "$directory"
exec 9>"$directory/queue.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

/opt/conda/envs/torch-base/bin/python -c \
    'import hashlib,json,pathlib,sys; q=json.load(open(sys.argv[1])); actual=hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest(); assert actual == q["queue_script_sha256"], "Queued validation script changed"' \
    "$root/run-preparation/wave5-candidate.json" "${BASH_SOURCE[0]}"

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    /opt/conda/envs/torch-base/bin/python "$task/tracking/snapshot.py" \
        --root "$root" --wave wave4 --out "$directory/$stamp.json"
    cp "$directory/$stamp.json" "$directory/latest.json"
    status="$(/opt/conda/envs/torch-base/bin/python -c \
        'import json,pathlib,sys; s=json.load(open(sys.argv[1])); status=next(r["status"] for r in s["runs"] if r["node"] == "28.216.17.9"); log=pathlib.Path(sys.argv[2])/"runs/wave4/keeper-restore-28.216.17.9.log"; restored=log.exists() and "keeper start exit 0" in log.read_text(); print("restore_pending" if status == "complete" and not restored else status)' \
        "$directory/$stamp.json" "$root")"
    printf '%s predecessor_status=%s\n' "$(date -Is)" "$status"
    if [[ "$status" == complete ]]; then
        /opt/conda/envs/torch-base/bin/python "$task/tracking/validate_wave5.py" \
            --root "$root" --candidate "$candidate"
        printf '%s\n' "$(date -Is)" > "$directory/validation_passed_at"
        break
    fi
    [[ "$status" != stopped_incomplete ]] || exit 1
    sleep 900
done
