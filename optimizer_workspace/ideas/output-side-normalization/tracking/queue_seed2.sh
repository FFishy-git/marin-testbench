#!/usr/bin/env bash
set -euo pipefail

task="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?usage: queue_seed2.sh OUTPUT_ROOT}"
directory="$root/run-preparation/queue-seed2-companion"
mkdir -p "$directory"
exec 9>"$directory/queue.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$directory/pid"
trap 'printf "%s\n" "$(date -Is)" > "$directory/stopped_at"' EXIT

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    /opt/conda/envs/torch-base/bin/python "$task/tracking/snapshot.py" \
        --root "$root" --wave wave4 --out "$directory/$stamp.json"
    cp "$directory/$stamp.json" "$directory/latest.json"
    status="$(/opt/conda/envs/torch-base/bin/python -c \
        'import json,pathlib,sys; s=json.load(open(sys.argv[1])); status=next(r["status"] for r in s["runs"] if r["node"] == "28.216.17.144"); log=pathlib.Path(sys.argv[2])/"runs/wave4/keeper-restore-28.216.17.144.log"; restored=log.exists() and "keeper start exit 0" in log.read_text(); print("restore_pending" if status == "complete" and not restored else status)' \
        "$directory/$stamp.json" "$root")"
    printf '%s predecessor_status=%s\n' "$(date -Is)" "$status"
    if [[ "$status" == complete ]]; then
        ssh -o BatchMode=yes -o ConnectTimeout=10 root@28.216.17.144 \
            "cd '$task' && PYTHONPATH='$task' /opt/conda/envs/torch-base/bin/python tracking/seed2_companion.py --root '$root'"
        /opt/conda/envs/torch-base/bin/python -c \
            'import pathlib,subprocess,sys; root,task=map(pathlib.Path,sys.argv[1:]); directory=root/"run-preparation/monitor-wave4-baseline"; directory.mkdir(exist_ok=True); log=(directory/"monitor.log").open("a"); p=subprocess.Popen(["bash",str(task/"tracking/monitor.sh"),str(root),"wave4-baseline"],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print("baseline monitor PID",p.pid)' \
            "$root" "$task"
        printf '%s\n' "$(date -Is)" > "$directory/companion_launched_at"
        break
    fi
    [[ "$status" != stopped_incomplete ]] || exit 1
    sleep 900
done
