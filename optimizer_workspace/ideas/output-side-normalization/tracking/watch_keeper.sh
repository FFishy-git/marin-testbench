#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: watch_keeper.sh TRAIN_PID RUN_DIRECTORY}"
out="${2:?missing run directory}"
[[ "$pid" =~ ^[1-9][0-9]*$ ]]
identity="$(ps -p "$pid" -o args=)"
[[ "$identity" == *"-m torch.distributed.run"* && "$identity" == *"$out"* ]]
started="$(awk '{print $22}' "/proc/$pid/stat")"
printf '%s watching training PID %s for %s\n' "$(date -Is)" "$pid" "$out"

while [[ -r "/proc/$pid/stat" ]]; do
    read -r state current < <(awk '{print $3, $22}' "/proc/$pid/stat")
    [[ "$state" != Z ]] || break
    if [[ "$current" != "$started" ]]; then
        printf '%s PID reused; keeper restore refused\n' "$(date -Is)"
        exit 1
    fi
    sleep 15
done

printf '%s training PID %s exited\n' "$(date -Is)" "$pid"
occupied="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$occupied" ]]; then
    printf '%s CUDA occupancy remains; keeper restore refused: %s\n' "$(date -Is)" "$occupied"
    exit 1
fi
rc=0
env GPU_RESOURCE_MANAGER_LOCAL_ONLY=1 \
    bash /root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh start || rc=$?
printf '%s keeper start exit %s\n' "$(date -Is)" "$rc"
exit "$rc"
