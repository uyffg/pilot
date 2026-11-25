#!/usr/bin/env bash
# Start the experiment on this node: one queue worker per GPU set, then wait for them.
#
#   bash experiments/pilot/launch.sh
#
# Each worker takes jobs from the shared table (experiments/pilot/pilot.py) until none is left
# for it. Starting it again later resumes: finished jobs are skipped, the runner continues a
# training run at its last finished round, and an evaluation continues at its last saved
# episode. Run it on every node that takes part, with the same PILOT_OUT.
#
# PILOT_GPU_SETS="0,1 2,3 4,5 6,7"   one GPU set per worker (default: the node's GPUs in sets of
#                                    PILOT_GPUS_PER_RUN, 2)
# PILOT_STAGGER_SECONDS=45           pause between worker starts
# DRY_RUN=1                          print the plan and the command of every job; start nothing
# PILOT_SKIP_GATES=1                 start without a passed preflight and smoke (not for real runs)
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=experiments/pilot/env.sh
source "$HERE/env.sh"
cd "$PILOT_REPO"
PY=$PILOT_VENV/bin/python
PER_RUN=${PILOT_GPUS_PER_RUN:-2}

if [[ ${DRY_RUN:-0} == 1 ]]; then
    "${PY:-python3}" -m experiments.pilot.pilot plan --commands 2>/dev/null ||
        python3 -m experiments.pilot.pilot plan --commands
    exit 0
fi

default_sets() {
    local count i set sets=()
    count=$(nvidia-smi -L | wc -l)
    for ((i = 0; i + PER_RUN <= count; i += PER_RUN)); do
        set=$(seq -s, "$i" $((i + PER_RUN - 1)))
        sets+=("$set")
    done
    echo "${sets[*]}"
}
read -r -a SETS <<<"${PILOT_GPU_SETS:-$(default_sets)}"
((${#SETS[@]})) || { echo "launch.sh: no GPU set" >&2; exit 2; }
for set in "${SETS[@]}"; do
    IFS=, read -r -a devices <<<"$set"
    if ((${#devices[@]} != PER_RUN)); then
        echo "launch.sh: GPU set '$set' has ${#devices[@]} GPUs, PILOT_GPUS_PER_RUN is $PER_RUN" >&2
        exit 2
    fi
done

if [[ ${PILOT_SKIP_GATES:-0} != 1 ]]; then
    "$PY" - <<'PY'
import json, os, socket, sys
from pathlib import Path
out = Path(os.environ['PILOT_OUT'])
preflight = out / 'preflight' / f'{socket.gethostname()}.json'
if not preflight.exists() or not json.loads(preflight.read_text()).get('passed'):
    sys.exit(f'launch.sh: no passed preflight for this node ({preflight}); run preflight.py first')
if not (out / 'smoke' / 'smoke_passed.json').exists():
    sys.exit(f'launch.sh: no passed smoke in {out / "smoke"}; run smoke.sh first')
PY
    "$PY" "$HERE/prepare_bcb_full.py" --check
fi

mkdir -p "$PILOT_OUT/logs"
git -C "$PILOT_REPO" rev-parse HEAD >"$PILOT_OUT/logs/source_commit_$(hostname).txt" 2>/dev/null || true
pids=()
stop_workers() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait || true
    exit 130
}
trap stop_workers INT TERM

for i in "${!SETS[@]}"; do
    log=$PILOT_OUT/logs/worker_$(hostname)_slot$i.log
    "$PY" -m experiments.pilot.pilot worker --slot "$i" --gpus "${SETS[$i]}" >>"$log" 2>&1 &
    pids+=("$!")
    echo "slot=$i gpus=${SETS[$i]} pid=$! log=$log"
    if ((i + 1 < ${#SETS[@]})); then
        sleep "${PILOT_STAGGER_SECONDS:-45}"
    fi
done

status=0
for i in "${!SETS[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "slot=$i ended with unfinished or failed jobs; see $PILOT_OUT/logs" >&2
        status=1
    fi
done
"$PY" -m experiments.pilot.pilot summary || true
exit "$status"
