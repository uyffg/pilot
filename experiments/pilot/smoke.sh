#!/usr/bin/env bash
# The gate between preflight and the runs: the real model on the real GPUs, at the smallest size.
#
#   bash experiments/pilot/smoke.sh
#
# 1. Training smoke, both prompt profiles at the same time on the first two GPU sets, with the
#    runner's own --smoke mode (2 rounds of 1 task, 3 turns per episode). Each one serves the
#    base model, collects an episode with reviewer assessments in the sandbox, does one real
#    FSDP2 update of the 35B model, merges the adapter, serves the merged model and does it
#    again. Running the two side by side also checks that two runs can share a node.
# 2. Evaluation smoke on the first GPU set: two held-out tasks with the untrained model
#    (General prompts), then two with the adapter of the Specific training smoke, which takes
#    the merge path. Serve, collect, check, grade.
# 3. A summary with the timings, written to $PILOT_OUT/smoke/smoke_passed.json. launch.sh
#    refuses to start without it.
#
# A finished step is skipped on a rerun. A smoke adapter is never used by a real run: the runs
# start from the base model in their own directories.
#
# PILOT_GPU_SETS="0,1 2,3"   GPU sets to use (default: the node's GPUs in sets of PILOT_GPUS_PER_RUN, 2)
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=experiments/pilot/env.sh
source "$HERE/env.sh"
cd "$PILOT_REPO"
PY=$PILOT_VENV/bin/python
PER_RUN=${PILOT_GPUS_PER_RUN:-2}
SMOKE=$PILOT_OUT/smoke
mkdir -p "$SMOKE/logs"

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
((${#SETS[@]})) || { echo "smoke.sh: no GPU set" >&2; exit 2; }
say() { echo "[pilot smoke $(date -u +%T)] $*"; }

train_smoke() {
    local profile=$1 slot=$2
    CUDA_VISIBLE_DEVICES=${SETS[$slot]} SLURM_JOB_ID=$((7900 + slot)) \
        "$PY" -m experiments.bcb_opd.run --config "$HERE/configs/${profile}_r28.json" \
        --out "$SMOKE/train/$profile" --scratch "$PILOT_SCRATCH/smoke/train/$profile" \
        --agent-env "$PILOT_AGENT_ENV" --gpus "$PER_RUN" --smoke >"$SMOKE/logs/train_$profile.log" 2>&1
}

started=$(date +%s)
if [[ -f $SMOKE/train/General/smoke_complete.json && -f $SMOKE/train/Specific/smoke_complete.json ]]; then
    say "training smoke already complete"
else
    say "training smoke: General on GPUs ${SETS[0]}, Specific on GPUs ${SETS[1]:-${SETS[0]} (after General)}"
    if ((${#SETS[@]} >= 2)); then
        train_smoke General 0 &
        first=$!
        sleep "${PILOT_STAGGER_SECONDS:-45}"
        train_smoke Specific 1 &
        second=$!
        status=0
        wait "$first" || status=1
        wait "$second" || status=1
    else
        status=0
        train_smoke General 0 || status=1
        train_smoke Specific 0 || status=1
    fi
    if ((status)) || [[ ! -f $SMOKE/train/General/smoke_complete.json || ! -f $SMOKE/train/Specific/smoke_complete.json ]]; then
        say "PILOT_SMOKE_FAILED training smoke; last lines of the logs:"
        tail -n 25 "$SMOKE/logs/train_General.log" "$SMOKE/logs/train_Specific.log" || true
        exit 1
    fi
fi
train_seconds=$(($(date +%s) - started))

started=$(date +%s)
eval_smoke() {
    local name=$1
    shift
    if ! CUDA_VISIBLE_DEVICES=${SETS[0]} "$PY" -m experiments.pilot.evaluate --taskset heldout74 --limit 2 \
        --out "$SMOKE/eval/$name" --scratch "$PILOT_SCRATCH/smoke/eval/$name" --agent-env "$PILOT_AGENT_ENV" \
        --gpus "$PER_RUN" --port 21900 "$@" >"$SMOKE/logs/eval_$name.log" 2>&1; then
        say "PILOT_SMOKE_FAILED evaluation smoke $name; last lines of its log:"
        tail -n 25 "$SMOKE/logs/eval_$name.log" || true
        exit 1
    fi
}
say "evaluation smoke: untrained model, General prompts"
eval_smoke General_base --profile General --label base
say "evaluation smoke: adapter of the Specific training smoke (merge path)"
eval_smoke Specific_r2 --profile Specific --label r2 --adapter "$SMOKE/train/Specific/round_001/adapter"
eval_seconds=$(($(date +%s) - started))

TRAIN_SECONDS=$train_seconds EVAL_SECONDS=$eval_seconds SMOKE=$SMOKE GPU_SETS="${SETS[*]}" "$PY" - <<'PY'
import json, os, socket, time
from pathlib import Path
smoke = Path(os.environ['SMOKE'])
report = {'host': socket.gethostname(), 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'gpu_sets': os.environ['GPU_SETS'].split(), 'train_smoke_seconds': int(os.environ['TRAIN_SECONDS']),
          'eval_smoke_seconds': int(os.environ['EVAL_SECONDS']), 'training': {}, 'evaluation': {}}
for profile in ('General', 'Specific'):
    updates = []
    for index in range(2):
        adapter = smoke / 'train' / profile / f'round_{index:03d}' / 'adapter'
        update = json.loads((adapter / 'opd_update.json').read_text())
        state = json.loads((adapter / 'trainer_state.json').read_text())
        runtime = next((entry.get('train_runtime') for entry in reversed(state.get('log_history', []))
                        if 'train_runtime' in entry), None)
        peak = max((entry.get('gpu_peak_gib', 0) for entry in state.get('log_history', [])), default=None)
        if update['optimizer_steps'] != 1 or not update['mean_kl'] > 0 or update['reviewer_in_student_history']:
            raise SystemExit(f'PILOT_SMOKE_FAILED unexpected update record: {update}')
        updates.append({'round': index, 'pairs': update['pairs'], 'world_size': update['world_size'],
                        'mean_kl': update['mean_kl'], 'update_seconds': runtime, 'gpu_peak_gib': peak})
    report['training'][profile] = updates
for name in ('General_base', 'Specific_r2'):
    complete = json.loads((smoke / 'eval' / name / 'complete.json').read_text())
    report['evaluation'][name] = {'score': complete['score']['final_solution'], 'timing': complete['timing'],
                                  'exit_status_counts': complete['exit_status_counts']}
(smoke / 'smoke_passed.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
print('PILOT_SMOKE_OK')
PY
