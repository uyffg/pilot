# Pilot experiments: launch layer

Everything needed to run the first two pilot experiments on a GPU node, without changing the
distillation code. The runner, the trainer, the server, the prompts and the sandbox are the 50
files this repository started with; this directory only adds files next to them.

## What runs

| | Experiment 1: more epochs on BCB-Hard | Experiment 2: full BigCodeBench |
|---|---|---|
| Training | One run per prompt profile (General, Specific) with the unchanged runner: 28 updates of 8 episodes instead of 10. That is three passes over the 74 training tasks, plus two tasks. Seed 42. | None. |
| Evaluated models | The untrained model, and the adapters after 10, 19 and 28 updates (about one, two and three passes). The first 10 updates are the original recipe. | The untrained model, and the adapter after 10 updates. |
| Evaluated tasks | The 74 held-out BCB-Hard tasks. | The 992 tasks outside BCB-Hard. With the 74 held-out tasks that is the 1,066 tasks that were never trained on. |

That is 26 jobs: 2 training runs, 8 evaluations on 74 tasks, and 16 evaluation parts of 248
tasks (each of the four evaluations on 992 tasks is split in four). Each job uses one GPU set
(two GPUs by default). `python -m experiments.pilot.pilot plan` lists them in the order a free
GPU set takes them.

The untrained model is evaluated once per profile, because the two profiles give the worker
different prompts.

## Steps on a node

On Merlin, every command below runs inside the training image: prefix it with
`bash experiments/pilot/merlin/enter.sh` (see that file for why and for what the node needs).
On another Linux host with a driver that supports CUDA 13, run the commands as they are, with
`PILOT_ROOT` set to a node-local directory.

```bash
bash experiments/pilot/setup_env.sh                       # 1. environments and sandbox (about 30 GB)
source experiments/pilot/env.sh
python experiments/pilot/fetch_model.py                   # 2. the model at the pinned revision (72 GB)
python experiments/pilot/preflight.py --gpus 8            # 3. checks of this node, about 10 minutes
bash experiments/pilot/smoke.sh                           # 4. the real model at the smallest size, 1-2 hours
bash experiments/pilot/launch.sh                          # 5. the experiment; returns when this node's work is done
```

1. `setup_env.sh` builds three environments from lock files and decides how the sandbox runs.
   It prints `PILOT_SETUP_OK` and writes the node's package lists to `$PILOT_OUT/env/<host>/`.
2. `fetch_model.py` downloads `Qwen/Qwen3.5-35B-A3B` at revision `59d61f3c`, compares every
   file with the published size and hash, and writes the `bcb_verified.json` the runner
   requires. It prints `PILOT_MODEL_OK`. A verified copy can be moved between nodes and
   re-checked with `--verify-only`.
3. `preflight.py` prints one PASS, WARN or FAIL line per check and `PILOT_PREFLIGHT_OK`.
4. `smoke.sh` runs the runner's own smoke mode for both profiles side by side, then two tiny
   evaluations. It prints `PILOT_SMOKE_OK` and the timings.
5. `launch.sh` refuses to start without a passed preflight of this node and a passed smoke.
   `DRY_RUN=1 bash experiments/pilot/launch.sh` prints the plan and every command.

`PILOT_OUT` holds all results. It defaults to `$PILOT_ROOT/out` on the node. When the GPU sets
are on several nodes, set it to one directory that all of them see (on Merlin, on the devbox
share) before every step: the evaluations on one node read the adapters trained on another,
and the job table is shared through it. The preflight checks that locking works there.

## While it runs

```bash
python -m experiments.pilot.pilot status      # every job: state, node, progress
python -m experiments.pilot.pilot summary     # the result table so far (markdown; --json for numbers)
python -m experiments.pilot.wandb_sync --loop 900    # optional mirror to Weights & Biases
```

- Logs: `$PILOT_OUT/logs/<job>.log` per job and `worker_<host>_slot<i>.log` per GPU set.
- A job that fails is recorded in `$PILOT_OUT/queue/<job>/failed.json` with the end of its log,
  and is not run again by itself. After the cause is removed:
  `python -m experiments.pilot.pilot retry <job>`, then `launch.sh` again.
- Starting `launch.sh` again always resumes. A training run continues after its last finished
  update, an evaluation after its last saved episode, and finished jobs are skipped.
- A job whose worker died is taken over by another worker after 15 minutes without a heartbeat.
- Stopping: send TERM to `launch.sh`. The workers stop their servers and give their jobs back.

## Results

```
$PILOT_OUT/train/<Profile>_r28/round_NNN/{episodes.jsonl,pairs.jsonl,adapter/}   the runner's own layout
$PILOT_OUT/eval/<taskset>/<Profile>/<model>/<part>/
    eval_episodes.jsonl      one row per task, with the final solution.py
    grades.json              per-task test results
    complete.json            score, timing, exit statuses, counts of the two special cases below
    trajectories.tar.gz      the full Mini-SWE trajectory of every episode
$PILOT_OUT/queue/            job states
$PILOT_OUT/{env,preflight,smoke}/   what each node installed and checked
```

`<model>` is `base` or `r<N>` (the adapter after N updates). The reported number is
`final_solution`: the share of tasks whose final `solution.py` passes the benchmark's tests,
with every task in the denominator. `first_submit_solution` is recorded too.

The runner ends each training run with its own evaluation of the last adapter on the 74
held-out tasks (`train/<Profile>_r28/grades.json`). The table reports the evaluation job
instead, so that all four models are evaluated by one procedure; the runner's number is listed
beside it as a second sample of the same model.

## Evaluation rules

`evaluate.py` repeats the runner's final evaluation: the same merge, server, collection and
grading commands, with the model and the task list replaced. `tests/test_pilot.py` compares the
commands with the runner's source. Two cases the runner's evaluation stops on are settled
here, the same way for every model, with every task kept in the denominator:

- An episode that ends in `ContextWindowExceededError` is kept and its final file is graded as
  it is. This is the rule of the original matched-base evaluation.
- An episode that ends in any other exception, or is missing, is an infrastructure failure.
  The task is collected again from an empty workspace, at most twice. If a failure remains,
  the evaluation is not graded and its job fails.

## What differs from the original runs

These are properties of the node, not choices of the experiment. They hold for every model
alike, so comparisons inside this experiment are not affected. Comparisons with the numbers of
the original runs are.

- **Hardware and server layout.** Two H100 per run (tensor-parallel 2). The original runs used
  four 48 GB GPUs or two 80 GB GPUs.
- **Package versions.** The original runs did not record theirs. See "Package versions".
- **Sandbox `/proc`.** On a node whose `/proc` is masked (Merlin), Bubblewrap cannot mount a new
  procfs, and the sandbox gets the host's `/proc` bound instead (`sandbox_probe.py`, one of
  the `bind-proc` modes). Code that looks up its own process id in `/proc` then sees another
  process.
- **User.** Inside the Merlin image everything runs as user 0 of a user namespace, including
  the agent's commands and the grading. A test that expects a permission error can behave
  differently.
- Among the 1,066 evaluated tasks, the tests of seven touch these: BigCodeBench/459, 460, 779,
  812 (file permissions), 21, 377 (`psutil`) and 1042 (user name; a held-out hard task). They
  stay in the denominator and are worth reporting separately.
- **Agent environment.** Python 3.9 from a relocatable build with the benchmark's own
  requirements, installed with uv. The original used a conda Python 3.9 and pip.

## Package versions

| Environment | Lock file | Fixed by |
|---|---|---|
| Training and serving | `requirements/train-serve.lock.txt` (203 packages) | vLLM 0.25.1, torch 2.11.0, transformers 5.15.1, FLA 0.5.2, liger-kernel 0.8.2, peft 0.20.0, accelerate 1.14.0; the rest as of 2026-08-31 |
| Mini-SWE | `requirements/frontend.lock.txt` (82 packages) | mini-swe-agent 2.4.6; the rest as of 2026-11-07, which gives litellm 1.108.2 and tokenizers 0.23.3 |
| Agent (sandbox, grading) | `requirements/agent.lock.txt` (170 packages) | the benchmark's `requirements-eval.txt`; the rest as of 2026-11-04 |

Where the versions come from: the repository's history names vLLM 0.25.1, torch 2.11.0+cu130
and "Transformers 5.15" for the training side, the trainer prints "FLA 0.5.2", and the original
work log names tokenizers 0.23.3 for the Mini-SWE side, which the resolution above reproduces.
liger-kernel, peft and accelerate are the releases that were current with those.

What was checked before any node ran this:

- Each pinned package was opened and searched for what the code imports or patches:
  `modeling_qwen3_5_moe.torch_chunk_gated_delta_rule` and `Qwen3_5MoeRMSNormGated` in
  transformers, `fsdp2_apply_ac` in accelerate, `model/qwen3_5_moe.py` and the model type
  `qwen3_5_moe` in liger-kernel, `ops/gated_delta_rule` in FLA, `exclude_modules` in peft,
  `BASH_TOOL` in mini-swe-agent, `--language-model-only` and `qwen3_coder` in vLLM.
- The three lock files resolve for Linux x86_64 (uv, Python 3.12 and 3.9).
- With torch 2.11.0, transformers 5.15.1 and peft 0.20.0 on a CPU: the repository's two
  torch tests pass, the chat template of the real tokenizer aligns the teacher and student
  views, and the repository's merge command round-trips a LoRA adapter on a tiny Qwen3.5 MoE
  built from the real model configuration.
- With the Mini-SWE lock file installed: all 53 unit tests of the repository pass.

What was not checked, because it needs a GPU node: vLLM serving the 35B model, the Triton
kernels, the FSDP2 update, Bubblewrap itself. The preflight and the smoke are those checks.

## Requirements per node

- GPUs: any number of sets of two H100 (`PILOT_GPU_SETS`, default all GPUs of the node in pairs).
- Disk under `PILOT_ROOT`: 72 GB for the model, about 30 GB for the environments, and up to
  72 GB per GPU set for a merged copy of the model while a job runs. Four sets: about 400 GB.
  An evaluation that finds less than 90 GB free waits before it merges
  (`PILOT_MIN_FREE_GB`, `PILOT_DISK_WAIT_MINUTES`).
- Memory: the trainer keeps the model in host memory. About 100 GB per GPU set.
- Network for steps 1 and 2 only (PyPI, GitHub, Hugging Face). The runs are offline.

## Files

| File | Purpose |
|---|---|
| `env.sh` | Directory layout and the process environment of the original jobs. Sourced by everything. |
| `setup_env.sh` | Step 1. |
| `requirements/` | The three lock files and the two files they were resolved from. |
| `sandbox_probe.py`, `tools/bwrap_shim.sh` | How the sandbox runs on this node, and proof that it isolates. |
| `fetch_model.py` | Step 2. |
| `preflight.py`, `run_unit_tests.py` | Step 3. |
| `smoke.sh` | Step 4. |
| `launch.sh` | Step 5: one worker per GPU set. |
| `pilot.py` | The job table, the queue, the workers, `status`, `summary`, `retry`. |
| `evaluate.py` | One evaluation job. `--dry-run` prints its commands. |
| `configs/` | The two training configurations: the originals with `rounds` 28 instead of 10. |
| `data/`, `prepare_bcb_full.py` | The 992 tasks outside BCB-Hard (`bigcode/bigcodebench`, split `v0.1.1`, pinned revision), their manifest and its check. |
| `wandb_sync.py` | Optional mirror of progress and scores to Weights & Biases. |
| `merlin/enter.sh` | Runs a command inside the Merlin training image. |

## Settings

All optional, read from the environment.

| Variable | Default | Meaning |
|---|---|---|
| `PILOT_ROOT` | `/arl/pilot` | Node-local working area. |
| `PILOT_OUT` | `$PILOT_ROOT/out` | Results; shared between nodes when there are several. |
| `PILOT_GPU_SETS` | all GPUs in pairs | For example `"0,1 2,3 4,5 6,7"`. |
| `PILOT_GPUS_PER_RUN` | `2` | GPUs per job. |
| `PILOT_FULL_SHARDS` | `4` | Parts of each 992-task evaluation. Must be the same on every node. |
| `PILOT_PROFILES` | `General Specific` | |
| `PILOT_ROUNDS`, `PILOT_EVAL_ROUNDS`, `PILOT_FULL_ROUND` | `28`, `10 19 28`, `10` | Changing these changes the experiment. |
