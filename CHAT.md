# CHAT.md: requests of the pilot experiments

Requests from Chongrui's local session, which writes the code, to the execution agent on the
cluster, which runs it. Newest entry first. This repository is public, so the agent's
acknowledgements and reports are not posted here: they go to the private channel through which
each request is announced, as entries whose titles start with `pilot:`.

The execution agent launches and reports. It does not change code, configurations or prompts: a
defect is reported in that channel with its evidence (the command, the log lines) and is fixed on
the local side. A push is not evidence of a launch, and a launch is not evidence of a result: keep receipt,
gate results and running jobs apart in the reports.

---

## 2026-11-25 22:50Z, Chongrui via Claude Code (local session): REQUEST PILOT-BCB-20261125. Two pilot experiments of on-policy distillation on BigCodeBench: more epochs on BCB-Hard, and the full benchmark

**Who asks, and what this is not.** Chongrui asks for this directly. It is a pilot with the code
of this repository (Qwen3.5-35B-A3B, Mini-SWE agent, reviewer-guided on-policy distillation). It resumes nothing of the 9B/30B work in `agentic_rl`, which stays paused, and no
node, job or evaluation of that work is part of it. The README lists a third pilot, SWE-Gym. It
is not part of this request; it has no code yet.

**Code.** `uyffg/pilot`, branch `main`, code at `3683243` (the launch layer `49de58f` and three fixes:
`e0f1170`, CUDA driver libraries on nodes with a driver newer than the image's compat copy;
`dfed964`, litellm's `mcp` extra in the Mini-SWE environment and an agent round trip in the preflight;
`3683243`, a job taken over from a lost node checks its lock and repairs incomplete episode lines first);
later commits on `main` that touch only documentation (`*.md`) are fine to run from. Check the repository out on the devbox share, so that the image sees it.
`experiments/pilot/README.md` is the runbook: the steps, the files, the settings, the result
layout. The distillation code itself is untouched: every file outside `experiments/pilot/`,
`tests/test_pilot.py`, `README.md`, `CHAT.md` and `AGENTS.md` is byte-identical to the first
commit of this repository (`c08cc69`); `git diff --stat c08cc69 3683243` shows no deleted or changed line in them.

**What runs.** 26 jobs, two H100 each, from one shared job table
(`python -m experiments.pilot.pilot plan --commands` prints them with their commands):

| Jobs | What | Depends on |
|---|---|---|
| 2 | Training, one per prompt profile (General, Specific): the unchanged runner with 28 updates of 8 episodes instead of 10 (three passes over the 74 training tasks). Seed 42. | nothing |
| 2 | Evaluation of the untrained model on the 74 held-out BCB-Hard tasks, once per profile | nothing |
| 6 | Evaluation of the adapters after 10, 19 and 28 updates on the same 74 tasks, per profile | that update |
| 8 | Evaluation of the untrained model on the 992 tasks outside BCB-Hard, per profile, in four parts | nothing |
| 8 | Evaluation of the adapter after 10 updates on the same 992 tasks, per profile, in four parts | update 10 |

Experiment 1 is the first three rows; experiment 2 is the last two, read together with the
74-task evaluations of the same two models (1,066 tasks never trained on). **Two 8 x H100 nodes,
so up to 16 H100 at a time** (eight GPU sets); fewer is fine and only takes longer. With two
nodes `PILOT_OUT` must be one directory on the devbox share, set the same for every step on
both nodes: the evaluations on one node read the adapters trained on the other.

**Before any run.** Each step ends in one line to post in the private channel, with its evidence. On Merlin every
command runs through `experiments/pilot/merlin/enter.sh` (the stack is torch 2.11 for CUDA 13, as
in the `vllm024.dev2` image, and needs that image's forward-compatibility libraries on the
driver-535 nodes). If the GPU sets are on more than one node, set `PILOT_OUT` to one directory on
the devbox share for every step.

1. **ACK** this entry with the commit you checked out, the nodes and GPU sets you plan to use,
   and where `PILOT_OUT` is.
2. **Node.** The image rootfs and the host driver libraries, as `ray_node.sh` with
   `PREP_ONLY=1` prepares them. The 30B model and the R2E data are not needed. Post the free
   disk under `/tmp/arl`: about 100 GB plus 72 GB per GPU set is needed (README, "Requirements
   per node").
3. `bash experiments/pilot/setup_env.sh` → `PILOT_SETUP_OK`. Post `versions.txt` from
   `$PILOT_OUT/env/<host>/` (it has the hashes of the installed package lists and the sandbox
   mode). On every node; the hashes and the sandbox mode must be the same on all of them.
4. `python experiments/pilot/fetch_model.py` → `PILOT_MODEL_OK` (72 GB from Hugging Face, every
   file checked against its published hash). For further nodes you may copy the verified
   directory and run `--verify-only`.
5. `python experiments/pilot/preflight.py --gpus <n>` → `PILOT_PREFLIGHT_OK`. Post the whole
   table, warnings included. On every node.
6. `bash experiments/pilot/smoke.sh` → `PILOT_SMOKE_OK`. Post `smoke_passed.json`. Once, on
   one node. It takes two GPU sets for one to two hours.
7. **Projection.** From the smoke: seconds per update, and seconds per evaluation episode. Post
   the projected GPU-hours for the 26 jobs before launching. No GPU-hour limit is set; my own
   estimate, not measured on this hardware, is 450 to 1,000 H100-hours and one to three days
   on 16 GPUs. The two training runs are the longest chain: 28 updates cannot be parallelized.
8. `bash experiments/pilot/launch.sh` on every node. It refuses to start without steps 5 and 6.

A step that fails stops the sequence: post the failing line and the end of the log, and wait.
The environment has never run on a GPU node. Every package pin was checked against the
published packages, and the CPU-only parts were run locally (README, "Package versions"), but
vLLM with the 35B model, the Triton kernels, the FSDP2 update and Bubblewrap run for the first
time in steps 3, 5 and 6. Expect to report something there.

**Yours to set, the same for every job, and to report.** The nodes and GPU sets; the location
of `PILOT_OUT`; `PILOT_FULL_SHARDS` (four parts per 992-task evaluation by default, the same
value on every node); HDFS copies of `PILOT_OUT`; your GPU holds and keep-warm; whether the
optional `wandb_sync.py --loop 900` runs (project `pilot-opd-bcb`, the entity our joint runs
use).

**Not to change.** Everything else: the code, the two configurations (`rounds` 28 is their only
difference from the originals), the prompts, the exports of `experiments/pilot/env.sh`, the lock
files, the sandbox mode once the probe has chosen it, the seed, the evaluation rules. In
particular the NCCL settings in `env.sh` are the original jobs' and stay, even if another
setting would be faster; say so if it matters and I will decide.

**Rules fixed before any score** (README, "Evaluation rules" and "What differs from the original
runs").

- The reported number is `final_solution`: tasks whose final `solution.py` passes the benchmark
  tests, over all tasks of the evaluation. One sample per task.
- An episode that ends in `ContextWindowExceededError` stays, and its final file is graded.
- An episode that ends in another exception is collected again from an empty workspace, at most
  twice. If a failure remains, that evaluation is not graded and its job fails.
- Seven of the 1,066 tasks have tests that can depend on the node (file permissions, `psutil`,
  the user name): BigCodeBench/459, 460, 779, 812, 21, 377, 1042. They stay in the denominator.
  Report their results separately at the end.

**How the result will be read.** Fixed here so that it is not fitted to the numbers.

- Experiment 1 compares, per profile, the 74-task accuracy of the untrained model and of the
  adapters after 10, 19 and 28 updates. One task is 1.35 points and one sample of 74 tasks has
  a standard error of about 5.5 points, so a difference of a few tasks between two rounds is
  noise. "More epochs help" is supported only if accuracy does not fall from 10 to 19 to 28
  updates in both profiles and the tasks gained from update 10 to update 28 clearly outnumber
  the tasks lost (paired by task). Otherwise the result is reported as a curve, with no claim.
- Experiment 2 compares the untrained model and the adapter after 10 updates on 1,066 tasks,
  paired by task, per profile. The standard error is about 1.4 points, so a difference of three
  points or more is informative, and a smaller one is reported as no clear difference.
- These numbers are not comparable with the original runs' numbers (other hardware, package
  versions and sandbox). The untrained model is evaluated here for that reason.

**Stop rules.**

- The runner stops a training run by itself on a non-finite loss, on a reviewer that does not
  return an assessment in three collection attempts, and on mixed model revisions. Do not work
  around these. Post the end of `$PILOT_OUT/logs/train.<Profile>.log`.
- A failed job is recorded and not repeated. If the cause is infrastructure (a node lost, a
  port, a full disk), remove the cause, run `python -m experiments.pilot.pilot retry <job>` and
  `launch.sh` again: once per job without asking. A second failure of the same job, or any
  failure whose cause is not infrastructure, waits for my answer.
- A training run that has finished updates is never started from scratch. `launch.sh` resumes
  it after its last finished update.
- The other jobs keep running when one fails, except jobs that need the failed training run.

**This repository is public.** Push nothing to it: the reports below go to the private channel,
and logs and result files stay on the cluster.

**Reports.**

- At launch: nodes, GPU sets, driver, sandbox mode, the package-list hashes, the job plan, and
  the projection of step 7.
- Then when update 10 of each profile is done, and about every six hours: the output of
  `python -m experiments.pilot.pilot summary`, the count of jobs per state from `status`, and
  anything that failed or was retried.
- At the end: the summary table; per job the GPU-hours, the number of context-limited and
  retried episodes and the exit statuses (all in `complete.json`); the seven tasks above; the
  mean KL per update of both training runs; where the results are kept (`PILOT_OUT` and its
  HDFS copy). Keep the adapters of updates 10, 19 and 28, every `eval_episodes.jsonl`,
  `grades.json` and `trajectories.tar.gz`, and the training `episodes.jsonl` and `pairs.jsonl`.

**Not authorized.** Another seed, another number of updates or another evaluated update; the
SWE-Gym pilot; more than two nodes (16 GPUs) at a time; any change of the files listed under "Not to
change"; a public upload of rollouts or adapters; deleting results; anything of the paused
9B/30B work.

**Local verification only.** All 53 unit tests pass: the repository's own 32 and the 21 of the
launch layer (in the locked Mini-SWE environment; the two torch tests with the pinned torch,
transformers and peft on a CPU). The evaluation's commands are compared with the runner's
source by test. The task files were built from `bigcode/bigcodebench` (`v0.1.1`, pinned
revision) and checked against the repository's 148 hard tasks. No GPU job has been started by
this entry.
