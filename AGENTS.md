# Cluster experiment instructions

## Request, 2026-11-25 (Chongrui): PILOT-BCB-20261125, two pilot experiments on BigCodeBench

Chongrui asks directly for two pilot experiments with the on-policy distillation code of this
repository (Qwen3.5-35B-A3B, Mini-SWE agent):

1. **More epochs on BCB-Hard.** One training run per prompt profile (General, Specific) with the
   unchanged runner, 28 updates instead of 10, seed 42. Unaided evaluations of the untrained
   model and of the adapters after 10, 19 and 28 updates on the 74 held-out tasks.
2. **Full BigCodeBench.** No further training. The untrained model and the adapter after 10
   updates on the 992 tasks outside BCB-Hard; with the 74 held-out tasks, the 1,066 tasks that
   were never trained on.

26 jobs, two H100 each, on **two 8 x H100 nodes (at most 16 H100 at a time)**. No GPU-hour limit; the projection from
the smoke is posted before the launch, as information. Code: `main` at `49de58f`. The runbook is
[experiments/pilot/README.md](experiments/pilot/README.md). The gates before any run, what may
and may not be set on the cluster, the rules fixed before any score, the stop rules and the
reports are in the 2026-11-25 entry of [CHAT.md](CHAT.md).

**Roles.** The execution agent launches and reports. It does not change code, configurations,
prompts, lock files or the exports of `experiments/pilot/env.sh`. A defect is reported in
`CHAT.md` with its evidence and fixed on the local side.

**Order.** ACK, node preparation, `setup_env.sh`, `fetch_model.py`, `preflight.py`, `smoke.sh`,
the projection, then `launch.sh`. A failed step stops the sequence until it is answered. A
pushed commit, a passed preflight and a running job are three different things; report them
apart.

**Not authorized.** Another seed, another number of updates or another evaluated update; the
SWE-Gym pilot (the third item of the README, which has no code yet); more than two nodes (16 GPUs) at a
time; a public upload of rollouts or adapters; deleting results; anything of the paused 9B/30B
work in `agentic_rl`, which this request does not resume.
