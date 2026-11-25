# Pilot Experiments: Mini-SWE On-Policy Distillation

This repository is for pilot experiments. It holds only the code needed for the
three experiments listed below.

Train Qwen3.5-35B-A3B on its own Mini-SWE rollouts using private reviewer guidance.
Teacher and student share weights; only student action tokens receive KL loss.
Evaluation runs without a reviewer.

| Prompt | Focus |
| --- | --- |
| [General](worker_reviewer_bench/opd/prompts/General.json) | The next useful check, repair, or submission. |
| [Specific](worker_reviewer_bench/opd/prompts/Specific.json) | Evidence, assumptions, failure diagnosis, and repair verification. |

Specific also uses benchmark-specific worker instructions.

## Run

```bash
pip install -e .
# Install train/serve extras in compatible CUDA environments.
export MODEL_PATH=/models/Qwen3.5-35B-A3B
./launch_opd.sh General --agent-env /path/to/agent-env --gpus 4
```

Use `Specific` for the other prompt; add `--smoke` for a short check.
Output and scratch directories are separated by prompt and smoke/full mode.
Override them with `--out` and `--scratch`. Requires CUDA, FLA/Liger/FSDP2, vLLM,
Bubblewrap, and benchmark dependencies in the agent environment.
Set `OPD_FRONTEND_PYTHON` if Mini-SWE uses a separate environment.

Training: 74 BCB-Hard tasks, 10 updates, 8 episodes/update, LoRA rank 16.
The runner then evaluates 74 held-out tasks.

## Pilot experiments

1. Big Code Bench: Add more epochs (e.g., 3).
   Train for more epochs to assess whether additional passes improve performance.
2. Full Big Code Bench.
   Evaluate the approach across the full benchmark to measure broader coding performance.
3. SWE-Gym.
   Explore the approach on SWE-Gym tasks to assess its applicability to software engineering.
