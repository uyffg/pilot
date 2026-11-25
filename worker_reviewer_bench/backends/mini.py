from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .mini_environment import make_environment


WORKER_SYSTEM = """You are the worker solving one benchmark programming problem.
Work only in the current task directory. Read the complete problem from the task
prompt, create an executable Python 3 file named solution.py, run public samples
and additional tests, and fix any failures. Do not read private tests or other
task directories. When the solution is ready, submit it with one command whose
first line is exactly COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. Do not run commands
after that submission command."""

WORKER_INSTANCE = """Solve this benchmark problem:

{{task}}

Write only the required solution.py in the current directory. If starter code is
provided, follow its required interface. Verify the actual file before submitting."""

REVIEWER_SYSTEM = """You are an independent reviewer for a benchmark solution.
The current directory contains solution.py and problem.txt. Inspect the problem
and the actual solution, run public examples and independently chosen edge cases,
and check complexity and input/output details. Do not modify solution.py, do not
read private tests or other task directories, and do not trust the worker's claims.
Write a JSON object to review.json with exactly these fields:
  decision: either \"pass\" or \"intervene\"
  reason: concise evidence
  check: one actionable check (empty string on pass)
  hint: short general guidance (empty string on pass)
After review.json is complete, submit with one command whose first line is exactly
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. Do not run commands after submitting."""

REVIEWER_INSTANCE = """Review the current solution for this problem:

{{task}}

Keep solution.py unchanged and write review.json before submitting."""


def _mini_imports():
    try:
        from minisweagent.agents import get_agent
        from minisweagent.models import get_model
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "mini-swe-agent is required; install it with `python -m pip install mini-swe-agent`"
        ) from exc
    return get_agent, get_model


def _make_agent(args: argparse.Namespace, cwd: Path, system: str, instance: str, output: Path, role="worker"):
    get_agent, get_model = _mini_imports()
    protocol_compat = role == "worker" and getattr(args, "mini_protocol_compat", False)
    model_kwargs: dict[str, Any] = {"drop_params": True}
    # mini-swe-agent 2.x routes through LiteLLM.  The endpoint and key must be
    # passed as LiteLLM model kwargs; setting only OPENAI_* environment vars is
    # insufficient when the user's global mini-swe-agent config is loaded.
    if args.base_url:
        model_kwargs["api_base"] = args.base_url
    if args.api_key:
        model_kwargs["api_key"] = args.api_key
    temperature = args.reviewer_temperature if role == "reviewer" else args.temperature
    if temperature is not None:
        model_kwargs["temperature"] = temperature
    model_kwargs["max_tokens"] = args.reviewer_max_tokens if role == "reviewer" else args.max_tokens
    if (getattr(args, 'opd_thinking_off', False) or getattr(args, 'disable_thinking', False)):
        model_kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
    model_config = {
            "model_name": (args.reviewer_model or args.worker_model) if role == "reviewer" else args.worker_model,
            "model_class": args.model_class,
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        }
    if protocol_compat:
        from .mini_protocol import FORMAT_ERROR_TEMPLATE, PROTOCOL_APPENDIX, ProtocolFeedbackModel
        model_config["format_error_template"] = FORMAT_ERROR_TEMPLATE
        if getattr(args, "mini_request_timeout", None) is not None:
            model_kwargs["timeout"] = args.mini_request_timeout
        system += PROTOCOL_APPENDIX
    model = get_model(config=model_config)
    if protocol_compat:
        model = ProtocolFeedbackModel(model)
    env = make_environment(args, cwd)
    return get_agent(
        model,
        env,
        {
            "agent_class": "default",
            "system_template": system,
            "instance_template": instance,
            "step_limit": args.reviewer_turns if role == "reviewer" else args.max_turns,
            "cost_limit": args.cost_limit,
            "output_path": output,
        },
        default_type="default",
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def run(task, args):
    from ..opd.profiles import mini_prompts
    root = task.prepare(args.workdir)
    task_text = task.text
    worker_result: dict[str, Any] = {}
    worker_error = ""
    review_result: dict[str, Any] | None = None
    review_error = ""
    review_history: list[dict[str, Any]] = []
    worker_task = task_text
    first_solution = ""
    first_captured = False
    solution = ""
    attempts = []
    for attempt in range(args.review_retries + 1):
        worker_result, worker_error = {}, ""
        system, instance = mini_prompts(args, task.benchmark, WORKER_SYSTEM, WORKER_INSTANCE)
        worker = _make_agent(args, root, system, instance, root / f"worker_{attempt}.traj.json")
        try:
            worker_result = worker.run(worker_task)
        except Exception as exc:  # keep one failed task from stopping the batch
            worker_error = f"{type(exc).__name__}: {exc}"
        attempts.append({"result": worker_result, "error": worker_error,
                         "trajectory": str(root / f"worker_{attempt}.traj.json")})
        solution = (root / "solution.py").read_text(encoding="utf-8") if (root / "solution.py").exists() else ""
        if not first_captured and worker_result.get("exit_status") == "Submitted":
            first_solution = solution
            first_captured = True
        if not solution:
            review_error = "worker did not create solution.py"
            break
        if worker_error or worker_result.get("exit_status") != "Submitted":
            review_error = "Worker session did not submit successfully"
            break
        if args.no_review:
            review_result = {"decision": "pass", "reason": "Review disabled", "check": "", "hint": ""}
            break
        review_prompt = (
            f"{task_text}\n\nWorker trajectory summary (context only; test the file yourself):\n"
            f"{json.dumps(worker_result, ensure_ascii=False)}"
        )
        (root / "review.json").unlink(missing_ok=True)
        reviewer = _make_agent(args, root, REVIEWER_SYSTEM, REVIEWER_INSTANCE, root / f"reviewer_{attempt}.traj.json", role="reviewer")
        try:
            reviewer_result = reviewer.run(review_prompt)
            if reviewer_result.get("exit_status") != "Submitted":
                review_error = "Reviewer session did not submit successfully"
                break
            review_result = _read_json(root / "review.json")
            if review_result is None or review_result.get("decision") not in {"pass", "intervene"}:
                review_error = "reviewer did not produce valid review.json"
                break
            review_history.append(review_result)
        except Exception as exc:
            review_error = f"{type(exc).__name__}: {exc}"
            break
        if review_result.get("decision") == "pass":
            break
        if attempt < args.review_retries:
            worker_task = (
                f"{task_text}\n\nReviewer intervention from the previous attempt:\n"
                f"reason: {review_result.get('reason', '')}\n"
                f"required check: {review_result.get('check', '')}\n"
                f"hint: {review_result.get('hint', '')}\n"
                "Inspect the current solution.py, address this concern, retest, and submit again."
            )

    decision = (review_result or {}).get("decision")
    submitted = bool(solution) and decision == "pass"
    return task.identify({
        "submitted": submitted,
        "worker_result": worker_result,
        "worker_attempts": attempts,
        "worker_error": worker_error,
        "worker_prompt_messages": [{k: v for k, v in m.items() if k in {'role', 'content'}}
                                   for m in getattr(worker, 'messages', [])[:2]],
        "review": review_result,
        "review_history": review_history,
        "review_error": review_error,
        "solution": solution,
        "first_submit_solution": first_solution,
        "first_submit_captured": first_captured,
        "final_solution": solution,
        "workspace": str(root),
    })
