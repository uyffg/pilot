"""Single batch/resume implementation for collection and worker-only evaluation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
from pathlib import Path
import sys

from .opd.profiles import adapt_task
from .execution import shell
from .io import read_rows, index_rows


def collect(args):
    from .backends import mini as backend
    if args.backend != 'mini-swe':
        raise ValueError('Only mini-swe is supported')
    key = 'task_id' if args.benchmark == 'bcb' else 'question_id'
    tasks = [adapt_task(row, args.benchmark, args.backend, args) for row in index_rows(read_rows(args.tasks), key).values()]
    names = [task.workspace_name for task in tasks]
    if len(set(names)) != len(names):
        raise ValueError('Task IDs collide after workspace filename normalization')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = index_rows(read_rows(output), key) if output.exists() else {}
    args.shell = partial(shell, sandbox=args.sandbox, agent_env=args.agent_env)
    failures = 0
    with output.open('a', encoding='utf8') as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(backend.run, task, args): task.id for task in tasks if task.id not in done}
        for job in as_completed(jobs):
            try:
                result = job.result()
                if result.get('worker_error') or result.get('review_error'):
                    failures += 1
                    print(f"session error {jobs[job]}: {result.get('worker_error')} {result.get('review_error')}", file=sys.stderr)
                result.update(backend=args.backend, evaluation_tests_visible=False)
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
                print('saved', jobs[job], flush=True)
            except Exception as exc:
                failures += 1
                print(f'failed {jobs[job]}: {type(exc).__name__}: {exc}', file=sys.stderr)
    return int(failures > 0)
