"""Unaided evaluation of the base model, or of one training round, on a fixed task list.

It repeats the evaluation at the end of experiments/bcb_opd/run.py, step for step, for a model
and a task list of our choice:

  1. merge the round's LoRA adapter into the base model (`worker_reviewer_bench merge`);
     the base model itself is served as it is;
  2. serve it with experiments/bcb_opd/serve.py (the runner's `Server`, default seed 42);
  3. collect one worker-only episode per task (`worker_reviewer_bench collect --no-review
     --disable-thinking`), in the Bubblewrap sandbox, with the profile's prompts;
  4. check the saved episodes (`worker_reviewer_bench.opd.evaluate.validate_evaluation`);
  5. grade the saved files with the benchmark tests (`worker_reviewer_bench grade`).

The collect and grade commands are the runner's, argument for argument (tests/test_pilot.py
compares them with the runner's source). Nothing here trains or uses a reviewer.

Two rules cover what the runner's own evaluation does not handle. Both are fixed here, before
any score exists, and apply to every model alike. Every task stays in the denominator.

  * An episode that ended in `ContextWindowExceededError` is kept. Its final file is graded as
    it is. This is the rule of the original matched-base evaluation.
  * An episode that ended in any other exception (a request timeout, a connection error), or
    that is missing, is an infrastructure failure. Its row is moved to `infra_failed.jsonl`,
    its workspace is deleted and the task is collected again, at most --infra-retries times.
    If failures remain, nothing is graded and the job ends with an error.

    python -m experiments.pilot.evaluate --profile General --taskset heldout74 --label base \
        --out $PILOT_OUT/eval/heldout74/General/base --scratch $PILOT_SCRATCH/eval/... \
        --agent-env $PILOT_AGENT_ENV --gpus 2 --port 21000 [--adapter <round dir>/adapter]
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / 'experiments' / 'bcb_opd'
DATA = Path(__file__).resolve().parent / 'data'
TASKSETS = ('heldout74', 'nonhard992')
CONTEXT_ERROR = 'ContextWindowExceededError:'
MODEL_REPO_ID = 'Qwen/Qwen3.5-35B-A3B'


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def write_rows(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values), encoding='utf8')
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def parse_shard(text):
    index, _, count = text.partition('/')
    try:
        index, count = int(index), int(count)
    except ValueError:
        raise argparse.ArgumentTypeError('Shard must be i/n') from None
    if not 1 <= index <= count:
        raise argparse.ArgumentTypeError('Shard must be i/n with 1 <= i <= n')
    return index, count


def select(values, shard, limit=None):
    """Shard i of n takes every n-th task, so each shard mixes the whole benchmark."""
    index, count = shard
    chosen = values[index - 1::count]
    return chosen[:limit] if limit else chosen


def task_lists(taskset, shard=(1, 1), limit=None):
    """Public rows (given to the agent) and grading rows (with tests) of one evaluation."""
    if taskset == 'heldout74':
        # As the runner builds its own evaluation: public rows in split order, tests in file order.
        split = json.loads((ASSETS / 'split.json').read_text())
        by_id = {row['task_id']: row for row in read_rows(ASSETS / 'public_tasks.jsonl')}
        public = [by_id[task_id] for task_id in split['eval_ids']]
        grading_source = read_rows(ASSETS / 'grading_tasks.jsonl')
    elif taskset == 'nonhard992':
        public = read_rows(DATA / 'bcb_nonhard_public.jsonl')
        grading_source = read_rows(DATA / 'bcb_nonhard_grading.jsonl')
    else:
        raise ValueError(f'Unknown task set: {taskset}')
    public = select(public, shard, limit)
    chosen = {row['task_id'] for row in public}
    if len(chosen) != len(public) or any('test' in row for row in public):
        raise ValueError('Duplicate tasks, or tests in the public task list')
    grading = [row for row in grading_source if row['task_id'] in chosen]
    if {row['task_id'] for row in grading} != chosen:
        raise ValueError('Grading tasks do not match the public tasks')
    return public, grading


def collect_command(cfg, profile, tasks, episodes, workdir, port, agent_env, python=None):
    """The worker-only collection command of experiments/bcb_opd/run.py, for these paths."""
    python = python or os.environ.get('OPD_FRONTEND_PYTHON', sys.executable)
    return [python, '-m', 'worker_reviewer_bench', 'collect', '--benchmark', 'bcb', '--backend', 'mini-swe',
            '--tasks', tasks, '--output', episodes,
            '--workdir', workdir, '--no-review', '--disable-thinking', '--review-retries', '0',
            '--worker-model', 'openai/opd-worker',
            '--prompt-profile', profile,
            '--base-url', f'http://127.0.0.1:{port}/v1', '--api-key', 'local', '--workers', cfg['workers'],
            '--max-turns', cfg['max_turns'], '--max-tokens', cfg['max_tokens'],
            '--temperature', cfg['temperature'], '--sandbox', 'bwrap', '--agent-env', agent_env,
            '--cost-limit', '0']


def grade_command(cfg, grading, episodes, grades, agent_env):
    """The grading command of experiments/bcb_opd/run.py, for these paths."""
    return [Path(agent_env) / 'bin/python', '-m', 'worker_reviewer_bench', 'grade',
            '--benchmark', 'bcb', '--tasks', grading,
            '--rows', episodes, '--out', grades,
            '--timeout', cfg['eval_timeout'], '--workers', cfg['workers']]


def grading_environment():
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONPATH=str(REPO))
    if os.environ.get('OPD_DEPENDENCY_OVERLAY'):
        env['PYTHONPATH'] = os.environ['OPD_DEPENDENCY_OVERLAY'] + os.pathsep + str(REPO)
    return env


def adapter_digest(adapter):
    digest = hashlib.sha256()
    for name in ['adapter_config.json', 'adapter_model.safetensors']:
        with (Path(adapter) / name).open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 ** 2), b''):
                digest.update(chunk)
    return digest.hexdigest()


def is_context_limited(row):
    return ((row.get('worker_error') or '').startswith(CONTEXT_ERROR)
            and not row.get('submitted') and not row.get('review_history'))


def classify(saved, expected_ids):
    """Sort saved episodes into context-limited ones, infrastructure failures and missing tasks."""
    by_id = {}
    for row in saved:
        if row['task_id'] in by_id:
            raise ValueError(f"Duplicate evaluation episode: {row['task_id']}")
        by_id[row['task_id']] = row
    unexpected = set(by_id) - set(expected_ids)
    if unexpected:
        raise ValueError(f'Episodes of tasks outside this evaluation: {sorted(unexpected)}')
    context = sorted(task_id for task_id, row in by_id.items() if is_context_limited(row))
    infra = sorted(task_id for task_id, row in by_id.items()
                   if row.get('worker_error') and not is_context_limited(row))
    missing = sorted(set(expected_ids) - set(by_id))
    return {'context_limited': context, 'infrastructure': infra, 'missing': missing}


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def wait_for_disk(path, need_gb, wait_minutes):
    deadline = time.time() + wait_minutes * 60
    while True:
        free_gb = shutil.disk_usage(path).free / 1e9
        if free_gb >= need_gb:
            return free_gb
        if time.time() >= deadline:
            raise RuntimeError(f'{path}: {free_gb:.0f} GB free, {need_gb:.0f} GB needed for a merged model')
        print(f'PILOT_DISK_WAIT {free_gb:.0f} GB free of {need_gb:.0f} GB needed at {path}', flush=True)
        time.sleep(60)


def alive(token):
    """A user token is `<host>-<pid>`; merged models live on node disk, so the host is this one."""
    try:
        os.kill(int(token.name.rsplit('-', 1)[1]), 0)
        return True
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


@contextmanager
def merged_model(base, adapter, digest, scratch_root, label, run):
    """One merged copy per adapter and node, shared by the jobs that evaluate it at the same time.

    The copy is as large as the base model (about 72 GB). It is deleted when its last user
    finishes, unless PILOT_KEEP_MERGED=1.
    """
    root = Path(scratch_root) / 'merged'
    root.mkdir(parents=True, exist_ok=True)
    target, users, lock = root / digest[:16], root / (digest[:16] + '.users'), root / (digest[:16] + '.lock')
    token = users / f'{socket.gethostname()}-{os.getpid()}'
    with locked(lock):
        users.mkdir(exist_ok=True)
        token.write_text(label)
        if not (target / '.pilot_merged.json').exists():
            if target.exists():
                shutil.rmtree(target)
            wait_for_disk(root, float(os.environ.get('PILOT_MIN_FREE_GB', '90')),
                          float(os.environ.get('PILOT_DISK_WAIT_MINUTES', '120')))
            # The runner's merge command.
            run([sys.executable, '-m', 'worker_reviewer_bench', 'merge', '--model', base, '--adapter', adapter,
                 '--out', target])
            write_json(target / '.pilot_merged.json', {'adapter': str(adapter), 'adapter_sha256': digest})
    try:
        yield target
    finally:
        with locked(lock):
            token.unlink(missing_ok=True)
            for other in list(users.iterdir()):
                if not alive(other):
                    other.unlink(missing_ok=True)
            if not any(users.iterdir()) and os.environ.get('PILOT_KEEP_MERGED') != '1':
                shutil.rmtree(target, ignore_errors=True)


def clear_unfinished_workspaces(public, done, workdir):
    """A task that is collected (again) starts from an empty workspace, never from leftovers."""
    from worker_reviewer_bench.benchmarks import adapt
    for row in public:
        if row['task_id'] not in done:
            shutil.rmtree(Path(workdir) / adapt(row, 'bcb').workspace_name, ignore_errors=True)


def archive_trajectories(workdir, target):
    """Keep the full Mini-SWE trajectories with the results; the workspaces are on node-local scratch."""
    try:
        temporary = Path(str(target) + '.tmp')
        with tarfile.open(temporary, 'w:gz') as archive:
            for path in sorted(Path(workdir).glob('*/worker_*.traj.json')):
                archive.add(path, arcname=f'{path.parent.name}/{path.name}')
        temporary.replace(target)
    except OSError as error:
        print(f'PILOT_EVAL_TRAJECTORY_ARCHIVE_FAILED {type(error).__name__}: {error}', flush=True)


def identity(args, cfg, public, digest):
    """What defines this evaluation. A rerun with a different identity is refused."""
    blob = ''.join(json.dumps(row, sort_keys=True) for row in public).encode()
    return {'label': args.label, 'prompt_profile': args.profile, 'taskset': args.taskset,
            'shard': list(args.shard), 'limit': args.limit, 'tasks': len(public),
            'tasks_sha256': hashlib.sha256(blob).hexdigest(),
            'base_revision': cfg['base_revision'], 'adapter_sha256': digest,
            'max_turns': cfg['max_turns'], 'max_tokens': cfg['max_tokens'], 'temperature': cfg['temperature'],
            'workers': cfg['workers'], 'eval_timeout': cfg['eval_timeout'], 'serve_seed': 42,
            'reviewer': False, 'thinking': False, 'backend': 'mini-swe',
            'context_policy': 'Grade the unchanged final file; no regeneration or exclusion',
            'infrastructure_policy': f'Collect the task again from an empty workspace, at most {args.infra_retries} times'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', choices=['General', 'Specific'], required=True)
    parser.add_argument('--taskset', choices=TASKSETS, required=True)
    parser.add_argument('--label', required=True, help='base, or r<N> for the adapter after N updates')
    parser.add_argument('--adapter', help='Round adapter directory; omit for the base model')
    parser.add_argument('--config', help='Profile settings; default experiments/bcb_opd/<profile>.json')
    parser.add_argument('--shard', type=parse_shard, default=(1, 1), metavar='i/n')
    parser.add_argument('--limit', type=int, help='First N tasks only (smoke)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--agent-env', required=True)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--port', type=int, default=21000 + os.getpid() % 1000)
    parser.add_argument('--infra-retries', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true', help='Print the identity and the commands; start nothing')
    args = parser.parse_args(argv)
    if (args.label == 'base') != (args.adapter is None):
        parser.error('--label base goes without --adapter; any other label needs --adapter')

    cfg = json.loads(Path(args.config or ASSETS / f'{args.profile}.json').read_text())
    if cfg['prompt_profile'] != args.profile:
        parser.error('--config belongs to the other prompt profile')
    public, grading = task_lists(args.taskset, args.shard, args.limit)
    expected = [row['task_id'] for row in public]
    out, scratch = Path(args.out), Path(args.scratch)
    episodes, workdir = out / 'eval_episodes.jsonl', scratch / 'evaluation'

    if args.dry_run:
        print(json.dumps(identity(args, cfg, public, None if args.adapter is None else '<adapter sha256>'), indent=2))
        for command in (collect_command(cfg, args.profile, out / 'eval_tasks.jsonl', episodes, workdir, args.port,
                                        args.agent_env),
                        grade_command(cfg, out / 'grading_tasks.jsonl', episodes, out / 'grades.json', args.agent_env)):
            print('DRY_RUN', ' '.join(map(str, command)))
        return 0

    # The runner's helpers: same server start, same subprocess handling.
    from experiments.bcb_opd.run import Server, run
    from worker_reviewer_bench.opd.evaluate import validate_evaluation

    base = Path(os.environ['MODEL_PATH'])
    verified = json.loads((base / 'bcb_verified.json').read_text())
    if verified['revision'] != cfg['base_revision'] or verified['repo_id'] != MODEL_REPO_ID:
        raise ValueError('Unexpected base model provenance')
    if args.adapter and not (Path(args.adapter) / 'opd_update.json').exists():
        raise ValueError(f'{args.adapter} is not a finished training round')
    digest = adapter_digest(args.adapter) if args.adapter else None

    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    with locked(out / '.eval.lock'):
        if (out / 'complete.json').exists():
            print('PILOT_EVAL_ALREADY_COMPLETE', out, flush=True)
            return 0
        this = identity(args, cfg, public, digest)
        if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text())['identity'] != this:
            raise ValueError(f'{out} holds a different evaluation')
        write_json(out / 'config.json', {'identity': this, 'adapter': args.adapter, 'gpus': args.gpus,
                                         'host': socket.gethostname()})
        write_rows(out / 'eval_tasks.jsonl', public)
        write_rows(out / 'grading_tasks.jsonl', grading)
        timing = json.loads((out / 'timing.json').read_text()) if (out / 'timing.json').exists() else {}

        if not (out / 'generation_complete.json').exists():
            started = time.time()
            status = None
            for attempt in range(args.infra_retries + 1):
                done = {row['task_id'] for row in read_rows(episodes)} if episodes.exists() else set()
                clear_unfinished_workspaces(public, done, workdir)
                command = collect_command(cfg, args.profile, out / 'eval_tasks.jsonl', episodes, workdir,
                                          args.port, args.agent_env)

                def serve_and_collect(model, attempt=attempt, command=command):
                    with Server(model, args.port, args.gpus, out / f'vllm.attempt{attempt}.log'):
                        # Collection returns 1 for finished episodes that did not submit, as in the runner.
                        run(command, accepted_returncodes=(0, 1))

                if args.adapter:
                    with merged_model(base, args.adapter, digest, scratch.parent, f'{args.label}:{out}', run) as model:
                        serve_and_collect(model)
                else:
                    serve_and_collect(base)
                saved = read_rows(episodes) if episodes.exists() else []
                status = classify(saved, expected)
                if not status['infrastructure'] and not status['missing']:
                    break
                failed = [row for row in saved if row['task_id'] in status['infrastructure']]
                with (out / 'infra_failed.jsonl').open('a', encoding='utf8') as handle:
                    for row in failed:
                        handle.write(json.dumps(dict(row, pilot_attempt=attempt), ensure_ascii=False) + '\n')
                write_rows(episodes, [row for row in saved if row['task_id'] not in status['infrastructure']])
                print('PILOT_EVAL_INFRASTRUCTURE', json.dumps({'attempt': attempt, **status}), flush=True)
            else:
                write_json(out / 'incomplete.json', dict(status, attempts=args.infra_retries + 1))
                print('PILOT_EVAL_INCOMPLETE', out, json.dumps(status), flush=True)
                return 3
            (out / 'incomplete.json').unlink(missing_ok=True)
            saved = read_rows(episodes)
            context = set(status['context_limited'])
            if context:
                write_rows(out / 'validation_subset.jsonl', [row for row in saved if row['task_id'] not in context])
                summary = validate_evaluation(out / 'validation_subset.jsonl', set(expected) - context, 'mini-swe')
                summary.update(tasks=len(saved), episodes_sha256=hashlib.sha256(episodes.read_bytes()).hexdigest())
            else:
                summary = validate_evaluation(episodes, expected, 'mini-swe')
            summary['context_limit_tasks'] = sorted(context)
            summary['infrastructure_retried'] = (len(read_rows(out / 'infra_failed.jsonl'))
                                                 if (out / 'infra_failed.jsonl').exists() else 0)
            archive_trajectories(workdir, out / 'trajectories.tar.gz')
            write_json(out / 'generation_complete.json', summary)
            timing['collection_seconds'] = round(timing.get('collection_seconds', 0) + time.time() - started, 1)
            write_json(out / 'timing.json', timing)

        started = time.time()
        run(grade_command(cfg, out / 'grading_tasks.jsonl', episodes, out / 'grades.json', args.agent_env),
            env=grading_environment())
        timing['grading_seconds'] = round(time.time() - started, 1)
        write_json(out / 'timing.json', timing)
        report = {entry['field']: {key: entry[key] for key in
                                   ['passed', 'expected_tasks', 'accuracy', 'n', 'missing_tasks',
                                    'tasks_with_executed_tests']}
                  for entry in json.loads((out / 'grades.json').read_text())}
        if any(entry['expected_tasks'] != len(expected) or entry['missing_tasks'] for entry in report.values()):
            raise ValueError('Grading did not cover every task of this evaluation')
        generation = json.loads((out / 'generation_complete.json').read_text())
        write_json(out / 'complete.json', {'identity': this, 'score': report, 'timing': timing,
                                           'submitted': generation['submitted'],
                                           'exit_status_counts': generation['exit_status_counts'],
                                           'context_limit_tasks': generation['context_limit_tasks'],
                                           'infrastructure_retried': generation['infrastructure_retried']})
        final = report['final_solution']
        print('PILOT_EVAL_DONE', args.label, args.profile, args.taskset, f'{args.shard[0]}/{args.shard[1]}',
              'final_solution', f"{final['passed']}/{final['expected_tasks']}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
