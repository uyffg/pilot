"""Alternating on-policy collection/full-batch update, followed by unaided evaluation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import shlex
import tempfile
import subprocess
import sys
import time

import requests

from worker_reviewer_bench.opd.evaluate import validate_evaluation

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('OPD_EXPERIMENT_ASSETS', str(CODE))).resolve()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(command, env=None, attempts=1, accepted_returncodes=(0,)):
    for attempt in range(attempts):
        print('RUN', ' '.join(map(str, command)), flush=True)
        result = subprocess.run(list(map(str, command)), env=env)
        if result.returncode in accepted_returncodes:
            return
        if attempt + 1 < attempts:
            print('RETRY_INCOMPLETE_ON_POLICY_COLLECTION', attempt + 1, result.returncode, flush=True)
            time.sleep(2)
        else:
            raise subprocess.CalledProcessError(result.returncode, command)



def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def dump(path, values):
    path.write_text(''.join(json.dumps(r) + '\n' for r in values))


def fingerprint(adapter, config):
    digest = hashlib.sha256(config['base_revision'].encode())
    from worker_reviewer_bench.opd.profiles import PROMPTS
    digest.update((PROMPTS / (config['prompt_profile'] + '.json')).read_bytes())
    if adapter:
        for name in ['adapter_config.json', 'adapter_model.safetensors']:
            with (adapter / name).open('rb') as handle:
                for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
                    digest.update(chunk)
    return digest.hexdigest()


def finish_evaluation(args, cfg, out):
    """Grade every saved episode, including normal model/turn-limit failures."""
    eval_ids = json.loads((ROOT / 'split.json').read_text())['eval_ids']
    summary = validate_evaluation(out / 'eval_episodes.jsonl', eval_ids, args.backend)
    write(out / 'generation_complete.json', summary)
    dump(out / 'grading_tasks.jsonl', [r for r in rows(ROOT / 'grading_tasks.jsonl') if r['task_id'] in eval_ids])
    grading_env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONPATH=str(ROOT.parents[1]))
    if os.environ.get('OPD_DEPENDENCY_OVERLAY'):
        grading_env['PYTHONPATH'] = os.environ['OPD_DEPENDENCY_OVERLAY'] + os.pathsep + str(ROOT.parents[1])
    run([Path(args.agent_env) / 'bin/python', '-m', 'worker_reviewer_bench', 'grade',
        '--benchmark', 'bcb', '--tasks', out / 'grading_tasks.jsonl',
        '--rows', out / 'eval_episodes.jsonl', '--out', out / 'grades.json',
        '--timeout', cfg['eval_timeout'], '--workers', cfg['workers']], env=grading_env)
    write(out / 'complete.json', {'backend': args.backend, 'rounds': cfg['rounds'],
        'adapter': str(out / f"round_{cfg['rounds'] - 1:03d}" / 'adapter'),
        'eval_tasks': len(eval_ids), 'reviewer_at_eval': False})


class Server:
    def __init__(self, model, port, gpus, logfile, seed=42):
        self.model, self.port, self.gpus, self.logfile = model, port, gpus, logfile
        self.seed = seed

    def __enter__(self):
        self.log = self.logfile.open('w')
        self.proc = subprocess.Popen([sys.executable, str(CODE / 'serve.py'), '--model', str(self.model),
            '--port', str(self.port), '--gpus', str(self.gpus), '--seed', str(self.seed)], stdout=self.log,
            stderr=subprocess.STDOUT, start_new_session=True)
        session = requests.Session()
        session.trust_env = False
        try:
            for _ in range(240):
                if self.proc.poll() is not None:
                    raise RuntimeError(f'vLLM failed; see {self.logfile}')
                try:
                    if session.get(f'http://127.0.0.1:{self.port}/health', timeout=5).ok:
                        return self
                except requests.RequestException:
                    pass
                time.sleep(5)
            raise RuntimeError('vLLM readiness timed out')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        except ProcessLookupError:
            pass
        self.log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['mini-swe'], default='mini-swe')
    p.add_argument('--out', required=True)
    p.add_argument('--scratch', required=True)
    p.add_argument('--agent-env', required=True)
    p.add_argument('--gpus', type=int, default=4)
    p.add_argument('--config', required=True)
    p.add_argument('--smoke', action='store_true', help='Two fresh rounds, one train task each, short episodes; no held-out evaluation')
    p.add_argument('--finalize-evaluation', action='store_true',
                   help='Validate and grade existing complete held-out episodes; never generate or train')
    args = p.parse_args()
    if args.finalize_evaluation and args.smoke:
        p.error('--finalize-evaluation cannot be used with --smoke')
    cfg = json.loads(Path(args.config).read_text())
    if args.smoke:
        cfg.update(rounds=2, tasks_per_round=1, max_turns=3,
                   max_tokens=1400,
                   reviewer_turns=1, reviewer_max_tokens=512, review_every=1, workers=1)
    out, scratch = Path(args.out), Path(args.scratch)
    # Serialize all phases sharing one Slurm allocation, even when two
    # orchestration entrypoints advance different backend/output directories.
    allocation_lock_path = Path(tempfile.gettempdir()) / ('opd_allocation_' + os.environ.get('SLURM_JOB_ID', 'local') + '.lock')
    allocation_lock_path.parent.mkdir(parents=True, exist_ok=True)
    allocation_lock = allocation_lock_path.open('a')
    fcntl.flock(allocation_lock, fcntl.LOCK_EX)
    out.mkdir(parents=True, exist_ok=True)
    run_lock = (out / '.run.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX)
    scratch.mkdir(parents=True, exist_ok=True)
    expected_config = dict(cfg, backend=args.backend, smoke=args.smoke)
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != expected_config:
        raise ValueError('Existing experiment configuration differs')
    write(out / 'config.json', expected_config)
    if not args.smoke and (scratch.parent / 'smoke' / 'merged').exists():
        shutil.rmtree(scratch.parent / 'smoke' / 'merged')
    if (out / ('smoke_complete.json' if args.smoke else 'complete.json')).exists():
        print('OPD_PHASE_ALREADY_COMPLETE', out, flush=True)
        return
    if args.finalize_evaluation:
        for iteration in range(cfg['rounds']):
            if not (out / f'round_{iteration:03d}' / 'adapter' / 'opd_update.json').exists():
                raise ValueError('Cannot finalize evaluation before all training updates finish')
        finish_evaluation(args, cfg, out)
        return
    base = Path(os.environ.get('MODEL_PATH', cfg['base_model']))
    verified = json.loads((base / 'bcb_verified.json').read_text())
    if verified['revision'] != cfg['base_revision'] or verified['repo_id'] != 'Qwen/Qwen3.5-35B-A3B':
        raise ValueError('Unexpected base model provenance')
    if not (out / 'environment_preflight.json').exists():
        from worker_reviewer_bench.execution import shell
        code = ('import json,numpy,pandas,scipy,matplotlib,sklearn,nltk,sympy,PIL,requests; '
                'print(json.dumps({"numpy":numpy.__version__,"pandas":pandas.__version__, '
                '"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}))')
        with tempfile.TemporaryDirectory(dir=scratch, prefix='environment-probe-') as td:
            observed = shell('python -c ' + shlex.quote(code), Path(td), 120,
                             sandbox='bwrap', agent_env=args.agent_env)
        if not observed.startswith('exit_code=0'):
            raise RuntimeError('BCB dependencies unavailable inside worker sandbox: ' + observed)
        write(out / 'environment_preflight.json', {'agent_env': args.agent_env, 'output': observed})
    manifest = json.loads((ROOT / 'split.json').read_text())
    train_ids, eval_ids = manifest['train_ids'], manifest['eval_ids']
    if len(train_ids) != 74 or len(eval_ids) != 74 or set(train_ids) & set(eval_ids):
        raise ValueError('Expected disjoint fixed BCB-Hard 74/74 split')
    tasks = {r['task_id']: r for r in rows(ROOT / 'public_tasks.jsonl')}
    if set(tasks) != set(train_ids) | set(eval_ids) or any('test' in r for r in tasks.values()):
        raise ValueError('Public tasks mismatch or contain hidden tests')
    order = list(train_ids)
    random.Random(cfg['seed']).shuffle(order)
    port = 20000 + int(os.environ.get('SLURM_JOB_ID', '42')) % 15000
    python_cli = [sys.executable, '-m', 'worker_reviewer_bench']
    collect_cli = [os.environ.get('OPD_FRONTEND_PYTHON', sys.executable), '-m', 'worker_reviewer_bench']
    adapter = None
    for iteration in range(cfg['rounds']):
        phase = out / f'round_{iteration:03d}'
        phase.mkdir(exist_ok=True)
        next_adapter = phase / 'adapter'
        if (next_adapter / 'opd_update.json').exists():
            update = json.loads((next_adapter / 'opd_update.json').read_text())
            if update['round'] != iteration or update['student_revision'] != fingerprint(adapter, cfg):
                raise ValueError('Invalid resumed update provenance')
            adapter = next_adapter
            continue
        ids = [order[(iteration * cfg['tasks_per_round'] + j) % len(order)]
               for j in range(cfg['tasks_per_round'])]
        dump(phase / 'tasks.jsonl', [tasks[t] for t in ids])
        revision = fingerprint(adapter, cfg)
        write(phase / 'manifest.json', {'round': iteration, 'train_ids': ids,
            'student_revision': revision, 'adapter': str(adapter) if adapter else None})
        pairs = phase / 'pairs.jsonl'
        if not pairs.exists():
            serving_model = base
            if adapter:
                serving_model = scratch / 'merged'
                # Only delete the experiment-owned materialized merge.
                if serving_model.exists():
                    shutil.rmtree(serving_model)
                run(python_cli + ['merge', '--model', base, '--adapter', adapter, '--out', serving_model])
            with Server(serving_model, port, args.gpus, phase / 'vllm.log'):
                run(collect_cli + ['opd-collect', '--backend', args.backend,
                    '--tasks', phase / 'tasks.jsonl', '--output', phase / 'episodes.jsonl', '--pairs', pairs,
                    '--workdir', scratch / f'round_{iteration:03d}' / 'students',
                    '--review-root', scratch / 'reviews', '--round', iteration,
                    '--student-revision', revision, '--base-url', f'http://127.0.0.1:{port}/v1',
                    '--worker-model', 'opd-worker', '--api-key', 'local', '--workers', cfg['workers'],
                    '--prompt-profile', cfg['prompt_profile'],
                    '--max-turns', cfg['max_turns'], '--max-tokens', cfg['max_tokens'],
                    '--temperature', cfg['temperature'], '--review-every', cfg['review_every'],
                    '--reviewer-turns', cfg['reviewer_turns'], '--reviewer-max-tokens', cfg['reviewer_max_tokens'],
                    '--command-timeout', cfg['command_timeout'], '--sandbox', 'bwrap', '--agent-env', args.agent_env], attempts=3)
        # Generation processes have exited before training acquires the GPUs.
        env = dict(os.environ, MODEL_PATH=str(base))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={args.gpus}', '--module', 'worker_reviewer_bench.opd.train',
            '--pairs', pairs, '--out', next_adapter, '--lr', cfg['lr'], '--rank', cfg['rank'],
            '--max-len', cfg['max_len'], '--seed', cfg['seed'] + iteration]
        if adapter:
            command += ['--adapter', adapter]
        run(command, env=env)
        adapter = next_adapter
        write(out / 'progress.json', {'completed_rounds': iteration + 1, 'adapter': str(adapter)})
    if args.smoke:
        if (scratch / 'merged').exists():
            shutil.rmtree(scratch / 'merged')
        write(out / 'smoke_complete.json', {'rounds': cfg['rounds'], 'backend': args.backend, 'adapter': str(adapter)})
        return
    if (scratch / 'merged').exists():
        shutil.rmtree(scratch / 'merged')
    serving_model = scratch / 'final_merged'
    if not (out / 'generation_complete.json').exists():
        if serving_model.exists():
            shutil.rmtree(serving_model)
        run(python_cli + ['merge', '--model', base, '--adapter', adapter, '--out', serving_model])
        dump(out / 'eval_tasks.jsonl', [tasks[t] for t in eval_ids])
        with Server(serving_model, port, args.gpus, out / 'eval_vllm.log'):
            # Match each backend to itself. Neither reviewer nor private feedback
            # is constructed on the held-out evaluation path.
            run(collect_cli + ['collect', '--benchmark', 'bcb', '--backend', args.backend,
                '--tasks', out / 'eval_tasks.jsonl', '--output', out / 'eval_episodes.jsonl',
                '--workdir', scratch / 'evaluation', '--no-review', '--disable-thinking', '--review-retries', '0',
                '--worker-model', 'openai/opd-worker',
                '--prompt-profile', cfg['prompt_profile'],
                '--base-url', f'http://127.0.0.1:{port}/v1', '--api-key', 'local', '--workers', cfg['workers'],
                '--max-turns', cfg['max_turns'], '--max-tokens', cfg['max_tokens'],
                '--temperature', cfg['temperature'], '--sandbox', 'bwrap', '--agent-env', args.agent_env,
                '--cost-limit', '0'], accepted_returncodes=(0, 1))
    # Collection also returns 1 for completed episodes that fail to submit.
    # Validation still rejects missing rows and API/environment failures.
    finish_evaluation(args, cfg, out)


if __name__ == '__main__':
    main()
