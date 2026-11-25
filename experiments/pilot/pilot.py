"""Job table, claim queue and GPU-slot workers of the two pilot experiments.

Experiment 1 (more epochs on BCB-Hard): one training run per prompt profile with the unchanged
runner, 28 updates instead of 10, and an unaided evaluation on the 74 held-out tasks of the
untrained model and of the adapters after 10, 19 and 28 updates.

Experiment 2 (full BigCodeBench): no further training. The untrained model and the adapter
after 10 updates are evaluated on the 992 tasks outside BCB-Hard. With the 74 held-out tasks
of experiment 1 that covers the 1,066 tasks that were never trained on.

A slot is one set of GPUs on one node. Each slot runs one worker, and a worker runs one job at
a time: the first job in the table that nobody has taken and whose adapter exists. Jobs are
taken by creating a directory under PILOT_OUT/queue, so workers on several nodes can share
one table when PILOT_OUT is on shared storage.

    python -m experiments.pilot.pilot plan [--commands]     # the jobs, in order, with their state
    python -m experiments.pilot.pilot worker --slot 0 --gpus 0,1
    python -m experiments.pilot.pilot status                # progress of every job
    python -m experiments.pilot.pilot summary [--json]      # the result tables
    python -m experiments.pilot.pilot retry <job id>        # let a failed job run again

Settings (environment, all optional): PILOT_PROFILES ("General Specific"), PILOT_ROUNDS (28),
PILOT_EVAL_ROUNDS ("10 19 28"), PILOT_FULL_ROUND (10), PILOT_FULL_SHARDS (4).
"""
import argparse
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HEARTBEAT_SECONDS = 60
STALE_SECONDS = 15 * 60
POLL_SECONDS = 60
# The runner derives its server port and its allocation lock from SLURM_JOB_ID. Each slot gets
# its own value, so that several runs can share a node. Ports: runner 27000 + slot, evaluation 21000 + slot.
SLOT_JOB_ID_BASE = 7000
EVAL_PORT_BASE = 21000


@dataclass(frozen=True)
class Job:
    id: str
    kind: str                 # 'train' or 'eval'
    profile: str
    priority: int
    label: str = ''           # 'base', or 'r<N>' for the adapter after N updates
    taskset: str = ''
    shard: tuple = (1, 1)
    needs_round: int = 0      # the training round whose adapter this job evaluates; 0 for none


class Settings:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.out = Path(env.get('PILOT_OUT', ''))
        self.scratch = Path(env.get('PILOT_SCRATCH', ''))
        self.agent_env = env.get('PILOT_AGENT_ENV', '')
        self.profiles = env.get('PILOT_PROFILES', 'General Specific').split()
        self.rounds = int(env.get('PILOT_ROUNDS', '28'))
        self.eval_rounds = sorted({int(value) for value in env.get('PILOT_EVAL_ROUNDS', '10 19 28').split()})
        self.full_round = int(env.get('PILOT_FULL_ROUND', '10'))
        self.full_shards = int(env.get('PILOT_FULL_SHARDS', '4'))
        if any(profile not in ('General', 'Specific') for profile in self.profiles) or not self.profiles:
            raise ValueError('PILOT_PROFILES must list General and/or Specific')
        if any(not 1 <= value <= self.rounds for value in self.eval_rounds + [self.full_round]):
            raise ValueError('Evaluation rounds must lie between 1 and PILOT_ROUNDS')
        if self.full_shards < 1:
            raise ValueError('PILOT_FULL_SHARDS must be positive')

    def config(self, profile):
        path = HERE / 'configs' / f'{profile}_r{self.rounds}.json'
        if not path.exists():
            raise FileNotFoundError(f'No training configuration for {self.rounds} rounds: {path}')
        return path

    def train_out(self, profile):
        return self.out / 'train' / f'{profile}_r{self.rounds}'

    def train_scratch(self, profile):
        return self.scratch / 'train' / f'{profile}_r{self.rounds}'

    def adapter(self, profile, round_count):
        """The adapter after `round_count` updates; the runner numbers its round directories from 0."""
        return self.train_out(profile) / f'round_{round_count - 1:03d}' / 'adapter'

    def adapter_ready(self, profile, round_count):
        return (self.adapter(profile, round_count) / 'opd_update.json').exists()

    def training_complete(self, profile):
        return self.adapter_ready(profile, self.rounds)

    def eval_out(self, job):
        index, count = job.shard
        part = 'all' if count == 1 else f's{index}of{count}'
        return self.out / 'eval' / job.taskset / job.profile / job.label / part

    def eval_scratch(self, job):
        return self.scratch / 'eval' / job.id

    def queue(self, job):
        return self.out / 'queue' / job.id

    def log(self, job):
        return self.out / 'logs' / f'{job.id}.log'


def jobs(settings):
    """Every job of both experiments, in the order in which a free slot takes them."""
    table = []
    for profile in settings.profiles:
        table.append(Job(f'train.{profile}', 'train', profile, 0))
        table.append(Job(f'eval.heldout74.{profile}.base', 'eval', profile, 1, 'base', 'heldout74'))
        for round_count in settings.eval_rounds:
            priority = 2 if round_count == settings.full_round else 5
            table.append(Job(f'eval.heldout74.{profile}.r{round_count}', 'eval', profile, priority,
                             f'r{round_count}', 'heldout74', (1, 1), round_count))
        for label, priority, needs in (('base', 3, 0), (f'r{settings.full_round}', 4, settings.full_round)):
            for index in range(1, settings.full_shards + 1):
                suffix = '' if settings.full_shards == 1 else f'.s{index}of{settings.full_shards}'
                table.append(Job(f'eval.nonhard992.{profile}.{label}{suffix}', 'eval', profile, priority,
                                 label, 'nonhard992', (index, settings.full_shards), needs))
    order = {profile: position for position, profile in enumerate(settings.profiles)}
    table.sort(key=lambda job: (job.priority, job.needs_round, order[job.profile], job.shard))
    if len({job.id for job in table}) != len(table):
        raise ValueError('Duplicate job identifiers')
    return table


def command(job, settings, gpus, slot):
    """The process a job runs. Training is the unchanged runner, as launch_opd.sh starts it."""
    count = len(gpus.split(','))
    if job.kind == 'train':
        return [sys.executable, '-m', 'experiments.bcb_opd.run', '--config', str(settings.config(job.profile)),
                '--out', str(settings.train_out(job.profile)), '--scratch', str(settings.train_scratch(job.profile)),
                '--agent-env', settings.agent_env, '--gpus', str(count)]
    result = [sys.executable, '-m', 'experiments.pilot.evaluate', '--profile', job.profile,
              '--taskset', job.taskset, '--label', job.label, '--shard', f'{job.shard[0]}/{job.shard[1]}',
              '--out', str(settings.eval_out(job)), '--scratch', str(settings.eval_scratch(job)),
              '--agent-env', settings.agent_env, '--gpus', str(count), '--port', str(EVAL_PORT_BASE + slot)]
    if job.needs_round:
        result += ['--adapter', str(settings.adapter(job.profile, job.needs_round))]
    return result


def artifact_complete(job, settings):
    if job.kind == 'train':
        return settings.training_complete(job.profile)
    return (settings.eval_out(job) / 'complete.json').exists()


def heartbeat_age(directory):
    try:
        return time.time() - float((directory / 'heartbeat').read_text().strip())
    except (OSError, ValueError):
        try:
            return time.time() - directory.stat().st_mtime
        except OSError:
            return float('inf')


def state(job, settings, table=None):
    """done, failed, running, stale (its worker stopped reporting), ready, waiting or blocked."""
    directory = settings.queue(job)
    if (directory / 'done.json').exists() or artifact_complete(job, settings):
        return 'done'
    if (directory / 'failed.json').exists():
        return 'failed'
    if directory.exists():
        return 'running' if heartbeat_age(directory) < STALE_SECONDS else 'stale'
    if not job.needs_round or settings.adapter_ready(job.profile, job.needs_round):
        return 'ready'
    producer = next(other for other in (table or jobs(settings))
                    if other.kind == 'train' and other.profile == job.profile)
    # The adapter is missing. It can still appear only while its training run has not ended.
    return 'blocked' if state(producer, settings) in ('failed', 'done') else 'waiting'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def claim(job, settings, slot, gpus):
    directory = settings.queue(job)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError:
        if state(job, settings) != 'stale':
            return False
        try:
            directory.rename(directory.parent / f'.stale.{job.id}.{int(time.time())}.{os.getpid()}')
            directory.mkdir()
        except OSError:
            return False
    (directory / 'heartbeat').write_text(str(time.time()))
    write_json(directory / 'claim.json', {'host': socket.gethostname(), 'slot': slot, 'gpus': gpus,
                                          'pid': os.getpid(), 'started': timestamp()})
    return True


def release(job, settings, reason):
    directory = settings.queue(job)
    try:
        directory.rename(directory.parent / f'.{reason}.{job.id}.{int(time.time())}.{os.getpid()}')
    except OSError:
        pass


def become_subreaper():
    """Orphaned descendants (a server whose runner died) are re-parented to this worker, which stops them."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        pass


def descendants(root):
    parents = {}
    try:
        entries = list(Path('/proc').iterdir())
    except OSError:
        return []
    for entry in entries:
        if entry.name.isdigit():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
            except (OSError, IndexError, ValueError):
                continue
    found, frontier = [], [root]
    while frontier:
        parent = frontier.pop()
        for pid, ppid in parents.items():
            if ppid == parent and pid not in found:
                found.append(pid)
                frontier.append(pid)
    return found


def stop_descendants(grace=45):
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
        pids = descendants(os.getpid())
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + wait
        while time.time() < deadline and descendants(os.getpid()):
            try:
                os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pass
            time.sleep(1)


def gpu_memory_used(gpus):
    """MiB in use per GPU of this slot, or None when nvidia-smi is not available."""
    try:
        output = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', gpus],
                                capture_output=True, text=True, timeout=60, check=True).stdout
        return [int(line) for line in output.split()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def wait_for_free_gpus(gpus, limit_mib=4000, minutes=10):
    deadline = time.time() + minutes * 60
    while True:
        used = gpu_memory_used(gpus)
        if used is None or max(used) < limit_mib:
            return used
        if time.time() >= deadline:
            raise RuntimeError(f'GPUs {gpus} are in use by another process ({used} MiB); not starting a job on them')
        time.sleep(15)


def tail(path, lines=40):
    try:
        return path.read_text(errors='replace').splitlines()[-lines:]
    except OSError:
        return []


def run_job(job, settings, slot, gpus):
    directory, log = settings.queue(job), settings.log(job)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, SLURM_JOB_ID=str(SLOT_JOB_ID_BASE + slot))
    argv = command(job, settings, gpus, slot)
    started = time.time()
    record = {'job': job.id, 'host': socket.gethostname(), 'slot': slot, 'gpus': gpus, 'command': argv}
    stop = threading.Event()

    def beat():
        while not stop.wait(HEARTBEAT_SECONDS):
            try:
                (directory / 'heartbeat').write_text(str(time.time()))
            except OSError:
                pass

    threading.Thread(target=beat, daemon=True).start()
    try:
        wait_for_free_gpus(gpus)
        with log.open('a') as handle:
            handle.write(f"\n===== {timestamp()} {job.id} on {record['host']} slot {slot} gpus {gpus} =====\n"
                         f"{' '.join(argv)}\n")
            handle.flush()
            returncode = subprocess.run(argv, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    except Exception as error:  # the job could not be started; record it instead of losing the slot
        returncode = -1
        record['error'] = f'{type(error).__name__}: {error}'
    finally:
        stop.set()
        stop_descendants()
    record.update(returncode=returncode, seconds=round(time.time() - started, 1), finished=timestamp())
    if job.kind == 'train':
        # The runner ends with its own 74-task evaluation. The reported score after the last update
        # comes from the evaluation job, so a failure in that last step does not fail the training.
        succeeded = settings.training_complete(job.profile)
        record['runner_evaluation'] = ('complete' if (settings.train_out(job.profile) / 'complete.json').exists()
                                       else 'not complete')
        for leftover in ('merged', 'final_merged'):
            shutil.rmtree(settings.train_scratch(job.profile) / leftover, ignore_errors=True)
    else:
        succeeded = returncode == 0 and artifact_complete(job, settings)
    if succeeded:
        write_json(directory / 'done.json', record)
    else:
        write_json(directory / 'failed.json', dict(record, log=str(log), log_tail=tail(log)))
    print(f"PILOT_JOB {'DONE' if succeeded else 'FAILED'} {job.id} rc={returncode} seconds={record['seconds']}", flush=True)
    return succeeded


def worker(settings, slot, gpus):
    become_subreaper()
    table = jobs(settings)
    current = {'job': None}

    def interrupted(signum, _frame):
        stop_descendants()
        if current['job'] is not None:
            release(current['job'], settings, 'interrupted')
        print(f'PILOT_WORKER_INTERRUPTED slot={slot} signal={signum}', flush=True)
        os._exit(130)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print(f'PILOT_WORKER_START host={socket.gethostname()} slot={slot} gpus={gpus} jobs={len(table)} '
          f'out={settings.out}', flush=True)
    last_note = 0
    while True:
        states = {job.id: state(job, settings, table) for job in table}
        taken = None
        for job in table:
            if states[job.id] in ('ready', 'stale') and claim(job, settings, slot, gpus):
                taken = job
                break
        if taken is not None:
            current['job'] = taken
            print(f'PILOT_JOB START {taken.id} slot={slot} gpus={gpus}', flush=True)
            run_job(taken, settings, slot, gpus)
            current['job'] = None
            continue
        waiting = [job_id for job_id, value in states.items() if value == 'waiting']
        if not waiting:
            counts = {value: sum(1 for other in states.values() if other == value) for value in sorted(set(states.values()))}
            print(f'PILOT_WORKER_EXIT slot={slot} {json.dumps(counts)}', flush=True)
            return 0 if all(value in ('done', 'running') for value in states.values()) else 1
        if time.time() - last_note > 600:
            print(f'PILOT_WORKER_WAITING slot={slot} for adapters of {len(waiting)} job(s), e.g. {waiting[0]}', flush=True)
            last_note = time.time()
        time.sleep(POLL_SECONDS)


def training_progress(settings, profile):
    out = settings.train_out(profile)
    rounds, updates = 0, []
    while (settings.adapter(profile, rounds + 1) / 'opd_update.json').exists():
        rounds += 1
        update = json.loads((settings.adapter(profile, rounds) / 'opd_update.json').read_text())
        updates.append({'round': rounds, 'mean_kl': update.get('mean_kl'), 'pairs': update.get('pairs')})
    collecting = out / f'round_{rounds:03d}' / 'episodes.jsonl'
    episodes = sum(1 for line in collecting.read_text().splitlines() if line.strip()) if collecting.exists() else 0
    return {'completed_rounds': rounds, 'of': settings.rounds, 'episodes_in_current_round': episodes, 'updates': updates}


def eval_progress(job, settings):
    out = settings.eval_out(job)
    if (out / 'complete.json').exists():
        complete = json.loads((out / 'complete.json').read_text())
        return {'tasks': complete['identity']['tasks'], 'saved': complete['identity']['tasks'], 'score': complete['score'],
                'context_limit_tasks': complete['context_limit_tasks'],
                'infrastructure_retried': complete['infrastructure_retried'], 'timing': complete['timing'],
                'exit_status_counts': complete['exit_status_counts']}
    episodes = out / 'eval_episodes.jsonl'
    saved = sum(1 for line in episodes.read_text().splitlines() if line.strip()) if episodes.exists() else 0
    tasks = None
    if (out / 'config.json').exists():
        tasks = json.loads((out / 'config.json').read_text())['identity']['tasks']
    return {'tasks': tasks, 'saved': saved}


def status(settings):
    table = jobs(settings)
    report = []
    for job in table:
        entry = {'job': job.id, 'state': state(job, settings, table)}
        directory = settings.queue(job)
        if (directory / 'claim.json').exists():
            claimed = json.loads((directory / 'claim.json').read_text())
            entry.update(host=claimed['host'], slot=claimed['slot'], gpus=claimed['gpus'], started=claimed['started'])
        if (directory / 'failed.json').exists():
            entry['returncode'] = json.loads((directory / 'failed.json').read_text()).get('returncode')
        entry['progress'] = training_progress(settings, job.profile) if job.kind == 'train' else eval_progress(job, settings)
        report.append(entry)
    return report


def summary(settings):
    """Per profile and model: passed / tasks on the held-out hard tasks, the other tasks and both."""
    table = jobs(settings)
    result = {'reported_metric': 'final_solution', 'profiles': {}, 'training': {}}
    for profile in settings.profiles:
        labels = {}
        for job in table:
            if job.kind != 'eval' or job.profile != profile:
                continue
            progress = eval_progress(job, settings)
            cell = labels.setdefault(job.label, {}).setdefault(
                job.taskset, {'passed': 0, 'first_submit_passed': 0, 'tasks': 0, 'shards_done': 0, 'shards': 0,
                              'context_limit_tasks': 0, 'infrastructure_retried': 0})
            cell['shards'] += 1
            if 'score' in progress:
                cell['shards_done'] += 1
                cell['passed'] += progress['score']['final_solution']['passed']
                cell['first_submit_passed'] += progress['score']['first_submit_solution']['passed']
                cell['tasks'] += progress['score']['final_solution']['expected_tasks']
                cell['context_limit_tasks'] += len(progress['context_limit_tasks'])
                cell['infrastructure_retried'] += progress['infrastructure_retried']
        for cells in labels.values():
            for cell in cells.values():
                cell['complete'] = cell['shards_done'] == cell['shards']
                cell['accuracy'] = cell['passed'] / cell['tasks'] if cell['complete'] and cell['tasks'] else None
            if all(name in cells and cells[name]['complete'] for name in ('heldout74', 'nonhard992')):
                passed = cells['heldout74']['passed'] + cells['nonhard992']['passed']
                tasks = cells['heldout74']['tasks'] + cells['nonhard992']['tasks']
                cells['never_trained'] = {'passed': passed, 'tasks': tasks, 'accuracy': passed / tasks, 'complete': True}
        result['profiles'][profile] = labels
        progress = training_progress(settings, profile)
        runner = settings.train_out(profile) / 'grades.json'
        if runner.exists():
            grade = next(entry for entry in json.loads(runner.read_text()) if entry['field'] == 'final_solution')
            progress['runner_evaluation'] = {'passed': grade['passed'], 'tasks': grade['expected_tasks']}
        result['training'][profile] = progress
    return result


def markdown(result):
    def cell(value):
        if not value:
            return 'n/a'
        if not value.get('complete'):
            return f"not finished ({value.get('shards_done', 0)}/{value.get('shards', 0)} parts)"
        return f"{value['passed']}/{value['tasks']} ({100 * value['accuracy']:.2f}%)"

    lines = ['Passed / tasks, graded on the final `solution.py` of each episode. One sample per task.', '',
             '| Prompt | Model | Held-out BCB-Hard (74) | Outside BCB-Hard (992) | Never trained on (1,066) |',
             '|---|---|---|---|---|']
    for profile, labels in result['profiles'].items():
        for label in sorted(labels, key=lambda name: (name != 'base', int(name[1:]) if name != 'base' else 0)):
            cells = labels[label]
            name = 'untrained' if label == 'base' else f'after {label[1:]} updates'
            lines.append(f"| {profile} | {name} | {cell(cells.get('heldout74'))} | {cell(cells.get('nonhard992'))} | "
                         f"{cell(cells.get('never_trained'))} |")
    lines.append('')
    for profile, progress in result['training'].items():
        note = f"{profile}: {progress['completed_rounds']}/{progress['of']} updates"
        if progress['updates']:
            note += f", mean KL of the last update {progress['updates'][-1]['mean_kl']}"
        if 'runner_evaluation' in progress:
            note += (f"; the runner's own evaluation after the last update: "
                     f"{progress['runner_evaluation']['passed']}/{progress['runner_evaluation']['tasks']}")
        lines.append('- ' + note)
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='action', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--commands', action='store_true', help='Also print the command of every job')
    run = commands.add_parser('worker')
    run.add_argument('--slot', type=int, required=True, help='Index of this GPU set on this node, from 0')
    run.add_argument('--gpus', required=True, help='Comma-separated GPU indices of this slot')
    commands.add_parser('status')
    show = commands.add_parser('summary')
    show.add_argument('--json', action='store_true')
    again = commands.add_parser('retry')
    again.add_argument('job')
    args = parser.parse_args(argv)
    settings = Settings()
    if args.action == 'plan':
        table = jobs(settings)
        for job in table:
            print(f'{job.priority}  {state(job, settings, table) if settings.out.parts else "-":8s}  {job.id}')
            if args.commands:
                print('     ' + ' '.join(command(job, settings, '0,1', 0)))
        return 0
    for name in ('PILOT_OUT', 'PILOT_SCRATCH', 'PILOT_AGENT_ENV'):
        if not os.environ.get(name):
            parser.error(f'{name} is not set; source experiments/pilot/env.sh first')
    if args.action == 'worker':
        return worker(settings, args.slot, args.gpus)
    if args.action == 'status':
        print(json.dumps(status(settings), indent=2))
        return 0
    if args.action == 'summary':
        result = summary(settings)
        print(json.dumps(result, indent=2) if args.json else markdown(result))
        return 0
    job = next((entry for entry in jobs(settings) if entry.id == args.job), None)
    if job is None:
        parser.error(f'Unknown job: {args.job}')
    if state(job, settings) not in ('failed', 'stale'):
        parser.error(f'{args.job} is {state(job, settings)}; only a failed or stale job can be retried')
    release(job, settings, 'retried')
    print('PILOT_RETRY', args.job)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
