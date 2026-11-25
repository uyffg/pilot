"""Collect independent student rollouts with side-channel teacher assessments."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import shutil

from ..backends import mini
from ..execution import shell
from ..io import read_rows
from .data import clean_messages
from .reviewer import StateReviewer
from .profiles import adapt_task, mini_prompts


class Recorder:
    def __init__(self, args, task, workspace, reviewer):
        self.args, self.task, self.workspace, self.reviewer = args, task, workspace, reviewer
        self.turn, self.records = 0, []

    def observe(self, history, action, tools=None):
        turn = self.turn
        self.turn += 1
        text = json.dumps(action.get('tool_calls', []), ensure_ascii=False)
        terminal = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in text
        if turn % self.args.review_every != 0 and not terminal:
            return
        # No proposed action is passed to the reviewer. It sees h_k only.
        review, audit = self.reviewer(deepcopy(history), self.workspace)
        self.records.append({'id': f'{self.task.id}:turn{turn}', 'task_id': self.task.id,
            'round': self.args.round, 'student_revision': self.args.student_revision,
            'backend': self.args.backend, 'turn': turn,
            'prompt_profile': getattr(self.args, 'prompt_profile', 'General'),
            'student_messages': clean_messages(history), 'action': clean_messages([action])[0],
            'tools': tools, 'review': review, 'reviewer_messages': audit})


class RecordingModel:
    """Keep mini's native query, parsing, actions, observations and termination."""
    def __init__(self, wrapped, recorder):
        self.wrapped, self.recorder = wrapped, recorder

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def query(self, messages, **kwargs):
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL
        response = self.wrapped.query(messages, **kwargs)
        self.recorder.observe(messages, response, [BASH_TOOL])
        return response


def collect_one(task, args):
    # An unfinished task is restarted from an empty task-owned workspace.
    old_workspace = Path(args.workdir) / task.workspace_name
    ownership = {'task_id': task.id, 'student_revision': args.student_revision,
                 'round': args.round, 'backend': args.backend}
    ownership_root = Path(args.workdir) / '.opd_ownership'
    ownership_root.mkdir(parents=True, exist_ok=True)
    marker = ownership_root / (task.workspace_name + '.json')
    if old_workspace.exists():
        if not marker.exists() or json.loads(marker.read_text()) != ownership:
            raise ValueError(f'Refusing to overwrite a workspace not owned by this OPD episode: {old_workspace}')
        shutil.rmtree(old_workspace)
    workspace = task.prepare(args.workdir)
    # Coordinator metadata lives outside the worker's sandbox.
    marker.write_text(json.dumps(ownership))
    reviewer = StateReviewer(args)
    recorder = Recorder(args, task, workspace, reviewer)
    from copy import copy
    mini_args = copy(args)
    if not mini_args.worker_model.startswith('openai/'):
        mini_args.worker_model = 'openai/' + mini_args.worker_model
    system, instance = mini_prompts(args, task.benchmark, mini.WORKER_SYSTEM, mini.WORKER_INSTANCE)
    worker = mini._make_agent(mini_args, workspace, system, instance,
                              workspace / 'student.traj.json')
    worker.model = RecordingModel(worker.model, recorder)
    native = worker.run(task.text)
    code = (workspace / 'solution.py').read_text() if (workspace / 'solution.py').exists() else ''
    result = {'messages': clean_messages(worker.messages), 'worker_result': native,
              'submitted': native.get('exit_status') == 'Submitted',
              'final_solution': code, 'first_submit_solution': code if native.get('exit_status') == 'Submitted' else ''}
    if not result['submitted'] and native.get('exit_status') not in {'LimitsExceeded', 'RepeatedFormatError'}:
        raise RuntimeError(f'Mini student failed: {native}')
    return task.identify(dict(result, backend=args.backend, opd_records=recorder.records,
                              student_revision=args.student_revision, round=args.round))


def collect(args):
    if args.backend != 'mini-swe':
        raise ValueError('Only mini-swe is supported')
    if args.review_every < 1:
        raise ValueError('review-every must be positive')
    tasks = [adapt_task(row, 'bcb', args.backend, args) for row in read_rows(args.tasks)]
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError('Duplicate tasks')
    args.opd_thinking_off = True
    args.shell = partial(shell, sandbox=args.sandbox, agent_env=args.agent_env)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {r['task_id']: r for r in read_rows(output)} if output.exists() else {}
    if any(r['student_revision'] != args.student_revision for r in done.values()):
        raise ValueError('Refusing to mix student revisions in one rollout batch')
    if not set(done) <= {t.id for t in tasks}:
        raise ValueError('Existing output has unexpected task IDs')
    with output.open('a') as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect_one, t, args): t.id for t in tasks if t.id not in done}
        for future in as_completed(futures):
            result = future.result()  # Fail closed: an API/reviewer failure is not a completed batch.
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            print('OPD_EPISODE', result['task_id'], len(result['opd_records']), flush=True)
    episodes = list(read_rows(output))
    records = [r for e in episodes for r in e['opd_records']]
    if not records:
        raise ValueError('No OPD supervision collected')
    pairs = Path(args.pairs)
    pairs.parent.mkdir(parents=True, exist_ok=True)
    temporary = pairs.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    temporary.replace(pairs)
    return 0
