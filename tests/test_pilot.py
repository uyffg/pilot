"""Checks of the pilot launch layer (experiments/pilot) that need no GPU and no model."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.pilot import evaluate, pilot  # noqa: E402

RUNNER = (REPO / 'experiments' / 'bcb_opd' / 'run.py').read_text()
CFG = json.loads((REPO / 'experiments' / 'bcb_opd' / 'General.json').read_text())


def flags(tokens):
    return [str(token) for token in tokens if str(token).startswith('--')]


def runner_flags(start, end):
    """The option names of one command of the runner, in order, read from its source."""
    segment = RUNNER[RUNNER.index(start):]
    segment = segment[:segment.index(end) + len(end)]
    return re.findall(r"'(--[a-z-]+)'", segment)


def episode(task_id, **changes):
    row = {'task_id': task_id, 'backend': 'mini-swe', 'evaluation_tests_visible': False, 'submitted': True,
           'worker_result': {'exit_status': 'Submitted'}, 'worker_error': '', 'review_error': '', 'review_history': [],
           'review': {'decision': 'pass', 'reason': 'Review disabled', 'check': '', 'hint': ''},
           'final_solution': 'def task_func():\n    return 1\n', 'first_submit_solution': 'def task_func():\n    return 1\n',
           'first_submit_captured': True}
    row.update(changes)
    return row


class Fixed(unittest.TestCase):
    def test_configurations_differ_only_in_rounds(self):
        for profile in ('General', 'Specific'):
            original = json.loads((REPO / 'experiments' / 'bcb_opd' / f'{profile}.json').read_text())
            ours = json.loads((REPO / 'experiments' / 'pilot' / 'configs' / f'{profile}_r28.json').read_text())
            self.assertEqual({key for key in original if original[key] != ours[key]}, {'rounds'})
            self.assertEqual(set(original), set(ours))
            self.assertEqual((original['rounds'], ours['rounds']), (10, 28))
            # 28 rounds of 8 tasks pass the 74 training tasks three times (222) plus two tasks.
            self.assertEqual(ours['rounds'] * ours['tasks_per_round'], 3 * 74 + 2)

    def test_task_files_match_their_manifest(self):
        result = subprocess.run([sys.executable, str(REPO / 'experiments' / 'pilot' / 'prepare_bcb_full.py'), '--check'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('BCB_FULL_CHECK_OK', result.stdout)

    def test_heldout_tasks_are_the_runners(self):
        split = json.loads((REPO / 'experiments' / 'bcb_opd' / 'split.json').read_text())
        public, grading = evaluate.task_lists('heldout74')
        self.assertEqual([row['task_id'] for row in public], split['eval_ids'])
        self.assertEqual({row['task_id'] for row in grading}, set(split['eval_ids']))
        self.assertFalse(set(split['train_ids']) & {row['task_id'] for row in public})
        self.assertTrue(all('test' in row for row in grading))

    def test_shards_partition_the_other_tasks(self):
        whole, _ = evaluate.task_lists('nonhard992')
        seen = []
        for index in range(1, 5):
            public, grading = evaluate.task_lists('nonhard992', (index, 4))
            self.assertEqual(len(public), 248)
            self.assertEqual({row['task_id'] for row in public}, {row['task_id'] for row in grading})
            seen += [row['task_id'] for row in public]
        self.assertEqual(sorted(seen), sorted(row['task_id'] for row in whole))
        self.assertEqual(len(set(seen)), 992)
        train = set(json.loads((REPO / 'experiments' / 'bcb_opd' / 'split.json').read_text())['train_ids'])
        self.assertFalse(train & set(seen))
        self.assertEqual(len(evaluate.task_lists('heldout74', limit=2)[0]), 2)

    def test_shard_argument(self):
        self.assertEqual(evaluate.parse_shard('2/4'), (2, 4))
        for bad in ('0/4', '5/4', 'x/4', '3'):
            with self.assertRaises(Exception):
                evaluate.parse_shard(bad)


class Commands(unittest.TestCase):
    """The evaluation must issue the runner's commands. These fail if either side drifts."""

    def test_collect_command_has_the_runners_options(self):
        ours = evaluate.collect_command(CFG, 'General', 'T', 'E', 'W', 21000, '/agent', python='python')
        self.assertEqual(flags(ours), runner_flags("['collect', '--benchmark', 'bcb'", "'--cost-limit', '0']"))
        text = ' '.join(map(str, ours))
        for literal in ('collect --benchmark bcb --backend mini-swe', '--no-review --disable-thinking --review-retries 0',
                        '--worker-model openai/opd-worker', '--prompt-profile General',
                        '--base-url http://127.0.0.1:21000/v1 --api-key local', '--workers 4',
                        '--max-turns 160 --max-tokens 1400 --temperature 0.7', '--sandbox bwrap --agent-env /agent',
                        '--cost-limit 0'):
            self.assertIn(literal, text)

    def test_grade_command_has_the_runners_options(self):
        ours = evaluate.grade_command(CFG, 'G', 'E', 'O', '/agent')
        self.assertEqual(flags(ours), runner_flags("'-m', 'worker_reviewer_bench', 'grade'", "cfg['workers']]"))
        self.assertEqual(' '.join(map(str, ours)),
                         '/agent/bin/python -m worker_reviewer_bench grade --benchmark bcb --tasks G --rows E --out O '
                         '--timeout 240 --workers 4')

    def test_training_command_is_the_launchers(self):
        settings = pilot.Settings({'PILOT_OUT': '/out', 'PILOT_SCRATCH': '/scratch', 'PILOT_AGENT_ENV': '/agent'})
        job = next(job for job in pilot.jobs(settings) if job.id == 'train.Specific')
        command = pilot.command(job, settings, '2,3', 1)
        self.assertEqual(command[1:3], ['-m', 'experiments.bcb_opd.run'])
        launcher = (REPO / 'launch_opd.sh').read_text()
        self.assertIn('-m experiments.bcb_opd.run', launcher)
        for option in ('--config', '--out', '--scratch', '--agent-env'):
            self.assertIn(option, launcher)
            self.assertIn(option, command)
        self.assertEqual(command[command.index('--gpus') + 1], '2')
        self.assertTrue(command[command.index('--config') + 1].endswith('experiments/pilot/configs/Specific_r28.json'))
        self.assertEqual(command[command.index('--out') + 1], '/out/train/Specific_r28')

    def test_merge_command_has_the_runners_options(self):
        self.assertEqual(runner_flags("['merge', '--model'", "'--out', serving_model]")[:3], ['--model', '--adapter', '--out'])
        source = (REPO / 'experiments' / 'pilot' / 'evaluate.py').read_text()
        self.assertIn("'worker_reviewer_bench', 'merge', '--model', base, '--adapter', adapter,", source)


class Policy(unittest.TestCase):
    def test_episode_classes(self):
        saved = [episode('a'), episode('b', worker_error='ContextWindowExceededError: maximum context length is 131072', submitted=False),
                 episode('c', worker_error='Timeout: request timed out', submitted=False),
                 episode('d', worker_result={'exit_status': 'LimitsExceeded'}, submitted=False)]
        status = evaluate.classify(saved, ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(status, {'context_limited': ['b'], 'infrastructure': ['c'], 'missing': ['e']})
        with self.assertRaises(ValueError):
            evaluate.classify(saved + [episode('a')], ['a', 'b', 'c', 'd'])
        with self.assertRaises(ValueError):
            evaluate.classify(saved, ['a'])


class FakeServer:
    started = []

    def __init__(self, model, port, gpus, logfile, seed=42):
        self.model = model

    def __enter__(self):
        FakeServer.started.append(str(self.model))
        return self

    def __exit__(self, *args):
        return False


class EndToEnd(unittest.TestCase):
    """evaluate.main with the server, the collection and the grading replaced by stand-ins."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        model = self.root / 'model'
        model.mkdir()
        (model / 'bcb_verified.json').write_text(json.dumps({'repo_id': 'Qwen/Qwen3.5-35B-A3B', 'revision': CFG['base_revision']}))
        self.adapter = self.root / 'round_009' / 'adapter'
        self.adapter.mkdir(parents=True)
        for name in ('adapter_config.json', 'adapter_model.safetensors', 'opd_update.json'):
            (self.adapter / name).write_text(name)
        self.environment = patch.dict(os.environ, {'MODEL_PATH': str(model), 'PILOT_MIN_FREE_GB': '0'})
        self.environment.start()
        self.collections = 0
        self.failing = {}
        FakeServer.started = []

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def fake_run(self, command, env=None, attempts=1, accepted_returncodes=(0,)):
        command = list(map(str, command))
        value = lambda name: command[command.index(name) + 1]  # noqa: E731
        if 'merge' in command:
            Path(value('--out')).mkdir(parents=True)
            (Path(value('--out')) / 'model.safetensors').write_text('merged')
        elif 'collect' in command:
            self.collections += 1
            tasks = evaluate.read_rows(value('--tasks'))
            output = Path(value('--output'))
            done = {row['task_id'] for row in evaluate.read_rows(output)} if output.exists() else set()
            with output.open('a') as handle:
                for row in tasks:
                    if row['task_id'] in done:
                        continue
                    workspace = Path(value('--workdir')) / row['task_id'].replace('/', '_')
                    self.assertFalse(workspace.exists(), 'a task was collected into a used workspace')
                    workspace.mkdir(parents=True)
                    (workspace / 'worker_0.traj.json').write_text('{}')
                    changes = self.failing.get(row['task_id'], [])
                    handle.write(json.dumps(episode(row['task_id'], **(changes.pop(0) if changes else {}))) + '\n')
        elif 'grade' in command:
            episodes = evaluate.read_rows(value('--rows'))
            expected = [row['task_id'] for row in evaluate.read_rows(value('--tasks'))]
            Path(value('--out')).write_text(json.dumps([
                {'field': field, 'n': len(episodes), 'passed': 1, 'accuracy': 1 / len(expected), 'expected_tasks': len(expected),
                 'missing_tasks': sorted(set(expected) - {row['task_id'] for row in episodes}), 'tasks_with_executed_tests': 1,
                 'results': {}} for field in ('first_submit_solution', 'final_solution')]))
        else:
            raise AssertionError(command)

    def evaluate(self, *extra):
        argv = ['--profile', 'General', '--taskset', 'heldout74', '--limit', '3', '--out', str(self.root / 'out'),
                '--scratch', str(self.root / 'scratch' / 'job'), '--agent-env', '/agent', '--port', '21000', *extra]
        with patch('experiments.bcb_opd.run.Server', FakeServer), patch('experiments.bcb_opd.run.run', self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            return evaluate.main(argv)

    def test_base_model_is_served_unmerged(self):
        self.assertEqual(self.evaluate('--label', 'base'), 0)
        complete = json.loads((self.root / 'out' / 'complete.json').read_text())
        self.assertEqual(complete['score']['final_solution']['expected_tasks'], 3)
        self.assertEqual(complete['identity']['adapter_sha256'], None)
        self.assertEqual(FakeServer.started, [str(self.root / 'model')])
        self.assertTrue((self.root / 'out' / 'trajectories.tar.gz').exists())
        self.assertEqual(self.evaluate('--label', 'base'), 0)
        self.assertEqual(self.collections, 1, 'a finished evaluation was collected again')

    def test_adapter_is_merged_once_and_the_copy_removed(self):
        self.assertEqual(self.evaluate('--label', 'r10', '--adapter', str(self.adapter)), 0)
        merged = self.root / 'scratch' / 'merged'
        self.assertEqual(len(FakeServer.started), 1)
        self.assertTrue(FakeServer.started[0].startswith(str(merged)))
        self.assertFalse(Path(FakeServer.started[0]).exists(), 'the merged copy was left on disk')
        complete = json.loads((self.root / 'out' / 'complete.json').read_text())
        self.assertEqual(complete['identity']['adapter_sha256'], evaluate.adapter_digest(self.adapter))

    def test_infrastructure_failure_is_collected_again_and_context_limit_is_kept(self):
        first, second, third = (row['task_id'] for row in evaluate.task_lists('heldout74', limit=3)[0])
        self.failing = {first: [{'worker_error': 'APIConnectionError: connection reset', 'submitted': False}],
                        second: [{'worker_error': 'ContextWindowExceededError: maximum context length is 131072',
                                  'submitted': False, 'first_submit_captured': False, 'first_submit_solution': '',
                                  'review': None, 'review_error': 'Worker session did not submit successfully'}]}
        self.assertEqual(self.evaluate('--label', 'base'), 0)
        self.assertEqual(self.collections, 2)
        complete = json.loads((self.root / 'out' / 'complete.json').read_text())
        self.assertEqual(complete['context_limit_tasks'], [second])
        self.assertEqual(complete['infrastructure_retried'], 1)
        saved = {row['task_id']: row for row in evaluate.read_rows(self.root / 'out' / 'eval_episodes.jsonl')}
        self.assertEqual(set(saved), {first, second, third})
        self.assertFalse(saved[first]['worker_error'])
        self.assertEqual(len(evaluate.read_rows(self.root / 'out' / 'infra_failed.jsonl')), 1)

    def test_persistent_failure_is_not_graded(self):
        first = evaluate.task_lists('heldout74', limit=3)[0][0]['task_id']
        self.failing = {first: [{'worker_error': 'Timeout: request timed out', 'submitted': False}] * 5}
        self.assertEqual(self.evaluate('--label', 'base', '--infra-retries', '1'), 3)
        self.assertEqual(self.collections, 2)
        self.assertTrue((self.root / 'out' / 'incomplete.json').exists())
        self.assertFalse((self.root / 'out' / 'grades.json').exists())
        self.assertFalse((self.root / 'out' / 'complete.json').exists())

    def test_another_evaluation_in_the_same_directory_is_refused(self):
        self.assertEqual(self.evaluate('--label', 'base'), 0)
        (self.root / 'out' / 'complete.json').unlink()
        with self.assertRaises(ValueError):
            self.evaluate('--label', 'r10', '--adapter', str(self.adapter))


class Queue(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = pilot.Settings({'PILOT_OUT': str(root / 'out'), 'PILOT_SCRATCH': str(root / 'scratch'),
                                        'PILOT_AGENT_ENV': '/agent'})
        self.table = pilot.jobs(self.settings)
        self.by_id = {job.id: job for job in self.table}

    def tearDown(self):
        self.directory.cleanup()

    def finish_round(self, profile, count):
        adapter = self.settings.adapter(profile, count)
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / 'opd_update.json').write_text(json.dumps({'mean_kl': 0.01, 'pairs': 30, 'round': count - 1}))

    def test_job_table(self):
        self.assertEqual(len(self.table), 26)
        self.assertEqual([job.id for job in self.table[:4]],
                         ['train.General', 'train.Specific', 'eval.heldout74.General.base', 'eval.heldout74.Specific.base'])
        self.assertEqual(self.by_id['eval.heldout74.General.r10'].needs_round, 10)
        self.assertEqual(self.by_id['eval.nonhard992.Specific.r10.s3of4'].shard, (3, 4))
        self.assertEqual(self.by_id['eval.nonhard992.Specific.base.s3of4'].needs_round, 0)
        priorities = [job.priority for job in self.table]
        self.assertEqual(priorities, sorted(priorities))
        self.assertEqual(self.settings.adapter('General', 10).parent.name, 'round_009')
        one = pilot.jobs(pilot.Settings({'PILOT_PROFILES': 'Specific', 'PILOT_FULL_SHARDS': '1', 'PILOT_EVAL_ROUNDS': '10'}))
        self.assertEqual([job.id for job in one], ['train.Specific', 'eval.heldout74.Specific.base', 'eval.heldout74.Specific.r10',
                                                   'eval.nonhard992.Specific.base', 'eval.nonhard992.Specific.r10'])
        with self.assertRaises(ValueError):
            pilot.Settings({'PILOT_EVAL_ROUNDS': '10 40'})

    def test_states_and_claims(self):
        train, base, later = self.by_id['train.General'], self.by_id['eval.heldout74.General.base'], self.by_id['eval.heldout74.General.r10']
        self.assertEqual(pilot.state(train, self.settings), 'ready')
        self.assertEqual(pilot.state(base, self.settings), 'ready')
        self.assertEqual(pilot.state(later, self.settings, self.table), 'waiting')
        self.assertTrue(pilot.claim(train, self.settings, 0, '0,1'))
        self.assertFalse(pilot.claim(train, self.settings, 1, '2,3'))
        self.assertEqual(pilot.state(train, self.settings), 'running')
        (self.settings.queue(train) / 'heartbeat').write_text(str(time.time() - 10 * pilot.STALE_SECONDS))
        self.assertEqual(pilot.state(train, self.settings), 'stale')
        self.assertTrue(pilot.claim(train, self.settings, 1, '2,3'))
        self.assertEqual(json.loads((self.settings.queue(train) / 'claim.json').read_text())['slot'], 1)
        self.finish_round('General', 10)
        self.assertEqual(pilot.state(later, self.settings, self.table), 'ready')
        (self.settings.queue(train) / 'failed.json').write_text('{}')
        self.assertEqual(pilot.state(self.by_id['eval.heldout74.General.r19'], self.settings, self.table), 'blocked')
        pilot.release(train, self.settings, 'retried')
        self.assertEqual(pilot.state(train, self.settings), 'ready')
        self.finish_round('General', 28)
        self.assertEqual(pilot.state(train, self.settings), 'done')

    def test_workers_finish_every_job_and_respect_the_adapters(self):
        settings, order = self.settings, []
        script = (
            'import json, sys, pathlib\n'
            'kind, target, rounds = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])\n'
            'if kind == "train":\n'
            '    for index in range(rounds):\n'
            '        adapter = target / f"round_{index:03d}" / "adapter"; adapter.mkdir(parents=True, exist_ok=True)\n'
            '        (adapter / "opd_update.json").write_text(json.dumps({"mean_kl": 0.02, "pairs": 29}))\n'
            'else:\n'
            '    assert not sys.argv[4] or (pathlib.Path(sys.argv[4]) / "opd_update.json").exists(), "adapter missing"\n'
            '    target.mkdir(parents=True, exist_ok=True)\n'
            '    score = {"passed": 5, "expected_tasks": int(sys.argv[5])}\n'
            '    (target / "complete.json").write_text(json.dumps({"identity": {"tasks": int(sys.argv[5])},\n'
            '        "score": {"final_solution": score, "first_submit_solution": score}, "timing": {},\n'
            '        "context_limit_tasks": [], "infrastructure_retried": 0, "exit_status_counts": {}}))\n')

        def fake_command(job, settings_, gpus, slot):
            order.append(job.id)
            if job.kind == 'train':
                return [sys.executable, '-c', script, 'train', str(settings.train_out(job.profile)), str(settings.rounds)]
            adapter = str(settings.adapter(job.profile, job.needs_round)) if job.needs_round else ''
            tasks = 74 if job.taskset == 'heldout74' else 248
            return [sys.executable, '-c', script, 'eval', str(settings.eval_out(job)), '0', adapter, str(tasks)]

        with patch.object(pilot, 'command', fake_command), patch.object(pilot, 'gpu_memory_used', lambda gpus: None), \
                patch.object(pilot, 'POLL_SECONDS', 0.05), patch.object(pilot.signal, 'signal', lambda *args: None), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.worker(settings, 0, '0,1'), 0)
        self.assertEqual(sorted(order), sorted(self.by_id))
        self.assertLess(order.index('train.General'), order.index('eval.heldout74.General.r10'))
        self.assertTrue(all(pilot.state(job, settings, self.table) == 'done' for job in self.table))
        result = pilot.summary(settings)
        cells = result['profiles']['General']['r10']
        self.assertEqual((cells['heldout74']['passed'], cells['heldout74']['tasks']), (5, 74))
        self.assertEqual((cells['nonhard992']['passed'], cells['nonhard992']['tasks']), (20, 992))
        self.assertEqual((cells['never_trained']['passed'], cells['never_trained']['tasks']), (25, 1066))
        self.assertNotIn('nonhard992', result['profiles']['General']['r19'])
        self.assertEqual(result['training']['Specific']['completed_rounds'], 28)
        text = pilot.markdown(result)
        self.assertIn('| General | after 10 updates | 5/74 (6.76%) | 20/992 (2.02%) | 25/1066 (2.35%) |', text)
        self.assertIn('| Specific | untrained |', text)

    def test_failed_job_is_recorded_and_not_repeated(self):
        settings = self.settings
        with patch.object(pilot, 'command', lambda job, s, gpus, slot: [sys.executable, '-c', 'raise SystemExit(7)']), \
                patch.object(pilot, 'gpu_memory_used', lambda gpus: None), patch.object(pilot, 'POLL_SECONDS', 0.05), \
                patch.object(pilot.signal, 'signal', lambda *args: None), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.worker(settings, 0, '0,1'), 1)
        failed = json.loads((settings.queue(self.by_id['train.General']) / 'failed.json').read_text())
        self.assertEqual(failed['returncode'], 7)
        states = {job.id: pilot.state(job, settings, self.table) for job in self.table}
        self.assertEqual(states['train.General'], 'failed')
        self.assertEqual(states['eval.heldout74.General.r10'], 'blocked')
        self.assertEqual(states['eval.nonhard992.General.base.s1of4'], 'failed')


class Resume(unittest.TestCase):
    """A job taken over from a worker whose node disappeared."""
    ROW_A = json.dumps({'task_id': 'BigCodeBench/1', 'final_solution': 'x = 1'})
    ROW_B = json.dumps({'task_id': 'BigCodeBench/2', 'final_solution': 'y = 2'})

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = pilot.Settings({'PILOT_OUT': str(root / 'out'), 'PILOT_SCRATCH': str(root / 'scratch'),
                                        'PILOT_AGENT_ENV': '/agent', 'PILOT_PROFILES': 'General',
                                        'PILOT_FULL_SHARDS': '1', 'PILOT_EVAL_ROUNDS': '10'})
        self.table = pilot.jobs(self.settings)
        self.by_id = {job.id: job for job in self.table}

    def tearDown(self):
        self.directory.cleanup()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def quiet(self):
        return contextlib.redirect_stdout(io.StringIO())

    def test_incomplete_lines_are_dropped_and_the_original_kept(self):
        job = self.by_id['eval.heldout74.General.base']
        out = self.settings.eval_out(job)
        episodes = self.write(out / 'eval_episodes.jsonl', (self.ROW_A + '\n').encode() + b'\0' * 300 + b'\n'
                              + (self.ROW_B + '\n').encode() + self.ROW_A[:25].encode())
        infra = self.write(out / 'infra_failed.jsonl', self.ROW_B.encode())
        with self.quiet():
            repairs = pilot.prepare_resume(job, self.settings, wait_seconds=1)
        self.assertEqual(episodes.read_text(), self.ROW_A + '\n' + self.ROW_B + '\n')
        self.assertEqual(infra.read_text(), self.ROW_B + '\n')
        by_file = {Path(entry['file']).name: entry for entry in repairs}
        self.assertEqual((by_file['eval_episodes.jsonl']['rows_kept'], by_file['eval_episodes.jsonl']['lines_dropped']), (2, 2))
        self.assertTrue(by_file['infra_failed.jsonl']['newline_added'])
        original = Path(by_file['eval_episodes.jsonl']['original'])
        self.assertTrue(original.read_bytes().endswith(self.ROW_A[:25].encode()))
        # Whole files are left alone, and a finished collection is never touched.
        with self.quiet():
            self.assertEqual(pilot.prepare_resume(job, self.settings, wait_seconds=1), [])
        (out / 'generation_complete.json').write_text('{}')
        self.write(out / 'eval_episodes.jsonl', b'{"broken"')
        with self.quiet():
            self.assertEqual(pilot.prepare_resume(job, self.settings, wait_seconds=1), [])
        self.assertEqual((out / 'eval_episodes.jsonl').read_bytes(), b'{"broken"')

    def test_training_rows_only_of_an_unfinished_collection(self):
        job = self.by_id['train.General']
        out = self.settings.train_out(job.profile)
        collected = self.write(out / 'round_000' / 'episodes.jsonl', b'{"broken"')
        self.write(out / 'round_000' / 'pairs.jsonl', b'{}\n')
        partial = self.write(out / 'round_001' / 'episodes.jsonl', (self.ROW_A + '\n' + self.ROW_B[:10]).encode())
        with self.quiet():
            repairs = pilot.prepare_resume(job, self.settings, wait_seconds=1)
        self.assertEqual([Path(entry['file']) for entry in repairs], [partial])
        self.assertEqual(partial.read_text(), self.ROW_A + '\n')
        self.assertEqual(collected.read_bytes(), b'{"broken"')

    def hold(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        holder = subprocess.Popen([sys.executable, '-c', 'import fcntl, sys, time\n'
                                   'handle = open(sys.argv[1], "a"); fcntl.flock(handle, fcntl.LOCK_EX)\n'
                                   'print("held", flush=True); time.sleep(60)', str(path)],
                                  stdout=subprocess.PIPE, text=True)
        self.addCleanup(holder.wait)
        self.addCleanup(holder.kill)
        self.assertEqual(holder.stdout.readline().strip(), 'held')
        return holder

    def test_a_lock_that_stays_held_fails_the_job_instead_of_blocking_it(self):
        job = self.by_id['train.General']
        holder = self.hold(self.settings.train_out(job.profile) / '.run.lock')
        marker = Path(self.directory.name) / 'started'
        self.assertTrue(pilot.claim(job, self.settings, 0, '0,1'))
        run = [sys.executable, '-c', 'import pathlib, sys; pathlib.Path(sys.argv[1]).touch()', str(marker)]
        with patch.object(pilot, 'LOCK_WAIT_SECONDS', 0.3), patch.object(pilot, 'gpu_memory_used', lambda gpus: None), \
                patch.object(pilot, 'command', lambda *args: run), self.quiet():
            self.assertFalse(pilot.run_job(job, self.settings, 0, '0,1'))
        failed = json.loads((self.settings.queue(job) / 'failed.json').read_text())
        self.assertIn('still locked', failed['error'])
        self.assertFalse(marker.exists())
        # Once the holder is gone (its lease ended), the job starts.
        holder.kill()
        holder.wait()
        with self.quiet():
            self.assertEqual(pilot.prepare_resume(job, self.settings, wait_seconds=1), [])

    def test_a_worker_takes_over_the_jobs_of_a_deleted_node(self):
        settings = self.settings
        train, base = self.by_id['train.General'], self.by_id['eval.heldout74.General.base']
        for job in (train, base):   # claims of a node that stopped 20 minutes ago
            queue = settings.queue(job)
            queue.mkdir(parents=True)
            (queue / 'heartbeat').write_text(str(time.time() - 20 * 60))
            (queue / 'claim.json').write_text(json.dumps({'host': 'deleted-node', 'slot': 0, 'gpus': '0,1'}))
        self.write(settings.train_out('General') / 'round_000' / 'episodes.jsonl',
                   (self.ROW_A + '\n' + self.ROW_B + '\n' + self.ROW_A[:30]).encode())
        self.write(settings.eval_out(base) / 'eval_episodes.jsonl', (self.ROW_A + '\n' + self.ROW_B).encode())
        script = (
            'import json, sys, pathlib\n'
            'kind, target = sys.argv[1], pathlib.Path(sys.argv[2])\n'
            'def rows(path):\n'
            '    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []\n'
            'if kind == "train":\n'
            '    assert len(rows(target / "round_000" / "episodes.jsonl")) == 2\n'
            '    for index in range(28):\n'
            '        adapter = target / f"round_{index:03d}" / "adapter"; adapter.mkdir(parents=True, exist_ok=True)\n'
            '        (adapter / "opd_update.json").write_text(json.dumps({"mean_kl": 0.02, "pairs": 29}))\n'
            'else:\n'
            '    seen = rows(target / "eval_episodes.jsonl")\n'
            '    assert len(seen) == len({row["task_id"] for row in seen})\n'
            '    with (target / "eval_episodes.jsonl").open("a") as handle:\n'
            '        handle.write(json.dumps({"task_id": "BigCodeBench/3"}) + "\\n")\n'
            '    assert len(rows(target / "eval_episodes.jsonl")) == len(seen) + 1\n'
            '    score = {"passed": 1, "expected_tasks": 74}\n'
            '    (target / "complete.json").write_text(json.dumps({"identity": {"tasks": 74},\n'
            '        "score": {"final_solution": score, "first_submit_solution": score}, "timing": {},\n'
            '        "context_limit_tasks": [], "infrastructure_retried": 0, "exit_status_counts": {}}))\n')

        def fake_command(job, settings_, gpus, slot):
            target = settings.train_out(job.profile) if job.kind == 'train' else settings.eval_out(job)
            return [sys.executable, '-c', script, job.kind, str(target)]

        with patch.object(pilot, 'command', fake_command), patch.object(pilot, 'gpu_memory_used', lambda gpus: None), \
                patch.object(pilot, 'POLL_SECONDS', 0.05), patch.object(pilot.signal, 'signal', lambda *args: None), \
                self.quiet():
            self.assertEqual(pilot.worker(settings, 1, '2,3'), 0)
        self.assertTrue(all(pilot.state(job, settings, self.table) == 'done' for job in self.table))
        for job in (train, base):
            done = json.loads((settings.queue(job) / 'done.json').read_text())
            self.assertEqual(done['slot'], 1)
            self.assertEqual(len(done['resume_repairs']), 1)
            moved = list(settings.queue(job).parent.glob(f'.stale.{job.id}.*'))
            self.assertEqual(len(moved), 1)
            self.assertEqual(json.loads((moved[0] / 'claim.json').read_text())['host'], 'deleted-node')


class CudaLibraryPath(unittest.TestCase):
    """env.sh drops library directories whose libcuda is older than the kernel module."""

    def resolve(self, kernel, layout):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for name, version in layout:
                folder = root / name
                folder.mkdir()
                if version:
                    (folder / f'libcuda.so.{version}').write_text('')
                    (folder / 'libcuda.so.1').symlink_to(f'libcuda.so.{version}')
                entries.append(str(folder))
            script = 'source experiments/pilot/env.sh; printf "%s\\n%s" "$LD_LIBRARY_PATH" "${PILOT_CUDA_DROPPED_LIBRARY_DIRS:-}"'
            env = dict(os.environ, PILOT_ROOT=str(root / 'pilot'), PILOT_KERNEL_DRIVER_VERSION=kernel,
                       LD_LIBRARY_PATH=':'.join(entries))
            output = subprocess.run(['bash', '-c', script], cwd=REPO, env=env, capture_output=True, text=True, check=True).stdout
            kept, _, dropped = output.partition('\n')
            return [Path(entry).name for entry in kept.split(':') if entry], dropped

    def test_newer_host_driver_drops_the_compat_copy(self):
        kept, dropped = self.resolve('580.167.08', [('compat', '580.95.05'), ('hostnv', '580.167.08'), ('fastrak', None)])
        self.assertEqual(kept, ['hostnv', 'fastrak'])
        self.assertIn('libcuda 580.95.05', dropped)

    def test_older_host_driver_keeps_forward_compatibility(self):
        kept, dropped = self.resolve('535.129.03', [('compat', '580.95.05'), ('hostnv', '535.129.03')])
        self.assertEqual(kept, ['compat', 'hostnv'])
        self.assertEqual(dropped, '')

    def test_equal_versions_change_nothing(self):
        kept, _ = self.resolve('580.95.05', [('compat', '580.95.05'), ('hostnv', '580.95.05')])
        self.assertEqual(kept, ['compat', 'hostnv'])


class Shim(unittest.TestCase):
    def run_shim(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bin').mkdir()
            real = root / 'real_bwrap'
            real.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            real.chmod(0o755)
            shim = root / 'bin' / 'bwrap'
            shim.write_text((REPO / 'experiments' / 'pilot' / 'tools' / 'bwrap_shim.sh').read_text())
            shim.chmod(0o755)
            (root / 'pilot_bwrap.conf').write_text(f'real={real}\nmode={mode}\n')
            arguments = ['--unshare-user', '--proc', '/proc', '--dev', '/dev', '--chdir', '/workspace',
                         '/bin/bash', '-c', 'mount --proc /proc; echo "a b"']
            return arguments, subprocess.run([str(shim), *arguments], capture_output=True, text=True, check=True).stdout.split('\n')[:-1]

    def test_native_mode_changes_nothing(self):
        arguments, seen = self.run_shim('native')
        self.assertEqual('\n'.join(seen), '\n'.join(arguments))

    def test_bind_modes_replace_only_the_sandbox_option(self):
        for mode, option in (('bind-proc', '--bind'), ('dev-bind-proc', '--dev-bind'), ('ro-bind-proc', '--ro-bind')):
            arguments, seen = self.run_shim(mode)
            expected = ['--unshare-user', option, '/proc', '/proc', '--dev', '/dev', '--chdir', '/workspace',
                        '/bin/bash', '-c', 'mount --proc /proc; echo "a b"']
            self.assertEqual('\n'.join(seen), '\n'.join(expected))


if __name__ == '__main__':
    unittest.main()
