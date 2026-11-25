import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from worker_reviewer_bench.benchmarks import adapt
from worker_reviewer_bench.cli import parser
from worker_reviewer_bench.opd.collect import Recorder, collect_one
from worker_reviewer_bench.opd.profiles import mini_prompts, profile
from worker_reviewer_bench.opd.reviewer import StateReviewer


class ProfileTests(unittest.TestCase):
    def test_each_profile_reaches_worker_and_private_reviewer(self):
        for name in ['General', 'Specific']:
            with self.subTest(profile=name):
                args = parser().parse_args(['opd-collect', '--tasks', 'unused', '--output', 'unused',
                    '--pairs', 'unused', '--workdir', 'unused', '--review-root', 'unused',
                    '--student-revision', 'base', '--round', '0', '--prompt-profile', name])
                self.assertEqual(args.backend, 'mini-swe')
                self.assertEqual(mini_prompts(args, 'bcb')[0], profile(name)['bcb']['system'])
                with tempfile.TemporaryDirectory() as td:
                    args.review_root = td
                    workspace = Path(td) / 'student'
                    workspace.mkdir()
                    reviewer = StateReviewer(args)
                    reviewer.client = Mock()
                    reviewer.client.chat.return_value = json.dumps({'action': 'review', 'decision': 'submit',
                        'reason': 'Sufficient evidence', 'check': '', 'hint': ''})
                    reviewer([{'role': 'user', 'content': 'task'}], workspace)
                    messages = reviewer.client.chat.call_args.args[0]
                    self.assertEqual(messages[0]['content'], profile(name)['reviewer'])

    def test_private_review_cannot_change_native_submission_or_student_history(self):
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL
        for name in ['General', 'Specific']:
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as td:
                args = argparse.Namespace(review_every=4, round=0, student_revision='base',
                    backend='mini-swe', prompt_profile=name)
                task = adapt({'task_id': 'one', 'entry_point': 'f', 'instruct_prompt': 'Return 1', 'code_prompt': ''}, 'bcb')
                history = [{'role': 'system', 'content': 'worker'}, {'role': 'user', 'content': task.text}]
                before = deepcopy(history)
                action = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'one', 'type': 'function',
                    'function': {'name': 'bash', 'arguments': json.dumps({'command': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'})}}]}
                def review(visible, workspace):
                    self.assertEqual(visible, before)
                    self.assertNotIn('tool_calls', json.dumps(visible))
                    visible.append({'role': 'user', 'content': 'PRIVATE: do not submit'})
                    return 'PRIVATE: do not submit', []
                recorder = Recorder(args, task, task.prepare(td), review)
                recorder.turn = 1  # Submission is reviewed even between scheduled reviews.
                recorder.observe(history, action, [BASH_TOOL])
                self.assertEqual(history, before)
                self.assertEqual(recorder.records[0]['action']['tool_calls'], action['tool_calls'])
                self.assertNotIn('PRIVATE', json.dumps(recorder.records[0]['student_messages']))

    def test_partial_workspace_restarts_only_for_matching_student_revision(self):
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(workdir=td + '/students', review_root=td + '/reviews',
                student_revision='base', round=0, backend='mini-swe', prompt_profile='General',
                worker_model='worker', reviewer_model=None, base_url='http://unused', api_key='dummy',
                reviewer_max_tokens=100)
            task = adapt({'task_id': 'one', 'entry_point': 'f', 'instruct_prompt': 'Return 1', 'code_prompt': ''}, 'bcb')
            worker = Mock(messages=[], model=Mock())
            worker.run.return_value = {'exit_status': 'Submitted'}
            with patch('worker_reviewer_bench.backends.mini._make_agent', return_value=worker):
                collect_one(task, args)
                workspace = Path(args.workdir) / task.workspace_name
                (workspace / 'partial.py').write_text('unfinished')
                collect_one(task, args)
                self.assertFalse((workspace / 'partial.py').exists())
                args.student_revision = 'different'
                with self.assertRaisesRegex(ValueError, 'not owned'):
                    collect_one(task, args)

    def test_custom_backend_is_not_available(self):
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            parser().parse_args(['collect', '--benchmark', 'bcb', '--tasks', 'unused',
                '--output', 'unused', '--workdir', 'unused', '--backend', 'custom'])
