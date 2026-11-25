import json
from pathlib import Path
import tempfile
import unittest

from worker_reviewer_bench.opd.evaluate import validate_evaluation


class EvaluationValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'episodes.jsonl'

    def validate(self, episodes):
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in episodes))
        return validate_evaluation(self.path, ['t'], 'mini-swe')

    def episode(self, status='Submitted'):
        submitted = status == 'Submitted'
        return {'task_id': 't', 'backend': 'mini-swe', 'evaluation_tests_visible': False,
                'submitted': submitted, 'worker_result': {'exit_status': status},
                'worker_error': '', 'review_history': [],
                'review_error': '' if submitted else 'Worker session did not submit successfully',
                'review': {'decision': 'pass', 'reason': 'Review disabled'} if submitted else None,
                'first_submit_captured': submitted, 'first_submit_solution': 'solution' if submitted else '',
                'final_solution': 'possibly incomplete solution'}

    def test_model_failures_retained_without_retry_or_code_changes(self):
        for status in ['Submitted', 'LimitsExceeded', 'RepeatedFormatError']:
            with self.subTest(status=status):
                episode = self.episode(status)
                summary = self.validate([episode])
                self.assertEqual(summary['submitted'], int(status == 'Submitted'))
                self.assertEqual(len(summary['unsuccessful_episodes']), int(status != 'Submitted'))
                self.assertEqual(json.loads(self.path.read_text()), episode)

    def test_incomplete_duplicate_and_unknown_ids_rejected(self):
        for episodes in [[], [self.episode(), self.episode()], [dict(self.episode(), task_id='other')]]:
            with self.subTest(episodes=episodes), self.assertRaises(ValueError):
                self.validate(episodes)

    def test_api_failures_and_unexpected_exit_status_rejected(self):
        for episode in [dict(self.episode(), worker_error='API timeout'), self.episode('RuntimeError')]:
            with self.subTest(episode=episode), self.assertRaises(ValueError):
                self.validate([episode])

    def test_review_or_inconsistent_failure_rejected(self):
        for episode in [dict(self.episode(), review_history=[{'decision': 'intervene'}]),
                        dict(self.episode(), review={'decision': 'pass', 'reason': 'Checked'}),
                        dict(self.episode('LimitsExceeded'), first_submit_captured=True),
                        dict(self.episode('LimitsExceeded'), submitted=True)]:
            with self.subTest(episode=episode), self.assertRaises(ValueError):
                self.validate([episode])



if __name__ == '__main__':
    unittest.main()
