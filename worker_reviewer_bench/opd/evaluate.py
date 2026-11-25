"""Validate saved unaided evaluations without retrying unsuccessful episodes."""
import hashlib
import json
from collections import Counter


def validate_evaluation(path, expected_ids, backend, benchmark='bcb'):
    if backend != 'mini-swe':
        raise ValueError('Only mini-swe is supported')
    content = path.read_bytes()
    episodes = [json.loads(line) for line in content.splitlines() if line.strip()]
    key = 'task_id' if benchmark == 'bcb' else 'question_id'
    ids = [row[key] for row in episodes]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('Incomplete, duplicate, or unexpected held-out evaluation IDs')
    failures = []
    for row in episodes:
        if row.get('backend') != backend or row.get('evaluation_tests_visible') is not False:
            raise ValueError('Unexpected evaluation backend or test visibility')
        if row.get('worker_error'):
            raise ValueError(f"Evaluation infrastructure error: {row[key]}: {row['worker_error']}")
        if row.get('review_history'):
            raise ValueError('Reviewer feedback in unaided evaluation')
        review = row.get('review')
        if review and (review.get('decision') != 'pass' or review.get('reason') != 'Review disabled'):
            raise ValueError('Unexpected review in unaided evaluation')
        error = row.get('review_error')
        if backend == 'mini-swe':
            status = row.get('worker_result', {}).get('exit_status')
            if status == 'Submitted':
                empty = error == 'worker did not create solution.py' and not row.get('final_solution')
                if not empty and (error or row.get('submitted') is not True):
                    raise ValueError('Inconsistent submitted evaluation episode')
            elif status in {'LimitsExceeded', 'RepeatedFormatError'}:
                if (error not in {'Worker session did not submit successfully', 'worker did not create solution.py'}
                        or row.get('submitted') is not False
                        or row.get('first_submit_captured') is not False
                        or row.get('first_submit_solution')):
                    raise ValueError('Inconsistent unsuccessful evaluation episode')
                failures.append({key: row[key], 'exit_status': status})
            else:
                raise ValueError(f"Unexpected evaluation exit status: {row[key]}: {status}")
    return {'tasks': len(episodes), 'reviewer': False,
            'episodes_sha256': hashlib.sha256(content).hexdigest(),
            'submitted': sum(bool(row.get('submitted')) for row in episodes),
            'unsuccessful_episodes': failures,
            'exit_status_counts': dict(Counter(row.get('worker_result', {}).get('exit_status',
                'Submitted' if row.get('submitted') else 'NotSubmitted') for row in episodes))}
