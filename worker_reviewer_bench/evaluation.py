"""One grading pipeline for both benchmarks and both agent backends."""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import pickle
import zlib

from .benchmarks.bcb import grade as grade_bcb
from .benchmarks.lcb import run_solution as grade_lcb
from .io import read_rows, index_rows, write_json

INTERACTIVE = {'abc337_e', 'abc355_e'}


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Only data are permitted in test payloads')


def decode_cases(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        cases = json.loads(value)
    except ValueError:
        payload = zlib.decompress(base64.b64decode(value))
        decoded = DataOnlyUnpickler(io.BytesIO(payload)).load()
        cases = json.loads(decoded) if isinstance(decoded, str) else decoded
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError('Test payload must be a list of test cases')
    return cases


def evaluate(args):
    key = 'task_id' if args.benchmark == 'bcb' else 'question_id'
    tasks = index_rows(read_rows(args.tasks), key)
    rows = index_rows(read_rows(args.rows), key)
    unknown = set(rows) - set(tasks)
    if unknown:
        raise ValueError(f'Prediction IDs absent from task manifest: {sorted(unknown)}')
    excluded = set(args.exclude or [])
    if args.benchmark == 'lcb':
        excluded |= INTERACTIVE
    expected = set(tasks) - excluded
    rows = {k:v for k,v in rows.items() if k in expected}
    # Cache keys include the scorer, code, tests, field and timeout.
    cache_path = Path(args.cache) if args.cache else None
    cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() +
        Path(__file__).with_name('benchmarks').joinpath(args.benchmark + '.py').read_bytes()).hexdigest()
    report = []
    for field in ['first_submit_solution', 'final_solution']:
        results = {}
        def score(task_id):
            row, task = rows[task_id], tasks[task_id]
            if field not in row or (field == 'first_submit_solution' and row.get('first_submit_captured') is False):
                return task_id, None, {'status':'missing_snapshot', 'passed':False, 'tests_run':0}
            code = row[field] or ''
            source = 'dataset' if args.benchmark == 'bcb' else args.test_source
            payload = task.get('test', '') if args.benchmark == 'bcb' else task.get(source + '_test_cases', [])
            signature = hashlib.sha256(json.dumps([fingerprint, task_id, field, code, payload, source,
                                                   args.timeout], sort_keys=True).encode()).hexdigest()
            if signature in cache:
                return task_id, signature, cache[signature]
            if args.benchmark == 'bcb':
                result = grade_bcb({'test':payload, field:code}, field, args.timeout)
            else:
                cases = decode_cases(payload)
                result = grade_lcb(code, cases, args.timeout)
                result['test_source'] = source
            return task_id, signature, result
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(score, task_id):task_id for task_id in rows}
            for job in as_completed(jobs):
                try:
                    task_id, signature, result = job.result()
                except Exception as exc:
                    task_id, signature = jobs[job], None
                    result = {'status':'error', 'passed':False, 'tests_run':0, 'error':str(exc)}
                results[task_id] = result
                if signature:
                    cache[signature] = result
                if cache_path:
                    write_json(cache_path, cache)
        passed = sum(bool(r.get('passed')) for r in results.values())
        report.append({'field':field, 'n':len(results), 'passed':passed,
            'accuracy':passed/len(expected) if expected else None,
            'evaluated_accuracy':passed/len(results) if results else None,
            'expected_tasks':len(expected), 'missing_tasks':sorted(expected - set(rows)),
            'excluded_tasks':sorted(set(tasks) & excluded),
            'tasks_with_executed_tests':sum(r.get('tests_run', 0) > 0 for r in results.values()),
            'results':dict(sorted(results.items()))})
    write_json(args.out, report)
    print(json.dumps([{k:v for k,v in r.items() if k!='results'} for r in report], indent=2))
    return int(any(v.get('status') == 'error' for r in report for v in r['results'].values()))
