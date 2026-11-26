"""Replay the reviewer of one failed training episode against a running server, timing every request.

Both training runs failed in `StateReviewer.__call__` (`worker_reviewer_bench/opd/reviewer.py`) with
`Model request failed after 4 attempts`, which `worker_reviewer_bench/client.py` prints for any
`requests` exception and without the cause. This tool runs that same reviewer, with the arguments
the runner gives `opd-collect` (from the training configuration), on the history saved in the failed
episode's `student.traj.json`, and prints for every request what `client.py` does not: the elapsed
seconds, the exception type, the HTTP status, the token counts, the finish reason and the start and
end of the answer. The request itself is unchanged, including `client.py`'s own retries; only the
time this tool waits for an answer is longer (`--timeout`), so that a request slower than
`client.py`'s 180 s is still measured.

Run it with the Python of `opd-collect` (`OPD_FRONTEND_PYTHON`), inside the image with `env.sh`, against
a server that serves the weights of the failed round. The reviewer's commands, if it asks for any,
run in a sandbox snapshot of a copy of the episode's workspace, as in training. The tool writes only
under a temporary directory, which it removes at the end.

    "$OPD_FRONTEND_PYTHON" experiments/pilot/tools/reviewer_replay.py \\
        --traj "$PILOT_SCRATCH/train/General_r28/round_010/students/<episode>/student.traj.json" \\
        --config experiments/pilot/configs/General_r28.json --base-url http://127.0.0.1:<port>/v1
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

PROXY_VARIABLES = ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
                   'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY')


def describe(response, entry):
    """Add what the answer says to `entry`; never raises."""
    try:
        data = response.json()
        choice = data['choices'][0]
        message = choice.get('message') or {}
        content = message.get('content') or ''
        entry.update(usage=data.get('usage'), finish_reason=choice.get('finish_reason'),
                     tool_calls=bool(message.get('tool_calls')), content_chars=len(content),
                     whitespace_chars=sum(c.isspace() for c in content),
                     head=content[:300], tail=content[-200:])
    except Exception as error:  # the body is part of the evidence, whatever it is
        entry.update(body_error=type(error).__name__, body_head=response.text[:300])


def install_timer(requests, timeout, client_timeout, log):
    real_post = requests.post

    def timed_post(url, **kwargs):
        kwargs['timeout'] = timeout
        body = kwargs.get('json') or {}
        entry = {'request': len(log) + 1, 'messages': len(body.get('messages', [])),
                 'max_tokens': body.get('max_tokens')}
        started = time.time()
        try:
            response = real_post(url, **kwargs)
        except requests.RequestException as error:
            entry.update(seconds=round(time.time() - started, 1), exception=type(error).__name__,
                         detail=str(error)[:300])
            log.append(entry)
            print('REQUEST', json.dumps(entry, ensure_ascii=False), flush=True)
            raise
        entry.update(seconds=round(time.time() - started, 1), status=response.status_code)
        describe(response, entry)
        entry['over_client_timeout'] = entry['seconds'] > client_timeout
        log.append(entry)
        print('REQUEST', json.dumps(entry, ensure_ascii=False), flush=True)
        return response

    requests.post = timed_post


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--traj', required=True, help="the failed episode's student.traj.json")
    p.add_argument('--config', required=True, help='the training configuration of the run')
    p.add_argument('--base-url', required=True, help='a server with the weights of the failed round')
    p.add_argument('--agent-env', default=os.environ.get('PILOT_AGENT_ENV', ''))
    p.add_argument('--repo', default=os.environ.get('PILOT_REPO', str(Path(__file__).resolve().parents[3])))
    p.add_argument('--client-timeout', type=float, default=180, help="client.py's timeout, for the verdict only")
    p.add_argument('--timeout', type=float, default=900, help='seconds this replay waits for one answer')
    p.add_argument('--repeat', type=int, default=1)
    args = p.parse_args(argv)

    sys.path.insert(0, args.repo)
    import requests
    from worker_reviewer_bench.opd.reviewer import StateReviewer
    cfg = json.loads(Path(args.config).read_text())

    set_names = [name for name in PROXY_VARIABLES if os.environ.get(name)]
    print('PROXY variables set:', ', '.join(set_names) or 'none', '| used for the base URL:',
          bool(requests.utils.get_environ_proxies(args.base_url)), flush=True)
    session = requests.Session()
    session.trust_env = False
    models = session.get(args.base_url.rstrip('/') + '/models', timeout=30).json()
    print('MODELS', json.dumps([{'id': m.get('id'), 'root': m.get('root'), 'max_model_len': m.get('max_model_len')}
                                for m in models.get('data', [])]), flush=True)

    traj = json.loads(Path(args.traj).read_text())
    history = [m for m in traj.get('messages', []) if m.get('role') != 'exit']
    print('EPISODE', json.dumps({'traj': str(args.traj), 'messages': len(history),
                                 'exit_status': traj.get('info', {}).get('exit_status'),
                                 'profile': cfg['prompt_profile']}), flush=True)

    log = []
    install_timer(requests, args.timeout, args.client_timeout, log)
    outcomes = []
    with tempfile.TemporaryDirectory(prefix='reviewer-replay-') as td:
        # The episode failed before its first command ran, and before mini saved its trajectory.
        workspace = Path(td) / 'workspace'
        shutil.copytree(Path(args.traj).parent, workspace, symlinks=True,
                        ignore=shutil.ignore_patterns('student.traj.json'))
        review_root = Path(td) / 'reviews'
        review_root.mkdir()
        reviewer = StateReviewer(SimpleNamespace(
            reviewer_model=None, worker_model='opd-worker', base_url=args.base_url, api_key='local',
            reviewer_max_tokens=cfg['reviewer_max_tokens'], reviewer_turns=cfg['reviewer_turns'],
            command_timeout=cfg['command_timeout'], prompt_profile=cfg['prompt_profile'],
            review_root=str(review_root), sandbox='bwrap', agent_env=args.agent_env))
        for attempt in range(args.repeat):
            first = len(log)
            started = time.time()
            try:
                review, _ = reviewer(json.loads(json.dumps(history)), workspace)
                outcome = {'attempt': attempt + 1, 'result': 'review', 'review': json.loads(review)}
            except Exception as error:
                outcome = {'attempt': attempt + 1, 'result': type(error).__name__, 'detail': str(error)[:300]}
            outcome.update(seconds=round(time.time() - started, 1), requests=len(log) - first)
            outcomes.append(outcome)
            print('ATTEMPT', json.dumps(outcome, ensure_ascii=False), flush=True)

    print('REPLAY_RESULT', json.dumps({
        'attempts': len(outcomes), 'reviews': sum(o['result'] == 'review' for o in outcomes),
        'requests': len(log), 'exceptions': sorted({e['exception'] for e in log if 'exception' in e}),
        'statuses': sorted({e['status'] for e in log if 'status' in e}),
        'longest_seconds': max((e['seconds'] for e in log), default=0),
        'over_client_timeout': sum(bool(e.get('over_client_timeout')) for e in log)}), flush=True)


if __name__ == '__main__':
    main()
