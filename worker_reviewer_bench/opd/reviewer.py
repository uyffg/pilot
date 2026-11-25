"""State-only reviewer in a disposable snapshot, outside the student's history."""
from functools import partial
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from ..client import Chat, parse_json
from ..execution import shell
from .data import clean_messages

from .profiles import profile


DECISIONS = ['continue', 'check', 'repair', 'submit']
REVIEW_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'action': {'type': 'string', 'enum': ['review']},
        'decision': {'type': 'string', 'enum': DECISIONS},
        'reason': {'type': 'string', 'maxLength': 256},
        'check': {'type': 'string', 'maxLength': 256},
        'hint': {'type': 'string', 'maxLength': 256},
    },
    'required': ['action', 'decision', 'reason', 'check', 'hint'],
}
BASH_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'action': {'type': 'string', 'enum': ['bash']},
                   'command': {'type': 'string', 'maxLength': 2048}},
    'required': ['action', 'command'],
}


def response_format(final):
    schema = REVIEW_SCHEMA if final else {'oneOf': [BASH_SCHEMA, REVIEW_SCHEMA]}
    return {'type': 'json_schema', 'json_schema': {
        'name': 'state_review', 'strict': True, 'schema': schema}}


def valid_review(value):
    return (value.get('action') == 'review' and value.get('decision') in DECISIONS
            and all(isinstance(value.get(key), str) for key in ['reason', 'check', 'hint']))


class StateReviewer:
    def __init__(self, args):
        self.args = args
        self.client = Chat(args.reviewer_model or args.worker_model, args.base_url,
                           args.api_key, 0.0, args.reviewer_max_tokens)
        self.client.extra_body = {'chat_template_kwargs': {'enable_thinking': False}}

    def __call__(self, history, workspace):
        # Reject symlinks rather than following them outside the task snapshot.
        with tempfile.TemporaryDirectory(prefix='opd-review-', dir=self.args.review_root) as td:
            snapshot = Path(td) / 'workspace'
            def ignore(directory, names):
                return [n for n in names if (Path(directory) / n).is_symlink()
                        or n in {'.tmp', '__pycache__', '.git'}]
            shutil.copytree(workspace, snapshot, ignore=ignore)
            specific = getattr(self.args, 'prompt_profile', 'General') == 'Specific'
            if specific:
                (snapshot / 'reviewer_work').mkdir(exist_ok=True)
            run = partial(shell, sandbox=self.args.sandbox, agent_env=self.args.agent_env,
                          workspace_readonly=specific, writable_paths=('reviewer_work',) if specific else ())
            session = [{'role': 'system', 'content': profile(getattr(self.args, 'prompt_profile', 'General'))['reviewer']},
                       {'role': 'user', 'content': 'Current worker history:\n' + json.dumps(
                           clean_messages(history), ensure_ascii=False)}]
            # Last call must return a decision, not an unexecuted command.
            for turn in range(self.args.reviewer_turns):
                final = turn == self.args.reviewer_turns - 1
                self.client.extra_body = {'chat_template_kwargs': {'enable_thinking': False},
                                          'response_format': response_format(final)}
                if final:
                    session.append({'role': 'user', 'content': 'Return the final review JSON now; no more commands.'})
                raw = self.client.chat(session)
                try:
                    decision = parse_json(raw)
                except ValueError:
                    session += [{'role': 'assistant', 'content': raw},
                                {'role': 'user', 'content': 'Return valid review JSON.'}]
                    continue
                if decision.get('action') == 'bash' and turn < self.args.reviewer_turns - 1:
                    output = run(decision['command'], snapshot, self.args.command_timeout)
                    # Keep the JSON-chat protocol compatible with Qwen templates.
                    session += [{'role': 'assistant', 'content': raw},
                                {'role': 'user', 'content': 'Command output:\n' + output}]
                elif valid_review(decision):
                    return json.dumps(decision, ensure_ascii=False), session
                else:
                    session += [{'role': 'assistant', 'content': raw},
                                {'role': 'user', 'content': 'Return the required review JSON.'}]
            failures = Path(self.args.review_root) / 'failures'
            failures.mkdir(parents=True, exist_ok=True)
            audit_path = failures / (uuid.uuid4().hex + '.json')
            audit_path.write_text(json.dumps({'student_history': clean_messages(history),
                                             'reviewer_messages': session}, ensure_ascii=False))
            raise RuntimeError(f'Reviewer did not return a valid state assessment; audit: {audit_path}')
