"""Opt-in native Mini-SWE protocol guidance and lossless format-error context."""
from copy import deepcopy

PROTOCOL_VERSION = 'native-bash-compat-v1'
PROTOCOL_APPENDIX = '''

Interaction protocol: every response must invoke the native bash tool with a
nonempty string argument named command. Keep any prose before the call brief.
Do not print a tool-call JSON object as ordinary text. Use compact commands and
split long calculations or derivations across calls, writing or appending notes
and the solution to answer.txt as needed. A long plain-text proof can exhaust the
response budget before a tool call is emitted. Only executed commands change files.
Finish with the same COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT command described above.
'''
FORMAT_ERROR_TEMPLATE = '''{{ error }}
Use the native bash tool with arguments {"command":"a short shell command"}.
Keep the next call compact; split long work across calls and save it in answer.txt.
{% if finish_reason == 'length' %}The previous response reached its token limit.
{% endif %}An invalid or incomplete tool call has not executed. Continue from any
visible prior work, then make one complete native bash tool call.
'''


class ProtocolFeedbackModel:
    """Keep valid responses unchanged; recover context without inventing actions."""
    def __init__(self, model):
        self.wrapped = model
        self.format_recoveries = []

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def query(self, messages, **kwargs):
        from minisweagent.exceptions import FormatError
        try:
            return self.wrapped.query(messages, **kwargs)
        except FormatError as error:
            feedback = error.messages[0]
            extra = feedback.get('extra', {})
            response = extra.get('response')
            if not isinstance(response, dict):
                raise
            choices = response.get('choices') or []
            if not choices:
                raise
            choice = choices[0]
            raw = choice.get('message') or {}
            content = raw.get('content')
            usage = response.get('usage') or {}
            maximum = self.wrapped.config.model_kwargs.get('max_tokens')
            truncated = choice.get('finish_reason') == 'length' or (
                isinstance(maximum, int) and usage.get('completion_tokens', 0) >= maximum)
            detail = {'finish_reason': choice.get('finish_reason'), 'truncated': truncated,
                      'completion_tokens': usage.get('completion_tokens'),
                      'retained_content': isinstance(content, str) and bool(content.strip())}
            self.format_recoveries.append(detail)
            if truncated:
                feedback['content'] += (
                    '\nThe response exhausted the output-token budget, even if its finish reason '
                    'says tool_calls. Shorten the next command; an incomplete call did not run.')
            if isinstance(content, str) and content.strip():
                # Preserve ordinary assistant text only. Never promote broken tool arguments
                # to executable commands or leave unmatched tool-call IDs in API history.
                prior = {'role': 'assistant', 'content': content,
                         'extra': {'format_recovery': deepcopy(detail), 'cost': extra.get('cost', 0.0)}}
                error.messages = (prior, *error.messages)
            raise
