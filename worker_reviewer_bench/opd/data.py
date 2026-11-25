"""Exact chat-template alignment for teacher/student views of one sampled action."""
from copy import deepcopy
import json

TEACHER_INSTRUCTION = '''Training-only private assessment of the current state:
{review}
Continue as the WORKER using its original action format. Choose the next useful
action from the task and visible evidence. If evidence is missing, collect it
with a tool before assuming the result. This assessment is guidance, not an
observation in the worker's history. Do not output a review, refer to this
assessment, or claim an unexecuted test ran. Submit when evidence is sufficient;
do not add pointless checks.'''


def clean_messages(messages):
    """Match the mini API view; retain tool calls/IDs but remove bookkeeping."""
    allowed = {'role', 'content', 'tool_calls', 'tool_call_id', 'name', 'reasoning_content'}
    return [{k: deepcopy(v) for k, v in m.items() if k in allowed and v is not None}
            for m in messages if m['role'] != 'exit']


def teacher_messages(messages, review):
    result = clean_messages(messages)
    assert result and result[0]['role'] == 'system'
    # Put privileged context in the system prompt: tool-call/result adjacency
    # and the student's entire interaction history remain valid and unchanged.
    result[0]['content'] += '\n\n' + TEACHER_INSTRUCTION.format(review=review)
    return result


def encode_pair(row, tokenizer, max_len):
    kwargs = {'tokenize': False, 'enable_thinking': False}
    if row.get('tools'):
        kwargs['tools'] = row['tools']
    history = clean_messages(row['student_messages'])
    action = clean_messages([row['action']])[0]
    if action['role'] != 'assistant':
        raise ValueError('Only a student assistant action may be a distillation target')
    views = [history, teacher_messages(history, row['review'])]
    encoded, suffixes = [], []
    for messages in views:
        # OpenAI tool-call arguments are JSON strings, while Qwen's native
        # Jinja template requires mappings (as vLLM's parser supplies them).
        messages = template_messages(messages)
        template_action = template_messages([action])[0]
        prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
        full = tokenizer.apply_chat_template(messages + [template_action], add_generation_prompt=False, **kwargs)
        if not full.startswith(prefix):
            raise ValueError(f'Chat template rewrites the action prefix: {row["id"]}')
        # Tokenize separately, exactly as a generation prompt plus continuation.
        prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix = tokenizer.encode(full[len(prefix):], add_special_tokens=False)
        if not suffix or not prompt_ids:
            raise ValueError('Empty prompt or action')
        ids = prompt_ids + suffix
        if len(ids) > max_len:
            raise ValueError(f'Overlong OPD view: {row["id"]}: {len(ids)} > {max_len}; no truncation')
        encoded.append({'input_ids': ids, 'start': len(prompt_ids) - 1})
        suffixes.append(suffix)
    if suffixes[0] != suffixes[1]:
        raise ValueError('Teacher/student action tokenization differs')
    return {'student': encoded[0], 'teacher': encoded[1], 'target_ids': suffixes[0]}


def template_messages(messages):
    result = deepcopy(messages)
    for message in result:
        for call in message.get('tool_calls', []):
            function = call['function']
            arguments = function.get('arguments', {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must be a JSON object')
            function['arguments'] = arguments
    return result
