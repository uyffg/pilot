"""General and Specific prompt snapshots for Mini-SWE OPD."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
from ..benchmarks import adapt

PROMPTS = Path(__file__).with_name('prompts')

@lru_cache(maxsize=2)
def profile(name):
    if name not in {'General', 'Specific'}:
        raise ValueError('Prompt must be General or Specific')
    return json.loads((PROMPTS / (name + '.json')).read_text())

def mini_prompts(args, benchmark, system=None, instance=None):
    prompt = profile(getattr(args, 'prompt_profile', 'General'))[benchmark]
    return prompt['system'], prompt['instance']

def adapt_task(row, benchmark, backend, args):
    if backend != 'mini-swe':
        raise ValueError('Only mini-swe is supported')
    task = adapt(row, benchmark)
    if getattr(args, 'prompt_profile', 'General') == 'General':
        return task
    if benchmark == 'bcb':
        text = (f"BigCodeBench problem {task.id.replace('/', '__')} ()\n"
                f"Required entry point: {row['entry_point']}\n\n{row['instruct_prompt']}\n\n"
                f"Code prompt:\n{row['code_prompt']}\n\nImplement the required function in solution.py with the exact interface. "
                'Evaluation tests are withheld. Write and execute your own tests. '
                'Do not access benchmark dataset files or other task workspaces.')
    else:
        text = f"LiveCodeBench problem {task.id} ({row.get('platform', '')})\n{row['question_content']}"
        if row.get('starter_code'):
            text += '\n\nStarter code:\n' + row['starter_code']
    return replace(task, text=text)
