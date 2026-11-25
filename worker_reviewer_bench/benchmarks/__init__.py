"""Task adapters: benchmark differences belong here, never in episode loops."""
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Task:
    id: str
    benchmark: str
    text: str
    metadata: dict

    @property
    def workspace_name(self):
        return re.sub(r'[^A-Za-z0-9_.-]', '_', self.id).replace('..', '__')

    def prepare(self, workdir):
        workspace = Path(workdir).resolve() / self.workspace_name
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / 'problem.txt').write_text(self.text, encoding='utf8')
        return workspace

    def identify(self, result):
        key = 'task_id' if self.benchmark == 'bcb' else 'question_id'
        return {**self.metadata, **result, key: self.id, 'benchmark': self.benchmark}


def adapt(row, benchmark):
    if benchmark == 'bcb':
        task_id = str(row['task_id'])
        text = (f"BigCodeBench task {task_id}\nRequired entry point: {row['entry_point']}\n\n"
                f"{row['instruct_prompt']}\n\nCode prompt:\n{row['code_prompt']}\n\n"
                'Implement the exact function interface in solution.py. Evaluation tests are withheld. '
                'Write and run your own specification-based tests.')
        metadata = {'entry_point': row['entry_point']}
    else:
        task_id = str(row['question_id'])
        text = f"LiveCodeBench release_v6 problem {task_id}\n{row['question_content']}"
        if row.get('starter_code'):
            text += '\n\nStarter code:\n' + row['starter_code']
        text += '\n\nWrite the required Python 3 implementation in solution.py.'
        metadata = {k: row[k] for k in ['contest_id','platform','difficulty','contest_date'] if k in row}
        metadata['interactive'] = task_id in {'abc337_e', 'abc355_e'}
    if not task_id or task_id in {'.', '..'}:
        raise ValueError('Invalid task ID')
    return Task(task_id, benchmark, text, metadata)
