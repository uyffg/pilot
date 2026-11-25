"""Shared JSONL I/O with explicit IDs and atomic report writes."""
import gzip
import json
from pathlib import Path


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_rows(rows, key):
    result = {}
    for row in rows:
        task_id = str(row[key])
        if task_id in result:
            raise ValueError(f'Duplicate task ID: {task_id}')
        result[task_id] = row
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    temporary.replace(path)
