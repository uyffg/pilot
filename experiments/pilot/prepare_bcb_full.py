"""Task files for the full-BigCodeBench evaluation, and their integrity check.

The repository's BCB-Hard files (experiments/bcb_opd/{public,grading}_tasks.jsonl) hold the 148
hard tasks. The full benchmark has 1,140 tasks. The evaluation covers the 1,066 tasks that were
never trained on: the 74 held-out hard tasks, which the 74-task evaluation already covers with
the repository's own files, and the 992 tasks outside BCB-Hard, which this script writes:

    experiments/pilot/data/bcb_nonhard_public.jsonl    992 tasks, no tests   (given to the agent)
    experiments/pilot/data/bcb_nonhard_grading.jsonl   the same tasks with their tests
    experiments/pilot/data/manifest.json               source, counts and hashes

Source: Hugging Face dataset `bigcode/bigcodebench`, split `v0.1.1`, at a pinned revision. That
split is the one the repository's hard tasks came from: all 148 agree with it in every prompt
field, and 147 agree in the test (BigCodeBench/1006, a training task, carries an older test).

    python experiments/pilot/prepare_bcb_full.py --check      # no network, standard library only
    python experiments/pilot/prepare_bcb_full.py --build      # needs huggingface_hub and pyarrow

The generated files are committed, so a node only runs --check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / 'bcb_opd'
DATA = HERE / 'data'
REPO_ID = 'bigcode/bigcodebench'
REVISION = 'b74c0d0bf70d2c0bc459be537895cca163007f1a'
SPLIT = 'v0.1.1'
PUBLIC_KEYS = ['task_id', 'instruct_prompt', 'code_prompt', 'entry_point', 'libs']
# Field order of the repository's own grading file.
GRADING_KEYS = ['task_id', 'instruct_prompt', 'code_prompt', 'test', 'entry_point', 'libs']
# The only field of the 148 repository tasks that differs from the pinned split.
KNOWN_DIFFERENCES = {('BigCodeBench/1006', 'test')}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(task_id):
    return int(task_id.rsplit('/', 1)[1])


def expected_ids():
    split = json.loads((ASSETS / 'split.json').read_text())
    hard = [row['task_id'] for row in rows(ASSETS / 'public_tasks.jsonl')]
    train, heldout = split['train_ids'], split['eval_ids']
    if len(train) != 74 or len(heldout) != 74 or set(train) & set(heldout) or set(train) | set(heldout) != set(hard):
        raise ValueError('The fixed 74/74 split no longer matches the 148 repository tasks')
    return set(train), set(heldout), set(hard)


def build():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    train, heldout, hard = expected_ids()
    path = hf_hub_download(REPO_ID, f'data/{SPLIT}-00000-of-00001.parquet', repo_type='dataset', revision=REVISION)
    source = pq.read_table(path).to_pylist()
    by_id = {row['task_id']: row for row in source}
    if len(source) != 1140 or len(by_id) != 1140:
        raise ValueError(f'Expected 1,140 distinct tasks, found {len(source)}')
    if not hard <= set(by_id):
        raise ValueError('Repository hard tasks are missing from the pinned split')
    # The repository's own rows must agree with the pinned split, so both evaluations use one benchmark version.
    differences = set()
    for row in rows(ASSETS / 'grading_tasks.jsonl'):
        for key in GRADING_KEYS:
            if by_id[row['task_id']][key] != row[key]:
                differences.add((row['task_id'], key))
    if differences != KNOWN_DIFFERENCES:
        raise ValueError(f'Unexpected differences from {REPO_ID}@{SPLIT}: {sorted(differences)}')
    nonhard = sorted((row for row in source if row['task_id'] not in hard), key=lambda row: number(row['task_id']))
    if len(nonhard) != 992:
        raise ValueError(f'Expected 992 tasks outside BCB-Hard, found {len(nonhard)}')
    DATA.mkdir(parents=True, exist_ok=True)
    public = DATA / 'bcb_nonhard_public.jsonl'
    grading = DATA / 'bcb_nonhard_grading.jsonl'
    public.write_text(''.join(json.dumps({key: row[key] for key in PUBLIC_KEYS}) + '\n' for row in nonhard), encoding='utf8')
    grading.write_text(''.join(json.dumps({key: row[key] for key in GRADING_KEYS}) + '\n' for row in nonhard),
                       encoding='utf8')
    manifest = {
        'source': {'repo_id': REPO_ID, 'repo_type': 'dataset', 'revision': REVISION, 'split': SPLIT,
                   'parquet_sha256': sha256(path)},
        'counts': {'benchmark': 1140, 'bcb_hard': 148, 'training': 74, 'heldout_hard': 74, 'nonhard': 992,
                   'never_trained': 1066},
        'repository_rows_differing_from_source': sorted(map(list, KNOWN_DIFFERENCES)),
        'files': {public.name: {'rows': len(nonhard), 'sha256': sha256(public)},
                  grading.name: {'rows': len(nonhard), 'sha256': sha256(grading)}},
        'repository_files': {name: sha256(ASSETS / name)
                             for name in ['public_tasks.jsonl', 'grading_tasks.jsonl', 'split.json']},
    }
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('BCB_FULL_BUILT', json.dumps(manifest['counts']))


def check():
    train, heldout, hard = expected_ids()
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, expected in manifest['repository_files'].items():
        if sha256(ASSETS / name) != expected:
            raise ValueError(f'experiments/bcb_opd/{name} changed since the task files were built')
    for name, entry in manifest['files'].items():
        if sha256(DATA / name) != entry['sha256']:
            raise ValueError(f'{name} does not match its manifest hash')
    public, grading = rows(DATA / 'bcb_nonhard_public.jsonl'), rows(DATA / 'bcb_nonhard_grading.jsonl')
    ids = [row['task_id'] for row in public]
    if ids != [row['task_id'] for row in grading] or len(ids) != 992 or len(set(ids)) != 992:
        raise ValueError('Public and grading task files disagree')
    if set(ids) & hard:
        raise ValueError('A BCB-Hard task leaked into the non-hard file')
    if any('test' in row for row in public):
        raise ValueError('The public task file contains tests')
    if any(list(row) != PUBLIC_KEYS for row in public) or any(list(row) != GRADING_KEYS for row in grading):
        raise ValueError('Unexpected task fields')
    if any({key: g[key] for key in PUBLIC_KEYS} != p for p, g in zip(public, grading)):
        raise ValueError('Public fields differ between the two files')
    if any(not row['test'].strip() for row in grading):
        raise ValueError('A task has no test')
    if len(set(ids) | heldout) != 1066 or (set(ids) | heldout) & train:
        raise ValueError('The never-trained set is not 1,066 tasks disjoint from the training tasks')
    print('BCB_FULL_CHECK_OK', json.dumps({'nonhard': len(ids), 'heldout_hard': len(heldout), 'training_excluded': len(train),
                                           'revision': manifest['source']['revision'], 'split': manifest['source']['split']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    build() if args.build else check()


if __name__ == '__main__':
    sys.exit(main())
