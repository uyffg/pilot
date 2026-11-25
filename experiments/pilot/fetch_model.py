"""Download Qwen/Qwen3.5-35B-A3B at the pinned revision and write the provenance file the runner checks.

The runner refuses to start unless <model>/bcb_verified.json names the repository and the
revision of its configuration (`base_revision`). This script downloads exactly that revision
(27 files, about 72 GB; the repository's `main` has moved on since), compares every file with
the size and hash that Hugging Face publishes for it, and only then writes the file.

    python experiments/pilot/fetch_model.py                  # download into $PILOT_MODEL and verify
    python experiments/pilot/fetch_model.py --verify-only    # re-hash an existing copy against its bcb_verified.json

--verify-only needs no network. Use it after copying a verified directory from shared storage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[2]
REPO_ID = 'Qwen/Qwen3.5-35B-A3B'
PROVENANCE = 'bcb_verified.json'


def pinned_revision():
    revisions = {json.loads((REPO / 'experiments' / 'bcb_opd' / f'{name}.json').read_text())['base_revision']
                 for name in ('General', 'Specific')}
    if len(revisions) != 1:
        raise ValueError('The two profiles pin different base revisions')
    return revisions.pop()


def file_hashes(path):
    """sha256 of the content, and the git blob id that Hugging Face reports for small files."""
    size = path.stat().st_size
    sha256, git = hashlib.sha256(), hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 ** 2), b''):
            sha256.update(chunk)
            git.update(chunk)
    return {'size': size, 'sha256': sha256.hexdigest(), 'git_blob': git.hexdigest()}


def verify(dest, expected, workers):
    names = sorted(expected)
    missing = [name for name in names if not (dest / name).is_file()]
    if missing:
        raise SystemExit(f'PILOT_MODEL_FAILED missing files: {missing}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        observed = dict(zip(names, pool.map(lambda name: file_hashes(dest / name), names)))
    bad = []
    for name in names:
        want, got = expected[name], observed[name]
        if want.get('size') is not None and want['size'] != got['size']:
            bad.append(f'{name}: size {got["size"]} != {want["size"]}')
        key = 'sha256' if want.get('sha256') else 'git_blob'
        if want.get(key) and want[key] != got[key]:
            bad.append(f'{name}: {key} differs')
    if bad:
        raise SystemExit('PILOT_MODEL_FAILED ' + '; '.join(bad))
    return observed


def published_files(revision):
    from huggingface_hub import HfApi
    info = HfApi().model_info(REPO_ID, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise SystemExit(f'PILOT_MODEL_FAILED Hugging Face resolved {revision} to {info.sha}')
    expected = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, 'lfs', None)
        sha256 = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
        expected[sibling.rfilename] = {'size': sibling.size, 'sha256': sha256,
                                       'git_blob': None if sha256 else getattr(sibling, 'blob_id', None)}
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dest', default=os.environ.get('PILOT_MODEL'))
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not args.dest:
        parser.error('Set PILOT_MODEL (source experiments/pilot/env.sh) or pass --dest')
    dest, revision = Path(args.dest), pinned_revision()

    if args.verify_only:
        record = json.loads((dest / PROVENANCE).read_text())
        if record['revision'] != revision or record['repo_id'] != REPO_ID:
            raise SystemExit('PILOT_MODEL_FAILED the copy records another repository or revision')
        expected = {name: {'size': entry['size'], 'sha256': entry['sha256']} for name, entry in record['files'].items()}
        verify(dest, expected, args.workers)
        print(f'PILOT_MODEL_OK {dest} revision={revision} files={len(expected)} (re-hashed)')
        return

    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ.pop(name, None)
    from huggingface_hub import snapshot_download
    expected = published_files(revision)
    started = time.time()
    snapshot_download(REPO_ID, revision=revision, local_dir=str(dest), max_workers=args.workers)
    observed = verify(dest, expected, args.workers)
    total = sum(entry['size'] for entry in observed.values())
    record = {'repo_id': REPO_ID, 'revision': revision,
              'verified_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'total_bytes': total,
              'files': {name: {'size': entry['size'], 'sha256': entry['sha256']} for name, entry in observed.items()}}
    temporary = dest / (PROVENANCE + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(dest / PROVENANCE)
    print(f'PILOT_MODEL_OK {dest} revision={revision} files={len(observed)} bytes={total} '
          f'seconds={time.time() - started:.0f}')


if __name__ == '__main__':
    main()
