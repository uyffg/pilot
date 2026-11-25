"""Decide how the Bubblewrap sandbox runs on this node, and prove that it isolates.

worker_reviewer_bench/execution.py starts every agent and reviewer command as
`bwrap --unshare-user --unshare-pid ... --proc /proc ... /bin/bash -c <command>`, and looks for
`bwrap` in "<agent env>/bin:/usr/bin:/bin". Two things can stop that on a node: there is no
bwrap in /usr/bin, and the node is a container with a masked /proc, where mounting a new procfs
is refused. This script tries, in this order, and keeps the first that works:

  system         /usr/bin/bwrap, the command exactly as written (nothing is installed)
  native         a wrapper in <agent env>/bin that only locates a bwrap binary elsewhere
  bind-proc      the same wrapper, replacing `--proc /proc` by `--bind /proc /proc`
  dev-bind-proc  ... by `--dev-bind /proc /proc`
  ro-bind-proc   ... by `--ro-bind /proc /proc`

    python experiments/pilot/sandbox_probe.py --install [--bwrap /path/to/bwrap]
    python experiments/pilot/sandbox_probe.py --check     # the recorded mode still works; changes nothing

The result is written to <PILOT_ROOT>/sandbox_mode.json. Every run of the experiment must use
the same mode: the mode is part of the environment in which the agent tests its code.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from worker_reviewer_bench.execution import shell  # noqa: E402

SHIM = Path(__file__).resolve().parent / 'tools' / 'bwrap_shim.sh'
MODES = ('native', 'bind-proc', 'dev-bind-proc', 'ro-bind-proc')
# The import check of the runner's own environment preflight.
BCB_IMPORTS = ('import json,numpy,pandas,scipy,matplotlib,sklearn,nltk,sympy,PIL,requests; '
               'print(json.dumps({"numpy":numpy.__version__,"pandas":pandas.__version__, '
               '"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}))')


def sandboxed(command, workspace, agent_env, **kwargs):
    try:
        return shell(command, workspace, 120, sandbox='bwrap', agent_env=agent_env, **kwargs)
    except FileNotFoundError as error:
        return f'bwrap not found: {error}'


def works(scratch, agent_env):
    with tempfile.TemporaryDirectory(dir=scratch, prefix='sandbox-probe-') as directory:
        observed = sandboxed('python -c "print(6 * 7)"', Path(directory), agent_env)
    return observed.startswith('exit_code=0') and '42' in observed, observed


def remove_shim(agent_env):
    for path in (Path(agent_env) / 'bin' / 'bwrap', Path(agent_env) / 'pilot_bwrap.conf'):
        path.unlink(missing_ok=True)


def install_shim(agent_env, real, mode):
    target = Path(agent_env) / 'bin' / 'bwrap'
    shutil.copyfile(SHIM, target)
    target.chmod(0o755)
    (Path(agent_env) / 'pilot_bwrap.conf').write_text(f'real={real}\nmode={mode}\n')


def find_real(explicit):
    candidates = [explicit, os.environ.get('PILOT_BWRAP_REAL'), '/usr/bin/bwrap', '/bin/bwrap']
    if os.environ.get('PILOT_TOOLS'):
        candidates.append(str(Path(os.environ['PILOT_TOOLS']) / 'bwrap' / 'usr' / 'bin' / 'bwrap'))
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    return None


def isolation(scratch, agent_env, hidden):
    """What the sandbox must and must not allow, checked with the functions the experiment uses."""
    checks = {}
    with tempfile.TemporaryDirectory(dir=scratch, prefix='sandbox-probe-') as directory:
        workspace = Path(directory)
        observed = sandboxed('touch /workspace/written && echo WRITABLE', workspace, agent_env)
        checks['workspace_writable'] = 'WRITABLE' in observed and (workspace / 'written').exists()
        observed = sandboxed('python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect((\'1.1.1.1\', 53))"',
                             workspace, agent_env)
        checks['network_blocked'] = not observed.startswith('exit_code=0')
        observed = sandboxed(f'touch {shlex.quote(str(Path(agent_env) / "probe_write"))} && echo WROTE', workspace, agent_env)
        checks['agent_env_read_only'] = 'WROTE' not in observed
        for name, path in hidden.items():
            observed = sandboxed(f'ls {shlex.quote(str(path))} >/dev/null 2>&1 && echo VISIBLE', workspace, agent_env)
            checks[f'{name}_hidden'] = 'VISIBLE' not in observed
        observed = sandboxed('python -c ' + shlex.quote(BCB_IMPORTS), workspace, agent_env)
        checks['benchmark_packages'] = observed.startswith('exit_code=0')
        checks['benchmark_packages_output'] = observed[:400]
        # The Specific reviewer runs with a read-only workspace and one writable directory.
        (workspace / 'reviewer_work').mkdir()
        observed = sandboxed('touch reviewer_work/a && ! touch b 2>/dev/null && echo READ_ONLY_OK', workspace, agent_env,
                             workspace_readonly=True, writable_paths=('reviewer_work',))
        checks['reviewer_read_only_workspace'] = 'READ_ONLY_OK' in observed
        observed = sandboxed('id -u; head -1 /proc/self/status; ls /proc | grep -c "^[0-9]"', workspace, agent_env)
        checks['identity_and_proc'] = observed[:300]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--install', action='store_true')
    action.add_argument('--check', action='store_true')
    parser.add_argument('--bwrap', help='A bwrap binary outside /usr/bin (default: PILOT_TOOLS/bwrap/usr/bin/bwrap)')
    parser.add_argument('--agent-env', default=os.environ.get('PILOT_AGENT_ENV'))
    args = parser.parse_args()
    root = Path(os.environ['PILOT_ROOT'])
    scratch = Path(os.environ.get('PILOT_SCRATCH', root / 'scratch'))
    scratch.mkdir(parents=True, exist_ok=True)
    record_path = root / 'sandbox_mode.json'
    agent_env = args.agent_env
    if not agent_env or not (Path(agent_env) / 'bin' / 'python').exists():
        parser.error('The agent environment does not exist yet; run setup_env.sh first')

    if args.check:
        record = json.loads(record_path.read_text())
        ok, observed = works(scratch, agent_env)
        if not ok:
            raise SystemExit(f'PILOT_SANDBOX_FAILED recorded mode {record["mode"]}: {observed}')
    else:
        remove_shim(agent_env)
        real = find_real(args.bwrap)
        attempts = []
        record = None
        if any(os.access(path, os.X_OK) for path in ('/usr/bin/bwrap', '/bin/bwrap')):
            ok, observed = works(scratch, agent_env)
            attempts.append({'mode': 'system', 'ok': ok, 'observed': observed[-600:]})
            if ok:
                record = {'mode': 'system', 'real': real}
        for mode in MODES:
            if record or not real:
                break
            install_shim(agent_env, real, mode)
            ok, observed = works(scratch, agent_env)
            attempts.append({'mode': mode, 'ok': ok, 'observed': observed[-600:]})
            if ok:
                record = {'mode': mode, 'real': real}
        if not record:
            remove_shim(agent_env)
            print(json.dumps(attempts, indent=2))
            raise SystemExit('PILOT_SANDBOX_FAILED no bwrap binary found' if not real else
                             'PILOT_SANDBOX_FAILED the sandbox command does not run in any mode')
        record['attempts'] = attempts
        version = subprocess.run([record['real'], '--version'], capture_output=True, text=True)
        record['bwrap_version'] = (version.stdout or version.stderr).strip()

    hidden = {'results': os.environ.get('PILOT_OUT', root / 'out'), 'model': os.environ.get('PILOT_MODEL', root / 'models')}
    checks = isolation(scratch, agent_env, hidden)
    record.update(host=socket.gethostname(), agent_env=str(agent_env), checks=checks)
    failed = [name for name, value in checks.items() if value is False]
    if args.install or not failed:
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: record[key] for key in ('mode', 'real', 'checks')}, indent=2))
    if failed:
        raise SystemExit(f'PILOT_SANDBOX_FAILED checks: {", ".join(failed)}')
    print(f'PILOT_SANDBOX_OK mode={record["mode"]}')


if __name__ == '__main__':
    main()
