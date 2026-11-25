"""One shell executor for local and Bubblewrap-isolated sessions."""
import os
from pathlib import Path
import signal
import subprocess


def shell(command, cwd, timeout, *, sandbox='local', agent_env=None,
          workspace_readonly=False, writable_paths=()):
    cwd = Path(cwd).resolve()
    env = dict(os.environ, MPLBACKEND='Agg', QT_QPA_PLATFORM='offscreen',
               DISPLAY='', WAYLAND_DISPLAY='', BROWSER='true')
    if sandbox == 'bwrap':
        if not agent_env:
            raise ValueError('--agent-env is required with --sandbox bwrap')
        venv = Path(agent_env).resolve()
        temporary = cwd / '.tmp'
        temporary.mkdir(exist_ok=True)
        command_args = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc',
            '--unshare-uts', '--unshare-net', '--die-with-parent', '--ro-bind', '/usr', '/usr',
            '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
            '--symlink', 'usr/lib64', '/lib64', '--ro-bind', '/etc', '/etc',
            '--proc', '/proc', '--dev', '/dev', '--bind', str(temporary), '/tmp',
            '--ro-bind', str(venv), str(venv), '--ro-bind' if workspace_readonly else '--bind', str(cwd), '/workspace',
            '--chdir', '/workspace', '/bin/bash', '-c', command]
        overlay = os.environ.get('OPD_DEPENDENCY_OVERLAY')
        python_path = '/workspace/_packages'
        if overlay:
            overlay = Path(overlay).resolve()
            if not overlay.is_dir():
                raise ValueError(f'OPD dependency overlay is missing: {overlay}')
            # Insert mount options before the command, keeping dependencies
            # immutable and outside the worker/reviewer writable workspace.
            command_index = command_args.index('--chdir')
            command_args[command_index:command_index] = ['--ro-bind', str(overlay), '/opd-dependencies']
            python_path += ':/opd-dependencies'
        for name in writable_paths:
            path = (cwd / name).resolve()
            relative = path.relative_to(cwd)
            if not path.exists():
                raise ValueError(f'Writable reviewer path is missing: {path}')
            index = command_args.index('--chdir')
            command_args[index:index] = ['--bind', str(path), '/workspace/' + str(relative)]
        env = {k: env[k] for k in ['MPLBACKEND','QT_QPA_PLATFORM','DISPLAY','WAYLAND_DISPLAY','BROWSER']}
        env.update(PATH=f'{venv}/bin:/usr/bin:/bin', HOME='/workspace', TMPDIR='/tmp',
                   LANG='C.UTF-8', PYTHONNOUSERSITE='1', PYTHONPATH=python_path,
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    else:
        command_args = ['/bin/bash', '-c', command]
    process = subprocess.Popen(command_args, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, errors='replace', start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return f'exit_code={process.returncode}\n{stdout}\n{stderr}'.strip()[-12000:]
    except subprocess.TimeoutExpired:
        return f'timeout_after_seconds={timeout}'
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
