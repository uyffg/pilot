"""Adapt the shared executor to mini-swe-agent's environment protocol."""
from pathlib import Path


def make_environment(args, cwd):
    from minisweagent.environments.local import LocalEnvironment

    class SharedEnvironment(LocalEnvironment):
        def execute(self, action, cwd='', *, timeout=None):
            text = args.shell(action.get('command', ''), Path(cwd or self.config.cwd),
                              timeout or self.config.timeout)
            if text.startswith('exit_code='):
                first, _, body = text.partition('\n')
                output = {'output': body, 'returncode': int(first.split('=')[1]), 'exception_info': ''}
            else:
                output = {'output': text, 'returncode': -1, 'exception_info': text}
            self._check_finished(output)
            return output

    return SharedEnvironment(cwd=str(cwd), timeout=args.command_timeout)
