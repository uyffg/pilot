"""Mini-SWE on-policy distillation and unaided evaluation."""
import argparse
import os


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    p = commands.add_parser('collect', help='Collect trajectories; --no-review generates worker-only eval rollouts')
    p.add_argument('--benchmark', choices=['lcb','bcb'], required=True)
    p.add_argument('--backend', choices=['mini-swe'], default='mini-swe')
    for key in ['tasks','output','workdir']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--worker-model', default=os.getenv('WORKER_MODEL','Qwen/Qwen3.5-35B-A3B'))
    p.add_argument('--reviewer-model', default=os.getenv('REVIEWER_MODEL'))
    p.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL','http://localhost:8000/v1'))
    p.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY','dummy'))
    p.add_argument('--workers', type=positive, default=8)
    p.add_argument('--max-turns', type=positive, default=160)
    p.add_argument('--reviewer-turns', type=positive, default=160)
    p.add_argument('--command-timeout', type=positive, default=120)
    p.add_argument('--review-retries', type=int, default=2, help='mini-swe repair rounds after the initial attempt')
    p.add_argument('--no-review', action='store_true')
    p.add_argument('--disable-thinking', action='store_true', help='Match non-thinking training chat templates')
    p.add_argument('--prompt-profile', choices=['General','Specific'], default='General')
    p.add_argument('--model-class', default='litellm')
    p.add_argument('--cost-limit', type=float, default=3.0)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--reviewer-temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=positive, default=1400)
    p.add_argument('--reviewer-max-tokens', type=positive, default=2200)
    p.add_argument('--sandbox', choices=['local','bwrap'], default='local')
    p.add_argument('--agent-env', help='Self-contained Python environment mounted in Bubblewrap')
    p = commands.add_parser('grade', help='Grade first and final snapshots with independent tests')
    p.add_argument('--benchmark', choices=['lcb','bcb'], required=True)
    p.add_argument('--tasks','--raw-tasks', required=True)
    p.add_argument('--rows', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=positive, default=8)
    p.add_argument('--timeout', type=positive, default=60)
    p.add_argument('--test-source', choices=['public','private'], default='private')
    p.add_argument('--exclude', nargs='*', default=[])
    p.add_argument('--cache', help='Optional per-code/test grading cache')
    p = commands.add_parser('merge', help='Merge a finite LoRA adapter into a local base model')
    for key in ['model','adapter','out']:
        p.add_argument('--'+key, required=True)
    p = commands.add_parser('serve', help='Serve a base or merged model with vLLM')
    p.add_argument('--model', required=True)
    p.add_argument('--name', default='worker-reviewer')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=positive, default=8000)
    p.add_argument('--gpus', type=positive, default=2)
    p.add_argument('--max-len', type=positive, default=131072)
    p.add_argument('--max-seqs', type=positive, default=8)
    p.add_argument('--gpu-memory', type=float, default=0.94)
    p.add_argument('--seed', type=int, default=42)
    p = commands.add_parser('opd-collect', help='Independent BCB student rollout with side-channel reviews')
    for key in ['tasks', 'output', 'pairs', 'workdir', 'review-root', 'student-revision']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--backend', choices=['mini-swe'], default='mini-swe')
    p.add_argument('--prompt-profile', choices=['General','Specific'], default='General')
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--worker-model', default='opd-worker')
    p.add_argument('--reviewer-model', default=None)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY', 'dummy'))
    p.add_argument('--workers', type=positive, default=4)
    p.add_argument('--max-turns', type=positive, default=160)
    p.add_argument('--max-tokens', type=positive, default=1400)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--reviewer-temperature', type=float, default=0.0)
    p.add_argument('--reviewer-max-tokens', type=positive, default=2200)
    p.add_argument('--reviewer-turns', type=positive, default=4)
    p.add_argument('--review-every', type=positive, default=4)
    p.add_argument('--command-timeout', type=positive, default=120)
    p.add_argument('--sandbox', choices=['local', 'bwrap'], default='local')
    p.add_argument('--agent-env')
    p.add_argument('--model-class', choices=['litellm'], default='litellm')
    p.add_argument('--cost-limit', type=float, default=0.0)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    if args.command == 'opd-collect':
        if args.sandbox == 'bwrap' and not args.agent_env:
            root.error('--agent-env is required with --sandbox bwrap')
        from .opd.collect import collect
        return collect(args)
    if args.command == 'collect':
        if args.review_retries < 0:
            root.error('--review-retries must be nonnegative')
        if args.sandbox == 'bwrap' and not args.agent_env:
            root.error('--agent-env is required with --sandbox bwrap')
        from .collection import collect
        return collect(args)
    if args.command == 'grade':
        from .evaluation import evaluate
        return evaluate(args)
    from . import models
    return getattr(models, args.command)(args)


if __name__ == '__main__':
    raise SystemExit(main())
