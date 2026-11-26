"""Read-only check: how long are the reviewer's requests of a training round?

`worker_reviewer_bench/opd/reviewer.py` sends the reviewer the whole worker history as one JSON
string, and `experiments/bcb_opd/serve.py` serves at most 131072 tokens per request, of which each
reviewer request reserves `reviewer_max_tokens` (2200 in both configurations) for its answer. This
script rebuilds those requests and counts their tokens with the model's chat template:

- for every student trajectory left in a round's student workspaces (`student.traj.json`; mini-swe
  also saves an episode that ended in an exception, up to the request that failed), the reviewer's
  first request on that history;
- for the reviews saved in `episodes.jsonl` files (`reviewer_messages` is the input of the
  review's last request), the longest ones.

It reads files only: no GPU, no server, no model weights, and it writes nothing.

    python experiments/pilot/tools/reviewer_context.py --model "$PILOT_MODEL" --profile Specific \\
        --students "$PILOT_SCRATCH/train/Specific_r28/round_003/students" \\
        --episodes "$PILOT_OUT"/train/Specific_r28/round_*/episodes.jsonl
"""
import argparse
import json
import os
from pathlib import Path
import sys

CONTEXT = 131072  # --max-model-len in experiments/bcb_opd/serve.py


def reviewer_session(prompt, history, clean_messages):
    """The first request of StateReviewer.__call__ (worker_reviewer_bench/opd/reviewer.py)."""
    return [{'role': 'system', 'content': prompt},
            {'role': 'user', 'content': 'Current worker history:\n' + json.dumps(
                clean_messages(history), ensure_ascii=False)}]


def count(tokenizer, messages):
    """Tokens of a chat request as vLLM builds it: the rendered template, tokenized without extra specials."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])


def worker_count(tokenizer, history, clean_messages):
    try:
        return count(tokenizer, clean_messages(history))
    except Exception as error:  # the template may reject a message shape; this number is secondary
        return f'n/a ({type(error).__name__})'


def trajectories(roots):
    for root in roots:
        yield from sorted(Path(root).glob('**/student.traj.json'))


def saved_reviews(path):
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                yield None, None, None
                continue
            for record in episode.get('opd_records', []):
                yield episode.get('task_id'), record.get('turn'), record.get('reviewer_messages') or []


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='model directory with the tokenizer (PILOT_MODEL)')
    p.add_argument('--profile', required=True, choices=['General', 'Specific'])
    p.add_argument('--repo', default=os.environ.get('PILOT_REPO', str(Path(__file__).resolve().parents[3])))
    p.add_argument('--reviewer-max-tokens', type=int, default=2200)
    p.add_argument('--students', nargs='*', default=[], help='student workspace directories of a round')
    p.add_argument('--episodes', nargs='*', default=[], help='episodes.jsonl files')
    p.add_argument('--top', type=int, default=5, help='longest saved reviews to count per file')
    args = p.parse_args(argv)

    sys.path.insert(0, args.repo)
    from worker_reviewer_bench.opd.data import clean_messages
    prompt = json.loads((Path(args.repo) / 'worker_reviewer_bench/opd/prompts' / f'{args.profile}.json')
                        .read_text())['reviewer']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    limit = CONTEXT - args.reviewer_max_tokens
    print(f'LIMIT prompt tokens <= {limit} (max-model-len {CONTEXT} - reviewer max_tokens {args.reviewer_max_tokens})')

    largest = {'trajectory': 0, 'saved review': 0}
    for path in trajectories(args.students):
        data = json.loads(path.read_text())
        history = [m for m in data.get('messages', []) if m.get('role') != 'exit']
        tokens = count(tokenizer, reviewer_session(prompt, history, clean_messages))
        largest['trajectory'] = max(largest['trajectory'], tokens)
        print(f"TRAJ {path.parent.name} messages={len(history)} exit={data.get('info', {}).get('exit_status') or '-'} "
              f"reviewer_first_request={tokens} {'OVER' if tokens > limit else 'ok'} "
              f"worker_request_without_tool_schema={worker_count(tokenizer, history, clean_messages)}")

    for path in args.episodes:
        reviews, unreadable, kept = 0, 0, []
        for task, turn, messages in saved_reviews(path):
            if messages is None:
                unreadable += 1
                continue
            reviews += 1
            size = len(json.dumps(messages, ensure_ascii=False))
            kept = sorted(kept + [(size, task, turn, messages)], key=lambda item: -item[0])[:args.top]
        counted = [(count(tokenizer, m), task, turn) for _, task, turn, m in kept]
        if counted:
            largest['saved review'] = max(largest['saved review'], max(c[0] for c in counted))
        print(f'EPISODES {path} reviews={reviews} unreadable_lines={unreadable} longest: ' +
              (', '.join(f'{t}@turn{u}={n}' for n, t, u in sorted(counted, reverse=True)) or '-'))

    print(f"REVIEWER_CONTEXT largest_trajectory_request={largest['trajectory']} "
          f"largest_saved_review_request={largest['saved review']} limit={limit}")


if __name__ == '__main__':
    main()
