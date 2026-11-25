"""Mirror the progress and the results of the runs to Weights & Biases.

The trainer of this repository does not log to W&B, and it is not changed for the pilot. This
script reads PILOT_OUT and writes what it finds: one W&B run per training run (step = update,
with the mean KL and the number of teacher/student pairs of that update) and one run named
`evaluation` whose summary holds every finished evaluation score. It only reads the results.

    python -m experiments.pilot.wandb_sync                 # once
    python -m experiments.pilot.wandb_sync --loop 900      # every 15 minutes until every job is done
    python -m experiments.pilot.wandb_sync --dry-run       # print what would be logged; no W&B needed

Project: WANDB_PROJECT or `pilot-opd-bcb`. Entity: WANDB_ENTITY or the account's default.
"""
import argparse
import json
import os
import time

from experiments.pilot.pilot import Settings, jobs, state, summary


def payload(settings):
    """Everything to mirror: per training run its updates, and the finished scores as flat keys."""
    result = summary(settings)
    scores = {}
    for profile, labels in result['profiles'].items():
        for label, cells in labels.items():
            for taskset, cell in cells.items():
                if cell.get('complete'):
                    scores[f'{profile}/{label}/{taskset}/accuracy'] = cell['accuracy']
                    scores[f'{profile}/{label}/{taskset}/passed'] = cell['passed']
                    scores[f'{profile}/{label}/{taskset}/tasks'] = cell['tasks']
    training = {profile: progress['updates'] for profile, progress in result['training'].items()}
    return training, scores


def sync(settings, project, dry_run):
    training, scores = payload(settings)
    record_path = settings.out / 'wandb' / 'synced.json'
    synced = json.loads(record_path.read_text()) if record_path.exists() else {}
    if dry_run:
        print(json.dumps({'training': training, 'scores': scores}, indent=2))
        return
    import wandb
    entity = os.environ.get('WANDB_ENTITY') or None
    for profile, updates in training.items():
        name = f'train-{profile}-r{settings.rounds}'
        new = [update for update in updates if update['round'] > synced.get(name, 0)]
        if not new:
            continue
        config = json.loads(settings.config(profile).read_text())
        run = wandb.init(project=project, entity=entity, id=name, name=name, resume='allow', config=config,
                         dir=str(settings.out / 'wandb'))
        for update in new:
            run.log({'mean_kl': update['mean_kl'], 'pairs': update['pairs'], 'update': update['round']},
                    step=update['round'])
        run.finish()
        synced[name] = new[-1]['round']
    if scores and scores != synced.get('scores'):
        run = wandb.init(project=project, entity=entity, id='evaluation', name='evaluation', resume='allow',
                         dir=str(settings.out / 'wandb'))
        run.summary.update(scores)
        run.finish()
        synced['scores'] = scores
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(synced, indent=2) + '\n')
    print('PILOT_WANDB_SYNCED', json.dumps({key: value for key, value in synced.items() if key != 'scores'}),
          f'{len(scores)} score values', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', default=os.environ.get('WANDB_PROJECT', 'pilot-opd-bcb'))
    parser.add_argument('--loop', type=int, default=0, help='Seconds between passes; 0 for a single pass')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('PILOT_OUT'):
        parser.error('PILOT_OUT is not set; source experiments/pilot/env.sh first')
    settings = Settings()
    (settings.out / 'wandb').mkdir(parents=True, exist_ok=True)
    while True:
        sync(settings, args.project, args.dry_run)
        table = jobs(settings)
        if not args.loop or all(state(job, settings, table) in ('done', 'failed', 'blocked') for job in table):
            return 0
        time.sleep(args.loop)


if __name__ == '__main__':
    raise SystemExit(main())
