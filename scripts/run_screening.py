"""Run a resumable, time-limited local ensemble screening batch.

Run from the repository root with .venv/bin/python scripts/run_screening.py.
Each fit runs in a subprocess with a timeout. The final test partition is
reserved before all sampling and is never scored by this script.
"""

import argparse
import csv
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback

import pandas as pd

from assignment4.dec_tree.id3 import id3, predict
from assignment4.dec_tree.info_metrics import information_gains
from assignment4.dec_tree.reduced_error_pruning import reduced_error_pruning
from assignment4.ensemble import bagging, boosting
from assignment4.progress import ProgressBar, format_duration


CATEGORICAL = ['proto', 'service', 'state', 'is_sm_ips_ports']


def stratified_sample(df, size, seed):
    """Sample approximately proportionally, retaining every available class."""
    if size >= len(df):
        return df.copy()
    groups = [group for _, group in df.groupby(df.columns[-1], sort=True)]
    if size < len(groups):
        raise ValueError('Sample size must cover every target class.')
    desired = [size * len(group) / len(df) for group in groups]
    counts = [max(1, int(value)) for value in desired]
    while sum(counts) < size:
        index = max((i for i in range(len(groups)) if counts[i] < len(groups[i])),
                    key=lambda i: desired[i] - counts[i])
        counts[index] += 1
    while sum(counts) > size:
        index = max((i for i in range(len(groups)) if counts[i] > 1),
                    key=lambda i: counts[i] - desired[i])
        counts[index] -= 1
    rng = random.Random(seed)
    selected = []
    for group, count in zip(groups, counts):
        positions = list(range(len(group)))
        rng.shuffle(positions)
        selected.append(group.iloc[positions[:count]])
    return pd.concat(selected).sample(frac=1, random_state=seed).reset_index(drop=True)


def predict_tree(model, row):
    """Use training-majority fallback for categories unseen during training."""
    if any(row[col] not in values for col, values in model['domains'].items()):
        return model['fallback']
    return predict(model['tree'], row)


def train_tree(sample, numerical, domains, progress=False):
    """Adapt ID3 to the generic ensemble callback interface."""
    return {
        'tree': id3(sample, information_gains, continuous_features=numerical,
                    domains=domains, progress=progress),
        'domains': domains,
        'fallback': sample.iloc[:, -1].mode().iloc[0],
    }


def metrics(targets, predictions, classes):
    """Compute multiclass metrics and a confusion matrix without dependencies."""
    positions = {label: i for i, label in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for actual, predicted in zip(targets, predictions):
        matrix[positions[actual]][positions[predicted]] += 1
    per_class = {}
    for i, label in enumerate(classes):
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = matrix[i][i] / predicted_count if predicted_count else 0.0
        recall = matrix[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = dict(support=support, precision=precision, recall=recall, f1=f1)
    return {
        'accuracy': sum(matrix[i][i] for i in range(len(classes))) / len(targets),
        'macro_f1': sum(item['f1'] for item in per_class.values()) / len(classes),
        'balanced_accuracy': sum(item['recall'] for item in per_class.values()) / len(classes),
        'per_class': per_class,
        'confusion_matrix': matrix,
        'class_order': classes,
    }


def evaluate(model, validation, classes):
    """Score only the tuning partition, recording batch prediction time."""
    started = time.monotonic()
    if isinstance(model, dict):
        predictions = [predict_tree(model, row)
                       for row in validation.iloc[:, :-1].to_dict(orient='records')]
    else:
        predictions = model.predict_many(validation).tolist()
    result = metrics(validation.iloc[:, -1].tolist(), predictions, classes)
    result['prediction_seconds'] = time.monotonic() - started
    result['validation_rows'] = len(validation)
    return result


def write_json(path, value):
    """Replace a JSON checkpoint only after its contents are complete."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def worker(output, job):
    """Fit one configuration, score it, and save metrics and fitted models."""
    state = dict(phase='loading data')
    last_report = 0.0

    def report(force=False, **updates):
        nonlocal last_report
        state.update(updates)
        now = time.monotonic()
        if force or now - last_report >= 0.2:
            write_json(output / 'jobs' / (job['id'] + '.progress.json'), state)
            last_report = now

    def row_progress(completed, total):
        report(force=completed in (0, total), phase='training',
               rows_completed=completed, rows_total=total)

    def model_progress(completed, total):
        report(force=True, models_completed=completed, models_total=total)

    report(force=True)
    pools = pd.read_pickle(output / 'development_pools.pkl')
    sample = stratified_sample(pools['grow'], job['training_rows'], job['seed'])
    prune = stratified_sample(pools['prune'], 2000, 2026)
    validation = pools['validation']
    numerical = [col for col in sample.columns[:-1] if col not in CATEGORICAL]
    domains = {col: pools['grow'][col].unique().tolist() for col in CATEGORICAL}
    trainer = partial(train_tree, numerical=numerical, domains=domains, progress=row_progress)
    classes = sorted(pools['grow'].iloc[:, -1].unique().tolist())
    started = time.monotonic()
    results = []
    if job['method'] == 'tree':
        model = trainer(sample)
        fit_seconds = time.monotonic() - started
        report(force=True, phase='scoring tree')
        result = evaluate(model, validation, classes)
        result.update(method='tree', fit_seconds=fit_seconds, actual_estimators=1)
        results.append(result)
        with (output / 'models' / (job['id'] + '.pkl')).open('wb') as file:
            pickle.dump(model, file)
        pruning_started = time.monotonic()
        report(force=True, phase='pruning')
        pruned = dict(model)
        pruned['tree'] = reduced_error_pruning(model['tree'], sample, prune)
        pruning_seconds = time.monotonic() - pruning_started
        report(force=True, phase='scoring pruned tree')
        result = evaluate(pruned, validation, classes)
        result.update(method='pruned_tree', fit_seconds=fit_seconds + pruning_seconds,
                      pruning_seconds=pruning_seconds, actual_estimators=1)
        results.append(result)
        with (output / 'models' / (job['id'] + '-pruned.pkl')).open('wb') as file:
            pickle.dump(pruned, file)
    else:
        kwargs = dict(n_estimators=job['n_estimators'], random_state=job['seed'],
                      progress=model_progress)
        if job['method'] == 'boosting':
            model = boosting(sample, trainer, predict_tree, **kwargs)
        else:
            if job['method'] == 'subbagging':
                kwargs.update(bootstrap=False, sample_size=max(1, len(sample) // 2))
            model = bagging(sample, trainer, predict_tree, **kwargs)
        fit_seconds = time.monotonic() - started
        report(force=True, phase='scoring ensemble')
        result = evaluate(model, validation, classes)
        result.update(method=job['method'], fit_seconds=fit_seconds,
                      actual_estimators=len(model.estimators),
                      estimator_errors=model.estimator_errors,
                      estimator_weights=model.estimator_weights)
        results.append(result)
        with (output / 'models' / (job['id'] + '.pkl')).open('wb') as file:
            pickle.dump(model, file)
    for result in results:
        result.update(training_rows=len(sample), seed=job['seed'],
                      requested_estimators=job['n_estimators'])
    write_json(output / 'jobs' / (job['id'] + '.json'),
               dict(status='complete', job=job, results=results))
    report(force=True, phase='complete')


def prepare(output, data):
    """Freeze code and stratified partitions before any experiment fitting."""
    output.mkdir(parents=True, exist_ok=False)
    for name in ['jobs', 'models', 'logs']:
        (output / name).mkdir()
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / 'src', output / 'code' / 'src',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, output / 'code' / 'run_screening.py')
    shutil.copy2(root / 'pyproject.toml', output / 'code' / 'pyproject.toml')
    shutil.copy2(root / 'uv.lock', output / 'code' / 'uv.lock')
    df = pd.read_csv(data)
    if df.columns[-1] != 'attack_cat' or df.isna().any().any():
        raise ValueError('Expected cleaned data, no missing values, attack_cat last.')
    indices = {name: [] for name in ['grow', 'prune', 'validation', 'test']}
    rng = random.Random(2026)
    for _, group in df.groupby('attack_cat', sort=True):
        positions = group.index.tolist()
        rng.shuffle(positions)
        n_test = max(1, int(0.2 * len(positions)))
        indices['test'].extend(positions[:n_test])
        remaining = positions[n_test:]
        n_grow, n_prune = int(0.6 * len(remaining)), int(0.2 * len(remaining))
        indices['grow'].extend(remaining[:n_grow])
        indices['prune'].extend(remaining[n_grow:n_grow + n_prune])
        indices['validation'].extend(remaining[n_grow + n_prune:])
    write_json(output / 'split_indices.json', indices)
    pools = {name: df.loc[indices[name]].reset_index(drop=True)
             for name in ['grow', 'prune', 'validation']}
    pd.to_pickle(pools, output / 'development_pools.pkl')
    write_json(output / 'manifest.json', dict(
        dataset=str(data.resolve()), dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        python=sys.version, split_seed=2026,
        partition_rows={name: len(value) for name, value in indices.items()},
        partition_counts={name: df.loc[value, 'attack_cat'].value_counts().sort_index().to_dict()
                          for name, value in indices.items()},
        notes='Preliminary screening, all cleaned features. Final test partition is never scored. '
              'Training subsets retain every class; pruning uses a separate 2000-row sample. '
              'Validation uses its full natural class distribution. Unseen categories use '
              'training-majority fallback. Seeds vary training subsamples and ensemble sampling; '
              'the validation partition is fixed. Timings include local machine load.'))


def collect(output):
    """Rebuild the summary from completed, durable job checkpoints."""
    rows = []
    for path in sorted((output / 'jobs').glob('*.json')):
        if path.name.endswith('.progress.json'):
            continue
        checkpoint = json.loads(path.read_text())
        for result in checkpoint.get('results', []):
            rows.append({key: value for key, value in result.items()
                         if not isinstance(value, (dict, list))})
    if rows:
        columns = sorted(set().union(*(row.keys() for row in rows)))
        with (output / 'summary.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def run(output, seconds, job_timeout, progress=True):
    """Run the prioritised queue until completed or the wall-clock cap is met."""
    jobs = []
    # Early jobs give paired baselines and a first ensemble comparison before
    # spending time on bigger samples or more ensemble members.
    for size in [500, 1000, 2000]:
        for seed in [42, 43, 44]:
            for method in ['tree', 'bagging', 'subbagging', 'boosting']:
                count = 1 if method == 'tree' else 5
                jobs.append(dict(id=f'{size}-{seed}-{method}-{count}', method=method,
                                 training_rows=size, seed=seed, n_estimators=count))
    for size in [500, 1000, 2000]:
        for seed in [42, 43, 44]:
            for method in ['bagging', 'boosting']:
                jobs.append(dict(id=f'{size}-{seed}-{method}-15', method=method,
                                 training_rows=size, seed=seed, n_estimators=15))
    write_json(output / 'plan.json', dict(seconds=seconds, job_timeout=job_timeout, jobs=jobs))
    deadline = time.monotonic() + seconds
    environment = dict(os.environ, PYTHONPATH=str(output / 'code' / 'src'))
    attempted = sum((output / 'jobs' / (job['id'] + '.json')).exists() for job in jobs)
    with ProgressBar(len(jobs), 'Experiments', progress) as bar:
        bar.update(attempted, force=True)
        for job in jobs:
            checkpoint = output / 'jobs' / (job['id'] + '.json')
            if checkpoint.exists():
                continue
            remaining = deadline - time.monotonic()
            if remaining < 5:
                break
            write_json(output / 'status.json', dict(status='running', current_job=job,
                                                   pid=os.getpid()))
            bar.update(attempted, detail=job['id'] + ' | starting | budget left ' +
                       format_duration(remaining), force=True)
            job_deadline = min(deadline, time.monotonic() + job_timeout)
            progress_path = output / 'jobs' / (job['id'] + '.progress.json')
            with (output / 'logs' / (job['id'] + '.log')).open('w') as log:
                process = subprocess.Popen(
                    [sys.executable, str(output / 'code' / 'run_screening.py'),
                     '--output', str(output), '--worker', json.dumps(job)],
                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    while process.poll() is None:
                        remaining = job_deadline - time.monotonic()
                        if remaining <= 0:
                            process.kill()
                            process.wait()
                            write_json(checkpoint, dict(status='timeout', job=job))
                            break
                        state = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                        detail = f"{job['id']} | {state.get('phase', 'starting')}"
                        if state.get('phase') == 'training':
                            if 'models_total' in state:
                                detail += f" models {state['models_completed']}/{state['models_total']}"
                            if 'rows_total' in state:
                                fraction = state['rows_completed'] / max(1, state['rows_total'])
                                filled = int(10 * fraction)
                                detail += (' rows [' + '#' * filled + '-' * (10 - filled)
                                           + f"] {state['rows_completed']}/{state['rows_total']}")
                        detail += ' | budget left ' + format_duration(deadline - time.monotonic())
                        bar.update(attempted, detail=detail)
                        try:
                            process.wait(timeout=min(0.25, remaining))
                        except subprocess.TimeoutExpired:
                            pass
                    if not checkpoint.exists():
                        write_json(checkpoint, dict(status='failed', job=job,
                                                    returncode=process.returncode))
                finally:
                    # Ctrl-C must not leave the current worker running.
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            collect(output)
            attempted += 1
            bar.update(attempted, detail=job['id'] + ' ' +
                       json.loads(checkpoint.read_text())['status'], force=True)
    statuses = [json.loads((output / 'jobs' / (job['id'] + '.json')).read_text())['status']
                for job in jobs if (output / 'jobs' / (job['id'] + '.json')).exists()]
    write_json(output / 'status.json', dict(status='finished', attempted=len(statuses),
                                           completed=statuses.count('complete'),
                                           timeouts=statuses.count('timeout'),
                                           failed=statuses.count('failed'),
                                           planned=len(jobs), pid=os.getpid()))
    print('FINISHED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('notebooks/data/networkTraffic.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=7200)
    parser.add_argument('--job-timeout', type=float, default=360)
    parser.add_argument('--worker')
    parser.add_argument('--no-progress', action='store_true', help='Disable terminal progress bars.')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        # Persisted callbacks must have an importable module name, not __main__.
        from run_screening import worker as run_worker
        run_worker(output, json.loads(args.worker))
    else:
        if not (output / 'manifest.json').exists():
            prepare(output, args.data)
        try:
            run(output, args.seconds, args.job_timeout, progress=not args.no_progress)
        except KeyboardInterrupt:
            collect(output)
            write_json(output / 'status.json', dict(status='interrupted'))
            print('\nStopped; saved results remain available. Resume using the same --output.')
        except Exception:
            write_json(output / 'status.json', dict(status='failed', traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
