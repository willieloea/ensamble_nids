"""Paired correlation ablation using the previous screening's fixed partitions."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import pandas as pd

from assignment4.progress import ProgressBar, format_duration


HERE = Path(__file__).resolve().parent
CATEGORICAL = ['proto', 'service', 'state', 'is_sm_ips_ports']
VARIANTS = ['all_features', 'reduced_a', 'reduced_b']


def write_json(path, value):
    """Replace a checkpoint atomically."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def checksum(path):
    """Hash an input or source file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def correlated_groups(strength, threshold):
    """Merge strongest pairs only when every pair in the new group qualifies.

    This deterministic greedy complete-link rule avoids treating correlation
    as transitive. It need not eliminate every edge between distinct groups.
    """
    columns = strength.columns.tolist()
    groups = [[col] for col in columns]
    edges = sorted(
        [(float(strength.loc[left, right]), i, j)
         for i, left in enumerate(columns)
         for j, right in enumerate(columns[i + 1:], i + 1)
         if strength.loc[left, right] >= threshold],
        key=lambda edge: (-edge[0], edge[1], edge[2]),
    )
    for _, i, j in edges:
        first = next(k for k, group in enumerate(groups) if columns[i] in group)
        second = next(k for k, group in enumerate(groups) if columns[j] in group)
        if first == second:
            continue
        if all(strength.loc[a, b] >= threshold
               for a in groups[first] for b in groups[second]):
            merged = sorted(groups[first] + groups[second], key=columns.index)
            groups = [group for k, group in enumerate(groups)
                      if k not in (first, second)] + [merged]
    return sorted([group for group in groups if len(group) > 1],
                  key=lambda group: columns.index(group[0]))


def choose_features(grow, threshold):
    """Build two target-independent reduction policies from growth predictors."""
    predictors = grow.columns[:-1].tolist()
    numerical = [col for col in predictors if col not in CATEGORICAL]
    pearson = grow[numerical].corr(method='pearson')
    spearman = grow[numerical].corr(method='spearman')
    strength = pearson.abs().combine(spearman.abs(), lambda a, b: a.combine(b, max)).fillna(0)
    groups = correlated_groups(strength, threshold)
    feature_sets = {'all_features': predictors}
    for variant, representative in [('reduced_a', 0), ('reduced_b', -1)]:
        dropped = {col for group in groups for col in group if col != group[representative]}
        feature_sets[variant] = [col for col in predictors if col not in dropped]
    pairs = [dict(feature_1=a, feature_2=b, pearson=float(pearson.loc[a, b]),
                  spearman=float(spearman.loc[a, b]), strength=float(strength.loc[a, b]))
             for i, a in enumerate(numerical) for b in numerical[i + 1:]
             if strength.loc[a, b] >= threshold]
    residual = {variant: [pair for pair in pairs
                          if pair['feature_1'] in kept and pair['feature_2'] in kept]
                for variant, kept in feature_sets.items()}
    return dict(groups=[dict(members=group, representative_a=group[0], representative_b=group[-1])
                        for group in groups],
                feature_sets=feature_sets, strong_pairs=pairs, residual_pairs=residual)


def validate_config(config):
    """Validate fixed policies and counts before preparing or fitting."""
    for key in ['training_rows', 'pruning_rows', 'job_timeout']:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if config['pruning_rows'] != 2000 or config['pruning_seed'] != 2026:
        raise ValueError('The frozen screening worker requires pruning_rows=2000 and pruning_seed=2026.')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Provide distinct sampling seeds.')
    if any(type(seed) is not int or seed < 0 or seed >= 2**32 for seed in config['seeds']):
        raise ValueError('Seeds must be nonnegative 32-bit integers.')
    if not math.isfinite(config['seconds']) or not 0 < config['seconds'] <= 1800:
        raise ValueError('seconds must be positive and at most 1800.')
    if not 0 < config['correlation_threshold'] <= 1:
        raise ValueError('correlation_threshold must be in (0, 1].')
    if (config['representatives_a'] != 'first_in_original_column_order'
            or config['representatives_b'] != 'last_in_original_column_order'):
        raise ValueError('Supported representative policies are first for A and last for B.')


def prepare(output, config, source):
    """Freeze feature choices, code, and identical rows before examining scores."""
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is nonempty but has no manifest; choose a fresh --output directory.')
    source_manifest = json.loads((source / 'manifest.json').read_text())
    pools = pd.read_pickle(source / 'development_pools.pkl')
    columns = pools['grow'].columns
    if columns[-1] != 'attack_cat' or not columns.is_unique:
        raise ValueError('Expected unique columns with attack_cat last.')
    for name in ['grow', 'prune', 'validation']:
        frame = pools[name]
        if not frame.columns.equals(columns) or frame.isna().any().any():
            raise ValueError('All development partitions must have matching columns and no missing values.')
        if len(frame) != source_manifest['partition_rows'][name]:
            raise ValueError(f'{name} row count disagrees with the source manifest.')
    if config['training_rows'] > len(pools['grow']):
        raise ValueError('training_rows exceeds the source growth pool.')
    if config['pruning_rows'] > len(pools['prune']):
        raise ValueError('pruning_rows exceeds the source pruning pool.')
    selection = choose_features(pools['grow'], config['correlation_threshold'])
    if not selection['groups']:
        raise ValueError('No qualifying groups were found; the proposed ablation is empty.')

    output.mkdir(parents=True, exist_ok=True)
    code = output / 'code'
    code.mkdir()
    # Use exactly the implementation used in the earlier experiment, not
    # whichever package source happens to be checked out when a worker starts.
    shutil.copytree(source / 'code/src', code / 'src',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source / 'code/run_screening.py', code / 'run_screening.py')
    shutil.copy2(__file__, code / 'ablation_run.py')
    for filename in ['pyproject.toml', 'uv.lock']:
        if (source / 'code' / filename).exists():
            shutil.copy2(source / 'code' / filename, code / filename)
    for variant, features in selection['feature_sets'].items():
        directory = output / 'variants' / variant
        for subdirectory in ['jobs', 'models', 'logs']:
            (directory / subdirectory).mkdir(parents=True, exist_ok=True)
        selected = features + ['attack_cat']
        filtered = {name: pools[name].loc[:, selected].copy()
                    for name in ['grow', 'prune', 'validation']}
        # Keep row order and targets identical; only predictor columns change.
        for name in filtered:
            assert filtered[name]['attack_cat'].equals(pools[name]['attack_cat'])
            assert filtered[name].index.equals(pools[name].index)
        pd.to_pickle(filtered, directory / 'development_pools.pkl')
    jobs = [dict(id=f'{variant}-{seed}', variant=variant, method='tree',
                 training_rows=config['training_rows'], seed=seed, n_estimators=1)
            for seed in config['seeds'] for variant in VARIANTS]
    write_json(output / 'feature_sets.json', selection)
    pd.DataFrame(selection['strong_pairs']).to_csv(output / 'correlations.csv', index=False)
    write_json(output / 'plan.json', dict(jobs=jobs))
    write_json(output / 'manifest.json', dict(
        config=config, source_experiment=str(source), source_manifest=source_manifest,
        source_pools_sha256=checksum(source / 'development_pools.pkl'),
        python=sys.version, feature_counts={key: len(value) for key, value in selection['feature_sets'].items()},
        input_hashes={variant: checksum(output / 'variants' / variant / 'development_pools.pkl')
                      for variant in VARIANTS},
        notes='Correlation selection uses only growth predictors, with first/last column-order '
              'representatives fixed before fitting. Rows and splits are shared between variants. '
              'No final test data is loaded. Raw and pruned tree results come from each fit.'))


def tree_statistics(model):
    """Count stored decision nodes, leaves, and maximum edge depth."""
    stack = [(model['tree'], 0)]
    decisions = leaves = depth = 0
    while stack:
        node, level = stack.pop()
        depth = max(depth, level)
        if isinstance(node, dict):
            decisions += 1
            stack.extend((child, level + 1) for child in next(iter(node.values())).values())
        else:
            leaves += 1
    return dict(decision_nodes=decisions, leaf_count=leaves, tree_depth=depth)


def collect(output):
    """Save all scores and within-seed differences against the all-feature fit."""
    records = []
    class_records = []
    manifest = json.loads((output / 'manifest.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())['jobs']
    for job in plan:
        path = output / 'variants' / job['variant'] / 'jobs' / (job['id'] + '.json')
        if not path.exists():
            continue
        checkpoint = json.loads(path.read_text())
        for result in checkpoint.get('results', []):
            record = {key: value for key, value in result.items()
                      if not isinstance(value, (list, dict))}
            record.update(feature_set=job['variant'], job_id=job['id'],
                          feature_count=manifest['feature_counts'][job['variant']])
            suffix = '-pruned' if result['method'] == 'pruned_tree' else ''
            model_path = output / 'variants' / job['variant'] / 'models' / (job['id'] + suffix + '.pkl')
            with model_path.open('rb') as file:
                record.update(tree_statistics(pickle.load(file)))
            record['model_bytes'] = model_path.stat().st_size
            records.append(record)
            class_records.extend(dict(feature_set=job['variant'], seed=result['seed'],
                                      method=result['method'], class_id=label, **values)
                                 for label, values in result['per_class'].items())
    if not records:
        return
    results = pd.DataFrame(records)
    baseline = results[results.feature_set == 'all_features'].set_index(['seed', 'method'])
    for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'fit_seconds',
                   'prediction_seconds', 'decision_nodes', 'leaf_count', 'tree_depth', 'model_bytes']:
        results[metric + '_delta'] = [
            row[metric] - baseline.loc[(row['seed'], row['method']), metric]
            if (row['seed'], row['method']) in baseline.index else float('nan')
            for row in records]
    results.to_csv(output / 'summary.csv', index=False)
    classes = pd.DataFrame(class_records)
    baseline_classes = classes[classes.feature_set == 'all_features'].set_index(['seed', 'method', 'class_id'])
    for metric in ['recall', 'f1']:
        classes[metric + '_delta'] = [
            row[metric] - baseline_classes.loc[(row['seed'], row['method'], row['class_id']), metric]
            if (row['seed'], row['method'], row['class_id']) in baseline_classes.index else float('nan')
            for row in class_records]
    classes.to_csv(output / 'per_class.csv', index=False)
    grouped = results.groupby(['feature_set', 'method']).agg(
        completed=('seed', 'count'), accuracy_mean=('accuracy', 'mean'),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        fit_seconds_mean=('fit_seconds', 'mean'), decision_nodes_mean=('decision_nodes', 'mean'),
        model_bytes_mean=('model_bytes', 'mean'))
    grouped.to_csv(output / 'aggregate.csv')
    differences = results[results.feature_set != 'all_features'].dropna(subset=['macro_f1_delta'])
    lines = ['# Correlation-ablation results', '',
             'All values below are tuning-set scores. The reserved final test set was not scored.', '',
             'See `summary.csv` for every completed model, `aggregate.csv` for descriptive means, '
             'and `feature_sets.json` for exact retained features. Group means may cover different '
             'seeds when jobs are incomplete; use matched differences for comparisons.', '',
             '## Matched differences from all features', '',
             'Positive accuracy/F1 differences favour the reduced set. Negative fit-time '
             'differences mean faster training. Differences are in fractional score units, not percentages.', '',
             '| Feature set | Tree | Pairs | Mean delta accuracy | Mean delta macro-F1 | '
             'Mean delta balanced accuracy | Mean delta fit seconds |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for (variant, method), group in differences.groupby(['feature_set', 'method']):
        values = [f'{group[metric + "_delta"].mean():+.4f}'
                  for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'fit_seconds']]
        lines.append('| ' + ' | '.join([variant, method, str(len(group))] + values) + ' |')
    lines += ['', 'Compare signs across individual seeds and per-class recall in the job JSON '
              'files. Small or inconsistent changes are inconclusive. Even consistent changes '
              'apply to these feature sets, tree implementation, and small training samples; '
              'they do not establish individual feature redundancy or ensemble performance.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))


def run(output, config, deadline, progress=True):
    """Run each fit in an isolated subprocess, enforcing global and job limits."""
    jobs = json.loads((output / 'plan.json').read_text())['jobs']
    environment = dict(os.environ, PYTHONPATH=str(output / 'code/src'))
    attempted = sum((output / 'variants' / job['variant'] / 'jobs' /
                     (job['id'] + '.json')).exists() for job in jobs)
    with ProgressBar(len(jobs), 'Feature ablation', progress) as bar:
        for job in jobs:
            directory = output / 'variants' / job['variant']
            checkpoint = directory / 'jobs' / (job['id'] + '.json')
            if checkpoint.exists():
                continue
            if deadline - time.monotonic() < 2:
                break
            write_json(output / 'status.json', dict(status='running', current_job=job,
                                                   attempted=attempted, planned=len(jobs)))
            bar.update(attempted, detail=job['id'] + ' starting', force=True)
            job_deadline = min(deadline, time.monotonic() + config['job_timeout'])
            progress_path = directory / 'jobs' / (job['id'] + '.progress.json')
            with (directory / 'logs' / (job['id'] + '.log')).open('w') as log:
                process = subprocess.Popen(
                    [sys.executable, str(output / 'code/run_screening.py'),
                     '--output', str(directory), '--worker', json.dumps(job)],
                    env=environment, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        remaining = job_deadline - time.monotonic()
                        if remaining <= 0:
                            process.kill()
                            process.wait()
                            write_json(checkpoint, dict(status='timeout', job=job))
                            break
                        state = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                        detail = job['id'] + ' | ' + state.get('phase', 'loading')
                        if state.get('phase') == 'training':
                            done, total = state.get('rows_completed', 0), state.get('rows_total', 1)
                            filled = int(10 * done / max(1, total))
                            detail += ' [' + '#' * filled + '-' * (10 - filled) + f'] {done}/{total}'
                        detail += ' | left ' + format_duration(deadline - time.monotonic())
                        bar.update(attempted, detail=detail)
                        try:
                            process.wait(timeout=min(0.25, remaining))
                        except subprocess.TimeoutExpired:
                            pass
                    if not checkpoint.exists():
                        write_json(checkpoint, dict(status='failed', job=job,
                                                    returncode=process.returncode))
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            attempted += 1
            collect(output)
            bar.update(attempted, detail=job['id'] + ' ' +
                       json.loads(checkpoint.read_text())['status'], force=True)
    outcomes = []
    for job in jobs:
        path = output / 'variants' / job['variant'] / 'jobs' / (job['id'] + '.json')
        if path.exists():
            outcomes.append(json.loads(path.read_text())['status'])
    write_json(output / 'status.json', dict(
        status='finished', planned=len(jobs), completed=outcomes.count('complete'),
        timeouts=outcomes.count('timeout'), failed=outcomes.count('failed'),
        not_started=len(jobs) - len(outcomes)))
    collect(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--prepare-only', action='store_true', help='Freeze and inspect feature sets without training.')
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    validate_config(config)
    output = args.output.resolve()
    source = (args.config.resolve().parent / config['source_experiment']).resolve()
    # Reserve two seconds for final checkpoint/summary writes. Preparation is
    # included in the budget; interrupted/resumed invocations get a fresh cap.
    deadline = started + config['seconds'] - 2
    if (output / 'manifest.json').exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['config'] != config or manifest['source_experiment'] != str(source):
            raise ValueError('Configuration differs from the frozen run; choose a fresh --output directory.')
        for variant, expected in manifest['input_hashes'].items():
            if checksum(output / 'variants' / variant / 'development_pools.pkl') != expected:
                raise ValueError('Prepared input changed; choose a fresh --output directory.')
    else:
        prepare(output, config, source)
    print('Feature choices:', output / 'feature_sets.json', flush=True)
    print('Results:', output, flush=True)
    if args.prepare_only:
        return
    try:
        run(output, config, deadline, progress=not args.no_progress)
    except KeyboardInterrupt:
        collect(output)
        write_json(output / 'status.json', dict(status='interrupted'))
        print('\nStopped. Resume unattempted jobs with the same command.')
    except Exception as error:
        write_json(output / 'status.json', dict(status='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
