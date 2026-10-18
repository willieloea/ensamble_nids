"""Importable model adapters, checkpointed workers and overnight reporting."""
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from assignment4.dec_tree.id3 import id3
from assignment4.dec_tree.info_metrics import (
    information_gains, gini_information_gains, gain_ratios,
)
from assignment4.dec_tree.reduced_error_pruning import reduced_error_pruning
from assignment4.ensemble import bagging, boosting, stacking
from assignment4.ensemble._common import VotingEnsemble, validate_training
from assignment4.progress import ProgressBar, format_duration
from run_screening import stratified_sample, predict_tree, evaluate, metrics, write_json

CRITERIA = {'entropy': information_gains, 'gain_ratio': gain_ratios,
            'gini': gini_information_gains}


def configured_criteria(config):
    """Validate explicit scorer names; older configurations retain entropy only."""
    criteria = config.get('criteria', ['entropy'])
    if (not isinstance(criteria, list) or not criteria
            or any(not isinstance(name, str) or name not in CRITERIA for name in criteria)
            or len(set(criteria)) != len(criteria)):
        raise ValueError(f'criteria must be a nonempty list of unique names from {list(CRITERIA)}')
    return criteria


def save_model(path, model):
    """Atomically replace a trusted local pickle checkpoint."""
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def balanced_sample(df, seed):
    """Equal class quotas at a fixed row budget; only supplied rows are sampled."""
    groups = list(df.groupby(df.columns[-1], sort=True))
    rng = random.Random(seed)
    pieces = []
    for i, (_, group) in enumerate(groups):
        count = len(df) // len(groups) + (i < len(df) % len(groups))
        positions = (rng.choices(range(len(group)), k=count) if count > len(group)
                     else rng.sample(range(len(group)), count))
        pieces.append(group.iloc[positions])
    # Keep source indices for distinct-row accounting; ID3 does not use them.
    return pd.concat(pieces).sample(frac=1, random_state=seed)


def counts(df):
    return {str(k): int(v) for k, v in df.iloc[:, -1].value_counts().items()}


@dataclass
class TreeTrainer:
    """Pickle-compatible tree fit callback; balancing stays inside stacking splits."""
    numerical: list
    domains: dict
    max_depth: int | None = None
    criterion: str = 'entropy'
    policy: str = 'natural'
    seed: int = 42
    progress_path: str | None = None
    bootstrap: bool = False
    record_fit: bool = False

    def __call__(self, sample):
        if self.criterion not in CRITERIA:
            raise ValueError(f'Unknown split criterion: {self.criterion!r}')
        if self.policy == 'balanced':
            fitted = balanced_sample(sample, self.seed)
        elif self.bootstrap:
            fitted = sample.sample(n=len(sample), replace=True, random_state=self.seed)
        else:
            fitted = sample
        metric = CRITERIA[self.criterion]
        last_update = -float('inf')
        def row_progress(completed, total):
            nonlocal last_update
            if self.progress_path and (time.monotonic() - last_update >= 2 or completed == total):
                path = Path(self.progress_path)
                state = json.loads(path.read_text()) if path.exists() else {}
                state.update(phase='training tree', tree_completed_rows=completed, tree_total_rows=total)
                write_json(path, state)
                last_update = time.monotonic()
        model = {
            'tree': id3(fitted, metric, domains=self.domains,
                        continuous_features=self.numerical, max_depth=self.max_depth,
                        progress=row_progress),
            'domains': self.domains,
            'fallback': fitted.iloc[:, -1].mode().iloc[0],
            'fit_rows': len(fitted), 'fit_class_counts': counts(fitted),
            'distinct_source_rows': int(fitted.index.nunique()),
            'criterion': self.criterion,
            'sampling_seed': self.seed,
            'sampling': ('equal_class' if self.policy == 'balanced'
                         else 'bootstrap' if self.bootstrap else 'none'),
        }
        if self.record_fit:
            model['fit_metrics'] = evaluate(model, fitted,
                                            sorted(sample.iloc[:, -1].unique().tolist()))
        return model


def balanced_bagging(df, train_model, *, n_estimators, random_state, on_estimator,
                     on_sample):
    """Fit fresh equal-quota samples from the full source pool for every member."""
    classes = validate_training(df, train_model, predict_tree, n_estimators)
    ensemble = VotingEnsemble([], [], classes,
                              df.columns[:-1].tolist(), predict_tree)
    with ProgressBar(n_estimators, 'Balanced bagging models', True) as bar:
        for i in range(n_estimators):
            sample = balanced_sample(df, random_state + i)
            member = train_model(sample)
            # IDs refer to positions in the original, unresampled growth subsample.
            member['fit_source_counts'] = {str(int(k)): int(v)
                                            for k, v in sample.index.value_counts().items()}
            member['sampling_seed'] = random_state + i
            member['sampling'] = 'equal_class'
            ensemble.estimators.append(member)
            ensemble.estimator_weights.append(1.0)
            on_sample(sample)
            bar.advance()
            on_estimator(ensemble, i + 1)
    return ensemble


def tree_shape(tree):
    """Return total nodes, leaves and maximum edge depth, including empty branches."""
    if not isinstance(tree, dict):
        return 1, 1, 0
    children = [tree_shape(child) for child in next(iter(tree.values())).values()]
    return (1 + sum(x[0] for x in children), sum(x[1] for x in children),
            1 + max(x[2] for x in children))


def split_diagnostics(tree):
    """Describe the root and counts of categorical/numerical internal nodes."""
    result = {'root_feature': None, 'root_split_type': 'leaf',
              'categorical_splits': 0, 'numerical_splits': 0}
    if not isinstance(tree, dict):
        return result
    root = next(iter(tree))
    result['root_feature'] = root[0] if isinstance(root, tuple) else root
    result['root_split_type'] = 'numerical' if isinstance(root, tuple) else 'categorical'
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            split, branches = next(iter(node.items()))
            key = 'numerical_splits' if isinstance(split, tuple) else 'categorical_splits'
            result[key] += 1
            pending.extend(branches.values())
    return result


def validate_pools(output, config):
    configured_criteria(config)
    with (output / 'development_pools.pkl').open('rb') as file:
        pools = pickle.load(file)
    if set(pools) != {'grow', 'prune', 'validation'}:
        raise ValueError('Expected only growth, pruning and validation pools; no final test data.')
    grow = pools['grow']
    if len(grow.columns) - 1 != config['expected_predictors']:
        raise ValueError('Unexpected predictor count; all cleaned predictors are required.')
    if not set(config['categorical']) <= set(grow.columns[:-1]):
        raise ValueError('Categorical predictors are missing.')
    classes = set(grow.iloc[:, -1])
    for name, frame in pools.items():
        if list(frame.columns) != list(grow.columns) or frame.isna().any().any():
            raise ValueError(f'{name}: inconsistent columns or missing values.')
        if set(frame.iloc[:, -1]) != classes:
            raise ValueError(f'{name}: missing target classes.')
    return pools


class EnsembleEvaluationCache:
    """Reuse immutable member predictions on one fixed evaluation DataFrame.

    Normal append-only trajectories add only new votes. Replacements or changed
    weights rebuild votes from cached predictions, preserving member order and
    first-class tie resolution. The cache belongs to a worker, never the model.
    Prediction callbacks must be deterministic and members must not be mutated.
    """

    def __init__(self, frame, classes):
        self.frame = frame
        self.classes = classes
        self.targets = frame.iloc[:, -1].tolist()
        self._signature = None
        self._callback = None
        self._predictions = {}
        self._members = []
        self._weights = []

    def evaluate(self, ensemble, row_counts=None):
        """Return exact classification metrics and incremental scoring timings."""
        started = time.monotonic()
        signature = (tuple(ensemble.feature_names), tuple(ensemble.classes))
        if self._signature != signature or self._callback is not ensemble.predict_model:
            self._signature = signature
            self._callback = ensemble.predict_model
            self._rows = self.frame.loc[:, ensemble.feature_names].to_dict(orient='records')
            self._row_positions = np.arange(len(self._rows))
            self._class_positions = {label: i for i, label in enumerate(ensemble.classes)}
            self._predictions = {}
            self._members = []
            self._weights = []
            self._votes = np.zeros((len(self._rows), len(ensemble.classes)), dtype=float)
        prefix_length = len(self._members)
        if (len(ensemble.estimators) < prefix_length
                or any(old is not new for old, new in zip(self._members, ensemble.estimators))
                or self._weights != ensemble.estimator_weights[:prefix_length]):
            self._members = []
            self._weights = []
            self._votes.fill(0)
            prefix_length = 0
        newly_scored = 0
        for member, weight in zip(ensemble.estimators[prefix_length:],
                                  ensemble.estimator_weights[prefix_length:]):
            key = id(member)
            if key not in self._predictions:
                predictions = []
                for row in self._rows:
                    label = ensemble.predict_model(member, row)
                    if label not in self._class_positions:
                        raise ValueError(f'Base learner predicted unknown class: {label!r}')
                    predictions.append(self._class_positions[label])
                # Retain the object too, preventing reuse of its Python object ID.
                self._predictions[key] = (member, np.asarray(predictions, dtype=np.intp))
                newly_scored += 1
            positions = self._predictions[key][1]
            self._votes[self._row_positions, positions] += weight
            self._members.append(member)
            self._weights.append(weight)
        predictions = [ensemble.classes[i] for i in self._votes.argmax(axis=1)]
        targets = self.targets
        if row_counts is not None:
            if (len(row_counts) != len(targets)
                    or any(not isinstance(count, (int, np.integer)) or count < 0 for count in row_counts)
                    or sum(row_counts) == 0):
                raise ValueError('row_counts must contain nonnegative integer counts with positive total')
            # Expand labels only, never feature rows or model predictions.
            targets = [target for target, count in zip(targets, row_counts) for _ in range(count)]
            predictions = [label for label, count in zip(predictions, row_counts) for _ in range(count)]
        result = metrics(targets, predictions, self.classes)
        result.update(prediction_seconds=time.monotonic() - started,
                      validation_rows=len(targets), evaluation_mode='cached_members',
                      new_estimators_scored=newly_scored)
        return result


def worker(output, job, config):
    """Fit one job; save every accepted member before expensive scoring."""
    job_id = job['id']
    state_path = output / 'progress' / f'{job_id}.json'
    started = time.monotonic()
    overhead = 0.0
    last_round = 0
    last_scored = None
    state = {'job': job, 'phase': 'loading', 'completed_models': 0}

    def progress(phase, **values):
        state.update(phase=phase, wall_seconds=time.monotonic() - started, **values)
        write_json(state_path, state)

    progress('loading')
    pools = validate_pools(output, config)
    grow, validation = pools['grow'], pools['validation']
    raw = stratified_sample(grow, job['n_train'], job['seed']).reset_index(drop=True)
    shared_balanced_fit = job['policy'] == 'balanced' and job['kind'] in ('tree', 'boosting')
    fitted = balanced_sample(raw, job['seed']) if shared_balanced_fit else raw
    fit_draw_counts = np.zeros(len(raw), dtype=np.int64)
    shared_fit_counts = np.bincount(fitted.index.to_numpy(), minlength=len(raw))
    numerical = [c for c in grow.columns[:-1] if c not in config['categorical']]
    domains = {c: grow[c].unique().tolist() for c in config['categorical']}
    classes = sorted(grow.iloc[:, -1].unique().tolist())
    criterion = job.get('criterion', 'entropy')
    trainer = TreeTrainer(numerical, domains, job.get('max_depth'),
                          criterion=criterion, seed=job['seed'], progress_path=str(state_path))
    metadata = dict(job, actual_train_rows=len(raw), train_class_counts=counts(raw),
                    criterion=criterion,
                    fit_class_counts=counts(fitted), distinct_fit_source_rows=int(fitted.index.nunique()),
                    validation_rows=len(validation), predictors=list(raw.columns[:-1]))
    evaluation_caches = {split: EnsembleEvaluationCache(frame, classes)
                         for split, frame in [('train', raw), ('validation', validation)]}

    def score(model, variant, round_number=0):
        nonlocal overhead, last_scored
        tick = time.monotonic()
        fit_seconds = tick - started - overhead
        point_id = f'{job_id}__{variant}__{round_number:03}'
        progress('scoring ' + variant, round=round_number)
        model_path = output / 'models' / f'{point_id}.pkl'
        save_model(model_path, model)
        result = dict(metadata, point_id=point_id, variant=variant,
                      round=round_number, fit_seconds=fit_seconds,
                      actual_estimators=len(model.estimators) if hasattr(model, 'estimators') else 1,
                      model_file=str(model_path.relative_to(output)), model_bytes=model_path.stat().st_size)
        if isinstance(model, dict):
            result['nodes'], result['leaves'], result['tree_depth'] = tree_shape(model['tree'])
            result.update(split_diagnostics(model['tree']))
            result['criterion'] = model.get('criterion', criterion)
        for split, frame in [('train', raw), ('validation', validation)]:
            result[split] = (evaluation_caches[split].evaluate(model)
                             if isinstance(model, VotingEnsemble) else evaluate(model, frame, classes))
        result['fit'] = None
        result['fit_evaluation_scope'] = 'not_recorded'
        if shared_balanced_fit:
            if isinstance(model, VotingEnsemble):
                result['fit'] = evaluation_caches['train'].evaluate(model, row_counts=shared_fit_counts)
            else:
                result['fit'] = evaluate(model, fitted, classes)
            result['fit_evaluation_scope'] = 'shared_balanced_training_population'
        elif job['policy'] == 'balanced' and job['kind'] == 'bagging':
            result['fit'] = evaluation_caches['train'].evaluate(model, row_counts=fit_draw_counts)
            result['fit_evaluation_scope'] = 'cumulative_member_fit_draws'
            result['distinct_fit_source_rows'] = int(np.count_nonzero(fit_draw_counts))
            result['fit_class_counts'] = {str(label): int(fit_draw_counts[
                raw.iloc[:, -1].to_numpy() == label].sum()) for label in classes}
        elif job['policy'] == 'balanced' and isinstance(model, dict) and 'fit_metrics' in model:
            result['fit'] = model['fit_metrics']
            result['fit_evaluation_scope'] = 'own_fitted_rows'
            result['fit_class_counts'] = model['fit_class_counts']
            result['distinct_fit_source_rows'] = model['distinct_source_rows']
        if result['fit'] is not None:
            result['fit_rows'] = sum(entry['support'] for entry in result['fit']['per_class'].values())
        result['score_seconds'] = time.monotonic() - tick
        write_json(output / 'points' / f'{point_id}.json', result)
        collect(output)
        overhead += time.monotonic() - tick
        last_scored = round_number
        progress('checkpoint saved', round=round_number)

    def checkpoint(ensemble, round_number):
        nonlocal overhead, last_round
        tick = time.monotonic()
        last_round = round_number
        save_model(output / 'models' / f'{job_id}__latest.pkl', ensemble)
        progress('training', round=round_number, completed_models=len(ensemble.estimators))
        overhead += time.monotonic() - tick
        if round_number in config['checkpoints']:
            score(ensemble, job['kind'], round_number)

    progress('training')
    if job['kind'] == 'tree':
        model = trainer(fitted)
        score(model, 'raw')
        if job.get('stage') != 'depth':
            progress('pruning')
            prune = stratified_sample(pools['prune'], config['pruning_rows'], 2026)
            tick = time.monotonic()
            pruned = dict(model, tree=reduced_error_pruning(model['tree'], fitted, prune))
            prune_seconds = time.monotonic() - tick
            # Keep pruning separate from raw induction time in the point metadata.
            metadata['prune_seconds'] = prune_seconds
            score(pruned, 'pruned')
    elif job['kind'] in ('bagging', 'boosting'):
        trainer.max_depth = job.get('max_depth')
        arguments = dict(n_estimators=config['n_estimators'], random_state=job['seed'],
                         progress=True, on_estimator=checkpoint)
        if job['kind'] == 'boosting':
            arguments['learning_rate'] = job['learning_rate']
        if job['kind'] == 'bagging' and job['policy'] == 'balanced':
            def record_sample(sample):
                indices, counts_ = np.unique(sample.index.to_numpy(), return_counts=True)
                fit_draw_counts[indices] += counts_
            metadata['fit_class_counts'] = None
            metadata['distinct_fit_source_rows'] = None
            model = balanced_bagging(raw, trainer, n_estimators=config['n_estimators'],
                                     random_state=job['seed'], on_estimator=checkpoint,
                                     on_sample=record_sample)
        else:
            method = bagging if job['kind'] == 'bagging' else boosting
            model = method(fitted, trainer, predict_tree, **arguments)
        if last_scored != last_round:
            score(model, job['kind'], last_round)
    elif job['kind'] == 'stacking':
        # Never balance the input before splitting: duplicates could cross the holdout.
        learners = {
            name: (TreeTrainer(numerical, domains, depth, 'entropy', job['policy'],
                               job['seed'] + i, str(state_path), bootstrap=True,
                               record_fit=job['policy'] == 'balanced'), predict_tree)
            for i, (name, depth) in enumerate([('entropy', None), ('depth3', 3), ('depth8', 8)])
        }
        meta_domains = {f'base_{i}_class_{j}': [0, 1]
                        for i in range(len(learners)) for j in range(len(classes))}
        meta = TreeTrainer([], meta_domains, 3, policy=job['policy'], seed=job['seed'],
                           progress_path=str(state_path), record_fit=job['policy'] == 'balanced')
        model = stacking(raw, learners, meta, predict_tree,
                         holdout_fraction=config['stack_holdout_fraction'],
                         random_state=job['seed'], progress=True)
        metadata['fit_class_counts'] = None
        metadata['distinct_fit_source_rows'] = None
        metadata['stack_fit_details'] = {
            name: {k: value[k] for k in ['fit_rows', 'fit_class_counts', 'distinct_source_rows',
                                        'sampling', 'sampling_seed']}
            for name, value in model.estimators.items()}
        metadata['stack_fit_details']['meta'] = {
            k: model.meta_estimator[k] for k in ['fit_rows', 'fit_class_counts', 'distinct_source_rows']}
        if job['policy'] == 'balanced':
            metadata['stack_fit_details']['meta']['fit_metrics'] = model.meta_estimator['fit_metrics']
        # These are the exact fitted constituents, not separately refitted trees.
        for name, estimator in model.estimators.items():
            score(estimator, 'base_' + name)
        vote = VotingEnsemble(list(model.estimators.values()), [1.0] * len(learners),
                              model.classes, model.feature_names, predict_tree)
        score(vote, 'majority_vote')
        score(model, 'stacking')
    else:
        raise ValueError('Unknown job kind')
    progress('complete')


def make_job(kind, n_train, seed, policy='natural', max_depth=None, learning_rate=None,
             stage='core', criterion='entropy'):
    if criterion not in CRITERIA and not (kind == 'stacking' and criterion == 'mixed'):
        raise ValueError(f'Unknown split criterion: {criterion!r}')
    depth_label = 'full' if max_depth is None else str(max_depth)
    rate_label = '' if learning_rate is None else f'_lr{learning_rate}'
    return dict(id=f'{kind}_n{n_train}_s{seed}_{policy}_d{depth_label}{rate_label}_{criterion}',
                kind=kind, n_train=n_train, seed=seed, policy=policy, max_depth=max_depth,
                learning_rate=learning_rate, stage=stage, criterion=criterion)


def plan(config, main_rows):
    """Put matched small-sample comparisons before optional expensive extensions."""
    seeds = config['seeds']
    criteria = configured_criteria(config)
    # Complete paired unrestricted baselines before expanding the other sweeps.
    jobs = [make_job('tree', main_rows, seed, policy, stage='criterion', criterion=criterion)
            for seed in seeds for policy in ['natural', 'balanced'] for criterion in criteria]
    jobs += [make_job('tree', n, seed, stage='learning', criterion=criterion)
             for n in config['learning_sizes'] if n <= main_rows for seed in seeds
             for criterion in criteria]
    jobs += [make_job('tree', main_rows, seed, policy, depth, stage='depth', criterion=criterion)
             for depth in config['depths'] for seed in seeds for policy in ['natural', 'balanced']
             for criterion in criteria]
    # Interleave methods and policies within each seed so partial batches remain useful.
    for seed in seeds:
        for policy in ['natural', 'balanced']:
            jobs.extend(make_job('bagging', main_rows, seed, policy, stage='ensemble', criterion=criterion)
                        for criterion in criteria)
            jobs.extend(make_job('boosting', main_rows, seed, policy, config['boost_depth'], rate,
                                 stage='ensemble', criterion=criterion)
                        for rate in config['learning_rates'] for criterion in criteria)
    jobs += [make_job('stacking', main_rows, seed, policy, stage='stacking', criterion='entropy')
             for seed in seeds for policy in ['natural', 'balanced']]
    jobs += [make_job('tree', n, seed, stage='larger_learning', criterion=criterion)
             for n in config['learning_sizes'] if n > main_rows for seed in seeds
             for criterion in criteria]
    # The main-size learning point is the same fit as its criterion baseline.
    unique = {}
    for job in jobs:
        unique.setdefault(job['id'], job)
    return list(unique.values())


def archive_previous_attempt(output, job_id, previous):
    """Preserve a previous attempt before restarting, keeping active points separate."""
    root = output / 'attempts' / job_id
    prior_numbers = [int(path.name) for path in root.glob('*') if path.name.isdigit()]
    prior_attempt = max([int(previous.get('attempt', 0)), *prior_numbers], default=0)
    paths = []
    for name in ['points', 'models', 'progress', 'logs', 'jobs']:
        folder = output / name
        prefix = job_id + ('__' if name in ['points', 'models'] else '.')
        paths.extend(path for path in folder.iterdir() if path.name.startswith(prefix))
    if paths:
        prior_attempt = max(1, prior_attempt)
        # A numbered directory may already exist after an interrupted archival.
        while (root / str(prior_attempt)).exists():
            prior_attempt += 1
        archive = root / str(prior_attempt)
        archive.mkdir(parents=True)
        for path in paths:
            destination = archive / path.parent.name / path.name
            destination.parent.mkdir(exist_ok=True)
            path.rename(destination)
        (archive / 'README.md').write_text(
            'Historical attempt; excluded from current summaries.\n'
            'Model paths in point files are relative to this attempt directory.\n')
    return prior_attempt + 1


def execute_job(output, job, config, deadline, completed=0, total=1):
    """Run or defer one job; only completed jobs are skipped on subsequent runs."""
    status_path = output / 'jobs' / (job['id'] + '.json')
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get('status') == 'complete':
        return 'already_complete'
    limit = config['tree_timeout_seconds'] if job['kind'] == 'tree' else config['ensemble_timeout_seconds']
    remaining = deadline - time.monotonic() - 30
    if remaining < limit:
        print(f'Deferred {job["id"]}: {format_duration(max(0, remaining))} available; '
              f'full timeout is {format_duration(limit)}.', flush=True)
        return 'deferred'
    attempt = archive_previous_attempt(output, job['id'], previous)
    job = dict(job, attempt=attempt)
    started = time.monotonic()
    status = dict(job=job, status='running', timeout_seconds=limit, attempt=attempt)
    write_json(status_path, status)
    write_json(output / 'status.json', dict(status, considered_jobs=completed, planned_jobs=total))
    collect(output)
    print(f'\n[{completed + 1}/{total}] {job["id"]}, attempt {attempt}; budget remaining '
          f'{format_duration(max(0, deadline - time.monotonic()))}', flush=True)
    process = None

    def stop_process():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    try:
        with (output / 'logs' / (job['id'] + '.log')).open('w') as log:
            process = subprocess.Popen([sys.executable, str(output / 'code/run.py'),
                                        '--output', str(output), '--worker', json.dumps(job)],
                                       stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                state_file = output / 'progress' / (job['id'] + '.json')
                state = json.loads(state_file.read_text()) if state_file.exists() else {}
                detail = f'{state.get("phase", "starting")} round={state.get("round", 0)}'
                if 'tree_total_rows' in state:
                    detail += f' tree rows={state["tree_completed_rows"]}/{state["tree_total_rows"]}'
                print(f'\r  {detail}; job {format_duration(elapsed)} / '
                      f'{format_duration(limit)}', end='', flush=True)
                if elapsed >= limit:
                    stop_process()
                    status['status'] = 'timeout'
                    break
                try:
                    # Wake immediately when a short worker finishes, retaining live updates.
                    process.wait(timeout=min(5, max(.001, limit - elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            else:
                status['status'] = 'complete' if process.returncode == 0 else 'failed'
    except KeyboardInterrupt:
        stop_process()
        status['status'] = 'interrupted'
        raise
    except Exception as error:
        stop_process()
        status.update(status='failed', error=repr(error))
        print(f'\nJob failed: {error}', flush=True)
    finally:
        status['wall_seconds'] = time.monotonic() - started
        write_json(status_path, status)
        print(flush=True)
        collect(output)
    return status['status']


def run(output, config, deadline):
    """Prevent concurrent training invocations from retrying each other's jobs."""
    import fcntl
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A training runner is already active in this output directory.') from None
        _run(output, config, deadline)


def _run(output, config, deadline):
    """Consider each job once per invocation, deferring jobs without a full timeout."""
    validate_pools(output, config)
    attempted = set()
    handled = set()
    deferred = {}

    def execute(job, completed, total):
        if job['id'] in handled:
            return
        outcome = execute_job(output, job, config, deadline, completed, total)
        if outcome == 'deferred':
            deferred[job['id']] = job
        else:
            handled.add(job['id'])
            if outcome != 'already_complete':
                attempted.add(job['id'])
            deferred.pop(job['id'], None)

    def finish(reason, **values):
        write_json(output / 'deferred_jobs.json', list(deferred.values()))
        summarize(output, config)
        write_json(output / 'status.json', dict(phase='stopped', reason=reason,
                   deferred_jobs=len(deferred), attempted_this_invocation=len(attempted), **values))

    try:
        pilot_criterion = configured_criteria(config)[0]
        pilots = [make_job('tree', n, config['seeds'][0], stage='pilot', criterion=pilot_criterion)
                  for n in config['pilot_sizes']]
        for i, job in enumerate(pilots):
            execute(job, i, len(pilots))
        selection_path = output / 'selection.json'
        if selection_path.exists():
            main_rows = json.loads(selection_path.read_text())['main_rows']
        else:
            eligible = []
            finished = []
            for job in pilots:
                point = output / 'points' / f'{job["id"]}__raw__000.json'
                if point.exists():
                    seconds = json.loads(point.read_text())['fit_seconds']
                    finished.append(job['n_train'])
                    if seconds <= config['pilot_fit_limit_seconds']:
                        eligible.append(job['n_train'])
            if not finished:
                finish('insufficient time for pilot job' if deferred else 'no completed pilot fit')
                print('No pilot finished. Inspect status/logs; main sample size was not selected.')
                return
            main_rows = max(eligible) if eligible else min(finished)
            write_json(selection_path, dict(main_rows=main_rows,
                       rule='Largest pilot with raw induction <= fit limit; otherwise smallest completed pilot.',
                       fit_limit_seconds=config['pilot_fit_limit_seconds']))
        jobs = plan(config, main_rows)
        write_json(output / 'plan.json', jobs)
        with ProgressBar(len(jobs), 'Overnight queue considered', True) as bar:
            for i, job in enumerate(jobs):
                # Continue past long jobs: later tree jobs may still fit their full timeout.
                execute(job, i, len(jobs))
                bar.advance()
        finish('insufficient time for pending jobs' if deferred else 'plan exhausted',
               main_rows=main_rows, attempted_jobs=len(list((output / 'jobs').glob('*.json'))))
    except KeyboardInterrupt:
        write_json(output / 'deferred_jobs.json', list(deferred.values()))
        summarize(output, config)
        write_json(output / 'status.json', dict(phase='interrupted', results_preserved=True))
        print('Stopped. Completed points and model checkpoints were preserved.', flush=True)


def collect(output):
    """Refresh flat CSVs from atomic point files, including timed-out trajectories."""
    rows, per_class, points = [], [], []
    for path in sorted((output / 'points').glob('*.json')):
        try:
            point = json.loads(path.read_text())
        except FileNotFoundError:
            # A retry may have just archived this point during --summarize-only.
            continue
        point.setdefault('criterion', 'entropy')
        point.setdefault('attempt', 1)
        points.append(point)
        row = {key: point.get(key) for key in ['point_id', 'id', 'attempt', 'kind', 'stage', 'n_train',
               'actual_train_rows', 'seed', 'policy', 'criterion', 'max_depth', 'learning_rate', 'variant',
               'round', 'actual_estimators', 'fit_seconds', 'prune_seconds', 'score_seconds',
               'nodes', 'leaves', 'tree_depth', 'root_feature', 'root_split_type',
               'categorical_splits', 'numerical_splits', 'model_bytes',
               'distinct_fit_source_rows', 'fit_evaluation_scope', 'fit_rows', 'model_file']}
        status_file = output / 'jobs' / (point['id'] + '.json')
        row['job_status'] = (json.loads(status_file.read_text())['status']
                             if status_file.exists() else 'running')
        for split in ['train', 'validation', 'fit']:
            entry = point.get(split)
            for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'prediction_seconds']:
                row[f'{split}_{key}'] = entry[key] if entry else None
            row[f'{split}_error'] = 1 - entry['accuracy'] if entry else None
            row[f'{split}_evaluation_mode'] = entry.get('evaluation_mode', 'full') if entry else None
            row[f'{split}_new_estimators_scored'] = entry.get('new_estimators_scored') if entry else None
            for label, values in (entry['per_class'].items() if entry else []):
                per_class.append(dict(point_id=point['point_id'], split=split, label=label, **values))
        rows.append(row)
    def write_csv(name, values, empty_header):
        path = output / name
        # Concurrent --summarize-only must not share a temporary file with a worker.
        with tempfile.NamedTemporaryFile('w', newline='', dir=output, delete=False,
                                         prefix=name + '.', suffix='.tmp') as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=list(values[0]) if values else empty_header)
            writer.writeheader()
            writer.writerows(values)
        temporary.replace(path)
    write_csv('summary.csv', rows, ['point_id', 'validation_macro_f1'])
    write_csv('per_class.csv', per_class, ['point_id', 'split', 'label', 'support', 'precision', 'recall', 'f1'])
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for column in ['fit_accuracy', 'fit_macro_f1', 'fit_balanced_accuracy',
                       'fit_prediction_seconds', 'fit_error']:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    write_criterion_comparison(output, frame)
    return frame, points


def write_criterion_comparison(output, frame):
    """Pair completed entropy/gain-ratio points on seed and all other settings."""
    keys = ['kind', 'n_train', 'actual_train_rows', 'seed', 'policy', 'max_depth',
            'learning_rate', 'variant', 'round']
    metrics = ['train_error', 'validation_error', 'train_macro_f1', 'validation_macro_f1',
               'validation_accuracy', 'validation_balanced_accuracy', 'fit_seconds']
    columns = ['point_id', *keys, *metrics]
    if frame.empty:
        paired = pd.DataFrame(columns=keys + ['point_id_entropy', 'point_id_gain_ratio'])
    else:
        # Fixed mixed-criterion stacks are a separate experiment, not criterion pairs.
        eligible = frame[frame.kind.isin(['tree', 'bagging', 'boosting'])]
        entropy_points = eligible[eligible.criterion == 'entropy'][columns]
        ratio_points = eligible[eligible.criterion == 'gain_ratio'][columns]
        paired = entropy_points.merge(ratio_points, on=keys, suffixes=('_entropy', '_gain_ratio'),
                                      validate='one_to_one')
        for metric in metrics:
            paired[f'{metric}_delta'] = paired[f'{metric}_gain_ratio'] - paired[f'{metric}_entropy']
    with tempfile.NamedTemporaryFile('w', dir=output, delete=False,
                                     prefix='criterion_comparison.', suffix='.tmp') as file:
        temporary = Path(file.name)
        paired.to_csv(file, index=False)
    temporary.replace(output / 'criterion_comparison.csv')


def plots(output, frame):
    """Plot matched training/validation curves; variability is descriptive across seeds."""
    if frame.empty:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    def curve(data, x_column, title, name, labels=None, logarithmic=False):
        if data.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for axis, metric, ylabel in zip(axes, ['error', 'macro_f1'], ['Classification error', 'Macro-F1']):
            lines = [('train', '#2471a3'), ('validation', '#c0392b')]
            if data[f'fit_{metric}'].notna().any():
                lines.append(('fit', '#239b56'))
            for split, color in lines:
                groups = data.groupby(x_column)[f'{split}_{metric}']
                mean, std, count = groups.mean(), groups.std().fillna(0), groups.count()
                x = mean.index.to_numpy(dtype=float)
                axis.plot(x, mean.values, 'o-', color=color,
                          label='Fit population' if split == 'fit' else split.capitalize())
                axis.fill_between(x, np.clip(mean.values - std.values, 0, 1),
                                  np.clip(mean.values + std.values, 0, 1), alpha=.12, color=color)
                if split == 'validation':
                    for a, b, n in zip(x, mean, count):
                        axis.annotate(f'n={n}', (a, b), xytext=(0, 9),
                                      textcoords='offset points', ha='center', fontsize=8)
            axis.set_ylim(-.02, 1.05)
            axis.set_ylabel(ylabel)
            axis.set_xlabel(x_column.replace('_', ' ').capitalize())
            axis.grid(alpha=.2)
            axis.legend()
            if logarithmic:
                axis.set_xscale('log')
            if labels:
                axis.set_xticks(list(labels), list(labels.values()), rotation=30)
        fig.suptitle(title + '\nMean ± sample SD across available seeds; n = seeds per point', fontsize=10)
        fig.tight_layout()
        fig.savefig(output / 'plots' / (name + '.png'), dpi=160)
        plt.close(fig)

    # IDs are unique; the pilot and main plan reuse the same unrestricted fit.
    trees = frame[frame.kind == 'tree']
    for criterion, criterion_trees in trees.groupby('criterion'):
        for variant in ['raw', 'pruned']:
            data = criterion_trees[(criterion_trees.policy == 'natural')
                                   & criterion_trees.max_depth.isna()
                                   & (criterion_trees.variant == variant)]
            curve(data, 'n_train', f'{criterion}: natural distribution, {variant} learning curve',
                  f'learning_{variant}_{criterion}', logarithmic=True)
    selection = output / 'selection.json'
    if selection.exists():
        main_rows = json.loads(selection.read_text())['main_rows']
        for (policy, criterion), group in trees.groupby(['policy', 'criterion']):
            data = group[(group.n_train == main_rows) & (group.variant == 'raw')].copy()
            full_x = max(config_depths := json.loads((output / 'config.json').read_text())['depths']) + 4
            data['depth_limit'] = data.max_depth.fillna(full_x)
            labels = {d: str(d) for d in config_depths}
            labels[full_x] = 'Unrestricted'
            curve(data, 'depth_limit', f'Tree depth: {criterion}, {policy}, {main_rows:,} source rows',
                  f'depth_{policy}_{criterion}', labels=labels)
    ensemble = frame[frame.kind.isin(['bagging', 'boosting'])].copy()
    ensemble['depth_group'] = ensemble.max_depth.fillna(-1)
    ensemble['rate_group'] = ensemble.learning_rate.fillna(-1)
    for key, data in ensemble.groupby(['kind', 'n_train', 'policy', 'depth_group', 'rate_group', 'criterion']):
        kind, n_train, policy, depth, rate, criterion = key
        name = f'{kind}_n{n_train}_{policy}_d{depth}_lr{rate}_{criterion}'
        curve(data, 'round', f'{kind.capitalize()}: {criterion}, {policy}, n={n_train}, depth={depth}, rate={rate}', name)
    stack = frame[frame.kind == 'stacking']
    for policy, data in stack.groupby('policy'):
        groups = data.groupby('variant').validation_macro_f1
        mean, std = groups.mean(), groups.std().fillna(0)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(mean.index, mean, yerr=std, capsize=4, color='#2471a3')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Validation macro-F1')
        ax.set_title(f'Stacking and its exact constituents: {policy}\nMean ± sample SD across available seeds')
        ax.tick_params(axis='x', rotation=20)
        fig.tight_layout()
        fig.savefig(output / 'plots' / f'stacking_{policy}.png', dpi=160)
        plt.close(fig)


def summarize(output, config):
    """Regenerate tables, plots and a short interpretation guide without fitting."""
    frame, points = collect(output)
    plots(output, frame)
    statuses = [json.loads(p.read_text()) for p in (output / 'jobs').glob('*.json')]
    status_counts = {}
    for status in statuses:
        key = status['status']
        status_counts[key] = status_counts.get(key, 0) + 1
    text = ['# Overnight learning curves', '',
            f'Saved score points: {len(points)}. Attempted job statuses: {status_counts}.', '',
            f'Configured split criteria: {", ".join(configured_criteria(config))}.', '',
            'All cleaned predictors are used. The reserved final test set is never loaded or scored.', '',
            'See `summary.csv`, `per_class.csv`, `points/`, `models/`, `logs/`, and `plots/`.', '',
            'Split criteria are kept separate in curves and aggregates. '
            '`criterion_comparison.csv` pairs information-gain (entropy) and gain-ratio points '
            'on seed and all other settings. Deltas are gain ratio minus information gain: '
            'positive F1/accuracy deltas and negative error deltas favour gain ratio. '
            'Only completed pairs appear; missing partners are not treated as zero. '
            'This is ID3-style induction with gain ratio, not the complete C4.5 algorithm.', '',
            'Training metrics use the original, unresampled growth subsample. Validation uses the full '
            'fixed development validation pool. Pruning has its own pool. Equal-quota balancing keeps '
            'the fitting row budget fixed but changes both class proportions and distinct-row coverage.', '',
            'Balanced models also report fit-population metrics: the shared balanced population for '
            'single trees/boosting, cumulative member fitting draws for bagging, and own fitting rows '
            'for stacking constituents. Combined stacking has no single fit-population score; its '
            'meta-learner fit score is recorded separately. See fit_evaluation_scope.', '',
            'Depth curves can reveal underfitting/overfitting. Sample-size curves show data efficiency. '
            'Ensemble curves compare prefixes of the same trajectory. Bagging need not show increasing '
            'validation error. These curves do not estimate a mathematical bias/variance decomposition.', '',
            'Bands are descriptive sample standard deviations across sampling seeds, not confidence '
            'intervals: all seeds share validation observations. Labels show available repeats. '
            'Partial runs are included; missing points are never replaced by zero.', '',
            'Boosting uses weighted resampling (SAMME), with depth-limited trees. Perfect learners '
            'can end training early and replace earlier members. The round and actual member count '
            'are both recorded. Balanced training need not reduce unweighted source training error.', '',
            'Stacking uses a stratified base/meta holdout inside the growth subsample. Resampling '
            'happens within each fitting callback. Its training score is descriptive because source '
            'rows train one layer or the other. Compare the exact fitted constituents, their majority '
            'vote and stack together; their fit-data budgets differ from full-data single trees.', '',
            'Macro-F1 is the primary tuning metric; inspect minority-class recall and accuracy too. '
            'Any best result below is selected using validation and is not an unbiased final-test score.', '']
    if not frame.empty:
        best = frame.sort_values('validation_macro_f1', ascending=False).head(10)
        text += ['## Highest validation macro-F1 points', '',
                 '| Point | Validation macro-F1 | Train macro-F1 | Validation accuracy |',
                 '| --- | ---: | ---: | ---: |']
        for _, row in best.iterrows():
            text.append(f'| {row.point_id} | {row.validation_macro_f1:.4f} | '
                        f'{row.train_macro_f1:.4f} | {row.validation_accuracy:.4f} |')
        # Aggregate only identical conditions; retain counts to expose incomplete comparisons.
        aggregate = frame.copy()
        aggregate['max_depth'] = aggregate.max_depth.fillna('unrestricted')
        aggregate['learning_rate'] = aggregate.learning_rate.fillna('unused')
        aggregate.groupby(['kind', 'n_train', 'policy', 'criterion', 'max_depth', 'learning_rate', 'variant', 'round'],
                          dropna=False)[['train_error', 'validation_error', 'fit_error', 'train_macro_f1',
                                       'fit_macro_f1',
                                       'validation_macro_f1', 'validation_balanced_accuracy']].agg(
                                           ['mean', 'std', 'count']).to_csv(output / 'aggregate.csv')
    (output / 'RESULTS.md').write_text('\n'.join(text) + '\n')
