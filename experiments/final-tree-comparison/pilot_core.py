"""Fixed-population balancing, timed SAMME rounds, and runtime forecasts."""

from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import pickle
import random
import resource
import time

import numpy as np
import pandas as pd

from assignment4.dec_tree.reduced_error_pruning import reduced_error_pruning
from assignment4.ensemble._common import VotingEnsemble, validate_training
from assignment4.progress import ProgressBar, format_duration
from overnight_core import TreeTrainer, EnsembleEvaluationCache, save_model, tree_shape
from run_screening import evaluate, predict_tree, write_json


def validate_config(config):
    """Reject unsupported settings rather than silently changing the experiment."""
    if (config['criterion'] != 'entropy' or config['pruning_objective'] != 'accuracy'
            or config['prune_ensemble_members'] or config['runtime_pilot']['evaluate_final_test']):
        raise ValueError('Pilot requires information gain, accuracy pruning, and no final-test access.')
    lambdas = config['balancing_lambdas']
    if not lambdas or len(set(lambdas)) != len(lambdas) or any(not 0 <= x <= 1 for x in lambdas):
        raise ValueError('Balancing strengths must be distinct values between zero and one.')
    for value in [config['training_draws'], config['runtime_pilot']['initial_members'],
                  *config['ensemble_checkpoints'], *config['runtime_pilot']['adaptive_boosting_members']]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Sample sizes and member counts must be positive integers.')
    if config['runtime_pilot']['job_timeout_seconds'] <= 0:
        raise ValueError('Job timeout must be positive.')
    if config['runtime_pilot']['slowdown_ratio'] <= 1:
        raise ValueError('Slowdown ratio must exceed one.')
    if config['runtime_pilot']['contingency_fraction'] < 0:
        raise ValueError('Contingency must be nonnegative.')
    if not config['runtime_pilot']['sequential_jobs']:
        raise ValueError('The runtime pilot requires sequential execution.')
    if config['tree_max_depth'] is not None or config['bagging_max_depth'] is not None:
        raise ValueError('Standalone and bagging trees must have unrestricted depth.')
    if any(not isinstance(d, int) or isinstance(d, bool) or d < 0
           for d in config['boosting_max_depths']):
        raise ValueError('Boosting depths must be nonnegative integers.')
    if any(not math.isfinite(r) or r <= 0 for r in config['boosting_learning_rates']):
        raise ValueError('Boosting rates must be finite and positive.')


def load_pools(output, config):
    """Load trusted local development data and reject any additional partitions."""
    with (output / 'development_pools.pkl').open('rb') as file:
        pools = pickle.load(file)
    if set(pools) != {'grow', 'prune', 'validation'}:
        raise ValueError('Expected only growth, pruning, and validation partitions.')
    grow = pools['grow']
    classes = set(grow.iloc[:, -1])
    if len(grow.columns) - 1 != config['expected_predictors']:
        raise ValueError('Unexpected predictor count.')
    if not set(config['categorical']) <= set(grow.columns[:-1]):
        raise ValueError('Categorical predictors are missing.')
    for name, frame in pools.items():
        if (len(frame) != config['partition_rows'][name]
                or list(frame.columns) != list(grow.columns)
                or frame.isna().any().any() or set(frame.iloc[:, -1]) != classes):
            raise ValueError(f'Invalid partition: {name}')
    if config['training_draws'] != len(grow):
        raise ValueError('The pilot requires one full-growth-pool row budget.')
    return pools


def balanced_sample(frame, strength, n_rows, seed):
    """Draw interpolated class quotas; preserve original positional source IDs."""
    if frame.empty or not 0 <= strength <= 1 or n_rows < 1:
        raise ValueError('Invalid balancing inputs.')
    groups = list(frame.groupby(frame.columns[-1], sort=True))
    quotas = [n_rows * ((1 - strength) * len(group) / len(frame) + strength / len(groups))
              for _, group in groups]
    counts = [math.floor(value) for value in quotas]
    order = sorted(range(len(groups)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[:n_rows - sum(counts)]:
        counts[i] += 1
    rng = random.Random(seed)
    pieces = []
    for (_, group), count in zip(groups, counts):
        positions = (rng.sample(range(len(group)), count) if count <= len(group)
                     else rng.choices(range(len(group)), k=count))
        pieces.append(group.iloc[positions])
    return pd.concat(pieces).sample(frac=1, random_state=seed)


class Recorder:
    """Append durable non-overlapping phase timings and current worker progress."""

    def __init__(self, path, job):
        self.path = path
        self.started = time.monotonic()
        self.events = []
        self.state = {'job': job, 'accepted_members': 0}

    def progress(self, phase, **values):
        self.state.update(phase=phase, elapsed_seconds=time.monotonic() - self.started, **values)
        write_json(self.path / 'progress.json', self.state)

    @contextmanager
    def phase(self, name, **values):
        self.progress(name, **values)
        started = time.monotonic()
        try:
            yield
        finally:
            event = dict(phase=name, seconds=time.monotonic() - started, **values)
            self.events.append(event)
            with (self.path / 'timings.jsonl').open('a') as file:
                file.write(json.dumps(event) + '\n')


def timed_boosting(frame, trainer, *, n_estimators, learning_rate, seed, recorder,
                   checkpoint):
    """Mirror package SAMME exactly, exposing separate round-phase timings.

    Equivalence to assignment4.ensemble.boosting is checked on deterministic
    fixtures. No balancing occurs inside SAMME: frame is its fixed population.
    """
    classes = validate_training(frame, trainer, predict_tree, n_estimators)
    if len(classes) < 2:
        raise ValueError('Boosting requires at least two classes.')
    rng = random.Random(seed)
    with recorder.phase('boosting_setup'):
        rows = frame.iloc[:, :-1].to_dict(orient='records')
        targets = frame.iloc[:, -1].tolist()
        log_weights = [-math.log(len(frame))] * len(frame)
    ensemble = VotingEnsemble([], [], classes, frame.columns[:-1].tolist(), predict_tree)
    reason = 'member_limit'
    for round_number in range(1, n_estimators + 1):
        with recorder.phase('round_sampling', round=round_number):
            weights = [math.exp(value) for value in log_weights]
            indices = rng.choices(range(len(frame)), weights=weights, k=len(frame))
            sample = frame.iloc[indices].copy().reset_index(drop=True)
        with recorder.phase('induction', round=round_number):
            model = trainer(sample)
            # The generic callback receives a reset index. Restore source-pool
            # accounting separately from its unchanged tree induction.
            model['distinct_source_rows'] = int(frame.iloc[indices].index.nunique())
            model['distinct_population_rows'] = len(set(indices))
            model['sampling'] = 'samme_weighted_resampling'
        with recorder.phase('boosting_training_predictions', round=round_number):
            predictions = [predict_tree(model, row) for row in rows]
            if any(label not in classes for label in predictions):
                raise ValueError('Base learner predicted an unknown class.')
        with recorder.phase('boosting_weight_update', round=round_number):
            wrong = [a != b for a, b in zip(predictions, targets)]
            error = math.fsum(w for w, bad in zip(weights, wrong) if bad) / math.fsum(weights)
            if error >= 1 - 1 / len(classes):
                reason = 'rejected_learner'
            elif not any(wrong):
                ensemble.estimators = [model]
                ensemble.estimator_weights = [1.0]
                ensemble.estimator_errors = [0.0]
                reason = 'perfect_learner'
            else:
                wrong_logs = [v for v, bad in zip(log_weights, wrong) if bad]
                largest_wrong = max(wrong_logs)
                log_error = (largest_wrong
                             + math.log(math.fsum(math.exp(v - largest_wrong) for v in wrong_logs))
                             - math.log(math.fsum(weights)))
                alpha = learning_rate * (math.log1p(-error) - log_error + math.log(len(classes) - 1))
                if not math.isfinite(alpha):
                    raise ValueError('Nonfinite boosting weight.')
                ensemble.estimators.append(model)
                ensemble.estimator_weights.append(alpha)
                ensemble.estimator_errors.append(error)
                updated = [v + (alpha if bad else 0.0) for v, bad in zip(log_weights, wrong)]
                largest = max(updated)
                normaliser = math.log(math.fsum(math.exp(v - largest) for v in updated))
                log_weights = [(v - largest) - normaliser for v in updated]
        accepted = reason != 'rejected_learner'
        recorder.progress('round complete', round=round_number, accepted=accepted,
                          accepted_members=len(ensemble.estimators), weighted_error=error)
        with (recorder.path / 'rounds.jsonl').open('a') as file:
            file.write(json.dumps({'round': round_number, 'accepted': accepted,
                                   'weighted_error': error, 'reason': reason,
                                   'distinct_source_rows': model['distinct_source_rows'],
                                   'distinct_population_rows': model['distinct_population_rows']}) + '\n')
        if accepted:
            checkpoint(ensemble, round_number, force=reason == 'perfect_learner')
        if reason != 'member_limit':
            if not ensemble.estimators:
                raise ValueError('First boosting learner failed the SAMME error bound.')
            break
    return ensemble, reason


def worker(output, job, config):
    """Train one timed configuration and persist metrics at actual checkpoints."""
    path = output / 'jobs' / job['id']
    recorder = Recorder(path, job)
    with recorder.phase('loading'):
        pools = load_pools(output, config)
        raw = pools['grow'].reset_index(drop=True)
        classes = sorted(raw.iloc[:, -1].unique().tolist())
        numerical = [c for c in raw.columns[:-1] if c not in config['categorical']]
        domains = {c: raw[c].unique().tolist() for c in config['categorical']}
        trainer = TreeTrainer(numerical, domains, job['depth'], progress_path=str(path / 'progress.json'))
        caches = {split: EnsembleEvaluationCache(frame, classes)
                  for split, frame in [('grow', raw), ('validation', pools['validation'])]}
    drawn_counts = np.zeros(len(raw), dtype=np.int64)
    fitted = None
    if job['kind'] != 'bagging':
        with recorder.phase('initial_sampling'):
            fitted = balanced_sample(raw, job['lambda'], config['training_draws'], job['seed'])
            drawn_counts = np.bincount(fitted.index.to_numpy(), minlength=len(raw))
    last_checkpoint = None

    def score(model, label, round_number):
        nonlocal last_checkpoint
        file_path = path / f'{label}-{round_number:03}.pkl'
        with recorder.phase('saving', round=round_number):
            save_model(file_path, model)
        result = {'variant': label, 'round': round_number,
                  'model_bytes': file_path.stat().st_size,
                  'distinct_source_rows': int(np.count_nonzero(drawn_counts)),
                  'training_draws': int(drawn_counts.sum()),
                  'source_class_draws': {str(c): int(drawn_counts[raw.iloc[:, -1].to_numpy() == c].sum())
                                         for c in classes},
                  'actual_estimators': len(model.estimators) if isinstance(model, VotingEnsemble) else 1}
        for split, frame in [('grow', raw), ('validation', pools['validation'])]:
            with recorder.phase('evaluate_' + split, round=round_number):
                result[split] = (caches[split].evaluate(model) if isinstance(model, VotingEnsemble)
                                 else evaluate(model, frame, classes))
        with recorder.phase('evaluate_fit', round=round_number):
            result['fit'] = (caches['grow'].evaluate(model, row_counts=drawn_counts)
                             if isinstance(model, VotingEnsemble) else evaluate(model, fitted, classes))
        result['fit_scope'] = ('cumulative_member_draws' if job['kind'] == 'bagging'
                               else 'fixed_balanced_population')
        if isinstance(model, dict):
            result['nodes'], result['leaves'], result['depth'] = tree_shape(model['tree'])
        with recorder.phase('writing_metrics', round=round_number):
            write_json(path / f'{label}-{round_number:03}.json', result)
        last_checkpoint = round_number

    checkpoints = set(config['ensemble_checkpoints']) | {job['members']}

    def checkpoint(model, round_number, force=False):
        recorder.progress('member complete', accepted_members=len(model.estimators), round=round_number)
        if force or round_number in checkpoints:
            score(model, job['kind'], round_number)

    reason = 'member_limit'
    if job['kind'] == 'tree':
        with recorder.phase('induction', round=1):
            model = trainer(fitted)
        score(model, 'raw', 1)
        with recorder.phase('pruning'):
            pruned = dict(model, tree=reduced_error_pruning(model['tree'], fitted, pools['prune']))
        score(pruned, 'pruned', 1)
    elif job['kind'] == 'bagging':
        model = VotingEnsemble([], [], raw.iloc[:, -1].unique().tolist(), raw.columns[:-1].tolist(), predict_tree)
        for round_number in range(1, job['members'] + 1):
            with recorder.phase('round_sampling', round=round_number):
                sample = balanced_sample(raw, job['lambda'], config['training_draws'],
                                         job['seed'] + round_number - 1)
                drawn_counts += np.bincount(sample.index.to_numpy(), minlength=len(raw))
            with recorder.phase('induction', round=round_number):
                member = trainer(sample)
                member['sampling_seed'] = job['seed'] + round_number - 1
                model.estimators.append(member)
                model.estimator_weights.append(1.0)
            checkpoint(model, round_number)
    else:
        model, reason = timed_boosting(fitted, trainer, n_estimators=job['members'],
                                      learning_rate=job['learning_rate'], seed=job['seed'],
                                      recorder=recorder, checkpoint=checkpoint)
        if last_checkpoint != recorder.state['round']:
            # A rejected final attempt adds no member, but retain the accepted prefix.
            score(model, 'boosting', recorder.state['round'])
    elapsed = time.monotonic() - recorder.started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux KiB -> bytes.
    totals = {}
    for event in recorder.events:
        totals[event['phase']] = totals.get(event['phase'], 0.0) + event['seconds']
    write_json(path / 'result.json', {
        'status': 'complete', 'job': job, 'stop_reason': reason,
        'wall_seconds': elapsed, 'phase_seconds': totals,
        'unattributed_seconds': max(0.0, elapsed - sum(totals.values())),
        'peak_rss_bytes': peak, 'final_test_loaded': False,
        'accepted_members': len(model.estimators) if isinstance(model, VotingEnsemble) else 1,
    })
    recorder.progress('complete')


def make_job(kind, strength, config, depth=None, rate=None, members=None):
    """Create stable IDs, including the requested pilot length."""
    members = members or (1 if kind == 'tree' else config['runtime_pilot']['initial_members'])
    key = f'{kind}_lambda{strength:g}_depth{depth}_rate{rate}_n{members}'
    return {'id': key, 'kind': kind, 'lambda': strength, 'depth': depth,
            'learning_rate': rate, 'members': members, 'seed': config['runtime_pilot']['seed']}


def initial_jobs(config):
    """Order standalone fits first, then bagging, then the boosting grid."""
    jobs = [make_job(kind, strength, config) for kind in ('tree', 'bagging')
            for strength in config['balancing_lambdas']]
    jobs.extend(make_job('boosting', strength, config, depth, rate)
                for strength in config['balancing_lambdas']
                for depth in config['boosting_max_depths']
                for rate in config['boosting_learning_rates'])
    return jobs


def completed_results(output):
    return [value for path in sorted((output / 'jobs').glob('*/result.json'))
            if (value := json.loads(path.read_text())).get('status') == 'complete']


def events_for(output, job):
    return [json.loads(line) for line in
            (output / 'jobs' / job['id'] / 'timings.jsonl').read_text().splitlines()]


ROUND_PHASES = {'round_sampling', 'induction', 'boosting_training_predictions', 'boosting_weight_update'}


def round_costs(events):
    costs = {}
    for event in events:
        if event['phase'] in ROUND_PHASES and 'round' in event:
            costs[event['round']] = costs.get(event['round'], 0.0) + event['seconds']
    return costs


def run(output, config, execute, deadline):
    """Run the initial grid and deterministic adaptive checks; restart partial fits."""
    jobs = initial_jobs(config)
    timeout = config['runtime_pilot']['job_timeout_seconds']
    handled = []
    requested = list(jobs)
    def processed(bar, job):
        bar.advance()
        result_path = output / 'jobs' / job['id'] / 'result.json'
        status = json.loads(result_path.read_text())['status'] if result_path.exists() else 'deferred'
        print(f'Pilot jobs processed {bar.completed}/{bar.total}'
              f' elapsed {format_duration(time.monotonic() - bar.started)}'
              f" | {job['id']} | {status}", flush=True)

    # Keep counting without timed renders, carriage returns, or start/end lines.
    with ProgressBar(len(jobs), 'Pilot jobs processed', False) as bar:
        for job in jobs:
            execute(output, job, timeout, deadline, bar)
            handled.append(job['id'])
            summarize(output, config)
            processed(bar, job)
        # Later checks deliberately restart, preserving independent 5/10/25 runs.
        for strength in config['balancing_lambdas']:
            candidates = [r for r in completed_results(output)
                          if r['job']['kind'] == 'boosting' and r['job']['lambda'] == strength
                          and r['job']['members'] == config['runtime_pilot']['initial_members']
                          and r['stop_reason'] == 'member_limit']
            if not candidates:
                continue
            slowest = max(candidates, key=lambda r: np.mean(list(round_costs(events_for(output, r['job'])).values())))
            base = slowest['job']
            for length in config['runtime_pilot']['adaptive_boosting_members']:
                if length <= base['members']:
                    continue
                extension = make_job('boosting', strength, config, base['depth'], base['learning_rate'], length)
                requested.append(extension)
                bar.total += 1
                success = execute(output, extension, timeout, deadline, bar)
                handled.append(extension['id'])
                summarize(output, config)
                processed(bar, extension)
                if not success:
                    break
                result = json.loads((output / 'jobs' / extension['id'] / 'result.json').read_text())
                costs = list(round_costs(events_for(output, extension)).values())
                if result['stop_reason'] != 'member_limit' or len(costs) < 10:
                    break
                ratio = np.mean(costs[-5:]) / max(np.mean(costs[:5]), 1e-12)
                if ratio <= config['runtime_pilot']['slowdown_ratio']:
                    break
    summarize(output, config)
    done = {r['job']['id'] for r in completed_results(output)}
    write_json(output / 'status.json', {'initial_jobs': len(jobs),
               'initial_complete': sum(j['id'] in done for j in jobs),
               'completed_jobs': len(done), 'considered': handled,
               'incomplete_jobs': [j['id'] for j in requested if j['id'] not in done],
               'final_test_loaded': False})
    print(f'Results and forecasts: {output}', flush=True)
    return all(j['id'] in done for j in requested)


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, config):
    """Forecast the fixed grid, disclosing incomplete coverage and extrapolation."""
    results = completed_results(output)
    summary = []
    timing_rows = []
    best = {}
    for result in results:
        job = result['job']
        events = events_for(output, job)
        for event in events:
            timing_rows.append({'job_id': job['id'], 'phase': event['phase'],
                                'round': event.get('round', ''), 'seconds': event['seconds']})
        summary.append({**job, 'stop_reason': result['stop_reason'],
                        'accepted_members': result['accepted_members'],
                        'wall_seconds': result['wall_seconds'],
                        'process_wall_seconds': result.get('process_wall_seconds', ''),
                        'peak_rss_bytes': result['peak_rss_bytes']})
        key = (job['kind'], job['lambda'], job['depth'], job['learning_rate'])
        if key not in best or job['members'] > best[key]['job']['members']:
            best[key] = result
    forecasts = []
    for result in best.values():
        job = result['job']
        events = events_for(output, job)
        costs = list(round_costs(events).values())
        measured_checkpoints = len({e['round'] for e in events if e['phase'] == 'saving'})
        scoring = sum(e['seconds'] for e in events if e['phase'].startswith('evaluate_')
                      or e['phase'] == 'writing_metrics')
        saving = sum(e['seconds'] for e in events if e['phase'] == 'saving')
        # Incremental cached predictions scale with members; metric aggregation
        # scales with checkpoint count. The conservative forecast charges all
        # measured scoring at every checkpoint to cover that uncertainty.
        overhead = result['wall_seconds'] - sum(costs) - scoring - saving
        for members in ([1] if job['kind'] == 'tree' else config['ensemble_checkpoints']):
            if job['kind'] == 'tree':
                central = result.get('process_wall_seconds', result['wall_seconds'])
                conservative = central
            else:
                checkpoints = [n for n in config['ensemble_checkpoints'] if n <= members]
                measured_members = max(1, result['accepted_members'])
                scale = members / measured_members
                check_scale = len(checkpoints) / max(1, measured_checkpoints)
                # Model serialization grows roughly with the number of members.
                saved_rounds = sum(e['round'] for e in events if e['phase'] == 'saving')
                save_scale = sum(checkpoints) / max(1, saved_rounds)
                startup = max(0.0, result.get('process_wall_seconds', result['wall_seconds']) - result['wall_seconds'])
                central = (overhead + startup + np.mean(costs) * members
                           + scoring * max(scale, check_scale) + saving * save_scale)
                conservative = (overhead + startup + max(costs) * members
                                + scoring * max(scale, len(checkpoints))
                                + saving * max(save_scale, scale * len(checkpoints)))
            conservative = max(central, conservative) * (1 + config['runtime_pilot']['contingency_fraction'])
            forecasts.append({'kind': job['kind'], 'lambda': job['lambda'], 'depth': job['depth'],
                              'learning_rate': job['learning_rate'], 'members': members,
                              'observed_members': result['accepted_members'],
                              'observed_round_seconds_mean': float(np.mean(costs)),
                              'observed_round_seconds_min': float(min(costs)),
                              'observed_round_seconds_max': float(max(costs)),
                              'observed_round_seconds_sd': float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
                              'stop_reason': result['stop_reason'],
                              'central_seconds': float(central), 'planning_seconds': float(conservative),
                              'peak_rss_bytes_observed': result['peak_rss_bytes'],
                              'later_rounds_unverified': job['kind'] == 'boosting' and members > job['members']})
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'timings.csv', timing_rows)
    write_csv(output / 'projections.csv', forecasts)
    expected = len(initial_jobs(config))
    complete = len({(j['kind'], j['lambda'], j['depth'], j['learning_rate']) for j in initial_jobs(config)} & set(best))
    totals = []
    for seeds in config['runtime_pilot']['forecast_seed_counts']:
        chosen = [r for r in forecasts if r['kind'] == 'tree' or r['members'] == max(config['ensemble_checkpoints'])]
        totals.append({'seeds': seeds, 'central_hours': sum(r['central_seconds'] for r in chosen) * seeds / 3600,
                       'planning_hours': sum(r['planning_seconds'] for r in chosen) * seeds / 3600,
                       'coverage_complete': complete == expected})
    write_csv(output / 'total_projections.csv', totals)
    lines = ['# Runtime pilot forecast', '', f'Configuration coverage: {complete}/{expected}.', '',
             'Totals with incomplete coverage are partial, not full-run estimates.',
             'Tree costs include both raw and pruned results. Ensemble costs include checkpoint evaluation and saving.',
             'Adaptive extensions restart from round one; all pilot work remains in summary.csv.',
             'Only the longest completed run per configuration informs forecasts.',
             'Early-stopped boosting is projected as a hypothetical full-length run, not a promised attainable ensemble size.',
             'Central projections scale cached evaluation by member/checkpoint counts; planning projections use slower rounds,',
             'more conservative checkpoint overhead, and the configured contingency.',
             'Memory values are observed peaks, not 50-member peak-memory predictions.',
             'Single-seed timing variation does not quantify variability across seeds or system load.', '',
             '| Seeds | Central hours | Planning hours | Full coverage |',
             '| ---: | ---: | ---: | :--- |']
    lines.extend(f"| {r['seeds']} | {r['central_hours']:.2f} | {r['planning_hours']:.2f} | {r['coverage_complete']} |" for r in totals)
    (output / 'FORECAST.md').write_text('\n'.join(lines) + '\n')
