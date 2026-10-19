"""Independent final tasks, resumable boosting, and checked result exchange."""

import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import resource
import shutil
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd

from assignment4.dec_tree.reduced_error_pruning import reduced_error_pruning
from assignment4.ensemble._common import VotingEnsemble
from overnight_core import TreeTrainer, EnsembleEvaluationCache, save_model
from pilot_core import balanced_sample, load_pools, Recorder, timed_boosting, write_csv
from run_screening import evaluate, predict_tree, write_json


def validate(config):
    """Check the locked primary comparison and nonoverlapping sample streams."""
    if config['balancing_lambda'] != 0.5 or config['criterion'] != 'entropy':
        raise ValueError('This comparison requires lambda 0.5 and information gain.')
    seeds = config['seeds']
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError('At least two distinct matched seeds are required.')
    checkpoints = config['ensemble_checkpoints']
    if sorted(set(checkpoints)) != checkpoints or checkpoints[-1] != config['primary_members']:
        raise ValueError('Checkpoints must be unique, increasing, and end at the primary size.')
    used = set()
    for seed in seeds:
        stream = {member_seed(config, seed, i) for i in range(config['primary_members'])}
        if used & stream:
            raise ValueError('Bagging repetitions must not share member sampling seeds.')
        used |= stream


def member_seed(config, seed, index):
    return config['bagging_member_seed_bases'][str(seed)] + index


def jobs(config):
    """Assign CPU-intensive tasks once, across two fixed machine roles."""
    result = []
    for kind in ('boosting', 'tree'):
        for seed in config['seeds']:
            result.append({'id': f'{kind}-s{seed}', 'kind': kind, 'seed': seed, 'owner': 'vostro'})
    for index in range(config['primary_members']):
        for seed in config['seeds']:
            result.append({'id': f'bag-s{seed}-m{index + 1:03}', 'kind': 'bag_member',
                           'seed': seed, 'index': index,
                           'sampling_seed': member_seed(config, seed, index),
                           'owner': 'vostro' if index % 3 == 0 else 'thinkbook'})
    return result


def plan_id(output):
    return json.loads((output / 'manifest.json').read_text())['plan_id']


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(output, job, **details):
    path = output / 'tasks' / job['id']
    artifacts = {p.name: fingerprint(p) for p in path.glob('*.pkl')}
    write_json(path / 'result.json', {'status': 'complete', 'job': job,
               'plan_id': plan_id(output), 'artifacts': artifacts,
               'machine': platform.node(), **details})


def worker(output, job, config):
    """Fit a tree/member or continue an ensemble from its last accepted round."""
    path = output / 'tasks' / job['id']
    if job['owner'] == 'vostro':
        os.nice(10)  # Report editing and foreground applications keep priority.
    recorder = Recorder(path, job)
    with recorder.phase('loading'):
        pools = load_pools(output, config)
        raw = pools['grow'].reset_index(drop=True)
        domains = {c: raw[c].unique().tolist() for c in config['categorical']}
        numerical = [c for c in raw.columns[:-1] if c not in config['categorical']]
        depth = config['boosting_max_depth'] if job['kind'] == 'boosting' else None
        trainer = TreeTrainer(numerical, domains, depth, progress_path=str(path / 'progress.json'))
    sample_seed = job.get('sampling_seed', job['seed'])
    with recorder.phase('initial_sampling'):
        fitted = balanced_sample(raw, config['balancing_lambda'], config['training_draws'], sample_seed)
    if job['kind'] in ('tree', 'bag_member'):
        raw_model = path / 'raw.pkl'
        if raw_model.exists():
            with raw_model.open('rb') as file:
                model = pickle.load(file)
        else:
            with recorder.phase('induction'):
                model = trainer(fitted)
                model.update(sampling_seed=sample_seed, balancing_lambda=config['balancing_lambda'])
            with recorder.phase('saving'):
                save_model(raw_model, model)
        if job['kind'] == 'tree':
            with recorder.phase('pruning'):
                pruned = dict(model, tree=reduced_error_pruning(model['tree'], fitted, pools['prune']))
                save_model(path / 'pruned.pkl', pruned)
    else:
        state_path = path / 'state.pkl'
        resume = None
        if state_path.exists():
            with state_path.open('rb') as file:
                resume = pickle.load(file)
            if resume.get('plan_id') != plan_id(output) or resume.get('seed') != job['seed']:
                raise ValueError('Boosting checkpoint does not match this plan/seed.')

        def state_changed(state):
            state.update(plan_id=plan_id(output), seed=job['seed'])
            with recorder.phase('saving_state'):
                save_model(state_path, state)

        def checkpoint(model, round_number, force=False):
            if force or round_number in config['ensemble_checkpoints']:
                with recorder.phase('saving_checkpoint', round=round_number):
                    save_model(path / f'checkpoint-{round_number:03}.pkl', model)

        model, reason = timed_boosting(fitted, trainer,
            n_estimators=config['primary_members'], learning_rate=config['boosting_learning_rate'],
            seed=job['seed'], recorder=recorder, checkpoint=checkpoint,
            resume=resume, on_state=state_changed)
        # Handles a resumed state saved just before checkpoint serialization,
        # and the valid smaller ensemble after SAMME early stopping.
        with recorder.phase('saving_final'):
            save_model(path / 'primary.pkl', model)
        complete(output, job, stop_reason=reason, actual_members=len(model.estimators),
                 last_worker_seconds=time.monotonic() - recorder.started,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        recorder.progress('complete', accepted_members=len(model.estimators))
        return
    complete(output, job, last_worker_seconds=time.monotonic() - recorder.started,
             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    recorder.progress('complete')


def check_task(output, job):
    """Check model hashes and plan identity before reuse or aggregation."""
    path = output / 'tasks' / job['id']
    result = path / 'result.json'
    if not result.exists():
        return False
    value = json.loads(result.read_text())
    if value.get('status') != 'complete':
        return False
    if value['plan_id'] != plan_id(output) or value['job'] != job:
        raise ValueError(f"Task belongs to another plan: {job['id']}")
    for name, digest in value['artifacts'].items():
        if fingerprint(path / name) != digest:
            raise ValueError(f'Corrupted model artifact: {path / name}')
    return True


def import_pilot(output, source, config):
    """Reuse checked seed-42 pilot trees/members and reconstruct SAMME state."""
    if (output / 'pilot_import.json').exists():
        return
    old = json.loads((source / 'config.json').read_text())
    if (old['training_draws'] if 'training_draws' in old else config['training_draws']) != config['training_draws']:
        raise ValueError('Pilot row budget differs.')
    if fingerprint(source / 'development_pools.pkl') != fingerprint(output / 'development_pools.pkl'):
        raise ValueError('Pilot development data differs.')
    old_manifest = json.loads((source / 'manifest.json').read_text())
    # The timed runner gained continuation hooks; verify the actual induction
    # package and prediction/trainer adapters rather than presentation code.
    frozen = json.loads((output / 'manifest.json').read_text())
    for name, digest in old_manifest['sha256'].items():
        if name.startswith('code/src/') or name in ('code/run_screening.py', 'code/overnight_core.py'):
            if frozen['sha256'].get(name) != digest or fingerprint(source / name) != digest:
                raise ValueError(f'Pilot training dependency differs: {name}')
    seed = old['runtime_pilot']['seed']
    if seed not in config['seeds'] or 0.5 not in old['balancing_lambdas']:
        raise ValueError('Pilot settings cannot be reused.')
    imported = []
    all_jobs = jobs(config)

    def pilot_job(kind, depth=None, rate=None):
        matches = []
        for file in (source / 'jobs').glob('*/result.json'):
            value = json.loads(file.read_text())
            job = value.get('job', {})
            if (value.get('status') == 'complete' and job.get('kind') == kind
                    and job.get('lambda') == 0.5 and job.get('seed') == seed
                    and job.get('depth') == depth and job.get('learning_rate') == rate):
                matches.append((file.parent, value))
        return matches[0] if matches else (None, None)

    folder, _ = pilot_job('tree')
    if folder:
        job = next(j for j in all_jobs if j['kind'] == 'tree' and j['seed'] == seed)
        path = output / 'tasks' / job['id']
        path.mkdir(exist_ok=True)
        for name, prefix in [('raw.pkl', 'raw'), ('pruned.pkl', 'pruned')]:
            shutil.copy2(folder / f'{prefix}-001.pkl', path / name)
        complete(output, job, imported_from=str(folder), training_time_excluded=True)
        imported.append(job['id'])
    folder, _ = pilot_job('bagging')
    if folder:
        file = sorted(folder.glob('bagging-*.pkl'))[-1]
        with file.open('rb') as handle:
            ensemble = pickle.load(handle)
        for i, model in enumerate(ensemble.estimators[:config['primary_members']]):
            job = next(j for j in all_jobs if j['kind'] == 'bag_member' and j['seed'] == seed and j['index'] == i)
            if model['sampling_seed'] != job['sampling_seed']:
                raise ValueError('Pilot bagging sample stream differs.')
            path = output / 'tasks' / job['id']
            path.mkdir(exist_ok=True)
            save_model(path / 'raw.pkl', model)
            complete(output, job, imported_from=str(file), training_time_excluded=True)
            imported.append(job['id'])
    folder, result = pilot_job('boosting', config['boosting_max_depth'], config['boosting_learning_rate'])
    if folder and result['stop_reason'] == 'member_limit':
        file = sorted(folder.glob('boosting-*.pkl'))[-1]
        with file.open('rb') as handle:
            original = pickle.load(handle)
        job = next(j for j in all_jobs if j['kind'] == 'boosting' and j['seed'] == seed)
        path = output / 'tasks' / job['id']
        path.mkdir(exist_ok=True)
        raw = load_pools(output, config)['grow'].reset_index(drop=True)
        fitted = balanced_sample(raw, 0.5, config['training_draws'], seed)
        members = iter(original.estimators)
        recorder = Recorder(path, job)
        state = {}
        replay, _ = timed_boosting(fitted, lambda sample: next(members),
            n_estimators=len(original.estimators), learning_rate=config['boosting_learning_rate'],
            seed=seed, recorder=recorder, checkpoint=lambda *a, **kw: None,
            on_state=lambda value: state.update(value))
        if replay.estimator_weights != original.estimator_weights or replay.estimator_errors != original.estimator_errors:
            raise ValueError('Pilot SAMME replay differs; refusing to reuse.')
        state.update(plan_id=plan_id(output), seed=seed)
        save_model(path / 'state.pkl', state)
        save_model(path / f'checkpoint-{len(original.estimators):03}.pkl', replay)
        imported.append(job['id'] + ':state')
    write_json(output / 'pilot_import.json', {'source': str(source), 'reused': imported,
               'boosting_replayed_without_induction': True})
    print('Reused pilot artifacts: ' + ', '.join(imported), flush=True)


def export_results(output, destination, role, config):
    """Export only completed tasks owned by this machine role."""
    selected = [j for j in jobs(config) if j['owner'] == role and check_task(output, j)]
    with tempfile.TemporaryDirectory() as temporary:
        meta = Path(temporary) / 'transfer.json'
        write_json(meta, {'plan_id': plan_id(output), 'role': role, 'jobs': selected})
        with tarfile.open(destination, 'w:gz', compresslevel=1) as archive:
            archive.add(meta, arcname='transfer.json')
            for job in selected:
                archive.add(output / 'tasks' / job['id'], arcname='tasks/' + job['id'])
            machine = output / f'machine-{role}.json'
            if machine.exists():
                archive.add(machine, arcname=machine.name)
    print(f'Exported {len(selected)} complete tasks: {destination}')


def import_results(output, archive_path, config):
    """Validate provenance, ownership and hashes before importing remote tasks."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with tarfile.open(archive_path) as archive:
            archive.extractall(root, filter='data')
        transfer = json.loads((root / 'transfer.json').read_text())
        if transfer['plan_id'] != plan_id(output):
            raise ValueError('Transferred results belong to another frozen plan.')
        expected = {j['id']: j for j in jobs(config)}
        # Reuse check_task against the extracted layout and original plan.
        shutil.copy2(output / 'manifest.json', root / 'manifest.json')
        for job in transfer['jobs']:
            if job != expected.get(job['id']) or job['owner'] != transfer['role']:
                raise ValueError('Invalid transferred task ownership.')
            if not check_task(root, job):
                raise ValueError('Transfer includes an incomplete task.')
            if check_task(output, job):
                local = json.loads((output / 'tasks' / job['id'] / 'result.json').read_text())
                incoming = json.loads((root / 'tasks' / job['id'] / 'result.json').read_text())
                if local['artifacts'] != incoming['artifacts']:
                    raise ValueError('Conflicting completed results; refusing overwrite.')
                continue
            target = output / 'tasks' / job['id']
            if target.exists():
                raise ValueError(f'Refusing to overwrite partial local task: {target}')
            shutil.copytree(root / 'tasks' / job['id'], target)
        machine = root / f"machine-{transfer['role']}.json"
        if machine.exists():
            shutil.copy2(machine, output / machine.name)
    print(f"Imported {len(transfer['jobs'])} remote tasks.")


def collect(output, config, dataset=None, *, evaluate_test=False):
    """Build ordered ensemble prefixes and score development/frozen primary models."""
    all_jobs = jobs(config)
    available = {j['id'] for j in all_jobs if check_task(output, j)}
    missing = [j['id'] for j in all_jobs if j['id'] not in available]
    if evaluate_test and missing:
        raise ValueError(f'Primary comparison incomplete: {missing}')
    pools = load_pools(output, config)
    raw = pools['grow'].reset_index(drop=True)
    classes = sorted(raw.iloc[:, -1].unique().tolist())
    destination = output / 'evaluation'
    destination.mkdir(exist_ok=True)
    models = []

    def load(path):
        with path.open('rb') as file:
            return pickle.load(file)

    for seed in config['seeds']:
        tree_id = f'tree-s{seed}'
        if tree_id in available:
            for variant, name in [('unpruned', 'raw'), ('pruned', 'pruned')]:
                models.append((variant, seed, 1, load(output / 'tasks' / tree_id / f'{name}.pkl')))
        members = []
        for i in range(config['primary_members']):
            key = f'bag-s{seed}-m{i+1:03}'
            if key not in available:
                break
            members.append(load(output / 'tasks' / key / 'raw.pkl'))
            if i + 1 in config['ensemble_checkpoints']:
                ensemble = VotingEnsemble(list(members), [1.0] * len(members),
                    raw.iloc[:, -1].unique().tolist(), raw.columns[:-1].tolist(), predict_tree)
                models.append(('bagging', seed, i + 1, ensemble))
        key = f'boosting-s{seed}'
        if key in available:
            path = output / 'tasks' / key
            primary = load(path / 'primary.pkl')
            for checkpoint in config['ensemble_checkpoints']:
                file = path / f'checkpoint-{checkpoint:03}.pkl'
                if checkpoint == config['primary_members']:
                    continue
                if file.exists():
                    models.append(('boosting', seed, checkpoint, load(file)))
                elif len(primary.estimators) >= checkpoint:
                    prefix = VotingEnsemble(primary.estimators[:checkpoint],
                        primary.estimator_weights[:checkpoint], primary.classes,
                        primary.feature_names, primary.predict_model,
                        primary.estimator_errors[:checkpoint])
                    models.append(('boosting', seed, checkpoint, prefix))
            models.append(('boosting', seed, config['primary_members'], primary))
    if evaluate_test:
        if dataset is None:
            raise ValueError('--evaluate-test requires --dataset pointing to the original cleaned CSV.')
        source = json.loads((output / 'source_manifest.json').read_text())
        if fingerprint(dataset) != source['dataset_sha256']:
            raise ValueError('Original dataset checksum differs; refusing final-test evaluation.')
        # All settings were locked in config before this first access.
        table = pd.read_csv(dataset)
        indices = json.loads((output / 'split_indices.json').read_text())
        for name, frame in pools.items():
            pd.testing.assert_frame_equal(table.loc[indices[name]].reset_index(drop=True), frame)
        test = table.loc[indices['test']].reset_index(drop=True)
        if len(test) != config['partition_rows']['final_test']:
            raise ValueError('Unexpected final-test size.')
        pools = {'generalization': test}
        models = [m for m in models if m[0] in ('unpruned', 'pruned') or m[2] == config['primary_members']]
        lock = {'plan_id': plan_id(output), 'models': [(a, b, c) for a, b, c, _ in models],
                'note': 'All primary fitted models; no seed or setting selection on this set.'}
        write_json(output / 'generalization_access.json', lock)
    else:
        pools = {'grow': raw, 'validation': pools['validation']}
    rows = []
    per_class = []
    for method, seed, count, model in models:
        record = {'method': method, 'seed': seed, 'members': count,
                  'actual_members': len(model.estimators) if isinstance(model, VotingEnsemble) else 1}
        for split, frame in pools.items():
            value = evaluate(model, frame, classes)
            write_json(destination / f'{split}-{method}-s{seed}-n{count}.json', value)
            rows.append({**record, 'split': split, **{k: value[k] for k in
                         ('accuracy', 'macro_f1', 'balanced_accuracy', 'prediction_seconds')}})
            for label, scores in value['per_class'].items():
                per_class.append({**record, 'split': split, 'class': label, **scores})
        save_model(destination / f'{method}-s{seed}-n{count}.pkl', model)
    prefix = 'generalization' if evaluate_test else 'development'
    write_csv(output / f'{prefix}_summary.csv', rows)
    write_csv(output / f'{prefix}_per_class.csv', per_class)
    write_json(output / 'coverage.json', {'complete': len(available), 'expected': len(all_jobs), 'missing': missing})
    lines = [f'# {prefix.title()} comparison', '', f'Complete tasks: {len(available)}/{len(all_jobs)}.', '',
             '| Method | Split | Members | Mean macro-F1 | Sample SD across seeds | Seeds |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for method, split, count in sorted({(r['method'], r['split'], r['members']) for r in rows}):
        values = [r['macro_f1'] for r in rows if (r['method'], r['split'], r['members']) == (method, split, count)]
        sd = f'{np.std(values, ddof=1):.4f}' if len(values) > 1 else 'n/a'
        lines.append(f'| {method} | {split} | {count} | {np.mean(values):.4f} | {sd} | {len(values)} |')
    lines += ['', 'Seed variation uses the same evaluation observations; it is not a confidence interval.',
              'Growth scores use the natural growth pool, not an in-sample balanced fitting population.',
              'Machine-specific task timing logs include concurrent contention; imported pilot timings are excluded.']
    (output / f'{prefix.upper()}_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(f'{prefix} scores saved; task coverage {len(available)}/{len(all_jobs)}.', flush=True)
