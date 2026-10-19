"""Development-only bagging continuation and matched split-criterion ablation."""

import json
from pathlib import Path
import pickle
import resource
import shutil
import time

import numpy as np

from assignment4.dec_tree.reduced_error_pruning import reduced_error_pruning
from assignment4.ensemble._common import VotingEnsemble
from final_core import check_task, complete, fingerprint, jobs as original_jobs, worker as member_worker
from overnight_core import TreeTrainer, EnsembleEvaluationCache, save_model, tree_shape
from pilot_core import balanced_sample, load_pools, Recorder, write_csv
from run_screening import evaluate, predict_tree, write_json


def validate(config):
    """Require the agreed development-only follow-ups and independent streams."""
    if config['generalization_evaluation'] or config['balancing_lambda'] != 0.5:
        raise ValueError('Follow-ups require lambda 0.5 and development evaluation only.')
    if config['tree_criteria'] != ['entropy', 'gain_ratio']:
        raise ValueError('Expected information gain and gain ratio.')
    depth = config.get('tree_max_depth')
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
        raise ValueError('A bounded criterion comparison needs a positive integer depth.')
    if config['primary_members'] <= config['source_members']:
        raise ValueError('The new ensemble must extend the existing member count.')
    if config['ensemble_checkpoints'][-1] != config['primary_members']:
        raise ValueError('Last checkpoint must equal the target size.')
    streams = []
    for seed in config['seeds']:
        base = config['bagging_member_seed_bases'][str(seed)]
        stream = set(range(base, base + config['primary_members']))
        if any(stream & previous for previous in streams):
            raise ValueError('Member seed streams overlap between repetitions.')
        streams.append(stream)


def jobs(config):
    """Schedule the two new gain-ratio fits first, then independent members."""
    result = []
    for criterion in reversed(config['tree_criteria']):
        for seed in config['seeds']:
            result.append({'id': f'tree-{criterion}-s{seed}', 'kind': 'tree',
                           'criterion': criterion, 'seed': seed, 'owner': 'thinkbook'})
    if not config.get('include_bagging', True):
        return result
    for i in range(config['primary_members']):
        for seed in config['seeds']:
            result.append({'id': f'bag-s{seed}-m{i+1:03}', 'kind': 'bag_member',
                           'seed': seed, 'index': i,
                           'sampling_seed': config['bagging_member_seed_bases'][str(seed)] + i,
                           'owner': 'thinkbook'})
    return result


def import_baselines(output, source, config):
    """Copy exactly the original ordered member prefixes and entropy trees."""
    old = json.loads((source / 'config.json').read_text())
    for name in ('training_draws', 'balancing_lambda', 'seeds', 'categorical',
                 'expected_predictors', 'bagging_member_seed_bases'):
        if old[name] != config[name]:
            raise ValueError(f'Original setting differs: {name}')
    if old['primary_members'] != config['source_members'] or old['criterion'] != 'entropy':
        raise ValueError('Unexpected original ensemble size or split criterion.')
    if fingerprint(source / 'development_pools.pkl') != fingerprint(output / 'development_pools.pkl'):
        raise ValueError('Development data differs from the original experiment.')
    old_manifest = json.loads((source / 'manifest.json').read_text())
    new_manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in old_manifest['sha256'].items():
        if name.startswith('code/src/') or name in ('code/run_screening.py', 'code/overnight_core.py'):
            if new_manifest['sha256'].get(name) != digest or fingerprint(source / name) != digest:
                raise ValueError(f'Training dependency differs: {name}')
    required = [j for j in original_jobs(old) if j['kind'] in ('tree', 'bag_member')]
    if not all(check_task(source, job) for job in required):
        raise ValueError('Original standalone trees or bagging members are incomplete.')
    old_by_id = {j['id']: j for j in required}
    imported = []
    for job in jobs(config):
        if job['kind'] == 'tree':
            if job['criterion'] != 'entropy':
                continue
            if config.get('tree_max_depth') is not None:
                # Unrestricted entropy trees are not matched depth-cap baselines.
                continue
            original_id = f"tree-s{job['seed']}"
            files = ['raw.pkl', 'pruned.pkl']
        else:
            if job['index'] >= config['source_members']:
                continue
            original_id = job['id']
            original_job = old_by_id[original_id]
            if original_job['sampling_seed'] != job['sampling_seed']:
                raise ValueError('Original member sample stream differs.')
            files = ['raw.pkl']
        target = output / 'tasks' / job['id']
        target.mkdir(exist_ok=True)
        for name in files:
            shutil.copy2(source / 'tasks' / original_id / name, target / name)
        complete(output, job, imported_from=str(source / 'tasks' / original_id),
                 source_plan_id=old_manifest['plan_id'], training_time_excluded=True)
        imported.append(job['id'])
    write_json(output / 'baseline_import.json', {'source_plan_id': old_manifest['plan_id'],
               'source': str(source), 'reused_tasks': imported,
               'original_models_copied_without_retraining': True})
    print(f'Reused {len(imported)} original tasks without retraining.', flush=True)


def worker(output, job, config):
    """Fit one fresh bagging member or a matched gain-ratio raw/pruned pair."""
    if job['kind'] == 'bag_member':
        member_worker(output, job, config)
        return
    path = output / 'tasks' / job['id']
    recorder = Recorder(path, job)
    with recorder.phase('loading'):
        pools = load_pools(output, config)
        raw = pools['grow'].reset_index(drop=True)
        domains = {c: raw[c].unique().tolist() for c in config['categorical']}
        numerical = [c for c in raw.columns[:-1] if c not in config['categorical']]
        trainer = TreeTrainer(numerical, domains, max_depth=config.get('tree_max_depth'),
                              criterion=job['criterion'], seed=job['seed'],
                              progress_path=str(path / 'progress.json'))
    with recorder.phase('initial_sampling'):
        fitted = balanced_sample(raw, 0.5, config['training_draws'], job['seed'])
    if (path / 'raw.pkl').exists():
        with (path / 'raw.pkl').open('rb') as file:
            model = pickle.load(file)
    else:
        with recorder.phase('induction'):
            model = trainer(fitted)
        with recorder.phase('saving'):
            save_model(path / 'raw.pkl', model)
    with recorder.phase('pruning'):
        pruned = dict(model, tree=reduced_error_pruning(model['tree'], fitted, pools['prune']))
        save_model(path / 'pruned.pkl', pruned)
    complete(output, job, configured_max_depth=config.get('tree_max_depth'),
             last_worker_seconds=time.monotonic() - recorder.started,
             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    recorder.progress('complete')


def collect(output, config):
    """Evaluate ordered prefixes and paired criteria on growth/validation only."""
    all_jobs = jobs(config)
    available = {j['id'] for j in all_jobs if check_task(output, j)}
    pools = load_pools(output, config)
    raw = pools['grow'].reset_index(drop=True)
    classes = sorted(raw.iloc[:, -1].unique().tolist())
    evaluation = output / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    rows, class_rows = [], []

    def record(model, method, seed, members, criterion, cache=None):
        base = {'method': method, 'seed': seed, 'members': members, 'criterion': criterion}
        if isinstance(model, dict):
            base['nodes'], base['leaves'], base['depth'] = tree_shape(model['tree'])
        else:
            base.update(nodes=None, leaves=None, depth=None)
        base['model_bytes'] = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        scores = {}
        for split, frame in [('grow', raw), ('validation', pools['validation'])]:
            value = cache[split].evaluate(model) if cache is not None else evaluate(model, frame, classes)
            rows.append({**base, 'split': split, **{key: value[key] for key in
                         ('accuracy', 'macro_f1', 'balanced_accuracy', 'prediction_seconds')}})
            class_rows.extend({**base, 'split': split, 'class': label, **values}
                              for label, values in value['per_class'].items())
            scores[split] = value
        write_json(evaluation / f'{method}-{criterion}-s{seed}-n{members}.json', scores)
        # Only the newly extended endpoint needs a new composite model file.
        if method == 'bagging' and members == config['primary_members']:
            save_model(evaluation / f'bagging-s{seed}-n{members}.pkl', model)

    for seed in config['seeds']:
        for criterion in config['tree_criteria']:
            key = f'tree-{criterion}-s{seed}'
            if key in available:
                for method, file_name in [('unpruned', 'raw.pkl'), ('pruned', 'pruned.pkl')]:
                    with (output / 'tasks' / key / file_name).open('rb') as file:
                        record(pickle.load(file), method, seed, 1, criterion)
        caches = {split: EnsembleEvaluationCache(frame, classes)
                  for split, frame in [('grow', raw), ('validation', pools['validation'])]}
        members = []
        for i in range(config['primary_members']):
            key = f'bag-s{seed}-m{i+1:03}'
            if key not in available:
                break
            with (output / 'tasks' / key / 'raw.pkl').open('rb') as file:
                members.append(pickle.load(file))
            if i + 1 in config['ensemble_checkpoints']:
                model = VotingEnsemble(list(members), [1.0] * len(members),
                    raw.iloc[:, -1].unique().tolist(), raw.columns[:-1].tolist(), predict_tree)
                record(model, 'bagging', seed, i + 1, 'entropy', caches)
    write_csv(output / 'summary.csv', rows)
    write_csv(output / 'per_class.csv', class_rows)
    missing = [j['id'] for j in all_jobs if j['id'] not in available]
    write_json(output / 'coverage.json', {'complete': len(available), 'expected': len(all_jobs),
               'missing': missing, 'generalization_evaluated': False})
    timing_rows = []
    for job in all_jobs:
        path = output / 'tasks' / job['id'] / 'timings.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                value = json.loads(line)
                timing_rows.append({'task': job['id'], 'phase': value['phase'], 'seconds': value['seconds']})
    write_csv(output / 'timings.csv', timing_rows)
    lines = ['# Supplementary development results', '',
             f'Task coverage: {len(available)}/{len(all_jobs)}. Generalization data were not evaluated.', '',
             f"Standalone tree maximum depth: {config.get('tree_max_depth', 'unrestricted')}.", '',
             '| Method | Criterion | Split | Members | Macro-F1 mean | Sample SD | Seeds |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for key in sorted({(r['method'], r['criterion'], r['split'], r['members']) for r in rows}):
        values = [r['macro_f1'] for r in rows if (r['method'], r['criterion'], r['split'], r['members']) == key]
        sd = f'{np.std(values, ddof=1):.4f}' if len(values) > 1 else 'n/a'
        method, criterion, split, n = key
        lines.append(f'| {method} | {criterion} | {split} | {n} | {np.mean(values):.4f} | {sd} | {len(values)} |')
    lines += ['', 'Growth scores use the natural source pool, not the balanced fitting draws.',
              'Cached prediction_seconds measure incremental evaluation work, not inference latency.',
              'Seed SD is descriptive; all seeds share the same evaluation observations.',
              'Original entropy trees/members are reused. Their old training time is not measured again.',
              'Mixed-machine, concurrent timings do not isolate the runtime effect of split criterion.',
              'These follow-ups were designed after examining final-test findings and are exploratory.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    plots(output, rows)
    print(f'Development follow-ups saved: {len(available)}/{len(all_jobs)} tasks complete.', flush=True)


def plots(output, rows):
    """Save standalone figures with descriptive seed variation and clear labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if any(r['method'] == 'bagging' for r in rows):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for split, label in [('grow', 'Natural growth pool'), ('validation', 'Validation')]:
            selected = [r for r in rows if r['method'] == 'bagging' and r['split'] == split]
            sizes = sorted({r['members'] for r in selected})
            for ax, metric in zip(axes, ['macro_f1', 'accuracy']):
                values = [[r[metric] for r in selected if r['members'] == n] for n in sizes]
                ax.errorbar(sizes, [np.mean(v) for v in values],
                    yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in values], marker='o', label=label)
                ax.set(xlabel='Bagging members', ylabel=metric.replace('_', ' ').title())
                ax.grid(alpha=0.2)
                ax.legend()
        fig.suptitle('Bagging extension: mean +/- seed SD (development data)')
        fig.tight_layout()
        fig.savefig(output / 'bagging_curves.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for offset, criterion in enumerate(['entropy', 'gain_ratio']):
        for ax, metric in zip(axes, ['macro_f1', 'leaves']):
            groups = [[r[metric] for r in rows if r['method'] == method
                       and r['criterion'] == criterion and r['split'] == 'validation']
                      for method in ['unpruned', 'pruned']]
            means = [np.mean(v) if v else float('nan') for v in groups]
            ax.bar(np.arange(2) + (offset - 0.5) * 0.35, means, width=0.35,
                   label='Information gain' if criterion == 'entropy' else 'Gain ratio')
            ax.set_xticks([0, 1], ['Unpruned', 'Pruned'])
            ax.set_ylabel('Validation macro-F1' if metric == 'macro_f1' else 'Mean leaves')
            ax.legend()
    fig.suptitle('Matched split-criterion comparison (development data)')
    fig.tight_layout()
    fig.savefig(output / 'criterion_comparison.png', dpi=160)
    plt.close(fig)
