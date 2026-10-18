"""Analyze frozen overnight outputs without fitting models or scoring the final test.

Run with .venv/bin/python experiments/overnight-learning-curves/analyze_results.py.
Only the results/analysis directory is written; original run outputs are preserved.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ['Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits',
               'Analysis', 'Fuzzers', 'Worms', 'Shellcode', 'Generic']


def ceiling(frame):
    """Empirical accuracy upper bound for deterministic predictions of these features."""
    features = frame.columns[:-1].tolist()
    target = frame.columns[-1]
    frequencies = frame.groupby(features + [target], sort=False, observed=True,
                                dropna=False).size().rename('count').reset_index()
    groups = frequencies.groupby(features, sort=False, observed=True, dropna=False)['count']
    sizes = groups.size()
    return dict(distinct_vectors=len(sizes), mixed_vectors=int((sizes > 1).sum()),
                accuracy_ceiling=int(groups.max().sum()) / len(frame))


def flatten_tree(tree):
    if not isinstance(tree, dict):
        return [tree]
    return [label for child in next(iter(tree.values())).values() for label in flatten_tree(child)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    root = args.results.resolve()
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path[:0] = [str(root / 'code'), str(root / 'code/src')]
    import overnight_core as core
    from assignment4.ensemble.stacking import _partitions, META_TARGET

    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    frame = pd.read_csv(root / 'summary.csv')
    per_class = pd.read_csv(root / 'per_class.csv')
    points = {p['point_id']: p for path in (root / 'points').glob('*.json')
              for p in [json.loads(path.read_text())]}
    jobs = [json.loads(path.read_text()) for path in (root / 'jobs').glob('*.json')]
    planned = json.loads((root / 'plan.json').read_text())
    assert {job['job']['id'] for job in jobs} == {job['id'] for job in planned}
    assert set(frame.point_id) == set(points) and frame.point_id.is_unique
    assert len(jobs) == 84 and all(job['status'] == 'complete' for job in jobs)
    assert set(frame.criterion) == {'entropy'}
    assert set(frame.attempt) == {1}
    with (root / 'development_pools.pkl').open('rb') as file:
        pools = pickle.load(file)
    classes = sorted(pools['grow'].iloc[:, -1].unique().tolist())
    # Independently reconstruct all reported classification metrics from counts.
    for _, row in frame.iterrows():
        point = points[row.point_id]
        assert (root / point['model_file']).exists()
        for split in ['train', 'fit', 'validation']:
            if point.get(split) is None:
                continue
            entry = point[split]
            matrix = np.asarray(entry['confusion_matrix'])
            tp = np.diag(matrix)
            support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
            precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted != 0)
            recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support != 0)
            f1 = np.divide(2*precision*recall, precision+recall,
                           out=np.zeros(len(tp)), where=precision+recall != 0)
            expected = dict(accuracy=float(tp.sum()/matrix.sum()), macro_f1=float(f1.mean()),
                            balanced_accuracy=float(recall.mean()))
            for metric, value in expected.items():
                assert abs(value-entry[metric]) < 1e-12, (row.point_id, split, metric)
                assert abs(value-row[f'{split}_{metric}']) < 1e-12
            for i, label in enumerate(classes):
                saved = entry['per_class'][str(label)]
                for metric, values in [('precision', precision), ('recall', recall), ('f1', f1)]:
                    assert abs(saved[metric]-values[i]) < 1e-12
                assert saved['support'] == support[i]
            if split == 'validation':
                assert matrix.sum() == 41234
    for row in per_class.itertuples(index=False):
        expected = points[row.point_id][row.split]['per_class'][str(row.label)]
        assert row.support == expected['support']
        for metric in ['precision', 'recall', 'f1']:
            assert abs(getattr(row, metric)-expected[metric]) < 1e-12
    metrics = ['validation_macro_f1', 'validation_accuracy', 'validation_balanced_accuracy',
               'train_error', 'fit_error', 'fit_seconds']
    full = frame[(frame.kind == 'tree') & frame.max_depth.isna()]
    cases = {}
    for n in [2000, 5000, 10000, 20000, 50000]:
        for variant in ['raw', 'pruned']:
            cases[f'{variant.capitalize()} tree {n//1000}k'] = full[
                (full.n_train == n) & (full.policy == 'natural') & (full.variant == variant)]
    for variant in ['raw', 'pruned']:
        cases[f'Balanced {variant} tree 10k'] = full[(full.n_train == 10000)
            & (full.policy == 'balanced') & (full.variant == variant)]
    for policy in ['natural', 'balanced']:
        cases[f'{policy.capitalize()} bagging 25'] = frame[(frame.kind == 'bagging')
            & (frame.policy == policy) & (frame['round'] == 25)]
        for rate in [.5, 1.]:
            cases[f'{policy.capitalize()} boosting 25 lr={rate}'] = frame[(frame.kind == 'boosting')
                & (frame.policy == policy) & (frame['round'] == 25) & (frame.learning_rate == rate)]
        for variant in ['base_entropy', 'base_depth3', 'base_depth8', 'majority_vote', 'stacking']:
            cases[f'{policy.capitalize()} stack {variant}'] = frame[(frame.kind == 'stacking')
                & (frame.policy == policy) & (frame.variant == variant)]
    method_rows = []
    for name, data in cases.items():
        assert len(data) == 3, name
        row = dict(configuration=name, repeats=3)
        for metric in metrics:
            row[metric+'_mean'] = float(data[metric].mean())
            row[metric+'_sd'] = float(data[metric].std())
        method_rows.append(row)
    method_table = pd.DataFrame(method_rows)
    method_table.to_csv(output / 'method_comparison.csv', index=False)
    pairs = [
        ('Natural bagging 25', 'Raw tree 10k'),
        ('Natural bagging 25', 'Pruned tree 10k'),
        ('Balanced bagging 25', 'Balanced raw tree 10k'),
        ('Balanced bagging 25', 'Natural bagging 25'),
        ('Balanced bagging 25', 'Balanced pruned tree 10k'),
        ('Balanced pruned tree 10k', 'Pruned tree 10k'),
        ('Raw tree 50k', 'Pruned tree 50k'),
    ]
    paired = []
    for a, b in pairs:
        for seed in [42, 43, 44]:
            left = cases[a].set_index('seed').loc[seed]
            right = cases[b].set_index('seed').loc[seed]
            paired.append(dict(comparison=f'{a} minus {b}', seed=seed,
                          macro_f1_delta=left.validation_macro_f1-right.validation_macro_f1,
                          accuracy_delta=left.validation_accuracy-right.validation_accuracy))
    pd.DataFrame(paired).to_csv(output / 'paired_differences.csv', index=False)
    class_rows = []
    for name, data in cases.items():
        subset = per_class[(per_class.split == 'validation') & per_class.point_id.isin(data.point_id)]
        for label, values in subset.groupby('label'):
            assert len(values) == 3
            class_rows.append(dict(configuration=name, label=label, class_name=CLASS_NAMES[label],
                                   support=int(values.support.iloc[0]),
                                   **{metric: values[metric].mean() for metric in ['precision','recall','f1']}))
    class_table = pd.DataFrame(class_rows)
    class_table.to_csv(output / 'per_class_comparison.csv', index=False)
    # Increasing training error can reflect contradictory labels, rather than capacity.
    duplicate_rows = []
    for n in [2000, 5000, 10000, 20000, 50000]:
        for seed in [42, 43, 44]:
            sample = core.stratified_sample(pools['grow'], n, seed)
            measured = cases[f'Raw tree {n//1000}k'].set_index('seed').loc[seed, 'train_accuracy']
            diagnostics = ceiling(sample)
            assert abs(measured-diagnostics['accuracy_ceiling']) < 1e-12
            duplicate_rows.append(dict(n_train=n, seed=seed, **diagnostics,
                    train_accuracy=measured, worms_rows=int((sample.iloc[:, -1] == 7).sum())))
    pd.DataFrame(duplicate_rows).to_csv(output / 'training_conflicts.csv', index=False)
    # Inspect saved stacking models and reconstruct their original meta fitting inputs.
    stack_rows = []
    for policy in ['natural', 'balanced']:
        for seed in [42,43,44]:
            record = cases[f'{policy.capitalize()} stack stacking'].set_index('seed').loc[seed]
            with (root / record.model_file).open('rb') as file:
                model = pickle.load(file)
            raw = core.stratified_sample(pools['grow'], 10000, seed).reset_index(drop=True)
            _, heldout = _partitions(raw.iloc[:, -1].tolist(), model.classes, None, .2,
                                    random.Random(seed))[0]
            meta = model.transform_many(raw.iloc[heldout]).reset_index(drop=True)
            meta[META_TARGET] = raw.iloc[heldout, -1].tolist()
            fitted = core.balanced_sample(meta, seed) if policy == 'balanced' else meta
            fit_metrics = core.evaluate(model.meta_estimator, fitted, classes)
            shape = core.tree_shape(model.meta_estimator['tree'])
            predicted_classes = sorted(set(flatten_tree(model.meta_estimator['tree'])))
            diagnostic = dict(policy=policy, seed=seed, meta_rows=len(meta),
                 original_meta_worms=int((meta.iloc[:, -1] == 7).sum()),
                 meta_nodes=shape[0], meta_leaves=shape[1], meta_depth=shape[2],
                 predicted_classes=','.join(map(str,predicted_classes)),
                 meta_fit_accuracy=fit_metrics['accuracy'], meta_fit_macro_f1=fit_metrics['macro_f1'],
                 meta_pattern_accuracy_ceiling=ceiling(fitted)['accuracy_ceiling'])
            stack_rows.append(diagnostic)
            matrix = np.asarray(points[record.point_id]['validation']['confusion_matrix'])
            diagnostic['validation_shellcode_predictions'] = int(matrix[:,8].sum())
            # Verify persisted ensemble scores independently of the old checkpoint cache.
            if seed == 42:
                verified = core.evaluate(model, pools['validation'], classes)
                assert abs(verified['macro_f1']-record.validation_macro_f1) < 1e-12
    pd.DataFrame(stack_rows).to_csv(output / 'stacking_diagnostics.csv', index=False)
    # Verify raw/bagging persisted models too, not only saved confusion matrices.
    for name in ['Raw tree 50k', 'Natural bagging 25', 'Balanced bagging 25']:
        record = cases[name].set_index('seed').loc[42]
        with (root / record.model_file).open('rb') as file:
            model = pickle.load(file)
        verified = core.evaluate(model, pools['validation'], classes)
        assert abs(verified['macro_f1']-record.validation_macro_f1) < 1e-12
    worker_times = pd.DataFrame([dict(kind=job['job']['kind'], wall_seconds=job['wall_seconds']) for job in jobs])
    worker_times.groupby('kind').wall_seconds.agg(['sum','mean','max','count']).to_csv(output / 'worker_times.csv')
    # Publication-sized overview figures, kept separate from original plots.
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    learning = full[full.policy == 'natural']
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for variant, color in [('raw','#2166ac'),('pruned','#b2182b')]:
        data=learning[learning.variant==variant]
        for metric, ax in [('validation_macro_f1',axes[0]),('validation_accuracy',axes[1])]:
            groups=data.groupby('n_train')[metric]; mean=groups.mean(); sd=groups.std()
            ax.errorbar(mean.index, mean, yerr=sd, marker='o', capsize=3,color=color,label=variant.capitalize())
            ax.set_xscale('log'); ax.set_xticks([2000,5000,10000,20000,50000],['2k','5k','10k','20k','50k'])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel('Original growth sample size');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Validation macro-F1');axes[1].set_ylabel('Validation accuracy')
    fig.suptitle('More original training observations improve validation scores\nMean and sample SD across three sampling seeds')
    fig.tight_layout();fig.savefig(output/'learning_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    raw_trees=frame[(frame.kind=='tree')&(frame.n_train==10000)&(frame.variant=='raw')]
    depth_labels=[0,1,2,4,8,12,20,24]
    for policy,ax in zip(['natural','balanced'],axes):
        data=raw_trees[raw_trees.policy==policy].copy();data['depth']=data.max_depth.fillna(24)
        lines=[('train_error','Original growth','#2166ac'),('validation_error','Validation','#b2182b')]
        if policy=='balanced':lines.append(('fit_error','Balanced fitting rows','#238b45'))
        for metric,label,color in lines:
            groups=data.groupby('depth')[metric];mean=groups.mean();sd=groups.std()
            ax.plot(mean.index,mean,'o-',label=label,color=color)
            ax.fill_between(mean.index,(mean-sd).values,(mean+sd).values,color=color,alpha=.12)
        ax.set_xticks(depth_labels,['0','1','2','4','8','12','20','Full'],rotation=30)
        ax.set_xlabel('Maximum edge depth');ax.set_ylabel('Classification error');ax.set_title(policy.capitalize())
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Tree depth: score populations must be distinguished (10,000 source rows)')
    fig.tight_layout();fig.savefig(output/'depth_tradeoffs.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for policy,color in [('natural','#2166ac'),('balanced','#b2182b')]:
        data=frame[(frame.kind=='bagging')&(frame.policy==policy)]
        for metric,ax in [('validation_macro_f1',axes[0]),('validation_accuracy',axes[1])]:
            groups=data.groupby('round')[metric];mean=groups.mean();sd=groups.std()
            ax.errorbar(mean.index,mean,yerr=sd,marker='o',capsize=3,label=policy.capitalize(),color=color)
            ax.set_xlabel('Number of trees');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Validation macro-F1');axes[1].set_ylabel('Validation accuracy')
    fig.suptitle('Bagging improves through about 15 trees, then largely plateaus\nMeans and sample SD, same 10,000-row source samples')
    fig.tight_layout();fig.savefig(output/'bagging_comparison.png',dpi=180);plt.close(fig)
    names=['Raw tree 10k','Pruned tree 10k','Balanced pruned tree 10k','Natural bagging 25',
           'Balanced bagging 25','Natural boosting 25 lr=1.0','Natural stack stacking',
           'Balanced stack stacking','Raw tree 50k','Pruned tree 50k']
    data=method_table.set_index('configuration').loc[names]
    fig,ax=plt.subplots(figsize=(10,6))
    ax.barh(range(len(names)),data.validation_macro_f1_mean,
            xerr=data.validation_macro_f1_sd,capsize=3,color=['#4575b4']*8+['#d73027']*2)
    ax.set_yticks(range(len(names)),[name.replace('stack stacking','stacking') for name in names])
    ax.invert_yaxis();ax.set_xlabel('Validation macro-F1')
    ax.set_xlim(0,.62);ax.grid(axis='x',alpha=.2)
    ax.set_title('Larger-data trees and 10k-row ensemble comparisons\n50k rows use five times the source data; mean ± sample SD')
    fig.tight_layout();fig.savefig(output/'model_comparison.png',dpi=180);plt.close(fig)
    # Recall and precision together expose false-positive tradeoffs.
    names=['Raw tree 50k','Balanced pruned tree 10k','Natural bagging 25',
           'Balanced bagging 25','Balanced boosting 25 lr=0.5']
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for metric,ax in zip(['recall','precision'],axes):
        grid=class_table[class_table.configuration.isin(names)].pivot(index='class_name',columns='configuration',values=metric).reindex(index=CLASS_NAMES,columns=names)
        ax.imshow(grid,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
        ax.set_xticks(range(len(names)),['Raw 50k','Bal. pruned\n10k','Natural bag\n25','Balanced bag\n25','Balanced boost\n25, lr=.5'],rotation=35,ha='right')
        ax.set_yticks(range(10),CLASS_NAMES);ax.set_title('Validation '+metric)
        for (i,j),value in np.ndenumerate(grid.to_numpy()):
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if value>.65 else 'black',fontsize=8)
    fig.suptitle('Minority recall gains can carry substantial false positives\nMean per-class scores over three seeds')
    fig.tight_layout();fig.savefig(output/'class_tradeoffs.png',dpi=180);plt.close(fig)
    summary=dict(jobs=len(jobs),points=len(frame),job_statuses=dict(Counter(job['status'] for job in jobs)),
                 worker_wall_seconds=float(worker_times.wall_seconds.sum()),
                 manifest_hashes_verified=True,confusion_metrics_verified=True,
                 deterministic_training_ceilings_verified=len(duplicate_rows),
                 saved_models_rescored=['natural/balanced stacking seed42','raw50k seed42',
                                        'natural/balanced bagging25 seed42'],
                 validation_rows=41234,final_test_evaluated=False)
    summary['source_csv_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                     for name in ['summary.csv', 'per_class.csv']}
    (output/'verification.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(stack_rows).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
