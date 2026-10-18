"""Depth bounds, ensemble checkpoints and a complete synthetic overnight batch."""
from functools import partial
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from assignment4.dec_tree.id3 import id3, predict
from assignment4.dec_tree.info_metrics import information_gains, gain_ratios
from assignment4.ensemble import bagging, boosting

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/overnight-learning-curves'
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('overnight_core', EXPERIMENT / 'overnight_core.py')
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def lookup(model, row):
    return model[row['x']]


class OvernightTests(unittest.TestCase):
    def test_depth_bounds_and_default_predictions(self):
        frame = pd.DataFrame({'x': range(12), 'target': [0, 1] * 6})
        train = partial(id3, info_metric=information_gains, continuous_features=['x'])
        full = train(frame)
        self.assertEqual([predict(full, row) for row in frame.to_dict('records')], frame.target.tolist())
        self.assertEqual(train(frame, max_depth=0), 0)
        for depth in [1, 2, 3]:
            tree = train(frame, max_depth=depth)
            self.assertLessEqual(core.tree_shape(tree)[2], depth)
        self.assertEqual(full, train(frame, max_depth=None))
        for invalid in [-1, True, 1.2]:
            with self.assertRaises(ValueError):
                train(frame, max_depth=invalid)
        categorical = pd.DataFrame({'a': ['x', 'x', 'y', 'y'], 'b': [0, 1, 0, 1],
                                    'target': [0, 1, 1, 0]})
        self.assertEqual(core.tree_shape(id3(categorical, information_gains, max_depth=1))[2], 1)

    def test_gain_ratio_changes_high_cardinality_split(self):
        frame = pd.DataFrame({'identifier': range(12),
                              'signal': [0] * 5 + [1, 0] + [1] * 5,
                              'target': [0] * 6 + [1] * 6})
        domains = {c: frame[c].unique().tolist() for c in frame.columns[:-1]}
        entropy_model = core.TreeTrainer([], domains, 1, criterion='entropy')(frame)
        ratio_model = core.TreeTrainer([], domains, 1, criterion='gain_ratio')(frame)
        self.assertEqual(next(iter(entropy_model['tree'])), 'identifier')
        self.assertEqual(next(iter(ratio_model['tree'])), 'signal')
        self.assertEqual(ratio_model['criterion'], 'gain_ratio')
        self.assertEqual(core.split_diagnostics(ratio_model['tree'])['root_feature'], 'signal')
        with self.assertRaisesRegex(ValueError, 'Unknown split criterion'):
            core.TreeTrainer([], domains, criterion='typo')(frame)
        numeric = pd.DataFrame({'x': range(8), 'target': [0, 0, 1, 1, 0, 1, 1, 1]})
        actual = core.TreeTrainer(['x'], {}, criterion='gain_ratio')(numeric)
        reference = id3(numeric, gain_ratios, continuous_features=['x'])
        self.assertEqual(actual['tree'], reference)
        self.assertEqual(core.split_diagnostics(actual['tree'])['root_split_type'], 'numerical')

    def test_default_plan_uses_information_gain_only(self):
        config = json.loads((EXPERIMENT / 'config.json').read_text())
        self.assertEqual(config['criteria'], ['entropy'])
        self.assertEqual({job['criterion'] for job in core.plan(config, 5000)}, {'entropy'})

    def test_criterion_pairs_in_plan(self):
        config = json.loads((EXPERIMENT / 'config.json').read_text())
        config['criteria'] = ['entropy', 'gain_ratio']
        jobs = core.plan(config, 5000)
        self.assertEqual(len(jobs), len({job['id'] for job in jobs}))
        by_id = {job['id']: job for job in jobs}
        for job in jobs:
            if job['kind'] == 'stacking':
                self.assertEqual(job['criterion'], 'entropy')
                continue
            if job['criterion'] == 'entropy':
                partner = by_id[job['id'].removesuffix('_entropy') + '_gain_ratio']
                for key in ['kind', 'n_train', 'seed', 'policy', 'max_depth', 'learning_rate']:
                    self.assertEqual(job[key], partner[key])
        self.assertEqual([j['criterion'] for j in jobs[:2]], ['entropy', 'gain_ratio'])
        for invalid in [['typo'], [], ['entropy', 'entropy']]:
            with self.assertRaises(ValueError):
                core.plan(dict(config, criteria=invalid), 5000)

    def test_bagging_prefixes_match_independent_runs(self):
        frame = pd.DataFrame({'x': range(8), 'target': [0, 1] * 4})
        train = partial(id3, info_metric=information_gains, continuous_features=['x'], max_depth=2)
        prefixes = []
        observed = bagging(frame, train, predict, n_estimators=3, random_state=4,
                           on_estimator=lambda model, n: prefixes.append((n, pickle.dumps(model))))
        self.assertEqual([n for n, _ in prefixes], [1, 2, 3])
        for n, blob in prefixes:
            reference = bagging(frame, train, predict, n_estimators=n, random_state=4)
            self.assertEqual(pickle.loads(blob).estimators, reference.estimators)
        self.assertEqual(observed.estimators, pickle.loads(prefixes[-1][1]).estimators)

    def test_boosting_callback_tracks_perfect_replacement(self):
        frame = pd.DataFrame({'x': range(6), 'target': ['a', 'a', 'b', 'b', 'c', 'c']})
        models = [dict(enumerate(['a', 'a', 'b', 'b', 'c', 'b'])),
                  dict(enumerate(frame.target))]
        calls = []
        result = boosting(frame, lambda sample: models.pop(0), lookup, n_estimators=3,
                          on_estimator=lambda model, n: calls.append((n, len(model.estimators))))
        self.assertEqual(calls, [(1, 1), (2, 1)])
        self.assertEqual(result.estimator_errors, [0.0])

    def test_balancing_uses_only_provided_source_rows(self):
        frame = pd.DataFrame({'x': range(12), 'target': [0] * 10 + [1] * 2}, index=range(100, 112))
        balanced = core.balanced_sample(frame, 42)
        self.assertEqual(len(balanced), len(frame))
        self.assertEqual(balanced.target.value_counts().tolist(), [6, 6])
        self.assertTrue(set(balanced.index) <= set(frame.index))
        self.assertLess(balanced.index.nunique(), len(balanced))

    def test_checkpoint_cache_reuses_predictions_and_matches_votes(self):
        frame = pd.DataFrame({'x': range(6), 'target': ['b', 'a', 'c', 'b', 'a', 'c']},
                             index=[8] * 6)
        models = [dict(enumerate(labels)) for labels in
                  [['b', 'a', 'c', 'a', 'b', 'c'], ['a', 'b', 'c', 'b', 'a', 'c'],
                   ['c', 'a', 'b', 'b', 'a', 'c']]]
        calls = []
        def counted_predict(model, row):
            self.assertEqual(set(row), {'x'})
            calls.append((id(model), row['x']))
            return lookup(model, row)
        cache = core.EnsembleEvaluationCache(frame, ['a', 'b', 'c'])
        ensemble = core.VotingEnsemble([], [], ['b', 'a', 'c'], ['x'], counted_predict)
        def check():
            reference = core.VotingEnsemble(list(ensemble.estimators),
                       list(ensemble.estimator_weights), list(ensemble.classes), ['x'], lookup)
            expected = core.evaluate(reference, frame, ['a', 'b', 'c'])
            actual = cache.evaluate(ensemble)
            for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'per_class',
                        'confusion_matrix', 'class_order']:
                self.assertEqual(actual[key], expected[key])
            return actual
        for model, weight in zip(models, [.1, .1, .2]):
            ensemble.estimators.append(model)
            ensemble.estimator_weights.append(weight)
            self.assertEqual(check()['new_estimators_scored'], 1)
        self.assertEqual(len(calls), len(frame) * 3)
        self.assertEqual(check()['new_estimators_scored'], 0)
        ensemble.estimator_weights[0] = .9
        self.assertEqual(check()['new_estimators_scored'], 0)
        self.assertEqual(len(calls), len(frame) * 3)
        # A perfect boosting learner replaces earlier votes, even after one member.
        perfect = dict(enumerate(frame.target))
        ensemble.estimators = [perfect]
        ensemble.estimator_weights = [1.0]
        self.assertEqual(check()['accuracy'], 1.0)
        self.assertEqual(len(calls), len(frame) * 4)
        # Cache state never travels with a saved estimator.
        restored = pickle.loads(pickle.dumps(reference := core.VotingEnsemble(
            [perfect], [1.0], ['b', 'a', 'c'], ['x'], lookup)))
        self.assertEqual(restored.predict_many(frame).tolist(), reference.predict_many(frame).tolist())

    def test_checkpoint_cache_handles_real_boosting_replacement(self):
        frame = pd.DataFrame({'x': range(6), 'target': ['a', 'a', 'b', 'b', 'c', 'c']})
        models = [dict(enumerate(['a', 'a', 'b', 'b', 'c', 'b'])),
                  dict(enumerate(frame.target))]
        cache = core.EnsembleEvaluationCache(frame, ['a', 'b', 'c'])
        observed = []
        def checkpoint(model, round_number):
            actual = cache.evaluate(model)
            expected = core.evaluate(model, frame, ['a', 'b', 'c'])
            self.assertEqual(actual['confusion_matrix'], expected['confusion_matrix'])
            observed.append((round_number, actual['new_estimators_scored']))
        boosting(frame, lambda sample: models.pop(0), lookup,
                 n_estimators=3, on_estimator=checkpoint)
        self.assertEqual(observed, [(1, 1), (2, 1)])

    def test_checkpoint_survives_scoring_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in ['models', 'points', 'progress', 'jobs']:
                (output / name).mkdir()
            frame = pd.DataFrame({'x': range(18), 'target': [0, 1, 2] * 6})
            with (output / 'development_pools.pkl').open('wb') as file:
                pickle.dump({name: frame.copy() for name in ['grow', 'prune', 'validation']}, file)
            config = json.loads((EXPERIMENT / 'config.json').read_text())
            config.update(categorical=[], expected_predictors=1, n_estimators=3, checkpoints=[1])
            job = core.make_job('bagging', 18, 42)
            with patch.object(core.EnsembleEvaluationCache, 'evaluate',
                              side_effect=RuntimeError('scoring interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'scoring interrupted'):
                    core.worker(output, job, config)
            with (output / 'models' / (job['id'] + '__latest.pkl')).open('rb') as file:
                model = pickle.load(file)
            self.assertEqual(len(model.estimators), 1)
            self.assertEqual(len(model.predict_many(frame)), len(frame))
            self.assertFalse(list((output / 'points').glob('*.json')))

    def test_frozen_runner_smoke_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            pools = {}
            for i, name in enumerate(['grow', 'prune', 'validation']):
                n = 60
                pools[name] = pd.DataFrame({
                    'x': list(range(n)), 'x_copy': list(range(n)),
                    'proto': ['tcp', 'udp'] * (n // 2), 'service': ['other'] * n,
                    'state': ['FIN'] * n, 'is_sm_ips_ports': [0] * n,
                    'target': [(j // 5) % 3 for j in range(n)]}, index=range(i*n, (i+1)*n))
            with (source / 'development_pools.pkl').open('wb') as file:
                pickle.dump(pools, file)
            (source / 'manifest.json').write_text('{"reserved_test_rows": 20}')
            config = json.loads((EXPERIMENT / 'config.json').read_text())
            config.update(source_experiment=str(source), pilot_sizes=[30], learning_sizes=[30, 45],
                          seeds=[42], depths=[0, 2], n_estimators=3, checkpoints=[1, 2, 3],
                          learning_rates=[0.5], pruning_rows=30, expected_predictors=6,
                          budget_seconds=240, pilot_fit_limit_seconds=120,
                          tree_timeout_seconds=10, ensemble_timeout_seconds=10,
                          criteria=['entropy', 'gain_ratio'])
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            output = root / 'results'
            command = [sys.executable, str(EXPERIMENT / 'run.py'), '--config', str(config_path),
                       '--output', str(output)]
            prepared = subprocess.run(command + ['--prepare-only'], capture_output=True, text=True)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            self.assertFalse(list((output / 'points').glob('*.json')))
            result = subprocess.run(command, capture_output=True, text=True, timeout=200)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            statuses = [json.loads(p.read_text()) for p in (output / 'jobs').glob('*.json')]
            failures = [s for s in statuses if s['status'] != 'complete']
            logs = '\n'.join(p.read_text() for p in (output / 'logs').glob('*.log'))
            self.assertFalse(failures, str(failures) + logs)
            frame = pd.read_csv(output / 'summary.csv')
            self.assertTrue({'raw', 'pruned', 'bagging', 'boosting', 'stacking', 'majority_vote'} <= set(frame.variant))
            self.assertTrue((frame.validation_macro_f1.between(0, 1)).all())
            for kind in ['tree', 'bagging', 'boosting']:
                self.assertEqual(set(frame.loc[frame.kind == kind, 'criterion']), {'entropy', 'gain_ratio'})
            self.assertEqual(set(frame.loc[frame.kind == 'stacking', 'criterion']), {'entropy'})
            pairs = pd.read_csv(output / 'criterion_comparison.csv')
            self.assertFalse(pairs.empty)
            self.assertEqual(set(pairs.kind), {'tree', 'bagging', 'boosting'})
            for _, pair in pairs.iterrows():
                self.assertAlmostEqual(pair.validation_macro_f1_delta,
                                       pair.validation_macro_f1_gain_ratio - pair.validation_macro_f1_entropy)
            aggregate = pd.read_csv(output / 'aggregate.csv', header=[0, 1],
                                    index_col=list(range(8)))
            self.assertIn('criterion', aggregate.index.names)
            self.assertTrue((aggregate[('validation_macro_f1', 'count')] == 1).all())
            depth_points = frame[frame.stage == 'depth']
            self.assertFalse(depth_points.empty)
            self.assertEqual(set(depth_points.variant), {'raw'})
            self.assertTrue((frame.attempt == 1).all())
            tree_points = frame[frame.kind == 'tree']
            self.assertTrue(tree_points.root_split_type.isin(['leaf', 'categorical', 'numerical']).all())
            self.assertTrue((frame.train_error.between(0, 1)).all())
            self.assertEqual(set(frame.validation_accuracy.notna()), {True})
            self.assertTrue(list((output / 'plots').glob('*.png')))
            before = {p.name: p.stat().st_mtime_ns for p in (output / 'points').glob('*.json')}
            resumed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in (output / 'points').glob('*.json')})
            self.assertFalse(json.loads((output / 'manifest.json').read_text())['final_test_evaluated'])
            # Keep one plot for visual inspection outside the ephemeral test directory.
            import shutil
            shutil.copy2(output / 'plots/depth_natural_gain_ratio.png', '/tmp/assignment4-overnight-smoke.png')


if __name__ == '__main__':
    unittest.main()
