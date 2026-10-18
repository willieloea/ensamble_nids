"""Independent resampling, fitting-population scores and holdout isolation."""
from contextlib import redirect_stdout
import io
import json
import pickle
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_overnight_experiment import core, EXPERIMENT, lookup
from assignment4.ensemble.stacking import _partitions


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'x': range(60), 'target': [0] * 42 + [1] * 12 + [2] * 6})

    def test_balanced_members_use_new_samples_from_entire_pool(self):
        def fit():
            samples, checkpoints = [], []
            def train(sample):
                samples.append(sample.copy())
                return {'tree': 0, 'domains': {}, 'fallback': 0}
            with redirect_stdout(io.StringIO()):
                result = core.balanced_bagging(self.frame, train, n_estimators=12, random_state=42,
                         on_sample=lambda sample: None,
                         on_estimator=lambda model, n: checkpoints.append(n))
            return samples, checkpoints, result
        first, checkpoints, model = fit()
        again, _, _ = fit()
        self.assertEqual(checkpoints, list(range(1, 13)))
        for a, b, member in zip(first, again, model.estimators):
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(a.target.value_counts().to_dict(), {0: 20, 1: 20, 2: 20})
            self.assertTrue(set(a.x) <= set(self.frame.x))
            self.assertEqual(sum(member['fit_source_counts'].values()), len(self.frame))
        selected = [set(sample.loc[sample.target == 0, 'x']) for sample in first]
        self.assertGreater(len(set.union(*selected)), len(selected[0]))
        self.assertNotEqual(first[0].x.tolist(), first[1].x.tolist())

    def test_weighted_fit_scores_reuse_cached_predictions(self):
        frame = pd.DataFrame({'x': range(6), 'target': [0, 0, 0, 0, 1, 1]})
        fitted = core.balanced_sample(frame, 42)
        calls = []
        def predict(model, row):
            calls.append(row['x'])
            return model[row['x']]
        member = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0}
        model = core.VotingEnsemble([member], [1.0], [0, 1], ['x'], predict)
        cache = core.EnsembleEvaluationCache(frame, [0, 1])
        cache.evaluate(model)
        counts = np.bincount(fitted.index.to_numpy(), minlength=len(frame))
        actual = cache.evaluate(model, row_counts=counts)
        reference = core.VotingEnsemble([member], [1.0], [0, 1], ['x'], lookup)
        expected = core.evaluate(reference, fitted, [0, 1])
        for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'confusion_matrix', 'per_class']:
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(actual['new_estimators_scored'], 0)
        self.assertEqual(len(calls), len(frame))

    def test_stacking_samples_are_independent_and_exclude_meta_holdout(self):
        frame = pd.DataFrame({'x': range(60), 'target': [0, 1, 2] * 20})
        classes = frame.target.unique().tolist()
        base_positions, holdout_positions = _partitions(frame.target.tolist(), classes, None, .2,
                                                       random.Random(42))[0]
        allowed = set(frame.iloc[base_positions].x)
        forbidden = set(frame.iloc[holdout_positions].x)
        for policy in ['natural', 'balanced']:
            captured = []
            original = core.id3
            def spy(fitted, *args, **kwargs):
                if 'x' in fitted:
                    captured.append(fitted.copy())
                return original(fitted, *args, **kwargs)
            learners = {f'base{i}': (core.TreeTrainer(['x'], {}, depth, policy=policy, seed=42+i,
                                      bootstrap=True, record_fit=policy == 'balanced'), core.predict_tree)
                        for i, depth in enumerate([None, 3, 8])}
            meta = core.TreeTrainer([], {f'base_{i}_class_{j}': [0, 1] for i in range(3)
                                         for j in range(3)}, 3)
            with patch.object(core, 'id3', side_effect=spy):
                model = core.stacking(frame, learners, meta, core.predict_tree,
                                      holdout_fraction=.2, random_state=42)
            self.assertEqual(len(captured), 3)
            self.assertNotEqual(captured[0].x.tolist(), captured[1].x.tolist())
            for sample in captured:
                self.assertTrue(set(sample.x) <= allowed)
                self.assertFalse(set(sample.x) & forbidden)
            self.assertEqual({v['sampling_seed'] for v in model.estimators.values()}, {42, 43, 44})
            if policy == 'balanced':
                self.assertTrue(all('fit_metrics' in m for m in model.estimators.values()))

    def test_natural_only_reports_leave_fit_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in ['points', 'models', 'progress', 'jobs', 'plots']:
                (output / name).mkdir()
            config = json.loads((EXPERIMENT / 'config.json').read_text())
            config.update(categorical=[], pruning_rows=30)
            (output / 'config.json').write_text(json.dumps(config))
            pools = {name: self.frame.copy() for name in ['grow', 'prune', 'validation']}
            with patch.object(core, 'validate_pools', return_value=pools):
                core.worker(output, core.make_job('tree', 60, 42), config)
            core.summarize(output, config)
            frame = pd.read_csv(output / 'summary.csv')
            self.assertTrue(frame.fit_macro_f1.isna().all())
            self.assertTrue((output / 'aggregate.csv').exists())

    def test_balanced_workers_export_fit_metrics_and_scopes(self):
        pools = {name: self.frame.copy() for name in ['grow', 'prune', 'validation']}
        config = json.loads((EXPERIMENT / 'config.json').read_text())
        config.update(categorical=[], pruning_rows=30, n_estimators=2, checkpoints=[1, 2])
        for kind in ['tree', 'bagging', 'boosting', 'stacking']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                for name in ['points', 'models', 'progress', 'jobs']:
                    (output / name).mkdir()
                job = core.make_job(kind, 60, 42, policy='balanced',
                                    max_depth=3 if kind == 'boosting' else None,
                                    learning_rate=.5 if kind == 'boosting' else None)
                with patch.object(core, 'validate_pools', return_value=pools), redirect_stdout(io.StringIO()):
                    core.worker(output, job, config)
                points = [json.loads(p.read_text()) for p in (output / 'points').glob('*.json')]
                summary = pd.read_csv(output / 'summary.csv')
                self.assertIn('fit_macro_f1', summary)
                if kind in ['tree', 'boosting']:
                    fitted = core.balanced_sample(self.frame, 42)
                    for point in points:
                        with (output / point['model_file']).open('rb') as file:
                            model = pickle.load(file)
                        expected = core.evaluate(model, fitted, [0, 1, 2])
                        self.assertEqual(point['fit']['confusion_matrix'], expected['confusion_matrix'])
                        self.assertEqual(point['fit_evaluation_scope'], 'shared_balanced_training_population')
                elif kind == 'bagging':
                    for point in points:
                        self.assertEqual(point['fit_rows'], point['round'] * len(self.frame))
                        self.assertEqual(point['fit_evaluation_scope'], 'cumulative_member_fit_draws')
                        self.assertEqual(point['fit']['new_estimators_scored'], 0)
                        draws = pd.concat([core.balanced_sample(self.frame, 42+i)
                                           for i in range(point['round'])])
                        with (output / point['model_file']).open('rb') as file:
                            fitted_model = pickle.load(file)
                        expected = core.evaluate(fitted_model, draws, [0, 1, 2])
                        self.assertEqual(point['fit']['confusion_matrix'], expected['confusion_matrix'])
                        self.assertGreater(point['distinct_fit_source_rows'], 0)
                else:
                    for point in points:
                        if point['variant'].startswith('base_'):
                            self.assertIsNotNone(point['fit'])
                            self.assertEqual(point['fit_evaluation_scope'], 'own_fitted_rows')
                        else:
                            self.assertIsNone(point['fit'])
                        self.assertIn('fit_metrics', point['stack_fit_details']['meta'])
                per_class = pd.read_csv(output / 'per_class.csv')
                self.assertIn('fit', set(per_class.split))


if __name__ == '__main__':
    unittest.main()
