"""Check out-of-sample meta training, generic callbacks, and stack inference."""

from functools import partial
import pickle
import unittest

import pandas as pd

from assignment4.dec_tree.id3 import id3, predict
from assignment4.dec_tree.info_metrics import information_gains, gini_information_gains
from assignment4.ensemble import StackingEnsemble, stacking


def train_neighbours(sample):
    return sample.copy()


def predict_neighbour(model, row):
    position = (model['x'] - row['x']).abs().to_numpy().argmin()
    return model.iloc[position, -1]


def train_first_prediction(meta):
    return meta.columns[0]


def predict_first_prediction(column, row):
    return row[column]


class StackingTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'row_id': range(18), 'x': range(18),
                                'target': ['a', 'b', 'c'] * 6}, index=[7] * 18)

    def spy_learners(self):
        trained, predicted = [], []

        def train(sample):
            ids = set(sample.row_id)
            trained.append(ids)
            return ids

        def predict_model(ids, row):
            self.assertNotIn('target', row)
            self.assertNotIn(row['row_id'], ids)
            predicted.append(row['row_id'])
            return ['a', 'b', 'c'][row['row_id'] % 3]

        return {'first': (train, predict_model), 'second': (train, predict_model)}, trained, predicted

    def test_holdout_meta_predictions_never_use_their_training_rows(self):
        learners, trained, predicted = self.spy_learners()
        captured = []

        def train_meta(meta):
            captured.append(meta.copy())
            return 'base_0'

        model = stacking(self.df, learners, train_meta, predict_first_prediction,
                         meta_encoding='labels', holdout_fraction=1 / 3, random_state=7)
        self.assertIsInstance(model, StackingEnsemble)
        self.assertEqual(model.training_scheme, 'holdout')
        self.assertEqual(len(trained), 2)  # No refit on the meta holdout.
        self.assertEqual(trained[0], trained[1])
        self.assertEqual(len(trained[0]), 12)
        self.assertEqual(set(predicted), set(range(18)) - trained[0])
        meta = captured[0]
        self.assertEqual(meta.columns.tolist(), ['base_0', 'base_1', '__stacking_target__'])
        self.assertEqual(len(meta), 6)
        self.assertEqual(meta.base_0.tolist(), meta.iloc[:, -1].tolist())

    def test_oof_alignment_refits_and_progress_with_duplicate_indices(self):
        learners, trained, predicted = self.spy_learners()
        captured, progress = [], []

        def train_meta(meta):
            captured.append(meta.copy())
            return 'base_0'

        stacking(self.df, learners, train_meta, predict_first_prediction,
                 n_folds=3, meta_encoding='labels', random_state=42,
                 progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(len(trained), 8)  # 3 folds * 2 models + 2 full fits.
        self.assertTrue(all(len(ids) == 12 for ids in trained[:6]))
        self.assertEqual(trained[-2:], [set(range(18)), set(range(18))])
        self.assertEqual(sorted(predicted), sorted(list(range(18)) * 2))
        meta = captured[0]
        self.assertEqual(meta.iloc[:, -1].tolist(), self.df.target.tolist())
        self.assertEqual(meta.base_0.tolist(), self.df.target.tolist())
        self.assertEqual(meta.base_1.tolist(), self.df.target.tolist())
        self.assertEqual(progress, [(i, 9) for i in range(10)])

    def test_one_hot_multiclass_schema_and_target_name_collision(self):
        captured = []
        df = self.df.rename(columns={'target': 'base_0_class_0'})

        def train_meta(meta):
            captured.append(meta)
            return 'a'

        model = stacking(df, {'a': (lambda sample: 'a', lambda model, row: model),
                              'c': (lambda sample: 'c', lambda model, row: model)},
                         train_meta, lambda model, row: model, random_state=1)
        meta = captured[0]
        self.assertEqual(meta.shape[1], 7)
        self.assertEqual(meta.iloc[:, -1].name, '__stacking_target__')
        expected = [1, 0, 0, 0, 0, 1]
        self.assertTrue(all(row == expected for row in meta.iloc[:, :-1].values.tolist()))
        self.assertEqual(list(model.transform({'row_id': 99, 'x': 99}).values()), expected)
        self.assertEqual(model.predict({'row_id': 99, 'x': 99}), 'a')

    def test_reproducible_splits_without_mutating_input(self):
        original = self.df.copy(deep=True)
        def fit(seed):
            learners, trained, _ = self.spy_learners()
            stacking(self.df, learners, train_first_prediction, predict_first_prediction,
                     n_folds=3, random_state=seed)
            return trained
        self.assertEqual(fit(8), fit(8))
        self.assertNotEqual(fit(8), fit(9))
        pd.testing.assert_frame_equal(self.df, original)

    def test_heterogeneous_id3_and_neighbours_and_pickle(self):
        df = pd.DataFrame({'x': [0., 1., 2., 3., 4., 5., 6., 7.],
                           'target': ['low'] * 4 + ['high'] * 4})
        tree = partial(id3, info_metric=information_gains, continuous_features=['x'])
        for folds in [None, 2]:
            with self.subTest(folds=folds):
                model = stacking(df, {'tree': (tree, predict),
                                      'knn': (train_neighbours, predict_neighbour)},
                                 train_first_prediction, predict_first_prediction,
                                 n_folds=folds, meta_encoding='labels', random_state=3)
                loaded = pickle.loads(pickle.dumps(model))
                observations = pd.DataFrame({'x': [0.5, 6.5]}, index=[4, 4])
                self.assertEqual(loaded.predict_many(observations).tolist(), ['low', 'high'])
                self.assertEqual(loaded.predict_many(observations).index.tolist(), [4, 4])
                self.assertEqual(model.predict_many(observations).tolist(),
                                 loaded.predict_many(observations).tolist())
                self.assertTrue(loaded.predict_many(observations.iloc[:0]).empty)

    def test_id3_meta_learner_with_one_hot_features(self):
        df = pd.DataFrame({'x': list(range(18)), 'target': [0] * 9 + [1] * 9})
        entropy = partial(id3, info_metric=information_gains, continuous_features=['x'])
        gini = partial(id3, info_metric=gini_information_gains, continuous_features=['x'])
        meta_domains = {f'base_{i}_class_{j}': [0, 1] for i in range(2) for j in range(2)}
        meta = partial(id3, info_metric=information_gains, domains=meta_domains)
        model = stacking(df, {'entropy': (entropy, predict), 'gini': (gini, predict)},
                         meta, predict, n_folds=3, random_state=2)
        self.assertEqual(model.predict_many(pd.DataFrame({'x': [1, 16]})).tolist(), [0, 1])

    def test_reject_singletons_or_too_many_folds_before_fitting(self):
        def should_not_fit(sample):
            self.fail('Input must be rejected before fitting.')
        learners = {'model': (should_not_fit, predict_neighbour)}
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            stacking(self.df.iloc[:3], learners, should_not_fit, predict_first_prediction)
        with self.assertRaisesRegex(ValueError, 'at least 7'):
            stacking(self.df, learners, should_not_fit, predict_first_prediction, n_folds=7)

    def test_invalid_parameters_and_predictions(self):
        learners = {'knn': (train_neighbours, predict_neighbour)}
        for kwargs in [{'n_folds': 1}, {'n_folds': True}, {'n_folds': 2.5},
                       {'holdout_fraction': 0}, {'holdout_fraction': 1},
                       {'holdout_fraction': float('nan')}, {'meta_encoding': 'probabilities'}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                stacking(self.df, learners, train_first_prediction, predict_first_prediction, **kwargs)
        with self.assertRaises(ValueError):
            stacking(self.df, {}, train_first_prediction, predict_first_prediction)
        with self.assertRaises(TypeError):
            stacking(self.df, {'bad': (None, predict_neighbour)},
                     train_first_prediction, predict_first_prediction)
        with self.assertRaisesRegex(ValueError, 'known class'):
            stacking(self.df, {'bad': (lambda sample: None, lambda model, row: 'unknown')},
                     train_first_prediction, predict_first_prediction)
        model = stacking(self.df, learners, lambda meta: None,
                         lambda model, row: ['a'], random_state=2)
        with self.assertRaisesRegex(ValueError, 'known class'):
            model.predict({'row_id': 99, 'x': 99})


if __name__ == '__main__':
    unittest.main()
