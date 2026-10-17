"""Deterministic checks for sampling, voting, SAMME, and ID3 integration."""

from functools import partial
import math
import pickle
import random
import unittest
import warnings

import pandas as pd

from assignment4.dec_tree.id3 import id3, predict
from assignment4.dec_tree.info_metrics import information_gains
from assignment4.ensemble import VotingEnsemble, bagging, boosting


def predict_lookup(model, row):
    return model[row['x']]


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'x': range(6), 'target': ['a', 'a', 'b', 'b', 'c', 'c']})

    def test_bootstrap_and_reproducibility(self):
        def train(sample):
            return sample['x'].tolist()

        original = self.df.copy(deep=True)
        first = bagging(self.df, train, predict_lookup, n_estimators=3, random_state=4)
        second = bagging(self.df, train, predict_lookup, n_estimators=3, random_state=4)
        rng = random.Random(4)
        expected = [rng.choices(range(6), k=6) for _ in range(3)]
        self.assertEqual(first.estimators, expected)
        self.assertEqual(first.estimators, second.estimators)
        self.assertTrue(any(len(set(sample)) < 6 for sample in expected))
        pd.testing.assert_frame_equal(self.df, original)

    def test_subbagging(self):
        samples = []

        def train(sample):
            samples.append(sample['x'].tolist())
            return 'a'

        bagging(self.df, train, lambda model, row: model, n_estimators=4,
                sample_size=3, bootstrap=False, random_state=5)
        self.assertTrue(all(len(sample) == len(set(sample)) == 3 for sample in samples))

    def test_weighted_votes_ties_and_batch_index(self):
        model = VotingEnsemble(['a', 'b', 'b'], [3, 1, 1], ['a', 'b'], ['x'],
                               lambda model, row: model)
        self.assertEqual(model.predict({'x': 2}), 'a')
        model.estimator_weights = [2, 1, 1]
        self.assertEqual(model.predict({'x': 2}), 'a')
        result = model.predict_many(pd.DataFrame({'x': [1, 2]}, index=[7, 7]))
        self.assertEqual(result.tolist(), ['a', 'a'])
        self.assertEqual(result.index.tolist(), [7, 7])

    def test_samme_error_vote_and_resampling_update(self):
        first = dict(enumerate(['a', 'a', 'b', 'b', 'c', 'b']))
        second = dict(enumerate(['b', 'a', 'b', 'b', 'c', 'c']))
        samples = []

        def train(sample):
            samples.append(sample['x'].tolist())
            return [first, second][len(samples) - 1]

        model = boosting(self.df, train, predict_lookup, n_estimators=2, random_state=8)
        self.assertAlmostEqual(model.estimator_errors[0], 1 / 6)
        self.assertAlmostEqual(model.estimator_weights[0], math.log(10))
        self.assertAlmostEqual(model.estimator_errors[1], 1 / 15)
        self.assertAlmostEqual(model.estimator_weights[1], math.log(28))
        rng = random.Random(8)
        self.assertEqual(samples[0], rng.choices(range(6), weights=[1 / 6] * 6, k=6))
        self.assertEqual(samples[1], rng.choices(range(6), weights=[1 / 15] * 5 + [2 / 3], k=6))
        self.assertEqual(model.predict({'x': 5}), 'c')

    def test_binary_update_and_learning_rate(self):
        df = pd.DataFrame({'x': range(4), 'target': [0, 0, 1, 1]})
        models = [{0: 0, 1: 0, 2: 1, 3: 0}, {0: 1, 1: 0, 2: 1, 3: 1}]
        model = boosting(df, lambda sample: models.pop(0), predict_lookup,
                         n_estimators=2, learning_rate=0.5, random_state=1)
        self.assertAlmostEqual(model.estimator_weights[0], 0.5 * math.log(3))
        expected_error = 1 / (3 + math.sqrt(3))
        self.assertAlmostEqual(model.estimator_errors[1], expected_error)

    def test_perfect_learner_stops(self):
        lookup = dict(enumerate(self.df['target']))
        calls = []

        def train(sample):
            calls.append(sample)
            return lookup

        model = boosting(self.df, train, predict_lookup, n_estimators=10)
        self.assertEqual(len(calls), 1)
        self.assertEqual(model.estimator_errors, [0.0])
        self.assertEqual(model.predict_many(self.df).tolist(), self.df['target'].tolist())

    def test_perfect_learner_dominates_previous_members(self):
        imperfect = dict(enumerate(['a', 'a', 'b', 'b', 'c', 'b']))
        perfect = dict(enumerate(self.df['target']))
        models = [imperfect, perfect]
        model = boosting(self.df, lambda sample: models.pop(0), predict_lookup,
                         n_estimators=2)
        self.assertEqual(model.estimators, [perfect])

    def test_reject_bad_first_learner(self):
        with self.assertRaisesRegex(ValueError, 'weighted error'):
            boosting(self.df, lambda sample: 'a', lambda model, row: model)

    def test_reject_later_learner_and_keep_accepted_models(self):
        good = dict(enumerate(['a', 'a', 'b', 'b', 'c', 'b']))
        bad = dict(enumerate(['b', 'b', 'c', 'c', 'a', 'a']))
        models = [good, bad]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = boosting(self.df, lambda sample: models.pop(0), predict_lookup,
                             n_estimators=2)
        self.assertEqual(model.estimators, [good])
        self.assertEqual(len(caught), 1)

    def test_id3_categorical_domains_and_pickle(self):
        df = pd.DataFrame({'colour': ['red', 'blue', 'green'], 'target': [1, 0, 2]})
        trainer = partial(id3, info_metric=information_gains,
                          domains={'colour': df['colour'].tolist()})
        model = bagging(df, trainer, predict, sample_size=1,
                        n_estimators=3, random_state=9)
        loaded = pickle.loads(pickle.dumps(model))
        self.assertEqual(model.predict_many(df).tolist(), loaded.predict_many(df).tolist())

    def test_id3_continuous_boosting(self):
        df = pd.DataFrame({'x': [1., 2., 3., 4., 5., 6.], 'target': [0, 0, 1, 1, 2, 2]})
        trainer = partial(id3, info_metric=information_gains, continuous_features=['x'])
        model = boosting(df, trainer, predict, random_state=3, n_estimators=4)
        self.assertTrue(set(model.predict_many(df)).issubset({0, 1, 2}))

    def test_nearest_neighbour_callbacks(self):
        def train(sample):
            return sample

        def nearest_neighbour(model, row):
            position = (model['x'] - row['x']).abs().to_numpy().argmin()
            return model.iloc[position, -1]

        for fit in [bagging, boosting]:
            with self.subTest(fit=fit.__name__):
                model = fit(self.df, train, nearest_neighbour,
                            random_state=3, n_estimators=4)
                self.assertEqual(len(model.predict_many(self.df)), len(self.df))
                self.assertTrue(set(model.predict_many(self.df)).issubset({'a', 'b', 'c'}))

    def test_tiny_boosting_weights_are_not_treated_as_perfect(self):
        first = dict(enumerate(['a', 'a', 'b', 'b', 'c', 'b']))
        second = dict(enumerate(['b', 'a', 'b', 'b', 'c', 'c']))
        models = [first, second]
        model = boosting(self.df, lambda sample: models.pop(0), predict_lookup,
                         n_estimators=2, learning_rate=500, random_state=1)
        self.assertEqual(len(model.estimators), 2)
        self.assertTrue(all(math.isfinite(weight) for weight in model.estimator_weights))

    def test_invalid_inputs(self):
        train = lambda sample: 'a'
        predict_constant = lambda model, row: model
        for count in [0, -1, 1.5, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                bagging(self.df, train, predict_constant, n_estimators=count)
        with self.assertRaises(ValueError):
            bagging(self.df, train, predict_constant, bootstrap=False, sample_size=7)
        with self.assertRaises(ValueError):
            bagging(self.df.iloc[:0], train, predict_constant)
        for rate in [0, -1, float('nan'), float('inf')]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                boosting(self.df, train, predict_constant, learning_rate=rate)
        with self.assertRaises(ValueError):
            boosting(self.df.assign(target='a'), train, predict_constant)
        with self.assertRaisesRegex(ValueError, 'absent'):
            boosting(self.df, train, lambda model, row: 'unknown')


if __name__ == '__main__':
    unittest.main()
