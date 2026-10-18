"""Compare incremental split scores with the original Boolean-partition method."""

import math
import random
import unittest
from unittest.mock import patch

import pandas as pd

from assignment4.dec_tree import info_metrics
from assignment4.dec_tree.id3 import id3, predict
from assignment4.dec_tree.reduced_error_pruning import reduced_error_pruning


def reference_scores(df, feature, info_metric=info_metrics.information_gains):
    """The original scoring loop, deliberately retaining pandas partitions."""
    scores = {}
    for threshold in info_metrics.candidate_thresholds(df, feature):
        split = df.loc[:, [feature, df.columns[-1]]].copy()
        split[feature] = df[feature] >= threshold
        scores[threshold] = info_metric(split)[feature]
    return scores


class ContinuousScoringTests(unittest.TestCase):
    def setUp(self):
        self.metrics = [info_metrics.information_gains,
                        info_metrics.gini_information_gains, info_metrics.gain_ratios]

    def assert_scores_match(self, df):
        for metric in self.metrics:
            with self.subTest(metric=metric.__name__, dtype=str(df.x.dtype)):
                expected = reference_scores(df, 'x', metric)
                actual = info_metrics.continuous_information_gains(df, 'x', metric)
                self.assertEqual(list(actual), list(expected))
                self.assertEqual(actual, expected)
                if expected:
                    self.assertEqual(max(actual, key=actual.get), max(expected, key=expected.get))

    def test_mixed_equal_values_and_same_label_boundary_filter(self):
        df = pd.DataFrame({'x': [0, 0, 1, 1, 2, 3],
                           'target': ['a', 'b', 'b', 'b', 'b', 'c']}, index=[7] * 6)
        self.assertEqual(info_metrics.candidate_thresholds(df, 'x'), [.5, 2.5])
        self.assert_scores_match(df)

    def test_random_multiclass_scores_are_exactly_equal(self):
        rng = random.Random(42)
        for trial in range(25):
            count = rng.randrange(2, 100)
            df = pd.DataFrame({'x': [rng.randrange(-10, 11) / 3 for _ in range(count)],
                               'target': [rng.choice(['a', 'b', 'c', 'd']) for _ in range(count)]})
            with self.subTest(trial=trial):
                self.assert_scores_match(df)

    def test_empty_constant_single_row_and_pure_labels(self):
        for values, labels in [([], []), ([1, 1], [0, 1]), ([1], [0]),
                               ([0, 1, 2], [0, 0, 0])]:
            self.assert_scores_match(pd.DataFrame({'x': pd.Series(values, dtype=float),
                                                   'target': labels}))

    def test_numeric_dtypes_and_precision_sensitive_fallbacks(self):
        for dtype, values in [('int32', [0, 1, 2, 3]), ('Int64', [0, 1, 2, 3]),
                              ('UInt64', [0, 1, 2, 3]), ('Float64', [0., 1., 2., 3.]),
                              ('float32', [1., 1. + 2**-23, 2., 3.]),
                              ('Float32', [1., 1. + 2**-23, 2., 3.]),
                              ('int64', [2**53 + i for i in [1, 2, 3, 4]]),
                              ('uint64', [2**63 + i for i in [1, 2, 3, 4]]),
                              ('Int64', [2**53 + i for i in [1, 2, 3, 4]]),
                              ('boolean', [False, False, True, True])]:
            self.assert_scores_match(pd.DataFrame({'x': pd.Series(values, dtype=dtype),
                                                   'target': [0, 1, 1, 0]}))
        # pandas cannot group float16 indexes; preserve that existing failure.
        unsupported = pd.DataFrame({'x': pd.Series([1., 2.], dtype='float16'), 'target': [0, 1]})
        with self.assertRaises(NotImplementedError):
            reference_scores(unsupported, 'x')
        with self.assertRaises(NotImplementedError):
            info_metrics.continuous_information_gains(unsupported, 'x')

    def test_adjacent_float64_and_extreme_finite_values(self):
        for values in [[1., math.nextafter(1., 2.), 2., math.nextafter(2., 3.)],
                       [-1e308, -1e307, 1e307, 1e308]]:
            self.assert_scores_match(pd.DataFrame({'x': values, 'target': [0, 1, 0, 1]}))

    def test_missing_and_categorical_targets_preserve_legacy_scoring(self):
        self.assert_scores_match(pd.DataFrame({'x': [0., 1., 2., 3.],
                                               'target': [0., float('nan'), 1., 0.]}))
        df = pd.DataFrame({'x': [0., 1., 2., 3.],
                           'target': pd.Categorical(['a', 'b', 'a', 'b'], categories=['a', 'b', 'unused'])})
        self.assertEqual(reference_scores(df, 'x', info_metrics.gini_information_gains),
                         info_metrics.continuous_information_gains(df, 'x', info_metrics.gini_information_gains))
        # The existing entropy helper rejects zero-probability unused categories;
        # this optimisation deliberately does not change that separate behaviour.
        with self.assertRaises(ValueError):
            reference_scores(df, 'x')
        with self.assertRaises(ValueError):
            info_metrics.continuous_information_gains(df, 'x')

    def test_custom_scorer_receives_original_boolean_frames(self):
        seen = []
        df = pd.DataFrame({'x': [1., 2., 3., 4.], 'target': [0, 1, 0, 1]}, index=[9] * 4)
        original = df.copy(deep=True)

        def scorer(frame):
            self.assertEqual(frame.columns.tolist(), ['x', 'target'])
            self.assertEqual(frame.x.dtype, bool)
            seen.append(frame.copy())
            return {'x': float(frame.x.mean())}

        actual = info_metrics.continuous_information_gains(df, 'x', scorer)
        self.assertEqual(actual, {1.5: .75, 2.5: .5, 3.5: .25})
        self.assertEqual(len(seen), 3)
        pd.testing.assert_frame_equal(df, original)

    def test_invalid_features_raise_the_same_errors(self):
        for values in [['a', 'b'], [0., float('nan')], [0., float('inf')],
                       [float('-inf'), 0.]]:
            df = pd.DataFrame({'x': values, 'target': [0, 1]})
            with self.assertRaises(ValueError):
                info_metrics.continuous_information_gains(df, 'x')

    def test_identical_trees_predictions_and_pruning(self):
        rng = random.Random(8)
        df = pd.DataFrame({'x': [rng.randrange(8) for _ in range(45)],
                           'z': [rng.randrange(5) / 2 for _ in range(45)],
                           'kind': [rng.choice(['a', 'b']) for _ in range(45)],
                           'target': [rng.randrange(3) for _ in range(45)]})
        train, validation = df.iloc[:35].copy(), df.iloc[35:].copy()
        for metric in self.metrics:
            with self.subTest(metric=metric.__name__):
                fast = id3(train, metric, continuous_features=['x', 'z'])
                with patch.object(info_metrics, 'continuous_information_gains', reference_scores):
                    reference = id3(train, metric, continuous_features=['x', 'z'])
                self.assertEqual(fast, reference)
                self.assertEqual([predict(fast, row) for row in validation.to_dict('records')],
                                 [predict(reference, row) for row in validation.to_dict('records')])
                self.assertEqual(reduced_error_pruning(fast, train, validation),
                                 reduced_error_pruning(reference, train, validation))


if __name__ == '__main__':
    unittest.main()
