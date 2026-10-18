"""Check bottom-up error reuse against the original pruning decisions."""
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

import pandas as pd

from assignment4.dec_tree.id3 import id3
from assignment4.dec_tree.info_metrics import information_gains, gain_ratios
from assignment4.dec_tree.reduced_error_pruning import reduced_error_pruning, error_count


def reference_prune(tree, training, validation):
    """Original post-order algorithm, re-predicting every retained subtree."""
    target = training.columns[-1]

    def mask(frame, split, branch):
        if isinstance(split, tuple):
            return frame[split[0]] < split[1] if branch == '<' else frame[split[0]] >= split[1]
        return frame[split] == branch

    def prune(node, train, valid):
        if not isinstance(node, dict) or train.empty:
            return node
        majority = train[target].mode().iloc[0]
        split, branches = next(iter(node.items()))
        for branch, child in list(branches.items()):
            branches[branch] = prune(child, train.loc[mask(train, split, branch)],
                                    valid.loc[mask(valid, split, branch)])
        return majority if int((valid[target] != majority).sum()) <= error_count(node, valid) else node

    return prune(deepcopy(tree), training, validation)


class PruningTests(unittest.TestCase):
    def test_same_pruned_trees_on_mixed_multiclass_data(self):
        for seed in range(8):
            rng = random.Random(seed)
            frame = pd.DataFrame({'category': [rng.choice(['a', 'b', 'c']) for _ in range(70)],
                                  'x': [rng.randrange(12) for _ in range(70)],
                                  'target': [rng.randrange(3) for _ in range(70)]})
            train, valid = frame.iloc[:45].copy(), frame.iloc[45:].copy()
            train.index = [7] * len(train)
            valid.index = [8] * len(valid)
            for metric in [information_gains, gain_ratios]:
                tree = id3(train, metric, domains={'category': ['a', 'b', 'c', 'unused']},
                           continuous_features=['x'])
                original = deepcopy(tree)
                expected = reference_prune(tree, train, valid)
                with patch('assignment4.dec_tree.reduced_error_pruning.error_count',
                           side_effect=AssertionError('subtree must not be re-predicted')):
                    actual = reduced_error_pruning(tree, train, valid)
                self.assertEqual(actual, expected)
                self.assertEqual(tree, original)

    def test_ties_pure_leaves_and_unvisited_branches(self):
        train = pd.DataFrame({'c': ['a', 'a', 'b', 'b'], 'x': [0, 1, 0, 1],
                              'target': [0, 1, 1, 0]})
        tree = {'c': {'a': {('x', .5): {'<': 0, '>=': 1}},
                      'b': {('x', .5): {'<': 1, '>=': 0}}, 'unused': 2}}
        valid = train.iloc[:2].copy()
        self.assertEqual(reduced_error_pruning(tree, train, valid), reference_prune(tree, train, valid))
        self.assertEqual(reduced_error_pruning(2, train, valid), 2)
        tied = pd.DataFrame({'c': ['a', 'b'], 'target': [0, 1]})
        self.assertEqual(reduced_error_pruning({'c': {'a': 0, 'b': 0}}, tied, tied), 0)

    def test_empty_training_subtree_is_retained_and_scored_once(self):
        train = pd.DataFrame({'c': ['a', 'a'], 'x': [0, 1], 'target': [0, 0]})
        valid = pd.DataFrame({'c': ['a', 'b', 'b'], 'x': [0, 0, 1], 'target': [0, 1, 2]})
        tree = {'c': {'a': 0, 'b': {('x', .5): {'<': 1, '>=': 2}}}}
        expected = reference_prune(tree, train, valid)
        with patch('assignment4.dec_tree.reduced_error_pruning.error_count', wraps=error_count) as score:
            actual = reduced_error_pruning(tree, train, valid)
        self.assertEqual(actual, expected)
        self.assertEqual(score.call_count, 1)

    def test_unknown_validation_category_still_raises(self):
        train = pd.DataFrame({'c': ['a', 'b'], 'target': [0, 1]})
        valid = pd.DataFrame({'c': ['unknown'], 'target': [0]})
        with self.assertRaises(KeyError):
            reduced_error_pruning({'c': {'a': 0, 'b': 1}}, train, valid)


if __name__ == '__main__':
    unittest.main()
