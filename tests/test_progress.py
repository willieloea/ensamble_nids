"""Verify that progress hooks report real work without changing predictions."""

from contextlib import redirect_stderr
import io
import unittest

import pandas as pd

from assignment4.dec_tree.id3 import id3
from assignment4.dec_tree.info_metrics import information_gains
from assignment4.ensemble import bagging, boosting


class ProgressTests(unittest.TestCase):
    def test_id3_reports_each_row_once(self):
        df = pd.DataFrame({'x': [1, 2, 3, 4], 'target': [0, 0, 1, 1]})
        updates = []
        model = id3(df, information_gains, continuous_features=['x'],
                    progress=lambda completed, total: updates.append((completed, total)))
        self.assertEqual(model, id3(df, information_gains, continuous_features=['x']))
        self.assertEqual(updates[0], (0, 4))
        self.assertEqual(updates[-1], (4, 4))
        self.assertEqual([done for done, _ in updates], sorted(done for done, _ in updates))

    def test_id3_leaf_reports_completion(self):
        for df in [pd.DataFrame({'target': [0, 1, 1]}),
                   pd.DataFrame({'x': [1, 1, 1], 'target': [0, 1, 1]}),
                   pd.DataFrame({'x': [1, 2, 3], 'target': [1, 1, 1]})]:
            with self.subTest(columns=df.columns.tolist()):
                updates = []
                id3(df, information_gains,
                    continuous_features=['x'] if 'x' in df else [],
                    progress=lambda completed, total: updates.append((completed, total)))
                self.assertEqual(updates[-1], (3, 3))

    def test_bagging_reports_completed_models(self):
        df = pd.DataFrame({'x': [1, 2], 'target': [0, 1]})
        updates = []
        model = bagging(df, lambda sample: 0, lambda model, row: model,
                        n_estimators=3, random_state=1,
                        progress=lambda completed, total: updates.append((completed, total)))
        self.assertEqual(len(model.estimators), 3)
        self.assertEqual(updates, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_boosting_early_stop_keeps_actual_round_count(self):
        df = pd.DataFrame({'x': [1, 2], 'target': [0, 1]})
        updates = []
        model = boosting(df, lambda sample: {1: 0, 2: 1}, lambda model, row: model[row['x']],
                         n_estimators=5,
                         progress=lambda completed, total: updates.append((completed, total)))
        self.assertEqual(len(model.estimators), 1)
        self.assertEqual(updates, [(0, 5), (1, 5)])

    def test_terminal_output_is_opt_in(self):
        df = pd.DataFrame({'x': [1, 2], 'target': [0, 1]})
        output = io.StringIO()
        with redirect_stderr(output):
            id3(df, information_gains)
        self.assertEqual(output.getvalue(), '')
        with redirect_stderr(output):
            id3(df, information_gains, progress=True)
        self.assertIn('ID3 training rows', output.getvalue())
        self.assertIn('2/2', output.getvalue())


if __name__ == '__main__':
    unittest.main()
