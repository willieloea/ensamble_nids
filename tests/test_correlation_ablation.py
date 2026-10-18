"""Checks for grouping and matched comparisons in the correlation experiment."""

import importlib.util
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'correlation_ablation', ROOT / 'experiments/correlation-ablation/run.py')
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


class CorrelationAblationTests(unittest.TestCase):
    def test_non_transitive_correlations_are_not_collapsed_together(self):
        strength = pd.DataFrame([[1, .99, .5], [.99, 1, .98], [.5, .98, 1]],
                                columns=['a', 'b', 'c'], index=['a', 'b', 'c'])
        self.assertEqual(ablation.correlated_groups(strength, .95), [['a', 'b']])

    def test_representatives_are_target_independent(self):
        grow = pd.DataFrame({'x': range(100), 'copy': [2 * i for i in range(100)],
                             'z': [(i * 37) % 101 for i in range(100)],
                             'attack_cat': [i % 3 for i in range(100)]})
        original = ablation.choose_features(grow, .95)
        changed = ablation.choose_features(grow.assign(attack_cat=0), .95)
        self.assertEqual(original, changed)
        self.assertEqual(original['feature_sets']['reduced_a'], ['x', 'z'])
        self.assertEqual(original['feature_sets']['reduced_b'], ['copy', 'z'])
        self.assertNotIn('attack_cat', original['feature_sets']['all_features'])

    def test_small_subprocess_batch_uses_identical_rows_and_pairs(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source'
            source.mkdir()
            grow = pd.DataFrame({
                'x': range(90), 'copy': [2 * i for i in range(90)],
                'proto': ['tcp'] * 90, 'service': ['other'] * 90,
                'state': ['FIN'] * 90, 'is_sm_ips_ports': [0] * 90,
                'attack_cat': [i // 30 for i in range(90)],
            })
            pools = {'grow': grow, 'prune': pd.concat([grow] * 23, ignore_index=True),
                     'validation': grow.copy()}
            pd.to_pickle(pools, source / 'development_pools.pkl')
            (source / 'manifest.json').write_text(json.dumps({
                'partition_rows': {name: len(frame) for name, frame in pools.items()}}))
            (source / 'code').mkdir()
            shutil.copytree(ROOT / 'src', source / 'code/src',
                            ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT / 'scripts/run_screening.py', source / 'code/run_screening.py')
            config = json.loads((ROOT / 'experiments/correlation-ablation/config.json').read_text())
            config.update(training_rows=20, seeds=[42], seconds=30)
            output = Path(temporary) / 'output'
            ablation.prepare(output, config, source)
            for variant in ablation.VARIANTS:
                actual = pd.read_pickle(output / 'variants' / variant / 'development_pools.pkl')
                for name in pools:
                    pd.testing.assert_series_equal(actual[name].attack_cat, pools[name].attack_cat)
                    self.assertEqual(actual[name].index.tolist(), pools[name].index.tolist())
            ablation.run(output, config, time.monotonic() + 25, progress=False)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['completed'], 3, status)
            self.assertEqual(status['failed'], 0)
            results = pd.read_csv(output / 'summary.csv')
            self.assertEqual(len(results), 6)
            self.assertTrue(results.macro_f1_delta.notna().all())
            self.assertEqual(set(results.feature_set), set(ablation.VARIANTS))
            self.assertTrue(results.decision_nodes.ge(0).all())
            self.assertTrue(results.model_bytes.gt(0).all())
            self.assertTrue(results.decision_nodes_delta.notna().all())
            per_class = pd.read_csv(output / 'per_class.csv')
            self.assertEqual(len(per_class), 18)
            self.assertTrue(per_class.recall_delta.notna().all())
            self.assertTrue((output / 'RESULTS.md').exists())

            # An absent all-feature seed must not inherit another seed's score.
            (output / 'variants/all_features/jobs/all_features-42.json').unlink()
            ablation.collect(output)
            incomplete = pd.read_csv(output / 'summary.csv')
            self.assertTrue(incomplete.macro_f1_delta.isna().all())

    def test_global_budget_is_capped_at_thirty_minutes(self):
        config = json.loads((ROOT / 'experiments/correlation-ablation/config.json').read_text())
        config['seconds'] = 1801
        with self.assertRaises(ValueError):
            ablation.validate_config(config)


if __name__ == '__main__':
    unittest.main()
