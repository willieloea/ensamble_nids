"""Actual transfer bundle, reused prefixes, matched criteria, and development-only scores."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile
import tempfile
import unittest

import pandas as pd

from test_final_runtime_pilot import EXPERIMENT, fixture
import followup_core as core


class FollowupTests(unittest.TestCase):
    def test_task_budget_and_nonoverlapping_seed_ranges(self):
        config = json.loads((EXPERIMENT / 'followup_config.json').read_text())
        core.validate(config)
        jobs = core.jobs(config)
        self.assertEqual(len(jobs), 104)
        self.assertEqual(sum(j['kind'] == 'tree' and j['criterion'] == 'gain_ratio' for j in jobs), 2)
        self.assertEqual(sum(j['kind'] == 'bag_member' and j['index'] >= 25 for j in jobs), 50)
        config['generalization_evaluation'] = True
        with self.assertRaisesRegex(ValueError, 'development evaluation only'):
            core.validate(config)

    def test_bundle_extension_and_criterion_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = fixture()
            pd.to_pickle({name: frame.copy() for name in ('grow', 'prune', 'validation')}, root / 'pools.pkl')
            (root / 'source.json').write_text('{}')
            (root / 'indices.json').write_text('{}')
            original = json.loads((EXPERIMENT / 'final_config.json').read_text())
            original.update(development_pools='pools.pkl', source_manifest='source.json',
                split_indices='indices.json', training_draws=90,
                partition_rows={'grow': 90, 'prune': 90, 'validation': 90, 'final_test': 90},
                categorical=['proto'], expected_predictors=2, boosting_max_depth=1,
                ensemble_checkpoints=[1, 2, 3], primary_members=3,
                training_deadline=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                timeouts_seconds={'tree': 60, 'bag_member': 60, 'boosting': 60})
            original_config = root / 'original_config.json'
            original_config.write_text(json.dumps(original))
            original_output = root / 'original'
            def run(script, *args, success=True):
                result = subprocess.run([sys.executable, str(EXPERIMENT / script), *map(str, args)],
                                        capture_output=True, text=True, timeout=90)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            for role in ['vostro', 'thinkbook']:
                run('final_run.py', '--config', original_config, '--output', original_output,
                    '--role', role, '--workers', 2)
            config = json.loads((EXPERIMENT / 'followup_config.json').read_text())
            config.update(source_final='original', development_pools='original/development_pools.pkl',
                training_draws=90, partition_rows=original['partition_rows'], categorical=['proto'],
                expected_predictors=2, source_members=3, primary_members=5,
                ensemble_checkpoints=[1, 3, 4, 5], workers=2,
                timeouts_seconds={'tree': 60, 'bag_member': 60})
            config_path = root / 'followup_config.json'
            config_path.write_text(json.dumps(config))
            output = root / 'followups'
            bundle = root / 'followups.tar.gz'
            run('followup_run.py', '--config', config_path, '--output', output,
                '--prepare-only', '--bundle', bundle)
            imported = json.loads((output / 'baseline_import.json').read_text())
            self.assertEqual(len(imported['reused_tasks']), 8)
            for seed in [42, 43]:
                for i in range(1, 4):
                    old = original_output / 'tasks' / f'bag-s{seed}-m{i:03}' / 'raw.pkl'
                    new = output / 'tasks' / f'bag-s{seed}-m{i:03}' / 'raw.pkl'
                    self.assertEqual(old.read_bytes(), new.read_bytes())
            with tarfile.open(bundle) as archive:
                names = set(archive.getnames())
                self.assertIn('followups/code/uv.lock', names)
                self.assertNotIn('followups/source_manifest.json', names)
                self.assertNotIn('followups/split_indices.json', names)
                archive.extractall(root / 'remote', filter='data')
            remote = root / 'remote/followups'
            command = [sys.executable, str(remote / 'code/followup_run.py'), '--output', str(remote), '--workers', '2']
            first = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn('6 new/incomplete tasks; 8 cached', first.stdout)
            coverage = json.loads((remote / 'coverage.json').read_text())
            self.assertEqual(coverage['complete'], 14)
            self.assertFalse(coverage['missing'])
            self.assertFalse(coverage['generalization_evaluated'])
            scores = pd.read_csv(remote / 'summary.csv')
            self.assertEqual(set(scores.split), {'grow', 'validation'})
            self.assertEqual(set(scores.loc[scores.method == 'bagging', 'members']), {1, 3, 4, 5})
            self.assertEqual(set(scores.loc[scores.method == 'pruned', 'criterion']), {'entropy', 'gain_ratio'})
            for seed in [42, 43]:
                with (remote / 'tasks' / f'tree-gain_ratio-s{seed}' / 'raw.pkl').open('rb') as file:
                    model = pickle.load(file)
                self.assertEqual(model['criterion'], 'gain_ratio')
                self.assertEqual(model['sampling_seed'], seed)
                with (remote / 'tasks' / f'tree-entropy-s{seed}' / 'raw.pkl').open('rb') as file:
                    baseline = pickle.load(file)
                self.assertEqual(model['fit_class_counts'], baseline['fit_class_counts'])
            self.assertTrue((remote / 'bagging_curves.png').is_file())
            self.assertTrue((remote / 'criterion_comparison.png').is_file())
            second = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn('0 new/incomplete tasks; 14 cached', second.stdout)
            # After unrestricted gain-ratio failures, a separate capped study
            # must train BOTH criteria, without importing unbounded baselines.
            bounded = dict(config, include_bagging=False, tree_max_depth=2)
            bounded_path = root / 'bounded.json'
            bounded_path.write_text(json.dumps(bounded))
            bounded_output = root / 'bounded'
            result = run('followup_run.py', '--config', bounded_path,
                         '--output', bounded_output, '--workers', 2)
            self.assertIn('4 new/incomplete tasks; 0 cached', result.stdout)
            imported = json.loads((bounded_output / 'baseline_import.json').read_text())
            self.assertFalse(imported['reused_tasks'])
            bounded_scores = pd.read_csv(bounded_output / 'summary.csv')
            self.assertEqual(set(bounded_scores.method), {'unpruned', 'pruned'})
            self.assertTrue((bounded_scores.depth <= 2).all())
            self.assertEqual(set(bounded_scores.criterion), {'entropy', 'gain_ratio'})


if __name__ == '__main__':
    unittest.main()
