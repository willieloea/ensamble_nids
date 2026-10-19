"""Round continuation and two-machine result exchange on small deterministic data."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

import pandas as pd

from test_final_runtime_pilot import EXPERIMENT, fixture, pilot, TreeTrainer, predict_tree
from assignment4.ensemble import boosting
import final_core as core


class DistributedTests(unittest.TestCase):
    def test_independent_streams_and_complete_ownership(self):
        config = json.loads((EXPERIMENT / 'final_config.json').read_text())
        core.validate(config)
        tasks = core.jobs(config)
        self.assertEqual(len(tasks), 54)
        self.assertEqual(len({j['id'] for j in tasks}), 54)
        self.assertEqual({j['owner'] for j in tasks}, {'vostro', 'thinkbook'})
        first = {core.member_seed(config, 42, i) for i in range(25)}
        second = {core.member_seed(config, 43, i) for i in range(25)}
        self.assertFalse(first & second)
        config['bagging_member_seed_bases']['43'] = 43
        with self.assertRaisesRegex(ValueError, 'share member'):
            core.validate(config)

    def test_boosting_resume_matches_uninterrupted(self):
        frame = fixture()
        trainer = TreeTrainer(['x'], {'proto': ['a', 'b']}, max_depth=1)
        expected = boosting(frame, trainer, predict_tree, n_estimators=5,
                            learning_rate=1, random_state=42)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            def state_changed(state):
                with (path / 'state.pkl').open('wb') as file:
                    pickle.dump(state, file)
            def stop(model, round_number, **kwargs):
                if round_number == 2:
                    raise RuntimeError('simulated interruption')
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                pilot.timed_boosting(frame, trainer, n_estimators=5, learning_rate=1,
                    seed=42, recorder=pilot.Recorder(path, {}), checkpoint=stop,
                    on_state=state_changed)
            with (path / 'state.pkl').open('rb') as file:
                state = pickle.load(file)
            self.assertEqual(state['next_round'], 3)
            calls = []
            def train(sample):
                calls.append(len(sample))
                return trainer(sample)
            actual, _ = pilot.timed_boosting(frame, train, n_estimators=5, learning_rate=1,
                seed=42, recorder=pilot.Recorder(path, {}), checkpoint=lambda *a, **kw: None,
                resume=state)
            self.assertEqual(len(calls), 3)
            self.assertEqual(actual.estimator_weights, expected.estimator_weights)
            self.assertEqual(actual.estimator_errors, expected.estimator_errors)
            self.assertEqual(actual.predict_many(frame).tolist(), expected.predict_many(frame).tolist())

    def test_two_machine_cli_transfer_and_locked_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = fixture()
            pools = {name: frame.copy() for name in ('grow', 'prune', 'validation')}
            pd.to_pickle(pools, root / 'pools.pkl')
            dataset = root / 'original.csv'
            pd.concat([frame] * 4, ignore_index=True).to_csv(dataset, index=False)
            (root / 'source.json').write_text(json.dumps({'dataset_sha256': core.fingerprint(dataset)}))
            (root / 'indices.json').write_text(json.dumps({name: list(range(i * 90, (i + 1) * 90))
                for i, name in enumerate(('grow', 'prune', 'validation', 'test'))}))
            config = json.loads((EXPERIMENT / 'final_config.json').read_text())
            config.update(development_pools='pools.pkl', source_manifest='source.json',
                          split_indices='indices.json', training_draws=90,
                          partition_rows={'grow': 90, 'prune': 90, 'validation': 90, 'final_test': 90},
                          categorical=['proto'], expected_predictors=2,
                          boosting_max_depth=1, ensemble_checkpoints=[1, 2, 3], primary_members=3,
                          training_deadline=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                          timeouts_seconds={'tree': 60, 'bag_member': 60, 'boosting': 60})
            config_path = root / 'final_config.json'
            config_path.write_text(json.dumps(config))
            local = root / 'local'
            remote = root / 'remote'
            base = [sys.executable, str(EXPERIMENT / 'final_run.py'), '--config', str(config_path)]
            def run(output, *args, success=True):
                result = subprocess.run(base + ['--output', str(output), *map(str, args)],
                                        capture_output=True, text=True, timeout=60)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            run(local, '--prepare-only', '--bundle', root / 'bundle.tar.gz')
            # Exercise the actual bundle, not a filesystem copy that would
            # conceal omitted inputs such as the required dependency lockfile.
            with tarfile.open(root / 'bundle.tar.gz') as archive:
                names = set(archive.getnames())
                manifest = json.loads(archive.extractfile('local/manifest.json').read())
                self.assertTrue(all('local/' + name in names for name in manifest['sha256']))
                archive.extractall(remote, filter='data')
            remote = remote / 'local'
            run(local, '--evaluate-test', '--dataset', dataset, success=False)
            self.assertFalse((local / 'generalization_access.json').exists())
            run(local, '--role', 'vostro', '--workers', 2)
            run(remote, '--role', 'thinkbook', '--workers', 2)
            run(remote, '--role', 'thinkbook', '--export-results', root / 'returned.tar.gz')
            run(local, '--import-results', root / 'returned.tar.gz', '--collect')
            coverage = json.loads((local / 'coverage.json').read_text())
            self.assertFalse(coverage['missing'])
            self.assertEqual(coverage['complete'], 10)
            results_before = {p: p.stat().st_mtime_ns for p in (local / 'tasks').glob('*/result.json')}
            run(local, '--role', 'vostro', '--workers', 2)
            self.assertEqual(results_before, {p: p.stat().st_mtime_ns for p in results_before})
            run(local, '--evaluate-test', '--dataset', dataset)
            scores = pd.read_csv(local / 'generalization_summary.csv')
            self.assertEqual(len(scores), 8)
            self.assertEqual(set(scores.method), {'unpruned', 'pruned', 'bagging', 'boosting'})
            self.assertEqual(set(scores.seed), {42, 43})
            self.assertEqual(set(scores.loc[scores.method.isin(['bagging', 'boosting']), 'members']), {3})
            self.assertTrue((local / 'generalization_access.json').exists())
            # A changed model must be rejected before aggregation or reuse.
            model = next((local / 'tasks').glob('bag-*/raw.pkl'))
            model.write_bytes(model.read_bytes() + b'changed')
            run(local, '--collect', success=False)


if __name__ == '__main__':
    unittest.main()
