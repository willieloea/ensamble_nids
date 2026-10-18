"""Sampling, SAMME equivalence, and frozen subprocess pilot integration."""

import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/final-tree-comparison'
for path in (ROOT / 'src', ROOT / 'scripts', ROOT / 'experiments/overnight-learning-curves', EXPERIMENT):
    sys.path.insert(0, str(path))
import pilot_core as pilot
from assignment4.ensemble import boosting
from overnight_core import TreeTrainer
from run_screening import predict_tree

spec = importlib.util.spec_from_file_location('final_pilot_run', EXPERIMENT / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    frame = pd.DataFrame({'x': list(range(90)), 'proto': ['a', 'b', 'a'] * 30,
                          'target': [0] * 54 + [1] * 27 + [2] * 9})
    frame.loc[[8, 16, 32, 64, 80], 'target'] = [1, 2, 1, 0, 0]
    return frame


class RuntimePilotTests(unittest.TestCase):
    def test_quotas_repeatability_and_source_ids(self):
        frame = pd.DataFrame({'x': range(60), 'target': [0] * 42 + [1] * 12 + [2] * 6})
        for strength, expected in [(0, {0: 42, 1: 12, 2: 6}),
                                   (0.5, {0: 31, 1: 16, 2: 13}),
                                   (1, {0: 20, 1: 20, 2: 20})]:
            first = pilot.balanced_sample(frame, strength, 60, 42)
            self.assertEqual(first.target.value_counts().to_dict(), expected)
            pd.testing.assert_frame_equal(first, pilot.balanced_sample(frame, strength, 60, 42))
            self.assertTrue(set(first.index) <= set(frame.index))
        first = pilot.balanced_sample(frame, 1, 60, 42)
        second = pilot.balanced_sample(frame, 1, 60, 43)
        self.assertNotEqual(first.index.tolist(), second.index.tolist())
        rounded = pilot.balanced_sample(frame, 1, 62, 42)
        self.assertEqual(rounded.target.value_counts().to_dict(), {0: 21, 1: 21, 2: 20})

    def test_timed_samme_matches_package(self):
        frame = fixture()
        trainer = TreeTrainer(['x'], {'proto': ['a', 'b']}, max_depth=1)
        for rate in (0.5, 1.0):
            for seed in (42, 43):
                expected = boosting(frame, trainer, predict_tree, n_estimators=5,
                                    learning_rate=rate, random_state=seed)
                with tempfile.TemporaryDirectory() as folder:
                    recorder = pilot.Recorder(Path(folder), {})
                    actual, _ = pilot.timed_boosting(frame, trainer, n_estimators=5,
                        learning_rate=rate, seed=seed, recorder=recorder,
                        checkpoint=lambda *args, **kwargs: None)
                self.assertEqual([m['tree'] for m in actual.estimators],
                                 [m['tree'] for m in expected.estimators])
                self.assertTrue(all(m['distinct_source_rows'] <= len(frame) for m in actual.estimators))
                self.assertEqual(actual.estimator_weights, expected.estimator_weights)
                self.assertEqual(actual.estimator_errors, expected.estimator_errors)
                self.assertEqual(actual.predict_many(frame).tolist(), expected.predict_many(frame).tolist())

    def test_perfect_and_rejected_learners(self):
        frame = pd.DataFrame({'x': range(6), 'target': [0, 0, 0, 1, 1, 1]})
        for perfect in (False, True):
            trainer = (TreeTrainer(['x'], {}, max_depth=1) if perfect else
                       lambda sample: {'tree': 0, 'domains': {}, 'fallback': 0})
            with tempfile.TemporaryDirectory() as folder:
                recorder = pilot.Recorder(Path(folder), {})
                if perfect:
                    actual, reason = pilot.timed_boosting(frame, trainer, n_estimators=5,
                        learning_rate=1, seed=42, recorder=recorder,
                        checkpoint=lambda *args, **kwargs: None)
                    self.assertEqual(reason, 'perfect_learner')
                    self.assertEqual(actual.estimator_errors, [0])
                else:
                    with self.assertRaisesRegex(ValueError, 'First boosting'):
                        pilot.timed_boosting(frame, trainer, n_estimators=5,
                            learning_rate=1, seed=42, recorder=recorder,
                            checkpoint=lambda *args, **kwargs: None)

    def test_full_cli_resume_retry_and_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            frame = fixture()
            with (path / 'pools.pkl').open('wb') as file:
                pickle.dump({name: frame.copy() for name in ('grow', 'prune', 'validation')}, file)
            config = json.loads((EXPERIMENT / 'config.json').read_text())
            config.update(development_pools='pools.pkl', training_draws=90,
                          partition_rows={'grow': 90, 'prune': 90, 'validation': 90, 'final_test': 0},
                          categorical=['proto'], expected_predictors=2,
                          boosting_max_depths=[1, 2], ensemble_checkpoints=[1, 2, 3])
            config['runtime_pilot'].update(initial_members=2, adaptive_boosting_members=[3],
                                           job_timeout_seconds=60)
            config_path = path / 'config.json'
            config_path.write_text(json.dumps(config))
            output = path / 'results'
            command = [sys.executable, str(EXPERIMENT / 'run.py'), '--config', str(config_path),
                       '--output', str(output)]
            first = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            statuses = [json.loads(p.read_text()) for p in (output / 'jobs').glob('*/result.json')]
            self.assertTrue(all(s['status'] == 'complete' for s in statuses), statuses)
            progress_lines = [line for line in first.stdout.splitlines()
                              if line.startswith('Pilot jobs processed ')]
            self.assertEqual(len(progress_lines), len(statuses))
            self.assertNotIn('tree rows', first.stdout + first.stderr)
            self.assertEqual(json.loads((output / 'status.json').read_text())['initial_complete'], 12)
            self.assertIn('Configuration coverage: 12/12', (output / 'FORECAST.md').read_text())
            for result in statuses:
                self.assertLessEqual(sum(result['phase_seconds'].values()), result['wall_seconds'] + 0.01)
            result_path = next((output / 'jobs').glob('*/result.json'))
            before = result_path.stat().st_mtime_ns
            second = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(before, result_path.stat().st_mtime_ns)
            result = json.loads(result_path.read_text())
            result['status'] = 'timeout'
            result_path.write_text(json.dumps(result))
            retry = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
            self.assertEqual(json.loads(result_path.read_text())['status'], 'complete')
            self.assertTrue(list((output / 'attempts' / result['job']['id']).iterdir()))
            frozen = output / 'code/pilot_core.py'
            frozen.write_text(frozen.read_text() + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                runner.verify(output)

    def test_deadline_defers_without_creating_job(self):
        import time
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'jobs').mkdir()
            bar = type('Bar', (), {'update': lambda *a, **kw: None})()
            self.assertFalse(runner.execute(output, {'id': 'deferred'}, 60,
                                             time.monotonic() + 1, bar))
            self.assertEqual(list((output / 'jobs').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
