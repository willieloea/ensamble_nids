"""Deadline deferral, isolated retries and depth-only growth checks."""
from contextlib import redirect_stdout
import io
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from test_overnight_experiment import core, EXPERIMENT


def folders(output):
    for name in ['jobs', 'progress', 'points', 'models', 'logs']:
        (output / name).mkdir()


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.config = {'tree_timeout_seconds': 100, 'ensemble_timeout_seconds': 300,
                       'criteria': ['entropy'], 'pilot_sizes': [30], 'seeds': [42]}

    def test_deferred_job_leaves_prior_attempt_intact(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            folders(output)
            job = core.make_job('bagging', 30, 42)
            status = output / 'jobs' / (job['id'] + '.json')
            status.write_text(json.dumps({'status': 'timeout', 'attempt': 1}))
            with patch.object(core.time, 'monotonic', return_value=100), \
                 patch.object(core.subprocess, 'Popen') as spawn, redirect_stdout(io.StringIO()):
                outcome = core.execute_job(output, job, self.config, deadline=300)
            self.assertEqual(outcome, 'deferred')
            spawn.assert_not_called()
            self.assertEqual(json.loads(status.read_text())['status'], 'timeout')
            self.assertFalse((output / 'attempts').exists())

    def test_retries_archive_old_points_and_only_complete_is_skipped(self):
        for prior_status in ['timeout', 'interrupted', 'failed', 'running']:
            with self.subTest(status=prior_status), tempfile.TemporaryDirectory() as name:
                output = Path(name)
                folders(output)
                job = core.make_job('bagging', 30, 42)
                status = output / 'jobs' / (job['id'] + '.json')
                status.write_text(json.dumps({'status': prior_status, 'attempt': 1}))
                point = output / 'points' / (job['id'] + '__bagging__025.json')
                point.write_text('{"old": true}')
                model = output / 'models' / (job['id'] + '__latest.pkl')
                model.write_bytes(b'old model')
                log = output / 'logs' / (job['id'] + '.log')
                log.write_text('old log')
                process = Mock(returncode=0)
                process.poll.return_value = 0
                with patch.object(core.time, 'monotonic', return_value=100), \
                     patch.object(core.subprocess, 'Popen', return_value=process) as spawn, \
                     patch.object(core, 'collect') as collect, redirect_stdout(io.StringIO()):
                    outcome = core.execute_job(output, job, self.config, deadline=500)
                self.assertEqual(outcome, 'complete')
                self.assertEqual(spawn.call_count, 1)
                self.assertFalse(point.exists())
                self.assertFalse(model.exists())
                archive = output / 'attempts' / job['id'] / '1'
                self.assertEqual((archive / 'models' / model.name).read_bytes(), b'old model')
                self.assertTrue((archive / 'points' / point.name).exists())
                self.assertEqual((archive / 'logs' / log.name).read_text(), 'old log')
                current = json.loads(status.read_text())
                self.assertEqual(current['attempt'], 2)
                self.assertEqual(current['timeout_seconds'], 300)
                args = spawn.call_args.args[0]
                self.assertEqual(json.loads(args[-1])['attempt'], 2)
                with patch.object(core.subprocess, 'Popen') as spawn, redirect_stdout(io.StringIO()):
                    self.assertEqual(core.execute_job(output, job, self.config, deadline=500),
                                     'already_complete')
                spawn.assert_not_called()

    def test_timeout_and_interrupt_are_recorded(self):
        for interrupted in [False, True]:
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as name:
                output = Path(name)
                folders(output)
                job = core.make_job('tree', 30, 42)
                process = Mock(returncode=None)
                process.poll.return_value = None
                clock = [100, 100, 100, 100, 201, 201]
                with patch.object(core.time, 'monotonic', side_effect=clock), \
                     patch.object(core.subprocess, 'Popen', return_value=process), \
                     patch.object(core, 'collect'), redirect_stdout(io.StringIO()):
                    if interrupted:
                        process.poll.side_effect = [KeyboardInterrupt(), None]
                        with self.assertRaises(KeyboardInterrupt):
                            core.execute_job(output, job, self.config, deadline=500)
                    else:
                        self.assertEqual(core.execute_job(output, job, self.config, deadline=500), 'timeout')
                status = json.loads((output / 'jobs' / (job['id'] + '.json')).read_text())
                self.assertEqual(status['status'], 'interrupted' if interrupted else 'timeout')
                process.terminate.assert_called_once()
                self.assertEqual(status['timeout_seconds'], 100)

    def test_runner_considers_shorter_jobs_after_deferral_and_attempts_once(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            folders(output)
            pilot = core.make_job('tree', 30, 42, stage='pilot')
            long = core.make_job('bagging', 30, 42)
            short = core.make_job('tree', 30, 43)
            core.write_json(output / 'selection.json', {'main_rows': 30})
            calls = []
            def execute(output, job, config, deadline, completed, total):
                calls.append(job['id'])
                if job['kind'] == 'bagging':
                    return 'deferred'
                return 'timeout'  # The pilot must not retry again in the main plan.
            with patch.object(core, 'validate_pools'), \
                 patch.object(core, 'plan', return_value=[long, pilot, short]), \
                 patch.object(core, 'execute_job', side_effect=execute), \
                 patch.object(core, 'summarize'), redirect_stdout(io.StringIO()):
                core.run(output, self.config, deadline=500)
            self.assertEqual(calls, [pilot['id'], long['id'], short['id']])
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['deferred_jobs'], 1)
            self.assertEqual(status['attempted_this_invocation'], 2)
            self.assertEqual(status['reason'], 'insufficient time for pending jobs')
            self.assertFalse((output / 'jobs' / (long['id'] + '.json')).exists())

    def test_depth_worker_does_not_prune_but_baseline_does(self):
        frame = pd.DataFrame({'x': range(12), 'target': [0, 1] * 6})
        pools = {name: frame.copy() for name in ['grow', 'prune', 'validation']}
        config = dict(self.config, categorical=[], pruning_rows=6)
        for stage, depth, expected_calls in [('depth', 2, 0), ('criterion', None, 1)]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as name:
                output = Path(name)
                folders(output)
                job = core.make_job('tree', 12, 42, max_depth=depth, stage=stage)
                with patch.object(core, 'validate_pools', return_value=pools), \
                     patch.object(core, 'reduced_error_pruning', wraps=core.reduced_error_pruning) as prune:
                    core.worker(output, job, config)
                self.assertEqual(prune.call_count, expected_calls)
                points = [json.loads(p.read_text()) for p in (output / 'points').glob('*.json')]
                self.assertEqual({point['variant'] for point in points},
                                 {'raw'} if stage == 'depth' else {'raw', 'pruned'})
                self.assertEqual(json.loads((output / 'progress' / (job['id'] + '.json')).read_text())['phase'],
                                 'complete')

    def test_real_retry_replaces_current_scores_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'source'
            source.mkdir()
            frame = pd.DataFrame({'x': range(18), 'target': [0, 1, 2] * 6})
            with (source / 'development_pools.pkl').open('wb') as file:
                pickle.dump({name: frame.copy() for name in ['grow', 'prune', 'validation']}, file)
            (source / 'manifest.json').write_text('{}')
            config = json.loads((EXPERIMENT / 'config.json').read_text())
            config.update(source_experiment=str(source), categorical=[], expected_predictors=1,
                          n_estimators=2, checkpoints=[1, 2], tree_timeout_seconds=10,
                          ensemble_timeout_seconds=10)
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            output = root / 'results'
            prepared = subprocess.run([sys.executable, str(EXPERIMENT / 'run.py'),
                       '--config', str(path), '--output', str(output), '--prepare-only'],
                       capture_output=True, text=True, timeout=20)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            job = core.make_job('bagging', 18, 42)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(core.execute_job(output, job, config, time.monotonic() + 60), 'complete')
            status_path = output / 'jobs' / (job['id'] + '.json')
            status = json.loads(status_path.read_text())
            status['status'] = 'timeout'
            core.write_json(status_path, status)
            point = json.loads(next((output / 'points').glob('*.json')).read_text())
            point.update(point_id=job['id'] + '__bagging__999', round=999)
            old_path = output / 'points' / (point['point_id'] + '.json')
            core.write_json(old_path, point)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(core.execute_job(output, job, config, time.monotonic() + 60), 'complete')
            current = pd.read_csv(output / 'summary.csv')
            self.assertEqual(set(current.attempt), {2})
            self.assertEqual(set(current['round']), {1, 2})
            self.assertFalse(old_path.exists())
            archive = output / 'attempts' / job['id'] / '1'
            self.assertTrue((archive / 'points' / old_path.name).exists())
            self.assertTrue((archive / 'models' / (job['id'] + '__latest.pkl')).exists())
            self.assertEqual(json.loads(status_path.read_text())['attempt'], 2)

    def test_concurrent_runner_is_rejected(self):
        import fcntl
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            with (output / '.run.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    core.run(output, self.config, deadline=500)


if __name__ == '__main__':
    unittest.main()
