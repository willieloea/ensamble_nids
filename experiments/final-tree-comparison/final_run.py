"""Parallel final comparison, with fixed machine ownership and frozen inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def prepare(output, config_path):
    """Freeze the pilot dependencies plus final-run code and split provenance."""
    import run as pilot_run
    existed = (output / 'manifest.json').exists()
    pilot_run.prepare(output, config_path)
    if existed:
        if not json.loads((output / 'manifest.json').read_text()).get('final_comparison'):
            raise ValueError('Select a fresh output directory for the final comparison.')
        return
    config = json.loads(config_path.read_text())
    for name in ('final_run.py', 'final_core.py'):
        shutil.copy2(HERE / name, output / 'code' / name)
    for key in ('source_manifest', 'split_indices'):
        source = (config_path.parent / config[key]).resolve()
        shutil.copy2(source, output / f'{key}.json')
    manifest = json.loads((output / 'manifest.json').read_text())
    for path in [output / 'code/final_run.py', output / 'code/final_core.py',
                 output / 'source_manifest.json', output / 'split_indices.json']:
        manifest['sha256'][str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['final_comparison'] = True
    manifest['plan_id'] = hashlib.sha256(json.dumps(manifest['sha256'], sort_keys=True).encode()).hexdigest()
    pilot_run.write_json(output / 'manifest.json', manifest)
    (output / 'tasks').mkdir()


def execute(output, job, timeout, deadline, stop=None, worker_script='final_run.py'):
    """Supervise one process; preserve member/state checkpoints after interruption."""
    from run_screening import write_json
    from final_core import check_task
    path = output / 'tasks' / job['id']
    result_path = path / 'result.json'
    if check_task(output, job):
        return 'cached'
    if stop is not None and stop.is_set():
        return 'interrupted'
    if deadline - time.time() < timeout:
        return 'deferred'
    path.mkdir(parents=True, exist_ok=True)
    attempt = time.time_ns()
    write_json(result_path, {'status': 'running', 'job': job, 'attempt': attempt})
    command = [sys.executable, str(output / 'code' / worker_script), '--output', str(output),
               '--worker', json.dumps(job)]
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    started = time.monotonic()
    process = None
    status = 'failed'
    try:
        with (path / f'worker-{attempt}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
            while process.poll() is None:
                if stop is not None and stop.is_set():
                    status = 'interrupted'
                    break
                if time.monotonic() - started >= timeout or time.time() >= deadline:
                    status = 'timeout'
                    break
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            else:
                status = 'complete' if process.returncode == 0 else 'failed'
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        result = json.loads(result_path.read_text())
        if status == 'complete' and result.get('status') != 'complete':
            status = 'failed'
        if status != 'complete' or result.get('status') != 'complete':
            result.update(status=status, job=job)
        result.update(last_attempt_seconds=time.monotonic() - started, attempt=attempt)
        write_json(result_path, result)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/final')
    parser.add_argument('--config', type=Path, default=HERE / 'final_config.json')
    parser.add_argument('--role', choices=['thinkbook', 'vostro'])
    parser.add_argument('--workers', type=int)
    parser.add_argument('--deadline', help='Timezone-aware ISO timestamp overriding the training cutoff')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--import-pilot', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--export-results', type=Path)
    parser.add_argument('--import-results', type=Path)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.worker:
        if not args.config.exists() and (output / 'config.json').exists():
            args.config = output / 'config.json'
        prepare(output, args.config.resolve())
        frozen = output / 'code/final_run.py'
        if Path(__file__).resolve() != frozen.resolve():
            command = [sys.executable, str(frozen), '--output', str(output),
                       '--config', str(output / 'config.json')]
            for name in ('role', 'workers', 'deadline', 'import_pilot', 'bundle',
                         'export_results', 'import_results', 'dataset'):
                value = getattr(args, name)
                if value is not None:
                    command.extend(['--' + name.replace('_', '-'),
                                    str(value.resolve()) if isinstance(value, Path) else str(value)])
            for name in ('prepare_only', 'collect', 'evaluate_test'):
                if getattr(args, name):
                    command.append('--' + name.replace('_', '-'))
            os.execv(sys.executable, command)
    sys.path[:0] = [str(output / 'code'), str(output / 'code/src')]
    import final_core as core
    config = json.loads((output / 'config.json').read_text())
    core.validate(config)
    if args.worker:
        core.worker(output, json.loads(args.worker), config)
        return
    if args.import_pilot:
        core.import_pilot(output, args.import_pilot.resolve(), config)
    if args.import_results:
        core.import_results(output, args.import_results.resolve(), config)
    if args.bundle:
        with tarfile.open(args.bundle, 'w:gz', compresslevel=1) as archive:
            archive.add(output, arcname=output.name,
                        filter=lambda info: None if '__pycache__' in Path(info.name).parts
                        or Path(info.name).name in {'.run.lock', '.final-run.lock'} else info)
        print(f'Transfer bundle: {args.bundle}')
    if args.export_results:
        if not args.role:
            parser.error('--export-results requires --role')
        core.export_results(output, args.export_results, args.role, config)
    if args.collect or args.evaluate_test:
        core.collect(output, config, args.dataset.resolve() if args.dataset else None,
                     evaluate_test=args.evaluate_test)
    if any([args.prepare_only, args.bundle, args.export_results, args.import_results,
            args.collect, args.evaluate_test]):
        print(f'Prepared/results: {output}. No training started.')
        return
    if not args.role:
        parser.error('Training requires --role thinkbook or --role vostro')
    workers = args.workers or config['default_workers'][args.role]
    if workers < 1 or workers > (os.cpu_count() or 1):
        parser.error('--workers must be between 1 and the logical CPU count')
    cutoff = datetime.fromisoformat(args.deadline or config['training_deadline'])
    if cutoff.tzinfo is None:
        parser.error('Deadline must include its timezone offset')
    with (output / '.final-run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another runner is active in this output directory')
        jobs = [j for j in core.jobs(config) if j['owner'] == args.role]
        print(f"{args.role}: {workers} worker processes, {len(jobs)} tasks; deadline {cutoff.isoformat()}", flush=True)
        from run_screening import write_json
        write_json(output / f'machine-{args.role}.json', {'host': platform.node(),
                   'platform': platform.platform(), 'workers': workers,
                   'logical_cpus': os.cpu_count(), 'deadline': cutoff.isoformat()})
        statuses = {}
        stop = threading.Event()
        # A thread only supervises a child process. CPU work is in separate
        # interpreters, so it is not serialized by Python's GIL.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, output, job, config['timeouts_seconds'][job['kind']],
                                   cutoff.timestamp(), stop): job for job in jobs}
            try:
                for future in as_completed(futures):
                    job = futures[future]
                    statuses[job['id']] = future.result()
                    print(f"Tasks processed {len(statuses)}/{len(jobs)} | {job['id']} | {statuses[job['id']]}", flush=True)
                    write_json(output / f'status-{args.role}.json', statuses)
            except KeyboardInterrupt:
                # Workers have their own durable per-round state. On Ctrl+C the
                # terminal also signals child processes in the foreground group.
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        if any(s not in ('complete', 'cached') for s in statuses.values()):
            raise SystemExit('Some tasks are incomplete; inspect task logs and resume with the same command.')


if __name__ == '__main__':
    main()
