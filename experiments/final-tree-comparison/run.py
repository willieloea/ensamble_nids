"""Freeze and run the sequential runtime pilot without final-test access."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare(output, config_path):
    """Snapshot all executable dependencies and development data once."""
    config = json.loads(config_path.read_text())
    if (output / 'manifest.json').exists():
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('Configuration changed; select a fresh --output directory.')
        verify(output)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output without a manifest; select a fresh --output.')
    source = (config_path.parent / config['development_pools']).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    code = output / 'code'
    code.mkdir()
    shutil.copytree(ROOT / 'src', code / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('run.py', 'pilot_core.py'):
        shutil.copy2(HERE / name, code / name)
    shutil.copy2(ROOT / 'experiments/overnight-learning-curves/overnight_core.py', code)
    shutil.copy2(ROOT / 'scripts/run_screening.py', code)
    shutil.copy2(source, output / 'development_pools.pkl')
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copy2(ROOT / name, code)
    write_json(output / 'config.json', config)
    (output / 'jobs').mkdir()
    (output / 'attempts').mkdir()
    paths = [output / 'config.json', output / 'development_pools.pkl',
             *code.rglob('*.py'), code / 'pyproject.toml', code / 'uv.lock']
    write_json(output / 'manifest.json', {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python': sys.version, 'platform': platform.platform(),
        'logical_cpus': os.cpu_count(), 'processor': platform.processor(),
        'load_average_at_preparation': os.getloadavg(), 'source_data': str(source),
        'final_test_loaded': False,
        'sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(paths)},
    })


def verify(output):
    """Reject corrupted or edited frozen inputs before resuming."""
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {name}')


def execute(output, job, timeout, deadline, bar):
    """Run a fresh worker; preserve incomplete attempts and enforce full timeouts."""
    path = output / 'jobs' / job['id']
    result_path = path / 'result.json'
    if result_path.exists() and json.loads(result_path.read_text())['status'] == 'complete':
        return True
    if deadline is not None and deadline - time.monotonic() < timeout:
        return False
    if path.exists():
        archive = output / 'attempts' / job['id']
        archive.mkdir(exist_ok=True)
        shutil.move(str(path), archive / str(time.time_ns()))
    path.mkdir()
    write_json(result_path, {'job': job, 'status': 'running'})
    command = [sys.executable, str(output / 'code/run.py'), '--output', str(output),
               '--worker', json.dumps(job)]
    started = time.monotonic()
    process = None
    status = 'failed'
    try:
        with (path / 'worker.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    status = 'timeout'
                    process.terminate()
                    break
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            if status != 'timeout':
                status = 'complete' if process.returncode == 0 else 'failed'
    except KeyboardInterrupt:
        status = 'interrupted'
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        result = json.loads(result_path.read_text())
        if result.get('status') != 'complete' or status != 'complete':
            result.update(status=status, job=job)
        result['process_wall_seconds'] = time.monotonic() - started
        write_json(result_path, result)
    return status == 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/runtime-pilot')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--seconds', type=float, help='Invocation budget; jobs get full configured timeouts')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error('--seconds must be positive')
    output = args.output.resolve()
    if not args.worker:
        if args.summarize_only:
            verify(output)
        else:
            prepare(output, args.config.resolve())
        # Resume the frozen controller too, not just frozen worker/model code.
        frozen_entry = output / 'code/run.py'
        if Path(__file__).resolve() != frozen_entry.resolve():
            command = [sys.executable, str(frozen_entry), '--output', str(output),
                       '--config', str(output / 'config.json')]
            if args.prepare_only:
                command.append('--prepare-only')
            if args.summarize_only:
                command.append('--summarize-only')
            if args.seconds is not None:
                command.extend(['--seconds', str(args.seconds)])
            os.execv(sys.executable, command)
    sys.path[:0] = [str(output / 'code'), str(output / 'code/src')]
    import pilot_core as core
    config = json.loads((output / 'config.json').read_text())
    core.validate_config(config)
    if args.worker:
        core.worker(output, json.loads(args.worker), config)
    elif args.prepare_only:
        core.load_pools(output, config)
        print(f'Prepared {output}. No training started.')
    elif args.summarize_only:
        core.summarize(output, config)
    else:
        with (output / '.run.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error('Another pilot runner is using this output directory.')
            deadline = None if args.seconds is None else time.monotonic() + args.seconds
            if not core.run(output, config, execute, deadline):
                raise SystemExit('Pilot incomplete: inspect status.json and jobs/*/worker.log; rerun to retry.')


if __name__ == '__main__':
    main()
