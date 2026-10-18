"""Prepare and run the frozen overnight experiment; no final test access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare(output, config_path):
    """Freeze code, configuration and development partitions before fitting."""
    config = json.loads(config_path.read_text())
    if (output / 'manifest.json').exists():
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('Configuration changed: use a new --output directory.')
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is nonempty without a manifest; use a fresh directory.')
    source = (config_path.parent / config['source_experiment']).resolve()
    required = [source / 'development_pools.pkl', source / 'manifest.json']
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    code = output / 'code'
    code.mkdir()
    shutil.copytree(ROOT / 'src', code / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['run.py', 'overnight_core.py']:
        shutil.copy2(HERE / name, code / name)
    shutil.copy2(ROOT / 'scripts/run_screening.py', code / 'run_screening.py')
    for name in ['pyproject.toml', 'uv.lock']:
        shutil.copy2(ROOT / name, code / name)
    shutil.copy2(required[0], output / 'development_pools.pkl')
    shutil.copy2(required[1], output / 'source_manifest.json')
    atomic_json(output / 'config.json', config)
    for name in ['points', 'models', 'jobs', 'progress', 'logs', 'plots']:
        (output / name).mkdir()
    hashes = {}
    for path in sorted([output / 'development_pools.pkl', output / 'source_manifest.json',
                        output / 'config.json', *code.rglob('*.py'), code / 'uv.lock',
                        code / 'pyproject.toml']):
        hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(output / 'manifest.json', {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python': sys.version, 'source_experiment': str(source), 'sha256': hashes,
        'final_test_evaluated': False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--seconds', type=float, help='Override this invocation wall-clock budget')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.monotonic()
    if not args.worker:
        if args.summarize_only:
            if not (output / 'manifest.json').exists():
                parser.error('No prepared results at --output.')
        else:
            prepare(output, args.config.resolve())
    sys.path.insert(0, str(output / 'code/src'))
    sys.path.insert(0, str(output / 'code'))
    import overnight_core as core
    config = json.loads((output / 'config.json').read_text())
    if args.worker:
        core.worker(output, json.loads(args.worker), config)
    elif args.prepare_only:
        core.validate_pools(output, config)
        core.summarize(output, config)
        print(f'Prepared {output}. No training started.', flush=True)
    elif args.summarize_only:
        core.summarize(output, config)
    else:
        seconds = config['budget_seconds'] if args.seconds is None else args.seconds
        if seconds <= 0:
            parser.error('--seconds must be positive')
        core.run(output, config, started + seconds)


if __name__ == '__main__':
    main()
