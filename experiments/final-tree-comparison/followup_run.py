"""Prepare and run bagging extensions and split-criterion follow-ups."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import threading

HERE = Path(__file__).resolve().parent


def prepare(output, config_path):
    """Freeze development-only inputs, then import checked existing models."""
    import run as pilot_run
    existed = (output / 'manifest.json').exists()
    config = json.loads(config_path.read_text())
    pilot_run.prepare(output, config_path)
    if existed:
        if not json.loads((output / 'manifest.json').read_text()).get('followups'):
            raise ValueError('Use a fresh follow-up output directory.')
        if not (output / 'baseline_import.json').exists():
            raise ValueError('Baseline import did not finish; use a fresh follow-up output directory.')
        return
    for name in ('followup_run.py', 'followup_core.py', 'final_run.py', 'final_core.py'):
        shutil.copy2(HERE / name, output / 'code' / name)
    shutil.copy2(HERE / 'FOLLOWUPS.md', output / 'FOLLOWUPS.md')
    manifest = json.loads((output / 'manifest.json').read_text())
    for path in [*[(output / 'code' / name) for name in
                  ('followup_run.py', 'followup_core.py', 'final_run.py', 'final_core.py')],
                 output / 'FOLLOWUPS.md']:
        manifest['sha256'][str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['followups'] = True
    manifest['plan_id'] = hashlib.sha256(json.dumps(manifest['sha256'], sort_keys=True).encode()).hexdigest()
    pilot_run.write_json(output / 'manifest.json', manifest)
    (output / 'tasks').mkdir()
    sys.path[:0] = [str(output / 'code'), str(output / 'code/src')]
    import followup_core as core
    core.validate(config)
    core.import_baselines(output, (config_path.parent / config['source_final']).resolve(), config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/followups')
    parser.add_argument('--config', type=Path, default=HERE / 'followup_config.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--deadline', help='Optional timezone-aware ISO cutoff; otherwise finish all tasks')
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--export-results', type=Path)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.worker:
        if not args.config.exists() and (output / 'config.json').exists():
            args.config = output / 'config.json'
        prepare(output, args.config.resolve())
        frozen = output / 'code/followup_run.py'
        if Path(__file__).resolve() != frozen.resolve():
            command = [sys.executable, str(frozen), '--output', str(output),
                       '--config', str(output / 'config.json')]
            for name in ('bundle', 'workers', 'deadline', 'export_results'):
                value = getattr(args, name)
                if value is not None:
                    command.extend(['--' + name.replace('_', '-'),
                                    str(value.resolve()) if isinstance(value, Path) else str(value)])
            for name in ('prepare_only', 'collect'):
                if getattr(args, name):
                    command.append('--' + name.replace('_', '-'))
            os.execv(sys.executable, command)
    sys.path[:0] = [str(output / 'code'), str(output / 'code/src')]
    import followup_core as core
    from final_core import check_task
    from final_run import execute
    from run_screening import write_json
    config = json.loads((output / 'config.json').read_text())
    core.validate(config)
    if args.worker:
        core.worker(output, json.loads(args.worker), config)
        return
    if args.bundle:
        with tarfile.open(args.bundle, 'w:gz', compresslevel=1) as archive:
            archive.add(output, arcname=output.name,
                filter=lambda info: None if '__pycache__' in Path(info.name).parts
                or Path(info.name).name in {'.run.lock', '.final-run.lock', '.followup.lock'} else info)
        # Verify the actual archive, including uv.lock, before offering it.
        with tarfile.open(args.bundle) as archive:
            names = set(archive.getnames())
            manifest = json.loads((output / 'manifest.json').read_text())
            missing = [p for p in manifest['sha256'] if output.name + '/' + p not in names]
            if missing:
                raise ValueError(f'Bundle omitted inputs: {missing}')
        print(f'Bundle verified: {args.bundle}')
    if args.collect:
        core.collect(output, config)
    if args.export_results:
        with tarfile.open(args.export_results, 'w:gz', compresslevel=1) as archive:
            archive.add(output, arcname=output.name,
                filter=lambda info: None if '__pycache__' in Path(info.name).parts
                or Path(info.name).name in {'.run.lock', '.final-run.lock', '.followup.lock'} else info)
        print(f'Results exported: {args.export_results}')
    if args.prepare_only or args.bundle or args.collect or args.export_results:
        return
    workers = config['workers'] if args.workers is None else args.workers
    if workers < 1 or workers > (os.cpu_count() or 1):
        parser.error('Worker count must be within the logical CPU count.')
    deadline = float('inf')
    if args.deadline:
        value = datetime.fromisoformat(args.deadline)
        if value.tzinfo is None:
            parser.error('--deadline requires a timezone offset.')
        deadline = value.timestamp()
    with (output / '.followup.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another follow-up runner is active in this output directory.')
        all_jobs = core.jobs(config)
        pending = [j for j in all_jobs if not check_task(output, j)]
        print(f'Follow-ups: {workers} processes; {len(pending)} new/incomplete tasks; '
              f'{len(all_jobs) - len(pending)} cached. Development data only.', flush=True)
        stop = threading.Event()
        statuses = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, output, j, config['timeouts_seconds'][j['kind']],
                       deadline, stop, 'followup_run.py'): j for j in pending}
            try:
                for future in as_completed(futures):
                    job = futures[future]
                    statuses[job['id']] = future.result()
                    print(f"Tasks processed {len(statuses)}/{len(pending)} | {job['id']} | {statuses[job['id']]}", flush=True)
                    write_json(output / 'status.json', statuses)
            except KeyboardInterrupt:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        core.collect(output, config)
        if any(value not in ('complete', 'cached') for value in statuses.values()):
            raise SystemExit('Some follow-up tasks are incomplete; resume with the same command.')


if __name__ == '__main__':
    main()
