#!/usr/bin/env python3
"""Run the build with a resource guard. Never stop existing services."""
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/remote-build'
GIB = 1024 ** 3


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resources():
    fields = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
    return {'at_utc': now(), 'available_memory_bytes': fields['MemAvailable'] * 1024, 'free_disk_bytes': shutil.disk_usage(ROOT).free, 'load_average': os.getloadavg()}


def stop_build(process):
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'started_at': now(), 'status': 'starting', 'max_jobs': int(os.environ.get('COMPAT_MAX_JOBS', '4'))}
    state_file = OUT / 'build-status.json'
    def save():
        temporary = state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_file)
    initial = resources()
    if initial['available_memory_bytes'] < 96 * GIB or initial['free_disk_bytes'] < 200 * GIB:
        state.update(status='resource_guard', resources=initial, exit_code=125)
        save()
        raise SystemExit(125)
    with (OUT / 'build.log').open('a', buffering=1) as log, (OUT / 'resources.jsonl').open('a', buffering=1) as samples:
        log.write('\nBUILD START ' + now() + '\n')
        process = subprocess.Popen(['bash', 'scripts/build-compat.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='running', pid=process.pid)
        save()
        guarded = False
        next_sample = 0
        try:
            while process.poll() is None:
                if time.monotonic() >= next_sample:
                    sample = resources()
                    samples.write(json.dumps(sample) + '\n')
                    if sample['available_memory_bytes'] < 96 * GIB or sample['free_disk_bytes'] < 200 * GIB:
                        guarded = True
                        state['resource_guard_sample'] = sample
                        stop_build(process)
                        break
                    next_sample = time.monotonic() + 60
                time.sleep(1)
        except BaseException:
            if process.poll() is None:
                stop_build(process)
            raise
        result = process.wait()
        state.update(status='resource_guard' if guarded else ('passed' if result == 0 else 'failed'), completed_at=now(), exit_code=result)
        save()
    raise SystemExit(125 if guarded else result)


if __name__ == '__main__':
    main()
