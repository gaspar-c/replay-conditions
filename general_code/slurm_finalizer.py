"""
slurm_finalizer.py
Called by the SLURM finalizer job after all array tasks finish.
Queries sacct for per-task stats and writes a summary to the group log.

Usage (internal): python -m general_code.slurm_finalizer <job_ids_csv> <group_log> <label>
"""
import sys
import os
import subprocess
from datetime import datetime
from general_code.aux_functions import xprint, seconds_to_hhmmss

_FAIL_STATES = {'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'}


def _elapsed_to_seconds(elapsed):
    """Convert sacct elapsed string (HH:MM:SS or D-HH:MM:SS) to integer seconds."""
    if '-' in elapsed:
        days, rest = elapsed.split('-', 1)
        h, m, s = rest.split(':')
        return int(days) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
    parts = elapsed.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def main(job_ids_csv, group_log, label):
    now = datetime.now()
    submitted_at = datetime.fromtimestamp(os.path.getmtime(group_log))
    wall_secs = int((now - submitted_at).total_seconds())
    xprint(
        f'Finalizer running at {now.strftime("%Y-%m-%d %H:%M:%S")}  '
        f'(submitted {submitted_at.strftime("%H:%M:%S")}, '
        f'total wall time {seconds_to_hhmmss(wall_secs)})',
        group_log,
    )

    result = subprocess.run(
        ['sacct', '-j', job_ids_csv,
         '--format=JobID,NodeList,Elapsed,State,ExitCode',
         '--noheader', '--parsable2'],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        xprint(f'[finalizer] sacct failed: {result.stderr.strip()}', group_log)
        return

    rows = []
    n_ok = n_fail = n_other = 0
    times = []
    node_times = {}   # node -> list of seconds

    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split('|')
        if len(parts) < 5:
            continue
        job_id, node, elapsed, state, exit_code = parts[:5]
        # Skip batch/extern sub-steps (e.g. "257052_1.batch")
        if '.' in job_id:
            continue
        # Skip the warmup job (non-array job submitted before arrays)
        if '_' not in job_id:
            continue

        secs = _elapsed_to_seconds(elapsed)
        times.append(secs)
        node_times.setdefault(node, []).append(secs)

        if state == 'COMPLETED':
            n_ok += 1
        elif state in _FAIL_STATES:
            n_fail += 1
        else:
            n_other += 1

        rows.append((job_id, node, elapsed, state, exit_code))

    total = n_ok + n_fail + n_other
    xprint(
        f'--- Batch {label} finished: {n_ok}/{total} OK, '
        f'{n_fail} FAILED, {n_other} other ---',
        group_log,
    )

    if times:
        avg_s = int(sum(times) / len(times))
        xprint(
            f'    Duration  min={seconds_to_hhmmss(min(times))}  '
            f'avg={seconds_to_hhmmss(avg_s)}  '
            f'max={seconds_to_hhmmss(max(times))}',
            group_log,
        )

    for node, t in sorted(node_times.items()):
        avg_s = int(sum(t) / len(t))
        xprint(
            f'    {node:<20} {len(t):>4} jobs  '
            f'min={seconds_to_hhmmss(min(t))}  '
            f'avg={seconds_to_hhmmss(avg_s)}  '
            f'max={seconds_to_hhmmss(max(t))}',
            group_log,
        )

    # Log all non-OK jobs; skip the (usually long) list of successes
    failed_rows = [(jid, nd, el, st, ex) for jid, nd, el, st, ex in rows if st != 'COMPLETED']
    if failed_rows:
        xprint(f'    Failed / non-completed jobs ({len(failed_rows)}):', group_log)
        for job_id, node, elapsed, state, exit_code in failed_rows:
            xprint(
                f'      {job_id:<24} {node:<20} {elapsed:>10}  {state:<16} exit={exit_code}',
                group_log,
            )
    else:
        xprint('    All jobs completed successfully.', group_log)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python -m general_code.slurm_finalizer <job_ids_csv> <group_log> <label>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
