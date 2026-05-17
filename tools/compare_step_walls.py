#!/usr/bin/env python3
"""Diff per-step wall times between CPU and GPU end-to-end runs.

Reads two summary.tsv files produced by run_bench.sh (one per
run_end_to_end_bench.sh invocation) and prints per-step deltas. Splits
the steps into two classes per Issue #21 acceptance criteria:

    GPU-able    invert_network, correct_topography  → expect speedup
    CPU-only    everything else                     → expect ±5% control

Reports the headline GPU-able subtotal speedup, fails if any CPU-only
step's wall changes by more than ±5% (regression signal), and fails
if any step exited non-zero in either run.

Input summary.tsv format (tab-separated, with header):
    step  wall_seconds  max_rss_kb  exit_code
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# Steps that route through a torch dispatch when the relevant solver
# flag is set. All other steps in the default 18-step chain stay on
# CPU regardless of solver flags and serve as the I/O / cache control.
GPU_ABLE_STEPS: frozenset[str] = frozenset({
    'invert_network',
    'correct_topography',
})

# Tolerance for CPU-only step wall regression. Set per Issue #21
# acceptance criteria; wider than typical run-to-run noise (~1-2%
# on warm SSD) so that steady-state load fluctuations don't trip it.
CPU_ONLY_TOLERANCE_FRAC = 0.05


def _read_summary(path: Path) -> dict[str, dict]:
    """Parse summary.tsv into {step_name: {wall_s, rss_kb, exit_code}}."""
    rows: dict[str, dict] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows[row['step']] = {
                'wall_s': float(row['wall_seconds']),
                'rss_kb': int(row['max_rss_kb']),
                'exit_code': int(row['exit_code']),
            }
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cpu-summary', required=True,
                   help='Path to summary.tsv from the CPU run')
    p.add_argument('--gpu-summary', required=True,
                   help='Path to summary.tsv from the GPU (torch) run')
    p.add_argument('--out', default=None,
                   help='Optional path to write the full diff report as JSON')
    p.add_argument('--cpu-tolerance', type=float,
                   default=CPU_ONLY_TOLERANCE_FRAC,
                   help=f'CPU-only step wall tolerance (default '
                        f'{CPU_ONLY_TOLERANCE_FRAC:.2f} = '
                        f'±{CPU_ONLY_TOLERANCE_FRAC * 100:.0f}%%)')
    args = p.parse_args()

    cpu_rows = _read_summary(Path(args.cpu_summary))
    gpu_rows = _read_summary(Path(args.gpu_summary))

    # Walk the union of step names so missing entries surface as None.
    all_steps = list(dict.fromkeys(list(cpu_rows) + list(gpu_rows)))

    per_step: list[dict] = []
    cpu_only_breaches: list[str] = []
    nonzero_exit: list[str] = []
    gpu_able_cpu_total = 0.0
    gpu_able_gpu_total = 0.0
    cpu_only_cpu_total = 0.0
    cpu_only_gpu_total = 0.0

    for step in all_steps:
        c = cpu_rows.get(step)
        g = gpu_rows.get(step)
        is_gpu_able = step in GPU_ABLE_STEPS
        entry: dict = {
            'step': step,
            'class': 'gpu_able' if is_gpu_able else 'cpu_only',
            'cpu_wall_s': c['wall_s'] if c else None,
            'gpu_wall_s': g['wall_s'] if g else None,
            'cpu_exit': c['exit_code'] if c else None,
            'gpu_exit': g['exit_code'] if g else None,
        }

        if c and g and c['wall_s'] > 0:
            speedup = c['wall_s'] / g['wall_s'] if g['wall_s'] > 0 else float('inf')
            delta_frac = (g['wall_s'] - c['wall_s']) / c['wall_s']
            entry['speedup'] = speedup
            entry['delta_frac'] = delta_frac
            if is_gpu_able:
                gpu_able_cpu_total += c['wall_s']
                gpu_able_gpu_total += g['wall_s']
            else:
                cpu_only_cpu_total += c['wall_s']
                cpu_only_gpu_total += g['wall_s']
                if abs(delta_frac) > args.cpu_tolerance:
                    entry['regression'] = True
                    cpu_only_breaches.append(step)
                else:
                    entry['regression'] = False

        if c and c['exit_code'] != 0:
            nonzero_exit.append(f'{step}[cpu]')
        if g and g['exit_code'] != 0:
            nonzero_exit.append(f'{step}[gpu]')

        per_step.append(entry)

    headline_gpu_speedup = (
        gpu_able_cpu_total / gpu_able_gpu_total
        if gpu_able_gpu_total > 0 else float('nan')
    )
    cpu_only_ratio = (
        cpu_only_gpu_total / cpu_only_cpu_total
        if cpu_only_cpu_total > 0 else float('nan')
    )

    report = {
        'cpu_summary': str(Path(args.cpu_summary).resolve()),
        'gpu_summary': str(Path(args.gpu_summary).resolve()),
        'cpu_only_tolerance_frac': args.cpu_tolerance,
        'gpu_able_steps': sorted(GPU_ABLE_STEPS),
        'totals': {
            'gpu_able_cpu_wall_s': gpu_able_cpu_total,
            'gpu_able_gpu_wall_s': gpu_able_gpu_total,
            'gpu_able_speedup': headline_gpu_speedup,
            'cpu_only_cpu_wall_s': cpu_only_cpu_total,
            'cpu_only_gpu_wall_s': cpu_only_gpu_total,
            'cpu_only_ratio_gpu_over_cpu': cpu_only_ratio,
        },
        'cpu_only_regressions': cpu_only_breaches,
        'nonzero_exits': nonzero_exit,
        'per_step': per_step,
    }

    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))

    # Exit non-zero on any CPU-only regression or any non-zero step
    # exit — the harness uses this as the regression gate.
    if cpu_only_breaches or nonzero_exit:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
