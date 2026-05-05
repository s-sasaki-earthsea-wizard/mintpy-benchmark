#!/usr/bin/env python3
"""Compare two timeseries.h5 files produced by different invert_network solvers.

Computes per-pixel RMS of ``ts_a - ts_b`` over the time axis, normalised
by the per-pixel temporal std of ``ts_b`` (the reference). The normalised
RMS is the relevant quantity for the solver comparison: it asks "how
large is the disagreement relative to the signal amplitude this pixel
actually carries". An absolute RMS without that normalisation would be
dominated by quiet pixels (where both solutions are tiny and noise is
amplified by division), or hide large absolute errors at high-amplitude
pixels.

Pixels where ``std(ts_b, axis=0)`` is below ``--min-signal`` are excluded
from the normalised statistics (otherwise a pixel with no signal would
produce a 0/0 = NaN, or near-zero / near-zero = inflated ratio). They
are still counted in the absolute-RMS statistics.

Output:
  stdout      human-readable summary
  --json-out  optional JSON dump for machine consumption (report ingest)

Usage:
  python compare_solutions.py --a timeseries_chol.h5 --b timeseries_lstsq.h5
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def load_ts(path: Path) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        return f['timeseries'][:]


def per_pixel_rms(diff: np.ndarray) -> np.ndarray:
    """RMS along the time axis, returns shape (H, W)."""
    return np.sqrt(np.mean(diff.astype(np.float64) ** 2, axis=0))


def percentile_table(values: np.ndarray) -> dict:
    """Standard summary statistics, robust to NaN."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'count': 0}
    return {
        'count': int(finite.size),
        'min': float(np.min(finite)),
        'median': float(np.median(finite)),
        'p99': float(np.percentile(finite, 99)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, type=Path,
                   help='First timeseries.h5 (typically cholesky)')
    p.add_argument('--b', required=True, type=Path,
                   help='Second timeseries.h5, used as reference for normalisation '
                        '(typically lstsq)')
    p.add_argument('--min-signal', type=float, default=1e-6,
                   help='Pixels with std(ts_b) below this are excluded from '
                        'normalised stats (default: 1e-6 m)')
    p.add_argument('--json-out', type=Path, default=None,
                   help='Optional JSON path for machine-readable output')
    args = p.parse_args()

    print(f'[compare_solutions] a = {args.a}', flush=True)
    print(f'[compare_solutions] b = {args.b} (reference for normalisation)',
          flush=True)

    ts_a = load_ts(args.a)
    ts_b = load_ts(args.b)

    if ts_a.shape != ts_b.shape:
        print(f'ERROR: shape mismatch: a={ts_a.shape} vs b={ts_b.shape}',
              file=sys.stderr)
        return 1

    diff = ts_a - ts_b
    rms = per_pixel_rms(diff)
    signal = np.std(ts_b.astype(np.float64), axis=0)

    valid = signal >= args.min_signal
    rms_norm = np.where(valid, rms / np.maximum(signal, args.min_signal), np.nan)

    out = {
        'shape': list(ts_a.shape),
        'min_signal_threshold_m': args.min_signal,
        'pixels_total': int(rms.size),
        'pixels_above_signal_threshold': int(valid.sum()),
        'absolute_rms_m': percentile_table(rms),
        'normalised_rms_dimensionless': percentile_table(rms_norm),
        'reference_signal_std_m': percentile_table(signal[valid]),
    }

    print()
    print('shape:', out['shape'])
    print(f'pixels total                    : {out["pixels_total"]:>10d}')
    print(f'pixels with std(ts_b) >= {args.min_signal:g}: '
          f'{out["pixels_above_signal_threshold"]:>10d}')
    print()

    print('absolute RMS (m), all pixels:')
    for k, v in out['absolute_rms_m'].items():
        print(f'  {k:<8} = {v}')
    print()
    print('normalised RMS = per_pixel_rms(diff) / per_pixel_std(ts_b):')
    for k, v in out['normalised_rms_dimensionless'].items():
        print(f'  {k:<8} = {v}')
    print()
    print('reference signal std (m), valid pixels only:')
    for k, v in out['reference_signal_std_m'].items():
        print(f'  {k:<8} = {v}')

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f'\n[compare_solutions] JSON written -> {args.json_out}', flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
