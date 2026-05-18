#!/usr/bin/env python3
"""Diff CPU vs GPU end-to-end pipeline products (HDF5).

Loads matching HDF5 products from two `smallbaselineApp.py` workdirs
(CPU baseline and GPU candidate) and prints per-dataset deviation
metrics. Asserts a float32 round-off gate (`rms/cpu_abs_max < 1e-5`)
per Issue #21 acceptance criteria.

Targets seven products spanning upstream / downstream / final stages
of the pipeline:

    upstream of invert_network:
        timeseries.h5            (dataset: timeseries)
        temporalCoherence.h5     (dataset: temporalCoherence)
        numInvIfgram.h5          (dataset: mask)
    downstream of correct_topography:
        timeseries*demErr.h5     (dataset: timeseries, glob — actual
                                  filename depends on the template's
                                  tropo/deramp suffix accumulation,
                                  e.g. timeseries_ERA5_ramp_demErr.h5
                                  on Fernandina vs timeseries_demErr.h5
                                  on Kuju where tropo/deramp are off)
        demErr.h5                (dataset: dem)
    final pipeline products:
        velocity.h5              (dataset: velocity)
        geo_velocity.h5          (dataset: velocity, optional)

Files that are absent in both workdirs are skipped silently (e.g.
geo_velocity.h5 when geocoding is disabled by the template). Files
present in only one workdir are flagged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


# (subdir, pattern, dataset_key, required) — pattern may be a literal
# filename or a glob; for globs the *longest* match wins (so the most
# suffix-accumulated product is picked under MintPy's processing-
# chain naming convention).
PRODUCTS: tuple[tuple[str, str, str, bool], ...] = (
    ('.', 'timeseries.h5', 'timeseries', True),
    ('.', 'temporalCoherence.h5', 'temporalCoherence', True),
    ('.', 'numInvIfgram.h5', 'mask', True),
    ('.', 'timeseries*demErr.h5', 'timeseries', True),
    ('.', 'demErr.h5', 'dem', True),
    ('.', 'velocity.h5', 'velocity', True),
    ('geo', 'geo_velocity.h5', 'velocity', False),
)

# Gate per Issue #21 acceptance criteria: rms/scale < 1e-5 (float32
# round-off envelope). invert_network uses 1e-4 upstream; tightening
# to 1e-5 here is justified by the observed correct_topography
# headroom (rms/scale 2e-8 to 7e-7 on Galapagos, ~1.5 decades below
# the gate per memory `project_correct_topography_plan.md`).
GATE_RMS_OVER_SCALE = 1e-5


def _read_dataset(h5path: Path, key: str) -> np.ndarray:
    """Load the named dataset from an HDF5 file as a NumPy array."""
    with h5py.File(h5path, 'r') as f:
        if key not in f:
            raise KeyError(f'{h5path}: dataset {key!r} not found '
                           f'(available: {list(f.keys())})')
        return f[key][()]


def _diff_metrics(cpu: np.ndarray, gpu: np.ndarray) -> dict:
    """Compute deviation metrics between two arrays of identical shape.

    Mirrors compare_dem_error_outputs._diff_metrics(): NaN-safe, uses
    float64 accumulation, reports rms_over_scale as the gate-relevant
    relative metric.
    """
    if cpu.shape != gpu.shape:
        raise ValueError(f'shape mismatch: cpu={cpu.shape} gpu={gpu.shape}')

    cpu64 = cpu.astype(np.float64, copy=False)
    gpu64 = gpu.astype(np.float64, copy=False)
    diff = cpu64 - gpu64

    finite = np.isfinite(diff)
    diff_finite = diff[finite]
    cpu_finite = cpu64[finite]

    abs_max = float(np.abs(cpu_finite).max()) if cpu_finite.size else float('nan')
    rms = float(np.sqrt(np.mean(diff_finite**2))) if diff_finite.size else float('nan')
    max_abs = float(np.abs(diff_finite).max()) if diff_finite.size else float('nan')
    rms_rel = rms / abs_max if abs_max > 0 else float('nan')

    return {
        'shape': list(cpu.shape),
        'dtype_cpu': str(cpu.dtype),
        'dtype_gpu': str(gpu.dtype),
        'rms': rms,
        'max_abs_diff': max_abs,
        'cpu_abs_max': abs_max,
        'rms_over_scale': rms_rel,
        'nan_count_cpu': int(np.isnan(cpu64).sum()),
        'nan_count_gpu': int(np.isnan(gpu64).sum()),
    }


def _resolve(workdir: Path, subdir: str, pattern: str) -> Path | None:
    """Resolve a (subdir, pattern) spec to a file path in workdir.

    Literal patterns return workdir/subdir/pattern if it exists. Glob
    patterns return the *longest* match by filename — under MintPy's
    suffix-accumulation convention this is the most-processed product
    (e.g. timeseries_ERA5_ramp_demErr.h5 wins over timeseries_demErr.h5
    when both exist).
    """
    base = workdir / subdir if subdir != '.' else workdir
    if '*' in pattern or '?' in pattern:
        matches = sorted(base.glob(pattern), key=lambda p: len(p.name),
                         reverse=True)
        return matches[0] if matches else None
    p = base / pattern
    return p if p.is_file() else None


def _compare_one(cpu_wd: Path, gpu_wd: Path,
                 subdir: str, pattern: str,
                 key: str, required: bool) -> dict:
    """Diff one product across the two workdirs."""
    cpu_h5 = _resolve(cpu_wd, subdir, pattern)
    gpu_h5 = _resolve(gpu_wd, subdir, pattern)
    spec = f'{subdir}/{pattern}' if subdir != '.' else pattern

    if cpu_h5 is None and gpu_h5 is None:
        return {'file': spec, 'status': 'absent_in_both',
                'required': required}
    if cpu_h5 is None or gpu_h5 is None:
        return {'file': spec, 'status': 'absent_in_one',
                'cpu_path': str(cpu_h5) if cpu_h5 else None,
                'gpu_path': str(gpu_h5) if gpu_h5 else None,
                'required': required}

    # For globs, cpu and gpu should resolve to the same filename;
    # if not, flag it as a structural mismatch (different processing
    # chains produced different product names).
    if cpu_h5.name != gpu_h5.name:
        return {'file': spec, 'status': 'filename_mismatch',
                'cpu_path': str(cpu_h5),
                'gpu_path': str(gpu_h5),
                'required': required}

    try:
        cpu = _read_dataset(cpu_h5, key)
        gpu = _read_dataset(gpu_h5, key)
    except KeyError as e:
        return {'file': spec, 'status': 'dataset_key_error',
                'error': str(e),
                'required': required}

    metrics = _diff_metrics(cpu, gpu)
    rms_rel = metrics['rms_over_scale']
    gate_pass = (np.isfinite(rms_rel) and rms_rel < GATE_RMS_OVER_SCALE)
    return {
        'file': spec,
        'resolved_filename': cpu_h5.name,
        'status': 'compared',
        'cpu_path': str(cpu_h5),
        'gpu_path': str(gpu_h5),
        'dataset': key,
        'gate_threshold': GATE_RMS_OVER_SCALE,
        'gate_pass': bool(gate_pass),
        **metrics,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cpu-workdir', required=True,
                   help='Workdir of the CPU end-to-end run')
    p.add_argument('--gpu-workdir', required=True,
                   help='Workdir of the GPU (torch) end-to-end run')
    p.add_argument('--out', default=None,
                   help='Optional path to write the full diff report as JSON')
    args = p.parse_args()

    cpu_wd = Path(args.cpu_workdir).resolve()
    gpu_wd = Path(args.gpu_workdir).resolve()

    results = []
    for subdir, pattern, key, required in PRODUCTS:
        results.append(_compare_one(cpu_wd, gpu_wd, subdir, pattern,
                                    key, required))

    # Summary roll-up.
    compared = [r for r in results if r['status'] == 'compared']
    n_compared = len(compared)
    n_pass = sum(1 for r in compared if r['gate_pass'])
    n_fail = n_compared - n_pass
    bad_status = ('absent_in_one', 'absent_in_both', 'filename_mismatch',
                  'dataset_key_error')
    n_missing_required = sum(
        1 for r in results
        if r['status'] in bad_status and r.get('required')
    )

    report = {
        'cpu_workdir': str(cpu_wd),
        'gpu_workdir': str(gpu_wd),
        'gate_threshold_rms_over_scale': GATE_RMS_OVER_SCALE,
        'summary': {
            'n_compared': n_compared,
            'n_gate_pass': n_pass,
            'n_gate_fail': n_fail,
            'n_missing_required': n_missing_required,
        },
        'products': results,
    }

    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))

    # Exit non-zero if any gate failed or any required product is
    # missing — lets the harness short-circuit on regression.
    if n_fail > 0 or n_missing_required > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
