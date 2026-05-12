#!/usr/bin/env python3
"""Direct-Python-call benchmark for the correct_topography (DEM error) step.

Calls ``mintpy.dem_error.correct_dem_error_patch`` with ``solver='cpu'`` or
``'torch'`` on the full image extent of a MintPy work directory, then writes
the three output arrays (``delta_z``, ``ts_cor``, ``ts_res``) as ``.npy``
files alongside a ``metrics.json`` recording wall time, peak host RSS, and
(for the torch solver) peak CUDA VRAM.

Why direct call vs ``smallbaselineApp.py --dostep correct_topography``: the
production CLI wraps the patch call in update-mode + h5 writeback + per-box
split. None of that is on the GPU's critical path. Calling the patch
function directly isolates the solver-dispatched cost we are trying to
measure and lets us reuse the same fitted output arrays for an apples-to-
apples numerical diff between CPU and GPU.

Production fidelity preserved: ``G_defo`` and ``date_flag`` are constructed
from the same template helpers that ``mintpy.dem_error.correct_dem_error``
uses (``get_design_matrix4defo`` + ``read_exclude_date``), so polynomial
order, step dates, and excluded dates match what the CLI would have
produced.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mintpy.dem_error import (
    correct_dem_error_patch,
    get_design_matrix4defo,
    read_exclude_date,
)
from mintpy.objects import timeseries
from mintpy.utils import ptime


def _build_inps(ts_file: str,
                poly_order: int,
                step_dates: list[str],
                exclude_dates: list[str],
                phase_velocity: bool) -> SimpleNamespace:
    """Mirror the subset of ``inps`` that ``get_design_matrix4defo`` reads.

    ``get_design_matrix4defo`` consumes ``polyOrder``, ``stepDate``,
    ``periodic``, ``phaseVelocity``, ``ts_file``. We construct only those
    attributes plus ``excludeDate`` (used downstream for ``date_flag``).
    """
    return SimpleNamespace(
        ts_file=ts_file,
        polyOrder=poly_order,
        stepDate=ptime.yyyymmdd(step_dates) if step_dates else [],
        excludeDate=ptime.yyyymmdd(exclude_dates) if exclude_dates else [],
        periodic=[],
        phaseVelocity=phase_velocity,
    )


def _max_rss_kb() -> int:
    """Peak resident set size of this process in kilobytes (Linux semantics)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--ts-file', required=True,
                   help='Time-series HDF5 (e.g. timeseries_ERA5_ramp.h5)')
    p.add_argument('--geom-file', required=True,
                   help='Geometry HDF5 (geometryRadar.h5 or geometryGeo.h5)')
    p.add_argument('--solver', required=True, choices=['cpu', 'torch'],
                   help='cpu = scipy per-pixel loop, torch = batched Cholesky on CUDA')
    p.add_argument('--log-dir', required=True,
                   help='Output directory for .npy arrays + metrics.json + stdout log')
    p.add_argument('--poly-order', type=int, default=2,
                   help='Polynomial order of the deformation model (default: 2, '
                        'matches mintpy.topographicResidual.polyOrder = auto)')
    p.add_argument('--step-date', nargs='*', default=[],
                   help='Step-jump dates (YYYYMMDD), pass through to G_defo')
    p.add_argument('--exclude-date', nargs='*', default=[],
                   help='Dates to exclude from the fit')
    p.add_argument('--chunk-size', type=int, default=None,
                   help='GPU chunk size (pixels). None => auto from free VRAM. '
                        'Ignored when --solver=cpu.')
    p.add_argument('--phase-velocity', action='store_true',
                   help='Minimise phase velocity instead of phase. Not supported '
                        'on the torch solver yet; included for parity.')
    args = p.parse_args()

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build design matrix + date mask the same way the CLI does, so we
    # cannot drift from production with respect to which dates / what
    # polynomial basis enter the fit.
    inps = _build_inps(
        ts_file=args.ts_file,
        poly_order=args.poly_order,
        step_dates=args.step_date,
        exclude_dates=args.exclude_date,
        phase_velocity=args.phase_velocity,
    )
    G_defo = get_design_matrix4defo(inps)

    ts_obj = timeseries(args.ts_file)
    ts_obj.open(print_msg=False)
    date_flag = read_exclude_date(inps.excludeDate, ts_obj.dateList, print_msg=False)[0]

    if args.poly_order > int(np.sum(date_flag)):
        raise ValueError(
            f'poly_order={args.poly_order} > number of acquisitions used '
            f'({int(np.sum(date_flag))}); reduce it'
        )

    metrics: dict = {
        'solver': args.solver,
        'ts_file': args.ts_file,
        'geom_file': args.geom_file,
        'num_date': int(ts_obj.numDate),
        'image_shape': [int(ts_obj.length), int(ts_obj.width)],
        'poly_order': args.poly_order,
        'step_date': args.step_date,
        'exclude_date': args.exclude_date,
        'num_param_defo': int(G_defo.shape[1]),
        'chunk_size': args.chunk_size,
        'phase_velocity': bool(args.phase_velocity),
    }

    # VRAM peak instrumentation only matters for the torch path.
    if args.solver == 'torch':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('--solver=torch requires CUDA but torch.cuda.is_available() is False')
        # Reset peak stats so we measure this run only.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    patch_kwargs: dict = dict(
        G_defo=G_defo,
        ts_file=args.ts_file,
        geom_file=args.geom_file,
        box=None,  # full image — Fernandina fits in 1 box at default maxMemory
        date_flag=date_flag,
        phase_velocity=args.phase_velocity,
        solver=args.solver,
    )

    t0 = time.perf_counter()
    delta_z, ts_cor, ts_res, box = correct_dem_error_patch(**patch_kwargs)
    if args.solver == 'torch':
        import torch
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    metrics['wall_seconds'] = wall
    metrics['max_rss_kb'] = _max_rss_kb()

    if args.solver == 'torch':
        import torch
        metrics['cuda_peak_alloc_bytes'] = int(torch.cuda.max_memory_allocated())
        metrics['cuda_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
        metrics['cuda_device_name'] = torch.cuda.get_device_name(0)

    # Save outputs as .npy for downstream diff. Shapes:
    #   delta_z (num_row, num_col)
    #   ts_cor  (num_date, num_row, num_col)
    #   ts_res  (num_date, num_row, num_col)
    np.save(log_dir / 'delta_z.npy', delta_z)
    np.save(log_dir / 'ts_cor.npy', ts_cor)
    np.save(log_dir / 'ts_res.npy', ts_res)

    with open(log_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n[bench] solver={args.solver}  wall={wall:.3f}s  '
          f'max_rss={_max_rss_kb()/1024:.1f}MiB', flush=True)
    if args.solver == 'torch':
        print(f'[bench] cuda_peak_alloc={metrics["cuda_peak_alloc_bytes"]/2**20:.1f}MiB  '
              f'reserved={metrics["cuda_peak_reserved_bytes"]/2**20:.1f}MiB  '
              f'device={metrics["cuda_device_name"]}', flush=True)
    print(f'[bench] outputs => {log_dir}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
