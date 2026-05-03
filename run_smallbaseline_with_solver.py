#!/usr/bin/env python3
"""Run smallbaselineApp.py with the torch backend's _SOLVER overridden by env var.

Why this exists:
  ``mintpy.ifgram_inversion_gpu._SOLVER`` is a module-level constant pinned
  to ``'cholesky'`` (the chosen default after PR #9). Comparing the
  retained ``'lstsq'`` path against ``'cholesky'`` would normally require
  editing the source. This wrapper monkey-patches the constant before the
  CLI is invoked, so the comparison harness stays entirely in this
  sibling repo and the upstream-tracked source is not touched.

Usage (drop-in for ``smallbaselineApp.py``):
  MINTPY_SOLVER=cholesky python run_smallbaseline_with_solver.py <template> --dostep invert_network
  MINTPY_SOLVER=lstsq    python run_smallbaseline_with_solver.py <template> --dostep invert_network

If ``MINTPY_SOLVER`` is unset, the in-tree default is used unchanged.
"""
import os
import sys


def main() -> int:
    import mintpy.ifgram_inversion_gpu as gpu_mod

    requested = os.environ.get('MINTPY_SOLVER')
    if requested is not None:
        if requested not in ('cholesky', 'lstsq'):
            print(f"ERROR: MINTPY_SOLVER must be 'cholesky' or 'lstsq', got {requested!r}",
                  file=sys.stderr)
            return 2
        gpu_mod._SOLVER = requested
    print(f"[run_smallbaseline_with_solver] _SOLVER={gpu_mod._SOLVER}", flush=True)

    # Strip our wrapper name so smallbaselineApp.main() sees the same argv
    # it would have via the [project.scripts] shim.
    sys.argv = ['smallbaselineApp.py', *sys.argv[1:]]

    from mintpy.cli import smallbaselineApp
    try:
        smallbaselineApp.main()
    except SystemExit as e:
        return int(getattr(e, 'code', 0) or 0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
