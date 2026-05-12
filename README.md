# mintpy-benchmark

Benchmark and profiling harness for the GPU-acceleration work tracked in
[`s-sasaki-earthsea-wizard/MintPy`](https://github.com/s-sasaki-earthsea-wizard/MintPy)
(a fork of [`insarlab/MintPy`](https://github.com/insarlab/MintPy)).
This repository is **not** a fork of MintPy — it lives as a sibling to
the fork to keep machine-dependent bench artifacts out of the MintPy
git history.

## Layout

```
.
├── Makefile               # thin wrapper, run `make help` to list targets
├── scripts/               # bash harness (one entry per target)
│   ├── run_bench.sh
│   ├── run_chunk_sweep.sh
│   ├── run_profile_pyspy.sh
│   ├── run_profile_torch.sh
│   ├── run_correct_topography_bench.sh       # (a) direct-call CPU vs GPU for dem_error
│   ├── run_correct_topography_bench.py       # python harness driven by (a)
│   ├── run_correct_topography_step_bench.sh  # (b) step-wall via smallbaselineApp.py --dostep
│   ├── run_correct_topography_step_bench.py  # python driver driven by (b), monkeypatches solver
│   └── lib/setup_ulimit.sh    # shared OOM safety net (sourced by every harness)
├── tools/                 # Python utilities the harness or analysis depend on
│   ├── compare_solutions.py   # h5 RMS diff between two solver outputs
│   ├── compare_dem_error_outputs.py # .npy RMS diff for correct_topography outputs
│   ├── parse_trace.py         # offline reduction of torch.profiler trace JSON
│   └── profile_torch.py       # torch.profiler driver invoked by run_profile_torch.sh
├── fixtures/
│   └── FernandinaSenDT128_torch.txt  # GPU template
├── reports/               # commit-pinned by external links — do not rename / move
│   ├── report_baseline.md
│   ├── report_torch.md
│   ├── report_chunk_sweep.md
│   ├── report_profile.md
│   ├── report_solver_comparison.md
│   ├── report_large_scene.md
│   └── dem_error/         # per-step subdirectories for newer GPU steps
│       ├── report_fernandina.md
│       └── report_galapagos.md
├── requirements.txt       # bench-only Python deps
└── logs_*/                # untracked, machine-dependent — see .gitignore
```

Per-run logs land under `logs_<tag>/` at the top of this repo and are
**untracked** by design (see [`.gitignore`](.gitignore)). Numerical
findings are transcribed into the corresponding `reports/report_*.md`
by hand; raw artifacts stay on the developer's machine.

## Reports

| File | Subject |
|---|---|
| [`report_baseline.md`](reports/report_baseline.md) | NAS / SSD baselines on the CPU path |
| [`report_torch.md`](reports/report_torch.md) | GPU torch backend on `invert_network` (Fernandina) |
| [`report_chunk_sweep.md`](reports/report_chunk_sweep.md) | `gpuChunkSize` sweep on the torch backend |
| [`report_profile.md`](reports/report_profile.md) | py-spy + torch.profiler breakdown of `invert_network` |
| [`report_solver_comparison.md`](reports/report_solver_comparison.md) | Cholesky vs lstsq solver comparison |
| [`report_large_scene.md`](reports/report_large_scene.md) | GPU torch backend on `invert_network` (Galapagos) |
| [`dem_error/report_fernandina.md`](reports/dem_error/report_fernandina.md) | GPU torch backend on `correct_topography` (Fernandina) |
| [`dem_error/report_galapagos.md`](reports/dem_error/report_galapagos.md) | GPU torch backend on `correct_topography` (Galapagos, 6.15× speedup) |

The MintPy fork's [`docs/gpu.md`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/main/docs/gpu.md)
links to these reports as commit-pinned permalinks; existing SHA-pinned
links keep working across moves, but new links must use the
`reports/report_*.md` paths.

## Prerequisites

- The **MintPy fork** must be checked out and installed with the
  `[gpu]` extra. The harness scripts resolve `REPO_ROOT` as the
  fork root (two levels up from `scripts/`) and call
  `${REPO_ROOT}/.venv/bin/python`, so the layout is expected to be:

  ```
  MintPy/                 # the fork repo
  ├── .venv/              # uv-managed virtualenv with [gpu] extras
  ├── src/mintpy/         # MintPy source
  ├── docs/templates/     # upstream templates (used by run_bench.sh)
  └── benchmark/          # this repo, cloned as a sibling
  ```

- Tutorial dataset (`FernandinaSenDT128/`) extracted under the fork's
  `REPO_ROOT`. The harness reads `WORK_DIR=${WORK_DIR:-${REPO_ROOT}/FernandinaSenDT128/mintpy}`,
  so override `WORK_DIR` to point at an SSD copy when comparing
  storage paths (see [`report_torch.md`](reports/report_torch.md) §2).

- Bench-only Python deps (in addition to MintPy's own env):

  ```bash
  uv pip install -r requirements.txt
  ```

## Running

The Makefile wraps each harness script. List targets with `make help`:

```bash
make help
```

Common invocations:

```bash
# 18-step CPU baseline on the default WORK_DIR (fork's tutorial dataset)
make bench LOG_DIR=logs_baseline

# 18-step GPU torch run (uses fixtures/FernandinaSenDT128_torch.txt by default)
make bench-torch \
    WORK_DIR=$HOME/MintPy_bench/FernandinaSenDT128/mintpy \
    LOG_DIR=logs_torch

# chunk_size sweep on invert_network only (12 runs × ~270 s ≈ 55 min)
make chunk-sweep WORK_DIR=$HOME/MintPy_bench/FernandinaSenDT128/mintpy

# py-spy + torch.profiler on invert_network
make profile-pyspy
make profile-torch

# correct_topography (DEM error) CPU vs GPU — direct-call wall + numeric diff
bash scripts/run_correct_topography_bench.sh logs_correct_topo_fernandina_r2

# correct_topography step wall via smallbaselineApp.py --dostep
bash scripts/run_correct_topography_step_bench.sh logs_correct_topo_step_fernandina_r2
```

Argument-passing contract:

- **`LOG_DIR`** is the first positional argument of each underlying
  script. Override it via `make <target> LOG_DIR=<dir>`. Empty / unset
  falls through to the script's default (`logs_baseline`,
  `logs_chunk_sweep`, or a timestamped `logs_profile_*`).
- **`WORK_DIR` and `TEMPLATE`** are environment variables read by the
  scripts directly. Override either at the make level
  (`make bench WORK_DIR=...`) or via env (`WORK_DIR=... make bench`);
  the Makefile forwards them with `export`.

Direct shell invocation also works (the Makefile is just a wrapper):

```bash
bash scripts/run_bench.sh logs_baseline
WORK_DIR=$HOME/.../mintpy bash scripts/run_chunk_sweep.sh
```

`/usr/bin/time -v` and `scripts/lib/setup_ulimit.sh` (which caps host VA
at 80% of physical RAM) are sourced from each harness so OOM events
stop the process cleanly rather than reaching SIGKILL.

## See also

- [Wiki: Performance-Benchmarks](https://github.com/s-sasaki-earthsea-wizard/MintPy/wiki/Performance-Benchmarks)
  — index of bench results with commit-pinned permalinks
- [`docs/gpu.md`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/main/docs/gpu.md)
  in the fork — user-facing documentation of the torch backend
