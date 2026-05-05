# mintpy-benchmark

Benchmark and profiling harness for the GPU-acceleration work tracked in
[`s-sasaki-earthsea-wizard/MintPy`](https://github.com/s-sasaki-earthsea-wizard/MintPy)
(a fork of [`insarlab/MintPy`](https://github.com/insarlab/MintPy)).
This repository is **not** a fork of MintPy — it lives as a sibling to
the fork to keep machine-dependent bench artifacts out of the MintPy
git history.

## What lives here

| Kind | Files |
|---|---|
| Bench harness | [`run_bench.sh`](run_bench.sh), [`run_chunk_sweep.sh`](run_chunk_sweep.sh) |
| Profile harness | [`run_profile_pyspy.sh`](run_profile_pyspy.sh), [`run_profile_torch.sh`](run_profile_torch.sh) + [`profile_torch.py`](profile_torch.py) |
| Trace post-processor | [`parse_trace.py`](parse_trace.py) |
| Solver-output diff | [`compare_solutions.py`](compare_solutions.py) |
| Shared OOM safety net | [`lib/setup_ulimit.sh`](lib/setup_ulimit.sh) |
| GPU template fixture | [`FernandinaSenDT128_torch.txt`](FernandinaSenDT128_torch.txt) |
| Bench-only deps | [`requirements.txt`](requirements.txt) |
| Reports | [`report_*.md`](.) — see below |

Per-run logs live under `logs_<tag>/` and are **untracked** by design
(see [`.gitignore`](.gitignore)). Numerical findings are transcribed
into the corresponding `report_*.md` by hand; raw artifacts stay on
the developer's machine.

## Reports

| File | Subject |
|---|---|
| [`report_baseline.md`](report_baseline.md) | NAS / SSD baselines on the CPU path |
| [`report_torch.md`](report_torch.md) | GPU torch backend on the same dataset |
| [`report_chunk_sweep.md`](report_chunk_sweep.md) | `gpuChunkSize` sweep on the torch backend |
| [`report_profile.md`](report_profile.md) | py-spy + torch.profiler breakdown of `invert_network` |
| [`report_solver_comparison.md`](report_solver_comparison.md) | Cholesky vs lstsq solver comparison |

The MintPy fork's [`docs/gpu.md`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/main/docs/gpu.md)
links to these reports as commit-pinned permalinks; do not move or
rename `report_*.md` files without updating those links.

## Prerequisites

- The **MintPy fork** must be checked out and installed with the
  `[gpu]` extra. The harness scripts resolve `REPO_ROOT` as the
  parent of this directory and call `${REPO_ROOT}/.venv/bin/python`,
  so the layout is expected to be:

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
  storage paths (see [`report_torch.md`](report_torch.md) §2).

- Bench-only Python deps (in addition to MintPy's own env):

  ```bash
  uv pip install -r requirements.txt
  ```

## Running

```bash
# 18-step baseline run on default WORK_DIR
bash run_bench.sh logs_baseline

# Same with the GPU template
TEMPLATE=$PWD/FernandinaSenDT128_torch.txt \
WORK_DIR=$HOME/MintPy_bench/FernandinaSenDT128/mintpy \
    bash run_bench.sh logs_torch

# chunk_size sweep (12 runs × ~270 s ≈ 55 min)
WORK_DIR=$HOME/MintPy_bench/FernandinaSenDT128/mintpy \
    bash run_chunk_sweep.sh

# py-spy + torch.profiler on invert_network only
bash run_profile_pyspy.sh
bash run_profile_torch.sh
```

`/usr/bin/time -v` and `lib/setup_ulimit.sh` (which caps host VA at 80%
of physical RAM) are sourced from each script so OOM events stop the
process cleanly rather than reaching SIGKILL.

## See also

- [Wiki: Performance-Benchmarks](https://github.com/s-sasaki-earthsea-wizard/MintPy/wiki/Performance-Benchmarks)
  — index of bench results with commit-pinned permalinks
- [`docs/gpu.md`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/main/docs/gpu.md)
  in the fork — user-facing documentation of the torch backend
