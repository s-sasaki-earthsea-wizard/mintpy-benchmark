#!/usr/bin/env bash
# Step-wall bench for ``correct_topography`` via smallbaselineApp.py.
# Companion to run_correct_topography_bench.sh: this one drives the full
# CLI step (update-mode reset + box split + h5 writeback) so the wall
# is directly comparable to the invert_network step-wall numbers in
# report_torch.md / report_large_scene.md.
#
# Pre-conditions:
#   * MintPy fork venv is the active interpreter (.venv/bin/python).
#   * Fork branch carries the GPU dispatch (perf/correct-topography-torch
#     or later). The driver monkeypatches the dispatch site because the
#     CLI doesn't yet expose --solver on dem_error.
#   * geometryRadar.h5 with non-degenerate slantRangeDistance / bperp
#     exists in WORK_DIR/inputs/.
#
# Usage:
#   bash run_correct_topography_step_bench.sh [LOG_DIR]
# Env overrides:
#   WORK_DIR       MintPy work dir (default: REPO_ROOT/FernandinaSenDT128/mintpy)
#   TEMPLATE       MintPy template (default: REPO_ROOT/docs/templates/FernandinaSenDT128.txt)
#   TS_BASENAME    input ts basename without .h5 (default: timeseries_ERA5_ramp)
set -u

source "$(dirname "$0")/lib/setup_ulimit.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
DRIVER_PY="$(cd "$(dirname "$0")" && pwd)/run_correct_topography_step_bench.py"

WORK_DIR="${WORK_DIR:-${REPO_ROOT}/FernandinaSenDT128/mintpy}"
TEMPLATE="${TEMPLATE:-${REPO_ROOT}/docs/templates/FernandinaSenDT128.txt}"
TS_BASENAME="${TS_BASENAME:-timeseries_ERA5_ramp}"
LOG_DIR="${1:-${REPO_ROOT}/benchmark/logs_correct_topo_step_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$(realpath -m -- "${LOG_DIR}")"

mkdir -p "${LOG_DIR}/cpu" "${LOG_DIR}/torch"
echo "[step-bench] log dir:   ${LOG_DIR}"
echo "[step-bench] work_dir:  ${WORK_DIR}"
echo "[step-bench] template:  ${TEMPLATE}"
echo "[step-bench] ts_base:   ${TS_BASENAME}"

run_one() {
    local solver="$1"
    local out_dir="${LOG_DIR}/${solver}"
    local log_file="${out_dir}/stdout.log"
    echo
    echo "==== step solver=${solver}  $(date -Iseconds) ===="
    /usr/bin/time -v -o "${out_dir}/usrbin_time.txt" \
        "${PYTHON_BIN}" "${DRIVER_PY}" \
            --template "${TEMPLATE}" \
            --work-dir "${WORK_DIR}" \
            --ts-basename "${TS_BASENAME}" \
            --solver "${solver}" \
            --log-dir "${out_dir}" \
            > "${log_file}" 2>&1
    local rc=$?
    echo "  -> exit=${rc}  log=${log_file}"
    if [[ ${rc} -ne 0 ]]; then
        echo "[step-bench] solver=${solver} failed; see ${log_file}" >&2
        return ${rc}
    fi
    grep -E '^\[step-bench\]' "${log_file}" || true
}

# Run CPU first to populate page cache, then GPU. Per-run wall reflects
# warm-cache state for both solvers — same convention as
# run_correct_topography_bench.sh; see that script's docstring for the
# DROP_CACHES alternative when page-cache fairness matters.
run_one cpu
run_one torch

echo
echo "[step-bench] all artifacts under ${LOG_DIR}"
