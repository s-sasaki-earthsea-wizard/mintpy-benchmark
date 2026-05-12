#!/usr/bin/env bash
# Direct-Python-call CPU vs GPU bench for correct_topography on a single
# MintPy work directory. Writes per-solver .npy outputs + metrics.json
# into LOG_DIR/{cpu,torch}/ and a CPU-vs-GPU diff into LOG_DIR/diff.json.
#
# Pre-conditions:
#   * MintPy fork venv is the active interpreter (.venv/bin/python).
#   * Fork branch carries the GPU dispatch (perf/correct-topography-torch
#     or later). Without it, --solver=torch will raise.
#   * geometryRadar.h5 with non-degenerate slantRangeDistance / bperp
#     exists in WORK_DIR/inputs/.
#
# Usage:
#   bash run_correct_topography_bench.sh [LOG_DIR]
# Env overrides:
#   WORK_DIR       MintPy work dir (default: REPO_ROOT/FernandinaSenDT128/mintpy)
#   TS_FILE        time-series h5 (default: WORK_DIR/timeseries_ERA5_ramp.h5)
#   GEOM_FILE      geometry h5    (default: WORK_DIR/inputs/geometryRadar.h5)
#   POLY_ORDER     deformation polynomial order (default: 2)
#   STEP_DATES     comma-separated step dates  (default: 20170910,20180613,
#                  matching Fernandina smallbaselineApp.cfg)
#   EXCLUDE_DATES  comma-separated excluded dates (default: empty)
#   CHUNK_SIZE     GPU chunk size in pixels (default: auto)
#   DROP_CACHES    set to 1 to flush page cache between runs via sudo
#                  drop_caches=3 (default: 0; requires NOPASSWD sudo)
set -u

source "$(dirname "$0")/lib/setup_ulimit.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
BENCH_PY="$(cd "$(dirname "$0")" && pwd)/run_correct_topography_bench.py"
DIFF_PY="$(cd "$(dirname "$0")/../tools" && pwd)/compare_dem_error_outputs.py"

WORK_DIR="${WORK_DIR:-${REPO_ROOT}/FernandinaSenDT128/mintpy}"
TS_FILE="${TS_FILE:-${WORK_DIR}/timeseries_ERA5_ramp.h5}"
GEOM_FILE="${GEOM_FILE:-${WORK_DIR}/inputs/geometryRadar.h5}"
POLY_ORDER="${POLY_ORDER:-2}"
# Use `${VAR-default}` (no colon) so an explicitly-empty STEP_DATES="" — meaning
# "no step jumps, e.g. for Galapagos" — does not fall back to the Fernandina
# defaults. The colon form would treat empty as unset.
STEP_DATES="${STEP_DATES-20170910,20180613}"
EXCLUDE_DATES="${EXCLUDE_DATES:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
LOG_DIR="${1:-${REPO_ROOT}/benchmark/logs_correct_topography_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$(realpath -m -- "${LOG_DIR}")"

mkdir -p "${LOG_DIR}/cpu" "${LOG_DIR}/torch"
echo "[bench] log dir: ${LOG_DIR}"
echo "[bench] ts_file:   ${TS_FILE}"
echo "[bench] geom_file: ${GEOM_FILE}"
echo "[bench] poly_order=${POLY_ORDER}  step_dates=${STEP_DATES}  exclude=${EXCLUDE_DATES}"

# Render step / exclude dates as space-separated argv tokens (nargs='*').
STEP_ARGV=()
if [[ -n "${STEP_DATES}" ]]; then
    IFS=',' read -ra STEP_ARGV <<< "${STEP_DATES}"
fi
EX_ARGV=()
if [[ -n "${EXCLUDE_DATES}" ]]; then
    IFS=',' read -ra EX_ARGV <<< "${EXCLUDE_DATES}"
fi
CHUNK_ARGV=()
if [[ -n "${CHUNK_SIZE}" ]]; then
    CHUNK_ARGV=(--chunk-size "${CHUNK_SIZE}")
fi

drop_caches() {
    if [[ "${DROP_CACHES:-0}" == "1" ]]; then
        echo "[bench] flushing page cache (sudo drop_caches=3)" >&2
        sync
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    fi
}

run_one() {
    local solver="$1"
    local out_dir="${LOG_DIR}/${solver}"
    local log_file="${out_dir}/stdout.log"
    echo
    echo "==== solver=${solver}  $(date -Iseconds) ===="
    drop_caches
    /usr/bin/time -v -o "${out_dir}/usrbin_time.txt" \
        "${PYTHON_BIN}" "${BENCH_PY}" \
            --ts-file "${TS_FILE}" \
            --geom-file "${GEOM_FILE}" \
            --solver "${solver}" \
            --log-dir "${out_dir}" \
            --poly-order "${POLY_ORDER}" \
            --step-date "${STEP_ARGV[@]}" \
            --exclude-date "${EX_ARGV[@]}" \
            "${CHUNK_ARGV[@]}" \
            > "${log_file}" 2>&1
    local rc=$?
    echo "  -> exit=${rc}  log=${log_file}"
    if [[ ${rc} -ne 0 ]]; then
        echo "[bench] solver=${solver} failed; see ${log_file}" >&2
        return ${rc}
    fi
    grep -E '^\[bench\]' "${log_file}" || true
}

run_one cpu
run_one torch

echo
echo "==== diff cpu vs torch  $(date -Iseconds) ===="
"${PYTHON_BIN}" "${DIFF_PY}" \
    --cpu-dir "${LOG_DIR}/cpu" \
    --gpu-dir "${LOG_DIR}/torch" \
    --out "${LOG_DIR}/diff.json"
echo
echo "[bench] all artifacts under ${LOG_DIR}"
