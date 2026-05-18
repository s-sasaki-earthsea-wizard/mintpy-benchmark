#!/usr/bin/env bash
# Benchmark FernandinaSenDT128 step-by-step using --dostep.
# Logs per-step wall time + max RSS via /usr/bin/time -v, and aggregates a TSV summary.
set -u

# Cap host VA at 80% of physical RAM (see lib/setup_ulimit.sh for why)
source "$(dirname "$0")/lib/setup_ulimit.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# WORK_DIR / TEMPLATE may be overridden via env var to run on a copy of the
# tutorial dataset placed elsewhere (e.g. SSD vs NAS, to isolate I/O cost).
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/FernandinaSenDT128/mintpy}"
TEMPLATE="${TEMPLATE:-${REPO_ROOT}/docs/templates/FernandinaSenDT128.txt}"
LOG_DIR="${1:-${REPO_ROOT}/benchmark/logs_baseline}"
# Resolve LOG_DIR to an absolute path so log writes still work after we cd
# into WORK_DIR below. -m allows the path not to exist yet.
LOG_DIR="$(realpath -m -- "${LOG_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SBA_BIN="${SBA_BIN:-${REPO_ROOT}/.venv/bin/smallbaselineApp.py}"

mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/summary.tsv"
printf 'step\twall_seconds\tmax_rss_kb\texit_code\n' > "${SUMMARY}"

# Default: full 18-step pipeline. Override with BENCH_STEPS env var
# (whitespace-separated) to run a subset, e.g.
#   BENCH_STEPS="invert_network" bash scripts/run_bench.sh ...
DEFAULT_STEPS=(load_data modify_network reference_point quick_overview correct_unwrap_error \
       invert_network correct_LOD correct_SET correct_ionosphere correct_troposphere \
       deramp correct_topography residual_RMS reference_date velocity geocode \
       google_earth hdfeos5)
if [[ -n "${BENCH_STEPS:-}" ]]; then
    read -ra STEPS <<< "${BENCH_STEPS}"
else
    STEPS=("${DEFAULT_STEPS[@]}")
fi

cd "${WORK_DIR}"

OVERALL_START=$(date +%s)
echo "Benchmark started at $(date -Iseconds)" | tee "${LOG_DIR}/_overview.log"
echo "Work dir: ${WORK_DIR}"            | tee -a "${LOG_DIR}/_overview.log"
echo "Template: ${TEMPLATE}"            | tee -a "${LOG_DIR}/_overview.log"

for step in "${STEPS[@]}"; do
    LOG="${LOG_DIR}/${step}.log"
    TIMEFILE="${LOG_DIR}/${step}.time"
    echo ; echo "==== ${step} @ $(date -Iseconds) ====" | tee -a "${LOG_DIR}/_overview.log"

    /usr/bin/time -v -o "${TIMEFILE}" \
        "${PYTHON_BIN}" "${SBA_BIN}" "${TEMPLATE}" --dostep "${step}" \
        > "${LOG}" 2>&1
    EC=$?

    WALL=$(awk -F': ' '/Elapsed \(wall clock\)/ {print $2}' "${TIMEFILE}")
    RSS=$(awk -F': ' '/Maximum resident set size/ {print $2}' "${TIMEFILE}")
    # Convert h:mm:ss(.ss) or m:ss(.ss) to seconds
    SECS=$(echo "$WALL" | awk -F: '{
        if (NF==3) print $1*3600+$2*60+$3;
        else if (NF==2) print $1*60+$2;
        else print $1;
    }')
    printf '%s\t%s\t%s\t%s\n' "${step}" "${SECS}" "${RSS}" "${EC}" >> "${SUMMARY}"
    echo "  -> wall=${SECS}s  rss=${RSS}kb  exit=${EC}" | tee -a "${LOG_DIR}/_overview.log"
done

OVERALL_END=$(date +%s)
TOTAL=$((OVERALL_END - OVERALL_START))
echo ; echo "Total wall time: ${TOTAL} s ($(date -d@${TOTAL} -u +%H:%M:%S))" | tee -a "${LOG_DIR}/_overview.log"
echo "Summary TSV: ${SUMMARY}"           | tee -a "${LOG_DIR}/_overview.log"
