#!/usr/bin/env bash
# Chunk-size sweep for the torch GPU backend on `--dostep invert_network`.
#
# Goal: locate the sweet spot of mintpy.networkInversion.gpuChunkSize on this
# RTX 5080 (16 GiB) machine, and quantify how much of the total invert_network
# wall time is driven by GPU lstsq vs. the surrounding host-side overhead
# (chunk launch, host<->device copies, read_stack_obs, calc_weight_sqrt).
#
# Each run executes `smallbaselineApp.py <template> --dostep invert_network`
# in a fresh process; the template is copied per chunk size with one extra
# line appended:
#   mintpy.networkInversion.gpuChunkSize = <N>
# A value of 0 means "auto-size from free VRAM" (the current default).
#
# Output layout:
#   ${LOG_DIR}/template_<cs>.txt        # per-chunk template
#   ${LOG_DIR}/cs<cs>_r<round>.log      # full stdout/stderr
#   ${LOG_DIR}/cs<cs>_r<round>.time     # /usr/bin/time -v rusage
#   ${LOG_DIR}/summary.tsv              # one row per run
#   ${LOG_DIR}/_overview.log            # high-level progress

set -u

# Cap host VA at 80% of physical RAM (see lib/setup_ulimit.sh for why)
source "$(dirname "$0")/lib/setup_ulimit.sh"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${WORK_DIR:?set WORK_DIR to the SSD copy of FernandinaSenDT128/mintpy}"
TEMPLATE_BASE="${TEMPLATE:-${REPO_ROOT}/benchmark/FernandinaSenDT128_torch.txt}"
LOG_DIR="${1:-${REPO_ROOT}/benchmark/logs_chunk_sweep}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
SBA_BIN="${REPO_ROOT}/.venv/bin/smallbaselineApp.py"

# Sweep matrix. 0 == auto (free VRAM driven). 40000 is the practical ceiling
# on RTX 5080 (16 GiB) for this dataset; 80000 OOMs the gels workspace.
CHUNK_SIZES=(1000 5000 10000 0 30000 40000)
ROUNDS=2

mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/summary.tsv"
OVERVIEW="${LOG_DIR}/_overview.log"
printf 'chunk_size\tround\twall_seconds\tinternal_seconds\tnum_chunks\tresolved_chunk_size\tmax_rss_kb\texit_code\n' > "${SUMMARY}"

cd "${WORK_DIR}"

OVERALL_START=$(date +%s)
{
    echo "Chunk sweep started at $(date -Iseconds)"
    echo "Repo root:     ${REPO_ROOT}"
    echo "Work dir:      ${WORK_DIR}"
    echo "Template base: ${TEMPLATE_BASE}"
    echo "Chunk sizes:   ${CHUNK_SIZES[*]}"
    echo "Rounds:        ${ROUNDS}"
} | tee "${OVERVIEW}"

# Generate per-chunk-size templates once. Always include the gpuChunkSize
# line (even for cs=0): MintPy preserves stale values in the work_dir's
# smallbaselineApp.cfg when the template does not override, which would
# silently re-use the previous run's chunk_size instead of triggering auto.
for cs in "${CHUNK_SIZES[@]}"; do
    TEMPLATE="${LOG_DIR}/template_${cs}.txt"
    cp "${TEMPLATE_BASE}" "${TEMPLATE}"
    printf '\n# chunk-sweep override (0 = auto from free VRAM)\nmintpy.networkInversion.gpuChunkSize = %s\n' "${cs}" >> "${TEMPLATE}"
done

# Round-robin order: sweep all chunk sizes in round 1, then in round 2, ...
# This balances any thermal / page-cache drift across the two rounds.
for r in $(seq 1 ${ROUNDS}); do
    for cs in "${CHUNK_SIZES[@]}"; do
        TAG="cs${cs}_r${r}"
        TEMPLATE="${LOG_DIR}/template_${cs}.txt"
        LOG="${LOG_DIR}/${TAG}.log"
        TIMEFILE="${LOG_DIR}/${TAG}.time"
        echo | tee -a "${OVERVIEW}"
        echo "==== chunk_size=${cs} round=${r} @ $(date -Iseconds) ====" | tee -a "${OVERVIEW}"

        # Force re-inversion. smallbaselineApp invokes ifgram_inversion with
        # --update, which checks output mtime + a fixed list of "key config
        # parameters" -- gpuChunkSize is NOT in that list, so changing only
        # chunk_size would short-circuit to skip. Remove outputs to override.
        rm -f "${WORK_DIR}/timeseries.h5" \
              "${WORK_DIR}/temporalCoherence.h5" \
              "${WORK_DIR}/numInvIfgram.h5"

        /usr/bin/time -v -o "${TIMEFILE}" \
            "${PYTHON_BIN}" "${SBA_BIN}" "${TEMPLATE}" --dostep invert_network \
            > "${LOG}" 2>&1
        EC=$?

        WALL=$(awk -F': ' '/Elapsed \(wall clock\)/ {print $2}' "${TIMEFILE}")
        RSS=$(awk -F': ' '/Maximum resident set size/ {print $2}' "${TIMEFILE}")
        SECS=$(echo "${WALL}" | awk -F: '{
            if (NF==3) print $1*3600+$2*60+$3;
            else if (NF==2) print $1*60+$2;
            else print $1;
        }')

        # Internal "Time used:" line printed by ifgram_inversion at end of step.
        # Format examples seen in MintPy:  "Time used: 04 mins 17.4 secs"
        INTERNAL=$(awk '/^Time used:/ {
            t = 0;
            for (i = 1; i <= NF; i++) {
                if ($i == "mins" || $i == "min")   t += $(i-1) * 60;
                else if ($i == "secs" || $i == "sec") t += $(i-1);
                else if ($i == "hrs"  || $i == "hr")  t += $(i-1) * 3600;
            }
            print t;
            exit;
        }' "${LOG}")
        [ -z "${INTERNAL}" ] && INTERNAL="NA"

        # Authoritative resolved chunk_size + num_chunks from the universally
        # printed line (auto path doesn't print its own diagnostic if the
        # caller passed an explicit value, so we read from this one line).
        # Line: "estimating time-series via torch batched WLS in 14 chunk(s) of up to 19403 pixels ..."
        RESOLVED=$(awk '/estimating time-series via torch batched/ {
            num_chunks = ""; resolved = "";
            for (i = 1; i <= NF; i++) {
                if ($i == "in" && num_chunks == "") num_chunks = $(i+1);
                if ($i == "of" && $(i+1) == "up" && $(i+2) == "to") resolved = $(i+3);
            }
            print num_chunks "\t" resolved;
            exit;
        }' "${LOG}")
        NUM_CHUNKS=$(echo "${RESOLVED}" | cut -f1)
        RESOLVED_CS=$(echo "${RESOLVED}" | cut -f2)
        [ -z "${NUM_CHUNKS}" ]  && NUM_CHUNKS="NA"
        [ -z "${RESOLVED_CS}" ] && RESOLVED_CS="${cs}"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${cs}" "${r}" "${SECS}" "${INTERNAL}" "${NUM_CHUNKS}" "${RESOLVED_CS}" "${RSS}" "${EC}" \
            >> "${SUMMARY}"
        echo "  -> wall=${SECS}s  internal=${INTERNAL}s  chunks=${NUM_CHUNKS}  resolved_cs=${RESOLVED_CS}  rss=${RSS}kb  exit=${EC}" \
            | tee -a "${OVERVIEW}"
    done
done

OVERALL_END=$(date +%s)
TOTAL=$((OVERALL_END - OVERALL_START))
{
    echo
    echo "Total wall time: ${TOTAL} s ($(date -d@${TOTAL} -u +%H:%M:%S))"
    echo "Summary TSV: ${SUMMARY}"
} | tee -a "${OVERVIEW}"
