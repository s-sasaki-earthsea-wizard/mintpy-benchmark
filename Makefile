# Thin wrapper around scripts/run_*.sh. Run `make help` to list targets.
#
# Argument-passing contract (matches the underlying scripts):
#
#   - LOG_DIR    -> first positional argument of each script
#                   (e.g. `make bench LOG_DIR=logs_first_run`)
#   - WORK_DIR   -> env (forwarded; required by chunk-sweep, optional elsewhere)
#   - TEMPLATE   -> env (forwarded; bench-torch sets a sensible default if unset)
#
# Empty / unset values are safe: the scripts use `${VAR:-default}` so passing
# an empty string falls through to the script's own default.

.DEFAULT_GOAL := help
.PHONY: help bench bench-torch bench-galapagos bench-galapagos-torch chunk-sweep profile-pyspy profile-torch

# Forward WORK_DIR / TEMPLATE to recipe shells. LOG_DIR is positional, not env,
# so it's not exported here -- each recipe passes "$(LOG_DIR)" explicitly.
export WORK_DIR
export TEMPLATE

help: ## Show this help
	@awk 'BEGIN { FS = ":.*## "; print "Targets:" } \
	      /^[a-zA-Z_-]+:.*## / { printf "  %-16s %s\n", $$1, $$2 }' \
	     $(MAKEFILE_LIST)

bench: ## CPU baseline, all 18 steps (override TEMPLATE / WORK_DIR / LOG_DIR via env)
	bash scripts/run_bench.sh "$(LOG_DIR)"

bench-torch: ## GPU torch, all 18 steps (defaults TEMPLATE to fixtures/FernandinaSenDT128_torch.txt)
	TEMPLATE="$${TEMPLATE:-$(CURDIR)/fixtures/FernandinaSenDT128_torch.txt}" \
	    bash scripts/run_bench.sh "$(LOG_DIR)"

bench-galapagos: ## CPU galapagos large scene (WORK_DIR required, defaults TEMPLATE to embedded inputs/GalapagosSenDT128.template)
	TEMPLATE="$${TEMPLATE:-$${WORK_DIR}/inputs/GalapagosSenDT128.template}" \
	    bash scripts/run_bench.sh "$(LOG_DIR)"

bench-galapagos-torch: ## GPU torch galapagos large scene (WORK_DIR required, defaults TEMPLATE to fixtures/GalapagosSenDT128_torch.txt)
	TEMPLATE="$${TEMPLATE:-$(CURDIR)/fixtures/GalapagosSenDT128_torch.txt}" \
	    bash scripts/run_bench.sh "$(LOG_DIR)"

chunk-sweep: ## gpuChunkSize sweep on invert_network (WORK_DIR required, ~55 min)
	bash scripts/run_chunk_sweep.sh "$(LOG_DIR)"

profile-pyspy: ## py-spy flamegraph on invert_network
	bash scripts/run_profile_pyspy.sh "$(LOG_DIR)"

profile-torch: ## torch.profiler kernel breakdown on invert_network
	bash scripts/run_profile_torch.sh "$(LOG_DIR)"
