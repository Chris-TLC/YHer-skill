PYTHON ?= python3
PAPER_RESULTS_MANIFEST ?= data/sim_store/confirmatory/confirmatory-v1/manifest.json
PAPER_RESULTS_DIR ?= /tmp/yher_sprint2/paper_results
PAPER_RESULTS_CONTRACT ?= docs/paper/results_contract.md
PAPER_H5_RAW_ROOT ?= data/sim_store/llm_personas/llm-personas-v1
PAPER_H5_DIR ?= /tmp/yher_sprint2/h5_results
PAPER_H5_COLLECTION ?= $(PAPER_H5_DIR)/h5_collection_manifest.json
PAPER_H5_RESULTS ?= $(PAPER_H5_DIR)/h5_results.json
PAPER_MAIN ?= docs/paper/main.md
PAPER_YAU ?= docs/paper/yau_award_4page.md
PAPER_FIGURE_OUTPUT ?= docs/paper/generated

.PHONY: paper-results figures paper-h5-lock paper-h5-finalize paper-h5-analyze paper-h5-merge paper-h5-merge-existing paper-bind paper-check paper-all paper-final

paper-results:
	PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=. $(PYTHON) -m analysis --manifest "$(PAPER_RESULTS_MANIFEST)" --output "$(PAPER_RESULTS_DIR)" --results-contract "$(PAPER_RESULTS_CONTRACT)"

figures: paper-all

paper-h5-lock:
	PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=. $(PYTHON) -m analysis.h5 lock --raw-root "$(PAPER_H5_RAW_ROOT)" --repo-root .

paper-h5-finalize:
	mkdir -p "$(PAPER_H5_DIR)"
	PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=. $(PYTHON) -m analysis.h5 finalize --raw-root "$(PAPER_H5_RAW_ROOT)" --output "$(PAPER_H5_COLLECTION)" --repo-root .

paper-h5-analyze: paper-h5-finalize
	mkdir -p "$(PAPER_H5_DIR)"
	PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=. $(PYTHON) -m analysis.h5 analyze --collection "$(PAPER_H5_COLLECTION)" --raw-root "$(PAPER_H5_RAW_ROOT)" --output-dir "$(PAPER_H5_DIR)" --repo-root .

paper-h5-merge: paper-h5-analyze
	$(MAKE) --no-print-directory paper-h5-merge-existing

paper-h5-merge-existing:
	PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=. $(PYTHON) -m analysis.h5 merge --contract "$(PAPER_RESULTS_CONTRACT)" --h5-results "$(PAPER_H5_RESULTS)" --artifact-root "$(PAPER_RESULTS_DIR)"

paper-bind:
	PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=. $(PYTHON) -m analysis.paper --contract "$(PAPER_RESULTS_CONTRACT)" --artifact-root "$(PAPER_RESULTS_DIR)" --main "$(PAPER_MAIN)" --yau "$(PAPER_YAU)" --figure-output-dir "$(PAPER_FIGURE_OUTPUT)"

paper-check:
	PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=. $(PYTHON) -m analysis.paper --contract "$(PAPER_RESULTS_CONTRACT)" --artifact-root "$(PAPER_RESULTS_DIR)" --main "$(PAPER_MAIN)" --yau "$(PAPER_YAU)" --figure-output-dir "$(PAPER_FIGURE_OUTPUT)" --check

paper-all:
	$(MAKE) --no-print-directory paper-results
	$(MAKE) --no-print-directory paper-h5-merge
	$(MAKE) --no-print-directory paper-bind
	$(MAKE) --no-print-directory paper-check

paper-final:
	$(MAKE) --no-print-directory paper-h5-merge-existing
	$(MAKE) --no-print-directory paper-bind
	$(MAKE) --no-print-directory paper-check
