TEX_DIR := paper
MAIN := main
PDF := $(TEX_DIR)/$(MAIN).pdf

EVAL_DIR := noninterference_eval
EVAL_SRC := $(EVAL_DIR)/src/run_e2e_eval.py
PYTHON := python3.11
PIP := pip3

.PHONY: all clean check-eval-deps install-eval-deps test-e2e e2e-smoke e2e-benchmark e2e-aggregate e2e-tables

all: $(PDF)

$(PDF): $(TEX_DIR)/$(MAIN).tex $(TEX_DIR)/definitions.tex $(TEX_DIR)/proof.tex $(TEX_DIR)/references.bib
	cd $(TEX_DIR) && latexmk -pdf -interaction=nonstopmode $(MAIN).tex

clean:
	cd $(TEX_DIR) && latexmk -C $(MAIN).tex

check-eval-deps:
	@test -f $(EVAL_DIR)/requirements.txt || (echo "Missing $(EVAL_DIR)/requirements.txt" && exit 1)
	@test -f $(EVAL_SRC) || (echo "Missing $(EVAL_SRC)" && exit 1)
	@$(PYTHON) -c "from importlib.util import find_spec; import sys; required=['yaml','pytest']; missing=[name for name in required if find_spec(name) is None]; sys.exit('Missing Python packages: ' + ', '.join(missing) + '. Install with \'make install-eval-deps\'.') if missing else print('Evaluation dependencies present.')"

install-eval-deps:
	$(PIP) install -r $(EVAL_DIR)/requirements.txt

test-e2e: check-eval-deps
	cd $(EVAL_DIR) && $(PYTHON) -m pytest -q tests/test_guarded_runtime.py tests/test_declassification.py tests/test_e2e_benchmark.py tests/test_e2e_metrics.py

e2e-smoke: check-eval-deps
	cd $(EVAL_DIR) && $(PYTHON) src/run_e2e_eval.py smoke --output-dir ./results/e2e

e2e-benchmark: check-eval-deps
	cd $(EVAL_DIR) && $(PYTHON) src/run_e2e_eval.py benchmark --output-dir ./results/e2e

e2e-aggregate: check-eval-deps
	@if [ -z "$(RUN_DIRS)" ]; then echo "Set RUN_DIRS to one or more existing run directories"; exit 1; fi
	cd $(EVAL_DIR) && $(PYTHON) src/run_e2e_eval.py aggregate $(RUN_DIRS) --output-dir ./results/e2e

e2e-tables: check-eval-deps
	@if [ -z "$(RUN_DIRS)" ]; then echo "Set RUN_DIRS to one or more existing run directories"; exit 1; fi
	cd $(EVAL_DIR) && $(PYTHON) src/run_e2e_eval.py tables $(RUN_DIRS) --output-dir ./results/e2e
