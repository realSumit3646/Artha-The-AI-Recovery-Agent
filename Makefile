# Artha — experiment harness.
PYTHON ?= python

.PHONY: install test lint reproduce figures freeze

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

# No linter is a declared dependency yet, and none is to be added without
# being asked. Until one is requested this is a syntax check only.
lint:
	$(PYTHON) -m compileall -q src tests scripts
	@echo "lint: syntax check only - no linter configured"

# Every experiment, from stored configuration. No API key required: the
# experiments that carry results use no model, and the model layer runs from
# the committed cache in llm_cache/.
reproduce:
	$(PYTHON) scripts/reproduce_all.py

# The freeze check on its own, for CI and for a quick sanity check.
freeze:
	$(PYTHON) -m pytest tests/sim/test_freeze.py -q
