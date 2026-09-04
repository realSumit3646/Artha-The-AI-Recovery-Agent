# Artha — experiment harness.
PYTHON ?= python

.PHONY: install test lint reproduce

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

# No linter is a declared dependency yet, and none is to be added without
# being asked. Until one is requested this is a syntax check only.
lint:
	$(PYTHON) -m compileall -q src tests
	@echo "lint: syntax check only — no linter configured"

reproduce:
	@echo "not yet implemented"
