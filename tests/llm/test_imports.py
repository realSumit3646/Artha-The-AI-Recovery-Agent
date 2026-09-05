"""Every module must import cleanly on its own, in any order.

A circular import between `llm.diagnosis` and `policies.heuristic` survived
until milestone 5 because the test suite always imported `policies` first.
Anyone importing the llm layer directly hit an ImportError. These tests import
each module in a fresh interpreter, which is the only way to catch it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MODULES = [
    "mandate_recovery.types",
    "mandate_recovery.calibration",
    "mandate_recovery.costs",
    "mandate_recovery.figures",
    "mandate_recovery.sim",
    "mandate_recovery.sim.outcomes",
    "mandate_recovery.sim.response_codes",
    "mandate_recovery.sim.freeze",
    "mandate_recovery.agent",
    "mandate_recovery.agent.scheduler",
    "mandate_recovery.agent.validator",
    "mandate_recovery.agent.audit",
    "mandate_recovery.policies",
    "mandate_recovery.policies.heuristic",
    "mandate_recovery.policies.llm_agent",
    "mandate_recovery.harness",
    "mandate_recovery.llm",
    "mandate_recovery.llm.client",
    "mandate_recovery.llm.cache",
    "mandate_recovery.llm.diagnosis",
    "mandate_recovery.llm.intervention",
    "mandate_recovery.llm.messaging",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_first_in_a_fresh_interpreter(module):
    """Import order must not matter. This is the regression test."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing {module} first fails:\n{result.stderr.strip()[-600:]}"
    )
