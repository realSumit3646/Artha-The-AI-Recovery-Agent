"""Tests for the routed diagnosis stage.

Three things carry weight here: the model is called only on the residual, the
prompt contains nothing latent, and a low-confidence answer is discarded
rather than believed.
"""

from __future__ import annotations

import pytest

from mandate_recovery.llm.client import StubClient
from mandate_recovery.llm.diagnosis import (
    MINIMUM_CONFIDENCE,
    DiagnosisReply,
    DiagnosisRouter,
    render_prompt,
)
from mandate_recovery.policies.heuristic import Diagnosis
from mandate_recovery.types import Observation, ObservedAttempt, Rail

CONFIDENT = DiagnosisReply(
    cause="LIMIT", confidence=0.9, reasoning="Amount exceeds every past payment."
)
UNSURE = DiagnosisReply(
    cause="LIMIT", confidence=0.2, reasoning="Not enough to go on."
)


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": 880_000,
        "due_day": 5,
        "current_day": 10,
        "current_hour": 6,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def _history(*codes: str):
    return tuple(
        ObservedAttempt(day=10, hour=6, rail=Rail.UPI_AUTOPAY, raw_code=code)
        for code in codes
    )


def _router(reply=CONFIDENT, **kwargs) -> DiagnosisRouter:
    return DiagnosisRouter(StubClient({"DiagnosisReply": reply}, **kwargs))


# --------------------------------------------------------------------------
# Routing: the point of the commit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["PS-91", "AB3301", "SF_PEAK", "PS-14"])
def test_a_code_the_rules_resolve_never_reaches_the_model(code):
    router = _router()
    result = router.diagnose(_observation(attempt_history=_history(code)))

    assert result.source == "rule"
    assert router.llm_invoked == 0
    assert router.stats()["llm_invocation_rate"] == 0.0


@pytest.mark.parametrize("code", ["DECLINED", "", "NA"])
def test_an_uninformative_code_is_routed_to_the_model(code):
    router = _router()
    result = router.diagnose(_observation(attempt_history=_history(code)))

    assert result.source == "llm"
    assert router.llm_invoked == 1


def test_the_contradiction_case_is_routed_to_the_model():
    """A funds code on an amount larger than anything they have paid."""
    router = _router()
    result = router.diagnose(
        _observation(
            amount_paise=900_000,
            attempt_history=_history("PS-51"),
            max_historical_success_amount_paise=500_000,
        )
    )
    assert result.source == "llm"
    assert result.diagnosis is Diagnosis.LIMIT


def test_a_funds_code_the_history_clears_never_reaches_the_model():
    router = _router()
    result = router.diagnose(
        _observation(
            amount_paise=500_000,
            attempt_history=_history("PS-51"),
            max_historical_success_amount_paise=900_000,
        )
    )
    assert result.source == "rule"
    assert router.llm_invoked == 0


def test_the_invocation_rate_is_measured_and_exposed():
    """The evidence that the model was used where judgment was needed."""
    router = _router()
    for code in ("PS-91", "AB3301", "DECLINED", "NA"):
        router.diagnose(_observation(attempt_history=_history(code)))

    stats = router.stats()
    assert stats["diagnoses"] == 4
    assert stats["rule_resolved"] == 2
    assert stats["llm_invoked"] == 2
    assert stats["llm_invocation_rate"] == pytest.approx(0.5)


def test_the_invocation_rate_is_zero_before_anything_is_diagnosed():
    assert _router().stats()["llm_invocation_rate"] == 0.0


# --------------------------------------------------------------------------
# The prompt carries nothing latent
# --------------------------------------------------------------------------


def test_the_rendered_prompt_contains_no_latent_values():
    """The asymmetry invariant, enforced against a real simulated customer.

    Checks for latent *values*, not latent words. The prompt deliberately
    tells the model it does not have the balance or the salary date — that is
    good prompt design, not a leak. What must never appear is a number only
    the simulator knows.
    """
    import numpy as np

    from mandate_recovery.calibration import DEFAULT_CALIBRATION
    from mandate_recovery.harness import ExperimentConfig, build_world_and_mandates
    from mandate_recovery.harness.runner import _MandateState, build_observation

    config = ExperimentConfig(experiment_id="prompt-leak", seeds=[7], n_customers=25, n_mandates=25)
    world, mandates = build_world_and_mandates(7, config)

    leaked = []
    for index, mandate in enumerate(mandates[:25]):
        latent = world.latent_state(index)
        state = _MandateState()
        state.cycle_history = list(_history("DECLINED"))
        state.max_success_amount_paise = latent.salary_amount_paise // 3
        prompt = render_prompt(build_observation(mandate, state, day=10, hour=6))

        # Only the large latent numbers are checked: a salary *day* is 1-31
        # and collides with ordinary prose, but a balance in paise is a
        # seven-digit fingerprint that could only have come from the world.
        for name in (
            "balance_paise",
            "salary_amount_paise",
            "spend_rate_paise_per_day",
            "per_txn_limit_paise",
        ):
            value = getattr(latent, name)
            if value > 9999 and str(value) in prompt:
                leaked.append((mandate.id, name, value))

    assert not leaked, f"latent values reached the prompt: {leaked}"


def test_the_prompt_does_not_leak_raw_amounts():
    """Raw numbers would make the cache useless and the experiment impossible."""
    observation = _observation(
        amount_paise=880_000, max_historical_success_amount_paise=500_000
    )
    prompt = render_prompt(observation)
    assert "880000" not in prompt
    assert "500000" not in prompt


def test_the_prompt_is_canonical_across_different_raw_numbers():
    """Two observations that differ only in scale share one cache entry."""
    left = render_prompt(
        _observation(
            amount_paise=900_000,
            attempt_history=_history("DECLINED"),
            max_historical_success_amount_paise=400_000,
        )
    )
    right = render_prompt(
        _observation(
            amount_paise=1_800_000,
            attempt_history=_history("DECLINED"),
            max_historical_success_amount_paise=800_000,
        )
    )
    assert left == right


def test_the_prompt_still_separates_the_cases_that_matter():
    """Bucketing must not collapse the distinction the diagnosis turns on."""
    above = render_prompt(
        _observation(
            amount_paise=900_000,
            attempt_history=_history("DECLINED"),
            max_historical_success_amount_paise=400_000,
        )
    )
    below = render_prompt(
        _observation(
            amount_paise=400_000,
            attempt_history=_history("DECLINED"),
            max_historical_success_amount_paise=900_000,
        )
    )
    assert above != below


def test_the_prompt_comes_from_a_versioned_file():
    """Prompt changes must show up in a diff, not hide inside Python."""
    from mandate_recovery.llm.diagnosis import PROMPT_PATH

    assert PROMPT_PATH.exists()
    assert PROMPT_PATH.suffix == ".md"
    assert "payments recovery analyst" in render_prompt(_observation())


def test_the_prompt_names_every_cause_it_may_return():
    prompt = render_prompt(_observation(attempt_history=_history("DECLINED")))
    for cause in ("INSUFFICIENT_FUNDS", "TECHNICAL", "LIMIT", "WINDOW", "UNKNOWN"):
        assert cause in prompt


# --------------------------------------------------------------------------
# Confidence and failure
# --------------------------------------------------------------------------


def test_a_low_confidence_answer_is_discarded():
    """An honest UNKNOWN beats a confident guess."""
    router = _router(UNSURE)
    result = router.diagnose(_observation(attempt_history=_history("DECLINED")))

    assert result.diagnosis is Diagnosis.UNKNOWN
    assert result.confident is False
    assert router.llm_low_confidence == 1
    assert router.llm_resolved == 0


def test_an_answer_just_above_the_threshold_is_kept():
    reply = DiagnosisReply(
        cause="TECHNICAL", confidence=MINIMUM_CONFIDENCE + 0.01, reasoning="ok"
    )
    router = _router(reply)
    result = router.diagnose(_observation(attempt_history=_history("DECLINED")))
    assert result.diagnosis is Diagnosis.TECHNICAL


def test_a_discarded_answer_is_still_explained_in_the_rationale():
    router = _router(UNSURE)
    result = router.diagnose(_observation(attempt_history=_history("DECLINED")))
    assert "below the" in result.rationale
    assert "Not enough to go on" in result.rationale


def test_a_model_failure_falls_back_rather_than_crashing():
    router = DiagnosisRouter(StubClient(always_fail=True))
    result = router.diagnose(_observation(attempt_history=_history("DECLINED")))

    assert result.diagnosis is Diagnosis.UNKNOWN
    assert result.source == "fallback"
    assert router.llm_fallbacks == 1


def test_the_rationale_records_which_stage_answered():
    router = _router()
    ruled = router.diagnose(_observation(attempt_history=_history("PS-91")))
    modelled = router.diagnose(_observation(attempt_history=_history("DECLINED")))

    assert "PS-91" in ruled.rationale
    assert "The model read it as" in modelled.rationale


def test_counters_reset_between_arms():
    router = _router()
    router.diagnose(_observation(attempt_history=_history("DECLINED")))
    router.reset()
    assert router.stats()["diagnoses"] == 0
