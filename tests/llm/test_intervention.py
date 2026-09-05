"""Tests for intervention selection.

The one that matters: a model proposal that violates a compliance rule never
executes. The rest pin the split — the model picks a kind of action and a
rough time, and nothing it says moves money.
"""

from __future__ import annotations

import pytest

from mandate_recovery.agent.validator import Budget, ComplianceLimits, Validator
from mandate_recovery.llm.client import StubClient
from mandate_recovery.llm.diagnosis import RoutedDiagnosis
from mandate_recovery.llm.intervention import (
    InterventionReply,
    InterventionSelector,
    render_prompt,
)
from mandate_recovery.policies.heuristic import Diagnosis
from mandate_recovery.types import (
    CollectPartial,
    EscalateHuman,
    Observation,
    ObservedAttempt,
    Rail,
    RetrySilent,
    SendNudge,
    Stop,
    SwitchRail,
)

LIMITS = ComplianceLimits()


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": 880_000,
        "due_day": 25,
        "current_day": 10,
        "current_hour": 10,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def _history(count: int = 1, code: str = "DECLINED"):
    return tuple(
        ObservedAttempt(day=10, hour=6, rail=Rail.UPI_AUTOPAY, raw_code=code)
        for _ in range(count)
    )


def _diagnosis(cause=Diagnosis.INSUFFICIENT_FUNDS, confident=True):
    return RoutedDiagnosis(cause, "llm", confident, "because")


def _selector(reply: InterventionReply, **kwargs) -> InterventionSelector:
    return InterventionSelector(
        StubClient({"InterventionReply": reply}), **kwargs
    )


def _reply(**overrides) -> InterventionReply:
    kwargs = {
        "action": "RETRY_SILENT",
        "timing": "AFTER_NEXT_SALARY",
        "tone_level": 1,
        "reasoning": "wait for payday",
    }
    kwargs.update(overrides)
    return InterventionReply(**kwargs)


# --------------------------------------------------------------------------
# The gate holds against the model
# --------------------------------------------------------------------------


def test_a_proposal_that_violates_a_rule_never_executes():
    """The required test. The model insists; the customer is not contacted."""
    selector = _selector(_reply(action="SEND_NUDGE"))
    result = selector.select(
        _observation(current_hour=23, attempt_history=_history()),
        _diagnosis(),
        Budget(),
    )

    assert not isinstance(result.action, SendNudge)
    assert result.validated is False
    assert result.validator_rule == "outside_contact_hours"
    assert selector.refused_by_validator == 1


def test_a_proposal_past_the_attempt_cap_never_executes():
    selector = _selector(_reply(action="RETRY_SILENT"))
    result = selector.select(
        _observation(attempt_history=_history(LIMITS.max_attempts_per_cycle)),
        _diagnosis(),
        Budget(),
    )
    assert isinstance(result.action, Stop)
    assert result.validated is False


def test_a_proposal_on_an_exhausted_budget_never_executes():
    selector = _selector(_reply(action="RETRY_SILENT"))
    result = selector.select(
        _observation(attempt_history=_history()),
        _diagnosis(),
        Budget(spent_paise=LIMITS.max_cost_paise_per_mandate),
    )
    assert isinstance(result.action, Stop)
    assert result.validated is False


def test_a_card_switch_without_a_card_never_executes():
    selector = _selector(_reply(action="SWITCH_RAIL"))
    result = selector.select(
        _observation(attempt_history=_history(), has_card_on_file=False),
        _diagnosis(Diagnosis.LIMIT),
        Budget(),
    )
    assert not isinstance(result.action, SwitchRail)
    assert result.validator_rule == "no_card_on_file"


def test_the_proposed_action_is_kept_alongside_the_executed_one():
    """The audit trail must show what was asked for and what actually ran."""
    selector = _selector(_reply(action="SEND_NUDGE"))
    result = selector.select(
        _observation(current_hour=23, attempt_history=_history()),
        _diagnosis(),
        Budget(),
    )
    assert isinstance(result.proposed_action, SendNudge)
    assert isinstance(result.action, Stop)


# --------------------------------------------------------------------------
# The model does not move money
# --------------------------------------------------------------------------


def test_the_reply_schema_contains_no_amount_day_hour_or_rail():
    """Invariant 6, enforced on the schema itself."""
    fields = set(InterventionReply.model_fields)
    for forbidden in ("amount", "amount_paise", "day", "hour", "rail", "slot"):
        assert forbidden not in fields
    assert fields == {"action", "timing", "tone_level", "reasoning"}


def test_a_partial_amount_comes_from_history_not_from_the_model():
    selector = _selector(_reply(action="COLLECT_PARTIAL"))
    result = selector.select(
        _observation(
            amount_paise=900_000,
            attempt_history=_history(),
            max_historical_success_amount_paise=400_000,
        ),
        _diagnosis(Diagnosis.LIMIT),
        Budget(),
    )
    assert isinstance(result.action, CollectPartial)
    assert result.action.amount_paise == 400_000


def test_a_partial_with_no_settled_history_stops_rather_than_guessing():
    selector = _selector(_reply(action="COLLECT_PARTIAL"))
    result = selector.select(
        _observation(
            attempt_history=_history(), max_historical_success_amount_paise=0
        ),
        _diagnosis(Diagnosis.LIMIT),
        Budget(),
    )
    assert isinstance(result.action, Stop)


def test_the_retry_slot_is_computed_by_the_scheduler_not_the_model():
    selector = _selector(_reply(action="RETRY_SILENT"))
    result = selector.select(
        _observation(attempt_history=_history()), _diagnosis(), Budget()
    )
    assert isinstance(result.action, RetrySilent)
    assert not LIMITS.is_restricted(result.action.scheduled_hour)
    assert result.action.scheduled_day > 10


@pytest.mark.parametrize("timing,ceiling", [("SOON", 14), ("AFTER_NEXT_SALARY", 27)])
def test_the_timing_preference_bounds_the_search_window(timing, ceiling):
    selector = _selector(_reply(action="RETRY_SILENT", timing=timing))
    result = selector.select(
        _observation(attempt_history=_history()), _diagnosis(), Budget()
    )
    assert result.action.scheduled_day <= ceiling


def test_next_cycle_timing_stops_whatever_action_was_named():
    selector = _selector(_reply(action="RETRY_SILENT", timing="NEXT_CYCLE"))
    result = selector.select(
        _observation(attempt_history=_history()), _diagnosis(), Budget()
    )
    assert isinstance(result.action, Stop)


def test_an_out_of_range_tone_is_clamped():
    reply = InterventionReply(
        action="SEND_NUDGE", timing="SOON", tone_level=3, reasoning="firm"
    )
    selector = _selector(reply)
    result = selector.select(
        _observation(attempt_history=_history()), _diagnosis(), Budget()
    )
    assert 1 <= result.action.tone_level <= 3


def test_the_schema_rejects_a_tone_outside_the_scale():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        InterventionReply(
            action="SEND_NUDGE", timing="SOON", tone_level=9, reasoning="x"
        )


def test_the_schema_rejects_an_invented_action():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        InterventionReply(
            action="WIRE_TEN_LAKH", timing="SOON", reasoning="do it"
        )


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------


def test_the_prompt_carries_no_raw_amounts():
    prompt = render_prompt(
        _observation(
            amount_paise=880_000,
            attempt_history=_history(),
            max_historical_success_amount_paise=500_000,
        ),
        _diagnosis(),
        Budget(spent_paise=700),
    )
    assert "880000" not in prompt
    assert "500000" not in prompt
    assert "700" not in prompt


def test_the_prompt_is_canonical_across_scales():
    left = render_prompt(
        _observation(amount_paise=900_000, attempt_history=_history(),
                     max_historical_success_amount_paise=400_000),
        _diagnosis(), Budget())
    right = render_prompt(
        _observation(amount_paise=1_800_000, attempt_history=_history(),
                     max_historical_success_amount_paise=800_000),
        _diagnosis(), Budget())
    assert left == right


def test_the_prompt_states_the_churn_economics():
    """The model must know contact is expensive or it will nudge constantly."""
    prompt = render_prompt(
        _observation(attempt_history=_history()), _diagnosis(), Budget()
    )
    # Collapse the markdown wrapping: the claim spans a line break.
    flat = " ".join(prompt.lower().split())
    assert "churn" in flat
    assert "silent retry costs almost nothing" in flat


def test_the_prompt_comes_from_a_versioned_file():
    from mandate_recovery.llm.intervention import PROMPT_PATH

    assert PROMPT_PATH.exists() and PROMPT_PATH.suffix == ".md"


def test_the_prompt_names_the_diagnosis_it_was_given():
    prompt = render_prompt(
        _observation(attempt_history=_history()),
        _diagnosis(Diagnosis.TECHNICAL),
        Budget(),
    )
    assert "TECHNICAL" in prompt


# --------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------


def test_a_model_failure_returns_none_so_the_caller_falls_back():
    selector = InterventionSelector(StubClient(always_fail=True))
    result = selector.select(
        _observation(attempt_history=_history()), _diagnosis(), Budget()
    )
    assert result is None
    assert selector.stats()["fallbacks"] == 1


def test_stats_track_proposals_and_refusals():
    selector = _selector(_reply(action="SEND_NUDGE"))
    selector.select(
        _observation(current_hour=23, attempt_history=_history()),
        _diagnosis(),
        Budget(),
    )
    assert selector.stats() == {
        "proposals": 1,
        "fallbacks": 0,
        "refused_by_validator": 1,
    }


def test_an_escalation_proposal_is_translated_and_gated():
    selector = _selector(_reply(action="ESCALATE"))
    result = selector.select(
        _observation(current_hour=12, attempt_history=_history()),
        _diagnosis(),
        Budget(),
    )
    assert isinstance(result.action, EscalateHuman)
    assert result.validated is True
