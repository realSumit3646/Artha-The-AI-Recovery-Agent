"""Tests for the compliance validator.

Each rule is tested on its own, and then the rule that matters most: an action
violating a rule is never executed, however insistent the policy proposing it.
"""

from __future__ import annotations

import pytest

from mandate_recovery.agent.validator import (
    Budget,
    ComplianceLimits,
    ValidationResult,
    Validator,
    validate,
)
from mandate_recovery.types import (
    CollectPartial,
    EscalateHuman,
    Observation,
    ObservedAttempt,
    Rail,
    RetrySilent,
    SendNudge,
    Stop,
)
from mandate_recovery.types import NudgeChannel

LIMITS = ComplianceLimits()


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": 880_000,
        "due_day": 5,
        "current_day": 10,
        "current_hour": 10,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def _history(count: int, day: int = 10, hour: int = 6):
    return tuple(
        ObservedAttempt(
            day=day, hour=hour, rail=Rail.UPI_AUTOPAY, raw_code="PS-51"
        )
        for _ in range(count)
    )


def _retry(day: int = 20, hour: int = 6) -> RetrySilent:
    return RetrySilent(
        scheduled_day=day, scheduled_hour=hour, rail=Rail.UPI_AUTOPAY
    )


def _nudge() -> SendNudge:
    return SendNudge(channel=NudgeChannel.SMS, tone_level=1)


# --------------------------------------------------------------------------
# Rule: attempt cap
# --------------------------------------------------------------------------


def test_the_attempt_cap_stops_further_retries():
    validator = Validator()
    observation = _observation(
        attempt_history=_history(LIMITS.max_attempts_per_cycle)
    )
    result = validator.validate(_retry(), observation)

    assert result.approved is False
    assert result.rule == "attempt_cap_reached"
    assert isinstance(result.action, Stop)


def test_one_attempt_below_the_cap_is_allowed():
    observation = _observation(
        attempt_history=_history(LIMITS.max_attempts_per_cycle - 1)
    )
    assert validate(_retry(day=30), observation).approved is True


# --------------------------------------------------------------------------
# Rule: minimum gap between attempts
# --------------------------------------------------------------------------


def test_a_retry_too_soon_is_pushed_forward_not_refused():
    """A timing error with an obviously right answer is corrected."""
    validator = Validator()
    observation = _observation(attempt_history=_history(1, day=10, hour=6))
    result = validator.validate(_retry(day=10, hour=8), observation)

    assert result.approved is True
    assert result.was_substituted
    elapsed = (result.action.scheduled_day - 10) * 24 + (
        result.action.scheduled_hour - 6
    )
    assert elapsed >= LIMITS.min_hours_between_attempts
    assert validator.substitutions.get("min_gap") == 1


def test_a_retry_with_enough_spacing_is_untouched():
    observation = _observation(attempt_history=_history(1, day=10, hour=6))
    result = validate(_retry(day=12, hour=6), observation)
    assert result.approved is True
    assert not result.was_substituted


def test_the_correction_rolls_over_into_the_next_day():
    observation = _observation(attempt_history=_history(1, day=10, hour=20))
    result = validate(_retry(day=10, hour=22), observation)
    assert result.action.scheduled_day > 10


# --------------------------------------------------------------------------
# Rule: restricted window (defence in depth)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hour", [10, 11, 12, 17, 18, 19, 20])
def test_a_retry_inside_the_restricted_window_is_moved_out(hour):
    """The scheduler already prevents this; this catches a policy that doesn't."""
    validator = Validator()
    result = validator.validate(_retry(day=30, hour=hour), _observation())

    assert result.approved is True
    assert not LIMITS.is_restricted(result.action.scheduled_hour)
    assert validator.substitutions.get("restricted_window") == 1


def test_a_retry_outside_the_window_is_left_alone():
    result = validate(_retry(day=30, hour=6), _observation())
    assert result.approved is True
    assert result.action.scheduled_hour == 6


def test_a_late_evening_window_hour_moves_to_a_clear_hour_the_same_day():
    result = validate(_retry(day=30, hour=20), _observation())
    assert result.action.scheduled_hour == 21


# --------------------------------------------------------------------------
# Rule: contact frequency and hours
# --------------------------------------------------------------------------


def test_the_contact_cap_refuses_further_contact():
    validator = Validator()
    observation = _observation(
        contacts_in_last_7_days=LIMITS.max_contacts_per_7_days
    )
    result = validator.validate(_nudge(), observation)

    assert result.approved is False
    assert result.rule == "contact_cap_reached"
    assert isinstance(result.action, Stop)


def test_a_contact_below_the_cap_is_allowed():
    observation = _observation(
        contacts_in_last_7_days=LIMITS.max_contacts_per_7_days - 1
    )
    assert validate(_nudge(), observation).approved is True


@pytest.mark.parametrize("hour", [21, 22, 23])
def test_contact_after_permitted_hours_is_refused(hour):
    """Too late is a violation: the day is gone, there is nothing to queue."""
    result = validate(_nudge(), _observation(current_hour=hour))
    assert result.approved is False
    assert result.rule == "outside_contact_hours"


@pytest.mark.parametrize("hour", [0, 5, 8])
def test_a_nudge_decided_too_early_is_deferred_not_refused(hour):
    """Too early is a queuing question, not a violation.

    A collector that notices a failure at 05:00 does not wake the customer; it
    sends the message at 09:00. Refusing here killed the entire contact path
    in the first heuristic run — 20,366 nudges rejected, zero contacts made.
    """
    validator = Validator()
    result = validator.validate(_nudge(), _observation(current_hour=hour))

    assert result.approved is True
    assert result.action.send_hour == LIMITS.earliest_contact_hour
    assert result.rule == "contact_deferred"
    assert validator.substitutions["contact_deferred"] == 1


def test_an_escalation_is_never_deferred():
    """A human agent picking up a phone is not a queued message."""
    result = validate(EscalateHuman(reason="repeat failure"), _observation(current_hour=5))
    assert result.approved is False
    assert result.rule == "outside_contact_hours"


def test_an_explicit_send_hour_is_validated_rather_than_the_decision_hour():
    late = SendNudge(channel=NudgeChannel.SMS, tone_level=1, send_hour=22)
    assert validate(late, _observation(current_hour=10)).approved is False

    fine = SendNudge(channel=NudgeChannel.SMS, tone_level=1, send_hour=10)
    assert validate(fine, _observation(current_hour=5)).approved is True


@pytest.mark.parametrize("hour", [9, 12, 15, 20])
def test_contact_inside_permitted_hours_is_allowed(hour):
    assert validate(_nudge(), _observation(current_hour=hour)).approved is True


def test_escalation_obeys_the_same_contact_rules():
    """A human agent calling is still a contact."""
    result = validate(
        EscalateHuman(reason="repeat failure"), _observation(current_hour=23)
    )
    assert result.approved is False


def test_a_silent_retry_is_not_subject_to_contact_hours():
    """Retrying at 03:00 disturbs nobody."""
    result = validate(_retry(day=30, hour=3), _observation(current_hour=3))
    assert result.approved is True


# --------------------------------------------------------------------------
# Rule: cost budget
# --------------------------------------------------------------------------


def test_an_exhausted_budget_substitutes_a_stop():
    validator = Validator()
    budget = Budget(spent_paise=LIMITS.max_cost_paise_per_mandate)
    result = validator.validate(_retry(day=30), _observation(), budget)

    assert result.approved is False
    assert result.rule == "cost_budget_exhausted"
    assert isinstance(result.action, Stop)


def test_a_budget_with_room_permits_the_action():
    budget = Budget(spent_paise=LIMITS.max_cost_paise_per_mandate // 2)
    assert validate(_retry(day=30), _observation(), budget).approved is True
    assert budget.remaining_paise() > 0
    assert budget.is_exhausted() is False


def test_the_budget_blocks_contact_as_well_as_retries():
    budget = Budget(spent_paise=LIMITS.max_cost_paise_per_mandate)
    assert validate(_nudge(), _observation(), budget).approved is False


# --------------------------------------------------------------------------
# Rule: rail switching
# --------------------------------------------------------------------------


def test_switching_to_card_without_a_card_is_refused():
    from mandate_recovery.types import SwitchRail

    result = validate(
        SwitchRail(target_rail=Rail.CARD), _observation(has_card_on_file=False)
    )
    assert result.approved is False
    assert result.rule == "no_card_on_file"


def test_switching_to_card_with_a_card_is_allowed():
    from mandate_recovery.types import SwitchRail

    result = validate(
        SwitchRail(target_rail=Rail.CARD), _observation(has_card_on_file=True)
    )
    assert result.approved is True


def test_switching_to_another_upi_rail_needs_no_card():
    from mandate_recovery.types import SwitchRail

    result = validate(
        SwitchRail(target_rail=Rail.NACH), _observation(has_card_on_file=False)
    )
    assert result.approved is True


# --------------------------------------------------------------------------
# The gate cannot be talked around
# --------------------------------------------------------------------------


def test_a_violating_action_is_never_returned_for_execution():
    """The rule that matters. However insistent the policy, the gate holds.

    A caller that executes ``result.action`` is always compliant, whether the
    verdict was approval, correction, or refusal.
    """
    validator = Validator()
    insistent = [
        (_retry(day=30, hour=11), _observation()),
        (_retry(day=30, hour=18), _observation()),
        (_nudge(), _observation(current_hour=3)),
        (_nudge(), _observation(current_hour=22)),
        (_nudge(), _observation(contacts_in_last_7_days=99)),
        (_retry(day=30), _observation(attempt_history=_history(99))),
        (EscalateHuman(reason="now"), _observation(current_hour=2)),
    ]

    for action, observation in insistent:
        result = validator.validate(action, observation)
        executed = result.action

        if isinstance(executed, RetrySilent):
            assert not LIMITS.is_restricted(executed.scheduled_hour)
        if isinstance(executed, (SendNudge, EscalateHuman)):
            # The hour that matters is when it is *sent*, not when it was
            # decided; a deferred nudge carries its own send hour.
            effective = getattr(executed, "send_hour", None)
            if effective is None:
                effective = observation.current_hour
            assert (
                LIMITS.earliest_contact_hour
                <= effective
                < LIMITS.latest_contact_hour
            )
            assert (
                observation.contacts_in_last_7_days
                < LIMITS.max_contacts_per_7_days
            )


def test_repeating_a_refused_action_never_wears_the_gate_down():
    validator = Validator()
    observation = _observation(current_hour=22)
    for _ in range(50):
        assert validator.validate(_nudge(), observation).approved is False
    assert validator.rejections["outside_contact_hours"] == 50


def test_stopping_is_always_permitted():
    result = validate(
        Stop(reason="giving up"),
        _observation(attempt_history=_history(99), current_hour=3),
        Budget(spent_paise=10**9),
    )
    assert result.approved is True


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------


def test_every_rejection_is_counted_by_reason():
    validator = Validator()
    validator.validate(_nudge(), _observation(current_hour=22))
    validator.validate(_nudge(), _observation(contacts_in_last_7_days=99))
    validator.validate(_retry(), _observation(attempt_history=_history(99)))

    assert validator.rejections == {
        "outside_contact_hours": 1,
        "contact_cap_reached": 1,
        "attempt_cap_reached": 1,
    }
    assert validator.total_rejections == 3


def test_counters_start_empty_and_can_be_reset():
    validator = Validator()
    assert validator.rejections == {}
    validator.validate(_nudge(), _observation(current_hour=22))
    assert validator.total_rejections == 1
    validator.reset()
    assert validator.total_rejections == 0


def test_limits_are_configurable():
    strict = Validator(ComplianceLimits(max_attempts_per_cycle=1))
    observation = _observation(attempt_history=_history(1))
    assert strict.validate(_retry(day=30), observation).approved is False


def test_the_result_reports_which_rule_fired():
    result = validate(_nudge(), _observation(current_hour=22))
    assert isinstance(result, ValidationResult)
    assert result.rule == "outside_contact_hours"
    assert "09:00" in result.reason


def test_a_partial_collection_is_subject_to_the_attempt_cap():
    observation = _observation(attempt_history=_history(99))
    result = validate(CollectPartial(amount_paise=1000), observation)
    assert result.approved is False
