"""Tests for the fixed-schedule baseline.

Several of these exist to prove the baseline is *not* handicapped. If the
agent's advantage later turns out to come from the baseline retrying at a
stupid hour or giving up too early, the result is worthless — so the schedule,
the hour and the exhaustion point are all pinned here.
"""

from __future__ import annotations

import pytest

from mandate_recovery.calibration import DEFAULT_CALIBRATION
from mandate_recovery.policies.fixed_schedule import (
    DEFAULT_RETRY_HOUR,
    DEFAULT_RETRY_OFFSETS_DAYS,
    FixedSchedulePolicy,
)
from mandate_recovery.types import (
    Observation,
    ObservedAttempt,
    Rail,
    RetrySilent,
    Stop,
)

FIRST_FAILURE_DAY = 5


def _history(*days: int, code: str = "DECLINED") -> tuple[ObservedAttempt, ...]:
    return tuple(
        ObservedAttempt(day=day, hour=9, rail=Rail.UPI_AUTOPAY, raw_code=code)
        for day in days
    )


def _observation(history=(), current_day: int = FIRST_FAILURE_DAY, **overrides):
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": 880_000,
        "due_day": FIRST_FAILURE_DAY,
        "current_day": current_day,
        "attempt_history": history,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


# --------------------------------------------------------------------------
# The schedule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retries_already_made, expected_offset", [(0, 1), (1, 3), (2, 5)]
)
def test_it_retries_at_the_configured_offsets(
    retries_already_made, expected_offset
):
    history = _history(*([FIRST_FAILURE_DAY] * (retries_already_made + 1)))
    decision = FixedSchedulePolicy().decide(_observation(history))

    assert isinstance(decision.action, RetrySilent)
    assert decision.action.scheduled_day == FIRST_FAILURE_DAY + expected_offset


def test_it_stops_once_the_schedule_is_exhausted():
    """Three retries and done. It does not retry forever."""
    history = _history(*([FIRST_FAILURE_DAY] * 4))  # original + 3 retries
    decision = FixedSchedulePolicy().decide(_observation(history))

    assert isinstance(decision.action, Stop)
    assert "exhausted" in decision.action.reason


def test_it_does_nothing_before_the_first_failure():
    decision = FixedSchedulePolicy().decide(_observation(()))
    assert isinstance(decision.action, Stop)


def test_offsets_are_measured_from_the_first_failure_not_the_last_attempt():
    """T+3 means three days after the original failure, not after retry one."""
    history = _history(FIRST_FAILURE_DAY, FIRST_FAILURE_DAY + 1)
    decision = FixedSchedulePolicy().decide(
        _observation(history, current_day=FIRST_FAILURE_DAY + 1)
    )
    assert decision.action.scheduled_day == FIRST_FAILURE_DAY + 3


def test_it_never_schedules_into_the_past():
    """A late decision takes the earliest slot still available."""
    history = _history(FIRST_FAILURE_DAY)
    late = _observation(history, current_day=FIRST_FAILURE_DAY + 40)
    decision = FixedSchedulePolicy().decide(late)
    assert decision.action.scheduled_day >= late.current_day


# --------------------------------------------------------------------------
# The baseline is not handicapped
# --------------------------------------------------------------------------


def test_the_default_retry_hour_is_outside_the_restricted_window():
    """The single most important fairness check in this file.

    A baseline retrying inside the NPCI window would hand the agent a large
    advantage on a decision no competent merchant gets wrong, and every
    downstream comparison would be inflated.
    """
    windows = DEFAULT_CALIBRATION.restricted_window_hours.value
    assert not any(
        start <= DEFAULT_RETRY_HOUR < end for start, end in windows
    ), (
        f"the baseline retries at {DEFAULT_RETRY_HOUR:02d}:00, inside the "
        "restricted window; that is a strawman, not a baseline"
    )


def test_the_default_schedule_is_the_common_industry_default():
    assert DEFAULT_RETRY_OFFSETS_DAYS == (1, 3, 5)


def test_it_makes_three_attempts_before_giving_up():
    """Not one, not ten. Enough to catch a short cash gap."""
    assert len(DEFAULT_RETRY_OFFSETS_DAYS) == 3


def test_every_retry_uses_the_configured_hour_and_rail():
    policy = FixedSchedulePolicy(retry_hour=7, rail=Rail.NACH)
    for retries in range(3):
        history = _history(*([FIRST_FAILURE_DAY] * (retries + 1)))
        action = policy.decide(_observation(history)).action
        assert action.scheduled_hour == 7
        assert action.rail is Rail.NACH


# --------------------------------------------------------------------------
# It really is dumb
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code", ["DECLINED", "", "NA", "PS-51", "AB3301", "SF_SYSERR"]
)
def test_the_response_code_makes_no_difference(code):
    """No diagnosis. That is the entire point of the baseline."""
    history = _history(FIRST_FAILURE_DAY, code=code)
    decision = FixedSchedulePolicy().decide(_observation(history))
    assert decision.action.scheduled_day == FIRST_FAILURE_DAY + 1


def test_it_never_contacts_the_customer():
    """No channel choice, no nudges — only silent retries."""
    policy = FixedSchedulePolicy()
    for retries in range(5):
        history = _history(*([FIRST_FAILURE_DAY] * (retries + 1)))
        action = policy.decide(_observation(history)).action
        assert isinstance(action, (RetrySilent, Stop))


def test_it_ignores_how_much_money_is_at_stake():
    small = _observation(_history(FIRST_FAILURE_DAY), amount_paise=100)
    large = _observation(_history(FIRST_FAILURE_DAY), amount_paise=50_000_000)
    policy = FixedSchedulePolicy()
    assert policy.decide(small).action == policy.decide(large).action


def test_it_ignores_the_customers_payment_history():
    policy = FixedSchedulePolicy()
    reliable = _observation(
        _history(FIRST_FAILURE_DAY), historical_success_count=24
    )
    hopeless = _observation(
        _history(FIRST_FAILURE_DAY), historical_failure_count=24
    )
    assert policy.decide(reliable).action == policy.decide(hopeless).action


# --------------------------------------------------------------------------
# Configuration and mechanics
# --------------------------------------------------------------------------


def test_offsets_are_configurable():
    policy = FixedSchedulePolicy(retry_offsets_days=(2, 4, 6, 8))
    history = _history(FIRST_FAILURE_DAY)
    assert policy.decide(_observation(history)).action.scheduled_day == (
        FIRST_FAILURE_DAY + 2
    )
    assert policy.retry_offsets_days == (2, 4, 6, 8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retry_offsets_days": (3, 1)},
        {"retry_offsets_days": (-1,)},
        {"retry_hour": 24},
        {"retry_hour": -1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FixedSchedulePolicy(**kwargs)


def test_it_is_deterministic():
    """No randomness anywhere: the same observation gives the same action."""
    policy = FixedSchedulePolicy()
    observation = _observation(_history(FIRST_FAILURE_DAY))
    assert policy.decide(observation) == policy.decide(observation)


def test_decisions_are_unvalidated_and_explained():
    decision = FixedSchedulePolicy().decide(_observation(_history(FIRST_FAILURE_DAY)))
    assert decision.validated is False
    assert decision.source == "rule"
    assert "T+1" in decision.rationale


def test_it_is_bound_by_the_observation_boundary():
    assert FixedSchedulePolicy.reads_latent_state is False
