"""Tests for the retry scheduler.

Two required properties: it never returns a slot inside the restricted window,
and it prefers post-salary days when history supports it. The rest guard the
separation between what the agent believes and what the simulator knows.
"""

from __future__ import annotations

import pytest

from mandate_recovery.agent.scheduler import (
    ASSUMED_TIER_AVAILABILITY,
    CODE_PREFIX_TO_TIER,
    DAYS_IN_MONTH,
    HOURS_IN_DAY,
    RetrySlot,
    SchedulerConstraints,
    infer_bank_tier,
    next_retry_slot,
)
from mandate_recovery.types import Observation, ObservedAttempt, Rail


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": 880_000,
        "due_day": 5,
        "current_day": 10,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def _history(*codes: str, day: int = 10) -> tuple[ObservedAttempt, ...]:
    return tuple(
        ObservedAttempt(day=day, hour=9, rail=Rail.UPI_AUTOPAY, raw_code=code)
        for code in codes
    )


# --------------------------------------------------------------------------
# The hard exclusion
# --------------------------------------------------------------------------


def test_it_never_returns_a_slot_inside_the_restricted_window():
    """Required property. The window is a hard exclusion, never a penalty."""
    constraints = SchedulerConstraints()
    for current_day in range(0, 40):
        for successes in ((), (1,), (15,), (30,)):
            slot = next_retry_slot(
                _observation(
                    current_day=current_day, successful_days_of_month=successes
                ),
                constraints,
            )
            assert slot is not None
            assert not constraints.is_restricted(slot.hour), (
                f"scheduled into the restricted window at {slot.hour:02d}:00"
            )


def test_a_wider_window_is_still_excluded_completely():
    """The exclusion follows the constraints, not a hardcoded pair of ranges."""
    constraints = SchedulerConstraints(restricted_windows=((0, 20),))
    slot = next_retry_slot(_observation(), constraints)
    assert slot is not None
    assert slot.hour >= 20


def test_no_slot_exists_when_every_hour_is_excluded():
    constraints = SchedulerConstraints(restricted_windows=((0, 24),))
    assert next_retry_slot(_observation(), constraints) is None


# --------------------------------------------------------------------------
# Salary timing
# --------------------------------------------------------------------------


def _day_of_month(day: int) -> int:
    return (day % DAYS_IN_MONTH) + 1


def test_it_prefers_days_this_customer_has_actually_paid_on():
    """Required property: observed behaviour beats the population prior."""
    # Day 14 is mid-month and weak under the folk prior. Telling the
    # scheduler this customer pays on the 15th should pull it there.
    observed_day = 15
    slot = next_retry_slot(
        _observation(
            current_day=10, successful_days_of_month=(observed_day,)
        )
    )
    assert slot is not None
    assert _day_of_month(slot.day) == observed_day


def test_without_history_it_falls_back_to_the_month_end_prior():
    slot = next_retry_slot(_observation(current_day=10))
    assert slot is not None
    chosen = _day_of_month(slot.day)
    assert chosen >= 25 or chosen <= 7, (
        f"chose the {chosen}th with no history; expected a payday window"
    )


def test_it_schedules_soon_after_payday_not_long_after():
    """Money drains, so a slot just after payday beats one a week later.

    The cooling-off period rules out same-day, so the best available slot is
    the day after payday rather than payday itself.
    """
    slot = next_retry_slot(
        _observation(current_day=0, successful_days_of_month=(1,))
    )
    assert slot is not None
    assert _day_of_month(slot.day) <= 3, (
        f"chose the {_day_of_month(slot.day)}th; expected within days of payday"
    )


def test_history_on_a_weak_day_still_beats_a_strong_day_without_history():
    strong_prior_only = next_retry_slot(_observation(current_day=10))
    with_history = next_retry_slot(
        _observation(current_day=10, successful_days_of_month=(14,))
    )
    assert _day_of_month(with_history.day) == 14
    assert _day_of_month(strong_prior_only.day) != 14


# --------------------------------------------------------------------------
# Bank inference
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code, expected",
    [
        ("AB1200", "large_private"),
        ("AB9001", "large_private"),
        ("PS-51", "psu"),
        ("SF_NOFUNDS", "small_finance"),
    ],
)
def test_the_bank_is_inferred_from_the_code_vocabulary(code, expected):
    assert infer_bank_tier([code]) == expected


@pytest.mark.parametrize("code", ["DECLINED", "", "NA"])
def test_uninformative_codes_identify_nothing(code):
    assert infer_bank_tier([code]) == "unknown"


def test_an_informative_code_is_found_among_uninformative_ones():
    assert infer_bank_tier(["DECLINED", "", "PS-91"]) == "psu"


def test_the_agents_code_book_matches_the_simulators_vocabulary():
    """The agent learned this from data; it must not have drifted from reality.

    This test is the *only* place the agent's beliefs are compared against the
    simulator. The agent code itself imports nothing from `sim`.
    """
    from mandate_recovery.calibration import BankTier
    from mandate_recovery.sim.response_codes import BANK_CODE_VOCABULARY

    for tier in BankTier:
        for code in BANK_CODE_VOCABULARY[tier].values():
            assert infer_bank_tier([code]) == tier.value, (
                f"agent cannot place {code}, which {tier.value} really emits"
            )


def test_the_agent_module_does_not_import_the_simulator():
    """Beliefs must not be ground truth smuggled in through an import."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mandate_recovery"
        / "agent"
        / "scheduler.py"
    ).read_text(encoding="utf-8")
    assert "from ..sim" not in source
    assert "mandate_recovery.sim" not in source
    assert "calibration" not in source.split('"""')[2]  # not in the code body


def test_bank_priors_are_beliefs_not_the_calibrated_truth():
    """If these ever match the calibration exactly, the agent is cheating."""
    from mandate_recovery.calibration import DEFAULT_CALIBRATION

    true_values = {
        tier.value: value
        for tier, value in DEFAULT_CALIBRATION.bank_availability_by_tier.value.items()
    }
    for tier, believed in ASSUMED_TIER_AVAILABILITY.items():
        if tier in true_values:
            assert believed != true_values[tier], (
                f"the agent's prior for {tier} equals the simulator's truth"
            )


# --------------------------------------------------------------------------
# Cooling off
# --------------------------------------------------------------------------


def test_it_respects_the_cooling_off_period():
    history = _history("PS-51", day=10)
    slot = next_retry_slot(
        _observation(current_day=10, attempt_history=history),
        SchedulerConstraints(cooling_off_days=3),
    )
    assert slot is not None
    assert slot.day >= 13


def test_a_longer_cooling_off_pushes_the_slot_later():
    history = _history("PS-51", day=10)
    early = next_retry_slot(
        _observation(current_day=10, attempt_history=history),
        SchedulerConstraints(cooling_off_days=1),
    )
    late = next_retry_slot(
        _observation(current_day=10, attempt_history=history),
        SchedulerConstraints(cooling_off_days=6),
    )
    assert early.day >= 11
    assert late.day >= 16


def test_the_horizon_bounds_the_search():
    slot = next_retry_slot(
        _observation(current_day=0), SchedulerConstraints(horizon_days=3)
    )
    assert slot is not None
    assert 0 <= slot.day <= 3


# --------------------------------------------------------------------------
# Hour choice and rationale
# --------------------------------------------------------------------------


def test_it_prefers_early_hours_to_catch_the_balance():
    slot = next_retry_slot(_observation())
    assert slot is not None
    assert 4 <= slot.hour <= 9


def test_the_rationale_is_written_for_a_human():
    # Day 20 onward covers the 21st to the 4th, so the 30th is reachable.
    slot = next_retry_slot(
        _observation(
            current_day=20,
            successful_days_of_month=(30,),
            attempt_history=_history("PS-51", day=20),
        )
    )
    assert slot is not None
    rationale = slot.rationale

    assert len(rationale) > 80
    assert "psu" in rationale or "PSU" in rationale.upper()
    assert "restricted window" in rationale
    assert "30th" in rationale
    assert rationale.endswith(".")


def test_an_unidentified_bank_says_so_in_the_rationale():
    slot = next_retry_slot(_observation(attempt_history=_history("DECLINED")))
    assert "could not be identified" in slot.rationale


def test_the_result_is_deterministic():
    observation = _observation(successful_days_of_month=(28,))
    assert next_retry_slot(observation) == next_retry_slot(observation)


def test_the_slot_exposes_a_plain_day_hour_tuple():
    slot = next_retry_slot(_observation())
    assert isinstance(slot, RetrySlot)
    assert slot.as_tuple() == (slot.day, slot.hour)
    assert 0 <= slot.hour < HOURS_IN_DAY


def test_every_known_prefix_maps_to_a_tier_with_a_prior():
    for tier in CODE_PREFIX_TO_TIER.values():
        assert tier in ASSUMED_TIER_AVAILABILITY
