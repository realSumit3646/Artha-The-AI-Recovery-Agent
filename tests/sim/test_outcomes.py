"""Tests for attempt resolution and mandate revocation.

Every branch is forced deliberately by recalibrating the world rather than by
hunting for a seed that happens to hit it, so a failure names the condition
that broke rather than "the seed moved".
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from mandate_recovery.calibration import DEFAULT_CALIBRATION, BankTier, CalibrationSet
from mandate_recovery.sim import World
from mandate_recovery.sim.response_codes import (
    BANK_CODE_VOCABULARY,
    GENERIC_CODE,
    MISSING_CODES,
)
from mandate_recovery.sim.outcomes import (
    MIN_INSUFFICIENT_FUNDS_FAILURES_FOR_REVOCATION,
    daily_revocation_probability,
    resolve_attempt,
    revoke_eligible_mandates,
)
from mandate_recovery.types import (
    Attempt,
    AttemptOutcome,
    Mandate,
    MandateStatus,
    Rail,
)

QUIET_HOUR = 3  # outside every calibrated restricted window
BUSY_HOUR = 11  # inside the first calibrated restricted window
HUGE = 10**12


def _recalibrated(**overrides) -> CalibrationSet:
    """DEFAULT_CALIBRATION with parameter values replaced.

    Uses ``model_copy``, so cross-parameter validators do not re-run and a
    branch can be forced with an extreme value.
    """
    updates = {
        name: getattr(DEFAULT_CALIBRATION, name).model_copy(update={"value": value})
        for name, value in overrides.items()
    }
    return DEFAULT_CALIBRATION.model_copy(update=updates)


def _all_tiers(value):
    return {tier: value for tier in BankTier}


def _world(calibration: CalibrationSet | None = None, seed: int = 0) -> World:
    return World(
        calibration if calibration is not None else DEFAULT_CALIBRATION,
        np.random.default_rng(seed),
        n_customers=4,
        n_days=10,
    )


def _mandate(amount_paise: int, status=MandateStatus.ACTIVE) -> Mandate:
    return Mandate(
        id="m1",
        customer_id="c000000",
        amount_paise=amount_paise,
        day_of_month=5,
        created_on_day=0,
        status=status,
    )


def _attempt(hour: int = QUIET_HOUR) -> Attempt:
    return Attempt(
        mandate_id="m1",
        scheduled_at=datetime(2026, 4, 5, hour, 0),
        rail=Rail.UPI_AUTOPAY,
    )


def _healthy() -> CalibrationSet:
    """A world where nothing fails unless the test makes it fail."""
    return _recalibrated(
        bank_availability_by_tier=_all_tiers(1.0),
        per_txn_limit_paise_by_tier=_all_tiers(HUGE),
        restricted_window_rejection_probability=0.0,
    )


# --------------------------------------------------------------------------
# Each branch, triggered by its own condition
# --------------------------------------------------------------------------


def test_revoked_mandate_is_never_presented():
    world = _world(_healthy())
    response = resolve_attempt(
        world,
        _mandate(1000, status=MandateStatus.REVOKED),
        _attempt(),
        np.random.default_rng(0),
    )
    assert response.outcome is AttemptOutcome.MANDATE_REVOKED


def test_restricted_window_rejects_at_the_calibrated_probability():
    calibration = _recalibrated(
        bank_availability_by_tier=_all_tiers(1.0),
        per_txn_limit_paise_by_tier=_all_tiers(HUGE),
        restricted_window_rejection_probability=1.0,
    )
    world = _world(calibration)
    response = resolve_attempt(
        world, _mandate(1000), _attempt(BUSY_HOUR), np.random.default_rng(0)
    )
    assert response.outcome is AttemptOutcome.WINDOW_REJECTED


def test_restricted_window_is_a_probability_not_a_block():
    """With rejection at zero, a peak-hour attempt still goes through."""
    world = _world(_healthy())
    response = resolve_attempt(
        world, _mandate(1000), _attempt(BUSY_HOUR), np.random.default_rng(0)
    )
    assert response.outcome is AttemptOutcome.SUCCESS


def test_unavailable_bank_gives_a_technical_decline():
    calibration = _recalibrated(
        bank_availability_by_tier=_all_tiers(0.0),
        per_txn_limit_paise_by_tier=_all_tiers(HUGE),
        restricted_window_rejection_probability=0.0,
    )
    world = _world(calibration)
    response = resolve_attempt(
        world, _mandate(1000), _attempt(), np.random.default_rng(0)
    )
    assert response.outcome is AttemptOutcome.TECHNICAL_DECLINE


def test_amount_over_the_per_transaction_ceiling_is_limit_exceeded():
    calibration = _recalibrated(
        bank_availability_by_tier=_all_tiers(1.0),
        per_txn_limit_paise_by_tier=_all_tiers(500),
        restricted_window_rejection_probability=0.0,
    )
    world = _world(calibration)
    response = resolve_attempt(
        world, _mandate(501), _attempt(), np.random.default_rng(0)
    )
    assert response.outcome is AttemptOutcome.LIMIT_EXCEEDED


def test_amount_over_the_balance_is_insufficient_funds():
    world = _world(_healthy())
    balance = world.balance_paise(0)
    response = resolve_attempt(
        world, _mandate(balance + 1), _attempt(), np.random.default_rng(0)
    )
    assert response.outcome is AttemptOutcome.INSUFFICIENT_FUNDS


def test_an_affordable_attempt_succeeds():
    world = _world(_healthy())
    balance = world.balance_paise(0)
    response = resolve_attempt(
        world, _mandate(balance), _attempt(), np.random.default_rng(0)
    )
    assert response.outcome is AttemptOutcome.SUCCESS


def test_every_outcome_is_reachable():
    """All six outcomes are produced by the branches above."""
    produced = set()

    world = _world(_healthy())
    produced.add(
        resolve_attempt(
            world,
            _mandate(1000, status=MandateStatus.REVOKED),
            _attempt(),
            np.random.default_rng(0),
        ).outcome
    )
    produced.add(
        resolve_attempt(
            world, _mandate(1000), _attempt(), np.random.default_rng(0)
        ).outcome
    )
    produced.add(
        resolve_attempt(
            world,
            _mandate(world.balance_paise(0) + 1),
            _attempt(),
            np.random.default_rng(0),
        ).outcome
    )

    blocked = _world(
        _recalibrated(
            bank_availability_by_tier=_all_tiers(0.0),
            per_txn_limit_paise_by_tier=_all_tiers(HUGE),
            restricted_window_rejection_probability=0.0,
        )
    )
    produced.add(
        resolve_attempt(
            blocked, _mandate(1000), _attempt(), np.random.default_rng(0)
        ).outcome
    )

    capped = _world(
        _recalibrated(
            bank_availability_by_tier=_all_tiers(1.0),
            per_txn_limit_paise_by_tier=_all_tiers(500),
            restricted_window_rejection_probability=0.0,
        )
    )
    produced.add(
        resolve_attempt(
            capped, _mandate(501), _attempt(), np.random.default_rng(0)
        ).outcome
    )

    peak = _world(
        _recalibrated(
            bank_availability_by_tier=_all_tiers(1.0),
            per_txn_limit_paise_by_tier=_all_tiers(HUGE),
            restricted_window_rejection_probability=1.0,
        )
    )
    produced.add(
        resolve_attempt(
            peak, _mandate(1000), _attempt(BUSY_HOUR), np.random.default_rng(0)
        ).outcome
    )

    assert produced == set(AttemptOutcome)


# --------------------------------------------------------------------------
# Resolution order
# --------------------------------------------------------------------------


def test_limit_is_checked_before_the_balance():
    """An over-limit amount is refused without consulting the account."""
    calibration = _recalibrated(
        bank_availability_by_tier=_all_tiers(1.0),
        per_txn_limit_paise_by_tier=_all_tiers(500),
        restricted_window_rejection_probability=0.0,
    )
    world = _world(calibration)
    world.debit(0, world.balance_paise(0))  # empty the account
    assert world.balance_paise(0) == 0

    response = resolve_attempt(
        world, _mandate(501), _attempt(), np.random.default_rng(0)
    )
    assert response.outcome is AttemptOutcome.LIMIT_EXCEEDED


def test_revocation_is_checked_before_everything_else():
    """A revoked mandate reports revocation even in a broken world."""
    calibration = _recalibrated(
        bank_availability_by_tier=_all_tiers(0.0),
        per_txn_limit_paise_by_tier=_all_tiers(1),
        restricted_window_rejection_probability=1.0,
    )
    world = _world(calibration)
    response = resolve_attempt(
        world,
        _mandate(HUGE, status=MandateStatus.REVOKED),
        _attempt(BUSY_HOUR),
        np.random.default_rng(0),
    )
    assert response.outcome is AttemptOutcome.MANDATE_REVOKED


# --------------------------------------------------------------------------
# Money movement
# --------------------------------------------------------------------------


def test_success_reduces_the_balance_by_exactly_the_amount():
    world = _world(_healthy())
    before = world.balance_paise(0)
    amount = before // 3

    response = resolve_attempt(
        world, _mandate(amount), _attempt(), np.random.default_rng(0)
    )
    assert response.outcome is AttemptOutcome.SUCCESS
    assert world.balance_paise(0) == before - amount


@pytest.mark.parametrize(
    "calibration_kwargs, mandate_amount, hour, status",
    [
        ({"bank_availability_by_tier": _all_tiers(0.0)}, 1000, QUIET_HOUR, MandateStatus.ACTIVE),
        ({"per_txn_limit_paise_by_tier": _all_tiers(1)}, 1000, QUIET_HOUR, MandateStatus.ACTIVE),
        ({"restricted_window_rejection_probability": 1.0}, 1000, BUSY_HOUR, MandateStatus.ACTIVE),
        ({}, 1000, QUIET_HOUR, MandateStatus.REVOKED),
    ],
)
def test_a_failed_attempt_moves_no_money(
    calibration_kwargs, mandate_amount, hour, status
):
    base = {
        "bank_availability_by_tier": _all_tiers(1.0),
        "per_txn_limit_paise_by_tier": _all_tiers(HUGE),
        "restricted_window_rejection_probability": 0.0,
    }
    world = _world(_recalibrated(**{**base, **calibration_kwargs}))
    before = world.balances_paise()

    response = resolve_attempt(
        world, _mandate(mandate_amount, status=status), _attempt(hour),
        np.random.default_rng(0),
    )
    assert response.outcome is not AttemptOutcome.SUCCESS
    assert world.balances_paise() == before


def test_success_never_overdraws():
    """Spending the whole balance lands exactly on zero."""
    world = _world(_healthy())
    balance = world.balance_paise(0)
    resolve_attempt(world, _mandate(balance), _attempt(), np.random.default_rng(0))
    assert world.balance_paise(0) == 0


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_identical_attempt_with_the_same_seed_gives_the_same_outcome():
    """The window branch is genuinely stochastic, so this exercises the rng."""
    calibration = _recalibrated(
        bank_availability_by_tier=_all_tiers(1.0),
        per_txn_limit_paise_by_tier=_all_tiers(HUGE),
        restricted_window_rejection_probability=0.5,
    )

    def run(seed: int) -> list[tuple[str, str]]:
        world = _world(calibration, seed=99)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(40):
            response = resolve_attempt(
                world, _mandate(1000), _attempt(BUSY_HOUR), rng
            )
            out.append((response.outcome.value, response.raw_code))
        return out

    assert run(2026) == run(2026)
    assert run(1) != run(2), "rng is not actually driving the window branch"


def test_a_quiet_hour_success_consumes_no_randomness():
    """No window draw, and success is reported cleanly, so the stream holds."""
    world = _world(_healthy())
    rng = np.random.default_rng(5)
    resolve_attempt(world, _mandate(1000), _attempt(QUIET_HOUR), rng)
    assert rng.random() == np.random.default_rng(5).random()


def test_a_failure_consumes_randomness_for_its_code():
    """Encoding a failure is stochastic, so the generator must move."""
    world = _world(_healthy())
    rng = np.random.default_rng(5)
    resolve_attempt(
        world, _mandate(world.balance_paise(0) + 1), _attempt(QUIET_HOUR), rng
    )
    assert rng.random() != np.random.default_rng(5).random()


def test_resolution_requires_an_explicit_generator():
    world = _world(_healthy())
    for bad in (None, 42, np.random.RandomState(0)):
        with pytest.raises(TypeError):
            resolve_attempt(world, _mandate(1000), _attempt(), bad)


# --------------------------------------------------------------------------
# The response itself
# --------------------------------------------------------------------------


def test_a_success_reports_its_bank_own_clean_code():
    world = _world(_healthy())
    response = resolve_attempt(
        world, _mandate(1000), _attempt(), np.random.default_rng(0)
    )
    tier = world.bank_tier_for(0)
    assert response.outcome is AttemptOutcome.SUCCESS
    assert response.raw_code == BANK_CODE_VOCABULARY[tier][AttemptOutcome.SUCCESS]
    assert response.bank_id == world.bank_id_for(0)
    assert response.timestamp == _attempt().scheduled_at


def test_a_failure_reports_a_code_from_the_messiness_layer():
    """The code is the bank's, the generic bucket, or missing entirely."""
    world = _world(_healthy())
    tier = world.bank_tier_for(0)
    allowed = set(BANK_CODE_VOCABULARY[tier].values()) | {GENERIC_CODE}
    allowed |= set(MISSING_CODES)

    rng = np.random.default_rng(4)
    for _ in range(50):
        response = resolve_attempt(
            world, _mandate(world.balance_paise(0) + 1), _attempt(), rng
        )
        assert response.outcome is AttemptOutcome.INSUFFICIENT_FUNDS
        assert response.raw_code in allowed


def test_unknown_customer_is_rejected_loudly():
    world = _world(_healthy())
    stray = _mandate(1000).model_copy(update={"customer_id": "c999999"})
    with pytest.raises(KeyError):
        resolve_attempt(world, stray, _attempt(), np.random.default_rng(0))


# --------------------------------------------------------------------------
# Revocation
# --------------------------------------------------------------------------


def _mandates(count: int = 3) -> tuple[Mandate, ...]:
    return tuple(
        Mandate(
            id=f"m{index}",
            customer_id=f"c{index:06d}",
            amount_paise=49900,
            day_of_month=5,
            created_on_day=0,
            status=MandateStatus.ACTIVE,
        )
        for index in range(count)
    )


def test_daily_revocation_rate_compounds_to_the_monthly_rate():
    monthly = DEFAULT_CALIBRATION.monthly_mandate_revocation_rate.value
    daily = daily_revocation_probability(DEFAULT_CALIBRATION)
    assert 0.0 < daily < monthly
    assert 1.0 - (1.0 - daily) ** 31 == pytest.approx(monthly)


def test_a_mandate_that_has_not_repeatedly_failed_is_never_revoked():
    """Certain hazard, but too few funds failures to qualify."""
    certain = _recalibrated(monthly_mandate_revocation_rate=1.0)
    mandates = _mandates()
    failures = {
        mandate.id: MIN_INSUFFICIENT_FUNDS_FAILURES_FOR_REVOCATION - 1
        for mandate in mandates
    }
    updated = revoke_eligible_mandates(
        mandates, failures, certain, np.random.default_rng(0)
    )
    assert all(m.status is MandateStatus.ACTIVE for m in updated)


def test_a_repeatedly_failing_mandate_is_revoked_at_a_certain_hazard():
    certain = _recalibrated(monthly_mandate_revocation_rate=1.0)
    mandates = _mandates()
    failures = {
        mandate.id: MIN_INSUFFICIENT_FUNDS_FAILURES_FOR_REVOCATION
        for mandate in mandates
    }
    updated = revoke_eligible_mandates(
        mandates, failures, certain, np.random.default_rng(0)
    )
    assert all(m.status is MandateStatus.REVOKED for m in updated)


def test_no_revocation_at_zero_hazard():
    never = _recalibrated(monthly_mandate_revocation_rate=0.0)
    mandates = _mandates()
    failures = {m.id: 10 for m in mandates}
    updated = revoke_eligible_mandates(
        mandates, failures, never, np.random.default_rng(0)
    )
    assert all(m.status is MandateStatus.ACTIVE for m in updated)


def test_revocation_leaves_already_revoked_mandates_alone():
    certain = _recalibrated(monthly_mandate_revocation_rate=1.0)
    mandates = tuple(
        m.model_copy(update={"status": MandateStatus.COMPLETED})
        for m in _mandates()
    )
    failures = {m.id: 10 for m in mandates}
    updated = revoke_eligible_mandates(
        mandates, failures, certain, np.random.default_rng(0)
    )
    assert all(m.status is MandateStatus.COMPLETED for m in updated)


def test_revocation_is_deterministic_under_a_seed():
    partial = _recalibrated(monthly_mandate_revocation_rate=0.5)
    mandates = _mandates(40)
    failures = {m.id: 5 for m in mandates}

    def run(seed: int) -> list[str]:
        updated = revoke_eligible_mandates(
            mandates, failures, partial, np.random.default_rng(seed)
        )
        return [m.status.value for m in updated]

    assert run(3) == run(3)
    assert run(3) != run(4)


def test_revocation_preserves_order_and_identity():
    certain = _recalibrated(monthly_mandate_revocation_rate=1.0)
    mandates = _mandates(5)
    failures = {m.id: 5 for m in mandates}
    updated = revoke_eligible_mandates(
        mandates, failures, certain, np.random.default_rng(0)
    )
    assert [m.id for m in updated] == [m.id for m in mandates]
    assert len(updated) == len(mandates)


def test_revocation_requires_an_explicit_generator():
    with pytest.raises(TypeError):
        revoke_eligible_mandates(_mandates(), {}, DEFAULT_CALIBRATION, None)
