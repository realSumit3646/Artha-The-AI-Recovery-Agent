"""Tests for the perfect-information oracle.

Two jobs here. First, prove the oracle actually finds the best slot — an
oracle that misses recoverable money understates the headroom and flatters
every policy measured against it. Second, prove its exemption is contained:
it may read latent state, but it must still take nothing but an `Observation`
in `decide`, or the policy interface has been quietly widened for everyone.
"""

from __future__ import annotations

import inspect
import typing

import numpy as np
import pytest

from mandate_recovery.calibration import DEFAULT_CALIBRATION, BankTier
from mandate_recovery.policies.oracle import OraclePolicy
from mandate_recovery.sim import World
from mandate_recovery.sim.world import DAYS_IN_MONTH
from mandate_recovery.types import (
    Decision,
    Observation,
    Rail,
    RetrySilent,
    Stop,
)

HUGE = 10**12


def _recalibrated(**overrides):
    updates = {
        name: getattr(DEFAULT_CALIBRATION, name).model_copy(update={"value": value})
        for name, value in overrides.items()
    }
    return DEFAULT_CALIBRATION.model_copy(update=updates)


def _all_tiers(value):
    return {tier: value for tier in BankTier}


def _world(calibration=None, seed: int = 0, n_days: int = 60) -> World:
    return World(
        calibration if calibration is not None else DEFAULT_CALIBRATION,
        np.random.default_rng(seed),
        n_customers=4,
        n_days=n_days,
    )


def _oracle(world: World, **kwargs) -> OraclePolicy:
    mapping = {
        f"m{index}": world.customer_id_for(index)
        for index in range(world.n_customers)
    }
    return OraclePolicy(world, mapping, **kwargs)


def _observation(amount_paise: int, current_day: int = 0, mandate_id: str = "m0"):
    return Observation(
        mandate_id=mandate_id,
        amount_paise=amount_paise,
        due_day=5,
        current_day=current_day,
    )


# --------------------------------------------------------------------------
# The exemption is declared, and contained
# --------------------------------------------------------------------------


def test_the_oracle_declares_that_it_reads_latent_state():
    assert OraclePolicy.reads_latent_state is True


def test_it_is_the_only_policy_with_the_exemption():
    """If a second policy ever claims this, something has gone wrong."""
    from mandate_recovery.policies import POLICY_REGISTRY

    exempt = [
        policy.__name__
        for policy in POLICY_REGISTRY.values()
        if policy.reads_latent_state
    ]
    assert exempt == ["OraclePolicy"], f"unexpected exempt policies: {exempt}"


def test_the_docstring_says_it_is_not_shippable():
    docstring = inspect.getdoc(OraclePolicy) or ""
    assert "UPPER BOUND" in docstring.upper()
    assert "not shippable" in docstring.lower()


def test_decide_still_takes_only_an_observation():
    """The exemption must not widen the policy interface itself.

    The World arrives through the constructor. If it ever appears in `decide`,
    the boundary has been relaxed for every policy, not just this one.
    """
    signature = inspect.signature(OraclePolicy.decide)
    parameters = [p for p in signature.parameters if p != "self"]
    assert parameters == ["observation"]

    hints = typing.get_type_hints(OraclePolicy.decide)
    assert hints["observation"] is Observation
    assert hints["return"] is Decision


def test_the_world_is_injected_explicitly_at_construction():
    """The cheat is visible at the call site, not hidden inside the class."""
    hints = typing.get_type_hints(OraclePolicy.__init__)
    assert hints["world"] is World


# --------------------------------------------------------------------------
# It finds the money
# --------------------------------------------------------------------------


def test_it_schedules_a_retry_when_the_balance_already_covers_the_amount():
    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(1.0)))
    affordable = world.balance_paise(0)
    decision = _oracle(world).decide(_observation(affordable))

    assert isinstance(decision.action, RetrySilent)
    assert decision.action.scheduled_day == world.current_day


def _project_balances(world: World, index: int, horizon: int) -> dict[int, int]:
    """Replay the world's balance arithmetic forward, independently.

    Deliberately a separate implementation from the oracle's, so the two
    agreeing is evidence rather than tautology.
    """
    latent = world.latent_state(index)
    balance = latent.balance_paise
    projected = {world.current_day: balance}
    for day in range(world.current_day + 1, world.current_day + horizon + 1):
        if (day % DAYS_IN_MONTH) + 1 == latent.salary_day:
            balance += latent.salary_amount_paise
        balance = max(0, balance - latent.spend_rate_paise_per_day)
        projected[day] = balance
    return projected


def test_it_waits_for_payday_when_the_balance_is_short_today():
    """The core claim of the whole project, in one test.

    A debit that fails today succeeds after the salary lands, and the oracle
    schedules for the later day rather than giving up.
    """
    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(1.0)))

    # Pick the customer whose balance grows most over the horizon, and an
    # amount that is out of reach today but comfortably covered later.
    best_index, projected = max(
        (
            (index, _project_balances(world, index, DAYS_IN_MONTH))
            for index in range(world.n_customers)
        ),
        key=lambda pair: max(pair[1].values()) - pair[1][world.current_day],
    )
    today = projected[world.current_day]
    peak = max(projected.values())
    assert peak > today, "no customer gains balance over the horizon"
    amount = today + (peak - today) // 2

    decision = _oracle(world).decide(
        _observation(amount, mandate_id=f"m{best_index}")
    )

    assert isinstance(decision.action, RetrySilent), decision.rationale
    assert decision.action.scheduled_day > world.current_day
    assert projected[decision.action.scheduled_day] >= amount


def test_it_never_schedules_inside_the_restricted_window():
    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(1.0)))
    windows = DEFAULT_CALIBRATION.restricted_window_hours.value
    oracle = _oracle(world)

    for index in range(world.n_customers):
        action = oracle.decide(
            _observation(world.balance_paise(index), mandate_id=f"m{index}")
        ).action
        if isinstance(action, RetrySilent):
            assert not any(
                start <= action.scheduled_hour < end for start, end in windows
            )


def test_it_only_schedules_when_the_bank_is_actually_up():
    world = _world(seed=5)
    oracle = _oracle(world)
    for index in range(world.n_customers):
        action = oracle.decide(
            _observation(world.balance_paise(index), mandate_id=f"m{index}")
        ).action
        if isinstance(action, RetrySilent):
            assert world.bank_available(
                world.bank_id_for(index),
                action.scheduled_day,
                action.scheduled_hour,
            )


def test_it_picks_the_earliest_workable_slot_not_merely_a_workable_one():
    """An oracle that settles for a late slot understates the headroom."""
    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(1.0)))
    amount = world.balance_paise(0)
    action = _oracle(world).decide(_observation(amount)).action

    windows = DEFAULT_CALIBRATION.restricted_window_hours.value
    earliest_clear_hour = next(
        hour
        for hour in range(24)
        if not any(start <= hour < end for start, end in windows)
    )
    assert action.scheduled_day == world.current_day
    assert action.scheduled_hour == earliest_clear_hour


# --------------------------------------------------------------------------
# It knows when to give up
# --------------------------------------------------------------------------


def test_it_stops_when_the_amount_exceeds_the_ceiling():
    """Perfect information includes knowing what is simply impossible."""
    world = _world(_recalibrated(per_txn_limit_paise_by_tier=_all_tiers(500)))
    decision = _oracle(world).decide(_observation(501))

    assert isinstance(decision.action, Stop)
    assert "ceiling" in decision.action.reason


def test_it_stops_when_the_balance_never_covers_the_amount():
    world = _world(
        _recalibrated(
            bank_availability_by_tier=_all_tiers(1.0),
            per_txn_limit_paise_by_tier=_all_tiers(HUGE),
        )
    )
    decision = _oracle(world).decide(_observation(HUGE - 1))

    assert isinstance(decision.action, Stop)
    assert "not recoverable" in decision.rationale


def test_it_stops_when_the_bank_is_never_available():
    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(0.0)))
    decision = _oracle(world).decide(_observation(world.balance_paise(0)))
    assert isinstance(decision.action, Stop)


def test_the_search_horizon_is_one_pay_cycle_by_default():
    from mandate_recovery.sim.world import DAYS_IN_MONTH

    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(1.0)))
    signature = inspect.signature(OraclePolicy.__init__)
    assert signature.parameters["max_days_ahead"].default == DAYS_IN_MONTH

    narrow = _oracle(world, max_days_ahead=1)
    latent = world.latent_state(0)
    amount = latent.balance_paise + latent.salary_amount_paise // 2
    # A slot exists next cycle, but not within one day.
    assert isinstance(narrow.decide(_observation(amount)).action, Stop)


def test_an_invalid_horizon_is_rejected():
    world = _world()
    with pytest.raises(ValueError):
        _oracle(world, max_days_ahead=0)


def test_an_unmapped_mandate_fails_loudly():
    world = _world()
    oracle = OraclePolicy(world, {})
    with pytest.raises(KeyError):
        oracle.decide(_observation(1000))


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------


def test_it_is_deterministic():
    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(1.0)))
    oracle = _oracle(world)
    observation = _observation(world.balance_paise(0))
    assert oracle.decide(observation) == oracle.decide(observation)


def test_it_never_contacts_the_customer():
    """It bounds timing, not persuasion. Only retries and stops."""
    world = _world(seed=9)
    oracle = _oracle(world)
    for index in range(world.n_customers):
        action = oracle.decide(
            _observation(world.balance_paise(index), mandate_id=f"m{index}")
        ).action
        assert isinstance(action, (RetrySilent, Stop))


def test_decisions_are_unvalidated_and_explained():
    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(1.0)))
    decision = _oracle(world).decide(_observation(world.balance_paise(0)))
    assert decision.validated is False
    assert decision.rationale.strip()


def test_the_projection_does_not_mutate_the_world():
    """Looking ahead must not move the world it is looking at."""
    world = _world(_recalibrated(bank_availability_by_tier=_all_tiers(1.0)))
    before_day = world.current_day
    before_balances = world.balances_paise()

    _oracle(world).decide(_observation(world.balance_paise(0) * 3))

    assert world.current_day == before_day
    assert world.balances_paise() == before_balances
