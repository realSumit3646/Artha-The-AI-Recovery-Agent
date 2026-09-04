"""Tests for the simulator's private world.

The three required properties are determinism under a seed, balances that
never go negative, and salary landing on the configured day. The rest guard
the boundary and the determinism invariant that make those meaningful.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from mandate_recovery.calibration import DEFAULT_CALIBRATION, BankTier
from mandate_recovery.sim import DAYS_IN_MONTH, HOURS_IN_DAY, World
from mandate_recovery.sim import world as world_module

N_CUSTOMERS = 25
N_DAYS = 95  # ~3 pay cycles, so every salary day comes round more than once


def _world(seed: int, n_customers: int = N_CUSTOMERS, n_days: int = N_DAYS) -> World:
    return World(
        DEFAULT_CALIBRATION,
        np.random.default_rng(seed),
        n_customers=n_customers,
        n_days=n_days,
    )


def _run(world: World) -> list[tuple[int, ...]]:
    """Balances for every day of the run, day 0 first."""
    trajectory = [world.balances_paise()]
    while world.current_day + 1 < world.n_days:
        world.advance_day()
        trajectory.append(world.balances_paise())
    return trajectory


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_produces_identical_balance_trajectories():
    """Two worlds seeded alike are the same world."""
    assert _run(_world(20260904)) == _run(_world(20260904))


def test_different_seeds_produce_different_trajectories():
    """Guards the test above against passing because nothing varies."""
    assert _run(_world(1)) != _run(_world(2))


def test_same_seed_produces_identical_populations():
    left, right = _world(7), _world(7)
    for index in range(N_CUSTOMERS):
        assert left.latent_state(index) == right.latent_state(index)
        assert left.bank_id_for(index) == right.bank_id_for(index)


def test_same_seed_produces_identical_bank_availability():
    left, right = _world(7), _world(7)
    for bank_id in (tier.value for tier in BankTier):
        for day in range(0, N_DAYS, 7):
            for hour in range(HOURS_IN_DAY):
                assert left.bank_available(bank_id, day, hour) == (
                    right.bank_available(bank_id, day, hour)
                )


def test_world_requires_an_explicit_generator():
    """Invariant 5: no global random state, ever."""
    for bad in (None, 42, np.random.RandomState(42)):
        with pytest.raises(TypeError):
            World(
                DEFAULT_CALIBRATION, bad, n_customers=3, n_days=5
            )  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Balances
# --------------------------------------------------------------------------


def test_balances_never_go_negative():
    for day_balances in _run(_world(11)):
        assert all(balance >= 0 for balance in day_balances)


def test_balances_never_go_negative_when_spending_outruns_income():
    """The floor holds in the case that would actually break it.

    A customer who spends far more than they earn is driven to zero and kept
    there, rather than going overdrawn.
    """
    starved = DEFAULT_CALIBRATION.model_copy(
        update={
            "monthly_spend_share_of_salary": (
                DEFAULT_CALIBRATION.monthly_spend_share_of_salary.model_copy(
                    update={"value": 20.0}
                )
            )
        }
    )
    world = World(
        starved, np.random.default_rng(3), n_customers=10, n_days=N_DAYS
    )
    seen_zero = False
    while world.current_day + 1 < world.n_days:
        world.advance_day()
        balances = world.balances_paise()
        assert all(balance >= 0 for balance in balances)
        seen_zero = seen_zero or any(balance == 0 for balance in balances)
    assert seen_zero, "spend was not high enough to exercise the floor"


def test_balances_are_integer_paise():
    world = _world(5)
    for balance in world.balances_paise():
        assert isinstance(balance, int) and not isinstance(balance, bool)
    assert isinstance(world.balance_paise(0), int)


# --------------------------------------------------------------------------
# Salary
# --------------------------------------------------------------------------


def test_salary_credits_land_on_the_configured_day():
    """Balance rises only on a customer's own salary day, and does rise."""
    world = _world(13)
    salary_days = [
        world.latent_state(index).salary_day for index in range(world.n_customers)
    ]
    trajectory = _run(world)

    rises_seen = 0
    for index, salary_day in enumerate(salary_days):
        for day in range(1, len(trajectory)):
            if trajectory[day][index] > trajectory[day - 1][index]:
                day_of_month = (day % DAYS_IN_MONTH) + 1
                assert day_of_month == salary_day, (
                    f"customer {index} gained money on day {day} "
                    f"(day-of-month {day_of_month}) but is paid on {salary_day}"
                )
                rises_seen += 1

    assert rises_seen >= world.n_customers, (
        "expected at least one salary credit per customer over the run"
    )


def test_balance_trajectory_matches_the_stated_dynamics():
    """Salary credited at the start of the day, then the day's spend, floored.

    Reconstructs every balance from the latent parameters and compares, which
    pins the arithmetic rather than only its direction.
    """
    world = _world(17)
    latent = [world.latent_state(i) for i in range(world.n_customers)]
    trajectory = _run(world)

    expected = list(trajectory[0])
    for day in range(1, len(trajectory)):
        day_of_month = (day % DAYS_IN_MONTH) + 1
        for index, state in enumerate(latent):
            credit = (
                state.salary_amount_paise
                if day_of_month == state.salary_day
                else 0
            )
            expected[index] = max(
                expected[index] + credit - state.spend_rate_paise_per_day, 0
            )
        assert tuple(expected) == trajectory[day], f"diverged on day {day}"


def test_salary_days_come_from_the_calibrated_distribution():
    world = _world(23, n_customers=200)
    allowed = set(DEFAULT_CALIBRATION.salary_credit_day_distribution.value)
    observed = {
        world.latent_state(index).salary_day for index in range(world.n_customers)
    }
    assert observed <= allowed


# --------------------------------------------------------------------------
# Banks and the restricted window
# --------------------------------------------------------------------------


def test_bank_available_is_stable_across_repeated_calls():
    """A bank is up or down at an hour; asking again does not re-roll it."""
    world = _world(29)
    answers = [world.bank_available("psu", 4, 11) for _ in range(20)]
    assert len(set(answers)) == 1


def test_bank_availability_tracks_the_calibrated_rate():
    world = _world(31, n_customers=2, n_days=300)
    calibrated = DEFAULT_CALIBRATION.bank_availability_by_tier.value
    for tier, rate in calibrated.items():
        up = sum(
            world.bank_available(tier.value, day, hour)
            for day in range(world.n_days)
            for hour in range(HOURS_IN_DAY)
        )
        observed = up / (world.n_days * HOURS_IN_DAY)
        assert abs(observed - rate) < 0.02, f"{tier.value}: {observed} vs {rate}"


def test_bank_available_rejects_unknown_banks_and_out_of_range_slots():
    world = _world(37)
    with pytest.raises(KeyError):
        world.bank_available("not_a_bank", 0, 0)
    with pytest.raises(IndexError):
        world.bank_available("psu", world.n_days, 0)
    with pytest.raises(IndexError):
        world.bank_available("psu", 0, HOURS_IN_DAY)


def test_in_restricted_window_matches_the_calibrated_windows():
    world = _world(41)
    windows = DEFAULT_CALIBRATION.restricted_window_hours.value
    for hour in range(HOURS_IN_DAY):
        expected = any(start <= hour < end for start, end in windows)
        assert world.in_restricted_window(hour) is expected

    # Half-open: the end hour is outside the window.
    for start, end in windows:
        assert world.in_restricted_window(start) is True
        assert world.in_restricted_window(end - 1) is True
        if end < HOURS_IN_DAY:
            assert world.in_restricted_window(end) is any(
                s <= end < e for s, e in windows
            )


# --------------------------------------------------------------------------
# The calendar
# --------------------------------------------------------------------------


def test_day_of_month_cycles_within_the_month():
    world = _world(43, n_days=DAYS_IN_MONTH + 2)
    seen = [world.day_of_month]
    while world.current_day + 1 < world.n_days:
        world.advance_day()
        seen.append(world.day_of_month)
    assert seen[0] == 1
    assert seen[:DAYS_IN_MONTH] == list(range(1, DAYS_IN_MONTH + 1))
    assert seen[DAYS_IN_MONTH] == 1  # wraps


def test_advance_day_stops_at_the_end_of_the_run():
    world = _world(47, n_days=3)
    world.advance_day()
    world.advance_day()
    assert world.current_day == 2
    with pytest.raises(RuntimeError):
        world.advance_day()


def test_world_rejects_an_empty_run():
    for kwargs in ({"n_customers": 0}, {"n_days": 0}):
        with pytest.raises(ValueError):
            World(
                DEFAULT_CALIBRATION,
                np.random.default_rng(0),
                **{"n_customers": 1, "n_days": 1, **kwargs},
            )


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------


def test_world_never_hands_out_an_observation():
    """Invariant 2: this side of the harness does not build policy inputs.

    ``World`` is ground truth. If it grows a method that returns an
    ``Observation``, the boundary has moved into the simulator, where it
    cannot be seen.
    """
    assert not hasattr(world_module, "Observation")

    for name, member in inspect.getmembers(World, inspect.isfunction):
        annotation = inspect.signature(member).return_annotation
        assert "Observation" not in str(annotation), (
            f"World.{name} returns {annotation!r}"
        )


def test_world_docstring_states_it_is_simulator_private():
    docstring = inspect.getdoc(World) or ""
    assert "SIMULATOR-PRIVATE" in docstring
    assert "MUST NEVER BE PASSED TO A POLICY" in docstring
