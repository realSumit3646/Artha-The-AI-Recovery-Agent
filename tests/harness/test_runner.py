"""Tests for the experiment runner and result storage.

The paired-design tests are the ones that matter. If arms do not face
bit-identical worlds, every comparison downstream is measuring the luck of the
seed draw and calling it a policy difference — and it would still look
perfectly plausible.
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from mandate_recovery.harness import (
    ExperimentConfig,
    build_observation,
    build_world_and_mandates,
    run_experiment,
)
from mandate_recovery.harness.runner import _MandateState, _run_arm
from mandate_recovery.harness.storage import (
    ExperimentExistsError,
    read_experiment,
    write_experiment,
)
from mandate_recovery.policies import (
    DoNothingPolicy,
    FixedSchedulePolicy,
    OraclePolicy,
)
from mandate_recovery.sim.freeze import SIMULATOR_HASH
from mandate_recovery.types import LatentCustomerState, Observation

SEED = 42


def _config(**overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_id": "test",
        "seeds": [SEED],
        "n_customers": 60,
        "n_mandates": 60,
        "n_days": 70,
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _factories() -> dict:
    return {
        "do_nothing": lambda world, mapping: DoNothingPolicy(),
        "fixed_schedule": lambda world, mapping: FixedSchedulePolicy(),
        "oracle": lambda world, mapping: OraclePolicy(world, mapping),
    }


# --------------------------------------------------------------------------
# The paired design
# --------------------------------------------------------------------------


def test_every_arm_starts_from_a_bit_identical_world():
    """Seed 42's world is the same world for every arm, down to each field."""
    config = _config()
    world, _ = build_world_and_mandates(SEED, config)
    left, right = copy.deepcopy(world), copy.deepcopy(world)

    assert left.balances_paise() == right.balances_paise()
    for index in range(world.n_customers):
        assert left.latent_state(index) == right.latent_state(index)
        assert left.bank_id_for(index) == right.bank_id_for(index)

    for bank in {left.bank_id_for(i) for i in range(world.n_customers)}:
        for day in range(0, config.n_days, 11):
            for hour in (0, 7, 11, 18, 23):
                assert left.bank_available(bank, day, hour) == (
                    right.bank_available(bank, day, hour)
                )


def test_two_arms_see_identical_balance_trajectories():
    """The assertion the build plan asks for, stated on seed 42.

    Two non-intervening arms on the same seed must trace the same balances
    day by day. Any divergence means the worlds were not truly paired.
    """
    config = _config()
    base_world, mandates = build_world_and_mandates(SEED, config)

    trajectories = []
    for arm in ("arm_a", "arm_b"):
        world = copy.deepcopy(base_world)
        _run_arm(arm, DoNothingPolicy(), world, mandates, SEED, config)
        trajectories.append(world.balances_paise())

    assert trajectories[0] == trajectories[1]


def test_one_arm_cannot_move_another_arms_world():
    """Isolation: an arm that debits heavily leaves the base world untouched."""
    config = _config()
    base_world, mandates = build_world_and_mandates(SEED, config)
    before = base_world.balances_paise()

    busy = copy.deepcopy(base_world)
    _run_arm("busy", FixedSchedulePolicy(), busy, mandates, SEED, config)

    assert base_world.balances_paise() == before
    assert base_world.current_day == 0
    assert busy.balances_paise() != before, "the busy arm did not run"


def test_the_same_seed_reproduces_the_same_experiment():
    config = _config()
    first, _ = run_experiment(_factories(), config)
    second, _ = run_experiment(_factories(), config)

    assert [e.to_row() for e in first] == [e.to_row() for e in second]


def test_different_seeds_produce_different_results():
    """Guards the determinism test against passing because nothing varies."""
    one, _ = run_experiment(_factories(), _config(seeds=[1]))
    two, _ = run_experiment(_factories(), _config(seeds=[2]))
    assert [e.net_recovery_paise for e in one] != [
        e.net_recovery_paise for e in two
    ]


# --------------------------------------------------------------------------
# What a run produces
# --------------------------------------------------------------------------


def test_every_arm_produces_one_episode_per_mandate():
    config = _config()
    episodes, _ = run_experiment(_factories(), config)

    assert len(episodes) == len(_factories()) * config.n_mandates
    for arm in _factories():
        arm_episodes = [e for e in episodes if e.arm == arm]
        assert len(arm_episodes) == config.n_mandates
        assert len({e.mandate_id for e in arm_episodes}) == config.n_mandates


def test_mandates_recur_so_customers_build_a_history():
    """A 70-day run is two cycles, not one. Without recurrence there is no
    payment history for any policy to reason about."""
    episodes, _ = run_experiment(_factories(), _config())
    assert max(e.cycles for e in episodes) >= 2


def test_decisions_are_recorded_with_a_rationale():
    _, decisions = run_experiment(_factories(), _config())
    assert decisions
    for decision in decisions:
        assert decision.rationale.strip(), decision
        assert decision.action_kind
        assert decision.arm in _factories()


def test_only_failures_produce_decisions():
    """A policy is asked what to do when something went wrong, never before."""
    episodes, decisions = run_experiment(_factories(), _config())
    per_arm_decisions = {}
    for decision in decisions:
        per_arm_decisions[decision.arm] = per_arm_decisions.get(decision.arm, 0) + 1
    for arm, count in per_arm_decisions.items():
        failures = sum(e.failures for e in episodes if e.arm == arm)
        assert count == failures


def test_costs_are_populated_and_internally_consistent():
    episodes, _ = run_experiment(_factories(), _config())
    for episode in episodes:
        assert episode.total_cost_paise == (
            episode.gateway_cost_paise
            + episode.contact_cost_paise
            + episode.churn_cost_paise
        )
        assert episode.net_recovery_paise == (
            episode.recovered_paise - episode.total_cost_paise
        )


def test_all_money_on_an_episode_is_integer_paise():
    episodes, _ = run_experiment(_factories(), _config())
    for episode in episodes[:200]:
        for field in (
            "amount_paise",
            "recovered_paise",
            "gateway_cost_paise",
            "total_cost_paise",
            "net_recovery_paise",
        ):
            value = getattr(episode, field)
            assert isinstance(value, int) and not isinstance(value, bool), field


def test_the_arms_land_in_the_expected_order():
    """Floor below baseline below ceiling. Not a strict requirement of the
    runner, but if this inverts something is wrong with the harness rather
    than with the policies."""
    episodes, _ = run_experiment(_factories(), _config(seeds=[1, 2, 3]))
    net = {}
    for episode in episodes:
        net[episode.arm] = net.get(episode.arm, 0) + episode.net_recovery_paise

    assert net["do_nothing"] < net["fixed_schedule"] < net["oracle"]


# --------------------------------------------------------------------------
# The counterfactual
# --------------------------------------------------------------------------


def test_the_counterfactual_is_computed_and_is_not_all_false():
    episodes, _ = run_experiment(_factories(), _config())
    flags = [e.would_have_paid_without_intervention for e in episodes]
    assert any(flags), "no mandate ever recovers without intervention"
    assert not all(flags), "every mandate recovers without intervention"


def test_the_counterfactual_is_the_same_for_every_arm():
    """It describes the world, not the policy, so arms must agree on it."""
    episodes, _ = run_experiment(_factories(), _config())
    by_arm: dict[str, dict[str, bool]] = {}
    for episode in episodes:
        by_arm.setdefault(episode.arm, {})[episode.mandate_id] = (
            episode.would_have_paid_without_intervention
        )
    reference = by_arm["do_nothing"]
    for arm, flags in by_arm.items():
        assert flags == reference, f"{arm} disagrees about the counterfactual"


def test_a_silent_arm_never_over_intervenes():
    episodes, _ = run_experiment(_factories(), _config())
    for episode in episodes:
        if episode.sms_sent == 0 and not episode.escalated_to_human:
            assert episode.over_intervention is False


# --------------------------------------------------------------------------
# The observation boundary
# --------------------------------------------------------------------------


def test_the_harness_hands_policies_a_clean_observation():
    """The runner is simulator-side; it must still build a lawful view."""
    config = _config()
    _, mandates = build_world_and_mandates(SEED, config)
    observation = build_observation(mandates[0], _MandateState(), day=3)

    assert isinstance(observation, Observation)
    latent_fields = set(LatentCustomerState.model_fields)
    assert set(Observation.model_fields).isdisjoint(latent_fields)


def test_the_observation_carries_only_this_cycles_attempts():
    """`attempt_history` is the current recovery episode, not all of time."""
    config = _config()
    _, mandates = build_world_and_mandates(SEED, config)
    state = _MandateState(successes=4, failures=9, max_success_amount_paise=5000)
    observation = build_observation(mandates[0], state, day=10)

    assert observation.attempt_history == ()
    assert observation.historical_success_count == 4
    assert observation.historical_failure_count == 9
    assert observation.max_historical_success_amount_paise == 5000


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def _frames():
    episodes, decisions = run_experiment(_factories(), _config())
    return (
        pd.DataFrame([e.to_row() for e in episodes]),
        pd.DataFrame([d.__dict__ for d in decisions]),
    )


def test_storage_writes_the_full_experiment_directory(tmp_path):
    episodes, decisions = _frames()
    directory = write_experiment(
        "exp1",
        _config().to_dict(),
        episodes,
        decisions,
        {"net_recovery_paise": 1},
        results_root=tmp_path,
    )

    assert (directory / "config.json").exists()
    assert (directory / "metrics.json").exists()
    assert (directory / "raw" / "episodes.parquet").exists()
    assert (directory / "raw" / "decisions.parquet").exists()


def test_stored_config_records_which_world_produced_it(tmp_path):
    episodes, decisions = _frames()
    directory = write_experiment(
        "exp2", _config().to_dict(), episodes, decisions, {}, results_root=tmp_path
    )
    config = json.loads((directory / "config.json").read_text("utf-8"))

    assert config["simulator_hash"] == SIMULATOR_HASH
    assert "git_sha" in config
    assert "written_at" in config
    assert config["seeds"] == [SEED]
    assert "calibration" in config


def test_storage_refuses_to_overwrite_an_existing_experiment(tmp_path):
    episodes, decisions = _frames()
    write_experiment(
        "exp3", _config().to_dict(), episodes, decisions, {}, results_root=tmp_path
    )
    with pytest.raises(ExperimentExistsError):
        write_experiment(
            "exp3",
            _config().to_dict(),
            episodes,
            decisions,
            {},
            results_root=tmp_path,
        )


def test_overwrite_is_possible_but_must_be_asked_for(tmp_path):
    episodes, decisions = _frames()
    for _ in range(2):
        write_experiment(
            "exp4",
            _config().to_dict(),
            episodes,
            decisions,
            {},
            results_root=tmp_path,
            overwrite=True,
        )


def test_a_stored_experiment_reads_back_intact(tmp_path):
    episodes, decisions = _frames()
    write_experiment(
        "exp5",
        _config().to_dict(),
        episodes,
        decisions,
        {"headline": 42},
        results_root=tmp_path,
    )
    loaded = read_experiment("exp5", results_root=tmp_path)

    assert loaded["metrics"]["headline"] == 42
    assert len(loaded["episodes"]) == len(episodes)
    assert len(loaded["decisions"]) == len(decisions)
    assert loaded["episodes"]["net_recovery_paise"].dtype.kind == "i", (
        "paise came back as something other than an integer"
    )


def test_reading_a_missing_experiment_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_experiment("never_written", results_root=tmp_path)
