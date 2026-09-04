"""Tests for metrics and the paired arm comparison.

The comparison tests carry the weight. A bug here would not crash anything —
it would quietly produce a confidence interval that is too narrow, or a loss
rate that reads zero, and the result would look better than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mandate_recovery.harness.metrics import (
    compare_arms,
    compute_metrics,
    compute_metrics_by_arm,
    summarise_comparison,
)


def _episode(**overrides) -> dict:
    row = {
        "seed": 0,
        "arm": "test",
        "mandate_id": "m0",
        "customer_id": "c000000",
        "amount_paise": 880_000,
        "cycles": 3,
        "attempts": 3,
        "decisions": 0,
        "successes": 3,
        "failures": 0,
        "sms_sent": 0,
        "voice_calls_made": 0,
        "escalated_to_human": False,
        "recovered_paise": 2_640_000,
        "days_to_recovery": None,
        "gateway_cost_paise": 600,
        "contact_cost_paise": 0,
        "churn_cost_paise": 0,
        "total_cost_paise": 600,
        "net_recovery_paise": 2_639_400,
        "over_intervention": False,
        "would_have_paid_without_intervention": False,
    }
    row.update(overrides)
    return row


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame([_episode(**row) for row in rows])


# --------------------------------------------------------------------------
# Summary metrics
# --------------------------------------------------------------------------


def test_recovery_rate_is_per_cycle_not_per_mandate():
    """A mandate collected twice out of three cycles recovered two thirds."""
    metrics = compute_metrics(_frame([{"cycles": 3, "successes": 2}]))
    assert metrics["recovery_rate"] == pytest.approx(2 / 3)


def test_headline_totals_are_summed_across_episodes():
    metrics = compute_metrics(
        _frame(
            [
                {"mandate_id": "a", "recovered_paise": 100, "net_recovery_paise": 60},
                {"mandate_id": "b", "recovered_paise": 200, "net_recovery_paise": 90},
            ]
        )
    )
    assert metrics["recovered_paise"] == 300
    assert metrics["net_recovery_paise"] == 150
    assert metrics["n_episodes"] == 2


def test_attempts_and_contacts_per_recovery():
    metrics = compute_metrics(
        _frame([{"attempts": 10, "successes": 4, "sms_sent": 2, "voice_calls_made": 2}])
    )
    assert metrics["attempts_per_recovery"] == pytest.approx(2.5)
    assert metrics["contacts_per_recovery"] == pytest.approx(1.0)


def test_escalation_counts_as_a_contact():
    metrics = compute_metrics(
        _frame([{"successes": 1, "escalated_to_human": True}])
    )
    assert metrics["contacts_per_recovery"] == pytest.approx(1.0)


def test_ratios_are_none_rather_than_infinite_when_nothing_recovered():
    """An undefined ratio is reported as undefined, not as a huge number."""
    metrics = compute_metrics(
        _frame([{"successes": 0, "cycles": 3, "recovered_paise": 0}])
    )
    assert metrics["attempts_per_recovery"] is None
    assert metrics["contacts_per_recovery"] is None
    assert metrics["cost_per_100_rupees_recovered"] is None
    assert metrics["recovery_rate"] == 0.0


def test_days_to_recovery_median_and_p90():
    rows = [
        {"mandate_id": str(i), "days_to_recovery": day}
        for i, day in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    ]
    metrics = compute_metrics(_frame(rows))
    assert metrics["median_days_to_recovery"] == pytest.approx(5.5)
    assert metrics["p90_days_to_recovery"] > metrics["median_days_to_recovery"]


def test_days_to_recovery_ignores_mandates_that_never_recovered():
    rows = [
        {"mandate_id": "a", "days_to_recovery": 4},
        {"mandate_id": "b", "days_to_recovery": None},
    ]
    assert compute_metrics(_frame(rows))["median_days_to_recovery"] == 4


def test_over_intervention_rate_is_a_share_of_episodes():
    rows = [
        {"mandate_id": "a", "over_intervention": True},
        {"mandate_id": "b", "over_intervention": False},
        {"mandate_id": "c", "over_intervention": False},
        {"mandate_id": "d", "over_intervention": False},
    ]
    assert compute_metrics(_frame(rows))["over_intervention_rate"] == 0.25


def test_cost_per_hundred_rupees_is_scale_free():
    """Doubling both cost and recovery leaves the ratio unchanged."""
    small = compute_metrics(
        _frame([{"recovered_paise": 100_000, "total_cost_paise": 5_000}])
    )
    large = compute_metrics(
        _frame([{"recovered_paise": 200_000, "total_cost_paise": 10_000}])
    )
    assert small["cost_per_100_rupees_recovered"] == pytest.approx(5.0)
    assert large["cost_per_100_rupees_recovered"] == pytest.approx(5.0)


def test_empty_input_produces_a_well_formed_result():
    metrics = compute_metrics(pd.DataFrame())
    assert metrics["n_episodes"] == 0
    assert metrics["recovery_rate"] is None
    assert compute_metrics_by_arm(pd.DataFrame()) == {}


def test_metrics_split_by_arm():
    frame = _frame(
        [
            {"arm": "a", "mandate_id": "1", "net_recovery_paise": 10},
            {"arm": "b", "mandate_id": "2", "net_recovery_paise": 20},
        ]
    )
    by_arm = compute_metrics_by_arm(frame)
    assert set(by_arm) == {"a", "b"}
    assert by_arm["b"]["net_recovery_paise"] == 20


# --------------------------------------------------------------------------
# The paired comparison
# --------------------------------------------------------------------------


def _paired(deltas: list[int], control_net: int = 1_000_000) -> pd.DataFrame:
    """Two arms across len(deltas) seeds, with the given per-seed deltas."""
    rows = []
    for seed, delta in enumerate(deltas):
        rows.append(
            _episode(
                seed=seed,
                arm="control",
                mandate_id=f"c{seed}",
                net_recovery_paise=control_net,
            )
        )
        rows.append(
            _episode(
                seed=seed,
                arm="treatment",
                mandate_id=f"t{seed}",
                net_recovery_paise=control_net + delta,
            )
        )
    return pd.DataFrame(rows)


def test_per_seed_deltas_are_exact():
    deltas = [100, -50, 25]
    result = compare_arms(
        _paired(deltas),
        "treatment",
        "control",
        rng=np.random.default_rng(0),
        n_bootstrap=500,
    )
    assert result["per_seed_delta_paise"] == {0: 100, 1: -50, 2: 25}
    assert result["mean_delta_paise"] == pytest.approx(25.0)
    assert result["n_seeds"] == 3


def test_loss_rate_counts_the_seeds_where_treatment_lost():
    """The number that stops a fragile win reading as a solid one."""
    result = compare_arms(
        _paired([100, 100, 100, -10, -10]),
        "treatment",
        "control",
        rng=np.random.default_rng(0),
        n_bootstrap=500,
    )
    assert result["loss_rate"] == pytest.approx(0.4)
    assert result["n_seeds_lost"] == 2


def test_loss_rate_survives_a_large_mean_win():
    """A policy can win hugely on average and still lose a third of the time.

    If loss rate tracked the mean, it would be useless.
    """
    result = compare_arms(
        _paired([10_000_000, 10_000_000, -1, -1]),
        "treatment",
        "control",
        rng=np.random.default_rng(0),
        n_bootstrap=500,
    )
    assert result["mean_delta_paise"] > 0
    assert result["loss_rate"] == pytest.approx(0.5)


def test_a_tie_is_neither_a_win_nor_a_loss():
    result = compare_arms(
        _paired([0, 0, 10]),
        "treatment",
        "control",
        rng=np.random.default_rng(0),
        n_bootstrap=500,
    )
    assert result["n_seeds_tied"] == 2
    assert result["loss_rate"] == 0.0


def test_the_confidence_interval_brackets_the_mean():
    result = compare_arms(
        _paired([120, 80, 100, 90, 110, 95, 105]),
        "treatment",
        "control",
        rng=np.random.default_rng(7),
        n_bootstrap=4000,
    )
    assert result["ci_low_paise"] <= result["mean_delta_paise"]
    assert result["mean_delta_paise"] <= result["ci_high_paise"]
    assert result["ci_excludes_zero"] is True


def test_a_noisy_result_has_an_interval_spanning_zero():
    result = compare_arms(
        _paired([500, -480, 300, -520, 100, -90]),
        "treatment",
        "control",
        rng=np.random.default_rng(11),
        n_bootstrap=4000,
    )
    assert result["ci_low_paise"] < 0 < result["ci_high_paise"]
    assert result["ci_excludes_zero"] is False


def test_more_seeds_narrow_the_interval():
    """A basic sanity property; if it fails the bootstrap is not resampling."""
    spread = [100, -20, 60, 10, 90, -30, 70, 20]

    def width(deltas):
        result = compare_arms(
            _paired(deltas),
            "treatment",
            "control",
            rng=np.random.default_rng(3),
            n_bootstrap=4000,
        )
        return result["ci_high_paise"] - result["ci_low_paise"]

    assert width(spread * 4) < width(spread)


def test_the_bootstrap_is_reproducible_under_a_seed():
    frame = _paired([120, 80, -10, 100])
    kwargs = dict(n_bootstrap=2000)
    first = compare_arms(
        frame, "treatment", "control", rng=np.random.default_rng(5), **kwargs
    )
    second = compare_arms(
        frame, "treatment", "control", rng=np.random.default_rng(5), **kwargs
    )
    assert first == second


def test_comparison_requires_an_explicit_generator():
    frame = _paired([1, 2])
    for bad in (None, 42, np.random.RandomState(0)):
        with pytest.raises(TypeError):
            compare_arms(frame, "treatment", "control", rng=bad)


def test_unpaired_seed_sets_are_refused():
    """Comparing arms run on different seeds is not a paired comparison."""
    frame = _paired([10, 20])
    frame = frame[~((frame["arm"] == "treatment") & (frame["seed"] == 1))]
    with pytest.raises(ValueError, match="same seeds"):
        compare_arms(
            frame, "treatment", "control", rng=np.random.default_rng(0)
        )


def test_an_unknown_arm_fails_loudly():
    with pytest.raises(KeyError):
        compare_arms(
            _paired([1]), "nope", "control", rng=np.random.default_rng(0)
        )


def test_the_summary_line_always_states_the_loss_rate():
    result = compare_arms(
        _paired([100, -50]),
        "treatment",
        "control",
        rng=np.random.default_rng(0),
        n_bootstrap=500,
    )
    line = summarise_comparison(result)
    assert "lost on" in line
    assert "1/2" in line
    assert "CI" in line
