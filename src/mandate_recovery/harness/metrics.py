"""Metrics, and the paired comparison that decides whether an arm actually won.

Summary metrics describe one arm. :func:`compare_arms` is the one that settles
anything: because the harness runs every arm on paired worlds, the same seed
gives a matched pair, and the per-seed delta is a within-subject comparison
rather than a difference of two noisy averages.

On loss rate
------------
``loss_rate`` is the fraction of seeds on which the treatment arm did **worse**
than the control. It is reported next to the mean delta, always, and it is
never suppressed, rounded away, or relegated to an appendix.

A policy that wins by a large margin on average while losing on a third of
seeds is a different proposition from one that wins by a smaller margin and
almost never loses, and a mean with a confidence interval does not distinguish
them. For a system that moves other people's money, the second is usually the
one you want. Reporting only the mean is the single easiest way to make a
fragile result look solid, and this module refuses to make that easy.

The bootstrap resamples the per-seed deltas, not the episodes. Episodes within
a seed are not independent — they share a world, a bank, and a calendar — so
resampling them would understate the interval badly.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "compute_metrics",
    "compute_metrics_by_arm",
    "compare_arms",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
]

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """A ratio, or None when it is undefined rather than a fake infinity."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def compute_metrics(episodes: pd.DataFrame) -> dict[str, Any]:
    """Summary metrics for one arm's episodes.

    ``recovery_rate`` is per *billing cycle*, not per mandate: a mandate that
    runs three cycles and is collected twice recovered two thirds of what it
    was owed, and the mandate-level view would hide that.
    """
    if episodes.empty:
        return {
            "n_episodes": 0,
            "n_cycles": 0,
            "n_seeds": 0,
            "recovery_rate": None,
            "recovered_paise": 0,
            "net_recovery_paise": 0,
            "total_cost_paise": 0,
            "attempts_per_recovery": None,
            "contacts_per_recovery": None,
            "median_days_to_recovery": None,
            "p90_days_to_recovery": None,
            "over_intervention_rate": None,
            "cost_per_100_rupees_recovered": None,
        }

    successes = int(episodes["successes"].sum())
    cycles = int(episodes["cycles"].sum())
    attempts = int(episodes["attempts"].sum())
    contacts = int(
        episodes["sms_sent"].sum()
        + episodes["voice_calls_made"].sum()
        + episodes["escalated_to_human"].astype(int).sum()
    )
    recovered = int(episodes["recovered_paise"].sum())
    total_cost = int(episodes["total_cost_paise"].sum())

    days = episodes["days_to_recovery"].dropna()

    return {
        "n_episodes": int(len(episodes)),
        "n_cycles": cycles,
        "n_seeds": int(episodes["seed"].nunique()),
        "recovery_rate": _safe_ratio(successes, cycles),
        "recovered_paise": recovered,
        "net_recovery_paise": int(episodes["net_recovery_paise"].sum()),
        "total_cost_paise": total_cost,
        "gateway_cost_paise": int(episodes["gateway_cost_paise"].sum()),
        "contact_cost_paise": int(episodes["contact_cost_paise"].sum()),
        "churn_cost_paise": int(episodes["churn_cost_paise"].sum()),
        "attempts_per_recovery": _safe_ratio(attempts, successes),
        "contacts_per_recovery": _safe_ratio(contacts, successes),
        "median_days_to_recovery": (
            float(days.median()) if len(days) else None
        ),
        "p90_days_to_recovery": (
            float(days.quantile(0.90)) if len(days) else None
        ),
        "over_intervention_rate": float(episodes["over_intervention"].mean()),
        # Rupees of cost per 100 rupees collected. Unit-free, so it survives
        # a change of scale in the mandate amount distribution.
        "cost_per_100_rupees_recovered": (
            None if recovered == 0 else 100.0 * total_cost / recovered
        ),
    }


def compute_metrics_by_arm(episodes: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summary metrics for every arm present in the frame."""
    if episodes.empty:
        return {}
    return {
        str(arm): compute_metrics(group)
        for arm, group in episodes.groupby("arm", sort=True)
    }


def _net_by_seed(episodes: pd.DataFrame, arm: str) -> pd.Series:
    subset = episodes[episodes["arm"] == arm]
    if subset.empty:
        raise KeyError(
            f"arm {arm!r} is not in these episodes; available arms are "
            f"{sorted(episodes['arm'].unique())}"
        )
    return subset.groupby("seed")["net_recovery_paise"].sum().sort_index()


def compare_arms(
    episodes: pd.DataFrame,
    treatment: str,
    control: str,
    *,
    rng: np.random.Generator,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired comparison of two arms, seed by seed.

    Returns the per-seed deltas, the mean delta with a bootstrap confidence
    interval, and ``loss_rate`` — the share of seeds where the treatment did
    worse. **Report the loss rate wherever you report the mean.**

    Args:
        rng: explicit generator, per the determinism invariant. The bootstrap
            is stochastic and must be reproducible from the run config.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an explicit numpy.random.Generator; the harness "
            "never uses global random state"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    treatment_net = _net_by_seed(episodes, treatment)
    control_net = _net_by_seed(episodes, control)

    shared_seeds = treatment_net.index.intersection(control_net.index)
    if len(shared_seeds) == 0:
        raise ValueError(
            f"{treatment!r} and {control!r} share no seeds; the comparison "
            "would not be paired"
        )
    if len(shared_seeds) != len(treatment_net) or len(shared_seeds) != len(
        control_net
    ):
        raise ValueError(
            "arms were run on different seed sets; a paired comparison "
            "requires the same seeds on both sides"
        )

    deltas = (treatment_net[shared_seeds] - control_net[shared_seeds]).to_numpy()
    n_seeds = len(deltas)

    resampled = rng.choice(deltas, size=(n_bootstrap, n_seeds), replace=True)
    means = resampled.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])

    return {
        "treatment": treatment,
        "control": control,
        "n_seeds": int(n_seeds),
        "per_seed_delta_paise": {
            int(seed): int(delta) for seed, delta in zip(shared_seeds, deltas)
        },
        "mean_delta_paise": float(deltas.mean()),
        "median_delta_paise": float(np.median(deltas)),
        "ci_low_paise": float(low),
        "ci_high_paise": float(high),
        "confidence": confidence,
        "n_bootstrap": int(n_bootstrap),
        # The number that stops a fragile win being reported as a solid one.
        "loss_rate": float((deltas < 0).mean()),
        "n_seeds_lost": int((deltas < 0).sum()),
        "n_seeds_tied": int((deltas == 0).sum()),
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def summarise_comparison(comparison: Mapping[str, Any]) -> str:
    """One human-readable line, with the loss rate attached to the claim."""
    mean_rupees = comparison["mean_delta_paise"] / 100.0
    low = comparison["ci_low_paise"] / 100.0
    high = comparison["ci_high_paise"] / 100.0
    return (
        f"{comparison['treatment']} vs {comparison['control']}: "
        f"mean delta Rs {mean_rupees:,.0f} "
        f"(95% CI Rs {low:,.0f} to Rs {high:,.0f}), "
        f"lost on {comparison['n_seeds_lost']}/{comparison['n_seeds']} seeds "
        f"({comparison['loss_rate']:.1%})"
    )
