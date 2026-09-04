"""Prove the simulator produces the distributions the calibration claims.

Runs the world with **no policy at all**: every mandate is presented once per
cycle on its due day, and a failure is simply not retried. That is the null
world the whole project is measured against, and it is the only setting in
which the emergent failure mix can be compared with the calibrated targets
without a policy confounding it.

Run it with::

    python scripts/validate_simulator.py

Output lands in ``results/simulator_validation/``.

On fitted parameters
--------------------
Three quantities here are **fitted, not observed**: the mandate amount
distribution and the share of attempts presented inside the restricted window,
both defined in this module. They are free structural choices, and they were
chosen so the emergent failure mix matches the calibrated targets.

That direction is deliberate. Overall failure rate and the failure-mode shares
are the figures most likely to be replaced by real published data, so they are
treated as targets. The amount distribution and presentment timing are
properties of a particular merchant's book that no public source will ever
settle, so they absorb the fit. None of it is empirical, and none of it is
dressed up as such -- see ``docs/CALIBRATION.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow running without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mandate_recovery.calibration import (  # noqa: E402
    DEFAULT_CALIBRATION,
    CalibrationSet,
)
from mandate_recovery.sim.outcomes import (  # noqa: E402
    resolve_attempt,
    revoke_eligible_mandates,
)
from mandate_recovery.sim.world import DAYS_IN_MONTH, HOURS_IN_DAY, World  # noqa: E402
from mandate_recovery.types import (  # noqa: E402
    Attempt,
    AttemptOutcome,
    Mandate,
    MandateStatus,
    Rail,
)

# --------------------------------------------------------------------------
# Run shape
# --------------------------------------------------------------------------

N_SEEDS: Final = 50
N_MANDATES: Final = 500
N_DAYS: Final = 90

#: One mandate per customer. Two mandates competing for one balance is
#: realistic but confounds the funds-failure rate, and this run exists to
#: measure that rate cleanly.
N_CUSTOMERS: Final = N_MANDATES


# --------------------------------------------------------------------------
# Fitted structural parameters (see the module docstring)
# --------------------------------------------------------------------------

#: Mandate amounts are lognormal in paise. Fitted so that the tail crossing
#: the per-transaction ceiling reproduces the calibrated limit share, and the
#: body crossing typical balances reproduces the calibrated funds share.
MANDATE_AMOUNT_PAISE_MEDIAN: Final = 880_000
MANDATE_AMOUNT_LOGNORMAL_SIGMA: Final = 1.20

#: Share of attempts presented inside the NPCI restricted window. Fitted to
#: reproduce the calibrated window-rejection share given the calibrated
#: rejection probability.
SHARE_OF_ATTEMPTS_PRESENTED_IN_RESTRICTED_WINDOW: Final = 0.086

FAILURE_MODES: Final = (
    "INSUFFICIENT_FUNDS",
    "TECHNICAL_DECLINE",
    "LIMIT_EXCEEDED",
    "WINDOW_REJECTED",
)


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Everything one seed produced."""

    seed: int
    attempts: int = 0
    failures: int = 0
    outcome_counts: Counter = field(default_factory=Counter)
    code_counts: Counter = field(default_factory=Counter)
    attempts_by_hour: list[int] = field(
        default_factory=lambda: [0] * HOURS_IN_DAY
    )
    failures_by_hour: list[int] = field(
        default_factory=lambda: [0] * HOURS_IN_DAY
    )
    revoked: int = 0
    balance_trajectories: list[list[int]] = field(default_factory=list)
    salary_days: list[int] = field(default_factory=list)


def _restricted_hours(calibration: CalibrationSet) -> tuple[list[int], list[int]]:
    """(hours inside the window, hours outside it)."""
    windows = calibration.restricted_window_hours.value
    inside = [
        hour
        for hour in range(HOURS_IN_DAY)
        if any(start <= hour < end for start, end in windows)
    ]
    outside = [hour for hour in range(HOURS_IN_DAY) if hour not in inside]
    return inside, outside


def _presentment_hours(
    count: int,
    calibration: CalibrationSet,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hours at which a batch of debits is presented."""
    inside, outside = _restricted_hours(calibration)
    in_window = (
        rng.random(count) < SHARE_OF_ATTEMPTS_PRESENTED_IN_RESTRICTED_WINDOW
    )
    hours = np.where(
        in_window,
        rng.choice(inside, size=count),
        rng.choice(outside, size=count),
    )
    return hours


def simulate_seed(
    seed: int,
    calibration: CalibrationSet = DEFAULT_CALIBRATION,
    *,
    n_mandates: int = N_MANDATES,
    n_days: int = N_DAYS,
    trajectory_sample: int = 0,
) -> SeedResult:
    """Run one seed of the null world and return what happened.

    Every stochastic draw comes from generators seeded from ``seed`` alone, so
    the whole run is reproducible from the seed and the calibration.
    """
    world_rng = np.random.default_rng([seed, 1])
    setup_rng = np.random.default_rng([seed, 2])
    outcome_rng = np.random.default_rng([seed, 3])
    revocation_rng = np.random.default_rng([seed, 4])

    world = World(
        calibration, world_rng, n_customers=n_mandates, n_days=n_days
    )

    amounts = np.rint(
        setup_rng.lognormal(
            mean=float(np.log(MANDATE_AMOUNT_PAISE_MEDIAN)),
            sigma=MANDATE_AMOUNT_LOGNORMAL_SIGMA,
            size=n_mandates,
        )
    ).astype(np.int64)
    np.maximum(amounts, 100, out=amounts)  # a mandate is at least one rupee
    due_days = setup_rng.integers(1, DAYS_IN_MONTH + 1, size=n_mandates)

    mandates = [
        Mandate(
            id=f"m{index:06d}",
            customer_id=world.customer_id_for(index),
            amount_paise=int(amounts[index]),
            day_of_month=int(due_days[index]),
            created_on_day=0,
            status=MandateStatus.ACTIVE,
        )
        for index in range(n_mandates)
    ]

    result = SeedResult(seed=seed)
    funds_failures: Counter = Counter()
    sampled = list(range(min(trajectory_sample, n_mandates)))
    if sampled:
        result.salary_days = [
            world.latent_state(index).salary_day for index in sampled
        ]
        result.balance_trajectories = [[] for _ in sampled]

    for day in range(n_days):
        for position, index in enumerate(sampled):
            result.balance_trajectories[position].append(
                world.balance_paise(index)
            )

        due = [
            position
            for position, mandate in enumerate(mandates)
            if mandate.status is MandateStatus.ACTIVE
            and mandate.day_of_month == world.day_of_month
        ]
        if due:
            hours = _presentment_hours(len(due), calibration, outcome_rng)
            for position, hour in zip(due, hours):
                mandate = mandates[position]
                attempt = Attempt(
                    mandate_id=mandate.id,
                    # Only the hour is read; the world supplies the day.
                    scheduled_at=_stamp(int(hour)),
                    rail=Rail.UPI_AUTOPAY,
                )
                response = resolve_attempt(
                    world, mandate, attempt, outcome_rng
                )

                result.attempts += 1
                result.attempts_by_hour[int(hour)] += 1
                result.outcome_counts[response.outcome.value] += 1
                result.code_counts[response.raw_code] += 1
                if response.outcome is not AttemptOutcome.SUCCESS:
                    result.failures += 1
                    result.failures_by_hour[int(hour)] += 1
                if response.outcome is AttemptOutcome.INSUFFICIENT_FUNDS:
                    funds_failures[mandate.id] += 1

        mandates = list(
            revoke_eligible_mandates(
                mandates, funds_failures, calibration, revocation_rng
            )
        )

        if day + 1 < n_days:
            world.advance_day()

    result.revoked = sum(
        1 for mandate in mandates if mandate.status is MandateStatus.REVOKED
    )
    return result


def _stamp(hour: int):
    from datetime import datetime

    return datetime(2026, 1, 1, hour, 0)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(results: Iterable[SeedResult]) -> dict:
    """Observed rates across every seed, plus the calibrated targets."""
    results = list(results)
    attempts = sum(r.attempts for r in results)
    failures = sum(r.failures for r in results)

    outcome_counts: Counter = Counter()
    for result in results:
        outcome_counts.update(result.outcome_counts)

    mode_counts = {mode: outcome_counts.get(mode, 0) for mode in FAILURE_MODES}
    mode_total = sum(mode_counts.values()) or 1

    calibration = DEFAULT_CALIBRATION
    return {
        "n_seeds": len(results),
        "n_attempts": attempts,
        "n_failures": failures,
        "observed": {
            "failure_rate": failures / attempts if attempts else 0.0,
            "share_of_failures_insufficient_funds": (
                mode_counts["INSUFFICIENT_FUNDS"] / mode_total
            ),
            "share_of_failures_technical": (
                mode_counts["TECHNICAL_DECLINE"] / mode_total
            ),
            "share_of_failures_limit": mode_counts["LIMIT_EXCEEDED"] / mode_total,
            "share_of_failures_window_rejected": (
                mode_counts["WINDOW_REJECTED"] / mode_total
            ),
        },
        "calibrated": {
            "failure_rate": calibration.upi_autopay_execution_failure_rate.value,
            "share_of_failures_insufficient_funds": (
                calibration.share_of_failures_insufficient_funds.value
            ),
            "share_of_failures_technical": (
                calibration.share_of_failures_technical.value
            ),
            "share_of_failures_limit": calibration.share_of_failures_limit.value,
            "share_of_failures_window_rejected": (
                calibration.share_of_failures_window_rejected.value
            ),
        },
        "outcome_counts": dict(outcome_counts),
        "mandates_revoked": sum(r.revoked for r in results),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(output_root: Path | None = None) -> dict:
    from mandate_recovery import figures

    output = Path(output_root or REPO_ROOT / "results" / "simulator_validation")
    (output / "figures").mkdir(parents=True, exist_ok=True)

    seeds = list(range(N_SEEDS))
    results = [
        simulate_seed(seed, trajectory_sample=6 if seed == seeds[0] else 0)
        for seed in seeds
    ]
    metrics = aggregate(results)

    (output / "config.json").write_text(
        json.dumps(
            {
                "git_sha": _git_sha(),
                "seeds": seeds,
                "n_mandates": N_MANDATES,
                "n_days": N_DAYS,
                "n_customers": N_CUSTOMERS,
                "fitted_parameters": {
                    "mandate_amount_paise_median": MANDATE_AMOUNT_PAISE_MEDIAN,
                    "mandate_amount_lognormal_sigma": (
                        MANDATE_AMOUNT_LOGNORMAL_SIGMA
                    ),
                    "share_of_attempts_presented_in_restricted_window": (
                        SHARE_OF_ATTEMPTS_PRESENTED_IN_RESTRICTED_WINDOW
                    ),
                },
                "calibration": json.loads(DEFAULT_CALIBRATION.model_dump_json()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    _write_figures(output / "figures", results, metrics, figures)
    return metrics


def _write_figures(directory: Path, results, metrics, figures) -> None:
    observed = metrics["observed"]
    calibrated = metrics["calibrated"]

    shares = {
        mode: observed[key]
        for mode, key in zip(
            FAILURE_MODES,
            (
                "share_of_failures_insufficient_funds",
                "share_of_failures_technical",
                "share_of_failures_limit",
                "share_of_failures_window_rejected",
            ),
        )
    }
    figures.save_figure(
        figures.failure_mode_breakdown(shares),
        directory / "failure_mode_breakdown",
        f"Failure modes across {metrics['n_failures']:,} failed attempts in "
        f"{metrics['n_seeds']} seeded runs with no recovery policy. "
        "Insufficient funds dominates, which is why retry timing rather than "
        "retry volume is the lever this project tests.",
    )

    attempts_by_hour = [0] * HOURS_IN_DAY
    failures_by_hour = [0] * HOURS_IN_DAY
    for result in results:
        for hour in range(HOURS_IN_DAY):
            attempts_by_hour[hour] += result.attempts_by_hour[hour]
            failures_by_hour[hour] += result.failures_by_hour[hour]

    figures.save_figure(
        figures.failure_rate_by_hour(
            attempts_by_hour,
            failures_by_hour,
            DEFAULT_CALIBRATION.restricted_window_hours.value,
        ),
        directory / "failure_rate_by_hour",
        "Failure rate by hour of day. The shaded bands are the NPCI window "
        "in which recurring debits are deprioritised; the step up inside them "
        "is the cost of presenting a debit at peak, and it is why the "
        "scheduler treats those hours as a hard exclusion.",
    )

    with_trajectories = next(
        (r for r in results if r.balance_trajectories), None
    )
    if with_trajectories is not None:
        figures.save_figure(
            figures.balance_trajectory_sample(
                with_trajectories.balance_trajectories,
                with_trajectories.salary_days,
            ),
            directory / "balance_trajectory_sample",
            "Six customer balance trajectories over 90 simulated days. The "
            "sawtooth is the salary cycle: a debit presented two days before "
            "payday fails where the identical debit two days later succeeds. "
            "The policy never sees these curves.",
        )

    figures.save_figure(
        figures.observed_vs_calibrated(
            observed,
            calibrated,
            tolerance={
                "failure_rate": 0.02,
                "share_of_failures_insufficient_funds": 0.03,
                "share_of_failures_technical": 0.03,
                "share_of_failures_limit": 0.03,
                "share_of_failures_window_rejected": 0.03,
            },
        ),
        directory / "observed_vs_calibrated",
        "What the calibration claims against what the simulator produced, "
        "with the tolerance bands the test suite enforces (2 points on the "
        "overall failure rate, 3 on each mode share). Agreement here is not "
        "evidence the parameters are right -- they are placeholders -- only "
        "that the simulator does what it says it does.",
    )


if __name__ == "__main__":
    summary = main()
    observed, calibrated = summary["observed"], summary["calibrated"]
    print(f"attempts: {summary['n_attempts']:,}  failures: {summary['n_failures']:,}")
    print(f"{'metric':<44} {'observed':>10} {'target':>10} {'delta':>8}")
    for key in calibrated:
        delta = observed[key] - calibrated[key]
        print(f"{key:<44} {observed[key]:>10.4f} {calibrated[key]:>10.4f} {delta:>+8.4f}")
