"""Does the agent's advantage survive being wrong about the world?

This is the direct answer to "you built the simulator to favour your agent."
Every calibrated parameter in this project is an unsourced placeholder, so the
only defensible claim is a *conditional* one: here is the range of worlds in
which the result holds, and here is where it stops holding.

The grid varies four things, each by a lever that actually changes the
mechanics rather than by editing the target it is measured against:

===================  ==========================================  =======
Axis                 Lever                                       Levels
===================  ==========================================  =======
Failure regime       mandate amount distribution                 3
Funds-share regime   per-transaction ceiling                     3
Restricted window    NPCI window width (none / calibrated / wide) 3
Bank availability    per-tier uptime                             2
===================  ==========================================  =======

54 cells, 40 seeds each. The *observed* failure rate and funds share are
recorded per cell, so the axis labels are grounded in what the simulator
actually produced rather than in what the label asserts.

Which agent is swept
--------------------
The **heuristic** agent, against the fixed-schedule baseline. The build plan
says to sweep "the winning agent", but there is no winner: the ablation at
commit 26 is blocked on API quota, and at commit 20 the heuristic *tied* the
baseline rather than beating it. Sweeping the deterministic agent is the
honest reading — it is the best arm that has actually been measured, it needs
no API key, and a reviewer can reproduce it. When the ablation runs, whichever
arm wins should be swept the same way.

Run it with::

    python scripts/run_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mandate_recovery import figures  # noqa: E402
from mandate_recovery.calibration import DEFAULT_CALIBRATION, BankTier  # noqa: E402
from mandate_recovery.harness import (  # noqa: E402
    ExperimentConfig,
    compare_arms,
    compute_metrics_by_arm,
    run_experiment,
    write_experiment,
)
from mandate_recovery.policies import FixedSchedulePolicy, HeuristicPolicy  # noqa: E402

EXPERIMENT_ID = "sensitivity"
N_SEEDS = 40
N_MANDATES = 250
N_DAYS = 90
BOOTSTRAP_SEED = 20260904

TREATMENT = "heuristic"
CONTROL = "fixed_schedule"

#: Scales the mandate amount distribution. Larger mandates fail more often,
#: which is the mechanical way to move the overall failure rate.
FAILURE_REGIMES = {"low": 0.55, "calibrated": 1.0, "high": 1.75}

#: Scales the per-transaction ceiling. A tighter ceiling converts funds
#: failures into limit failures, moving the funds share without touching the
#: overall rate much.
FUNDS_SHARE_REGIMES = {"low": 0.30, "calibrated": 1.0, "high": 4.0}

WINDOW_REGIMES = {
    "none": (),
    "calibrated": ((10, 13), (17, 21)),
    "wide": ((8, 14), (16, 22)),
}

AVAILABILITY_REGIMES = {"degraded": 0.93, "calibrated": 1.0}


def _calibration_for(failure: str, funds: str, window: str, availability: str):
    """One grid cell's world, built by copying the frozen calibration."""
    base = DEFAULT_CALIBRATION
    limits = {
        tier: int(value * FUNDS_SHARE_REGIMES[funds])
        for tier, value in base.per_txn_limit_paise_by_tier.value.items()
    }
    uptime = {
        tier: min(1.0, value * AVAILABILITY_REGIMES[availability])
        for tier, value in base.bank_availability_by_tier.value.items()
    }
    return base.model_copy(
        update={
            "per_txn_limit_paise_by_tier": (
                base.per_txn_limit_paise_by_tier.model_copy(
                    update={"value": limits}
                )
            ),
            "bank_availability_by_tier": (
                base.bank_availability_by_tier.model_copy(update={"value": uptime})
            ),
            "restricted_window_hours": (
                base.restricted_window_hours.model_copy(
                    update={"value": WINDOW_REGIMES[window]}
                )
            ),
        }
    )


def _observed(frame: pd.DataFrame) -> dict[str, float]:
    """What the cell actually produced, so labels are grounded in measurement."""
    control = frame[frame["arm"] == CONTROL]
    attempts = int(control["attempts"].sum()) or 1
    failures = int(control["failures"].sum())
    codes = ",".join(control["attempt_outcomes"]).split(",")
    codes = [code for code in codes if code and code != "SUCCESS"]
    funds = sum(1 for code in codes if code == "INSUFFICIENT_FUNDS")
    # Per *attempt*, including retries, so this is not comparable with the
    # null-world rate in commit 7 -- retries fail far more often than first
    # presentments, which is why these numbers run much higher.
    return {
        "observed_failure_rate_per_attempt": failures / attempts,
        "observed_funds_share": funds / (len(codes) or 1),
    }


def main(results_root: Path | None = None, overwrite: bool = True) -> dict:
    cells = []
    total = (
        len(FAILURE_REGIMES)
        * len(FUNDS_SHARE_REGIMES)
        * len(WINDOW_REGIMES)
        * len(AVAILABILITY_REGIMES)
    )
    index = 0

    for failure in FAILURE_REGIMES:
        for funds in FUNDS_SHARE_REGIMES:
            for window in WINDOW_REGIMES:
                for availability in AVAILABILITY_REGIMES:
                    index += 1
                    config = ExperimentConfig(
                        experiment_id=f"{EXPERIMENT_ID}-{index}",
                        seeds=list(range(N_SEEDS)),
                        n_customers=N_MANDATES,
                        n_mandates=N_MANDATES,
                        n_days=N_DAYS,
                        calibration=_calibration_for(
                            failure, funds, window, availability
                        ),
                        mandate_amount_paise_median=int(
                            880_000 * FAILURE_REGIMES[failure]
                        ),
                    )
                    episodes, _ = run_experiment(
                        {
                            CONTROL: lambda world, mapping: FixedSchedulePolicy(),
                            TREATMENT: lambda world, mapping: HeuristicPolicy(
                                calibration=config.calibration
                            ),
                        },
                        config,
                    )
                    frame = pd.DataFrame([e.to_row() for e in episodes])
                    comparison = compare_arms(
                        frame,
                        TREATMENT,
                        CONTROL,
                        rng=np.random.default_rng(BOOTSTRAP_SEED),
                    )
                    by_arm = compute_metrics_by_arm(frame)

                    cells.append(
                        {
                            "cell": index,
                            "failure_regime": failure,
                            "funds_share_regime": funds,
                            "window_regime": window,
                            "availability_regime": availability,
                            "mean_delta_paise": comparison["mean_delta_paise"],
                            "ci_low_paise": comparison["ci_low_paise"],
                            "ci_high_paise": comparison["ci_high_paise"],
                            "loss_rate": comparison["loss_rate"],
                            "ci_excludes_zero": comparison["ci_excludes_zero"],
                            "treatment_net_paise": by_arm[TREATMENT][
                                "net_recovery_paise"
                            ],
                            "control_net_paise": by_arm[CONTROL][
                                "net_recovery_paise"
                            ],
                            **_observed(frame),
                        }
                    )
                    print(
                        f"[{index:>2}/{total}] {failure:<10} {funds:<10} "
                        f"{window:<10} {availability:<10} "
                        f"delta={comparison['mean_delta_paise'] / 100_000:>+8.1f}k "
                        f"loss={comparison['loss_rate']:>5.0%} "
                        f"fail={cells[-1]['observed_failure_rate_per_attempt']:>5.1%}",
                        flush=True,
                    )

    cell_frame = pd.DataFrame(cells)
    survived = int((cell_frame["mean_delta_paise"] > 0).sum())
    metrics = {
        "n_cells": len(cells),
        "n_seeds_per_cell": N_SEEDS,
        "treatment": TREATMENT,
        "control": CONTROL,
        "cells_where_advantage_holds": survived,
        "advantage_survival_rate": survived / len(cells),
        "cells_with_ci_excluding_zero": int(cell_frame["ci_excludes_zero"].sum()),
        "worst_cell": cell_frame.loc[
            cell_frame["mean_delta_paise"].idxmin()
        ].to_dict(),
        "best_cell": cell_frame.loc[
            cell_frame["mean_delta_paise"].idxmax()
        ].to_dict(),
        "cells": cells,
    }

    directory = write_experiment(
        EXPERIMENT_ID,
        {
            "n_cells": len(cells),
            "n_seeds_per_cell": N_SEEDS,
            "n_mandates": N_MANDATES,
            "n_days": N_DAYS,
            "failure_regimes": FAILURE_REGIMES,
            "funds_share_regimes": FUNDS_SHARE_REGIMES,
            "window_regimes": {k: list(v) for k, v in WINDOW_REGIMES.items()},
            "availability_regimes": AVAILABILITY_REGIMES,
        },
        cell_frame,
        pd.DataFrame(),
        metrics,
        results_root=results_root,
        overwrite=overwrite,
    )
    _write_figures(directory / "figures", cells, metrics)
    _print_summary(metrics)
    return metrics


def _write_figures(directory: Path, cells, metrics) -> None:
    figures.save_figure(
        figures.sensitivity_heatmap(
            cells,
            "failure_regime",
            "window_regime",
            row_label="Failure regime (mandate amount scale)",
            column_label="NPCI restricted window",
        ),
        directory / "sensitivity_heatmap",
        "Mean per-seed advantage of the heuristic agent over the fixed "
        "schedule, averaged across the ceiling and availability axes. Green "
        "is an advantage, red is a loss. Note that the 'none' and "
        "'calibrated' window columns are identical: neither arm ever "
        "presents a debit inside the calibrated NPCI window, so widening it "
        "is the only change that reaches them.",
    )

    figures.save_figure(
        figures.advantage_survival(cells),
        directory / "advantage_survival",
        f"Every one of the {metrics['n_cells']} regimes, sorted by advantage. "
        f"The agent is ahead in {metrics['cells_where_advantage_holds']} of "
        f"them ({metrics['advantage_survival_rate']:.0%}). A bounded, honest "
        "claim is what this figure supports: not that the agent always wins, "
        "but exactly where it does and does not.",
    )


def _print_summary(metrics) -> None:
    print(
        f"\nadvantage holds in {metrics['cells_where_advantage_holds']}"
        f"/{metrics['n_cells']} cells "
        f"({metrics['advantage_survival_rate']:.0%}); "
        f"{metrics['cells_with_ci_excluding_zero']} have a CI excluding zero"
    )
    for label in ("worst_cell", "best_cell"):
        cell = metrics[label]
        print(
            f"  {label:<11} {cell['failure_regime']}/{cell['funds_share_regime']}"
            f"/{cell['window_regime']}/{cell['availability_regime']}"
            f"  delta={cell['mean_delta_paise'] / 100_000:+.1f}k"
            f"  loss={cell['loss_rate']:.0%}"
        )


if __name__ == "__main__":
    main()
