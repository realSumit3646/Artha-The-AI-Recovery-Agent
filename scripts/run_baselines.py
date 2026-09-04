"""The first real numbers: three arms, 120 paired seeds.

Runs the no-intervention floor, the industry-standard fixed schedule, and the
perfect-information oracle against identical worlds, and writes everything a
later claim would need to cite.

Run it with::

    python scripts/run_baselines.py

Output lands in ``results/baselines/``.

What this run establishes
-------------------------
The **bounds**. `do_nothing` is what a merchant collects by doing nothing at
all; `oracle` is what perfect knowledge of every balance and every bank outage
could collect by retiming alone. Everything real lives between them, and the
gap between `fixed_schedule` and `oracle` is the headroom any agent is
competing for. Quoting a recovery improvement without that denominator is how
a 2% gain gets sold as transformative.

Every comparison here is paired: arm A and arm B face bit-identical worlds on
each seed, so the per-seed delta is a within-subject measurement. The loss
rate — the share of seeds where an arm did *worse* than its control — is
printed alongside every mean, and is not optional.
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
from mandate_recovery.harness import (  # noqa: E402
    ExperimentConfig,
    compare_arms,
    compute_metrics_by_arm,
    run_experiment,
    summarise_comparison,
    write_experiment,
)
from mandate_recovery.harness.storage import experiment_directory  # noqa: E402
from mandate_recovery.policies import (  # noqa: E402
    DoNothingPolicy,
    FixedSchedulePolicy,
    OraclePolicy,
)

EXPERIMENT_ID = "baselines"
N_SEEDS = 120
N_MANDATES = 500
N_DAYS = 90

#: Seed for the bootstrap itself, so published intervals are reproducible.
BOOTSTRAP_SEED = 20260904

ARMS = {
    "do_nothing": lambda world, mapping: DoNothingPolicy(),
    "fixed_schedule": lambda world, mapping: FixedSchedulePolicy(),
    "oracle": lambda world, mapping: OraclePolicy(world, mapping),
}

#: Every comparison this run makes, as (treatment, control). Stated up front
#: so the count is visible: three comparisons, not "however many looked good".
COMPARISONS = (
    ("fixed_schedule", "do_nothing"),
    ("oracle", "fixed_schedule"),
    ("oracle", "do_nothing"),
)


def main(results_root: Path | None = None, overwrite: bool = True) -> dict:
    config = ExperimentConfig(
        experiment_id=EXPERIMENT_ID,
        seeds=list(range(N_SEEDS)),
        n_customers=N_MANDATES,
        n_mandates=N_MANDATES,
        n_days=N_DAYS,
    )

    episodes, decisions = run_experiment(ARMS, config)
    episode_frame = pd.DataFrame([e.to_row() for e in episodes])
    decision_frame = pd.DataFrame([d.__dict__ for d in decisions])

    by_arm = compute_metrics_by_arm(episode_frame)
    comparisons = {
        f"{treatment}_vs_{control}": compare_arms(
            episode_frame,
            treatment,
            control,
            rng=np.random.default_rng(BOOTSTRAP_SEED),
        )
        for treatment, control in COMPARISONS
    }

    net_by_arm = {arm: by_arm[arm]["net_recovery_paise"] for arm in ARMS}
    headroom = net_by_arm["oracle"] - net_by_arm["fixed_schedule"]
    captured = (
        net_by_arm["fixed_schedule"] - net_by_arm["do_nothing"]
    ) / max(1, net_by_arm["oracle"] - net_by_arm["do_nothing"])

    metrics = {
        "n_seeds": N_SEEDS,
        "n_mandates": N_MANDATES,
        "n_days": N_DAYS,
        "n_comparisons_run": len(COMPARISONS),
        "by_arm": by_arm,
        "comparisons": comparisons,
        "headroom_paise": int(headroom),
        "baseline_share_of_available_headroom": float(captured),
    }

    directory = write_experiment(
        EXPERIMENT_ID,
        config.to_dict(),
        episode_frame,
        decision_frame,
        metrics,
        results_root=results_root,
        overwrite=overwrite,
    )
    _write_figures(directory / "figures", net_by_arm, comparisons, metrics)
    _print_summary(by_arm, comparisons, metrics)
    return metrics


def _write_figures(directory: Path, net_by_arm, comparisons, metrics) -> None:
    headroom_lakh = metrics["headroom_paise"] / 10_000_000

    figures.save_figure(
        figures.recovery_bounds(net_by_arm),
        directory / "recovery_bounds",
        f"Net recovery across {metrics['n_seeds']} paired seeds and "
        f"{metrics['n_mandates']} mandates over {metrics['n_days']} days. "
        "The no-intervention floor is what a merchant collects by doing "
        "nothing; the oracle ceiling is what perfect knowledge of every "
        "balance and outage could collect by retiming alone. The headroom "
        f"between the fixed schedule and the oracle is Rs {headroom_lakh:,.1f} "
        "lakh — that gap, not the raw recovery rate, is what any agent is "
        "competing for.",
    )

    figures.save_figure(
        figures.arm_comparison_bars(
            net_by_arm,
            {
                "fixed_schedule": comparisons["fixed_schedule_vs_do_nothing"],
                "oracle": comparisons["oracle_vs_do_nothing"],
            },
        ),
        directory / "arm_comparison_bars",
        "Net recovery by arm with bootstrap 95% confidence intervals on the "
        "paired per-seed delta against the no-intervention floor. Intervals "
        "come from resampling seeds, not episodes: episodes inside a seed "
        "share a world and are not independent.",
    )

    baseline_vs_floor = comparisons["fixed_schedule_vs_do_nothing"]
    deltas = list(baseline_vs_floor["per_seed_delta_paise"].values())
    figures.save_figure(
        figures.paired_delta_distribution(
            deltas, treatment="fixed_schedule", control="do_nothing"
        ),
        directory / "paired_delta_distribution",
        "Per-seed net recovery delta, fixed schedule minus no intervention. "
        f"The fixed schedule lost on {baseline_vs_floor['n_seeds_lost']} of "
        f"{baseline_vs_floor['n_seeds']} seeds "
        f"({baseline_vs_floor['loss_rate']:.1%}). The mean is one number; "
        "this is the shape behind it, and the losing seeds are shown rather "
        "than averaged away.",
    )


def _print_summary(by_arm, comparisons, metrics) -> None:
    print(
        f"\n{metrics['n_seeds']} seeds x {metrics['n_mandates']} mandates x "
        f"{metrics['n_days']} days\n"
    )
    header = (
        f"{'arm':<18}{'net (Rs lakh)':>15}{'recovery':>11}"
        f"{'att/rec':>9}{'over-int':>10}{'cost/Rs100':>12}"
    )
    print(header)
    print("-" * len(header))
    for arm, metric in by_arm.items():
        print(
            f"{arm:<18}"
            f"{metric['net_recovery_paise'] / 10_000_000:>15,.1f}"
            f"{metric['recovery_rate']:>11.1%}"
            f"{metric['attempts_per_recovery']:>9.2f}"
            f"{metric['over_intervention_rate']:>10.1%}"
            f"{metric['cost_per_100_rupees_recovered']:>12.3f}"
        )

    print(f"\ncomparisons run: {metrics['n_comparisons_run']}")
    for comparison in comparisons.values():
        print("  " + summarise_comparison(comparison))

    print(
        f"\nheadroom (oracle - fixed schedule): Rs "
        f"{metrics['headroom_paise'] / 10_000_000:,.1f} lakh"
    )
    print(
        "fixed schedule captures "
        f"{metrics['baseline_share_of_available_headroom']:.1%} of the "
        "available headroom above doing nothing"
    )


if __name__ == "__main__":
    main()
