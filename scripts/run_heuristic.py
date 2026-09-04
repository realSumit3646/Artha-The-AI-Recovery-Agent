"""The heuristic agent against the same 120 seeds as the baselines.

Runs all four non-LLM arms together — floor, industry baseline, deterministic
agent, oracle ceiling — so the heuristic's position between the baseline and
the ceiling is measured on identical worlds rather than inferred by comparing
two separate runs.

Run it with::

    python scripts/run_heuristic.py

Output lands in ``results/heuristic/``.

The number to watch is not the recovery figure. It is
``unknown_diagnosis_rate``: the share of failures a code book cannot resolve
even with the two contradiction rules applied. That is the honest size of the
gap a model stage would have to fill, and it is the argument for the next
milestone — or the argument against it.
"""

from __future__ import annotations

import sys
from collections import Counter
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
from mandate_recovery.policies import (  # noqa: E402
    DoNothingPolicy,
    FixedSchedulePolicy,
    HeuristicPolicy,
    OraclePolicy,
)

EXPERIMENT_ID = "heuristic"
N_SEEDS = 120
N_MANDATES = 500
N_DAYS = 90
BOOTSTRAP_SEED = 20260904

#: Declared up front so the count is visible in the stored metrics.
COMPARISONS = (
    ("heuristic", "fixed_schedule"),
    ("heuristic", "do_nothing"),
    ("oracle", "heuristic"),
)


def main(results_root: Path | None = None, overwrite: bool = True) -> dict:
    config = ExperimentConfig(
        experiment_id=EXPERIMENT_ID,
        seeds=list(range(N_SEEDS)),
        n_customers=N_MANDATES,
        n_mandates=N_MANDATES,
        n_days=N_DAYS,
    )

    # Keep a handle on every heuristic instance so its diagnosis counters can
    # be aggregated after the run; the runner builds one per seed.
    built: list[HeuristicPolicy] = []

    def heuristic_factory(world, mapping):
        policy = HeuristicPolicy()
        built.append(policy)
        return policy

    arms = {
        "do_nothing": lambda world, mapping: DoNothingPolicy(),
        "fixed_schedule": lambda world, mapping: FixedSchedulePolicy(),
        "heuristic": heuristic_factory,
        "oracle": lambda world, mapping: OraclePolicy(world, mapping),
    }

    episodes, decisions = run_experiment(arms, config)
    episode_frame = pd.DataFrame([e.to_row() for e in episodes])
    decision_frame = pd.DataFrame([d.__dict__ for d in decisions])

    diagnosis_counts: Counter = Counter()
    rejection_counts: Counter = Counter()
    for policy in built:
        diagnosis_counts.update(policy.diagnosis_counts)
        rejection_counts.update(policy.validator.rejections)

    diagnosed = sum(diagnosis_counts.values()) or 1
    unknown_rate = diagnosis_counts.get("UNKNOWN", 0) / diagnosed

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

    net = {arm: by_arm[arm]["net_recovery_paise"] for arm in arms}
    available = net["oracle"] - net["do_nothing"]
    metrics = {
        "n_seeds": N_SEEDS,
        "n_mandates": N_MANDATES,
        "n_days": N_DAYS,
        "n_comparisons_run": len(COMPARISONS),
        "by_arm": by_arm,
        "comparisons": comparisons,
        "diagnosis_counts": dict(diagnosis_counts),
        "unknown_diagnosis_rate": unknown_rate,
        "validator_rejections": dict(rejection_counts),
        "headroom_captured": {
            arm: (net[arm] - net["do_nothing"]) / max(1, available)
            for arm in arms
        },
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
    _write_figures(
        directory / "figures", net, comparisons, decision_frame, metrics
    )
    _print_summary(by_arm, comparisons, metrics)
    return metrics


def _write_figures(directory, net, comparisons, decision_frame, metrics) -> None:
    heuristic_vs_baseline = comparisons["heuristic_vs_fixed_schedule"]
    deltas = list(heuristic_vs_baseline["per_seed_delta_paise"].values())

    figures.save_figure(
        figures.paired_delta_distribution(
            deltas, treatment="heuristic", control="fixed_schedule"
        ),
        directory / "heuristic_vs_baseline_paired",
        "Per-seed net recovery delta, heuristic agent minus fixed schedule, "
        f"across {metrics['n_seeds']} paired seeds. The agent lost on "
        f"{heuristic_vs_baseline['n_seeds_lost']} of "
        f"{heuristic_vs_baseline['n_seeds']} seeds "
        f"({heuristic_vs_baseline['loss_rate']:.1%}). Losing seeds are shown "
        "rather than averaged away.",
    )

    figures.save_figure(
        figures.diagnosis_coverage(metrics["diagnosis_counts"]),
        directory / "diagnosis_coverage",
        "What a rule-based code book can and cannot resolve. "
        f"{metrics['unknown_diagnosis_rate']:.1%} of failures return UNKNOWN "
        "— a generic code, a missing code, or a funds code the customer's "
        "payment history cannot separate from a ceiling breach. That share is "
        "the honest size of the gap a model stage would have to fill.",
    )

    counts_by_arm = {
        str(arm): dict(group["action_kind"].value_counts())
        for arm, group in decision_frame.groupby("arm", sort=True)
    }
    figures.save_figure(
        figures.intervention_mix(counts_by_arm),
        directory / "intervention_mix",
        "The share of each arm's decisions by action type. The fixed schedule "
        "can only retry or stop; the heuristic agent also collects partial "
        "amounts against a known ceiling and contacts the customer when "
        "silent retries have failed.",
    )

    figures.save_figure(
        figures.recovery_bounds(net),
        directory / "all_arms_bounds",
        "Net recovery across four arms on identical worlds. The heuristic "
        "agent sits between the industry baseline and the perfect-information "
        "ceiling; the remaining gap is what a model stage would have to earn.",
    )


def _print_summary(by_arm, comparisons, metrics) -> None:
    print(
        f"\n{metrics['n_seeds']} seeds x {metrics['n_mandates']} mandates x "
        f"{metrics['n_days']} days\n"
    )
    header = (
        f"{'arm':<18}{'net (Rs lakh)':>15}{'recovery':>11}{'att/rec':>9}"
        f"{'contacts/rec':>14}{'over-int':>10}{'headroom':>10}"
    )
    print(header)
    print("-" * len(header))
    for arm, metric in by_arm.items():
        print(
            f"{arm:<18}"
            f"{metric['net_recovery_paise'] / 10_000_000:>15,.1f}"
            f"{metric['recovery_rate']:>11.1%}"
            f"{metric['attempts_per_recovery']:>9.2f}"
            f"{metric['contacts_per_recovery']:>14.3f}"
            f"{metric['over_intervention_rate']:>10.1%}"
            f"{metrics['headroom_captured'][arm]:>10.1%}"
        )

    print(f"\ncomparisons run: {metrics['n_comparisons_run']}")
    for comparison in comparisons.values():
        print("  " + summarise_comparison(comparison))

    print(f"\ndiagnosis coverage over {sum(metrics['diagnosis_counts'].values()):,} failures:")
    for name, count in sorted(
        metrics["diagnosis_counts"].items(), key=lambda kv: -kv[1]
    ):
        share = count / sum(metrics["diagnosis_counts"].values())
        print(f"  {name:<22}{count:>10,}{share:>8.1%}")
    print(f"\nUNKNOWN diagnosis rate: {metrics['unknown_diagnosis_rate']:.1%}")

    if metrics["validator_rejections"]:
        print("\ncompliance gate refusals:")
        for rule, count in sorted(
            metrics["validator_rejections"].items(), key=lambda kv: -kv[1]
        ):
            print(f"  {rule:<28}{count:>10,}")


if __name__ == "__main__":
    main()
