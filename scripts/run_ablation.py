"""The deciding experiment: heuristic agent against LLM agent, head to head.

Five arms on the same 120 paired seeds as every earlier run — floor, industry
baseline, deterministic agent, LLM agent, oracle ceiling. The comparison that
matters is `llm_agent` against `heuristic`, because both use the same
scheduler, the same compliance validator and the same cost model. The only
difference between them is whether a language model resolves the residual
diagnoses and chooses the intervention.

Run it with::

    python scripts/run_ablation.py            # live, warms the cache
    python scripts/run_ablation.py --offline  # cache only, no API key needed

The first live run costs a few hundred model calls; everything after is served
from `llm_cache/`, which is committed. `--offline` is what `make reproduce`
uses, so a reviewer with no key gets the same numbers.

On reporting
------------
Whatever this run says is what `docs/ABLATION.md` reports, in the first line,
including the loss rate. If the heuristic wins, that is the finding: recovery
is a scheduling problem and the model belongs only in residual diagnosis. The
one thing that must not happen is re-running with a tuned prompt until the
answer changes.
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
from mandate_recovery.llm.client import DEFAULT_MODEL, LLMClient  # noqa: E402
from mandate_recovery.policies import (  # noqa: E402
    DoNothingPolicy,
    FixedSchedulePolicy,
    HeuristicPolicy,
    LLMAgentPolicy,
    OraclePolicy,
)

EXPERIMENT_ID = "ablation"
N_SEEDS = 120
N_MANDATES = 500
N_DAYS = 90
BOOTSTRAP_SEED = 20260904

#: Declared up front. Four comparisons, and the count goes into metrics.json
#: so a reader knows how many were run rather than how many were reported.
COMPARISONS = (
    ("llm_agent", "heuristic"),
    ("llm_agent", "fixed_schedule"),
    ("heuristic", "fixed_schedule"),
    ("oracle", "llm_agent"),
)


def main(
    results_root: Path | None = None,
    overwrite: bool = True,
    offline: bool = False,
) -> dict:
    config = ExperimentConfig(
        experiment_id=EXPERIMENT_ID,
        seeds=list(range(N_SEEDS)),
        n_customers=N_MANDATES,
        n_mandates=N_MANDATES,
        n_days=N_DAYS,
    )

    # One client across every seed so the cache is shared and warms once.
    client = LLMClient(cache_dir=REPO_ROOT / "llm_cache", offline=offline)
    built: list[LLMAgentPolicy] = []
    heuristics: list[HeuristicPolicy] = []

    def llm_factory(world, mapping):
        policy = LLMAgentPolicy(client)
        built.append(policy)
        return policy

    def heuristic_factory(world, mapping):
        policy = HeuristicPolicy()
        heuristics.append(policy)
        return policy

    arms = {
        "do_nothing": lambda world, mapping: DoNothingPolicy(),
        "fixed_schedule": lambda world, mapping: FixedSchedulePolicy(),
        "heuristic": heuristic_factory,
        "llm_agent": llm_factory,
        "oracle": lambda world, mapping: OraclePolicy(world, mapping),
    }

    episodes, decisions = run_experiment(arms, config)
    episode_frame = pd.DataFrame([e.to_row() for e in episodes])
    decision_frame = pd.DataFrame([d.__dict__ for d in decisions])

    agent_stats = _aggregate_agent_stats(built)
    heuristic_diagnoses: Counter = Counter()
    for policy in heuristics:
        heuristic_diagnoses.update(policy.diagnosis_counts)

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
        "model": DEFAULT_MODEL,
        "offline": offline,
        "by_arm": by_arm,
        "comparisons": comparisons,
        "llm": agent_stats,
        "client": client.stats(),
        "heuristic_diagnosis_counts": dict(heuristic_diagnoses),
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
    _write_figures(directory / "figures", net, comparisons, metrics)
    _print_summary(by_arm, comparisons, metrics)
    return metrics


def _aggregate_agent_stats(agents) -> dict:
    """Sum the per-seed agent counters into one picture of the model layer."""
    totals: Counter = Counter()
    stages: Counter = Counter()
    rejections: Counter = Counter()
    for agent in agents:
        stats = agent.stats()
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key.endswith("_rate"):
                    continue
                totals[key] += value
        stages.update(stats.get("fallbacks_by_stage", {}))
        rejections.update(stats.get("validator_rejections", {}))

    diagnoses = totals.get("diagnoses", 0) or 1
    decisions = totals.get("decisions", 0) or 1
    return {
        **dict(totals),
        "fallbacks_by_stage": dict(stages),
        "validator_rejections": dict(rejections),
        "llm_invocation_rate": totals.get("llm_invoked", 0) / diagnoses,
        "residual_resolution_rate": (
            totals.get("llm_resolved", 0) / (totals.get("llm_invoked", 0) or 1)
        ),
        "stage_fallbacks": sum(stages.values()),
        "fallback_rate": sum(stages.values()) / decisions,
    }


def _write_figures(directory: Path, net, comparisons, metrics) -> None:
    llm = metrics["llm"]
    head_to_head = comparisons["llm_agent_vs_heuristic"]

    figures.save_figure(
        figures.recovery_bounds(net),
        directory / "all_arms",
        "Net recovery across five arms on identical worlds, "
        f"{metrics['n_seeds']} paired seeds. Floor, industry baseline, "
        "deterministic agent, LLM agent, and the perfect-information ceiling.",
    )

    figures.save_figure(
        figures.paired_delta_distribution(
            list(head_to_head["per_seed_delta_paise"].values()),
            treatment="llm_agent",
            control="heuristic",
        ),
        directory / "heuristic_vs_llm_paired",
        "The deciding comparison: per-seed net recovery delta, LLM agent minus "
        f"heuristic agent. The LLM agent lost on {head_to_head['n_seeds_lost']} "
        f"of {head_to_head['n_seeds']} seeds "
        f"({head_to_head['loss_rate']:.1%}). Both arms share the same "
        "scheduler, validator and cost model; the only difference is whether a "
        "model resolves residual diagnoses and picks the intervention.",
    )

    figures.save_figure(
        figures.llm_invocation_rate(
            int(llm.get("rule_resolved", 0)),
            int(llm.get("llm_invoked", 0)),
            int(llm.get("llm_resolved", 0)),
        ),
        directory / "llm_invocation_rate",
        f"The model was called on {llm['llm_invocation_rate']:.1%} of "
        "diagnosed failures — only those a rule-based code book could not "
        "settle. The rest were resolved deterministically. Narrow use is the "
        "evidence for the judgement claim, not the fact that a model appears.",
    )

    reliability = {
        **metrics["client"],
        "stage_fallbacks": llm.get("stage_fallbacks", 0),
        "message_fallbacks": llm.get("message_fallbacks", 0),
    }
    figures.save_figure(
        figures.llm_reliability(reliability),
        directory / "llm_reliability",
        "What the model layer actually did across the whole experiment: cache "
        "hits against live calls, and every schema failure, transport failure "
        "and fallback. A cached run reproduces these numbers without an API "
        "key.",
    )


def _print_summary(by_arm, comparisons, metrics) -> None:
    llm = metrics["llm"]
    print(
        f"\n{metrics['n_seeds']} seeds x {metrics['n_mandates']} mandates x "
        f"{metrics['n_days']} days   model={metrics['model']}\n"
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

    print("\nmodel layer:")
    print(f"  llm invocation rate      {llm['llm_invocation_rate']:>10.1%}")
    print(f"  residual resolved        {llm['residual_resolution_rate']:>10.1%}")
    print(f"  stage fallback rate      {llm['fallback_rate']:>10.2%}")
    print(f"  live calls               {metrics['client']['calls_made']:>10,}")
    print(f"  cache hits               {metrics['client']['cache_hits']:>10,}")
    print(f"  cache entries            {metrics['client']['cache']['entries']:>10,}")
    print(f"  schema failures          {metrics['client']['schema_failures']:>10,}")
    print(f"  total tokens             {metrics['client']['total_tokens']:>10,}")
    if llm.get("fallbacks_by_stage"):
        print(f"  fallbacks by stage       {llm['fallbacks_by_stage']}")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
