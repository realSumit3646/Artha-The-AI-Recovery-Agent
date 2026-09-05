"""Reproduce every experiment in this project, from stored configuration.

    python scripts/reproduce_all.py

No API key is required. The experiments that carry results use no model at
all, and the model layer runs from the committed cache in ``llm_cache/``.

What it runs, in order:

1. Simulator validation — does the world produce the distributions the
   calibration claims?
2. Baselines — floor, industry schedule, oracle ceiling.
3. Heuristic — the deterministic agent against all three.
4. Sensitivity — 54 regimes, does the advantage survive?

The ablation is **skipped by default** and says why. Its cache is not warm
enough to run it offline, and running it against a cold cache would produce an
LLM arm that had silently fallen back to the heuristic on every decision — a
result that looks like a null finding rather than an absent one. Pass
``--with-ablation`` to attempt it anyway once the cache has been warmed with a
key.

Takes roughly seven minutes on a laptop, most of it the sensitivity sweep.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: Below this many cached responses the ablation cannot run meaningfully.
#: A 10-seed measurement needed 198 distinct prompts; the full run needs more.
MINIMUM_CACHE_ENTRIES_FOR_ABLATION = 150

STEPS = (
    ("simulator validation", "validate_simulator.py", "results/simulator_validation"),
    ("baselines", "run_baselines.py", "results/baselines"),
    ("heuristic", "run_heuristic.py", "results/heuristic"),
    ("sensitivity", "run_sensitivity.py", "results/sensitivity"),
)


def _load(script_name: str):
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _cache_entries() -> int:
    cache = REPO_ROOT / "llm_cache"
    return sum(1 for _ in cache.rglob("*.json")) if cache.exists() else 0


def _rupees_lakh(paise: float) -> str:
    return f"Rs {paise / 10_000_000:,.1f} L"


def main(with_ablation: bool = False) -> int:
    started = time.time()
    print("=" * 74)
    print("Reproducing every experiment in this project. No API key required.")
    print("=" * 74)

    for label, script, output in STEPS:
        step_started = time.time()
        print(f"\n>>> {label} ({script})")
        module = _load(script)
        if script == "validate_simulator.py":
            module.main()
        else:
            module.main(overwrite=True)
        print(f"    done in {time.time() - step_started:.0f}s -> {output}/")

    entries = _cache_entries()
    if with_ablation:
        if entries < MINIMUM_CACHE_ENTRIES_FOR_ABLATION:
            print(
                f"\n>>> ablation SKIPPED: only {entries} cached responses "
                f"(need ~{MINIMUM_CACHE_ENTRIES_FOR_ABLATION}). Running it now "
                "would produce an LLM arm that fell back to the heuristic on "
                "every decision. See docs/ABLATION.md."
            )
        else:
            print("\n>>> ablation (run_ablation.py, offline)")
            _load("run_ablation.py").main(overwrite=True, offline=True)
    else:
        print(
            f"\n>>> ablation not attempted (cache has {entries} responses; "
            "pass --with-ablation once it is warm). See docs/ABLATION.md."
        )

    _print_summary()
    print(f"\nTotal: {time.time() - started:.0f}s")
    return 0


def _read(path: str) -> dict | None:
    file = REPO_ROOT / path / "metrics.json"
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def _print_summary() -> None:
    print("\n" + "=" * 74)
    print("HEADLINE METRICS")
    print("=" * 74)

    validation = _read("results/simulator_validation")
    if validation:
        observed = validation["observed"]["failure_rate"]
        target = validation["calibrated"]["failure_rate"]
        print(
            f"\nSimulator validation: observed failure rate {observed:.1%} "
            f"against a calibrated target of {target:.0%} "
            f"(delta {observed - target:+.2%}, tolerance 2 points)"
        )

    heuristic = _read("results/heuristic")
    if heuristic:
        print(f"\n{'arm':<18}{'net recovery':>16}{'recovery':>11}{'headroom':>11}")
        print("-" * 56)
        for arm, metric in heuristic["by_arm"].items():
            print(
                f"{arm:<18}{_rupees_lakh(metric['net_recovery_paise']):>16}"
                f"{metric['recovery_rate']:>11.1%}"
                f"{heuristic['headroom_captured'][arm]:>11.1%}"
            )
        comparison = heuristic["comparisons"]["heuristic_vs_fixed_schedule"]
        print(
            f"\nHeuristic vs fixed schedule: mean delta Rs "
            f"{comparison['mean_delta_paise'] / 100:,.0f} "
            f"(95% CI Rs {comparison['ci_low_paise'] / 100:,.0f} to "
            f"Rs {comparison['ci_high_paise'] / 100:,.0f}), "
            f"lost on {comparison['n_seeds_lost']}/{comparison['n_seeds']} seeds "
            f"({comparison['loss_rate']:.1%})"
        )
        print(
            f"Rule-based diagnosis returned UNKNOWN on "
            f"{heuristic['unknown_diagnosis_rate']:.1%} of failures"
        )

    sensitivity = _read("results/sensitivity")
    if sensitivity:
        print(
            f"\nSensitivity: the advantage holds in "
            f"{sensitivity['cells_where_advantage_holds']}/"
            f"{sensitivity['n_cells']} regimes "
            f"({sensitivity['advantage_survival_rate']:.0%}); "
            f"{sensitivity['cells_with_ci_excluding_zero']} cells have a CI "
            "excluding zero"
        )
        worst = sensitivity["worst_cell"]
        print(
            f"  worst regime: {worst['failure_regime']}/"
            f"{worst['funds_share_regime']}/{worst['window_regime']}/"
            f"{worst['availability_regime']} at "
            f"Rs {worst['mean_delta_paise'] / 100:,.0f} per seed, "
            f"losing on {worst['loss_rate']:.0%}"
        )

    ablation = _read("results/ablation")
    print(
        "\nAblation: "
        + (
            "see results/ablation/metrics.json"
            if ablation
            else "NOT RUN - blocked on API quota, see docs/ABLATION.md"
        )
    )
    print(
        "\nEvery number above is conditional on an entirely unsourced "
        "calibration. See docs/LIMITATIONS.md."
    )


if __name__ == "__main__":
    raise SystemExit(main(with_ablation="--with-ablation" in sys.argv))
