"""Shared plotting for every figure this project produces.

One style, applied at import, so figures from different scripts sit together
without looking like they came from different projects. Every figure is saved
three ways by :func:`save_figure`: a 300-dpi PNG, an SVG, and a ``.txt``
caption alongside them. The captions are written to be reused verbatim in the
README and the pitch video, so they say what the figure shows and what to
conclude, not "Figure 3".

Design rules, enforced by habit rather than by code:

* No chartjunk. No gridlines competing with the data, no 3-D, no shadows.
* Readable at video resolution -- large type, few series, direct labels where
  they fit.
* Colourblind-safe throughout: the palette is Okabe-Ito, which stays legible
  under deuteranopia and protanopia and survives greyscale printing.
* Every axis carries units. A number without a unit is not a result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # figures are files, never windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "PALETTE",
    "OUTCOME_COLOURS",
    "set_style",
    "save_figure",
    "failure_mode_breakdown",
    "failure_rate_by_hour",
    "balance_trajectory_sample",
    "observed_vs_calibrated",
    "recovery_bounds",
    "arm_comparison_bars",
    "paired_delta_distribution",
    "diagnosis_coverage",
    "intervention_mix",
    "llm_invocation_rate",
    "llm_reliability",
]


#: Okabe-Ito: eight hues distinguishable under the common colour vision
#: deficiencies, and distinguishable from each other in greyscale.
PALETTE: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
)

#: Stable colour per failure mode, so the same cause is the same colour in
#: every figure in the project.
OUTCOME_COLOURS: Final[Mapping[str, str]] = {
    "INSUFFICIENT_FUNDS": PALETTE[0],
    "TECHNICAL_DECLINE": PALETTE[1],
    "LIMIT_EXCEEDED": PALETTE[2],
    "WINDOW_REJECTED": PALETTE[3],
    "MANDATE_REVOKED": PALETTE[7],
    "SUCCESS": PALETTE[5],
}

_INK: Final = "#222222"
_MUTED: Final = "#666666"


def set_style() -> None:
    """Apply the project style. Called once at import."""
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 5.5),
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelcolor": _INK,
            "axes.edgecolor": _MUTED,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
            "xtick.color": _INK,
            "ytick.color": _INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 11,
            "lines.linewidth": 2.0,
            "text.color": _INK,
        }
    )


set_style()


def save_figure(figure: plt.Figure, path_stem: Path, caption: str) -> dict[str, Path]:
    """Save a figure as PNG (300 dpi), SVG, and a caption text file.

    ``path_stem`` is a path without an extension; the three files are written
    beside each other. Returns the paths written, keyed by extension.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {})):
        target = path_stem.with_suffix(f".{suffix}")
        figure.savefig(target, **kwargs)
        written[suffix] = target

    caption_path = path_stem.with_suffix(".txt")
    caption_path.write_text(caption.strip() + "\n", encoding="utf-8")
    written["txt"] = caption_path

    plt.close(figure)
    return written


def _rupees(paise: float) -> float:
    return paise / 100.0


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def failure_mode_breakdown(shares: Mapping[str, float]) -> plt.Figure:
    """Horizontal bars: what share of failures each cause accounts for."""
    ordered = sorted(shares.items(), key=lambda item: item[1])
    labels = [name.replace("_", " ").title() for name, _ in ordered]
    values = [share for _, share in ordered]
    colours = [OUTCOME_COLOURS.get(name, PALETTE[7]) for name, _ in ordered]

    figure, axes = plt.subplots()
    bars = axes.barh(labels, values, color=colours, height=0.62)
    for bar, value in zip(bars, values):
        axes.text(
            value + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=11,
            color=_INK,
        )

    axes.set_xlabel("Share of all failed attempts")
    axes.set_title("What actually goes wrong")
    axes.set_xlim(0, max(values) * 1.18 if values else 1)
    axes.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes.grid(axis="y", visible=False)
    return figure


def failure_rate_by_hour(
    attempts_by_hour: Sequence[int],
    failures_by_hour: Sequence[int],
    restricted_windows: Iterable[tuple[int, int]],
) -> plt.Figure:
    """Failure rate against hour of day, with the restricted window shaded.

    The restricted window should be visible as a step up in the failure rate;
    if it is not, the window is not doing anything.
    """
    hours = np.arange(len(attempts_by_hour))
    attempts = np.asarray(attempts_by_hour, dtype=float)
    failures = np.asarray(failures_by_hour, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(attempts > 0, failures / attempts, np.nan)

    figure, axes = plt.subplots()
    for index, (start, end) in enumerate(restricted_windows):
        axes.axvspan(
            start - 0.5,
            end - 0.5,
            color=PALETTE[3],
            alpha=0.13,
            label="NPCI restricted window" if index == 0 else None,
        )
    axes.plot(hours, rate, color=PALETTE[0], marker="o", markersize=4)

    axes.set_xlabel("Hour of day (local, 24h)")
    axes.set_ylabel("Failure rate (share of attempts)")
    axes.set_title("Failures cluster in the restricted window")
    axes.set_xticks(range(0, 24, 2))
    axes.set_xlim(-0.5, 23.5)
    axes.set_ylim(0, np.nanmax(rate) * 1.25 if np.isfinite(rate).any() else 1)
    axes.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axes.legend(loc="upper left")
    return figure


def balance_trajectory_sample(
    trajectories_paise: Sequence[Sequence[int]],
    salary_days: Sequence[int] | None = None,
) -> plt.Figure:
    """A handful of customer balance trajectories over the run."""
    figure, axes = plt.subplots()
    for index, trajectory in enumerate(trajectories_paise):
        label = None
        if salary_days is not None and index < len(salary_days):
            label = f"paid on day {salary_days[index]}"
        axes.plot(
            [_rupees(value) for value in trajectory],
            color=PALETTE[index % len(PALETTE)],
            label=label,
            alpha=0.9,
        )

    axes.set_xlabel("Simulation day")
    axes.set_ylabel("Balance (rupees)")
    axes.set_title("Balances sawtooth around the salary cycle")
    axes.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
    if salary_days is not None:
        axes.legend(loc="upper right", ncol=2)
    return figure


def observed_vs_calibrated(
    observed: Mapping[str, float],
    calibrated: Mapping[str, float],
    tolerance: Mapping[str, float] | None = None,
) -> plt.Figure:
    """Side-by-side bars: what the calibration claims vs what the sim produced.

    This is the figure that says whether the simulator is honest about itself.
    """
    names = list(calibrated)
    labels = [name.replace("_", " ").replace("share ", "").title() for name in names]
    calibrated_values = [calibrated[name] for name in names]
    observed_values = [observed.get(name, float("nan")) for name in names]

    positions = np.arange(len(names))
    width = 0.38

    figure, axes = plt.subplots()
    axes.bar(
        positions - width / 2,
        calibrated_values,
        width,
        label="Calibrated target",
        color=PALETTE[7],
    )
    axes.bar(
        positions + width / 2,
        observed_values,
        width,
        label="Observed in simulator",
        color=PALETTE[0],
    )

    if tolerance:
        for index, name in enumerate(names):
            band = tolerance.get(name)
            if band is None:
                continue
            axes.errorbar(
                positions[index] - width / 2,
                calibrated[name],
                yerr=band,
                fmt="none",
                ecolor=_MUTED,
                capsize=5,
                linewidth=1.2,
            )

    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=20, ha="right")
    axes.set_ylabel("Share of attempts / of failures")
    axes.set_title("Observed distributions against their calibrated targets")
    axes.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axes.legend()
    axes.grid(axis="x", visible=False)
    return figure


# --------------------------------------------------------------------------
# Experiment figures
# --------------------------------------------------------------------------

#: Stable colour per arm, so an arm looks the same in every figure.
ARM_COLOURS: Final[Mapping[str, str]] = {
    "do_nothing": PALETTE[7],
    "fixed_schedule": PALETTE[1],
    "heuristic": PALETTE[2],
    "llm_agent": PALETTE[0],
    "oracle": PALETTE[4],
}

ARM_LABELS: Final[Mapping[str, str]] = {
    "do_nothing": "No intervention",
    "fixed_schedule": "Fixed schedule",
    "heuristic": "Heuristic agent",
    "llm_agent": "LLM agent",
    "oracle": "Oracle (perfect info)",
}


def _arm_label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm.replace("_", " ").title())


def _arm_colour(arm: str) -> str:
    return ARM_COLOURS.get(arm, PALETTE[5])


def recovery_bounds(
    net_recovery_paise_by_arm: Mapping[str, float],
    *,
    floor_arm: str = "do_nothing",
    ceiling_arm: str = "oracle",
    baseline_arm: str = "fixed_schedule",
) -> plt.Figure:
    """Floor, baseline and ceiling, with the headroom between them named.

    The figure that frames the whole project: it says how much of the lost
    money is recoverable at all, and how much of that the industry default
    already gets.
    """
    order = [
        arm
        for arm in (floor_arm, baseline_arm, ceiling_arm)
        if arm in net_recovery_paise_by_arm
    ]
    order += [arm for arm in net_recovery_paise_by_arm if arm not in order]
    values = [net_recovery_paise_by_arm[arm] / 10_000_000 for arm in order]

    figure, axes = plt.subplots(figsize=(9.0, 4.6))
    bars = axes.barh(
        [_arm_label(arm) for arm in order],
        values,
        color=[_arm_colour(arm) for arm in order],
        height=0.6,
    )
    for bar, value in zip(bars, values):
        axes.text(
            value + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"Rs {value:,.1f}L",
            va="center",
            fontsize=11,
        )

    if baseline_arm in net_recovery_paise_by_arm and (
        ceiling_arm in net_recovery_paise_by_arm
    ):
        baseline = net_recovery_paise_by_arm[baseline_arm] / 10_000_000
        ceiling = net_recovery_paise_by_arm[ceiling_arm] / 10_000_000
        # Sit the arrow between the two bars it describes, not above them.
        span_y = (order.index(baseline_arm) + order.index(ceiling_arm)) / 2.0
        axes.annotate(
            "",
            xy=(ceiling, span_y),
            xytext=(baseline, span_y),
            arrowprops=dict(arrowstyle="<->", color=_INK, linewidth=1.6),
        )
        axes.text(
            (baseline + ceiling) / 2,
            span_y + 0.17,
            f"headroom Rs {ceiling - baseline:,.1f}L",
            ha="center",
            fontsize=11,
            color=_INK,
        )

    axes.set_xlabel("Net recovery (lakh rupees, summed across seeds)")
    axes.set_title("How much of the lost money is recoverable at all")
    axes.set_xlim(0, max(values) * 1.20)
    axes.grid(axis="y", visible=False)
    return figure


def arm_comparison_bars(
    net_recovery_paise_by_arm: Mapping[str, float],
    comparisons: Mapping[str, Mapping[str, float]] | None = None,
) -> plt.Figure:
    """Net recovery by arm, with bootstrap CIs on the paired deltas."""
    arms = list(net_recovery_paise_by_arm)
    values = [net_recovery_paise_by_arm[arm] / 10_000_000 for arm in arms]

    figure, axes = plt.subplots()
    bars = axes.bar(
        [_arm_label(arm) for arm in arms],
        values,
        color=[_arm_colour(arm) for arm in arms],
        width=0.6,
    )

    if comparisons:
        for bar, arm in zip(bars, arms):
            comparison = comparisons.get(arm)
            if not comparison:
                continue
            low = comparison["ci_low_paise"] / 10_000_000
            high = comparison["ci_high_paise"] / 10_000_000
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                "\n".join(
                    [
                        "vs baseline",
                        "95% CI",
                        f"{low:+,.1f}L to {high:+,.1f}L",
                    ]
                ),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
            )

    axes.set_ylabel("Net recovery (lakh rupees)")
    axes.set_title("Net recovery by arm")
    axes.grid(axis="x", visible=False)
    plt.setp(axes.get_xticklabels(), rotation=12, ha="right")
    return figure


def paired_delta_distribution(
    deltas_paise: Sequence[float],
    *,
    treatment: str = "treatment",
    control: str = "control",
) -> plt.Figure:
    """Per-seed deltas, with the losing seeds shown rather than averaged away.

    The mean is one number; this is the shape behind it. Seeds below zero are
    coloured separately because that is the loss rate made visible.
    """
    deltas = np.asarray(list(deltas_paise), dtype=float) / 100_000.0
    losses = deltas < 0

    figure, axes = plt.subplots()
    axes.hist(
        deltas[~losses],
        bins=24,
        color=PALETTE[2],
        label=f"won ({(~losses).sum()} seeds)",
    )
    if losses.any():
        axes.hist(
            deltas[losses],
            bins=12,
            color=PALETTE[3],
            label=f"lost ({losses.sum()} seeds)",
        )
    axes.axvline(0, color=_MUTED, linewidth=1.2)
    axes.axvline(
        deltas.mean(),
        color=_INK,
        linestyle="--",
        linewidth=1.6,
        label=f"mean {deltas.mean():+,.0f}k",
    )

    axes.set_xlabel("Per-seed net recovery delta (thousand rupees)")
    axes.set_ylabel("Seeds")
    axes.set_title(f"{_arm_label(treatment)} minus {_arm_label(control)}, by seed")
    axes.legend()
    axes.grid(axis="x", visible=False)
    return figure


DIAGNOSIS_COLOURS: Final[Mapping[str, str]] = {
    "INSUFFICIENT_FUNDS": PALETTE[0],
    "TECHNICAL": PALETTE[1],
    "LIMIT": PALETTE[2],
    "WINDOW": PALETTE[3],
    "REVOKED": PALETTE[5],
    "UNKNOWN": PALETTE[7],
}

ACTION_COLOURS: Final[Mapping[str, str]] = {
    "retry_silent": PALETTE[0],
    "send_nudge": PALETTE[1],
    "collect_partial": PALETTE[2],
    "switch_rail": PALETTE[5],
    "escalate_human": PALETTE[4],
    "stop": PALETTE[7],
}


def diagnosis_coverage(counts: Mapping[str, int]) -> plt.Figure:
    """How many failures the code book resolved, and how many it could not.

    The UNKNOWN share is the argument for a model stage, so it is drawn last,
    in grey, and labelled with its percentage rather than left to be eyeballed.
    """
    total = sum(counts.values()) or 1
    resolved = {k: v for k, v in counts.items() if k != "UNKNOWN"}
    ordered = sorted(resolved.items(), key=lambda item: item[1])
    ordered.append(("UNKNOWN", counts.get("UNKNOWN", 0)))

    labels = [name.replace("_", " ").title() for name, _ in ordered]
    values = [count / total for _, count in ordered]
    colours = [DIAGNOSIS_COLOURS.get(name, PALETTE[7]) for name, _ in ordered]

    figure, axes = plt.subplots(figsize=(9.0, 4.8))
    bars = axes.barh(labels, values, color=colours, height=0.62)
    for bar, value in zip(bars, values):
        axes.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=11,
        )

    unknown_share = counts.get("UNKNOWN", 0) / total
    axes.set_xlabel("Share of diagnosed failures")
    axes.set_title(
        f"A code book resolves {1 - unknown_share:.0%} of failures"
    )
    axes.set_xlim(0, max(values) * 1.18)
    axes.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes.grid(axis="y", visible=False)
    return figure


def intervention_mix(counts_by_arm: Mapping[str, Mapping[str, int]]) -> plt.Figure:
    """What each arm actually did, as a share of its decisions."""
    arms = list(counts_by_arm)
    kinds: list[str] = []
    for counts in counts_by_arm.values():
        for kind in counts:
            if kind not in kinds:
                kinds.append(kind)
    kinds.sort(key=lambda k: -sum(c.get(k, 0) for c in counts_by_arm.values()))

    figure, axes = plt.subplots(figsize=(9.0, 4.8))
    bottoms = np.zeros(len(arms))
    for kind in kinds:
        shares = []
        for arm in arms:
            counts = counts_by_arm[arm]
            total = sum(counts.values()) or 1
            shares.append(counts.get(kind, 0) / total)
        shares_array = np.asarray(shares)
        axes.bar(
            [_arm_label(arm) for arm in arms],
            shares_array,
            bottom=bottoms,
            color=ACTION_COLOURS.get(kind, PALETTE[6]),
            label=kind.replace("_", " "),
            width=0.55,
        )
        bottoms += shares_array

    axes.set_ylabel("Share of decisions")
    axes.set_title("What each arm actually did")
    axes.set_ylim(0, 1)
    axes.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    axes.grid(axis="x", visible=False)
    return figure


def llm_invocation_rate(
    rule_resolved: int, llm_invoked: int, llm_resolved: int
) -> plt.Figure:
    """How much of the work the model actually did.

    The evidence for the AI-judgement claim is not that a model was used; it
    is that it was used *narrowly*, on the events rules could not settle.
    """
    total = rule_resolved + llm_invoked or 1
    unresolved = llm_invoked - llm_resolved
    segments = [
        ("Resolved by rules", rule_resolved, PALETTE[7]),
        ("Resolved by the model", llm_resolved, PALETTE[0]),
        ("Still undetermined", unresolved, PALETTE[3]),
    ]

    figure, axes = plt.subplots(figsize=(9.0, 2.9))
    left = 0.0
    for label, value, colour in segments:
        share = value / total
        if share <= 0:
            continue
        axes.barh([""], [share], left=left, color=colour, height=0.55, label=label)
        if share > 0.04:
            axes.text(
                left + share / 2,
                0,
                f"{share:.0%}",
                ha="center",
                va="center",
                color="white" if colour != PALETTE[7] else _INK,
                fontsize=12,
                fontweight="bold",
            )
        left += share

    axes.set_xlim(0, 1)
    axes.set_xlabel("Share of all diagnosed failures")
    axes.set_title(
        f"The model was called on {llm_invoked / total:.0%} of failures"
    )
    axes.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes.set_yticks([])
    axes.grid(visible=False)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    return figure


def llm_reliability(counters: Mapping[str, float]) -> plt.Figure:
    """Schema failures, fallbacks and cache hits — what actually went wrong."""
    rows = [
        ("Cache hits", counters.get("cache_hits", 0), PALETTE[2]),
        ("Live calls", counters.get("calls_made", 0), PALETTE[0]),
        ("Retries", counters.get("retries", 0), PALETTE[1]),
        ("Schema failures", counters.get("schema_failures", 0), PALETTE[3]),
        ("Transport failures", counters.get("transport_failures", 0), PALETTE[4]),
        ("Stage fallbacks", counters.get("stage_fallbacks", 0), PALETTE[7]),
        ("Message fallbacks", counters.get("message_fallbacks", 0), PALETTE[5]),
    ]
    rows = [row for row in rows if row[1] > 0] or rows[:2]
    rows.sort(key=lambda row: row[1])

    figure, axes = plt.subplots(figsize=(9.0, 4.6))
    bars = axes.barh(
        [name for name, _, _ in rows],
        [value for _, value, _ in rows],
        color=[colour for _, _, colour in rows],
        height=0.62,
    )
    largest = max(value for _, value, _ in rows) or 1
    for bar, (_, value, _) in zip(bars, rows):
        axes.text(
            value + largest * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
            fontsize=11,
        )

    axes.set_xlabel("Events across the whole experiment (log scale)")
    axes.set_xscale("symlog")
    axes.set_title("Model layer reliability")
    axes.grid(axis="y", visible=False)
    return figure
