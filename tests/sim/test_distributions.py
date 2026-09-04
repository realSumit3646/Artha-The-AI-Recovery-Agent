"""Does the simulator actually produce what the calibration claims?

A calibration file is a promise. These tests are the only thing that keeps it
honest: they run the null world -- no policy, every mandate presented once per
cycle -- and check the emergent failure mix against the calibrated targets
within the tolerances stated in ``docs/CALIBRATION.md``.

Agreement here is *not* evidence the parameters are right. They are
placeholders. It is evidence only that the simulator does what its own
configuration says it does, which is the weaker claim this project is entitled
to make.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mandate_recovery import figures  # noqa: E402
from mandate_recovery.calibration import DEFAULT_CALIBRATION  # noqa: E402
from mandate_recovery.sim.world import HOURS_IN_DAY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_simulator.py"

#: Tolerances from the task and from docs/CALIBRATION.md.
FAILURE_RATE_TOLERANCE = 0.02
MODE_SHARE_TOLERANCE = 0.03

#: Fewer seeds than the full validation run, enough for a stable estimate.
TEST_SEEDS = 15


def _load_script():
    """Import the validation script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("validate_simulator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_simulator"] = module
    spec.loader.exec_module(module)
    return module


validate_simulator = _load_script()


@pytest.fixture(scope="module")
def observed():
    """One null-world run, shared by every test in this module."""
    results = [
        validate_simulator.simulate_seed(seed) for seed in range(TEST_SEEDS)
    ]
    return validate_simulator.aggregate(results), results


# --------------------------------------------------------------------------
# The calibration promise
# --------------------------------------------------------------------------


def test_observed_failure_rate_matches_the_calibrated_target(observed):
    metrics, _ = observed
    target = metrics["calibrated"]["failure_rate"]
    actual = metrics["observed"]["failure_rate"]
    assert abs(actual - target) < FAILURE_RATE_TOLERANCE, (
        f"simulator produces a {actual:.1%} failure rate against a calibrated "
        f"target of {target:.1%}"
    )


@pytest.mark.parametrize(
    "share",
    [
        "share_of_failures_insufficient_funds",
        "share_of_failures_technical",
        "share_of_failures_limit",
        "share_of_failures_window_rejected",
    ],
)
def test_observed_failure_mode_shares_match_their_targets(observed, share):
    metrics, _ = observed
    target = metrics["calibrated"][share]
    actual = metrics["observed"][share]
    assert abs(actual - target) < MODE_SHARE_TOLERANCE, (
        f"{share}: simulator produces {actual:.1%} against a calibrated "
        f"target of {target:.1%}"
    )


def test_the_failure_shares_observed_still_partition_the_failures(observed):
    metrics, _ = observed
    total = sum(
        metrics["observed"][key]
        for key in metrics["observed"]
        if key.startswith("share_of_failures_")
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_the_run_is_large_enough_to_mean_anything(observed):
    metrics, _ = observed
    assert metrics["n_attempts"] > 10_000
    assert metrics["n_failures"] > 2_000


# --------------------------------------------------------------------------
# The restricted window has to be visible
# --------------------------------------------------------------------------


def test_failures_are_more_common_inside_the_restricted_window(observed):
    """If the window is not visible in the data, it is not doing anything."""
    _, results = observed
    windows = DEFAULT_CALIBRATION.restricted_window_hours.value

    attempts = np.zeros(HOURS_IN_DAY)
    failures = np.zeros(HOURS_IN_DAY)
    for result in results:
        attempts += np.asarray(result.attempts_by_hour, dtype=float)
        failures += np.asarray(result.failures_by_hour, dtype=float)

    inside = np.array(
        [any(start <= hour < end for start, end in windows) for hour in range(24)]
    )
    inside_rate = failures[inside].sum() / attempts[inside].sum()
    outside_rate = failures[~inside].sum() / attempts[~inside].sum()

    assert inside_rate > outside_rate, (
        f"failure rate inside the window ({inside_rate:.1%}) is not higher "
        f"than outside it ({outside_rate:.1%})"
    )
    assert inside_rate - outside_rate > 0.10, (
        "the restricted window barely shows in the data; it should be a "
        "visible step, not a rounding difference"
    )


# --------------------------------------------------------------------------
# Determinism of the whole run
# --------------------------------------------------------------------------


def test_a_seed_reproduces_its_whole_run():
    left = validate_simulator.simulate_seed(3, n_mandates=120, n_days=45)
    right = validate_simulator.simulate_seed(3, n_mandates=120, n_days=45)
    assert left.outcome_counts == right.outcome_counts
    assert left.code_counts == right.code_counts
    assert left.attempts_by_hour == right.attempts_by_hour


def test_different_seeds_produce_different_runs():
    left = validate_simulator.simulate_seed(3, n_mandates=120, n_days=45)
    right = validate_simulator.simulate_seed(4, n_mandates=120, n_days=45)
    assert left.outcome_counts != right.outcome_counts


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def test_save_figure_writes_png_svg_and_caption(tmp_path):
    figure = figures.failure_mode_breakdown(
        {"INSUFFICIENT_FUNDS": 0.55, "TECHNICAL_DECLINE": 0.45}
    )
    written = figures.save_figure(
        figure, tmp_path / "example", "A caption a reader can reuse."
    )

    assert written["png"].exists() and written["png"].stat().st_size > 0
    assert written["svg"].exists() and written["svg"].stat().st_size > 0
    assert written["txt"].read_text(encoding="utf-8").strip() == (
        "A caption a reader can reuse."
    )


def test_every_figure_function_produces_a_saveable_figure(tmp_path):
    made = {
        "breakdown": figures.failure_mode_breakdown({"INSUFFICIENT_FUNDS": 1.0}),
        "by_hour": figures.failure_rate_by_hour(
            [10] * HOURS_IN_DAY, [3] * HOURS_IN_DAY, ((10, 13), (17, 21))
        ),
        "balances": figures.balance_trajectory_sample([[100, 200, 50]], [5]),
        "observed": figures.observed_vs_calibrated(
            {"failure_rate": 0.29}, {"failure_rate": 0.30}
        ),
    }
    for name, figure in made.items():
        written = figures.save_figure(figure, tmp_path / name, f"{name} caption")
        assert written["png"].exists()
        assert written["svg"].exists()


def test_the_palette_is_colourblind_safe_okabe_ito():
    assert figures.PALETTE[0] == "#0072B2"
    assert len(set(figures.PALETTE)) == len(figures.PALETTE)
