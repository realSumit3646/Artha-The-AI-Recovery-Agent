"""Tests for the calibration layer.

Two of these are the point of the commit: the failure shares must partition
the failures, and no number may enter the simulator without saying where it
came from. The rest guard the honesty rules against quiet erosion.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pydantic
import pytest

from mandate_recovery.calibration import (
    DEFAULT_CALIBRATION,
    FAILURE_SHARE_FIELDS,
    NO_PUBLIC_SOURCE,
    SALARY_CREDIT_DAYS,
    TODO_MARKER,
    BankTier,
    CalibratedValue,
    CalibrationSet,
)

CALIBRATION_DOC = Path(__file__).resolve().parents[1] / "docs" / "CALIBRATION.md"


# --------------------------------------------------------------------------
# The two required assertions
# --------------------------------------------------------------------------


def test_failure_shares_sum_to_one():
    """The four failure-share parameters partition all failures.

    Named explicitly rather than read from FAILURE_SHARE_FIELDS, so that
    dropping a share from both the model and the constant still fails here.
    """
    calibration = DEFAULT_CALIBRATION
    shares = [
        calibration.share_of_failures_insufficient_funds.value,
        calibration.share_of_failures_technical.value,
        calibration.share_of_failures_limit.value,
        calibration.share_of_failures_window_rejected.value,
    ]

    assert math.isclose(sum(shares), 1.0, abs_tol=1e-9), (
        f"failure shares sum to {sum(shares)!r}, not 1.0"
    )
    assert set(FAILURE_SHARE_FIELDS) == {
        "share_of_failures_insufficient_funds",
        "share_of_failures_technical",
        "share_of_failures_limit",
        "share_of_failures_window_rejected",
    }


def test_every_calibrated_value_has_a_non_empty_source():
    """No number enters the simulator unlabelled."""
    parameters = DEFAULT_CALIBRATION.parameters()
    assert parameters, "no parameters discovered"

    for name, calibrated in parameters.items():
        assert isinstance(calibrated, CalibratedValue), (
            f"{name} is not a CalibratedValue"
        )
        assert calibrated.source.strip(), f"{name} has a blank source"
        assert calibrated.unit.strip(), f"{name} has a blank unit"


# --------------------------------------------------------------------------
# Honesty rules
# --------------------------------------------------------------------------


def test_no_parameter_claims_a_source_it_does_not_have():
    """A placeholder may never be labelled `published` or `derived`."""
    for name, calibrated in DEFAULT_CALIBRATION.parameters().items():
        unsourced = (
            TODO_MARKER in calibrated.source
            or calibrated.source.startswith(NO_PUBLIC_SOURCE)
        )
        if unsourced:
            assert calibrated.confidence == "assumption", (
                f"{name} is unsourced but claims confidence="
                f"{calibrated.confidence!r}"
            )


def test_every_unsourced_parameter_says_so_in_words():
    """Each source is either the assumption wording or a TODO placeholder."""
    for name, calibrated in DEFAULT_CALIBRATION.parameters().items():
        if calibrated.confidence != "assumption":
            continue
        assert calibrated.source.startswith(NO_PUBLIC_SOURCE) or (
            TODO_MARKER in calibrated.source
        ), f"{name} is an assumption but its source does not admit it"


def test_calibrated_value_rejects_a_dressed_up_guess():
    """The honesty rule is structural, not a convention."""
    for confidence in ("published", "derived"):
        with pytest.raises(pydantic.ValidationError):
            CalibratedValue(
                value=0.5,
                unit="fraction",
                source=f"{TODO_MARKER}: not really sourced",
                confidence=confidence,
            )
        with pytest.raises(pydantic.ValidationError):
            CalibratedValue(
                value=0.5,
                unit="fraction",
                source=NO_PUBLIC_SOURCE,
                confidence=confidence,
            )


def test_calibrated_value_rejects_a_blank_source():
    for blank in ("", "   "):
        with pytest.raises(pydantic.ValidationError):
            CalibratedValue(
                value=0.5, unit="fraction", source=blank, confidence="assumption"
            )


# --------------------------------------------------------------------------
# Cross-parameter validation
# --------------------------------------------------------------------------


def _with(**overrides) -> dict:
    """The default calibration as a dict, with parameters replaced."""
    data = DEFAULT_CALIBRATION.model_dump()
    for name, value in overrides.items():
        data[name] = {**data[name], "value": value}
    return data


def test_failure_shares_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(pydantic.ValidationError):
        CalibrationSet.model_validate(_with(share_of_failures_technical=0.9))


def test_salary_distribution_must_sum_to_one():
    weights = dict(DEFAULT_CALIBRATION.salary_credit_day_distribution.value)
    weights[1] = weights[1] + 0.05
    with pytest.raises(pydantic.ValidationError):
        CalibrationSet.model_validate(_with(salary_credit_day_distribution=weights))


def test_salary_distribution_rejects_days_outside_the_pay_window():
    weights = dict(DEFAULT_CALIBRATION.salary_credit_day_distribution.value)
    weights[15] = weights.pop(1)
    with pytest.raises(pydantic.ValidationError):
        CalibrationSet.model_validate(_with(salary_credit_day_distribution=weights))


def test_salary_distribution_covers_only_the_declared_days():
    weights = DEFAULT_CALIBRATION.salary_credit_day_distribution.value
    assert set(weights) <= set(SALARY_CREDIT_DAYS)
    assert set(SALARY_CREDIT_DAYS) == set(range(1, 8)) | set(range(25, 32))


def test_all_bank_tiers_are_calibrated():
    availability = DEFAULT_CALIBRATION.bank_availability_by_tier.value
    assert set(availability) == set(BankTier)

    incomplete = {k: v for k, v in availability.items() if k is not BankTier.PSU}
    with pytest.raises(pydantic.ValidationError):
        CalibrationSet.model_validate(_with(bank_availability_by_tier=incomplete))


@pytest.mark.parametrize(
    "name", ["upi_autopay_execution_failure_rate", "card_penetration_rate"]
)
@pytest.mark.parametrize("bad", [-0.1, 1.4])
def test_probabilities_stay_within_zero_and_one(name, bad):
    with pytest.raises(pydantic.ValidationError):
        CalibrationSet.model_validate(_with(**{name: bad}))


# --------------------------------------------------------------------------
# Money and immutability
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "gateway_cost_per_attempt_paise",
        "sms_cost_paise",
        "voice_call_cost_paise",
    ],
)
def test_cost_parameters_are_integer_paise(name):
    value = DEFAULT_CALIBRATION.parameters()[name].value
    assert isinstance(value, int) and not isinstance(value, bool)


@pytest.mark.parametrize("bad", [200.5, 200.0, -5])
def test_cost_parameters_reject_non_integer_paise(bad):
    """Integral floats are rejected too: 200.0 is not an integer paise value."""
    with pytest.raises(pydantic.ValidationError):
        CalibrationSet.model_validate(_with(gateway_cost_per_attempt_paise=bad))


def test_calibration_set_is_frozen():
    with pytest.raises(pydantic.ValidationError):
        DEFAULT_CALIBRATION.card_penetration_rate = None


def test_calibrated_value_is_frozen():
    with pytest.raises(pydantic.ValidationError):
        DEFAULT_CALIBRATION.card_penetration_rate.value = 0.9


def test_calibration_set_rejects_unknown_parameters():
    """A number cannot be smuggled in without a home in the model."""
    data = DEFAULT_CALIBRATION.model_dump()
    data["invented_parameter"] = data["card_penetration_rate"]
    with pytest.raises(pydantic.ValidationError):
        CalibrationSet.model_validate(data)


def test_calibration_survives_a_json_round_trip():
    """Invariant 4: an experiment is reproducible from its stored config."""
    restored = CalibrationSet.model_validate_json(
        DEFAULT_CALIBRATION.model_dump_json()
    )
    assert restored == DEFAULT_CALIBRATION


# --------------------------------------------------------------------------
# Documentation cannot drift from the code
# --------------------------------------------------------------------------


def _documented_rows() -> dict[str, str]:
    """Parameter name -> confidence, read from the CALIBRATION.md table."""
    rows: dict[str, str] = {}
    for line in CALIBRATION_DOC.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*`(\w+)`\s*\|.*\|\s*`(\w+)`\s*\|\s*$", line)
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def test_every_parameter_is_documented():
    documented = _documented_rows()
    assert documented, f"no parameter rows parsed from {CALIBRATION_DOC.name}"

    missing = set(DEFAULT_CALIBRATION.parameters()) - set(documented)
    assert not missing, f"undocumented parameters: {sorted(missing)}"


def test_documented_confidence_matches_the_code():
    documented = _documented_rows()
    for name, calibrated in DEFAULT_CALIBRATION.parameters().items():
        assert documented[name] == calibrated.confidence, (
            f"{name}: docs say {documented[name]!r}, "
            f"code says {calibrated.confidence!r}"
        )


def test_doc_states_that_nothing_is_published_yet():
    """The headline honesty claim is present, not just implied by the table."""
    text = CALIBRATION_DOC.read_text(encoding="utf-8")
    assert "Nothing in this table is a published figure yet" in text
    assert "No citation in this project is invented" in text


def test_doc_explains_the_range_and_the_sweep():
    text = CALIBRATION_DOC.read_text(encoding="utf-8")
    assert "vary widely" in text
    assert "conservative" in text
    assert "commit 27" in text
