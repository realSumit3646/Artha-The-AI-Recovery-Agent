"""Tests for the bank code messiness layer.

The load-bearing test here is the recoverable-fraction ceiling: if a lookup
table can resolve most failures from the code alone, the messiness is too weak
and a diagnosis stage is not justifiable.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from mandate_recovery.calibration import DEFAULT_CALIBRATION, BankTier
from mandate_recovery.sim.response_codes import (
    BANK_CODE_VOCABULARY,
    GENERIC_CODE,
    MISSING_CODES,
    SHARE_OF_FAILURES_GENERIC,
    SHARE_OF_FAILURES_MISSING,
    SHARE_OF_LIMIT_FAILURES_MISCODED_AS_FUNDS,
    TRUE_CAUSE_RECOVERABLE_FRACTION,
    encode_response,
)
from mandate_recovery.types import AttemptOutcome

FAILURE_OUTCOMES = (
    AttemptOutcome.INSUFFICIENT_FUNDS,
    AttemptOutcome.TECHNICAL_DECLINE,
    AttemptOutcome.LIMIT_EXCEEDED,
    AttemptOutcome.WINDOW_REJECTED,
)


def _failure_mix() -> list[float]:
    """The calibrated share of each failure cause."""
    return [
        DEFAULT_CALIBRATION.share_of_failures_insufficient_funds.value,
        DEFAULT_CALIBRATION.share_of_failures_technical.value,
        DEFAULT_CALIBRATION.share_of_failures_limit.value,
        DEFAULT_CALIBRATION.share_of_failures_window_rejected.value,
    ]


def _sample(n: int, seed: int = 0) -> list[tuple[AttemptOutcome, str]]:
    """(true cause, emitted code) pairs drawn from the calibrated mix."""
    rng = np.random.default_rng(seed)
    tiers = list(BankTier)
    tier_probs = [DEFAULT_CALIBRATION.bank_tier_mix.value[t] for t in tiers]

    cause_index = rng.choice(len(FAILURE_OUTCOMES), size=n, p=_failure_mix())
    tier_index = rng.choice(len(tiers), size=n, p=tier_probs)
    return [
        (
            FAILURE_OUTCOMES[cause],
            encode_response(FAILURE_OUTCOMES[cause], tiers[tier].value, rng),
        )
        for cause, tier in zip(cause_index, tier_index)
    ]


def _recoverable_fraction(pairs) -> float:
    """Share of failures whose code identifies exactly one cause."""
    causes_by_code: dict[str, set[AttemptOutcome]] = defaultdict(set)
    for cause, code in pairs:
        causes_by_code[code].add(cause)
    unambiguous = sum(1 for _, code in pairs if len(causes_by_code[code]) == 1)
    return unambiguous / len(pairs)


# --------------------------------------------------------------------------
# The design constraint
# --------------------------------------------------------------------------


def test_true_cause_is_not_recoverable_from_the_code_alone():
    """The ceiling that justifies having a diagnosis stage at all.

    If a lookup table can resolve more than three quarters of failures, the
    messiness is too weak: diagnosis becomes a dictionary and any measured
    benefit from a model is an artefact of the simulator being kind.
    """
    observed = _recoverable_fraction(_sample(200_000))
    assert observed < 0.75, (
        f"{observed:.1%} of failures are unambiguous; the messiness layer is "
        "too weak to justify a diagnosis stage"
    )


def test_documented_recoverable_fraction_matches_reality():
    """The module constant is not allowed to drift from what the code does."""
    observed = _recoverable_fraction(_sample(200_000, seed=7))
    assert observed == pytest.approx(TRUE_CAUSE_RECOVERABLE_FRACTION, abs=0.02)


# --------------------------------------------------------------------------
# Each source of mess
# --------------------------------------------------------------------------


def test_the_same_cause_reads_differently_at_different_banks():
    codes = {
        tier: BANK_CODE_VOCABULARY[tier][AttemptOutcome.INSUFFICIENT_FUNDS]
        for tier in BankTier
    }
    assert len(set(codes.values())) == len(BankTier)


def test_every_bank_has_a_distinct_code_for_every_outcome():
    for tier in BankTier:
        vocabulary = BANK_CODE_VOCABULARY[tier]
        assert set(vocabulary) == set(AttemptOutcome)
        assert len(set(vocabulary.values())) == len(AttemptOutcome)


def test_generic_and_missing_codes_appear_at_their_configured_shares():
    codes = [code for _, code in _sample(100_000, seed=11)]
    generic = codes.count(GENERIC_CODE) / len(codes)
    missing = sum(1 for code in codes if code in MISSING_CODES) / len(codes)

    assert generic == pytest.approx(SHARE_OF_FAILURES_GENERIC, abs=0.01)
    assert missing == pytest.approx(SHARE_OF_FAILURES_MISSING, abs=0.01)


def test_the_generic_code_covers_at_least_three_causes():
    causes_by_code: dict[str, set[AttemptOutcome]] = defaultdict(set)
    for cause, code in _sample(50_000, seed=13):
        causes_by_code[code].add(cause)
    assert len(causes_by_code[GENERIC_CODE]) >= 3


def test_missing_codes_are_emitted_and_are_uninformative():
    causes_by_code: dict[str, set[AttemptOutcome]] = defaultdict(set)
    for cause, code in _sample(50_000, seed=17):
        causes_by_code[code].add(cause)
    for missing in MISSING_CODES:
        assert missing in causes_by_code, f"{missing!r} is never emitted"
        assert len(causes_by_code[missing]) >= 3


def test_limit_breaches_sometimes_report_a_funds_code():
    """The contradiction case: only history can disambiguate it."""
    rng = np.random.default_rng(19)
    tier = BankTier.PSU
    funds_code = BANK_CODE_VOCABULARY[tier][AttemptOutcome.INSUFFICIENT_FUNDS]

    codes = [
        encode_response(AttemptOutcome.LIMIT_EXCEEDED, tier.value, rng)
        for _ in range(50_000)
    ]
    miscoded = codes.count(funds_code) / len(codes)

    specific = 1.0 - SHARE_OF_FAILURES_GENERIC - SHARE_OF_FAILURES_MISSING
    expected = specific * SHARE_OF_LIMIT_FAILURES_MISCODED_AS_FUNDS
    assert miscoded > 0.0
    assert miscoded == pytest.approx(expected, abs=0.01)


def test_a_funds_code_is_never_conclusive_on_its_own():
    causes_by_code: dict[str, set[AttemptOutcome]] = defaultdict(set)
    for cause, code in _sample(100_000, seed=23):
        causes_by_code[code].add(cause)

    for tier in BankTier:
        funds = BANK_CODE_VOCABULARY[tier][AttemptOutcome.INSUFFICIENT_FUNDS]
        assert AttemptOutcome.LIMIT_EXCEEDED in causes_by_code[funds], (
            f"{funds} is unambiguous; the contradiction case is not firing"
        )


# --------------------------------------------------------------------------
# Clean outcomes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcome", [AttemptOutcome.SUCCESS, AttemptOutcome.MANDATE_REVOKED]
)
def test_clean_outcomes_are_never_messy(outcome):
    """A merchant knows whether money arrived, and whether they still hold a
    mandate. Neither is a diagnostic puzzle."""
    rng = np.random.default_rng(29)
    for tier in BankTier:
        expected = BANK_CODE_VOCABULARY[tier][outcome]
        emitted = {
            encode_response(outcome, tier.value, rng) for _ in range(2_000)
        }
        assert emitted == {expected}


def test_clean_outcomes_consume_no_randomness():
    rng = np.random.default_rng(31)
    encode_response(AttemptOutcome.SUCCESS, BankTier.PSU.value, rng)
    assert rng.random() == np.random.default_rng(31).random()


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------


def test_encoding_is_deterministic_under_a_seed():
    def run(seed: int) -> list[str]:
        rng = np.random.default_rng(seed)
        return [
            encode_response(
                AttemptOutcome.INSUFFICIENT_FUNDS, BankTier.PSU.value, rng
            )
            for _ in range(200)
        ]

    assert run(37) == run(37)
    assert run(37) != run(41)


def test_encoding_requires_an_explicit_generator():
    for bad in (None, 42, np.random.RandomState(0)):
        with pytest.raises(TypeError):
            encode_response(AttemptOutcome.TECHNICAL_DECLINE, "psu", bad)


def test_unknown_bank_is_rejected_loudly():
    with pytest.raises(KeyError):
        encode_response(
            AttemptOutcome.TECHNICAL_DECLINE,
            "not_a_bank",
            np.random.default_rng(0),
        )
