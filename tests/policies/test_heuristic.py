"""Tests for the deterministic heuristic agent.

This is the ablation control, so several of these exist to prove it is *not*
handicapped: it must diagnose everything a code book honestly can, act on the
contradiction cases, and never be quietly worse than the fixed schedule at the
things both can do.
"""

from __future__ import annotations

import pytest

from mandate_recovery.agent.validator import ComplianceLimits
from mandate_recovery.policies.heuristic import (
    DIAGNOSIS_CODE_BOOK,
    UNINFORMATIVE_CODES,
    Diagnosis,
    HeuristicPolicy,
    diagnose,
)
from mandate_recovery.types import (
    CollectPartial,
    Observation,
    ObservedAttempt,
    Rail,
    RetrySilent,
    SendNudge,
    Stop,
)

LIMITS = ComplianceLimits()


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": 880_000,
        "due_day": 5,
        "current_day": 10,
        "current_hour": 9,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def _history(*codes: str, day: int = 10):
    return tuple(
        ObservedAttempt(day=day, hour=6, rail=Rail.UPI_AUTOPAY, raw_code=code)
        for code in codes
    )


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code, expected",
    [
        ("AB9001", Diagnosis.TECHNICAL),
        ("PS-91", Diagnosis.TECHNICAL),
        ("SF_SYSERR", Diagnosis.TECHNICAL),
        ("AB3301", Diagnosis.LIMIT),
        ("PS-77", Diagnosis.WINDOW),
        ("SF_CANCELLED", Diagnosis.REVOKED),
    ],
)
def test_unambiguous_codes_are_diagnosed_from_the_code_book(code, expected):
    result = diagnose(_observation(attempt_history=_history(code)))
    assert result.diagnosis is expected
    assert result.confident is True


@pytest.mark.parametrize("code", sorted(UNINFORMATIVE_CODES))
def test_uninformative_codes_return_unknown_rather_than_a_guess(code):
    """A rules agent that claimed to diagnose these would be lying."""
    result = diagnose(_observation(attempt_history=_history(code)))
    assert result.diagnosis is Diagnosis.UNKNOWN
    assert result.confident is False


def test_an_unseen_code_returns_unknown():
    result = diagnose(_observation(attempt_history=_history("ZZ999")))
    assert result.diagnosis is Diagnosis.UNKNOWN


def test_the_code_book_matches_the_simulators_vocabulary():
    """The agent learned this from data; it must not have drifted."""
    from mandate_recovery.calibration import BankTier
    from mandate_recovery.sim.response_codes import BANK_CODE_VOCABULARY
    from mandate_recovery.types import AttemptOutcome

    expected = {
        AttemptOutcome.INSUFFICIENT_FUNDS: Diagnosis.INSUFFICIENT_FUNDS,
        AttemptOutcome.TECHNICAL_DECLINE: Diagnosis.TECHNICAL,
        AttemptOutcome.LIMIT_EXCEEDED: Diagnosis.LIMIT,
        AttemptOutcome.WINDOW_REJECTED: Diagnosis.WINDOW,
        AttemptOutcome.MANDATE_REVOKED: Diagnosis.REVOKED,
    }
    for tier in BankTier:
        for outcome, diagnosis in expected.items():
            code = BANK_CODE_VOCABULARY[tier][outcome]
            assert DIAGNOSIS_CODE_BOOK.get(code) is diagnosis, (
                f"{code} really means {outcome.value} but the agent reads it "
                f"as {DIAGNOSIS_CODE_BOOK.get(code)}"
            )


def _imported_modules(relative_path: str) -> set[str]:
    """Every module a file imports, read from its AST.

    Text search would be fooled by a docstring that merely mentions the
    simulator, which is exactly what these modules do.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "mandate_recovery"
    tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(("." * node.level) + (node.module or ""))
    return modules


def test_the_heuristic_does_not_import_the_simulator():
    modules = _imported_modules("policies/heuristic.py")
    assert not any("sim" in module.split(".") for module in modules), modules


# --------------------------------------------------------------------------
# The contradiction case
# --------------------------------------------------------------------------


def test_a_funds_code_is_trusted_when_history_clears_the_ceiling():
    """Rule 1: they have paid this much before, so the ceiling permits it."""
    result = diagnose(
        _observation(
            amount_paise=500_000,
            attempt_history=_history("PS-51"),
            max_historical_success_amount_paise=900_000,
        )
    )
    assert result.diagnosis is Diagnosis.INSUFFICIENT_FUNDS
    assert result.confident is True
    assert "ceiling permits" in result.rationale


def test_a_funds_code_is_distrusted_when_the_amount_exceeds_every_payment():
    """Rule 2: a limit breach would look identical, so say so."""
    result = diagnose(
        _observation(
            amount_paise=900_000,
            attempt_history=_history("PS-51"),
            max_historical_success_amount_paise=500_000,
        )
    )
    assert result.diagnosis is Diagnosis.UNKNOWN
    assert result.confident is False
    assert "ceiling" in result.rationale


def test_a_funds_code_with_no_history_is_taken_at_face_value_unconfidently():
    result = diagnose(
        _observation(
            attempt_history=_history("PS-51"),
            max_historical_success_amount_paise=0,
        )
    )
    assert result.diagnosis is Diagnosis.INSUFFICIENT_FUNDS
    assert result.confident is False


def test_the_unknown_rate_is_measured_and_exposed():
    """The number that argues for adding a model stage."""
    policy = HeuristicPolicy()
    for code in ("DECLINED", "", "PS-91", "AB1200"):
        policy.decide(
            _observation(
                attempt_history=_history(code),
                max_historical_success_amount_paise=10_000_000,
            )
        )
    assert policy.diagnosis_counts["UNKNOWN"] == 2
    assert policy.unknown_diagnosis_rate == pytest.approx(0.5)


def test_the_unknown_rate_is_zero_before_anything_is_diagnosed():
    assert HeuristicPolicy().unknown_diagnosis_rate == 0.0


# --------------------------------------------------------------------------
# The decision table
# --------------------------------------------------------------------------


def test_a_revoked_mandate_is_abandoned_immediately():
    decision = HeuristicPolicy().decide(
        _observation(attempt_history=_history("PS-14"))
    )
    assert isinstance(decision.action, Stop)


def test_a_limit_breach_collects_the_largest_amount_known_to_work():
    decision = HeuristicPolicy().decide(
        _observation(
            amount_paise=900_000,
            attempt_history=_history("AB3301"),
            max_historical_success_amount_paise=400_000,
        )
    )
    assert isinstance(decision.action, CollectPartial)
    assert decision.action.amount_paise == 400_000


def test_a_limit_breach_with_no_known_lower_bound_stops():
    decision = HeuristicPolicy().decide(
        _observation(
            attempt_history=_history("AB3301"),
            max_historical_success_amount_paise=0,
        )
    )
    assert isinstance(decision.action, Stop)


def test_a_technical_decline_is_simply_retried():
    decision = HeuristicPolicy().decide(
        _observation(attempt_history=_history("PS-91"))
    )
    assert isinstance(decision.action, RetrySilent)


def test_it_nudges_only_after_silent_retries_have_failed():
    """Contact is expensive, so it is not the first move."""
    policy = HeuristicPolicy()
    first = policy.decide(
        _observation(
            attempt_history=_history("DECLINED"), current_day=10, due_day=25
        )
    )
    assert isinstance(first.action, RetrySilent)

    third = policy.decide(
        _observation(
            attempt_history=_history("DECLINED", "DECLINED"),
            current_day=10,
            due_day=25,
        )
    )
    assert isinstance(third.action, SendNudge)


def test_it_never_nudges_twice():
    decision = HeuristicPolicy().decide(
        _observation(
            attempt_history=_history("DECLINED", "DECLINED"),
            contacts_sent=1,
            current_day=10,
            due_day=25,
        )
    )
    assert not isinstance(decision.action, SendNudge)


def test_it_stops_when_the_cycle_is_about_to_lapse():
    decision = HeuristicPolicy().decide(
        _observation(
            attempt_history=_history("PS-91"), current_day=10, due_day=12
        )
    )
    assert isinstance(decision.action, Stop)
    assert "lapse" in decision.action.reason
    assert "supersedes this one" in decision.rationale


# --------------------------------------------------------------------------
# The gate is not bypassed
# --------------------------------------------------------------------------


def test_every_action_it_emits_has_passed_the_validator():
    policy = HeuristicPolicy()
    for attempts in range(1, 8):
        for code in ("DECLINED", "PS-51", "AB9001"):
            decision = policy.decide(
                _observation(
                    attempt_history=_history(*([code] * attempts)),
                    current_day=10,
                    due_day=25,
                )
            )
            action = decision.action
            if isinstance(action, RetrySilent):
                assert not LIMITS.is_restricted(action.scheduled_hour)


def test_it_stops_once_the_attempt_cap_is_reached():
    policy = HeuristicPolicy()
    decision = policy.decide(
        _observation(
            attempt_history=_history(
                *(["PS-91"] * LIMITS.max_attempts_per_cycle)
            ),
            current_day=10,
            due_day=25,
        )
    )
    assert isinstance(decision.action, Stop)
    assert decision.validated is False
    assert policy.validator.rejections["attempt_cap_reached"] == 1


def test_a_refused_decision_is_marked_unvalidated():
    policy = HeuristicPolicy()
    decision = policy.decide(
        _observation(
            attempt_history=_history(*(["PS-91"] * 9)),
            current_day=10,
            due_day=25,
        )
    )
    assert decision.validated is False
    assert "compliance gate refused" in decision.rationale


def test_an_approved_decision_is_marked_validated():
    decision = HeuristicPolicy().decide(
        _observation(attempt_history=_history("PS-91"), current_day=10, due_day=25)
    )
    assert decision.validated is True


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------


def test_it_is_deterministic():
    observation = _observation(attempt_history=_history("PS-51"))
    assert HeuristicPolicy().decide(observation) == HeuristicPolicy().decide(
        observation
    )


def test_every_decision_carries_a_rationale_naming_the_evidence():
    decision = HeuristicPolicy().decide(
        _observation(attempt_history=_history("PS-51"))
    )
    assert "PS-51" in decision.rationale
    assert len(decision.rationale) > 60
    assert decision.source == "rule"


def test_reset_clears_counters_between_seeds():
    policy = HeuristicPolicy()
    policy.decide(_observation(attempt_history=_history("DECLINED")))
    assert policy.diagnosis_counts
    policy.reset()
    assert policy.diagnosis_counts == {}
    assert policy.validator.total_rejections == 0


def test_it_is_bound_by_the_observation_boundary():
    assert HeuristicPolicy.reads_latent_state is False
