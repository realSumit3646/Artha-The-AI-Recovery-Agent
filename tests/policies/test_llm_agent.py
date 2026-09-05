"""Tests for the assembled LLM agent.

The headline test is the resilience proof: with a client that always raises,
this agent produces decisions *identical* to the heuristic agent's. That is
what lets the ablation claim the model was compared on judgement rather than
on availability.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandate_recovery.harness import ExperimentConfig, run_experiment
from mandate_recovery.llm.client import StubClient
from mandate_recovery.llm.diagnosis import DiagnosisReply
from mandate_recovery.llm.intervention import InterventionReply
from mandate_recovery.llm.messaging import MessageReply
from mandate_recovery.policies import DoNothingPolicy, HeuristicPolicy
from mandate_recovery.policies.llm_agent import LLMAgentPolicy
from mandate_recovery.types import (
    Observation,
    ObservedAttempt,
    Rail,
    RetrySilent,
    SendNudge,
    Stop,
)

GOOD_TEMPLATE = (
    "Hi! Aapka {amount} ka payment {due_date} ko fail hua. Please balance "
    "rakhein - {merchant} (Ref {reference})."
)

WORKING_REPLIES = {
    "DiagnosisReply": DiagnosisReply(
        cause="LIMIT", confidence=0.9, reasoning="above every past payment"
    ),
    "InterventionReply": InterventionReply(
        action="RETRY_SILENT",
        timing="AFTER_NEXT_SALARY",
        tone_level=1,
        reasoning="wait for payday",
    ),
    "MessageReply": MessageReply(message_template=GOOD_TEMPLATE),
}


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m000123",
        "amount_paise": 880_000,
        "due_day": 25,
        "current_day": 10,
        "current_hour": 10,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def _history(count: int = 1, code: str = "DECLINED"):
    return tuple(
        ObservedAttempt(day=10, hour=6, rail=Rail.UPI_AUTOPAY, raw_code=code)
        for _ in range(count)
    )


# --------------------------------------------------------------------------
# The resilience proof
# --------------------------------------------------------------------------


def test_with_a_dead_model_it_decides_exactly_as_the_heuristic_does():
    """The required test, on a wide range of observations.

    Not "similar" — identical. If the model is unavailable this agent *is*
    the heuristic agent, so it can never lose the ablation on availability.
    """
    agent = LLMAgentPolicy(StubClient(always_fail=True))
    heuristic = HeuristicPolicy()

    for code in ("DECLINED", "", "NA", "PS-51", "PS-91", "AB3301", "PS-14"):
        for attempts in (1, 2, 3, 5):
            for ceiling in (0, 400_000, 900_000):
                observation = _observation(
                    attempt_history=_history(attempts, code),
                    max_historical_success_amount_paise=ceiling,
                )
                assert agent.decide(observation) == heuristic.decide(observation)


def test_a_dead_model_produces_identical_experiment_results():
    """The same claim at the level that actually matters: the numbers."""
    config = ExperimentConfig(
        experiment_id="resilience", seeds=[3], n_customers=60, n_mandates=60, n_days=70
    )
    arms = {
        "heuristic": lambda world, mapping: HeuristicPolicy(),
        "llm_dead": lambda world, mapping: LLMAgentPolicy(
            StubClient(always_fail=True)
        ),
    }
    episodes, _ = run_experiment(arms, config)

    heuristic = {e.mandate_id: e for e in episodes if e.arm == "heuristic"}
    dead = {e.mandate_id: e for e in episodes if e.arm == "llm_dead"}
    assert heuristic.keys() == dead.keys()
    for mandate_id, left in heuristic.items():
        right = dead[mandate_id]
        assert left.net_recovery_paise == right.net_recovery_paise
        assert left.recovered_paise == right.recovered_paise
        assert left.attempts == right.attempts


def test_it_never_raises_whatever_the_model_does():
    """An experiment must not die because an API had a bad minute."""

    class Exploding:
        model = "exploding"
        offline = True

        def complete(self, *args, **kwargs):
            raise RuntimeError("something entirely unexpected")

    agent = LLMAgentPolicy(Exploding())
    decision = agent.decide(_observation(attempt_history=_history()))
    assert decision is not None
    assert agent.stats()["fallbacks_by_stage"]


def test_fallbacks_are_counted_by_stage():
    agent = LLMAgentPolicy(StubClient(always_fail=True))
    agent.decide(_observation(attempt_history=_history(1, "DECLINED")))
    agent.decide(_observation(attempt_history=_history(1, "PS-91")))

    stages = agent.stats()["fallbacks_by_stage"]
    assert stages["diagnosis"] == 1  # residual code, model unavailable
    assert stages["intervention"] == 1  # rule-diagnosed, proposal unavailable
    assert agent.stats()["fallback_rate"] == 1.0


# --------------------------------------------------------------------------
# The working path
# --------------------------------------------------------------------------


def test_a_working_model_produces_an_llm_sourced_decision():
    agent = LLMAgentPolicy(StubClient(WORKING_REPLIES))
    decision = agent.decide(
        _observation(
            attempt_history=_history(1, "DECLINED"),
            max_historical_success_amount_paise=400_000,
        )
    )
    assert decision.source == "llm"
    assert isinstance(decision.action, RetrySilent)
    assert agent.stats()["llm_decisions"] == 1
    assert agent.stats()["fallback_rate"] == 0.0


def test_a_rule_resolvable_code_still_reaches_the_intervention_stage():
    """Routing applies to diagnosis only; the model still picks the action."""
    agent = LLMAgentPolicy(StubClient(WORKING_REPLIES))
    agent.decide(_observation(attempt_history=_history(1, "PS-91")))

    stats = agent.stats()
    assert stats["rule_resolved"] == 1
    assert stats["llm_invoked"] == 0
    assert stats["proposals"] == 1


def test_the_invocation_rate_is_reported():
    agent = LLMAgentPolicy(StubClient(WORKING_REPLIES))
    for code in ("PS-91", "AB3301", "DECLINED", "NA"):
        agent.decide(_observation(attempt_history=_history(1, code)))
    assert agent.stats()["llm_invocation_rate"] == pytest.approx(0.5)


def test_a_nudge_carries_a_verified_message():
    replies = dict(WORKING_REPLIES)
    replies["InterventionReply"] = InterventionReply(
        action="SEND_NUDGE", timing="SOON", tone_level=1, reasoning="ask them"
    )
    agent = LLMAgentPolicy(StubClient(replies))
    decision = agent.decide(_observation(attempt_history=_history(2, "DECLINED")))

    assert isinstance(decision.action, SendNudge)
    assert "Rs 8,800.00" in decision.rationale
    assert "M000123" in decision.rationale
    assert agent.stats()["messages_generated"] == 1


def test_a_bad_message_degrades_the_wording_not_the_decision():
    replies = dict(WORKING_REPLIES)
    replies["InterventionReply"] = InterventionReply(
        action="SEND_NUDGE", timing="SOON", tone_level=1, reasoning="ask them"
    )
    replies["MessageReply"] = MessageReply(
        message_template="Pay Rs 99999 now {amount} {due_date} {reference} {merchant}"
    )
    agent = LLMAgentPolicy(StubClient(replies))
    decision = agent.decide(_observation(attempt_history=_history(2, "DECLINED")))

    assert isinstance(decision.action, SendNudge)
    assert "99999" not in decision.rationale
    assert agent.stats()["message_verification_failures"] == 1


# --------------------------------------------------------------------------
# The gate still holds
# --------------------------------------------------------------------------


def test_a_model_proposal_that_breaks_a_rule_still_never_executes():
    replies = dict(WORKING_REPLIES)
    replies["InterventionReply"] = InterventionReply(
        action="SEND_NUDGE", timing="SOON", tone_level=3, reasoning="now"
    )
    agent = LLMAgentPolicy(StubClient(replies))
    decision = agent.decide(
        _observation(current_hour=23, attempt_history=_history(2, "DECLINED"))
    )

    assert not isinstance(decision.action, SendNudge)
    assert decision.validated is False


def test_it_is_bound_by_the_observation_boundary():
    assert LLMAgentPolicy.reads_latent_state is False


def test_it_registers_itself():
    from mandate_recovery.policies import POLICY_REGISTRY

    assert POLICY_REGISTRY["llm_agent"] is LLMAgentPolicy


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def test_reset_clears_every_stage_counter():
    agent = LLMAgentPolicy(StubClient(WORKING_REPLIES))
    agent.decide(_observation(attempt_history=_history(1, "DECLINED")))
    agent.reset()

    stats = agent.stats()
    assert stats["decisions"] == 0
    assert stats["llm_decisions"] == 0
    assert stats["diagnoses"] == 0
    assert stats["fallbacks_by_stage"] == {}


def test_stats_expose_everything_the_reliability_figure_needs():
    agent = LLMAgentPolicy(StubClient(WORKING_REPLIES))
    agent.decide(_observation(attempt_history=_history(1, "DECLINED")))
    stats = agent.stats()

    for key in (
        "llm_invocation_rate",
        "fallback_rate",
        "fallbacks_by_stage",
        "message_fallbacks",
        "refused_by_validator",
        "validator_rejections",
    ):
        assert key in stats
