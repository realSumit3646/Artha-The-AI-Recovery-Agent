"""The assembled LLM agent, with a deterministic floor under every stage.

The pipeline::

    detect      deterministic   a failure happened; the harness said so
    diagnose    rules, then the model on the residual only
    propose     the model picks a kind of action and a rough time
    schedule    deterministic   the exact slot
    validate    deterministic   approve, correct, or refuse
    message     the model writes it; a verifier checks every fact
    audit       deterministic   what was proposed, what ran, and why

**Every stage that can fail falls back to the heuristic agent**, and the
fallback is total rather than partial: on any model failure this policy
returns the heuristic's own decision, unchanged. That is what makes the
resilience claim checkable — with a client that always raises, this agent and
`HeuristicPolicy` produce *identical* decisions, which is a test rather than
an assurance.

The design consequence worth stating: this agent can never be worse than the
heuristic because the model is unavailable. If it loses the ablation, it will
have lost on the quality of its judgement, which is the only comparison worth
running.
"""

from __future__ import annotations

from collections import Counter
from typing import ClassVar, Mapping

from ..agent.scheduler import SchedulerConstraints
from ..agent.validator import Budget, ComplianceLimits, Validator
from ..calibration import DEFAULT_CALIBRATION, CalibrationSet
from ..llm.diagnosis import DiagnosisRouter
from ..llm.intervention import InterventionSelector
from ..llm.messaging import MessageGenerator
from ..types import Decision, Observation, SendNudge
from .base import Policy
from .heuristic import HeuristicPolicy

__all__ = ["LLMAgentPolicy"]


class LLMAgentPolicy(Policy):
    """Rules, then a model on the residual, then deterministic execution."""

    name: ClassVar[str] = "llm_agent"

    def __init__(
        self,
        client,
        calibration: CalibrationSet = DEFAULT_CALIBRATION,
        constraints: SchedulerConstraints | None = None,
        limits: ComplianceLimits | None = None,
        merchant_name: str = "Artha Payments",
    ) -> None:
        self._calibration = calibration
        self._constraints = constraints or SchedulerConstraints()
        self._validator = Validator(limits)

        self._router = DiagnosisRouter(client)
        self._selector = InterventionSelector(
            client, validator=self._validator, constraints=self._constraints
        )
        self._messenger = MessageGenerator(client, merchant_name=merchant_name)

        # The floor. Shares nothing with the LLM path except the observation,
        # so a fallback is a clean handover rather than a half-built decision.
        self._heuristic = HeuristicPolicy(
            calibration=calibration, constraints=self._constraints, limits=limits
        )

        self._fallbacks: Counter = Counter()
        self._decisions = 0
        self._llm_decisions = 0

    # ------------------------------------------------------------------

    @property
    def validator(self) -> Validator:
        return self._validator

    @property
    def router(self) -> DiagnosisRouter:
        return self._router

    def stats(self) -> dict[str, object]:
        return {
            "decisions": self._decisions,
            "llm_decisions": self._llm_decisions,
            "fallback_rate": (
                sum(self._fallbacks.values()) / self._decisions
                if self._decisions
                else 0.0
            ),
            "fallbacks_by_stage": dict(self._fallbacks),
            **self._router.stats(),
            **self._selector.stats(),
            **self._messenger.stats(),
            "validator_rejections": dict(self._validator.rejections),
            "heuristic_unknown_rate": self._heuristic.unknown_diagnosis_rate,
        }

    def reset(self) -> None:
        self._fallbacks.clear()
        self._decisions = self._llm_decisions = 0
        self._router.reset()
        self._selector.reset()
        self._messenger.reset()
        self._heuristic.reset()
        self._validator.reset()

    # ------------------------------------------------------------------

    def _fall_back(self, observation: Observation, stage: str) -> Decision:
        """Hand the whole decision to the heuristic. Nothing partial survives."""
        self._fallbacks[stage] += 1
        return self._heuristic.decide(observation)

    def _spent_paise(self, observation: Observation) -> int:
        gateway = self._calibration.gateway_cost_per_attempt_paise.value
        sms = self._calibration.sms_cost_paise.value
        return (
            len(observation.attempt_history) * gateway
            + observation.contacts_sent * sms
        )

    def decide(self, observation: Observation) -> Decision:
        """Never raises. A failure anywhere becomes the heuristic's decision."""
        self._decisions += 1
        try:
            return self._decide(observation)
        except Exception:  # noqa: BLE001 - an experiment must not die here
            return self._fall_back(observation, "unexpected_error")

    def _decide(self, observation: Observation) -> Decision:
        routed = self._router.diagnose(observation)
        if routed.source == "fallback":
            return self._fall_back(observation, "diagnosis")

        budget = Budget(
            spent_paise=self._spent_paise(observation),
            limits=self._validator.limits,
        )
        proposal = self._selector.select(observation, routed, budget)
        if proposal is None:
            return self._fall_back(observation, "intervention")

        rationale = f"{routed.rationale} {proposal.rationale}"

        # A nudge is only useful if there is something to send. The message
        # has its own verifier and its own static fallback, so a failure here
        # degrades the wording rather than the decision.
        if isinstance(proposal.action, SendNudge) and proposal.validated:
            message = self._messenger.generate(
                observation,
                reason=routed.diagnosis.value.lower().replace("_", " "),
            )
            if message.source == "fallback":
                self._fallbacks["messaging"] += 1
            rationale += (
                f' The customer will be sent ({message.source}, tone '
                f'{message.tone_level}): "{message.text}"'
            )

        self._llm_decisions += 1
        return self.decision(
            observation,
            proposal.action,
            rationale=rationale,
            source="llm",
            validated=proposal.validated,
        )
