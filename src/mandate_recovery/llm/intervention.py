"""Intervention selection: the model proposes, deterministic code disposes.

**This module is the clearest expression of invariant 6 in the codebase.** The
split is deliberate and worth stating precisely, because "the LLM decides"
would be both the easy implementation and the wrong one:

===========================  ===========================================
The model chooses            Deterministic code chooses
===========================  ===========================================
*Which kind* of action       The exact day and hour of a retry
*Roughly when* (soon /       The exact amount of a partial collection
after salary / next cycle)   Whether the action is permitted at all
The tone of a message        The rail, and whether a card exists
===========================  ===========================================

Nothing the model emits moves money. It cannot name a rupee figure — the
partial amount is computed from the customer's own settled history, not
proposed — it cannot pick a slot, and it cannot approve itself. A proposal
that violates a compliance rule is refused by the validator and replaced,
which a test drives directly: the model is told to nudge at 3am, and the
customer is not contacted.

That arrangement also happens to be the honest answer to prompt injection. A
response saying "collect ten lakh rupees immediately" parses into an enum
member and a timing preference, and then meets a function it cannot argue
with.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..agent.scheduler import SchedulerConstraints, next_retry_slot
from ..agent.validator import Budget, Validator
from ..policies.heuristic import Diagnosis
from ..types import (
    Action,
    CollectPartial,
    Decision,
    EscalateHuman,
    NudgeChannel,
    Observation,
    RetrySilent,
    SendNudge,
    Stop,
    SwitchRail,
    Rail,
)
from .client import LLMFallback
from .diagnosis import RoutedDiagnosis, _amount_vs_history, _attempts_bucket
from .diagnosis import _days_bucket, _history_bucket

__all__ = [
    "InterventionReply",
    "ProposedIntervention",
    "InterventionSelector",
    "render_prompt",
]

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "intervention.md"

#: How the model's timing preference becomes a search window. "Soon" narrows
#: the horizon; waiting for salary opens it up so the payday prior can win.
_TIMING_HORIZON_DAYS = {"SOON": 3, "AFTER_NEXT_SALARY": 16}


class InterventionReply(BaseModel):
    """The schema every intervention call must satisfy.

    Note what is absent: no amount, no day, no hour, no rail.
    """

    action: Literal[
        "RETRY_SILENT",
        "SEND_NUDGE",
        "COLLECT_PARTIAL",
        "SWITCH_RAIL",
        "ESCALATE",
        "STOP",
    ]
    timing: Literal["SOON", "AFTER_NEXT_SALARY", "NEXT_CYCLE"]
    tone_level: int = Field(ge=1, le=3, default=1)
    reasoning: str


@dataclass(frozen=True)
class ProposedIntervention:
    """What will actually be executed, and how it got there."""

    action: Action
    source: Literal["llm", "fallback"]
    rationale: str
    validated: bool
    proposed_action: Action
    validator_rule: str | None = None
    validator_reason: str = ""


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, _, body = text.split("---", 2)
        return body.strip()
    return text.strip()


def _budget_bucket(budget: Budget) -> str:
    remaining = budget.remaining_paise()
    cap = budget.limits.max_cost_paise_per_mandate or 1
    share = remaining / cap
    if remaining == 0:
        return "none left, no further spend is authorised"
    if share < 0.34:
        return "nearly exhausted"
    if share < 0.67:
        return "about half remaining"
    return "most of it remaining"


def render_prompt(
    observation: Observation, diagnosis: RoutedDiagnosis, budget: Budget
) -> str:
    """Canonical, bucketed, and built only from Observation fields."""
    code = (
        observation.attempt_history[-1].raw_code
        if observation.attempt_history
        else ""
    )
    return load_prompt_template().format(
        diagnosis=diagnosis.diagnosis.value,
        diagnosis_confidence=(
            "confident" if diagnosis.confident else "not confident"
        ),
        code=code or "(no code returned)",
        attempts=_attempts_bucket(observation),
        contacts=min(observation.contacts_sent, 3),
        amount_vs_history=_amount_vs_history(observation),
        history=_history_bucket(observation),
        days_to_lapse=_days_bucket(observation),
        budget=_budget_bucket(budget),
    )


class InterventionSelector:
    """Turns a model proposal into a validated, executable action."""

    def __init__(
        self,
        client,
        validator: Validator | None = None,
        constraints: SchedulerConstraints | None = None,
    ) -> None:
        self._client = client
        self._validator = validator or Validator()
        self._constraints = constraints or SchedulerConstraints()
        self.proposals = 0
        self.fallbacks = 0
        self.refused_by_validator = 0

    @property
    def validator(self) -> Validator:
        return self._validator

    def stats(self) -> dict[str, int]:
        return {
            "proposals": self.proposals,
            "fallbacks": self.fallbacks,
            "refused_by_validator": self.refused_by_validator,
        }

    def reset(self) -> None:
        self.proposals = self.fallbacks = self.refused_by_validator = 0
        self._validator.reset()

    # ------------------------------------------------------------------

    def select(
        self,
        observation: Observation,
        diagnosis: RoutedDiagnosis,
        budget: Budget,
    ) -> ProposedIntervention | None:
        """Propose, translate, validate. Returns None if the model was no help.

        ``None`` means the caller should use its deterministic path; the
        model layer never leaves an experiment without an action.
        """
        try:
            reply = self._client.complete(
                render_prompt(observation, diagnosis, budget),
                InterventionReply,
                system_instruction=None,
            )
        except LLMFallback:
            self.fallbacks += 1
            return None

        self.proposals += 1
        proposed = self._translate(reply, observation)
        verdict = self._validator.validate(proposed, observation, budget)

        rationale = f"Model proposed {reply.action} ({reply.timing}): {reply.reasoning}"
        if not verdict.approved:
            self.refused_by_validator += 1
            rationale += f" The compliance gate refused it: {verdict.reason}"
        elif verdict.was_substituted:
            rationale += f" {verdict.reason}"

        return ProposedIntervention(
            action=verdict.action,
            source="llm",
            rationale=rationale,
            validated=verdict.approved,
            proposed_action=proposed,
            validator_rule=verdict.rule,
            validator_reason=verdict.reason,
        )

    # ------------------------------------------------------------------

    def _translate(
        self, reply: InterventionReply, observation: Observation
    ) -> Action:
        """Turn an enum choice into a concrete action. All money set here."""
        if reply.action == "STOP" or reply.timing == "NEXT_CYCLE":
            return Stop(reason="model advised no further action this cycle")

        if reply.action == "SEND_NUDGE":
            # The hour is left to the validator, which defers to the first
            # permitted contact hour rather than waking anyone.
            return SendNudge(
                channel=NudgeChannel.SMS,
                tone_level=max(1, min(3, reply.tone_level)),
            )

        if reply.action == "COLLECT_PARTIAL":
            # The amount is NOT the model's to choose. It is the largest sum
            # this customer has actually settled, which is the only figure
            # with evidence behind it.
            ceiling = observation.max_historical_success_amount_paise
            if 0 < ceiling < observation.amount_paise:
                return CollectPartial(amount_paise=ceiling)
            return Stop(reason="no settled payment to size a partial against")

        if reply.action == "SWITCH_RAIL":
            return SwitchRail(target_rail=Rail.CARD)

        if reply.action == "ESCALATE":
            return EscalateHuman(reason="model escalated after repeated failure")

        slot = next_retry_slot(
            observation,
            SchedulerConstraints(
                restricted_windows=self._constraints.restricted_windows,
                horizon_days=_TIMING_HORIZON_DAYS.get(
                    reply.timing, self._constraints.horizon_days
                ),
                cooling_off_days=self._constraints.cooling_off_days,
                allowed_rails=self._constraints.allowed_rails,
            ),
        )
        if slot is None:
            return Stop(reason="no compliant retry slot available")
        return RetrySilent(
            scheduled_day=slot.day,
            scheduled_hour=slot.hour,
            rail=self._constraints.allowed_rails[0],
        )
