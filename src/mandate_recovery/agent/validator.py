"""The compliance gate. Every action passes through here before execution.

This layer is never bypassed. Not by the heuristic agent, not by the LLM agent
that arrives later, not by a policy that is very confident. A model may
propose; this module approves, corrects, or refuses — and it is deterministic
arithmetic, so what it did can be replayed from a stored config.

That separation is invariant 6 made concrete: **no money-moving or
compliance-gating decision is made by a language model.** The interesting
consequence is that the LLM arm cannot break the rules even if a prompt
injection tells it to, because nothing it emits reaches the rail without
passing a function it cannot see.

The rules
---------
======================================  ==================================
Rule                                    On violation
======================================  ==================================
Max attempts per mandate per cycle       Stop
Minimum hours between attempts           slot corrected forward
Max contacts per rolling 7 days          Stop
No contact outside 09:00-21:00           Stop
Cumulative cost budget per mandate       Stop
No silent retry in the restricted window slot corrected to a clear hour
SwitchRail to CARD without a card        Stop
======================================  ==================================

Two of these correct rather than refuse. A retry that lands an hour too early
or inside the NPCI window is a *timing* error with an obviously right answer,
and refusing it would throw away a recovery to punish a rounding mistake.
Everything else refuses outright: there is no safe correction for "you have
contacted this customer too many times this week".

Every rejection is counted by reason, so the ablation can report how often
each arm had to be stopped rather than only how well it scored.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

from ..types import (
    Action,
    CollectPartial,
    EscalateHuman,
    Observation,
    Rail,
    RetrySilent,
    SendNudge,
    Stop,
    SwitchRail,
)

__all__ = [
    "ComplianceLimits",
    "Budget",
    "ValidationResult",
    "Validator",
    "validate",
]

HOURS_IN_DAY = 24


@dataclass(frozen=True)
class ComplianceLimits:
    """The hard limits. Every one of these is an author's assumption.

    NPCI does cap re-presentment of a failed recurring mandate, and Indian
    telecom rules do restrict outbound commercial contact hours, but the exact
    numbers here are not taken from a published circular and are not presented
    as such. They are the shape of the constraint, not its citation.
    """

    max_attempts_per_cycle: int = 4
    min_hours_between_attempts: int = 12
    max_contacts_per_7_days: int = 3
    earliest_contact_hour: int = 9
    latest_contact_hour: int = 21  # exclusive: 21:00 is already too late
    max_cost_paise_per_mandate: int = 2_000
    restricted_windows: tuple[tuple[int, int], ...] = ((10, 13), (17, 21))

    def is_restricted(self, hour: int) -> bool:
        return any(start <= hour < end for start, end in self.restricted_windows)

    def first_clear_hour_at_or_after(self, hour: int) -> int | None:
        """The next hour outside every restricted window, same day."""
        for candidate in range(hour, HOURS_IN_DAY):
            if not self.is_restricted(candidate):
                return candidate
        return None


@dataclass(frozen=True)
class Budget:
    """What this mandate has already consumed, plus the limits it runs under."""

    spent_paise: int = 0
    limits: ComplianceLimits = field(default_factory=ComplianceLimits)

    def remaining_paise(self) -> int:
        return max(0, self.limits.max_cost_paise_per_mandate - self.spent_paise)

    def is_exhausted(self) -> bool:
        return self.spent_paise >= self.limits.max_cost_paise_per_mandate


@dataclass(frozen=True)
class ValidationResult:
    """The gate's verdict.

    ``approved`` describes the *returned* action. When a slot was corrected the
    result is approved and ``substituted_action`` holds the corrected form, so
    a caller that executes ``result.action`` is always compliant.
    """

    approved: bool
    reason: str
    action: Action
    substituted_action: Action | None = None
    rule: str | None = None

    @property
    def was_substituted(self) -> bool:
        return self.substituted_action is not None


class Validator:
    """Applies the compliance rules and counts every rejection by reason."""

    def __init__(self, limits: ComplianceLimits | None = None) -> None:
        self._limits = limits or ComplianceLimits()
        self._rejections: Counter = Counter()
        self._substitutions: Counter = Counter()

    @property
    def limits(self) -> ComplianceLimits:
        return self._limits

    @property
    def rejections(self) -> Mapping[str, int]:
        """How many times each rule refused an action."""
        return dict(self._rejections)

    @property
    def substitutions(self) -> Mapping[str, int]:
        """How many times each rule corrected an action instead of refusing."""
        return dict(self._substitutions)

    @property
    def total_rejections(self) -> int:
        return sum(self._rejections.values())

    def reset(self) -> None:
        self._rejections.clear()
        self._substitutions.clear()

    # ------------------------------------------------------------------

    def validate(
        self,
        action: Action,
        observation: Observation,
        budget: Budget | None = None,
    ) -> ValidationResult:
        """Approve, correct, or refuse one proposed action."""
        budget = budget or Budget(limits=self._limits)
        limits = self._limits

        # Stopping is always allowed. Refusing to refuse would be absurd.
        if isinstance(action, Stop):
            return ValidationResult(True, "stopping is always permitted", action)

        if budget.is_exhausted():
            return self._refuse(
                "cost_budget_exhausted",
                action,
                f"This mandate has spent {budget.spent_paise} paise against a "
                f"cap of {limits.max_cost_paise_per_mandate}. Further spend "
                "is not authorised.",
            )

        if isinstance(action, (RetrySilent, CollectPartial, SwitchRail)):
            return self._validate_debit(action, observation, budget)
        if isinstance(action, (SendNudge, EscalateHuman)):
            return self._validate_contact(action, observation)

        return ValidationResult(True, "no rule applies to this action", action)

    # ------------------------------------------------------------------

    def _validate_debit(
        self, action: Action, observation: Observation, budget: Budget
    ) -> ValidationResult:
        limits = self._limits
        attempts = len(observation.attempt_history)

        if attempts >= limits.max_attempts_per_cycle:
            return self._refuse(
                "attempt_cap_reached",
                action,
                f"{attempts} attempts have already been made this cycle "
                f"against a cap of {limits.max_attempts_per_cycle}.",
            )

        if isinstance(action, SwitchRail):
            if action.target_rail is Rail.CARD and not observation.has_card_on_file:
                return self._refuse(
                    "no_card_on_file",
                    action,
                    "Cannot move this mandate to card: the customer has no "
                    "card on file.",
                )
            return ValidationResult(True, "rail switch permitted", action)

        if not isinstance(action, RetrySilent):
            return ValidationResult(True, "debit permitted", action)

        corrected_day = action.scheduled_day
        corrected_hour = action.scheduled_hour
        notes: list[str] = []
        # Both corrections can fire on one action -- pushing a retry forward
        # can land it inside the window. Each is counted separately, or the
        # substitution counts would silently under-report the first one.
        applied: list[str] = []

        # Minimum spacing, measured from the last attempt.
        if observation.attempt_history:
            last = observation.attempt_history[-1]
            elapsed_hours = (corrected_day - last.day) * HOURS_IN_DAY + (
                corrected_hour - last.hour
            )
            if elapsed_hours < limits.min_hours_between_attempts:
                shortfall = limits.min_hours_between_attempts - elapsed_hours
                total = corrected_day * HOURS_IN_DAY + corrected_hour + shortfall
                corrected_day, corrected_hour = divmod(total, HOURS_IN_DAY)
                notes.append(
                    f"pushed forward {shortfall}h to respect the "
                    f"{limits.min_hours_between_attempts}h minimum gap"
                )
                applied.append("min_gap")

        # Defence in depth: the scheduler already excludes the window.
        if limits.is_restricted(corrected_hour):
            clear = limits.first_clear_hour_at_or_after(corrected_hour)
            if clear is None:
                corrected_day += 1
                clear = limits.first_clear_hour_at_or_after(0)
            notes.append(
                f"moved out of the NPCI restricted window to {clear:02d}:00"
            )
            applied.append("restricted_window")
            corrected_hour = clear

        if not notes:
            return ValidationResult(True, "retry permitted", action)

        substitute = RetrySilent(
            scheduled_day=corrected_day,
            scheduled_hour=corrected_hour,
            rail=action.rail,
        )
        for rule in applied:
            self._substitutions[rule] += 1
        return ValidationResult(
            approved=True,
            reason="Retry corrected before approval: " + "; ".join(notes) + ".",
            action=substitute,
            substituted_action=substitute,
            rule="+".join(applied),
        )

    def _validate_contact(
        self, action: Action, observation: Observation
    ) -> ValidationResult:
        limits = self._limits

        if observation.contacts_in_last_7_days >= limits.max_contacts_per_7_days:
            return self._refuse(
                "contact_cap_reached",
                action,
                f"{observation.contacts_in_last_7_days} contacts in the last "
                f"7 days against a cap of {limits.max_contacts_per_7_days}.",
            )

        hour = observation.current_hour
        if not limits.earliest_contact_hour <= hour < limits.latest_contact_hour:
            return self._refuse(
                "outside_contact_hours",
                action,
                f"It is {hour:02d}:00. Customers may only be contacted "
                f"between {limits.earliest_contact_hour:02d}:00 and "
                f"{limits.latest_contact_hour:02d}:00.",
            )

        return ValidationResult(True, "contact permitted", action)

    def _refuse(self, rule: str, action: Action, reason: str) -> ValidationResult:
        """Refuse an action and substitute a stop."""
        self._rejections[rule] += 1
        substitute = Stop(reason=rule.replace("_", " "))
        return ValidationResult(
            approved=False,
            reason=reason,
            action=substitute,
            substituted_action=substitute,
            rule=rule,
        )


def validate(
    action: Action,
    observation: Observation,
    budget: Budget | None = None,
    *,
    limits: ComplianceLimits | None = None,
) -> ValidationResult:
    """One-off validation without keeping counters. See :class:`Validator`."""
    return Validator(limits).validate(action, observation, budget)
