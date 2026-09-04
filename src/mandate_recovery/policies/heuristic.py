"""A competent deterministic agent. No language model anywhere in it.

This is the ablation control, and it decides what this project's honest claim
turns out to be. If it beats the LLM agent at commit 26, that is a finding to
report — that recovery is a scheduling problem and the model adds nothing —
not a failure to engineer around. **It is therefore built in good faith and
deliberately not handicapped.**

The pipeline
------------
``diagnose`` (rules) -> ``select`` (decision table) -> ``schedule``
(deterministic) -> ``validate`` (compliance gate) -> act.

Diagnosis is a lookup over the bank's response code, plus two disambiguation
rules for the contradiction case where a limit breach is reported with a funds
code. When the code is generic or missing, diagnosis returns ``UNKNOWN``
rather than guessing, and the **UNKNOWN rate is measured and exposed** —
`diagnosis_counts` — because that number is the argument for adding a model
stage later. A rules-only agent that claimed to diagnose everything would be
lying about the hard part.

What the agent knows
--------------------
Its code book is its own, built from settlement reports, and this module
imports nothing from ``mandate_recovery.sim``. It reads the *calibrated cost
figures*, which is legitimate: a merchant knows their own gateway fees and SMS
prices. It does not read failure rates, balances, or anything about the
customer beyond the observation.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import ClassVar, Mapping

from ..agent.scheduler import SchedulerConstraints, next_retry_slot
from ..agent.validator import Budget, ComplianceLimits, Validator
from ..calibration import DEFAULT_CALIBRATION, CalibrationSet
from ..types import (
    CollectPartial,
    Decision,
    NudgeChannel,
    Observation,
    RetrySilent,
    SendNudge,
    Stop,
)
from .base import Policy

__all__ = ["Diagnosis", "DIAGNOSIS_CODE_BOOK", "diagnose", "HeuristicPolicy"]

DAYS_IN_MONTH = 31


class Diagnosis(str, Enum):
    """What the agent believes went wrong."""

    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    TECHNICAL = "TECHNICAL"
    LIMIT = "LIMIT"
    WINDOW = "WINDOW"
    REVOKED = "REVOKED"
    UNKNOWN = "UNKNOWN"


#: The agent's own code book, learned from its settlement reports. Not
#: imported from the simulator; a test asserts the two agree.
DIAGNOSIS_CODE_BOOK: Mapping[str, Diagnosis] = {
    "AB1200": Diagnosis.INSUFFICIENT_FUNDS,
    "PS-51": Diagnosis.INSUFFICIENT_FUNDS,
    "SF_NOFUNDS": Diagnosis.INSUFFICIENT_FUNDS,
    "AB9001": Diagnosis.TECHNICAL,
    "PS-91": Diagnosis.TECHNICAL,
    "SF_SYSERR": Diagnosis.TECHNICAL,
    "AB3301": Diagnosis.LIMIT,
    "PS-61": Diagnosis.LIMIT,
    "SF_CAP": Diagnosis.LIMIT,
    "AB7702": Diagnosis.WINDOW,
    "PS-77": Diagnosis.WINDOW,
    "SF_PEAK": Diagnosis.WINDOW,
    "AB6600": Diagnosis.REVOKED,
    "PS-14": Diagnosis.REVOKED,
    "SF_CANCELLED": Diagnosis.REVOKED,
}

#: Codes that identify nothing. Everything else the agent has never seen also
#: lands here.
UNINFORMATIVE_CODES = frozenset({"", "NA", "DECLINED"})


class DiagnosisResult:
    """A diagnosis, whether it is trusted, and why."""

    __slots__ = ("diagnosis", "confident", "rationale")

    def __init__(self, diagnosis: Diagnosis, confident: bool, rationale: str):
        self.diagnosis = diagnosis
        self.confident = confident
        self.rationale = rationale

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Diagnosis {self.diagnosis.value} confident={self.confident}>"


def diagnose(observation: Observation) -> DiagnosisResult:
    """Read the last response code, and correct it where history contradicts it.

    Two disambiguation rules handle the contradiction case, where a limit
    breach is reported with a funds code:

    1. If the customer has previously been collected at **this amount or
       more**, their ceiling clearly permits it, so a funds code means what it
       says.
    2. If the largest amount ever collected from them is **below** this one, a
       funds code is unreliable — the ceiling is a live possibility and the
       response cannot distinguish them.
    """
    if not observation.attempt_history:
        return DiagnosisResult(
            Diagnosis.UNKNOWN, False, "No attempt has been made yet."
        )

    code = observation.attempt_history[-1].raw_code

    if code in UNINFORMATIVE_CODES:
        return DiagnosisResult(
            Diagnosis.UNKNOWN,
            False,
            f"The bank returned {code!r}, which several different causes "
            "share. The failure cannot be diagnosed from the code.",
        )

    diagnosis = DIAGNOSIS_CODE_BOOK.get(code)
    if diagnosis is None:
        return DiagnosisResult(
            Diagnosis.UNKNOWN,
            False,
            f"The code {code!r} is not in the code book.",
        )

    if diagnosis is not Diagnosis.INSUFFICIENT_FUNDS:
        return DiagnosisResult(
            diagnosis,
            True,
            f"The bank returned {code!r}, which means "
            f"{diagnosis.value.lower().replace('_', ' ')}.",
        )

    ceiling_evidence = observation.max_historical_success_amount_paise

    if ceiling_evidence >= observation.amount_paise:
        return DiagnosisResult(
            Diagnosis.INSUFFICIENT_FUNDS,
            True,
            f"The bank returned {code!r}. This customer has previously paid "
            f"{ceiling_evidence} paise, at or above the {observation.amount_paise} "
            "being collected, so their ceiling permits this amount and the "
            "funds reading is trustworthy.",
        )

    if ceiling_evidence > 0:
        return DiagnosisResult(
            Diagnosis.UNKNOWN,
            False,
            f"The bank returned {code!r}, but the most this customer has ever "
            f"paid is {ceiling_evidence} paise, below the "
            f"{observation.amount_paise} being collected. A per-transaction "
            "ceiling would look identical, so the cause is undetermined.",
        )

    return DiagnosisResult(
        Diagnosis.INSUFFICIENT_FUNDS,
        False,
        f"The bank returned {code!r}. There is no settled payment from this "
        "customer to compare the amount against, so the funds reading is "
        "taken at face value but is not confirmed.",
    )


class HeuristicPolicy(Policy):
    """Rule-based diagnosis, a decision table, then scheduling and validation."""

    name: ClassVar[str] = "heuristic"

    def __init__(
        self,
        calibration: CalibrationSet = DEFAULT_CALIBRATION,
        constraints: SchedulerConstraints | None = None,
        limits: ComplianceLimits | None = None,
    ) -> None:
        self._calibration = calibration
        self._constraints = constraints or SchedulerConstraints()
        self._validator = Validator(limits)
        self._diagnosis_counts: Counter = Counter()

    # ------------------------------------------------------------------

    @property
    def validator(self) -> Validator:
        return self._validator

    @property
    def diagnosis_counts(self) -> Mapping[str, int]:
        return dict(self._diagnosis_counts)

    @property
    def unknown_diagnosis_rate(self) -> float:
        """Share of failures the code book could not resolve.

        The argument for a model stage, stated as a number rather than a
        feeling.
        """
        total = sum(self._diagnosis_counts.values())
        if total == 0:
            return 0.0
        return self._diagnosis_counts[Diagnosis.UNKNOWN.value] / total

    def reset(self) -> None:
        self._diagnosis_counts.clear()
        self._validator.reset()

    # ------------------------------------------------------------------

    def _spent_paise(self, observation: Observation) -> int:
        """What this mandate has cost so far, at the merchant's own prices."""
        gateway = self._calibration.gateway_cost_per_attempt_paise.value
        sms = self._calibration.sms_cost_paise.value
        return (
            len(observation.attempt_history) * gateway
            + observation.contacts_sent * sms
        )

    @staticmethod
    def _days_to_lapse(observation: Observation) -> int:
        """Days until this cycle is superseded by the next one."""
        current_dom = (observation.current_day % DAYS_IN_MONTH) + 1
        remaining = (observation.due_day - current_dom) % DAYS_IN_MONTH
        return remaining or DAYS_IN_MONTH

    def decide(self, observation: Observation) -> Decision:
        result = diagnose(observation)
        self._diagnosis_counts[result.diagnosis.value] += 1

        proposed, why = self._select(observation, result)
        budget = Budget(
            spent_paise=self._spent_paise(observation),
            limits=self._validator.limits,
        )
        verdict = self._validator.validate(proposed, observation, budget)

        rationale = f"{result.rationale} {why}"
        if not verdict.approved:
            rationale += f" The compliance gate refused this: {verdict.reason}"
        elif verdict.was_substituted:
            rationale += f" {verdict.reason}"

        return self.decision(
            observation,
            verdict.action,
            rationale=rationale,
            source="rule",
            validated=verdict.approved,
        )

    # ------------------------------------------------------------------

    def _select(self, observation: Observation, result: DiagnosisResult):
        """The decision table over diagnosis, attempts, contacts and lapse."""
        attempts = len(observation.attempt_history)
        contacts = observation.contacts_sent
        days_left = self._days_to_lapse(observation)

        if result.diagnosis is Diagnosis.REVOKED:
            return (
                Stop(reason="mandate revoked"),
                "A revoked mandate cannot be collected, so nothing further is "
                "attempted.",
            )

        if result.diagnosis is Diagnosis.LIMIT:
            ceiling = observation.max_historical_success_amount_paise
            if 0 < ceiling < observation.amount_paise:
                return (
                    CollectPartial(amount_paise=ceiling),
                    f"The ceiling is the binding constraint, so a partial "
                    f"{ceiling} paise is attempted at the largest amount this "
                    "customer has actually paid before.",
                )
            return (
                Stop(reason="amount exceeds the ceiling with no known lower bound"),
                "The amount is above the customer's per-transaction ceiling "
                "and there is no settled payment to size a partial against.",
            )

        # Everything below is a timing or persuasion problem.
        if days_left <= 1:
            return (
                Stop(reason="cycle is about to lapse"),
                f"Only {days_left} day(s) remain before the next cycle "
                "supersedes this one, so further spend would not pay back.",
            )

        wants_contact = (
            result.diagnosis
            in (Diagnosis.INSUFFICIENT_FUNDS, Diagnosis.UNKNOWN)
            and attempts >= 2
            and contacts == 0
            and days_left > 2
        )
        if wants_contact:
            return (
                SendNudge(channel=NudgeChannel.SMS, tone_level=1),
                f"Two silent retries have failed and the customer has not "
                "been contacted, so a single low-tone reminder is sent before "
                "spending further gateway fees.",
            )

        slot = next_retry_slot(observation, self._constraints)
        if slot is None:
            return (
                Stop(reason="no compliant retry slot available"),
                "No slot inside the horizon clears the restricted window and "
                "the cooling-off period.",
            )
        return (
            RetrySilent(
                scheduled_day=slot.day,
                scheduled_hour=slot.hour,
                rail=self._constraints.allowed_rails[0],
            ),
            slot.rationale,
        )
