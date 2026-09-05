"""Diagnosis, with the model on the residual only.

**Routing is the point of this module.** The rule-based code book from
`policies.heuristic` runs first and resolves whatever it honestly can. The
model is invoked *only* on what is left: generic codes, missing codes, and the
contradiction case where a funds code cannot be separated from a ceiling
breach.

That split is measured, not asserted. `DiagnosisRouter.stats()` reports
`llm_invocation_rate` — the share of failures that reached the model — as a
first-class metric. Being able to say "the model was called on 23% of events
and the other 77% resolved deterministically" is the evidence that judgment
was applied where judgment was needed, rather than a model being sprayed at a
problem rules already solve.

Two design constraints
----------------------
**The prompt receives only Observation fields.** A test renders the prompt for
observations carrying deliberately distinctive latent-looking values and
asserts none of them appear. Nothing here can reach the simulator.

**The prompt is canonical and bucketed.** A model call takes about five
seconds and a full experiment produces tens of thousands of residual failures,
so raw amounts would make the cache useless and the experiment impossible.
Rendering `amount_vs_history` as "larger than anything they have paid" rather
than "880000 against 500000" collapses thousands of observations onto a few
hundred prompts. The buckets are chosen to preserve exactly the distinctions
the diagnosis actually turns on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..policies.heuristic import (
    DIAGNOSIS_CODE_BOOK,
    UNINFORMATIVE_CODES,
    Diagnosis,
    diagnose as rule_diagnose,
)
from ..types import Observation
from .client import LLMFallback

__all__ = [
    "MINIMUM_CONFIDENCE",
    "DiagnosisReply",
    "RoutedDiagnosis",
    "DiagnosisRouter",
    "render_prompt",
    "load_prompt_template",
]

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "diagnosis.md"

#: Below this the model's answer is discarded and the failure is treated as
#: undiagnosed. A confident wrong diagnosis is worse than an honest UNKNOWN,
#: because it sends money and goodwill in the wrong direction.
MINIMUM_CONFIDENCE = 0.55


class DiagnosisReply(BaseModel):
    """The schema every diagnosis call must satisfy."""

    cause: Literal[
        "INSUFFICIENT_FUNDS", "TECHNICAL", "LIMIT", "WINDOW", "UNKNOWN"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


@dataclass(frozen=True)
class RoutedDiagnosis:
    """A diagnosis and, importantly, where it came from."""

    diagnosis: Diagnosis
    source: Literal["rule", "llm", "fallback"]
    confident: bool
    rationale: str


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    """The prompt, read from its versioned file with front matter stripped."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, _, body = text.split("---", 2)
        return body.strip()
    return text.strip()


# --------------------------------------------------------------------------
# Bucketing: what makes the cache work
# --------------------------------------------------------------------------


def _amount_vs_history(observation: Observation) -> str:
    ceiling = observation.max_historical_success_amount_paise
    if ceiling == 0:
        return "no settled payment from this customer to compare against"
    ratio = observation.amount_paise / ceiling
    if ratio <= 0.5:
        return "well below the largest amount they have paid before"
    if ratio <= 1.0:
        return "at or below the largest amount they have paid before"
    if ratio <= 2.0:
        return "somewhat larger than anything they have paid before"
    return "far larger than anything they have paid before"


def _attempts_bucket(observation: Observation) -> str:
    attempts = len(observation.attempt_history)
    if attempts <= 1:
        return "1 (this is the first failure this cycle)"
    if attempts == 2:
        return "2"
    return "3 or more"


def _history_bucket(observation: Observation) -> str:
    successes = observation.historical_success_count
    failures = observation.historical_failure_count
    if successes == 0 and failures <= 1:
        return "new customer, almost no history"
    if successes == 0:
        return "has never successfully paid us"
    if successes >= failures:
        return "pays reliably more often than not"
    return "fails more often than they pay"


def _days_bucket(observation: Observation) -> str:
    current_dom = (observation.current_day % 31) + 1
    remaining = (observation.due_day - current_dom) % 31 or 31
    if remaining <= 2:
        return "1-2 (about to lapse)"
    if remaining <= 7:
        return "3-7"
    return "more than a week"


def _code_meaning(code: str) -> str:
    if code in UNINFORMATIVE_CODES:
        return (
            "nothing specific -- this bank uses it for several unrelated "
            "causes, and some banks return it when no code was recorded"
        )
    known = DIAGNOSIS_CODE_BOOK.get(code)
    if known is Diagnosis.INSUFFICIENT_FUNDS:
        return (
            "a shortfall in the account -- but this bank also returns it for "
            "some ceiling breaches, so it is not conclusive"
        )
    if known is None:
        return "not a code this merchant has seen before"
    return known.value.lower().replace("_", " ")


def render_prompt(observation: Observation) -> str:
    """Render the canonical prompt. Only Observation fields may appear here."""
    code = (
        observation.attempt_history[-1].raw_code
        if observation.attempt_history
        else ""
    )
    return load_prompt_template().format(
        code=code or "(no code returned)",
        code_meaning=_code_meaning(code),
        amount_vs_history=_amount_vs_history(observation),
        attempts=_attempts_bucket(observation),
        history=_history_bucket(observation),
        days_to_lapse=_days_bucket(observation),
    )


# --------------------------------------------------------------------------
# The router
# --------------------------------------------------------------------------


class DiagnosisRouter:
    """Rules first; the model only on what the rules could not resolve."""

    def __init__(self, client) -> None:
        self._client = client
        self.rule_resolved = 0
        self.llm_invoked = 0
        self.llm_resolved = 0
        self.llm_low_confidence = 0
        self.llm_fallbacks = 0

    @property
    def total(self) -> int:
        return self.rule_resolved + self.llm_invoked

    @property
    def llm_invocation_rate(self) -> float:
        """Share of failures that reached the model. The headline routing metric."""
        return self.llm_invoked / self.total if self.total else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "diagnoses": self.total,
            "rule_resolved": self.rule_resolved,
            "llm_invoked": self.llm_invoked,
            "llm_resolved": self.llm_resolved,
            "llm_low_confidence": self.llm_low_confidence,
            "llm_fallbacks": self.llm_fallbacks,
            "llm_invocation_rate": self.llm_invocation_rate,
            "residual_resolution_rate": (
                self.llm_resolved / self.llm_invoked if self.llm_invoked else 0.0
            ),
        }

    def reset(self) -> None:
        self.rule_resolved = self.llm_invoked = self.llm_resolved = 0
        self.llm_low_confidence = self.llm_fallbacks = 0

    def diagnose(self, observation: Observation) -> RoutedDiagnosis:
        """Diagnose one failure, using the model only if the rules cannot."""
        ruled = rule_diagnose(observation)

        if ruled.diagnosis is not Diagnosis.UNKNOWN:
            self.rule_resolved += 1
            return RoutedDiagnosis(
                ruled.diagnosis, "rule", ruled.confident, ruled.rationale
            )

        self.llm_invoked += 1
        try:
            reply = self._client.complete(
                render_prompt(observation),
                DiagnosisReply,
                system_instruction=None,
            )
        except LLMFallback as error:
            self.llm_fallbacks += 1
            return RoutedDiagnosis(
                Diagnosis.UNKNOWN,
                "fallback",
                False,
                f"{ruled.rationale} The model was unavailable "
                f"({error}), so the failure stays undiagnosed.",
            )

        if reply.confidence < MINIMUM_CONFIDENCE:
            self.llm_low_confidence += 1
            return RoutedDiagnosis(
                Diagnosis.UNKNOWN,
                "llm",
                False,
                f"{ruled.rationale} The model offered "
                f"{reply.cause} at {reply.confidence:.0%} confidence, below "
                f"the {MINIMUM_CONFIDENCE:.0%} threshold, so it is discarded: "
                f"{reply.reasoning}",
            )

        self.llm_resolved += 1
        return RoutedDiagnosis(
            Diagnosis(reply.cause),
            "llm",
            True,
            f"{ruled.rationale} The model read it as {reply.cause} at "
            f"{reply.confidence:.0%} confidence: {reply.reasoning}",
        )
