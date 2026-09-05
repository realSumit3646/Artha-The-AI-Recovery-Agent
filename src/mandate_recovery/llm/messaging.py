"""Hinglish nudges, with every fact templated and verified.

**A hallucinated rupee figure in a payment message is a fintech red line.**
This module is built so the model never has the opportunity: it is not shown
the amount, the date, the reference or the merchant name. It writes a
*template* containing four placeholders, and Python substitutes the true
values afterwards.

That is stronger than checking the model's arithmetic. A model that cannot see
a number cannot get it wrong, and the verifier's job reduces to something
absolute: **the template must contain no digits at all.** Not "the right
digits" — none. There is no partial credit and no judgement call.

Two verification passes
-----------------------
1. **On the template**, before substitution: all four placeholders present
   exactly once, no digits anywhere, no forbidden language.
2. **On the rendered message**, after substitution: every fact appears
   verbatim, and every digit sequence in the message traces back to the fact
   set. This catches a substitution bug as well as a model failure.

Either failure discards the message, sends a static template instead, and
increments a counter that the reliability figure reports. Falling back is not
an error path to be embarrassed about; it is the designed behaviour.

The cache, incidentally, loves this. Because the model never sees real values,
the prompt varies only by tone and reason — a couple of dozen distinct prompts
for the entire experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from ..types import Observation
from .client import LLMFallback

__all__ = [
    "MessageFacts",
    "MessageReply",
    "GeneratedMessage",
    "MessageGenerator",
    "PLACEHOLDERS",
    "verify_template",
    "verify_rendered",
    "static_message",
]

PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "message_hinglish.md"
)

#: Exactly these, exactly once each.
PLACEHOLDERS = ("{amount}", "{due_date}", "{reference}", "{merchant}")

_DIGITS = re.compile(r"\d+")
_DIGIT_RUN = re.compile(r"\d[\d,.]*\d|\d")

#: Language a payment reminder must never contain, at any tone.
_FORBIDDEN = (
    "legal action",
    "lawyer",
    "court",
    "police",
    "credit score",
    "cibil",
    "recovery agent",
    "blacklist",
    "defaulter",
)

#: Sent when the model fails verification or is unavailable. Deliberately
#: plain: it is a safety net, not a product.
_STATIC_TEMPLATES = {
    1: "Hi! Aapka {amount} ka payment {due_date} ko process nahi ho paya. "
       "Please account mein balance rakhein - {merchant} (Ref {reference}).",
    2: "Reminder: {amount} ka payment {due_date} se pending hai aur dobara "
       "fail hua hai. Please jaldi balance rakhein - {merchant} "
       "(Ref {reference}).",
    3: "Final notice: {amount} ka payment {due_date} se pending hai. Balance "
       "na hone par aapka mandate lapse ho sakta hai - {merchant} "
       "(Ref {reference}).",
}


@dataclass(frozen=True)
class MessageFacts:
    """The four facts the model is never shown."""

    amount_text: str
    due_date_text: str
    reference: str
    merchant_name: str

    @classmethod
    def from_observation(
        cls, observation: Observation, merchant_name: str = "Artha Payments"
    ) -> "MessageFacts":
        rupees = observation.amount_paise / 100
        return cls(
            amount_text=f"Rs {rupees:,.2f}",
            due_date_text=f"{observation.due_day}{_ordinal(observation.due_day)}",
            reference=observation.mandate_id.upper(),
            merchant_name=merchant_name,
        )

    def as_mapping(self) -> dict[str, str]:
        return {
            "amount": self.amount_text,
            "due_date": self.due_date_text,
            "reference": self.reference,
            "merchant": self.merchant_name,
        }

    def digit_runs(self) -> set[str]:
        """Every digit sequence that is legitimately allowed to appear."""
        runs: set[str] = set()
        for value in self.as_mapping().values():
            runs.update(_DIGIT_RUN.findall(value))
        return runs


class MessageReply(BaseModel):
    """What the model returns: a template, never a finished message."""

    message_template: str


@dataclass(frozen=True)
class GeneratedMessage:
    """The message that will actually be sent, and where it came from."""

    text: str
    source: str  # "llm" | "fallback"
    tone_level: int
    verification_error: str | None = None


def _ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, _, body = text.split("---", 2)
        return body.strip()
    return text.strip()


def render_prompt(tone_level: int, reason: str) -> str:
    """Canonical: varies only by tone and reason, so the cache is tiny."""
    return load_prompt_template().format(tone_level=tone_level, reason=reason)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_template(template: str) -> str | None:
    """Check the model's template. Returns an error string, or None if clean."""
    if not template.strip():
        return "the model returned an empty message"

    for placeholder in PLACEHOLDERS:
        count = template.count(placeholder)
        if count == 0:
            return f"missing the {placeholder} placeholder"
        if count > 1:
            return f"{placeholder} appears {count} times"

    # The absolute rule. A template with no digits cannot carry a wrong one.
    stripped = template
    for placeholder in PLACEHOLDERS:
        stripped = stripped.replace(placeholder, "")
    if _DIGITS.search(stripped):
        found = _DIGITS.findall(stripped)
        return f"the model wrote digits it was not given: {found}"

    lowered = template.lower()
    for phrase in _FORBIDDEN:
        if phrase in lowered:
            return f"the message contains forbidden language: {phrase!r}"

    unknown = set(re.findall(r"\{(\w+)\}", template)) - {
        p.strip("{}") for p in PLACEHOLDERS
    }
    if unknown:
        return f"unknown placeholders: {sorted(unknown)}"

    return None


def verify_rendered(message: str, facts: MessageFacts) -> str | None:
    """Check the finished message. Catches a substitution bug too."""
    for name, value in facts.as_mapping().items():
        if value not in message:
            return f"the {name} fact does not appear verbatim in the message"

    allowed = facts.digit_runs()
    for run in _DIGIT_RUN.findall(message):
        if run not in allowed:
            return f"the message contains an unsourced number: {run!r}"

    if "{" in message or "}" in message:
        return "an unsubstituted placeholder remains in the message"

    return None


def static_message(facts: MessageFacts, tone_level: int) -> str:
    """The safety net. Always passes verification by construction."""
    template = _STATIC_TEMPLATES[max(1, min(3, tone_level))]
    return template.format(**facts.as_mapping())


# --------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------


class MessageGenerator:
    """Generates a nudge, or falls back to a static template and says so."""

    def __init__(self, client, merchant_name: str = "Artha Payments") -> None:
        self._client = client
        self._merchant_name = merchant_name
        self.generated = 0
        self.fallbacks = 0
        self.verification_failures = 0

    def stats(self) -> dict[str, int]:
        return {
            "messages_generated": self.generated,
            "message_fallbacks": self.fallbacks,
            "message_verification_failures": self.verification_failures,
        }

    def reset(self) -> None:
        self.generated = self.fallbacks = self.verification_failures = 0

    @staticmethod
    def tone_for(contacts_sent: int) -> int:
        """Tone rises with insistence, and stops at a firm notice."""
        return max(1, min(3, contacts_sent + 1))

    def generate(
        self,
        observation: Observation,
        reason: str = "the payment could not be collected",
        tone_level: int | None = None,
    ) -> GeneratedMessage:
        facts = MessageFacts.from_observation(observation, self._merchant_name)
        tone = tone_level or self.tone_for(observation.contacts_sent)

        try:
            reply = self._client.complete(
                render_prompt(tone, reason),
                MessageReply,
                system_instruction=None,
            )
        except LLMFallback as error:
            self.fallbacks += 1
            return GeneratedMessage(
                static_message(facts, tone), "fallback", tone, str(error)
            )

        error = verify_template(reply.message_template)
        if error is None:
            rendered = reply.message_template.format(**facts.as_mapping())
            error = verify_rendered(rendered, facts)
            if error is None:
                self.generated += 1
                return GeneratedMessage(rendered, "llm", tone)

        self.verification_failures += 1
        self.fallbacks += 1
        return GeneratedMessage(
            static_message(facts, tone), "fallback", tone, error
        )
