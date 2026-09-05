"""Tests for Hinglish message generation and fact verification.

These are not optional. A hallucinated rupee figure in a payment message is a
fintech red line, so the adversarial cases below — a model that invents an
amount, doubles a placeholder, or threatens the customer — are the point of
the file rather than edge cases appended to it.
"""

from __future__ import annotations

import pytest

from mandate_recovery.llm.client import StubClient
from mandate_recovery.llm.messaging import (
    PLACEHOLDERS,
    GeneratedMessage,
    MessageFacts,
    MessageGenerator,
    MessageReply,
    render_prompt,
    static_message,
    verify_rendered,
    verify_template,
)
from mandate_recovery.types import Observation

GOOD = (
    "Hi! Aapka {amount} ka payment {due_date} ko process nahi ho paya. "
    "Please balance rakhein - {merchant} (Ref {reference})."
)


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m000123",
        "amount_paise": 880_000,
        "due_day": 5,
        "current_day": 10,
        "current_hour": 10,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def _facts(**overrides) -> MessageFacts:
    return MessageFacts.from_observation(_observation(**overrides))


def _generator(template: str = GOOD, **kwargs) -> MessageGenerator:
    return MessageGenerator(
        StubClient({"MessageReply": MessageReply(message_template=template)}),
        **kwargs,
    )


# --------------------------------------------------------------------------
# The red line: the model never sees, and never writes, a number
# --------------------------------------------------------------------------


def test_the_prompt_never_contains_the_real_facts():
    """The model cannot hallucinate an amount it was never shown."""
    prompt = render_prompt(1, "insufficient funds")
    assert "880000" not in prompt
    assert "8,800" not in prompt
    assert "m000123" not in prompt.lower()


@pytest.mark.parametrize(
    "bad_template",
    [
        "Aapka Rs 5000 ka payment {due_date} {amount} {reference} {merchant}",
        "{amount} {due_date} {reference} {merchant} - pay within 24 hours",
        "{amount} {due_date} {reference} {merchant} call 1800123456",
        "{amount} {due_date} {reference} {merchant} - 2 din mein karein",
    ],
)
def test_a_template_containing_any_digit_is_rejected(bad_template):
    """Not 'the wrong digits' — any digit at all. No partial credit."""
    error = verify_template(bad_template)
    assert error is not None
    assert "digits" in error


def test_a_model_that_invents_an_amount_never_reaches_the_customer():
    generator = _generator(
        "Aapka Rs 99,999 ka payment fail hua {amount} {due_date} "
        "{reference} {merchant}"
    )
    message = generator.generate(_observation())

    assert message.source == "fallback"
    assert "99,999" not in message.text
    assert "Rs 8,800.00" in message.text
    assert generator.verification_failures == 1


def test_the_rendered_message_only_contains_sourced_numbers():
    message = _generator().generate(_observation()).text
    assert verify_rendered(message, _facts()) is None


def test_an_unsourced_number_in_a_rendered_message_is_caught():
    """Defence against a substitution bug, not just a model failure."""
    facts = _facts()
    tampered = static_message(facts, 1).replace("Ref", "Ref 777")
    assert "unsourced number" in verify_rendered(tampered, facts)


# --------------------------------------------------------------------------
# Placeholders
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", PLACEHOLDERS)
def test_a_missing_placeholder_is_rejected(missing):
    template = GOOD.replace(missing, "")
    error = verify_template(template)
    assert error is not None
    assert missing in error


def test_a_duplicated_placeholder_is_rejected():
    error = verify_template(GOOD + " {amount}")
    assert "appears 2 times" in error


def test_an_invented_placeholder_is_rejected():
    error = verify_template(GOOD + " {account_number}")
    assert "unknown placeholders" in error


def test_an_empty_message_is_rejected():
    assert "empty" in verify_template("   ")


def test_a_clean_template_passes():
    assert verify_template(GOOD) is None


def test_an_unsubstituted_placeholder_is_caught_after_rendering():
    facts = _facts()
    assert "unsubstituted" in verify_rendered(
        "Rs 8,800.00 5th M000123 Artha Payments {amount}", facts
    )


# --------------------------------------------------------------------------
# Tone
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "contacts, expected", [(0, 1), (1, 2), (2, 3), (5, 3), (99, 3)]
)
def test_tone_rises_with_contacts_and_stops_at_three(contacts, expected):
    assert MessageGenerator.tone_for(contacts) == expected


def test_the_prompt_carries_the_tone_level():
    assert "Tone level: 2" in render_prompt(2, "reason")


def test_the_prompt_forbids_threatening_language_at_every_tone():
    prompt = render_prompt(3, "reason")
    flat = " ".join(prompt.lower().split())
    assert "never threatening" in flat
    assert "legal action" in flat


@pytest.mark.parametrize(
    "phrase",
    ["legal action", "recovery agent", "credit score", "CIBIL", "defaulter"],
)
def test_threatening_language_is_rejected_even_if_otherwise_valid(phrase):
    template = GOOD + f" Warning: {phrase} follows."
    error = verify_template(template)
    assert "forbidden language" in error


def test_a_threatening_message_falls_back_to_the_static_template():
    generator = _generator(GOOD + " We will involve a recovery agent.")
    message = generator.generate(_observation())
    assert message.source == "fallback"
    assert "recovery agent" not in message.text


# --------------------------------------------------------------------------
# Facts and rendering
# --------------------------------------------------------------------------


def test_facts_are_derived_from_the_observation_in_rupees():
    facts = _facts(amount_paise=880_000, due_day=5)
    assert facts.amount_text == "Rs 8,800.00"
    assert facts.due_date_text == "5th"
    assert facts.reference == "M000123"


@pytest.mark.parametrize(
    "day, suffix", [(1, "st"), (2, "nd"), (3, "rd"), (4, "th"), (11, "th"), (21, "st")]
)
def test_due_dates_are_written_the_way_a_person_would(day, suffix):
    assert _facts(due_day=day).due_date_text == f"{day}{suffix}"


def test_every_fact_appears_verbatim_in_the_sent_message():
    """The required assertion: templated facts survive generation intact."""
    message = _generator().generate(_observation()).text
    facts = _facts()
    for value in facts.as_mapping().values():
        assert value in message


def test_the_static_fallback_passes_its_own_verification():
    """The safety net must never itself be unsendable."""
    facts = _facts()
    for tone in (1, 2, 3):
        assert verify_rendered(static_message(facts, tone), facts) is None


def test_the_static_fallback_is_available_at_every_tone():
    facts = _facts()
    messages = {static_message(facts, tone) for tone in (1, 2, 3)}
    assert len(messages) == 3


# --------------------------------------------------------------------------
# Failure counting
# --------------------------------------------------------------------------


def test_an_unavailable_model_falls_back_and_is_counted():
    generator = MessageGenerator(StubClient(always_fail=True))
    message = generator.generate(_observation())

    assert message.source == "fallback"
    assert message.text
    assert generator.stats()["message_fallbacks"] == 1
    assert generator.stats()["message_verification_failures"] == 0


def test_a_verification_failure_is_counted_separately_from_an_outage():
    """The reliability figure must distinguish 'wrong' from 'unreachable'."""
    generator = _generator("no placeholders here at all")
    generator.generate(_observation())
    stats = generator.stats()
    assert stats["message_verification_failures"] == 1
    assert stats["message_fallbacks"] == 1


def test_a_good_message_is_counted_as_generated():
    generator = _generator()
    generator.generate(_observation())
    assert generator.stats() == {
        "messages_generated": 1,
        "message_fallbacks": 0,
        "message_verification_failures": 0,
    }


def test_generation_always_returns_something_sendable():
    """No path leaves the caller without a message."""
    for template in (GOOD, "garbage", "{amount} only", "Rs 500 {amount}"):
        generator = _generator(template)
        message = generator.generate(_observation())
        assert isinstance(message, GeneratedMessage)
        assert message.text.strip()
        assert verify_rendered(message.text, _facts()) is None
