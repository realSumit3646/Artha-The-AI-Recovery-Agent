"""Tests for the decision audit trail.

Two required properties: every executed action has an entry, and no entry is
missing a rationale. The rendering tests exist because this output is meant to
be read on screen — if it stops being readable, it stops doing its job.
"""

from __future__ import annotations

import pytest

from mandate_recovery.agent.audit import (
    AuditEntry,
    AuditLog,
    observation_fingerprint,
)
from mandate_recovery.types import (
    NudgeChannel,
    Observation,
    ObservedAttempt,
    Rail,
    RetrySilent,
    SendNudge,
    Stop,
)


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m000123",
        "amount_paise": 880_000,
        "due_day": 5,
        "current_day": 5,
        "current_hour": 9,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def _history(code: str = "PS-51", day: int = 5):
    return (
        ObservedAttempt(day=day, hour=9, rail=Rail.UPI_AUTOPAY, raw_code=code),
    )


def _retry(day: int = 34, hour: int = 6) -> RetrySilent:
    return RetrySilent(
        scheduled_day=day, scheduled_hour=hour, rail=Rail.UPI_AUTOPAY
    )


def _record(log: AuditLog, **overrides) -> AuditEntry:
    kwargs = dict(
        seed=42,
        arm="heuristic",
        observation=_observation(attempt_history=_history()),
        proposed_action=_retry(),
        source="rule",
        rationale=(
            "The bank reported a funds failure and this customer is paid at "
            "month end, so the retry is placed just after the next salary."
        ),
        executed_action=_retry(),
        validator_approved=True,
        validator_reason="retry permitted",
        running_cost_paise=200,
    )
    kwargs.update(overrides)
    return log.record(**kwargs)


# --------------------------------------------------------------------------
# The two required properties
# --------------------------------------------------------------------------


def test_every_executed_action_has_an_audit_entry():
    log = AuditLog()
    for day in range(5):
        _record(log, observation=_observation(current_day=day, attempt_history=_history()))

    assert len(log) == 5
    assert all(entry.executed_action for entry in log)
    assert len(log.to_dataframe()) == 5


def test_no_entry_can_be_recorded_without_a_rationale():
    """An action nobody can explain is an action that should not be taken."""
    log = AuditLog()
    for blank in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="rationale"):
            _record(log, rationale=blank)
    assert len(log) == 0


def test_every_recorded_entry_carries_a_rationale():
    log = AuditLog()
    _record(log)
    _record(log, proposed_action=Stop(reason="done"), executed_action=Stop(reason="done"))
    assert all(entry.rationale.strip() for entry in log)


# --------------------------------------------------------------------------
# What an entry captures
# --------------------------------------------------------------------------


def test_an_entry_captures_the_observation_it_was_made_on():
    log = AuditLog()
    observation = _observation(attempt_history=_history("AB1200"))
    entry = _record(log, observation=observation)

    assert entry.observation_hash == observation_fingerprint(observation)
    assert entry.amount_paise == observation.amount_paise
    assert entry.attempts_this_cycle == 1
    assert entry.last_raw_code == "AB1200"


def test_identical_observations_fingerprint_identically():
    left, right = _observation(), _observation()
    assert observation_fingerprint(left) == observation_fingerprint(right)


def test_a_changed_observation_changes_the_fingerprint():
    assert observation_fingerprint(_observation()) != observation_fingerprint(
        _observation(current_day=6)
    )


def test_the_decision_source_is_recorded():
    log = AuditLog()
    _record(log, source="rule")
    _record(log, source="llm")
    _record(log, source="fallback")
    assert [e.source for e in log] == ["rule", "llm", "fallback"]
    assert log.summary()["decisions_by_source"] == {
        "rule": 1,
        "llm": 1,
        "fallback": 1,
    }


def test_a_refusal_records_the_rule_that_fired():
    log = AuditLog()
    entry = _record(
        log,
        executed_action=Stop(reason="contact cap reached"),
        validator_approved=False,
        validator_rule="contact_cap_reached",
        validator_reason="3 contacts in the last 7 days against a cap of 3.",
    )
    assert entry.validator_approved is False
    assert entry.validator_rule == "contact_cap_reached"
    assert log.summary()["n_refused"] == 1


def test_entries_use_simulation_time_not_the_wall_clock():
    """A trail stamped with datetime.now() could not be reproduced."""
    log = AuditLog()
    entry = _record(log, observation=_observation(current_day=17, current_hour=6))
    assert entry.day == 17
    assert entry.hour == 6
    assert not any(
        "timestamp" in field for field in entry.to_row() if "day" not in field
    )


def test_a_diagnosis_is_recorded_when_present():
    log = AuditLog()
    entry = _record(log, diagnosis="insufficient funds (confident)")
    assert entry.diagnosis == "insufficient funds (confident)"
    assert _record(log, diagnosis=None).diagnosis is None


# --------------------------------------------------------------------------
# Rendering for a human
# --------------------------------------------------------------------------


def test_the_human_readable_trail_tells_the_whole_story():
    log = AuditLog()
    _record(log)
    _record(
        log,
        observation=_observation(current_day=34, attempt_history=_history("DECLINED", day=34)),
        proposed_action=SendNudge(channel=NudgeChannel.SMS, tone_level=1),
        executed_action=SendNudge(channel=NudgeChannel.SMS, tone_level=1),
        diagnosis="unknown (generic code)",
        rationale="The code was uninformative, so the customer is asked directly.",
        running_cost_paise=415,
    )
    rendered = log.to_human_readable("m000123")

    assert "Mandate m000123" in rendered
    assert "Rs 8,800.00" in rendered
    assert "5th of each month" in rendered
    assert "Decision 1" in rendered and "Decision 2" in rendered
    assert "retry silently on day 34 at 06:00" in rendered
    assert "contact the customer by SMS" in rendered
    assert "unknown (generic code)" in rendered
    assert "Rs 4.15" in rendered
    assert "2 decision(s)" in rendered


def test_it_renders_rupees_not_paise():
    """Nobody reads paise off a screen."""
    log = AuditLog()
    _record(log, running_cost_paise=123_456)
    rendered = log.to_human_readable("m000123")
    assert "Rs 1,234.56" in rendered
    assert "123456" not in rendered


def test_a_refusal_is_shouted_not_buried():
    log = AuditLog()
    _record(
        log,
        executed_action=Stop(reason="contact cap reached"),
        validator_approved=False,
        validator_rule="contact_cap_reached",
        validator_reason="3 contacts in the last 7 days against a cap of 3.",
    )
    rendered = log.to_human_readable("m000123")
    assert "REFUSED" in rendered
    assert "contact_cap_reached" in rendered
    assert "1 refused by the compliance gate" in rendered


def test_a_corrected_action_shows_both_what_was_asked_and_what_ran():
    log = AuditLog()
    _record(
        log,
        proposed_action=_retry(day=34, hour=11),
        executed_action=_retry(day=34, hour=13),
        validator_approved=True,
        validator_reason="moved out of the NPCI restricted window to 13:00",
    )
    rendered = log.to_human_readable("m000123")
    assert "11:00" in rendered and "13:00" in rendered
    assert "approved with changes" in rendered


def test_long_rationales_are_wrapped_to_stay_readable():
    log = AuditLog()
    _record(log, rationale="word " * 60)
    rendered = log.to_human_readable("m000123")
    assert max(len(line) for line in rendered.splitlines()) < 90


def test_an_unknown_mandate_renders_a_plain_message():
    assert "No decisions" in AuditLog().to_human_readable("nope")


# --------------------------------------------------------------------------
# Tabular output
# --------------------------------------------------------------------------


def test_the_dataframe_has_one_row_per_decision():
    log = AuditLog()
    for _ in range(3):
        _record(log)
    frame = log.to_dataframe()
    assert len(frame) == 3
    for column in ("mandate_id", "rationale", "source", "validator_approved"):
        assert column in frame.columns


def test_an_empty_log_produces_an_empty_frame():
    log = AuditLog()
    assert log.to_dataframe().empty
    assert log.summary()["n_decisions"] == 0


def test_entries_are_filtered_per_mandate():
    log = AuditLog()
    _record(log, observation=_observation(mandate_id="a", attempt_history=_history()))
    _record(log, observation=_observation(mandate_id="b", attempt_history=_history()))
    _record(log, observation=_observation(mandate_id="a", attempt_history=_history()))

    assert len(log.entries_for("a")) == 2
    assert log.mandate_ids() == ["a", "b"]
