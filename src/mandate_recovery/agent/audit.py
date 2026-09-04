"""The decision audit trail.

This artifact does double duty. It is the audit trail a payments system is
expected to produce — every action traceable to the state that justified it —
and it is the honesty evidence for this project: you can read one mandate's
entire story and see exactly which decisions were rules, which were a model,
and which were overruled by the compliance gate.

``to_human_readable`` is written to be *read*, not parsed. It is the output
that gets screen-recorded, so it renders rupees rather than paise, spells out
what the bank said, and quotes the rationale in full.

On timestamps
-------------
Entries are stamped with **simulation** day and hour, never wall-clock time.
An audit trail carrying ``datetime.now()`` could not be reproduced from a
stored config, and invariant 4 says every experiment must be. The real clock
is the one thing about a run that cannot be replayed.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..types import Action, Observation

__all__ = ["AuditEntry", "AuditLog", "observation_fingerprint"]


def observation_fingerprint(observation: Observation) -> str:
    """A short stable hash of the exact observation a decision was made on.

    Lets a reader confirm two decisions saw identical inputs without storing
    the whole object on every row.
    """
    payload = observation.model_dump_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _describe(action: Action | None) -> str:
    """One line describing an action, for a human."""
    if action is None:
        return "-"
    kind = getattr(action, "kind", type(action).__name__)
    if kind == "retry_silent":
        return (
            f"retry silently on day {action.scheduled_day} at "
            f"{action.scheduled_hour:02d}:00 via {action.rail.value}"
        )
    if kind == "send_nudge":
        return (
            f"contact the customer by {action.channel.value} "
            f"at tone level {action.tone_level}"
        )
    if kind == "collect_partial":
        return f"collect a partial {_rupees(action.amount_paise)}"
    if kind == "switch_rail":
        return f"move future attempts to {action.target_rail.value}"
    if kind == "escalate_human":
        return f"escalate to a human agent ({action.reason})"
    if kind == "stop":
        return f"stop ({action.reason})"
    return kind


def _rupees(paise: int | None) -> str:
    if paise is None:
        return "-"
    return f"Rs {paise / 100:,.2f}"


@dataclass(frozen=True)
class AuditEntry:
    """One decision, and everything that justified and followed it."""

    seed: int
    arm: str
    mandate_id: str
    day: int
    hour: int

    observation_hash: str
    amount_paise: int
    due_day: int
    attempts_this_cycle: int
    contacts_sent: int
    last_raw_code: str | None

    diagnosis: str | None
    proposed_action: str
    source: str
    rationale: str

    validator_approved: bool
    validator_rule: str | None
    validator_reason: str

    executed_action: str
    outcome: str | None
    running_cost_paise: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """Every decision made during a run, in order."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> Sequence[AuditEntry]:
        return tuple(self._entries)

    def record(
        self,
        *,
        seed: int,
        arm: str,
        observation: Observation,
        proposed_action: Action,
        source: str,
        rationale: str,
        executed_action: Action,
        validator_approved: bool,
        validator_reason: str,
        validator_rule: str | None = None,
        diagnosis: str | None = None,
        outcome: str | None = None,
        running_cost_paise: int = 0,
    ) -> AuditEntry:
        """Record one decision. Every argument here is required to explain it.

        ``rationale`` may not be blank: an action nobody can explain is an
        action that should not have been taken.
        """
        if not rationale or not rationale.strip():
            raise ValueError(
                f"decision for {observation.mandate_id} has no rationale; "
                "every action must be explainable"
            )

        entry = AuditEntry(
            seed=seed,
            arm=arm,
            mandate_id=observation.mandate_id,
            day=observation.current_day,
            hour=observation.current_hour,
            observation_hash=observation_fingerprint(observation),
            amount_paise=observation.amount_paise,
            due_day=observation.due_day,
            attempts_this_cycle=len(observation.attempt_history),
            contacts_sent=observation.contacts_sent,
            last_raw_code=(
                observation.attempt_history[-1].raw_code
                if observation.attempt_history
                else None
            ),
            diagnosis=diagnosis,
            proposed_action=_describe(proposed_action),
            source=source,
            rationale=rationale.strip(),
            validator_approved=validator_approved,
            validator_rule=validator_rule,
            validator_reason=validator_reason,
            executed_action=_describe(executed_action),
            outcome=outcome,
            running_cost_paise=running_cost_paise,
        )
        self._entries.append(entry)
        return entry

    # ------------------------------------------------------------------

    def entries_for(self, mandate_id: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.mandate_id == mandate_id]

    def mandate_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for entry in self._entries:
            seen.setdefault(entry.mandate_id, None)
        return list(seen)

    def to_dataframe(self):
        import pandas as pd

        if not self._entries:
            return pd.DataFrame()
        return pd.DataFrame([entry.to_row() for entry in self._entries])

    def to_human_readable(self, mandate_id: str) -> str:
        """One mandate's whole story, as text a person can read aloud."""
        entries = self.entries_for(mandate_id)
        if not entries:
            return f"No decisions were recorded for mandate {mandate_id}."

        first = entries[0]
        lines = [
            f"Mandate {mandate_id}  -  {_rupees(first.amount_paise)} "
            f"due on the {first.due_day}{_ordinal(first.due_day)} of each month",
            f"Arm: {first.arm}   Seed: {first.seed}",
            "=" * 72,
            "",
        ]

        for index, entry in enumerate(entries, start=1):
            lines.append(
                f"Decision {index}  -  day {entry.day}, {entry.hour:02d}:00"
            )
            lines.append(
                f"  Attempt {entry.attempts_this_cycle} this cycle"
                + (
                    f", bank said {entry.last_raw_code!r}"
                    if entry.last_raw_code is not None
                    else ", no attempt yet"
                )
            )
            if entry.diagnosis:
                lines.append(f"  Diagnosis:  {entry.diagnosis}")
            lines.append(f"  Proposed:   {entry.proposed_action}  [{entry.source}]")
            lines.append(f"  Because:    {_wrap(entry.rationale)}")

            if entry.validator_approved and entry.proposed_action == (
                entry.executed_action
            ):
                lines.append("  Validator:  approved")
            elif entry.validator_approved:
                lines.append(
                    "  Validator:  approved with changes - "
                    + _wrap(entry.validator_reason)
                )
            else:
                lines.append(
                    f"  Validator:  REFUSED ({entry.validator_rule})"
                )
                lines.append(f"              {_wrap(entry.validator_reason)}")

            lines.append(f"  Executed:   {entry.executed_action}")
            if entry.outcome:
                lines.append(f"  Result:     {entry.outcome}")
            lines.append(
                f"  Cost so far: {_rupees(entry.running_cost_paise)}"
            )
            lines.append("")

        lines.append("-" * 72)
        lines.append(
            f"{len(entries)} decision(s), "
            f"{sum(1 for e in entries if not e.validator_approved)} refused by "
            f"the compliance gate, "
            f"final running cost {_rupees(entries[-1].running_cost_paise)}."
        )
        return "\n".join(lines)

    def summary(self) -> Mapping[str, Any]:
        """Counts a run can report without loading the whole trail."""
        by_source: dict[str, int] = {}
        for entry in self._entries:
            by_source[entry.source] = by_source.get(entry.source, 0) + 1
        return {
            "n_decisions": len(self._entries),
            "n_mandates": len(self.mandate_ids()),
            "n_refused": sum(
                1 for e in self._entries if not e.validator_approved
            ),
            "decisions_by_source": by_source,
        }


def _ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _wrap(text: str, width: int = 62, indent: str = " " * 14) -> str:
    """Wrap a rationale so the rendered trail stays inside a terminal."""
    import textwrap

    wrapped = textwrap.wrap(text, width=width)
    if not wrapped:
        return text
    return ("\n" + indent).join(wrapped)
