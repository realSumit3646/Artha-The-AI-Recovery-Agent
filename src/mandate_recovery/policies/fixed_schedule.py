"""The industry-standard baseline: retry at T+1, T+3, T+5, then give up.

This is what a competent merchant does today without any recovery
intelligence. It reads no response codes, chooses no channel, and knows
nothing about when the customer gets paid. It retries on a calendar.

**Built in good faith, deliberately.** A strawman baseline would make the
agent's advantage look larger and mean nothing; every number downstream would
be measuring how badly this file was written rather than how well the agent
works. Two choices in particular were made in the baseline's favour:

* **The retry hour defaults to 09:00, outside the NPCI restricted window.** A
  real ops team schedules its retry batch in business hours and knows to avoid
  peak; NPCI's deprioritisation is public. Putting the baseline inside the
  window would hand the agent a large advantage for free, on a decision no
  competent merchant gets wrong. The agent has to win on the things that are
  genuinely hard: reading the customer's cash cycle, diagnosing an ambiguous
  code, and choosing whether to make contact at all.
* **T+1 / T+3 / T+5 is the common default**, not the worst schedule
  available. It front-loads retries while a temporary decline might clear and
  spaces them enough to catch a short cash gap.

The offsets, hour and rail are all constructor arguments, so the sensitivity
sweep can ask whether the agent's advantage survives a better-tuned baseline.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from ..types import Decision, Observation, Rail, RetrySilent, Stop
from .base import Policy

__all__ = ["DEFAULT_RETRY_OFFSETS_DAYS", "DEFAULT_RETRY_HOUR", "FixedSchedulePolicy"]

#: The common industry default: retry the day after the failure, then two days
#: later, then two days after that. Not a published standard -- it is the
#: schedule most merchant recovery guides and gateway docs describe.
DEFAULT_RETRY_OFFSETS_DAYS: tuple[int, ...] = (1, 3, 5)

#: 09:00 local. Business hours, and outside the calibrated restricted window.
DEFAULT_RETRY_HOUR: int = 9


class FixedSchedulePolicy(Policy):
    """Retry on a fixed calendar offset from the first failure, then stop.

    Holds no state: everything it needs is on the observation, which is what
    makes it reproducible and what makes it dumb.
    """

    name: ClassVar[str] = "fixed_schedule"

    def __init__(
        self,
        retry_offsets_days: Sequence[int] = DEFAULT_RETRY_OFFSETS_DAYS,
        retry_hour: int = DEFAULT_RETRY_HOUR,
        rail: Rail = Rail.UPI_AUTOPAY,
    ) -> None:
        offsets = tuple(int(offset) for offset in retry_offsets_days)
        if any(offset < 0 for offset in offsets):
            raise ValueError(f"retry offsets must be >= 0, got {offsets}")
        if list(offsets) != sorted(offsets):
            raise ValueError(f"retry offsets must be ascending, got {offsets}")
        if not 0 <= retry_hour <= 23:
            raise ValueError(f"retry_hour must be in 0..23, got {retry_hour}")

        self._offsets = offsets
        self._retry_hour = int(retry_hour)
        self._rail = rail

    @property
    def retry_offsets_days(self) -> tuple[int, ...]:
        return self._offsets

    @property
    def retry_hour(self) -> int:
        return self._retry_hour

    @property
    def rail(self) -> Rail:
        return self._rail

    def decide(self, observation: Observation) -> Decision:
        history = observation.attempt_history

        if not history:
            return self.decision(
                observation,
                Stop(reason="no failure to recover from"),
                rationale=(
                    "No attempt has been made on this mandate yet, so there "
                    "is nothing for a retry schedule to act on."
                ),
            )

        first_failure_day = history[0].day
        retries_made = len(history) - 1

        if retries_made >= len(self._offsets):
            return self.decision(
                observation,
                Stop(reason="retry schedule exhausted"),
                rationale=(
                    f"All {len(self._offsets)} scheduled retries "
                    f"(T+{', T+'.join(str(o) for o in self._offsets)}) have "
                    "been used. The fixed schedule has nothing left to try."
                ),
            )

        offset = self._offsets[retries_made]
        scheduled_day = first_failure_day + offset
        # Never schedule into the past: if the harness reaches this decision
        # after the slot has gone, take the earliest one still available.
        scheduled_day = max(scheduled_day, observation.current_day)

        return self.decision(
            observation,
            RetrySilent(
                scheduled_day=scheduled_day,
                scheduled_hour=self._retry_hour,
                rail=self._rail,
            ),
            rationale=(
                f"Retry {retries_made + 1} of {len(self._offsets)}: the fixed "
                f"schedule places it at T+{offset} from the first failure on "
                f"day {first_failure_day}, at {self._retry_hour:02d}:00 on "
                f"{self._rail.value}. No diagnosis was performed and the "
                "response code was not read."
            ),
        )
