"""When to re-present a failed debit. Deterministic, no model involved.

Timing is constrained optimisation, not judgment. There is a finite set of
(day, hour) slots, a hard exclusion, and three priors that can be scored
arithmetically. A language model adds nothing here except non-determinism and
latency, so it is not asked.

What the scheduler believes
---------------------------
Everything in this module is the *agent's* belief about the world, not the
simulator's ground truth. It is stated here, in the agent package, precisely
so the difference is visible:

* **The restricted window** is knowledge a merchant legitimately has — NPCI
  publishes its peak-hour policy. It arrives through
  :class:`SchedulerConstraints` rather than being read from the simulator.
* **Bank uptime by tier** is a prior the agent holds, inferred from the code
  vocabulary it has seen. A merchant learns over months that one bank's
  declines cluster differently from another's. It is deliberately cruder than
  the true availability figures.
* **Salary timing** is a folk prior — Indian payroll clusters at month-end and
  in the first week — sharpened by the days this customer has actually paid on
  before. The agent never sees a salary date.

None of these are read from `CalibrationSet`, and nothing here imports the
simulator. If the agent's priors happen to be wrong, it schedules worse; that
is the experiment working.

Scoring
-------
A candidate slot scores as the product of four terms: funds-present
likelihood on that day, the bank's availability prior, an hour preference, and
a cooling-off penalty. The argmax wins, and the reasons are rendered into a
rationale string that goes into the audit trail — so write it for a reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..types import Observation, Rail

__all__ = [
    "DAYS_IN_MONTH",
    "HOURS_IN_DAY",
    "CODE_PREFIX_TO_TIER",
    "ASSUMED_TIER_AVAILABILITY",
    "SALARY_DAY_PRIOR",
    "SchedulerConstraints",
    "RetrySlot",
    "infer_bank_tier",
    "next_retry_slot",
]

DAYS_IN_MONTH = 31
HOURS_IN_DAY = 24

#: What the agent has learned about whose codes look like what. A merchant
#: builds this from their own settlement reports; it is not imported from the
#: simulator, and a test asserts the two have not drifted apart.
CODE_PREFIX_TO_TIER: Mapping[str, str] = {
    "AB": "large_private",
    "PS-": "psu",
    "SF_": "small_finance",
}

#: The agent's belief about how often each tier is up. Deliberately rounder
#: than the calibrated truth: the agent has impressions, not measurements.
ASSUMED_TIER_AVAILABILITY: Mapping[str, float] = {
    "large_private": 0.95,
    "psu": 0.90,
    "small_finance": 0.85,
    "unknown": 0.90,
}

#: Folk prior over payday. Month-end and the first week carry the mass. Not a
#: calibrated distribution — a belief anyone collecting in India would hold.
SALARY_DAY_PRIOR: Mapping[int, float] = {
    **{day: 0.055 for day in range(1, 8)},
    **{day: 0.045 for day in range(25, 32)},
}

#: How long money stays in the account after payday, as a per-day decay.
_FUNDS_DECAY_PER_DAY = 0.72

#: Mass given to a day this customer has actually paid us on. Set as an
#: absolute floor rather than a multiplier: multiplying a mid-month day's tiny
#: prior leaves it far below the month-end cluster, so observed behaviour
#: would never overturn the population average — which is the whole point of
#: having it.
_OBSERVED_DAY_MASS = 0.30

#: Days with no payday mass at all still have a little: people are paid
#: bonuses, refunded, and lent money.
_WEAK_DAY_MASS = 0.004


@dataclass(frozen=True)
class SchedulerConstraints:
    """The hard limits and the search space.

    ``restricted_windows`` is public NPCI policy, passed in rather than read
    from the simulator so the dependency is visible at the call site.
    """

    restricted_windows: tuple[tuple[int, int], ...] = ((10, 13), (17, 21))
    horizon_days: int = 14
    cooling_off_days: int = 1
    allowed_rails: tuple[Rail, ...] = (Rail.UPI_AUTOPAY,)

    def is_restricted(self, hour: int) -> bool:
        return any(start <= hour < end for start, end in self.restricted_windows)


@dataclass(frozen=True)
class RetrySlot:
    """A chosen slot, its score, and why it was chosen."""

    day: int
    hour: int
    score: float
    rationale: str

    def as_tuple(self) -> tuple[int, int]:
        return (self.day, self.hour)


def infer_bank_tier(raw_codes: Sequence[str]) -> str:
    """Guess the customer's bank tier from the codes it has returned.

    Legitimately observable: the codes are on the observation. Generic and
    missing codes carry no signal, so they are skipped. Returns ``"unknown"``
    when nothing identifies a tier, which is common early on.
    """
    for code in raw_codes:
        if not code:
            continue
        for prefix, tier in CODE_PREFIX_TO_TIER.items():
            if code.startswith(prefix):
                return tier
    return "unknown"


def _funds_prior(day_of_month: int, successful_days: Sequence[int]) -> float:
    """Likelihood the account has money on this day of the month.

    Money arrives on payday and drains afterwards, so a day scores for its own
    payday mass plus the decayed mass of the paydays just before it.
    """
    observed = set(successful_days)
    total = 0.0
    for lag in range(6):
        day = ((day_of_month - 1 - lag) % DAYS_IN_MONTH) + 1
        mass = SALARY_DAY_PRIOR.get(day, _WEAK_DAY_MASS)
        if day in observed:
            mass = max(mass, _OBSERVED_DAY_MASS)
        total += mass * (_FUNDS_DECAY_PER_DAY**lag)
    return total


def _hour_prior(hour: int) -> float:
    """Prefer early hours: salary credits land overnight, spending follows.

    Re-presenting at 06:00 catches the balance before the day eats it. This is
    a belief, not a measurement, and it is the cheapest lever the agent has.
    """
    if 4 <= hour <= 9:
        return 1.0
    if hour < 4:
        return 0.85
    if hour <= 13:
        return 0.6
    return 0.45


def _cooling_penalty(days_since_last_attempt: int, cooling_off_days: int) -> float:
    """Hammering the same mandate daily annoys the bank and wastes fees."""
    if days_since_last_attempt < cooling_off_days:
        return 0.0
    if days_since_last_attempt == cooling_off_days:
        return 0.7
    return 1.0


def next_retry_slot(
    observation: Observation,
    constraints: SchedulerConstraints | None = None,
) -> RetrySlot | None:
    """Best (day, hour) to re-present this debit, or None if there is none.

    Returns ``None`` only when every candidate slot is excluded — in practice
    when the horizon is shorter than the cooling-off period.
    """
    constraints = constraints or SchedulerConstraints()

    last_attempt_day = (
        observation.attempt_history[-1].day
        if observation.attempt_history
        else observation.current_day
    )
    tier = infer_bank_tier([a.raw_code for a in observation.attempt_history])
    bank_prior = ASSUMED_TIER_AVAILABILITY.get(
        tier, ASSUMED_TIER_AVAILABILITY["unknown"]
    )

    best: RetrySlot | None = None
    for offset in range(0, constraints.horizon_days + 1):
        day = observation.current_day + offset
        days_since = day - last_attempt_day
        cooling = _cooling_penalty(days_since, constraints.cooling_off_days)
        if cooling == 0.0:
            continue

        day_of_month = (day % DAYS_IN_MONTH) + 1
        funds = _funds_prior(day_of_month, observation.successful_days_of_month)

        for hour in range(HOURS_IN_DAY):
            if constraints.is_restricted(hour):
                continue  # hard exclusion, never scored
            score = funds * bank_prior * _hour_prior(hour) * cooling
            if best is None or score > best.score:
                best = RetrySlot(
                    day=day,
                    hour=hour,
                    score=score,
                    rationale=_rationale(
                        day=day,
                        hour=hour,
                        day_of_month=day_of_month,
                        tier=tier,
                        bank_prior=bank_prior,
                        observation=observation,
                        days_since=days_since,
                    ),
                )
    return best


def _rationale(
    *,
    day: int,
    hour: int,
    day_of_month: int,
    tier: str,
    bank_prior: float,
    observation: Observation,
    days_since: int,
) -> str:
    """Why this slot. Written for a human reading the audit trail."""
    reasons = [
        f"Day {day} (the {day_of_month}{_ordinal(day_of_month)}) at "
        f"{hour:02d}:00."
    ]

    if day_of_month in set(observation.successful_days_of_month):
        reasons.append(
            f"This customer has paid us on the {day_of_month}"
            f"{_ordinal(day_of_month)} before, which outweighs the "
            "population payday prior."
        )
    elif day_of_month <= 7 or day_of_month >= 25:
        reasons.append(
            "Month-end and the first week are when salaries land, so funds "
            "are most likely present."
        )
    else:
        reasons.append(
            "Mid-month is a weak window for funds; this was the best slot "
            "available within the horizon."
        )

    if tier == "unknown":
        reasons.append(
            "The bank could not be identified from the response codes seen "
            "so far, so a neutral availability prior was used."
        )
    else:
        reasons.append(
            f"Response codes identify the bank as {tier.replace('_', ' ')}, "
            f"assumed available {bank_prior:.0%} of the time."
        )

    reasons.append(
        f"{hour:02d}:00 is outside the NPCI restricted window and early "
        "enough to catch the balance before the day's spending."
    )
    reasons.append(f"{days_since} day(s) since the last attempt.")
    return " ".join(reasons)


def _ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
