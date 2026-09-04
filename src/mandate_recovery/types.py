"""Core domain types for the recovery harness.

THE OBSERVATION BOUNDARY IS THE CENTRAL INVARIANT OF THIS MODULE.

The simulator owns latent state -- the customer's balance trajectory, when
their salary lands, how fast they spend, how close they are to leaving, and
what their per-transaction ceiling is. That state lives in
:class:`LatentCustomerState` and is SIMULATOR-PRIVATE.

Policies never see it. A policy -- rule-based, learned, or LLM-backed --
receives exactly one input type, :class:`Observation`, which carries only what
a real collector could actually know at decision time. Nothing derived from
:class:`LatentCustomerState` may ever appear on :class:`Observation`, in a
policy signature, or in an LLM prompt. The two field sets are asserted
disjoint by ``tests/test_types.py::test_observation_contains_no_latent_fields``.
If a policy needs a fact it does not have, that is the experiment working as
designed; it is not a reason to widen the boundary.

Money is integer paise everywhere. Every ``*_paise`` field is a strict ``int``
so that a float is rejected at construction rather than silently truncated.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Paise",
    "NonNegativePaise",
    "PositivePaise",
    "MandateStatus",
    "Rail",
    "AttemptOutcome",
    "NudgeChannel",
    "Mandate",
    "LatentCustomerState",
    "BankResponse",
    "Attempt",
    "ObservedAttempt",
    "Observation",
    "RetrySilent",
    "SendNudge",
    "CollectPartial",
    "SwitchRail",
    "EscalateHuman",
    "Stop",
    "Action",
    "DecisionSource",
    "Decision",
]


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------
# All monetary amounts are integer paise, never floats. ``strict=True`` makes
# pydantic reject a float outright instead of coercing 1.0 -> 1, which would
# quietly admit float arithmetic into the money path.

Paise = Annotated[int, Field(strict=True)]
NonNegativePaise = Annotated[int, Field(strict=True, ge=0)]
PositivePaise = Annotated[int, Field(strict=True, gt=0)]


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class MandateStatus(str, Enum):
    """Lifecycle state of a recurring mandate."""

    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    COMPLETED = "COMPLETED"


class Rail(str, Enum):
    """Payment rail an attempt is presented on."""

    UPI_AUTOPAY = "UPI_AUTOPAY"
    CARD = "CARD"
    NACH = "NACH"


class AttemptOutcome(str, Enum):
    """Classified result of a debit attempt.

    This is the simulator's own classification of what happened. Policies see
    the bank's ``raw_code`` string instead -- see :class:`ObservedAttempt`.
    """

    SUCCESS = "SUCCESS"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    TECHNICAL_DECLINE = "TECHNICAL_DECLINE"
    WINDOW_REJECTED = "WINDOW_REJECTED"
    MANDATE_REVOKED = "MANDATE_REVOKED"


class NudgeChannel(str, Enum):
    """Channel a customer-facing nudge is sent on."""

    SMS = "SMS"
    WHATSAPP = "WHATSAPP"
    EMAIL = "EMAIL"
    IVR = "IVR"


DecisionSource = Literal["rule", "llm", "fallback"]


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------


class _FrozenModel(BaseModel):
    """Immutable, closed base for every domain type.

    ``frozen`` keeps records from being edited after construction so a stored
    experiment trace means what it said. ``extra="forbid"`` is load-bearing on
    :class:`Observation`: it makes smuggling a latent field across the
    boundary a construction-time error rather than a silent extra attribute.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------
# Mandate
# --------------------------------------------------------------------------


class Mandate(_FrozenModel):
    """A recurring-payment authorisation to collect from one customer."""

    id: str
    customer_id: str
    amount_paise: PositivePaise
    day_of_month: Annotated[int, Field(strict=True, ge=1, le=31)]
    created_on_day: Annotated[int, Field(strict=True, ge=0)]
    status: MandateStatus


# --------------------------------------------------------------------------
# Simulator-private state
# --------------------------------------------------------------------------


class LatentCustomerState(_FrozenModel):
    """SIMULATOR-PRIVATE. Ground truth about a customer.

    **This type must never cross the observation boundary.** It may be held by
    the simulator, read by the simulator, and written to raw experiment
    traces. It may not be passed to a policy, embedded in an
    :class:`Observation`, rendered into an LLM prompt, or reached through any
    object a policy holds.

    If you are writing a policy and want something from this class, the answer
    is no. Derive a legitimate signal from :class:`Observation` instead.
    """

    balance_paise: Paise
    salary_day: Annotated[int, Field(strict=True, ge=1, le=31)]
    salary_amount_paise: NonNegativePaise
    spend_rate_paise_per_day: NonNegativePaise
    churn_intent: Annotated[float, Field(ge=0.0, le=1.0)]
    per_txn_limit_paise: NonNegativePaise


# --------------------------------------------------------------------------
# Attempts
# --------------------------------------------------------------------------


class BankResponse(_FrozenModel):
    """What the bank returned for a single debit attempt."""

    raw_code: str
    outcome: AttemptOutcome
    bank_id: str
    timestamp: datetime


class Attempt(_FrozenModel):
    """One debit presented on one rail.

    ``response`` is ``None`` while the attempt is scheduled but not yet
    resolved.
    """

    mandate_id: str
    scheduled_at: datetime
    rail: Rail
    response: BankResponse | None = None


# --------------------------------------------------------------------------
# The observation boundary
# --------------------------------------------------------------------------


class ObservedAttempt(_FrozenModel):
    """One past attempt, as a policy is allowed to see it.

    Carries the bank's ``raw_code`` string, not the simulator's
    :class:`AttemptOutcome`. A real collector reads response codes and has to
    infer what they mean; handing over the classified outcome would leak the
    simulator's interpretation of its own latent state.
    """

    day: Annotated[int, Field(strict=True, ge=0)]
    hour: Annotated[int, Field(strict=True, ge=0, le=23)]
    rail: Rail
    raw_code: str


class Observation(_FrozenModel):
    """THE ONLY TYPE POLICIES SEE.

    Everything on this class is something a real collector could know at
    decision time: the mandate's own terms, the calendar, what has already
    been tried, what was already sent to the customer, and a summary of this
    customer's own payment history with us.

    Nothing here is derived from :class:`LatentCustomerState`. There is no
    balance, no salary day, no spend rate, no churn score, no per-transaction
    limit, and no proxy computed from them by the simulator.
    ``test_observation_contains_no_latent_fields`` asserts the two field sets
    stay disjoint, and ``extra="forbid"`` blocks adding one at runtime.

    ``max_historical_success_amount_paise`` is the one field that deserves a
    note: it is the largest amount we have actually collected from this
    customer before. It is a fact about our own settled history, not a read of
    the customer's current limit or balance.
    """

    mandate_id: str
    amount_paise: PositivePaise
    # Day of the month the mandate falls due, matching Mandate.day_of_month.
    due_day: Annotated[int, Field(strict=True, ge=1, le=31)]
    # Simulation day index, not a day of the month.
    current_day: Annotated[int, Field(strict=True, ge=0)]
    # Local clock hour right now. A collector knows what time it is; the
    # validator needs it to enforce contact hours.
    current_hour: Annotated[int, Field(strict=True, ge=0, le=23)] = 9
    attempt_history: tuple[ObservedAttempt, ...] = ()
    contacts_sent: Annotated[int, Field(strict=True, ge=0)] = 0
    # Contacts inside the trailing 7 days, which is the window the contact
    # cap is written against. Cumulative contacts_sent cannot express it.
    contacts_in_last_7_days: Annotated[int, Field(strict=True, ge=0)] = 0
    days_since_last_contact: Optional[Annotated[int, Field(strict=True, ge=0)]] = None
    # Whether this customer has a card usable for recurring debits. Set from
    # the calibrated card-penetration rate; it bounds the SwitchRail action.
    has_card_on_file: bool = False
    historical_success_count: Annotated[int, Field(strict=True, ge=0)] = 0
    historical_failure_count: Annotated[int, Field(strict=True, ge=0)] = 0
    max_historical_success_amount_paise: NonNegativePaise = 0
    # Days of the month this customer has actually paid us on. A merchant
    # knows this from their own settlement history; it is not a read of the
    # customer's salary date, only of when money has previously arrived.
    successful_days_of_month: tuple[
        Annotated[int, Field(strict=True, ge=1, le=31)], ...
    ] = ()


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


class RetrySilent(_FrozenModel):
    """Re-present the debit without contacting the customer."""

    kind: Literal["retry_silent"] = "retry_silent"
    scheduled_day: Annotated[int, Field(strict=True, ge=0)]
    scheduled_hour: Annotated[int, Field(strict=True, ge=0, le=23)]
    rail: Rail


class SendNudge(_FrozenModel):
    """Contact the customer. ``tone_level`` rises with insistence."""

    kind: Literal["send_nudge"] = "send_nudge"
    channel: NudgeChannel
    tone_level: Annotated[int, Field(strict=True, ge=1)]


class CollectPartial(_FrozenModel):
    """Attempt a smaller amount than the mandate's full value."""

    kind: Literal["collect_partial"] = "collect_partial"
    amount_paise: PositivePaise


class SwitchRail(_FrozenModel):
    """Move future attempts to a different rail."""

    kind: Literal["switch_rail"] = "switch_rail"
    target_rail: Rail


class EscalateHuman(_FrozenModel):
    """Hand the case to a human agent."""

    kind: Literal["escalate_human"] = "escalate_human"
    reason: str


class Stop(_FrozenModel):
    """Give up on this mandate for this cycle."""

    kind: Literal["stop"] = "stop"
    reason: str


Action = Annotated[
    RetrySilent
    | SendNudge
    | CollectPartial
    | SwitchRail
    | EscalateHuman
    | Stop,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------


class Decision(_FrozenModel):
    """An action chosen for an observation, plus how it was chosen.

    ``source`` records what produced the action; ``"llm"`` is a proposal, not
    an authorisation. ``validated`` is set only by a deterministic validator
    that has approved the action -- never by a model, and never by the code
    that proposed the action. Nothing acts on a :class:`Decision` whose
    ``validated`` is ``False``.
    """

    observation: Observation
    action: Action
    source: DecisionSource
    rationale: str
    validated: bool = False
