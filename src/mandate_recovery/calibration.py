"""The single source of every number the simulator uses.

No magic constants anywhere else in the harness. If the simulator needs a
rate, a probability, a cost or a distribution, it reads it from a
:class:`CalibrationSet` -- and every one of those numbers arrives wearing a
label saying where it came from and how much to trust it.

That labelling is the point. A simulation whose parameters cannot be traced
is a simulation whose results cannot be argued with. Each parameter is a
:class:`CalibratedValue` carrying its ``unit``, its ``source``, and a
``confidence`` of ``"published"``, ``"derived"`` or ``"assumption"``.

**Nothing in the current defaults is marked ``published`` or ``derived``.**
Every default here is a placeholder chosen by the author, and every ``source``
string says so -- either with :data:`NO_PUBLIC_SOURCE` where no public figure
exists to find, or with a ``TODO(sumit)`` note naming where a real figure
should come from. No citation is invented. See ``docs/CALIBRATION.md``.

Parameter values are deliberately NOT tuned here. Once a policy has been
evaluated against the simulator, these numbers are frozen; the honest way to
test whether a result depends on them is the sensitivity sweep (commit 27),
not a quiet edit.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Final, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import NonNegativePaise

__all__ = [
    "NO_PUBLIC_SOURCE",
    "TODO_MARKER",
    "Confidence",
    "BankTier",
    "CalibratedValue",
    "CalibrationSet",
    "FAILURE_SHARE_FIELDS",
    "SALARY_CREDIT_DAYS",
    "DEFAULT_CALIBRATION",
]


# --------------------------------------------------------------------------
# Honesty markers
# --------------------------------------------------------------------------

#: The exact wording required in the ``source`` of any parameter for which no
#: public figure exists. Kept as a constant so it cannot drift and so tests
#: can assert on it.
NO_PUBLIC_SOURCE: Final = "no public source — author's assumption"

#: Prefix marking a placeholder whose real figure is expected to exist in
#: public data but has not been filled in yet.
TODO_MARKER: Final = "TODO(sumit)"

Confidence = Literal["published", "derived", "assumption"]


class BankTier(str, Enum):
    """Bank cohorts that fail at materially different rates."""

    LARGE_PRIVATE = "large_private"
    PSU = "psu"
    SMALL_FINANCE = "small_finance"


#: Days of the month on which salaries are modelled as landing.
SALARY_CREDIT_DAYS: Final = tuple(range(1, 8)) + tuple(range(25, 32))

#: The failure-share parameters, which partition all failures and must sum
#: to exactly 1.
FAILURE_SHARE_FIELDS: Final = (
    "share_of_failures_insufficient_funds",
    "share_of_failures_technical",
    "share_of_failures_limit",
    "share_of_failures_window_rejected",
)

_TOLERANCE: Final = 1e-9

#: Parameters that are genuinely not probabilities and so are exempt from the
#: [0, 1] check. Listed explicitly rather than guessed from the name, because
#: a distribution shape parameter reads like a rate and is not one.
_UNBOUNDED_PARAMETERS: Final = frozenset(
    {
        "restricted_window_hours",
        "monthly_salary_lognormal_sigma",
        "initial_churn_intent_alpha",
        "initial_churn_intent_beta",
    }
)


# --------------------------------------------------------------------------
# CalibratedValue
# --------------------------------------------------------------------------

T = TypeVar("T")


class CalibratedValue(BaseModel, Generic[T]):
    """One parameter, with its provenance attached.

    ``source`` is mandatory and non-empty by construction. An unlabelled
    number cannot enter the simulator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: T
    unit: str = Field(min_length=1)
    source: str = Field(min_length=1)
    confidence: Confidence

    @model_validator(mode="after")
    def _check_source_is_honest(self) -> CalibratedValue[T]:
        if not self.source.strip():
            raise ValueError("source must not be blank")

        # A placeholder or an admitted assumption may never be dressed up as
        # a real figure. This is the honesty rule made structural.
        if self.confidence in ("published", "derived"):
            if TODO_MARKER in self.source or self.source == NO_PUBLIC_SOURCE:
                raise ValueError(
                    f"confidence={self.confidence!r} claims a real figure, but "
                    f"source={self.source!r} says otherwise"
                )
        return self


def _assumption(
    value: T, unit: str, note: str | None = None
) -> CalibratedValue[T]:
    """A parameter with no public figure to find. Says so, plainly."""
    source = NO_PUBLIC_SOURCE if note is None else f"{NO_PUBLIC_SOURCE} ({note})"
    return CalibratedValue(
        value=value, unit=unit, source=source, confidence="assumption"
    )


def _placeholder(value: T, unit: str, expected_source: str) -> CalibratedValue[T]:
    """A placeholder awaiting a real published figure.

    ``expected_source`` names where to look. It is not a citation and the
    confidence stays ``"assumption"`` until the number is actually replaced.
    """
    return CalibratedValue(
        value=value,
        unit=unit,
        source=f"{TODO_MARKER}: placeholder, not sourced — expected source: "
        f"{expected_source}",
        confidence="assumption",
    )


# --------------------------------------------------------------------------
# CalibrationSet
# --------------------------------------------------------------------------


class CalibrationSet(BaseModel):
    """Every number the simulator is allowed to use.

    Defaults are placeholders at the conservative (pessimistic) end of the
    plausible range; see ``docs/CALIBRATION.md`` for the reasoning and for the
    sweep that tests whether conclusions survive the optimistic end.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -- Failure rate and its composition -------------------------------
    upi_autopay_execution_failure_rate: CalibratedValue[float] = _placeholder(
        0.30,
        "fraction of scheduled executions that fail",
        "NPCI UPI monthly statistics / payment-gateway autopay success "
        "reports; pick a single reporting period and cite it",
    )
    share_of_failures_insufficient_funds: CalibratedValue[float] = _placeholder(
        0.55,
        "fraction of failed executions",
        "gateway or issuer decline-reason breakdown for recurring debits",
    )
    share_of_failures_technical: CalibratedValue[float] = _placeholder(
        0.25,
        "fraction of failed executions",
        "gateway or issuer decline-reason breakdown for recurring debits",
    )
    share_of_failures_limit: CalibratedValue[float] = _placeholder(
        0.10,
        "fraction of failed executions",
        "gateway or issuer decline-reason breakdown for recurring debits",
    )
    share_of_failures_window_rejected: CalibratedValue[float] = _placeholder(
        0.10,
        "fraction of failed executions",
        "gateway or issuer decline-reason breakdown for recurring debits",
    )

    # -- The restricted (peak) window ------------------------------------
    restricted_window_hours: CalibratedValue[
        tuple[tuple[int, int], ...]
    ] = _placeholder(
        ((10, 13), (17, 21)),
        "local clock hours, half-open [start, end)",
        "NPCI circular on processing windows for recurring e-mandates — "
        "confirm the exact peak hours and the circular number",
    )
    restricted_window_rejection_probability: CalibratedValue[float] = _assumption(
        0.35,
        "probability a debit presented inside the window is deprioritised",
        "NPCI states recurring debits are deprioritised at peak, but "
        "publishes no rejection rate",
    )

    # -- Bank health ------------------------------------------------------
    bank_availability_by_tier: CalibratedValue[
        dict[BankTier, float]
    ] = _placeholder(
        {
            BankTier.LARGE_PRIVATE: 0.985,
            BankTier.PSU: 0.950,
            BankTier.SMALL_FINANCE: 0.920,
        },
        "probability the issuer is available for a debit",
        "NPCI publishes bank-wise UPI technical decline rates monthly — "
        "aggregate them into these three tiers and cite the period",
    )

    # -- Mandate lifecycle -------------------------------------------------
    monthly_mandate_revocation_rate: CalibratedValue[float] = _assumption(
        0.02,
        "fraction of active mandates revoked per month",
        "revocation is a customer action merchants report privately, if "
        "at all",
    )

    # -- Customer cash cycle ----------------------------------------------
    salary_credit_day_distribution: CalibratedValue[
        dict[int, float]
    ] = _assumption(
        {
            1: 0.18, 2: 0.10, 3: 0.06, 4: 0.04, 5: 0.04, 6: 0.03, 7: 0.03,
            25: 0.04, 26: 0.04, 27: 0.05, 28: 0.06, 29: 0.08,
            30: 0.13, 31: 0.12,
        },
        "probability mass by day of month",
        "no public dataset of Indian payroll credit dates; shape reflects "
        "month-end and first-week clustering",
    )

    # -- Rail availability -------------------------------------------------
    card_penetration_rate: CalibratedValue[float] = _placeholder(
        0.25,
        "fraction of customers holding a card usable for recurring debits",
        "RBI monthly card statistics, bounded to the customer segment this "
        "experiment models",
    )

    # -- Costs (integer paise) ---------------------------------------------
    gateway_cost_per_attempt_paise: CalibratedValue[
        NonNegativePaise
    ] = _placeholder(
        200,
        "paise per debit attempt",
        "published payment-gateway pricing for recurring mandates",
    )
    sms_cost_paise: CalibratedValue[NonNegativePaise] = _placeholder(
        15,
        "paise per message",
        "published transactional SMS / DLT pricing",
    )
    voice_call_cost_paise: CalibratedValue[NonNegativePaise] = _placeholder(
        120,
        "paise per completed call",
        "published outbound IVR or agent-call pricing",
    )

    # -- Customer experience damage ----------------------------------------
    churn_probability_increment_per_contact: CalibratedValue[float] = _assumption(
        0.015,
        "probability added to churn intent per customer contact",
        "the cost of nagging a customer is not something merchants publish",
    )

    # -- Population shape (who the customers are) --------------------------
    # These describe the population the simulator generates. They live here,
    # not in the simulator, so that they are captured in a stored experiment
    # config and swept along with everything else.
    bank_tier_mix: CalibratedValue[dict[BankTier, float]] = _placeholder(
        {
            BankTier.LARGE_PRIVATE: 0.45,
            BankTier.PSU: 0.40,
            BankTier.SMALL_FINANCE: 0.15,
        },
        "fraction of customers banking with each tier",
        "RBI / NPCI bank-wise account or UPI volume share, narrowed to the "
        "customer segment this experiment models",
    )
    monthly_salary_paise_median: CalibratedValue[
        NonNegativePaise
    ] = _placeholder(
        3_500_000,
        "paise per month, median of the salary distribution",
        "PLFS or EPFO wage distribution for the salaried segment",
    )
    monthly_salary_lognormal_sigma: CalibratedValue[float] = _placeholder(
        0.55,
        "sigma of log salary (dimensionless)",
        "derive from two published wage percentiles; record the calculation "
        "and re-mark this parameter as derived",
    )
    monthly_spend_share_of_salary: CalibratedValue[float] = _placeholder(
        0.75,
        "fraction of monthly salary spent over the month",
        "household consumption survey (MPCE) against the same wage segment",
    )
    initial_churn_intent_alpha: CalibratedValue[float] = _assumption(
        1.5,
        "alpha of the Beta prior on initial churn intent",
        "churn intent is not observable, so no public figure can exist; "
        "alpha/beta chosen to put most customers near zero intent",
    )
    initial_churn_intent_beta: CalibratedValue[float] = _assumption(
        28.5,
        "beta of the Beta prior on initial churn intent",
        "paired with initial_churn_intent_alpha for a mean of 0.05",
    )
    per_txn_limit_paise_by_tier: CalibratedValue[
        dict[BankTier, NonNegativePaise]
    ] = _placeholder(
        {
            BankTier.LARGE_PRIVATE: 10_000_000,
            BankTier.PSU: 10_000_000,
            BankTier.SMALL_FINANCE: 5_000_000,
        },
        "paise, per-transaction ceiling for a recurring debit",
        "NPCI UPI transaction-limit circulars plus per-bank published "
        "mandate limits",
    )

    # ----------------------------------------------------------------------
    # Cross-parameter validation
    # ----------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_failure_shares_partition(self) -> CalibrationSet:
        total = sum(getattr(self, name).value for name in FAILURE_SHARE_FIELDS)
        if not math.isclose(total, 1.0, abs_tol=_TOLERANCE):
            raise ValueError(
                f"failure shares must sum to 1.0, got {total!r} from "
                f"{FAILURE_SHARE_FIELDS}"
            )
        return self

    @model_validator(mode="after")
    def _check_salary_distribution(self) -> CalibrationSet:
        weights = self.salary_credit_day_distribution.value
        unexpected = set(weights) - set(SALARY_CREDIT_DAYS)
        if unexpected:
            raise ValueError(
                f"salary_credit_day_distribution has days outside "
                f"{SALARY_CREDIT_DAYS}: {sorted(unexpected)}"
            )
        total = sum(weights.values())
        if not math.isclose(total, 1.0, abs_tol=_TOLERANCE):
            raise ValueError(
                f"salary_credit_day_distribution must sum to 1.0, got {total!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_bank_tier_mix(self) -> CalibrationSet:
        mix = self.bank_tier_mix.value
        missing = set(BankTier) - set(mix)
        if missing:
            raise ValueError(
                f"bank_tier_mix is missing tiers: "
                f"{sorted(t.value for t in missing)}"
            )
        total = sum(mix.values())
        if not math.isclose(total, 1.0, abs_tol=_TOLERANCE):
            raise ValueError(f"bank_tier_mix must sum to 1.0, got {total!r}")
        return self

    @model_validator(mode="after")
    def _check_per_txn_limits_complete(self) -> CalibrationSet:
        missing = set(BankTier) - set(self.per_txn_limit_paise_by_tier.value)
        if missing:
            raise ValueError(
                f"per_txn_limit_paise_by_tier is missing tiers: "
                f"{sorted(t.value for t in missing)}"
            )
        return self

    @model_validator(mode="after")
    def _check_beta_shape_is_positive(self) -> CalibrationSet:
        for name in ("initial_churn_intent_alpha", "initial_churn_intent_beta"):
            shape = getattr(self, name).value
            if shape <= 0.0:
                raise ValueError(f"{name} must be > 0, got {shape!r}")
        return self

    @model_validator(mode="after")
    def _check_bank_tiers_complete(self) -> CalibrationSet:
        availability = self.bank_availability_by_tier.value
        missing = set(BankTier) - set(availability)
        if missing:
            raise ValueError(
                f"bank_availability_by_tier is missing tiers: "
                f"{sorted(t.value for t in missing)}"
            )
        return self

    @model_validator(mode="after")
    def _check_probabilities_in_range(self) -> CalibrationSet:
        for name, calibrated in self.parameters().items():
            for label, number in _probability_like(name, calibrated.value):
                if not 0.0 <= number <= 1.0:
                    raise ValueError(
                        f"{label} is a probability but is {number!r}"
                    )
        return self

    # ----------------------------------------------------------------------
    # Access
    # ----------------------------------------------------------------------

    def parameters(self) -> dict[str, CalibratedValue[Any]]:
        """Every parameter by name, for stored configs and provenance dumps."""
        return {name: getattr(self, name) for name in type(self).model_fields}


def _probability_like(name: str, value: Any) -> list[tuple[str, float]]:
    """Numbers on this parameter that must lie in [0, 1].

    Costs are paise and unbounded, as are the parameters named in
    _UNBOUNDED_PARAMETERS. Everything else named as a rate, share, probability
    or distribution weight is bounded.
    """
    if "paise" in name or name in _UNBOUNDED_PARAMETERS:
        return []
    if isinstance(value, dict):
        return [(f"{name}[{key!r}]", number) for key, number in value.items()]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [(name, float(value))]
    return []


#: The calibration the harness runs with unless an experiment overrides it.
DEFAULT_CALIBRATION: Final = CalibrationSet()
