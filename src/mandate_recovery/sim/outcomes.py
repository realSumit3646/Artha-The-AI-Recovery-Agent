"""Attempt resolution: what actually happens when a debit is presented.

This is the heart of the simulator. Given the latent world -- balance,
per-transaction ceiling, bank uptime, the calendar -- it decides whether a
debit succeeds, and if not, why.

Nothing here is visible to a policy. A policy learns only what a real
collector would learn: the bank's ``raw_code`` on the resulting
:class:`BankResponse`. It never sees the balance that caused an
``INSUFFICIENT_FUNDS``, nor the uptime draw behind a ``TECHNICAL_DECLINE``.

Resolution order is fixed and total. Each check answers a question a real
rail answers in roughly this sequence, and the first one that fails ends the
attempt:

1. Is the mandate still live?          -> ``MANDATE_REVOKED``
2. Is this a deprioritised hour?       -> ``WINDOW_REJECTED`` (probabilistic)
3. Is the issuer up?                   -> ``TECHNICAL_DECLINE``
4. Is the amount within the ceiling?   -> ``LIMIT_EXCEEDED``
5. Is there money in the account?      -> ``INSUFFICIENT_FUNDS``
6. Otherwise                           -> ``SUCCESS``, and the money moves.

The order matters and is not arbitrary: a revoked mandate is never presented,
a rejected window means the bank never looked at the account, and a limit
breach is refused before the balance is consulted. A customer with an empty
account and an over-limit amount reports ``LIMIT_EXCEEDED``, not
``INSUFFICIENT_FUNDS``.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

import numpy as np

from ..calibration import CalibrationSet
from ..types import (
    Attempt,
    AttemptOutcome,
    BankResponse,
    Mandate,
    MandateStatus,
)
from .world import DAYS_IN_MONTH, World

__all__ = [
    "SYNTHETIC_RAW_CODES",
    "MIN_INSUFFICIENT_FUNDS_FAILURES_FOR_REVOCATION",
    "resolve_attempt",
    "daily_revocation_probability",
    "revoke_eligible_mandates",
]


# --------------------------------------------------------------------------
# Bank response codes
# --------------------------------------------------------------------------

#: Codes returned to policies in place of the classified outcome.
#:
#: **These are SYNTHETIC. They are not real NPCI, UPI or issuer codes**, and
#: they are deliberately named so that nobody mistakes them for real ones.
#: A real bank returns a noisier vocabulary, with several codes per condition
#: and some genuine ambiguity between them.
#:
#: TODO(sumit): replace with a real code mapping before any result is quoted.
#: Note that the mapping here is one-to-one, so a policy can invert a code
#: back to the exact outcome. That is defensible -- gateways really do report
#: decline reasons -- but it makes the simulator kinder than reality, where
#: a technical decline is sometimes miscoded as a funds failure.
SYNTHETIC_RAW_CODES: Final[Mapping[AttemptOutcome, str]] = {
    AttemptOutcome.SUCCESS: "SIM_OK",
    AttemptOutcome.INSUFFICIENT_FUNDS: "SIM_NSF",
    AttemptOutcome.LIMIT_EXCEEDED: "SIM_LIMIT",
    AttemptOutcome.TECHNICAL_DECLINE: "SIM_TECH",
    AttemptOutcome.WINDOW_REJECTED: "SIM_WINDOW",
    AttemptOutcome.MANDATE_REVOKED: "SIM_REVOKED",
}

#: How many insufficient-funds failures make a mandate a revocation candidate.
#:
#: "Repeatedly failed" is read as more than once. This is a structural reading
#: of the word, not a calibrated figure; if a result turns out to depend on
#: it, it belongs in `CalibrationSet` so the sweep can move it.
MIN_INSUFFICIENT_FUNDS_FAILURES_FOR_REVOCATION: Final = 2


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def resolve_attempt(
    world: World,
    mandate: Mandate,
    attempt: Attempt,
    rng: np.random.Generator,
) -> BankResponse:
    """Decide what the bank does with one presented debit.

    On ``SUCCESS`` the amount is deducted from the customer's latent balance;
    every other outcome leaves the world unchanged.

    The attempt is resolved on the world's *current* day. Only the hour is
    read from ``attempt.scheduled_at`` -- the date on it is not consulted,
    because the world counts days from its own start and has no calendar
    epoch to compare a ``datetime`` against. Callers advance the world to the
    day they mean to resolve on.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an explicit numpy.random.Generator; the harness "
            "never uses global random state"
        )

    customer_index = world.index_for_customer_id(mandate.customer_id)
    calibration = world.calibration
    hour = attempt.scheduled_at.hour
    day = world.current_day

    def responded(outcome: AttemptOutcome) -> BankResponse:
        return BankResponse(
            raw_code=SYNTHETIC_RAW_CODES[outcome],
            outcome=outcome,
            bank_id=world.bank_id_for(customer_index),
            timestamp=attempt.scheduled_at,
        )

    # 1. A revoked mandate is never presented to the rail at all.
    if mandate.status is not MandateStatus.ACTIVE:
        return responded(AttemptOutcome.MANDATE_REVOKED)

    # 2. Peak hours: recurring debits are deprioritised, not banned, so this
    #    is a probability rather than a hard block.
    if world.in_restricted_window(hour):
        rejection = calibration.restricted_window_rejection_probability.value
        if float(rng.random()) < rejection:
            return responded(AttemptOutcome.WINDOW_REJECTED)

    # 3. The issuer has to be up for anyone to look at the account.
    if not world.bank_available(world.bank_id_for(customer_index), day, hour):
        return responded(AttemptOutcome.TECHNICAL_DECLINE)

    latent = world.latent_state(customer_index)

    # 4. The ceiling is checked before the balance: a bank refuses an
    #    over-limit mandate without reference to what is in the account.
    if mandate.amount_paise > latent.per_txn_limit_paise:
        return responded(AttemptOutcome.LIMIT_EXCEEDED)

    # 5. Then, and only then, is there enough money?
    if mandate.amount_paise > latent.balance_paise:
        return responded(AttemptOutcome.INSUFFICIENT_FUNDS)

    # 6. The money moves.
    world.debit(customer_index, mandate.amount_paise)
    return responded(AttemptOutcome.SUCCESS)


# --------------------------------------------------------------------------
# Revocation
# --------------------------------------------------------------------------


def daily_revocation_probability(calibration: CalibrationSet) -> float:
    """The per-day hazard equivalent to the calibrated monthly rate.

    Converted as ``1 - (1 - monthly) ** (1 / 31)`` rather than
    ``monthly / 31``, so that compounding over a month reproduces the monthly
    figure instead of overshooting it.
    """
    monthly = calibration.monthly_mandate_revocation_rate.value
    return 1.0 - (1.0 - monthly) ** (1.0 / DAYS_IN_MONTH)


def revoke_eligible_mandates(
    mandates: Sequence[Mandate],
    insufficient_funds_failures: Mapping[str, int],
    calibration: CalibrationSet,
    rng: np.random.Generator,
    *,
    min_failures: int = MIN_INSUFFICIENT_FUNDS_FAILURES_FOR_REVOCATION,
) -> tuple[Mandate, ...]:
    """Revoke a day's worth of mandates, returning the updated sequence.

    A customer who has been told "insufficient funds" repeatedly is the one
    who cancels the mandate. Eligibility is therefore conditional on having
    failed for funds at least ``min_failures`` times; the hazard applied to
    eligible mandates is the calibrated monthly rate expressed per day.

    Mandates that are not ACTIVE, and mandates whose customer has not failed
    often enough, are returned untouched.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an explicit numpy.random.Generator; the harness "
            "never uses global random state"
        )

    hazard = daily_revocation_probability(calibration)
    updated: list[Mandate] = []

    for mandate in mandates:
        eligible = (
            mandate.status is MandateStatus.ACTIVE
            and insufficient_funds_failures.get(mandate.id, 0) >= min_failures
        )
        # The draw is taken for every eligible mandate, in order, so that the
        # stream does not depend on which of them happen to be revoked.
        if eligible and float(rng.random()) < hazard:
            updated.append(
                mandate.model_copy(update={"status": MandateStatus.REVOKED})
            )
        else:
            updated.append(mandate)

    return tuple(updated)
