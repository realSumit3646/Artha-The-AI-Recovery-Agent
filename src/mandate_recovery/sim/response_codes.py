"""Bank response codes, and the mess that makes diagnosis a real problem.

The simulator knows the true cause of every failure. If it handed back a clean
one-to-one code for each cause, any policy could invert it with a dictionary
and a "diagnosis" stage would be theatre -- an elaborate way of reading a
label the simulator had already written. Real merchants do not get clean
codes. This module reproduces that, deliberately.

Three sources of mess, all applied to failures only:

* **Bank-specific vocabularies.** The same cause reads differently depending
  on who declined it. "No funds" is ``AB1200`` at one bank, ``PS-51`` at
  another and ``SF_NOFUNDS`` at a third.
* **A generic bucket.** A share of failures come back as a single
  uninformative ``DECLINED``, which could be any of several causes.
* **Missing codes.** A smaller share come back empty or as ``NA``.
* **Contradictions.** A limit breach sometimes reports a funds code. From the
  code alone that case is indistinguishable from a genuine funds failure; it
  can only be caught by noticing that this customer has previously succeeded
  at a *higher* amount, which is why
  ``Observation.max_historical_success_amount_paise`` exists.

``SUCCESS`` and ``MANDATE_REVOKED`` are left clean. Neither is a diagnostic
puzzle: a merchant knows perfectly well whether money arrived and whether they
still hold a mandate.

**Every share in this module is an author's assumption.** No public dataset of
bank code ambiguity exists, and none is claimed. See ``docs/MESSINESS.md``.
"""

from __future__ import annotations

from typing import Final, Mapping

import numpy as np

from ..calibration import BankTier
from ..types import AttemptOutcome

__all__ = [
    "GENERIC_CODE",
    "MISSING_CODES",
    "BANK_CODE_VOCABULARY",
    "SHARE_OF_FAILURES_GENERIC",
    "SHARE_OF_FAILURES_MISSING",
    "SHARE_OF_LIMIT_FAILURES_MISCODED_AS_FUNDS",
    "TRUE_CAUSE_RECOVERABLE_FRACTION",
    "encode_response",
]


# --------------------------------------------------------------------------
# The vocabularies
# --------------------------------------------------------------------------

#: The uninformative code. Several causes collapse onto this one string.
GENERIC_CODE: Final = "DECLINED"

#: What a missing code looks like coming off a real gateway.
MISSING_CODES: Final = ("", "NA")

#: Per-tier code vocabularies. **Synthetic.** These are not real NPCI, UPI or
#: issuer codes, and they are shaped differently per tier on purpose so that a
#: policy has to learn each bank's dialect rather than one global table.
BANK_CODE_VOCABULARY: Final[Mapping[BankTier, Mapping[AttemptOutcome, str]]] = {
    BankTier.LARGE_PRIVATE: {
        AttemptOutcome.SUCCESS: "AB0000",
        AttemptOutcome.INSUFFICIENT_FUNDS: "AB1200",
        AttemptOutcome.TECHNICAL_DECLINE: "AB9001",
        AttemptOutcome.LIMIT_EXCEEDED: "AB3301",
        AttemptOutcome.WINDOW_REJECTED: "AB7702",
        AttemptOutcome.MANDATE_REVOKED: "AB6600",
    },
    BankTier.PSU: {
        AttemptOutcome.SUCCESS: "PS-00",
        AttemptOutcome.INSUFFICIENT_FUNDS: "PS-51",
        AttemptOutcome.TECHNICAL_DECLINE: "PS-91",
        AttemptOutcome.LIMIT_EXCEEDED: "PS-61",
        AttemptOutcome.WINDOW_REJECTED: "PS-77",
        AttemptOutcome.MANDATE_REVOKED: "PS-14",
    },
    BankTier.SMALL_FINANCE: {
        AttemptOutcome.SUCCESS: "SF_OK",
        AttemptOutcome.INSUFFICIENT_FUNDS: "SF_NOFUNDS",
        AttemptOutcome.TECHNICAL_DECLINE: "SF_SYSERR",
        AttemptOutcome.LIMIT_EXCEEDED: "SF_CAP",
        AttemptOutcome.WINDOW_REJECTED: "SF_PEAK",
        AttemptOutcome.MANDATE_REVOKED: "SF_CANCELLED",
    },
}

#: Outcomes that are reported cleanly, without messiness applied.
_CLEAN_OUTCOMES: Final = frozenset(
    {AttemptOutcome.SUCCESS, AttemptOutcome.MANDATE_REVOKED}
)


# --------------------------------------------------------------------------
# The messiness shares (all author's assumptions)
# --------------------------------------------------------------------------

#: Share of failures reported as the uninformative generic code.
SHARE_OF_FAILURES_GENERIC: Final = 0.18

#: Share of failures reported with no usable code at all.
SHARE_OF_FAILURES_MISSING: Final = 0.05

#: Share of specifically-coded limit breaches that report a *funds* code
#: instead. This is the contradiction case, and it is the reason a lookup
#: table cannot be trusted on a funds code.
SHARE_OF_LIMIT_FAILURES_MISCODED_AS_FUNDS: Final = 0.25

#: Share of failures whose code identifies exactly one true cause.
#:
#: Computed under the default calibration's failure mix. A code counts as
#: recoverable only if no other cause ever emits it: the generic code and the
#: missing codes are shared by construction, and a funds code is shared with
#: miscoded limit breaches, so only technical, window and cleanly-coded limit
#: failures qualify. ``tests/sim/test_response_codes.py`` recomputes this
#: empirically and asserts it stays below 0.75 -- above that, a lookup table
#: would resolve most cases and a diagnosis stage would not be justifiable.
TRUE_CAUSE_RECOVERABLE_FRACTION: Final = 0.33


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def encode_response(
    outcome: AttemptOutcome, bank_id: str, rng: np.random.Generator
) -> str:
    """The code a bank reports for this outcome. Messy on purpose.

    ``SUCCESS`` and ``MANDATE_REVOKED`` are always reported cleanly. Every
    other outcome is subject to the generic bucket, missing codes, and -- for
    limit breaches -- the funds-code contradiction.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an explicit numpy.random.Generator; the harness "
            "never uses global random state"
        )

    try:
        tier = BankTier(bank_id)
    except ValueError:
        raise KeyError(
            f"unknown bank_id {bank_id!r}; known banks are "
            f"{sorted(tier.value for tier in BankTier)}"
        ) from None

    vocabulary = BANK_CODE_VOCABULARY[tier]

    if outcome in _CLEAN_OUTCOMES:
        return vocabulary[outcome]

    roll = float(rng.random())
    if roll < SHARE_OF_FAILURES_MISSING:
        index = int(rng.integers(len(MISSING_CODES)))
        return MISSING_CODES[index]
    if roll < SHARE_OF_FAILURES_MISSING + SHARE_OF_FAILURES_GENERIC:
        return GENERIC_CODE

    if outcome is AttemptOutcome.LIMIT_EXCEEDED:
        if float(rng.random()) < SHARE_OF_LIMIT_FAILURES_MISCODED_AS_FUNDS:
            return vocabulary[AttemptOutcome.INSUFFICIENT_FUNDS]

    return vocabulary[outcome]
