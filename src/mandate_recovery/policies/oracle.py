"""The perfect-information oracle.

**THIS IS AN UPPER-BOUND INSTRUMENT. IT IS NOT A SHIPPABLE POLICY.**

**IT READS SIMULATOR-PRIVATE STATE AND COULD NOT EXIST IN PRODUCTION.** It
knows every customer's exact balance on every future day, knows which hours
each bank will be up, and knows the per-transaction ceiling. No merchant has
any of that. Nothing in this file describes a strategy anyone could deploy.

It exists to answer one question: **how much of the lost money is recoverable
at all by better timing?** Without that number, a policy that recovers 40% of
failures is unreadable — it could be leaving almost nothing on the table or
almost everything. The oracle turns the comparison from "our agent beat the
baseline" into "our agent captured N% of the available headroom", which is the
only version of the claim worth making.

It is the ceiling, `DoNothingPolicy` is the floor, and every real policy is
scored in the gap between them.

The exemption
-------------
This is the only class in the project that sets ``reads_latent_state = True``.
That flag takes it out of the boundary checks in
``tests/policies/test_base.py`` — deliberately, and for this class alone. The
`World` arrives through the **constructor**, never through ``decide``, so the
policy interface itself stays clean and a test still asserts that ``decide``
takes nothing but an `Observation`.

What it optimises
-----------------
The first future hour at which the debit would actually succeed: balance
covers the amount, the bank is up, and the hour is outside the restricted
window. It only retries — it never contacts the customer, so it is an upper
bound on *timing* alone, not on recovery in general. A policy that nudges
someone into topping up their account could in principle beat it, and if one
ever does, that is a real finding rather than a bug.
"""

from __future__ import annotations

from typing import ClassVar, Mapping

from ..sim.world import DAYS_IN_MONTH, HOURS_IN_DAY, World
from ..types import Decision, Observation, Rail, RetrySilent, Stop
from .base import Policy

__all__ = ["OraclePolicy"]


class OraclePolicy(Policy):
    """Retries at the first hour the debit would actually go through.

    **UPPER BOUND INSTRUMENT — reads latent state, not shippable.** See the
    module docstring.
    """

    name: ClassVar[str] = "oracle"

    #: The one permitted exemption from the observation boundary.
    reads_latent_state: ClassVar[bool] = True

    def __init__(
        self,
        world: World,
        customer_id_by_mandate_id: Mapping[str, str],
        *,
        max_days_ahead: int = DAYS_IN_MONTH,
        rail: Rail = Rail.UPI_AUTOPAY,
    ) -> None:
        """
        Args:
            world: simulator-private ground truth, injected explicitly so the
                cheat is visible at the call site rather than hidden.
            customer_id_by_mandate_id: which customer each mandate belongs to.
                The `Observation` carries only a mandate id, and resolving it
                is the harness's job.
            max_days_ahead: how far to look before declaring the mandate
                unrecoverable. One pay cycle by default: beyond that the next
                cycle supersedes this one.
        """
        if max_days_ahead < 1:
            raise ValueError(f"max_days_ahead must be >= 1, got {max_days_ahead}")

        self._world = world
        self._customer_id_by_mandate_id = dict(customer_id_by_mandate_id)
        self._max_days_ahead = int(max_days_ahead)
        self._rail = rail

    # ------------------------------------------------------------------
    # Projection over latent state
    # ------------------------------------------------------------------

    def _projected_balance_paise(self, customer_index: int, day: int) -> int:
        """The customer's balance on a future day, if nothing debits them.

        Replays the world's own balance arithmetic forward: salary credited at
        the start of the day, then the day's spend, floored at zero. Ignores
        other mandates, which makes this an optimistic projection — correct
        for an upper bound.
        """
        world = self._world
        latent = world.latent_state(customer_index)
        balance = latent.balance_paise

        for step in range(world.current_day + 1, day + 1):
            day_of_month = (step % DAYS_IN_MONTH) + 1
            if day_of_month == latent.salary_day:
                balance += latent.salary_amount_paise
            balance = max(0, balance - latent.spend_rate_paise_per_day)
        return balance

    # ------------------------------------------------------------------
    # The policy interface
    # ------------------------------------------------------------------

    def decide(self, observation: Observation) -> Decision:
        world = self._world
        customer_id = self._customer_id_by_mandate_id.get(observation.mandate_id)
        if customer_id is None:
            raise KeyError(
                f"oracle has no customer mapped for mandate "
                f"{observation.mandate_id!r}"
            )
        customer_index = world.index_for_customer_id(customer_id)
        latent = world.latent_state(customer_index)
        bank_id = world.bank_id_for(customer_index)
        amount = observation.amount_paise

        # Perfect information includes knowing what is simply impossible.
        if amount > latent.per_txn_limit_paise:
            return self.decision(
                observation,
                Stop(reason="amount exceeds the per-transaction ceiling"),
                rationale=(
                    f"The mandate is for {amount} paise against a ceiling of "
                    f"{latent.per_txn_limit_paise}. No retry at any hour can "
                    "succeed, so the ceiling is the binding constraint."
                ),
            )

        last_day = min(
            world.current_day + self._max_days_ahead, world.n_days - 1
        )
        for day in range(world.current_day, last_day + 1):
            if self._projected_balance_paise(customer_index, day) < amount:
                continue
            for hour in range(HOURS_IN_DAY):
                if world.in_restricted_window(hour):
                    continue
                if not world.bank_available(bank_id, day, hour):
                    continue
                return self.decision(
                    observation,
                    RetrySilent(
                        scheduled_day=day,
                        scheduled_hour=hour,
                        rail=self._rail,
                    ),
                    rationale=(
                        f"Perfect information: on day {day} at "
                        f"{hour:02d}:00 the balance covers the amount, "
                        f"{bank_id} is available, and the hour is outside the "
                        "restricted window. This is the earliest slot at "
                        "which the debit would actually succeed."
                    ),
                )

        return self.decision(
            observation,
            Stop(reason="no recoverable slot before the mandate lapses"),
            rationale=(
                f"Searched every hour from day {world.current_day} to "
                f"{last_day}. The balance never covers {amount} paise at an "
                "hour when the bank is up and the window is clear, so this "
                "mandate is not recoverable by timing alone."
            ),
        )
