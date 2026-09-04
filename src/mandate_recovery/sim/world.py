"""The simulator's private world: customers, balances, banks, the calendar.

Everything in this module is ground truth. None of it is visible to a policy.

The calendar is a uniform 31-day month: simulation day ``d`` falls on day of
month ``(d % 31) + 1``. Real months vary and February has no 31st, but the
calibrated salary distribution puts mass on days 1-7 and 25-31, and a uniform
month keeps every one of those days reachable in every cycle rather than
silently folding month-end mass onto the 28th.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from ..calibration import BankTier, CalibrationSet
from ..types import LatentCustomerState

__all__ = ["DAYS_IN_MONTH", "HOURS_IN_DAY", "World"]

#: Length of the simulated month. See the module docstring.
DAYS_IN_MONTH: Final = 31

HOURS_IN_DAY: Final = 24


class World:
    """The latent state of the simulated world.

    **THIS OBJECT IS SIMULATOR-PRIVATE AND MUST NEVER BE PASSED TO A POLICY.**

    It holds exactly the things a policy is forbidden to know: every
    customer's current balance, the day their salary lands, how fast they
    spend, how close they are to leaving, their per-transaction ceiling, and
    whether their bank is up at a given hour. A policy that could read this
    object would be able to time a retry perfectly, and the experiment would
    measure nothing.

    So: no ``World``, no :class:`LatentCustomerState`, and no value read off
    either may be passed to a policy, embedded in an ``Observation``, or
    rendered into an LLM prompt. The simulator uses this to decide what
    *happens*; it builds a separate, deliberately impoverished ``Observation``
    to tell a policy what is *visible*. If a policy needs something from here,
    the answer is no.

    Determinism
    -----------
    Every stochastic draw happens once, in ``__init__``, from the caller's
    explicit ``numpy.random.Generator``. Two worlds built from equally seeded
    generators with the same calibration are identical, and stay identical
    under the same sequence of calls. No global random state is touched.

    Bank availability is pre-drawn for the whole run rather than sampled per
    call, so ``bank_available`` is a lookup. Sampling on demand would make the
    answer depend on how many times it had been asked, which would mean a
    policy that retries more often would silently shift the bank's uptime.
    """

    def __init__(
        self,
        calibration: CalibrationSet,
        rng: np.random.Generator,
        n_customers: int,
        n_days: int,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be an explicit numpy.random.Generator; the harness "
                "never uses global random state"
            )
        if n_customers < 1:
            raise ValueError(f"n_customers must be >= 1, got {n_customers}")
        if n_days < 1:
            raise ValueError(f"n_days must be >= 1, got {n_days}")

        self._calibration = calibration
        self._n_customers = int(n_customers)
        self._n_days = int(n_days)
        self._day = 0

        self._sample_population(rng)
        self._draw_bank_availability(rng)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _sample_population(self, rng: np.random.Generator) -> None:
        """Draw the customer population. All money lands as integer paise."""
        n = self._n_customers
        cal = self._calibration

        # Bank tier, in the enum's declaration order so the mapping from
        # sampled index to tier is stable across runs.
        self._tiers: tuple[BankTier, ...] = tuple(BankTier)
        mix = cal.bank_tier_mix.value
        tier_probs = np.array([mix[tier] for tier in self._tiers], dtype=float)
        self._customer_tier_index = rng.choice(
            len(self._tiers), size=n, p=tier_probs
        )

        # Salary day, from the calibrated day-of-month distribution.
        weights = cal.salary_credit_day_distribution.value
        days = np.array(sorted(weights), dtype=np.int64)
        day_probs = np.array([weights[int(d)] for d in days], dtype=float)
        self._salary_days = rng.choice(days, size=n, p=day_probs)

        # Salary amount: lognormal with the calibrated median.
        median = cal.monthly_salary_paise_median.value
        sigma = cal.monthly_salary_lognormal_sigma.value
        drawn = rng.lognormal(mean=math.log(median), sigma=sigma, size=n)
        self._salary_amounts_paise = np.rint(drawn).astype(np.int64)
        np.maximum(self._salary_amounts_paise, 0, out=self._salary_amounts_paise)

        # Daily spend is derived from salary rather than drawn independently,
        # so a customer's outgoings track their income.
        share = cal.monthly_spend_share_of_salary.value
        self._spend_rates_paise = np.rint(
            self._salary_amounts_paise * share / DAYS_IN_MONTH
        ).astype(np.int64)
        np.maximum(self._spend_rates_paise, 0, out=self._spend_rates_paise)

        self._churn_intents = rng.beta(
            cal.initial_churn_intent_alpha.value,
            cal.initial_churn_intent_beta.value,
            size=n,
        )

        limits = cal.per_txn_limit_paise_by_tier.value
        limit_by_index = np.array(
            [limits[tier] for tier in self._tiers], dtype=np.int64
        )
        self._per_txn_limits_paise = limit_by_index[self._customer_tier_index]

        self._balances_paise = self._opening_balances()

    def _opening_balances(self) -> np.ndarray:
        """Balances on day 0, wound forward from the last salary credit.

        Starting everyone at a full salary would put the whole population in
        lockstep on day 0 and make the first cycle unrepresentative. Instead
        each customer starts however far through their own pay cycle day 0
        happens to fall.
        """
        days_since_salary = (
            self.day_of_month - self._salary_days
        ) % DAYS_IN_MONTH
        spent = self._spend_rates_paise * days_since_salary
        return np.maximum(self._salary_amounts_paise - spent, 0)

    def _draw_bank_availability(self, rng: np.random.Generator) -> None:
        """Pre-draw uptime for every bank, day and hour of the run.

        One representative bank per tier: ``bank_id`` is the tier's value.
        Modelling several banks per tier would need a bank count, and there is
        no calibrated figure for one.
        """
        availability = self._calibration.bank_availability_by_tier.value
        rates = np.array(
            [availability[tier] for tier in self._tiers], dtype=float
        )
        draws = rng.random((len(self._tiers), self._n_days, HOURS_IN_DAY))
        self._bank_up = draws < rates[:, None, None]
        self._bank_index_by_id = {
            tier.value: index for index, tier in enumerate(self._tiers)
        }

    # ------------------------------------------------------------------
    # The calendar
    # ------------------------------------------------------------------

    @property
    def current_day(self) -> int:
        """Simulation day index, starting at 0."""
        return self._day

    @property
    def day_of_month(self) -> int:
        """Day of month for the current simulation day, in 1..31."""
        return (self._day % DAYS_IN_MONTH) + 1

    @property
    def n_customers(self) -> int:
        return self._n_customers

    @property
    def n_days(self) -> int:
        return self._n_days

    def advance_day(self) -> int:
        """Step the world forward one day and return the new day index.

        Salary is credited at the start of the day, then the day's spending is
        taken out. Balances are floored at zero: a customer with no money has
        no money, they do not go overdrawn.
        """
        if self._day + 1 >= self._n_days:
            raise RuntimeError(
                f"cannot advance past the end of the run: day {self._day} of "
                f"n_days={self._n_days}"
            )

        self._day += 1
        credited = np.where(
            self._salary_days == self.day_of_month,
            self._salary_amounts_paise,
            0,
        )
        self._balances_paise = np.maximum(
            self._balances_paise + credited - self._spend_rates_paise, 0
        )
        return self._day

    # ------------------------------------------------------------------
    # Latent state (simulator-only reads)
    # ------------------------------------------------------------------

    def latent_state(self, customer_index: int) -> LatentCustomerState:
        """Ground truth for one customer. Never hand this to a policy."""
        self._check_customer(customer_index)
        return LatentCustomerState(
            balance_paise=int(self._balances_paise[customer_index]),
            salary_day=int(self._salary_days[customer_index]),
            salary_amount_paise=int(
                self._salary_amounts_paise[customer_index]
            ),
            spend_rate_paise_per_day=int(
                self._spend_rates_paise[customer_index]
            ),
            churn_intent=float(self._churn_intents[customer_index]),
            per_txn_limit_paise=int(
                self._per_txn_limits_paise[customer_index]
            ),
        )

    def balance_paise(self, customer_index: int) -> int:
        self._check_customer(customer_index)
        return int(self._balances_paise[customer_index])

    def balances_paise(self) -> tuple[int, ...]:
        """Every customer's balance today, as a snapshot copy."""
        return tuple(int(balance) for balance in self._balances_paise)

    def bank_tier_for(self, customer_index: int) -> BankTier:
        self._check_customer(customer_index)
        return self._tiers[self._customer_tier_index[customer_index]]

    def bank_id_for(self, customer_index: int) -> str:
        return self.bank_tier_for(customer_index).value

    # ------------------------------------------------------------------
    # Banks and the restricted window
    # ------------------------------------------------------------------

    def bank_available(self, bank_id: str, day: int, hour: int) -> bool:
        """Whether ``bank_id`` is up at this day and hour.

        Pre-drawn at construction, so asking twice gives the same answer.
        """
        try:
            index = self._bank_index_by_id[bank_id]
        except KeyError:
            raise KeyError(
                f"unknown bank_id {bank_id!r}; known banks are "
                f"{sorted(self._bank_index_by_id)}"
            ) from None
        if not 0 <= day < self._n_days:
            raise IndexError(
                f"day {day} is outside the run (0..{self._n_days - 1})"
            )
        if not 0 <= hour < HOURS_IN_DAY:
            raise IndexError(f"hour {hour} is outside 0..{HOURS_IN_DAY - 1}")
        return bool(self._bank_up[index, day, hour])

    def in_restricted_window(self, hour: int) -> bool:
        """Whether ``hour`` falls in a window where recurring debits are
        deprioritised. Windows are half-open ``[start, end)`` clock hours."""
        if not 0 <= hour < HOURS_IN_DAY:
            raise IndexError(f"hour {hour} is outside 0..{HOURS_IN_DAY - 1}")
        return any(
            start <= hour < end
            for start, end in self._calibration.restricted_window_hours.value
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_customer(self, customer_index: int) -> None:
        if not 0 <= customer_index < self._n_customers:
            raise IndexError(
                f"customer_index {customer_index} is outside "
                f"0..{self._n_customers - 1}"
            )
