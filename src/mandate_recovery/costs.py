"""What an intervention costs, and what recovery is actually worth.

Without a cost model the optimal recovery policy is to retry forever and phone
the customer every morning. That policy would score beautifully on recovery
rate, be illegal under NPCI attempt limits, and destroy the customer base it
was collecting from. This module is what stops the experiment rewarding it.

Three costs are charged:

* **Gateway** -- every attempt costs money whether or not it succeeds. This is
  what makes retrying-forever expensive.
* **Contact** -- every SMS and every voice call has a price.
* **Churn** -- every contact raises the probability the customer leaves. The
  expected cost is that probability multiplied by what the mandate would have
  been worth over its remaining cycles. This is the one that hurts, and it is
  the reason a policy cannot buy recovery with nagging.

And one flag, which is not a cost but a diagnosis:

* **Over-intervention** -- the customer was contacted or escalated, and the
  counterfactual run on the same seed shows they would have paid on the next
  natural cycle anyway. The money was recovered, the cost was real, and the
  intervention bought nothing. A policy can look good on net recovery while
  over-intervening constantly, so the rate is reported separately.

``net_recovery_paise = recovered_paise - total_cost_paise`` is the headline
metric of this project. Recovery rate, attempts per recovery and days to
recovery are all secondary to it: they are diagnostics that explain the
headline, not competitors to it.

All amounts are integer paise. The churn term is an expectation over a
probability, so it is computed in float and rounded once, at the end, before
it re-enters the money path.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .calibration import DEFAULT_CALIBRATION, CalibrationSet
from .types import NonNegativePaise, Paise, PositivePaise

__all__ = ["Episode", "CostBreakdown", "CostModel"]

_Count = Annotated[int, Field(strict=True, ge=0)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Episode(_FrozenModel):
    """One mandate's complete history under one policy, ready to be scored.

    This is a summary of what happened, not a live object: the harness builds
    it after a mandate has run to completion or lapsed.
    """

    mandate_id: str
    amount_paise: PositivePaise

    #: Every debit presented, successful or not. Each one costs gateway fees.
    attempts: _Count = 0
    sms_sent: _Count = 0
    voice_calls_made: _Count = 0
    escalated_to_human: bool = False

    recovered_paise: NonNegativePaise = 0

    #: Cycles this mandate would still have run if the customer stays. Sets
    #: what churn actually costs; a mandate on its last cycle is cheap to lose.
    remaining_cycles: _Count = 0

    #: From the paired counterfactual run on the same seed: would this
    #: customer have paid on the next natural cycle with no intervention at
    #: all? Only meaningful when the policy did intervene.
    would_have_paid_without_intervention: bool = False


class CostBreakdown(_FrozenModel):
    """What one episode cost, and what it was worth."""

    mandate_id: str
    gateway_cost_paise: NonNegativePaise
    contact_cost_paise: NonNegativePaise
    churn_cost_paise: NonNegativePaise
    total_cost_paise: NonNegativePaise
    recovered_paise: NonNegativePaise

    #: The headline. Negative when a policy spent more than it collected.
    net_recovery_paise: Paise

    #: Probability the customer leaves *because of* this episode's contacts.
    churn_probability: float

    #: Contacted or escalated, when the counterfactual says they would have
    #: paid anyway.
    over_intervention: bool


class CostModel:
    """Scores episodes against a calibration.

    Holds no state between episodes, so the same model can score every arm of
    an experiment without one arm's history leaking into another's.
    """

    def __init__(self, calibration: CalibrationSet = DEFAULT_CALIBRATION) -> None:
        self._calibration = calibration

    @property
    def calibration(self) -> CalibrationSet:
        return self._calibration

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def gateway_cost_paise(self, episode: Episode) -> int:
        """Every attempt is billed, successful or not."""
        per_attempt = self._calibration.gateway_cost_per_attempt_paise.value
        return episode.attempts * per_attempt

    def contact_cost_paise(self, episode: Episode) -> int:
        """SMS and voice, at their calibrated unit prices."""
        sms = self._calibration.sms_cost_paise.value
        voice = self._calibration.voice_call_cost_paise.value
        return episode.sms_sent * sms + episode.voice_calls_made * voice

    def contacts_made(self, episode: Episode) -> int:
        """Everything the customer experienced as being contacted.

        Escalation counts: from the customer's side a human agent calling is a
        contact, and it carries the same churn risk. It carries no separate
        monetary charge here because no agent-time figure is calibrated.
        """
        return (
            episode.sms_sent
            + episode.voice_calls_made
            + (1 if episode.escalated_to_human else 0)
        )

    def churn_probability(self, episode: Episode) -> float:
        """Marginal probability the customer leaves because of this episode.

        Only the risk *this policy created* is charged. The customer's
        underlying churn intent is latent state; charging a policy for churn
        it did not cause would make the cost depend on something no policy can
        see, and would punish arms that happened to draw unhappy customers.
        """
        increment = (
            self._calibration.churn_probability_increment_per_contact.value
        )
        return min(1.0, self.contacts_made(episode) * increment)

    def remaining_lifetime_value_paise(self, episode: Episode) -> int:
        """What the mandate is still worth if the customer stays."""
        return episode.amount_paise * episode.remaining_cycles

    def churn_cost_paise(self, episode: Episode) -> int:
        """Expected value of the business lost to intervention-driven churn."""
        expected = self.churn_probability(
            episode
        ) * self.remaining_lifetime_value_paise(episode)
        return int(round(expected))

    def is_over_intervention(self, episode: Episode) -> bool:
        """Did the policy spend a contact on someone who would have paid?"""
        return (
            self.contacts_made(episode) > 0
            and episode.would_have_paid_without_intervention
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, episode: Episode) -> CostBreakdown:
        """The full breakdown for one episode."""
        gateway = self.gateway_cost_paise(episode)
        contact = self.contact_cost_paise(episode)
        churn = self.churn_cost_paise(episode)
        total = gateway + contact + churn

        return CostBreakdown(
            mandate_id=episode.mandate_id,
            gateway_cost_paise=gateway,
            contact_cost_paise=contact,
            churn_cost_paise=churn,
            total_cost_paise=total,
            recovered_paise=episode.recovered_paise,
            net_recovery_paise=episode.recovered_paise - total,
            churn_probability=self.churn_probability(episode),
            over_intervention=self.is_over_intervention(episode),
        )

    def score_all(self, episodes) -> tuple[CostBreakdown, ...]:
        return tuple(self.score(episode) for episode in episodes)

    def net_recovery_paise(self, episodes) -> int:
        """Headline metric across a set of episodes."""
        return sum(self.score(episode).net_recovery_paise for episode in episodes)
