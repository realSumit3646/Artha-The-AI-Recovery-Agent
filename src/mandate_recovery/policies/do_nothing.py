"""The no-intervention arm: the floor every other policy is measured against.

It does nothing. Every failed mandate is abandoned where it fell. This is not
a strawman — it is the honest counterfactual. A merchant who never chases a
failed autopay still recovers whatever the customer pays voluntarily on the
next cycle, and pays nothing to get it.

Any policy that cannot beat this one is worse than switching recovery off.
"""

from __future__ import annotations

from typing import ClassVar

from ..types import Decision, Observation, Stop
from .base import Policy

__all__ = ["DoNothingPolicy"]


class DoNothingPolicy(Policy):
    """Always stops. The loss floor and the cost floor at once."""

    name: ClassVar[str] = "do_nothing"

    def decide(self, observation: Observation) -> Decision:
        return self.decision(
            observation,
            Stop(reason="no intervention arm"),
            rationale=(
                "The no-intervention arm never acts. This mandate is left to "
                "recover on its own or not at all."
            ),
        )
