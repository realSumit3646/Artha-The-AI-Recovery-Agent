"""The policy interface, and the boundary it enforces.

A policy decides what to do about a failing mandate. It gets exactly one
input:

    decide(self, observation: Observation) -> Decision

That signature is the information-asymmetry invariant expressed as a type. No
policy receives a :class:`~mandate_recovery.sim.world.World`, a
:class:`~mandate_recovery.types.LatentCustomerState`, or anything else from
the ``sim`` package. It cannot see the balance, the salary date, the bank's
uptime draw or the customer's churn intent, because a real collector cannot
see them either.

This is not left to discipline. Every subclass registers itself in
:data:`POLICY_REGISTRY`, and ``tests/policies/test_base.py`` walks the
registry and fails if any policy's ``decide`` or ``__init__`` so much as
mentions a simulator type. Adding a policy that peeks breaks the suite.

The one permitted exception
---------------------------
A policy may set ``reads_latent_state = True`` to opt out. Exactly one thing
in this project is entitled to: the oracle, which exists to compute an upper
bound on what perfect information could achieve, and is not a shippable
policy. The declaration is a class attribute rather than a line in a test, so
the exemption is visible in the code that uses it. Anything declaring it must
say in its own docstring that it is an upper-bound instrument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..types import Decision, DecisionSource, Observation

__all__ = ["POLICY_REGISTRY", "Policy"]

#: Every policy class defined anywhere in the project, by name. Populated on
#: subclass creation so the boundary test cannot miss one.
POLICY_REGISTRY: dict[str, type["Policy"]] = {}


class Policy(ABC):
    """Base class for every recovery policy.

    Subclasses implement :meth:`decide` and nothing else is required. State
    that must persist across mandates belongs on the instance; the harness
    constructs one policy per arm per seed.
    """

    #: Short identifier used as the arm name in stored results.
    name: ClassVar[str] = ""

    #: Opt-out from the observation boundary. See the module docstring. If you
    #: are setting this on something you intend to ship, you are wrong.
    reads_latent_state: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        POLICY_REGISTRY[cls.name or cls.__name__] = cls

    @abstractmethod
    def decide(self, observation: Observation) -> Decision:
        """Choose what to do about this mandate, right now.

        The observation is everything the policy is allowed to know. Returning
        a :class:`Decision` with ``validated=False`` is correct: approval is
        the validator's job, not the policy's.
        """

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def decision(
        observation: Observation,
        action,
        *,
        rationale: str,
        source: DecisionSource = "rule",
        validated: bool = False,
    ) -> Decision:
        """Build a :class:`Decision`, defaulting to unvalidated.

        Every policy goes through here so that no policy can accidentally mark
        its own action approved.
        """
        return Decision(
            observation=observation,
            action=action,
            source=source,
            rationale=rationale,
            validated=validated,
        )

    def reset(self) -> None:
        """Clear any per-run state. Called by the harness before each seed."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
