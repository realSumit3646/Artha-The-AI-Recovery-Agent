"""The runtime enforcement of the observation boundary.

`tests/test_types.py` proves the `Observation` type carries no latent field.
This proves no *policy* can get at latent state by another route: not through
its `decide` signature, not through its constructor, not by importing the sim
package and reaching for it.

If someone adds a policy that peeks, this file is what fails.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import typing

import pytest

import mandate_recovery.policies as policies_package
from mandate_recovery.policies import POLICY_REGISTRY, DoNothingPolicy, Policy
from mandate_recovery.types import Decision, Observation, Stop

#: Names no policy may accept, in any argument, under any alias.
FORBIDDEN_TYPE_NAMES = {
    "World",
    "LatentCustomerState",
    "BankResponse",
    "AttemptOutcome",
}

#: Any annotation mentioning the simulator is forbidden outright.
FORBIDDEN_MODULE_MARKER = "mandate_recovery.sim"


def _import_every_policy_module() -> None:
    """Make sure the registry is complete before we inspect it."""
    for module in pkgutil.iter_modules(policies_package.__path__):
        importlib.import_module(f"{policies_package.__name__}.{module.name}")


_import_every_policy_module()


def _annotations_of(function) -> list[str]:
    """Every annotation on a callable, rendered as text."""
    try:
        hints = typing.get_type_hints(function)
    except Exception:  # noqa: BLE001 - unresolvable hints are inspected raw
        hints = getattr(function, "__annotations__", {})
    rendered = [str(value) for value in hints.values()]
    signature = inspect.signature(function)
    rendered.extend(
        str(parameter.annotation)
        for parameter in signature.parameters.values()
        if parameter.annotation is not inspect.Parameter.empty
    )
    return rendered


def _bounded_policies() -> list[type[Policy]]:
    """Policies subject to the boundary, i.e. everything but the oracle."""
    return [
        policy
        for policy in POLICY_REGISTRY.values()
        if not policy.reads_latent_state
    ]


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------


def test_the_registry_is_not_empty():
    """Guards every test below against passing because nothing was found."""
    assert POLICY_REGISTRY, "no policies registered"
    assert _bounded_policies(), "no boundary-bound policies registered"


@pytest.mark.parametrize("policy", _bounded_policies(), ids=lambda p: p.__name__)
def test_decide_accepts_only_an_observation(policy):
    """The signature *is* the invariant."""
    signature = inspect.signature(policy.decide)
    parameters = [
        name for name in signature.parameters if name not in ("self", "cls")
    ]
    assert parameters == ["observation"], (
        f"{policy.__name__}.decide takes {parameters}; it may take only an "
        "Observation"
    )

    hints = typing.get_type_hints(policy.decide)
    assert hints.get("observation") is Observation
    assert hints.get("return") is Decision


@pytest.mark.parametrize("policy", _bounded_policies(), ids=lambda p: p.__name__)
def test_decide_mentions_no_simulator_type(policy):
    for annotation in _annotations_of(policy.decide):
        assert FORBIDDEN_MODULE_MARKER not in annotation, (
            f"{policy.__name__}.decide is annotated with a simulator type: "
            f"{annotation}"
        )
        for forbidden in FORBIDDEN_TYPE_NAMES:
            assert forbidden not in annotation, (
                f"{policy.__name__}.decide mentions {forbidden}"
            )


@pytest.mark.parametrize("policy", _bounded_policies(), ids=lambda p: p.__name__)
def test_the_constructor_cannot_smuggle_latent_state_in(policy):
    """Closing the obvious workaround: take the World in __init__ instead."""
    if policy.__init__ is object.__init__:
        return
    for annotation in _annotations_of(policy.__init__):
        assert FORBIDDEN_MODULE_MARKER not in annotation, (
            f"{policy.__name__}.__init__ accepts a simulator type: {annotation}"
        )
        for forbidden in FORBIDDEN_TYPE_NAMES:
            assert forbidden not in annotation, (
                f"{policy.__name__}.__init__ accepts {forbidden}"
            )


def test_an_exempt_policy_must_declare_itself_an_upper_bound():
    """`reads_latent_state` is not a flag you get to set quietly."""
    for policy in POLICY_REGISTRY.values():
        if not policy.reads_latent_state:
            continue
        docstring = (inspect.getdoc(policy) or "").upper()
        assert "UPPER BOUND" in docstring or "UPPER-BOUND" in docstring, (
            f"{policy.__name__} reads latent state but does not declare "
            "itself an upper-bound instrument in its docstring"
        )


def test_the_boundary_check_would_actually_catch_a_violation():
    """A policy that takes the World must fail, or the tests above are theatre."""

    class PeekingPolicy(Policy):
        name = "_peeking_test_only"

        def decide(
            self, observation: "mandate_recovery.sim.world.World"
        ) -> Decision:  # noqa: F821
            raise AssertionError("never called")

    try:
        annotations = _annotations_of(PeekingPolicy.decide)
        assert any(
            FORBIDDEN_MODULE_MARKER in annotation for annotation in annotations
        ), "the annotation walker missed a simulator type"
    finally:
        POLICY_REGISTRY.pop("_peeking_test_only", None)


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


def test_policy_cannot_be_instantiated_without_decide():
    class Incomplete(Policy):
        name = "_incomplete_test_only"

    try:
        with pytest.raises(TypeError):
            Incomplete()
    finally:
        POLICY_REGISTRY.pop("_incomplete_test_only", None)


def test_every_policy_registers_itself():
    assert POLICY_REGISTRY["do_nothing"] is DoNothingPolicy


def test_decisions_are_unvalidated_by_default():
    """A policy never approves its own action."""
    observation = _observation()
    decision = DoNothingPolicy().decide(observation)
    assert decision.validated is False


def test_reset_exists_and_is_harmless():
    DoNothingPolicy().reset()


# --------------------------------------------------------------------------
# The do-nothing arm
# --------------------------------------------------------------------------


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": 880_000,
        "due_day": 5,
        "current_day": 12,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def test_do_nothing_always_stops():
    policy = DoNothingPolicy()
    for day in (0, 1, 30, 89):
        decision = policy.decide(_observation(current_day=day))
        assert isinstance(decision.action, Stop)


def test_do_nothing_stops_however_bad_things_get():
    """No history, however grim, moves it. It is a floor, not a strategy."""
    from mandate_recovery.types import ObservedAttempt, Rail

    battered = _observation(
        attempt_history=tuple(
            ObservedAttempt(day=d, hour=9, rail=Rail.UPI_AUTOPAY, raw_code="DECLINED")
            for d in range(5)
        ),
        contacts_sent=4,
        days_since_last_contact=1,
        historical_failure_count=9,
    )
    assert isinstance(DoNothingPolicy().decide(battered).action, Stop)


def test_do_nothing_explains_itself():
    decision = DoNothingPolicy().decide(_observation())
    assert decision.rationale.strip()
    assert decision.source == "rule"


def test_do_nothing_is_bound_by_the_boundary():
    assert DoNothingPolicy.reads_latent_state is False
