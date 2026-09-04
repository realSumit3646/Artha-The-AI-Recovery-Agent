"""Tests for the domain model, and for the two invariants it encodes.

The observation boundary and the integer-paise rule are not conventions here;
they are the reason this module exists. These tests are the enforcement.
"""

from __future__ import annotations

import inspect
from datetime import datetime

import pydantic
import pytest
from pydantic import BaseModel

from mandate_recovery import types
from mandate_recovery.types import (
    Attempt,
    CollectPartial,
    Decision,
    LatentCustomerState,
    Mandate,
    MandateStatus,
    NudgeChannel,
    ObservedAttempt,
    Observation,
    Rail,
    SendNudge,
)


def _domain_models() -> list[type[BaseModel]]:
    """Every public pydantic model defined in ``mandate_recovery.types``."""
    return [
        obj
        for name, obj in inspect.getmembers(types, inspect.isclass)
        if not name.startswith("_")
        and issubclass(obj, BaseModel)
        and obj.__module__ == types.__name__
    ]


def _money_fields() -> list[tuple[type[BaseModel], str]]:
    """Every field on every domain model that holds an amount of money.

    Detected by name: an amount is only ever expressed in paise, so a money
    field always says so. ``spend_rate_paise_per_day`` is why this is a
    substring test and not a suffix test.
    """
    return [
        (model, field_name)
        for model in _domain_models()
        for field_name in model.model_fields
        if "paise" in field_name
    ]


# --------------------------------------------------------------------------
# The observation boundary
# --------------------------------------------------------------------------


def test_observation_contains_no_latent_fields():
    """Observation and LatentCustomerState share no field name.

    This is the central invariant. The simulator knows the balance, the salary
    date, the spend rate and the churn intent; a policy knows none of them. If
    this test fails, either a latent fact has leaked onto the observation or
    someone has renamed around the check -- both are the same bug.
    """
    observation_fields = set(Observation.model_fields)
    latent_fields = set(LatentCustomerState.model_fields)

    # Guard against a vacuous pass if either model is emptied or renamed.
    assert observation_fields, "Observation has no fields"
    assert latent_fields, "LatentCustomerState has no fields"

    leaked = observation_fields & latent_fields
    assert not leaked, f"latent state leaked onto Observation: {sorted(leaked)}"


def test_observation_has_no_nested_latent_fields():
    """The boundary holds transitively, not just on the top-level fields.

    Nesting a latent field one model deeper would satisfy the disjointness
    check above while leaking exactly the same information.
    """
    latent_fields = set(LatentCustomerState.model_fields)

    seen: set[type[BaseModel]] = set()
    stack: list[type[BaseModel]] = [Observation]
    checked_any_nested = False

    while stack:
        model = stack.pop()
        if model in seen:
            continue
        seen.add(model)

        leaked = set(model.model_fields) & latent_fields
        assert not leaked, (
            f"latent state leaked onto {model.__name__} "
            f"(reachable from Observation): {sorted(leaked)}"
        )

        for field in model.model_fields.values():
            for arg in (field.annotation, *getattr(field.annotation, "__args__", ())):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    checked_any_nested = True
                    stack.append(arg)

    assert checked_any_nested, "no nested models traversed; check the walker"


def test_latent_state_is_not_reachable_from_observation():
    """LatentCustomerState is not embedded in Observation under any name."""
    assert LatentCustomerState not in {
        arg
        for field in Observation.model_fields.values()
        for arg in (field.annotation, *getattr(field.annotation, "__args__", ()))
        if isinstance(arg, type)
    }


def test_observation_rejects_unknown_fields():
    """A latent fact cannot be smuggled in as an extra keyword."""
    with pytest.raises(pydantic.ValidationError):
        Observation(
            mandate_id="m1",
            amount_paise=49900,
            due_day=5,
            current_day=12,
            churn_intent=0.9,
        )


def test_observed_attempt_exposes_raw_code_not_outcome():
    """Policies read bank codes; the classified outcome stays simulator-side."""
    assert "raw_code" in ObservedAttempt.model_fields
    assert "outcome" not in ObservedAttempt.model_fields


# --------------------------------------------------------------------------
# Money is integer paise
# --------------------------------------------------------------------------


def test_all_money_fields_are_typed_int():
    """Every monetary field on every domain model is an ``int``, never a float."""
    money_fields = _money_fields()

    # Guard against a vacuous pass if the naming convention drifts.
    expected = {
        (Mandate, "amount_paise"),
        (LatentCustomerState, "balance_paise"),
        (LatentCustomerState, "salary_amount_paise"),
        (LatentCustomerState, "spend_rate_paise_per_day"),
        (LatentCustomerState, "per_txn_limit_paise"),
        (Observation, "amount_paise"),
        (Observation, "max_historical_success_amount_paise"),
        (CollectPartial, "amount_paise"),
    }
    assert expected <= set(money_fields)

    for model, field_name in money_fields:
        annotation = model.model_fields[field_name].annotation
        assert annotation is int, (
            f"{model.__name__}.{field_name} is typed {annotation!r}, "
            "but money is integer paise"
        )


@pytest.mark.parametrize("amount", [49900.0, 4.5, "49900"])
def test_money_fields_reject_non_int(amount):
    """A float amount is rejected outright, not coerced.

    ``49900.0`` matters most: without strict ints pydantic would silently
    accept it, and float arithmetic would be in the money path from then on.
    """
    with pytest.raises(pydantic.ValidationError):
        CollectPartial(amount_paise=amount)


def test_churn_intent_is_a_float_and_bounded():
    """churn_intent is a probability, not money, so it is legitimately float."""
    assert LatentCustomerState.model_fields["churn_intent"].annotation is float
    with pytest.raises(pydantic.ValidationError):
        _latent(churn_intent=1.5)


# --------------------------------------------------------------------------
# Model mechanics
# --------------------------------------------------------------------------


def _latent(**overrides) -> LatentCustomerState:
    kwargs = {
        "balance_paise": 120000,
        "salary_day": 1,
        "salary_amount_paise": 5000000,
        "spend_rate_paise_per_day": 90000,
        "churn_intent": 0.1,
        "per_txn_limit_paise": 10000000,
    }
    kwargs.update(overrides)
    return LatentCustomerState(**kwargs)


def _observation(**overrides) -> Observation:
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": 49900,
        "due_day": 5,
        "current_day": 12,
    }
    kwargs.update(overrides)
    return Observation(**kwargs)


def test_all_domain_models_are_frozen():
    """Every domain model rejects mutation, so a stored trace stays true."""
    models = _domain_models()
    assert models, "no domain models discovered"
    for model in models:
        assert model.model_config.get("frozen") is True, f"{model.__name__} is mutable"


def test_frozen_is_enforced_at_runtime():
    observation = _observation()
    with pytest.raises(pydantic.ValidationError):
        observation.current_day = 99


def test_observation_is_hashable():
    """attempt_history is a tuple, so a frozen Observation really is hashable."""
    assert isinstance(hash(_observation()), int)


def test_mandate_rejects_out_of_range_day_of_month():
    with pytest.raises(pydantic.ValidationError):
        Mandate(
            id="m1",
            customer_id="c1",
            amount_paise=49900,
            day_of_month=32,
            created_on_day=0,
            status=MandateStatus.ACTIVE,
        )


def test_action_union_is_discriminated():
    """The action union resolves by ``kind`` and survives a JSON round trip."""
    decision = Decision(
        observation=_observation(),
        action={"kind": "send_nudge", "channel": "SMS", "tone_level": 2},
        source="rule",
        rationale="salary lands tomorrow; ask rather than retry",
    )
    assert isinstance(decision.action, SendNudge)
    assert decision.action.channel is NudgeChannel.SMS

    round_tripped = Decision.model_validate_json(decision.model_dump_json())
    assert round_tripped == decision
    assert isinstance(round_tripped.action, SendNudge)


def test_decision_is_unvalidated_by_default():
    """No decision counts as approved until a deterministic validator says so."""
    decision = Decision(
        observation=_observation(),
        action={"kind": "stop", "reason": "mandate revoked"},
        source="llm",
        rationale="customer asked us to stop",
    )
    assert decision.validated is False


def test_attempt_response_is_optional_until_resolved():
    attempt = Attempt(
        mandate_id="m1",
        scheduled_at=datetime(2026, 4, 5, 9, 0),
        rail=Rail.UPI_AUTOPAY,
    )
    assert attempt.response is None
