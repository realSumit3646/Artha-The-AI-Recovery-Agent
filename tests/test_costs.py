"""Tests for the intervention cost model.

The two that matter are the bounds: retrying forever must lose money, and
doing nothing must cost nothing. Between those, a policy has to actually
choose.
"""

from __future__ import annotations

import pydantic
import pytest

from mandate_recovery.calibration import DEFAULT_CALIBRATION
from mandate_recovery.costs import CostBreakdown, CostModel, Episode

MANDATE_PAISE = 880_000  # the validated median mandate


@pytest.fixture
def model() -> CostModel:
    return CostModel(DEFAULT_CALIBRATION)


def _episode(**overrides) -> Episode:
    kwargs = {
        "mandate_id": "m1",
        "amount_paise": MANDATE_PAISE,
        "remaining_cycles": 11,
    }
    kwargs.update(overrides)
    return Episode(**kwargs)


# --------------------------------------------------------------------------
# The bounds
# --------------------------------------------------------------------------


def test_retrying_fifty_times_loses_money(model):
    """The whole reason this module exists."""
    episode = _episode(attempts=50, recovered_paise=0)
    scored = model.score(episode)

    assert scored.net_recovery_paise < 0
    assert scored.gateway_cost_paise == 50 * (
        DEFAULT_CALIBRATION.gateway_cost_per_attempt_paise.value
    )


def test_retrying_and_nagging_can_cost_more_than_it_recovers(model):
    """Even a successful recovery can be a net loss once churn is charged."""
    episode = _episode(
        attempts=50,
        sms_sent=20,
        voice_calls_made=10,
        recovered_paise=MANDATE_PAISE,
    )
    scored = model.score(episode)

    assert scored.recovered_paise > 0
    assert scored.net_recovery_paise < 0, (
        "30 contacts on an 11-cycle mandate should cost more than one "
        "recovery is worth"
    )


def test_doing_nothing_costs_nothing_and_recovers_nothing(model):
    scored = model.score(_episode())

    assert scored.gateway_cost_paise == 0
    assert scored.contact_cost_paise == 0
    assert scored.churn_cost_paise == 0
    assert scored.total_cost_paise == 0
    assert scored.recovered_paise == 0
    assert scored.net_recovery_paise == 0
    assert scored.churn_probability == 0.0
    assert scored.over_intervention is False


def test_a_single_cheap_recovery_is_profitable(model):
    """The model must not make every intervention a loss."""
    scored = model.score(
        _episode(attempts=2, sms_sent=1, recovered_paise=MANDATE_PAISE)
    )
    assert scored.net_recovery_paise > 0


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def test_gateway_cost_is_charged_per_attempt_regardless_of_outcome(model):
    per_attempt = DEFAULT_CALIBRATION.gateway_cost_per_attempt_paise.value
    failed = model.score(_episode(attempts=3, recovered_paise=0))
    succeeded = model.score(_episode(attempts=3, recovered_paise=MANDATE_PAISE))

    assert failed.gateway_cost_paise == 3 * per_attempt
    assert succeeded.gateway_cost_paise == failed.gateway_cost_paise


def test_contact_cost_uses_the_calibrated_channel_prices(model):
    sms = DEFAULT_CALIBRATION.sms_cost_paise.value
    voice = DEFAULT_CALIBRATION.voice_call_cost_paise.value
    scored = model.score(_episode(sms_sent=4, voice_calls_made=2))
    assert scored.contact_cost_paise == 4 * sms + 2 * voice


def test_a_voice_call_costs_more_than_an_sms(model):
    sms_only = model.score(_episode(sms_sent=1))
    voice_only = model.score(_episode(voice_calls_made=1))
    assert voice_only.contact_cost_paise > sms_only.contact_cost_paise


def test_churn_probability_rises_with_every_contact(model):
    increment = (
        DEFAULT_CALIBRATION.churn_probability_increment_per_contact.value
    )
    for contacts in range(5):
        scored = model.score(_episode(sms_sent=contacts))
        assert scored.churn_probability == pytest.approx(contacts * increment)


def test_churn_probability_is_capped_at_one(model):
    scored = model.score(_episode(sms_sent=10_000))
    assert scored.churn_probability == 1.0


def test_escalation_counts_as_a_contact_for_churn(model):
    """A human agent calling is still the customer being contacted."""
    quiet = model.score(_episode())
    escalated = model.score(_episode(escalated_to_human=True))

    assert escalated.churn_probability > quiet.churn_probability
    # ...but carries no separate monetary charge, since none is calibrated.
    assert escalated.contact_cost_paise == quiet.contact_cost_paise


def test_churn_cost_scales_with_what_is_left_to_lose(model):
    """Losing a customer on their last cycle is cheap; losing a new one is not."""
    nearly_done = model.score(_episode(sms_sent=3, remaining_cycles=1))
    plenty_left = model.score(_episode(sms_sent=3, remaining_cycles=24))

    assert plenty_left.churn_cost_paise > nearly_done.churn_cost_paise


def test_churn_cost_is_zero_when_nothing_remains(model):
    scored = model.score(_episode(sms_sent=5, remaining_cycles=0))
    assert scored.churn_cost_paise == 0
    assert scored.churn_probability > 0.0


def test_churn_ignores_latent_intent_and_charges_only_what_the_policy_caused(
    model,
):
    """Two identical episodes cost the same, whoever the customer is.

    Charging a policy for churn it did not cause would make cost depend on
    latent state and punish arms that drew unhappy customers.
    """
    left = model.score(_episode(mandate_id="a", sms_sent=2))
    right = model.score(_episode(mandate_id="b", sms_sent=2))
    assert left.churn_cost_paise == right.churn_cost_paise


# --------------------------------------------------------------------------
# Over-intervention
# --------------------------------------------------------------------------


def test_over_intervention_needs_both_a_contact_and_a_counterfactual(model):
    contacted_and_would_have_paid = model.score(
        _episode(sms_sent=1, would_have_paid_without_intervention=True)
    )
    contacted_but_would_not_have = model.score(
        _episode(sms_sent=1, would_have_paid_without_intervention=False)
    )
    silent_but_would_have_paid = model.score(
        _episode(attempts=3, would_have_paid_without_intervention=True)
    )

    assert contacted_and_would_have_paid.over_intervention is True
    assert contacted_but_would_not_have.over_intervention is False
    assert silent_but_would_have_paid.over_intervention is False


def test_a_silent_retry_is_never_over_intervention(model):
    """Retrying costs money, but it does not bother the customer."""
    scored = model.score(
        _episode(attempts=10, would_have_paid_without_intervention=True)
    )
    assert scored.over_intervention is False
    assert scored.gateway_cost_paise > 0


def test_escalation_alone_can_be_over_intervention(model):
    scored = model.score(
        _episode(
            escalated_to_human=True, would_have_paid_without_intervention=True
        )
    )
    assert scored.over_intervention is True


# --------------------------------------------------------------------------
# Money discipline
# --------------------------------------------------------------------------


def test_every_cost_is_integer_paise(model):
    scored = model.score(
        _episode(attempts=7, sms_sent=3, voice_calls_made=2, recovered_paise=1234)
    )
    for field in (
        "gateway_cost_paise",
        "contact_cost_paise",
        "churn_cost_paise",
        "total_cost_paise",
        "recovered_paise",
        "net_recovery_paise",
    ):
        value = getattr(scored, field)
        assert isinstance(value, int) and not isinstance(value, bool), field


def test_total_is_the_sum_of_its_parts(model):
    scored = model.score(_episode(attempts=4, sms_sent=2, voice_calls_made=1))
    assert scored.total_cost_paise == (
        scored.gateway_cost_paise
        + scored.contact_cost_paise
        + scored.churn_cost_paise
    )
    assert scored.net_recovery_paise == (
        scored.recovered_paise - scored.total_cost_paise
    )


def test_episode_rejects_float_money():
    with pytest.raises(pydantic.ValidationError):
        Episode(mandate_id="m", amount_paise=880_000.0)


def test_breakdown_is_frozen(model):
    scored = model.score(_episode())
    with pytest.raises(pydantic.ValidationError):
        scored.net_recovery_paise = 1


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_net_recovery_sums_across_episodes(model):
    episodes = [
        _episode(mandate_id="a", attempts=1, recovered_paise=MANDATE_PAISE),
        _episode(mandate_id="b", attempts=50, recovered_paise=0),
    ]
    total = model.net_recovery_paise(episodes)
    assert total == sum(model.score(e).net_recovery_paise for e in episodes)
    assert len(model.score_all(episodes)) == 2
    assert all(isinstance(s, CostBreakdown) for s in model.score_all(episodes))
