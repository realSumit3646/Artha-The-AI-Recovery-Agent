"""The experiment runner. Everything downstream depends on this being right.

The paired design
-----------------
**For a given seed, every arm faces a bit-identical world.** The world is
built from the seed before any policy exists, and each arm gets its own deep
copy. Arm A's retries cannot move the balance Arm B sees; Arm A's bank uptime
draws are Arm B's draws.

This is not a nicety. Recovery outcomes vary enormously between seeds — one
seed hands you customers who were going to pay anyway, another hands you a
bank outage. Comparing arms across *different* worlds measures the luck of the
draw and calls it a policy difference. With paired worlds the per-seed delta
is a clean within-subject comparison, which is what makes the bootstrap at
commit 14 mean anything.

If the pairing breaks, every number in this project becomes noise, so
``tests/harness/test_runner.py`` asserts two arms on the same seed observe
identical latent balance trajectories.

Cycles
------
A mandate recurs. It comes due on its day of the month, every month, for the
whole horizon. A cycle opens on the due day and closes when the debit
succeeds, when the policy stops, or when the month runs out. Ninety days is
roughly three cycles, which is what gives a customer a payment *history* for a
policy to reason about.

``Observation.attempt_history`` carries the **current cycle's** attempts —
that is the recovery episode a policy is working on. History across earlier
cycles arrives as ``historical_success_count``,
``historical_failure_count`` and ``max_historical_success_amount_paise``.

The counterfactual
------------------
Each seed also runs a silent no-intervention pass, used only to fill in
``would_have_paid_without_intervention``. That is what makes over-intervention
a real measurement: a policy that contacts someone who would have paid anyway
is charged for it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..calibration import DEFAULT_CALIBRATION, CalibrationSet
from ..costs import CostModel, Episode
from ..policies.base import Policy
from ..sim.outcomes import resolve_attempt, revoke_eligible_mandates
from ..sim.world import DAYS_IN_MONTH, World
from ..types import (
    Attempt,
    AttemptOutcome,
    CollectPartial,
    EscalateHuman,
    Mandate,
    MandateStatus,
    Observation,
    ObservedAttempt,
    Rail,
    RetrySilent,
    SendNudge,
    Stop,
    SwitchRail,
)

__all__ = [
    "ExperimentConfig",
    "EpisodeRecord",
    "DecisionRecord",
    "ArmResult",
    "PolicyFactory",
    "run_experiment",
    "build_world_and_mandates",
    "build_observation",
]

#: arm name -> factory taking (world, mandate_id->customer_id). Only the
#: oracle uses either argument; boundary-bound policies ignore them.
PolicyFactory = Callable[[World, Mapping[str, str]], Policy]

#: Assumed mandate lifetime, in cycles, for costing churn. A modelling choice,
#: not a calibrated figure: it sets the scale of what losing a customer costs.
ASSUMED_MANDATE_LIFETIME_CYCLES = 12


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce a run."""

    experiment_id: str
    seeds: Sequence[int]
    n_customers: int = 500
    n_mandates: int = 500
    n_days: int = 90
    calibration: CalibrationSet = DEFAULT_CALIBRATION

    #: Mandate amount distribution, fitted at commit 7. Held here rather than
    #: in a script so it lands in the stored config, per invariant 4.
    mandate_amount_paise_median: int = 880_000
    mandate_amount_lognormal_sigma: float = 1.20

    #: The hour a mandate is first presented on its due day.
    default_presentment_hour: int = 9

    def to_dict(self) -> dict[str, Any]:
        import json

        return {
            "experiment_id": self.experiment_id,
            "seeds": list(self.seeds),
            "n_customers": self.n_customers,
            "n_mandates": self.n_mandates,
            "n_days": self.n_days,
            "mandate_amount_paise_median": self.mandate_amount_paise_median,
            "mandate_amount_lognormal_sigma": self.mandate_amount_lognormal_sigma,
            "default_presentment_hour": self.default_presentment_hour,
            "assumed_mandate_lifetime_cycles": ASSUMED_MANDATE_LIFETIME_CYCLES,
            "calibration": json.loads(self.calibration.model_dump_json()),
        }


@dataclass
class DecisionRecord:
    """One decision, flattened for the audit trail."""

    seed: int
    arm: str
    mandate_id: str
    day: int
    action_kind: str
    scheduled_day: int | None
    scheduled_hour: int | None
    source: str
    rationale: str
    validated: bool


@dataclass
class EpisodeRecord:
    """One mandate's whole life under one arm, across every cycle."""

    seed: int
    arm: str
    mandate_id: str
    customer_id: str
    amount_paise: int
    cycles: int = 0
    attempts: int = 0
    decisions: int = 0
    successes: int = 0
    failures: int = 0
    sms_sent: int = 0
    voice_calls_made: int = 0
    escalated_to_human: bool = False
    recovered_paise: int = 0
    days_to_recovery: int | None = None
    gateway_cost_paise: int = 0
    contact_cost_paise: int = 0
    churn_cost_paise: int = 0
    total_cost_paise: int = 0
    net_recovery_paise: int = 0
    over_intervention: bool = False
    would_have_paid_without_intervention: bool = False
    attempt_outcomes: str = ""
    raw_codes: str = ""

    def to_row(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ArmResult:
    arm: str
    episodes: list[EpisodeRecord]
    decisions: list[DecisionRecord]


@dataclass
class _MandateState:
    """Live per-mandate bookkeeping for one arm."""

    cycle_open: bool = False
    cycle_history: list[ObservedAttempt] = field(default_factory=list)
    cycle_first_failure_day: int | None = None
    scheduled: tuple[int, int, Rail] | None = None
    successes: int = 0
    failures: int = 0
    contacts: int = 0
    last_contact_day: int | None = None
    max_success_amount_paise: int = 0
    all_outcomes: list[str] = field(default_factory=list)
    all_codes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Observation: the only thing a policy sees
# --------------------------------------------------------------------------


def build_observation(
    mandate: Mandate, state: _MandateState, day: int
) -> Observation:
    """Assemble the policy's view. Nothing latent may enter here.

    Every field is something a real collector could look up: the mandate's own
    terms, the calendar, what has been tried this cycle, what was sent to the
    customer, and this customer's settled history with us.
    """
    return Observation(
        mandate_id=mandate.id,
        amount_paise=mandate.amount_paise,
        due_day=mandate.day_of_month,
        current_day=day,
        attempt_history=tuple(state.cycle_history),
        contacts_sent=state.contacts,
        days_since_last_contact=(
            None if state.last_contact_day is None else day - state.last_contact_day
        ),
        historical_success_count=state.successes,
        historical_failure_count=state.failures,
        max_historical_success_amount_paise=state.max_success_amount_paise,
    )


# --------------------------------------------------------------------------
# World construction
# --------------------------------------------------------------------------


def build_world_and_mandates(
    seed: int, config: ExperimentConfig
) -> tuple[World, list[Mandate]]:
    """The world one seed produces, before any policy has seen anything.

    Built once per seed and deep-copied per arm; that copy is the pairing.
    """
    world = World(
        config.calibration,
        np.random.default_rng([seed, 1]),
        n_customers=config.n_customers,
        n_days=config.n_days,
    )
    setup_rng = np.random.default_rng([seed, 2])

    amounts = np.rint(
        setup_rng.lognormal(
            mean=float(np.log(config.mandate_amount_paise_median)),
            sigma=config.mandate_amount_lognormal_sigma,
            size=config.n_mandates,
        )
    ).astype(np.int64)
    np.maximum(amounts, 100, out=amounts)
    due_days = setup_rng.integers(1, DAYS_IN_MONTH + 1, size=config.n_mandates)

    mandates = [
        Mandate(
            id=f"m{index:06d}",
            customer_id=world.customer_id_for(index % config.n_customers),
            amount_paise=int(amounts[index]),
            day_of_month=int(due_days[index]),
            created_on_day=0,
            status=MandateStatus.ACTIVE,
        )
        for index in range(config.n_mandates)
    ]
    return world, mandates


# --------------------------------------------------------------------------
# One arm
# --------------------------------------------------------------------------


def _stamp(hour: int) -> datetime:
    return datetime(2026, 1, 1, hour, 0)


def _run_arm(
    arm: str,
    policy: Policy,
    world: World,
    mandates: Sequence[Mandate],
    seed: int,
    config: ExperimentConfig,
    *,
    record_decisions: bool = True,
) -> ArmResult:
    """Run one policy against one world for the whole horizon."""
    outcome_rng = np.random.default_rng([seed, 3])
    revocation_rng = np.random.default_rng([seed, 4])

    mandates = list(mandates)
    episodes = {
        mandate.id: EpisodeRecord(
            seed=seed,
            arm=arm,
            mandate_id=mandate.id,
            customer_id=mandate.customer_id,
            amount_paise=mandate.amount_paise,
        )
        for mandate in mandates
    }
    states = {mandate.id: _MandateState() for mandate in mandates}
    decisions: list[DecisionRecord] = []
    funds_failures: dict[str, int] = {}

    for day in range(config.n_days):
        for mandate in mandates:
            state = states[mandate.id]
            record = episodes[mandate.id]

            if mandate.status is not MandateStatus.ACTIVE:
                continue

            due_today = (
                mandate.day_of_month == world.day_of_month and not state.cycle_open
            )
            retry_today = (
                state.scheduled is not None and state.scheduled[0] == day
            )
            if not (due_today or retry_today):
                continue

            if due_today:
                state.cycle_open = True
                state.cycle_history = []
                state.cycle_first_failure_day = None
                state.scheduled = None
                record.cycles += 1
                hour = config.default_presentment_hour
                rail = Rail.UPI_AUTOPAY
            else:
                _, hour, rail = state.scheduled
                state.scheduled = None

            attempt = Attempt(
                mandate_id=mandate.id,
                scheduled_at=_stamp(hour),
                rail=rail,
            )
            response = resolve_attempt(world, mandate, attempt, outcome_rng)

            record.attempts += 1
            state.all_outcomes.append(response.outcome.value)
            state.all_codes.append(response.raw_code)
            state.cycle_history.append(
                ObservedAttempt(
                    day=day, hour=hour, rail=rail, raw_code=response.raw_code
                )
            )

            if response.outcome is AttemptOutcome.SUCCESS:
                record.recovered_paise += mandate.amount_paise
                record.successes += 1
                state.successes += 1
                state.max_success_amount_paise = max(
                    state.max_success_amount_paise, mandate.amount_paise
                )
                if (
                    record.days_to_recovery is None
                    and state.cycle_first_failure_day is not None
                ):
                    record.days_to_recovery = day - state.cycle_first_failure_day
                state.cycle_open = False
                continue

            record.failures += 1
            state.failures += 1
            if state.cycle_first_failure_day is None:
                state.cycle_first_failure_day = day
            if response.outcome is AttemptOutcome.INSUFFICIENT_FUNDS:
                funds_failures[mandate.id] = funds_failures.get(mandate.id, 0) + 1

            # A failure is the only place a policy gets to act.
            decision = policy.decide(build_observation(mandate, state, day))
            record.decisions += 1
            action = decision.action

            if record_decisions:
                decisions.append(
                    DecisionRecord(
                        seed=seed,
                        arm=arm,
                        mandate_id=mandate.id,
                        day=day,
                        action_kind=action.kind,
                        scheduled_day=getattr(action, "scheduled_day", None),
                        scheduled_hour=getattr(action, "scheduled_hour", None),
                        source=decision.source,
                        rationale=decision.rationale,
                        validated=decision.validated,
                    )
                )

            _apply(action, state, record, day, config)

        mandates = list(
            revoke_eligible_mandates(
                mandates, funds_failures, config.calibration, revocation_rng
            )
        )
        if day + 1 < config.n_days:
            world.advance_day()

    for mandate_id, state in states.items():
        episodes[mandate_id].attempt_outcomes = ",".join(state.all_outcomes)
        episodes[mandate_id].raw_codes = ",".join(state.all_codes)

    return ArmResult(arm=arm, episodes=list(episodes.values()), decisions=decisions)


def _apply(
    action,
    state: _MandateState,
    record: EpisodeRecord,
    day: int,
    config: ExperimentConfig,
) -> None:
    """Carry out a decision's effect on this cycle."""
    if isinstance(action, (RetrySilent, CollectPartial, SwitchRail)):
        if isinstance(action, RetrySilent):
            target_day = max(action.scheduled_day, day + 1)
            hour = action.scheduled_hour
            rail = action.rail
        else:
            target_day = day + 1
            hour = config.default_presentment_hour
            rail = getattr(action, "target_rail", Rail.UPI_AUTOPAY)

        if target_day < config.n_days:
            state.scheduled = (target_day, hour, rail)
        else:
            state.cycle_open = False
        return

    if isinstance(action, SendNudge):
        record.sms_sent += 1
        state.contacts += 1
        state.last_contact_day = day
        # A nudge does not itself re-present the debit. The cycle closes
        # here; nudge-then-retry needs harness support, see PROGRESS.md.
        state.cycle_open = False
        return

    if isinstance(action, EscalateHuman):
        record.escalated_to_human = True
        state.contacts += 1
        state.last_contact_day = day
        state.cycle_open = False
        return

    if isinstance(action, Stop):
        state.cycle_open = False


# --------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------


def run_experiment(
    policy_factories: Mapping[str, PolicyFactory],
    config: ExperimentConfig,
) -> tuple[list[EpisodeRecord], list[DecisionRecord]]:
    """Run every arm across every seed on paired worlds."""
    from ..policies.do_nothing import DoNothingPolicy

    cost_model = CostModel(config.calibration)
    all_episodes: list[EpisodeRecord] = []
    all_decisions: list[DecisionRecord] = []

    for seed in config.seeds:
        base_world, base_mandates = build_world_and_mandates(seed, config)
        customer_by_mandate = {m.id: m.customer_id for m in base_mandates}

        counterfactual = _run_arm(
            "_counterfactual",
            DoNothingPolicy(),
            copy.deepcopy(base_world),
            base_mandates,
            seed,
            config,
            record_decisions=False,
        )
        would_have_paid = {
            episode.mandate_id: episode.recovered_paise > 0
            for episode in counterfactual.episodes
        }

        for arm, factory in policy_factories.items():
            arm_world = copy.deepcopy(base_world)
            policy = factory(arm_world, customer_by_mandate)
            policy.reset()

            result = _run_arm(arm, policy, arm_world, base_mandates, seed, config)

            for episode in result.episodes:
                episode.would_have_paid_without_intervention = would_have_paid.get(
                    episode.mandate_id, False
                )
                scored = cost_model.score(
                    Episode(
                        mandate_id=episode.mandate_id,
                        amount_paise=episode.amount_paise,
                        attempts=episode.attempts,
                        sms_sent=episode.sms_sent,
                        voice_calls_made=episode.voice_calls_made,
                        escalated_to_human=episode.escalated_to_human,
                        recovered_paise=episode.recovered_paise,
                        remaining_cycles=max(
                            0,
                            ASSUMED_MANDATE_LIFETIME_CYCLES - episode.cycles,
                        ),
                        would_have_paid_without_intervention=(
                            episode.would_have_paid_without_intervention
                        ),
                    )
                )
                episode.gateway_cost_paise = scored.gateway_cost_paise
                episode.contact_cost_paise = scored.contact_cost_paise
                episode.churn_cost_paise = scored.churn_cost_paise
                episode.total_cost_paise = scored.total_cost_paise
                episode.net_recovery_paise = scored.net_recovery_paise
                episode.over_intervention = scored.over_intervention

            all_episodes.extend(result.episodes)
            all_decisions.extend(result.decisions)

    return all_episodes, all_decisions
