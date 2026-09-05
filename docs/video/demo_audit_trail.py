"""Render one mandate's full decision trail, for screen recording.

    python docs/video/demo_audit_trail.py                 # the curated mandate
    python docs/video/demo_audit_trail.py m000060         # a different one
    python docs/video/demo_audit_trail.py --list          # candidates, ranked

Deterministic: seed 7 always produces the same trail, so the recording can be
retaken as many times as needed and the output will not move.

This exists because the FastAPI service and React viewer were cut from the
build. The audit trail is the thing worth showing anyway — it is the honesty
evidence, and it reads better in a terminal than it would in a dashboard.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mandate_recovery.agent.audit import AuditLog  # noqa: E402
from mandate_recovery.harness import (  # noqa: E402
    ExperimentConfig,
    build_world_and_mandates,
)
from mandate_recovery.harness.runner import _MandateState, build_observation  # noqa: E402
from mandate_recovery.policies import HeuristicPolicy  # noqa: E402
from mandate_recovery.sim.outcomes import resolve_attempt  # noqa: E402
from mandate_recovery.types import (  # noqa: E402
    Attempt,
    AttemptOutcome,
    ObservedAttempt,
    Rail,
)

SEED = 7
N_MANDATES = 200
N_DAYS = 90
CURATED_MANDATE = "m000147"


def build_log() -> AuditLog:
    """Replay one seed under the heuristic agent, recording every decision."""
    config = ExperimentConfig(
        experiment_id="demo",
        seeds=[SEED],
        n_customers=N_MANDATES,
        n_mandates=N_MANDATES,
        n_days=N_DAYS,
    )
    world, mandates = build_world_and_mandates(SEED, config)
    policy = HeuristicPolicy()
    log = AuditLog()
    states = {mandate.id: _MandateState() for mandate in mandates}
    outcome_rng = np.random.default_rng([SEED, 3])
    scheduled: dict[str, tuple[int, int, Rail]] = {}
    spent: dict[str, int] = {}

    for day in range(N_DAYS):
        for mandate in mandates:
            state = states[mandate.id]
            due = mandate.day_of_month == world.day_of_month and not state.cycle_open
            retry = scheduled.get(mandate.id, (None,))[0] == day
            if not (due or retry):
                continue

            if due:
                state.cycle_open, state.cycle_history = True, []
                hour, rail = 9, Rail.UPI_AUTOPAY
            else:
                _, hour, rail = scheduled.pop(mandate.id)

            response = resolve_attempt(
                world,
                mandate,
                Attempt(
                    mandate_id=mandate.id,
                    scheduled_at=datetime(2026, 1, 1, hour, 0),
                    rail=rail,
                ),
                outcome_rng,
            )
            spent[mandate.id] = spent.get(mandate.id, 0) + 200
            state.cycle_history.append(
                ObservedAttempt(
                    day=day, hour=hour, rail=rail, raw_code=response.raw_code
                )
            )

            if response.outcome is AttemptOutcome.SUCCESS:
                state.successes += 1
                state.max_success_amount_paise = max(
                    state.max_success_amount_paise, mandate.amount_paise
                )
                state.successful_days_of_month.append(world.day_of_month)
                state.cycle_open = False
                continue

            state.failures += 1
            observation = build_observation(mandate, state, day, hour)
            decision = policy.decide(observation)
            log.record(
                seed=SEED,
                arm="heuristic",
                observation=observation,
                proposed_action=decision.action,
                source=decision.source,
                rationale=decision.rationale,
                executed_action=decision.action,
                validator_approved=decision.validated,
                validator_reason="see rationale",
                outcome=response.outcome.value,
                running_cost_paise=spent[mandate.id],
            )

            action = decision.action
            if getattr(action, "kind", "") == "retry_silent" and (
                action.scheduled_day < N_DAYS
            ):
                scheduled[mandate.id] = (
                    max(action.scheduled_day, day + 1),
                    action.scheduled_hour,
                    action.rail,
                )
            else:
                state.cycle_open = False

        if day + 1 < N_DAYS:
            world.advance_day()

    return log


def rank(log: AuditLog) -> list[tuple[int, int, str]]:
    """Mandates whose story shows the most: more decisions, more code variety."""
    ranked = []
    for mandate_id in log.mandate_ids():
        entries = log.entries_for(mandate_id)
        score = (
            len(entries)
            + 3 * sum(1 for e in entries if not e.validator_approved)
            + 2 * len({e.last_raw_code for e in entries})
        )
        ranked.append((score, len(entries), mandate_id))
    return sorted(ranked, reverse=True)


def main(argv: list[str]) -> int:
    log = build_log()

    if "--list" in argv:
        print(f"{len(log)} decisions across {len(log.mandate_ids())} mandates\n")
        print(f"{'score':>6}{'decisions':>11}  mandate")
        for score, count, mandate_id in rank(log)[:10]:
            print(f"{score:>6}{count:>11}  {mandate_id}")
        return 0

    chosen = next((a for a in argv[1:] if not a.startswith("-")), CURATED_MANDATE)
    print(log.to_human_readable(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
