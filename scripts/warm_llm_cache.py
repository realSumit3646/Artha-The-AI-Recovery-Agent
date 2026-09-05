"""Populate the response cache so the ablation can run offline.

    python scripts/warm_llm_cache.py            # warm everything missing
    python scripts/warm_llm_cache.py --dry-run  # count what is missing, call nothing

Why this is a separate step from the ablation: if the quota runs out
*during* an experiment, the LLM agent silently falls back to the heuristic for
the remainder and the arm becomes a blend of two policies. That is worse than
a failed run, because it looks like a result. So the cache is filled first,
and `run_ablation.py --offline` then executes with every response already on
disk and no possibility of a partial fallback.

How the prompt set is found
---------------------------
The full 120-seed experiment is replayed against a collector that records
prompts and returns canned replies without touching the network. Because
prompts are canonical and bucketed, ~186,000 calls collapse onto ~216 distinct
prompts. Only those are sent.

Pacing
------
Groq's free tier allows 8,000 tokens per minute, and a call here measures
~1,400 tokens, so requests are spaced to stay under it. Without pacing the run
spends its time in 429 backoff instead of doing work.

A per-minute throttle and a per-day exhaustion both arrive as HTTP 429. They
are told apart from the message text: a throttle waits and continues, an
exhaustion stops. The provider's raw message is printed either way, because
the first run guessed and mislabelled a throttle as a daily limit.

Prompts are warmed in descending order of how many calls each one serves. The
nine message prompts answer 118,888 calls between them; the 180 intervention
prompts answer 150,948. Warming alphabetically spends the allowance on the
cheap prompts and leaves the valuable ones cold -- which is what happened on
the first run, leaving 39% of calls uncached despite 90% of prompts being
warmed.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mandate_recovery.harness import ExperimentConfig, run_experiment  # noqa: E402
from mandate_recovery.llm.cache import cache_key  # noqa: E402
from mandate_recovery.llm.client import LLMClient, LLMFallback  # noqa: E402
from mandate_recovery.llm.diagnosis import DiagnosisReply  # noqa: E402
from mandate_recovery.llm.intervention import InterventionReply  # noqa: E402
from mandate_recovery.llm.messaging import MessageReply  # noqa: E402
from mandate_recovery.policies.llm_agent import LLMAgentPolicy  # noqa: E402

N_SEEDS = 120
N_MANDATES = 500
N_DAYS = 90

#: Free-tier tokens per minute. Pace under it rather than backing off into it.
TOKENS_PER_MINUTE = 8_000
#: Measured over the first warming run: 268,240 tokens across 195 calls.
#: The initial 900 estimate was low, which paced requests at ~10,800
#: tokens/minute -- over the limit, so the run spent time in 429s.
ESTIMATED_TOKENS_PER_CALL = 1_400

SCHEMAS = {
    "DiagnosisReply": DiagnosisReply,
    "InterventionReply": InterventionReply,
    "MessageReply": MessageReply,
}

#: Canned replies for the collection pass. Chosen to drive the pipeline down
#: every branch -- a nudge so messaging prompts are reached at all.
CANNED = {
    "DiagnosisReply": DiagnosisReply(
        cause="INSUFFICIENT_FUNDS", confidence=0.8, reasoning="collection pass"
    ),
    "InterventionReply": InterventionReply(
        action="SEND_NUDGE",
        timing="AFTER_NEXT_SALARY",
        tone_level=1,
        reasoning="collection pass",
    ),
    "MessageReply": MessageReply(
        message_template="Hi {amount} {due_date} {reference} {merchant}"
    ),
}


class _Collector:
    """Records prompts, calls nothing."""

    model = "collect"
    offline = True

    def __init__(self) -> None:
        self.prompts: dict[str, dict[str, None]] = {}
        self.calls: Counter = Counter()

    def complete(self, prompt, schema, system_instruction=None):
        bucket = self.prompts.setdefault(schema.__name__, {})
        bucket[prompt] = bucket.get(prompt, 0) + 1
        self.calls[schema.__name__] += 1
        return CANNED[schema.__name__]

    def stats(self):
        return {}


def collect_prompts() -> tuple[dict[str, dict[str, int]], Counter]:
    """Every distinct prompt the full ablation will ask for."""
    collector = _Collector()
    config = ExperimentConfig(
        experiment_id="warm",
        seeds=list(range(N_SEEDS)),
        n_customers=N_MANDATES,
        n_mandates=N_MANDATES,
        n_days=N_DAYS,
    )
    run_experiment({"llm_agent": lambda world, mapping: LLMAgentPolicy(collector)}, config)
    return collector.prompts, collector.calls


def main(dry_run: bool = False) -> int:
    print("Enumerating the prompt set (no API calls)...", flush=True)
    started = time.time()
    prompts, calls = collect_prompts()
    total = sum(len(p) for p in prompts.values())
    print(f"  done in {time.time() - started:.0f}s\n")

    for name in sorted(prompts):
        per_prompt = calls[name] / max(1, len(prompts[name]))
        print(
            f"  {name:<20} {calls[name]:>8,} calls collapse onto "
            f"{len(prompts[name]):>4} prompts  ({per_prompt:>8,.0f} calls each)"
        )
    print(f"  {'TOTAL':<20} {sum(calls.values()):>8,} calls -> {total} prompts\n")

    client = LLMClient(cache_dir=REPO_ROOT / "llm_cache")
    # Warm in descending order of how many calls each prompt serves, NOT in
    # schema order. The nine MessageReply prompts answer 118,888 calls while
    # the 180 InterventionReply prompts answer 150,948 -- so a message prompt
    # is worth ~13,000 calls and an intervention prompt ~840. Warming
    # alphabetically spent an entire daily allowance on the cheap ones and
    # left the most valuable nine uncached, which is exactly what happened on
    # the first run.
    missing = sorted(
        (
            (name, prompt, weight)
            for name, items in prompts.items()
            for prompt, weight in items.items()
            if cache_key(client.provider, client.model, prompt, name)
            not in client.cache
        ),
        key=lambda item: -item[2],
    )
    print(
        f"cache: {len(client.cache)} entries present, {len(missing)} missing "
        f"({client.provider} / {client.model}, {client.n_keys} key(s))"
    )

    if dry_run:
        estimate = len(missing) * ESTIMATED_TOKENS_PER_CALL
        print(f"\ndry run: would send {len(missing)} calls, ~{estimate:,} tokens")
        return 0
    if not missing:
        print("\nnothing to do; run: python scripts/run_ablation.py --offline")
        return 0

    delay = 60.0 / max(1, TOKENS_PER_MINUTE // ESTIMATED_TOKENS_PER_CALL)
    print(
        f"warming at one call per {delay:.1f}s to stay under "
        f"{TOKENS_PER_MINUTE:,} tokens/min "
        f"(~{len(missing) * delay / 60:.0f} min)\n",
        flush=True,
    )

    warmed = failed = 0
    started = time.time()
    covered = 0
    outstanding = sum(weight for _, _, weight in missing)
    for index, (name, prompt, weight) in enumerate(missing, start=1):
        try:
            client.complete(prompt, SCHEMAS[name], system_instruction=None)
            warmed += 1
            covered += weight
        except LLMFallback as error:
            failed += 1
            text = str(error).lower()
            if "quota" in text or "429" in text or "rate limit" in text:
                print(
                    f"\n[{index}/{len(missing)}] daily allowance exhausted after "
                    f"{warmed} new entries.\n"
                    "The cache is on disk: re-run tomorrow to resume where this "
                    "stopped. Nothing is lost.",
                    flush=True,
                )
                break
            print(f"  [{index}/{len(missing)}] {name}: {str(error)[:90]}", flush=True)

        if index % 10 == 0 or index == len(missing):
            rate = (time.time() - started) / index
            print(
                f"  [{index:>3}/{len(missing)}] warmed={warmed} failed={failed} "
                f"calls_covered={covered / max(1, outstanding):.0%} "
                f"tokens={client.counters.total_tokens:,} "
                f"eta={(len(missing) - index) * rate / 60:.0f}min",
                flush=True,
            )
        time.sleep(delay)

    print(
        f"\nwarmed {warmed} new entries ({failed} failed), "
        f"{client.counters.total_tokens:,} tokens, "
        f"{len(client.cache)} in cache"
    )
    remaining = len(missing) - warmed - failed
    if remaining > 0:
        print(f"{remaining} prompts still missing - re-run to continue")
    else:
        print("\nready: python scripts/run_ablation.py --offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(dry_run="--dry-run" in sys.argv))
