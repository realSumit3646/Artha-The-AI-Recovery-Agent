"""Populate the response cache so the ablation can run offline.

    python scripts/warm_llm_cache.py            # warm until converged
    python scripts/warm_llm_cache.py --dry-run  # count what is missing, call nothing

Why this is a separate step from the ablation: if the quota runs out *during*
an experiment, the LLM agent silently falls back to the heuristic for the
remainder and the arm becomes a blend of two policies. That is worse than a
failed run, because it looks like a result. So the cache is filled first, and
``run_ablation.py --offline`` then executes with every response already on
disk.

How the prompt set is found, and why it takes several rounds
-----------------------------------------------------------
The prompt set is **not** a fixed list. What the agent asks depends on what it
was told a moment ago: a reply of "send a nudge" leads to a different next
situation, and therefore a different next question, than "retry silently". The
set of prompts is a property of the agent *driven by real replies*.

The first version of this script enumerated prompts by replaying the
experiment against canned replies. That describes the prompt set of a
different policy -- one that always nudges -- and it warmed 216 prompts which
turned out to cover only 47% of the real run's calls. The ablation built on it
had a 65% fallback rate and was discarded.

This version replays the experiment against the **real cache**, records every
prompt that misses, warms those, and repeats until a round adds nothing. Each
round makes the agent behave more like its final self, so the set converges.

Pacing
------
Groq's free tier allows 8,000 tokens per minute and a call here measures
~1,400 tokens, so requests are spaced to stay under it. A per-minute throttle
and a per-day exhaustion both arrive as HTTP 429; they are told apart from the
message text, because the first version guessed and mislabelled one as the
other. The provider's raw message is printed either way.

Within a round, prompts are warmed in descending order of how many calls each
serves, so an allowance that runs out mid-round leaves only rare prompts cold.
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
ESTIMATED_TOKENS_PER_CALL = 1_400

SCHEMAS = {
    "DiagnosisReply": DiagnosisReply,
    "InterventionReply": InterventionReply,
    "MessageReply": MessageReply,
}


class _Discoverer:
    """Serves cached replies, and records every prompt that is not cached.

    Answering from the real cache is what makes the discovered prompt set the
    *actual* one. A miss raises, so the agent takes its documented fallback
    exactly as it would in a real offline run.
    """

    offline = True

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.model = client.model
        self.provider = client.provider
        self.hits: Counter = Counter()
        self.misses: Counter = Counter()
        self.missing: dict[str, dict[str, int]] = {}

    def complete(self, prompt, schema, system_instruction=None):
        name = schema.__name__
        key = cache_key(self.provider, self.model, prompt, name)
        if key in self._client.cache:
            self.hits[name] += 1
            return self._client.complete(
                prompt, schema, system_instruction=system_instruction
            )
        self.misses[name] += 1
        bucket = self.missing.setdefault(name, {})
        bucket[prompt] = bucket.get(prompt, 0) + 1
        raise LLMFallback("not cached")

    def stats(self):
        return {}


def discover(client: LLMClient) -> _Discoverer:
    """Replay the full experiment against the real cache, recording misses."""
    watcher = _Discoverer(client)
    config = ExperimentConfig(
        experiment_id="warm",
        seeds=list(range(N_SEEDS)),
        n_customers=N_MANDATES,
        n_mandates=N_MANDATES,
        n_days=N_DAYS,
    )
    run_experiment(
        {"llm_agent": lambda world, mapping: LLMAgentPolicy(watcher)}, config
    )
    return watcher


def main(dry_run: bool = False, max_rounds: int = 8) -> int:
    client = LLMClient(cache_dir=REPO_ROOT / "llm_cache")
    print(
        f"{client.provider} / {client.model}, {client.n_keys} key(s), "
        f"{len(client.cache)} cached\n"
    )

    total_warmed = 0
    for round_number in range(1, max_rounds + 1):
        print(f"--- round {round_number}: replaying against the cache ---", flush=True)
        started = time.time()
        found = discover(client)

        hits = sum(found.hits.values())
        misses = sum(found.misses.values())
        coverage = hits / max(1, hits + misses)
        missing = sorted(
            (
                (name, prompt, weight)
                for name, items in found.missing.items()
                for prompt, weight in items.items()
            ),
            key=lambda item: -item[2],
        )
        print(
            f"  {time.time() - started:.0f}s   call coverage {coverage:.1%}   "
            f"({hits:,} hits, {misses:,} misses)   {len(missing)} new prompts"
        )
        for name in sorted(set(found.hits) | set(found.misses)):
            print(
                f"    {name:<20} hits={found.hits[name]:>8,}  "
                f"misses={found.misses[name]:>8,}  "
                f"new={len(found.missing.get(name, {})):>4}"
            )

        if not missing:
            print(
                f"\nCONVERGED: every prompt is cached, call coverage "
                f"{coverage:.1%}.\n{len(client.cache)} entries, "
                f"{total_warmed} added this session.\n\n"
                "ready: python scripts/run_ablation.py --offline"
            )
            return 0

        if dry_run:
            estimate = len(missing) * ESTIMATED_TOKENS_PER_CALL
            print(
                f"\ndry run: round {round_number} would send {len(missing)} "
                f"calls, ~{estimate:,} tokens"
            )
            return 0

        delay = 60.0 / max(1, TOKENS_PER_MINUTE // ESTIMATED_TOKENS_PER_CALL)
        print(
            f"  warming {len(missing)} prompts, one per {delay:.0f}s "
            f"(~{len(missing) * delay / 60:.0f} min)",
            flush=True,
        )

        warmed = 0
        covered = 0
        outstanding = sum(weight for _, _, weight in missing)
        exhausted = False
        started = time.time()

        for index, (name, prompt, weight) in enumerate(missing, start=1):
            try:
                client.complete(prompt, SCHEMAS[name], system_instruction=None)
                warmed += 1
                covered += weight
            except LLMFallback as error:
                text = str(error).lower()
                throttled = any(m in text for m in ("quota", "429", "rate limit"))
                daily = any(
                    m in text
                    for m in ("per day", "rpd", "tpd", "daily", "resource_exhausted")
                )
                if throttled and daily:
                    print(
                        f"\n  allowance exhausted after {warmed} new entries, "
                        f"covering {covered / max(1, outstanding):.0%} of this "
                        f"round's outstanding calls."
                        f"\n  provider said: {str(error)[:160]}"
                        "\n  The cache is on disk; re-run to resume.",
                        flush=True,
                    )
                    exhausted = True
                    break
                if throttled:
                    print("    throttled, waiting 60s", flush=True)
                    time.sleep(60)
                    continue
                print(f"    [{index}] {name}: {str(error)[:110]}", flush=True)

            if index % 25 == 0 or index == len(missing):
                rate = (time.time() - started) / index
                print(
                    f"    [{index:>3}/{len(missing)}] warmed={warmed} "
                    f"calls_covered={covered / max(1, outstanding):.0%} "
                    f"tokens={client.counters.total_tokens:,} "
                    f"eta={(len(missing) - index) * rate / 60:.0f}min",
                    flush=True,
                )
            time.sleep(delay)

        total_warmed += warmed
        print(
            f"  round {round_number}: +{warmed} entries, "
            f"{len(client.cache)} in cache\n"
        )
        if exhausted:
            print("re-run once the allowance resets to continue.")
            return 1

    print(
        f"\nstopped after {max_rounds} rounds without converging. The prompt "
        "set is still growing; inspect before spending more."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(dry_run="--dry-run" in sys.argv))
