# Ablation: heuristic agent vs LLM agent

> **STATUS: NOT YET RUN. This document reports no result, because there is no
> result to report.** The experiment is implemented, tested and ready; it is
> blocked on API quota. Details below. Nothing in this file should be read as
> a finding, and no number from a partial run has been recorded here.

## What the experiment is

Five arms on the same 120 paired seeds used by every earlier run:

| Arm | What it is |
| --- | --- |
| `do_nothing` | the no-intervention floor |
| `fixed_schedule` | the industry-standard T+1/T+3/T+5 baseline |
| `heuristic` | the deterministic agent — no model anywhere |
| `llm_agent` | the same agent, with a model on residual diagnosis and intervention choice |
| `oracle` | the perfect-information ceiling |

The comparison that decides the project's claim is **`llm_agent` against
`heuristic`**. Both use the same scheduler, the same compliance validator, the
same cost model and the same fallback path. The only difference between them
is whether a language model resolves the diagnoses a code book cannot settle
and chooses the intervention. That is as clean an ablation as this design
allows, and it is why the two arms were built to share everything else.

Four comparisons are declared in `scripts/run_ablation.py` and their count is
written into `metrics.json`, so a reader can see how many were run rather than
how many were reported.

## Why it has not run

The Gemini free tier allows **20 requests per day per model per project**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quota value `20`). The
experiment needs roughly 250–300 *distinct* model calls to warm its cache.

The prompt design already does everything it can to reduce that number.
Prompts are canonical and bucketed — `amount_vs_history` renders as "somewhat
larger than anything they have paid before" rather than a rupee figure — so a
measured run over 10 seeds and 5,000 mandates produced **12,329 model calls
collapsing onto just 198 distinct prompts**. Every one of those 12,329 calls
after the first of its kind is served from disk. That is a 62× reduction, and
it is still an order of magnitude more than a 20/day quota allows.

A first live attempt warmed **17 cache entries** before the quota was
exhausted. Those entries are committed. After that point every call returned
`429 RESOURCE_EXHAUSTED`, retries were exhausted, and the LLM agent fell back
to the heuristic on every decision — which would have produced an `llm_agent`
arm numerically identical to `heuristic` and an ablation that measured
nothing. **The run was stopped rather than recorded.** An arm that is silently
the control is worse than no arm at all, because it looks like a result.

## What unblocks it

1. **Enable billing on the Google AI Studio project.** The paid tier's limits
   are far above what this needs; the whole experiment is a few hundred calls
   and a few million tokens. Then run `python scripts/run_ablation.py`.
2. Or supply a key on a plan with a higher daily allowance.

Once the cache is warm, `python scripts/run_ablation.py --offline` reproduces
the result from `llm_cache/` with no key at all, which is the mode
`make reproduce` uses.

## How the result will be reported

Written down before the number is known, so it cannot be shaded afterwards:

- **If the LLM agent wins:** the margin, the bootstrap 95% CI, the loss rate,
  the model-invocation rate and the token cost, plus a statement of *where*
  the gain came from — residual diagnosis, intervention choice, or messaging.
- **If the heuristic wins or ties:** that goes in the first line of this
  document. The honest conclusion would be that recovery is a scheduling
  problem, that the deterministic policy is what ships for intervention
  selection, and that the model is retained only for residual diagnosis and
  message drafting if it earns those.
- Either way the **loss rate is reported next to the mean**, never omitted.
- The result will **not** be re-run with a tuned prompt to reverse it. If a
  prompt improvement is attempted later it will appear as a separate, labelled
  run alongside the original, not in place of it.

## A caveat that applies whichever way it goes

Both agent arms are currently handicapped by the same harness limitation: a
nudge schedules its follow-up debit at a fixed offset and default hour,
discarding the slot the scheduler would have chosen (see `PROGRESS.md`,
commits 20 and 25). Because it affects both arms identically the head-to-head
remains internally valid, but both are understated against the fixed-schedule
baseline. That should be fixed before any of these numbers are published as a
claim about what an agent can do.
