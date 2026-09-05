# Ablation: heuristic agent vs LLM agent

> **The LLM agent wins. +Rs 117,900 per seed against the deterministic
> heuristic (95% CI Rs 81,461 to Rs 153,122), losing on 33 of 120 seeds
> (27.5%). It is also the first arm to beat the industry baseline with a
> confidence interval that excludes zero: +Rs 102,635, losing on 23.3%.**
>
> **It does not win by recovering more money. It recovers slightly *less*.
> It wins by not contacting customers.**

120 paired seeds, 500 mandates, 90 days, `openai/gpt-oss-120b` at temperature
0, served entirely from the committed cache: **166,116 cache hits, zero live
calls, zero schema failures, 0.00% stage fallback rate.**

## The result

| Arm | Net recovery | Recovery rate | Attempts/rec | Contacts/rec | Over-intervention | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| do_nothing | Rs 11,866 L | 73.1% | 1.37 | 0.000 | 0.0% | 0% |
| fixed_schedule | Rs 13,641 L | 81.3% | 2.06 | 0.000 | 0.0% | 50.1% |
| heuristic | Rs 13,623 L | 82.4% | 1.78 | 0.082 | 8.4% | 49.6% |
| **llm_agent** | **Rs 13,764 L** | 82.0% | 1.94 | **0.008** | **0.0%** | **53.6%** |
| oracle (cheats) | Rs 15,409 L | 84.8% | 1.38 | 0.000 | 0.0% | 100% |

| Comparison | Mean delta | 95% CI | Loss rate |
| --- | ---: | --- | ---: |
| llm_agent vs heuristic | **+Rs 117,900** | Rs 81,461 to 153,122 | **27.5%** |
| llm_agent vs fixed_schedule | **+Rs 102,635** | Rs 76,192 to 130,046 | **23.3%** |
| heuristic vs fixed_schedule | −Rs 15,265 | −Rs 52,047 to +22,500 | 52.5% |
| oracle vs llm_agent | +Rs 1,370,777 | Rs 1,329,403 to 1,413,557 | 0.0% |

Four comparisons were declared before the run and all four are reported.

## Where the gain came from

Not from recovering more. The LLM agent recovers **82.0%** against the
heuristic's **82.4%**, and uses *more* attempts per recovery (1.94 vs 1.78).
On the metric most projects would headline, it is slightly worse.

The difference is restraint:

| | heuristic | llm_agent |
| --- | ---: | ---: |
| Contacts per recovery | 0.082 | **0.008** — 10x fewer |
| Over-intervention rate | 8.4% | **0.0%** |
| Cost per Rs 100 recovered | Rs 4.89 | **Rs 1.54** — a third |

The heuristic's decision table nudges after two failed silent retries, full
stop. The model was told in its prompt what a contact actually costs — roughly
a tenth of the mandate's remaining value in churn risk — and chose to nudge
about a tenth as often. Its over-intervention rate, the share of episodes
where a contacted customer would have paid anyway, fell to zero.

**The model's contribution was knowing when not to act.** That is a more
interesting finding than "the AI recovered more", and it is the opposite of
what a recovery-rate headline would have shown.

## How narrowly the model was used

| | |
| --- | ---: |
| Diagnoses resolved by rules | 103,080 (76.9%) |
| Diagnoses routed to the model | 30,948 (**23.1%**) |
| ...of which the model resolved confidently | 27,677 (89.4%) |
| Stage fallbacks | **0** |
| Schema failures | **0** |
| Message verification failures | **0** |
| Actions refused by the compliance validator | 20,853 |

The model was consulted on under a quarter of failures — only those a
rule-based code book genuinely cannot settle — and the deterministic validator
still refused 20,853 of the actions that came out of the pipeline.

## The run that was discarded, and why it matters

**An earlier run of this exact experiment produced the opposite answer.** It
reported the LLM agent *losing* by Rs 42,597 with a confidence interval
excluding zero, on the same 120 seeds and the same code.

It was wrong because the cache was only 47% covered by call volume. Every
missing prompt raised a fallback, so **65% of the LLM arm's decisions were
actually heuristic decisions** — the arm was mostly its own control, and the
small genuine LLM contribution was swamped.

The cause was a flaw in the warming script rather than in the agent. The
prompt set was enumerated by replaying the experiment against *canned* replies
(always "send a nudge"). But what the agent asks depends on what it was told a
moment ago: a nudge leads to a different next situation, and a different next
question, than a silent retry. So the enumeration described the prompt set of
a **different policy**, and warming it covered 216 prompts that answered fewer
than half the real run's calls.

The fix was to make warming iterative — replay against the *real* cache,
record what misses, warm it, repeat. It converged in five rounds:

| Round | Call coverage | New prompts |
| --- | ---: | ---: |
| 1 | 46.9% | 181 |
| 2 | 99.0% | 16 |
| 3 | 99.96% | 13 |
| 4 | 100.0% | 1 |
| 5 | **100.0%** | **0 — converged** |

444 cached entries, ~298,000 tokens, all committed to this repository.

**Two things follow from this that are worth more than the result itself.**
First, a partially-cached LLM arm does not fail loudly — it quietly becomes
its control and returns a confident, wrong, statistically significant answer.
Second, the discipline of refusing to report the first run was what made the
correct answer reachable. Had it been published, this document would say the
opposite with the same apparent rigour.

## Caveats

- **The whole world is unsourced.** All 22 calibrated parameters are
  assumptions. This result is conditional on a simulator nobody has validated
  against reality.
- **The sensitivity sweep covers the heuristic, not this agent.** The sweep at
  commit 27 ran before the ablation, when no arm had won. Sweeping the LLM
  agent would need cache warming across all 54 regimes, since each regime
  produces different observations and therefore different prompts. Until that
  is done, **there is no evidence this advantage survives a different world** —
  and the heuristic's advantage survived only 37% of regimes.
- **The nudge follow-up still ignores the scheduler**, which handicaps both
  agent arms identically. The comparison holds; both are understated against
  the baseline.
- One model, one provider, one temperature. Nothing here says the result
  generalises to another model.
- The agent's restraint may be an artefact of the prompt stating the churn
  economics explicitly. A prompt that omitted them would likely nudge more and
  score worse. That is prompt design doing real work, and it should be
  reported as part of the system rather than as a property of the model.
