# Sensitivity: where the advantage holds, and where it collapses

> **The heuristic agent's advantage over the fixed-schedule baseline does not
> generalise. It holds in 20 of 54 regimes (37%), and in 40 of 54 the
> confidence interval excludes zero — so most of the losses are as solid as
> the wins.**

This document exists to answer "you built the world to favour your agent."
Every calibrated parameter in this project is an unsourced placeholder, so the
only defensible claim was always a conditional one. This is that condition,
stated in full including the parts that are unflattering.

## The grid

54 cells, 40 seeds each, heuristic agent against the fixed schedule on paired
worlds. Each axis is moved by a lever that changes the *mechanics*, never by
editing the target it is measured against:

| Axis | Lever | Levels |
| --- | --- | --- |
| Failure regime | mandate amount distribution (×0.55, ×1, ×1.75) | 3 |
| Funds-share regime | per-transaction ceiling (×0.3, ×1, ×4) | 3 |
| Restricted window | NPCI window width — none / calibrated / wide | 3 |
| Bank availability | per-tier uptime (×0.93, ×1) | 2 |

Which agent: the **heuristic**. The plan says to sweep "the winning agent",
but there is no winner — the ablation is blocked on API quota and the
heuristic *tied* the baseline at commit 20. Sweeping the deterministic agent
is the honest reading: it is the best arm actually measured, and it needs no
API key to reproduce.

## Where it holds

| Axis level | Mean delta | Holds in |
| --- | ---: | ---: |
| Failure regime **low** | +24.7k | **72%** of its cells |
| Failure regime calibrated | −27.5k | 33% |
| Failure regime **high** | **−193.9k** | **6%** |
| Funds share **calibrated** | +2.9k | 61% |
| Funds share high | −55.5k | 44% |
| Funds share **low** (tight ceiling) | **−144.1k** | **6%** |
| Window none / calibrated | −92.4k | 28% |
| Window **wide** | −11.9k | **56%** |
| Availability calibrated | −55.6k | 37% |
| Availability degraded | −75.5k | 37% |

Best cell: `calibrated / calibrated / wide / degraded`, **+118.1k**, losing on
30% of seeds. Worst cell: `high / low / wide / degraded`, **−337.7k**, losing
on **100%** of seeds.

## Where it collapses, and why

**When failures are frequent (high regime): 6% of cells hold.** The agent
spends its budget diagnosing and retiming failures that were going to fail
again regardless. The fixed schedule's three cheap retries are a better bet
when the base rate is high, because the agent's extra attempts and its nudges
cost real money against a population that mostly cannot pay. **The agent's
advantage is a low-failure-rate phenomenon.**

**When the per-transaction ceiling is tight (funds share low): 6% of cells
hold.** A tight ceiling converts funds failures into limit failures, and the
agent's response to a limit breach — collect a partial at the largest amount
previously settled — collects less money per success while still paying the
gateway fee. It is doing something sensible and losing anyway.

**The two compound.** `high` failure with a `low` funds share is the worst
corner of the grid, losing on 100% of seeds in every window and availability
combination.

## A finding about the experiment, not the agent

**The calibrated NPCI restricted window never binds.** In 100% of cells, the
`none` and `calibrated` window levels produce *identical* deltas, to the
rupee. Neither arm ever presents a debit inside 10:00–13:00 or 17:00–21:00:
the baseline presents at 09:00 by design (chosen at commit 11 precisely so it
would not be a strawman), and the agent's scheduler prefers 04:00–09:00. Only
the `wide` level — 08:00–14:00 and 16:00–22:00 — reaches either of them.

Two consequences, both worth stating plainly:

1. **The effective grid is 36 distinct regimes, not 54.** Eighteen cells are
   exact duplicates of another eighteen.
2. **An earlier claim in this repository was wrong.** The commit 20 write-up
   and an initial figure caption said the agent's edge comes partly from
   timing around the restricted window. It does not — the window is inert for
   both arms. The caption has been corrected; the claim should not be repeated
   in the README or the video. The agent's edge, where it has one, comes from
   the salary-cycle prior and from diagnosing which failures are worth
   retrying at all.

## What this means for the claim

The honest headline is not "the agent beats the baseline." It is:

> On the calibrated world the deterministic agent ties the industry baseline,
> and across a 54-cell sweep of plausible alternative worlds its advantage
> survives in about a third. It is ahead when failures are relatively rare and
> ceilings are generous, and it is clearly behind when failures are frequent
> or ceilings are tight.

A bounded claim with a named failure regime is worth more than an unbounded
one, and this sweep is the reason the bounded version can be stated at all.

## Caveats

- `observed_failure_rate_per_attempt` in `metrics.json` counts every attempt
  including retries, so it runs far above the ~29% null-world rate validated
  at commit 7. Retries fail much more often than first presentments. The two
  numbers are not comparable and are not meant to be.
- Both arms remain handicapped by the nudge follow-up ignoring the scheduler
  (see `PROGRESS.md`, commits 20 and 25). It affects both identically, so the
  comparison stands, but the agent is understated in absolute terms.
- 40 seeds per cell is enough for a mean and a rough interval, not for a fine
  distinction between adjacent cells.
- Every regime here is a scaling of an unsourced placeholder. The sweep shows
  the result is fragile *within the model*; it says nothing about which regime
  the real world is in.
