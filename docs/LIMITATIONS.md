# Limitations

Written before anyone asked, and specific enough to act on. A vague
limitations section reads as performance; this one names the things most
likely to break the conclusions and says which conclusion each one breaks.

## 1. The data is simulated, and the simulator is calibrated to nothing

**Every one of the 22 calibrated parameters is an author's assumption.** Not
one is marked `published` or `derived`. There are no citations in this
repository because I would not stand behind any I could produce from memory,
and inventing them would have been worse than admitting the gap.

`docs/CALIBRATION.md` records which numbers were treated as targets and which
were fitted to them. In short: the overall failure rate and the four
failure-mode shares were treated as targets; the mandate amount distribution,
the presentment timing and — uncomfortably — the bank availability figures
were fitted so the simulator would reproduce those targets.

**The bank availability fit is the weakest link.** NPCI actually publishes
bank-wise technical decline rates monthly, so it is the one fitted quantity a
public source could settle. It was adjusted because two placeholder sets
written at different times disagreed: the original availability figures
implied a 3.9% technical-decline rate while the failure mix demanded 7.5%.
Making two placeholders agree is not evidence either is right.

**What would change if this were wrong:** everything downstream. The bounds,
the headroom, the agent comparison and the sweep are all conditional on this
world. They are internally consistent, not externally validated.

## 2. The messiness distribution is invented

The share of failures returning a generic code (18%), a missing code (5%), and
the rate at which a limit breach is miscoded as a funds failure (25%) are
modelling choices, not measurements. They set how hard diagnosis is, and
therefore how much room a diagnosis stage — rules or model — has to add value.

`TRUE_CAUSE_RECOVERABLE_FRACTION` is 0.33: a code book resolves about a third
of failures unambiguously. Turn the ambiguity down and the rules look better;
turn it up and the model stage looks more necessary. **I chose those numbers
before measuring either.** That ordering is the only thing that makes the
result non-circular, and it is checkable in the git history — the messiness
layer is commit 6, the heuristic agent is commit 19.

## 3. The LLM ablation has not run

The deciding experiment is blocked on API quota (20 requests/day/model on the
free tier against ~250–300 distinct calls needed). **No LLM result exists**,
so nothing in this repository supports a claim about whether a language model
helps. The model layer is built and tested; it is unmeasured.

A partial run was discarded rather than recorded, because with every call
failing the LLM arm would have been numerically identical to the heuristic
control and looked like a null result rather than an absent one.

## 4. The agent's advantage is fragile, and the sweep says so

It holds in 20 of 54 regimes. It collapses when failures are frequent (6% of
cells) or per-transaction ceilings are tight (6%). **The honest claim is
bounded**: this agent helps in a low-failure, generous-ceiling world and hurts
in the opposite one. Nothing here identifies which world a real merchant is
in.

## 5. Known defects in the harness

- **The nudge follow-up ignores the scheduler.** After a contact, the debit is
  re-presented at a fixed offset and the default hour, discarding the
  salary-timed slot the scheduler would have chosen. This handicaps every arm
  that contacts customers. It affects the heuristic and LLM arms identically
  so head-to-head comparisons stand, but both are understated against the
  fixed schedule. **This is the first thing to fix.**
- **The calibrated NPCI window never binds.** Neither arm presents inside it,
  so the sweep's window axis is effectively binary and 18 of 54 cells are
  duplicates. An earlier version of this repository claimed the agent's edge
  came partly from window timing; that claim was wrong and has been corrected.
- **The freeze is incomplete.** `SIMULATOR_HASH` covers the calibration and
  three simulator modules, but the fitted mandate-amount parameters live in
  `ExperimentConfig` and sit outside it. They are captured in every stored
  config, so runs are reproducible, but changing them does not trip the freeze
  test.

## 6. Modelling simplifications

- A uniform 31-day month, so every salary day 1–31 is reachable every cycle.
- One mandate per customer. The oracle and the churn cost both assume this;
  a customer with several mandates would break both.
- Churn is charged as expected value at the moment of contact, never realised.
  No customer actually leaves, so there is no feedback loop from
  over-intervention into the population.
- `EscalateHuman` costs nothing but churn, because no agent-time figure is
  calibrated. That makes escalation look cheaper than it is.
- Partial collections and rail switches are executed as next-day retries
  without modelling a reduced amount settling differently or a card rail
  behaving differently.

## 7. What real merchant data would change

Given a real book of failed UPI Autopay mandates, in order of how much it
would move the conclusions:

1. **The response-code distribution.** If real codes are cleaner than modelled,
   the rules do more and the model stage loses its justification. If they are
   messier, the reverse. This single input drives the project's central
   question.
2. **The churn cost of a contact.** The finding that the agent's timing edge
   is eaten by contact costs rests entirely on the calibrated 0.015 increment.
   Halve it and contact becomes clearly worth it; double it and no policy
   should ever nudge.
3. **The mandate amount distribution.** It was fitted, and it drives both the
   funds-failure rate and the limit-breach rate.
4. **Bank availability by tier.** Published monthly; would replace the
   weakest fitted parameter with a real one.

**Which conclusion is most at risk:** that contacting customers is
value-destroying. It follows directly from one unsourced number multiplied by
an assumed twelve-cycle mandate lifetime. It is the most consequential claim
in the project and the least supported.

## 8. What this project is not

Not a product, not a pilot, and not evidence about the Indian payments market.
It is an evaluation harness with a frozen simulator, an enforced information
boundary, and a set of conditional results about a world that was made up on
purpose and documented as such.
