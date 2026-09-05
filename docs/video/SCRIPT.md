# Pitch video script — 5 minutes

The honest architecture is the differentiator, not the recovery number. The
number is a tie. Lead with how the thing was built to be checkable, and let
the result be what it is.

**One rule while recording:** do not say the agent beats the baseline. It does
not. Every phrasing below is written to be true.

---

## 0:00–0:45 — The problem

> "A UPI Autopay mandate fails. The merchant sees a response code and nothing
> else. They cannot see that the customer's account was empty three days
> before payday, or that their bank was down for an hour, or that they had
> already decided to cancel. So the industry does the only thing it can: retry
> on a fixed calendar. T+1, T+3, T+5. That schedule spends customer goodwill
> on people who were never going to pay, and gives up on people who would have
> paid on Friday."

> "I wanted to know whether reasoning about *why* a payment failed does better
> than retrying on a timer. Published failure figures for UPI Autopay vary
> widely by source and reporting period, so rather than pick a flattering one,
> I built a simulator, labelled every parameter as an assumption, and made the
> whole thing reproducible."

**On screen:** `docs/CALIBRATION.md`, scrolled to the parameter table so the
`assumption` column is visible on every row.

---

## 0:45–1:15 — Why this is hard

> "The hard part is that the code lies. The same failure returns a different
> string at different banks. About a fifth come back as a generic 'DECLINED'
> or blank. And a breach of the customer's transaction ceiling is sometimes
> reported using the *insufficient funds* code — which you can only catch by
> noticing they previously paid you a larger amount."

> "So I built that ambiguity in deliberately, before writing any policy. A
> lookup table resolves about a third of failures cleanly. That number is a
> design constraint with a test enforcing it, because if the codes were clean,
> a diagnosis stage would be theatre."

**On screen:** `docs/MESSINESS.md`, then figure `4_diagnosis_coverage.png`.

---

## 1:15–2:00 — The design

> "Three things make this checkable rather than just plausible."

> "One — the simulator was frozen before any policy existed. There's a hash
> over the world's source and its parameters, and a test that fails if it
> moves. You can check the commit order in git rather than take my word."

> "Two — information asymmetry is enforced by types. Policies receive one
> object containing only what a real collector could look up. A test walks
> every policy and fails if its signature *or its constructor* mentions
> simulator state. There's one exception, an oracle with perfect information,
> and it exists purely to compute the ceiling."

> "Three — every arm faces a bit-identical world on a given seed. That makes
> each comparison a paired measurement rather than a difference of two noisy
> averages."

**On screen:** the mermaid diagram in `README.md`, pausing on the one-way
arrow into `Observation`.

---

## 2:00–3:15 — The demo

Run in a clean terminal, full screen, large font:

```
python docs/video/demo_audit_trail.py
```

> "This is one mandate's entire life under the agent. Rupees thirty-two
> thousand, due on the 17th."

> "First attempt fails with 'DECLINED' — generic, undiagnosable. The agent
> says so rather than guessing, then schedules a retry for the 30th, because
> month-end is when salaries land."

> "Second failure. Two silent retries have now failed, so it decides to
> message the customer — and look at this line: *contact deferred from 04:00
> to 09:00*. The agent wanted to act at four in the morning. The compliance
> validator queued it for business hours. The model never gets to make that
> call."

> "Later the bank returns 'PS-51', and the agent identifies the bank from its
> code vocabulary — but says the funds reading is *not confirmed*, because
> this customer has never settled a payment it could compare the amount
> against."

> "Every decision has a reason written for a human, and every rupee figure in
> a customer message is templated and verified. The model is never shown an
> amount, so it cannot get one wrong."

**Timing note:** the trail is 5 decisions. Scroll slowly; the deferral line
and the "not confirmed" line are the two moments worth pausing on.

---

## 3:15–4:15 — Results, including the parts that hurt

> "Four arms, 120 paired seeds, half a million mandate-cycles."

**On screen:** `1_recovery_bounds.png`.

> "Doing nothing recovers 73%. The fixed schedule gets to 81%, capturing half
> the headroom to a perfect-information ceiling. The agent reaches 82.4% with
> fewer attempts — and ties on money."

> "Mean difference: minus fifteen thousand rupees per seed. Confidence
> interval spans zero. **It lost on 63 of 120 seeds.** I'm saying that out
> loud because a mean without a loss rate is how a fragile result gets sold as
> a solid one."

**On screen:** `5_paired_deltas.png`, then `6_advantage_survival.png`.

> "Why? Contacting a customer raises their churn probability. On a mandate
> with a year left to run that's about thirteen percent of its value, spent to
> buy one point of recovery. The agent's timing edge pays for the contact that
> produced it."

> "And the sweep says the advantage doesn't generalise: it holds in 20 of 54
> alternative worlds. It collapses when failures are frequent or ceilings are
> tight."

---

## 4:15–5:00 — What broke, and what's next

> "Three things I'd want you to know."

> "The LLM ablation hasn't run. The model layer is built, tested, and the
> whole pipeline works — but the free tier allows twenty requests a day and it
> needs about three hundred. A partial run had the model falling back on every
> decision, which would have looked like a null result instead of an absent
> one. I threw it away rather than report it."

> "The first time I enabled the agent's messaging, its contact path was
> completely dead — twenty thousand nudges refused because the scheduler
> retries at 4am and you may not text customers at 4am. Fixing it made the
> agent *worse*, because contact genuinely costs more than it recovers. Both
> numbers are in the log."

> "And there's a defect I know about and haven't fixed: after a nudge, the
> follow-up debit ignores the scheduler. It handicaps both agents equally so
> the comparison stands, but both are understated."

> "With real merchant data, the number I'd want first is the actual
> distribution of response codes. If real codes are cleaner than I modelled,
> the rules do the work and the model isn't justified. If they're messier, the
> opposite. That single input decides this project's central question, and
> right now it's an assumption I wrote down myself."

---

## Figures, in order of use

| # | File | Used at |
| --- | --- | --- |
| 4 | `4_diagnosis_coverage.png` | 0:45 |
| — | README mermaid diagram | 1:15 |
| 1 | `1_recovery_bounds.png` | 3:15 |
| 5 | `5_paired_deltas.png` | 3:45 |
| 6 | `6_advantage_survival.png` | 4:00 |
| 2 | `2_simulator_validation.png` | spare — if asked "is the simulator honest?" |
| 3 | `3_restricted_window.png` | spare — see the caveat in `DEMO_NOTES.md` |

## Phrases to avoid

- ~~"the agent beats the baseline"~~ — it ties. Say "matches, with fewer
  attempts, and I'll show you why that isn't a win."
- ~~"the agent times around the NPCI restricted window"~~ — **false.** The
  sweep proved the window never binds for either arm.
- ~~"our calibration is based on published figures"~~ — none of it is.
- ~~"the LLM agent..."~~ — it has not been measured. Say "the LLM layer is
  built but unmeasured."
