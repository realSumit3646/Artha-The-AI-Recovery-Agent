# Calibration

Every number the simulator uses comes from `mandate_recovery.calibration`, and
every one of them is labelled with where it came from. This file is the
human-readable record of those labels.

## Read this before quoting any number below

**Nothing in this table is a published figure yet.** Every value currently in
the code is a placeholder chosen by the author. Each is marked
`confidence = "assumption"`, and each `source` string says plainly that it is
unsourced — either "no public source — author's assumption" where no public
figure exists to find, or a `TODO(sumit)` note naming where a real figure
should come from.

No citation in this project is invented. A `TODO(sumit)` entry is a pointer to
where to look, **not** a claim that the number came from there. Until a value
is actually replaced with a sourced figure, its confidence stays
`"assumption"` — a placeholder is never promoted to `"derived"` merely because
it looks reasonable.

The `confidence` vocabulary is:

| value | meaning |
| --- | --- |
| `published` | taken directly from a named public document, with the figure and reporting period recorded |
| `derived` | computed from published figures by a stated calculation |
| `assumption` | the author picked it; no public figure stands behind it |

`CalibratedValue` refuses at construction time to carry `published` or
`derived` alongside a `TODO(sumit)` or "no public source" string, so the two
columns cannot drift apart.

## Why the failure rate is a range, not a number

Published UPI Autopay failure figures vary widely — by source, by reporting
period, by whether "failure" counts only the first execution attempt or the
whole retry sequence, and by whether the reporter is NPCI, an issuing bank, a
payment gateway, or a merchant with a particular customer mix. Figures drawn
from different quarters are not comparable, and a gateway's numbers reflect
its own merchant base rather than the system as a whole.

This project therefore does two things:

1. **Defaults sit at the conservative (pessimistic) end of the plausible
   range.** A recovery policy that looks good in a harsh environment is a
   safer claim than one that needs a forgiving one. If a policy only beats the
   baseline when failures are rare, that is worth knowing before it ships.
2. **The sensitivity sweep (commit 27) re-runs every arm across the optimistic
   and pessimistic ends of each parameter's range.** The question that sweep
   answers is not "what is the true failure rate" but "does the ranking of
   policies survive being wrong about it". A result that flips under the sweep
   is reported as fragile, whatever the headline number says.

A single number in the table below is the point estimate the simulator runs
with. It is not a claim about reality, and no conclusion in this project
should rest on it alone.

## Parameters

| parameter | value used | source | confidence |
| --- | --- | --- | --- |
| `upi_autopay_execution_failure_rate` | `0.30` | `TODO(sumit)` — NPCI UPI monthly statistics / gateway autopay success reports; pick one reporting period and cite it | `assumption` |
| `share_of_failures_insufficient_funds` | `0.55` | `TODO(sumit)` — gateway or issuer decline-reason breakdown for recurring debits | `assumption` |
| `share_of_failures_technical` | `0.25` | `TODO(sumit)` — gateway or issuer decline-reason breakdown for recurring debits | `assumption` |
| `share_of_failures_limit` | `0.10` | `TODO(sumit)` — gateway or issuer decline-reason breakdown for recurring debits | `assumption` |
| `share_of_failures_window_rejected` | `0.10` | `TODO(sumit)` — gateway or issuer decline-reason breakdown for recurring debits | `assumption` |
| `restricted_window_hours` | `((10, 13), (17, 21))` | `TODO(sumit)` — NPCI circular on processing windows for recurring e-mandates; confirm exact peak hours and circular number | `assumption` |
| `restricted_window_rejection_probability` | `0.35` | no public source — author's assumption (NPCI deprioritises recurring debits at peak but publishes no rejection rate) | `assumption` |
| `bank_availability_by_tier` | `large_private 0.985`, `psu 0.950`, `small_finance 0.920` | `TODO(sumit)` — NPCI publishes bank-wise UPI technical decline rates monthly; aggregate into these three tiers and cite the period | `assumption` |
| `monthly_mandate_revocation_rate` | `0.02` | no public source — author's assumption (revocation is a customer action merchants report privately, if at all) | `assumption` |
| `salary_credit_day_distribution` | 14 days, mass on 1–7 and 25–31, peaking on day 1 (`0.18`) and day 30 (`0.13`) | no public source — author's assumption (no public dataset of Indian payroll credit dates) | `assumption` |
| `card_penetration_rate` | `0.25` | `TODO(sumit)` — RBI monthly card statistics, bounded to the customer segment modelled here | `assumption` |
| `gateway_cost_per_attempt_paise` | `200` (₹2.00) | `TODO(sumit)` — published payment-gateway pricing for recurring mandates | `assumption` |
| `sms_cost_paise` | `15` (₹0.15) | `TODO(sumit)` — published transactional SMS / DLT pricing | `assumption` |
| `voice_call_cost_paise` | `120` (₹1.20) | `TODO(sumit)` — published outbound IVR or agent-call pricing | `assumption` |
| `churn_probability_increment_per_contact` | `0.015` | no public source — author's assumption (the cost of nagging a customer is not something merchants publish) | `assumption` |
| `bank_tier_mix` | `large_private 0.45`, `psu 0.40`, `small_finance 0.15` | `TODO(sumit)` — RBI/NPCI bank-wise account or UPI volume share, narrowed to the segment modelled | `assumption` |
| `monthly_salary_paise_median` | `3_500_000` (₹35,000/month) | `TODO(sumit)` — PLFS or EPFO wage distribution for the salaried segment | `assumption` |
| `monthly_salary_lognormal_sigma` | `0.55` | `TODO(sumit)` — derive from two published wage percentiles, then re-mark as `derived` | `assumption` |
| `monthly_spend_share_of_salary` | `0.75` | `TODO(sumit)` — household consumption survey (MPCE) against the same wage segment | `assumption` |
| `initial_churn_intent_alpha` | `1.5` | no public source — author's assumption (churn intent is unobservable; shape puts most customers near zero) | `assumption` |
| `initial_churn_intent_beta` | `28.5` | no public source — author's assumption (paired with alpha for a mean of 0.05) | `assumption` |
| `per_txn_limit_paise_by_tier` | `large_private 10_000_000`, `psu 10_000_000`, `small_finance 5_000_000` | `TODO(sumit)` — NPCI UPI transaction-limit circulars plus per-bank mandate limits | `assumption` |

The `source` strings in `calibration.py` are authoritative; the column above is
abbreviated for reading. `tests/test_calibration.py` asserts that every
parameter appears in this table and that the confidence column matches the
code, so the two cannot silently diverge.

## Notes on individual parameters

**Failure shares partition the failures.** The four `share_of_failures_*`
values are fractions *of failed executions*, not of all executions, and they
are validated to sum to exactly 1.0. `upi_autopay_execution_failure_rate` is
the separate question of how often an execution fails at all.

**`restricted_window_hours` are half-open** `[start, end)` local clock hours.
`((10, 13), (17, 21))` means 10:00–12:59 and 17:00–20:59. NPCI's
deprioritisation of recurring debits during peak hours is a real policy; the
exact hours here are a placeholder pending the circular.

**Costs are integer paise**, never rupees and never floats, per the project's
money invariant. `200` is ₹2.00.

**`salary_credit_day_distribution` is a probability distribution**, validated
to sum to 1.0 over days 1–7 and 25–31 only. The shape encodes month-end and
first-week clustering. It is an assumption about payroll timing, and it is one
of the parameters most worth sweeping: the whole premise that retry *timing*
matters depends on salary credits being predictable.

**The population parameters describe who the customers are.**
`bank_tier_mix`, the two salary parameters, `monthly_spend_share_of_salary`,
the two churn-intent Beta shapes and `per_txn_limit_paise_by_tier` are what
the simulator samples a population from. They live in `CalibrationSet` rather
than inside the simulator so that they are captured in a stored experiment
config and swept with everything else. Salary is lognormal with the stated
median; daily spend is derived from salary rather than drawn independently,
so a customer's outgoings track their income.

**`card_penetration_rate` bounds the `SwitchRail` action.** A policy cannot
move a customer to card if the customer has no usable card, so this parameter
caps how often that action is available at all.

**`churn_probability_increment_per_contact` is the price of nagging.** It is
what stops a policy from trivially winning by contacting every customer every
day. It is an assumption, it is doing a lot of work, and it should be swept.
