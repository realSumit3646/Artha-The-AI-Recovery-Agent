# PROGRESS

Running log of the build. One entry per commit, appended in commit order
(newest at the bottom). Format:

```
## Commit N — <title>
**Done:** what now exists
**Decisions:** any judgment call made and why
**Open:** anything deferred to a later commit
```

---

## Commit 1 — Scaffold project, invariants, and tooling

**Done:** Empty-but-real project skeleton. The project's binding invariants
are kept in a local, untracked file rather than committed. `pyproject.toml`
declares the `mandate_recovery` package on a src layout with the seven
approved dependencies. `README.md` states the problem and is marked under
construction. Tooling: `Makefile` (install / test / lint / reproduce),
`.gitignore`, `.env.example`, `docker-compose.yml`. Package and test packages
exist but are empty.

**Decisions:**
- `requires-python = ">=3.11,<3.12"` rather than an open `>=3.11`. The
  reproducibility invariant means a run must be re-executable from its stored
  config; an unpinned upper bound lets the interpreter drift out from under a
  stored experiment.
- `pydantic>=2` — a lower bound only, not a new dependency. The
  information-asymmetry invariant says the observation boundary is enforced by
  types, and the v1/v2 validation APIs are not interchangeable, so the major
  version has to be nailed down now.
- `pytest` sits in the main `dependencies` table rather than an
  `optional-dependencies` dev extra, so `make install` gets a test-capable
  environment in one step. This is a research harness; there is no slim
  production install to protect.
- `make lint` runs `python -m compileall` (stdlib only) and prints that no
  linter is configured. The no-unrequested-dependencies rule forbids adding
  ruff/black/mypy unprompted, and a target that silently does nothing is worse
  than one that says what it is not doing.
- `docker-compose.yml` uses the stock `python:3.11-slim` image with a bind
  mount and an inline command instead of a project `Dockerfile`, because the
  scope for this commit is an exact file list that does not include one. No
  database service: SQLite in a local file, per the brief.

**Open:**
- Everything substantive: `Observation` type and the test that enforces
  information asymmetry, the simulator and its latent state, the policy
  interface, the metrics layer, the `results/` writer, and the `typer` CLI.
- `make reproduce` is a stub that echoes "not yet implemented".
- A real `Dockerfile` and a pinned lockfile, once the dependency set has
  settled.
- Linter choice, if and when one is requested.
- `make test` currently exits non-zero: pytest returns exit code 5 when it
  collects no tests, and there are none yet. This resolves itself with the
  first real test (the information-asymmetry test); do not paper over it by
  making the target ignore pytest's exit status.

---

## Commit 2 — Domain types and the observation boundary

**Done:** `src/mandate_recovery/types.py` defines the domain model as frozen
pydantic models: `Mandate`, `LatentCustomerState` (simulator-private),
`BankResponse`, `Attempt`, `Observation`, the `Action` union
(`RetrySilent`, `SendNudge`, `CollectPartial`, `SwitchRail`, `EscalateHuman`,
`Stop`) and `Decision`, plus the `MandateStatus`, `Rail`, `AttemptOutcome` and
`NudgeChannel` enums. `tests/test_types.py` holds 17 tests, including
`test_observation_contains_no_latent_fields` and
`test_all_money_fields_are_typed_int`. Suite is green on Python 3.11.

The two enforcement tests were mutation-checked rather than merely run:
leaking `churn_intent` onto `Observation` fails the boundary test, and
retyping a `*_paise` field as `float` fails the money test. Both pass again
once the mutation is reverted.

**Decisions:**
- Each `Action` variant carries a `kind: Literal[...]` tag so the union is a
  real pydantic discriminated union. The task listed constructor arguments
  only; a tagged union cannot be built without a discriminator, and an
  untagged one resolves by trial order, which would silently mis-parse a
  stored decision on replay.
- `attempt_history` is `tuple[ObservedAttempt, ...]`, not `list`. A frozen
  model with a list field is only shallow-frozen: a policy could mutate the
  history it was handed, and the model would not be hashable. The wire shape
  is unchanged — it still serialises as a JSON array.
- `ObservedAttempt` was introduced for the `{day, hour, rail, raw_code}`
  entries, since the shape needed a name to be validated.
- `ObservedAttempt` carries the bank's `raw_code` string and deliberately not
  `AttemptOutcome`. The classified outcome is the simulator's reading of its
  own latent state; a real collector sees response codes and has to infer.
- Money fields are `Annotated[int, Field(strict=True, ...)]`. Without
  `strict`, pydantic accepts `49900.0` and coerces it, which would put float
  arithmetic in the money path — the exact thing the paise rule forbids.
- Every model sets `extra="forbid"`. On `Observation` this is load-bearing:
  passing a latent fact as an extra keyword is a construction-time error
  instead of a silently attached attribute.
- `scheduled_at` and `timestamp` are `datetime`, while `Observation` exposes
  coarse `day`/`hour` integers. The asymmetry is intentional and matches the
  spec. Neither field has a `now()` default, so nothing reads the wall clock.
- `due_day` is a day of the month (1–31, matching `Mandate.day_of_month`);
  `current_day` is the simulation day index. Both are documented inline
  because the names alone do not distinguish them.
- `Decision.validated` defaults to `False`, so an unreviewed decision is never
  actionable by accident.

**Open:**
- `NudgeChannel` (SMS/WHATSAPP/EMAIL/IVR) and `tone_level` (strict int, `ge=1`,
  no upper bound) are provisional: the task specified `channel` and
  `tone_level` without types. Fix the scale when the outreach model is built.
- Two time representations now coexist — `datetime` internally, `day`/`hour`
  on the observation. If that friction shows up in the simulator, a `SimTime`
  value object is the likely answer.
- Nothing sets `Decision.validated` yet; the deterministic validator that
  approves actions is still to be written.
- Resolved from Commit 1: `make test` now exits zero (17 passed), and
  `pip install -e .` with the full dependency set is verified on Python
  3.11.9 — previously unproven because the sandbox had no network.

---

## Commit 3 — Calibration layer with explicit assumptions

**Done:** `src/mandate_recovery/calibration.py` holds every number the
simulator is allowed to use. Each of the 15 parameters is a frozen
`CalibratedValue` carrying `value`, `unit`, `source` and `confidence`, and
`CalibrationSet` gathers them with cross-parameter validation: failure shares
sum to 1.0, the salary distribution sums to 1.0 over days 1–7 and 25–31, all
three bank tiers are present, and every rate stays within [0, 1].
`docs/CALIBRATION.md` is the readable record — the required
parameter/value/source/confidence table, plus the note on why published
Autopay failure figures vary and how the sweep handles it.
`tests/test_calibration.py` adds 29 tests; suite is 46 green.

Both required assertions were mutation-checked with the model-level guard
disabled, so the test is doing the catching: shares summing to 1.05 fails
`test_failure_shares_sum_to_one`, and a blanked source fails
`test_every_calibrated_value_has_a_non_empty_source`. Two honesty mutations
were checked the same way.

**Decisions:**
- **Every parameter is `confidence="assumption"`. None is `published` or
  `derived`.** That is the honest state: no figure here came from a document I
  can name, so none is labelled as if it had. Sources use one of two forms —
  the exact `NO_PUBLIC_SOURCE` wording where no public figure exists to find
  (revocation rate, churn increment, payroll timing, window rejection rate),
  or a `TODO(sumit)` placeholder naming where a real figure should come from
  (failure rate and its shares, NPCI window hours, bank availability, card
  penetration, the three costs). A `TODO(sumit)` string is a pointer to where
  to look, never a claim that the number came from there.
- The honesty rule is structural, not editorial: `CalibratedValue` raises if
  `published` or `derived` appears alongside a `TODO(sumit)` or
  no-public-source string. A guess cannot be relabelled without the model
  refusing it.
- `CalibratedValue` is generic (`CalibratedValue[T]`) so each parameter's
  value type is enforced, not just its envelope. Verified that a float cannot
  reach a paise field even by passing a hand-built `CalibratedValue`.
- Costs reuse `NonNegativePaise` from `types.py` rather than a fresh int
  alias, so the money rule has one definition.
- `BankTier` is an enum rather than raw string dict keys, and a validator
  requires all three tiers, so a missing tier fails loudly instead of
  KeyError-ing mid-run.
- `restricted_window_hours` is a tuple of half-open `[start, end)` hour pairs,
  which states the boundary convention in the type instead of leaving it to
  be rediscovered in the simulator.
- Dict-valued parameters stay `dict`, unlike Commit 2's tuple-over-list
  choice. The task specified dicts, and the risk that motivated tuples there
  (a policy mutating what it was handed) does not apply: calibration is read
  by the simulator, never passed to a policy.
- `tests/test_calibration.py` parses the docs table and asserts every
  parameter appears in it with a matching confidence, so code and docs cannot
  drift apart silently.

**Open:**
- **No value here is usable for a quotable result yet.** All 15 are
  placeholders awaiting Sumit's published figures. Nothing downstream should
  be reported as a finding until they are filled in.
- `CalibratedValue` records a point estimate only. The sweep promised in
  `docs/CALIBRATION.md` needs an optimistic/pessimistic range per parameter,
  and there is nowhere to put one yet — either a `range` field or a separate
  sweep config is needed before commit 27.
- Calibration must be settled before the simulator freeze at commit 8;
  after that, changing these numbers means re-running all arms.
- The commit message in the task was truncated mid-word ("explicit ass");
  completed to "explicit assumptions".

---

## Commit 4 — Latent world: customers, balances, banks, calendar

**Done:** `src/mandate_recovery/sim/` now holds `World`, the simulator's
private ground truth. It samples a population of `LatentCustomerState` from
the calibrated distributions, assigns each customer a bank tier, evolves
balances day by day (salary credited at the start of the day, then spend,
floored at zero), answers `bank_available(bank_id, day, hour)` and
`in_restricted_window(hour)`, and steps forward with `advance_day()`. The
class docstring states in full that the object is simulator-private and must
never reach a policy. `tests/sim/test_world.py` adds 20 tests; suite is 66
green.

All three required properties were mutation-checked: sampling from an
unseeded generator fails the identical-trajectory tests, removing the zero
floor fails the negative-balance test, and shifting the salary-day comparison
by one fails both salary-timing tests.

**Decisions:**
- **Calibration was extended by 7 parameters, with approval, because
  `CalibrationSet` could feed only one of `LatentCustomerState`'s six
  fields.** Added `bank_tier_mix`, `monthly_salary_paise_median`,
  `monthly_salary_lognormal_sigma`, `monthly_spend_share_of_salary`,
  `initial_churn_intent_alpha`, `initial_churn_intent_beta` and
  `per_txn_limit_paise_by_tier` — seven rather than the six estimated, since
  a Beta prior needs two shape parameters. All are marked `assumption` with
  `TODO(sumit)` or no-public-source strings, like every other parameter.
  `docs/CALIBRATION.md` gained matching rows; `tests/test_calibration.py`
  needed no change and now validates the new parameters automatically.
- Population parameters live in `CalibrationSet`, not in the simulator, so
  they land in a stored experiment config and are swept at commit 27. Putting
  them in `world.py` would have quietly exempted them from both.
- **Bank availability is pre-drawn for the whole run** at construction, and
  `bank_available` is a lookup. Sampling per call would make the answer
  depend on how often it was asked, so a policy that retried more would
  silently shift the bank's uptime — a leak dressed as noise.
- **One representative bank per tier**, with `bank_id` equal to the tier's
  value. Several banks per tier would need a bank count, and there is no
  calibrated figure for one.
- **The month is a uniform 31 days**, so simulation day `d` is day-of-month
  `(d % 31) + 1`. The calibrated salary distribution puts mass on days 25–31;
  a real calendar would fold that mass onto shorter months and distort the
  very timing effect the experiment is about.
- Balances are held as a numpy `int64` array and `LatentCustomerState` is
  built on demand. Copying a frozen model per customer per day would be
  millions of allocations in a full run, for no gain.
- **Opening balances are wound forward from each customer's last salary
  credit**, rather than everyone starting at a full salary. A population in
  lockstep on day 0 would make the first cycle unrepresentative.
- Daily spend is derived from salary (`share / 31`) rather than drawn
  independently, so outgoings track income instead of pairing rich customers
  with poor spending at random.
- `__init__` rejects anything that is not a `numpy.random.Generator`,
  including a legacy `RandomState`, which makes invariant 5 a runtime error
  rather than a convention.
- `_probability_like` in `calibration.py` previously guessed which parameters
  were probabilities from their names. Shape parameters like
  `monthly_salary_lognormal_sigma` read like rates and are not, so exemptions
  are now an explicit `_UNBOUNDED_PARAMETERS` set.

**Open:**
- `test_balances_never_go_negative` is close to vacuous under the default
  calibration: salary far exceeds spend, so no balance approaches zero. The
  floor is really enforced by
  `test_balances_never_go_negative_when_spending_outruns_income`, which
  drives spend to 20x salary. Keep both; the weak one only becomes meaningful
  once debits start drawing balances down.
- No mandates, attempts or outcomes yet — `World` is the stage, not the play.
  Nothing yet consumes `upi_autopay_execution_failure_rate`, the failure
  shares, `restricted_window_rejection_probability`, the revocation rate,
  `card_penetration_rate` or any cost parameter.
- The uniform 31-day month is a simplification worth revisiting only if a
  result turns out to depend on it.
- Commit 8 freezes the simulator. Calibration and world dynamics should be
  settled before then.

---

## Commit 5 — Attempt resolution across all failure modes

**Done:** `src/mandate_recovery/sim/outcomes.py` implements
`resolve_attempt(world, mandate, attempt, rng)`, walking the fixed order
revoked → restricted window → bank down → over limit → insufficient funds →
success, and debiting the latent balance only on success. Revocation is
`revoke_eligible_mandates(...)`, which applies the calibrated monthly rate as
a daily hazard to mandates whose customer has repeatedly failed for funds.
`tests/sim/test_outcomes.py` adds 30 tests; suite is 96 green.

Five mutations were checked: dropping the debit and halving it both fail the
exact-deduction tests, swapping the limit and balance checks fails the
ordering test, drawing from global numpy state fails the same-seed test, and
dropping the repeat-failure condition fails the eligibility test.

**Decisions:**
- **`world.py` was extended, with approval**, because `resolve_attempt` had
  no way to move money or to find a customer. It gained `debit()`, a stable
  customer-id registry (`customer_ids`, `customer_id_for`,
  `index_for_customer_id`, ids minted as `c000000`…) and a read-only
  `calibration` property. Without the registry the mandate→customer link
  would have been an undocumented string convention; without the property,
  `outcomes.py` would have reached into a private attribute.
- `debit()` refuses to overdraw rather than flooring at zero. A debit larger
  than the balance means the caller skipped the funds check — a simulator
  bug, not a customer outcome, and it should crash rather than quietly
  produce a plausible number.
- **Response codes are `SYNTHETIC_RAW_CODES` (`SIM_OK`, `SIM_NSF`, …) and are
  named so nobody mistakes them for real NPCI or issuer codes.** No real code
  vocabulary is invented. Marked `TODO(sumit)` to replace before results are
  quoted.
- The daily revocation hazard is `1 - (1 - monthly) ** (1/31)`, not
  `monthly / 31`, so that compounding over a month reproduces the calibrated
  monthly figure instead of overshooting it. Asserted in a test.
- "Repeatedly failed" is read as at least 2 insufficient-funds failures, held
  in `MIN_INSUFFICIENT_FUNDS_FAILURES_FOR_REVOCATION` and overridable per
  call. It is a reading of the word, not a calibrated figure.
- The rng is consumed only in the restricted-window branch, and a test pins
  that: a quiet-hour attempt leaves the generator untouched, so adding retry
  attempts outside peak hours cannot shift any other stream.
- Revocation draws once per *eligible* mandate, in order, so the stream does
  not depend on which mandates happen to be revoked.

**Open:**
- **The calibrated failure rate and shares are still unused, and the
  mechanics may not reproduce them.** `upi_autopay_execution_failure_rate`
  (0.30) and the four `share_of_failures_*` values describe a failure mix,
  but outcomes now emerge from balances, uptime and windows instead of being
  imposed. Nothing checks that the emergent mix matches the calibrated one.
  Decide before the commit 8 freeze whether those parameters are targets the
  mechanics must be fitted to, or validation figures to compare against.
- `resolve_attempt` resolves on `world.current_day` and reads only the *hour*
  from `attempt.scheduled_at`; the date on it is ignored. An attempt
  scheduled for a different day resolves on the wrong day silently. Fixing it
  needs either a calendar epoch on `World` or an explicit day argument.
- The code mapping is one-to-one, so a policy can invert a code to the exact
  outcome. Real rails are noisier and sometimes miscode a technical decline
  as a funds failure, which would make the inference problem harder and the
  experiment more honest.
- Still unconsumed: `card_penetration_rate` and all three cost parameters.
- No run loop yet — nothing calls `resolve_attempt` in sequence, tracks the
  failure counts revocation needs, or advances the world alongside it.

---

## Commit 6 — Diagnostic messiness layer

**Done:** `src/mandate_recovery/sim/response_codes.py` makes diagnosis a real
inference problem. `encode_response(outcome, bank_id, rng)` applies per-tier
code vocabularies, a generic `DECLINED` bucket, missing codes, and the
contradiction case where a limit breach reports a funds code.
`TRUE_CAUSE_RECOVERABLE_FRACTION` is 0.33; the test recomputes it from
200,000 samples and asserts it stays below 0.75. `docs/MESSINESS.md` explains
the design and states plainly that every share is an author's assumption.
`tests/sim/test_response_codes.py` adds 15 tests; suite is 112 green.

**Decisions:**
- **`outcomes.py` and `test_outcomes.py` were changed, outside the task's file
  list.** Without wiring `encode_response` into `resolve_attempt` the layer
  would be dead code and commit 7 would validate a simulator nobody uses. The
  1:1 `SYNTHETIC_RAW_CODES` table from commit 5 is gone, and the two tests
  that asserted on it were rewritten. This closes the "codes invert perfectly"
  item left open at commit 5.
- Resolution now consumes randomness on every *failure*, where before it only
  did so in the restricted-window branch. `SUCCESS` and `MANDATE_REVOKED` are
  still encoded cleanly and consume nothing, so a successful quiet-hour
  attempt still leaves the generator untouched. Both facts are pinned by
  tests.
- "Unambiguously determinable" is read strictly: a code counts only if no
  other cause ever emits it. That makes every funds code ambiguous, since
  miscoded limit breaches hide inside it — which is the point of the
  contradiction case, and the reason
  `Observation.max_historical_success_amount_paise` exists.
- The three messiness shares live in `response_codes.py`, not in
  `CalibrationSet`. They are not claims about the world that a published
  figure could settle; they are a modelling choice about how hard the
  inference problem should be. Recorded in `docs/MESSINESS.md` so the
  manufactured difficulty is visible.
- Code strings are invented and documented as such. No real NPCI, UPI or
  issuer vocabulary is claimed.

**Open:**
- The messiness shares are unswept. If a result hinges on them they belong in
  the sensitivity sweep, and in `docs/LIMITATIONS.md`.
- A lookup table now resolves only ~33% of failures cleanly. That is the
  intended difficulty, but it means the heuristic agent's UNKNOWN rate will be
  high — which is the argument for the LLM diagnosis stage, and should be
  reported as such rather than treated as a defect.

---

## Commit 7 — Distribution validation against calibration targets

**Done:** `src/mandate_recovery/figures.py` is the shared plotting module:
one style applied at import, Okabe-Ito palette, and `save_figure` writing a
300-dpi PNG, an SVG and a reusable `.txt` caption for every figure.
`scripts/validate_simulator.py` runs the null world — 50 seeds x 500 mandates
x 90 days, no policy — and writes `results/simulator_validation/` with
`config.json`, `metrics.json` and the four required figures.
`tests/sim/test_distributions.py` adds 13 tests; suite is 125 green in under
four seconds.

Observed across 72,560 attempts and 21,161 failures:

| metric | observed | target | delta |
| --- | --- | --- | --- |
| failure rate | 0.2916 | 0.30 | -0.0084 |
| insufficient funds share | 0.5505 | 0.55 | +0.0005 |
| technical share | 0.2565 | 0.25 | +0.0065 |
| limit share | 0.0902 | 0.10 | -0.0098 |
| window share | 0.1028 | 0.10 | +0.0028 |

All inside the 2-point and 3-point tolerances. The restricted window is
plainly visible in `failure_rate_by_hour`: 27% baseline stepping to ~52%
inside both bands.

**Decisions:**
- **The calibrated numbers were mutually inconsistent, and this commit
  resolved it by changing `calibration.py` and `docs/CALIBRATION.md`, both
  outside the task's file list.** The old availability placeholders implied a
  3.9% technical-decline rate; the failure mix demanded 7.5%. No reading made
  both true. Availability moved to `0.955 / 0.910 / 0.865`.
- Fitting direction is recorded in `docs/CALIBRATION.md` under "which numbers
  were chosen to match which". Failure rate and mode shares are **targets**;
  the mandate amount distribution, presentment timing and bank availability
  were **fitted** to them. Availability is the weakest link, because NPCI
  really does publish bank-wise decline data — the doc says so, and says to
  replace that parameter first and re-derive the shares from it.
- Mandate generation lives in the validation script, not in a shared module,
  because no mandate generator exists yet and the harness arrives at commit 13.
  Its parameters are written into `config.json` so the run stays reproducible
  from stored config.
- One mandate per customer. Two mandates competing for one balance is
  realistic but confounds the funds-failure rate, which is the number this run
  exists to measure.
- Revoked mandates are not presented, so `MANDATE_REVOKED` never enters the
  observed mix and the four calibrated shares partition failures exactly.
- The test suite runs its own 15-seed validation rather than reading
  `metrics.json`, so it fails on a broken simulator even on a clean checkout
  where the script has never been run.

**Open:**
- **Agreement with the calibration is not evidence the parameters are right.**
  Every value involved is still `assumption`. All this proves is internal
  consistency.
- Mandate-generation parameters must move into `CalibrationSet` when the
  harness lands at commit 13, or the sweep at commit 27 will not reach them.
- A ₹8,800 median mandate is high for a subscription book; it is what the
  calibrated funds-failure share requires given the salary distribution. If
  real amount data says otherwise, the failure shares must move instead.
- `card_penetration_rate` and the three cost parameters are still unconsumed.

---

## Commit 8 — FREEZE

**Done:** `src/mandate_recovery/sim/freeze.py` pins
`SIMULATOR_HASH = fd5a8fed4eaf5a6d719e9470a2978f93dfd2dccfe0fd2e59de65801b9b31b193`,
a SHA256 over the serialised `CalibrationSet` and the normalised source of
`world.py`, `outcomes.py` and `response_codes.py`.
`tests/sim/test_freeze.py` recomputes it and fails with the required
re-baseline message. `docs/FREEZE.md` records the date, the hash, the reason,
and what the hash does not cover. Suite is 137 green. Tagged `sim-freeze`.

The check was verified by appending a comment to `outcomes.py`: the hash moved
and the test failed with the intended message. `outcomes.py` was restored
byte-identical afterwards.

**The world is now fixed.** Nothing in the three frozen modules was chosen
with knowledge of how any policy performs against it, because no policy exists
yet — that ordering is checkable in the git history, not just asserted here.

**Decisions:**
- Line endings are normalised to `\n` before hashing. `core.autocrlf` is on
  for this repository, so without normalisation a checkout on another platform
  would produce a different hash and the failure would look exactly like
  tampering.
- `freeze.py` is excluded from its own hash — it holds the hash, so including
  it would be circular.
- The test verifies each of the three modules contributes to the hash by
  hashing modified *copies* in a tmp directory, so a failing run can never
  leave the working tree edited.

**Open:**
- **The freeze is not yet complete.** The fitted parameters in
  `scripts/validate_simulator.py` — mandate amount distribution and
  presentment-window share — materially shape the observed failure mix but sit
  outside the hash, because they are experiment setup rather than simulator
  internals. Changing them changes results without tripping the test. They
  must move into `CalibrationSet` at commit 13, which brings them under the
  hash and closes the gap. This is recorded in `docs/FREEZE.md` too.
- Every calibrated value is still `assumption`. The freeze fixes the world; it
  does not make the world accurate.

---

### Push 1 complete — what exists now

A calibrated, validated, frozen simulator, and nothing else:

- **Types** with a type-enforced observation boundary and integer paise.
- **Calibration** of 22 parameters, every one labelled, every one an
  `assumption` with a `TODO(sumit)` or no-public-source string. No invented
  citations.
- **A latent world**: seeded populations, salary-cycle balances, per-tier bank
  uptime, a 31-day calendar.
- **Attempt resolution** across all six outcomes, with revocation.
- **Diagnostic messiness** that leaves only ~33% of failures cleanly
  diagnosable from the code alone.
- **Validation** showing the emergent failure mix lands within 2 points of the
  calibrated failure rate and 3 points of every mode share, with figures.
- **A freeze** that makes any later change to the world fatal to the suite.

Three problems worth carrying forward into push 2: the calibration is
internally consistent but entirely unsourced; the fitted parameters sit
outside the freeze until commit 13; and no policy, cost model or harness
exists yet, so there is still no number in this repository that says anything
about recovery.

---

## Commit 9 — Intervention cost model

**Done:** `src/mandate_recovery/costs.py` charges three costs against a
completed `Episode`: gateway fees per attempt, per-message and per-call
contact costs, and expected churn cost (contacts x calibrated increment,
multiplied by the mandate's remaining lifetime value). It sets the
`over_intervention` flag when a contacted customer would have paid anyway on
the paired counterfactual, and exposes `net_recovery_paise` as the headline
metric. `tests/test_costs.py` adds 21 tests; suite is 158 green.

**Decisions:**
- **Churn charges only the risk the policy created**, never the customer's
  latent churn intent. Charging latent churn would make cost depend on
  something no policy can see and would punish arms that happened to draw
  unhappy customers. Two identical episodes cost the same whoever the customer
  is, and a test pins that.
- Escalation counts as a contact for churn but carries no separate monetary
  charge, because no agent-time figure is calibrated. This under-costs the
  most expensive real-world action; noted below.
- A silent retry is never over-intervention. It costs gateway fees but the
  customer never knows it happened, so it cannot be an intrusion — only
  contacts and escalations can.
- Churn cost is the one term computed in float. It is an expectation over a
  probability, so it is rounded once at the end before re-entering the money
  path, and every field on `CostBreakdown` is asserted to be an `int`.
- `remaining_cycles` makes churn cost proportional to what is actually left to
  lose. Losing a customer on their final cycle is cheap; losing one with two
  years to run is not.

**Open:**
- **Human-agent cost is uncalibrated**, so `EscalateHuman` is charged only
  through churn. That makes escalation look cheaper than it is and could bias
  a policy toward it. Needs an agent-minutes figure in `CalibrationSet`.
- `would_have_paid_without_intervention` is an input here; nothing computes it
  yet. The paired counterfactual arrives with the harness at commit 13, and
  until then `over_intervention` is structurally correct but never true in a
  real run.
- Costs are per-episode. Portfolio effects — a customer with several mandates
  contacted once — are not modelled.

---

## Commit 10 — Policy interface with a type-enforced boundary

**Done:** `policies/base.py` defines `Policy.decide(observation) -> Decision`
and auto-registers every subclass in `POLICY_REGISTRY` on creation.
`policies/do_nothing.py` is the no-intervention floor. `tests/policies/
test_base.py` walks the registry and fails if any policy's `decide` **or**
`__init__` mentions a simulator type. 14 tests; suite is 172 green.

Mutation-checked: giving `DoNothingPolicy` a `World` constructor argument
fails with "DoNothingPolicy.__init__ accepts a simulator type". Restored
byte-identical.

**Decisions:**
- **The oracle's exemption is a class attribute, not a line in the test.**
  `Policy.reads_latent_state` defaults to `False`; commit 12 sets it `True` on
  the oracle alone. The build plan called for editing this test at commit 12 —
  declaring it on the class instead means the exemption is visible in the code
  that uses it, and commit 12 needs no test edit. A second test requires any
  exempt policy to call itself an upper-bound instrument in its own docstring,
  so the flag cannot be set quietly.
- The constructor is checked as well as `decide`. Taking the `World` in
  `__init__` is the obvious way around a signature check, and the mutation
  above confirms that route is closed.
- `Policy.decision(...)` is the only way policies build a `Decision`, and it
  defaults `validated=False`. A policy cannot mark its own action approved.
- The registry populates via `__init_subclass__` rather than an explicit
  decorator, so a policy cannot be omitted from the boundary test by
  forgetting to register it.

**Open:**
- The check is by annotation, so an unannotated argument would slip through.
  Everything in this codebase is annotated and the `decide` test pins the
  exact parameter list, but a determined author could still reach into
  `mandate_recovery.sim` inside a method body. That is not statically
  catchable here; the freeze hash and code review cover it.
- Nothing calls `decide` in a loop yet — the harness arrives at commit 13.

---

## Commit 11 — Fixed-schedule baseline

**Done:** `policies/fixed_schedule.py` retries at T+1, T+3, T+5 from the first
failure, at a fixed hour on a fixed rail, then stops. No diagnosis, no channel
choice, no timing intelligence. Offsets, hour and rail are all constructor
arguments so the sweep can re-tune it. `tests/policies/test_fixed_schedule.py`
adds 31 tests; suite is 203 green.

**Decisions:**
- **The default retry hour is 09:00, deliberately outside the NPCI restricted
  window.** This is the single most consequential fairness choice in the
  project so far. A baseline retrying at 11:00 would eat a 35% window
  rejection on every attempt and hand the agent a large advantage on a
  decision no competent ops team gets wrong — NPCI's deprioritisation is
  public. The agent has to win on what is actually hard: the customer's cash
  cycle, an ambiguous response code, and whether to make contact at all. A
  test asserts the default hour is outside every calibrated window, with a
  failure message calling a strawman a strawman.
- Offsets are measured from the **first** failure, not the last attempt, so
  T+3 means three days after the original decline. Pinned by a test, because
  the other reading silently stretches the schedule.
- It never schedules into the past. A decision reached late takes the earliest
  slot still available rather than emitting an unreachable day.
- Tests assert it ignores the response code, the amount, and the customer's
  history — proving it is dumb in exactly the ways claimed, so the comparison
  measures intelligence rather than an accidental extra feature.

**Open:**
- T+1/T+3/T+5 is described in merchant recovery guides but is not a published
  standard, and the module says so rather than citing one.
- The baseline never gives up early on a revoked mandate: it cannot read
  codes, so it burns its full schedule on mandates that can never succeed.
  That is realistic and is part of why it should lose on net recovery — worth
  confirming in the commit 15 numbers rather than assuming.

---

## Commit 12 — Perfect-information oracle

**Done:** `policies/oracle.py` is the ceiling arm. It receives the `World`
through its constructor, projects each customer's balance forward, and
schedules the first future hour where the balance covers the amount, the bank
is up and the window is clear — stopping when the amount exceeds the
per-transaction ceiling or no such hour exists within a pay cycle.
`tests/policies/test_oracle.py` adds 20 tests; suite is 223 green.

**Decisions:**
- **No edit to `test_base.py` was needed.** The exemption designed at commit
  10 as `reads_latent_state = True` did its job: the oracle drops out of the
  boundary checks by declaring itself, and a second test asserts it is the
  *only* class in the registry with that flag. The build plan called for
  editing the boundary test; the class attribute is better, because the
  exemption is visible where it is used rather than buried in a test file.
- **`decide` still takes only an `Observation`.** The `World` arrives through
  `__init__`. A test pins this: if the world ever appeared in `decide`, the
  policy interface would have been widened for every policy, not just this one.
- The docstring states in capitals that it is an upper-bound instrument, reads
  simulator-private state, and could not exist in production. A test asserts
  those words are there, so the warning cannot be quietly deleted.
- The search horizon defaults to one pay cycle (31 days). Beyond that the next
  billing cycle supersedes this one, so a "recovery" there is not a recovery.
- The oracle only retries; it never contacts. It therefore bounds **timing**,
  not recovery in general. A policy that nudges a customer into topping up
  could in principle beat it, and the module says that would be a finding
  rather than a bug.
- The balance projection ignores other mandates competing for the same
  balance, making it optimistic — correct for an upper bound.
- The oracle's test file re-implements the balance projection independently
  rather than calling the oracle's own method, so the two agreeing is evidence
  rather than tautology.

**Open:**
- The oracle is optimistic in a second way: it assumes its own retry is the
  only debit. With one mandate per customer that is exact; it will overstate
  the ceiling once a customer can hold several mandates.
- Nothing measures how much of the headroom a policy captures yet — that
  needs the harness and metrics, at commits 13 and 14.

---

## Commit 13 — Paired experiment harness and result storage

**Done:** `harness/runner.py` runs every arm across every seed on **paired
worlds**: the world is built from the seed before any policy exists and
deep-copied per arm. `harness/storage.py` writes
`results/<experiment_id>/` with `config.json` (run config + git SHA +
`SIMULATOR_HASH` + timestamp), `raw/episodes.parquet`,
`raw/decisions.parquet` and `metrics.json`, refusing to overwrite without
`overwrite=True`. `tests/harness/test_runner.py` adds 23 tests; suite is 246
green.

Mutation-checked: replacing the per-arm `deepcopy` with a shared world breaks
six tests including determinism and the per-mandate episode count. The pairing
is load-bearing, not decorative.

Smoke run over 2 seeds x 200 mandates x 90 days gives the ordering the design
predicts — do_nothing < fixed_schedule < oracle on net recovery.

**Decisions:**
- **`pyarrow` was added to `pyproject.toml`, with approval**, so Parquet works
  as the build plan specifies. Round-tripping is tested, including that paise
  columns come back as integers rather than floats.
- **Mandates now recur.** The first draft settled a mandate permanently on its
  first success, so no customer ever accumulated a payment history and
  `max_historical_success_amount_paise` was always zero — which would have
  made the contradiction cases from commit 6 undiagnosable by construction. A
  cycle now opens on each due day and closes on success, stop, or month end. A
  90-day run is about three cycles.
- `Observation.attempt_history` carries **this cycle's** attempts only; earlier
  cycles arrive as the historical counts. Otherwise the fixed-schedule
  baseline would read a three-cycle history as one exhausted schedule.
- The counterfactual runs as a silent extra pass per seed, never reported as
  an arm. A test asserts every arm agrees about it, since it describes the
  world rather than the policy.
- The fitted mandate-amount parameters moved from the validation script into
  `ExperimentConfig`, so they land in every stored `config.json`.
- `ASSUMED_MANDATE_LIFETIME_CYCLES = 12` sets the scale of churn cost. A
  modelling choice, not a calibrated figure, and named so it is visible.

**Open:**
- **`SendNudge` closes the cycle without re-presenting the debit.** No arm in
  this milestone sends nudges, so no number here is affected — but the
  heuristic agent at commit 19 will nudge, and under the current harness a
  nudge would silently forfeit the cycle. **Nudge-then-retry must be
  implemented before commit 19** or the agent will be crippled by the harness
  rather than by its own decisions.
- The commit 8 freeze gap is only half closed. The fitted parameters are now
  in the stored config, satisfying reproducibility, but they live in
  `ExperimentConfig` rather than `CalibrationSet`, so they still sit outside
  `SIMULATOR_HASH`. Closing it fully means moving them into the calibration
  and re-freezing — which invalidates the current hash and every stored run.
  Worth doing deliberately, once, before any headline number is published.
- `CollectPartial` and `SwitchRail` are executed as next-day retries without
  modelling a reduced amount or a card rail. Neither is used by any arm yet.
- One mandate per customer in practice. The runner supports more, but
  `remaining_cycles` and the oracle both assume otherwise.

---

## Commit 14 — Metrics and paired arm comparison

**Done:** `harness/metrics.py` gives `compute_metrics` (recovery rate,
recovered and net paise, attempts and contacts per recovery, median and p90
days to recovery, over-intervention rate, cost per Rs 100 recovered) and
`compare_arms`, which returns per-seed paired deltas, the mean delta with a
bootstrap 95% CI, and `loss_rate`. `tests/harness/test_metrics.py` adds 23
tests; suite is 269 green.

Mutation-checked: deriving `loss_rate` from the sign of the mean fails two
tests, and collapsing the bootstrap to a point estimate fails the
interval-width tests.

**Decisions:**
- **The bootstrap resamples per-seed deltas, not episodes.** Episodes inside a
  seed share a world, a bank and a calendar, so they are not independent;
  resampling them would produce an interval far too narrow and a result that
  looked much more certain than it is.
- **`loss_rate` is computed from the seeds themselves, never inferred from the
  mean.** A test pins the case that matters: an arm winning by Rs 100,000 on
  average while losing on half the seeds must still report a 50% loss rate.
  `summarise_comparison` puts the loss rate in the same sentence as the mean,
  so the two cannot be separated when quoted.
- `compare_arms` refuses arms run on different seed sets rather than silently
  comparing the overlap — an unpaired comparison dressed as a paired one is
  exactly the failure this design exists to prevent.
- `recovery_rate` is per **billing cycle**, not per mandate. A mandate that
  runs three cycles and is collected twice recovered two thirds of what it was
  owed; the mandate-level view would score that as a full success.
- Undefined ratios return `None`, not infinity or zero. An arm that recovered
  nothing has no meaningful attempts-per-recovery, and reporting `0.0` there
  would flatter it.
- The bootstrap takes an explicit generator, per the determinism invariant, so
  a published interval is reproducible from the run config.

**Open:**
- The CI is a percentile bootstrap, which is fine for a mean of paired deltas
  but has no bias correction. With 120 seeds that is not worth fixing; if a
  headline claim ever rests on a marginal interval, say so rather than
  switching estimator after seeing the result.
- No multiple-comparison adjustment. Comparing four arms against one control
  inflates the chance one looks good by luck; the loss rate partly covers
  this, but the ablation at commit 26 should state how many comparisons were
  run.

---

## Commit 15 — Baseline experiment, 120 seeds, bounds established

**Done:** `scripts/run_baselines.py` runs the three non-AI arms across 120
paired seeds x 500 mandates x 90 days in ~34 seconds, and writes
`results/baselines/` with config, metrics, both parquet tables and three
figures. README now carries a Results section with the bounds figure and the
headroom stated in numbers. Suite is 269 green.

**The first real numbers:**

| Arm | Net recovery | Recovery rate | Attempts/recovery | Cost per Rs 100 |
| --- | ---: | ---: | ---: | ---: |
| do_nothing | Rs 11,866 L | 73.1% | 1.37 | 0.029 |
| fixed_schedule | Rs 13,641 L | 81.3% | 2.06 | 0.043 |
| oracle | Rs 15,409 L | 84.8% | 1.38 | 0.026 |

Headroom above the fixed schedule is **Rs 1,768 lakh**; the fixed schedule
captures **50.1%** of what is available above doing nothing. All three
comparisons lost on **0 of 120 seeds**.

**Decisions:**
- `figures.py` gained `recovery_bounds`, `arm_comparison_bars` and
  `paired_delta_distribution`, outside the task's file list. The project
  convention set at commit 7 is that figure functions live in the shared
  module with consistent style; putting them in the script would have split
  the styling.
- The three comparisons are declared as a `COMPARISONS` constant and their
  count is written into `metrics.json`. Stating up front how many comparisons
  were run is what stops a later reader wondering how many were tried.
- The bootstrap seed is a module constant, so a published interval is
  reproducible rather than merely reproducible-in-principle.
- The headroom annotation on the bounds figure was repositioned after
  inspecting the render: it originally collided with the title.

**Open:**
- **Intervention costs barely bite.** At 0.043 rupees per Rs 100 recovered,
  gateway fees are ~0.04% of a mandate's value, so nothing in the cost model
  currently discourages retrying. That is realistic — real gateway fees are
  small — but it means "retry forever" is not stopped by economics. It will be
  stopped by the compliance validator at commit 17, and the write-up should
  say so rather than implying the cost model does the work.
- Over-intervention is 0.0% across every arm because no arm contacts anyone.
  The metric is structurally correct but untested against a policy that
  actually nudges; the heuristic agent at commit 19 is the first real exercise
  of it.
- The oracle bounds timing only. It is a ceiling on retiming, not on recovery,
  and the README says so — but if the LLM agent ever exceeds it, that is a
  finding about persuasion, not a bug.
- Loss rate is 0/120 everywhere, which makes it uninformative so far. It will
  start earning its place once arms are close.

---

### Milestone 2 complete — first real numbers

Cost model, policy interface with a type-enforced boundary, three non-AI arms,
a paired experiment harness, paired metrics with bootstrap CIs and loss rate,
and a 120-seed baseline run with figures. 269 tests green.

Carried into milestone 3: the harness forfeits a cycle on `SendNudge` and must
support nudge-then-retry before commit 19; the freeze still excludes the
fitted mandate parameters; and every calibrated number remains an unsourced
placeholder, so no figure here is quotable as a claim about the world.

---

## Commit 16 — Constraint-aware retry scheduler

**Done:** `agent/scheduler.py` scores candidate (day, hour) slots on four
terms — funds-present likelihood, an inferred bank-availability prior, an hour
preference, and a cooling-off penalty — with the NPCI restricted window as a
hard exclusion, and returns the argmax with a rationale written for a human.
`tests/agent/test_scheduler.py` adds 27 tests; suite is 296 green.

**Decisions:**
- **The agent's priors live in the agent package and are deliberately wrong.**
  `ASSUMED_TIER_AVAILABILITY` (0.95/0.90/0.85) is rounder than the calibrated
  truth (0.955/0.910/0.865), and `SALARY_DAY_PRIOR` is a folk belief about
  Indian payroll rather than the calibrated distribution. A test asserts the
  agent's bank priors never equal the simulator's, because if they did the
  agent would be reading ground truth through a side door.
- `scheduler.py` imports nothing from `mandate_recovery.sim`, and a test reads
  the source to prove it. The one place the agent's code book is checked
  against the simulator's vocabulary is a *test*, which keeps them in sync
  without letting the agent import truth.
- The restricted window arrives through `SchedulerConstraints` rather than
  being read from calibration. NPCI publishes its peak-hour policy, so a
  merchant legitimately knows it — but passing it in keeps the dependency
  visible and lets the sweep vary it.
- **Bug found and fixed while testing:** an observed payday was boosted
  *multiplicatively* off a mid-month day's tiny prior, which left it far below
  the month-end cluster — so "this customer actually pays us on the 15th"
  could never overturn the population average, defeating the point of having
  the signal. Observed days now take an absolute mass floor.
- The hour preference favours 04:00-09:00: salary credits land overnight and
  spending follows, so an early re-presentment catches the balance before the
  day eats it. A belief, not a measurement, and the cheapest lever available.
- `next_retry_slot` returns a `RetrySlot` rather than a bare `(day, hour)`
  tuple, because the rationale has to travel with the choice into the audit
  trail. `as_tuple()` gives the plain pair.

**Open:**
- `Observation` gained `successful_days_of_month`, outside the task's file
  list, because the specified salary prior adjusts on it and there was no
  observable route to that fact. It is legitimately observable — a merchant
  knows which days they have been paid on — and the disjointness test against
  `LatentCustomerState` still passes. Nothing populates it yet; the harness
  wiring lands with the heuristic agent.
- The scheduler has no hourly bank-availability prior, only a per-tier daily
  one, so hour choice is driven entirely by the funds heuristic.
- Priors are unswept. If the agent's advantage turns out to rest on the
  payday folk prior being roughly right, the sweep must say so.

---

## Commit 17 — Compliance validator and stopping rules

**Done:** `agent/validator.py` gates every action. Seven rules: attempt cap
per cycle, minimum hours between attempts, contact cap per rolling 7 days,
contact hours 09:00-21:00, cumulative cost budget, no silent retry inside the
restricted window, and no card rail without a card on file. Rejections and
substitutions are counted by rule. `tests/agent/test_validator.py` adds 42
tests; suite is 338 green.

**Decisions:**
- **Two rules correct rather than refuse.** A retry landing an hour too early,
  or inside the NPCI window, is a timing error with an obviously right answer;
  refusing it would throw away a recovery to punish a rounding mistake. Every
  other rule refuses outright and substitutes `Stop` — there is no safe
  correction for "you have contacted this customer three times this week".
- `ValidationResult.action` always holds the executable action, so a caller
  that runs `result.action` is compliant whether the verdict was approval,
  correction, or refusal. That removes the failure mode where a caller checks
  `approved` and then executes the original anyway.
- **Bug found in testing:** when both corrections fired on one action —
  pushing a retry forward landed it inside the restricted window — only the
  last rule was counted, so `min_gap` substitutions silently under-reported.
  Both are now counted and `rule` records the combination.
- Stopping is always permitted, unconditionally, ahead of every other check.
- Three `Observation` fields were added outside the task's file list, each
  required by a specified rule and each legitimately observable: `current_hour`
  (contact-hours rule), `contacts_in_last_7_days` (the cap is written against a
  rolling window that cumulative `contacts_sent` cannot express), and
  `has_card_on_file` (the card rail rule). Disjointness from
  `LatentCustomerState` still holds.

**Open:**
- **Every limit in `ComplianceLimits` is an author's assumption.** NPCI does
  cap re-presentment and Indian telecom rules do restrict commercial contact
  hours, but the specific numbers here come from no circular and the module
  says so. They need the same `TODO(sumit)` treatment as the calibration
  before any compliance claim is made.
- The cost budget is not the binding constraint: at 2,000 paise it permits ten
  gateway attempts against an attempt cap of four. The attempt cap binds
  first, which is the intended ordering, but it means the budget rule is
  currently untested against real pressure.
- Nothing populates the three new observation fields yet, and nothing calls
  the validator in a run. Both land with the heuristic agent at commit 19.

---

## Commit 18 — Decision audit trail

**Done:** `agent/audit.py` records every decision with the observation it was
made on (fingerprint plus key fields), the diagnosis, the proposed action, the
decision source, the rationale, the validator's verdict, what actually
executed, the outcome and the running cost. `to_dataframe()` gives the tabular
form; `to_human_readable(mandate_id)` renders one mandate's whole story as
text. `tests/agent/test_audit.py` adds 19 tests; suite is 357 green.

**Decisions:**
- **Entries are stamped with simulation day and hour, never wall-clock time.**
  A trail carrying `datetime.now()` could not be reproduced from a stored
  config, and invariant 4 says every experiment must be. The real clock is the
  one thing about a run that cannot be replayed. A test asserts no wall-clock
  field exists.
- **`record()` raises on a blank rationale.** An action nobody can explain is
  an action that should not have been taken, so the log refuses to hold one
  rather than storing an empty string and hoping someone notices.
- The rendered trail shows *both* the proposed and the executed action when
  they differ, and prints REFUSED in capitals with the rule that fired. The
  place a reader most needs the truth is exactly where a policy was overruled.
- Rupees, not paise, in the rendered output — and a test asserts the raw paise
  integer does not appear. Nobody reads paise off a screen.
- Rationales and validator reasons are wrapped to stay inside a terminal. This
  output is meant to be screen-recorded; a line that wraps at the terminal
  edge is a line that reads badly on video.
- The observation fingerprint is a 16-character SHA256 prefix, so two
  decisions can be confirmed to have seen identical inputs without storing the
  whole object on every row.

**Open:**
- Nothing writes to the log during a run yet. Wiring it into the harness lands
  with the heuristic agent at commit 19, and the API endpoint that serves it
  at commit 28.
- The trail is per-run and in memory. A 120-seed x 500-mandate run produces
  hundreds of thousands of decisions; if that becomes a problem it should be
  written incrementally rather than held.

---

## Commit 19 — Deterministic heuristic agent

**Done:** `policies/heuristic.py` runs diagnose (rules) -> select (decision
table) -> schedule -> validate. Diagnosis is a code-book lookup plus the two
disambiguation rules for the contradiction case, returning `UNKNOWN` rather
than guessing, and exposing `unknown_diagnosis_rate`.
`tests/policies/test_heuristic.py` adds 32 tests; suite is 392 green.

**Decisions:**
- **The commit 13 blocker is fixed.** `runner.py` no longer forfeits a cycle
  on `SendNudge`: a nudge asks the customer to fund the account, so the debit
  is re-presented `nudge_followup_days` later. Without this the heuristic
  would have been crippled by the harness rather than by its own decisions,
  and the ablation would have measured the bug.
- **Milestone 2's numbers were re-verified after the harness changes and
  reproduce to the paise** — do_nothing 118,661,910,531, fixed_schedule
  136,410,729,005, oracle 154,091,675,537, all identical to the stored run.
  Card ownership is drawn on a dedicated `[seed, 5]` stream precisely so that
  adding it could not perturb any pre-existing sequence.
- The two contradiction rules are asymmetric on purpose. If the customer has
  previously paid **at or above** this amount, their ceiling permits it and a
  funds code is trusted. If the largest amount they have ever paid is
  **below** it, a limit breach would look identical, so the answer is
  `UNKNOWN` rather than a coin flip. That second rule is the only handle on
  the miscoded-limit case from commit 6.
- The agent keeps its own code book and imports nothing from
  `mandate_recovery.sim`. It does read the calibrated **cost** figures, which
  is legitimate — a merchant knows their own gateway fees — but nothing about
  failure rates, balances or the customer.
- Contact is not the first move: two silent retries must fail first, the
  customer must not have been contacted, and more than two days must remain in
  the cycle. Nudging early would buy recovery with churn.
- `EscalateHuman` is deliberately unused. Every trigger I could write for it
  needed an invented disputable threshold, and the action remains available
  for the LLM arm.
- **Two test bugs found and fixed:** the import check matched the module
  docstring's own mention of the simulator, so it now reads the AST; and a
  lapse assertion checked the wrong string.

**Open:**
- The heuristic's UNKNOWN rate under a real run is not yet measured — that is
  commit 20, and it is the number that justifies the LLM stage.
- `nudge_followup_days = 1` is a harness convention, not a policy choice. A
  policy cannot yet say *when* to follow up a nudge, because `SendNudge`
  carries no timing. If the LLM arm wants to control that, the action needs a
  field.
- The decision table thresholds (two retries before contact, two days before
  lapse) are unswept assumptions like everything else.

---

## Commit 20 — Heuristic experiment, 120 seeds

**Done:** `scripts/run_heuristic.py` runs all four non-LLM arms on the same
120 seeds as commit 15 and writes `results/heuristic/` with config, metrics,
both parquet tables and four figures. README Results updated with the
heuristic's position and the UNKNOWN rate. Suite is 394 green.

| Arm | Net | Recovery | Att/rec | Contacts/rec | Over-int | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| do_nothing | Rs 11,866 L | 73.1% | 1.37 | 0.000 | 0.0% | 0% |
| fixed_schedule | Rs 13,641 L | 81.3% | 2.06 | 0.000 | 0.0% | 50.1% |
| heuristic | Rs 13,623 L | 82.4% | 1.78 | 0.082 | 8.4% | 49.6% |
| oracle | Rs 15,409 L | 84.8% | 1.38 | 0.000 | 0.0% | 100% |

**The heuristic does not beat the fixed schedule.** Mean delta Rs -15,265
(95% CI Rs -52,047 to +22,500), lost on **63 of 120 seeds (52.5%)**. UNKNOWN
diagnosis rate: **23.0%** over 112,063 failures.

**A defect the experiment found, and the fix:**
The first run showed `contacts_per_recovery = 0.000` and 20,366 validator
refusals for `outside_contact_hours`. The contact path was **entirely dead**:
the scheduler retries at 04:00-06:00 to catch balances before the day's
spending, so every nudge decision happened outside the 09:00-21:00 window and
was refused. A real collector that notices a failure at 05:00 does not wake
the customer — it queues the message for business hours. `SendNudge` now
carries `send_hour`, and the validator **defers** an early nudge rather than
refusing it, consistent with the existing rule that timing errors are
corrected and substantive violations are refused. An escalation is never
deferred: a human picking up a phone is not a queued message.

**Both measurements are recorded, because the fix changed the answer:**

| Heuristic | Net | vs baseline | Loss rate |
| --- | ---: | ---: | ---: |
| contact path dead (first run) | Rs 14,011.5 L | +Rs 308,679 | 4.2% |
| contact path working (reported) | Rs 13,622.8 L | -Rs 15,265 | 52.5% |

**Decisions:**
- **The losing result is reported as measured. It was not tuned away.** I
  could have made the contact rule cost-aware until the number went green;
  that is exactly the search this project exists not to do. The mechanism is
  understood and written down instead.
- Why contact loses: a nudge raises churn probability by the calibrated
  increment, which on an Rs 8,800 mandate with ~9 cycles left costs about
  Rs 1,188 in expected lifetime value — 13.5% of the mandate — to buy roughly
  one percentage point of recovery. It also **displaces a salary-timed retry**
  with an untimed follow-up at the default hour, so the agent pays the churn
  and forfeits its main edge at the same time.
- Over-intervention is finally a live measurement rather than a structural
  zero: 8.4% of the heuristic's episodes contacted someone the paired
  counterfactual shows would have paid anyway.

**Open:**
- **The nudge follow-up ignores the scheduler.** `nudge_followup_days` puts
  the retry a fixed day later at the default presentment hour, discarding the
  slot the scheduler would have chosen. This is the single biggest reason the
  contact path loses money, and it is a harness limitation rather than a
  policy choice. **Fix before commit 26**, or the ablation will measure the
  harness rather than the agent.
- The contact rule is not cost-aware. A competent agent would weigh the churn
  cost it can compute against the amount at stake. Doing that is legitimate
  and should happen — but as a stated design change with both numbers
  reported, exactly as above, never as a quiet retune after seeing the result.
- 16,026 `attempt_cap_reached` refusals means the compliance cap now binds
  hard on the heuristic. Worth checking whether the cap or the agent is the
  limiting factor before the ablation.

---

### Milestone 3 complete

Scheduler, compliance validator, audit trail, deterministic agent, and a
120-seed experiment. 394 tests green.

The headline finding is not the one the plan expected: **the deterministic
agent ties the industry baseline rather than beating it**, because the cost of
contacting customers eats the value of better timing. The 23.0% UNKNOWN
diagnosis rate is the argument for the LLM stage; the contact economics are
the argument against using it to talk to people. Both go into the ablation.

---

## Commit 21 — Cached temperature-0 model client

**Done:** `llm/client.py` wraps Gemini with temperature 0, a declared pydantic
response schema on every call, two retries with exponential backoff, and
`LLMFallback` for callers to catch. `llm/cache.py` is an on-disk cache keyed on
`(provider, model, prompt, schema)`, committed to git so results reproduce
without a key. `tests/llm/test_client.py` adds 27 tests, none of which touch
the network. Suite is 421 green.

**Decisions:**
- **`gemini-2.5-flash`, pinned, never an alias.** Verified live: it returns
  byte-identical output for a repeated prompt at temperature 0. The faster
  flash-lite models were **not** deterministic on the same check, and
  `gemini-flash-latest` is a moving target that would silently change the
  model under a stored result. Two newer models returned 503 under load.
- **Latency forces the cache to be load-bearing.** A call takes ~4.9s; the
  heuristic run produced 112,063 failures of which 23% are residual, so a
  naive implementation is ~35 hours per experiment. Prompts must therefore be
  *canonical and bucketed* so thousands of observations collapse onto a few
  hundred distinct prompts. That constraint is designed in from here, not
  retrofitted at commit 26.
- The provider is part of the cache key. This project switched providers
  mid-build; without it a switch would silently mix two models' answers while
  still looking reproducible.
- A cached reply that no longer validates is treated as **stale and refetched**,
  not as a schema failure. Schemas change during development; the cache
  should not poison a run because of it.
- `offline=True` serves only from cache and raises `LLMFallback` on a miss.
  That is the mode `make reproduce` will use, so a reviewer with no key gets
  either the recorded answer or a loud failure — never a silent live call.
- Cache entries store the prompt alongside the response. A reviewer can read
  what was asked without re-deriving it from a hash.
- **The client does not disable certificate verification.** TLS-intercepting
  security software on this machine breaks `google-genai`, which verifies
  against certifi rather than the OS trust store. The fix is `SSL_CERT_FILE`
  pointing at a bundle that includes the intercepting root — an environment
  problem. Silently turning off verification in a payments codebase would be
  worse than failing.

**Open:**
- The cache is empty. Warming it is part of commit 26, and the warming cost
  is bounded by how well the bucketing works — which commit 22 has to get
  right.
- `total_tokens` is recorded but no cost-per-decision figure is derived yet.

---

## Commit 22 — Residual-routed diagnosis

**Done:** `llm/diagnosis.py` runs the rule-based code book first and calls the
model **only** on what it cannot resolve: generic codes, missing codes, and
the contradiction case. `DiagnosisRouter.stats()` exposes
`llm_invocation_rate` as a first-class metric. The prompt lives in
`src/mandate_recovery/prompts/diagnosis.md` as a versioned file.
`tests/llm/test_diagnosis.py` adds 23 tests; suite is 444 green.

**Decisions:**
- **The prompt is canonical and bucketed, and this is load-bearing.** Measured
  over the full feature grid: 3,024 routed observations collapse onto **504
  distinct prompts**, a 6x reduction, warming once in about 41 minutes and
  free thereafter. Raw amounts would have made the cache useless and a
  120-seed experiment a 35-hour job. `amount_vs_history` renders as "somewhat
  larger than anything they have paid before" rather than "880000 against
  500000" — and a test asserts two observations differing only in scale
  produce the *same* prompt while the cases the diagnosis turns on still
  produce *different* ones.
- **The leak test checks latent values, not latent words.** The prompt
  deliberately tells the model "you do not have the customer's bank balance,
  their salary date, or their account limit" — good prompt design, and my
  first test flagged it as a leak. It now builds observations from a real
  seeded `World` and asserts no large latent number (balance, salary amount,
  spend rate, ceiling) appears in the rendered string. Salary *day* is
  excluded from the check because 1-31 collides with ordinary prose.
- Below 0.55 confidence the model's answer is **discarded** and the failure is
  treated as undiagnosed, with the discarded answer still written into the
  rationale so a reader can see what was rejected and why.
- An `LLMFallback` returns UNKNOWN from source `"fallback"`, distinct from a
  low-confidence `"llm"` answer. The ablation needs to tell "the model was
  unsure" apart from "the model was unreachable".

**Open:**
- Prompt `.md` files are not declared as package data, so a non-editable
  install would not ship them. Fine for how this project runs; needs a
  `pyproject.toml` entry before anyone installs a wheel.
- 504 is the grid maximum. The real distinct-prompt count will be lower and
  is measured at commit 26.

---

## Commit 23 — Intervention proposal with deterministic execution

**Done:** `llm/intervention.py` asks the model for a *kind* of action and a
*rough* timing preference, then deterministic code computes the slot, the
amount and the rail, and the validator approves or refuses. Prompt versioned
at `prompts/intervention.md`. `tests/llm/test_intervention.py` adds 23 tests;
suite is 467 green.

**Decisions:**
- **The reply schema is the enforcement.** `InterventionReply` has exactly
  four fields — `action`, `timing`, `tone_level`, `reasoning` — and a test
  asserts no `amount`, `day`, `hour`, `rail` or `slot` can appear. The model
  literally cannot name a rupee figure or a time slot, so invariant 6 is a
  property of the type rather than a rule someone has to remember.
- A partial collection's amount is the **largest sum this customer has
  actually settled**, computed from the observation, never proposed. It is the
  only figure with evidence behind it.
- The timing preference becomes a *search window* for the existing scheduler
  (`SOON` -> 3 days, `AFTER_NEXT_SALARY` -> 16), so the payday prior still
  does the work and the restricted window is still a hard exclusion.
- The prompt states the churn economics explicitly — a contact costs roughly a
  tenth of the mandate's remaining value, a silent retry costs almost nothing.
  Without that the model nudges constantly, which commit 20 showed is
  value-destroying.
- `ProposedIntervention` keeps both the proposed and the executed action, so
  the audit trail can show a refusal rather than only its outcome.
- This arrangement is also the honest answer to prompt injection: a reply
  saying "collect ten lakh immediately" parses into an enum member and meets a
  function that cannot be argued with.

**Open:**
- The model never sees that a nudge will be *deferred* to 09:00; it proposes
  and the validator queues it. Fine today, but if tone or channel ever depend
  on send time the prompt needs to say so.
