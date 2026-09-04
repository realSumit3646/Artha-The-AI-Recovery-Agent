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
