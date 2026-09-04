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
