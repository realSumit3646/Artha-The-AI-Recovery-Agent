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
