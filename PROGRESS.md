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
