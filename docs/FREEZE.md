# Simulator freeze

**Frozen:** 4 September 2026, at commit 8, before any policy existed.

**`SIMULATOR_HASH`:** `fd5a8fed4eaf5a6d719e9470a2978f93dfd2dccfe0fd2e59de65801b9b31b193`

## Why the freeze exists

A simulator whose parameters can be adjusted after you have seen how a policy
scores is not an experiment. It is a search for flattering settings, and it
will find them — not through dishonesty, but because every knob has a
defensible-sounding reason to move in the direction that helps, and nobody
records the twenty small nudges that produced the final number.

Freezing removes the temptation rather than relying on restraint. From this
commit, any change to the world shows up as a moved hash, fails the test
suite, and demands an explicit re-baseline. That is a deliberately annoying
process, and the annoyance is the feature.

The ordering matters as much as the mechanism: the world was fixed **before**
the first policy was written. Nothing in `world.py`, `outcomes.py` or
`response_codes.py` was chosen with knowledge of how any recovery strategy
would perform against it, because no recovery strategy existed yet. A reader
can check that claim against the git history rather than taking it on trust.

## What the hash covers

| Covered | Why |
| --- | --- |
| Serialised `CalibrationSet` | Every number the simulator uses |
| `sim/world.py` | Balances, salary cycle, bank uptime, the calendar |
| `sim/outcomes.py` | The resolution order and revocation |
| `sim/response_codes.py` | How much diagnostic ambiguity the world contains |

Line endings are normalised to `\n` before hashing, so a Windows checkout with
`core.autocrlf` on produces the same hash as a Linux one. Without that, the
freeze would break for anyone on a different platform and the failure would
look exactly like tampering.

`freeze.py` is not hashed. It holds the hash, so hashing it would be circular.

## What the hash does *not* cover

Stated here rather than discovered later:

- **The fitted parameters in `scripts/validate_simulator.py`** — the mandate
  amount distribution and the share of attempts presented inside the
  restricted window. These materially shape the observed failure mix and sit
  outside the freeze because they are experiment setup rather than simulator
  internals. **They should move into `CalibrationSet` when the harness lands
  at commit 13**, at which point they come under the hash and the freeze
  becomes complete. Until then, changing them changes results without
  tripping this test.
- Policies, the cost model, metrics, and everything else built after this
  commit. Those are supposed to change; that is the entire remaining project.

## If this test fails

The message is in `FREEZE_FAILURE_MESSAGE` and it means what it says:

> Simulator changed after freeze. If this was intentional, ALL experiment arms
> must be re-run from scratch and the re-baseline recorded in PROGRESS.md.
> Update SIMULATOR_HASH only after doing so.

The order is not negotiable. Fix the bug, re-run every arm from scratch, write
the re-baseline into `PROGRESS.md` explaining what changed and which results
were invalidated, and only then update `SIMULATOR_HASH`.

Updating the constant to make the suite green is the one thing that would
render every number in this repository meaningless, and it is the easiest
thing in the world to do at 2 a.m. Do not.
