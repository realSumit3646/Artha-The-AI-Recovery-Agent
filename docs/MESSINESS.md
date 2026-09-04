# Diagnostic messiness

## Why this layer exists

The simulator knows the true cause of every failure. If it handed that cause
back as a clean, one-to-one bank code, any policy could invert it with a
fifteen-line dictionary, and every "diagnosis" stage in this project — rules
or model — would be theatre: an elaborate way of reading a label the simulator
had already written. The measured advantage of a diagnosis stage would be an
artefact of the simulator's generosity, not a finding.

Real merchants do not get clean codes. They get a different vocabulary from
every bank, a large bucket of undifferentiated declines, a steady trickle of
blank fields, and occasional codes that are simply wrong. This layer
reproduces that so the inference problem is real.

## What the layer does

`mandate_recovery.sim.response_codes.encode_response(outcome, bank_id, rng)`
applies four effects, to **failures only**:

| Effect | Behaviour |
| --- | --- |
| Bank-specific vocabularies | The same cause reads differently per tier: no-funds is `AB1200`, `PS-51` or `SF_NOFUNDS` depending on who declined |
| Generic bucket | A share of failures return the single uninformative code `DECLINED`, shared by four causes |
| Missing codes | A smaller share return `""` or `NA` |
| Contradictions | A share of limit breaches return the bank's *funds* code instead |

`SUCCESS` and `MANDATE_REVOKED` are reported cleanly and are excluded from all
of the above. Neither is a diagnostic puzzle: a merchant knows whether money
arrived, and knows whether they still hold a mandate.

### The contradiction case is the interesting one

A limit breach that reports a funds code is indistinguishable from a genuine
funds failure *from the code alone*. It can only be caught by noticing that
this customer has previously succeeded at a **higher** amount than the one
that just failed — which is why `Observation` carries
`max_historical_success_amount_paise`. That field is not decoration; it is the
only handle on this class of error, and a diagnosis stage that ignores it will
mis-classify every one of them.

## How messy is it?

`TRUE_CAUSE_RECOVERABLE_FRACTION` records the share of failures whose code
identifies exactly one true cause. A code counts as recoverable only if no
other cause ever emits it, so:

- `DECLINED`, `""` and `NA` are never recoverable — four causes share them.
- A **funds** code is never recoverable, because miscoded limit breaches hide
  inside it.
- Technical, window, and cleanly-coded limit failures are recoverable.

Under the default calibration's failure mix this comes to roughly **0.33**.
`tests/sim/test_response_codes.py` recomputes it empirically from 200,000
samples and asserts it stays **below 0.75**. That ceiling is the design
constraint: if a lookup table could resolve more than three quarters of cases,
the messiness would be too weak and a model-based diagnosis stage would not be
justifiable. The test fails loudly if a future change makes the world too kind.

There is deliberately no lower bound. If the fraction drops, diagnosis gets
harder and the honest consequence is a higher `UNKNOWN` rate, not a broken
experiment.

## Honesty

**Every share in this layer is an author's assumption.** There is no public
dataset of bank response-code ambiguity, no published distribution of generic
versus specific decline codes, and no reported rate at which Indian issuers
miscode a limit breach as a funds failure. None is claimed.

The code strings themselves — `AB1200`, `PS-51`, `SF_NOFUNDS`, `DECLINED` —
are **invented**. They are not real NPCI, UPI or issuer codes, and they are
shaped differently per tier on purpose so that a policy has to learn each
bank's dialect rather than one global table.

The specific values:

| Constant | Value | Basis |
| --- | --- | --- |
| `SHARE_OF_FAILURES_GENERIC` | `0.18` | no public source — author's assumption |
| `SHARE_OF_FAILURES_MISSING` | `0.05` | no public source — author's assumption |
| `SHARE_OF_LIMIT_FAILURES_MISCODED_AS_FUNDS` | `0.25` | no public source — author's assumption |

These are not in `CalibrationSet` because they are not claims about the world
that a published figure could ever settle; they are a modelling choice about
how hard the inference problem should be. They are recorded here, and in the
module, so that a reader can see exactly how much of the diagnosis difficulty
was manufactured. If a result turns out to hinge on them, that dependence
belongs in the sensitivity sweep and in `docs/LIMITATIONS.md`.
