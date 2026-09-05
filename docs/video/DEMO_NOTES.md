# Demo notes

Everything needed to reproduce the on-screen material, with the exact seeds
and commands. All of it is deterministic — retake as often as you like, the
output will not move.

## The replay

```bash
python docs/video/demo_audit_trail.py
```

**Seed 7, mandate `m000147`**, heuristic agent, 200 mandates over 90 days.
Rs 32,412.62 due on the 17th. Five decisions, no compliance refusals, ends
unrecovered.

Verified identical across repeated runs. Other candidates:

```bash
python docs/video/demo_audit_trail.py --list      # ranked alternatives
python docs/video/demo_audit_trail.py m000060     # a specific one
```

`m000060`, `m000046` and `m000025` score equally and are reasonable
substitutes if you want a different shape of story.

### The two moments to pause on

1. **Decision 2** — `Contact deferred from 04:00 to 09:00, the first hour
   customers may be contacted.` The agent wanted to act at 4am; the
   compliance validator queued it. This is the clearest single line showing a
   deterministic gate overruling the decision layer.
2. **Decision 3** — `The bank returned 'PS-51'... the funds reading is taken
   at face value but is not confirmed.` The agent identifies the bank from its
   code vocabulary and *still* declines to be confident, because the customer
   has never settled an amount it could compare against. This is the
   contradiction case from `docs/MESSINESS.md` being handled honestly.

### Terminal setup

- Width **at least 80 columns** — the trail wraps at 72 and the box rules
  assume it.
- Large font, dark-on-light reads better than light-on-dark for the rule
  lines.
- Run from the repository root; the script resolves paths relative to itself
  but the output is cleaner from the root.
- It takes about 20 seconds to replay the seed before printing. **Start
  recording after the output appears**, or cut the pause.

## Regenerating every figure

```bash
python scripts/reproduce_all.py       # ~9.5 minutes, no API key needed
```

Then re-copy the curated set:

| Curated file | Source |
| --- | --- |
| `1_recovery_bounds.png` | `results/heuristic/figures/all_arms_bounds.png` |
| `2_simulator_validation.png` | `results/simulator_validation/figures/observed_vs_calibrated.png` |
| `3_restricted_window.png` | `results/simulator_validation/figures/failure_rate_by_hour.png` |
| `4_diagnosis_coverage.png` | `results/heuristic/figures/diagnosis_coverage.png` |
| `5_paired_deltas.png` | `results/heuristic/figures/heuristic_vs_baseline_paired.png` |
| `6_advantage_survival.png` | `results/sensitivity/figures/advantage_survival.png` |

Every figure also exists as SVG next to its PNG, if you want to scale one up
for a full-screen shot without it going soft.

## Numbers you may be asked for

| Question | Answer |
| --- | --- |
| How many tests? | 528 |
| How many seeds? | 120 paired, 500 mandates each, 90 days |
| Simulator validation | 29.2% observed failure rate vs a 30% target, inside a 2-point tolerance |
| Agent vs baseline | Rs −15,265 mean, 95% CI −52,047 to +22,500, lost on 63/120 seeds |
| Undiagnosable failures | 23.0% return UNKNOWN from the code book |
| Sensitivity | advantage holds in 20/54 regimes; 40/54 CIs exclude zero |
| Reproduction time | 9m27s from a clean clone, verified |
| Published figures used | **none** — every parameter is a labelled assumption |

## Caveats to hold in mind while recording

- **Do not use figure 3 to claim the agent times around the NPCI window.** The
  figure is real — the null world does show a failure spike inside the window
  — but the sensitivity sweep proved that *neither arm ever presents a debit
  inside it*, so it plays no part in the agent's behaviour. Figure 3 is
  evidence the simulator works, not evidence the agent is clever. Using it the
  other way would be the single most misleading thing available in this repo.
- The ablation has not run. If asked about the LLM results, the answer is
  "built, tested, unmeasured — blocked on API quota, and I discarded the
  partial run rather than report it."
- Recovery rates are per billing cycle, not per mandate.
- Every rupee figure on screen is simulated money in an invented world.
