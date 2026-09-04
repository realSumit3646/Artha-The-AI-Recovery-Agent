# Artha — The AI Recovery Agent

## Problem

UPI Autopay recurring mandates fail for reasons the collecting merchant cannot
see: an insufficient balance three days before payday, a bank-side outage, a
customer who has quietly decided to leave. The industry default is a fixed
retry schedule, which spends customer goodwill on payers who were never going
to pay and gives up on payers who would have paid on Friday. Artha is a
research harness that simulates these failures with explicit latent state and
evaluates recovery policies against them — on recovery rate, cost, and
customer-experience damage — under strict information asymmetry: a policy sees
only what a real collector could see, never the simulator's ground truth.

## Status

🚧 **Under construction.** Scaffolding only at this commit — no simulator, no
policies, no metrics yet. See `PROGRESS.md` for the running log.
