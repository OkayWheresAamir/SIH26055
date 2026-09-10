# Iteration ledger

One row per training run: what launched it, what it produced, where it stands. Requested as item
6 of the 2026-09-10 review directive (see `docs/project/DECISIONS.md` D60–D63) and built here for
the first time, alongside D64.

**Scope.** Starts at D60 (`36405b1`, 2026-09-10) — the first commit any run in this ledger could
possibly postdate cleanly. Every run before that commit trained on `EmitterPool.from_train()`, the
leaking pool, regardless of which reward or hyperparameters it used; those runs are narrated in
`scratch/TRAINING_JOURNEY.md` §§1–13 and are not re-tabulated here, because a ledger's job is to
make re-running and comparing entries trustworthy, and none of them can be compared to anything
made after the split without the same caveat repeated on every row.

**Columns.** *Run* — the `--run-name` / manifest stem. *Reward* — the candidate trained on.
*Split* — always "D60" for anything in this ledger (the training-half pool); recorded anyway
because a future entry that trains against `EmitterPool.from_train()` directly, by mistake or on
purpose, needs to stand out in this column, not hide in the reward one. *Hyperparameters* — the
`--hyperparam` overrides; blank means SB3 defaults. *Validation (D61)* — net dominance of the
selected checkpoint against rung 5 on the 12 validation configs; "not yet selected" if
`rfenv.selection` hasn't been run on the run's checkpoints. *Headline* — the `compare.py` full-
ladder result for the selected checkpoint; "not yet run" until `EVALUATION.md` §5 has a row for it.
*Outcome* — one line, including for a run that produced nothing usable.

**To re-run any row exactly**, read the `command` field out of the checkpoint's own manifest JSON
(`runs/checkpoints/<name>.json`) and run it verbatim — every manifest records its full argv,
resolved hyperparameters, git commit and package versions at launch, which is more than this table
carries per row. The `Command` column below is the same string, kept here so the ledger is
self-contained without opening a JSON file for the common case.

---

## `clean_lstm` — the control arm, `reward_balance`

| | |
|---|---|
| **Reward** | `reward_balance` |
| **Split** | D60 (`training_pool()`, 35 configs / 1,431 emitters) |
| **Hyperparameters** | `ent_coef=0.01 gamma=0.997 n_steps=8192` (else SB3 `RecurrentPPO` defaults) |
| **Seed** | 0 |
| **Timesteps** | 400,000, checkpointed every 100,000 |
| **Started** | 2026-09-10T15:47:46Z, commit `176e6a9` (post-D60, post-D62/D63) |
| **Wall clock** | 4,419 s (~74 min) for the full 400k |
| **Manifests** | `runs/checkpoints/clean_lstm_s{1,2,3,4}.json` |
| **Validation (D61)** | `clean_lstm_s4` (400k) selected, **net dominance +36.1%** (47.2% dominates / 11.1% dominated, 36 paired episodes). Full table in D64. |
| **Headline** | **Run 2026-09-10** (`runs/clean_paired_comparison/`, rung `10d`): paired against recency, **ratio 73.7%, cTTI 31.6%, both 22.8%** (684-episode run, shared with the treatment arm below). Full table in D65. |
| **Outcome** | First run in this repository clean on both axes (D60 pool, D61 selection). Selected checkpoint is `runs/checkpoints/clean_lstm_s4.zip`, registered as rung `10d`. |

```
venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo --reward reward_balance --timesteps 400000 \
  --seed 0 --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
  --checkpoint runs/checkpoints/lstm_balance_clean.zip --checkpoint-freq 100000 \
  --run-name clean_lstm --description "attempt 1 under D60 split, D61 selection"
```

## `lstm_balance_improved` — the treatment arm, `reward_balance_improved`

| | |
|---|---|
| **Reward** | `reward_balance_improved` (built after D62/D63; not yet screened or ratified as a D47 candidate) |
| **Split** | D60 (`training_pool()`) |
| **Hyperparameters** | `ent_coef=0.01 gamma=0.997 n_steps=8192` — identical to `clean_lstm`, the only deliberate difference is the reward |
| **Seed** | 0 |
| **Timesteps** | 400,000 (target), checkpointed every 100,000 |
| **Started** | 2026-09-10T17:16:32Z, commit `176e6a9` (same commit as `clean_lstm`) |
| **Manifests** | `runs/checkpoints/lstm_balance_improved_s{1,2,3,4}.json`, all four landed |
| **Validation (D61)** | `lstm_balance_improved_s2` (200k) selected, **net dominance +25.0%** (36.1% dominates / 11.1% dominated). Below the control's +36.1%. Full table in D65. |
| **Headline** | **Run 2026-09-10** (`runs/clean_paired_comparison/`, rung `11b`): paired against recency, **ratio 83.0%, cTTI 39.8%, both 31.0%** — ahead of the control's 73.7%/31.6%/22.8% on this measure. **Reverses the validation ranking; see D65 for why this is not read as "the reward wins."** |
| **Outcome** | Complete. Selected checkpoint `runs/checkpoints/lstm_balance_improved_s2.zip`, registered as rung `11b`. Single seed per arm — the headline reversal against validation is the expected size of noise for one training seed, not settled evidence either way. |

```
python -m rfenv.rl.recurrent_ppo --reward reward_balance_improved --timesteps 400000 --seed 0 \
  --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
  --checkpoint runs/checkpoints/lstm_balance_improved.zip --checkpoint-freq 100000 \
  --run-name lstm_balance_improved --description "treatment: density-weighted occupancy, D60 split"
```

---

## Maintaining this file

Add a row when a run under D60 finishes (or is abandoned — a run that crashed, diverged, or was
killed early belongs here with that as its Outcome, not silently omitted). Update the Validation
and Headline cells in place as `rfenv.selection` and `rfenv.compare` are run against a row's
checkpoints; do not create a second row for the same run. A row's Command block must match its
manifest's `command` field exactly — if they'd ever disagree, the manifest wins and this file is
wrong.
