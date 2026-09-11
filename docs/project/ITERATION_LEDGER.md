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

## `clean_lstm_seed1`/`seed2`, `lstm_balance_improved_seed1`/`seed2` — the crashed matched-seed attempt

| | |
|---|---|
| **Reward** | `reward_balance` (×2 seeds) and `reward_balance_improved` (×2 seeds) |
| **Split** | D60 (`training_pool()`) |
| **Hyperparameters** | Identical to `clean_lstm`/`lstm_balance_improved`: `ent_coef=0.01 gamma=0.997 n_steps=8192` |
| **Seeds** | 1 and 2, one run per (reward, seed) pair — 4 runs, launched together in the background |
| **Started** | 2026-09-10, on the pre-D67 (146-wide) observation |
| **Manifests** | None — no run reached its first 100k checkpoint before the crash |
| **Validation (D61)** | N/A |
| **Headline** | N/A |
| **Outcome** | **Crashed.** The machine crashed partway through; all 4 runs stopped with no checkpoint saved (`runs/checkpoints/` held no `seed1`/`seed2` files afterward — confirmed, not assumed). Nothing salvageable, nothing lost beyond wall-clock time, since none had passed its first save point. Superseded rather than retried as-was: D67 (below) landed before the retry, changing the observation width these runs would have trained against. Lesson taken: train one model at a time in the background from here on, not several in parallel.

## `lstm_balance_d67_control` — first retrain under D67 (183-wide observation)

| | |
|---|---|
| **Reward** | `reward_balance` |
| **Split** | D60 (`training_pool()`) |
| **Hyperparameters** | `ent_coef=0.01 gamma=0.997 n_steps=8192` — identical to `clean_lstm` (D64) |
| **Seed** | 0 |
| **Timesteps** | 400,000 (target), checkpointed every 100,000 |
| **Observation** | 183-wide (D67) — **not comparable to any pre-D67 checkpoint's numbers directly** |
| **Started** | 2026-09-10, single run, no other training running concurrently |
| **Manifests** | `runs/checkpoints/lstm_balance_d67_control_s{1,2,3,4}.json`, all landed |
| **Validation (D61)** | `lstm_balance_d67_control_s3` (300k) selected, **net dominance +25.0%** (36.1% dominates / 11.1% dominated) — lower than D64's original 146-wide control (+36.1%), not directly comparable (different observation width). |
| **Headline** | **Run 2026-09-10** (`runs/d67_paired_comparison/`, rung `14c`): paired against recency, **ratio 62.0%, cTTI 39.8%, both 25.7%** — a few points above the pre-D67 equivalent's 22.8% (D64/D65), despite no camping behaviour appearing (see Outcome). |
| **Outcome** | Complete. **Streak analysis (3 sample scenarios): no change.** Max dwell streak 4-6 slots, same order of magnitude as every pre-D67 checkpoint (D64/D66 measured max 6). The hoped-for commitment behavior has not appeared on this checkpoint, yet the headline `both` column still moved up a few points against its pre-D67 equivalent — not attributed to the hypothesised mechanism, since that mechanism didn't show up. See D67 for the full accounting alongside the treatment arm. |

```
venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo --reward reward_balance --timesteps 400000 \
  --seed 0 --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
  --checkpoint runs/checkpoints/lstm_balance_d67_control.zip --checkpoint-freq 100000 \
  --run-name lstm_balance_d67_control \
  --description "control: reward_balance, D60 split, D67 observation (183-wide, hit_streak)"
```

## `lstm_balance_d67_treatment` — treatment arm under D67 (183-wide observation)

| | |
|---|---|
| **Reward** | `reward_balance_improved` |
| **Split** | D60 (`training_pool()`) |
| **Hyperparameters** | `ent_coef=0.01 gamma=0.997 n_steps=8192` — identical to control |
| **Seed** | 0 |
| **Timesteps** | 400,000 (target), checkpointed every 100,000 |
| **Observation** | 183-wide (D67) |
| **Started** | 2026-09-10, single run, launched only after the control arm fully finished |
| **Manifests** | `runs/checkpoints/lstm_balance_d67_treatment_s{1,2,3,4}.json`, all landed |
| **Validation (D61)** | `lstm_balance_d67_treatment_s1` (100k) selected, **net dominance +33.3%** (38.9% dominates / 5.6% dominated). Higher than the D67 control's +25.0% -- treatment beats control on validation this time (reversed under D65's 146-wide setup). |
| **Headline** | **Run 2026-09-10** (`runs/d67_paired_comparison/`, rung `15a`): paired against recency, **ratio 77.2%, cTTI 49.7%, both 35.1%** -- ahead of the control's 62.0%/39.8%/25.7% on the same run, and both a few points above their pre-D67 (D65) equivalents (22.8%/31.0%). |
| **Outcome** | Complete. **Streak analysis: no commitment behaviour** -- max dwell streak 4-6 slots on all three sample scenarios, same as every pre-D67 checkpoint. The headline moved up modestly for both arms anyway; not attributed to the hypothesised mechanism since that mechanism didn't appear. See D67 for the full accounting. |

```
venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo --reward reward_balance_improved --timesteps 400000 \
  --seed 0 --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
  --checkpoint runs/checkpoints/lstm_balance_d67_treatment.zip --checkpoint-freq 100000 \
  --run-name lstm_balance_d67_treatment \
  --description "treatment: reward_balance_improved, D60 split, D67 observation (183-wide, hit_streak)"
```

## The D47/D68 matched-seed queue — 4 runs, one at a time

Launched to resolve D68's escalation: `reward_balance` (65.5%) vs `reward_balance_improved`
(64.3%) landed 1.2 pp apart on single-seed-each checkpoints, inside D47's 5 pp no-selection margin.
Adding seeds 1 and 2 to each arm (3 seeds total per reward) to see whether the gap survives being
measured with less noise. Same hyperparameters as the original D67 runs throughout; only `--seed`
and the checkpoint name differ per row.

| run | reward | seed | status |
|---|---|---|---|
| `lstm_balance_d67_control_seed1` | `reward_balance` | 1 | **Complete** 2026-09-11 00:23 — `s1`-`s4` all landed |
| `lstm_balance_d67_control_seed2` | `reward_balance` | 2 | **Complete** 2026-09-11 01:18 — `s1`-`s4` all landed |
| `lstm_balance_d67_treatment_seed1` | `reward_balance_improved` | 1 | **Complete** 2026-09-11 02:18 — `s1`-`s4` all landed |
| `lstm_balance_d67_treatment_seed2` | `reward_balance_improved` | 2 | **Complete** 2026-09-11 04:17 — `s1`-`s4` all landed |

All four ran strictly one at a time, as the standing instruction after the crash requires. The
queue's own cost: roughly 4 hours of wall clock for ~70 minutes of training each, plus the gaps
between one finishing and the next being noticed and launched.

```
venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo --reward <reward> --timesteps 400000 \
  --seed <seed> --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
  --checkpoint runs/checkpoints/<run>.zip --checkpoint-freq 100000 --run-name <run> \
  --description "<control|treatment>: <reward>, D60 split, D67 observation (183-wide), matched-seed for D47/D68"
```

**Resolved.** D61 re-run across all 12 checkpoints per arm: the control pick moved from seed 0's
300k (rung 14c, +25.0% net dominance) to seed 2's 300k (rung 17c, **+36.1%**); the treatment pick
was unchanged (rung 15a, +33.3%, same checkpoint both times). D47 re-applied on the new pair,
paired against round-robin (`runs/d68_rerun_paired_comparison/`): `reward_balance` **81.9%** both,
`reward_balance_improved` **64.3%** both — a 17.6 pp gap, decisively outside the 5 pp margin.
**`reward_balance` selected.** Full accounting in D68 (`DECISIONS.md`); the 16 new checkpoints are
registered as ladder rungs 16a-16d, 17a-17d, 18a-18d, 19a-19d.

A broader comparison — every one of these 16 checkpoints plus both previous best-known
checkpoints (14c, 15a), round-robin and recency, 20 rungs total — is running separately
(`runs/d68_full_matched_seed_comparison/`) to put the complete picture in one table, not just the
two D61 picks. Result to follow once it lands.

---

## Maintaining this file

Add a row when a run under D60 finishes (or is abandoned — a run that crashed, diverged, or was
killed early belongs here with that as its Outcome, not silently omitted). Update the Validation
and Headline cells in place as `rfenv.selection` and `rfenv.compare` are run against a row's
checkpoints; do not create a second row for the same run. A row's Command block must match its
manifest's `command` field exactly — if they'd ever disagree, the manifest wins and this file is
wrong.
