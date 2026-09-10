# Decision draft — the RL lane

Drafted 2026-09-09. `RL_TEAM_HANDOFF.md` §18 format; §16.1's `D-1 … D-13` numbering. **These are
drafts.** Aamir assigns the real `D4x` numbers and merges them into
`docs/project/DECISIONS.md`; nothing here has been written into that file.

**Scope.** Four decisions taken over this lane are already recorded, committed as `ae2b5ff`, and
are **not repeated here**:

| already in `DECISIONS.md` | covers |
|---|---|
| **D48** | RL joins the ladder: rungs 7 (DQN), 8 (PPO), 9 (RecurrentPPO); the two policy adapters; the parameterised-factory pattern |
| **D49** | The observation vector extended past D34's base three — `current_band`, `camp_time`, `measured_dbm`; 109 → 145 → 146 → 147 |
| **D50** | `DEFAULT_REWARD` moved off `hit_z`; `weighted_camp` registered and retired |
| **D51** | `newly` credits per `(emitter, band)` rather than once per episode; measured 2.50×/2.52× |

So **D-14 (observation change) → see D49**, **D-15 (reward rewrite) → see D51 and D50**,
**D-16 (`weighted_camp` retired) → see D50**. This file covers only what those four do not:
hyperparameters, the training command, the seed, inference-time action selection, the
checkpoint-selection criterion, the evaluation configuration, and compute.

**Evidence tags**, per the recovery brief: `[RECOVERED]` — read off a file or a command run in
this session, cited inline. `[RECALLED]` — stated in session but not verifiable against the repo.
`[UNRECOVERABLE]` — the evidence is gone; stated and left blank, never inferred.

---

## D-1 — Algorithm and library, with pinned versions

**Status:** SETTLED (2026-09-09)

**The decision.** Three algorithms, not one: `stable_baselines3.DQN` (rung 7),
`stable_baselines3.PPO` (rung 8), `sb3_contrib.RecurrentPPO` with `MlpLstmPolicy` (rung 9).
`stable-baselines3 2.9.0`, `sb3-contrib 2.9.0`, `torch 2.14.0+cpu`, `gymnasium 1.3.0`,
`numpy 2.5.3`, Python 3.12.10.

**Evidence.** `[RECOVERED]` — read out of every checkpoint's own `system_info.txt` and
`_stable_baselines3_version` members this session; all nineteen agree. **Gymnasium was not
downgraded**: 1.3.0 is what §3.1 flagged as needing a compatibility check, and the suite runs
against it (287 passed, 42 skipped, 1 pre-existing environment-dependent failure).

**Alternatives rejected.** None deliberately — rung 9 was added because a feed-forward policy
conditions each action on one observation alone, and the scan-history signal the task needs is
sequential. That reasoning is D48's, not repeated here.

**How we know.** `python` reading each `.zip`'s JSON `data` member; `pytest tests -q`, this session.

---

## D-2 — Policy network architecture

**Status:** SETTLED (2026-09-09) — by default, not by choice.

**The decision.** The library default for every algorithm. `policy_kwargs == {}` on all nineteen
checkpoints: no `net_arch` override, no activation override, no value-head change. For
RecurrentPPO that also means sb3-contrib's default LSTM hidden size (256), untouched.

**Evidence.** `[RECOVERED]` — `policy_kwargs` read from every checkpoint archive, this session.

**Alternatives rejected.** None were tried. §16.1 says "anything other than the library default
needs a reason"; the default needed none and got none.

**How we know.** The archives. Note what this means: **no architecture has ever been compared
against another in this lane.**

---

## D-3 — The full hyperparameter set, as exact numbers

**Status:** SETTLED (2026-09-09) as a record of what was *used*. **NOT** settled as a
recommendation — see the note below, which is the important part.

**The decision, as trained.** Every value is the library default; nothing was tuned.

| | DQN (rungs 7, 7a) | PPO (rungs 8, 8a–8c) | RecurrentPPO (rung 9, series A–D) |
|---|---|---|---|
| learning_rate | 1e-4 | 3e-4 | 3e-4 |
| γ_RL | 0.99 | 0.99 | 0.99 |
| n_steps | 1 | 2048 | **128** |
| batch_size | 32 | 64 | **128** |
| n_epochs | — | 10 | 10 |
| **ent_coef** | — (ε-greedy) | **0.0** | **0.0** |
| clip_range | — | 0.2 | 0.2 |
| gae_lambda | — | 0.95 | 0.95 |
| vf_coef / max_grad_norm | — / 10 | 0.5 / 0.5 | 0.5 / 0.5 |
| exploration_fraction / final_eps | 0.1 / 0.05 | — | — |
| buffer_size / learning_starts / tau | 1e6 / 100 / 1.0 | — | — |
| total_timesteps | 20,000–60,000 | 61,440–2,000,896 | 100,000–1,000,064 |

**Evidence.** `[RECOVERED]` — the `data` member of each archive, this session. Exact
`num_timesteps` per checkpoint is in `scratch/TRAINING_JOURNEY.md` §1.

**The note that matters.** This set produces a **camper**, and should not be reused. At full
scale — `--seeds 3 --sampled 10`, 2,223 episodes, `runs/baselines`, 2026-09-09 `[RECOVERED]` — the
RL rungs score **0.0%–1.8% paired-both** against round-robin, against `recency`'s 70.2%. But the
shape of the failure is more specific than "it does not work":

| rung | ratio | cTTI (s) | coverage | wins on ratio | **both** |
|---|---|---|---|---|---|
| 2 `round_robin` | 0.0605 | 4.18 | 0.8650 | — | — |
| 5 `recency` | 0.1104 | 3.20 | 0.8968 | 93.6% | **70.2%** |
| 4 `camper` *(the degenerate exploit)* | 0.2088 | 9.67 | 0.4971 | 69.0% | **1.8%** |
| 9b `recurrent_ppo_200k` | 0.2265 | 16.26 | 0.2518 | 78.9% | **0.0%** |
| 9d `recurrent_ppo_400k` | 0.2052 | 15.49 | 0.2823 | 78.9% | **0.0%** |

**The RL policies beat `recency` on interception ratio** — 0.2265 against 0.1104, winning that
column on 78.9% of episodes — and lose the `both` column outright because censored intercept time
is four times worse and coverage is a third. That profile is rung 4's, closely: **the agents
rediscovered the camper**, which is the exploit rung 4 exists to demonstrate is possible (D14).
An RL result that reproduces the known degenerate strategy is a more useful negative result than
one that merely underperforms, and it is the strongest thing this lane has produced.

Three suspected causes were nominated, in the order they were thought worth testing:

1. **γ_RL = 0.99** — a ~100-step effective horizon against a 300–600-step episode, so the return
   cannot see far enough for exploration to pay.
2. **n_steps = 128** — a fifth of an episode per rollout, re-fit 10 times each (`n_epochs=10`),
   78,130 updates across the 1M run.
3. **ent_coef = 0.0** — nothing in the loss opposes entropy collapse.

**Alternatives rejected.** None, at the time — no sweep was ever run.

**The diagnostic run, 2026-09-09** `[RECOVERED]`. γ_RL = 0.997, ent_coef = 0.01, n_steps = 2048,
100,352 steps, seed 0, reward `reward_balance`, 18 min CPU —
`runs/checkpoints/lstm_gamma997.zip`, sha256 `2a56a5395f97246d…`, manifest beside it.

**It did not break the camping, under `deterministic=True`.** One episode, `config_2` stare,
seed 0: band 3 for 598 of 600 slots, coverage 0.0526, ratio 0.0069 — *worse* on every metric than
the untuned 100k control (band 6 for 600/600, coverage 0.2632, ratio 0.0548).

**But that comparison turns out to be measuring the wrong thing** — see D-6. The same checkpoint
queried with sampled actions uses 32 of 36 bands with a longest streak of 4 slots and coverage
0.7895. The hyperparameters are not what is producing the camping signature; the inference setting
is. **Test D-6 before spending compute on another sweep**, and note that
`RL_TEAM_HANDOFF.md` §18's worked example prescribes exactly the γ/entropy fix that did not work
here.

---

## D-4 — The training scenario draw

**Status:** SETTLED (2026-09-09)

**The decision.** `rfenv.rl.common.make_train_env()` → `ScanEnv(pool=EmitterPool.from_train(),
reward=...)`: a fresh scenario per `reset()`, drawn from the **train pool only** (D25, D32).
`n_envs = 1`. Vec-env type: **`DummyVecEnv`**, and not by choice — SB3 wraps a bare `Env`
in one automatically; `SubprocVecEnv` was never used. Training seed **0** for every checkpoint
except the PPO `_fi`/`wt_cmp` family, which used seed **3**.

**Evidence.** `[RECOVERED]` — `rfenv/rl/common.py:58`; `seed` read from each archive. The held-out
guard sits under `EmitterPool.from_train` (D8), so no test-split emitter can enter training.

**Alternatives rejected.** `SubprocVecEnv` — not considered; throughput was never the constraint.

**How we know.** The source and the archives. **Why seed 3 for one family is
`[UNRECOVERABLE]`** — nothing records the reason it differs from every other run.

---

## D-5 — Which reward

**Status:** `OPEN`. **This lane did not settle it, and cannot.**

**Why not.** D47's rule needs the candidate that beats round-robin on *both* headline metrics on
the most paired episodes. **Every RL rung scores 0.0% on the `both` column**, so the rule selects
nothing — not "within 5 pp and escalate", but a floor that no candidate clears.

**Worse, the comparison cannot be attributed.** `[UNRECOVERABLE]` — four RecurrentPPO training
series (A–D) exist on disk with **identical seeds and identical recorded hyperparameters but four
different sets of weights**. Since seed and hyperparameters are held fixed, the variable is the
reward — and SB3 does not serialise the reward function. The training commands appear nowhere in
any session transcript. **Only the human who ran them knows which reward each used.**

This is the single largest hole in the record, and it is what the checkpoint manifest (see D-13)
now exists to make impossible in future.

**Note on the candidate set.** It has churned: `weighted_camp` (retired, D50), two
`first_intercept` shapes (D50, D51), and now `reward_balance`. `REWARDS` today is
`{hit_z, hit_y, reward_balance}` with `DEFAULT_REWARD = reward_balance`.

---

## D-6 — Inference-time action selection

**Status:** SETTLED (2026-09-09)

**The decision.** `deterministic=True`, everywhere, for every algorithm.

**Evidence.** `[RECOVERED]` — `rfenv/rl/common.py`, both adapters: `RLScheduler.__call__` and
`RecurrentRLScheduler.__call__` each pass `deterministic=True`. There is no code path that
samples, so the setting cannot silently differ between a debug run and a scored one.

**Alternatives rejected.** Sampling — never tried during the lane. **It should have been, and
this decision is now the prime suspect for the entire collapse result.**

**Measured 2026-09-09** `[RECOVERED]`, one episode, `config_2` stare, seed 0, the same policy
evaluated both ways:

| checkpoint | inference | distinct bands | longest streak | coverage | ratio |
|---|---|---|---|---|---|
| `lstm_ppo4_100000` | `deterministic=True` | **1** / 36 | 600 slots | 0.2632 | 0.0548 |
| `lstm_ppo4_100000` | sampled | **27** / 36 | 20 slots | **0.7895** | 0.0779 |
| `lstm_gamma997` | `deterministic=True` | **2** / 36 | 598 slots | 0.0526 | 0.0069 |
| `lstm_gamma997` | sampled | **32** / 36 | 4 slots | **0.7895** | 0.0635 |

The learned distribution is **not** collapsed — `entropy_loss = -2.58` against a `ln(36) = 3.58`
maximum, so it is broad. `deterministic=True` takes its argmax, and the argmax is one band. The
camping is therefore a property of **how the policy is queried**, not of what it learned.

**Consequence.** `deterministic=True` is not a detail, exactly as §16.1 warns. Every RL number in
this repository was produced under it, so every one of them measures the argmax of a broad
distribution. Coverage triples and airtime spreads across 27–32 bands the moment it is relaxed.
**This must be re-measured at `--seeds 3 --sampled 10` before either setting is settled** — the
table above is one episode and is a signal, not a result.

**How we know.** `rfenv/rl/common.py` for the setting; the table from a script run this session
that drives the same checkpoint through `rollout.run_episode` with `deterministic` toggled.

---

## D-7 — Checkpoint-selection criterion

**Status:** **NOT SETTLED, and it was needed before any run rather than after.**

**What happened instead.** No criterion was fixed in advance. Checkpoints were saved every N steps
and then compared after the fact — `runs/baseline_6`, `_8`, `_9`, `_10` each score a different
subset of snapshots. Picking the best of those after seeing the numbers is exactly the failure D39
names for gate thresholds, and §16.1 flags it as the thing a judge will ask about.

**Why it did not bite.** Only because there is nothing to pick: every snapshot scores 0.0% paired
both. A selection criterion applied to that set selects nothing regardless of when it was fixed.

**What to fix before the next run.** Fix the criterion now, in writing, while the numbers are still
absent: **highest paired-both against round-robin on train scenarios only, at
`--seeds 3 --sampled 10`, ties broken by the earlier checkpoint.** Written here so it predates the
next result.

---

## D-8 — D30: do AoA and PulseWidth enter the observation?

**Status:** **NO**, deferred (2026-09-09). `[RECALLED]` — the human's own answer this session:
*"AoA and PW will not be used for our usecase right now."*

**The decision.** They do not enter the observation. The wording is a **deferral, not a
rejection** — "right now" is the human's phrasing and is preserved deliberately, because D30 is
the only `OPEN` decision blocking the freeze and closing it as a permanent "no" claims more than
was said.

**Evidence.** None measured. **There is no ablation** `[UNRECOVERABLE]` — no with/without pair was
ever trained, so §16.1's "with the paired comparison that decided it" cannot be satisfied. The
decision is a scope call, not a measurement, and must be recorded as one.

**Consequence.** No observation-length change, no new slice constants, `tests/test_baselines.py`'s
slice assertions unaffected. Artefact **A-3** (the D30 pair) is not owed.

**Flagged for review.** A related tension that D49 does not mention: `measured_dbm` **does**
enter the observation, and D34's "What it deliberately excludes" names "peak amplitude within the
dwell" alongside AoA/PulseWidth. `measured_dbm` is the dwell *mean*, not the peak, so this may be
a distinction that holds — but D49 justifies the addition on D19 observability grounds and never
cites the exclusion it reverses. **Aamir should review this before anything is built on
`measured_dbm`**, and D49 likely owes an amendment either way.

---

## D-9 — Observation preprocessing

**Status:** SETTLED (2026-09-09)

**The decision.** **None.** No `VecNormalize`, no frame stacking, no observation transform of any
kind. The observation is already in `[0, 1]` by construction — `ScanEnv`'s `Box(low=0.0, high=1.0)`
— and `measured_dbm` is clamped to `[-120, -20]` dBm and rescaled inside `_observation()` for
exactly that reason.

**Evidence.** `[RECOVERED]` — `make_train_env` returns a bare `ScanEnv`; the only wrappers are the
`Monitor` and `DummyVecEnv` SB3 adds itself, neither of which transforms observations.

**Alternatives rejected.** `VecNormalize` — not needed, and it would have made the running
statistics part of the policy, which then have to ship with every checkpoint and be applied
identically at evaluation. The simplest answer is the correct one here.

---

## D-10 — Evaluation configuration

**Status:** SETTLED (2026-09-09)

**The decision.** `python -m rfenv.compare --seeds 3 --sampled 10 --figures --out runs/baselines`
— 47 stare replays + 10 sampled scenarios × 3 seeds, train split only, scan replays refused (D36).
One command, matching the existing §5 table.

**Evidence.** `[RECOVERED]` — run this session, 2026-09-09T20:30:50Z: 57 scenarios (47 stare
replays + 10 sampled) × 3 seeds = **2,223 episodes**, artefacts and figures in `runs/baselines`.

**What it corrects.** **No RL result before today used this configuration.** Every prior
comparison ran at 8–30 episodes over 2 scenarios (`runs/baseline_2` … `_10`), against this
command's 2,223. At that scale the interquartile ranges carry almost nothing, and the numbers moved
between runs: `recency` measured 33.3% and 50.0% paired-both in two small runs.

**The re-run validates itself against a committed figure:** `recency` comes out at **70.2%**
paired-both, exactly matching D46's committed value. That is the control that says the 2,223-episode
numbers can be trusted and the 8-to-30-episode ones could not. No small-run number in this lane
should be treated as final.

---

## D-11 — Stop/fail criteria and the fallback

**Status:** **NOT AGREED IN ADVANCE.** Recorded as a gap.

**What §16.1 asked for.** An agreement, fixed before the work, of the form "if rung 7 has not
cleared X by end of Day 3, we ship rung 5 as the headline and report rung 7 as an honest negative
result."

**What exists.** Nothing written. The condition it was meant to cover has nevertheless occurred:
rung 5 (`recency`) is the deployable headline, and rungs 7/8/9 are an honest negative result. That
outcome is being reported as one — see `scratch/TRAINING_JOURNEY.md` §3 and §4 — but it was reached
by observation rather than by a criterion agreed beforehand, which is a weaker position.

---

## D-12 — Compute envelope

**Status:** SETTLED (2026-09-09)

**The decision / the measurement.** All training was **CPU-only on one laptop**: Windows 11,
Intel64 Family 6 Model 165 (12 logical cores), no CUDA (`torch 2.14.0+cpu`, GPU disabled). Wall
clock, from checkpoint mtimes `[RECOVERED]`: RecurrentPPO ran **~100,000 steps per 18–20 minutes**
(series D: 100k at 11:12, 200k at 11:32, 300k at 11:51, 400k at 12:09), so the 1,000,064-step run
was roughly **3 hours**. DQN/PPO runs were minutes, not hours.

**What this means for reproduction.** Yes, on a laptop — a 1M-step RecurrentPPO run is an
afternoon, not a cluster job. That also sets the price of the retraining question: re-training the
six dead DQN/PPO rungs is hours, not days.

**Evidence.** `system_info.txt` in every archive; mtimes from `runs/checkpoints/`. From here on
this is recorded automatically — `wall_clock_s` and `hardware` are manifest fields (D-13).

---

## D-13 — Deviations from the brief

**Status:** SETTLED (2026-09-09). Every known deviation, stated.

1. **Three algorithms, not one.** The brief is written throughout for "rung 7"; the lane built
   rungs 7, 8 and 9. Recorded in D48.
2. **The observation changed.** The brief assumes D34's 109-wide vector; it is now 147. Recorded
   in D49. Five documents still say 109 (`ENVIRONMENT_SPEC.md`, `STATE_ACTION_FORMULATION.md`,
   `EDGE_LANE_HANDOFF.md`, `RL_TEAM_HANDOFF.md`, and the `technical-references/` duplicates) —
   **`EDGE_LANE_HANDOFF.md` has already been sent to the edge team and is wrong.** Not yet fixed.
3. **The reward set changed twice** beyond D29's three. Recorded in D50 and D51.
4. **No hyperparameter sweep was run**, which §16.1's D-3 assumes. See D-3.
5. **No checkpoint-selection criterion was fixed in advance.** See D-7.
6. **No stop/fail criterion was agreed.** See D-11.
7. **Checkpoints now carry a manifest.** Not in the brief; added 2026-09-09 in response to the
   D-5 failure above. Every `.zip` any trainer writes gets a `<name>.json` recording argv, git
   commit + dirty flag, reward, **observation width**, hyperparameters actually passed vs resolved,
   seed, timesteps, wall-clock, hardware, library versions and the zip's SHA-256; `load_checkpoint`
   refuses a width mismatch up front with the rebuilding command in the error. Snapshots are named
   `<run>_s1.zip`, `<run>_s2.zip`, … rather than by timestep count, because counts have twice
   drifted from the rung keys that quote them (rung 8a is labelled "800k" for a 600,064-step
   checkpoint; rung 7a says "5,000 timesteps" for a 20,000-step one). See `run.md`.
8. **`compare.py` no longer dies on an unbuildable rung.** It warns, names the rung and the
   reason, and continues — so the headline command works on a machine with no training stack, as
   §16.5.3 requires. It also names how many rungs were dropped, because a comparison that quietly
   contains no RL rows looks like a success and is not one.

---

## Not owed, and why

- **A-3** (the D30 with/without-AoA pair) — D-8 came out "no". Not owed.
- **A-2** (the three reward checkpoints behind the D-5 comparison) — cannot be supplied. The
  checkpoints exist but which reward each used is `[UNRECOVERABLE]`; handing over four
  indistinguishable archives would be worse than saying so.
- **E-5** (the D30 ablation table) — no ablation was run. See D-8.
