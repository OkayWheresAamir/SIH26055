# Model comparison: every trained model, its observation space, and what it measured

Compiled 2026-09-11 by reading `runs/checkpoints/*.json` (56 manifests) and every `comparison.md`
under `runs/` (22 of them). Every number here is quoted from an artefact on disk, not from another
document. Where two runs disagree about the same model, **both are shown** and the reason is stated
— they are almost always different scenario sets.

**The one rule for reading this file: a number is only comparable to another number measured on the
same scenario set with the same seeds.** Section 3 groups runs by that. Comparing a 47-scenario
single-seed figure against a 57-scenario three-seed figure is not a comparison.

---

## 1. Why the observation width dominates everything

The observation is the vector fed to the policy network every step. Its width is fixed in the
network's input layer, so **a model trained at one width cannot be loaded at another** —
`rfenv/rl/common.py::require_loadable()` refuses it outright rather than letting it fail silently.
Every time the observation changed, every existing checkpoint died.

| width | what it added | decision | checkpoints on disk | loadable today |
|---|---|---|---|---|
| 147 | — (pre-D55 layout) | D49 era | 1 | **no** |
| 146 | D55 rescaled `visit_density`/`staleness`, removed `camp_time` | D55 | 25 | **no** |
| **183** | D67 appended `hit_streak` (36) + `current_hit_streak` (1) | **D67** | **30** | **yes** |
| 326 | D71 (2026-09-14) added PulseWidth (36), per-band amplitude (36, replacing a 1-wide global scalar), AoA sin/cos (72) | **D71** | 3 (`lstm_balance_v2_control_seed2_s1/s2/s3`, 385,024/400,000 steps, unfinished) | **no** — superseded by D72 the same day |
| **362** | D72 (2026-09-14) appended `pulse_count` (36), gated `C` | **D72** | 2, both finished (`lstm_balance_v2_d72_seed2`, `reward_balance`, 401,408 steps; `lstm_balance_improved_v2_d72_seed2`, `reward_balance_improved_v2`, 404,800 steps, resumed twice after crashes — D73) | **yes** |

**Updated 2026-09-14 (D71, then D72 the same day): this is no longer one number.** Every row above
183 in this table *replaced* the one before it — every old checkpoint died the moment the width
changed (D49, D55, D67). "v2" is different: it is a second, opt-in layout
(`ScanEnv(obs_version="v2")`) sitting alongside "v1", and every 183-wide checkpoint stays exactly as
loadable as it was (`obs_version` defaults to `"v1"`, unchanged). But "v2" itself is not immune to
widening in place — D72 moved it from 326 to 362 the same session D71 introduced it, and the three
snapshots D71's own retrain had already produced (`observation_width: 326` in their manifests, not
yet finished per `train.log`) are now permanently unloadable, the same cost every past observation
change has carried, paid again here.
`rfenv.rl.common.current_observation_width()` (the function this section used to cite) still only
ever answers for "v1" — the function that now matters for "is this checkpoint loadable at all" is
`rfenv.rl.common.known_observation_widths()`, which returns `{183, 362}`. `rfenv.env.obs_width("v1"
| "v2")` is the single source both read from.

**Consequence, stated plainly: 26 of the 56 trained checkpoints (all below 183) can never be run
again.** Their measured numbers remain valid as history — the episodes really happened — but no new
comparison can include them, and nothing can be re-measured about them. This is the fourth time
that specific cost has been paid (D49, D55, D67); D71 added a layout instead of replacing one, so it
paid nothing on "v1"'s 30 checkpoints — but **D72, the same day, widened "v2" itself in place** and
did pay a cost, on the three snapshots D71's own retrain had produced by then (not counted in the
56-manifest total above, compiled 2026-09-11, three days before either existed). "v1" remains
untouched by both D71 and D72; "v2" has now paid the same cost "v1" paid three times, once, this
early in its life.

---

## 2. Every trained model

All 56 are RecurrentPPO. Grouped by observation width; within a width, by reward.
`sd` = training seed. `steps` = total timesteps at that snapshot.

### Width 183 — "v1", the stable era (30)

| checkpoint | reward | steps | sd | rung | trained |
|---|---|---|---|---|---|
| `lstm_balance_d67_control_s1` | `reward_balance` | 100k | 0 | 14a | 2026-09-10 |
| `lstm_balance_d67_control_s2` | `reward_balance` | 200k | 0 | 14b | 2026-09-10 |
| `lstm_balance_d67_control_s3` | `reward_balance` | 300k | 0 | 14c | 2026-09-10 |
| `lstm_balance_d67_control_s4` | `reward_balance` | 400k | 0 | 14d | 2026-09-10 |
| `lstm_balance_d67_control` | `reward_balance` | 401k | 0 | — | 2026-09-10 |
| `lstm_balance_d67_control_seed1_s1`…`_s4` | `reward_balance` | 100–400k | 1 | 16a–16d | 2026-09-11 |
| `lstm_balance_d67_control_seed2_s1`…`_s4` | `reward_balance` | 100–400k | 2 | 17a–17d | 2026-09-11 |
| `lstm_balance_d67_treatment_s1` | `reward_balance_improved` | 100k | 0 | 15a | 2026-09-10 |
| `lstm_balance_d67_treatment_s2` | `reward_balance_improved` | 200k | 0 | 15b | 2026-09-10 |
| `lstm_balance_d67_treatment_s3` | `reward_balance_improved` | 300k | 0 | 15c | 2026-09-10 |
| `lstm_balance_d67_treatment_s4` | `reward_balance_improved` | 400k | 0 | 15d | 2026-09-10 |
| `lstm_balance_d67_treatment` | `reward_balance_improved` | 401k | 0 | — | 2026-09-10 |
| `lstm_balance_d67_treatment_seed1_s1`…`_s4` | `reward_balance_improved` | 100–400k | 1 | 18a–18d | 2026-09-11 |
| `lstm_balance_d67_treatment_seed2_s1`…`_s4` | `reward_balance_improved` | 100–400k | 2 | 19a–19d | 2026-09-11 |

Control and treatment are identical in every respect except the reward — same hyperparameters
(`ent_coef=0.01`, `gamma=0.997`, `n_steps=8192`), same D60 training split, same observation, same
seed per matched pair. That is what makes the comparison attributable to the reward.

### Width 362 — "v2" widened, D72 (2)

| checkpoint | reward | steps | seed | rung | trained |
|---|---|---|---|---|---|
| `lstm_balance_v2_d72_seed2` | `reward_balance` | 401,408 | 2 | 20d | 2026-09-14 |
| `lstm_balance_improved_v2_d72_seed2` | `reward_balance_improved_v2` | 404,800 | 2 | 21a | 2026-09-14 to 2026-09-18 |

Same pairing discipline as the 183-wide control/treatment rows above (same split, hyperparameters,
seed — only the reward differs), but on `ScanEnv(obs_version="v2")`, D72's 362-wide layout.
`lstm_balance_improved_v2_d72_seed2` was interrupted by two laptop crashes and resumed both times
via `RecurrentPPO.load()` + `learn(reset_num_timesteps=False)` from its own last `--checkpoint-freq`
snapshot — recorded in its own manifest description, not silently absorbed into the step count. Full
comparison: D73, `EVALUATION.md` §5.

The three 326-wide `lstm_balance_v2_control_seed2_s1/s2/s3` snapshots (rungs 20a–20c) predate this
pair, trained on D71's original "v2" before D72 widened it, never finished (385,024/400,000 steps),
and are now permanently unloadable — `326` is no longer in `known_observation_widths()` at all.

### Width 398 — "v2p", band-priority reward, measured null (D74) (2, complete)

| checkpoint | reward | steps | seed | rung | trained | notes |
|---|---|---|---|---|---|---|
| `lstm_balance_v2p_priority_seed2` | `reward_balance` | 401,408 | 2 | 22a | 2026-09-18 | Treatment: `band_priority=True`, `priority_uniform=False` — real per-episode elevated bands. Complete. |
| `lstm_balance_v2p_uniform_seed2` | `reward_balance` | 400,000 | 2 | 22b | 2026-09-18 | Control: same reward-term scale, `priority_uniform=True` — `band_priority` stays all-ones every episode. Complete. |

Same pairing discipline again (same split, hyperparameters, seed as the 362-wide pair above), but on
`ScanEnv(obs_version="v2p", band_priority=True, priority_coef=0.5, priority_n_bands=(3,6))`. This
pair is not a reward-formula comparison like the two rows above it — both rungs train on the same
`reward_balance` — it is a **control-arm** comparison: the two checkpoints differ only in whether
`band_priority` ever carries real information, isolating whether a trained policy actually
conditions on it versus just benefiting from the reward-scale increase the term adds uniformly.

**The comparison ran 2026-09-18** — two separate `compare.py` runs (round_robin + recency + one
rung each, 513 episodes apiece: 3 rungs × 57 scenarios × 3 seeds), then reproduced in a single
combined run after `compare.py` gained per-rung priority resolution (`resolve_priority_kwargs`,
same task) — identical numbers both ways:

| rung | scheduler | ratio | cTTI (s) | coverage | beats recency on **both** |
|---|---|---|---|---|---|
| 2 | round_robin | 0.060 | 4.18–4.20 | 0.864–0.865 | — |
| 5 | recency | 0.109–0.111 | 3.34–3.36 | 0.887–0.891 | — |
| **22a** — treatment | 0.120 | 3.28 | 0.891 | **44.4%** |
| **22b** — control | **0.128** | **2.70** | **0.908** | **61.4%** |

**The control beat the treatment on every column** — the opposite of what the feature is meant to
show. Three follow-up checks, all direct measurements rather than assumptions:

1. **Camping ruled out.** Treatment checkpoint, 20 episodes: **1.076** mean fair-share airtime on
   elevated bands against **1.007** on ordinary ones (`1.0` = an equal cut) — about 7% more
   attention, not camping, and inconsistent in direction across episodes (several showed *less*
   attention to elevated bands). Structurally unlikely regardless — `reward_balance`'s own
   `-3.0 × visit_density × n_slots` term already prices concentrated airtime, and the priority bonus
   only pays on a *new* discovery, never on occupancy.
2. **Treatment is more diffuse than control across the whole spectrum**, not just the priority
   bands. Same 20 episodes, entropy of the final `visit_density` distribution over all 36 bands:
   treatment **3.255** against control **3.051** (max possible `ln 36 = 3.584`); bands touched
   meaningfully (`visit_density > 0.5`): treatment **24.95** against control **18.95**, out of 36.
   Control's `band_priority` input is *constant* every episode — nothing to condition on, an easier,
   more stationary training problem — while treatment's genuinely varies episode to episode; more
   likely explanation for the gap than anything about the priority signal's content.
3. **Permutation ablation, decisive: the agent isn't reading `band_priority` at all.** 30 episodes,
   treatment checkpoint, each run twice on the identical scenario/seed/receiver-noise draw and the
   identical torch sampling seed — real `band_priority` in one run, the same values shuffled across
   bands in the other:

   | | corr(airtime, true priority) | ratio | cTTI (s) | coverage |
   |---|---|---|---|---|
   | real priority fed | +0.018 | 0.111 | 2.83 | 0.892 |
   | shuffled priority fed | +0.019 | 0.114 | 2.58 | 0.903 |

   Correlation between airtime and true priority is statistically indistinguishable whether the
   signal is real or garbage (real beats shuffled in 13/30 episodes — a coin flip), and performance
   does not degrade when the signal is corrupted.

**Conclusion (D74, `MEASURED`): the band-priority reward, at this scale, was never learned.**
Treatment's underperformance is a training-difficulty story (2, above), not a misused-signal story
— ruled out directly by (1) and (3). Single seed, one comparison run — not read as settled beyond
this configuration, same caveat D65/D73 give and do not resolve. **Not adopted, not promoted, code
not removed**: nothing defaults to `band_priority=True`, so the mechanism stays registered and
available; a larger coefficient, elevation multiplier, more training, or more LSTM capacity are
named, untried, unscoped next steps if this is picked up again, not a revision of these numbers.
Mechanism: `OBSERVATION_SPACE.md` §2.4. Full artefacts: `runs/d74_treatment_comparison/`,
`runs/d74_control_comparison/` — each also has `priority_animation_config_*.gif` (treatment only;
the control never elevates a band, so there is nothing to mark), a `compare.py --figures` output
that highlights the episode's elevated band(s) directly on the animated schedule, generated
automatically whenever a compared rung is tagged as priority-trained (`Rung.band_priority`,
`ladder.py`). Full account: `DECISIONS.md` D74; narrative: `scratch/TRAINING_JOURNEY.md` §18.

**Follow-up, same day (2026-09-19): the named next steps were tried, together, at a substantial
multiple of the original scale.**

| checkpoint | reward | steps | seed | rung | notes |
|---|---|---|---|---|---|
| `lstm_balance_v2p_priority_strong_seed2` | `reward_balance` | 802,816 | 2 | 23a | Treatment: `priority_coef=2.0` (was 0.5), `priority_high=5.0` (was 3.0), plus a new decaying per-slot occupancy term (`occupancy_coef=0.3`, `occupancy_decay_cap=6.0`, `priority_reward_bonus` in `env.py`). 512-wide LSTM (doubled), 800k timesteps (doubled). |
| `lstm_balance_v2p_uniform_strong_seed2` | `reward_balance` | 800,000 | 2 | 23b | Control, paired against 23a: identical except `priority_uniform=True`. Same LSTM size and timesteps. |

| rung | ratio | cTTI (s) | coverage | beats recency on **both** |
|---|---|---|---|---|
| 23a — treatment (strengthened) | **0.154** | **2.43** | 0.907 | **73.1%** |
| 23b — control (strengthened) | 0.121 | 2.74 | **0.911** | 45.6% |

**Treatment beat control by a wide margin — the opposite direction from the original pair above**
(control led there, 61.4% vs 44.4%). 23a's 73.1% is the highest "beats recency both" figure measured
anywhere in this project, ahead of D68's settled best (17c, 54.4%).

**A second permutation ablation says this is not the priority mechanism working.** Same method, on
the strengthened treatment checkpoint: real `band_priority` fed vs. the same values shuffled across
bands, 30 paired episodes, identical torch sampling seed per pair.

| | corr(airtime, true priority) | ratio | cTTI (s) | coverage |
|---|---|---|---|---|
| real priority fed | +0.031 | 0.145 | 1.80 | 0.960 |
| shuffled priority fed | +0.031 | 0.148 | 1.82 | 0.929 |

Identical correlation to three decimal places, real beats shuffled in only 12/30 episodes (40%),
performance unchanged when the signal is corrupted. **The agent still does not read `band_priority`**
— at roughly 4x the original discovery incentive, a whole new occupancy term, double the network,
and double the training budget. 23a's lead over 23b is read as single-seed training-run variance
(the direction flipped completely between the two pairs, while the ablation answer — "no, it doesn't
read the signal" — stayed the same both times), not a real, reproducible effect of the mechanism.

**One thing this follow-up does establish, kept separate from the priority question**: a 512-wide
LSTM trained for 800k steps outperforms every smaller/shorter checkpoint measured in this project so
far, on both arms. Whether that holds with `band_priority` removed entirely is untested — neither
23a nor 23b isolates capacity/budget from the mechanism's presence, since both still carry it, just
with real vs. uniform values.

**Verdict unchanged: not adopted, not promoted, code not removed.** The specific paths this entry's
original verdict left untried (bigger coefficient, more capacity, more training) have now been tried
together, at a substantial multiple of scale, and the result is the same. Full account: `DECISIONS.md`
D74 (same entry, extended); narrative: `scratch/TRAINING_JOURNEY.md` §18. Artefacts:
`runs/d74_followup_treatment_comparison/`, `runs/d74_followup_control_comparison/`.

### Width 146 — dead, but this is where several headline numbers came from (25)

| checkpoint family | reward | steps | rung | note |
|---|---|---|---|---|
| `lstm_balance_1M`, `_s1`…`_s10` | `reward_balance` | 100k–1M | 9a–9j | the 1M-step run |
| `clean_lstm_s1`…`_s4`, `lstm_balance_clean` | `reward_balance` | 100–401k | 10a–10d | **D64**, first clean post-leak run |
| `lstm_balance_improved_s1`…`_s4`, `lstm_balance_improved` | `reward_balance_improved` | 100–401k | 11a–11d | **D65**, treatment arm |
| `lstm_weighted_s1`…`_s4` | `weighted` | 100–400k | — | never registered; `weighted` fails D62 |

### Width 147 — one orphan (1)

`lstm_gamma997` (`reward_balance`, 100k, 2026-09-09). Pre-D55. Not registered on the ladder.

---

## 3. Comparison runs: which ones can be trusted, and for what

22 `comparison.md` files exist. They fall into four classes, and **only the first class supports
model-to-model conclusions.**

### Class A — full protocol: 57 scenarios (47 stare replays + 10 sampled) × 3 seeds = 171 episodes/rung

These are the only runs whose numbers belong in a headline claim.

| run | date | what it scored | status |
|---|---|---|---|
| `baselines/` | 09-09 | the original 7-scheduler ladder (D46) | valid, pre-RL |
| `acceptance_2026-09-10/` | 09-10 | 9a–9d + full ladder + oracles | **CONTAMINATED — train/eval leak (D60)** |
| `clean_paired_comparison/` | 09-10 | 10d vs 11b (D64/D65) | valid, 146-wide |
| `phase_switch_comparison/` | 09-10 | rungs 12/13 (D66) | valid, 146-wide |
| `d67_paired_comparison/` | 09-10 | 14c vs 15a (D67) | valid, 183-wide |
| `d68_rerun_paired_comparison/` | 09-11 | 17c vs 15a (D68 resolution) | valid, 183-wide |
| `d68_full_matched_seed_comparison/` | 09-11 | 20 rungs, all 16 matched-seed checkpoints | valid, 183-wide — **the deciding run, §5–6** |

### Class B — single-seed, 47 stare replays only, no sampled scenarios (seed 0)

`baselines_all/`, `baselines_all_2/`, `clean_baseline_100k/`, `clean_baseline_200k_all/`,
`clean_baseline_300k_all/`, `clean_baseline_all/`, `smoke_final_all_scenarios/`.

Useful for a broad sweep across many rungs at once — `baselines_all_2` scores 23 rungs in one table
— but **one seed is one roll of the dice.** D68 is the worked example of why that matters: a
single-seed checkpoint selection put two rewards 1.2 pp apart, and adding two seeds moved the gap to
17.6 pp. Treat Class B as indicative, never decisive.

### Class C — 2-scenario smoke tests

`baseline_2/`, `_4`, `_5`, `_6`, `_8`, `_9`, `_10`, `_11`, `_12`. Two stare replays each. These were
plumbing checks during development, not measurements. Ignore for comparison purposes.

### Class D — contaminated

`acceptance_2026-09-10/` (also Class A by protocol). Every checkpoint in it trained on the emitters
it was scored against, because training sampled `EmitterPool.from_train()` — all 47 development
configs — while evaluation used those same 47 (D60). Its rung 9 rows (9c at 48.0% / 13.5% paired
dominance) are arithmetically sound and permanently excluded from `EVALUATION.md` §5.

---

## 4. Headline performance, Class A runs only

All rows below: 57 scenarios × 3 seeds. Directly comparable to each other.

### Baselines, identical across every Class A run (the fixed points)

| rung | scheduler | interception ratio | censored intercept time (s) | coverage |
|---|---|---|---|---|
| 2 | `round_robin` | 0.0605 | 4.18 | 0.8650 |
| 5 | `recency` | 0.1105 | 3.34 | 0.8874 |
| 4 | `camper` | 0.2088 | 9.67 | 0.4971 |
| 6a | `apfeld_active_rfs` | 0.1320 | 4.32 | 0.8602 |
| — | `camper_oracle` *(reference, reads truth)* | 0.5680 | 15.71 | 0.2960 |
| — | `oracle_pulse` *(reference, reads truth)* | 0.6579 | 8.01 | 0.6912 |

Note the camper: **highest ratio of any deployable policy (0.2088) and the worst intercept time
(9.67 s) with half the coverage.** That is D14's trap in one row, and why no single metric decides
anything here.

### Trained models — all 20 rungs from `d68_full_matched_seed_comparison`, plus the two 146-wide rows for context

`d68_full_matched_seed_comparison` is the largest single run in this project: 20 rungs × 171
episodes each = 3,420 episodes, completed 2026-09-11. It is the only run that scores every
individual matched-seed checkpoint (16a–19d), not just the two D61 picked. **This supersedes
`d67_paired_comparison` and `d68_rerun_paired_comparison`** for any rung it covers — same
protocol, same scenario set, and the four shared rows (`round_robin`, `recency`, 14c, 15a) are
identical across all three runs, confirming nothing drifted between them.

| rung | model | width | ratio | cTTI (s) | coverage |
|---|---|---|---|---|---|
| 10d | `lstm_balance_clean_400k` | 146 | 0.1230 | 3.87 | 0.8613 |
| 11b | `lstm_balance_improved_200k_400k` | 146 | **0.1428** | 3.51 | 0.8596 |
| 14c | control, seed 0, 300k | 183 | 0.1096 | 3.93 | 0.8699 |
| 15a | treatment, seed 0, 100k | 183 | 0.1285 | 3.29 | 0.9051 |
| 16a | control, seed 1, 100k | 183 | 0.1121 | 3.32 | 0.9100 |
| 16b | control, seed 1, 200k | 183 | 0.1105 | 3.89 | 0.8777 |
| 16c | control, seed 1, 300k | 183 | 0.1115 | 3.50 | 0.8881 |
| 16d | control, seed 1, 400k | 183 | 0.1119 | 3.70 | 0.8960 |
| 17a | control, seed 2, 100k | 183 | 0.1243 | 3.49 | 0.8700 |
| 17b | control, seed 2, 200k | 183 | 0.1272 | 3.10 | 0.8891 |
| **17c** | **control, seed 2, 300k — D61-selected** | 183 | 0.1304 | **3.04** | 0.9019 |
| 17d | control, seed 2, 400k | 183 | 0.1263 | 3.36 | 0.8729 |
| 18a | treatment, seed 1, 100k | 183 | 0.1201 | 3.10 | 0.9106 |
| 18b | treatment, seed 1, 200k | 183 | 0.1160 | 3.52 | 0.8615 |
| 18c | treatment, seed 1, 300k | 183 | 0.1114 | 3.64 | 0.8880 |
| 18d | treatment, seed 1, 400k | 183 | 0.1095 | 4.00 | 0.8691 |
| 19a | treatment, seed 2, 100k | 183 | 0.1363 | 3.35 | 0.8635 |
| 19b | treatment, seed 2, 200k | 183 | **0.1424** | 3.50 | 0.8633 |
| 19c | treatment, seed 2, 300k | 183 | 0.1264 | 3.30 | 0.8897 |
| 19d | treatment, seed 2, 400k | 183 | 0.1070 | 3.31 | 0.8913 |

### Paired dominance — the column the selection rules actually use

`both` = fraction of the 171 paired episodes where the model beats the reference on **both** headline
metrics simultaneously (same scenario, same seed, same truth grid).

| rung | model | vs `round_robin` (floor) | vs `recency` (**the bar**) |
|---|---|---|---|
| — | `recency` | 67.3% | — |
| — | `round_robin` | — | 3.5% |
| 10d | `lstm_balance_clean_400k` | 59.1% | 22.8% |
| 11b | `lstm_balance_improved_200k_400k` | 66.7% | 31.0% |
| 14c | control, seed 0, 300k | 65.5% | 25.7% |
| 15a | treatment, seed 0, 100k | 64.3% | 35.1% |
| 16a | control, seed 1, 100k | 69.6% | 37.4% |
| 16b | control, seed 1, 200k | 60.8% | 28.7% |
| 16c | control, seed 1, 300k | 67.3% | 36.3% |
| 16d | control, seed 1, 400k | 73.1% | 36.8% |
| 17a | control, seed 2, 100k | 71.3% | 42.1% |
| 17b | control, seed 2, 200k | 74.9% | 44.4% |
| **17c** | **control, seed 2, 300k** | **81.9%** | **54.4%** |
| 17d | control, seed 2, 400k | 64.3% | 30.4% |
| 18a | treatment, seed 1, 100k | 76.6% | 40.9% |
| 18b | treatment, seed 1, 200k | 69.6% | 41.5% |
| 18c | treatment, seed 1, 300k | 65.5% | 28.7% |
| 18d | treatment, seed 1, 400k | 50.9% | 18.7% |
| 19a | treatment, seed 2, 100k | 70.2% | 43.9% |
| 19b | treatment, seed 2, 200k | 64.9% | 36.8% |
| 19c | treatment, seed 2, 300k | 64.9% | 30.4% |
| 19d | treatment, seed 2, 400k | 71.9% | 35.1% |

**17c leads every other 183-wide checkpoint on paired dominance against both references**, by a
clear margin — the next-best against the bar is 17b at 44.4%, ten points behind.

---

## 5. What the numbers say

**17c is the best model this project has produced, on the metric that matters most (paired
dominance) and one of the two raw headlines (cTTI) — but not on raw interception ratio.** This is
now settled directly, rather than inferred, now that every one of the 16 matched-seed checkpoints
has been individually scored (`d68_full_matched_seed_comparison`, §6):

- **Lowest censored intercept time of any of the 20 rungs measured: 3.04 s** (17c), against
  recency's 3.34 and round-robin's 4.18. No other checkpoint — old or new — beats it.
- **Not the highest raw interception ratio.** Two checkpoints beat 17c's 0.1304: rung 19b
  (`treatment, seed 2, 200k`, 0.1424) and rung 19a (`treatment, seed 2, 100k`, 0.1363). Both lose
  badly on paired dominance (19b: 64.9%/36.8%, roughly half of 17c's 81.9%/54.4%) and on coverage
  (0.8633 against 17c's 0.9019) — a higher ratio bought by looking away from more emitters, D14's
  trap in miniature. Chasing raw ratio alone would have picked a worse-rounded checkpoint.
- **By far the best paired dominance of any 183-wide checkpoint**, against both references. The
  next-best against the actual bar (rung 5) is 17b at 44.4% — ten points behind 17c's 54.4%, and
  every other checkpoint from both arms sits lower still.

This is exactly why D61/D47 select on paired dominance rather than on a raw metric: raw ratio alone
would have handed the project a checkpoint that trades away coverage and intercept time for a
number that looks better in isolation.

**Seed noise is large, now that it is directly measurable.** Within the control arm alone, paired
dominance against rung 5 ranges from 28.7% (16b) to 54.4% (17c) — the four snapshots of seed 0 alone
(14a–14d, not all shown above) plus the eight from seeds 1–2 span a comparable range. **A single
training seed is not a reliable way to judge a reward or a checkpoint**, which is the whole reason
this session ran the matched-seed retrain rather than trusting D68's first, single-seed result.

**But the single highest interception ratio ever measured on any protocol belongs to a model that
can no longer be run.** Rung 11b (`lstm_balance_improved_200k_400k`, 146-wide) reached **0.1428**,
just above 19b's 0.1424 and above 17c's 0.1304. It also loses badly on paired dominance (31.0%
against the bar, roughly half of 17c's). Not reachable now regardless — D67's observation change
killed it, along with every other 146-wide checkpoint.

**The control/treatment verdict reversed when seeds were added.** At one seed per arm, treatment
looked ahead — 15a beat 14c on every column. With three seeds per arm, D61 found a much stronger
control checkpoint (17c), and D47's re-application moved from a 1.2 pp gap (no selection) to a
17.6 pp gap in control's favour. **`reward_balance` is the selected reward (D68).** The lesson is
not that treatment is bad; it is that a single-seed checkpoint selection was picking a checkpoint
16.4 points weaker on the headline than the same reward could produce. Notably, treatment's own
best individual snapshot by paired dominance (18a, 76.6%/40.9%) also was not D61's seed-0 pick
(15a, 64.3%/35.1%) — the same single-seed limitation likely affected both arms, not only control's.

**No model is close to the oracles, and that is expected.** `oracle_pulse` reaches 0.6579 ratio by
reading the truth grid. `camper_oracle` reaches 0.5680 — and takes 15.71 s to first intercept with
0.2960 coverage, which is why a high ratio alone is not the goal.

---

## 6. Provenance of the deciding run

`runs/d68_full_matched_seed_comparison/` — 20 rungs (round-robin, recency, 14c, 15a, and all 16
matched-seed checkpoints 16a–19d) × 57 scenarios × 3 seeds = 3,420 episodes, completed 2026-09-11.
The largest comparison run in this project to date, and the source for every number in section 4's
expanded tables and section 5's analysis above. Figures (`--figures`) generated alongside: Pareto
plot, per-config timelines, discovery curves and animations for 3 sample configs.
