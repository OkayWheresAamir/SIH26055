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

Current width is **183** (`N_BANDS * 5 + 3`, N_BANDS = 36), verified this session via
`rfenv.rl.common.current_observation_width()`.

**Consequence, stated plainly: 26 of the 56 trained checkpoints can never be run again.** Their
measured numbers remain valid as history — the episodes really happened — but no new comparison can
include them, and nothing can be re-measured about them. This is the fourth time the cost has been
paid (D49, D55, D67), and it is the reason D67's own entry argues for batching future observation
changes rather than making them one at a time.

---

## 2. Every trained model

All 56 are RecurrentPPO. Grouped by observation width; within a width, by reward.
`sd` = training seed. `steps` = total timesteps at that snapshot.

### Width 183 — the current era, the only loadable models (30)

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
