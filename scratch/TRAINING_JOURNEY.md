# Training journey — reconstructed 2026-09-09

Evidence tags, per RECOVERY_TASK.md:

- `[RECOVERED]` — a file, a command's output in this session, or a commit. Cited inline.
- `[RECALLED]` — stated in a session transcript (including the human's own answers in this
  session) but not verifiable against the repo. Quoted, and said so.
- `[UNRECOVERABLE]` — evidence is gone. Stated, left blank, not inferred.

**Re-inventoried 2026-09-09 12:37 local**, after the human restored the DQN/PPO checkpoints and
after four further training runs and three further comparison runs landed. Everything below is
re-read from disk at that time, not carried over.

---

## 0. Two corrections to the brief's premise, before anything else

**(a) The decisions the brief says are owed are already written.** `[RECOVERED]` —
`docs/project/DECISIONS.md` contains **D48–D51**, uncommitted in the working tree
(`git diff --stat` → 158 insertions):

| entry | covers | the brief's "owed" item |
|---|---|---|
| **D48** | RL joins the ladder: rungs 7 (DQN), 8 (PPO), 9 (RecurrentPPO); the two policy adapters; the parameterised-factory pattern | — |
| **D49** | Observation extended past D34's base three: `current_band`, `camp_time`, `measured_dbm`; 109 → 145 → 146 → 147 | **D-14** |
| **D50** | `DEFAULT_REWARD` moved off `hit_z`; `weighted_camp` registered and retired | **D-16** |
| **D51** | `newly` credits per `(emitter, band)` rather than once per episode; measured 2.50×/2.52× inflation; the naming tension against D28 | **D-15** |

D34's status line was amended in place to point forward to D49. Phase 4a would be renumbering
work that already exists.

**(b) The environment did not run for part of this session — since fixed.** `[RECOVERED]` Two
independent breakages in `rfenv/env.py`, both introduced by mid-session edits and both repaired
later the same day (see §8):

1. `DEFAULT_REWARD = "first_intercept_train"` named no key of `REWARDS`, so every `ScanEnv()` or
   `make_train_env()` call without an explicit `reward=` raised.
2. `step()` called `_reward_fn` with six arguments to suit the new `reward_balance`, while
   `reward_hit_z`/`reward_hit_y` still took three — so only one of the three registered rewards
   could run at all.

Together these accounted for 42 of the 79 test failures measured at the time.
`reward_first_intercept` no longer exists as a function; it survives only as a commented-out
`reward_first_intercept_test` draft in `env.py`.

---

## 1. The checkpoints, as they exist now `[RECOVERED]`

Read this session from the `.zip`s themselves (`data` member + `system_info.txt`). All nineteen:
`stable-baselines3 2.9.0`, PyTorch 2.14.0+cpu, Python 3.12.10, `Discrete(36)`, `policy_kwargs {}`
(library-default network, no `net_arch` override).

### The non-recurrent family — all restored, all stale

| file | sha256 (16) | mtime (local) | obs | timesteps | updates | seed |
|---|---|---|---|---|---|---|
| `checkpoints/deep_q_network.zip` | `ee66b17e5f1ca1df` | 09-08 10:58 | **145** | 60,000 | 14,975 | 0 |
| `checkpoints/deep_q_network_hit_y.zip` | `d7507d26588bdc7b` | 09-08 10:59 | **145** | 20,000 | 4,975 | 0 |
| `checkpoints/ppo.zip` | `b649f341d053b7c0` | 09-08 11:00 | **145** | 61,440 | 300 | 0 |
| `checkpoints/ppo_fi.zip` | `2f9545022ace38d6` | 09-08 11:23 | **145** | 600,064 | 2,930 | 3 |
| `checkpoints/ppo_fi_100k.zip` | `c52d1172f7144972` | 09-08 11:49 | **145** | 100,352 | 490 | 3 |
| `checkpoints/ppo_fi_2M.zip` | `88b3f1d7ebe13fe9` | 09-08 13:04 | **145** | 2,000,896 | 9,770 | 3 |
| `checkpoints/ppo_wt_cmp_1M.zip` | `8ffffdb4cc6077a3` | 09-08 14:48 | **146** | 1,001,472 | 4,890 | 3 |

These were restored by the human between the first inventory and this one `[RECALLED]` — *"I added
them back i had removed them since my observation space had changed and i couldnt use those files
anymore"*.

**Every one is stale against the current 147-wide observation** `[RECOVERED]`. Six are 145-wide, so
they predate both `camp_time` and `measured_dbm`; `ppo_wt_cmp_1M` is 146-wide, so it predates
`measured_dbm` alone. Note the correction this forces: the DQN/PPO era was **145**, never 109 —
`current_band` was already in the vector when every one of them was trained.

Two notable details: `ppo_fi.zip` is a **600k** run despite rung 8a labelling it "800k"
(`ladder.py:151`), and `deep_q_network_hit_y.zip` is a **20,000**-step run despite `ladder.py:139`
describing it as 5,000 timesteps. Both labels are wrong; the archives are not.

### The recurrent family — four separate training series

| file | sha256 (16) | policy.pth sha (16) | mtime | obs | timesteps |
|---|---|---|---|---|---|
| **A** `old_checkpoints/recurrent_ppo.zip` | `bea8f01de18139ba` | — | 09-09 07:37 | 147 | 1,000,064 |
| **A** `old_checkpoints/lstm_ppo_100000_steps.zip` | `103dc0c5859027b5` | — | 09-09 07:57 | 147 | 100,000 |
| **A** `old_checkpoints/lstm_ppo_200000_steps.zip` | `67a3fd54be7615d0` | — | 09-09 08:15 | 147 | 200,000 |
| **A** `old_checkpoints/lstm_ppo_300000_steps.zip` | `816b418fa8403f88` | — | 09-09 08:34 | 147 | 300,000 |
| **B** `checkpoints/lstm_ppo_100000_steps.zip` | `a801ba05fb7b2607` | `9b592740826c36ad` | 09-09 09:17 | 147 | 100,000 |
| **B** `checkpoints/lstm_ppo_200000_steps.zip` | `d5965145375aeeab` | `342d2a5eb09c42f8` | 09-09 09:36 | 147 | 200,000 |
| **C** `checkpoints/lstm_ppo3_100000_steps.zip` | `03063148e60833df` | `4dbef9b4bd00e382` | 09-09 10:37 | 147 | 100,000 |
| **C** `checkpoints/lstm_ppo3_200000_steps.zip` | `925ca8d44755f13f` | `52c896dfef55581e` | 09-09 10:54 | 147 | 200,000 |
| **D** `checkpoints/lstm_ppo4_100000_steps.zip` | `861b6473cf107946` | `6dfeaca6f3128898` | 09-09 11:12 | 147 | 100,000 |
| **D** `checkpoints/lstm_ppo4_200000_steps.zip` | `772a2afc90235c2a` | `6b729fceff7acffb` | 09-09 11:32 | 147 | 200,000 |
| **D** `checkpoints/lstm_ppo4_300000_steps.zip` | `75d54da0c6f0e65d` | `a1a90cb3ae72b9e2` | 09-09 11:51 | 147 | 300,000 |
| **D** `checkpoints/lstm_ppo4_400000_steps.zip` | `ce06fb8fd00f86a1` | `e51b76c9f332f5ea` | 09-09 12:09 | 147 | 400,000 |

**Series B is not series A re-saved** `[RECOVERED]` — same filenames, same `num_timesteps`, same
`seed=0`, same hyperparameters, **different SHA-256**. Four series were trained with identical
recorded constructor arguments and produced four different sets of weights. Since seed and
hyperparameters are held fixed, the thing that differed between them is not in the archive — it is
the **reward function**, which SB3 does not serialise. That is what makes §5's open question the
load-bearing one.

Hyperparameters, every checkpoint, **all library defaults, nothing tuned**:

| | DQN (7, 7a) | PPO (8, 8a–8c, 8d) | RecurrentPPO (9, series A–D) |
|---|---|---|---|
| learning_rate | 0.0001 | 0.0003 | 0.0003 |
| gamma | 0.99 | 0.99 | 0.99 |
| n_steps | 1 | 2048 | **128** |
| batch_size | 32 | 64 | **128** |
| n_epochs | — | 10 | 10 |
| **ent_coef** | — (ε-greedy) | **0.0** | **0.0** |
| gae_lambda | — | 0.95 | 0.95 |
| vf_coef / max_grad_norm | — / 10 | 0.5 / 0.5 | 0.5 / 0.5 |
| exploration_fraction / final_eps | 0.1 / 0.05 | — | — |
| buffer_size / learning_starts / tau | 1e6 / 100 / 1.0 | — | — |

`ent_coef = 0.0` across every policy-gradient checkpoint was initially read as the single most
load-bearing recovered number. **§4.2 supersedes that**: the policies are not entropy-collapsed,
and the camping is an inference-time artefact. `ent_coef = 0.0` remains true and remains untuned;
it is no longer the leading explanation of anything.

### Loadability against the current environment `[RECOVERED]`

`load_checkpoint()` binds no env, so **all nineteen used to load without error** — the mismatch
only fired later, at `predict()`. A stale checkpoint therefore failed at *comparison* time rather
than at registration time, which is why `ladder.py` registering six dead rungs looked healthy until
`compare` ran, and why restoring the six stale DQN/PPO checkpoints to disk turned ~30 clean test
skips into failures with a bare shape error.

**Fixed 2026-09-09** (§8): `load_checkpoint()` now calls `require_loadable()` first, which reads
the recorded observation width from the checkpoint's manifest or, for pre-manifest checkpoints,
from the archive's own pickled `observation_space`, and refuses with a named reason.

---

## 2. The runs, dated `[RECOVERED]`

From each run's own `comparison.md` header (UTC) and file mtime (local, UTC-7).

| artefact dir | written | scale | rungs present |
|---|---|---|---|
| `runs/baseline_2` | 09-08 11:56 | 18 ep | `deep_q_network_z_60k`, `ppo_first_intercept_{100k,800k}`, recency, round_robin |
| `runs/baseline_3` | 09-08 13:08 | 24 ep | `ppo_first_intercept_{2M,800k}`, recency, round_robin — **deleted from `runs/` since; survives only in the backup** |
| `runs/baseline_4` | 09-08 13:18 | 24 ep | same four |
| `runs/baseline_5` | 09-08 14:52 | 18 ep | `ppo_weighted_camp`, recency, round_robin |
| `runs/baselines` | 09-09 08:36 | 12 ep | 20 rungs — every heuristic + oracles + all DQN/PPO + `recurrent_ppo_*_{100k,200k,300k,1M}` |
| `runs/baseline_6` | 09-09 08:43 | 12 ep | round_robin, recency, `recurrent_ppo_*_{100k,200k,300k,1M}` |
| `runs/baseline_8` | 09-09 10:39 | 8 ep | round_robin, recency, `recurrent_ppo_*_{100k,200k}` |
| `runs/baseline_9` | 09-09 11:33 | 24 ep | round_robin, recency, `recurrent_ppo_*_{100k,200k}` |
| `runs/baseline_10` | 09-09 12:19 | 30 ep | round_robin, recency, `recurrent_ppo_*_{100k,300k,400k}` |

Phase 0's backup earned itself: **`baseline_3` no longer exists in `runs/`** and is preserved only
in `runs_backup_2026-09-09/`.

**Rung-key churn is visible in the directory names** `[RECOVERED]`: `deep_q_network` →
`deep_q_network_60k` → `deep_q_network_z_60k`; `ppo` → `ppo_hit_z_60k`; `ppo_first_intercept_80k`
vs `_100k` vs `_800k`. Several of those keys exist in no surviving version of `ladder.py`.

**No run has used the acceptance scale.** The brief's `--seeds 3 --sampled 10` is 1,539 episodes;
the largest run above is 30. Interquartile ranges over 2 scenarios carry very little.

**`ladder.py`'s rung 9 block has drifted out of sync with disk** `[RECOVERED]` — 9a–9d now point at
`lstm_ppo4_*` (series D), but 9e/9f/9c point at `lstm_ppo_{500000,600000,700000}_steps.zip`, none of
which exist. Rung 9 itself still points at `runs/checkpoints/recurrent_ppo.zip`, which is now in
`old_checkpoints/`. And **"9c" is used twice** (300k and 700k) — the same key collision that
previously ate a row out of the Pareto figure.

---

## 3. The headline result, at full scale `[RECOVERED]`

**Superseded 2026-09-09.** What follows replaces the earlier §3/§4, which were written from
8-to-30-episode runs and from `deterministic=True` inference alone. Both conclusions moved.

### 3.1 The measurement that counts

`python -m rfenv.compare --seeds 3 --sampled 10 --figures --out runs/baselines`, run
2026-09-09T20:30:50Z: 57 scenarios (47 stare replays + 10 sampled) × 3 seeds × 13 buildable
rungs = **2,223 episodes**. Reward `reward_balance`.

**The run validates itself against a committed figure.** `recency` scores **70.2%** paired-both,
matching `EVALUATION.md` §5 / D46 exactly, as do every other heuristic row and both oracles. The
two earlier small runs put `recency` at 33.3% and 50.0%. That is the control: the 2,223-episode
numbers reproduce a committed result, the 8-to-30-episode ones did not, and **no small-run number
from this lane should be quoted.**

| rung | scheduler | ratio | cTTI (s) | coverage | wins ratio | **both** |
|---|---|---|---|---|---|---|
| 2 | `round_robin` | 0.0605 | 4.18 | 0.8650 | — | — |
| 5 | `recency` | 0.1104 | 3.20 | 0.8968 | 93.6% | **70.2%** |
| 6 | `apfeld` | 0.2455 | 14.86 | 0.3672 | 83.6% | **4.1%** |
| 4 | `camper` *(the degenerate exploit)* | 0.2088 | 9.67 | 0.4971 | 69.0% | **1.8%** |
| 9a | `recurrent_ppo` 100k | 0.1251 | 19.16 | 0.1248 | 42.1% | **1.8%** |
| 9b | `recurrent_ppo` 200k | **0.2265** | 16.26 | 0.2518 | 78.9% | **0.0%** |
| 9c | `recurrent_ppo` 300k | 0.1695 | 16.26 | 0.2483 | 69.0% | **1.8%** |
| 9d | `recurrent_ppo` 400k | 0.2052 | 15.49 | 0.2823 | 78.9% | **0.0%** |
| — | `camper_oracle` *(reference)* | 0.5680 | 15.71 | 0.2960 | 100.0% | 7.0% |

### 3.2 The result: the agents rediscovered the camper

The RL rungs **beat `recency` on interception ratio** — 0.2265 against 0.1104, winning that column
on 78.9% of episodes — and lose the `both` column outright because censored intercept time is four
to five times worse and coverage is a third. Read the rows together and the profile is rung 4's,
and `apfeld`'s: high ratio, ruinous intercept time, collapsed coverage.

**That is the finding.** Rung 4 exists in the ladder precisely to demonstrate that a single metric
can be gamed by camping (D14). An agent trained on this environment converges on exactly that
exploit, unprompted. A negative result that reproduces the known degenerate strategy says something
about the reward and the metric pair; one that merely underperforms says nothing.

Note also: **rungs 7 and 8 (DQN, PPO) are absent from the table entirely.** All six of their
checkpoints predate the observation change and cannot run (§1). The RL rows above are rung 9 only.

---

## 4. Why the camping happens — and it is not what §4 first said

### 4.1 The original diagnosis, and its error

The first pass attributed the camping to **behavioural collapse during training**, on the evidence
that `ent_coef = 0.0` on every policy-gradient checkpoint and that 100k/200k/1M produced identical
metrics. The first half is a fact; the conclusion drawn from it was wrong.

The tell was in the diagnostic run's own log: **`entropy_loss = -2.58`, against a `ln(36) = 3.58`
maximum.** A collapsed policy does not have an action distribution that broad.

### 4.2 What is actually happening `[RECOVERED]`

Measured 2026-09-09: the same checkpoint, the same episode (`config_2` stare, seed 0), driven
through `rollout.run_episode` twice, changing only `deterministic`:

| checkpoint | inference | distinct bands | longest streak | coverage | ratio |
|---|---|---|---|---|---|
| `lstm_ppo4_100000` | `deterministic=True` | **1** / 36 | 600 slots | 0.2632 | 0.0548 |
| `lstm_ppo4_100000` | sampled | **27** / 36 | 20 slots | **0.7895** | 0.0779 |
| `lstm_gamma997` | `deterministic=True` | **2** / 36 | 598 slots | 0.0526 | 0.0069 |
| `lstm_gamma997` | sampled | **32** / 36 | 4 slots | **0.7895** | 0.0635 |

**The camping is a property of how the policy is queried, not of what it learned.** The learned
distribution is broad; `deterministic=True` takes its argmax, and the argmax is a single band.
Relax it and airtime spreads over 27–32 of 36 bands and coverage triples.

Every RL number in this repository — including §3.1's — was produced under `deterministic=True`
(`rfenv/rl/common.py`, both adapters). They all measure the argmax of a broad distribution.

**This is a signal, not yet a result:** one episode, one scenario, one seed. It needs the same
`--seeds 3 --sampled 10` treatment §3.1 got before it is a claim.

### 4.3 The hyperparameter hypothesis, tested and not supported

`runs/checkpoints/lstm_gamma997.zip` (sha256 `2a56a5395f97246d…`, manifest beside it): γ_RL 0.997,
`ent_coef` 0.01, `n_steps` 2048, 100,352 steps, seed 0, reward `reward_balance`, 18 min CPU. The
reasoning was sound — γ 0.99 is a ~100-step horizon against a 300–600-step episode, and
`n_steps=128` is a fifth of an episode per rollout — and it is the fix `RL_TEAM_HANDOFF.md` §18's
own worked example prescribes.

**Under `deterministic=True` it camped harder**, not less: band 3 for 598 of 600 slots, coverage
0.0526, ratio 0.0069 — worse than the untuned control on every metric. Under sampling it is the
*least* camped policy measured (32 bands, longest streak 4).

So the ranked suspects have changed. `deterministic=True` is first; the hyperparameters are not
excluded but are no longer the leading explanation. Still true and still untested: no run has
compared sampled against deterministic inference at scale.

### 4.4 What the observation change bought, still unmeasured

`camp_time` and `current_band` were added in D49 specifically to make a camping streak visible to
the policy. `[UNRECOVERABLE]` whether they helped — no ablation exists and none can now be run
(§1: no 109- or 145-wide checkpoint can execute in the current environment).

---

## 5. The human's answers, 2026-09-09 `[RECALLED]`

Given directly in session; not independently verifiable against the repo, so tagged as recall even
where the code corroborates them.

| # | answer | corroboration |
|---|---|---|
| 1 | *"we use first intercept for the reward function"* | Partly `[RECOVERED]`: `baseline_8/9/10`'s headers record `Reward first_intercept_train`, and `DEFAULT_REWARD` still names it. But no function of that name exists in `env.py` now — see §0(b). |
| 2 | *"it was rewritten to incentivise exploration and punish camping"* | `[RECOVERED]` in the code: `reward_balance` (`env.py:126-152`) pays `0.5·hit_rate[a] + (1.5 − visit_density[a])·staleness[a] + 0.5·ΣZ − 0.5·camp_slots`. Three of its four terms are exploration or anti-camping terms. **This is the answer to D-15/D-16 that the record was missing.** |
| 3 | *"i had changed the observation space from 109 to 147 dimensions"* | `[RECOVERED]` — matches D49 and the archives. Refines the brief's "109 → 146": 146 was one intermediate state of three (145 → 146 → 147). |
| 4 | *"I added them back i had removed them since my observation space had changed"* | `[RECOVERED]` — the seven non-recurrent checkpoints are back on disk (§1) and all are 145/146-wide, i.e. exactly as stale as the removal implies. |
| 5 | *"AoA and PW will not be used for our usecase right now"* | **This closes D30/D-8**, which was the only OPEN decision blocking the freeze. It needs writing up as a decision with "right now" preserved — a deferral, not a rejection. |

---

## 6. What remains missing

| question | status |
|---|---|
| Why 109 → 147? | **Answered** — D49, and answer 3 above. |
| Was the extension ever measured against the 109 (or 145) vector? | `[UNRECOVERABLE]` — **no ablation exists**, and none can be run now: no 109-wide checkpoint survives and the 145/146-wide ones cannot execute in the current env. See also §4.4. |
| Why was the first-intercept reward rewritten? | **Answered** by answer 2: to incentivise exploration and punish camping. |
| What was `weighted_camp`, why retired? | **Answered** in D50. Its checkpoint (`ppo_wt_cmp_1M.zip`, 146-wide) survives but cannot run. |
| Checkpoint training order | **Recovered** (§1, §2) by mtime, `num_timesteps` and observation width. |
| Which checkpoint is best, by what measure? | **Recovered: none of them.** At full scale every RL rung scores 0.0–1.8% paired-both against round-robin, against `recency`'s 70.2% (§3.1). No checkpoint-selection criterion was ever fixed in advance; one is now written down in `HANDOVER/DECISIONS_DRAFT.md` D-7. |
| What did not work? | **Recovered and substantial** — §3, §4. This is the real deliverable. |
| **Which reward did each of the four LSTM series train on?** | `[UNRECOVERABLE]` **and it is now the one hole that matters.** Four series share seed and hyperparameters and differ only in weights, so reward is the variable — but SB3 does not serialise it, the training commands appear nowhere in the transcript (grepped), and the reward function they named has since been deleted from `env.py`. Without it, §3's numbers cannot be attributed to a reward and D47's selection rule has nothing to select between. **Only the human can supply this.** |
| D30 / D-8 (AoA, PulseWidth) | **Answered** by answer 5 — deferred, not rejected. Needs writing up. |

---

## 7. Evidence protected

`runs/` → `runs_backup_2026-09-09/` — 224 MB, 8,851 files, verified. It has already paid for itself:
`runs/baseline_3` has been deleted from `runs/` since the copy was taken and exists nowhere else.

The backup predates the checkpoint restoration and series C/D, so it does **not** contain
`deep_q_network*.zip`, `ppo*.zip`, `lstm_ppo3_*` or `lstm_ppo4_*`. Those exist in `runs/` only.

---

## 8. Changes made 2026-09-09, and why `[RECOVERED]`

Logged here because the whole point of this document is that work which is not written down stops
existing. Nothing below is committed except where stated.

### 8.1 Committed

| commit | what |
|---|---|
| `ae2b5ff` | **D48–D51 written into `docs/project/DECISIONS.md`** and pushed to `origin/rl-baselines-modularity`. 158 insertions. They had been sitting uncommitted in the working tree — the same failure this document records, one level up. Not pushed to the `zatiyab` remote. |

### 8.2 The checkpoint manifest — reproducibility by construction

Every `.zip` any trainer writes now gets a `<name>.json` beside it, from one shared helper
(`rfenv/rl/common.py::write_manifest`), so DQN, PPO, RecurrentPPO and anything added later are
self-describing without anyone remembering. It records what the SB3 archive cannot: verbatim
`argv` plus a paste-able module-form command, git commit + dirty flag, **reward name**,
**observation width at training time**, hyperparameters *actually passed* alongside those
*resolved* off the model, seed, `total_timesteps`, wall-clock, hardware, library versions, and the
zip's SHA-256.

**The reward field is the one that pays for the rest**, and §6 says why: four training series exist
on disk with identical seeds and identical recorded hyperparameters but four different sets of
weights, and the variable — the reward — is the one thing SB3 does not serialise. That hole cannot
open again.

`load_checkpoint()` now calls `require_loadable()` first, which refuses a width mismatch up front
with the rebuilding command in the message, rather than letting it fail later inside `predict()`
with a shape error that names neither the checkpoint nor the cause. Pre-manifest checkpoints fall
back to the archive's own recorded `observation_space`; a checkpoint with neither is allowed
through, since there is then nothing to check against.

Snapshots from `--checkpoint-freq` are named `<run>_s1.zip`, `<run>_s2.zip`, … rather than by
timestep count. A count in a filename gets transcribed into a rung key by hand and then drifts from
the archive, which has happened twice here: rung 8a is labelled "800k" for a 600,064-step
checkpoint, and rung 7a says "5,000 timesteps" for a 20,000-step one. New CLI flags on all three
trainers: `--run-name`, `--description`, and repeatable `--hyperparam KEY=VALUE` — without the
last, "hyperparameters actually passed" would always be empty and the field would be hollow.

### 8.3 The evaluation harness

- **`compare.py` no longer dies on the first unbuildable rung.** `buildable_rungs()` probes each
  rung once, drops the ones that cannot be constructed — no training stack, untrained checkpoint,
  or stale observation width — and names each on stderr, plus a count. Before this, one missing
  checkpoint killed a 2,223-episode run with a traceback, and the headline command did not work at
  all on a machine without `stable_baselines3`. The count is printed deliberately: a comparison
  that quietly contains no RL rows looks like a success and is not one.
- **`ladder.py`**: removed three rungs pointing at checkpoints that were never trained
  (`lstm_ppo_{500000,600000,700000}_steps.zip`). One of them was numbered **`9c`, colliding with
  the 300k rung** — the same duplicate-key bug that previously ate a row out of the Pareto figure.
- **`tests/test_baselines.py`**: the per-rung skip checked only whether a checkpoint file *exists*.
  It now also checks whether it can *run*, so a restored-but-stale checkpoint skips rather than
  failing with a bare shape error.

### 8.4 `env.py` repairs

Both breakages in §0(b), fixed. All three rewards now share one call signature — the convention the
file's own comment block already mandated — and `newly` was restored to the call, without which the
commented-out first-intercept draft cannot be re-enabled. **No reward's return value changes**;
these are signature-only edits, so no checkpoint is invalidated by them.

### 8.5 Test suite

**79 failed → 1 failed, 287 passed, 42 skipped.** The single remaining failure,
`test_heldout_split_is_refused_without_an_explicit_flag`, predates all of this and is
environment-dependent: it expects the 45 held-out test pairs and finds none on this machine. Not
touched — `data/turing/**/test_*` is off limits.

### 8.6 Still open

- **Which reward each LSTM series trained on** — `[UNRECOVERABLE]`, and the reason the manifest now
  exists. Only the human can supply it.
- **Sampled vs deterministic inference** — **RESOLVED 2026-09-09, see §9.1a and D54.** The
  distribution itself was read rather than inferred from behaviour: mean entropy 2.369 against
  `ln 36 = 3.584`, modal band holding 0.206 of the mass. The camping was the argmax, not the policy.
  Still outstanding is the *protocol-scale* re-run (`--seeds 3 --sampled 10`) against a checkpoint
  trained on the corrected reward — §9.3's run is that checkpoint.
- **`measured_dbm` against D34's exclusion list** — flagged for human review; D49 justifies the
  addition on D19 grounds and never cites the exclusion it reverses. Nothing is being built on it
  meanwhile.
- **Rungs 7 and 8 are permanently dead** at 145/146 wide. Retraining is hours, not days (§8.2's
  manifest records ~100k RecurrentPPO steps per 18–20 min, CPU-only) — a compute decision, not a
  technical one.

---

# 9. The collapse diagnosed, and the first run on a corrected reward — 2026-09-09

§4 left the camping unexplained and §8.6 named the sampled-vs-deterministic measurement as the
highest-value thing outstanding. Both are now resolved, and the answer was three separate defects
stacked on each other. Every number in this section was measured this session; none is carried over
from a document.

## 9.1 The three defects

**(a) Inference took the argmax of a policy that had not collapsed — D54.**

Reading the action distribution directly off `model.policy.get_distribution` for
`runs/checkpoints/lstm_gamma997.zip`, over a full seed-0 episode:

```
steps                586
mean entropy         2.369   (uniform over 36 = ln 36 = 3.584; fully collapsed = 0)
min / max entropy    2.293 / 3.553
mean max-probability 0.206   (min 0.044, max 0.215)
argmax histogram     band 5 on 580 steps, band 3 on 6
```

The policy is **broad**. Its mode is merely sticky, and `RecurrentRLScheduler` hardcoded
`deterministic=True`, so every rung 9 row in `EVALUATION.md` §5 reported that sticky mode rather
than the policy. Same checkpoint, same seeds, both ways:

| inference | distinct bands | interception ratio | coverage | episode reward |
|---|---|---|---|---|
| `deterministic=True`, seed 0 | 2 | 0.1042 | 0.247 | −167,871 |
| `deterministic=True`, seed 1 | 2 | 0.0003 | 0.041 | −178,922 |
| `deterministic=False`, seed 0 | **31** | 0.0618 | **0.603** | **+114.0** |
| `deterministic=False`, seed 1 | **31** | 0.2006 | **0.714** | **+52.7** |

PPO optimises expected return under the sampled policy — the +114, not the −167,871 — so it had no
gradient pointing at the mode and no reason to fix it. **§4's "the agent rediscovered the camper"
is withdrawn.**

**(b) The reward's exploration term was inverted — D52.**

`ScanEnv.step` stored `_staleness_array[action] = _last_slot[action] / N_SLOTS` — *when* a band was
last seen — while `_observation()` reports `(t - _last_slot) / N_SLOTS`, *how long ago*. Those run
in opposite directions over an episode. Sampled scenario, seed 0, at `t = 503`, after one dwell on
band 0 at slot 0 and 250 on band 18:

| band | exploration term `(1.5 − visit_density) × staleness`, before | after |
|---|---|---|
| 18, just left | **+0.419** | +0.00084 |
| 0, untouched for 500 slots | **−0.0025** | +1.5 |

The negative came from the `-1` sentinel: a band chosen once and never again kept `-1/600` for the
rest of the episode. Nothing caught this because `_observation()` recomputes from the raw counters
and was always right — only the reward-facing copy was wrong, so the agent's *inputs* were correct
and only its *incentives* were backwards.

**(c) The camping penalty had a free workaround — D53.**

`-1.0 * camp_slots` resets the instant the action changes, so a 2-band ping-pong and a full 36-band
sweep both pay exactly `N_SLOTS`. Four fixed policies, 3 sampled scenarios each (seeds 0/1/2), with
(b) already fixed in both columns:

| policy | ratio | cTTI | coverage | `−1.0 × camp_slots` | `−3.0 × visit_density × n_slots` |
|---|---|---|---|---|---|
| recency (rung 5) | 0.0995 | 6.49 s | 0.782 | −206.6 | **+329.5** |
| round-robin | 0.1179 | 2.47 s | **0.921** | −228.5 | **+324.3** |
| alternate, 2 bands | 0.1985 | 16.55 s | 0.261 | **−218.9** | −505.4 |
| camp one band | 0.2350 | 16.84 s | 0.245 | −89,867 | −1,361 |

**The old term ranked the ping-pong above round-robin.** Coverage 0.261 against 0.921. So there was
no gradient toward sweeping at all, which is the direct explanation for (a)'s entropy of 2.369
after 100k steps: "do not repeat the same band twice consecutively" is the only coherent thing that
reward taught, and a near-uniform policy already satisfies it.

## 9.2 What changed in the code

| file | change |
|---|---|
| `rfenv/env.py` | `_staleness_array` uses the observation's formula (D52); three dead locals holding the *correct* formula removed |
| `rfenv/env.py` | `reward_balance`'s fourth term is `-3.0 * visit_density[action] * n_slots` (D53) |
| `rfenv/rl/common.py` | `RLScheduler` / `RecurrentRLScheduler` take `deterministic`, default `False` (D54) |
| `rfenv/baselines/ladder.py` | DQN factory passes `deterministic=True`; `_seed_torch(rng)` in the PPO and RecurrentPPO factories |
| `tests/test_env.py` | two regressions: staleness pinned to the observation, sweep pinned above ping-pong |

Suite after: **289 passed, 42 skipped, 1 failed** — the failure is
`test_heldout_split_is_refused_without_an_explicit_flag`, unchanged and environment-dependent as
§8.5 records.

**Reproducibility check:** with `_seed_torch`, a sampled rung 9 at the same seed gives a
byte-identical action sequence across runs (verified, seeds 0 and 1), so `EVALUATION.md` §7's
"identical scenarios and seeds" still holds now that inference is stochastic.

## 9.3 The run in flight

> **VOID — killed by D55, see §10.** The observation was rescaled from 147 to 146 while this run
> was training, so its checkpoint cannot load. The command and the reading table below still stand
> as the recipe for the *next* run; only this particular execution of it is gone. Note §10.4's
> correction to the table: **+325 is both the target and roughly the cap.**


Started 2026-09-09, on the corrected reward. This is the first RecurrentPPO run whose exploration
term points the right way and whose camping cost cannot be alternated around.

```
venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo \
  --reward reward_balance --timesteps 1000000 --seed 0 \
  --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=2048 \
  --checkpoint runs/checkpoints/lstm_balance_1M.zip --checkpoint-freq 200000 \
  --run-name lstm_balance_1M --description "first run on the corrected reward_balance"
```

Hyperparameters carried over from `lstm_gamma997` unchanged, deliberately: §4.3 tested them and
found them not to be the problem, and changing them in the same run as the reward would confound
the two. `gamma=0.997` gives a ~333-slot horizon against a 300–600-step episode; `n_steps=2048` is
roughly four episodes per rollout; `ent_coef=0.01` against sb3-contrib's default of 0.0.

Snapshots land at 200k/400k/600k/800k plus the final, so the series is comparable the way §1's
recurrent family is — but this time every one carries a manifest naming its reward (§8.2).

**The number to watch is `ep_rew_mean`**, and it now has a meaningful scale, which is new. Under
this reward the ladder's own rows sit at **round-robin +324.3 and recency +329.5** (§9.1c). So:

| `ep_rew_mean` | reading |
|---|---|
| below 0 | still concentrating airtime; the reward is dominated by the density penalty |
| ~0 to +300 | spreading out, not yet matching the floor |
| **~+325** | **matched round-robin on its own reward** |
| **above +330** | **beat rung 5, the actual bar** (D46) |

`lstm_gamma997` sampled scored +114.0 and +52.7 on single episodes, so that is the gap to close.
Note these are single-episode and 3-seed sampled-scenario figures, **not** the `--seeds 3
--sampled 10` protocol — they set a scale to read training against, they are not ladder rows.

### Result

*Pending — the run was still in flight when this section was written. Fill in from the manifest at
`runs/checkpoints/lstm_balance_1M.json` (wall clock, resolved hyperparameters, sha256) and from the
compare run against the snapshots.*

## 9.4 What this run cannot settle

- **Nothing here is a protocol-scale result.** `EVALUATION.md` §5's rung 9 table is superseded on
  both counts and needs `python -m rfenv.compare --seeds 3 --sampled 10 --figures` re-run against
  the new checkpoint before any row is quoted.
- **D47's selection rule is still un-applied.** D53 shows the new camping term orders two known
  policies correctly; it does not claim `reward_balance` is the right candidate. That is a
  paired-dominance measurement nobody has run.
- **Rungs 7 and 8 remain dead** at 145/146 wide, unchanged from §8.6. Any retrain of them should
  now use `reward_balance` and will need `--reward` passed explicitly.
- **D30 (AoA / pulse width) stays `PROPOSED`** — the human's answer on 2026-09-09 was that neither
  is in scope for now, which closes it as a live question without changing its status line.

---

# 10. The observation rescaled, and what it cost — 2026-09-09

**§9.3's run is void.** It was killed mid-flight by the change in this section, with the cost
stated in advance and accepted: D55 takes the vector from 147 to 146, so that checkpoint could not
have loaded. Nothing in §9.1's diagnosis or §9.2's fixes is affected — those are about the reward
and about inference, and both survive D55 numerically intact (see 10.3).

## 10.1 Why the observation was looked at at all

§9 fixed what the agent was *paid*. This section is about what it was *shown*. Measured over 1,506
round-robin steps and 1,409 rung-5 steps, 3 seeds each:

| block | mean | p99 | max | verdict |
|---|---|---|---|---|
| `hit_rate` | 0.399 | 1.000 | 1.000 | healthy, full range |
| `visit_density` | **0.0278** | 0.065 | 1.000 | bottom tenth of its range |
| `staleness` | **0.0697** | **1.000** | 1.000 | bimodal, not small |
| `current_band` | 0.0278 | 1.000 | 1.000 | one-hot, fine |
| `camp_time` | 0.0020 | — | **0.0033** | **two distinct values** |
| `measured_dbm` | 0.183 | — | 1.000 | discriminates: 0.406 on a declared hit, 0.0135 without |

`visit_density`'s 0.0278 is exactly 1/36 and is not a property of the policy: the block sums to 1
across bands by construction, so its mean is pinned there for **every** scheduler that will ever
run. `staleness`'s p99 landing exactly on 1.0 is the giveaway that it was bimodal rather than
merely compressed — everything visited near zero, everything never-visited on the ceiling.

**The decisive evidence was already in the repository.** `baselines/recency.py` multiplied
staleness straight back out by `N_SLOTS / SWEEP_SLOTS` before scoring with it, and its docstring
records what happens otherwise: *"the rung silently collapses into rung 4: coverage 0.526 against
round-robin's 0.895."* The policy the RL rungs have to beat could not use the feature as shipped.

Two further findings, both verified rather than assumed:

- **`current_band` is exactly redundant.** `argmin(staleness) == argmax(current_band)` on
  **1,506 of 1,506** steps. Kept anyway — recovering it costs the network an argmax over 36 dims,
  and 36 input weights per neuron is the cheaper side of that trade.
- **`camp_time` cannot move under good behaviour.** Exactly `{0.001667, 0.003333}` — one or two
  slots over 600 — because the streak resets on every action change. Its only consumer was removed
  by D53.

## 10.2 What changed (D55)

The vector is **146** wide and its box is **no longer the unit interval**, deliberately:

| block | before | after | 1.0 now means |
|---|---|---|---|
| `visit_density` | slots / elapsed | **× `N_BANDS`** | an equal cut of airtime (ceiling 36.0) |
| `staleness` | (t − last) / `N_SLOTS` | **/ `SWEEP_SLOTS`** | one full pass overdue (ceiling 13.95) |
| `camp_time` | streak / `N_SLOTS` | **removed** | — |

After: `visit_density` mean **1.0000**, p99 2.32; `staleness` mean **0.9720**, p99 13.95. Both now
centre near 1.0 instead of 0.028 and 0.070.

`recency.py` no longer divides — the correction moved into `_observation()`, so the heuristic and
the agent read the same well-scaled number and the agent no longer has to rediscover a constant
that follows from the frozen dwell schedule.

## 10.3 The two invariants, verified

A rescale that quietly moved rung 5 or the reward would invalidate D46 and D53 without anyone
noticing, so both were checked against explicit recomputations in the old units:

| invariant | result |
|---|---|
| rung 5's ranking identical to the pre-D55 formula | **1,408 / 1,408 steps** |
| `reward_balance` per-step vs old-units recomputation | max abs error **1.6e-7** |
| D53's round-robin row | **+324.3**, reproduces exactly |
| D53's camp-one-band row | **−1360.9**, reproduces exactly |

`reward_balance` converts both rescaled inputs back at the top of the function rather than carrying
re-tuned coefficients — that is the only way D53's measurement survives an observation change.
D52's requirement still holds: the reward and the observation read the same quantities, now in the
same units with one visible conversion line.

D53's other two rows move slightly (rung 5 +324.8, ping-pong −509.0). Rung 5 breaks ties randomly,
and the ping-pong's band pair is not recorded in D53, so neither is evidence of a change.

## 10.4 What it cost, and one thing it exposed

**Every checkpoint in `runs/checkpoints/` is dead.** The suite's skip count goes 42 → 66. Rungs 7
and 8 were already unloadable (D49); rung 9's four now join them, as does the killed §9.3 run.
Suite: **265 passed, 66 skipped, 1 failed** — the pre-existing held-out-split guard.

**And the measurement that matters most for the next run — D56.** Checking whether the reward could
tell the ladder's two best policies apart, over 8 seeds:

| policy | reward mean | sd | coverage |
|---|---|---|---|
| recency (rung 5) | +277.2 | 81.6 | 0.768 |
| round-robin | +274.9 | 77.7 | 0.932 |

Per-seed difference **+2.3 ± 11.7**, rung 5 ahead on **4 of 8 seeds**. The signal between the best
deployable heuristic and the floor is about 3% of the reward's own scenario-to-scenario noise.

**So the ceiling on the next run is round-robin, not rung 5.** `reward_balance` separates
catastrophe from competence by a huge margin — camping −1361 against round-robin +324, which is
what D53 fixed and what should stop the agent camping — and barely separates competence from
excellence. An agent that reaches ~+325 has learned everything this reward can teach it. The
§9.3 reading table should be understood that way: **+325 is the target and also roughly the cap**,
and D46's actual Pareto goal is not encoded in the reward at all.

Not acted on, deliberately: re-weighting now would confound a fourth change into the next run, and
D47's paired-dominance rule — the project's own procedure for choosing a reward — has still never
been run. That procedure, not another coefficient, is what should settle it.

## 10.5 Still open after this section

- **D56.** The reward cannot express D46's target. Highest-value open question in the lane.
- **Time since last *hit*, per band.** `_last_hit_slot` is already tracked and still excluded.
  `staleness` says when the agent last *looked*; nothing says when a band was last *active*. Held
  back from D55 on purpose so a retrain stays attributable.
- **D47 has never been run.** Unchanged from §9.4.
- **Rungs 7 and 8** need retraining from scratch at 146 wide, with `--reward` passed explicitly.

---

# 11. Training readiness, measured before the next 1M run — 2026-09-09

§10 left the environment changed and every checkpoint dead. This section is the check that the
next run can actually start, plus the two numbers that say what it can be expected to reach.

## 11.1 The pipeline runs

A 4,000-step RecurrentPPO run on the 146-wide environment, `reward_balance`, seed 0,
`ent_coef=0.01 gamma=0.997 n_steps=2048`. It trained, wrote a checkpoint, and wrote a manifest
beside it. Two readings from its own log worth keeping:

- `entropy_loss = -3.58` at initialisation, which is exactly `ln 36 = 3.584` — a perfectly uniform
  policy. With `ent_coef=0.01` opposing collapse this is the right starting point given D54.
- `explained_variance = 0.036` after 10 updates — the value head has fitted essentially nothing
  yet, which is expected this early and is the number to watch for the reward being learnable at
  all.

## 11.2 What the reward's scale actually allows

Measured this session, 8 seeds, sampled scenarios, `reward_balance`:

| policy | reward | sd |
|---|---|---|
| SB3's first `ep_rew_mean`, untrained | ~+169 | — |
| uniform random | **+211.9** | 68.6 |
| round-robin | **+274.9** | 77.7 |
| recency (rung 5) | +277.2 | 81.6 |

**Total learnable headroom is about 65 points**, from a random policy to the effective ceiling,
against a per-episode standard deviation of about 78. The whole signal is under one standard
deviation of episode-to-episode noise. That is the practical form of D56: the reward is not merely
unable to separate rung 5 from round-robin, it gives the optimiser a narrow target relative to the
noise it has to average through.

**Consequence for `n_steps`.** At roughly 500 steps per episode, `n_steps=2048` is about four
episodes per rollout, so the noise on a rollout's return estimate is about `78/sqrt(4) = 39` —
around 60% of the entire 65-point signal. Raising it to 8192 (~16 episodes) puts that at ~20, and
16384 (~32 episodes) at ~14. Fewer, cleaner updates is the right trade here now that D52 and D53
have made the reward point the right way; the previous runs' problem was never a shortage of
updates.

**Not a decision, and not applied to any recorded run.** This is arithmetic on a measured standard
deviation, offered as a starting point rather than a tuned value. No `n_steps` sweep was run --
that would be tuning against the metric, and §9.3's rule about changing one thing at a time still
applies.

## 11.3 A small API change, recorded because it is easy to misread

`make_train_env` gained a `render_mode` parameter, threaded to `ScanEnv`. It is **inert for
training**: all three trainers call `make_train_env(reward=reward)` and none passes it, and
`model.learn()` never calls `env.render()` regardless. Setting it does not produce frames during
training -- it only makes `render()` legal if something calls it. The path that actually produces a
visual of a policy is `compare --figures`/`--animate`, which writes `animation_<config>.gif`.

Kept rather than reverted: it is the correct hook for a future recording callback, and it changes
nothing today.

## 11.4 The next run

Register the resulting checkpoint as a rung before comparing -- a `.zip` with no `Rung(...)` entry
is not in the ladder, and the intermediate `--checkpoint-freq` snapshots each want their own
lettered rung so the series shows whether the policy is still improving at 1M or plateaued earlier.

The reading table from §9.3 stands, with §10.4's correction: **~+275 is both the target and roughly
the cap.** Stalling near +212 means it is not beating random.

---

# 12. Acceptance scale, a withdrawn measurement, and the leak — 2026-09-10

## 12.1 The acceptance run

`python -m rfenv.compare --seeds 3 --sampled 10 --figures`, 57 scenarios × 3 seeds =
**2,223 episodes**, artefacts in `runs/acceptance_2026-09-10/`. First run at protocol scale since
D52/D53/D54/D55, and the first ever to compute the paired comparison against **rung 5** as well as
the floor — until now only the floor comparison existed, so every RL claim in this repository had
been measured against the wrong reference.

Paired both directions, because `wins on both` is not symmetric — an episode can split:

| rung | dominates rung 5 | dominated by it | neither | ratio |
|---|---|---|---|---|
| 9b · 200k | 43.9% | **4.7%** | 51.5% | **9.3 : 1** |
| 9c · 300k | **48.0%** | 13.5% | 38.6% | 3.6 : 1 |
| 9a · 100k | 30.4% | 11.7% | 57.9% | 2.6 : 1 |
| 9d · 400k | 25.7% | 27.5% | 46.8% | 0.9 : 1 |
| 6a · apfeld | 23.4% | 21.6% | 55.0% | 1.1 : 1 |

Against the floor: 9c **80.1%**, 9b 78.9%, rung 5 67.3%.

**Stated precisely: 48.0% is not a majority.** What the models do is dominate rung 5 far more often
than the reverse, where every heuristic sits at or below 1.1:1. 400k is *worse* than 200k and 300k
— performance peaks around 200–300k and degrades, which is the concrete argument for a
pre-registered selection rule.

**And it is contaminated** (§12.3). Not written into `EVALUATION.md` §5.

## 12.2 D56 was measured wrong

The brief asked for D56 to be reconciled with the result before anything was built on it, listing
three hypotheses. Manifests were read rather than recalled: all three checkpoints trained on
`reward_balance`, obs 146, seed 0, `n_steps=8192`, `ent_coef=0.01`, `gamma=0.997`. **H1 eliminated
by evidence.**

H2 is the answer, and the bug is ours:

| quantity | separation | seeds with rung 5 ahead |
|---|---|---|
| `recency − rung 2` (correct) | **+59.0 ± 19.4** | **8 / 8** |
| `recency − step % 36` (what D56 measured) | +2.0 ± 13.5 | 5 / 8 |

D56 scored a **hand-written `step % N_BANDS` sweep** and called it `round_robin`. Rung 2 is
`EQUAL_AIRTIME_CYCLE` (D43) — equal *airtime* per band, not one dwell per band — and the naive
version hands the seven wide bands twice the airtime, scoring **+57.0 ± 18.5** higher with coverage
0.932 against 0.793. The separation is three times the seed noise, not three percent of it.

D53 and D57 used the same stand-in. Their conclusions survive; only their `round_robin` rows move.
All three entries corrected in place; D56 withdrawn.

**The lesson, now a test:** a rung has a registered implementation for a reason, and a measurement
that substitutes an obvious-looking reimplementation is not measuring the ladder.

## 12.3 The train/evaluation leak (D60)

Training sampled `EmitterPool.from_train()` — every emitter in all 47 development configs — while
evaluation ran those same 47 replays plus scenarios sampled from that same pool. The heuristic
rungs do not train, so the asymmetry ran one way, ours.

Split rule written into `rfenv/split.py` **before anyone looked at which configs landed where**:
order by detectable-emitter count ascending, take every 4th from index 1 into validation.
Systematic along the difficulty variable rather than random, because difficulty spans 2 to 99
emitters and dominates every §4 metric.

| | n | min | median | max | mean |
|---|---|---|---|---|---|
| training | 35 | 1 | 38 | 82 | **40.9** |
| validation | 12 | 1 | 37 | 80 | **40.2** |

It fell balanced and **was not re-drawn after that was seen** — re-rolling until a split looks good
is the error the rule prevents.

Splitting the config list is not enough: the pool is assembled *from* the configs.
`EmitterPool.from_configs` is the constructor that matters. Measured: **2,600 contributions /
1,431 emitters training, 843 / 482 validation, 0 shared.**

## 12.4 The reward screen (D62)

| candidate | rung 2 | rung 5 | rung 4 | rung 6a | sep | camper | verdict |
|---|---|---|---|---|---|---|---|
| `reward_balance` | 217.9 | 276.9 | **−414.1** | 278.7 | 3.0σ | 7.3σ | **PASS** |
| `weighted` | 1455.7 | 1817.2 | 1323.0 | 1928.4 | 2.6σ | 0.4σ | FAIL |
| `hit_z` | 196.4 | 299.1 | **409.8** | 376.6 | 3.1σ | −2.3σ | FAIL |
| `hit_y` | 165.4 | 270.5 | **372.4** | 332.0 | 2.6σ | −1.9σ | FAIL |
| `greedy` | 459.0 | 776.6 | **1198.3** | 984.2 | 2.7σ | −2.7σ | FAIL |
| `explore` | 1277.0 | 1238.1 | −205.2 | 1034.0 | **−2.5σ** | 31.5σ | FAIL |

**`hit_z` and `hit_y` — D29's original two — both fail.** They rank rung 4 above every sweeping
policy. Every DQN and PPO run in this repository trained on one of them (D48). The screen costs
minutes and would have saved all of it.

## 12.5 Selection, and its first execution

D61: highest `dominates − dominated` against rung 5 on the validation half, ties toward fewer
steps. The rule was first proposed as a win-rate rule together with a false claim about its
behaviour; the discrepancy was raised and net dominance ratified in its place.

| checkpoint | steps | dominates | dominated | net |
|---|---|---|---|---|
| `lstm_balance_1M_s2` | 200k | 55.6% | 8.3% | **+47.2%** ← selected |
| `lstm_balance_1M_s3` | 300k | 36.1% | 5.6% | +30.6% |
| `lstm_balance_1M_s1` | 100k | 36.1% | 11.1% | +25.0% |

**Selects nothing** — all three predate D60 and trained on these emitters. It demonstrates the rule
executes. Note 200k and 300k tie on win rate at 36.1%, which under the rejected version would have
fallen to the tie-break.

## 12.6 Where this leaves the lane

- **No clean RL number exists.** Every checkpoint predates D60.
- ~~**The human declined a retrain** on 2026-09-10 (cost), and will train `greedy` to 500k
  instead.~~ **Superseded by §14, same day** — a clean retrain on `reward_balance` was run after
  all, and a second, on `reward_balance_improved`, followed it as a paired comparison. Neither
  trained `greedy`.
- **Still open:** the iteration ledger (one row per run, including failures, with a command to
  re-run any past run from its manifest alone), and D47, which has never been run. The ledger is
  now started — see `docs/project/ITERATION_LEDGER.md` — beginning with the D60-forward runs;
  every run before D60 trained on a leaking pool and is narrated above instead of re-tabulated.

---

# 13. `hit_z`/`hit_y` retired from `REWARDS` — 2026-09-10

Following directly from §12.4's screen: `reward_hit_z` and `reward_hit_y` (D29's original two
candidates) removed from `rfenv/env.py` entirely, by direct edit, confirmed when raised. `REWARDS`
now holds four keys: `reward_balance`, `greedy`, `explore`, `weighted`. Recorded as **D63**, with
D62 (the screen that produced the finding) cross-referenced forward to it.

**Consequence, not cause.** D62 measured both at −2.3σ and −1.9σ against the 1.0σ camper-margin
bar — both rank the camper above every sweeping policy. Every DQN and PPO checkpoint in this
repository trained on one of the two, but both were already permanently unloadable from D49's
observation-width change, so nothing currently loadable is lost.

**Blast radius, all fixed, suite green (323 passed / 60 skipped / 1 pre-existing failure):**

- Two `test_env.py` tests pinned D31's per-slot invariant to `hit_z` specifically, since it was a
  bare per-slot `Z` count and none of the four survivors is. Rewritten to inject a raw reward
  function directly via `env._reward_fn`, testing the environment's slot-summation mechanics
  independent of registry contents — more durable than depending on which candidates exist.
- `test_reward_gate.py`'s known-bad control swapped from `hit_z` to `greedy` (D57's exploit corner,
  which fails the same check by construction).
- ~35 other test failures were generic fixtures that happened to default to `reward="hit_z"` with
  no dependency on its shape; swapped to `DEFAULT_REWARD`.
- `rfenv/baselines/ladder.py` (rungs 7/7a/8 docstrings), `rfenv/rl/__init__.py` (usage examples,
  which used `--reward hit_z` as a copy-pasteable command that now raises), `run.md`,
  `ENVIRONMENT_SPEC.md`, `EVALUATION.md`, `EDGE_LANE_HANDOFF.md`, and the three RL handoff docs'
  amendment banners all corrected.

**Not touched**: the deep historical worked-examples inside the three long handoff documents
(day-by-day task tables, file-tree listings) — those already carry a top-of-document amendment
banner saying to read it before trusting anything below.

---

# 14. The first clean retrain, and a paired reward comparison — 2026-09-10

## 14.1 The control run

`reward_balance`, 400k timesteps, under D60/D61 for the first time — the earliest run in this
repository to start (commit `176e6a9`) after **both** the split (D60, `36405b1`) and the reward
screen (D62/D63, `36405b1`/`46d8449`) were already in the tree. Every RL checkpoint before this one
predates at least one of the two.

```
venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo --reward reward_balance --timesteps 400000 \
  --seed 0 --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
  --checkpoint runs/checkpoints/lstm_balance_clean.zip --checkpoint-freq 100000 \
  --run-name clean_lstm --description "attempt 1 under D60 split, D61 selection"
```

Four snapshots, `clean_lstm_s1`–`s4` at 100k/200k/300k/400k, 4,419 s wall clock for the full run
(~74 min).

## 14.2 Selection (D61), run for real

Against the 12 validation configs, 3 seeds, paired per episode against rung 5 — the rule fixed in
`rfenv/selection.py` before this run existed:

```
venv/Scripts/python.exe -m rfenv.selection runs/checkpoints/clean_lstm_s1.zip \
  runs/checkpoints/clean_lstm_s2.zip runs/checkpoints/clean_lstm_s3.zip \
  runs/checkpoints/clean_lstm_s4.zip --reward reward_balance
```

| checkpoint | steps | dominates | dominated | net |
|---|---|---|---|---|
| `clean_lstm_s4` | 400k | 47.2% | 11.1% | **+36.1%** ← selected |
| `clean_lstm_s2` | 200k | 41.7% | 11.1% | +30.6% |
| `clean_lstm_s1` | 100k | 36.1% | 11.1% | +25.0% |
| `clean_lstm_s3` | 300k | 19.4% | 19.4% | +0.0% |

Monotonic in steps except a dip at 300k (net dominance drops to zero, then recovers at 400k) —
noted, not chased; the rule picks the checkpoint it picks. **This is the first result this
repository has produced that is both clean (post-D60 pool) and pre-registered (D61's rule, not a
number picked after looking at the comparison).**

## 14.3 What was still outstanding for the control arm at this point in the session

D61 answers "which checkpoint," not "how good is it." The headline — `compare.py`'s full run over
all 47 stare replays plus sampled scenarios, on `clean_lstm_s4` specifically — had not yet been run
here. It has since (§14.6 below). An earlier comparison (`runs/clean_baseline_100k`, `--seeds 1 --rungs
round_robin,recency,lstm_balance_clean_100k_500k,lstm_balance_200k_1M,lstm_balance_300k_1M
--figures`) was run *before* selection completed, scored the **100k** snapshot only (rung `10a`
was still registered against `clean_lstm_s1.zip`, not the eventual winner), and mixed it with two
pre-D60 checkpoints (`lstm_balance_200k_1M`, `lstm_balance_300k_1M`) that trained on the pool these
same 47 replays are drawn from. Read on its own terms: `lstm_balance_clean_100k_500k` (i.e.
`clean_lstm_s1`, 100k steps) scored 74.5%/68.1%/51.1% (ratio/cTTI/both) paired against recency —
ahead of both contaminated checkpoints despite an order of magnitude less training — but that run
is **not** a substitute for scoring the D61-selected winner, and the two contaminated rows in it
are not evidence about anything (same status as every pre-D60 number: sound arithmetic,
contaminated comparison).

`rfenv/baselines/ladder.py` registers all four checkpoints as their own rungs, `10a`–`10d`
(`clean_lstm_s1`…`s4` respectively). `10d` is the D61-selected checkpoint, and §14.6 below is that
comparison run for real.

## 14.4 The treatment run, launched in parallel

To settle whether `reward_balance_improved` (§ below D63, in `rfenv/env.py`) helps a learner rather
than just passing the D62 screen more narrowly, the human launched a second
run identical to the control in everything but the reward:

```
python -m rfenv.rl.recurrent_ppo --reward reward_balance_improved --timesteps 400000 --seed 0 \
  --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
  --checkpoint runs/checkpoints/lstm_balance_improved.zip --checkpoint-freq 100000 \
  --run-name lstm_balance_improved --description "treatment: density-weighted occupancy, D60 split"
```

Same seed, same hyperparameters, same split — the only variable that differs from §14.1 is the
reward function. **This is a single seed per arm.** A difference between the two arms is
suggestive, not conclusive: the control run's own trajectory swings by double digits between
adjacent 100k checkpoints (§14.2), so an effect smaller than that swing is inside seed noise, not
evidence about the reward.

## 14.5 The treatment arm, selected

Finished the same day. D61 selection on its four checkpoints, same rule, same validation set:

| checkpoint | steps | dominates | dominated | net |
|---|---|---|---|---|
| `lstm_balance_improved_s2` | 200k | 36.1% | 11.1% | **+25.0%** ← selected |
| `lstm_balance_improved_s4` | 400k | 33.3% | 11.1% | +22.2% |
| `lstm_balance_improved_s1` | 100k | 22.2% | 11.1% | +11.1% |
| `lstm_balance_improved_s3` | 300k | 36.1% | 27.8% | +8.3% |

Selected `lstm_balance_improved_s2` (200k) — **below** the control's +36.1% (§14.2) on validation.
Both arms registered on the ladder: `10a`–`10d` (control), `11a`–`11d` (treatment).

## 14.6 The headline, both winners together — and a reversal

```
venv/Scripts/python.exe -m rfenv.compare --rungs round_robin,recency,lstm_balance_clean_400k,\
  lstm_balance_improved_200k_400k --seeds 3 --sampled 10 --figures --out runs/clean_paired_comparison
```

684 episodes (4 rungs × 57 scenarios × 3 seeds), the full development set per D60. Paired against
recency:

| scheduler | ratio | cTTI | both |
|---|---|---|---|
| `round_robin` | 7.0% | 28.7% | 3.5% |
| control (`reward_balance`, 400k) | 73.7% | 31.6% | 22.8% |
| treatment (`reward_balance_improved`, 200k) | **83.0%** | **39.8%** | **31.0%** |

Both clean checkpoints clear rung 5 comfortably. **The treatment is ahead on the headline; the
control was ahead on validation.** Not read as "the improved reward wins" — one training seed per
arm, the winning checkpoints sit at different step counts (400k vs 200k), and the control's own
checkpoints alone swing by more than the 8.2-point gap between the two arms across their four
snapshots. It is the first evidence, weak as it is, in the direction argued when the reward was
called "unproven rather than rejected": the D62 screen measures discrimination between six fixed
heuristics, which is a different signal than the gradient a learner actually consumes. Settling it
for real needs matched seed counts per arm — see D65 for the full accounting.

---

# 15. A crash, a fix it exposed, and the observation grows to 183 (D67) — 2026-09-10

## 15.1 The matched-seed attempt that didn't survive

Following D65, the plan was to settle the control/treatment reversal with matched seed counts:
`clean_lstm_seed1`/`seed2` and `lstm_balance_improved_seed1`/`seed2`, four runs, launched together
in the background against the 146-wide observation. The laptop crashed partway through. No
checkpoint from any of the four had reached its first 100k snapshot (~15-20 min in, well short of
the ~18 min/100k pace observed on every prior run), so nothing was salvageable and nothing was
lost beyond the wall-clock time -- confirmed by `runs/checkpoints/` holding no `seed1`/`seed2`
files after the crash. The instruction that followed: train one model at a time in the background
from here on, not several in parallel.

## 15.2 Option A lands before the retry (D67)

Discussed the same day, before the crash: whether the trained RecurrentPPO's never-camps behaviour
(D64-D66) came from a missing signal or a reward that doesn't pay for using one. Agreed plan:
Option A over Option C (a jointly-learned phase/band action, `docs/project/PHASE_SWITCH_FUTURE_WORK.md`)
-- add the streak signal, keep the current architecture, retrain. Confirmed spec: both the scalar
(current-band streak) and the full per-band block together, `146 -> 183`, rather than doing it in
two separate width-bumping passes later -- "make any changes necessary to the observation space
now" to minimise future churn.

Landed in `rfenv/env.py`/`guard.py`/`rl/common.py` (full account: D67). The crash happened with
this code already on disk but before it had been tested end-to-end; re-verified after restart:
`ScanEnv` builds a self-consistent 183-wide box, `reset()`/`step()` populate `hit_streak`
correctly, `current_observation_width()` reads it off `guard.py` without a hardcoded literal.

## 15.3 What the first test run after the crash actually found

`pytest tests/test_baselines.py`: **85 failures**, not the clean skips a stale-checkpoint change is
supposed to produce. Cause: `_unusable_checkpoint`'s hand-maintained rung-number -> checkpoint-path
table was written for rungs 7-9d and never extended as 9e-9j/10a-d/11a-d were registered across
this session -- so every checkpoint outside that table hit the observation-width mismatch as a bare
test failure instead of a skip, even though `load_checkpoint()` already raises the exact
informative D49 error via `require_loadable()`. Fixed by building each rung directly inside the
check and catching `(FileNotFoundError, ValueError)`, removing the table: self-healing for any rung
registered after this one, nothing left to fall behind again. Full suite after: **287 passed / 144
skipped / 1 pre-existing failure** (up from 60 skipped -- every now-stale RL checkpoint skips
cleanly instead of erroring).

## 15.4 The retrain, restarted

One model at a time this time. Control arm first:

```
venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo --reward reward_balance --timesteps 400000 \
  --seed 0 --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
  --checkpoint runs/checkpoints/lstm_balance_d67_control.zip --checkpoint-freq 100000 \
  --run-name lstm_balance_d67_control \
  --description "control: reward_balance, D60 split, D67 observation (183-wide, hit_streak)"
```

Treatment (`reward_balance_improved`) queued to start only once this one finishes, per the
one-at-a-time instruction. This folds the never-run matched-seed question into the same retrain --
there was no clean 146-wide answer to preserve, so the question is answered on the current
observation rather than a superseded one.

## 15.5 Both arms finished, and the answer to the actual question

Control finished first (401,408 timesteps, ~58 min single-run), D61-selected `s3` (300k), net
dominance +25.0%. Treatment launched immediately after as the sole background job, finished
cleanly, D61-selected `s1` (100k), net dominance **+33.3%** -- ahead of the control on validation,
where under the 146-wide setup (D65) the treatment had lost.

**The question Option A exists to answer, checked directly rather than inferred from a metric:**
does either selected checkpoint actually commit to a band now that it can see a streak signal?
Same method as D66 -- run an episode, measure the logged band sequence's run-lengths, on
`config_81`/`config_2`/`config_921`. Both checkpoints: **maximum dwell streak 4-6 slots, on every
scenario checked.** Identical order of magnitude to every pre-D67 checkpoint (D64/D66 measured max
6). The feature exists in the input; neither policy learned to act on it by staying put.

**The headline still moved, a little, in both arms, in the same direction as D65:**

```
venv/Scripts/python.exe -m rfenv.compare --rungs round_robin,recency,\
  lstm_balance_d67_control_300k_400k,lstm_balance_d67_treatment_100k_400k \
  --seeds 3 --sampled 10 --figures --out runs/d67_paired_comparison
```

| | control | treatment |
|---|---|---|
| both, D67 (183-wide) | 25.7% | **35.1%** |
| both, pre-D67 equivalent (D64/D65, 146-wide) | 22.8% | 31.0% |

Both arms up a few points; treatment still ahead of control, same direction D65 found on a wholly
independent pair of checkpoints. **Read carefully, not triumphantly.** The mechanism this change
was built to produce -- commitment -- did not appear on either checkpoint, so the few-point gain
cannot honestly be credited to it. Equally plausible: 17 extra input dimensions gave the network
marginally more capacity for reasons that have nothing to do with streaks, or this is ordinary
single-seed noise -- D64's own four snapshots swung by 36 points on this exact measure. What can be
said without hedging: the streak feature has not, on either checkpoint tried so far, taught a
policy to camp. Full accounting in D67.

## 15.6 D47, run for the first time (D68)

Rather than another architecture experiment, the next question asked was procedural: three
decision-cycles (D62, D66, D67) had shipped since D47 -- the actual pre-registered rule for
*choosing* between reward candidates -- was written, and it had never once executed. Checked D62
eligibility fresh (`python -m rfenv.reward_gate --rewards reward_balance,reward_balance_improved`,
this session, not trusted from an earlier unrecorded claim): both pass, `reward_balance` at
3.0σ/7.3σ, `reward_balance_improved` at 2.3σ/5.6σ. Then applied D47's rule -- paired-both against
round-robin, largest fraction wins, within 5 pp nothing is selected -- to the same D61-selected
checkpoints and the same `compare.py` run §15.5 already produced, no new run needed:
`reward_balance` 65.5%, `reward_balance_improved` 64.3%. **1.2 points apart. No candidate
selected**, exactly as D47's own text says should happen at this margin. Full accounting in D68.

## 16. PulseWidth, AoA and a second observation layout (D71, 2026-09-14)

D30 had sat open since 2026-09-04 -- the case for AoA (96.7%/86.1% attribution accuracy, the
only observable separating a new find from a re-find) and PulseWidth was made and measured, but
the trigger this entry set for itself (a trained agent failing to explore for want of it) was
never separately checked before this was approved directly.

**What made this different from D49/D55/D67:** every one of those widened the observation in
place and killed every existing checkpoint doing it. This time the ask was explicit -- keep both,
switchable -- so `ScanEnv._observation_blocks()` was restructured to return every block as a named
dict, and `OBS_LAYOUTS["v1"|"v2"]` (`rfenv/env.py`) picks and orders the subset that actually ships
into the flat vector SB3 sees. "v1" is the old 183, byte-for-byte (verified: an identical seeded
episode run on both "v1" and "v2" envs in parallel produces identical values on every block they
share). "v2" is 326 -- PulseWidth and AoA sin/cos, both gated on a declared hit the same way
`measured_dbm` already is, plus amplitude upgraded from one global last-dwell scalar to a per-band
block that actually persists (`measured_dbm`'s real weakness, that it forgets every band but the
one just left the instant the agent moves on).

**The AoA encoding decision:** stored as `(sin θ, cos θ)`, not a raw angle -- 359° and 1° are the
same bearing, and a raw value would teach the network otherwise. `(0.5, 0.5)` (the rescaled form of
raw `(0, 0)`) turned out to be a free, provably-unreachable "never measured" sentinel: every real
angle satisfies `sin²+cos²=1`, so the origin is off the unit circle and no genuine reading can ever
land there. Not designed in up front -- noticed once the encoding was written, then tested directly.

**What this is not:** D30's own measurement found AoA's real value is telling a new emitter from an
already-seen one, which needs comparing a bearing against a *set* of bearings already logged for
that band -- clustering, not a single reading. That is not built. "v2" ships one raw last-bearing
number per band and nothing more.

**Checkpoints, same session:** `runs/checkpoints/` had grown to 150 tracked files in one flat
directory -- ladder.py's own path literals were the only thing keeping it navigable. Reorganised
(`git mv`) into `runs/checkpoints/v1/<model_name>/`, 23 families, grouped by stripping trailing
snapshot suffixes (`_sN`, `_NNN_steps`, `_NNNk`/`_NNNM`) off each filename's stem. `v2/` is the
sibling directory for anything trained on the new layout. `require_loadable()` had a real latent
bug surfaced by this: it checked a checkpoint's recorded width against one hardcoded "current"
value (always "v1"'s), so the very first "v2" checkpoint saved would have been unloadable through
the normal path the moment anyone tried. Fixed: `known_observation_widths()` returns every width
`OBS_LAYOUTS` defines, and the check accepts any of them now.

**Training, launched, not yet finished:** `lstm_balance_v2_control_seed2` -- `reward_balance`, seed
2, 400k timesteps, `ent_coef=0.01`, `gamma=0.997`, `n_steps=8192`, otherwise an exact mirror of
`lstm_balance_d67_control_seed2` (rung 17c's own training run) except `obs_version=v2`, so the
comparison once it lands is like-for-like. Expected wall clock ~40 minutes, based on that same
config's own recorded 2398.8 s on this machine. `runs/checkpoints/v2/lstm_balance_v2_control_seed2/`.
Full accounting in D71.

## 17. `pulse_count` widens "v2" to 362, and its first result (D72, D73, 2026-09-14 to 2026-09-18)

**The widening.** Same session as §16, a second direct request: literal `C` in the observation was
asked for first, and had already been measured and declined the same session
(`PDW_COMPLETENESS_AND_BAND_DENSITY_BRIEF.md` -- `C` carries no gamma gate at all, so it counts
sub-threshold contributions a real receiver never resolved). The second ask -- gate it by `Y`, the
same rule already applied to PulseWidth/AoA -- is different and is what shipped: a sixth "v2"-only
block, `pulse_count`, appended after `aoa_cos`, `log1p`-normalised against the same
`_DENSITY_REF_PULSES = 64` constant `reward_balance_improved` already uses for `C` on the reward
side. "v2" moved from 326 to 362. Cost: the three 326-wide snapshots §16's own retrain had already
produced (`lstm_balance_v2_control_seed2_s1/s2/s3`, 385,024/400,000 steps, never finished) are now
permanently unloadable -- the same tax D49/D55/D67 already charged, paid again on purpose. Full
account: D72.

**Two fresh 362-wide runs, from scratch** (the dead 326-wide run could not be resumed into a wider
vector): `lstm_balance_v2_d72_seed2` (`reward_balance`, rung 20d, 401,408 steps, one uninterrupted
run) and `lstm_balance_improved_v2_d72_seed2` (`reward_balance_improved_v2`, rung 21a, 404,800 steps,
survived two laptop crashes via `RecurrentPPO.load()` + `learn(reset_num_timesteps=False)` from the
last `--checkpoint-freq` snapshot each time -- the crash-resume pattern §15 established, reused
without incident).

**The result (D73, `MEASURED`, single seed).** `python -m rfenv.compare
--rungs round_robin,recency,lstm_balance_v2_d72_seed2_400k,lstm_balance_improved_v2_d72_seed2_400k
--seeds 3 --sampled 10 --figures --obs-version v2 --out runs/d72_paired_comparison` -- 684 episodes.
`reward_balance` (20d): 0.111 ratio / 3.30 s cTTI / 36.3% paired-both-vs-recency.
`reward_balance_improved_v2` (21a): 0.136 / 2.82 s / **45.6%** -- ahead on every column, the same
direction D65 found on "v1" between the un-widened pair. **Not read as settled** -- one seed per
arm, same caveat D65 gave and never resolved, and not directly comparable to the "v1" D64/D65 numbers
(width, reward formula and the sampled-scenario draw all differ at once). Neither row promoted over
the other.

## 18. A band-priority reward, "v2p", 398 wide -- built, measured null, tried stronger, same answer (D74, 2026-09-18/19)

**Not a continuation of D70.** D70 ("the scheduler takes a threat priority from outside; it does not
compute one") sourced its priority vector from an external threat-classification library and had its
own partial implementation (`rfenv/threat.py`, priority-reweighting in `rl/common.py`) deleted --
uncommitted, unexplained, discovered this session when a related request was raised, and left
unresolved after the user chose to cancel that line of work rather than dig into the deletion. This
feature is a **separate, fully-specified, library-free request**, deliberately dropping the threat
library entirely: `band_priority` is synthetic, sampled fresh each episode from `ScanEnv`'s own RNG
(3-6 of 36 bands elevated to priority `3.0`, the rest at `1.0`), not read from any emitter-type table.

**What shipped.** A fourth observation layout, `OBS_LAYOUTS["v2p"]` -- "v2" (362) plus one more
36-wide block, `band_priority`, appended last (398 total). Paired with it, `ScanEnv.step()` adds a
reward term directly, after whichever `REWARDS` candidate already ran and before `total_reward +=`:
`reward += priority_coef * band_priority[dwell.band] * len(newly)`. **Gated on discovering something
new (`newly`), not on occupying the band** -- a per-slot multiplier would reopen D53's camping
exploit (a policy paid repeatedly just for sitting on a high-priority band). This is additive, not a
new `REWARDS` entry, and does not touch `reward_gate.py`'s D62 screen -- composable with whichever
reward is selected, verified by a bit-for-bit reward-identity test with the term disabled.

**The control arm, and why it has to run the same reward term, not a switched-off one.**
`band_priority` defaults to `1.0` even on ordinary bands, so the reward term fires on *every*
discovery, not only ones on elevated bands -- a uniform scale increase, plus a differential for
elevated ones. `priority_uniform=True` keeps the same scale increase but removes the differential
(every band stays `1.0` all episode), which is what makes it a genuine control rather than a
before/after comparison that would confound "the agent used the signal" with "the reward got
bigger." `ScanEnv.__init__` refuses `band_priority=True` against any layout that doesn't carry the
block (i.e., anything but "v2p").

**Two RecurrentPPO checkpoints, matched pair, same split/hyperparameters/seed as the D72 pair
(§17):** `lstm_balance_v2p_priority_seed2` (treatment, `priority_uniform=False`, rung 22a) --
401,408 steps. `lstm_balance_v2p_uniform_seed2` (control, `priority_uniform=True`, rung 22b) --
400,000 steps, launched automatically after the treatment run finished. Both restarted from scratch
on GPU partway through this work (see below); only ~50k CPU steps existed at that point, not worth
resuming across the device change. **Both complete.**

**GPU/CUDA added the same session, on direct instruction.** Real hardware (GTX 1650) was present but
torch was CPU-only (`2.14.0+cpu`); code-level `device` support was added first (safe, no risk to the
then-running job), and the dependency swap itself was deliberately held until asked directly whether
to stop the running job and do it now -- confirmed, then `TaskStop` on the CPU run, `torch==2.14.0+cu126`
installed, verified with a direct CUDA matmul and the full `test_rl.py` suite (27/27), both training
runs relaunched from scratch. Measured fps: ~95-248 on GPU (settling ~93-95 once the model warms up)
against ~67-88 on CPU for this specific LSTM policy -- a real, if modest, win, contrary to SB3's usual
caution that GPU rarely helps a small MLP-sized network.

**`compare.py` gained matching flags** (`--band-priority`, `--priority-coef`, `--priority-n-bands`,
`--priority-uniform`), threaded through `run_one`/`compare`/`figures` into `ScanEnv(...)` --
initially global-only, so evaluating the treatment and control checkpoints against round_robin/
recency needed two separate `compare.py` invocations, one per arm, each with its own `--out`. Fixed
later the same session (see "per-rung priority resolution" below).

**The comparison ran 2026-09-18** -- two separate runs (round_robin + recency + one rung each, 513
episodes apiece: 3 rungs x 57 scenarios x 3 seeds):

| rung | scheduler | ratio | cTTI (s) | coverage | beats recency on both |
|---|---|---|---|---|---|
| 2 | round_robin | 0.060 | 4.18-4.20 | 0.864-0.865 | -- |
| 5 | recency | 0.109-0.111 | 3.34-3.36 | 0.887-0.891 | -- |
| 22a treatment | 0.120 | 3.28 | 0.891 | 44.4% |
| 22b control | **0.128** | **2.70** | **0.908** | **61.4%** |

**The control beat the treatment on every column** -- the opposite of what the feature is meant to
show. The obvious hypothesis, raised directly in conversation, was that the treatment agent camps on
its priority bands, starving the rest of the spectrum. Checked directly rather than assumed: 20
episodes with the treatment checkpoint, measuring `env._visit_density_array` split by
`env._band_priority > 1.5` (elevated) vs not. Result: **1.076** mean fair-share airtime on elevated
bands against **1.007** on ordinary ones -- about 7% more, not camping, and the direction was
inconsistent episode to episode (several episodes showed *less* attention to elevated bands, not
more). Camping was already structurally unlikely by design -- `reward_balance`'s own
`-3.0 * visit_density * n_slots` term prices concentrated airtime, and the priority bonus only pays
on a *new* discovery, never on occupying the band -- and this measurement confirms it didn't happen
anyway. Read together with the pre-training magnitude estimate (the priority term is roughly 4-5% of
total episode reward, worked out before this pair was trained, in response to the same "will this
even move the needle" question), the likelier explanation is a signal too small and inconsistent to
learn from reliably -- added training noise, not a harmful bias. Single seed, single comparison run,
not read as settled (same caveat D65/D73 give and do not resolve).

**Per-rung priority resolution, same day, prompted by wanting the priority animation (below) to
"just work" without remembering CLI flags.** `Rung` (`ladder.py`) gained its own
`band_priority`/`priority_uniform`/`priority_coef`/`priority_n_bands` fields (default `False`/unused
-- no effect on any existing rung); rungs 22a/22b are tagged with their real training config.
`compare.py`'s new `resolve_priority_kwargs(key, ...)` lets a rung's own metadata override whatever
the caller passed on the CLI, used inside `run_one`. This closes a real, previously-silent risk:
forgetting `--band-priority` when evaluating a priority-trained checkpoint used to mean it got
scored under an all-ones vector -- matching the control condition -- with no error raised. It also
means 22a and 22b can now sit in the *same* `compare.py` invocation; re-running the comparison this
way reproduced the exact same numbers as the two-invocation version, confirming the refactor changed
nothing about existing behaviour. 271 passed, 162 skipped, `test_compare.py`/`test_baselines.py`/
`test_render.py`.

**A "priority animation," on direct request.** `render.compare_animation` takes an optional
`band_priority` array and draws a translucent stripe across every row for each elevated band, static
for the whole clip since the assignment doesn't change mid-episode. `compare.py`'s `figures()` now
writes a second file, `priority_animation_config_*.gif`, automatically whenever a compared rung
resolves to real (non-uniform) priority -- driven by the rung's own tag, not a separate flag. The
control arm never elevates a band, so it never triggers this; verified both ways (present for a run
containing 22a, absent for one containing only 22b or only "v1" rungs).

**A follow-up question, raised directly: does treatment simply explore more broadly than
control -- not camping on priority bands specifically, but more diffuse overall?** Checked, not
assumed: same 20 episodes, entropy of the final `visit_density` distribution across all 36 bands
(not split by priority this time). Treatment **3.255** against control **3.051** (max possible
`ln 36 = 3.584`); bands touched meaningfully (`visit_density > 0.5`): treatment **24.95** against
control **18.95**, out of 36. Confirmed -- treatment genuinely spreads its attention across about 6
more bands per episode than control. Plausible mechanism, not yet proven on its own: control's
`band_priority` input is a *constant* every single episode (all-ones, always) -- nothing to
condition on, so from an information-theoretic standpoint that block carries zero per-episode
signal, and control's whole training problem is stationary in a way treatment's genuinely is not
(a different elevated-band pattern every episode). A harder, non-stationary problem converging to a
less focused policy in the same 400k-step budget is a training-difficulty story, distinct from
"control learned to decode a constant" (which isn't possible -- a value that never varies carries
no information to decode).

**Permutation ablation, run 2026-09-19 -- the decisive check.** Method: the trained treatment
checkpoint, 30 episodes, each run *twice* on the identical scenario/seed/receiver-noise draw (a
separate RNG permutes `band_priority` across bands; `env.np_random` itself is never touched, so
nothing else about the episode differs) and the identical torch sampling seed for both runs of a
pair (`torch.manual_seed`, matching `_seed_torch`'s convention in `ladder.py`) -- so if behaviour
still diverges between the two runs, it is attributable to the observation difference alone, not to
fresh independent sampling noise. `env._observation()` is rebuilt immediately after the shuffle, so
the very first observation the policy sees is consistent with it too (the array `reset()` already
returned would otherwise be stale).

Real `band_priority` in one run of each pair, the same values permuted across bands in the other:

```
                          corr(airtime, true priority)   ratio   cTTI (s)   coverage
real priority fed                  +0.018                0.111    2.83      0.892
shuffled priority fed              +0.019                0.114    2.58      0.903
real beat shuffled: 13/30 episodes (43% -- a coin flip)
```

Correlation between where the agent actually spends its time and where the true priority is turns
out to be statistically indistinguishable whether the fed signal is real or garbage, and none of
the three headline metrics degrade when the signal is corrupted -- if anything the reverse, by a
margin small enough to be noise. **The agent never learned to use `band_priority` at all.**
Treatment's underperformance against control is therefore a training-difficulty story (previous
paragraph), not a misused-signal story -- both possibilities were live going in, and this is what
separates them.

**`DECISIONS.md` D74 written up 2026-09-19, `MEASURED`.** Unlike D71/D72 (§16/§17), written up as
`SETTLED` the day they were built, ahead of any trained result, this write-up was deliberately held
until both the control-arm comparison and the permutation ablation existed, on direct instruction.
**Not adopted, not promoted, code not removed** -- nothing defaults to `band_priority=True`
anywhere, so the mechanism stays registered and available. Unlike D66's phase-switch gate (a
specific mechanism measured actively worse than its own baseline, removed after being measured),
this is a null result at one configuration (`priority_coef=0.5`, 3-6/36 bands, sb3-contrib's
default LSTM capacity, 400k steps, one seed) with named, untried paths -- a larger coefficient or
elevation multiplier, more training steps, more network capacity -- not scoped or recommended here,
a genuinely new experiment if picked up again, not a revision of these numbers. Full account:
`DECISIONS.md` D74; mechanism: `OBSERVATION_SPACE.md` §2.4; numbers: `MODEL_COMPARISON.md`
Width 398. The camping, spread and permutation-ablation diagnostics above were scratch scripts, not
committed to the repository -- reported here in full, with method and numbers, per this
repository's own provenance rules on what counts as a measurement.

**Same day, requested directly: try it stronger before concluding anything from one configuration.**
Bigger discovery bonus, a genuine per-slot "camp a little on the priority band" incentive despite the
D53 risk that phrase raises, and more capacity/budget for the retrain.

**The D53 risk, worked through before writing any code.** A literal per-slot occupancy bonus is
exactly what D53's own exploit was built on -- `camp_slots`, a *consecutive*-dwell counter, reset to
zero the instant the action changed, so a two-band ping-pong paid nothing close to what sustained
camping should have. The fix has to anchor to something that does not reset on a band switch. D55
already solved this once, for `reward_balance`'s own camping cost: `visit_density`, cumulative
fair-share airtime across the *whole episode*, immune to being gamed by alternating away and back.
Reused directly: `decay = max(0, 1 - visit_density_value / occupancy_decay_cap)`. Checked, not just
argued -- a new test rolls out a two-band ping-pong against a full 36-band sweep under the occupancy
term alone (`priority_coef=0.0`) and asserts the ping-pong does not out-earn the sweep. It doesn't,
comfortably: `visit_density` on a heavily concentrated pair of bands saturates within 1-2 visits
(measured directly: two bands split evenly reach ~16-18 fair-share units almost immediately, an
order of magnitude past any sane decay cap), so the occupancy term self-extinguishes for both bands
almost as fast as literal camping would, and faster than it would for a policy spreading attention
across many bands instead.

**A new function, not a change to any existing reward.** `priority_reward_bonus` (`env.py`) --
`discovery` unchanged from D74's original formula; `occupancy` new, both terms scaled by
`band_priority`'s own *value*, not a binary "is this elevated" gate. That last detail matters for the
same reason D74's original discovery term used the same convention: the control arm's constant `1.0`
still has to fire both terms at the same uniform scale, or the comparison stops isolating what it's
meant to isolate. `priority_high` (the elevated value, was the hardcoded module constant
`_PRIORITY_HIGH`) is now a `ScanEnv` kwarg too, needed to raise it past 3.0 for this run.

**A real bug, found by actually running `--check-env` with the new flags rather than assuming they'd
compose cleanly.** `observation_space`'s declared ceiling for `band_priority` was still reading
`_BLOCK_SPECS`'s static 3.0, so `priority_high=5.0` produced episodes whose real values (up to 5.0)
fell outside the box the env itself declared -- `gymnasium`'s own `check_env` catches this
(`AssertionError: ... not within the observation space`), SB3 would not have, at least not loudly.
Fixed by making the declared ceiling for that one block track `self._priority_high`; regression test
added so it can't silently regress again.

**Training config:** `policy_kwargs={"lstm_hidden_size": 512}` (SB3-contrib's own kwarg name,
confirmed via `inspect.signature` before trusting it, then confirmed again on the *loaded* checkpoint
-- `model.policy.lstm_actor.hidden_size == 512`, not just the flag that was passed), doubled from the
default 256. `--timesteps 800000`, doubled from 400k. `priority_coef=2.0` (was 0.5), `priority_high=5.0`
(was 3.0), `occupancy_coef=0.3`, `occupancy_decay_cap=6.0`. A short smoke test (2048 steps) ran first,
end to end including the JSON-quoted `--hyperparam policy_kwargs=...` flag on this shell, before
committing to an 800k-step run on the strength of an untested command line.

**Two runs, same matched-pair discipline:** `lstm_balance_v2p_priority_strong_seed2` (treatment,
rung 23a, 802,816 steps, ~103 fps settling to ~95-97) and `lstm_balance_v2p_uniform_strong_seed2`
(control, rung 23b, 800,000 steps, launched automatically after, ~90-106 fps). Roughly 4 hours
combined wall clock on this machine -- reported honestly as "how much train left" was asked, from the
raw log file directly rather than the task-output tool's occasionally-stale snapshot (a lesson from
earlier this session, applied consistently this time).

**The comparison, same 513-episode design as before:** 23a **0.154 / 2.43s / 0.907 / 73.1%**
beats-recency-both -- the best figure measured anywhere in this project, ahead of D68's own settled
best (17c, 54.4%). 23b (control) **0.121 / 2.74s / 0.911 / 45.6%**. Treatment beat control by a wide
margin -- the *opposite* direction from the original D74 pair.

**Taken alone, that result would read as the mechanism finally working. It doesn't survive a second
ablation.** Same method as the first (30 episodes, identical scenario/seed/receiver-noise draw per
pair, identical torch sampling seed per pair, real `band_priority` in one run and the same values
shuffled across bands in the other): correlation between airtime and true priority **+0.031 real,
+0.031 shuffled** -- identical to three decimal places. Real beat shuffled in 12 of 30 episodes (40%,
below even odds if anything). Performance did not degrade when the signal was corrupted (ratio 0.145
vs 0.148, cTTI 1.80 vs 1.82, coverage 0.960 vs 0.929). Airtime on elevated bands was a little more
tilted than the original run's (1.129 vs 0.970, ~16% more, against 1.076 vs 1.007 the first time),
but the ablation shows directly that the tilt doesn't track the true signal -- noise wearing the
shape of a pattern, not a pattern.

**Read together: 23a's lead over 23b is single-seed training-run variance, not the priority mechanism.**
A checkpoint shown, directly, not to condition its behaviour on a signal cannot be winning *because of*
that signal. The direction flipped completely between the two pairs -- control ahead the first time,
treatment ahead the second -- while the one thing actually measured both times, whether the agent
reads `band_priority` at all, came back "no" on both occasions. That consistency is the finding; which
arm happened to score higher on a single seed each time isn't.

**One thread left open on purpose, separate from the priority question:** a 512-wide, 800k-step
checkpoint beat every smaller/shorter checkpoint measured in this project, on both arms. Whether that
holds with `band_priority` removed entirely -- a plain capacity/budget question, nothing to do with
priority -- is untested; neither 23a nor 23b isolates it, since both still carry the mechanism, just
with real vs. uniform values. Not scoped here.

**Verdict unchanged: not adopted, not promoted, code not removed.** The named untried paths from the
first write-up -- bigger coefficient, more capacity, more training -- have now been tried together, at
several times the original scale, and the answer didn't move. `DECISIONS.md` D74 was extended the
same day with this follow-up, not given a new decision number -- it is the same question, answered
again, more carefully. Full account: `DECISIONS.md` D74; numbers: `MODEL_COMPARISON.md` Width 398.
`tests/test_band_priority.py` grew to 26 tests (the occupancy formula in isolation, the D53 ping-pong
check, the `observation_space` ceiling regression); full suite re-run clean (470 passed, 1
pre-existing unrelated failure, 27/27 RL, 26/26 band-priority) before any of this was trusted.

## 19. Making it online: "v3", missions that do not end, and a view (D75/D76/D77, 2026-09-19)

Three requests in one: make the recurrent policy an *online* learner, let episodes run as long as we
like, and let us actually watch the environment while it runs.

Worth writing down first: the starting premise was half-right in a useful way. A recurrent policy
**already is** an online learner — its hidden state integrates hits and misses across a mission and
changes behaviour with no gradient step at all. That is the RL² result, and the most literal answer
to the PS's "absence of prior reliable intelligence". What this repository had never done is give
that recurrence the two inputs the construction depends on: what the agent last did, and what came
back.

### The Y/Z conversation, which changed the design

The request was initially to feed the *training* reward into the observation, on the reasoning that
a real scheduler does not hypothesise an emitter and then scan — it just scans and either gets data
or does not, so `Y` and `Z` should be the same thing for a slot it looked at.

That is right about what a scheduler does and wrong about this simulator, and the gap is
**sensitivity**, not hypothesis. `receiver.py`'s entire detection model is one line,
`Y[b,t] = 1 iff S[b,t] + n >= gamma`. An emitter can be transmitting into a band the receiver is
staring straight at and still go undeclared, because its received level sits below gamma — too weak,
too far, or in an antenna null. Re-read from `runs/d74_followup_treatment_comparison/metrics.json`
while answering this: **Pd = 0.8421**. Roughly one truly-occupied cell in six that you look directly
at comes back silent. And Pfa = 1.35e-3 in the other direction.

The decisive argument was the PS's own figures of merit. If `Y == Z` then Pd = 1 and Pfa = 0 by
construction, and "probability of detection", "probability of false alarm" and "sensitivity" — three
things the problem statement names explicitly as deliverables — all go degenerate.

So the design split. `reward_balance` (reads `Z`) stays exactly as it was and still trains and scores
every arm: D29 untouched, its source bytes untouched. A second function, `reward_balance_obs`, is the
same formula with `Y` substituted, written as
`reward_balance(dataclasses.replace(dwell, Z=dwell.Y), ...)` so the two cannot drift apart. *That* is
what the policy sees. Ordinary asymmetric actor-critic — privileged critic, deployable actor — and it
is what keeps a "v3" rung `deployable=True` rather than a reference line beside `oracle_pulse`.

Why it matters concretely: the agent already holds `hit_rate`, `visit_density`, `staleness` and
`n_slots` in its vector. Hand it `reward_balance`'s value and it can solve the remaining term for
`0.5 * dwell.Z.sum()` — and thereby learn about emitters it never detected. Strictly more than any
real instrument has.

### The awkward finding, found before training rather than after

"v3" is "v2p" + `prev_action` (36) + `prev_reward` (1) + `prev_hit` (1) = 436. While writing it I
checked what `_prev_action` actually holds at the point `_observation()` runs, and found `step()`
assigns `self._current_band = action` (line 1026) and `self._prev_action = action` (line 1108) from
the same value in the same step.

**So `prev_action` is bit-identical to the existing `current_band` block at every step after the
first.** `prev_hit` is likewise just `current_hit_streak > 0`. 36 of the 38 new columns restate
blocks that were already there. They differ only in the cold-start sentinel: `current_band` reads
one-hot at band 0 after `reset()` — claiming a dwell that never happened — while `prev_action` reads
the zero vector, off the one-hot simplex and unreachable by any real action.

I kept the layout, because the RL² interface is conventional and 38 columns is cheap, but two things
follow and both are now in D75, `OBSERVATION_SPACE.md` §2.5, `CLAUDE.md` and a test. **`prev_reward`
is the only genuinely new information in "v3".** And **an ablation corrupting `prev_action` alone
would read null by construction** — so the pre-registered ablation corrupts it together with
`current_band`, and carries a `hit_rate` positive-control arm. That control is the direct lesson of
D74: a null tells you nothing unless you have shown the instrument can detect a positive.

### Long missions, without touching the freeze

`N_SLOTS` and `EPISODE_S` live in `constants.py`, which D42 froze and `tests/test_freeze.py` pins
with per-value literals plus a digest tripwire; D25 says moving anything there re-validates from gate
1 and re-runs every baseline. So `episode_slots` is a per-*environment* length defaulting to
`N_SLOTS`, and the frozen file is untouched.

Before changing anything I took golden sha256 digests of all seven checkpoint-free rungs' per-slot
logs and of the full v1/v2/v2p observation byte streams, at `d4f4361`, and landed them as
`tests/test_backward_compat_hashes.py` in the first commit. "Bit-identical" is not checkable by
reading a diff, and this change touched `env.py`, `truth.py`, `receiver.py`, `artefacts.py`,
`apfeld.py` and `ladder.py`. The digests held through all of it.

Three things the long path exposed that the short one never could:

- **`clock` and `staleness` would have left their declared Box.** At 72,000 slots a neglected band
  reads 1,674 sweeps against a declared ceiling of 13.95, and the clock reads 120. The clip has to go
  at *both* sites that write staleness — the observation's and the reward's — or the reward stops
  pricing the number the agent sees, which is the promise D52 exists to keep. It provably never binds
  at 600: the most stale a band can be is 599/43 = 13.930 against 600/43 = 13.953.
- **`episode_metrics()` had a latent correctness bug**, not just a scaling issue. It censored a
  missed emitter at the *full episode*, so an hour-long mission would charge a segment-0 miss 3,600 s
  and the number would stop being comparable with every 30 s figure here. Now censored at the
  emitter's own segment end, which evaluates to exactly 600 at the default.
- **`EmitterContribution.cells` is int16.** 32767 // 600 = 54 segments = 27 simulated minutes, past
  which offsetting slots would wrap silently rather than fail. Upcast to int32 in the stitched copies.

Tiling one recording to fill an hour was rejected outright — it would hand the agent an exactly
periodic world to memorise — so long worlds are stitched from independent draws and `ScanEnv` refuses
`episode_slots > N_SLOTS` with a fixed `scenario=`. The cost, stated in `stitch`'s own docstring
rather than buried: emitter identity is per segment, so there is no temporal continuity across a
seam; at every 600-slot boundary the world is replaced at once.

Measured before building: ~0.65 MB per segment, so **~78 MB per simulated hour**, and an hour-long
grid builds in under a second. The per-slot log turned out to be the other memory term (~20 MB/hour
of dict overhead), hence `log_window_slots` — and a windowed env is *refused* by `write_run` rather
than writing a header claiming a length its rows do not have.

### The arithmetic that shaped the fine-tuning design

The request included gradient updates at deploy time. The honest constraint is arithmetic: a 30 s
episode is 300–600 decisions, and these checkpoints trained at `n_steps=8192`. One mission cannot
fill a fourteenth of a single rollout, and at `gamma=0.997` the effective horizon (~333 steps) is
comparable to the whole episode, so the advantages would be dominated by the terminal bootstrap. A
per-mission update is not a small update, it is a noisy one.

So `rfenv/rl/online.py` does not offer one. Fine-tuning exists only on a continuous grid, at
`n_steps=2048` — about two simulated minutes, ~30 updates per simulated hour — pre-registered in D77
rather than tuned afterwards, with a conservative `learning_rate=1e-5`, `clip_range=0.1` and
`target_kl=0.02`, because adaptation starts from a policy that already works and the risk is wrecking
it. `ent_coef=0.01` is retained specifically because D54's documented failure mode for this rung is
collapse onto one band.

SB3 owns the loop. The recurrent rollout buffer and GAE-through-LSTM-state are already correct in
sb3-contrib, and reimplementing `collect_rollouts` is where a silent bug would live. One detail there
fails silently and took a moment to get right: `n_steps` is baked into the checkpoint and read while
the rollout buffer is constructed inside `_setup_model()`, so it must go through
`RecurrentPPO.load(custom_objects=...)`. Assigning `model.n_steps` after loading leaves the old
buffer and the run quietly collects 8192 transitions per update. There is now a test for exactly that.

This is also the repository's **first resume path**. `train()` has always built a fresh model;
nothing before this called `learn(reset_num_timesteps=False)`. Rung 21a's description records a human
doing the equivalent by hand after two laptop crashes.

The two tests I trust most here: a tiny fine-tune moves the weights **and** advances `num_timesteps`
from the checkpoint's own count (so it is a continuation, not a restart), and **at
`learning_rate=0.0` the weights come back bit-identical** — the no-op control that rules out the
harness perturbing the policy by any route other than the optimiser.

### The view

`rfenv/live.py` is numpy and stdlib only, and has to stay that way — `rfenv/render/__init__.py` calls
`matplotlib.use("Agg")` at package import, so anything under `rfenv/render/` both pulls in matplotlib
and lands on a non-interactive backend. Hence the split: the ANSI heat strip at the top level, the
matplotlib panel in `rfenv/render/live.py`, imported lazily.

The first real run of it was more useful than expected — recency's sweep is immediately visible as a
diagonal drift, and the seven wide bands (0, 1, 6, 7, 17, 18, 19) stand out because their dwells draw
two columns instead of one. Degradation is ordered and tested: not a tty means **zero escape bytes**
(that stream is usually a log file, where control codes are corruption), `NO_COLOR` or a console that
refuses virtual-terminal mode means ASCII glyphs, and no interactive matplotlib backend means a
warning and the terminal view — a plotting backend must never be able to kill a one-hour run.

### Where this stands

All three are `BUILT`, none is `MEASURED`. 79 new tests; full suite green apart from the known
pre-existing `test_heldout_split_is_refused_without_an_explicit_flag` (the 45 held-out pairs are not
on this machine). The matched pair — "v2p" control against "v3" treatment, identical but for
`--obs-version`, three seeds per arm because D68 escalated over a 1.2 pp gap — and the four-arm
ablation are specified in D75 and not yet run. **Nothing in this section is a claim about whether any
of it works.**

### 19.1 `prev_action` removed the next day, and two live views got a real redesign (2026-09-20)

Two follow-ups landed the day after §19, both from direct questions rather than anything I set out
to build.

**The question that actually mattered: doesn't the LSTM already store the previous action?** Yes —
and the code already half-admits it. `current_band` (every layout since "v1") is set from `action`
in the same line `_prev_action` is, so the recurrent state has always had direct, unmediated access
to the last band tuned. `prev_action` was never a new channel, only a second copy of one that already
existed, and §19 already said so in these words when "v3" first shipped: *"`prev_reward` is the only
genuinely new information in this layout."* Asked to justify keeping the redundant block anyway,
there wasn't a good one, so it came out — `OBS_LAYOUTS["v3"]` narrows from 436 to 400 wide, in place,
the same class of change D49/D55/D67/D72 made before it. `lstm_v3_seed0` and `lstm_v3_seed1`, both
complete 800k-step runs, are now permanently unloadable. The seed-0 comparison already run against
them stays on record as what was measured on the old shape; nothing about it is retracted, it just
cannot be extended.

`prev_reward` survives the same scrutiny for a different reason, worth stating precisely rather than
by analogy: an LSTM's hidden state can only carry forward what appeared in its *input* at some point,
and no layout before "v3" ever put the raw per-step reward there — `hit_rate`/`hit_streak`/`staleness`
are running aggregate statistics, not the scalar the policy is actually optimised against. The RL²
argument for handing it over explicitly anyway (Duan et al. 2016) is about gradient path length, not
about information the architecture otherwise lacks a route to — and D74 is the standing reason not to
trust that argument on faith: it already showed, twice, that this exact setup does not reliably learn
to use even a stronger, equally-unavailable-elsewhere signal. The ablation this was always going to
need is unchanged in what it tests, only in the header row: four arms now, not five, since the arm
that tested `prev_action`'s redundancy no longer has a block to corrupt.

**Separately, and unrelated: the live views only ever showed the receiver's own log.** Asked directly
to colour and label where the emitters are, and colour hits and misses — genuinely different requests
that exposed the same gap. Three states (unseen/looked/hit) could never show an emitter that hadn't
been scanned yet, and collapsed "looked, heard nothing" and "never looked" into one colour, so a
missed detection and a gap in coverage looked identical. Six states now, truth crossed with
declaration, both views. Picking colours for them the `dataviz` skill's way (its validated status
palette, not eyeballed) caught something worth having caught before shipping it: true hit and missed
detection — the two states that matter most in the whole picture — measured 4.1 OKLab Delta E under a
simulated deuteranopia, under even the 6.0 floor, the classic red/green collision landing on exactly
the pair carrying the most meaning. `node` wasn't on this machine to run the skill's own validator, so
its math got ported to Python and run directly rather than skipped. Every state has its own glyph now,
in both the coloured and the ASCII stream, so nothing depends on hue alone. One more thing turned up
while wiring the terminal counts into the fine-tuning path: `_callbacks()`'s check for "was a real
view asked for" used `isinstance(view, NullView)`, which is true for every real view too, since both
backends subclass it to share its no-op `open`/`close`. Online fine-tuning with `--view light` had
been running and silently drawing nothing this whole time. Fixed with an exact type comparison,
caught only because a smoke test finally looked for actual output instead of a clean exit.

### 19.2 Two architecture ablations on rung 9, and a second narrowing of "v3" (D78/D79, 2026-09-20)

Rung 9's policy has been `MlpLstmPolicy` since it existed — `FlattenExtractor` is a no-op on an
already-flat observation, so the architecture every checkpoint has ever trained on is really
`obs -> LSTM -> actor/critic`. Two requests, same day, each asking whether a different piece of
structure ahead of the LSTM would help.

**D78 — a flat MLP first.** `obs -> 2-layer LayerNorm MLP (256) -> LSTM -> actor/critic`,
`MlpFeatureLstmPolicy`, registered opt-in via `--policy`. Trained the matched pair against the
already-trained `lstm_v2p_ctrl_seed0` (reused as the control, not retrained — identical config,
differing only in `--policy`). **The baseline won, decisively for one seed**: paired against
recency, control beat treatment 62.4% to 46.8% on beats-recency-both, three times the 5pp margin
this repo treats as real. Ratio was essentially tied; the gap was almost entirely censored intercept
time — the deeper network was measurably slower to first-detect, not worse at eventually covering
the spectrum. A convergence check (training-reward curve flattened by ~82k of 800k steps; evaluating
the run's own 200k/400k/600k/800k snapshots found no clean late-training climb) said this looked like
a genuine plateau, not an unfinished run — and a follow-up comparison of the "best-looking" 600k
snapshot against rung 23a (the project's strongest checkpoint) **corrected** that snapshot's own
apparent edge rather than confirming it: on the full 47-config/3-seed set the 600k snapshot scored
worse than the final 800k one, not better. A 12-scenario probe, it turns out, isn't enough to rank
snapshots by, only enough to say training has stopped moving.

**D79 — represent each band, then pool.** A more structural ask: since the observation already *is*
11 interleaved per-band arrays (in "v3") plus a few global scalars, make that explicit instead of
handing the network one undifferentiated vector. `obs -> shared per-band encoder -> mean pool across
bands -> concat encoded global features -> LSTM -> actor/critic`, `BandEncoderLstmPolicy`. The
interesting part wasn't the network, which is two lines of `nn.Sequential` — it was the gather:
the flat vector interleaves *blocks* (`hit_rate[0:36]`, `visit_density[0:36]`, ...), not *bands*, so
turning it into `(36, n_band_features)` needed a precomputed strided index, not a reshape, or it
would have silently mixed one band's `hit_rate` with a different band's `staleness`. Built
`rfenv.env.band_layout()` to compute that index purely from `_BLOCK_SPECS`'s declared widths —
per-band exactly when a block is 36 wide, global otherwise, nothing hardcoded by name — checked
against synthetic values that encode their own source block before trusting it near anything real.
One shared encoder, not 36: `nn.Linear`/`nn.LayerNorm` already only touch a tensor's last dimension,
so calling it on a `(rows, 36, K)` tensor applies the same weights to every band for free, checked
directly (parameter count independent of band count; permuting which band holds which feature vector
leaves mean-pooled output unchanged). No training run yet.

**Then, asked directly: what does "v3" even add if I'm training offline, and can we strip it down to
just `prev_reward` and try that first?** `prev_hit` was always the weaker of "v3"'s two additions —
§19.1 already called it "pre-existing information... lacking the same direct duplicate" `prev_action`
had, and kept it anyway on that weaker footing. Asked plainly whether that was a reason to keep it or
just an excuse, it came out too: `OBS_LAYOUTS["v3"]` narrows a second time, 400 -> 399, so a first
training run tests exactly one hypothesis (does the raw per-step reward help) instead of two tangled
ones. No checkpoint cost this time — nothing had ever been successfully trained on the 400-wide shape
it replaces. Also asked for: band-priority off for this run (not literal zero — `band_priority`'s
declared range is `[1.0, priority_high]`, so zero would violate `ScanEnv`'s own `observation_space`
the same way D74's `priority_high` bug once did; "off" still means the reset default, all-ones,
uninformative but in-bounds), and the architecture to use is D79's brand-new `BandEncoderLstmPolicy`
— untested combination, both the narrowed observation and the representation architecture unproven
together, picked deliberately over the safer one-variable-at-a-time choice. Training launched at
23a's own scale (512-wide LSTM, `n_steps=8192`, 800k steps, `reward_balance`) so a result, if it
comes, is comparable to the project's best-known number. Full accounts: D78, D79, D75's second
amendment.

### 19.3 The run finished, mid-training checks kept getting overturned by the next one, and a real dataset property fell out of asking why (2026-09-20)

The seed-2 BandEncoder run (§19.2) finished at 66.7% beats-recency-both — second only to 23a's
73.8%, and ahead of every other no-priority checkpoint measured tonight. Checked at 300k (mid-run,
a quick 12-scenario probe) it already read 66.7%; checked properly at 800k (the full 47-config/3-seed
set) it read exactly the same 66.7%. Training reward had stopped climbing by ~250k steps and spent
the rest of the run oscillating in a band (203 → 250 → dip to 220 at 491,520 steps → back up to
234–262) rather than settling flat the way D78's run did — noisier, but the same underlying story:
whatever this run was going to learn, it had mostly learned it well before 800k.

**Asked to compare every checkpoint-freq snapshot against 23a, not just the final one** — a fair
question, since the small 12-scenario probe had already been shown (D78's own §19.2 story) to pick
the wrong snapshot once. All 16 snapshots, properly scored: none beat 23a. The best of the whole run
was 70.2%, reached twice (100k, then again at 400k/450k) — 3.6 points short of 23a, and the final
checkpoint (66.7%) wasn't even this run's own best. A real dip shows up at 500k (57.4%, the worst
point in the whole series).

**Asked why 23a keeps winning, given its own priority mechanism was already shown twice not to be
read at all (D74) — so what's actually different about it?** Checked directly rather than guessed:
built the truth grid for all 47 comparison scenarios and summed occupancy per band. **Twelve of the
36 bands have never once had a pulse in any of the 47 scenarios** — bands 0, 13, 14, 25–30, 33–35,
zero exceptions. Checked the scan replays of the same 47 configs too: eleven of the same twelve
match exactly; band 0 is the one difference, dead in every stare recording but carrying real traffic
in scan (427,078 pulses across the 47 files) — unexplained, not chased further.

Then the natural next question: does airtime on those twelve bands track the actual score gap?
Pulled `episode_log.csv` from the comparison runs already sitting on disk (no retraining, no new
eval) and summed dwell time on the dead twelve, per checkpoint: **23a spends 1.73% of its airtime
there; the seed-2 BandEncoder run, 4.73%; the plain "v2p" baseline, 9.94%; D78's MLP-feature run,
12.53%.** The same order as the scores, every time. Inside the seed-2 run's own 16 snapshots,
dead-band airtime and score correlate at r = −0.48, and the single worst snapshot on each measure
(500k steps) is the same one. **This, not priority, not architecture, looks like most of what
separates these checkpoints**: how reliably each one has learned that a third of the spectrum is
simply never worth checking.

**Requested: a second seed, before building real priority on top of an unconfirmed result.** Ran
seed 0 of the identical config. It finished at **51.8%** — 15 points below seed 2's 66.7%, and
below even the plain baseline's 62.4%. Checked its dead-band airtime expecting the same explanation:
it wasn't. Seed 0 wastes only 5.64% on the dead bands, barely more than seed 2's 4.73% — nowhere
near enough to explain a 15-point gap. The real difference is censored intercept time on the *live*
bands (2.70 s vs 1.97 s) — a second, still-unexplained source of run-to-run variance this session
didn't get to the bottom of. Averaged, the two seeds put this architecture at ~59.3%, *below* the
single-seed plain baseline it was supposed to be beating. Stopped here, on request, rather than
chasing a third seed or building priority on an unconfirmed number.

**What this leaves as the actual finding of the night, separate from any one architecture's score:**
interception ratio (and `reward_balance`'s own airtime-shaping terms, which price the identical
behaviour during training) cannot tell "learned to prioritise spectrum that's plausibly worth
watching" apart from "learned exactly which of this one finite synthetic dataset's 36 dwell
frequencies its generator happens to never populate." Both look the same in the numbers. Whether the
twelve dead bands are a fact about a real operating environment or an artifact of this dataset's own
construction isn't answerable from the 47-config development set alone, because every checkpoint
here trained and was scored against the same population. The 45-config held-out split would answer
it directly, and has not been touched. Written up as D80, and as a third entry beside D14's two
traps in `EVALUATION.md` §4 — not a code or environment change, a caution about what the existing
numbers have been able to hide.

### 19.4 D77 finally got run for real, and the two speeds turned out to be different things (2026-09-20)

D77 (online fine-tuning) has existed since the day before yesterday and had never once been run to
completion. Asked directly to actually do it: build a world where the model's own known blind spot
(the ten bands it gives zero airtime to, from D80) becomes the *only* place anything happens, and
watch how long it takes to stop ignoring them.

**Built a synthetic world by hand.** Not from the real dataset -- a fabricated `EmitterPool`, two
emitters per target band, transmitting almost continuously, strong enough that a miss is essentially
physically impossible (10 standard deviations above the detection threshold). Every other band left
completely silent. `band_priority` held at its constant, uninformative default throughout -- whatever
happened had to come from hits and misses, nothing handed to it.

**First surprise: the frozen model, untouched, already handled it.** Watching it live with no
training at all: nothing in the first 30 seconds, 2 finds in the next 30, then 18-20 out of 20 new
emitters every single 30-second window from 90 seconds onward, for the full half hour measured. That's
the recurrent hidden state doing exactly the thing D75 always claimed it could -- adjusting within a
single mission, no gradient step, for free. Genuinely fast, and genuinely already there.

**Then online fine-tuning ran for the first time ever, in two pieces.** The first session got
Ctrl-C'd partway through -- on request, specifically to check whether progress was actually being
saved along the way. It wasn't: `online.py` only ever saved on a clean stop, so a crash or a closed
terminal would have thrown away everything. Fixed properly (`checkpoint_freq`, on by default here,
unlike offline training where it's opt-in) rather than just noted and left. The second session then
resumed cleanly from exactly where the first one stopped -- the whole point of building the fix
before continuing rather than after. Total: about 26 minutes of real time, about 72 minutes of
simulated mission time, across the two sessions.

**The result that actually mattered: tested cold, on a mission it had never seen, not a continuation
of anything it trained on.** 314 hits out of 600 slots in the first 30 seconds, no ramp-up at all.
That's the number worth sitting with next to the frozen model's own 90-second warm-up, because they
are not the same kind of fast. The frozen model's speed is real but it resets every single time a new
mission starts -- it has to rediscover the pattern from nothing, every time, for free, in about 90
seconds. What the 26 minutes of actual training bought was making that discovery permanent -- baked
into the weights themselves, available instantly on any future mission, no 90-second tax paid ever
again. Conflating "it reacts fast within a mission" with "it learned something" would have missed
that these are genuinely two different mechanisms doing two different jobs, one free and one that
needed real gradient steps.

**One more thing caught along the way, on myself.** Building a fourth colour into the animation
GIFs (correct silence, alongside hit/miss/false-alarm) turned up a real mistake in how I'd been
reading my own printouts a few messages earlier -- I'd counted "any declaration" as a hit, which
quietly mislabelled a genuine false alarm (one real event, slot 162, band 18, in the very first
frozen-model segment) as a success. Caught only by going back and checking the actual rendered pixel
data directly instead of trusting the summary line I'd already told the user. Fixed, and said so
plainly rather than letting the earlier wrong claim stand uncorrected. Written up as D81.
