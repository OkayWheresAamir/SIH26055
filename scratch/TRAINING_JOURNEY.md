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
- **The human declined a retrain** on 2026-09-10 (cost), and will train `greedy` to 500k instead.
  `greedy` **fails** D62's screen at −2.7σ, so the screen predicts a camper — rung 4's profile.
  That run is therefore the screen's **first falsification test** and is worth doing as a control
  with the prediction recorded first. It also runs under D60 automatically, since
  `make_train_env` now defaults to the training half.
- **Still open:** the iteration ledger (one row per run, including failures, with a command to
  re-run any past run from its manifest alone), and D47, which has never been run.

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
