# Notes — negative results and the training log

`RL_TEAM_HANDOFF.md` §16.4 E-7 asks for the negative-results list. This project already treats
those as pitch material and has four (D36, D43, D44, D45). Here are the RL lane's.

Every number below was produced by a command run on 2026-09-09 and is cited to its artefact.
Nothing here is quoted from memory or from a transcript.

---

## E-6 — Which component earns the gain: none of them, and that is the finding

The ladder is the ablation. Over **2,223 episodes** (`runs/baselines`,
2026-09-09T20:30:50Z — 57 scenarios × 3 seeds × 13 rungs):

| rung | scheduler | ratio | cTTI (s) | coverage | **both** |
|---|---|---|---|---|---|
| 2 | `round_robin` — the floor | 0.0605 | 4.18 | 0.865 | — |
| 5 | `recency` — **the bar** | 0.1104 | **3.20** | **0.897** | **70.2%** |
| 6a | `apfeld_active_rfs` | 0.1320 | 4.32 | 0.860 | 53.2% |
| 6 | `apfeld` | 0.2455 | 14.86 | 0.367 | 4.1% |
| 4 | `camper` — the degenerate exploit | 0.2088 | 9.67 | 0.497 | 1.8% |
| 9b | Recurrent PPO, 200k | **0.2265** | 16.26 | 0.252 | **0.0%** |

**Rung 5 is still the answer.** A one-line index policy — `argmax(hit rate + gap in sweeps)` —
Pareto-dominates the floor on 70.2% of episodes. Nothing added after it improves on that: Apfeld's
published adaptive strategy manages 4.1%, its own ablation 53.2%, and the RL rungs 0.0–1.8%.

**What earns the gain is adaptivity that stays cheap.** Rung 5 wins because it keeps moving while
responding to what it has seen. Every rung that scores a *higher interception ratio* than rung 5 —
Apfeld, the camper, and all four RL rungs — does so by concentrating airtime, and pays for it in
intercept time and coverage.

---

## E-7 — The negative results

### N-1. Reinforcement learning rediscovered the camper

**The result.** Trained on this environment, a Recurrent PPO agent converges on the strategy rung 4
exists to demonstrate is available: park on the band with the densest observed activity and stay.
Its scorecard is rung 4's, not rung 5's — high ratio, ruinous intercept time, collapsed coverage.

**Why it is worth reporting.** Rung 4 was put in the ladder deliberately, before any RL work, to
show that a single headline metric can be gamed (D14). Rung 9 is the measurement that **an agent
will in fact game it** when the ladder permits. That is a statement about the reward and the metric
pair, not about the algorithm, and it is why `EVALUATION.md` §4 prints all three metrics together
and never one alone. An RL row reported on interception ratio by itself would read as beating every
heuristic in the table.

**Evidence.** `runs/baselines/comparison.md`, rungs 4, 9a–9d.

### N-2. The prescribed hyperparameter fix did not break the camping

**The hypothesis**, and it was a good one: γ_RL = 0.99 gives a ~100-step effective horizon against
a 300–600-step episode, `n_steps = 128` is a fifth of an episode per rollout re-fit ten times, and
`ent_coef = 0.0` leaves nothing in the loss opposing entropy collapse. This is also the exact fix
`RL_TEAM_HANDOFF.md` §18's worked example prescribes.

**The run.** γ_RL = 0.997, `ent_coef` = 0.01, `n_steps` = 2048, 100,352 steps, seed 0, reward
`reward_balance`, 18 min CPU. `runs/checkpoints/lstm_gamma997.zip`, sha256 `2a56a5395f97246d…`,
manifest beside it with the exact command.

**The result: it camped harder.** Under `deterministic=True`, band 3 for 598 of 600 slots,
coverage 0.0526, ratio 0.0069 — worse on every metric than the untuned 100k control (band 6 for
600/600, coverage 0.2632, ratio 0.0548).

**Caveat that matters:** see N-3. Measured the other way, this is the *least* camped policy of the
five tested. The hyperparameters are not exonerated, but they are not the leading suspect either.

### N-3. The camping is an inference artefact, not a learned policy — probably

**What was believed.** That the policies had collapsed during training, on the evidence that
`ent_coef = 0.0` throughout and that 100k/200k/1M produced metrics identical to four decimals.

**The tell that contradicted it.** `entropy_loss = -2.58` in the diagnostic run's own log, against
a `ln(36) = 3.58` maximum. A collapsed policy does not have a distribution that broad.

**The measurement.** Same checkpoint, same episode (`config_2` stare, seed 0), changing only
`deterministic`:

| checkpoint | inference | distinct bands | longest streak | coverage |
|---|---|---|---|---|
| `lstm_ppo4_100000` | `deterministic=True` | 1 / 36 | 600 slots | 0.2632 |
| `lstm_ppo4_100000` | sampled | **27** / 36 | 20 slots | **0.7895** |
| `lstm_gamma997` | `deterministic=True` | 2 / 36 | 598 slots | 0.0526 |
| `lstm_gamma997` | sampled | **32** / 36 | 4 slots | **0.7895** |

The learned distribution is broad; `deterministic=True` takes its argmax, and the argmax is one
band. **Every RL number in this repository was produced under `deterministic=True`**, so all of
them measure the argmax of a broad distribution.

**Why "probably".** One episode, one scenario, one seed. This is a signal, not a result. It needs
the same `--seeds 3 --sampled 10` treatment the table in E-6 got — roughly 15 minutes of compute —
before it can be claimed. **It is the highest-value measurement outstanding in this lane.**

### N-4. Nothing recorded what a checkpoint was, and it cost the result

Four Recurrent PPO training series sit on disk with **identical seeds, identical recorded
hyperparameters, and four different sets of weights**. The variable between them can only be the
reward function — which SB3 does not serialise, and which no log, transcript or commit records.

**Consequence:** D-5 (which reward) cannot be answered, D47's selection rule has nothing to select
between, and artefact A-2 cannot be handed over. The checkpoints exist and are useless as evidence.

**The fix, built 2026-09-09.** Every checkpoint any trainer writes now carries a `<name>.json`
manifest — argv, git commit + dirty flag, reward, **observation width**, hyperparameters actually
passed, seed, timesteps, wall-clock, hardware, versions, SHA-256 — and `load_checkpoint()` refuses
a width mismatch up front with the rebuilding command in the message. See `run.md`.

### N-5. An observation change silently killed six checkpoints

D49 moved the observation vector 109 → 145 → 146 → 147 in a single session. SB3 sizes a policy's
input layer at construction, so **every width change is a retrain, not a reload**. All six DQN and
PPO checkpoints predate it and cannot execute; rungs 7 and 8 therefore have no row in
`EVALUATION.md` §5 at all.

Worse, the failure was silent: `.load()` binds no environment and so checks nothing, and the
mismatch surfaced only much later inside `predict()`, mid-comparison, as a bare shape error naming
neither the checkpoint nor the cause. Restoring the six stale checkpoints to disk turned ~30 clean
test skips into failures. Fixed by the same manifest work — and for pre-manifest checkpoints, by
reading the recorded `observation_space` out of the archive itself.

---

## Training log

Every Recurrent PPO run on record. Series A–D were trained outside any session transcript; their
rewards are `[UNRECOVERABLE]`, which is N-4.

| series | checkpoints | timesteps | γ | ent_coef | n_steps | reward | trained |
|---|---|---|---|---|---|---|---|
| A | `recurrent_ppo`, `lstm_ppo_{100,200,300}k` | 100k–1,000,064 | 0.99 | 0.0 | 128 | `[UNRECOVERABLE]` | 09-09 07:37–08:34 |
| B | `lstm_ppo_{100,200}k` | 100k–200k | 0.99 | 0.0 | 128 | `[UNRECOVERABLE]` | 09-09 09:17–09:36 |
| C | `lstm_ppo3_{100,200}k` | 100k–200k | 0.99 | 0.0 | 128 | `[UNRECOVERABLE]` | 09-09 10:37–10:54 |
| D | `lstm_ppo4_{100,200,300,400}k` | 100k–400k | 0.99 | 0.0 | 128 | `[UNRECOVERABLE]` | 09-09 11:12–12:09 |
| — | **`lstm_gamma997`** | 100,352 | **0.997** | **0.01** | **2048** | **`reward_balance`** | 09-09 13:04–13:22 |

Only the last row is reproducible, and only because it has a manifest. That contrast is the whole
argument for N-4's fix.

**Throughput, CPU-only** (Windows 11, 12 logical cores, no CUDA, `torch 2.14.0+cpu`): Recurrent PPO
runs at roughly **100,000 steps per 18–20 minutes**, so a 1M-step run is about three hours and
retraining the six dead DQN/PPO rungs is hours, not days. Reproducible on a laptop.

---

## Evaluation harness

The acceptance command, unchanged:

```
python -m rfenv.compare --seeds 3 --sampled 10 --figures --out runs/baselines
```

57 scenarios (47 stare replays + 10 sampled) × 3 seeds × however many rungs build — 2,223 episodes
at 13 rungs, about 15 minutes. Train split only; scan replays refused (D36).

Two changes made 2026-09-09, both to stop the harness lying:

1. **An unbuildable rung skips instead of killing the run.** `buildable_rungs()` probes each rung
   once and drops those that cannot be constructed — no training stack installed, checkpoint not
   trained, or checkpoint stale — naming each on stderr with its reason, plus a count. Before this,
   one missing checkpoint ended a 2,223-episode run with a traceback, and the headline command did
   not work at all on a machine without `stable_baselines3`, which §16.5.3 requires.
2. **The count of dropped rungs is printed deliberately.** A comparison that quietly contains no RL
   rows looks like a success and is not one.

**The run validates itself.** Every heuristic row and both reference lines reproduce D46's
committed 2026-09-04 figures exactly — `recency` at 70.2% included. Two earlier small runs put
`recency` at 33.3% and 50.0%. **No number from an 8-to-30-episode run in this lane should be
quoted**; that reproduction is the control that says which scale can be trusted.
