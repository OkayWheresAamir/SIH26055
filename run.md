# run.md — training and comparing schedulers

Practical commands only: what to type to train an RL scheduler, register it, and compare it
against the baseline ladder. For what the metrics *mean*, see `docs/project/EVALUATION.md` — this
file doesn't redefine anything there.

All commands assume an activated venv at the repo root:

```
venv/Scripts/python.exe -m ...          # Windows
.venv/bin/python -m ...                 # Linux/macOS, per requirements.txt
```

Examples below use `venv/Scripts/python.exe`; substitute your own interpreter path.

## The ladder, as it stands

```
venv/Scripts/python.exe -m rfenv.baselines
```

Prints every registered rung (key, number, label, scheduler vs. reference line) plus the D43
airtime-per-band comparison. This is the live source of truth for what's registered — the table
below is illustrative and will drift as rungs get added.

| Rung | Kind | Needs training? |
|---|---|---|
| 1 random, 2 round_robin, 3 turing_sweep | open-loop | no |
| 4 camper, 5 recency | heuristic | no |
| 6/6a apfeld, apfeld_active_rfs | published adaptive | no |
| 7, 7a, ... | DQN variants | **yes** — `rfenv.rl` |
| 8, 8a, ... | PPO variants | **yes** — `rfenv.rl.ppo` |
| 9, 9a, ... | RecurrentPPO (LSTM) variants | **yes** — `rfenv.rl.recurrent_ppo` |
| camper_oracle, oracle_pulse | reference lines (read truth) | no — not competitors, not in `SCHEDULERS` |

Rungs 1–6a and the two reference lines need nothing built — they're plain Python policies, ready
to compare out of the box. Only the RL rungs (7/8/9 and their lettered variants) need a checkpoint
trained first.

## Training a DQN scheduler (rung 7)

```
venv/Scripts/python.exe -m rfenv.rl --reward reward_balance --timesteps 60000 --seed 0 \
    --checkpoint runs/checkpoints/deep_q_network.zip
```

**`hit_z` and `hit_y` are retired from `REWARDS`** (2026-09-10, D62 -- both fail the reward screen, ranking the camper above every sweeping policy), so `--reward hit_z`/`hit_y` now raises rather than training. `reward_balance` is the only candidate that currently passes; run `python -m rfenv.reward_gate` before training on anything else.

| Flag | Default | Meaning |
|---|---|---|
| `--reward` | `env.DEFAULT_REWARD` (currently `reward_balance`) | one of `REWARDS` (D29) — run `python -c "from rfenv.env import REWARDS, DEFAULT_REWARD; print(DEFAULT_REWARD, sorted(REWARDS))"` for the live set and default |
| `--timesteps` | `20000` | `model.learn(total_timesteps=...)` |
| `--seed` | `0` | SB3 model seed |
| `--checkpoint` | `runs/checkpoints/deep_q_network.zip` | where the final `.zip` is written |
| `--check-env` | off | run `gymnasium.utils.env_checker.check_env` on the training env and exit — no training |
| `--print-episode-metrics` | off | print `ScanEnv.episode_metrics()` (interception ratio, censored intercept time, coverage, ...) after every training episode |
| `--checkpoint-freq` | none | also save a snapshot every N steps, e.g. `--checkpoint-freq 200000` — see below |
| `--run-name` | the checkpoint's stem | names this run in the manifest and in snapshot filenames |
| `--description` | empty | what this run is trying — recorded in the manifest, and where a rung's label should come from |
| `--hyperparam` | none | `KEY=VALUE` passed straight to the algorithm's constructor, repeatable (e.g. `--hyperparam ent_coef=0.01`) |

### Screen a reward before you train on it

```
venv/Scripts/python.exe -m rfenv.reward_gate
```

Minutes, no GPU, no agent. Scores rungs 2, 4, 5 and 6a under every registered candidate over 8
seeds and reports PASS/FAIL against criteria fixed in `rfenv/reward_gate.py`. **As of 2026-09-10
only `reward_balance` passes** -- `hit_z` and `hit_y` both rank rung 4 above every sweeping policy,
so an agent trained on either cannot be expected to beat the floor (D62). Run it before spending
hours of compute; that is the entire point of it.

### Pick a checkpoint by the pre-registered rule

```
venv/Scripts/python.exe -m rfenv.selection runs/checkpoints/myrun_*.zip
```

Scores each checkpoint on the **12 validation configs** (D60) paired against rung 5 and picks the
highest `dominates - dominated`, ties toward fewer steps (D61). Do not pick by eye from a compare
table -- that fits the evaluation set through the choice, and the bias grows every time the lane
iterates.

### Training now uses the training half automatically

`make_train_env` defaults to `split.training_pool()` -- 35 configs, zero emitters shared with the
12 validation configs. Nothing to pass; it is the default. `python -m rfenv.split` prints the
split. Passing `EmitterPool.from_train()` explicitly re-opens the leak D60 closed.

### The reward candidates

Six registered (D29's cap of three was lifted by D57). Run
`python -c "from rfenv.env import REWARDS, DEFAULT_REWARD; print(DEFAULT_REWARD, sorted(REWARDS))"`
for the live set.

**Verdicts below are `python -m rfenv.reward_gate`'s (D62), not the earlier by-hand table** — run
it yourself before trusting a stale copy.

| `--reward` | what it pays for | D62 screen |
|---|---|---|
| `hit_z` | +1 per true hit `Z`, per slot | **FAIL** — ranks the camper above every sweeping policy |
| `hit_y` | +1 per declared hit `Y`, per slot | **FAIL** — same failure |
| `reward_balance` | exploit + explore + occupancy − airtime concentration | **PASS** — the only survivor of six |
| `greedy` | declarations + remembered hit rate; no explore term, no camping cost | FAIL — camps, by design, the exploit corner |
| `explore` | staleness − airtime concentration + new `(emitter, band)` discoveries | FAIL — ranks round-robin above rung 5, by design |
| `weighted` | `0.3 * greedy * 4.08 + 0.7 * explore` | FAIL — against the real rung 4, camper sits only 0.4σ below the sweeps, short of the 1.0σ bar |

`reward_balance` separates rung 5 from round-robin by **+59.0 +/- 19.4 on 8/8 seeds** (re-measured
against the real rung — the earlier `-1.3` figure came from scoring a hand-written stand-in and is
withdrawn, D56). `greedy` and `explore` are the two corners of D14's tension and are *expected* to
fail their opposite check — they exist so the axis spans something, not as proposals. `weighted` is
the knob
between them, and `env.make_reward_weighted(alpha)` builds any other point on the curve without
touching the registry:

```python
from rfenv.env import make_reward_weighted
env = ScanEnv(pool=pool, reward="reward_balance")   # any registered key -- overwritten below
env._reward_fn = make_reward_weighted(0.4)    # anywhere in [0, 1]
```

**`WEIGHTED_ALPHA = 0.3` sits in the middle of a measured window.** Below 0.2 the explore half
dominates and round-robin outscores rung 5; from 0.5 up a 2-band ping-pong outscores round-robin,
which is D53's failure mode arriving through the greedy half. D57 has the sweep.

**None of this ranks the candidates.** Those are ordering checks against known policies, not D47's
paired-dominance rule over the evaluation protocol — which has still never been run.

**Always pass `--reward` explicitly.** The CLI's default tracks `env.DEFAULT_REWARD`, and that has
moved twice (`hit_z` -> `first_intercept` -> `reward_balance`, D50/D53). Rung 7's own registered
checkpoint was trained with `--reward hit_z`, so its manifest names a reward that **no longer
exists in `REWARDS`** -- `hit_z` was retired 2026-09-10 (D62), and reproducing that specific
checkpoint is no longer possible without reinstating it. The manifest beside each `.zip` still
records which reward a checkpoint was actually trained on -- when a rung key and a manifest
disagree, the manifest is right, even for a reward that has since been retired.

`runs/` is gitignored; checkpoints are rebuilt locally, never committed. A checkpoint's shape is
tied to `ScanEnv`'s observation vector (D34) — if `env.py`'s observation changes (adding a
component, for instance), every existing checkpoint stops loading and needs retraining.

**This has happened four times** (109 → 145 → 146 → 147 → 146; D49, D55), so assume it will happen
again: `--run-name` and `--description` are what make a dead checkpoint's manifest still tell you
what it was, and they cost nothing to pass. As of D55 the vector is **146** wide and its box is
**not** `[0, 1]` — `visit_density` reads in fair shares (ceiling 36.0) and `staleness` in reference
sweeps (ceiling 13.95), so a policy that assumed unit-interval inputs needs retraining, not
rescaling.

### Every checkpoint carries a manifest

Each `.zip` written by any of the three trainers gets a `.json` beside it — `ppo_fi.zip` gets
`ppo_fi.json` — recording what the archive itself cannot: the reward, the exact argv, the commit
and whether the tree was dirty, the hyperparameters actually passed (as opposed to the defaults
they were left at), the seed, the timestep count, wall-clock, hardware, library versions, the
SHA-256 of the `.zip`, and **the observation width the environment had at training time**.

That last field is the one that pays for the rest. SB3 binds no environment at load time, so a
checkpoint trained against a narrower observation loads perfectly happily and only fails much
later, inside `predict()` during a comparison run, with a shape error that names neither the
checkpoint nor the reason. `load_checkpoint()` now reads the manifest first and refuses up front
instead, quoting the training command that would rebuild it:

```
ppo_fi.zip was trained against a 145-wide observation, but ScanEnv now builds a 146-wide one --
this checkpoint cannot be used and will fail inside predict() if forced.
  run:       ppo_fi
  reward:    hit_z
  trained:   2026-09-08T18:23:04Z at commit de8b6a8 (dirty)
  retrain:   venv/Scripts/python.exe -m rfenv.rl.ppo --reward hit_z --timesteps 600000 ...
```

The `retrain:` line is quoted verbatim from the manifest and is now itself stale: `hit_z` was retired from `REWARDS` 2026-09-10 (D62), so running it raises immediately rather than retraining. This is illustrative of the refusal message's format, not a command to run.

A checkpoint with no manifest is *not* refused — everything trained before manifests existed is
legitimate and still loads. It only means the width check cannot run for that file.

Nothing about a run is reproducible from the `.zip` alone: SB3 serialises the algorithm's
constructor arguments but not the reward function, so two runs that differ only in reward are
indistinguishable inside the archive. Keep the `.json` with the `.zip`.

### Comparing checkpoints from different points in one training run

```
venv/Scripts/python.exe -m rfenv.rl --reward reward_balance --timesteps 1000000 --seed 0 \
    --checkpoint runs/checkpoints/deep_q_network.zip --checkpoint-freq 200000
```

Writes `deep_q_network_s1.zip` … `deep_q_network_s4.zip` (plus a `.json` each) alongside the final
`deep_q_network.zip` at 1,000,000 steps, in the same directory as `--checkpoint`. Each one is a
complete, independently loadable checkpoint, not a diff — register the ones you actually want to
compare as their own ladder rungs (see "Registering a trained checkpoint as a rung" below), e.g.
rungs `7a`/`7b`/`7c` for three points along the same run. Same flag, same behavior, on all three
trainers (`rfenv.rl`, `rfenv.rl.ppo`, `rfenv.rl.recurrent_ppo`).

Snapshots are named for the **run and an ordinal**, not for their timestep count: a count in a
filename gets transcribed into a rung key and a rung label by hand and then drifts from the
archive, which has already happened twice here (rung 8a is labelled "800k" for a 600,064-step
checkpoint; rung 7a says "5,000 timesteps" for a 20,000-step one). `cat` the manifest for the
count — `total_timesteps` is read from the model, not typed by anyone:

```
venv/Scripts/python.exe -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['run'], m['total_timesteps'], m['reward'], m['description'])" runs/checkpoints/deep_q_network_s3.json
```

## Training a PPO scheduler (rung 8)

```
venv/Scripts/python.exe -m rfenv.rl.ppo --reward reward_balance --timesteps 100000 --seed 0 \
    --checkpoint runs/checkpoints/ppo_balance.zip
```

Same flags as `rfenv.rl`, plus:

| Flag | Default | Meaning |
|---|---|---|
| `--policy` | `MlpPolicy` | `MlpPolicy` or `CnnPolicy` |

`--check-env` works identically to the DQN CLI.

## Training a RecurrentPPO scheduler (rung 9)

```
venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo --reward reward_balance --timesteps 20000 --seed 0 \
    --checkpoint runs/checkpoints/recurrent_ppo.zip
```

Same flags as `rfenv.rl.ppo`, plus:

| Flag | Default | Meaning |
|---|---|---|
| `--policy` | `MlpLstmPolicy` | `MlpLstmPolicy`, `CnnLstmPolicy`, or `MultiInputLstmPolicy` |

This is `sb3-contrib`'s `RecurrentPPO`, not plain SB3's `PPO` — plain PPO has no recurrent policy
at all. The observation is the same D34 vector every other rung gets; what's different is that an
LSTM inside the policy carries a hidden state across the whole episode, so the action can in
principle depend on the accumulated scan history rather than only the latest look.

**Inference needs a different adapter.** A recurrent policy's `.predict()` takes and returns
hidden state across calls (`state=...`, `episode_start=...`) — `RLScheduler` (used by rungs 7/8)
has no slot for that. Rung 9 is wrapped in `RecurrentRLScheduler` instead
(`rfenv/rl/common.py`), which carries the state itself and infers "new episode" from
`info["slot"] == 0`. If you ever drive a rung-9 checkpoint directly from Python rather than through
the ladder, use `RecurrentRLScheduler`, not `RLScheduler` — the latter will silently reset the LSTM's
hidden state to `None` on every single call (via its own default `state=None` inside `.predict()`),
which runs without error but defeats the entire point of the LSTM.

`--check-env` works identically to the other two CLIs.

### Inference samples; it does not take the argmax (D54)

`RecurrentRLScheduler` and `RLScheduler` both take a `deterministic` argument, **defaulting to
`False`**. This matters more than it sounds: an on-policy algorithm optimises expected return under
its sampled distribution and never evaluates its own mode, so nothing in training constrains where
the argmax lands. Measured on `lstm_gamma997`, the policy's action distribution has mean entropy
2.369 against `ln 36 = 3.584` and a modal band holding 0.206 of the mass — broad, not collapsed —
yet the argmax sat on one band for 580 of 586 steps. Argmaxed it visits 2 bands; sampled it visits
31.

- **Rungs 8 and 9 sample.** Leave the default alone.
- **Rung 7 (DQN) passes `deterministic=True`** in its ladder factory, deliberately: a DQN's greedy
  action *is* its policy, and SB3's `deterministic=False` there means ε-greedy exploration noise
  (`exploration_final_eps`, 0.05) — a training artefact.
- **Seeded runs stay reproducible.** SB3 samples from torch's *global* generator and takes no
  generator argument, so `ladder._seed_torch(rng)` folds each rung's own seeded stream into torch
  before the scheduler is built. Same seed gives a byte-identical action sequence.

Pass `deterministic=True` only to reproduce a row from before 2026-09-09.

## Registering a trained checkpoint as a rung

Training writes a `.zip`; it isn't compared until it has a `Rung(...)` entry in
`rfenv/baselines/ladder.py`. Both RL algorithms use the same factory pattern — one line per
variant, so multiple checkpoints coexist instead of overwriting each other:

```python
Rung("deep_q_network_hit_y", "7b", "DQN (hit_y)",
     "Same algorithm as rung 7, trained on hit_y instead of hit_z -- both "
     "retired from REWARDS after failing D62's screen (D29).",
     _dqn_rung_factory(Path("runs/checkpoints/deep_q_network_hit_y.zip"))),
```

```python
Rung("ppo_balance_100k", "8b", "PPO (reward_balance, 100k)",
     "Ours. Trained on reward_balance (D29, D53), 100k timesteps.",
     _ppo_rung_factory(Path("runs/checkpoints/ppo_balance_100k.zip"))),
```

```python
Rung("recurrent_ppo_hit_y_20k", "9a", "Recurrent PPO (hit_y)",
     "Same algorithm as rung 9, trained on hit_y instead of hit_z -- both "
     "retired from REWARDS after failing D62's screen (D29).",
     _recurrent_ppo_rung_factory(Path("runs/checkpoints/recurrent_ppo_hit_y.zip"))),
```

- **`key`** must be unique across the whole ladder — it's the identifier `--rungs`/`--animate`
  take on the command line and the dict key `BY_KEY` is built from.
- **`rung`** is the display number (`"7a"`, `"8b"`, ...). Keep it unique too: two rungs sharing a
  number breaks anything that looks up "the" rung by number, and two rungs sharing a **label**
  collapse to one point in the Pareto figure (`render.pareto()` keys its legend by label unless
  told otherwise — see `compare.py`'s call site for the pattern that avoids this).
- Import `load_checkpoint` from the specific algorithm module (`rfenv.rl.dqn` / `rfenv.rl.ppo` /
  `rfenv.rl.recurrent_ppo`), not the `rfenv.rl` package re-export — the factory closures do this
  already; match it if you add a new one.
- Rung 9's factory wraps its checkpoint in `RecurrentRLScheduler`, not `RLScheduler` — see the
  RecurrentPPO section above for why. A fourth algorithm gets its own equivalent factory, mirroring
  whichever adapter its `.predict()` shape actually needs.

## Comparing rungs

```
venv/Scripts/python.exe -m rfenv.compare --seeds 3 --sampled 10 --figures
```

Runs every registered scheduler (everything in `SCHEDULERS` — deployable rungs only, reference
lines excluded by construction) over every stare replay plus `--sampled` extra scenarios, times
`--seeds` noise seeds, and prints the comparison table.

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `runs/baselines` | artefact directory |
| `--seeds` | `3` | noise seeds per scenario |
| `--configs` | all | limit to the first N stare replays (smoke test) |
| `--sampled` | `0` | add N scenarios drawn fresh from the train pool |
| `--rungs` | all schedulers | comma-separated subset, e.g. `--rungs round_robin,deep_q_network_z_60k,ppo_hit_z_60k` |
| `--reward` | `env.DEFAULT_REWARD` (currently `reward_balance`) | **one shared reward for every rung's `ScanEnv` this run** — affects the printed reward column only, not what any RL model was trained on (D7) |
| `--figures` | off | write `pareto.png`, `timeline_*.png`, `discovery_*.png`, and `animation_*.gif` per scenario |
| `--gif-stride` | `8` | slots between animation frames, `--figures` only |
| `--gif-fps` | `12` | `--figures` only |

**`--reward` is a trap if misread**: it sets the reward every `ScanEnv` in *this comparison run*
uses to compute the printed reward number — it does not change what a trained RL checkpoint
learned from. A model trained on `reward_balance` still gets scored on `round_robin`'s numbers unless
you pass `--reward reward_balance` to this specific `compare` invocation.

Smoke test before a full run:

```
venv/Scripts/python.exe -m rfenv.compare --configs 2 --seeds 1 \
    --rungs round_robin,recency,deep_q_network_z_60k
```

Full comparison matching `EVALUATION.md` §5's numbers:

```
venv/Scripts/python.exe -m rfenv.compare --seeds 3 --sampled 10 --figures
```

## Eyeballing one scenario or one comparison

```
# Plain waterfall of one scenario's truth grid
venv/Scripts/python.exe -m rfenv.render config_2 stare out.png

# Animated side-by-side of several rungs on the same scenario/seed
venv/Scripts/python.exe -m rfenv.render config_2 stare out.gif \
    --animate round_robin,deep_q_network_z_60k,ppo_hit_z_60k --seed 0
```

| Flag | Default | Meaning |
|---|---|---|
| `--animate` | off | comma-separated rung keys — switches from a plain waterfall to `compare_animation` |
| `--seed` | `0` | shared seed across every animated rung |
| `--stride` | `4` | slots between animation frames, `--animate` only |
| `--fps` | `15` | `--animate` only |

`out`'s extension isn't inspected either way — `.png` for a plain waterfall, `.gif` for
`--animate`, by convention only.

## Validation gates (not scheduler-specific, included for completeness)

```
venv/Scripts/python.exe -m rfenv.validate
```

Runs the four gates against the train split (`EVALUATION.md` §6). Not part of training/comparing
schedulers — this checks the environment itself against the recordings.

## Everything runs, nothing's trained

```
venv/Scripts/python.exe -m pytest tests -q
```

RL-dependent tests (`tests/test_rl.py`, the RL rows of `tests/test_baselines.py`'s generic
per-rung tests) build their own untrained model in-process and need no checkpoint on disk — they
still run, and still assert real properties (legality, determinism, the deployability guard),
without anything in `runs/checkpoints/` existing. Tests that *do* need a specific rung's checkpoint
skip cleanly with a message naming the exact training command, rather than failing.
