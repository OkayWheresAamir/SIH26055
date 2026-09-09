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
| camper_oracle, oracle_pulse | reference lines (read truth) | no — not competitors, not in `SCHEDULERS` |

Rungs 1–6a and the two reference lines need nothing built — they're plain Python policies, ready
to compare out of the box. Only the RL rungs (7/8 and their lettered variants) need a checkpoint
trained first.

## Training a DQN scheduler (rung 7)

```
venv/Scripts/python.exe -m rfenv.rl --reward hit_z --timesteps 60000 --seed 0 \
    --checkpoint runs/checkpoints/deep_q_network.zip
```

| Flag | Default | Meaning |
|---|---|---|
| `--reward` | `hit_z` | one of `REWARDS` (D29) — run `python -c "from rfenv.env import REWARDS; print(sorted(REWARDS))"` for the live set |
| `--timesteps` | `20000` | `model.learn(total_timesteps=...)` |
| `--seed` | `0` | SB3 model seed |
| `--checkpoint` | `runs/checkpoints/deep_q_network.zip` | where the `.zip` is written |
| `--check-env` | off | run `gymnasium.utils.env_checker.check_env` on the training env and exit — no training |

`runs/` is gitignored; checkpoints are rebuilt locally, never committed. A checkpoint's shape is
tied to `ScanEnv`'s observation vector (D34) — if `env.py`'s observation changes (adding a
component, for instance), every existing checkpoint stops loading and needs retraining.

## Training a PPO scheduler (rung 8)

```
venv/Scripts/python.exe -m rfenv.rl.ppo --reward first_intercept --timesteps 100000 --seed 0 \
    --checkpoint runs/checkpoints/ppo_fi.zip
```

Same flags as `rfenv.rl`, plus:

| Flag | Default | Meaning |
|---|---|---|
| `--policy` | `MlpPolicy` | `MlpPolicy` or `CnnPolicy` |

`--check-env` works identically to the DQN CLI.

## Registering a trained checkpoint as a rung

Training writes a `.zip`; it isn't compared until it has a `Rung(...)` entry in
`rfenv/baselines/ladder.py`. Both RL algorithms use the same factory pattern — one line per
variant, so multiple checkpoints coexist instead of overwriting each other:

```python
Rung("deep_q_network_hit_y", "7b", "DQN (hit_y)",
     "Same algorithm as rung 7, trained on hit_y instead of hit_z (D29).",
     _dqn_rung_factory(Path("runs/checkpoints/deep_q_network_hit_y.zip"))),
```

```python
Rung("ppo_first_intercept_100k", "8b", "PPO (first_intercept, 100k)",
     "Ours. Trained on first_intercept (D29), 100k timesteps.",
     _ppo_rung_factory(Path("runs/checkpoints/ppo_fi_100k.zip"))),
```

- **`key`** must be unique across the whole ladder — it's the identifier `--rungs`/`--animate`
  take on the command line and the dict key `BY_KEY` is built from.
- **`rung`** is the display number (`"7a"`, `"8b"`, ...). Keep it unique too: two rungs sharing a
  number breaks anything that looks up "the" rung by number, and two rungs sharing a **label**
  collapse to one point in the Pareto figure (`render.pareto()` keys its legend by label unless
  told otherwise — see `compare.py`'s call site for the pattern that avoids this).
- Import `load_checkpoint` from the specific algorithm module (`rfenv.rl.dqn` / `rfenv.rl.ppo`),
  not the `rfenv.rl` package re-export — the factory closures do this already; match it if you add
  a new one.

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
| `--reward` | `hit_z` | **one shared reward for every rung's `ScanEnv` this run** — affects the printed reward column only, not what any RL model was trained on (D7) |
| `--figures` | off | write `pareto.png`, `timeline_*.png`, `discovery_*.png`, and `animation_*.gif` per scenario |
| `--gif-stride` | `8` | slots between animation frames, `--figures` only |
| `--gif-fps` | `12` | `--figures` only |

**`--reward` is a trap if misread**: it sets the reward every `ScanEnv` in *this comparison run*
uses to compute the printed reward number — it does not change what a trained RL checkpoint
learned from. A `ppo_fi` model trained on `first_intercept` still gets scored on `hit_z`'s numbers
unless you pass `--reward first_intercept` to this specific `compare` invocation.

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
