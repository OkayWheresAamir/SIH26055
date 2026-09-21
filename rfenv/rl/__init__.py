"""Rungs 7, 8 and 9 of the baseline ladder: reinforcement-learning schedulers.

All deliberately naive first passes (RL_TEAM_HANDOFF.md §15 Day 1):
library-default hyperparameters, nothing tuned. Each base variant (rungs 7, 8,
9 -- not their lettered siblings) was specifically trained with `--reward hit_z`
(D29's original candidate 1). **That flag no longer reproduces anything**:
`hit_z` and `hit_y` were retired from `REWARDS` 2026-09-10 after both failed
D62's screen, so rungs 7 and 7a/8's base variants are permanently unloadable
(they already were, from D49's observation-width change) and unreproducible.
`env.DEFAULT_REWARD` is `reward_balance` -- the only candidate that currently
passes the screen; see `python -m rfenv.reward_gate` before training on
anything else. See `dqn.py` (rung 7), `ppo.py` (rung 8) and
`recurrent_ppo.py` (rung 9).

**Package layout, built for more than one algorithm.** `make_train_env` (the
`ScanEnv(pool=..., reward=...)` factory) is algorithm-agnostic -- any SB3(-contrib)
algorithm trains on the same env -- and lives in `common.py`, alongside the two
policy adapters `common.py`'s own docstring explains (`RLScheduler` for a plain
feed-forward `.predict()`, `RecurrentRLScheduler` for a recurrent one). Only
`train()`, `load_checkpoint()` and the default checkpoint path are
algorithm-specific, one file per algorithm (`dqn.py`, `ppo.py`,
`recurrent_ppo.py`). A fourth algorithm is another sibling file, not a rewrite.
Top-level `rfenv.rl.train`/`load_checkpoint`/`DEFAULT_CHECKPOINT` stay aliased
to `dqn.py` specifically (today's original algorithm, kept for anything
already using those names); PPO and RecurrentPPO are reached explicitly as
`rfenv.rl.ppo.*` / `rfenv.rl.recurrent_ppo.*`.

Usage::

    python -m rfenv.rl --check-env                                        # DQN sanity check only, no training
    python -m rfenv.rl --reward reward_balance                           # train DQN, 20,000 steps -- the only candidate D62 currently passes
    python -m rfenv.rl --reward greedy --timesteps 100000 --seed 1       # a different candidate (D29, D57) -- run reward_gate first
    python -m rfenv.rl --checkpoint runs/checkpoints/v1/dqn_greedy/dqn_greedy.zip --reward greedy

    python -m rfenv.rl.ppo --check-env                                    # PPO's own CLI, same shape
    python -m rfenv.rl.ppo --checkpoint runs/checkpoints/v1/ppo_balance/ppo_balance.zip --reward reward_balance

    python -m rfenv.rl.recurrent_ppo --check-env                          # RecurrentPPO's own CLI, same shape
    python -m rfenv.rl.recurrent_ppo --checkpoint runs/checkpoints/v1/recurrent_ppo_balance/recurrent_ppo_balance.zip --reward reward_balance

**Checkpoints live one folder per model** (D75): `runs/checkpoints/<obs_version>/<model_name>/`,
not flat -- 150 files in one directory was unnavigable. `<obs_version>` is "v1"
(the 183-wide vector every checkpoint before D30 was trained on) or "v2" (D30/D76's
362-wide vector); pick the directory that matches whichever `ScanEnv(obs_version=...)`
the run trains against. `finish_training` (`common.py`) creates the nested
directory automatically -- there is nothing else to set up.

**Every checkpoint written here gets a `.json` manifest beside it**, from the
shared helper in `common.py` -- so `dqn.py`, `ppo.py`, `recurrent_ppo.py` and
any algorithm added later are self-describing by construction rather than by
anyone remembering. It records what the SB3 archive cannot: the reward (SB3
serialises the algorithm's constructor arguments, never the reward function,
so two runs differing only in reward are indistinguishable inside the `.zip`),
the exact argv, the commit and whether the tree was dirty, and the observation
width at training time. `load_checkpoint()` reads that width first and refuses
a stale checkpoint up front, with the training command that rebuilds it,
instead of letting it fail later inside `predict()` with a bare shape error.
Checkpoints predating manifests have none and still load. See `run.md`.

Training writes a checkpoint; it does not by itself run the agent against
anything. To actually run it -- against the rest of the ladder, on real
scenarios, scored the same way every other rung is::

    python -m rfenv.compare --rungs deep_q_network,ppo,recency,round_robin --seeds 3 --sampled 10 --figures

`baselines.make("deep_q_network", ...)` / `baselines.make("ppo", ...)` load
each rung's own checkpoint (via `_dqn_rung_factory`/`_ppo_rung_factory` in
`baselines/ladder.py`) and wrap it in `guarded()` automatically -- nothing in
`compare.py` needs to know a rung is a neural net rather than a heuristic, or
which algorithm trained it.

To drive one episode directly from Python, without the ladder::

    from rfenv.env import ScanEnv
    from rfenv.rl import RLScheduler, load_checkpoint
    from rfenv.rollout import run_episode
    from rfenv.scenario import Scenario

    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"))
    policy = RLScheduler(load_checkpoint())   # raises with the exact fix if untrained
    run_episode(env, policy, seed=0)
    print(env.episode_metrics())
"""

from __future__ import annotations

from rfenv.rl._cli import main
from rfenv.rl.common import RLScheduler, make_train_env
from rfenv.rl.dqn import DEFAULT_CHECKPOINT, load_checkpoint, train

__all__ = [
    "DEFAULT_CHECKPOINT",
    "RLScheduler",
    "load_checkpoint",
    "main",
    "make_train_env",
    "train",
]
