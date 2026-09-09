"""Rungs 7 and 8 of the baseline ladder: reinforcement-learning schedulers.

Both deliberately naive first passes (RL_TEAM_HANDOFF.md §15 Day 1):
library-default hyperparameters, reward = hit_z (env.DEFAULT_REWARD), nothing
tuned. See `dqn.py` (rung 7) and `ppo.py` (rung 8).

**Package layout, built for more than one algorithm.** `RLScheduler` (the
`(obs, info) -> band` adapter around `model.predict()`) and `make_train_env`
(the `ScanEnv(pool=..., reward=...)` factory) are algorithm-agnostic --
any SB3 algorithm exposes the same `.predict()` API and trains on the same
env -- and live in `common.py`. Only `train()`, `load_checkpoint()` and the
default checkpoint path are algorithm-specific, one file per algorithm
(`dqn.py`, `ppo.py`). A third algorithm is another sibling file, not a rewrite.
Top-level `rfenv.rl.train`/`load_checkpoint`/`DEFAULT_CHECKPOINT` stay aliased
to `dqn.py` specifically (today's original algorithm, kept for anything
already using those names); PPO is reached explicitly as `rfenv.rl.ppo.*`.

Usage::

    python -m rfenv.rl --check-env                                   # DQN sanity check only, no training
    python -m rfenv.rl                                                # train DQN on hit_z, 20,000 steps (the defaults)
    python -m rfenv.rl --reward hit_y --timesteps 100000 --seed 1     # a different reward candidate (D29)
    python -m rfenv.rl --checkpoint runs/checkpoints/dqn_hit_y.zip --reward hit_y

    python -m rfenv.rl.ppo --check-env                                # PPO's own CLI, same shape
    python -m rfenv.rl.ppo --checkpoint runs/checkpoints/ppo_hit_y.zip --reward hit_y

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
