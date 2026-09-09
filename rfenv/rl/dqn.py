"""Rung 7: the SB3 DQN algorithm specifically.

First pass, deliberately naive (RL_TEAM_HANDOFF.md §15 Day 1): library-default
hyperparameters, reward = hit_z (env.DEFAULT_REWARD), nothing tuned.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN

from rfenv.env import DEFAULT_REWARD
from rfenv.rl.common import make_train_env

DEFAULT_CHECKPOINT = Path("runs/checkpoints/deep_q_network.zip")


def train(
    *,
    dqn_type: str = "MlpPolicy",
    reward: str = DEFAULT_REWARD,
    total_timesteps: int = 60_000,
    seed: int = 0,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    verbose: int = 1,
) -> DQN:
    """Build SB3 DQN with library defaults and train it. Tune nothing (Day 1).

    SB3's `gamma` below is the RL discount factor (constructor default 0.99) --
    NOT rfenv's GAMMA_DBM (-111 dBm detection threshold). Same name, unrelated
    quantities. Left at SB3's default for this first pass; RL_TEAM_HANDOFF.md
    flags a short horizon here as a likely later tuning target, not something
    to fix now.
    """
    env = make_train_env(reward=reward)
    model = DQN(dqn_type, env, seed=seed, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    return model


def load_checkpoint(path: str | Path | None = None) -> DQN:
    """Load a trained DQN, or fail with the exact command that fixes it."""
    path = Path(path) if path is not None else DEFAULT_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(
            f"no DQN checkpoint at {path}. Train one first, e.g.:\n"
            f"    venv/Scripts/python.exe -m rfenv.rl --reward {DEFAULT_REWARD} "
            f"--timesteps 20000 --checkpoint {path}\n"
            "(runs/ is gitignored -- checkpoints are rebuilt, never committed.)"
        )
    return DQN.load(path)
