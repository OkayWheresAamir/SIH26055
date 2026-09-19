"""Rung 7: the SB3 DQN algorithm specifically.

First pass, deliberately naive (RL_TEAM_HANDOFF.md §15 Day 1): library-default
hyperparameters, nothing tuned. The registered checkpoint (rung 7's base
variant) was specifically trained with `reward="hit_z"` -- historically true,
but no longer reproducible: `hit_z` was retired from `REWARDS` 2026-09-10 after
failing D62's screen (D63), so `--reward hit_z` now raises rather than training.
The checkpoint itself was already permanently unloadable from D49's observation
change. `env.DEFAULT_REWARD` is `reward_balance`.
"""

from __future__ import annotations

from pathlib import Path

import time

from stable_baselines3 import DQN

from rfenv.env import DEFAULT_REWARD
from rfenv.rl.common import (
    finish_training, make_train_env, require_loadable, training_callbacks,
)

DEFAULT_CHECKPOINT = Path("runs/checkpoints/v1/deep_q_network/deep_q_network.zip")


def train(
    *,
    dqn_type: str = "MlpPolicy",
    reward: str = DEFAULT_REWARD,
    total_timesteps: int = 60_000,
    seed: int = 0,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    verbose: int = 1,
    print_episode_metrics: bool = False,
    checkpoint_freq: int | None = None,
    run: str | None = None,
    description: str = "",
    hyperparameters: dict | None = None,
    obs_version: str = "v1",
    band_priority: bool = False,
    priority_coef: float = 0.5,
    priority_n_bands: tuple[int, int] = (3, 6),
    priority_uniform: bool = False,
    priority_high: float = 3.0,
    occupancy_coef: float = 0.0,
    occupancy_decay_cap: float = 2.0,
    device: str = "auto",
) -> DQN:
    """Build SB3 DQN with library defaults and train it. Tune nothing (Day 1).

    SB3's `gamma` below is the RL discount factor (constructor default 0.99) --
    NOT rfenv's GAMMA_DBM (-111 dBm detection threshold). Same name, unrelated
    quantities. Left at SB3's default for this first pass; RL_TEAM_HANDOFF.md
    flags a short horizon here as a likely later tuning target, not something
    to fix now.

    `print_episode_metrics=True` prints `ScanEnv.episode_metrics()` after every
    training episode via `EpisodeMetricsCallback` (`common.py`) -- off by
    default since it's one line of noisy stdout per episode, and a full
    training run is thousands of them.

    `checkpoint_freq`, when set, saves an extra snapshot every that-many steps
    (e.g. `checkpoint_freq=200_000`) alongside the final checkpoint this
    function always writes at the end -- see `common.training_callbacks` for
    the exact naming and why you'd want several checkpoints from one run.

    `hyperparameters` is forwarded verbatim to the `DQN(...)` constructor and
    recorded verbatim in the manifest, so what is written down is the same dict
    the algorithm was handed rather than a description of it. An empty dict
    genuinely means every knob was left at the library default.
    """
    started_at = time.time()
    hyperparameters = dict(hyperparameters or {})
    env = make_train_env(reward=reward, obs_version=obs_version,
                          band_priority=band_priority, priority_coef=priority_coef,
                          priority_n_bands=priority_n_bands, priority_uniform=priority_uniform,
                          priority_high=priority_high, occupancy_coef=occupancy_coef,
                          occupancy_decay_cap=occupancy_decay_cap)
    model = DQN(dqn_type, env, seed=seed, verbose=verbose, device=device, **hyperparameters)
    manifest_kwargs = {"reward": reward, "hyperparameters": hyperparameters,
                       "started_at": started_at, "description": description}
    callbacks = training_callbacks(
        print_episode_metrics=print_episode_metrics,
        checkpoint_freq=checkpoint_freq,
        checkpoint=checkpoint,
        run=run or Path(checkpoint).stem,
        manifest_kwargs=manifest_kwargs,
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    finish_training(model, checkpoint=checkpoint, run=run, **manifest_kwargs)
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
    require_loadable(path)
    return DQN.load(path)
