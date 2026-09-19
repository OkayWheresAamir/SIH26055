"""Rung 8: the SB3 PPO algorithm.

Second algorithm in this package, and the reason `common.py` exists: `RLScheduler`
and `make_train_env` needed nothing changed to support it, since any SB3 algorithm
exposes the same `.predict()` API and trains on the same env. Mirrors `dqn.py`'s
shape exactly. First pass, deliberately naive (RL_TEAM_HANDOFF.md §15 Day 1):
library-default hyperparameters, nothing tuned. The registered checkpoint
(rung 8's base variant) was specifically trained with `reward="hit_z"` --
historically true, but no longer reproducible: `hit_z` was retired from
`REWARDS` 2026-09-10 after failing D62's screen (D63), so `--reward hit_z` now
raises rather than training. The checkpoint itself was already permanently
unloadable from D49's observation change. `env.DEFAULT_REWARD` is
`reward_balance`.
"""

from __future__ import annotations

from pathlib import Path

import time

from stable_baselines3 import PPO

from rfenv.env import DEFAULT_REWARD
from rfenv.rl.common import (
    add_manifest_arguments, finish_training, make_train_env,
    parse_hyperparameters, require_loadable, training_callbacks,
)

DEFAULT_CHECKPOINT = Path("runs/checkpoints/v1/ppo/ppo.zip")


def train(
    *,
    policy: str = "MlpPolicy",
    reward: str = DEFAULT_REWARD,
    total_timesteps: int = 20_000,
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
) -> PPO:
    """Build SB3 PPO with library defaults and train it. Tune nothing (Day 1).

    Same caveat as `dqn.train`: SB3's `gamma` here is PPO's own discount factor
    (constructor default 0.99), unrelated to rfenv's GAMMA_DBM. PPO is on-policy
    and typically wants a vectorised env for real throughput; this first pass
    stays as naive as `dqn.train` -- a single env, nothing tuned -- so the two
    algorithms are compared on equally untuned terms, not one hand-optimised
    against the other.

    `print_episode_metrics=True` prints `ScanEnv.episode_metrics()` after every
    training episode via `EpisodeMetricsCallback` (`common.py`) -- off by
    default since it's one line of noisy stdout per episode, and a full
    training run is thousands of them.

    `checkpoint_freq`, when set, saves an extra snapshot every that-many steps
    (e.g. `checkpoint_freq=200_000`) alongside the final checkpoint this
    function always writes at the end -- see `common.training_callbacks` for
    the exact naming and why you'd want several checkpoints from one run.

    `hyperparameters` is forwarded verbatim to the `PPO(...)` constructor and
    recorded verbatim in the manifest, which is what makes the manifest's
    hyperparameter record true rather than aspirational: what is written down
    is the same dict the algorithm was handed, not a description of it. An
    empty dict genuinely means every knob was left at the library default.
    """
    started_at = time.time()
    hyperparameters = dict(hyperparameters or {})
    env = make_train_env(reward=reward, obs_version=obs_version,
                          band_priority=band_priority, priority_coef=priority_coef,
                          priority_n_bands=priority_n_bands, priority_uniform=priority_uniform,
                          priority_high=priority_high, occupancy_coef=occupancy_coef,
                          occupancy_decay_cap=occupancy_decay_cap)
    model = PPO(policy, env, seed=seed, verbose=verbose, device=device, **hyperparameters)
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


def load_checkpoint(path: str | Path | None = None) -> PPO:
    """Load a trained PPO model, or fail with the exact command that fixes it."""
    path = Path(path) if path is not None else DEFAULT_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(
            f"no PPO checkpoint at {path}. Train one first, e.g.:\n"
            f"    venv/Scripts/python.exe -m rfenv.rl.ppo --reward {DEFAULT_REWARD} "
            f"--timesteps 20000 --checkpoint {path}\n"
            "(runs/ is gitignored -- checkpoints are rebuilt, never committed.)"
        )
    require_loadable(path)
    return PPO.load(path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from rfenv.env import REWARDS

    ap = argparse.ArgumentParser(
        prog="python -m rfenv.rl.ppo",
        description="Train the SB3 PPO scheduler (rung 8). "
                     "Day-1 pass: library defaults, nothing tuned.",
    )
    ap.add_argument("--policy", default="MlpPolicy", choices=["MlpPolicy", "CnnPolicy"],
                     help="SB3 PPO policy type (default: MlpPolicy, a 2-layer MLP)")
    ap.add_argument("--reward", default=DEFAULT_REWARD, choices=sorted(REWARDS),
                     help="reward candidate, passed to ScanEnv(reward=...) (D29)")
    ap.add_argument("--timesteps", type=int, default=20_000,
                     help="total_timesteps for model.learn() (default: a token amount)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                     help="where to write the .zip")
    ap.add_argument("--check-env", action="store_true",
                     help="run gymnasium's check_env on the training env and exit")
    ap.add_argument("--print-episode-metrics",default=True, action="store_true",
                     help="print ScanEnv.episode_metrics() after every training episode")
    ap.add_argument("--checkpoint-freq", type=int, default=None,
                     help="also save a snapshot every N steps, alongside the final one "
                          "(e.g. 200000), named <run>_s1.zip, <run>_s2.zip, ...")
    add_manifest_arguments(ap)
    args = ap.parse_args(argv)

    if args.check_env:
        from gymnasium.utils.env_checker import check_env
        check_env(make_train_env(reward=args.reward, obs_version=args.obs_version,
                                  band_priority=args.band_priority,
                                  priority_coef=args.priority_coef,
                                  priority_n_bands=tuple(args.priority_n_bands),
                                  priority_uniform=args.priority_uniform,
                                  priority_high=args.priority_high,
                                  occupancy_coef=args.occupancy_coef,
                                  occupancy_decay_cap=args.occupancy_decay_cap),
                  skip_render_check=True)
        print("check_env: OK")
        return 0

    train(policy=args.policy, reward=args.reward, total_timesteps=args.timesteps,
          seed=args.seed, checkpoint=args.checkpoint,
          print_episode_metrics=args.print_episode_metrics,
          checkpoint_freq=args.checkpoint_freq,
          run=args.run_name, description=args.description,
          hyperparameters=parse_hyperparameters(args.hyperparam),
          obs_version=args.obs_version,
          band_priority=args.band_priority, priority_coef=args.priority_coef,
          priority_n_bands=tuple(args.priority_n_bands), priority_uniform=args.priority_uniform,
          priority_high=args.priority_high, occupancy_coef=args.occupancy_coef,
          occupancy_decay_cap=args.occupancy_decay_cap,
          device=args.device)
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved manifest:   {Path(args.checkpoint).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
