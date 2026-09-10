"""`python -m rfenv.rl` -- train the DQN scheduler (rung 7). Day-1 pass:
library defaults, nothing tuned."""

from __future__ import annotations

import argparse
from pathlib import Path

from rfenv.env import DEFAULT_REWARD, REWARDS
from rfenv.rl.common import (
    add_manifest_arguments, make_train_env, parse_hyperparameters,
)
from rfenv.rl.dqn import DEFAULT_CHECKPOINT, train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rfenv.rl",
        description="Train the SB3 DQN scheduler (rung 7, deep_q_network). "
                     "Day-1 pass: library defaults, nothing tuned.",
    )
    ap.add_argument("--reward", default=DEFAULT_REWARD, choices=sorted(REWARDS),
                     help="reward candidate, passed to ScanEnv(reward=...) (D29)")
    ap.add_argument("--timesteps", type=int, default=20_000,
                     help="total_timesteps for model.learn() (default: a token amount)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                     help="where to write the .zip")
    ap.add_argument("--check-env", action="store_true",
                     help="run gymnasium's check_env on the training env and exit")
    ap.add_argument("--print-episode-metrics", action="store_true",
                     help="print ScanEnv.episode_metrics() after every training episode")
    ap.add_argument("--checkpoint-freq", type=int, default=None,
                     help="also save a snapshot every N steps, alongside the final one "
                          "(e.g. 200000), named <run>_s1.zip, <run>_s2.zip, ...")
    add_manifest_arguments(ap)
    args = ap.parse_args(argv)

    if args.check_env:
        from gymnasium.utils.env_checker import check_env
        check_env(make_train_env(reward=args.reward), skip_render_check=True)
        print("check_env: OK")
        return 0

    train(dqn_type="MlpPolicy", reward=args.reward, total_timesteps=args.timesteps,
          seed=args.seed, checkpoint=args.checkpoint,
          print_episode_metrics=args.print_episode_metrics,
          checkpoint_freq=args.checkpoint_freq,
          run=args.run_name, description=args.description,
          hyperparameters=parse_hyperparameters(args.hyperparam))
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved manifest:   {Path(args.checkpoint).with_suffix('.json')}")
    return 0
