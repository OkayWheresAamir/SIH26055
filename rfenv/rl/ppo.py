"""Rung 8: the SB3 PPO algorithm.

Second algorithm in this package, and the reason `common.py` exists: `RLScheduler`
and `make_train_env` needed nothing changed to support it, since any SB3 algorithm
exposes the same `.predict()` API and trains on the same env. Mirrors `dqn.py`'s
shape exactly. First pass, deliberately naive (RL_TEAM_HANDOFF.md §15 Day 1):
library-default hyperparameters, reward = hit_z (env.DEFAULT_REWARD), nothing
tuned.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from rfenv.env import DEFAULT_REWARD
from rfenv.rl.common import make_train_env

DEFAULT_CHECKPOINT = Path("runs/checkpoints/ppo.zip")


def train(
    *,
    policy: str = "MlpPolicy",
    reward: str = DEFAULT_REWARD,
    total_timesteps: int = 20_000,
    seed: int = 0,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    verbose: int = 1,
) -> PPO:
    """Build SB3 PPO with library defaults and train it. Tune nothing (Day 1).

    Same caveat as `dqn.train`: SB3's `gamma` here is PPO's own discount factor
    (constructor default 0.99), unrelated to rfenv's GAMMA_DBM. PPO is on-policy
    and typically wants a vectorised env for real throughput; this first pass
    stays as naive as `dqn.train` -- a single env, nothing tuned -- so the two
    algorithms are compared on equally untuned terms, not one hand-optimised
    against the other.
    """
    env = make_train_env(reward=reward)
    model = PPO(policy, env, seed=seed, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
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
    args = ap.parse_args(argv)

    if args.check_env:
        from gymnasium.utils.env_checker import check_env
        check_env(make_train_env(reward=args.reward), skip_render_check=True)
        print("check_env: OK")
        return 0

    train(policy=args.policy, reward=args.reward, total_timesteps=args.timesteps,
          seed=args.seed, checkpoint=args.checkpoint)
    print(f"saved checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
