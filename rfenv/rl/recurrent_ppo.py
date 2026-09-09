"""Rung 9: PPO with an LSTM-recurrent policy (sb3-contrib), not plain SB3.

Plain `stable_baselines3.PPO` has no recurrent policy -- `MlpPolicy` is
feed-forward only, so it conditions each action on exactly one D34
observation and nothing else. `sb3_contrib.RecurrentPPO` swaps in
`MlpLstmPolicy`: the same flat 147-vector goes in, but an LSTM inside the
policy carries a hidden state across the episode, so the action can in
principle depend on the whole scan history so far, not just the latest look.

Mirrors `ppo.py`'s shape (`train`/`load_checkpoint`/`DEFAULT_CHECKPOINT`/CLI)
with one structural difference: inference needs `RecurrentRLScheduler`
(`common.py`), not `RLScheduler` -- a recurrent `.predict()` takes and returns
hidden state across calls, which the plain adapter has no slot for. A
`RecurrentPPO` checkpoint is not interchangeable with a `PPO` one either way;
`PPO.load()` cannot read it and vice versa.

First pass, deliberately naive (mirrors RL_TEAM_HANDOFF.md §15 Day 1 for the
other two algorithms): library-default hyperparameters, nothing tuned --
including the LSTM's own hidden size (sb3-contrib's default, 256) and every
other recurrent-specific knob. The registered checkpoint (rung 9's base
variant, `recurrent_ppo_hit_z_20k`) is meant to be trained with
`reward="hit_z"` -- pass that explicitly; `env.DEFAULT_REWARD` is
`first_intercept`, not `hit_z`, so the unqualified default would silently
mismatch the key's own name.
"""

from __future__ import annotations

from pathlib import Path

import time

from sb3_contrib import RecurrentPPO

from rfenv.env import DEFAULT_REWARD
from rfenv.rl.common import (
    add_manifest_arguments, finish_training, make_train_env,
    parse_hyperparameters, require_loadable, training_callbacks,
)

DEFAULT_CHECKPOINT = Path("runs/checkpoints/recurrent_ppo.zip")


def train(
    *,
    policy: str = "MlpLstmPolicy",
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
) -> RecurrentPPO:
    """Build sb3-contrib's RecurrentPPO with library defaults and train it.

    Same caveat as `dqn.train`/`ppo.train`: SB3's `gamma` here is PPO's own
    discount factor, unrelated to rfenv's GAMMA_DBM. A single env, nothing
    tuned -- same naive-first-pass terms as the other two algorithms, so all
    three are compared on equally untuned footing.

    `print_episode_metrics=True` prints `ScanEnv.episode_metrics()` after every
    training episode via `EpisodeMetricsCallback` (`common.py`) -- off by
    default since it's one line of noisy stdout per episode, and a full
    training run is thousands of them.

    `checkpoint_freq`, when set, saves an extra snapshot every that-many steps
    (e.g. `checkpoint_freq=200_000`) alongside the final checkpoint this
    function always writes at the end -- see `common.training_callbacks` for
    the exact naming and why you'd want several checkpoints from one run.

    `hyperparameters` is forwarded verbatim to the `RecurrentPPO(...)`
    constructor and recorded verbatim in the manifest. This is the lever that
    matters most for this rung specifically: every RecurrentPPO checkpoint on
    disk was trained at `ent_coef=0.0` (sb3-contrib's default), which is what
    lets a policy sharpen onto a single band and stay there -- pass
    `--hyperparam ent_coef=0.01` to try otherwise, and the manifest will say
    which of the two any given checkpoint was.
    """
    started_at = time.time()
    hyperparameters = dict(hyperparameters or {})
    env = make_train_env(reward=reward)
    model = RecurrentPPO(policy, env, seed=seed, verbose=verbose, **hyperparameters)
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


def load_checkpoint(path: str | Path | None = None) -> RecurrentPPO:
    """Load a trained RecurrentPPO model, or fail with the exact command that fixes it."""
    path = Path(path) if path is not None else DEFAULT_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(
            f"no RecurrentPPO checkpoint at {path}. Train one first, e.g.:\n"
            f"    venv/Scripts/python.exe -m rfenv.rl.recurrent_ppo --reward {DEFAULT_REWARD} "
            f"--timesteps 20000 --checkpoint {path}\n"
            "(runs/ is gitignored -- checkpoints are rebuilt, never committed.)"
        )
    require_loadable(path)
    return RecurrentPPO.load(path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from rfenv.env import REWARDS

    ap = argparse.ArgumentParser(
        prog="python -m rfenv.rl.recurrent_ppo",
        description="Train the sb3-contrib RecurrentPPO scheduler (rung 9). "
                     "Day-1 pass: library defaults, nothing tuned.",
    )
    ap.add_argument("--policy", default="MlpLstmPolicy",
                     choices=["MlpLstmPolicy", "CnnLstmPolicy", "MultiInputLstmPolicy"],
                     help="sb3-contrib recurrent policy type (default: MlpLstmPolicy)")
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

    train(policy=args.policy, reward=args.reward, total_timesteps=args.timesteps,
          seed=args.seed, checkpoint=args.checkpoint,
          print_episode_metrics=args.print_episode_metrics,
          checkpoint_freq=args.checkpoint_freq,
          run=args.run_name, description=args.description,
          hyperparameters=parse_hyperparameters(args.hyperparam))
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved manifest:   {Path(args.checkpoint).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
