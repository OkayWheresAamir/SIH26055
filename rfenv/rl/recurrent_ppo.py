"""Rung 9: PPO with an LSTM-recurrent policy (sb3-contrib), not plain SB3.

Plain `stable_baselines3.PPO` has no recurrent policy -- `MlpPolicy` is
feed-forward only, so it conditions each action on exactly one D34
observation and nothing else. `sb3_contrib.RecurrentPPO` swaps in
`MlpLstmPolicy`: the same flat 183-vector goes in, but an LSTM inside it
carries a hidden state across the episode, so the action can in
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
other recurrent-specific knob. Whichever reward a checkpoint is trained on,
pass `--reward` explicitly rather than leaning on `env.DEFAULT_REWARD`: that
default has moved more than once, and a rung key naming a reward the checkpoint
was not trained on is exactly the mismatch the manifest (`common.py`) now
exists to make visible.

**Inference on this rung samples rather than taking the argmax.**
`RecurrentRLScheduler` defaults to `deterministic=False`, because measured on
`lstm_gamma997` the policy's action distribution is broad (mean entropy 2.369
against 3.584 for uniform-over-36, mean max-probability 0.206) while its mode
is sticky -- the argmax landed on one band for 580 of 586 steps. Every "the
agent learned to camp" result on this rung came from taking that argmax; the
same checkpoint sampled visits 31 of 36 bands. See `common.py` for the full
note.
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
from rfenv.rl.policies import POLICY_ALIASES

DEFAULT_CHECKPOINT = Path("runs/checkpoints/v1/recurrent_ppo/recurrent_ppo.zip")


def _resolve_policy(policy: str):
    """`"MlpFeatureLstmPolicy"` -> the class; anything else passes through.

    `RecurrentPPO`'s own `policy_aliases` only knows its three library
    policies (`MlpLstmPolicy`/`CnnLstmPolicy`/`MultiInputLstmPolicy`) --
    `rfenv/rl/policies.py` deliberately does not mutate that class-level dict
    (it is shared by every `RecurrentPPO` instance in the process, library
    code, not ours to patch). Passing the class object itself instead of a
    string bypasses that lookup entirely; SB3 accepts either
    (`policy: str | type[RecurrentActorCriticPolicy]`) and records whichever
    was given, so a checkpoint trained this way still round-trips through
    `RecurrentPPO.load()` with no special casing there.
    """
    return POLICY_ALIASES.get(policy, policy)


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
    obs_version: str = "v1",
    band_priority: bool = False,
    priority_coef: float = 0.5,
    priority_n_bands: tuple[int, int] = (3, 6),
    priority_uniform: bool = False,
    priority_high: float = 3.0,
    occupancy_coef: float = 0.0,
    occupancy_decay_cap: float = 2.0,
    device: str = "auto",
    pool=None,
    from_checkpoint: str | Path | None = None,
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

    `pool` defaults to `None`, which is `make_train_env`'s own default --
    `split.training_pool()`, the real train-split emitters (D60). Passing an
    `EmitterPool` here trains entirely on it instead, real data or a
    hand-built synthetic one (`online.py`'s `fine_tune` already supports this
    for adapting an *existing* checkpoint via `env_kwargs={"pool": ...}`;
    this is the same idea for training a policy *from scratch* on a custom
    distribution rather than fine-tuning a pretrained one onto it). A
    checkpoint trained this way is otherwise ordinary -- same manifest, same
    `load_checkpoint()`, same deployable contract -- only the distribution of
    scenarios it ever saw during training differs from every other rung's.

    `from_checkpoint`, when set, loads that checkpoint instead of building a
    fresh model, and keeps training it (`reset_num_timesteps=False`) rather
    than starting a new one from scratch -- offline training's own resume
    path, deliberately built to mirror `online.py`'s (same `custom_objects`
    load, same reasoning: `n_steps` is baked into the rollout buffer at
    `_setup_model()`, so a hyperparameter override has to go through
    `custom_objects`, not an attribute set after loading). The difference
    from `online.fine_tune()` is what it does *not* change: this still runs
    ordinary fixed-length episodes drawn fresh from `pool` each reset, and
    the gradient still sees `step()`'s own (truth-reading) reward -- no
    `ObservableRewardWrapper`, no continuous grid, none of D77's online-mode
    machinery. This is what lets "does further *offline* training on a new
    distribution adapt a checkpoint the same way online fine-tuning did"
    become a real, answerable comparison, changing only the one variable
    (online mechanism vs. plain continued offline training) rather than
    also changing the reward the gradient sees and the episode length at
    the same time.
    """
    started_at = time.time()
    hyperparameters = dict(hyperparameters or {})
    env = make_train_env(reward=reward, obs_version=obs_version, pool=pool,
                          band_priority=band_priority, priority_coef=priority_coef,
                          priority_n_bands=priority_n_bands, priority_uniform=priority_uniform,
                          priority_high=priority_high, occupancy_coef=occupancy_coef,
                          occupancy_decay_cap=occupancy_decay_cap)
    if from_checkpoint is not None:
        require_loadable(from_checkpoint)
        # custom_objects, not model.n_steps = ... after loading -- n_steps is
        # baked into the rollout buffer at _setup_model()/load time, same
        # reasoning online.py's own resume path follows.
        model = RecurrentPPO.load(from_checkpoint, env=env, device=device,
                                  custom_objects=hyperparameters)
        model.verbose = verbose
    else:
        model = RecurrentPPO(_resolve_policy(policy), env, seed=seed, verbose=verbose,
                              device=device, **hyperparameters)
    manifest_kwargs = {"reward": reward, "hyperparameters": hyperparameters,
                       "started_at": started_at, "description": description}
    callbacks = training_callbacks(
        print_episode_metrics=print_episode_metrics,
        checkpoint_freq=checkpoint_freq,
        checkpoint=checkpoint,
        run=run or Path(checkpoint).stem,
        manifest_kwargs=manifest_kwargs,
    )
    # Continuing a checkpoint's own step count is the whole point of resuming
    # it, the same as online.py's fine_tune().
    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                reset_num_timesteps=from_checkpoint is None)

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
                     choices=["MlpLstmPolicy", "CnnLstmPolicy", "MultiInputLstmPolicy",
                              "MlpFeatureLstmPolicy", "BandEncoderLstmPolicy"],
                     help="sb3-contrib recurrent policy type (default: MlpLstmPolicy, "
                          "obs -> LSTM -> actor/critic). rfenv/rl/policies.py adds two "
                          "opt-in alternatives, each a matched-pair architecture "
                          "comparison against the baseline, not a replacement: "
                          "MlpFeatureLstmPolicy (D82, obs -> 2-layer LayerNorm MLP -> "
                          "LSTM -> actor/critic) and BandEncoderLstmPolicy (D83, obs -> "
                          "per-band encoder, shared across all 36 bands, mean-pooled, "
                          "concatenated with an encoded global-feature vector -> LSTM -> "
                          "actor/critic). LSTM hidden size and actor/critic heads are "
                          "unchanged by either.")
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
