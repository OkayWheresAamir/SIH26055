"""Online fine-tuning: a trained policy that keeps learning while it scans (D77).

Everything else in `rfenv.rl` trains a policy, freezes it, and evaluates the
frozen thing. This module is the other mode -- load a checkpoint, put it in a
mission that runs as long as you let it, and keep taking gradient steps from what
comes back. It is the repository's first resume path: `train()` always builds a
fresh model, and nothing before this ever called `learn(reset_num_timesteps=False)`.

**The gradient sees `reward_balance_obs`, not `step()`'s reward.** This is the
load-bearing decision and the reason `ObservableRewardWrapper` exists. `step()`
returns `reward_balance`, whose third term is `0.5 * dwell.Z.sum()` -- truth, not
measurement. A fielded receiver has no `Z`. An agent fine-tuned against it would
be optimising a quantity it could never compute outside the simulator, so the
"adapts in the field" claim would be false. The wrapper substitutes the `Y`-based
twin (D75), which is exactly what the receiver declared. Scoring is untouched:
`total_reward` and `episode_metrics()` still use the training reward, so a
fine-tuned checkpoint is judged on the same yardstick as every other rung.

**Why updates are per rollout and not per mission.** A 30 s episode is 300-600
decisions; the checkpoints here trained at `n_steps=8192`. One mission cannot
fill even a fourteenth of one rollout, and at `gamma=0.997` the effective horizon
(~333 steps) is comparable to the whole episode, so the advantages would be
dominated by the terminal bootstrap. A per-mission update is therefore not a
small update, it is a noisy one, and this module does not offer it. Fine-tuning
runs on a continuous grid (`episode_slots`, D76), where a simulated hour is
40k-72k steps, and updates at `--n-steps` (default 2048, about two simulated
minutes, ~30 updates per simulated hour). That number is pre-registered here
rather than tuned later.

**SB3 owns the loop.** The recurrent rollout buffer, sequence masking and
GAE-through-LSTM-state are fiddly and already correct in sb3-contrib;
reimplementing `collect_rollouts` is where a silent correctness bug would live.
The cost, stated plainly: during fine-tuning the acting loop is SB3's, not
`RecurrentRLScheduler`, so the deployable `(obs, info) -> int` contract is not
exercised here. It does not need to be -- that contract matters at *evaluation*,
which happens afterwards on the saved checkpoint through the ordinary ladder,
unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from rfenv.constants import N_SLOTS, SLOT_S
from rfenv.env import DEFAULT_REWARD, ScanEnv
from rfenv.live import make_live_view

# Defaults, chosen for adaptation rather than training and pre-registered here.
# A low learning rate and a tight clip because this starts from a policy that
# already works: the risk is destroying it, not failing to move it. `ent_coef`
# is kept at the value the recent checkpoints trained with -- this rung's
# documented failure mode (D54) is collapsing onto one band, and fine-tuning on
# a narrow slice of the world is exactly when that would happen again.
ONLINE_DEFAULTS = {
    "n_steps": 2048,
    "batch_size": 256,
    "learning_rate": 1.0e-5,
    "clip_range": 0.1,
    "target_kl": 0.02,
    "ent_coef": 0.01,
}


def make_observable_reward_wrapper():
    """Built lazily so importing this module never requires gymnasium."""
    import gymnasium as gym

    class ObservableRewardWrapper(gym.Wrapper):
        """Hand the learner the reward a real receiver could have computed.

        Substitution only -- the environment is untouched, `total_reward` still
        accumulates the training reward, and `episode_metrics()` still reports
        it. The swap is exactly `0.5 * dwell.Z.sum()` -> `0.5 * dwell.Y.sum()`,
        which differ by the receiver's own Pd (0.842) and Pfa (1.35e-3).
        """

        def step(self, action):
            obs, _truth_reward, terminated, truncated, info = self.env.step(action)
            return (obs, self.env.unwrapped.last_reward_obs,
                    terminated, truncated, info)

    return ObservableRewardWrapper


def make_online_env(*, reward: str = DEFAULT_REWARD, obs_version: str = "v3",
                    episode_slots: int = 120 * N_SLOTS, pool=None,
                    observable_reward: bool = True, **env_kwargs):
    """One continuous-grid `ScanEnv`, wrapped so the gradient sees `Y`.

    `pool` defaults to the training half (D60), the same source
    `rl.common.make_train_env` uses -- fine-tuning is still training and the
    validation half stays untouched.
    """
    if pool is None:
        from rfenv.split import training_pool

        pool = training_pool()
    env = ScanEnv(pool=pool, reward=reward, obs_version=obs_version,
                  episode_slots=episode_slots, **env_kwargs)
    if observable_reward:
        env = make_observable_reward_wrapper()(env)
    return env


def _callbacks(view, segments_path: Path | None):
    """The two online-only callbacks, built lazily (they subclass SB3's base)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class OnlineViewCallback(BaseCallback):
        """Drive the live view from inside SB3's own rollout loop.

        The view rate-limits itself, so calling this every step is cheap.
        """

        def __init__(self, view):
            super().__init__()
            self.view = view
            self._opened = False

        def _on_step(self) -> bool:
            env = self.training_env.envs[0].unwrapped
            if not self._opened:
                self.view.open(env)
                self._opened = True
            else:
                self.view.update(env)
            return True

    class SegmentMetricsCallback(BaseCallback):
        """Coverage and reward per 30 s segment, appended as JSONL.

        **This series is the result for this capability.** Whether online
        adaptation helps is the question "does coverage per segment trend up
        across a mission", and nothing else here answers it. Recorded live
        because a run that is interrupted after 40 minutes should still have
        said something.
        """

        def __init__(self, path: Path | None):
            super().__init__()
            self.path = path
            self._segment = 0
            self._mission = 0
            self._last_found = 0
            self._last_reward = 0.0
            self.rows: list[dict] = []

        def _on_step(self) -> bool:
            env = self.training_env.envs[0].unwrapped
            segment = int(env.t) // N_SLOTS

            # `DummyVecEnv` auto-resets the instant a mission ends, so the clock
            # going backwards is the only signal that a new one began -- there is
            # no callback hook that fires in between.
            if segment < self._segment:
                self._segment, self._last_found, self._last_reward = 0, 0, 0.0
                self._mission += 1
                return True
            # A row is written when a boundary is *crossed*, describing the
            # segment that just finished -- never before one has.
            if segment == self._segment:
                return True

            found, reward = len(env.tracks), float(env.total_reward)
            row = {
                "mission": self._mission,
                "segment": self._segment,
                "slot": int(env.t),
                "time_s": round(int(env.t) * SLOT_S, 2),
                "num_timesteps": int(self.num_timesteps),
                "found_total": found,
                "found_this_segment": found - self._last_found,
                "n_detectable": len(env.detectable),
                "reward_this_segment": round(reward - self._last_reward, 4),
                "coverage_so_far": round(
                    found / len(env.detectable), 6) if env.detectable else None,
            }
            self.rows.append(row)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            self._segment, self._last_found, self._last_reward = segment, found, reward
            return True

    out = [SegmentMetricsCallback(segments_path)]
    if not isinstance(view, type(make_live_view("off"))):
        out.insert(0, OnlineViewCallback(view))
    return out


def fine_tune(
    *,
    checkpoint: str | Path,
    out_checkpoint: str | Path,
    total_timesteps: int = 200_000,
    episode_slots: int = 120 * N_SLOTS,
    reward: str = DEFAULT_REWARD,
    obs_version: str = "v3",
    view: str = "off",
    out: str | Path | None = None,
    hyperparameters: dict | None = None,
    observable_reward: bool = True,
    device: str = "auto",
    run: str | None = None,
    description: str = "",
    env_kwargs: dict | None = None,
):
    """Load a checkpoint, keep training it on a continuous grid, save the result.

    Returns the fine-tuned model. Ctrl-C is a supported way to stop: the
    checkpoint is saved with a manifest recording the interruption, which is what
    makes "run it until you have seen enough" a usable workflow rather than a way
    to lose an hour of adaptation. The rollout in progress is discarded, which is
    correct -- a partial rollout has no advantages to learn from.
    """
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rfenv.rl.common import finish_training, require_loadable

    started_at = time.time()
    checkpoint = Path(checkpoint)
    require_loadable(checkpoint)

    resolved = dict(ONLINE_DEFAULTS)
    resolved.update(hyperparameters or {})

    env = DummyVecEnv([
        lambda: make_online_env(
            reward=reward, obs_version=obs_version, episode_slots=episode_slots,
            observable_reward=observable_reward, **(env_kwargs or {})
        )
    ])

    # `custom_objects`, NOT `model.n_steps = ...` after loading. `n_steps` is
    # baked into the checkpoint (8192 for these) and is read while the rollout
    # buffer is being constructed inside `_setup_model()`; assigning afterwards
    # leaves a buffer of the old size and the run silently collects 8192
    # transitions per update instead of the 2048 asked for.
    model = RecurrentPPO.load(checkpoint, env=env, device=device,
                              custom_objects=resolved)

    live = make_live_view(view)
    callbacks = _callbacks(live, Path(out) / "segments.jsonl" if out else None)
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)

    interrupted = False
    try:
        with live:
            model.learn(
                total_timesteps=total_timesteps,
                # The whole point: continue this policy's life rather than
                # restarting its clock, so the manifest's `total_timesteps`
                # keeps counting from what the checkpoint already had.
                reset_num_timesteps=False,
                callback=callbacks,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted -- saving the adapted checkpoint", file=sys.stderr)

    note = description or f"online fine-tune of {checkpoint.name}"
    if interrupted:
        note += " (interrupted by the user mid-rollout; the partial rollout was discarded)"
    finish_training(
        model,
        checkpoint=out_checkpoint,
        run=run or Path(out_checkpoint).stem,
        reward=reward,
        hyperparameters=resolved,
        started_at=started_at,
        description=note,
    )
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rfenv.rl.online",
        description="Fine-tune a trained checkpoint online, on a continuous grid (D77).",
    )
    ap.add_argument("--checkpoint", required=True, help="the checkpoint to adapt")
    ap.add_argument("--out-checkpoint", default=None,
                    help="where to save the adapted policy (default: <checkpoint>_online.zip)")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--episode-slots", type=int, default=120 * N_SLOTS,
                    help=f"mission length in slots; must be a multiple of {N_SLOTS} "
                         f"(default 72000 = 1 simulated hour)")
    ap.add_argument("--reward", default=DEFAULT_REWARD,
                    help="the scoring reward; the gradient sees its observable twin")
    ap.add_argument("--obs-version", default="v3", choices=("v1", "v2", "v2p", "v3"))
    ap.add_argument("--view", default="light", choices=("off", "light", "full"),
                    help="live view: terminal strip, matplotlib window, or none")
    ap.add_argument("--fps", type=float, default=10.0, help="live view refresh cap")
    ap.add_argument("--out", default=None, help="directory for segments.jsonl")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--description", default="")
    ap.add_argument("--truth-reward", action="store_true",
                    help="fine-tune on the TRUTH reward instead of the observable "
                         "one. Not deployable -- the agent optimises a quantity no "
                         "receiver can compute. Label any result from it accordingly.")
    ap.add_argument("--hyperparam", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    from rfenv.rl.common import parse_hyperparameters

    checkpoint = Path(args.checkpoint)
    out_checkpoint = Path(
        args.out_checkpoint
        or checkpoint.with_name(f"{checkpoint.stem}_online.zip")
    )

    model = fine_tune(
        checkpoint=checkpoint,
        out_checkpoint=out_checkpoint,
        total_timesteps=args.timesteps,
        episode_slots=args.episode_slots,
        reward=args.reward,
        obs_version=args.obs_version,
        view=args.view,
        out=args.out,
        hyperparameters=parse_hyperparameters(args.hyperparam),
        observable_reward=not args.truth_reward,
        device=args.device,
        run=args.run_name,
        description=args.description,
    )
    print(f"saved  {out_checkpoint}")
    print(f"       {out_checkpoint.with_suffix('.json')}")
    print(f"steps  {model.num_timesteps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
