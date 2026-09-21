"""Watch a scheduler run, live (D76).

`rollout.run_episode` is the scored driver and stays untouched -- no view, no
pacing, as fast as Python allows, because that is what every metric in this
repository was measured against. This is a second, unscored driver for the
other use: looking at a schedule as it happens, in the terminal or a window,
optionally paced to something close to real time.

Any registered rung works -- heuristic or trained, `Rung.deployable` or not
(`B.make` already handles the difference). A checkpoint's `obs_version` is
resolved the same way `compare.py` resolves it, via the same functions, so a
rung's own training config builds its env correctly here too.
"""

from __future__ import annotations

import argparse
import time

from rfenv.baselines import ladder as B
from rfenv.compare import resolve_obs_version, resolve_priority_kwargs
from rfenv.constants import N_SLOTS, SLOT_S
from rfenv.env import DEFAULT_REWARD, ScanEnv
from rfenv.live import make_live_view
from rfenv.scenario import Scenario
from rfenv.truth import TruthGrid


def watch(
    key: str,
    *,
    config_id: str | None = "config_2",
    source: str = "stare",
    seed: int = 0,
    episode_slots: int = N_SLOTS,
    reward: str = DEFAULT_REWARD,
    view: str = "light",
    fps: float = 10.0,
    speed: float = 4.0,
    device: str = "cpu",
) -> dict:
    """Run one episode of `key`, driving `view` as it goes.

    `speed` paces real time to `dwell_slots * SLOT_S / speed` per step -- 1.0 is
    the true 50 ms/slot clock, 0 (or negative) skips pacing entirely and lets
    the view's own fps cap set the pace instead. `device="cpu"` is the default
    on purpose: this is meant to run *alongside* whatever is training, and an
    RL rung's checkpoint otherwise loads on `device="auto"`, which picks CUDA
    and would compete with it. Set before anything imports torch, so it has to
    happen here, first.
    """
    if device == "cpu":
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    if episode_slots > N_SLOTS:
        from rfenv.split import training_pool
        scenario, pool = None, training_pool()
    else:
        scenario, pool = Scenario.replay(config_id, source), None

    grid = TruthGrid.from_scenario(scenario) if scenario is not None else None
    (band_priority, priority_coef, priority_n_bands, priority_uniform,
     priority_high, occupancy_coef, occupancy_decay_cap) = resolve_priority_kwargs(
        key, False, 0.5, (3, 6), False, 3.0, 0.0, 2.0,
    )
    env = ScanEnv(scenario=scenario, pool=pool, reward=reward,
                  obs_version=resolve_obs_version(key, "v1"),
                  band_priority=band_priority, priority_coef=priority_coef,
                  priority_n_bands=priority_n_bands, priority_uniform=priority_uniform,
                  priority_high=priority_high, occupancy_coef=occupancy_coef,
                  occupancy_decay_cap=occupancy_decay_cap, episode_slots=episode_slots)

    policy = B.make(key, seed=seed, grid=grid)
    obs, info = env.reset(seed=seed)

    # `make_live_view` forwards kwargs to the backend's constructor, which
    # takes `min_interval_s` -- translated here so this function's own knob
    # stays in the units a human picks a pace with. Harmless for "off": that
    # branch returns `NullView()` before looking at any kwarg.
    live = make_live_view(view, min_interval_s=(1.0 / fps if fps > 0 else 0.0))

    terminated = False
    with live:
        live.open(env)
        try:
            while not terminated:
                obs, _reward, terminated, _, info = env.step(int(policy(obs, info)))
                live.update(env)
                if speed > 0:
                    time.sleep(info.get("dwell_slots", 1) * SLOT_S / speed)
        except KeyboardInterrupt:
            pass
        live.update(env, force=True)

    metrics = env.episode_metrics()
    print(f"\n{key}: ratio {metrics['interception_ratio']:.4f}  "
          f"cTTI {metrics['censored_mean_intercept_time_s']:.2f}s  "
          f"coverage {metrics['emitter_coverage']:.4f}  "
          f"({env.t}/{env.episode_slots} slots)")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rfenv.watch",
        description="Watch a scheduler run live, in the terminal or a window (D76).",
    )
    ap.add_argument("--rung", required=True, help=f"one of {sorted(B.BY_KEY)}")
    ap.add_argument("--config-id", default="config_2",
                    help="stare/scan replay to watch; ignored if --episode-slots > 600")
    ap.add_argument("--source", default="stare", choices=("stare", "scan"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-slots", type=int, default=N_SLOTS,
                    help="mission length in slots; above 600 draws a continuous "
                         "grid from the training pool instead of one replay")
    ap.add_argument("--reward", default=DEFAULT_REWARD)
    ap.add_argument("--view", default="light", choices=("off", "light", "full"))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=4.0,
                    help="real-time multiplier; 1.0 = true 50ms/slot, 0 = unpaced")
    ap.add_argument("--device", default="cpu", choices=("cpu", "auto"))
    args = ap.parse_args(argv)

    if args.rung not in B.BY_KEY:
        ap.error(f"unknown rung {args.rung!r}; choose from {sorted(B.BY_KEY)}")

    watch(args.rung, config_id=args.config_id, source=args.source, seed=args.seed,
          episode_slots=args.episode_slots, reward=args.reward, view=args.view,
          fps=args.fps, speed=args.speed, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
