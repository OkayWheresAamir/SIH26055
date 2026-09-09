"""`python -m rfenv.render config_2 stare out.png` -- eyeball one scenario.

`--animate a,b,c` switches to `compare_animation`: run those rungs on the
same scenario and seed, then write the side-by-side GIF to `out` instead of
a plain waterfall PNG. `out`'s extension is not inspected either way -- pass
`.png` for a waterfall, `.gif` for `--animate`.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rfenv.render.comparison import compare_animation
from rfenv.render.episode import waterfall


def main(argv: list[str] | None = None) -> int:
    from rfenv import baselines as B
    from rfenv import metrics as M
    from rfenv.env import ScanEnv
    from rfenv.rollout import run_episode
    from rfenv.scenario import Scenario
    from rfenv.truth import TruthGrid

    ap = argparse.ArgumentParser(
        prog="python -m rfenv.render",
        description="Render one scenario's truth grid, or (--animate) an animated "
                     "side-by-side comparison of several schedulers on it.",
    )
    ap.add_argument("config_id")
    ap.add_argument("source", choices=("scan", "stare"))
    ap.add_argument("out", help="output path: .png for a waterfall, .gif for --animate")
    ap.add_argument("--animate", default=None,
                     help="comma-separated rung keys, e.g. deep_q_network,recency,round_robin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4,
                     help="slots between animation frames (--animate only)")
    ap.add_argument("--fps", type=int, default=15, help="--animate only")
    args = ap.parse_args(argv)

    scenario = Scenario.replay(args.config_id, args.source)
    grid = TruthGrid.from_scenario(scenario)

    if args.animate is None:
        waterfall(grid, path=args.out)
        print(f"{args.out}  <-  {args.config_id} {args.source}: {grid.summary()}")
        return 0

    keys = [k.strip() for k in args.animate.split(",") if k.strip()]
    unknown = [k for k in keys if k not in B.BY_KEY]
    if unknown:
        ap.error(f"unknown rung(s) {unknown}; choose from {sorted(B.BY_KEY)}")

    with tempfile.TemporaryDirectory() as tmp:
        runs = {}
        for key in keys:
            env = ScanEnv(scenario=scenario)
            run_episode(env, B.make(key, seed=args.seed, grid=grid), seed=args.seed)
            runs[key] = M.write_run(Path(tmp) / key, env, scheduler=key, seed=args.seed)
        out = compare_animation(runs, grid, args.out, stride=args.stride, fps=args.fps)

    print(f"{out}  <-  {args.config_id} {args.source}, rungs {keys}, seed {args.seed}")
    return 0
