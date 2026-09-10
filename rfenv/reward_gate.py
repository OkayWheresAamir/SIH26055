"""A screen every reward candidate passes before anything trains on it.

D47 fixed the rule for *selecting* a reward: the candidate beating round-robin on
both headline metrics on the most paired episodes. It has never been run, and it
is expensive -- it needs a trained agent per candidate, which is hours of compute
each. This module is the cheap screen that goes in front of it, and it needs no
agent at all: **score the ladder's own rungs under a candidate and ask whether
that candidate can even tell them apart.**

The argument is short. If a reward cannot rank rung 5 above rung 2 -- two fixed,
known policies, one measurably better than the other on the metrics that decide
the project -- then no policy trained on it can be expected to discover the
difference either, whatever the hyperparameters. Gradient descent optimises the
reward it is given; it cannot recover an ordering the reward does not encode.
Such a candidate is excluded here, in minutes on a CPU, instead of after a
1M-step run.

**The criteria are fixed in this file and asserted by a test, deliberately.**
`validate.py::GATES` does the same thing for the four validation gates, for the
reason D39 gives: a threshold chosen once the measurement is visible is not a
threshold. If a candidate we like fails, the answer is to change the candidate,
not the number below.

What this screen is **not**: it is not a ranking, and passing it is not evidence
that a candidate is good. It is a necessary condition, not a sufficient one. A
reward can order four fixed policies perfectly and still be unlearnable, badly
scaled, or aligned with the wrong thing at the margin. D47 remains the selection
rule; this only decides what D47 is allowed to consider.

Run it with `python -m rfenv.reward_gate`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from rfenv import baselines as B
from rfenv.baselines.guard import restrict
from rfenv.env import REWARDS, ScanEnv
from rfenv.scenario import EmitterPool

# --------------------------------------------------------------------------- #
# The criteria. Fixed before the first run; see the module docstring.
# --------------------------------------------------------------------------- #

# The four rungs the screen scores. Two sweeping policies either side of the
# question, one degenerate policy, one published adaptive alternative:
#   2   round-robin, equal airtime (D43)   -- the floor
#   5   recency / activity                 -- the bar (EVALUATION.md §5)
#   4   greedy camper, observation-fed     -- the exploit D14 says is available
#   6a  Apfeld "Active RFs" (D45)          -- an adaptive policy we did not write
# Rung 6a is in the set precisely because we did not design it: a reward tuned
# until it happens to like our own two policies would still be caught out by one
# that arrives from a paper.
SCREEN_RUNGS = ("round_robin", "recency", "camper", "apfeld_active_rfs")

BAR = "recency"          # rung 5
FLOOR = "round_robin"    # rung 2
EXPLOIT = "camper"       # rung 4

N_SEEDS = 8              # the brief's "8+ seeds, per seed not averaged"

# **Separation is measured paired and in units of its own noise.** The absolute
# gap is meaningless across candidates -- they have wildly different scales, from
# `reward_balance`'s hundreds to `explore`'s thousands -- so what has to clear a bar is
# the ratio of the mean paired difference to its standard deviation. 2.0 is two
# standard errors of the per-seed difference, the conventional "not noise" line,
# and it is the same quantity D56 got wrong by measuring against the wrong
# policy.
MIN_SEPARATION_SIGMA = 2.0

# ...and it must also be consistent, not merely large on average. A candidate
# that wins by a mile on two seeds and loses on six is not encoding an ordering.
MIN_SEEDS_BAR_ABOVE_FLOOR = 7   # of N_SEEDS

# The camper must sit clearly below **both** sweeping policies. "Far below" is
# read the same way: in units of the pooled per-seed spread, so a candidate
# cannot pass by being noisy.
MIN_EXPLOIT_MARGIN_SIGMA = 1.0


@dataclass(frozen=True)
class Screen:
    """One candidate's result. `passed` is the conjunction of the three checks."""

    reward: str
    scores: dict[str, np.ndarray]
    separation_sigma: float
    seeds_bar_above_floor: int
    exploit_margin_sigma: float

    @property
    def bar_clears_floor(self) -> bool:
        return (self.separation_sigma >= MIN_SEPARATION_SIGMA
                and self.seeds_bar_above_floor >= MIN_SEEDS_BAR_ABOVE_FLOOR)

    @property
    def exploit_is_below(self) -> bool:
        return self.exploit_margin_sigma >= MIN_EXPLOIT_MARGIN_SIGMA

    @property
    def passed(self) -> bool:
        return self.bar_clears_floor and self.exploit_is_below

    @property
    def verdict(self) -> str:
        if self.passed:
            return "PASS"
        why = []
        if self.separation_sigma < MIN_SEPARATION_SIGMA:
            why.append(f"bar-floor {self.separation_sigma:.1f}sigma")
        if self.seeds_bar_above_floor < MIN_SEEDS_BAR_ABOVE_FLOOR:
            why.append(f"only {self.seeds_bar_above_floor}/{N_SEEDS} seeds")
        if not self.exploit_is_below:
            why.append(f"camper {self.exploit_margin_sigma:+.1f}sigma")
        return "FAIL (" + ", ".join(why) + ")"


def score_rung(reward: str, key: str, seeds, pool: EmitterPool) -> np.ndarray:
    """Total episode reward for one rung under one candidate, one value per seed.

    Uses `baselines.make`, never a hand-written stand-in. That distinction is not
    pedantry: D56 was measured against a hand-rolled `step % N_BANDS` sweep
    labelled "round_robin", which is **not** rung 2 -- rung 2 is
    `EQUAL_AIRTIME_CYCLE` (D43), equal airtime per band rather than one dwell per
    band, and the two differ by about 57 reward and 0.14 of coverage. That single
    substitution inverted D56's conclusion.
    """
    out = []
    for seed in seeds:
        env = ScanEnv(pool=pool, reward=reward)
        obs, info = env.reset(seed=seed)
        policy = B.make(key, seed=seed)
        while True:
            obs, _, terminated, _, info = env.step(int(policy(obs, restrict(info))))
            if terminated:
                out.append(env.episode_metrics()["total_reward"])
                break
    return np.asarray(out, dtype=float)


def screen(reward: str, seeds=None, pool: EmitterPool | None = None) -> Screen:
    """Run the screen for one registered reward candidate."""
    seeds = tuple(range(N_SEEDS)) if seeds is None else tuple(seeds)
    pool = pool or EmitterPool.from_train()
    scores = {k: score_rung(reward, k, seeds, pool)
              for k in SCREEN_RUNGS if k in B.SCHEDULERS}

    bar, floor = scores[BAR], scores[FLOOR]
    # Paired per seed -- same scenario draw, same noise -- so the scenario-to-
    # scenario spread cancels instead of swamping the difference.
    diff = bar - floor
    sep = float(diff.mean() / diff.std()) if diff.std() > 0 else float("inf")
    n_above = int((diff > 0).sum())

    exploit = scores.get(EXPLOIT)
    if exploit is None:
        margin = float("inf")
    else:
        worst_sweep = np.minimum(bar, floor)
        gap = worst_sweep - exploit
        margin = float(gap.mean() / gap.std()) if gap.std() > 0 else float("inf")

    return Screen(reward, scores, sep, n_above, margin)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rfenv.reward_gate",
        description="Screen every reward candidate before training on one.")
    ap.add_argument("--rewards", help="comma-separated subset (default: all registered)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args(argv)

    names = args.rewards.split(",") if args.rewards else sorted(REWARDS)
    seeds = tuple(range(args.seeds))
    pool = EmitterPool.from_train()

    print(f"Reward screen -- {len(names)} candidates x {len(SCREEN_RUNGS)} rungs "
          f"x {len(seeds)} seeds")
    print(f"criteria (fixed in rfenv/reward_gate.py, D39 discipline): "
          f"bar-floor >= {MIN_SEPARATION_SIGMA}sigma on >= "
          f"{MIN_SEEDS_BAR_ABOVE_FLOOR}/{N_SEEDS} seeds, "
          f"camper >= {MIN_EXPLOIT_MARGIN_SIGMA}sigma below both\n")

    header = f"{'candidate':<16}" + "".join(f"{k[:11]:>13}" for k in SCREEN_RUNGS)
    print(header + f"{'sep':>8}{'seeds':>7}{'camper':>9}   verdict")
    print("-" * (len(header) + 40))

    results = []
    for name in names:
        if name not in REWARDS:
            print(f"{name:<16} not a registered candidate -- skipped")
            continue
        r = screen(name, seeds=seeds, pool=pool)
        results.append(r)
        cells = "".join(f"{r.scores[k].mean():>13.1f}" if k in r.scores else f"{'--':>13}"
                        for k in SCREEN_RUNGS)
        print(f"{name:<16}{cells}{r.separation_sigma:>8.1f}"
              f"{r.seeds_bar_above_floor:>4}/{len(seeds)}"
              f"{r.exploit_margin_sigma:>9.1f}   {r.verdict}")

    survivors = [r.reward for r in results if r.passed]
    print(f"\nsurvivors -> D47: {', '.join(survivors) if survivors else 'NONE'}")
    if not survivors:
        print("No candidate encodes the ordering the ladder measures. The next work "
              "is reward design, not hyperparameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
