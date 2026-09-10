"""Which checkpoint from a training run is *the* model.

A 1M-step run with `--checkpoint-freq 100000` produces ten checkpoints, and they
differ a great deal -- on the 2026-09-10 acceptance run the 100k/200k/300k/400k
snapshots of one run scored 30.4%/43.9%/48.0%/25.7% against rung 5. One of them
has to be picked.

**Picking it by looking at evaluation numbers fits the evaluation set through the
choice.** Ten hypotheses are tried and the best is reported without that being
said, so the reported figure carries an optimistic bias of roughly the spread
across checkpoints -- here, tens of percentage points -- and the bias is invisible
in the number itself. It is D39's argument about thresholds, applied to model
selection: a criterion chosen once the measurement is visible is not a criterion.
It also gets monotonically worse the more we iterate, which is why it is written
down here instead of applied by eye.

So: **the rule is fixed in this file, it runs on the validation half only, and it
picks whatever it picks.** Including when that is not the checkpoint anyone
hoped for.

### The rule

For each checkpoint, over the 12 validation configs (D60 -- zero emitters shared
with the training pool) at `SELECTION_SEEDS` seeds, paired per episode against
rung 5 on the same scenario, same seed, same truth grid:

    net dominance = P(checkpoint Pareto-dominates recency)
                  - P(recency Pareto-dominates checkpoint)

Highest net dominance wins; ties break toward fewer training steps.

Three choices inside that, each load-bearing:

**Against `recency`, not `round_robin`.** Rung 5 is the bar (`EVALUATION.md` §5).
Selecting on the floor picks whichever checkpoint is best at clearing something
the project does not care about.

**Pareto-dominance, not means.** A mean can clear a mean while losing most
scenarios, and D14's whole finding is that no trivial strategy is good at both
objectives -- so "wins on both" is the question and either axis alone is not.

**Net, not the raw win rate.** The two differ, and the difference is the point.
On the acceptance run 300k dominated rung 5 on 48.0% of episodes against 200k's
43.9%, so a win-rate rule picks 300k -- but 300k is *dominated* on 13.5% against
200k's 4.7%, so net picks 200k (+39.2 against +34.5). Being strictly beaten is a
real cost, and a model that rarely loses outright is the better bet for
surviving a held-out run than one that sometimes wins bigger. Ratified by the
human 2026-09-10 after the win-rate version was proposed and corrected.

Ties break toward **fewer** steps: given two checkpoints that cannot be
distinguished on validation, the less-trained one has had less opportunity to
memorise the training pool, and it is cheaper to reproduce.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rfenv import baselines as B
from rfenv.baselines.guard import restrict
from rfenv.env import DEFAULT_REWARD, ScanEnv
from rfenv.scenario import Scenario
from rfenv.split import validation_configs

# The bar. Not the floor -- see the module docstring.
SELECTION_REFERENCE = "recency"

# Seeds per validation config. 12 configs x 3 = 36 paired episodes per checkpoint,
# which is the same seed count the acceptance protocol uses.
SELECTION_SEEDS = (0, 1, 2)

# Where the comparison scenarios come from. Stare replays of the validation
# configs: D36 excludes scan replays from scheduler comparison, and the metrics
# are the §4 headline pair.
SELECTION_SOURCE = "stare"


@dataclass(frozen=True)
class Candidate:
    """One checkpoint's validation result."""

    checkpoint: Path
    steps: int | None
    dominates: float          # fraction of episodes beating rung 5 on both
    dominated: float          # fraction where rung 5 beats it on both
    n_episodes: int

    @property
    def net_dominance(self) -> float:
        """The quantity the rule maximises."""
        return self.dominates - self.dominated


def _episode(env_factory, policy_factory, scenario, seed) -> tuple[float, float]:
    """One episode's (interception ratio, censored mean intercept time)."""
    env = env_factory(scenario)
    obs, info = env.reset(seed=seed)
    policy = policy_factory(seed)
    while True:
        obs, _, terminated, _, info = env.step(int(policy(obs, restrict(info))))
        if terminated:
            m = env.episode_metrics()
            return m["interception_ratio"], m["censored_mean_intercept_time_s"]


def evaluate(policy_factory, *, reward: str = DEFAULT_REWARD,
             configs=None, seeds=SELECTION_SEEDS) -> Candidate:
    """Score one policy against rung 5 on the validation half.

    `policy_factory` takes a seed and returns an `(obs, info) -> int` callable,
    which is exactly the ladder's own scheduler contract -- so a checkpoint, a
    heuristic rung, or a hand-written probe all go through the same path and are
    measured identically.
    """
    configs = tuple(configs) if configs is not None else validation_configs()

    def make_env(sc):
        return ScanEnv(scenario=sc, reward=reward)

    wins = losses = n = 0
    for config_id in configs:
        scenario = Scenario.replay(config_id, SELECTION_SOURCE)
        for seed in seeds:
            mine = _episode(make_env, policy_factory, scenario, seed)
            theirs = _episode(
                make_env,
                lambda s: B.make(SELECTION_REFERENCE, seed=s),
                scenario, seed)
            # Lower intercept time is better; higher ratio is better.
            i_win = mine[0] > theirs[0] and mine[1] < theirs[1]
            i_lose = theirs[0] > mine[0] and theirs[1] < mine[1]
            wins += i_win
            losses += i_lose
            n += 1
    return Candidate(Path("<policy>"), None, wins / n, losses / n, n)


def select(checkpoints, *, reward: str = DEFAULT_REWARD,
           seeds=SELECTION_SEEDS) -> tuple[Candidate, list[Candidate]]:
    """Apply the rule. Returns `(winner, every candidate scored)`.

    Every checkpoint is scored even though only one is returned, because the
    table is the evidence that the rule was applied rather than a result chosen
    and justified afterwards. The iteration ledger records the whole table.
    """
    from rfenv.rl.common import RecurrentRLScheduler, read_manifest
    from rfenv.rl.recurrent_ppo import load_checkpoint

    scored: list[Candidate] = []
    for path in [Path(c) for c in checkpoints]:
        manifest = read_manifest(path) or {}
        model = load_checkpoint(path)
        cand = evaluate(
            lambda s, m=model: RecurrentRLScheduler(m),
            reward=reward, seeds=seeds)
        scored.append(Candidate(path, manifest.get("total_timesteps"),
                                cand.dominates, cand.dominated, cand.n_episodes))
    if not scored:
        raise ValueError("no checkpoints to select from")
    # Highest net dominance; ties toward fewer steps (see the module docstring).
    winner = max(scored, key=lambda c: (c.net_dominance, -(c.steps or 0)))
    return winner, scored


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rfenv.selection",
        description="Pick a checkpoint by the pre-registered validation rule.")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--reward", default=DEFAULT_REWARD)
    args = ap.parse_args(argv)

    configs = validation_configs()
    print(f"selection: net dominance against `{SELECTION_REFERENCE}`, "
          f"{len(configs)} validation configs x {len(SELECTION_SEEDS)} seeds")
    print(f"rule fixed in rfenv/selection.py; validation half is D60's\n")

    winner, scored = select(args.checkpoints, reward=args.reward)
    print(f"{'checkpoint':<38}{'steps':>9}{'domin':>8}{'dom-by':>8}{'NET':>9}")
    print("-" * 72)
    for c in sorted(scored, key=lambda c: -c.net_dominance):
        mark = "  <- selected" if c.checkpoint == winner.checkpoint else ""
        print(f"{c.checkpoint.name:<38}{c.steps or '?':>9}{c.dominates:>8.1%}"
              f"{c.dominated:>8.1%}{c.net_dominance:>+9.1%}{mark}")
    print(f"\nselected: {winner.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
