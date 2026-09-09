"""Rung 5 -- the recency / activity heuristic."""

from __future__ import annotations

import numpy as np

from rfenv.baselines._util import _argmax_random_tie
from rfenv.baselines.guard import HIT_RATE, STALENESS
from rfenv.baselines.simple import SWEEP_SLOTS
from rfenv.constants import N_SLOTS


class RecencyActivity:
    """`argmax over bands of (hit rate + gap measured in reference sweeps)`.

    Two of the three observation components (D34) and nothing else. It is the
    smallest policy that can express both halves of the D14 tension at once: hit
    rate is the exploit term, the gap since the last look is the explore term, and
    the whole difficulty of the problem is that optimising either alone gives you
    rung 4 or rung 2.

    **The explore term is measured in sweeps, and that is what makes the rung
    work.** D34's staleness is normalised by the *episode* (600 slots), so it is
    at most 1.0 -- the same range as the hit rate. Added with unit weight, a band
    declaring on every look scores 1 + 0.002 one slot after being visited, which
    no other band can ever beat, and the rung silently collapses into rung 4:
    measured on `config_2` stare, coverage 0.526 against round-robin's 0.895. The
    fix is the denominator, not a weight. A gap divided by the frozen 43-slot
    reference sweep is unbounded, so a neglected band always eventually outranks a
    perfect one, and the policy is restless rather than greedy.

    **The unit weight is then a statement, not a knob**: one sweep of neglect is
    worth as much as a band that declares on every look. Its effect is legible --
    a band with hit rate 1.0 is pulled forward by exactly one sweep in the revisit
    order, so it earns roughly one extra look per cycle rather than the whole
    episode. The 43 slots come from `constants.DWELL_TIMES_S`, which is frozen
    (D3), so nothing here was chosen against a score, and no weight sweep was run:
    that would be tuning a baseline on the metric it is judged by, which would
    make it a weak learned scheduler rather than a benchmark. If the RL rung only
    beats a *tuned* version of this one, that is worth knowing and is a fair
    question for a reviewer -- not something the ladder should hide by tuning it.
    """

    key = "recency"
    weight = 1.0

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, obs, info) -> int:
        # staleness is (t - last visit) / N_SLOTS (D34); re-express it in sweeps.
        gap_sweeps = np.asarray(obs[STALENESS], dtype=np.float64) * N_SLOTS / SWEEP_SLOTS
        score = np.asarray(obs[HIT_RATE], dtype=np.float64) + self.weight * gap_sweeps
        return _argmax_random_tie(score, self.rng)
