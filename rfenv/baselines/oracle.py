"""The ceiling line -- not a baseline."""

from __future__ import annotations

import numpy as np

from rfenv.baselines._util import _argmax_random_tie
from rfenv.constants import DWELL_SLOTS, N_BANDS


class PulseCaptureOracle:
    """Greedy on illuminations per slot, with the truth grid in hand.

    §5 lists it under a dash, not a number, and that is load-bearing: **it is not
    a scheduler**. It reads `grid.C`, which no fielded receiver has, and it exists
    to say how much of the pulse stream was capturable at all. Reported as a
    reference line and never as a competitor.

    Greedy per look on captured illuminations **per slot**, since airtime is the
    only currency (D31) -- a wide band must be worth twice a narrow one to be
    worth taking. That makes it myopic, so it is an achievable reference rather
    than a proven bound: the true optimum is a 600-slot scheduling problem and
    D14's own note applies, "each column has a different optimum, which is itself
    the point" -- an intercept-time-optimal oracle would look nothing like this.
    """

    key = "oracle_pulse"

    def __init__(self, grid, rng: np.random.Generator):
        self.C = np.asarray(grid.C, dtype=np.float64)
        self.rng = rng

    def __call__(self, obs, info) -> int:
        t = int(info["slot"])
        rate = np.array(
            [
                self.C[b, t: t + int(DWELL_SLOTS[b])].sum() / float(DWELL_SLOTS[b])
                for b in range(N_BANDS)
            ]
        )
        return _argmax_random_tie(rate, self.rng)
