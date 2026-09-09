"""Rungs 1-3 -- the non-adaptive open-loop baselines: Random, Round-robin, Turing's sweep."""

from __future__ import annotations

import numpy as np

from rfenv.constants import DWELL_SLOTS, N_BANDS, N_SLOTS, dwell_schedule, slots_for_dwell

# --------------------------------------------------------------------------- #
# Rung 1 -- Random
# --------------------------------------------------------------------------- #

class RandomBands:
    """Uniform over the 36 bands, independently each step.

    §5's lower bound and its coverage-heavy extreme: measured on the train set it
    reaches 93.5% coverage and 5.5% interception ratio (D14 amendment). It is the
    control for "did the adaptive part do anything", because it spends the same
    airtime as every other rung and spends it with no information at all.

    Apfeld's own Random "makes sure that each band is selected about the same
    number of times" (paper §III-A), which is a shuffled round-robin rather than
    an i.i.d. draw. Ours is i.i.d.: rung 2 already covers the equal-visit case,
    and keeping them distinct is the point of D43.
    """

    key = "random"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, obs, info) -> int:
        return int(self.rng.integers(N_BANDS))


# --------------------------------------------------------------------------- #
# Rung 2 -- Round-robin, equal airtime (D43)
# --------------------------------------------------------------------------- #
#
# Two passes over the 36 bands. Wide bands cost two slots per visit and are
# visited on the first pass only; narrow bands cost one and are visited on both.
# Every band therefore receives exactly 2 slots per 72-slot (3.6 s) cycle.

def _equal_airtime_cycle() -> list[int]:
    first = list(range(N_BANDS))
    second = [b for b in range(N_BANDS) if DWELL_SLOTS[b] == 1]
    return first + second


EQUAL_AIRTIME_CYCLE = _equal_airtime_cycle()
EQUAL_AIRTIME_CYCLE_SLOTS = int(sum(DWELL_SLOTS[b] for b in EQUAL_AIRTIME_CYCLE))  # 72

# One full reference sweep, in slots (43 = 2.15 s). Shared with `camper.py`
# (the probe length) and `recency.py` (the explore-term's denominator).
SWEEP_SLOTS = int(sum(DWELL_SLOTS))


class RoundRobin:
    """Open-loop, equal airtime per band. The floor the problem statement names.

    **Why this is not Turing's sweep** (D43). D36 measured that "step to the next
    band in order, each for its native dwell" *is* `constants.dwell_schedule()`,
    and it reproduced the sweep's row to four decimal places on both replays --
    so §5's rungs 2 and 3 were one policy counted twice. D36 named three ways to
    separate them: a uniform dwell length, a different starting phase, or a
    shuffled band order. A uniform *dwell* is not expressible -- an action is a
    band and its length is frozen (D3, D16) -- so the expressible form of the same
    idea is uniform **airtime**, which is the currency anyway (D31).

    The difference is a real strategic choice and not a tie-break. Turing's sweep
    gives the seven wide bands double airtime (2 slots per 2.15 s sweep against 1)
    and D36 measured that those bands hold the densest emitter population, so the
    sweep carries a weak prior about where the emitters are. This rung refuses that
    prior: every band gets 2 slots per 3.6 s. It buys uniformity with a 1.67x
    longer revisit period, which is exactly the trade an open-loop scheduler has
    to make with no knowledge, and it makes rungs 2 and 3 two measurements instead
    of one.

    A band's two narrow visits sit ~43 and ~29 slots apart rather than adjacent,
    so the longer cycle does not come with a longer worst-case gap than it has to.
    """

    key = "round_robin"

    def __init__(self):
        self.i = 0

    def __call__(self, obs, info) -> int:
        band = EQUAL_AIRTIME_CYCLE[self.i % len(EQUAL_AIRTIME_CYCLE)]
        self.i += 1
        return band


# --------------------------------------------------------------------------- #
# Rung 3 -- Turing's reference sweep
# --------------------------------------------------------------------------- #

def _band_at_slot() -> np.ndarray:
    band_at = np.zeros(N_SLOTS, dtype=np.int64)
    for start, end, band in dwell_schedule():
        s0, s1 = slots_for_dwell(start, end)
        band_at[s0:s1] = int(band)
    return band_at


BAND_AT_SLOT = _band_at_slot()


class TuringSweep:
    """The dataset's own schedule, `constants.dwell_schedule()`, as a policy.

    Bands 0..35 in ascending frequency, each for its native Turing dwell, 2.15 s
    per sweep, phase 0. It is on the freeze list, so this rung is the one whose
    numbers are directly comparable to the recordings -- and the reason §5 keeps
    it as its own rung rather than folding it into round-robin.

    Indexed by `info["slot"]` rather than by a step counter so it stays the
    reference schedule even if something else has advanced the clock. On this
    environment the two agree exactly, and a test asserts it.
    """

    key = "turing_sweep"

    def __call__(self, obs, info) -> int:
        return int(BAND_AT_SLOT[info["slot"]])
