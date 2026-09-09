"""Rung 4 -- the greedy static camper, and its truth-fed reference-line twin."""

from __future__ import annotations

import numpy as np

from rfenv.baselines._util import _argmax_random_tie
from rfenv.baselines.guard import HIT_RATE
from rfenv.baselines.simple import BAND_AT_SLOT, SWEEP_SLOTS

PROBE_SWEEPS = 3   # see GreedyCamper


class GreedyCamper:
    """Probe for three sweeps, then park on the busiest band *seen* and never move.

    §5 keeps this rung *precisely* to show that a single metric can be gamed: on
    the train set a camper wins interception ratio and loses censored intercept
    time by a factor of two (D14 amendment), because the band it camps on really
    does hold most of the pulses -- a documented TSRD property, label imbalance
    "at a proportion of up to 99.7%".

    **This camper is observation-fed; D14's was truth-fed.** D14's read the
    occupancy grid at t = 0 and camped on the band with the most illuminations,
    which no fielded receiver can do -- illumination counts are truth-side (D29).
    This one probes on the reference sweep and then camps on the largest observed
    hit rate. The two are not interchangeable and the ladder carries both:
    `camper_oracle` below is D14's, kept as a reference line, and the gap between
    them is a result rather than an implementation detail.

    **Three sweeps of probe, fixed in advance for a stated reason.** After one
    sweep a narrow band has been looked at once, so its observed hit rate is one
    bit and most bands tie at 1.0 -- the camper would then be choosing at random
    among every band that happened to declare. Three is the smallest probe that
    gives the rate more than one bit per band; it costs 129 slots, 21.5% of the
    episode, and that cost is part of what the rung measures. Not searched.

    If nothing declared during the probe there is nothing to camp on, so it probes
    another three sweeps. On a quiet scenario that is a live path, not a defensive
    branch.

    Once camped it never re-evaluates. "Never moves again" is the behaviour under
    test (D14); a camper that drifts is rung 5.
    """

    key = "camper"

    def __init__(self, rng: np.random.Generator, probe_sweeps: int = PROBE_SWEEPS):
        self.rng = rng
        self.probe_slots = int(probe_sweeps) * SWEEP_SLOTS
        self.probe_step = self.probe_slots
        self.camp: int | None = None

    def __call__(self, obs, info) -> int:
        if self.camp is not None:
            return self.camp
        if info["slot"] < self.probe_slots:
            return int(BAND_AT_SLOT[info["slot"]])
        hit_rate = np.asarray(obs[HIT_RATE], dtype=np.float64)
        if hit_rate.max() > 0.0:
            self.camp = _argmax_random_tie(hit_rate, self.rng)
            return self.camp
        self.probe_slots += self.probe_step
        return int(BAND_AT_SLOT[info["slot"]])


class OracleCamper:
    """D14's camper: camp from slot 0 on the band holding the most illuminations.

    **Not deployable, and therefore not a rung.** It reads `grid.C`, so it sits in
    the ladder as a second reference line beside the oracle. It is here for one
    reason: it is the exact policy behind §5's quantified target -- interception
    ratio 57.4%, censored intercept time 23.78 s -- and without it that row cannot
    be reproduced under the frozen environment, only quoted.

    The gap between this and `camper` measures something worth knowing on its own:
    how much of the camper's strength came from knowing, in advance, where the
    pulses were.
    """

    key = "camper_oracle"

    def __init__(self, grid, rng: np.random.Generator):
        self.band = _argmax_random_tie(
            np.asarray(grid.C, dtype=np.float64).sum(axis=1), rng
        )

    def __call__(self, obs, info) -> int:
        return self.band
