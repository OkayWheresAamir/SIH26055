"""The baseline ladder of `docs/project/EVALUATION.md` §5.

Seven schedulers and one reference line, all consuming L3 and nothing else. They
live outside the environment on purpose (`ENVIRONMENT_SPEC.md` §Build order): a
baseline that could reach inside `ScanEnv` would be able to cheat, and the whole
value of the ladder is that every rung is scored through the same interface the
RL policy will use.

**Every rung except the oracle is deployable.** A scheduler sees its own scan
history and the receiver's declaration `Y`, and nothing else -- D19, D20, D34.
That constraint is enforced here rather than trusted: `guarded()` strips `info`
down to `OBSERVABLE_INFO` before the policy sees it, so a baseline that reaches
for `Z`, `pulses` or `first_intercept` raises a `KeyError` instead of quietly
scoring well. The reward-side asymmetry of D29 does not help a baseline: none of
these reads a reward at all.

**The oracle is exempt, and is not a baseline.** `PulseCaptureOracle` is handed
the truth grid at construction. It is the ceiling line of §5, marked as such
everywhere it is reported, and it is never a competitor.

**Every rung is a fresh object per episode.** D20 starts each episode cold -- no
threat library, no carry-over -- so the ladder is a dict of *factories*, not of
policies. Nothing here survives a `reset()`.

**Where the rungs came from.** Rungs 1-5 are D13's set. Rung 3 is Turing's own
frozen `constants.dwell_schedule()`. Rungs 6 and 6a are Apfeld et al.,
`docs/reference/scheduling/paperSSPD (1).pdf` §II, adapted to a binary-detection
receiver -- see `Apfeld` for exactly what was adapted and why. Rung 2 was
*changed* by D43: as written, `EVALUATION.md` §5's round-robin and Turing sweep
were the same policy and reproduced each other to four decimals (D36), so the
ladder had six rungs and claimed seven.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rfenv.constants import (
    DWELL_SLOTS,
    N_BANDS,
    N_SLOTS,
    SLOT_S,
    dwell_schedule,
    slots_for_dwell,
)

# --------------------------------------------------------------------------- #
# What a scheduler is allowed to see
# --------------------------------------------------------------------------- #
#
# The observation vector's layout (D34), named once so no policy indexes it with
# a bare integer. `env.ScanEnv._observation` builds it and
# `tests/test_baselines.py::test_the_observation_slices_match_the_environment`
# asserts these slices still describe it.

HIT_RATE = slice(0, N_BANDS)          # declared hits per slot looked, per band
VISIT_DENSITY = slice(N_BANDS, 2 * N_BANDS)      # fraction of airtime spent there
STALENESS = slice(2 * N_BANDS, 3 * N_BANDS)      # (t - last visit) / N_SLOTS
CLOCK = 3 * N_BANDS                   # normalised episode time

# The keys of `info` a deployable scheduler may read. `slot`/`time_s` are the
# receiver's own clock; `band`, `dwell_slots` and `Y` are what its last look did
# and what it declared. Everything else in `info` is truth-side and belongs to
# the evaluator (D29): `Z`, `pulses`, `newly_intercepted`, `first_intercept`,
# `detectable`, `n_detectable`, `emitter_table`, `total_pulses`.
OBSERVABLE_INFO = frozenset({"slot", "time_s", "band", "dwell_slots", "Y"})


def restrict(info: dict) -> dict:
    """`info` as a deployable scheduler is allowed to see it."""
    return {k: v for k, v in info.items() if k in OBSERVABLE_INFO}


def guarded(policy: Callable[[np.ndarray, dict], int]):
    """Wrap a policy so it is *given* only what it is allowed to read.

    Enforcement, not documentation. A policy that reaches for a truth key gets a
    `KeyError` on the first step, in a test, rather than an unexplained good
    score in a results table.
    """
    return lambda obs, info: policy(obs, restrict(info))


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


# --------------------------------------------------------------------------- #
# Rung 4 -- Greedy static camper
# --------------------------------------------------------------------------- #

SWEEP_SLOTS = int(sum(DWELL_SLOTS))   # 43 -- one full reference sweep, 2.15 s
PROBE_SWEEPS = 3                      # see GreedyCamper


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


def _argmax_random_tie(scores: np.ndarray, rng: np.random.Generator) -> int:
    """`argmax` with ties broken by the policy's own RNG.

    Ties are not an edge case here: at slot 0 every band has hit rate 0 and
    staleness 1, so a plain `np.argmax` would hand band 0 a permanent head start
    and turn an index policy into a band-order artefact.
    """
    best = np.flatnonzero(scores >= scores.max())
    return int(best[0] if best.size == 1 else best[rng.integers(best.size)])


# --------------------------------------------------------------------------- #
# Rung 5 -- Recency / activity heuristic
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Rungs 6a and 6 -- Apfeld et al., adapted
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ApfeldParams:
    """Apfeld §II's parameters. The paper names every one and fixes none of them.

    So these are *our* values, chosen once from the geometry of our episode and
    never searched -- the same rule rung 5's weight follows, and for the same
    reason. Each is justified below by something other than the score it produces.
    """

    # "the receiver stays tuned to that RF band for another d dwells" (§II).
    # 2 extra looks: enough for the autocorrelation to have a second sample of a
    # short period, cheap enough that a false alarm costs 3 slots, not 10.
    d: int = 2
    # Algorithm 1's scaling parameter and cap: the probability of taking a band
    # off the tentative list is min(|tentative| * y, z). See `_algorithm_1` for
    # why that is the prose's reading and not the printed pseudo-code's.
    # y = 0.1 gives one known-active band a tenth of the receiver's looks -- the
    # smallest share that is unambiguously more than the 1/36 a uniform scheduler
    # would give it, so the rung is exploiting on purpose rather than by accident.
    # z = 0.8 keeps a fifth of all airtime for exploration however much of the
    # spectrum is known active, which is what "avoid dwelling on just very few
    # frequencies with a very high probability" asks for.
    y: float = 0.1
    z: float = 0.8
    # "If the standard deviation of the last j period estimates is smaller than a
    # threshold T_std, the period is declared stable". j = 3 is the smallest
    # number for which a standard deviation means anything; T_std = 1 slot is the
    # clock's own resolution, so "stable" means "agreeing to the resolution of the
    # instrument" rather than to a tuned tolerance.
    j: int = 3
    t_std_slots: float = 1.0
    # "After s misdetections, the period estimate is reset ... thus restarting the
    # exploration phase." Our episode is 30 s against the paper's 5 min, so a long
    # patience would spend the whole episode wrong: s = 2.
    s: int = 2
    # "several dwells are executed around the expected time of the next (local)
    # SNR maximum" -- +-1 slot, the clock resolution again.
    window_slots: int = 1
    # A period estimate needs two periods of evidence to be an estimate rather
    # than an artefact of the series' length, so lags run to half the history.
    max_lag_fraction: float = 0.5


class Apfeld:
    """Apfeld, Charlish and Koch's adaptive search strategy (§II), adapted.

    D13 settled this as the strong non-learning bar: published, on our problem,
    non-learning, and described in enough detail to reimplement. §5: "Beating
    Apfeld is the claim worth making."

    **The algorithm, as implemented.** Start with a random search. When a look
    declares a detection on a band, add that band to the tentative list and stay
    on it for `d` more dwells. Autocorrelate the band's own intercepted series to
    estimate the emitter's scan period (Eq. 2, Eq. 3); once `j` successive
    estimates agree to `T_std`, declare the period stable, drop the band from the
    tentative list, and schedule visits at the last detection plus integer
    multiples of the estimate. A scheduled visit that declares nothing counts as a
    misdetection; after `s` of them the estimate is discarded and the band returns
    to the tentative list. With nothing scheduled, pick a band by Algorithm 1 --
    explore off the tentative list with probability `min(|tentative|*y, z)`,
    otherwise exploit a band on it.

    **Three adaptations, all forced by the environment, none of them optional.**

    1. *SNR time series -> binary detection series.* The paper autocorrelates the
       intercepted SNR. Our receiver declares `Y` and nothing else (D26, D28):
       there is no amplitude in the observation and putting one there would be a
       change to D34. So `f` is the binary declaration series, zero where the
       receiver was tuned elsewhere -- which is exactly how the paper's own `f`
       behaves ("at (most of) the points in time where the SNR is zero, the
       receiver is tuned to a different frequency", Fig. 1b). Eq. 3's `argmax`
       over non-zero lags is unchanged.

    2. *Scheduling anchor: last detection, not `SNR_max`.* The paper plans future
       dwells at the last global SNR maximum plus multiples of the period, and
       reschedules when a new maximum appears. Binary declarations have no
       maximum, so the anchor is the most recent declaration on that band. The
       misdetection threshold `T_snr = x * SNR_max` (Eq. 4) collapses with it: a
       scheduled visit either declares or it does not.

    3. *No tracking-dwell handling.* The paper's tracking branch needs the receiver
       to tell search dwells from tracking dwells by waveform. Our PDW stream
       carries no such label and D12 rules out building one. This is therefore the
       paper's own "Adaptive, no tracking" variant -- which §III-B finds performs
       equally well ("scheduling for the tracking dwells doesn't seem to make a
       major difference"), so the reduction costs the comparison nothing that the
       authors themselves measured.

    `use_period_estimation=False` gives Apfeld's third strategy, **"Active RFs"**:
    the tentative list and Algorithm 1 with no period estimation at all. D13 named
    it as the rung that turns the ladder into an ablation -- it isolates how much
    of rung 6 is the period estimate and how much is just "revisit what was loud".
    """

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        use_period_estimation: bool = True,
        params: ApfeldParams = ApfeldParams(),
    ):
        self.rng = rng
        self.p = params
        self.use_period_estimation = bool(use_period_estimation)
        self.key = "apfeld" if use_period_estimation else "apfeld_active_rfs"

        self.series = np.zeros((N_BANDS, N_SLOTS), dtype=np.float64)
        self.tentative: set[int] = set()
        self.stable: dict[int, float] = {}      # band -> period estimate, in slots
        self.estimates: dict[int, list[float]] = {}
        self.scheduled: dict[int, list[int]] = {}
        self.last_detection: dict[int, int] = {}
        self.misses: dict[int, int] = {}
        self.stay = 0
        self.current: int | None = None
        self._from_schedule = False

    # ------------------------------------------------------------ perceiving --

    def _observe(self, info: dict) -> None:
        """Fold the dwell that just finished into the band's own history."""
        band = int(info["band"])
        slot0 = int(info["slot"]) - int(info["dwell_slots"])
        declared = [bool(v) for v in info["Y"]]
        for i, y in enumerate(declared):
            if y:
                self.series[band, slot0 + i] = 1.0
                self.last_detection[band] = slot0 + i

        if any(declared):
            self.misses[band] = 0
            # "Once the SNR on a band crosses the detection threshold ... the
            # receiver stays tuned to that RF band for another d dwells" (§II):
            # on the band *entering* the tentative list, not on every later
            # detection. Re-arming it on every hit makes a dense scenario stick
            # the receiver on the first band it finds -- measured, 11.1% emitter
            # coverage on `config_921` stare.
            if band not in self.stable and band not in self.tentative:
                self.tentative.add(band)
                self.stay = self.p.d
                self.current = band
            if self.use_period_estimation:
                self._update_period(band, upto=slot0 + len(declared))
        elif self._from_schedule:
            self.misses[band] = self.misses.get(band, 0) + 1
            if self.misses[band] >= self.p.s:
                self._forget(band)

    def _update_period(self, band: int, upto: int) -> None:
        """Eq. 2 and Eq. 3 on the band's declaration series."""
        p_hat = _autocorrelation_period(
            self.series[band, :upto], self.p.max_lag_fraction
        )
        if p_hat is None:
            return
        hist = self.estimates.setdefault(band, [])
        hist.append(float(p_hat))
        del hist[: -self.p.j]
        if len(hist) < self.p.j or float(np.std(hist)) >= self.p.t_std_slots:
            return
        # Stable: leave the exploration phase and schedule on the estimate.
        self.stable[band] = float(np.mean(hist))
        self.tentative.discard(band)
        self.misses[band] = 0
        self._reschedule(band, upto)

    def _reschedule(self, band: int, now: int) -> None:
        period = max(1, int(round(self.stable[band])))
        anchor = self.last_detection.get(band, now)
        visits = list(range(anchor + period, N_SLOTS, period))
        self.scheduled[band] = [v for v in visits if v >= now]

    def _forget(self, band: int) -> None:
        self.stable.pop(band, None)
        self.scheduled.pop(band, None)
        self.estimates.pop(band, None)
        self.misses[band] = 0
        self.tentative.add(band)

    # --------------------------------------------------------------- choosing --

    def __call__(self, obs, info) -> int:
        if "band" in info:
            self._observe(info)
        slot = int(info["slot"])
        self._from_schedule = False

        # Stay on a band that just declared, for d more dwells (§II).
        if self.stay > 0 and self.current is not None:
            self.stay -= 1
            return self.current

        # A scheduled visit that is due now, or overdue.
        due = [
            (min(v for v in visits if v >= slot - self.p.window_slots), band)
            for band, visits in self.scheduled.items()
            if any(v >= slot - self.p.window_slots for v in visits)
        ]
        due = [(v, b) for v, b in due if v <= slot + self.p.window_slots]
        if due:
            _, band = min(due)
            self.scheduled[band] = [
                v for v in self.scheduled[band] if v > slot + self.p.window_slots
            ]
            self.current, self._from_schedule = band, True
            return band

        self.current = self._algorithm_1()
        return self.current

    def _algorithm_1(self) -> int:
        """Algorithm 1: exploit the tentative list with probability min(n*y, z).

        **The paper's prose and its printed pseudo-code disagree, and we follow the
        prose.** §II says: "The probability for choosing each tentative frequency is
        scaled by y according to the number of frequencies in the list of tentative
        RFs until the scaled value reaches a maximum z. This is done to avoid
        dwelling on just very few frequencies with a very high probability."  That
        makes `min(n*y, z)` the probability of choosing a **tentative** band, rising
        with the list and capped so exploration never dies. Algorithm 1 as printed
        returns a *non*-tentative band on that branch, which inverts it: with one
        tentative band, `min(1*y, z)` is small, so the receiver would pick that one
        band with probability `1 - y` -- dwelling on exactly one frequency with a
        very high probability, the thing the sentence says the cap exists to
        prevent.

        Measured, the inversion is not cosmetic: implemented as printed with
        y = 1/36, the rung camps on the first band that declares and reaches 11.1%
        emitter coverage on `config_921` stare, against Apfeld's own reported
        result that the adaptive strategies detect most radars and only lose to
        Random on that criterion (§III-B). Recorded as D44.
        """
        others = [b for b in range(N_BANDS) if b not in self.tentative]
        if not self.tentative:
            return int(self.rng.integers(N_BANDS))
        exploit_p = min(len(self.tentative) * self.p.y, self.p.z)
        if self.rng.random() >= exploit_p and others:
            return int(others[self.rng.integers(len(others))])
        pool = sorted(self.tentative)
        return int(pool[self.rng.integers(len(pool))])


def _autocorrelation_period(f: np.ndarray, max_lag_fraction: float) -> float | None:
    """Eq. 3: the lag of the autocorrelation's largest non-zero-lag peak, in slots.

    `f` is the intercepted series -- zero wherever the receiver was tuned
    elsewhere, exactly as in the paper's Fig. 1b. Returns `None` when there is not
    yet enough history for a lag to mean anything: fewer than two declarations, or
    a series too short to contain two periods.
    """
    f = np.asarray(f, dtype=np.float64)
    if f.sum() < 2:
        return None
    max_lag = int(len(f) * max_lag_fraction)
    if max_lag < 1:
        return None
    full = np.correlate(f, f, mode="full")
    positive = full[len(f):][:max_lag]     # lags 1 .. max_lag
    if not positive.size or positive.max() <= 0:
        return None
    return float(np.argmax(positive) + 1)


# --------------------------------------------------------------------------- #
# The ceiling line -- not a baseline
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Rung:
    """One row of `EVALUATION.md` §5, with the factory that builds it."""

    key: str
    rung: str
    label: str
    purpose: str
    factory: Callable
    deployable: bool = True   # False = reads truth, reported as a reference line
    needs_grid: bool = False


LADDER: tuple[Rung, ...] = (
    Rung("random", "1", "Random",
         "Lower bound; the coverage-heavy extreme.",
         lambda rng, grid: RandomBands(rng)),
    Rung("round_robin", "2", "Round-robin (equal airtime)",
         "The open-loop strategy the PS targets. The floor to beat (D43).",
         lambda rng, grid: RoundRobin()),
    Rung("turing_sweep", "3", "Turing reference sweep",
         "The dataset's own schedule; makes our numbers comparable to the recordings.",
         lambda rng, grid: TuringSweep()),
    Rung("camper", "4", "Greedy static (camper)",
         "The degenerate exploit; included to show a single metric can be gamed.",
         lambda rng, grid: GreedyCamper(rng)),
    Rung("recency", "5", "Recency / activity heuristic",
         "Simple adaptive benchmark.",
         lambda rng, grid: RecencyActivity(rng)),
    Rung("apfeld_active_rfs", "6a", "Apfeld: Active RFs",
         "Apfeld's own ablation: the tentative list with no period estimation (D13).",
         lambda rng, grid: Apfeld(rng, use_period_estimation=False)),
    Rung("apfeld", "6", "Apfeld adaptive (no tracking)",
         "Published non-learning adaptive strategy. The serious bar.",
         lambda rng, grid: Apfeld(rng, use_period_estimation=True)),
    Rung("camper_oracle", "—", "Greedy static, truth-fed (D14's camper)",
         "Reference line: D14's camper, which knew where the pulses were.",
         lambda rng, grid: OracleCamper(grid, rng),
         deployable=False, needs_grid=True),
    Rung("oracle_pulse", "—", "Pulse-capture oracle",
         "Ceiling. Not a baseline -- a reference line; reads the truth grid.",
         lambda rng, grid: PulseCaptureOracle(grid, rng),
         deployable=False, needs_grid=True),
)

BY_KEY: dict[str, Rung] = {r.key: r for r in LADDER}

# The seven rungs that are schedulers. `oracle_pulse` is excluded by construction
# rather than by remembering to exclude it.
SCHEDULERS: tuple[str, ...] = tuple(r.key for r in LADDER if r.deployable)


def make(key: str, *, seed: int, grid=None):
    """Build one rung, fresh, for one episode.

    A fresh object every episode is D20 made mechanical: no threat library, no
    carry-over between scenarios, nothing to memorise. The RNG is derived from the
    episode seed and the rung's own name, so two rungs in the same episode do not
    share a random stream and a rung's draw does not depend on where it sits in
    the ladder.

    Returns a callable ready for `metrics.run_episode`. Deployable rungs come back
    wrapped in `guarded()`; the oracle does not, because it has the grid anyway
    and wrapping it would only hide that fact.
    """
    if key not in BY_KEY:
        raise KeyError(f"unknown baseline {key!r}; choose from {sorted(BY_KEY)}")
    spec = BY_KEY[key]
    if spec.needs_grid and grid is None:
        raise ValueError(f"{key} needs the truth grid: make({key!r}, seed=..., grid=grid)")

    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=int(seed), spawn_key=(_key_entropy(key),))
    )
    policy = spec.factory(rng, grid)
    return guarded(policy) if spec.deployable else policy


def _key_entropy(key: str) -> int:
    """A stable per-rung integer, so rung streams are independent but reproducible."""
    return int.from_bytes(key.encode(), "big") % (2 ** 31)


def band_at_slot_from_log(log: list[dict]) -> np.ndarray:
    """The band tuned at each slot, from an episode log. Used by the renders."""
    out = np.full(N_SLOTS, -1, dtype=np.int64)
    for row in log:
        out[row["slot"]] = row["band"]
    return out


def cycle_summary() -> dict:
    """What the two open-loop rungs actually spend, per band. For the write-up.

    Printed by `python -m rfenv.baselines`, so the D43 claim -- equal airtime
    against Turing's 2:1 weighting -- is a command's output rather than a
    sentence in a document.
    """
    sweep = np.zeros(N_BANDS, dtype=np.int64)
    for b in BAND_AT_SLOT:
        sweep[b] += 1
    rr = np.zeros(N_BANDS, dtype=np.int64)
    for b in EQUAL_AIRTIME_CYCLE:
        rr[b] += int(DWELL_SLOTS[b])
    return {
        "turing_sweep_cycle_slots": int(sum(DWELL_SLOTS)),
        "turing_sweep_slots_per_band_per_cycle": {
            "narrow": 1, "wide": 2,
        },
        "turing_sweep_episode_slots_per_band": sweep,
        "round_robin_cycle_slots": EQUAL_AIRTIME_CYCLE_SLOTS,
        "round_robin_slots_per_band_per_cycle": rr,
        "round_robin_cycle_s": EQUAL_AIRTIME_CYCLE_SLOTS * SLOT_S,
        "turing_sweep_cycle_s": float(sum(DWELL_SLOTS)) * SLOT_S,
    }


if __name__ == "__main__":
    summary = cycle_summary()
    print(__doc__)
    print("Rungs:")
    for r in LADDER:
        print(f"  {r.rung:>2}  {r.key:<18} {r.label:<32} "
              f"{'reference line' if not r.deployable else 'scheduler'}")
    print()
    print(f"Turing sweep cycle : {summary['turing_sweep_cycle_slots']} slots "
          f"({summary['turing_sweep_cycle_s']:.2f} s), 1 slot/narrow band, 2/wide")
    print(f"Round-robin cycle  : {summary['round_robin_cycle_slots']} slots "
          f"({summary['round_robin_cycle_s']:.2f} s), "
          f"{set(summary['round_robin_slots_per_band_per_cycle'].tolist())} slot(s) per band")
