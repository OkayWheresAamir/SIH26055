"""Rungs 6a and 6 -- Apfeld et al., adapted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfenv.constants import N_BANDS, N_SLOTS


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
        n_slots: int = N_SLOTS,
    ):
        self.rng = rng
        self.p = params
        self.use_period_estimation = bool(use_period_estimation)
        self.key = "apfeld" if use_period_estimation else "apfeld_active_rfs"

        # `n_slots` is the episode this instance will run in (D76). It is the
        # constant for every 30 s episode, and longer only for a stitched
        # mission -- but it cannot be left as the constant, because `series` is
        # indexed by *absolute* slot below and a long episode would run off the
        # end of it. This rung is rung 6, the serious bar, so it has to survive
        # any episode the rungs it is compared against survive.
        self.n_slots = int(n_slots)
        self.series = np.zeros((N_BANDS, self.n_slots), dtype=np.float64)
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
        visits = list(range(anchor + period, self.n_slots, period))
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
