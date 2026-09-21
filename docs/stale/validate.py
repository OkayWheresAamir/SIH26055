"""The four validation gates of `docs/project/EVALUATION.md` §6, as a runnable script.

**Why this module exists.** The 2026-09-03 consistency audit found two headline
numbers produced by scratch scripts whose comparison conventions were never written
down, and named that as this repository's standing risk: "until a gate is a runnable
check, its number is a claim." This is the structural fix. The pass criteria live in
`GATES` below, in code, decided before the first run -- a threshold chosen once the
measurement is visible is not a gate, it is a post-hoc description of whatever the
code did (D39).

`GATES` is deliberately not a config file and not a set of CLI flags. A threshold you
can pass on the command line is a threshold you can tune after seeing the number.

**What each gate is, and what settles it.**

1. **Out-of-sample prediction** -- truth from *stare only*, Turing's scan schedule
   replayed through it, scored **per dwell against the raw scan ToA stream** (D37).
   The scan recordings are never used in construction, so this is the one gate that
   is a genuine prediction rather than a fit. **Reported, not gated**: D37 fixes the
   convention and says "whatever `validate.py` returns under it becomes the number",
   and no pass threshold has been decided. Inventing one here would be exactly the
   failure the module exists to prevent, so gate 1 prints MEASURED.

2. **Per-band structure** -- grid built from the scan recording, replayed on the
   schedule that produced it, thresholded not at all. A pipeline self-consistency
   test, not a gamma calibration (D23). Both sides count the same 502 dwells per
   config, so there is **no sampling noise to hide behind** and the tolerance is a
   mechanism bound: dwell boundaries are exact slot multiples, so the only ways the
   two sides can differ are the truncated final dwell and slot bucketing. Hence the
   tight 0.5 pp / 1.0 pp in `GATES`, and hence 2c -- bucketing can lose a pulse at a
   boundary but can never invent one, so the discrepancy has a predicted sign.

3. **Theory** -- a controlled periodic case against Koksal's closed forms
   (`docs/reference/scheduling/optimumsearch.pdf`; body p.18 for Eq. 3.8 and body
   p.72 for Table 6-1, which are PDF pp. 33 and 87 -- the thesis has 15 pages of
   front matter). See `_KOKSAL` below for what those pages actually say and for the
   one form that is **not** usable as a criterion.

   **Gate 3 is an environment check and never a scheduler claim.** Koksal assumes
   "a pre-knowledge about radars to be intercepted is available"; D20 starts every
   episode with zero prior emitter knowledge, which is the problem statement's own
   title condition. These do not conflict -- the pre-knowledge in this gate is the
   analyst's, never the agent's -- but the distinction is easy to lose, and "our
   system matches Koksal" would be a claim the architecture does not support.

4. **Extremes** -- `config_81` (2 transmitters) and `config_921` (99). "Sensibly" is
   not a predicate, so this is a fixed list of assertions, each of which restates a
   decision (D28, D35, D38) or a metric definition. They are true or the code is
   wrong; none of them has a tuneable tolerance.

**Scope.** Train split only -- `allow_heldout` is never passed anywhere in this
module, so `scenario._guard_heldout` is the enforcement rather than my discipline.
Gate 4 drives the environment with `constants.dwell_schedule()`, which is Turing's
own schedule and already frozen; **no baseline policy is implemented here.** The
baseline ladder (`EVALUATION.md` §5) comes after the gates pass and the environment
freezes.

**Scan replays appear in gates 1 and 2 only** (D36). There the reference sweep's
footprint is the mechanism under test, not a contaminant. Gate 4 is behavioural and
runs on stare replays.

Usage::

    python -m rfenv.validate                       # all four gates
    python -m rfenv.validate --gate 3,4            # a subset
    python -m rfenv.validate --out runs/validation --seed 0

Exits non-zero if any *gated* check fails. Gate 1 is MEASURED and never fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from rfenv.constants import (
    BAND_CENTRES_MHZ,
    BAND_HALFWIDTH_MHZ,
    DWELL_TIMES_S,
    EPISODE_S,
    GAMMA_DBM,
    N_BANDS,
    N_SLOTS,
    NOISE_SIGMA_DB,
    SLOT_S,
    SWEEP_S,
    dwell_schedule,
    slots_for_dwell,
)
from rfenv.env import ScanEnv
from rfenv.metrics import read_run, scheduler_metrics, write_run
from rfenv.receiver import Receiver, operating_point
from rfenv.rollout import run_episode
from rfenv.scenario import (
    EmitterContribution,
    Scenario,
    list_configs,
    load_contributions,
    recording_path,
)
from rfenv.truth import TruthGrid

# --------------------------------------------------------------------------- #
# The criteria. Fixed before the first run (D39).
# --------------------------------------------------------------------------- #
#
# Every number below is justified by something other than the measurement it
# judges -- that is D39's requirement, and where it could not be met the check is
# reported rather than gated.

GATES: dict[str, dict] = {
    "1": {
        "name": "out-of-sample prediction",
        "gated": False,
        # D37 fixes the convention and leaves the threshold undecided. A number
        # invented here would be the very thing this module exists to prevent.
        "criterion": "MEASURED -- convention fixed by D37; no pass threshold decided",
    },
    "2": {
        "name": "per-band structure",
        "gated": True,
        # A self-consistency test over the same 502 dwells on both sides: no
        # sampling, so the tolerance is a mechanism bound rather than a standard
        # error. 0.5 pp is ~2.5 dwells per config out of 502; 1.0 pp is ~6 dwells
        # per band out of the 611-658 each band gets across the 47 configs.
        "aggregate_pp": 0.5,
        "per_band_pp": 1.0,
        # Bucketing can lose a pulse at a boundary; it cannot invent one.
        "direction": "replayed <= recorded",
    },
    "3": {
        "name": "theory (Koksal)",
        "gated": True,
        # The environment must reproduce the *discretised* geometry. The reference
        # is `_koksal_discrete_p12`, an independent pure-numpy model of the same two
        # pulse trains that imports nothing from L1/L2 -- so the tolerance measures
        # the environment, not the clock. The clock's own bias against the
        # continuous closed form is reported separately (3c) rather than hidden.
        #
        # 0.01 = the clock term for this case (0.0027, computed from the numpy
        # model) plus ~3.7x margin for the phase sweep being 60 offsets rather than
        # continuous. The detector contributes nothing: the emitter sits at
        # gamma + 5 sigma, so the per-look miss probability is Phi(-5) = 2.9e-7.
        "p12_abs_tol": 0.01,
        # Eq. (3.8), read off body p.18. A lower bound on the *maximum* intercept
        # time over phase, so the check is one-sided.
        "eq38_one_sided": True,
    },
    "4": {
        "name": "extremes",
        "gated": True,
        # Each assertion restates a decision or a metric definition. No tolerances.
        # Five per config (D28, §4 ranges, censoring, D35, D38) plus two orderings
        # over the pair: 5 x 2 + 2 = 12.
        "configs": ("config_81", "config_921"),
        "assertions": 12,
    },
}


# --------------------------------------------------------------------------- #
# Koksal's closed forms, transcribed from the primary
# --------------------------------------------------------------------------- #
#
# Read visually from `docs/reference/scheduling/optimumsearch.pdf` rendered at
# 150 dpi, not from a text extraction and not from a search snippet -- the text
# layer mangles subscripts (`T_rcv` extracts as "rcv \n T"), which is exactly why
# CLAUDE.md requires opening the primary at the cited page.
#
# The thesis has 15 pages of front matter, so body page N is PDF page N + 15.
#
# `EVALUATION.md` §6 cites "ch. 3.2, 6.1". What is actually there:
#
#   * **Eq. (3.8), body p.18 (PDF p.33).** The last line of §3.1, immediately
#     before §3.2, and the equation §3.2's Figure 3-5 plots as "theoretical lower
#     bound for intercept time":
#
#         intercept time >= T_rcv / eps = (T_emit * T_rcv) / (tau_emit + tau_rcv)
#
#     A lower bound on the maximum intercept time over phase -- not a prediction of
#     intercept time. Used here as a one-sided check.
#
#   * **§3.2.1-3.2.4, body pp.19-27.** Diophantine approximation and Farey series.
#     A *procedure*, not a closed form: §3.2.3 ends "the intercept time is
#     T_rcv x (Q + q - kappa*q)" where p, q, P, Q and kappa come from (3.13)/(3.14).
#     Implementing it is a small number-theory project and it is not used here.
#     (Its step 1 also says "calculate alpha as in (2.3)", but (2.3) on body p.12 is
#     `duty cycle = tau_rcv / T_rcv` -- the thesis's own cross-reference is wrong,
#     which is a further reason not to build a gate on that path.)
#
#   * **Table 6-1, body p.72 (PDF p.87).** The usable closed form, attributed to
#     Hatcher [8]. Transcribed in `_koksal_p12_first` below.
#
# **The form that is deliberately NOT a criterion.** Table 6-1 also gives
#
#     P12(T) = 1 - [1 - P12(T1)]^(T/T1)
#
# which compounds the first-period probability as though successive receiver
# periods were independent Bernoulli trials -- body p.71 states the assumption in
# Koksal's own words, that Hatcher derived these "assuming that the starting time
# instants of the pulse trains are independent". For two *strictly periodic* trains
# at a fixed phase they are not independent at all: the relative phase drifts
# deterministically and sweeps the phase space systematically, so coverage is far
# faster than an independence model allows. Measured over the four candidate cases,
# every phase intercepts within 30 s (P = 1.0000) against Koksal's 0.751-0.938.
# Gating on P12(T) would fail a correct environment. It is reported as 3c with its
# cause, and Koksal's own ch. 3 -- which yields finite *guaranteed* intercept times
# for the deterministic case -- is the consistent statement (D40).

_KOKSAL = "docs/reference/scheduling/optimumsearch.pdf"


def _koksal_p12_first(tau1: float, T1: float, tau2: float, T2: float) -> float:
    """Table 6-1's `P12(T1)`: coincidence during the first period of train 1.

    Train 1 must be the **shorter-period** train (`T1 <= T2`); `tau` is pulse width.
    The four cases are the table's two rows by two columns, transcribed verbatim
    from body p.72.
    """
    if T1 > T2:
        raise ValueError(f"train 1 must have the shorter period: T1={T1} > T2={T2}")
    if tau1 <= T2 - T1:
        if tau2 <= T1:
            return (tau1 + tau2 * (1.0 - tau2 / (2.0 * T1))) / T2
        return (tau1 + T1 / 2.0) / T2
    if tau2 <= T1:
        return 1.0 - ((T1 - tau2) ** 2 + (T2 - tau1) ** 2) / (2.0 * T1 * T2)
    return 1.0 - (T2 - tau1) ** 2 / (2.0 * T1 * T2)


def _koksal_p12_over_T(tau1: float, T1: float, tau2: float, T2: float, T: float) -> float:
    """Table 6-1's `P12(T)`. **Reported only** -- see the module note above."""
    return 1.0 - (1.0 - _koksal_p12_first(tau1, T1, tau2, T2)) ** (T / T1)


def _koksal_eq38(T_emit: float, T_rcv: float, tau_emit: float, tau_rcv: float) -> float:
    """Eq. (3.8), body p.18: the lower bound on **maximum** intercept time."""
    return T_emit * T_rcv / (tau_emit + tau_rcv)


def _koksal_discrete_p12(
    tau1: float, T1: float, tau2: float, T2: float, slot: float = SLOT_S
) -> float:
    """`P12(T1)` for the same two trains on a slot grid, by exhaustive phase sweep.

    **This is the gate-3 reference, and it deliberately imports nothing from L1 or
    L2.** Holding the environment directly to the continuous closed form would
    conflate two different questions -- "is the environment right" and "is our 50 ms
    clock coarse" -- and would make the tolerance underivable. This function answers
    the second; the gate measures the environment against it and reports the
    difference between it and the closed form separately.

    Every whole-slot phase of train 2 within its period is enumerated, so this is
    exact for the discretised problem rather than sampled.
    """
    P1, W1 = int(round(T1 / slot)), int(round(tau1 / slot))
    P2, W2 = int(round(T2 / slot)), int(round(tau2 / slot))
    hits = 0
    for phase in range(P2):
        train1 = np.zeros(P1, dtype=bool)
        train1[:W1] = True  # train 1's pulse opens its first period
        train2 = np.zeros(P1, dtype=bool)
        for k in range(-P2, P1, P2):
            a = k + phase
            if a + W2 > 0:
                train2[max(a, 0) : a + W2] = True
        hits += bool((train1 & train2).any())
    return hits / P2


# --------------------------------------------------------------------------- #
# Result plumbing
# --------------------------------------------------------------------------- #

@dataclass
class GateResult:
    """One gate's outcome: what was asked, what was measured, and whether it passed.

    `gated=False` means MEASURED -- the gate reports a number and cannot fail.
    Only gate 1 is in that state, because D37 fixed its convention and left its
    threshold undecided.
    """

    gate: str
    name: str
    gated: bool
    passed: bool | None
    criterion: dict
    measured: dict
    checks: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.gated:
            return "MEASURED"
        return "PASS" if self.passed else "FAIL"

    def to_json(self) -> dict:
        return {
            "gate": self.gate,
            "name": self.name,
            "status": self.status,
            "gated": self.gated,
            "passed": self.passed,
            "criterion": self.criterion,
            "measured": self.measured,
            "checks": self.checks,
            "notes": self.notes,
        }


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    """Matthews correlation. The one figure a "predict nothing" model cannot game.

    `EVALUATION.md` §7 forbids reporting bare accuracy: with sparse occupancy,
    always predicting "no transmission" scores well and is operationally useless.
    """
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else float("nan")


def _rates(tp: int, fp: int, tn: int, fn: int) -> dict:
    n = tp + fp + tn + fn
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "mcc": _mcc(tp, fp, tn, fn),
        "base_rate": (tp + fn) / n if n else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Shared: what the raw scan recording actually contains, per dwell
# --------------------------------------------------------------------------- #

def _raw_pulses(config_id: str, source: str = "scan") -> tuple[np.ndarray, np.ndarray]:
    """`(toa_s, freq_mhz)` for one recording, straight out of the HDF5.

    Deliberately **not** `scenario.build_contributions`, which buckets pulses into
    (band, slot) cells. Gate 1's whole claim is that it is a prediction rather than
    a fit (`EVALUATION.md` §6), and every construction step inserted between the
    prediction and the data weakens it -- so the observation side goes to the raw
    ToA stream and shares no band-assignment or slot-clock code with the prediction
    side (D37, alternative 2).

    `data` columns are ToA in microseconds, frequency in MHz, then AoA, PW and
    amplitude; only the first two are needed to say whether a dwell heard anything.
    Train split only -- `recording_path` is called with its default split and the
    held-out guard is never bypassed.
    """
    with h5py.File(recording_path(config_id, source), "r") as fh:
        data = fh["data"][:, :2]
    toa_s = data[:, 0].astype(np.float64) / 1e6
    freq_mhz = data[:, 1].astype(np.float64)
    inside = (toa_s >= 0.0) & (toa_s < EPISODE_S)
    return toa_s[inside], freq_mhz[inside]


def _recorded_per_dwell(config_id: str, schedule: np.ndarray) -> np.ndarray:
    """For each dwell of `schedule`, did the raw scan stream hold a pulse in it?

    A pulse counts if its frequency is inside the tuned band's +-500 MHz window
    (D3's measured geometry, the same window the grid uses) and its ToA falls in
    the dwell's half-open interval. Pulses are sorted per band once and located
    with `searchsorted`, so this is two binary searches per dwell rather than a
    scan over ~90k pulses.

    This is the observation side of **both** gate 1 and gate 2. They differ only in
    what they compare against it: gate 1 predicts from a stare-built grid (truth the
    scan recording never touched), gate 2 replays a scan-built grid (the same
    recording, so any disagreement is a pipeline defect rather than a modelling
    error).
    """
    toa_s, freq_mhz = _raw_pulses(config_id, "scan")
    per_band = []
    for centre in BAND_CENTRES_MHZ:
        in_band = np.abs(freq_mhz - centre) <= BAND_HALFWIDTH_MHZ
        per_band.append(np.sort(toa_s[in_band]))

    out = np.zeros(len(schedule), dtype=bool)
    for i, (start, end, band) in enumerate(schedule):
        t = per_band[int(band)]
        lo = np.searchsorted(t, start, side="left")
        hi = np.searchsorted(t, end, side="left")
        out[i] = hi > lo
    return out


def _predicted_per_dwell(grid: TruthGrid, schedule: np.ndarray, rng) -> np.ndarray:
    """For each dwell, does the environment **declare** a detection?

    `Y`, not `S >= gamma`. `EVALUATION.md` §2 defines the model-level metric as
    agreement between "the environment's predicted detection" and the recording,
    and a detection is what the receiver declares -- which is `Y`, with the noise
    draw in it (D4, D26). Using the noiseless occupancy instead would score a
    quantity the environment never actually emits.

    A dwell truncated by the 30 s boundary is scored on the slots it really covers:
    `Receiver.dwell` always looks for the band's full native length, so its `Y` is
    sliced back to the schedule's own slot range.
    """
    receiver = Receiver(GAMMA_DBM, NOISE_SIGMA_DB, rng=rng)
    out = np.zeros(len(schedule), dtype=bool)
    for i, (start, end, band) in enumerate(schedule):
        s0, s1 = slots_for_dwell(start, end)
        out[i] = bool(receiver.dwell(grid, int(band), s0).Y[: s1 - s0].any())
    return out


def _per_band_rate(bands: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Fraction of each band's dwells that were non-empty. NaN for an unvisited band."""
    rate = np.full(N_BANDS, np.nan)
    for b in range(N_BANDS):
        m = bands == b
        if m.any():
            rate[b] = flags[m].mean()
    return rate


# --------------------------------------------------------------------------- #
# Gate 1 -- out-of-sample prediction (D17, D37). MEASURED, not gated.
# --------------------------------------------------------------------------- #

def gate1(configs: list[str], seed: int) -> GateResult:
    """Truth from stare only; Turing's scan schedule replayed; scored per dwell.

    The only gate that is a genuine prediction rather than a fit: the scan
    recordings are never used in construction (D17), so nothing on the prediction
    side has seen the data on the observation side. It is also why no physics signal
    model is fitted to these same recordings (D25) -- that would turn this gate into
    a fit.

    **Scored per dwell, not per cell** (D37, alternative 1). A dwell is the unit a
    receiver produces a declaration for, and scoring all 36x600 cells would put ~97%
    of the denominator on cells the scan recording could never have observed -- the
    sweep visits 2.78% of them (D36) -- making most of the score unfalsifiable by
    construction. It is also the accounting gate 2 already uses, so the two numbers
    can be read side by side.

    **Known limitation, stated rather than patched.** Band 0 (250 MHz) is heavily
    occupied in the recordings and cannot be predicted at all, because stare's
    `freq_range_mhz` starts at 500 MHz (D10, D24). The per-band bar plot shows it.
    """
    schedule = dwell_schedule()
    bands = schedule[:, 2].astype(int)
    rng = np.random.default_rng(seed)

    tp = fp = tn = fn = 0
    pred_all, obs_all, band_all = [], [], []
    per_config = []
    for config_id in configs:
        grid = TruthGrid.from_scenario(Scenario.replay(config_id, "stare"))
        pred = _predicted_per_dwell(grid, schedule, rng)
        obs = _recorded_per_dwell(config_id, schedule)
        tp += int((pred & obs).sum())
        fp += int((pred & ~obs).sum())
        tn += int((~pred & ~obs).sum())
        fn += int((~pred & obs).sum())
        pred_all.append(pred)
        obs_all.append(obs)
        band_all.append(bands)
        per_config.append({
            "config": config_id,
            **_rates(int((pred & obs).sum()), int((pred & ~obs).sum()),
                     int((~pred & ~obs).sum()), int((~pred & obs).sum())),
        })

    pred_all = np.concatenate(pred_all)
    obs_all = np.concatenate(obs_all)
    band_all = np.concatenate(band_all)

    pred_band = _per_band_rate(band_all, pred_all)
    obs_band = _per_band_rate(band_all, obs_all)
    ok = ~np.isnan(pred_band) & ~np.isnan(obs_band)
    # Pearson r over the 36 band rates -- "not just the aggregate", the same
    # question gate 2 asks, asked here about the prediction.
    r = float(np.corrcoef(pred_band[ok], obs_band[ok])[0, 1]) if ok.sum() > 1 else float("nan")

    overall = _rates(tp, fp, tn, fn)
    return GateResult(
        gate="1",
        name=GATES["1"]["name"],
        gated=False,
        passed=None,
        criterion={"convention": "per dwell, against the raw scan ToA stream (D37)",
                   "threshold": GATES["1"]["criterion"]},
        measured={
            **overall,
            "per_band_pearson_r": r,
            "n_configs": len(configs),
            "n_dwells_per_config": len(schedule),
            "seed": seed,
            "gamma_dbm": GAMMA_DBM,
            "band0_predicted": float(pred_band[0]),
            "band0_recorded": float(obs_band[0]),
            "per_band_predicted": pred_band.tolist(),
            "per_band_recorded": obs_band.tolist(),
        },
        checks=per_config,
        notes=[
            "MEASURED, not gated: D37 fixed the convention and left the threshold "
            "undecided; a number invented here would be the failure this module exists "
            "to prevent.",
            f"Band 0 is {obs_band[0]:.2%} occupied in the recordings and "
            f"{pred_band[0]:.2%} predicted -- stare cannot see below 500 MHz (D10, D24). "
            "Stated, not patched.",
            "Never report bare accuracy (EVALUATION.md §7): MCC is the figure a "
            "'predict nothing' model cannot game.",
        ],
    )


# --------------------------------------------------------------------------- #
# Gate 2 -- per-band structure (D23). Self-consistency, tightly gated.
# --------------------------------------------------------------------------- #

def gate2(configs: list[str]) -> GateResult:
    """Grid built from the scan recording, replayed on the schedule that produced it.

    Thresholded not at all: this asks whether the pipeline round-trips, so `gamma`
    is not in it. It is **not** a gamma calibration -- that procedure was confounded
    and is retracted (D23).

    The scan replay is correct here and only here (with gate 1). D36 showed a
    scan-derived grid carries the reference sweep's own footprint, which disqualifies
    it for scheduler comparison; for this gate the shared origin is exactly the point.

    Both sides count the same dwells, so there is no sampling noise and the
    tolerances in `GATES` are mechanism bounds rather than standard errors. Every
    dwell boundary is an exact multiple of the 50 ms slot, so the only ways the two
    sides can disagree are the dwell truncated at 30 s and slot bucketing at the
    edges. 2c predicts the *sign* of any residual: `_bucket` floors a ToA into a
    slot and `bands_covering` uses the same window as the recording check, so
    bucketing can drop a pulse at a boundary but cannot invent one.
    """
    schedule = dwell_schedule()
    bands = schedule[:, 2].astype(int)

    rep_all, rec_all, band_all, per_config = [], [], [], []
    for config_id in configs:
        grid = TruthGrid.from_scenario(Scenario.replay(config_id, "scan"))
        replayed = np.zeros(len(schedule), dtype=bool)
        for i, (start, end, band) in enumerate(schedule):
            s0, s1 = slots_for_dwell(start, end)
            replayed[i] = bool(grid.C[int(band), s0:s1].any())
        recorded = _recorded_per_dwell(config_id, schedule)
        rep_all.append(replayed)
        rec_all.append(recorded)
        band_all.append(bands)
        per_config.append({
            "config": config_id,
            "replayed": float(replayed.mean()),
            "recorded": float(recorded.mean()),
            "delta_pp": float((replayed.mean() - recorded.mean()) * 100.0),
        })

    rep_all = np.concatenate(rep_all)
    rec_all = np.concatenate(rec_all)
    band_all = np.concatenate(band_all)

    agg_rep, agg_rec = float(rep_all.mean()), float(rec_all.mean())
    agg_delta_pp = (agg_rep - agg_rec) * 100.0

    rep_band = _per_band_rate(band_all, rep_all)
    rec_band = _per_band_rate(band_all, rec_all)
    band_delta_pp = (rep_band - rec_band) * 100.0
    worst = int(np.nanargmax(np.abs(band_delta_pp)))

    crit = GATES["2"]
    c_agg = abs(agg_delta_pp) <= crit["aggregate_pp"]
    c_band = float(np.nanmax(np.abs(band_delta_pp))) <= crit["per_band_pp"]
    c_dir = agg_rep <= agg_rec

    return GateResult(
        gate="2",
        name=crit["name"],
        gated=True,
        passed=bool(c_agg and c_band and c_dir),
        criterion={
            "2a aggregate": f"|replayed - recorded| <= {crit['aggregate_pp']} pp",
            "2b per band": f"max over 36 bands |replayed - recorded| <= {crit['per_band_pp']} pp",
            "2c direction": crit["direction"],
        },
        measured={
            "aggregate_replayed": agg_rep,
            "aggregate_recorded": agg_rec,
            "aggregate_delta_pp": agg_delta_pp,
            "max_band_delta_pp": float(np.nanmax(np.abs(band_delta_pp))),
            "worst_band": worst,
            "worst_band_centre_mhz": float(BAND_CENTRES_MHZ[worst]),
            "n_configs": len(configs),
            "n_dwells": int(rep_all.size),
            "per_band_replayed": rep_band.tolist(),
            "per_band_recorded": rec_band.tolist(),
        },
        checks=[
            {"check": "2a aggregate", "passed": bool(c_agg),
             "value_pp": agg_delta_pp, "limit_pp": crit["aggregate_pp"]},
            {"check": "2b per band", "passed": bool(c_band),
             "value_pp": float(np.nanmax(np.abs(band_delta_pp))), "limit_pp": crit["per_band_pp"]},
            {"check": "2c direction", "passed": bool(c_dir),
             "value": crit["direction"], "replayed": agg_rep, "recorded": agg_rec},
        ] + per_config,
        notes=[
            "Self-consistency, not calibration (D23). Both sides count the same "
            "dwells, so the tolerance is a mechanism bound, not a standard error.",
            "Scan replay is correct for this gate and disqualified for scheduler "
            "comparison (D36).",
        ],
    )


# --------------------------------------------------------------------------- #
# Gate 3 -- theory (Koksal). An environment check, never a scheduler claim.
# --------------------------------------------------------------------------- #

# The controlled case, fixed in advance. Chosen by stated rules, not by outcome:
#
#   * **Band 0.** Its native dwell is 100 ms = 2 slots. A 1-slot dwell makes the
#     discrete overlap all-or-nothing and inflates the clock's error by an order of
#     magnitude (measured: 0.0365 against 0.0027 for a 2-slot dwell), which would
#     swamp what the gate is trying to measure.
#   * **One band only.** Deliberately unlike a real emitter, which straddles two
#     overlapping windows (D3) -- a controlled case needs an unambiguous alpha.
#   * **tau_emit = 0.50 s, T_emit = 3.00 s.** alpha = T_emit / T_rcv = 60/43 in
#     slots: a large Farey denominator, so no synchronisation and no infinite
#     intercept time, and T0(P=0.9) = 24.9 s fits inside the 30 s horizon.
#   * **Level gamma + 5 sigma.** Gate 3 tests timing geometry, not the detector.
#     At 5 sigma the per-look miss probability is Phi(-5) = 2.9e-7.
_G3_BAND = 0
_G3_TAU_EMIT_S = 0.50
_G3_T_EMIT_S = 3.00
_G3_LEVEL_DBM = GAMMA_DBM + 5.0 * NOISE_SIGMA_DB


def _g3_contribution(phase_slots: int) -> EmitterContribution:
    """One periodic emitter in band 0, offset by `phase_slots`, as a contribution.

    Synthetic: no HDF5, no pool, no sampling. `total_pulses` counts one illumination
    per occupied slot, which is all the interception-ratio denominator needs here.
    """
    period = int(round(_G3_T_EMIT_S / SLOT_S))
    width = int(round(_G3_TAU_EMIT_S / SLOT_S))
    on = np.zeros(N_SLOTS, dtype=bool)
    for k in range(-period, N_SLOTS, period):
        a = k + phase_slots
        if a + width > 0:
            on[max(a, 0) : a + width] = True
    slots = np.flatnonzero(on).astype(np.int16)
    cells = np.stack([np.full(slots.size, _G3_BAND, dtype=np.int16), slots], axis=1)
    return EmitterContribution(
        config_id="synthetic", source="koksal", label=0,
        cells=cells,
        peak_dbm=np.full(slots.size, _G3_LEVEL_DBM, dtype=np.float32),
        n_pulses=np.ones(slots.size, dtype=np.int32),
        total_pulses=int(slots.size),
    )


def _g3_sweep_policy():
    """Turing's own schedule as a `policy(obs, info) -> band`.

    `constants.dwell_schedule()` is the frozen reference sweep, so this is a lookup
    into it rather than a baseline: `EVALUATION.md` §5's ladder lives outside the
    environment and is not this module's business.
    """
    band_at = np.zeros(N_SLOTS, dtype=int)
    for start, end, band in dwell_schedule():
        s0, s1 = slots_for_dwell(start, end)
        band_at[s0:s1] = int(band)
    return lambda obs, info: int(band_at[info["slot"]])


def gate3(seed: int) -> GateResult:
    """A controlled periodic case against Eq. (3.8) and Table 6-1.

    Independent of the dataset entirely -- that is the point of the gate. The
    emitter is synthesised, so nothing here touches `data/`.

    **3a** compares the environment's phase-averaged `P12(T1)` against
    `_koksal_discrete_p12`, a pure-numpy model of the same two pulse trains. The
    environment must reproduce the *discretised* geometry to 0.01. **3b** checks
    Eq. (3.8)'s one-sided bound on the maximum intercept time over phase. **3c** is
    reported and not gated: how far the discretised geometry sits from the
    continuous closed form (the clock's own bias), and how badly `P12(T)`'s
    cross-period independence assumption fails for a deterministic periodic pair.

    Splitting 3a from 3c is what makes the tolerance derivable. Held directly to the
    continuous closed form, the gate would be measuring the clock and the environment
    at once and neither number would mean anything on its own.
    """
    tau_rcv, T_rcv = float(DWELL_TIMES_S[_G3_BAND]), SWEEP_S
    tau_emit, T_emit = _G3_TAU_EMIT_S, _G3_T_EMIT_S
    # Table 6-1 indexes train 1 as the shorter period.
    if T_rcv <= T_emit:
        t1, T1, t2, T2 = tau_rcv, T_rcv, tau_emit, T_emit
    else:
        t1, T1, t2, T2 = tau_emit, T_emit, tau_rcv, T_rcv

    period = int(round(T_emit / SLOT_S))
    first_period_slots = int(round(T1 / SLOT_S))
    policy = _g3_sweep_policy()

    coincided, first_slots = [], []
    for phase in range(period):
        scenario = Scenario(
            name=f"synthetic:koksal/phase{phase}",
            contributions=[_g3_contribution(phase)],
        )
        env = run_episode(ScanEnv(scenario=scenario), policy, seed=seed + phase)
        first = env.first_intercept.get(0)
        first_slots.append(N_SLOTS if first is None else first)
        # P12(T1): a coincidence during the first period of the shorter train.
        coincided.append(first is not None and first < first_period_slots)

    p12_env = float(np.mean(coincided))
    p12_numpy = _koksal_discrete_p12(t1, T1, t2, T2)
    p12_closed = _koksal_p12_first(t1, T1, t2, T2)
    max_first_s = float(np.max(first_slots)) * SLOT_S
    bound_s = _koksal_eq38(T_emit, T_rcv, tau_emit, tau_rcv)

    crit = GATES["3"]
    c_p12 = abs(p12_env - p12_numpy) <= crit["p12_abs_tol"]
    c_eq38 = max_first_s >= bound_s

    p12_T_koksal = _koksal_p12_over_T(t1, T1, t2, T2, EPISODE_S)
    p12_T_actual = float(np.mean(np.array(first_slots) < N_SLOTS))

    return GateResult(
        gate="3",
        name=crit["name"],
        gated=True,
        passed=bool(c_p12 and c_eq38),
        criterion={
            "3a P12(T1)": f"|env - discrete reference| <= {crit['p12_abs_tol']}",
            "3b Eq(3.8)": f"max first intercept over phase >= {bound_s:.2f} s",
            "3c": "reported, not gated -- clock bias and the P12(T) divergence",
        },
        measured={
            "case": {
                "band": _G3_BAND,
                "tau_rcv_s": tau_rcv, "T_rcv_s": T_rcv,
                "tau_emit_s": tau_emit, "T_emit_s": T_emit,
                "alpha": T_emit / T_rcv,
                "level_dbm": _G3_LEVEL_DBM,
                "n_phases": period,
            },
            "p12_first_env": p12_env,
            "p12_first_discrete_reference": p12_numpy,
            "p12_first_closed_form": p12_closed,
            "p12_env_vs_reference": abs(p12_env - p12_numpy),
            "clock_bias_reference_vs_closed": abs(p12_numpy - p12_closed),
            "max_first_intercept_s": max_first_s,
            "eq38_bound_s": bound_s,
            "mean_first_intercept_s": float(np.mean(first_slots)) * SLOT_S,
            "p12_over_T_koksal": p12_T_koksal,
            "p12_over_T_actual": p12_T_actual,
            "seed": seed,
        },
        checks=[
            {"check": "3a P12(T1)", "passed": bool(c_p12),
             "env": p12_env, "reference": p12_numpy,
             "abs_error": abs(p12_env - p12_numpy), "limit": crit["p12_abs_tol"]},
            {"check": "3b Eq(3.8)", "passed": bool(c_eq38),
             "max_first_intercept_s": max_first_s, "bound_s": bound_s},
            {"check": "3c clock bias", "gated": False,
             "reference_vs_closed": abs(p12_numpy - p12_closed)},
            {"check": "3c P12(T) divergence", "gated": False,
             "koksal": p12_T_koksal, "actual": p12_T_actual},
        ],
        notes=[
            "An ENVIRONMENT check. Koksal assumes prior emitter knowledge; D20 "
            "refuses it. The pre-knowledge here is the analyst's, never the agent's. "
            "No output of this gate is a scheduler result.",
            f"Eq. (3.8) is {_KOKSAL} body p.18 (PDF p.33); Table 6-1 is body p.72 "
            "(PDF p.87). Both read visually from the rendered page.",
            f"P12(T) predicts {p12_T_koksal:.4f} by 30 s; the environment reaches "
            f"{p12_T_actual:.4f}. Hatcher's cross-period independence assumption does "
            "not hold for a deterministic periodic pair -- reported, not gated (D40).",
        ],
    )


# --------------------------------------------------------------------------- #
# Gate 4 -- extremes. Structural assertions, no tolerances.
# --------------------------------------------------------------------------- #

def gate4(out_dir: Path, seed: int) -> GateResult:
    """`config_81` (2 transmitters) and `config_921` (99) must behave sensibly.

    "Sensibly" is not a predicate, so this is a fixed list of assertions, each
    restating a decision or a metric definition rather than carrying a tuneable
    threshold. They hold or the code is wrong.

    **Stare replays** (D36): gate 4 is behavioural, so the scan replay's imprint
    would make it meaningless. Driven by `dwell_schedule()`, Turing's own frozen
    schedule -- no baseline policy is implemented here; the ladder comes after the
    freeze.

    Assertion 5 is D38's artefact contract re-checked at the extremes: the run is
    written to disk and scored back from the files, so a serialisation bug at 2 or
    99 emitters fails loudly instead of scoring low.
    """
    policy = _g3_sweep_policy()
    configs = GATES["4"]["configs"]
    checks: list[dict] = []
    summaries: dict[str, dict] = {}

    for config_id in configs:
        scenario = Scenario.replay(config_id, "stare")
        env = run_episode(ScanEnv(scenario=scenario), policy, seed=seed)
        live = env.episode_metrics()
        run = write_run(out_dir / "runs" / config_id, env,
                        scheduler="turing_reference_sweep", seed=seed)
        scored = scheduler_metrics(read_run(out_dir / "runs" / config_id))
        summaries[config_id] = {
            **{k: v for k, v in scored.items()},
            "total_pulses": int(run.header["total_pulses"]),
        }

        # 1. first_e >= on_e for every intercepted emitter (D28).
        bad = [r["emitter"] for r in run.emitters
               if r["first_intercept_slot"] is not None
               and r["first_intercept_slot"] < r["on_slot"]]
        checks.append({"check": f"{config_id}: first_e >= on_e (D28)",
                       "passed": not bad, "violations": bad})

        # 2. Every reported metric inside its definitional range (§4).
        in_range = (
            0.0 <= scored["interception_ratio"] <= 1.0
            and 0.0 <= scored["emitter_coverage"] <= 1.0
            and 0.0 <= scored["censored_mean_intercept_time_s"] <= EPISODE_S
        )
        checks.append({"check": f"{config_id}: metrics inside definitional range",
                       "passed": bool(in_range),
                       "interception_ratio": scored["interception_ratio"],
                       "emitter_coverage": scored["emitter_coverage"],
                       "censored_mean_intercept_time_s":
                           scored["censored_mean_intercept_time_s"]})

        # 3. No NaN or inf anywhere -- censoring is mandatory, so an unfound
        #    emitter contributes the episode length and never an infinity (§4).
        finite = all(np.isfinite(v) for v in scored.values() if isinstance(v, float))
        checks.append({"check": f"{config_id}: no NaN or inf (censoring is mandatory)",
                       "passed": bool(finite)})

        # 4. 600 slots exactly, 300-600 steps (D35).
        clock = len(run.log) == N_SLOTS and 300 <= scored["n_steps"] <= N_SLOTS
        checks.append({"check": f"{config_id}: 600 slots, 300-600 steps (D35)",
                       "passed": bool(clock),
                       "n_slots": len(run.log), "n_steps": scored["n_steps"]})

        # 5. The artefacts reproduce the environment's own numbers (D38).
        shared = set(live) & set(scored)
        mismatched = {k: (live[k], scored[k]) for k in sorted(shared)
                      if not (isinstance(live[k], float) and np.isnan(live[k]))
                      and live[k] != scored[k]}
        checks.append({"check": f"{config_id}: artefacts reproduce env.episode_metrics() (D38)",
                       "passed": not mismatched, "mismatched": mismatched})

    # 6. Neither grid is degenerate.
    checks.append({"check": "config_81 has at least one detectable emitter",
                   "passed": summaries["config_81"]["n_detectable"] >= 1,
                   "n_detectable": summaries["config_81"]["n_detectable"]})

    # 7. Ordering that follows from the data, not from the environment: 99
    #    transmitters against 2. Deliberately *not* an ordering on coverage --
    #    D28 scores each emitter on its own level, so an emitter's detectability
    #    does not depend on how crowded the scenario is, and that ordering cannot
    #    be derived. It is reported below instead.
    order = (summaries["config_921"]["n_detectable"] > summaries["config_81"]["n_detectable"]
             and summaries["config_921"]["total_pulses"] > summaries["config_81"]["total_pulses"])
    checks.append({"check": "config_921 exceeds config_81 in emitters and illuminations",
                   "passed": bool(order),
                   "n_detectable": [summaries["config_81"]["n_detectable"],
                                    summaries["config_921"]["n_detectable"]],
                   "total_pulses": [summaries["config_81"]["total_pulses"],
                                    summaries["config_921"]["total_pulses"]]})

    gated = [c for c in checks if c.get("passed") is not None]
    return GateResult(
        gate="4",
        name=GATES["4"]["name"],
        gated=True,
        passed=all(c["passed"] for c in gated),
        criterion={"assertions": f"{len(gated)} structural assertions, all must hold",
                   "configs": list(configs),
                   "replay": "stare (D36 -- gate 4 is behavioural)",
                   "schedule": "constants.dwell_schedule()"},
        measured={"n_assertions": len(gated),
                  "n_passed": sum(c["passed"] for c in gated),
                  "summaries": summaries,
                  "seed": seed},
        checks=checks,
        notes=[
            "Driven by Turing's frozen schedule. No baseline policy is implemented "
            "here: the ladder (EVALUATION.md §5) comes after the gates pass and the "
            "environment freezes.",
            "Coverage is reported per config but not asserted to fall with emitter "
            "count -- D28 scores each emitter on its own level, so that ordering does "
            "not follow from the design and would be a threshold read off the answer.",
        ],
    )


# --------------------------------------------------------------------------- #
# Average intercept-time error (EVALUATION.md §2). Model-level, per emitter.
# --------------------------------------------------------------------------- #
#
# The second of the problem statement's two *prediction* figures of merit, and the
# only one of the seven that had a definition (EVALUATION.md §2) but no
# implementation. Added additively for the figures-of-merit report (D70); it changes
# no gate and no frozen constant.


def intercept_time_error(configs: list[str], seed: int) -> dict:
    """`EVALUATION.md` §2's average intercept-time error: `|predicted - measured|`.

    Like gate 1 (`% correct predictions`), this is a **model-level** metric: it asks
    whether the stare-built environment reproduces *Turing's own sweep's* per-emitter
    intercept timing, not whether any scheduler is fast (D70). It is therefore
    policy-independent and reads the same for every rung, exactly as P_d, P_fa and
    sensitivity do (D21, §3) -- the correct outcome, to be stated and not debugged.

    **The two sides, matched by transmitter label.**

    * *Predicted* -- truth from **stare only** (data the scan recording never touched,
      D17), Turing's own `dwell_schedule()` replayed through `ScanEnv`, each detectable
      emitter's `first_intercept_slot` read from the emitter table (D28). An emitter the
      sweep never catches is censored at the 30 s horizon, as §4 censors intercept time.
    * *Measured* -- the first slot that emitter's pulses appear in the **actual scan
      recording** (`scenario.load_contributions(..., "scan")`). A scan recording holds
      only what Turing's sweep was tuned to (D36), so an emitter's first appearance in it
      *is* the sweep's real first intercept of it.

    Both sides live on the 50 ms slot grid, so the error is a whole multiple of 0.05 s.
    Only emitters detectable in stare **and** present in scan are scored; the count kept
    and the counts dropped on each side are all reported.

    **Two limitations, stated not patched** -- the same two gate 1 carries:

    * scan and stare are independent simulation runs, so the same emitter has disjoint
      activity in each (D24). Part of this error is that realisation divergence rather
      than a modelling defect -- but an *out-of-sample* prediction (stare -> scan) is
      exposed to exactly that divergence, so it belongs in the number, not removed from
      it. This is why the figure is larger than the per-dwell agreement of gate 1.
    * band 0 (250 MHz) is heavily occupied in the recordings and invisible to stare
      (D10), so its emitters are predicted as never-intercepted and inflate the error
      wherever scan caught them early.
    """
    schedule_policy = _g3_sweep_policy()
    per_config: list[dict] = []
    all_errors: list[float] = []
    n_matched = n_pred_only = n_meas_only = 0

    for config_id in configs:
        env = run_episode(
            ScanEnv(scenario=Scenario.replay(config_id, "stare")),
            schedule_policy, seed=seed,
        )
        # label -> predicted first-intercept time (s), censored at the horizon.
        predicted: dict[int, float] = {}
        for row in env.emitter_table():
            label = int(row["uid"].rsplit("/", 1)[1])
            slot = row["first_intercept_slot"]
            predicted[label] = (N_SLOTS if slot is None else slot) * SLOT_S
        # label -> measured first-appearance time (s) in the actual scan recording.
        measured = {
            int(c.label): int(c.slots.min()) * SLOT_S
            for c in load_contributions(config_id, "scan")
        }

        labels = set(predicted) & set(measured)
        errors = [abs(predicted[l] - measured[l]) for l in labels]
        all_errors.extend(errors)
        n_matched += len(labels)
        n_pred_only += len(set(predicted) - set(measured))
        n_meas_only += len(set(measured) - set(predicted))
        per_config.append({
            "config": config_id,
            "n_matched": len(labels),
            "mean_abs_error_s": float(np.mean(errors)) if errors else float("nan"),
        })

    errs = np.array(all_errors, dtype=np.float64)
    have = errs.size > 0
    return {
        "metric": "average intercept-time error (EVALUATION.md §2)",
        "mean_abs_error_s": float(errs.mean()) if have else float("nan"),
        "median_abs_error_s": float(np.median(errs)) if have else float("nan"),
        "p25_abs_error_s": float(np.percentile(errs, 25)) if have else float("nan"),
        "p75_abs_error_s": float(np.percentile(errs, 75)) if have else float("nan"),
        "n_emitters_matched": n_matched,
        "n_predicted_not_in_scan": n_pred_only,
        "n_scan_not_predicted": n_meas_only,
        "n_configs": len(configs),
        "seed": seed,
        "gamma_dbm": GAMMA_DBM,
        "policy_independent": True,
        "per_config": per_config,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _detail(res: GateResult) -> str:
    """The measurement, with the threshold it was judged against beside it.

    A number without its criterion is a claim, not a result, so the two are formatted
    together and never separately.
    """
    m = res.measured
    if res.gate == "1":
        return (f"acc {m['accuracy']:.4f}  MCC {m['mcc']:.4f}  "
                f"prec {m['precision']:.4f}  rec {m['recall']:.4f}  "
                f"r {m['per_band_pearson_r']:.3f}  (base rate {m['base_rate']:.4f})")
    if res.gate == "2":
        return (f"agg |d| {abs(m['aggregate_delta_pp']):.3f} pp <= "
                f"{GATES['2']['aggregate_pp']}   max band |d| "
                f"{m['max_band_delta_pp']:.3f} pp <= {GATES['2']['per_band_pp']}   "
                f"replayed {m['aggregate_replayed']:.5f} vs recorded "
                f"{m['aggregate_recorded']:.5f}")
    if res.gate == "3":
        return (f"|d| {m['p12_env_vs_reference']:.4f} <= {GATES['3']['p12_abs_tol']}   "
                f"Eq(3.8) {m['max_first_intercept_s']:.2f} s >= {m['eq38_bound_s']:.2f} s")
    return f"{m['n_passed']}/{m['n_assertions']} assertions"


def _line(res: GateResult) -> str:
    return f"gate {res.gate}  {res.name:<26}  {res.status:<8}  {_detail(res)}"


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell.

    Gate criteria are full of absolute-value bars, which would otherwise be read as
    column separators and shred the table.
    """
    return str(text).replace("|", "\\|")


def _report_md(results: list[GateResult], point: dict, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = [
        "# Validation gates",
        "",
        f"`python -m rfenv.validate` -- {stamp}, seed {seed}, train split only.",
        "",
        "Criteria were fixed in `rfenv/validate.py::GATES` before this ran (D39). "
        "Gate 1 is MEASURED: D37 fixed its convention and left its threshold undecided.",
        "",
        "| gate | name | status | criterion | measured |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        crit = "<br>".join(f"{_cell(k)}: {_cell(v)}" for k, v in r.criterion.items())
        out.append(f"| {r.gate} | {r.name} | **{r.status}** | {crit} | {_cell(_detail(r))} |")
    out += [
        "",
        "## Operating point",
        "",
        f"gamma = {point['gamma_dbm']} dB, sigma = {point['sigma_db']} dB, "
        f"Pd = {point['pd']:.4f}, Pfa = {point['pfa']:.3e}, "
        f"sensitivity = {point['sensitivity_dbm']:.2f} dB, "
        f"population = {point['population']} ({point['n_occupied']} occupied cells).",
        "",
        "## Notes",
        "",
    ]
    for r in results:
        out.append(f"**Gate {r.gate} -- {r.name}**")
        out += [f"- {n}" for n in r.notes]
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rfenv.validate",
        description="Run EVALUATION.md §6's four validation gates. Train split only.",
    )
    ap.add_argument("--gate", default="1,2,3,4",
                    help="comma-separated subset, e.g. --gate 3,4 (default: all)")
    ap.add_argument("--out", default="runs/validation", help="artefact directory")
    ap.add_argument("--seed", type=int, default=0, help="receiver noise seed")
    ap.add_argument("--configs", type=int, default=None,
                    help="limit gates 1-2 to the first N train configs (smoke test)")
    args = ap.parse_args(argv)

    wanted = [g.strip() for g in args.gate.split(",") if g.strip()]
    unknown = [g for g in wanted if g not in GATES]
    if unknown:
        ap.error(f"unknown gate(s) {unknown}; choose from {sorted(GATES)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = list_configs("scan")
    if args.configs is not None:
        configs = configs[: args.configs]

    results: list[GateResult] = []
    if "1" in wanted:
        results.append(gate1(configs, args.seed))
    if "2" in wanted:
        results.append(gate2(configs))
    if "3" in wanted:
        results.append(gate3(args.seed))
    if "4" in wanted:
        results.append(gate4(out_dir, args.seed))

    # The operating point is stamped on every validation run: EVALUATION.md §7
    # requires gamma and Pfa be stated alongside any table, and the ROC population
    # is on the freeze list (D33).
    point = operating_point(
        [TruthGrid.from_scenario(Scenario.replay(c, "scan")) for c in configs]
    )

    for res in results:
        (out_dir / f"gate{res.gate}.json").write_text(
            json.dumps(res.to_json(), indent=2, default=_jsonable) + "\n", encoding="utf-8"
        )

    _render(out_dir, results)

    summary = {
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "split": "train",
        "n_configs": len(configs),
        "operating_point": point,
        "gates": [{"gate": r.gate, "name": r.name, "status": r.status,
                   "gated": r.gated, "passed": r.passed} for r in results],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report_md(results, point, args.seed), encoding="utf-8")

    print()
    for res in results:
        print(_line(res))
    print()
    failed = [r for r in results if r.gated and not r.passed]
    print(f"artefacts: {out_dir}")
    if failed:
        print(f"FAILED: {', '.join('gate ' + r.gate for r in failed)}")
        return 1
    gated = [r for r in results if r.gated]
    print(f"all {len(gated)} gated check(s) passed"
          + ("; gate 1 MEASURED" if any(not r.gated for r in results) else ""))
    return 0


def _render(out_dir: Path, results: list[GateResult]) -> None:
    """Per-band bars for gates 1 and 2. Import is local: matplotlib is render.py's."""
    from rfenv import render

    for res in results:
        if res.gate == "1":
            render.band_profile(
                {"predicted (stare-built)": np.nan_to_num(res.measured["per_band_predicted"]),
                 "recorded (scan)": np.nan_to_num(res.measured["per_band_recorded"])},
                out_dir / "gate1_bands.png",
                title="Gate 1 -- predicted vs recorded non-empty dwell rate, per band",
                ylabel="non-empty dwell rate",
            )
        elif res.gate == "2":
            render.band_profile(
                {"replayed (scan-built)": np.nan_to_num(res.measured["per_band_replayed"]),
                 "recorded (scan)": np.nan_to_num(res.measured["per_band_recorded"])},
                out_dir / "gate2_bands.png",
                title="Gate 2 -- replayed vs recorded non-empty dwell rate, per band",
                ylabel="non-empty dwell rate",
            )


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
