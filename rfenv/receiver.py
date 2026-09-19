"""L2 — the receiver layer.

The instrument. L1 holds the world; this module is what happens when you point a
real detector at one cell of it and ask whether anything is there.

    Y[b, t] = 1  iff  S[b, t] + n >= gamma,   n ~ N(0, sigma)

That single line is the whole detection model, and everything the problem
statement asks for at the receiver level falls out of it. `Y` is the scheduler's
only observable. It differs from truth in both directions -- an empty cell can
declare a hit (Pfa > 0) and an occupied one can stay silent (Pd < 1) -- and that
disagreement *is* Pd and Pfa (D26). A natively binary environment has Pfa = 0 by
construction and the metric carries no information, which is why D4 made the
level continuous in the first place.

Three things this layer is deliberately not:

* **Not the scheduler.** It looks where it is told. Pd and Pfa are properties of
  this layer at the frozen gamma and are identical for every scheduler (D15,
  D21); what a scheduler controls is *which* cells get looked at.
* **Not free to choose its dwell length.** An action is a band; the dwell then
  runs that band's native Turing length, 2 slots for bands 0, 1, 6, 7, 17, 18, 19
  and 1 slot otherwise (D3, D16). Retuning is free -- measured under 1 us, under
  0.002% of a slot (D31) -- so airtime is the only currency in the problem.
* **Not a per-pulse detector.** `Y` is declared once per cell on the combined `S`,
  because a real receiver cannot un-mix a cell (D28). The honest form is per-pulse
  detection with per-emitter noise draws; that is v2, after the gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from rfenv.constants import (
    DWELL_SLOTS,
    GAMMA_DBM,
    N_SLOTS,
    NOISE_SIGMA_DB,
    PD_POPULATION,
    dwell_schedule,
    slots_for_dwell,
)
from rfenv.truth import TruthGrid

# Phi and its inverse, from the standard library. scipy is not a dependency of
# this project and this is the only place that would have needed it. `frompyfunc`
# vectorises the scalar cdf at about 400k values per 55 ms, which is ample: the
# ROC sweep is ~12k cells per gamma.
_NORMAL = NormalDist()
_phi = np.frompyfunc(_NORMAL.cdf, 1, 1)


def phi(x: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF, elementwise."""
    return np.asarray(_phi(np.asarray(x, dtype=np.float64)), dtype=np.float64)


# --------------------------------------------------------------------------- #
# One dwell
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DwellResult:
    """What one action produced. Per-slot arrays, because everything is per slot.

    `Y` is the only field the agent may see unconditionally (and even then only
    through the history-derived observation, D34). The rest is evaluator-side raw
    truth: `Z`, `C`, `pulse_width_us` and `aoa_deg` are all truth-grid slices with
    no gamma gate applied here -- `env.py` is where `pulse_width_us`/`aoa_deg`
    become observable, and only on slots where `Y` is also true (D30): a real
    receiver only measures a pulse's width and bearing on a pulse it actually
    detected, so folding these into the agent's memory ungated would be reading
    truth the receiver never had, the same failure D29 already rules out for `C`.
    """

    band: int
    slot0: int
    level_dbm: np.ndarray      # (n_slots,) float64 -- true S, the noise-free level
    measured_dbm: np.ndarray   # (n_slots,) float64 -- S + n, what the detector read
    Y: np.ndarray              # (n_slots,) bool    -- declared detections
    Z: np.ndarray              # (n_slots,) bool    -- true occupancy of those cells
    C: np.ndarray              # (n_slots,) int32   -- illuminations inside the window
    pulse_width_us: np.ndarray  # (n_slots,) float32 -- D30, truth-side; env.py gates by Y
    aoa_deg: np.ndarray         # (n_slots,) float32 -- D30, truth-side; env.py gates by Y
    intercepts: tuple[tuple[int, int], ...]  # (emitter_index, slot) meeting D28

    @property
    def n_slots(self) -> int:
        return len(self.Y)

    @property
    def pulses(self) -> int:
        """Illuminations the dwell was tuned across -- the interception-ratio
        numerator, per illumination and not per dwell (EVALUATION.md §4)."""
        return int(self.C.sum())

    @property
    def slots(self) -> np.ndarray:
        return np.arange(self.slot0, self.slot0 + self.n_slots)

    @property
    def hit(self) -> bool:
        """Did the dwell declare anything at all? A convenience for baselines."""
        return bool(self.Y.any())


class Receiver:
    """The frozen detector: a threshold, a noise scale, and a random number stream.

    Nothing here is learned or tuned. gamma and sigma come from the freeze list
    and no result may move them (D25); the RNG only realises the noise.
    """

    def __init__(
        self,
        gamma_dbm: float = GAMMA_DBM,
        sigma_db: float = NOISE_SIGMA_DB,
        rng: np.random.Generator | None = None,
    ):
        self.gamma = float(gamma_dbm)
        self.sigma = float(sigma_db)
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ look --

    def dwell(self, grid: TruthGrid, band: int, slot0: int) -> DwellResult:
        """Tune to `band` at `slot0` and look for that band's native dwell length.

        One **independent noise draw per slot**, not one per dwell. That is what
        makes a 100 ms dwell two measurements rather than one long one: a marginal
        emitter sitting at S = gamma yields Y with probability 0.5 per look and
        0.75 over the pair. Collapsing the dwell to a single draw would erase
        exactly the signal D31 exists to preserve.

        A dwell that would run past the end of the episode is clipped, matching
        `constants.dwell_schedule()`, which already truncates its final dwell at
        the 30 s boundary. Clipping rather than forbidding keeps every band legal
        at every slot, so the action space never has to change shape (D35).
        """
        band, slot0 = int(band), int(slot0)
        if not 0 <= band < len(DWELL_SLOTS):
            raise ValueError(f"band {band} outside 0..{len(DWELL_SLOTS) - 1}")
        if not 0 <= slot0 < N_SLOTS:
            raise ValueError(f"slot {slot0} outside 0..{N_SLOTS - 1}")

        n = min(int(DWELL_SLOTS[band]), N_SLOTS - slot0)
        sl = slice(slot0, slot0 + n)

        S = grid.S[band, sl].astype(np.float64)
        measured = S + self.rng.normal(0.0, self.sigma, size=n)
        Y = measured >= self.gamma

        # Interception ratio counts illuminations, not dwells, and is
        # threshold-free -- it is the opportunity metric, gamma is not in it
        # (EVALUATION.md §4, D28). Summing C over the visited cells cannot
        # double-count: C double-counts a pulse across the two *adjacent bands*
        # whose windows both contain it, and the receiver occupies exactly one
        # band per slot, so it can never collect both copies.
        return DwellResult(
            band=band, slot0=slot0,
            level_dbm=S, measured_dbm=measured, Y=Y,
            Z=grid.Z[band, sl].copy(), C=grid.C[band, sl].copy(),
            pulse_width_us=grid.PW[band, sl].copy(), aoa_deg=grid.AOA[band, sl].copy(),
            intercepts=self._intercepts(grid, band, slot0, Y),
        )

    def _intercepts(
        self, grid: TruthGrid, band: int, slot0: int, Y: np.ndarray
    ) -> tuple[tuple[int, int], ...]:
        """Emitters this dwell intercepted, by D28's three clauses.

        An emitter `e` is intercepted at slot `t` iff

            1. the tuned band is one `e` puts pulses into at `t`;
            2. **`e`'s own** received level in that cell clears gamma;
            3. the receiver declared `Y(t) = 1`.

        Clause 2 is what stops a quiet emitter inheriting a loud neighbour's
        detectability, and it is the same rule that fixes `on_e` (D27) -- so
        numerator and denominator of censored intercept time are measured one way
        and `first_e >= on_e` holds by construction rather than by luck.

        Note clause 3 tests `Y`, which was declared on the *combined* `S`. Since
        `S` is the max over contributors, `own >= gamma` implies `S >= gamma`, so
        clauses 2 and 3 never contradict each other; the effect is that a
        qualifying weak emitter sharing a cell with a louder one is near-certain
        to be credited. Accepted as the cost of a cell-level model (D28).
        """
        found: list[tuple[int, int]] = []
        for i, declared in enumerate(Y):
            if not declared:
                continue
            slot = slot0 + i
            owners, own_peak = grid.contributors_at(band, slot)
            for e in owners[own_peak >= self.gamma]:
                found.append((int(e), slot))
        return tuple(found)

    # ------------------------------------------------------- characterisation --

    def detection_probability(self, level_dbm: np.ndarray | float) -> np.ndarray:
        """`P(Y = 1)` for a cell at this true level: `Phi((S - gamma) / sigma)`.

        Analytic rather than sampled. The noise is the only stochastic element in
        the layer, and its distribution is known exactly, so averaging this over a
        population gives the same answer as a Monte-Carlo draw with none of the
        variance -- a ROC that is identical from run to run.
        """
        return phi((np.asarray(level_dbm, dtype=np.float64) - self.gamma) / self.sigma)


# --------------------------------------------------------------------------- #
# Receiver characterisation: the ROC (EVALUATION.md §3)
# --------------------------------------------------------------------------- #

def reference_sweep_cells(grid: TruthGrid) -> tuple[np.ndarray, np.ndarray]:
    """The cells Turing's own dwell schedule looks at, as (levels, occupied).

    This is the frozen `PD_POPULATION` (D33). Pd has to be averaged over *some*
    population and the figure depends on it entirely -- 0.819, 0.837 and 0.851 at
    the same gamma over three different ones. "Cells the receiver actually looked
    at" cannot be it, because then a camper parked on loud cells would report a
    better Pd than a sweeper and Pd would stop being a receiver property (D21).
    Turing's schedule never varies, so this population is per-look *and*
    scheduler-independent.

    A cell visited twice by the sweep is counted twice: these are per-look
    probabilities, so the weighting by how often the schedule visits a cell is
    correct rather than an artefact.
    """
    levels, occupied = [], []
    for start, end, band in dwell_schedule():
        s0, s1 = slots_for_dwell(start, end)
        b = int(band)
        levels.append(grid.S[b, s0:s1].astype(np.float64))
        occupied.append(grid.Z[b, s0:s1])
    return np.concatenate(levels), np.concatenate(occupied)


def all_occupied_cells(grid: TruthGrid) -> tuple[np.ndarray, np.ndarray]:
    """Every cell in the grid, as (levels, occupied). Not the frozen population.

    Kept because the ROC's *shape* is worth being able to draw over the whole grid
    for the write-up, and because gate 4's extremes are easier to read this way.
    Never used for the quoted operating point -- D33 settled that on
    `reference_sweep_cells`, and mixing the two is how the withdrawn 0.822 happened.
    """
    return grid.S.ravel().astype(np.float64), grid.Z.ravel()


POPULATIONS = {
    "reference_sweep": reference_sweep_cells,
    "all_cells": all_occupied_cells,
}


def roc(
    grids,
    gammas: np.ndarray,
    sigma_db: float = NOISE_SIGMA_DB,
    population: str = PD_POPULATION,
) -> dict:
    """Sweep gamma over one or more grids; return Pd, Pfa and the cell counts.

    The sweep *is* the receiver deliverable (D15) -- Pd, Pfa and sensitivity are
    three of the problem statement's figures of merit and only exist as a curve.
    A single operating point is a reading off it, never the whole answer.

    Pd is conditioned on threshold-free `Z`, not on a second copy of `S`
    thresholded at the same gamma (D26). The latter is degenerate: it forces
    Pd >= 0.5 for every gamma, so the curve cannot sweep and carries no
    information.
    """
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {sorted(POPULATIONS)}")
    select = POPULATIONS[population]

    levels, occupied = zip(*(select(g) for g in np.atleast_1d(grids)))
    levels, occupied = np.concatenate(levels), np.concatenate(occupied)
    on, off = levels[occupied], levels[~occupied]

    gammas = np.atleast_1d(np.asarray(gammas, dtype=np.float64))
    pd = np.array([phi((on - g) / sigma_db).mean() if on.size else np.nan for g in gammas])
    pfa = np.array([phi((off - g) / sigma_db).mean() if off.size else np.nan for g in gammas])
    return {
        "gamma_dbm": gammas,
        "pd": pd,
        "pfa": pfa,
        "population": population,
        "n_occupied": int(on.size),
        "n_empty": int(off.size),
    }


def operating_point(
    grids,
    gamma_dbm: float = GAMMA_DBM,
    sigma_db: float = NOISE_SIGMA_DB,
    pd_target: float = 0.9,
    population: str = PD_POPULATION,
) -> dict:
    """Pd, Pfa and sensitivity at one gamma. Never quote one without the others.

    Pfa and sensitivity are analytic in gamma and sigma alone and do not depend on
    the data at all: with empty cells sitting exactly at the noise floor,
    `Pfa = 1 - Phi((gamma - N0) / sigma)`, which at the frozen gamma = N0 + 3 sigma
    is the Gaussian 3-sigma tail. Sensitivity -- the level at which Pd reaches
    `pd_target` -- inverts `Phi((S - gamma) / sigma) = pd_target` to
    `gamma + Phi^-1(pd_target) * sigma`. Only Pd depends on the population.
    """
    out = roc(grids, np.array([gamma_dbm]), sigma_db, population)
    return {
        "gamma_dbm": float(gamma_dbm),
        "sigma_db": float(sigma_db),
        "pd": float(out["pd"][0]),
        "pfa": float(out["pfa"][0]),
        "sensitivity_dbm": float(gamma_dbm + _NORMAL.inv_cdf(pd_target) * sigma_db),
        "pd_target": float(pd_target),
        "population": population,
        "n_occupied": out["n_occupied"],
        "n_empty": out["n_empty"],
    }
