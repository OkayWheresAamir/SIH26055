"""L1 — the truth layer.

The world the scheduler is looking into: a 36 x 600 grid of what is actually out
there, for every band and every 50 ms slot, whether or not anyone looked.

Three arrays, and the distinction between them is the whole detection model (D26):

    Z[b, t]   bool     is any emitter transmitting into this cell?  Threshold-free.
                       This is the problem statement's "transmission or a
                       non-transmission" status.
    S[b, t]   float32  the continuous received level in dB: the peak pulse
                       amplitude where Z is true, the noise floor where it is not.
    C[b, t]   int32    how many pulses landed in the cell -- the numerator of
                       interception ratio, which is counted per illumination and
                       not per dwell (EVALUATION.md §4).

Two more (D30), same shape, describing **whichever single pulse set `S`** in that
cell -- not an average over every contributor, a matched reading of one real pulse:

    PW[b, t]   float32  that pulse's width, microseconds. 0.0 where Z is false.
    AOA[b, t]  float32  that pulse's angle of arrival, degrees. 0.0 where Z is false.

What is *not* here: noise, thresholds applied to make a decision, and anything
stochastic. Truth is the world; the receiver is the instrument. `receiver.py`
draws the noise and declares the hit. This module has no RNG and no I/O, so a
Scenario always yields exactly the same grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rfenv.constants import GAMMA_DBM, N_BANDS, N_SLOTS, NOISE_FLOOR_DBM
from rfenv.scenario import EmitterContribution, Scenario


@dataclass
class TruthGrid:
    """The 36 x 600 world for one scenario.

    Emitters superpose by taking the maximum level per cell -- the same rule used
    to combine pulses within one emitter, and the right one because a cell's level
    is the peak the receiver would measure there. Pulse counts add.
    """

    scenario: Scenario
    S: np.ndarray            # (36, 600) float32, dB
    C: np.ndarray            # (36, 600) int32
    Z: np.ndarray            # (36, 600) bool
    PW: np.ndarray           # (36, 600) float32, us -- D30, the S-setting pulse's width
    AOA: np.ndarray          # (36, 600) float32, deg -- D30, the S-setting pulse's bearing
    _cell_id: np.ndarray     # (n,) int64, sorted: band * N_SLOTS + slot
    _owner: np.ndarray       # (n,) int32, index into `contributions`
    _owner_peak: np.ndarray  # (n,) float32, that emitter's own level in that cell

    # ------------------------------------------------------------------ build --

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "TruthGrid":
        S = np.full((N_BANDS, N_SLOTS), NOISE_FLOOR_DBM, dtype=np.float32)
        C = np.zeros((N_BANDS, N_SLOTS), dtype=np.int32)
        Z = np.zeros((N_BANDS, N_SLOTS), dtype=bool)
        PW = np.zeros((N_BANDS, N_SLOTS), dtype=np.float32)
        AOA = np.zeros((N_BANDS, N_SLOTS), dtype=np.float32)

        ids, owners, peaks = [], [], []
        for i, c in enumerate(scenario.contributions):
            if not len(c):
                continue
            b, t = c.cells[:, 0].astype(np.int64), c.cells[:, 1].astype(np.int64)
            # D30: PW/AOA must follow whichever contribution's peak_dbm actually
            # wins the max at each cell -- so read the *pre-update* running max
            # here, before `np.maximum.at` folds this contribution in, and only
            # overwrite PW/AOA where this contribution's own peak beats it. Two
            # contributions landing in the same cell across different loop
            # iterations are handled correctly because `S` is monotonically
            # non-decreasing across the loop: whichever one is truly loudest is
            # always the last to satisfy `better` for that cell.
            beats_current_max = c.peak_dbm > S[b, t]
            np.maximum.at(S, (b, t), c.peak_dbm)
            np.add.at(C, (b, t), c.n_pulses)
            Z[b, t] = True
            wb, wt = b[beats_current_max], t[beats_current_max]
            PW[wb, wt] = c.pulse_width_us[beats_current_max]
            AOA[wb, wt] = c.aoa_deg[beats_current_max]
            ids.append(b * N_SLOTS + t)
            owners.append(np.full(len(c), i, dtype=np.int32))
            peaks.append(c.peak_dbm)

        if ids:
            cell_id = np.concatenate(ids)
            owner = np.concatenate(owners)
            owner_peak = np.concatenate(peaks)
            order = np.argsort(cell_id, kind="stable")
            cell_id, owner, owner_peak = cell_id[order], owner[order], owner_peak[order]
        else:
            cell_id = np.zeros(0, np.int64)
            owner = np.zeros(0, np.int32)
            owner_peak = np.zeros(0, np.float32)

        # An empty cell sits at the floor exactly; nothing may have pushed it below.
        S[~Z] = NOISE_FLOOR_DBM
        return cls(scenario=scenario, S=S, C=C, Z=Z, PW=PW, AOA=AOA,
                   _cell_id=cell_id, _owner=owner, _owner_peak=owner_peak)

    @classmethod
    def stitch(cls, grids: list["TruthGrid"], *, name: str) -> "TruthGrid":
        """Lay grids end to end into one longer world (D76).

        This is how an episode runs longer than 30 s. A recording *is* 30 s, so
        there is no such thing as a one-hour scenario to load; a long mission is
        built by concatenating independent draws along the time axis.

        **Emitter identity is per segment, deliberately.** Each segment is its own
        pool draw, so the same physical emitter may appear in two of them with a
        different realisation, position and beam phase. Merging those into one
        identity would mean an emitter found in segment 0 could never be found
        again for the rest of the mission, which would silently destroy coverage.
        Two segments are two emitters and two opportunities; `config_id` is
        suffixed so `uid` stays unique and the emitter table stays readable.

        The honest limitation, stated rather than hidden: **there is no temporal
        continuity across a seam.** At each boundary the whole world is replaced
        at once. Nothing in the environment pretends otherwise, and a dwell that
        straddles a seam simply reads two cells from two different worlds -- which
        needs no special case, but is not physics.

        `stitch([g])` reproduces `g` exactly, index included, which is what makes
        the single-segment path testable against the multi-segment one.
        """
        if not grids:
            raise ValueError("stitch needs at least one grid")

        widths = [int(g.S.shape[1]) for g in grids]
        offsets = np.concatenate([[0], np.cumsum(widths[:-1])]).astype(np.int64)
        total = int(sum(widths))

        # `contributions` is rebuilt first, because the cell index below must
        # point into *this* list, not into any segment's own.
        contributions: list[EmitterContribution] = []
        for k, (grid, offset) in enumerate(zip(grids, offsets)):
            for c in grid.contributions:
                # int32, not the declared int16: a slot index past 32767 is only
                # 54 segments away (27 simulated minutes), and int16 would wrap
                # there silently rather than fail.
                cells = c.cells.astype(np.int32, copy=True)
                if len(cells):
                    cells[:, 1] += int(offset)
                contributions.append(
                    replace(c, cells=cells, config_id=f"{c.config_id}@s{k}")
                )

        ids, owners, peaks = [], [], []
        for i, c in enumerate(contributions):
            # Empty contributions keep their slot in the list -- `_owner` indexes
            # into it -- but contribute no cells, exactly as `from_scenario` does.
            if not len(c):
                continue
            b = c.cells[:, 0].astype(np.int64)
            t = c.cells[:, 1].astype(np.int64)
            ids.append(b * total + t)      # the stride is the *stitched* length
            owners.append(np.full(len(c), i, dtype=np.int32))
            peaks.append(c.peak_dbm)

        if ids:
            cell_id = np.concatenate(ids)
            owner = np.concatenate(owners)
            owner_peak = np.concatenate(peaks)
            order = np.argsort(cell_id, kind="stable")
            cell_id, owner, owner_peak = cell_id[order], owner[order], owner_peak[order]
        else:
            cell_id = np.zeros(0, np.int64)
            owner = np.zeros(0, np.int32)
            owner_peak = np.zeros(0, np.float32)

        return cls(
            scenario=Scenario(name=name, contributions=contributions),
            S=np.concatenate([g.S for g in grids], axis=1),
            C=np.concatenate([g.C for g in grids], axis=1),
            Z=np.concatenate([g.Z for g in grids], axis=1),
            PW=np.concatenate([g.PW for g in grids], axis=1),
            AOA=np.concatenate([g.AOA for g in grids], axis=1),
            _cell_id=cell_id, _owner=owner, _owner_peak=owner_peak,
        )

    # ------------------------------------------------------------- properties --

    @property
    def n_slots(self) -> int:
        """This grid's length in slots -- `N_SLOTS` for a recording, longer if stitched.

        A property rather than a field because the field list is positional in
        `from_scenario`'s own `cls(...)` call and in every test that builds a grid
        by hand; adding a field would break all of them for no gain.
        """
        return int(self.S.shape[1])

    @property
    def contributions(self) -> list[EmitterContribution]:
        return self.scenario.contributions

    @property
    def n_emitters(self) -> int:
        return len(self.contributions)

    @property
    def total_pulses(self) -> int:
        """Total illuminations in the scenario -- the interception-ratio denominator.

        Summed from each emitter's true pulse count, not from `C`, because a pulse
        falls inside two adjacent band windows and is counted in both cells (D3).
        """
        return int(sum(c.total_pulses for c in self.contributions))

    # ---------------------------------------------------------------- queries --

    def occupancy(self, gamma: float = GAMMA_DBM) -> np.ndarray:
        """`S >= gamma`, the noise-free thresholded view of the world.

        For reporting and for the waterfall render. The receiver's *declaration*
        is not this -- it adds a noise draw first, which is what makes Pd < 1 and
        Pfa > 0 (`receiver.py`).
        """
        return self.S >= gamma

    def contributors_at(self, band: int, slot: int) -> tuple[np.ndarray, np.ndarray]:
        """Emitters contributing to one cell, and **each one's own level** there.

        The own-level array is the point. `S[b, t]` is the max over contributors, so
        it cannot answer "does *this* emitter clear gamma here?" -- and that is
        D28's clause 2, the one that stops a quiet emitter inheriting a loud
        neighbour's detectability. The data was always stored; this exposes it.

        Evaluator-side only. Neither array ever reaches the agent: the scheduler
        runs cold, with no emitter knowledge beyond its own scan history (D19, D20).
        """
        # The stride is this grid's own length, not the constant: a stitched grid
        # (`stitch`) is longer than 600 and indexes with its own width.
        key = int(band) * self.n_slots + int(slot)
        lo, hi = np.searchsorted(self._cell_id, [key, key + 1])
        return self._owner[lo:hi], self._owner_peak[lo:hi]

    def emitters_at(self, band: int, slot: int) -> np.ndarray:
        """Indices of the emitters contributing to one cell."""
        return self.contributors_at(band, slot)[0]

    def detectable_interval(
        self, emitter_index: int, gamma: float = GAMMA_DBM
    ) -> tuple[int, int] | None:
        """First and last slot at which this emitter is detectable, or None.

        Deliberately *not* called an activity window. This is the interval over
        which the emitter's own received level clears gamma -- activity that is
        detectable under this receiver, not the emitter's physical transmission
        window. An emitter can be transmitting the whole episode while its signal
        sits below gamma, and this interval will not show it.

        Judged on the emitter's own level, not on the combined `S`, so a quiet
        emitter sharing a band with a loud one does not inherit the loud one's
        detectability.

        `on_e` for censored intercept time (EVALUATION.md §4) is the first element.
        """
        c = self.contributions[emitter_index]
        if not len(c):
            return None
        above = c.peak_dbm >= gamma
        if not above.any():
            return None
        slots = c.cells[above, 1]
        return int(slots.min()), int(slots.max())

    def detectable_emitters(self, gamma: float = GAMMA_DBM) -> np.ndarray:
        """`E` -- the emitters a scheduler could possibly find (EVALUATION.md §1).

        Coverage is measured against this, not against every transmitter in the
        metadata: 19.0% of train transmitters are never detected in either
        recording (some transmit above 18 GHz), and scoring a scheduler for
        missing those would be meaningless.
        """
        return np.array(
            [i for i in range(self.n_emitters)
             if self.detectable_interval(i, gamma) is not None],
            dtype=np.int32,
        )

    def summary(self, gamma: float = GAMMA_DBM) -> dict:
        occ = self.occupancy(gamma)
        return {
            "scenario": self.scenario.name,
            "n_emitters": self.n_emitters,
            "n_detectable": int(len(self.detectable_emitters(gamma))),
            "total_pulses": self.total_pulses,
            "cells_occupied_Z": int(self.Z.sum()),
            "cells_above_gamma": int(occ.sum()),
            "occupancy_fraction": float(occ.mean()),
            "gamma_dbm": gamma,
        }
