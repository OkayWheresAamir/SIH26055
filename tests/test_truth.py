"""L1 checks, plus the one that matters most: the pipeline reproduces the recording.

Every expected number was measured against data/turing this session.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from rfenv import constants as K
from rfenv.scenario import Scenario, list_configs, recording_path
from rfenv.truth import TruthGrid


def grid_for(config_id: str, source: str) -> TruthGrid:
    return TruthGrid.from_scenario(Scenario.replay(config_id, source))


def test_grid_shape_and_dtypes():
    g = grid_for("config_2", "stare")
    assert g.S.shape == g.C.shape == g.Z.shape == (36, 600)
    assert g.S.dtype == np.float32 and g.C.dtype == np.int32 and g.Z.dtype == bool


def test_empty_cells_sit_exactly_at_the_noise_floor():
    """Z and S must agree: an unoccupied cell is at the floor and nowhere else."""
    g = grid_for("config_921", "stare")
    assert (g.S[~g.Z] == K.NOISE_FLOOR_DBM).all()
    assert (g.C[~g.Z] == 0).all()
    assert (g.C[g.Z] > 0).all()


def test_superposition_is_max_of_level_and_sum_of_counts():
    g = grid_for("config_2", "stare")
    for band, slot in [(9, 100), (17, 300), (5, 42)]:
        owners = g.emitters_at(band, slot)
        if not len(owners):
            continue
        levels, counts = [], []
        for i in owners:
            c = g.contributions[i]
            m = (c.cells[:, 0] == band) & (c.cells[:, 1] == slot)
            levels.append(c.peak_dbm[m].max())
            counts.append(int(c.n_pulses[m].sum()))
        assert g.S[band, slot] == pytest.approx(max(levels))
        assert g.C[band, slot] == sum(counts)


def test_pulse_totals_are_not_the_band_multiplied_counts():
    """A pulse falls inside two adjacent band windows, so C double-counts it.
    total_pulses is the honest interception-ratio denominator."""
    g = grid_for("config_59", "stare")
    with h5py.File(recording_path("config_59", "stare"), "r") as fh:
        n_file = fh["data"].shape[0]
    assert g.total_pulses == n_file
    assert n_file < g.C.sum() <= 2 * n_file


def test_grid_is_deterministic():
    a, b = grid_for("config_59", "scan"), grid_for("config_59", "scan")
    assert np.array_equal(a.S, b.S) and np.array_equal(a.C, b.C)


def test_detectable_interval_uses_the_emitters_own_level():
    """A quiet emitter sharing a band with a loud one must not inherit its
    detectability -- the interval is judged on that emitter's own peak."""
    g = grid_for("config_2", "stare")
    for i in range(g.n_emitters):
        span = g.detectable_interval(i)
        if span is None:
            continue
        on, off = span
        c = g.contributions[i]
        assert 0 <= on <= off < 600
        assert (c.peak_dbm[c.cells[:, 1] < on] < K.GAMMA_DBM).all()
        assert c.peak_dbm[c.cells[:, 1] == on].max() >= K.GAMMA_DBM


def test_detectable_emitters_shrink_as_gamma_rises():
    g = grid_for("config_921", "stare")
    counts = [len(g.detectable_emitters(gamma)) for gamma in (-150, -120, -111, -90, -60)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] >= counts[-1]


def test_emitters_at_matches_the_contributions():
    g = grid_for("config_1366", "scan")
    for band, slot in [(1, 200), (6, 55), (17, 400)]:
        expected = {
            i for i, c in enumerate(g.contributions)
            if ((c.cells[:, 0] == band) & (c.cells[:, 1] == slot)).any()
        }
        assert set(g.emitters_at(band, slot).tolist()) == expected


def test_extremes_behave(): 
    """EVALUATION.md gate 4. Both runs of each config, since a scenario carries one
    recording: config_81 has 2 transmitters, config_921 has 99."""
    small = TruthGrid.from_scenario(
        Scenario(name="both", contributions=[
            *Scenario.replay("config_81", "stare").contributions,
            *Scenario.replay("config_81", "scan").contributions])
    )
    big = TruthGrid.from_scenario(
        Scenario(name="both", contributions=[
            *Scenario.replay("config_921", "stare").contributions,
            *Scenario.replay("config_921", "scan").contributions])
    )
    assert int(small.Z.sum()) == 386
    assert int(big.Z.sum()) == 11753
    assert small.Z.sum() < big.Z.sum()


# --------------------------------------------------------------------------- #
# The anchor: does the pipeline reproduce the recording it was built from?
# --------------------------------------------------------------------------- #

def _nonempty_dwell_rate(grid: TruthGrid, gamma: float | None = None) -> tuple[int, int]:
    """Replay Turing's own sweep through a grid: how many dwells look non-empty?

    `gamma=None` means no threshold at all -- test physical occupancy Z, not the
    level S. Testing `S >= -inf` would be vacuous now that empty cells sit at the
    noise floor rather than at -inf.
    """
    sched = K.dwell_schedule()
    field = grid.Z if gamma is None else (grid.S >= gamma)
    hits = 0
    for start, end, band in sched:
        s0, s1 = K.slots_for_dwell(start, end)
        if field[int(band), s0:s1].any():
            hits += 1
    return hits, len(sched)


def test_pipeline_reproduces_the_recorded_dwell_rate():
    """The self-consistency check that replaces the old gamma calibration (D23).

    Build the grid from the *scan* recording, replay the schedule that produced
    it, and count non-empty dwells with no threshold. It must return what the raw
    file says: 35.70% across the 47 train configs. This is a plumbing test -- band
    assignment, slot clock and dwell schedule all have to be right for it to pass --
    not evidence about detection, which is why gamma is no longer fitted to it.
    """
    pred_hits = pred_total = actual_hits = 0
    for config_id in list_configs("scan", "train"):
        g = grid_for(config_id, "scan")
        h, n = _nonempty_dwell_rate(g)  # no threshold: Z, not S
        pred_hits += h
        pred_total += n
        with h5py.File(recording_path(config_id, "scan"), "r") as fh:
            toa = np.sort(fh["data"][:, 0].astype(np.float64) / 1e6)
        for start, end, _ in K.dwell_schedule():
            lo = np.searchsorted(toa, start, "left")
            hi = np.searchsorted(toa, end, "left")
            actual_hits += hi > lo

    assert pred_total == 23594
    recorded = 100 * actual_hits / pred_total
    replayed = 100 * pred_hits / pred_total
    assert recorded == pytest.approx(35.70, abs=0.05)
    # Residual is slot-boundary quantisation: a 50 ms dwell starting mid-slot spans
    # two 50 ms slots, so the replay sees a little more than the dwell did.
    assert replayed == pytest.approx(recorded, abs=0.5)


def test_stare_grid_cannot_see_below_500_mhz():
    """Band 0 (centre 250 MHz) is 59.12% occupied in the scan recordings and empty
    in every stare-built grid: stare's freq_range_mhz starts at 500 MHz (D10).
    A known, stated limitation of the out-of-sample gate, not a defect to patch.
    """
    stare_band0 = sum(
        int(grid_for(c, "stare").Z[0].sum()) for c in list_configs("scan", "train")
    )
    scan_band0 = sum(
        int(grid_for(c, "scan").Z[0].sum()) for c in list_configs("scan", "train")
    )
    assert stare_band0 == 0
    assert scan_band0 > 0
