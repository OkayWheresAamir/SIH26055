"""L2 checks: the detection channel, the dwell clock, and the ROC.

Every expected number here was measured against data/turing this session, or is
analytic in gamma and sigma alone and says so. A failure means either the code or
that measurement is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from statistics import NormalDist

from rfenv import constants as K
from rfenv.receiver import Receiver, operating_point, roc
from rfenv.scenario import EmitterContribution, Scenario, list_configs
from rfenv.truth import TruthGrid

TRAIN_CONFIGS = list_configs("scan", "train")


def synthetic(band, slots, level_dbm, label=0, pulses=5):
    """One emitter as a bare contribution -- no HDF5, no recording.

    `EmitterContribution` is a plain frozen dataclass and `TruthGrid` does no I/O,
    so a controlled emitter can be written directly. That is what lets these tests
    pin an exact rule instead of hunting the recordings for a cell that happens to
    exercise it, and it is also what gate 3's periodic case will need.
    """
    slots = np.asarray(slots, dtype=np.int16)
    return EmitterContribution(
        config_id="synthetic", source="synthetic", label=label,
        cells=np.stack([np.full(len(slots), band, np.int16), slots], axis=1),
        peak_dbm=np.full(len(slots), level_dbm, np.float32),
        n_pulses=np.full(len(slots), pulses, np.int32),
        total_pulses=pulses * len(slots),
    )


def grid_of(*contribs):
    return TruthGrid.from_scenario(Scenario(name="synthetic", contributions=list(contribs)))


# --------------------------------------------------------------------------- #
# The dwell clock (D3, D16, D31)
# --------------------------------------------------------------------------- #

def test_dwell_length_is_the_bands_native_length():
    """An action is a band; the dwell is the band's, never the agent's. Seven
    bands run 100 ms (2 slots), twenty-nine run 50 ms (1 slot)."""
    g = grid_of(synthetic(10, [0], -100.0))
    rx = Receiver(rng=np.random.default_rng(0))
    wide = [b for b in range(K.N_BANDS) if rx.dwell(g, b, 0).n_slots == 2]
    assert wide == [0, 1, 6, 7, 17, 18, 19]
    assert sum(rx.dwell(g, b, 0).n_slots == 1 for b in range(K.N_BANDS)) == 29


def test_a_wide_dwell_at_the_last_slot_is_clipped():
    """Clipped, not forbidden, so every band stays legal at every slot and the
    action space never changes shape (D35). Matches dwell_schedule(), which
    already truncates its final dwell at the 30 s boundary."""
    g = grid_of(synthetic(10, [0], -100.0))
    rx = Receiver(rng=np.random.default_rng(0))
    assert rx.dwell(g, 0, K.N_SLOTS - 1).n_slots == 1
    assert rx.dwell(g, 0, K.N_SLOTS - 2).n_slots == 2
    with pytest.raises(ValueError):
        rx.dwell(g, 0, K.N_SLOTS)


def test_each_slot_of_a_dwell_gets_its_own_noise_draw():
    """Two slots, two independent measurements -- not one long look (D31).

    This is what gives a marginal emitter two chances at declaring: 0.5 per look
    at S = gamma, 0.75 over the pair. A single draw per dwell would erase it.
    """
    g = grid_of(synthetic(0, range(600), -100.0))
    rx = Receiver(rng=np.random.default_rng(0))
    pairs = np.array([rx.dwell(g, 0, t).measured_dbm for t in range(0, 500, 2)])
    assert (pairs[:, 0] != pairs[:, 1]).all()
    # ...and the pair is not correlated with itself: a shared draw would give 0 spread
    assert np.std(pairs[:, 0] - pairs[:, 1]) == pytest.approx(K.NOISE_SIGMA_DB * np.sqrt(2), rel=0.1)


# --------------------------------------------------------------------------- #
# Detection (D26)
# --------------------------------------------------------------------------- #

def test_pfa_is_the_three_sigma_tail():
    """Empty cells sit exactly at N0, and gamma is N0 + 3 sigma, so the
    false-alarm rate is the Gaussian 3-sigma tail: 1 - Phi(3) = 1.3499e-3.
    Analytic, not fitted -- and it is why gamma has the value it has (D23)."""
    analytic = 1 - NormalDist().cdf(3.0)
    assert analytic == pytest.approx(1.3499e-3, rel=1e-3)

    g = grid_of(synthetic(10, [0], -100.0))  # band 5 is empty everywhere
    rx = Receiver(rng=np.random.default_rng(0))
    draws = np.concatenate([rx.dwell(g, 5, t).Y
                            for _ in range(100) for t in range(K.N_SLOTS)])
    assert len(draws) == 60_000
    assert draws.mean() == pytest.approx(analytic, abs=6e-4)


def test_detection_probability_matches_a_monte_carlo_draw():
    """Pd is computed analytically as the mean of Phi((S-gamma)/sigma) rather than
    sampled, so the ROC is identical run to run. It must agree with sampling."""
    rx = Receiver(rng=np.random.default_rng(0))
    for level in (-120.0, -114.0, -111.0, -108.0, -105.0):
        g = grid_of(synthetic(10, range(600), level))
        sampled = np.concatenate([rx.dwell(g, 10, t).Y
                                  for _ in range(20) for t in range(600)]).mean()
        assert rx.detection_probability(level) == pytest.approx(sampled, abs=0.012)
    # the anchor points of the curve, exact
    assert rx.detection_probability(K.GAMMA_DBM) == pytest.approx(0.5)
    assert rx.detection_probability(K.GAMMA_DBM + 3.0) == pytest.approx(0.841, abs=1e-3)


# --------------------------------------------------------------------------- #
# Crediting an intercept (D28)
# --------------------------------------------------------------------------- #

def test_a_quiet_emitter_does_not_inherit_a_loud_neighbours_detectability():
    """D28 clause 2, the whole reason the rule has three clauses.

    Two emitters in one cell: one at -100 dB (well above gamma), one at -118 dB
    (below it). The receiver declares Y on the *combined* S = max = -100, because
    a real receiver cannot un-mix a cell. Only the loud one may be credited.
    """
    loud, quiet = synthetic(10, [100], -100.0, label=0), synthetic(10, [100], -118.0, label=1)
    g = grid_of(loud, quiet)
    d = Receiver(rng=np.random.default_rng(0)).dwell(g, 10, 100)
    assert d.Y[0]                                   # declared, on the combined level
    assert g.S[10, 100] == pytest.approx(-100.0)    # max, not sum
    assert [e for e, _ in d.intercepts] == [0]      # loud only


def test_intercepts_require_a_declaration():
    """D28 clause 3. Raise gamma above both emitters and nothing is credited even
    though both are still physically transmitting -- Z stays true."""
    g = grid_of(synthetic(10, [100], -100.0))
    d = Receiver(gamma_dbm=-50.0, rng=np.random.default_rng(0)).dwell(g, 10, 100)
    assert d.Z[0] and not d.Y[0]
    assert d.intercepts == ()


def test_pulses_are_counted_without_reference_to_gamma():
    """Interception ratio is the threshold-free opportunity metric (D28), so the
    pulse count in a dwell is the same at any gamma -- only Y moves."""
    g = grid_of(synthetic(10, [100], -100.0, pulses=7))
    counts = {Receiver(gamma_dbm=gam, rng=np.random.default_rng(0)).dwell(g, 10, 100).pulses
              for gam in (-150.0, -111.0, -50.0)}
    assert counts == {7}


# --------------------------------------------------------------------------- #
# The ROC (EVALUATION.md §3, D33)
# --------------------------------------------------------------------------- #

def test_roc_sweeps_because_pd_is_conditioned_on_Z():
    """Pd must fall as gamma rises. Conditioning on (S >= gamma) instead of on
    threshold-free Z forces Pd >= 0.5 for every gamma, so the curve cannot sweep
    and carries no information (D26). This is that check."""
    grids = [TruthGrid.from_scenario(Scenario.replay(c, "scan")) for c in TRAIN_CONFIGS[:10]]
    out = roc(grids, np.arange(-120.0, -99.0, 2.0))
    assert (np.diff(out["pd"]) < 0).all()
    assert (np.diff(out["pfa"]) <= 0).all()
    assert out["pd"][0] > out["pd"][-1] + 0.1   # a real sweep, not a flat line


def test_operating_point_over_the_47_train_configs():
    """The frozen operating point, re-measured this session over all 47 train
    scan replays at gamma = -111 (D33: the reference-sweep population).

    Pfa and sensitivity are analytic in gamma and sigma and depend on no data:
    1 - Phi(3), and gamma + Phi^-1(0.9)*sigma. Only Pd depends on the population,
    which is exactly why the population is on the freeze list.
    """
    grids = [TruthGrid.from_scenario(Scenario.replay(c, "scan")) for c in TRAIN_CONFIGS]
    op = operating_point(grids)
    assert op["population"] == K.PD_POPULATION == "reference_sweep"
    assert op["n_occupied"] == 11710
    assert op["pd"] == pytest.approx(0.851, abs=0.001)
    assert op["pfa"] == pytest.approx(1.3499e-3, rel=1e-3)
    assert op["sensitivity_dbm"] == pytest.approx(-107.16, abs=0.01)


def test_the_population_changes_pd_which_is_why_it_is_frozen():
    """0.851 over the reference sweep against 0.837 over every occupied cell, at
    the same gamma on the same grids. The withdrawn 0.822 came from leaving this
    unstated; naming the population is the fix (D33)."""
    grids = [TruthGrid.from_scenario(Scenario.replay(c, "scan")) for c in TRAIN_CONFIGS]
    sweep = operating_point(grids, population="reference_sweep")["pd"]
    every = operating_point(grids, population="all_cells")["pd"]
    assert sweep == pytest.approx(0.851, abs=0.001)
    assert every == pytest.approx(0.837, abs=0.001)
    assert sweep != every
