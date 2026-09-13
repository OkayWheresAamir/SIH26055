"""The example threat library and the per-band priority vector `p` it produces (D70).

`EMITTER_CLASS` is fixed before any scheduler is scored against it (the brief's own rule --
see `rfenv/threat.py`'s module docstring), so these tests pin the invariants a fixed library
must hold, and check `compute_priority` against the PDF brief's own worked properties rather
than against a hand-picked example.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from rfenv import threat as T
from rfenv.constants import N_BANDS
from rfenv.scenario import Scenario, list_configs
from rfenv.truth import TruthGrid


def test_every_function_string_in_the_data_is_classified():
    """The library is fixed *before* scoring, per the brief -- so if the dataset
    ever carries a `function` value this doesn't know about, that must fail
    loudly here, not silently default a new emitter to some class."""
    seen: set[str] = set()
    for cfg in list_configs("scan", "train"):
        sc = Scenario.replay(cfg, "stare")
        for c in sc.contributions:
            fn = c.meta.get("function")
            if fn is not None:
                seen.add(fn)
    assert seen, "no function strings found -- something else is broken"
    unclassified = seen - set(T.EMITTER_CLASS)
    assert not unclassified, f"unclassified function names: {sorted(unclassified)}"


def test_the_library_has_68_entries():
    """Pinned as a literal -- the brief's own count, confirmed against
    `metadata/feature_names` this session. A change here is a change to the
    dataset's emitter roster, not a typo to silently absorb."""
    assert T.N_CLASSIFIED == 68


def test_classify_raises_on_an_unknown_name():
    with pytest.raises(ValueError, match="threat library"):
        T.classify("Not A Real Radar")


def test_the_high_share_is_close_to_the_briefs_own_measurement():
    """The brief's own §0 gate measured HIGH at 653/1704 = 38.3% (PDF) /
    656/38.5% (the `.md` draft). This is the one external check available for
    a library that is otherwise our own judgement call (`rfenv/threat.py`'s
    module docstring) -- verified this session at 656/1704 = 38.5%, within 2
    emitters of the PDF's own count. A wide band, not an exact match: the
    two documents don't even agree with each other to the emitter.
    """
    counts = {T.ThreatClass.HIGH: 0, T.ThreatClass.MEDIUM: 0, T.ThreatClass.LOW: 0}
    total = 0
    for cfg in list_configs("scan", "train"):
        sc = Scenario.replay(cfg, "stare")
        grid = TruthGrid.from_scenario(sc)
        detectable = set(grid.detectable_emitters().tolist())
        for i, c in enumerate(sc.contributions):
            if i not in detectable:
                continue
            counts[T.classify(c.meta.get("function"))] += 1
            total += 1
    assert total == 1704
    high_share = counts[T.ThreatClass.HIGH] / total
    assert 0.36 <= high_share <= 0.40, f"HIGH share {high_share:.3f} outside the brief's own range"


def test_compute_priority_shape_and_bounds():
    sc = Scenario.replay("config_2", "stare")
    grid = TruthGrid.from_scenario(sc)
    p = T.compute_priority(sc, grid)
    assert p.shape == (N_BANDS,)
    assert p.dtype == np.float32
    assert np.all(p >= 0.0)
    assert np.all(np.isfinite(p))


def test_uniform_priority_means_the_mean_over_occupied_bands_is_one():
    """"rescaled so 1.0 = ordinary" (PDF p.3) -- checked directly: the mean
    priority over bands that actually hold traffic must be 1.0, so a
    scheduler with no way to tell bands apart sees an uninformative vector
    centred on "ordinary", not one biased high or low by construction."""
    sc = Scenario.replay("config_2", "stare")
    grid = TruthGrid.from_scenario(sc)
    p = T.compute_priority(sc, grid)
    band_total = np.zeros(N_BANDS)
    for c in sc.contributions:
        if len(c):
            np.add.at(band_total, c.cells[:, 0].astype(np.int64), c.n_pulses)
    occupied = band_total > 0
    assert occupied.any()
    assert p[occupied].mean() == pytest.approx(1.0, abs=1e-5)


def test_empty_bands_read_as_ordinary_not_zero():
    """The module docstring's own stated default for PDF §4's open item: an
    unoccupied band carries no threat evidence, so it reads as 1.0
    ("ordinary"), never 0.0 ("definitely safe") -- the data doesn't support
    that stronger claim."""
    sc = Scenario.replay("config_2", "stare")
    grid = TruthGrid.from_scenario(sc)
    p = T.compute_priority(sc, grid)
    band_total = np.zeros(N_BANDS)
    for c in sc.contributions:
        if len(c):
            np.add.at(band_total, c.cells[:, 0].astype(np.int64), c.n_pulses)
    empty = band_total == 0
    if empty.any():
        assert np.all(p[empty] == 1.0)


def test_a_band_with_only_high_class_traffic_scores_above_a_low_only_band():
    """The direction the whole feature depends on: more HIGH-weighted share
    in a band must score at least as high as a band with none, once both are
    occupied. Built from a synthetic scenario rather than a real config, so
    the direction is checked independent of which real bands happen to be
    HIGH- or LOW-heavy this session."""
    from dataclasses import replace

    sc = Scenario.replay("config_2", "stare")
    # Two synthetic single-cell contributions on two different, otherwise
    # unused-by-each-other bands, one HIGH-classified, one LOW-classified,
    # equal pulse counts -- isolates the class weight as the only variable.
    high_name = next(n for n, c in T.EMITTER_CLASS.items() if c == T.ThreatClass.HIGH)
    low_name = next(n for n, c in T.EMITTER_CLASS.items() if c == T.ThreatClass.LOW)
    band_a, band_b = 20, 21
    high_c = replace(
        sc.contributions[0],
        cells=np.array([[band_a, 0]], dtype=np.int16),
        peak_dbm=np.array([-90.0], dtype=np.float32),
        n_pulses=np.array([10], dtype=np.int32),
        meta={"function": high_name},
    )
    low_c = replace(
        sc.contributions[0],
        cells=np.array([[band_b, 0]], dtype=np.int16),
        peak_dbm=np.array([-90.0], dtype=np.float32),
        n_pulses=np.array([10], dtype=np.int32),
        meta={"function": low_name},
    )
    synthetic = replace(sc, contributions=[high_c, low_c])
    grid = TruthGrid.from_scenario(synthetic)
    p = T.compute_priority(synthetic, grid)
    assert p[band_a] > p[band_b]
