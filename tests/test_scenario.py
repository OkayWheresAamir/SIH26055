"""L0 checks. Every expected number here was measured against data/turing this
session; a failure means either the code or that measurement is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from rfenv import constants as K
from rfenv.scenario import (
    EmitterPool,
    HeldOutDataError,
    Scenario,
    list_configs,
    load_contributions,
    load_receiver,
    recording_path,
)

TRAIN_CONFIGS = list_configs("scan", "train")


def test_train_split_is_47_pairs():
    assert len(TRAIN_CONFIGS) == 47
    assert list_configs("stare", "train") == TRAIN_CONFIGS


def test_receiver_geometry_matches_frozen_constants():
    """36 centres 250->17750 on 500 MHz spacing; seven 100 ms dwells and
    twenty-nine 50 ms, summing to 2.15 s per sweep. Identical in all 47 scan files."""
    for config_id in TRAIN_CONFIGS:
        rx = load_receiver(recording_path(config_id, "scan"))
        assert np.array_equal(rx["dwell_centres_mhz"], K.BAND_CENTRES_MHZ)
        assert np.allclose(rx["dwell_times_s"], K.DWELL_TIMES_S)
    assert K.N_BANDS == 36
    assert K.BAND_CENTRES_MHZ[0] == 250.0 and K.BAND_CENTRES_MHZ[-1] == 17750.0
    assert (K.DWELL_TIMES_S == 0.10).sum() == 7
    assert (K.DWELL_TIMES_S == 0.05).sum() == 29
    assert K.SWEEP_S == pytest.approx(2.15)


def test_stare_files_carry_no_dwell_schedule():
    """A staring receiver has no sweep, so these datasets exist but are empty."""
    rx = load_receiver(recording_path("config_2", "stare"))
    assert rx["scan_mode"] == "Stare"
    assert rx["dwell_centres_mhz"].size == 0
    assert rx["dwell_times_s"].size == 0


def test_dwell_schedule_is_502_dwells_over_30s():
    """13.953 sweeps fit in 30 s. The 502nd dwell ends at exactly 30.000 s, so no
    dwell is truncated: 13 full sweeps (468 dwells) plus 34 of the 14th."""
    sched = K.dwell_schedule()
    assert len(sched) == 502
    complete = np.isclose(sched[:, 1] - sched[:, 0], K.DWELL_TIMES_S[sched[:, 2].astype(int)])
    assert complete.all()
    assert sched[:, 1].max() == pytest.approx(K.EPISODE_S)
    assert sched[0, 2] == 0 and sched[35, 2] == 35  # cycles bands in order
    # 502 dwells x 47 train configs = the 23,594 that the 35.70% rate is measured over
    assert len(sched) * 47 == 23594


def test_band_membership_multiplicity():
    """Centres are 500 MHz apart and the half-width is 500, so a pulse falls in two
    adjacent bands generically and one at the spectrum edges -- never three.
    Measured on config_921: 2 bands for 95.2% of pulses, 1 for 4.8%."""
    import h5py

    with h5py.File(recording_path("config_921", "scan"), "r") as fh:
        freq = fh["data"][:, 1].astype(float)
    mult = np.sum([m for m in K.bands_covering(freq)], axis=0)
    assert set(np.unique(mult)) <= {1, 2}
    assert np.mean(mult == 2) == pytest.approx(0.952, abs=0.005)
    assert np.mean(mult == 1) == pytest.approx(0.048, abs=0.005)


def test_labels_index_the_transmitter_metadata():
    """labels[:, 0] indexes metadata/transmitters/transmitters_N directly.

    Checked by frequency: every pulse of an emitter lands inside that
    transmitter's own nominal frequency span. Not "near a nominal value" -- the
    RandomRange mode transmits uniformly *between* its two listed frequencies
    (config_2 label 17, nominal [10000, 12000], observed median 10948), so only
    the span holds across all freq_modes."""
    import h5py

    with h5py.File(recording_path("config_2", "stare"), "r") as fh:
        data, labels = fh["data"][:], fh["labels"][:, 0]
    for c in load_contributions("config_2", "stare"):
        observed = data[labels == c.label, 1].astype(float)
        nominal = c.meta["freqs_mhz"]
        assert (observed >= nominal.min() - 50.0).all()
        assert (observed <= nominal.max() + 50.0).all()


def test_contribution_cells_are_in_range():
    for c in load_contributions("config_921", "stare"):
        assert c.cells.dtype == np.int16
        assert (c.bands >= 0).all() and (c.bands < K.N_BANDS).all()
        assert (c.slots >= 0).all() and (c.slots < K.N_SLOTS).all()
        assert len(c.peak_dbm) == len(c.cells) == len(c.n_pulses)
        # a pulse lands in 1 or 2 bands, so the band-multiplied count brackets it
        assert c.total_pulses <= c.n_pulses.sum() <= 2 * c.total_pulses


def test_cache_roundtrip_is_exact():
    fresh = load_contributions("config_59", "scan", rebuild=True)
    cached = load_contributions("config_59", "scan")
    assert len(fresh) == len(cached)
    for a, b in zip(fresh, cached):
        assert a.label == b.label and a.total_pulses == b.total_pulses
        assert np.array_equal(a.cells, b.cells)
        assert np.array_equal(a.peak_dbm, b.peak_dbm)
        assert np.array_equal(a.n_pulses, b.n_pulses)
        assert np.allclose(a.meta["freqs_mhz"], b.meta["freqs_mhz"])


def test_heldout_split_is_refused_without_an_explicit_flag():
    """D8: the 45 test pairs are touched once, at the end. Mechanically enforced."""
    heldout = list_configs("scan", "test")
    assert len(heldout) == 45
    with pytest.raises(HeldOutDataError):
        load_contributions(heldout[0], "scan", split="test")
    with pytest.raises(HeldOutDataError):
        Scenario.replay(heldout[0], "stare", split="test")


def test_pool_size():
    """1,739 emitters appear in scan and 1,704 in stare across the 47 train
    configs, giving 3,443 contributions over 1,913 distinct emitters."""
    pool = EmitterPool.from_train()
    assert len(pool) == 3443
    assert pool.n_distinct_emitters == 1913
    assert sum(c.source == "scan" for c in pool.contributions) == 1739
    assert sum(c.source == "stare" for c in pool.contributions) == 1704
    # Difficulty spread: emitters *detectable* per config, which is what a
    # scheduler faces. The metadata lists 2-99 transmitters per config; 1-82 of
    # them are detectable in either run, and those sum to the 1,913 distinct
    # emitters. 19.04% of the 2,363 train transmitters are never detectable at all.
    assert pool.emitter_counts.sum() == 1913
    assert pool.emitter_counts.min() == 1 and pool.emitter_counts.max() == 82


def test_sampled_scenarios_differ_and_are_reproducible():
    pool = EmitterPool.from_train()
    a = Scenario.sample(pool, np.random.default_rng(0))
    b = Scenario.sample(pool, np.random.default_rng(0))
    c = Scenario.sample(pool, np.random.default_rng(1))
    assert [x.uid for x in a.contributions] == [x.uid for x in b.contributions]
    assert [x.uid for x in a.contributions] != [x.uid for x in c.contributions]
    assert len(Scenario.sample(pool, np.random.default_rng(2), n_emitters=12)) == 12


def test_sampled_scenarios_never_repeat_a_physical_emitter():
    """One emitter, at most one contribution per scenario (D32).

    1,530 of the 1,913 pool emitters have both a scan and a stare realisation, so
    a uniform draw over contributions returned the same physical emitter twice in
    23.9% of scenarios (59.0% at n=82) -- same position, same beam phase, disjoint
    activity, and double-counted in `E`. Drawing over emitters instead of over
    contributions is what makes this impossible rather than merely unlikely.
    """
    pool = EmitterPool.from_train()
    both = sum(1 for v in pool.by_emitter.values() if len(v) > 1)
    assert both == 1530

    rng = np.random.default_rng(0)
    for _ in range(200):
        keys = [(c.config_id, c.label) for c in Scenario.sample(pool, rng).contributions]
        assert len(keys) == len(set(keys))

    # and at the top of the difficulty range, where it used to happen 59% of the time
    for _ in range(50):
        keys = [(c.config_id, c.label)
                for c in Scenario.sample(pool, rng, n_emitters=82).contributions]
        assert len(keys) == len(set(keys)) == 82


def test_sample_can_draw_either_realisation_of_an_emitter():
    """Both runs still feed the pool -- the fix picks one realisation, it does not
    discard stare (D17's conclusion, D24's mechanism, D32's correction)."""
    pool = EmitterPool.from_train()
    rng = np.random.default_rng(3)
    sources = set()
    for _ in range(30):
        sources.update(c.source for c in Scenario.sample(pool, rng, n_emitters=60).contributions)
    assert sources == {"scan", "stare"}


def test_replay_carries_exactly_one_recording():
    """A contribution is always one self-consistent realisation: scan and stare are
    independent simulation runs, so they are never stitched together (D24)."""
    s = Scenario.replay("config_2", "stare")
    assert {c.source for c in s.contributions} == {"stare"}
    assert {c.config_id for c in s.contributions} == {"config_2"}
