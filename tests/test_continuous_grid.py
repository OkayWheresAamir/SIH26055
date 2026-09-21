"""Episodes longer than one recording: `episode_slots` and `TruthGrid.stitch` (D76).

`constants.py` is frozen (D42) and none of this moves it -- `N_SLOTS` is still
600 and every default-length episode is bit-identical, which
`tests/test_backward_compat_hashes.py` pins independently. What is tested here is
the other length: that a stitched world is really the segments laid end to end,
that a seam is not an episode boundary, that the two blocks which would otherwise
leave their declared Box (`clock`, `staleness`) stay inside it, and that censored
intercept time is charged per segment rather than per mission -- which is a
correctness bug at length, not a rescale.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfenv import constants as K
from rfenv.env import _SWEEPS_PER_EPISODE, SWEEP_SLOTS, ScanEnv
from rfenv.scenario import Scenario
from rfenv.split import training_pool
from rfenv.truth import TruthGrid


def _grid(config_id="config_2"):
    return TruthGrid.from_scenario(Scenario.replay(config_id, "stare"))


def _long_env(slots, **kwargs):
    kwargs.setdefault("reward", "reward_balance")
    return ScanEnv(pool=training_pool(), episode_slots=slots, **kwargs)


# --------------------------------------------------------------------------- #
# stitch
# --------------------------------------------------------------------------- #

def test_stitching_one_grid_reproduces_it_exactly():
    """The single-segment path must be the multi-segment path, not a special case."""
    g = _grid()
    s = TruthGrid.stitch([g], name="one")
    for name in ("S", "C", "Z", "PW", "AOA", "_cell_id", "_owner", "_owner_peak"):
        assert np.array_equal(getattr(g, name), getattr(s, name)), name
    assert s.n_slots == g.n_slots == K.N_SLOTS
    assert s.total_pulses == g.total_pulses
    assert s.n_emitters == g.n_emitters


def test_stitching_one_grid_preserves_every_cell_query():
    g, s = _grid(), None
    s = TruthGrid.stitch([_grid()], name="one")
    for band in range(0, K.N_BANDS, 4):
        for slot in range(0, K.N_SLOTS, 29):
            go, gp = g.contributors_at(band, slot)
            so, sp = s.contributors_at(band, slot)
            assert np.array_equal(go, so)
            assert np.array_equal(gp, sp)


def test_a_stitched_grid_is_the_segments_laid_end_to_end():
    grids = [_grid("config_2"), _grid("config_81"), _grid("config_2")]
    s = TruthGrid.stitch(grids, name="three")
    assert s.n_slots == 3 * K.N_SLOTS
    for k, g in enumerate(grids):
        sl = slice(k * K.N_SLOTS, (k + 1) * K.N_SLOTS)
        assert np.array_equal(s.Z[:, sl], g.Z)
        assert np.array_equal(s.S[:, sl], g.S)
        assert np.array_equal(s.C[:, sl], g.C)
    assert s.total_pulses == sum(g.total_pulses for g in grids)
    assert s.n_emitters == sum(g.n_emitters for g in grids)


def test_cell_queries_resolve_into_the_right_segment():
    a, b = _grid("config_2"), _grid("config_81")
    s = TruthGrid.stitch([a, b], name="two")
    for band in range(0, K.N_BANDS, 3):
        for slot in range(0, K.N_SLOTS, 53):
            # Segment 1's emitters are offset by that segment's own base index.
            b_owners, b_peaks = b.contributors_at(band, slot)
            s_owners, s_peaks = s.contributors_at(band, K.N_SLOTS + slot)
            assert np.array_equal(s_owners, b_owners + a.n_emitters)
            assert np.array_equal(s_peaks, b_peaks)


def test_a_segment_ones_emitter_is_detectable_inside_segment_one():
    a, b = _grid("config_2"), _grid("config_81")
    s = TruthGrid.stitch([a, b], name="two")
    for i in s.detectable_emitters():
        lo, hi = s.detectable_interval(int(i))
        segment = 0 if i < a.n_emitters else 1
        assert segment * K.N_SLOTS <= lo <= hi < (segment + 1) * K.N_SLOTS


def test_the_same_scenario_twice_gives_two_emitters_not_one():
    """Identity is per segment on purpose -- see `stitch`'s docstring.

    Merging would mean an emitter found in segment 0 could never be found again
    for the rest of the mission, which would silently destroy coverage.
    """
    g = _grid()
    s = TruthGrid.stitch([g, g], name="twice")
    assert s.n_emitters == 2 * g.n_emitters
    uids = [c.uid for c in s.contributions]
    assert len(set(uids)) == len(uids)
    assert len(s.detectable_emitters()) == 2 * len(g.detectable_emitters())


def test_slot_offsets_are_upcast_past_the_int16_ceiling():
    """`cells` ships as int16; 32767 // 600 = 54 segments = 27 simulated minutes."""
    g = _grid()
    s = TruthGrid.stitch([g] * 56, name="long")
    assert s.n_slots == 56 * K.N_SLOTS > 32767
    last = [c for c in s.contributions if len(c)][-1]
    assert last.cells.dtype == np.int32
    assert last.cells[:, 1].max() > 32767
    # And the index still resolves at the far end.
    assert s.Z[:, 55 * K.N_SLOTS:].sum() == g.Z.sum()


def test_stitch_refuses_an_empty_list():
    with pytest.raises(ValueError):
        TruthGrid.stitch([], name="none")


# --------------------------------------------------------------------------- #
# episode_slots: the default is unchanged
# --------------------------------------------------------------------------- #

def test_the_default_length_is_n_slots_and_is_identical_to_passing_it():
    scenario = Scenario.replay("config_2", "stare")
    a = ScanEnv(scenario=scenario)
    b = ScanEnv(scenario=scenario, episode_slots=K.N_SLOTS)
    assert a.episode_slots == b.episode_slots == K.N_SLOTS
    oa, _ = a.reset(seed=0)
    ob, _ = b.reset(seed=0)
    assert np.array_equal(oa, ob)
    k, done = 0, False
    while not done:
        oa, ra, done, _, _ = a.step((7 * k) % 36)
        ob, rb, _, _, _ = b.step((7 * k) % 36)
        assert np.array_equal(oa, ob) and ra == rb
        k += 1
    assert a.episode_metrics() == b.episode_metrics()


def test_the_staleness_clip_provably_never_binds_at_the_default_length():
    """It is the clip that keeps a long mission inside the Box; at 600 it is a no-op.

    The most stale a band can be at slot 599 is 599/43 = 13.930, and the declared
    ceiling is 600/43 = 13.953.
    """
    assert (K.N_SLOTS - 1) / SWEEP_SLOTS < _SWEEPS_PER_EPISODE
    assert _SWEEPS_PER_EPISODE == K.N_SLOTS / SWEEP_SLOTS


@pytest.mark.parametrize("bad", [0, -1, 900, 601])
def test_an_unusable_episode_length_is_refused(bad):
    with pytest.raises(ValueError):
        ScanEnv(pool=training_pool(), episode_slots=bad)


def test_a_long_episode_needs_a_pool_not_a_fixed_recording():
    """A 30 s recording tiled into an hour would be exactly periodic."""
    with pytest.raises(ValueError, match="pool"):
        ScanEnv(scenario=Scenario.replay("config_2", "stare"),
                episode_slots=2 * K.N_SLOTS)


# --------------------------------------------------------------------------- #
# episode_slots: the long path
# --------------------------------------------------------------------------- #

def test_a_long_episode_runs_to_its_declared_length():
    env = _long_env(3 * K.N_SLOTS)
    env.reset(seed=0)
    assert env.grid.n_slots == 3 * K.N_SLOTS
    steps, done = 0, False
    while not done:
        _, _, done, _, _ = env.step(steps % 36)
        steps += 1
    assert env.t == 3 * K.N_SLOTS
    assert len(env.log) == 3 * K.N_SLOTS
    with pytest.raises(RuntimeError):
        env.step(0)


def test_the_first_segment_is_the_scenario_reset_already_chose():
    """A seeded run's opening 30 s must not depend on how long the mission is."""
    short = ScanEnv(pool=training_pool(), episode_slots=K.N_SLOTS)
    long_ = _long_env(3 * K.N_SLOTS)
    short.reset(seed=11)
    long_.reset(seed=11)
    assert np.array_equal(long_.grid.Z[:, :K.N_SLOTS], short.grid.Z)


@pytest.mark.parametrize("version,kwargs", [
    ("v1", {}),
    ("v3", {"band_priority": True}),
])
def test_every_observation_of_a_long_camping_episode_stays_inside_the_box(version, kwargs):
    """Camping is the worst case for both blocks that scale with episode length.

    Without the `clock` denominator and the `staleness` clip this fails at slot
    601: a band untouched for three segments reads 41.8 sweeps against a declared
    ceiling of 13.95, and the clock reads 3.0 against a declared 1.0.
    """
    env = _long_env(3 * K.N_SLOTS, obs_version=version, **kwargs)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    done = False
    while not done:
        obs, _, done, _, _ = env.step(0)
        assert env.observation_space.contains(obs), f"left the Box at t={env.t}"


def test_a_dwell_straddling_a_seam_is_not_clipped_but_the_mission_end_is():
    """A seam is not a boundary: the clock does not stop at slot 600."""
    env = _long_env(2 * K.N_SLOTS)
    env.reset(seed=0)
    wide = int(np.flatnonzero(np.asarray(K.DWELL_SLOTS) == 2)[0])

    while env.t < K.N_SLOTS - 1:
        env.step(wide if env.t + 2 <= K.N_SLOTS - 1 else 2)
    assert env.t == K.N_SLOTS - 1
    env.step(wide)                      # 599 -> 601, straddling the seam
    assert env.t == K.N_SLOTS + 1

    while env.t < 2 * K.N_SLOTS - 1:    # walk to the very last slot
        env.step(wide if env.t + 2 <= 2 * K.N_SLOTS - 1 else 2)
    env.step(wide)                      # clipped by the end of the world
    assert env.t == 2 * K.N_SLOTS


def test_the_log_window_bounds_memory_and_blocks_artefact_writing(tmp_path):
    from rfenv.metrics.artefacts import write_run

    env = _long_env(2 * K.N_SLOTS, log_window_slots=500)
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.t % 36)
        assert len(env.log) <= 500
    with pytest.raises(ValueError, match="windowed log"):
        write_run(tmp_path, env, scheduler="x", seed=0)


# --------------------------------------------------------------------------- #
# Censoring
# --------------------------------------------------------------------------- #

def test_a_missed_emitter_is_censored_at_its_own_segment_not_the_whole_mission():
    """The latent bug at length: charging a segment-0 miss the full hour.

    Recomputed here from `detectable`/`tracks` rather than trusting the number,
    so the assertion is about the rule and not about one scenario's arithmetic.
    """
    env = _long_env(3 * K.N_SLOTS)
    env.reset(seed=0)
    done = False
    while not done:                      # camp: misses almost everything
        _, _, done, _, _ = env.step(0)

    delays = []
    for e, (on_e, _) in env.detectable.items():
        track = env.tracks.get(e)
        end = ((on_e // K.N_SLOTS) + 1) * K.N_SLOTS
        delays.append((end if track is None else track["first"]) - on_e)
        assert delays[-1] <= K.N_SLOTS

    assert env.episode_metrics()["censored_mean_intercept_time_s"] == pytest.approx(
        float(np.mean(delays)) * K.SLOT_S
    )
    # A whole-mission censor would run to 90 s; a per-segment one cannot pass 30.
    assert env.episode_metrics()["censored_mean_intercept_time_s"] <= K.EPISODE_S


def test_the_intercept_rate_is_per_second_of_this_episode():
    env = _long_env(2 * K.N_SLOTS)
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.t % 36)
    m = env.episode_metrics()
    assert m["avg_intercept_rate_per_s"] == pytest.approx(
        len(env.tracks) / (2 * K.N_SLOTS * K.SLOT_S)
    )
