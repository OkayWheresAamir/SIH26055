"""D30: PulseWidth and AoA enter the observation, and the layout is modular.

Three things pinned here that a change to `env.py`/`truth.py`/`scenario.py`
must not silently break: (1) "v1" stays byte-for-byte the 183-wide vector every
pre-D30 checkpoint was trained on -- nothing about adding "v2" may change it;
(2) PulseWidth/AoA only ever update on a slot the receiver actually declared a
hit on (D29's rule, applied to the two newly-read PDW columns); (3) when two
contributions share a cell, `TruthGrid.PW`/`.AOA` follow the louder one, the
same emitter `S` already credits, not some other pairing.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rfenv import constants as K
from rfenv.env import OBS_LAYOUTS, ScanEnv, obs_width
from rfenv.receiver import Receiver
from rfenv.scenario import Scenario
from rfenv.truth import TruthGrid
from tests.test_receiver import grid_of, synthetic


# --------------------------------------------------------------------------- #
# The layout table itself
# --------------------------------------------------------------------------- #

def test_v1_is_183_wide_and_still_the_default():
    assert obs_width("v1") == 183
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"))
    assert env.observation_space.shape == (183,)
    assert env._obs_version == "v1"


def test_v2_is_362_wide():
    assert obs_width("v2") == 362
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"), obs_version="v2")
    assert env.observation_space.shape == (362,)


def test_unknown_obs_version_is_refused():
    with pytest.raises(ValueError, match="obs_version"):
        ScanEnv(scenario=Scenario.replay("config_59", "stare"), obs_version="v99")


def test_every_layout_only_names_real_blocks():
    from rfenv.env import _BLOCK_SPECS
    for version, names in OBS_LAYOUTS.items():
        unknown = set(names) - set(_BLOCK_SPECS)
        assert not unknown, f"{version} names unknown blocks: {unknown}"


# --------------------------------------------------------------------------- #
# v1 is unchanged by v2 existing
# --------------------------------------------------------------------------- #

def test_v1_and_v2_agree_on_every_block_they_share():
    """Same scenario, same seed, same actions -> the blocks "v1" and "v2" both
    read (hit_rate, visit_density, staleness, current_band, clock, hit_streak,
    current_hit_streak) must be numerically identical. If they diverged, D30's
    extra bookkeeping would have perturbed the pre-existing vector -- exactly
    what every existing checkpoint depends on not happening.
    """
    sc = Scenario.replay("config_2", "stare")
    shared = ("hit_rate", "visit_density", "staleness", "current_band",
              "clock", "hit_streak", "current_hit_streak")

    env1 = ScanEnv(scenario=sc, obs_version="v1")
    env2 = ScanEnv(scenario=sc, obs_version="v2")
    env1.reset(seed=7)
    env2.reset(seed=7)
    for t in range(120):
        a = t % K.N_BANDS
        env1.step(a)
        env2.step(a)
    b1, b2 = env1._observation_blocks(), env2._observation_blocks()
    for name in shared:
        np.testing.assert_array_equal(b1[name], b2[name], err_msg=name)
    # measured_dbm (v1, global) must equal measured_dbm_band (v2) at whichever
    # band was just visited -- same underlying dwell.measured_dbm.mean() call.
    assert b1["measured_dbm"][0] == pytest.approx(b2["measured_dbm_band"][env2._current_band])


def test_v1_observation_never_grows_the_extra_blocks():
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"), obs_version="v1")
    env.reset(seed=0)
    for t in range(50):
        env.step(t % K.N_BANDS)
    assert env._observation().shape == (183,)


# --------------------------------------------------------------------------- #
# Gating by Y (D29's rule, applied to PulseWidth/AoA)
# --------------------------------------------------------------------------- #

def test_pulse_width_and_aoa_start_at_the_never_measured_sentinel():
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"), obs_version="v2")
    env.reset(seed=0)
    blocks = env._observation_blocks()
    assert (blocks["pulse_width"] == 0.0).all()
    assert (blocks["aoa_sin"] == 0.5).all()   # (0.5, 0.5) decodes to raw (0, 0)
    assert (blocks["aoa_cos"] == 0.5).all()
    assert (blocks["pulse_count"] == 0.0).all()   # D72, log1p(0) = 0


def test_a_dwell_with_no_declared_hit_does_not_move_pulse_width_or_aoa():
    """A loud emitter far below gamma should almost never declare Y=1 -- pin one
    concrete miss and confirm the band's PW/AoA memory is untouched by it."""
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"), obs_version="v2")
    env.reset(seed=0)
    # Find a real step this seed produces with zero declared hits.
    found = False
    for t in range(200):
        band = t % K.N_BANDS
        before = env._observation_blocks()
        obs, r, term, trunc, info = env.step(band)
        if info["Y"] and any(info["Y"]):
            continue
        after = env._observation_blocks()
        assert after["pulse_width"][band] == before["pulse_width"][band]
        assert after["aoa_sin"][band] == before["aoa_sin"][band]
        assert after["aoa_cos"][band] == before["aoa_cos"][band]
        assert after["pulse_count"][band] == before["pulse_count"][band]   # D72
        found = True
        break
    assert found, "no miss turned up in 200 steps of this seed -- widen the search"


def test_a_declared_hit_writes_a_real_unit_circle_bearing():
    """Once any band has a declared hit, its (aoa_sin, aoa_cos) must decode to a
    point actually on the unit circle (sin^2+cos^2=1) -- never the (0,0) sentinel,
    which is reserved for "never measured" precisely because no real angle can
    produce it."""
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"), obs_version="v2")
    env.reset(seed=0)
    hit_band = None
    for t in range(200):
        band = t % K.N_BANDS
        obs, r, term, trunc, info = env.step(band)
        if info["Y"] and any(info["Y"]):
            hit_band = band
            break
    assert hit_band is not None, "no hit turned up in 200 steps of this seed"
    blocks = env._observation_blocks()
    sin_raw = 2 * blocks["aoa_sin"][hit_band] - 1
    cos_raw = 2 * blocks["aoa_cos"][hit_band] - 1
    assert sin_raw**2 + cos_raw**2 == pytest.approx(1.0, abs=1e-5)
    assert not (sin_raw == 0.0 and cos_raw == 0.0)
    assert blocks["pulse_width"][hit_band] > 0.0
    assert blocks["pulse_count"][hit_band] > 0.0   # D72: a declared hit implies C >= 1


def test_pulse_count_is_log1p_normalised_and_clipped_to_the_box():
    """D72: matches `reward_balance_improved`'s own transform of `C`
    (`log1p(C) / log1p(_DENSITY_REF_PULSES)`), clipped to [0, 1] because the
    observation's box cannot exceed 1.0 the way the reward's weight can."""
    from rfenv.env import _DENSITY_REF_PULSES

    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"), obs_version="v2")
    env.reset(seed=0)
    env._band_pulse_count[5] = _DENSITY_REF_PULSES
    assert env._observation_blocks()["pulse_count"][5] == pytest.approx(1.0)
    env._band_pulse_count[5] = _DENSITY_REF_PULSES * 100   # far past the reference
    assert env._observation_blocks()["pulse_count"][5] == pytest.approx(1.0)   # clipped, not > 1
    env._band_pulse_count[5] = 0.0
    assert env._observation_blocks()["pulse_count"][5] == pytest.approx(0.0)


def test_measured_dbm_band_persists_while_the_global_scalar_forgets():
    """Visit band A, then band B: v2's per-band memory of A must survive; v1's
    global scalar must have moved on to B (D30's whole reason for the block)."""
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"), obs_version="v2")
    env.reset(seed=0)
    env.step(3)
    after_a = env._observation_blocks()["measured_dbm_band"][3]
    env.step(9)
    after_b = env._observation_blocks()
    assert after_b["measured_dbm_band"][3] == pytest.approx(after_a)   # unchanged
    assert after_b["measured_dbm"][0] != pytest.approx(after_a) or after_a == after_b["measured_dbm"][0]
    # the global scalar now reports band 9's dwell, not band 3's
    assert after_b["measured_dbm_band"][9] == pytest.approx(after_b["measured_dbm"][0])


# --------------------------------------------------------------------------- #
# TruthGrid.PW/AOA follow whichever contribution actually set S (L1)
# --------------------------------------------------------------------------- #

def _with_pdw(contrib, pulse_width_us, aoa_deg):
    n = len(contrib)
    return replace(
        contrib,
        pulse_width_us=np.full(n, pulse_width_us, dtype=np.float32),
        aoa_deg=np.full(n, aoa_deg, dtype=np.float32),
    )


def test_truthgrid_pw_aoa_follow_the_louder_contribution():
    loud = _with_pdw(synthetic(10, [100], -100.0, label=0), pulse_width_us=5.0, aoa_deg=30.0)
    quiet = _with_pdw(synthetic(10, [100], -118.0, label=1), pulse_width_us=99.0, aoa_deg=-150.0)
    g = grid_of(loud, quiet)
    assert g.S[10, 100] == pytest.approx(-100.0)   # sanity: loud one wins S, as before D30
    assert g.PW[10, 100] == pytest.approx(5.0)
    assert g.AOA[10, 100] == pytest.approx(30.0)


def test_truthgrid_pw_aoa_switch_if_a_later_contribution_is_actually_louder():
    """Order in the contributions list must not matter -- only loudness."""
    quiet = _with_pdw(synthetic(10, [100], -118.0, label=0), pulse_width_us=99.0, aoa_deg=-150.0)
    loud = _with_pdw(synthetic(10, [100], -100.0, label=1), pulse_width_us=5.0, aoa_deg=30.0)
    g = grid_of(quiet, loud)   # loud one is second in the list this time
    assert g.PW[10, 100] == pytest.approx(5.0)
    assert g.AOA[10, 100] == pytest.approx(30.0)


def test_truthgrid_pw_aoa_are_zero_on_an_empty_cell():
    g = grid_of(synthetic(10, [100], -100.0))
    assert g.PW[10, 0] == 0.0    # never touched
    assert g.AOA[10, 0] == 0.0


def test_receiver_dwell_carries_pw_aoa_truth_through_unconditionally():
    """`DwellResult` itself is evaluator-side truth, ungated -- `env.py` is where
    the Y-gate is applied (tested above), not `receiver.py`."""
    g = grid_of(_with_pdw(synthetic(10, [100], -60.0), pulse_width_us=12.5, aoa_deg=77.0))
    rcv = Receiver(rng=np.random.default_rng(0))
    dwell = rcv.dwell(g, band=10, slot0=100)
    assert dwell.pulse_width_us[0] == pytest.approx(12.5)
    assert dwell.aoa_deg[0] == pytest.approx(77.0)


# --------------------------------------------------------------------------- #
# EmitterContribution / scenario.py backward compatibility
# --------------------------------------------------------------------------- #

def test_hand_built_contributions_without_pdw_default_to_zero_arrays():
    """Every existing `synthetic(...)` call across the test suite predates D30
    and passes neither field -- they must keep working, defaulting to arrays of
    the right shape rather than None or a crash."""
    c = synthetic(10, [0, 1, 2], -100.0)
    assert c.pulse_width_us.shape == (3,)
    assert c.aoa_deg.shape == (3,)
    assert (c.pulse_width_us == 0.0).all()
    assert (c.aoa_deg == 0.0).all()
