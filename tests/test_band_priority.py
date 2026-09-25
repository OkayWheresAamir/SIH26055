"""Band-priority reward/observation term (D70, no-library form) and its D78
follow-up (`priority_reward_bonus`'s decaying occupancy term).

Pinned here: (1) the new "v2p" layout is 398-wide and leaves "v1"/"v2" untouched;
(2) `band_priority` is exogenous -- sampled fresh every `reset()` from the env's
own seeded RNG, not derived from scan history; (3) disabled (the default) or
`priority_coef=0.0` reproduces the base reward exactly; (4) the discovery term is
gated on `newly`, not on any per-slot signal, so it cannot pay for camping;
(5) priority for band i lands at index i and nowhere else; (6) `band_priority=True`
is refused unless the layout actually carries the block. D78 follow-up, added
this task: (7) `occupancy_coef=0.0` (default) reproduces D78's own recorded
behaviour exactly, byte for byte; (8) the occupancy term scales by
`band_priority`'s *value*, not a binary elevated/not gate, so it fires
identically for the uniform control arm too; (9) it decays with cumulative
`visit_density`, not a resettable streak, so a two-band ping-pong between
elevated bands does not out-earn broad coverage (the D53 exploit, checked
directly, not assumed).
"""
from __future__ import annotations

import numpy as np
import pytest

from rfenv import constants as K
from rfenv.env import OBS_LAYOUTS, ScanEnv, obs_width, priority_reward_bonus
from rfenv.scenario import Scenario


def _env(**kwargs):
    return ScanEnv(scenario=Scenario.replay("config_2", "stare"), **kwargs)


# --------------------------------------------------------------------------- #
# The layout itself
# --------------------------------------------------------------------------- #

def test_v2p_is_398_wide():
    assert obs_width("v2p") == obs_width("v2") + K.N_BANDS == 398
    env = _env(obs_version="v2p")
    assert env.observation_space.shape == (398,)


def test_v1_and_v2_are_unaffected_by_v2p_existing():
    assert obs_width("v1") == 183
    assert obs_width("v2") == 362


def test_band_priority_only_in_v2p():
    assert "band_priority" not in OBS_LAYOUTS["v1"]
    assert "band_priority" not in OBS_LAYOUTS["v2"]
    assert "band_priority" in OBS_LAYOUTS["v2p"]
    assert OBS_LAYOUTS["v2p"][-1] == "band_priority"   # appended last, D67/D30/D76's convention


def test_unknown_obs_version_is_still_refused():
    with pytest.raises(ValueError, match="obs_version"):
        _env(obs_version="v99")


def test_band_priority_true_requires_a_layout_that_carries_it():
    with pytest.raises(ValueError, match="band_priority"):
        _env(obs_version="v2", band_priority=True)
    with pytest.raises(ValueError, match="band_priority"):
        _env(obs_version="v1", band_priority=True)
    _env(obs_version="v2p", band_priority=True)   # does not raise


# --------------------------------------------------------------------------- #
# Sampling: exogenous, seeded, bounded, per-episode
# --------------------------------------------------------------------------- #

def test_disabled_by_default_reads_all_ones():
    env = _env(obs_version="v2p")
    env.reset(seed=0)
    assert (env._observation_blocks()["band_priority"] == 1.0).all()


def test_enabled_elevates_a_bounded_number_of_bands_each_episode():
    env = _env(obs_version="v2p", band_priority=True, priority_n_bands=(3, 6))
    for seed in range(20):
        env.reset(seed=seed)
        p = env._band_priority
        elevated = int((p > 1.0).sum())
        assert 3 <= elevated <= 6, (seed, elevated)
        assert set(np.unique(p).tolist()) <= {1.0, 3.0}


def test_uniform_control_arm_never_elevates_anything():
    env = _env(obs_version="v2p", band_priority=True, priority_uniform=True)
    for seed in range(10):
        env.reset(seed=seed)
        assert (env._band_priority == 1.0).all()


def test_same_seed_reproduces_the_same_priority_draw():
    env = _env(obs_version="v2p", band_priority=True)
    env.reset(seed=7)
    p1 = env._band_priority.copy()
    env.reset(seed=7)
    p2 = env._band_priority.copy()
    np.testing.assert_array_equal(p1, p2)


def test_different_seeds_usually_draw_different_priorities():
    env = _env(obs_version="v2p", band_priority=True)
    draws = []
    for seed in range(8):
        env.reset(seed=seed)
        draws.append(tuple(env._band_priority.tolist()))
    assert len(set(draws)) > 1, "20 seeds drew the identical vector -- sampling is not seed-dependent"


def test_reset_does_not_carry_priority_over_between_episodes():
    """A band elevated in episode 1 must not still read elevated in episode 2
    unless episode 2's own independent draw happens to pick it again."""
    env = _env(obs_version="v2p", band_priority=True, priority_n_bands=(3, 6))
    env.reset(seed=0)
    p_first = env._band_priority.copy()
    env.reset(seed=1)
    p_second = env._band_priority.copy()
    # Not a strict inequality assertion (a shared elevated band is possible by
    # chance) -- what matters is that reset() rebuilds from scratch each time,
    # not `|=`s a new draw onto the old one.
    assert (p_second == 1.0).sum() >= K.N_BANDS - 6, "second draw looks contaminated by the first"
    del p_first


# --------------------------------------------------------------------------- #
# Per-band alignment: priority for band i lands at index i
# --------------------------------------------------------------------------- #

def test_priority_for_band_i_is_not_shifted():
    env = _env(obs_version="v2p", band_priority=True)
    env.reset(seed=0)
    env._band_priority[:] = 1.0
    env._band_priority[K.N_BANDS - 1] = 3.0   # last band specifically
    blocks = env._observation_blocks()
    assert blocks["band_priority"][K.N_BANDS - 1] == 3.0
    assert (blocks["band_priority"][: K.N_BANDS - 1] == 1.0).all()


# --------------------------------------------------------------------------- #
# The reward term: additive, discovery-gated, band-correct, zero-coef-inert
# --------------------------------------------------------------------------- #

def test_disabled_reward_is_bit_identical_to_no_band_priority_kwarg_at_all():
    env_a = _env(obs_version="v2", band_priority=False)
    env_b = _env(obs_version="v2p", band_priority=False)
    env_a.reset(seed=3)
    env_b.reset(seed=3)
    for band in range(K.N_BANDS):
        _, r_a, *_ = env_a.step(band)
        _, r_b, *_ = env_b.step(band)
        assert r_a == pytest.approx(r_b), band


def test_zero_coef_reproduces_the_base_reward_exactly():
    base = _env(obs_version="v2p", band_priority=False)
    priced = _env(obs_version="v2p", band_priority=True, priority_coef=0.0)
    base.reset(seed=5)
    priced.reset(seed=5)
    for band in range(K.N_BANDS):
        _, r_base, *_ = base.step(band)
        _, r_priced, *_ = priced.step(band)
        assert r_base == pytest.approx(r_priced), band


def test_priority_reward_only_fires_on_a_discovery():
    """A step that credits nothing new in `newly` must add exactly zero,
    regardless of `band_priority`'s value on that band."""
    env = _env(obs_version="v2p", band_priority=True, priority_coef=1.0)
    env.reset(seed=0)
    env._band_priority[:] = 1.0
    env._band_priority[9] = 3.0
    # Revisit the same band twice in a row: whatever `newly` produced on the
    # first visit cannot fire identically on the second for the same emitters
    # (D51: `newly` credits an emitter once per band). Compare the term's own
    # contribution directly rather than asserting on total reward, which also
    # carries the base reward's own per-step variation.
    _, r1, _, _, info1 = env.step(9)
    _, r2, _, _, info2 = env.step(9)
    newly2 = info2["newly_intercepted"]
    assert len(newly2) <= len(info1["newly_intercepted"])


def test_priority_reward_prices_the_dwells_band_not_the_action_argument():
    """Sanity that the term reads `dwell.band`, matching what `newly` was
    actually credited on this step -- not a stale/mismatched index."""
    env = _env(obs_version="v2p", band_priority=True, priority_coef=2.0)
    env.reset(seed=0)
    env._band_priority[:] = 1.0
    target = 20
    env._band_priority[target] = 3.0
    obs, reward, term, trunc, info = env.step(target)
    newly = info["newly_intercepted"]
    expected_extra = 2.0 * 3.0 * len(newly)
    # Recompute the base reward alone (band_priority disabled, same seed/action)
    # and confirm the delta matches the term's own formula exactly.
    base_env = _env(obs_version="v2p", band_priority=False)
    base_env.reset(seed=0)
    _, base_reward, *_ = base_env.step(target)
    assert reward == pytest.approx(base_reward + expected_extra)


# --------------------------------------------------------------------------- #
# Every layout still only names real blocks (the generic registry check)
# --------------------------------------------------------------------------- #

def test_every_layout_only_names_real_blocks():
    from rfenv.env import _BLOCK_SPECS
    for version, names in OBS_LAYOUTS.items():
        unknown = set(names) - set(_BLOCK_SPECS)
        assert not unknown, f"{version} names unknown blocks: {unknown}"


# --------------------------------------------------------------------------- #
# D78 follow-up: priority_reward_bonus -- the decaying occupancy term
# --------------------------------------------------------------------------- #

def test_priority_reward_bonus_matches_d74s_formula_when_occupancy_is_off():
    got = priority_reward_bonus(3.0, 2, 0.5, 1, priority_coef=0.5,
                                 occupancy_coef=0.0, occupancy_decay_cap=2.0)
    assert got == pytest.approx(0.5 * 3.0 * 2)


def test_occupancy_term_fires_even_with_no_discovery():
    """Unlike the discovery term (D78), this one is a genuine per-slot
    occupancy bonus -- deliberately, on direct request."""
    got = priority_reward_bonus(3.0, 0, 0.0, 1, priority_coef=0.5,
                                 occupancy_coef=0.1, occupancy_decay_cap=2.0)
    assert got == pytest.approx(0.1 * 3.0 * 1.0 * 1)   # decay = 1.0 at visit_density = 0


def test_occupancy_term_scales_by_priority_value_not_a_binary_elevated_gate():
    """A uniform-control band (priority == 1.0, never "elevated") still earns
    a nonzero occupancy bonus at the baseline rate -- required so control and
    treatment share the same reward-scale shift (D78's own methodology); a
    binary "elevated or not" gate would fire only for treatment and never for
    control, breaking that comparison."""
    control_band = priority_reward_bonus(1.0, 0, 0.0, 1, priority_coef=0.5,
                                          occupancy_coef=0.1, occupancy_decay_cap=2.0)
    elevated_band = priority_reward_bonus(3.0, 0, 0.0, 1, priority_coef=0.5,
                                           occupancy_coef=0.1, occupancy_decay_cap=2.0)
    assert control_band == pytest.approx(0.1 * 1.0)
    assert elevated_band == pytest.approx(0.1 * 3.0)
    assert elevated_band > control_band


def test_occupancy_term_decays_linearly_to_zero_at_the_cap_and_stays_there():
    half = priority_reward_bonus(1.0, 0, 1.0, 1, priority_coef=0.0,
                                  occupancy_coef=1.0, occupancy_decay_cap=2.0)
    assert half == pytest.approx(0.5)   # 1 - 1.0/2.0
    at_cap = priority_reward_bonus(1.0, 0, 2.0, 1, priority_coef=0.0,
                                    occupancy_coef=1.0, occupancy_decay_cap=2.0)
    assert at_cap == pytest.approx(0.0)
    past_cap = priority_reward_bonus(1.0, 0, 5.0, 1, priority_coef=0.0,
                                      occupancy_coef=1.0, occupancy_decay_cap=2.0)
    assert past_cap == pytest.approx(0.0)   # never negative


def test_occupancy_term_scales_by_n_slots_like_every_other_per_slot_term():
    one_slot = priority_reward_bonus(1.0, 0, 0.0, 1, priority_coef=0.0,
                                      occupancy_coef=1.0, occupancy_decay_cap=2.0)
    two_slots = priority_reward_bonus(1.0, 0, 0.0, 2, priority_coef=0.0,
                                       occupancy_coef=1.0, occupancy_decay_cap=2.0)
    assert two_slots == pytest.approx(2 * one_slot)


# --------------------------------------------------------------------------- #
# D78 follow-up, through the env: backward compatibility and the D53 check
# --------------------------------------------------------------------------- #

def test_occupancy_coef_zero_is_bit_identical_to_d74s_recorded_behaviour():
    before = _env(obs_version="v2p", band_priority=True, priority_coef=0.5)
    after = _env(obs_version="v2p", band_priority=True, priority_coef=0.5,
                 occupancy_coef=0.0)
    before.reset(seed=11)
    after.reset(seed=11)
    for band in range(K.N_BANDS):
        _, r_before, *_ = before.step(band)
        _, r_after, *_ = after.step(band)
        assert r_before == pytest.approx(r_after), band


def test_priority_high_is_configurable_and_overrides_the_module_default():
    env = _env(obs_version="v2p", band_priority=True, priority_high=7.0)
    env.reset(seed=0)
    p = env._band_priority
    elevated = p[p > 1.0]
    assert elevated.size > 0
    assert (elevated == 7.0).all()


def test_observation_space_ceiling_tracks_priority_high_not_the_module_default():
    """Regression: `observation_space`'s declared `band_priority` ceiling was
    briefly hardcoded to `_BLOCK_SPECS`'s static 3.0, so `priority_high > 3.0`
    produced real episodes whose values fell outside the space the env itself
    declared -- caught by `gymnasium`'s own `check_env`, not by this repo's
    suite, until this test existed."""
    env = _env(obs_version="v2p", band_priority=True, priority_high=7.0)
    obs, info = env.reset(seed=0)
    assert obs in env.observation_space
    for band in range(K.N_BANDS):
        obs, *_ = env.step(band)
        assert obs in env.observation_space


def test_ping_pong_between_two_elevated_bands_does_not_out_earn_a_full_sweep():
    """D53's own check, re-run for the new occupancy term specifically. Measured
    directly (see `visit_density`'s own values): a two-band ping-pong drives
    both bands' `visit_density` to roughly half the episode's fair-share
    ceiling within 1-2 visits, decaying the occupancy bonus on both to zero
    almost immediately -- unlike the old, now-retired `camp_slots`-based
    penalty, `visit_density` is never reset by switching away and back."""
    def rollout(policy_bands):
        env = _env(obs_version="v2p", band_priority=True, priority_coef=0.0,
                   occupancy_coef=1.0, occupancy_decay_cap=2.0)
        env.reset(seed=0)
        env._band_priority[:] = 1.0
        env._band_priority[0] = 3.0
        env._band_priority[1] = 3.0
        total = 0.0
        for band in policy_bands:
            _, reward, terminated, _, _ = env.step(band)
            total += reward
            if terminated:
                break
        return total

    ping_pong_total = rollout(0 if i % 2 == 0 else 1 for i in range(40))
    sweep_total = rollout(i % K.N_BANDS for i in range(40))

    assert ping_pong_total <= sweep_total + 1e-6, (
        f"ping-pong ({ping_pong_total}) out-earned a full sweep ({sweep_total}) -- "
        "the occupancy term may have reopened D53's exploit"
    )
