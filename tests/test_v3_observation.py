"""The "v3" in-context observation layout and `reward_balance_obs` (D75).

Pinned here: (1) "v3" is 436 wide and leaves "v1"/"v2"/"v2p" byte-identical;
(2) the cold-start sentinels are what they claim to be, including the deliberate
disagreement between `prev_action` and `current_band` at t=0; (3) `prev_action`
*is* `current_band` after any step -- the known redundancy, pinned so a later
change to either is noticed rather than discovered; (4) `reward_balance_obs` is
`reward_balance` with `Y` for `Z`, exactly, and two dwells with the same `Y` and
different `Z` are indistinguishable to it -- which is the deployability claim in
mechanical form; (5) it never reaches `total_reward`; (6) the ablation's block
corruption preserves the receiver's noise stream, so a paired run stays paired.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rfenv import constants as K
from rfenv.env import (
    _BLOCK_SPECS,
    _REWARD_OBS_CLAMP,
    OBS_LAYOUTS,
    ScanEnv,
    _normalise_prev_reward,
    obs_width,
    reward_balance,
    reward_balance_obs,
)
from rfenv.receiver import DwellResult
from rfenv.scenario import Scenario


def _env(**kwargs):
    kwargs.setdefault("obs_version", "v3")
    kwargs.setdefault("band_priority", True)
    return ScanEnv(scenario=Scenario.replay("config_2", "stare"), **kwargs)


def _offsets(version: str) -> dict[str, slice]:
    out, i = {}, 0
    for name in OBS_LAYOUTS[version]:
        width = _BLOCK_SPECS[name][0]
        out[name] = slice(i, i + width)
        i += width
    return out


OFF = _offsets("v3")


def _dwell(Y, Z, *, band=0, slot0=0) -> DwellResult:
    Y = np.asarray(Y, dtype=bool)
    Z = np.asarray(Z, dtype=bool)
    n = len(Y)
    return DwellResult(
        band=band, slot0=slot0,
        level_dbm=np.full(n, -100.0), measured_dbm=np.full(n, -100.0),
        Y=Y, Z=Z, C=np.zeros(n, dtype=np.int32),
        pulse_width_us=np.zeros(n, dtype=np.float32),
        aoa_deg=np.zeros(n, dtype=np.float32),
        intercepts={},
    )


# --------------------------------------------------------------------------- #
# The layout
# --------------------------------------------------------------------------- #

def test_v3_is_436_wide():
    assert obs_width("v3") == obs_width("v2p") + 2 * 1 + K.N_BANDS == 436
    assert _env().observation_space.shape == (436,)


def test_v3_is_v2p_plus_exactly_three_named_blocks_in_order():
    assert OBS_LAYOUTS["v3"][: len(OBS_LAYOUTS["v2p"])] == OBS_LAYOUTS["v2p"]
    assert OBS_LAYOUTS["v3"][len(OBS_LAYOUTS["v2p"]):] == (
        "prev_action", "prev_reward", "prev_hit"
    )


def test_earlier_layouts_are_untouched_by_v3_existing():
    assert obs_width("v1") == 183
    assert obs_width("v2") == 362
    assert obs_width("v2p") == 398


def test_the_three_new_blocks_declare_the_unit_interval_even_at_a_raised_priority_high():
    """They must not inherit `band_priority`'s `_high_for` override (D74's bug)."""
    env = _env(priority_high=5.0)
    high = env.observation_space.high
    for name in ("prev_action", "prev_reward", "prev_hit"):
        assert np.all(high[OFF[name]] == 1.0), name
    assert np.all(high[OFF["band_priority"]] == 5.0)


def test_v3_passes_the_gymnasium_env_checker():
    check_env(_env(), skip_render_check=True)


# --------------------------------------------------------------------------- #
# Cold start
# --------------------------------------------------------------------------- #

def test_prev_action_is_all_zero_at_reset_while_current_band_is_one_hot_at_zero():
    """Asserted together, because the disagreement is the point.

    `current_band` claims band 0 after `reset()` -- a dwell that never happened.
    `prev_action` reads the zero vector instead, which is off the one-hot simplex
    and so unreachable by any real action. That difference is the only
    information `prev_action` carries over `current_band` in the whole layout.
    """
    obs, _ = _env().reset(seed=0)
    assert obs[OFF["prev_action"]].sum() == 0.0
    assert obs[OFF["current_band"]].sum() == 1.0
    assert obs[OFF["current_band"]].argmax() == 0


def test_prev_reward_is_exactly_one_half_at_reset():
    """Zero reward, and the symmetric clamp puts zero at the midpoint exactly."""
    obs, _ = _env().reset(seed=0)
    assert obs[OFF["prev_reward"]][0] == 0.5
    assert _normalise_prev_reward(0.0) == 0.5


def test_prev_hit_is_zero_at_reset():
    obs, _ = _env().reset(seed=0)
    assert obs[OFF["prev_hit"]][0] == 0.0


def test_a_second_episode_carries_nothing_from_the_first(_band=13):
    env = _env()
    env.reset(seed=0)
    for _ in range(20):
        env.step(_band)
    obs, _ = env.reset(seed=1)
    assert obs[OFF["prev_action"]].sum() == 0.0
    assert obs[OFF["prev_reward"]][0] == 0.5
    assert obs[OFF["prev_hit"]][0] == 0.0


# --------------------------------------------------------------------------- #
# The known redundancy
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("band", [0, 7, 17, 35])
def test_prev_action_is_one_hot_at_the_last_action_and_equals_current_band(band):
    """Pins the redundancy documented in OBS_LAYOUTS["v3"].

    If this ever fails, one of the two blocks changed meaning, and any ablation
    conclusion drawn about `prev_action` needs revisiting.
    """
    env = _env()
    env.reset(seed=0)
    obs, *_ = env.step(band)
    assert obs[OFF["prev_action"]].argmax() == band
    assert obs[OFF["prev_action"]].sum() == 1.0
    assert np.array_equal(obs[OFF["prev_action"]], obs[OFF["current_band"]])


def test_prev_hit_agrees_with_current_hit_streak_being_positive():
    env = _env()
    env.reset(seed=0)
    for k in range(60):
        obs, *_ = env.step((5 * k) % 36)
        assert bool(obs[OFF["prev_hit"]][0]) == bool(obs[OFF["current_hit_streak"]][0] > 0)


# --------------------------------------------------------------------------- #
# reward_balance_obs, in isolation
# --------------------------------------------------------------------------- #

def _args(rng):
    return (
        set(),
        0,
        rng.random(K.N_BANDS),
        rng.random(K.N_BANDS) * K.N_BANDS,
        rng.random(K.N_BANDS) * 13.9,
        int(rng.integers(K.N_BANDS)),
    )


def test_it_equals_reward_balance_whenever_Y_and_Z_agree():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 3))
        same = rng.random(n) < 0.5
        d = _dwell(same, same)
        rest = _args(rng)
        assert reward_balance_obs(d, *rest) == reward_balance(d, *rest)


def test_the_difference_is_exactly_half_the_Y_minus_Z_count():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 3))
        d = _dwell(rng.random(n) < 0.5, rng.random(n) < 0.5)
        rest = _args(rng)
        delta = reward_balance_obs(d, *rest) - reward_balance(d, *rest)
        assert delta == pytest.approx(0.5 * (int(d.Y.sum()) - int(d.Z.sum())))


def test_two_dwells_with_the_same_Y_and_different_Z_are_indistinguishable():
    """The deployability claim, mechanically: nothing truth-side survives.

    A fielded receiver knows `Y` and cannot know `Z`. If this ever fails, the
    "v3" `prev_reward` block is leaking the simulator into the policy's input
    path and the rung is no longer `deployable=True`.
    """
    rng = np.random.default_rng(2)
    Y = np.array([True, False])
    rest = _args(rng)
    a = reward_balance_obs(_dwell(Y, [True, True]), *rest)
    b = reward_balance_obs(_dwell(Y, [False, False]), *rest)
    assert a == b


def test_it_leaves_the_dwell_untouched():
    """`dataclasses.replace` must not mutate the original -- arrays are shared."""
    d = _dwell([True, False], [False, True])
    before_Y, before_Z = d.Y.copy(), d.Z.copy()
    reward_balance_obs(d, *_args(np.random.default_rng(3)))
    assert np.array_equal(d.Y, before_Y)
    assert np.array_equal(d.Z, before_Z)


# --------------------------------------------------------------------------- #
# The normaliser
# --------------------------------------------------------------------------- #

def test_the_normaliser_is_monotone_centred_and_saturating():
    xs = np.linspace(-3 * _REWARD_OBS_CLAMP, 3 * _REWARD_OBS_CLAMP, 501)
    ys = np.array([_normalise_prev_reward(x) for x in xs])
    assert np.all(np.diff(ys) >= 0)
    assert np.all((ys >= 0.0) & (ys <= 1.0))
    assert _normalise_prev_reward(0.0) == 0.5
    assert _normalise_prev_reward(-_REWARD_OBS_CLAMP) == 0.0
    assert _normalise_prev_reward(+_REWARD_OBS_CLAMP) == 1.0
    assert _normalise_prev_reward(-1e9) == 0.0
    assert _normalise_prev_reward(+1e9) == 1.0


def test_the_clamp_covers_the_analytic_range_of_the_formula():
    """Terms bound to [-6, +3]; the clamp is set from that, not from a percentile.

    +0.5*hit_rate <= 0.5; +(1.5-vd)*st <= 1.5; +0.5*Y.sum() <= 1.0 (n_slots<=2);
    -3.0*vd*n_slots >= -6.0. So the clip is incapable of binding on real data.
    """
    assert _REWARD_OBS_CLAMP >= 6.0


def test_the_prev_reward_slot_tracks_last_reward_obs_at_every_step():
    env = _env()
    env.reset(seed=0)
    for k in range(80):
        obs, *_ = env.step((7 * k) % 36)
        assert obs[OFF["prev_reward"]][0] == pytest.approx(
            _normalise_prev_reward(env.last_reward_obs), abs=1e-7
        )


def test_the_observable_reward_never_reaches_total_reward():
    """It is what the policy sees, not what it is paid."""
    a = ScanEnv(scenario=Scenario.replay("config_2", "stare"), obs_version="v2p",
                band_priority=True)
    b = _env()
    a.reset(seed=5)
    b.reset(seed=5)
    differed = False
    for k in range(200):
        a.step((3 * k) % 36)
        b.step((3 * k) % 36)
        differed |= b.last_reward_obs != b.total_reward
    assert a.total_reward == b.total_reward
    assert differed


def test_the_observable_and_training_rewards_really_do_differ_somewhere():
    """Pd = 0.842 -- if these never diverged, Y and Z would be the same thing."""
    env = _env()
    env.reset(seed=0)
    diffs = 0
    for k in range(300):
        _, reward, done, _, _ = env.step((13 * k) % 36)
        diffs += env.last_reward_obs != reward
        if done:
            break
    assert diffs > 0


# --------------------------------------------------------------------------- #
# v2p is bit-identical underneath v3
# --------------------------------------------------------------------------- #

def test_v3_contains_v2p_byte_for_byte_and_pays_the_same_reward():
    scenario = Scenario.replay("config_2", "stare")
    a = ScanEnv(scenario=scenario, obs_version="v2p", band_priority=True)
    b = ScanEnv(scenario=scenario, obs_version="v3", band_priority=True)
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    assert np.array_equal(oa, ob[:398])
    k, done = 0, False
    while not done:
        oa, ra, done, _, _ = a.step((5 * k) % 36)
        ob, rb, _, _, _ = b.step((5 * k) % 36)
        assert np.array_equal(oa, ob[:398])
        assert ra == rb
        k += 1
    assert a.total_reward == b.total_reward
    assert a.episode_metrics() == b.episode_metrics()


# --------------------------------------------------------------------------- #
# The ablation hook
# --------------------------------------------------------------------------- #

def test_corruption_does_not_move_the_receivers_noise_stream():
    """Without this the clean/corrupted pair is not a paired comparison."""
    scenario = Scenario.replay("config_2", "stare")

    def levels(corrupt):
        env = ScanEnv(scenario=scenario, obs_version="v3", band_priority=True,
                      corrupt_obs_blocks=corrupt)
        env.reset(seed=7)
        out, k, done = [], 0, False
        while not done:
            _, _, done, _, _ = env.step((11 * k) % 36)
            out.append(env._last_measured_dbm)
            k += 1
        return np.array(out)

    clean = levels(())
    assert np.array_equal(clean, levels(("prev_reward",)))
    assert np.array_equal(clean, levels(("hit_rate",)))


def test_corruption_changes_the_block_it_names_and_leaves_the_others_alone():
    scenario = Scenario.replay("config_2", "stare")
    clean = ScanEnv(scenario=scenario, obs_version="v3", band_priority=True)
    dirty = ScanEnv(scenario=scenario, obs_version="v3", band_priority=True,
                    corrupt_obs_blocks=("prev_reward",))
    clean.reset(seed=1)
    dirty.reset(seed=1)
    moved, untouched = 0, True
    for k in range(150):
        oc, *_ = clean.step((9 * k) % 36)
        od, *_ = dirty.step((9 * k) % 36)
        moved += oc[OFF["prev_reward"]][0] != od[OFF["prev_reward"]][0]
        untouched &= np.array_equal(oc[OFF["hit_rate"]], od[OFF["hit_rate"]])
    assert moved > 0
    assert untouched


def test_corruption_only_ever_emits_values_the_block_really_took():
    """Resampling preserves the marginal; it does not invent out-of-range values."""
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"), obs_version="v3",
                  band_priority=True, corrupt_obs_blocks=("prev_action",))
    env.reset(seed=2)
    for k in range(120):
        obs, *_ = env.step((4 * k) % 36)
        block = obs[OFF["prev_action"]]
        assert block.sum() in (0.0, 1.0)
        assert set(np.unique(block)).issubset({0.0, 1.0})


@pytest.mark.parametrize("blocks,version", [
    (("no_such_block",), "v3"),
    (("band_priority",), "v1"),
])
def test_corrupt_obs_blocks_refuses_a_block_the_layout_cannot_show(blocks, version):
    with pytest.raises(ValueError):
        ScanEnv(scenario=Scenario.replay("config_2", "stare"),
                obs_version=version, corrupt_obs_blocks=blocks)
