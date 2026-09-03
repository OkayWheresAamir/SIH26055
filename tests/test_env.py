"""L3 checks: the Gymnasium contract, the clock, and the guarantees that make the
scheduler metrics mean what they say.

The sharpest one is `first_e >= on_e`. It holds by construction because D28's
clause 2 and D27's `on_e` are the same own-level rule -- if it ever fails, the
two halves of censored intercept time have drifted apart.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rfenv import constants as K
from rfenv.env import DEFAULT_REWARD, REWARDS, ScanEnv
from rfenv.scenario import EmitterPool, Scenario
from rfenv.truth import TruthGrid
from tests.test_receiver import grid_of, synthetic


def episode(env, policy, seed=0):
    """Run one episode to termination; return the final info."""
    env.reset(seed=seed)
    terminated = False
    while not terminated:
        _, _, terminated, truncated, info = env.step(policy(env))
        assert not truncated  # D35: the horizon terminates, it does not truncate
    return info


ROUND_ROBIN = lambda env: env.n_steps % K.N_BANDS


# --------------------------------------------------------------------------- #
# The Gymnasium contract
# --------------------------------------------------------------------------- #

def test_passes_the_gymnasium_env_checker():
    check_env(ScanEnv(scenario=Scenario.replay("config_59", "stare")), skip_render_check=True)


def test_spaces_are_the_specified_ones():
    """Discrete(36) and a 36x3+1 = 109 box (D34). Every component of the
    observation is natively a fraction, so the box is the unit interval."""
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"))
    assert env.action_space.n == 36
    assert env.observation_space.shape == (109,)
    assert env.observation_space.dtype == np.float32

    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, *_ = env.step(int(env.np_random.integers(36)))
        assert env.observation_space.contains(obs)
        assert (obs >= 0).all() and (obs <= 1).all()


def test_rejects_an_action_outside_the_band_set():
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(36)


# --------------------------------------------------------------------------- #
# The clock (D3, D16, D31, D35)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("policy,name", [
    (ROUND_ROBIN, "round-robin"),
    (lambda env: 0, "camper on a wide band"),
    (lambda env: 5, "camper on a narrow band"),
    (lambda env: int(env.np_random.integers(36)), "random"),
])
def test_every_episode_covers_exactly_600_slots(policy, name):
    """An episode is 600 slots, never 600 steps. Whatever the policy, the wall
    clock lands exactly on 30 s -- what varies is how many decisions bought it."""
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"))
    episode(env, policy)
    assert env.t == K.N_SLOTS
    assert len(env.log) == K.N_SLOTS
    assert [r["slot"] for r in env.log] == list(range(K.N_SLOTS))


def test_step_count_varies_with_dwell_width_but_airtime_does_not():
    """Camping a 100 ms band costs 300 decisions; a 50 ms band costs 600. Both
    spend the same 30 s. That is D31's invariant: airtime is the only currency."""
    sc = Scenario.replay("config_59", "stare")
    wide = ScanEnv(scenario=sc); episode(wide, lambda env: 0)
    narrow = ScanEnv(scenario=sc); episode(narrow, lambda env: 5)
    assert wide.n_steps == 300 and narrow.n_steps == 600
    assert wide.t == narrow.t == K.N_SLOTS


# --------------------------------------------------------------------------- #
# Reward (D29, D31)
# --------------------------------------------------------------------------- #

def test_a_wide_dwell_scores_both_its_slots():
    """D31: reward is per slot and a dwell's reward is the sum of its slots', so a
    100 ms dwell on an occupied pair earns +2. The rejected alternative -- +1 per
    dwell -- would halve a wide band's rate and make those seven bands strictly
    dominated."""
    env = ScanEnv(scenario=Scenario(name="synthetic",
                                    contributions=[synthetic(0, [0, 1], -80.0)]))
    env.reset(seed=0)
    _, reward, *_ = env.step(0)          # band 0 is wide: slots 0 and 1, both occupied
    assert env.t == 2
    assert reward == 2.0


def test_reward_per_unit_time_is_equal_across_dwell_widths():
    """The invariant D31 exists to protect. Two bands, identical occupancy on
    every slot, one wide and one narrow: the same reward per slot of airtime."""
    sc = Scenario(name="synthetic", contributions=[
        synthetic(0, range(600), -80.0, label=0),   # wide band, always on
        synthetic(5, range(600), -80.0, label=1),   # narrow band, always on
    ])
    wide = ScanEnv(scenario=sc); episode(wide, lambda env: 0)
    narrow = ScanEnv(scenario=sc); episode(narrow, lambda env: 5)
    assert wide.total_reward == narrow.total_reward == float(K.N_SLOTS)


def test_all_three_reward_candidates_run_and_differ():
    """Exactly three, and the set stays at three (D29). Nothing here ranks them --
    the selection rule is open and is the human's."""
    assert set(REWARDS) == {"hit_z", "hit_y", "first_intercept"}
    assert DEFAULT_REWARD == "hit_z"

    sc = Scenario.replay("config_921", "stare")
    totals = {}
    for name in REWARDS:
        env = ScanEnv(scenario=sc, reward=name)
        episode(env, ROUND_ROBIN)
        totals[name] = env.total_reward
    # candidate 3 counts each emitter once, so it is bounded by |E| and far smaller
    assert totals["first_intercept"] < totals["hit_y"]
    assert totals["first_intercept"] <= len(env.detectable)
    # 1 and 2 are both per-slot counts over the same looks, so they are close but
    # not identical: Y misses weak occupied cells and fires on empty ones.
    assert totals["hit_z"] != totals["hit_y"]


def test_an_unknown_reward_is_refused():
    with pytest.raises(ValueError):
        ScanEnv(scenario=Scenario.replay("config_59", "stare"), reward="staleness")


# --------------------------------------------------------------------------- #
# The guarantees the metrics rest on
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("config_id", ["config_2", "config_59", "config_921", "config_81"])
def test_first_intercept_never_precedes_the_detectable_interval(config_id):
    """`first_e >= on_e`, for every emitter, in every episode.

    True by construction, not by luck: D28's clause 2 judges an intercept on the
    emitter's *own* level clearing gamma, and D27's `on_e` is the first slot that
    same level clears it. One rule for the numerator and the denominator of
    censored intercept time. The gamma-free alternative breaks this -- it was
    measured producing an intercept time of -0.05 s (D28).
    """
    env = ScanEnv(scenario=Scenario.replay(config_id, "stare"))
    episode(env, ROUND_ROBIN)
    assert env.detectable, "scenario has no detectable emitters to check"
    for emitter, first in env.first_intercept.items():
        on_e, off_e = env.detectable[emitter]
        assert first >= on_e
        assert first <= off_e


def test_censoring_uses_the_full_episode_for_emitters_never_found():
    """An emitter never intercepted counts at the episode end, not dropped from
    the average -- averaging over only what you found rewards not looking, and
    that is measured, not hypothetical (EVALUATION.md §4 trap 2)."""
    sc = Scenario(name="synthetic", contributions=[
        synthetic(10, range(600), -80.0, label=0),   # loud, on band 10 all episode
        synthetic(30, range(600), -80.0, label=1),   # loud, on band 30, never looked at
    ])
    env = ScanEnv(scenario=sc)
    episode(env, lambda e: 10)
    m = env.episode_metrics()
    assert m["n_detectable"] == 2 and m["n_intercepted"] == 1
    assert m["emitter_coverage"] == 0.5
    # found at slot 0 (delay 0) and never found (delay 600) -> mean 300 slots = 15 s
    assert m["censored_mean_intercept_time_s"] == pytest.approx(15.0)


def test_interception_ratio_counts_illuminations_not_dwells():
    """Per illumination, not per dwell (EVALUATION.md §4). Driving Turing's own
    sweep through the env must reproduce the ratio computed straight off the grid
    -- the plumbing check that env and metrics agree before metrics.py exists."""
    sc = Scenario.replay("config_921", "stare")
    env = ScanEnv(scenario=sc)
    episode(env, ROUND_ROBIN)

    grid = TruthGrid.from_scenario(sc)
    expected = 0
    for start, end, band in K.dwell_schedule():
        s0, s1 = K.slots_for_dwell(start, end)
        expected += int(grid.C[int(band), s0:s1].sum())
    assert env.pulses_intercepted == expected
    assert env.episode_metrics()["interception_ratio"] == pytest.approx(
        expected / grid.total_pulses)


# --------------------------------------------------------------------------- #
# Cold start, and what the agent may see (D19, D20, D29, D34)
# --------------------------------------------------------------------------- #

def test_the_observation_reads_no_truth_at_all():
    """The observation ships; the reward does not (D29). So the vector must be a
    pure function of the agent's own scan history.

    Proved directly: run an episode partway, then swap the truth grid underneath
    the env for a completely different one. The observation must not move.
    """
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"))
    env.reset(seed=0)
    for i in range(50):
        env.step(i % 36)
    before = env._observation().copy()

    env.grid = grid_of(synthetic(3, range(600), -20.0))   # a different world entirely
    assert np.array_equal(env._observation(), before)


def test_every_episode_starts_cold():
    """No carry-over between episodes: no threat library, no map of who transmits
    where (D20). The first observation of any episode is the same zero-history
    vector whatever ran before it."""
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"))
    first, _ = env.reset(seed=0)
    episode(env, ROUND_ROBIN, seed=1)
    again, _ = env.reset(seed=2)
    assert np.array_equal(first, again)
    # staleness is 1.0 everywhere (nothing seen), hit rate and density are 0
    assert (first[:36] == 0).all() and (first[36:72] == 0).all()
    assert (first[72:108] == 1).all() and first[108] == 0


# --------------------------------------------------------------------------- #
# Reproducibility (EVALUATION.md §7)
# --------------------------------------------------------------------------- #

def test_a_seed_reproduces_an_episode_exactly():
    """Schedulers are compared on identical scenarios *and seeds*, so the seed has
    to fix the noise draw as well as the scenario draw."""
    sc = Scenario.replay("config_2", "stare")
    runs = []
    for _ in range(2):
        env = ScanEnv(scenario=sc)
        episode(env, lambda e: int(e.np_random.integers(36)), seed=7)
        runs.append((env.total_reward, env.pulses_intercepted, dict(env.first_intercept),
                     [r["measured_dbm"] for r in env.log]))
    assert runs[0] == runs[1]


def test_sampled_scenarios_come_from_the_pool_and_vary():
    """Training draws a fresh scenario each reset, so no episode repeats and there
    is nothing to memorise (D25, D32)."""
    env = ScanEnv(pool=EmitterPool.from_train())
    names = set()
    for seed in range(5):
        env.reset(seed=seed)
        names.add(tuple(c.uid for c in env.scenario.contributions))
        assert env.scenario.name.startswith("sample:")
    assert len(names) == 5


def test_needs_a_scenario_or_a_pool():
    with pytest.raises(ValueError):
        ScanEnv()


def test_the_emitter_table_carries_more_than_first_sightings():
    """`EVALUATION.md` §8 artefact 2, which `first_e` alone cannot supply.

    Measured under round-robin: 92.8% of config_2's intercept events and 94.6% of
    config_921's are re-sightings rather than first sightings, and the per-slot
    episode log has no emitter attribution to recover them from. So the env keeps
    the whole track; the alternative was re-deriving D28's three clauses inside
    metrics.py, putting one rule in two places.
    """
    env = ScanEnv(scenario=Scenario.replay("config_921", "stare"))
    info = episode(env, ROUND_ROBIN)

    table = env.emitter_table()
    assert len(table) == len(env.detectable)          # one row per emitter in E
    assert info["emitter_table"] == table             # and it reaches the evaluator

    seen = [r for r in table if r["intercept_count"] > 0]
    assert seen, "round-robin should intercept something in config_921"
    assert sum(r["intercept_count"] for r in seen) > len(seen)   # re-sightings kept
    for r in seen:
        assert r["on_slot"] <= r["first_intercept_slot"] <= r["last_intercept_slot"]
        assert r["bands_seen_in"] and all(0 <= b < K.N_BANDS for b in r["bands_seen_in"])
        assert r["uid"].endswith(f"/{r['emitter']}") or "/" in r["uid"]

    # an emitter never intercepted is present and scored as a miss, not dropped
    for r in table:
        if r["intercept_count"] == 0:
            assert r["first_intercept_slot"] is None and r["bands_seen_in"] == []


def test_first_intercept_still_reads_as_before():
    """The property is a view over the tracks, so nothing downstream shifted."""
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"))
    episode(env, ROUND_ROBIN)
    assert env.first_intercept == {e: t["first"] for e, t in env.tracks.items()}
    assert all(env.detectable[e][0] <= s for e, s in env.first_intercept.items())
