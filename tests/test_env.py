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
from rfenv.env import DEFAULT_REWARD, REWARDS, ScanEnv, make_reward_weighted
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


def test_passes_the_gymnasium_render_check():
    """With a render_mode actually set, the render check need not be skipped."""
    check_env(ScanEnv(scenario=Scenario.replay("config_81", "stare"), render_mode="rgb_array"))


# --------------------------------------------------------------------------- #
# Optional rendering -- off unless render_mode is set (nothing about
# reset()/step() changes either way)
# --------------------------------------------------------------------------- #

def test_render_is_off_by_default():
    env = ScanEnv(scenario=Scenario.replay("config_81", "stare"))
    env.reset(seed=0)
    assert env.render() is None


def test_render_mode_rejects_an_unknown_value():
    with pytest.raises(ValueError):
        ScanEnv(scenario=Scenario.replay("config_81", "stare"), render_mode="human")


def test_render_before_reset_raises():
    env = ScanEnv(scenario=Scenario.replay("config_81", "stare"), render_mode="rgb_array")
    with pytest.raises(RuntimeError):
        env.render()


def test_rgb_array_render_is_a_valid_frame():
    env = ScanEnv(scenario=Scenario.replay("config_81", "stare"), render_mode="rgb_array")
    env.reset(seed=0)
    env.step(0)
    frame = env.render()
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_spaces_are_the_specified_ones():
    """Discrete(36) and a 36x5+3 = 183 box (D34, extended by D49, rescaled by
    D55, extended again by D67).

    The box is **not** the unit interval, which is the point of D55: two blocks
    declare ceilings above 1.0 so that 1.0 means something inside them -- equal
    airtime for visit_density, one full pass overdue for staleness. A box that
    lied about those bounds would be caught by `check_env`, and a box that kept
    them at 1.0 would be the compressed scaling D55 removed.
    """
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"))
    assert env.action_space.n == 36
    assert env.observation_space.shape == (183,)
    assert env.observation_space.dtype == np.float32

    high = env.observation_space.high
    assert (high[:36] == 1.0).all()                                   # hit_rate
    assert high[36:72] == pytest.approx(36.0)                         # visit_density, fair shares
    assert high[72:108] == pytest.approx(600 / 43, rel=1e-6)          # staleness, sweeps
    assert (high[108:] == 1.0).all()                                  # one-hot, clock, dbm, hit_streak

    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, *_ = env.step(int(env.np_random.integers(36)))
        assert env.observation_space.contains(obs)


def test_rejects_an_action_outside_the_band_set():
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(36)


def test_episode_metrics_is_snapshotted_into_terminal_info():
    """info["episode_metrics"] exists only on the terminating step, and matches
    episode_metrics() called at that exact moment.

    This is specifically for SB3's DummyVecEnv, which auto-resets a sub-env
    the instant its episode ends -- before any training callback gets to run
    -- so a callback reading env.episode_metrics() after the fact would
    silently see the *next* episode instead (measured: n_steps reads back 0).
    Snapshotting it into info at the moment of termination is what a callback
    actually needs; see rfenv/rl/common.py's EpisodeMetricsCallback.
    """
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"))
    env.reset(seed=0)
    terminated = False
    info = {}
    while not terminated:
        assert "episode_metrics" not in info
        expected = None
        _, _, terminated, _, info = env.step(ROUND_ROBIN(env))
        if terminated:
            expected = env.episode_metrics()
    assert info["episode_metrics"] == expected
    assert info["episode_metrics"]["n_steps"] == env.n_steps > 0


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
    dominated.

    **Not pinned to a registered candidate.** This used to run against `hit_z`,
    the simplest candidate that was purely a per-slot Z count -- but `hit_z` and
    `hit_y` were removed after both failed D62's screen (they rank the camper
    above every sweeping policy), and none of the four remaining candidates is a
    bare per-slot count: each mixes in a state-dependent term. So this tests
    `ScanEnv.step`'s own slot-summation mechanics directly, via a reward function
    built for exactly that and nothing else -- decoupled from whichever
    candidates happen to be registered.
    """
    raw_z_count = lambda dwell, newly, camp_slots, hr, vd, st, a: float(dwell.Z.sum())
    env = ScanEnv(scenario=Scenario(name="synthetic",
                                    contributions=[synthetic(0, [0, 1], -80.0)]),
                  reward="reward_balance")
    env._reward_fn = raw_z_count
    env.reset(seed=0)
    _, reward, *_ = env.step(0)          # band 0 is wide: slots 0 and 1, both occupied
    assert env.t == 2
    assert reward == 2.0


def test_reward_per_unit_time_is_equal_across_dwell_widths():
    """The invariant D31 exists to protect. Two bands, identical occupancy on
    every slot, one wide and one narrow: the same reward per slot of airtime.

    **Not pinned to a registered candidate**, for the reason the test above
    gives -- `hit_z`/`hit_y` are gone and no surviving candidate is a bare
    per-slot count. Same raw-Z-count function as above.
    """
    raw_z_count = lambda dwell, newly, camp_slots, hr, vd, st, a: float(dwell.Z.sum())
    sc = Scenario(name="synthetic", contributions=[
        synthetic(0, range(600), -80.0, label=0),   # wide band, always on
        synthetic(5, range(600), -80.0, label=1),   # narrow band, always on
    ])
    wide = ScanEnv(scenario=sc, reward="reward_balance"); wide._reward_fn = raw_z_count
    episode(wide, lambda env: 0)
    narrow = ScanEnv(scenario=sc, reward="reward_balance"); narrow._reward_fn = raw_z_count
    episode(narrow, lambda env: 5)
    assert wide.total_reward == narrow.total_reward == float(K.N_SLOTS)


def test_all_reward_candidates_run_and_differ():
    """The registered set: six since D57, which lifted D29's cap of three.

    Candidate 3 has been rewritten more than once (`weighted_camp` and two
    `first_intercept` shapes were each registered and retired), and D57 added
    `greedy`/`explore`/`weighted` as a designed axis rather than three more
    guesses. What has held throughout is that `DEFAULT_REWARD` names a
    registered key and that the candidates are genuinely different functions.
    Those are what this pins. Nothing here ranks them -- the selection rule is
    open and is the human's (D29, D47).

    `DEFAULT_REWARD` is asserted to be *a key of REWARDS* rather than a
    specific name, because naming one that does not exist is the failure this
    catches: `ScanEnv.__init__` validates against REWARDS, so a stale default
    makes every unqualified `ScanEnv()` raise -- which is exactly what happened
    when candidate 3 was renamed and the default was not.
    """
    assert set(REWARDS) == {"reward_balance", "reward_balance_improved",
                           "greedy", "explore", "weighted"}
    assert DEFAULT_REWARD in REWARDS

    sc = Scenario.replay("config_921", "stare")
    totals = {}
    n_steps = {}
    for name in REWARDS:
        env = ScanEnv(scenario=sc, reward=name)
        episode(env, ROUND_ROBIN)
        totals[name] = env.total_reward
        n_steps[name] = env.n_steps
    # reward_balance is the only one of the four that can go negative: its
    # camping term subtracts against the chosen band's airtime share, with
    # nothing bounding it below. That asymmetry is the candidate's whole point,
    # so assert it is possible rather than asserting a range that would hide it.
    assert isinstance(totals["reward_balance"], float)
    # All four differ -- four different shapes of reward, not variations on one.
    values = list(totals.values())
    assert len(set(values)) == len(values)


def test_the_reward_reads_the_same_staleness_the_observation_reports():
    """`reward_balance` promises it prices exploration off observation
    components. `_staleness_array` once held `_last_slot[action] / N_SLOTS` --
    *when* a band was last seen, the inverse of *how long ago* -- so the
    exploration term paid most for revisiting the freshest band and paid a small
    negative for returning to one abandoned hundreds of slots earlier.

    Nothing caught it because the array is reward-facing only: `_observation()`
    recomputes staleness from the raw counters and was always right. This pins
    the two together at the one index the reward actually reads.
    """
    env = ScanEnv(pool=EmitterPool.from_train(), reward="reward_balance")
    env.reset(seed=0)
    env.step(0)                             # band 0 seen early, then abandoned
    for _ in range(200):
        env.step(18)                        # burn the clock elsewhere

    # The array is written before the dwell it prices, the observation after it,
    # so read the array at the moment of choice and compare against what the
    # observation said one step earlier.
    for band in (0, 9):                     # long-abandoned, never-visited
        before = env._observation()[2 * K.N_BANDS + band]
        env.step(band)
        assert env._staleness_array[band] == pytest.approx(before)

    # Never visited is maximally stale, not minimally -- the inverted version
    # read -1/600 here, because the element still held the sentinel `_last_slot`
    # of -1 divided by the episode length. Since D55 "maximally stale" is the
    # episode measured in reference sweeps, N_SLOTS / SWEEP_SLOTS, not 1.0.
    assert env._staleness_array[9] == pytest.approx(K.N_SLOTS / 43)

    # And the ordering the exploration term depends on: a band abandoned 200
    # slots ago must read staler than the one just left. A fresh episode,
    # because the loop above has just re-visited band 0.
    env = ScanEnv(pool=EmitterPool.from_train(), reward="reward_balance")
    env.reset(seed=0)
    env.step(0)
    for _ in range(200):
        env.step(18)
    env.step(0)                             # re-price band 0 after the neglect
    env.step(18)                            # re-price band 18, just left
    assert env._staleness_array[0] > env._staleness_array[18]


def test_reward_balance_ranks_a_sweep_above_a_two_band_pingpong():
    """The hole that `-1.0 * camp_slots` left open.

    `camp_slots` resets the instant the action changes, so alternating between
    two bands paid exactly what a full sweep paid. Measured over three sampled
    scenarios, that version scored a 2-band ping-pong (coverage 0.261, censored
    intercept time 16.55 s) above round-robin (coverage 0.921, 2.47 s) -- it
    ranked the failure mode D14 exists to demonstrate above the floor the ladder
    is built on. Charging against `visit_density` closes it, because alternating
    holds density near 0.5 where a sweep holds it near 1/36.

    Asserted as an ordering, not against literals: the numbers move with the
    scenario draw, the ordering is the property.
    """
    pool = EmitterPool.from_train()

    def total(pick):
        rewards = []
        for seed in (0, 1, 2):
            env = ScanEnv(pool=pool, reward="reward_balance")
            env.reset(seed=seed)
            step = 0
            while True:
                _, _, terminated, _, _ = env.step(pick(step))
                step += 1
                if terminated:
                    break
            rewards.append(env.total_reward)
        return float(np.mean(rewards))

    sweep = total(lambda s: s % K.N_BANDS)
    pingpong = total(lambda s: 18 + s % 2)
    camp = total(lambda s: 18)

    assert sweep > pingpong, "a ping-pong must not out-score a full sweep"
    assert pingpong > camp, "camping must still be the worst of the three"
    assert sweep > 0 > camp


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
    # staleness is maximal everywhere (nothing seen) -- since D55 that ceiling is
    # the episode's worth of sweeps, 600/43, not 1.0. Hit rate and density are 0,
    # current_band is a one-hot on band 0 (reset()'s initial _current_band), and
    # the clock and measured_dbm scalars are 0 -- the last because reset() sets
    # _last_measured_dbm to the clamp floor (D20: cold start).
    assert (first[:36] == 0).all() and (first[36:72] == 0).all()
    assert first[72:108] == pytest.approx(600 / 43, rel=1e-6)
    assert first[108] == 1 and (first[109:144] == 0).all()
    assert first[144] == 0 and first[145] == 0


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


def test_the_greedy_explore_axis_has_the_shape_it_claims():
    """D57's three candidates are an axis, not three guesses, and this pins the
    axis rather than any coefficient.

    Three properties, each of which is what makes a corner a corner:

      1. `greedy` prefers camping to sweeping. It has no exploration term and no
         camping cost by design, so a policy that parks on one band must
         outscore round-robin. If this ever fails, the "pure exploit" corner has
         quietly acquired an explore term and the axis no longer spans anything.
      2. `explore` prefers sweeping to camping, by a wide margin.
      3. `make_reward_weighted` interpolates between them: alpha = 0 reproduces
         `explore` exactly, and alpha = 1 reproduces `greedy` up to the positive
         constant `_GREEDY_GAIN` (which cannot reorder policies).

    Asserted as orderings and as an exact-endpoint identity, never against
    literals -- the totals move with the scenario draw, and pinning them would
    make this a change-detector rather than a test.
    """
    sc = Scenario.replay("config_921", "stare")

    def total(reward_name=None, fn=None, policy="camp"):
        env = ScanEnv(scenario=sc, reward=reward_name or DEFAULT_REWARD)
        if fn is not None:
            env._reward_fn = fn
        env.reset(seed=0)
        step = 0
        while True:
            action = 18 if policy == "camp" else step % K.N_BANDS
            _, _, terminated, _, _ = env.step(action)
            step += 1
            if terminated:
                return env.total_reward

    # 1. the exploit corner prefers the exploit
    assert total("greedy", policy="camp") > total("greedy", policy="sweep")

    # 2. the explore corner prefers the sweep
    assert total("explore", policy="sweep") > total("explore", policy="camp")

    # 3. the blend's endpoints are its corners
    at_zero = make_reward_weighted(0.0)
    assert total(fn=at_zero, policy="sweep") == pytest.approx(
        total("explore", policy="sweep"))
    assert total(fn=at_zero, policy="camp") == pytest.approx(
        total("explore", policy="camp"))

    at_one = make_reward_weighted(1.0)
    gain_sweep = total(fn=at_one, policy="sweep") / total("greedy", policy="sweep")
    gain_camp = total(fn=at_one, policy="camp") / total("greedy", policy="camp")
    assert gain_sweep == pytest.approx(gain_camp)      # a constant, not a reshaping
    assert gain_sweep > 0                              # positive: ordering preserved

    with pytest.raises(ValueError):
        make_reward_weighted(1.5)


def test_the_registered_blend_outranks_the_degenerate_policies():
    """D57's registered alpha sits where the reward still orders the ladder.

    The blend has a valid window and `WEIGHTED_ALPHA` is inside it: below it the
    explore half dominates and round-robin outscores rung 5, above it a 2-band
    ping-pong outscores round-robin -- D53's failure mode, reintroduced through
    the greedy half. This asserts the half that would silently un-fix D53: both
    sweeping policies must beat both degenerate ones.

    Sampled scenarios rather than one replay, because the window was measured
    that way and a single scenario is a much noisier draw.
    """
    def total(policy):
        out = []
        for seed in (0, 1, 2):
            env = ScanEnv(pool=EmitterPool.from_train(), reward="weighted")
            env.reset(seed=seed)
            step = 0
            while True:
                if policy == "sweep":
                    action = step % K.N_BANDS
                elif policy == "alternate":
                    action = 18 if step % 2 else 19
                else:
                    action = 18
                _, _, terminated, _, _ = env.step(action)
                step += 1
                if terminated:
                    out.append(env.total_reward)
                    break
        return sum(out) / len(out)

    sweep = total("sweep")
    assert sweep > total("alternate"), "a 2-band ping-pong must not outscore a sweep (D53)"
    assert sweep > total("camp"), "camping must not outscore a sweep (D14)"
