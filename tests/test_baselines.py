"""The baseline ladder: is every rung legal, reproducible, and actually distinct?

Four things are checked, and the third is the one this file exists for.

**Legal.** Every rung returns a band in 0..35 at every step of a full episode, on
scenarios spanning the whole difficulty range. An illegal action raises inside
`ScanEnv`, so these are end-to-end runs rather than spot checks.

**Reproducible.** `EVALUATION.md` §7 compares schedulers "on identical scenarios
and seeds", which is only meaningful if a seed reproduces an episode exactly --
policy randomness included, not just the receiver's noise.

**Distinct.** D36 found that §5's rungs 2 and 3 were the same policy and had been
reproducing each other to four decimal places. D43 separated them, and the tests
below pin the separation as a mechanism -- equal airtime against Turing's 2:1
weighting -- not as a difference in output that a later edit could quietly undo.

**Deployable.** A scheduler may read its own scan history and `Y`, and nothing
else (D19, D20, D29). `baselines.guarded` enforces it by *removing* the truth keys
rather than by asking politely, and `test_a_rung_that_reaches_for_truth_fails`
proves the enforcement is live.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfenv import baselines as B
from rfenv import constants as K
from rfenv.env import ScanEnv
from rfenv.rollout import run_episode
from rfenv.scenario import Scenario
from rfenv.truth import TruthGrid



def _unusable_checkpoint(key: str, grid) -> str | None:
    """Why this rung cannot be built here, or None if it can.

    **Builds it directly rather than pre-checking a hand-maintained path
    table.** Every RL-backed factory's `load_checkpoint()` already calls
    `require_loadable()` before deserialising -- a cheap manifest/archive read,
    not a full model load -- so `B.make()` already raises exactly the
    informative error this needs: D49's `ValueError` for an observation-width
    mismatch, `FileNotFoundError` for a missing checkpoint. A hand-maintained
    `_UNTRAINED_RUNG_CHECKPOINTS` table lived here before and reliably fell
    behind -- it covered rungs 7-9d and was never extended as 9e-9j/10a-d/11a-d
    were registered, so the moment an observation change (D67) made every
    checkpoint stale at once, those newer rungs' tests failed outright with a
    bare shape error instead of skipping cleanly (measured: 85 failures the
    first time). A real `grid` must be passed so a rung that needs one
    (`needs_grid=True`) doesn't get misread as unusable for lacking it.
    """
    try:
        B.make(key, seed=0, grid=grid)
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)
    return None


def _skip_if_untrained(key: str, grid) -> None:
    """These generic per-rung tests check a property of ANY rung, not that a
    specific checkpoint exists -- that belongs to tests/test_rl.py, which
    builds its own untrained model and needs no file on disk."""
    reason = _unusable_checkpoint(key, grid)
    if reason is not None:
        pytest.skip(reason)

# The gate-4 extremes plus a middling one: 1 detectable emitter, 19, and 72.
# Stare replays, never scan (D36).
CONFIGS = ("config_81", "config_2", "config_921")


@pytest.fixture(scope="module")
def worlds():
    out = {}
    for config_id in CONFIGS:
        scenario = Scenario.replay(config_id, "stare")
        out[config_id] = (scenario, TruthGrid.from_scenario(scenario))
    return out


def bands_taken(key, scenario, grid, seed=0):
    """The band tuned at each of the 600 slots, for one rung on one episode."""
    env = ScanEnv(scenario=scenario)
    run_episode(env, B.make(key, seed=seed, grid=grid), seed=seed)
    return np.array([row["band"] for row in env.log], dtype=np.int64), env


# --------------------------------------------------------------------------- #
# Legal
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", [r.key for r in B.LADDER])
@pytest.mark.parametrize("config_id", CONFIGS)
def test_every_rung_plays_a_legal_episode(worlds, key, config_id):
    """A full episode with no exception, covering exactly 600 slots in 300-600 steps.

    `ScanEnv.step` rejects an out-of-range action itself, so reaching the end is
    the assertion; the counts restate D35, which is the property most easily broken
    by a policy that miscounts a two-slot dwell.
    """
    scenario, grid = worlds[config_id]
    _skip_if_untrained(key, grid)
    bands, env = bands_taken(key, scenario, grid)
    assert len(bands) == K.N_SLOTS
    assert bands.min() >= 0 and bands.max() < K.N_BANDS
    assert K.N_SLOTS // 2 <= env.n_steps <= K.N_SLOTS


@pytest.mark.parametrize("key", [r.key for r in B.LADDER])
def test_a_rung_never_names_a_band_that_is_not_an_action(worlds, key):
    """Actions are plain Python ints in the discrete space, not numpy scalars.

    `run_episode` casts, so a rung returning `np.int64` would pass every other
    test here and then fail wherever a policy is called directly -- including in
    whatever drives the RL rung.
    """
    scenario, grid = worlds["config_2"]
    _skip_if_untrained(key, grid)
    env = ScanEnv(scenario=scenario)
    policy = B.make(key, seed=0, grid=grid)
    obs, info = env.reset(seed=0)
    for _ in range(20):
        action = policy(obs, info)
        assert isinstance(action, int), f"{key} returned {type(action).__name__}"
        assert env.action_space.contains(action)
        obs, _, terminated, _, info = env.step(action)
        if terminated:
            break


# --------------------------------------------------------------------------- #
# Reproducible
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", [r.key for r in B.LADDER])
def test_the_same_seed_reproduces_the_episode_exactly(worlds, key):
    """Byte-identical band sequence and identical §4 row, twice, same seed.

    Exactly, not approximately: both the receiver's noise and the policy's own
    randomness run off the seed, so any tolerance here would be hiding a source of
    randomness that was never seeded.
    """
    scenario, grid = worlds["config_2"]
    _skip_if_untrained(key, grid)
    a, env_a = bands_taken(key, scenario, grid, seed=7)
    b, env_b = bands_taken(key, scenario, grid, seed=7)
    assert np.array_equal(a, b)
    assert env_a.episode_metrics() == env_b.episode_metrics()


@pytest.mark.parametrize("key", ["random", "recency", "apfeld", "apfeld_active_rfs"])
def test_a_stochastic_rung_actually_varies_with_the_seed(worlds, key):
    """The other half of determinism: a seeded rung must not ignore its seed.

    Without this, `make(..., seed=k)` could be dropping the seed entirely and
    every "reproducible" test above would still pass.
    """
    scenario, grid = worlds["config_2"]
    a, _ = bands_taken(key, scenario, grid, seed=0)
    b, _ = bands_taken(key, scenario, grid, seed=1)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("key", ["round_robin", "turing_sweep"])
def test_an_open_loop_rung_ignores_the_seed(worlds, key):
    """Rungs 2 and 3 are open loop: the schedule is fixed before the episode starts.

    Their metrics still move with the seed -- the receiver's noise does -- but the
    band sequence must not, or they are not open-loop schedules.
    """
    scenario, grid = worlds["config_2"]
    a, _ = bands_taken(key, scenario, grid, seed=0)
    b, _ = bands_taken(key, scenario, grid, seed=11)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Distinct -- D43
# --------------------------------------------------------------------------- #

def test_round_robin_and_the_turing_sweep_are_not_the_same_policy(worlds):
    """D36's finding, pinned. They differed by nothing until D43 separated them.

    Asserted on the band sequence rather than on a metric: two policies can score
    alike by luck, but if they tune the same band at every one of 600 slots they
    are one policy and the ladder has six rungs while claiming seven.
    """
    scenario, grid = worlds["config_2"]
    rr, _ = bands_taken("round_robin", scenario, grid)
    sweep, _ = bands_taken("turing_sweep", scenario, grid)
    assert not np.array_equal(rr, sweep)
    disagree = np.mean(rr != sweep)
    assert disagree > 0.5, f"only {disagree:.1%} of slots differ -- too close to be two rungs"


def test_round_robin_spends_equal_airtime_on_every_band():
    """D43's stated difference, checked on the cycle rather than on an episode.

    Equal airtime is the *definition* of rung 2, so it is checked against
    `DWELL_SLOTS` directly. An episode is 600 slots and 72 does not divide it, so
    a per-episode count would be equal only up to the truncated final cycle.
    """
    per_band = np.zeros(K.N_BANDS, dtype=np.int64)
    for band in B.EQUAL_AIRTIME_CYCLE:
        per_band[band] += int(K.DWELL_SLOTS[band])
    assert set(per_band.tolist()) == {2}
    assert B.EQUAL_AIRTIME_CYCLE_SLOTS == 72
    assert sorted(set(B.EQUAL_AIRTIME_CYCLE)) == list(range(K.N_BANDS))


def test_the_turing_sweep_spends_double_on_the_wide_bands():
    """The prior rung 2 refuses, measured: 2 slots per sweep against 1.

    This is the mechanism D43 says the two rungs differ by, so it is asserted on
    the frozen schedule rather than inferred from the scores it produces.
    """
    per_band = np.zeros(K.N_BANDS, dtype=np.int64)
    for start, end, band in K.dwell_schedule(K.SWEEP_S):
        s0, s1 = K.slots_for_dwell(start, end)
        per_band[int(band)] += s1 - s0
    wide = K.DWELL_SLOTS == 2
    assert set(per_band[wide].tolist()) == {2}
    assert set(per_band[~wide].tolist()) == {1}


def test_the_turing_sweep_rung_replays_the_frozen_schedule(worlds):
    """Rung 3 *is* `constants.dwell_schedule()`, slot for slot.

    §5 keeps it as its own rung because it makes our numbers comparable to the
    recordings, and that only holds if it is the recordings' own schedule.
    """
    scenario, grid = worlds["config_2"]
    taken, _ = bands_taken("turing_sweep", scenario, grid)
    expected = np.zeros(K.N_SLOTS, dtype=np.int64)
    for start, end, band in K.dwell_schedule():
        s0, s1 = K.slots_for_dwell(start, end)
        expected[s0:s1] = int(band)
    assert np.array_equal(taken, expected)


# --------------------------------------------------------------------------- #
# Deployable
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", B.SCHEDULERS)
def test_a_scheduler_never_sees_truth(worlds, key):
    """`guarded` hands the rung a stripped `info`, and the rung still works.

    The truth keys are not merely unused -- they are absent. That is the
    difference between a convention and an enforcement, and D29's asymmetry
    (reward may read truth, observation may not) is only credible as the latter.
    """
    scenario, grid = worlds["config_2"]
    _skip_if_untrained(key, grid)
    env = ScanEnv(scenario=scenario)
    seen: set[str] = set()
    policy = B.make(key, seed=0, grid=grid)

    obs, info = env.reset(seed=0)
    terminated = False
    while not terminated:
        seen.update(B.restrict(info))
        obs, _, terminated, _, info = env.step(int(policy(obs, info)))
    assert seen <= B.OBSERVABLE_INFO
    for forbidden in ("Z", "pulses", "newly_intercepted", "n_detectable", "total_pulses"):
        assert forbidden not in B.OBSERVABLE_INFO


def test_a_rung_that_reaches_for_truth_fails(worlds):
    """The guard is live: a policy asking for `Z` raises rather than scoring well."""
    scenario, _ = worlds["config_2"]
    env = ScanEnv(scenario=scenario)
    cheat = B.guarded(lambda obs, info: int(np.argmax(info["Z"])))
    obs, info = env.reset(seed=0)
    obs, _, _, _, info = env.step(0)
    with pytest.raises(KeyError):
        cheat(obs, info)


def test_the_observation_slices_match_the_environment(worlds):
    """`baselines.HIT_RATE` and friends still describe what `env` builds (D34).

    A silent reordering of the 183-vector would leave every rung running and every
    other test passing while rung 5 optimised staleness as though it were hit rate.
    The units matter as much as the order since D55: rung 5 reads staleness as a
    sweep count and would silently collapse into rung 4 if it arrived as an
    episode fraction again.
    """
    scenario, _ = worlds["config_2"]
    env = ScanEnv(scenario=scenario)
    obs, info = env.reset(seed=0)
    for _ in range(120):
        obs, _, terminated, _, info = env.step(6)
        if terminated:
            break

    hit_rate = np.asarray(obs[B.HIT_RATE], dtype=np.float64)
    visit = np.asarray(obs[B.VISIT_DENSITY], dtype=np.float64)
    stale = np.asarray(obs[B.STALENESS], dtype=np.float64)
    current = np.asarray(obs[B.CURRENT_BAND], dtype=np.float64)
    streak = np.asarray(obs[B.HIT_STREAK], dtype=np.float64)

    slot = info["slot"]
    assert obs[B.CLOCK] == pytest.approx(slot / K.N_SLOTS, abs=1e-6)
    # Only band 6 was ever looked at, so it holds all the airtime: in fair shares
    # (D55) that is N_BANDS, not 1.0 -- one band with all 36 bands' worth.
    assert visit[6] == pytest.approx(float(K.N_BANDS), abs=1e-4)
    assert visit[np.arange(K.N_BANDS) != 6].sum() == pytest.approx(0.0, abs=1e-6)
    # Never visited reads maximally stale: the episode in sweeps (D55), not 1.0.
    assert stale[0] == pytest.approx(K.N_SLOTS / 43, rel=1e-5)
    assert stale[6] < 0.25                         # under a quarter of a sweep old
    assert 0.0 <= hit_rate[6] <= 1.0
    assert hit_rate[np.arange(K.N_BANDS) != 6].sum() == pytest.approx(0.0)
    assert current[6] == pytest.approx(1.0)
    assert current.sum() == pytest.approx(1.0)     # one-hot: exactly one band current
    # `CAMP_TIME` was asserted here against the clock until D55 dropped it from
    # the vector: camped on band 6 since t=0, the streak equalled elapsed time.
    # visit_density[6] above now carries that, and carries it for every band.
    assert 0.0 <= obs[B.MEASURED_DBM] <= 1.0
    assert not hasattr(B, "CAMP_TIME"), "D55 removed camp_time from the observation"
    # hit_streak (D67): only band 6 was ever visited, so every other band's
    # streak is still its reset value of 0 -- and current_hit_streak, which
    # reads whichever band current_band is one-hot on, must agree with
    # hit_streak[6] exactly, since band 6 is the current band throughout.
    assert 0.0 <= streak[6] <= 1.0
    assert streak[np.arange(K.N_BANDS) != 6].sum() == pytest.approx(0.0)
    assert obs[B.CURRENT_HIT_STREAK] == pytest.approx(streak[6])


# --------------------------------------------------------------------------- #
# Each rung does the thing it is in the ladder to do
# --------------------------------------------------------------------------- #

def test_the_camper_camps(worlds):
    """Rung 4 probes, then never moves again -- the behaviour D14 measured."""
    scenario, grid = worlds["config_921"]
    bands, _ = bands_taken("camper", scenario, grid)
    tail = bands[3 * int(K.DWELL_SLOTS.sum()):]
    assert len(set(tail.tolist())) == 1, "a camper that drifts is rung 5, not rung 4"


def test_the_camper_trap_reproduces(worlds):
    """§4's two traps, in code: the camper wins ratio and loses intercept time.

    Run against `camper_oracle`, which is D14's own truth-fed camper, because that
    is the policy behind §5's quantified target. If this ever fails, either the
    metric definitions moved or the dataset's dominance property did, and both are
    much bigger news than a failing test.
    """
    ratio_wins = tti_losses = 0
    for config_id in CONFIGS:
        scenario, grid = worlds[config_id]
        _, camp = bands_taken("camper_oracle", scenario, grid)
        _, rr = bands_taken("round_robin", scenario, grid)
        c, r = camp.episode_metrics(), rr.episode_metrics()
        ratio_wins += c["interception_ratio"] > r["interception_ratio"]
        tti_losses += (c["censored_mean_intercept_time_s"]
                       > r["censored_mean_intercept_time_s"])
    assert ratio_wins == len(CONFIGS), "the camper should win interception ratio"
    assert tti_losses >= len(CONFIGS) - 1, "the camper should lose censored intercept time"


def test_the_oracle_captures_more_pulses_than_any_scheduler(worlds):
    """The ceiling line is above the ladder on the axis it is a ceiling for.

    Only on interception ratio, and deliberately only there: it is greedy on
    illuminations per slot, so it is not a ceiling for intercept time and §5 is
    careful never to call it one. `test_the_oracle_is_not_a_ceiling_for_intercept_time`
    is the other half of that claim.
    """
    scenario, grid = worlds["config_921"]
    _, oracle = bands_taken("oracle_pulse", scenario, grid)
    best = oracle.episode_metrics()["interception_ratio"]
    for key in B.SCHEDULERS:
        if _unusable_checkpoint(key, grid) is not None:
            continue
        _, env = bands_taken(key, scenario, grid)
        assert env.episode_metrics()["interception_ratio"] <= best + 1e-12, key


def test_the_oracle_is_not_a_ceiling_for_intercept_time(worlds):
    """Measured: the pulse oracle loses censored intercept time to round-robin.

    D14's note -- "each column has a different optimum, which is itself the point"
    -- as an assertion. A reader who takes `oracle_pulse` for *the* ceiling will
    read the whole comparison wrong.
    """
    scenario, grid = worlds["config_921"]
    _, oracle = bands_taken("oracle_pulse", scenario, grid)
    _, rr = bands_taken("round_robin", scenario, grid)
    assert (oracle.episode_metrics()["censored_mean_intercept_time_s"]
            > rr.episode_metrics()["censored_mean_intercept_time_s"])


# --------------------------------------------------------------------------- #
# The ladder itself
# --------------------------------------------------------------------------- #

def test_the_ladder_is_well_formed():
    keys = [r.key for r in B.LADDER]
    assert len(keys) == len(set(keys))
    assert set(B.SCHEDULERS) == {r.key for r in B.LADDER if r.deployable}
    assert "oracle_pulse" not in B.SCHEDULERS and "camper_oracle" not in B.SCHEDULERS
    for rung in B.LADDER:
        assert rung.purpose and rung.label


def test_a_reference_line_refuses_to_be_built_without_the_grid():
    """The oracles need truth, and saying so is better than silently degrading."""
    for key in ("oracle_pulse", "camper_oracle"):
        with pytest.raises(ValueError, match="needs the truth grid"):
            B.make(key, seed=0)


def test_an_unknown_rung_is_an_error():
    with pytest.raises(KeyError):
        B.make("not_a_rung", seed=0)


def test_rungs_do_not_share_a_random_stream(worlds):
    """Two rungs in the same episode draw independently.

    Otherwise a rung's behaviour would depend on where it sits in the ladder,
    and reordering `LADDER` would silently change the results table.
    """
    scenario, grid = worlds["config_2"]
    a, _ = bands_taken("random", scenario, grid, seed=3)
    b, _ = bands_taken("recency", scenario, grid, seed=3)
    assert not np.array_equal(a, b)
