"""Rungs 7 (DQN), 8 (PPO) and 9 (RecurrentPPO) plumbing: env factory, training
entry point, policy adapter, checkpoint I/O.

Untrained models are used throughout except small end-to-end train() smoke
tests. DQN/PPO("MlpPolicy", env=...) initialise real weights on construction, so
.predict() already returns valid, typed, legal actions without calling .learn() --
sufficient for testing legality/typing/guarding, not learned behaviour. Same
logic for RecurrentPPO("MlpLstmPolicy", ...).

The generic tests (env factory, RLScheduler legality/typing/guarding) exercise
DQN's `untrained_model` only, deliberately -- `RLScheduler`/`make_train_env` are
algorithm-agnostic (rfenv/rl/common.py), so covering them once and reusing the
same coverage for PPO via the ladder's own generic per-rung tests
(tests/test_baselines.py) is the point, not a gap. What's PPO-specific and
therefore tested separately here is `rl.ppo`'s own `train()`/`load_checkpoint()`.
RecurrentPPO gets its own section too, since `RecurrentRLScheduler` is a
genuinely different adapter (stateful), not just a different algorithm.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3 import DQN, PPO

sb3_contrib = pytest.importorskip("sb3_contrib")
from sb3_contrib import RecurrentPPO

from rfenv import baselines as B
from rfenv import rl
from rfenv.rl import common, dqn, ppo, recurrent_ppo
from rfenv.rl.common import RecurrentRLScheduler
from rfenv.env import DEFAULT_REWARD, ScanEnv
from rfenv.rollout import run_episode
from rfenv.scenario import Scenario

CONFIG_ID = "config_81"   # 1 detectable emitter -- the cheap/fast case

# Derived, never hardcoded: the observation has changed width four times
# (109 -> 145 -> 146 -> 147 -> 146 -> 183, D34/D49/D55/D67) and every literal in
# this file had to be chased down each time.
OBS_WIDTH = common.current_observation_width()


def _key_for_rung(rung_number: str) -> str:
    """The registered key for a rung number, not a hardcoded string.

    Checkpoint-tagged rung keys (e.g. "deep_q_network_z_60k") churn as new
    variants get trained (D-line, this session already renamed twice) -- the
    rung *number* is the stable identity, so look up by that instead of
    hardcoding a key that will go stale again.

    **Matches the lettered family, not just the bare number.** An algorithm's
    rungs are numbered `9`, `9a`, `9b`, ... and which of them exists depends on
    what has been trained: when the pre-D55 RecurrentPPO family was retired the
    only bare `9` went with it, leaving `9a`..`9d`. These tests are asking "the
    RecurrentPPO rung", not "rung 9 exactly", so an exact match made them fail on
    a change that was not about them. Falls back to the first lettered variant,
    in ladder order.
    """
    exact = [r.key for r in B.LADDER if r.rung == rung_number]
    if exact:
        return exact[0]
    family = [r.key for r in B.LADDER if r.rung.startswith(rung_number)]
    if not family:
        raise AssertionError(
            f"no rung numbered {rung_number!r} or {rung_number!r}<letter> in the ladder"
        )
    return family[0]


@pytest.fixture(scope="module")
def scenario():
    return Scenario.replay(CONFIG_ID, "stare")


@pytest.fixture(scope="module")
def untrained_model():
    env = rl.make_train_env(reward=DEFAULT_REWARD)
    return DQN("MlpPolicy", env, seed=0)


def test_make_train_env_builds_a_valid_scan_env():
    from gymnasium.utils.env_checker import check_env
    env = rl.make_train_env()
    assert env.observation_space.shape == (OBS_WIDTH,)
    assert env.action_space.n == 36
    check_env(env, skip_render_check=True)


def test_make_train_env_threads_reward_through():
    """Any registered candidate, not a hardcoded one: candidate 3 has been
    renamed more than once (and `hit_z`/`hit_y` were retired entirely after
    D62), so pinning a specific name here only re-breaks this test each time.
    `greedy` is picked because it is not the default, so threading is what is
    being observed rather than the default leaking through."""
    env = rl.make_train_env(reward="greedy")
    assert env.reward_name == "greedy"
    with pytest.raises(ValueError):
        rl.make_train_env(reward="not_a_reward")


def test_rl_scheduler_plays_a_legal_episode(scenario, untrained_model):
    env = ScanEnv(scenario=scenario)
    policy = rl.RLScheduler(untrained_model)
    run_episode(env, policy, seed=0)
    bands = np.array([row["band"] for row in env.log], dtype=np.int64)
    assert len(bands) == 600
    assert bands.min() >= 0 and bands.max() < 36
    assert 300 <= env.n_steps <= 600


def test_rl_scheduler_returns_plain_python_int(scenario, untrained_model):
    env = ScanEnv(scenario=scenario)
    policy = rl.RLScheduler(untrained_model)
    obs, info = env.reset(seed=0)
    for _ in range(20):
        action = policy(obs, info)
        assert isinstance(action, int), type(action).__name__
        assert env.action_space.contains(action)
        obs, _, terminated, _, info = env.step(action)
        if terminated:
            break


def test_rl_scheduler_ignores_info(untrained_model):
    obs = np.zeros(OBS_WIDTH, dtype=np.float32)
    action = rl.RLScheduler(untrained_model)(obs, {})
    assert isinstance(action, int)
    assert 0 <= action < 36


def test_deep_q_network_is_registered_in_the_ladder():
    spec = B.BY_KEY[_key_for_rung("7")]
    assert spec.rung == "7"
    assert spec.deployable is True
    assert spec.needs_grid is False


def test_deep_q_network_rung_never_sees_truth(monkeypatch, scenario, untrained_model):
    # Patch rfenv.rl.dqn (where ladder.py's factory actually imports
    # load_checkpoint from), not the rfenv.rl package re-export -- patching the
    # re-export doesn't reach ladder.py's `from rfenv.rl.dqn import load_checkpoint`,
    # so this silently fell through to the real on-disk checkpoint before.
    monkeypatch.setattr(dqn, "load_checkpoint", lambda *a, **k: untrained_model)
    env = ScanEnv(scenario=scenario)
    policy = B.make(_key_for_rung("7"), seed=0)
    seen: set[str] = set()
    obs, info = env.reset(seed=0)
    terminated = False
    while not terminated:
        seen.update(B.restrict(info))
        obs, _, terminated, _, info = env.step(int(policy(obs, info)))
    assert seen <= B.OBSERVABLE_INFO


def test_load_checkpoint_missing_file_raises_actionable_error(tmp_path):
    missing = tmp_path / "nope.zip"
    with pytest.raises(FileNotFoundError, match="rfenv.rl"):
        rl.load_checkpoint(missing)


def test_checkpoint_round_trips(tmp_path, untrained_model):
    path = tmp_path / "m.zip"
    untrained_model.save(path)
    loaded = rl.load_checkpoint(path)
    obs = np.zeros(OBS_WIDTH, dtype=np.float32)
    a1, _ = untrained_model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert int(a1) == int(a2)


def test_train_produces_a_loadable_checkpoint(tmp_path):
    path = tmp_path / "ckpt.zip"
    rl.train(reward=DEFAULT_REWARD, total_timesteps=10, seed=0, checkpoint=path)
    assert path.exists()
    model = rl.load_checkpoint(path)
    action, _ = model.predict(np.zeros(OBS_WIDTH, dtype=np.float32), deterministic=True)
    assert 0 <= int(action) < 36


def test_recurrent_ppo_train_threads_a_custom_pool_through(tmp_path):
    """`pool=` lets a caller train entirely on something other than the real
    training pool -- `online.py`'s `fine_tune` already supports this for
    adapting an *existing* checkpoint; this is the same idea for training a
    fresh one. `None` (the default) must still fall through to
    `make_train_env`'s own default (`split.training_pool()`), unchanged."""
    from rfenv.scenario import EmitterContribution, EmitterPool

    cells = np.array([[5, s] for s in range(500)], dtype=np.int16)
    contrib = EmitterContribution(
        config_id="t", source="synthetic", label=0, cells=cells,
        peak_dbm=np.full(500, -80.0, dtype=np.float32),
        n_pulses=np.full(500, 4, dtype=np.int32), total_pulses=2000,
    )
    pool = EmitterPool(contributions=[contrib], emitter_counts=np.array([1]))

    model = recurrent_ppo.train(
        reward=DEFAULT_REWARD, pool=pool, total_timesteps=200, seed=0,
        checkpoint=tmp_path / "m.zip", verbose=0,
        hyperparameters={"n_steps": 64, "batch_size": 16},
    )
    inner = model.env.envs[0].unwrapped
    assert inner._pool is pool


def test_print_episode_metrics_prints_a_real_episode(tmp_path, capsys):
    """`print_episode_metrics=True` prints something -- and specifically the
    episode that finished, not a fresh, all-zero one (the exact bug
    EpisodeMetricsCallback exists to avoid; see its docstring). 700 timesteps
    is enough to guarantee DQN completes at least one full episode (300-600
    steps) within the budget.
    """
    path = tmp_path / "ckpt.zip"
    dqn.train(reward=DEFAULT_REWARD, total_timesteps=700, seed=0, checkpoint=path,
              print_episode_metrics=True)
    out = capsys.readouterr().out
    assert "n_steps" in out
    assert "'n_steps': 0" not in out


def test_checkpoint_freq_saves_intermediate_checkpoints(tmp_path):
    """checkpoint_freq=N writes a loadable snapshot every N steps, named
    <run>_s<k>.zip in the final checkpoint's own directory, in addition to
    (not instead of) the final checkpoint train() always writes.

    Named by run and ordinal rather than by timestep count deliberately: a
    count in a filename gets copied into a rung key by hand and then drifts
    from the archive, which has already happened twice. The count lives in the
    manifest, where it is read rather than transcribed.
    """
    path = tmp_path / "deep_q_network.zip"
    dqn.train(reward=DEFAULT_REWARD, total_timesteps=1000, seed=0, checkpoint=path,
              checkpoint_freq=300)
    assert path.exists()
    intermediate = tmp_path / "deep_q_network_s1.zip"
    assert intermediate.exists()
    model = dqn.load_checkpoint(intermediate)
    action, _ = model.predict(np.zeros(OBS_WIDTH, dtype=np.float32), deterministic=True)
    assert 0 <= int(action) < 36


# --------------------------------------------------------------------------- #
# The checkpoint manifest (common.py) -- every .zip gets a .json beside it
# --------------------------------------------------------------------------- #

def test_training_writes_a_manifest_beside_every_checkpoint(tmp_path):
    """Both the final checkpoint and every intermediate snapshot get a sidecar,
    recording what the archive itself cannot: the reward, the argv, the commit,
    and the observation width the env had at training time.
    """
    path = tmp_path / "run_a.zip"
    dqn.train(reward=DEFAULT_REWARD, total_timesteps=1000, seed=0, checkpoint=path,
              checkpoint_freq=300, description="a test run",
              hyperparameters={"learning_rate": 5e-4})

    for zipped in (path, tmp_path / "run_a_s1.zip"):
        manifest = common.read_manifest(zipped)
        assert manifest is not None, f"no manifest for {zipped.name}"
        assert manifest["run"] == "run_a"
        assert manifest["reward"] == DEFAULT_REWARD
        assert manifest["seed"] == 0
        assert manifest["observation_width"] == common.current_observation_width()
        assert manifest["algorithm"] == "DQN"
        assert manifest["argv"]
        assert "learning_rate" in manifest["hyperparameters"]
        # what was *passed*, not merely what the default happened to be
        assert manifest["hyperparameters"]["learning_rate"] == 5e-4
        assert manifest["resolved"]["learning_rate"] == 5e-4
        assert manifest["description"] == "a test run"


def test_load_checkpoint_refuses_a_checkpoint_of_the_wrong_observation_width(tmp_path):
    """The failure this whole sidecar exists to remove.

    Without the manifest a stale checkpoint loads perfectly happily -- `.load()`
    binds no env and so checks nothing -- and only fails much later inside
    `predict()`, with a shape error naming neither the checkpoint nor the cause
    (D49: an observation change invalidates every checkpoint that predates it).
    """
    path = tmp_path / "run_b.zip"
    dqn.train(reward=DEFAULT_REWARD, total_timesteps=300, seed=0, checkpoint=path)

    manifest = common.read_manifest(path)
    manifest["observation_width"] = manifest["observation_width"] - 2
    common.manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="wide observation"):
        dqn.load_checkpoint(path)
    # the error has to carry the command that rebuilds it, or it is not actionable
    with pytest.raises(ValueError, match=re.escape(manifest["command"])):
        dqn.load_checkpoint(path)


def test_a_checkpoint_without_a_manifest_still_loads(tmp_path):
    """Every checkpoint trained before manifests existed has no sidecar.
    Refusing those would break every rung already registered, so a missing
    manifest is allowed through -- it only means the width check cannot run.
    """
    path = tmp_path / "run_c.zip"
    dqn.train(reward=DEFAULT_REWARD, total_timesteps=300, seed=0, checkpoint=path)
    common.manifest_path(path).unlink()
    assert dqn.load_checkpoint(path) is not None


def test_parse_hyperparameters_types_values_rather_than_passing_strings():
    """SB3 type-checks several constructor arguments, and a string "512" fails
    unhelpfully deep inside them -- so KEY=VALUE goes through json.loads."""
    parsed = common.parse_hyperparameters(["ent_coef=0.01", "n_steps=512",
                                              "device=cpu"])
    assert parsed == {"ent_coef": 0.01, "n_steps": 512, "device": "cpu"}
    assert isinstance(parsed["n_steps"], int)

    with pytest.raises(ValueError, match="KEY=VALUE"):
        common.parse_hyperparameters(["ent_coef"])


# --------------------------------------------------------------------------- #
# PPO (rung 8) -- its own train()/load_checkpoint(), same shape as DQN's above
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ppo_untrained_model():
    env = rl.make_train_env(reward=DEFAULT_REWARD)
    return PPO("MlpPolicy", env, seed=0)


def test_ppo_rung_is_registered_in_the_ladder():
    spec = B.BY_KEY[_key_for_rung("8")]
    assert spec.rung == "8"
    assert spec.deployable is True
    assert spec.needs_grid is False


def test_ppo_rung_never_sees_truth(monkeypatch, scenario, ppo_untrained_model):
    monkeypatch.setattr(ppo, "load_checkpoint", lambda *a, **k: ppo_untrained_model)
    env = ScanEnv(scenario=scenario)
    policy = B.make(_key_for_rung("8"), seed=0)
    seen: set[str] = set()
    obs, info = env.reset(seed=0)
    terminated = False
    while not terminated:
        seen.update(B.restrict(info))
        obs, _, terminated, _, info = env.step(int(policy(obs, info)))
    assert seen <= B.OBSERVABLE_INFO


def test_ppo_load_checkpoint_missing_file_raises_actionable_error(tmp_path):
    missing = tmp_path / "nope.zip"
    with pytest.raises(FileNotFoundError, match="rfenv.rl.ppo"):
        ppo.load_checkpoint(missing)


def test_ppo_checkpoint_round_trips(tmp_path, ppo_untrained_model):
    path = tmp_path / "m.zip"
    ppo_untrained_model.save(path)
    loaded = ppo.load_checkpoint(path)
    obs = np.zeros(OBS_WIDTH, dtype=np.float32)
    a1, _ = ppo_untrained_model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert int(a1) == int(a2)


def test_ppo_train_produces_a_loadable_checkpoint(tmp_path):
    path = tmp_path / "ckpt.zip"
    ppo.train(reward=DEFAULT_REWARD, total_timesteps=64, seed=0, checkpoint=path)
    assert path.exists()
    model = ppo.load_checkpoint(path)
    action, _ = model.predict(np.zeros(OBS_WIDTH, dtype=np.float32), deterministic=True)
    assert 0 <= int(action) < 36


# --------------------------------------------------------------------------- #
# RecurrentPPO (rung 9) -- its own train()/load_checkpoint(), plus the extra
# state/episode_start plumbing RecurrentRLScheduler carries that RLScheduler
# doesn't need
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def recurrent_untrained_model():
    env = rl.make_train_env(reward=DEFAULT_REWARD)
    return RecurrentPPO("MlpLstmPolicy", env, seed=0)


def test_recurrent_ppo_rung_is_registered_in_the_ladder():
    spec = B.BY_KEY[_key_for_rung("9")]
    # `9` or a lettered variant of it -- which exists depends on what has been
    # trained, and the bare number went when the pre-D55 family was retired.
    assert spec.rung.startswith("9")
    assert spec.deployable is True
    assert spec.needs_grid is False


def test_recurrent_ppo_rung_never_sees_truth(monkeypatch, scenario, recurrent_untrained_model):
    monkeypatch.setattr(recurrent_ppo, "load_checkpoint", lambda *a, **k: recurrent_untrained_model)
    env = ScanEnv(scenario=scenario)
    policy = B.make(_key_for_rung("9"), seed=0)
    seen: set[str] = set()
    obs, info = env.reset(seed=0)
    terminated = False
    while not terminated:
        seen.update(B.restrict(info))
        obs, _, terminated, _, info = env.step(int(policy(obs, info)))
    assert seen <= B.OBSERVABLE_INFO


def test_recurrent_ppo_load_checkpoint_missing_file_raises_actionable_error(tmp_path):
    missing = tmp_path / "nope.zip"
    with pytest.raises(FileNotFoundError, match="rfenv.rl.recurrent_ppo"):
        recurrent_ppo.load_checkpoint(missing)


def test_recurrent_ppo_checkpoint_round_trips(tmp_path, recurrent_untrained_model):
    path = tmp_path / "m.zip"
    recurrent_untrained_model.save(path)
    loaded = recurrent_ppo.load_checkpoint(path)
    obs = np.zeros(OBS_WIDTH, dtype=np.float32)
    episode_start = np.array([True])
    a1, _ = recurrent_untrained_model.predict(obs, state=None, episode_start=episode_start, deterministic=True)
    a2, _ = loaded.predict(obs, state=None, episode_start=episode_start, deterministic=True)
    assert int(a1) == int(a2)


def test_recurrent_ppo_train_produces_a_loadable_checkpoint(tmp_path):
    path = tmp_path / "ckpt.zip"
    recurrent_ppo.train(reward=DEFAULT_REWARD, total_timesteps=64, seed=0, checkpoint=path)
    assert path.exists()
    model = recurrent_ppo.load_checkpoint(path)
    action, _ = model.predict(
        np.zeros(OBS_WIDTH, dtype=np.float32), state=None, episode_start=np.array([True]), deterministic=True,
    )
    assert 0 <= int(action) < 36


def test_recurrent_rl_scheduler_carries_state_across_calls(scenario, recurrent_untrained_model):
    """The property that distinguishes this adapter from `RLScheduler`: the
    hidden state passed into call N+1 is whatever call N returned, not always
    `None` -- otherwise the LSTM would restart from zero every dwell and the
    "recurrent" part of RecurrentPPO would be doing nothing.
    """
    env = ScanEnv(scenario=scenario)
    policy = RecurrentRLScheduler(recurrent_untrained_model)
    obs, info = env.reset(seed=0)
    assert policy._lstm_state is None
    policy(obs, info)
    assert policy._lstm_state is not None
    first_state = policy._lstm_state
    obs, _, terminated, _, info = env.step(0)
    policy(obs, info)
    assert policy._lstm_state is not None
    # Not asserting the two states differ by value -- an untrained network's
    # LSTM update for one step can legitimately be tiny -- only that state is
    # being threaded through at all (was None, now consistently isn't).
    assert first_state[0].shape == policy._lstm_state[0].shape
