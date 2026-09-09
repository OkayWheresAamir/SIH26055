"""Rungs 7 (DQN) and 8 (PPO) plumbing: env factory, training entry point,
policy adapter, checkpoint I/O.

Untrained models are used throughout except two small end-to-end train() smoke
tests. DQN/PPO("MlpPolicy", env=...) initialise real weights on construction, so
.predict() already returns valid, typed, legal actions without calling .learn() --
sufficient for testing legality/typing/guarding, not learned behaviour.

The generic tests (env factory, RLScheduler legality/typing/guarding) exercise
DQN's `untrained_model` only, deliberately -- `RLScheduler`/`make_train_env` are
algorithm-agnostic (rfenv/rl/common.py), so covering them once and reusing the
same coverage for PPO via the ladder's own generic per-rung tests
(tests/test_baselines.py) is the point, not a gap. What's PPO-specific and
therefore tested separately here is `rl.ppo`'s own `train()`/`load_checkpoint()`.
"""
from __future__ import annotations

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3 import DQN, PPO

from rfenv import baselines as B
from rfenv import rl
from rfenv.rl import dqn, ppo
from rfenv.env import ScanEnv
from rfenv.rollout import run_episode
from rfenv.scenario import Scenario

CONFIG_ID = "config_81"   # 1 detectable emitter -- the cheap/fast case


def _key_for_rung(rung_number: str) -> str:
    """The registered key for a rung number, not a hardcoded string.

    Checkpoint-tagged rung keys (e.g. "deep_q_network_z_60k") churn as new
    variants get trained (D-line, this session already renamed twice) -- the
    rung *number* is the stable identity, so look up by that instead of
    hardcoding a key that will go stale again.
    """
    return next(r.key for r in B.LADDER if r.rung == rung_number)


@pytest.fixture(scope="module")
def scenario():
    return Scenario.replay(CONFIG_ID, "stare")


@pytest.fixture(scope="module")
def untrained_model():
    env = rl.make_train_env(reward="hit_z")
    return DQN("MlpPolicy", env, seed=0)


def test_make_train_env_builds_a_valid_scan_env():
    from gymnasium.utils.env_checker import check_env
    env = rl.make_train_env()
    assert env.observation_space.shape == (146,)
    assert env.action_space.n == 36
    check_env(env, skip_render_check=True)


def test_make_train_env_threads_reward_through():
    env = rl.make_train_env(reward="first_intercept")
    assert env.reward_name == "first_intercept"
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
    obs = np.zeros(146, dtype=np.float32)
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
    obs = np.zeros(146, dtype=np.float32)
    a1, _ = untrained_model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert int(a1) == int(a2)


def test_train_produces_a_loadable_checkpoint(tmp_path):
    path = tmp_path / "ckpt.zip"
    rl.train(reward="hit_z", total_timesteps=10, seed=0, checkpoint=path)
    assert path.exists()
    model = rl.load_checkpoint(path)
    action, _ = model.predict(np.zeros(146, dtype=np.float32), deterministic=True)
    assert 0 <= int(action) < 36


# --------------------------------------------------------------------------- #
# PPO (rung 8) -- its own train()/load_checkpoint(), same shape as DQN's above
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ppo_untrained_model():
    env = rl.make_train_env(reward="hit_z")
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
    obs = np.zeros(146, dtype=np.float32)
    a1, _ = ppo_untrained_model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert int(a1) == int(a2)


def test_ppo_train_produces_a_loadable_checkpoint(tmp_path):
    path = tmp_path / "ckpt.zip"
    ppo.train(reward="hit_z", total_timesteps=64, seed=0, checkpoint=path)
    assert path.exists()
    model = ppo.load_checkpoint(path)
    action, _ = model.predict(np.zeros(146, dtype=np.float32), deterministic=True)
    assert 0 <= int(action) < 36
