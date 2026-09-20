"""Online fine-tuning (D77).

The only test module here that needs the training stack, so it skips at module
level rather than importing it at the top -- a module-level torch import aborts
collection of the *whole* suite on a machine without one, which has happened
twice and which `scripts/doctor.py` checks for.

Pinned here: (1) the gradient is fed the observable reward, not `step()`'s
truth-fed one, which is the whole deployability claim; (2) fine-tuning actually
moves the weights **and** continues the checkpoint's step count rather than
restarting it; (3) at `learning_rate=0` it moves nothing at all -- the no-op
control, which is what rules out the harness perturbing the policy by some route
other than the optimiser; (4) `custom_objects` really does resize the rollout
buffer, which is the one detail that fails silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("torch")

import torch  # noqa: E402

from rfenv.constants import N_SLOTS  # noqa: E402
from rfenv.rl.online import (  # noqa: E402
    ONLINE_DEFAULTS,
    fine_tune,
    make_observable_reward_wrapper,
    make_online_env,
)

TINY = {"n_steps": 64, "batch_size": 16, "policy_kwargs": {"lstm_hidden_size": 16}}


@pytest.fixture(scope="module")
def base_checkpoint(tmp_path_factory) -> Path:
    """One tiny v3 checkpoint, trained once and shared -- training is the slow part."""
    from rfenv.rl.recurrent_ppo import train

    path = tmp_path_factory.mktemp("online") / "v3_tiny.zip"
    train(reward="reward_balance", obs_version="v3", band_priority=True,
          total_timesteps=128, seed=0, checkpoint=path, verbose=0, device="cpu",
          hyperparameters=dict(TINY))
    return path


def _state_dict(path):
    from sb3_contrib import RecurrentPPO

    return {k: v.clone()
            for k, v in RecurrentPPO.load(path, device="cpu").policy.state_dict().items()}


# --------------------------------------------------------------------------- #
# The reward the gradient sees
# --------------------------------------------------------------------------- #

def test_the_wrapper_returns_the_observable_reward_at_every_step():
    env = make_online_env(episode_slots=2 * N_SLOTS, obs_version="v3",
                          band_priority=True)
    env.reset(seed=0)
    inner = env.unwrapped
    for k in range(200):
        _, reward, done, _, _ = env.step((7 * k) % 36)
        assert reward == inner.last_reward_obs
        if done:
            break


def test_the_observable_reward_really_does_differ_from_the_truth_reward():
    """If it never did, `Y` and `Z` would be the same and Pd would be 1."""
    truth = make_online_env(episode_slots=N_SLOTS, obs_version="v3",
                            band_priority=True, observable_reward=False)
    obs_arm = make_online_env(episode_slots=N_SLOTS, obs_version="v3",
                              band_priority=True, observable_reward=True)
    truth.reset(seed=0)
    obs_arm.reset(seed=0)
    differed, k, done = 0, 0, False
    while not done:
        _, rt, done, _, _ = truth.step((13 * k) % 36)
        _, ro, _, _, _ = obs_arm.step((13 * k) % 36)
        differed += rt != ro
        k += 1
    assert differed > 0


def test_scoring_stays_on_the_truth_reward_even_while_the_gradient_sees_Y():
    """A fine-tuned checkpoint must be judged on the same yardstick as every rung."""
    env = make_online_env(episode_slots=N_SLOTS, obs_version="v3", band_priority=True)
    env.reset(seed=0)
    total_returned, done, k = 0.0, False, 0
    while not done:
        _, reward, done, _, _ = env.step((5 * k) % 36)
        total_returned += reward
        k += 1
    assert env.unwrapped.total_reward != pytest.approx(total_returned)
    assert env.unwrapped.episode_metrics()["total_reward"] == env.unwrapped.total_reward


def test_the_wrapper_is_opt_out():
    env = make_online_env(episode_slots=N_SLOTS, obs_version="v3",
                          band_priority=True, observable_reward=False)
    assert not isinstance(env, make_observable_reward_wrapper())


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #

def test_fine_tuning_moves_the_weights_and_continues_the_step_count(
    base_checkpoint, tmp_path
):
    """`reset_num_timesteps=False` is what makes this a continuation, not a restart."""
    before = _state_dict(base_checkpoint)
    out = tmp_path / "adapted.zip"
    model = fine_tune(
        checkpoint=base_checkpoint, out_checkpoint=out, total_timesteps=200,
        episode_slots=2 * N_SLOTS, obs_version="v3", view="off", device="cpu",
        hyperparameters={"n_steps": 100, "batch_size": 25},
        env_kwargs={"band_priority": True},
    )
    after = _state_dict(out)
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert model.num_timesteps > 128            # the checkpoint's own count carried over
    assert out.exists() and out.with_suffix(".json").exists()


def test_a_zero_learning_rate_leaves_the_policy_bit_identical(base_checkpoint, tmp_path):
    """The no-op control: nothing in this harness perturbs the policy but the optimiser."""
    before = _state_dict(base_checkpoint)
    out = tmp_path / "noop.zip"
    fine_tune(
        checkpoint=base_checkpoint, out_checkpoint=out, total_timesteps=200,
        episode_slots=2 * N_SLOTS, obs_version="v3", view="off", device="cpu",
        hyperparameters={"n_steps": 100, "batch_size": 25, "learning_rate": 0.0},
        env_kwargs={"band_priority": True},
    )
    after = _state_dict(out)
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_custom_objects_actually_resizes_the_rollout_buffer(base_checkpoint):
    """The detail that fails silently.

    `n_steps` is baked into the checkpoint and read while the rollout buffer is
    built inside `_setup_model()`. Assigning `model.n_steps` afterwards leaves
    the old buffer, and the run collects the checkpoint's `n_steps` per update
    instead of the one asked for -- with no error and no warning.
    """
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: make_online_env(
        episode_slots=N_SLOTS, obs_version="v3", band_priority=True)])
    model = RecurrentPPO.load(base_checkpoint, env=env, device="cpu",
                              custom_objects={"n_steps": 100, "batch_size": 25})
    assert model.n_steps == 100
    assert model.rollout_buffer.buffer_size == 100


def test_the_only_departures_from_the_training_regime_are_brakes():
    """Adaptation starts from a policy that works; the risk is wrecking it.

    The three that move are all brakes. Everything shaping the *estimator* --
    rollout length, batch size -- stays at the training value, so the advantage
    horizon does not shift underneath a policy that already works.
    """
    assert ONLINE_DEFAULTS["learning_rate"] < 3e-4
    assert ONLINE_DEFAULTS["clip_range"] < 0.2
    assert ONLINE_DEFAULTS["target_kl"] is not None
    assert ONLINE_DEFAULTS["ent_coef"] > 0.0        # D54: collapse is the failure mode


def test_the_rollout_is_not_shortened_to_buy_more_updates():
    """`n_steps` is the BPTT window, not just a batch size.

    `RecurrentPPO` carries the LSTM state across rollout boundaries but not the
    gradient, so shortening this truncates the long-horizon credit assignment an
    in-context ("v3") agent exists to learn. Adaptation belongs in the hidden
    state; the weights learn the rule that produces it, and that is the long
    problem. Pinned because "more updates per hour" is a persuasive-sounding
    reason to shrink it and optimises the wrong one of the two.
    """
    assert ONLINE_DEFAULTS["n_steps"] == 8192
    assert ONLINE_DEFAULTS["batch_size"] == 128


def test_per_segment_metrics_are_written_and_never_describe_an_unfinished_segment(
    base_checkpoint, tmp_path
):
    """Coverage per segment against segment index *is* the result for this mode."""
    fine_tune(
        checkpoint=base_checkpoint, out_checkpoint=tmp_path / "m.zip",
        total_timesteps=1400, episode_slots=2 * N_SLOTS, obs_version="v3",
        view="off", out=str(tmp_path), device="cpu",
        hyperparameters={"n_steps": 200, "batch_size": 50},
        env_kwargs={"band_priority": True},
    )
    rows = [json.loads(line)
            for line in (tmp_path / "segments.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        assert row["segment"] >= 0          # never a boundary that has not been crossed
        assert row["mission"] >= 0
        assert row["found_this_segment"] >= 0
        assert 0 <= row["found_total"] <= row["n_detectable"]


def test_a_real_view_actually_draws_during_fine_tuning(base_checkpoint, tmp_path, capsys):
    """Regression: `isinstance(view, NullView)` is true for every real view too.

    `AnsiHeatStrip`/`MatplotlibLiveView` are themselves subclasses of `NullView`
    (they share its no-op `open`/`close`), so an `isinstance` check meant to
    single out "no view was asked for" matched every view -- `_callbacks()`
    silently never attached `OnlineViewCallback` for `--view light` or
    `--view full`. The fine-tune ran correctly and nothing ever drew, with no
    error to notice it by. Caught only by looking for actual output, which no
    prior test in this file did (every one of them runs at `view="off"`).
    """
    fine_tune(
        checkpoint=base_checkpoint, out_checkpoint=tmp_path / "watched.zip",
        total_timesteps=200, episode_slots=2 * N_SLOTS, obs_version="v3",
        view="light", device="cpu",
        hyperparameters={"n_steps": 100, "batch_size": 25},
        env_kwargs={"band_priority": True},
    )
    out = capsys.readouterr().out
    assert "slot" in out and "found" in out, (
        "the live view drew nothing during a --view light fine-tune"
    )


def test_the_manifest_records_the_adapted_checkpoint(base_checkpoint, tmp_path):
    out = tmp_path / "adapted.zip"
    fine_tune(
        checkpoint=base_checkpoint, out_checkpoint=out, total_timesteps=200,
        episode_slots=2 * N_SLOTS, obs_version="v3", view="off", device="cpu",
        hyperparameters={"n_steps": 100, "batch_size": 25},
        env_kwargs={"band_priority": True}, description="a test adaptation",
    )
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert manifest["observation_width"] == 400
    assert manifest["algorithm"] == "RecurrentPPO"
    assert "a test adaptation" in manifest["description"]
    assert manifest["total_timesteps"] > 128
    assert manifest["hyperparameters"]["n_steps"] == 100


# --------------------------------------------------------------------------- #
# The deployable contract still holds afterwards
# --------------------------------------------------------------------------- #

def test_an_adapted_checkpoint_still_runs_through_the_ordinary_deployable_driver(
    base_checkpoint, tmp_path
):
    """Fine-tuning uses SB3's acting loop; evaluation must still use `run_episode`.

    That is the point of saving and re-loading rather than scoring in place: the
    `(obs, info) -> int` contract, and `guarded()`'s allowlist with it, are
    exercised exactly as they are for every other rung.
    """
    from rfenv.baselines.guard import guarded
    from rfenv.env import ScanEnv
    from rfenv.rl.common import RecurrentRLScheduler
    from rfenv.rl.recurrent_ppo import load_checkpoint
    from rfenv.rollout import run_episode
    from rfenv.scenario import Scenario

    out = tmp_path / "adapted.zip"
    fine_tune(
        checkpoint=base_checkpoint, out_checkpoint=out, total_timesteps=200,
        episode_slots=2 * N_SLOTS, obs_version="v3", view="off", device="cpu",
        hyperparameters={"n_steps": 100, "batch_size": 25},
        env_kwargs={"band_priority": True},
    )
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"),
                  obs_version="v3", band_priority=True)
    policy = guarded(RecurrentRLScheduler(load_checkpoint(out)))
    run_episode(env, policy, seed=0)
    assert len(env.log) == N_SLOTS
    assert np.isfinite(env.episode_metrics()["emitter_coverage"])
