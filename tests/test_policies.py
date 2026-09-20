"""The opt-in `MlpFeatureLstmPolicy` (D78): obs -> MLP -> LSTM -> actor/critic.

Skips at module level without the training stack, same reason `test_online.py`
does -- a module-level torch/sb3_contrib import aborts collection of the whole
suite on a machine without either, which `scripts/doctor.py` checks for.

Pinned here: (1) the baseline (`MlpLstmPolicy`) is untouched -- its extractor
is still the library's no-op `FlattenExtractor` and its LSTM still takes the
raw observation width; (2) `MlpFeatureLstmPolicy` really does run the MLP
first -- the LSTM's input width is the extractor's `features_dim` (256), not
the observation width; (3) `lstm_hidden_size` is not silently widened by
adding the extractor -- it stays at whatever is passed, library default
included; (4) the extractor treats every row of a `(batch, obs_dim)` tensor
independently, so a whole flattened rollout and a single live timestep
produce the same per-row output -- the "never flattens the sequence into one
vector" requirement, pinned directly rather than inferred from the class not
containing an `nn.Flatten` over the wrong axis; (5) a checkpoint trained under
the new policy reloads through the ordinary `load_checkpoint()` and predicts,
same as any other rung's checkpoint.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("torch")

import torch  # noqa: E402

from rfenv.rl.policies import MlpFeaturesExtractor, MlpFeatureLstmPolicy, POLICY_ALIASES  # noqa: E402
from rfenv.rl.recurrent_ppo import _resolve_policy  # noqa: E402

TINY = {"n_steps": 64, "batch_size": 16}


def _train(policy, tmp_path, **hyperparameters):
    from rfenv.rl.recurrent_ppo import train

    path = tmp_path / "m.zip"
    hp = dict(TINY)
    hp.update(hyperparameters)
    model = train(policy=policy, total_timesteps=128, seed=0, checkpoint=path,
                  verbose=0, device="cpu", hyperparameters=hp)
    return model, path


# --------------------------------------------------------------------------- #
# The baseline is untouched
# --------------------------------------------------------------------------- #

def test_the_baseline_policy_still_uses_the_library_default_extractor(tmp_path):
    from stable_baselines3.common.torch_layers import FlattenExtractor

    model, _ = _train("MlpLstmPolicy", tmp_path)
    assert isinstance(model.policy.features_extractor, FlattenExtractor)
    assert model.policy.lstm_actor.input_size == model.observation_space.shape[0]


# --------------------------------------------------------------------------- #
# The new policy runs the MLP first
# --------------------------------------------------------------------------- #

def test_the_feature_policy_puts_an_mlp_ahead_of_the_lstm(tmp_path):
    model, _ = _train("MlpFeatureLstmPolicy", tmp_path)
    assert isinstance(model.policy.features_extractor, MlpFeaturesExtractor)
    # The LSTM's input is the extractor's output width, not the raw observation.
    assert model.policy.lstm_actor.input_size == 256
    assert model.policy.lstm_actor.input_size != model.observation_space.shape[0]


def test_lstm_hidden_size_is_not_widened_just_because_the_extractor_was_added(tmp_path):
    baseline, _ = _train("MlpLstmPolicy", tmp_path)
    feature, _ = _train("MlpFeatureLstmPolicy", tmp_path)
    assert baseline.policy.lstm_actor.hidden_size == feature.policy.lstm_actor.hidden_size


def test_lstm_hidden_size_still_honours_an_explicit_override(tmp_path):
    model, _ = _train("MlpFeatureLstmPolicy", tmp_path,
                       policy_kwargs={"lstm_hidden_size": 32})
    assert model.policy.lstm_actor.hidden_size == 32
    # The extractor's own width is independent of the LSTM's.
    assert model.policy.lstm_actor.input_size == 256


def test_the_actor_and_critic_heads_are_unmodified_recurrent_ppo_ones(tmp_path):
    from stable_baselines3.common.torch_layers import MlpExtractor

    baseline, _ = _train("MlpLstmPolicy", tmp_path)
    feature, _ = _train("MlpFeatureLstmPolicy", tmp_path)
    assert type(baseline.policy.mlp_extractor) is type(feature.policy.mlp_extractor)
    assert isinstance(feature.policy.mlp_extractor, MlpExtractor)
    assert type(baseline.policy.action_net) is type(feature.policy.action_net)
    assert type(baseline.policy.value_net) is type(feature.policy.value_net)
    assert feature.policy.action_net.out_features == baseline.policy.action_net.out_features


# --------------------------------------------------------------------------- #
# The extractor treats each timestep independently
# --------------------------------------------------------------------------- #

def test_the_extractor_never_mixes_rows_of_the_batch():
    """A stand-in for "does not flatten the sequence into one vector": every
    row of a `(batch, obs_dim)` tensor must transform independently of every
    other row, whether that batch is one live timestep or a whole flattened
    rollout. Verified directly by checking that permuting the batch dimension
    permutes the output the same way, and that a single row processed alone
    reproduces its row from the batched pass exactly.
    """
    space = _flat_box(183)
    extractor = MlpFeaturesExtractor(space, features_dim=32, hidden_dim=32)
    extractor.eval()
    batch = torch.randn(5, 183)
    with torch.no_grad():
        out = extractor(batch)
        permuted_out = extractor(batch[[4, 3, 2, 1, 0]])
        single = extractor(batch[2:3])
    assert torch.allclose(permuted_out, out[[4, 3, 2, 1, 0]], atol=1e-6)
    assert torch.allclose(single[0], out[2], atol=1e-6)


def test_the_extractor_shape_is_obs_dim_to_hidden_to_features_dim():
    space = _flat_box(183)
    extractor = MlpFeaturesExtractor(space, features_dim=64, hidden_dim=48)
    linears = [m for m in extractor.mlp if isinstance(m, torch.nn.Linear)]
    assert [tuple(l.weight.shape) for l in linears] == [(48, 183), (64, 48)]
    norms = [m for m in extractor.mlp if isinstance(m, torch.nn.LayerNorm)]
    assert [n.normalized_shape for n in norms] == [(48,), (64,)]


# --------------------------------------------------------------------------- #
# Round-trip through the checkpoint machinery
# --------------------------------------------------------------------------- #

def test_a_checkpoint_trained_under_the_feature_policy_reloads_and_predicts(tmp_path):
    from rfenv.rl.recurrent_ppo import load_checkpoint

    _, path = _train("MlpFeatureLstmPolicy", tmp_path)
    reloaded = load_checkpoint(path)
    assert isinstance(reloaded.policy, MlpFeatureLstmPolicy)
    obs = np.zeros((1, reloaded.observation_space.shape[0]), dtype=np.float32)
    action, _ = reloaded.predict(obs, episode_start=np.array([True]))
    assert reloaded.action_space.contains(int(action[0]))


def test_resolve_policy_maps_the_cli_string_to_the_class_only_for_the_new_policy():
    assert _resolve_policy("MlpFeatureLstmPolicy") is MlpFeatureLstmPolicy
    assert _resolve_policy("MlpLstmPolicy") == "MlpLstmPolicy"
    assert set(POLICY_ALIASES) == {"MlpFeatureLstmPolicy"}


def _flat_box(width):
    from gymnasium import spaces

    return spaces.Box(low=-1.0, high=1.0, shape=(width,), dtype=np.float32)
