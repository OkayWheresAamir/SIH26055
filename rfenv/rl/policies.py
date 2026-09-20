"""An opt-in policy architecture for rung 9 (`recurrent_ppo.py`): a 2-layer
LayerNorm MLP ahead of the LSTM, instead of the library-default no-op
`FlattenExtractor`.

    BASELINE  obs -> LSTM -> actor/critic          ("MlpLstmPolicy", unchanged)
    NEW       obs -> MLP -> LSTM -> actor/critic    ("MlpFeatureLstmPolicy")

Registered as a second, selectable policy (`--policy MlpFeatureLstmPolicy`)
rather than a replacement for the default -- the two are meant to be trained
as a matched pair and compared, the same convention every other architecture
question in this repository has followed (D71 "v2", D74 band-priority, D75
"v3"), not a change made unilaterally to what every existing rung already
trains on.

Everything below the extractor -- the LSTM, its hidden state handling,
episode-start masking, truncated-BPTT sequence reshaping, the rollout buffer,
and the post-LSTM actor/critic heads -- is `sb3_contrib`'s own
`RecurrentActorCriticPolicy` code, untouched. `MlpFeatureLstmPolicy` changes
exactly one thing: which `features_extractor_class` runs on each timestep's
flat observation before it reaches the LSTM.
"""

from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MlpFeaturesExtractor(BaseFeaturesExtractor):
    """`Linear -> LayerNorm -> activation`, twice, applied per timestep.

    `obs_dim -> hidden_dim -> features_dim`, each stage `Linear` then
    `LayerNorm` then `activation_fn()`. `features_dim` is the extractor's
    output width and therefore also the LSTM's input width
    (`RecurrentActorCriticPolicy.__init__` builds `nn.LSTM(self.features_dim,
    lstm_hidden_size, ...)`) -- this class has no say over `lstm_hidden_size`
    itself, only over what feeds into it.

    **Runs independently on each timestep, never across a sequence.**
    `RecurrentActorCriticPolicy.forward`/`evaluate_actions` call
    `self.extract_features(obs)` on a `(batch, obs_dim)` tensor -- during
    rollout collection `batch` is `n_envs` (one timestep), during a gradient
    update it is the whole flattened rollout (`n_seq * seq_len` timesteps
    stacked in the batch dimension). Either way this extractor sees a plain
    2D `(rows, obs_dim)` tensor and transforms each row the same way; the
    reshape into `(seq_len, n_seq, features_dim)` for the LSTM happens
    *after*, in `_process_sequence`, which this class never touches. So a
    change here cannot see, and cannot leak information across, sequence
    boundaries.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        hidden_dim: int = 256,
        activation_fn: type[nn.Module] = nn.Tanh,
    ):
        super().__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, features_dim),
            nn.LayerNorm(features_dim),
            activation_fn(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)


class MlpFeatureLstmPolicy(RecurrentActorCriticPolicy):
    """`MlpLstmPolicy` with `MlpFeaturesExtractor` ahead of the LSTM.

    Only `features_extractor_class`/`features_extractor_kwargs` change from
    the library default; every other constructor argument -- `net_arch`
    (post-LSTM actor/critic heads), `lstm_hidden_size`, `n_lstm_layers`,
    `shared_lstm`, `enable_critic_lstm` -- passes straight through
    unmodified. In particular `lstm_hidden_size` is **not** widened here just
    because the extractor was added; it stays at whatever the caller passes
    (library default 256 if nothing is), the same as `MlpLstmPolicy`.

    `activation_fn` defaults to `nn.Tanh`, matching `ActorCriticPolicy`'s own
    default (used throughout this codebase's PPO/RecurrentPPO policies) --
    if a caller overrides `activation_fn` via `policy_kwargs` for the
    post-LSTM heads, the extractor picks up the same one rather than
    silently mismatching it.
    """

    def __init__(self, *args, **kwargs):
        activation_fn = kwargs.get("activation_fn", nn.Tanh)
        kwargs.setdefault("features_extractor_class", MlpFeaturesExtractor)
        kwargs.setdefault("features_extractor_kwargs", {
            "features_dim": 256, "hidden_dim": 256, "activation_fn": activation_fn,
        })
        super().__init__(*args, **kwargs)


# `RecurrentPPO(policy=..., ...)` accepts a class directly (bypassing its own
# `policy_aliases` string lookup, which only knows the three library
# policies) -- see `recurrent_ppo.py::_resolve_policy`. This mirrors that.
POLICY_ALIASES: dict[str, type[RecurrentActorCriticPolicy]] = {
    "MlpFeatureLstmPolicy": MlpFeatureLstmPolicy,
}
