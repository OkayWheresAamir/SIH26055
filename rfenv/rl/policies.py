"""Opt-in policy architectures for rung 9 (`recurrent_ppo.py`), each changing
only which `features_extractor_class` runs on each timestep's flat
observation before it reaches the LSTM.

    BASELINE  obs -> LSTM -> actor/critic                            ("MlpLstmPolicy", unchanged)
    D78       obs -> 2-layer LayerNorm MLP -> LSTM -> actor/critic    ("MlpFeatureLstmPolicy")
    D79       obs -> per-band encoder (shared) + mean pool
                   -> concat global -> LSTM -> actor/critic           ("BandEncoderLstmPolicy")

Each is registered as a second, selectable policy (`--policy ...`) rather
than a replacement for the default -- meant to be trained as a matched pair
against the baseline and compared, the same convention every other
architecture question in this repository has followed (D71 "v2", D74
band-priority, D75 "v3"), not a change made unilaterally to what every
existing rung already trains on.

Everything below each extractor -- the LSTM, its hidden state handling,
episode-start masking, truncated-BPTT sequence reshaping, the rollout buffer,
and the post-LSTM actor/critic heads -- is `sb3_contrib`'s own
`RecurrentActorCriticPolicy` code, untouched by either policy here.
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


class BandEncoderFeaturesExtractor(BaseFeaturesExtractor):
    """Represent each of the 36 bands as its own vector, encode all 36 with
    **one shared** encoder, mean-pool across bands, then concatenate the
    global features -- the "Band Encoder + LSTM" architecture (D79):

        obs -> {per-band (36, K), global (G)} -> shared band_encoder + mean pool
            -> band_context (band_embed_dim) ++ global_encoder(global) -> LSTM -> actor/critic

    in place of `MlpFeatureLstmPolicy`'s flat-vector MLP (D78) or
    `MlpLstmPolicy`'s library-default no-op flatten.

    **The gather, not a reshape.** The flat observation interleaves *blocks*
    (`hit_rate[0:36]`, `visit_density[0:36]`, ...), not *bands* (band 0's full
    feature vector, then band 1's, ...), so turning it into `(36,
    n_band_features)` is a strided gather, done once at construction time via
    `rfenv.env.band_layout(obs_version)` and applied every `forward()` with
    plain advanced indexing (`observations[:, self.band_index]` ->
    `(rows, 36, n_band_features)`, `observations[:, self.global_index]` ->
    `(rows, n_global_features)`). Which blocks are per-band vs global is never
    hardcoded here -- `band_layout` reads it off `_BLOCK_SPECS`'s declared
    width, so `prev_action` (D75 shipped it in "v3", removed the next day --
    still a registered `N_BANDS`-wide block, just unused by any current
    `OBS_LAYOUTS` entry) would fall into the per-band bucket automatically if
    a layout ever included it again, with nothing here to update.

    **One `band_encoder`, not thirty-six.** `nn.Linear` and `nn.LayerNorm`
    both operate only on a tensor's *last* dimension and broadcast over every
    leading one, so calling `self.band_encoder(x)` on `x` shaped
    `(rows, 36, n_band_features)` applies the identical weights to every band
    -- no reshape needed, no per-band parameter, and no way to accidentally
    create 36 separate encoders the way naming them `encoder_0..encoder_35`
    would invite. `tests/test_policies.py` checks this directly (parameter
    count independent of the band count; permuting which physical band holds
    which feature vector leaves the pooled output unchanged, since mean
    pooling over a permutation of the same multiset is invariant to the
    permutation).

    **Mean pooling only, deliberately** -- `band_embeddings.mean(dim=-2)`,
    no attention. The point of this experiment is to isolate whether
    structured per-band encoding helps at all before spending a second
    experiment on how the bands are combined.

    **Global features get their own small encoder**, `Linear ->
    activation_fn()`, not a raw concatenation -- so `global_dim` is
    independently configurable from both `n_global_features` (fixed by
    `obs_version`) and `band_embed_dim`, matching `band_context`'s own
    learned-representation treatment rather than handing the LSTM a few raw,
    differently-scaled scalars next to 128 learned ones.

    **`features_dim` (and therefore the LSTM's input width) is
    `band_embed_dim + global_dim`, not `obs_dim`.** `lstm_hidden_size` is
    untouched by this class either way -- same convention `MlpFeatureLstmPolicy`
    set: more input structure is not a reason to also grow the recurrent core.

    **Runs independently per timestep, same as `MlpFeaturesExtractor` (D78).**
    `RecurrentActorCriticPolicy` always calls `extract_features` on a flat
    `(rows, obs_dim)` tensor -- one live timestep during rollout collection or
    a whole flattened rollout during a gradient update -- and only reshapes
    into `(seq_len, n_seq, features_dim)` for the LSTM afterwards, in
    `_process_sequence`, which this class never touches. It has no temporal
    state of its own; only the LSTM downstream carries memory across steps.

    `obs_version` picks the band/global split. Left `None` (the default), it
    is inferred from `observation_space`'s own width via
    `rfenv.env.obs_version_for_width` -- SB3 always constructs this class
    from the real training env's `observation_space`, so the common case
    needs no manual wiring and cannot silently drift from what the env
    actually builds; every registered width is unique today, so the inference
    is unambiguous (`obs_version_for_width` raises rather than guessing if
    that ever stops being true).
    """

    def __init__(
        self,
        observation_space,
        obs_version: str | None = None,
        band_embed_dim: int = 128,
        band_hidden_dim: int = 128,
        global_dim: int = 64,
        activation_fn: type[nn.Module] = nn.Tanh,
    ):
        from rfenv.env import band_layout, obs_version_for_width
        from rfenv.env import obs_width as env_obs_width

        width = int(np.prod(observation_space.shape))
        if obs_version is None:
            obs_version = obs_version_for_width(width)
        elif env_obs_width(obs_version) != width:
            raise ValueError(
                f"obs_version={obs_version!r} builds a {env_obs_width(obs_version)}-wide "
                f"observation, but observation_space is {width}-wide"
            )
        layout = band_layout(obs_version)

        super().__init__(observation_space, band_embed_dim + global_dim)
        self.obs_version = obs_version
        self.register_buffer("band_index", th.as_tensor(layout.band_index, dtype=th.long))
        self.register_buffer("global_index", th.as_tensor(layout.global_index, dtype=th.long))
        n_band_features = layout.band_index.shape[1]
        n_global_features = layout.global_index.shape[0]

        self.band_encoder = nn.Sequential(
            nn.Linear(n_band_features, band_hidden_dim),
            nn.LayerNorm(band_hidden_dim),
            activation_fn(),
            nn.Linear(band_hidden_dim, band_embed_dim),
            nn.LayerNorm(band_embed_dim),
            activation_fn(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(n_global_features, global_dim),
            activation_fn(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        band_features = observations[:, self.band_index]        # (rows, 36, n_band_features)
        global_features = observations[:, self.global_index]    # (rows, n_global_features)
        band_embeddings = self.band_encoder(band_features)      # (rows, 36, band_embed_dim), one shared encoder
        band_context = band_embeddings.mean(dim=-2)              # (rows, band_embed_dim), mean pool over bands
        global_context = self.global_encoder(global_features)   # (rows, global_dim)
        return th.cat([band_context, global_context], dim=-1)


class BandEncoderLstmPolicy(RecurrentActorCriticPolicy):
    """`MlpLstmPolicy` with `BandEncoderFeaturesExtractor` ahead of the LSTM.

    Only `features_extractor_class`/`features_extractor_kwargs` change from
    the library default; `lstm_hidden_size` and the post-LSTM actor/critic
    heads pass straight through unmodified, same convention as
    `MlpFeatureLstmPolicy`. `activation_fn` defaults to `nn.Tanh`, matching
    `ActorCriticPolicy`'s own default; a caller overriding it via
    `policy_kwargs` for the post-LSTM heads gets the same one in the band and
    global encoders rather than a silent mismatch.
    """

    def __init__(self, *args, **kwargs):
        activation_fn = kwargs.get("activation_fn", nn.Tanh)
        kwargs.setdefault("features_extractor_class", BandEncoderFeaturesExtractor)
        kwargs.setdefault("features_extractor_kwargs", {
            "band_embed_dim": 128, "band_hidden_dim": 128, "global_dim": 64,
            "activation_fn": activation_fn,
        })
        super().__init__(*args, **kwargs)


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
    "BandEncoderLstmPolicy": BandEncoderLstmPolicy,
}
