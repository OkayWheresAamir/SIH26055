"""Algorithm-agnostic pieces: the training env factory and the policy adapter.

Every SB3 algorithm trains on the same env and exposes the same `.predict()`
API, so neither of these belongs to any one algorithm -- only `train()`,
`load_checkpoint()` and the default checkpoint path (in `dqn.py`, and future
siblings) are algorithm-specific.
"""

from __future__ import annotations

from stable_baselines3 import DQN

from rfenv.env import DEFAULT_REWARD, ScanEnv
from rfenv.scenario import EmitterPool


def make_train_env(*, reward: str = DEFAULT_REWARD) -> ScanEnv:
    """A fresh-scenario-per-reset training env (D25, D32) -- what RL trains on.

    `reward` is threaded straight through to ScanEnv, which is the whole
    "reward as a hyperparameter" requirement -- ScanEnv already validates it
    against `REWARDS`.
    """
    return ScanEnv(pool=EmitterPool.from_train(), reward=reward)


class RLScheduler:
    """A trained model as a `(obs, info) -> band` callable, per rollout.run_episode.

    Never reads `info` -- the policy conditions on the D34 observation alone,
    which is what makes it deployable. `guarded()` in baselines/ still strips
    `info` before this ever sees it; this class simply never asks.

    Wraps any SB3 algorithm that exposes `.predict()` -- today that's only
    `DQN` (see `dqn.py`), but nothing here is DQN-specific.
    """

    def __init__(self, model: DQN):
        self.model = model

    def __call__(self, obs, info) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
