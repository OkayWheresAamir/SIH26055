"""Golden digests taken at commit d4f4361, before `episode_slots`, the "v3"
observation blocks and `reward_balance_obs` were built.

These exist for one reason: every one of those three changes is claimed to leave
the existing paths *bit-identical*, and that claim is not checkable by reading
the diff. `tests/test_freeze.py` makes the same bet for `constants.py` and makes
it the same way -- per-value literals, not symbols, so the test cannot drift with
the thing it is pinning.

What a failure here means: an existing rung's schedule, an existing observation
layout's bytes, or the reward attached to either has moved. That is either a bug
or a deliberate change that invalidates every checkpoint trained before it (D49,
D55, D67, D72 each paid that cost). It is never a reason to re-take the hashes.

Numpy-only on purpose -- no training stack at module level, or collection of the
whole suite aborts on a machine without torch (`scripts/doctor.py` checks this).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from rfenv import baselines as B
from rfenv.env import ScanEnv
from rfenv.rollout import run_episode
from rfenv.scenario import Scenario
from rfenv.truth import TruthGrid

CONFIG = "config_2"

# config_2/stare, seed 0, reward_balance, 600 slots each.
RUNG_LOG_SHA256 = {
    "random": "b9ffa795c6ec056245ca2846a7f713012e4aff538938493b862ea97aa3b3ef62",
    "round_robin": "c2f0893848c2b49227683eeeca2e83458c23586e5a39d526a5ccc2ea8021dd43",
    "turing_sweep": "fc4e35ca132db9dcc352d4df4e7a6fe339ca1aba7ee060585d08010570aaa959",
    "camper": "7c83c2b146600b14e3918fe0e38c5b01d61a655c08c2cdc54af4d461f2a0c837",
    "recency": "6131402f04764d471dffd032c8d9e760eba8790a21fad2791e9469d7928177f3",
    "apfeld": "496f1594ec381244e63feb85d63a76ed471c5da1005246fd7c610d3fd95ccd38",
    "apfeld_active_rfs": "1c83b7d4dfb9431f6d6dac6d14185e4b24dfc89ed979f9a63735b5cca1cd7482",
}

# Same scenario/seed, fixed action sequence a_k = (7*k) % 36 (502 steps -- it
# lands on the wide bands often enough to exercise the 2-slot dwell path).
# `total_reward` is identical for v1 and v2 by construction: the observation is
# the policy's input path and the reward never reads it (D29). v2p differs only
# because `band_priority=True` adds `priority_reward_bonus` on top (D74).
OBS_TRAJECTORY_SHA256 = {
    "v1": "a6d5ef1680f126b18f8fe3c39be0fdb4a959ba6cb2540608baceddf211b2773e",
    "v2": "f4dde7255d23248b71f31c263681d30c35b2181e63bca82fc54d0b16a4838a86",
    "v2p": "18e5a0c3386051e256baa40c559e7106c87a165c9a6f66385df961c6eb9fa685",
}

OBS_TRAJECTORY_TOTAL_REWARD = {"v1": 206.123302, "v2": 206.123302, "v2p": 238.918288}


def _log_digest(env) -> str:
    return hashlib.sha256(
        json.dumps(env.log, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _obs_trajectory(version: str) -> tuple[str, float, int]:
    """Replay the fixed action sequence; digest every observation including reset's."""
    kwargs = {"band_priority": True} if version == "v2p" else {}
    env = ScanEnv(scenario=Scenario.replay(CONFIG, "stare"),
                  reward="reward_balance", obs_version=version, **kwargs)
    obs, _ = env.reset(seed=0)
    h = hashlib.sha256(np.ascontiguousarray(obs, dtype=np.float32).tobytes())
    k, done, total = 0, False, 0.0
    while not done:
        obs, reward, done, _, _ = env.step((7 * k) % 36)
        h.update(np.ascontiguousarray(obs, dtype=np.float32).tobytes())
        total += reward
        k += 1
    return h.hexdigest(), total, obs.shape[0]


@pytest.mark.parametrize("key", sorted(RUNG_LOG_SHA256))
def test_heuristic_rung_schedule_is_unchanged(key):
    """Every checkpoint-free rung's per-slot log, byte for byte.

    These seven need no training stack, so they pin the ladder on any machine.
    """
    scenario = Scenario.replay(CONFIG, "stare")
    env = ScanEnv(scenario=scenario, reward="reward_balance")
    policy = B.make(key, seed=0, grid=TruthGrid.from_scenario(scenario))
    run_episode(env, policy, seed=0)

    assert len(env.log) == 600
    assert _log_digest(env) == RUNG_LOG_SHA256[key], (
        f"{key}'s schedule moved. Every number this rung has ever produced is "
        f"now suspect -- do not re-take this hash to make the test pass."
    )


@pytest.mark.parametrize("version", sorted(OBS_TRAJECTORY_SHA256))
def test_observation_trajectory_is_unchanged(version):
    """The full float32 byte stream of an existing layout, across an episode.

    Catches a block that moved, changed dtype, changed scaling, or acquired a
    neighbour -- none of which a width assertion would see.
    """
    digest, total, width = _obs_trajectory(version)
    assert width == {"v1": 183, "v2": 362, "v2p": 398}[version]
    assert digest == OBS_TRAJECTORY_SHA256[version], (
        f'"{version}" observation bytes moved; every checkpoint on this layout '
        f"is invalidated. See D49/D55/D67/D72 for what that costs."
    )
    assert total == pytest.approx(OBS_TRAJECTORY_TOTAL_REWARD[version], abs=1e-4)


def test_the_observation_does_not_touch_the_reward():
    """v1 and v2 differ only in the policy's input path, so the reward is equal.

    Pinned because it is the mechanical form of D29's split: were a future block
    to feed back into `step()`'s reward, this is what would notice.
    """
    _, total_v1, _ = _obs_trajectory("v1")
    _, total_v2, _ = _obs_trajectory("v2")
    assert total_v1 == pytest.approx(total_v2, abs=1e-9)
