"""The ladder: `Rung`, `LADDER`, and the `make()` factory that ties every rung together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from rfenv.baselines.apfeld import Apfeld
from rfenv.baselines.camper import GreedyCamper, OracleCamper
from rfenv.baselines.guard import guarded
from rfenv.baselines.oracle import PulseCaptureOracle
from rfenv.baselines.recency import RecencyActivity
from rfenv.baselines.simple import (
    BAND_AT_SLOT,
    EQUAL_AIRTIME_CYCLE,
    EQUAL_AIRTIME_CYCLE_SLOTS,
    RandomBands,
    RoundRobin,
    TuringSweep,
)
from rfenv.constants import DWELL_SLOTS, N_BANDS, N_SLOTS, SLOT_S


@dataclass(frozen=True)
class Rung:
    """One row of `EVALUATION.md` §5, with the factory that builds it."""

    key: str
    rung: str
    label: str
    purpose: str
    factory: Callable
    deployable: bool = True   # False = reads truth, reported as a reference line
    needs_grid: bool = False


def _dqn_rung_factory(checkpoint_path: Path) -> Callable:
    """A `Rung` factory for one trained DQN checkpoint.

    Parameterised so multiple trained variants can coexist as separate rungs
    instead of one checkpoint overwriting another -- the same pattern rungs 6
    and 6a already use for Apfeld's two forms. To register a new variant:
    train it to its own checkpoint file --

        python -m rfenv.rl --checkpoint runs/checkpoints/<name>.zip [--reward ... --timesteps ...]

    -- then add one `Rung(...)` line below using this factory, e.g.

        Rung("deep_q_network_hit_y", "7a", "DQN (hit_y)",
             "Same algorithm, trained on hit_y -- retired from REWARDS "
             "after failing D62's screen (D29).",
             _dqn_rung_factory(Path("runs/checkpoints/deep_q_network_hit_y.zip"))),

    A different algorithm gets its own equivalent factory -- see
    `_ppo_rung_factory` immediately below for the worked example.

    Lazy import inside the closure: torch/stable-baselines3 stay optional for
    everything that isn't training or running an RL rung (mirrors
    compare.figures()'s local `from rfenv import render`).
    """
    def factory(rng, grid):
        from rfenv.rl import RLScheduler
        from rfenv.rl.dqn import load_checkpoint
        # `deterministic=True`, against `RLScheduler`'s own default, and the one
        # rung that should be: a DQN has no stochastic policy to sample from, so
        # SB3's `deterministic=False` means epsilon-greedy *exploration* noise
        # (`exploration_final_eps`, 0.05 by default) -- a training artefact, not
        # a learned distribution. The greedy argmax is this rung's policy.
        return RLScheduler(load_checkpoint(checkpoint_path), deterministic=True)
    return factory


def _ppo_rung_factory(checkpoint_path: Path) -> Callable:
    """A `Rung` factory for one trained PPO checkpoint -- `_dqn_rung_factory`'s
    sibling for a different algorithm, same shape, one line different
    (`rfenv.rl.ppo.load_checkpoint` instead of `rfenv.rl.dqn`'s)."""
    def factory(rng, grid):
        from rfenv.rl import RLScheduler
        from rfenv.rl.ppo import load_checkpoint
        _seed_torch(rng)
        return RLScheduler(load_checkpoint(checkpoint_path))
    return factory


def _recurrent_ppo_rung_factory(checkpoint_path: Path) -> Callable:
    """A `Rung` factory for one trained RecurrentPPO checkpoint (rung 9).

    Same parameterised-checkpoint shape as `_dqn_rung_factory`/`_ppo_rung_factory`,
    but wraps the checkpoint in `RecurrentRLScheduler`, not `RLScheduler` -- a
    recurrent policy's `.predict()` takes and returns hidden state across
    calls, which `RLScheduler` has no slot for (see `common.py`). sb3-contrib,
    not plain stable-baselines3, is what `rfenv.rl.recurrent_ppo.load_checkpoint`
    pulls in -- lazy-imported here for the same reason torch/SB3 are.
    """
    def factory(rng, grid):
        from rfenv.rl.common import RecurrentRLScheduler
        from rfenv.rl.recurrent_ppo import load_checkpoint
        _seed_torch(rng)
        return RecurrentRLScheduler(load_checkpoint(checkpoint_path))
    return factory


def _seed_torch(rng) -> None:
    """Fold this episode's rung stream into torch's RNG, so a sampled policy
    still reproduces from `(seed, rung key)` alone.

    `RLScheduler`/`RecurrentRLScheduler` default to sampling rather than taking
    the argmax (see `common.py` for the measurement behind that), and SB3's
    `.predict()` draws that sample from torch's *global* generator -- it takes no
    generator argument. Without this, two runs of `compare.py` at the same seed
    would give a sampled rung different actions, and `EVALUATION.md` §7's
    "identical scenarios and seeds" would quietly stop holding for rungs 8 and 9.

    Global state, unavoidably, which is why it is set here per rung rather than
    once per process: `make()` already derives `rng` from the episode seed and
    the rung's own key, so each rung re-seeds torch from its own stream and no
    rung's draw depends on where it sits in the ladder.
    """
    import torch
    torch.manual_seed(int(rng.integers(2 ** 31)))


# Literals, not imports from rfenv.rl.dqn/ppo/recurrent_ppo.DEFAULT_CHECKPOINT --
# importing anything from rfenv.rl here would make torch/stable-baselines3 (and,
# for rung 9, sb3-contrib) a hard dependency of this module for everyone, even
# those who never touch an RL rung. tests/test_baselines.py's
# _UNTRAINED_RUNG_CHECKPOINTS duplicates the same paths for the same reason.
_DQN_DEFAULT_CHECKPOINT = Path("runs/checkpoints/deep_q_network.zip")
_PPO_DEFAULT_CHECKPOINT = Path("runs/checkpoints/ppo.zip")
_RECURRENT_PPO_DEFAULT_CHECKPOINT = Path("runs/checkpoints/recurrent_ppo.zip")


LADDER: tuple[Rung, ...] = (
    Rung("random", "1", "Random",
         "Lower bound; the coverage-heavy extreme.",
         lambda rng, grid: RandomBands(rng)),
    Rung("round_robin", "2", "Round-robin (equal airtime)",
         "The open-loop strategy the PS targets. The floor to beat (D43).",
         lambda rng, grid: RoundRobin()),
    Rung("turing_sweep", "3", "Turing reference sweep",
         "The dataset's own schedule; makes our numbers comparable to the recordings.",
         lambda rng, grid: TuringSweep()),
    Rung("camper", "4", "Greedy static (camper)",
         "The degenerate exploit; included to show a single metric can be gamed.",
         lambda rng, grid: GreedyCamper(rng)),
    Rung("recency", "5", "Recency / activity heuristic",
         "Simple adaptive benchmark.",
         lambda rng, grid: RecencyActivity(rng)),
    Rung("apfeld_active_rfs", "6a", "Apfeld: Active RFs",
         "Apfeld's own ablation: the tentative list with no period estimation (D13).",
         lambda rng, grid: Apfeld(rng, use_period_estimation=False)),
    Rung("apfeld", "6", "Apfeld adaptive (no tracking)",
         "Published non-learning adaptive strategy. The serious bar.",
         lambda rng, grid: Apfeld(rng, use_period_estimation=True)),

    # Rungs 7 and 7a were both trained on `hit_z`/`hit_y`, D29's original two
    # candidates -- retired from `REWARDS` after both failed D62's screen
    # (each ranks the camper above every sweeping policy). Already permanently
    # unloadable regardless, since their checkpoints predate D49's observation
    # width change; kept registered so the ladder's skip-with-a-warning path
    # (`checkpoint_is_usable`) is what reports that, rather than a KeyError.
    Rung("deep_q_network_z_60k", "7", "DQN scheduler",
         "Ours. First naive pass: untuned SB3 DQN, trained on hit_z -- a "
         "candidate retired from REWARDS after failing D62's screen (D29).",
         _dqn_rung_factory(_DQN_DEFAULT_CHECKPOINT)),

    Rung("deep_q_network_hit_y_20k", "7a", "DQN (hit_y)",
         "Same algorithm as rung 7, trained on hit_y instead of hit_z -- both "
         "retired from REWARDS after failing D62's screen (D29). A worked "
         "example of the variant-registration pattern this docstring "
         "describes, not yet a tuned comparison (5,000 timesteps).",
         _dqn_rung_factory(Path("runs/checkpoints/deep_q_network_hit_y.zip"))),

    # More trained DQN variants register here, one Rung(...) line each -- see
    # _dqn_rung_factory's docstring for the exact pattern (rungs "7b", "7c", ...,
    # mirroring how 6/6a already coexist for Apfeld's two forms).

    Rung("ppo_hit_z_60k", "8", "PPO scheduler",
         "Ours. First naive pass: untuned SB3 PPO, trained on hit_z -- a "
         "candidate retired from REWARDS after failing D62's screen (D29).",
         _ppo_rung_factory(_PPO_DEFAULT_CHECKPOINT)),

    Rung("ppo_first_intercept_800k", "8a", "PPO (first_intercept, 800k)",
            "Ours. Second naive pass: untuned SB3 PPO, trained on first_intercept (D29). "
            "Sees only the D34 observation.",
            _ppo_rung_factory(Path("runs/checkpoints/ppo_fi.zip"))),

    Rung("ppo_first_intercept_100k", "8b", "PPO (first_intercept, 100k)",
            "Ours. Second naive pass: untuned SB3 PPO, trained on first_intercept (D29). "
            "Sees only the D34 observation.",
            _ppo_rung_factory(Path("runs/checkpoints/ppo_fi_100k.zip"))),

    Rung("ppo_first_intercept_2M", "8c", "PPO (first_intercept, 2M)",
                "Ours. Second naive pass: untuned SB3 PPO, trained on first_intercept (D29). "
                "Sees only the D34 observation.",
                _ppo_rung_factory(Path("runs/checkpoints/ppo_fi_2M.zip"))),
    # Commented out, not deleted: weighted_camp (the reward this was trained on)
    # was retired from REWARDS (env.py), and the observation shape has since
    # moved past what this checkpoint was trained against too. Re-enable only
    # after re-registering weighted_camp and retraining runs/checkpoints/ppo_wt_cmp_1M.zip.
    # Rung("ppo_weighted_camp", "8d", "PPO (weighted_camp, 1M)",
    #     "Ours. Second naive pass: untuned SB3 PPO, trained on weighted_camp (D29). "
    #     "Sees only the D34 observation.",
    #     _ppo_rung_factory(Path("runs/checkpoints/ppo_wt_cmp_1M.zip"))),
    # More trained PPO variants register the same way, one Rung(...) line each
    # using _ppo_rung_factory (rungs "8d", "8e", ...) -- give each a label that
    # names what's different about it (D29 candidate, timestep count, ...), not
    # a shared "PPO scheduler": two rungs with the same label collapse to one
    # point in render.pareto() and one row anywhere else keyed by label, since
    # compare.py's `_means`/figure-building code keys by label, not by key.

    # ----------------------------------------------------------------- #
    # The pre-D55 RecurrentPPO family, retired 2026-09-09.
    #
    # Commented out rather than deleted, matching rung 8d's precedent. Every one
    # of these was trained on a reward key that no longer exists in `REWARDS`
    # (`first_intercept`, renamed and rewritten -- D50/D53) against a 147-wide
    # observation (D55 took it to 146), so they are permanently unloadable: not
    # stale, dead. They cannot produce a row again and their published results
    # are already marked superseded in `EVALUATION.md` §5.
    #
    # They are commented out **because their rung numbers collided** with the
    # corrected-reward family below -- 9a..9d appeared twice, which breaks every
    # lookup by number and stamps duplicate labels on the Pareto markers. git
    # holds them if a retrain ever wants the numbering back.
    # ----------------------------------------------------------------- #
    # Rung("recurrent_ppo_first_intercept_1M", "9", "Recurrent PPO (LSTM),1M",
    #      "Ours. Third algorithm: sb3-contrib RecurrentPPO (MlpLstmPolicy), "
    #      "trained on hit_z (D29). Sees the same D34 observation as every "
    #      "other rung, but the policy carries an LSTM hidden state across "
    #      "the episode instead of acting on each look alone.",
    #      _recurrent_ppo_rung_factory(_RECURRENT_PPO_DEFAULT_CHECKPOINT)),
    # Rung("recurrent_ppo_first_intercept_100k", "9a", "Recurrent PPO (LSTM),100k",
    #         "Ours. Third algorithm: sb3-contrib RecurrentPPO (MlpLstmPolicy), "
    #         "trained on hit_z (D29). Sees the same D34 observation as every "
    #         "other rung, but the policy carries an LSTM hidden state across "
    #         "the episode instead of acting on each look alone.",
    #         _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_ppo5_100000_steps.zip"))),
    # Rung("recurrent_ppo_first_intercept_200k", "9b", "Recurrent PPO (LSTM),200k",
    #         "Ours. Third algorithm: sb3-contrib RecurrentPPO (MlpLstmPolicy), "
    #         "trained on hit_z (D29). Sees the same D34 observation as every "
    #         "other rung, but the policy carries an LSTM hidden state across "
    #         "the episode instead of acting on each look alone.",
    #         _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_ppo5_200000_steps.zip"))),
    # Rung("recurrent_ppo_first_intercept_300k", "9c", "Recurrent PPO (LSTM),300k",
    #         "Ours. Third algorithm: sb3-contrib RecurrentPPO (MlpLstmPolicy), "
    #         "trained on hit_z (D29). Sees the same D34 observation as every "
    #         "other rung, but the policy carries an LSTM hidden state across "
    #         "the episode instead of acting on each look alone.",
    #         _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_ppo4_300000_steps.zip"))),
    # Rung("recurrent_ppo_first_intercept_400k", "9d", "Recurrent PPO (LSTM),400k",
    #         "Ours. Third algorithm: sb3-contrib RecurrentPPO (MlpLstmPolicy), "
    #         "trained on hit_z (D29). Sees the same D34 observation as every "
    #         "other rung, but the policy carries an LSTM hidden state across "
    #         "the episode instead of acting on each look alone.",
    #         _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_ppo4_400000_steps.zip"))),


    Rung("lstm_balance_100k_1M", "9a", "Recurrent PPO (reward_balance, 100k)",
     "Ours. Trained on reward_balance after D52/D53/D55, 100k timesteps.",
     _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s1.zip"))),
    
    Rung("lstm_balance_200k_1M", "9b", "Recurrent PPO (reward_balance, 200k)",
     "Ours. Trained on reward_balance after D52/D53/D55, 200k timesteps.",
     _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s2.zip"))),

    Rung("lstm_balance_300k_1M", "9c", "Recurrent PPO (reward_balance, 300k)",
         "Ours. Trained on reward_balance after D52/D53/D55, 300k timesteps.",
         _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s3.zip"))),

    Rung("lstm_balance_400k_1M", "9d", "Recurrent PPO (reward_balance, 400k)",
        "Ours. Trained on reward_balance after D52/D53/D55, 400k timesteps.",
        _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s4.zip"))),

    Rung("lstm_balance_500k_1M", "9e", "Recurrent PPO (reward_balance, 500k)",
            "Ours. Trained on reward_balance after D52/D53/D55, 500k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s5.zip"))),
    
    Rung("lstm_balance_600k_1M", "9f", "Recurrent PPO (reward_balance, 600k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 600k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s6.zip"))),
    
    Rung("lstm_balance_700k_1M", "9g", "Recurrent PPO (reward_balance, 700k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 700k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s7.zip"))),
        
    
    Rung("lstm_balance_800k_1M", "9h", "Recurrent PPO (reward_balance, 800k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 800k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s8.zip"))),
    
    Rung("lstm_balance_900k_1M", "9i", "Recurrent PPO (reward_balance, 900k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 900k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M_s9.zip"))),
    
    Rung("lstm_balance_1M", "9j", "Recurrent PPO (reward_balance, 1M)",
                "Ours. Trained on reward_balance after D52/D53/D55, 1M timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_1M.zip"))),





    Rung("lstm_balance_clean_100k_400k", "10a", "Recurrent PPO (reward_balance clean, 100k-400k)",
            "Ours. Trained on reward_balance after D52/D53/D55, 100k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/clean_lstm_s1.zip"))),

      Rung("lstm_balance_clean_200k_400k", "10b", "Recurrent PPO (reward_balance clean, 200k-400k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 200k out of 400k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/clean_lstm_s2.zip"))),

     Rung("lstm_balance_clean_300k_400k", "10c", "Recurrent PPO (reward_balance clean, 300k-400k)",
                    "Ours. Trained on reward_balance after D52/D53/D55, 300k out of 400k timesteps.",
                    _recurrent_ppo_rung_factory(Path("runs/checkpoints/clean_lstm_s3.zip"))),

     Rung("lstm_balance_clean_400k", "10d", "Recurrent PPO (reward_balance clean, 400k)",
                    "Ours. Trained on reward_balance after D52/D53/D55, 400k out of 400k timesteps.",
                    _recurrent_ppo_rung_factory(Path("runs/checkpoints/clean_lstm_s4.zip"))),





    Rung("lstm_balance_improved_100k_400k", "11a", "Recurrent PPO (reward_balance_improved, 100k-400k)",
            "Ours. Treatment arm of the D64 paired comparison: reward_balance_improved, "
            "100k out of 400k timesteps, same split/hyperparameters/seed as rung 10a.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_improved_s1.zip"))),

    Rung("lstm_balance_improved_200k_400k", "11b", "Recurrent PPO (reward_balance_improved, 200k-400k)",
            "Ours. Treatment arm of the D64 paired comparison: reward_balance_improved, "
            "200k out of 400k timesteps, same split/hyperparameters/seed as rung 10b.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_improved_s2.zip"))),

    Rung("lstm_balance_improved_300k_400k", "11c", "Recurrent PPO (reward_balance_improved, 300k-400k)",
            "Ours. Treatment arm of the D64 paired comparison: reward_balance_improved, "
            "300k out of 400k timesteps, same split/hyperparameters/seed as rung 10c.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_improved_s3.zip"))),

    Rung("lstm_balance_improved_400k", "11d", "Recurrent PPO (reward_balance_improved, 400k)",
            "Ours. Treatment arm of the D64 paired comparison: reward_balance_improved, "
            "400k out of 400k timesteps, same split/hyperparameters/seed as rung 10d.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/lstm_balance_improved_s4.zip"))),


    Rung("camper_oracle", "—", "Greedy static, truth-fed (D14's camper)",
         "Reference line: D14's camper, which knew where the pulses were.",
         lambda rng, grid: OracleCamper(grid, rng),
         deployable=False, needs_grid=True),
    Rung("oracle_pulse", "—", "Pulse-capture oracle",
         "Ceiling. Not a baseline -- a reference line; reads the truth grid.",
         lambda rng, grid: PulseCaptureOracle(grid, rng),
         deployable=False, needs_grid=True),
)

BY_KEY: dict[str, Rung] = {r.key: r for r in LADDER}

# The seven rungs that are schedulers. `oracle_pulse` is excluded by construction
# rather than by remembering to exclude it.
SCHEDULERS: tuple[str, ...] = tuple(r.key for r in LADDER if r.deployable)


def make(key: str, *, seed: int, grid=None):
    """Build one rung, fresh, for one episode.

    A fresh object every episode is D20 made mechanical: no threat library, no
    carry-over between scenarios, nothing to memorise. The RNG is derived from the
    episode seed and the rung's own name, so two rungs in the same episode do not
    share a random stream and a rung's draw does not depend on where it sits in
    the ladder.

    Returns a callable ready for `rollout.run_episode`. Deployable rungs come back
    wrapped in `guarded()`; the oracle does not, because it has the grid anyway
    and wrapping it would only hide that fact.
    """
    if key not in BY_KEY:
        raise KeyError(f"unknown baseline {key!r}; choose from {sorted(BY_KEY)}")
    spec = BY_KEY[key]
    if spec.needs_grid and grid is None:
        raise ValueError(f"{key} needs the truth grid: make({key!r}, seed=..., grid=grid)")

    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=int(seed), spawn_key=(_key_entropy(key),))
    )
    policy = spec.factory(rng, grid)
    return guarded(policy) if spec.deployable else policy


def _key_entropy(key: str) -> int:
    """A stable per-rung integer, so rung streams are independent but reproducible."""
    return int.from_bytes(key.encode(), "big") % (2 ** 31)


def band_at_slot_from_log(log: list[dict]) -> np.ndarray:
    """The band tuned at each slot, from an episode log. Used by the renders."""
    out = np.full(N_SLOTS, -1, dtype=np.int64)
    for row in log:
        out[row["slot"]] = row["band"]
    return out


def cycle_summary() -> dict:
    """What the two open-loop rungs actually spend, per band. For the write-up.

    Printed by `python -m rfenv.baselines`, so the D43 claim -- equal airtime
    against Turing's 2:1 weighting -- is a command's output rather than a
    sentence in a document.
    """
    sweep = np.zeros(N_BANDS, dtype=np.int64)
    for b in BAND_AT_SLOT:
        sweep[b] += 1
    rr = np.zeros(N_BANDS, dtype=np.int64)
    for b in EQUAL_AIRTIME_CYCLE:
        rr[b] += int(DWELL_SLOTS[b])
    return {
        "turing_sweep_cycle_slots": int(sum(DWELL_SLOTS)),
        "turing_sweep_slots_per_band_per_cycle": {
            "narrow": 1, "wide": 2,
        },
        "turing_sweep_episode_slots_per_band": sweep,
        "round_robin_cycle_slots": EQUAL_AIRTIME_CYCLE_SLOTS,
        "round_robin_slots_per_band_per_cycle": rr,
        "round_robin_cycle_s": EQUAL_AIRTIME_CYCLE_SLOTS * SLOT_S,
        "turing_sweep_cycle_s": float(sum(DWELL_SLOTS)) * SLOT_S,
    }
