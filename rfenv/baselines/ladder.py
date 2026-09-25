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
    obs_version: str = "v1"   # D30/D71: which ScanEnv(obs_version=...) this rung needs
    # This task: what this rung was itself trained/built with, so `compare.py` can
    # build its env correctly -- and, in `figures()`, decide whether to draw a
    # priority-marked animation -- without the caller having to pass matching
    # `--band-priority`/`--priority-uniform`/... flags by hand and risk a silent
    # mismatch. `band_priority=False` (the default) leaves every existing rung
    # exactly as it was; only a rung that actually needs the block sets it True.
    band_priority: bool = False
    priority_uniform: bool = False
    priority_coef: float = 0.5
    priority_n_bands: tuple[int, int] = (3, 6)
    # D74 follow-up: the elevated value and the decaying per-slot occupancy
    # term, both off/at-D74's-own-defaults unless a rung declares otherwise.
    priority_high: float = 3.0
    occupancy_coef: float = 0.0
    occupancy_decay_cap: float = 2.0


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
             _dqn_rung_factory(Path("runs/checkpoints/v1/deep_q_network_hit_y/deep_q_network_hit_y.zip"))),

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
_DQN_DEFAULT_CHECKPOINT = Path("runs/checkpoints/v1/deep_q_network/deep_q_network.zip")
_PPO_DEFAULT_CHECKPOINT = Path("runs/checkpoints/v1/ppo/ppo.zip")
_RECURRENT_PPO_DEFAULT_CHECKPOINT = Path("runs/checkpoints/v1/recurrent_ppo/recurrent_ppo.zip")


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
         _dqn_rung_factory(Path("runs/checkpoints/v1/deep_q_network_hit_y/deep_q_network_hit_y.zip"))),

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
            _ppo_rung_factory(Path("runs/checkpoints/v1/ppo_fi/ppo_fi.zip"))),

    Rung("ppo_first_intercept_100k", "8b", "PPO (first_intercept, 100k)",
            "Ours. Second naive pass: untuned SB3 PPO, trained on first_intercept (D29). "
            "Sees only the D34 observation.",
            _ppo_rung_factory(Path("runs/checkpoints/v1/ppo_fi/ppo_fi_100k.zip"))),

    Rung("ppo_first_intercept_2M", "8c", "PPO (first_intercept, 2M)",
                "Ours. Second naive pass: untuned SB3 PPO, trained on first_intercept (D29). "
                "Sees only the D34 observation.",
                _ppo_rung_factory(Path("runs/checkpoints/v1/ppo_fi/ppo_fi_2M.zip"))),
    # Commented out, not deleted: weighted_camp (the reward this was trained on)
    # was retired from REWARDS (env.py), and the observation shape has since
    # moved past what this checkpoint was trained against too. Re-enable only
    # after re-registering weighted_camp and retraining runs/checkpoints/v1/ppo_wt_cmp/ppo_wt_cmp_1M.zip.
    # Rung("ppo_weighted_camp", "8d", "PPO (weighted_camp, 1M)",
    #     "Ours. Second naive pass: untuned SB3 PPO, trained on weighted_camp (D29). "
    #     "Sees only the D34 observation.",
    #     _ppo_rung_factory(Path("runs/checkpoints/v1/ppo_wt_cmp/ppo_wt_cmp_1M.zip"))),
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
    #         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_ppo5/lstm_ppo5_100000_steps.zip"))),
    # Rung("recurrent_ppo_first_intercept_200k", "9b", "Recurrent PPO (LSTM),200k",
    #         "Ours. Third algorithm: sb3-contrib RecurrentPPO (MlpLstmPolicy), "
    #         "trained on hit_z (D29). Sees the same D34 observation as every "
    #         "other rung, but the policy carries an LSTM hidden state across "
    #         "the episode instead of acting on each look alone.",
    #         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_ppo5/lstm_ppo5_200000_steps.zip"))),
    # Rung("recurrent_ppo_first_intercept_300k", "9c", "Recurrent PPO (LSTM),300k",
    #         "Ours. Third algorithm: sb3-contrib RecurrentPPO (MlpLstmPolicy), "
    #         "trained on hit_z (D29). Sees the same D34 observation as every "
    #         "other rung, but the policy carries an LSTM hidden state across "
    #         "the episode instead of acting on each look alone.",
    #         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_ppo4/lstm_ppo4_300000_steps.zip"))),
    # Rung("recurrent_ppo_first_intercept_400k", "9d", "Recurrent PPO (LSTM),400k",
    #         "Ours. Third algorithm: sb3-contrib RecurrentPPO (MlpLstmPolicy), "
    #         "trained on hit_z (D29). Sees the same D34 observation as every "
    #         "other rung, but the policy carries an LSTM hidden state across "
    #         "the episode instead of acting on each look alone.",
    #         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_ppo4/lstm_ppo4_400000_steps.zip"))),


    Rung("lstm_balance_100k_1M", "9a", "Recurrent PPO (reward_balance, 100k)",
     "Ours. Trained on reward_balance after D52/D53/D55, 100k timesteps.",
     _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s1.zip"))),
    
    Rung("lstm_balance_200k_1M", "9b", "Recurrent PPO (reward_balance, 200k)",
     "Ours. Trained on reward_balance after D52/D53/D55, 200k timesteps.",
     _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s2.zip"))),

    Rung("lstm_balance_300k_1M", "9c", "Recurrent PPO (reward_balance, 300k)",
         "Ours. Trained on reward_balance after D52/D53/D55, 300k timesteps.",
         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s3.zip"))),

    Rung("lstm_balance_400k_1M", "9d", "Recurrent PPO (reward_balance, 400k)",
        "Ours. Trained on reward_balance after D52/D53/D55, 400k timesteps.",
        _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s4.zip"))),

    Rung("lstm_balance_500k_1M", "9e", "Recurrent PPO (reward_balance, 500k)",
            "Ours. Trained on reward_balance after D52/D53/D55, 500k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s5.zip"))),
    
    Rung("lstm_balance_600k_1M", "9f", "Recurrent PPO (reward_balance, 600k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 600k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s6.zip"))),
    
    Rung("lstm_balance_700k_1M", "9g", "Recurrent PPO (reward_balance, 700k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 700k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s7.zip"))),
        
    
    Rung("lstm_balance_800k_1M", "9h", "Recurrent PPO (reward_balance, 800k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 800k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s8.zip"))),
    
    Rung("lstm_balance_900k_1M", "9i", "Recurrent PPO (reward_balance, 900k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 900k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M_s9.zip"))),
    
    Rung("lstm_balance_1M", "9j", "Recurrent PPO (reward_balance, 1M)",
                "Ours. Trained on reward_balance after D52/D53/D55, 1M timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance/lstm_balance_1M.zip"))),





    Rung("lstm_balance_clean_100k_400k", "10a", "Recurrent PPO (reward_balance clean, 100k-400k)",
            "Ours. Trained on reward_balance after D52/D53/D55, 100k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/clean_lstm/clean_lstm_s1.zip"))),

      Rung("lstm_balance_clean_200k_400k", "10b", "Recurrent PPO (reward_balance clean, 200k-400k)",
                "Ours. Trained on reward_balance after D52/D53/D55, 200k out of 400k timesteps.",
                _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/clean_lstm/clean_lstm_s2.zip"))),

     Rung("lstm_balance_clean_300k_400k", "10c", "Recurrent PPO (reward_balance clean, 300k-400k)",
                    "Ours. Trained on reward_balance after D52/D53/D55, 300k out of 400k timesteps.",
                    _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/clean_lstm/clean_lstm_s3.zip"))),

     Rung("lstm_balance_clean_400k", "10d", "Recurrent PPO (reward_balance clean, 400k)",
                    "Ours. Trained on reward_balance after D52/D53/D55, 400k out of 400k timesteps.",
                    _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/clean_lstm/clean_lstm_s4.zip"))),





    Rung("lstm_balance_improved_100k_400k", "11a", "Recurrent PPO (reward_balance_improved, 100k-400k)",
            "Ours. Treatment arm of the D64 paired comparison: reward_balance_improved, "
            "100k out of 400k timesteps, same split/hyperparameters/seed as rung 10a.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_improved/lstm_balance_improved_s1.zip"))),

    Rung("lstm_balance_improved_200k_400k", "11b", "Recurrent PPO (reward_balance_improved, 200k-400k)",
            "Ours. Treatment arm of the D64 paired comparison: reward_balance_improved, "
            "200k out of 400k timesteps, same split/hyperparameters/seed as rung 10b.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_improved/lstm_balance_improved_s2.zip"))),

    Rung("lstm_balance_improved_300k_400k", "11c", "Recurrent PPO (reward_balance_improved, 300k-400k)",
            "Ours. Treatment arm of the D64 paired comparison: reward_balance_improved, "
            "300k out of 400k timesteps, same split/hyperparameters/seed as rung 10c.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_improved/lstm_balance_improved_s3.zip"))),

    Rung("lstm_balance_improved_400k", "11d", "Recurrent PPO (reward_balance_improved, 400k)",
            "Ours. Treatment arm of the D64 paired comparison: reward_balance_improved, "
            "400k out of 400k timesteps, same split/hyperparameters/seed as rung 10d.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_improved/lstm_balance_improved_s4.zip"))),


    # D67: the observation gains hit_streak, 146 -> 183. Every rung above this
    # point trained against the narrower vector and is now permanently
    # unloadable (D49's rule, paid again). Rungs 14/15 are the first trained
    # against the current 183-wide observation.
    Rung("lstm_balance_d67_control_100k_400k", "14a", "Recurrent PPO (reward_balance, D67 obs, 100k-400k)",
            "Ours. Control arm retrained under D67's 183-wide observation (hit_streak), "
            "100k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control/lstm_balance_d67_control_s1.zip"))),

    Rung("lstm_balance_d67_control_200k_400k", "14b", "Recurrent PPO (reward_balance, D67 obs, 200k-400k)",
            "Ours. Control arm retrained under D67's 183-wide observation (hit_streak), "
            "200k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control/lstm_balance_d67_control_s2.zip"))),

    Rung("lstm_balance_d67_control_300k_400k", "14c", "Recurrent PPO (reward_balance, D67 obs, 300k-400k)",
            "Ours. Control arm retrained under D67's 183-wide observation (hit_streak), "
            "300k out of 400k timesteps -- the single-seed D61 pick (+25.0% net dominance); "
            "superseded by rung 17c (+36.1%) once seeds 1/2 were added for D47/D68's re-run.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control/lstm_balance_d67_control_s3.zip"))),

    Rung("lstm_balance_d67_control_400k", "14d", "Recurrent PPO (reward_balance, D67 obs, 400k)",
            "Ours. Control arm retrained under D67's 183-wide observation (hit_streak), "
            "400k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control/lstm_balance_d67_control_s4.zip"))),

    Rung("lstm_balance_d67_treatment_100k_400k", "15a", "Recurrent PPO (reward_balance_improved, D67 obs, 100k-400k)",
            "Ours. Treatment arm retrained under D67's 183-wide observation (hit_streak), "
            "100k out of 400k timesteps -- the D61-selected checkpoint, both before and after "
            "seeds 1/2 were added for D47/D68's re-run (+33.3% net dominance either way).",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment/lstm_balance_d67_treatment_s1.zip"))),

    Rung("lstm_balance_d67_treatment_200k_400k", "15b", "Recurrent PPO (reward_balance_improved, D67 obs, 200k-400k)",
            "Ours. Treatment arm retrained under D67's 183-wide observation (hit_streak), "
            "200k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment/lstm_balance_d67_treatment_s2.zip"))),

    Rung("lstm_balance_d67_treatment_300k_400k", "15c", "Recurrent PPO (reward_balance_improved, D67 obs, 300k-400k)",
            "Ours. Treatment arm retrained under D67's 183-wide observation (hit_streak), "
            "300k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment/lstm_balance_d67_treatment_s3.zip"))),

    Rung("lstm_balance_d67_treatment_400k", "15d", "Recurrent PPO (reward_balance_improved, D67 obs, 400k)",
            "Ours. Treatment arm retrained under D67's 183-wide observation (hit_streak), "
            "400k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment/lstm_balance_d67_treatment_s4.zip"))),


    # D68's escalation (1.2 pp gap, inside D47's 5 pp margin) prompted a matched-seed
    # retrain: 2 more seeds per arm, run strictly one at a time after the crash. Rungs
    # 16/17 are the control arm's seeds 1/2, rungs 18/19 the treatment arm's -- same
    # hyperparameters and D67 observation as 14/15, only --seed differs. D61 is run
    # across all 12 checkpoints per arm (rungs 14+16+17, and 15+18+19); whichever
    # checkpoint wins gets used for D47's re-application, whether or not it is one
    # of these eight.
    Rung("lstm_balance_d67_control_seed1_100k_400k", "16a", "Recurrent PPO (reward_balance, D67 obs, seed 1, 100k-400k)",
            "Ours. Control arm, matched-seed retrain for D47/D68, seed 1, "
            "100k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control_seed1/lstm_balance_d67_control_seed1_s1.zip"))),

    Rung("lstm_balance_d67_control_seed1_200k_400k", "16b", "Recurrent PPO (reward_balance, D67 obs, seed 1, 200k-400k)",
            "Ours. Control arm, matched-seed retrain for D47/D68, seed 1, "
            "200k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control_seed1/lstm_balance_d67_control_seed1_s2.zip"))),

    Rung("lstm_balance_d67_control_seed1_300k_400k", "16c", "Recurrent PPO (reward_balance, D67 obs, seed 1, 300k-400k)",
            "Ours. Control arm, matched-seed retrain for D47/D68, seed 1, "
            "300k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control_seed1/lstm_balance_d67_control_seed1_s3.zip"))),

    Rung("lstm_balance_d67_control_seed1_400k", "16d", "Recurrent PPO (reward_balance, D67 obs, seed 1, 400k)",
            "Ours. Control arm, matched-seed retrain for D47/D68, seed 1, "
            "400k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control_seed1/lstm_balance_d67_control_seed1_s4.zip"))),

    Rung("lstm_balance_d67_control_seed2_100k_400k", "17a", "Recurrent PPO (reward_balance, D67 obs, seed 2, 100k-400k)",
            "Ours. Control arm, matched-seed retrain for D47/D68, seed 2, "
            "100k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control_seed2/lstm_balance_d67_control_seed2_s1.zip"))),

    Rung("lstm_balance_d67_control_seed2_200k_400k", "17b", "Recurrent PPO (reward_balance, D67 obs, seed 2, 200k-400k)",
            "Ours. Control arm, matched-seed retrain for D47/D68, seed 2, "
            "200k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control_seed2/lstm_balance_d67_control_seed2_s2.zip"))),

    Rung("lstm_balance_d67_control_seed2_300k_400k", "17c", "Recurrent PPO (reward_balance, D67 obs, seed 2, 300k-400k)",
            "Ours. Control arm, matched-seed retrain for D47/D68, seed 2, "
            "300k out of 400k timesteps -- the D61-selected checkpoint once all 12 "
            "control-arm checkpoints (seeds 0/1/2) were scored together (+36.1% net dominance, "
            "ahead of seed 0's own 300k at +25.0%, rung 14c).",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control_seed2/lstm_balance_d67_control_seed2_s3.zip"))),

    Rung("lstm_balance_d67_control_seed2_400k", "17d", "Recurrent PPO (reward_balance, D67 obs, seed 2, 400k)",
            "Ours. Control arm, matched-seed retrain for D47/D68, seed 2, "
            "400k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_control_seed2/lstm_balance_d67_control_seed2_s4.zip"))),

    Rung("lstm_balance_d67_treatment_seed1_100k_400k", "18a", "Recurrent PPO (reward_balance_improved, D67 obs, seed 1, 100k-400k)",
            "Ours. Treatment arm, matched-seed retrain for D47/D68, seed 1, "
            "100k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment_seed1/lstm_balance_d67_treatment_seed1_s1.zip"))),

    Rung("lstm_balance_d67_treatment_seed1_200k_400k", "18b", "Recurrent PPO (reward_balance_improved, D67 obs, seed 1, 200k-400k)",
            "Ours. Treatment arm, matched-seed retrain for D47/D68, seed 1, "
            "200k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment_seed1/lstm_balance_d67_treatment_seed1_s2.zip"))),

    Rung("lstm_balance_d67_treatment_seed1_300k_400k", "18c", "Recurrent PPO (reward_balance_improved, D67 obs, seed 1, 300k-400k)",
            "Ours. Treatment arm, matched-seed retrain for D47/D68, seed 1, "
            "300k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment_seed1/lstm_balance_d67_treatment_seed1_s3.zip"))),

    Rung("lstm_balance_d67_treatment_seed1_400k", "18d", "Recurrent PPO (reward_balance_improved, D67 obs, seed 1, 400k)",
            "Ours. Treatment arm, matched-seed retrain for D47/D68, seed 1, "
            "400k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment_seed1/lstm_balance_d67_treatment_seed1_s4.zip"))),

    Rung("lstm_balance_d67_treatment_seed2_100k_400k", "19a", "Recurrent PPO (reward_balance_improved, D67 obs, seed 2, 100k-400k)",
            "Ours. Treatment arm, matched-seed retrain for D47/D68, seed 2, "
            "100k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment_seed2/lstm_balance_d67_treatment_seed2_s1.zip"))),

    Rung("lstm_balance_d67_treatment_seed2_200k_400k", "19b", "Recurrent PPO (reward_balance_improved, D67 obs, seed 2, 200k-400k)",
            "Ours. Treatment arm, matched-seed retrain for D47/D68, seed 2, "
            "200k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment_seed2/lstm_balance_d67_treatment_seed2_s2.zip"))),

    Rung("lstm_balance_d67_treatment_seed2_300k_400k", "19c", "Recurrent PPO (reward_balance_improved, D67 obs, seed 2, 300k-400k)",
            "Ours. Treatment arm, matched-seed retrain for D47/D68, seed 2, "
            "300k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment_seed2/lstm_balance_d67_treatment_seed2_s3.zip"))),

    Rung("lstm_balance_d67_treatment_seed2_400k", "19d", "Recurrent PPO (reward_balance_improved, D67 obs, seed 2, 400k)",
            "Ours. Treatment arm, matched-seed retrain for D47/D68, seed 2, "
            "400k out of 400k timesteps.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v1/lstm_balance_d67_treatment_seed2/lstm_balance_d67_treatment_seed2_s4.zip"))),

    # D30/D71: first checkpoint trained on the 326-wide "v2" observation
    # (PulseWidth/AoA/per-band amplitude). Otherwise mirrors
    # lstm_balance_d67_control_seed2 (rung 17) exactly -- same reward, split,
    # hyperparameters, seed -- so this is comparable to that run's own
    # snapshots. Only usable against a ScanEnv built with obs_version="v2";
    # require_loadable() (common.py) now accepts either known width, but the
    # env itself must still match or predict() raises mid-episode.
    Rung("lstm_balance_v2_control_seed2_100k_400k", "20a", "Recurrent PPO (reward_balance, D71 v2 obs, seed 2, 100k)",
            "Ours. v2-observation control run, seed 2, 100k out of 400k timesteps. "
            "Needs ScanEnv(obs_version='v2').",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2/lstm_balance_v2_control_seed2/lstm_balance_v2_control_seed2_s1.zip")),
            obs_version="v2"),

    Rung("lstm_balance_v2_control_seed2_200k_400k", "20b", "Recurrent PPO (reward_balance, D71 v2 obs, seed 2, 200k)",
            "Ours. v2-observation control run, seed 2, 200k out of 400k timesteps. "
            "Needs ScanEnv(obs_version='v2').",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2/lstm_balance_v2_control_seed2/lstm_balance_v2_control_seed2_s2.zip")),
            obs_version="v2"),

    Rung("lstm_balance_v2_control_seed2_300k_400k", "20c", "Recurrent PPO (reward_balance, D71 v2 obs, seed 2, 300k)",
            "Ours. v2-observation control run, seed 2, 300k out of 400k timesteps. "
            "Needs ScanEnv(obs_version='v2').",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2/lstm_balance_v2_control_seed2/lstm_balance_v2_control_seed2_s3.zip")),
            obs_version="v2"),

    # D72: the seed-2 control/treatment pair above (rung 20a-c) trained on the
    # 326-wide "v2" that predates pulse_count and never finished (385,024/400,000
    # steps) -- D72's width bump to 362 made those three snapshots permanently
    # unloadable (`known_observation_widths()` no longer contains 326 at all),
    # so this is not a continuation of that run, it is a fresh retrain on the
    # widened layout. Mirrors D71's own pairing: same split, hyperparameters
    # (ent_coef=0.01, gamma=0.997, n_steps=8192) and seed (2), only the reward
    # differs between the two rungs below, per D65's paired-comparison method.
    Rung("lstm_balance_v2_d72_seed2_400k", "20d", "Recurrent PPO (reward_balance, D72 v2 obs, seed 2, 400k)",
            "Ours. D72-observation (362-wide, pulse_count added) control run, seed 2, "
            "401,408 out of 400k timesteps, completed in one uninterrupted run. "
            "Needs ScanEnv(obs_version='v2').",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2/lstm_balance_v2_d72_seed2/lstm_balance_v2_d72_seed2.zip")),
            obs_version="v2"),

    Rung("lstm_balance_improved_v2_d72_seed2_400k", "21a", "Recurrent PPO (reward_balance_improved_v2, D72 v2 obs, seed 2, 400k)",
            "Ours. Paired against lstm_balance_v2_d72_seed2_400k (rung 20d) -- same "
            "split, hyperparameters and seed, only the reward differs (D65's "
            "methodology). 404,800 out of 400k timesteps; interrupted by two laptop "
            "crashes and resumed both times from its own last checkpoint-freq snapshot "
            "(RecurrentPPO.load + learn(reset_num_timesteps=False)), recorded in the "
            "checkpoint's own manifest description. Needs ScanEnv(obs_version='v2').",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2/lstm_balance_improved_v2_d72_seed2/lstm_balance_improved_v2_d72_seed2.zip")),
            obs_version="v2"),

    # This task: band-priority reward, additive on top of reward_balance, not a
    # new REWARDS candidate (env.py's step()). New "v2p" layout (398-wide, D71's
    # "v2" plus band_priority) -- both rungs need ScanEnv(obs_version="v2p").
    # `band_priority`/`priority_uniform`/`priority_coef`/`priority_n_bands` are
    # each set below, per rung (`Rung`'s own fields, added this task) -- so
    # `compare.py` builds each one's env correctly on its own, without the
    # caller having to pass matching `--band-priority`/`--priority-uniform`
    # flags by hand and risk silently evaluating the treatment checkpoint under
    # an all-ones vector. The two can now sit in the same `compare.py`
    # invocation and the same figure; `--band-priority` etc. remain available
    # as a manual override for a rung that doesn't declare its own (every rung
    # above this comment).
    Rung("lstm_balance_v2p_priority_seed2_400k", "22a", "Recurrent PPO (reward_balance, v2p+priority, seed 2, 400k)",
            "Ours. Treatment arm: band_priority=True, priority_uniform=False -- 3-6 of "
            "36 bands elevated to priority 3.0 each episode, real signal to condition "
            "on. Same split/hyperparameters/seed as the D72 pair (rungs 20d/21a). "
            "Needs ScanEnv(obs_version='v2p', band_priority=True).",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2p/lstm_balance_v2p_priority_seed2/lstm_balance_v2p_priority_seed2.zip")),
            obs_version="v2p", band_priority=True, priority_uniform=False,
            priority_coef=0.5, priority_n_bands=(3, 6)),

    Rung("lstm_balance_v2p_uniform_seed2_400k", "22b", "Recurrent PPO (reward_balance, v2p+priority, control, seed 2, 400k)",
            "Ours. Control arm, paired against rung 22a: identical in every other "
            "respect, priority_uniform=True -- band_priority stays all-ones every "
            "episode (same reward-term scale, no real signal to use). Isolates "
            "whether rung 22a's behaviour (if different) comes from reading the "
            "priority vector or just the reward-scale increase the term adds "
            "uniformly. Needs ScanEnv(obs_version='v2p', band_priority=True, "
            "priority_uniform=True).",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2p/lstm_balance_v2p_uniform_seed2/lstm_balance_v2p_uniform_seed2.zip")),
            obs_version="v2p", band_priority=True, priority_uniform=True,
            priority_coef=0.5, priority_n_bands=(3, 6)),

    # D74 follow-up: stronger discovery bonus (priority_coef 0.5->2.0,
    # priority_high 3.0->5.0), plus the new decaying occupancy term
    # (priority_reward_bonus, env.py) -- occupancy_coef/occupancy_decay_cap
    # are Rung fields too now, resolved by compare.py the same automatic way
    # as the other priority fields. LSTM hidden size (512, doubled) is not a
    # Rung field -- RecurrentPPO.load() reconstructs the policy architecture
    # from the checkpoint's own saved policy_kwargs, nothing to declare here.
    Rung("lstm_balance_v2p_priority_strong_seed2_800k", "23a", "Recurrent PPO (reward_balance, v2p+priority strengthened, seed 2, 800k)",
            "Ours. Treatment arm, D74 follow-up: priority_coef=2.0, priority_high=5.0, "
            "occupancy_coef=0.3, occupancy_decay_cap=6.0, 512-wide LSTM (doubled), "
            "800k timesteps (doubled) -- testing whether D74's null result was a "
            "signal-strength/capacity/budget limitation. Paired against "
            "lstm_balance_v2p_uniform_strong_seed2_800k (rung 23b). "
            "Needs ScanEnv(obs_version='v2p', band_priority=True).",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2p/lstm_balance_v2p_priority_strong_seed2/lstm_balance_v2p_priority_strong_seed2.zip")),
            obs_version="v2p", band_priority=True, priority_uniform=False,
            priority_coef=2.0, priority_n_bands=(3, 6), priority_high=5.0,
            occupancy_coef=0.3, occupancy_decay_cap=6.0),

    Rung("lstm_balance_v2p_uniform_strong_seed2_800k", "23b", "Recurrent PPO (reward_balance, v2p+priority strengthened, control, seed 2, 800k)",
            "Ours. Control arm, paired against rung 23a: identical except "
            "priority_uniform=True. Needs ScanEnv(obs_version='v2p', "
            "band_priority=True, priority_uniform=True).",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2p/lstm_balance_v2p_uniform_strong_seed2/lstm_balance_v2p_uniform_strong_seed2.zip")),
            obs_version="v2p", band_priority=True, priority_uniform=True,
            priority_coef=2.0, priority_n_bands=(3, 6), priority_high=5.0,
            occupancy_coef=0.3, occupancy_decay_cap=6.0),

    # D75 matched pair: "v3" (treatment -- prev_reward) vs "v2p"
    # (control, the extra block absent). Neither arm's --band-priority flag
    # was set during training, so `_band_priority` stays at its reset()
    # default (all ones) in both -- no confound with D74's own mechanism,
    # which is off here. Same base hyperparameters as 23a/23b (512-wide LSTM,
    # n_steps=8192, 800k steps), trained strictly sequentially so the two arms
    # of a seed never compete for the GPU with each other.
    #
    # **Rung 24a's own checkpoint is dead, twice over now.** "v3" was 436-wide
    # (a third block, `prev_action`) when this checkpoint trained; `prev_action`
    # was removed from "v3" the next day once confirmed bit-identical to
    # `current_band` (D75's amendment), narrowing "v3" to 400-wide, then
    # `prev_hit` was removed too, the same day, narrowing it again to 399-wide
    # (D75's second amendment) -- the same class of width change D49/D55/D67/
    # D72 made before it. `require_loadable` now refuses this file by name,
    # correctly and loudly. Left registered rather than deleted, matching how
    # every other width casualty in this file stays registered as a record of
    # what was measured, not pruned. No checkpoint was ever successfully
    # trained on the 400-wide shape in between -- the second amendment costs
    # nothing additional. Retraining on the current 399-wide "v3" would need a
    # fresh checkpoint at this path.
    Rung("lstm_v3_seed0", "24a", "Recurrent PPO (reward_balance, D75 v3 obs, seed 0, 800k)",
            "Ours. Treatment arm, D75: obs_version='v3' -- prev_reward "
            "appended to v2p. Paired against lstm_v2p_ctrl_seed0 (rung 24b), "
            "identical except obs_version. Needs ScanEnv(obs_version='v3'). "
            "DEAD as of D75's amendments (2026-09-20): trained on the 436-wide "
            "'v3' that existed before prev_action and prev_hit were removed; "
            "unloadable now.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v3/lstm_v3_seed0/lstm_v3_seed0.zip")),
            obs_version="v3"),

    Rung("lstm_v2p_ctrl_seed0", "24b", "Recurrent PPO (reward_balance, v2p obs, D75 control, seed 0, 800k)",
            "Ours. Control arm, paired against rung 24a: identical except "
            "obs_version='v2p' (no prev_reward block). Needs "
            "ScanEnv(obs_version='v2p'). Unaffected by the 'v3' width changes "
            "-- this checkpoint is on 'v2p', still loadable.",
            _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2p/lstm_v2p_ctrl_seed0/lstm_v2p_ctrl_seed0.zip")),
            obs_version="v2p"),


    # D78 matched pair: architecture, not observation. Identical to rung 24b
    # in every training input (reward_balance, obs_version='v2p', seed 0,
    # ent_coef=0.01, gamma=0.997, n_steps=8192, 512-wide LSTM, 800k steps) --
    # differs only in --policy. Reuses 24b as the control arm rather than
    # retraining a duplicate baseline.
    Rung("lstm_v2p_mlpfeature_seed0", "25a", "Recurrent PPO (reward_balance, v2p obs, MlpFeatureLstmPolicy, seed 0, 800k)",
         "Ours. Treatment arm, D78: --policy MlpFeatureLstmPolicy (obs -> 2-layer "
         "LayerNorm MLP -> LSTM -> actor/critic, rfenv/rl/policies.py) in place of "
         "the library-default MlpLstmPolicy (obs -> LSTM -> actor/critic). Paired "
         "against lstm_v2p_ctrl_seed0 (rung 24b), identical except --policy. Needs "
         "ScanEnv(obs_version='v2p'); RecurrentPPO.load() reconstructs the policy "
         "class from the checkpoint's own saved data, nothing extra to declare here.",
         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2p/lstm_v2p_mlpfeature_seed0/lstm_v2p_mlpfeature_seed0.zip")),
         obs_version="v2p"),

    # D78 convergence check: the 600k-step checkpoint-freq snapshot of rung 25a,
    # not the final 800k one. A quick 12-scenario probe across the run's own
    # snapshots (s4/s8/s12/s16 = 200k/400k/600k/800k) found no clean upward
    # trend late in training -- s12 scored marginally best of the four on that
    # small sample. Registered so it can be scored properly (full development
    # set, matched seeds) rather than trusted off 12 scenarios.
    Rung("lstm_v2p_mlpfeature_seed0_s12", "25b", "Recurrent PPO (reward_balance, v2p obs, MlpFeatureLstmPolicy, seed 0, 600k snapshot)",
         "Ours. Same run as rung 25a, checkpoint-freq snapshot at 600k steps "
         "instead of the final 800k -- see D78's convergence-check note: training "
         "reward flattened by ~80k steps and stayed flat, and evaluation metrics "
         "across snapshots do not climb steadily toward 800k, so this is a check "
         "on whether an earlier snapshot happens to score better, not a claim "
         "that it should.",
         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2p/lstm_v2p_mlpfeature_seed0/lstm_v2p_mlpfeature_seed0_s12.zip")),
         obs_version="v2p"),

    # D79 x D75's second amendment: BandEncoderLstmPolicy on the narrowed "v3"
    # (399-wide, prev_reward only), band_priority off (reset default, all-ones,
    # no real signal -- not a matched pair with rung 23a, which has real
    # priority active; this isolates architecture+prev_reward from priority
    # entirely). Otherwise mirrors 23a's own scale exactly: 512-wide LSTM,
    # n_steps=8192, 800k steps, reward_balance, seed 2.
    Rung("lstm_v3_bandenc_noprior_seed2", "26a", "Recurrent PPO (reward_balance, D75-narrowed v3 obs, BandEncoderLstmPolicy, no priority, seed 2, 800k)",
         "Ours. D79's BandEncoderLstmPolicy (obs -> shared per-band encoder -> mean "
         "pool -> concat global -> LSTM -> actor/critic) trained on 'v3' after its "
         "second narrowing (prev_reward only, no prev_hit -- D75's amendment), "
         "band_priority left at its uninformative reset default. First pass: does "
         "prev_reward, read through a per-band-structured encoder, train well "
         "before real priority is added back in. Needs ScanEnv(obs_version='v3').",
         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v3/lstm_v3_bandenc_noprior_seed2/lstm_v3_bandenc_noprior_seed2.zip")),
         obs_version="v3"),

    # Second seed of 26a -- identical config, seed 0 instead of 2. Confirming
    # whether 26a's result (66.7% beats-recency-both) holds across seeds
    # before building real band_priority on top of it.
    Rung("lstm_v3_bandenc_noprior_seed0", "26b", "Recurrent PPO (reward_balance, D75-narrowed v3 obs, BandEncoderLstmPolicy, no priority, seed 0, 800k)",
         "Ours. Same as rung 26a in every respect except seed (0, not 2). "
         "Needs ScanEnv(obs_version='v3').",
         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v3/lstm_v3_bandenc_noprior_seed0/lstm_v3_bandenc_noprior_seed0.zip")),
         obs_version="v3"),

    Rung("lstm_v2p_noprior_online_scratch_seed2", "27", "Recurrent PPO (reward_balance, v2p, no priority, trained FROM SCRATCH online, seed 2, 800k)",
         "Ours. The first policy in this repository trained entirely through D77's "
         "online regime rather than offline `train()`: continuous-grid missions "
         "(120 x 600 slots) instead of 30 s episodes, and the gradient fed "
         "`reward_balance_obs` (Y-derived, what a real receiver could compute) "
         "instead of the truth-reading `reward_balance` every other rung trained "
         "on. Scored on `reward_balance` like everything else, so the yardstick is "
         "unchanged. Matched to rung 23a on observation width (398), LSTM size "
         "(512), n_steps/batch/epochs/gamma/ent_coef, seed and budget, with "
         "band_priority stripped entirely -- which also tests the open thread "
         "MODEL_COMPARISON.md flags, that 23a's lead may be capacity and budget "
         "rather than a priority mechanism two ablations showed it never reads. "
         "Needs ScanEnv(obs_version='v2p').",
         _recurrent_ppo_rung_factory(Path("runs/checkpoints/v2p/lstm_v2p_noprior_online_scratch_seed2/lstm_v2p_noprior_online_scratch_seed2.zip")),
         obs_version="v2p"),

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
    # Sized from the log itself, not the constant, so a stitched mission (D76)
    # renders too. Identical for a 600-slot episode, which covers every slot.
    n_slots = max((row["slot"] for row in log), default=N_SLOTS - 1) + 1
    out = np.full(max(n_slots, N_SLOTS), -1, dtype=np.int64)
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
