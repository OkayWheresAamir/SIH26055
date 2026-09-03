"""L3 — the agent interface.

A standard Gymnasium environment wrapping L0–L2. The scheduler's whole job, and
its whole world:

    action       choose one of 36 bands
    observation  what its own scan history has taught it -- nothing else
    reward       one of three candidates, all of which may read truth
    time         advances by the chosen band's native dwell, 1 or 2 slots

Three properties of this interface are load-bearing and none of them is a
Gymnasium convention, so they are worth saying out loud.

**The episode is 600 slots, not 600 steps.** Seven bands carry a 100 ms dwell and
consume two slots (D3, D16), so an episode runs 300 to 600 `step()` calls
depending on what the agent picks, while always covering exactly 30 s. Airtime is
the only currency -- retuning is free, measured (D31) -- so a wide band genuinely
costs twice as much of the episode as a narrow one.

**The observation ships; the reward does not.** Training is offline and the policy
is frozen before deployment, so the reward is a training-time construct and may
read `Z`, per-emitter levels, `first_e`, anything (D29). The observation may not:
it is built only from the agent's own scan history, because that is all a fielded
receiver has (D19, D20, D34). This asymmetry is deliberate and is the reason
`info` carries truth while the observation vector carries none of it.

**Every episode starts cold.** No threat library, no carry-over between scenarios,
no prior on who transmits where (D20) -- the problem statement's *"absence of
prior reliable intelligence"* made literal. It is also what keeps the
explore/exploit tension real: with a prior, camping would be defensible.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rfenv.constants import (
    EPISODE_S,
    GAMMA_DBM,
    N_BANDS,
    N_SLOTS,
    NOISE_SIGMA_DB,
    SLOT_S,
)
from rfenv.receiver import DwellResult, Receiver
from rfenv.scenario import EmitterPool, Scenario
from rfenv.truth import TruthGrid

# --------------------------------------------------------------------------- #
# Reward candidates (D7, D29)
# --------------------------------------------------------------------------- #
#
# Exactly three, and the set stays at three (D29): every candidate scored on the
# same 47 scenarios is another draw, and best-of-many is partly selection noise.
#
# All three are **per slot**, and a dwell's reward is the sum over its slots
# (D31), so a 100 ms dwell can earn up to +2. That keeps reward per unit time
# equal across wide and narrow bands. The rejected alternative -- +1 per dwell
# regardless of length -- would make the seven wide bands strictly dominated, and
# those are measurably the bands holding the densest emitter populations and the
# slowest rotators. It would also be the only per-dwell quantity in a project
# whose every judging metric is per-slot or per-illumination.
#
# D28 puts the two headline metrics on opposite sides of the Y/Z line -- censored
# intercept time needs Y = 1, interception ratio does not -- so no single reward
# is aligned with both. That is why there are three rather than one, and why the
# rule for choosing between them is a human decision recorded as open in D29.
# Nothing here ranks them.


def reward_hit_z(dwell: DwellResult, newly: set[int]) -> float:
    """Candidate 1, the default: +1 per **true** hit, cell-level `Z` (D5).

    Values every occupied cell equally, whatever its level, so it prices
    opportunity rather than detectability. Predicted to favour interception ratio,
    which is itself the threshold-free opportunity metric.
    """
    return float(dwell.Z.sum())


def reward_hit_y(dwell: DwellResult, newly: set[int]) -> float:
    """Candidate 2: +1 per **declared** hit `Y` -- what the receiver actually said.

    Weights cells by loudness, since `P(Y=1) = Phi((S-gamma)/sigma)` runs from 0.5
    at `S = gamma` upward, and so teaches the agent that a marginal emitter needs
    repeated looks before it yields a declaration. Under D28 that is exactly what
    lowers `first_e`. Predicted to favour censored intercept time.

    It also pays out on false alarms. That cannot be helped and should not be
    penalised into submission: Pfa is a frozen receiver property and no reward can
    move it (D15, D21). At the operating point it is 1.35e-3 anyway -- about 0.8
    spurious reward over a whole episode of empty looks.
    """
    return float(dwell.Y.sum())


def reward_first_intercept(dwell: DwellResult, newly: set[int]) -> float:
    """Candidate 3: +1 per emitter intercepted for the **first** time (D28).

    The discovery weighting D5 flagged as open; D28 gives it the precise
    definition it lacked. Unaffected by the per-slot rule -- an emitter is
    first-intercepted once, whichever slot of the dwell delivers it.

    Reads truth, which is allowed (D29) and necessary: novelty is not something
    binary hit/miss can perceive. Whether a policy can *learn* to act on a signal
    it cannot observe is the open question D30 raises; it does not block training.
    """
    return float(len(newly))


REWARDS = {
    "hit_z": reward_hit_z,
    "hit_y": reward_hit_y,
    "first_intercept": reward_first_intercept,
}
DEFAULT_REWARD = "hit_z"


# --------------------------------------------------------------------------- #

class ScanEnv(gym.Env):
    """The scan-scheduling environment.

    Built fresh to `ENVIRONMENT_SPEC.md` §L3. `docs/teammate-work/rf_env_grounded.py`
    reached a near-identical interface independently from the same PS text, which
    is corroboration that the shape is right; nothing is inherited from it.

    Give it a `scenario` for a fixed world (what the validation gates use), or a
    `pool` to draw a fresh one each reset (what RL trains on, so no episode
    repeats and there is nothing to memorise -- D25, D32).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenario: Scenario | None = None,
        pool: EmitterPool | None = None,
        reward: str = DEFAULT_REWARD,
        gamma_dbm: float = GAMMA_DBM,
        sigma_db: float = NOISE_SIGMA_DB,
    ):
        super().__init__()
        if scenario is None and pool is None:
            raise ValueError("ScanEnv needs a `scenario` to replay or a `pool` to sample from")
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {sorted(REWARDS)}, got {reward!r}")

        self._scenario = scenario
        self._pool = pool
        self.reward_name = reward
        self._reward_fn = REWARDS[reward]
        self.gamma = float(gamma_dbm)
        self.sigma = float(sigma_db)

        self.action_space = spaces.Discrete(N_BANDS)
        # 36 x 3 + 1 = 109 (D34). Every component is natively a fraction, so the
        # box is the unit interval and no scaling layer is needed anywhere.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_BANDS * 3 + 1,), dtype=np.float32
        )

        self.grid: TruthGrid | None = None
        self.receiver: Receiver | None = None
        self.log: list[dict] = []

    # ----------------------------------------------------------------- reset --

    def reset(self, seed=None, options=None):
        """Start an episode. `options={"scenario": ...}` overrides for this one.

        Seeding seeds *everything* stochastic: the scenario draw and the
        receiver's noise both run off `self.np_random`, so a seed reproduces an
        episode exactly -- which is what `EVALUATION.md` §7 means by comparing
        schedulers "on identical scenarios and seeds".
        """
        super().reset(seed=seed)

        scenario = (options or {}).get("scenario") or self._scenario
        if scenario is None:
            scenario = Scenario.sample(self._pool, self.np_random)
        self.scenario = scenario
        self.grid = TruthGrid.from_scenario(scenario)
        self.receiver = Receiver(self.gamma, self.sigma, rng=self.np_random)

        self.t = 0
        self._slots_looked = np.zeros(N_BANDS, dtype=np.int64)
        self._hits = np.zeros(N_BANDS, dtype=np.int64)
        self._last_slot = np.full(N_BANDS, -1, dtype=np.int64)

        # Evaluator-side episode state. E is the coverage denominator: emitters
        # with a non-empty detectable interval, not every transmitter in the
        # metadata (D27) -- 19% of train transmitters are never detectable at all
        # and scoring a scheduler for missing those would be meaningless.
        self.detectable = {
            int(i): self.grid.detectable_interval(int(i), self.gamma)
            for i in self.grid.detectable_emitters(self.gamma)
        }
        self.tracks: dict[int, dict] = {}
        self.pulses_intercepted = 0
        self.total_reward = 0.0
        self.n_steps = 0
        self.log = []

        return self._observation(), self._info()

    # ------------------------------------------------------------------ step --

    def step(self, action):
        """Tune to `action` for that band's native dwell; advance the clock by it."""
        if self.grid is None:
            raise RuntimeError("call reset() before step()")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} outside 0..{N_BANDS - 1}")
        if self.t >= N_SLOTS:
            raise RuntimeError("episode is over; call reset()")

        dwell = self.receiver.dwell(self.grid, action, self.t)

        # Per-emitter tracks, per D28. Slots inside a dwell arrive in order, so
        # the first sighting recorded for an emitter is its earliest.
        #
        # Every intercept is kept, not only the first. `first` alone answers
        # censored intercept time, but EVALUATION.md §8 artefact 2 also wants last
        # intercept, count and bands-seen-in, and none of those survives in the
        # per-slot episode log -- the log carries no emitter attribution. The
        # alternative was to re-derive them in `metrics.py` from `measured_dbm`,
        # which would put D28's three-clause rule in two places. Measured: under
        # round-robin, 92.8% of config_2's intercept events and 94.6% of
        # config_921's are not first sightings, so this is most of the signal.
        newly: set[int] = set()
        for emitter, slot in dwell.intercepts:
            track = self.tracks.get(emitter)
            if track is None:
                self.tracks[emitter] = {
                    "first": slot, "last": slot, "count": 1, "bands": {dwell.band},
                }
                newly.add(emitter)
            else:
                track["last"] = slot
                track["count"] += 1
                track["bands"].add(dwell.band)

        reward = float(self._reward_fn(dwell, newly))

        self._slots_looked[action] += dwell.n_slots
        self._hits[action] += int(dwell.Y.sum())
        self._last_slot[action] = dwell.slot0 + dwell.n_slots - 1
        self.pulses_intercepted += dwell.pulses
        self.total_reward += reward
        self.n_steps += 1
        self.t += dwell.n_slots
        self._append_log(dwell)

        # `terminated`, not `truncated` (D35): the 30 s horizon is the task
        # definition -- Turing's own `collection_time_s`, and the extent of the
        # world the grid describes -- not an artificial cap on an ongoing task.
        # There is no state beyond slot 600 to bootstrap a value from.
        terminated = self.t >= N_SLOTS
        return self._observation(), reward, terminated, False, self._info(dwell, newly)

    @property
    def first_intercept(self) -> dict[int, int]:
        """`first_e` per emitter -- the censored-intercept-time numerator (D28)."""
        return {e: t["first"] for e, t in self.tracks.items()}

    def emitter_table(self) -> list[dict]:
        """Artefact 2 of `EVALUATION.md` §8: one row per detectable emitter.

        The shape of a real ESM intercept log, and -- with the per-slot episode
        log -- everything §4 needs. `E` is the detectable set (D27), so an emitter
        that was never findable is absent rather than scored as a miss.
        """
        rows = []
        for emitter, (on_e, off_e) in sorted(self.detectable.items()):
            track = self.tracks.get(emitter)
            contribution = self.grid.contributions[emitter]
            rows.append({
                "emitter": emitter,
                "uid": contribution.uid,
                "on_slot": on_e,
                "off_slot": off_e,
                "first_intercept_slot": track["first"] if track else None,
                "last_intercept_slot": track["last"] if track else None,
                "intercept_count": track["count"] if track else 0,
                "bands_seen_in": sorted(track["bands"]) if track else [],
            })
        return rows

    # ----------------------------------------------------------- observation --

    def _observation(self) -> np.ndarray:
        """The 109-vector (D34), built from the agent's own scan history alone.

        Each of the three per-band quantities maps onto one of the PS's own
        figures of merit, which is why these three:

          hit rate       what a scheduler needs to estimate detection probability
          visit density  intercept rate -- how its airtime is being spent
          staleness      what makes this restless rather than a plain bandit: a
                         band ignored for 10 s may have become busy without
                         telling you

        Nothing truth-side appears here. `Z`, per-emitter levels and `first_e` are
        all available to the *reward* (D29) and all absent from this vector.
        """
        elapsed = max(self.t, 1)

        looked = self._slots_looked > 0
        hit_rate = np.zeros(N_BANDS, dtype=np.float64)
        np.divide(self._hits, self._slots_looked, out=hit_rate, where=looked)

        # Sums to 1 across bands once anything has been looked at: it is the
        # fraction of spent airtime, and airtime is the only currency (D31).
        visit_density = self._slots_looked / elapsed

        # Never-visited reads 1.0, maximally stale -- an unexplored band should
        # look at least as attractive as one last seen at t=0.
        staleness = np.where(
            self._last_slot >= 0, (self.t - self._last_slot) / N_SLOTS, 1.0
        )

        return np.concatenate([
            hit_rate, visit_density, staleness, [self.t / N_SLOTS]
        ]).astype(np.float32)

    # ------------------------------------------------------------------ info --

    def _info(self, dwell: DwellResult | None = None, newly: set[int] | None = None) -> dict:
        """Truth-side quantities, for the evaluator only. Never fed to the agent.

        Everything `EVALUATION.md` §4 needs is computable from this plus the
        episode log: `first_intercept` and `detectable` give censored intercept
        time and coverage, `pulses_intercepted` over `total_pulses` gives
        interception ratio.
        """
        info = {
            "slot": self.t,
            "time_s": self.t * SLOT_S,
            "n_detectable": len(self.detectable),
            "n_intercepted": len(self.tracks),
            "pulses_intercepted": self.pulses_intercepted,
            "total_pulses": self.grid.total_pulses,
            "total_reward": self.total_reward,
        }
        if dwell is not None:
            info.update({
                "band": dwell.band,
                "dwell_slots": dwell.n_slots,
                "Y": dwell.Y.tolist(),
                "Z": dwell.Z.tolist(),
                "pulses": dwell.pulses,
                "newly_intercepted": sorted(newly or ()),
            })
        if self.t >= N_SLOTS:
            info["first_intercept"] = self.first_intercept
            info["detectable"] = dict(self.detectable)
            info["emitter_table"] = self.emitter_table()
        return info

    def _append_log(self, dwell: DwellResult) -> None:
        """One row **per slot**, as `EVALUATION.md` §8 specifies.

        Per slot rather than per dwell so that `metrics.py` never has to unpick a
        two-slot dwell, and so the log lines up cell-for-cell with the waterfall
        render. The env accumulates rows and writes nothing; serialising them is
        `metrics.py`'s job.
        """
        for i in range(dwell.n_slots):
            slot = dwell.slot0 + i
            self.log.append({
                "slot": slot,
                "time_s": slot * SLOT_S,
                "band": dwell.band,
                "dwell_slots": dwell.n_slots,
                "Y": bool(dwell.Y[i]),
                "Z": bool(dwell.Z[i]),
                "pulses": int(dwell.C[i]),
                "level_dbm": float(dwell.level_dbm[i]),
                "measured_dbm": float(dwell.measured_dbm[i]),
            })

    # --------------------------------------------------------------- summary --

    def episode_metrics(self) -> dict:
        """The scheduler-level table for this episode (`EVALUATION.md` §4).

        Interception ratio, censored mean intercept time and coverage are printed
        together, never one alone -- a single scalar hides the entire problem, and
        both traps in §4 were measured on real data. Censoring is the part that is
        easy to get wrong: an emitter never found counts at the **full episode
        length**, not dropped from the average, because averaging over only the
        emitters you did find rewards not looking.

        The full comparison across schedulers, with distributions and repeated
        seeds, belongs to `metrics.py`. This is one episode's row of it.
        """
        n_e = len(self.detectable)
        delays = []
        for e, (on_e, _) in self.detectable.items():
            track = self.tracks.get(e)
            delays.append((N_SLOTS if track is None else track["first"]) - on_e)

        return {
            "scenario": self.scenario.name,
            "reward": self.reward_name,
            "interception_ratio": (
                self.pulses_intercepted / self.grid.total_pulses
                if self.grid.total_pulses else float("nan")
            ),
            "censored_mean_intercept_time_s": (
                float(np.mean(delays)) * SLOT_S if delays else float("nan")
            ),
            "emitter_coverage": len(self.tracks) / n_e if n_e else float("nan"),
            "avg_intercept_rate_per_s": len(self.tracks) / EPISODE_S,
            "total_reward": self.total_reward,
            "n_detectable": n_e,
            "n_intercepted": len(self.tracks),
            # Steps, not slots: an episode is always 600 slots but 300-600
            # decisions, depending how many wide bands were chosen (D31).
            "n_steps": self.n_steps,
        }
