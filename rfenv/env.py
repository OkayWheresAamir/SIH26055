"""L3 — the agent interface.

A standard Gymnasium environment wrapping L0–L2. The scheduler's whole job, and
its whole world:

    action       choose one of 36 bands
    observation  what its own scan history has taught it -- nothing else
    reward       one of REWARDS' registered candidates, all of which may read truth
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
    NOISE_FLOOR_DBM,
    NOISE_SIGMA_DB,
    SLOT_S,
)
from rfenv.receiver import DwellResult, Receiver
from rfenv.scenario import EmitterPool, Scenario
from rfenv.truth import TruthGrid

# Clamp range for normalising measured_dbm into the observation's [0,1] box --
# mirrors render._common's LEVEL_VMIN_DB/LEVEL_VMAX_DB (the waterfall's own
# color scale), duplicated as literals rather than imported so env.py stays
# free of matplotlib (render/__init__.py calls matplotlib.use("Agg") on import,
# and every non-render part of the environment must work without it).
_DBM_CLAMP_MIN = NOISE_FLOOR_DBM   # -120.0
_DBM_CLAMP_MAX = -20.0

# --------------------------------------------------------------------------- #
# Reward candidates (D7, D29)
# --------------------------------------------------------------------------- #
#
# Exactly three, and the set stays at three (D29): every candidate scored on the
# same 47 scenarios is another draw, and best-of-many is partly selection noise.
# A camping-penalised fourth candidate was registered and retired within this
# same working session (see git history / `reward_weighted_camp`, kept below
# as a commented-out draft alongside a further `reward_hybrid` sketch) --
# neither is active; REWARDS holds exactly the original three again.
#
# Candidates 1 and 2 are **per slot**, and a dwell's reward is the sum over its
# slots (D31), so a 100 ms dwell can earn up to +2. That keeps reward per unit
# time equal across wide and narrow bands -- the alternative, +1 per dwell
# regardless of length, would make the seven wide bands strictly dominated, and
# those are measurably the bands holding the densest emitter populations and the
# slowest rotators.
#
# Candidate 3 is the exception: it is flat per dwell by design (a discovery is
# worth the same whether the dwell that found it was 1 or 2 slots), so it does
# not carry D31's per-slot invariant -- see `reward_first_intercept`'s own
# docstring for why.
#
# D28 puts the two headline metrics on opposite sides of the Y/Z line -- censored
# intercept time needs Y = 1, interception ratio does not -- so no single reward
# is aligned with both. That is why there are three rather than one, and why the
# rule for choosing between them is a human decision recorded as open in D29.
# Nothing here ranks them.
#
# Every candidate takes the same seven arguments even though no candidate reads
# all seven:
#
#     self._reward_fn(dwell, newly, camp_slots,
#                     hit_rate, visit_density, staleness, action)
#
# `dwell` and `newly` are truth-side (D29 permits both); `camp_slots` is the
# streak counter `ScanEnv.step` maintains; the three arrays and `action` are
# the same observation components `_observation()` builds, handed over so a
# reward can price exploration against the agent's own view of the world
# rather than against truth.
#
# One call signature, not a special case per candidate, so a new candidate can
# be registered -- or a retired one re-enabled -- without touching the call
# site. Unused parameters are the price of that and are deliberate.


def reward_hit_z(
    dwell: DwellResult,
    newly: set[int],
    camp_slots: int,
    hit_rate_array: np.ndarray,
    visit_density_array: np.ndarray,
    staleness_array: np.ndarray,
    action: int,
) -> float:
    """Candidate 1: +1 per **true** hit, cell-level `Z` (D5). Not the current
    default -- see `DEFAULT_REWARD` -- but still every rung's registered
    baseline reward for training rungs 7/8/9's own base variant.

    Values every occupied cell equally, whatever its level, so it prices
    opportunity rather than detectability. Predicted to favour interception ratio,
    which is itself the threshold-free opportunity metric.
    """
    return float(dwell.Z.sum())


def reward_hit_y(
    dwell: DwellResult,
    newly: set[int],
    camp_slots: int,
    hit_rate_array: np.ndarray,
    visit_density_array: np.ndarray,
    staleness_array: np.ndarray,
    action: int,
) -> float:
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


def reward_balance(
    dwell: DwellResult,
    newly: set[int],
    camp_slots: int,
    hit_rate_array: np.ndarray,
    visit_density_array: np.ndarray,
    staleness_array: np.ndarray,
    action: int,
) -> float:
    """Candidate 3: exploitation, exploration and an explicit camping cost, priced
    against the agent's own observation rather than against truth.

    Four terms, and the last three are the point. Candidates 1 and 2 pay only
    for what a dwell found, which makes camping the densest wide band a local
    optimum -- measured, every RecurrentPPO policy trained to date collapses
    onto exactly that, tuning one band for all 600 slots (coverage 0.25 against
    round-robin's 1.00). So:

      `+0.5 * hit_rate[action]`                     exploit what has paid out
      `+(1.5 - visit_density[action]) * staleness`  favour bands that are both
                                                    under-visited and overdue
      `+0.5 * Z.sum()`                              real occupancy, as candidate 1
      `-0.5 * camp_slots`                           a cost that grows with the
                                                    length of the current streak

    The camping term is charged per slot of the *current streak*, not per dwell,
    so it compounds the longer a policy stays put -- which is what makes leaving
    cheaper than staying once a streak is long, rather than merely less good.

    Reads `hit_rate`/`visit_density`/`staleness` at `action`, all three of which
    are observation components, so nothing here is information the policy could
    not have acted on itself.
    """
    reward = 0.0

    # Exploitation
    reward += 0.5 * hit_rate_array[action]

    # Exploration / airtime balance
    reward += (
        1.0
        * (1.5 - visit_density_array[action])
        * staleness_array[action]
    )

    # Actual useful occupancy
    reward += 0.5 * float(dwell.Z.sum())

    # Increasing cost for consecutive camping
    reward -= 1.0 * camp_slots

    return float(reward)
    

# def reward_first_intercept_test(dwell: DwellResult,
#     camp_slots: int,
#     hit_rate: float,
#     visit_density: float,
#     staleness: float) -> float:
#     """Candidate 3: a flat +3 per dwell that finds at least one emitter for the
#     **first time in a given band** (D28's own-first-intercept rule, extended
#     per-band rather than per-episode -- see `ScanEnv.step`'s `newly`), whatever
#     the count of new emitters that dwell credits. A dwell that finds nothing new
#     still earns +1 if it lands on an occupied cell at all (`Z`, even a cell
#     belonging to an already-found emitter), minus a flat 0.5 -- so an empty look
#     nets -0.5, a look that re-touches known occupancy nets +0.5, and a look that
#     surfaces a new (emitter, band) pair nets +3 outright.

#     Deliberately flat rather than `+1 * len(newly)`: this candidate prices *that*
#     a dwell discovered something at all, not how many, and does not scale with
#     dwell width the way candidates 1/2 and D31 do -- a discovery is worth the
#     same whether it lands on a 1- or 2-slot band.

#     Reads truth, which is allowed (D29) and necessary: novelty is not something
#     binary hit/miss can perceive. Whether a policy can *learn* to act on a signal
#     it cannot observe is the open question D30 raises; it does not block training.
#     """
#     reward = 0.0
#     if len(newly) == 0:
#         if float(dwell.Y.sum()) > 0:
#             reward += 0.5*(dwell.Y.sum())
#             if camp_slots > 1:
#                 reward -= 0.5*(camp_slots-1)
#         else:
#             reward -= 0.5
#             if camp_slots > 1:
#                 reward -= 0.5*(camp_slots-1)
#     else:
#         reward += 3.0*len(newly)
#     return reward



REWARDS = {
    "hit_z": reward_hit_z,
    "hit_y": reward_hit_y,
    "reward_balance": reward_balance,
}

DEFAULT_REWARD = "reward_balance"  # the one used to train the registered checkpoints, and the one rung 9's base variant is scored on


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

    # "human" is deliberately not offered: this project's rendering stack forces
    # the Agg backend (render.py) so tests and CI never try to open a window.
    # Use "rgb_array" and assemble frames yourself, or see
    # `render.compare_animation` for a ready-made multi-scheduler version.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        scenario: Scenario | None = None,
        pool: EmitterPool | None = None,
        reward: str = DEFAULT_REWARD,
        gamma_dbm: float = GAMMA_DBM,
        sigma_db: float = NOISE_SIGMA_DB,
        render_mode: str | None = None,
    ):
        super().__init__()
        if scenario is None and pool is None:
            raise ValueError("ScanEnv needs a `scenario` to replay or a `pool` to sample from")
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {sorted(REWARDS)}, got {reward!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}"
            )
        self.SWEEP_SLOTS = 43
        self._scenario = scenario          # fixed world to replay every reset() (validation gates)
        self._pool = pool                  # draw a fresh scenario each reset() instead (RL training, D25/D32)
        self.reward_name = reward          # the REWARDS key, kept around for episode_metrics()/logging
        self._reward_fn = REWARDS[reward]  # resolved once here, not looked up by string every step()
        self.gamma = float(gamma_dbm)      # receiver's detection threshold, dBm (frozen, D25)
        self.sigma = float(sigma_db)       # receiver's noise stdev, dB (frozen, D25)
        self.render_mode = render_mode     # None (default, render() is a no-op) or "rgb_array"

        self.action_space = spaces.Discrete(N_BANDS)   # one of the 36 bands, chosen every step()
        # 36 x 4 + 3 = 147 (D34). Every component is natively a fraction (the
        # current-band block is a one-hot, still 0/1-valued; measured_dbm is
        # clamped and linearly rescaled), so the box is the unit interval and
        # no scaling layer is needed anywhere.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_BANDS * 4 + 3,), dtype=np.float32
        )

        # Placeholders only -- the real values are per-episode state, built
        # fresh by reset() (never carried from one episode into the next, D20).
        self.grid: TruthGrid | None = None       # this episode's L1 truth grid
        self.receiver: Receiver | None = None    # this episode's L2 detector
        self.log: list[dict] = []                # per-slot rows so far, see _append_log

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
        # -------------------------------------------------- observation-facing --
        # Every array below is per-band (length N_BANDS); _observation() reads
        # them straight, nothing more is tracked than what it needs.
        self._current_band = 0        # band the most recent dwell was on -> current_band's one-hot
        self.t = 0                    # slots elapsed this episode, 0..N_SLOTS -> the clock scalar
        self._slots_looked = np.zeros(N_BANDS, dtype=np.int64)   # airtime spent per band, in slots -> visit_density
        self._hits = np.zeros(N_BANDS, dtype=np.int64)           # declared hits (Y) accumulated per band -> hit_rate
        self._last_slot = np.full(N_BANDS, -1, dtype=np.int64)     # slot of the last dwell per band, -1 = never -> staleness
        self._last_hit_slot = np.full(N_BANDS, -1, dtype=np.int64)  # slot of the last declared hit per band, -1 = never (episode log only, not in the observation)
        self._prev_action = -1        # last action taken; -1 sentinel so step() 1 always starts a fresh streak
        self._camp_slots = 0          # length of the current same-band streak, in slots -> camp_time
        # Never-measured reads as the noise floor -- quietest possible, not the
        # loudest -- consistent with every episode starting cold (D20): no
        # measurement carries over from a previous episode.
        self._last_measured_dbm = NOISE_FLOOR_DBM   # raw dBm, un-normalised -> measured_dbm (clamped+scaled in _observation)
        self._hit_rate_array = np.zeros(N_BANDS, dtype=np.float32)  # hit_rate per band, 0..1 -> hit_rate_array 
        self._visit_density_array = np.zeros(N_BANDS, dtype=np.float32)  # visit_density per band, 0..1 -> visit_density_array
        self._staleness_array = np.ones(N_BANDS, dtype=np.float32)  # staleness per band, 0..1 -> staleness_array
        # -------------------------------------------------------- evaluator-side --
        # Everything below feeds `info`/episode_metrics()/emitter_table -- never
        # the observation. E is the coverage denominator: emitters with a
        # non-empty detectable interval, not every transmitter in the metadata
        # (D27) -- 19% of train transmitters are never detectable at all and
        # scoring a scheduler for missing those would be meaningless.
        self.detectable = {
            int(i): self.grid.detectable_interval(int(i), self.gamma)
            for i in self.grid.detectable_emitters(self.gamma)
        }
        self.tracks: dict[int, dict] = {}   # per-emitter: first/last intercept slot, count, bands seen (D28)
        self.pulses_intercepted = 0         # running total of dwell.pulses -- interception ratio's numerator
        self.total_reward = 0.0             # running sum of this episode's per-step rewards
        self.n_steps = 0                    # step() calls so far: 300-600 over the episode, never exactly 600 (D31)
        self.log = []                       # per-slot rows, reset fresh each episode; see _append_log
        
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
        # Staleness of the band BEFORE this dwell begins.
        if self._last_slot[action] >= 0:
            staleness_before = (
                self.t - self._last_slot[action]
            ) / N_SLOTS
        else:
            # Never visited = maximally stale
            staleness_before = 1.0
        
        
        self._current_band = action
        elapsed = max(self.t, 1)

        # Hit rate for selected band
        self._hit_rate_array[action] = (
            self._hits[action] / self._slots_looked[action]
            if self._slots_looked[action] > 0
            else 0.0
        )
        if self._slots_looked[action] > 0:
            hit_rate = (
                self._hits[action]
                / self._slots_looked[action]
            )
        else:
            hit_rate = 0.0
        self._visit_density_array[action] = (
            self._slots_looked[action]
            / elapsed
        )

        # Fraction of episode airtime already spent on this band
        visit_density = (
            self._slots_looked[action]
            / elapsed
        )

        self._staleness_array[action] = (
            self._last_slot[action]
            / N_SLOTS
        )

        # How long this band has been neglected
        if self._last_slot[action] >= 0:
            staleness = (
                self.t - self._last_slot[action]
            ) / N_SLOTS
        else:
            staleness = 1.0
        dwell = self.receiver.dwell(self.grid, action, self.t)

        # This dwell's mean measured level -- S + noise, what the receiver's
        # detector actually read, not the truth-side S alone (D19: observable).
        self._last_measured_dbm = float(dwell.measured_dbm.mean())

        # Camping streak: consecutive slots spent on this same band, reset the
        # moment the action changes. Slots, not dwells, so a run of narrow
        # (1-slot) dwells on one band doesn't read as "less camped" than the
        # same airtime spent on a wide (2-slot) one -- consistent with airtime
        # being the only currency (D31).
        if action == self._prev_action:
            self._camp_slots += dwell.n_slots
        else:
            self._camp_slots = dwell.n_slots
        self._prev_action = action

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
                # `newly` fires again on a band this emitter hasn't been
                # credited in before -- not just its true first-ever intercept.
                # Deliberate (reward_first_intercept): an emitter generically
                # spans 2 overlapping bands (D3), so this rewards sweeping an
                # emitter's full footprint, not only its first sighting.
                # `first`/`first_e` (censored intercept time) is untouched --
                # only this reward-facing set is widened.
                if dwell.band not in track["bands"]:
                    newly.add(emitter)
                track["bands"].add(dwell.band)


        

        self._slots_looked[action] += dwell.n_slots
        self._hits[action] += int(dwell.Y.sum())
        
        self._last_slot[action] = dwell.slot0 + dwell.n_slots - 1
        hit_slots = np.flatnonzero(dwell.Y)
        if hit_slots.size:
            self._last_hit_slot[action] = dwell.slot0 + int(hit_slots[-1])
        self.pulses_intercepted += dwell.pulses
        self.n_steps += 1
        self.t += dwell.n_slots
        self._append_log(dwell)

        reward = float(
                        self._reward_fn(
                            dwell,
                            newly,
                            self._camp_slots,
                            self._hit_rate_array,
                            self._visit_density_array,
                            self._staleness_array,
                            self._current_band
                        )
                    )

        self.total_reward += reward
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
        """The 147-vector (D34), built from the agent's own scan history alone.

        Each of the three per-band quantities maps onto one of the PS's own
        figures of merit, which is why these three:

          hit rate       what a scheduler needs to estimate detection probability
          visit density  intercept rate -- how its airtime is being spent
          staleness      what makes this restless rather than a plain bandit: a
                         band ignored for 10 s may have become busy without
                         telling you

        A fourth, `current_band`, is a one-hot of the band this observation is
        being computed after -- so the agent can tell "I am here now" apart from
        "I last looked here a while ago" (`staleness` alone conflates the two,
        since the currently-tuned band always reads staleness 0 too).

        A scalar, `camp_time`, is how long the *current streak* on this band has
        run: consecutive slots spent on whatever band `current_band` points to,
        reset the instant the action changes (see `ScanEnv.step`), normalised by
        `N_SLOTS`. `visit_density` is the wrong signal for this -- it is airtime
        share over the *whole* episode, so a band camped early and abandoned
        still reads high density long after the agent moved on; `camp_time`
        collapses back to 0 the moment it leaves.

        A final scalar, `measured_dbm`, is the mean of `S + noise` over the most
        recent dwell's slots -- what the receiver's detector actually read
        (D19: observable, unlike the truth-side `S` alone), clamped to
        `[_DBM_CLAMP_MIN, _DBM_CLAMP_MAX]` and linearly rescaled to `[0, 1]` to
        fit the box. Global, not per-band: it reports only the band just left,
        the same scope `camp_time` and `current_band` already have. Before any
        dwell it reads as the clamp floor (quietest possible) -- see `reset()`.

        Nothing truth-side appears here. `Z`, per-emitter levels and `first_e` are
        all available to the *reward* (D29) and all absent from this vector.
        """
        elapsed = max(self.t, 1)   # slots so far, floored at 1 so the divisions below never hit 0/0

        # hit_rate[b] = declared hits on b / slots looked at b, only where b has
        # been looked at at least once (elsewhere left at 0 by np.zeros below --
        # "never looked" and "looked, zero hits" would otherwise be indistinguishable
        # from a bare division, but np.divide's `where` skips the untouched bands
        # rather than raising or writing 0/0 = nan into them).
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

        # One-hot: 1.0 at whichever band the most recent dwell was on, 0.0
        # everywhere else -- "where am I right now", distinct from staleness
        # (which the current band also reads as 0, but so would a band left
        # one slot ago).
        current_band = np.zeros(N_BANDS, dtype=np.float64)
        current_band[self._current_band] = 1.0

        # Already a fraction of N_SLOTS by construction (_camp_slots counts up
        # to at most N_SLOTS, an episode camped start-to-finish), so no clamp
        # needed here the way measured_dbm below needs one.
        camp_time = self._camp_slots / N_SLOTS

        # measured_dbm needs a manual normalisation step, unlike every other
        # component above: it is a real dBm value with no natural [0,1] range
        # (S + noise can fall anywhere), whereas hit_rate/visit_density/
        # staleness/camp_time/current_band/the clock are all ratios or one-hot
        # flags that land in [0,1] on their own.
        #   1. Clamp the raw reading into the fixed window
        #      [_DBM_CLAMP_MIN, _DBM_CLAMP_MAX] = [-120, -20] dBm, so one
        #      unusually loud or quiet dwell can't blow the box's declared
        #      bounds (Box(low=0, high=1) would otherwise be violated, and
        #      gymnasium's check_env / SB3 both assume the space is honest).
        clamped = min(max(self._last_measured_dbm, _DBM_CLAMP_MIN), _DBM_CLAMP_MAX)
        #   2. Min-max rescale that clamped value onto [0, 1]: the clamp floor
        #      maps to exactly 0.0, the clamp ceiling to exactly 1.0, and
        #      everything else linearly in between.
        measured_dbm = (clamped - _DBM_CLAMP_MIN) / (_DBM_CLAMP_MAX - _DBM_CLAMP_MIN)

        return np.concatenate([
            hit_rate, visit_density, staleness, current_band,
            [self.t / N_SLOTS], [camp_time], [measured_dbm]]).astype(np.float32)

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
            # A vectorised env (SB3's DummyVecEnv, used by every rfenv.rl
            # trainer) auto-resets a sub-env the instant its episode ends,
            # *before* handing control back to a training callback -- so by
            # the time anything outside step()/reset() can react to
            # terminated=True, self.episode_metrics() would already describe
            # the fresh episode that followed, not the one that just finished
            # (confirmed: n_steps reads back as 0). Snapshotting it into the
            # terminal info dict, here, is what survives that reset -- a
            # callback reads info["episode_metrics"], never calls the method.
            info["episode_metrics"] = self.episode_metrics()
        return info

    # --------------------------------------------------------------- render --

    def render(self):
        """Optional visualisation -- off unless `render_mode` was set at construction.

        Returns an (H, W, 3) uint8 array for `render_mode="rgb_array"`: the
        truth grid as background, with the tuning path and declared hits
        accumulated in `self.log` so far. `None` if `render_mode` is `None`
        (the default), which is the whole "optional" part -- nothing about
        `reset()`/`step()` changes either way.

        Lazy import: `render.py` is the only module that imports matplotlib,
        and every other part of the environment must keep working without a
        plotting stack installed (`rfenv/render.py`'s own module docstring).
        """
        if self.render_mode is None:
            return None
        if self.grid is None:
            raise RuntimeError("call reset() before render()")
        from rfenv import render as R
        return R.env_frame(self)

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
                "last_hit_slot": int(self._last_hit_slot[dwell.band]),
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
