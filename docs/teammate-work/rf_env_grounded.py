"""
RF Environment for Smart Scan Scheduling — RL Representation
Grounded strictly in the Problem Statement text.

PS references are quoted inline to justify every design choice.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ─────────────────────────────────────────────────────────────────────────────
# 1. THE CORE DATA STRUCTURE — what the PS calls the "environment"
# ─────────────────────────────────────────────────────────────────────────────
#
# PS: "The status of environment for each frequency band at each time step
#      can be recorded as a transmission or a non-transmission."
#
# This sentence alone defines the ground truth:
#
#   E[n, t]  ∈  {0, 1}
#   └── n: band index   (0 … N-1)   ← "many bands"
#   └── t: time step index (0 … T-1) ← "each time slot"
#   └── 0 = no transmission, 1 = transmission
#
# This is a binary matrix of shape (N_bands × T_steps).
# It is the HIDDEN state — the environment holds it, the agent cannot see
# all of it at once.
#
# PS: "Sensors with typically high sensitivity but with at least an order
#      lower instantaneous bandwidth compared to overall bandwidth"
#
# → The receiver can observe exactly ONE band per time step.
#   It cannot see the full E[:, t] vector — only E[action, t].
#   This makes the problem a Partially Observable MDP (POMDP).
# ─────────────────────────────────────────────────────────────────────────────


class RFEnvironment(gym.Env):
    """
    Simulated RF environment for ES receiver scan scheduling.

    Every design decision below is traced to a specific PS statement.
    No components have been invented beyond what the PS requires.

    Parameters
    ----------
    N_bands : int
        Number of discrete frequency bands the receiver can tune to.
        PS: "The frequency spectrum for own receiver consists of many bands."

    T_steps : int
        Number of discrete time steps in one episode (one scan mission).
        PS: "status of environment for each frequency band at each time step"

    E_truth : np.ndarray of shape (N_bands, T_steps), dtype int8, optional
        Pre-computed binary activity grid loaded from stare data.
        PS: "measurements obtained from a simulated RF environment which has
             truth information on status of emitters in each band and at
             each time slot."
        If None, a placeholder zero-grid is used (emitter simulation
        to be plugged in separately).
    """

    metadata = {"render_modes": []}

    def __init__(self, N_bands: int, T_steps: int, E_truth: np.ndarray = None):
        super().__init__()

        self.N = N_bands
        self.T = T_steps

        # ── Hidden ground truth ────────────────────────────────────────────
        # PS: "truth information on status of emitters in each band and at
        #      each time slot"
        # Shape: (N_bands, T_steps), values in {0, 1}
        # This is INTERNAL to the environment — never passed to the agent.
        if E_truth is not None:
            assert E_truth.shape == (N_bands, T_steps)
            self._E_truth = E_truth.astype(np.int8)
        else:
            self._E_truth = np.zeros((N_bands, T_steps), dtype=np.int8)

        # ── Episode state ──────────────────────────────────────────────────
        self._t          = 0          # current discrete time step
        self._hit_counts  = None      # accumulated hits per band
        self._obs_counts  = None      # how many times each band was observed
        self._last_obs_t  = None      # last time step each band was observed

        # ── Action Space ───────────────────────────────────────────────────
        # PS: "Interception of signals is a two dimensional search problem
        #      since it involves adjusting receiver's frequency at correct time."
        #
        # The agent controls the FREQUENCY dimension: which band to tune to.
        # The TIME dimension advances automatically each step.
        # Therefore: action ∈ {0, 1, ..., N_bands - 1}   (discrete integer)
        #
        # The agent picks ONE band per time step.
        # PS: receiver has instantaneous BW << total BW → one band at a time.
        self.action_space = spaces.Discrete(N_bands)

        # ── Observation Space ──────────────────────────────────────────────
        # What can the agent actually know at each step?
        #
        # PS: "absence of prior reliable intelligence of emitters"
        # → The agent has no pre-loaded map of who is where.
        #
        # PS: "The model should then be trained based on hits and misses."
        # → The only signal the agent receives is: did it detect something?
        #   From this, it must accumulate its own belief about each band.
        #
        # The observation vector the agent receives is built from its own
        # accumulated history. Three quantities per band are PS-justified:
        #
        #   (a) empirical_hit_rate[n]   — fraction of times band n had a
        #       transmission when observed.
        #       PS: "probability of detection" (figure of merit) — the agent
        #       needs a running estimate of per-band activity to inform
        #       scheduling decisions.
        #
        #   (b) obs_density[n]   — how frequently band n has been visited
        #       (observations / elapsed steps).
        #       PS: "Avg intercept rate" — needed to balance coverage.
        #
        #   (c) staleness[n]   — normalised time since band n was last checked.
        #       PS: "average intercept time error" — the agent needs to track
        #       how stale its knowledge of each band is, so it can decide
        #       whether revisiting is worth the dwell cost.
        #
        # + one scalar: normalised current time step (gives the agent temporal
        #   context; the mission has a finite horizon).
        #
        # Total observation dimension: N_bands * 3 + 1
        obs_dim = N_bands * 3 + 1
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # ── Reset ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t          = 0
        self._hit_counts  = np.zeros(self.N, dtype=np.float32)
        self._obs_counts  = np.zeros(self.N, dtype=np.float32)
        self._last_obs_t  = np.full(self.N, -1, dtype=np.float32)  # -1 = never seen
        return self._build_observation(), {}

    # ── Step ───────────────────────────────────────────────────────────────
    def step(self, action: int):
        """
        Execute one scan decision.

        Parameters
        ----------
        action : int
            Which band to observe at the current time step.
            PS: "adjusting receiver's frequency at correct time."

        Returns
        -------
        observation : np.ndarray
            Agent's accumulated belief state (see _build_observation).
        reward : float
            Scalar feedback signal.
            PS: "Avg Reward/cost function" as figure of merit;
                "trained based on hits and misses."
        terminated : bool
            True when T_steps elapsed (mission window closed).
        truncated : bool
            Always False (no truncation logic needed here).
        info : dict
            Raw quantities for logging PS figures of merit.
        """
        assert 0 <= action < self.N, f"Invalid band {action}"
        assert self._t < self.T, "Episode already terminated."

        # ── Ground truth lookup ────────────────────────────────────────────
        # PS: "transmission or a non-transmission" at this band, this step.
        # This is the ONLY moment the hidden truth is revealed to the agent
        # (indirectly, through the hit/miss signal).
        hit = int(self._E_truth[action, self._t])   # 0 or 1

        # ── Update agent's history ─────────────────────────────────────────
        self._obs_counts[action]  += 1
        self._hit_counts[action]  += hit
        self._last_obs_t[action]  = float(self._t)

        # ── Compute reward ─────────────────────────────────────────────────
        reward = self._compute_reward(hit, action)

        # ── Advance time ───────────────────────────────────────────────────
        self._t += 1
        terminated = (self._t >= self.T)

        # ── Info dict: PS figures of merit (for logging, not for agent) ───
        info = self._compute_figures_of_merit(hit, action)

        return self._build_observation(), reward, terminated, False, info

    # ── Observation builder ────────────────────────────────────────────────
    def _build_observation(self) -> np.ndarray:
        """
        Construct the agent's observation vector from its own accumulated
        history. This is what the RL policy receives as input.

        All three components are justified by PS figures of merit.
        """
        t_elapsed = max(self._t, 1)

        # (a) Empirical hit rate per band — running P(transmission | band n)
        #     Estimated purely from the agent's own observations.
        #     Unobserved bands → 0.0 (no information yet).
        hit_rate = np.where(
            self._obs_counts > 0,
            self._hit_counts / self._obs_counts,
            0.0,
        ).astype(np.float32)

        # (b) Observation density per band — how often has each band been visited?
        #     Normalised by elapsed steps to stay in [0, 1].
        obs_density = (self._obs_counts / t_elapsed).astype(np.float32)

        # (c) Staleness per band — normalised steps since each band was last seen.
        #     Never-observed bands → 1.0 (maximally stale / unexplored).
        #     Recently observed band → value near 0.0.
        staleness = np.where(
            self._last_obs_t >= 0,
            (self._t - self._last_obs_t) / self.T,
            1.0,
        ).astype(np.float32)

        # (d) Normalised current time step
        t_norm = np.array([self._t / self.T], dtype=np.float32)

        return np.concatenate([hit_rate, obs_density, staleness, t_norm])

    # ── Reward function ────────────────────────────────────────────────────
    def _compute_reward(self, hit: int, action: int) -> float:
        """
        Reward grounded in PS objectives — no invented components.

        PS explicitly states two objectives:
          1. "minimize intercept time"
          2. "ensure a high interception rate"

        And one training signal:
          3. "trained based on hits and misses"

        Component A — Hit reward (interception rate objective):
            +1 when a transmission is detected.
            0 when the chosen band has no transmission.
            PS: "trained based on hits and misses."

        Component B — Opportunity cost (intercept time objective):
            Small negative when receiver dwells on a band it recently
            checked and found empty, while other bands remain unexplored.
            This is the direct computational translation of the PS statement:
            "Open loop strategies ... may lose time to nonthreatening
             emitters by not giving time to new or threatening ones."
            The penalty nudges the scheduler away from redundant revisits.

        Note: probability of false alarm (Pfa) is a figure of merit for
        evaluation but the environment here has no false alarm in the
        strict sense (E is binary truth). Pfa is logged in info, not reward.
        """
        # Component A — core hit/miss signal
        reward = float(hit)

        # Component B — opportunity cost for wasteful revisit
        # Only applied when: (i) band was observed recently, and
        #                    (ii) it produced no hit now.
        if self._last_obs_t[action] >= 0 and hit == 0:
            steps_since_last = self._t - self._last_obs_t[action]
            if steps_since_last <= 1:
                # Revisited immediately with no hit → wasted dwell
                reward -= 0.1   # small, not dominating the +1 hit reward

        return reward

    # ── Figures of merit logger ────────────────────────────────────────────
    def _compute_figures_of_merit(self, hit: int, action: int) -> dict:
        """
        Log the PS-specified evaluation metrics at each step.
        These are NOT used as reward inputs — they are evaluation metrics.

        PS figures of merit:
          - probability of detection (Pd)
          - probability of false alarm (Pfa)
          - Avg intercept rate
          - percentage of correct predictions
          - average intercept time error
        """
        total_obs = int(self._obs_counts.sum())
        total_hits = int(self._hit_counts.sum())

        # Pd: fraction of observations that yielded a hit so far
        Pd = total_hits / total_obs if total_obs > 0 else 0.0

        # Pfa: here 0 by construction (E is binary truth — no noise model yet)
        # Will become meaningful once receiver noise / sensitivity is added.
        Pfa = 0.0

        # Avg intercept rate: hits per unit time
        avg_intercept_rate = total_hits / max(self._t, 1)

        # Bands never observed (completely unexplored)
        bands_unexplored = int((self._obs_counts == 0).sum())

        return {
            "hit":                  hit,
            "chosen_band":          action,
            "time_step":            self._t,
            "Pd":                   round(Pd, 4),
            "Pfa":                  Pfa,
            "avg_intercept_rate":   round(avg_intercept_rate, 4),
            "bands_unexplored":     bands_unexplored,
            "total_hits":           total_hits,
            "total_observations":   total_obs,
        }




# ─────────────────────────────────────────────────────────────────────────────
# 2. THE EMITTER MODEL — what drives E[n, t]
# ─────────────────────────────────────────────────────────────────────────────
#
# PS: "The model should enable prediction of intercept time and interception
#      ratio of a scanning receiver against spatially scanning and
#      frequency agile emitters."
#
# The PS names exactly two emitter behaviour types that need to be modelled.
# No others are mentioned. We model only these two.
# ─────────────────────────────────────────────────────────────────────────────


class FrequencyAgileEmitter:
    """
    PS: "frequency agile emitters"

    A frequency-agile emitter hops across bands over time.
    Its contribution to E[n, t] therefore moves between band indices.
    The receiver must predict WHERE the emitter will be at time t,
    not just WHERE it was.

    Parameters
    ----------
    hop_sequence : list[int]
        Sequence of band indices the emitter visits in order.
        The emitter cycles through this sequence.
        PS does not constrain the hop pattern — it can be linear,
        sawtooth, or random. Derived from archive freq_mode attribute.
    dwell_steps : int
        How many time steps the emitter stays on each frequency
        before hopping. Derived from PRI and dwell parameters.
    start_offset : int
        Phase offset into the hop_sequence at t=0.
        Accounts for PS requirement: "absence of prior reliable intelligence."
        The receiver does not know start_offset ahead of time.
    """

    def __init__(self, hop_sequence: list, dwell_steps: int, start_offset: int = 0):
        self.hop_sequence  = hop_sequence
        self.dwell_steps   = dwell_steps
        self.start_offset  = start_offset

    def active_band(self, t: int) -> int:
        """Return which band this emitter occupies at time step t."""
        effective_t = t + self.start_offset
        hop_index   = (effective_t // self.dwell_steps) % len(self.hop_sequence)
        return self.hop_sequence[hop_index]

    def stamp_on_grid(self, E: np.ndarray, T: int):
        """Write this emitter's transmissions into the ground truth grid."""
        for t in range(T):
            band = self.active_band(t)
            if 0 <= band < E.shape[0]:
                E[band, t] = 1


class SpatiallyScanningEmitter:
    """
    PS: "spatially scanning emitters"

    A spatially-scanning emitter has a rotating or sweeping antenna.
    Even if its RF frequency is fixed, it only illuminates the receiver
    during the angular window when its beam points toward the receiver.

    This makes the interception problem TIME-critical on top of
    frequency-critical — the PS's "two dimensional search problem."

    Parameters
    ----------
    fixed_band : int
        The frequency band this emitter occupies (fixed RF).
    illumination_period_steps : int
        How many steps between successive beam sweeps past the receiver.
        Derived from scan_rate_rpm in the dataset metadata.
    illumination_window_steps : int
        How many consecutive steps the beam illuminates the receiver
        during each sweep pass.
        Derived from beam_width_deg / (scan_rate_rpm * 360/60).
    phase_offset : int
        Time step at which the first illumination window begins.
        Unknown to the receiver a priori — the receiver must estimate it.
        PS: "absence of prior reliable intelligence."
    """

    def __init__(
        self,
        fixed_band: int,
        illumination_period_steps: int,
        illumination_window_steps: int,
        phase_offset: int = 0,
    ):
        self.fixed_band                  = fixed_band
        self.illumination_period_steps   = illumination_period_steps
        self.illumination_window_steps   = illumination_window_steps
        self.phase_offset                = phase_offset

    def is_illuminating(self, t: int) -> bool:
        """True if receiver is within the beam illumination window at time t."""
        phase = (t - self.phase_offset) % self.illumination_period_steps
        return phase < self.illumination_window_steps

    def stamp_on_grid(self, E: np.ndarray, T: int):
        """Write this emitter's illumination windows into the ground truth grid."""
        for t in range(T):
            if self.is_illuminating(t):
                if 0 <= self.fixed_band < E.shape[0]:
                    E[self.fixed_band, t] = 1


def build_ground_truth(
    N_bands: int,
    T_steps: int,
    frequency_agile_emitters: list,
    spatially_scanning_emitters: list,
) -> np.ndarray:
    """
    Assemble the binary ground truth grid E[N_bands, T_steps] from
    a list of emitter objects.

    PS: "A system model for the receiver needs to be developed with
         measurements obtained from a simulated RF environment which has
         truth information on status of emitters in each band and at
         each time slot."

    In the real implementation this grid is derived from the TSRD stare
    data directly. Here it can be populated from parameterised emitters
    drawn from the archive metadata.
    """
    E = np.zeros((N_bands, T_steps), dtype=np.int8)
    for emitter in frequency_agile_emitters:
        emitter.stamp_on_grid(E, T_steps)
    for emitter in spatially_scanning_emitters:
        emitter.stamp_on_grid(E, T_steps)
    return E




# ─────────────────────────────────────────────────────────────────────────────
# 3. SMOKE TEST — validates the structure without any RL algorithm
#    Just confirms the environment behaves correctly under a round-robin
#    open-loop sweep (the PS baseline it is designed to beat).
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    N_BANDS = 36   # matches dataset: 36 dwell positions at 500 MHz each
    T_STEPS = 60   # 60 time slots × 500 ms = 30 s mission window

    # --- Build a small synthetic ground truth ---
    # Two frequency-agile emitters and two spatially-scanning emitters,
    # grounded in the PS's explicit emitter taxonomy.

    fa_emitters = [
        FrequencyAgileEmitter(
            hop_sequence=[2, 8, 14, 20, 26],   # hops across 5 bands
            dwell_steps=3,                      # stays 3 steps per band
            start_offset=0,
        ),
        FrequencyAgileEmitter(
            hop_sequence=[5, 10, 17, 22],
            dwell_steps=4,
            start_offset=2,
        ),
    ]

    ss_emitters = [
        SpatiallyScanningEmitter(
            fixed_band=12,
            illumination_period_steps=10,   # beam returns every 10 steps
            illumination_window_steps=2,    # illuminates for 2 steps per sweep
            phase_offset=3,
        ),
        SpatiallyScanningEmitter(
            fixed_band=25,
            illumination_period_steps=15,
            illumination_window_steps=1,    # very narrow beam: 1 step per sweep
            phase_offset=7,
        ),
    ]

    E_truth = build_ground_truth(N_BANDS, T_STEPS, fa_emitters, ss_emitters)

    print(f"Ground truth grid: {E_truth.shape}  "
          f"(active cells: {E_truth.sum()} / {N_BANDS * T_STEPS})")

    # --- Instantiate the environment ---
    env = RFEnvironment(N_bands=N_BANDS, T_steps=T_STEPS, E_truth=E_truth)
    obs, _ = env.reset()

    print(f"\nObservation space: {env.observation_space.shape}  "
          f"(= {N_BANDS}×3 + 1 = {N_BANDS*3+1})")
    print(f"Action space:      Discrete({env.action_space.n})")
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs (first 10 values): {obs[:10]}")

    # --- Run one episode with a round-robin (open-loop) policy ---
    # This is the BASELINE the PS describes as the status quo.
    # An RL agent should significantly outperform this.
    obs, _ = env.reset()
    total_reward   = 0.0
    total_hits     = 0
    total_steps    = 0

    print("\n── Round-robin open-loop sweep (baseline) ──")
    print(f"{'Step':>4}  {'Band':>4}  {'Hit':>3}  {'Pd':>6}  {'Reward':>7}")
    print("─" * 35)

    for t in range(T_STEPS):
        action = t % N_BANDS          # simple round-robin: cycle through bands
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_steps  += 1
        if info["hit"]:
            total_hits += 1

        if t < 12 or t >= T_STEPS - 3:
            print(f"{t:>4}  {action:>4}  {info['hit']:>3}  "
                  f"{info['Pd']:>6.3f}  {reward:>7.2f}")
        elif t == 12:
            print("  ... (intermediate steps omitted) ...")

        if terminated:
            break

    # --- Final PS figures of merit ---
    total_active = int(E_truth.sum())
    print("\n── Episode summary ──")
    print(f"  Total steps         : {total_steps}")
    print(f"  Active grid cells   : {total_active}  (transmissions that existed)")
    print(f"  Hits                : {total_hits}")
    print(f"  Misses              : {total_active - total_hits}")
    print(f"  Interception rate   : {total_hits / total_active * 100:.1f}%"
          f"   ← PS figure of merit")
    print(f"  Cumulative reward   : {total_reward:.1f}"
          f"   ← PS 'Avg Reward/cost function'")
    print(f"  Bands never visited : {int((env._obs_counts == 0).sum())}"
          f"   ← directly quantifies open-loop's coverage gap")
    print()
    print("An RL scheduler trained on this environment should achieve")
    print("a higher interception rate than the round-robin baseline above.")

