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

import dataclasses

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rfenv.constants import (
    DWELL_SLOTS,
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

# One pass over all 36 bands, in slots (43). Derived from the frozen dwell
# schedule (D3/D42), not chosen -- `baselines.simple.SWEEP_SLOTS` computes the
# identical quantity for the heuristic rungs. It is the natural unit for "how
# overdue is this band": staleness is reported in these, not in episodes (D55).
SWEEP_SLOTS = int(DWELL_SLOTS.sum())
_SWEEPS_PER_EPISODE = N_SLOTS / SWEEP_SLOTS   # 600 / 43 = 13.95, staleness's ceiling

# `hit_streak`'s ceiling (D67): consecutive declared hits on one band, across
# separate visits, capped and divided down to [0, 1]. Not fit to a measurement
# -- chosen structurally, the same way `PROBE_SWEEPS`/`ENTER_HITS` were: 5 is
# comfortably above the 2-hit threshold the (now-removed) hard gate used to
# decide a band was worth committing to, D66, so the feature can still
# distinguish "just crossed that bar" from "been hot for a while", and low
# enough that it saturates within a number of revisits a 30 s episode can
# actually afford (a narrow band's own equal-airtime cadence -- rung 2's own
# reference -- revisits it roughly every 72 slots, so 5 consecutive hits
# already implies real, sustained activity, not noise).
_STREAK_CAP = 5.0

# PulseWidth clamp range for the observation, microseconds (D30). Measured
# 2026-09-13 over 6 real train recordings (`data/turing/stare/train_stare`):
# min 0.007, max 220.0, p99 102.3. Clamped at a round 200 -- comfortably above
# p99, bounding the rare outlier without compressing the bulk of the range.
_PW_CLAMP_MIN = 0.0
_PW_CLAMP_MAX = 200.0

# Band-priority reward (D70, this task, no library). Two-level, not continuous --
# a band is either "tasked" or ordinary; anything more graded needs a source for
# the grading, which is exactly what D70 measured has none (best AUC 0.514,
# a coin flip, across three independent attempts to derive threat from the
# receiver's own signals). 1.0 means ordinary so an episode with the block
# enabled but nothing tasked reads identically to one without it, the same
# convention visit_density/staleness already use (D55).
_PRIORITY_HIGH = 3.0

# `pulse_count`'s reference for the log1p normalisation below (D72): reuses
# `_DENSITY_REF_PULSES` (defined further down, read here only at call time),
# the same constant `reward_balance_improved` already prices `C` against, so
# the observation and the reward that reads the same underlying quantity
# agree on what "a lot of illuminations in one cell" means.

# --------------------------------------------------------------------------- #
# The modular observation layout (D30)
# --------------------------------------------------------------------------- #
#
# Every block `_observation_blocks()` can build, named, with the width and the
# [low, high] bound each one is declared to stay inside -- the single source
# `__init__` reads to build `observation_space` and `_observation()` reads to
# concatenate the flat vector SB3 actually sees. Two ways in, one way out: a
# block exists here once, `OBS_LAYOUTS` only ever selects and orders it.
#
# Two blocks carry a bound past 1.0 on purpose (D55) -- see their own
# docstrings in `_observation_blocks()`. Every block D30 adds is normalised
# into [0, 1] like the rest of D34/D49/D67's, including `aoa_sin`/`aoa_cos`,
# whose natural range is [-1, 1]: stored here as `(x + 1) / 2` so the box stays
# one convention throughout rather than gaining a second bound style for two
# fields out of twelve.
_BLOCK_SPECS: dict[str, tuple[int, float, float]] = {
    "hit_rate":           (N_BANDS, 0.0, 1.0),
    "visit_density":      (N_BANDS, 0.0, float(N_BANDS)),
    "staleness":          (N_BANDS, 0.0, _SWEEPS_PER_EPISODE),
    "current_band":       (N_BANDS, 0.0, 1.0),
    "clock":              (1, 0.0, 1.0),
    "measured_dbm":       (1, 0.0, 1.0),          # D34/D49: global, last dwell only
    "hit_streak":         (N_BANDS, 0.0, 1.0),    # D67
    "current_hit_streak": (1, 0.0, 1.0),          # D67
    "measured_dbm_band":  (N_BANDS, 0.0, 1.0),    # D30: per-band replacement for measured_dbm
    "pulse_width":        (N_BANDS, 0.0, 1.0),    # D30
    "aoa_sin":            (N_BANDS, 0.0, 1.0),    # D30, (sin+1)/2
    "aoa_cos":            (N_BANDS, 0.0, 1.0),    # D30, (cos+1)/2
    "pulse_count":        (N_BANDS, 0.0, 1.0),    # D72, log1p(C)/log1p(ref), Y-gated
    "band_priority":      (N_BANDS, 1.0, _PRIORITY_HIGH),   # this task, externally sampled/supplied, not computed
    "prev_action":        (N_BANDS, 0.0, 1.0),    # D75, one-hot of the last band chosen
    "prev_reward":        (1, 0.0, 1.0),          # D75, `reward_balance_obs` rescaled
    "prev_hit":           (1, 0.0, 1.0),          # D75, did the last dwell declare anything
}

# "v1" is the frozen 183-wide vector (D34, D49, D55, D67) -- every checkpoint in
# `runs/checkpoints/` before D30 was trained on exactly this, in exactly this
# order, and stays loadable only as long as this entry never changes. "v2" is
# D30's addition: `measured_dbm` (global, forgets every band but the last one
# visited) is replaced by `measured_dbm_band` (persists per band, same
# convention `hit_streak` already set), plus `pulse_width` and the two AoA
# blocks. `pulse_count` (D72) is appended last, after AoA rather than inserted
# among D30's four, so every "v2" offset from that change keeps its position --
# same append-only convention D67 set for "v1" and D30 kept. Default stays
# "v1" (see `ScanEnv.__init__`) so every existing call site -- tests,
# `compare.py`, every trained checkpoint -- is unaffected until something
# opts into "v2" explicitly.
OBS_LAYOUTS: dict[str, tuple[str, ...]] = {
    "v1": ("hit_rate", "visit_density", "staleness", "current_band", "clock",
           "measured_dbm", "hit_streak", "current_hit_streak"),
    "v2": ("hit_rate", "visit_density", "staleness", "current_band", "clock",
           "measured_dbm_band", "hit_streak", "current_hit_streak",
           "pulse_width", "aoa_sin", "aoa_cos", "pulse_count"),
    # This task: "v2" plus `band_priority`, appended last so nothing already
    # loadable under "v1"/"v2" shifts offset. 362 + 36 = 398 wide. Built on "v2"
    # only, not "v1" -- there is no reason to maintain two priority variants,
    # and "v2"'s richer per-band state is the more sensible base to condition
    # a priority-aware policy on.
    "v2p": ("hit_rate", "visit_density", "staleness", "current_band", "clock",
            "measured_dbm_band", "hit_streak", "current_hit_streak",
            "pulse_width", "aoa_sin", "aoa_cos", "pulse_count", "band_priority"),
    # D75: "v2p" plus two in-context blocks, appended last so every "v2p"
    # offset holds. 398 + 2 = 400 wide. This is the RL^2 interface -- a
    # recurrent policy that is shown what its last action returned can run an
    # adaptation rule inside the episode, in its hidden state, with no
    # gradient step. That is what makes it "online" in the sense the PS asks
    # for: no prior library, learning from hits and misses.
    #
    # **`prev_action` was here too, originally, and was removed the next day
    # (2026-09-20) on a direct question: doesn't the LSTM already carry the
    # previous action forward on its own?** For this one block, yes --
    # `current_band` (every layout since "v1") is bit-identical to `prev_action`
    # at every step after the first, since `step()` assigns `_current_band =
    # action` and `_prev_action = action` from the same value. Recorded as a
    # known redundancy when "v3" first shipped, then actually removed once
    # asked rather than kept for the RL^2 interface's own sake. `prev_hit`
    # stays -- it is `current_hit_streak > 0`, also pre-existing information,
    # but there is no companion block making that one redundant in the same
    # direct way, and it costs one column. `prev_reward` is, and remains, the
    # only block here carrying information nothing else in the vector does:
    # an LSTM's hidden state can only carry forward what appeared in its
    # *input* at some point, and no layout before "v3" ever exposed the raw
    # per-step reward -- `hit_rate`/`hit_streak`/`staleness` are running
    # aggregate statistics, not the scalar the policy is actually optimised
    # against. Whether this agent actually reads it is untested; see D75's
    # ablation.
    #
    # **This is an in-place width change** (436 -> 400), the same class of
    # change D49/D55/D67/D72 made before it: every checkpoint trained on the
    # 436-wide shape (`lstm_v3_seed0`, `lstm_v3_seed1`) is now permanently
    # unloadable. Not free, paid on purpose -- see D75's amendment.
    "v3": ("hit_rate", "visit_density", "staleness", "current_band", "clock",
           "measured_dbm_band", "hit_streak", "current_hit_streak",
           "pulse_width", "aoa_sin", "aoa_cos", "pulse_count", "band_priority",
           "prev_reward", "prev_hit"),
}


def obs_width(version: str) -> int:
    """Flat width for one layout: 183 "v1", 362 "v2", 398 "v2p", 400 "v3"."""
    if version not in OBS_LAYOUTS:
        raise ValueError(f"obs_version must be one of {sorted(OBS_LAYOUTS)}, got {version!r}")
    return sum(_BLOCK_SPECS[name][0] for name in OBS_LAYOUTS[version])


def obs_version_for_width(width: int) -> str:
    """The layout whose `obs_width()` is `width` -- the reverse of `obs_width()`.

    For a policy that needs to know its own layout but is only ever handed a
    `gymnasium.spaces.Box` (D79's `BandEncoderFeaturesExtractor`, built by SB3
    from nothing but `observation_space`): every registered width is unique
    today (183/362/398/400), so this is unambiguous, but raises rather than
    guessing if that ever stops being true, and raises if `width` matches no
    registered layout at all -- both cheaper to catch here than as a bad
    gather three layers down.
    """
    matches = [v for v in OBS_LAYOUTS if obs_width(v) == width]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no registered obs_version builds a {width}-wide observation "
            f"(known: {sorted(obs_width(v) for v in OBS_LAYOUTS)})"
        )
    raise ValueError(
        f"{width}-wide is ambiguous between {sorted(matches)} -- pass obs_version explicitly"
    )


@dataclasses.dataclass(frozen=True)
class BandLayout:
    """How one `obs_version`'s flat vector splits into per-band and global parts.

    `band_index[i, j]` is the flat-vector column holding per-band block
    `per_band_blocks[j]`'s value for band `i` -- shape `(N_BANDS,
    len(per_band_blocks))`. `global_index[k]` is the flat-vector column for
    the `k`-th scalar among `global_blocks`, in block order -- shape
    `(sum(width of global_blocks),)`. Both are plain `int64` arrays so a
    caller can gather with them directly (`flat[:, band_index]` ->
    `(rows, N_BANDS, len(per_band_blocks))` in both numpy and torch).
    """

    version: str
    per_band_blocks: tuple[str, ...]
    global_blocks: tuple[str, ...]
    band_index: np.ndarray
    global_index: np.ndarray


def band_layout(version: str) -> BandLayout:
    """Split `version`'s flat observation into its per-band and global parts.

    Purely from `_BLOCK_SPECS`'s declared widths and `OBS_LAYOUTS`' block
    order -- nothing about which *named* block is per-band or global is
    hardcoded here. A block is per-band exactly when `_BLOCK_SPECS` declares
    it `N_BANDS` wide (today: `hit_rate`, `visit_density`, `staleness`,
    `current_band`, `measured_dbm_band`, `hit_streak`, `pulse_width`,
    `aoa_sin`, `aoa_cos`, `pulse_count`, `band_priority`, `prev_action`) and
    global otherwise (`clock`, `measured_dbm`, `current_hit_streak`,
    `prev_reward`, `prev_hit`). This is what lets `prev_action` be handled
    correctly with no special case at all: no `OBS_LAYOUTS` entry uses it
    today (D75 shipped it in "v3", then removed it the next day once
    `current_band` was shown to already carry the same information -- see
    "v3"'s own comment), but it is still `N_BANDS` wide in `_BLOCK_SPECS`, so
    a layout that *did* include it would have it fall into the per-band
    bucket automatically, exactly like any other `N_BANDS`-wide block --
    never silently dropped into "global" or left out of the gather.

    Built for `rfenv/rl/policies.py`'s `BandEncoderFeaturesExtractor` (D79),
    which needs to turn the flat vector into `(N_BANDS, n_band_features)` +
    `(n_global_features,)` without ever hardcoding which columns are which --
    the flat vector interleaves blocks (`hit_rate[0:36]`, `visit_density[0:36]`,
    ...), not bands (`band_0`'s full feature vector, then `band_1`'s, ...), so
    this is a strided gather, not a reshape.
    """
    if version not in OBS_LAYOUTS:
        raise ValueError(f"obs_version must be one of {sorted(OBS_LAYOUTS)}, got {version!r}")

    per_band_blocks: list[str] = []
    global_blocks: list[str] = []
    band_offsets: list[int] = []
    global_columns: list[int] = []

    offset = 0
    for name in OBS_LAYOUTS[version]:
        width = _BLOCK_SPECS[name][0]
        if width == N_BANDS:
            per_band_blocks.append(name)
            band_offsets.append(offset)
        else:
            global_blocks.append(name)
            global_columns.extend(range(offset, offset + width))
        offset += width

    band_index = (np.asarray(band_offsets, dtype=np.int64)[np.newaxis, :]
                  + np.arange(N_BANDS, dtype=np.int64)[:, np.newaxis])
    global_index = np.asarray(global_columns, dtype=np.int64)

    return BandLayout(version=version, per_band_blocks=tuple(per_band_blocks),
                       global_blocks=tuple(global_blocks),
                       band_index=band_index, global_index=global_index)


# --------------------------------------------------------------------------- #
# Reward candidates (D7, D29)
# --------------------------------------------------------------------------- #
#
# **Six since D57, and D29's cap of three is formally lifted there.** The cap
# existed for a real reason and it has not gone away: every candidate scored on
# the same 47 scenarios is another draw, and best-of-many is partly selection
# noise. What changed is that the three added by D57 are not three more guesses
# competing for the same prize -- `greedy` and `explore` are the two corners of
# D14's tension, deliberately expected to fail (one camps, one cannot express
# the bar), and `weighted` is the single knob between them. They are a designed
# axis, so the multiple-comparisons argument applies to choosing a point on it,
# not to six independent tries. D47's selection rule is what must still be run,
# and D57 records that it has not been.
#
# An earlier camping-penalised fourth candidate was registered and retired
# within one working session (see git history / `reward_weighted_camp`, kept
# below as a commented-out draft alongside a further `reward_hybrid` sketch) --
# neither is active.
#
# D29's original candidates 1 (`hit_z`) and 2 (`hit_y`) were **per slot**, and
# a dwell's reward is the sum over its slots (D31), so a 100 ms dwell could earn
# up to +2. That kept reward per unit time equal across wide and narrow bands --
# the alternative, +1 per dwell regardless of length, would make the seven wide
# bands strictly dominated, and those are measurably the bands holding the
# densest emitter populations and the slowest rotators. **Both were retired from
# `REWARDS` 2026-09-10 after failing D62's screen** -- each ranks the camper
# above every sweeping policy, D14's tension in reward form -- but D31's
# per-slot invariant they established still binds every candidate below.
#
# Candidate 3, `reward_balance`, carries the invariant: its camping cost is
# scaled by `dwell.n_slots` so it is charged per slot of airtime spent (D53).
# Its predecessor under this slot, `reward_first_intercept`, was flat per dwell
# instead -- a discovery being worth the same whether the dwell that found it ran
# 1 or 2 slots -- and is kept as a commented-out draft at the end of this file.
# D57's `greedy`/`explore`/`weighted` carry it too, each multiplying its
# state-dependent terms by `dwell.n_slots` for the same reason.
#
# D28 puts the two headline metrics on opposite sides of the Y/Z line -- censored
# intercept time needs Y = 1, interception ratio does not -- so no single reward
# is aligned with both. That is part of why there is more than one candidate,
# and why the rule for choosing between them is a human decision recorded as
# open in D29, now gated behind D62's screen. Nothing here ranks them.
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
      `-3.0 * visit_density[action] * n_slots`      a cost on airtime already
                                                    concentrated on this band

    The camping cost is charged against **airtime share**, not against a
    consecutive-repeat streak. An earlier draft used `-1.0 * camp_slots`, which
    a policy defeats for free by alternating between two bands: `camp_slots`
    resets the instant the action changes, so a 2-band ping-pong paid exactly
    the same total as a full sweep. Measured over 3 seeds, that draft scored the
    ping-pong (coverage 0.261, censored intercept time 16.55 s) at -218.9 and
    round-robin (coverage 0.921, 2.47 s) at -228.5 -- it ranked the failure mode
    *above* the sweep. `visit_density` has no such hole: sustained camping drives
    it to 1.0 and alternating still holds it near 0.5, while round-robin keeps it
    near 1/36. Under this term the same four policies score recency +327.7,
    round-robin +324.3, ping-pong -505.4, camping -1361 -- correctly ordered, and
    a ~1,700-wide range rather than the ~90,000 an unbounded streak counter
    produced, which is a scale the value head can actually fit.

    Scaled by `n_slots` so the cost is per slot of airtime spent, keeping D31's
    per-slot invariant: a wide band costs twice as much because it consumes twice
    as much of the episode.

    Reads `hit_rate`/`visit_density`/`staleness` at `action`, all three of which
    are observation components, so nothing here is information the policy could
    not have acted on itself.
    """
    reward = 0.0

    # D55 rescaled two of this function's three array inputs for the *policy's*
    # benefit -- visit_density into fair shares, staleness into sweeps. The
    # coefficients below were calibrated against the episode-normalised versions
    # (D53's four-policy table), so convert back here rather than re-tuning:
    # this keeps the reward numerically identical across D55, which is the only
    # way the D53 measurement survives an observation change. The reward and the
    # observation still read the same quantities, as D52 requires -- the same
    # quantities in different units.
    visit_density = float(visit_density_array[action]) / N_BANDS
    staleness = float(staleness_array[action]) / _SWEEPS_PER_EPISODE

    # Exploitation
    reward += 0.5 * hit_rate_array[action]

    # Exploration / airtime balance
    reward += 1.0 * (1.5 - visit_density) * staleness

    # Actual useful occupancy
    reward += 0.5 * float(dwell.Z.sum())

    # Cost on concentrated airtime -- see the docstring for why this is charged
    # against visit_density rather than against `camp_slots`.
    reward -= 3.0 * visit_density * dwell.n_slots

    return float(reward)


# The pulse count at which `reward_balance_improved`'s occupancy term pays
# exactly what `reward_balance`'s flat one does. Measured over the **training
# half only** (D60): `expm1(mean(log1p(C)))` across all 252,270 occupied cells of
# the 35 training configs is 62.80, rounded to 64 so the constant reads as a
# chosen reference rather than a fitted decimal -- the rounding moves every
# weight by under 1%.
#
# Fitting it on the validation half instead would give 77.66, and using that
# number would bake twelve held-back scenarios into the environment itself.
# Which is why it is not used, and why the difference is recorded here.
_DENSITY_REF_PULSES = 64.0

# How far the occupancy term moves from flat toward fully density-weighted.
#
#     weight = Z * [ 1 + lambda * (log1p(C)/log1p(64) - 1) ]
#
# `lambda = 0` reproduces `reward_balance`'s flat `0.5 * Z.sum()` exactly, and
# `lambda = 1` is full density weighting. It is a shrinkage parameter toward a
# candidate already known to pass D62's screen, which is what makes the endpoint
# meaningful rather than arbitrary.
#
# **0.5 is justified by what it bounds, not by what it scores.** At this value
# the weight is confined to [0.583, 1.509] -- a 2.6:1 spread against full
# weighting's 12.1:1 -- so **no occupied cell is ever worth less than 58% of what
# `reward_balance` paid for it.** Density can reorder which occupied cell a
# policy should prefer; it cannot make a genuinely occupied cell nearly
# worthless, which is the failure mode that would quietly reintroduce the
# coverage collapse this whole reward exists to prevent.
_DENSITY_SHRINKAGE = 0.5


def reward_balance_improved(
    dwell: DwellResult,
    newly: set[int],
    camp_slots: int,
    hit_rate_array: np.ndarray,
    visit_density_array: np.ndarray,
    staleness_array: np.ndarray,
    action: int,
) -> float:
    """Candidate 7: `reward_balance` with its occupancy term weighted by pulse density.

    **One term differs from `reward_balance`, and nothing else.** That is the
    point: the other three coefficients passed D62's screen at their current
    values, so changing one variable keeps the comparison interpretable and lets
    the screen attribute any failure to density weighting rather than to a
    rebalance.

        reward_balance           `+0.5 * Z.sum()`
        reward_balance_improved  `+0.5 * sum(log1p(C) / log1p(64))`

    ### The mismatch this closes

    `Z` is boolean occupancy and `C` is the pulse count (`truth.py`: `Z[b,t] =
    True` against `np.add.at(C, (b,t), c.n_pulses)`). The **evaluation metric
    counts pulses** -- interception ratio is `pulses_intercepted / total_pulses`
    (`metrics/scoring.py`) -- while `reward_balance` pays a flat 0.5 per occupied
    cell. Measured on the development set, a cell holds between 1 and 4,543
    pulses, the top 10% of occupied cells hold 46-59% of all pulses, and the
    reward pays identically for the sparsest and the densest. The agent is
    optimising "touch many occupied cells" while being scored on "capture many
    pulses", and those diverge exactly as far as pulse mass is concentrated.

    Measured consequence, not a prediction: on `config_921` the two orderings
    already disagree. `reward_balance` ranks rung 5 first (413.2) and the 300k
    RecurrentPPO checkpoint second (407.9); interception ratio ranks that
    checkpoint first (0.1144) and rung 5 **third** (0.0856).

    ### Why `log1p` rather than `C` itself

    Raw `C` spans 1 to 4,543 within a single episode. Paying it linearly makes
    one lucky dwell worth more than the rest of the episode combined and hands
    the value head a target with a three-order-of-magnitude range -- the
    unfittable-scale failure D53 removed from this reward once already, and D57
    removed from `explore` a second time. `log1p` compresses that 4,543:1 span to
    **12:1** (weight 0.17 at one pulse, 2.03 at 4,543): enough that density
    changes the ranking of two candidate dwells, bounded enough that it cannot
    dominate the episode.

    ### Why the magnitude is preserved rather than increased

    `_DENSITY_REF_PULSES` is set so the mean weight over occupied cells is 1.0,
    which keeps this term's episode total where `reward_balance` had it (~42% of
    total reward magnitude) and changes only how it is *distributed* across
    cells. Scaling the term up as well as reshaping it would move two things at
    once, and the camping cost is the only thing holding the policy off the
    densest band -- which is precisely rung 4's exploit. **The risk this
    candidate carries is that density weighting re-creates the camper**, so it
    goes through `python -m rfenv.reward_gate` before anything trains on it, and
    that screen exists to fail it if so.

    Per-slot by construction (D31): `dwell.C` is the per-slot illumination array
    and the term sums over it, so a 2-slot dwell is scored on both its cells.

    Reads `C`, which is truth-side and permitted (D29) -- the same permission
    `reward_balance` already uses for `Z`. The observation was density-blind in
    its per-band blocks when this was written; closing that was an observation
    change (D34/D49/D55), separate from this and not done here.

    **Updated 2026-09-14 (D71): that observation change now exists, opt-in
    (`ScanEnv(obs_version="v2")` adds `measured_dbm_band`, a persistent
    per-band signal correlated with density).** `reward_balance_improved_v2`,
    just below, is the candidate that tests whether that changes anything --
    still shrunk here at 0.5, unchanged, because this candidate's own
    justification (D62-passing at this value) does not depend on which
    observation trains it and there is no reason to move it.
    """
    visit_density = float(visit_density_array[action]) / N_BANDS
    staleness = float(staleness_array[action]) / _SWEEPS_PER_EPISODE

    # Exploitation
    reward = 0.5 * hit_rate_array[action]

    # Exploration / airtime balance
    reward += 1.0 * (1.5 - visit_density) * staleness

    # Useful occupancy, weighted by how much traffic the cell actually held.
    # `Z` is implicit: an unoccupied cell has C = 0 and log1p(0) = 0, so it pays
    # nothing without needing a mask.
    occupied = np.asarray(dwell.Z, dtype=np.float64)
    density = np.log1p(np.asarray(dwell.C, dtype=np.float64)) / np.log1p(_DENSITY_REF_PULSES)
    weight = occupied * (1.0 + _DENSITY_SHRINKAGE * (density - 1.0))
    reward += 0.5 * float(weight.sum())

    # Cost on concentrated airtime -- unchanged, and load-bearing here.
    reward -= 3.0 * visit_density * dwell.n_slots

    return float(reward)


# `reward_balance_improved`'s own docstring named `lambda = 1.0` as untested --
# "has never been swept through the cheap screen." **Swept this session, both
# ends measured, not assumed:**
#
#     lambda = 1.0  (full weighting)   bar-floor 1.9sigma  -- FAILS (< 2.0 bar)
#     lambda = 0.75 (this candidate)   bar-floor 2.1sigma, camper 4.9sigma -- PASSES
#     lambda = 0.5  (reward_balance_improved)  bar-floor 2.3sigma, camper 5.6sigma
#
# `python -m rfenv.reward_gate --rewards reward_balance_improved,reward_balance_improved_v2`,
# this session. Full weighting is not safe *regardless of which observation
# trains it* -- the screen scores four **fixed heuristic** policies (rungs 2,
# 4, 5, 6a), none of which read `measured_dbm_band` or anything obs_version
# could change, so D71's "v2" observation cannot be what saves it here; the
# failure is in the reward's own incentive shape, full stop. 0.75 is the
# highest value that still clears both bars with real margin -- a measured
# choice, not a rounded compromise between 0.5 and the failed 1.0.
_DENSITY_SHRINKAGE_V2 = 0.75


def reward_balance_improved_v2(
    dwell: DwellResult,
    newly: set[int],
    camp_slots: int,
    hit_rate_array: np.ndarray,
    visit_density_array: np.ndarray,
    staleness_array: np.ndarray,
    action: int,
) -> float:
    """`reward_balance_improved` with density weighting pushed from 0.5 to 0.75
    (`_DENSITY_SHRINKAGE_V2`) -- **not** the full 1.0 this was originally built
    to test; 1.0 measurably fails D62's screen (see the constant's own comment
    above for both numbers). Motivated by D71: the "v2" observation (326-wide,
    `ScanEnv(obs_version="v2")`) gives the policy `measured_dbm_band` -- a
    genuine, persistent, per-band signal correlated with a cell's density,
    where "v1" only ever had one global scalar that forgets the instant the
    agent moves on. `reward_balance_improved`'s own docstring named the
    untested case directly: *"lambda = 1.0 has never been swept through the
    cheap screen"* -- shrunk to 0.5 specifically because scoring the policy on
    density it could not anticipate was the exact mismatch D29 rules out.
    That objection weakens under "v2": the correlate is no longer invisible.
    **It does not weaken all the way to 1.0** -- that value fails the screen
    on the reward's own terms, before any question of what a policy can
    perceive even enters it (see above). 0.75 is the highest value measured
    to still pass.

    **Passing D62 is not the same claim as "safe to train."** `measured_dbm_band`
    is correlated with density, not equal to it (`TruthGrid` combines co-located
    emitters by peak amplitude, not by count -- the same limitation
    `PDW_COMPLETENESS_AND_BAND_DENSITY_BRIEF.md`'s `BAND_POWER` proposal named).
    A policy could still learn nothing from the correlation and this candidate
    would then behave exactly like scoring blind density, which is the
    regime `_DENSITY_SHRINKAGE` was capped at 0.5 to bound the damage from.
    That is a training-time question this candidate's existence does not
    answer by itself -- **only screened here (`reward_gate.py`, D62), not
    trained on.** Meant to run *only* against `obs_version="v2"` environments;
    nothing stops it running under "v1", but doing so reopens exactly the
    mismatch this candidate exists to close.

    Every other term is unchanged from `reward_balance_improved` -- same
    exploit, explore, camping-cost, same `_DENSITY_REF_PULSES` reference.
    """
    visit_density = float(visit_density_array[action]) / N_BANDS
    staleness = float(staleness_array[action]) / _SWEEPS_PER_EPISODE

    reward = 0.5 * hit_rate_array[action]
    reward += 1.0 * (1.5 - visit_density) * staleness

    occupied = np.asarray(dwell.Z, dtype=np.float64)
    density = np.log1p(np.asarray(dwell.C, dtype=np.float64)) / np.log1p(_DENSITY_REF_PULSES)
    weight = occupied * (1.0 + _DENSITY_SHRINKAGE_V2 * (density - 1.0))
    reward += 0.5 * float(weight.sum())

    reward -= 3.0 * visit_density * dwell.n_slots

    return float(reward)






# --------------------------------------------------------------------------- #
# The greedy / explore pair, and the blend between them (D57)
# --------------------------------------------------------------------------- #
#
# `reward_balance` mixes exploitation, exploration and a camping cost with fixed
# coefficients, which makes it one point in a space rather than a way of moving
# through it. These three make that space explicit: `reward_greedy` is the pure
# exploit corner, `reward_explore` the pure explore corner, and
# `reward_weighted` a single knob between them. The point is not that any one of
# them is better -- it is that D14's tension can now be measured as a curve
# instead of argued about, and D47's selection rule has something to select
# between.
#
# All three keep D31's per-slot invariant: every term that prices a *state*
# (rather than counting events that already happened per slot) is multiplied by
# `dwell.n_slots`, so a 100 ms dwell on a wide band is worth exactly two 50 ms
# dwells and the seven wide bands are not silently dominated.
#
# All three read the observation arrays in D55's units: `visit_density` in fair
# shares (1.0 = an equal cut of airtime, 36.0 = fully camped) and `staleness` in
# reference sweeps (1.0 = one full pass overdue, 13.95 = untouched all episode).
# That is what makes the coefficients below legible -- each is "how much is one
# natural unit of this worth".


def reward_greedy(
    dwell: DwellResult,
    newly: set[int],
    camp_slots: int,
    hit_rate_array: np.ndarray,
    visit_density_array: np.ndarray,
    staleness_array: np.ndarray,
    action: int,
) -> float:
    """Candidate 4: pure exploitation. Pays for what this band has paid before.

      `+1.0 * Y.sum()`                       what the receiver actually declared
      `+2.0 * hit_rate[action] * n_slots`    for being on a band with a record

    **Deliberately has no exploration term and no camping cost**, and is
    therefore expected to camp. That is what makes it useful: it is the corner
    of the space, the reward-side twin of rung 4, and the control that says how
    much of any blended reward's behaviour comes from its exploit half.

    Distinct from `hit_y` (D29's second candidate, retired from `REWARDS` after
    failing D62's screen), which was also exploit-only and paid only for the
    current dwell's declarations, so a band that had paid out for 400 slots and
    a band never looked at were worth the same until the dwell resolved.
    `reward_greedy` adds the *remembered* rate, so it prices the decision at the
    moment it is made rather than only its outcome -- which is what "greedy" in
    the bandit sense actually means, and what an agent can act on. It is priced
    entirely off the observation plus the dwell, so nothing here is information
    the policy could not have conditioned on itself.

    The 2.0 makes a perfect band (hit rate 1.0) worth about as much per slot as
    two declared hits, so the remembered term can outvote a single unlucky dwell
    without drowning the signal that the band has actually gone quiet.
    """
    reward = 1.0 * float(dwell.Y.sum())
    reward += 2.0 * float(hit_rate_array[action]) * dwell.n_slots
    return float(reward)


def reward_explore(
    dwell: DwellResult,
    newly: set[int],
    camp_slots: int,
    hit_rate_array: np.ndarray,
    visit_density_array: np.ndarray,
    staleness_array: np.ndarray,
    action: int,
) -> float:
    """Candidate 5: pure exploration. Pays for going where it has not been.

      `+1.00 * staleness[action] * n_slots`       how overdue this band was
      `-0.10 * visit_density[action] * n_slots`   what it has already spent here
      `+2.00 * len(newly)`                        emitters found in a new band

    **Not the same thing as uniform coverage**, which is why the third term is
    here. A reward built only from `staleness` and `visit_density` is satisfied
    by any policy that spreads airtime evenly, including one that sweeps past
    every emitter without ever dwelling long enough to declare. `newly` is
    truth-side and D29 permits it; it credits the first intercept of each
    `(emitter, band)` pair (D51), so it pays for *discovering* rather than for
    merely moving. Flat per dwell rather than per slot, deliberately -- a
    discovery is worth the same whether the dwell that found it ran 1 or 2 slots.

    The first two terms are opposite sides of the same quantity and both are
    needed. `staleness` alone is a pull toward the most-overdue band and says
    nothing about a band being over-served; `visit_density` alone is a push away
    from over-served bands and says nothing about which of the rest to pick.

    **The 0.10 is set by the units, not by taste.** `visit_density` reads in fair
    shares since D55, so it reaches 36.0 on a camped band -- a coefficient of 1.0
    would charge 36 per slot and give a camped episode a reward near -22,000
    against a good episode's few hundred. That is the same unfittable-scale
    failure D53 removed from `reward_balance`, reintroduced from the other
    direction, and it was measured here before being fixed: at 2.0 the spread
    across these four policies was 43,000 wide. At 0.10 a camped band costs 3.6
    per slot, which is comparable to the staleness term's own ceiling of 13.95
    and keeps the episode range in the low thousands.

    Expected to score round-robin above rung 5, and that is a real limitation
    rather than an accident: nothing here rewards finding the *busy* bands
    faster, so it cannot express D46's Pareto target on its own. It is the
    corner of the space, not a proposal.
    """
    reward = 1.0 * float(staleness_array[action]) * dwell.n_slots
    reward -= 0.10 * float(visit_density_array[action]) * dwell.n_slots
    reward += 2.0 * float(len(newly))
    return float(reward)


# The blend point `reward_weighted` is registered at.
#
# **0.3 is the midpoint of the only window where the reward orders the ladder
# correctly**, measured across alpha in 0.0..1.0 at 0.1 steps over the four
# reference policies at seeds 0/1/2:
#
#   alpha   recency  round_robin  alternate     camp   sweeps>degen  rung5 top
#     0.0    1286.6       1384.8     -909.1  -2078.3            yes        no
#     0.1    1549.5       1591.4     -229.2  -1179.0            yes        no
#     0.2    1812.3       1798.0      450.7   -279.7            yes       yes
#     0.3    2075.2       2004.7     1130.6    619.7            yes       yes
#     0.4    2338.0       2211.3     1810.5   1519.0            yes       yes
#     0.5    2600.9       2417.9     2490.4   2418.3             NO       yes
#     0.6+   greedy dominates; camping outscores every sweeping policy
#
# Two constraints, and they close from opposite ends. Below 0.2 the explore half
# dominates and round-robin outscores rung 5, so the reward cannot express the
# bar. From 0.5 up, a 2-band ping-pong outscores round-robin -- D53's exact
# failure mode, reintroduced through the greedy half. Only 0.2..0.4 satisfies
# both, and 0.3 is its middle.
#
# **This is a sanity constraint, not tuning against a score.** Nothing here was
# chosen to make a policy perform better on the D27 metrics; what was checked is
# that two known-good policies outrank two known-degenerate ones, which is the
# same check D53 applied. The performance question -- which candidate to
# actually select -- is D47's paired-dominance rule, and it has still never been
# run. `make_reward_weighted` is public so that sweep can build the rest of the
# curve without touching the registry.
WEIGHTED_ALPHA = 0.3


def make_reward_weighted(alpha: float):
    """Candidate 6, as a family: `alpha * greedy + (1 - alpha) * explore`.

    `alpha = 1.0` is `reward_greedy` up to a positive constant (`_GREEDY_GAIN`,
    which cannot change an optimal policy), `alpha = 0.0` is exactly
    `reward_explore`, and the registered `reward_weighted` sits at
    `WEIGHTED_ALPHA`. One knob, both corners reachable, and the whole curve
    between them buildable without editing this file -- which is the difference
    between a reward you can run D47's paired-dominance rule over and a reward
    you can only argue about.

    **The two halves are scaled to be comparable before blending**, or `alpha`
    would not mean what it says -- a half with ten times the spread wins every
    blend regardless of the weight. `_GREEDY_SCALE` divides the greedy half by
    the measured ratio of the two spreads across the four reference policies, so
    a unit of `alpha` moves the blend by comparable amounts at both ends. See
    that constant for the numbers it came from.

    Raises on an alpha outside [0, 1]: a blend weight outside the corners is
    almost always a typo, and silently extrapolating produces a reward that
    punishes the thing it names.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")

    def reward_weighted(
        dwell: DwellResult,
        newly: set[int],
        camp_slots: int,
        hit_rate_array: np.ndarray,
        visit_density_array: np.ndarray,
        staleness_array: np.ndarray,
        action: int,
    ) -> float:
        greedy = reward_greedy(dwell, newly, camp_slots, hit_rate_array,
                               visit_density_array, staleness_array, action)
        explore = reward_explore(dwell, newly, camp_slots, hit_rate_array,
                                 visit_density_array, staleness_array, action)
        return float(alpha * greedy * _GREEDY_GAIN + (1.0 - alpha) * explore)

    reward_weighted.__doc__ = (
        f"`{alpha:.2f} * greedy * {_GREEDY_GAIN:g} + {1.0 - alpha:.2f} * explore` "
        f"-- see `make_reward_weighted`."
    )
    return reward_weighted


# What the greedy half is multiplied by before blending, so that `alpha` moves
# the result by comparable amounts at both ends. Measured over the four reference
# policies at seeds 0/1/2: the greedy half spans 849.0 across them (camp +1694.8
# down to round-robin +845.8) and the explore half spans 3463.1 (round-robin
# +1384.8 down to camp -2078.3), a ratio of 4.08.
#
# **This is a units correction, not a tuned coefficient.** Multiplying a reward
# by a positive constant leaves its optimal policy untouched, so nothing about
# `reward_greedy` or `reward_explore` alone depends on it -- it exists only so
# that `alpha = 0.5` is actually half-and-half rather than the 10:1 explore-
# dominated blend it was before this was measured.
_GREEDY_GAIN = 4.08

reward_weighted = make_reward_weighted(WEIGHTED_ALPHA)


REWARDS = {
    "reward_balance": reward_balance,
    "reward_balance_improved": reward_balance_improved,
    "reward_balance_improved_v2": reward_balance_improved_v2,
    "greedy": reward_greedy,
    "explore": reward_explore,
    "weighted": reward_weighted,
}

DEFAULT_REWARD = "reward_balance"  # the one used to train the registered checkpoints, and the one rung 9's base variant is scored on


def priority_reward_bonus(
    band_priority_value: float,
    n_newly: int,
    visit_density_value: float,
    n_slots: int,
    *,
    priority_coef: float,
    occupancy_coef: float,
    occupancy_decay_cap: float,
) -> float:
    """D74 follow-up: the band-priority reward, now with a second, decaying
    per-slot term -- a deliberately separate function, not a `REWARDS` entry
    and not a change to `reward_balance`/any existing candidate. Composable
    the same way D74's single-term version was: `ScanEnv.step()` adds this on
    top of whichever registered reward is selected.

    **Two terms, both scaled by `band_priority_value` directly** (not a binary
    "is this band elevated" gate) -- the same convention D74's own discovery
    term used, and for the same reason: `priority_uniform=True` (the control
    arm) sets every band's priority to a constant `1.0`, so both terms still
    fire, at the same uniform scale, on every band -- there is no elevated/
    ordinary split for either term to react to differently. Only under real
    (non-uniform) priority does either term actually discriminate between
    bands. Getting this wrong -- gating the occupancy term on "elevated or
    not" instead -- would have made it fire only for the treatment arm and
    never for the control, breaking the "same reward-scale, only the
    differential differs" comparison D74's whole methodology depends on.

    `discovery = priority_coef * band_priority_value * n_newly` -- unchanged
    from D74, still gated on a **new** discovery (`newly`), never occupancy,
    so it inherits none of D53's camping exploit.

    `occupancy = occupancy_coef * band_priority_value * decay * n_slots`,
    where `decay = max(0, 1 - visit_density_value / occupancy_decay_cap)`.
    This is new -- a genuine per-slot bonus, requested directly despite the
    D53 risk, so it earns its own justification for why it does not reopen
    that hole: D53's exploit worked because `camp_slots` was a *consecutive*-
    dwell counter that reset to zero the instant the action changed, so a
    two-band ping-pong never paid the cost a real camper did. `decay` here is
    anchored to `visit_density_value` instead -- the same fix D55 already
    applied to `reward_balance`'s own camping cost -- which accumulates
    across the *whole episode* regardless of what the agent did in between,
    is never reset by switching away and back, and is already the box's own
    per-band airtime-share reading (`1.0` = an equal cut). A two-band
    ping-pong between elevated bands still drives both of their
    `visit_density` up over time, still decaying the bonus on both, same as
    genuine sustained camping would. `occupancy_coef=0.0` (`ScanEnv`'s
    default) makes this term exactly zero always -- D74's own recorded runs,
    and every call site that predates this function, are unaffected.
    """
    discovery = priority_coef * band_priority_value * n_newly
    occupancy = 0.0
    if occupancy_coef and occupancy_decay_cap > 0:
        decay = max(0.0, 1.0 - visit_density_value / occupancy_decay_cap)
        occupancy = occupancy_coef * band_priority_value * decay * n_slots
    return discovery + occupancy


# The widest `reward_balance_obs` can be, by construction rather than by
# measurement -- each term is separately bounded:
#
#     +0.5 * hit_rate[a]        hit_rate in [0, 1]             ->  [0.0, +0.5]
#     +1.0 * (1.5 - vd) * st    vd, st in [0, 1]               ->  [0.0, +1.5]
#     +0.5 * Y.sum()            Y.sum() <= n_slots <= 2        ->  [0.0, +1.0]
#     -3.0 * vd * n_slots       vd in [0, 1], n_slots <= 2     ->  [-6.0,  0.0]
#                                                      total   ->  [-6.0, +3.0]
#
# Checked against the code rather than trusted: over 23,025 steps of
# round_robin/recency/camper x 3 seeds x 5 stare replays the realised range was
# [-4.5053, +2.5000] (p1 -3.3578, p99 +1.5000, mean -0.0065, median +0.0392),
# and a deliberate camper pinned to a wide 2-slot band -- the worst case the
# negative term admits -- reached -5.9992, approaching the analytic -6.0 without
# crossing it. Both measured 2026-09-19.
#
# Symmetric on purpose: it puts exactly-zero reward at exactly 0.5 after
# rescaling, which is what makes the first-step sentinel in `_observation_blocks`
# honest rather than an arbitrary fill value. Set from the analytic bound, not
# the measured percentiles, so the clip is mathematically incapable of binding.
_REWARD_OBS_CLAMP = 6.0

# How many past values of a corrupted block the ablation resamples from. Long
# enough that the pool is a fair sample of the block's distribution across the
# episode, short enough that it stays bounded on an hour-long mission.
_CORRUPT_HISTORY_CAP = 256


def reward_balance_obs(
    dwell: DwellResult,
    newly: set[int],
    camp_slots: int,
    hit_rate_array: np.ndarray,
    visit_density_array: np.ndarray,
    staleness_array: np.ndarray,
    action: int,
) -> float:
    """`reward_balance`, with `Y` in place of `Z` -- what the dwell *came back with*.

    This is not a reward candidate and is deliberately **not** in `REWARDS`:
    nothing trains on it, `reward_gate.py` does not screen it, and D29's
    truth-may-be-read rule is untouched. It exists only to be shown to the policy
    as the "v3" layout's `prev_reward` block (D75).

    **Why a separate function at all.** `reward_balance`'s third term is
    `0.5 * dwell.Z.sum()`, and `Z` is threshold-free truth -- whether an emitter
    was transmitting, whether or not the receiver could hear it. A deployed
    receiver never learns that. Feeding the training reward into the observation
    would therefore hand the agent a channel no real instrument has: it already
    holds `hit_rate`, `visit_density`, `staleness` and `n_slots` in its input, so
    given the reward it could solve the remaining term for `Z.sum()` -- including
    emitters it never detected. Substituting `Y`, what the detector actually
    declared, closes that: the two differ by exactly the receiver's own Pd and
    Pfa (0.8421 and 1.35e-3 at the frozen operating point), which is the
    sensitivity limit rather than a missing oracle.

    This is ordinary asymmetric actor-critic, the two halves kept apart:
    `reward_balance` (privileged, reads `Z`) is what the gradient optimises;
    this (deployable, reads `Y`) is what the policy *sees*. Because of that
    split, a "v3" rung stays `deployable=True` and its numbers stay directly
    comparable to every other rung in the ladder.

    Implemented by substitution rather than by copying the formula, so the two
    can never drift apart and `reward_balance`'s own source is untouched -- a
    fact about the file, not an intention. `DwellResult` is frozen, so `replace`
    shares every array by reference and copies nothing.

    Pure: no RNG, no state, no side effects. It is called on every step
    regardless of the layout in use, so that had better remain true -- a single
    draw from `self.np_random` here would shift the receiver's noise stream and
    silently break every seeded comparison in the repository.
    """
    return reward_balance(
        dataclasses.replace(dwell, Z=dwell.Y),
        newly, camp_slots, hit_rate_array, visit_density_array,
        staleness_array, action,
    )


def _normalise_prev_reward(reward_obs: float) -> float:
    """`reward_balance_obs` -> [0, 1], for the observation's `prev_reward` slot.

    Clip then affine -- the same recipe `measured_dbm` already uses to fold an
    open-ended dBm reading into the unit interval. Zero maps to exactly 0.5
    because the clamp is symmetric, which is what lets `reset()` fill this slot
    with "nothing earned yet" truthfully instead of inventing a value.
    """
    clipped = max(-_REWARD_OBS_CLAMP, min(_REWARD_OBS_CLAMP, float(reward_obs)))
    return (clipped + _REWARD_OBS_CLAMP) / (2.0 * _REWARD_OBS_CLAMP)


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
        obs_version: str = "v1",
        band_priority: bool = False,
        priority_coef: float = 0.5,
        priority_n_bands: tuple[int, int] = (3, 6),
        priority_uniform: bool = False,
        priority_high: float = _PRIORITY_HIGH,
        occupancy_coef: float = 0.0,
        occupancy_decay_cap: float = 2.0,
        episode_slots: int | None = None,
        log_window_slots: int | None = None,
        corrupt_obs_blocks: tuple[str, ...] = (),
    ):
        super().__init__()
        if scenario is None and pool is None:
            raise ValueError("ScanEnv needs a `scenario` to replay or a `pool` to sample from")
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {sorted(REWARDS)}, got {reward!r}")
        if obs_version not in OBS_LAYOUTS:
            raise ValueError(f"obs_version must be one of {sorted(OBS_LAYOUTS)}, got {obs_version!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}"
            )
        # This task: a policy cannot condition on `p` if `p` isn't in the vector
        # it sees. Refusing this combination up front is cheaper than a training
        # run whose permutation ablation (see the implementation prompt's
        # Validation section) would only tell you the same thing hours later --
        # the agent "ignored" a signal that was never observable to begin with.
        if band_priority and "band_priority" not in OBS_LAYOUTS[obs_version]:
            raise ValueError(
                f"band_priority=True needs an obs_version whose layout includes "
                f"'band_priority' (e.g. 'v2p'), got obs_version={obs_version!r}"
            )
        self._scenario = scenario          # fixed world to replay every reset() (validation gates)
        self._pool = pool                  # draw a fresh scenario each reset() instead (RL training, D25/D32)
        self.reward_name = reward          # the REWARDS key, kept around for episode_metrics()/logging
        self._reward_fn = REWARDS[reward]  # resolved once here, not looked up by string every step()
        self.gamma = float(gamma_dbm)      # receiver's detection threshold, dBm (frozen, D25)
        self.sigma = float(sigma_db)       # receiver's noise stdev, dB (frozen, D25)
        self.render_mode = render_mode     # None (default, render() is a no-op) or "rgb_array"
        self._obs_version = obs_version    # "v1" (183-wide, D34/D49/D55/D67) or "v2" (362-wide, D30/D72) or "v2p" (398-wide, this task)
        self._obs_layout = OBS_LAYOUTS[obs_version]
        # This task: band-priority reward, off by default (every existing call
        # site -- tests, compare.py, every trained checkpoint -- is unaffected
        # until a caller opts in explicitly, the same convention obs_version
        # itself uses). `priority_uniform=True` is the control arm: `p` is
        # resampled every reset() like normal, except it always comes out all
        # ones -- same reward term, same scale, no elevated bands, so a
        # control-vs-treatment comparison isn't confounded by the reward-scale
        # shift the term's own docstring (in step()) flags.
        self._priority_enabled = band_priority
        self._priority_coef = float(priority_coef)
        self._priority_n_bands = priority_n_bands
        self._priority_uniform = bool(priority_uniform)
        # D74 follow-up: `priority_high` makes the elevated value itself
        # tunable (was the hardcoded module constant `_PRIORITY_HIGH`) --
        # D74's own recorded runs used the default (3.0) and stay reproducible
        # unchanged. `occupancy_coef=0.0` (default) exactly reproduces every
        # behaviour that existed before this task: the occupancy bonus below
        # is strictly additive on top of the discovery bonus D74 measured, off
        # unless explicitly asked for.
        self._priority_high = float(priority_high)
        self._occupancy_coef = float(occupancy_coef)
        self._occupancy_decay_cap = float(occupancy_decay_cap)
        # D76: how long one episode runs. `None` means `N_SLOTS`, and every
        # existing call site takes that path unchanged -- `constants.py` is
        # frozen (D42) and nothing here moves it; this is a per-env length, not
        # a new constant. Above 600 the world is stitched from whole recordings
        # (`TruthGrid.stitch`), so the length has to be a multiple of one.
        self.episode_slots = N_SLOTS if episode_slots is None else int(episode_slots)
        if self.episode_slots < 1:
            raise ValueError(f"episode_slots must be >= 1, got {self.episode_slots}")
        if self.episode_slots > N_SLOTS and self.episode_slots % N_SLOTS:
            raise ValueError(
                f"episode_slots above {N_SLOTS} must be a whole number of "
                f"{N_SLOTS}-slot recordings, got {self.episode_slots}"
            )
        # A fixed scenario is one 30 s recording. There is no honest way to
        # stretch it over an hour -- tiling it would hand the agent an exactly
        # periodic world to memorise -- so refuse rather than invent one.
        if self.episode_slots > N_SLOTS and self._pool is None:
            raise ValueError(
                "episode_slots > N_SLOTS needs a `pool` to draw segments from; a "
                "single fixed `scenario` is one 30 s recording and tiling it "
                "would make the world exactly periodic"
            )
        # The per-slot log is the dominant memory term over a long mission
        # (measured: ~20 MB/simulated hour of dict overhead alone, against ~80 MB
        # for the grid). A window bounds it, at the cost of the artefact writer,
        # which needs the whole episode -- `metrics.artefacts.write_run` refuses
        # a windowed env rather than writing a log that silently isn't one.
        self.log_window_slots = None if log_window_slots is None else int(log_window_slots)
        if self.log_window_slots is not None and self.log_window_slots < 1:
            raise ValueError(
                f"log_window_slots must be >= 1 or None, got {self.log_window_slots}"
            )
        # Ablation support (D75): named blocks are replaced, per step, by a draw
        # from their own recent history before the vector is assembled. This is
        # a property of the *environment a checkpoint is scored in*, not of the
        # checkpoint, which is why it lives here and not on a rung -- the same
        # trained weights are run clean and corrupted, and only this differs.
        #
        # "Corrupted", not "permuted": permuting a width-1 block is the identity,
        # so it would silently test nothing. Resampling from the block's own
        # recent values preserves its marginal distribution while destroying the
        # correspondence between the value and the state that produced it, which
        # is the thing under test, and it means the same rule applies to a
        # one-hot and to a scalar.
        self._corrupt_obs_blocks = tuple(corrupt_obs_blocks)
        unknown = [b for b in self._corrupt_obs_blocks if b not in _BLOCK_SPECS]
        if unknown:
            raise ValueError(
                f"corrupt_obs_blocks names no such block(s): {unknown}; "
                f"choose from {sorted(_BLOCK_SPECS)}"
            )
        not_in_layout = [
            b for b in self._corrupt_obs_blocks if b not in OBS_LAYOUTS[obs_version]
        ]
        if not_in_layout:
            raise ValueError(
                f"corrupt_obs_blocks names {not_in_layout}, which obs_version="
                f"{obs_version!r} does not carry -- corrupting a block the policy "
                f"never sees would measure nothing"
            )
        self._corrupt_history: dict[str, list[np.ndarray]] = {}
        # A separate stream, never `self.np_random`. Drawing the corruption from
        # the env's own generator would shift the receiver's noise draws, so the
        # clean and corrupted runs would no longer be the same episode and the
        # paired comparison would stop being paired -- the exact failure mode
        # `_seed_torch`'s docstring in `ladder.py` exists to prevent.
        self._corrupt_rng = np.random.default_rng(0)

        self.action_space = spaces.Discrete(N_BANDS)   # one of the 36 bands, chosen every step()
        # Built from `_BLOCK_SPECS`/`OBS_LAYOUTS` (D30), not a literal shape --
        # "v1" is 36 x 5 + 3 = 183 (D34, extended by D49, rescaled by D55,
        # extended again by D67); "v2" is 326 (D30) + 36 (`pulse_count`, D72) = 362.
        #
        # **The box is no longer the unit interval**, and that is the whole point
        # of D55. Two blocks are deliberately scaled past 1.0 so that 1.0 means
        # something in each of them rather than "the theoretical maximum nobody
        # reaches": visit_density is airtime share over *fair* share, so 1.0 is
        # equal airtime and 36.0 is a fully camped episode; staleness is measured
        # in reference sweeps, so 1.0 is "one full pass overdue" and 13.95 is a
        # band untouched all episode. Under the old episode-normalised scaling
        # both sat in the bottom tenth of [0, 1] -- measured, visit_density had
        # mean 0.0278 and p99 0.065, staleness mean 0.070 -- which is why rung 5
        # has to divide staleness back out by hand to work at all
        # (`baselines/recency.py`). An honest box beats a tidy one; SB3 does not
        # rescale inputs, so the numbers the network sees are these.
        # D74 follow-up: `band_priority`'s declared ceiling has to track
        # `priority_high`, not `_BLOCK_SPECS`'s own static default (3.0) --
        # otherwise a caller passing `priority_high > 3.0` produces episodes
        # whose actual values fall outside the space `observation_space`
        # itself declares, and `gymnasium.utils.env_checker.check_env` (and
        # SB3, less visibly) catches exactly that as a contract violation.
        # Every other block is unaffected: this only overrides one name.
        def _high_for(name: str) -> float:
            return self._priority_high if name == "band_priority" else _BLOCK_SPECS[name][2]

        low = np.concatenate(
            [np.full(_BLOCK_SPECS[n][0], _BLOCK_SPECS[n][1], dtype=np.float32) for n in self._obs_layout]
        )
        high = np.concatenate(
            [np.full(_BLOCK_SPECS[n][0], _high_for(n), dtype=np.float32) for n in self._obs_layout]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Placeholders only -- the real values are per-episode state, built
        # fresh by reset() (never carried from one episode into the next, D20).
        self.grid: TruthGrid | None = None       # this episode's L1 truth grid
        self.receiver: Receiver | None = None    # this episode's L2 detector
        self.log: list[dict] = []                # per-slot rows so far, see _append_log

    # ----------------------------------------------------------------- reset --

    @property
    def last_reward_obs(self) -> float:
        """The most recent step's `reward_balance_obs` -- the deployable reward.

        Exposed because online fine-tuning (D77) optimises *this*, not `step()`'s
        return value: a gradient taken against a reward that reads `Z` describes
        a simulator, not a receiver. `rfenv.rl.online.ObservableRewardWrapper`
        is the only consumer.
        """
        return self._prev_reward_obs

    def _build_grid(self, scenario: Scenario) -> TruthGrid:
        """This episode's world: one recording, or several laid end to end (D76).

        At the default length this is `TruthGrid.from_scenario(scenario)` and
        nothing else -- the same call, consuming the RNG the same way -- so every
        existing episode is bit-identical. The first segment is always the
        scenario already chosen by `reset()`, so a seeded run's opening 30 s does
        not depend on whether the mission is long.
        """
        if self.episode_slots <= N_SLOTS:
            return TruthGrid.from_scenario(scenario)

        n_segments = self.episode_slots // N_SLOTS
        grids = [TruthGrid.from_scenario(scenario)]
        for _ in range(n_segments - 1):
            grids.append(
                TruthGrid.from_scenario(Scenario.sample(self._pool, self.np_random))
            )
        return TruthGrid.stitch(
            grids, name=f"mission:{n_segments}x{N_SLOTS}:{scenario.name}"
        )

    def reset(self, seed=None, options=None):
        """Start an episode. `options={"scenario": ...}` overrides for this one.

        Seeding seeds *everything* stochastic: the scenario draw and the
        receiver's noise both run off `self.np_random`, so a seed reproduces an
        episode exactly -- which is what `EVALUATION.md` §7 means by comparing
        schedulers "on identical scenarios and seeds".
        """
        super().reset(seed=seed)

        # The ablation's own stream, reseeded per episode from the episode seed
        # but drawn from a separate generator (D75). Derived from `seed` rather
        # than from `self.np_random` so that not one bit of the receiver's noise
        # sequence moves between a clean run and a corrupted one -- that is what
        # makes the pair a paired comparison.
        self._corrupt_history = {}
        if seed is not None:
            self._corrupt_rng = np.random.default_rng(
                np.random.SeedSequence(entropy=int(seed), spawn_key=(0xC0BB1E,))
            )

        scenario = (options or {}).get("scenario") or self._scenario
        if scenario is None:
            scenario = Scenario.sample(self._pool, self.np_random)
        self.grid = self._build_grid(scenario)
        self.scenario = self.grid.scenario
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
        # D75, the "v3" in-context blocks. Cold-start values, consistent with
        # every other array here (D20): nothing has been earned and nothing has
        # been declared, so 0.0 and False are the literal truth rather than a
        # fill. `_prev_reward_obs` reaches the observation through
        # `_normalise_prev_reward`, where 0.0 maps to exactly 0.5.
        self._prev_reward_obs = 0.0
        self._prev_hit = False
        self._camp_slots = 0          # length of the current same-band streak, in slots (reward arg only since D55)
        self._hit_streak = np.zeros(N_BANDS, dtype=np.int64)   # consecutive declared hits per band, across visits -> hit_streak (D67)
        # Never-measured reads as the noise floor -- quietest possible, not the
        # loudest -- consistent with every episode starting cold (D20): no
        # measurement carries over from a previous episode.
        self._last_measured_dbm = NOISE_FLOOR_DBM   # raw dBm, un-normalised -> measured_dbm (clamped+scaled in _observation)
        # D30, "v2"-only state -- built regardless of `self._obs_version` (cheap,
        # and keeps `_observation_blocks()` a single unconditional function; only
        # the layout selection in `_observation()` decides what's actually seen).
        self._band_measured_dbm = np.full(N_BANDS, NOISE_FLOOR_DBM, dtype=np.float64)  # per band, un-normalised -> measured_dbm_band
        self._band_pulse_width = np.zeros(N_BANDS, dtype=np.float64)   # us, 0.0 = never a declared hit here -> pulse_width
        self._band_aoa_sin = np.zeros(N_BANDS, dtype=np.float64)       # raw sin(theta); (0,0) together = never measured -> aoa_sin
        self._band_aoa_cos = np.zeros(N_BANDS, dtype=np.float64)       # raw cos(theta); (0,0) together = never measured -> aoa_cos
        self._band_pulse_count = np.zeros(N_BANDS, dtype=np.float64)   # raw C at the last declared hit, 0.0 = never -> pulse_count (D72)
        self._hit_rate_array = np.zeros(N_BANDS, dtype=np.float32)  # hit_rate per band, 0..1 -> hit_rate_array 
        self._visit_density_array = np.zeros(N_BANDS, dtype=np.float32)  # visit_density per band, 0..1 -> visit_density_array
        self._staleness_array = np.ones(N_BANDS, dtype=np.float32)  # staleness per band, 0..1 -> staleness_array
        # This task: band-priority vector, resampled every episode so the
        # policy has to read it rather than memorise one setting (D70's open
        # item 2). Uses self.np_random, not np.random/np.random.default_rng --
        # reset()'s own docstring guarantee ("seeding seeds everything
        # stochastic") has to hold for this too, or paired-seed comparisons
        # (EVALUATION.md §7, reward_gate's own screen convention) silently
        # stop being paired. All-ones (1.0 = "ordinary") is both the disabled
        # default and the untasked value even when enabled -- 0 bands elevated
        # this episode is a legitimate draw, not a special case.
        self._band_priority = np.ones(N_BANDS, dtype=np.float32)
        if self._priority_enabled and not self._priority_uniform:
            lo, hi = self._priority_n_bands
            k = int(self.np_random.integers(lo, hi + 1))
            elevated = self.np_random.choice(N_BANDS, size=k, replace=False)
            self._band_priority[elevated] = self._priority_high
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
        if self.t >= self.episode_slots:
            raise RuntimeError("episode is over; call reset()")
        # The three arrays below are read by the reward, at `action` only, and
        # every one of them is measured BEFORE this dwell is applied -- so a
        # reward prices the band as the agent saw it when it chose, not as the
        # dwell it is being paid for has already made it. They mirror the
        # corresponding blocks of `_observation()` exactly; nothing here is a
        # quantity the policy could not have conditioned on itself (D29).
        self._current_band = action
        elapsed = max(self.t, 1)

        # Declared hits per slot of airtime spent on this band, 0 where it has
        # never been looked at.
        self._hit_rate_array[action] = (
            self._hits[action] / self._slots_looked[action]
            if self._slots_looked[action] > 0
            else 0.0
        )

        # Airtime share on this band, in fair shares: 1.0 is an equal cut of the
        # episode so far, 36.0 is all of it. Same scaling as `_observation()`
        # (D55) -- these arrays exist to be the observation's numbers.
        self._visit_density_array[action] = (
            self._slots_looked[action]
            / elapsed
            * N_BANDS
        )

        # How long this band has been neglected: slots since its last dwell, in
        # reference sweeps (D55), with never-visited reading the episode's worth
        # of sweeps (maximally stale). This is the same formula and the same
        # convention `_observation()` uses, and it has to be -- `reward_balance` prices
        # exploration off this array and its docstring promises the agent could
        # have acted on the same numbers. It previously stored
        # `_last_slot[action] / N_SLOTS`, which is *when* the band was last seen
        # rather than *how long ago*: the two are inverses, so the exploration
        # term paid most for revisiting the freshest band and paid a small
        # negative for returning to one abandoned 500 slots earlier (its element
        # was still the -1/600 written on its first-ever visit). Measured on a
        # seed-0 episode at t=503: +0.419 for the band just left against -0.0025
        # for a band untouched since slot 1.
        if self._last_slot[action] >= 0:
            staleness = (
                self.t - self._last_slot[action]
            ) / SWEEP_SLOTS
        else:
            staleness = _SWEEPS_PER_EPISODE
        # D76: clipped to the declared ceiling, which is what keeps the matching
        # observation block inside its Box on a mission longer than one recording
        # -- a band untouched for a simulated hour would otherwise read 1674
        # against a declared 13.95. At the default length the clip provably never
        # binds (the most stale a band can be is (N_SLOTS-1)/SWEEP_SLOTS = 13.930,
        # below the 13.953 ceiling), so this is bit-identical for every episode
        # that ever ran. It has to be applied here *and* in `_observation_blocks`
        # identically, or the reward stops pricing the number the agent sees,
        # which is the promise D52 exists to keep.
        self._staleness_array[action] = min(staleness, _SWEEPS_PER_EPISODE)
        dwell = self.receiver.dwell(self.grid, action, self.t)

        # This dwell's mean measured level -- S + noise, what the receiver's
        # detector actually read, not the truth-side S alone (D19: observable).
        self._last_measured_dbm = float(dwell.measured_dbm.mean())
        self._band_measured_dbm[action] = self._last_measured_dbm

        # D30: PulseWidth/AoA, gated on `Y` -- a real receiver only measures a
        # pulse's width and bearing on a pulse it actually detected, unlike
        # amplitude (a continuous quantity present in every slot). Within a
        # multi-slot dwell with more than one declared hit, the loudest one is
        # the representative reading -- consistent with how `grid.PW`/`grid.AOA`
        # already resolve multiple *emitters* sharing one cell (`truth.py`).
        #
        # D72: `dwell.C`, gated the same way. `C` counts illuminations inside
        # the cell with no gamma gate at all (`truth.py`) -- it includes
        # contributions that would never individually cross the threshold, so
        # it is not literally "how many PDWs a real receiver logged here."
        # Gating on `Y` narrows that gap to cells the receiver actually
        # declared -- a real ES receiver's own PDW stream does carry a count
        # of overlapping detections it resolved, which `C` on a declared hit
        # is the closest quantity this simulator has to that. The remaining
        # gap (sub-threshold contributors folded into a hit's count) is
        # recorded, not hidden -- see D72.
        hit_idx = np.flatnonzero(dwell.Y)
        if hit_idx.size:
            loudest = hit_idx[np.argmax(dwell.measured_dbm[hit_idx])]
            self._band_pulse_width[action] = float(dwell.pulse_width_us[loudest])
            aoa_rad = np.deg2rad(float(dwell.aoa_deg[loudest]))
            self._band_aoa_sin[action] = float(np.sin(aoa_rad))
            self._band_aoa_cos[action] = float(np.cos(aoa_rad))
            self._band_pulse_count[action] = float(dwell.C[loudest])

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

        # This band's own consecutive-hit streak (D67): incremented on any
        # declaration this dwell, reset to 0 on a dwell with none -- tracked per
        # band across visits, not per action-in-a-row (`_camp_slots` already
        # covers that). A band visited rarely but hot every time it is stays hot
        # here; `visit_density` cannot say that, only how much airtime went to it.
        if dwell.Y.sum() > 0:
            self._hit_streak[action] += 1
        else:
            self._hit_streak[action] = 0

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

        # D75: the observable twin of the reward, for the "v3" layout's
        # `prev_reward` block. Computed from *this same* pre-dwell snapshot --
        # the three arrays were written at `action` above and nothing touches
        # them again this step, so the two values price the identical state and
        # differ only in `Z` versus `Y`. Computed unconditionally, for every
        # layout, matching `_observation_blocks`'s own rule of building
        # everything and letting the layout select; it is pure, so doing so
        # cannot perturb a run that never reads it.
        #
        # It is never added to `reward` and never reaches `total_reward`: this
        # is what the policy *sees*, not what it is *paid*. A test pins that.
        self._prev_reward_obs = float(
            reward_balance_obs(
                dwell,
                newly,
                self._camp_slots,
                self._hit_rate_array,
                self._visit_density_array,
                self._staleness_array,
                self._current_band,
            )
        )
        self._prev_hit = bool(dwell.Y.any())

        # This task, extended by D74's follow-up (`priority_reward_bonus`):
        # band-priority reward, additive on top of whatever `self._reward_fn`
        # produced -- it does not touch `REWARDS` or `reward_gate.py`'s
        # screen, which still runs against the base reward candidates
        # unmodified. `dwell.band`, not `action`: every emitter in `newly`
        # this step was found on the band this dwell actually covered, and
        # `visit_density_array[dwell.band]` at this point already reflects
        # this dwell's slots (updated above). See `priority_reward_bonus`'s
        # own docstring for the discovery/occupancy split, the D53 exploit
        # this has to stay clear of, and why both terms scale by
        # `band_priority_value` directly rather than an elevated/not gate.
        if self._priority_enabled:
            reward += priority_reward_bonus(
                float(self._band_priority[dwell.band]),
                len(newly),
                float(self._visit_density_array[dwell.band]),
                dwell.n_slots,
                priority_coef=self._priority_coef,
                occupancy_coef=self._occupancy_coef,
                occupancy_decay_cap=self._occupancy_decay_cap,
            )

        self.total_reward += reward
        # `terminated`, not `truncated` (D35): the 30 s horizon is the task
        # definition -- Turing's own `collection_time_s`, and the extent of the
        # world the grid describes -- not an artificial cap on an ongoing task.
        # There is no state beyond slot 600 to bootstrap a value from.
        terminated = self.t >= self.episode_slots
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
        """The flat vector SB3 sees: `_observation_blocks()`, concatenated in
        `self._obs_layout`'s order (`OBS_LAYOUTS[self._obs_version]`, D30) --
        183-wide for "v1", 362-wide for "v2" (D72 added `pulse_count`, 326 -> 362).
        `observation_space` (`__init__`) is built from the same layout, so the
        two can never disagree.
        """
        blocks = self._observation_blocks()
        if self._corrupt_obs_blocks:
            blocks = self._corrupt(blocks)
        return np.concatenate([blocks[name] for name in self._obs_layout]).astype(np.float32)

    def _corrupt(self, blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Replace the named blocks with a draw from their own recent history.

        The ablation (D75). Each corrupted block keeps its own marginal -- the
        values are ones this block really did take, in this episode -- but loses
        any correspondence with the state that produced them. A policy that
        conditions on the block degrades; a policy that ignores it does not
        notice, which is the whole measurement.

        Draws from `self._corrupt_rng`, never `self.np_random`, so the receiver's
        noise sequence is bit-identical between the clean and corrupted runs of
        the same seed. Without that the two are different episodes and the paired
        test is meaningless.

        The first observation of an episode has no history to draw from, so it
        passes through uncorrupted -- one step out of several hundred, and the
        alternative is inventing values from outside the block's own support.
        """
        out = dict(blocks)
        for name in self._corrupt_obs_blocks:
            history = self._corrupt_history.setdefault(name, [])
            if history:
                out[name] = history[self._corrupt_rng.integers(len(history))]
            # Record the *true* value, so the pool stays the block's real
            # distribution rather than a resampling of its own resamplings.
            history.append(np.array(blocks[name], copy=True))
            if len(history) > _CORRUPT_HISTORY_CAP:
                del history[0]
        return out

    def _observation_blocks(self) -> dict[str, np.ndarray]:
        """Every named block this environment knows how to build, keyed by name
        (`_BLOCK_SPECS`) -- built unconditionally regardless of `self._obs_version`
        (cheap: 36-element numpy ops), so this stays the one function that can
        disagree with the layout table, and any block can be inspected or tested
        in isolation without threading a version flag through the computation
        below. `_observation()` is the only caller that then picks a subset.

        The 183-vector (D34, extended by D49, rescaled by D55, extended again
        by D67), built from the agent's own scan history alone.

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

        **Two of the three per-band blocks are scaled so that 1.0 means something
        (D55).** `visit_density` is airtime share divided by *fair* share, so 1.0
        is an equal cut and 36.0 a fully camped episode; `staleness` is measured
        in reference sweeps (43 slots), so 1.0 is one full pass overdue and
        600/43 is a band untouched all episode. D34 divided both by the episode
        instead, which pinned visit_density's mean at 1/36 and made staleness
        bimodal -- measured means of 0.028 and 0.070 with everything unvisited
        piled on the 1.0 ceiling. Rung 5 could not use staleness in that form and
        multiplied it back out itself; that correction now lives here, where
        every policy gets it.

        A scalar `camp_time` sat between the clock and `measured_dbm` until D55.
        It was the current same-band streak over `N_SLOTS`, and measured under a
        non-camping policy it took exactly two values -- it could not move unless
        the agent was already camping, and `visit_density` says the same thing
        continuously and earlier.

        A final scalar, `measured_dbm`, is the mean of `S + noise` over the most
        recent dwell's slots -- what the receiver's detector actually read
        (D19: observable, unlike the truth-side `S` alone), clamped to
        `[_DBM_CLAMP_MIN, _DBM_CLAMP_MAX]` and linearly rescaled to `[0, 1]` to
        fit the box. Global, not per-band: it reports only the band just left,
        the same scope `current_band` already has. Before any dwell it reads as
        the clamp floor (quietest possible) -- see `reset()`.

        Nothing truth-side appears here. `Z`, per-emitter levels and `first_e` are
        all available to the *reward* (D29) and all absent from this vector.

        **Two more blocks, appended after `measured_dbm` rather than inserted
        among the four above, so every existing slice constant keeps its offset
        (D67).** `hit_streak` is a fifth per-band block: consecutive declared
        hits on that band across separate visits, capped at `_STREAK_CAP` and
        divided down to `[0, 1]`. It exists because `hit_rate` is cumulative over
        the whole episode -- a band hot for the first five slots and cold since
        reads the same as one that just turned hot -- and D66 measured that no
        RecurrentPPO checkpoint trained on the 146-wide vector ever commits to a
        band at all (max dwell streak 6 slots, indistinguishable from
        round-robin's 1-2). Whether that is because the information was missing
        or because nothing in the reward pays for using it is exactly what this
        change tests; it changes only what the agent can see, not what it is
        scored on. `current_hit_streak`, the final scalar, is `hit_streak` at
        whichever band `current_band` is one-hot on -- redundant with the
        per-band block plus a dot product, kept anyway as a direct scalar for
        the same reason `current_band` itself exists alongside `staleness`: not
        every consumer should have to learn to compute it.

        **Four more blocks, "v2" only (D30).** `measured_dbm_band` is
        `measured_dbm`'s per-band form: the global scalar forgets every band but
        the one just left the moment the agent moves on, so this keeps one
        reading per band, same persistence rule `hit_streak` already set.
        `pulse_width` and `aoa_sin`/`aoa_cos` are the two previously-discarded
        PDW columns (raw dataset columns 2 and 3, `rfenv/scenario.py`), read from
        whichever pulse actually set that band's last declared hit (`step()`),
        gated on `Y` -- a real receiver only measures a pulse's width and bearing
        on a pulse it detected, so writing these on a miss would read truth the
        receiver never had, same reasoning D29 already applies to `C`. AoA is
        circular (−180° and 180° are the same bearing), so it is stored as
        `(sin θ, cos θ)` rather than a raw angle, each then rescaled `(x+1)/2`
        into the box's [0, 1] convention; `(0.5, 0.5)` together decode back to
        raw `(0, 0)`, off the unit circle, which no real bearing measurement can
        produce -- a clean "never measured" flag that falls out of the encoding
        for free, not a separate sentinel. **What this does not give the agent:**
        D30's own measurement found AoA's real value is distinguishing an
        already-seen emitter from a new one in a crowded band, which needs
        comparing this reading against a *set* of bearings seen earlier in the
        episode (clustering) -- out of scope here; this block is one raw
        reading, "which direction was the last hit on this band", nothing more.

        **A fifth "v2" block, `pulse_count` (D72), reopens D29/D34's own
        exclusion of `C`.** Both had explicitly ruled truth-side illumination
        count out of the observation -- D34: "what it deliberately excludes:
        pulse count... within the dwell"; D29's governing line: "the
        observation... must contain only what a deployed receiver has." D72 is
        the human sign-off reopening that, on the argument that a real
        receiver's own PDW output *does* carry a count of the detections it
        resolved -- unlike literal `C`, which `truth.py` builds with no gamma
        gate at all and therefore includes contributions that would never
        individually cross the threshold (recorded precisely in
        `docs/project/PDW_COMPLETENESS_AND_BAND_DENSITY_BRIEF.md`, which
        reached the opposite conclusion on the evidence available at the time).
        Gating on `Y`, the same rule `pulse_width`/`aoa_sin`/`aoa_cos` already
        use, closes part of that gap -- only cells the receiver actually
        declared a hit on ever write a value -- but not all of it: `C` on a
        declared-hit cell can still include sub-threshold co-located emitters
        folded into the count. That residual gap is the one honest limit D72
        did not resolve, only narrow. Stored per band, same persistence
        convention as `measured_dbm_band`/`pulse_width` (the last declared
        hit's reading, kept until the next one), and normalised
        `log1p(C) / log1p(_DENSITY_REF_PULSES)` then clipped to `[0, 1]` --
        the identical transform `reward_balance_improved` already applies to
        `C` on the reward side, reused here so the observation and that reward
        agree on what "busy" means, and clipped because raw `C` can exceed the
        reference pulse count within a single episode (recorded: up to 4,543)
        while the box cannot.

        **A "v2p"-only sixth block, `band_priority` (D70, this task).** Unlike
        every block above, it is not computed from the agent's scan history at
        all -- it is exogenous, sampled fresh in `reset()` from `self.np_random`
        (2-6 bands drawn elevated to `_PRIORITY_HIGH`, everything else at 1.0
        = "ordinary"), and read-only from here. D70 already settled that threat
        cannot be derived from receiver signals (best AUC 0.514 across three
        independent attempts) and must come from outside; this is the
        no-library form of that interface -- synthetic tasking standing in for
        an operator, with no claim that any of it is a real threat label.
        Persists across the whole episode (no per-step update; `step()` never
        writes it), so unlike every gated block above it needs no sentinel --
        the value at index *b* is simply "was band *b* tasked this episode,"
        knowable from slot 0 onward. `band_priority=False` (the default)
        leaves it at the all-ones array `reset()` always builds first, so this
        block reads identically whether or not the feature is enabled; only
        the *reward* term in `step()` is gated on `band_priority`, not this
        block's presence in the vector.
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

        # Airtime share, expressed in **fair shares** rather than as a fraction
        # of the episode (D55). The raw fraction sums to 1 across bands (airtime
        # is the only currency, D31), which pins its mean at exactly 1/36 =
        # 0.0278 and -- measured over 1,506 round-robin steps -- its p99 at 0.065.
        # A feature that never leaves the bottom tenth of its declared range is a
        # feature the first layer has to amplify before it can use it, and this
        # is the one `reward_balance` prices camping off. Multiplied by N_BANDS,
        # 1.0 is "this band got exactly its equal share", <1 under-visited, >1
        # over-visited, and 36.0 a fully camped episode.
        visit_density = self._slots_looked / elapsed * N_BANDS

        # Neglect, in **reference sweeps** rather than in episodes (D55). D34
        # divided by N_SLOTS, which made this bimodal and useless in between:
        # measured mean 0.070 with p99 exactly 1.0, because everything visited
        # sits near zero and everything never-visited sits at the 1.0 ceiling.
        # Rung 5 -- the bar the RL rungs have to clear -- could not use it in
        # that form and multiplies it straight back out by N_SLOTS/SWEEP_SLOTS;
        # its own docstring records that skipping that step collapses the rung
        # into rung 4 (coverage 0.526 against round-robin's 0.895). Doing the
        # division here instead means the heuristic and the agent read the same
        # well-scaled number, and the agent no longer has to rediscover a
        # constant that follows from the frozen dwell schedule.
        #
        # 1.0 now means "one full pass overdue". Never-visited reads the ceiling,
        # _SWEEPS_PER_EPISODE -- maximally stale, as before, and still at least
        # as attractive as any band last seen at t=0.
        # Clipped to that same ceiling (D76) so a mission longer than one
        # recording cannot leave the declared Box; never binds at the default
        # length. The identical clip is applied in `step()` where the reward's
        # copy of this array is written -- the two must agree (D52).
        staleness = np.minimum(
            np.where(
                self._last_slot >= 0,
                (self.t - self._last_slot) / SWEEP_SLOTS,
                _SWEEPS_PER_EPISODE,
            ),
            _SWEEPS_PER_EPISODE,
        )

        # One-hot: 1.0 at whichever band the most recent dwell was on, 0.0
        # everywhere else -- "where am I right now", distinct from staleness
        # (which the current band also reads as 0, but so would a band left
        # one slot ago).
        current_band = np.zeros(N_BANDS, dtype=np.float64)
        current_band[self._current_band] = 1.0

        # `camp_time` (the current same-band streak / N_SLOTS) was here until
        # D55 and is deliberately gone. Measured under a non-camping policy it
        # took exactly two values, 1/600 and 2/600 -- a constant, because the
        # streak cannot exceed one dwell unless the agent is already camping. It
        # is a gauge that only moves once the wrong thing is happening, and
        # visit_density says the same thing earlier and continuously. Its only
        # consumer, `reward_balance`'s `-1.0 * camp_slots` term, was replaced in
        # D53. `self._camp_slots` is still maintained: every reward candidate
        # takes it in its signature (D31's uniform call shape) and
        # `episode_metrics()` reports the streak.

        # measured_dbm needs a manual normalisation step, unlike every other
        # component above: it is a real dBm value with no natural [0,1] range
        # (S + noise can fall anywhere), whereas hit_rate/current_band/the clock
        # are ratios or one-hot flags that land in [0,1] on their own, and
        # visit_density/staleness carry their own declared ceilings (D55).
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

        # Consecutive declared hits per band, across visits -- not reset by a
        # visit to a *different* band, only by a miss on this one (D67). Capped
        # so one exceptionally persistent emitter cannot make the feature
        # unbounded, and divided down to match every other per-band block's [0, 1]
        # range.
        hit_streak = np.minimum(self._hit_streak, _STREAK_CAP) / _STREAK_CAP
        current_hit_streak = float(hit_streak[self._current_band])

        # D30, "v2" blocks -- same clamp-then-rescale recipe as `measured_dbm`
        # above, applied per band instead of to one global scalar.
        clamped_band = np.clip(self._band_measured_dbm, _DBM_CLAMP_MIN, _DBM_CLAMP_MAX)
        measured_dbm_band = (clamped_band - _DBM_CLAMP_MIN) / (_DBM_CLAMP_MAX - _DBM_CLAMP_MIN)
        pulse_width = np.clip(self._band_pulse_width, _PW_CLAMP_MIN, _PW_CLAMP_MAX) / _PW_CLAMP_MAX
        aoa_sin = (self._band_aoa_sin + 1.0) / 2.0
        aoa_cos = (self._band_aoa_cos + 1.0) / 2.0

        # D72: same log1p transform `reward_balance_improved` prices `C`
        # with (`_DENSITY_REF_PULSES = 64`), clipped into [0, 1] -- unlike the
        # reward's use, this has to stay inside the box, and raw `C` can
        # exceed the reference within a single episode.
        pulse_count = np.clip(
            np.log1p(self._band_pulse_count) / np.log1p(_DENSITY_REF_PULSES),
            0.0, 1.0,
        )

        # D75, the three in-context blocks. `prev_action` is all zeros before the
        # first step -- deliberately *not* a one-hot at band 0, which is what
        # `current_band` reads there and is a small untruth it inherited (it
        # claims a dwell that never happened). The zero vector sits off the
        # one-hot simplex, so no real action can produce it and the cold start is
        # unambiguous without a magic value; the same argument the AoA blocks
        # make for their (0.5, 0.5) centre decoding to a point off the unit
        # circle. Thereafter this block equals `current_band` exactly -- see
        # OBS_LAYOUTS["v3"] for why that redundancy is accepted rather than
        # removed.
        prev_action = np.zeros(N_BANDS, dtype=np.float64)
        if self._prev_action >= 0:
            prev_action[self._prev_action] = 1.0

        return {
            "hit_rate": hit_rate.astype(np.float32),
            "visit_density": visit_density.astype(np.float32),
            "staleness": staleness.astype(np.float32),
            "current_band": current_band.astype(np.float32),
            "clock": np.array([self.t / self.episode_slots], dtype=np.float32),
            "measured_dbm": np.array([measured_dbm], dtype=np.float32),
            "hit_streak": hit_streak.astype(np.float32),
            "current_hit_streak": np.array([current_hit_streak], dtype=np.float32),
            "measured_dbm_band": measured_dbm_band.astype(np.float32),
            "pulse_width": pulse_width.astype(np.float32),
            "aoa_sin": aoa_sin.astype(np.float32),
            "aoa_cos": aoa_cos.astype(np.float32),
            "pulse_count": pulse_count.astype(np.float32),
            "band_priority": self._band_priority.astype(np.float32),
            "prev_action": prev_action.astype(np.float32),
            "prev_reward": np.array(
                [_normalise_prev_reward(self._prev_reward_obs)], dtype=np.float32
            ),
            "prev_hit": np.array([float(self._prev_hit)], dtype=np.float32),
        }

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
        if self.t >= self.episode_slots:
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
        # D76: a bounded log for long missions. `None` (the default) keeps every
        # row, which every existing caller relies on -- `metrics.artefacts` reads
        # the whole episode back off it and checks its length.
        if self.log_window_slots is not None:
            overflow = len(self.log) + dwell.n_slots - self.log_window_slots
            if overflow > 0:
                del self.log[:overflow]
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
        easy to get wrong: an emitter never found counts at the **end of the 30 s
        segment it lives in**, not dropped from the average, because averaging
        over only the emitters you did find rewards not looking. On a single
        recording -- every episode this repository has ever scored -- the segment
        and the episode are the same 600 slots; they differ only on a stitched
        mission (D76), where charging a segment-0 miss the whole hour would make
        the figure incomparable with every other number here.

        The full comparison across schedulers, with distributions and repeated
        seeds, belongs to `metrics.py`. This is one episode's row of it.
        """
        n_e = len(self.detectable)
        delays = []
        for e, (on_e, _) in self.detectable.items():
            track = self.tracks.get(e)
            # An emitter never found is censored at the end of **its own 30 s
            # segment**, not at the end of the mission (D76). On a single
            # recording those are the same slot and this is bit-identical to the
            # `N_SLOTS` it replaced. On a stitched one they are not: an emitter
            # that only ever existed in segment 0 would otherwise be charged the
            # whole hour, and the figure would stop being comparable with every
            # 30 s number in this repository. A segment is the whole world an
            # emitter can be found in, so it is the honest censoring horizon.
            segment_end = ((on_e // N_SLOTS) + 1) * N_SLOTS
            delays.append((segment_end if track is None else track["first"]) - on_e)

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
            # Per second of *this* episode, which is `EPISODE_S` at the default
            # length and longer on a stitched mission (D76).
            "avg_intercept_rate_per_s": (
                len(self.tracks) / (self.episode_slots * SLOT_S)
            ),
            "total_reward": self.total_reward,
            "n_detectable": n_e,
            "n_intercepted": len(self.tracks),
            # Steps, not slots: an episode is always 600 slots but 300-600
            # decisions, depending how many wide bands were chosen (D31).
            "n_steps": self.n_steps,
        }
