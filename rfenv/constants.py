"""The freeze list, literally.

Everything in this module is frozen before any scheduler number is quoted
(`docs/project/EVALUATION.md` §6). Nothing here may be changed because an RL result
came out badly: if one of these moves, the environment is re-validated from gate 1
and every baseline is re-run.

The receiver geometry is Turing's, unchanged (D3). It is *asserted* against every
HDF5 file opened rather than hardcoded and hoped for — `scenario.load_receiver`
reads the real fields and checks them against the values below.
"""

from __future__ import annotations

import numpy as np

# --- Band geometry (D3) -------------------------------------------------------
# 36 dwell centres on 500 MHz spacing, read from `metadata/receiver/dwell_centres_mhz`.
# Verified identical across all 47 train scan configs.
BAND_CENTRES_MHZ = np.arange(250.0, 18000.0, 500.0)
N_BANDS = len(BAND_CENTRES_MHZ)

# Half-width, not full width. The dataset paper says "500 MHz steps and 500 MHz
# bandwidth", implying a disjoint tiling; the files disagree and the files win
# (CLAUDE.md authority table). Measured: 99.985% of 4,393,233 train scan pulses fall
# within +-500 MHz of the dwell centre active at their ToA, with 4 pulses in
# [500, 520) and a hard edge at exactly 500. A disjoint +-250 tiling would put only
# 50.92% in band. Adjacent bands therefore overlap by half.
BAND_HALFWIDTH_MHZ = 500.0

# Native Turing dwell schedule: seven 100 ms dwells, twenty-nine 50 ms, 2.15 s/sweep.
# Read from `metadata/receiver/dwell_times_s`. A band keeps its native dwell length
# when our scheduler selects it (D3, D16).
DWELL_TIMES_S = np.array(
    [0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05,
     0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05,
     0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
)
SWEEP_S = float(DWELL_TIMES_S.sum())  # 2.15

# --- Time base (D16) ----------------------------------------------------------
# 50 ms slots, Turing's minimum dwell. The seven 100 ms dwells span two slots.
SLOT_S = 0.05
EPISODE_S = 30.0  # `metadata.attrs["collection_time_s"]`, 30.0 for all 47 train pairs
N_SLOTS = int(round(EPISODE_S / SLOT_S))  # 600

# Each band's native dwell as a whole number of slots: 2 for bands 0, 1, 6, 7, 17,
# 18, 19 and 1 for the other twenty-nine. This is what an action costs (D3, D16) --
# the agent picks a band, never a dwell length -- and since retuning is free (D31,
# measured under 1 us) airtime is the only currency in the problem.
DWELL_SLOTS = np.round(DWELL_TIMES_S / SLOT_S).astype(np.int64)

# --- Receiver detection model (D4, D23, D26) ----------------------------------
# The truth grid holds a continuous level S. Empty cells sit at the noise floor;
# when the receiver looks it measures S + n and declares a hit iff that beats gamma.
#
# NOISE_FLOOR_DBM is anchored to the receiver's own fields: `sensitivity_dbm` (-110)
# minus `gain_db` (10). It is deliberately NOT the dataset paper's -100 dB ambient
# noise -- that figure describes the *generator's* probabilistic detector, which
# admitted weak pulses and then blurred their amplitudes (TSRD §II), which is why
# 13.25% of recorded scan pulses sit below it. The recordings we build S from are
# already post-detection; -100 dB is not a floor our threshold must clear.
NOISE_FLOOR_DBM = -120.0

# Chosen, not measured. This is our receiver's design parameter.
NOISE_SIGMA_DB = 3.0

# Default operating point: 3 sigma above the floor -> Pfa = 1.35e-3 exactly
# (it is 1 - Phi(3)), sensitivity (the level at which Pd = 0.9) = gamma + 1.2816*sigma
# = -107.2 dB. Both re-run 2026-09-03. The full sweep is the deliverable (D15).
#
GAMMA_DBM = NOISE_FLOOR_DBM + 3.0 * NOISE_SIGMA_DB  # -111.0

# Which cells Pd is averaged over (D33). Frozen with gamma, N0 and sigma, because
# the figure depends on it entirely: measured 2026-09-03, Pd at gamma = -111 comes
# out 0.819 over stare-replay cells, 0.837 over scan-replay cells and 0.851 over
# these. The previously recorded 0.822 matched none of them and is withdrawn.
#
# "reference_sweep" = the occupied cells Turing's own dwell_schedule() looks at.
# It is per-look, which is what EVALUATION.md §3 means by a detection probability,
# and it is scheduler-independent, which D21 requires -- the schedule is Turing's
# and never varies, so no scheduler can move the population by looking elsewhere.
# The rejected alternative was all occupied cells (0.837): a larger sample and
# schedule-free, but it characterises the detector over cells no receiver visits,
# which is not what "per look" means.
PD_POPULATION = "reference_sweep"

# --- Data layout --------------------------------------------------------------
DATA_ROOT = "data/turing"
CACHE_ROOT = "data/cache/contrib"
SOURCES = ("scan", "stare")

# Held-out: 45 scan/stare pairs under data/turing/*/test_* (D8). Touched once, at the
# end, after the system is frozen. `scenario.py` refuses these paths unless an
# explicit allow_heldout=True is passed, and logs every such call.
HELDOUT_SPLIT = "test"
TRAIN_SPLIT = "train"


def dwell_schedule(episode_s: float = EPISODE_S) -> np.ndarray:
    """Turing's own sweep as absolute (start_s, end_s, band) rows over one episode.

    This is the reference schedule: the receiver cycles bands 0..35 in order, each
    for its native dwell length, repeating every SWEEP_S. Over 30 s that is 13.953
    sweeps -> 502 dwells, the last one truncated by the episode end.

    Used by the Turing-sweep baseline and by the validation replays.
    """
    edges = np.concatenate([[0.0], np.cumsum(DWELL_TIMES_S)])
    rows = []
    sweep = 0
    while sweep * SWEEP_S < episode_s:
        base = sweep * SWEEP_S
        for band in range(N_BANDS):
            start = base + edges[band]
            end = min(start + DWELL_TIMES_S[band], episode_s)
            # Float accumulation can leave a sliver of a dwell at the episode
            # boundary; a zero-length look is not a dwell.
            if end - start <= 1e-9:
                break
            rows.append((start, end, band))
        sweep += 1
    return np.array(rows, dtype=np.float64)


def slots_for_dwell(start_s: float, end_s: float) -> tuple[int, int]:
    """Half-open slot range [s0, s1) a dwell covers.

    Every dwell boundary lands on an exact slot boundary -- dwells are 50 or 100 ms
    and a sweep is 2.15 s, all whole multiples of the 50 ms slot -- so this rounds
    rather than truncates. Truncating is an off-by-one waiting to happen: float
    accumulation makes 6.45 / 0.05 come out as 128.99999999999999, which would
    silently shift a dwell one slot early and hand it a cell it never looked at.
    """
    s0 = int(round(start_s / SLOT_S))
    s1 = int(round(end_s / SLOT_S))
    return s0, max(s1, s0 + 1)


def bands_covering(freq_mhz: np.ndarray) -> list[np.ndarray]:
    """For each band, the boolean mask of pulses its +-500 MHz window contains.

    A pulse belongs to *every* band whose window covers it, not just the one the
    recording's receiver happened to be tuned to. That is what makes the grid
    counterfactual (D1): if the scheduler had tuned band b at that slot, it would
    have heard the pulse, so cell (b, slot) must say so.

    Because centres are 500 MHz apart and the half-width is 500 MHz, a pulse falls in
    2 bands generically and 1 near the spectrum edges. Measured on config_921:
    2 bands for 95.2% of pulses, 1 for 4.8%, never 3.
    """
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    return [
        np.abs(freq_mhz - centre) <= BAND_HALFWIDTH_MHZ for centre in BAND_CENTRES_MHZ
    ]
