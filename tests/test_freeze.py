"""The freeze list, as literals. `rfenv/constants.py` calls itself "the freeze list,
literally" -- until this file existed, it had no literal.

**Why this test is different from every other one.** Every other test in this suite
reads the constants symbolically (`K.GAMMA_DBM`, `K.SLOT_S`, ...). That is correct for
them -- they test behaviour, and behaviour should follow the constant. But it means the
whole suite would stay green if someone changed gamma, the slot clock or the band
geometry: every expectation would move with the value and nothing would notice. This
file is the one place that says what the values **are**, so that a change to the freeze
list is a deliberate act with a failing test attached to it.

**The freeze** (D42, 2026-09-04). The four validation gates of `EVALUATION.md` §6 ran
and passed, so `EVALUATION.md` §6's "on pass, **freeze** everything in
`rfenv/constants.py`" is now in force. Nothing here may move because an RL result came
out badly. If one of these does change, D25's rule applies: the environment is
re-validated from gate 1 and every baseline is re-run.

**What is not frozen**, and must not be added here: reward candidates and the
observation vector's optional extensions stay the RL lane's (D29, D30, D34), and the
per-episode draw -- which emitters, and the seed -- is free by construction.

**If this test fails**, do not update the numbers to match the code. Either revert the
change to `constants.py`, or -- if the change is intended -- record the decision, update
these literals in the same commit, and re-run the gates.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from rfenv import constants as K

# --------------------------------------------------------------------------- #
# Band geometry (D3)
# --------------------------------------------------------------------------- #

def test_band_geometry_is_turings_unchanged():
    """36 centres on 500 MHz spacing from 250 MHz, read from the files (D3).

    Checked against the recordings themselves on every load by
    `scenario.load_receiver`, which raises if a file disagrees. This pins our side of
    that comparison so both halves are nailed down.
    """
    assert K.N_BANDS == 36
    assert K.BAND_CENTRES_MHZ[0] == 250.0
    assert K.BAND_CENTRES_MHZ[-1] == 17750.0
    assert np.array_equal(K.BAND_CENTRES_MHZ, np.arange(250.0, 18000.0, 500.0))


def test_band_halfwidth_is_500_and_is_the_load_bearing_reconstruction():
    """±500 MHz, so adjacent bands **overlap by half**.

    This is the single most load-bearing reconstructed parameter in the project and
    the one with the weakest safety net, so it gets its own test and its own note:

    * The TSRD paper says the receiver sweeps *"in 500 MHz steps and 500 MHz
      bandwidth"* — a disjoint tiling. **We do not reproduce that**, deliberately: the
      files disagree with it and `CLAUDE.md`'s authority table says the files win.
      Measured, 99.9851% of 4,393,233 train scan pulses fall within ±500 MHz of the
      dwell centre active at their ToA; a disjoint ±250 tiling puts only 50.92% in
      band (D3).
    * **No validation gate can detect an error here.** Gates 1 and 2 both apply this
      constant to both sides of their comparison, so both pass unchanged with a wrong
      ±250 (verified by fault injection, D42). D3's measurement above is the sole
      evidence, which is exactly why the number is pinned here.
    """
    assert K.BAND_HALFWIDTH_MHZ == 500.0
    # Overlap by half is a consequence, not a separate choice: spacing 500, width 1000.
    spacing = float(np.diff(K.BAND_CENTRES_MHZ)[0])
    assert 2 * K.BAND_HALFWIDTH_MHZ == 2 * spacing


# --------------------------------------------------------------------------- #
# Time base (D16) and the native dwell schedule (D3)
# --------------------------------------------------------------------------- #

def test_slot_clock_and_episode_length():
    assert K.SLOT_S == 0.05
    assert K.EPISODE_S == 30.0
    assert K.N_SLOTS == 600


def test_native_dwell_schedule():
    """Seven 100 ms dwells, twenty-nine 50 ms, 2.15 s per sweep — Turing's own.

    The seven wide bands are why an episode is 600 slots but 300-600 steps (D35) and
    why reward is per slot rather than per dwell (D31). Their identity is not
    incidental: they are the densest bands in the dataset.
    """
    expected = [0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05,
                0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05,
                0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    assert np.allclose(K.DWELL_TIMES_S, expected)
    # SWEEP_S is `DWELL_TIMES_S.sum()`, so it carries float accumulation error and is
    # 2.1499999999999995, not 2.15. Pinned as it actually is rather than rounded: the
    # same accumulation is why `slots_for_dwell` rounds instead of truncating.
    assert K.SWEEP_S == pytest.approx(2.15, abs=1e-12)
    assert K.SWEEP_S == float(np.asarray(expected).sum())
    assert [int(v) for v in K.DWELL_SLOTS] == [2 if t > 0.05 else 1 for t in expected]
    assert [i for i, v in enumerate(K.DWELL_SLOTS) if v == 2] == [0, 1, 6, 7, 17, 18, 19]


def test_the_reference_sweep_covers_the_episode_as_expected():
    """502 dwells over 30 s, 13-14 per band. Derived, but pinned: gate 2's per-band
    tolerance was sized against these counts (D39)."""
    sched = K.dwell_schedule()
    assert len(sched) == 502
    counts = np.bincount(sched[:, 2].astype(int), minlength=K.N_BANDS)
    assert counts.min() == 13 and counts.max() == 14


# --------------------------------------------------------------------------- #
# Receiver detection model (D4, D23, D33)
# --------------------------------------------------------------------------- #

def test_noise_floor_sigma_and_gamma():
    """`N₀` is anchored to the receiver's own fields; `σ` is chosen; `γ = N₀ + 3σ`.

    `γ` is **not** calibrated against the recorded dwell rate — that procedure was
    confounded and is retracted (D23). It is a swept receiver parameter whose default
    operating point falls out of the noise model, which is why Pfa is analytic
    (`1 − Φ(3)`) rather than fitted.
    """
    assert K.NOISE_FLOOR_DBM == -120.0
    assert K.NOISE_SIGMA_DB == 3.0
    assert K.GAMMA_DBM == -111.0
    assert K.GAMMA_DBM == K.NOISE_FLOOR_DBM + 3.0 * K.NOISE_SIGMA_DB


def test_pd_population_is_frozen_with_gamma():
    """D33. On the freeze list because Pd depends on it entirely: 0.851 here against
    0.819 over stare-replay cells and 0.837 over scan-replay cells, at one γ. The
    withdrawn 0.822 matched none of them, which is what a population left unstated
    costs."""
    assert K.PD_POPULATION == "reference_sweep"


# --------------------------------------------------------------------------- #
# Data layout and the held-out guard (D8)
# --------------------------------------------------------------------------- #

def test_split_names_are_frozen():
    """The held-out guard keys off these. A typo here silently disarms D8."""
    assert K.TRAIN_SPLIT == "train"
    assert K.HELDOUT_SPLIT == "test"
    assert K.SOURCES == ("scan", "stare")


# --------------------------------------------------------------------------- #
# One digest over the whole list
# --------------------------------------------------------------------------- #

FREEZE_DIGEST = "e309d85286ecedb8e1423c7243d1d99d4951d22cb044b210fb205b29c20b6c74"


def test_the_freeze_list_digest_is_unchanged():
    """A single tripwire over everything above, so a value added or altered anywhere
    on the list fails even if no named test covers it yet.

    The per-field tests exist to say *what broke*; this exists so nothing can break
    silently. Update both together, in the commit that records the decision.
    """
    payload = json.dumps(
        {
            "centres": [float(x) for x in K.BAND_CENTRES_MHZ],
            "halfwidth": K.BAND_HALFWIDTH_MHZ,
            "dwell_times": [float(x) for x in K.DWELL_TIMES_S],
            "dwell_slots": [int(x) for x in K.DWELL_SLOTS],
            "slot_s": K.SLOT_S,
            "episode_s": K.EPISODE_S,
            "n_slots": K.N_SLOTS,
            "n0": K.NOISE_FLOOR_DBM,
            "sigma": K.NOISE_SIGMA_DB,
            "gamma": K.GAMMA_DBM,
            "pd_pop": K.PD_POPULATION,
        },
        sort_keys=True,
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == FREEZE_DIGEST, (
        "The freeze list changed. Do not update this digest to make the test pass: "
        "revert the change, or record the decision, re-run the gates (D25) and update "
        "the literals in this file in the same commit."
    )
