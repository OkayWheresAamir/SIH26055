"""Tests for `rfenv.validate` -- the gates, and the criteria they are judged against.

The point of most of these is not that the gates *pass*. It is that the criteria
cannot drift: `GATES` is asserted against the numbers written into
`docs/project/DECISIONS.md` D39, so a threshold cannot be edited in one place only.
That is the whole structural fix D39 asked for -- a criterion that lives in two
places and is checked against itself cannot be quietly retuned after a bad run.

Gate 3 runs end to end here because it needs no data at all: the emitter is
synthesised, so the test is fast and works on a machine without `data/turing`.
Gates 1, 2 and 4 need the recordings, so they are exercised on a couple of configs
and skipped when the data is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfenv.constants import (
    DWELL_TIMES_S,
    EPISODE_S,
    GAMMA_DBM,
    N_SLOTS,
    NOISE_SIGMA_DB,
    SLOT_S,
    SWEEP_S,
)
from rfenv import validate as V

DATA = Path("data/turing")
needs_data = pytest.mark.skipif(not DATA.exists(), reason="Turing recordings not present")


# --------------------------------------------------------------------------- #
# The criteria themselves
# --------------------------------------------------------------------------- #

def test_the_criteria_match_what_D39_records():
    """`GATES` is the decision, in code. If it drifts from D39, one of them is wrong.

    D39 is the decision that gates 2, 3 and 4 needed pass criteria fixed *before* the
    first run. These are those criteria. A threshold that can be changed in the code
    without changing the record is a threshold that can be tuned after seeing the
    measurement, which is exactly what D39 forbids.
    """
    assert V.GATES["2"]["aggregate_pp"] == 0.5
    assert V.GATES["2"]["per_band_pp"] == 1.0
    assert V.GATES["2"]["direction"] == "replayed <= recorded"
    assert V.GATES["3"]["p12_abs_tol"] == 0.01
    assert V.GATES["4"]["configs"] == ("config_81", "config_921")
    assert V.GATES["4"]["assertions"] == 12


def test_gate1_is_measured_and_the_rest_are_gated():
    """D37 fixed gate 1's convention and left its threshold undecided.

    So gate 1 reports a number and cannot fail. Inventing a threshold for it would be
    the failure this module exists to prevent, and this test is what stops someone
    adding one without reopening D37.
    """
    assert V.GATES["1"]["gated"] is False
    assert all(V.GATES[g]["gated"] for g in ("2", "3", "4"))


# --------------------------------------------------------------------------- #
# Koksal's closed forms, as transcribed
# --------------------------------------------------------------------------- #

def test_koksal_table_6_1_picks_the_right_case():
    """Table 6-1 is two rows by two columns; each must be reachable and in range."""
    # tau1 <= T2 - T1, tau2 <= T1  -- the gate-3 case
    assert 0.0 < V._koksal_p12_first(0.1, 2.15, 0.5, 3.0) < 1.0
    # tau1 <= T2 - T1, T1 <= tau2
    assert 0.0 < V._koksal_p12_first(0.1, 1.0, 2.0, 3.0) < 1.0
    # T2 - T1 <= tau1, tau2 <= T1
    assert 0.0 < V._koksal_p12_first(0.9, 2.0, 1.0, 2.5) < 1.0
    # T2 - T1 <= tau1, T1 <= tau2
    assert 0.0 < V._koksal_p12_first(0.9, 1.0, 1.5, 1.5) < 1.0


def test_koksal_requires_train_1_to_be_the_shorter_period():
    """Table 6-1 is stated for `T1 <= T2`; applying it the other way round is silent
    nonsense, so it raises instead."""
    with pytest.raises(ValueError):
        V._koksal_p12_first(0.1, 3.0, 0.5, 2.15)


def test_the_gate3_case_reproduces_the_transcribed_closed_form():
    """The value read off body p.72 for the pre-registered case.

    `P12(T1) = (1/T2)[tau1 + tau2(1 - tau2/(2 T1))]` with the gate-3 parameters
    (tau_rcv 0.1, T_rcv 2.15, tau_emit 0.5, T_emit 3.0) is 0.180620. Hardcoded
    because it is a transcription of a published table, not a computed result: if
    the code stops agreeing with it, the transcription has been edited.
    """
    assert V._koksal_p12_first(0.1, SWEEP_S, 0.5, 3.0) == pytest.approx(0.1806201550, abs=1e-9)


def test_eq_3_8_is_the_bound_on_the_page():
    """`intercept time >= T_emit T_rcv / (tau_emit + tau_rcv)`, body p.18."""
    assert V._koksal_eq38(3.0, 2.15, 0.5, 0.1) == pytest.approx(3.0 * 2.15 / 0.6)


def test_the_discrete_reference_imports_nothing_from_the_environment():
    """Gate 3's reference must be independent of what it judges.

    `_koksal_discrete_p12` models the same two pulse trains in pure numpy. If it ever
    reached into `TruthGrid` or `Receiver`, gate 3 would be comparing the environment
    against itself and its tolerance would mean nothing.
    """
    import inspect

    src = inspect.getsource(V._koksal_discrete_p12)
    for forbidden in ("TruthGrid", "Receiver", "ScanEnv", "Scenario"):
        assert forbidden not in src


def test_the_discrete_reference_matches_the_hand_count():
    """11 phases of 60 put the emitter under band 0's opening dwell -> 11/60.

    Counted by hand: the emitter is on for 10 of every 60 slots; band 0's dwell in
    the first sweep is slots 0 and 1; the phases covering either are {0, 1} and
    {51..59}, which is 11. This is the quantised answer the environment must
    reproduce, so it is worth pinning independently of the code that computes it.
    """
    assert V._koksal_discrete_p12(0.1, SWEEP_S, 0.5, 3.0) == pytest.approx(11 / 60)


def test_the_clock_bias_is_small_and_reported_not_hidden():
    """The 50 ms clock costs ~0.0027 against continuous theory, well inside the
    0.01 the gate allows for the environment itself."""
    bias = abs(V._koksal_discrete_p12(0.1, SWEEP_S, 0.5, 3.0)
               - V._koksal_p12_first(0.1, SWEEP_S, 0.5, 3.0))
    assert bias < V.GATES["3"]["p12_abs_tol"]


# --------------------------------------------------------------------------- #
# Gate 3 end to end -- no data needed
# --------------------------------------------------------------------------- #

def test_the_synthetic_emitter_is_periodic_in_one_band_above_threshold():
    """Gate 3's controlled case, checked against its own stated design.

    One band (unambiguous alpha), 10 slots on per 60 (tau/T = 0.5/3.0 s), and a level
    5 sigma above gamma so the detector contributes nothing to the timing check.
    """
    c = V._g3_contribution(phase_slots=0)
    assert set(c.cells[:, 0].tolist()) == {V._G3_BAND}
    slots = np.sort(c.cells[:, 1])
    assert np.all(c.peak_dbm >= GAMMA_DBM + 5.0 * NOISE_SIGMA_DB)
    # 600 slots / 60-slot period x 10 on-slots
    assert slots.size == N_SLOTS // int(round(3.0 / SLOT_S)) * int(round(0.5 / SLOT_S))


def test_gate3_passes_and_reports_the_P12_T_divergence():
    """The gate itself: 3a within tolerance, 3b's bound holds, 3c recorded.

    3c is the one that matters for the record. `P12(T) = 1 - [1-P12(T1)]^(T/T1)`
    assumes successive receiver periods are independent Bernoulli trials; for two
    strictly periodic trains the relative phase drifts deterministically, so every
    phase intercepts well inside 30 s while Koksal predicts ~0.94. Gating on it would
    fail a correct environment (D40).
    """
    res = V.gate3(seed=0)
    assert res.passed is True
    assert res.measured["p12_env_vs_reference"] <= V.GATES["3"]["p12_abs_tol"]
    assert res.measured["max_first_intercept_s"] >= res.measured["eq38_bound_s"]
    # The divergence is real and is reported rather than gated.
    assert res.measured["p12_over_T_actual"] > res.measured["p12_over_T_koksal"]


def test_gate3_is_framed_as_an_environment_check_not_a_scheduler_claim():
    """D39's framing risk, kept where it cannot be lost.

    Koksal assumes prior emitter knowledge; D20 refuses it. The gate is still valid --
    the pre-knowledge is the analyst's, never the agent's -- but "our system matches
    Koksal" is a claim the architecture does not support, so the caveat travels with
    the result rather than living in a chat log.
    """
    notes = " ".join(V.gate3(seed=0).notes).lower()
    assert "environment" in notes and "d20" in notes


# --------------------------------------------------------------------------- #
# Reporting contract
# --------------------------------------------------------------------------- #

def test_every_gate_prints_its_criterion_beside_its_measurement():
    """A number without the threshold it was judged against is a claim, not a result."""
    res = V.gate3(seed=0)
    assert res.criterion and res.measured
    line = V._line(res)
    assert "PASS" in line and str(V.GATES["3"]["p12_abs_tol"]) in line


def test_a_failing_gated_check_exits_non_zero(tmp_path, monkeypatch):
    """`validate.py` is meant to be runnable in CI, so failure must be an exit code.

    Forced by tightening gate 3's tolerance to something no discretised environment
    can meet, rather than by breaking the environment.
    """
    monkeypatch.setitem(V.GATES["3"], "p12_abs_tol", -1.0)
    assert V.main(["--gate", "3", "--out", str(tmp_path)]) == 1
    assert json.loads((tmp_path / "gate3.json").read_text())["status"] == "FAIL"


def test_summary_stamps_the_operating_point(tmp_path):
    """EVALUATION.md §7: state gamma and Pfa alongside any table.

    The ROC population is on the freeze list (D33), so it is stamped too -- a Pd with
    no population attached is the withdrawn 0.822 all over again.
    """
    pytest.importorskip("h5py")
    if not DATA.exists():
        pytest.skip("Turing recordings not present")
    V.main(["--gate", "3", "--out", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    point = summary["operating_point"]
    assert point["gamma_dbm"] == GAMMA_DBM
    assert point["sigma_db"] == NOISE_SIGMA_DB
    assert point["population"] and "pfa" in point and "pd" in point
    assert summary["split"] == "train"


# --------------------------------------------------------------------------- #
# The data-backed gates
# --------------------------------------------------------------------------- #

@needs_data
def test_gate1_and_gate2_share_one_observation_side():
    """Both gates ask the recording the same question, so they count the same way.

    That is D37's fourth reason for the per-dwell convention: gate 1 and gate 2 become
    readable side by side. If they ever computed the recorded side differently, the
    two numbers would stop being comparable and nobody would notice.
    """
    from rfenv.constants import dwell_schedule

    schedule = dwell_schedule()
    a = V._recorded_per_dwell("config_81", schedule)
    b = V._recorded_per_dwell("config_81", schedule)
    assert np.array_equal(a, b)
    assert a.shape == (len(schedule),)


@needs_data
def test_gate2_is_a_self_consistency_test_and_agrees_exactly():
    """Grid built from the scan recording, replayed on the schedule that produced it.

    Both sides count the same dwells and every dwell boundary is an exact slot
    multiple, so the mechanism predicts near-exact agreement -- which is why the
    tolerance is 0.5 pp rather than a standard error.
    """
    res = V.gate2(["config_81", "config_921"])
    assert res.passed is True
    assert abs(res.measured["aggregate_delta_pp"]) <= V.GATES["2"]["aggregate_pp"]
    assert res.measured["max_band_delta_pp"] <= V.GATES["2"]["per_band_pp"]


@needs_data
def test_gate1_predicts_band0_as_empty_and_says_so():
    """The standing limitation, asserted rather than trusted.

    Stare's `freq_range_mhz` starts at 500 MHz, so band 0 (250 MHz) cannot be
    predicted at all (D10, D24). It is stated and plotted, never patched -- and this
    test fails if someone ever "fixes" it by quietly widening the band.
    """
    res = V.gate1(["config_81", "config_921"], seed=0)
    assert res.measured["band0_predicted"] == 0.0
    assert res.measured["band0_recorded"] > 0.0
    assert any("band 0" in n.lower() for n in res.notes)


@needs_data
def test_gate4_asserts_every_structural_invariant():
    """All twelve, and they are structural: they hold or the code is wrong."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        res = V.gate4(Path(tmp), seed=0)
    assert res.passed is True
    assert res.measured["n_passed"] == res.measured["n_assertions"] == 12
    s = res.measured["summaries"]
    assert s["config_921"]["n_detectable"] > s["config_81"]["n_detectable"]
    for cfg in s.values():
        assert 0.0 <= cfg["interception_ratio"] <= 1.0
        assert 0.0 <= cfg["emitter_coverage"] <= 1.0
        assert 0.0 <= cfg["censored_mean_intercept_time_s"] <= EPISODE_S


@needs_data
def test_validate_never_touches_the_held_out_split():
    """D8: the 45 test pairs are touched once, at the end, after the freeze.

    `validate.py` passes `allow_heldout` nowhere, so `scenario._guard_heldout` is the
    enforcement rather than anyone's discipline. Checked over the parsed syntax tree
    rather than the raw text, because the module's own prose explains the rule and a
    substring search would match the explanation instead of a bypass.
    """
    import ast

    tree = ast.parse(Path("rfenv/validate.py").read_text())
    passed = [
        kw.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg in ("allow_heldout", "split", "reason")
    ]
    assert passed == [], f"validate.py passes {passed} into a loader; train split only (D8)"
