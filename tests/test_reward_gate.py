"""The reward screen that runs in front of D47.

The criteria live in `rfenv/reward_gate.py` and are asserted here as literals,
the same way `tests/test_validate.py` pins `validate.py::GATES` against D39. The
reason is identical and is the whole point of both files: a threshold edited
after the measurement is visible is not a threshold. If a candidate we want fails
the screen, the fix is the candidate.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfenv import baselines as B
from rfenv import reward_gate as G
from rfenv.env import REWARDS
from rfenv.scenario import EmitterPool


def test_the_criteria_are_what_was_written_down():
    """Pinned as literals so a later edit is a deliberate, reviewed change."""
    assert G.MIN_SEPARATION_SIGMA == 2.0
    assert G.MIN_SEEDS_BAR_ABOVE_FLOOR == 7
    assert G.MIN_EXPLOIT_MARGIN_SIGMA == 1.0
    assert G.N_SEEDS == 8
    assert G.SCREEN_RUNGS == ("round_robin", "recency", "camper", "apfeld_active_rfs")
    assert (G.FLOOR, G.BAR, G.EXPLOIT) == ("round_robin", "recency", "camper")


def test_every_screened_rung_is_a_real_registered_rung():
    """The bug that inverted D56, as a test.

    D56 scored a hand-written `step % N_BANDS` sweep and called it `round_robin`.
    That is not rung 2 -- rung 2 is `EQUAL_AIRTIME_CYCLE` (D43), equal *airtime*
    per band rather than one dwell per band -- and the substitution moved the
    measured separation from +59.0 to +2.0, inverting the conclusion. Every rung
    this screen scores must come from the ladder.
    """
    for key in G.SCREEN_RUNGS:
        assert key in B.BY_KEY, f"{key} is not a registered rung"
    assert not B.BY_KEY["round_robin"].deployable is False  # it is a real scheduler


def test_the_verdict_is_the_conjunction_of_the_three_checks():
    """Constructed, not measured: the decision logic, on numbers chosen to sit
    either side of each threshold. Keeps the rule testable without an episode."""
    def make(sep, seeds, camper):
        return G.Screen("x", {}, sep, seeds, camper)

    assert make(3.0, 8, 5.0).passed
    assert not make(1.9, 8, 5.0).passed            # separation too small
    assert not make(3.0, 6, 5.0).passed            # inconsistent across seeds
    assert not make(3.0, 8, 0.9).passed            # camper not far enough below
    assert make(2.0, 7, 1.0).passed                # exactly on every threshold

    v = make(1.0, 3, -2.0).verdict
    assert v.startswith("FAIL") and "sigma" in v and "seeds" in v


def test_the_screen_separates_a_known_good_reward_from_a_known_bad_one():
    """One end-to-end check on two candidates whose behaviour is settled.

    `greedy` (the exploit corner, D57) has no explore term and no camping cost
    by design, so it cannot help but rank the camper top -- D14's tension and the
    reason rung 4 is in the ladder at all. `hit_z`/`hit_y` were this repository's
    original known-bad example and shared the same failure; they are gone from
    `REWARDS` after failing this exact screen (D62), which is itself evidence the
    screen works -- `greedy` is the one still registered to check it against.
    `reward_balance` prices airtime concentration and does not share the
    failure. If the screen ever stops distinguishing those two it has stopped
    working.

    Two seeds rather than eight: this asserts the *direction*, and the criteria
    themselves are pinned above without paying for episodes.
    """
    pool = EmitterPool.from_train()
    seeds = (0, 1)

    bad = G.screen("greedy", seeds=seeds, pool=pool)
    good = G.screen("reward_balance", seeds=seeds, pool=pool)

    # The camper is the discriminator: above both sweeps under greedy, far below
    # them under reward_balance.
    assert bad.scores["camper"].mean() > bad.scores["round_robin"].mean()
    assert good.scores["camper"].mean() < good.scores["round_robin"].mean()
    assert good.exploit_margin_sigma > bad.exploit_margin_sigma


def test_screen_only_accepts_registered_candidates():
    """`ScanEnv.__init__` is the validator, and the screen must not route around
    it -- an unregistered name has to fail loudly rather than score as zero."""
    with pytest.raises(ValueError, match="reward must be one of"):
        G.screen("not_a_reward", seeds=(0,))


def test_every_registered_candidate_can_be_screened():
    """No candidate is un-screenable -- a new one cannot dodge the gate by
    having a signature the screen cannot call."""
    assert set(REWARDS)  # non-empty
    for name in REWARDS:
        assert callable(REWARDS[name])


def test_no_module_outside_baselines_hand_rolls_a_band_cycle():
    """D56, generalised: nothing outside `rfenv/baselines/` may construct a
    scheduling policy by hand -- every measurement goes through `baselines.make()`.

    D56 scored a hand-written `step % N_BANDS` sweep and called it `round_robin`;
    it was not rung 2, and the substitution inverted a conclusion (see
    `test_every_screened_rung_is_a_real_registered_rung` above). This is the same
    check made structural: parse every top-level `rfenv/*.py` module -- deliberately
    not `rfenv/baselines/` or `rfenv/rl/`, where constructing a policy or training
    one is the point -- and fail if any of them contains `... % N_BANDS` as actual
    code (an AST walk, not a text grep, so mentioning the pattern in a docstring,
    as this file and `reward_gate.py` both do, does not trip it).
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "rfenv"
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)):
                continue
            operand = node.right
            name = (operand.id if isinstance(operand, ast.Name) else
                     operand.attr if isinstance(operand, ast.Attribute) else None)
            if name == "N_BANDS":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"hand-rolled band-cycle ('... % N_BANDS') outside rfenv/baselines/: "
        f"{offenders} -- construct the policy through baselines.make() instead")
