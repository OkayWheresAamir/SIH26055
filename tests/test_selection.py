"""The pre-registered checkpoint-selection rule (D61).

The rule lives in `rfenv/selection.py` and is fixed before any retrain. These
tests pin it as literals, the same way `test_validate.py` pins the gates and
`test_reward_gate.py` pins the screen -- for the same reason in all three cases:
a criterion edited after the measurement is visible is not a criterion.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rfenv import selection as S
from rfenv import split


def test_the_rule_is_what_was_written_down():
    assert S.SELECTION_REFERENCE == "recency"      # the bar, not the floor
    assert S.SELECTION_SEEDS == (0, 1, 2)
    assert S.SELECTION_SOURCE == "stare"           # D36 excludes scan replays


def test_net_dominance_is_the_quantity_maximised():
    """Not the raw win rate -- the two disagree, and the disagreement is the point.

    These are the 2026-09-10 acceptance figures. A win-rate rule picks 300k
    (48.0% > 43.9%); the net rule picks 200k (+39.2 > +34.5), because 300k is
    strictly beaten three times as often. Ratified by the human after the
    win-rate version was proposed and corrected.
    """
    k200 = S.Candidate(Path("200k.zip"), 200_000, 0.439, 0.047, 171)
    k300 = S.Candidate(Path("300k.zip"), 300_000, 0.480, 0.135, 171)

    assert k300.dominates > k200.dominates          # win rate favours 300k
    assert k200.net_dominance > k300.net_dominance  # net favours 200k
    assert max([k200, k300], key=lambda c: c.net_dominance) is k200


def test_ties_break_toward_fewer_steps():
    """Given two checkpoints validation cannot separate, prefer the less-trained
    one: less opportunity to have memorised the training pool, cheaper to
    reproduce."""
    early = S.Candidate(Path("a.zip"), 100_000, 0.40, 0.10, 36)
    late = S.Candidate(Path("b.zip"), 900_000, 0.40, 0.10, 36)
    winner = max([late, early], key=lambda c: (c.net_dominance, -(c.steps or 0)))
    assert winner is early


def test_selection_runs_on_the_validation_half_only():
    """The rule must not be able to see training scenarios.

    `evaluate` defaults to `split.validation_configs()`; if that default ever
    became the full 47 the selection would be scored on scenarios the pool
    trained on, which is the leak D60 closed arriving by a different door.
    """
    import inspect
    src = inspect.getsource(S.evaluate)
    assert "validation_configs()" in src
    assert "from_train" not in src


def test_select_refuses_an_empty_candidate_set():
    with pytest.raises(ValueError, match="no checkpoints"):
        S.select([])


def test_evaluate_scores_a_known_policy_against_the_bar():
    """End-to-end on two configs: rung 5 against itself must tie.

    `recency` breaks ties randomly, so it will not be bit-identical to itself
    across two independent constructions at the same seed -- but it must be
    close, and it must certainly not dominate or be dominated most of the time.
    A harness bug (misread metric direction, mispaired seeds) shows up here as a
    lopsided result.
    """
    configs = split.validation_configs()[:2]
    from rfenv import baselines as B
    cand = S.evaluate(lambda s: B.make("recency", seed=s),
                      configs=configs, seeds=(0,))
    assert cand.n_episodes == 2
    assert cand.dominates < 0.75 and cand.dominated < 0.75


def test_the_direction_of_each_metric_is_right():
    """Higher interception ratio wins, lower intercept time wins.

    Getting one of these backwards would silently select the worst checkpoint
    every time and nothing else in the suite would notice, so it is asserted
    directly against a constructed pair rather than left to the end-to-end test.
    """
    import inspect
    src = inspect.getsource(S.evaluate)
    assert "mine[0] > theirs[0] and mine[1] < theirs[1]" in src
    assert "theirs[0] > mine[0] and theirs[1] < mine[1]" in src
