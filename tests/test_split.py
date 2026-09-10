"""The development split (D60), and the leak it closes.

The rule lives in `rfenv/split.py` and was written before anyone looked at which
configs landed where. These tests exist so that ordering survives: they pin the
rule's parameters as literals and assert the properties that make the split worth
having, so a later edit is a deliberate reviewed change rather than a drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfenv import split
from rfenv.rl.common import make_train_env
from rfenv.scenario import TRAIN_SPLIT, EmitterPool, list_configs


@pytest.fixture(scope="module")
def halves():
    return split.training_configs(), split.validation_configs()


def test_the_rule_is_what_was_written_down():
    """Pinned as literals, like `validate.py::GATES` and `reward_gate`'s criteria.

    Changing these re-draws the split, which silently invalidates every
    validation score ever recorded against it -- so it has to be loud.
    """
    assert split.VALIDATION_EVERY == 4
    assert split.VALIDATION_OFFSET == 1


def test_the_halves_partition_the_development_set(halves):
    train, val = halves
    assert len(val) == 12 and len(train) == 35
    assert not set(train) & set(val), "a config cannot be in both halves"
    assert set(train) | set(val) == set(list_configs("scan", TRAIN_SPLIT))


def test_the_pools_share_no_emitters(halves):
    """**The test that actually closes the leak.**

    Splitting the config list is not enough: `EmitterPool` is assembled *from*
    the configs, so a scenario sampled from a pool built over all 47 can contain
    an emitter belonging to a config that was supposed to be held back. Disjoint
    config lists with overlapping emitter pools would look like a split and
    behave like none.
    """
    train_pool, val_pool = split.training_pool(), split.validation_pool()
    shared = set(train_pool.by_emitter) & set(val_pool.by_emitter)
    assert not shared, f"{len(shared)} emitters appear in both pools"
    assert len(train_pool) + len(val_pool) == len(EmitterPool.from_train())


def test_both_halves_span_the_difficulty_range(halves):
    """Why the rule samples systematically along emitter count rather than at random.

    Scenario difficulty spans 2 to 99 emitters and dominates every §4 metric, so
    a uniform draw can hand validation a systematically easier or harder set
    without anyone noticing. Taking every 4th along the sorted order makes the
    coverage structural. Asserted as a loose band, not as the measured means --
    the point is that neither half is lopsided, not that they match to a decimal.
    """
    train, val = halves
    counts = dict(zip(list_configs("scan", TRAIN_SPLIT),
                      EmitterPool.from_train().emitter_counts.tolist()))
    ct = np.array([counts[c] for c in train])
    cv = np.array([counts[c] for c in val])

    assert cv.min() <= ct.min() + 5 and cv.max() >= ct.max() - 10
    assert abs(ct.mean() - cv.mean()) < 0.25 * ct.mean()


def test_training_does_not_sample_the_evaluation_pool():
    """`make_train_env` must not default to all 47 configs again.

    This is the regression that matters: the default was `EmitterPool.from_train()`,
    which is the same population `compare.py` draws its sampled evaluation
    scenarios from. Reverting it would re-open the leak while every other test
    stayed green.
    """
    env = make_train_env(reward="hit_z")
    assert len(env._pool) == len(split.training_pool())
    assert len(env._pool) < len(EmitterPool.from_train())


def test_from_configs_refuses_anything_outside_the_development_set():
    """D8's held-out split is reachable only through its explicit flag, and this
    constructor is not a way around that."""
    with pytest.raises(ValueError, match="not train-split configs"):
        EmitterPool.from_configs(["config_does_not_exist"])
    with pytest.raises(ValueError):
        EmitterPool.from_configs([])


def test_the_split_is_deterministic():
    """Same rule, same answer -- selection scores are comparable across sessions
    only if the validation half is the same set every time."""
    assert split.training_configs() == split.training_configs()
    assert split.validation_configs() == split.validation_configs()
