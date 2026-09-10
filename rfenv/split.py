"""The development split: which of the 47 train configs train, which validate.

**The rule is fixed here, and it was written before anyone looked at which
configs land on which side.** That ordering is the entire value of the file. A
split chosen after seeing which configs make a model look good is not a split, in
exactly the way D39 says a threshold chosen after seeing the measurement is not a
threshold.

Why this exists at all: training samples scenarios from `EmitterPool`, and until
now that pool was built from all 47 train configs while evaluation ran on those
same 47 replays plus scenarios sampled from that same pool. The baselines do not
train, so the asymmetry ran one way -- ours -- and any margin it produced was not
a real advantage over rung 5. Splitting the *config list* is not enough on its
own, because the pool is assembled from the configs: the emitters leak unless the
pool is rebuilt from the training half only, which is what
`EmitterPool.from_configs(training_configs())` is for.

**The rule.** Order the 47 train configs by how many emitters were detectable in
them (ascending), breaking ties by config id. Take every 4th, starting from index
1, into validation; everything else trains. That yields 12 validation and 35
training configs.

Systematic sampling on the difficulty variable rather than a hash, for one
reason: scenario difficulty spans 2 to 99 emitters and dominates every metric in
`EVALUATION.md` §4, so a uniform random draw can easily hand validation a set
that is systematically easier or harder than training, and we would not know
which. Taking every 4th along the sorted order guarantees both halves span the
whole difficulty range by construction. The offset of 1 rather than 0 keeps the
single easiest config in training, where a degenerate 2-emitter scenario is less
likely to distort a small validation set.

Emitter counts come from `EmitterPool.emitter_counts` -- recording metadata, not
a result. Nothing about any scheduler's performance enters this rule.

**The 45 held-out test pairs are not touched here and are not part of this.**
This splits the *development* set. The held-out split stays sealed until there is
a winner worth spending it on, and `scenario.list_configs` still refuses it
without an explicit flag (D8).
"""

from __future__ import annotations

from functools import lru_cache

from rfenv.scenario import TRAIN_SPLIT, EmitterPool, list_configs

# Take every Nth config into validation, starting at OFFSET, along the
# difficulty-sorted order. 47 configs -> 12 validation, 35 training.
VALIDATION_EVERY = 4
VALIDATION_OFFSET = 1


@lru_cache(maxsize=1)
def _ordered() -> tuple[str, ...]:
    """The 47 train configs, ascending by detectable-emitter count, ties by id."""
    configs = list_configs("scan", TRAIN_SPLIT)
    counts = EmitterPool.from_train().emitter_counts
    if len(counts) != len(configs):
        raise RuntimeError(
            f"emitter_counts has {len(counts)} entries for {len(configs)} configs -- "
            "EmitterPool.from_train() and list_configs disagree, so the split cannot "
            "be ordered reproducibly"
        )
    return tuple(c for _, c in sorted(zip(counts.tolist(), configs)))


def validation_configs() -> tuple[str, ...]:
    """The 12 held-back configs. Checkpoints are selected on these and never trained on."""
    return tuple(sorted(_ordered()[VALIDATION_OFFSET::VALIDATION_EVERY]))


def training_configs() -> tuple[str, ...]:
    """The 35 configs the emitter pool is built from and training samples."""
    held = set(validation_configs())
    return tuple(sorted(c for c in _ordered() if c not in held))


def training_pool(**kwargs) -> EmitterPool:
    """The pool RL trains against: training-half emitters only.

    This is the call that actually closes the leak. `EmitterPool.from_train()`
    still exists and still means all 47 -- it is what `compare.py` samples the
    headline scenarios from, which is correct, because the headline is reported
    on the full development set. Training must not use it.
    """
    return EmitterPool.from_configs(training_configs(), **kwargs)


def validation_pool(**kwargs) -> EmitterPool:
    """The pool checkpoint selection samples from. Disjoint from `training_pool`."""
    return EmitterPool.from_configs(validation_configs(), **kwargs)


def describe() -> str:
    train, val = training_configs(), validation_configs()
    return (
        f"development split (rfenv/split.py): {len(train)} training, {len(val)} validation\n"
        f"  rule: every {VALIDATION_EVERY}th config from offset {VALIDATION_OFFSET}, "
        f"ordered by detectable-emitter count ascending\n"
        f"  validation: {', '.join(val)}\n"
    )


if __name__ == "__main__":
    print(describe())
