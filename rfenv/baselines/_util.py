"""Tiny helpers shared by more than one rung."""

from __future__ import annotations

import numpy as np


def _argmax_random_tie(scores: np.ndarray, rng: np.random.Generator) -> int:
    """`argmax` with ties broken by the policy's own RNG.

    Ties are not an edge case here: at slot 0 every band has hit rate 0 and
    staleness 1, so a plain `np.argmax` would hand band 0 a permanent head start
    and turn an index policy into a band-order artefact.
    """
    best = np.flatnonzero(scores >= scores.max())
    return int(best[0] if best.size == 1 else best[rng.integers(best.size)])
