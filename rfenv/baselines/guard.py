"""What a scheduler is allowed to see.

The observation vector's layout (D34), named once so no policy indexes it with
a bare integer. `env.ScanEnv._observation` builds it and
`tests/test_baselines.py::test_the_observation_slices_match_the_environment`
asserts these slices still describe it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rfenv.constants import N_BANDS

HIT_RATE = slice(0, N_BANDS)          # declared hits per slot looked, per band
VISIT_DENSITY = slice(N_BANDS, 2 * N_BANDS)      # airtime share / fair share; 1.0 = equal cut (D55)
STALENESS = slice(2 * N_BANDS, 3 * N_BANDS)      # (t - last visit) / SWEEP_SLOTS, in sweeps (D55)
CURRENT_BAND = slice(3 * N_BANDS, 4 * N_BANDS)   # one-hot of the band just dwelt on
CLOCK = 4 * N_BANDS                   # normalised episode time
MEASURED_DBM = 4 * N_BANDS + 1        # last dwell's mean measured dBm, clamped+scaled

# `CAMP_TIME` sat at 4 * N_BANDS + 1 until D55 removed it from the vector. It is not
# reinstated here: a policy that wants the streak reads visit_density, which says
# the same thing without waiting for the agent to commit to camping first.

# The keys of `info` a deployable scheduler may read. `slot`/`time_s` are the
# receiver's own clock; `band`, `dwell_slots` and `Y` are what its last look did
# and what it declared. Everything else in `info` is truth-side and belongs to
# the evaluator (D29): `Z`, `pulses`, `newly_intercepted`, `first_intercept`,
# `detectable`, `n_detectable`, `emitter_table`, `total_pulses`.
OBSERVABLE_INFO = frozenset({"slot", "time_s", "band", "dwell_slots", "Y"})


def restrict(info: dict) -> dict:
    """`info` as a deployable scheduler is allowed to see it."""
    return {k: v for k, v in info.items() if k in OBSERVABLE_INFO}


def guarded(policy: Callable[[np.ndarray, dict], int]):
    """Wrap a policy so it is *given* only what it is allowed to read.

    Enforcement, not documentation. A policy that reaches for a truth key gets a
    `KeyError` on the first step, in a test, rather than an unexplained good
    score in a results table.
    """
    return lambda obs, info: policy(obs, restrict(info))
