"""The baseline ladder of `docs/project/EVALUATION.md` §5.

Seven schedulers and one reference line, all consuming L3 and nothing else. They
live outside the environment on purpose (`ENVIRONMENT_SPEC.md` §Build order): a
baseline that could reach inside `ScanEnv` would be able to cheat, and the whole
value of the ladder is that every rung is scored through the same interface the
RL policy will use.

**Every rung except the oracle is deployable.** A scheduler sees its own scan
history and the receiver's declaration `Y`, and nothing else -- D19, D20, D34.
That constraint is enforced here rather than trusted: `guarded()` strips `info`
down to `OBSERVABLE_INFO` before the policy sees it, so a baseline that reaches
for `Z`, `pulses` or `first_intercept` raises a `KeyError` instead of quietly
scoring well. The reward-side asymmetry of D29 does not help a baseline: none of
these reads a reward at all.

**The oracle is exempt, and is not a baseline.** `PulseCaptureOracle` is handed
the truth grid at construction. It is the ceiling line of §5, marked as such
everywhere it is reported, and it is never a competitor.

**Every rung is a fresh object per episode.** D20 starts each episode cold -- no
threat library, no carry-over -- so the ladder is a dict of *factories*, not of
policies. Nothing here survives a `reset()`.

**Where the rungs came from.** Rungs 1-5 are D13's set. Rung 3 is Turing's own
frozen `constants.dwell_schedule()`. Rungs 6 and 6a are Apfeld et al.,
`docs/reference/scheduling/paperSSPD (1).pdf` §II, adapted to a binary-detection
receiver -- see `Apfeld` for exactly what was adapted and why. Rung 2 was
*changed* by D43: as written, `EVALUATION.md` §5's round-robin and Turing sweep
were the same policy and reproduced each other to four decimals (D36), so the
ladder had six rungs and claimed seven.

**Package layout.** One rung family per file: `guard.py` (the deployability
contract), `simple.py` (rungs 1-3), `camper.py` (rung 4 + its truth-fed twin),
`recency.py` (rung 5), `apfeld.py` (rungs 6/6a), `oracle.py` (the ceiling
line), `ladder.py` (the registry that ties every rung together).

**A hard explore/exploit gate (rungs 12/13) was tried and removed (D66).**
Measured worse than the camper it was meant to fix. See
`docs/project/PHASE_SWITCH_FUTURE_WORK.md` for what's still worth trying.
"""

from __future__ import annotations

from rfenv.baselines.apfeld import Apfeld, ApfeldParams
from rfenv.baselines.camper import GreedyCamper, OracleCamper, PROBE_SWEEPS
from rfenv.baselines.guard import (
    CLOCK,
    CURRENT_BAND,
    CURRENT_HIT_STREAK,
    HIT_RATE,
    HIT_STREAK,
    MEASURED_DBM,
    OBSERVABLE_INFO,
    STALENESS,
    VISIT_DENSITY,
    guarded,
    restrict,
)
from rfenv.baselines.ladder import (
    BY_KEY,
    LADDER,
    Rung,
    SCHEDULERS,
    band_at_slot_from_log,
    cycle_summary,
    make,
)
from rfenv.baselines.oracle import PulseCaptureOracle
from rfenv.baselines.recency import RecencyActivity
from rfenv.baselines.simple import (
    BAND_AT_SLOT,
    EQUAL_AIRTIME_CYCLE,
    EQUAL_AIRTIME_CYCLE_SLOTS,
    RandomBands,
    RoundRobin,
    SWEEP_SLOTS,
    TuringSweep,
)

__all__ = [
    "Apfeld",
    "ApfeldParams",
    "BAND_AT_SLOT",
    "BY_KEY",
    "CLOCK",
    "CURRENT_BAND",
    "CURRENT_HIT_STREAK",
    "EQUAL_AIRTIME_CYCLE",
    "EQUAL_AIRTIME_CYCLE_SLOTS",
    "GreedyCamper",
    "HIT_RATE",
    "HIT_STREAK",
    "LADDER",
    "MEASURED_DBM",
    "OBSERVABLE_INFO",
    "OracleCamper",
    "PROBE_SWEEPS",
    "PulseCaptureOracle",
    "RandomBands",
    "RecencyActivity",
    "RoundRobin",
    "Rung",
    "SCHEDULERS",
    "STALENESS",
    "SWEEP_SLOTS",
    "TuringSweep",
    "VISIT_DENSITY",
    "band_at_slot_from_log",
    "cycle_summary",
    "guarded",
    "make",
    "restrict",
]
