"""The artefacts, and the scorecard computed from them.

`EVALUATION.md` §8 names four things the environment must emit so that evaluation
is possible at all. This module owns three of them -- the episode log, the emitter
table and `metrics.json` -- plus the run header they turned out to need. The
fourth, the waterfall, is `render.py`.

**The scorecard is computed from the artefacts, not from the live environment.**
That is the whole point of the module and it is not a stylistic preference:
`EVALUATION.md` §6 requires that "a validation script reproduces all four gates
from these artefacts alone", and the only way to know that is true is to never
take a shortcut through `ScanEnv`. So `scheduler_metrics()` reads CSV rows and a
JSON header, exactly as a reader six months from now would. `env.episode_metrics()`
computes the same table from live state, and
`tests/test_metrics.py::test_the_artefacts_reproduce_the_environments_own_numbers`
asserts the two agree exactly. If they ever diverge, the artefacts are wrong and
every gate built on them is wrong with them.

**The run header, and why §8's artefact list was one short.** §8 says everything in
§4 is computable from artefacts 1 and 2 alone. It is not, by two items. Interception
ratio's *denominator* is the scenario's total illuminations, which is a grid-level
quantity appearing in neither the per-slot log nor the emitter table -- the log
carries only the numerator. And average reward appears nowhere at all: reward is
per slot (D31) but candidate 3 pays out per dwell on set membership, so there is no
honest per-slot column for it. Both are scalars, so the fix is a scalar header
rather than a wider log (D38).

Everything here is per scenario and per seed. `EVALUATION.md` §7 forbids reporting a
grand mean alone -- scenario difficulty spans 2 to 99 emitters -- so `aggregate()`
returns a distribution and never a bare average.

**Package layout.** Split by concern: `_serialize.py` (CSV/JSON plumbing),
`artefacts.py` (`RunArtefacts`, read/write), `scoring.py` (the §4 table,
`per_band`), `views.py` (arrays a picture is drawn from), `aggregate.py`
(cross-episode distributions, `metrics.json`). `run_episode` moved out entirely,
to `rfenv/rollout.py` -- it never touched a CSV, an artefact, or a score, and
living here was a historical accident.
"""

from __future__ import annotations

from rfenv.metrics._serialize import (
    EMITTER_FIELDS,
    EMITTER_NAME,
    HEADER_NAME,
    LOG_FIELDS,
    LOG_NAME,
    METRICS_NAME,
)
from rfenv.metrics.aggregate import aggregate, write_metrics_json
from rfenv.metrics.artefacts import RunArtefacts, read_run, run_dir, write_run
from rfenv.metrics.scoring import per_band, scheduler_metrics
from rfenv.metrics.views import discovery, emitter_activity, schedule_series

__all__ = [
    "EMITTER_FIELDS",
    "EMITTER_NAME",
    "HEADER_NAME",
    "LOG_FIELDS",
    "LOG_NAME",
    "METRICS_NAME",
    "RunArtefacts",
    "aggregate",
    "discovery",
    "emitter_activity",
    "per_band",
    "read_run",
    "run_dir",
    "schedule_series",
    "scheduler_metrics",
    "write_metrics_json",
    "write_run",
]
