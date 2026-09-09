"""Cross-episode distributions, and `metrics.json` (artefact 4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rfenv.metrics._serialize import _jsonable

_SPREAD_KEYS = ("interception_ratio", "censored_mean_intercept_time_s", "emitter_coverage",
                "avg_intercept_rate_per_s", "total_reward", "n_detectable",
                "n_intercepted", "n_steps")


def aggregate(rows: list[dict], keys: tuple[str, ...] = _SPREAD_KEYS) -> dict:
    """Distributions across episodes -- never a bare mean.

    `EVALUATION.md` §7: report per-scenario results and repeated-run statistics
    with spread, because scenario difficulty spans 2 to 99 emitters and a grand
    mean over that is close to meaningless. Quartiles rather than a standard
    deviation alone, since these distributions are not symmetric -- censored
    intercept time piles up at the 30 s censoring point.

    NaNs are dropped per key rather than per row: a scenario with no illuminations
    has no interception ratio but still has a coverage figure worth keeping.
    """
    out: dict[str, dict] = {"n_runs": len(rows)}
    for key in keys:
        values = np.array([r[key] for r in rows if key in r], dtype=np.float64)
        values = values[~np.isnan(values)]
        if not values.size:
            out[key] = {"n": 0}
            continue
        out[key] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)),
            "max": float(values.max()),
        }
    return out


def write_metrics_json(
    path: str | Path,
    *,
    scheduler_runs: dict[str, list[dict]] | None = None,
    receiver: dict | None = None,
    model: dict | None = None,
    operating_point: dict | None = None,
    notes: str | None = None,
) -> Path:
    """Artefact 4: the three metric families, one file per evaluation run.

    The families are kept apart because mixing them is what most of this
    project's early confusion was made of (`EVALUATION.md` §0). Model-level asks
    whether the environment is faithful, receiver-level asks how good the detector
    is, and only scheduler-level varies with the scheduler at all. A scheduler
    comparison reported on receiver-level metrics is a category error, and keeping
    them in separate blocks of one file makes that hard to do by accident.

    The operating point is stamped on every file, whatever the file contains --
    §7's reporting rule: never state a scheduler table without the gamma and the
    Pfa it was measured at.

    `model` is left empty here. It is filled by `validate.py`, because both
    model-level metrics compare the environment against the actual recordings and
    that comparison is the validation gate, not a per-run summary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scheduler_block = {
        name: {"per_scenario": rows, "aggregate": aggregate(rows)}
        for name, rows in (scheduler_runs or {}).items()
    }
    payload = {
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operating_point": operating_point or {},
        "model": model or {},
        "receiver": receiver or {},
        "scheduler": scheduler_block,
    }
    if notes:
        payload["notes"] = notes
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    return path
