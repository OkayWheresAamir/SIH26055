"""CSV/JSON plumbing for the episode log, emitter table and `metrics.json`.

CSV rather than parquet. An episode is 600 rows of ten columns; the whole
scheduler comparison -- 7 schedulers x 47 scenarios x 5 seeds -- is about a
million rows, which CSV handles without a dependency. Parquet would buy
compression we do not need and cost a build dependency on pyarrow.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

LOG_FIELDS = (
    "slot", "time_s", "band", "dwell_slots", "Y", "Z", "pulses",
    "level_dbm", "measured_dbm", "last_hit_slot",
)

EMITTER_FIELDS = (
    "emitter", "uid", "on_slot", "off_slot",
    "first_intercept_slot", "last_intercept_slot", "intercept_count", "bands_seen_in",
)

LOG_NAME = "episode_log.csv"
EMITTER_NAME = "emitter_table.csv"
HEADER_NAME = "run.json"
METRICS_NAME = "metrics.json"


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _encode(row[k]) for k in fields})


def _encode(value):
    """CSV cell for one Python value.

    Bools become 1/0 rather than True/False so the column parses as a number in
    any tool. `None` becomes empty, which is how an emitter that was never
    intercepted records its missing first-intercept slot -- distinct from slot 0.
    A list of bands becomes space-separated ints; it is the only non-scalar we
    write and it is never long.
    """
    if isinstance(value, bool):
        return 1 if value else 0
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(int(v)) for v in sorted(value))
    return value


def _int_or_none(text: str):
    return None if text == "" else int(text)


_LOG_TYPES = {
    "slot": int, "time_s": float, "band": int, "dwell_slots": int,
    "Y": lambda s: bool(int(s)), "Z": lambda s: bool(int(s)), "pulses": int,
    "level_dbm": float, "measured_dbm": float, "last_hit_slot": int,
}

_EMITTER_TYPES = {
    "emitter": int, "uid": str, "on_slot": int, "off_slot": int,
    "first_intercept_slot": _int_or_none, "last_intercept_slot": _int_or_none,
    "intercept_count": int,
    "bands_seen_in": lambda s: [int(b) for b in s.split()] if s else [],
}


def _read_csv(path: Path, types: dict) -> list[dict]:
    with path.open(newline="") as fh:
        return [{k: types[k](v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _jsonable(value):
    """numpy scalars and arrays are not JSON; everything else is a bug."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")
