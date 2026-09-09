"""Views a picture needs (`EVALUATION.md` §8 artefact 4).

Derived from the artefacts and returning arrays, not figures. Kept here rather
than in `render/` so the numbers behind every scheduler plot can be asserted
without a plotting stack, and so two plots of the same run cannot disagree
about what the run did.
"""

from __future__ import annotations

import numpy as np

from rfenv.constants import SLOT_S
from rfenv.metrics.artefacts import RunArtefacts


def schedule_series(run: RunArtefacts) -> dict[str, np.ndarray]:
    """One episode as the arrays a timeline is drawn from.

    `slot`, `band` and `hit` are per slot and cover every slot exactly once --
    `read_run` has already refused the file otherwise. `dwell_start` is the
    boolean "a new look begins here", which is what makes a two-slot dwell
    visible as one decision rather than two; `dwell_slot0` lists those starts.

    A dwell boundary is a change of band *or* the end of a previous dwell's run,
    so it is read off the log's own `dwell_slots` column rather than guessed from
    band changes: two consecutive one-slot dwells on the same band are two
    decisions and must draw as two.
    """
    rows = sorted(run.log, key=lambda r: r["slot"])
    slot = np.array([r["slot"] for r in rows], dtype=np.int64)
    band = np.array([r["band"] for r in rows], dtype=np.int64)
    hit = np.array([r["Y"] for r in rows], dtype=bool)
    occupied = np.array([r["Z"] for r in rows], dtype=bool)
    length = np.array([r["dwell_slots"] for r in rows], dtype=np.int64)

    start = np.zeros(len(rows), dtype=bool)
    i = 0
    while i < len(rows):
        start[i] = True
        i += max(1, int(length[i]))
    return {
        "slot": slot,
        "time_s": slot * SLOT_S,
        "band": band,
        "hit": hit,
        "occupied": occupied,
        "dwell_slots": length,
        "dwell_start": start,
        "dwell_slot0": slot[start],
    }


def discovery(run: RunArtefacts) -> dict[str, np.ndarray]:
    """Distinct emitters intercepted so far, against time -- the coverage curve.

    Coverage is a single number at the end of the episode; this is the path it
    took to get there, and it is the one picture in which round-robin's advantage
    over a camper is visible rather than inferred. The ceiling is `|E|`, the
    detectable set (D27), so an emitter that was never findable does not depress
    the curve.

    `first_intercept_slot` is `None` for an emitter never found; those simply
    never step the curve, which is the censoring rule drawn rather than averaged.
    """
    firsts = sorted(
        row["first_intercept_slot"] for row in run.emitters
        if row["first_intercept_slot"] is not None
    )
    n_slots = int(run.header["n_slots"])
    found = np.zeros(n_slots + 1, dtype=np.int64)
    for slot in firsts:
        found[int(slot) + 1:] += 1
    return {
        "slot": np.arange(n_slots + 1, dtype=np.int64),
        "time_s": np.arange(n_slots + 1, dtype=np.float64) * SLOT_S,
        "n_found": found,
        "n_detectable": run.n_detectable,
    }


def emitter_activity(run: RunArtefacts) -> list[dict]:
    """Per-emitter detectable interval and first intercept, ordered by onset.

    The rows a Gantt strip is drawn from. Ordering by `on_slot` and then by first
    intercept puts the emitters that appear late at the top of the strip, which is
    where the differences between schedulers actually live: everything is easy in
    the first two seconds.
    """
    return sorted(
        (
            {
                "emitter": row["emitter"],
                "uid": row["uid"],
                "on_slot": row["on_slot"],
                "off_slot": row["off_slot"],
                "first_intercept_slot": row["first_intercept_slot"],
                "intercept_count": row["intercept_count"],
            }
            for row in run.emitters
        ),
        key=lambda r: (r["on_slot"], r["first_intercept_slot"] is None,
                       r["first_intercept_slot"] or 0),
    )
