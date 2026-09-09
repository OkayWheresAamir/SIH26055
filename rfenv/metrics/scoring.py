"""The scheduler-level table (`EVALUATION.md` §4) and per-band structure (gate 2)."""

from __future__ import annotations

import numpy as np

from rfenv.constants import SLOT_S
from rfenv.metrics.artefacts import RunArtefacts


def scheduler_metrics(run: RunArtefacts) -> dict:
    """`EVALUATION.md` §4 for one episode, from the artefacts alone.

    The two headline metrics are the problem statement's own objectives and are
    reported together, always, with coverage beside them -- §4's rule, and both
    traps behind it were measured on real data. Two of the definitions are easy to
    get wrong in the same direction:

    **Interception ratio is per illumination, not per dwell.** The numerator is
    pulses the tuned window contained, summed over the log; the denominator is the
    scenario's total illuminations, from the header. A per-dwell rate makes a
    camper look near-optimal at 85-90% while it captures 30% of emitters.

    **Censored intercept time counts a miss at the full episode**, not as a
    dropped row. Averaging over only the emitters you found rewards not looking:
    measured, that makes the camper look *faster* than round-robin.

    Delays are measured from `on_e`, the start of the detectable interval (D27),
    so an emitter that only switches on at 20 s is not charged for the 20 s before
    it existed. `first_e >= on_e` holds by construction (D28).
    """
    header = run.header
    n_slots = int(header["n_slots"])

    pulses_intercepted = sum(row["pulses"] for row in run.log)
    total_pulses = int(header["total_pulses"])

    delays = [
        (n_slots if row["first_intercept_slot"] is None else row["first_intercept_slot"])
        - row["on_slot"]
        for row in run.emitters
    ]
    n_found = sum(1 for row in run.emitters if row["first_intercept_slot"] is not None)
    n_e = len(run.emitters)

    return {
        "scheduler": header["scheduler"],
        "scenario": header["scenario"],
        "seed": header["seed"],
        "reward": header["reward"],
        "interception_ratio": pulses_intercepted / total_pulses if total_pulses else float("nan"),
        "censored_mean_intercept_time_s": float(np.mean(delays)) * SLOT_S if delays else float("nan"),
        "emitter_coverage": n_found / n_e if n_e else float("nan"),
        "avg_intercept_rate_per_s": n_found / float(header["episode_s"]),
        "total_reward": float(header["total_reward"]),
        "n_detectable": n_e,
        "n_intercepted": n_found,
        "pulses_intercepted": pulses_intercepted,
        "total_pulses": total_pulses,
        "n_steps": int(header["n_steps"]),
    }


def per_band(run: RunArtefacts) -> dict[str, np.ndarray]:
    """Per-band airtime, declarations, occupancy and captured illuminations.

    Gate 2 asks whether band-level structure matches the recordings rather than
    just the aggregate, and the waterfall wants the same numbers as a margin plot.

    Note what is *not* here: a per-band interception **ratio**. Its denominator is
    how many illuminations each band held in the scenario, which is grid-level and
    not in the artefacts -- the log only knows about bands the scheduler visited.
    Gate 2 takes that denominator from the truth grid directly.
    """
    n = int(run.header["n_bands"])
    out = {k: np.zeros(n, dtype=np.int64) for k in ("slots_looked", "hits", "occupied", "pulses")}
    for row in run.log:
        b = row["band"]
        out["slots_looked"][b] += 1
        out["hits"][b] += row["Y"]
        out["occupied"][b] += row["Z"]
        out["pulses"][b] += row["pulses"]

    looked = out["slots_looked"] > 0
    hit_rate = np.zeros(n, dtype=np.float64)
    np.divide(out["hits"], out["slots_looked"], out=hit_rate, where=looked)
    out["hit_rate"] = hit_rate
    return out
