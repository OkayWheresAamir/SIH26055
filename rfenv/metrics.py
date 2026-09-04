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
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rfenv.constants import (
    EPISODE_S,
    N_BANDS,
    N_SLOTS,
    SLOT_S,
)

# --------------------------------------------------------------------------- #
# File format
# --------------------------------------------------------------------------- #
#
# CSV rather than parquet. An episode is 600 rows of ten columns; the whole
# scheduler comparison -- 7 schedulers x 47 scenarios x 5 seeds -- is about a
# million rows, which CSV handles without a dependency. Parquet would buy
# compression we do not need and cost a build dependency on pyarrow.

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


# --------------------------------------------------------------------------- #
# Driving an episode
# --------------------------------------------------------------------------- #

def run_episode(env, policy, *, seed: int | None = None, scenario=None):
    """Run one episode to termination and return the environment.

    `policy(observation, info) -> band`. Both arguments, because a schedule that
    depends on the clock -- Turing's own sweep is the obvious one -- needs the slot,
    and `info["slot"]` is where it lives. The observation alone would work
    (`obs[-1]` is normalised episode time) but reading the slot back out of a
    normalised float is a trap waiting for whoever writes the next baseline.

    This is a driver, not a baseline. It defines no policy: the baseline ladder
    (`EVALUATION.md` §5) lives outside the environment and consumes only L3 and
    these artefacts.
    """
    options = {"scenario": scenario} if scenario is not None else None
    obs, info = env.reset(seed=seed, options=options)
    terminated = False
    while not terminated:
        obs, _, terminated, _, info = env.step(int(policy(obs, info)))
    return env


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #

@dataclass
class RunArtefacts:
    """One episode as it exists on disk: a header, a per-slot log, an emitter table.

    This is the only object `scheduler_metrics()` will look at. Nothing here
    remembers a `ScanEnv`, and that is deliberate.
    """

    header: dict
    log: list[dict]
    emitters: list[dict]

    @property
    def n_detectable(self) -> int:
        """`|E|`, the coverage denominator (D27) -- one row per detectable emitter."""
        return len(self.emitters)


def run_dir(root: str | Path, scheduler: str, scenario: str, seed: int | None) -> Path:
    """Where one episode's artefacts go: `<root>/<scheduler>/<scenario>__seed<k>`.

    Scenario names carry colons and slashes (`replay:train/stare/config_2`), so
    they are flattened. The flattened name stays readable, which matters more than
    being reversible -- the unflattened name is in the header.
    """
    safe = "".join(c if c.isalnum() or c in "-." else "_" for c in scenario)
    tag = "seedNone" if seed is None else f"seed{seed}"
    return Path(root) / scheduler / f"{safe}__{tag}"


def write_run(
    directory: str | Path,
    env,
    *,
    scheduler: str,
    seed: int | None = None,
    extra: dict | None = None,
) -> RunArtefacts:
    """Write one finished episode's three artefacts, and return them as read back.

    Returning the *read-back* copy rather than the in-memory rows is not
    ceremony: it means every caller -- including the tests -- scores the bytes on
    disk, so a serialisation bug cannot hide behind live objects.

    `seed` is passed in because the environment does not keep one. Gymnasium
    seeds `self.np_random` and discards the integer, and a seed that is not
    recorded is a run that cannot be reproduced (`EVALUATION.md` §7).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    log = list(env.log)
    if log and tuple(log[0]) != LOG_FIELDS:
        raise ValueError(
            f"episode log columns changed: env writes {tuple(log[0])}, "
            f"this module writes {LOG_FIELDS}. Update both together."
        )

    emitters = env.emitter_table()
    header = {
        "scheduler": scheduler,
        "scenario": env.scenario.name,
        "seed": seed,
        "reward": env.reward_name,
        "gamma_dbm": env.gamma,
        "sigma_db": env.sigma,
        "episode_s": EPISODE_S,
        "n_slots": N_SLOTS,
        "n_bands": N_BANDS,
        # The two quantities §4 needs and artefacts 1 and 2 do not carry (D38).
        "total_pulses": env.grid.total_pulses,
        "total_reward": env.total_reward,
        # Redundant with the emitter table's row count and the log's dwell runs,
        # deliberately: `read_run` checks them against each other, so a truncated
        # or half-written artefact set fails loudly instead of scoring low.
        "n_detectable": len(env.detectable),
        "n_steps": env.n_steps,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    header.update(extra or {})

    (directory / HEADER_NAME).write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")
    _write_csv(directory / LOG_NAME, LOG_FIELDS, log)
    _write_csv(directory / EMITTER_NAME, EMITTER_FIELDS, emitters)
    return read_run(directory)


def read_run(directory: str | Path) -> RunArtefacts:
    """Read one episode's artefacts back, checking them against each other."""
    directory = Path(directory)
    header = json.loads((directory / HEADER_NAME).read_text())
    log = _read_csv(directory / LOG_NAME, _LOG_TYPES)
    emitters = _read_csv(directory / EMITTER_NAME, _EMITTER_TYPES)

    if len(log) != header["n_slots"]:
        raise ValueError(
            f"{directory}: log has {len(log)} rows but the header says "
            f"{header['n_slots']} slots. An episode covers every slot exactly once."
        )
    if len(emitters) != header["n_detectable"]:
        raise ValueError(
            f"{directory}: emitter table has {len(emitters)} rows but the header "
            f"says {header['n_detectable']} detectable emitters."
        )
    return RunArtefacts(header=header, log=log, emitters=emitters)


# --------------------------------------------------------------------------- #
# The scheduler-level table (EVALUATION.md §4)
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Across episodes
# --------------------------------------------------------------------------- #

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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")
    return path


def _jsonable(value):
    """numpy scalars and arrays are not JSON; everything else is a bug."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")
