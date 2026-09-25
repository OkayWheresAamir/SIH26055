"""`RunArtefacts`, and reading/writing one episode's three files to disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rfenv.constants import N_BANDS, N_SLOTS, SLOT_S
from rfenv.metrics._serialize import (
    _EMITTER_TYPES,
    _LOG_TYPES,
    _read_csv,
    _write_csv,
    EMITTER_FIELDS,
    EMITTER_NAME,
    HEADER_NAME,
    LOG_FIELDS,
    LOG_NAME,
)


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

    # `read_run` checks the log covers every slot exactly once, so a windowed
    # log (D80, for long missions) cannot be written as if it were an episode.
    # Refusing is the honest outcome: the alternative is a header that claims a
    # length the rows do not have, which is exactly the class of silently-wrong
    # artefact the cross-checks below exist to catch.
    if getattr(env, "log_window_slots", None) is not None:
        raise ValueError(
            "this env keeps a windowed log (log_window_slots="
            f"{env.log_window_slots}), so its rows are not a whole episode and "
            "cannot be written as run artefacts. Use the per-segment metrics "
            "stream instead, or run with log_window_slots=None."
        )

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
        # This episode's own length, not the constant: identical at the default,
        # and the only thing that keeps `read_run`'s slot-count cross-check
        # meaningful on a stitched mission (D80).
        "episode_s": getattr(env, "episode_slots", N_SLOTS) * SLOT_S,
        "n_slots": getattr(env, "episode_slots", N_SLOTS),
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

    (directory / HEADER_NAME).write_text(
        json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
