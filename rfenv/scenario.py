"""L0 — the scenario pipeline.

Turns Turing recordings into the unit the environment is built from: an
**emitter contribution**, which is one emitter, in one recording, as a sparse set of
(band, slot, peak level, pulse count) cells.

Why that unit. TSRD built each config by sampling emitter instances from a 68-type
library and placing them independently, with no emitter-emitter interaction (paper
§II: "line-of-sight path loss without multi-path interference"). So one emitter's
recorded contribution is a valid sample of "one emitter of this type, at a plausible
position and beam phase, over 30 s", and emitters compose by taking the maximum
level per cell. A new scenario is therefore just a draw of N emitters from the
pool -- the same generative process TSRD used, one level up, with no invented
physics and no fitted parameters (D25). One emitter contributes at most once to a
scenario, so an emitter detected in both runs never appears twice (D32).

One contribution comes from exactly one recording. scan and stare are independent
simulation runs, not two views of one world -- the same emitter gets disjoint
activity in each (D24) -- so a contribution is always a single self-consistent
realisation. Both runs feed the pool; neither is discarded (D17's conclusion),
but they are never stitched into one timeline (D17's premise, corrected).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from rfenv.constants import (
    BAND_CENTRES_MHZ,
    CACHE_ROOT,
    DATA_ROOT,
    DWELL_TIMES_S,
    EPISODE_S,
    HELDOUT_SPLIT,
    N_SLOTS,
    SLOT_S,
    SOURCES,
    TRAIN_SPLIT,
    bands_covering,
)

HELDOUT_USE_LOG = Path("docs/project/HELDOUT_USE_LOG.md")


class HeldOutDataError(RuntimeError):
    """Raised when the 45 held-out test pairs are touched without saying so."""


# --------------------------------------------------------------------------- #
# Paths and the held-out guard
# --------------------------------------------------------------------------- #

def recording_path(config_id: str, source: str, split: str = TRAIN_SPLIT) -> Path:
    """Path to one Turing recording. `config_id` is e.g. "config_2"."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return Path(DATA_ROOT) / source / f"{split}_{source}" / f"{config_id}.h5"


def _guard_heldout(path: Path, allow_heldout: bool, reason: str = "") -> None:
    """Refuse the held-out split unless the caller explicitly asked for it (D8).

    The 45 test pairs exist so the final evaluation means something. They are
    touched once, at the end, after the system is frozen -- and every use is
    recorded. This makes that mechanical rather than a matter of remembering.
    """
    if HELDOUT_SPLIT not in Path(path).parts[-2]:
        return
    if not allow_heldout:
        raise HeldOutDataError(
            f"{path} is in the held-out test split (D8). It is not to be touched "
            "during development. If this really is the final evaluation, pass "
            "allow_heldout=True and give a reason -- the call will be logged."
        )
    HELDOUT_USE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not HELDOUT_USE_LOG.exists():
        HELDOUT_USE_LOG.write_text(
            "# Held-out data use log\n\n"
            "Every read of `data/turing/*/test_*` (D8). Appended automatically by\n"
            "`rfenv.scenario`. An entry here is a claim that the system was frozen\n"
            "at the time.\n\n",
            encoding="utf-8",
        )
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with HELDOUT_USE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{stamp}` — `{path}` — {reason or 'no reason given'}\n")


# --------------------------------------------------------------------------- #
# Receiver geometry, read from the files rather than trusted
# --------------------------------------------------------------------------- #

def load_receiver(path: str | Path) -> dict:
    """Read the receiver block and check it against the frozen constants.

    Only *scan* files carry the dwell schedule: in a stare file
    `dwell_centres_mhz` and `dwell_times_s` are present but zero-length, because a
    staring receiver has no sweep. Everything else in the block is shared.
    """
    with h5py.File(path, "r") as fh:
        rx = fh["metadata/receiver"]
        block = {
            "scan_mode": rx.attrs["scan_mode"],
            "bandwith_mhz": float(rx.attrs["bandwith_mhz"]),  # sic, spelled thus in the files
            "sensitivity_dbm": float(rx.attrs["sensitivity_dbm"]),
            "gain_db": float(rx.attrs["gain_db"]),
            "collection_time_s": float(rx.attrs["collection_time_s"]),
            "freq_range_mhz": rx["freq_range_mhz"][:].astype(float),
            "start_position_km": rx["start_position_km"][:].astype(float),
            "dwell_centres_mhz": rx["dwell_centres_mhz"][:].astype(float),
            "dwell_times_s": rx["dwell_times_s"][:].astype(float),
        }
    if block["collection_time_s"] != EPISODE_S:
        raise ValueError(
            f"{path}: collection_time_s is {block['collection_time_s']}, "
            f"but the frozen episode length is {EPISODE_S}"
        )
    if block["dwell_centres_mhz"].size:  # scan file
        if not np.array_equal(block["dwell_centres_mhz"], BAND_CENTRES_MHZ):
            raise ValueError(f"{path}: dwell centres differ from the frozen geometry (D3)")
        if not np.allclose(block["dwell_times_s"], DWELL_TIMES_S):
            raise ValueError(f"{path}: dwell times differ from the frozen schedule (D3)")
    return block


def _read_emitter_metadata(group: h5py.Group) -> dict:
    """One transmitter's config block.

    Beam and position fields are carried through but never computed with: v1 needs
    no beam model, and `scan_rate_rpm` is not literal rpm (it takes 116 distinct
    values across the train set, including 0.0 and 0.1) so its semantics are
    unconfirmed. Kept so the emitter table is complete for the write-up.
    """
    freq, pri, pw = group["frequency_config"], group["pri_config"], group["pulse_width_config"]
    scan, pos, power = group["scan_config"], group["position_config"], group["power_config"]
    return {
        "function": str(group.attrs.get("function", "")),
        "freqs_mhz": freq["freqs_mhz"][:].astype(float),
        "freq_mode": str(freq.attrs.get("freq_mode", "")),
        "pris_us": pri["pris_us"][:].astype(float),
        "pri_mode": str(pri.attrs.get("pri_mode", "")),
        "pws_us": pw["pws_us"][:].astype(float),
        "pw_mode": str(pw.attrs.get("pw_mode", "")),
        "beam_width_deg": float(scan.attrs.get("beam_width_deg", np.nan)),
        "scan_rate_rpm": float(scan.attrs.get("scan_rate_rpm", np.nan)),
        "scan_start_angle": float(scan.attrs.get("scan_start_angle", np.nan)),
        "scan_type": str(scan.attrs.get("scan_type", "")),
        "start_position_km": pos["start_position_km"][:].astype(float),
        "speed_km_s": float(pos.attrs.get("speed_km_s", np.nan)),
        "travel_angle_deg": float(pos.attrs.get("travel_angle_deg", np.nan)),
        "power_w": float(power.attrs.get("power_w", np.nan)),
        "gain": float(power.attrs.get("gain", np.nan)),
    }


# --------------------------------------------------------------------------- #
# The unit: one emitter's contribution to the grid
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EmitterContribution:
    """One emitter, in one recording, as sparse grid cells.

    `cells` is (n, 2) int16 of (band, slot); `peak_dbm`, `n_pulses`, `pulse_width_us`
    and `aoa_deg` are all aligned with it. A cell appears once per band the pulse's
    frequency falls in, so an emitter generally occupies two adjacent bands per
    slot -- the overlap is real (D3) and carrying it is what makes counterfactual
    band queries answerable.

    `pulse_width_us` and `aoa_deg` (D30) are the raw PDW columns 2 and 3, at
    **whichever single pulse set `peak_dbm` in that cell** -- the same pulse, not
    an average over every pulse the cell absorbs, so a cell's three per-pulse
    readings (`peak_dbm`, `pulse_width_us`, `aoa_deg`) always describe one real
    physical pulse together, never a blend of several.
    """

    config_id: str
    source: str
    label: int
    cells: np.ndarray             # (n, 2) int16  -> band, slot
    peak_dbm: np.ndarray          # (n,) float32  -> max pulse amplitude in that cell
    n_pulses: np.ndarray          # (n,) int32    -> pulses in that cell
    pulse_width_us: np.ndarray = None    # (n,) float32 -> PW of the peak_dbm pulse (D30)
    aoa_deg: np.ndarray = None           # (n,) float32 -> AoA of the peak_dbm pulse, degrees (D30)
    total_pulses: int = 0   # pulses actually emitted; NOT n_pulses.sum(), which
                            # counts a pulse once per band its window falls in.
                            # This is the denominator of interception ratio.
    meta: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Defaulted rather than required so every existing call site that builds
        # an `EmitterContribution` by hand (tests, `dataclasses.replace` in
        # `tests/test_threat.py`) keeps working without knowing about D30 -- but
        # never silently: an empty cell set legitimately has no pulses to report,
        # everything else must carry real (n,)-shaped arrays or D30's per-cell
        # invariant ("these three readings are one real pulse together") breaks
        # without anyone noticing.
        if self.pulse_width_us is None:
            object.__setattr__(self, "pulse_width_us", np.zeros(len(self.cells), dtype=np.float32))
        if self.aoa_deg is None:
            object.__setattr__(self, "aoa_deg", np.zeros(len(self.cells), dtype=np.float32))

    @property
    def uid(self) -> str:
        return f"{self.config_id}/{self.source}/{self.label}"

    @property
    def bands(self) -> np.ndarray:
        return self.cells[:, 0]

    @property
    def slots(self) -> np.ndarray:
        return self.cells[:, 1]

    def __len__(self) -> int:
        return len(self.cells)


def _bucket(toa_s, freq_mhz, amp_dbm, pw_us, aoa_deg):
    """Pulses -> (band, slot) cells with peak level, count, and (D30) the
    PulseWidth/AoA of whichever single pulse in that cell set the peak level.

    Slot is floor(t / 50 ms), clipped to the last slot so a pulse landing exactly at
    30.000 s does not fall off the end.

    Grouped by `np.lexsort((-amp_dbm, slot))` rather than a plain stable sort on
    slot alone: primary key `slot` (ascending), secondary key `amp_dbm`
    (descending) -- so within one (band, slot) group, the *first* row after
    sorting is always the loudest pulse, and taking each group's first row
    (`start`, from `np.unique`'s own index) for `peak`/`pw`/`aoa` together reads
    off one real pulse's amplitude, width and bearing as a matched triple, fully
    vectorised (no per-group Python loop).
    """
    slot = np.minimum((toa_s / SLOT_S).astype(np.int64), N_SLOTS - 1)
    keys, peaks, counts, pws, aoas = [], [], [], [], []
    for band, mask in enumerate(bands_covering(freq_mhz)):
        if not mask.any():
            continue
        s, a, pw, ao = slot[mask], amp_dbm[mask], pw_us[mask], aoa_deg[mask]
        # group by slot; within a slot, loudest pulse first
        order = np.lexsort((-a, s))
        s, a, pw, ao = s[order], a[order], pw[order], ao[order]
        uniq, start = np.unique(s, return_index=True)
        peak = a[start]          # the loudest pulse's own amplitude in each group
        pw_sel = pw[start]       # -- and that same pulse's width...
        aoa_sel = ao[start]      # -- and bearing.
        count = np.diff(np.append(start, len(s)))
        keys.append(np.stack([np.full(len(uniq), band, dtype=np.int16),
                              uniq.astype(np.int16)], axis=1))
        peaks.append(peak.astype(np.float32))
        counts.append(count.astype(np.int32))
        pws.append(pw_sel.astype(np.float32))
        aoas.append(aoa_sel.astype(np.float32))
    if not keys:
        empty_f = np.zeros(0, np.float32)
        return (np.zeros((0, 2), np.int16), empty_f, np.zeros(0, np.int32), empty_f, empty_f)
    return (np.concatenate(keys), np.concatenate(peaks), np.concatenate(counts),
            np.concatenate(pws), np.concatenate(aoas))


def build_contributions(
    config_id: str,
    source: str,
    split: str = TRAIN_SPLIT,
    allow_heldout: bool = False,
    reason: str = "",
) -> list[EmitterContribution]:
    """Read one recording and return one contribution per emitter seen in it.

    `labels[:, 0]` indexes `metadata/transmitters/transmitters_N` directly --
    verified by each label's observed median frequency matching that transmitter's
    `freqs_mhz`. Emitters with no pulses in this recording produce no contribution:
    of 2,363 train transmitters, 1,739 appear in scan and 1,704 in stare.
    """
    path = recording_path(config_id, source, split)
    _guard_heldout(path, allow_heldout, reason)
    load_receiver(path)  # geometry assertion

    with h5py.File(path, "r") as fh:
        data = fh["data"][:]
        labels = fh["labels"][:, 0].astype(np.int64)
        meta = {
            i: _read_emitter_metadata(fh[f"metadata/transmitters/transmitters_{i}"])
            for i in range(len(fh["metadata/transmitters"]))
        }

    toa_s = data[:, 0].astype(np.float64) / 1e6  # ToA column is microseconds
    freq_mhz = data[:, 1].astype(np.float64)
    pw_us = data[:, 2].astype(np.float64)     # D30: previously read and discarded
    aoa_deg = data[:, 3].astype(np.float64)   # D30: previously read and discarded
    amp_dbm = data[:, 4].astype(np.float64)

    inside = (toa_s >= 0.0) & (toa_s < EPISODE_S)
    toa_s, freq_mhz, pw_us, aoa_deg, amp_dbm, labels = (
        toa_s[inside], freq_mhz[inside], pw_us[inside], aoa_deg[inside],
        amp_dbm[inside], labels[inside]
    )

    out = []
    for label in np.unique(labels):
        m = labels == label
        cells, peaks, counts, pws, aoas = _bucket(
            toa_s[m], freq_mhz[m], amp_dbm[m], pw_us[m], aoa_deg[m]
        )
        if not len(cells):
            continue
        out.append(EmitterContribution(
            config_id=config_id, source=source, label=int(label),
            cells=cells, peak_dbm=peaks, n_pulses=counts,
            pulse_width_us=pws, aoa_deg=aoas,
            total_pulses=int(m.sum()), meta=meta.get(int(label), {}),
        ))
    return out


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def _cache_path(config_id: str, source: str, split: str) -> Path:
    # `_v2` (D30): the payload gained `pulse_width_us`/`aoa_deg` arrays that a
    # pre-D30 cache file does not have. A different filename means an old cache
    # is simply never found (silently rebuilt), instead of `load_contributions`
    # raising `KeyError: 'pw_<label>'` on every cold machine that already has a
    # `runs/cache/` from before this change.
    return Path(CACHE_ROOT) / f"{split}_{source}_{config_id}_v2.npz"


def load_contributions(
    config_id: str,
    source: str,
    split: str = TRAIN_SPLIT,
    *,
    rebuild: bool = False,
    allow_heldout: bool = False,
    reason: str = "",
) -> list[EmitterContribution]:
    """Contributions for one recording, from cache when available.

    A cold build is 1-3 s per config; the cache makes repeated episode setup free.
    """
    cache = _cache_path(config_id, source, split)
    if cache.exists() and not rebuild:
        with np.load(cache, allow_pickle=True) as z:
            metas = z["meta"].item()
            totals = z["total_pulses"]
            return [
                EmitterContribution(
                    config_id=config_id, source=source, label=int(lab),
                    cells=z[f"cells_{lab}"], peak_dbm=z[f"peak_{lab}"],
                    n_pulses=z[f"count_{lab}"],
                    pulse_width_us=z[f"pw_{lab}"], aoa_deg=z[f"aoa_{lab}"],
                    total_pulses=int(tot),
                    meta=metas.get(int(lab), {}),
                )
                for lab, tot in zip(z["labels"], totals)
            ]

    contribs = build_contributions(
        config_id, source, split, allow_heldout=allow_heldout, reason=reason
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {"labels": np.array([c.label for c in contribs], dtype=np.int64),
               "total_pulses": np.array([c.total_pulses for c in contribs], dtype=np.int64),
               "meta": np.array({c.label: c.meta for c in contribs}, dtype=object)}
    for c in contribs:
        payload[f"cells_{c.label}"] = c.cells
        payload[f"peak_{c.label}"] = c.peak_dbm
        payload[f"count_{c.label}"] = c.n_pulses
        payload[f"pw_{c.label}"] = c.pulse_width_us
        payload[f"aoa_{c.label}"] = c.aoa_deg
    np.savez_compressed(cache, **payload)
    return contribs


def list_configs(source: str = "scan", split: str = TRAIN_SPLIT) -> list[str]:
    """Config ids present for a split, sorted."""
    d = Path(DATA_ROOT) / source / f"{split}_{source}"
    return sorted(p.stem for p in d.glob("config_*.h5"))


# --------------------------------------------------------------------------- #
# Pool and scenarios
# --------------------------------------------------------------------------- #

@dataclass
class EmitterPool:
    """Every emitter contribution available for sampling.

    Built from the train split only, so the 45 held-out pairs stay clean no matter
    how many scenarios are sampled.
    """

    contributions: list[EmitterContribution]
    emitter_counts: np.ndarray  # per-config count of emitters detected at all
    _by_emitter: dict | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_train(cls, *, rebuild: bool = False, sources=SOURCES) -> "EmitterPool":
        """Every train-split emitter -- all 47 configs.

        **This is the evaluation pool, not the training pool.** `compare.py` draws
        its sampled scenarios from here, which is right: the headline is reported
        over the whole development set. RL training must use
        `split.training_pool()` instead, or the emitters it learns on are the same
        ones it is scored against and the margin over a non-training baseline is
        not real (see `rfenv/split.py`).
        """
        return cls.from_configs(list_configs("scan", TRAIN_SPLIT),
                                rebuild=rebuild, sources=sources)

    @classmethod
    def from_configs(cls, config_ids, *, rebuild: bool = False,
                     sources=SOURCES) -> "EmitterPool":
        """A pool built from a named subset of the train split.

        Splitting the config *list* is not enough to separate training from
        evaluation, because the pool is assembled from the configs -- a scenario
        sampled from a pool built over all 47 can contain an emitter from a config
        that was supposed to be held back. This is the constructor that keeps the
        two disjoint.
        """
        config_ids = list(config_ids)
        if not config_ids:
            raise ValueError("an emitter pool needs at least one config")
        known = set(list_configs("scan", TRAIN_SPLIT))
        unknown = [c for c in config_ids if c not in known]
        if unknown:
            raise ValueError(
                f"not train-split configs: {unknown}. `from_configs` is for "
                "subsets of the development set; the held-out split is reached "
                "only through the explicit flag D8 requires."
            )
        contribs, counts = [], []
        for config_id in config_ids:
            seen = set()
            for source in sources:
                got = load_contributions(config_id, source, TRAIN_SPLIT, rebuild=rebuild)
                contribs.extend(got)
                seen.update(c.label for c in got)
            # The count that matters for difficulty is how many emitters were
            # detectable at all, not how many the metadata lists: 19.0% of train
            # transmitters never appear in either recording.
            counts.append(len(seen))
        return cls(contributions=contribs, emitter_counts=np.array(counts))

    def __len__(self) -> int:
        return len(self.contributions)

    @property
    def by_emitter(self) -> dict[tuple[str, int], list[EmitterContribution]]:
        """Contributions grouped by physical emitter, `(config_id, label)`.

        An emitter detected in both runs has two entries: the scan realisation and
        the stare realisation. They are alternative samples of the same emitter,
        never two emitters (D24), which is why `sample` picks one per emitter (D32).
        """
        if self._by_emitter is None:
            index: dict[tuple[str, int], list[EmitterContribution]] = {}
            for c in self.contributions:
                index.setdefault((c.config_id, c.label), []).append(c)
            self._by_emitter = index
        return self._by_emitter

    @property
    def emitter_keys(self) -> list[tuple[str, int]]:
        """Sorted `(config_id, label)` keys -- the population `sample` draws over.

        Sorted so a seeded draw is reproducible across runs: dict insertion order
        follows the order recordings happened to be read.
        """
        return sorted(self.by_emitter)

    @property
    def n_distinct_emitters(self) -> int:
        """Emitters seen in at least one run (a scan and a stare sighting of the
        same transmitter are two contributions but one emitter)."""
        return len(self.by_emitter)


@dataclass
class Scenario:
    """A set of emitter contributions, and the world they make.

    Two ways in. `replay` is one real recording, unmodified -- that is what the
    validation gates run on. `sample` draws from the pool -- that is what RL trains
    on, so no episode repeats and there is nothing to memorise (D25).
    """

    name: str
    contributions: list[EmitterContribution]
    seed: int | None = None

    @classmethod
    def replay(
        cls,
        config_id: str,
        source: str,
        split: str = TRAIN_SPLIT,
        *,
        allow_heldout: bool = False,
        reason: str = "",
    ) -> "Scenario":
        contribs = load_contributions(
            config_id, source, split, allow_heldout=allow_heldout, reason=reason
        )
        return cls(name=f"replay:{split}/{source}/{config_id}", contributions=contribs)

    @classmethod
    def sample(
        cls,
        pool: EmitterPool,
        rng: np.random.Generator,
        n_emitters: int | None = None,
    ) -> "Scenario":
        """Draw a scenario from the pool.

        `n_emitters` defaults to a draw from the per-config count of emitters that
        were *detectable at all* (1 to 82 across the 47 train configs, summing to
        1,913), so sampled scenarios keep the difficulty spread the dataset
        actually has rather than a flat one. Not the metadata transmitter count
        (2 to 99): 19.0% of transmitters never appear in either recording.

        One emitter appears at most once (D32). 1,530 of the 1,913 pool emitters
        have both a scan and a stare realisation, so a uniform draw over
        contributions returned the same physical emitter twice in 23.9% of
        scenarios -- at identical position and beam phase, with disjoint activity.
        That is not the plausible independent placement D25 justifies sampling
        with, and it double-counts one emitter in `E`. So the draw is over
        emitters, and one realisation is chosen per emitter drawn.
        """
        if n_emitters is None:
            n_emitters = int(rng.choice(pool.emitter_counts))
        keys = pool.emitter_keys
        n_emitters = min(n_emitters, len(keys))
        idx = rng.choice(len(keys), size=n_emitters, replace=False)
        picked = [
            pool.by_emitter[keys[i]][rng.integers(len(pool.by_emitter[keys[i]]))]
            for i in idx
        ]
        return cls(name=f"sample:n={n_emitters}", contributions=picked)

    def __len__(self) -> int:
        return len(self.contributions)


# --------------------------------------------------------------------------- #

def _build_cache_cli() -> None:
    import time

    t0 = time.time()
    n = 0
    for source in SOURCES:
        for config_id in list_configs(source, TRAIN_SPLIT):
            got = load_contributions(config_id, source, TRAIN_SPLIT, rebuild=True)
            n += 1
            print(f"  {source:5s} {config_id:14s} {len(got):3d} emitters", flush=True)
    print(f"\ncached {n} recordings in {time.time() - t0:.1f} s -> {CACHE_ROOT}")


if __name__ == "__main__":
    import sys

    if "--build-cache" in sys.argv:
        _build_cache_cli()
    else:
        print(__doc__)
