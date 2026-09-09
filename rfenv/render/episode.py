"""Single-episode / single-frame views: the waterfall, and its live sibling.

Artefact 3 of `EVALUATION.md` §8. The waterfall exists for one reason: **the
same picture can be drawn from the raw Turing recording**, so the environment
and the data are compared by eye with no interpretation in between. Nothing
else in this project has that property -- every other check runs through a
metric definition that could itself be wrong.

Drawing it from the recording is not a separate function. A scan replay's truth
grid *is* the recording, bucketed into cells:

    waterfall(TruthGrid.from_scenario(Scenario.replay("config_2", "scan")))

so the comparison is two calls with different arguments, which is what makes it
trustworthy. Be careful reading a scan replay for anything else, though: its
content sits where Turing's own sweep was tuned, enriched 14.5x at swept cells
against 1.23x for a stare replay, so a sweeping scheduler's path will trace the
bright cells and that means nothing about the scheduler (D36).

**What the heatmap shows.** Level where occupied, background where not. The
problem statement's occupancy is binary and §8 asks for "the occupancy grid as a
heatmap"; drawing the level instead loses nothing -- the coloured footprint *is*
the occupancy -- and gains the 50-odd dB of antenna-pattern structure that made
D4 reject a binary world model in the first place. The colour scale is fixed to
the noise floor rather than to each grid's own range, so two waterfalls can be
laid side by side and believed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfenv.constants import EPISODE_S, GAMMA_DBM, N_BANDS, SLOT_S
from rfenv.render._common import LEVEL_VMAX_DB, LEVEL_VMIN_DB, _band_axis, _full_band_axis, _save


def waterfall(
    grid,
    run=None,
    path: str | Path | None = None,
    *,
    title: str | None = None,
    gamma_dbm: float = GAMMA_DBM,
    show_path: bool = True,
    vmin_db: float = LEVEL_VMIN_DB,
    vmax_db: float = LEVEL_VMAX_DB,
    figsize: tuple[float, float] = (13.0, 5.0),
):
    """Frequency vs time, with the scheduler's path and its declared hits over it.

    `grid` is a `TruthGrid`; `run` is an optional `metrics.RunArtefacts` whose log
    supplies the overlay. Passing no run draws the world alone, which is the form
    used to compare the environment against the raw recording.

    The overlay is deliberately two marks and not three. The **path** is where the
    receiver was tuned -- one band per slot, so it is a complete account of how the
    30 s was spent. The **hits** are slots where it declared `Y = 1`, which is all
    a real receiver knows; truth is already the background, so drawing a
    true-positive marker as well would be drawing the answer key on top of the
    exam. Where a hit sits on a dark cell, that is a false alarm, and it should be
    legible as one.
    """
    S = np.asarray(grid.S, dtype=np.float64)
    Z = np.asarray(grid.Z, dtype=bool)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        np.ma.masked_where(~Z, S),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, EPISODE_S, -0.5, N_BANDS - 0.5),
        cmap="viridis",
        vmin=vmin_db,
        vmax=vmax_db,
    )
    ax.set_facecolor("#101010")
    fig.colorbar(image, ax=ax, pad=0.01, extend="max",
                 label="peak received level (dB)")

    if run is not None and show_path:
        t = np.array([row["slot"] for row in run.log], dtype=np.float64)
        b = np.array([row["band"] for row in run.log], dtype=np.float64)
        step = EPISODE_S / len(run.log)
        ax.step(t * step, b, where="post", color="white", linewidth=0.7, alpha=0.65,
                label="receiver tuning")
        y = np.array([row["Y"] for row in run.log], dtype=bool)
        if y.any():
            ax.scatter(t[y] * step + step / 2, b[y], s=9, color="#ff3b30",
                       edgecolors="none", label="declared hit (Y=1)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    ax.set_xlabel("time (s)")
    ax.set_xlim(0.0, EPISODE_S)
    _band_axis(ax)

    if title is None:
        title = getattr(getattr(grid, "scenario", None), "name", "truth grid")
        if run is not None:
            title = f"{run.header['scheduler']} — {title}"
    subtitle = f"γ = {gamma_dbm:g} dB"
    if run is not None:
        subtitle += f", seed {run.header['seed']}, reward {run.header['reward']}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    return _save(fig, path)


def env_frame(env, *, figsize: tuple[float, float] = (13.0, 5.0), dpi: int = 100) -> np.ndarray:
    """One `rgb_array` frame of a live `ScanEnv`: `ScanEnv.render()`'s backend.

    Truth background plus the tuning path and declared hits accumulated in
    `env.log` so far -- a live version of `waterfall()`, one call per frame
    rather than one call per finished episode. Every band is labelled
    (`_full_band_axis`): a single live frame has no neighbouring row's
    compactness to protect, and "which exact band" is exactly what a viewer
    following something live needs.

    A fresh figure per call, matching every other function here: this module
    is optimised to be simple and correct, not fast. `compare_animation`
    reuses figures across frames instead, because 600 calls to this function
    would be markedly slower.
    """
    Z = np.asarray(env.grid.Z, dtype=bool)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        np.ma.masked_where(~Z, np.ones_like(Z, dtype=float)),
        origin="lower", aspect="auto", interpolation="nearest",
        extent=(0.0, EPISODE_S, -0.5, N_BANDS - 0.5),
        cmap="Greys_r", vmin=0.0, vmax=1.6, alpha=0.55,
    )
    ax.set_facecolor("#101010")

    if env.log:
        t = np.array([row["slot"] for row in env.log], dtype=np.float64) * SLOT_S
        band = np.array([row["band"] for row in env.log], dtype=np.float64)
        ax.step(t, band, where="post", color="white", linewidth=1.0, alpha=0.9)
        hits = np.array([row["Y"] for row in env.log], dtype=bool)
        if hits.any():
            ax.scatter(t[hits] + SLOT_S / 2, band[hits], s=12, color="#ff3b30",
                       edgecolors="none", zorder=4)
        ax.plot(t[-1], band[-1], "o", color="white", markersize=6, zorder=5,
                markeredgecolor="black", markeredgewidth=0.8)

    ax.set_ylim(-0.5, N_BANDS - 0.5)
    ax.set_xlim(0.0, EPISODE_S)
    ax.set_xlabel("time (s)")
    _full_band_axis(ax)
    ax.set_title(f"t = {env.t * SLOT_S:.2f} s / {EPISODE_S:.0f} s", fontsize=10)
    fig.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
    plt.close(fig)
    return frame
