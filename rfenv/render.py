"""The waterfall, and the two plots that go beside it.

Artefact 3 of `EVALUATION.md` §8, plus the receiver's ROC and the per-band bar
gate 2 reads. The waterfall is the point of the module and it exists for one
reason: **the same picture can be drawn from the raw Turing recording**, so the
environment and the data are compared by eye with no interpretation in between.
Nothing else in this project has that property -- every other check runs through
a metric definition that could itself be wrong.

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

matplotlib is a dependency of this module alone. `metrics.py`, `env.py` and the
layers below it never import it, so the environment and its tests run without a
plotting stack installed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rfenv.constants import (  # noqa: E402
    BAND_CENTRES_MHZ,
    EPISODE_S,
    GAMMA_DBM,
    N_BANDS,
    NOISE_FLOOR_DBM,
)

DPI = 150

# The waterfall colour scale, fixed rather than per-grid, so any two waterfalls
# can be laid side by side and believed -- which is the artefact's whole job.
#
# The bottom is the noise floor: an empty cell sits exactly at N0 and is masked
# out, so nothing is ever drawn there. The top is a clip, not a maximum. Measured
# this session over all 47 train configs, both runs, per-emitter peak level: the
# median emitter peaks at -75.8 dB (scan) / -58.3 dB (stare) and the 95th
# percentile at -30.4 / -33.9, while the loudest single emitter in the whole train
# set reaches +6.40 dB. Scaling to that maximum pushes almost every emitter into
# the bottom fifth of the colourmap and the antenna-pattern structure disappears.
# So the handful of cells above -20 dB saturate, and the 100 dB where the data
# actually lives gets the whole scale.
LEVEL_VMIN_DB = NOISE_FLOOR_DBM
LEVEL_VMAX_DB = -20.0


def _save(fig, path: str | Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def _band_axis(ax) -> None:
    """Label the y axis by frequency, because that is what an operator reads.

    Band index is linear in frequency -- centres are 250 MHz + 500 MHz per band --
    so the ticks can carry GHz directly with no distortion.
    """
    ticks = np.arange(0, N_BANDS, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{BAND_CENTRES_MHZ[b] / 1000:.2f}" for b in ticks])
    ax.set_ylabel("band centre (GHz)")


# --------------------------------------------------------------------------- #
# The waterfall
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Receiver characterisation
# --------------------------------------------------------------------------- #

def roc(
    curve: dict,
    path: str | Path | None = None,
    *,
    operating: dict | None = None,
    title: str = "Receiver ROC",
    figsize: tuple[float, float] = (5.5, 5.0),
):
    """Pd against Pfa over a gamma sweep -- the receiver deliverable (D15).

    `curve` is what `receiver.roc()` returns; `operating` is what
    `receiver.operating_point()` returns, and marks the frozen point on it.

    **The population is written on the figure**, as D33 requires. Pd depends on it
    entirely -- 0.819, 0.837 and 0.851 at the same gamma over three different
    populations -- and the previously quoted 0.822 was withdrawn precisely because
    a plot and a number had gone out into the world without one.

    Pfa is on a log axis: it runs from about 1 down to the Gaussian tail as gamma
    rises, and on a linear axis the entire useful part of the curve collapses onto
    the left edge.
    """
    pfa = np.asarray(curve["pfa"], dtype=np.float64)
    pd = np.asarray(curve["pd"], dtype=np.float64)
    floor = 1e-12  # log axis cannot show an exactly-zero tail

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.maximum(pfa, floor), pd, "-", color="#0a84ff", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("P_fa")
    ax.set_ylabel("P_d")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, which="both")

    if operating is not None:
        ax.plot(max(operating["pfa"], floor), operating["pd"], "o", color="#ff3b30", zorder=5)
        ax.annotate(
            f"γ = {operating['gamma_dbm']:g} dB\n"
            f"P_d = {operating['pd']:.4f}\n"
            f"P_fa = {operating['pfa']:.3e}\n"
            f"sensitivity {operating['sensitivity_dbm']:.2f} dB"
            f" (at P_d = {operating['pd_target']:g})",
            xy=(max(operating["pfa"], floor), operating["pd"]),
            xytext=(12, -46), textcoords="offset points", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", alpha=0.9),
        )

    ax.set_title(
        f"{title}\npopulation: {curve['population']} "
        f"({curve['n_occupied']:,} occupied / {curve['n_empty']:,} empty cells)",
        fontsize=10,
    )
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Per-band structure (gate 2)
# --------------------------------------------------------------------------- #

def band_profile(
    series: dict[str, np.ndarray],
    path: str | Path | None = None,
    *,
    title: str = "Per-band structure",
    ylabel: str = "fraction",
    figsize: tuple[float, float] = (13.0, 3.6),
):
    """Grouped bars over the 36 bands -- gate 2's "not just the aggregate" check.

    `series` maps a label to a length-36 array, so it draws predicted against
    recorded, or any pair worth comparing band by band. Deliberately unopinionated
    about what the arrays mean: gate 2 owns that definition, this draws it.

    Band 0 is expected to be the visible failure when stare-built predictions are
    compared against scan recordings -- stare cannot see below 500 MHz, so the
    band is 59.12% occupied in the recordings and 0.00% predicted (D10, D24). It
    is a stated limitation, and the plot should show it rather than hide it.
    """
    labels = list(series)
    n = len(labels)
    x = np.arange(N_BANDS)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, label in enumerate(labels):
        ax.bar(x + (i - (n - 1) / 2) * width, np.asarray(series[label], dtype=np.float64),
               width=width, label=label)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{BAND_CENTRES_MHZ[b] / 1000:.1f}" for b in x[::2]], fontsize=8)
    ax.set_xlabel("band centre (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, N_BANDS)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    return _save(fig, path)


# --------------------------------------------------------------------------- #

def _cli() -> None:
    """`python -m rfenv.render config_2 stare out.png` -- eyeball one scenario."""
    import sys

    from rfenv.scenario import Scenario
    from rfenv.truth import TruthGrid

    args = sys.argv[1:]
    if len(args) != 3:
        print(__doc__)
        print("usage: python -m rfenv.render <config_id> <scan|stare> <out.png>")
        raise SystemExit(2)

    config_id, source, out = args
    grid = TruthGrid.from_scenario(Scenario.replay(config_id, source))
    waterfall(grid, path=out)
    print(f"{out}  <-  {config_id} {source}: {grid.summary()}")


if __name__ == "__main__":
    _cli()
