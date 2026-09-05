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
    SLOT_S,
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
# Comparing schedulers (EVALUATION.md §5)
# --------------------------------------------------------------------------- #
#
# Three pictures, and each answers a question the §4 table can state but not
# show. The timeline shows *how* the 30 s was spent; the discovery curve shows
# *when* coverage was earned; the Pareto plot shows that no rung is good at both
# objectives, which is the whole finding of D14. All three take the artefacts, so
# they draw the same numbers the scorecard is computed from.

HIT_COLOUR = "#ff3b30"
PATH_COLOUR = "white"


def schedule_timeline(
    runs,
    grid=None,
    path: str | Path | None = None,
    *,
    title: str | None = None,
    show_emitters: bool = True,
    figsize_per_row: float = 2.1,
):
    """One row per scheduler: band against time, with dwells, hits and intercepts.

    `runs` is `{label: RunArtefacts}`, drawn in the order given -- normally the
    ladder's own order, so a reader goes down the figure from random to the
    reference line. `grid` is the optional `TruthGrid` the runs share; passing it
    puts the same faint occupancy behind every row, which is what makes the rows
    comparable rather than merely adjacent.

    **Every row draws the same three things.** The white step line is where the
    receiver was tuned, one band per slot, so it accounts for the whole episode.
    The tick under it marks the start of each dwell, which is the only way a
    two-slot look on a wide band is visible as one decision instead of two -- and
    the count of ticks is the episode's step count, 300 to 600 (D35). Red dots are
    declared hits (`Y = 1`), all a real receiver knows; where one sits off the
    occupancy it is a false alarm and should read as one.

    **The right-hand strip is the emitter view**: one line per detectable emitter
    over its detectable interval (D27), turning solid at the slot it was first
    intercepted. Rows that end mostly faint found few emitters, whatever their
    interception ratio says -- the camper and round-robin are opposite pictures
    and that is §4's rule about never publishing one metric drawn instead of
    argued.
    """
    from rfenv import metrics as M

    labels = list(runs)
    n = len(labels)
    fig, axes = plt.subplots(
        n, 2 if show_emitters else 1,
        figsize=(15.0, max(figsize_per_row * n, 2.4)),
        squeeze=False,
        gridspec_kw={"width_ratios": [3, 1]} if show_emitters else None,
        sharex=True,
    )

    Z = None if grid is None else np.asarray(grid.Z, dtype=bool)
    for row, label in enumerate(labels):
        run = runs[label]
        ax = axes[row][0]
        if Z is not None:
            ax.imshow(
                np.ma.masked_where(~Z, np.ones_like(Z, dtype=float)),
                origin="lower", aspect="auto", interpolation="nearest",
                extent=(0.0, EPISODE_S, -0.5, N_BANDS - 0.5),
                cmap="Greys_r", vmin=0.0, vmax=1.6, alpha=0.55,
            )
        ax.set_facecolor("#101010")

        series = M.schedule_series(run)
        t = series["time_s"]
        ax.step(t, series["band"], where="post", color=PATH_COLOUR,
                linewidth=0.8, alpha=0.85)
        starts = series["dwell_start"]
        ax.plot(t[starts], series["band"][starts], "|", color="#7fd1ff",
                markersize=3.5, alpha=0.9)
        hit = series["hit"]
        if hit.any():
            ax.scatter(t[hit] + SLOT_S / 2, series["band"][hit], s=7,
                       color=HIT_COLOUR, edgecolors="none", zorder=4)

        metrics = M.scheduler_metrics(run)
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
        ax.set_ylim(-0.5, N_BANDS - 0.5)
        ax.set_yticks([0, N_BANDS // 2, N_BANDS - 1])
        ax.set_yticklabels(
            [f"{BAND_CENTRES_MHZ[b] / 1000:.0f}G" for b in (0, N_BANDS // 2, N_BANDS - 1)],
            fontsize=7,
        )
        ax.text(
            0.005, 0.94,
            f"ratio {metrics['interception_ratio']:.3f}   "
            f"cTTI {metrics['censored_mean_intercept_time_s']:.2f} s   "
            f"cov {metrics['emitter_coverage']:.3f}   "
            f"{metrics['n_steps']} looks",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="#000000", ec="none", alpha=0.55),
        )

        if show_emitters:
            _emitter_strip(axes[row][1], run)

    axes[-1][0].set_xlabel("time (s)")
    axes[-1][0].set_xlim(0.0, EPISODE_S)
    if show_emitters:
        axes[-1][1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title, fontsize=10)
        fig.subplots_adjust(top=0.94)
    fig.align_ylabels()
    return _save(fig, path)


def _emitter_strip(ax, run) -> None:
    """Detectable intervals, faint until first intercept and solid after (D27, D28)."""
    from rfenv import metrics as M

    rows = M.emitter_activity(run)
    ax.set_facecolor("#f6f6f6")
    for y, row in enumerate(rows):
        on, off = row["on_slot"] * SLOT_S, row["off_slot"] * SLOT_S
        ax.plot([on, off], [y, y], color="#c8c8c8", linewidth=1.1, solid_capstyle="butt")
        first = row["first_intercept_slot"]
        if first is not None:
            ax.plot([first * SLOT_S, off], [y, y], color="#0a84ff",
                    linewidth=1.1, solid_capstyle="butt")
            ax.plot([first * SLOT_S], [y], "o", color=HIT_COLOUR, markersize=2.2)
    found = sum(1 for r in rows if r["first_intercept_slot"] is not None)
    ax.set_ylim(-1, max(len(rows), 1))
    ax.set_yticks([])
    ax.text(0.98, 0.94, f"{found}/{len(rows)} emitters", transform=ax.transAxes,
            va="top", ha="right", fontsize=7, color="#333")


def discovery_curves(
    runs,
    path: str | Path | None = None,
    *,
    title: str = "Emitters found against time",
    figsize: tuple[float, float] = (8.0, 4.6),
):
    """Cumulative distinct emitters intercepted, one line per scheduler.

    Coverage is one number; this is the path it took, and it is where censored
    intercept time becomes visible -- a curve that rises early and flattens is a
    low mean, a curve that never leaves the floor is 30 s of censoring on every
    emitter it missed. `|E|`, the detectable set, is the dashed ceiling.

    `runs` is `{label: RunArtefacts}`; every run must be the same scenario and
    seed, or the ceiling means nothing. The caller owns that -- `compare.py`
    passes one scenario at a time.
    """
    from rfenv import metrics as M

    fig, ax = plt.subplots(figsize=figsize)
    ceiling = 0
    for label, run in runs.items():
        d = M.discovery(run)
        ax.step(d["time_s"], d["n_found"], where="post", linewidth=1.4, label=label)
        ceiling = max(ceiling, d["n_detectable"])
    if ceiling:
        ax.axhline(ceiling, color="#888", linestyle="--", linewidth=1.0)
        ax.text(EPISODE_S, ceiling, f"  |E| = {ceiling}", va="center", fontsize=8,
                color="#666")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("distinct emitters intercepted")
    ax.set_xlim(0.0, EPISODE_S)
    ax.set_ylim(0, ceiling * 1.12 if ceiling else 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=10)
    return _save(fig, path)


def pareto(
    points,
    path: str | Path | None = None,
    *,
    title: str = "The two PS objectives, against each other",
    figsize: tuple[float, float] = (9.0, 5.6),
):
    """Interception ratio against censored intercept time, one point per scheduler.

    **This is D14's finding as a picture** and the reason §4 forbids publishing
    either axis alone: the camper sits top-right (captures the pulses, finds
    nobody) and round-robin bottom-left (finds everybody, captures nothing), and
    no trivial strategy is in the top-left corner. Up and to the left is better,
    so the target for an adaptive scheduler is stated by the geometry rather than
    by a sentence.

    Coverage is drawn as marker area, because §4's rule is that it is always
    printed beside the two headline metrics -- here it is impossible to read one
    without it, and it is repeated in the legend so the figure is still readable
    in grayscale.

    **Names go in the legend, not on the markers.** The interesting rungs cluster
    in the bottom-left corner -- five of them inside one second and 0.08 of ratio
    -- and annotating each in place makes that corner illegible, which is the one
    part of the plot a reader has come for. Each marker carries only its rung
    number.

    `points` is `{label: {interception_ratio, censored_mean_intercept_time_s,
    emitter_coverage, rung, reference}}`, normally the per-scheduler aggregate
    means. `rung` and `reference` are optional; without `rung` the marker is
    numbered by position.

    Reference lines -- the oracles -- are drawn hollow. They are not competitors
    (§5) and a filled marker would invite reading them as one.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, row) in enumerate(points.items(), 1):
        reference = bool(row.get("reference", False))
        coverage = float(row.get("emitter_coverage", 0.0))
        rung = str(row.get("rung", i))
        x = row["censored_mean_intercept_time_s"]
        y = row["interception_ratio"]
        ax.scatter(
            x, y, s=60 + 340 * coverage,
            facecolors="none" if reference else None,
            edgecolors="#444" if reference else "none",
            alpha=0.85, zorder=3,
            label=f"{rung:>2}  {label}   (cov {coverage:.2f})",
        )
        ax.annotate(rung, (x, y), ha="center", va="center", fontsize=7,
                    color="#222" if reference else "white", zorder=4,
                    fontweight="bold")

    ax.set_xlabel("censored mean intercept time (s)   ->  worse")
    ax.set_ylabel("interception ratio   ->  better")
    # Headroom, and room for the legend outside the axes.
    xs = [r["censored_mean_intercept_time_s"] for r in points.values()] or [1.0]
    ys = [r["interception_ratio"] for r in points.values()] or [1.0]
    ax.set_xlim(0.0, max(xs) * 1.12 + 0.5)
    ax.set_ylim(0.0, max(ys) * 1.12 + 0.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
              frameon=False, labelspacing=0.9, borderpad=0.0,
              title="rung (hollow = reference line)", title_fontsize=8)
    ax.set_title(f"{title}\nbetter is up and to the left; marker area is coverage",
                 fontsize=10)
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
