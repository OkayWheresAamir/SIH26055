"""Comparing schedulers (`EVALUATION.md` §5).

Four pictures, and each answers a question the §4 table can state but not
show. The timeline shows *how* the 30 s was spent; the discovery curve shows
*when* coverage was earned; the Pareto plot shows that no rung is good at both
objectives, which is the whole finding of D14; `compare_animation` is the
timeline in motion. All four take the artefacts, so they draw the same numbers
the scorecard is computed from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfenv.constants import BAND_CENTRES_MHZ, EPISODE_S, N_BANDS, SLOT_S
from rfenv.render._common import _band_axis, _save

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


def compare_animation(
    runs,
    grid,
    path: str | Path | None = None,
    *,
    stride: int = 4,
    fps: int = 15,
    figsize_per_row: float = 2.4,
    dpi: int = 100,
    band_priority: np.ndarray | None = None,
):
    """`schedule_timeline`'s rows, in motion -- the path and hits revealed slot
    by slot instead of all 600 at once.

    One deliberate difference from `schedule_timeline`: **no metrics text box**.
    `ratio`/`cTTI`/`coverage` summarise a *finished* episode; showing it while
    the episode is still unfolding is backwards. Read the final numbers off
    `EVALUATION.md`-style artefacts instead.

    The y-axis labels every 5th band (`_band_axis`), same as `schedule_timeline`
    and `waterfall` -- with several rows stacked at `figsize_per_row` height each,
    every-band labels (`_full_band_axis`, used by the single-row `env_frame`
    instead) crowd out at this scale. `env_frame` has a full-height row to
    itself and keeps the finer labelling.

    `runs` is `{label: RunArtefacts}`, same convention as `schedule_timeline`;
    every run must share `grid`, the scenario and the seed, or the rows are not
    comparable. Saved as a GIF (Pillow ships with matplotlib -- no ffmpeg
    dependency). Figures/artists are built once and only their data is updated
    per frame, not rebuilt -- 600 slots at `stride=1` would be prohibitively
    slow otherwise.

    `band_priority` (this task, "v2p"): the episode's `(N_BANDS,)` priority
    array, `1.0` ordinary / `3.0` elevated, sampled once per episode and fixed
    for its whole duration (`ScanEnv.reset()`) -- so unlike the path and hits,
    the highlight it draws is static, one translucent stripe per elevated band,
    drawn once behind every row rather than updated per frame. `None` (the
    default) draws nothing extra, unchanged from before this parameter existed.
    """
    from matplotlib.animation import PillowWriter

    from rfenv import metrics as M
    from rfenv.constants import N_SLOTS

    labels = list(runs)
    n = len(labels)
    fig, axes = plt.subplots(
        n, 1, figsize=(13.0, max(figsize_per_row * n, 3.0)), squeeze=False, sharex=True
    )
    Z = np.asarray(grid.Z, dtype=bool)
    series = {label: M.schedule_series(runs[label]) for label in labels}
    priority_bands = (
        np.flatnonzero(np.asarray(band_priority) > 1.5) if band_priority is not None else ()
    )

    dynamic = {}
    for row, label in enumerate(labels):
        ax = axes[row][0]
        ax.imshow(
            np.ma.masked_where(~Z, np.ones_like(Z, dtype=float)),
            origin="lower", aspect="auto", interpolation="nearest",
            extent=(0.0, EPISODE_S, -0.5, N_BANDS - 0.5),
            cmap="Greys_r", vmin=0.0, vmax=1.6, alpha=0.55,
        )
        for b in priority_bands:
            ax.axhspan(b - 0.5, b + 0.5, color="#ffd60a", alpha=0.22, zorder=1, linewidth=0)
        ax.set_facecolor("#101010")
        ax.set_ylim(-0.5, N_BANDS - 0.5)
        _band_axis(ax)
        ax.text(
            0.01, 0.97, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"),
        )
        if row == 0 and len(priority_bands):
            ax.text(
                0.99, 0.97,
                f"priority band{'s' if len(priority_bands) != 1 else ''}: "
                f"{', '.join(str(b) for b in priority_bands)}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                color="#ffd60a", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"),
            )
        (path_line,) = ax.step([], [], where="post", color=PATH_COLOUR, linewidth=1.0, alpha=0.9)
        hit_scatter = ax.scatter([], [], s=10, color=HIT_COLOUR, zorder=4)
        (now_marker,) = ax.plot([], [], "o", color="white", markersize=5.5, zorder=5,
                                 markeredgecolor="black", markeredgewidth=0.7)
        dynamic[label] = (path_line, hit_scatter, now_marker)

    axes[-1][0].set_xlabel("time (s)")
    axes[-1][0].set_xlim(0.0, EPISODE_S)
    title = fig.suptitle("t = 0.00 s", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    cutoffs = list(range(stride, N_SLOTS, stride)) + [N_SLOTS]

    with writer.saving(fig, path, dpi=dpi):
        for cutoff in cutoffs:
            title.set_text(f"t = {cutoff * SLOT_S:.2f} s")
            for label in labels:
                s = series[label]
                mask = s["slot"] < cutoff
                t = s["time_s"][mask]
                band = s["band"][mask].astype(float)
                path_line, hit_scatter, now_marker = dynamic[label]
                path_line.set_data(t, band)
                hit_mask = mask & s["hit"]
                if hit_mask.any():
                    hit_scatter.set_offsets(
                        np.column_stack([s["time_s"][hit_mask] + SLOT_S / 2, s["band"][hit_mask]])
                    )
                else:
                    hit_scatter.set_offsets(np.empty((0, 2)))
                if len(t):
                    now_marker.set_data([t[-1]], [band[-1]])
            writer.grab_frame()

    plt.close(fig)
    return path


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


def _iqr_arm(row: dict, metric: str, centre: float):
    """`[[lower], [upper]]` arm lengths for one axis, or None if not supplied.

    matplotlib wants distances from the point, not absolute positions, and it
    rejects negative arms. A p25 above the centre (or p75 below it) can only mean
    the caller mixed statistics -- passing means with somebody else's percentiles
    is exactly how that happens -- so clamp at zero rather than raise: a figure
    with one degenerate whisker is more useful than no figure.
    """
    lo, hi = row.get(f"{metric}_p25"), row.get(f"{metric}_p75")
    if lo is None or hi is None:
        return None
    return [[max(0.0, centre - float(lo))], [max(0.0, float(hi) - centre)]]


def pareto(
    points,
    path: str | Path | None = None,
    *,
    title: str = "The two PS objectives, against each other",
    figsize: tuple[float, float] = (9.0, 5.6),
):
    """Interception ratio against censored intercept time, one point per scheduler.

    **This is D14's finding as a picture** and the reason §4 forbids publishing
    either axis alone: the camper sits far right (captures the pulses, finds
    nobody) and round-robin bottom-left (finds everybody, captures nothing), and
    no trivial strategy is in the top-left corner. Up and to the left is better,
    so the target for an adaptive scheduler is stated by the geometry rather than
    by a sentence.

    **Points are medians and the whiskers are the interquartile range (D58).**
    A single marker cannot show a distribution, so the statistic it collapses to
    must be one a minority of episodes cannot move -- and drawn as means this
    figure said something false. The camper's interception ratio has mean 0.2065
    against median 0.1257 over 47 stare replays, because it either lands on a
    busy band and scores 0.32 or a quiet one and scores 0.05 (IQR
    [0.0538, 0.3228], std 0.19 against every adaptive rung's 0.04). Since the
    axis limits come from the maximum, that inflated mean also stretched the
    y-axis and pushed every other rung into the lower half of the plot: rung 9
    read as "worse on ratio than the camper" when its median is level with it and
    it wins the paired per-episode count on intercept time nearly 100% of the
    time.

    The whiskers are the honest part. A median point alone has the same failure
    mode as a mean point -- one dot, no spread -- and the camper's arm being four
    times longer than anyone else's is the single most informative mark on the
    figure. They are optional: a caller passing only the three headline metrics
    gets bare points, which is what the tests do.

    Coverage is drawn as marker area, because §4's rule is that it is always
    printed beside the two headline metrics -- here it is impossible to read one
    without it, and it is repeated in the legend so the figure is still readable
    in grayscale.

    **Names go in the legend, not on the markers.** The interesting rungs cluster
    in the bottom-left corner -- five of them inside one second and 0.08 of ratio
    -- and annotating each in place makes that corner illegible, which is the one
    part of the plot a reader has come for. Markers carry no text at all --
    plain coloured dots, identified only through the legend (rung number and
    label together).

    `points` is `{key: {interception_ratio, censored_mean_intercept_time_s,
    emitter_coverage, rung, reference, label}}`, normally the per-scheduler
    aggregate means. `rung`, `reference` and `label` are optional; without
    `rung` the marker is numbered by position, and without `label` the outer
    dict key is shown as-is. **`key` must be unique per rung -- `label` need
    not be**: two rungs sharing a display label (e.g. two PPO variants both
    called "PPO scheduler") is legitimate and must still draw two points, which
    is why the legend text comes from `row["label"]` rather than from the dict
    key itself -- a caller that keyed this dict by label directly would silently
    collapse same-labelled rungs to one point before this function ever saw them.

    Reference lines -- the oracles -- are drawn hollow. They are not competitors
    (§5) and a filled marker would invite reading them as one.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for i, (key, row) in enumerate(points.items(), 1):
        label = row.get("label", key)
        reference = bool(row.get("reference", False))
        coverage = float(row.get("emitter_coverage", 0.0))
        rung = str(row.get("rung", i))
        x = row["censored_mean_intercept_time_s"]
        y = row["interception_ratio"]

        # Interquartile whiskers, drawn first so the marker sits on top of them.
        # Optional: a caller passing only the three headline metrics still gets a
        # bare point, which is what the tests and any ad-hoc caller do.
        xerr = _iqr_arm(row, "censored_mean_intercept_time_s", x)
        yerr = _iqr_arm(row, "interception_ratio", y)
        if xerr is not None or yerr is not None:
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                fmt="none", ecolor="#888", elinewidth=0.9,
                capsize=2.5, capthick=0.9, alpha=0.55, zorder=2,
            )

        ax.scatter(
            x, y, s=60 + 340 * coverage,
            facecolors="none" if reference else None,
            edgecolors="#444" if reference else "none",
            alpha=0.85, zorder=3,
            label=f"{rung:>2}  {label}   (cov {coverage:.2f})",
        )

    ax.set_xlabel("censored mean intercept time (s)   ->  worse")
    ax.set_ylabel("interception ratio   ->  better")
    # Headroom, and room for the legend outside the axes. Taken from the whisker
    # tips where there are whiskers, so a wide rung is not clipped out of its own
    # interval -- the camper's p75 is what needs the room, and it is exactly the
    # rung whose spread the reader has to see.
    xs = [r.get("censored_mean_intercept_time_s_p75",
                r["censored_mean_intercept_time_s"]) for r in points.values()] or [1.0]
    ys = [r.get("interception_ratio_p75",
                r["interception_ratio"]) for r in points.values()] or [1.0]
    ax.set_xlim(0.0, max(xs) * 1.12 + 0.5)
    ax.set_ylim(0.0, max(ys) * 1.12 + 0.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
              frameon=False, labelspacing=0.9, borderpad=0.0,
              title="rung (hollow = reference line)", title_fontsize=8)
    ax.set_title(
        f"{title}\n"
        "median over the run, whiskers are the interquartile range\n"
        "better is up and to the left; marker area is coverage",
        fontsize=9)
    return _save(fig, path)
