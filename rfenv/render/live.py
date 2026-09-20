"""The full-fidelity live view: the schedule panel, redrawn as it happens.

Separate from `rfenv/live.py` because importing `rfenv.render` sets the Agg
backend for the whole process (`render/__init__.py`), which is right for writing
files and wrong for opening a window -- so this module is imported lazily, by
`rfenv.live.make_live_view("full")`, and switches the backend itself.

Truth background plus path plus declarations -- `waterfall`'s picture
(`render/episode.py`), live, at `env_frame`'s per-frame cost: artists built
once and only their data updated (`set_data`), not the fresh-figure-per-call
`env_frame` does, which is too slow to sit inside a step loop.

**Not the pre-D77 version's plain looked/hit strip.** That only showed the
receiver's own log, so a viewer could never see where an emitter really was
until the schedule happened to land on it, and "looked, heard nothing" and
"never looked" both looked identical to a genuine miss. Truth is materialised
for the whole mission before the first slot, and showing it is not new
information reaching the agent (D19/D20 govern the *observation*, not a human
render) -- `waterfall`/`compare_animation` already draw it, and this is that
same license.
"""

from __future__ import annotations

import sys

import numpy as np

from rfenv.constants import N_BANDS, SLOT_S
from rfenv.live import FALSE_ALARM, MISS, TRUE_HIT, _RateLimited, _headline

_INTERACTIVE_BACKENDS = ("QtAgg", "TkAgg", "MacOSX", "Qt5Agg")

# From the `dataviz` skill's validated status palette (`references/palette.md`):
# good/warning/critical. Checked, not assumed, against the `good`/`critical`
# pair specifically -- `node` was not available here, so the skill's own
# OKLab/Delta-E math was ported to Python and run directly: true hit vs miss
# measures only 4.1 Delta E under a simulated deuteranopia, under the 6.0
# floor. Marker *shape* carries the distinction here for exactly that reason
# -- circle / triangle / X -- never colour alone.
_GOOD, _WARNING, _CRITICAL = "#0ca30c", "#fab219", "#d03b3b"


def _select_backend():
    """Switch to the first interactive backend that actually imports.

    Raises if none does, which `make_live_view` catches and turns into the
    terminal view plus a warning.
    """
    import matplotlib

    for name in _INTERACTIVE_BACKENDS:
        try:
            matplotlib.use(name, force=True)
            import matplotlib.pyplot as plt  # noqa: F401

            return name
        except Exception:
            continue
    raise RuntimeError(
        f"no interactive matplotlib backend available (tried {', '.join(_INTERACTIVE_BACKENDS)})"
    )


class MatplotlibLiveView(_RateLimited):
    """A scrolling band-vs-time window over the running episode.

    `window_slots` bounds what is drawn, not what is kept: an hour-long mission
    is 72,000 slots and drawing all of them would get slower every frame, so the
    x-axis scrolls and the artists stay a fixed size.
    """

    def __init__(self, *, min_interval_s: float = 0.1, window_slots: int = 1200,
                 figsize: tuple[float, float] = (11.0, 5.5)):
        super().__init__(min_interval_s=min_interval_s)
        self.window_slots = int(window_slots)
        self.backend = _select_backend()

        import matplotlib.pyplot as plt

        from rfenv.render._common import LEVEL_VMAX_DB, LEVEL_VMIN_DB, _band_axis

        self._plt = plt
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlabel("time (s)")
        self.ax.set_facecolor("#101010")
        self.ax.set_ylim(-0.5, N_BANDS - 0.5)
        _band_axis(self.ax)

        # Truth background -- level where an emitter is transmitting, masked
        # (transparent) elsewhere, the same convention and colour scale
        # `waterfall()` uses so a screenshot of either reads the same way.
        # Visible for the *whole* window from the first frame: truth exists
        # for the mission up front, whether or not the schedule has reached
        # it yet -- this is "where the emitters are", live.
        self._truth = self.ax.imshow(
            np.zeros((N_BANDS, self.window_slots)),
            aspect="auto", origin="lower", interpolation="nearest",
            cmap="viridis", vmin=LEVEL_VMIN_DB, vmax=LEVEL_VMAX_DB,
            extent=(0.0, self.window_slots * SLOT_S, -0.5, N_BANDS - 0.5),
        )
        self.fig.colorbar(self._truth, ax=self.ax, pad=0.01, extend="max",
                          label="truth level where transmitting (dB)")

        (self._path,) = self.ax.step(
            [], [], where="post", color="white", linewidth=1.0, alpha=0.75,
            label="receiver tuning", zorder=3,
        )
        # Three separate marks, not one -- a false alarm and a miss are
        # opposite failures (declared-but-wrong vs silent-but-wrong) and a
        # viewer needs to tell them apart from a true hit at a glance. Shape
        # carries the distinction as much as colour does (see the module
        # docstring): circle / triangle / X.
        self._hits = self.ax.scatter(
            [], [], s=26, marker="o", color=_GOOD, edgecolors="none",
            zorder=5, label="true hit",
        )
        self._false_alarms = self.ax.scatter(
            [], [], s=26, marker="^", color=_WARNING, edgecolors="none",
            zorder=5, label="false alarm",
        )
        self._misses = self.ax.scatter(
            [], [], s=30, marker="x", color=_CRITICAL, linewidths=1.6,
            zorder=5, label="missed detection",
        )
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.85, ncols=2)
        self._title = self.ax.set_title("", fontsize=9, family="monospace", loc="left")

        plt.ion()
        self.fig.tight_layout()
        self.fig.show()

    def open(self, env) -> None:
        self.update(env, force=True)

    def update(self, env, *, force: bool = False) -> None:
        self._tally_new_rows(env)
        if not self._due(force=force):
            return
        if not self._plt.fignum_exists(self.fig.number):
            return                                  # the window was closed; keep running

        end = int(env.t)
        start = max(0, end - self.window_slots)
        t0, t1 = start * SLOT_S, (start + self.window_slots) * SLOT_S

        # Truth up to *now*, not the whole window: `env.grid` is materialised
        # for the whole mission up front and could technically be read past
        # `env.t`, but drawing it would be a look-ahead spoiler, not "where
        # the emitters are" -- same clip the ANSI view applies. Only the
        # early-mission window (before it has scrolled) has any padding to
        # fill; once `end - start == window_slots` there is none.
        S = np.asarray(env.grid.S[:, start:end], dtype=np.float64)
        Z = np.asarray(env.grid.Z[:, start:end], dtype=bool)
        level = np.ma.masked_where(~Z, S)
        if level.shape[1] < self.window_slots:
            pad = np.ma.masked_all((N_BANDS, self.window_slots - level.shape[1]))
            level = np.ma.concatenate([level, pad], axis=1)
        self._truth.set_data(level)
        self._truth.set_extent((t0, t1, -0.5, N_BANDS - 0.5))
        self.ax.set_xlim(t0, t1)

        # Path and declarations come from the log alone -- the receiver's own
        # record, same source `waterfall`/`env_frame` read.
        t_path, b_path, t_hit, b_hit, t_fa, b_fa, t_miss, b_miss = (
            [], [], [], [], [], [], [], []
        )
        for row in reversed(env.log):
            slot = int(row["slot"])
            if slot < start:
                break
            if slot >= end:
                continue
            t_path.append(slot * SLOT_S)
            b_path.append(row["band"])
            y, z = bool(row["Y"]), bool(row["Z"])
            if y and z:
                t_hit.append(slot * SLOT_S); b_hit.append(row["band"])
            elif y and not z:
                t_fa.append(slot * SLOT_S); b_fa.append(row["band"])
            elif not y and z:
                t_miss.append(slot * SLOT_S); b_miss.append(row["band"])

        order = np.argsort(t_path) if t_path else np.array([], dtype=int)
        self._path.set_data(np.asarray(t_path)[order], np.asarray(b_path)[order])
        self._hits.set_offsets(np.column_stack([t_hit, b_hit]) if t_hit else np.empty((0, 2)))
        self._false_alarms.set_offsets(np.column_stack([t_fa, b_fa]) if t_fa else np.empty((0, 2)))
        self._misses.set_offsets(np.column_stack([t_miss, b_miss]) if t_miss else np.empty((0, 2)))

        self._title.set_text(_headline(env, self.tally))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:                            # pragma: no cover
            print("live view: failed to close the figure", file=sys.stderr)
