"""The full-fidelity live view: the schedule panel, redrawn as it happens.

Separate from `rfenv/live.py` because importing `rfenv.render` sets the Agg
backend for the whole process (`render/__init__.py`), which is right for writing
files and wrong for opening a window -- so this module is imported lazily, by
`rfenv.live.make_live_view("full")`, and switches the backend itself.

The same picture `compare_animation` produces, live: band against time, the
schedule path over it, declared hits marked. It is built the way that function
builds it -- artists once, `set_data` per frame -- and **not** the way
`episode.env_frame` does, which constructs a whole figure per call and is far too
slow to sit inside a step loop.
"""

from __future__ import annotations

import sys

import numpy as np

from rfenv.constants import N_BANDS, SLOT_S
from rfenv.live import HIT, _RateLimited, _headline, _tail_state

_INTERACTIVE_BACKENDS = ("QtAgg", "TkAgg", "MacOSX", "Qt5Agg")


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
                 figsize: tuple[float, float] = (11.0, 5.0)):
        super().__init__(min_interval_s=min_interval_s)
        self.window_slots = int(window_slots)
        self.backend = _select_backend()

        import matplotlib.pyplot as plt

        self._plt = plt
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("band")
        self.ax.set_ylim(-0.5, N_BANDS - 0.5)

        # Artists built once. `imshow` carries the looked/hit strip; the step
        # line is the schedule itself; the scatter marks declarations.
        self._im = self.ax.imshow(
            np.zeros((N_BANDS, self.window_slots), dtype=np.uint8),
            aspect="auto", origin="lower", interpolation="nearest",
            vmin=0, vmax=2, cmap="magma",
            extent=(0.0, self.window_slots * SLOT_S, -0.5, N_BANDS - 0.5),
        )
        (self._path,) = self.ax.step(
            [], [], where="post", color="white", linewidth=1.0, alpha=0.9
        )
        self._hits = self.ax.scatter([], [], s=10, color="#ff3b30", zorder=4)
        self._title = self.ax.set_title("", fontsize=9, family="monospace", loc="left")

        plt.ion()
        self.fig.tight_layout()
        self.fig.show()

    def open(self, env) -> None:
        self.update(env, force=True)

    def update(self, env, *, force: bool = False) -> None:
        if not self._due(force=force):
            return
        if not self._plt.fignum_exists(self.fig.number):
            return                                  # the window was closed; keep running

        grid, start = _tail_state(env, self.window_slots)
        self._im.set_data(grid)
        t0, t1 = start * SLOT_S, (start + self.window_slots) * SLOT_S
        self._im.set_extent((t0, t1, -0.5, N_BANDS - 0.5))
        self.ax.set_xlim(t0, t1)

        slots = np.arange(start, start + self.window_slots)
        looked = grid.argmax(axis=0)
        any_looked = grid.any(axis=0)
        self._path.set_data(slots[any_looked] * SLOT_S, looked[any_looked])

        hit_band, hit_slot = np.nonzero(grid == HIT)
        self._hits.set_offsets(
            np.column_stack([(hit_slot + start) * SLOT_S, hit_band])
            if hit_band.size else np.empty((0, 2))
        )

        self._title.set_text(_headline(env))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:                            # pragma: no cover
            print("live view: failed to close the figure", file=sys.stderr)
