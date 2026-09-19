"""Watching an episode while it runs (D76).

Until now the only way to see a schedule was to finish the episode and render a
GIF. That is fine for a 30 s replay and useless for a mission that runs for an
hour, which is exactly what `episode_slots` makes possible -- so this module
draws the same picture as it happens.

**Two fidelities, and the light one is the point.** `AnsiHeatStrip` paints 36
band rows of coloured blocks straight to the terminal: no matplotlib, no window,
no display, works over SSH, and costs little enough to leave on during training.
`MatplotlibLiveView` (in `rfenv.render.live`, because importing `rfenv.render`
forces the Agg backend) is the full panel, for when you want to look properly.

**Why this is not `ScanEnv.render()`.** The environment knows nothing about any
of this. A view is handed the env by whichever loop is driving it and reads
`log`, `t` and `episode_slots`; `render_mode` and `render()` are untouched. The
default view is `NullView`, so the zero-cost path is the default path and nothing
that does not ask for a picture pays for one.

The contract is structural, like `rollout.run_episode`'s policy: any object with
`open(env)`, `update(env)`, `close()` works. No ABC, no Protocol.

    with make_live_view("light") as view:
        view.open(env)
        while not done:
            obs, r, done, _, info = env.step(action)
            view.update(env)

numpy and the standard library only. Nothing here may import matplotlib at
module level or the light view stops being light.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np

from rfenv.constants import N_BANDS, SLOT_S

# Cell states in the strip. "Looked and heard nothing" and "never looked" are
# different facts and the picture has to say which -- a blank column is a gap in
# coverage, a dark-blue one is a dwell that came back empty.
UNSEEN, LOOKED, HIT = 0, 1, 2

# xterm-256 indices. Chosen to survive both light and dark terminal themes, and
# to keep HIT unmistakable at a glance, since finding something is the event.
_BG = {UNSEEN: 236, LOOKED: 24, HIT: 196}
_ASCII = {UNSEEN: ".", LOOKED: ":", HIT: "#"}

_DEFAULT_MIN_INTERVAL_S = 0.1     # 10 Hz; a mission is 20 slots/s at real time


def _tail_state(env, width: int) -> tuple[np.ndarray, int]:
    """The last `width` slots of the episode as a (36, width) uint8 strip.

    Reads `env.log`, which is the same per-slot record `metrics/` serialises, so
    the live picture and the saved one cannot disagree about what happened.
    Walks backwards and stops early, so cost is bounded by `width` rather than by
    the length of a mission that may have been running for an hour.
    """
    grid = np.zeros((N_BANDS, width), dtype=np.uint8)
    log = env.log
    if not log:
        return grid, 0
    end = int(log[-1]["slot"]) + 1
    start = max(0, end - width)
    for row in reversed(log):
        slot = int(row["slot"])
        if slot < start:
            break
        grid[int(row["band"]), slot - start] = HIT if row["Y"] else LOOKED
    return grid, start


def _headline(env) -> str:
    """One line of state, shared by both backends."""
    detectable = len(getattr(env, "detectable", ()) or ())
    found = len(getattr(env, "tracks", ()) or ())
    coverage = found / detectable if detectable else float("nan")
    segment = int(env.t) // 600
    return (
        f"t {env.t * SLOT_S:8.2f}s  slot {int(env.t):6d}/{int(env.episode_slots)}"
        f"  seg {segment:3d}  found {found:4d}/{detectable:<4d}"
        f" ({coverage:5.1%})  reward {env.total_reward:10.1f}"
    )


class NullView:
    """The default: draws nothing, costs nothing.

    Exists so every driver can call `view.update(env)` unconditionally instead of
    branching on whether a view was asked for.
    """

    def open(self, env) -> None:
        pass

    def update(self, env) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _RateLimited(NullView):
    """Shared refresh throttle, so both backends inherit the same behaviour.

    A mission steps far faster than any display is worth updating -- and far
    faster than a human can read -- so `update()` is cheap to call every step and
    only actually draws on a clock.
    """

    def __init__(self, *, min_interval_s: float = _DEFAULT_MIN_INTERVAL_S):
        self.min_interval_s = float(min_interval_s)
        self._last = 0.0

    def _due(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last < self.min_interval_s:
            return False
        self._last = now
        return True


def _enable_windows_ansi() -> bool:
    """Turn on virtual-terminal processing, or report that we could not.

    Without this a Windows console prints the escape sequences literally, which
    is worse than not drawing at all -- hence the fallback to ASCII rather than a
    best-effort attempt.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class AnsiHeatStrip(_RateLimited):
    """36 band rows scrolling in the terminal. The light view.

    Degrades in three steps rather than failing: without a tty it prints a plain
    progress line and **not one escape byte** (the piped-to-a-log case, where
    control codes would be corruption); with `NO_COLOR` set, or on a console that
    refuses virtual-terminal mode, it draws ASCII glyphs instead of colour.
    """

    def __init__(self, *, min_interval_s: float = _DEFAULT_MIN_INTERVAL_S,
                 stream=None, width: int | None = None):
        super().__init__(min_interval_s=min_interval_s)
        self.stream = stream if stream is not None else sys.stdout
        self._forced_width = width
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.colour = (
            self.is_tty and not os.environ.get("NO_COLOR") and _enable_windows_ansi()
        )
        self._opened = False

    def _width(self) -> int:
        if self._forced_width is not None:
            return self._forced_width
        # 5 columns of band label, 2 of margin.
        return max(20, shutil.get_terminal_size((100, 40)).columns - 7)

    def open(self, env) -> None:
        self._opened = True
        if self.is_tty:
            self.stream.write("\x1b[2J")      # clear once; later frames repaint
        self.update(env, force=True)

    def update(self, env, *, force: bool = False) -> None:
        if not self._due(force=force):
            return
        if not self.is_tty:
            # No cursor movement, no colour: one appendable line.
            self.stream.write(_headline(env) + "\n")
            self.stream.flush()
            return

        width = self._width()
        grid, start = _tail_state(env, width)
        out = ["\x1b[H", _headline(env), "\x1b[K\n"]
        for band in range(N_BANDS):
            out.append(f"{band:3d} ")
            row = grid[band]
            if self.colour:
                # Runs of one colour are emitted as a single SGR sequence: a
                # 36x200 frame is ~7k cells and a per-cell escape would be
                # slower than the environment step it is drawing.
                prev = None
                for state in row:
                    if state != prev:
                        out.append(f"\x1b[48;5;{_BG[int(state)]}m")
                        prev = state
                    out.append(" ")
                out.append("\x1b[0m")
            else:
                out.append("".join(_ASCII[int(s)] for s in row))
            out.append("\x1b[K\n")
        out.append(
            f"    slots {start}..{start + width - 1}"
            f"   \x1b[48;5;{_BG[UNSEEN]}m \x1b[0m unseen"
            f"  \x1b[48;5;{_BG[LOOKED]}m \x1b[0m looked"
            f"  \x1b[48;5;{_BG[HIT]}m \x1b[0m hit\x1b[K\n"
            if self.colour else
            f"    slots {start}..{start + width - 1}   . unseen  : looked  # hit\n"
        )
        self.stream.write("".join(out))
        self.stream.flush()

    def close(self) -> None:
        if self._opened and self.is_tty:
            self.stream.write("\x1b[0m\n")
            self.stream.flush()
        self._opened = False


def make_live_view(kind: str = "off", **kwargs):
    """`"off"` | `"light"` | `"full"` -> a view object.

    `"full"` imports matplotlib, and therefore `rfenv.render`, only when asked
    for. If that import fails, or there is no display to open a window on, it
    warns and returns the ANSI view: a plotting backend must never be able to
    kill a one-hour run.
    """
    if kind in ("off", "none", None):
        return NullView()
    if kind == "light":
        return AnsiHeatStrip(**kwargs)
    if kind == "full":
        try:
            from rfenv.render.live import MatplotlibLiveView

            return MatplotlibLiveView(**kwargs)
        except Exception as exc:                      # pragma: no cover - env-dependent
            print(
                f"live view: falling back to the terminal strip ({exc})",
                file=sys.stderr,
            )
            return AnsiHeatStrip(
                **{k: v for k, v in kwargs.items() if k != "figsize"}
            )
    raise ValueError(f"live view must be 'off', 'light' or 'full', got {kind!r}")
