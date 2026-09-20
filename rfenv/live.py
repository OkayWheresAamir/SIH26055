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

# Six cell states -- what truth says (an emitter really transmitting there or
# not) crossed with what the receiver did about it (never looked, or looked
# and declared Y=0/Y=1). Collapsing "looked, no hit" and "never looked" into
# one grey, as the first version of this view did, hid the two facts a viewer
# actually wants live: **where the emitters are** (visible even before
# anything has looked, since the truth grid is materialised for the whole
# mission up front) and **whether a declaration was right** -- a false alarm
# and a missed detection are opposite failures and look nothing alike here.
UNSEEN_EMPTY = 0     # never looked, and truly nothing there
UNSEEN_TRUTH = 1     # never looked, but an emitter is transmitting here now
CORRECT_SILENCE = 2  # looked, correctly declared nothing (Y=0, Z=0)
TRUE_HIT = 3         # looked, correctly declared a hit (Y=1, Z=1)
FALSE_ALARM = 4      # looked, declared a hit where nothing was there (Y=1, Z=0)
MISS = 5             # looked straight at it and still missed it (Y=0, Z=1)

# Kept for callers built against the pre-truth-aware version.
UNSEEN, LOOKED, HIT = UNSEEN_EMPTY, CORRECT_SILENCE, TRUE_HIT

# xterm-256 indices, chosen from the validated status palette
# (`dataviz` skill, `references/palette.md`): good #0ca30c, warning #fab219,
# critical #d03b3b, sequential blue step 300 #6da7ec for the unvisited-truth
# ghost. **Not relied on alone**: `good` (true hit) and `critical` (miss) --
# the two states that matter most here -- measure only 4.1 OKLab Delta E under
# a deuteranopia simulation (computed with the skill's own validator math;
# `node` was not available in this environment, so the check was ported to
# Python rather than skipped), well under the 6.0 floor. Every state below
# therefore also gets its own glyph, in both the coloured and the ASCII
# fallback, so no distinction here depends on hue.
_BG = {
    UNSEEN_EMPTY: 236,      # near-black -- nothing to report
    UNSEEN_TRUTH: 111,      # light blue -- "something is here, unconfirmed"
    CORRECT_SILENCE: 24,    # dim blue -- looked, correctly quiet
    TRUE_HIT: 34,           # good, green
    FALSE_ALARM: 214,       # warning, amber
    MISS: 160,              # critical, red
}
_ASCII = {
    UNSEEN_EMPTY: ".",
    UNSEEN_TRUTH: "?",
    CORRECT_SILENCE: ":",
    TRUE_HIT: "#",
    FALSE_ALARM: "!",
    MISS: "X",
}
_LEGEND = (
    ". unseen  ? emitter here, unseen  : looked, quiet  "
    "# hit  ! false alarm  X missed"
)

_DEFAULT_MIN_INTERVAL_S = 0.1     # 10 Hz; a mission is 20 slots/s at real time


def _tail_state(env, width: int) -> tuple[np.ndarray, int]:
    """The last `width` slots of the episode as a (36, width) uint8 strip of
    the six states above.

    Two sources, combined: `env.grid.Z`, truth for the whole mission,
    materialised up front and independent of what has been scanned -- reading
    it is not new information reaching the agent (D19/D20 still govern the
    *observation*; this is a picture for a human, the same license
    `waterfall`/`compare_animation` already use); and `env.log`, the same
    per-slot record `metrics/` serialises, for what was actually declared.
    The log walk is bounded by `width`, not by how long the mission has been
    running.
    """
    end = int(env.t)
    start = max(0, end - width)
    truth = env.grid.Z[:, start:min(end, env.grid.n_slots)]
    w = truth.shape[1]
    state = np.where(truth, UNSEEN_TRUTH, UNSEEN_EMPTY).astype(np.uint8)
    if w < width:
        state = np.concatenate(
            [state, np.full((N_BANDS, width - w), UNSEEN_EMPTY, dtype=np.uint8)],
            axis=1,
        )
    for row in reversed(env.log):
        slot = int(row["slot"])
        if slot < start:
            break
        if slot >= end:
            continue
        y, z = bool(row["Y"]), bool(row["Z"])
        code = (TRUE_HIT if z else FALSE_ALARM) if y else (MISS if z else CORRECT_SILENCE)
        state[int(row["band"]), slot - start] = code
    return state, start


def _headline(env, tally: dict[int, int] | None = None) -> str:
    """One line of state, shared by both backends.

    `tally` (from `_RateLimited._tally_new_rows`) adds the mission's running
    true-hit/false-alarm/miss counts -- the summary a viewer wants alongside
    "found X/Y", since coverage alone does not say whether the misses on the
    way there were close calls or a band never even looked at properly.
    """
    detectable = len(getattr(env, "detectable", ()) or ())
    found = len(getattr(env, "tracks", ()) or ())
    coverage = found / detectable if detectable else float("nan")
    segment = int(env.t) // 600
    line = (
        f"t {env.t * SLOT_S:8.2f}s  slot {int(env.t):6d}/{int(env.episode_slots)}"
        f"  seg {segment:3d}  found {found:4d}/{detectable:<4d}"
        f" ({coverage:5.1%})  reward {env.total_reward:10.1f}"
    )
    if tally is not None:
        line += (
            f"   hits {tally.get(TRUE_HIT, 0):4d}"
            f"  false alarms {tally.get(FALSE_ALARM, 0):4d}"
            f"  misses {tally.get(MISS, 0):4d}"
        )
    return line


class NullView:
    """The default: draws nothing, costs nothing.

    Exists so every driver can call `view.update(env)` unconditionally instead of
    branching on whether a view was asked for.
    """

    def open(self, env) -> None:
        pass

    def update(self, env, *, force: bool = False) -> None:
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
        self.tally: dict[int, int] = {
            TRUE_HIT: 0, FALSE_ALARM: 0, MISS: 0, CORRECT_SILENCE: 0,
        }
        self._tallied_upto = 0

    def _due(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last < self.min_interval_s:
            return False
        self._last = now
        return True

    def _tally_new_rows(self, env) -> None:
        """Fold every log row since the last call into the running tally.

        Called every `update()`, before the throttle check -- counting has to
        see every row, even on a frame that ends up not being drawn, or a
        long mission's counts would silently undercount between redraws.
        Bounded by new rows only, same as `_tail_state`.

        If `log_window_slots` (D76) is set, old rows are dropped from the
        front of `env.log` once the window fills, which would make this
        index run past the end. Detected and reset rather than raising -- the
        tally then resumes from what remains, undercounting whatever was
        already tallied and has since aged out. A known, documented cost of
        the memory bound, not a correctness bug in the common (unbounded-log)
        case this view is mainly built for.
        """
        log = env.log
        if self._tallied_upto > len(log):
            self._tallied_upto = 0
        for row in log[self._tallied_upto:]:
            y, z = bool(row["Y"]), bool(row["Z"])
            code = (TRUE_HIT if z else FALSE_ALARM) if y else (MISS if z else CORRECT_SILENCE)
            self.tally[code] = self.tally.get(code, 0) + 1
        self._tallied_upto = len(log)


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
        self._tally_new_rows(env)
        if not self._due(force=force):
            return
        if not self.is_tty:
            # No cursor movement, no colour: one appendable line.
            self.stream.write(_headline(env, self.tally) + "\n")
            self.stream.flush()
            return

        width = self._width()
        grid, start = _tail_state(env, width)
        out = ["\x1b[H", _headline(env, self.tally), "\x1b[K\n"]
        for band in range(N_BANDS):
            out.append(f"{band:3d} ")
            row = grid[band]
            if self.colour:
                # A cell is one glyph, not one blank, coloured -- a run of the
                # same state is still one SGR sequence (a 36x200 frame is ~7k
                # cells; a per-cell escape would be slower than the env step
                # it draws), but the character itself now also carries the
                # state, so colour is never the only way to read a cell.
                prev = None
                for state in row:
                    state = int(state)
                    if state != prev:
                        out.append(f"\x1b[38;5;15m\x1b[48;5;{_BG[state]}m")
                        prev = state
                    out.append(_ASCII[state])
                out.append("\x1b[0m")
            else:
                out.append("".join(_ASCII[int(s)] for s in row))
            out.append("\x1b[K\n")
        out.append(f"    slots {start}..{start + width - 1}   {_LEGEND}\x1b[K\n")
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
