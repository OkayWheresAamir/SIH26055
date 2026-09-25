"""The live views (D80).

Pinned here, in rough order of how much damage the failure would do:
(1) `rfenv.live` never imports matplotlib -- the whole point of the light view;
(2) importing `rfenv.render` leaves the process on Agg, so a live window can
never leak an interactive backend into a test run or a headless training job;
(3) writing to something that is not a terminal emits **no escape bytes at all**,
because that stream is usually a log file and control codes in one are
corruption; (4) the refresh throttle actually throttles; (5) a missing plotting
backend degrades to the terminal instead of killing the run.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pytest

from rfenv.constants import N_BANDS
from rfenv.env import ScanEnv
from rfenv.live import (
    CORRECT_SILENCE,
    FALSE_ALARM,
    MISS,
    TRUE_HIT,
    UNSEEN_EMPTY,
    UNSEEN_TRUTH,
    AnsiHeatStrip,
    NullView,
    _tail_state,
    make_live_view,
)
from rfenv.scenario import Scenario


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _run(env, steps=40, view=None):
    for k in range(steps):
        env.step((7 * k) % 36)
        if view is not None:
            view.update(env)
    return env


def _env(**kwargs):
    env = ScanEnv(scenario=Scenario.replay("config_2", "stare"), **kwargs)
    env.reset(seed=0)
    return env


# --------------------------------------------------------------------------- #
# The dependency boundary
# --------------------------------------------------------------------------- #

def test_the_light_view_module_does_not_import_matplotlib():
    """If this fails, the terminal view is no longer dependency-free.

    Run in a clean interpreter and checked against `sys.modules`, because that
    is the actual property -- a source scan would be fooled by a comment, and
    checking in-process would be fooled by any earlier test having imported it.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import rfenv.live; "
         "assert 'matplotlib' not in sys.modules, sorted(sys.modules)[:0] or 'leaked'; "
         "print('clean')"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_importing_the_render_package_leaves_the_process_on_agg():
    """`rfenv/render/__init__.py` forces Agg; the live backend must not undo it."""
    matplotlib = pytest.importorskip("matplotlib")
    import rfenv.render  # noqa: F401

    assert matplotlib.get_backend().lower() == "agg"


# --------------------------------------------------------------------------- #
# _tail_state
# --------------------------------------------------------------------------- #

def test_the_tail_strip_classifies_every_looked_cell_against_truth():
    """Y crossed with Z, not Y alone -- a hit can be a false alarm, a quiet
    cell can be a missed detection, and the strip has to say which."""
    env = _env()
    _run(env, 40)
    grid, start = _tail_state(env, 64)
    assert grid.shape == (N_BANDS, 64)
    assert set(np.unique(grid)).issubset(
        {UNSEEN_EMPTY, UNSEEN_TRUTH, CORRECT_SILENCE, TRUE_HIT, FALSE_ALARM, MISS}
    )
    for row in env.log:
        slot = int(row["slot"])
        if not (start <= slot < start + 64):
            continue
        y, z = bool(row["Y"]), bool(row["Z"])
        expected = (TRUE_HIT if z else FALSE_ALARM) if y else (MISS if z else CORRECT_SILENCE)
        assert grid[int(row["band"]), slot - start] == expected


def test_truth_is_visible_on_a_band_before_anything_ever_looked_at_it():
    """The point of D81's redesign: where the emitters are, not just what was
    declared. A band with a real emitter transmitting shows UNSEEN_TRUTH even
    on a slot nothing ever tuned to."""
    env = _env()
    _run(env, 30)
    grid, start = _tail_state(env, 64)
    looked_bands = {int(row["band"]) for row in env.log
                    if start <= int(row["slot"]) < start + 64}
    elapsed = int(env.t) - start
    for band in range(N_BANDS):
        if band in looked_bands:
            continue
        truth = env.grid.Z[band, start:start + elapsed]   # elapsed part only
        if truth.any():
            assert UNSEEN_TRUTH in grid[band, :elapsed]
            break
    else:
        pytest.skip("no unlooked band had a true emitter in this window")


def test_the_tail_strip_never_shows_truth_past_the_current_slot():
    """No look-ahead: truth beyond `env.t` would be a spoiler, not a picture
    of the mission so far."""
    env = _env()
    _run(env, 15)          # episode_slots default is 600; well short of it
    grid, start = _tail_state(env, 64)
    elapsed = int(env.t) - start
    assert np.all(grid[:, elapsed:] == UNSEEN_EMPTY)


def test_the_tail_strip_is_bounded_by_its_width_not_by_the_episode():
    env = _env()
    _run(env, 250)
    grid, start = _tail_state(env, 32)
    assert grid.shape == (N_BANDS, 32)
    assert start == max(0, int(env.log[-1]["slot"]) + 1 - 32)


def test_the_tail_strip_of_a_fresh_episode_is_empty():
    grid, start = _tail_state(_env(), 20)
    assert start == 0
    assert not grid.any()


# --------------------------------------------------------------------------- #
# The ANSI view
# --------------------------------------------------------------------------- #

def test_a_non_tty_stream_receives_no_escape_bytes():
    """A piped run's output is a log file. Escape codes in one are corruption."""
    buf = io.StringIO()
    view = AnsiHeatStrip(stream=buf, min_interval_s=0.0)
    view.open(_env())
    _run(_env(), 20, view)
    view.close()
    assert "\x1b" not in buf.getvalue()
    assert buf.getvalue().strip()          # but it did say something


def test_a_tty_frame_carries_exactly_one_row_per_band():
    buf = _FakeTTY()
    view = AnsiHeatStrip(stream=buf, min_interval_s=0.0, width=40)
    env = _env()
    view.open(env)
    _run(env, 30)
    view.update(env, force=True)
    frame = buf.getvalue().split("\x1b[H")[-1]
    # Row format is "{marker}{band:3d} ..." -- marker is " " or the current-band
    # ">", always exactly one column, so the band number always sits at [1:4].
    rows = [ln for ln in frame.split("\n") if ln[1:4].strip().isdigit()]
    assert len(rows) == N_BANDS


def test_the_legend_appears_before_the_band_rows_not_after():
    """It used to sit in the footer, after 36 rows of glyphs a viewer had
    already had to guess at -- moved to right under the headline instead."""
    from rfenv.live import _LEGEND

    buf = _FakeTTY()
    view = AnsiHeatStrip(stream=buf, min_interval_s=0.0, width=40)
    env = _env()
    view.open(env)
    _run(env, 10)
    view.update(env, force=True)
    frame = buf.getvalue().split("\x1b[H")[-1]
    legend_line = next(i for i, ln in enumerate(frame.split("\n")) if _LEGEND in ln)
    first_band_row = next(
        i for i, ln in enumerate(frame.split("\n")) if ln[1:4].strip().isdigit()
    )
    assert legend_line < first_band_row


def test_exactly_one_row_is_marked_as_the_current_band():
    buf = _FakeTTY()
    view = AnsiHeatStrip(stream=buf, min_interval_s=0.0, width=40)
    env = _env()
    view.open(env)
    _run(env, 15)
    view.update(env, force=True)
    frame = buf.getvalue().split("\x1b[H")[-1]
    rows = [ln for ln in frame.split("\n") if ln[1:4].strip().isdigit()]
    marked = [ln for ln in rows if ln[0] == ">"]
    assert len(marked) == 1
    marked_band = int(marked[0][1:4])
    assert marked_band == int(env.log[-1]["band"])


def test_the_refresh_throttle_collapses_a_burst_into_one_write():
    buf = io.StringIO()
    view = AnsiHeatStrip(stream=buf, min_interval_s=999.0)
    env = _env()
    view.open(env)                        # forced, so exactly one line
    for _ in range(100):
        view.update(env)
    assert len(buf.getvalue().splitlines()) == 1


def test_no_color_falls_back_to_ascii_glyphs(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    buf = _FakeTTY()
    view = AnsiHeatStrip(stream=buf, min_interval_s=0.0, width=24)
    env = _env()
    view.open(env)
    _run(env, 20)
    view.update(env, force=True)
    frame = buf.getvalue()
    assert not view.colour
    assert "48;5;" not in frame           # no colour sequences
    assert "unseen" in frame


def test_the_headline_reports_this_episodes_own_length():
    buf = io.StringIO()
    view = AnsiHeatStrip(stream=buf, min_interval_s=0.0)
    env = _env()
    view.open(env)
    assert f"/{env.episode_slots}" in buf.getvalue()


def test_the_running_tally_matches_the_log_exactly():
    """Incremental counting (`_tally_new_rows`), checked against a direct scan.

    Called even on throttled frames that never draw -- a `min_interval_s` long
    enough to skip every intermediate frame must not skip counting either, or
    a long mission's totals would silently drift from what actually happened.
    """
    buf = io.StringIO()
    view = AnsiHeatStrip(stream=buf, min_interval_s=999.0)   # throttles almost everything
    env = _env()
    view.open(env)
    _run(env, 60, view)
    view.update(env, force=True)

    expected = {TRUE_HIT: 0, FALSE_ALARM: 0, MISS: 0, CORRECT_SILENCE: 0}
    for row in env.log:
        y, z = bool(row["Y"]), bool(row["Z"])
        code = (TRUE_HIT if z else FALSE_ALARM) if y else (MISS if z else CORRECT_SILENCE)
        expected[code] += 1
    assert view.tally == expected


def test_a_coloured_frame_still_carries_the_glyph_not_just_a_colour():
    """The `good`-vs-`critical` pair measures 4.1 deuteranopia Delta E (below
    the skill's own 6.0 floor) -- checked directly, not assumed. Every cell's
    character must survive independently of colour, so the ASCII glyph for a
    given state has to appear in the coloured stream too, not just a coloured
    blank."""
    buf = _FakeTTY()
    view = AnsiHeatStrip(stream=buf, min_interval_s=0.0, width=50)
    view.colour = True   # exercise the rendering path directly -- pytest's
    # captured stdout has no real console handle, so `_enable_windows_ansi()`
    # (tested on its own elsewhere) cannot be relied on to succeed here.
    env = _env()
    view.open(env)
    _run(env, 40, view)
    view.update(env, force=True)
    frame = buf.getvalue().split("\x1b[H")[-1]
    # At minimum the always-present states -- unseen and correct-silence --
    # must show up as literal characters, not just as SGR background codes.
    assert "." in frame
    for row in env.log:
        if not row["Y"] and not row["Z"]:
            assert ":" in frame
            break


def test_the_legend_names_every_state():
    buf = _FakeTTY()
    view = AnsiHeatStrip(stream=buf, min_interval_s=0.0, width=30)
    env = _env()
    view.open(env)
    _run(env, 20, view)
    view.update(env, force=True)
    frame = buf.getvalue()
    for word in ("unseen", "quiet", "hit", "false alarm", "missed"):
        assert word in frame


# --------------------------------------------------------------------------- #
# The factory
# --------------------------------------------------------------------------- #

def test_the_default_view_draws_nothing():
    view = make_live_view("off")
    assert isinstance(view, NullView)
    env = _env()
    with view:
        view.open(env)
        _run(env, 5, view)


def test_light_selects_the_terminal_view():
    assert isinstance(make_live_view("light"), AnsiHeatStrip)


def test_full_falls_back_to_the_terminal_when_no_backend_can_be_opened(
    monkeypatch, capsys
):
    """A plotting backend must never be able to kill a one-hour mission."""
    monkeypatch.setitem(sys.modules, "rfenv.render.live", None)
    view = make_live_view("full")
    assert isinstance(view, AnsiHeatStrip)
    assert "falling back" in capsys.readouterr().err


def test_an_unknown_view_kind_is_refused():
    with pytest.raises(ValueError):
        make_live_view("holographic")


def test_every_view_satisfies_the_same_structural_contract():
    """`open`/`update`/`close`, plus the context-manager pair. No ABC by design."""
    for view in (NullView(), make_live_view("off"), make_live_view("light")):
        for name in ("open", "update", "close", "__enter__", "__exit__"):
            assert callable(getattr(view, name)), (type(view).__name__, name)
