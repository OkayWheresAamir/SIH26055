"""The live views (D76).

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
    HIT,
    LOOKED,
    UNSEEN,
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

def test_the_tail_strip_distinguishes_unseen_from_looked_from_hit():
    env = _env()
    _run(env, 40)
    grid, start = _tail_state(env, 64)
    assert grid.shape == (N_BANDS, 64)
    assert set(np.unique(grid)).issubset({UNSEEN, LOOKED, HIT})
    # Every logged slot in range is marked, and its band matches.
    for row in env.log:
        slot = int(row["slot"])
        if start <= slot < start + 64:
            assert grid[int(row["band"]), slot - start] == (HIT if row["Y"] else LOOKED)


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
    rows = [ln for ln in frame.split("\n") if ln[:3].strip().isdigit()]
    assert len(rows) == N_BANDS


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
