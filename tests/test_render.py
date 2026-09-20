"""Waterfall checks: does the picture show what the artefacts say?

A plot is the one artefact nobody diffs, so these assert on the figure's own data
rather than on the file existing. The two that matter are that the overlaid path
is the log's band column verbatim -- a waterfall drawing a schedule the scheduler
did not fly is worse than no waterfall -- and that the colour scale is fixed, so
the environment and the raw recording can be laid side by side, which is the only
reason this artefact exists (`EVALUATION.md` §8 artefact 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from rfenv import constants as K
from rfenv import metrics as M
from rfenv import render as R
from rfenv.env import ScanEnv
from rfenv.receiver import operating_point, roc
from rfenv.rollout import run_episode
from rfenv.scenario import Scenario
from rfenv.truth import TruthGrid

CAMPER = lambda obs, info: 6
ROUND_ROBIN = lambda obs, info: info["slot"] % K.N_BANDS


@pytest.fixture(scope="module")
def grid():
    return TruthGrid.from_scenario(Scenario.replay("config_59", "stare"))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"))
    run_episode(env, ROUND_ROBIN, seed=0)
    d = tmp_path_factory.mktemp("render")
    return M.write_run(d, env, scheduler="round_robin", seed=0)


# --------------------------------------------------------------------------- #
# The waterfall
# --------------------------------------------------------------------------- #

def test_the_waterfall_draws_the_path_the_scheduler_actually_flew(grid, run):
    fig = R.waterfall(grid, run)
    ax = fig.axes[0]
    (line,) = [ln for ln in ax.lines if ln.get_label() == "receiver tuning"]
    assert list(line.get_ydata()) == [row["band"] for row in run.log]


# --------------------------------------------------------------------------- #
# The live/animated views (ScanEnv.render() and its multi-scheduler sibling)
# --------------------------------------------------------------------------- #

def test_env_frame_is_a_real_picture_not_a_blank_one():
    env = ScanEnv(scenario=Scenario.replay("config_59", "stare"), render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(20):
        _, _, terminated, _, _ = env.step(6)
        if terminated:
            break
    frame = R.env_frame(env)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 1


def test_compare_animation_writes_a_nonempty_gif(tmp_path):
    """A coarse stride keeps this to a handful of frames -- correctness, not speed."""
    scenario = Scenario.replay("config_81", "stare")
    small_grid = TruthGrid.from_scenario(scenario)
    runs = {}
    for key, policy in (("round_robin", ROUND_ROBIN), ("camper", CAMPER)):
        env = ScanEnv(scenario=scenario)
        run_episode(env, policy, seed=0)
        runs[key] = M.write_run(tmp_path / key, env, scheduler=key, seed=0)

    out = R.compare_animation(runs, small_grid, tmp_path / "cmp.gif", stride=150, fps=10)
    assert out.exists()
    assert out.stat().st_size > 0


def test_compare_animation_marks_all_three_outcomes_with_distinct_colours(tmp_path):
    """Hit (Y=1,Z=1) red, miss (Z=1,Y=0 -- the receiver's own Pd<1) green,
    false alarm (Y=1,Z=0 -- the receiver's own Pfa) blue. `compare_animation`
    closes its figure inside `PillowWriter`, so it isn't introspectable after
    the fact the way `waterfall`'s returned figure is; this asserts on the
    data it actually draws from (`schedule_series`'s `occupied`/`hit`
    arrays) and that the render still completes with all three outcomes
    present, rather than on pixels. `config_2` band 4 (seed 0) has a real
    example of all three at once: 163 hits, 17 misses, 1 false alarm --
    found by sweeping every band/config combo, not picked in advance."""
    from rfenv.metrics.views import schedule_series
    from rfenv.render.comparison import (
        FALSE_ALARM_COLOUR,
        HIT_COLOUR,
        MISS_COLOUR,
        compare_animation,
    )

    BAND4_CAMPER = lambda obs, info: 4
    scenario = Scenario.replay("config_2", "stare")
    small_grid = TruthGrid.from_scenario(scenario)
    env = ScanEnv(scenario=scenario)
    run_episode(env, BAND4_CAMPER, seed=0)
    run = M.write_run(tmp_path / "camper", env, scheduler="camper", seed=0)
    n_hits = sum(1 for row in run.log if row["Y"] and row["Z"])
    n_misses = sum(1 for row in run.log if row["Z"] and not row["Y"])
    n_false_alarms = sum(1 for row in run.log if row["Y"] and not row["Z"])
    assert n_hits > 0 and n_misses > 0 and n_false_alarms > 0  # otherwise colours are untestable

    s = schedule_series(run)
    assert int((s["hit"] & s["occupied"]).sum()) == n_hits
    assert int((s["occupied"] & ~s["hit"]).sum()) == n_misses
    assert int((s["hit"] & ~s["occupied"]).sum()) == n_false_alarms
    assert len({HIT_COLOUR, MISS_COLOUR, FALSE_ALARM_COLOUR}) == 3  # all distinct

    out = compare_animation({"camper": run}, small_grid, tmp_path / "cmp.gif", stride=150, fps=10)
    assert out.exists() and out.stat().st_size > 0


def test_declared_hits_are_marked_and_true_occupancy_is_not(grid, run):
    """`Y`, not `Z`. Truth is already the background image; drawing a
    true-positive marker over it would be drawing the answer key on the exam, and
    a hit sitting on a dark cell should be legible as the false alarm it is."""
    fig = R.waterfall(grid, run)
    ax = fig.axes[0]
    (hits,) = [c for c in ax.collections if c.get_label() == "declared hit (Y=1)"]
    assert len(hits.get_offsets()) == sum(row["Y"] for row in run.log)
    assert sum(row["Y"] for row in run.log) != sum(row["Z"] for row in run.log)


def test_a_grid_renders_without_a_run(grid):
    """The raw-recording form: no scheduler, no overlay, same picture otherwise.
    This is the call that makes environment-versus-data a visual comparison."""
    fig = R.waterfall(grid)
    ax = fig.axes[0]
    assert not ax.lines and not ax.collections
    assert len(ax.images) == 1


def test_the_colour_scale_is_fixed_so_two_waterfalls_compare(grid):
    """Not scaled to each grid's own range. config_59 is a quiet scenario and
    config_921 a loud one; if the scale followed the data, the two figures would
    use the same colours for levels 40 dB apart."""
    loud = TruthGrid.from_scenario(Scenario.replay("config_921", "stare"))
    assert grid.S.max() != loud.S.max()

    clims = [R.waterfall(g).axes[0].images[0].get_clim() for g in (grid, loud)]
    assert clims[0] == clims[1] == (R.LEVEL_VMIN_DB, R.LEVEL_VMAX_DB)


def test_the_waterfall_covers_the_whole_episode_and_every_band(grid, run):
    ax = R.waterfall(grid, run).axes[0]
    assert ax.get_xlim() == (0.0, K.EPISODE_S)
    assert tuple(ax.images[0].get_extent()) == (0.0, K.EPISODE_S, -0.5, K.N_BANDS - 0.5)


def test_the_operating_point_is_on_the_figure(grid, run):
    """§7: state gamma alongside anything measured at it."""
    title = R.waterfall(grid, run).axes[0].get_title()
    assert f"{K.GAMMA_DBM:g}" in title and "round_robin" in title


def test_the_waterfall_writes_a_file(tmp_path, grid, run):
    path = tmp_path / "nested" / "waterfall.png"
    R.waterfall(grid, run, path)
    assert path.exists() and path.stat().st_size > 10_000


# --------------------------------------------------------------------------- #
# ROC and per-band
# --------------------------------------------------------------------------- #

def test_the_roc_plot_states_its_population(grid):
    """D33: Pd depends on the population entirely -- 0.819, 0.837 and 0.851 at the
    same gamma over three of them -- and the withdrawn 0.822 happened because a
    figure went out without one. Every ROC carries its population."""
    curve = roc([grid], np.arange(-125.0, -95.0, 1.0))
    fig = R.roc(curve, operating=operating_point([grid]))
    title = fig.axes[0].get_title()
    assert K.PD_POPULATION in title
    assert f"{curve['n_occupied']:,}" in title


def test_the_roc_curve_sweeps(grid):
    """Pd falls as gamma rises. Conditioned on `S >= gamma` instead of on
    threshold-free `Z` it could not (D26), and the plot would be a flat line."""
    gammas = np.arange(-125.0, -95.0, 1.0)
    line = R.roc(roc([grid], gammas)).axes[0].lines[0]
    pd = np.asarray(line.get_ydata())
    assert pd[0] > pd[-1]
    assert np.all(np.diff(pd) <= 1e-12)


def test_band_profile_draws_every_band_for_every_series():
    a, b = np.linspace(0, 1, K.N_BANDS), np.linspace(1, 0, K.N_BANDS)
    fig = R.band_profile({"predicted": a, "recorded": b})
    assert len(fig.axes[0].containers) == 2
    assert all(len(c.patches) == K.N_BANDS for c in fig.axes[0].containers)


def test_band_profile_shows_band_0_rather_than_hiding_it():
    """Stare cannot see below 500 MHz, so band 0 is 59.12% occupied in the
    recordings and 0.00% predicted (D10, D24). A stated limitation belongs on the
    plot, not off the left edge of it."""
    predicted = np.zeros(K.N_BANDS)
    recorded = np.zeros(K.N_BANDS)
    recorded[0] = 0.5912
    fig = R.band_profile({"predicted": predicted, "recorded": recorded})
    heights = [p.get_height() for p in fig.axes[0].containers[1].patches]
    assert heights[0] == pytest.approx(0.5912)
    assert fig.axes[0].get_xlim()[0] < 0
