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
    M.run_episode(env, ROUND_ROBIN, seed=0)
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
