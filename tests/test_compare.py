"""The comparison runner and the pictures drawn from it.

`compare.py` is the ladder's equivalent of `validate.py`: the thing that makes a
results table a command's output rather than a claim. So what is tested here is
the machinery that could make the table wrong without making it *look* wrong --
running a rung on the scenario D36 forbids, scoring one rung by a different path
than another, or drawing a picture whose numbers disagree with the scorecard
beside it.

The visualisation tests assert **the data behind the pictures**, not the pixels.
A figure that renders is not a figure that is right, and `metrics.schedule_series`
and `metrics.discovery` exist precisely so the numbers can be checked without
comparing PNGs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rfenv import baselines as B
from rfenv import compare as C
from rfenv import constants as K
from rfenv import metrics as M
from rfenv.env import ScanEnv
from rfenv.scenario import Scenario
from rfenv.truth import TruthGrid

SMALL = ("random", "round_robin", "turing_sweep", "recency", "oracle_pulse")


@pytest.fixture(scope="module")
def scenario():
    return Scenario.replay("config_2", "stare")


@pytest.fixture(scope="module")
def grid(scenario):
    return TruthGrid.from_scenario(scenario)


@pytest.fixture(scope="module")
def runs(tmp_path_factory, scenario, grid):
    """One episode per rung on one scenario, at one seed -- what a figure needs."""
    root = tmp_path_factory.mktemp("compare")
    return root, {
        key: C.run_one(key, scenario, 0, root, grid=grid) for key in SMALL
    }


# --------------------------------------------------------------------------- #
# D36: where the comparison is allowed to run
# --------------------------------------------------------------------------- #

def test_a_scan_replay_is_refused(tmp_path, grid):
    """The single most damaging thing this module could do quietly.

    A scan grid holds only the pulses Turing's own sweep was tuned to, so a
    sweeping scheduler scores 0.9999 on it (D36). That is not a bug that shows up
    as an error; it shows up as an excellent result. Hence a refusal.
    """
    scan = Scenario.replay("config_2", "scan")
    with pytest.raises(C.ScanReplayRefused):
        C.run_one("turing_sweep", scan, 0, tmp_path)
    with pytest.raises(C.ScanReplayRefused):
        C.compare([scan], ("random",), [0], tmp_path, progress=False)


def test_the_comparison_set_is_stare_replays_and_samples(tmp_path):
    scenarios = C.comparison_scenarios(n_configs=3, n_sampled=2)
    assert len(scenarios) == 5
    assert all("/scan/" not in s.name for s in scenarios)
    assert sum(s.name.startswith("replay:") for s in scenarios) == 3
    # Two draws of the same size must not collide in `run_dir` (they would
    # overwrite each other's artefacts and one rung would score the other's run).
    sampled = [s.name for s in scenarios if not s.name.startswith("replay:")]
    assert len(set(sampled)) == len(sampled)


def test_a_truncated_run_is_a_prefix_of_the_full_one():
    """`--configs N` must be comparable to the full run, not a different sample."""
    assert [s.name for s in C.comparison_scenarios(n_configs=3)] == [
        s.name for s in C.comparison_scenarios(n_configs=5)
    ][:3]


# --------------------------------------------------------------------------- #
# Every rung goes through the same path
# --------------------------------------------------------------------------- #

def test_every_rung_is_scored_by_the_same_function(runs):
    """Identical metric keys for every rung, computed from artefacts on disk.

    If one rung were scored a different way -- from live state, or with a metric
    the others do not have -- the table would still print and would be comparing
    different quantities down its own column.
    """
    _, artefacts = runs
    keysets = {key: set(M.scheduler_metrics(run)) for key, run in artefacts.items()}
    assert len({frozenset(v) for v in keysets.values()}) == 1
    for key, run in artefacts.items():
        row = M.scheduler_metrics(run)
        assert row["scheduler"] == key
        for metric in C.HEADLINE:
            assert metric in row


def test_the_artefacts_reproduce_the_live_environment(scenario, grid, tmp_path):
    """`compare` inherits `metrics.py`'s claim: disk and live state agree exactly.

    Re-asserted here rather than assumed, because `run_one` is a second way into
    `write_run` and a bug in it would be invisible to `tests/test_metrics.py`.
    """
    run = C.run_one("recency", scenario, 0, tmp_path, grid=grid)
    env = ScanEnv(scenario=scenario)
    M.run_episode(env, B.make("recency", seed=0, grid=grid), seed=0)
    live, disk = env.episode_metrics(), M.scheduler_metrics(run)
    for metric in C.HEADLINE + ("n_steps", "n_detectable", "n_intercepted"):
        assert disk[metric] == pytest.approx(live[metric]), metric


def test_the_runner_writes_a_full_artefact_set(runs):
    root, artefacts = runs
    for key in artefacts:
        directory = M.run_dir(root, key, "replay:train/stare/config_2", 0)
        for name in (M.LOG_NAME, M.EMITTER_NAME, M.HEADER_NAME):
            assert (directory / name).exists(), f"{key} is missing {name}"


# --------------------------------------------------------------------------- #
# The paired comparison
# --------------------------------------------------------------------------- #

def _row(scheduler, scenario, seed, ratio, tti):
    return {"scheduler": scheduler, "scenario": scenario, "seed": seed,
            "interception_ratio": ratio, "censored_mean_intercept_time_s": tti}


def test_paired_wins_counts_pareto_dominance_not_averages():
    """`both` means better on both, on the same scenario and seed.

    Built on rows whose means are deliberately misleading: `mixed` beats the floor
    on ratio in one scenario and on intercept time in the other, so it never
    Pareto-dominates -- and a comparison of means would say it wins both.
    """
    rows = {
        "round_robin": [_row("round_robin", "a", 0, 0.10, 10.0),
                        _row("round_robin", "b", 0, 0.10, 10.0)],
        "mixed": [_row("mixed", "a", 0, 0.20, 12.0),
                  _row("mixed", "b", 0, 0.05, 8.0)],
        "dominant": [_row("dominant", "a", 0, 0.20, 8.0),
                     _row("dominant", "b", 0, 0.20, 8.0)],
        "tied": [_row("tied", "a", 0, 0.10, 10.0),
                 _row("tied", "b", 0, 0.10, 10.0)],
    }
    wins = C.paired_wins(rows)
    assert wins["mixed"] == {"n": 2, "beats_on_ratio": 0.5,
                             "beats_on_intercept_time": 0.5, "beats_on_both": 0.0}
    assert wins["dominant"]["beats_on_both"] == 1.0
    assert wins["tied"]["beats_on_both"] == 0.0   # a tie is not a win
    assert "round_robin" not in wins


def test_paired_wins_ignores_episodes_the_floor_did_not_run():
    rows = {
        "round_robin": [_row("round_robin", "a", 0, 0.1, 10.0)],
        "other": [_row("other", "a", 0, 0.2, 9.0), _row("other", "b", 0, 0.2, 9.0)],
    }
    assert C.paired_wins(rows)["other"]["n"] == 1


# --------------------------------------------------------------------------- #
# The data behind the pictures
# --------------------------------------------------------------------------- #

def test_the_timeline_series_accounts_for_every_slot_once(runs):
    """A timeline is a complete account of the 30 s, or it is decoration."""
    _, artefacts = runs
    for key, run in artefacts.items():
        series = M.schedule_series(run)
        assert np.array_equal(series["slot"], np.arange(K.N_SLOTS)), key
        assert series["band"].min() >= 0 and series["band"].max() < K.N_BANDS
        assert series["time_s"][-1] == pytest.approx(K.EPISODE_S - K.SLOT_S)


def test_dwell_boundaries_count_the_decisions_not_the_slots(runs):
    """One tick per look, so a 100 ms dwell draws as one decision (D31, D35).

    The count is the episode's step count, which is the property most easily lost:
    marking a boundary wherever the band changes would merge two consecutive
    one-slot dwells on the same band into one.
    """
    _, artefacts = runs
    for key, run in artefacts.items():
        series = M.schedule_series(run)
        assert int(series["dwell_start"].sum()) == run.header["n_steps"], key
        assert series["dwell_start"][0]
        assert np.array_equal(series["dwell_slot0"],
                              series["slot"][series["dwell_start"]])
        # Every dwell start is followed by exactly its own length before the next.
        starts = series["dwell_slot0"]
        lengths = series["dwell_slots"][series["dwell_start"]]
        assert np.array_equal(np.diff(starts), lengths[:-1])


def test_a_repeated_band_over_two_dwells_is_two_decisions(runs):
    """The camper case, in the extreme: 600 one-slot looks at one band.

    `band` never changes, so a band-change rule would report a single dwell.
    """
    _, artefacts = runs
    run = artefacts["round_robin"]
    fake = M.RunArtefacts(
        header={**run.header, "n_steps": K.N_SLOTS},
        log=[{**row, "band": 5, "dwell_slots": 1} for row in run.log],
        emitters=run.emitters,
    )
    series = M.schedule_series(fake)
    assert int(series["dwell_start"].sum()) == K.N_SLOTS


def test_the_discovery_curve_matches_the_scorecard(runs):
    """The curve's endpoint is the coverage the table prints. Same numbers, or the
    picture and the table are two different claims."""
    _, artefacts = runs
    for key, run in artefacts.items():
        d = M.discovery(run)
        row = M.scheduler_metrics(run)
        assert d["n_found"][-1] == row["n_intercepted"], key
        assert d["n_detectable"] == row["n_detectable"], key
        assert np.all(np.diff(d["n_found"]) >= 0), "coverage cannot go down"
        assert d["n_found"][0] == 0
        if row["n_detectable"]:
            assert (d["n_found"][-1] / d["n_detectable"]
                    == pytest.approx(row["emitter_coverage"]))


def test_the_discovery_curve_steps_where_the_emitter_table_says(runs):
    _, artefacts = runs
    run = artefacts["round_robin"]
    d = M.discovery(run)
    for row in run.emitters:
        first = row["first_intercept_slot"]
        if first is None:
            continue
        # The emitter is counted from the slot after its first intercept.
        assert d["n_found"][first + 1] > d["n_found"][first]


def test_emitter_activity_covers_the_detectable_set(runs):
    """One row per emitter in `E` (D27), ordered by onset, and `first_e >= on_e`."""
    _, artefacts = runs
    for key, run in artefacts.items():
        rows = M.emitter_activity(run)
        assert len(rows) == run.n_detectable, key
        assert [r["on_slot"] for r in rows] == sorted(r["on_slot"] for r in rows)
        for r in rows:
            assert r["on_slot"] <= r["off_slot"]
            if r["first_intercept_slot"] is not None:
                assert r["first_intercept_slot"] >= r["on_slot"], "D28"


# --------------------------------------------------------------------------- #
# The figures render
# --------------------------------------------------------------------------- #

def test_the_comparison_figures_are_written(tmp_path, runs, grid):
    """Pixels are not checked; that a figure is produced for every one is.

    matplotlib is imported here and nowhere in the environment layers, which is
    what keeps `ScanEnv` runnable without a plotting stack.
    """
    from rfenv import render

    _, artefacts = runs
    render.schedule_timeline(artefacts, grid, tmp_path / "timeline.png",
                             title="test")
    render.discovery_curves(artefacts, tmp_path / "discovery.png")
    render.pareto(
        {key: {**M.scheduler_metrics(run), "reference": key.startswith("oracle")}
         for key, run in artefacts.items()},
        tmp_path / "pareto.png",
    )
    for name in ("timeline.png", "discovery.png", "pareto.png"):
        assert (tmp_path / name).stat().st_size > 5_000, name


def test_the_timeline_survives_a_single_row(tmp_path, runs, grid):
    """`plt.subplots` collapses axes for one row; the caller must not."""
    from rfenv import render

    _, artefacts = runs
    render.schedule_timeline({"random": artefacts["random"]}, grid,
                             tmp_path / "one.png")
    assert (tmp_path / "one.png").exists()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_the_cli_writes_a_complete_comparison(tmp_path):
    """A real, tiny run: every output file, and a table with the §4 rule kept.

    Interception ratio, censored intercept time and coverage appear together in
    the report or §4's reporting rule is being broken by the tool that exists to
    enforce it.
    """
    out = tmp_path / "cmp"
    assert C.main(["--out", str(out), "--configs", "2", "--seeds", "1",
                   "--rungs", "round_robin,turing_sweep,recency"]) == 0

    payload = json.loads((out / "metrics.json").read_text())
    assert set(payload["scheduler"]) == {"round_robin", "turing_sweep", "recency"}
    assert payload["operating_point"]["gamma_dbm"] == K.GAMMA_DBM
    for block in payload["scheduler"].values():
        assert block["aggregate"]["n_runs"] == 2
        assert block["aggregate"]["interception_ratio"]["n"] == 2

    summary = json.loads((out / "summary.json").read_text())
    assert summary["meta"]["n_scenarios"] == 2
    assert summary["meta"]["source"] == "stare"
    assert summary["ladder"]["round_robin"]["rung"] == "2"
    assert summary["meta"]["paired"]["recency"]["n"] == 2

    report = (out / "comparison.md").read_text()
    for phrase in ("interception ratio", "censored intercept time",
                   "emitter coverage", "γ = "):
        assert phrase in report
    assert "D36" in report


def test_the_cli_rejects_an_unknown_rung(tmp_path):
    with pytest.raises(SystemExit):
        C.main(["--out", str(tmp_path), "--rungs", "ppo"])
