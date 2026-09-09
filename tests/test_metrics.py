"""Artefact checks: do the files on disk say what the environment said?

The sharpest test here is `test_the_artefacts_reproduce_the_environments_own_numbers`.
`EVALUATION.md` §6 requires that a validation script reproduce all four gates
"from these artefacts alone", and the only way that claim can be checked is to
score the bytes on disk and compare them against the live environment. If the two
ever disagree, every gate built on the artefacts is wrong and the CSVs are the
thing to fix.

The rest guard the two ways an artefact set fails quietly: a column silently
changing shape, and a half-written run scoring low instead of erroring.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rfenv import constants as K
from rfenv import metrics as M
from rfenv.env import ScanEnv
from rfenv.rollout import run_episode
from rfenv.scenario import Scenario

ROUND_ROBIN = lambda obs, info: info["slot"] % K.N_BANDS
CAMPER = lambda obs, info: 6


@pytest.fixture(scope="module")
def scenario():
    """A stare replay, not a scan one. A scan replay's content sits where Turing's
    own sweep was tuned (D36), so a sweeping policy scores ~1.0 on it and the
    metrics under test would all be pinned at their maxima."""
    return Scenario.replay("config_2", "stare")


def finished(scenario, policy=ROUND_ROBIN, seed=0, reward="hit_z"):
    env = ScanEnv(scenario=scenario, reward=reward)
    return run_episode(env, policy, seed=seed)


def written(tmp_path, scenario, policy=ROUND_ROBIN, seed=0, scheduler="round_robin"):
    env = finished(scenario, policy, seed)
    run = M.write_run(tmp_path / scheduler, env, scheduler=scheduler, seed=seed)
    return env, run


# --------------------------------------------------------------------------- #
# The claim the whole module rests on
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("policy,name", [(ROUND_ROBIN, "round-robin"), (CAMPER, "camper")])
def test_the_artefacts_reproduce_the_environments_own_numbers(tmp_path, scenario, policy, name):
    """Scored from CSV and JSON on disk, §4 comes out identical to the live env.

    Exactly, not approximately -- both sides do the same arithmetic on the same
    integers, so any tolerance here would be hiding a bug rather than absorbing
    float noise.
    """
    env, run = written(tmp_path, scenario, policy, scheduler=name.replace("-", "_"))
    from_disk = M.scheduler_metrics(run)
    from_env = env.episode_metrics()

    for key in ("interception_ratio", "censored_mean_intercept_time_s", "emitter_coverage",
                "avg_intercept_rate_per_s", "total_reward", "n_detectable",
                "n_intercepted", "n_steps"):
        assert from_disk[key] == from_env[key], key


def test_a_run_round_trips_through_disk(tmp_path, scenario):
    """Types survive: bools stay bool, a never-intercepted emitter stays None
    rather than becoming slot 0, and the band list stays a list of ints."""
    env, run = written(tmp_path, scenario, CAMPER, scheduler="camper")

    assert len(run.log) == K.N_SLOTS
    assert [row["slot"] for row in run.log] == list(range(K.N_SLOTS))
    assert all(isinstance(row["Y"], bool) and isinstance(row["Z"], bool) for row in run.log)

    assert run.n_detectable == len(env.detectable)
    missed = [r for r in run.emitters if r["first_intercept_slot"] is None]
    assert missed, "the camper should miss somebody, or this test proves nothing"
    assert all(r["intercept_count"] == 0 and r["bands_seen_in"] == [] for r in missed)

    found = [r for r in run.emitters if r["first_intercept_slot"] is not None]
    assert all(all(isinstance(b, int) for b in r["bands_seen_in"]) for r in found)
    # D28: numerator and denominator of censored intercept time share one rule.
    assert all(r["first_intercept_slot"] >= r["on_slot"] for r in found)


def test_the_header_carries_the_two_things_the_tables_cannot(tmp_path, scenario):
    """D38: §4 is not computable from artefacts 1 and 2 alone.

    Interception ratio's denominator is the scenario's total illuminations, a
    grid-level quantity the per-slot log cannot hold -- the log carries only the
    numerator, and the two differ by a lot. Average reward is in neither table.
    """
    env, run = written(tmp_path, scenario)
    numerator = sum(row["pulses"] for row in run.log)

    assert run.header["total_pulses"] > numerator > 0
    assert set(M.LOG_FIELDS).isdisjoint({"total_pulses", "total_reward"})
    assert set(M.EMITTER_FIELDS).isdisjoint({"total_pulses", "total_reward"})
    assert M.scheduler_metrics(run)["interception_ratio"] == numerator / run.header["total_pulses"]


# --------------------------------------------------------------------------- #
# Failing loudly
# --------------------------------------------------------------------------- #

def test_a_truncated_artefact_set_is_refused(tmp_path, scenario):
    """A half-written run must error, not score low. This is the failure mode the
    repository exists to prevent: a number that looks like a result."""
    _, run = written(tmp_path, scenario)
    d = tmp_path / "round_robin"

    lines = (d / M.LOG_NAME).read_text().splitlines()
    (d / M.LOG_NAME).write_text("\n".join(lines[:-5]) + "\n")
    with pytest.raises(ValueError, match="log has"):
        M.read_run(d)

    (d / M.LOG_NAME).write_text("\n".join(lines) + "\n")
    rows = (d / M.EMITTER_NAME).read_text().splitlines()
    (d / M.EMITTER_NAME).write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="emitter table has"):
        M.read_run(d)


def test_a_changed_log_column_fails_loudly(tmp_path, scenario):
    """The env writes the log rows and this module writes the CSV header. If one
    side gains a column the other must not silently drop it."""
    env = finished(scenario)
    env.log[0] = dict(env.log[0]) | {"unexpected": 1}
    with pytest.raises(ValueError, match="episode log columns changed"):
        M.write_run(tmp_path / "bad", env, scheduler="round_robin", seed=0)


# --------------------------------------------------------------------------- #
# The metric definitions that are easy to get wrong (EVALUATION.md §4)
# --------------------------------------------------------------------------- #

def test_censoring_charges_a_missed_emitter_the_full_episode(tmp_path, scenario):
    """Not dropped from the average -- averaging over only the emitters you found
    rewards not looking, and measured, that makes the camper look *faster* than
    round-robin."""
    _, run = written(tmp_path, scenario, CAMPER, scheduler="camper")
    scored = M.scheduler_metrics(run)["censored_mean_intercept_time_s"]

    delays = [(K.N_SLOTS if r["first_intercept_slot"] is None else r["first_intercept_slot"])
              - r["on_slot"] for r in run.emitters]
    assert scored == float(np.mean(delays)) * K.SLOT_S

    found_only = [d for d, r in zip(delays, run.emitters) if r["first_intercept_slot"] is not None]
    assert np.mean(found_only) < np.mean(delays), "censoring must be the pessimistic one"


def test_interception_ratio_is_per_illumination_not_per_dwell(tmp_path, scenario):
    """§4 trap 1, in code. A camper's per-dwell hit rate is enormous while the
    fraction of the scenario's illuminations it captures is small."""
    _, run = written(tmp_path, scenario, CAMPER, scheduler="camper")
    ratio = M.scheduler_metrics(run)["interception_ratio"]
    per_dwell = np.mean([row["Y"] for row in run.log])
    assert per_dwell > 5 * ratio


def test_per_band_airtime_accounts_for_every_slot(tmp_path, scenario):
    """One band per slot, all 600 of them: airtime is the only currency (D31)."""
    _, run = written(tmp_path, scenario)
    bands = M.per_band(run)
    assert bands["slots_looked"].sum() == K.N_SLOTS
    assert (bands["hits"] <= bands["slots_looked"]).all()
    looked = bands["slots_looked"] > 0
    assert np.allclose(bands["hit_rate"][looked],
                       bands["hits"][looked] / bands["slots_looked"][looked])
    assert (bands["hit_rate"][~looked] == 0).all()


# --------------------------------------------------------------------------- #
# Across episodes
# --------------------------------------------------------------------------- #

def test_aggregate_reports_spread_not_just_a_mean(tmp_path, scenario):
    """§7: distributions and repeated-run statistics, never a grand mean alone."""
    rows = []
    for seed in range(4):
        env = finished(scenario, ROUND_ROBIN, seed=seed)
        run = M.write_run(tmp_path / f"s{seed}", env, scheduler="round_robin", seed=seed)
        rows.append(M.scheduler_metrics(run))

    agg = M.aggregate(rows)
    assert agg["n_runs"] == 4
    for key in ("interception_ratio", "emitter_coverage", "censored_mean_intercept_time_s"):
        block = agg[key]
        assert block["n"] == 4
        assert block["min"] <= block["p25"] <= block["median"] <= block["p75"] <= block["max"]
        assert "std" in block and "mean" in block


def test_aggregate_drops_nan_per_key_not_per_row():
    """A scenario with no illuminations has no interception ratio but still has a
    coverage figure worth keeping."""
    rows = [
        {"interception_ratio": float("nan"), "emitter_coverage": 0.5},
        {"interception_ratio": 0.25, "emitter_coverage": 0.75},
    ]
    agg = M.aggregate(rows, keys=("interception_ratio", "emitter_coverage"))
    assert agg["interception_ratio"]["n"] == 1
    assert agg["emitter_coverage"]["n"] == 2


def test_metrics_json_stamps_the_operating_point(tmp_path, scenario):
    """§7's reporting rule: never a scheduler table without the gamma it was
    measured at. The three families stay in separate blocks (§0)."""
    _, run = written(tmp_path, scenario)
    row = M.scheduler_metrics(run)

    path = M.write_metrics_json(
        tmp_path / M.METRICS_NAME,
        scheduler_runs={"round_robin": [row]},
        receiver={"pd": 0.851, "pfa": 1.35e-3, "population": K.PD_POPULATION},
        operating_point={"gamma_dbm": K.GAMMA_DBM, "sigma_db": K.NOISE_SIGMA_DB},
    )
    payload = json.loads(path.read_text())

    assert payload["operating_point"]["gamma_dbm"] == K.GAMMA_DBM
    assert set(payload) >= {"model", "receiver", "scheduler", "operating_point"}
    assert payload["model"] == {}, "model-level metrics are validate.py's to fill"
    block = payload["scheduler"]["round_robin"]
    assert block["aggregate"]["n_runs"] == 1
    assert block["per_scenario"][0]["scenario"] == run.header["scenario"]


def test_metrics_json_serialises_numpy_scalars(tmp_path):
    """`per_band` and the ROC hand back numpy types; json cannot encode them."""
    path = M.write_metrics_json(
        tmp_path / M.METRICS_NAME,
        receiver={"pd": np.float64(0.851), "n_occupied": np.int64(11710)},
    )
    assert json.loads(path.read_text())["receiver"] == {"pd": 0.851, "n_occupied": 11710}
