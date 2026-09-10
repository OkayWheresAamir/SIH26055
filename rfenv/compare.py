"""The scheduler comparison of `docs/project/EVALUATION.md` §4-§5, as a runnable script.

`validate.py` is to the gates what this is to the ladder: the point of both is
that a result is a command's output rather than a claim in a document. Every rung
of `baselines.py` runs on the same scenarios, at the same seeds, at the frozen
operating point, through the same `ScanEnv` and the same artefacts, and is scored
by the same `metrics.scheduler_metrics()`. Nothing here knows which rung it is
running.

**Where it runs, and why not everywhere.** Stare replays plus sampled scenarios,
never scan replays (D36). A scan recording holds only the pulses Turing's own
sweeping receiver was tuned to, so a grid built from one hands any sweeping
scheduler its answer -- measured, interception ratio 0.9999 and censored intercept
time 0.00 s. `_check_comparison_scenario` refuses one rather than trusting whoever
passes `--scenarios`.

**Reporting rules are enforced, not remembered** (`EVALUATION.md` §7). Every table
this module writes carries interception ratio, censored intercept time and
coverage together; `aggregate()` returns a distribution and never a bare mean; the
operating point is stamped on `metrics.json`; and the reference lines are marked
as reference lines in every output, because a ceiling read as a competitor is the
easiest mistake in the file.

**What this module deliberately does not do.** It does not choose a reward. All
rungs run at `env.DEFAULT_REWARD` because §4's two headline metrics do not depend
on it -- the reward only moves the `total_reward` column, which §4 says is
"comparable only within a reward family; never used to rank across different
rewards" (D7). Selecting among D29's three candidates is a human decision and is
still open. It also implements no learning: rung 7 is not here yet.

Usage::

    python -m rfenv.compare                          # full ladder, 47 stare replays
    python -m rfenv.compare --seeds 5                # five noise seeds each
    python -m rfenv.compare --sampled 20             # add 20 sampled scenarios
    python -m rfenv.compare --configs 3 --figures    # smoke test, with the plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rfenv import baselines as B
from rfenv.constants import GAMMA_DBM, NOISE_SIGMA_DB
from rfenv.env import DEFAULT_REWARD, ScanEnv
from rfenv.metrics import (
    aggregate,
    run_dir,
    scheduler_metrics,
    write_metrics_json,
    write_run,
)
from rfenv.receiver import operating_point
from rfenv.rollout import run_episode
from rfenv.scenario import EmitterPool, Scenario, list_configs
from rfenv.truth import TruthGrid

# The two headline metrics and the diagnostic that must be printed beside them
# (EVALUATION.md §4). Ordered as they are reported, so a table cannot quietly
# drop one.
HEADLINE = (
    "interception_ratio",
    "censored_mean_intercept_time_s",
    "emitter_coverage",
    "avg_intercept_rate_per_s",
)

# The scenarios the comparison is allowed to run on (D36).
COMPARISON_SOURCE = "stare"


class ScanReplayRefused(ValueError):
    """Raised when a scan replay is handed to the scheduler comparison (D36)."""


def _check_comparison_scenario(scenario: Scenario) -> None:
    if "/scan/" in scenario.name:
        raise ScanReplayRefused(
            f"{scenario.name} is a scan replay. A scan recording holds only the "
            "pulses Turing's own sweeping receiver was tuned to, so any sweeping "
            "scheduler scores ~1.0 on it (D36). The scheduler comparison runs on "
            "stare replays and sampled scenarios; scan replays belong to gates 1 "
            "and 2, where that imprint is the mechanism under test."
        )


# --------------------------------------------------------------------------- #
# One episode, one rung
# --------------------------------------------------------------------------- #

def run_one(
    key: str,
    scenario: Scenario,
    seed: int,
    out_root: Path | None = None,
    *,
    grid: TruthGrid | None = None,
    reward: str = DEFAULT_REWARD,
    gamma_dbm: float = GAMMA_DBM,
    sigma_db: float = NOISE_SIGMA_DB,
):
    """Run one rung on one scenario at one seed; return its artefacts.

    The grid is passed in rather than rebuilt because every rung on a scenario
    shares it -- and because the two reference lines need it, so building it here
    would mean building it twice for them and once for everyone else.

    The **same seed** goes to the environment and to the policy factory, which is
    what `EVALUATION.md` §7's "identical scenarios and seeds" has to mean: the
    receiver's noise draw and the policy's own randomness both reproduce. They do
    not collide -- `baselines.make` derives an independent stream per rung name.
    """
    _check_comparison_scenario(scenario)
    if grid is None:
        grid = TruthGrid.from_scenario(scenario)

    env = ScanEnv(scenario=scenario, reward=reward, gamma_dbm=gamma_dbm, sigma_db=sigma_db)
    policy = B.make(key, seed=seed, grid=grid)
    run_episode(env, policy, seed=seed)

    directory = (
        run_dir(out_root, key, scenario.name, seed)
        if out_root is not None
        else None
    )
    if directory is None:
        raise ValueError("run_one needs an out_root: the scorecard is read from disk")
    return write_run(directory, env, scheduler=key, seed=seed)


# --------------------------------------------------------------------------- #
# The comparison set
# --------------------------------------------------------------------------- #

def comparison_scenarios(
    n_configs: int | None = None,
    n_sampled: int = 0,
    sample_seed: int = 0,
) -> list[Scenario]:
    """The stare replays, plus optional sampled scenarios (D36, D25).

    Replays first and in config order, so a truncated `--configs` run is a prefix
    of the full one and the two are directly comparable. Sampled scenarios come
    from the train pool only; `scenario.EmitterPool.from_train` is the enforcement
    and the held-out guard sits under it (D8).
    """
    configs = list_configs(COMPARISON_SOURCE)
    if n_configs is not None:
        configs = configs[:n_configs]
    out = [Scenario.replay(c, COMPARISON_SOURCE) for c in configs]

    if n_sampled:
        pool = EmitterPool.from_train()
        rng = np.random.default_rng(sample_seed)
        for i in range(n_sampled):
            scenario = Scenario.sample(pool, rng)
            # `Scenario.sample` names every draw "sample:n=<k>", so two draws of
            # the same size would collide in `run_dir` and silently overwrite.
            out.append(Scenario(name=f"{scenario.name}:draw{i}",
                                contributions=scenario.contributions))
    return out


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #

def buildable_rungs(keys: tuple[str, ...], grid: TruthGrid) -> tuple[str, ...]:
    """`keys`, minus the rungs that cannot be constructed on this machine.

    A rung is unbuildable for three reasons, all of them legitimate and none of
    them a reason to lose the whole comparison: the training stack is not
    installed (`stable_baselines3`/`sb3_contrib` are optional -- no heuristic
    rung needs them), its checkpoint has not been trained yet, or its
    checkpoint predates an observation change and can no longer run (D49).
    Before this, the first such rung killed the entire run with a traceback,
    which meant the headline command did not work at all on a machine without a
    training stack.

    Probed once, by actually building each policy, rather than by guessing from
    the rung's name or checking for a file: whether a rung can be built is
    exactly the question "does building it raise", and only the factory knows.

    Warnings go to stderr, and the dropped rungs are named -- a comparison that
    quietly contains no RL rows looks like a success and is not one.
    """
    ok, dropped = [], []
    for key in keys:
        try:
            B.make(key, seed=0, grid=grid)
        except Exception as exc:                      # noqa: BLE001
            dropped.append(key)
            first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            print(f"  ! rung {B.BY_KEY[key].rung} ({key}) skipped: {first}",
                  file=sys.stderr)
        else:
            ok.append(key)
    if dropped:
        print(f"  ! {len(dropped)} of {len(keys)} rungs skipped: "
              f"{', '.join(dropped)}", file=sys.stderr)
    return tuple(ok)


def compare(
    scenarios: list[Scenario],
    keys: tuple[str, ...],
    seeds: list[int],
    out_root: Path,
    *,
    reward: str = DEFAULT_REWARD,
    progress: bool = True,
) -> dict[str, list[dict]]:
    """Every rung on every scenario at every seed. Returns `{key: [row, ...]}`.

    Scenario-major so each truth grid is built once and shared across the whole
    ladder -- and so every rung on a scenario sees exactly the same world, which
    is what makes the comparison paired rather than merely averaged.
    """
    rows: dict[str, list[dict]] = {k: [] for k in keys}
    t0 = time.time()
    for i, scenario in enumerate(scenarios, 1):
        _check_comparison_scenario(scenario)
        grid = TruthGrid.from_scenario(scenario)
        for key in keys:
            for seed in seeds:
                run = run_one(key, scenario, seed, out_root, grid=grid, reward=reward)
                rows[key].append(scheduler_metrics(run))
        if progress:
            print(f"  [{i:3d}/{len(scenarios)}] {scenario.name:<34} "
                  f"{time.time() - t0:6.1f} s", flush=True)
    return rows


def summarise(rows: dict[str, list[dict]]) -> dict[str, dict]:
    """Per-rung aggregate, with the rung's ladder metadata attached.

    `reference` travels with the numbers rather than being reattached at print
    time: the oracles are ceilings and every table that shows them has to say so
    (§5), so the flag belongs in the data.
    """
    out = {}
    for key, runs in rows.items():
        spec = B.BY_KEY[key]
        out[key] = {
            "rung": spec.rung,
            "label": spec.label,
            "purpose": spec.purpose,
            "reference": not spec.deployable,
            "aggregate": aggregate(runs),
            "n_runs": len(runs),
        }
    return out


def _means(summary: dict[str, dict]) -> dict[str, dict]:
    """`{label: {metric: mean}}` for the Pareto plot and the printed table."""
    out = {}
    for key, block in summary.items():
        agg = block["aggregate"]
        row = {m: agg[m]["mean"] for m in HEADLINE if agg.get(m, {}).get("n")}
        row["reference"] = block["reference"]
        row["rung"] = block["rung"]
        out[key] = row
    return out


# --------------------------------------------------------------------------- #
# Paired comparison against the floor
# --------------------------------------------------------------------------- #

BASELINE_TO_BEAT = "round_robin"


def paired_wins(rows: dict[str, list[dict]], against: str = BASELINE_TO_BEAT) -> dict:
    """Per-episode head-to-head against the floor, not a difference of means.

    `EVALUATION.md` §5: "Beating round-robin is the minimum." A mean can clear a
    mean while losing most scenarios -- difficulty spans 2 to 99 emitters and the
    hard ones dominate an average -- so the claim is checked **paired**: the same
    scenario, the same seed, the same truth grid, one rung against the other.

    `both` is the column that matters. D28 puts the two headline metrics on
    opposite sides of the Y/Z line and D14 measured that no trivial strategy is
    good at both, so *Pareto*-dominating the floor is the bar, and winning one
    column is not evidence of anything.
    """
    if against not in rows:
        return {}
    reference = {(r["scenario"], r["seed"]): r for r in rows[against]}
    out = {}
    for key, runs in rows.items():
        if key == against:
            continue
        ratio = tti = both = n = 0
        for row in runs:
            ref = reference.get((row["scenario"], row["seed"]))
            if ref is None:
                continue
            n += 1
            # Strictly better: a tie is not a win.
            r_win = row["interception_ratio"] > ref["interception_ratio"]
            t_win = (row["censored_mean_intercept_time_s"]
                     < ref["censored_mean_intercept_time_s"])
            ratio += r_win
            tti += t_win
            both += r_win and t_win
        if n:
            out[key] = {
                "n": n, "beats_on_ratio": ratio / n, "beats_on_intercept_time": tti / n,
                "beats_on_both": both / n,
            }
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _table(summary: dict[str, dict]) -> list[str]:
    lines = [
        f"{'rung':>4}  {'scheduler':<18} {'ratio':>16}  {'cTTI (s)':>16}  "
        f"{'coverage':>16}  {'rate/s':>7}",
        f"{'':>4}  {'':<18} {'mean [p25 p75]':>16}  {'mean [p25 p75]':>16}  "
        f"{'mean [p25 p75]':>16}  {'mean':>7}",
        "-" * 92,
    ]
    for key, block in summary.items():
        agg = block["aggregate"]

        def cell(metric, fmt="{:.3f}"):
            a = agg.get(metric)
            if not a or not a["n"]:
                return f"{'-':>16}"
            return (f"{fmt.format(a['mean'])} "
                    f"[{fmt.format(a['p25'])} {fmt.format(a['p75'])}]").rjust(16)

        mark = "*" if block["reference"] else " "
        rate = agg.get("avg_intercept_rate_per_s", {})
        lines.append(
            f"{block['rung']:>4}{mark} {key:<18} {cell('interception_ratio')}  "
            f"{cell('censored_mean_intercept_time_s', '{:.2f}')}  "
            f"{cell('emitter_coverage')}  "
            f"{rate.get('mean', float('nan')):7.3f}"
        )
    lines.append("")
    lines.append("* reference line, not a scheduler: reads the truth grid (EVALUATION.md §5).")
    return lines


def _paired_table(wins: dict, summary: dict[str, dict]) -> list[str]:
    lines = [
        f"paired against {BASELINE_TO_BEAT}, per episode "
        f"(same scenario, same seed, same grid)",
        f"{'scheduler':<18} {'ratio':>8} {'cTTI':>8} {'both':>8}",
        "-" * 46,
    ]
    for key, w in wins.items():
        mark = "*" if summary[key]["reference"] else " "
        lines.append(f"{key:<17}{mark} {w['beats_on_ratio']:7.1%} "
                     f"{w['beats_on_intercept_time']:7.1%} {w['beats_on_both']:7.1%}")
    return lines


def _report_md(summary: dict[str, dict], meta: dict) -> str:
    stamp = meta["written_utc"]
    out = [
        "# Scheduler comparison — the baseline ladder",
        "",
        f"`python -m rfenv.compare` — {stamp}. "
        f"{meta['n_scenarios']} scenarios x {len(meta['seeds'])} seed(s) = "
        f"{meta['n_runs']} episodes, train split only, "
        f"stare replays and sampled scenarios only (D36).",
        "",
        f"Operating point (§7): γ = {meta['operating_point']['gamma_dbm']} dB, "
        f"σ = {meta['operating_point']['sigma_db']} dB, "
        f"**P_fa = {meta['operating_point']['pfa']:.3e}**, "
        f"sensitivity {meta['operating_point']['sensitivity_dbm']:.2f} dB. "
        f"γ, σ, P_fa and sensitivity are frozen and data-independent (D25).",
        "",
        f"P_d here is **{meta['operating_point']['pd']:.4f}** over the "
        f"{meta['operating_point']['population']} cells of *these "
        f"{meta['n_scenarios']} comparison grids* "
        f"({meta['operating_point']['n_occupied']:,} occupied). It is not the "
        "0.85058 of `runs/validation`: D33 froze the *population rule*, not the set "
        "of grids, and that figure is over the 47 scan-replay grids `validate.py` "
        "builds. Same rule, different worlds. Quote whichever one you name the "
        "grids for, and never one without them (D33).",
        "",
        f"Reward `{meta['reward']}` throughout. It moves only the reward column and "
        "none of the metrics below (D7, D29); selecting among D29's three candidates "
        "is still an open human decision.",
        "",
        "Interception ratio, censored mean intercept time and emitter coverage are "
        "printed together and never one alone: a single scalar hides the entire "
        "problem (`EVALUATION.md` §4). Means come with the interquartile range "
        "because scenario difficulty spans 2 to 99 emitters (§7).",
        "",
        "| rung | scheduler | interception ratio | censored intercept time (s) | "
        "emitter coverage | intercept rate (/s) |",
        "|---|---|---|---|---|---|",
    ]

    def cell(agg, metric, fmt="{:.4f}"):
        a = agg.get(metric)
        if not a or not a["n"]:
            return "—"
        return (f"{fmt.format(a['mean'])} <br><sub>[{fmt.format(a['p25'])}, "
                f"{fmt.format(a['p75'])}]</sub>")

    for key, block in summary.items():
        agg = block["aggregate"]
        name = f"`{key}`" + (" *(reference line)*" if block["reference"] else "")
        out.append(
            f"| {block['rung']} | {name} | {cell(agg, 'interception_ratio')} | "
            f"{cell(agg, 'censored_mean_intercept_time_s', '{:.2f}')} | "
            f"{cell(agg, 'emitter_coverage')} | "
            f"{agg.get('avg_intercept_rate_per_s', {}).get('mean', float('nan')):.3f} |"
        )

    out += [
        "",
        "Cells are `mean` over the interquartile range. A **reference line** reads "
        "the truth grid and is not a scheduler (§5).",
        "",
        f"## Head to head against `{BASELINE_TO_BEAT}`, paired per episode",
        "",
        "Same scenario, same seed, same truth grid. A mean can clear a mean while "
        "losing most scenarios, so §5's \"beating round-robin is the minimum\" is "
        "checked paired. **`both` is the column that matters**: D14 measured that no "
        "trivial strategy is good at both objectives, so Pareto-dominating the floor "
        "is the bar and winning one column alone is not evidence.",
        "",
        "| scheduler | wins on ratio | wins on intercept time | **wins on both** |",
        "|---|---|---|---|",
    ]
    for key, w in meta.get("paired", {}).items():
        name = f"`{key}`" + (" *(reference line)*" if summary[key]["reference"] else "")
        out.append(
            f"| {name} | {w['beats_on_ratio']:.1%} | "
            f"{w['beats_on_intercept_time']:.1%} | **{w['beats_on_both']:.1%}** |"
        )
    out += [
        "",
        f"Over {meta['n_scenarios']} scenarios x {len(meta['seeds'])} seeds.",
        "",
        "## The ladder",
        "",
        "| rung | what it is |",
        "|---|---|",
    ]
    for key, block in summary.items():
        out.append(f"| {block['rung']} `{key}` | {block['purpose']} |")
    out += [
        "",
        "## Provenance",
        "",
        f"- Scenarios: {meta['n_scenarios']} "
        f"({meta['n_replays']} stare replays, {meta['n_sampled']} sampled), "
        f"seeds {meta['seeds']}.",
        f"- Every row is scored by `metrics.scheduler_metrics()` from the artefacts "
        f"under `{meta['out_root']}`, never from live environment state.",
        "- Scan replays are refused by `compare._check_comparison_scenario` (D36).",
        "- Rung 7 (RL) is absent: it does not exist yet.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

FIGURE_SCENARIOS = ("config_81", "config_2", "config_921")


def figures(out_root: Path, summary: dict[str, dict], keys: tuple[str, ...],
            seed: int, reward: str, *, gif_stride: int = 8, gif_fps: int = 12) -> list[Path]:
    """The three comparison pictures, a per-scenario timeline, and an animated
    GIF of the same rows, for each of the fixed scenarios.

    Import is local: matplotlib belongs to `render.py` and the comparison must run
    without a plotting stack.

    The scenarios are fixed -- the two gate-4 extremes and `config_2` -- so the
    figure set is the same every run and two runs can be laid side by side. They
    are re-run here rather than reloaded from `out_root` because a figure of one
    scenario needs every rung's artefacts for that scenario together, and finding
    them by path would be reconstructing what `compare()` already knows.

    **One combined GIF per scenario, not one per rung.** `render.compare_animation`
    already draws every compared rung as a synced row in one file -- that's what
    "compare" means, and it's the same artefacts `schedule_timeline` draws from, so
    the still and the animated picture cannot disagree. `gif_stride`/`gif_fps` default
    coarser than `compare_animation`'s own (stride=4) on purpose: this runs
    automatically on every `--figures` call, for as many rungs as were compared, so
    the default has to stay reasonable rather than the smoothest possible.
    """
    from rfenv import render

    written = []
    render.pareto(
        # Keyed by rung key (unique) with the display label carried inside the
        # row, not used as the key itself -- two rungs sharing a label (e.g.
        # two PPO variants) must still draw as two points (see render.pareto's
        # docstring for the bug this avoids).
        {k: {**row, "label": B.BY_KEY[k].label} for k, row in _means(summary).items()},
        out_root / "pareto.png",
        title="The baseline ladder on the two PS objectives",
    )
    written.append(out_root / "pareto.png")

    for config_id in FIGURE_SCENARIOS:
        scenario = Scenario.replay(config_id, COMPARISON_SOURCE)
        grid = TruthGrid.from_scenario(scenario)
        runs = {}
        for key in keys:
            runs[key] = run_one(key, scenario, seed, out_root, grid=grid, reward=reward)
        render.schedule_timeline(
            runs, grid, out_root / f"timeline_{config_id}.png",
            title=f"Receiver tuning over 30 s — {scenario.name}, seed {seed}, "
                  f"γ = {GAMMA_DBM:g} dB",
        )
        render.discovery_curves(
            runs, out_root / f"discovery_{config_id}.png",
            title=f"Emitters found against time — {scenario.name}, seed {seed}",
        )
        render.compare_animation(
            runs, grid, out_root / f"animation_{config_id}.gif",
            stride=gif_stride, fps=gif_fps,
        )
        written += [out_root / f"timeline_{config_id}.png",
                    out_root / f"discovery_{config_id}.png",
                    out_root / f"animation_{config_id}.gif"]
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rfenv.compare",
        description="Run EVALUATION.md §5's baseline ladder. Train split only.",
    )
    ap.add_argument("--out", default="runs/baselines", help="artefact directory")
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of noise seeds per scenario (default 3)")
    ap.add_argument("--configs", type=int, default=None,
                    help="limit to the first N stare replays (smoke test)")
    ap.add_argument("--sampled", type=int, default=0,
                    help="add N sampled scenarios drawn from the train pool")
    ap.add_argument("--rungs", default=None,
                    help=f"comma-separated subset of {sorted(B.BY_KEY)}")
    ap.add_argument("--reward", default=DEFAULT_REWARD,
                    help="reward candidate; affects the reward column only (D7)")
    ap.add_argument("--figures", action="store_true", help="write the comparison plots")
    ap.add_argument("--gif-stride", type=int, default=8,
                     help="slots between animation frames, --figures only (default 8)")
    ap.add_argument("--gif-fps", type=int, default=12, help="--figures only")
    args = ap.parse_args(argv)

    keys = tuple(r.key for r in B.LADDER)
    if args.rungs:
        keys = tuple(k.strip() for k in args.rungs.split(",") if k.strip())
        unknown = [k for k in keys if k not in B.BY_KEY]
        if unknown:
            ap.error(f"unknown rung(s) {unknown}; choose from {sorted(B.BY_KEY)}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    scenarios = comparison_scenarios(args.configs, args.sampled)
    # Settle which rungs can actually be built before announcing the episode
    # count, so the number printed is the number that will run.
    if scenarios:
        keys = buildable_rungs(keys, TruthGrid.from_scenario(scenarios[0]))
    if not keys:
        ap.error("no rung could be built -- nothing to compare")
    n_replays = sum(1 for s in scenarios if s.name.startswith("replay:"))
    print(f"{len(keys)} rungs x {len(scenarios)} scenarios x {len(seeds)} seeds "
          f"= {len(keys) * len(scenarios) * len(seeds)} episodes -> {out_root}")

    rows = compare(scenarios, keys, seeds, out_root, reward=args.reward)
    summary = summarise(rows)
    wins = paired_wins(rows)

    point = operating_point([TruthGrid.from_scenario(s) for s in scenarios])
    meta = {
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seeds": seeds,
        "reward": args.reward,
        "split": "train",
        "source": COMPARISON_SOURCE,
        "n_scenarios": len(scenarios),
        "n_replays": n_replays,
        "n_sampled": len(scenarios) - n_replays,
        "n_runs": sum(len(v) for v in rows.values()),
        "out_root": str(out_root),
        "operating_point": point,
        "paired": wins,
        "paired_against": BASELINE_TO_BEAT,
    }

    write_metrics_json(
        out_root / "metrics.json",
        scheduler_runs=rows,
        operating_point=point,
        notes=(
            "EVALUATION.md §5 baseline ladder. Stare replays and sampled scenarios "
            "only (D36). `camper_oracle` and `oracle_pulse` are reference lines that "
            "read the truth grid, not schedulers. Rung 7 (RL) does not exist yet."
        ),
    )
    (out_root / "summary.json").write_text(
        json.dumps({"meta": meta, "ladder": summary}, indent=2, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    (out_root / "comparison.md").write_text(_report_md(summary, meta), encoding="utf-8")

    if args.figures:
        for path in figures(out_root, summary, keys, seeds[0], args.reward,
                            gif_stride=args.gif_stride, gif_fps=args.gif_fps):
            print(f"  figure: {path}")

    print()
    for line in _table(summary):
        print(line)
    print()
    for line in _paired_table(wins, summary):
        print(line)
    print(f"\nartefacts: {out_root}")
    return 0


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
