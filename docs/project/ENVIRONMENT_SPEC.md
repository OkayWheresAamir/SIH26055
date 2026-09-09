# RF Environment — Consolidated Specification

**What this is.** The single buildable specification of the RF environment, consolidating every
decision in `docs/project/DECISIONS.md` (referenced as D1–D42) into one coherent design. A planning
session should be able to read this file plus `DECISIONS.md` and start building without
re-deriving anything. Written 2026-09-01; L0/L1 corrected 2026-09-01 after D23–D27, again 2026-09-03
after the third consistency audit (D32–D34), and again 2026-09-04 when the validation gates first
ran (D39–D41).

**Design stance:** three layers, each simple enough to explain in a sentence. Complexity lives
in *calibration and validation*, not in the architecture.

```text
                 ┌──────────────────────────────────────────────┐
   Turing data → │ L0  SCENARIO PIPELINE  (offline, per config) │
                 └──────────────────┬───────────────────────────┘
                                    ↓  emitter contributions
                 ┌──────────────────────────────────────────────┐
                 │ L1  TRUTH LAYER    Z, S, C [36 bands × 600]  │
                 │     occupancy Z; level S; pulse count C      │
                 └──────────────────┬───────────────────────────┘
                                    ↓  looked-at cells only
                 ┌──────────────────────────────────────────────┐
                 │ L2  RECEIVER LAYER   dwell → hit/miss        │
                 └──────────────────┬───────────────────────────┘
                                    ↓  gymnasium API
                 ┌──────────────────────────────────────────────┐
                 │ L3  AGENT INTERFACE  action=band, obs=history │
                 └──────────────────────────────────────────────┘
```

---

## L0 — Scenario pipeline (offline, one pass per config)

**Input:** one Turing config — `metadata/transmitters` (identical in scan and stare, verified)
plus both recordings. They are **independent simulation runs**, not two views of one world
(D24), so each contributes its own emitter realisations; they are never stitched into one grid.

**Output per scenario:**

1. **Emitter contributions** — the unit the environment is built from. One per emitter *per
   recording*: label, `freqs_mhz`, `freq_mode`, `pris_us`, `pws_us`, beam
   (`beam_width_deg`, `scan_rate_rpm`, `scan_start_angle`), position, `power_w`/`gain`, and a
   **sparse set of (band, slot, peak level, pulse count) cells**. Beam and position fields are
   carried through but not computed with — `scan_rate_rpm` is not literal rpm and its semantics
   are unconfirmed (D25). No activity-window field: `on_e`/`off_e` are derived from the grid as
   the **detectable activity interval** (D27).
2. **Signal grid** `S[36, 600]` — peak received level (dB) per (band, 50 ms slot), the maximum
   over the scenario's emitter contributions. Empty cells sit at the receiver's noise floor
   `N₀ = −120 dB` (= `sensitivity_dbm` −110 minus `gain_db` 10). **Not the TSRD paper's −100 dB
   ambient noise** — that describes the generator's probabilistic detector, and 13.25% of
   recorded scan pulses sit below it; the recordings are already post-detection (D23). The noise
   draw is applied by the receiver at look time, which is what makes Pfa real.

**Scenario sampling (D25):** the loader either **replays** one recording deterministically (what
the validation gates run on) or **samples** a scenario as a draw of N emitter contributions from
the pool of 3,443 (what RL trains on). Because TSRD placed emitters independently with no
emitter–emitter interaction, a recombination is as physically valid as an original config — it
is the same generative process TSRD used, one level up, with nothing invented and nothing
fitted. `N` is drawn from the empirical per-config count of detectable emitters (1 to 82), and the draw
is over **emitters, not contributions** — an emitter detected in both runs offers two
realisations and exactly one is used, so no scenario contains the same physical emitter twice
(D32).

**Not built, and not deferred either (D25):** a physics signal model generating `S` from power,
range and beam pattern. Its antenna pattern and power scale are published nowhere and would have
to be **fitted to the same recordings the primary validation gate scores** — turning that gate
from a prediction into a fit. The recording-derived grid keeps it a prediction rather than a fit —
**measured 2026-09-04 by gate 1 at accuracy 0.8585, MCC 0.6854** over 23,594 dwells of data never
used in construction (D17, D37; `EVALUATION.md` §6).

## L1 — Truth layer

- **Bands:** Turing's 36 centres (250 + 500k MHz), window = centre ± 500 MHz (measured; bands
  overlap by half — D3). **Clock:** 50 ms slots; 600 per 30 s episode (D16).
- **State:** three arrays. `Z[b, t]` — physical occupancy, threshold-free: is any emitter
  transmitting into this cell? That is the PS's binary transmission/non-transmission.
  `S[b, t]` — the continuous level, `N₀` where `Z` is false (D4). `C[b, t]` — pulse count, the
  interception-ratio numerator.
- **γ is frozen with the environment before any agent runs** (D15). It is **not** calibrated to
  the recorded ~35% dwell rate — that procedure was confounded and is retracted (D23). γ is a
  swept receiver parameter; the default operating point is `γ = N₀ + 3σ = −111 dB` with
  `σ = 3 dB` (chosen, not measured), giving **Pfa = 1.35e−3** and **sensitivity −107.2 dB** —
  both analytic in γ and σ, both re-run 2026-09-03. **Pd = 0.851**, averaged over the
  reference-sweep population — the occupied cells Turing's own schedule looks at (D33, `SETTLED`;
  `PD_POPULATION` in `rfenv/constants.py`). The population is on the freeze list because the
  figure depends on it entirely: 0.819 over stare-replay cells, 0.837 over scan-replay cells, and
  the previously quoted 0.822 matched none of them and is withdrawn. Re-measured from
  `rfenv.receiver` on 2026-09-03 over all 47 train scan replays: Pd 0.8506, Pfa 1.350e−3,
  sensitivity −107.16 dB, 11,710 occupied cells. The full sweep (ROC) is the receiver
  characterisation.
- The recorded non-empty dwell rate is now a **pipeline self-consistency test**: build the grid
  from the scan recording, replay the schedule that produced it, threshold nothing. **Measured
  2026-09-04 by gate 2: replayed and recorded agree exactly at 35.403%** — 0.000 pp aggregate and
  0.000 pp on every band. The previously recorded "35.403% replayed against 35.700% recorded" is
  withdrawn: the 0.3 pp residual was a band-blind recorded side, not slot quantisation (D41).

## L2 — Receiver layer

- **Action = choose a band.** The dwell then runs that band's native Turing length — 100 ms
  (2 slots) for the seven wide-dwell bands, 50 ms (1 slot) otherwise (D3, D16).
- **Observation channel:** for each slot of the dwell the receiver measures `S[band, t] + n`,
  `n ~ N(0, σ)`, and declares `Y = 1` iff that beats γ. A hit can be false (Pfa > 0) and a weak
  transmission missed (Pd < 1). Pd and Pfa are measured against **physical occupancy `Z`**, not
  against a second thresholded copy of `S` — referencing them to `(S ≥ γ)` is degenerate and
  cannot sweep (D26). They are **properties of this layer at the frozen γ and do not depend on
  the scheduler** (D15, D21). What the scheduler controls is *which cells get looked at*.
- **Crediting an intercept to an emitter** (D28): `e` is intercepted at slot `t` iff the tuned
  band is one `e` pulses into at `t`, **`e`'s own** level there clears γ, and `Y(t) = 1`. `Y` is
  declared on the combined `S`, so a weak emitter qualifying under its own level and sharing a
  cell with a louder one is near-certain to be credited — accepted; per-pulse detection is the
  v2 form. The own-level clause is what stops a quiet emitter inheriting a loud neighbour's
  detectability, and it is the same rule that fixes `on_e` (D27).
- **No retune cost — measured, not assumed** (D31). The sweep period is exactly 2.15 s =
  `sum(dwell_times_s)`; assuming even 1 µs of dead time per retune drops the schedule replay's
  in-band fraction from **99.985% to 99.829%** over 4,393,233 train scan pulses, and 100 µs drops
  it to 82.9%. Retuning costs under 1 µs, i.e. under 0.002% of a slot. **Airtime is therefore the
  only currency in this problem.** If a retune cost is added later it is a reward term, not a
  receiver change.

## L3 — Agent interface

**Built fresh.** A standard Gymnasium environment. The interface below follows ordinary
Gymnasium conventions and the PS's own wording — not any existing implementation.
`docs/teammate-work/rf_env_grounded.py` is kept as a **cross-check, not a base**: a teammate
independently derived a near-identical interface from the same PS text, which is corroboration
that the shape is right. Nothing is inherited from it; its truth grid is an all-zeros
placeholder, which is precisely the part this spec replaces.

- `action_space = Discrete(36)`
- `observation_space`: dimension **36×4 + 3 = 147**. The base 36×3 + 1 — per-band empirical hit
  rate, per-band visit density, per-band staleness, plus normalised episode time — **recorded as
  D34** (`SETTLED`), then **extended by D49** with `current_band` (36-wide one-hot of the band just dwelt on), `camp_time` (consecutive slots on that band / N_SLOTS) and `measured_dbm` (the last dwell's mean measured level, clamped and rescaled). D34's own text left extensions to the RL lane; D49 is that
  extension, and every observation-width change invalidates every trained checkpoint. D34 —
  the choice lived only in this spec until the 2026-09-03 audit found it ungated while its
  proposed extension (D30) correctly was. Ratified as-is; it is not on the freeze list, so it
  stays the RL lane's to extend. These three quantities are derived
  from the PS's own figures of merit (they are what a scheduler needs to estimate detection
  probability, intercept rate and staleness respectively), and are built **only** from the
  agent's own scan history — no prior emitter intelligence (D19, D20). The RL lane may extend
  this; extensions get logged as decisions.
- `reward`: pluggable (D7). Candidates are trained separately and judged on the scheduler-level
  metrics below. Default first candidate: +1 per true hit, with censored intercept time doing the
  discovery-pricing at evaluation (D5, D14).
  **The reward may read truth-side state; the observation may not** (D29). Training is offline
  and the policy is frozen before deployment, so the reward is a training-time construct that is
  discarded at inference — only the observation ships, and only it carries the deployability
  constraint. D28 makes this a live choice rather than a formality: `first_e` requires `Y = 1`,
  so censored intercept time is `Y`-conditioned while interception ratio stays threshold-free,
  and no single reward is aligned with both. The three D7 candidates are therefore +1 per true
  hit (`Z`), +1 per declared hit (`Y`), and +1 per first intercept of an emitter (D28's three
  clauses). **No reward can move P<sub>d</sub> or P<sub>fa</sub>** — those are frozen receiver
  properties (D15, D21); a false-alarm penalty prices a wasted dwell, nothing more.
  **Reward is defined per slot, and a dwell's reward is the sum of its slots'** (D31) — so a
  100 ms dwell on one of the seven wide bands is scored on both its cells and can earn up to +2.
  This holds for all three candidates, and it keeps reward-per-unit-time equal across wide and
  narrow bands: a wide band costs twice the airtime and can earn twice the credit. The rejected
  alternative, +1 per dwell regardless of length, would make those seven bands strictly dominated
  — and they are measurably the bands that matter most, holding the densest emitter populations
  (mean 764.6 emitter-frequency placements against 170.4) and the slowest rotators (bands 0 and 1
  at median 3.00 rpm). Every metric that judges a reward is already per-slot or per-illumination
  (`EVALUATION.md` §4), so per-dwell scoring would be the only per-dwell quantity in the project.
  Implementation: a wide-band action returns one `step()` with `reward = r[t] + r[t+1]` and
  advances the clock by two slots.
- `info` dict: truth-side quantities for the evaluator only — never fed to the agent. It must
  carry **the whole per-emitter track, not just first-intercept slots**: first *and last*
  intercept, intercept count, and bands seen in, which is `EVALUATION.md` §8 artefact 2.
  First sightings alone are not enough and the per-slot episode log cannot make up the
  difference — it carries no emitter attribution. Measured under round-robin, 92.8% of
  `config_2`'s intercept events and 94.6% of `config_921`'s are re-sightings. Accumulating
  them in L3 keeps D28's three-clause rule in one place; the alternative was re-deriving it
  in `metrics.py` from the logged `measured_dbm`.
- `reset(seed, options={scenario})` takes either a deterministic replay or a sampled scenario
  (D25). Validation always uses replays; training uses samples. Seeding fixes both the
  scenario draw and the receiver's noise, so a seed reproduces an episode exactly.
- **An episode is 600 slots but 300–600 `step()` calls** (D35), because the seven wide bands
  consume two slots each — the wall clock is always 30 s, the number of decisions is not.
  The horizon returns `terminated`, not `truncated`: 30 s is the task, not a harness cap, so
  there is no state past it to bootstrap from. A wide dwell starting at slot 599 is clipped
  rather than forbidden, so every band stays legal at every slot.

## Cold-start operating assumption (D20)

Every episode starts with zero emitter knowledge — no threat library, no carry-over between
scenarios. This is the PS's *"absence of prior reliable intelligence"* made literal, and it
matches the project's intent reading: current systems lean on historical emitter libraries;
the situation being targeted is one where emitters are not where any library says they are.

---

## Outputs and evaluation

**`docs/project/EVALUATION.md` is the authority on metrics, baselines and gates** — exact
definitions, formulas and the comparison protocol live there so this spec stays about the
environment. What follows is the environment's obligation: the artefacts it must emit so that
evaluation is possible at all.

### How we see that it works

The environment's per-episode output is designed to be *checked by eye against the raw data*,
in the formats real spectrum-surveillance systems actually use (waterfall display + intercept
log):

1. **Episode log** (`episode_log.csv`, one per run): per slot — time, band chosen, dwell length,
   declared hit, true occupancy, pulse count, peak level.
2. **Waterfall render** (`.png`, one per run): the 36×600 occupancy grid as a
   frequency-vs-time heatmap with the scheduler's path drawn over it and hits marked. This is
   the standard ESM operator view, and — crucially — the same picture can be drawn directly
   from the raw Turing recording, so environment and data are compared visually with no
   interpretation in between. Drawn from a **stare** replay for that comparison; a scan
   replay's content lies along Turing's own sweep and reads as a scheduler success that is not
   one (D36).
3. **Emitter track table** (`emitter_table.csv`): per-emitter intercept summary (emitter,
   band(s), first/last intercept, count) — the shape of a real ESM intercept log.
4. **Run header** (`run.json`): the episode's scalars — scenario, scheduler, seed, reward, γ, σ,
   total illuminations, total reward. The two of those that §4 needs and neither table can hold
   are why it exists (D38).
5. **`metrics.json`**: the three metric families below, one file per evaluation run.

A validation script reproduces all four gates from these artefacts alone — which is a testable
claim, not a hope: `metrics.py` scores §4 by reading the files back, and a test asserts it
matches what the environment said in memory.

## Metrics and gates — see `docs/project/EVALUATION.md`

Three metric families (model-level, receiver-level, scheduler-level), the nine PS figures of
merit mapped one-to-one onto them, the baseline ladder, and the four validation gates are all
defined in `docs/project/EVALUATION.md`. Kept in one place deliberately: duplicating metric
definitions across documents is how they drift apart.

## Freeze list

**Frozen 2026-09-04 (D42)** — the gates passed, the freeze is taken, and
`tests/test_freeze.py` pins every value with a digest tripwire over the whole list. These are no
longer open questions — and they live in `rfenv/constants.py`, so the freeze list is a literal file: band
geometry, the slot clock, native dwell lengths, the truth-construction rule, `N₀`, `σ`, γ, the
metric definitions, and the scenario sampling distribution. **Free to vary per episode:** only
the draw — which emitters, and the seed. **Stays open for the RL lane:** reward candidates and
observation extensions. **Never:** no RL result may move anything on the frozen list; if one
does, the environment is re-validated from gate 1 and every baseline re-run (D25).

## Build order (for the planning session)

1. `scenario.py` — L0: emitter contributions, pool, replay and sampled scenarios. ✅ built
   (sampler corrected 2026-09-03, D32)
2. `truth.py` — L1: the `Z`/`S`/`C` grid, detectable intervals. ✅ built
3. `receiver.py` — L2: dwell mechanics, the noise draw, `Y`. ✅ built
4. `env.py` — L3: gymnasium wrapper. ✅ built
5. `metrics.py` + `render.py` — outputs above. ✅ built (2026-09-04). `metrics.py`
   scores `EVALUATION.md` §4 **from the artefacts on disk**, never from live env
   state, so §6's "reproduces the gates from these artefacts alone" is a tested
   claim rather than an aspiration; the artefact set gained a run header (D38).
   `render.py` owns the waterfall, the ROC and the per-band bar, and is the only
   module that imports matplotlib.
6. `validate.py` — gates 1–4 as a runnable script. ✅ built (2026-09-04). The pass
   criteria live in `validate.py::GATES`, fixed *before* the first run and asserted
   against D39 by a test, so a threshold cannot be retuned after a bad result — the
   structural fix the 2026-09-03 audit asked for. Gate 1's convention is D37's; gates
   2, 3 and 4 are D39's resolution. Gate 3 needed **D40** (Köksal's `P₁₂(T)` assumes
   independent successive periods and does not apply to a deterministic periodic pair —
   reported, never gated), and gate 2's first run produced **D41** (the recorded dwell
   rate is 35.403%, not the withdrawn 35.700%). **All four have now run: three PASS,
   gate 1 MEASURED** (`EVALUATION.md` §6).

7. `baselines.py` + `compare.py` — `EVALUATION.md` §5's ladder and the runner that scores it.
   ✅ built (2026-09-04), **after** the freeze (D42), which is the order §7 requires. `baselines.py`
   holds seven schedulers and two truth-reading reference lines and imports nothing below L3;
   `compare.py` runs every rung on every scenario at every seed through the same `ScanEnv` and the
   same `metrics.scheduler_metrics()`, refuses scan replays (D36), and writes `comparison.md`,
   `metrics.json`, `summary.json` and the comparison figures. Rungs 2 and 3 were one policy until
   **D43**; Apfeld needed **D44** (its Algorithm 1 contradicts its own prose) and gained its own
   ablation as rung 6a (**D45**). First run: **D46**.

Eight small modules mirroring the layers. The baselines live outside the environment and consume
only L3 + the episode artefacts — `baselines.guarded` strips `info` down to what a fielded
receiver has, so a rung that reaches for `Z` raises rather than scoring well. `validate.py`
implements no policy of its own: gates 3 and 4 drive the environment with
`constants.dwell_schedule()`, which is Turing's own frozen schedule.

---

*Provenance: every measured number in this file traces to a command run in-session and is
recorded with its evidence in `docs/project/DECISIONS.md` and `docs/project/RESEARCH_MAP.md`. The PS text is in
`docs/project/SIH26055_PROBLEM_STATEMENT.md` and outranks this spec where they disagree.*
