# RF Environment — Consolidated Specification

**What this is.** The single buildable specification of the RF environment, consolidating every
decision in `docs/project/DECISIONS.md` (referenced as D1–D27) into one coherent design. A planning
session should be able to read this file plus `DECISIONS.md` and start building without
re-deriving anything. Written 2026-09-01; L0/L1 corrected 2026-09-01 after D23–D27.

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
fitted. `N` is drawn from the empirical per-config count of detectable emitters (1 to 82).

**Not built, and not deferred either (D25):** a physics signal model generating `S` from power,
range and beam pattern. Its antenna pattern and power scale are published nowhere and would have
to be **fitted to the same recordings the primary validation gate scores** — turning that gate
from a prediction into a fit. The recording-derived grid keeps it a prediction (86.19% accuracy
on held-out data, D24).

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
  `σ = 3 dB` (chosen, not measured), giving **Pd = 0.822, Pfa = 1.35e−3, sensitivity −107.2 dB**
  measured over the 47 train configs. The full sweep (ROC) is the receiver characterisation.
- The recorded **35.70%** non-empty dwell rate is now a **pipeline self-consistency test**:
  build the grid from the scan recording, replay the schedule that produced it, threshold
  nothing. Measured 35.403% replayed against 35.700% recorded.

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
- No retune cost (Turing's own sweep has none observable); if one is added later it is a
  reward term, not a receiver change.

## L3 — Agent interface

**Built fresh.** A standard Gymnasium environment. The interface below follows ordinary
Gymnasium conventions and the PS's own wording — not any existing implementation.
`docs/teammate-work/rf_env_grounded.py` is kept as a **cross-check, not a base**: a teammate
independently derived a near-identical interface from the same PS text, which is corroboration
that the shape is right. Nothing is inherited from it; its truth grid is an all-zeros
placeholder, which is precisely the part this spec replaces.

- `action_space = Discrete(36)`
- `observation_space`: dimension 36×3 + 1 — per-band empirical hit rate, per-band visit
  density, per-band staleness, plus normalised episode time. These three quantities are derived
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
- `info` dict: truth-side quantities for the evaluator only (per-emitter first-intercept slots
  under D28, cell occupancy) — never fed to the agent.
- `reset(seed, options={scenario})` takes either a deterministic replay or a sampled scenario
  (D25). Validation always uses replays; training uses samples.

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

1. **Episode log** (one CSV/parquet per run): per slot — time, band chosen, dwell length,
   declared hit, true occupancy, pulse count, peak level; per emitter — first seen, last seen,
   intercept count, bands seen in.
2. **Waterfall render** (one PNG/HTML per run): the 36×600 occupancy grid as a
   frequency-vs-time heatmap with the scheduler's path drawn over it and hits marked. This is
   the standard ESM operator view, and — crucially — the same picture can be drawn directly
   from the raw Turing recording, so environment and data are compared visually with no
   interpretation in between.
3. **Emitter track table**: per-emitter intercept summary (emitter, band(s), first/last
   intercept, count) — the shape of a real ESM intercept log.
4. **`metrics.json`**: the three metric families below, one file per evaluation run.

A validation script reproduces all four gates from these artefacts alone.

## Metrics and gates — see `docs/project/EVALUATION.md`

Three metric families (model-level, receiver-level, scheduler-level), the nine PS figures of
merit mapped one-to-one onto them, the baseline ladder, and the four validation gates are all
defined in `docs/project/EVALUATION.md`. Kept in one place deliberately: duplicating metric
definitions across documents is how they drift apart.

## Freeze list

When the four gates in `docs/project/EVALUATION.md` pass, these freeze and stop being open
questions — and they live in `rfenv/constants.py`, so the freeze list is a literal file: band
geometry, the slot clock, native dwell lengths, the truth-construction rule, `N₀`, `σ`, γ, the
metric definitions, and the scenario sampling distribution. **Free to vary per episode:** only
the draw — which emitters, and the seed. **Stays open for the RL lane:** reward candidates and
observation extensions. **Never:** no RL result may move anything on the frozen list; if one
does, the environment is re-validated from gate 1 and every baseline re-run (D25).

## Build order (for the planning session)

1. `scenario.py` — L0: emitter contributions, pool, replay and sampled scenarios. ✅ built
2. `truth.py` — L1: the `Z`/`S`/`C` grid, detectable intervals. ✅ built
3. `receiver.py` — L2: dwell mechanics, the noise draw, `Y`.
4. `env.py` — L3: gymnasium wrapper.
5. `render.py` + `metrics.py` — outputs above.
6. `validate.py` — gates 1–4 as a runnable script.

Five small modules mirroring the layers; baselines (random, round-robin, Turing sweep, greedy
camper, Apfeld) live outside the environment and consume only L3 + episode logs.

---

*Provenance: every measured number in this file traces to a command run in-session and is
recorded with its evidence in `docs/project/DECISIONS.md` and `docs/project/RESEARCH_MAP.md`. The PS text is in
`docs/project/SIH26055_PROBLEM_STATEMENT.md` and outranks this spec where they disagree.*
