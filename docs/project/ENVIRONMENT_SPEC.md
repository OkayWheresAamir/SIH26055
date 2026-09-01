# RF Environment — Consolidated Specification (v1)

**What this is.** The single buildable specification of the RF environment, consolidating every
decision in `docs/project/DECISIONS.md` (referenced as D1–D22) into one coherent design. A planning
session should be able to read this file plus `DECISIONS.md` and start building without
re-deriving anything. Written 2026-09-01.

**Design stance:** three layers, each simple enough to explain in a sentence. Complexity lives
in *calibration and validation*, not in the architecture.

```text
                 ┌──────────────────────────────────────────────┐
   Turing data → │ L0  SCENARIO PIPELINE  (offline, per config) │
                 └──────────────────┬───────────────────────────┘
                                    ↓  emitter table + signal grid
                 ┌──────────────────────────────────────────────┐
                 │ L1  TRUTH LAYER      S[36 bands × 600 slots] │
                 │     continuous signal level; occupancy = S≥γ │
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
plus **both** recordings (union of evidence, D17).

**Output per scenario:**

1. **Emitter table** — one row per transmitter: label, `freqs_mhz`, `freq_mode`, `pris_us`,
   `pws_us`, beam (`beam_width_deg`, rotation period = `30/scan_rate_rpm` [measured — the field
   is not literal rpm], `scan_start_angle`), position, `power_w`/`gain`, and the **activity
   window** `[t_start, t_end]` recovered from the union of recordings (D2). 76.8% of emitters
   are a single continuous window; multi-burst emitters get their observed burst list.
2. **Signal grid** `S[36, 600]` — peak received level (dB) per (band, 50 ms slot), built from
   the union of both recordings. Empty cells sit at the ambient noise floor (TSRD paper:
   −100 dB) plus a small noise draw, so that thresholding produces a real Pfa.

**Scenario variation (D18):** the loader replays a config deterministically (for validation) or
randomises within it (for training): activity-window placement, beam phase, and positions,
behind a single seed. The 47 train configs are templates for a scenario *distribution*, not 47
fixed worlds.

**Deferred, deliberately (keep it simple):** generating `S` from emitter physics (Apfeld Eq. 1:
power, range, beam pattern) instead of from recordings. The recording-built grid is the v1
truth; the generative signal model is a v2 upgrade whose acceptance test already exists —
it must reproduce the v1 grid. Do not build both at once.

## L1 — Truth layer

- **Bands:** Turing's 36 centres (250 + 500k MHz), window = centre ± 500 MHz (measured; bands
  overlap by half — D3). **Clock:** 50 ms slots; 600 per 30 s episode (D16).
- **State:** `S[b, t]` continuous. **Occupancy** `O[b, t] = (S[b, t] ≥ γ)` — the PS's binary
  transmission/non-transmission, derived, per D4.
- **γ is frozen with the environment before any agent runs** (D15). Chosen by calibration:
  replaying Turing's own sweep must reproduce the recorded scan statistics (~35% non-empty
  dwell rate — measured 35.3–35.7% by convention — and per-band structure). The full threshold sweep (ROC) is reported once as the
  receiver characterisation.

## L2 — Receiver layer

- **Action = choose a band.** The dwell then runs that band's native Turing length — 100 ms
  (2 slots) for the seven wide-dwell bands, 50 ms (1 slot) otherwise (D3, D16).
- **Observation channel:** for each slot of the dwell, the receiver sees hit/miss = `O[band, t]`.
  Because empty cells carry noise, a hit can be false (Pfa > 0) and a weak transmission can be
  missed (Pd < 1). Pd and Pfa are **properties of this layer at the frozen γ — they do not
  depend on the scheduler** (D15). What the scheduler controls is *which cells get looked at*.
- No retune cost in v1 (Turing's own sweep has none observable); if one is added later it is a
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
  metrics below. Default v1 candidate: +1 per true hit, with censored intercept time doing the
  discovery-pricing at evaluation (D5, D14).
- `info` dict: truth-side quantities for the evaluator only (per-emitter first-intercept slots,
  cell occupancy) — never fed to the agent.
- `reset(seed, options={config, randomize})` selects scenario and variation.

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
questions: band geometry, the detection threshold γ, the truth pipeline, and the metric
definitions. **Stays open for the RL lane:** reward candidates, observation extensions, and how
much scenario randomisation to train with.

## Build order (for the planning session)

1. `scenario.py` — L0 loader: emitter table + signal grid from one config; seedable variation.
2. `truth.py` — L1: grid container, γ, occupancy.
3. `receiver.py` — L2: dwell mechanics, detection channel.
4. `env.py` — L3: gymnasium wrapper (start from the interface of `rf_env_grounded.py`).
5. `render.py` + `metrics.py` — outputs above.
6. `validate.py` — gates 1–4 as a runnable script.

Five small modules mirroring the layers; baselines (random, round-robin, Turing sweep, greedy
camper, Apfeld) live outside the environment and consume only L3 + episode logs.

---

*Provenance: every measured number in this file traces to a command run in-session and is
recorded with its evidence in `docs/project/DECISIONS.md` and `docs/project/RESEARCH_MAP.md`. The PS text is in
`docs/project/SIH26055_PROBLEM_STATEMENT.md` and outranks this spec where they disagree.*
