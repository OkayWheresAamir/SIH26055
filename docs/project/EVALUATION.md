# Evaluation

**The single authority on how anything in this project is measured.** Metric definitions, the
baseline ladder, the validation gates, and the comparison protocol. Rationale for each choice
lives in `docs/project/DECISIONS.md`; this file is the implementable version.

Kept separate from `ENVIRONMENT_SPEC.md` on purpose: the environment must not define its own
scorecard, and duplicated metric definitions drift apart.

---

## 0. The three families, and why the split matters

The PS lists nine figures of merit. Most confusion in this project came from mixing three
different kinds of question. Separating them is what makes the evaluation honest (D21).

| Family | Asks | Varies with the scheduler? |
|---|---|---|
| **Model-level** | Is the environment a faithful model of the data? | No |
| **Receiver-level** | How good is the receiver when it looks? | No |
| **Scheduler-level** | How well did we aim it? | **Yes** |

A scheduler cannot improve P<sub>d</sub> — it can only point the detector at more of the right
cells. Reporting a scheduler comparison on receiver-level metrics is a category error.

---

## 1. Notation

For one scenario over one 30 s episode:

- Bands `b ∈ {0..35}`, slots `t ∈ {0..599}` (50 ms each) — D3, D16.
- `S[b,t]` — true received signal level (dB). `O[b,t] = (S[b,t] ≥ γ)` — true occupancy, D4.
- `a(t)` — band the scheduler is tuned to at slot `t`.
- `Y(t) ∈ {0,1}` — the receiver's **declared** detection at slot `t`, i.e. what the scheduler
  actually observes. `Y` may differ from `O[a(t),t]` because of noise: that difference is
  exactly what P<sub>d</sub> and P<sub>fa</sub> measure.
- `E` — set of emitters detectable in the scenario. `on_e` — emitter `e`'s activity start.
- `first_e` — slot of the first true intercept of `e`; `∞` if never intercepted.

**Every metric below is computed identically for every scheduler, including baselines.**

---

## 2. Model-level metrics — do we believe the environment?

Computed by running a **known** schedule (Turing's own sweep) through the environment and
comparing to the actual recordings. These validate the simulator, not any scheduler.

| Metric | Definition |
|---|---|
| **% correct predictions** | Fraction of (band, slot) cells where the environment's predicted detection matches the recorded scan data, under Turing's own schedule. |
| **Average intercept-time error** | Mean absolute difference between per-emitter first-intercept time predicted by the environment and the value measured from the actual scan recording. |

> **PS reading.** *"The model should enable prediction of intercept time and interception ratio
> of a scanning receiver…"* assigns prediction to the **system model**. That is why these two
> live here and not in the scheduler family.

---

## 3. Receiver-level metrics — how good is the detector?

Computed once, at environment freeze time, by sweeping γ. Reported as an ROC curve plus the
chosen operating point. **Identical for every scheduler** (D15, D21).

| Metric | Definition |
|---|---|
| **P<sub>d</sub>** | `P(Y=1 | O=1)` — declared a hit given the cell was truly occupied. |
| **P<sub>fa</sub>** | `P(Y=1 | O=0)` — declared a hit given the cell was truly empty. |
| **Sensitivity** | The signal level at which P<sub>d</sub> reaches a stated value (e.g. 0.9) at the operating P<sub>fa</sub>. Quoted *with* both, never alone. |

Only cells the receiver actually looked at contribute — these are per-look conditional
probabilities, not properties of the whole grid.

---

## 4. Scheduler-level metrics — the actual comparison

This is the table that compares schedulers. **The first two are the PS's primary objectives and
must always be reported together** (D14).

| Metric | Definition | Direction |
|---|---|---|
| **Interception ratio** | Intercepted illuminations ÷ total illuminations — a pulse counts as intercepted iff the scheduler's tuned band window contained it at its ToA. **Per illumination, not per dwell.** | higher better |
| **Censored mean intercept time** | `mean over e ∈ E of (first_e − on_e)`, with **`first_e` set to episode end (30 s) for any emitter never intercepted.** | lower better |
| **Emitter coverage** | `|{e : first_e < ∞}| ÷ |E|` — supporting diagnostic, always printed beside the two above. | higher better |
| **Average intercept rate** | Distinct emitter-intercepts per second of episode. | higher better |
| **Average reward / cost** | The scheduler's own accumulated reward, reported as a scalar. Comparable only within a reward family; never used to rank across different rewards (D7). | — |

### Two traps, both measured on real data (D14)

1. **Per-dwell hit rate is misleading.** A camper that parks on the busiest band scores 85–90%
   per-dwell and looks near-optimal, while capturing only 30% of emitters. The per-illumination
   definition above is the honest one and is what the literature uses.
2. **Uncensored intercept time rewards not looking.** Averaged over only *found* emitters, the
   camper appears *faster* (6.5 s) than round-robin — because it only ever finds the loudest
   emitters. Censoring at episode end reverses it correctly (23.8 s vs 9.7 s).

**Rule: never publish interception ratio without coverage and censored intercept time beside
it.** A single scalar hides the entire problem.

---

## 5. Baseline ladder

Every scheduler runs on identical scenarios and seeds, at the same frozen γ (D13).

| # | Baseline | Purpose |
|---|---|---|
| 1 | **Random** | Lower bound; also the coverage-heavy extreme. |
| 2 | **Round-robin** | The open-loop strategy the PS explicitly targets. The floor to beat. |
| 3 | **Turing reference sweep** | The dataset's own schedule — makes our numbers comparable to the recordings. |
| 4 | **Greedy static (camper)** | The degenerate exploit. Included *precisely* to show a single metric can be gamed. |
| 5 | **Recency / activity heuristic** | Simple adaptive benchmark. |
| 6 | **Apfeld adaptive** | Published non-learning adaptive strategy (`docs/reference/scheduling/paperSSPD (1).pdf` §II). The serious bar. |
| 7 | **RL scheduler** | Ours. |
| — | **Pulse-capture oracle** | Ceiling. Not a baseline — a reference line. |

Beating round-robin is the minimum. **Beating Apfeld is the claim worth making.**

### The target, quantified

Measured on the 47 train scenarios (D14 amendment): the camper wins interception ratio (57.4%)
and loses intercept time (23.78 s); round-robin wins intercept time (9.72 s) and loses ratio
(5.5%). **No trivial strategy is good at both.** The RL scheduler's job is to Pareto-dominate
that pair — approach round-robin's intercept time while multiplying its interception ratio.

---

## 6. Validation gates — pass before any scheduler number is quoted

| # | Gate | Why it matters |
|---|---|---|
| **1** | **Out-of-sample prediction.** Build truth from **stare only**, replay Turing's scan schedule, compare predicted detections against the **actual scan recordings** — data never used in construction (D17). | The only gate that is a genuine prediction rather than a fit. If one gate is run, run this one. |
| **2** | **Per-band structure.** Band-level interception ratios match the recordings, not just the aggregate ~35% non-empty dwell rate. | An aggregate can match while the structure is wrong. |
| **3** | **Theory.** A controlled periodic case matches Köksal's closed-form intercept time and probability of intercept (`docs/reference/scheduling/optimumsearch.pdf` ch. 3.2, 6.1). | Independent of the dataset entirely. |
| **4** | **Extremes.** `config_81` (2 emitters) and `config_921` (99) both behave sensibly. | Catches failures that averages hide. |

On pass, **freeze**: band geometry, γ, truth pipeline, metric definitions.

---

## 7. Protocol

1. **Develop and tune** on the 47 train scenarios only.
2. **Validate** the environment (gates 1–4). No scheduler result is quoted before this.
3. **Freeze** the environment and publish the receiver ROC.
4. **Compare** all schedulers on identical scenarios and seeds; report the full
   scheduler-level table, never a single metric.
5. **Ablate** — the baseline ladder is the ablation: it shows which component earns the gain.
6. **Test once.** The 45 held-out pairs are touched a single time, at the end, after the system
   is frozen (D8). Record that use.

### Reporting rules

- Report **distributions and per-scenario results**, not just a grand mean — scenario difficulty
  spans 2 to 99 emitters.
- Report **repeated-run statistics** (multiple seeds) with spread, not a single run.
- **Never report bare accuracy.** With sparse occupancy, "predict nothing" scores well and is
  operationally useless.
- State the operating point (γ, P<sub>fa</sub>) alongside any scheduler table.

---

## 8. Output artefacts the environment must emit

Evaluation is only possible if the environment logs these (see `ENVIRONMENT_SPEC.md` §Outputs):

1. **Episode log** — per slot: time, band chosen, dwell length, declared hit `Y`, true occupancy
   `O`, pulse count, peak level.
2. **Emitter table** — per emitter: activity window, first/last intercept slot, intercept count,
   bands seen in. *(Everything in §4 is computable from artefacts 1 and 2 alone.)*
3. **Waterfall render** — the 36×600 grid as a frequency-vs-time heatmap with the scheduler's
   path and hits overlaid. The same picture drawn from the raw Turing recording should match:
   the standard ESM operator view, and the fastest way to see that the environment is sane.
4. **`metrics.json`** — the three families, one file per run.

---

*Provenance: every measured figure quoted here traces to a command run against the Turing HDF5
files and is recorded with its evidence in `docs/project/DECISIONS.md`. Where this file and
`docs/project/SIH26055_PROBLEM_STATEMENT.md` disagree, the PS wins.*
