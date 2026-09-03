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
- `Z[b,t]` — physical occupancy: is any emitter transmitting into this cell? Threshold-free,
  and the PS's binary transmission/non-transmission status (D26).
- `S[b,t]` — true received signal level (dB); the noise floor `N₀` where `Z` is false (D4).
- `a(t)` — band the scheduler is tuned to at slot `t`.
- `Y(t) ∈ {0,1}` — the receiver's **declared** detection at slot `t`, i.e. what the scheduler
  actually observes: `Y = 1` iff `S[a(t),t] + n ≥ γ`, `n ~ N(0, σ)`. `Y` may differ from
  `Z[a(t),t]` because of noise: that difference is exactly what P<sub>d</sub> and
  P<sub>fa</sub> measure.
- `E` — emitters with a non-empty **detectable activity interval**, i.e. whose own received
  level clears γ at some slot (D27). Not every transmitter in the metadata: 19.0% of train
  transmitters never appear in either recording. `on_e` — the start of that interval.
- `first_e` — slot of the first true intercept of `e`; `∞` if never intercepted. An
  intercept of `e` at slot `t` requires **all three** (D5 ∧ D27, restated as D28): the
  scheduler was tuned to a band `e` puts pulses into (`a(t)` ∈ bands of `e` at `t`); **`e`'s
  own** received level in that cell clears γ; and the receiver declared `Y(t) = 1`. The
  own-level clause is what stops a quiet emitter inheriting a loud neighbour's detectability —
  the same rule that defines `on_e`, so numerator and denominator agree and `first_e ≥ on_e`
  always. `Y` itself is declared on the **combined** `S[a(t),t]`, because that is all a real
  receiver has (D28).

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
| **P<sub>d</sub>** | `P(Y=1 | Z=1)` — declared a hit given an emitter really was transmitting there. |
| **P<sub>fa</sub>** | `P(Y=1 | Z=0)` — declared a hit given the cell was truly empty. |
| **Sensitivity** | The signal level at which P<sub>d</sub> reaches a stated value (e.g. 0.9) at the operating P<sub>fa</sub>. Quoted *with* both, never alone. |

These are per-look conditional probabilities, not properties of the whole grid.

**The cells they are averaged over: the reference-sweep population** — the occupied cells
Turing's own dwell schedule looks at (D33, `SETTLED`; `PD_POPULATION` in `rfenv/constants.py`).
"Cells the receiver actually looked at" cannot be the answer as written: different schedulers
look at different cells, and per-cell detection probability is not uniform (`Φ((S−γ)/σ)`, D29),
so that population would make P<sub>d</sub> scheduler-dependent and contradict D21. Turing's
schedule never varies, so this population is per-look *and* scheduler-independent. Every ROC is
reported with its population stated on it.

**Conditioned on `Z`, not on `(S ≥ γ)`** (D26). Referencing P<sub>d</sub> to a second copy of
`S` thresholded at the same γ is degenerate — it forces `P_d ≥ 0.5` for every γ and the curve
can never sweep. Against threshold-free `Z` it does: measured over the 47 train configs,
P<sub>d</sub> falls as γ rises. At the default operating point `γ = N₀ + 3σ = −111 dB`:
**P<sub>fa</sub> = 1.35e−3** (exact — it is `1 − Φ(3)`) and **sensitivity −107.2 dB**
(= `γ + 1.2816σ`); both re-run 2026-09-03 and confirmed.

**P<sub>d</sub> = 0.851** at γ = −111 over the reference-sweep population (D33). The previously
quoted 0.822 is withdrawn: it matched no population, and the 2026-09-03 re-run gives 0.819 over
stare-replay cells and 0.837 over scan-replay cells, so the figure depends entirely on a choice
that had never been stated. The sweep's *shape* is unaffected by it, and the ROC — not the single
point — remains the deliverable (D15).

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

**Rewards are judged from outside, and they read what they like** (D29). A reward function is a
training-time construct — training is offline and the policy is frozen before deployment — so it
may read truth-side state (`Z`, per-emitter own levels, `first_e`). Only the *observation* carries
the deployability constraint. Note that D28 puts the first two metrics on opposite sides of that
line: **censored intercept time requires `Y = 1`, interception ratio does not**, so no single
reward is aligned with both, and the D7 comparison has to resolve a Pareto front rather than a
scalar. **No reward can move P<sub>d</sub> or P<sub>fa</sub>** (§3, D15, D21) — penalising false
alarms prices a wasted dwell, it does not improve the receiver.

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
| **1** | **Out-of-sample prediction.** Build truth from **stare only**, replay Turing's scan schedule, compare predicted detections against the **actual scan recordings** — data never used in construction (D17). **Pre-gate measurement, convention unrecorded — not yet a gate result.** A 2026-09-01 scratch script reported accuracy 86.19%, precision 87.87%, recall 71.14%, MCC 0.694, per-band r = 0.940 at γ = −110 (note: *not* the frozen γ = −111). Re-run from `rfenv` on 2026-09-03 across four comparison conventions, none reproduces those figures exactly; the range is **accuracy 83.5–86.0%, precision 88.5–89.5%, recall 68.2–69.5%, MCC 0.66–0.69, per-band r ≈ 0.93** against a 35.70% base rate, at the frozen γ. Directionally the gate passes. **`validate.py` must define the convention in code, and whatever it returns becomes the number.** Known limitation, not a defect: band 0 (250 MHz) is 59.12% occupied in the recordings and 0.00% predicted, because stare cannot see below 500 MHz (D10). | The only gate that is a genuine prediction rather than a fit. If one gate is run, run this one. This is also why no physics signal model is fitted to these same recordings (D25). |
| **2** | **Per-band structure.** Band-level interception ratios match the recordings, not just the aggregate. The ~35% dwell rate is no longer a γ calibration (D23) — it is a pipeline self-consistency test: grid built from the scan recording, replayed on the schedule that produced it, thresholded not at all. Measured **35.403% replayed against 35.700% recorded.** | An aggregate can match while the structure is wrong. |
| **3** | **Theory.** A controlled periodic case matches Köksal's closed-form intercept time and probability of intercept (`docs/reference/scheduling/optimumsearch.pdf` ch. 3.2, 6.1). | Independent of the dataset entirely. |
| **4** | **Extremes.** `config_81` (2 emitters) and `config_921` (99) both behave sensibly. | Catches failures that averages hide. |

On pass, **freeze** everything in `rfenv/constants.py`: band geometry, slot clock, native dwell
lengths, truth pipeline, `N₀`, `σ`, γ, metric definitions, and the scenario sampling
distribution (D25).

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
   `Z`, pulse count, peak level.
2. **Emitter table** — per emitter: detectable activity interval (D27), first/last intercept
   slot, intercept count, bands seen in. *(Everything in §4 is computable from artefacts 1 and 2 alone.)*
3. **Waterfall render** — the 36×600 grid as a frequency-vs-time heatmap with the scheduler's
   path and hits overlaid. The same picture drawn from the raw Turing recording should match:
   the standard ESM operator view, and the fastest way to see that the environment is sane.
4. **`metrics.json`** — the three families, one file per run.

---

*Provenance: every measured figure quoted here traces to a command run against the Turing HDF5
files and is recorded with its evidence in `docs/project/DECISIONS.md`. Where this file and
`docs/project/SIH26055_PROBLEM_STATEMENT.md` disagree, the PS wins.*
