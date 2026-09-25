# Figures of merit — definitions and formulas

**The seven measurements `SIH26055_PROBLEM_STATEMENT.md` names, specified so they can be
implemented correctly against any model, in any repository.** This file is the definition;
`EVALUATION.md` remains the authority on the *scheduler* metrics (interception ratio, censored
intercept time, coverage) and on the reporting protocol. Where this file and `EVALUATION.md`
disagree about a shared quantity, `EVALUATION.md` wins.

Written 2026-09-20. Every formula is in symbols, not in our numbers, so it ports.

---

## How to produce them

```bash
python -m rfenv.compare --figures-of-merit          # adds figures_of_merit.md to the run
python -m rfenv.compare --seeds 3 --sampled 10 --figures-of-merit --out runs/fom
```

Writes `figures_of_merit.md` beside `comparison.md` in the output directory, in the
three-block layout of §8, and puts the same figures into `summary.json` under
`meta.figures_of_merit` so they travel with the run rather than only with the report.

- Blocks A and B are computed over the **stare replay grids only**, never the sampled
  scenarios — a sampled scenario has no underlying scan recording to predict against, so
  #6 and #7 are undefined on it and P<sub>d</sub>'s population would silently change.
- Block C uses whatever scenario set the run itself used.
- The two model-level figures come from `rfenv.validate.gate1` and
  `rfenv.validate.intercept_time_error`; P<sub>d</sub>/P<sub>fa</sub>/sensitivity from
  `rfenv.receiver.operating_point`. Nothing is transcribed.

Implementing this somewhere else: §1–§7 below are the specification, §8 the presentation,
§9 the rules that apply to all seven.

---

## 0. Read this first

The PS sentence these come from:

> *"...building up figures of merit for interception performance such as probability of detection,
> probability of false alarm, sensitivity, Avg intercept rate, Avg Reward / cost function,
> percentage of correct predictions and average intercept time error."*

**Interception ratio and intercept time are named separately, in the next sentence, and are not
part of this seven.** They are the headline objectives (`EVALUATION.md` §4). Do not fold them in.

**Five of the seven do not vary with the scheduler, and that is correct.** Three are properties of
the receiver at a frozen threshold; two are properties of the simulated environment. Only
**average intercept rate** and **average reward** are results about a search strategy.

| # | Metric | What it characterises | Varies by scheduler? |
|---|---|---|---|
| 1 | Probability of detection, P<sub>d</sub> | the receiver | no |
| 2 | Probability of false alarm, P<sub>fa</sub> | the receiver | no (analytic) |
| 3 | Sensitivity | the receiver | no (analytic) |
| 4 | Average intercept rate | **the scheduler** | **yes** |
| 5 | Average reward / cost | **the scheduler** | **yes** |
| 6 | Percentage of correct predictions | the environment | no |
| 7 | Average intercept-time error | the environment | no |

If a table shows five columns repeating identically down every row, the numbers are probably right
and the table shape is wrong — see §5.

**But "no" and "no (analytic)" are different answers, and P<sub>d</sub> is the one that is not
analytic.** P<sub>fa</sub> and sensitivity are closed forms in `γ`, `N₀` and `σ` alone — measured
2026-09-21, P<sub>fa</sub> is identically `1.349898e-03` over every cell population in this
repository, which is what "exact" means. P<sub>d</sub> is an **empirical average over a chosen
population**, and it is scheduler-invariant only because D33 pinned that population to one fixed
reference sweep. Average it over *"the cells this scheduler looked at"* instead and it stops being
a receiver property altogether: measured on the same 47 stare grids at the same `γ` and `σ`, a
camper parked on each grid's busiest band reports **P<sub>d</sub> = 0.916** against the reference
sweep's **0.840**, and a camper on the quietest band reports **no P<sub>d</sub> at all** (zero
occupied cells in its denominator). *"P<sub>d</sub> is a property of the threshold"* is the wrong
reason for the right answer, and it is the reason that breaks the moment someone reimplements
this.

### Symbols

| | |
|---|---|
| `Z[b,t]` | physical occupancy — is any emitter transmitting into band `b` at slot `t`? Threshold-free. The PS's *"transmission or a non-transmission"*. |
| `S[b,t]` | continuous received level in dB — peak amplitude where `Z` is true, the noise floor `N₀` where it is false. |
| `Y[b,t]` | what the receiver **declares** when it looks: `Y = 1` iff `S + n ≥ γ`. |
| `n` | noise draw, `n ~ Normal(0, σ²)`, redrawn per look. |
| `γ` | detection threshold — a receiver design parameter, never learned (D15). |
| `N₀`, `σ` | noise floor and noise standard deviation, both frozen (D42). |
| `Φ` | standard normal CDF; `Φ⁻¹` its inverse. |
| `E` | the emitter population being scored — **always state which** (§6). |

This project: `γ = −111 dB`, `N₀ = −120 dB`, `σ = 3 dB`, episode `T = 30 s`, slot `= 50 ms`.

---

## 1. Probability of detection — P<sub>d</sub>

**In words.** Given something really was transmitting and the receiver looked at it, how often does
it say so?

```
P_d = P(Y = 1 | Z = 1)
    = count(Y = 1 AND Z = 1) / count(Z = 1)        over the chosen cell population
```

**Population.** The occupied cells visited by **one fixed reference schedule** — here, the
dataset's own sweep (`PD_POPULATION = reference_sweep`, D33). Not "cells the scheduler looked at":
different schedulers look at different cells and per-cell detection probability is not uniform, so
that choice makes P<sub>d</sub> scheduler-dependent, which contradicts D21 — measured, **0.916 for
a busiest-band camper against 0.840 for the sweep** on the same 47 grids.

**Report as** a single value, *always* with the grid set it was averaged over named beside it. The
same rule over different grids gives different numbers and they are not interchangeable — re-measured
2026-09-21, 0.83951 (stare replays) / 0.84211 (stare replays + 10 sampled) / 0.85058 (scan replays)
over three different grid sets in this project (D33).

**One run directory prints two of those three.** `comparison.md` states the operating point over
every scenario the ladder actually ran, sampled ones included; `figures_of_merit.md` states it over
the stare replay grids only, because #6 and #7 are undefined on a sampled scenario. At
`--sampled 10` that is 0.8421 in one file and 0.8395 in the other, from the same command. Both are
labelled and neither is wrong — but quoting either as *"our P<sub>d</sub>"* without naming the grid
set is.

> **Trap — this one silently destroys the ROC.** Condition on `Z`, the threshold-free truth, never
> on `S ≥ γ`. Conditioning on a second copy of the signal thresholded at the same `γ` forces
> `P_d ≥ 0.5` for every `γ` by construction, so the curve is flat and carries no information.
> Against `Z` it sweeps properly — measured, P<sub>d</sub> falls 0.894 → 0.681 as `γ` goes
> −120 → −100 (D26).

---

## 2. Probability of false alarm — P<sub>fa</sub>

**In words.** When nothing is there, how often does the receiver claim otherwise?

```
P_fa = P(Y = 1 | Z = 0)

Where Z = 0 the level is exactly N₀, so Y = 1 requires n ≥ γ − N₀:

P_fa = 1 − Φ( (γ − N₀) / σ )

With γ = N₀ + 3σ:   P_fa = 1 − Φ(3) = 1.3499e−3
```

**Population.** None. It is exact and data-independent.

**Report as** the exact value. No distribution, no interquartile range — showing a spread here is a
mistake. Measure it empirically once as a unit test, then quote the closed form.

---

## 3. Sensitivity

**In words.** How faint can a signal be and still be reliably heard? A level in dB, not a
probability.

```
A signal at true level S is detected with probability
    P(detect | S) = Φ( (S − γ) / σ )

Sensitivity = the level at which that reaches a stated probability p:
    S_p = γ + σ · Φ⁻¹(p)

At p = 0.9:   S = γ + 1.2816 σ = −111 + 1.2816(3) = −107.16 dB
```

**Report as** a level in dB, **always quoted with both the P<sub>d</sub> target it refers to and the
operating P<sub>fa</sub>**. "Sensitivity = −107.2 dB" alone is meaningless — any receiver is
arbitrarily sensitive if you accept enough false alarms.

---

## 4. Average intercept rate

**In words.** How many distinct emitters does it find per second of mission?

```
rate = |{ e ∈ E : e intercepted at least once }| / T_episode
```

where an emitter counts **once**, however many times it is seen, and `T_episode` is the mission
length in seconds.

**Population.** `E` = emitters *detectable at all* in this scenario — those whose own received level
clears `γ` at some point (D27). Scoring a scheduler for missing an emitter it could never have heard
is meaningless; measured, 19% of transmitters are never detectable.

**Report as** the mean across episodes with its interquartile range. Scenario difficulty spans 2 to
99 emitters here, so a bare mean hides the distribution (`EVALUATION.md` §7).

> **Trap — distinct, not total.** Counting every re-detection turns this into a measure of how long
> you camped on a busy band, which rewards exactly the failure mode the ladder exists to expose
> (D14).

---

## 5. Average reward / cost

**In words.** The total score the scheduler accumulated under whatever reward function it was
trained with.

```
avg_reward = mean over episodes of  Σ_t r(t)
```

**Report as** the mean with its interquartile range, and **the name of the reward function printed
beside it**.

> **Trap — this is not a ranking.** Never compare it across two different reward functions. It is a
> per-family scale, not a score (D7): two rewards can rank the same schedulers identically and
> produce numbers orders of magnitude apart. Using a reward to decide which scheduler is better is
> circular — that is what the other six are for.

---

## 6. Percentage of correct predictions

**In words.** When the environment says "this look would have heard something", was it right?

This is a **model-level** metric: it asks whether the simulator reproduces the real recordings. The
PS's own wording assigns *prediction* to the system model, which is why it lives here and not with
the scheduler metrics.

**Construction.** Build the environment from one recording; replay the *other* recording's known
schedule through it; compare. The environment must never have seen the recording it is tested
against, or this is a fit rather than a prediction (D17, D37).

```
For each look in the known schedule:
    predicted = did the environment declare a hit?   (Y, with noise — not S ≥ γ)
    actual    = did the real recording hold a pulse there?

    TP = predicted 1, actual 1      FP = predicted 1, actual 0
    TN = predicted 0, actual 0      FN = predicted 0, actual 1

accuracy = (TP + TN) / (TP + FP + TN + FN)

MCC = (TP·TN − FP·FN) / sqrt( (TP+FP)(TP+FN)(TN+FP)(TN+FN) )
```

**Report as** accuracy **and MCC together**, plus the base rate.

**Decide and state the unit.** A "look" is either one time-slot cell or one whole dwell. Both are
defensible and they give different numbers. **This project scores per dwell** (D37) — scoring all
36×600 cells would put ~97% of the denominator on cells the sweep never visits, making most of the
score unfalsifiable. `EVALUATION.md` §2 said *cells* when this file was written; it was corrected
to *dwells* on 2026-09-20 (commit `ca36dec`) and the two now agree. No number moved — only the
text was wrong.

> **Trap — never report bare accuracy.** Occupancy is sparse; a model that predicts "nothing, ever"
> scores well and is useless (`EVALUATION.md` §7). MCC is not decoration here — it is what makes the
> number mean anything.

---

## 7. Average intercept-time error

**In words.** The environment predicts when each emitter would first be caught. How far off is that
from when it really was?

Model-level, like #6, and computed the same out-of-sample way.

```
For each emitter e in the stated population:

  predicted_e = first time the environment says e is intercepted,
                replaying the known schedule through the out-of-sample grid
  actual_e    = first time e is detected in the real recording

REPORT THREE NUMBERS, not one:

  (a) mean | predicted_e − actual_e |   over emitters detected on BOTH sides
  (b) agreement rate — fraction of the population where both sides agree
      on whether e was detected at all
  (c) signed mean ( predicted_e − actual_e )   — is the model early or late?
```

**Population.** State it explicitly. "Detectable emitters" (D27) and "all emitter instances" are
different sets and give materially different answers.

**Both sides must use the same detection rule.** If `predicted` requires a declared detection
(D28: tuned to the band, own level ≥ `γ`, and `Y = 1`) then `actual` must also require a detection,
not merely the presence of a pulse. An ungated `actual` fires earlier than a gated `predicted` on
every emitter whose first pulses are sub-threshold, which inflates the error systematically.

> **Trap — do not censor into the mean.** If the environment predicts an emitter is never
> intercepted, do **not** set `predicted = T_episode` and average it in. That silently mixes
> *"predicted the wrong time"* with *"predicted the wrong outcome"*, and afterwards nobody can
> decompose it.

**The two traps in this section compound, so measure them one at a time.** Re-measured 2026-09-21
over this project's 47 train pairs at seed 0, on one fixed 1,530-emitter matched population, moving
one thing at a time:

| recorded side | misses | mean \|error\| | what that number is |
|---|---|---|---|
| ungated (any pulse) | censored to 30 s | **8.42 s** | both traps at once |
| gated (`≥ γ`) | censored to 30 s | **8.05 s** | the censoring trap alone |
| gated (`≥ γ`) | excluded | **6.60 s** | what this section specifies, with 87.3% agreement |

So of that 8.42 s, **1.46 s is missed detections and 0.37 s is the mismatched detection rule.**

> **`8.42 s` is not "the censored figure" — it is the figure you get by tripping both traps.** This
> file, `EVALUATION.md` §2 and `validate.py` all described it as the censored form until
> 2026-09-21, and all three attributed the whole 1.8 s gap to missed detections. The censored-only
> figure is 8.05 s. A single number that mixes two defects cannot be attributed to either, which is
> the same argument this section makes about mixing timing with outcome — it just caught us first.

**Expect the timing error to be large, and check why before calling it a defect.** Where the two
recordings are independent simulation runs rather than two views of one world (D24), the same
emitter has different activity in each, and part of the error is that divergence. The diagnostic is
whether the *distributions* agree even when individual emitters do not. Re-measured 2026-09-21 over
the 1,295 emitters detected on both sides: predicted mean **8.50 s** against recorded **8.57 s**,
agreeing within 1.1 s at every decile from p10 to p90, with a per-emitter correlation of only
**r = 0.066**. **The environment reproduces the statistics of intercept time almost
exactly and an individual emitter's barely at all** — which is the expected behaviour for this
problem, not a modelling failure, and it is exactly what licenses comparing schedulers over
distributions.

---

## 8. How to print them

One table with all seven produces five columns that repeat identically down every row. The values
are right; the shape invites the reader to think something is broken. Print three blocks:

- **Block A — the receiver, stated once.** P<sub>d</sub>, P<sub>fa</sub>, sensitivity, with the
  operating point (`γ`, `σ`) and the cell population named on the same line.
- **Block B — the environment, stated once.** Percentage of correct predictions (with MCC and base
  rate), and intercept-time error (all three numbers).
- **Block C — the schedulers, one row each.** Average intercept rate and average reward, each as a
  mean over its interquartile range, with the reward function named in the header.

This also makes the honest point visible rather than buried: five of the seven characterise the
*system*, and only two are a verdict on the search strategy.

---

## 9. Rules that apply to all seven

1. **Name the population, every time.** A figure without its population is not quotable.
2. **Name the scenario set and the seeds.** Absolute values move with both — measured, ~20% between
   scenario sets in this project.
3. **Report means over interquartile ranges** for anything with a distribution. P<sub>fa</sub> and
   sensitivity have none; do not invent one.
4. **Re-derive, never transcribe.** Every figure comes from the code that computes it, in the run
   that prints it.
5. **Label reference lines.** Anything that reads ground truth is a ceiling, not a competitor
   (`EVALUATION.md` §5).
6. **Sanity-check the analytic two** against their closed forms before trusting a run. If
   P<sub>fa</sub> or sensitivity drift, the operating point has moved underneath you.
