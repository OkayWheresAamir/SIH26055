# Decisions

Every decision this project has taken, why, and what evidence backed it. Written so that
months later — or while building the PPT — anyone can reconstruct *why* the system looks the
way it does, and so no settled question gets silently reopened.

**Status vocabulary**

| Status | Meaning |
|---|---|
| `SETTLED` | Decided. Do not reopen without new evidence. Build on it. |
| `PROPOSED` | Recommendation with evidence, **awaiting a human decision.** `CLAUDE.md` requires this for anything shaping the environment, receiver, ground truth, reward, evaluation or scheduler. |
| `OPEN` | Genuinely undecided. Needs discussion or research. |
| `CLOSED` | Was a question, turned out not to need a decision. Recorded so it stays closed. |

**Evidence vocabulary.** *Measured* = a command run against the HDF5 files, reproducible.
*Sourced* = quoted from a document in `docs/`, cited. *Reasoned* = follows from the above; no
new evidence. Anything else is unverified and says so.

---

## D1 — The environment is generative, not a replay of the recordings

**Status:** `SETTLED` (2026-08-28)

We build a simulated RF environment carrying its own truth state, and use the Turing
recordings to construct and validate it. We do not replay pulse logs as if they were the world.

**Why.** The problem statement requires it outright: *"A system model for the receiver needs to
be developed with measurements obtained from a simulated RF environment which has truth
information on status of emitters in each band and at each time slot."* Independently, a replay
cannot answer counterfactuals — "what would the receiver have seen in band 12 at t=7.3 s if it
had looked there?" — and answering those is the entire point of having an environment.

**Evidence.** Sourced: `docs/SIH26055_PROBLEM_STATEMENT.md`. Reached independently before the
PS was available, then confirmed by it.

**Consequence.** `docs/PROJECT_ARCHITECTURE.md` §3 was right; the pulse table is not a complete
description of the hidden world.

---

## D2 — Truth is built from metadata for *who/what/where*, and from the recordings for *when*

**Status:** `SETTLED` (2026-08-28)

Emitter identity, frequency, PRI, pulse width, beam geometry, position and power come from
`metadata/transmitters`. Activity windows — when each emitter is transmitting at all — are
recovered from the recordings.

**Why.** The metadata is complete but has no notion of when an emitter turns on. The recordings
have that information but are incomplete in other ways. Together they cover the state we need.

**Evidence.** Measured: transmitter metadata is byte-identical between scan and stare across all
47 train pairs (every attribute and dataset compared recursively). Measured: 63 of 82 emitters
across six scenarios (76.8%) transmit in a single continuous window — e.g. `config_2` emitter 3
runs 4.14 s to 16.98 s with no gap over 100 ms — and nothing under
`metadata/transmitters/transmitters_3` encodes that window.

**Known limit, to be stated in the write-up.** An emitter that neither recording ever captured
is invisible to us. We do not claim otherwise.

---

## D3 — Adopt Turing's receiver geometry unchanged

**Status:** `SETTLED` (2026-08-28, reaffirmed 2026-08-29)

36 bands on 500 MHz centres from 250 MHz to 17750 MHz, each ±500 MHz wide, on Turing's native
non-uniform dwell schedule (seven 100 ms, twenty-nine 50 ms, 2.150 s per sweep). A band keeps
its native dwell length when our scheduler selects it.

**Why.** The reference comparison is the project's value proposition. Changing the geometry
breaks comparability with the Turing sweep and with every baseline number measured against it.

**Evidence.** Measured: 99.985% of 4,393,233 scan pulses fall within ±500 MHz of the dwell
centre active at their ToA, with a hard edge at exactly 500 MHz and zero pulses in [500, 520).

**Note — bands overlap by half, and the dataset paper disagrees.** The paper says the receiver
sweeps *"in 500 MHz steps and 500 MHz bandwidth"*, implying a disjoint tiling. The files say
otherwise: pulses spread evenly across ±500 MHz (52.63% within ±250; a 500 MHz-total window
would give ~100%). Effective window is 1000 MHz on 500 MHz centres. Most likely
`bandwith_mhz = 500` is applied as a half-width in their generator. Per the `CLAUDE.md`
authority table, **the files win.**

**Interpretive consequence.** Two *adjacent* band choices share half their spectrum, so for
neighbours "picked a different band" is not "looked somewhere else". For non-adjacent choices it
genuinely is. This affects how we read exploration behaviour; it does not invalidate the setup.

---

## D4 — Environment state is a per-(band, slot) signal level; the PS's binary occupancy is derived from it by threshold

**Status:** `PROPOSED` — **this is the decision to make next.** (2026-08-29)

The environment holds a continuous received-signal level for every (band, time slot) cell. The
binary transmission / non-transmission status the PS asks for is produced by thresholding that
level. The threshold is an explicit receiver parameter.

**Why this rather than a native binary model.** Three independent lines agree:

1. **The data is not a window function.** Measured on `config_2` stare: folding each emitter's
   pulse times at its rotation period, peak received amplitude swings **53–62 dB** across one
   revolution in a smooth antenna-pattern shape, and **all 36 phase bins contain pulses**. The
   emitter is detectable at every beam angle, just far weaker off-boresight. A binary
   illuminating / not-illuminating model cannot represent that.
2. **The dataset was generated that way.** The TSRD paper §II: ambient noise −100 dB, received
   amplitude falling quadratically with distance, and *"the probability of pulse detection
   increases the more distinct the signal is from the noise floor."*
3. **The literature made this exact correction.** Apfeld, Charlish & Koch (2016),
   `docs/paperSSPD (1).pdf`, argue the window-function model used by Clarkson, Köksal and others
   is *"rather simplistic"* and replace it with SNR time series precisely so sidelobe intercepts
   are representable. Our measurement is that correction, reproduced independently in Turing data.

**What it buys us.** Probability of false alarm and sensitivity — both PS-mandated metrics —
are only definable as a threshold on a noisy continuous quantity. A natively binary environment
cannot be wrong, so its Pfa is identically zero and the metric carries no information. This
resolves what was tracked as Q4.

**It still satisfies the PS.** *"The status of environment for each frequency band at each time
step can be recorded as a transmission or a non-transmission"* — it can, and is. The PS
constrains the state we expose to the scheduler, not the machinery underneath.

**Feasibility measured.** Prototype grid built from stare recordings: 36 bands × 600 slots of
50 ms over 30 s. Occupancy is a smooth function of threshold — `config_2` runs 25.0% at −120 dB
down to 4.9% at −80 dB; `config_921` 48.8% to 16.4%; `config_59` 6.3% to 1.3%. The set of bands
that are ever active stays stable across thresholds (20, 21 and 8 of 36 respectively), so the
structure is robust while the operating point is tunable.

**What needs deciding by a human:** whether to accept this, and where the default threshold
sits. Everything in D5 and D6 follows from it.

---

## D5 — A hit is: tuned to the band, during the slot, with signal above threshold

**Status:** `PROPOSED`, follows from D4 (2026-08-29)

**Why.** It is the PS's own definition once D4 supplies the occupancy: *"the model should then
be trained based on hits and misses."* No extra machinery.

**Still open inside this:** whether all hits score equally, or whether first-interception of a
previously unseen emitter is worth more. The PS names two objectives that pull apart —
*minimise intercept time* favours weighting discovery, *high interception rate* favours raw
volume. **Empirical warning from the literature:** Apfeld et al. found their Random baseline had
the **best** percentage of radars detected at least once while having the **worst** efficiency —
pure exploration wins coverage, exploitation wins efficiency. Whatever we choose will move those
two metrics in opposite directions, and we should report both.

---

## D6 — Pfa and sensitivity come from the detection threshold

**Status:** `PROPOSED`, follows from D4 (2026-08-29)

Sweeping the threshold traces an ROC curve; that is where Pd, Pfa and sensitivity come from,
and it gives us a principled operating point rather than an arbitrary one.

**Evidence.** Reasoned from D4. Enabled by the measured threshold sweep in D4.

---

## D7 — The reward is a hyperparameter, selected on the PS's own metrics

**Status:** `SETTLED` (2026-08-29)

Define two or three candidate reward functions, train under each, then score all of them on
*intercept time* and *interception rate*. Whichever produces the best PS-mandated metrics wins.

**Why.** Scoring a reward by results it itself generated is circular. Intercept time and
interception rate are mandated by the PS and are reward-independent, so they can judge a reward
from outside. Reward is decided *before* training and *chosen* after comparison.

**Evidence.** Reasoned. Metric list sourced from `docs/SIH26055_PROBLEM_STATEMENT.md`.

---

## D8 — Held-out test set: 45 pairs, rule fixed in advance

**Status:** `SETTLED` (2026-08-28)

45 scan/stare pairs from the test split, in `data/turing/*/test_*/`. Selection rule and config
ids recorded in `docs/RESEARCH_MAP.md`. **Not to be touched until the system is frozen.**

**Why not all 250?** Train was chosen by stratified sampling on stare file size. If test were
the full split, train and test would have different size distributions and any performance gap
could be distribution shift rather than a genuine generalisation failure. Matching the selection
rule keeps them comparable. What makes a held-out set honest is that the rule was fixed before
any result was seen — not its size. 45 scenarios is ample for a confidence interval.

---

## D9 — `sensitivity_dbm` is not a detection threshold

**Status:** `CLOSED` (2026-08-29)

**Why it came up.** The receiver attribute reads −110.0, yet 4.49% of scan pulses and 4.01% of
stare pulses sit below it, with no cliff in the amplitude histogram at −110 or at −120
(−110 minus the 10 dB `gain_db`).

**Resolution.** The TSRD paper §II states detection is probabilistic against a −100 dB ambient
noise floor. There was never meant to be a hard threshold. The field is configuration that does
not gate the Amplitude column. Our measurement was correct; the dataset card simply did not
document the model. **We define our own threshold (D4); we do not inherit theirs.**

---

## D10 — Scan and stare are not nested, and that is a property, not a defect

**Status:** `CLOSED` (2026-08-29)

**Measured:** across the 47 train pairs, 209 emitter-instances appear in scan but not stare, and
174 the other way. Neither is a superset.

**Resolution.** TSRD paper §II: *"Pulses were dropped when the Rx was not tuned to the correct
frequency band, when the Tx was too far for detection, or when the pulse width dropped below a
threshold (0.0069µs)."* Verified — minimum pulse width across 72,031,672 train pulses is exactly
0.006900 µs, with none below. These rules apply to **both** modes, plus random drops. Stare is
an oracle in *coverage*, not in *detection*. The HF dataset card's "detecting all signals" is
wrong; the paper it summarises is not.

---

## D11 — No additional datasets

**Status:** `SETTLED` (2026-08-29)

Turing only.

**Why.** Our difficulties were documentation gaps, not data gaps, and they are now closed
(D9, D10). Of the alternatives considered: the **Radar Emitter Database** by John C. Wise MBE
([radars.org.uk](https://www.radars.org.uk/)) — the "JC Wise" in the PS's dataset line — is a
commercial reference product of 16,500 emitter identities sold with a handbook, not a
downloadable training set; it is a parameter reference of the kind Turing used to set its
emitter ranges. **RadioML 2018** is communications modulation classification on IQ streams —
wrong problem, wrong data type, and the TSRD paper explicitly rejects that family as
*"insufficient for congested radar environments"*. Mixing sources would create exactly the
environment-mismatch problem we would be trying to avoid.

---

## D12 — Clustering is not the deliverable

**Status:** `SETTLED` (2026-08-29)

Emitter clustering may later serve as a *supporting* component — for example, estimating how
many distinct emitters have been seen in a band. It is not the system we are asked to build.

**Why.** Clustering pulses by emitter is **deinterleaving**, which is what the Turing dataset
was built for and what its challenge scores. Our PS asks for a *scheduler*: *"Expected Solution:
Machine learning based Electronic Support receiver scheduler software."* Building a clusterer
would be solving a different, already-benchmarked problem.

**If we ever want it:** the TSRD paper publishes HDBSCAN baselines on raw PDWs — V-measure 0.54
stare, 0.19 scan (Table IV) — so there is a citable reference point without us doing the work.

---

## D13 — Baseline set

**Status:** `SETTLED` (2026-08-29)

Random; round-robin; **Turing's own reference sweep**; a recency/activity heuristic; and
**Apfeld's adaptive strategy** as the strong non-learning baseline. RL is compared against all
of them under identical conditions.

**Why Apfeld specifically.** It is published, directly on our problem, non-learning, and
described in enough detail to reimplement (`docs/paperSSPD (1).pdf` §II, plus its Algorithm 1).
Beating a real published adaptive strategy is a far stronger claim than beating round-robin.
Its own baselines — Random, "Active RFs", adaptive-without-tracking — give us a ladder that
maps onto the ablation study we need anyway.

---

## D14 — The problem is only hard if we measure it with both metrics at once

**Status:** `SETTLED` as a finding (2026-08-29). Measured, and it should shape the reward.

A trivial scheduler that picks the single busiest band at t=0 and **never moves again** achieves
**90.3%** hit rate against an oracle's 91.9%, and beats round-robin on **47 of 47** scenarios.
On hit rate alone, this problem is close to solved by a scheduler that does nothing.

It is saved by the second metric. Measured across all 47 train scenarios on a stare-derived
occupancy grid (36 bands × 600 slots of 50 ms):

| Scheduler | Hit rate | Emitter coverage | Mean time to first intercept |
|---|---|---|---|
| Oracle (knows the future) | 91.9% | — | — |
| Greedy static (camp on busiest band) | **90.3%** | **27.9%** | 6.47 s |
| Round-robin | 33.3% | **92.8%** | 8.70 s |
| Random | 33.3% | 89.3% | 8.96 s |

**Why this matters more than anything else measured so far.**

1. **Optimising hit rate alone produces a scheduler that camps on one band and ignores 72% of
   the emitters.** That is operationally useless and would score well on a naive metric.
2. **Optimising coverage alone gives you round-robin** — which is the open-loop baseline the PS
   is explicitly asking us to beat.
3. **Nothing in the table is good at both.** Greedy has 2.7× round-robin's hit rate and less
   than a third of its coverage. That gap is exactly the space an adaptive scheduler should
   occupy, and it is now measured rather than assumed.
4. It reproduces, in Turing data, precisely what Apfeld et al. found: their Random baseline was
   best on radars-detected-at-least-once and worst on efficiency. Two independent datasets, same
   tension.
5. It vindicates not promoting the 35.7% figure into the environment definition — a single
   scalar hides all of this.

**Consequences.**
- The reward (D7) must express both, or be selected against both.
- **Never report hit rate, interception ratio or efficiency without coverage beside it.**
- The oracle row is our ceiling; round-robin is the floor the PS names. Both go in every table.

**A metric trap inside this result.** Greedy's mean time-to-first-intercept (6.47 s) looks
*better* than round-robin's (8.70 s) — but it is conditioned on the 27.9% of emitters greedy
ever finds, which are the loudest and easiest. Time-to-intercept averaged over *found* emitters
rewards not looking. Either average over all detectable emitters with a censoring penalty for
misses, or report coverage alongside it every time.

**Caveats on these numbers.** Truth is stare-derived, so emitters below 500 MHz that only scan
sees are missing (D10). Occupancy is "≥1 pulse in band during slot" with no detection threshold,
which is the permissive end — D4's threshold will lower every row. Band assignment for pulses in
the overlap between adjacent bands takes the last matching band rather than both. Directional
conclusions are robust to all three; the exact figures are not final.

---

# Evaluation plan

What we will measure, why, and the traps. Drafted 2026-08-29; not yet exercised against
anything, so treat as a plan rather than a protocol.

### The seven metrics the PS mandates

Probability of detection · probability of false alarm · sensitivity · average intercept rate ·
average reward/cost · percentage of correct predictions · average intercept time error. Plus
intercept time and interception ratio, named separately in the same paragraph. **These are not
ours to choose** — reporting a different set is a failure to answer the PS.

### How each becomes computable

| Metric | Where it comes from | Depends on |
|---|---|---|
| Pd, Pfa, sensitivity | Sweeping the detection threshold — an ROC over the signal grid | D4 |
| Interception ratio | Fraction of occupied band-slots the scheduler was tuned to | D4, D5 |
| Intercept time | Delay from an emitter becoming active to its first detection | D2 activity windows |
| Avg intercept time error | Predicted minus actual intercept time | a scheduler that ranks bands |
| % correct predictions | Was the top-ranked band actually occupied | ranking, not a separate model |
| Avg reward / cost | The chosen reward, reported as a scalar | D7 |

### Three traps to design around

**A "predict nothing" model can score well.** Flagged in
`docs/Smart Spectrum Surveillance ... [BASICS].pdf` §6: with sparse occupancy, always predicting
"no transmission" gives high accuracy and zero operational value. Our own measurement shows how
sparse — thresholded occupancy runs 1.3%–48.8% depending on scenario and threshold. **Never
report bare accuracy.** Use interception ratio and intercept time, which a null predictor cannot
game.

**Coverage and efficiency move in opposite directions.** Apfeld et al. measured Random as best
on "percentage of radars detected at least once" and worst on efficiency. Reporting only one
lets any scheduler look good. **Report both, always, on the same table.**

**A per-scenario mean hides everything.** Scenario difficulty spans a huge range — our 47 hold
2 to 99 transmitters. Report distributions and per-scenario results, not just a grand mean.

### Protocol

Develop and tune on the 47 train scenarios. Validate the environment (below) before any
scheduler number is quoted. Freeze the environment, then run every scheduler on identical
scenarios and seeds. Touch the 45 held-out test pairs **once**, at the end, and record that use.

### Environment validation gates

Before any scheduler result is believed:

1. **Reproduce the reference.** Run Turing's own sweep inside our environment; recover the
   measured **35.7%** non-empty dwell rate (8,422 of 23,594 visits across 47 scenarios).
2. **Reproduce per-band structure.** Band-level interception ratios should match the recordings,
   not just the aggregate.
3. **Check against theory.** For a controlled periodic case, intercept time and probability of
   intercept should match Köksal's closed forms (`docs/optimumsearch.pdf` ch. 3.2, 6.1).
4. **Sanity-check the extremes.** `config_81` (2 emitters) and `config_921` (99) should behave
   sensibly at both ends.

Gate 1 is the one that matters most: it is a single number, measured from real data, that our
environment either reproduces or does not.
