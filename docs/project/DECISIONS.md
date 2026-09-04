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

**Evidence.** Sourced: `docs/project/SIH26055_PROBLEM_STATEMENT.md`. Reached independently before the
PS was available, then confirmed by it.

**Consequence.** `docs/project/PROJECT_ARCHITECTURE.md` §3 was right; the pulse table is not a complete
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

**Status:** `SETTLED` — **accepted by the team 2026-09-01.**

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
   `docs/reference/scheduling/paperSSPD (1).pdf`, argue the window-function model used by Clarkson, Köksal and others
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

**Accepted 2026-09-01.** The architecture is settled. The numeric value of γ is not a separate
judgement call — it is produced by the calibration procedure in D15 and reported as a full ROC
sweep, so no arbitrary constant is ever chosen by hand. D5, D6 and D15 settle with it.

---

## D5 — A hit is: tuned to the band, during the slot, with signal above threshold

**Status:** `SETTLED` with D4 (2026-09-01)

**Why.** It is the PS's own definition once D4 supplies the occupancy: *"the model should then
be trained based on hits and misses."* No extra machinery.

**The sub-question this raised is now closed — see D29.** "Whether all hits score equally, or
whether first-interception of a previously unseen emitter is worth more" became D7 candidate 3
(`+1` per first intercept of an emitter), given a precise three-clause definition by D28. It is
no longer an open question and `RESEARCH_MAP.md` no longer lists it as one. The original framing
is kept below because the tension it names is real and still governs how the candidates are
judged. The PS names two objectives that pull apart —
*minimise intercept time* favours weighting discovery, *high interception rate* favours raw
volume. **Empirical warning from the literature:** Apfeld et al. found their Random baseline had
the **best** percentage of radars detected at least once while having the **worst** efficiency —
pure exploration wins coverage, exploitation wins efficiency. Whatever we choose will move those
two metrics in opposite directions, and we should report both.

---

## D6 — Pfa and sensitivity come from the detection threshold

**Status:** `SETTLED` with D4 (2026-09-01)

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

**Evidence.** Reasoned. Metric list sourced from `docs/project/SIH26055_PROBLEM_STATEMENT.md`.

---

## D8 — Held-out test set: 45 pairs, rule fixed in advance

**Status:** `SETTLED` (2026-08-28)

45 scan/stare pairs from the test split, in `data/turing/*/test_*/`. Selection rule and config
ids recorded in `docs/project/RESEARCH_MAP.md`. **Not to be touched until the system is frozen.**

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
described in enough detail to reimplement (`docs/reference/scheduling/paperSSPD (1).pdf` §II, plus its Algorithm 1).
Beating a real published adaptive strategy is a far stronger claim than beating round-robin.
Its own baselines — Random, "Active RFs", adaptive-without-tracking — give us a ladder that
maps onto the ablation study we need anyway.

---

## D14 — The problem is only hard if we measure it with both metrics at once

**Status:** `SETTLED` as a finding (2026-08-29). **AMENDED 2026-08-30 — read the amendment; it corrects the interpretation below.**

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
5. It vindicates not promoting the ~35% figure into the environment definition — a single
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

**AMENDMENT 2026-08-30 — the metrics below were naively instantiated; the corrected ones
change the conclusion.**

The 2026-08-29 table used a *per-dwell* hit rate, which is not how the literature defines
interception. Gul & Erer (Fig. 2) define interception ratio as *"the percentage of the total
amount of intercepted illuminations on the total amount of illuminations"* — per illumination,
not per dwell. And intercept time must be **censored**: an emitter never found counts at the
full 30 s, not dropped from the average (the 2026-08-29 table averaged only over found emitters,
which rewards not looking). Re-measured across all 47 scenarios with capture defined per pulse
(a pulse is intercepted iff the scheduler's band window contains it at its slot):

| Scheduler | Per-dwell hit | Interception ratio (per pulse) | Coverage | Censored mean TTI |
|---|---|---|---|---|
| Pulse-capture oracle | 91.9% | 66.8% | 75.1% | 15.04 s |
| Greedy static (camp) | 85.0% | **57.4%** | 30.4% | **23.78 s** |
| Round-robin | 33.3% | **5.5%** | 95.7% | **9.72 s** |
| Random | 33.3% | 5.5% | 93.5% | 10.01 s |

("Oracle" here maximises per-slot pulse capture; a TTI-optimal oracle would look different —
each column has a different optimum, which is itself the point.)

**What survives from 2026-08-29:** the tension is real, and single-metric reporting is fatal.

**What changes:**
1. **The camper's strength on interception ratio is genuine, not an artifact of per-dwell
   accounting** — it captures 57.4% of all pulses because the busiest band really does hold a
   mean 57.4% of a scenario's pulses (median 52.2%; >50% in 27 of 47 scenarios). And this
   dominance is a *documented dataset property*: the TSRD paper introduces label imbalance "at a
   proportion of up to 99.7%" and states *"the high proportion of strongly dominating
   transmitters is likely exaggerated."* So camping is strong because the dataset deliberately
   exaggerates dominance — universal across the dataset, not a bias in our 47 (which are
   stratified samples of it).
2. **The PS's own second metric already defeats the camper.** With censored intercept time,
   greedy scores 23.78 s against round-robin's 9.72 s. The 2026-08-29 claim that coverage —
   "a metric SIH didn't specify" — was what rescued the problem is **wrong**: intercept time is
   PS-mandated, and correctly computed it does the rescuing. Coverage remains a useful
   diagnostic, but we do not need off-spec metrics to make the problem honest.
3. **The two PS objectives are in direct, measured tension.** Interception ratio champion:
   greedy (57.4%, TTI 23.78 s). Intercept-time champion: round-robin (9.72 s, ratio 5.5%).
   No trivial strategy is good at both. **The problem, measured: Pareto-dominate that pair —
   an adaptive scheduler must approach round-robin's intercept time while multiplying its
   interception ratio.** That is exactly the explore-then-exploit arc Apfeld's algorithm
   embodies, and it is now a quantified target rather than a slogan.

**Caveats on these numbers.** Truth is stare-derived, so emitters below 500 MHz that only scan
sees are missing (D10). Occupancy is "≥1 pulse in band during slot" with no detection threshold,
which is the permissive end — D4's threshold will lower every row. Band assignment for pulses in
the overlap between adjacent bands takes the last matching band rather than both. Directional
conclusions are robust to all three; the exact figures are not final.

---

## D15 — The detection threshold is calibrated and swept, not trained

**Status:** `SETTLED` with D4 (2026-09-01)

The threshold that turns the continuous signal level (D4) into binary occupancy is a **receiver
design parameter**, not something the RL agent learns. Two commitments:

1. **Report the whole curve.** Sweeping the threshold traces Pd against Pfa — the ROC. The PS
   asks for Pd, Pfa *and* sensitivity as figures of merit; the sweep *is* that deliverable, and
   every scheduler is evaluated at the same operating point on it.
2. **Calibrate the default operating point against the recordings.** Choose the threshold at
   which replaying Turing's own sweep through our environment best reproduces the observed scan
   recordings (the ~35% non-empty dwell rate and the per-band structure). The TSRD paper's
   −100 dB ambient noise floor anchors the plausible range.

**Why not train it?** It defines the world the agent lives in. Training it alongside the agent
would let the optimiser move the goalposts — a scheduler could "improve" by making detection
easier. Fixed world, competing schedulers: that is the whole experimental design (D3, D7).

---

## D16 — Time base: 50 ms slots; native dwells span one or two slots

**Status:** `SETTLED` (2026-08-30) — routine implementation inside D3

The occupancy/signal grid uses a 50 ms slot — Turing's minimum dwell. The seven 100 ms dwells
in the reference schedule occupy two consecutive slots. An action is "choose a band"; the dwell
then lasts that band's native length per D3. Verified compatible: the dwell-schedule replay that
assigned 99.985% of scan pulses correctly was built on exactly this clock.

---

## D17 — Truth grid from the union of both recordings; validation kept out-of-sample

**Status:** `SETTLED` (2026-09-01) — adopted alongside D4; flag if you disagree

**Construction:** the truth state is built from the union of scan- and stare-derived evidence.
Neither recording is "the truth" (D1/D2 already say truth is the *constructed* state); they are
two incomplete observations of one world, and discarding either throws away real emitters.
Measured: adding scan on top of stare adds a mean **6.15%** more occupied cells — and up to
**+158%** in the outlier (`config_1089`, where scan recorded more pulses than stare, 1,693 vs
923) — plus the sub-500 MHz emitters stare's frequency floor cuts off entirely (D10).

**The circularity worry, and its fix.** If truth is built partly *from* the scan recording,
then validating by replaying the scan schedule and comparing to that same recording is partly
self-fulfilling. So the primary validation gate is **out-of-sample**: build a truth grid from
**stare only**, replay Turing's scan schedule through it, and compare the predicted detections
against the *actual scan recording* — data that never touched the construction. That is a
genuine prediction test (stare evidence → scan observations). The union grid is then used for
the final environment. The scan-only content (mostly sub-500 MHz) has no independent recording
to test against; that limitation is stated, not hidden.

---

## D18 — The environment must support scenario variation, not just 47 fixed replays

**Status:** `SETTLED` as a design requirement (2026-08-30)

The scenario loader takes one of the 47 configs as a **template** and can randomise within it:
activity-window placement, emitter beam phase (`scan_start_angle` is already per-instance
randomised in the data), positions within plausible range, and which emitters are active.
Deterministic replay of the exact recorded scenario remains available for validation.

**Why.** 47 scenarios × 600 slots is ~28k decision steps per pass — small if the agent can only
ever see 47 fixed worlds, and an agent trained on fixed replays can memorise them. Because the
environment is generative (D1), the 47 configs plus the 68 documented transmitter types are a
*scenario distribution*, not a fixed set — randomisation gives unlimited distinct episodes while
staying Turing-grounded. This preserves the original "exhaustiveness over volume" rationale for
the 47 (they are stratified for diversity) and is the standard sim-training practice (domain
randomisation). Whether and how much randomisation to use during training is the RL lane's
call; the environment's job is to make it available.

**Consequence for construction:** build the scenario loader with a seedable randomisation knob
from day one. It does not delay anything else.

---

## D19 — Deinterleaving is not required; the observation vector is the RL lane's open choice

**Status:** deinterleaving question `CLOSED` (2026-08-30); observation contents **default taken
in `ENVIRONMENT_SPEC.md` §L3 and recorded as D34** (2026-09-03) — extensions remain the RL lane's
call and get logged as decisions.

**Deinterleaving — separating interleaved pulses by emitter — is the Turing dataset's native
task, not ours, and nothing in the PS requires the scheduler to do it.** The training signal is
hit/miss per dwell; the environment knows every pulse's emitter internally (the `labels`
dataset, verified) and can score per-emitter metrics (coverage, intercept time) without the
scheduler ever attributing pulses. Note Apfeld et al. *assumed one radar per frequency
precisely to avoid deinterleaving* — Turing violates that heavily, so any future design that
needs per-emitter attribution from observations inherits a deinterleaving problem. Avoid
needing it.

**Open for the RL lane:** what the scheduler observes beyond binary hit/miss — pulse count in
the dwell? peak amplitude? **AoA or pulse width, both of which are measured PDW fields we
currently discard (D30)?** nothing? Richer observations help learning but move away from the
PS's minimal hit/miss framing. To be decided when Lane E starts, not now.

---

## D20 — Cold start: zero prior emitter knowledge each episode

**Status:** `SETTLED` (2026-09-01)

Every episode begins with no emitter library, no map of who transmits where, and no carry-over
between scenarios. The scheduler knows only what its own scan history has accumulated.

**Why — three sources agree, which is rare:**
- **[PS]** The title condition: *"in the absence of prior reliable intelligence of emitters and
  their operating characteristics."*
- **[intent, user 2026-09-01]** The project targets *active* warfare — emitters are not where any
  historical library says, so a library is a liability, not an asset. The ADITI 4.0 Cognitive EW
  problem statement (Indian Army, `docs/reference/problem-context/iDEX ADITI 4.0 (Go to Page 12).pdf` p.12 — the same problem family)
  frames exactly this: *"Historically, EW systems were developed based on knowledge of specific,
  previously learned threats,"* to be replaced by a system that *"sense, adapt and self-learn
  environment changes."* In the ADITI Q&A the Army would not even commit to sharing a threat
  library, reinforcing that the system cannot assume one.
- **[lit]** `docs/reference/scheduling/Dynamic Scan Scheduling.pdf` (Dutertre, SRI, RTSS'02): *"Today's systems rely
  on a fixed schedule, computed offline from an a priori table of known emitter types,"* which he
  identifies as the limitation to remove.

**Consequence.** The observation vector is built purely from scan history (D19). No pretraining
on emitter identities. This is also what makes the exploration/exploitation tension real: with a
prior, camping would be defensible; without one, you must keep discovering.

---

## D21 — Pd and Pfa are receiver properties at the frozen threshold, not agent-dependent quantities

**Status:** `SETTLED` — clarification (2026-09-01)

Raised by the team: if the threshold is frozen with the environment, why do Pd and Pfa — which
sound like they should depend on the scheduler — depend on the threshold?

**Resolution: they are two different questions, and separating them is correct.**

- **Pd and Pfa are per-look detection quantities.** Given the receiver looked at a band-slot,
  Pd = P(declare hit | truly occupied), Pfa = P(declare hit | truly empty). Both are fixed by
  the detector's threshold γ against the noise floor — pure receiver characterisation. They are
  reported once, by sweeping γ (the ROC), and are **identical for every scheduler** because they
  do not depend on *which* cells you look at, only on what the detector does *when* it looks.
- **The scheduler controls coverage, not detection.** What the agent changes is which
  band-slots get looked at — measured by interception ratio, intercept time, intercept rate.
  A scheduler cannot improve Pd; it can only point the detector at more of the right cells.

**Why freezing γ is right, not contradictory.** If γ were trainable alongside the agent, the
optimiser could lower it to manufacture hits — improving apparent interception by degrading Pfa.
Freezing the receiver (γ, geometry, noise) and letting only the schedulers vary is what makes
the comparison fair — the same reasoning as D3 (fixed receiver) and D7 (reward chosen, not the
world). This is the standard separation in the detection literature (`docs/reference/background/SIH- Smart Scan Strategy.pdf` §3–5: sensitivity is specified *together with* a required Pd and Pfa at a fixed
operating point; `docs/reference/scheduling/Dwell_Time_Optimization_of_Alert-Confirm_Detection.pdf` treats detection threshold and scheduling
as separate layers). **The apparent paradox dissolves: Pd/Pfa answer "how good is the
receiver?", the scheduler metrics answer "how well did we aim it?".**

---

## D22 — Keep the architecture at three layers; defer the generative signal model to v2

**Status:** `SETTLED` — scope discipline (2026-09-01)

The environment is three layers (truth grid → receiver → agent interface), consolidated in
`docs/project/ENVIRONMENT_SPEC.md`. The v1 truth grid is built **from the recordings** (union, D17). The
physics-based generative signal model (Apfeld Eq. 1 from emitter position/power/beam) is a **v2
upgrade**, not built now.

**Why defer, not skip.** The team flagged (2026-09-01) that judges value an architecture they
can follow, and that decisions-on-decisions risk complexity and inconsistency. The recording-built
grid satisfies the PS end to end and is directly checkable against the raw data. The generative
model buys richer counterfactuals (D1's original motivation) but adds a calibration surface we do
not need for a working, explainable v1. Its acceptance test already exists — it must reproduce the
v1 grid — so it can be added later without reopening anything. **One path now; the upgrade is
staged, not preserved as a live alternative** (matches `CLAUDE.md` "one path, not a menu").

**This slightly narrows D1 for v1.** D1 (generative environment) remains the direction; v1
realises it as "generative *scenario structure* — activity windows, variation, counterfactual
band/time queries — over a recording-derived signal grid," which is enough for every counterfactual
the scheduler actually poses (what if I look at band b at time t). Full physical generativity is
v2. Flagged here so the narrowing is explicit, not silent.

---

## D23 — γ is not calibrated against the recorded dwell rate; it is a swept receiver parameter

**Status:** `SETTLED` (2026-09-01). **Retracts D15 commitment 2.**

D15 said: choose γ so that replaying Turing's own sweep reproduces the recorded ~35% non-empty
dwell rate. Applied to the D17 union grid that returns γ ≈ −111 dB. **That procedure is
confounded and the number it produced is withdrawn.**

**Why it is wrong.** Measured this session — Turing's schedule replayed through grids built from
different sources, 47 train configs, 23,594 dwells, recorded non-empty rate **35.70%**:

| γ | scan-only grid | stare-only grid | union grid |
|---|---|---|---|
| no threshold | 36.14% | 34.73% | 41.54% |
| −150 | 35.72% | 34.30% | 40.96% |
| −110 | 30.13% | 28.90% | 35.35% |

The **scan-only grid with no threshold already returns the recorded rate**. The union grid's
excess is not undetected signal — it is stare's *independent simulation run* (D24) contributing
emitters and timings the scan run never had. Raising γ to cancel that excess mislabels
run-mixing as a detection threshold.

**The dataset is not contradictory.** Raised by the team: how can the recordings calibrate below
their own stated −100 dB noise floor? They do not. TSRD §II says detection was **probabilistic**
(*"the probability of pulse detection increases the more distinct the signal is from the noise
floor"*) and that *"amplitude ... [was] blurred using the OU process"* **after** reception. Weak
pulses therefore appear below −100 dB in the recording — measured, 13.25% of scan pulses and
15.96% of stare pulses do. This is exactly D9's finding about `sensitivity_dbm` (−110, with
4.49% of pulses below it), applied to a second field. **The recordings are already
post-detection**; −100 dB is not a floor our threshold has to clear.

**What replaces it.**
1. **γ is swept, and the sweep is the deliverable** — D15 commitment 1, unchanged.
2. **The default operating point is set by the receiver's own noise, not by a fit.** Noise floor
   `N₀ = −120 dB` (`sensitivity_dbm` −110 minus `gain_db` 10 — the anchor D9 noted); `σ = 3 dB`,
   **chosen, not measured**; `γ = N₀ + 3σ = −111 dB`, giving Pfa = 1.35e−3. Measured ROC over the
   47 train configs at that point: **Pd = 0.822, sensitivity (level at which Pd = 0.9) = −107.2 dB.**
   *(Amended 2026-09-03: sensitivity re-ran exactly; **`Pd = 0.822` is withdrawn** — the cell
   population it was averaged over was never recorded, and the three candidates measure 0.819,
   0.837 and 0.851. See D33. Nothing else in D23 is affected: γ, N₀, σ and Pfa are unchanged.)*
   The same number as the discredited calibration, reached for a defensible reason.
3. **The 35.70% becomes a pipeline self-consistency test**, which is what it should always have
   been: build the grid from the scan recording, replay the schedule that produced it, threshold
   nothing. Implemented as `tests/test_truth.py::test_pipeline_reproduces_the_recorded_dwell_rate`
   — measured **35.403% replayed against 35.700% recorded**, the 0.3 pp residual being slot
   quantisation. It tests band assignment, slot clock and dwell schedule; it says nothing about
   detection, which is why γ is no longer fitted to it.

**Evidence.** Measured: all figures above, this session, over all 47 train pairs. Sourced: TSRD
paper §II.

---

## D24 — scan and stare are independent simulation runs, not two views of one world

**Status:** `SETTLED` (2026-09-01). **Amends D17's premise; keeps its conclusion.**

D17 builds truth from the union of both recordings, premised on *"two incomplete observations of
one world."* **That premise is false.**

**Evidence.** Measured: nearest-neighbour ToA matching between the scan and stare files of the
same config finds no correspondence — median |Δt| ranges from 185 µs to 4.81 s against a
`toa_noise_scale_us` of 0.025. The same emitter gets disjoint activity in the two runs:

| config_2 | scan | stare |
|---|---|---|
| label 13 | 2.47 – 17.70 s | 19.01 – 28.21 s |
| label 16 | 4.96 – 13.65 s | **26.40 – 29.93 s** (disjoint) |
| label 28 | never seen | 2.58 – 14.00 s |

Emitter metadata is identical between the two files, so position, beam phase and power are
fixed; what differs is *when each emitter transmits*, which nothing in
`metadata/transmitters` encodes (already noted in D2). It is a per-run draw.

**Consequence.** A union grid is one incoherent timeline stitched from two. D17's *conclusion* —
use both recordings, discard neither — survives and is honoured by D25: both runs contribute
realisations to the emitter pool. They are simply never stitched into a single grid. Each
scenario carries contributions from exactly one recording, so every scenario is
self-consistent.

**What this does not change.** D17's out-of-sample gate stands and passes. Measured this
session: stare-only truth grid, replay Turing's scan schedule, predict the **scan** recording
(never used in construction) — **accuracy 86.19%, precision 87.87%, recall 71.14%, MCC 0.694,
per-band r = 0.940**, against a 35.70% base rate. Aggregate band occupancy is robust to which
individual emitter is on, even though individual activity is not.

One systematic failure, fully explained by D10: **band 0 (centre 250 MHz) is 59.12% occupied in
the recordings and 0.00% predicted**, because stare's `freq_range_mhz` starts at 500 MHz. Stated
as a limitation of the gate, not patched.

---

## D25 — the environment is a generative scenario sampler over recorded emitter contributions

**Status:** `SETTLED` (2026-09-01). **Supersedes D22's v1/v2 framing and narrows D18.**

Raised by the team: if the goal is a validated *generative* environment, why validate a
recording-derived one first and defer physics to a "v2"? And why treat physically justified
variation as messing with the environment rather than normal scenario sampling?

**Resolution: the v1/v2 split was the wrong axis. It is dropped.** The environment is generative
now — what is recording-derived is the *signal values*, not the environment's structure.

**Why no physics signal model in the validated environment.** To generate `S[b,t]` from emitter
physics we would need: range `R(t)` (✅ exact from metadata); beam angle `θ(t)` (⚠️
`scan_rate_rpm` takes 116 distinct values across the train set including 0.0 and 0.1, so it is
not literal rpm and its semantics are unconfirmed — folding config_2 emitter 3 gives max/mean
bin 4.59 at 30/rpm against 3.92 at 60/rpm, which is not decisive); the antenna pattern `G_t(θ)`
(❌ sidelobe structure published nowhere — **must be invented**); the absolute power scale onto
Turing's amplitude column (❌ **must be fitted**); and whether an emitter transmits at all (❌ not
in metadata, and it differs between runs per D24 — **still needs the recordings**).

So a physics model is not more fundamental than the recording-derived grid. It is that grid plus
an interpolator carrying ≥2 fitted parameters and one unconfirmed convention, still taking
activity from the recordings.

**The cost is methodological, not effort.** Those parameters would be fitted against the same
recordings the primary validation gate scores. **The gate would stop being a prediction and
become a fit.** It currently predicts held-out data at 86.19% accuracy (D24). Trading that for
realism we cannot independently verify is a bad trade. **What is lost:** placing an emitter at a
position or beam phase never recorded, and continuous control of difficulty. Both real; neither
required by the PS.

**What the environment is instead.** TSRD built each config by sampling emitter instances from a
68-type library and placing them independently, with **no emitter–emitter interaction** (paper
§II: *"line-of-sight path loss without multi-path interference"*). Therefore one emitter's
recorded contribution to the (band, slot) grid is a valid sample of *one emitter of that type at
a plausible position and beam phase over 30 s*, emitters compose by `max`, and **a new scenario
is a draw of N contributions from the pool.**

This is not an approximation of a physics model — it is the *same generative process TSRD used*,
one level up, with **zero invented parameters and nothing fitted**. Measured pool: **3,443
contributions** (1,739 scan-seen + 1,704 stare-seen) over **1,913 distinct emitters**, from
2,363 train transmitters — 19.0% are never detectable in either run.

**Dropped from D18: the per-emitter time-shift knob.** Circular-shifting an emitter in time
desynchronises its beam phase from its position track, so it is arbitrary randomisation of
exactly the kind we want to avoid. Emitter recombination alone defeats memorisation and is
physically justified; the shift is neither.

**The architecture, in four sentences** (unchanged three layers, D22):

> An emitter is a recorded contribution to the time–frequency grid.
> A scenario is a set of emitters; the world is their maximum.
> The receiver looks at one band at a time and declares a hit when what it hears beats its threshold.
> The scheduler chooses the band.

**Freeze list** — frozen before any scheduler number is quoted, and it lives in
`rfenv/constants.py` so the list is a literal file: band geometry, slot clock, native dwell
lengths, the truth-construction rule, `N₀`, `σ`, `γ`, the metric definitions, and the scenario
sampling distribution. **Free to vary per episode:** only the draw — which emitters, and the
seed. **Never:** no RL result may motivate a change to the frozen list; if one does, the
environment is re-validated from gate 1 and every baseline re-run.

**Memorisation vs validation contamination — separate tracks.**

| Track | Scenarios |
|---|---|
| Validation (gates 1–4) | the 47 **deterministic** single-run replays, unmodified |
| RL training | sampled draws from the pool (train configs only) — no episode repeats |
| Scheduler comparison | one fixed seeded set of sampled scenarios *plus* the 47 replays, identical for every scheduler |
| Final claim | the 45 held-out pairs, **once** (D8) |

Sampled `n` is drawn from the empirical per-config count of *detectable* emitters (measured:
1 to 82, summing to 1,913), so sampled scenarios keep the difficulty spread the data has.

---

## D26 — Pd and Pfa reference physical occupancy Z, not a second thresholded copy of S

**Status:** `SETTLED` (2026-09-01) — corrects a degeneracy in `EVALUATION.md` §3.

Three quantities, and only two of them are truth:

- **`Z[b,t]`** — physical occupancy: is any emitter transmitting into this cell? Threshold-free.
  This is the PS's *"transmission or a non-transmission"* status.
- **`S[b,t]`** — the continuous level: peak recorded amplitude where `Z` is true, `N₀` where not (D4).
- **`Y`** — the receiver's declaration when it looks: `Y = 1` iff `S + n ≥ γ`, `n ~ N(0, σ)`.

`Pd = P(Y=1 | Z=1)`, `Pfa = P(Y=1 | Z=0)`.

**Why not against `O = (S ≥ γ)`.** EVALUATION.md §3 defined Pd against `O`, which is itself `S`
thresholded at the same γ. That is degenerate: `Pd = P(S+n ≥ γ | S ≥ γ) ≥ 0.5` by construction,
for every γ, so it can never sweep and the ROC carries no information. Referencing the
threshold-free `Z` gives a real curve — measured, Pd falls 0.894 → 0.681 as γ goes −120 → −100.

This is the Z-versus-Y framing the 2026-09-01 consistency audit already recorded as consistent
with `docs/reference/background/SIH- Smart Scan Strategy.pdf` §3–5. D21 is unaffected: Pd and
Pfa remain receiver properties at frozen γ, identical for every scheduler.

---

## D27 — "detectable activity interval", not "activity window"

**Status:** `SETTLED` (2026-09-01) — terminology, raised by the team.

`on_e` and `off_e` are derived from the truth grid: the first and last slot at which the
emitter's **own** received level clears γ. No separate window field is stored.

**Why derived.** Self-consistent by construction — nothing can be "active" that is not in the
world the scheduler faces, so censored intercept time is measured against exactly the grid it
was earned on. It also needs no extra decision, which matters given D24 killed the idea of a
single true window.

**Why the name.** This is activity *detectable under this receiver at this γ*, not the emitter's
physical transmission window. An emitter can be transmitting for the whole episode with its
received signal below γ, and this interval will not show it. Judged on the emitter's own level
rather than the combined `S`, so a quiet emitter sharing a band with a loud one does not inherit
the loud one's detectability.

**Consequence for `E`** (EVALUATION.md §1, the coverage denominator): `E` is the set of emitters
with a non-empty detectable interval — not every transmitter in the metadata. Measured, 19.0% of
train transmitters never appear in either recording, several because they transmit above 18 GHz;
scoring a scheduler for missing those would be meaningless.

---

## D28 — What counts as intercepting an emitter: D5 and D27 conjoined

**Status:** `SETTLED` (2026-09-03) — not a new decision. Records the answer to a question the
implementation lane raised in `first_e` terminology, which D5 and D27 already settle jointly.

An emitter `e` is **intercepted at slot `t`** iff all three hold:

1. `a(t)` is a band `e` puts pulses into at `t` — the scheduler was looking where it radiates;
2. **`e`'s own** received level in that cell clears γ;
3. the receiver declared `Y(t) = 1`, i.e. `S[a(t),t] + n ≥ γ`.

(1) ∧ (3) is D5 verbatim. (2) is D27's own-level rule, the same one that defines `on_e` — so
numerator and denominator of censored intercept time are measured by one rule and
`first_e ≥ on_e` holds by construction.

**Why the question arose, and the misreading to avoid.** It reads as though γ replaces the
"double coincidence" of receiver-tuned × emitter-illuminating with a generic noise threshold.
It does not. D4 already converted the second factor from a boolean gate into a continuous
level, on measured evidence, and clause (2) *is* that factor — an emitter pointed away is
represented by its own level collapsing, not by a flag going false.

**Evidence — D4's beam measurement, re-run 2026-09-03** on `config_2` stare, folding each
emitter's ToAs at `30/scan_rate_rpm` into 36 phase bins: **15 of 19 emitters occupy all 36
bins**, median peak-amplitude swing across phase **53.8 dB** (max 72.0). The four that do not
are sampling-limited, not gated — label 18 has 32 pulses over 17.1 revolutions (1.9/rev),
label 30 spans 0.1 of a revolution. The single `Omni` emitter (label 28) swings **3.9 dB**
against 53.8 dB median for `Circular` ones, which is the antenna pattern appearing exactly
where it should. A rotating emitter is interceptable through its sidelobes; there is no
off-state to coincide with.

**Why not the γ-free alternative** ("intercepted = tuned to a band it pulses into, no detection
required"). It mirrors interception ratio's wording, but it contradicts D5, credits intercepts
no receiver declared, and breaks `first_e ≥ on_e`. Measured on the same file: label 24 has
pulses in band 18 at slot 35 at own level −120.4 dB while its detectable interval starts at
slot 36, so a scheduler tuned there at slot 35 would score an intercept time of **−0.05 s**.
The division of labour is already correct without it: **interception ratio** is the
threshold-free opportunity metric, **intercept time** is the detection metric (D14).

**Implementation choice taken inside this design** (routine, not gated): `Y` is declared on the
**combined** `S[a(t),t]`, per D26 — not on the emitter's own level plus noise. `Y` is the
scheduler's only observable and a real receiver cannot un-mix a cell. Since `S = max` over
contributors, `own ≥ γ` implies `S ≥ γ`, so clauses (2) and (3) never conflict; the effect is
that at a cell shared with a louder emitter, a qualifying weak emitter's detection is close to
assured. Accepted as the cost of a cell-level model.

**Known limitation, deferred to v2.** The honest form is per-pulse detection — a real receiver
deinterleaves PDWs, so each pulse should clear γ on its own amplitude rather than the cell
maximum deciding for every contributor. That needs per-emitter noise draws at L2 and belongs
after the gates, not before.

**Consequence.** `EVALUATION.md` §1 now carries the three-clause `first_e` definition.
`receiver.py` implements it; nothing in `truth.py` changes — `detectable_interval` already
judges on `c.peak_dbm`, the emitter's own array.

---

## D29 — The reward may read truth; the observation may not. And D28 splits the metrics across that line

**Status:** `SETTLED` (2026-09-03) — not a new decision. Records the answer to a question the
implementation lane raised ("should reward count declared hits `Y` or true hits `Z`?"), which
D7 and the offline-training workflow already settle, plus a consequence of D28 that does
change the candidate set.

**The workflow this rests on.** Train offline in the simulator, freeze the policy, evaluate.
`PROJECT_ARCHITECTURE.md` §10 (`Develop RL scheduler → Freeze final system → Final held-out
evaluation`) and `EVALUATION.md` §7 both describe only this. **There is no online learning after
deployment anywhere in the architecture**, and nothing in the PS asks for it — *"trained based on
hits and misses"* names the learning signal, not when learning happens.

**Consequence, and the misreading to avoid.** It is tempting to argue that the reward must be
restricted to what a fielded receiver could compute. It must not. The two are different objects:

- **The observation is the policy's input path.** It ships. It must contain only what a deployed
  receiver has — the agent's own scan history and nothing else (D19, D20).
- **The reward is a training-time construct and is discarded at deployment.** Nothing evaluates
  it at inference. It may read `Z`, per-emitter own levels, `first_e`, or any other simulator
  state. This is ordinary privileged-information training; asymmetric actor-critic is the named
  case.

The restriction would only bite if we ever fine-tuned online, which we do not.

**What D28 changes.** D28 requires `Y(t) = 1` for an intercept to be credited to an emitter, so
`first_e` — and therefore **censored mean intercept time** — is `Y`-conditioned. Interception
ratio stays threshold-free (D28: *"interception ratio is the threshold-free opportunity metric,
intercept time is the detection metric"*). **The two headline metrics now sit on opposite sides
of the `Y`/`Z` line, so no single reward is aligned with both.** This is D14's tension appearing
in the reward's information source, not just its shape.

**Why this is not settled by "the agent cannot control Pd" (D21).** D21 is about the *aggregate*
Pd being scheduler-independent. Per-cell detection probability is not uniform: under the L2
model, `P(Y=1 | cell) = Φ((S − γ)/σ)`, which with the frozen `σ = 3 dB` runs 0.500 at `S = γ`,
0.841 at `γ+3`, 0.977 at `γ+6`, 0.999 at `γ+9`. The recorded aggregate 0.822 is a mixture over
that curve. So:

- a `Z`-reward values every detectable cell equally;
- a `Y`-reward weights cells by loudness, and — the part that matters — teaches the agent that a
  marginal emitter needs **repeated looks** before it yields a declaration. Under D28 that is
  exactly what lowers its `first_e`. A `Z`-reward never teaches it.

**A false-alarm penalty is not a Pfa improvement.** Pfa is receiver-level and frozen; no reward
can move it (D15, D21, `EVALUATION.md` §3). At the operating point the false-alarm channel is
also small: `Pfa = 1.35e−3` is the Gaussian 3σ tail (`1 − Φ(3) = 1.3499e−3`, computed
2026-09-03), so an all-empty band earns ~0.81 spurious reward over 600 slots. What an FA penalty
actually prices is a wasted dwell — name it that.

**Decision.** No change to the default: **candidate 1 stays `+1` per true hit** (`Z`, D5), as
`ENVIRONMENT_SPEC.md` §L3 already states. D28 promotes the alternatives from variants to
principled D7 candidates with a predicted trade, which is what makes them worth running:

| # | Reward | Reads | Predicted to favour |
|---|---|---|---|
| 1 | `+1` per true hit (D5 cell-level `Z`) | truth | interception ratio |
| 2 | `+1` per declared hit (`Y`) | receiver only | censored intercept time |
| 3 | `+1` per **first** intercept of an emitter (D28's three clauses) | truth | censored intercept time, coverage |

Candidate 3 is the discovery weighting D5 flagged as *"still open inside this"*; D28 gives it a
precise definition it did not have before. Running several is D7 as written, not a departure
from it — the judging metrics are PS-mandated and reward-independent, so they rank rewards from
outside without circularity.

**Two constraints on that comparison, and one genuinely open question.**

1. Keep the set to the three above. Every candidate scored on the same 47 scenarios is another
   draw; best-of-many is partly selection noise. D8's single-use held-out set is the backstop.
2. `EVALUATION.md` §4 already bars ranking across reward families by accumulated reward.
3. **Open, and not decided here:** the selection rule when candidates Pareto-dominate the
   baselines but not each other. D14 measured the two objectives in direct tension, so "whichever
   scores most" has no referent. A scalarisation, a lexicographic rule, or reporting the front —
   to be fixed **before** training, so the choice is not made after seeing results. Not blocking:
   `receiver.py` and `env.py` do not depend on it.

**Evidence.** Reasoned from D5, D7, D14, D21, D26, D28 and the architecture's own stage order.
The `Φ((S−γ)/σ)` curve is the L2 model definition, not a measurement. `Pd = 0.822` was quoted from
`EVALUATION.md` §3 (measured 2026-09-01) and not re-run for this entry — **it has since been
withdrawn (D33)**; read the two mentions above as "the aggregate Pd, whatever D33 fixes it at,
is a mixture over the per-cell curve". The argument does not depend on the value.

---

## D30 — AoA is a measured PDW field we discard; proposal to reconsider it for the observation

**Status:** `OPEN` (2026-09-04) — **owned by the RL lane, and off the pre-RL critical path.**
Re-scoped 2026-09-04 from "awaiting a human decision": the question is real and the measurements
below stand, but it cannot be answered well yet and it does not block anything before the freeze.

**Why it is not a pre-freeze decision.** The observation vector is **not on the freeze list**
(`ENVIRONMENT_SPEC.md` §Freeze list), the four validation gates do not read it, and D19 already
assigns observation contents to the RL lane. So adding AoA later costs a retrain of the policy's
input layer — not a re-validation of the environment, and not a re-run of the baselines.

**Why it should not be closed either way now.** Both answers would be speculation. Closing it
"out" asserts the 109-vector is sufficient before any agent has run. Closing it "in" pays a real
cost — L1 gains an angular dimension, breaking the `max`-composition model, and the observation
needs a fixed-width encoding of a variable-length bearing set — on the strength of an attribution
figure that is a **ceiling**, computed from true labels and whole-episode medians, and that
already falls 96.7% → 86.1% from 19 emitters to 99, i.e. worst where scheduling is hardest.

**The trigger that decides it.** A trained agent, or Apfeld, failing to explore in a way that
per-band hit rate, visit density and staleness demonstrably cannot fix — most plausibly showing
up as a policy that camps despite a novelty-shaped reward (D29 candidate 3, whose target is
truth-side and therefore imperceptible to the current vector). That is measurable once the
baselines and one trained policy exist, and meaningless before. `PulseWidth` rides on the same
decision.

**Consequence for the pre-RL lane:** none. `receiver.py`, `env.py`, `render.py`, `metrics.py`,
`validate.py` and the baselines are all unaffected.

**The situation.** Turing PDWs carry five fields — `metadata/feature_names` is `['ToA',
'Frequency', 'PulseWidth', 'AoA', 'Amplitude']`. `rfenv/scenario.py` reads columns 0, 1 and 4
and silently drops **`PulseWidth` and `AoA`**. No decision anywhere records that choice; it is
an omission, not a ruling. This entry exists so it becomes one either way.

**Why dropping it is defensible.** Three real reasons: the PS calls interception *"a two
dimensional search problem"* (frequency × time), so the **action** space is correctly angle-free
— this receiver tunes, it does not steer, and AoA can never be something a dwell is spent on.
L1's grid is `(band, slot) → S/C/Z` combined by `max`; AoA is per-emitter, so a shared cell has
several bearings and `max` is meaningless on them. And D19 explicitly steers away from
per-emitter attribution: *"any future design that needs per-emitter attribution from
observations inherits a deinterleaving problem. Avoid needing it."*

**Why it is nevertheless worth reconsidering — the information gap, measured 2026-09-03.**

To an agent seeing only binary hit/miss, a dwell that finds a **new** emitter and one that
re-finds a **known** emitter are indistinguishable. That ambiguity is the camper pathology of
D14, stated in information terms. Its size, walking 600 slots at truth level (D28 clauses 1–2,
noise draw skipped — this is an information question, not a detection one):

| scenario | policy | novel dwells | redundant | redundant / hits |
|---|---|---|---|---|
| `config_2` | round-robin | 16 | 104 | 86.7% |
| `config_2` | camper (band 6) | 5 | 486 | **99.0%** |
| `config_921` | round-robin | 51 | 220 | 81.2% |
| `config_921` | camper (band 1) | 7 | 590 | **98.8%** |

The camper's dwells are ~99% redundant and *every one of them looks like a hit*. Nothing in the
current observation vector (per-band hit rate, visit density, staleness) separates the two
columns.

**AoA does separate them.** Assigning each pulse to the emitter with the nearest median bearing,
scored against the true `labels` column, per config over all bands carrying more than one
emitter:

| scenario | multi-emitter bands | pulses | attribution accuracy |
|---|---|---|---|
| `config_2` | 11 | 1,272,772 | **96.7%** |
| `config_59` | 3 | 74,776 | **99.3%** |
| `config_921` | 21 | 4,321,146 | **86.1%** |
| `config_81` | 0 | — | n/a (no band holds two emitters) |

Bearings are well separated relative to their spread — in `config_2` band 18, five emitters sit
at −130.2° (σ 8.9), −114.2° (5.4), −61.9° (6.2), −43.2° (1.1) and −29.6° (1.7).

**Why this matters *more* after D29, not less.** D29 rules that the reward may read truth freely
while the observation may not. Its candidate 3 is *"+1 per first intercept of an emitter"* — a
novelty reward. But novelty is a truth-side quantity: the agent is trained to value something it
**cannot perceive at inference**, because binary hit/miss cannot distinguish new from seen. The
policy would have to approximate novelty with staleness, which is a proxy for *time since
looked*, not for *have I already got everyone here*. AoA is the observable that makes D29's
candidate 3 actionable rather than merely scorable.

**Honest limits — three, and none of them small.**

1. **The accuracy above is a ceiling, not an achievable figure.** Medians were computed over the
   whole episode from the true labels. A deployed agent clusters bearings online with no labels
   and does worse, by an unmeasured amount.
2. **AoA discriminates seen-vs-unseen; it does not find anything.** It cannot say where an
   unobserved emitter is. Its exploration value is negative information — *"this band is
   exhausted, leave"* — which is real but is not a search heuristic.
3. **It degrades exactly where the data is hardest.** 18 of 32 `config_2` transmitters have
   `speed_km_s ≠ 0`, so bearings drift (label 23's AoA σ is 122.6° — it crosses the circle), and
   close pairs are unresolvable (`config_2` band 6 holds emitters 0.8° apart against σ ≈ 0.5°).
   Accuracy falls 96.7% → 86.1% from 19 emitters to 99, i.e. worst in the crowded scenarios where
   the scheduling problem is hardest.

**Cost if adopted.** L1 gains an angular dimension it does not have (per-cell bearing lists, not
a `max`), and the observation vector — currently a fixed 36×3+1 — needs a fixed-width encoding
of a variable-length bearing set, i.e. binning or online clustering. Neither is free. `PulseWidth`
is dropped for the same non-reason and is also a standard deinterleaving feature; the same
decision covers it.

**What is being asked.** Either record *"AoA and PulseWidth stay out, and here is why"*, or
*"they enter the observation, at this cost."* Not blocking: `receiver.py` and `env.py` do not
depend on the answer, and the four validation gates do not touch the observation vector at all.

**Evidence.** Measured 2026-09-03 against `data/turing/stare/train_stare/*.h5` (`config_2`,
`config_59`, `config_81`, `config_921`) via throwaway scripts; the redundancy walk uses
`rfenv.truth` at the frozen γ = −111. Field list confirmed against `metadata/feature_names`.
Column-drop confirmed by reading `rfenv/scenario.py`.

---

## D31 — Reward is defined per slot; a dwell's reward is the sum of its slots'

**Status:** `SETTLED` (2026-09-03) — accepted by the team. Records the answer to the third
question the implementation lane raised. Unlike D28 and D29, this one was **not** already
settled elsewhere: no prior decision covers how reward accounts for a two-slot dwell. It is
nevertheless tightly constrained, because the metrics that judge a reward (D7) are already
per-slot and per-illumination.

Seven bands — 0, 1, 6, 7, 17, 18, 19 (250, 750, 3250, 3750, 8750, 9250, 9750 MHz) — carry a
native 100 ms dwell, so choosing one consumes two 50 ms slots (D3, D16). **Reward is scored on
each slot the dwell covers, and the dwell's reward is their sum.** A 2-slot dwell can therefore
earn up to +2 under a per-hit reward. The rejected alternative is OR-over-slots: +1 per dwell
regardless of length.

**Why, in order of force.**

1. **Nothing else in the system is per-dwell.** Interception ratio is *per illumination*
   (`EVALUATION.md` §4, explicit: *"Per illumination, not per dwell"*). Censored intercept time
   is per emitter via `first_e`, which is per slot (D28). D7 selects a reward by scoring it on
   exactly these metrics, so a per-dwell reward would be the only per-dwell quantity in the
   project — and `EVALUATION.md` §4 trap 1 already records per-dwell accounting as a *measured*
   trap: the camper scores 85–90% per-dwell while capturing 30% of emitters.
2. **L2 performs two measurements, not one long one.** A 100 ms dwell is `S[b,t] + n₀` at slot
   `t` and `S[b,t+1] + n₁` at `t+1` — two independent noise draws, two chances at `Y = 1`. Under
   D29's `P(Y=1) = Φ((S−γ)/σ)`, a marginal emitter at `S = γ` yields 0.5 per look and **0.75 over
   two**. D29's case for a `Y`-reward is precisely that it teaches *"a marginal emitter needs
   repeated looks before it yields a declaration"*; OR-over-slots erases that signal.
3. **OR-over-slots trains the agent to abandon the best part of the spectrum.** It makes the
   seven wide bands strictly dominated — twice the airtime for the same maximum reward — so a
   trained agent learns to avoid them. Measured this session across all 47 train scan configs,
   those are the bands that matter most:

   | | wide bands | narrow bands |
   |---|---|---|
   | emitter-frequency placements | mean **764.6** | mean **170.4** |
   | median `scan_rate_rpm` | **10.00** | **35.00** |

   Correlation of *is-wide* with density **+0.629**, with median scan rate **−0.425**. Five of
   the seven rank in the top 8 of 36 by density. Bands 0 and 1 are not dense (ranks 16 and 13)
   but hold the slowest rotators in the dataset — **median 3.00 rpm**, one revolution per 20 s,
   the hardest emitters to catch. Avoiding these bands would be D14's camper pathology
   reintroduced by an accounting artifact rather than by the physics.
4. **The second slot is not filler.** Measured this session, 47 train scan configs, γ = −111 dB:
   persistence `P(Z[b,t+1] | Z[b,t])` in wide bands **77.72%** (narrow control 60.59%); over
   16,000 wide dwells starting on an occupied slot, mean **307.63** pulses in slot 1 against
   **254.84** in slot 2 (**82.8%**); and over 17,584 non-empty wide slot-pairs, **32.89%** contain
   an emitter detectable in slot 2 that was not detectable in slot 1 (D28's own-level rule).

**What this keeps invariant.** Reward per unit time is the same for wide and narrow bands: a
wide band costs twice the airtime and can earn twice the credit. Since retuning is free (below),
airtime is the only currency in the problem, and this is the accounting that leaves the agent
free to trade it on the merits rather than on a units artifact.

**Uniform across all three D7 candidates.** Candidates 1 (`+1` per true hit, `Z`) and 2 (`+1` per
declared hit, `Y`) are per-slot quantities and simply sum. Candidate 3 (`+1` per **first**
intercept of an emitter) is unaffected — an emitter is first-intercepted once, whichever slot of
the dwell delivers it.

**Implementation** (routine, inside this design): a wide-band action returns one `step()` with
`reward = r[t] + r[t+1]` and advances the clock by two slots. `env.py` exposes one action per
band; dwell length is the band's, never the agent's.

**Supporting measurement — retuning is free, so airtime is the only cost.** The sweep period is
exactly 2.15 s = `sum(dwell_times_s)`. If each retune cost `dt`, the true period would be
`2.15 + 36·dt` and band assignment would drift across a 30 s episode. Replayed against all
**4,393,233** train scan pulses:

| assumed dead time per retune | pulses in the predicted band |
|---|---|
| **0 µs** | **99.985%** |
| 1 µs | 99.829% |
| 5 µs | 99.146% |
| 20 µs | 96.549% |
| 100 µs | 82.896% |

No drift within the episode either: 99.996% / 99.984% / 99.973% across the three 10 s thirds.
Retune costs well under 1 µs — under 0.002% of a slot. This confirms by measurement what
`ENVIRONMENT_SPEC.md` §L2 previously asserted ("no retune cost"). `dwell_times_s` and
`dwell_centres_mhz` are byte-identical to `rfenv/constants.py` in **47/47** train scan configs,
so the wide-dwell set is a fixed receiver property, not a per-config one.

**What is *not* represented, and is out of scope.** Staying on a band has operational value our
scorecard does not score: contiguous observation is what lets a real ES receiver estimate PRI and
scan period. No metric in `EVALUATION.md` prices it. That is consistent with the PS, whose primary
objective is *"minimize intercept time and ensure a high interception rate"* and not
characterisation quality — but the asymmetry is stated rather than hidden.

**Noted, not acted on — the dwell schedule encodes prior intelligence.** The seven 100 ms bands
sit exactly where the emitters are densest and where the beams come round slowest. That is sound
receiver design, and we adopt it under D3, but it means the *receiver* carries prior knowledge of
the emitter population even though D20 starts the *scheduler* cold. Not a defect and not a change
to anything; recorded so the write-up states it before a reviewer notices it.

**Evidence.** Measured this session (2026-09-03) over all 47 train scan pairs via scratch scripts,
using `rfenv.truth` at the frozen γ = −111 dB; dwell-schedule and retune figures read directly
from `data/turing/scan/train_scan/*.h5`. Sourced: `EVALUATION.md` §4, D3, D7, D14, D16, D28, D29.

---

## D32 — a sampled scenario draws at most one contribution per physical emitter

**Status:** `SETTLED` (2026-09-03) — implementation correction inside D25, found by the
consistency audit. Lands before the freeze, so nothing is re-validated.

`Scenario.sample` drew uniformly from all 3,443 pool contributions. But 1,530 emitters appear in
the pool **twice** — once from the scan run, once from the stare run (3,443 contributions over
1,913 distinct emitters). So a draw could return both realisations of the same physical emitter.

**Why that is wrong.** D25 justifies the sampler on the grounds that one recorded contribution is
"a valid sample of *one emitter of that type at a plausible position and beam phase* over 30 s".
Two contributions of the same `(config_id, label)` are the same emitter at **identical** position
and beam phase with **disjoint** activity (D24) — which is not a plausible independent placement,
and is precisely the arbitrary randomisation D25 rejected the time-shift knob for. It also
double-counts one emitter in `E`, inflating both emitter coverage and censored mean intercept
time.

**Measured this session.** Over 2,000 sampled scenarios at the default `n` draw: **23.9%**
contained at least one duplicated physical emitter, mean 0.306 duplicates per scenario, max 4.
At the top of the difficulty range (`n = 82`): **59.0%** of scenarios, mean 0.854.

**Decision.** Sample distinct `(config_id, label)` keys, then take one contribution per key. Both
runs still feed the pool (D17's conclusion, D24's mechanism); a scenario simply never contains
the same emitter twice.

**Consequence.** The scenario sampling distribution is on the freeze list (D25), so this had to
land before the gates run — it now has. `n` continues to be drawn from the empirical per-config
count of *detectable* emitters, re-verified this session as **1 to 82, summing to 1,913 over 47
configs**, matching D25 exactly.

**Evidence.** Measured 2026-09-03 via `rfenv.scenario` against `data/turing`; pool figures
re-run and identical to D25's.

---

## D33 — which cells P<sub>d</sub> is averaged over

**Status:** `SETTLED` (2026-09-03) — **accepted by the team: the reference-sweep population.**
Receiver characterisation, so it was gated by `CLAUDE.md`. It is what unblocked `receiver.py`'s
ROC; `PD_POPULATION = "reference_sweep"` is now on the freeze list in `rfenv/constants.py`.

**The problem.** `EVALUATION.md` §3 said "only cells the receiver actually looked at contribute".
Taken literally that makes P<sub>d</sub> **scheduler-dependent** — different schedulers look at
different cells, and per-cell detection probability is not uniform (`P(Y=1 | cell) =
Φ((S−γ)/σ)`, D29), so a camper parked on loud cells would report a higher P<sub>d</sub> than a
sweeper. That contradicts D21 and §0, both of which state P<sub>d</sub> is identical for every
scheduler. The population was never specified, and the figure depends on it entirely.

**Measured this session** — ROC re-run from `rfenv` over the 47 train configs at the frozen
γ = −111 dB, analytic in the noise draw (`Pd = mean of Φ((S−γ)/σ)` over the population;
confirmed against a Monte-Carlo draw to three decimals):

| Population | n occupied cells | P<sub>d</sub> |
|---|---|---|
| All occupied cells, scan replays | 29,707 | **0.837** |
| All occupied cells, stare replays | 336,210 | **0.819** |
| Only cells Turing's reference sweep looks at, scan replays | 11,710 | **0.851** |

The previously quoted **0.822** is closest to the stare population but matches none of them, and
is withdrawn. P<sub>fa</sub> = 1.35e−3 and sensitivity −107.2 dB are unaffected — both are
analytic in γ and σ (`1 − Φ(3)` and `γ + 1.2816σ`), and both re-ran exactly.

**Decision: the reference-sweep population** — the occupied cells Turing's own
`dwell_schedule()` looks at, giving **P<sub>d</sub> = 0.851** at the operating point. It keeps
the "per-look" reading `EVALUATION.md` §3 already commits to, it is **fixed and
scheduler-independent** as D21 requires — the schedule is Turing's own and never varies, so no
scheduler can move the population by looking elsewhere — and it is the same fixed schedule every
other model-level check already uses (D3, gates 1 and 2, baseline 3).

**Rejected, and recorded once so it does not come back: all occupied cells** (0.837). A larger
sample and independent of any schedule, but it characterises the detector over cells no receiver
ever visits, which is not what a per-look probability means.

**The population is on the freeze list** beside γ, N₀ and σ, as `PD_POPULATION` in
`rfenv/constants.py`, and the ROC is reported as a curve with the population stated on it. The
*shape* of the sweep does not depend on this choice; only the quoted operating point does.

**Evidence.** Measured 2026-09-03 via `rfenv.truth` over all 47 train pairs. Reasoned from D21,
D26 and D29. Re-measured from `rfenv.receiver` when L2 landed (2026-09-03) — see the
`test_receiver.py` operating-point test, which asserts the figure rather than quoting it.

---

## D34 — the base observation vector, recorded

**Status:** `SETTLED` (2026-09-04) — **ratified in the implementation lane, not escalated.**
Records a choice already made in `ENVIRONMENT_SPEC.md` §L3, built in `rfenv/env.py` and covered
by `tests/test_env.py`. The human retains a one-line veto; nothing downstream assumes otherwise.

**Why this did not need escalating, when D30 does.** `CLAUDE.md` gates decisions that shape the
agent interface because they are expensive to reverse. This one is not: the observation vector is
**explicitly absent from the freeze list** (`ENVIRONMENT_SPEC.md` §Freeze list — *"Stays open for
the RL lane: reward candidates and observation extensions"*), and the four gates never touch it.
So ratifying costs nothing and un-ratifying costs a retrain rather than a re-validation. There is
also no live alternative to weigh: every quantity this vector excludes is either D30's separate
question or already excluded by D12 and D19. Recording a built, tested, spec-carried choice with
no competing option is bookkeeping, not architecture.

**Why this entry exists.** `ENVIRONMENT_SPEC.md` §L3 fixes `observation_space` at **36×3 + 1 =
109**: per-band empirical hit rate, per-band visit density, per-band staleness, plus normalised
episode time. That choice appears in no decision entry — no evidence line, no alternatives, no
recorded approval. Meanwhile D19's status still reads *"observation contents `OPEN`"*, and D30 is
correctly being held for a human decision **because it changes the observation vector**. So the
base vector went through no gate while its proposed extension waits at one. The 2026-09-03 audit
flagged the asymmetry; this entry closes it either way.

**The case for the three quantities.** Each maps onto one of the PS's own figures of merit, which
is why they were chosen: **hit rate** is what a scheduler needs to estimate detection
probability, **visit density** is intercept rate, **staleness** is what makes the problem
restless rather than a plain bandit — a band ignored for 10 s may have become busy without
telling you. All three are computed **only from the agent's own scan history**, so the vector
carries no prior emitter intelligence (D19, D20) and nothing truth-side leaks into the policy's
input path (D29).

**What it deliberately excludes.** Pulse count and peak amplitude within the dwell (available at
L2, richer than the PS's minimal hit/miss framing); AoA and PulseWidth (D30, separately gated);
anything per-emitter (D19: avoid needing deinterleaving).

**Consequence.** `D19` is amended from *"observation contents `OPEN`"* to **"default taken here;
extensions remain the RL lane's call and get logged as decisions."** The four validation gates do
not touch the observation vector, so nothing in validation depends on this.

**Evidence.** Sourced: `ENVIRONMENT_SPEC.md` §L3, `SIH26055_PROBLEM_STATEMENT.md` (figures of
merit). Reasoned from D19, D20, D29. No new measurement.

---

## D35 — the episode terminates at 600 slots, and an overrunning dwell is clipped

**Status:** `SETTLED` (2026-09-03) — routine implementation inside D16 and D31, recorded because
the first half changes how an RL algorithm bootstraps and the second is a boundary rule someone
will otherwise have to rediscover by reading `env.py`.

**Two rules, taken while building L3.**

1. **The 600-slot horizon returns `terminated=True`, not `truncated=True`.** Gymnasium separates
   the two so that an agent knows whether to bootstrap a value estimate past the boundary:
   `truncated` means the task continues and the *harness* stopped it, `terminated` means the
   world ended. Here 30 s is the task — it is Turing's own `collection_time_s` (D16), and it is
   the entire extent of the grid the scenario describes. There is no slot 600 to have a value.
   Reporting `truncated` would invite an agent to bootstrap from a state that does not exist.
2. **A wide-band action at slot 599 is clipped to one slot** rather than being made illegal.
   This matches `constants.dwell_schedule()`, which already truncates its final dwell at the
   30 s boundary, and it keeps all 36 bands legal at every slot — so the action space never has
   to change shape mid-episode, and no policy needs a masking layer for one edge case.

**Consequence worth stating plainly, because it surprises people.** An episode is **600 slots but
300 to 600 `step()` calls**: seven bands consume two slots each (D3, D16, D31), so the number of
decisions depends on what the agent chooses while the wall clock does not. Measured this session
— camping band 0 (wide) takes 300 steps, camping band 5 (narrow) takes 600, and both cover
exactly 30 s. Anything computing a per-step average must know this; anything computing a
per-second average is unaffected, which is another reason every metric in `EVALUATION.md` §4 is
per-illumination or per-emitter rather than per-step.

**Evidence.** Reasoned from D16 and D31 plus the Gymnasium API contract. The step-count figures
are measured, and are asserted in `tests/test_env.py::test_step_count_varies_with_dwell_width_but_airtime_does_not`.

---

## D36 — a scan replay is not a scheduler-comparison scenario: it carries the reference sweep's own footprint

**Status:** `SETTLED` (2026-09-04) — **accepted by the team.** Evaluation semantics, so it was
gated by `CLAUDE.md`. Found while building `metrics.py`; nothing in D1–D35 covers it.

**The finding.** A Turing **scan** recording contains only the pulses its sweeping receiver was
tuned to at their ToA — that is D3's own headline measurement (99.985% of 4,393,233 train scan
pulses fall inside the active dwell's window), read the other way round. So a truth grid built
from a scan recording has its content sitting where that one fixed schedule looked, and any
scheduler that sweeps is handed the answer.

**Measured this session**, `config_2`, one seed, both replays of the same config:

| policy | scan replay | | | stare replay | | |
|---|---|---|---|---|---|---|
| | ratio | coverage | cTTI | ratio | coverage | cTTI |
| Turing reference sweep | **0.9999** | 1.000 | **0.00 s** | 0.0677 | 0.842 | 4.57 s |
| camper (band 6) | 0.1740 | 0.250 | 17.75 s | 0.0548 | 0.263 | 15.36 s |
| random | 0.0076 | 0.300 | 19.63 s | 0.0685 | 0.895 | 4.36 s |

A censored mean intercept time of **0.00 s** is the tell: every emitter is found in the first
slot it is detectable, because it only became detectable when the sweep arrived.

**Noticed while measuring this, and owed to whoever writes the baselines: `EVALUATION.md` §5's
baselines 2 and 3 are the same policy** unless one is deliberately changed. "Round-robin" —
step to the next band in order, each for its native dwell — *is* Turing's reference sweep, and
it reproduced the sweep's row above to four decimal places on both replays because it is not a
second measurement. They need to differ by something stated (a uniform dwell length, a different
starting phase, or a shuffled band order), or the ladder has six rungs and not seven. Not fixed
here: baselines are outside the environment and are not this session's scope.

**The mechanism, measured over all 47 train configs.** Comparing each contribution's cells
against the band `dwell_schedule()` is tuned to at that slot:

| | contributions | cells | mean cells each | on the swept band | within one band |
|---|---|---|---|---|---|
| scan-derived | 1,739 | 86,206 | 49.6 | **45.29%** | **99.42%** |
| stare-derived | 1,704 | 1,089,251 | 639.2 | 3.40% | 9.97% |

The reference sweep occupies one band per slot, so the no-bias base rate is 1/36 = 2.78%.
Stare sits at 3.40% — the small excess is the seven wide-dwell bands, where the sweep spends
double the airtime and the emitters are densest (D31). Scan sits at 45.29%, a **16× enrichment**,
and 99.42% within one band. The gap between 45% and 100% is the band overlap, not leakage: a
pulse falls in two adjacent ±500 MHz windows (D3) and the sweep is tuned to one of them, so
about half of a scan contribution's cells are the swept band and about half its overlapping
neighbour. **A scan-derived contribution is the emitter as seen through Turing's schedule**, not
the emitter.

At grid level: occupied cells are enriched at swept positions **14.5×** on a `config_2` scan
replay and **1.23×** on the stare replay. It is visible without any statistic — the waterfall of
a raw scan recording draws the sweep's sawtooth in signal (`rfenv/render.py`).

**Decision.** The scheduler comparison (`EVALUATION.md` §5) runs on the **47 stare replays plus
sampled scenarios. Never on scan replays.** Scan replays keep their existing roles, where the
imprint is not a contaminant but the mechanism being tested: gate 1 predicts the scan recording
*from stare* and never builds truth from scan at all, and gate 2's self-consistency check
(35.403% replayed against 35.700% recorded, D23) works precisely *because* the grid and the
schedule share an origin.

**What this does not change.** D25's pool, D24, D17's conclusion and D32 all stand — both runs
still feed the emitter pool, and no contribution is discarded. Nothing on the freeze list moves.

**The residual, stated rather than removed.** A *sampled* scenario draws one realisation per
emitter (D32) from a pool that is 1,739 scan and 1,704 stare contributions, so it inherits a
diluted version of the same imprint. Measured over 30 sampled scenarios at the default `n` draw
(seed 0): **53.4%** of the emitters drawn are scan-derived, but only **11.8%** of their cells,
because a scan contribution carries 49.6 cells against stare's 639.2. Composed grid enrichment
is **2.40×** on average (range 1.19–19.77; the tail is a low-`n` draw dominated by one scan
contribution).

**Why that residual is accepted rather than engineered away.** The obvious fix — sample only
stare-derived contributions — costs more than it buys. Stare's `freq_range_mhz` starts at
500 MHz, so **band 0 would be permanently empty in every training scenario** (D10), which is
exactly the known gate-1 limitation imported into the training distribution; it would also drop
the 383 emitters seen in only one run, and it moves the scenario sampling distribution, which is
on the freeze list (D25). A 2.40× residual on the training set is a smaller problem than a
structurally absent band. **Recorded so the RL lane knows it is there**, and it is a fair thing
for a reviewer to ask about.

**Consequence.** `EVALUATION.md` §5 now names the comparison set; §8 warns about reading a scan
waterfall; `ENVIRONMENT_SPEC.md` §Outputs says which replay the visual comparison is drawn from.
`rfenv/render.py` and `tests/test_metrics.py` carry the warning where it is easiest to trip over.

**Evidence.** Measured 2026-09-04 from `rfenv.scenario`, `rfenv.truth`, `rfenv.env` and
`rfenv.constants.dwell_schedule()` over `data/turing/*/train_*`, at the frozen γ = −111 dB.
Reasoned from D3 (the 99.985% in-band figure), D10, D24, D25 and D31.

---

## D37 — gate 1 is scored per dwell, against the raw scan ToA stream

**Status:** `SETTLED` (2026-09-04) — **accepted by the team.** Fixes the convention
`EVALUATION.md` §6 explicitly deferred to `validate.py`, **before** `validate.py` is written and
therefore before any number is seen.

**Why this needed deciding at all.** The 2026-09-03 audit found gate 1's headline figures
(accuracy 86.19%, precision 87.87%, recall 71.14%, MCC 0.694, r = 0.940) irreproducible: they
came from a scratch script at γ = −110, not the frozen γ = −111, whose comparison convention was
never written down. Four defensible conventions rebuilt from `rfenv` spanned **accuracy
83.5–86.0%**. The audit's own conclusion was that this — a headline number from a script whose
convention was not recorded — is *the* failure mode this repository exists to prevent, and that
the structural fix is to define the convention in code.

**Decision: per dwell, against the raw scan ToA stream.** For each dwell in
`dwell_schedule()`, the environment (truth built from **stare only**, D17) predicts whether that
dwell declares a detection; the observation is whether the **scan recording** actually contains
a pulse in that band window during that dwell. Score the 2×2 table over all dwells of all 47
train configs.

**Why, against the three alternatives.**

1. **Per dwell, not per cell.** A dwell is the unit a receiver produces a declaration for, and
   §2 defines the metric as "the environment's predicted detection matches the recorded scan
   data" — a detection is per look. Scoring all 36×600 cells instead would put ~97% of the
   denominator on cells the scan recording could never have observed (the sweep visits 2.78% of
   them, D36), making most of the score unfalsifiable by construction.
2. **Against the raw ToA stream, not against a scan-built grid.** Grid-versus-grid is tidier —
   both sides pass through identical band-assignment and slot-clock code — but that is precisely
   what disqualifies it. It would compare our pipeline against our pipeline, and gate 1's whole
   claim is that it is *"the only gate that is a genuine prediction rather than a fit"*
   (`EVALUATION.md` §6). Every construction step inserted between the prediction and the data
   weakens that claim.
3. **Not "report the range."** Publishing 83.5–86.0% would be honest about convention
   sensitivity and useless as a gate: it leaves the project with no gate-1 number and invites a
   reader to quote the flattering end.
4. It is the accounting the **pipeline self-consistency test already uses** — the recorded
   non-empty *dwell* rate, 35.403% replayed against 35.700% recorded (D23) — so gate 1 and gate
   2 are counted the same way and their numbers can be read side by side.

**What is fixed, and what is still open.** This settles *what is compared*. It does not settle
gate 2's and gate 4's pass criteria, which are still owed and must likewise be fixed before the
first run rather than after.

**Standing limitation, unchanged.** Band 0 (250 MHz) is 59.12% occupied in the recordings and
0.00% predicted, because stare cannot see below 500 MHz (D10, D24). Stated, not patched, and the
per-band bar plot shows it rather than hiding it.

**Evidence.** Sourced: `EVALUATION.md` §2, §6 and the 2026-09-03 audit finding 1. Reasoned from
D3, D10, D17, D23, D24 and D36. **No new measurement — deliberately.** The convention is fixed
before the gate runs, so that the number `validate.py` returns is a result and not a selection.

---

## D38 — the artefact set gains a run header; §4 was not computable from two tables

**Status:** `SETTLED` (2026-09-04) — implementation correction inside `EVALUATION.md` §8, taken
while building `metrics.py`. Recorded rather than silently patched because it changes an artefact
contract `validate.py` will be written against.

**The gap.** §8 asserted that "everything in §4 is computable from artefacts 1 and 2 alone" — the
per-slot episode log and the per-emitter table. It is not, by two of §4's five rows:

- **Interception ratio's denominator.** The numerator is in the log (illuminations inside the
  tuned window, summed over slots); the denominator is the scenario's *total* illuminations,
  which is grid-level and appears in neither table. Measured on a `config_2` stare replay under
  round-robin, the two differ by a factor of about 15 — the numerator alone would silently
  become a different metric.
- **Average reward / cost.** In neither table. Reward is per slot (D31), but candidate 3 pays out
  on set membership over a whole dwell, so there is no honest per-slot column for it.

**Decision.** A third artefact, `run.json`: scenario, scheduler, seed, reward name, γ, σ,
episode length, slot and band counts, **total illuminations**, **total reward**, `|E|` and step
count. Both missing quantities are scalars, so a scalar header is the right shape — widening the
600-row log to carry a constant would be worse.

**A second job it does.** `|E|` and the step count are recoverable from the tables, and are
stored anyway: `metrics.read_run()` checks the header against the row counts, so a truncated or
half-written artefact set raises rather than scoring low. A run that silently scores low looks
like a result, which is the failure this repository is organised around.

**Consequence.** `EVALUATION.md` §8 now lists five artefacts and states that §4 is computable
from 1–3. `ENVIRONMENT_SPEC.md` §Outputs matches. `metrics.py` scores §4 by reading the files
back rather than from live environment state, and
`tests/test_metrics.py::test_the_artefacts_reproduce_the_environments_own_numbers` asserts that
result is identical to `env.episode_metrics()` — which is what makes §6's "reproduces all four
gates from these artefacts alone" a checked claim instead of an intention.

**Evidence.** Measured 2026-09-04: the artefacts written by `rfenv.metrics` for `config_2` stare
replays under round-robin and a camper reproduce `ScanEnv.episode_metrics()` exactly on all eight
shared keys. Sourced: `EVALUATION.md` §4, §8; D31.

---

## Consistency audit — 2026-08-30

Requested by the team: a check that the decisions form one coherent story. Result: **two real
inconsistencies found and fixed, three tensions clarified.** Everything else holds together.

| # | Check | Outcome |
|---|---|---|
| 1 | D14's claim that only off-spec "coverage" rescued the problem | **Wrong — fixed.** PS-mandated intercept time, censored properly, defeats the camper (see D14 amendment). We do not need metrics the PS didn't ask for. |
| 2 | Measurement scripts used two different pulse→band conventions (all-covering-bands vs last-band-wins) | **Fixed.** Canonical rule: a pulse is intercepted iff the scheduler's tuned window contains it at its slot. The 2026-08-29 exact figures shifted slightly (e.g. greedy 90.3%→85.0% per-dwell); directions unchanged. |
| 3 | D1 (generative environment) vs D4's prototype grid (built empirically from recordings) | **Resolution below is OVERTAKEN BY D25 (2026-09-01) — do not build from this row.** It said the target signal model was *generated* (Apfeld's Eq. 1 fed by Turing metadata) and that generated-vs-empirical agreement would be "itself a validation gate". D25 dropped the physics signal model outright: its antenna pattern is published nowhere and its power scale would have to be fitted to the same recordings the primary gate scores. **There is no fifth gate**; the four in `EVALUATION.md` §6 are all of them. What survives is the framing D25 gives: the environment is generative in its *scenario structure*, over recording-derived signal values. |
| 4 | Union-truth construction vs replay validation (circularity) | **Real risk — addressed by D17's out-of-sample gate.** |
| 5 | D3 "adopt receiver exactly" vs undefined slot length | **Closed by D16.** |
| 6 | "Stare as partial ground truth" phrasing vs union construction | **Consistent** — D1/D2 already define truth as constructed, recordings as evidence. Union uses all evidence; no framing change. |
| 7 | 47-scenario scale vs RL training needs | **Consistent via D18** — generative env turns 47 templates into a distribution. The env-construction dependency chain ("RL depends on env; env from 47; good env → RL fine") holds with the D18 requirement added. |
| 8 | D7 (reward as hyperparameter) vs D14 amendment | **Consistent and sharpened** — candidate rewards are judged on censored TTI *and* interception ratio jointly; a candidate that wins one by sacrificing the other loses. |

---

## Consistency audit — 2026-09-01 (after ADITI, teammate docs, env spec)

Second pass, triggered by new documents and the consolidated `ENVIRONMENT_SPEC.md`. No
contradictions with D1–D19 introduced. Points checked:

| Check | Outcome |
|---|---|
| ADITI 4.0 CEW PS vs our SIH PS | **Sibling, not identical.** ADITI (Indian Army) is the broader Cognitive EW system — ES + EA, fusion, jamming, geolocation. Ours (SIH26055, DRDO) is the **ES scan-scheduling slice** of that vision. ADITI is valid *context* for intent (D20); it does not expand our scope to jamming or DF. Recorded so nobody imports EA requirements. |
| Teammate env `rf_env_grounded.py` vs our spec | **Adopted at the interface (L3), superseded inside.** Its action space, POMDP framing and history-based observation are sound and PS-grounded; we keep them and credit it. Its truth grid is a binary placeholder; L0–L2 replace that with the measured signal grid. No conflict — it stops where our data work begins. |
| Teammate crash-course `SIH- Smart Scan Strategy.pdf` | **Consistent and confirmatory.** Its detection model (Z truth vs Y declaration, Pd/Pfa, ROC, γ), its noise floor arithmetic (kTBF ≈ −109 dBm at 1 MHz), and its bandit framing all match D4/D6/D21 independently. Good onboarding + PPT source. |
| `Dynamic Scan Scheduling.pdf` (Dutertre) vs D3 | **Note only.** Dutertre assumes *disjoint* bands and treats the schedule as NP-hard time-allocation. Our bands overlap (D3) and our action is next-band, not a full cyclic schedule. Useful as a baseline-family reference and for the D20 "fixed a-priori table is the limitation" quote; not adopted wholesale. |
| PassiveRadar.pdf | **Off-topic, filed BACKGROUND.** Passive *bistatic* radar (transmitters of opportunity) is a different sense of "passive" than passive ES receiving. No bearing on our design. Guard against conflation. |
| Deinterleaving papers (Nuhoglu & Cirpan; Radar_Signal_Deinterleaving) | **Consistent with D12/D19** — deinterleaving stays out of scope; these are references if a future observation channel ever needs per-emitter attribution. |
| ENVIRONMENT_SPEC three-layer design vs D1–D19 | **Faithful consolidation.** Every layer cites its decisions; the one narrowing (v1 recording-built grid) is made explicit in D22, not smuggled. |

**Net:** the new material *confirmed* the direction more than it changed it. The one genuine
scope risk — ADITI's much larger CEW brief bleeding jamming/DF/fusion into our ES-only task — is
now fenced off explicitly.

---

## Consistency audit — 2026-09-03 (D1–D31, architecture, spec, code)

Third pass, requested by the team before L2/L3 implementation: check that every decision is
consistent with every other one and with the built code, and re-run whatever could be re-run.
Scope was D1–D31, `PROJECT_ARCHITECTURE.md`, `ENVIRONMENT_SPEC.md`, `EVALUATION.md`,
`RESEARCH_MAP.md`, `RL_LANE_HANDOFF`, and `rfenv/` L0–L1 with its tests.

**Result: the decision spine has no contradictions. Three quoted numbers did not reproduce, one
implementation did not match its decision, and the surrounding documents had drifted.**

### The spine holds

D1 → D4 → D26 → D5/D27 → D28 is coherent end to end, and `first_e ≥ on_e` is true by
construction rather than by luck. D24 → D25 amends D17 and D22 without leaving a rejected path
alive. D31 is consistent with D3, D16, D7 and D14; D29 with D7 and D21. Nothing in D1–D31 needs
reopening.

### Re-run and exact

Measured this session against `data/turing`, all 47 train pairs:

| Check | Documented | Re-run | |
|---|---|---|---|
| Pool contributions | 3,443 | 3,443 | exact |
| Distinct emitters in pool | 1,913 | 1,913 | exact |
| Per-config detectable emitters | 1–82 | 1–82 | exact |
| Transmitters never detectable | 19.0% | 19.04% | exact |
| Recorded non-empty dwell rate | 35.70% | 35.70% | exact |
| P<sub>fa</sub> at γ = −111 | 1.35e−3 | 1.35e−3 | exact |
| Sensitivity at γ = −111 | −107.2 dB | −107.16 dB | exact |
| Test suite | 23 pass | 23 pass | exact |

New, and no issue found: **zero** of the 4,393,233 train scan pulses fall outside every band
window, and zero fall outside the 30 s episode — so the interception-ratio denominator contains
no unreachable illuminations.

### Findings, and what was done about each

| # | Finding | Action |
|---|---|---|
| 1 | **Gate 1's headline numbers do not reproduce.** `EVALUATION.md` §6 quoted accuracy 86.19%, precision 87.87%, recall 71.14%, MCC 0.694, per-band r = 0.940 at γ = −110 — *not* the frozen γ = −111. Rebuilt from `rfenv` across four comparison conventions (per-dwell and per-cell, against the raw ToA stream and against a scan-built grid), none reproduces them; the range is accuracy 83.5–86.0%, precision 88.5–89.5%, recall 68.2–69.5%, MCC 0.66–0.69, per-band r ≈ 0.93. Directionally the gate passes, and band 0 behaves exactly as documented (59.12% recorded, 0.00% predicted). The original convention was never recorded, so the discrepancy cannot be attributed. | `EVALUATION.md` §6 now labels these **pre-gate, convention unrecorded**, not "measured". `validate.py` must define the convention in code; whatever it returns becomes the number. |
| 2 | **P<sub>d</sub> = 0.822 depends on an unstated cell population**, and `EVALUATION.md` §3's own wording ("only cells the receiver actually looked at") would make P<sub>d</sub> scheduler-dependent, contradicting D21. | 0.822 **withdrawn**; the three candidate populations measured and recorded as **D33** (`PROPOSED`). |
| 3 | **`Scenario.sample` could draw the same physical emitter twice** — 23.9% of sampled scenarios, 59.0% at `n = 82` — because 1,530 emitters sit in the pool as both a scan and a stare realisation. Contradicts D25's own justification and double-counts in `E`. | Fixed in `scenario.py` and recorded as **D32**. Landed before the freeze. |
| 4 | **The base observation vector (36×3+1) was never gated**, while D30 — its proposed extension — correctly is. D19 still read "observation contents `OPEN`". | Recorded as **D34** (`PROPOSED`); D19's status amended. |
| 5 | **`RL_LANE_HANDOFF` contradicts D29 and D31.** Its §6 assigns the per-dwell accounting question and the reward-on-`Y`-or-`Z` question to the RL lane as undecided; D31 and D29 settled both. Its §9 candidate list includes a staleness-shaped reward, while D29 fixes the set at three. This is the document the teammates have already read. | HTML patched to point at D29/D31. **The PDF is stale until regenerated — owner: the human team.** |
| 6 | **A superseded evaluation draft still sat inside this file**, redefining "% correct predictions" and "average intercept time error" differently from `EVALUATION.md` §2, keying intercept time to D2 activity windows (D27 replaced them), and phrasing gate 1 against the union grid (D24/D25 abolished it). | Section retitled **SUPERSEDED** with the four conflicts listed inline. Kept for history. |
| 7 | **The 2026-08-30 audit row 3 still promised a physics signal model** and asserted a fifth validation gate ("generated and empirical grids must agree"). D25 dropped the model; `EVALUATION.md` §6 has four gates. | Row annotated as overtaken by D25. |
| 8 | Smaller drift: `PROJECT_ARCHITECTURE.md` said "40 scenarios" in four places (it is 47); `EVALUATION.md` §8 still named occupancy `O`, the symbol D26 abolished; `RESEARCH_MAP.md` §Unresolved still listed D4 and D5's sub-question as awaiting a human, both since closed; D5's own text still said "still open inside this"; `scenario.py`'s sampler docstring described metadata transmitter counts (2–99) where the code correctly uses detectable counts (1–82). | All corrected. |

### Not re-run, and therefore still quoted rather than verified

D14's scheduler comparison table, D28's beam-phase measurement, D30's AoA attribution accuracy,
and D31's band-density, persistence and retune figures were **not** re-measured in this pass.
They remain as recorded, with their original session's evidence.

### Standing risk, recorded not fixed

Findings 1 and 2 are the same failure: a headline number produced by a scratch script whose
convention was not written down. That is the failure mode this repository was created to prevent.
The structural fix is `validate.py` — until a gate is a runnable check, its number is a claim.

---

# Evaluation plan — SUPERSEDED, kept for history

> **Do not build from this section. `docs/project/EVALUATION.md` is the single authority on every
> metric, baseline and gate** (`CLAUDE.md`; a metric is not defined in two places). This was the
> first draft, written 2026-08-29 before EVALUATION.md existed. The 2026-09-03 audit found it
> still in conflict with later decisions in four places, which is exactly the drift the
> single-authority rule exists to prevent:
>
> - **"% correct predictions"** is defined below as *was the top-ranked band actually occupied* —
>   a scheduler-ranking metric. `EVALUATION.md` §2 defines it as *environment-predicted vs
>   recorded detection agreement*, a model-level metric. Two different quantities under one PS
>   name; §2 is correct.
> - **"Avg intercept time error"** is defined below as a *scheduler's* prediction error;
>   `EVALUATION.md` §2 makes it the *environment's* prediction error, per the PS's own wording
>   ("the model should enable prediction of…").
> - **Intercept time** is keyed below to "D2 activity windows"; D27 replaced those with the
>   **detectable activity interval** derived from the grid.
> - **Gate 1** is phrased below against "the final union-built environment"; D24 and D25
>   abolished the union grid (scan and stare are independent simulation runs).
>
> Retained unedited because it records what we thought before measuring, which is worth keeping.

### The seven metrics the PS mandates

Probability of detection · probability of false alarm · sensitivity · average intercept rate ·
average reward/cost · percentage of correct predictions · average intercept time error. Plus
intercept time and interception ratio, named separately in the same paragraph. **These are not
ours to choose** — reporting a different set is a failure to answer the PS.

### How each becomes computable

| Metric | Where it comes from | Depends on |
|---|---|---|
| Pd, Pfa, sensitivity | Sweeping the detection threshold — an ROC over the signal grid | D4 |
| Interception ratio | **Per illumination** (Gul & Erer Fig. 2): fraction of all pulses whose band the scheduler was tuned to at their ToA — *not* per-dwell | D4, D5 |
| Intercept time | Delay from an emitter becoming active to first detection, **censored at episode end for emitters never found** — averaging only over found emitters rewards not looking (D14 amendment) | D2 activity windows |
| Avg intercept time error | Predicted minus actual intercept time | a scheduler that ranks bands |
| % correct predictions | Was the top-ranked band actually occupied | ranking, not a separate model |
| Avg reward / cost | The chosen reward, reported as a scalar | D7 |

### Three traps to design around

**A "predict nothing" model can score well.** Flagged in
`docs/reference/background/Smart Spectrum Surveillance for Electronic Support (ES) [BASICS].pdf` §6: with sparse occupancy, always predicting
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

1. **Out-of-sample prediction (primary).** Build truth from **stare only**, replay Turing's
   scan schedule through it, and compare predicted detections against the actual scan
   recordings — data never used in construction (D17). Then, on the final union-built
   environment, the replay should also recover the measured **~35%** non-empty dwell rate
   (35.3–35.7% depending on the final-partial-dwell convention) as a consistency check.
2. **Reproduce per-band structure.** Band-level interception ratios should match the recordings,
   not just the aggregate.
3. **Check against theory.** For a controlled periodic case, intercept time and probability of
   intercept should match Köksal's closed forms (`docs/reference/scheduling/optimumsearch.pdf` ch. 3.2, 6.1).
4. **Sanity-check the extremes.** `config_81` (2 emitters) and `config_921` (99) should behave
   sensibly at both ends.

Gate 1 is the one that matters most: it is a single number, measured from real data, that our
environment either reproduces or does not.
