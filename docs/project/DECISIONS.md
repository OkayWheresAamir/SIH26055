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

**Status:** `SETTLED` (2026-09-03). **Its three-candidate cap was lifted 2026-09-09 — see D57;**
the truth/observation split below is untouched by that and still governs. Not a new decision.
Records the answer to a question the
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
3. ~~**Open, and not decided here:**~~ **RESOLVED by D47 (2026-09-05).** The selection rule when
   candidates Pareto-dominate the baselines but not each other. D14 measured the two objectives in
   direct tension, so "whichever scores most" has no referent. A scalarisation, a lexicographic
   rule, or reporting the front — to be fixed **before** training, so the choice is not made after
   seeing results. **Fixed as paired dominance count over round-robin; see D47.** It was never
   blocking: `receiver.py` and `env.py` do not depend on it.

**Evidence.** Reasoned from D5, D7, D14, D21, D26, D28 and the architecture's own stage order.
The `Φ((S−γ)/σ)` curve is the L2 model definition, not a measurement. `Pd = 0.822` was quoted from
`EVALUATION.md` §3 (measured 2026-09-01) and not re-run for this entry — **it has since been
withdrawn (D33)**; read the two mentions above as "the aggregate Pd, whatever D33 fixes it at,
is a mixture over the per-cell curve". The argument does not depend on the value.

---

## D30 — AoA is a measured PDW field we discard; proposal to reconsider it for the observation

**Status:** `RESOLVED by D71` (2026-09-14) — **"they enter the observation, at this cost,"
graded.** AoA and PulseWidth both now enter, as an opt-in "v2" observation layout alongside the
original ("v1"), not a replacement for it. The trigger this entry set for itself was never
formally hit (no agent was measured failing to explore for want of these features) — D71 was
approved directly, on the strength of the case already made below, rather than waiting on that
trigger. **The cost this entry warned about was paid in a cheaper form than it predicted**: no
per-cell bearing list, no clustering, no deinterleaving problem (D19's concern) — see D71 for why,
and for what is deliberately still missing (the seen-vs-unseen discrimination this entry's own
measurement showed is AoA's actual value, which needs comparing bearings against each other, not
just reading one).

~~**Status:** `OPEN` (2026-09-04) — **owned by the RL lane, and off the pre-RL critical path.**
Re-scoped 2026-09-04 from "awaiting a human decision": the question is real and the measurements
below stand, but it cannot be answered well yet and it does not block anything before the freeze.~~

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
**EXTENDED 2026-09-09 — see D49, RESCALED see D55, and its amplitude exclusion lifted for `measured_dbm` by D59 (the base three are no longer the whole vector; `current_band`,
`current_band` and `measured_dbm` were added and it is now 36×4+2 = 146 wide, not 109 — D49, D55).**

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

## D39 — gates 2, 3 and 4 still need pass criteria, and they must be fixed before the first run

**Status:** `SETTLED` (2026-09-04) — **resolved in place; see §Resolution at the end of this
decision.** Criteria were fixed and written into `rfenv/validate.py::GATES` **before** the first
run, and `tests/test_validate.py::test_the_criteria_match_what_D39_records` asserts the code and
this record cannot drift apart. Recorded here rather than left in a chat log because it is the
exact failure the third audit named as this repository's standing risk.

**The problem.** `EVALUATION.md` §6 states four gates. Gate 1's comparison convention is now
fixed by **D37**. The other three are not testable as written:

| Gate | As written | What is missing |
|---|---|---|
| **2** | "Band-level interception ratios match the recordings, not just the aggregate." | *Match* by what measure, and how close is close enough. A per-band correlation, a per-band absolute error bound, and a rank agreement are three different tests that can disagree. |
| **3** | "A controlled periodic case matches Köksal's closed-form intercept time and probability of intercept." | Which controlled case (emitter period, dwell schedule, γ), which closed form, and what tolerance. |
| **4** | "`config_81` (2 emitters) and `config_921` (99) both behave sensibly." | *Sensibly* is not a predicate. Needs concrete assertions. |

**The constraint that makes this urgent, not tidy-up.** A pass threshold chosen *after* seeing
the measurement is not a gate — it is a post-hoc description of whatever the code did. The
2026-09-03 audit's closing note says so directly: "until a gate is a runnable check, its number
is a claim", and both of its findings 1 and 2 were headline numbers produced by scratch scripts
whose conventions were never written down. **So the criteria are decided first, and
`validate.py` reports pass or fail against them.** Where a threshold genuinely cannot be set
without knowing the scale, the honest form is a criterion justified by something other than the
measurement it judges — the base rate, a published figure, or a stated engineering requirement.

**Gate 3 carries a framing risk worth naming now.** Köksal's setup assumes the opposite of ours
on the one axis the problem statement cares about most: *"it is assumed that a pre-knowledge
about radars to be intercepted is available. Otherwise, there can not be any search strategy
that guarantee finite intercept times."* **D20** starts every episode with zero prior emitter
knowledge, which is the problem statement's own title condition. These do not actually conflict
— gate 3 checks that the **environment** reproduces an analytic result for a known periodic
case, where the pre-knowledge is the analyst's and never the agent's — but the distinction is
easy to lose, and "our system matches Köksal" would be a claim the architecture does not
support. State it as an environment check, never as a scheduler result.

*(That quotation is from a `docsearch` snippet of `docs/reference/scheduling/optimumsearch.pdf`
p.19 and has **not** been verified by opening the page. `EVALUATION.md` §6 cites ch. 3.2 and 6.1
for the closed forms, which may or may not be p.19. The next session must open the primary at
the cited pages before writing gate 3 — search locates a page, it does not read mathematics off
it.)*

**What is not open.** Gate 1 (D37), the comparison scenario set (D36), the artefact contract
(D38), γ and the population (D23, D33). Only the three criteria above.

**Evidence.** Sourced: `EVALUATION.md` §6, the 2026-09-03 consistency audit's closing note, and
a docsearch hit on `optimumsearch.pdf` p.19 (unverified against the page). Reasoned from D20,
D36 and D37. No new measurement — deliberately, for the reason in the constraint above.

### Resolution (2026-09-04)

The criteria below were decided and committed to code **before** `validate.py` was run for the
first time. They live in `rfenv/validate.py::GATES`, not in a config file and not as CLI flags:
a threshold you can pass on the command line is a threshold you can tune after seeing the number.

**Gate 2 — per-band structure.**

| | criterion | why this number |
|---|---|---|
| **2a** aggregate | `\|replayed − recorded\| ≤ 0.5 pp` | Both sides count the same 502 dwells per config, so there is **no sampling noise** and the tolerance is a mechanism bound, not a standard error. Every dwell boundary is an exact multiple of the 50 ms slot, so the only ways the two sides can disagree are the dwell truncated at 30 s and slot bucketing at the edges. 0.5 pp ≈ 2.5 dwells per config out of 502. |
| **2b** per band | `max over 36 bands \|replayed − recorded\| ≤ 1.0 pp` | 1.0 pp ≈ 6 dwells out of the 611–658 each band receives across the 47 configs (measured this session from `dwell_schedule()`: 502 dwells per episode, 13–14 per band). |
| **2c** direction | `replayed ≤ recorded` | `_bucket` floors a ToA into a slot and `bands_covering` uses the same ±500 MHz window as the recording check, so bucketing can drop a pulse at a boundary but cannot invent one. A predicted *sign*, which is stronger than a magnitude. |

**Gate 2's scope was also corrected.** `EVALUATION.md` §6's headline said "band-level
*interception ratios* match the recordings" while its own body described the *non-empty dwell
rate* self-consistency test. Those are different metrics, and the interception-ratio reading is
disqualified by **D36**: on a scan replay the reference sweep scores 0.9999 by construction, so
that clause would have measured nothing. Gate 2 is the dwell-rate test its body already
described; §6's wording now matches.

**Gate 3 — theory.** Controlled case fixed in advance, by stated rules rather than by outcome:
band 0 (native dwell 100 ms = 2 slots — a 1-slot dwell makes the discrete overlap
all-or-nothing and inflates the clock's error tenfold, measured 0.0365 against 0.0027); one band
only, so α is unambiguous; `τ_emit = 0.50 s`, `T_emit = 3.00 s`, giving `α = T_emit/T_rcv =
60/43` in slots — a large Farey denominator, so no synchronisation, and `T₀(P = 0.9) = 24.9 s`
fits inside the 30 s horizon; emitter level `γ + 5σ`, so the per-look miss probability is
`Φ(−5) = 2.9e−7` and the detector contributes nothing to a timing check.

| | criterion | why this number |
|---|---|---|
| **3a** `P₁₂(T₁)` | `\|env − discrete reference\| ≤ 0.01` | The reference is `validate._koksal_discrete_p12`, an independent pure-numpy model of the same two pulse trains that imports nothing from L1/L2 — asserted by a test. 0.01 = the clock term for this case (0.0027) plus ~3.7× margin for the phase sweep being 60 offsets rather than continuous. |
| **3b** Eq. (3.8) | `max first intercept over phase ≥ T_emit·T_rcv/(τ_emit+τ_rcv) = 10.75 s` | Read off the primary, body p.18. A lower bound on the **maximum** intercept time, so the check is one-sided. |
| **3c** | reported, **not** gated | The clock's own bias against continuous theory, and the `P₁₂(T)` divergence — see **D40**. |

Splitting 3a from 3c is what makes the tolerance derivable at all. Held directly to the
continuous closed form, gate 3 would measure the clock and the environment at once and neither
number would mean anything alone.

**Gate 3's framing risk, discharged.** The caveat this decision raised now travels with the
result instead of living in a chat log: `validate.gate3()` carries it in its docstring and in
`notes`, and `tests/test_validate.py::test_gate3_is_framed_as_an_environment_check_not_a_scheduler_claim`
asserts it is still there. Köksal's pre-knowledge is the analyst's, never the agent's; D20 is
untouched; no output of gate 3 is a scheduler result.

**Gate 4 — extremes.** Twelve structural assertions, no tolerances: five per config (`first_e ≥
on_e` per D28; every §4 metric inside its definitional range; no NaN or ∞, since censoring is
mandatory; 600 slots and 300–600 steps per D35; the artefacts reproducing
`env.episode_metrics()` per D38) plus two orderings over the pair (`config_81` has ≥ 1 detectable
emitter; `config_921` exceeds it in both detectable emitters and illuminations). Each restates a
decision or a metric definition — they hold or the code is wrong. Stare replays, driven by
`constants.dwell_schedule()`; **no baseline policy is implemented**, since the ladder comes after
the freeze.

**Deliberately excluded from gate 4:** "coverage falls as emitter count rises". D28 scores each
emitter on **its own** level, so an emitter's detectability does not depend on how crowded the
scenario is and the ordering does not follow from the design. Asserting it would have been a
threshold read off the answer. It is reported as a diagnostic instead.

**Gate 1 stays MEASURED.** D37 fixed the convention and explicitly left the threshold undecided
("whatever `validate.py` returns under it becomes the number"). This sat in the crack between
D37 and D39 and was surfaced while writing the module. Inventing a pass threshold here would be
the precise failure the gate machinery exists to prevent, so `validate.py` prints `MEASURED` for
gate 1, it cannot fail, and a test asserts that nobody adds a threshold without reopening D37.

**First run, 2026-09-04**, `python -m rfenv.validate`, 47 train configs, seed 0, γ = −111:

| gate | status | result |
|---|---|---|
| 1 out-of-sample prediction | **MEASURED** | accuracy 0.8585, precision 0.8819, recall 0.6932, MCC 0.6854, per-band r 0.935, base rate 0.3540, over 23,594 dwells (TP 5790 / FP 775 / TN 14466 / FN 2563) |
| 2 per-band structure | **PASS** | aggregate \|Δ\| **0.000 pp**, max band \|Δ\| **0.000 pp** — replayed 0.35403 against recorded 0.35403 |
| 3 theory (Köksal) | **PASS** | `P₁₂(T₁)` env 0.18333 vs reference 0.18333, \|Δ\| **0.0000**; Eq. (3.8) max first intercept 12.90 s ≥ 10.75 s |
| 4 extremes | **PASS** | 12/12 assertions |

Gate 1's figures land inside the audit's 2026-09-03 re-run range (accuracy 83.5–86.0%, recall
68.2–69.5%, MCC 0.66–0.69, r ≈ 0.93), and band 0 behaves exactly as D10 predicts: 58.97%
occupied in the recordings, 0.00% predicted. Operating point re-measured in the same run and
matching D33 exactly: Pd 0.85058, Pfa 1.3499e−3, sensitivity −107.155 dB over 11,710 occupied
reference-sweep cells.

**Evidence.** Sourced: `EVALUATION.md` §6, the 2026-09-03 consistency audit's closing note, and
`docs/reference/scheduling/optimumsearch.pdf` opened at body pp.18, 23 and 72 (PDF pp.33, 38 and
87 — the thesis carries 15 pages of front matter). Reasoned from D20, D36 and D37. The gate-3
sizing figures were computed from a standalone numpy model of the two pulse trains, deliberately
not from the environment. All four gate results measured 2026-09-04 by `python -m rfenv.validate`.

---

## D40 — Köksal's `P₁₂(T)` does not apply to a deterministic periodic pair, and is not gated

**Status:** `SETTLED` (2026-09-04) — found while sizing gate 3, before writing it. Evaluation
semantics, so it was gated by `CLAUDE.md`; nothing in D1–D39 covers it.

**The finding.** Table 6-1 of `docs/reference/scheduling/optimumsearch.pdf` (body p.72, PDF
p.87 — read visually from the rendered page, not from a text extraction) gives three things:
a single-period coincidence probability `P₁₂(T₁)` in four cases, a multi-period form

> `P₁₂(T) = 1 − [1 − P₁₂(T₁)]^(T/T₁)`

and an observation time `T₀ = T₁ · ln(1−P₀₁)/ln[1−P₁₂(T₁)]`. **The multi-period form compounds
the first-period probability as though successive receiver periods were independent Bernoulli
trials.** Body p.71 states the assumption in Köksal's own words: Hatcher derived these "by using
the pulse train model introduced in Sec. 2.2 **and assuming that the starting time instants of
the pulse trains are independent**."

**Why it fails here.** For two *strictly periodic* trains at a fixed phase, successive periods
are not independent at all — the relative phase drifts deterministically and sweeps the phase
space systematically, so coverage is far faster than an independence model allows.

**Measured**, over four candidate controlled cases, exact enumeration of every whole-slot phase:

| case (τ_rcv/T_rcv, τ_emit/T_emit) | `P₁₂(T₁)` closed | `P₁₂(T₁)` discrete | Köksal `P₁₂(30 s)` | **actual P(by 30 s)** |
|---|---|---|---|---|
| 0.10 / 2.15, 0.50 / 3.00 | 0.18062 | 0.18333 | 0.938 | **1.0000** |
| 0.05 / 2.15, 0.50 / 3.00 | 0.16395 | 0.16667 | 0.918 | **1.0000** |
| 0.05 / 2.15, 1.00 / 5.00 | 0.16349 | 0.20000 | 0.917 | **1.0000** |
| 0.10 / 2.15, 0.30 / 4.00 | 0.09477 | 0.08750 | 0.751 | **1.0000** |

The single-period form reproduces to ~0.003; the multi-period form is off by 6 to 25 points,
always in the same direction. Confirmed through the full environment stack in the first gate-3
run: Köksal 0.9379 against the environment's 1.0000.

**Decision.** Gate 3 gates on **`P₁₂(T₁)` and Eq. (3.8) only**. `P₁₂(T)` is computed, printed
and stored in `gate3.json` as a reported divergence with its cause, and is never a pass
criterion. **Gating on it would fail a correct environment** — which is the whole reason this is
a decision and not a footnote.

**Why this is consistent with Köksal rather than a rejection of him.** His ch. 3 is the
deterministic treatment (Diophantine approximation and Farey series) and yields *finite
guaranteed* intercept times for exactly this case; ch. 6 exists because he wants a probabilistic
alternative where ch. 3's optimisation fails. Applying ch. 6's independence model to a
deterministic pair is our error to avoid, not his.

**Third row is why the controlled case uses a wide band.** With `τ_rcv` equal to one 50 ms slot
the discrete overlap becomes all-or-nothing and the clock's error jumps to 0.0365. Band 0's
100 ms dwell keeps it at 0.0027, an order of magnitude below the 0.01 gate-3 allows for the
environment itself.

**Also noted, not acted on.** §3.2.3's step 1 says "calculate α as in (2.3)", but (2.3) on body
p.12 is `duty cycle = τ_rcv / T_rcv`. The thesis's own cross-reference is wrong. A further reason
gate 3 does not build on the Farey path, which would in any case require implementing (3.13) and
(3.14) — a small number-theory project for a check that Eq. (3.8) already covers one-sidedly.

**Evidence.** Sourced: `optimumsearch.pdf` body pp.18, 71, 72 (PDF pp.33, 86, 87), opened at the
page. Measured 2026-09-04: the four-case table from a standalone numpy model importing nothing
from `rfenv`; the environment figure from `python -m rfenv.validate --gate 3`.

---

## D41 — the recorded non-empty dwell rate is 35.403%, not 35.700%; the gap was a band-blind comparison

**Status:** `SETTLED` (2026-09-04) — found by gate 2 on its first run. **A published figure is
withdrawn**, so it is recorded rather than silently corrected.

**The finding.** D23 §3 and `EVALUATION.md` §6 record the pipeline self-consistency test as
"**35.403% replayed against 35.700% recorded**, the 0.3 pp residual being slot quantisation."
Gate 2 measures both sides at **35.403%** — they agree **exactly**, to 0.000 pp in the aggregate
and 0.000 pp on every one of the 36 bands.

**The cause, and it is not slot quantisation.** The recorded side of
`tests/test_truth.py::test_pipeline_reproduces_the_recorded_dwell_rate` **ignored frequency**: it
asked whether the recording held *any* pulse during the dwell, not whether it held one inside the
tuned band's ±500 MHz window. The replayed side reads `grid[band, slots]` and is band-aware by
construction, so the two sides were never like-for-like. Since 99.985% of scan pulses fall inside
the active dwell's window (D3), the two conventions differ by only ~0.3 pp — small enough to look
like a quantisation residual and be explained away as one.

**Why the band-aware convention is the correct one, independently of this result.** **D37**
already fixed it for gate 1: the observation is "whether the **scan recording** actually contains
a pulse **in that band window** during that dwell." Comparing a band-aware replayed side against
a band-blind recorded side is not a comparison. D37 also requires gates 1 and 2 to be counted the
same way so their numbers can be read side by side, which the old form prevented.

**Checked, not assumed.** Six conventions were measured this session before concluding the figure
does not reproduce: per dwell pooled (0.35403), mean of per-config rates (0.35403), excluding the
truncated final dwell (0.35403 — in fact `dwell_schedule()` truncates none), whole sweeps only
(0.36352), per slot looked at (0.41525), and per dwell via grid `Z` rather than `C` (0.35403).
None returns 35.700%.

**Consequence.** **35.700% is withdrawn.** The self-consistency test is now stated as *replayed
and recorded agree exactly at 35.403%*, which is a **stronger** claim than the one withdrawn: the
pipeline round-trips with no residual to explain. D23's conclusion is unaffected — γ is still not
calibrated against this rate, and the test still says nothing about detection. `test_truth.py` is
corrected to the band-aware convention and its tolerance tightened from 0.5 pp to 0.01 pp;
`EVALUATION.md` §6 and `ENVIRONMENT_SPEC.md` §L1 are updated.

**This is the fourth instance of one failure mode** — after gate 1's 86.19% (audit finding 1),
`Pd = 0.822` (audit finding 2) and D23's own withdrawn γ calibration: *a headline number produced
by a script whose comparison convention was never written down*. The 2026-09-03 audit called
`validate.py` the structural fix and this is it working as intended, on its first run, against a
number that had already survived one audit pass marked "exact".

**Evidence.** Measured 2026-09-04 by `python -m rfenv.validate --gate 2` over all 47 train scan
configs, 23,594 dwells; the six alternative conventions measured in the same session. Reasoned
from D3, D23 and D37.

---

## D42 — the environment is frozen

**Status:** `SETTLED` (2026-09-04) — **the freeze that `EVALUATION.md` §6 mandates on gate pass.**
Environment and pre-RL work is closed from here; it reopens only on evidence of an actual bug.

**What triggered it.** The four validation gates ran for the first time on 2026-09-04 and the
three gated checks passed (D39 §Resolution). `EVALUATION.md` §6: *"On pass, **freeze** everything
in `rfenv/constants.py`."* This records that the freeze is taken rather than merely due.

**What is frozen.** Everything in `rfenv/constants.py`: band geometry (36 centres, ±500 MHz
half-width), the slot clock (50 ms, 600 slots, 30 s), the native dwell schedule (seven 100 ms,
twenty-nine 50 ms, 2.15 s sweep), `N₀ = −120 dB`, `σ = 3 dB`, `γ = −111 dB`, `PD_POPULATION =
reference_sweep`, the split names the held-out guard keys off, and the truth-construction rule
and metric definitions that depend on them.

**How the freeze is enforced, which it previously was not.** Every other test in the suite reads
the constants *symbolically* (`K.GAMMA_DBM`, `K.SLOT_S`, …). That is correct for behaviour tests,
but it meant **the whole suite would have stayed green if someone changed γ, the slot clock or the
band geometry** — every expectation would move with the value and nothing would notice.
`rfenv/constants.py` called itself "the freeze list, literally" and had no literal.
`tests/test_freeze.py` is now that literal: nine tests pinning each value plus a SHA-256 digest
over the whole list as a single tripwire. Verified by injection this session — a γ change and a
band-half-width change both trip it.

**What is *not* frozen**, deliberately: reward candidates and observation-vector extensions stay
the RL lane's (D29, D30, D34), and the per-episode draw — which emitters, and the seed — is free
by construction (D25). **D29 item 3 (the reward-candidate selection rule, settled 2026-09-05 as D47) was
explicitly outside this freeze**: it is an RL-lane reward/Pareto decision, it touches nothing on the list, and the
freeze did not wait on it.

**Re-validation trigger.** D25's rule stands: if anything on the list moves, the environment is
re-validated from gate 1 and every baseline is re-run. `tests/test_freeze.py` says so in its own
failure message, so the rule is attached to the thing it governs rather than living only here.

### Validation limitations, recorded at freeze time

The gates passed, but they are not equally strong, and a reader should know which is which before
quoting one. Established by fault injection this session — each defect was *injected* and the gate
observed:

| Gate | What it genuinely tests | What it cannot detect |
|---|---|---|
| **1** out-of-sample prediction | **The only gate whose two sides use different data** (stare-built prediction vs the scan ToA stream). MCC 0.6854 over 23,594 dwells against a 0.3540 base rate is real evidence. | Band geometry — both sides apply the same half-width. Injected ±250: MCC *rose* to 0.5123 from 0.4988 on the same two configs. |
| **2** per-band structure | Slot-clock, dwell-schedule and bucketing consistency. Injected an off-by-one slot shift: **FAIL at −5.44 pp**, as it should. | **Its 0.000 pp is algebraically forced.** Dwell boundaries are exact slot multiples, so `s0 ≤ floor(t/Δ) < s1` ⟺ `start ≤ t < end`, and both sides share the band predicate — the two sides are the same function of the same input. Injected ±250 half-width on both sides with the cache rebuilt: **PASS at 0.000 pp**, with the rate moving 0.354 → 0.138. It is a consistency check, not a validity check. |
| **3** theory (Köksal) | Environment machinery — contribution → grid → receiver → D28 crediting → `first_intercept`. Injected an emitter 2σ *below* γ: **FAIL**, correctly. | Parameters. The reference is genuinely independent of L1/L2 (a test enforces it) but is **co-parameterised** — `gate3()` feeds both sides from the same `_G3_*` constants. Injected τ_emit 0.50 → 0.25: both moved to 0.100, **PASS**. Also: phase resolution is 1/60 = 0.0167, *above* the 0.01 tolerance, so 3a is exact-agreement-or-fail and the stated "3.7× margin" buys nothing. And 3b (Eq. 3.8) is a one-sided bound cleared with 20% headroom. |
| **4** extremes | Structural invariants and the full artefact round-trip at 1 and 72 detectable emitters. | Nothing beyond its twelve assertions; it is a smoke test by design. |

**The band half-width is the load-bearing assumption, and no gate covers it.** It is ±500 MHz, so
adjacent bands overlap by half. The TSRD paper says the receiver sweeps *"in 500 MHz steps and
500 MHz bandwidth"* — a disjoint tiling — and **we deliberately do not reproduce the paper here**:
the files disagree and `CLAUDE.md`'s authority table says the files win. Its sole evidence is D3's
measurement, **re-run this session: 99.9851% of 4,393,233 train scan pulses fall within ±500 MHz
of the dwell centre active at their ToA** (655 outside), against 50.92% for a disjoint ±250
tiling. That measurement is sound; it is simply not re-checked by any gate, which is why the
constant is pinned in `test_freeze.py` with this reasoning attached to it.

**No Turing performance result was reproduced, because none exists.** Checked at freeze time
rather than assumed: a full-text sweep of all six pages of the TSRD paper finds one line touching
this task (*"scan receiver model which sweeps the frequency"*, p.4) and no intercept, detection or
scheduling figures. `RESEARCH_MAP.md` already records it — *"Nothing about scheduling,
interception, or reward. It is a deinterleaving dataset paper."* Its §III evaluation framework is
the deinterleaving challenge's (V-measure), which D12 puts out of scope. **What was reproduced is
Turing's receiver *configuration*, exactly**: dwell centres and dwell times are asserted against
every HDF5 file by `scenario.load_receiver`, which raises on mismatch (verified this session on
both fields), sweep period `sum(dwell_times_s)` = 2.150000 s matches `SWEEP_S`, and
`collection_time_s` = 30.0 matches `EPISODE_S`. There is no stronger comparison available: the
recordings contain already-detected pulses, not a declared-detection stream, so no independent
reference exists for the receiver's *output*. **A check was not invented to fill that gap.**

**Pre-RL readiness evidence**, measured this session on the sampled-scenario path — which is what
RL trains on and which **no gate exercises**: 40 sampled episodes clean (no NaN, no truncation,
600-slot log every time, all metrics in range); observation stays inside its declared `[0, 1]` box;
same seed reproduces exactly and a different seed differs; all three reward candidates run;
the held-out guard fires on a `test_` path; throughput 36,140 steps/s (~4,300 episodes/min).

**Evidence.** Measured 2026-09-04: the gate results by `python -m rfenv.validate`; the fault
injections, the 99.9851% in-band figure, the receiver-configuration assertions and the
sampled-path checks by scripts run this session. Sourced: `EVALUATION.md` §6, `RESEARCH_MAP.md`,
TSRD paper p.4 and §III. Reasoned from D3, D8, D12, D25, D29, D30, D34 and D39.

---

## D43 — round-robin is equal-airtime, not Turing's sweep; the ladder now has seven rungs

**Status:** `SETTLED` (2026-09-04). Evaluation semantics, so it was gated by `CLAUDE.md`;
delegated by the team with the instruction "make them genuinely different but simple, defensible
benchmark policies."

**What was wrong.** D36 found, while building `metrics.py`, that `EVALUATION.md` §5's rungs 2 and
3 were **the same policy**. "Round-robin — step to the next band in order, each for its native
dwell" *is* `constants.dwell_schedule()`, and it reproduced the Turing sweep's row to four decimal
places on both replays because it was not a second measurement. The ladder had six rungs and
claimed seven. D36 named three possible separations — a uniform dwell length, a different starting
phase, or a shuffled band order — and left the choice to whoever wrote the baselines.

**Decision. Rung 2 is round-robin with equal airtime per band.** Two passes over the 36 bands per
cycle: the seven wide bands, which cost two slots per visit, are visited on the first pass only;
the twenty-nine narrow bands are visited on both. **Every band gets exactly 2 slots per 72-slot
(3.60 s) cycle**, against Turing's 1 (narrow) or 2 (wide) per 43-slot (2.15 s) sweep. A narrow
band's two visits sit ~43 and ~29 slots apart rather than adjacent, so the longer cycle does not
buy a longer worst-case gap than it has to. Implemented as `baselines.RoundRobin`.

**Why airtime and not phase or order.** A uniform *dwell length* — D36's first suggestion — is not
expressible: an action is a band and its dwell is frozen at that band's native Turing length
(D3, D16). The expressible form of the same idea is uniform **airtime**, which is the currency
anyway (D31, retuning measured under 1 µs). A shuffled order or a phase offset would leave the
airtime allocation identical to the sweep's and change only the coincidence phase against
individual emitters — a difference, but not a *strategic* one, and the two rungs would still be
measuring nearly the same thing.

**The two rungs now differ by a stated mechanism, and it is worth something.** Turing's sweep gives
the seven wide bands double airtime, and D36 measured that those are the bands where the emitter
population is densest — so the sweep carries a weak, correct prior about where the emitters are.
Rung 2 refuses that prior; that is what makes it the open-loop floor the PS names. **Measured over
57 scenarios × 3 seeds (below): the sweep's weighting is worth +33% interception ratio (0.0805 vs
0.0605) and 0.44 s of censored intercept time (3.74 s vs 4.18 s) at identical coverage (0.864 vs
0.865).** That is the ladder working: rung 3 is now telling us something rung 2 does not.

**Evidence.** `python -m rfenv.compare --seeds 3 --sampled 10`, 2026-09-04, 1,539 episodes at the
frozen γ = −111. The airtime claim itself is asserted against `constants.DWELL_SLOTS` by
`tests/test_baselines.py::test_round_robin_spends_equal_airtime_on_every_band` and
`::test_the_turing_sweep_spends_double_on_the_wide_bands`, and the two rungs are pinned as
distinct policies by `::test_round_robin_and_the_turing_sweep_are_not_the_same_policy` — on the
600-slot band sequence, not on a metric, because two policies can score alike by luck.

---

## D44 — Apfeld's Algorithm 1 contradicts its own prose; we implement the prose

**Status:** `SETTLED` (2026-09-04). A reading of an external source, recorded because the
alternative reading is defensible and someone will ask.

**The discrepancy.** `docs/reference/scheduling/paperSSPD (1).pdf` §II (p.2) says:

> "The probability for choosing each tentative frequency is scaled by *y* according to the number
> of frequencies in the list of tentative RFs until the scaled value reaches a maximum *z*. This is
> done to avoid dwelling on just very few frequencies with a very high probability."

Algorithm 1, printed immediately below it, reads:

> 1: `r ← random ∈ (0,1)` 2: `if r < min(|tentativeRFs|·y, z) then` 3: `return random band ∈ RFs \ tentativeRFs`

The pseudo-code makes `min(n·y, z)` the probability of choosing a **non**-tentative band. With one
tentative band that probability is `y`, so the receiver picks that single band with probability
`1 − y` — dwelling on exactly one frequency with a very high probability, which is the thing the
sentence says the cap exists to prevent. The prose reading makes `min(n·y, z)` the probability of
choosing a **tentative** band: it rises with the list and is capped so exploration never dies.

**Decision: follow the prose.** `baselines.Apfeld._algorithm_1` exploits the tentative list with
probability `min(n·y, z)`.

**Why, beyond the internal contradiction.** Implemented as printed with `y = 1/36`, the rung camps
on the first band that declares and reaches **11.1% emitter coverage on `config_921` stare**
(measured this session). Apfeld's own §III-B reports the opposite shape for their adaptive
strategies — they lose the "radars detected at least once" criterion *only* to Random, and by a
modest margin. An implementation that reproduces neither the prose nor the reported behaviour is
the wrong reading of an ambiguous source.

**A second bug of the same kind, and the same fix.** "Once the SNR on a band crosses the detection
threshold … the receiver stays tuned to that RF band for another *d* dwells" applies to the band
*entering* the tentative list. Re-arming the stay on every later detection sticks the receiver on
the first band it finds whenever the scenario is dense enough that every look declares — measured,
the same 11.1% coverage on `config_921`. Fixed; the comment sits on the line.

**Three adaptations, forced by the environment and not optional.** All are recorded in
`baselines.Apfeld`'s docstring and repeated here because they qualify every Apfeld number we
report:

1. **SNR series → binary detection series.** The paper autocorrelates the intercepted SNR
   (Eq. 2, Eq. 3). Our receiver declares `Y` and nothing else (D26, D28); there is no amplitude in
   the observation and putting one there would change D34. `f` is therefore the binary declaration
   series, zero where the receiver was tuned elsewhere — which is how the paper's own `f` behaves
   (Fig. 1b: *"at (most of) the points in time where the SNR is zero, the receiver is tuned to a
   different frequency"*).
2. **Scheduling anchor: last detection, not `SNR_max`.** Binary declarations have no maximum, so
   Eq. 4's `T_snr = x·SNR_max` collapses with it: a scheduled visit either declares or it does not.
3. **No tracking-dwell branch.** It needs the receiver to tell search dwells from tracking dwells
   by waveform; our PDW stream carries no such label and D12 rules out building one. This is
   therefore the paper's own **"Adaptive, no tracking"** variant, which §III-B finds performs
   equally well — so the reduction costs the comparison nothing the authors did not already
   measure.

**Parameters are ours, stated, and not searched.** The paper names `d`, `y`, `z`, `j`, `T_std`,
`s`, `x`, `i`, `k` and fixes none of them. `baselines.ApfeldParams` carries our values with a
justification per field that is independent of the score it produces (`d = 2`, `y = 0.1`,
`z = 0.8`, `j = 3`, `T_std = 1 slot`, `s = 2`). No sweep was run: tuning a baseline against the
metric it is judged on would make it a weak learned scheduler rather than a benchmark.

**Consequence for the claim.** §5 says "Beating Apfeld is the claim worth making." It remains
worth making, but it must be stated as **beating our adaptation of Apfeld's no-tracking variant on
a binary-detection receiver**, at our parameters. That is a fair claim; "beating Apfeld" without
those words is not.

---

## D45 — Apfeld's own ablation joins the ladder as rung 6a; the period estimate costs more than it buys here

**Status:** `SETTLED` (2026-09-04) for the rung; the measurement is a **finding**, not a decision.

**The rung.** D13 already named it: *"Its own baselines — Random, 'Active RFs',
adaptive-without-tracking — give us a ladder that maps onto the ablation study we need anyway."*
`baselines.Apfeld(use_period_estimation=False)` is Apfeld's third strategy, **Active RFs**: the
tentative list and Algorithm 1, with no autocorrelation and no scheduled revisits. It costs one
flag and it isolates how much of rung 6 is the period estimate and how much is just "revisit what
was loud". `EVALUATION.md` §5 gains it as rung **6a**.

**The finding: the period estimate makes things worse on this data.** Measured over 57 scenarios ×
3 seeds:

| rung | interception ratio | censored intercept time | coverage | beats round-robin on **both** |
|---|---|---|---|---|
| 6a Active RFs | 0.132 | 4.32 s | 0.860 | **53.2%** |
| 6 Apfeld (period estimation) | 0.245 | 14.86 s | 0.367 | 4.1% |

Rung 6 buys 1.9× the interception ratio and pays 3.4× the intercept time, ending with less than
half the coverage. It is not a Pareto improvement over its own ablation, and against round-robin
it Pareto-wins 4.1% of episodes against 6a's 53.2%.

**Why, and why this is not evidence the paper is wrong.** Two structural differences, both ours:

- **Our episode is 30 s; theirs is 5 min.** An autocorrelation needs at least two periods of
  evidence before a lag means anything, and a scheduled revisit only pays off if the episode
  outlasts several of them. In 600 slots there is often time to estimate a period and no time to
  profit from it.
- **Our series is binary** (D44 adaptation 1). The paper's autocorrelation runs on SNR, where the
  height of a peak carries information about beam geometry; ours runs on declarations, where every
  detection is a 1 and a false alarm is indistinguishable from a weak one.

So this is a result about **Apfeld's algorithm on a 30 s binary-detection problem**, not a
refutation of §III-B. Recorded because it is exactly the kind of ablation the write-up needs, and
because "the published adaptive strategy's clever half did not help here" is a claim we should be
able to defend with a number.

**Evidence.** `python -m rfenv.compare --seeds 3 --sampled 10`, 2026-09-04, artefacts under
`runs/baselines/`.

---

## D46 — the baseline ladder, first run: what it measured

**Status:** measured 2026-09-04. A **result**, recorded here so the numbers have a provenance and
a date; the decisions it depends on are D13, D14, D36, D43, D44 and D45.

**The run.** `python -m rfenv.compare --seeds 3 --sampled 10 --figures`, train split only, 47
stare replays + 10 sampled scenarios × 3 noise seeds = **1,539 episodes**, at the frozen
γ = −111 dB, σ = 3 dB, P_fa = 1.3499e−3, reward `hit_z`. Never on scan replays (D36) —
`compare._check_comparison_scenario` refuses one. Every row is scored by
`metrics.scheduler_metrics()` **from the artefacts on disk**, the same function and the same path
for every rung.

| # | scheduler | interception ratio | censored intercept time (s) | emitter coverage |
|---|---|---|---|---|
| 1 | random | 0.0669 | 4.16 | 0.858 |
| 2 | round-robin (equal airtime) | 0.0605 | 4.18 | 0.865 |
| 3 | Turing reference sweep | 0.0805 | 3.74 | 0.864 |
| 4 | greedy camper (observation-fed) | 0.2088 | 9.67 | 0.497 |
| 5 | recency / activity | **0.1104** | **3.20** | **0.897** |
| 6a | Apfeld: Active RFs | 0.1320 | 4.32 | 0.860 |
| 6 | Apfeld adaptive (no tracking) | 0.2455 | 14.86 | 0.367 |
| — | camper, truth-fed (D14's) | 0.5680 | 15.71 | 0.296 |
| — | pulse-capture oracle | 0.6579 | 8.01 | 0.691 |

Paired per episode against round-robin — same scenario, same seed, same truth grid — the fraction
of the 171 episodes each rung wins on **both** headline metrics: recency **70.2%**, Active RFs
53.2%, Turing sweep 51.5%, random 42.1%, Apfeld 4.1%, camper 1.8%.

**Four things this measured that were not known before.**

1. **D14's tension survives the frozen environment**, and it is the *truth-fed* camper that
   embodies it: ratio 0.568 against round-robin's 0.061, censored intercept time 15.71 s against
   4.18 s, coverage 0.296 against 0.865. The shape reproduces D14's amendment (57.4% / 23.78 s /
   30.4%); the intercept times are lower across the board because D27 measures the delay from
   `on_e` rather than from t = 0, which is a definition change made after D14 and not a
   disagreement.
2. **A camper that cannot see truth is a much weaker camper.** Rung 4 probes for three sweeps and
   camps on the largest *observed hit rate*; it reaches ratio 0.209 against the truth-fed 0.568.
   Illumination density is truth-side (D29) and binary declarations are a poor proxy for it — the
   busiest band and the most reliably-occupied band are not the same band. **The 57.4% headline in
   D14 was never achievable by a deployable scheduler**, and the ladder now says so with two rows
   instead of one.
3. **The pulse-capture oracle is a ceiling for one axis only.** It loses censored intercept time to
   plain round-robin on **80.7%** of episodes (8.01 s against 4.18 s on average). D14 said as much
   in a parenthesis — "each column has a different optimum, which is itself the point" — and it is
   now measured. Anyone reading `oracle_pulse` as *the* ceiling will read the whole table wrong,
   which is why `EVALUATION.md` §5 keeps it under a dash.
4. **The bar for the RL rung is rung 5, not rung 2.** A one-line index policy — `argmax(hit rate +
   gap in sweeps)` over two components of the D34 observation — Pareto-dominates the floor on 70%
   of episodes and beats the Turing sweep on all three reported metrics. Beating round-robin is
   not the interesting claim any more; beating `recency` is, and beating the deployable camper's
   ratio (0.209) *while* holding recency's intercept time is the Pareto target.

**Caveats, stated rather than buried.** (a) The reward is `hit_z` throughout and moves only the
reward column (D7, D29) — no reward was selected. (b) Sampled scenarios inherit D36's 2.40×
residual scan imprint. (c) Three seeds, not five; spread is in `runs/baselines/summary.json` and
every mean above is printed with its interquartile range in `comparison.md`. (d) P_d over these 57
comparison grids is 0.8421, **not** the 0.85058 in `runs/validation` — D33 froze the population
*rule*, not the set of grids, and the validation figure is over 47 scan-replay grids. Same rule,
different worlds; neither is quotable without naming its grids.

**Evidence.** `runs/baselines/{summary.json,metrics.json,comparison.md}` and the 1,539 artefact
sets under it, all written 2026-09-04.

---

## D47 — the reward-candidate selection rule: paired dominance over round-robin

**Status:** `SETTLED` (2026-09-05) — **decided by the team**, closing D29 item 3, the last thing
D29 left open. Fixed **before** any training run, which is the whole point of it.
**GATED 2026-09-10 by D62** — a candidate is only eligible for this rule once it has passed the
cheap screen, and **as of that date only `reward_balance` has**. Two of the three candidates D29
registered (`hit_z`, `hit_y`) rank rung 4 above every sweeping policy and are excluded.
**This rule has still never been run.**

**The question.** D7 says the reward is a hyperparameter chosen after comparison on the PS's own
metrics. D14 then measured that the PS's two metrics are in direct tension, so "whichever scores
most" has no referent — a candidate can win interception ratio and lose censored intercept time,
and usually will. D29 recorded the gap and deliberately left it open.

**Decision. The winning reward candidate is the one that beats round-robin on *both* headline
metrics on the largest fraction of paired episodes** — same scenario, same seed, same truth grid.
`compare.paired_wins()` already computes exactly this and prints it as the `both` column. If two
candidates land within **5 percentage points** of each other, no candidate is selected: both are
reported and the choice is escalated.

**Why this rule and not a priority order.** Three rules were put to the team: this one, censored
intercept time first, and interception ratio first. The team chose this one, on the grounds that
**the problem statement names both objectives and does not rank them**, so a rule that ranks them
would be importing a preference the PS does not state.

Two further reasons it is the right shape for *this* project:

- **It needs no exchange rate.** The two metrics are not commensurable — interception ratio is a
  fraction in [0, 1], censored intercept time is seconds in [0, 30]. Any weighted sum is a claim
  that one point of ratio is worth *N* seconds, and that claim would have to come from a mission
  we have not been given. A dominance count never forms the ratio.
- **Every trap this project has hit came from one metric winning alone** — the camper on
  interception ratio (D14), the uncensored intercept time that made the same camper look *fast*
  (D14's amendment). A rule that demands both is the one that keeps reproducing that lesson rather
  than falling for it.

**Its two weaknesses, recorded rather than argued away.**

1. **It is insensitive to margin.** Beating round-robin by 0.001 counts exactly as much as beating
   it by a factor of three. A candidate that is spectacular on one axis and merely adequate on the
   other scores worse than one that is mildly better on both.
2. **"Both equally" is itself a weighting**, just an implicit one. The rule does not escape having
   a preference; it declines to make the preference numeric.

Neither is fatal, because §4 requires the full table to be published beside the winner regardless —
the rule picks, it does not summarise.

**Two consequences for how the RL rung is reported.**

- **Round-robin is the reference in the rule, not the bar to clear.** It is the paired denominator
  because it is the PS's own named floor and it is open-loop, so it cannot have learned anything
  from the scenario. That it is a *weak* opponent — random beats it 42.1% of the time (D46) — is
  irrelevant to its use here: the same denominator is applied to every candidate, so it cancels.
  **The bar to clear is still rung 5 (D46), and it should be stated separately.**
- **A margin-based reading must be published alongside**, because of weakness 1. The full §4 table
  with interquartile ranges is already mandatory (§7); this just says it is not optional when the
  winner is announced.

**Noted for later, not decided: the two rejected rules have practical merit as a variant, not as
this rule.** The team observed that a priority order — speed first, or capture first — diverges
from the PS's own framing but could match a real mission: threat warning wants time, emitter
characterisation wants pulses. If a mission-specific variant is ever built, it is a **second
system with a stated mission**, chosen by a rule recorded at that time. It does not reopen this
one, and it may not be selected after seeing which corner looked better.

**Provenance of the recommendation, stated because it matters.** The rule recommended to the team
was the dominance count, and the counter it uses (`compare.paired_wins`) was written in the same
session by the same author, whose best-performing baseline scores highest under it. The team was
told this before choosing. The contamination is limited but real: the rule is applied to three
*trained policies* that do not exist yet, so no result the rule will judge has been seen — but
baseline numbers were shown as calibration, and calibrating a rule is one step from choosing it.
Recorded so a reviewer can weigh it.

**Evidence.** Reasoned from D7, D14, D28, D29 and D46. The `both` column and the 5 pp margin are
computed by `rfenv/compare.py::paired_wins`, which `tests/test_compare.py` pins against
hand-built rows including the case where a candidate wins each metric in a different scenario and
must therefore score zero.

---

## D48 — RL joins the ladder: rungs 7 (DQN), 8 (PPO), 9 (Recurrent PPO)

**Status:** `SETTLED` (2026-09-09) — built, tested, and registered in `rfenv/baselines/ladder.py`.
Closes the gap `CLAUDE.md`'s own build status named: *"Rung 7 (RL) is the only thing missing from
the ladder."*

**Three algorithms, one adapter shape each.** All three are `stable-baselines3` (rung 9 also needs
`sb3-contrib`, which SB3 itself does not ship), trained by `rfenv/rl/{dqn,ppo,recurrent_ppo}.py`
against `rfenv/rl/common.py::make_train_env` — `ScanEnv(pool=EmitterPool.from_train(), reward=...)`,
never a fixed replay, so every training episode draws a fresh scenario (D25, D32) and nothing is
ever memorised. Two policy adapters, not one, because a recurrent policy's `.predict()` has a
different shape than a feed-forward one's:

- **`RLScheduler`** wraps DQN/PPO's plain `predict(obs, deterministic=True)`. Stateless.
- **`RecurrentRLScheduler`** wraps rung 9's `predict(obs, state=..., episode_start=..., ...)`,
  carrying the LSTM's hidden state across calls within an episode and resetting it at each new
  one (inferred from `info["slot"] == 0`, not tracked by the caller).

Both read only what `guarded()` lets through — `RecurrentRLScheduler` touches exactly one `info`
field (`slot`, already in `OBSERVABLE_INFO`) purely to detect an episode boundary, never to
condition the action — so every RL rung is deployable under the same D19/D29 rule every other
rung already follows, not a special case.

**Multiple trained variants coexist; they do not overwrite each other.** One checkpoint per
algorithm was never going to be enough — different rewards, different timestep budgets, different
points along one run all want comparing. `_dqn_rung_factory`/`_ppo_rung_factory`/
`_recurrent_ppo_rung_factory` are parameterised by checkpoint path, so registering a new variant
is one `Rung(...)` line naming its own `.zip`, not a retrain-and-overwrite. Lettered sub-rungs
follow the numbering 6/6a already established for Apfeld's two forms: **7, 7a** (DQN); **8, 8a,
8b, 8c** (PPO, three `first_intercept` checkpoints at different timestep budgets); **9** (Recurrent
PPO). `run.md` documents the registration pattern and the training commands.

**Not decided here, still open:** which variant, if any, is the RL lane's answer to D47's
paired-dominance rule. No trained checkpoint has been run through `compare.paired_wins` against
round-robin as of this entry — rungs 7-9 exist and are comparable, nothing has been selected.

**Evidence.** `rfenv/rl/{common,dqn,ppo,recurrent_ppo}.py`, `rfenv/baselines/ladder.py`,
`tests/test_rl.py` (registration, deployability, checkpoint round-trips, and — specifically for
rung 9 — that hidden state actually threads across calls rather than resetting every dwell).

---

## D49 — the observation vector, extended past D34's base three

**Status:** `SETTLED` (2026-09-09) — built in `rfenv/env.py::ScanEnv._observation`, covered by
`tests/test_env.py`/`tests/test_baselines.py`. Exactly the kind of change D34 itself reserved:
*"extensions remain the RL lane's call and get logged as decisions."* D34's own status line is
amended below to point here.

**Three additions, in the order they landed, each keeping D19/D29's rule** (built only from the
agent's own scan history or receiver-observable quantities — never truth):

1. **`current_band`** — a one-hot of the band the most recent dwell was on. Distinguishes "I am
   here right now" from "I left here one slot ago", which `staleness` alone cannot: the currently
   tuned band and a band just vacated both read staleness ≈ 0.
2. **`camp_time`** — consecutive slots spent on the current streak, normalised by `N_SLOTS`, reset
   the instant the action changes. Not redundant with `visit_density` (airtime share over the
   *whole* episode): a band camped early and abandoned still reads high density long after the
   agent moved on, where `camp_time` collapses back to 0 the moment it leaves.
   **REMOVED 2026-09-09 by D55.** The reasoning above is sound but the component was measurably
   inert: under a non-camping policy it takes exactly two values, because the streak cannot exceed
   one dwell unless the agent is already camping.
3. **`measured_dbm`** — the most recent dwell's mean `S + noise`, i.e. what the receiver's
   detector actually read (D19: observable, unlike the truth-side `S` alone), clamped to a fixed
   `[-120, -20]` dBm window and linearly rescaled to `[0, 1]` to fit the `Box`. Global, not
   per-band, matching `camp_time`/`current_band`'s scope: it reports only the band just left.

**The vector's size moved three times in the same session**: D34's **109** (36×3+1) → **145**
(36×4+1, `current_band`) → **146** (+`camp_time`) → **147** (36×4+3, +`measured_dbm`), and then
**back to 146** when D55 dropped `camp_time` and rescaled two blocks. Every
existing trained checkpoint breaks at each step — SB3 sizes a policy network's input layer to
`observation_space.shape` at construction and cannot accept a differently-shaped vector afterward
— so a shape change is a retrain, not a reload, for every registered RL rung (D48).

**Evidence.** `rfenv/env.py::ScanEnv._observation`, `rfenv/baselines/guard.py` (`CURRENT_BAND`,
`MEASURED_DBM` slice constants — `CAMP_TIME` until D55 — kept in sync with the vector by
`tests/test_baselines.py::test_the_observation_slices_match_the_environment`).

---

## D50 — reward-candidate churn this session: the default moved, a fourth candidate did not stick

**Status:** `SETTLED` as a record of what changed (2026-09-09). **Not** a D47 rule application —
recorded as that explicitly, below. **SUPERSEDED on the default 2026-09-09 — see D53:**
`first_intercept` is no longer a registered key at all; candidate 3 is now `reward_balance` and
`DEFAULT_REWARD` names that. Everything below about the *fourth* candidate still stands.

**`DEFAULT_REWARD` moved from `hit_z` to `first_intercept`,** changed directly in `env.py` and
confirmed this session. No `compare.paired_wins` measurement (D47's own selection rule) was run
against this specific choice — this entry records that the default **is** now `first_intercept`,
not that D47's dominance-count procedure is what put it there. Anyone treating this as "the reward
question is closed" should re-read D47: the rule exists, but no evidence line here claims it was
applied. Every trainer's `--reward` CLI flag still defaults to `env.DEFAULT_REWARD`, so reproducing
any rung's base variant (all originally trained on `hit_z`, D48) now needs `--reward hit_z` passed
explicitly rather than left to the default.

**A fourth candidate, `reward_weighted_camp`, was registered and retired within the same working
session.** Discovery credit plus declared-hit credit, the declared-hit term taxed by a per-slot
fee proportional to the current camping streak — an attempt to price greed directly rather than
leave camping to be merely un-rewarded. It briefly went in as `REWARDS["weighted_camp"]`, a
deliberate, explicit exception to D29's "exactly three, and the set stays at three." It came back
out; **D29's three-candidate cap holds, unamended.** Kept as a commented-out draft in `env.py`
alongside a further, never-finished `reward_hybrid` sketch — neither is active, both are dead code
by choice, not by accident, in case either is picked back up.

**Evidence.** `rfenv/env.py` (`REWARDS`, `DEFAULT_REWARD`, the commented-out drafts and their
docstrings), `tests/test_env.py::test_all_reward_candidates_run_and_differ`.

---

## D51 — `newly` credits an emitter once per band, not once per episode

**Status:** `SETTLED` (2026-09-09) — changed in `rfenv/env.py::ScanEnv.step`, flagged as looking
like a bug, confirmed intentional by the human. Reward-side only; **D28's intercept definition and
every §4 metric are untouched.** Recorded because it is in a gated category (the reward) and
because the candidate's name now outlives its semantics — see the last paragraph.

**What changed.** The `newly` set — the only input `reward_first_intercept` (D29's candidate 3)
has beyond the dwell — used to fire once per emitter per episode: an emitter's true first
intercept and nothing after. It now fires **once per distinct `(emitter, band)` pair**: on the
first-ever intercept, and again the first time that emitter turns up in a band it has not been
credited in before. `track["bands"]` is what remembers which.

**What it does not change, which is the part that matters.** `track["first"]` — `first_e`, the
censored-intercept-time numerator D28 defines — is written **only** in the `track is None` branch,
never in the widened one. So it is still set exactly once per emitter, at the true first sighting.
Censored intercept time, emitter coverage, interception ratio and everything else in
`EVALUATION.md` §4 are computed from that and are bit-identical either way. Only the reward-facing
set was widened; the metric-facing one was not.

**Why widen it.** D3's bands overlap — an emitter generically falls inside about two band windows
— so under the old rule a scheduler that swept an emitter's whole footprint was paid exactly the
same as one that clipped its edge once and moved on. The widened set pays for covering the
footprint. Whether a policy can *learn* to act on it is D30's open question, unchanged by this.

**Measured effect: about 2.5×, not the ~2× D3's geometry alone would suggest.** Round-robin, seed
0, stare replays:

| scenario | distinct emitters intercepted | `newly` credits | credits per emitter |
|---|---|---|---|
| `config_2` | 16 | 40 | 2.50× |
| `config_921` | 61 | 154 | 2.52× |

Credits equal the `(emitter, band)` pair count exactly in both — confirming one credit per pair,
with no double-counting inside a band. It runs above 2× because a full 36-band sweep catches an
emitter in more windows than the nominal two it generically spans.

**The naming tension, recorded rather than fixed.** Candidate 3 is still called
`first_intercept`, and D29 and D28 both describe it in first-intercept language, but it no longer
pays for first intercepts — it pays for footprint coverage. The name is now wrong in a way that
will mislead anyone reading D29 without reading this. Renaming it is a checkpoint-invalidating
change to a registered `REWARDS` key (every rung trained on it names it), so it is **not** done
here; the mismatch is recorded instead, and D50 already notes this candidate is now the default.

**Evidence.** Measured 2026-09-09, this session, on `config_2` and `config_921` stare replays under
round-robin at seed 0, counting `info["newly_intercepted"]` per step against `len(env.tracks)` and
`sum(len(t["bands"]) for t in env.tracks.values())`. Code: `rfenv/env.py::ScanEnv.step` (the
`if dwell.band not in track["bands"]` branch) and `reward_first_intercept`'s own docstring.
Reasoned, from the same code, that `first`/`first_e` is unaffected: it is assigned in one branch only.

---

## D52 — the reward's staleness array was the inverse of the observation's, and paid for camping

**Status:** `SETTLED` (2026-09-09) — a bug fix, not a design change. Recorded anyway because it
sits in a gated category (the reward, D29) and because it silently invalidates every RL result
trained against `reward_balance` before this date.

**What was wrong.** `ScanEnv.step` maintains three per-band arrays that exist only to be handed to
the reward — `_hit_rate_array`, `_visit_density_array`, `_staleness_array`. The staleness one
stored

```python
self._staleness_array[action] = self._last_slot[action] / N_SLOTS
```

which is *when* the band was last seen. The quantity `_observation()` reports, and the one
`reward_balance`'s docstring promises it prices exploration against, is *how long ago*:
`(self.t - self._last_slot[action]) / N_SLOTS`. Over an episode those two run in opposite
directions. A band revisited constantly has a large and growing `_last_slot`; a band abandoned
early keeps a small one forever. So `(1.5 - visit_density) * staleness` — the term whose entire
job is to pull the agent toward neglected bands — paid **most** for returning to the band it had
just left.

The never-visited case was worse than merely inverted. `_last_slot` initialises to the sentinel
`-1`, and the array element for a band is written on the step that band is *chosen*, before the
dwell lands. A band chosen once at slot 0 and never again therefore kept `-1 / 600` — a small
**negative** staleness — for the rest of the episode.

**Measured, before the fix.** Sampled scenario, seed 0, at `t = 503`, after one dwell on band 0 at
slot 0 and 250 dwells on band 18:

| band | last seen | exploration term `(1.5 − visit_density) × staleness` |
|---|---|---|
| 18 — just left | slot ~502 | **+0.419** |
| 0 — untouched for 500 slots | slot 1 | **−0.0025** |

After the fix, the same two read **+0.00084** and **+1.5**.

**Why nothing caught it.** `_observation()` never reads these arrays; it recomputes all three from
the raw counters (`_slots_looked`, `_hits`, `_last_slot`) and was always correct. The bug lived
only in the reward-facing copy, so every observation test passed and the agent's *inputs* were
never wrong — only the price it was paid for acting on them. Two dead locals in the same function
(`staleness_before` and `staleness`) held the correct formula and went unused, which is what made
the wrong line read as if it were right.

**Consequence for existing results.** Every RecurrentPPO checkpoint trained on `reward_balance`
before 2026-09-09 was optimising a reward whose exploration term rewarded camping. Those runs are
not evidence about what the environment teaches; they are evidence about what that reward taught.
`EVALUATION.md` §5's rung 9 rows are marked accordingly.

**Evidence.** `rfenv/env.py::ScanEnv.step` (the `_staleness_array` assignment) against
`ScanEnv._observation` (the `staleness` block). Measured this session by stepping a
`ScanEnv(pool=EmitterPool.from_train(), reward="reward_balance")` at seed 0 and reading both
quantities directly. Pinned by
`tests/test_env.py::test_the_reward_reads_the_same_staleness_the_observation_reports`, which
asserts the array equals the observation's value at the index the reward reads, that a
never-visited band reads 1.0 rather than −1/600, and that a long-abandoned band reads staler than
one just left.

---

## D53 — `reward_balance`'s camping cost is charged against airtime share, not a repeat streak

**Status:** `SETTLED` (2026-09-09) — a change to the shape of a registered reward candidate,
proposed with the measurements below and **confirmed by the human** before it went in, per
CLAUDE.md's architecture gate. Candidate 3 only; D29's three-candidate cap holds, unamended.

**Naming, first, because D50 is now stale on it.** Candidate 3 is `reward_balance`, and
`DEFAULT_REWARD` names it. The key `first_intercept` that D50 recorded as the default no longer
exists in `REWARDS`; the rename and rewrite happened outside any recorded session, so this entry
records the state, not the moment it changed. D51's discussion of `newly` describes a candidate
that no longer reads `newly` at all — `reward_balance` ignores it. `newly` is still computed and
still published in `info["newly_intercepted"]`, so nothing about D51's measurement is withdrawn;
it simply has no consumer in the current candidate set.

**What changed.** The fourth term of `reward_balance`:

```python
reward -= 1.0 * camp_slots                                    # before
reward -= 3.0 * visit_density_array[action] * dwell.n_slots   # after
```

**Why.** `camp_slots` counts *consecutive* slots on one band and resets the instant the action
changes. A policy therefore defeats it for free by alternating between two bands: a 2-band
ping-pong and a full 36-band sweep both pay exactly `N_SLOTS` in total. That is not a small
loophole — it is the difference between the behaviour rung 4 exists to demonstrate and the floor
the whole ladder is built on. `visit_density` has no such hole, because it is airtime share over
the episode: sustained camping drives it to 1.0, alternating holds it near 0.5, a sweep near 1/36.

**Measured.** Four fixed policies, three sampled scenarios each (seeds 0, 1, 2), the staleness bug
of D52 already fixed in both columns so the two terms are compared on equal footing:

| policy | interception ratio | censored intercept time | emitter coverage | `−1.0 × camp_slots` | `−3.0 × visit_density × n_slots` |
|---|---|---|---|---|---|
| recency (rung 5) | 0.0995 | 6.49 s | 0.782 | −206.6 | **+329.5** |
| round-robin | 0.1179 | 2.47 s | **0.921** | −228.5 | **+324.3** |
| alternate, 2 bands | 0.1985 | 16.55 s | 0.261 | **−218.9** | −505.4 |
| camp one band | 0.2350 | 16.84 s | 0.245 | −89,867 | −1,361 |

> **CORRECTED 2026-09-10 — the "round-robin" row above is not rung 2.** It came from a
> hand-written `step % N_BANDS` sweep rather than `baselines.make("round_robin")`, the same
> substitution that inverted D56. Re-measured against the ladder's own rungs, 8 seeds, under the
> final `−3.0 × visit_density × n_slots` term: recency **+276.9**, **rung 2 +217.9**, rung 4
> (`camper`) **−414.1**, rung 6a **+278.7**, alternate **−560.9**, camp-one-band **−1447.7**.
>
> **The conclusion is unchanged and is in fact stronger.** Sweeping policies beat degenerate ones
> by **+778.8** rather than the +832.1 first reported, and rung 5 now sits **+59.0 ± 19.4 above
> rung 2 on 8/8 seeds** where the flawed comparison put it at +2.0. The ordering this entry exists
> to establish — that airtime share fixes a hole a repeat-streak counter left open — holds either
> way, because both sweeps order the same way against a ping-pong.

**The old term ranked the 2-band ping-pong above round-robin** — coverage 0.261 against 0.921,
censored intercept time 16.55 s against 2.47 s, and it scored 4% better. There was no gradient
toward sweeping; what little existed pointed the wrong way. That is the direct explanation for why
the rung 9 policy's action distribution stayed near-uniform (D54): "don't repeat the same band
twice in a row" is the only coherent thing that reward taught, and a near-random policy already
satisfies it.

Two secondary properties of the new term, both deliberate:

- **Scaled by `dwell.n_slots`**, so the cost is per slot of airtime spent and D31's per-slot
  invariant survives — a wide band costs twice as much because it consumes twice as much episode.
- **The reward's range collapses from ~90,000 to ~1,700.** An unbounded streak counter makes a
  fully-camped episode worth −89,867 against a good episode's +330, which is a 270× spread for a
  value head to fit. This is a training-stability argument, not a correctness one, and is recorded
  as such.

**What this does not claim.** These are four hand-written policies on three sampled scenarios, not
D47's paired-dominance procedure over the full protocol, and not a claim that `reward_balance` is
the right candidate. D47's selection rule remains un-applied, exactly as D50 left it. What is
claimed is narrower and sufficient: the previous term ordered two known policies backwards, and
this one does not.

**Evidence.** `rfenv/env.py::reward_balance`. Measured this session by scoring each policy through
`ScanEnv(pool=EmitterPool.from_train(), reward="reward_balance")` at seeds 0/1/2 and reading
`episode_metrics()["total_reward"]`. Pinned by
`tests/test_env.py::test_reward_balance_ranks_a_sweep_above_a_two_band_pingpong`, asserted as an
ordering rather than against literals because the numbers move with the scenario draw.

---

## D54 — RL inference samples the policy; it does not take the argmax

**Status:** `SETTLED` (2026-09-09) — a change to the evaluation protocol, proposed with the
measurement below and **confirmed by the human**. This is the entry that withdraws the "the agent
learned to camp" reading of `EVALUATION.md` §5.

**What was wrong.** `RLScheduler` and `RecurrentRLScheduler` both hardcoded `deterministic=True`,
so every RL row ever produced in this repository reported the **mode** of the policy rather than
the policy. For a DQN that is correct — the greedy argmax *is* a DQN's policy. For an on-policy
algorithm it is not: PPO optimises expected return under the sampled distribution and never
evaluates its own mode, so nothing in training constrains where the argmax lands.

**Measured, on `runs/checkpoints/lstm_gamma997.zip`, one seed-0 episode**, reading the action
distribution directly off `policy.get_distribution`:

```
mean entropy       2.369    (uniform over 36 bands = ln 36 = 3.584; collapsed = 0)
mean max-probability 0.206  the modal band holds about a fifth of the mass
argmax             band 5 on 580 of 586 steps
```

**The policy had not collapsed — it was broad. Its mode was sticky.** Taking the argmax of a broad
distribution whose peak barely moves turns a spread-out policy into a one-band camper. The same
checkpoint, same seeds, sampled instead:

| inference | distinct bands visited | emitter coverage |
|---|---|---|
| `deterministic=True`, seed 0 / 1 | 2 / 2 | 0.247 / 0.041 |
| `deterministic=False`, seed 0 / 1 | **31 / 31** | **0.603 / 0.714** |

This confirms at the level of the distribution what `EVALUATION.md` §5 already flagged as an
unmeasured caveat from a single episode. **The camping in every rung 9 row is an inference
artefact, not a learned policy.**

**The decision.** `deterministic` is now a constructor parameter on both adapters, defaulting to
`False`. Three consequences, each deliberate:

1. **Rung 7 (DQN) passes `deterministic=True` explicitly** in its ladder factory. SB3's
   `deterministic=False` on a DQN means ε-greedy *exploration* noise at
   `exploration_final_eps` (0.05 by default) — a training artefact, not a learned distribution.
   Sampling it would inject 5% random actions into an evaluation.
2. **Rungs 8 and 9 sample.** That is what their training return measured, so it is what an
   evaluation of them should measure.
3. **torch's global generator is seeded per rung, from the rung's own stream.** SB3's `.predict()`
   draws its sample from torch's global RNG and accepts no generator argument, so without this two
   runs of `compare.py` at the same seed would give a sampled rung different actions and
   `EVALUATION.md` §7's "identical scenarios and seeds" would quietly stop holding. `make()`
   already derives an RNG from the episode seed and the rung key; `_seed_torch` folds that into
   torch. Verified: same seed → byte-identical action sequence.

**What this does not settle.** Whether rung 9 *beats* anything under sampled inference is
unmeasured at protocol scale — the numbers above are two episodes on one checkpoint trained
against the pre-D52 reward. `EVALUATION.md` §5's rung 9 table is superseded on both counts and
needs re-running once a checkpoint trained on the corrected reward exists.

**Evidence.** `rfenv/rl/common.py` (`RLScheduler`, `RecurrentRLScheduler`),
`rfenv/baselines/ladder.py` (`_seed_torch` and the three rung factories). Measured this session on
`lstm_gamma997.zip` at seeds 0 and 1; entropy and max-probability read from
`model.policy.get_distribution(...).distribution.probs` over a full seed-0 episode.

---

## D55 — the observation is rescaled so 1.0 means something, and `camp_time` is dropped

**Status:** `SETTLED` (2026-09-09) — a change to the observation vector, which is a gated
category (D34, D42's "not frozen, deliberately" list). **Proposed with the measurements below and
confirmed by the human**, who also accepted the cost: it invalidates every checkpoint, including a
1M-step RecurrentPPO run that was in flight when the change went in.

**The vector goes 147 → 146.** Nothing was reordered; two blocks changed units and one scalar left.

| block | before (D34/D49) | after (D55) |
|---|---|---|
| `hit_rate` | hits / slots looked | unchanged |
| `visit_density` | slots looked / elapsed | **× `N_BANDS`** — airtime share over *fair* share |
| `staleness` | (t − last visit) / `N_SLOTS` | **/ `SWEEP_SLOTS`** — neglect in reference sweeps |
| `current_band` | one-hot | unchanged |
| `clock` | t / `N_SLOTS` | unchanged |
| `camp_time` | streak / `N_SLOTS` | **removed** |
| `measured_dbm` | clamped, rescaled | unchanged |

**The observation space is no longer the unit box**, and that is the point rather than a side
effect. `visit_density` declares a ceiling of `N_BANDS` (36.0, a fully camped episode) and
`staleness` one of `N_SLOTS / SWEEP_SLOTS` (13.95, a band untouched all episode). SB3 does not
rescale inputs, so the numbers in the box are the numbers the network sees; an honest box beats a
tidy one.

### Why: two of the three per-band blocks lived in the bottom tenth of their range

Measured this session over 1,506 round-robin steps and 1,409 rung-5 steps, 3 seeds each, sampled
scenarios:

| block | mean | p99 | max |
|---|---|---|---|
| `visit_density`, before | **0.0278** | 0.065 | 1.000 |
| `visit_density`, after | **1.0000** | 2.323 | 36.000 |
| `staleness`, before | **0.0697** | 1.000 | 1.000 |
| `staleness`, after | **0.9720** | 13.953 | 13.953 |

`visit_density`'s old mean is not an accident of the policy — it sums to 1 across bands by
construction (airtime is the only currency, D31), so its mean is pinned at exactly 1/36 for every
scheduler that ever runs. `staleness`'s old distribution was bimodal rather than small: everything
visited sat near zero and everything never-visited sat on the 1.0 ceiling, with the p99 landing
exactly on the ceiling.

**The decisive evidence is that rung 5 already had to correct one of them by hand.**
`baselines/recency.py` multiplied staleness straight back out by `N_SLOTS / SWEEP_SLOTS` before
using it, and its docstring records what happens without that step: *"the rung silently collapses
into rung 4: measured on `config_2` stare, coverage 0.526 against round-robin's 0.895."* The bar
the RL rungs have to clear was unusable on the raw feature. Since D55 the division happens in
`_observation()` and every policy gets it, rather than it remaining one heuristic's private
knowledge. `recency.py`'s `__call__` no longer divides; **rung 5's ranking is unchanged, verified
on 1,408 of 1,408 steps against an explicit recomputation in the old units.**

**`visit_density` is the one `reward_balance` prices camping off** (D53), so the term meant to
prevent camping was reading the block with the least resolution in the vector.

### Why `camp_time` went

Measured under a non-camping policy it takes **exactly two values**, 1/600 and 2/600 (std 0.0007) —
it cannot move unless the agent is already camping, because the streak resets on every action
change. It is a gauge that only registers once the wrong thing is happening, while `visit_density`
says the same thing continuously and earlier. Its only consumer, `-1.0 * camp_slots`, was replaced
in D53. `ScanEnv._camp_slots` is still maintained: every reward candidate takes it in its
signature (D31's one call shape) and `episode_metrics()` reports it.

### What was deliberately *not* done

- **`current_band` stays**, though it is exactly redundant: `argmin(staleness) == argmax(current_band)`
  on 1,506 of 1,506 steps, since the band just dwelt on always has minimum staleness. Recovering it
  costs the network an argmax over 36 dims; 36 input weights per neuron is the cheaper side of that
  trade. Recorded so nobody re-derives the redundancy and assumes it was missed.
- **Time since last *hit*, per band, was not added.** `ScanEnv._last_hit_slot` already tracks it and
  `_observation()` still excludes it. `staleness` says when the agent last *looked*; nothing says
  when a band was last *active*, and in a restless environment those want opposite actions. It is
  the obvious next component and it is held back deliberately: bundling it with a rescale would make
  the retrain unattributable. **Open, and it is the next observation question.**

### The reward is numerically unchanged, on purpose

`reward_balance` reads two of the rescaled arrays, so leaving it alone would have silently
re-tuned it and invalidated D53's table. It converts both back to the old units at the top of the
function instead of carrying new coefficients. Verified two ways: per-step reward matches an
explicit old-units recomputation to **1.6e-7** over 1,408 steps, and D53's two deterministic rows
reproduce exactly — round-robin **+324.3**, camp-one-band **−1360.9**. The two remaining rows move
within tie-breaking noise (rung 5 breaks ties randomly) and are not evidence of a change.

D52's requirement still holds and its test still passes: the reward and the observation read the
same quantities. They now read them in the same units too, and the conversion is one visible line
rather than a scale mismatch nobody can see.

### What it costs

**Every checkpoint in `runs/checkpoints/` is dead** — all five were 146- or 147-wide against a
different layout, and the test suite's skip count goes 42 → 66 accordingly. This includes the
in-flight `lstm_balance_1M` run, which was warned about before the change and accepted. Rungs 7 and
8 were already dead (D49) and remain so.

**Evidence.** `rfenv/env.py` (`SWEEP_SLOTS`, `_SWEEPS_PER_EPISODE`, `_observation`, the
`_visit_density_array`/`_staleness_array` writes in `step`, `reward_balance`'s conversion),
`rfenv/baselines/guard.py`, `rfenv/baselines/recency.py`. Measured this session on
`ScanEnv(pool=EmitterPool.from_train())` at seeds 0/1/2 under round-robin and rung 5. Pinned by
`tests/test_env.py::test_spaces_are_the_specified_ones` (which now asserts the two non-unit
ceilings), `::test_every_episode_starts_cold`,
`::test_the_reward_reads_the_same_staleness_the_observation_reports`, and
`tests/test_baselines.py::test_the_observation_slices_match_the_environment`. Suite: 265 passed,
66 skipped, 1 failed — the pre-existing held-out-split guard.

---

## D56 — `reward_balance` cannot separate rung 5 from round-robin, and that caps what training can reach

> ## WITHDRAWN 2026-09-10 — the measurement was wrong, and the conclusion inverts.
>
> **What it scored was not rung 2.** The "round_robin" column below came from a hand-written
> `step % N_BANDS` sweep, not from `baselines.make("round_robin")`. Rung 2 is
> `EQUAL_AIRTIME_CYCLE` (D43) — equal *airtime* per band, two slots per 72-slot cycle — while
> `step % N_BANDS` gives every band one *dwell* per cycle and so hands the seven wide bands twice
> the airtime. They are different policies, and the naive one scores **+57.0 ± 18.5** higher on
> `reward_balance` with coverage 0.932 against rung 2's 0.793.
>
> Re-measured against the ladder, 8 seeds, sampled scenarios:
>
> | quantity | separation | seeds with rung 5 ahead |
> |---|---|---|
> | `recency − rung 2` (correct) | **+59.0 ± 19.4** | **8 / 8** |
> | `recency − step % 36` (what D56 measured) | +2.0 ± 13.5 | 5 / 8 |
>
> So the separation is about **three times the seed noise, not three percent of it**, and every
> claim below built on that ratio falls with it. `reward_balance` distinguishes the bar from the
> floor perfectly well; it is the only one of six candidates to pass D62's screen.
>
> **What survives:** nothing of the conclusion. The instrument was right and the brief that called
> it "the right instrument, one step short" was right — D62 is what it became. What was wrong was
> comparing against a stand-in for a rung instead of the rung.
>
> **Where it propagated:** `CLAUDE.md`'s build status and `EVALUATION.md` §5 both carried the
> "round-robin is roughly the ceiling this reward can teach" line; both are corrected. D53 and D57
> used the same hand-written sweep and are corrected in place — their conclusions survive, only
> the `round_robin` rows move.
>
> **The lesson, which is why this is withdrawn in place rather than deleted:** a rung has a
> registered implementation for a reason, and a measurement that substitutes an obvious-looking
> reimplementation is not measuring the ladder.
> `tests/test_reward_gate.py::test_every_screened_rung_is_a_real_registered_rung` now enforces it.

**Status:** ~~`OPEN` (2026-09-09)~~ **`WITHDRAWN` (2026-09-10)** — a measured limitation, recorded rather than fixed. **Not** a
change to anything; `reward_balance` is untouched by this entry. Raised because it bounds what the
run logged in `scratch/TRAINING_JOURNEY.md` §9 can possibly achieve, and because it is the next
reward question after D52 and D53.

**Measured, 8 seeds, sampled scenarios, per seed rather than averaged:**

| policy | reward mean | sd | coverage |
|---|---|---|---|
| recency (rung 5) | **+277.2** | 81.6 | 0.768 |
| round-robin (rung 2) | **+274.9** | 77.7 | 0.932 |

Per-seed difference: **+2.3 ± 11.7**, with rung 5 ahead on **4 of 8 seeds**. Against a
scenario-to-scenario spread of about 80, the reward's signal between the best deployable heuristic
and the floor is roughly 3% of its own noise.

**What that implies.** `reward_balance` distinguishes catastrophe from competence with a very large
margin — camping scores −1361 against round-robin's +324, which is why D53's fix mattered and why an
agent trained on it should stop camping. It does **not** meaningfully distinguish competence from
excellence. An agent optimising it has almost no gradient telling it that rung 5's behaviour beats
round-robin's, so **round-robin is approximately the ceiling this reward can teach**, and D46's
stated target — hold `recency`'s 3.20 s while multiplying its interception ratio — is not encoded in
it at all.

Note also that round-robin scores *better* coverage (0.932 against 0.768) while rung 5 wins on
interception ratio; the reward is weighted toward the spread-out behaviour, so the two nearly cancel.

**Not acted on**, for two reasons. Re-weighting the reward now would confound the next training run
with D52/D53/D55, and D47's paired-dominance selection rule — the project's own procedure for
choosing between reward candidates — has still never been run. Any change here should come out of
that procedure rather than out of one more hand-tuned coefficient.

**Evidence.** Measured this session, 8 seeds, `ScanEnv(pool=EmitterPool.from_train(),
reward="reward_balance")`, comparing `baselines.make("recency")` against a `t % N_BANDS`
round-robin and reading `episode_metrics()["total_reward"]`.

---

## D57 — the reward set becomes an axis: `greedy`, `explore` and the blend between them

**Status:** `SETTLED` (2026-09-09) — three new candidates registered, and **D29's cap of exactly
three is formally lifted**. Both are gated changes (the reward), requested and confirmed by the
human. D47's selection rule is still un-applied and nothing here selects anything.

### What was added

| key | shape | expected to |
|---|---|---|
| `greedy` | `+1.0·Y.sum() + 2.0·hit_rate[a]·n_slots` | **camp** — no explore term, no camping cost |
| `explore` | `+1.0·staleness[a]·n_slots − 0.10·visit_density[a]·n_slots + 2.0·len(newly)` | **sweep** — cannot express the bar |
| `weighted` | `α·greedy·4.08 + (1−α)·explore`, α = **0.3** | sit between them |

`make_reward_weighted(alpha)` is public, so the whole curve is buildable without touching the
registry — which is what makes D47's paired-dominance rule runnable over this family rather than
only arguable about.

### Why lift D29's cap

D29's reason has not gone away: every candidate scored on the same 47 scenarios is another draw,
and best-of-many is partly selection noise. What changed is that these three are **not three more
independent guesses competing for one prize**. `greedy` and `explore` are the two corners of D14's
tension and are *expected to fail* — one camps, the other cannot rank rung 5 above the floor —
and `weighted` is the single knob between them. The multiple-comparisons argument applies to
choosing a point on a designed axis, not to six independent tries. D50's precedent stands: a fourth
candidate was once registered and retired, and that was treated as an exception; this is a change
to the rule, recorded as one.

### Measured: what each registered candidate actually teaches

Four reference policies — two sweeping, two degenerate — over 3 seeds on sampled scenarios:

| policy | `hit_z` | `hit_y` | `reward_balance` | `greedy` | `explore` | `weighted` |
|---|---|---|---|---|---|---|
| recency (rung 5) | 361.0 | 336.0 | 323.0 | 959.6 | 1286.6 | **2075.2** |
| round-robin | 347.3 | 307.0 | 324.3 | 845.8 | 1384.8 | 2004.7 |
| alternate, 2 bands | 504.7 | 478.3 | −509.0 | 1443.6 | −909.1 | 1130.6 |
| camp one band | 582.3 | 573.0 | −1360.9 | 1694.8 | −2078.3 | 619.7 |

> **CORRECTED 2026-09-10 — the "round-robin" row is a hand-written `step % 36`, not rung 2**, and
> the four policies above include no registered rung at all. Re-measured against the ladder, 8
> seeds — and with rung 4 and rung 6a in place of the hand-written camper, which is what D62's
> screen now uses:
>
> | rung | `reward_balance` | `greedy` | `explore` | `weighted` (α=0.3) |
> |---|---|---|---|---|
> | 5 `recency` | **+276.9** | +776.6 | +1238.1 | **+1817.2** |
> | 2 `round_robin` | +217.9 | +459.0 | +1277.0 | +1455.7 |
> | 4 `camper` | **−414.1** | **+1198.3** | −205.2 | +1323.0 |
> | 6a `apfeld_active_rfs` | +278.7 | +984.2 | +1034.0 | +1928.4 |
>
> **Every conclusion in this entry survives.** `greedy` still ranks the camper top (now against
> the real rung 4, which is the stronger statement); `explore` still puts round-robin above rung 5
> (−38.9, 0/8 seeds); `weighted` still orders the ladder. What changed is that `reward_balance` now
> separates rung 5 from rung 2 by +59.0 rather than −1.3 — see D56, withdrawn.

Two checks, neither of them a performance metric — both ask only whether the reward orders policies
the way the ladder already does:

| candidate | sweeps beat degenerate? | rung 5 − round-robin |
|---|---|---|
| `hit_z` | **no** (−235.0) | +13.7 |
| `hit_y` | **no** (−266.0) | +29.0 |
| `reward_balance` | yes (+832.1) | **−1.3** |
| `greedy` | **no** (−849.0) | +113.8 |
| `explore` | yes (+2195.7) | **−98.2** |
| **`weighted`** | **yes (+874.0)** | **+70.5** |

**`weighted` is the only registered candidate that passes both.**

> **AMENDED 2026-09-10 — under D62's screen, against the real rungs, this reverses.** With rung 4
> in place of the hand-written camper, `weighted` puts the camper at +1323.0 against rung 2's
> +1455.7 — only 0.4σ below, where the screen requires 1.0σ — and it **fails**. `reward_balance`
> is the only candidate of the six that passes (D62). `weighted` remains registered and remains the
> knob it was built to be; it is not currently a candidate D47 may consider.
>
> The paragraph this amends originally read: *"`hit_z` and `hit_y` rank camping above sweeping,
> which is D14's tension and exactly why the ladder carries rung 4. `reward_balance` orders the
> ladder correctly but cannot separate rung 5 from round-robin at all (D56). `greedy` and `explore`
> each fail the check their corner is defined by failing."* The `hit_z`/`hit_y`/`greedy`/`explore`
> sentences still hold. **The `reward_balance` sentence is withdrawn with D56**: it separates rung 5
> from round-robin by +59.0 ± 19.4 on 8/8 seeds once measured against the real rung, not "not at
> all" — see D56's own entry for the correction.

### How α = 0.3 was chosen

Swept 0.0 → 1.0 in 0.1 steps over the same four policies and seeds:

| α | recency | round-robin | alternate | camp | sweeps > degenerate | rung 5 top |
|---|---|---|---|---|---|---|
| 0.0 | 1286.6 | 1384.8 | −909.1 | −2078.3 | yes | no |
| 0.1 | 1549.5 | 1591.4 | −229.2 | −1179.0 | yes | no |
| **0.2** | 1812.3 | 1798.0 | 450.7 | −279.7 | **yes** | **yes** |
| **0.3** | 2075.2 | 2004.7 | 1130.6 | 619.7 | **yes** | **yes** |
| **0.4** | 2338.0 | 2211.3 | 1810.5 | 1519.0 | **yes** | **yes** |
| 0.5 | 2600.9 | 2417.9 | 2490.4 | 2418.3 | **no** | yes |
| 0.6–1.0 | — | — | — | — | **no** | no |

> **CORRECTED 2026-09-10 — re-derived against rung 2 and rung 5** rather than the hand-written
> sweep. The window *widens* and α = 0.3 stays comfortably inside it:
>
> | α | 0.0 | 0.1 | 0.2 | **0.3** | 0.4 | 0.5 | 0.6+ |
> |---|---|---|---|---|---|---|---|
> | sweeps > degenerate | yes | yes | yes | **yes** | yes | no | no |
> | rung 5 > rung 2 | 0/8 | **8/8** | 8/8 | **8/8** | 8/8 | 8/8 | 8/8 |
>
> Valid range **[0.1, 0.4]**, not [0.2, 0.4] — the lower bound moves because the flawed sweep
> scored too high and made rung 5 look beatable by the floor at α = 0.1. The upper bound is
> unchanged: from 0.5 the greedy half lets a 2-band ping-pong outscore rung 2, which is D53's
> failure mode arriving through a different term. **`WEIGHTED_ALPHA = 0.3` is unchanged and is
> still the middle of the window.**

**The window closes from both ends.** Below 0.2 the explore half dominates and round-robin
outscores rung 5, so the reward cannot express the bar it is supposed to teach toward. From 0.5 up,
a 2-band ping-pong outscores round-robin — **D53's exact failure mode, reintroduced through the
greedy half rather than through a streak counter**. Only 0.2–0.4 satisfies both; 0.3 is its middle.

**This is a sanity constraint, not tuning against a score.** Nothing was chosen to make a policy
perform better on D27's metrics. What was checked is that two known-good policies outrank two
known-degenerate ones — the same check D53 applied. Which candidate to actually *select* remains
D47's paired-dominance rule over the full protocol, and it has still never been run.

### Two things fixed during the work, recorded because they were nearly shipped

**`explore`'s density coefficient started at 2.0 and produced a 43,000-wide range.**
`visit_density` reads in fair shares since D55, reaching 36.0 on a camped band, so 2.0 charged 72
per slot and gave a camped episode roughly −43,000 against a good episode's few hundred. That is
the same unfittable-scale failure D53 removed from `reward_balance`, reintroduced from the other
direction. Measured and corrected to 0.10 before registration; the range is now ~3,500.

**`_GREEDY_GAIN` was initially a guess of 2.5 in the wrong direction**, which made α = 0.5 roughly
a 10:1 explore-dominated blend rather than half-and-half. It is now 4.08, the measured ratio of the
two halves' spreads across the four reference policies (explore 3463.1, greedy 849.0). It is a
units correction, not a coefficient: multiplying a reward by a positive constant cannot change its
optimal policy, so nothing about `greedy` or `explore` alone depends on its value.

### What this does not do

- **It does not change `DEFAULT_REWARD`**, which is still `reward_balance`. `weighted` scoring
  better on both sanity checks is an argument for running D47, not a substitute for having run it.
- **It does not invalidate any checkpoint.** The observation is untouched; only the registry grew.
- **It does not rank the candidates.** The tables above are ordering checks against known policies,
  not paired-dominance measurements over the evaluation protocol.

**Evidence.** `rfenv/env.py` (`reward_greedy`, `reward_explore`, `make_reward_weighted`,
`WEIGHTED_ALPHA`, `_GREEDY_GAIN`, `REWARDS`). Measured this session on
`ScanEnv(pool=EmitterPool.from_train())` at seeds 0/1/2, reading
`episode_metrics()["total_reward"]` for each of the four reference policies under each candidate.
Pinned by `tests/test_env.py::test_the_greedy_explore_axis_has_the_shape_it_claims` (the corners
prefer their corner; α = 0 reproduces `explore` exactly and α = 1 reproduces `greedy` up to a
single positive constant) and
`::test_the_registered_blend_outranks_the_degenerate_policies` (the half that would silently
un-fix D53). Suite: 267 passed, 84 skipped, 1 failed — the pre-existing held-out-split guard.

---

## D58 — the Pareto figure plots medians with interquartile whiskers, not means

**Status:** `SETTLED` (2026-09-09) — a change to how a published figure reports the §4 metrics,
requested and confirmed by the human after the mean-based version was found to be actively
misleading. **The metrics themselves are untouched**, and so is the printed table: `_report_md`
still reports `mean` over the IQR and `EVALUATION.md` §5's ratified rows are still means. Only the
figure's collapse changed.

### What the mean-based figure said, and why it was wrong

`render.pareto` draws one marker per scheduler. A marker cannot show a distribution, so the
statistic it collapses to has to be one a minority of episodes cannot move. For every rung on the
ladder except one that is a detail; for **rung 4 it decides what the figure says**.

Measured over 47 stare replays, seed 0 (`runs/smoke3_all_scenarios`, re-scored from the per-episode
logs this session):

| scheduler | ratio **mean** | ratio **median** | p25 | p75 | std |
|---|---|---|---|---|---|
| `camper` | **0.2065** | **0.1257** | 0.0538 | 0.3228 | **0.1909** |
| `lstm_balance_100k_1M` | 0.1247 | 0.1234 | 0.1038 | 0.1427 | 0.0388 |
| `lstm_balance_200k_1M` | 0.1259 | 0.1258 | 0.0996 | 0.1512 | 0.0390 |
| `lstm_balance_300k_1M` | 0.1172 | 0.1209 | 0.0956 | 0.1431 | 0.0373 |
| `recency` | 0.1134 | 0.1062 | 0.0958 | 0.1222 | 0.0404 |
| `round_robin` | 0.0589 | 0.0562 | 0.0544 | 0.0592 | 0.0110 |

**The camper's mean sits 64% above its own median**, and its standard deviation is roughly five
times every adaptive rung's. The mechanism is rung 4's whole design: it picks one band and holds
it, so it either lands on a busy one and scores 0.32 or a quiet one and scores 0.05. The mean is
dragged up by the lucky half.

**And because the axis limits are taken from the maximum, that inflated mean stretched the y-axis
for everybody else.** Positions on the drawn figure, as fractions of the axis:

| scheduler | height, means | height, medians |
|---|---|---|
| `camper` | **82%** | 78% |
| `lstm_balance_200k_1M` | 50% | **78%** |
| `lstm_balance_300k_1M` | 47% | **75%** |
| `recency` | 45% | 66% |
| `round_robin` | 23% | 35% |

Under means the figure read as "the RL rungs are well below the camper on interception ratio."
Under medians they are level with it, which is what every other artefact in the same run already
said: paired per-episode, the RL rungs beat the camper on censored intercept time on **97.9–100%**
of episodes and split the ratio column near 50/50. The figure was the only thing disagreeing, and
it was disagreeing because of one statistic.

This surfaced as a reported inconsistency between `comparison.md` and `pareto.png` — the table's
"wins on both" column (paired against `round_robin`) versus the figure's mean positions. Those two
were always answering different questions and both were correct; the figure was nonetheless giving
a false impression, which is what this fixes.

### What changed

- `compare._means` becomes `compare._centres`: it reads `median` from each aggregate instead of
  `mean`, and carries `p25`/`p75` alongside every headline metric.
- `render.pareto` draws interquartile whiskers when those keys are present, and takes its axis
  limits from the whisker tips so a wide rung is not clipped out of its own interval. Whiskers are
  optional — a caller passing only the three headline metrics still gets bare points, which is what
  `tests/test_compare.py` does.
- `render._iqr_arm` converts absolute percentiles into the arm lengths matplotlib wants, clamping
  at zero rather than raising: a p25 above the centre can only mean a caller mixed statistics, and
  one degenerate whisker is more useful than no figure.

**The whiskers are not decoration.** A median point alone has the same failure mode a mean point
does — one dot, no spread — and the camper's vertical arm being four times longer than any other
rung's is now the single most informative mark on the figure. It shows the reader *why* the camper
cannot be trusted at a glance, which two tables and a decision entry had been saying in prose.

### What this does not do

- **It does not change any metric.** D27's definitions, §4's rule that the three are printed
  together, and the paired-dominance counts are all untouched.
- **It does not change the printed table or `EVALUATION.md` §5.** Those keep means, because a table
  has room for an interval beside every number and a scatter plot does not. The two collapse
  differently on purpose, and `_centres`'s docstring says so.
- **It does not re-run anything.** The figure above was redrawn from the existing run's
  `summary.json`, which already carried the percentiles.

**Evidence.** `rfenv/compare.py::_centres`, `rfenv/render/comparison.py::pareto` and `::_iqr_arm`.
Measured this session by re-scoring all 282 per-episode artefacts under
`runs/smoke3_all_scenarios/` through `metrics.read_run` + `metrics.scheduler_metrics` and comparing
mean against median per scheduler. `tests/test_compare.py` and `tests/test_render.py`: 31 passed.

---

## D59 — `measured_dbm` stays in the observation; D34's amplitude exclusion is lifted for it

**Status:** `SETTLED` (2026-09-10) — **ratified by the human**, who is the only one who could:
D49 added the component on D19 observability grounds and never addressed the exclusion it was
reversing, and that gap was flagged as owing a decision rather than built on further.

**What D34 excluded.** D34's base observation deliberately left out "peak amplitude within the
dwell". `measured_dbm` is a *clamped mean* over the dwell's 1–2 slots rather than a peak — softer,
bounded, and averaged — but it is the same class of quantity, so the exclusion had to be answered
rather than sidestepped.

**What it is, precisely.** `receiver.dwell` computes `measured = S + N(0, sigma)` per slot and
declares `Y = measured >= gamma` (`rfenv/receiver.py`). `measured_dbm` is the mean of that
`measured` array, clamped to `[-120, -20]` dBm and rescaled to `[0, 1]` — **the quantity one step
before the binary declaration**, noise included. It is observable in D19's sense: a fielded
receiver has this number. `S` alone would be truth-side and is not exposed.

**Why it earns a slot.** Every other component of the observation is built from `Y`, which is
thresholded. `hit_rate` cannot distinguish an empty band from one holding an emitter sitting just
under gamma. `measured_dbm` is the only component carrying sub-threshold information.

**What is measured and what is not.** Over 1,506 round-robin dwells it reads **0.406** when the
dwell declared a hit against **0.0135** when it did not — it tracks the declaration cleanly, which
a threshold detector guarantees. **What was never measured is whether it adds anything beyond
`hit_rate`**: the sub-threshold case is the entire argument for the component, and how often a band
sits detectable-but-undeclared was not quantified before ratification. Recorded as a known gap, not
as evidence. Anyone claiming the component helps still owes that measurement, or an ablation.

**Scope.** One scalar, global rather than per-band: it reports only the band just left, the same
scope `current_band` has. Making it per-band is a separate change and is not authorised here.

**Evidence.** `rfenv/env.py::ScanEnv._observation` (the clamp and rescale), `rfenv/receiver.py`
(`measured = S + noise`, `Y = measured >= gamma`), `rfenv/baselines/guard.py::MEASURED_DBM`.
Conditional means measured 2026-09-09 this session under round-robin, seeds 0/1/2.

---

## D60 — the development split: 35 configs train, 12 validate, and the pool is rebuilt from the training half

**Status:** `SETTLED` (2026-09-10) — closes a train/evaluation leak that made every RL margin over
rung 5 unsafe. **The rule was written into `rfenv/split.py` before anyone looked at which configs
landed on which side**, which is the whole value of the entry; D39's reasoning about thresholds
applies unchanged to splits.

### The leak

RL training sampled `EmitterPool.from_train()` — every emitter in all 47 development configs.
Evaluation ran `compare.py` over those same 47 stare replays **plus sampled scenarios drawn from
that same pool**. The heuristic rungs do not train, so the asymmetry ran one way only: our agent
had seen the evaluation emitters and rung 5 had not. Any margin that produced is not a real
advantage, and it is exactly the kind that evaporates on the held-out split.

### The rule, fixed in advance

Order the 47 train configs by detectable-emitter count ascending, ties broken by config id. Take
every 4th from index 1 into validation; the rest train. `VALIDATION_EVERY = 4`,
`VALIDATION_OFFSET = 1`, both pinned as literals by `tests/test_split.py`.

**Systematic sampling along the difficulty variable, not a hash or a uniform draw.** Scenario
difficulty spans 2 to 99 emitters and dominates every §4 metric, so a random split can hand
validation a systematically easier or harder set and nobody would know which. Every-4th along the
sorted order makes coverage of the difficulty range structural rather than lucky. Offset 1 rather
than 0 keeps the single easiest config in training, where a degenerate 2-emitter scenario is less
able to distort a 12-config validation set.

### What it produced, reported as it fell

|  | n | min | median | max | mean |
|---|---|---|---|---|---|
| training | 35 | 1 | 38 | 82 | **40.9** |
| validation | 12 | 1 | 37 | 80 | **40.2** |

Validation configs: `config_81`, `config_418`, `config_535`, `config_658`, `config_706`,
`config_719`, `config_940`, `config_1635`, `config_1776`, `config_1902`, `config_2027`,
`config_2445`.

**The split was not re-drawn after seeing this.** It fell balanced; had it fallen lopsided that
would have been reported as a limitation, because re-rolling until a split looks good is the same
error the rule exists to prevent.

### Splitting the config list is not enough, and that is the part that matters

`EmitterPool` is assembled *from* the configs, so a scenario sampled from a pool built over all 47
can contain an emitter belonging to a held-back config. Disjoint config lists over a shared pool
would look like a split and behave like none. `EmitterPool.from_configs(...)` is the new
constructor; `split.training_pool()` and `split.validation_pool()` are built through it.

**Measured: 2,600 contributions / 1,431 distinct emitters in training, 843 / 482 in validation,
and 0 emitters shared.** `tests/test_split.py::test_the_pools_share_no_emitters` asserts it.

### What changed, and what deliberately did not

- `rl.common.make_train_env` now defaults to `split.training_pool()`. Its `pool` argument is
  injectable so checkpoint selection can score against `validation_pool()` without routing around
  the function. Passing `EmitterPool.from_train()` re-opens the leak, which is why it is no longer
  the default, and a test asserts the default did not drift back.
- **`compare.py` still samples from all 47, and that is correct.** The headline is reported over
  the whole development set; it is *training* that must not see it. Reporting the headline on the
  training half would be a different and equally wrong result.
- The 45 held-out test pairs are untouched and remain sealed behind D8's explicit flag.
  `from_configs` refuses any config outside the development set.

### Consequences to state plainly

**Every RL result recorded before this date was produced under the leak**, including the
2,223-episode acceptance run of 2026-09-10 (`runs/acceptance_2026-09-10/`), where rung 9c
Pareto-dominated rung 5 on 48.0% of episodes against being dominated on 13.5%. That measurement is
sound as arithmetic and is **not** evidence of a real advantage over rung 5, because the agent
trained on the emitters it was scored against. It is not written into `EVALUATION.md` §5. The
number to chase is the same comparison re-measured after a retrain on the training half.

**Evidence.** `rfenv/split.py`, `rfenv/scenario.py::EmitterPool.from_configs`,
`rfenv/rl/common.py::make_train_env`. Counts measured 2026-09-10 this session. Pinned by
`tests/test_split.py` (7 tests: rule literals, partition, zero shared emitters, difficulty
coverage, the `make_train_env` default, held-out refusal, determinism).

---

## D61 — the checkpoint-selection rule, fixed before the run that uses it

**Status:** `SETTLED` (2026-09-10) — the quantity it maximises was **ratified by the human** after
a first proposal was corrected (see "the correction", below). Fixed in `rfenv/selection.py` and
pinned by `tests/test_selection.py`.

### The problem it closes

A 1M-step run at `--checkpoint-freq 100000` produces ten checkpoints, and they differ enormously —
on the 2026-09-10 acceptance run, one run's 100k/200k/300k/400k snapshots scored 30.4% / 43.9% /
48.0% / 25.7% paired-both against rung 5. Picking among them *after* seeing evaluation numbers
fits the evaluation set through the choice: ten hypotheses are tried, the best is reported, and the
reported figure carries an optimistic bias of roughly the spread across checkpoints — tens of
percentage points here — which is invisible in the number itself. It is D39's argument about
thresholds applied to model selection, and it gets monotonically worse the more the lane iterates.

### The rule

For each checkpoint, over the **12 validation configs** (D60 — zero emitters shared with the
training pool) at seeds (0, 1, 2), stare replays only (D36), paired per episode against rung 5 on
the same scenario, same seed, same truth grid:

```
net dominance = P(checkpoint Pareto-dominates recency) - P(recency Pareto-dominates checkpoint)
```

Highest net dominance wins. Ties break toward **fewer** training steps.

Four choices, each load-bearing:

1. **Against `recency`, not `round_robin`.** Rung 5 is the bar (`EVALUATION.md` §5). Selecting on
   the floor picks whichever checkpoint is best at clearing something the project does not care
   about.
2. **Pareto-dominance, not means.** A mean can clear a mean while losing most scenarios, and D14's
   finding is that no trivial strategy is good at both objectives — so "wins on both" is the
   question and either axis alone is not.
3. **Net, not the raw win rate.** See below.
4. **Ties toward fewer steps.** Given two checkpoints validation cannot separate, the less-trained
   one has had less opportunity to memorise the training pool and is cheaper to reproduce.

### The correction, recorded because the human approved the wrong version first

The rule was first proposed as *"highest paired-both against recency, ties broken by fewest
episodes where recency dominates"*, together with the claim that it would have picked the 200k
checkpoint over the 300k one. **That claim was false.** The tie-break only fires on an exact tie,
and 48.0% > 43.9% is not a tie, so the win-rate version picks 300k:

| checkpoint | dominates rung 5 | dominated by rung 5 | net |
|---|---|---|---|
| 200k | 43.9% | **4.7%** | **+39.2** |
| 300k | **48.0%** | 13.5% | +34.5 |

The human had approved the rule on the strength of a statement about its behaviour that did not
hold, so the discrepancy was raised rather than quietly resolved, and **net dominance was ratified
in its place**. Being strictly beaten is a real cost: 300k is dominated three times as often as
200k, and a model that rarely loses outright is the better bet for surviving a held-out run than
one that sometimes wins bigger.

### First execution — machinery only, not a valid selection

Run against the three `lstm_balance_1M_s*` checkpoints, 12 validation configs × 3 seeds:

| checkpoint | steps | dominates | dominated | **net** |
|---|---|---|---|---|
| `lstm_balance_1M_s2` | 200,000 | 55.6% | 8.3% | **+47.2%** ← selected |
| `lstm_balance_1M_s3` | 300,000 | 36.1% | 5.6% | +30.6% |
| `lstm_balance_1M_s1` | 100,000 | 36.1% | 11.1% | +25.0% |

**This selects nothing.** All three were trained before D60, on a pool built from all 47 configs
including the 12 now called validation, so the model has seen these emitters. The table
demonstrates the rule executes; it is not evidence about any checkpoint. Note also that 200k and
300k **tie on win rate** here at 36.1% — under the rejected version this would have fallen to the
tie-break, which is the fragility net dominance removes.

Worth recording: validation net dominance reads +47.2% for 200k against +39.2% on the full
development set. Different scenario sets, so not a clean comparison — but it runs in the direction
expected if the policy is partly recalling emitters rather than generalising, which is why D60 had
to come before this entry.

### What is deliberately not covered

- **The rule does not choose hyperparameters, rewards or algorithms.** It picks among checkpoints
  *within* one run. Choosing across runs is the same failure mode one level up and needs the
  iteration ledger to be honest about how many configurations were tried.
- **It does not report anything.** The selected checkpoint's headline is measured on the full 47 by
  `compare.py`, with the number of configurations tried stated alongside.

**Evidence.** `rfenv/selection.py` (`SELECTION_REFERENCE`, `SELECTION_SEEDS`, `SELECTION_SOURCE`,
`Candidate.net_dominance`, `select`). Pinned by `tests/test_selection.py` — 7 tests including one
asserting `evaluate` cannot reach `EmitterPool.from_train` (the D60 leak arriving by a different
door) and one asserting the *direction* of each metric literally, because reversing "lower
intercept time is better" would silently select the worst checkpoint every time with the whole
suite green. Table above measured 2026-09-10 this session.

---

## D62 — every reward candidate is screened against the ladder before anything trains on it

**Status:** `SETTLED` (2026-09-10) — a new gate in front of D47, requested by the human. Criteria
fixed in `rfenv/reward_gate.py` before the first run and pinned by `tests/test_reward_gate.py`.

### Why a screen in front of D47

D47 fixed the rule for *selecting* a reward: the candidate beating round-robin on both headline
metrics on the most paired episodes. It has never been run, and it is expensive — it needs a
trained agent per candidate, hours each. This screen needs no agent at all.

The argument is short. If a reward cannot rank rung 5 above rung 2 — two fixed, known policies,
one measurably better on the metrics that decide the project — then no policy trained on it can be
expected to discover the difference either, whatever the hyperparameters. Gradient descent
optimises the reward it is given; it cannot recover an ordering the reward does not encode. Such a
candidate is excluded in minutes on a CPU instead of after a 1M-step run.

### The criteria, fixed in advance

Score rungs **2, 4, 5 and 6a** under each candidate, 8 seeds, paired per seed rather than averaged.
A candidate passes only if:

- `mean(rung5 − rung2) / std(rung5 − rung2) >= 2.0` — separation in units of its own noise, because
  absolute gaps are meaningless across candidates whose scales run from hundreds to thousands;
- rung 5 above rung 2 on **>= 7 of 8 seeds** — large on average is not enough if it is inconsistent;
- the camper **>= 1.0σ below both** sweeping policies.

Rung 6a is in the set precisely because we did not write it: a reward tuned until it happens to
like our own two policies would still be caught out by one that arrived from a paper (D45).

**The criteria are literals in the file and asserted by a test**, exactly as `validate.py::GATES`
is asserted against D39, and for the same reason: a threshold chosen once the measurement is
visible is not a threshold. If a candidate we like fails, the answer is to change the candidate.

### Measured, 6 candidates × 4 rungs × 8 seeds

| candidate | rung 2 | rung 5 | rung 4 | rung 6a | sep | seeds | camper | verdict |
|---|---|---|---|---|---|---|---|---|
| `reward_balance` | 217.9 | 276.9 | **−414.1** | 278.7 | 3.0σ | 8/8 | 7.3σ | **PASS** |
| `weighted` | 1455.7 | 1817.2 | 1323.0 | 1928.4 | 2.6σ | 8/8 | 0.4σ | FAIL |
| `hit_z` | 196.4 | 299.1 | **409.8** | 376.6 | 3.1σ | 8/8 | −2.3σ | FAIL |
| `hit_y` | 165.4 | 270.5 | **372.4** | 332.0 | 2.6σ | 8/8 | −1.9σ | FAIL |
| `greedy` | 459.0 | 776.6 | **1198.3** | 984.2 | 2.7σ | 8/8 | −2.7σ | FAIL |
| `explore` | 1277.0 | 1238.1 | −205.2 | 1034.0 | **−2.5σ** | **0/8** | 31.5σ | FAIL |

**`reward_balance` is the only survivor**, and it is what every current rung 9 checkpoint was
trained on.

### The finding that matters most

**`hit_z` and `hit_y` — D29's original two candidates, carried since 2026-09-03 — both fail.** They
rank rung 4 above every sweeping policy, which is D14's tension stated in reward form: a reward
that pays only for hits pays most for camping on the densest band. No agent trained on either can
be expected to beat the floor, and both have been live options for the RL lane the entire time. The
screen cost minutes and would have saved every DQN and PPO run in this repository, all of which
trained on `hit_z` or `hit_y` (D48).

**Consequence, same day: both were removed from `REWARDS` entirely — see D63.** This entry's
numbers (six candidates screened, `reward_balance` the sole survivor) describe the screen as run,
before that removal; D63 is the record of the registry edit that followed from it.

`greedy` failing is by construction — D57 built it as the exploit corner and predicted exactly
this. `explore` fails from the other side: it is the only candidate that ranks rung 2 *above* rung
5, on 0/8 seeds, because nothing in it rewards finding the busy bands faster.

### What the screen does not do

- **It is a necessary condition, not a sufficient one.** A reward can order four fixed policies
  perfectly and still be unlearnable, badly scaled, or wrong at the margin. Passing is not evidence
  a candidate is good.
- **It does not rank the survivors.** D47 does that, and D47 still has not been run — the screen
  only decides what D47 is allowed to consider.
- **It says nothing about hyperparameters.** A candidate that passes can still fail to train.

**Evidence.** `rfenv/reward_gate.py`. Measured 2026-09-10 this session over
`EmitterPool.from_train()` at seeds 0–7. Pinned by `tests/test_reward_gate.py` — 6 tests, including
`test_every_screened_rung_is_a_real_registered_rung`, which exists because D56 was inverted by
scoring a hand-written stand-in instead of a registered rung.

---

## D63 — `hit_z` and `hit_y` are removed from `REWARDS` entirely, as a consequence of D62

**Status:** `SETTLED` (2026-09-10) — a change to a gated category (the reward), made directly by
the human by editing `rfenv/env.py`, confirmed when raised. Recorded here because CLAUDE.md's
working rules require it and because the edit has a wide blast radius across code, tests and docs
that referenced the two by name.

### What changed

`reward_hit_z` and `reward_hit_y` — D29's original two candidates, registered since 2026-09-03 —
are deleted from `rfenv/env.py`, along with their `REWARDS` entries. `REWARDS` now holds exactly
**four** keys: `reward_balance`, `greedy`, `explore`, `weighted`. `sorted(REWARDS)` is
`['explore', 'greedy', 'reward_balance', 'weighted']`.

### Why

D62's screen, run the same day, found both fail decisively:

| candidate | separation (rung 5 − rung 2) | camper margin | verdict |
|---|---|---|---|
| `hit_z` | 3.1σ | **−2.3σ** | FAIL — camper ranks *above* both sweeps |
| `hit_y` | 2.6σ | **−1.9σ** | FAIL — same failure |

Both rank the camper above every sweeping policy — D14's tension stated in reward form, a reward
that pays only for hits pays most for camping on the densest band. No agent trained on either could
be expected to beat the floor, and both had been live default options for the RL lane since D29.
Every DQN and PPO checkpoint in this repository was trained on one of the two (D48) — already
permanently unloadable from D49's observation-width change, so nothing currently loadable is lost
by the removal itself.

D62 stopped at recording the finding; this entry is the registry edit that followed from it,
made directly rather than proposed first. Raised and confirmed rather than reverted, per the
working agreement that an unexpected change to a gated file gets checked before being built on.

### Blast radius

Removing two long-standing registry keys touched more than the registry:

- **`rfenv/env.py`** — the two functions and their `REWARDS` entries gone; the module-level
  comment block explaining D31's per-slot invariant rewritten (it previously used `hit_z`/`hit_y`
  as the worked examples); `reward_greedy`'s docstring, which compares itself against `hit_y`,
  updated to past tense.
- **`rfenv/baselines/ladder.py`** — rungs 7, 7a and 8's descriptions annotated: already
  permanently unloadable from D49, now doubly so since their registered reward no longer exists.
  The ladder still carries them (a `Rung` with a dead checkpoint skips with a warning via
  `checkpoint_is_usable`, rather than erroring) so nothing crashes; the docstrings now say why.
- **`rfenv/rl/__init__.py`** — the package's usage examples used `--reward hit_z`/`hit_y` as live,
  copy-pasteable commands; both now raise `ValueError` on the current registry, so the examples
  are rewritten to `reward_balance`/`greedy`.
- **Tests — 37 failures, all fixed.** Most were generic training/metrics fixtures that happened to
  default to `reward="hit_z"` as an arbitrary valid value with no dependency on its specific shape;
  swapped to `DEFAULT_REWARD`. Two were not generic and needed real rework:
  - `tests/test_env.py::test_a_wide_dwell_scores_both_its_slots` and
    `::test_reward_per_unit_time_is_equal_across_dwell_widths` pinned D31's per-slot invariant to
    `hit_z` specifically, because it was a bare per-slot `Z` count and none of the four survivors
    is. Rewritten to inject a raw `lambda dwell, ...: float(dwell.Z.sum())` via `env._reward_fn`
    directly, testing `ScanEnv.step`'s slot-summation mechanics independent of whichever
    candidates happen to be registered — a more durable test than depending on registry contents.
  - `tests/test_reward_gate.py::test_the_screen_separates_a_known_good_reward_from_a_known_bad_one`
    used `hit_z` as its known-bad control; swapped to `greedy` (D57's exploit corner, which fails
    the same check by construction) since `hit_z` no longer exists to screen.
- **Docs** — `run.md`, `ENVIRONMENT_SPEC.md`, `EVALUATION.md`, `EDGE_LANE_HANDOFF.md`, and the
  amendment banners in `RL_LANE_HANDOFF.md`/`RL_TEAM_HANDOFF.md`/`STATE_ACTION_FORMULATION.md`
  corrected: live example commands using `--reward hit_z`/`hit_y` rewritten (they now raise), and
  present-tense "six candidates, hit_z and hit_y fail" language moved to past tense with the
  removal stated. Deep historical worked-examples inside the three long handoff documents (their
  original day-by-day task tables, file-tree listings) are **not** individually rewritten — those
  documents already carry a prominent amendment banner at the top saying to read it before
  trusting anything below, and rewriting hundreds of lines of legacy planning prose for one
  registry edit is not proportional.

### What this does not do

- **It does not change `DEFAULT_REWARD`**, still `reward_balance`.
- **It does not invalidate any currently-loadable checkpoint.** Rungs 7/7a/8 were already
  unloadable from D49; nothing that worked yesterday stops working today.
- **It is not a statement that `hit_z`/`hit_y` were bad ideas at the time.** D14's tension was not
  known to be this severe until D62 measured it; they were D29's original two candidates and D47's
  selection rule was written to choose between exactly this kind of pair.
- **It does not touch D29's truth/observation split** (what a reward may read) — only which
  candidates are registered.

**Evidence.** `git diff` on `rfenv/env.py` (the deletion). D62's table, same session. Suite after:
323 passed, 60 skipped, 1 failed (the pre-existing held-out-split guard).

---

## D70 — the scheduler takes a threat priority from outside; it does not compute one

> **Renumbered from D64 on 2026-09-11.** Two lanes allocated D64 in parallel — this entry
> and the RL lane's first clean result — and both reached `main`. The RL chain D64–D69 is
> referenced from `CLAUDE.md`, `EVALUATION.md`, `ITERATION_LEDGER.md`, `MODEL_COMPARISON.md`,
> `PHASE_SWITCH_FUTURE_WORK.md` and `ladder.py`; this entry was referenced only by its own
> heading, so it is the one that moves. It keeps its position in the file rather than being
> shuffled to the end — the number is authoritative, the ordering is not.
> **Allocate new numbers from the highest on `main` after a pull, never from a local tree.**
> `scripts/doctor.py` now fails on a duplicate.

**Status:** `SETTLED` (2026-09-11) for the direction. Implementation brief:
`docs/project/THREAT_WEIGHTING_BRIEF.md`. Two things inside it are open and named at the end.

**The gap.** A modern ES system already knows which emitters are dangerous — the threat library is
in the processing unit, the operator has a mission brief. **None of it reaches the scan
scheduler**, which sweeps a schedule computed before the mission. The PS names this exactly:
*"Open loop strategies … may lose time to nonthreatening emitters by not giving time to new or
threatening ones."* The word "threat" appears **zero times** in `EVALUATION.md`.

**Decision.** The scheduler accepts a per-band priority vector `p`, learns to use it (sampled per
episode during training), and defaults to `p = 1` — which is the plain PS objective, unchanged.
`p` has two sources and they are **the same code**: a human operator, or a downstream categoriser.

**Threat is not ours to define, and this is measured, not assumed.** Threat is doctrine plus
mission context; a fire-control radar is dangerous because of what it is attached to, not because
of anything in its pulse train. Three attempts to derive it from data, all failed:

1. **Infer it from what the receiver observes.** Band level, which needs no attribution, over 772
   occupied (config, band) pairs: hit rate AUC **0.483**, mean level **0.405**, max level
   **0.381**, level std **0.429**, intermittency **0.617**. **Nothing clears 0.62.** Mechanism:
   `S = max` over contributors (D25) — a 0.3 kW fire-control radar sharing a band with a 750 kW
   weather radar never reaches that band's statistics. Emitter level is no better and needs
   attribution anyway (deinterleaving, out of scope by D12/D19).
2. **Use "hard to intercept" as a proxy.** **AUC 0.514** for predicting HIGH threat — a coin flip.
   Median duty does trend (HIGH 0.682, MEDIUM 0.890, LOW 0.966) but the distributions overlap
   almost completely. Difficulty is driven by transmit power (`r = +0.485`; duty 0.112 → 0.999
   across power quartiles), **not** by range (`r = −0.051`), and not by threat.
3. **"Threat = the busiest bands."** **Inverted.** Threat fraction by band-density quartile runs
   **0.379 / 0.227 / 0.116 / 0.091** from sparsest to busiest — 4.2× more threat-dense in the
   quietest bands. The camper parks precisely where the threats are not, which gives D14's camper
   pathology an operational reason as well as a metric one.

**Withdrawn in the process.** An earlier draft of this entry weighted the reward by
`1 + λ(1 − duty_e)` and called the result "threat-weighted". Finding 2 kills it; do not
reintroduce it. The 7.5% "threat signature" it was built on (frequency-agile ∧ footprint wider
than one band window) captures only **12%** of HIGH-threat emitters and was an unrepresentative
slice with memorable names in it.

**Consequence, and it is the point: there is no autonomous threat mode, and we do not claim one.**
A self-categorising scheduler would be false. What we claim is the interface — and **no
scan-scheduling paper in the reference set has it**: Köksal, Apfeld, Gul & Erer, Clarkson and
Teissier all give every emitter the same dwell.

**Where categorisation belongs.** ADITI 4.0's own decomposition (transcript p.2): *"the system
basically consists of antenna, receiver, **processing unit, database** to provide the means to
intercept, identify, analyze and localize."* Categorisation is a processing-unit function over the
full PDW stream from dwells the scheduler already won. We are a different box, and the
*"feedback based decision making mechanism"* ADITI asks for between them is this interface.
**SIH26055 does not ask for categorisation** and the 2026-09-01 audit records ADITI as context for
intent, not scope — so we specify that box and leave it out of scope.

Also relevant to D20: asked directly whether a threat library would be supplied, the officer
answered *"this I cannot comment now"* (p.8), and an academic in the room argued a cognitive system
should *"generate your own"* — uncontradicted. D20's cold start, corroborated from the customer's
own outreach. **D20 is unaffected by this entry:** the *scheduler* still starts with no emitter
intelligence; `p` is mission input, not a learned prior.

**The example threat library, for scoring and the demo.**
`metadata/transmitters/*/.attrs['function']` names 68 emitter types; classified as a real RWR
would: **HIGH** (fire control, engagement, missile guidance, counter-battery, LPI) **656, 38.5%**;
**MEDIUM** (air-defence search, early warning, surveillance, maritime patrol) **775, 45.5%**;
**LOW** (weather, marine navigation, airport/ground movement, SAR, GPR) **273, 16.0%**. **Labelled
as an example instance, never as our threat model**, fixed in code before any scheduler is scored
against it.

**What does not move.** `rfenv/constants.py` untouched — D42 puts reward candidates and
observation extensions deliberately outside the freeze list, D57 lifted D29's cap. No gate or
baseline re-runs. `DEFAULT_REWARD` stays `reward_balance` as the control arm. The cost that *is*
real: `p` widens the observation 146 → 182, so **every existing checkpoint dies**, as at D49 and
D55.

**Open inside this decision:**

1. **Threat-split reporting in `EVALUATION.md` §4 is not yet adopted.** A scorecard change, so
   gated. It needs no retrain and is step 0 of the brief — recommended, not taken.
2. **The `p` sampling distribution during training is unspecified.** It decides whether the policy
   generalises across priority settings or overfits one. Fix it before the retrain.

**Evidence.** Measured 2026-09-11 via scratch scripts against `data/turing/stare/train_stare/*.h5`
and `rfenv.truth`/`rfenv.scenario` at the frozen γ = −111 dB — all 47 stare replays, 1,704
detectable emitters, 2,363 train transmitters, 772 occupied band-pairs. Sourced: the PS Background
paragraph; ADITI transcript pp. 2 and 8, read 2026-09-11.

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

---

## D64 — the first clean RL result: `reward_balance` retrained and selected under D60/D61

**Status:** `MEASURED` (2026-09-10) — a checkpoint chosen entirely by the pre-registered D61 rule,
on a pool that never saw the checkpoint's own training data (D60). No previous RL number in this
repository can say both of those things.

### What ran

`reward_balance`, 400k timesteps, seed 0, `--hyperparam ent_coef=0.01 gamma=0.997 n_steps=8192`,
run name `clean_lstm`, started at commit `176e6a9` — after D60 (`36405b1`) and D62/D63
(`36405b1`/`46d8449`) were both already in the tree. Every RL checkpoint recorded before this one
predates at least one of the two, which is what "no clean RL result" has meant since D60. Full
command and manifest reasoning in `scratch/TRAINING_JOURNEY.md` §14.1.

### Selection (D61), run for the first time on real candidates

| checkpoint | steps | dominates rung 5 | dominated by rung 5 | net dominance |
|---|---|---|---|---|
| `clean_lstm_s4` | 400k | 47.2% | 11.1% | **+36.1%** ← selected |
| `clean_lstm_s2` | 200k | 41.7% | 11.1% | +30.6% |
| `clean_lstm_s1` | 100k | 36.1% | 11.1% | +25.0% |
| `clean_lstm_s3` | 300k | 19.4% | 19.4% | +0.0% |

12 validation configs × 3 seeds = 36 paired episodes per checkpoint, zero emitters shared with
training (D60). `clean_lstm_s4` (400k) selected — chosen by the rule, not by looking at which
number was largest on some other measure first. The dip at 300k is reported, not explained away.

### What this settles and what it does not

**Settles:** the split (D60) and the selection rule (D61) both run end-to-end on a real training
run, not a demonstration against stale checkpoints (§12.5's `lstm_balance_1M_s1..s3` run "selected
nothing" — all three predated D60). This is a checkpoint this repository can defend as clean.

**Did not settle at the time this entry was first written:** the headline. That gap is closed by
D65, written the same day — `compare.py` has since run against `clean_lstm_s4` (rung `10d`)
specifically, paired against recency: **73.7% ratio / 31.6% cTTI / 22.8% both**, now in
`EVALUATION.md` §5. The interim comparison mentioned in an earlier draft of this entry
(`runs/clean_baseline_100k`, which scored only the 100k snapshot mixed with two pre-D60
checkpoints) is superseded by that run and was never itself written into `EVALUATION.md` §5. All
four checkpoints are registered on the ladder — rungs `10a`–`10d` (`lstm_balance_clean_100k_400k` /
`200k_400k` / `300k_400k` / `400k`), one per checkpoint, `10d` pointing at the D61-selected
`clean_lstm_s4.zip`.

### A parallel treatment run

A second run, identical except `--reward reward_balance_improved`, was launched by the human to
compare against the control above — the paired experiment needed to say whether the density-
weighted occupancy term (built after D62/D63, discussed but not yet decided as a candidate) helps
a learner, isolated from the D60 split itself. Single seed per arm: a difference is suggestive
against the control's own double-digit swing between adjacent checkpoints (see the table above),
not conclusive. Tracked in `docs/project/ITERATION_LEDGER.md` and `scratch/TRAINING_JOURNEY.md`
§14.4; picked up here once selected.

**Evidence.** `runs/checkpoints/clean_lstm_s{1,2,3,4}.{zip,json}`, D61 selection output reproduced
this session. `docs/project/ITERATION_LEDGER.md` row `clean_lstm`.

---

## D65 — the paired reward comparison: `reward_balance` vs `reward_balance_improved`, both clean

**Status:** `MEASURED` (2026-09-10) — the treatment run from D64 finished, was selected the same
way as the control, and both winners were scored on the full development-set headline together.
**Single seed per arm throughout: read this as suggestive, not conclusive.**

### The treatment arm, selected the same way as the control

`lstm_balance_improved` — identical to `clean_lstm` (D64) except `--reward
reward_balance_improved`, same commit, same seed, same hyperparameters, same D60 split. D61
selection on its four checkpoints, 12 validation configs × 3 seeds:

| checkpoint | steps | dominates rung 5 | dominated by rung 5 | net dominance |
|---|---|---|---|---|
| `lstm_balance_improved_s2` | 200k | 36.1% | 11.1% | **+25.0%** ← selected |
| `lstm_balance_improved_s4` | 400k | 33.3% | 11.1% | +22.2% |
| `lstm_balance_improved_s1` | 100k | 22.2% | 11.1% | +11.1% |
| `lstm_balance_improved_s3` | 300k | 36.1% | 27.8% | +8.3% |

`lstm_balance_improved_s2` (200k) selected. On validation alone, **the control wins**: +36.1%
(D64) against the treatment's +25.0%. Both ladder entries: rungs `10a`–`10d` (control) and
`11a`–`11d` (treatment) in `rfenv/baselines/ladder.py`.

### The headline: `compare.py`, both winners together, full development set

```
venv/Scripts/python.exe -m rfenv.compare --rungs round_robin,recency,lstm_balance_clean_400k,\
  lstm_balance_improved_200k_400k --seeds 3 --sampled 10 --figures --out runs/clean_paired_comparison
```

4 rungs × 57 scenarios (47 stare + 10 sampled from the full 47-config pool, per D60's rule that the
headline reports over the whole development set) × 3 seeds = 684 episodes.

Paired against **recency** — rung 5, the bar:

| scheduler | ratio | cTTI | both |
|---|---|---|---|
| `round_robin` | 7.0% | 28.7% | 3.5% |
| `lstm_balance_clean_400k` (control, `reward_balance`) | 73.7% | 31.6% | 22.8% |
| `lstm_balance_improved_200k_400k` (treatment, `reward_balance_improved`) | **83.0%** | **39.8%** | **31.0%** |

**Both clean checkpoints clear rung 5 comfortably. On this measure, the treatment is ahead of the
control** — higher on both paired metrics and on the joint `both` column (31.0% against 22.8%).

### The reversal, stated plainly

D61 selection (validation, 36 paired episodes per checkpoint) ranks the control ahead. The
headline (684 episodes, full development set) ranks the treatment ahead. **Both numbers are real
measurements; they are not measuring the same thing.** Validation asks "which checkpoint from this
run looks best on 12 held-back configs," per-run, before anything else is seen — it answers a
different question than "how do the two arms' selected checkpoints compare on the full development
set," which is what the headline reports.

**Why this is not read as "the improved reward wins":**

1. **One training seed per arm.** The control's own checkpoints swing from +25.0% to +30.6% to
   +0.0% to +36.1% net dominance across four snapshots of the *same* training run (D64) — a swing
   larger than the +8.2-point headline gap between the two arms' selected checkpoints. A single
   seed cannot separate a reward effect from a training run finding a different point on that same
   kind of curve.
2. **The winning checkpoints are at different step counts** (400k for control, 200k for treatment),
   so the comparison is between "reward X's best snapshot" and "reward Y's best snapshot," not a
   controlled point-in-training comparison — which is correct per D61 (selection picks whatever it
   picks) but means the 8.2-point gap is not isolated to the reward term alone.
3. It does corroborate the direction from `docs/project/DESIGN_QA` discussion earlier the same
   day: `reward_balance_improved` was never rejected, only unproven on the D62 screen's own terms
   (screen measures discrimination between six fixed heuristics; a learner consumes a different
   signal — the gradient — which the screen cannot see). This headline is the first evidence, weak
   as it is, in the direction the "unproven rather than rejected" argument predicted.

**What would make this conclusive:** matching seed counts per arm (2–3 seeds each, ~2.5–4 hours
total) and comparing net dominance distributions rather than single point estimates, exactly the
gap flagged when this experiment was proposed.

**Evidence.** `runs/checkpoints/lstm_balance_improved_s{1,2,3,4}.{zip,json}`, D61 selection and
`compare.py` output both reproduced this session, artefacts in `runs/clean_paired_comparison/`.
`docs/project/ITERATION_LEDGER.md` rows `clean_lstm` and `lstm_balance_improved`.

---

## D66 — a hard explore/exploit gate, measured: worse than the camper it was meant to fix

**Status:** `MEASURED` (2026-09-10) — two new rungs, both negative results, recorded because a
result that says "this doesn't work" is exactly as much a fact as one that says it does.

**The code below was removed the same day**, on the strength of this measurement — `rungs 12/13`
are no longer registered, `rfenv/baselines/phase_switch.py` is deleted, and every reference in
`ladder.py`/`__init__.py`/`tests/test_baselines.py` is gone. This entry stays as the record of what
was tried and why it didn't work; `docs/project/PHASE_SWITCH_FUTURE_WORK.md` carries the two
directions (Options A and C, discussed but not built) that are still worth trying if this line of
work is picked up again.

### Why this was built

None of the clean D60/D61 checkpoints camp at all (D64/D65): maximum dwell streak 6 slots across
every sample episode checked, indistinguishable from round-robin's 1-2. Rather than "loosen the
reward and hope," the question was narrowed to something a heuristic can test cheaply: does a
*hard, external* commit/release gate — explicitly separate from the reward, from the observation,
from any retraining — produce useful phase-switching behaviour at all, before spending any effort
teaching a network to do it internally.

### What was built

`rfenv/baselines/phase_switch.py` — `_CommitReleaseGate`: commit to a band on `ENTER_HITS = 2`
consecutive declarations, release after `EXIT_MISSES = 2` consecutive misses. Both thresholds
reused verbatim from `ApfeldParams.d`/`ApfeldParams.s` (already justified in `apfeld.py`), not
searched against either rung's score.

- **Rung 12, `phase_switch`.** Explores via `EQUAL_AIRTIME_CYCLE` (rung 2's own sweep). Isolates
  the gate mechanism from everything else Apfeld's strategy does (period estimation, probabilistic
  mixing), the way rung 6a isolates period estimation from the tentative list (D45).
- **Rung 13, `rl_phase_switch`.** The identical gate, exploring with the D61-selected clean
  checkpoint's (`clean_lstm_s4`, rung 10d) own `.predict()` instead of the sweep — checkpoint used
  exactly as trained, no retraining, no reward or observation change. The hidden state advances on
  every step regardless of whether its suggestion is used, so the LSTM's memory tracks what
  actually happened, including gate-overridden slots.

### What was measured

`python -m rfenv.compare --rungs round_robin,recency,camper,apfeld_active_rfs,phase_switch,rl_phase_switch --seeds 3 --sampled 10 --out runs/phase_switch_comparison` — 57 scenarios × 3 seeds × 6
rungs. Paired against recency (the bar):

| scheduler | ratio | cTTI | both |
|---|---|---|---|
| `round_robin` | 7.0% | 28.7% | 3.5% |
| `camper` (rung 4) | 55.6% | 2.9% | 0.6% |
| `apfeld_active_rfs` (rung 6a) | 74.3% | 27.5% | 23.4% |
| `phase_switch` (rung 12) | 67.3% | 4.7% | 4.1% |
| `rl_phase_switch` (rung 13) | 58.5% | 4.1% | 2.9% |

Unpaired means: `phase_switch` ratio 0.256 / cTTI 12.83 / coverage 0.440; `rl_phase_switch` ratio
0.222 / cTTI 15.19 / coverage 0.348; `camper` ratio 0.209 / cTTI 9.67 / coverage 0.497.

### The finding

**Both gated rungs are worse than the plain camper on censored intercept time** (12.83s / 15.19s
against camper's 9.67s) **and worse on coverage** (0.440 / 0.348 against 0.497). Streak analysis
(measured directly on episode logs, not inferred) explains why: on a scenario with sustained
activity the gate commits and does not release, for the same reason `PhaseSwitch`'s own docstring
states plainly — a persistently active band gives the gate no evidence to release on, and staying
committed is the *correct* call given that evidence. The gate is not a broken camper; it is a
conditional one, and the condition triggers on exactly the scenarios where camping already hurts
most.

**Using the trained policy's own suggestions during explore made every metric worse, not better**
— ratio 0.222 vs 0.256, cTTI 15.19 vs 12.83, coverage 0.348 vs 0.440, both-vs-recency 2.9% vs
4.1%. Unverified hypothesis, not measured this session: `reward_balance`'s policy was trained
assuming continuous control of its own actions, and its LSTM hidden state — advanced every step
including gate-overridden ones — may be reasoning from a trajectory it never experienced during
training, degrading its suggestions once explore resumes after a release. Would need inspecting
the action distribution around a release specifically to confirm.

**Neither threshold was retuned after seeing this.** Both were fixed from Apfeld's own parameters
before either rung was measured, per the same discipline `reward_gate.py` and `validate.py::GATES`
already hold to. If different thresholds are wanted, that is a new, separately-justified rung, not
a revision of this one's numbers.

### What this does and does not close

Answers the narrow question it was built to answer: a hard external gate, in this form, is not the
missing piece. It does not touch D62 (no reward changed), D60/D61 (no training happened, no
checkpoint selection reopened), or D64/D65 (the clean checkpoints' own numbers are unchanged). The
open question from D64/D65 — why the trained policy never commits to anything on its own — is
narrower now: an external gate acting on the same policy's suggestions doesn't rescue it, so
whatever would produce useful phase-switching behaviour has to come from inside training (the
reward or the observation), which is exactly the two costlier options this was built to check
before spending on.

**Evidence.** `rfenv/baselines/phase_switch.py`, `rfenv/baselines/ladder.py` rungs 12/13,
`tests/test_baselines.py` (12 new tests, direct mechanism tests plus real-scenario behaviour).
`compare.py` output reproduced this session, artefacts in `runs/phase_switch_comparison/`.

---

## D67 — the observation gains a hit-streak feature: 146 → 183 (Option A)

**Status:** `SETTLED` (2026-09-10) — proposed with the exact spec and cost stated up front
(`docs/project/PHASE_SWITCH_FUTURE_WORK.md`'s Option A), confirmed by the human before `env.py` was
touched, per the working rule that an observation change is not this agent's call to make alone.

### What changed

Two blocks appended after `measured_dbm` — existing slice offsets (`HIT_RATE`, `VISIT_DENSITY`,
`STALENESS`, `CURRENT_BAND`, `CLOCK`, `MEASURED_DBM`) are untouched, so no heuristic rung needed
updating:

- `HIT_STREAK` (36-wide, per band): consecutive declared hits on that band **across visits**, not
  reset by a visit to a different band — only by an actual miss on this one. Capped at
  `_STREAK_CAP = 5` and divided down to `[0, 1]`, the same convention D55 already uses for
  `visit_density`/`staleness`.
- `CURRENT_HIT_STREAK` (1 scalar): `HIT_STREAK` at whichever band `CURRENT_BAND` is one-hot on.
  Redundant with the per-band block plus a dot product, kept anyway as a direct scalar for the same
  reason `current_band` exists alongside `staleness`.

`146 → 183` (`N_BANDS * 5 + 3`). Both `_STREAK_CAP = 5` and the decision to track only the current
band rather than a full per-band block were fixed before anything was measured, not tuned against
a score — `5` because it sits comfortably above the (now-removed) D66 gate's own 2-hit commit
threshold, giving room to distinguish "just crossed that bar" from "been hot a while", while still
saturating within a revisit cadence a 30 s episode can afford.

### Why

D66 measured that no RecurrentPPO checkpoint trained on the 146-wide vector ever commits to a band
at all — maximum dwell streak 6 slots, indistinguishable from round-robin's 1-2 — and that
wrapping a hand-coded gate around the trained policy's own suggestions made things *worse*, not
better. `hit_rate` is cumulative over the whole episode, so a band hot for five slots and cold
since reads the same as one that just turned hot; nothing in the 146-wide vector told the agent
"this specific band, right now, has hit twice in a row" — the exact quantity the deleted heuristic
gate computed and acted on. This change gives the agent that signal directly and tests whether the
absence of it, not the reward, was the bottleneck. It changes only what the agent can see, not what
it is scored on — no reward function's signature changed.

### Cost, paid in full

**Every checkpoint that predates this commit is now permanently unloadable** — `clean_lstm_s4`
(D64), `lstm_balance_improved_s2` (D65), every snapshot of both arms, every earlier RL checkpoint
in `runs/checkpoints/`. `require_loadable()` (D49) refuses them with the retrain command rather
than failing silently, exactly as designed.

**A gap in the guard that this exposed, and closed as part of this change.**
`tests/test_baselines.py::_unusable_checkpoint` pre-checked a hand-maintained table of rung-number
→ checkpoint-path, written for rungs 7-9d and never extended as 9e-9j/10a-d/11a-d were registered.
The moment this observation change made every checkpoint stale at once, the untracked rungs' tests
**failed outright** (85 failures) instead of skipping cleanly — `load_checkpoint()` already raises
the correct, informative error via `require_loadable()`, but nothing in the newer rungs' path ever
called it before the assertion ran. Rewritten to try building the rung directly and catch
`(FileNotFoundError, ValueError)`, removing the table entirely: self-healing for any future rung,
nothing left to fall behind.

### What is still open

The D65 matched-seed plan (settling whether `reward_balance` or `reward_balance_improved` really
differs, or whether the single-seed reversal was noise) was specified for the 146-wide observation
and has not been run on it — this change superseded it before those runs produced a checkpoint (see
`scratch/TRAINING_JOURNEY.md` §15 for the crash that interrupted the first attempt). It is folded
into the retrain this decision requires rather than run separately: fresh training under 183 is
needed regardless, so the matched-seed question is answered on the current observation, not the
superseded one.

**Evidence.** `rfenv/env.py` (`_observation`, `reset`, `step`, the `Box` construction),
`rfenv/baselines/guard.py` (new slice constants), `rfenv/rl/common.py::current_observation_width`.
Suite after: 287 passed / 144 skipped / 1 pre-existing failure (held-out data absent locally).

### The retrain, measured

Both arms retrained under 183, one at a time (a laptop crash interrupted the first attempt at
several in parallel — `scratch/TRAINING_JOURNEY.md` §15 — hence the change in practice). D61
selection, 12 validation configs × 3 seeds:

| arm | selected checkpoint | dominates rung 5 | dominated | net dominance |
|---|---|---|---|---|
| control (`reward_balance`) | 300k (`lstm_balance_d67_control_s3`, rung 14c) | 36.1% | 11.1% | +25.0% |
| treatment (`reward_balance_improved`) | 100k (`lstm_balance_d67_treatment_s1`, rung 15a) | 38.9% | 5.6% | **+33.3%** |

Both lower than their 146-wide predecessors' own validation scores (D64's control +36.1%, D65's
treatment +25.0%) — not a like-for-like comparison, different observation, but the ordering
between the two arms held: treatment beats control on validation this time (it lost to control
under 146).

**Did the streak feature produce the hoped-for commitment behaviour? No.** Measured directly on
episode logs, both selected checkpoints, three sample scenarios: maximum dwell streak 4–6 slots on
every one — the same order of magnitude as every pre-D67 checkpoint (D64/D66 measured max 6). The
feature exists in the observation now; neither policy has learned to act on it by committing to a
band.

**The headline moved anyway, modestly, in the same direction as D65.**
`compare.py --rungs round_robin,recency,lstm_balance_d67_control_300k_400k,lstm_balance_d67_treatment_100k_400k --seeds 3 --sampled 10 --figures` — paired against recency:

| scheduler | ratio | cTTI | both | (D65's pre-D67 equivalent) |
|---|---|---|---|---|
| control | 62.0% | 39.8% | **25.7%** | 22.8% |
| treatment | 77.2% | 49.7% | **35.1%** | 31.0% |

Both arms' `both` column ticked up a few points against their 146-wide predecessors, and the
treatment again beats the control on this measure — the same direction D65 found, now on a second,
independent pair of checkpoints. **Not read as "Option A worked."** The mechanism it was built to
test — commitment triggered by the streak signal — did not appear on either checkpoint, so this
modest gain cannot be attributed to that mechanism specifically; it is at least as plausible that
17 extra input dimensions gave the value/policy heads marginally more capacity for reasons
unrelated to the hypothesis, or that this is within the noise a single seed already carries (D64's
own four snapshots swung by 36 points on this same measure). What is not in question: the streak
feature has not, so far, taught either policy to camp.

**Evidence.** `runs/checkpoints/lstm_balance_d67_control_s{1,2,3,4}`,
`lstm_balance_d67_treatment_s{1,2,3,4}` (`.zip`/`.json`), streak analysis and `compare.py` output
both reproduced this session, artefacts in `runs/d67_paired_comparison/`. Ladder rungs `14a`–`14d`,
`15a`–`15d`. `docs/project/ITERATION_LEDGER.md`.

---

## D68 — D47 run for the first time: no reward candidate selected, escalated

**Status:** `MEASURED` (2026-09-10) — D47's rule executed against real, D62-screened, D61-selected
checkpoints for the first time since it was written (D47, 2026-09-05). Three decision-cycles
(D62, D66, D67) shipped ahead of this one; nothing blocked it, it simply had not been run.

### Eligibility, verified fresh this session

D47 is gated behind D62 (only a screened candidate may be considered). Rather than trust an
unrecorded prior claim, `python -m rfenv.reward_gate --rewards reward_balance,reward_balance_improved`
was re-run this session:

| candidate | round_robin | recency | camper | apfeld_active_rfs | sep | seeds | camper margin | verdict |
|---|---|---|---|---|---|---|---|---|
| `reward_balance` | 217.9 | 276.9 | −414.1 | 278.7 | 3.0σ | 8/8 | 7.3σ | **PASS** |
| `reward_balance_improved` | 216.6 | 280.0 | −388.4 | 276.6 | 2.3σ | 8/8 | 5.6σ | **PASS** |

Both eligible. No other candidate in `REWARDS` passes (`greedy`/`explore`/`weighted` all fail D62,
per D62/D57's own record) and no other candidate has a trained checkpoint regardless.

### The rule, applied

D47: the candidate beating round-robin on **both** headline metrics on the largest fraction of
paired episodes, via `compare.paired_wins()`; within 5 percentage points, no candidate is
selected. One D61-selected checkpoint stands in for each reward — the same checkpoints D67
retrained and the same `compare.py` run already on record (`runs/d67_paired_comparison/`), paired
against **round-robin specifically** (D47's own reference, not rung 5 — see D47's own reasoning for
why the weak floor is the right denominator here):

| candidate | checkpoint | ratio | cTTI | **both** |
|---|---|---|---|---|
| `reward_balance` | `lstm_balance_d67_control_s3` (300k) | 86.5% | 69.0% | **65.5%** |
| `reward_balance_improved` | `lstm_balance_d67_treatment_s1` (100k) | 90.6% | 69.6% | **64.3%** |

**Gap: 1.2 percentage points. Under the 5 pp margin. No candidate is selected.**

### What this means, plainly

This is not a null result in the sense of "the measurement failed" — it is D47's own rule doing
exactly what its own docstring says it would: "if two candidates land within 5 percentage points
of each other, no candidate is selected: both are reported and the choice is escalated." The rule
was written knowing this could happen (D47's own "weakness 1": insensitivity to margin) and chose
to accept it rather than force a pick. Both candidates clear round-robin decisively (65.5%/64.3%
against round-robin's own unpaired position) and both clear rung 5, the actual bar (D64/D65/D67).
Between them, D47 declines to choose.

**Escalated, per the rule's own text, rather than broken by a tie-break invented here.** The human
decision this surfaces: whether to (a) pick one anyway on a rationale D47 was deliberately built
not to encode (e.g. `reward_balance_improved`'s consistent edge on the recency-paired headline
across two independent checkpoint pairs, D65 and D67 — a real pattern, still built on single-seed
training each time), (b) run D47 again on matched multi-seed checkpoints in case the margin
sharpens past 5 pp with less noise, or (c) accept "no selection" as the answer and carry both
forward.

**Evidence.** `python -m rfenv.reward_gate --rewards reward_balance,reward_balance_improved`,
this session. `runs/d67_paired_comparison/comparison.md`'s "paired against round_robin" table,
same run D67 already recorded. No new training, no new comparison run — D47 applied to existing,
already-verified numbers.

### Resolved with matched seeds (2026-09-11): `reward_balance` selected, decisively

The escalation above chose option (b): 2 more seeds trained per arm (4 runs total, one at a time,
strictly sequential after a laptop crash earlier this session lost 4 parallel runs), then D61 and
D47 re-applied on the fuller 3-seed set. Both training runs and both re-applications ran this
session.

**D61, re-run across all 12 checkpoints per arm** (`python -m rfenv.selection`, 12 validation
configs × 3 seeds, net dominance against `recency`):

| checkpoint | steps | dominates | dominated | net |
|---|---|---|---|---|
| `lstm_balance_d67_control_seed2_s3` (rung 17c) | 300k | 44.4% | 8.3% | **+36.1% — selected** |
| `lstm_balance_d67_control_s3` (rung 14c, the single-seed pick) | 300k | 36.1% | 11.1% | +25.0% |
| `lstm_balance_d67_control_seed2_s2` | 200k | 38.9% | 16.7% | +22.2% |
| `lstm_balance_d67_control_seed2_s1` | 100k | 36.1% | 13.9% | +22.2% |
| (8 more control checkpoints, all lower) | | | | +16.7% down to −13.9% |

**The control arm's selected checkpoint changed** — seed 2's 300k snapshot displaces seed 0's,
registered as new rung **17c**. Rung 14c's docstring is annotated in place rather than deleted, so
the ladder still shows what was believed before more seeds existed.

| checkpoint | steps | dominates | dominated | net |
|---|---|---|---|---|
| `lstm_balance_d67_treatment_s1` (rung 15a, unchanged) | 100k | 38.9% | 5.6% | **+33.3% — selected** |
| `lstm_balance_d67_treatment_s2` | 200k | 38.9% | 13.9% | +25.0% |
| `lstm_balance_d67_treatment_seed1_s1` | 100k | 41.7% | 19.4% | +22.2% |
| (9 more treatment checkpoints, all lower) | | | | +19.4% down to −16.7% |

**The treatment arm's selection did not change** — the fuller 3-seed comparison re-picked the
exact same checkpoint (rung 15a) that the single-seed run had already chosen.

**D47, re-applied on the new pair** (`python -m rfenv.compare --rungs
round_robin,recency,lstm_balance_d67_control_seed2_300k_400k,lstm_balance_d67_treatment_100k_400k
--seeds 3 --sampled 10 --figures`, 57 configs × 3 seeds = 171 episodes/rung, artefacts in
`runs/d68_rerun_paired_comparison/`), paired against round-robin:

| candidate | checkpoint | ratio | cTTI | **both** |
|---|---|---|---|---|
| `reward_balance` | `lstm_balance_d67_control_seed2_s3` (rung 17c) | 89.5% | 85.4% | **81.9%** |
| `reward_balance_improved` | `lstm_balance_d67_treatment_s1` (rung 15a) | 90.6% | 69.6% | **64.3%** |

**Gap: 17.6 percentage points. Far outside the 5 pp margin. `reward_balance` is selected.**

Same ranking against rung 5 (`recency`, the actual bar): `reward_balance` 54.4% both, versus
`reward_balance_improved`'s 35.1%.

**What changed, and what didn't.** `reward_balance_improved`'s number is identical to D68's
original measurement (64.3%) — its selected checkpoint never changed, so there was nothing to
re-measure. `reward_balance`'s number moved from 65.5% to 81.9% because D61 found a materially
better checkpoint once seeds 1 and 2 gave it more to compare against — the single-seed selection
had picked a checkpoint 16.4 points weaker on this exact headline than one two more seeds of
training happened to produce. **This is the opposite of noise washing the gap out; it is noise
in the original single-seed selection being corrected by having more to select from.** D68's own
prediction was symmetric — either candidate's number could have moved — and this session did not
know which way it would break before running it.

**Decision: `reward_balance` is the selected reward candidate.** `reward_balance_improved` is not
discarded — it remains registered, passes D62, and stays available — but it is no longer carried
forward as a co-equal candidate. Future RL work in this repository trains on `reward_balance`
unless a new decision reopens the question.

**Evidence.** `python -m rfenv.selection` against both arms' full 12-checkpoint sets, this
session. `python -m rfenv.compare --rungs round_robin,recency,lstm_balance_d67_control_seed2_300k_400k,lstm_balance_d67_treatment_100k_400k
--seeds 3 --sampled 10 --figures`, this session, artefacts in `runs/d68_rerun_paired_comparison/`.
The 16 new checkpoints (2 arms × 2 seeds × 4 snapshots) are registered as ladder rungs 16a-16d
(control seed 1), 17a-17d (control seed 2), 18a-18d (treatment seed 1), 19a-19d (treatment seed 2).

## D69 — hand-rolled policies outside `baselines/` are now a structural test failure, not a review catch

D56 (`step % N_BANDS` scored and called `round_robin`) was the third time a policy built by hand
instead of through `baselines.make()` inverted a conclusion — D36 (a scan replay stood in for a
scenario), D43 (rungs 2 and 3 were accidentally the same policy), D56. Each was caught by a human
reading a diff, not by anything the suite would refuse to let happen again. `reward_gate.py`
already carries the fix for its own instance (`score_rung` calls `B.make(key, seed=seed)`, with a
docstring naming D56 directly) and `compare.py` and `selection.py` were checked this session and
already go through `B.make`/`baselines.make` everywhere they construct a policy — the code itself
had no live instance of the bug left. What was missing was the guard against a fourth recurrence.

**`tests/test_reward_gate.py::test_no_module_outside_baselines_hand_rolls_a_band_cycle`** parses
every top-level `rfenv/*.py` module with `ast` — deliberately not `rfenv/baselines/`, where
constructing a policy is the point, and not `rfenv/rl/`, which trains one rather than hand-writing
one — and fails if any of them contains `... % N_BANDS` as an AST node (a `BinOp` with `ast.Mod`
and `N_BANDS` on the right), rather than as a text grep that would also trip on the sentence
describing the bug in a docstring. Passes clean on the current tree; exists to fail loudly the next
time someone reaches for `step % N_BANDS` instead of `baselines.make("round_robin", ...)`.

**A second, unrelated gap closed in the same pass.** `tests/test_split.py` imported
`rfenv.rl.common` (which hard-imports `stable_baselines3`) at module level, so on a machine without
the training stack installed, `pytest tests -q` aborted at collection — nothing in the file ran,
not even the tests that need no training library at all. `tests/test_selection.py` was checked
against the same concern and did not have it: `rfenv.selection` only reaches `rfenv.baselines`,
whose one `rfenv.rl` dependency (`ladder.py::_make_deep_q_network`) is already lazy, imported
inside the function rather than at module level. Fixed in `test_split.py` by moving
`from rfenv.rl.common import make_train_env` into the one test that needs it
(`test_training_does_not_sample_the_evaluation_pool`), guarded by
`pytest.importorskip("stable_baselines3")` — the other five tests in the file now run on a clean
checkout with no training stack, and only that one test skips.

**Evidence.** `ast`-walked `rfenv/*.py` by hand this session (`grep` first, then the AST test, to
confirm the one docstring mention of `step % N_BANDS` in `reward_gate.py` does not trip a text-based
check) — no live instance of the bug found; the fix is prophylactic, not a correction to a wrong
number in any table. `venv/Scripts/python.exe -m pytest tests/test_reward_gate.py tests/test_split.py
tests/test_selection.py -q` → 21 passed.

---

## D71 — PulseWidth, AoA and per-band amplitude enter the observation, as an opt-in "v2" layout (D30 resolved)

**Status:** `SETTLED` (2026-09-14) — approved directly on the case D30 had already built; the
formal trigger D30 set for itself (a trained agent measurably failing to explore for want of these
features) was never separately measured before this was approved. Two of D30's three "honest
limits" were sidestepped rather than solved — see "What this does not give" below — so this closes
D30's question without fully claiming its hardest part.

**What changed, in one line.** `rfenv/scenario.py` now reads the two PDW columns it always silently
dropped (PulseWidth, AoA); `ScanEnv` can build either the original 183-wide observation ("v1",
still the default — every checkpoint on disk before today stays exactly as loadable as it was) or a
326-wide one ("v2") that adds them, selected per-instance via `ScanEnv(obs_version=...)`. Nothing
about "v1" changed by "v2" existing — verified byte-for-byte, see Evidence.

**Why now, not deferred further.** The user asked directly for it, having already read D30's own
brief (the "PDW completeness" planning pass, same session) and the mechanical/policy reasons literal
truth-side `C` cannot enter the observation (D29/D34, unaffected by this). PulseWidth and AoA are a
different case from `C`: both are receiver-measurable quantities on a pulse the receiver actually
detected — a real ES receiver's PDW output includes exactly these fields on every detection — so
gating them on `Y` (declared hit) keeps them inside D29's boundary the same way `measured_dbm`
already is, rather than reopening it.

**The layout is modular, not a replacement (the user's own request).** `ScanEnv._observation_blocks()`
returns every block by name in a dict; `OBS_LAYOUTS["v1"|"v2"]` (`rfenv/env.py`) selects and orders
a subset into the flat vector SB3 actually sees, and `observation_space` is built from the same
table so the two can never disagree. This is why "v1" staying exactly 183-wide, in exactly its old
order, was checkable rather than asserted — see the regression test below.

**Per layer:**

- **L0 (`scenario.py`).** `_bucket()` now groups pulses by `np.lexsort((-amplitude, slot))` instead
  of a plain stable sort — primary key slot, secondary key amplitude descending — so the first row
  of each (band, slot) group is always the loudest pulse, and `peak_dbm`, the new `pulse_width_us`
  and the new `aoa_deg` are read off that one row together: one real pulse's three readings, never
  a blend of several. Fully vectorised, no per-group Python loop. Cache bumped to `_v2.npz` so a
  pre-D30 cache is simply rebuilt, not read into a `KeyError`.
- **L1 (`truth.py`).** New `TruthGrid.PW`/`.AOA` grids, resolved the same way `S` already is: within
  `from_scenario`'s per-contribution loop, PW/AOA at a cell are overwritten only when this
  contribution's own peak beats the *running* max there — so whichever contribution ends up winning
  `S` (loudest overall) is provably the one PW/AOA end up reading, regardless of contribution order
  (tested both orderings).
- **L2 (`receiver.py`).** `DwellResult` carries `pulse_width_us`/`aoa_deg` unconditionally, truth-side,
  same as `Z`/`C` already do — `env.py`, not `receiver.py`, is where the `Y`-gate is applied.
- **L3 (`env.py`).** Four new blocks, all "v2"-only: `measured_dbm_band` (amplitude upgraded from one
  global last-dwell scalar to a per-band block that persists across the episode, same rule
  `hit_streak` set in D67 — the global scalar's real weakness, that it forgets every band but the
  one just left, is why this exists); `pulse_width` (clamped to [0, 200] µs — measured this session
  against 6 real train recordings: min 0.007, max 220.0, p99 102.3 — and rescaled to [0, 1]); `aoa_sin`/
  `aoa_cos` (AoA is circular, so it is stored as `(sin θ, cos θ)`, each rescaled `(x+1)/2` into the
  box's [0, 1] convention rather than as a raw angle that would teach the network 359° and 1° are
  opposites). All three of the new per-band blocks are gated on `Y`: within a multi-slot dwell with
  more than one declared hit, the loudest slot is the representative reading, mirroring how L1
  already resolves multiple *emitters* sharing one cell.

**A property that fell out of the encoding, not a separate mechanism.** `(0.5, 0.5)` in the stored
`aoa_sin`/`aoa_cos` decodes back to raw `(0, 0)`, which sits at the centre of the unit circle — a
point no real bearing measurement (always on the circle's edge, `sin²+cos²=1`) can ever produce.
That makes it a clean, provably-unreachable "never measured" flag, for free, rather than a magic
number chosen and hoped never to collide with a real reading. Tested directly (see Evidence).

**What this does not give the agent.** D30's own measurement found AoA's real value is
distinguishing an already-seen emitter from a new one in a crowded band — and that needs comparing
a bearing against a *set* of bearings already seen in that band this episode (clustering), which is
still not built. What ships here is one raw reading per band: "which direction did the last hit on
this band come from." D30's second honest limit (the ceiling-vs-achievable gap: whole-episode
medians against true labels, falling from 96.7% to 86.1% between sparse and crowded scenarios) is
therefore untested by this change, because nothing here does per-emitter attribution yet. Its third
limit (moving emitters, up to σ 122.6° of bearing drift) is structurally sidestepped rather than
solved: a single last-reading-per-band feature has no notion of drift to get wrong, because it never
compares two readings against each other in the first place. Whether the raw reading alone is useful
is untested — nothing has trained on "v2" long enough to say.

**Checkpoint housekeeping that came with it, same session.** `runs/checkpoints/` held 150 tracked
files flat in one directory — unnavigable, and every one of them implicitly "v1" now that the
concept exists. Reorganised (`git mv`, history-preserving) into `runs/checkpoints/v1/<model_name>/`,
grouped by stripping trailing `_sN`/`_NNN_steps`/`_NNNk`/`_NNNM` snapshot suffixes (150 files → 23
folders); future "v2" training lands in the sibling `runs/checkpoints/v2/<model_name>/`. Every
reference updated: `ladder.py`'s 56 checkpoint-path literals, the three `DEFAULT_CHECKPOINT`
constants, `doctor.py`'s checkpoint counter (`glob("*.zip")` → `glob("**/*.zip")`, or it would have
silently reported zero checkpoints after the move). `--obs-version {v1,v2}` is now a shared CLI flag
(`add_manifest_arguments`, `rfenv/rl/common.py`) on all three trainers.

**A second, unrelated gap this surfaced and fixed.** `require_loadable()` (`common.py`) — the check
every `load_checkpoint()` runs before trusting a checkpoint's shape — compared a checkpoint's
recorded width against a single hardcoded "current" value, always "v1"'s 183. A "v2" checkpoint,
correctly self-describing itself as 326-wide in its own manifest, would have failed this check the
moment anyone tried to load it back — "cannot be used," even though it is perfectly loadable under
`obs_version="v2"`. Fixed: `known_observation_widths()` returns every width `OBS_LAYOUTS` currently
defines, and the check now accepts a recorded width matching *any* of them, not only one.

**Evidence.** PulseWidth/AoA ranges measured this session against 6 real files under
`data/turing/stare/train_stare/` (`h5py`, direct column read, confirmed against
`metadata/feature_names`). New test file `tests/test_observation_d30.py`, 15 tests: layout widths
(183/v1, 326/v2), unknown-version rejection, v1/v2 block-for-block agreement on every shared block
run on an identical seeded episode, `Y`-gating (a confirmed real miss leaves PW/AoA untouched, a
confirmed real hit writes a genuine unit-circle bearing), per-band amplitude persistence against the
global scalar's forgetting, `TruthGrid.PW`/`.AOA` following the louder contribution under both
orderings, and `EmitterContribution`'s backward-compatible zero-default for every pre-D30 hand-built
contribution in the existing test suite. Full suite after: 405 non-RL tests + 27 RL tests +
these 15, all passing; `scripts/doctor.py` clean (checkpoints on disk: 84, unchanged by the reorg).
`venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/test_rl.py` and
`tests/test_rl.py`/`tests/test_observation_d30.py` separately, this session.

**Not yet done, deliberately out of scope here.** No checkpoint has trained on "v2" long enough to
report a result — a run (`lstm_balance_v2_control_seed2`, `reward_balance`, seed 2, 400k timesteps,
otherwise an exact mirror of rung 17c's own training config) started this session and is not yet
finished as this entry is written. The per-emitter bearing-clustering feature D30 actually measured
the value of is not designed, let alone built. Neither is scoped by this entry; both are separate,
future decisions if the raw "v2" reading turns out to be worth building on.

## D72 — `pulse_count` enters the "v2" observation, gated by `Y`: 326 → 362

**Status:** `SETTLED` (2026-09-14) — approved directly on the user's own request, made twice in
the same session. The first ask was literal `C` in the observation; that request had already been
measured and declined the same session, in writing, by
`docs/project/PDW_COMPLETENESS_AND_BAND_DENSITY_BRIEF.md` §1 — "mechanically, the current receiver
has no process that could produce `C`," because `Receiver.dwell()` collapses every pulse landing in
a cell into one combined draw before any threshold test runs, and `C` itself is built with **no
gamma gate at all**, counting contributions that would never individually cross the threshold. The
user's second ask — gate it by `Y`, the same rule D30 already applies to PulseWidth/AoA, on the
argument that a real receiver's own PDW stream does carry a count of the detections it resolved —
is a materially different request from the first and is what this entry approves. It does not
overturn the brief's mechanical finding; it narrows the gap the brief itself left open (a real
receiver counts detections it actually made, and `Y`-gating restricts the feature to cells the
receiver actually declared) without closing it (see "What this does not fix," below).

**What changed, in one line.** `rfenv/env.py`'s "v2" layout gains a sixth per-band block,
`pulse_count`: on a declared hit, the loudest slot's `dwell.C` (already computed, already flowing
through `DwellResult` — no L0/L1/L2 change needed) is stored per band, persisting until the next
declared hit on that band, normalised `log1p(C) / log1p(64)` then clipped to `[0, 1]`. "v2" moves
from 326 to 362 wide; "v1" is untouched — same append-only discipline D67 set and D30 kept, block
appended after `aoa_cos` rather than inserted among D30's four, so no existing "v2" offset moves
either. **Every checkpoint trained on the pre-D72 326-wide "v2" is invalidated by this, the same way
D49/D55/D67 invalidated their predecessors — and unlike those, this one has a real casualty.**
`runs/checkpoints/v2/lstm_balance_v2_control_seed2/` holds three snapshots (`_s1`/`_s2`/`_s3.zip`,
manifests recording `observation_width: 326`) from D71's own retrain, at 385,024 of its planned
400,000 timesteps per `train.log` — not finished, no training process currently running, and now
permanently unloadable regardless. Nothing else on disk trained on "v2" yet, so this is the full
extent of the cost.

**Why gating narrows the gap but does not close it.** `truth.py` builds `C[b,t]` from every
contributing emitter's raw pulse count landing in that cell, summed with `np.add.at`, independent of
whether any one contribution's amplitude would itself cross `gamma`. A declared hit (`Y=1`) means
the *combined* signal `S` crossed the threshold — it does not mean every pulse `C` counts did. So
`pulse_count` on a hit can still include sub-threshold co-located emitters folded into the number a
real receiver, resolving individual detections, would not have logged. This is the same "honest
limit" pattern D30/D71 already carries for AoA (a raw last-reading, not the clustering feature that
would make it truly useful) — the feature ships narrower than its ideal form, with the gap recorded
rather than hidden. The only way to close it fully is the sub-slot pulse-simulation receiver named
in D28 and re-named in the brief as "v2, after the gates" — a materially larger, separately-scoped
receiver-model change, not proposed here.

**Reuses an existing constant on purpose.** `_DENSITY_REF_PULSES = 64.0` already normalises `C` on
the reward side (`reward_balance_improved`'s `log1p(C)/log1p(64)` weighting, D57). `pulse_count`
reuses the identical transform so the observation and the one reward that already reads `C` agree on
what "busy" means, rather than inventing a second reference for the same underlying quantity. The
reward's use is unclipped (its weight can legitimately exceed 1); the observation's is clipped to
`[0, 1]` because the box cannot be violated and raw `C` is recorded (D34) as spanning up to 4,543
within a single episode.

**Evidence.** `tests/test_observation_d30.py` extended: `test_v2_is_362_wide` (was
`test_v2_is_326_wide`), `pulse_count` added to the never-measured-sentinel check (reads 0.0, since
`log1p(0) = 0`), the Y-gating miss/hit tests (a confirmed miss leaves `pulse_count` untouched, a
confirmed hit writes `pulse_count > 0`, since a declared hit implies `C >= 1`), and a dedicated
`test_pulse_count_is_log1p_normalised_and_clipped_to_the_box` pinning the transform at the reference
value, far past it (clipped to 1.0, not left free), and at zero. `tests/test_observation_d30.py`
and `tests/test_env.py` together: 51 passed. Full non-RL suite
(`venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/test_rl.py`): 421 passed, 162 skipped,
1 failed — the one failure (`test_heldout_split_is_refused_without_an_explicit_flag`) is a
pre-existing, unrelated data-availability gap (`list_configs("scan", "test")` returns 0, not 45 —
the held-out test split's files are not present on this machine), not touched by this change.

**Not done here, deliberately.** No checkpoint has trained on the 362-wide layout. `BAND_POWER`,
the brief's own proposed deployable density proxy, remains unbuilt and is not superseded by this —
they answer different parts of the same underlying goal and are not mutually exclusive. The
sub-slot receiver redesign that would close the remaining gap stays named, not scoped.

## D73 — the first result on the widened "v2" observation (D72), paired `reward_balance` vs `reward_balance_improved_v2`

**Status:** `MEASURED` (2026-09-18) — a single-seed measurement, reported as such; not a selection
decision (D47/D61 are not re-run here, since each reward has exactly one trained candidate at this
width, not several snapshots to choose between).

**What ran.** Two RecurrentPPO checkpoints, both `ScanEnv(obs_version="v2")` (362-wide, D72), both
the D60 training split, `ent_coef=0.01 gamma=0.997 n_steps=8192`, seed 2 — the same pairing
discipline D65 used, mirroring rung 17c/D71's own config:

- `lstm_balance_v2_d72_seed2` (rung 20d) — `reward_balance`, 401,408 steps, one uninterrupted run.
- `lstm_balance_improved_v2_d72_seed2` (rung 21a) — `reward_balance_improved_v2`, 404,800 steps.
  **Interrupted by two separate laptop crashes**, both times with zero progress lost beyond the
  last `--checkpoint-freq 50000` snapshot: `RecurrentPPO.load(...)` on the last `_sN.zip` plus
  `model.learn(reset_num_timesteps=False)` picked up exactly where the process died, rather than
  restarting from step 0. Recorded in the checkpoint's own manifest description, not only here.

Both registered in `rfenv/baselines/ladder.py` (rungs 20d, 21a) beside the now-permanently-dead
326-wide seed-2 pair D71 started and D72's width bump killed mid-training (rungs 20a–20c, three
snapshots, 385,024/400,000 steps, never finished).

**The comparison.** `python -m rfenv.compare --rungs round_robin,recency,lstm_balance_v2_d72_seed2_400k,lstm_balance_improved_v2_d72_seed2_400k --seeds 3 --sampled 10 --figures --obs-version v2 --out runs/d72_paired_comparison`
— 4 rungs × 57 scenarios (47 stare + 10 sampled, D60's whole-development-set rule) × 3 seeds =
**684 episodes**, same scale as D65's own headline run:

| # | scheduler | interception ratio | censored intercept time (s) | emitter coverage | beats recency on **both** |
|---|---|---|---|---|---|
| 2 | round_robin | 0.060 | 4.18 | 0.865 | 3.5% |
| 5 | recency | 0.111 | 3.34 | 0.887 | — |
| 20d | Recurrent PPO, D72 v2 (`reward_balance`, 400k) | 0.111 | 3.30 | 0.902 | 36.3% |
| 21a | Recurrent PPO, D72 v2 (`reward_balance_improved_v2`, 400k) | 0.136 | 2.82 | 0.912 | **45.6%** |

Paired against round_robin: recency 93.0%/70.2%/67.3%, rung 20d 89.5%/77.2%/72.5%, rung 21a
89.5%/83.6%/78.4% (ratio/cTTI/both). Full artefacts, both paired tables and figures:
`runs/d72_paired_comparison/`.

**Both checkpoints clear the bar.** Both beat recency on the joint metric far more often than
round_robin's 3.5% floor. **The treatment (`reward_balance_improved_v2`) is ahead of the control
(`reward_balance`) on every column** — ratio 0.136 vs 0.111, cTTI 2.82 s vs 3.30 s, coverage 0.912
vs 0.902, both 45.6% vs 36.3% — the same direction D65 found on "v1" between `reward_balance` and
`reward_balance_improved` (31.0% vs 22.8%).

**Not directly comparable to D64/D65's "v1" numbers, and not claimed to be.** Three things differ
at once: observation width (362 vs 183), reward formula (`reward_balance_improved_v2`'s
`_DENSITY_SHRINKAGE_V2 = 0.75` against `reward_balance_improved`'s 0.5), and the sampled-scenario
draws (this run's own `--sampled 10` draw, not verified identical to D65's). This entry measures
"v2, this session" against its own floor and bar, not "v2 beats v1."

**Not read as "the treatment wins," for the same reason D65 gave and did not resolve.** One
training seed per arm, and D64 already measured the control's own four checkpoints swinging by more
than the gap reported here (+25.0 to +36.1 net dominance across four snapshots of one run). This
result adds a second observation-width's worth of the same single-seed pattern; it does not settle
it. **Neither row is promoted as the number to carry forward.**

**Evidence.** Command and output above are this session's own; manifests for both checkpoints
(`runs/checkpoints/v2/lstm_balance_v2_d72_seed2/lstm_balance_v2_d72_seed2.json`,
`.../lstm_balance_improved_v2_d72_seed2/lstm_balance_improved_v2_d72_seed2.json`) record
`observation_width: 362`, `reward`, `seed: 2`, and `total_timesteps`. `runs/` is gitignored;
artefacts are reproducible from the command above, not committed.

## D74 — a band-priority reward, "v2p" (398-wide): built, validated, measured null at this scale

**Status:** `MEASURED` (2026-09-19, extended same day by a follow-up below) — a negative/null
result, recorded because a result that says "the agent isn't using this" is exactly as much a fact
as one that says it is. Write-up was deliberately held until both pieces of validation existed (the
control-arm comparison and a permutation ablation), on a direct instruction, rather than following
D71/D72's precedent of writing the design up as `SETTLED` the day it was built. **The follow-up
below re-ran the same test at a materially stronger setting (bigger coefficients, a new reward
term, double the network capacity, double the training budget) and found the same null result** —
this is not a single measurement at one arbitrary scale, it is the same answer twice.

### Why this was built

A fully-specified, library-free request, made directly this session: give the scheduler a
per-episode "pay more attention to these bands" signal and see whether a trained policy uses it. It
echoes the gap D70 named — *"Open loop strategies … may lose time to nonthreatening emitters by not
giving time to new or threatening ones"* — but is a **deliberately separate, synthetic successor to
D70, not a continuation of it**. D70 sourced its priority vector from an external threat
classification and had its own partial implementation (`rfenv/threat.py`, priority-reweighting in
`rl/common.py`) deleted uncommitted, unexplained, discovered and left unresolved earlier this
session when a related request was raised and then cancelled. This entry's `band_priority` has no
library behind it at all: values are synthetic, sampled fresh each episode from `ScanEnv`'s own
RNG, not read from any emitter-type table.

### What was built

- **A fourth observation layout, `OBS_LAYOUTS["v2p"]`** — "v2" (362-wide, D72) plus one more
  36-wide block, `band_priority`, appended last (398 total). `1.0` = ordinary (every band's
  default), `3.0` = elevated (3-6 of 36 bands, resampled every `reset()`). Not gated on `Y`, unlike
  every other "v2"-only block — nothing to gate, since it isn't something the receiver measures.
- **An additive reward term in `ScanEnv.step()`**, not a new `REWARDS` candidate and not touched by
  `reward_gate.py`'s D62 screen: `reward += priority_coef * band_priority[dwell.band] * len(newly)`
  (`priority_coef = 0.5`), applied after whichever registered reward already ran. **Gated on
  discovering something new, not on occupying the band** — a per-slot multiplier would reopen D53's
  camping exploit; this was a design constraint from the start, not something added after measuring
  a problem.
- **A control-arm setting, `priority_uniform`**, load-bearing for the whole validation: with it set,
  `band_priority` stays all-ones every episode, so the reward term still fires at the same scale but
  carries no differential information. Necessary because the term is not neutral at `p=1` vs
  switched off — it adds `priority_coef * len(newly)` to *every* discovery regardless of band — so a
  plain before/after comparison would have confounded "the signal helped" with "the reward got
  uniformly bigger."
- `compare.py` gained matching flags, later made per-rung (`Rung.band_priority`/`priority_uniform`/
  `priority_coef`/`priority_n_bands`, `resolve_priority_kwargs`) so a registered checkpoint's own
  training config is used automatically rather than depending on the caller passing matching CLI
  flags — closing a real, previously-silent risk (forgetting the flag would have scored a
  priority-trained checkpoint under an all-ones vector without raising). `--figures` also gained an
  automatic `priority_animation_config_*.gif`, highlighting the episode's elevated band(s) directly
  on the animated schedule, generated whenever a compared rung resolves to real (non-uniform)
  priority.

### What was measured

**Two RecurrentPPO checkpoints, matched pair**, `reward_balance`, D60 split, `ent_coef=0.01
gamma=0.997 n_steps=8192`, seed 2, `obs_version="v2p"` — same discipline as the D72 pair, only
`priority_uniform` differs:

- `lstm_balance_v2p_priority_seed2` (treatment, rung 22a) — 401,408 steps.
- `lstm_balance_v2p_uniform_seed2` (control, rung 22b) — 400,000 steps.

**The comparison** (`round_robin`/`recency` + each rung, 513 episodes apiece: 3 rungs × 57
scenarios × 3 seeds; reproduced afterward as one combined run via the per-rung fix above, identical
numbers both ways):

| rung | scheduler | ratio | cTTI (s) | coverage | beats recency on **both** |
|---|---|---|---|---|---|
| 2 | round_robin | 0.060 | 4.18–4.20 | 0.864–0.865 | — |
| 5 | recency | 0.109–0.111 | 3.34–3.36 | 0.887–0.891 | — |
| 22a — treatment | 0.120 | 3.28 | 0.891 | 44.4% |
| 22b — control | **0.128** | **2.70** | **0.908** | **61.4%** |

**The control beat the treatment on every column** — the opposite of what the feature is meant to
show, if it worked.

**Camping ruled out directly.** The obvious hypothesis — the treatment agent overcommits to its
priority bands — was checked, not assumed: 20 episodes, treatment checkpoint, mean fair-share
airtime on elevated bands **1.076** against **1.007** on ordinary ones (`1.0` = an equal cut) —
about 7% more, not camping, and inconsistent in direction episode to episode. Camping was already
structurally unlikely by construction (`reward_balance`'s own `-3.0 × visit_density × n_slots` term
prices concentrated airtime; the priority bonus only pays on a new discovery, never on occupancy),
and this measurement confirms it didn't happen anyway.

**Treatment is measurably more diffuse than control, across the whole spectrum, not just the
priority bands.** Same 20 episodes, entropy of the final `visit_density` distribution over all 36
bands: treatment **3.255** against control **3.051** (max possible `ln 36 = 3.584`); bands touched
meaningfully (`visit_density > 0.5`): treatment **24.95** against control **18.95**, out of 36.
Control converged onto a tighter, more concentrated sweep; treatment's is broader and less
committed — a real difference, plausibly because control's `band_priority` input is a *constant*
every episode (nothing to condition on, so nothing to make the training distribution harder), while
treatment's genuinely varies episode to episode, a harder problem to converge cleanly on on in the
same 400k-step budget. A constant input cannot itself be "read" for information — this is a
training-difficulty account, not "control learned to use the signal."

**Permutation ablation: the treatment agent does not read `band_priority` at all.** 30 episodes on
the trained treatment checkpoint, each run twice on the identical scenario/seed/receiver-noise draw
(a separate RNG shuffles `band_priority`, never `env.np_random`, so nothing else about the episode
differs) and the identical torch sampling seed for both runs of a pair — real `band_priority` in one,
the same values permuted across bands in the other:

| | corr(airtime, true priority) | ratio | cTTI (s) | coverage |
|---|---|---|---|---|
| real priority fed | +0.018 | 0.111 | 2.83 | 0.892 |
| shuffled priority fed | +0.019 | 0.114 | 2.58 | 0.903 |

Correlation between actual airtime and true priority is statistically indistinguishable whether the
signal is real or garbage; real beat shuffled in 13 of 30 episodes (43% — a coin flip); performance
does not degrade when the signal is corrupted, if anything the reverse by a margin small enough to
be noise. **The agent learned to ignore this block of the observation.**

### The finding

**Read together, not one measurement alone:** treatment's underperformance is not the agent
misusing a real signal (ruled out — it isn't reading the signal at all, ablation) and not the agent
overcommitting to a narrow set of bands (ruled out — camping check). The more diffuse, less-focused
policy it settled on (spread-check) is better explained by the harder, non-stationary training
distribution it faced than by anything about the priority mechanism itself. **At this scale
(`priority_coef = 0.5`, 3-6 of 36 bands, sb3-contrib's default LSTM capacity, 400k steps, one
seed), the band-priority reward has no measurable effect, positive or negative, through the
mechanism it was built to test** — it is simply not learned.

**This is not adopted as a working feature and is not promoted for further use as configured.**
Nothing in this repository defaults to it (`band_priority=False` everywhere it is a constructor
kwarg), so nothing else is affected by leaving it registered. The code is not removed — unlike
D66's phase-switch gate, which was a specific mechanism shown to be actively worse than the
baseline it was meant to improve on, this is a null result with plausible, named paths that were not
tried: a larger `priority_coef` or elevation multiplier (`_PRIORITY_HIGH`, currently `3.0`), more
training steps, or more network capacity (the LSTM hidden state, sb3-contrib's default 256,
untouched by this entry). None of those is scoped or recommended here — a genuinely new experiment,
not a revision of this one's numbers, matching the discipline D66's own "neither threshold was
retuned after seeing this" held to.

### What this does and does not close

Answers the narrow question it was built to answer: a synthetic, discovery-gated, additive priority
reward at this specific scale does not get learned in 400k steps against `reward_balance`. It does
not touch `REWARDS` or `reward_gate.py`'s D62 screen (nothing there changed). It does not resolve
D70's own open items (threat-split `EVALUATION.md` §4 reporting remains unadopted; the `p` sampling
distribution question D70 raised for its own, library-sourced retrain is untouched by this entry,
which used a materially different, uniform-random sampling scheme by construction) — this entry
answers a related but distinct question with a different mechanism, not D70's own retrain. Whether
a *stronger* version of this same mechanism (bigger coefficient, more capacity, more steps) would be
learned and would help is open, not closed by a null result at one configuration.

**Evidence.** `rfenv/env.py` (`OBS_LAYOUTS["v2p"]`, the reward term, `_PRIORITY_HIGH`),
`rfenv/rl/common.py`/`ppo.py`/`recurrent_ppo.py`/`_cli.py` (CLI wiring, `device` support added the
same session), `rfenv/compare.py` (`resolve_priority_kwargs`, priority-animation), `rfenv/baselines/
ladder.py` (rungs 22a/22b), `tests/test_band_priority.py` (17 tests). Comparison commands and
manifests: `runs/d74_treatment_comparison/`, `runs/d74_control_comparison/`
(`lstm_balance_v2p_priority_seed2.json`/`lstm_balance_v2p_uniform_seed2.json` record
`observation_width: 398`, `band_priority`/`priority_uniform`, seed, `total_timesteps`). Camping,
spread and permutation-ablation diagnostics were scratch scripts, not committed tests (not
committed-repeatable artefacts the way the compare.py run is) — run this session, reported here in
full with method and numbers rather than only a conclusion, per this repository's own provenance
rules. `runs/` is gitignored; the compare.py artefacts are reproducible from the commands in the
docs and `scratch/TRAINING_JOURNEY.md` §18; the diagnostic scripts themselves were not saved to the
repository.

### Follow-up (2026-09-19, same day): stronger incentive, more capacity, same null result

Requested directly, in response to the null result above: turn up the discovery bonus, add a genuine
per-slot "camp a little on the priority band" incentive, and give the retrain more capacity and more
steps — testing whether the null result was a signal-strength or a capacity/budget limitation rather
than the mechanism itself.

**What changed in the code.** A new function, `priority_reward_bonus` (`rfenv/env.py`) — deliberately
a separate function, not a change to `reward_balance` or any `REWARDS` entry, same discipline the
original term held to. Two terms now, both added to whichever reward is selected:

- `discovery = priority_coef * band_priority_value * n_newly` — unchanged formula, `priority_coef`
  raised `0.5 → 2.0` and the elevated value itself now configurable (`priority_high`, a new `ScanEnv`
  kwarg; was the hardcoded module constant `_PRIORITY_HIGH`), raised `3.0 → 5.0` for this run.
- `occupancy = occupancy_coef * band_priority_value * decay * n_slots`, new — a genuine per-slot
  occupancy bonus, requested directly despite the D53 risk this reopens in spirit. `decay = max(0, 1
  - visit_density_value / occupancy_decay_cap)`, anchored to **`visit_density`** (D55's own
  anti-camping fix for `reward_balance`'s camping cost), not a resettable consecutive-dwell streak —
  the mechanism D53's exploit actually depended on. A two-band ping-pong drives both bands'
  `visit_density` up over the whole episode regardless of what ran in between, decaying the bonus on
  both exactly as sustained camping would; checked directly, not assumed
  (`tests/test_band_priority.py::test_ping_pong_between_two_elevated_bands_does_not_out_earn_a_full_sweep`).
  **Both terms scale by `band_priority_value` directly, not a binary elevated/ordinary gate** — required
  so the control arm's constant `1.0` still fires both terms at the same uniform scale, preserving the
  "same reward-scale, only the differential differs" comparison this whole methodology depends on; a
  binary gate would have made the occupancy term fire only for treatment, never control, confounding
  the comparison from the start. `occupancy_coef = 0.0` (`ScanEnv`'s default) reproduces this entry's
  own original recorded runs exactly — verified with a dedicated regression test, not assumed.

A latent bug surfaced and fixed while wiring `priority_high` through: `observation_space`'s declared
`band_priority` ceiling was hardcoded to `_BLOCK_SPECS`'s static `3.0`, so `priority_high > 3.0`
produced real episodes whose values fell outside the space the env itself declared —
`gymnasium.utils.env_checker.check_env` catches this as a contract violation, and would have done so
silently for any SB3 training run that never called `check_env` first. Fixed: the declared ceiling for
this one block now tracks `self._priority_high`. Regression test added.

`Rung` (`ladder.py`) gained `priority_high`/`occupancy_coef`/`occupancy_decay_cap` fields alongside
the existing priority fields, resolved by `compare.py`'s `resolve_priority_kwargs` the same automatic,
per-rung way as before.

**Training config.** LSTM hidden size doubled (256 → 512, `policy_kwargs={"lstm_hidden_size": 512}`,
confirmed on the loaded checkpoint, not just the training flag); timesteps doubled (400k → 800k).
Same matched-pair discipline: `lstm_balance_v2p_priority_strong_seed2` (treatment, rung 23a,
802,816 steps) and `lstm_balance_v2p_uniform_strong_seed2` (control, rung 23b, 800,000 steps), same
split/seed/base hyperparameters as every other pair in this lineage.

**The comparison** (round_robin/recency + each rung, 513 episodes apiece, same design as before):

| rung | scheduler | ratio | cTTI (s) | coverage | beats recency on **both** |
|---|---|---|---|---|---|
| 23a — treatment (strengthened) | **0.154** | **2.43** | 0.907 | **73.1%** |
| 23b — control (strengthened) | 0.121 | 2.74 | **0.911** | 45.6% |

**Treatment beat control by a wide margin — the opposite direction from this entry's original pair**
(where control led, 61.4% vs 44.4%). Taken alone, this would look like the mechanism finally working.
23a's 73.1% is also the highest "beats recency on both" figure measured anywhere in this project to
date, ahead of D68's settled best (17c, 54.4%) — though, as below, not for the reason that comparison
would suggest.

**A second permutation ablation, on the strengthened treatment checkpoint, says otherwise.** Same
method as the original (30 episodes, each run twice on the identical scenario/seed/receiver-noise
draw and the identical torch sampling seed, real `band_priority` in one run of each pair and the same
values shuffled across bands in the other):

| | corr(airtime, true priority) | ratio | cTTI (s) | coverage |
|---|---|---|---|---|
| real priority fed | +0.031 | 0.145 | 1.80 | 0.960 |
| shuffled priority fed | +0.031 | 0.148 | 1.82 | 0.929 |

Correlation is identical to three decimal places whether the signal is real or garbage, real beat
shuffled in 12 of 30 episodes (40% — a coin flip, if anything below even odds), and performance does
not degrade when the signal is corrupted. Airtime on elevated bands is a little more tilted than the
original run's (1.129 vs 0.970 fair-share, ~16% more, against 1.076 vs 1.007 the first time) but the
ablation shows directly that this tilt does not track the true signal either — noise, not tracking.
**The agent still does not read `band_priority`**, at roughly 4x the discovery incentive, a new
occupancy term on top, double the network, and double the training budget.

**Conclusion: 23a's lead over 23b is not the priority mechanism working.** A checkpoint that has been
shown, directly, not to condition its behaviour on a signal cannot be beating another checkpoint
*because* of that signal. The far more likely explanation is plain single-seed training-run variance
— the same caveat every comparison in this lineage carries (D65, D73, this entry's own original
pair) — and the direction flipping completely between the two pairs (control ahead the first time,
treatment ahead the second) while the one thing actually measured both times, whether the agent reads
the signal, came back an unambiguous "no" on both occasions, is exactly the pattern that points at
noise rather than a real, reproducible effect. **What this follow-up does establish, and it is worth
keeping separate from the priority question:** a 512-wide LSTM trained for 800k steps outperforms
every 256-wide/300-400k-step checkpoint measured in this project so far, on both arms. Whether that
holds with `band_priority` removed entirely — the one comparison this follow-up does not contain,
since both 23a and 23b still carry the mechanism, just with real vs. uniform values — is untested and
open if capacity/budget is pursued as its own question, separate from priority.

**Not adopted, not promoted, code not removed — unchanged from this entry's original verdict.**
Nothing defaults to `occupancy_coef` above `0.0` or `priority_high` above `3.0`; every existing call
site is unaffected. The specific untried paths named in this entry's original verdict (bigger
coefficient, more capacity, more training) have now been tried, together, at a substantial multiple
of the original scale, and the mechanism still measures null. A materially different scale again, a
different architecture, or accepting that a synthetic per-episode priority signal may not be
learnable by this policy class in this environment at all are the remaining open directions — none
scoped or recommended here.

**Evidence.** `rfenv/env.py` (`priority_reward_bonus`, `priority_high`/`occupancy_coef`/
`occupancy_decay_cap` on `ScanEnv.__init__`, the `observation_space` ceiling fix),
`rfenv/rl/common.py`/`ppo.py`/`recurrent_ppo.py`/`dqn.py`/`_cli.py` (CLI wiring for the three new
flags), `rfenv/compare.py` (`resolve_priority_kwargs` extended to seven fields), `rfenv/baselines/
ladder.py` (rungs 23a/23b), `tests/test_band_priority.py` (extended to 26 tests: the occupancy
formula in isolation, backward compatibility at `occupancy_coef=0.0`, the D53 ping-pong check, and
the `observation_space` ceiling regression). Comparison commands and manifests:
`runs/d74_followup_treatment_comparison/`, `runs/d74_followup_control_comparison/`
(`lstm_balance_v2p_priority_strong_seed2.json`/`lstm_balance_v2p_uniform_strong_seed2.json` record
`observation_width: 398`, seed, `total_timesteps`, and the full `argv` including every new flag).
Both permutation-ablation runs (original and this follow-up) were scratch scripts, not committed
tests, reported here in full with method and numbers rather than only a conclusion, per this
repository's own provenance rules. `runs/` is gitignored; the compare.py artefacts are reproducible
from the commands in `scratch/TRAINING_JOURNEY.md` §18; the diagnostic scripts themselves were not
saved to the repository.
