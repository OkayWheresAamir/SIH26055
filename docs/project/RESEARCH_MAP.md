# Research Map

> Built from the documents actually present in `docs/` (and the dataset card in `data/turing/`)
> on 2026-08-28. **Updated the same day** when the official problem statement was added as
> `docs/project/SIH26055_PROBLEM_STATEMENT.md`; it outranks everything else here and settles several
> questions this map had listed as open. Every PDF below was opened and read with `pypdf` in that session; none was
> classified from its filename. Where a document's filename and its contents disagree, the
> contents decide.
>
> **Updated 2026-09-01** (eight documents) and **2026-09-04** (six more, including Teissier 2026,
> which this map had listed as wanted and absent). The 2026-09-04 additions were read with
> `pymupdf`, same rule: opened, not inferred. All 35 PDFs in `docs/` are now classified.
> `docsearch` (see `docsearch/README.md`) searches them and returns `file.pdf:p.7` citations —
> use it to check anything below against its page.

## How to read the Authority column

Per `CLAUDE_CODE_RESEARCH_PROTOCOL.md`: **A** = direct project/dataset facts; **B** = reputable
research/theory; **C** = proposed strategy; **D** = brainstorm/commentary. A document that
summarises another source is never stronger than the source it summarises, and is **D** when
that primary is not in this repository.

---

## Stage index

### The requirement itself
- `docs/project/SIH26055_PROBLEM_STATEMENT.md` — **Authority A, NOW, read before anything else.** The official DRDO problem statement. Defines the environment, the observation model, the training signal and the metric set. Every other document in this folder is a means to satisfying it.

### Turing / dataset understanding
- `data/turing/README.md` — the Hugging Face dataset card. **Partly contradicted by the files.** Use the HDF5 files, not this card.
- *(No PDF in `docs/` documents the Turing dataset. This is a real gap.)*

### Environment construction
- `docs/reference/scheduling/paperSSPD (1).pdf` — the SNR-time-series illumination model, which is what our data actually matches (see D4). **NOW.**
- `docs/reference/scheduling/me 2.0.pdf` §17–18 — the radar-state / physical-illumination / receiver-observation separation, and a module decomposition. **NOW**, for structure.
- `docs/reference/scheduling/optimumsearch.pdf` — sweep period / dwell time / duty cycle definitions and the coincidence model. **NOW.**
- `docs/reference/scheduling/118700Q.pdf` §4 — a concrete, minimal band-scheduling environment and its hit/miss observation rule. **NOW.**
- `docs/reference/background/TECHNICAL DIFFERENTIATION STRATEGY.pdf` §3, §4, §12 — a checklist of environment features. **LATER**, proposal only.

### Environment validation
- `docs/reference/scheduling/optimumsearch.pdf` ch. 2, 3, 6 — intercept-time and probability-of-intercept theory. Derivable equations we can check an environment against. **NOW.**
- `docs/reference/Interception_Model_...copy.pdf` §III–VI — **Teissier et al. 2026 (added 2026-09-04).** Intercept time and pulse interception ratio for a *randomly scanning* multichannel SHR against *frequency-agile* emitters with rotating antennas, validated in simulation. The second source of checkable equations, and the first covering frequency agility. **NOW.**

### Baselines
- `docs/reference/scheduling/paperSSPD (1).pdf` — **Apfeld et al. 2016. The strong non-learning baseline.** Adaptive band selection with autocorrelation-based scan-period estimation. Reimplementable from §II. **NOW.**
- `docs/reference/Interception_Model_...copy.pdf` §VII — Teissier's minimax-optimised random scan under parameter uncertainty. A *known-parameter* upper reference, not a cold-start method; useful as the "what if you knew the emitters" line on a comparison plot. **LATER.**
- `docs/reference/scheduling/optimumsearch.pdf` — periodic/probabilistic search strategies; Clarkson's strategy as the reference point. **NOW.**
- `docs/reference/scheduling/118700Q.pdf` §4 — "simple periodic search strategy" used as the bootstrap policy. **NOW.**
- `docs/reference/background/TECHNICAL DIFFERENTIATION STRATEGY.pdf` §2 — random / round-robin / fixed-priority / recency list. **LATER.**

### RL / adaptive scheduler
- `docs/reference/scheduling/118700Q.pdf` §2, §3 — PSR/TPSR + RPCA + multi-armed-bandit band selection. **LATER.**
- `docs/reference/background/TECHNICAL DIFFERENTIATION STRATEGY.pdf` §19 — RL framing, explicitly "only after strong baselines". **LATER.**

### Final evaluation
- `docs/reference/background/TECHNICAL DIFFERENTIATION STRATEGY.pdf` §13, §14, §27 — metric list and the "numerical claims not marketing claims" rule. **FINAL EVALUATION.**
- `docs/reference/scheduling/optimumsearch.pdf` ch. 6 — probability of intercept as a defined quantity. **FINAL EVALUATION.**

### Background / future
- `docs/reference/background/Electronic warfare (extended).pdf` — plain-language explanation of EW and the problem statement.
- `docs/reference/background/GPT BACKDOOR PAPER KHARKIV CONFRENCE.pdf` — synthesis of a conference proceedings not in this repository.
- `docs/reference/scheduling/Scanning Strategy Learning For Electronic Support Receivers by Robust Principal Component Analysis (3).pdf` — one-page summary of `docs/reference/scheduling/118700Q.pdf`.

### Unresolved

> Most of what was here on 2026-08-28 has since closed. Decisions and their evidence now live in
> `docs/project/DECISIONS.md`; this list holds only what genuinely remains.

**Needs a human decision (not more research):**
- **D30** — do `AoA` and `PulseWidth` enter the observation vector? They are measured PDW fields
  `rfenv/scenario.py` currently discards. Bearing-based attribution was measured at 96.7%
  (`config_2`) and 86.1% (`config_921`), and it is the only observable that separates a dwell
  finding a *new* emitter from one re-finding a known emitter — which is D14's camper pathology in
  information terms, and what would make D29's candidate 3 actionable rather than merely scorable.
  Costs L1 an angular dimension and the observation a fixed-width bearing encoding. **Not
  blocking** `receiver.py` or `env.py`, and the four gates do not touch the observation vector.
- **D33** — which cell population P<sub>d</sub> is averaged over. Three candidates measured
  (0.819 / 0.837 / 0.851 at the frozen γ); the previously quoted 0.822 is withdrawn.
  **`receiver.py` needs this to emit a ROC at all.**
- **D34** — ratify the base observation vector (36×3+1), which `ENVIRONMENT_SPEC.md` §L3 already
  fixes but which was never recorded as a decision. Build to it meanwhile.
- **D29's selection rule** — how to choose among the three reward candidates when they
  Pareto-dominate the baselines but not each other. Must be fixed **before** training, so the
  choice is not made after seeing results. Not blocking the environment build.
- **CPRIT / deinterleaving as a scheduler input — raised 2026-09-04, conflicts with D12 and D19.**
  `CPRIT(Combined PRI Transform).pdf` and `CPRITworkflow.pdf` propose a PRI-transform module
  between the receiver and the RL agent, on the argument that hit/miss bits alone cannot
  represent a frequency-agile, staggered-PRI emitter. Both are Authority C (proposed strategy),
  which is not new evidence, so **D12/D19 stand** and nothing has been built toward this. Its
  real cost is an added per-emitter attribution channel in the observation — the same class of
  change as D30, and larger. **A human decision, and not a blocking one.**
- **Is Teissier's PIR our interception ratio? — raised 2026-09-04.** The definitions look close
  but were not compared term by term this session. `EVALUATION.md` is the single authority on
  metrics; if they coincide, Teissier gives us a closed-form check on a metric we already
  report, which would be worth having. **Needs someone to read both definitions side by side.**
  Until then, treat them as different quantities that share a name.

> Closed since the 2026-08-28 list: **D4** (accepted by the team 2026-09-01 — the
> continuous-signal environment with derived binary occupancy) and **D5's sub-question** (do all
> hits score equally, or is first-interception worth more — it became D7 candidate 3, given a
> precise three-clause definition by D28 and recorded in D29).

**Needs domain review:**
- *"Approaches to intercept a periodic scan receiver optimally should be outlined"* (PS). Most likely the scan-on-scan problem. A written deliverable, owed to the evaluators; Köksal ch. 3 is the source. Needs someone with the domain reading to write it.

**Worth obtaining, not blocking:**
- ~~Teissier et al. (2026)~~ — **OBTAINED 2026-09-04.** Now
  `docs/reference/Interception_Model_of_Random_Scanning_Strategies_Against_Frequency-Agile_Radar_in_Electronic_Support_copy.pdf`
  and classified below (VALIDATION / BASELINE, Authority B). The claims previously carried here
  on the strength of a citation in `Electronic Support Scan Scheduling (1).pdf` can now be read
  at the primary.
- Clarkson (2005), *Optimal Periodic Sensor Scheduling in Electronic Support* — the dwell-time-allocation primary. Not in this repository.

### Closed since 2026-08-28
- **~~`sensitivity_dbm` semantics~~ — RESOLVED 2026-08-28.** The dataset paper (§II) states detection is probabilistic: ambient noise −100 dB, amplitude falling quadratically with distance, and *"the probability of pulse detection increases the more distinct the signal is from the noise floor."* There is no hard threshold, which is why we found no cliff at −110 or −120 and why 4% of pulses sit below the stated figure. `sensitivity_dbm` is a configuration value that does not gate the Amplitude column. Nothing further is needed from it.
- **~~Scan and stare are not nested~~ — EXPLAINED 2026-08-28, and the reason is stronger than first thought (D24, 2026-09-01).** 209 scan-only vs 174 stare-only emitter instances across the 47 pairs. The dataset paper gives one mechanism: pulses are dropped when the Rx is not tuned to the band, when the Tx is too far, or when pulse width falls below 0.0069 µs, plus random drops — applying to *both* modes. But **the two files are also independent simulation runs**, measured 2026-09-01: nearest-neighbour ToA matching finds no correspondence (median |Δt| 185 µs to 4.81 s against a 0.025 µs noise scale), and the same emitter gets disjoint activity windows in the two runs. Stare is an oracle in coverage, not in detection — and not a second observation of the scan run's timeline at all. Not a defect; a property to model, and the reason each scenario carries contributions from exactly one recording (D25).
- **`scan_rate_rpm` is not revolutions per minute.** Folding each emitter's pulse times at `60/scan_rate_rpm` gives **two** peaks per cycle; folding at `30/scan_rate_rpm` gives exactly one. Verified on labels 3, 11, 12, 16 of `config_2` stare. The empirical rotation period is `30/scan_rate_rpm` seconds. Do not take the field name at face value.
- **`beam_width_deg` is not the detectable window.** For the same emitters the main illumination peak is 40–50° wide against a nominal `beam_width_deg` of 1.8–2.5°, and pulses are recorded at 267–359° of the full revolution. Emitter beam pointing modulates how many pulses arrive; it does not gate them on and off. **Re-verified 2026-09-03 (D28):** folding all 19 `config_2` stare emitters into 36 phase bins, 15 occupy every bin, median peak swing across phase 53.8 dB; the four exceptions are sampling-limited (label 18: 1.9 pulses/revolution), and the one `Omni` emitter swings 3.9 dB. There is no emitter off-state for a scheduler to coincide with.
- **Emitters have on/off activity windows that are not in the metadata.** `config_2` label 3 transmits continuously from 4.14 s to 16.98 s and is silent outside it (no gap >100 ms within). Nothing under `metadata/transmitters/transmitters_3` encodes that window. This is the main obstacle to regenerating a scenario from configuration alone.

---

## Summary-of-a-sibling relationships (explicit, per instruction)

| Summary document | Primary it summarises | Primary present here? |
|---|---|---|
| `Scanning Strategy Learning For Electronic Support Receivers by Robust Principal Component Analysis (3).pdf` | `118700Q.pdf` — the same paper, same title | **Yes.** Always cite `118700Q.pdf`. |
| `GPT BACKDOOR PAPER KHARKIV CONFRENCE.pdf` | KhNUPS 2025 conference proceedings (888 pp) | **No.** Every claim it makes about that conference is unverified here. |
| `data/turing/README.md` | The TSRD dataset + the Turing challenge GitHub repo | **The data is present.** The card is not. Prefer the files. |
| `me 2.0.pdf` | `paperSSPD (1).pdf` — the Apfeld paper | **Yes.** Cite the paper. Its §17 and §18 are the author's own and may be cited as such. |
| `Electronic Support Scan Scheduling (1).pdf` | Five papers; **three** are here (`118700Q.pdf`, `paperSSPD (1).pdf`, and Teissier 2026 as of 2026-09-04), two are not | **Mostly.** Only Clarkson 2005 and Hatcher 1976 remain unverified. |
| `CPRIT(Combined PRI Transform).pdf` p.4 | `Radar_Signal_Deinterleaving_in_Electronic_Warfare_.pdf` — Nuhoglu & Cirpan | **Yes.** Cite the paper. Its pages 5+ are the author's own proposal and may be cited as such. |

`118700Q.pdf` also cites `optimumsearch.pdf` as its reference [1] (Köksal 2010). Both primaries are
in this repository, so the RPCA paper's account of periodic search can be checked against the thesis
directly rather than taken on trust.

---

## Document records

## SIH26055 — Smart Scan Strategy for Electronic Warfare (official problem statement)

**Path:** `docs/project/SIH26055_PROBLEM_STATEMENT.md`

**Role:** ENVIRONMENT / EVALUATION / RL — it is the requirement, so it touches every role

**Stage:** NOW, and permanently

**Authority:** **A** — this is the specification. It outranks `PROJECT_ARCHITECTURE.md`, which is our reading of it, and every PDF here.

### What it contributes
- The environment definition: *"a simulated RF environment which has truth information on status of emitters in each band and at each time slot."*
- The observation model: *"the status of environment for each frequency band at each time step can be recorded as a transmission or a non-transmission."*
- The training signal: *"the model should then be trained based on hits and misses."*
- The metric set, named explicitly: Pd, Pfa, sensitivity, average intercept rate, average reward/cost, percentage of correct predictions, average intercept time error, plus intercept time and interception ratio.
- The objective: minimise intercept time, maximise interception rate.
- The required emitter behaviours: spatially scanning **and** frequency agile.
- It names the Turing dataset as the data source, so our grounding is expected rather than chosen.

### What it does NOT establish
- Time-slot duration, band count, band width, or whether bands overlap.
- How a band-slot's transmission status is computed from emitter geometry and power.
- Which ML method. "Machine learning based" is as specific as it gets; RL is our reading of "trained based on hits and misses", not the PS's word.
- What generates false alarms, despite requiring Pfa as a metric.

### Assumptions
- That a binary transmission/non-transmission status per band-slot is an adequate abstraction of the RF world. The PS asserts this; the Turing PDW data is far richer, so we are deliberately discarding detail to meet the spec.

### Relevant project component
- All of them. Use it to settle scope disputes.

### Conflicts/questions
- **`docs/reference/background/TECHNICAL DIFFERENTIATION STRATEGY.pdf` §1** says treat classification as a supporting component and build a scheduler. The PS agrees. Where that document adds clustering, drift detection and explainability, the PS does not ask for them — they remain OPTIONAL.
- **The Turing data is PDW-level, the PS is band-occupancy-level.** Converting one to the other is the central modelling step, and the PS does not specify it. See the open questions in the team brief.
- *"Approaches to intercept a periodic scan receiver optimally"* is ambiguous and flagged UNRESOLVED in the PS file itself.

### Decision
- Treat as the specification. When this map and the PS disagree, the PS wins; when the PS is silent, `PROJECT_ARCHITECTURE.md` fills the gap; when both are silent, it is a team decision to be recorded.

---

## Scanning Strategy Learning For Electronic Support Receivers by Robust Principal Component Analysis

**Path:** `docs/reference/scheduling/118700Q.pdf`

**Role:** RL / BASELINE / ENVIRONMENT

**Stage:** NOW (for its §4 environment and baseline); LATER (for its RPCA/TPSR method)

**Authority:** B

### What it contributes
- Full bibliographic identity: Ismail Gul (ITU / ASELSAN) and Isın Erer (ITU), *Artificial Intelligence and Machine Learning in Defense Applications III*, Proc. SPIE Vol. 11870, 118700Q, 2021, doi 10.1117/12.2601109. 7 pages.
- A precise statement of our problem: a narrow-band ES receiver cannot cover the spectrum at once, so a frequency-scanning strategy must be planned; when emitter parameters are unknown, plan it by learning.
- An explicit, minimal environment (§4) we could reproduce: emitters modelled as cyclic binary pulse trains, one emitter per band, receiver picks one of K bands per time step.
- An explicit binary observation rule (§4): `o_t = 1` if the radar's pulse train is high at time `t` **and** the chosen band contains the radar's frequency; `0` otherwise — including when the signal is below receiver sensitivity or the transmit beam is not pointing at the receiver.
- Its simulation constants (§4): detection bandwidth 1 GHz, smallest detection time 100 ms, coverage 2–12 GHz, K = 10 bands, PRI 40–80 time steps, pulse width 10–20 time steps, 5% jitter, 300-step bootstrap with a periodic search strategy, 30×10⁴ total time steps.
- A bootstrap baseline: the receiver "begins with a simple periodic search strategy… the tuned frequency band changes incrementally from 1 to K after each time step".
- Method chain: TPSR for state, RPCA (replacing SVT) for subspace identification, an exponentially-weighted multi-armed-bandit rule (eq. 12) for band selection.
- Reported result (theirs, not ours): RPCA and SVT perform "quite equal"; per-band interception ratios in Table 1 range from 26.29% to 61.23% across ten 1 GHz bands.

### What it does NOT establish
- That RPCA beats SVT. The paper's own conclusion is that they are equivalent in interception performance; RPCA is offered as a substitute, and the stated future work is *computational speed*.
- Any result on the Turing dataset, on PDW-level data, or on any real receiver. §5 says evaluation "using a real ES support system instead of a simulation environment" is future work.
- That its 10-band, one-emitter-per-band world resembles Turing. It does not: Turing has up to 99 transmitters per scenario against 36 dwell positions, with many emitters sharing a band.
- A reward function for RL. Its selection rule is a bandit weighting, not an RL reward.

### Assumptions
- **Each radar operates in a different frequency band** (§4, stated outright). Turing violates this heavily.
- One emitter's activity per band, represented as a binary pulse train — no PDWs, no amplitude, no AoA.
- The receiver's observation is a single bit per time step.
- Time is discretised into uniform steps; the dwell is one step.
- A single receiver / single learning agent.

### Relevant project component
- Scheduler and the hit/miss feedback loop. Its §4 observation rule is the cleanest published statement of the "HIT / MISS" arrow in `PROJECT_ARCHITECTURE.md` §1.
- Its periodic bootstrap is a legitimate baseline for us.

### Conflicts/questions
- Its "one emitter per band" assumption conflicts with the Turing data (verified: config_2 has two `AN/APG-81` emitters, labels 17 and 19, both at 10000/12000 MHz). Any use of its environment must drop this assumption or say why it kept it.
- Its binary observation conflicts with what Turing actually records, which is a 5-column PDW per pulse. Collapsing PDWs to a bit is a modelling decision, not a given.

### Decision
- Use as the reference statement of the problem and as the source for the hit/miss observation rule and the periodic baseline. **Do not** implement TPSR/RPCA now — it is the paper's method, not its problem statement, and `PROJECT_ARCHITECTURE.md` §13 explicitly says a research paper's proposed architecture need not be implemented.

---

## Periodic Search Strategies for Electronic Countermeasure Receivers with Desired Probability of Intercept for Each Frequency Band

**Path:** `docs/reference/scheduling/optimumsearch.pdf`

**Role:** VALIDATION / BASELINE / ENVIRONMENT

**Stage:** NOW

**Authority:** B

### What it contributes
- Bibliographic identity: Emin Köksal, MSc thesis, Department of Electrical and Electronics Engineering, Middle East Technical University, January 2010, supervisor Prof. Dr. Mustafa Kuzuoğlu; 110 PDF pages (95 numbered). This is reference [1] of `118700Q.pdf`, so the primary behind that paper's "previous deterministic approaches" is in this repository.
- The vocabulary the whole field uses, defined in ch. 2.3: scan period, beamwidth, PRI, **sweep period**, **dwell time**, duty cycle. These map directly onto the Turing scan metadata.
- The coincidence / pulse-train (window function) model of interception (ch. 2.1–2.2) — the mathematical object behind "did the receiver and the emitter beam overlap".
- Intercept-time theory (ch. 3.2) via Diophantine approximation and Farey series, including maximum intercept time and a geometric construction.
- Min-max intercept time optimisation over a fixed sweep period and over a range of sweep periods (ch. 3.3).
- **Probability of intercept** as a defined, computable quantity, and dwell-time calculation to achieve a *desired* per-band POI (ch. 6.1–6.3). Table 6-1 gives POI and intercept time in terms of pulse-train parameters τ and T.
- A worked critique of Clarkson's strategy and the conditions under which it is undesirable.

### What it does NOT establish
- Anything learning-based or adaptive. It is deterministic/probabilistic scheduling with *a priori* emitter knowledge; the thesis is explicit that Clarkson's strategy "assumes that a priori knowledge about the radars that will be intercepted is available".
- Anything about PDW-level data, deinterleaving, or the Turing dataset (it predates it by 15 years).
- Any RL formulation, reward, or neural method.

### Assumptions
- Emitters are periodic and characterised by scan period, beamwidth, PRI and duty cycle — i.e. the emitter model is a periodic window function.
- The receiver sweep is periodic and the strategy is planned offline.
- Prior knowledge of the threat-emitter list (for the Clarkson-style strategies it evaluates).

### Relevant project component
- **Environment validation.** This is the strongest validation source in the repository: it gives closed-form expectations for intercept time and POI that a correct environment should reproduce for a controlled periodic case.
- **Baselines.** Periodic and probabilistic search are exactly the classical schedulers `PROJECT_ARCHITECTURE.md` §5 wants to compare against.
- **Metrics.** POI and intercept time are defined here rather than asserted.

### Conflicts/questions
- Its emitter model is a periodic window function; Turing emitters are described by richer configs (`freq_mode`, `pri_mode`, `pw_mode`, `scan_type`, `beam_width_deg`, `scan_rate_rpm`). Whether the Turing emitters reduce to the thesis's window function is an open, checkable question — not an assumption to make.
- Not read cover to cover this session: front matter, table of contents and abstract were read in full; chapters 3–7 were read via the table of contents and table list only. Treat specific equations as **unverified until opened**.

### Decision
- Adopt its terminology (sweep period, dwell time, duty cycle, POI, intercept time) as the project's vocabulary. Use ch. 2, 3.2 and 6.1 as the validation targets when the environment is built. Read the relevant chapters properly before quoting any equation.

---

## Technical Differentiation Strategy for the SIH Smart Scan Strategy

**Path:** `docs/reference/background/TECHNICAL DIFFERENTIATION STRATEGY.pdf`

**Role:** ENVIRONMENT / BASELINE / EVALUATION / FUTURE

**Stage:** LATER (with §2, §13, §14 relevant NOW as a checklist)

**Authority:** C — proposed strategy. Internal PDF title `SIH_Smart_Scan_Technical_Differentiation_Strategy`, author metadata "Abdullah Ansari", produced 2026-08-27. 7 pages, 30 numbered recommendations.

### What it contributes
- The single clearest statement of project framing in the repository: "Do not try to build merely a better signal classifier. Build a better receiver scheduler" (§1). This agrees with `PROJECT_ARCHITECTURE.md` §5.
- A baseline list (§2): random, round-robin, fixed-priority, recency/activity heuristic — "evaluated against these under identical simulated RF conditions".
- The partial-observability rule (§4): the simulator knows true emitter state, the scheduler must not; ground truth is for evaluation and hit/miss only.
- A layer separation (§5): detector / classifier / predictor / scheduler.
- A metric list (§13) and the argument that average intercept time is the persuasive metric (§14), with a worked definition: emitter active at t=10 s, detected at t=13 s → intercept delay 3 s.
- An ablation ladder (§21) from round-robin up to the full adaptive scheduler.
- Two pieces of discipline worth keeping: §27 "make numerical claims, not marketing claims", and §28 "do not claim to beat classified military systems".

### What it does NOT establish
- Any measured result. There are no numbers in this document that came from running anything — every figure in it is illustrative.
- That any of the 30 recommendations is necessary, or that the recommended differentiator (§29: unknown-signal discovery + temporal prediction + uncertainty-aware scheduling + hit/miss feedback + concept-drift adaptation) actually improves anything.
- That K-Means/DBSCAN, LSTM/GRU, or a decision-score formula (§11) are appropriate for our data. They are proposed.

### Assumptions
- That we will build our own RF simulator from scratch (§3) rather than ground scenarios in a dataset. It does not mention Turing at all.
- That emitters can be given priority classes (§17) — Turing metadata has a `function` string, not a priority.
- That the scheduler chooses "which band, when, and for how long" (§1) — i.e. variable dwell length is part of the action space. That is a bigger action space than the Turing scan receiver uses.

### Relevant project component
- Evaluation protocol and baseline set. §2, §13, §14, §21 are directly usable as a checklist once the environment exists.

### Conflicts/questions
- **Scope creep risk.** §6–§10 (clustering, LSTM prediction, uncertainty modelling) and §23 (concept drift) are a much larger system than `PROJECT_ARCHITECTURE.md` §10 asks for at this stage. The protocol's "no research-driven scope creep" rule applies: these are OPTIONAL/LATER, not requirements.
- §3 says build a simulator with invented emitters; `PROJECT_ARCHITECTURE.md` §3 says construct scenarios from Turing configuration information. The architecture wins — it is team-authored and higher authority than a strategy proposal.

### Decision
- Keep §2, §13, §14, §21, §27, §28 as the evaluation and baseline checklist. Treat §6–§11, §17, §23 as REFERENCE ONLY until baselines exist and the architecture calls for them.

---

## Ukrainian Military Research: AI, Adaptive EW, RF Signal Analysis (KhNUPS 2025 synthesis)

**Path:** `docs/reference/background/GPT BACKDOOR PAPER KHARKIV CONFRENCE.pdf`

**Role:** FUTURE — **candidate-idea source**

**Stage:** LATER (as an idea menu, once baselines exist); never NOW as evidence

**Authority:** D — it is a synthesis of a primary that is **not in this repository**.

### What it contributes
- Context that adaptive-EW, unknown-signal clustering and ML-based prediction are active published research areas.
- A metric list (§10) and baseline list (§11) that substantially duplicate `TECHNICAL DIFFERENTIATION STRATEGY.pdf`.
- §12 is genuinely useful and unusually honest: an explicit list of what the conference does *not* prove — no validated benchmark beating round-robin, no dataset, no Pd/Pfa/intercept-time results for this problem, no evidence of deployment.
- A correct framing of the project's actual contribution (§9): "Use learned information about the RF environment to dynamically allocate limited receiver observation opportunities."

### What it does NOT establish
- Anything about the KhNUPS 2025 proceedings that we can check. The 888-page primary is not in `docs/`; only a URL is given (§13). Under provenance rule 4, every claim here about what those authors wrote is **unverified in this repository**.
- Any technical method in enough detail to implement. It names K-Means, DBSCAN, SVM, CNN, LSTM, ACO without equations or parameters.
- Any connection to Turing, PDWs, or our receiver model.

### Assumptions
- That the named conference papers say what the synthesis says they say. Not checkable here.
- Its own §1 "evidence rule" — proposals are labelled proposals, not deployments — which it does follow.

### Relevant project component
- None directly. It is orientation material.

### Conflicts/questions
- The filename ("GPT BACKDOOR PAPER") does not describe the contents; the internal PDF title is `KhNUPS_2025_EW_AI_Technical_Synthesis_for_SIH`. Classified from contents, as required.
- Its §8 architecture and §10–§11 lists overlap heavily with `TECHNICAL DIFFERENTIATION STRATEGY.pdf` (same author metadata, same date range). Treat the two as one proposal, not two independent sources agreeing.

### Decision
- **Keep it as a list of ideas we may choose to try**, which is what it is good for. The team's position (2026-08-28) is that the methods it names — unknown-signal clustering (K-Means/DBSCAN), temporal activity prediction (LSTM/GRU), ML + optimisation for resource allocation — are plausible things to attempt on our problem and may produce good results. Nothing here is committed to; nothing here is ruled out.
- **The separation that matters:** it can motivate an experiment, it cannot support a claim. If we try one of these ideas, the justification is our own measured result, not this document.
- Do not cite it for any technical or factual claim. The 888-page primary is not in this repository and **we are not obtaining it** — access to it is not clean, and nothing in our plan depends on having it. The ideas stand on their own merits and will be judged by our own experiments.

---

## Scanning Strategy Learning… (one-page summary)

**Path:** `docs/reference/scheduling/Scanning Strategy Learning For Electronic Support Receivers by Robust Principal Component Analysis (3).pdf`

**Role:** BACKGROUND

**Stage:** REFERENCE ONLY

**Authority:** D — a summary of `118700Q.pdf`, which is in this repository.

### What it contributes
- A readable one-page explanation of PSR, TPSR, RPCA and MAB and how they chain together, useful for briefing a teammate.
- Nothing else. It is 1 page, produced in Canva (PDF producer metadata), author metadata "coder nesi", created 2026-08-27.

### What it does NOT establish
- Anything independent. Every technical statement in it comes from `118700Q.pdf`.
- It contains no equations, no simulation setup, no results, and no bibliographic reference to the paper it summarises.

### Assumptions
- Inherits all of `118700Q.pdf`'s assumptions, including one-emitter-per-band, without stating them. That omission is the main hazard of using it.

### Relevant project component
- Team communication only.

### Conflicts/questions
- Its near-identical filename to the primary makes it easy to cite by mistake. **Cite `118700Q.pdf`, never this file.**

### Decision
- Keep for onboarding. Never use as a source.

---

## Electronic Warfare (extended)

**Path:** `docs/reference/background/Electronic warfare (extended).pdf`

**Role:** BACKGROUND

**Stage:** REFERENCE ONLY

**Authority:** D — plain-language commentary on the problem statement. 14 pages, Google Docs export, headed "Electronic warfare | SIH26055".

### What it contributes
- The EA / EP / ES split, and the placement of our problem inside **ES**.
- A correct and well-put statement of why scanning exists at all: the receiver's instantaneous bandwidth is smaller than the spectrum it must cover, so "scanning is a hard physical necessity, not a design choice", and a fixed sweep is "essentially hoping" the emitter transmits while the receiver is looking.
- The open-loop vs closed-loop framing, which is the same distinction `PROJECT_ARCHITECTURE.md` §1 draws with its feedback arrow.
- The three emitter behaviours the problem statement cares about: frequency hopping, intermittent radiation, intelligence mismatch.
- The observation that detection is a two-grid problem — space/time **and** frequency — which matters for us because Turing emitters have `scan_config` beams as well as frequencies.

### What it does NOT establish
- Any quantity, equation, threshold or result. There is not a single number in it that could be used or checked.
- Any claim about current fielded systems that is sourced. Statements like "an order of magnitude" gap between instantaneous bandwidth and covered spectrum are illustrative.

### Assumptions
- That the problem statement's framing (rigid pre-programmed schedules failing against agile emitters) is accurate. Plausible, unsourced.

### Relevant project component
- Motivation and write-up. Useful for the report's introduction, not for design.

### Conflicts/questions
- None with the data or architecture. It is consistent with both, at a level of detail too coarse to conflict.

### Decision
- REFERENCE ONLY, for framing and write-up.

---

## The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving (the paper)

**Path:** `docs/reference/dataset/TSRD_dataset_paper_arXiv_2602.03856.pdf`

**Role:** DATASET — **the primary source for how our data was generated**

**Stage:** NOW

**Authority:** **B, and the highest available on dataset semantics.** Gunn, Hosford (Dstl), Jones, Zeitler, Groves, Nockles. arXiv:2602.03856v2, 7 Apr 2026, 6 pages. Retrieved and read in full 2026-08-28. This is the primary the dataset card summarises — the card is now demoted to a summary of this.

### What it establishes (§II, quoted)
- **Why pulses are missing — three explicit drop rules.** *"Pulses were dropped when the Rx was not tuned to the correct frequency band, when the Tx was too far for detection, or when the pulse width dropped below a threshold (0.0069µs). Consequently, not all emitters were visible to the Rx."* Verified against our files: minimum pulse width across 72,031,672 train pulses is **exactly 0.006900 µs with zero pulses below it**. That threshold is the `pw_res_us` receiver attribute.
- **Detection is probabilistic, not a hard threshold.** *"The simulation has an ambient noise of -100 dB. The received amplitude decreases quadratically with emitter distance, and the probability of pulse detection increases the more distinct the signal is from the noise floor."* This resolves the `sensitivity_dbm` puzzle — there was never meant to be a cliff. Detection probability is a smooth function of SNR against a −100 dB noise floor.
- **The scan receiver, exactly as we measured it.** *"the scan receiver model which sweeps the frequency spectrum at centre frequencies between 0.5 - 18 GHz in 500 MHz steps and 500 MHz bandwidth at deterministic but varying dwell times. Pulses sent on frequencies outside the tuned 500 MHz bandwidth were dropped."*
- **Stare drops pulses too.** *"Stare mode can be understood as an oracle receiver that can observe the entire EME (except randomly dropped pulses)"* (repo README). The paper's drop rules apply to both modes — which is why stare is not a superset of scan.
- **The data is emitted ground truth, not receiver output.** *"Data in the TSRD can be understood as the emitted ground truth in the environment rather than imitating receiver behaviour"*; *"To make the dataset mostly independent of Rx hardware characteristics, we focused on simulating realistic Tx properties and simplified signal detection."*
- **Noise model, in detail.** Additive Gaussian white noise on ToA and pulse width at both emission and reception; frequency jittered with an Ornstein-Uhlenbeck process at emission and *not* adjusted at the receiver; amplitude and AoA blurred with OU to model atmospheric interference. Line-of-sight path loss, no multipath.
- **68 transmitter types**, instances randomly sampled, initial positions uniform in a 250×250 km plane, straight-line constant-velocity motion.
- **A published baseline to beat/cite:** HDBSCAN on raw PDWs gives V-measure 0.54 stare / 0.19 scan (Table IV).

### What it does NOT establish
- Nothing about scheduling, interception, or reward. It is a deinterleaving dataset paper.
- No generator source code is released, and the paper gives no equation for the detection probability — only its qualitative form.
- It does not document `scan_rate_rpm` units or emitter activity windows, so our measurements on those stand as the only evidence.

### Conflicts resolved by this paper
- **`sensitivity_dbm` is not a threshold.** Confirmed by construction: detection is probabilistic against a −100 dB ambient noise floor. Our measurement (no cliff at −110 or −120, 4% of pulses below) is exactly what that model predicts. **This question is now closed.**
- **Stare is not an oracle in practice.** The paper calls it an oracle but also states pulses are randomly dropped and that emitters too far away are not detected. Both statements are in the same document; the drop rules explain our 209/174 asymmetry.
- **Collection time is 30 s**, stated in §II, matching the files and contradicting the HF card's "10 seconds".

### Where the paper is contradicted by the files
- **Stated bandwidth.** The paper says the scan receiver sweeps *"in 500 MHz steps and 500 MHz bandwidth"*. The files disagree: pulses assigned to a dwell are spread evenly over ±500 MHz of its centre — 52.63% within ±250, 99.98% within ±500, with a hard edge at exactly 500 and roughly uniform density across all four 250 MHz quartiles (measured on `config_921`, n=189,308). A 500 MHz-total window would place ~100% within ±250. **The effective window is 1000 MHz wide on 500 MHz centres, so adjacent bands overlap by half.** The likeliest explanation is that `bandwith_mhz = 500` is applied as a half-width in the generator. Per the CLAUDE.md authority table, the files win; treat the paper's phrasing as loose.

### Decision
- **Promote to the primary source on dataset semantics.** The HF dataset card is a summary of this and should not be cited where the paper covers the same ground. Cite this paper for anything about how the data was generated.

---

## The Turing Synthetic Radar Dataset — dataset card

**Path:** `data/turing/README.md`

**Role:** DATASET

**Stage:** NOW — but as a hypothesis to check, not a source.

**Authority:** **D — it is a summary of `docs/reference/dataset/TSRD_dataset_paper_arXiv_2602.03856.pdf`, which is now in this repository.** Cite the paper, not the card. Superseded by the files wherever either disagrees with an observed field.

### What it contributes
- Dataset identity: The Turing Synthetic Radar Dataset (TSRD), Gunn, Hosford, Jones, Zeitler, Groves, Nockles; Apache-2.0; supported by the Turing's Defence and Security programme.
- The PDW definition we can confirm against the files: 5 features — ToA (µs), Centre Frequency (MHz), Pulse Width (µs), AoA (deg), Amplitude (dB). **Confirmed** against `metadata/feature_names` this session.
- A pointer to the primary documentation: the `alan-turing-institute/turing-deinterleaving-challenge` GitHub repository. That repository is **not** in this project, so anything it says is unverified here.
- The intended framing of the dataset: pulse **deinterleaving**, evaluated with clustering metrics (V-measure, ARI, AMI, MCC, F1).

### What it does NOT establish
- The contents of our 47 pairs. It describes the full dataset (6,000 pulse trains; 2,500 train / 250 val / 250 test per mode). We have 47 scan/stare pairs, train split only.

### Assumptions
- That "stare" is an oracle. See below.

### Conflicts/questions — **all three verified against the files this session**
1. **Collection time.** The card says stare observes "over 10 seconds". Every one of the 94 files carries `metadata.attrs['collection_time_s'] = 30.0`, and observed ToA reaches 29.93 s. **The files win: 30 s.**
2. **Stare as an oracle.** The card says stare is an "Oracle receiver detecting all signals across the entire frequency spectrum (0–18 GHz)". Both modes carry `freq_range_mhz = [500, 18000]`, not 0–18000, and across the 47 pairs there are **174 emitter instances detected by stare but not scan and 209 detected by scan but not stare**. Stare is not a superset of scan and is not a full-spectrum oracle.
3. **Per-file pulse counts.** The card's averages (1.29 M stare, 94 k scan per train pulse train) are for the whole train split, not our subset. Our 47: 4,393,233 scan pulses and 67,638,439 stare pulses in total (counted this session).

### Decision
- Use the card only for dataset identity, authorship and the PDW column meanings. Take every quantity from the files. Where a design question depends on dataset semantics the card gets wrong, raise it rather than adopting the card.

### How our 47 were selected — verified 2026-08-28

The team's account is that the subset is *40 stratified + 6 extremes + 1 mid-range extra*.
That is exactly right, and the stratification variable is **stare file size**, not scan.
Reconstructed by listing the 2,500-file `stare/train_stare` split from the Hugging Face API
(file metadata only — nothing downloaded) and ranking our 47 within it:

- Divide the 2,500 files into **40 equal-count strata of 62.5 files** by stare size. Every one
  of the 40 strata contains **exactly one** of our files, at a consistent position within its
  stratum (ranks 104, 166, 228, 291, 353, … 2469 — a clean 62.5 step). Those are the 40.
- **6 extremes**: three at the bottom of stratum 0 — `config_81` (rank 11), `config_1089` (12),
  `config_2356` (13) — and three at the top of stratum 39 — `config_418` (2498),
  `config_1902` (2499), `config_1950` (2500, the single largest file in the split).
- **1 mid-range extra**: `config_706`, stare rank **322/2500** (12.9th percentile), which lands
  in stratum 5 alongside that stratum's regular pick `config_1710` (rank 353).

**`config_706` is therefore not an extreme** — it is the odd one out, and the suspicion that it
didn't belong with the extremes is correct. It is a second sample inside stratum 5.

Two smaller corrections to the mental model: `config_81` is our smallest but is **rank 11**, not
the smallest in the split (five files tie at 23,784 bytes); and `config_1534` (rank 42) also sits
in stratum 0, so stratum 0 holds four of our files rather than the one its share implies.

Ranking on *scan* size instead scatters the same 47 irregularly (gaps of 1 to 422 ranks), which
is why the pattern is invisible from the `scan/train_scan` sizes alone.

### The held-out test subset — fetched 2026-08-28

45 test pairs (90 files, 1.1 GB) downloaded into `data/turing/scan/test_scan/` and
`data/turing/stare/test_stare/`. Verified on arrival: 45 ids in each, id sets identical,
all files open, `collection_time_s = 30.0`, 36 dwell bins on scan and 0 on stare — same
structure as train.

**Selection rule, stated rather than reverse-engineered so it is reproducible:** rank all 250
`stare/test_stare` files by size; cut into 40 equal-count strata and take the **middle** file of
each; add the 3 smallest and 3 largest in the split. That yields 46 slots, of which one collides
(stratum 0's middle is also a bottom-3 extreme), giving **45**. Train used an arbitrary ~66%
within-stratum offset; the middle is the canonical choice and gives the same coverage.

**Chosen config ids:** 5, 7, 16, 17, 20, 21, 22, 37, 40, 42, 46, 49, 53, 63, 64, 71, 78, 86, 87,
88, 89, 97, 102, 105, 117, 119, 124, 131, 142, 145, 146, 155, 161, 172, 178, 188, 195, 197, 209,
210, 222, 235, 240, 242, 245.

These ids are **test-split ids and are unrelated to train ids of the same number** — e.g.
`test/config_64` and `train/config_64` are different scenarios.

**Discipline:** this subset was chosen by a rule fixed before any result was seen, and it must
stay untouched until the system is frozen. Every use of it should be recorded.

---

## An Adaptive Receiver Search Strategy for Electronic Support

**Path:** `docs/reference/scheduling/paperSSPD (1).pdf`

**Role:** BASELINE / ENVIRONMENT / VALIDATION

**Stage:** **NOW** — the strongest non-learning baseline available to us

**Authority:** **B, and the most directly relevant paper in this repository.** Sabine Apfeld, Alexander Charlish, Wolfgang Koch — Fraunhofer FKIE, Dept. Sensor Data and Information Fusion. 5 pages, 2016. Added by a teammate 2026-08-29; read in full the same day.

### What it contributes
- **A direct critique of the model everything else in this folder uses.** *"The majority of today's literature regarding this topic models the intercept problem as that of the coincidence of two or more periodic window functions. Since this model is rather simplistic, in this paper the radars' illumination patterns are described by signal-to-noise ratio time series."* This puts Köksal and Clarkson in context: their window-function model is the classical approach, and this paper is the correction to it.
- **Why the correction matters:** *"window functions usually only consider the main beam of the radar. In the presented approach, the radars can be intercepted and detected through the sidelobes as well."* **Our Turing measurements independently confirm this is the right model** — see D4 in `docs/project/DECISIONS.md`.
- **A complete, reimplementable adaptive algorithm** (§II): random start; SNR over threshold `T_D` promotes a band to a "tentative" list; tentative bands visited more often via Algorithm 1 (scaling `y`, cap `z`); autocorrelation of the intercepted SNR series estimates the emitter scan period (Eq. 3); once the estimate is stable (std over last `j` below `T_std`) dwells are scheduled at predicted SNR maxima plus integer multiples of the period; misdetections widen the search, and after `s` misses the band returns to exploration.
- **An SNR equation** (Eq. 1) with every term defined — peak power, pulse width, PRI, wavelength, transmit/receive gain, range, Boltzmann, noise temperature, bandwidth, losses.
- **A baseline ladder we can reuse directly:** Adaptive / Adaptive-without-tracking / "Active RFs" / Random.
- **Their metrics:** efficiency (percentage of dwells on bands with an active emitter that produced a detection), total detections, percentage of radars detected at least once.
- **Their setup:** 250 MHz instantaneous bandwidth, 50 ms dwell, 2–18 GHz, 10 radars, 5 minutes simulated, 30 runs per configuration.
- **A result that directly shapes our metric design:** Random scored **best** on percentage of radars detected at least once, and **worst** on efficiency — *"the random strategy only explores the environment without exploiting the information it obtains."* Coverage and efficiency trade against each other; reporting one alone is misleading.
- **An honest negative result:** scheduling for tracking dwells *"doesn't seem to make a major difference in performance"*, and they recommend the simpler variant on computational-cost grounds.

### What it does NOT establish
- Nothing about RL. It is a hand-designed adaptive heuristic; there is no learned policy, no reward, no training.
- No absolute performance claim — *"the results shown in the next section are to be seen as a comparison and not an absolute performance measure."*
- Its scenario assumes **one radar per frequency at a time**, explicitly to avoid implementing deinterleaving. Turing violates this heavily.
- No public code or data.

### Assumptions
- The receiver can distinguish search dwells from tracking dwells (justified by differing waveforms).
- Emitters are phased-array multifunction radars whose periodicity is broken by interleaved tracking — a *different* emitter model from Turing's mechanically-rotating `Circular` scan (95.6% of our transmitters).
- Fixed dwell time and fixed instantaneous bandwidth.

### Relevant project component
- **Baselines (Lane D)** — reimplement as the strong non-learning comparator.
- **Environment (Lane B)** — its SNR formulation is the model our data actually matches.
- **Metrics (Lane D)** — its efficiency/coverage split is a trap we would otherwise have walked into.

### Conflicts/questions
- **Against Köksal and Clarkson:** window function versus SNR time series. **Resolved in our favour by measurement** — Turing emitters show 53–62 dB of amplitude variation across a revolution with pulses in all 36 phase bins. The SNR model wins for our data. This does not make Köksal useless; his intercept-time theory remains the validation target.
- **Against the PS:** the PS mandates binary transmission/non-transmission per band-slot. Reconciled by D4 — continuous underneath, thresholded for the interface.

### Decision
- Adopt its SNR-based illumination model (D4). Reimplement its scheduler as our strong baseline (D13). Do not adopt its one-emitter-per-band assumption.

---

## Electronic Support Scan Scheduling (comparison note)

**Path:** `docs/reference/scheduling/Electronic Support Scan Scheduling (1).pdf`

**Role:** BACKGROUND — a map of the field

**Stage:** NOW, for orientation

**Authority:** **D** — a teammate's comparison note (Canva, "coder nesi", 2026-08-28). Two of the five papers it describes are in this repository; three are not.

### What it contributes
- **The most useful framing we have: what each paper actually treats as the decision variable.** Clarkson optimises *dwell time per band plus sweep period*, with frequency order explicitly not the interesting variable. Glaude et al. and Gul & Erer optimise *which band next*. Apfeld optimises *which band and when to revisit it*. Teissier et al. optimise *dwell time* under a random scan.
- A clear statement of the gap it argues we should occupy: a unified closed-loop scheduler allocating limited sensing across multiple uncertain emitters while optimising both interception probability and intercept time.

### What it does NOT establish
- Anything checkable about Clarkson (2005) or Teissier et al. (2026) — **neither is in this repository**, so those descriptions are unverified.
- No equations, no results, no evaluation.

### Conflicts/questions
- It states Gul & Erer's novelty is only the model-learning method, not a new scheduler action. That matches our own reading of `118700Q.pdf`. Two independent readings agreeing is worth something, but both are readings of the same primary.

### Decision
- Use as a map, not a source. Its identification of the decision variable per paper is genuinely useful for positioning our contribution. **Teissier et al. (2026) is worth obtaining** — it derives probability of intercept, intercept time and pulse interception ratio against *frequency-agile and spatially scanning* radar, which is exactly the emitter pair the PS demands.

---

## Smart Spectrum Surveillance for Electronic Support (ES) — BASICS

**Path:** `docs/reference/background/Smart Spectrum Surveillance for Electronic Support (ES) [BASICS].pdf`

**Role:** BACKGROUND / EVALUATION

**Stage:** REFERENCE ONLY, except §6 which is NOW

**Authority:** **D** — teammate brief (Abdullah Ansari, 2026-08-28). Vendor claims in §4 are explicitly flagged by its own author as *"design intent, not independently verified combat performance."*

### What it contributes
- A clean first-principles account of why scanning exists: instantaneous bandwidth, sensitivity, dwell/revisit time and dynamic range as four linked constraints, with `N = kTB` showing why wider coverage raises the noise floor.
- A survey of fielded systems (R&S, HENSOLDT, Saab, BAE, US Army) — useful for a PPT slide on the operational landscape, not for design.
- **§6 contains one genuinely important warning we have adopted:** *"a model predicting 'no transmission' can score high accuracy while being operationally poor."* Recorded in the evaluation plan of `docs/project/DECISIONS.md`.
- Two other framings worth keeping: the "wideband paradox" (wider collection shifts the bottleneck to deciding what deserves analysis), and "periodicity is not universal".

### What it does NOT establish
- No equations beyond `N = kTB`, no measurements, no citations to primaries.

### Decision
- Keep §2 and §6 for the write-up and the metric design. Treat §4's system survey as unverified vendor positioning.

---

## An Adaptive Receiver Search Strategy — teammate explainer

**Path:** `docs/reference/scheduling/me 2.0.pdf`

**Role:** BACKGROUND

**Stage:** REFERENCE ONLY — but genuinely useful for onboarding

**Authority:** **D** — a summary of `docs/reference/scheduling/paperSSPD (1).pdf`, which is in this repository. Farheen Khan, 10 pages, 2026-08-29. Image-based PDF with no extractable text layer; read by rendering the pages.

### What it contributes
- A patient walk-through of the Apfeld paper for someone new to the domain — dwells, why frequency alone is insufficient, the autocorrelation step.
- **Two things that go beyond the paper and are worth keeping.** §17 draws a three-way distinction we should build into the code: **radar state** (what the emitter is physically doing) vs **physical illumination** (the SNR that *would* be received if tuned there) vs **receiver observation** (the SNR actually obtained given the schedule) — with the critical case that a radar can be strongly illuminating while the receiver, tuned elsewhere, observes nothing. That is precisely the truth/observation separation our environment needs, and it is stated more clearly here than in the paper. §18 decomposes the paper into modules (emitter model, antenna/beam model, SNR generator, scheduler, detection, SNR history, period estimator, stability monitor, prediction) which is a reasonable starting shape for Lane B.

### What it does NOT establish
- Nothing independent — every technical claim traces to `paperSSPD (1).pdf`. The module breakdown and the truth/observation table are the author's own contribution and are interpretation, not source.

### Decision
- Use for onboarding teammates onto Apfeld. **Cite `paperSSPD (1).pdf`, not this.** Carry §17's three-way distinction into the environment design.

---

## Documents added 2026-09-01 — compact records

These were classified by opening each one this session. Kept brief; none changes a decision,
several confirm one.

### `iDEX ADITI 4.0 (Go to Page 12).pdf` — Cognitive EW System (Indian Army)
**Role:** BACKGROUND / intent. **Authority:** A for *context*, on the sibling problem. Page 12
is the ADITI 4.0 "Cognitive Electronic Warfare System" challenge: an autonomous ES+EA system
that senses, self-learns, and deploys countermeasures, explicitly replacing the historical
"previously learned threats" database model. **Our SIH26055 is the ES scan-scheduling slice of
this larger vision** — it grounds the D20 cold-start / active-warfare intent, but does **not**
extend our scope to jamming, DF or fusion. See the 2026-09-01 consistency audit.

### `ADITI 4.0 YT TRANSCRIPT.pdf` — official Q&A
**Role:** BACKGROUND / intent. **Authority:** B (verbatim official transcript). Covers the CEW
challenge among four. Substance for us: the Army frames CEW around AI that will *"sense, adapt
and self-learn environment challenges,"* and when asked whether a threat library would be
provided, would not commit — *"up to us to build a threat library."* Confirms D20: no assumed
prior intelligence. No mention of the SIH dataset or scan mechanics (it is the broader system).

### `SIH- Smart Scan Strategy.pdf` — teammate crash course
**Role:** BACKGROUND / EVALUATION. **Authority:** D (teammate synthesis), but technically sound
and independently confirmatory. Lays out truth Z vs declaration Y, the four detection outcomes,
Pd/Pfa, ROC vs threshold γ, noise floor kTBF (≈ −109 dBm at 1 MHz, NF 5 dB), intercept time, and
the bandit framing (UCB, Thompson). Matches D4/D6/D21 independently. Good PPT/onboarding source.

### `RFenvironment(baseline).pdf` + `rf_env_grounded.py` — teammate env
**Role:** ENVIRONMENT. **Authority:** C (a proposed design). A gymnasium env deriving action
space Discrete(N), a POMDP framing, and a history-based observation (per-band hit rate, visit
density, staleness + time) strictly from PS text. **We adopt its L3 interface and credit it;**
its binary placeholder truth grid is replaced by our measured L0–L2 (see `ENVIRONMENT_SPEC.md`).

### `Dynamic Scan Scheduling.pdf` — Dutertre, SRI, RTSS'02
**Role:** BASELINE / BACKGROUND. **Authority:** B. Online cyclic scan-schedule construction over
**disjoint** bands; NP-hard with an exploitable phase transition. Source of the D20 quote that
fixed a-priori-table scheduling is the limitation. Its disjoint-band, full-schedule model
differs from our overlapping-band, next-band action (D3) — reference, not template.

### `Dwell_Time_Optimization_of_Alert-Confirm_Detection.pdf` — JEES 2019
**Role:** VALIDATION / FUTURE. **Authority:** B. Alert–confirm (sequential-detection) dwell
optimisation for AESA radar. Relevant as a *possible v2 receiver refinement* (two-threshold
confirm-on-alert) and as support for treating detection threshold and dwell as a separate layer
from scheduling (D21). Not needed for v1.

### `Radar_Signal_Deinterleaving_in_Electronic_Warfare_.pdf` — Nuhoglu & Cirpan, IEEE Access 2023
**Role:** REFERENCE (deinterleaving). **Authority:** B. PRI-transform combined deinterleaving,
strong on staggered PRI. Out of scope per D12/D19; a reference only if an observation channel
ever needs per-emitter attribution.

### `PassiveRadar.pdf` — undergraduate survey
**Role:** BACKGROUND. **Authority:** D. Passive **bistatic** radar (transmitters of opportunity)
— a *different* meaning of "passive" than passive ES receiving. Filed to prevent conflation; no
bearing on our design.

### `samyukta_ew_system.md` — note on the Samyukta EW system
**Role:** BACKGROUND / intent. **Authority:** D (teammate note, unsourced). Describes an Indian
tactical ESM system (COMINT+ELINT passive sensors, 1.5 MHz–40 GHz). Colour for the operational
picture and PPT; not a design input.

---

## Documents added 2026-09-04 — compact records

Six additions, each opened and read this session with `pymupdf`; none classified from its
filename. **One is a significant acquisition** (Teissier, previously listed as wanted and
absent). **Two propose reopening a settled scope decision** and are flagged, not absorbed.

### `Interception_Model_of_Random_Scanning_Strategies_Against_Frequency-Agile_Radar_in_Electronic_Support_copy.pdf` — Teissier, Toumi & Khenchaf, IEEE TAES, published 2026-02-03
**Role:** VALIDATION / BASELINE. **Authority:** B (peer-reviewed, IEEE TAES, DOI
10.1109/TAES.2026.3660987; DGA/AID funded). **This is the paper the "worth obtaining" list
named and did not have.** It derives intercept time (IT) and **pulse interception ratio (PIR)**
for a *randomly scanning multichannel superheterodyne receiver* against *frequency-agile
emitters with rotating directional antennas* — the emitter class the PS names, and closer to our
setup than anything else in the repository. It also gives a minimax scheme for optimising a
random scan against multiple known emitters under parameter uncertainty, and validates the model
in a simulated EW environment. The authors position it explicitly as *"a theoretical baseline and
design tool for adaptive scanning strategies"* — i.e. as the thing an RL scheduler should be
measured against.

- **Why it matters to us:** it is the second document in the repository (with `optimumsearch.pdf`)
  carrying derivable equations an environment can be checked against, and the first whose emitter
  model includes frequency agility. **NOW** for validation; **NOW** for the baseline list.
- **What it does NOT establish:** it assumes *known* emitter parameters and constant dwell per
  band. Our problem is the cold-start case (D20), so its optimisation is a bound and a baseline,
  not our method.
- **Unverified / needs a human read:** whether its PIR is the same quantity as our interception
  ratio in `EVALUATION.md` §4. The names coincide and the definitions look close, but I have not
  checked them term by term, and `EVALUATION.md` is the single authority on metrics — **do not
  align the two without that check.** Flagged as an open question below.

### `The_VITA_49_Analog_RF-Digital_Interface.pdf` — Cooklev, Normoyle & Clendenen, *IEEE Circuits and Systems Magazine*, Q4 2012
**Role:** BACKGROUND / FUTURE (deployment). **Authority:** B (peer-reviewed tutorial). Describes
the VITA 49 / VRT packet protocol carrying digitised signal data plus context metadata (RF centre
frequency, bandwidth, IF, sample rate, gain, timestamps) between a radio's analog RF and digital
subsections. **REFERENCE ONLY for the current milestone** — it is about a real hardware interface,
and our receiver is simulated. Its value is the deployment-path argument for the write-up: it is
the standard a real ES receiver would use to hand our scheduler its observations.

### `Hardware Integration (2).pdf` — teammate note
**Role:** BACKGROUND / FUTURE (deployment). **Authority:** C (a proposed design, unsourced).
Sets out the receiver chain (antenna → RF front-end → downconversion → ADC → DSP → VITA-49) and a
control loop (observations → smart scheduler → `TUNE(frequency, bandwidth, dwell_time)` → receiver
API → retune), then maps our prototype onto it: the virtual receiver stands in for the hardware,
the receiver interface for VITA-49. **Useful, and consistent with our L2/L3 split** — it is the
same boundary `ENVIRONMENT_SPEC.md` draws, arrived at independently. Good PPT material for the
"how does this reach real hardware" question. Not a design input; nothing here changes the
environment. Cite the VITA 49 paper, not this, for anything about the standard itself.

### `CPRIT(Combined PRI Transform).pdf` — teammate explainer + proposal
**Role:** BACKGROUND, and **UNRESOLVED** for its proposal. **Authority:** D for the explanatory
half; **C** for the proposal. Pages 1–3 are an EW primer (zero-prior-intelligence scanning,
COMINT vs ELINT, interleaved vs deinterleaved pulses) — sound, and useful for onboarding and the
deck. Page 4 paraphrases `docs/reference/deinterleaving/Radar_Signal_Deinterleaving_in_Electronic_Warfare_.pdf`,
**which is in this repository — cite the primary, not this.** Pages 5+ are the author's own
proposal: fuse CPRIT as a feature extractor beneath the RL scheduler, on the argument that an
agent seeing only hit/miss bits cannot understand a frequency-agile, staggered-PRI emitter.

### `CPRITworkflow.pdf` — one-page workflow diagram
**Role:** **UNRESOLVED** (a proposal). **Authority:** C. Places a "CPRIT MODULE" (phase
clustering, fused test statistics) between the narrow-IBW receiver and the RL agent, running
during a "strategic micro-dwell" and outputting true PRI / low-false-alarm tracks.

> **⚠ These two conflict with a settled decision, and are recorded here rather than acted on.**
> **D12** (`SETTLED` 2026-08-29) says clustering is not the deliverable; **D19** (`CLOSED`
> 2026-08-30) says deinterleaving is not required. Both CPRIT documents propose making PRI-based
> deinterleaving a core module of the scheduler. Per `CLAUDE_CODE_RESEARCH_PROTOCOL.md` a
> proposal at Authority C is a *candidate design, not new evidence*, and `CLAUDE.md` says a
> `SETTLED`/`CLOSED` decision does not reopen without new evidence. **The conflict is surfaced,
> not merged.** Reopening D12/D19 is a human call — and note the cost is real: it would add a
> per-emitter attribution channel to the observation, which is the D30 question at a larger
> scale. Listed under Unresolved.

### `docs/reference/PPT/` — past SIH winning decks and two pitch guides
**Role:** BACKGROUND (presentation). **Authority:** D. Ten decks plus a scraped repository readme;
material for building our own deck, not for the design. **Seven of the ten have no text layer**
(slide exports as images) and must be read visually; only the TechDoodles playbook, the
easy-vs-difficult guide and the Cattle-Race deck carry text. `docsearch` isolates this folder in a
separate `presentation` collection so it cannot surface in a technical query.

---

## Prior art on the scheduling problem — searched 2026-08-28

Searched because the team asked whether this is an established problem with existing approaches
rather than something we invent. **It is established, but thinly, and nobody has solved our exact
version.** Summary of what a literature search surfaced; none of these papers is in this
repository, so **every claim here is unverified beyond its title and abstract** and should be
read before being relied on.

**The closest existing work is already in `docs/`.** `118700Q.pdf` (Gul & Erer 2021) and its
reference chain — Glaude et al. 2015 (PSR-based scanning strategy learning, MLSP), Clarkson 2006
and 2018 (sensor scheduling via Markov chains; intercepting beam-agile radar), Winsor & Hughes
2012 (receiver search strategy optimisation), and Köksal's thesis in `optimumsearch.pdf` — form
the actual lineage of this problem. That lineage is small: roughly a handful of groups over
twenty years. **This is the single most useful fact from the search.** Our reference list is
already better than it looked.

**Adjacent and much larger:** cognitive *radar* resource management. Substantial recent work
formulates radar scan/track time allocation as a POMDP and solves it with deep RL — constrained
DRL, multi-objective RL, belief-reward shaping. That literature is about a radar allocating its
own beam, not an ES receiver hunting unknown emitters, so the *problem* differs, but the
*formalism* transfers directly: partially observable state, discrete allocation action,
belief maintained over unobserved targets.

**A US patent exists** (US 10,523,342, "Autonomous reinforcement learning method of receiver scan
schedule control") covering RL for receiver scan scheduling. Relevant as evidence the idea is
established and as prior art to cite honestly; not an obstacle to research work.

**Not found:** any public implementation, benchmark or dataset for RL-based ES receiver
scheduling. No GitHub repository doing what we are doing. That gap is the project's opportunity
and also why we must build our own environment.

### What this means for us
- Reading `optimumsearch.pdf` and `118700Q.pdf` properly is worth more than a broad literature
  sweep. They *are* the field.
- The restless multi-armed bandit / POMDP framing is the right formal home, and the cognitive
  radar RRM literature is where to borrow method from.
- Nobody has published our benchmark. A validated environment plus honest baselines is a real
  contribution, independent of whether the RL agent wins.

---

## Documents referenced but absent — resolved

`docs/README.md` describes two files that are not in `docs/`. Both are accounted for
(confirmed by the repository owner, 2026-08-28) and **neither is missing in any way that
matters**:

- `SIH_RF_Brainstorming_Teammate_Guide.pdf` — was sent to teammates. It explains the
  architecture, which we already have in `docs/project/PROJECT_ARCHITECTURE.md`. Nothing in it is
  needed here.
- `CLAUDE_MD_UPDATE_PROMPT.md` — was a setup prompt belonging to the previous repository,
  for wiring up the documents, the research map and the working rules. That job has since
  been done directly, so the prompt is obsolete.

No action required. Recorded so a future session doesn't re-raise it as a gap.
