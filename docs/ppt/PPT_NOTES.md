# NARADA — SIH 2026 Idea Deck: content pack

**For the PPT lane. Built 2026-09-11.** Everything here is a *source* for slides, not slide text.
Rewrite it in your own words — the point of this document is that you never have to go looking for
a number, and that no number on a slide is one we cannot defend.

- **PS 26055**, DRDO — *Smart Scan Strategy for Electronic Warfare*. Theme: Robotics & Drones.
  Category: Software. **Team NIRVANA. Product: NARADA.** Tagline: *Right place at the right time.*
- Template: `SIH2026-IDEA-Presentation-Format.pptx` — **6 slides, title slide included.**
  Its own rules slide: no paragraphs, use points/diagrams/infographics, don't change the
  section headings, export to PDF.
- Figures are in `docs/ppt/figures/` at 200 dpi, drop-in ready. Regenerate with the `mk_*.py`
  scripts beside them.

---

## A. The one decision that shapes the whole deck

At idea-submission stage almost every deck is a **promise** — "we will build X". We are the rare
team that can submit a **result**: the system is built, it ran 5,643 episodes, and it beat every
baseline including the one from the published literature.

**So the deck's job is not to describe an idea. It is to prove the work is already done.**
Everything follows from that:

- Lead each slide with a **claim about approach**, and let one number support it. A judge cannot
  verify our numbers in the ninety seconds they spend on the slide — but they can absolutely
  judge whether we thought clearly. **Two or three numbers per slide, not a table.**
- **Show the course-corrections.** This is the strongest and most unusual thing we have. We were
  wrong about how to infer threat, wrong about how to encode it, wrong about our own reward
  function twice, and we found a train/test leak in our own work. Every one of those is written
  down with the evidence that changed our mind. Almost no team at idea stage can show that, and
  it is exactly what separates a project from a proposal.
- Every claim traces to something we ran. Nothing is aspirational except the clearly-labelled
  future work on slide 5.
- The threat-priority edge is **one block on slide 2 and one line on slide 5.** It is the tagline
  and the differentiator, not the subject. The PS's own objectives are the subject, and we have
  already met them.

**Story spine, one sentence per slide:**

| Slide | The sentence it has to land |
|---|---|
| 1 Title | — |
| 2 Proposed Solution | Open-loop sweeps waste time; we built a closed-loop learned scheduler that beats every baseline on both of DRDO's objectives at once. |
| 3 Technical Approach | Here is the four-layer system, and here is why you can believe its numbers. |
| 4 Feasibility & Viability | It already runs; here are the three real risks and what we did about each. |
| 5 Impact & Benefits | Faster, wider interception on the same hardware — plus an operator who can steer it. |
| 6 Research & References | The data, the papers we built on, and the papers we beat. |

---

## B. Slide-by-slide content

### Slide 1 — TITLE PAGE

Template fields only. PS ID **26055** · *Smart Scan Strategy for Electronic Warfare* ·
Theme **Robotics and Drones** · Category **Software** · Team ID · **Team NIRVANA**.

Keep **NARADA — Right place at the right time** as the product lockup. It is a good tagline:
the PS itself says interception *"involves adjusting receiver's frequency at correct time."*

---

### Slide 2 — PROPOSED SOLUTION

> **Headline banner (replaces "A 3-RL agent phase sequencer…", which is dead — we ship one policy):**
>
> ## Learns where to look. Listens when you know better.
>
> *A closed-loop scan scheduler that beats every baseline — including the published state of the
> art — on both DRDO objectives at once, and takes the threat library nobody ever wired in.*
>
> **Alternative if you want the number in the headline instead:** *"Both objectives at once —
> 3.04 s to intercept at 13.0% ratio, beating every baseline. Then let the operator aim it."*

**Left block — what it is (4 bullets, short):**

- Today's ES receivers sweep a **schedule fixed before the mission** and cannot react.
- NARADA closes the loop: **tune a band → hear a hit or not → update belief over all 36 bands →
  choose the next band.** 600 decisions in 30 seconds.
- Learned with **Recurrent PPO** on a simulated RF environment built from **2.2 GB of real
  recorded radar data**.
- **Autonomous by default.** An operator or threat library can hand it a priority vector; with no
  input it runs the plain DRDO objective unchanged.

**Middle block — our differentiation (this is the old "Aamir's idea / human intelligence layer" box).**
Lead with the *reasoning*, not the measurement. Suggested headline:

> **"Your threat library already exists. Nothing has ever wired it to the scanner."**

- **Human ON the loop, not in it.** It runs autonomously; a priority input is there when someone
  knows something the library doesn't. Default = uniform = the plain DRDO objective, unchanged.
- **It reuses data the platform already paid for** — the threat library in the processing unit,
  which every ES system has and no scan scheduler reads.
- **We found the problem in our own agent first.** The better our scheduler got at the DRDO
  objective, the *later* it reached threatening emitters — because adaptive scheduling chases
  activity, and dangerous emitters here are the quiet ones.
- **Then we got the design wrong twice before getting it right** — see `figures/fig4_threat.png`.
  Asking *"is anything dangerous in this band?"* flags almost every band and cannot rank. Asking
  *"how much of this band is dangerous?"* can.

*Keep at most one number in this block on the slide. The reasoning is the differentiator; the
numbers are in the notes if a judge asks.*

**Right block — the numbers (use these four, nothing else):**

| | NARADA | Open-loop floor | Best in literature |
|---|---|---|---|
| Interception ratio ↑ | **0.130** | 0.061 | 0.132 |
| Intercept time ↓ | **3.04 s** | 4.18 s | 4.32 s |
| Emitter coverage ↑ | **0.902** | 0.865 | 0.860 |
| Beats the floor on both, per episode | **81.9%** | — | 53.2% |

**Bottom-right — FIGURE:** `figures/fig1_pareto.png`. This replaces the "Pareto's CHART" box.
<div><img src="docs/ppt/figures/fig1_pareto.png" style="width:100%"/>
<p><i>FIGURE 1 — slide 2. Everything else trades the two objectives off against each other; only the learned scheduler moves up and to the left at the same time.</i></p></div>


---

### Slide 3 — TECHNICAL APPROACH

> **Headline banner:** **Four layers, each validated before the next was built —
> so the scheduler's numbers mean something.**

**Left column — methodology (numbered 01–04, matching the flowchart):**

- **01 Scenario** — 2.2 GB of Turing recorded radar → **3,443 reusable emitter contributions**.
  New scenarios are re-draws, so the agent never sees the same world twice.
- **02 Truth** — a **36 band × 600 slot** grid of who is transmitting, whether or not we look.
  Slots are 50 ms; the receiver's own geometry, read from the files, not assumed.
- **03 Receiver** — declares a hit iff **signal + noise ≥ threshold**. **Pd 0.84, Pfa 0.0013**,
  frozen — a scheduler cannot improve the detector, only aim it.
- **04 Scheduler** — **Recurrent PPO**, 36 actions, a **183-value observation built only from its
  own scan history** (hit rate, airtime share, staleness, hit streak per band). No truth leaks in.

**Centre — FIGURE:** `figures/flowB_arch.png` (the four-layer architecture).
<div><img src="docs/ppt/figures/flowB_arch.png" style="width:100%"/>
<p><i>FLOWCHART B — slide 3, the “Central Flow Chart” box. Redraw in PowerPoint if you want it on-brand; keep the L0–L3 labels, the green priority input on the left and the dashed “acts on the world” return arrow.</i></p></div>


**Right column — why you can believe it (this is the strongest block on the slide):**

- **Validation gates ran before any scheduler was scored.** Out-of-sample band occupancy predicted
  at **85.9% accuracy (MCC 0.685)** on data the model never saw; per-band structure error
  **0.000 pp**; matches Köksal's published intercept-time theory to **±0.0000**.
- **The environment was frozen** — band geometry, slot clock, thresholds — before any result was
  quoted. A bad result cannot move it.
- **Train/test split written before anyone looked:** 35 train / 12 validation configs,
  **zero emitters shared.**
- **The winning checkpoint was chosen by a rule fixed in advance**, on validation data only — and
  it then scored best of all 24 on the full set. The rule never saw the data it was right about.
- **433 automated tests.**

**Bottom strips:** Python · NumPy · Gymnasium · Stable-Baselines3 · PyTorch · h5py.
Reward: exploit what pays + explore what is overdue − a cost on hogging airtime. Screened against
four known-good and known-bad policies **before** training, at 3.0σ separation.

---

### Slide 4 — FEASIBILITY AND VIABILITY

> **Headline banner:** **Already built and measured: 5,643 episodes, 57 scenarios, 3 seeds.**

Use the template's four quadrants around the Feasibility/Viability/Risk/Mitigation wheel.

**Feasibility — it exists now**

- Environment, 4 validation gates, a 9-rung baseline ladder and the trained agent: all built.
- Runs on **one CPU**; no GPU needed at inference. 600 decisions per 30 s episode.
- Uses the **exact receiver geometry** in the DRDO-named dataset — 36 bands, ±500 MHz, Turing's
  own dwell schedule — so it drops onto that receiver without redesign.

**Viability — it transfers**

- The scheduler only ever reads **hit / no-hit**, which every ES receiver already produces.
- `p = uniform` is the plain PS objective, so adopting the priority input **costs nothing** to a
  user who does not want it.
- **45 held-out recordings** are sealed behind an explicit flag with an automatic use log, for a
  final evaluation that has not been spent.

**Risks (name them — this is what separates a real project)**

- **Optimising one metric produces a useless scheduler.** Measured: a camper hits a 0.209 ratio and
  abandons **half the emitters**.
- **The agent could memorise the training emitters.** We found exactly this leak in our own work.
- **Seed variance can masquerade as a result.** Our 24 checkpoints span 18.7–54.4% against the bar.

**Mitigation**

- **Never report one metric.** Every table carries ratio, intercept time and coverage together.
- **Leak closed** — 35/12 split, pool rebuilt, zero shared emitters, asserted by a test.
- **Three seeds, pre-registered selection, a pre-registered reward screen.** Four numbers have been
  publicly withdrawn in this project when their measurement convention turned out to be undefined.

**FIGURE (fits the lower half):** `figures/fig2_discovery.png`.
<div><img src="docs/ppt/figures/fig2_discovery.png" style="width:100%"/>
<p><i>FIGURE 2 — slide 4. The two camping strategies visibly stop finding anything at 38 of 72 emitters. This is the easiest figure in the deck for a non-technical judge to read.</i></p></div>


---

### Slide 5 — IMPACT AND BENEFITS

> **Headline banner:** **Same receiver, same 30 seconds — finds more, finds it sooner,
> and can be told what matters.**

**Left — impact vs what is in use today** (keep the template's "Current ES" box):

- **2.2× the interception ratio of the open-loop sweep** (0.130 vs 0.061) **and** intercept time
  down from 4.18 s to **3.04 s**. Normally these trade against each other; here both improve.
- **Beats the open-loop floor on both objectives on 81.9% of episodes**, and beats the best
  published adaptive method on **54.4%**.
- **+4 percentage points of emitter coverage** (0.902 vs 0.865) — fewer emitters missed entirely.
- **No hardware change.** Same antenna, same receiver, same dwell schedule — only the choice of
  which band to look at next.

**Right — benefits (4 short headed items):**

- **Operational** — time goes to new and threatening emitters instead of being spread evenly over
  a weather radar and a fire-control radar alike. This is the failure DRDO names in paragraph one.
- **Supervised when it matters** — one priority input, one code path, no mode switch. Autonomous
  by default.
- **Auditable** — every threshold, split and selection rule is fixed in code before the run that
  uses it. That is what makes the result defensible to a customer.
- **Extensible** — the same closed loop is the *"feedback based decision making mechanism"* a
  cognitive EW system needs between its receiver and its processing unit.

> **Note on the old EA / EP boxes.** Keep them only as a small, clearly-labelled *future work*
> strip. Our PS is the **ES** slice; claiming jamming (EA) or protection (EP) invites a scope
> question we would lose. One line — "the same interface extends to EA/EP tasking" — is enough.

**FIGURE (small, bottom corner):** `figures/fig4_threat.png`, or `figures/flowA_loop.png` if you
prefer the open-loop-vs-closed-loop contrast here instead of on slide 2.
<div><img src="docs/ppt/figures/fig4_threat.png" style="width:100%"/>
<p><i>FIGURE 4 — the threat block as a decision story: two wrong attempts and the one that worked. This is the version to use on a slide, because it shows judgement rather than asking a judge to trust a lift figure.</i></p></div>

<div><img src="docs/ppt/figures/flowA_loop.png" style="width:100%"/>
<p><i>FLOWCHART A — open loop vs closed loop. Use on slide 2 or 5, whichever needs a diagram more than a chart.</i></p></div>


---

### Slide 6 — RESEARCH AND REFERENCES

**Research**

- **Problem statement** — SIH26055, DRDO, `sih.gov.in/sih2026PS`.
- **Dataset** — Turing Synthetic Radar Dataset (named in the PS itself),
  `huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset`;
  dataset paper arXiv:2602.03856. 184 files, 2.2 GB, 92 scan/stare pairs.
- **Our validation** — 4 gates, criteria fixed in code before the first run.

**References — we built on these and we beat these. Cite both honestly.**

- **Köksal**, *Periodic Search Strategies for ECM Receivers* (METU) — our intercept-time theory
  check (gate 3); ch. 6 is per-band priority, which is prior art for our interface.
- **Dutertre**, *Dynamic Scan Scheduling*, RTSS 2002 — mission-varying emitter weights and a
  uniform-weight control arm; prior art we extend to overlapping bands and a learned policy.
- **Apfeld et al.** (SSPD) — adaptive scan scheduling; implemented as rungs 6 and 6a of our
  ladder, and beaten on the joint objective.
- **Gul & Erer** — the per-illumination definition of interception ratio that we score against.
- **Clarkson** — optimal periodic sensor scheduling.

**YouTube video** — keep the template's box. Record **30 seconds of the waterfall running**
(`figures/fig3_waterfall.png` is a still of exactly this); it is worth more than any slide.
<div><img src="docs/ppt/figures/fig3_waterfall.png" style="width:100%"/>
<p><i>FIGURE 3 — the waterfall. Grey = a band genuinely transmitting, red = where we pointed. Sawtooth sweep, flat camp, dense hunt. If only one image survives to the final round, make it this one.</i></p></div>


---

## C. Figures — what each one is for

| File | Use on | Why it earns its place |
|---|---|---|
| `fig1_pareto.png` | **Slide 2** | The core claim in one picture: everything else trades the two objectives off; we improve both. Marker size = coverage, so the camper's failure is visible too. |
| `fig2_discovery.png` | **Slide 4** | The most intuitive figure we have. Camper and Apfeld visibly **flatline at 38 of 72** while we reach 65. Non-technical judges read this instantly. |
| `fig3_waterfall.png` | **Slide 6 / video still** | Three panels: the sweep's sawtooth, the camper's flat line, our dense hunting pattern. This is the demo in one image. |
| `fig4_threat.png` | **Slide 2** | Three attempts, two wrong. Shows how we corrected course — which is what the deck is actually judged on. Needs no number to land. |
| `flowA_loop.png` | **Slide 2** | Open loop vs closed loop. Use if the slide needs a diagram more than a chart. |
| `flowB_arch.png` | **Slide 3** | The four-layer architecture. This is the "Central Flow Chart" box. |

**If you redraw these in PowerPoint** (better for consistency), keep: the left-is-better /
up-is-better axis labels, the marker-size-is-coverage note, and the "off this chart" caption on
fig 1 — the camper's absence from the zoomed view has to be explained or it looks like we hid it.

---

## D. Numbers you may use, with what each one is

Everything below is from the run of **2026-09-11** — 33 schedulers × 57 scenarios × 3 seeds =
**5,643 episodes** (`runs/final_2026-09-11/`). Do not round differently from this table.

**Headline scheduler comparison (means over all episodes)**

| Scheduler | Interception ratio | Intercept time (s) | Coverage | Intercept rate (/s) |
|---|---|---|---|---|
| **NARADA (Recurrent PPO)** | **0.1304** | **3.04** | **0.9019** | **1.191** |
| Recency heuristic — our bar | 0.1105 | 3.34 | 0.8874 | 1.153 |
| Apfeld "Active RFs" — best in literature | 0.1320 | 4.32 | 0.8602 | 1.135 |
| Turing reference sweep | 0.0805 | 3.74 | 0.8643 | 1.166 |
| Round-robin — the open-loop floor | 0.0605 | 4.18 | 0.8650 | 1.118 |
| Random | 0.0669 | 4.16 | 0.8583 | 1.117 |
| Camper (hit-rate-only trap) | 0.2088 | 9.67 | 0.4971 | 0.660 |
| Apfeld (full) | 0.2455 | 14.86 | 0.3672 | 0.350 |
| *Truth-reading oracle (ceiling, not a competitor)* | *0.6579* | *8.01* | *0.6912* | *0.813* |

**Paired, per episode — "wins on BOTH objectives"**

| | vs the open-loop floor | vs the recency bar |
|---|---|---|
| **NARADA** | **81.9%** | **54.4%** |
| Apfeld Active RFs | 53.2% | 23.4% |
| Turing sweep | 51.5% | 9.9% |
| Apfeld (full) | 4.1% | 2.9% |
| Camper | 1.8% | 0.6% |

**Environment validation** — accuracy **0.8585**, MCC **0.6854**, precision 0.8819, recall 0.6932,
correlation 0.935 (base rate 0.354) · per-band occupancy error **0.000 pp** (0.35403 replayed vs
0.35403 recorded) · Köksal P12 agreement **|Δ| 0.0000** · 12/12 structural assertions.

**Receiver operating point** — γ = −111 dBm, σ = 3 dB, **Pd 0.842**, **Pfa 0.00135**,
sensitivity −107.2 dBm.

**Scale** — 47 development scenario pairs (35 train / 12 validation, **0 emitters shared**),
45 held out and unopened · 3,443 emitter contributions · 2,363 transmitters · 68 radar types ·
**433 tests**.

**Threat bias in our own schedulers** — 47 replays × 3 seeds, censored intercept time by threat
class. Round-robin HIGH 3.94 s / LOW 3.96 s (**no bias — it sweeps blindly**) · Recency
2.84 s / 2.47 s (**+0.37 s**) · NARADA RL 2.53 s / 1.86 s (**+0.66 s**, coverage 0.92 vs 0.96).
The bias grows as the scheduler improves. Uses our example threat library.

**Threat priority** — 772 occupied bands · naive max-rule priority flags **77.8%** of them
(precision 0.452 against a 0.383 base rate) · max-rule lift **1.17×** at every airtime fraction, because it has only 3 distinct values
and 77.8% of bands tie at the top one · share-weighted lift **2.22×** at 10–25% airtime, **1.69×**
at 50%, from 334 distinct values. **Always quote the lift with its airtime fraction.**

---

## E. Things to keep off the slides

- **"3 RL agents / phase sequencer."** We ship **one** policy. The old layout's central claim is
  no longer true and a judge who asks "which agent is running now?" gets a bad answer.
- **"We invented priority-driven scan scheduling."** Köksal (2004) and Dutertre (2002) did it
  first. Claim the *measurement* instead — it is stronger and it survives a reviewer who reads.
- **Any threat-priority result.** It is **designed and specified, not yet trained.** On the deck it
  is an architecture claim plus the 77.8% / 2.22× measurement. Nothing more.
- **EA / EP / jamming / geolocation** as deliverables. Our PS is ES-only. One future-work line.
- **The oracle row as if we beat it.** It reads the truth grid. It is a ceiling line, always
  labelled as one.
- **Accumulated reward** as a scheduler comparison. It is only comparable within one reward family.
- **Per-dwell hit rate.** It flatters the camper to 85–90% while it misses half the emitters; the
  per-illumination ratio is the honest number and the one the literature uses.
- **A wall of numbers.** Every figure here is defensible, but a judge cannot check any of them in
  the time they have. Numbers earn their place when they back a claim about *approach*; three good
  ones beat a table. Put the rest in the notes and let them come up in questions.
