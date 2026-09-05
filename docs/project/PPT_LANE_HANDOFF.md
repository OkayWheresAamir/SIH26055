# PPT Lane — Handoff

**Version 1 · valid as of 2026-09-05.** For the 2 people building the deck.
**Owner of this file:** Aamir. **Regenerate the PDF** with
`python -m scripts.md2pdf docs/project/PPT_LANE_HANDOFF.md`.

> Slots marked `[ ] PENDING` are content that does not exist yet. Each says **who** closes it and
> **when**. Build the slide now with the slot visible — an empty box with a label is a task; an
> empty box without one is a hole you find at 2 a.m.

---

## 0. What you are presenting, in judge language

> Radar receivers can only listen to one narrow slice of the spectrum at a time, but they have to
> watch all of it. Today they sweep on a fixed schedule set before the mission — so they spend the
> same time on an empty band as on the one where a new threat just appeared. We built a simulator
> of that problem grounded in a real radar dataset, proved the simulator is faithful, measured
> seven classical scheduling strategies on it, and trained a scheduler that learns where to listen
> from nothing but its own hits and misses.

That is the pitch. Everything else on every slide is evidence for one of those five clauses.

**The problem statement is DRDO's, ID 26055, "Smart Scan strategy for Electronic Warfare."** Its
expected solution, verbatim: *"Machine learning based Electronic Support receiver scheduler
software."* Quote that line on the deck — it proves you read the ask.

---

## 1. Before anything else — two blocking checks

**1. Get the official SIH 2026 PPT template and follow it exactly.** Slide count, order and
format are scored, and the playbook's submission checklist lists *"follows the official SIH PPT
template exactly"* as a pass/fail item. **I have not verified what the 2026 template contains** —
it is not in this repository, and nothing here should be treated as a substitute for it. If the
official template conflicts with the structure in §3 below, **the official template wins** and §3
becomes a content checklist to redistribute across its slides.

> [ ] **PENDING — official template downloaded and its slide list written into §3.** Owner: PPT lane.
> **Day 1, before any slide is designed.**

**2. There are two artefacts, not one, and they have different jobs.**

| | Idea-submission deck | Finale / demo deck |
|---|---|---|
| audience | screening, seconds per slide | live judges, questions |
| governed by | the official template, strictly | you |
| optimised for | *is this real and is it feasible* | *do we believe the numbers* |
| our strength | the measured baseline table | the live demo + the "what didn't work" slide |

Build the idea deck first. The finale deck reuses its slides plus depth.

---

## 2. Work split for two

| Role | Owns |
|---|---|
| **P1 — Narrative & design** | the story order, the one-sentence claim, slide layout, visual consistency, the script |
| **P2 — Evidence & figures** | every number and figure on every slide, traced to a file and a section; the demo video; the reference slide |

**The rule that makes this work: P2 has veto over any number P1 wants to say.** Every figure on
the deck must come from `EVALUATION.md`, `DECISIONS.md`, or a `runs/` artefact — never from
memory, never from an earlier draft of the deck. This project has already withdrawn four published
figures that were repeated from documents instead of re-measured (D33, D41 and two in the
2026-09-03 audit). **Do not become the fifth.**

Both of you rehearse. The playbook's own readiness test: *any* member, not just the presenter, can
walk a stranger through problem → solution → demo in under two minutes.

---

## 3. Slide-by-slide

Structure follows the deck order in `docs/reference/PPT/SIH_2026_Playbook_TechDoodles_Final.pdf`
§08 (p.11), reordered where our project's strength differs from a generic one — **our evidence is
unusually strong, so results move earlier than the playbook suggests.**

### Slide 1 — Title & team
PS **26055**, "Smart Scan Strategy for Electronic Warfare", DRDO. Team name, institution, member
names and roles.

> [ ] **PENDING** — team name, member list. Owner: Aamir.

### Slide 2 — The problem, made physical
Not a paragraph of text. **One diagram**: 36 bands stacked vertically, time across, emitters
lighting up, and a single receiver cursor that can only be in one place at a time. The instant
someone sees that picture they understand the problem.

Say: the receiver's instantaneous bandwidth is *an order of magnitude smaller* than the spectrum
it must cover (PS, Background). Today's strategies are **open loop** — fixed before the mission —
and by the PS's own words *"may lose time to nonthreatening emitters by not giving time to new or
threatening ones."* **Quote the PS here.**

*Source:* `SIH26055_PROBLEM_STATEMENT.md`. *Figure:* build from `runs/baselines/timeline_config_2.png`,
which is exactly this picture with real data in it.

### Slide 3 — Proposed solution, in one sentence
One sentence, large. Then three bullets: **a validated simulator**, **a measured baseline ladder**,
**a learned scheduler**. Nothing else on this slide.

> [ ] **PENDING** — the one sentence, and the tagline. Owner: Aamir + the scheduler brainstorm.
> Closed by end of RL Day 2. See `RL_LANE_HANDOFF.md` §9 for the four candidates and the method.

### Slide 4 — Innovation & uniqueness
**This is the slide that decides whether you stand out**, and it is currently the emptiest.

The candidates, with the number that backs each — pick **one headline and one supporting**, do not
list all four:

| Angle | The number behind it | Status |
|---|---|---|
| **The operator's dial** — the two objectives provably conflict, so we ship a front, not a policy | D28: censored intercept time needs `Y=1`, interception ratio does not — no single reward serves both | needs a preference-conditioned agent |
| **Learning to see what the receiver can't measure** | truth-fed camper 0.568 vs observation-fed 0.209 — a **0.359 gap** that is the price of invisible information (D46) | measured ✅ |
| **We can prove our simulator is right** | 4 gates, criteria fixed *before* the run and asserted by a test (D39); D42 publishes the gates' own blind spots | measured ✅ |
| **What didn't work** | Apfeld's published period estimation *hurt*: 1.9× ratio for 3.4× the intercept time (D45) | measured ✅ |

> [ ] **PENDING** — the chosen angle. Owner: Aamir + brainstorm. **Blocks slides 3, 4 and 12.**

### Slide 5 — How it works
The loop, as a diagram, four boxes: **observe** (own scan history only) → **decide** (which band)
→ **listen** (one dwell, 50 or 100 ms) → **learn** (hit or miss). Then the line that makes it
credible to anyone technical:

> The agent sees **only what a real receiver would see** — its own hits and misses. Ground truth
> exists in the simulator, and it is used to *score* the agent, never to *inform* it.

That asymmetry (D29) is a real engineering discipline and it is worth 20 seconds of airtime.

### Slide 6 — Technical approach & architecture
The four-layer stack, one line each: **L0 scenario** (emitters drawn from real recordings) →
**L1 truth grid** (36 bands × 600 slots, who is transmitting when) → **L2 receiver** (a detector
with a frozen threshold and a measured ROC) → **L3 agent interface** (a standard Gymnasium
environment).

Stack: Python, NumPy, HDF5, Gymnasium, PyTorch/Stable-Baselines3, Matplotlib. **229 automated
tests.** Say the test count — almost no SIH deck can.

*Source:* `ENVIRONMENT_SPEC.md`, `rfenv/` module docstrings.

### Slide 7 — The data is real, and here is how we used it
The Turing Synthetic Radar Dataset: **184 HDF5 files, 2.2 GB, 47 train + 45 held-out scenario
pairs.** The held-out 45 are touched **once, at the end** (D8).

Two facts that show you actually opened the files rather than reading a dataset card:
- **19% of transmitters in the metadata are never detectable at all** — scoring a scheduler for
  missing those would be meaningless, so our coverage denominator excludes them (D27).
- **A scan recording contains only the pulses the dataset's own sweeping receiver was tuned to.**
  Benchmark on one and any sweeping scheduler scores 0.9999 — we caught that and excluded scan
  replays from every comparison (D36).

The second one is a *strong* slide. It says: we understood our data well enough to find the trap
inside it.

### Slide 8 — Results: the baseline ladder
**The table from `EVALUATION.md` §5**, plus `runs/baselines/pareto.png`. State beneath it:
**1,539 episodes, 3 seeds, γ = −111 dB, 2026-09-04.**

Three rules, non-negotiable:
1. **Never show interception ratio alone.** Ratio, censored intercept time and coverage together,
   always (D14). A camper scores 0.209 and looks great until 9.67 s and 0.497 coverage appear.
2. **Label the two oracle rows as reference lines** wherever they appear. They read the truth grid.
   Read as competitors they invert the meaning of the whole table.
3. **The bar is rung 5, not round-robin.** Say so. Claiming victory over a weak baseline is the
   fastest way to lose a technical judge.

> [ ] **PENDING** — the **rung 7 row**. Owner: RL lane, end of Day 3. Build the slide with the row
> present and empty; do not restructure it later.

### Slide 9 — Results: the agent
Before/after: rung 5 vs rung 7. `pareto.png` with rung 7 marked, and `discovery_config_921.png`
(distinct emitters found against time — the most intuitive figure we have).

> [ ] **PENDING** — all numbers. Owner: RL lane, end of Day 3.

### Slide 10 — Feasibility, and why you should believe the numbers
Feasibility here is not "can it be built" — it is built. It is **"is the simulator faithful?"**,
and that is our strongest and least-copyable slide:

- **Gate 1** — build truth from *stare* recordings only, replay the *scan* schedule, predict against
  the real scan recordings. Data never used in construction. **Accuracy 0.859, MCC 0.685** over
  23,594 dwells.
- **Gate 2** — pipeline self-consistency, per band. **0.000 pp** deviation.
- **Gate 3** — a controlled periodic case against published closed-form theory (Köksal). **Exact.**
- **Gate 4** — 12 structural assertions on the two extreme scenarios. **12/12.**

Then the line that separates us from every team that shows a validation slide:

> **Every pass criterion was written into the code before the first run, and a test asserts it
> against the record** (D39). And we publish what the gates *cannot* detect (D42) — including that
> no gate covers the receiver's band half-width.

**Also state the known limitation, on the slide:** band 0 is 59% occupied in the recordings and 0%
predicted, because the stare recordings cannot see below 500 MHz (D10). Naming your own limitation
before a judge finds it converts a weakness into evidence of rigour.

### Slide 11 — Impact & scalability
Who benefits, and the measurable outcome: a receiver that finds threats sooner, at no hardware
cost — it is a **scheduling policy**, deployable as software on the receiver that already exists.
Scalability: nothing in the method is specific to 36 bands or to this dataset.

> [ ] **PENDING** — impact framing. Owner: P1. **Keep it honest.** Differentiation-doc §28: do not
> claim superiority over fielded or classified military systems. We have a simulator result, and
> that is a strong, defensible claim on its own.

### Slide 12 — What we tried that did not work
One slide. Four bullets, from `DECISIONS.md`:

- **D45** — we implemented a published adaptive method; its period-estimation half made things
  worse here, and lost to its own ablation.
- **D44** — that paper's Algorithm 1 contradicts its own prose. We implement the prose and say so.
- **D43** — two of our own baselines turned out to be the same policy. We found it and split them.
- **D41 / D33** — we withdrew our own published figures when their measurement convention turned
  out to be unrecorded.

Most decks cannot fill this slide. It is disproportionately persuasive — it is the difference
between a team that got a number and a team that understands its number.

### Slide 13 — Team & roles
Why this team can execute *this* PS.

> [ ] **PENDING** — owner: Aamir.

### Slide 14 — References
The PS, the Turing dataset, the two scheduling papers (`optimumsearch.pdf` — Köksal;
`paperSSPD (1).pdf` — Apfeld et al.), Gymnasium. Cite the papers properly — we adapted one and
contradicted the other, and both are named honestly.

---

## 4. Figures that already exist

All in `runs/baselines/`, regenerated by one command:

```bash
python -m rfenv.compare --seeds 3 --sampled 10 --figures --out runs/baselines
```

| File | What it shows | Best slide |
|---|---|---|
| `pareto.png` | interception ratio vs censored intercept time, one marker per rung, marker area = coverage | 8, 9 |
| `timeline_config_2.png` | band vs time with occupancy, the tuning path, hits, and a per-emitter strip that turns solid at first intercept | 2, 5 |
| `discovery_config_921.png` | distinct emitters found against time, one line per rung, with the ceiling | 9 |
| `timeline_config_81.png` | the 1-emitter extreme — good for explaining the mechanism slowly | 5 |
| the waterfall (`render.waterfall`) | the 36×600 grid as a frequency-vs-time heatmap — the standard ESM operator view | 2, 7 |

`runs/` is **gitignored** — these are rebuilt, not committed. Regenerate before exporting slides so
the figures match the numbers beside them.

**Design note:** they are Matplotlib defaults. They are honest but not presentation-grade. Restyle
for the deck (larger fonts, fewer gridlines, one accent colour), **but never redraw a figure by
hand or in a spreadsheet** — a hand-drawn chart can disagree with the table beside it, which is
precisely why `render.py` draws all of them from the same artefacts.

> [ ] **PENDING** — a diagram of the 4-layer architecture for slide 6. Owner: P1. Not auto-generated;
> draw it.

---

## 5. Traps

1. **Never quote a number from an earlier draft of the deck.** Re-open the source. Four published
   figures in this project have already been withdrawn for exactly this.
2. **State the operating point** (γ = −111 dB, P_fa = 1.35e−3) beside any scheduler table, and name
   which population P_d was averaged over — it is not data-independent (D33, D46).
3. **Do not say "accuracy"** as a headline. With sparse occupancy, "predict nothing" scores well.
4. **Do not call the oracle rows baselines.**
5. **`RL_LANE_HANDOFF.pdf` and `.html` in this folder are stale.** Ignore them; the `.md` files are
   the source.
6. **Nothing from `../SIHProto`.** Not a number, not a diagram, not a slide.
7. **Don't oversell.** We have a validated simulator and a measured comparison. That is genuinely
   more than most entries have, and it survives questioning — which a bigger claim will not.

---

## 6. Timeline

| Day | P1 (narrative) | P2 (evidence) |
|---|---|---|
| **1** | official template downloaded; slide skeleton with every [ ] visible | pull every existing number into one sheet with its source; regenerate figures |
| **2** | slides 1, 2, 5, 6, 7 fully drafted | slides 8, 10, 12 built from `EVALUATION.md` and `DECISIONS.md` |
| **3** | slides 3, 4, 11 once the edge is chosen | rung 7 numbers land → slides 8, 9 |
| **4** | full pass for consistency and one visual language | **record the backup demo video locally**; verify every number against its source once more |
| **5** | two timed rehearsals, both members presenting | held-out results land → final number pass |
| **6** | buffer | buffer |

---

## 7. Submission checklist

From the playbook (p.12), kept only where it applies to us:

- [ ] Follows the official SIH PPT template exactly — slide count, order, format
- [ ] Every slide answers one clear question; no filler slides
- [ ] Core end-to-end flow works, even if unpolished
- [ ] **Backup demo video recorded locally** — the playbook's most-cited live failure
- [ ] Rehearsed as a team at least twice, against a timer
- [ ] Every member can explain the PS without notes
- [ ] Every number on every slide traced to a file and a section by P2
- [ ] Every reference line labelled as a reference line
- [ ] Named beneficiary and measurable outcome present
- [ ] No claim of superiority over fielded or classified systems

---

## 8. Where the content lives

| Need | Go to |
|---|---|
| the ask, verbatim | `docs/project/SIH26055_PROBLEM_STATEMENT.md` |
| every metric, baseline, gate figure | `docs/project/EVALUATION.md` §4, §5, §6 |
| "what we tried and what changed our mind" | `docs/project/DECISIONS.md` — D33, D36, D41, D43, D44, D45, D46 |
| the architecture, in words | `docs/project/PROJECT_ARCHITECTURE.md`, `docs/project/ENVIRONMENT_SPEC.md` |
| what the RL lane is doing and when it lands | `docs/project/RL_LANE_HANDOFF.md` |
| a claim inside a reference PDF | `.venv/bin/python -m docsearch "your question" -k 8` — returns `file.pdf:p.7` |
| past winners' decks | `docs/reference/PPT/` — 7 of the 10 are image-only, open them visually |
