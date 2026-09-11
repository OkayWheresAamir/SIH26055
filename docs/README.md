# docs/

Everything the project knows, sorted by what it is. **Folder tells you how much authority a
document has** — that is the whole point of the layout.

## Read these first

| Order | File | Why |
|---|---|---|
| 1 | `project/SIH26055_PROBLEM_STATEMENT.md` | The requirement. Outranks everything else here. |
| 2 | `project/DECISIONS.md` | Every decision taken, with evidence and status. Settles what is already closed. |
| 3 | `project/ENVIRONMENT_SPEC.md` | The buildable spec for the RF environment. |
| 4 | `project/EVALUATION.md` | How anything gets measured. |

---

## `project/` — authored, living, governs the build

These are ours. They are kept current; if a chat changes a decision, it changes these.

| File | What it is |
|---|---|
| `SIH26055_PROBLEM_STATEMENT.md` | The official DRDO problem statement, verbatim, plus what it settles and leaves open. |
| `PROJECT_ARCHITECTURE.md` | The human team's working architecture. Direction, not immutable. |
| `ENVIRONMENT_SPEC.md` | Consolidated three-layer spec: scenario pipeline → truth → receiver → agent interface, plus build order. |
| `EVALUATION.md` | **Single authority on metrics, baselines, validation gates and protocol.** |
| `DECISIONS.md` | D1–D42 with status, evidence and three consistency audits. **D42 freezes the environment and records what the gates cannot detect.** Read before proposing anything. |
| `RESEARCH_MAP.md` | Every document in `reference/` classified: what it establishes, what it does not, its authority level. |
| `RL_LANE_HANDOFF.md` (+ generated `.pdf`) | Onboarding handoff for the RL lane: the rung-7 target, role split for two or three people, a day-by-day task list, the evaluation rules, and the edge candidates. |
| `PPT_LANE_HANDOFF.md` (+ generated `.pdf`) | Onboarding handoff for the presentation lane: slide-by-slide content with its source for every number, the figures that already exist, and the traps. |
| `THREAT_WEIGHTING_BRIEF.md` | **D63 as something the RL lane can build.** The scheduler takes a threat priority from outside and learns to use it; the three measurements showing why it cannot compute one itself; the example threat library; the order of work. |

The handoffs are **derived from the four files above — where they disagree, those win.** The `.md` is the source; regenerate the PDFs after any edit with
`.venv/bin/python -m scripts.md2pdf docs/project/RL_LANE_HANDOFF.md docs/project/PPT_LANE_HANDOFF.md`.

## `protocol/` — how we work

| File | What it is |
|---|---|
| `CLAUDE_CODE_RESEARCH_PROTOCOL.md` | Rules for using reference material. Reference informs; it does not silently become requirement. |
| `RESEARCH_MAP_TEMPLATE.md` | Template the research map was built from. |
| `CLAUDE_MD_SUGGESTED_SECTION.md` | Suggested wiring for an existing `CLAUDE.md`. |

## `reference/` — external material

**Not requirements.** Classify before use, per the protocol. `project/RESEARCH_MAP.md` records
what each one does and does not establish.

| Folder | Contents |
|---|---|
| `dataset/` | The TSRD paper — how the Turing data was generated. Primary source on dataset semantics; the HF dataset card is only a summary of it and is wrong in two places. |
| `PPT/` | Past SIH winning decks and two pitch guides. **Pitch material, not project material** — `docsearch` keeps it in a separate `presentation` collection so it cannot surface in a technical answer. Seven of the ten decks are image-only and must be read visually. |
| `scheduling/` | The core literature: Köksal (intercept theory), **Apfeld et al.** (adaptive SNR-based search — our strong baseline), Gul & Erer (RPCA/TPSR), Dutertre (dynamic scan scheduling), alert–confirm dwell optimisation, plus two teammate summaries. |
| `deinterleaving/` | Pulse-separation literature. **Out of scope** (D12/D19); kept for reference only. |
| `background/` | Domain grounding, EW landscape, strategy notes, a crash course. Useful for the write-up; low authority for design. |
| `problem-context/` | The iDEX ADITI 4.0 Cognitive EW challenge (p.12) and its official Q&A transcript — the sibling Army problem that grounds our intent reading. |

## The code it governs — `rfenv/`

Not part of `docs/`, but this is what the documents above exist to constrain. One module per
layer of `ENVIRONMENT_SPEC.md`; `tests/` mirrors it.

| Module | Layer | Status |
|---|---|---|
| `rfenv/constants.py` | the freeze list, as a file | built |
| `rfenv/scenario.py` | L0 — emitter contributions, pool, replay and sampled scenarios | built |
| `rfenv/truth.py` | L1 — the `Z`/`S`/`C` grid, detectable intervals | built |
| `rfenv/receiver.py` | L2 — dwell mechanics, the noise draw, `Y`, the ROC | built |
| `rfenv/env.py` | L3 — gymnasium interface, the three reward candidates | built |
| `rfenv/metrics.py` | episode log, emitter table, run header, `metrics.json` — §4 scored from the artefacts, not from the env | built |
| `rfenv/render.py` | waterfall, ROC, per-band bar. The only module importing matplotlib | built |
| `rfenv/validate.py` | gates 1–4 as a runnable script; the pass criteria live in its `GATES` dict | built |

**All four gates ran for the first time on 2026-09-04** (`python -m rfenv.validate`, 47 train
configs, seed 0): gates 2, 3 and 4 **PASS**, gate 1 is **MEASURED** — D37 fixed its convention and
left its threshold undecided. Criteria were fixed in code before the run and are asserted against
D39 by a test. `EVALUATION.md` §6 has the numbers; scheduler comparison is unblocked, and the
baseline ladder is the next thing owed.

**The environment was frozen on 2026-09-04 (D42)** and `tests/test_freeze.py` enforces it. Nothing
in `rfenv/constants.py` may move because a result came out badly; if one does, the environment is
re-validated from gate 1 and every baseline re-run (D25). Environment and pre-RL work is closed.

## `teammate-work/`

A teammate's independently-derived Gymnasium environment (`rf_env_grounded.py` plus its write-up).
Kept as a **cross-check, not a base** — it arrived at a near-identical interface from the same PS
text, which is corroboration. Our environment is built fresh.

---

## Searching all of this — `docsearch/`

The 366 pages of PDF in here are invisible to `grep`. `docsearch` extracts them and returns a
`file.pdf:p.7` citation with every hit, so the protocol's "Relevant section/page" is cheap
instead of effortful.

```bash
.venv/bin/python -m docsearch "alert confirm dwell time" -k 8
.venv/bin/python -m docsearch "how was the dataset generated" --primary
.venv/bin/python -m docsearch.corpus --blind-spots     # the 8 PDFs with no text layer
```

Measured recall@10 is 100% over 22 verified questions; it answers `no strong match` rather than
ranking the least-bad page. See `docsearch/README.md`.

## Conventions

- **Cite the primary.** A summary of a source is not a second source. Where a summary and its
  primary are both here, the research map says so explicitly. `docsearch --primary` restricts
  to external sources — worth using, since our own summaries outrank the papers they summarise
  in about an eighth of primary-source queries.
- **The files beat the documents.** For any fact about the data, `data/turing/**/*.h5` wins —
  including against the dataset's own README.
- **Paths are repo-root-relative** in all cross-references, e.g.
  `docs/reference/scheduling/paperSSPD (1).pdf`.
