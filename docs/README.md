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
| `DECISIONS.md` | D1–D27 with status, evidence and two consistency audits. Read before proposing anything. |
| `RESEARCH_MAP.md` | Every document in `reference/` classified: what it establishes, what it does not, its authority level. |
| `RL_LANE_HANDOFF.pdf` (+ `.html` source) | Onboarding handoff for the RL lane: role split for three people, what is frozen, what is theirs to decide, what to learn. **Derived from the four files above — where they disagree, they win.** |

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
| `rfenv/receiver.py` | L2 — dwell mechanics, the noise draw, `Y` | not built |
| `rfenv/env.py` | L3 — gymnasium interface | not built |
| `rfenv/render.py`, `rfenv/metrics.py` | waterfall, episode log, metrics | not built |
| `rfenv/validate.py` | gates 1–4 as a runnable script | not built |

**No validation gate has been run yet.** `EVALUATION.md` §7 forbids quoting a scheduler number
before they pass.

## `teammate-work/`

A teammate's independently-derived Gymnasium environment (`rf_env_grounded.py` plus its write-up).
Kept as a **cross-check, not a base** — it arrived at a near-identical interface from the same PS
text, which is corroboration. Our environment is built fresh.

---

## Conventions

- **Cite the primary.** A summary of a source is not a second source. Where a summary and its
  primary are both here, the research map says so explicitly.
- **The files beat the documents.** For any fact about the data, `data/turing/**/*.h5` wins —
  including against the dataset's own README.
- **Paths are repo-root-relative** in all cross-references, e.g.
  `docs/reference/scheduling/paperSSPD (1).pdf`.
