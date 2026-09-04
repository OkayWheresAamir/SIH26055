# CLAUDE.md

## What this repository is

A fresh start on SIH26055. It holds the source material, the data, the decisions taken from
them, and the RF environment being built on top.

A previous attempt accumulated derived documents and measured claims that became difficult to
separate from their sources. This repository exists to avoid that. The rules below are the
whole point of it — follow them before doing anything else.

**Build status (2026-09-04).** `rfenv/` has L0 (`scenario.py`), L1 (`truth.py`), L2
(`receiver.py`), L3 (`env.py`) and the artefact layer (`metrics.py`, `render.py`) built, with
**110 passing tests** under `tests/`. Next and last in `docs/project/ENVIRONMENT_SPEC.md`
§Build order: `validate.py`. Gate 1's comparison convention is fixed in advance by **D37**;
**gates 2, 3 and 4 still need their pass criteria — D39, `OPEN`, and the last thing owed
before `validate.py` can be written.** They are fixed *before* the first run, not after: a
threshold chosen once the measurement is visible is not a gate.
**The four validation gates in `docs/project/EVALUATION.md` §6 have not been run yet** — no
scheduler number may be quoted until they have.

**A scan replay is not a scheduler-comparison scenario (D36, 2026-09-04).** A scan recording
holds only the pulses Turing's own sweeping receiver was tuned to, so a grid built from one
hands any sweeping scheduler its answer — measured, interception ratio 0.9999 and censored
intercept time 0.00 s. Compare schedulers on **stare replays and sampled scenarios**; scan
replays stay in gates 1 and 2, where that imprint is the mechanism under test.

The three questions L2 was blocked on are all answered: **D28** (what counts as intercepting an
emitter), **D29** (what the reward may read), **D31** (a dwell's reward is the sum of its slots').

**Third consistency audit run 2026-09-03** (`DECISIONS.md` §Consistency audit — 2026-09-03).
The decision spine D1–D31 holds with no contradictions. It found and fixed one implementation
mismatch (**D32** — the sampler could draw the same physical emitter twice) and withdrew two
figures whose measurement convention was never recorded: **`Pd = 0.822`** and **gate 1's
86.19%/71.14%**. Both are relabelled, not silently corrected. **`receiver.py` needs D33 decided**
(which cell population `Pd` averages over) before it can emit a ROC; **D34** ratifies the base
observation vector `env.py` builds to. Neither blocks starting.

## What is authoritative here

| Source | Authority |
|---|---|
| `docs/project/SIH26055_PROBLEM_STATEMENT.md` | **The requirement.** The official DRDO problem statement — what we are being asked to build. Settles scope disputes. |
| `data/turing/**/*.h5` | **Highest for facts about the data.** When anything disagrees with an observed field, the files win. |
| `docs/project/PROJECT_ARCHITECTURE.md` | The working architecture, written by the human team. Default direction; not immutable. |
| `docs/protocol/CLAUDE_CODE_RESEARCH_PROTOCOL.md` | How the reference documents may and may not be used. |
| `docs/project/ENVIRONMENT_SPEC.md` | The consolidated buildable spec for the RF environment. Follows from `DECISIONS.md`; read before building. |
| `docs/project/EVALUATION.md` | The single authority on metrics, baselines, validation gates and protocol. Do not redefine a metric anywhere else. |
| `docs/reference/dataset/TSRD_dataset_paper_arXiv_2602.03856.pdf` | How the data was generated. Primary source on dataset semantics; the HF dataset card is only a summary of it. |
| Everything under `docs/reference/` | Reference material. Classify before use, per the protocol. See `docs/project/RESEARCH_MAP.md`. |
| `rfenv/constants.py` | **The freeze list, as a file.** Band geometry, slot clock, `N₀`, `σ`, `γ`. Frozen once the gates pass; no result may move it (D25). |

`docs/` is organised by authority: `project/` is authored and governs the build, `protocol/` is
how we work, `reference/` is external material, `teammate-work/` is cross-check only. See
`docs/README.md`.

`rfenv/` is the environment, one module per layer of `ENVIRONMENT_SPEC.md`. Its module
docstrings carry the reasoning; the decisions themselves live in `DECISIONS.md`.

**Search `docs/` before re-reading it.** 366 pages of PDF are invisible to `grep`;
`docsearch` makes them searchable and returns a `file.pdf:p.7` citation with every hit, which
is the "Relevant section/page" the protocol asks for:

```bash
.venv/bin/python -m docsearch "alert confirm dwell time" -k 8
.venv/bin/python -m docsearch "how was the dataset generated" --primary   # skip our summaries
```

It answers `no strong match` when the corpus cannot support a claim, and lists the eight
image-only PDFs it cannot read (`--blind-spots`). See `docsearch/README.md`.

**When to search and when to open the file.** Search when you do not already know which
document holds the answer, and whenever the answer is likely in a PDF — those are unreachable
any other way. Read the file directly when you know which one it is and it is markdown;
`DECISIONS.md`, `ENVIRONMENT_SPEC.md` and `EVALUATION.md` are authoritative and greppable, so
go straight to them. **Either way, open the primary at the cited page before relying on a
number or an equation** — search locates a page, it does not read mathematics off it
(`T_rcv` extracts as `rcv \n T`). Prefer `--primary` when the question is what an external
source claims: our own records in `RESEARCH_MAP.md` and `DECISIONS.md` outrank the papers they
summarise in roughly a fifth of such queries.

Read `docs/project/PROJECT_ARCHITECTURE.md` and `docs/protocol/CLAUDE_CODE_RESEARCH_PROTOCOL.md` before
designing anything.

**`docs/project/DECISIONS.md` records every decision taken, with its evidence and status.** Read it
before proposing anything — a question marked `SETTLED` or `CLOSED` there does not get reopened
without new evidence, and one marked `PROPOSED` is waiting on a human, not on more research.
Add to it whenever a decision is made; that file is where project knowledge survives a session.

## Personal working preferences

Each of us keeps a gitignored `CLAUDE.local.md` at the repo root with our own working
preferences — how we want to be worked with, what we already know, what needs explaining.
Claude Code reads it automatically alongside this file. It is personal and never committed;
write your own rather than editing someone else's.

**Nothing in a `CLAUDE.local.md` overrides this file or the research protocol.** It covers
working style only — never provenance, authority, or the architecture gates below.

## Provenance rules

These are not style preferences. They are why this repository was recreated.

1. **Every factual claim must be traceable to a source you can name** — a specific HDF5 field,
   a specific page of a specific PDF, or a command whose output is in the transcript.
2. **State how you know.** "Verified by opening `config_2.h5`" and "as summarised in a
   document I have not checked against the primary" are different claims. Say which one you
   are making, every time.
3. **A measurement you did not run this session is not a fact.** Re-run it, or label it
   unverified. Never repeat a number because it appears in a document.
4. **A summary of a source is not a second source.** Cite the primary. If the primary is not
   in this repository, the claim is unverified — say so.
5. **Do not import anything from `../SIHProto`.** That repository's code, documents and
   numbers are deliberately not here. Nothing in it is established fact.

## Working rules

- **Do not decide the architecture unilaterally.** For anything that shapes the RF
  environment, the receiver model, ground truth, the reward, the evaluation protocol, or the
  scheduler design: propose it, give the evidence and the alternatives, and **wait for
  confirmation.** Routine implementation choices inside an agreed design do not need approval.
- **One path, not a menu.** Once something is decided, build it. Do not preserve rejected
  approaches as live alternatives, and do not hand back a decision the architecture already
  settles.
- **Keep it simple and writable.** Prefer a model that can be explained in a few sentences and
  implemented directly over one that is more faithful but nobody can hold in their head.
- **Naming.** Do not let an implementation detail become the project's identity.

## Data

`data/` is gitignored — 184 HDF5 files, 2.2 GB, already present locally. The dataset is the
Turing Synthetic Radar Dataset (gated on Hugging Face). Do not re-download it; it is here.

- **47 `scan`/`stare` pairs from the train split** — the development set.
- **45 `scan`/`stare` pairs from the test split** — held out, fetched 2026-08-28 by a rule fixed
  in advance (see `docs/project/RESEARCH_MAP.md`). **Do not touch these during development.** They exist
  so the final evaluation means something; every use must be recorded.
