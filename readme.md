# SIH26055

An interactive RF environment for evaluating receiver scheduling strategies, grounded in the
Turing Synthetic Radar Dataset, working toward a learned (RL) scheduler that decides which
frequency band a receiver should inspect next.

This repository is a **fresh start**. It currently contains source material and data only — no
code has been written, and no conclusions have been drawn. See `CLAUDE.md` for why, and for the
provenance rules that govern how anything gets added from here.

## Status

Stage: **understand the Turing data** (`docs/PROJECT_ARCHITECTURE.md` §10). Nothing downstream —
scenario construction, the environment, the scheduler, RL — has been built yet.

Two things are open and blocking real modelling decisions; see "Unresolved" in
`docs/RESEARCH_MAP.md`:
- what `sensitivity_dbm` actually means, given that ~4% of recorded pulses fall below it
- scan and stare are not nested in each other, despite the dataset card describing stare as an oracle

## Layout

| Path | What it is |
|---|---|
| `data/turing/` | The Turing dataset (94 HDF5 files, gitignored, already present locally). 47 scan/stare pairs, train split only. |
| `docs/PROJECT_ARCHITECTURE.md` | The working architecture — read this first. |
| `docs/CLAUDE_CODE_RESEARCH_PROTOCOL.md` | Rules for how the reference PDFs may and may not be used. |
| `docs/RESEARCH_MAP.md` | The actual inventory of every document in `docs/`, classified per the protocol, with what each one does and does not establish. |
| `docs/` (six PDFs) | Reference material — research papers, a thesis, and strategy notes. Authority varies; see the research map before citing any of them. |
| `CLAUDE.md` | Authoritative rules for this repository: source-of-truth table, provenance rules, working rules. |

## Working here

If you're picking this up with Claude Code, read `CLAUDE.md` and the two documents it points to
before doing anything else. The short version:

- The HDF5 files outrank every document, including the dataset's own README.
- Every factual claim needs a named source and a stated method ("verified by opening `config_2.h5`"
  vs. "the PDF asserts") — a number from a previous session or a summary of a source doesn't count.
- Architecture-shaping decisions (environment, receiver model, reward, evaluation protocol,
  scheduler design) get proposed with evidence and alternatives, not decided unilaterally.
- Nothing is imported from `../SIHProto` — that repository's code and numbers are deliberately
  not established fact here.
