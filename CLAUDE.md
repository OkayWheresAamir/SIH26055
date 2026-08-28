# CLAUDE.md

## What this repository is

A fresh start on SIH26055. It contains **source material and data only**. No code has been
written yet, and no conclusions have been drawn yet.

A previous attempt accumulated derived documents and measured claims that became difficult to
separate from their sources. This repository exists to avoid that. The rules below are the
whole point of it — follow them before doing anything else.

## What is authoritative here

| Source | Authority |
|---|---|
| `data/turing/**/*.h5` | **Highest.** The data itself. When anything disagrees with an observed field, the files win. |
| `docs/PROJECT_ARCHITECTURE.md` | The working architecture, written by the human team. Default direction; not immutable. |
| `docs/CLAUDE_CODE_RESEARCH_PROTOCOL.md` | How the reference documents may and may not be used. |
| The six PDFs in `docs/` | Reference material. Classify before use, per the protocol. |

Read `docs/PROJECT_ARCHITECTURE.md` and `docs/CLAUDE_CODE_RESEARCH_PROTOCOL.md` before
designing anything.

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

`data/` is gitignored — 94 HDF5 files, 1.1 GB, already present locally. The dataset is the
Turing Synthetic Radar Dataset (gated on Hugging Face). Do not re-download it; it is here.

The 47 `scan`/`stare` pairs are all from the **train** split. There is no held-out test data
in this repository yet.
