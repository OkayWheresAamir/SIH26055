# SIH26055

An interactive RF environment for evaluating receiver scheduling strategies, grounded in the
Turing Synthetic Radar Dataset, working toward a learned (RL) scheduler that decides which
frequency band a receiver should inspect next.

## Status

Stage: **building the RF environment** (`docs/project/PROJECT_ARCHITECTURE.md` §10, build order
in `docs/project/ENVIRONMENT_SPEC.md`).

Built so far:
- `rfenv/constants.py` — the freeze list
- `rfenv/scenario.py` — L0: emitter contributions, pool, replay/sampled scenarios, held-out guard
- `rfenv/truth.py` — L1: the Z/S/C truth grid
- Tests for both, passing (`tests/`)

Next: `receiver.py` (L2 — the noise draw and `Y`), then `env.py` (L3 — gymnasium), then
`render.py` + `metrics.py`, then `validate.py` (gates 1–4). Nothing blocks `receiver.py`: the
three questions it raised are answered by **D28** (what counts as intercepting an emitter),
**D29** (the reward may read truth; the observation may not) and **D31** (reward is per slot, so
a 100 ms dwell is scored on both its cells).

Earlier open questions (`sensitivity_dbm` semantics, scan/stare non-nesting) are closed — see
`docs/project/DECISIONS.md` (D9, D10, D24).

One decision is still open and should be fixed **before** training starts: D29's selection rule
for choosing among the three reward candidates when they Pareto-dominate the baselines but not
each other. One is `PROPOSED` and awaiting a human: D30 (whether AoA and PulseWidth enter the
observation).

## Layout

| Path | What it is |
|---|---|
| `rfenv/` | The environment implementation, built incrementally per `ENVIRONMENT_SPEC.md`. |
| `tests/` | Tests for `rfenv/`. |
| `data/turing/` | The Turing dataset (184 HDF5 files, gitignored, already present locally). 47 scan/stare pairs (train, development set) + 45 held-out test pairs. |
| `docs/project/PROJECT_ARCHITECTURE.md` | The working architecture — read this first. |
| `docs/project/ENVIRONMENT_SPEC.md` | The consolidated buildable spec for the RF environment. |
| `docs/project/EVALUATION.md` | The single authority on metrics, baselines, validation gates and protocol. |
| `docs/project/DECISIONS.md` | Every decision taken, with its evidence and status. |
| `docs/project/RESEARCH_MAP.md` | Inventory of every document in `docs/`, classified per the research protocol. |
| `docs/protocol/CLAUDE_CODE_RESEARCH_PROTOCOL.md` | Rules for how reference documents may and may not be used. |
| `docs/reference/` | Reference material — research papers, a thesis, and strategy notes. Authority varies; see `RESEARCH_MAP.md` before citing any of them. |
| `CLAUDE.md` | Authoritative rules for this repository: source-of-truth table, provenance rules, working rules. |

## Working here

If you're picking this up with Claude Code, read `CLAUDE.md` and the docs it points to before
doing anything else. The short version:

- The HDF5 files outrank every document, including the dataset's own README.
- Every factual claim needs a named source and a stated method ("verified by opening `config_2.h5`"
  vs. "the PDF asserts") — a number from a previous session or a summary of a source doesn't count.
- Architecture-shaping decisions (environment, receiver model, reward, evaluation protocol,
  scheduler design) get proposed with evidence and alternatives, not decided unilaterally.
- Nothing is imported from `../SIHProto` — that repository's code and numbers are deliberately
  not established fact here.
