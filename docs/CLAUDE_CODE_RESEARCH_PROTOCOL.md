# Claude Code Research & Reference Protocol

## Purpose

The repository already contains research papers, strategy notes, dataset documents, and other reference material. This file defines **how Claude Code should use them**.

These documents are a knowledge base. They are **not automatic implementation instructions**.

## Core rule

> Reference material informs project decisions; it does not silently become project requirements.

Before using a document to change code, Claude should determine:
1. What does this source actually say?
2. What problem does it address?
3. What assumptions does it make?
4. Does it apply to our Turing + CEWS setup?
5. Is it needed now, later, optional, or reference-only?
6. Does another source disagree?

If a conflict affects a major modeling or algorithmic decision, surface it instead of silently choosing.

---

## Document roles

Classify repository documents as one or more of:

- `DATASET` — Turing/data semantics.
- `ENVIRONMENT` — RF/simulation/environment construction.
- `VALIDATION` — equations, theory, expected behavior, validation methods.
- `BASELINE` — conventional scheduling methods.
- `RL` — reinforcement-learning/adaptive scheduling methods.
- `EVALUATION` — metrics and experimental protocols.
- `BACKGROUND` — useful context, not a direct implementation source.
- `FUTURE` — interesting but not part of the current milestone.
- `UNRESOLVED` — needs human/domain review.

Also classify when they should be used:

- `NOW`
- `LATER`
- `FINAL EVALUATION`
- `REFERENCE ONLY`

---

## When to read what

### Turing/data understanding
Prioritize `DATASET` documents.

Use them to understand:
- fields,
- file structure,
- metadata,
- scan/stare meaning,
- observations,
- known dataset limitations.

### Environment construction
Prioritize:
- `DATASET`
- `ENVIRONMENT`

Consult `VALIDATION` documents when they provide equations or expected behavior relevant to a modeling choice.

Do not copy a paper's assumptions into our simulator without checking that they fit our setup.

### Environment validation
Prioritize:
- `VALIDATION`
- relevant `DATASET`

Use:
- controlled theory,
- sanity checks,
- reference observations,
- reproducible comparisons.

Do not tune the environment simply until it produces a desired result.

### Scheduler/baselines
Prioritize:
- `BASELINE`
- `EVALUATION`

### RL/adaptive scheduler
Prioritize:
- `RL`
- `EVALUATION`

Do this after the environment is sufficiently validated.

### Final evaluation
Prioritize:
- `EVALUATION`

Do not use held-out test results for tuning.

---

## Authority hierarchy

Use the strongest applicable source:

### A — Direct project/dataset facts
Actual repository code and observed Turing file contents.

### B — Reputable research/theory
Useful for equations, known methods, and experimental expectations. Check assumptions.

### C — Proposed strategy
Useful as a candidate design or hypothesis. Not automatically correct.

### D — Brainstorm/commentary
Useful for ideas and context, not ground truth.

---

## Conflict protocol

If sources disagree:

1. State the exact disagreement.
2. Identify which source supports each claim.
3. Check the actual dataset/code if possible.
4. Decide whether one source has higher authority for this question.
5. If still ambiguous, ask the human/domain expert.
6. Record important decisions in `docs/DECISIONS/`.

Never silently merge incompatible assumptions.

---

## No research-driven scope creep

A document can contain a good idea that is irrelevant to the current milestone.

Do not implement it merely because it exists.

Mark it:
- NOW
- LATER
- OPTIONAL
- REFERENCE ONLY

---

## Human review gates

Ask for human/domain review when a proposed interpretation changes:
- RF/environment behavior,
- receiver behavior,
- Turing metadata semantics,
- ground truth,
- reward definition,
- evaluation protocol,
- major scheduler architecture.

Routine coding choices do not need a meeting.

---

## Required source note when implementing research

For a nontrivial research-derived change, record:

```text
Source:
Relevant section/page:
What the source claims:
Assumptions:
Why it applies to our setup:
Implementation choice:
Validation:
```

Keep this concise.

---

## Golden question

Before applying a research idea, Claude should be able to answer:

> What does this source establish, what assumptions does it make, and why does that apply to our specific project?
