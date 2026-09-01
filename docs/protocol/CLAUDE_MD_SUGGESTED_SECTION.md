# Suggested CLAUDE.md section

## Research and reference documentation

When a task depends on research papers, teammate strategy documents, dataset documentation, or other material under `docs/`, first consult:

- `docs/project/RESEARCH_MAP.md`
- `docs/protocol/CLAUDE_CODE_RESEARCH_PROTOCOL.md`

Use the research map to identify which sources are relevant to the current project stage.

Treat research documents as **references**, not automatic implementation instructions.

Before applying a research idea:
1. identify what the source actually claims,
2. identify its assumptions,
3. check whether those assumptions match our Turing + CEWS setup,
4. distinguish established facts from proposed methods,
5. surface conflicts between sources,
6. record important modeling decisions in `docs/DECISIONS/`.

Do not implement optional/future ideas unless the current task explicitly calls for them.

Do not silently resolve disagreements between sources. If a disagreement affects a major RF/environment/scheduler assumption, ask for human/domain review.
