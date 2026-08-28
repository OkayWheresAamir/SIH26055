# SIH Project Documentation Pack

## Human understanding

`SIH_RF_Brainstorming_Teammate_Guide.pdf`

This is the teammate guide. It is deliberately based on the brainstorming discussion only.

## Claude Code research handling

`CLAUDE_CODE_RESEARCH_PROTOCOL.md`

Rules for using the existing research/strategy/reference documents.

`RESEARCH_MAP_TEMPLATE.md`

Template for the repository's actual research inventory.

`CLAUDE_MD_UPDATE_PROMPT.md`

Prompt to give Claude Code to audit the existing docs and wire the research map/protocol into the existing Claude instructions.

`CLAUDE_MD_SUGGESTED_SECTION.md`

Suggested minimal section for an existing `CLAUDE.md`.

## Important separation

Do not merge the human brainstorming guide with the research knowledge base.

The two serve different purposes:

```text
Brainstorming guide
    ↓
helps humans understand the architecture

Research knowledge base
    ↓
helps Claude Code + team make informed implementation decisions
```
