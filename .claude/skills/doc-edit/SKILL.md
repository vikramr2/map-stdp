---
name: doc-edit
description: House rules for editing docs/derivation.md, docs/SPEC.md and CLAUDE.md in this repo — equation numbering, Proposal markers, Appendix A/B bookkeeping, cross-references, math rendering and the changelog. Use for any non-trivial edit to those docs.
---

# Editing the project docs

## Which file

| File | Holds |
| --- | --- |
| `docs/derivation.md` | **Canonical** math: rules, theorems, assumptions. |
| `docs/SPEC.md` | Goals, design considerations (Direction / **Proposal** / **Open**), metrics, ablations, milestones. |
| `docs/CHANGELOG.md` | One dated entry per change, decisions and reversals included. Use the `/changelog` skill. |
| `CLAUDE.md` | Conventions and constraints only. Keep it short. |

## Rules

1. **Check the math first.** Any new or changed gradient, modulator or rule needs `/derivation-check` *before* you edit. Record its numbers in Appendix A.
2. **Mark undecided ideas.** Write `**Proposal**` before them and never present them as settled. Unresolved questions go under **Open**, or into SPEC §8.
3. **Number equations** as `\qquad \text{(n)}`. Insert new equations with letter suffixes, (1a) or (4b), rather than renumbering. Keep the labels in reading order within a section.
4. **Avoid renumbering sections.** Add subsections at the end of a section, such as §4.6, or as sub-numbers. After any structural change, grep for `§` and `Eq.` in SPEC, derivation, CLAUDE.md and `.claude/agents/*.md`, and fix every reference.
5. **Appendix B.** Every substantive revision adds a numbered item: what changed and why, dated.
6. **Conventions** (CLAUDE.md):
   - $W_{ij}$ is pre $j$ → post $i$, and $T = W/d$.
   - Frames are indexed $f$; simulation steps $t$.
   - Module types are $\mathcal{C}$, $\mathcal{A}$, $\mathcal{L}$.
   - Rules must be crossbar-native.
   - The modulator is the marginal cost.
7. **Math rendering.**
   - Display `$$` goes on its own lines, with blank lines around it.
   - Use `\lbrace`/`\rbrace` and `\lVert`/`\rVert`.
   - Never use `\,`, `\;` or `\tag`.
   - Use braced arguments (`\mathbb{I}`, `\bar{e}`).
   - Don't put display math inside list items; use a paragraph with a bold lead-in instead.

   The PostToolUse hook runs `.claude/hooks/math_check.js` after each edit and reports any failure. Fix it immediately.
8. **Citations.** Only cite references whose author, year and venue you have verified. Add them to the sorted References list. Mark anything unverified.
9. **Preliminary numbers.** Numbers from a subagent's own simulation are labelled preliminary until they are reproduced.
10. **Finish** with a `/changelog` entry. Commits without a staged CHANGELOG are blocked by a hook.

## Final check

```bash
~/.conda/envs/map-stdp/bin/node .claude/hooks/math_check.js docs/derivation.md docs/SPEC.md docs/CHANGELOG.md CLAUDE.md
```
