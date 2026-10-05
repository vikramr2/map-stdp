---
name: changelog
description: Draft and insert a dated docs/CHANGELOG.md entry for the current uncommitted changes, in the file's Keep-a-Changelog style with decisions and reversals called out. Use before committing, or whenever docs or code changed.
---

# Changelog entry

CLAUDE.md requires a dated CHANGELOG entry for every change to docs or code, and a hook blocks commits that skip it.

## Steps

1. **See what changed:** `git status --short`, `git diff`, `git diff --cached`. Include untracked files.
2. **Read the latest entries** at the top of `docs/CHANGELOG.md` to match their style.
3. **Find the right heading.**
   - Use today's date as `## YYYY-MM-DD`.
   - If that date already has a section, add to it rather than creating a second one.
   - Sections are `### Added`, `### Changed` and `### Removed`, optionally qualified, as in `### Changed (simulator)`.
   - Newest dates go first.
4. **Write the bullets.**
   - Start each with the **bold file or topic**, followed by a colon or a short description.
   - Call out **decisions** (`**Decision: …**`) and **reversals** (`**Decision reversal: …**`) with their reasons.
   - Name new results with their numbers, for example "checked numerically: …".
   - Say which numbers are preliminary.
   - Mark undecided items as **Proposal**.
5. **Keep it factual.** Describe what changed and why. Don't paste diffs.
6. **Insert the entry** below the file's intro paragraph and above the previous newest date. The math-check hook will validate any math.

Do not commit unless the user asks.
