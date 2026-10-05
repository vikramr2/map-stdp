---
name: expert-review
description: Run the project's expert subagents (neuroscientist, snn-expert, and superneuro-expert when code is involved) in parallel on a doc section or piece of code, then merge their findings into one ranked list with agreements and conflicts. Use when the user asks for a review of the derivation, SPEC, a rule, or simulation code.
argument-hint: "<section, file or topic to review>"
---

# Expert review

Review target: **$ARGUMENTS**. If no target is given, ask what to review.

## 1. Launch the reviewers in parallel

Send all the Agent calls in one message, as background agents.

| Agent | When | Focus |
| --- | --- | --- |
| `neuroscientist` | always | biological plausibility, locality, timescales, analogies, citations |
| `snn-expert` | always | dynamics, learning, performance, stability. It may run numpy experiments in the scratchpad. |
| `superneuro-expert` | the target includes code or simulation plans | API correctness, per-frame loop, speed |

Each prompt must contain:

- **The target**, given as file paths and section numbers, and **the context** the agent needs: recent decisions, and what is a **Proposal**.
- **Review-only:** no repo edits. Throwaway scripts go in the session scratchpad.
- **Citations:** verify every one; never invent one.
- **Output:** a ranked list of issues (severity, location, problem, fix, with exact replacement text for wording changes), then "claims to soften" and "well supported", under about 1200 to 1500 words.

## 2. Merge

Once all the reports are back, give the user one merged summary:

- **Ranked issues.** Deduplicate, and mark where reviewers **agree**, since agreement strengthens a finding.
- **Conflicts.** Where performance and plausibility pull apart, state both sides.
- **Evidence status.** Say which numbers come from a reviewer's own simulation. Those are *preliminary*, and you have not re-run them.
- **Proposed edits.** Group them by file. Ask which to apply; the dual-term-style "keep as Proposal" choices are the user's.

## 3. Applying edits (only after the user agrees)

Follow `/doc-edit`. Run `/derivation-check` on any new math before writing it, and finish with `/changelog`.
