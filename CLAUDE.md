# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository is in a **docs-first research phase with no code**. The earlier PyTorch prototype was deleted on purpose; it is in git history at `fcedb0a` if you need it for reference. Do not restore it unless asked.

Research question: can an information-theoretic description length of neural activity flow (currently the map equation) serve as a surrogate for neurons' thermodynamic/metabolic cost? The learning rule under study is **Map-STDP**: STDP plus a gradient term on the map equation that pushes spiking networks to form cortical modules, with one output column per action or class.

## Docs

- `docs/derivation.md` is the **canonical** derivation. `derivation/mapstdp.tex` is the original and is kept as history only: it contains a sign error and inconsistent direction conventions that the markdown version corrects. Put math changes in the markdown.
- `docs/SPEC.md` holds the goals, the five design considerations (simulator, tasks, stimulus, description length vs. thermodynamics, 3-factor reduction), open questions and milestones. Items marked **Proposal** are undecided. Do not treat them as settled.
- `docs/CHANGELOG.md`: add a dated entry for any change to docs or code, including decisions and reversals.

## Conventions

- $W_{ij}$ is the synapse from **presynaptic $j$ to postsynaptic $i$**. The random walk follows spikes forward: $T_{ij} = W_{ij}/d_j$, where $d_j$ is $j$'s total outgoing weight.
- The stimulus enters the flow through teleportation: $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$.
- The simplified gating is indexed by the presynaptic neuron: $G_{sim}(i,j) = \mathbb I[i\notin m(j)] - \bar e_j$.
- Math in markdown uses `$…$` / `$$…$$`, which GitHub renders with MathJax.
- Before changing a derivation, check the gradient claims numerically (finite differences on a small random network).

## Environment

The planned simulator is **Brian2** (see SPEC §3). `environment.yml` is stale: it still lists the old PyTorch/snntorch/tonic stack. Update it when implementation starts.
