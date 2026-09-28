# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository is in a **docs-first research phase with no code**. The earlier PyTorch prototype was deleted on purpose; it is in git history at `fcedb0a` if you need it for reference. Do not restore it unless asked.

Research question: can an information-theoretic description length of neural activity flow (currently the map equation) serve as a surrogate for neurons' thermodynamic/metabolic cost? The learning rule under study is **Map-STDP**, a three-factor rule. Reward-modulated STDP learns the task. Cost-modulated STDP, whose per-module neuromodulator broadcasts the marginal description cost of each transition, pushes spiking networks to form cortical modules, with one output column per action or class. The map equation is the baseline description length, and seven alternatives are compared.

## Docs

- `docs/derivation.md` is the **canonical** derivation. Its Appendix B lists the corrections to an earlier version, which had a sign error and inconsistent direction conventions. Put math changes in the markdown.
- `docs/SPEC.md` holds the goals, the design considerations (simulator, tasks, stimulus, description length as neuromodulator, memristive hardware), open questions and milestones. Items marked **Proposal** are undecided. Do not treat them as settled.
- `docs/CHANGELOG.md`: add a dated entry for any change to docs or code, including decisions and reversals.

## Conventions

- $W_{ij}$ is the synapse from **presynaptic $j$ to postsynaptic $i$**. The random walk follows spikes forward: $T_{ij} = W_{ij}/d_j$, where $d_j$ is $j$'s total outgoing weight.
- The stimulus enters the flow through teleportation: $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$.
- Structural rule: $\Delta W_{ij} = -\lambda (g_{ij} - \bar{g}_j) \kappa_{ij}$, where the modulator $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal** cost, never the pointwise cost, and $\bar{g}_j$ is a per-neuron baseline. Its expectation is $-\lambda W_{ij} \partial D / \partial W_{ij}$. For the map equation it averages to the gating $G(i,j) = \mathbb{I}[i \notin m(j)] - \bar{e}_j$.
- **Hardware target: memristive crossbars.** Every rule must be crossbar-native: per-module broadcast modulators times pulse-coincidence STDP. Modulators must be $K \times K$ module-pair tables. No nonlinear function of an individual synapse's conductance, and no access to the transposed $W_{ji}$. The only per-synapse state beyond $W$ is the task eligibility trace.
- Math must render in GitHub, VS Code and Markdown Preview Enhanced: put display `$$` on their own lines with blank lines around them, use `\lbrace`/`\rbrace` and `\lVert`/`\rVert` instead of `\{`/`\}` and `\|`, avoid `\,`, `\;` and `\tag` (number equations as `\qquad \text{(n)}`), and use braced arguments (`\mathbb{I}`, `\bar{e}`).
- Before changing a derivation, check the gradient claims numerically (finite differences on a small random network), and check that a new modulator's three-factor average matches $-W \odot \nabla D$ by Monte Carlo.

## Environment

The planned simulator is **Brian2** (see SPEC §3). `environment.yml` is stale: it still lists the old PyTorch/snntorch/tonic stack. Update it when implementation starts.
