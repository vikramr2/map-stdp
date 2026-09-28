# Changelog

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-09-27

### Removed
- **The PyTorch prototype in `models/`.** It had:
  - an SBM starting-topology generator;
  - a shared LIF simulation with an E-step/M-step training loop;
  - 4-bit suppression STDP (Gautam & Kohno 2023);
  - two Map-STDP variants trained on N-MNIST: fixed-community ($G_{sim}$) and lateral-inhibition ($G_{LI}$).

  The project restarts from ideation. The code remains in git history (last at `fcedb0a`).

### Added
- **`docs/derivation.md`**, a corrected markdown rewrite of `derivation/mapstdp.tex`. The tex is kept unchanged. The rewrite:
  - fixes a sign error in the expansion of $L(M)$. The gradient's log factor becomes $\log\frac{q_\curvearrowright(p_m+q_m)}{q_m^2} \ge 0$, verified by finite differences.
  - adopts a single direction convention: $W_{ij}$ is pre $j$ → post $i$, and the walk follows spikes forward. As a result, $G_{sim}(i,j) = \mathbb I[i\notin m(j)] - \bar e_j$ is indexed by the presynaptic neuron.
  - adds the stimulus as a teleportation term, $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$.
  - turns the identification of the firing marginal with walk flow into an explicit mean-field assumption.
  - replaces the fixed-point "convergence" proof with a contraction proof.
  - restates the lateral-inhibition variant with a soft indicator $\chi_{ij}$, and flags the problems with its anti-Hebbian rule.
- **`docs/SPEC.md`**, the project spec: goal and hypothesis, architecture, and five considerations with proposals (Brian2; the RL and vision task ladder, with classification as a contextual bandit; stimulus as teleportation; candidate description lengths vs. thermodynamic entropy; 3-factor variants). It also lists metrics, ablations and milestones M0–M6.
- **`docs/CHANGELOG.md`**, this file.

### Changed
- **`CLAUDE.md`**, rewritten for the docs-first phase.
