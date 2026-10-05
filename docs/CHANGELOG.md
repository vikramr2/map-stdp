# Changelog

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-10-04

### Changed

- **Reformulation: communities as states.** The primary goal is now a state-space abstraction for spiking networks in which communities are the states. Map-STDP is framed as the **inverse** of community detection: the partition is given, and the rule learns weights that make it the minimum-description-length partition of the walk. The thermodynamic surrogate hypothesis becomes a secondary side tie, and the energy proxy a secondary metric.
- **`docs/derivation.md`:**
  - New title, goal and motivation. The controller community is motivated by the cerebral cortex, with Global Workspace Theory kept as an analogy.
  - Notation adds module types $\mathcal C$ (controller), $\mathcal A$ (actions) and $\mathcal L$ (latent), frame index $f$, hops $n$, and module-pair flows $J_{ba}$, $T^K_{ba}$.
  - New §2.3, one walk per frame:
    - per-frame flow $\pi_f$, computed implicitly by $n$ hops with error $\le 2(1-\alpha)^n$ (Eq. 1a);
    - software, crossbar and spiking realisations;
    - timescale ordering (hops, then frames, then plasticity);
    - a carry-over proposal, Eq. 1b, so latent communities persist across frames.
  - New §2.4, the extracted Markov model:
    - the module chain $T^K(\mathbf o)$ (Eq. 1c);
    - why a central controller makes it stimulus-conditioned;
    - policy readout from action communities (Eq. 1d);
    - a proposed across-frame latent chain as a world model.
  - New §3.4: the inverse problem $\min_W \mathbb{E}_f[L(M; \pi_f)]$ (Eq. 3a). Theorems 1–3 hold frame by frame, and re-detecting latent modules makes training an alternating minimisation.
  - §5: candidates table gains a "state abstraction" column. New §5.9 proposes the entropy rate of the latent chain as a candidate; its modulator is not yet derived.
  - §6: per-frame walk row in the crossbar table.
  - §8: training loop restructured by frame. Only latent modules may be re-detected.
  - §9: assumptions 9–11 added (convergence within a frame, carry-over, number of latent modules).
  - Appendix A and Appendix B item 8 added.

  New results, each checked numerically on a 15-node, 5-module network:
  - the per-frame and frame-averaged map-equation gradients match finite differences ($\le 10^{-9}$);
  - the three-factor rule averaged over frames matches $-W \odot \nabla D$ (correlation 0.99996, or 0.99993 with a pooled modulator);
  - modular networks contract at nearly the $1-\alpha$ bound (0.78 vs. 0.8);
  - with carry-over held fixed, the gradient has cosine 0.98 with the full gradient;
  - the controller column of $T^K$ carries most of the stimulus dependence.
- **`docs/SPEC.md`:**
  - §1 has the new goal and hypotheses.
  - §2 has the new architecture: controller, latent and action communities, with carry-over.
  - §4 is RL-only. The ladder is CartPole → Acrobot/MountainCar → LunarLander → a partially observable task. **Vision is deferred.**
  - §5 merges stimulus and timescale. Implicit walk per frame and carry-over are **Proposals**.
  - §6: the candidates table gains a state-abstraction column and the $h_{\mathcal L}$ proposal. Decision criteria now include extracted-model quality, and energy is secondary.
  - §8: pinned vs. free latent partitions; $\lvert \mathcal L \rvert$; the world-model open question.
  - Metrics and ablations are split into §9 and §10 and extended (policy KL, implicit vs. spiking walk, carry-over, latent communities).
  - Milestones M1–M6 are re-scoped.
- **`CLAUDE.md`:** research question rewritten; conventions for per-frame flow and module types.

## 2026-10-01

### Added

- **`docs/review/search.py`** and **`docs/review/data/`**: the Phase 2 search script and its output. It queries OpenAlex, PubMed and arXiv and returns 1,870 records, which come to 1,276 after deduplication.
- **`docs/review/screening/`**: screening rules (clarifications of protocol §B.2), the 50-record calibration set, a blank author sheet, and the AI's calibration decisions, committed before the author screens.
- **`docs/review/protocol.md`**, a draft PRISMA-P protocol (frozen at v1.2 after author approval; not registered, git commit serves as the timestamp) for a systematic review of how neural signalling dissipates energy and what evidence links information-theoretic measures to metabolic or thermodynamic cost. It covers three sub-questions (mechanism, information–cost linkage, network level) and separates cost classes M/T/W/P with no pooling across classes. It also adds a gap check for description-length measures, including the map equation, and records Devil's Advocate Checkpoint 1. The protocol is awaiting author approval, and no search has been run beyond pilot counts.

### Changed

- **Review protocol v1.3:** the search strings were revised after the seed recall check missed 3 of 10 eligible seeds. The revised strings retrieve all 10. Details are in the protocol's §B.13.

## 2026-09-27

### Removed

- **The original LaTeX derivation** and its template files, superseded by `docs/derivation.md`. Its training-loop figure was dropped because it depicted the old per-sample E-step/M-step loop.
- **The PyTorch prototype in `models/`.** It had:
  - an SBM starting-topology generator;
  - a shared LIF simulation with an E-step/M-step training loop;
  - 4-bit suppression STDP (Gautam & Kohno 2023);
  - two Map-STDP variants trained on N-MNIST: fixed-community ($G_{sim}$) and lateral-inhibition ($G_{LI}$).

  The project restarts from ideation. The code remains in git history (last at `fcedb0a`).

### Added

- **`docs/derivation.md`**, a corrected markdown rewrite of the original LaTeX derivation. The rewrite:
  - fixes a sign error in the expansion of $L(M)$. The gradient's log factor becomes $\log\frac{q_\curvearrowright(p_m+q_m)}{q_m^2} \ge 0$, verified by finite differences.
  - adopts a single direction convention: $W_{ij}$ is pre $j$ → post $i$, and the walk follows spikes forward. As a result, $G_{sim}(i,j) = \mathbb I[i\notin m(j)] - \bar e_j$ is indexed by the presynaptic neuron.
  - adds the stimulus as a teleportation term, $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$.
  - turns the identification of the firing marginal with walk flow into an explicit mean-field assumption.
  - replaces the fixed-point "convergence" proof with a contraction proof.
  - restates the lateral-inhibition variant with a soft indicator $\chi_{ij}$, and flags the problems with its anti-Hebbian rule.
- **`docs/SPEC.md`**, the project spec: goal and hypothesis, architecture, and five considerations with proposals (Brian2; the RL and vision task ladder, with classification as a contextual bandit; stimulus as teleportation; candidate description lengths vs. thermodynamic entropy; 3-factor variants). It also lists metrics, ablations and milestones M0–M6.
- **`docs/CHANGELOG.md`**, this file.

### Changed

- **`CLAUDE.md`**, rewritten for the docs-first phase. It now also lists math-rendering conventions.
- **`docs/derivation.md`**, consolidated to put interpretation first:
  - an "At a glance" section with the rule, a sign/magnitude table and a worked example;
  - a notation table saying where each quantity lives;
  - an intuition for the map equation as a two-level code, and the fact that leakage always costs bits;
  - a single assumptions list;
  - the corrections to the earlier version moved to Appendix B.

  New result: $\sum_i W_{ij} G(i,j) = 0$, so a multiplicative map term conserves each neuron's outgoing weight (checked numerically). All math now renders in KaTeX, GitHub and Markdown Preview Enhanced: no `\tag`, `\,`, `\{` or `\|`, and every span is validated with KaTeX.
- **`docs/SPEC.md`**, cross-references updated to the new section and equation numbers. $G_{sim}$ is renamed to $G$, and math spacing is fixed.
- **Decision: structure is a neuromodulated 3-factor rule.** The structural term is now cost-modulated STDP: $\Delta W_{ij} = -\lambda (g_{ij} - \bar{g}_j) \kappa_{ij}$. The neuromodulator broadcasts the *marginal* description cost $g_{ij} = \partial D / \partial J_{ij}$, a $K \times K$ module-pair table.
  - Chosen over a factorized-gradient hardware rule because of the new memristive-crossbar target.
  - The exact gradient remains the analysis reference and simulation baseline.
  - Supersedes SPEC 3-factor variants A/B/C.
- **`docs/derivation.md`**, restructured around the rule:
  - new §4: the rule, and Theorem 3, which shows the expected update is $-\lambda W_{ij} \partial D / \partial W_{ij}$; marginal vs. pointwise cost;
  - new §5: eight candidate description lengths as modulators (map equation, entropy rate, entropy production, cost-weighted map equation, Markov stability, SBM description length, predictive dissipation, cross-module information flow);
  - new §6: crossbar mapping;
  - neuromodulator/energy motivation with verified citations.

  New results, each checked numerically:
  - $M^{\ast}_m$, previously absorbed into the learning rate, is the map equation's per-module neuromodulator, firing only on module exits.
  - $G$ is exactly the lag-1 Markov-stability gradient.
  - Broadcasting pointwise codeword length is biased: correlation 0.976 with the true descent direction, against 0.9999 for the marginal cost.
  - Coarse-grained $\sigma_K$ lower-bounds $\sigma$, but $h_K$ has no general ordering with $h$.
- **`docs/SPEC.md`**, restructured:
  - §6 merges C4 and C5 into "Description length as a neuromodulator", with a candidate table and decision criteria;
  - new §7 (C6) sets memristive crossbars as the hardware target, with its constraints;
  - ablations, M1 and M5 updated.
- **`CLAUDE.md`** adds the crossbar-native constraint.
