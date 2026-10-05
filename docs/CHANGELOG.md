# Changelog

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-10-05

### Added

- **`map-stdp` conda env** (`~/.conda/envs/map-stdp`): Python 3.11 and Node.js 26, so the tooling has Node without a system-wide install.
- **ponytail plugin** (v4.11.0, [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)), installed at project scope through `.claude/settings.json`. It is a least-code skill. Its hooks were reviewed before install: they inject instructions and write small mode files under `~/.claude` and `~/.config/ponytail`, with no network calls and no subprocesses. It needs `node` on PATH, so activate the env before launching `claude`.
- **CodeGraph MCP server** (v1.6.2, [colbymchenry/codegraph](https://github.com/colbymchenry/codegraph)), installed in the env and registered at project scope in `.mcp.json`.
  - Telemetry is off, both by `codegraph telemetry off` and by `CODEGRAPH_TELEMETRY=0`.
  - The local index is in `.codegraph/`, which ignores itself.
  - The repo has no code yet, so the index is empty until implementation starts.
- **Project hooks** (`.claude/hooks/project_hooks.py` and `math_check.js`, registered in `.claude/settings.json`), each tested with sample events:
  - **Math check.** After any edit to `docs/*.md` or `CLAUDE.md`, KaTeX renders every math span in strict mode and banned macros (`\,` `\;` `\tag` `\{` `\}` `\|`) are flagged. Failures are fed back so they get fixed immediately.
  - **Changelog guard.** Blocks `git commit` when changes are staged without `docs/CHANGELOG.md`. Override with "skip changelog" in the message.
  - **No global installs.** Blocks pip, npm `-g` and conda installs that don't target the `map-stdp` env, a venv, or `--target`.
  - **Environment check.** At session start, warns if the `map-stdp` env or `node` is missing.
- **Project skills** (`.claude/skills/`):
  - **`/derivation-check`** (`scripts/fdcheck.py`): the finite-difference and Monte Carlo checks CLAUDE.md requires. Its selftest reproduces a map-gradient error of 7e-10 and a three-factor correlation of 0.99994.
  - **`/doc-edit`**: house rules for the docs.
  - **`/expert-review`**: runs the expert agents in parallel and merges their findings.
  - **`/changelog`**: drafts an entry in this file's style.
- **Official plugins at project scope:**
  - **math-proof**, for proofs of the derivation's theorems;
  - **pyright-lsp**, with pyright 1.1.414 installed in the env rather than globally;
  - **commit-commands**;
  - **hookify**.
- **`map-stdp` env:** gained katex, pyright, numpy and scipy.

## 2026-10-04

### Changed (simulator)

- **Decision reversal: SuperNeuroMAT replaces Brian2** as the spiking simulator (SPEC §3). Reasons:
  - its discrete-time matrix update makes one step one hop, and one crossbar read;
  - the per-frame closed loop is plain Python;
  - the plasticity code can be shared with the numpy reference;
  - it has CPU, JIT and GPU backends;
  - it can export to the NeuroCoreX FPGA platform (inference only).

  Brian2 stays as a fallback for continuous-time checks. SuperNeuroABM is an option for heterogeneous neurons.
- **Smoke test** (SuperNeuroMAT 3.5.0, installed only in the scratchpad):
  - `weight_mat()` is `[pre, post]` (the transpose of $W$);
  - built-in `aneg` is a per-step decay on non-coincident synapses, not acausal STDP, so Map-STDP is applied per frame outside `simulate`;
  - `add_spike` times are relative to the current step;
  - cost per frame is about 1 ms at ~100 neurons, 40–50 ms at ~500 and ~230 ms at ~1000, dominated by weight get/set in Python at larger sizes.
- **`docs/derivation.md` §2.3:** a discrete-time note. One step with delay 1 is one hop. A branching-process mode (infinite leak, stochastic input) is a Proposal for M1.
- **`CLAUDE.md`:** simulator, submodule, transpose convention, and no global installs.
- **`.claude/agents/snn-expert.md`:** tooling is no longer Brian2-specific.

### Added (simulator)

- **`.claude/agents/superneuro-expert.md`**: a subagent expert in coding SuperNeuroMAT and SuperNeuroABM. Its API facts were checked against the source, and it follows the project constraints.

### Changed (review revisions)

Applied the findings of the neuroscientist and SNN-expert reviews of `docs/derivation.md`.

- **Decision: new task term.** TD-modulated STDP with an action-gated eligibility trace (derivation Eq. 4b, 4c, §4.6) replaces $(R - \bar{R}) e_{ij}$. That term gets no action credit when the action is sampled from flow shares, and stalls under sparse reward. The new eligibility has the same crossbar form as the structural rule.
- **Proposal: dual routing term** (Eq. 4a, §4.5). The map term alone starves controller→action and controller→latent routing, because that routing is cross-module. The dual term holds routing at a target $q^{\ast}$, and the simplest variant exempts the routing pairs. It stays a Proposal until simulated.
- **Spiking timing.** Spiking frames take about $4\tau/\alpha$ (100–250 ms), not 20 ms. The RL ladder runs on a numpy flow-level reference first, and Brian2 is used for fidelity at M1 and M2.
- **Mean field.** The doc now states its three requirements: constant $d_j$, global normalisation, and causal-minus-acausal pairing counts.
- **Locality (§6).** Restated: the postsynaptic module is the local one. The doc also notes what neuromodulators can carry, and that the per-neuron baseline equals presynaptic renormalisation to first order.
- **Readout and memory.**
  - A race readout matches Eq. 1d, and the limits of the policy family are stated.
  - Linear carry-over forgets geometrically; a module-level nonlinearity is proposed.
- **Lateral inhibition (§7)** is routed through interneurons with inhibitory STDP, for Dale's law.
- **Training (§8)** gains stability measures and a $\lambda$ ramp. **Assumptions (§9)** gain items 12–15.
- **Claims softened:** controller ↔ cortex (now a functional abstraction with basal-ganglia, OFC and hippocampal analogues), the characterisation of Buesing et al., the metabolic reading of the map equation, Friston, and Lynn et al. (an fMRI measure).
- **References.** 14 added, each checked by title, venue and year during the review.
- **New numerical checks (Appendix A):**
  - action-gated eligibility: cosine 0.997 with the true $\nabla \log P$, and the Monte Carlo correlation is 0.997;
  - baseline as renormalisation: first order, relative difference $2.5\lambda$;
  - dual-term gradient: exact;
  - carry-over memory decays 0.29–0.39 per frame.

  The preliminary spiking and bandit numbers come from the review's own simulations and still need reproducing.
- **`docs/SPEC.md`:**
  - architecture analogies softened;
  - spiking timing corrected;
  - task-term decision recorded;
  - routing conflict and dual Proposal added;
  - new open questions (controller in $D$, $q^{\ast}$, nonlinear carry-over, device multiplicativity, Dale's law, efference copy);
  - new ablations;
  - M1 and M2 re-scoped.
- **`CLAUDE.md`:** task term and routing-conflict notes.

### Added

- **`.claude/agents/snn-expert.md`**: a subagent expert in SNNs, local and three-factor learning, Brian2 and closed-loop RL, focused on improving performance within the crossbar-native constraints. It can run and edit code.
- **`.claude/agents/neuroscientist.md`**: a read-only subagent that reviews biological plausibility (locality, timescales, neuromodulator specificity, Dale's law, strength of analogies) with verified citations.

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
