# Changelog

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-10-05

### Changed (expert review of the cleanup)

- **Review:** `/expert-review` of the cleanup diff by `snn-expert`, `superneuro-expert` and `neuroscientist`. All three found the cleanup numerically safe; `superneuro-expert` ran 20 spiking episodes on the old and new code and got identical frames, identical weights and the same 3.3 ms per frame. Applied, all names, docstrings and comments only; `pytest` 9/9 bit-identical and `ruff` clean afterwards:
  - **Renames in `experiments/m2.py`:** `local_plasticity` → `pre_td_updates` (all three reviewers: it is neither more local than the task term nor purely plasticity, and it mutates W and μ); `td_step` → `critic_step` (the task weight update stays in `episode`).
  - **`experiments/m2.py` docstrings:**
    - `pre_td_updates`: its update order, which the golden file locks;
    - `episode`: the task update lands one walk late;
    - `frame`: what κ and the action are in each backend;
    - `Agent`: the dual and the annealing;
    - `evaluate`: in spiking it consumes the training RNG.
  - **`mapstdp/spiking.py`:**
    - `pi_hat`: cited as the batch form of Eq. 2, replacing the bare §2.2 citation;
    - `run_frame`: what `reset()` clears;
    - `build` and `set_weights`: synapses are fixed at build time;
    - `pairings`: now says `lag`.
  - **`mapstdp/core.py`:**
    - `homeostasis_update`: marked as presynaptic conservation, an abstraction;
    - `flow`: a comment links `pi` to the iterates $x_k$.
  - **Not applied:** a separate evaluation RNG. `evaluate()` in spiking draws from the training RNG, so the logging interval changes trajectories. Fixing that would change `tests/golden.json`, so it is documented and deferred.

### Changed (docs aligned with the code; found by the codebase-cleaner)

- **`docs/derivation.md`:**
  - **Eq. 4c** now states the critic the code uses since iteration 001: $V = u \cdot \pi_{\mathcal{C}} / p_{\mathcal{C}}$ over controller neurons, with NLMS, reward −1 on termination and bootstrapping on truncation. The module-level critic is recorded as the weaker alternative (160 against 195, preliminary, iteration 001).
  - **§6 table** and **§9 item 13** are updated to match.
  - **§8 step 2:** the M2 frame is cold-started, $k = 100$ (was "about $4 \tau / \alpha$, warm-started").
  - **§8 step 5:** reordered to match the code. The pairings are the covariance count (was "causal minus acausal"), and the TD error and the annealed $\eta_e$ step come after the next frame's walk.
  - **§9 item 2:** the mean-field assumption needs the covariance count, not the balanced count. The note that it is untested in spiking is replaced by the M1 cosine of 0.992.
- **`docs/SPEC.md`:** the C2 critic line is updated to match Eq. 4c.

### Changed (code cleanup)

- **Behaviour-preserving cleanup** by the `codebase-cleaner`, with no change to numerics, RNG-call order, CLI flags, result-file names or JSON keys. `pytest -q tests` passes before and after (9/9, `golden.json` untouched). `m2.summary()` output and a short `m2.run` (flow and spiking, results written to a scratch directory) are byte-identical to the baseline.
  - **Lint:** `ruff check .` goes from 21 findings to 0. Long lines are wrapped (`m1.py` keeps the same AST), imports are sorted (`m2_viz.py`, `fdcheck.py`), `pairings` uses `lag` instead of `l`, the `core` self-check drops its semicolons and binds `beta` in `logP`, and `fdcheck.map_L` uses a `def` instead of a lambda.
  - **Pyright:** 18 errors go down to 16, after an annotation on `m2.Agent.evaluate`'s output dict. The rest are inference limits on `sum()`/`0` accumulators and optional arguments, and are left alone.
  - **Names:** `core.flow` names its iterate `pi` instead of `x`. `m2.Agent.C` becomes `Agent.ctrl`, since `C` means causal counts in `spiking`.
  - **Structure:** `m2.Agent.episode` is split along derivation §8 into `pre_td_updates` (module statistics, Eq. 4b trace, Eq. 4, Eq. 4d, Eq. 4a dual) and `critic_step` (TD error and NLMS critic). These names come from the expert review below; the cleaner first called them `local_plasticity` and `td_step`. Operation order is unchanged. The dual-target fraction 0.75 is now `m2.Q_STAR_FRAC`. Nested conditional expressions in `Agent.__init__` and `summary()` are now `if` statements.
  - **Docstrings:** added to functions that had none, with citations (`spiking.build`, `tau`, `pi_hat`, `run_frame`; `core._plogp`; `m2.Agent` and its methods, `random_return`, `run`, `summary`; `m1` helpers; `m2_viz.first_race_step`). The `m2.py` header now reads "models v1-v3".

### Added (code hygiene)

- **`.claude/agents/codebase-cleaner.md`:** adapted from another project for Map-STDP. Changes:
  - **Scope and layout:** the scope, the forbidden paths, and this repo's layout and dependency direction.
  - **Checks:** ruff, pytest and pyright in place of the original's tools.
  - **Hard limits for this code:** numerics, RNG-call order, the $W$/`W_snm` orientation, the crossbar-native rules, CLI and result-file formats, and no `strict=` on zips.
  - **Review:** it points to `/expert-review` instead of `design-guardian`, which doesn't exist here.
- **`tests/test_regression.py` and `tests/golden.json`:** a behaviour lock. Checked:
  - Fixed-seed runs are bit-identical across processes, spiking included.
  - The suite covers five closed-loop configurations, the module self-checks, pairing counts with the race readout, and `m2.summary()`.
  - Changing the weight floor by 0.01% fails all five closed-loop cases.
- **`ruff.toml`:**
  - **Rules pinned:** E, F, W, I and B, because ruff 0.16's defaults are much wider.
  - **B905 off:** `m2.evaluate` relies on zip truncation.
  - **No `ruff format`.**
  - **Baseline:** 21 findings, left for the cleaner.
- **`environment.yml`:** adds ruff and pytest; both are installed in the local env.

### Fixed

- **`m2.py --summary` crashed** on the replay recording, because its `m2_*.json` glob matched `m2_viz_episode_s3.json`. The viz outputs are renamed `viz_m2_*`.
- **`pyrightconfig.json`** no longer hard-codes the cluster env path. Pyright and the IDE now resolve imports from the active interpreter on any machine.

### Added (visualisation)

- **`experiments/m2_viz.py` and `experiments/m2_viz_template.html`:** a replay page of a trained spiking v3 agent. The left pane shows spikes and walk hops on the network, with the controller drawn as its 6×6 (θ, θ̇) grid and the race winner marked; the right pane shows CartPole. Commands:
  - `train` saves the weights (`viz_m2_W_s<seed>.npz`, renamed from `m2_viz_*` so `m2.py --summary` no longer picks them up; seeds 1–4 have last-100 returns of 193, 143, 207 and 192);
  - `record` keeps the best of 20 frozen-weight episodes (seed 3: 302 frames);
  - `page` embeds the recorded episode into the template.

### Changed (environment)

- **Local workstation env:** Miniforge in `~/miniforge3`, with the `map-stdp` env created from `environment.yml` plus katex, pyright and CodeGraph 1.6.2 (telemetry off). The math hook, the self-checks and the CodeGraph index (148 nodes) work there. The M2 runs used a scratchpad venv with the same packages before this env existed.
- **`CLAUDE.md` and `environment.yml`:** the "environment.yml is stale" note is removed; the file already lists the current stack. The setup comments now cover any machine as well as the cluster.

### Added (iteration 002: M2 done)

- **M2 results.** 110 runs, 5 seeds × 1500 episodes each, run locally on a 72-core workstation with no SLURM; results are in `experiments/results/m2_*_{v1,v2,v3}_s*.json`. Model v3, mean of the last 100 episodes, against a random policy of 22:
  - **Flow:** exempt 287 ± 21, dual 327 ± 46, none 39 ± 11, $\lambda = 0$ 187 ± 15.
  - **Spiking:** exempt 184 ± 13, $\lambda = 0$ 77 ± 4.
  - **Race vs. Eq. 1d:** KL 0.010, against a sampling floor of 0.010.

  All M2 criteria are met; SPEC §11 marks M2 done. Full record: `docs/iterations/002-m2.md` (v1 baseline, reviews, v2, annealing test, v3, outcome).
- **Reviews** by `snn-expert` (spiking diagnostics) and `neuroscientist` (plausibility), merged in the iteration record.
- **Code:**
  - `mapstdp/core.py`:
    - `homeostasis_update` (Eq. 4d);
    - `beta` in `policy`, `action_score` and `eligibility_step`, for the sharpened policy (Eq. 1e). The self-check now verifies the $\beta = 3$ score against finite differences.
  - `mapstdp/spiking.py`: `race(..., burn)`, with a self-check.
  - `experiments/m2.py`:
    - `--model {v1,v2,v3}` (default v3), `--beta`, `--eta`, `--eta-tau`;
    - the variant goes into the result filename;
    - `--summary` groups by model and variant.

  The v1 result files were renamed with a `_v1` suffix.
- **`docs/derivation.md`:**
  - **§2.2:** homeostasis as its own term (4d); weak-synapse rectification does not depend on the floor value; soft bounds (**Proposal**).
  - **§2.4:** the race equals (1d) only after relaxation, so a burn-in is adopted; sharpened policy (1e) (**Proposal**).
  - **§4.5:** M2 numbers; sealing caveat (persistence 0.996 against about 80% intrinsic input in cortex); exit-floor dual (**Proposal**, M4); the dual's $\mu$ diverges.
  - **§4.6:** $\bar{h}_j$ is presynaptic; the $\tau_e$ timescale.
  - **§8:** burn-in, (4d) and annealing in the training loop.
  - **§9, item 14:** updated.
  - **Appendix A:** iteration-002 checks.
  - **Appendix B:** item 11.
  - **References** (each checked by the `neuroscientist`): Chistiakova et al. 2014, Frank 2006, Gold & Shadlen 2007, Gurney et al. 2015, Markov et al. 2011, Royer & Paré 2003, Turrigiano et al. 1998, van Rossum et al. 2000, Zenke & Gerstner 2017.

### Changed (model v1 → v3, `docs/model.md`)

- **Decision: $d_j$ homeostasis is its own term on all plasticity** (Eq. 4d, $\epsilon_h = 5$, independent of $\lambda$), not part of the structural baseline. Inside $\lambda$ it was about 100× too weak, and it vanished at $\lambda = 0$. The task term ratcheted $d_{max}$ to 13 (flow) and 19 (spiking). Now $d_{max} \le 1.04$. Its expectation is still a column rescaling: spiking Monte Carlo cosine 0.992 (preliminary).
- **Decision: race burn-in of 10 steps.** Without it, the race read the cold-start transient (KL 0.022 against a floor of 0.010). With it, KL is at the floor.
- **Decision: anneal the task rate,** $\eta = 10$ decayed as $\eta / (1 + e/300)$ over episodes, in both backends (v3). v2 used a constant rate with spiking $\eta = 5$, and its return fell: spiking 140 → 120, flow at $\lambda = 0$ 210 → 145. The v1 ratchet had acted as hidden annealing. Constant $\eta = 10$ gives spiking 88, annealed gives 184.
- **Decision: routing exemption stays the default.** The dual is within about one sd ($t = 1.6$, $p = 0.15$), its order relative to the exemption flipped between v2 and v3, and its $\mu$ never converges.
- **Rejected for now: the sharpened policy (Eq. 1e).** At the flow level it collapsed on 3 of 5 seeds at $\beta = 4$ and on 1 of 5 at $\beta = 2$. It stays a **Proposal**, needing an entropy floor and a matched spiking readout.
- **Deferred: the exit floor on module sealing.** Sealing costs no return on CartPole (`snn-expert`), but it is implausible as anatomy (`neuroscientist`). Revisit at M4.
- **`docs/SPEC.md`:** status line, M2 marked done, the M2 routing results, and ablations (homeostasis placement, task-rate schedule, temperature, exit floor, race burn-in).

### Added (implementation, iteration 001: M1)

- **Code:**
  - `mapstdp/core.py`: numpy flow-level reference;
  - `mapstdp/spiking.py`: SuperNeuroMAT backend;
  - `experiments/m1.py`: the six M1 checks, with results in `experiments/results/m1.json`.

  Both modules have assert-based self-checks. Built by `superneuro-expert`. `pyrightconfig.json` points pyright at the `map-stdp` env.
- **`docs/model.md`:** the current model, versioned. v0 is the spec; v1 follows M1.
- **`docs/iterations/001-m1.md`:** the M1 record, with build, results, reviews (`snn-expert`, `neuroscientist`) and decisions.

### Added (iteration 002: M2, in progress)

- **Code** (built by `superneuro-expert`; self-checks pass):
  - `mapstdp/core.py`: the CartPole conjunctive encoder, $d_j$ homeostasis, and the action-gated eligibility (Eq. 4b);
  - `mapstdp/spiking.py`: the race readout;
  - `experiments/m2.py`: the closed-loop CLI and `--summary`.
- **`docs/iterations/002-m2.md`:** status, run plan, preliminary numbers, and open problems:
  - a $d_j$ ratchet from the task term;
  - the race readout reading the early transient.
- **No M2 results yet.** The SLURM jobs on `secondary` sat pending on priority for over 4 h and were cancelled on the user's instruction. M2 continues on a GPU machine that needs no SLURM.

### Changed (portability)

- **Paths:** the hooks, `.mcp.json` and the skills no longer hard-code `/scratch/vikramr2`.
  - The hooks use the active env (`CONDA_PREFIX`).
  - `math_check.js` finds katex next to the running `node`.
  - CodeGraph and the skills resolve from `PATH`.

  Launch Claude from `conda activate map-stdp` on any machine.

### Changed (iteration 001 decisions)

- **Decision: pairing count $\kappa$ is the covariance count** (derivation §2.2), replacing the balanced count. Cosine with $-W \odot \nabla D$: 0.992 covariance, 0.979 causal, 0.847 balanced, reproduced at 5k frames. References added: Sejnowski 1977 and Kempter et al. 1999.
- **Decision: presynaptic normalisation** as an output gain $1/d_j$, plus $d_j$ homeostasis $\epsilon (d_j - 1)$ in the baseline (derivation §2.2).
- **Decision: spiking uses branching mode,** with $k = 100$. Frame windows are stated in spikes per module (error ≈ $1.3 / \sqrt{S_{mod}}$). The $4 \tau / \alpha$ estimate in §2.3 was optimistic.
- **Decision: conjunctive encoding** (SPEC §4, derivation §2.4). Per-dimension fields cap the CartPole policy at about 84 steps.
- **Decision: routing exemption is the M2 default** (SPEC §6, derivation §4.5). In the flow-level closed loop, the map term without protection collapsed learning (≈53), while exempt routing at $\lambda = 0.05$ reached ≈343 against ≈195 for $\lambda = 0$. These numbers are preliminary. The dual term stays a Proposal and becomes an ablation. A constant $\lambda$ replaces the ramp, which becomes an ablation.
- **SPEC:**
  - M1 marked done;
  - status line updated;
  - ablations updated (pairing count, normalisation, encoding, schedule);
  - "Controller in $D$" provisionally answered (input layer).

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
- **SLURM template** (`slurm/job.sbatch`): the user's standard header (partition `zgdrasil`, 8 CPUs, 64 GB, 2 days). It activates `map-stdp` and runs the given command. `logs/` is gitignored.
  - CLAUDE.md gains a "Compute (SLURM)" section: long runs go to `sbatch`, never the login node.
  - The `snn-expert` and `superneuro-expert` agents point to it.
  - **Decision:** jobs go to the `secondary` partition (4 h, preemptible), on the user's instruction. Installs run on the login node.
- **Env moved to `/scratch/vikramr2/conda/envs/map-stdp`.** The `/u` home hit its 500k-file quota at 503k files (`~/.conda/pkgs` held ~224k).
  - `conda clean --all` freed ~107k files.
  - The env was rebuilt on scratch, with its package cache there too, and registered through `envs_dirs`. It now also has numba, gymnasium, matplotlib, networkx and superneuromat 3.5.0.
  - Hooks, skills and `.mcp.json` point to the new path.
  - `environment.yml` is rewritten to match; it had listed the old PyTorch stack.
  - The old home env is removed once Claude is relaunched from the new env.

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
