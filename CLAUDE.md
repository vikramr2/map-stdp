# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository is in a **docs-first research phase with no code**. The earlier PyTorch prototype was deleted on purpose; it is in git history at `fcedb0a` if you need it for reference. Do not restore it unless asked.

Research question: can a spiking network's communities serve as its states, giving it a state-space abstraction that solves RL tasks? **Map-STDP** inverts minimum description length: community detection finds the partition that minimises a walk's description length (the map equation, or one of seven alternatives) for a given graph, while Map-STDP fixes the partition and learns the weights. It is a three-factor rule. TD-modulated STDP with an action-gated eligibility trace learns the task. Cost-modulated STDP, whose per-module neuromodulator broadcasts the marginal description cost of each transition, makes the given partition the minimum-description-length one. The communities are a central controller (receives the stimulus), one action community per action, and latent communities. The coarse-grained module chain $T^K(\mathbf o)$ is the extracted, stimulus-conditioned Markov model. Each environment frame is one implicit random walk. Side hypothesis, secondary: description length tracks metabolic/thermodynamic cost.

## Docs

- `docs/derivation.md` is the **canonical** derivation. Its Appendix B lists the corrections to an earlier version, which had a sign error and inconsistent direction conventions. Put math changes in the markdown.
- `docs/SPEC.md` holds the goals, the design considerations (simulator, tasks, stimulus, description length as neuromodulator, memristive hardware), open questions and milestones. Items marked **Proposal** are undecided. Do not treat them as settled.
- `docs/CHANGELOG.md`: add a dated entry for any change to docs or code, including decisions and reversals.

## Conventions

- $W_{ij}$ is the synapse from **presynaptic $j$ to postsynaptic $i$**. The random walk follows spikes forward: $T_{ij} = W_{ij}/d_j$, where $d_j$ is $j$'s total outgoing weight.
- SuperNeuroMAT's `weight_mat()` is indexed `[pre, post]`, the transpose of $W$: `W_snm = W.T`. Its built-in STDP has no third factor, so it stays off; Map-STDP is applied per frame from `ispikes`.
- The stimulus enters the flow through teleportation: $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$.
- Flow is per frame: $\pi_f$ solves that equation with $v(\mathbf o_f)$, approximated by $n$ hops of power iteration. Frames are indexed $f$; $t$ is the simulation step. Module types are $\mathcal C$ (controller), $\mathcal A$ (actions) and $\mathcal L$ (latent); controller and action modules are pinned.
- Structural rule: $\Delta W_{ij} = -\lambda (g_{ij} - \bar{g}_j) \kappa_{ij}$, where the modulator $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal** cost, never the pointwise cost, and $\bar{g}_j$ is a per-neuron baseline. Its expectation is $-\lambda W_{ij} \partial D / \partial W_{ij}$. For the map equation it averages to the gating $G(i,j) = \mathbb{I}[i \notin m(j)] - \bar{e}_j$. Controller→action/latent routing is cross-module, so the map term alone starves it (derivation §4.5); the dual routing term is a **Proposal**.
- **Hardware target: memristive crossbars.** Every rule must be crossbar-native: per-module broadcast modulators times pulse-coincidence STDP. Modulators must be $K \times K$ module-pair tables. No nonlinear function of an individual synapse's conductance, and no access to the transposed $W_{ji}$. The only per-synapse state beyond $W$ is the task eligibility trace.
- Math must render in GitHub, VS Code and Markdown Preview Enhanced: put display `$$` on their own lines with blank lines around them, use `\lbrace`/`\rbrace` and `\lVert`/`\rVert` instead of `\{`/`\}` and `\|`, avoid `\,`, `\;` and `\tag` (number equations as `\qquad \text{(n)}`), and use braced arguments (`\mathbb{I}`, `\bar{e}`).
- Before changing a derivation, check the gradient claims numerically (finite differences on a small random network), and check that a new modulator's three-factor average matches $-W \odot \nabla D$ by Monte Carlo.

## Environment

The planned spiking simulator is **SuperNeuroMAT** (ORNL; see SPEC §3), with a numpy flow-level reference run first. Brian2 is only a fallback. `docs/superneuro` is the SuperNeuro reference submodule (README and tutorials). `environment.yml` lists the current stack (numpy, scipy, numba, gymnasium, superneuromat 3.5.0, Node.js); the npm tools are listed at its top. Never install packages globally without asking; use a scratchpad `--target` or a venv for experiments.

**Machines.** Work moves between machines. Paths in the hooks and MCP config resolve from the active env, so any machine needs only:

1. Clone the repo, then `git submodule update --init`.
2. Create the `map-stdp` env from `environment.yml`, plus the npm tools listed at its top.
3. Launch Claude from `conda activate map-stdp`.

The SLURM notes below apply only to the campus cluster. On a machine without SLURM, run experiments directly.

**Compute (SLURM, campus cluster only).** The login node is for editing, quick checks and short tests only. Anything longer than a few minutes goes to SLURM:

- **Submitting.** Use `sbatch [--job-name=NAME] slurm/job.sbatch <command...>` from the repo root. The template activates `map-stdp` and runs the command.
- **Partition: `secondary`** (the user's choice). Defaults are 1 node, 8 CPUs, 64 GB, 4 h.
  - **4 h is the hard limit.**
  - **Jobs can be preempted and requeued,** so long experiments must checkpoint and resume, or be split into jobs under 4 h, for example one per seed.
- **Other partitions** need the user's OK:
  - `zgdrasil` starts immediately, with 7 days and 2× A30 on 1 node;
  - `IllinoisComputes` needs `-A chackoge-ic` and allows 3 days, on 128-core nodes.
- **Logs.** They go to `logs/py_<jobid>.out` and `.err`; `logs/` is gitignored and must exist before submitting.
- **Monitoring.** Check jobs with `squeue -u $USER`, or `sacct -j <id>` after they finish.
- **Recording.** Put the job ID and the command in the iteration record in `docs/iterations/`.

**Tooling.** The `map-stdp` conda env has Python 3.11, Node.js and the packages in `environment.yml`. On the campus cluster it lives at `/scratch/vikramr2/conda/envs/map-stdp`, off `/u`, which has a 500k-file quota. On the local workstation it is `~/miniforge3/envs/map-stdp` (Miniforge in the home directory).

- On the cluster, `conda activate map-stdp` resolves to it through `envs_dirs` in `~/.condarc`.
- On the cluster, for installs, set `CONDA_PKGS_DIRS=/scratch/vikramr2/conda/pkgs`, and use `pip --no-cache-dir` and `npm --cache /scratch/vikramr2/conda/npm-cache`, so caches stay off `/u`.

- **ponytail** (project-scope plugin, `.claude/settings.json`): a least-code skill. Its hooks run `node`, so run `conda activate map-stdp` before launching `claude`.
- **CodeGraph** (project-scope MCP server, `.mcp.json`; telemetry off): a code-intelligence index. The index lives in `.codegraph/`, which is gitignored by its own `.gitignore`. Run `codegraph sync` after large changes.
- **Hooks** (`.claude/hooks/`):
  - KaTeX math check after editing docs;
  - commits blocked without a CHANGELOG entry ("skip changelog" overrides);
  - installs blocked outside the env;
  - an env warning at session start.
- **Skills:** `/derivation-check` (run before any derivation math change), `/doc-edit`, `/expert-review`, `/changelog`.
- **Plugins:** math-proof, pyright-lsp (pyright is in the env), commit-commands, hookify.
