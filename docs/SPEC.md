# Map-STDP Project Spec

Status: ideation, with no code yet. The math lives in [`derivation.md`](derivation.md), and changes are logged in [`CHANGELOG.md`](CHANGELOG.md).

Each consideration below states the user's direction first. Items marked **Proposal** are candidate approaches that have not been decided. **Open** items are unresolved.

## 1. Goal and hypothesis

Find a learning framework in which **information-dynamic entropy is a surrogate for the thermodynamic entropy (metabolic cost) of neurons**.

**Hypothesis.** A neuron pays more thermodynamic cost to communicate with neurons in cortices it does not normally talk to, and less within its usual partners. A description length of neural activity flow, such as the map equation, captures this: rare, cross-module transitions get long codewords and frequent, within-module transitions get short ones. A plasticity rule that descends this description length alongside a task-learning rule should therefore:

1. form modular cortices, one per concept;
2. keep task performance; and
3. reduce a biophysical energy proxy.

Map-STDP ([`derivation.md`](derivation.md), Eq. 4) is the current rule: cost-modulated STDP whose neuromodulator reports a description length. Which description length is right is itself a question (§6).

## 2. Architecture

```
stimulus o ──► receptive field ──► workspace community ──► output columns (one per action / class)
                                        ▲        │                 │
                                        └────────┴── recurrent ◄───┘
```

- **Output cortical columns.** Each discrete action or image class is a community (column). The readout is the column with the most spikes in the decision window.
- **Workspace.** A central community receives the stimulus and broadcasts to the columns, following Global Neuronal Workspace Theory (see `derivation.md` §8).
- **Conventions.** $W_{ij}$ is the synapse from pre $j$ to post $i$. The flow $\pi$ follows spikes forward.

## 3. C1: Simulator

**Direction: Brian2.**

Why it fits:

- The dynamics and plasticity are written as equations, so the model stays close to the math.
- Map-STDP's per-neuron sums map onto Brian2's `(summed)` synaptic variables. Per-presynaptic sums (`..._pre = ... (summed)`) give $d_j$ and $e_j$, and per-postsynaptic sums give the E-step drive in `derivation.md` Eq. 2.
- The community labels $m(\cdot)$ can be neuron variables, and synapses can read them via `m_pre` / `m_post`.

Risks:

- **Closed-loop RL** needs `network_operation` (or an equivalent Python callback) to step the gym environment and set input rates. That works only in runtime mode, not `cpp_standalone`, so it will be slow for large networks.
- **Scale.** Large open-loop vision runs may need Brian2CUDA or Brian2GeNN (standalone). If those are still too slow, a PyTorch SNN library with batching is the fallback, at the cost of equation-level fidelity.

**Proposal.** Use Brian2 for M1–M4 (§10). Revisit when M4 wall-clock times are known.

## 4. C2: Applications

**Direction:** a preliminary with a basic discrete-action RL gym environment and image classification, then more complex architectures. Each action or class is a cortical column.

**Task ladder:**

| Track  | Step 1             | Step 2                                  | Stretch     |
| --- | --- | --- | --- |
| RL     | CartPole (2 cols)  | LunarLander (4 cols)                    | CarRacing (5 discrete) |
| Vision | MNIST / N-MNIST (10 cols) | Imagenette (10-class ImageNet subset) | ImageNet-1k (1000 cols) |

- **Vision input.** For Imagenette and beyond, raw pixels are too large for a local-rule SNN. The proposal is a **frozen pretrained feature encoder** (e.g. ResNet-18 penultimate features) with Poisson rate coding. The caveat is that the SNN then learns on top of features it did not learn itself. That should be stated in any result and ablated against raw-pixel encoding at the MNIST step.
- **Proposal: classification as a contextual bandit.** Each image is a one-step episode, with reward $+1$ when the correct column wins and $0$ or $-1$ otherwise. A single reward-modulated rule (§6) then serves RL and vision alike. A teacher current into the correct column is an ablation, not the default.

## 5. C3: Stimulus in the equation

**Direction:** include the stimulus in the equations, possibly by making it part of the environment.

**Proposal (adopted in `derivation.md`): stimulus as teleportation.** $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$, where $v(\mathbf o)$ is the normalized external drive. This:

- makes the flow well defined, since the chain is irreducible and the fixed point is unique;
- makes $\pi$, and hence the gating, stimulus-conditioned; and
- gives $\alpha$ a measurable meaning: the external share of total drive.

**Proposal (extension): environment as nodes.** Add sensor and actuator nodes, so that flow runs from action columns through the environment and back into the receptive field. The environment becomes a community in the map equation, and the codelength then accounts for agent–environment coupling. Try this after teleportation works.

**Open:**

- Whether teleportation steps should count as module exits (Infomap's "recorded teleportation" choice). The derivation currently does not count them.
- Whether $\alpha$ should be fixed, or estimated from the input and recurrent currents.

## 6. C4 + C5: Description length as a neuromodulator

**Direction:** there may be a better information-theoretic surrogate for thermodynamic cost than the map equation, and the structural term should be a 3-factor rule, since neuromodulators gate costly plasticity. Each candidate description length gets its own neuromodulator.

**Decision (2026-09-27): cost-modulated STDP is the implemented form.** See `derivation.md` §4. The rule is

$$
\Delta W_{ij} = \eta \left( R - \bar{R} \right) e_{ij} - \lambda \left( g_{ij} - \bar{g}_j \right) \kappa_{ij}
$$

where:

- $\kappa_{ij}$ is the causal pre→post coincidence;
- $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal** description cost of the transition, a $K \times K$ module-pair table broadcast per module;
- $\bar{g}_j$ is a per-neuron baseline.

On average this is weight-scaled gradient descent on $D$ (`derivation.md` Theorem 3). The factorized exact gradient is kept only as the analysis reference and as simulation baseline A: the additive rule $\eta \mathrm{STDP} - \eta' G$. This supersedes the earlier variants A/B/C.

Why this form:

- **Crossbars.** It is a single rule for memristive crossbars (§7).
- **No trace for structure.** Description costs are instantaneous, so the structural term needs no eligibility trace. Only delayed task reward does.
- **Marginal, not pointwise.** The modulator must be the marginal cost. Broadcasting pointwise cost, such as each step's codeword length, is biased (checked in simulation).

**Candidates** (`derivation.md` §5 has the modulator, pointwise cost and caveats for each):

| Candidate | Modulator $g_{ba}$ for a step from module $a$ to module $b$ | Thermodynamic reading | Status |
| --- | --- | --- | --- |
| Map equation | $M^{\ast}_a \mathbb{I}[b \ne a]$: fires only on exits | per-transition surprise of crossing modules | baseline; closed form |
| Entropy rate $h_K$ | $-\log T_{ba}$ | minimum information per step | closed form; favours determinism, not modularity |
| Entropy production $\sigma_K$ | $\log (J_{ba}/J_{ab}) - J_{ab}/J_{ba}$ | is entropy production (coarse-grained lower bound) | closed form; no modular pressure alone |
| Cost-weighted map equation | map modulator $+ \lambda_E c_{ba}$ | explicit energy; $\lambda_E$ = metabolic state | closed form |
| Markov stability | $-\mathbb{I}[b = a]$ at lag 1 | none | closed form; lag 1 gives exactly $G$ |
| SBM description length | $\log \frac{1 - \omega_{ba}}{\omega_{ba}}$ at rewiring | none | gates structural plasticity |
| Predictive dissipation | per-module pointwise $I_{mem} - I_{pred}$ | lower bound on dissipated work | estimator only |
| Cross-module information flow | pointwise transfer entropy between modules | enters each module's entropy balance | estimator only |

**How to decide.** Score each candidate on:

1. task performance;
2. whether modules emerge and match the columns (NMI);
3. correlation with a simulated **energy proxy** (per-spike cost, plus a per-synaptic-event cost that scales with weight, plus a wiring-distance term);
4. learning variance and speed of the 3-factor estimate relative to its exact-gradient baseline.

**Open:**

- **Energy model.** A concrete one is still needed, e.g. Attwell & Laughlin-style ATP accounting.
- **Estimators.** Which ones to use for predictive dissipation and information flow.
- **Composite modulators.** Whether candidates should be combined. For example, map equation plus $\lambda \sigma_K$, because $\sigma$ alone exerts no modular pressure.
- **Lateral inhibition.** How $M^{\ast}$ is defined when modules are implicit.

## 7. C6: Hardware target: memristive crossbars

**Direction:** the rule must be implementable on memristive devices.

Constraints this places on every rule (see `derivation.md` §6):

- **Updates** are pulse-coincidence STDP scaled per row by a broadcast signal: one phase per target-module column mask.
- **No nonlinear function of an individual device's conductance** is allowed, and no access to the transposed element $W_{ji}$. This is why every modulator is defined at module level.
- **Allowed reads.** Per-row sums come from $K + 1$ masked crossbar reads ($d_j$, $e_j$, module flows). Modulators and baselines are computed in peripheral logic from $K \times K$ statistics.
- **Per-synapse state.** The only per-synapse state beyond $W$ is the task eligibility trace. Volatile, diffusive memristors are one candidate for it.

**Open:**

- device non-idealities (update asymmetry and nonlinearity, limited conductance levels, noise);
- whether to prototype on a crossbar simulator (e.g. with device models) after M3.

## 8. Open questions carried from the derivation

- **Partition $M$.** Static (SBM initialization) or periodically re-detected (Infomap, Leiden)? For fixed-output tasks, the columns probably pin $M$ for the output communities, and only the workspace would be re-detected.
- **Structural plasticity.** Threshold pruning ($W_{ij} < \epsilon$) vs. top-$k$ per neuron.
- **Initialization.** SBM (head start, bias) vs. Erdős–Rényi (neutral, slow).
- **Locality convention.** The forward walk ties flow to firing and maps onto crossbar row sums. The backward, dendritically normalized walk has postsynaptic sums but loses that tie. See `derivation.md` §6.
- **Lateral-inhibition variant.** The anti-Hebbian $\Delta I_{ij} = \eta\pi_i\pi_j$ is unbounded and treats co-active neurons as competitors, which conflicts with Hebbian STDP. It also ignores Dale's law. It needs a bounded, correctly signed rule, probably through interneuron populations.
- **Mean-field validity.** Does $\pi \approx$ normalized firing rate, and do causal pairings occur at a rate proportional to $J_{ij}$ (`derivation.md` §2.2)? Theorem 3 depends on both. Measure them at M1.

## 9. Metrics and ablations

**Metrics:**

- Task: accuracy, episodic return.
- Structure: $L(M)$ and each §6 candidate over training; NMI between the Infomap-detected partition and the intended columns; mean exit fraction $\bar e$.
- Energy: firing rates and the energy proxy.

**Ablations:** STDP only · exact-gradient baseline (additive $G$) · cost-modulated rule per candidate · per-module vs. global modulator · per-neuron vs. per-module baseline · teleportation on/off · static vs. re-detected $M$.

## 10. Milestones

| ID | Milestone | Done when |
| --- | --- | --- |
| M0 | Docs: corrected derivation, spec, changelog | This commit |
| M1 | Brian2 rule prototype on a small SBM graph (no task) | Online $\pi$ estimate matches power iteration; $L(M)$ decreases under both the cost-modulated rule and the exact-gradient baseline; mean-field and pairing-rate checks reported |
| M2 | CartPole, 2 columns, closed loop | Beats random policy; modules persist |
| M3 | MNIST as contextual bandit, 10 columns | Accuracy and NMI reported for all ablations |
| M4 | LunarLander (4) and Imagenette with frozen encoder | Runs end to end; wall-clock informs the simulator decision |
| M5 | Modulator comparison (§6): all eight candidates | Candidates scored on the four criteria |
| M6 | Scale-up: ImageNet-1k, hierarchical/multilevel map equation, richer architectures | Open |
