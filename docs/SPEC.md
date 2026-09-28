# Map-STDP Project Spec

Status: ideation, with no code yet. The math lives in [`derivation.md`](derivation.md), and changes are logged in [`CHANGELOG.md`](CHANGELOG.md).

Each consideration below states the user's direction first. Items marked **Proposal** are candidate approaches that have not been decided. **Open** items are unresolved.

## 1. Goal and hypothesis

Find a learning framework in which **information-dynamic entropy is a surrogate for the thermodynamic entropy (metabolic cost) of neurons**.

**Hypothesis.** A neuron pays more thermodynamic cost to communicate with neurons in cortices it does not normally talk to, and less within its usual partners. A description length of neural activity flow, such as the map equation, captures this: rare, cross-module transitions get long codewords and frequent, within-module transitions get short ones. A plasticity rule that descends this description length alongside a task-learning rule should therefore:

1. form modular cortices, one per concept;
2. keep task performance; and
3. reduce a biophysical energy proxy.

Map-STDP ([`derivation.md`](derivation.md), Eq. 6) is the current rule. Whether the map equation is the *right* description length is itself a question (§6).

## 2. Architecture

```
stimulus o ──► receptive field ──► workspace community ──► output columns (one per action / class)
                                        ▲        │                 │
                                        └────────┴── recurrent ◄───┘
```

- **Output cortical columns.** Each discrete action or image class is a community (column). The readout is the column with the most spikes in the decision window.
- **Workspace.** A central community receives the stimulus and broadcasts to the columns, following Global Neuronal Workspace Theory (see `derivation.md` §7).
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
- **Proposal: classification as a contextual bandit.** Each image is a one-step episode, with reward $+1$ when the correct column wins and $0$ or $-1$ otherwise. A single reward-modulated rule (§7) then serves RL and vision alike. A teacher current into the correct column is an ablation, not the default.

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

## 6. C4: The right description length

**Direction:** determine whether there is a better information-theoretic description-length formulation than the map equation to serve as a thermodynamic surrogate. If the map equation holds up, keep it.

**Proposal: candidates to compare.**

| Candidate | What it measures | Relation to thermodynamics | Notes |
| --- | --- | --- | --- |
| Map equation $L(M)$ | Two-level codelength of the flow given partition $M$ | Per-transition codeword length ≈ "cost" of a spike transmission; rare cross-module hops cost more | Baseline; gives Map-STDP |
| Entropy rate $h = -\sum_j \pi_j \sum_i T_{ij}\log T_{ij}$ | Unavoidable information per step | Partition-free lower bound, $L(M) \ge h$ | $L(M) - h$ = overhead of the modular code |
| Entropy production $\sigma = \tfrac12\sum_{ij}(J_{ij}-J_{ji})\log\frac{J_{ij}}{J_{ji}}$, $J_{ij}=\pi_jT_{ij}$ | Irreversibility of the flow | *Is* the thermodynamic entropy production of a Markov jump process (Schnakenberg) | Needs reciprocal links; zero for detailed balance |
| Cost-weighted codelength $L(M) + \lambda\sum_{ij} J_{ij}c_{ij}$ | Codelength plus metabolic or wiring cost per transmission | Landauer bridge: ≥ $k_BT\ln 2$ per bit | Per-edge cost $c_{ij}$ can encode distance or cross-cortex expense |
| Markov stability; SBM description length (Peixoto) | Partition quality at multiple timescales / as a generative-model MDL | Indirect | Alternatives if the map equation's partitions are poor |

**How to decide.** Score each candidate on:

1. whether its gradient (under the EM approximation) is local;
2. task performance when used as the structural term;
3. whether modules emerge and match the columns (NMI);
4. correlation with a simulated **energy proxy**: per-spike cost plus a per-synaptic-event cost that scales with weight and a wiring-distance term.

**Open:** a concrete energy model. One starting point is Attwell & Laughlin-style ATP accounting: spikes, synaptic transmission, resting potential.

## 7. C5: 3-factor reduction

**Direction:** consider reducing the rule to a 3-factor STDP rule.

Common structure: an eligibility trace $\dot e_{ij} = -e_{ij}/\tau_e + \mathrm{STDP}_{ij}(t)$, and $\Delta W_{ij} = \eta M(t) e_{ij}$, where $M(t)$ is a neuromodulator (reward-prediction error $R - \bar R$).

**Proposal: variants.**

- **(A) Additive baseline.** $\Delta W_{ij} = \eta M(t)e_{ij} - \eta' G(i,j)$. The map term stays a separate, unmodulated heterosynaptic term.
- **(B) Map term folded into the third factor.** $\Delta W_{ij} = \eta M(t) g(\bar e_j, \chi_{ij}) e_{ij}$, where $g$ gates plasticity by community leakage. For example, $g = 1 - \lambda(\chi_{ij} - \bar e_j)$ damps potentiation on cross-module synapses when the presynaptic neuron is leaky. This is the "true" 3-factor reduction and the main hypothesis.
- **(C) Two timescales.** Fast reward-modulated STDP, with the map term applied as a slow consolidation or homeostatic step (e.g. once per episode). This also relaxes the EM separation problem.

The key comparison is A vs. B: does folding the structure pressure into the modulator preserve modularity and performance?

**Open:** $G$ is signed and has no eligibility of its own. In (B), the map term acts only when $M(t) \neq 0$, so modules may not form in the absence of reward. This needs checking.

## 8. Open questions carried from the derivation

- **Partition $M$.** Static (SBM initialization) or periodically re-detected (Infomap, Leiden)? For fixed-output tasks, the columns probably pin $M$ for the output communities, and only the workspace would be re-detected.
- **Structural plasticity.** Threshold pruning ($W_{ij} < \epsilon$) vs. top-$k$ per neuron.
- **Initialization.** SBM (head start, bias) vs. Erdős–Rényi (neutral, slow).
- **Locality convention.** The forward walk ties flow to firing but needs presynaptic (axonal) sums. The backward, dendritically normalized walk has postsynaptic sums but loses that tie. See `derivation.md` §5.
- **Lateral-inhibition variant.** The anti-Hebbian $\Delta I_{ij} = \eta\pi_i\pi_j$ is unbounded and treats co-active neurons as competitors, which conflicts with Hebbian STDP. It also ignores Dale's law. It needs a bounded, correctly signed rule, probably through interneuron populations.
- **Mean-field validity.** Does $\pi \approx$ normalized firing rate (`derivation.md` §2.2) hold in the simulated regime? Measure it at M1.

## 9. Metrics and ablations

**Metrics:**

- Task: accuracy, episodic return.
- Structure: $L(M)$ and each §6 candidate over training; NMI between the Infomap-detected partition and the intended columns; mean exit fraction $\bar e$.
- Energy: firing rates and the energy proxy.

**Ablations:** STDP only · STDP + map term · 3-factor (A) · 3-factor (B) · teleportation on/off · static vs. re-detected $M$.

## 10. Milestones

| ID | Milestone | Done when |
| --- | --- | --- |
| M0 | Docs: corrected derivation, spec, changelog | This commit |
| M1 | Brian2 rule prototype on a small SBM graph (no task) | Online $\pi$ estimate matches power iteration; $L(M)$ decreases under Map-STDP; mean-field check reported |
| M2 | CartPole, 2 columns, closed loop | Beats random policy; modules persist |
| M3 | MNIST as contextual bandit, 10 columns | Accuracy and NMI reported for all ablations |
| M4 | LunarLander (4) and Imagenette with frozen encoder | Runs end to end; wall-clock informs the simulator decision |
| M5 | Description-length comparison (§6) | Candidates scored on the four criteria |
| M6 | Scale-up: ImageNet-1k, hierarchical/multilevel map equation, richer architectures | Open |
