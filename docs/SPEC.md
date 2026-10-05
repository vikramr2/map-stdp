# Map-STDP Project Spec

Status: ideation, with no code yet. The math lives in [`derivation.md`](derivation.md), and changes are logged in [`CHANGELOG.md`](CHANGELOG.md).

Each consideration below states the user's direction first. Items marked **Proposal** are candidate approaches that have not been decided. **Open** items are unresolved.

## 1. Goal and hypothesis

Give a spiking network a **state-space abstraction in which its communities are the states**, and use it to solve RL tasks.

**Approach.** A random walk on a graph with communities has a minimum description length: the map equation, or an alternative (§6). Community detection finds the partition that minimises that length for a given graph. **Map-STDP inverts this.** The partition is given, and a three-factor STDP rule shapes the weights until the given partition is the minimum-description-length one (`derivation.md` §3.4). The coarse-grained walk between communities is then a Markov model over the network's states, conditioned on the stimulus. It can be extracted, and it chooses the action (`derivation.md` §2.4).

**Hypotheses.**

1. **Modules form.** The communities emerge and match the intended controller, latent and action roles (NMI).
2. **The state model works.** The extracted chain $T^K(\mathbf o)$ gives a policy that solves basic RL tasks. With carry-over, the latent communities give a predictive model of hidden state.
3. **Side tie: thermodynamics.** A neuron pays more metabolic cost to communicate with cortices it does not normally talk to. If description length tracks that cost, the information dynamics the rule minimises are a model of thermodynamics in the brain, and the rule lowers a biophysical energy proxy. This is secondary: the state abstraction does not depend on it.

Map-STDP ([`derivation.md`](derivation.md), Eq. 4) is the rule: reward-modulated STDP for the task, plus cost-modulated STDP whose neuromodulator reports the marginal description length. Which description length is best is an open question (§6).

## 2. Architecture

```text
stimulus o_f ──► controller C ─────────────► action communities A ──► action
                   ▲      │                        ▲
                   │      ▼                        │
                   └── latent communities L ───────┘
                              │      ▲
                              └──────┘  carry-over ρ: frame f → f+1
```

There are three community types (`derivation.md` §2):

- **Controller $\mathcal C$.** One central community that receives the stimulus: $v(\mathbf o)$ is concentrated here. It routes flow to latent and action communities. It is the analogue of the cerebral cortex integrating input and selecting among motor programmes, and of the workspace in Global Workspace Theory. Because the stimulus enters here, the extracted chain is stimulus-conditioned (`derivation.md` §2.4).
- **Action communities $\mathcal A$.** One per discrete action. The policy is the flow share of each action community, or equivalently the column with the most spikes in the frame window (`derivation.md` Eq. 1d).
- **Latent communities $\mathcal L$.** Abstract states that are not tied to an action. With carry-over they persist across frames and form a learned world model.

**Conventions.** $W_{ij}$ is the synapse from pre $j$ to post $i$. The flow $\pi_f$ follows spikes forward and is computed once per frame $f$.

## 3. C1: Simulator

**Direction: Brian2.**

Why it fits:

- The dynamics and plasticity are written as equations, so the model stays close to the math.
- Map-STDP's per-neuron sums map onto Brian2's `(summed)` synaptic variables. Per-presynaptic sums (`..._pre = ... (summed)`) give $d_j$ and $e_j$, and per-postsynaptic sums give the E-step drive in `derivation.md` Eq. 2.
- The community labels $m(\cdot)$ can be neuron variables, and synapses can read them via `m_pre` / `m_post`.
- The implicit per-frame walk (§5) is $n$ sparse matrix–vector products, cheap to run in the same `network_operation` that steps the environment.

Risks:

- **Closed-loop RL** needs `network_operation` (or an equivalent Python callback) to step the gym environment and set input rates. That works only in runtime mode, not `cpp_standalone`, so it will be slow for large networks.
- **Scale.** If runtime mode is too slow, options are Brian2CUDA or Brian2GeNN (standalone, which complicates the closed loop), or a PyTorch SNN library with batching, at the cost of equation-level fidelity.

**Proposal.** Use Brian2 for M1–M4 (§11). Revisit when M3 wall-clock times are known.

## 4. C2: Applications

**Direction:** basic discrete-action RL environments first, several of them. Each action is a community.

**Task ladder (Proposal):**

| Step | Environment | Actions | What it tests |
| --- | --- | --- | --- |
| 1 | CartPole | 2 | closed loop, controller routing |
| 2 | Acrobot, MountainCar | 3 each | more actions; MountainCar's sparse reward |
| 3 | LunarLander | 4 | richer observation |
| 4 | a partially observable task, e.g. a T-maze or MiniGrid memory task | 3–4 | latent communities with carry-over must hold hidden state |

- **Observation encoding.** Continuous observations become $v(\mathbf o)$ over controller neurons, e.g. with Gaussian receptive fields per observation dimension. **Open**: the encoding and the controller size.
- **Deferred: vision.** Image classification as a contextual bandit (one-step episodes, one community per class), MNIST and Imagenette with a frozen feature encoder. This is out of scope until the RL ladder works.

## 5. C3 + C7: Stimulus, frames and timescale

**Direction:** the stimulus is in the equations, each environment frame is one random walk, and the walk must be implicit or very fast so that timescales do not become a problem.

**Adopted: stimulus as teleportation.** $\pi_f = (1-\alpha)T\pi_f + \alpha v_f$ with $v_f = v(\mathbf o_f)$. This:

- makes the flow well defined, since the chain is irreducible and the fixed point is unique;
- makes $\pi_f$, and hence the modulators and the extracted chain, stimulus-conditioned; and
- gives $\alpha$ a measurable meaning: the external share of total drive.

**Proposal: implicit walk per frame.** $\pi_f = \alpha (I - (1-\alpha)T)^{-1} v_f$ is approximated by $n$ hops of power iteration, with error $\le 2(1-\alpha)^n$ independent of network size (`derivation.md` Eq. 1a). Modular networks approach this bound, so budget for it: $\alpha = 0.2$ and $n = 20$ give $\le 2.3\%$. The three realisations are:

- software reference: $n$ sparse matrix–vector products;
- crossbar: $n$ analogue reads;
- spiking: about one synaptic delay per hop, so about 20 ms per frame.

The software reference is the default. The spiking estimate is the biological and hardware variant, and its agreement with the reference is a milestone (M1).

**Proposal: carry-over.** Latent state persists across frames by mixing the previous frame's latent flow into the teleportation vector with weight $\rho$ (`derivation.md` Eq. 1b). With $\rho = 0$, frames are independent, which suffices for fully observed tasks.

**Timescale ordering:** walk hops, then frames, then plasticity.

**Proposal (extension): environment as nodes.** Add sensor and actuator nodes, so that flow runs from action communities through the environment and back into the controller. The environment becomes a community in the map equation, and the codelength then accounts for agent–environment coupling. Try this after teleportation works.

**Open:**

- $n$, $\alpha$ and $\rho$, and whether $\alpha$ should be estimated from input and recurrent currents.
- Whether teleportation and carry-over steps should count as module exits (Infomap's "recorded teleportation" choice). The derivation currently does not count them.
- Whether the spiking estimate settles within a frame window short enough for the environment.

## 6. C4 + C5: Description length as a neuromodulator

**Direction:** there may be a better description length than the map equation, and the structural term should be a three-factor rule, since neuromodulators gate costly plasticity. Each candidate description length gets its own neuromodulator.

**Decision (2026-09-27): cost-modulated STDP is the implemented form.** See `derivation.md` §4. The rule is

$$
\Delta W_{ij} = \eta \left( R - \bar{R} \right) e_{ij} - \lambda \left( g_{ij} - \bar{g}_j \right) \kappa_{ij}
$$

where:

- $\kappa_{ij}$ is the causal pre→post coincidence;
- $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal** description cost of the transition, a $K \times K$ module-pair table broadcast per module;
- $\bar{g}_j$ is a per-neuron baseline.

On average this is weight-scaled gradient descent on $D$ (`derivation.md` Theorem 3). In an environment, $D$ is the frame-averaged description length, and the theorem holds frame by frame (`derivation.md` §3.4). The factorized exact gradient is kept only as the analysis reference and as simulation baseline A: the additive rule $\eta \mathrm{STDP} - \eta' G$.

Why this form:

- **Crossbars.** It is a single rule for memristive crossbars (§7).
- **No trace for structure.** Description costs are instantaneous, so the structural term needs no eligibility trace. Only delayed task reward does.
- **Marginal, not pointwise.** The modulator must be the marginal cost. Broadcasting pointwise cost, such as each step's codeword length, is biased (checked in simulation).

**Candidates** (`derivation.md` §5 has the modulator, pointwise cost and caveats for each):

| Candidate | Modulator $g_{ba}$ for a step from module $a$ to module $b$ | As a state abstraction | Thermodynamic reading | Status |
| --- | --- | --- | --- | --- |
| Map equation | $M^{\ast}_a \mathbb{I}[b \ne a]$: fires only on exits | sticky, well-separated states | per-transition surprise of crossing modules | baseline; closed form |
| Entropy rate $h_K$ | $-\log T_{ba}$ | deterministic transitions, not sticky states | minimum information per step | closed form; favours determinism, not modularity |
| Entropy production $\sigma_K$ | $\log (J_{ba}/J_{ab}) - J_{ab}/J_{ba}$ | directed state sequences | is entropy production (coarse-grained lower bound) | closed form; no modular pressure alone |
| Cost-weighted map equation | map modulator $+ \lambda_E c_{ba}$ | as map equation, avoiding costly transitions | explicit energy; $\lambda_E$ = metabolic state | closed form |
| Markov stability | $-\mathbb{I}[b = a]$ at lag 1 | states persisting for $\tau$ steps | none | closed form; lag 1 gives exactly $G$ |
| SBM description length | $\log \frac{1 - \omega_{ba}}{\omega_{ba}}$ at rewiring | block wiring under the states | none | gates structural plasticity |
| Predictive dissipation | per-module pointwise $I_{mem} - I_{pred}$ | states keep only what predicts | lower bound on dissipated work | estimator only |
| Cross-module information flow | pointwise transfer entropy between modules | states exchange little information | enters each module's entropy balance | estimator only |
| **Proposal:** latent-chain entropy rate $h_{\mathcal L}$ | not yet derived | predictable latent dynamics (world model) | none | needs carry-over; `derivation.md` §5.9 |

**How to decide.** Score each candidate on:

1. task performance (episodic return);
2. whether modules emerge and match the controller, latent and action roles (NMI);
3. quality of the extracted state model: agreement between the policy from $T^K(\mathbf o)$ and the spiking readout, and, with carry-over, how well the latent chain predicts the next latent state;
4. learning variance and speed of the three-factor estimate relative to its exact-gradient baseline;
5. secondary: correlation with a simulated **energy proxy** (per-spike cost, plus a per-synaptic-event cost that scales with weight, plus a wiring-distance term).

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
- **Allowed reads.** Per-row sums come from $K + 1$ masked crossbar reads ($d_j$, $e_j$, module flows). The per-frame walk is $n$ ordinary reads. Modulators and baselines are computed in peripheral logic from $K \times K$ statistics.
- **Per-synapse state.** The only per-synapse state beyond $W$ is the task eligibility trace. Volatile, diffusive memristors are one candidate for it.

**Open:**

- device non-idealities (update asymmetry and nonlinearity, limited conductance levels, noise);
- whether to prototype on a crossbar simulator (e.g. with device models) after M3.

## 8. Open questions carried from the derivation

- **Partition $M$.** The controller and action communities are pinned, because they carry the input and the policy. Latent communities: static (SBM initialization) or periodically re-detected (Infomap, Leiden), which makes training an alternating minimisation (`derivation.md` §3.4).
- **Number of latent communities** $\lvert \mathcal L \rvert$: a hyperparameter, or set by re-detection.
- **World model.** The definition and estimator of the across-frame latent chain (`derivation.md` §2.4) are proposals, not yet checked in simulation.
- **Structural plasticity.** Threshold pruning ($W_{ij} < \epsilon$) vs. top-$k$ per neuron.
- **Initialization.** SBM (head start, bias) vs. Erdős–Rényi (neutral, slow).
- **Locality convention.** The forward walk ties flow to firing and maps onto crossbar row sums. The backward, dendritically normalized walk has postsynaptic sums but loses that tie. See `derivation.md` §6.
- **Lateral-inhibition variant.** The anti-Hebbian $\Delta I_{ij} = \eta\pi_i\pi_j$ is unbounded and treats co-active neurons as competitors, which conflicts with Hebbian STDP. It also ignores Dale's law. It needs a bounded, correctly signed rule, probably through interneuron populations.
- **Mean-field validity.** Does $\pi \approx$ normalized firing rate, and do causal pairings occur at a rate proportional to $J_{ij}$ (`derivation.md` §2.2)? Theorem 3 depends on both. Measure them at M1.

## 9. Metrics

- **Task:** episodic return, compared with a random policy and a standard RL baseline.
- **Structure:** $L(M)$ and each §6 candidate over training; NMI between the Infomap-detected partition and the intended communities; mean exit fraction $\bar e$.
- **State model:**
  - KL divergence between the policy from $T^K(\mathbf o)$ (`derivation.md` Eq. 1d) and the spiking readout;
  - error of the implicit walk vs. the spiking estimate of $\pi_f$;
  - with carry-over, next-latent-state prediction accuracy.
- **Energy (secondary):** firing rates and the energy proxy.

## 10. Ablations

- STDP only
- exact-gradient baseline (additive $G$)
- cost-modulated rule per candidate
- per-module vs. global modulator
- per-neuron vs. per-module baseline
- per-frame vs. running-average modulator table
- implicit vs. spiking walk
- carry-over on/off ($\rho$)
- latent communities on/off
- static vs. re-detected latent partition

## 11. Milestones

| ID | Milestone | Done when |
| --- | --- | --- |
| M0 | Docs: corrected derivation, spec, changelog; reformulated around communities as states | Done (2026-09-27, reformulated 2026-10-04) |
| M1 | Brian2 rule prototype on a small graph with controller, latent and action modules (no task) | Implicit per-frame walk matches power iteration; spiking estimate agrees with it within a frame window; $L(M)$ decreases under the cost-modulated rule and the exact-gradient baseline; mean-field and pairing-rate checks reported |
| M2 | CartPole, closed loop | Beats random policy; modules persist; policy from $T^K$ agrees with the spiking readout |
| M3 | Acrobot, MountainCar, LunarLander | Runs end to end; returns reported for all ablations; wall-clock informs the simulator decision |
| M4 | Partially observable task with latent communities and carry-over | Carry-over beats $\rho = 0$; latent chain predicts the next latent state |
| M5 | Modulator comparison (§6): all candidates | Candidates scored on the five criteria |
| M6 | Scale-up: hierarchical/multilevel map equation, richer architectures, deferred vision track | Open |
