# Map-STDP: communities as states

Author: Vikram Ramavarapu

**Goal.** Give a spiking network a **state-space abstraction** whose states are its own communities. A random walk on a graph with communities has a minimum description length: the map equation, or one of the alternatives in §5. Community detection solves the *forward* problem, finding the partition that minimises the description length of a walk on a given graph. Map-STDP solves the **inverse** problem. The partition is given, with one controller community, one community per action and some latent communities, and plasticity shapes the synapses until that partition is the minimum-description-length one. The coarse-grained walk between communities is then a Markov model over the network's states, conditioned on the stimulus. It can be read out, and in an RL environment it chooses the action.

**Side tie: thermodynamics.** Neurons pay a real metabolic and thermodynamic cost to communicate. We hypothesise that this cost is higher when activity crosses into cortices a neuron rarely talks to. If description length tracks that cost, the information dynamics that the rule minimises double as a model of thermodynamics in the brain. This is a secondary hypothesis, tested through an energy proxy (§1).

STDP is local, so on its own it has no reason to organise a network into cortices: dense modules with sparse links between them. Map-STDP makes structure a *neuromodulated* term. Each module has a neuromodulator that reports how much a candidate **description length** $D$ grows when activity crosses a given transition, and that signal gates STDP. The map equation is the baseline $D$. Seven alternatives plug into the same rule (§5).

**Each environment frame is one walk.** The stimulus of frame $t$ re-injects the walker, and the walk is run implicitly to its per-frame flow $\pi_t$ in a fixed number of hops (§2.3). Description length, modulators and the extracted Markov model are all per frame.

This document is a corrected and consolidated rewrite of an earlier derivation, reformulated on 2026-10-04 around communities as states. Appendix B lists what changed.

## At a glance

For every causal spike pairing $j \to i$ (pre $j$ fires, then post $i$ within the STDP window; indicator $\kappa_{ij}(t)$), the synapse changes by

$$
\Delta W_{ij}(t) = \underbrace{\eta \left( R(t) - \bar{R} \right) e_{ij}(t)}_{\text{task: reward-modulated STDP}} - \underbrace{\lambda \left( g_{ij} - \bar{g}_j \right) \kappa_{ij}(t)}_{\text{structure: cost-modulated STDP}}
$$

where:

- $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal description cost** of one more unit of flow from $j$ to $i$. It depends only on the modules of $i$ and $j$, so each module's neuromodulator can broadcast it.
- $\bar{g}_j$ is its average over $j$'s outgoing transitions, a per-neuron baseline.
- $e_{ij}$ is an STDP eligibility trace, needed only because task reward arrives late.

How to read it:

- **The modulator reports cost.** A transition that makes activity more expensive to describe than $j$'s usual transitions ($g_{ij} > \bar{g}_j$) depresses the synapse that produced it. A cheaper-than-usual transition potentiates it.
- **On average it is gradient descent** (Theorem 3): $\mathbb{E}[\Delta W_{ij}] = -\lambda W_{ij} \partial D / \partial W_{ij}$. This is multiplicative, weight-scaled descent of the description length.
- **The modulator must be the *marginal* cost, not the cost itself** (§4.3). Broadcasting the codeword length of each step is biased.
- **For the map equation**, the modulator fires only when activity *leaves* a module, with strength $M^{\ast}_m$ equal to the marginal bits per exit. On average this reproduces the familiar gating:

| Synapse $j \to i$ | Average structural update | Effect |
| --- | --- | --- |
| $i$ in a **different** module from $j$ | $\propto -(1 - \bar{e}_j)$ | weakened |
| $i$ in the **same** module as $j$ | $\propto +\bar{e}_j$ | strengthened |

- **It suits memristive crossbars** (§6). The update is a pulse-coincidence STDP event scaled by a per-module signal. There are no per-synapse nonlinear reads, no transposed access, and no per-synapse trace for the structural term, because description costs are instantaneous.
- **The communities are the states** (§2.4). In each frame, the module-to-module flow defines a Markov chain $T_K(\mathbf o)$ over communities. Its flow into the action communities is the policy, and its flow between latent communities across frames is a learned world model.

## 1. Motivation

- **Communities as states.** An agent needs a compact state space, but a spiking network's microstate (which neurons fire) is far too large to serve as one. If the network is modular, the coarse-grained walk between modules is a small Markov chain that can be read out. When modules map onto actions and latent situations, the network carries its own abstract model of the task: which latent state it is in, and which action that state leads to.
- **A central controller.** The stimulus enters one central controller community, which routes activity to latent and action communities. This is loosely analogous to the cerebral cortex, which integrates sensory input and selects among motor programmes. It also follows the broadcast architecture of Global Workspace Theory [Baars 2005]. The controller is what makes the extracted chain depend on the stimulus (§2.4).
- **Cortices are useful codes.** When modules map onto concepts, population activity becomes a readable spatial code. Each discrete action gets a module.
- **Neural activity is a random walk.** Stochastic spiking networks sample from a distribution over activity patterns [Buesing et al. 2011]. The map equation [Rosvall & Bergstrom 2008] scores modularity by the description length of a random walk, so it applies naturally to spiking.
- **Plasticity is expensive, and neuromodulators gate it.** Memory formation has a measurable metabolic cost. Starving *Drosophila* switch off protein-synthesis-dependent long-term memory, and forcing it shortens their survival [Plaçais & Preat 2013]. Distributing learning between cheap transient changes and costly consolidation saves energy manifold [Li & van Rossum 2020]. Three-factor rules, where a local eligibility flag becomes a weight change only when a neuromodulator arrives, have direct experimental support [Gerstner et al. 2018].
- **Side hypothesis (thermodynamics).** The neuromodulator *carries a description-length signal*, so plasticity is spent where it reduces the information-theoretic cost of the network's activity. If that cost tracks metabolic cost, the same rule also makes the network thermodynamically efficient. The state abstraction does not depend on this hypothesis; it is tested separately against an energy proxy.
- **Free energy.** Under the free-energy view [Friston 2010], the sampling distribution is the network's recognition density. The structural term organises that same distribution into modules instead of adding an unrelated goal.

## 2. Setup: activity as flow

### Notation

| Symbol | Meaning | Where it lives |
| --- | --- | --- |
| $W_{ij} \ge 0$ | synapse from pre $j$ to post $i$ | synapse |
| $z_j(t) \in \lbrace 0, 1 \rbrace$ | spike of neuron $j$ at step $t$ | neuron $j$ |
| $d_j = \sum_k W_{kj}$ | total outgoing weight of $j$ | neuron $j$ (axonal) |
| $T_{ij} = W_{ij} / d_j$ | probability that the walker at $j$ steps to $i$ | synapse |
| $J_{ij} = \pi_j T_{ij}$ | flow along the link $j \to i$ | synapse |
| $v(\mathbf{o})$ | normalised external drive from stimulus $\mathbf{o}$ | input |
| $\alpha$ | share of activity injected by the stimulus each step | global constant |
| $\pi_j$, $\pi_{f,j}$ | flow through $j$; in frame $f$, $\pi_f = \pi(\mathbf{o}_f)$ | neuron $j$ |
| $f$, $n$ | environment frame; walk hops per frame | global |
| $m(j)$, $K$ | module of $j$; number of modules | neuron $j$ |
| $\mathcal{C}$, $\mathcal{A}$, $\mathcal{L}$ | controller module; action modules (one per action); latent modules, $K = 1 + \lvert \mathcal{A} \rvert + \lvert \mathcal{L} \rvert$ | module labels |
| $e_j = \sum_{i \notin m(j)} W_{ij}$, $\bar{e}_j = e_j / d_j$ | outgoing weight leaving $j$'s module, absolute and as a fraction | neuron $j$ (axonal) |
| $p_m = \sum_{j \in m} \pi_j$ | visit rate of module $m$ | module |
| $q_m = \sum_{j \in m} \pi_j \bar{e}_j$ | exit rate of module $m$ | module |
| $q_{\curvearrowright} = \sum_m q_m$ | total rate of switching modules | network |
| $J_{ba} = \sum_{i \in b, j \in a} J_{ij}$, $T^K_{ba} = J_{ba} / p_a$ | flow and transition probability from module $a$ to module $b$ | module pair |

The membrane equation of a sampling SNN [Buesing et al. 2011] is $u_i = b_i + \sum_j W_{ij} z_j(t)$, so activity flows from $j$ to $i$.

### 2.1 The walker follows the spikes

A walker moves along synapses in the direction spikes travel. From $j$ it steps to $i$ with probability $T_{ij}$. On each step there is also a chance $\alpha$ that it is instead re-injected by the stimulus at a neuron drawn from $v(\mathbf{o})$. The flow $\pi$ is the long-run fraction of time spent at each neuron:

$$
\pi = (1 - \alpha) T \pi + \alpha v(\mathbf{o}), \qquad \sum_i \pi_i = 1 \qquad \text{(1)}
$$

This is personalised PageRank with the stimulus as the personalisation vector. The stimulus is therefore part of the equation. For any $\alpha > 0$, $\pi$ exists, is unique, and power iteration reaches it (Theorem 1). In an environment, (1) is solved once per frame with that frame's stimulus (§2.3).

### 2.2 Estimating flow from spikes (E-step)

Each neuron estimates its own flow from its input:

$$
\hat{\pi}_i \leftarrow (1 - \beta) \hat{\pi}_i + \beta \left[ (1 - \alpha) \sum_j \frac{W_{ij}}{d_j} \hat{z}_j(t) + \alpha v_i \right] \qquad \text{(2)}
$$

where $\hat{z}(t) = z(t) / \sum_k z_k(t)$, taken as zero when nothing fires.

Within a frame, (2) runs over that frame's simulation steps, with $\hat{\pi}$ reset to $v(\mathbf{o}_f)$ at the start of the frame or carried over from the previous frame (§2.3).

**Assumption (mean field).** The normalised firing rates satisfy the same balance as (1). This holds approximately for linear rate responses, presynaptic normalisation, and a stimulus share $\alpha$ of the drive. Two things follow:

- $r = \pi$, and (2) tracks $\pi$ with $O(\beta)$ noise.
- **Causal spike pairings $j \to i$ occur at a rate proportional to $J_{ij}$.**

The second consequence is what lets spike coincidences stand in for walker steps in §4. Whether the assumption holds is something to measure.

### 2.3 One walk per frame

In an environment, each frame $f$ brings a stimulus $\mathbf{o}_f$, and the walk is re-injected from $v_f = v(\mathbf{o}_f)$. The frame's flow $\pi_f$ is the solution of (1) with $v_f$. It is **implicit**: $\pi_f = \alpha (I - (1 - \alpha) T)^{-1} v_f$, so the walk never has to be sampled step by step. It is also **fast**. Starting from $x_0 = v_f$, $n$ hops of $x_{k+1} = (1 - \alpha) T x_k + \alpha v_f$ give

$$
\lVert x_n - \pi_f \rVert_1 \le (1 - \alpha)^n \lVert v_f - \pi_f \rVert_1 \le 2 (1 - \alpha)^n \qquad \text{(1a)}
$$

by Theorem 1. The number of hops for a given accuracy does not depend on network size. For example, $\alpha = 0.2$ and $n = 20$ give an error of at most $2.3\%$. Modular networks mix slowly, so they approach this bound: a modular test network contracted by $0.78$ per hop, against the bound of $0.8$. Budget for the bound, not for the faster rate of a dense network.

There are three ways to run the $n$ hops:

| Realisation | One hop is | Cost per frame |
| --- | --- | --- |
| Software reference | one sparse matrix–vector product | $n$ products |
| Memristive crossbar | one analogue matrix–vector read | $n$ reads |
| Spiking network | one synaptic delay | about $n$ ms, e.g. 20 ms per frame, which fits a 50 Hz environment |

The spiking version estimates $\pi_f$ through (2), with sampling noise and under the mean-field assumption. How closely it matches the reference is measured, not assumed.

**Timescales.** Three timescales must be ordered: walk hops, then frames, then plasticity. The walk converges within a frame ($n$ hops), and plasticity is slow relative to frames ($\lambda, \eta$ small), so the EM split (§8) holds per frame.

**Proposal: carry-over.** If frames are independent, the network is a function of the current stimulus only. That is enough when the observation is the full state, but not otherwise. To let latent communities carry state, mix the previous frame's latent flow into the teleportation vector:

$$
v_f = (1 - \rho) v(\mathbf{o}_f) + \rho \frac{P_{\mathcal{L}} \pi_{f-1}}{p_{\mathcal{L}}(f - 1)} \qquad \text{(1b)}
$$

Here $P_{\mathcal{L}}$ keeps only the entries of latent neurons and $p_{\mathcal{L}}$ is their total flow. Setting $\rho = 0$ recovers independent frames. With $\pi_{f-1}$ held fixed, every gradient below is unchanged. The neglected dependence of $\pi_f$ on $W$ is small. The cosine between the fixed-flow gradient and the full gradient was $0.977$ at $\rho = 0$ and $0.984$ at $\rho = 0.5$ (Appendix A).

### 2.4 Communities as states: the extracted Markov model

Coarse-graining the frame's flow to modules gives a $K$-state Markov chain, conditioned on the stimulus:

$$
T^K_{ba}(\mathbf{o}_f) = \frac{J_{ba}}{p_a} = \sum_{j \in a} \frac{\pi_{f,j}}{p_a} \sum_{i \in b} T_{ij} \qquad \text{(1c)}
$$

Each neuron's split of outgoing weight across modules, $\sum_{i \in b} T_{ij}$, is fixed by $W$. **The stimulus enters the module chain only by choosing which neurons in a module carry its flow.**

**Why a central controller.** The stimulus is concentrated in the controller $\mathcal{C}$, so the distribution of flow within $\mathcal{C}$ changes the most with $\mathbf{o}$, and so does the controller's column of $T^K$, which routes flow to latent and action modules. In an untrained test network, the controller column varied about thirty times more across stimuli (standard deviation $0.0196$) than when stimuli differed only in how much drive each module received, spread uniformly within it ($0.0006$). Plasticity then has to make that routing depend sharply on the stimulus.

**Policy.** The action is read out from the flow into the action modules:

$$
P(a \mid \mathbf{o}_f) = \frac{p_a(f)}{\sum_{a' \in \mathcal{A}} p_{a'}(f)}, \qquad a \in \mathcal{A} \qquad \text{(1d)}
$$

The spiking analogue is the action column with the most spikes in the frame window.

**World model (with carry-over).** The flow is linear in the teleportation vector, $\pi_f = R v_f$ with $R = \alpha (I - (1 - \alpha) T)^{-1}$. So the latent mass carried out of module $a$ lands in latent module $b$ in proportion to the latent-$b$ share of $R u_a$, where $u_a$ is $\pi_{f-1}$ restricted to $a$ and normalised. This gives a latent-to-latent transition matrix $P(b \mid a, \mathbf{o}_f)$ across frames, an abstract model of how the task's hidden state evolves. **Proposal**: its definition and estimator are not yet checked in simulation.

Read together, the communities are the states of an abstract decision process. The controller is the stimulus-dependent router, the latent modules carry state between frames, and the action modules are output states that emit actions.

## 3. The map equation

### 3.1 Intuition

Describe the walk with a two-level code. Inside a module, name the next neuron with a short, module-local codeword, or name "exit". On exit, name the next module from a global index. Frequent moves get short codewords. A step inside a familiar module is cheap; a hop into a rarely visited module costs an exit codeword plus an index codeword. $L(M)$ is the average number of bits per step.

### 3.2 Formula

$$
L(M) = q_{\curvearrowright} \log q_{\curvearrowright} - 2 \sum_m q_m \log q_m - \sum_j \pi_j \log \pi_j + \sum_m (p_m + q_m) \log (p_m + q_m) \qquad \text{(3)}
$$

The first two terms are the cost of the module index, the third is the cost of naming neurons, and the last is the cost of each module's codebook.

### 3.3 Leakage always costs bits

With $\pi$ held fixed, $W$ enters $L$ only through the exit rates, and

$$
M^{\ast}_m := \frac{\partial L}{\partial q_m} = \log \frac{q_{\curvearrowright} (p_m + q_m)}{q_m^2} \ge 0
$$

because $q_{\curvearrowright} \ge q_m$ and $p_m + q_m \ge q_m$. More exit flow from any module never shortens the description. $M^{\ast}_m$ is **the marginal number of bits per unit of exit flow from module $m$**, and it becomes the map equation's neuromodulator in §4.

### 3.4 The inverse problem, averaged over frames

Community detection, such as Infomap, solves the forward problem $\min_M L(M; W)$: given the graph, find the partition. Map-STDP solves the **inverse** problem. The controller and action modules are pinned, and plasticity minimises the description length over the weights, averaged over the frames the environment produces:

$$
\min_W D(W), \qquad D(W) = \mathbb{E}_f \left[ L(M; \pi_f) \right] \qquad \text{(3a)}
$$

The latent modules may be pinned too, or re-detected from time to time (§8). Re-detection is a forward step, so the procedure becomes an alternating minimisation over $M$ and $W$.

With each $\pi_f$ held fixed, (3a) is an average of per-frame description lengths. Each depends on $W$ only through that frame's link flows. Every result in §4 therefore holds frame by frame with $\pi_f$ in place of $\pi$, and holds for $D$ by averaging. The modulator $M^{\ast}_m$ can be computed from each frame's module statistics or from running averages across frames. The second is biased in principle, because an average of ratios is not a ratio of averages, but in a test it made no practical difference (Appendix A).

## 4. Plasticity as a three-factor rule

### 4.1 The rule

Let $\kappa_{ij}(t) = 1$ when pre $j$ spikes and post $i$ follows within the causal STDP window, and $0$ otherwise. The structural update is

$$
\Delta W_{ij}(t) = -\lambda \left( g_{ij} - \bar{g}_j \right) \kappa_{ij}(t), \qquad g_{ij} = \frac{\partial D}{\partial J_{ij}}, \qquad \bar{g}_j = \sum_k T_{kj} g_{kj} \qquad \text{(4)}
$$

It has three factors:

- **Pre and post** (factors 1 and 2), through the coincidence $\kappa_{ij}$.
- **The neuromodulator** (factor 3), $g_{ij} - \bar{g}_j$.

For hardware and biological realism, $g_{ij}$ depends only on the module pair $(m(i), m(j))$. It is a $K \times K$ table computed from module-level statistics and broadcast by each module's modulator. The baseline $\bar{g}_j$ is a per-neuron running mean of the modulator over $j$'s own transitions.

The task term $\eta (R - \bar{R}) e_{ij}$ is standard reward-modulated STDP. Subtracting the expected reward $\bar{R}$ is required for it to learn the task instead of drifting [Frémaux et al. 2010]. It is the only part that needs an eligibility trace, because rewards are delayed. Description costs are known at the moment of the transition.

### 4.2 What the rule descends

For any description length $D$ that depends on $W$ through the link flows $J$, with $\pi$ held fixed (the EM split, §8):

$$
\frac{\partial D}{\partial W_{ij}} = \frac{\pi_j}{d_j} \left( g_{ij} - \bar{g}_j \right) \qquad \text{(5)}
$$

Causal pairings occur at rate $J_{ij} = \pi_j W_{ij} / d_j$ (§2.2), so averaging (4) gives (Theorem 3)

$$
\mathbb{E} \left[ \Delta W_{ij} \right] = -\lambda W_{ij} \frac{\partial D}{\partial W_{ij}}
$$

This is gradient descent on $D$ with each coordinate scaled by its own weight, which is positive, so the direction is still a descent direction. Strong synapses move more and silent synapses stay silent. Two points about the baseline:

- **Without the baseline** $\bar{g}_j$, an extra term $-\lambda J_{ij} \bar{g}_j$ shrinks every synapse of neuron $j$ in proportion to its traffic. It acts as a cost-weighted weight decay, not descent.
- **A per-module baseline** $\bar{g}_m$ is a cheaper approximation. Its bias is $-\lambda J_{ij} (\bar{g}_j - \bar{g}_{m(j)})$.

### 4.3 Marginal cost, not pointwise cost

Every candidate has two related quantities:

- **Pointwise cost** $c(t)$: what one step "costs", such as its codeword length or the entropy it produces. Its average is the description length: $\mathbb{E}[c] = D$. This is what $D$ *measures*.
- **Marginal cost** $g_{ij} = \partial D / \partial J_{ij}$: how much $D$ grows if flow along that transition increases. This is what the modulator must *broadcast*.

The two coincide only for some candidates, such as the entropy rate. For the map equation, broadcasting the codeword length of each step also rewards landing on high-traffic neurons, whose short codewords are cheap. In simulation that biases learning (correlation 0.976 with the true descent direction, against 0.9999 for the marginal cost). §5 therefore lists both quantities for every candidate.

### 4.4 Worked case: the map equation

With $\pi$ fixed, $L$ depends on $J$ only through the exit rates $q_m = \sum_{j \in m} \sum_{i \notin m} J_{ij}$. So

$$
g_{ij} = M^{\ast}_{m(j)} \mathbb{I}[i \notin m(j)], \qquad \bar{g}_j = M^{\ast}_{m(j)} \bar{e}_j \qquad \text{(6)}
$$

**The modulator of module $m$ is silent while activity stays inside and fires $M^{\ast}_m$ when activity leaves.** Substituting into (5) gives the exact gradient:

$$
\frac{\partial L}{\partial W_{ij}} = M^{\ast}_{m(j)} \cdot \frac{\pi_j}{d_j} \cdot G(i, j), \qquad G(i, j) = \mathbb{I}[i \notin m(j)] - \bar{e}_j \qquad \text{(7)}
$$

What the pieces mean:

- $G$ decides **which way**: cross-module synapses are weakened and within-module synapses strengthened.
- $\bar{e}_j$ sets **how hard**: leaky neurons are corrected most.
- $M^{\ast}_m$ sets **how costly this module's leakage is**.

**Conservation.** $\sum_i W_{ij} G(i, j) = 0$. Applied multiplicatively, which is exactly what (4) does on average, the structural term conserves each neuron's total outgoing weight and only reallocates it from cross-module to within-module targets.

**Additive reference rule.** The earlier form $\Delta W_{ij} = \eta \mathrm{STDP}(i, j) - \eta' G(i, j)$ drops the positive factors from (7). It remains the exact-gradient baseline in simulation. It is not the hardware rule.

## 5. Candidate description lengths as modulators

Each candidate is written at the level of module pairs. When a transition goes from module $a = m(j)$ to module $b = m(i)$, the modulator reads $g_{ba}$ from a $K \times K$ table. The table is built from module-level statistics:

- $p_a$: visit rate of module $a$;
- $J_{ba} = \sum_{i \in b, j \in a} J_{ij}$: flow from module $a$ to module $b$;
- $T_{ba} = J_{ba} / p_a$: module-to-module transition probability, $T^K_{ba}$ of §2.4. In an environment, all three are per frame.

Where a quantity is naturally per synapse, such as entropy rate or entropy production, we use its **coarse-grained** (module-level) version, which is what a crossbar can compute. For entropy production, the data-processing inequality makes the coarse version a lower bound on the full one. For the entropy rate there is no general ordering: $h_K$ is a different, coarser quantity, not a bound.

| Candidate | Measures | Pointwise cost $c$ (step $a \to b$) | Modulator $g_{ba}$ | Thermodynamic reading | As a state abstraction |
| --- | --- | --- | --- | --- | --- |
| Map equation | two-level codelength of flow | codeword length | $M^{\ast}_a \mathbb{I}[b \ne a]$ | per-transition "surprise" of crossing modules | sticky, well-separated states; transitions between states are rare and cheap to name |
| Entropy rate $h_K$ | unavoidable information per step | $-\log T_{ba}$ | $-\log T_{ba}$ | lower bound on any code for the chain it describes; no modularity | deterministic state transitions; does not by itself make states sticky |
| Entropy production $\sigma_K$ | irreversibility of module flow | $\log (J_{ba} / J_{ab})$ | $\log \frac{J_{ba}}{J_{ab}} - \frac{J_{ab}}{J_{ba}}$ | **is** entropy production of the coarse Markov process | directed, cycle-driven state sequences; reversible chains score zero |
| Cost-weighted map equation | codelength plus energy per transmission | codeword $+ \lambda_E c_{ba}$ | $M^{\ast}_a \mathbb{I}[b \ne a] + \lambda_E c_{ba}$ | explicit energy; $\lambda_E$ = metabolic state | as the map equation, with expensive transitions avoided |
| Markov stability $r(\tau)$ | persistence of activity in modules | $-(\mathbb{I}[\text{same module after } \tau] - p_a)$ | $-\mathbb{I}[b = a]$ (at $\tau = 1$) | none (partition quality) | states that persist for $\tau$ steps |
| SBM description length | cost of the wiring diagram itself | per edge | $\log \frac{1 - \omega_{ba}}{\omega_{ba}}$, at rewiring | none (structural prior) | block-structured wiring underlying the states |
| Predictive dissipation | memory kept that does not predict the stimulus | $i(x_t; s_t) - i(x_t; s_{t+1})$ | estimated, no closed form | lower bound on dissipated work | states that keep only what predicts the next stimulus |
| Cross-module information flow | information exchanged between modules | pointwise transfer entropy | estimated, no closed form | enters each module's entropy balance | states that exchange little information |

The first four have closed-form gradients, checked against finite differences (Appendix A).

### 5.1 Map equation (baseline)

This is §4.4. The modulator fires only on module exits, with strength $M^{\ast}_a$.

- **Pointwise cost.** Within module $b$, the codeword is $-\log \frac{\pi_i}{p_b + q_b}$. On an exit from $a$ to $b$, add $-\log \frac{q_a}{p_a + q_a} - \log \frac{q_b}{q_{\curvearrowright}}$.
- **Expected cost.** It averages to $L(M)$ exactly at $\alpha = 0$ and to within $O(\alpha)$ otherwise, since teleportation shifts arrival rates slightly away from $\pi$. The Monte Carlo error was $0.1\%$ at $\alpha = 0.15$.
- **Reading.** Crossing into a rarely visited module is the expensive event, matching the hypothesis that "unfamiliar cortices cost more".

### 5.2 Entropy rate

$$
h = -\sum_j \pi_j \sum_i T_{ij} \log T_{ij}
$$

- **Gradient.** $\partial h / \partial W_{ij} = \frac{\pi_j}{d_j} ( -\log T_{ij} - H_j )$, where $H_j$ is the entropy of $j$'s outgoing transitions.
- **Modulator.** The surprisal of the transition, which is also its pointwise cost; this is the one candidate where the two coincide.
- **Reading.** Every code costs at least $h$ bits per step, so $L(M) - h$ is the overhead of forcing a modular code.
- **Caveat.** Minimising $h$ favours *deterministic* flow, not modular flow. Paired with the map equation it separates "predictable" from "modular". On its own it would not form cortices.
- **Hardware.** Use the module-level $h_K$ with $-\log T_{ba}$. The per-synapse version would need $-\log W_{ij}$ read from every device.

### 5.3 Entropy production

$$
\sigma = \sum_{i, j} J_{ij} \log \frac{J_{ij}}{J_{ji}}
$$

- **What it is.** The Schnakenberg entropy production of a Markov jump process [Schnakenberg 1976]. Its trajectory version, $\log (J_{ij} / J_{ji})$ per step, is the stochastic-thermodynamic entropy produced along the path [Seifert 2005].
- **Gradient.** $\partial \sigma / \partial W_{ij} = \frac{\pi_j}{d_j} ( A_{ij} - \bar{A}_j )$ with $A_{ij} = \log \frac{J_{ij}}{J_{ji}} + 1 - \frac{J_{ji}}{J_{ij}}$.
- **Modulator.** $A$ minus its constant 1, which differs from the pointwise cost by $-J_{ji}/J_{ij}$. Broadcasting only the pointwise entropy produced would be biased.
- **Reading.** This is the candidate most literally "thermodynamic". Lynn et al. measured coarse-grained entropy production of human brain-state transitions and found that it rises with task demands [Lynn et al. 2021].
- **Caveats.**
  - It needs reciprocal flow, $J_{ji} > 0$.
  - A perfectly modular but reversible network has $\sigma = 0$, so $\sigma$ alone exerts no modular pressure.
  - With teleportation, $\sigma$ here measures link-flow irreversibility; teleportation contributes its own share of entropy production.
- **Hardware.** The module-level $\sigma_K$ needs only $J_{ab}$ and $J_{ba}$ from the $K \times K$ flow table. No transposed synapse access is required.

### 5.4 Cost-weighted map equation

$$
D = L(M) + \lambda_E \sum_{i, j} J_{ij} c_{ij}
$$

- **Cost term.** $c_{ij}$ is an energy per transmission, for example wiring length or cross-cortex distance. At module level it is a $K \times K$ table $c_{ba}$.
- **Modulator.** The map modulator plus $\lambda_E c_{ba}$.
- **Metabolic state.** $\lambda_E$ is a natural *global* signal. Raising it under energy scarcity makes plasticity favour cheap transmissions, which parallels the starvation result [Plaçais & Preat 2013].
- **Bits and energy.** Landauer's principle bounds the energy per bit at $\ge k_B T \ln 2$, so $L$ and the energy term can be read in the same units. Real neurons operate far above this bound, however, so the exchange rate $\lambda_E$ must be fitted, not derived.

### 5.5 Markov stability

$$
r(\tau) = \sum_m \left[ P(X_0 \in m, X_\tau \in m) - p_m^2 \right]
$$

[Delvenne et al. 2010]

- **Modulator at lag 1.** With $\pi$ fixed, the gradient of $-r(1)$ is $\frac{\pi_j}{d_j} G(i, j)$. **The simplified rule $G$ is exactly the lag-1 Markov-stability gradient.** The modulator is $-\mathbb{I}[b = a]$: a constant reward for staying.
- **What the map equation adds.** It is Markov stability at lag 1 with each module's pressure reweighted by its marginal codelength $M^{\ast}_m$.
- **Longer lags.** $\tau > 1$ probes slower, coarser modules. The modulator needs the module identity $\tau$ steps back: a lagged module trace, estimated in the periphery.
- **Reading.** No thermodynamic reading. It is included as a partition-quality control.

### 5.6 SBM description length

The description length of the wiring diagram under a stochastic block model [Peixoto 2014] is a cost on **structure**, not on activity.

- **Marginal cost.** For a Bernoulli SBM with block densities $\omega_{ba}$ at their maximum-likelihood values, adding one synapse from module $a$ to module $b$ changes the description length by $\log \frac{1 - \omega_{ba}}{\omega_{ba}}$. Dense blocks are cheap; sparse blocks are expensive.
- **When it acts.** It applies at **rewiring events** (synapse creation or pruning), not per spike, so it modulates structural plasticity (§8) rather than (4).
- **Reading.** No thermodynamic reading. It is a structural prior that stabilises the block pattern.

### 5.7 Predictive dissipation

A system driven by a stochastic signal $s$ dissipates at least $k_B T (I_{mem} - I_{pred})$ [Still et al. 2012]:

- $I_{mem} = I(x_t; s_t)$ is what the state retains about the present stimulus;
- $I_{pred} = I(x_t; s_{t+1})$ is how much of that predicts the next stimulus.

Non-predictive memory is thermodynamically wasted.

- **Cost.** Per module, $D_m = I(x^m_t; s_t) - I(x^m_t; s_{t+1})$, with $x^m$ a coarse state of module $m$ (e.g. binned rate).
- **Modulator.** A running estimate of the pointwise version, $i(x^m_t; s_t) - i(x^m_t; s_{t+1})$, from small per-module decoders in the periphery.
- **Caveat.** There is no closed-form marginal cost, so this is an estimator and its bias is unknown.
- **Reading.** It ties structure directly to the stimulus-as-environment view (§2.1). A module should keep only what predicts.

### 5.8 Cross-module information flow

For two coupled subsystems, each subsystem's entropy balance gains an information-flow term: the rate at which it learns about the other [Horowitz & Esposito 2014].

- **Setup.** Treat modules as subsystems. The information flow from module $a$ to module $b$ can be estimated as a pointwise transfer entropy between coarse module states, charged to transitions from $a$ to $b$.
- **Reading.** This is the most direct formalisation of "communicating with an unfamiliar cortex costs more": cross-module information flow appears explicitly in each module's thermodynamic ledger.
- **Caveat.** Estimator only, like §5.7.

### 5.9 Proposal: predictability of the latent chain

The candidates above score the flow within a frame. A world model also needs the latent states to evolve predictably *across* frames. A natural cost is the entropy rate of the across-frame latent chain of §2.4:

$$
h_{\mathcal{L}} = -\sum_{a \in \mathcal{L}} p_a \sum_{b \in \mathcal{L}} P(b \mid a) \log P(b \mid a)
$$

- **Reading.** Low $h_{\mathcal{L}}$ means that the current latent state and stimulus determine the next latent state.
- **Status.** Proposal only. $P(b \mid a)$ depends on $W$ through the resolvent $R$, not only through one frame's link flows, so the Lemma does not apply directly, and neither a marginal cost nor a crossbar-native modulator has been derived. It needs carry-over ($\rho > 0$).

## 6. Locality and hardware mapping (memristive crossbars)

Target: $W$ stored as conductances in a crossbar, with rows = presynaptic neurons and columns = postsynaptic neurons, and neurons and modulators in peripheral circuits.

| Operation | Crossbar implementation |
| --- | --- |
| Recurrent drive, E-step (2) | standard analogue matrix–vector read |
| Per-frame walk (1a) | $n$ matrix–vector reads per frame, with $\alpha v_f$ added in the periphery after each read |
| $d_j$ and $e_j$ (hence $\bar{e}_j$, and module flows) | $K + 1$ reads: drive all columns, then each module's columns, and sum the current per row |
| Coincidence $\kappa_{ij}$ | overlapping pre/post programming pulses, as in conventional memristive STDP |
| Modulator $g_{ba}$ | peripheral logic over $K \times K$ module statistics; broadcast per module |
| Baseline $\bar{g}_j$ | per-row peripheral register: $\bar{g}_j = \sum_b T_{bj} g_{b, m(j)}$, from the masked reads |
| Applying (4) | scale row-pulse amplitude by $-\lambda (g_{ba} - \bar{g}_j)$, one phase per target-module column mask. The map equation needs only 2 phases per module: exit columns and home columns |
| Task eligibility $e_{ij}$ | the only per-synapse state; one option is volatile, diffusive memristors whose decay emulates a trace [Wang et al. 2017] |

No step requires a nonlinear function of an individual device's conductance, or access to the transposed element $W_{ji}$. This is why every modulator is defined at module level.

**Locality in the biological reading.** Everything a synapse needs is local to one of three places:

- the synapse (coincidence);
- its presynaptic neuron (row sums, $\bar{g}_j$);
- the neuromodulator of its presynaptic module.

The only structural information is the *module* of the postsynaptic partner. In hardware that is a column mask. Biologically it remains the non-local ingredient that §7 replaces.

## 7. Removing explicit modules: lateral inhibition

In cortex, modules emerge from local competition rather than from labels. Let $I_{ij} \ge 0$ be inhibition from $j$ onto $i$, and replace the membership bit with a soft competition score:

$$
\chi_{ij} = \frac{I_{ij}}{I_{ij} + \kappa_I} \in [0, 1), \qquad G_{\chi}(i, j) = \chi_{ij} - \bar{e}^{\chi}_j, \qquad \bar{e}^{\chi}_j = \frac{1}{d_j} \sum_{i'} \chi_{i'j} W_{i'j} \qquad \text{(8)}
$$

When $\chi$ is binary, (8) reduces to $G$. In the three-factor form, the map modulator becomes $g_{ij} = M^{\ast} \chi_{ij}$, which needs a module-level $M^{\ast}$ even without explicit modules. That is an open question.

Compared with lateral inhibition alone, as in Kohonen maps, which produces whatever partition winner-take-all dynamics yield, Map-STDP adds a measurable objective and adaptive gain.

**Open problems.** The proposed inhibitory update $\Delta I_{ij} = \eta_{inh} \pi_i \pi_j$ has three:

- it is unbounded;
- it makes co-active neurons into competitors, which opposes Hebbian STDP;
- direct inhibitory partners ignore Dale's law.

## 8. Training

Training runs frame by frame. For each environment frame $f$:

1. **Observe.** Read $\mathbf{o}_f$ and form the teleportation vector $v_f$, concentrated on the controller. With carry-over, mix in the previous frame's latent flow (1b).
2. **Walk.** Run the $n$ hops of (1a). In software or on a crossbar this gives $\pi_f$ directly. In the spiking network, the neurons run for the frame window and (2) estimates $\hat{\pi}_f$.
3. **Act.** Read the action from the action modules, (1d), or from the column with the most spikes.
4. **Module statistics.** Compute $p_a$, $J_{ba}$ and $q_a$ for the frame, update the modulator table $g_{ba}$ (per frame or as a running average, §3.4), and refresh the per-row baselines $\bar{g}_j$.
5. **Plasticity.** Apply the structural term (4) to the frame's causal pairings. Apply the task term when reward arrives, through the eligibility traces.
6. **Carry over.** Keep the latent flow $P_{\mathcal{L}} \pi_f$ for the next frame.

**EM split.** $\pi_f$ and the module statistics are treated as fixed within each update. Two conditions are needed: the walk converges within a frame (enough hops $n$), and plasticity is slow relative to frames ($\lambda, \eta$ small). In the spiking variant, the estimate (2) must also settle within the frame window ($\beta$ large enough).

Periodically:

- **Re-detect latent modules (optional).** The controller and action modules stay pinned, because they carry the input and the policy. Latent modules may be re-detected with Infomap or Leiden, which is the forward step of the alternating minimisation in §3.4. Static modules only reinforce the initial partition. Any partition works with (4).
- **Rewire.** Prune synapses below $\epsilon$ or outside each neuron's top $k$, and create synapses. The SBM modulator (§5.6) can gate these events.

Initialisation is either a stochastic block model (head start, but biased toward its partition) or Erdős–Rényi (neutral, slow). The **controller** module receives the stimulus, concentrating $v(\mathbf{o})$, and routes to the latent and action modules. Its broadcast role is analogous to Global Workspace Theory [Baars 2005].

## 9. Assumptions and limitations

1. **EM split.** $\pi$ and the module statistics are held fixed during updates.
2. **Mean field.** Flow equals normalised firing rate, and causal pairings occur at a rate proportional to $J_{ij}$ (§2.2). This is untested. Theorem 3 depends on it.
3. **Covariance rule.** (4) is exact only in expectation. Its variance grows with the number of synapses sharing a modulator, which is why modulators are per module and baselines per neuron.
4. **Coarse-graining.** Module-level $\sigma_K$ lower-bounds the full $\sigma$ and misses irreversibility inside modules. In a test network it captured only 4% of $\sigma$ (0.018 vs. 0.41). $h_K$ has no general ordering with $h$.
5. **Estimator candidates.** Predictive dissipation and information flow (§5.7–5.8) rely on peripheral estimators with unknown bias.
6. **Teleportation.** It is not counted as an exit, and it perturbs the map equation's codeword averages by $O(\alpha)$.
7. **Divergence.** The map equation's $M^{\ast}_m$ diverges as $q_m \to 0$, for a fully sealed module. It needs clipping.
8. **Lateral inhibition.** The inhibitory plasticity is unresolved (§7).
9. **Convergence within a frame.** Equation (1a) bounds the error of $n$ hops, but the spiking estimate has extra sampling noise, and in a short frame window it may not settle.
10. **Carry-over.** $\rho$ is a free parameter. With $\pi_{f-1}$ held fixed, the gradient ignores the dependence through earlier frames (cosine $0.98$ with the full gradient in a test).
11. **Number of latent modules.** $\lvert \mathcal{L} \rvert$ is a hyperparameter unless latent modules are re-detected.

---

## Appendix A: Proofs and checks

**Theorem 1 (flow exists, is unique, and is reachable).** If $T$ is column-stochastic and $\alpha \in (0, 1]$, then (1) has one solution $\pi$ in the probability simplex, and power iteration with $F(x) = (1 - \alpha) T x + \alpha v$ converges to it geometrically.

*Proof.* $F$ maps the simplex into itself. A column-stochastic $T$ has $\lVert T \rVert_1 = 1$, so

$$
\lVert F(x) - F(y) \rVert_1 = (1 - \alpha) \lVert T (x - y) \rVert_1 \le (1 - \alpha) \lVert x - y \rVert_1
$$

By Banach's fixed-point theorem $F$ has a unique fixed point, and the error shrinks by at least $(1 - \alpha)$ per step. $\square$

*Online remark.* Under the mean-field assumption, the bracket in (2) is an unbiased sample of $F(r) = \pi$. Equation (2) is then an exponential moving average with steady-state variance $O(\beta)$.

**Lemma (chain rule through flows).** If $D$ depends on $W$ only through $J_{kj} = \pi_j W_{kj} / d_j$, with $\pi$ fixed, then (5) holds.

*Proof.* Differentiating $J_{kj}$ gives $\partial J_{kj} / \partial W_{ij} = \pi_j (\delta_{ki} - T_{kj}) / d_j$. Therefore

$$
\frac{\partial D}{\partial W_{ij}} = \sum_k g_{kj} \frac{\pi_j (\delta_{ki} - T_{kj})}{d_j} = \frac{\pi_j}{d_j} \left( g_{ij} - \bar{g}_j \right)
$$

and only column $j$ of $J$ depends on $W_{ij}$. $\square$

**Theorem 2 (map-equation gradient).** With $\pi$ fixed, (7) holds.

*Proof.* The $p_m$ and $\pi_j \log \pi_j$ terms of (3) are constant. Differentiating (3) with respect to one $q_m$, including through $q_{\curvearrowright} = \sum_m q_m$:

$$
\frac{\partial L}{\partial q_m} = \left( \log q_{\curvearrowright} + 1 \right) - 2 \left( \log q_m + 1 \right) + \left( \log (p_m + q_m) + 1 \right) = M^{\ast}_m
$$

Because $q_m = \sum_{j \in m} \sum_{i \notin m} J_{ij}$, we get $g_{ij} = M^{\ast}_{m(j)} \mathbb{I}[i \notin m(j)]$ and $\bar{g}_j = M^{\ast}_{m(j)} \bar{e}_j$. Apply the Lemma. $\square$

**Theorem 3 (the three-factor rule descends $D$).** Suppose causal pairings $j \to i$ occur with rate proportional to $J_{ij}$, and the modulator reports $g_{ij} - \bar{g}_j$. Then the expected update (4) per unit rate is $-\lambda W_{ij} \partial D / \partial W_{ij}$.

*Proof.* $\mathbb{E}[\Delta W_{ij}] = -\lambda J_{ij} (g_{ij} - \bar{g}_j) = -\lambda W_{ij} \frac{\pi_j}{d_j} (g_{ij} - \bar{g}_j)$. The result follows from the Lemma. $\square$

**Other gradients** (all with $\pi$ fixed, via the Lemma):

- **Entropy rate.** $g_{ij} = -\log T_{ij} - 1$, and the constant cancels in $g - \bar{g}$, giving $\frac{\pi_j}{d_j}(-\log T_{ij} - H_j)$.
- **Entropy production.** $\sigma = \sum_{i,j} J_{ij} \log (J_{ij}/J_{ji})$. $J_{ij}$ appears in the $(i,j)$ term and in the $(j,i)$ term, giving $g_{ij} = \log \frac{J_{ij}}{J_{ji}} + 1 - \frac{J_{ji}}{J_{ij}}$.
- **Cost-weighted map equation.** $g_{ij} = M^{\ast}_{m(j)} \mathbb{I}[i \notin m(j)] + \lambda_E c_{ij}$.
- **Markov stability at lag 1.** $-r(1) = -\sum_m \sum_{i, j \in m} J_{ij} + \text{const}$, so $g_{ij} = -\mathbb{I}[m(i) = m(j)]$ and $g_{ij} - \bar{g}_j = \mathbb{I}[i \notin m(j)] - \bar{e}_j = G(i, j)$.

**Expected pointwise costs.** With steps sampled with probability $J_{ij}$:

- $\mathbb{E}[-\log T_{ij}] = h$;
- $\mathbb{E}[\log (J_{ij}/J_{ji})] = \sigma$;
- $\mathbb{E}[-(\mathbb{I}[\text{same}] - p_{m(j)})] = -r(1)$, because $\mathbb{E}[p_{m(j)}] = \sum_m p_m^2$;
- the map equation's codeword average equals $L$ at $\alpha = 0$.

**Numerical checks** (random 12-node, 3-module network, $\alpha = 0.15$):

- all five closed-form gradients match finite differences to $\le 4 \times 10^{-7}$, and the module-level $h_K$ and $\sigma_K$ gradients to $\le 3 \times 10^{-9}$;
- the lag-1 Markov-stability gradient equals $\frac{\pi_j}{d_j} G$ to machine precision;
- Monte Carlo means ($4 \times 10^5$ steps) match $h$, $\sigma$ and $-r(1)$ within sampling error, and $L$ within $0.1\%$;
- the three-factor rule with marginal-cost modulators matches $-W \odot \nabla D$ with correlation $0.9999$ for the map equation, entropy rate and entropy production, in both per-synapse and module-level forms, against $0.976$ when the pointwise codeword length is broadcast;
- power-iteration error ratios stay below $1 - \alpha$;
- $\sum_i W_{ij} G(i, j) = 0$ to machine precision.

**Numerical checks for the per-frame formulation** (random 15-node network: a controller, 2 latent and 2 action modules of 3 neurons each; $\alpha = 0.2$; 8 stimuli concentrated on the controller):

- the per-frame map-equation gradient (7) under $\pi(\mathbf{o})$ matches finite differences to $\le 10^{-9}$ for every stimulus, and so does the frame-averaged gradient of $D$ in (3a);
- truncated iteration (1a): errors stay below the bound, with the largest per-hop contraction $0.28$ on a dense network and $0.78$ on a sparse modular one (bound $0.8$);
- three-factor rule averaged over frames ($4000$ frames of $200$ sampled pairings, modulator from each frame's statistics) matches $-W \odot \nabla D$ with correlation $0.99996$. With a modulator pooled across frames, the correlation is $0.99993$;
- with carry-over (1b), the fixed-flow gradient still matches finite differences to $\le 10^{-9}$. Its cosine with the full gradient, recomputing $\pi_f$ through $W$, is $0.977$ at $\rho = 0$ and $0.984$ at $\rho = 0.5$;
- the identity (1c) holds to machine precision. Across stimuli, the controller column of $T^K$ has total standard deviation $0.0196$, against $\le 0.0035$ for the other columns, and $0.0006$ when stimuli differ only in how much drive each module receives, spread uniformly within modules.

## Appendix B: Corrections to the earlier derivation

1. **Sign error.** The earlier version expanded $L(M)$ with $+$ on its second and third terms; both are negative. The gradient's log factor becomes $M^{\ast}_m = \log \frac{q_{\curvearrowright}(p_m+q_m)}{q_m^2} \ge 0$, not $\log \frac{q_{\curvearrowright}}{p_m+q_m} \le 0$.
2. **Direction convention.** $W_{ij}$ is the synapse from pre $j$ to post $i$ throughout, and the walk follows spikes. The earlier version row-normalised but wrote $\tilde{p} = \tilde{T} \tilde{p}$. Gating is now indexed by the presynaptic neuron.
3. **Teleportation.** The stimulus enters as $\alpha v(\mathbf{o})$, making the flow unique and the rule stimulus-dependent.
4. **Flow vs. firing.** The earlier version equated $p(z_j = 1 \mid \mathbf{o})$ with the stationary distribution. That is now an explicit assumption (§2.2).
5. **Convergence.** The earlier version showed only a fixed point. This version proves contraction.
6. **Lateral inhibition.** The earlier $\varphi_{ij}$ did not reduce to the indicator; it is replaced by $\chi_{ij}$, and the problems with the inhibitory update are flagged.
7. **Three-factor restructure.** The structural term is a neuromodulated, cost-gated STDP rule, (4). Its modulator is the marginal description cost, and eight candidate description lengths plug in. The factor $M^{\ast}_m$, previously absorbed into the learning rate, is the map equation's per-module neuromodulator. $G$ is the lag-1 Markov-stability gradient. Every modulator is defined at module level for memristive crossbars.
8. **Reformulation: communities as states (2026-10-04, new).** The primary goal is now a state-space abstraction, and thermodynamics is a secondary hypothesis. Modules are typed as controller, action and latent. Flow is computed once per environment frame by a truncated walk, (1a), with optional carry-over of latent flow, (1b). The module chain $T^K(\mathbf{o})$, (1c), is the extracted Markov model, and the policy, (1d), is read from the action modules. The rule descends the frame-averaged description length, (3a): the inverse of community detection. The rule (4) and Theorems 1–3 are unchanged; they apply per frame.

## References

- Baars, B. J. (2005). Global workspace theory of consciousness: toward a cognitive neuroscience of human experience. *Progress in Brain Research*, 150, 45–53.
- Buesing, L., Bill, J., Nessler, B., & Maass, W. (2011). Neural dynamics as sampling: a model for stochastic computation in recurrent networks of spiking neurons. *PLoS Computational Biology*, 7(11), e1002211.
- Dayan, P., & Abbott, L. F. (2001). Information theory. In *Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems* (pp. 123–150). MIT Press.
- Delvenne, J.-C., Yaliraki, S. N., & Barahona, M. (2010). Stability of graph communities across time scales. *PNAS*, 107(29), 12755–12760.
- Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional requirements for reward-modulated spike-timing-dependent plasticity. *Journal of Neuroscience*, 30(40), 13326–13337.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural Circuits*, 12, 53.
- Horowitz, J. M., & Esposito, M. (2014). Thermodynamics with continuous information flow. *Physical Review X*, 4, 031015.
- Li, H. L., & van Rossum, M. C. W. (2020). Energy efficient synaptic plasticity. *eLife*, 9, e50804.
- Lynn, C. W., Cornblath, E. J., Papadopoulos, L., Bertolero, M. A., & Bassett, D. S. (2021). Broken detailed balance and entropy production in the human brain. *PNAS*, 118(47), e2109889118.
- Peixoto, T. P. (2014). Hierarchical block structures and high-resolution model selection in large networks. *Physical Review X*, 4, 011047.
- Plaçais, P.-Y., & Preat, T. (2013). To favor survival under food shortage, the brain disables costly memory. *Science*, 339(6118), 440–442.
- Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex networks reveal community structure. *PNAS*, 105(4), 1118–1123.
- Schnakenberg, J. (1976). Network theory of microscopic and macroscopic behavior of master equation systems. *Reviews of Modern Physics*, 48(4), 571–585.
- Seifert, U. (2005). Entropy production along a stochastic trajectory and an integral fluctuation theorem. *Physical Review Letters*, 95, 040602.
- Shouval, H. Z., Wang, S. S.-H., & Wittenberg, G. M. (2010). Spike timing dependent plasticity: a consequence of more fundamental learning rules. *Frontiers in Computational Neuroscience*, 4.
- Still, S., Sivak, D. A., Bell, A. J., & Crooks, G. E. (2012). Thermodynamics of prediction. *Physical Review Letters*, 109, 120604.
- Wang, Z., Joshi, S., Savel'ev, S. E., et al. (2017). Memristors with diffusive dynamics as synaptic emulators for neuromorphic computing. *Nature Materials*, 16, 101–108.
