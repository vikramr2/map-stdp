# Map-STDP: communities as states

Author: Vikram Ramavarapu

**Goal.** Give a spiking network a **state-space abstraction** whose states are its own communities. A random walk on a graph with communities has a minimum description length: the map equation, or one of the alternatives in §5. Community detection solves the *forward* problem, finding the partition that minimises the description length of a walk on a given graph. Map-STDP solves the **inverse** problem. The partition is given, with one controller community, one community per action and some latent communities, and plasticity shapes the synapses until that partition is the minimum-description-length one. The coarse-grained walk between communities is then a Markov model over the network's states, conditioned on the stimulus. It can be read out, and in an RL environment it chooses the action.

**Side tie: thermodynamics.** Neurons pay a real metabolic and thermodynamic cost to communicate. We hypothesise that this cost is higher when activity crosses into cortices a neuron rarely talks to. If description length tracks that cost, the information dynamics that the rule minimises double as a model of thermodynamics in the brain. This is a secondary hypothesis, tested through an energy proxy (§1).

STDP is local, so on its own it has no reason to organise a network into cortices: dense modules with sparse links between them. Map-STDP makes structure a *neuromodulated* term. Each module has a modulatory signal that reports how much a candidate **description length** $D$ grows when activity crosses a given transition, and that signal gates STDP. A per-module modulator is an abstraction. Its closest biological carriers are regional dopamine or astrocytic domains (§6). The map equation is the baseline $D$. Seven alternatives plug into the same rule (§5).

**Each environment frame is one walk.** The stimulus of frame $f$ re-injects the walker, and the walk is run implicitly to its per-frame flow $\pi_f$ in a fixed number of hops (§2.3). Description length, modulators and the extracted Markov model are all per frame.

This document is a corrected and consolidated rewrite of an earlier derivation, reformulated on 2026-10-04 around communities as states. Appendix B lists what changed.

## At a glance

For every causal spike pairing $j \to i$ (pre $j$ fires, then post $i$ within the STDP window; indicator $\kappa_{ij}(t)$), the synapse changes by

$$
\Delta W_{ij}(t) = \underbrace{\eta \delta(t) e_{ij}(t)}_{\text{task: TD-modulated STDP}} - \underbrace{\lambda \left( g_{ij} - \bar{g}_j \right) \kappa_{ij}(t)}_{\text{structure: cost-modulated STDP}}
$$

where:

- $\delta$ is a global TD error from a critic readout (§4.6).
- $e_{ij}$ is an **action-gated** eligibility trace. Each pairing adds $\kappa_{ij} (h_{m(i)} - \bar{h}_j)$, where $h$ is a $K$-vector that marks the chosen action's module (§4.6). The trace is needed because reward arrives late.
- $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal description cost** of one more unit of flow from $j$ to $i$. It depends only on the modules of $i$ and $j$, so a per-module modulatory signal can broadcast it.
- $\bar{g}_j$ is its average over $j$'s outgoing transitions, a per-neuron baseline.

Both terms have the same form: a module-level broadcast, minus a per-neuron baseline, times a spike coincidence. A third term of the same form, $-\epsilon_h (d_j - 1) \kappa_{ij}$, holds each neuron's total output weight $d_j$ near 1 (§2.2, (4d)).

How to read the structural term:

- **The modulator reports cost.** A transition that makes activity more expensive to describe than $j$'s usual transitions ($g_{ij} > \bar{g}_j$) depresses the synapse that produced it. A cheaper-than-usual transition potentiates it.
- **On average it is gradient descent** (Theorem 3): $\mathbb{E}[\Delta W_{ij}] = -\lambda W_{ij} \partial D / \partial W_{ij}$. This is multiplicative, weight-scaled descent of the description length.
- **The modulator must be the *marginal* cost, not the cost itself** (§4.3). Broadcasting the codeword length of each step is biased.
- **For the map equation**, the modulator fires only when activity *leaves* a module, with strength $M^{\ast}_m$ equal to the marginal bits per exit. On average this reproduces the familiar gating:

| Synapse $j \to i$ | Average structural update | Effect |
| --- | --- | --- |
| $i$ in a **different** module from $j$ | $\propto -(1 - \bar{e}_j)$ | weakened |
| $i$ in the **same** module as $j$ | $\propto +\bar{e}_j$ | strengthened |

- **Known conflict** (§4.5). Controller-to-action routing is cross-module by definition, so the map term alone starves the action modules. A dual routing term is proposed as the fix.
- **It suits memristive crossbars** (§6). The update is a pulse-coincidence STDP event scaled by a per-module signal. There are no per-synapse nonlinear reads and no transposed access. In hardware, description costs are available at the moment of the transition, so the structural term needs no trace. A biological modulator would lag, and would need one.
- **The communities are the states** (§2.4). In each frame, the module-to-module flow defines a Markov chain $T^K(\mathbf o)$ over communities. Its flow into the action communities is the policy, and its flow between latent communities across frames is a learned world model.

## 1. Motivation

- **Communities as states.** An agent needs a compact state space, but a spiking network's microstate (which neurons fire) is far too large to serve as one. If the network is modular, the coarse-grained walk between modules is a small Markov chain that can be read out. When modules map onto actions and latent situations, the network carries its own abstract model of the task: which latent state it is in, and which action that state leads to.
- **A central controller.** The stimulus enters one central controller community, which routes activity to latent and action communities. The controller is what makes the extracted chain depend on the stimulus (§2.4). This is a functional abstraction, loosely inspired by the cerebral cortex integrating sensory input. A single stimulus-receiving router has no exact anatomical counterpart; it is closer to thalamus or sensory cortex, and cortex itself is highly modular. The readout of action communities resembles channel selection in the basal ganglia [Redgrave et al. 1999]. Latent communities resemble task-state representations in orbitofrontal cortex [Wilson et al. 2014] and latent-structure learning [Gershman & Niv 2010]. The broadcast role echoes Global Workspace Theory [Baars 2005].
- **Cortices are useful codes.** When modules map onto concepts, population activity becomes a readable spatial code. Each discrete action gets a module.
- **Activity as a walk on the neuron graph.** Stochastic spiking networks sample from a distribution over activity patterns [Buesing et al. 2011]. We model a different, simpler object: a walker on the neuron graph whose stationary flow approximates normalised firing rates (§2.2). The map equation [Rosvall & Bergstrom 2008] scores modularity by the description length of such a walk.
- **Plasticity is expensive, and neuromodulators gate it.** Memory formation has a measurable metabolic cost. Starving *Drosophila* switch off protein-synthesis-dependent long-term memory, and forcing it shortens their survival [Plaçais & Preat 2013]. Distributing learning between cheap transient changes and costly consolidation saves energy by up to an order of magnitude [Li & van Rossum 2020]. Three-factor rules, where a local eligibility flag becomes a weight change only when a neuromodulator arrives, have direct experimental support [Gerstner et al. 2018].
- **Side hypothesis (thermodynamics).** The neuromodulator *carries a description-length signal*, so plasticity is spent where it reduces the information-theoretic cost of the network's activity. If that cost tracks metabolic cost, the same rule also makes the network thermodynamically efficient. The state abstraction does not depend on this hypothesis; it is tested separately against an energy proxy.
- **Free energy.** Under the free-energy view [Friston 2010], one can read the sampling distribution as the network's recognition density. The structural term organises that same distribution into modules instead of adding an unrelated goal.

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

The membrane potential is $u_i = b_i + \sum_j W_{ij} z_j(t)$, as in sampling SNNs [Buesing et al. 2011], so activity flows from $j$ to $i$.

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

**Assumption (mean field).** The normalised firing rates satisfy the same balance as (1). Two things follow:

- $r = \pi$, and (2) tracks $\pi$ with $O(\beta)$ noise.
- **Causal spike pairings $j \to i$ occur at a rate proportional to $J_{ij}$.**

The second consequence is what lets spike coincidences stand in for walker steps in §4. The assumption needs three things that a spiking network does not provide by default:

1. **Constant presynaptic output.** Spiking drive depends on $W_{ij}$, not on $W_{ij}/d_j$, so (1) describes the rates only if each $d_j$ is held fixed. That requires presynaptic normalisation. Both plasticity terms conserve $d_j$ in expectation (§4.4, and $\sum_i T_{ij}(h_{m(i)} - \bar{h}_j) = 0$ for the task term), but sampled updates clipped at $w_{min}$ ratchet it upward. In M1, spiking learning ran away to rates of 0.65–0.71 per step without normalisation.

   **Adopted (M1):** a per-neuron output gain $1/d_j$, so the simulator implements $T$ directly. This is one column-sum read, so it is crossbar-native. The conductance range is held by a $d_j$ homeostasis term. Its expectation is a uniform rescaling of column $j$ ($\sum_i W_{ij} \partial D / \partial W_{ij} = 0$), so it leaves $T$ and $D$ unchanged. In M1, added to the baseline $\bar{g}_j$ as $\epsilon (d_j - 1)$, it brought the largest $d_j$ from 2.04 to 1.12 at $\epsilon = 1$, and $D$ moved only from 4.039 to 4.051.

   **Changed (iteration 002): homeostasis is its own term, on all plasticity.** Inside $\lambda$ it was too weak ($\lambda \epsilon = 0.05$), and it vanished at $\lambda = 0$. In the M2 closed loop the task term ratcheted $d_{max}$ to 13 (flow) and 19 (spiking) at $\lambda = 0$, and to 2.7 in spiking at $\lambda = 0.05$. The term is now

   $$
   \Delta W_{ij} = - \epsilon_h \left( d_j - 1 \right) \kappa_{ij} \qquad \text{(4d)}
   $$

   applied every frame whatever $\lambda$ is. It is a per-row register times the pairing pulse, so it is crossbar-native. With $\kappa = J$ in expectation, its average is $-\epsilon_h (d_j - 1) \pi_j W_{ij} / d_j$, again a column rescaling. In a spiking Monte Carlo (2000 frames, $d_j$ spread over $[0.6, 2]$) the average update had cosine 0.992 with this expectation (preliminary). Biologically it reads as fast heterosynaptic plasticity, a compensatory process that acts on all Hebbian change [Zenke & Gerstner 2017; Chistiakova et al. 2014]. Conservation of total synaptic weight is observed [Royer & Paré 2003], but that evidence is for the postsynaptic (input) side. Presynaptic output normalisation remains plausible but untested. Slow synaptic scaling [Turrigiano et al. 1998] is the wrong analogue: it is postsynaptic and acts over hours.
2. **Global divisive normalisation.** $\hat{z} = z / \sum_k z_k$ is a network-wide quantity. Biologically, it needs global inhibitory gain control, which the inhibitory pools of §7 can supply.
3. **No chance coincidences.** In a spiking network, the causal-pairing rate is about $J_{ij}$ plus a chance term proportional to $\tau_{STDP} r_i r_j$, which does not depend on $W_{ij}$ and biases Theorem 3. In a preliminary linear-Poisson (Hawkes) test, 85% of causal pairings in a 20 ms window were chance pairings. **Adopted fix (M1): the covariance count** $\kappa_{ij} = C_{ij} - c_i c_j (k-1)/k^2$. Here $C_{ij}$ is the number of lag-1 causal pairings in the frame, and $c_i$ is neuron $i$'s spike count over the $k$ steps. This is a frame-batched covariance rule [Sejnowski 1977]; the product-of-rates term is also what appears in spike-level STDP analyses [Kempter et al. 1999]. The subtracted term is a rank-1 product of per-neuron counters, so it costs one outer-product crossbar phase.

   M1 results (branching mode, 80 neurons):

   | Count | Correlation with $J_{ij}$ | Cosine of frame-averaged update with $-W \odot \nabla D$ |
   | --- | --- | --- |
   | Covariance | 0.91 | 0.992 |
   | Causal only | 0.79 | 0.979 |
   | Causal minus acausal (balanced) | 0.48 | 0.847 |

   The balanced count, proposed earlier, also subtracts genuine reverse transmission $J_{ji}$ on reciprocal synapses. Residual biases of the covariance count:

   - **Common inputs.** Neurons that share inputs covary within the same step, a two-hop term.
   - **Weak synapses.** There the count is zero-mean noise, which the $w_{min}$ floor rectifies, so keep the per-count $\lambda$ at about $5 \times 10^{-4}$ or less. The same holds for the task term. The rectification does not depend on the floor's value: lowering $w_{min}$ from $10^{-3}$ to $10^{-6}$ left the spiking $d_j$ drift unchanged (iteration 002). (4d) removes the drift in $d_j$ but not the noise itself. Weight-dependent soft bounds, with depression proportional to $W_{ij} - w_{min}$, would remove the bias at its source [van Rossum et al. 2000]; that is a linear function of conductance (**Proposal**).
   - **Non-causality.** The frame-batched subtraction is non-causal. The biological variant subtracts the product of slow pre and post traces online.
   - **Inhibition.** With inhibitory pools (§7), disynaptic inhibition gives competing modules negative covariance.

In M1, causal counts correlated 0.79 with $J_{ij}$, and acausal counts were 73% as large as causal ones: most of the excess is chance.

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
| Spiking network | not a discrete hop: the recurrent dynamics relax with time constant about $\tau/\alpha$ ($\tau$ the membrane or synaptic time constant) | about 100–250 ms per frame (4–10 Hz). A 50 Hz environment needs frame-skipping |

The spiking version estimates $\pi_f$ through (2), with sampling noise and under the mean-field assumption. Its transient decays roughly as $e^{-\alpha t / \tau}$, so reaching a 2% error takes about $4 \tau / \alpha$: 100–200 ms for $\tau = 5$–$10$ ms and $\alpha = 0.2$. At cortical rates, a 20 ms window also gives each neuron only 0.1–0.2 spikes, and it is about as long as the STDP window, so pairings would straddle frames. In a preliminary Hawkes test (200 neurons, 5 modules), the module-level L1 error of $\hat{\pi}$ in a 20 ms window was 0.24–0.50, against the software bound of 0.023. To reduce it:

- warm-start each frame from the previous one instead of resetting;
- read the policy at module level;
- use larger populations or a larger $\alpha$;
- state the window in units of $\tau / \alpha$.

How closely the spiking estimate matches the reference is measured, not assumed. The RL experiments run on the software reference first.

**M1 measurement.** The $4 \tau / \alpha$ estimate above was optimistic. In branching mode, the module-level error is sampling-limited at about $1.3 / \sqrt{S_{mod}}$, where $S_{mod}$ is the number of spikes per module per frame. Reaching 0.1 needs about 170 spikes per module, about 30 $\tau / \alpha$ for 16-neuron modules. So frame windows are stated in $S_{mod}$, and they shorten as modules grow. Finite leak has an error floor of 0.09–0.15, from spontaneous firing of integrated noise.

**Discrete-time spiking (the planned simulator, SuperNeuroMAT).** With delay 1, one time step is one synaptic transmission, $u \leftarrow u + W z$, which is one matrix–vector product and so one hop. Whether a step behaves like a *walk* hop depends on the leak:

- **Finite leak.** Neurons integrate over several steps, and the network relaxes over roughly $1/\alpha$ steps or more, as in continuous time.
- **Infinite leak with stochastic input** (**Proposal**). The state resets every step, so the network is a branching process in which one step is one hop. Whether its normalised spike counts track $\pi_f$ within $n$ steps is tested at M1.

**Timescales.** Three timescales must be ordered: walk hops, then frames, then plasticity. The walk converges within a frame ($n$ hops), and plasticity is slow relative to frames ($\lambda, \eta$ small), so the EM split (§8) holds per frame.

**Proposal: carry-over.** If frames are independent, the network is a function of the current stimulus only. That is enough when the observation is the full state, but not otherwise. To let latent communities carry state, mix the previous frame's latent flow into the teleportation vector:

$$
v_f = (1 - \rho) v(\mathbf{o}_f) + \rho \frac{P_{\mathcal{L}} \pi_{f-1}}{p_{\mathcal{L}}(f - 1)} \qquad \text{(1b)}
$$

Here $P_{\mathcal{L}}$ keeps only the entries of latent neurons and $p_{\mathcal{L}}$ is their total flow. Setting $\rho = 0$ recovers independent frames. With $\pi_{f-1}$ held fixed, every gradient below is unchanged. The neglected dependence of $\pi_f$ on $W$ is small. The cosine between the fixed-flow gradient and the full gradient was $0.977$ at $\rho = 0$ and $0.984$ at $\rho = 0.5$ (Appendix A).

**Limit: linear carry-over forgets.** Equation (1b) is a contraction, and linear PageRank has no attractors. The difference between two histories therefore shrinks geometrically. In a test network it shrank by a factor of 0.29–0.39 per frame ($\rho = 0.5$–$0.9$), down to $10^{-6}$–$10^{-8}$ after 10 frames. So (1b) alone cannot hold a cue across a delay. **Proposal:**

- **Make the carried vector nonlinear at module level.** Apply a soft winner-take-all over latent modules, e.g. weights $p_\ell^{\gamma} / \sum_{\ell'} p_{\ell'}^{\gamma}$ with $\gamma > 1$, in the periphery. That keeps it crossbar-native. With $\pi_{f-1}$ held fixed, the gradients are unchanged.
- **Give credit across the delay.** Set the eligibility time constant at or above the cue-to-reward delay. As an off-hardware baseline, use e-prop-style traces for recurrent latent synapses [Bellec et al. 2020].
- **Don't seal the latent modules.** The map term also seals latent modules, which blocks the cue from being written in. Controller-to-latent routing needs the dual term of §4.5.

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

Its spiking analogue is a **race readout**. The action is the module whose population spikes first in the frame. For independent Poisson populations with rates proportional to $p_a$, the race picks $a$ with probability exactly $p_a / \sum_{a'} p_{a'}$, so in the stationary regime, after the walk has relaxed, the race and (1d) are the same policy. The column with the most spikes is a *different* policy: nearly greedy at high spike counts and nearly random at low ones.

**Burn-in (adopted, iteration 002).** A cold-started frame is not stationary. The first action spike came at step 2–3, when the action modules' share still over-weights direct controller→action routing, and the per-step deviation from (1d) was 0.10–0.15 over steps 1–4. The deviation decays as $(1 - \alpha)^t$, and $0.8^{10} \approx 0.11$, so the race starts after a burn-in of $b = 10$ steps. That brought the KL from (1d) to the race down to its sampling floor from the start of training (0.011 against a floor of 0.010, compared with 0.049 without burn-in; preliminary, `snn-expert`). Spike-count shares over a late window matched (1d) to KL ≈ 0.0006, so the stationary spiking flow is right; only the race start was wrong. Biologically, the burn-in reads as tonic basal-ganglia inhibition, which withholds the response early in deliberation [Frank 2006]. A one-spike race is an accumulator with a bound of one spike [Gold & Shadlen 2007]. A many-spike race is more biological, but it is not (1d), so the eligibility score would have to match it (see (1e)). A warm start (§8) would also shorten the transient.

**Proposal: sharpened policy.** (1d) has no temperature (§9, item 14). A family with one is

$$
P_\beta(a \mid \mathbf{o}_f) = \frac{p_a(f)^\beta}{\sum_{a' \in \mathcal{A}} p_{a'}(f)^\beta} \qquad \text{(1e)}
$$

Its matched score in (4b) is $h_b = \beta \left( \mathbb{I}[b = a^{\ast}] - \tilde{P}_\beta(b) \right) / J^{in}_b$ on action modules, still a $K$-vector, and $\beta = 1$ recovers (4b). In spiking, a many-spike race or a late-window count sharpens the policy, but neither is exactly (1e). The readouts that sharpened the policy gave much higher spiking return (count argmax 186 against 87 for the burn-in race over episodes 201–300; preliminary), at a KL of 0.17 from (1d).

**Limits of this policy family.** $P(a \mid \mathbf{o})$ is a ratio of linear functions of $v$, so it is a mixture of each controller neuron's routing, and it has no temperature to sharpen it. Recurrence inside the controller blurs the stimulus. In a preliminary flow-level test, even ideal routing reached P(correct) = 0.92 with no controller recurrence, falling to 0.59 when 95% of the controller's outgoing weight stayed within the controller ($\alpha = 0.2$). So keep controller recurrence low, for example by giving the controller a larger $\alpha$, or by exempting the controller's own column from the map term (§4.5).

**Encoding matters as much as learning.** With one receptive field per observation dimension, the policy is an average of per-dimension votes. In flow-level CartPole tests, even ideal votes reached only 84 steps. One vote per conjunctive $(\theta, \dot{\theta})$ grid cell reached 491. So the controller uses conjunctive receptive fields (preliminary; see the iteration 001 record).

**World model (with carry-over).** The flow is linear in the teleportation vector, $\pi_f = R v_f$ with $R = \alpha (I - (1 - \alpha) T)^{-1}$. So the latent mass carried out of module $a$ lands in latent module $b$ in proportion to the latent-$b$ share of $R u_a$, where $u_a$ is $\pi_{f-1}$ restricted to $a$ and normalised. This gives a latent-to-latent transition matrix $P(b \mid a, \mathbf{o}_f)$ across frames, an abstract model of how the task's hidden state evolves. **Proposal**: its definition and estimator are not yet checked in simulation.

Read together, the communities are the states of an abstract decision process. The controller is the stimulus-dependent router, the latent modules carry state between frames, and the action modules are output states that emit actions.

**Biological reading.** The closest biology is clustered E/I networks whose clusters switch metastably [Litwin-Kumar & Doiron 2012], and HMM-decoded state sequences in cortex [Mazzucato et al. 2015]. There, module-to-module transitions take hundreds of milliseconds. That is a *frame* timescale, not a hop timescale, so the biologically meaningful state chain is the across-frame latent chain, not $T^K$ within a frame. Also, one anatomical module per state scales poorly: biological state codes are largely distributed and mixed.

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
- **The modulator** (factor 3), $g_{ij} - \bar{g}_j$.

For hardware, $g_{ij}$ depends only on the module pair $(m(i), m(j))$. It is a $K \times K$ table computed from module-level statistics and broadcast per module. The baseline $\bar{g}_j$ is a per-neuron running mean of the modulator over $j$'s own transitions.

A spiking implementation counts $\kappa_{ij}$ with the covariance count of §2.2, which removes chance coincidences in expectation. In hardware, description costs are available at the moment of the transition, so the structural term needs no trace. A biological modulator lags by 0.3–2 s [Yagishita et al. 2014], so a biological structural term would need an eligibility trace too.

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
- **The baseline as presynaptic renormalisation.** To first order in $\lambda$, the baselined rule equals the unbaselined rule followed by multiplicatively rescaling each neuron's outgoing weights so that $d_j$ is unchanged. The relative difference was $2.5 \lambda$ (Appendix A). This gives $\bar{g}_j$ a biological reading as a presynaptic resource constraint. That constraint is plausible but untested, since most documented scaling mechanisms are postsynaptic. It is also the normalisation that the mean-field assumption needs (§2.2). Row rescaling is not crossbar-native, so on hardware the baseline stays explicit.

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

### 4.5 Conflict: structure starves routing (Proposal: dual routing term)

In this architecture, the policy is carried by **cross-module** flow: controller to action, and controller to latent. The map term pushes every module to keep its flow inside, so it works against that routing.

- **What happens.** Applied multiplicatively, $G$ scales all of a neuron's cross-module synapses by the same factor. It never changes *which* action a controller neuron prefers. It only drives that neuron's exit fraction $\bar{e}_j$ toward 0.
- **Why that breaks learning.**
  - All controller-to-action flow is exit flow, so the action modules starve.
  - $M^{\ast}_{\mathcal{C}}$ diverges.
  - Pairings on controller-to-action synapses, which occur at a rate proportional to $J$, vanish, so the task term cannot learn either.
  - The most stimulus-driven controller neurons seal first, so the policy also loses its dependence on the stimulus.
- **Evidence.** In a preliminary flow-level simulation (24 neurons, structural term only, $\lambda = 0.5$), flow into the action modules fell from 0.21 to 0 within 500 iterations, and $M^{\ast}$ overflowed by 2000.

**Proposal: dual routing term.** Minimise $L$ subject to a target routing flow $q^{\ast}$ out of the controller. Its Lagrangian is $D_\mu = L - \mu (J_{route} - q^{\ast})$, with $J_{route} = \sum_{b \in \mathcal{A} \cup \mathcal{L}} J_{b \mathcal{C}}$. The modulator becomes

$$
g_{ba} = M^{\ast}_a \mathbb{I}[b \ne a] - \mu \mathbb{I}[a = \mathcal{C}, b \in \mathcal{A} \cup \mathcal{L}], \qquad \mu \leftarrow \max \left( 0, \mu + \eta_\mu \left( q^{\ast} - J_{route} \right) \right) \qquad \text{(4a)}
$$

This is still a $K \times K$ table, so it stays crossbar-native, and (5) and Theorem 3 apply to $D_\mu$ unchanged. The gradient was checked against finite differences (Appendix A). Because routing is held at $q^{\ast} > 0$, $q_{\mathcal{C}}$ stays positive and $M^{\ast}_{\mathcal{C}}$ stays finite.

The simplest variant **exempts** the routing pairs from the map term, setting $g = 0$ there.

In a preliminary flow-level contextual bandit (4 seeds, 6000 frames, action-gated eligibility), the final P(correct) was:

| Structural term | P(correct) | Collapsed runs |
| --- | --- | --- |
| none ($\lambda = 0$) | 0.657 | — |
| map term alone, $\lambda = 0.5$ | 0.50 | 3 of 4 |
| map term, routing pairs exempt | 0.693 | 0 |
| map term with dual (4a) | 0.762 | 0 |

So, once routing is protected, the structural term **helps**: it seals the action modules, and leakage out of the action modules costs P(correct). These are preliminary numbers that still need checking on real tasks. $q^{\ast}$ is a free parameter, and whether it should be learned from reward is open.

**Closed loop (preliminary).** Flow-level CartPole: conjunctive encoding, controller recurrence $p_{cc} = 0.1$, 1500 episodes, mean return over the final 50 episodes, 3 seeds.

| Structural term | Mean return |
| --- | --- |
| $\lambda = 0$ | ≈195 |
| $\lambda = 0.05$, routing exempt | **≈343** |
| $\lambda = 0.2$, routing exempt | ≈310 |
| $\lambda = 0.05$, no protection | ≈53 |

- **Starvation is real in the closed loop**, even at low controller recurrence. In open-loop M1 it bit only with a dense controller.
- **With routing exempt, the structural term helps.**
- **The exemption has no free parameter,** so it is the M2 default. Under it, the controller's column of $g$ is zero, so the controller drops out of the structural term and acts as an input layer.

**M2 (iteration 002; 5 seeds, 1500 episodes, mean of the last 100).** Model v3: homeostasis (4d), race burn-in, annealed task rate.

| Structural term | Flow | Spiking |
| --- | --- | --- |
| $\lambda = 0$ | 187 ± 15 | 77 ± 4 |
| $\lambda = 0.05$, routing exempt | **287 ± 21** | **184 ± 13** |

The structural term raises return in both backends. It also drives latent and action modules to near-complete sealing (persistence 0.996 at the flow level, 0.98 in spiking). That is far beyond the intrinsic share of cortical inputs, about 80% [Markov et al. 2011], and without an exit floor it is the map equation's optimum by construction, because leakage always costs bits (§3.3). On CartPole it costs no return. **Proposal: exit floor**, for tasks that need latent→action paths (M4). It is a per-module dual, $g_{ba} = (M^{\ast}_a - \nu_a) \mathbb{I}[b \ne a]$ with $\nu_a \leftarrow \max(0, \nu_a + \eta_\nu (q_{min} p_a - q_a))$. It is still a $K \times K$ table but needs `/derivation-check` before adoption.

**Dual (4a) in the closed loop.** With $q^{\ast} = 0.75 J_{route}(W_0)$, $\mu$ never settles: it ends at 2600–8400, because the map term keeps $J_{route}$ just below $q^{\ast}$. The dual then acts as strong routing potentiation, not as a constraint at its target. It was within one sd of the exemption in v1 (276 ± 26 vs. 258 ± 21), and worse in v2 (240 ± 52 vs. 261 ± 21).

**Schedule.** In these tests a constant $\lambda$ from frame 0 beat $\lambda = 0$, so the ramp is now an ablation.

### 4.6 The task term

**Why plain reward-modulated STDP is not enough.** The usual task term is $\eta (R - \bar{R}) e_{ij}$, where $e_{ij}$ is a low-pass trace of STDP, and subtracting $\bar{R}$ keeps it from drifting [Frémaux et al. 2010]. It has two problems here:

- **No action credit.** If the action is sampled from the flow shares (1d), independently of the spikes that build $e_{ij}$, then $\mathbb{E}[(R - \bar{R}) e_{ij}] = 0$ and nothing is learned. In a preliminary two-action contextual bandit, plain reward-modulated STDP stayed at chance.
- **Sparse reward.** With sparse or delayed reward (MountainCar, LunarLander), $R - \bar{R}$ is near zero on almost every frame. Spiking agents that solved control tasks, including cartpole and acrobot swing-up, used a TD error in an actor-critic instead [Frémaux et al. 2013].

**Action-gated eligibility.** With $\pi_f$ held fixed, take the surrogate policy $\tilde{P}(a \mid \mathbf{o}) = J^{in}_a / \sum_{a' \in \mathcal{A}} J^{in}_{a'}$, where $J^{in}_b = \sum_{i \in b} \sum_j J_{ij}$ is the link flow into module $b$. Its score with respect to $J_{ij}$ depends only on the module of $i$:

$$
h_b = \frac{\mathbb{I}[b = a^{\ast}]}{J^{in}_{a^{\ast}}} - \frac{\mathbb{I}[b \in \mathcal{A}]}{\sum_{a' \in \mathcal{A}} J^{in}_{a'}}, \qquad e_{ij} \leftarrow \left( 1 - \frac{1}{\tau_e} \right) e_{ij} + \kappa_{ij} \left( h_{m(i)} - \bar{h}_j \right), \qquad \bar{h}_j = \sum_k T_{kj} h_{m(k)} \qquad \text{(4b)}
$$

Here $a^{\ast}$ is the chosen action. By the Lemma, the expected trace increment in a frame is $W_{ij} \partial \log \tilde{P}(a^{\ast} \mid \mathbf{o}) / \partial W_{ij}$. That is a weight-scaled policy-gradient score, with exactly the form of (4): a $K$-vector broadcast, a per-neuron baseline and a coincidence. The surrogate is close to the true policy. The cosine with $W \odot \nabla \log P$ computed through the resolvent was $0.997$ (Appendix A).

**TD error as third factor.** The task update is $\Delta W_{ij} = \eta \delta_f e_{ij}$, with a global TD error from a linear critic on the controller's flow shares:

$$
\delta_f = r_f + \gamma V_{f+1} - V_f, \qquad V_f = u \cdot x_f, \qquad x_f = \frac{\pi_{\mathcal{C}}(f)}{p_{\mathcal{C}}(f)}, \qquad u \leftarrow u + \eta_V \delta_f \frac{x_f}{\lVert x_f \rVert^2} \qquad \text{(4c)}
$$

Here $\pi_{\mathcal{C}}$ is the flow on the controller neurons, so $x_f$ sums to 1. The update is normalised LMS. In CartPole the reward is $r_f = -1$ on termination and 0 otherwise. $V_{f+1} = 0$ on termination, and the critic bootstraps on truncation. The critic weights $u$ are a vector over controller neurons in the periphery, and $\delta$ is a global scalar, so the task term stays crossbar-native. The earlier module-level critic, $V_f = \sum_b u_b p_b(f)$, did worse in flow-level CartPole tests (160 against 195; preliminary, iteration 001). A dedicated value population is an untested alternative. For one-step tasks (bandits), $\delta = R - \bar{R}(\mathbf{o})$.

**Biological cost.** The vector $h$ requires the identity of the chosen action to be broadcast to the action modules. In basal-ganglia terms, the selected channel is tagged by its own activity when dopamine arrives [Gurney et al. 2015], which is more natural than a separate efference copy. The $1/J^{in}$ factors read as divisive normalisation, an abstraction. $\bar{h}_j$, like $\bar{g}_j$, is a presynaptic baseline over $j$'s targets and is not available at the postsynaptic synapse (§6).

**Timescale.** $\tau_e = 3$ frames is 0.3–0.75 s under the 100–250 ms frame mapping of §2.3, inside the 0.3–2 s window of dopamine-gated plasticity [Yagishita et al. 2014]. In CartPole's own time (20 ms per step) it is only 60 ms. The frame mapping is the one intended.

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

**What each modulator has to broadcast.** This matters for biology more than for hardware:

- **Map equation.** It needs only $K$ scalars $M^{\ast}_a$, plus a same/different-module mask and the global $q_{\curvearrowright}$. So it is the most plausible candidate for a per-region neuromodulator.
- **Markov stability at lag 1.** Only a constant and a mask.
- **Cost-weighted map equation, $h_K$ and $\sigma_K$.** These need full $K \times K$ tables. That is fine for a crossbar's periphery, but it is an abstraction biologically: no known neuromodulator carries a table of pairwise costs.

### 5.1 Map equation (baseline)

This is §4.4. The modulator fires only on module exits, with strength $M^{\ast}_a$.

- **Pointwise cost.** Within module $b$, the codeword is $-\log \frac{\pi_i}{p_b + q_b}$. On an exit from $a$ to $b$, add $-\log \frac{q_a}{p_a + q_a} - \log \frac{q_b}{q_{\curvearrowright}}$.
- **Expected cost.** It averages to $L(M)$ exactly at $\alpha = 0$ and to within $O(\alpha)$ otherwise, since teleportation shifts arrival rates slightly away from $\pi$. The Monte Carlo error was $0.1\%$ at $\alpha = 0.15$.
- **Reading.** Crossing into a rarely visited module is the expensive event *in bits*. It is metabolically expensive only if rare transitions use long or costly projections. Metabolic cost scales with spikes, synaptic events and wiring [Attwell & Laughlin 2001], regardless of how rare a transition is. §5.4 makes that link explicit.

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
- **Reading.** This is the candidate most literally "thermodynamic". Lynn et al. measured coarse-grained entropy production of human brain-state transitions and found that it rises with task demands [Lynn et al. 2021]. Their measure comes from fMRI and is not a metabolic measurement.
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
| Task eligibility $e_{ij}$ | the only per-synapse state; one option is volatile, diffusive memristors whose decay emulates a trace [Wang et al. 2017]. The action gating $h_{m(i)} - \bar{h}_j$ of (4b) is applied like (4): a $K$-vector by column mask, plus a per-row baseline |
| TD error $\delta$, critic $u$ | a global scalar and a vector over controller neurons in the periphery (4c) |

No step requires a nonlinear function of an individual device's conductance, or access to the transposed element $W_{ji}$. This is why every modulator is defined at module level.

**Locality in the biological reading.** A synapse $j \to i$ sits on the dendrite of $i$, inside $i$'s module. So a modulator released in a module reaches the synapses whose *postsynaptic* neuron is in that module. The local information is therefore the postsynaptic module $b$. What the synapse lacks is the identity of the presynaptic module $a$.

- **Partial help from dendrites.** Local and long-range inputs are partly segregated by dendritic compartment [Petreanu et al. 2009]. That could supply the bit $\mathbb{I}[a \ne b]$, but not $a$ itself.
- **Consequence for the map equation.** The modulator $g_{ba} = M^{\ast}_a \mathbb{I}[b \ne a]$ is indexed by the presynaptic module. Biologically it needs either $M^{\ast}$ roughly uniform across modules, or a projection-specific signal.
- **What neuromodulators can carry.** They are slow: dopamine gates spike-timing plasticity over a window of about 0.3–2 s [Yagishita et al. 2014]. They are spatially specific at most regionally, as in functional clustering in VTA [Engelhard et al. 2019] and localised striatal dopamine waves [Hamid et al. 2021]. A per-region scalar is plausible; a full $K \times K$ table is not. Astrocyte domains are an alternative local carrier: they integrate activity over seconds and gate NMDA-receptor-dependent LTP [Henneberger et al. 2010].
- **The baseline.** $\bar{g}_j$ can be read as presynaptic renormalisation (§4.2).

In hardware, the presynaptic module is a row label and the postsynaptic module a column mask, so none of this is an obstacle. §7 removes the explicit labels.

## 7. Removing explicit modules: lateral inhibition

In cortex, modules emerge from local competition rather than from labels. Let $I_{ij} \ge 0$ be inhibition from $j$ onto $i$, and replace the membership bit with a soft competition score:

$$
\chi_{ij} = \frac{I_{ij}}{I_{ij} + \kappa_I} \in [0, 1), \qquad G_{\chi}(i, j) = \chi_{ij} - \bar{e}^{\chi}_j, \qquad \bar{e}^{\chi}_j = \frac{1}{d_j} \sum_{i'} \chi_{i'j} W_{i'j} \qquad \text{(8)}
$$

When $\chi$ is binary, (8) reduces to $G$. In the three-factor form, the map modulator becomes $g_{ij} = M^{\ast} \chi_{ij}$, which needs a module-level $M^{\ast}$ even without explicit modules. That is an open question.

Compared with lateral inhibition alone, as in Kohonen maps, which produces whatever partition winner-take-all dynamics yield, Map-STDP adds a measurable objective and adaptive gain.

**Problems with direct inhibition.** The proposed inhibitory update $\Delta I_{ij} = \eta_{inh} \pi_i \pi_j$ has three:

- it is unbounded;
- it makes co-active neurons into competitors, which opposes Hebbian STDP;
- direct inhibition between excitatory neurons violates Dale's law, and the main network has no inhibitory population at all.

**Proposal: route inhibition through interneurons.** Give each module a pool of inhibitory, PV-like interneurons, with E→I→E connections, and let $\chi_{ij}$ be defined through disynaptic inhibition. Plasticity on I→E synapses follows a bounded, homeostatic inhibitory STDP rule that sets each excitatory neuron's rate to a target [Vogels et al. 2011]. The same pools provide the global divisive normalisation that the mean-field assumption needs (§2.2), and the rate homeostasis that keeps the network from running away or falling silent (§8).

## 8. Training

Training runs frame by frame. For each environment frame $f$:

1. **Observe.** Read $\mathbf{o}_f$ and form the teleportation vector $v_f$, concentrated on the controller. With carry-over, mix in the previous frame's latent flow (1b).
2. **Walk.** Run the $n$ hops of (1a). In software or on a crossbar this gives $\pi_f$ directly. In the spiking network, the neurons run for the frame window, $k = 100$ steps in M2, cold-started each frame (§2.3), and (2) estimates $\hat{\pi}_f$. A warm start would shorten the transient that the burn-in in step 3 skips.
3. **Act.** Sample the action from (1d), or use the race readout in the spiking network, after a burn-in of about $\ln 0.1 / \ln(1 - \alpha)$ steps (§2.4).
4. **Module statistics.** Compute $p_a$, $J_{ba}$ and $q_a$ for the frame. Update the modulator table $g_{ba}$ (per frame or as a running average, §3.4) and, if used, the routing dual $\mu$ (4a). Refresh the per-row baselines $\bar{g}_j$ and $\bar{h}_j$.
5. **Plasticity.**
   - Add the action-gated increments (4b) to the eligibility traces.
   - Apply the structural term (4) and the $d_j$ homeostasis (4d) to the frame's pairings. In spiking, the pairings are the covariance count (§2.2).
   - After the next frame's walk, compute the TD error (4c), update the critic, and apply $\eta_e \delta_f e_{ij}$ with the annealed rate $\eta_e$ (below).
6. **Carry over.** Keep the latent flow $P_{\mathcal{L}} \pi_f$ for the next frame.

**Stability measures.**

- **Clip $M^{\ast}$.** Clipping alone does not stop the drive toward $\bar{e} = 0$, because $M^{\ast} \ge 0$. That needs (4a).
- **Add an additive floor or rewiring** (below), because multiplicative updates freeze near-zero synapses, and finite conductance levels make them exactly zero.
- **Add rate homeostasis or inhibitory plasticity** (§7).
- **Log rates and the branching ratio** from M1 onward.
- **Ramp $\lambda$** from about 0 until return exceeds random (§4.5).
- **Anneal $\eta$.** A constant task rate stalls in the closed loop. In M2, $\eta_e = \eta / (1 + e / 300)$ over episodes $e$ raised spiking return from 88 (constant $\eta = 10$) to 184. Before (4d), a ratchet in $d_j$ had done this by accident: the step in $T = W/d$ is about $\eta \Delta W / d_j$. The schedule is a global scalar, so it is crossbar-native; biologically it is an abstraction, for example of a declining dopamine response gain.

**EM split.** $\pi_f$ and the module statistics are treated as fixed within each update. Two conditions are needed: the walk converges within a frame (enough hops $n$), and plasticity is slow relative to frames ($\lambda, \eta$ small). In the spiking variant, the estimate (2) must also settle within the frame window ($\beta$ large enough).

Periodically:

- **Re-detect latent modules (optional).** The controller and action modules stay pinned, because they carry the input and the policy. Latent modules may be re-detected with Infomap or Leiden, which is the forward step of the alternating minimisation in §3.4. Static modules only reinforce the initial partition. Any partition works with (4).
- **Rewire.** Prune synapses below $\epsilon$ or outside each neuron's top $k$, and create synapses. The SBM modulator (§5.6) can gate these events.

Initialisation is either a stochastic block model (head start, but biased toward its partition) or Erdős–Rényi (neutral, slow). The **controller** module receives the stimulus, concentrating $v(\mathbf{o})$, and routes to the latent and action modules. Keep its internal recurrence low (§2.4).

**Timescales in numbers.** These still need to be reported:

- the plasticity rates $\lambda$ and $\eta$ against the frame rate;
- per-frame against running-average $g_{ba}$ when the policy is non-stationary.

In biology, a frame of 100–250 ms sits between the walk relaxation (tens of ms) and neuromodulated plasticity (seconds and longer), so the ordering holds.

## 9. Assumptions and limitations

1. **EM split.** $\pi$ and the module statistics are held fixed during updates.
2. **Mean field.** Flow equals normalised firing rate, and causal pairings occur at a rate proportional to $J_{ij}$ (§2.2). Theorem 3 depends on both. It needs constant $d_j$, global normalisation and a pairing count that removes chance coincidences (the covariance count, §2.2). In spiking it held approximately: the covariance count's frame-averaged update had cosine 0.992 with $-W \odot \nabla D$ (M1).
3. **Covariance rule.** (4) is exact only in expectation. Its variance grows with the number of synapses sharing a modulator, which is why modulators are per module and baselines per neuron.
4. **Coarse-graining.** Module-level $\sigma_K$ lower-bounds the full $\sigma$ and misses irreversibility inside modules. In a test network it captured only 4% of $\sigma$ (0.018 vs. 0.41). $h_K$ has no general ordering with $h$.
5. **Estimator candidates.** Predictive dissipation and information flow (§5.7–5.8) rely on peripheral estimators with unknown bias.
6. **Teleportation.** It is not counted as an exit, and it perturbs the map equation's codeword averages by $O(\alpha)$.
7. **Divergence.** The map equation's $M^{\ast}_m$ diverges as $q_m \to 0$, for a fully sealed module. It needs clipping, and for the controller the routing dual (4a).
8. **Lateral inhibition.** The inhibitory plasticity is unresolved (§7).
9. **Convergence within a frame.** Equation (1a) bounds the error of $n$ hops, but the spiking estimate has extra sampling noise, and in a short frame window it may not settle.
10. **Carry-over.** $\rho$ is a free parameter. With $\pi_{f-1}$ held fixed, the gradient ignores the dependence through earlier frames (cosine $0.98$ with the full gradient in a test).
11. **Number of latent modules.** $\lvert \mathcal{L} \rvert$ is a hyperparameter unless latent modules are re-detected.
12. **Routing conflict.** The map term starves the controller's routing unless the dual (4a) or an exemption is used (§4.5). $q^{\ast}$ is a free parameter.
13. **Surrogate policy gradient.** The action-gated eligibility (4b) follows the gradient of the one-step inflow surrogate, not of (1d) exactly (cosine $0.997$ in a test). The TD critic (4c) is linear in the controller's flow shares.
14. **Policy family.** (1d) has no temperature, and controller recurrence blurs it (§2.4). The sharpened family (1e) was unstable at the flow level without an entropy floor (M2).
15. **Carry-over memory.** Linear carry-over forgets geometrically. Holding a cue needs the module-level nonlinearity proposed in §2.3.

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

**Numerical checks for the review revisions** (random 13-node network: a controller of 4, 1 latent and 2 action modules of 3; $\alpha = 0.2$):

- action-gated eligibility (4b): the surrogate score gradient matches finite differences to $\le 3 \times 10^{-10}$. Its cosine with $W \odot \nabla \log P$, computed through the exact resolvent, is $0.997$–$0.998$ over 5 stimuli. On a 6-stimulus bandit, the Monte Carlo average of $\delta \cdot$ (4b) over $2 \times 10^4$ frames of 300 sampled pairings matches $W \odot \nabla \mathbb{E}[R]$ (surrogate) with correlation $0.997$;
- baseline as renormalisation (§4.2): the relative difference between the baselined update and the unbaselined update followed by restoring $d_j$ is $2.5 \times 10^{-4}$, $2.5 \times 10^{-5}$ and $2.5 \times 10^{-6}$ at $\lambda = 10^{-2}, 10^{-3}, 10^{-4}$, i.e. first order;
- dual routing term (4a): its gradient via (5) matches finite differences of $D_\mu$ to $\le 6 \times 10^{-10}$;
- carry-over (1b): the L1 distance between two histories shrinks by $0.29$ per frame at $\rho = 0.5$ and $0.39$ at $\rho = 0.9$, to $3 \times 10^{-8}$ and $6 \times 10^{-7}$ after 10 frames.

The preliminary spiking (Hawkes) and flow-level bandit numbers quoted in §2.2, §2.3, §2.4 and §4.5 come from a separate review simulation. They have not been re-run, and should be reproduced at M1 and M2.

**Numerical checks from iteration 001** (M1; code in `mapstdp/`, records in `docs/iterations/001-m1.md`):

- **Pairing counts.** Monte Carlo over 5000 frames in a numpy branching simulator with the same law as SuperNeuroMAT branching mode (80 neurons, $\alpha = 0.2$, $k = 50$). Cosine between the average update and $-W \odot \nabla D$:

  | Count | Cosine |
  | --- | --- |
  | Expected (exact $\kappa$) | 1.0000 |
  | Covariance | 0.9916 |
  | Causal | 0.9787 |
  | Balanced | 0.8443 |

  This reproduces the reviewer's 20000-frame run (0.992, 0.979, 0.847).
- **`mapstdp/core.py` self-check:**
  - Eq. 7 matches finite differences;
  - $\sum_i W_{ij} \partial D / \partial W_{ij} = 0$, on which the $d_j$ homeostasis of §2.2 rests;
  - the three-factor Monte Carlo correlates 1.0000 with $-W \odot \nabla L$.

**Numerical checks from iteration 002** (M2; `mapstdp/core.py` self-check, records in `docs/iterations/002-m2.md`):

- the sharpened score of (1e) with $\beta = 3$: the expected eligibility increment ($\kappa = J$) equals $W \odot \partial \log \tilde{P}_\beta / \partial W$ by finite differences to $\le 10^{-6}$ on every synapse, as for $\beta = 1$ (80-neuron network);
- homeostasis (4d): in a spiking Monte Carlo (2000 frames, $d_j$ spread over $[0.6, 2]$), the average update has cosine 0.992 with its expectation $-\epsilon_h (d_j - 1) \pi_j W_{ij} / d_j$. The fitted per-column factor correlates 0.995 with the prediction (preliminary; `snn-expert`).

## Appendix B: Corrections to the earlier derivation

1. **Sign error.** The earlier version expanded $L(M)$ with $+$ on its second and third terms; both are negative. The gradient's log factor becomes $M^{\ast}_m = \log \frac{q_{\curvearrowright}(p_m+q_m)}{q_m^2} \ge 0$, not $\log \frac{q_{\curvearrowright}}{p_m+q_m} \le 0$.
2. **Direction convention.** $W_{ij}$ is the synapse from pre $j$ to post $i$ throughout, and the walk follows spikes. The earlier version row-normalised but wrote $\tilde{p} = \tilde{T} \tilde{p}$. Gating is now indexed by the presynaptic neuron.
3. **Teleportation.** The stimulus enters as $\alpha v(\mathbf{o})$, making the flow unique and the rule stimulus-dependent.
4. **Flow vs. firing.** The earlier version equated $p(z_j = 1 \mid \mathbf{o})$ with the stationary distribution. That is now an explicit assumption (§2.2).
5. **Convergence.** The earlier version showed only a fixed point. This version proves contraction.
6. **Lateral inhibition.** The earlier $\varphi_{ij}$ did not reduce to the indicator; it is replaced by $\chi_{ij}$, and the problems with the inhibitory update are flagged.
7. **Three-factor restructure.** The structural term is a neuromodulated, cost-gated STDP rule, (4). Its modulator is the marginal description cost, and eight candidate description lengths plug in. The factor $M^{\ast}_m$, previously absorbed into the learning rate, is the map equation's per-module neuromodulator. $G$ is the lag-1 Markov-stability gradient. Every modulator is defined at module level for memristive crossbars.
8. **Reformulation: communities as states (2026-10-04).** The primary goal is now a state-space abstraction, and thermodynamics is a secondary hypothesis. Modules are typed as controller, action and latent. Flow is computed once per environment frame by a truncated walk, (1a), with optional carry-over of latent flow, (1b). The module chain $T^K(\mathbf{o})$, (1c), is the extracted Markov model, and the policy, (1d), is read from the action modules. The rule descends the frame-averaged description length, (3a): the inverse of community detection. The rule (4) and Theorems 1–3 are unchanged; they apply per frame.
9. **Review revisions (2026-10-04).** These follow a neuroscience review and an SNN-performance review.
   - **Task term.** It now uses an action-gated eligibility trace (4b) with a TD error from a module-level critic (4c), because plain $R - \bar{R}$ STDP gets no action credit under (1d).
   - **Routing conflict.** The map term starves controller routing; a dual routing term (4a) is proposed (§4.5).
   - **Spiking timing.** Spiking frames take about $4 \tau / \alpha$ (100–250 ms), not one synaptic delay per hop.
   - **Mean field.** It needs constant $d_j$, global normalisation, and causal-minus-acausal pairing counting.
   - **Locality.** It is restated: the postsynaptic module is the local one.
   - **Baseline.** $\bar{g}_j$ can be read as presynaptic renormalisation.
   - **Readout and policy.** A race readout matches (1d), and the limits of the policy family are stated.
   - **Carry-over.** Linear carry-over forgets geometrically; a module-level nonlinearity is proposed.
   - **Lateral inhibition.** It is routed through interneurons with inhibitory STDP, so it respects Dale's law.
   - **Claims softened.** Controller ↔ cortex is now a functional abstraction (basal ganglia, OFC and hippocampal analogues), Buesing et al. is correctly characterised, and the metabolic reading of the map equation is qualified.
10. **Iteration 001, M1 (2026-10-05, new).** These come from the first implementation and its reviews.
    - **Pairing count.** $\kappa$ is the covariance count, replacing the balanced count (§2.2).
    - **Presynaptic normalisation.** Adopted as a per-neuron output gain $1/d_j$, plus $d_j$ homeostasis in the baseline (§2.2).
    - **Spiking frame length.** Stated in spikes per module, with error $\approx 1.3 / \sqrt{S_{mod}}$ (§2.3).
    - **Encoding.** Conjunctive receptive fields (§2.4).
    - **Routing.** Exemption is the default routing protection, and the closed-loop starvation numbers are added (§4.5).
11. **Iteration 002, M2 (2026-10-05).** These come from the CartPole closed loop and its reviews.
    - **Homeostasis.** It is now its own term on all plasticity, (4d), not part of the structural baseline. Inside $\lambda$ it was too weak, and the task term ratcheted $d_j$ (§2.2).
    - **Race.** The race equals (1d) only after the walk has relaxed, so a burn-in is adopted (§2.4).
    - **Sharpened policy.** (1e) and its matched score are a Proposal (§2.4).
    - **Annealed task rate** (§8).
    - **M2 results, sealing caveat, exit-floor Proposal, and dual behaviour** (§4.5).
    - **Biology.** Notes on $\bar{h}_j$ locality and on the $\tau_e$ timescale (§4.6).


## References

- Attwell, D., & Laughlin, S. B. (2001). An energy budget for signaling in the grey matter of the brain. *Journal of Cerebral Blood Flow & Metabolism*, 21(10), 1133–1145.
- Baars, B. J. (2005). Global workspace theory of consciousness: toward a cognitive neuroscience of human experience. *Progress in Brain Research*, 150, 45–53.
- Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*, 11, 3625.
- Buesing, L., Bill, J., Nessler, B., & Maass, W. (2011). Neural dynamics as sampling: a model for stochastic computation in recurrent networks of spiking neurons. *PLoS Computational Biology*, 7(11), e1002211.
- Chistiakova, M., Bannon, N. M., Bazhenov, M., & Volgushev, M. (2014). Heterosynaptic plasticity: multiple mechanisms and multiple roles. *The Neuroscientist*, 20(5), 483–498.
- Dayan, P., & Abbott, L. F. (2001). Information theory. In *Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems* (pp. 123–150). MIT Press.
- Delvenne, J.-C., Yaliraki, S. N., & Barahona, M. (2010). Stability of graph communities across time scales. *PNAS*, 107(29), 12755–12760.
- Engelhard, B., Finkelstein, J., Cox, J., et al. (2019). Specialized coding of sensory, motor and cognitive variables in VTA dopamine neurons. *Nature*, 570, 509–513.
- Frank, M. J. (2006). Hold your horses: a dynamic computational role for the subthalamic nucleus in decision making. *Neural Networks*, 19(8), 1120–1136.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional requirements for reward-modulated spike-timing-dependent plasticity. *Journal of Neuroscience*, 30(40), 13326–13337.
- Frémaux, N., Sprekeler, H., & Gerstner, W. (2013). Reinforcement learning using a continuous time actor-critic framework with spiking neurons. *PLoS Computational Biology*, 9(4), e1003024.
- Gershman, S. J., & Niv, Y. (2010). Learning latent structure: carving nature at its joints. *Current Opinion in Neurobiology*, 20(2), 251–256.
- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural Circuits*, 12, 53.
- Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision making. *Annual Review of Neuroscience*, 30, 535–574.
- Gurney, K. N., Humphries, M. D., & Redgrave, P. (2015). A new framework for cortico-striatal plasticity: behavioural theory meets in vitro data at the reinforcement-action interface. *PLoS Biology*, 13(1), e1002034.
- Hamid, A. A., Frank, M. J., & Moore, C. I. (2021). Wave-like dopamine dynamics as a mechanism for spatiotemporal credit assignment. *Cell*, 184(10), 2733–2749.
- Henneberger, C., Papouin, T., Oliet, S. H. R., & Rusakov, D. A. (2010). Long-term potentiation depends on release of D-serine from astrocytes. *Nature*, 463, 232–236.
- Horowitz, J. M., & Esposito, M. (2014). Thermodynamics with continuous information flow. *Physical Review X*, 4, 031015.
- Kempter, R., Gerstner, W., & van Hemmen, J. L. (1999). Hebbian learning and spiking neurons. *Physical Review E*, 59(4), 4498–4514.
- Li, H. L., & van Rossum, M. C. W. (2020). Energy efficient synaptic plasticity. *eLife*, 9, e50804.
- Litwin-Kumar, A., & Doiron, B. (2012). Slow dynamics and high variability in balanced cortical networks with clustered connections. *Nature Neuroscience*, 15(11), 1498–1505.
- Lynn, C. W., Cornblath, E. J., Papadopoulos, L., Bertolero, M. A., & Bassett, D. S. (2021). Broken detailed balance and entropy production in the human brain. *PNAS*, 118(47), e2109889118.
- Markov, N. T., Misery, P., Falchier, A., Lamy, C., Vezoli, J., Quilodran, R., Gariel, M. A., Giroud, P., Ercsey-Ravasz, M., Pilaz, L. R., Huissoud, C., Barone, P., Dehay, C., Toroczkai, Z., Van Essen, D. C., Kennedy, H., & Knoblauch, K. (2011). Weight consistency specifies regularities of macaque cortical networks. *Cerebral Cortex*, 21(6), 1254–1272.
- Mazzucato, L., Fontanini, A., & La Camera, G. (2015). Dynamics of multistable states during ongoing and evoked cortical activity. *Journal of Neuroscience*, 35(21), 8214–8231.
- Peixoto, T. P. (2014). Hierarchical block structures and high-resolution model selection in large networks. *Physical Review X*, 4, 011047.
- Petreanu, L., Mao, T., Sternson, S. M., & Svoboda, K. (2009). The subcellular organization of neocortical excitatory connections. *Nature*, 457, 1142–1145.
- Plaçais, P.-Y., & Preat, T. (2013). To favor survival under food shortage, the brain disables costly memory. *Science*, 339(6118), 440–442.
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a vertebrate solution to the selection problem? *Neuroscience*, 89(4), 1009–1023.
- van Rossum, M. C. W., Bi, G. Q., & Turrigiano, G. G. (2000). Stable Hebbian learning from spike timing-dependent plasticity. *Journal of Neuroscience*, 20(23), 8812–8821.
- Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex networks reveal community structure. *PNAS*, 105(4), 1118–1123.
- Royer, S., & Paré, D. (2003). Conservation of total synaptic weight through balanced synaptic depression and potentiation. *Nature*, 422(6931), 518–522.
- Schnakenberg, J. (1976). Network theory of microscopic and macroscopic behavior of master equation systems. *Reviews of Modern Physics*, 48(4), 571–585.
- Seifert, U. (2005). Entropy production along a stochastic trajectory and an integral fluctuation theorem. *Physical Review Letters*, 95, 040602.
- Sejnowski, T. J. (1977). Storing covariance with nonlinearly interacting neurons. *Journal of Mathematical Biology*, 4(4), 303–321.
- Shouval, H. Z., Wang, S. S.-H., & Wittenberg, G. M. (2010). Spike timing dependent plasticity: a consequence of more fundamental learning rules. *Frontiers in Computational Neuroscience*, 4.
- Still, S., Sivak, D. A., Bell, A. J., & Crooks, G. E. (2012). Thermodynamics of prediction. *Physical Review Letters*, 109, 120604.
- Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391(6670), 892–896.
- Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. *Science*, 334(6062), 1569–1573.
- Wang, Z., Joshi, S., Savel'ev, S. E., et al. (2017). Memristors with diffusive dynamics as synaptic emulators for neuromorphic computing. *Nature Materials*, 16, 101–108.
- Wilson, R. C., Takahashi, Y. K., Schoenbaum, G., & Niv, Y. (2014). Orbitofrontal cortex as a cognitive map of task space. *Neuron*, 81(2), 267–279.
- Yagishita, S., Hayashi-Takagi, A., Ellis-Davies, G. C. R., Urakubo, H., Ishii, S., & Kasai, H. (2014). A critical time window for dopamine actions on the structural plasticity of dendritic spines. *Science*, 345(6204), 1616–1620.
- Zenke, F., & Gerstner, W. (2017). Hebbian plasticity requires compensatory processes on multiple timescales. *Philosophical Transactions of the Royal Society B*, 372(1715), 20160259.
