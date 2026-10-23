# Map-STDP: communities as states

Author: Vikram Ramavarapu

**Goal.** Give a spiking network a **state-space abstraction** whose states are its own communities. A random walk on a graph with communities has a minimum description length, the map equation [Rosvall & Bergstrom 2008]. Community detection solves the *forward* problem: given the graph, find the partition that minimises it. Map-STDP solves the **inverse** problem. The partition is given (one controller community, one community per action, some latent communities), and plasticity shapes the synapses until that partition is the minimum-description-length one. The coarse-grained walk between communities is then a stimulus-conditioned Markov model over the network's states. It can be read out, and in an RL environment it chooses the action.

**The walk is the network's linear response.** One extra spike propagates through the network as a branching cascade with a mean offspring matrix $\Lambda$. That cascade, normalised, is the walk that the map equation codes (§2.1). The implemented engine (SuperNeuroMAT branching mode and its numpy twin) is the special case $\Lambda = (1 - \alpha) T$, and every implemented result in §§2–4 is stated for it. The general case, which a QIF network needs, is §5 (**Proposal**).

**Each environment frame is one walk.** The stimulus of frame $f$ re-injects the walker, and the walk is run implicitly to its per-frame flow $\pi_f$ in a fixed number of hops (§2.3). Description length, modulators and the extracted Markov model are all per frame.

**Side hypothesis (secondary).** Description length tracks the metabolic cost of communication, which is hypothesised to be higher when activity crosses into modules a neuron rarely talks to. The state abstraction does not depend on it.

## At a glance

For every causal spike pairing $j \to i$ (pre $j$ fires, then post $i$; count $\kappa_{ij}$), the synapse changes by

$$
\Delta W_{ij} = \underbrace{\eta \delta e_{ij}}_{\text{task: TD-modulated STDP}} - \underbrace{\lambda \left( g_{ij} - \bar{g}_j \right) \kappa_{ij}}_{\text{structure: cost-modulated STDP}} - \underbrace{\epsilon_h \left( d_j - 1 \right) \kappa_{ij}}_{\text{output homeostasis}}
$$

- $\delta$ is a global TD error from a critic (4c).
- $e_{ij}$ is an **action-gated** eligibility trace: each pairing adds $\kappa_{ij} (h_{m(i)} - \bar{h}_j)$, where the $K$-vector $h$ marks the chosen action's module (4b).
- $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal** description cost of one more unit of flow from $j$ to $i$, and $\bar{g}_j$ its mean over $j$'s transitions. For the map equation, $g$ fires only when activity *leaves* a module, with strength $M^{\ast}_m$, the marginal bits per exit (6).
- $d_j$ is $j$'s total output weight; the third term holds it near 1 (4d).

Every term is a module-level broadcast, minus a per-neuron baseline, times a spike coincidence, so the rule runs on a memristive crossbar (§6). On average the structural term is weight-scaled gradient descent, $\mathbb{E}[\Delta W_{ij}] = -\lambda W_{ij} \partial D / \partial W_{ij}$ (Theorem 3). For the map equation this weakens a cross-module synapse in proportion to $1 - \bar{e}_j$ and strengthens a within-module one in proportion to $\bar{e}_j$, where $\bar{e}_j$ is the fraction of $j$'s output that leaves its module (7). Controller→action and controller→latent routing is cross-module by definition, so those pairs are **exempt** from the map term (§4.5). In each frame the module-to-module flow is a Markov chain $T^K(\mathbf{o})$ over communities (1c); its flow into the action modules is the policy (1d).

**Proposal (§5).** On a general linear-response walk the same rule descends $D$ preconditioned by each neuron's offspring mean $\nu_j$, provided the baseline is the pairing-weighted mean (Theorem K), and the exact policy score is $\kappa_{ij} \Phi_i$ with a neuron-level cascade value $\Phi$ and no presynaptic baseline (K9); its module-level table $\Phi^K_{m(i)}$ is an approximation. Both need pairing counts proportional to the kernel, which no tested QIF operating point yet delivers.

## 1. Motivation

- **Communities as states.** A spiking network's microstate is far too large to be a state space. If the network is modular, the coarse-grained walk between modules is a small Markov chain that can be read out. Modules mapped onto actions and latent situations make it an abstract model of the task.
- **A central controller.** The stimulus enters one controller community, which routes activity to latent and action communities; this is what makes the extracted chain depend on the stimulus (§2.4). It is a functional abstraction, closer to thalamus or sensory cortex than to any single cortical router. The action readout resembles channel selection in the basal ganglia [Redgrave et al. 1999], latent communities resemble task-state codes in orbitofrontal cortex [Wilson et al. 2014] and latent-structure learning [Gershman & Niv 2010].
- **Neuromodulators gate costly plasticity.** Memory formation has a measurable metabolic cost [Plaçais & Preat 2013; Li & van Rossum 2020], and three-factor rules, where a local eligibility flag becomes a weight change only when a neuromodulator arrives, have direct experimental support [Gerstner et al. 2018]. Here the structural neuromodulator carries a description-length signal, so plasticity is spent where it makes the network's activity cheaper to describe.

## 2. Setup: activity as flow

### Notation

| Symbol | Meaning | Lives at |
| --- | --- | --- |
| $W_{ij} \ge 0$ | synapse from pre $j$ to post $i$ | synapse |
| $d_j = \sum_k W_{kj}$, $T_{ij} = W_{ij} / d_j$ | total output weight of $j$; weight-normalised step | neuron $j$; synapse |
| $r_i$, $G_i$ | firing rate and its slope (gain) at the operating point | neuron $i$ |
| $\Lambda_{ij}$, $\nu_j = \sum_i \Lambda_{ij}$ | mean extra spikes of $i$ caused by one spike of $j$ (offspring); offspring mean of $j$ | synapse; neuron $j$ |
| $\pi_j$, $\pi_{f,j}$ | flow (normalised rate) through $j$; in frame $f$, $\pi_f = \pi(\mathbf{o}_f)$ | neuron $j$ |
| $J_{ij} = \pi_j T_{ij}$ | flow along $j \to i$ | synapse |
| $v(\mathbf{o})$, $\alpha$ | normalised stimulus drive; share of activity it injects per step | input; global |
| $m(j)$, $K$ | module of $j$; number of modules | neuron $j$ |
| $\mathcal{C}$, $\mathcal{A}$, $\mathcal{L}$ | controller; action modules (one per action); latent modules | module labels |
| $e_j = \sum_{i \notin m(j)} W_{ij}$, $\bar{e}_j = e_j / d_j$ | output weight leaving $j$'s module, absolute and as a fraction | neuron $j$ |
| $p_m = \sum_{j \in m} \pi_j$, $q_m = \sum_{j \in m} \pi_j \bar{e}_j$, $q_{\curvearrowright} = \sum_m q_m$ | visit rate, exit rate, total switching rate | module; network |
| $J_{ba} = \sum_{i \in b, j \in a} J_{ij}$, $T^K_{ba} = J_{ba} / p_a$ | module-pair flow and transition probability ($K \times K$ tables indexed [post, pre]) | module pair |
| $f$, $t$, $n$ | environment frame; simulation step; walk hops per frame | global |

### 2.1 The walk: linear response of the network

Let neuron $i$ fire at rate $r_i = F_i(x_i)$, where its input is $x = B r + I^{ext}$, $B_{ij}$ is the charge one spike of $j$ delivers to $i$, and $I^{ext}$ is the stimulus drive. At the fixed point, with gains $G_i = F_i'(x_i)$,

$$
r = F(x), \qquad x = B r + I^{ext}, \qquad G_i = F_i'(x_i) \qquad \text{(K1)}
$$

Perturbing the input gives $\delta r = G \odot (B \delta r + \delta I^{ext})$, so

$$
\delta r = (I - \Lambda)^{-1} \mathrm{diag}(G) \delta I^{ext}, \qquad \Lambda = \mathrm{diag}(G) B \qquad \text{(K2)}
$$

$\Lambda_{ij}$ is the expected number of extra spikes of $i$ caused by one extra spike of $j$: the mean offspring matrix of a multivariate Hawkes process, $\int h_{ij}(s) ds = \Lambda_{ij}$ [Hawkes 1971]. This needs kicks small against the spread of excitabilities and a rate that follows its input linearly over one kick. The fixed point satisfies the exact identity

$$
r = \Lambda r + \mu, \qquad \mu = r - \Lambda r \qquad \text{(K3)}
$$

If $\Lambda \ge 0$, $\rho(\Lambda) < 1$ and $\mu \ge 0$, then $r = \sum_n \Lambda^n \mu$ is the visit measure of a branching walk [Hawkes & Oakes 1974]. Immigrants arrive at rate $\mu$. A visit to $j$ has $\nu_j$ offspring on average, each landing on $i$ with probability $T^{eff}_{ij} = \Lambda_{ij} / \nu_j$. When $\nu_j \le 1$, normalised, $\pi = r / \sum r$ is personalised PageRank with a per-neuron escape probability $1 - \nu_j$ and source $\mu / \sum \mu$.

**The implemented engine is the special case.** In SuperNeuroMAT branching mode (`leak=inf`, uniform threshold noise, $P(\text{spike}) = u / (2 \theta)$ with $2 \theta = 1$, output weights $(1 - \alpha) W / d$), $G = 1$ wherever $u \le 2 \theta$ and the stimulus is Bernoulli input at rate proportional to $v$. Then $\Lambda = (1 - \alpha) T$, $\nu_j = 1 - \alpha$, $T^{eff} = T$, $\mu \propto v(\mathbf{o})$, and (K3) normalised is

$$
\pi = (1 - \alpha) T \pi + \alpha v(\mathbf{o}), \qquad \sum_i \pi_i = 1 \qquad \text{(1)}
$$

the walk that follows spikes forward from $j$ to $i$ with probability $T_{ij}$ and is re-injected by the stimulus with probability $\alpha$ per step. For any $\alpha > 0$ it has a unique solution, reached by power iteration (Theorem 1). The one condition is that the linear escape never saturates, $u_i \le 2 \theta$; on CartPole the measured lag-1 offspring regresses on $0.8 T$ with slope 1.04 (Appendix A). §§2.2–4 use (1).

**A QIF network is not this special case.** For quadratic integrate-and-fire neurons, (K1) uses the QIF rate transfer of each neuron, as in the Montbrió–Pazó–Roxin mean field [Montbrió et al. 2015] (an exact population reduction, applied here per neuron), and $G$ varies by more than an order of magnitude across neurons. The map equation then codes visits with $\pi = r / \sum r$ and steps with $T^{eff}$; escape and immigration are not coded, as with unrecorded teleportation [Lambiotte & Rosvall 2012]. §5 states the rule for this walk.

### 2.2 Estimating flow and pairings from spikes

Each neuron can estimate its own flow from its input,

$$
\hat{\pi}_i \leftarrow (1 - \beta) \hat{\pi}_i + \beta \left[ (1 - \alpha) \sum_j \frac{W_{ij}}{d_j} \hat{z}_j(t) + \alpha v_i \right], \qquad \hat{z}(t) = \frac{z(t)}{\sum_k z_k(t)} \qquad \text{(2)}
$$

and in batch form $\hat{\pi}$ is the frame's normalised spike counts. **Assumption (mean field):** normalised rates satisfy (1), so $\hat{\pi}$ tracks $\pi$, and **causal pairings $j \to i$ occur at a rate proportional to $J_{ij}$**. The second half is what lets coincidences stand in for walk steps in §4. For a stationary Hawkes process the covariance density is $(I - \Lambda)^{-1} \mathrm{diag}(r) (I - \Lambda)^{-\top}$, whose first-order terms are the causal term $\Lambda \mathrm{diag}(r)$, the reverse term, common input and chains [Hawkes 1971; Pernice et al. 2011; Trousdale et al. 2012]. A causal window $w$ over lags $s > 0$ in a frame of length $T_f$ gives

$$
\mathbb{E} \left[ \kappa_{ij} \right] \approx T_f \omega_i \Lambda_{ij} r_j + O(\Lambda^2), \qquad \omega_i = \frac{\int w h_i}{\int h_i} \qquad \text{(K4)}
$$

so $\kappa \propto \Lambda \mathrm{diag}(r)$ only if the window efficiency $\omega_i$ is uniform (a window much longer than the response kernel) and chains and common input are small. The branching engine meets the assumption through three measures.

**Constant presynaptic output.** Spiking drive depends on $W_{ij}$, not $W_{ij} / d_j$. The engine applies a per-neuron output gain $1/d_j$, one column-sum read, so it implements $T$ directly. The conductance range is held by a homeostasis term on all plasticity,

$$
\Delta W_{ij} = -\epsilon_h \left( d_j - 1 \right) \kappa_{ij} \qquad \text{(4d)}
$$

whose average $-\epsilon_h (d_j - 1) \pi_j W_{ij} / d_j$ is a uniform rescaling of column $j$, so it leaves $T$ and $D$ unchanged. Without it the task term ratchets $d_j$ upward through the $w_{min}$ floor (iteration 002). Biologically it reads as fast heterosynaptic plasticity [Zenke & Gerstner 2017; Chistiakova et al. 2014]; conservation of total synaptic weight is documented on the input side [Royer & Paré 2003], so presynaptic conservation is plausible but untested.

**Global normalisation.** $\hat{z} = z / \sum_k z_k$ is network-wide; biologically it needs inhibitory gain control (§7).

**No chance coincidences.** The raw causal count adds a chance term $\propto r_i r_j$ that does not depend on $W_{ij}$. The engine uses the **covariance count** $\kappa_{ij} = C_{ij} - c_i c_j (k-1)/k^2$, with $C_{ij}$ the lag-1 causal pairings and $c_i$ the spike counts over the $k$ steps of the frame [Sejnowski 1977; Kempter et al. 1999]. The subtracted term is a rank-1 outer product, one crossbar phase. On weak synapses the count is zero-mean noise that the $w_{min}$ floor rectifies, so the per-count rate is kept at about $5 \times 10^{-4}$ or less; weight-dependent soft bounds would remove the bias at its source [van Rossum et al. 2000] (**Proposal**).

### 2.3 One walk per frame

Each frame $f$ brings a stimulus $\mathbf{o}_f$, and $\pi_f$ solves (1) with $v_f = v(\mathbf{o}_f)$. It is implicit, $\pi_f = \alpha (I - (1 - \alpha) T)^{-1} v_f$, and fast: $n$ hops of $x_{k+1} = (1 - \alpha) T x_k + \alpha v_f$ from $x_0 = v_f$ give

$$
\lVert x_n - \pi_f \rVert_1 \le (1 - \alpha)^n \lVert v_f - \pi_f \rVert_1 \le 2 (1 - \alpha)^n \qquad \text{(1a)}
$$

independently of network size; $\alpha = 0.2$ and $n = 20$ give at most 2.3%. Modular networks mix slowly and approach the bound (0.78 per hop against 0.8), so budget for it. One hop is a sparse matrix–vector product in software and one analogue read on a crossbar. In the branching engine one simulation step is one hop, and the module-level error of spike shares is sampling-limited at about $1.3 / \sqrt{S_{mod}}$, with $S_{mod}$ the spikes per module per frame, so frame windows are stated in spikes, not time (iteration 001). In continuous time the network relaxes with time constant about $\tau / \alpha$, a frame of 100–250 ms. Three timescales must be ordered: hops, then frames, then plasticity ($\lambda, \eta$ small), which is the EM split of §8.

**Proposal: carry-over.** To let latent communities carry state across frames, mix the previous frame's latent flow into the teleportation vector,

$$
v_f = (1 - \rho) v(\mathbf{o}_f) + \rho \frac{P_{\mathcal{L}} \pi_{f-1}}{p_{\mathcal{L}}(f - 1)} \qquad \text{(1b)}
$$

with $P_{\mathcal{L}}$ the restriction to latent neurons; $\rho = 0$ gives independent frames. With $\pi_{f-1}$ held fixed every gradient below is unchanged. Linear carry-over is a contraction with no attractors, so it forgets: two histories converge by a factor of 0.29–0.39 per frame ($\rho = 0.5$–$0.9$). Holding a cue needs a module-level nonlinearity on the carried vector (a soft winner-take-all over latent modules, still crossbar-native), an eligibility time constant at least as long as the cue-to-reward delay, and controller→latent routing that the map term does not seal (§4.5).

### 2.4 Communities as states: the extracted Markov model

Coarse-graining the frame's flow to modules gives a $K$-state, stimulus-conditioned Markov chain,

$$
T^K_{ba}(\mathbf{o}_f) = \frac{J_{ba}}{p_a} = \sum_{j \in a} \frac{\pi_{f,j}}{p_a} \sum_{i \in b} T_{ij} \qquad \text{(1c)}
$$

Each neuron's split of output across modules is fixed by $W$; **the stimulus enters the module chain only by choosing which neurons in a module carry its flow.** Concentrating the stimulus in the controller makes the controller's column of $T^K$ the stimulus-dependent one (standard deviation across stimuli 0.0196 against 0.0006 when stimuli differ only in per-module drive).

**Proposal: world model.** With carry-over, $\pi_f = R v_f$ with $R = \alpha (I - (1 - \alpha) T)^{-1}$, so the latent mass carried out of module $a$ lands in latent module $b$ in proportion to the latent-$b$ share of $R u_a$ ($u_a$ the normalised restriction of $\pi_{f-1}$ to $a$). That defines an across-frame latent chain $P(b \mid a, \mathbf{o}_f)$, not yet checked in simulation. Biologically, metastable cluster switching takes hundreds of milliseconds [Litwin-Kumar & Doiron 2012; Mazzucato et al. 2015], a frame timescale, so this across-frame chain, not $T^K$ within a frame, is the one to compare with cortical state sequences.

### 2.5 Action readout

The action is read from the flow into the action modules,

$$
P(a \mid \mathbf{o}_f) = \frac{p_a(f)}{\sum_{a' \in \mathcal{A}} p_{a'}(f)}, \qquad a \in \mathcal{A} \qquad \text{(1d)}
$$

**Spiking readout.** The spiking analogue of (1d) is a **race**: the action is the module whose population spikes first. For independent Poisson populations with rates proportional to $p_a$ the race picks $a$ with probability exactly $p_a / \sum_{a'} p_{a'}$, but only once the walk has relaxed. The race therefore starts after a burn-in of $b = 10$ steps ($0.8^{10} \approx 0.11$), which brings its KL from (1d) to the sampling floor (iteration 002); biologically, a "hold your horses" subthalamic signal withholding the response [Frank 2006].

**Proposal: sharpened and adaptive policy.** (1d) has no temperature. The family

$$
P_\beta(a \mid \mathbf{o}_f) = \frac{p_a(f)^\beta}{\sum_{a' \in \mathcal{A}} p_{a'}(f)^\beta} \qquad \text{(1e)}
$$

has the matched score $h_b = \beta \left( \mathbb{I}[b = a^{\ast}] - \tilde{P}_\beta(b) \right) / J^{in}_b$ in (4b). A fixed $\beta$ is unstable (M2). The adopted Proposal (model v4) adapts $\beta$ to the trend of episode return: after episode $e$, with running averages $\bar{R}_{fast}$, $\bar{R}_{slow}$ over 20 and 100 episodes and $s_R$ the running deviation about $\bar{R}_{slow}$,

$$
\log \beta \leftarrow \log \beta + \kappa_\beta z_e, \qquad z_e = \mathrm{clip} \left( \frac{\bar{R}_{fast} - \bar{R}_{slow}}{s_R}, -2, 2 \right), \qquad \beta \in [1, \beta_{max}] \qquad \text{(1g)}
$$

with $\kappa_\beta = 0.01$, $\beta_{max} = 16$, and no update over the first 20 episodes. It is one global scalar and not a gradient. Because the matched score carries the factor $\beta$, it is also a global step-size gain on the task term. Biologically it abstracts the utility-driven gain of the adaptive gain theory of locus coeruleus–noradrenaline [Aston-Jones & Cohen 2005], with reward-range normalisation as in dopamine neurons [Tobler et al. 2005]. It needs episode boundaries, and a $\beta$ that falls after a collapse does not recover, since $z_e \approx 0$ on a new plateau.

**Proposal (spiking only): a race on count differences to a bound.** With $n_a(t)$ the action spikes of module $a$ counted from the burn-in, the action is

$$
a^{\ast} = a \quad \text{at the first } t \text{ with } n_a(t) - \max_{b \ne a} n_b(t) \ge \theta \qquad \text{(1h)}
$$

or the leader at the end of the frame. If counted spikes are independent draws with probabilities $p$, then for two actions the lead is a $\pm 1$ walk absorbed at $\pm \theta$, and by gambler's ruin [Feller 1968] $P(a^{\ast} = a) = p_a^\theta / (p_a^\theta + p_b^\theta)$: (1e) with $\beta = \theta$ exactly, at an expected $\theta^2$ spikes at a tie [Wald 1947]. For three actions it is close to (1e) (KL at most 0.0065). It is a discrete drift-diffusion model with mutual inhibition [Usher & McClelland 2001; Bogacz et al. 2006], so $\beta$ maps onto a decision threshold set by cortico-striatal and subthalamic circuits [Lo & Wang 2006; Bogacz et al. 2010; Cavanagh et al. 2011]. Action modules are nearly closed, so spikes arrive in bursts with overdispersion $F$, giving $\beta_{eff} \approx \theta / F$; the agent maps $\beta$ to $\theta$ through a measured table (iteration 010).

**Limits of the policy family.** $P(a \mid \mathbf{o})$ is a ratio of functions linear in $v$, a mixture of each controller neuron's routing. Recurrence inside the controller blurs the stimulus, and its credit depends on each postsynaptic neuron's own routing, which the module-level score of (4b) cannot represent; on trained weights controller→controller synapses carried 15–78% of the true score with cosine 0.03–0.45 on that block. So the controller is built without recurrence ($p_{cc} = 0$, a feed-forward encoder; iteration 006). For the same reason the controller uses **conjunctive** receptive fields: a policy that averages per-dimension votes reached 84 steps on CartPole, one vote per conjunctive cell 491 (iteration 001).

## 3. The map equation

Describe the walk with a two-level code: inside a module, name the next neuron or "exit" with a module-local codeword; on exit, name the next module from a global index. $L(M)$ is the average number of bits per step [Rosvall & Bergstrom 2008]:

$$
L(M) = q_{\curvearrowright} \log q_{\curvearrowright} - 2 \sum_m q_m \log q_m - \sum_j \pi_j \log \pi_j + \sum_m (p_m + q_m) \log (p_m + q_m) \qquad \text{(3)}
$$

With $\pi$ held fixed, $W$ enters $L$ only through the exit rates, and

$$
M^{\ast}_m := \frac{\partial L}{\partial q_m} = \log \frac{q_{\curvearrowright} (p_m + q_m)}{q_m^2} \ge 0
$$

because $q_{\curvearrowright} \ge q_m$ and $p_m + q_m \ge q_m$. **Leakage always costs bits**, and $M^{\ast}_m$ is the marginal number of bits per unit of exit flow from module $m$. It diverges as $q_m \to 0$, so the implementation clips it.

**The inverse problem.** Community detection solves $\min_M L(M; W)$. Map-STDP pins the controller and action modules and minimises over the weights, averaged over the frames the environment produces:

$$
\min_W D(W), \qquad D(W) = \mathbb{E}_f \left[ L(M; \pi_f) \right] \qquad \text{(3a)}
$$

Latent modules may be pinned or re-detected from time to time (§8), which makes training an alternating minimisation. With each $\pi_f$ held fixed, $D$ is an average of per-frame description lengths, each depending on $W$ only through that frame's link flows, so every result of §4 holds frame by frame and for $D$ by averaging. The modulator can use each frame's module statistics or running averages across frames; the difference was negligible in a test (Appendix A).

## 4. Plasticity as a three-factor rule

### 4.1 The rule

With $\kappa_{ij}$ the pairing count of §2.2, the structural update is

$$
\Delta W_{ij} = -\lambda \left( g_{ij} - \bar{g}_j \right) \kappa_{ij}, \qquad g_{ij} = \frac{\partial D}{\partial J_{ij}}, \qquad \bar{g}_j = \sum_k T_{kj} g_{kj} \qquad \text{(4)}
$$

Pre and post enter through the coincidence $\kappa_{ij}$, and the third factor is $g_{ij} - \bar{g}_j$. For hardware, $g_{ij}$ depends only on the module pair $(m(i), m(j))$: a $K \times K$ table computed from module statistics and broadcast per module. The baseline $\bar{g}_j$ is a per-neuron running mean of the modulator over $j$'s own transitions. On a crossbar it is a per-row register; biologically it is an **abstraction** (as are $\bar{h}_j$ in (4b) and $\bar{g}^{eff}_j$ in §5): it averages over $j$'s axonal targets, so it needs a presynaptic or retrograde readout, which is untested. Description costs are available at the moment of the transition, so the structural term needs no trace; a biological modulator lags by 0.3–2 s [Yagishita et al. 2014] and would need one.

### 4.2 What the rule descends

If $D$ depends on $W$ only through $J$, with $\pi$ held fixed, then (Lemma)

$$
\frac{\partial D}{\partial W_{ij}} = \frac{\pi_j}{d_j} \left( g_{ij} - \bar{g}_j \right) \qquad \text{(5)}
$$

Pairings occur at rate $J_{ij} = \pi_j W_{ij} / d_j$, so averaging (4) gives (Theorem 3)

$$
\mathbb{E} \left[ \Delta W_{ij} \right] = -\lambda W_{ij} \frac{\partial D}{\partial W_{ij}}
$$

gradient descent with each coordinate scaled by its own positive weight: strong synapses move more, silent synapses stay silent. The baseline matters. Without it, an extra term $-\lambda J_{ij} \bar{g}_j$ acts as cost-weighted decay, not descent. To first order in $\lambda$, the baselined rule equals the unbaselined rule followed by rescaling each neuron's outputs back to the same $d_j$, so $\bar{g}_j$ reads as a presynaptic resource constraint (an abstraction; the documented weight conservation is on the input side, §2.2). Row rescaling is not crossbar-native, so on hardware the baseline stays explicit.

### 4.3 Marginal cost, not pointwise cost

The *pointwise* cost of a step (its codeword length) averages to $L$; the *marginal* cost $g_{ij} = \partial D / \partial J_{ij}$ is how much $D$ grows with more flow along that transition. The modulator must broadcast the marginal cost. Broadcasting the codeword length also rewards landing on high-traffic neurons with cheap codewords, and in simulation it correlated 0.976 with the descent direction, against 0.9999 for the marginal cost.

### 4.4 The map equation

With $\pi$ fixed, $L$ depends on $J$ only through $q_m = \sum_{j \in m} \sum_{i \notin m} J_{ij}$, so

$$
g_{ij} = M^{\ast}_{m(j)} \mathbb{I}[i \notin m(j)], \qquad \bar{g}_j = M^{\ast}_{m(j)} \bar{e}_j \qquad \text{(6)}
$$

**The modulator of module $m$ is silent while activity stays inside and fires $M^{\ast}_m$ when it leaves.** By (5), the exact gradient is

$$
\frac{\partial L}{\partial W_{ij}} = M^{\ast}_{m(j)} \cdot \frac{\pi_j}{d_j} \cdot G(i, j), \qquad G(i, j) = \mathbb{I}[i \notin m(j)] - \bar{e}_j \qquad \text{(7)}
$$

$G$ sets the direction (cross-module synapses weakened, within-module strengthened), $\bar{e}_j$ how hard (leaky neurons are corrected most), and $M^{\ast}_m$ how costly the module's leakage is. Since $\sum_i W_{ij} G(i, j) = 0$, the multiplicative update conserves each $d_j$ and only reallocates output from cross-module to within-module targets. The additive rule $\Delta W_{ij} = \eta \mathrm{STDP}(i, j) - \eta' G(i, j)$ is kept as the exact-gradient simulation baseline, not as the hardware rule. $G$ is exactly the centred modulator $g - \bar{g}$ of lag-1 Markov stability, whose gradient is $(\pi_j / d_j) G(i, j)$ [Delvenne et al. 2010]; the map equation reweights it per module by $M^{\ast}_m$.

### 4.5 Routing exemption and latent output

The policy is carried by cross-module flow (controller→action and controller→latent), and the map term pushes every module to keep its flow inside. Applied to those pairs it drives the controller's exit flow to zero: the action modules starve, $M^{\ast}_{\mathcal{C}}$ diverges, pairings on routing synapses vanish, and the task term cannot learn (flow CartPole return 53 against 343 with protection; iteration 001). The implemented choice therefore **exempts** the routing pairs, $g_{ba} = 0$ for $a = \mathcal{C}$, $b \in \mathcal{A} \cup \mathcal{L}$. It has no free parameter, and the controller drops out of the structural term as an input layer. With $M^{\ast}$ computed from the full exit rates it is not an exact gradient (cosine 0.985); it is exact if $M^{\ast}$ uses exit rates without the exempt pairs. With routing exempt, the map term raises return in both backends (iteration 002) and seals latent and action modules almost completely, which is the map equation's optimum by construction (§3).

A dual routing term, kept only as an ablation in `experiments/m1.py`, minimises $L - \mu (J_{route} - q^{\ast})$ with $J_{route} = \sum_{b \in \mathcal{A} \cup \mathcal{L}} J_{b \mathcal{C}}$:

$$
g_{ba} = M^{\ast}_a \mathbb{I}[b \ne a] - \mu \mathbb{I}[a = \mathcal{C}, b \in \mathcal{A} \cup \mathcal{L}], \qquad \mu \leftarrow \max \left( 0, \mu + \eta_\mu \left( q^{\ast} - J_{route} \right) \right) \qquad \text{(4a)}
$$

Once latents and actions seal, the target $q^{\ast}$ is infeasible and $\mu$ diverges, so it acts as routing potentiation rather than a constraint (iteration 003).

**Latent modules become sinks.** The task term removes latent→action flow, because latent flow does not depend on the stimulus and only dilutes (1d); the map term seals latents against other latents and the controller, which is where its benefit comes from (iteration 003).

**Proposal: latent-output barrier.** To give each latent module a path to the actions,

$$
D_{LA} = D - \beta_{LA} \sum_{\ell \in \mathcal{L}} \mathbb{E}_f \left[ p_\ell \ln J_{\mathcal{A} \ell} \right], \qquad g_{ba} = M^{\ast}_a \mathbb{I}[b \ne a] - \beta_{LA} \frac{p_a}{J_{\mathcal{A} a}} \mathbb{I}[a \in \mathcal{L}, b \in \mathcal{A}] \qquad \text{(4e)}
$$

with $J_{\mathcal{A} \ell} = \sum_{b \in \mathcal{A}} J_{b \ell}$. It is still a $K \times K$ table, and (5) and Theorem 3 apply to $D_{LA}$. Its marginal cost grows like $1 / J_{\mathcal{A} \ell}$ while $M^{\ast}$ grows only like $\log (1 / q)$, so an interior fixed point exists, and the growing modulator cancels the vanishing pairing rate, so it regrows sealed latent→action synapses. On fully observed tasks latent→action flow carries no stimulus information and (4e) trades a little return for latent output; it is meant for M4.

### 4.6 The task term

Plain reward-modulated STDP, $\eta (R - \bar{R}) e_{ij}$ [Frémaux et al. 2010], gets no action credit when the action is sampled from flow shares independently of the spikes that build $e_{ij}$, and it stalls under sparse reward; spiking actor-critics use a TD error instead [Frémaux et al. 2013].

**Action-gated eligibility.** With $\pi_f$ fixed, take the surrogate policy $\tilde{P}(a \mid \mathbf{o}) = J^{in}_a / \sum_{a' \in \mathcal{A}} J^{in}_{a'}$, where $J^{in}_b = \sum_{i \in b} \sum_j J_{ij}$. Its score with respect to $J_{ij}$ depends only on the module of $i$:

$$
h_b = \frac{\mathbb{I}[b = a^{\ast}]}{J^{in}_{a^{\ast}}} - \frac{\mathbb{I}[b \in \mathcal{A}]}{\sum_{a' \in \mathcal{A}} J^{in}_{a'}}, \qquad e_{ij} \leftarrow \left( 1 - \frac{1}{\tau_e} \right) e_{ij} + \kappa_{ij} \left( h_{m(i)} - \bar{h}_j \right), \qquad \bar{h}_j = \sum_k T_{kj} h_{m(k)} \qquad \text{(4b)}
$$

By the Lemma the expected trace increment is $W_{ij} \partial \log \tilde{P}(a^{\ast} \mid \mathbf{o}) / \partial W_{ij}$, a weight-scaled policy score of the same form as (4). It is the one-hop surrogate of the exact score through the resolvent, (K8) with $G = 1$: cosine 0.98–0.998 on untrained weights, 0.83–0.99 on trained weights without controller recurrence (Appendix A). $\tau_e = 3$ frames is 0.3–0.75 s at 100–250 ms per frame, inside the dopamine-gated window [Yagishita et al. 2014]. The chosen action must be broadcast to the action modules; in basal-ganglia terms the selected channel is tagged by its own activity when dopamine arrives [Gurney et al. 2015], which is plausible but untested here. Like $\bar{g}_j$, the baseline $\bar{h}_j$ is an abstraction (§4.1).

**TD error.** The task update is $\Delta W_{ij} = \eta \delta_f e_{ij}$, with a global TD error from a linear critic on the controller's flow shares, trained by normalised LMS:

$$
\delta_f = r_f + \gamma V_{f+1} - V_f, \qquad V_f = u \cdot x_f, \qquad x_f = \frac{\pi_{\mathcal{C}}(f)}{p_{\mathcal{C}}(f)}, \qquad u \leftarrow u + \eta_V \delta_f \frac{x_f}{\lVert x_f \rVert^2} \qquad \text{(4c)}
$$

$V_{f+1} = 0$ on termination, and the critic bootstraps on truncation. **Proposal (model v4): TD normalisation.** The task term uses $\delta'_f = \sigma_\delta \delta_f / \sqrt{\overline{\delta^2}}$, with $\overline{\delta^2}$ a running mean square over 1000 frames and $\sigma_\delta = 0.04$, so the task update has the same size on every task; the critic still learns from the raw $\delta_f$. The critic weights $u$ live in the periphery and $\delta$ is a global scalar, so the term stays crossbar-native.

**Proposal: a separate critic.** The critic reads its own fixed layer of nonnegative sensory receptive fields $y_f = \phi(\mathbf{o}_f)$ (a Gaussian grid, or a Fourier basis with ON/OFF rates):

$$
V_f = u \cdot y_f, \qquad u \leftarrow u + \eta_V \delta_f \frac{y_f}{\lVert y_f \rVert^2} \qquad \text{(4c'')}
$$

(The label is kept from an earlier version that also had a (4c'), now removed; code cites (4c'').)

The communities remain the actor's states; the value function's representation is designer-given, like a ventral-striatal critic on shared input [O'Doherty et al. 2004; Lansink et al. 2009; Foster et al. 2000]. On MountainCar the critic's resolution is what limits learning (iteration 012).

**Proposal: an entropy term on the action modules.** With $\tilde{P}(b)$ the frame's $\beta = 1$ action shares, a module-level modulator acts through the same coincidence and baseline as (4b), every frame and not gated by $\delta$:

$$
\Delta W_{ij} = \eta c_e \kappa_{ij} \left( \frac{g^H_{m(i)}}{J^{in}_{m(i)}} - \bar{g}^H_j \right), \qquad g^H_b = -\tilde{P}(b) \left( \log \tilde{P}(b) - \sum_{a \in \mathcal{A}} \tilde{P}(a) \log \tilde{P}(a) \right) \qquad \text{(4h)}
$$

on action modules ($g^H = 0$ elsewhere). With $\pi$ fixed and $\kappa = J$ it averages to $\eta c_e W \odot \nabla H(\tilde{P})$ [cf. Chung & Kozma 2020]. Optionally $c_e$ tracks a target entropy, $\log c_e \leftarrow \log c_e + k_H ( \phi_H \log \lvert \mathcal{A} \rvert - H(\tilde{P}) )$ per frame, with $\phi_H$ the target fraction of the maximum entropy [Haarnoja et al. 2018]. Logs in (4h) and here are natural (nats). It is useful only far below the task term's size, as a decaying exploration bonus (iteration 012). It abstracts usage homeostasis of action channels, not a neuromodulator; exploration-related noradrenaline acts on choice stochasticity, which $\beta$ already models [Tervo et al. 2014].

## 5. Proposal: the rule on the linear-response walk

This section restates §4 for the general walk of §2.1. None of it is implemented yet; its closed-loop test is pending. Hold $\pi = r / \sum r$ and $G$ fixed (the EM split), and restrict to the excitatory block, where $B_{ij} = \chi T_{ij}$, with $\chi$ the recurrent output gain times the membrane time constant. Define

$$
S_j = \sum_l G_l W_{lj}, \qquad T^{eff}_{ij} = \frac{G_i W_{ij}}{S_j}, \qquad \nu_j = \frac{\chi S_j}{d_j}, \qquad J^{eff}_{ij} = \pi_j T^{eff}_{ij}, \qquad J^\Lambda_{ij} = \nu_j J^{eff}_{ij} \qquad \text{(K5)}
$$

The module tables $p_a$, $J_{ba}$, $q_a$ and $L$ are those of §3 with $J^{eff}$ in place of $J$. Plasticity now shapes the gain-weighted walk: a synapse onto a low-gain neuron carries little coded flow however large $W_{ij}$ is. With $G = 1$ and $\chi = 1 - \alpha$, (K5) reduces to $T^{eff} = T$, $\nu_j = 1 - \alpha$, and everything below reduces to §4.

**Lemma K.** If $D$ depends on $W$ only through $J^{eff}$, then with $g_{ij} = \partial D / \partial J^{eff}_{ij}$

$$
\frac{\partial D}{\partial W_{ij}} = \frac{\pi_j G_i}{S_j} \left( g_{ij} - \bar{g}^{eff}_j \right), \qquad \bar{g}^{eff}_j = \sum_k T^{eff}_{kj} g_{kj} \qquad \text{(K6)}
$$

**Theorem K.** If $\mathbb{E}[\kappa_{ij}] = \zeta J^\Lambda_{ij}$ for a constant $\zeta > 0$, the rule $\Delta W_{ij} = -\lambda (g_{ij} - \bar{g}^{eff}_j) \kappa_{ij}$ averages to

$$
\mathbb{E} \left[ \Delta W_{ij} \right] = -\lambda \zeta \nu_j W_{ij} \frac{\partial D}{\partial W_{ij}}, \qquad \sum_i \mathbb{E} \left[ \Delta W_{ij} \right] = 0 \qquad \text{(K7)}
$$

So the rule is descent preconditioned by the presynaptic offspring mean $\nu_j > 0$, which never has to be computed: it is implicit in $\kappa$. The correct baseline is the **pairing-weighted** mean $\bar{g}^{eff}_j$, the average of the modulator over $j$'s actual transmissions, which is exactly what a per-row running mean over $j$'s pairings estimates (on a crossbar; biologically an abstraction, §4.1). The weight-normalised $\sum_k T_{kj} g_{kj}$ of (4) equals it only when $G$ is uniform.

**Policy score.** Let $s_i = \partial \log P / \partial r_i$, i.e. $\mathbb{I}[m(i) = a^{\ast}] / r_{a^{\ast}} - \mathbb{I}[m(i) \in \mathcal{A}] / r_{\mathcal{A}}$ with $r_b = \sum_{i \in b} r_i$. Differentiating (K1) implicitly gives the exact score through the fixed point,

$$
W_{ij} \frac{\partial \log P}{\partial W_{ij}} = \chi T_{ij} r_j \left( \psi_i - \bar{\psi}_j \right), \qquad \psi = G \odot \Phi, \qquad \Phi = (I - \Lambda)^{-\top} s, \qquad \bar{\psi}_j = \sum_k T_{kj} \psi_k \qquad \text{(K8)}
$$

$\Phi_i$ is the **cascade value** of a spike in $i$: $s$ summed over all of its linear-response descendants, inhibitory pools included. Since $\Lambda_{ij} r_j = \chi G_i T_{ij} r_j$,

$$
\chi T_{ij} r_j \left( \psi_i - \bar{\psi}_j \right) = \Lambda_{ij} r_j \Phi_i - W_{ij} \frac{\chi r_j \bar{\psi}_j}{d_j} \qquad \text{(K9)}
$$

The second term rescales column $j$ and changes neither $T$ nor $P$ to first order. Hence $e_{ij} \mathrel{+}= \kappa_{ij} \Phi_i$, **with no presynaptic baseline**, is the exact score modulo column rescaling when $\mathbb{E}[\kappa] \propto \Lambda \mathrm{diag}(r)$. Any baseline term $\kappa_{ij} b_j$ leaves a residual $\propto T_{ij} r_j b_j (G_i - \bar{G}_j)$ once $G$ varies, which dominates on trained weights. The crossbar-native version replaces $\Phi_i$ by the module-level $\Phi^K_{m(i)}$, an approximation (cosine 0.83–0.97 with the exact score): $\Phi^K = (I - \Lambda^{pop \top})^{-1} s^K$, with the population kernel table $\Lambda^{pop}_{ba} = \mathrm{mean}_{j \in a} \sum_{i \in b} \Lambda_{ij}$ over excitatory modules *and* inhibitory pools, is a $K$-vector; the pools must be included, while further hops add little over one. In the branching case $\Phi = (I - (1 - \alpha) T)^{-\top} s$, and (4b) is its one-hop surrogate with a baseline that is harmless because $G$ is uniform.

$\Phi^K$ is not a plausible neuromodulator payload: it needs an adjoint solve over an operating-point-dependent kernel and a signed per-module signal every frame. The closest biological analogues are heterogeneous, target-specific dopamine [Engelhard et al. 2019; Hamid et al. 2021] and a selected-channel tag times a global $\delta$ [Gurney et al. 2015]. Treat it as a peripheral (crossbar) computation. Its indexing by the *postsynaptic* module is itself a point in its favour, since volume transmission reaches synapses by their postsynaptic site (§6).

**Assumptions and open conditions.**

1. **Pairing count.** Theorem K and (K9) need $\mathbb{E}[\kappa] \propto J^\Lambda$, which by (K4) needs $\rho(\Lambda_{EE}) < 1$, a pairing window much longer than the response kernel, kicks small against the excitability spread, and Poisson-like neurons. No tested QIF operating point meets them. At `qif.MPR_POINT` the excitatory kernel is supercritical, $\rho(\Lambda_{EE}) = 1.1$–$1.7$, and per-module inhibition stabilises the network, an inhibition-stabilised network [Tsodyks et al. 1997; Ozeki et al. 2009]; the measured count does not track $\Lambda \mathrm{diag}(r)$, and (K7) with it has cosine 0.19–0.61 with its target.
2. **EM split.** Holding $\pi$ and $G$ fixed keeps cosine 0.25–0.44 of the full map-equation gradient on QIF networks, against 0.55–0.92 for the flow model on the same networks. A rule that also credits $d\pi / dW$ and $dG / dW$ is not derived; $\partial L / \partial \pi$ is not a $K \times K$ table.
3. **Homeostasis.** The average of (4d) becomes $\propto G_i W_{ij}$, no longer a pure column rescaling, so it biases $T^{eff}$ toward low-gain targets when $d_j > 1$.
4. **Not proved:** that $\pi$ and $T^{eff}$ form a consistent code when $\mu$ has negative entries.

## 6. Locality and hardware mapping (memristive crossbars)

Target: $W$ as conductances in a crossbar (rows presynaptic, columns postsynaptic), with neurons and modulators in peripheral circuits.

| Operation | Crossbar implementation |
| --- | --- |
| Per-frame walk (1a), E-step (2) | $n$ analogue matrix–vector reads, with $\alpha v_f$ added in the periphery |
| $d_j$, $e_j$, module flows | $K + 1$ masked reads, summing current per row |
| Coincidence $\kappa_{ij}$ | overlapping pre/post programming pulses; the covariance correction is one outer-product phase |
| Modulator $g_{ba}$, task vector $h_b$, cascade value $\Phi^K$ | peripheral logic over $K \times K$ module statistics, broadcast per module |
| Baselines $\bar{g}_j$, $\bar{h}_j$, $\bar{g}^{eff}_j$ | per-row registers (running means over $j$'s pairings) |
| Applying (4), (4b), (4d) | row-pulse amplitude scaled by the per-row factor, one phase per target-module column mask; the map equation needs two per module (exit and home columns) |
| Task eligibility $e_{ij}$ | the only per-synapse state; e.g. volatile diffusive memristors [Wang et al. 2017] |
| TD error $\delta$, critic $u$ | a global scalar and a peripheral vector (4c) |

No step needs a nonlinear function of a single device's conductance or the transposed element $W_{ji}$; this is why every modulator is a module-pair table. Two Proposal variants fall outside the current rules: the exact, $d_j$-conserving score $\kappa_{ij} (\Phi_i - \bar{\psi}_j / G_i)$ needs a per-postsynaptic $1/G_i$ register, and its equivalent form subtracts $W_{ij}$ times a row register, a weight-proportional decay phase the rules do not allow. Both are flagged, not adopted.

**Biological locality.** A synapse $j \to i$ sits on $i$'s dendrite, so a modulator released in a module reaches synapses by their *postsynaptic* module. The map modulator is indexed by the *presynaptic* module, so biologically it needs $M^{\ast}$ roughly uniform across modules or a projection-specific signal; dendritic segregation of local and long-range inputs [Petreanu et al. 2009] could supply the bit $\mathbb{I}[a \ne b]$ but not $a$, and a presynaptic-locus factor is the natural carrier of a presynaptic-module index. The backward, dendritically normalised walk ($W_{ij} / \sum_k W_{ik}$) would have postsynaptic sums, but it loses the tie of flow to firing that the forward walk has, so the forward convention is kept. Neuromodulators are slow (0.3–2 s [Yagishita et al. 2014]) and at most regionally specific [Engelhard et al. 2019; Hamid et al. 2021], so a per-region scalar is plausible and a full $K \times K$ table is not. Astrocytes gate plasticity locally [Henneberger et al. 2010] and tile exclusive domains [Bushong et al. 2002], but a domain covers about $10^5$ synapses, not a module pair.

## 7. Inhibition

The branching engine is excitatory only, with normalisation done in software (§2.2). The QIF engine adds fixed inhibitory (PV-like) pools, one per module, with E→I→E connections, which keeps Dale's law and supplies the gain control the mean-field assumption needs. With inhibition, $\Lambda$ is signed and is no longer a branching kernel. **Proposal: code the E→E walk only**: $T^{eff}$ is built from $\Lambda_{EE}$ as in (K5), and inhibition enters the map term only through $G$ and $\pi$. With per-module pools this is exact for the map term, because the inhibitory correction $\Lambda_{EI} (I - \Lambda_{II})^{-1} \Lambda_{IE}$ is zero on every cross-module pair and the map gradient sees only cross-module flow. Per-module pools with no cross-module E→I are an architectural choice: in cortex long-range inputs recruit feed-forward PV inhibition [Gabernet et al. 2005], which would make the cross-module correction nonzero. It is a coding device, not a branching process ($\nu_j > 1$ for many neurons). The policy score, by contrast, keeps the pools inside $\Phi$ (§5). With global or action-level pools the E→E walk is not exact across modules and must be measured before use. Plastic inhibition, a bounded homeostatic inhibitory STDP that sets each excitatory rate to a target [Vogels et al. 2011], is a further **Proposal**.

## 8. Training

Per environment frame $f$:

1. **Observe.** Form $v_f$ from $\mathbf{o}_f$, concentrated on the controller; with carry-over, mix in the previous latent flow (1b).
2. **Walk.** Run the $n$ hops of (1a) (flow), or the $k = 100$-step spiking frame from a cold start and take spike shares as $\hat{\pi}_f$ (spiking).
3. **Act.** Sample from (1d) or (1e), or run the race (or (1h)) after the burn-in.
4. **Module statistics.** Compute $p_a$, $J_{ba}$ and $q_a$, the modulator table $g_{ba}$ (with the exemption, and (4e) if used), and the per-row baselines $\bar{g}_j$, $\bar{h}_j$.
5. **Plasticity.** Add (4b) to the traces; apply (4) and (4d) to the frame's pairings (in spiking, the covariance count, with $\lambda$ divided by the frame's spike count); after the next frame's walk, compute the TD error (4c), update the critic, and apply $\eta_e \delta_f e_{ij}$.
6. **Carry over.** Keep $P_{\mathcal{L}} \pi_f$ for the next frame.

**Stability.** $M^{\ast}$ is clipped (at 20). The task rate is annealed, $\eta_e = \eta / (1 + e / 300)$ over episodes $e$: a constant rate stalls in the closed loop (iteration 002). It is a global scalar, crossbar-native. $\lambda$ is constant from frame 0. Multiplicative updates freeze near-zero synapses, so pruning and creating synapses (rewiring) is the route to new structure.

**EM split.** $\pi_f$ and the module statistics are fixed within each update. That needs the walk to converge within a frame and plasticity to be slow relative to frames. For the flow model the fixed-flow gradient keeps cosine 0.55–0.92 of the full gradient (Appendix A).

**Periodically.** Latent modules may be re-detected with Infomap or Leiden, the forward step of the alternating minimisation (3a); controller and action modules stay pinned. Initialisation is a stochastic block model with a recurrence-free controller.

## 9. Assumptions and limitations

1. **EM split and mean field.** $\pi$ (in §5 also $G$) is held fixed during updates, and pairings are assumed proportional to $J$. Both hold well in the branching engine and do not yet hold for QIF (§5). (4) is exact only in expectation.
2. **Coding choices.** Teleportation and escape are not counted as exits, and $M^{\ast}_m$ is clipped for sealed modules; the routing exemption is not an exact gradient (§4.5).
3. **Policy.** (4b) is a one-hop surrogate score, so the controller is recurrence-free; (1g) needs episode boundaries and does not recover after a collapse.
4. **Memory.** Linear carry-over forgets geometrically (§2.3); $\rho$ and $\lvert \mathcal{L} \rvert$ are hyperparameters unless latents are re-detected.

---

## Appendix A: Proofs and checks

**Theorem 1 (flow exists, is unique and is reachable).** If $T$ is column-stochastic and $\alpha \in (0, 1]$, (1) has one solution in the simplex, and power iteration with $F(x) = (1 - \alpha) T x + \alpha v$ converges to it geometrically. *Proof.* $F$ maps the simplex into itself, and $\lVert F(x) - F(y) \rVert_1 = (1 - \alpha) \lVert T (x - y) \rVert_1 \le (1 - \alpha) \lVert x - y \rVert_1$ since $\lVert T \rVert_1 = 1$. Banach's fixed-point theorem gives (1a). $\square$

**Lemma (chain rule through flows).** If $D$ depends on $W$ only through $J_{kj} = \pi_j W_{kj} / d_j$, with $\pi$ fixed, then (5) holds. *Proof.* $\partial J_{kj} / \partial W_{ij} = \pi_j (\delta_{ki} - T_{kj}) / d_j$, and only column $j$ depends on $W_{ij}$, so $\partial D / \partial W_{ij} = \sum_k g_{kj} \pi_j (\delta_{ki} - T_{kj}) / d_j = \pi_j (g_{ij} - \bar{g}_j) / d_j$. $\square$

**Theorem 2 (map-equation gradient).** With $\pi$ fixed, (7) holds. *Proof.* The $p_m$ and $\pi_j \log \pi_j$ terms of (3) are constant, and differentiating (3) in $q_m$, including through $q_{\curvearrowright}$, gives $(\log q_{\curvearrowright} + \log e) - 2 (\log q_m + \log e) + (\log (p_m + q_m) + \log e) = M^{\ast}_m$ (logs base 2; the $\log e$ terms cancel). Since $q_m = \sum_{j \in m} \sum_{i \notin m} J_{ij}$, (6) follows; apply the Lemma. $\square$

**Theorem 3 (the three-factor rule descends $D$).** If pairings $j \to i$ occur at a rate proportional to $J_{ij}$, then $\mathbb{E}[\Delta W_{ij}] = -\lambda J_{ij} (g_{ij} - \bar{g}_j) = -\lambda W_{ij} \frac{\pi_j}{d_j} (g_{ij} - \bar{g}_j) = -\lambda W_{ij} \partial D / \partial W_{ij}$ by the Lemma. $\square$

**Lemma K and Theorem K.** *Proof.* From (K5), $\partial T^{eff}_{kj} / \partial W_{ij} = G_i (\delta_{ki} - T^{eff}_{kj}) / S_j$, and only column $j$ depends on $W_{ij}$, which gives (K6). Then $J^\Lambda_{ij} = \nu_j W_{ij} \pi_j G_i / S_j$, so $-\lambda \zeta J^\Lambda_{ij} (g_{ij} - \bar{g}^{eff}_j) = -\lambda \zeta \nu_j W_{ij} \partial D / \partial W_{ij}$, and conservation follows from $\sum_i T^{eff}_{ij} (g_{ij} - \bar{g}^{eff}_j) = 0$. $\square$

**Score (K8, K9).** *Proof.* Differentiating (K1) gives $(I - \Lambda) dr = G \odot (dB r)$, so $d \log P = s^{\top} dr = \psi^{\top} (dB r)$. With $B_{kl} = \chi W_{kl} / d_l$, $\partial B_{kl} / \partial W_{ij} = \chi \delta_{lj} (\delta_{ki} - T_{kj}) / d_j$, which gives (K8) after multiplying by $W_{ij}$. Substituting $\chi G_i T_{ij} = \Lambda_{ij}$ gives (K9). $\square$

**Bounded race (1h), two actions.** Each counted spike moves the lead by $\pm 1$ with probabilities $p_a, p_b$; absorption at $+\theta$ before $-\theta$ has probability $1 / (1 + (p_b / p_a)^\theta)$ (gambler's ruin). With bursts of size $B$ per independent draw, the walk moves in steps of $B$, giving $\beta_{eff} \approx \theta \mathbb{E}[B] / \mathbb{E}[B^2]$ to leading order. $\square$

**Numerical checks.** Exact identities are checked by finite differences (FD); three-factor rules by Monte Carlo (MC) of sampled pairings against $-W \odot \nabla D$. Rows marked (s) come from scratch scripts and are preliminary; the rest are repeated by the `mapstdp` self-checks or the iteration records.

| Check | Network | Metric | Value |
| --- | --- | --- | --- |
| (1a) walk; (1c) identity; (1b) forgetting | 15 neurons, 5 modules, $\alpha = 0.2$, 8 stimuli | worst per-hop contraction (bound 0.8); (1c) error; history contraction per frame at $\rho = 0.5$ / 0.9 | 0.78 modular, 0.28 dense; machine precision; 0.29 / 0.39 |
| controller column of $T^K$ | same | std across stimuli: controller / other columns / per-module drive only | 0.0196 / $\le 0.0035$ / 0.0006 |
| (7) and (3a) gradient | 12 and 15 neurons, $\alpha$ 0.15–0.2 | max FD error | $\le 4 \times 10^{-7}$; $\le 10^{-9}$ per frame |
| (4) with (6); pointwise codeword instead | 15 neurons, 4000 frames × 200 pairings; 12 neurons | MC correlation | 0.99996 per-frame, 0.99993 pooled modulator; codeword 0.976 |
| baseline as renormalisation | 13 neurons | relative difference | $0.025 \lambda$ (first order) |
| pairing counts (branching); (4d) (s) | 80 neurons, 5000 / 2000 spiking frames | MC cosine: covariance / causal / balanced; (4d) with its expectation | 0.9916 / 0.9787 / 0.8443; 0.992 |
| (4a); (4e); routing exemption | 13–16 neurons, 6 stimuli; 80-neuron self-check | FD error; (4e) MC correlation and full-resolvent cosine ($\beta = 0.3$); exemption cosine with exact gradient | $\le 9 \times 10^{-10}$; 0.9999, 0.934; 0.985 |
| (4b) surrogate | 13 neurons; M3 networks | FD error; cosine with exact score | $\le 3 \times 10^{-10}$; untrained 0.981–0.998, trained $p_{cc} = 0.1$ 0.55–0.85, trained $p_{cc} = 0$ 0.83–0.99 |
| controller→controller credit (s) | trained M3, $p_{cc} = 0.1$ | share of exact score; block cosine of (4b) | 0.15–0.78; 0.03–0.45 |
| (1e) score; (1h) | 80 neurons, $\beta = 3$; exact absorbing chain and MC | FD error; $\lvert \mathcal{A} \rvert = 2$ identity, expected count, $\lvert \mathcal{A} \rvert = 3$ KL from (1e) | $\le 10^{-6}$; $\le 2 \times 10^{-15}$, $\le 10^{-12}$, $\le 0.0065$ |
| (4h) | 18 neurons, 3 seeds; repo SBM | FD error; MC correlation; full-resolvent cosine | $\le 2.1 \times 10^{-10}$; $\ge 0.9997$; 0.93 |
| branching = (K3) with $\Lambda = 0.8 T$ (s) | CartPole, 400 frames × 16 heads | offspring on $0.8 T$: slope, correlation; formulas vs `core` | 1.04, 0.94; $\le 9 \times 10^{-16}$ |
| (K2), (K6), (K8) (s) | QIF at `MPR_POINT` (CartPole, LunarLander, random 36); independent 12-neuron softplus network | FD error | (K2) 0.1–0.4%; (K6) $\le 1.8 \times 10^{-8}$; (K8) $\le 0.9$% (tabulated transfer); softplus (K6) $9 \times 10^{-10}$, (K8) $7 \times 10^{-9}$ relative |
| (K7) (s) | same; 4000 frames × 200 pairings $\propto J^\Lambda$; measured QIF $\kappa$, 3 stimuli × 400 frames × 32 heads | MC cosine with $-\nu_j W \odot \nabla D$ / with $-W \odot \nabla D$ / old baseline; measured $\kappa$ | 0.998–0.9997 / 0.88–0.97 / 0.86–0.99; 0.19–0.61 |
| EM split, map equation (s) | random 36, CartPole init, CartPole trained, LunarLander init | cosine of fixed-rate with full gradient: flow / QIF | 0.85, 0.92, 0.55, 0.92 / 0.44, 0.39, 0.25, 0.44 |
| (K9) score candidates (s) | CartPole, LunarLander, init and trained | cosine with (K8), column rescaling removed | $\Lambda r \Phi_i$ 1.000; module $\Phi^K$ with pools 0.83–0.97; without pools −0.36–0.93; (4b) on rates 0.07–0.82 |

## References

- Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. *Annual Review of Neuroscience*, 28, 403–450.
- Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. *Psychological Review*, 113(4), 700–765.
- Bogacz, R., Wagenmakers, E.-J., Forstmann, B. U., & Nieuwenhuis, S. (2010). The neural basis of the speed-accuracy tradeoff. *Trends in Neurosciences*, 33(1), 10–16.
- Bushong, E. A., Martone, M. E., Jones, Y. Z., & Ellisman, M. H. (2002). Protoplasmic astrocytes in CA1 stratum radiatum occupy separate anatomical domains. *Journal of Neuroscience*, 22(1), 183–192.
- Cavanagh, J. F., Wiecki, T. V., Cohen, M. X., Figueroa, C. M., Samanta, J., Sherman, S. J., & Frank, M. J. (2011). Subthalamic nucleus stimulation reverses mediofrontal influence over decision threshold. *Nature Neuroscience*, 14(11), 1462–1467.
- Chistiakova, M., Bannon, N. M., Bazhenov, M., & Volgushev, M. (2014). Heterosynaptic plasticity: multiple mechanisms and multiple roles. *The Neuroscientist*, 20(5), 483–498.
- Chung, S., & Kozma, R. (2020). Reinforcement learning with feedback-modulated TD-STDP. *arXiv:2008.13044*.
- Delvenne, J.-C., Yaliraki, S. N., & Barahona, M. (2010). Stability of graph communities across time scales. *PNAS*, 107(29), 12755–12760.
- Engelhard, B., Finkelstein, J., Cox, J., et al. (2019). Specialized coding of sensory, motor and cognitive variables in VTA dopamine neurons. *Nature*, 570, 509–513.
- Feller, W. (1968). *An Introduction to Probability Theory and Its Applications*, Vol. 1 (3rd ed.). Wiley.
- Foster, D. J., Morris, R. G. M., & Dayan, P. (2000). A model of hippocampally dependent navigation, using the temporal difference learning rule. *Hippocampus*, 10(1), 1–16.
- Frank, M. J. (2006). Hold your horses: a dynamic computational role for the subthalamic nucleus in decision making. *Neural Networks*, 19(8), 1120–1136.
- Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional requirements for reward-modulated spike-timing-dependent plasticity. *Journal of Neuroscience*, 30(40), 13326–13337.
- Frémaux, N., Sprekeler, H., & Gerstner, W. (2013). Reinforcement learning using a continuous time actor-critic framework with spiking neurons. *PLoS Computational Biology*, 9(4), e1003024.
- Gabernet, L., Jadhav, S. P., Feldman, D. E., Carandini, M., & Scanziani, M. (2005). Somatosensory integration controlled by dynamic thalamocortical feed-forward inhibition. *Neuron*, 48(2), 315–327.
- Gershman, S. J., & Niv, Y. (2010). Learning latent structure: carving nature at its joints. *Current Opinion in Neurobiology*, 20(2), 251–256.
- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural Circuits*, 12, 53.
- Gurney, K. N., Humphries, M. D., & Redgrave, P. (2015). A new framework for cortico-striatal plasticity: behavioural theory meets in vitro data at the reinforcement-action interface. *PLoS Biology*, 13(1), e1002034.
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic algorithms and applications. *arXiv:1812.05905*.
- Hamid, A. A., Frank, M. J., & Moore, C. I. (2021). Wave-like dopamine dynamics as a mechanism for spatiotemporal credit assignment. *Cell*, 184(10), 2733–2749.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83–90.
- Hawkes, A. G., & Oakes, D. (1974). A cluster process representation of a self-exciting process. *Journal of Applied Probability*, 11(3), 493–503.
- Henneberger, C., Papouin, T., Oliet, S. H. R., & Rusakov, D. A. (2010). Long-term potentiation depends on release of D-serine from astrocytes. *Nature*, 463, 232–236.
- Kempter, R., Gerstner, W., & van Hemmen, J. L. (1999). Hebbian learning and spiking neurons. *Physical Review E*, 59(4), 4498–4514.
- Lambiotte, R., & Rosvall, M. (2012). Ranking and clustering of nodes in networks with smart teleportation. *Physical Review E*, 85, 056107.
- Lansink, C. S., Goltstein, P. M., Lankelma, J. V., McNaughton, B. L., & Pennartz, C. M. A. (2009). Hippocampus leads ventral striatum in replay of place-reward information. *PLoS Biology*, 7(8), e1000173.
- Li, H. L., & van Rossum, M. C. W. (2020). Energy efficient synaptic plasticity. *eLife*, 9, e50804.
- Litwin-Kumar, A., & Doiron, B. (2012). Slow dynamics and high variability in balanced cortical networks with clustered connections. *Nature Neuroscience*, 15(11), 1498–1505.
- Lo, C.-C., & Wang, X.-J. (2006). Cortico-basal ganglia circuit mechanism for a decision threshold in reaction time tasks. *Nature Neuroscience*, 9(7), 956–963.
- Mazzucato, L., Fontanini, A., & La Camera, G. (2015). Dynamics of multistable states during ongoing and evoked cortical activity. *Journal of Neuroscience*, 35(21), 8214–8231.
- Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5(2), 021028.
- O'Doherty, J., Dayan, P., Schultz, J., Deichmann, R., Friston, K., & Dolan, R. J. (2004). Dissociable roles of ventral and dorsal striatum in instrumental conditioning. *Science*, 304(5669), 452–454.
- Ozeki, H., Finn, I. M., Schaffer, E. S., Miller, K. D., & Ferster, D. (2009). Inhibitory stabilization of the cortical network underlies visual surround suppression. *Neuron*, 62(4), 578–592.
- Pernice, V., Staude, B., Cardanobile, S., & Rotter, S. (2011). How structure determines correlations in neuronal networks. *PLoS Computational Biology*, 7(5), e1002059.
- Petreanu, L., Mao, T., Sternson, S. M., & Svoboda, K. (2009). The subcellular organization of neocortical excitatory connections. *Nature*, 457, 1142–1145.
- Plaçais, P.-Y., & Preat, T. (2013). To favor survival under food shortage, the brain disables costly memory. *Science*, 339(6118), 440–442.
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a vertebrate solution to the selection problem? *Neuroscience*, 89(4), 1009–1023.
- Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex networks reveal community structure. *PNAS*, 105(4), 1118–1123.
- Royer, S., & Paré, D. (2003). Conservation of total synaptic weight through balanced synaptic depression and potentiation. *Nature*, 422(6931), 518–522.
- Sejnowski, T. J. (1977). Storing covariance with nonlinearly interacting neurons. *Journal of Mathematical Biology*, 4(4), 303–321.
- Tervo, D. G. R., Proskurin, M., Manakov, M., Kabra, M., Vollmer, A., Branson, K., & Karpova, A. Y. (2014). Behavioral variability through stochastic choice and its gating by anterior cingulate cortex. *Cell*, 159(1), 21–32.
- Tobler, P. N., Fiorillo, C. D., & Schultz, W. (2005). Adaptive coding of reward value by dopamine neurons. *Science*, 307(5715), 1642–1645.
- Trousdale, J., Hu, Y., Shea-Brown, E., & Josić, K. (2012). Impact of network structure and cellular response on spike time correlations. *PLoS Computational Biology*, 8(3), e1002408.
- Tsodyks, M. V., Skaggs, W. E., Sejnowski, T. J., & McNaughton, B. L. (1997). Paradoxical effects of external modulation of inhibitory interneurons. *Journal of Neuroscience*, 17(11), 4382–4388.
- Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: the leaky, competing accumulator model. *Psychological Review*, 108(3), 550–592.
- van Rossum, M. C. W., Bi, G. Q., & Turrigiano, G. G. (2000). Stable Hebbian learning from spike timing-dependent plasticity. *Journal of Neuroscience*, 20(23), 8812–8821.
- Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. *Science*, 334(6062), 1569–1573.
- Wald, A. (1947). *Sequential Analysis*. Wiley.
- Wang, Z., Joshi, S., Savel'ev, S. E., et al. (2017). Memristors with diffusive dynamics as synaptic emulators for neuromorphic computing. *Nature Materials*, 16, 101–108.
- Wilson, R. C., Takahashi, Y. K., Schoenbaum, G., & Niv, Y. (2014). Orbitofrontal cortex as a cognitive map of task space. *Neuron*, 81(2), 267–279.
- Yagishita, S., Hayashi-Takagi, A., Ellis-Davies, G. C. R., Urakubo, H., Ishii, S., & Kasai, H. (2014). A critical time window for dopamine actions on the structural plasticity of dendritic spines. *Science*, 345(6204), 1616–1620.
- Zenke, F., & Gerstner, W. (2017). Hebbian plasticity requires compensatory processes on multiple timescales. *Philosophical Transactions of the Royal Society B*, 372(1715), 20160259.
