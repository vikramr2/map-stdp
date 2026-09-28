# Map-STDP: Does it form cortices?

Author: Vikram Ramavarapu

STDP is local, so on its own it has no reason to organise a network into cortices: dense modules with sparse links between them. Map-STDP adds a second plasticity term, the gradient of the **map equation**, which measures how many bits it takes to describe activity flowing through a network. Networks whose activity mostly stays inside modules have short descriptions. Descending this description length therefore pushes the network toward modularity while STDP keeps learning the task.

This document is a corrected and consolidated rewrite of `derivation/mapstdp.tex`. Appendix B lists what changed.

## At a glance

The learning rule is

$$
\Delta W_{ij} = \eta \cdot \mathrm{STDP}(i,j) - \eta' \cdot G(i,j), \qquad G(i,j) = \mathbb{I}[i \notin m(j)] - \bar{e}_j
$$

where $W_{ij}$ is the synapse from presynaptic $j$ to postsynaptic $i$, $m(j)$ is the community of $j$, and $\bar{e}_j$ is the fraction of $j$'s outgoing weight that leaves its community.

How to read it:

| Synapse $j \to i$ | $G(i,j)$ | Effect of $-\eta' G$ |
| --- | --- | --- |
| $i$ in a **different** community from $j$ | $1 - \bar{e}_j > 0$ | weakened |
| $i$ in the **same** community as $j$ | $-\bar{e}_j \le 0$ | strengthened |

- **Magnitude tracks leakage.** A neuron whose output already stays home ($\bar{e}_j \approx 0$) is barely changed. A leaky neuron ($\bar{e}_j \approx 1$) is corrected strongly. For example, with $\bar{e}_j = 0.3$, cross-community synapses get $-0.7\eta'$ and within-community synapses get $+0.3\eta'$.
- **It moves weight rather than adding it.** The weight-weighted sum over each neuron's outputs is zero: $\sum_i W_{ij} G(i,j) = 0$. If the term is applied multiplicatively, as $\Delta W_{ij} = -\eta' W_{ij} G(i,j)$, every neuron's total outgoing weight is exactly conserved and only reallocated from cross-community to within-community targets.
- **It is the true gradient up to a positive per-neuron scale** (§4). The sign is exact, not a heuristic.
- **Every quantity is presynaptic** apart from one bit about the partner, "is $i$ in my community?" (§5).

## 1. Motivation

- **Cortices are useful codes.** When modules map onto concepts, population activity becomes a readable spatial code. In classification, each class gets a module. In control, each action does: CarRacing's five discrete actions become five cortices.
- **Neural activity is a random walk.** Stochastic spiking networks sample from a distribution over activity patterns [Buesing et al. 2011]. The map equation [Rosvall & Bergstrom 2008] scores a network's modularity by the description length of a random walk on it, so it applies naturally to spiking activity.
- **One objective, two roles.** STDP maximises stimulus–response information at single synapses [Dayan & Abbott 2001]. The map term constrains structure at the population level. Under the free-energy view [Friston 2010], the sampling distribution is the network's recognition density, and the map term organises that same distribution into modules instead of adding an unrelated goal.

## 2. Setup: activity as flow

### Notation

| Symbol | Meaning | Where it lives |
| --- | --- | --- |
| $W_{ij} \ge 0$ | synapse from pre $j$ to post $i$ | synapse |
| $z_j(t) \in \lbrace 0, 1 \rbrace$ | spike of neuron $j$ at step $t$ | neuron $j$ |
| $d_j = \sum_k W_{kj}$ | total outgoing weight of $j$ | neuron $j$ (axonal) |
| $T_{ij} = W_{ij} / d_j$ | probability that the walker at $j$ steps to $i$ | synapse |
| $v(\mathbf{o})$ | normalised external drive from stimulus $\mathbf{o}$ | input |
| $\alpha$ | share of activity injected by the stimulus each step | global constant |
| $\pi_j$ | stationary flow through $j$ (how often the walker is there) | neuron $j$ |
| $m(j)$ | community of $j$ | neuron $j$ |
| $e_j = \sum_{i \notin m(j)} W_{ij}$, $\bar{e}_j = e_j / d_j$ | outgoing weight that leaves $j$'s community, absolute and as a fraction | neuron $j$ (axonal) |
| $p_m = \sum_{j \in m} \pi_j$ | how often the walker is in community $m$ | community |
| $q_m = \sum_{j \in m} \pi_j \bar{e}_j$ | how often the walker leaves $m$ | community |
| $q_{\curvearrowright} = \sum_m q_m$ | total rate of switching communities | network |

The membrane equation of a sampling SNN [Buesing et al. 2011] is $u_i = b_i + \sum_j W_{ij} z_j(t)$, so activity flows from $j$ to $i$.

### 2.1 The walker follows the spikes

Picture a walker that moves along synapses in the direction spikes travel. From neuron $j$ it steps to target $i$ with probability $T_{ij} = W_{ij}/d_j$. On each step there is also a chance $\alpha$ that it is instead re-injected by the stimulus at a neuron drawn from $v(\mathbf{o})$. The long-run fraction of time the walker spends at each neuron is the flow $\pi$:

$$
\pi = (1 - \alpha) T \pi + \alpha v(\mathbf{o}), \qquad \sum_i \pi_i = 1 \qquad \text{(1)}
$$

This is personalised PageRank with the stimulus as the personalisation vector. Two consequences:

- **The stimulus is part of the equation.** A change in $\mathbf{o}$ changes $\pi$, and through it the learning rule.
- **$\pi$ always exists, is unique, and power iteration converges to it** for any $\alpha > 0$ (Theorem 1). No separate irreducibility argument is needed.

### 2.2 Estimating flow from spikes (E-step)

Each neuron can estimate its own flow online from its input:

$$
\hat{\pi}_i \leftarrow (1 - \beta) \hat{\pi}_i + \beta \left[ (1 - \alpha) \sum_j \frac{W_{ij}}{d_j} \hat{z}_j(t) + \alpha v_i \right] \qquad \text{(2)}
$$

where $\hat{z}(t) = z(t) / \sum_k z_k(t)$ is the normalised spike vector, taken as zero when nothing fires. The bracket is neuron $i$'s recurrent drive, normalised presynaptically, plus its external drive. Neuron $i$ therefore has everything it needs.

**Assumption (mean field).** The normalised firing rates $r_i$ satisfy the same balance as (1). This holds approximately when rates respond linearly to input, each neuron spreads a fixed total efficacy over its synapses, and the stimulus supplies a fraction $\alpha$ of all drive. Under this assumption $r = \pi$, and (2) tracks $\pi$ with $O(\beta)$ noise. If the assumption fails, (2) still converges, but to a stimulus-weighted smoothing of the rates rather than to the walk's flow. Whether it holds in simulation is something to measure, not to assume.

## 3. The objective: the map equation

### 3.1 Intuition

To report where the walker goes, use a two-level code:

- **Within a module,** name the next neuron with a short, module-local codeword, or name "exit".
- **When exiting,** name the next module with a codeword from a global index.

Frequent moves get short codewords and rare moves get long ones. A step inside a familiar module is therefore cheap. A hop into a rarely visited module costs an exit codeword plus an index codeword. The map equation $L(M)$ is the average number of bits per step under the best such code for the partition $M$. Modular flow, where the walker rarely leaves, gives short descriptions.

### 3.2 Formula

$$
L(M) = q_{\curvearrowright} \log q_{\curvearrowright} - 2 \sum_m q_m \log q_m - \sum_j \pi_j \log \pi_j + \sum_m (p_m + q_m) \log (p_m + q_m) \qquad \text{(3)}
$$

In words: the first two terms are the cost of the global module index, the third is the cost of naming neurons, and the last is the cost of each module's codebook.

### 3.3 The key fact: leakage always costs bits

Hold $\pi$ fixed. Then $W$ enters $L$ only through the exit rates $q_m$, and

$$
\frac{\partial L}{\partial q_m} = \log \frac{q_{\curvearrowright} (p_m + q_m)}{q_m^2} \ge 0
$$

The inequality holds because $q_{\curvearrowright} \ge q_m$ and $p_m + q_m \ge q_m$. **Increasing any module's exit rate never shortens the description.** Minimising $L$ therefore always means reducing leakage, weighted by how costly each module's leakage currently is.

## 4. From objective to learning rule

The rule adds a structural term to STDP:

$$
\Delta W_{ij} = \eta \cdot \mathrm{STDP}(i, j) - \eta' \cdot G(i, j), \qquad \eta, \eta' > 0 \qquad \text{(4)}
$$

The two terms are additive, so each acts independently. STDP improves the response code, and $G$ descends $L$.

**EM approximation.** In truth $\pi$ depends on $W$ through (1). Following that dependence requires the global inverse $\left( I - (1 - \alpha) T \right)^{-1}$. We avoid it with an EM split. The E-step tracks $\pi$ via (2). The M-step updates $W$ with $\pi$ held fixed. This is sound when plasticity is slow relative to flow estimation, $\eta' \ll \beta$.

**Gradient.** With $\pi$ fixed, $W_{ij}$ affects only the exit rate of $j$'s own module (Theorem 2):

$$
\frac{\partial L}{\partial W_{ij}} = \underbrace{\frac{\pi_j}{d_j}}_{\text{how much flow}} \cdot \underbrace{\left( \mathbb{I}[i \notin m(j)] - \bar{e}_j \right)}_{\text{which way}} \cdot \underbrace{\log \frac{q_{\curvearrowright} (p_{m(j)} + q_{m(j)})}{q_{m(j)}^2}}_{\text{how costly this module's leakage is}} \qquad \text{(5)}
$$

The first and last factors are positive (§3.3), so they only rescale the step per neuron. Absorbing them into $\eta'$ gives the rule from "At a glance":

$$
G(i, j) = \mathbb{I}[i \notin m(j)] - \bar{e}_j \qquad \text{(6)}
$$

Dropping the factors is a choice with consequences. Keeping $\pi_j / d_j$ makes high-traffic neurons restructure faster. Keeping the log factor puts more pressure on modules whose leakage is most expensive. Both are cheap to retain if needed.

## 5. Locality: what a synapse needs

Because $W_{ij} = 0$ for non-neighbours, every sum runs over the presynaptic neuron's own outgoing synapses.

| Quantity | Computed from | Available to |
| --- | --- | --- |
| $d_j$, $e_j$, $\bar{e}_j$ | $j$'s axonal synapses | presynaptic neuron $j$, e.g. as a shared presynaptic resource pool |
| $\pi_j$ (optional) | $j$'s input, via (2) | neuron $j$ |
| $\mathbb{I}[i \notin m(j)]$ | community labels of both ends | **not local**: needs one bit about the partner |

The membership bit is the only non-local ingredient, and §6 replaces it.

**Trade-off.** A walk that runs backwards, from each neuron to its inputs, would place the sums on the postsynaptic dendrite, which is the classic locality story. It would lose the link between flow and firing in §2.2. This document uses the forward walk, and `SPEC.md` tracks the choice.

## 6. Removing explicit communities: lateral inhibition

In cortex, modules emerge from local competition, not from global labels. Let $I_{ij} \ge 0$ be inhibition from $j$ onto $i$, and replace the membership bit with a soft competition score:

$$
\chi_{ij} = \frac{I_{ij}}{I_{ij} + \kappa} \in [0, 1), \qquad G_{\chi}(i, j) = \chi_{ij} - \bar{e}^{\chi}_j, \qquad \bar{e}^{\chi}_j = \frac{1}{d_j} \sum_{i'} \chi_{i'j} W_{i'j} \qquad \text{(7)}
$$

Strong mutual inhibition means "different module". When $\chi$ is binary, (7) reduces exactly to (6).

Compared with lateral inhibition alone, as in Kohonen maps, which yields whatever partition winner-take-all dynamics happen to produce, Map-STDP adds:

- **a measurable objective**, the description length $L(M)$;
- **adaptive gain**: $\bar{e}^{\chi}_j$ says how far each neuron is from being contained, not just who its competitors are.

**Open problems.** The originally proposed update $\Delta I_{ij} = \eta_{inh} \pi_i \pi_j$ has three:

- It is unbounded.
- It makes co-active neurons into competitors, and therefore into different modules, which directly opposes Hebbian STDP.
- Pairing each excitatory synapse with a direct inhibitory one ignores Dale's law; inhibition should go through interneurons.

## 7. Training

![Training loop: the E-step runs dynamics and updates the flow estimate; the M-step applies Map-STDP with flow fixed; communities are static or periodically re-detected.](../derivation/algo_infograph.png)

Per stimulus presentation:

1. **E-step:** run the dynamics. Each neuron updates $\hat{\pi}$ via (2). STDP traces accumulate.
2. **M-step:** apply (4) with $G$ from (6) or (7), using the cached $d_j$ and $\bar{e}_j$.
3. **Periodically:** optionally re-detect communities $M$ and prune weak synapses.

State to keep: $W$, the community labels $M$, the flow estimate $\hat{\pi}$, the out-neighbour lists $N^{out}(j)$, and the per-neuron sums $d_j$ and $e_j$.

Design choices that are still open:

- **Static vs. re-detected $M$.** A static $M$ only reinforces the initial partition. Re-detection with Infomap or Leiden adapts to the network but costs repeated community detection. Any partition works with (6). Fixed-size methods suit fixed-output tasks, and Leiden-style methods suit open-ended representation.
- **Topology change.** Threshold pruning ($W_{ij} < \epsilon$) vs. keeping each neuron's top $k$ synapses. Both add a hyperparameter and can destabilise training.
- **Initialisation.** A stochastic block model gives a head start but biases the result toward its partition. Erdős–Rényi is neutral but slow.
- **Workspace.** A central community that receives the stimulus (concentrating $v(\mathbf{o})$) and broadcasts to output modules, in the spirit of Global Workspace Theory [Baars 2005].

## 8. Assumptions and limitations

1. **EM split.** $\pi$ is held fixed during the weight update. This needs $\eta' \ll \beta$.
2. **Mean field.** Flow equals normalised firing rates (§2.2). This is untested.
3. **Teleportation is not counted as an exit.** Only recurrent links contribute to $q_m$.
4. **Presynaptic locality.** The rule is local to the presynaptic neuron, plus one partner bit (§5).
5. **Undefined for fully sealed modules.** The full gradient (5) diverges as $q_m \to 0$. The simplified rule (6) does not.
6. **Lateral-inhibition plasticity is unresolved** (§6).

---

## Appendix A: Proofs

**Theorem 1 (flow exists, is unique, and is reachable).** If $T$ is column-stochastic and $\alpha \in (0, 1]$, then (1) has exactly one solution $\pi$ in the probability simplex, and power iteration $\pi^{(k+1)} = F(\pi^{(k)})$ with $F(x) = (1 - \alpha) T x + \alpha v$ converges to it geometrically.

*Proof.* $F$ maps the simplex into itself, because its entries are nonnegative and $\mathbf{1}^\top F(x) = (1 - \alpha) \mathbf{1}^\top x + \alpha = 1$. A column-stochastic $T$ has $\lVert T \rVert_1 = 1$, so

$$
\lVert F(x) - F(y) \rVert_1 = (1 - \alpha) \lVert T (x - y) \rVert_1 \le (1 - \alpha) \lVert x - y \rVert_1
$$

$F$ is a contraction. By the Banach fixed-point theorem it has a unique fixed point, and $\lVert \pi^{(k)} - \pi \rVert_1 \le (1 - \alpha)^k \lVert \pi^{(0)} - \pi \rVert_1$. $\square$

*Online remark.* Under the mean-field assumption, the bracket in (2) is an unbiased sample of $F(r) = \pi$. Equation (2) is therefore an exponential moving average with steady-state variance $O(\beta)$. A decaying $\beta_t$ with $\sum_t \beta_t = \infty$ and $\sum_t \beta_t^2 < \infty$ gives almost-sure convergence for a mixing spike process.

**Theorem 2 (gradient).** With $\pi$ held fixed, $\partial L / \partial W_{ij}$ is given by (5).

*Proof.* With $\pi$ fixed, the $p_m$ and $\pi_j \log \pi_j$ terms of (3) are constants, and $L$ depends on $W$ only through the $q_m$, including through $q_{\curvearrowright} = \sum_m q_m$. Differentiate (3) with respect to one $q_m$:

$$
\frac{\partial L}{\partial q_m} = \left( \log q_{\curvearrowright} + 1 \right) - 2 \left( \log q_m + 1 \right) + \left( \log (p_m + q_m) + 1 \right) = \log \frac{q_{\curvearrowright} (p_m + q_m)}{q_m^2}
$$

$W_{ij}$ appears only in column $j$ of $T$, and so only in the term $\pi_j \bar{e}_j$ of $q_{m(j)}$. By the quotient rule, with $\partial d_j / \partial W_{ij} = 1$ and $\partial e_j / \partial W_{ij} = \mathbb{I}[i \notin m(j)]$:

$$
\frac{\partial q_{m(j)}}{\partial W_{ij}} = \pi_j \frac{\mathbb{I}[i \notin m(j)] d_j - e_j}{d_j^2} = \frac{\pi_j}{d_j} \left( \mathbb{I}[i \notin m(j)] - \bar{e}_j \right)
$$

Multiplying the two by the chain rule gives (5). $\square$

*Conservation.* $\sum_i W_{ij} \left( \mathbb{I}[i \notin m(j)] - \bar{e}_j \right) = e_j - \bar{e}_j d_j = 0$, which gives the "moves weight" property in "At a glance".

*Numerical checks.* On random 9–12 node networks, (5) matches finite differences of (3) to about $2 \times 10^{-8}$, the power-iteration error ratio stays below $1 - \alpha$, and the conservation identity holds to machine precision.

## Appendix B: Changes from `derivation/mapstdp.tex`

The tex is kept unchanged for history.

1. **Sign error.** The tex expanded $L(M)$ with $+$ on its second and third terms; both are negative. The gradient's log factor becomes $\log \frac{q_{\curvearrowright}(p_m+q_m)}{q_m^2} \ge 0$ instead of $\log \frac{q_{\curvearrowright}}{p_m+q_m} \le 0$. As a result, absorbing the log into the learning rate is legitimate without a "negative learning rate" argument.
2. **Direction convention.** $W_{ij}$ is consistently the synapse from pre $j$ to post $i$, and the walk follows spikes. The tex row-normalised over inputs but wrote $\tilde{p} = \tilde{T} \tilde{p}$, which is inconsistent. The gating function is now indexed by the presynaptic neuron.
3. **Teleportation.** The stimulus enters the flow as $\alpha v(\mathbf{o})$. This makes the flow unique and makes the rule stimulus-dependent.
4. **Flow vs. firing.** The tex equated $p(z_j = 1 \mid \mathbf{o})$ with the walk's stationary distribution. That is now an explicit assumption (§2.2).
5. **Convergence.** The tex showed only a fixed point. The appendix now proves contraction. The singular $(I - T)^{-1}$ becomes the invertible $\left( I - (1 - \alpha) T \right)^{-1}$.
6. **Lateral inhibition.** The tex's $\varphi_{ij}$ summed to 1 per neuron and did not reduce to the indicator; it is replaced by $\chi_{ij}$. The problems with the inhibitory update are flagged.
7. **New interpretation.** Weight conservation under a multiplicative update, and the reading of $\partial L / \partial q_m \ge 0$ as "leakage always costs bits".

## References

- Baars, B. J. (2005). Global workspace theory of consciousness: toward a cognitive neuroscience of human experience. *Progress in Brain Research*, 150, 45–53.
- Buesing, L., Bill, J., Nessler, B., & Maass, W. (2011). Neural dynamics as sampling: a model for stochastic computation in recurrent networks of spiking neurons. *PLoS Computational Biology*, 7(11), e1002211.
- Dayan, P., & Abbott, L. F. (2001). Information theory. In *Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems* (pp. 123–150). MIT Press.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex networks reveal community structure. *PNAS*, 105(4), 1118–1123.
- Shouval, H. Z., Wang, S. S.-H., & Wittenberg, G. M. (2010). Spike timing dependent plasticity: a consequence of more fundamental learning rules. *Frontiers in Computational Neuroscience*, 4.
