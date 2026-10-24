# Map-STDP on four engines: flow, branching, LIF and QIF

A plain-language guide to what Map-STDP assumes, how each simulation engine meets (or fails) those assumptions, and what that means for the experiments. The canonical math is in [`derivation.md`](derivation.md); this document explains it and compares the engines side by side. Equation numbers such as (1) or (K4) refer to the derivation.

Status: written 2026-10-24. Numbers marked *diagnostic* come from frozen-weight measurements (no learning), recorded in `docs/iterations/013-tdstdp-backbone.md` §13. They are preliminary until reproduced through the repo CLI.

---

## 1. The idea in one page

**What we want.** A spiking network that solves a task by organising itself into *communities* (modules) that act as states: one module receives the stimulus (the *controller*), one module per action, and some *latent* modules. The flow of activity between modules is then a small Markov chain, a readable model of the task.

**The key move: treat spikes as a random walk.** Picture a single "walker" hopping from neuron to neuron along synapses, following the spikes. A stimulus keeps dropping new walkers into the controller. Over a frame, the fraction of time walkers spend on each neuron is that neuron's *flow*, $\pi_j$, and the number of hops along each synapse $j \to i$ is the synapse's *link flow*, $J_{ij}$.

**Why a walk.** A walk on a graph with communities has a known shortest description, the **map equation** (Rosvall and Bergstrom 2008). If walkers mostly stay inside modules and rarely cross, the walk is cheap to describe. Community detection normally asks "given this graph, which partition is cheapest?" Map-STDP asks the reverse: "given this partition, change the synapses until it is the cheapest one."

**The learning rule in words.** Every time neuron $j$ fires and then neuron $i$ fires (a *pairing*), the synapse $j \to i$ is eligible to change. A broadcast signal (a neuromodulator, one value per module) decides how:

- **Structure term:** if the pairing *left* $j$'s module, it costs bits, so the synapse is weakened; if it stayed inside, it is strengthened. The size of the signal is how expensive that module's leakage currently is.
- **Task term:** a TD error from a critic, times an eligibility trace that remembers which pairings led into the chosen action's module.
- **Homeostasis:** keeps each neuron's total output weight near 1.

**The one assumption everything rests on.** The rule counts pairings and treats them as walk steps. It does the right thing (descends the description length, follows the policy gradient) **only if pairings happen in proportion to the walk's link flow**:

$$
\mathbb{E} \left[ \kappa_{ij} \right] \propto J_{ij}
$$

where $\kappa_{ij}$ is the pairing count for $j \to i$. Each engine below is judged mainly on whether this holds, and on two companions:

1. **The flow matches the walk.** Each neuron's share of spikes equals its walk flow $\pi$.
2. **Pairings match link flow.** The equation above.
3. **The network is subcritical.** One extra spike causes fewer than one extra spike on average, so activity dies out unless the stimulus feeds it. A supercritical network runs away or needs inhibition to hold it, and then it is no longer a walk.

---

## 2. The general picture: a network's linear response is a walk

This section explains the common theory behind all four engines (derivation §2.1).

Give each neuron a firing rate $r_i$ that depends on its input. Now add one extra spike to neuron $j$. It nudges $j$'s targets, which may fire extra spikes, which nudge *their* targets, and so on: a **cascade**. Write

$$
\Lambda_{ij} = \text{expected number of extra spikes of } i \text{ caused by one extra spike of } j
$$

This is the *offspring matrix*. It equals the neuron's gain (how strongly its rate responds to input, $G_i$) times the synapse's strength. A cascade with offspring matrix $\Lambda$ is exactly a branching random walk: each spike of $j$ has on average $\nu_j = \sum_i \Lambda_{ij}$ offspring, each landing on $i$ with probability $\Lambda_{ij} / \nu_j$. The stimulus supplies new "immigrant" spikes.

So any network whose rates respond *linearly* to small inputs defines a walk. The walk's flow is the normalised firing rate, and its step probabilities are set by gain times weight. That is the bridge from spikes to the map equation.

**When the bridge holds:**

- **Linear response.** Small input changes give proportional rate changes. This needs some noise or spread in the neurons: a perfectly deterministic threshold is all-or-nothing, not linear.
- **Subcritical:** every $\nu_j < 1$ (strictly, the largest eigenvalue of $\Lambda$ below 1).
- **Pairings reflect $\Lambda$:** the pairing window has to catch a spike's offspring in proportion to $\Lambda$, without large contributions from long chains or common input.

The four engines differ in how well they meet these.

---

## 3. Flow: the exact reference

**What it is.** No spikes at all. Each frame, the code computes the walk's flow directly, by repeated matrix–vector products ("hops"):

$$
\pi = (1 - \alpha) T \pi + \alpha v(\mathbf{o}) \qquad \text{(1)}
$$

In words: a fraction $1 - \alpha$ of the flow keeps walking along the weight-normalised synapses $T_{ij} = W_{ij} / d_j$, and a fraction $\alpha$ is re-injected by the stimulus $v(\mathbf{o})$ into the controller. Twenty hops get within 2.3% of the exact answer (1a). Pairings are replaced by their expectations, $J_{ij} = \pi_j T_{ij}$.

**What the derivation guarantees.** Everything. By construction the flow *is* the walk and the "pairings" *are* the link flow, so the structural rule's average is exactly weight-scaled gradient descent on the description length (Theorem 3; Monte Carlo check correlation ≥ 0.999). The task term's eligibility (4b) is a one-hop approximation of the exact policy score (cosine 0.98–0.998 with the exact score on untrained weights, 0.83–0.99 on trained ones).

**Limitations.**

- **It is not a spiking network.** It is the noise-free, infinite-sample limit. It shows what the rule does when every assumption holds, not whether neurons can deliver it.
- **The derivation holds the flow fixed when taking gradients** (the "EM split"): it ignores how changing a synapse changes $\pi$ itself. On test networks the resulting direction keeps cosine 0.55–0.92 with the full gradient.
- **It is cheap**, about 0.5–1.3 ms per frame, which makes it the development tool for every other engine.

**Results so far.** Model v4 on flow: CartPole 40/40 solved (best configuration, median 296 episodes to solve), LunarLander 8/40 solved within 1000 episodes (trailing return 88), with no Adam. These are development results, not yet a reported study.

---

## 4. Branching: the implemented spiking engine

**What it is.** SuperNeuroMAT's LIF neuron with the leak set to infinity (the membrane forgets everything each step) and uniform threshold noise. Each step a neuron fires with probability proportional to its input:

$$
P(\text{spike}) = \frac{u}{2 \theta}, \qquad u = \text{summed input this step}
$$

Outputs are scaled by $1 / d_j$ (each neuron's total output weight) and by $1 - \alpha = 0.8$.

**Why it matches the walk.** With firing probability linear in input and no memory between steps, one spike of $j$ causes on average exactly $0.8 T_{ij}$ spikes of $i$ on the next step. That is precisely the walk of Eq. (1): **one simulation step is one hop**. Gains are uniform ($G = 1$), every $\nu_j = 0.8 < 1$ (subcritical), and pairings one step apart occur in proportion to $J$. Measured on CartPole: lag-1 offspring regress on $0.8 T$ with slope 1.04, and pairings on transitions with correlation 0.94.

*Diagnostic (initial weights):* flow error 0.02 (total variation), pairing correlation 0.98 with slope 0.87, rule average cosine 0.99 with its target. It passes every check.

**Limitations.**

- **Sampling noise.** Spikes are random, so a frame's estimate of module flow has error about $1.3 / \sqrt{S}$, with $S$ the spikes per module. More spikes cost more time.
- **Escape noise hurts precise control.** In the TD-STDP baseline, branching neurons learn as fast as deterministic LIF but fail CartPole's strict criterion (100 perfect episodes in a row) in 15 of 20 seeds; the noise-free version of the same rule solves 20/20. The noise level itself is realistic (Poisson-like, as in cortex).
- **Saturation.** Linearity breaks if a neuron's input exceeds $2 \theta$ (probability capped at 1).
- **Engineering conveniences.** The $1/d_j$ output gain and a network-wide normalisation are not obviously biological; they stand in for inhibitory gain control.
- **Trained weights (new, diagnostic).** On weights trained by flow v4, whose modules are almost sealed, the pairing count on weak cross-module synapses picks up whole-module fluctuations that do not scale with the synapse. The rule average then keeps only cosine 0.44–0.89 with its target, and the task credit 0.04–0.47. This is a bias, not noise (split-half reliability 0.89–0.99). The derivation's Appendix A check samples pairings proportional to $J$ by construction, so it could not see this. It needs a closed-loop test.

**Results so far.** Model v4 spiking (16 heads) on CartPole: last-100 return 154. TD-STDP on branching: CartPole 5/20 solved (strict), LunarLander 6/10 with Adam, 0/10 without.

---

## 5. LIF: the leaky integrate-and-fire neuron TD-STDP uses

**What it is.** The textbook neuron, as in BindsNET and the TD-STDP paper (Chung and Kozma 2020): the membrane voltage integrates input and leaks back to rest with a 100 ms time constant; when it crosses a threshold 13 mV above rest the neuron spikes and resets; steps are 1 ms; there is no noise in the neuron itself. Weights are in mV (one input spike adds $w$ mV). TD-STDP solves CartPole on it 20/20 without Adam, at the paper's speed.

**Why it might match the walk.** The general theory of §2 covers any neuron with a linear response, so in principle LIF defines a walk too: offspring = gain × weight. The derivation states this case as a **Proposal** (§5) and has never tested it.

**What the diagnostic found** (v4's network with recurrence, simulated with this neuron, frozen weights, CartPole and LunarLander):

| Check | Branching (control) | LIF as in TD-STDP | LIF + input noise (2–4 mV/√ms) |
| --- | --- | --- | --- |
| Flow matches the walk (TV ≤ 0.05) | 0.02, pass | 0.20–0.28, fail | 0.03–0.09, pass on CartPole only |
| Pairings track link flow (corr ≥ 0.9, slope 0.8–1.25) | 0.98 / 0.87, pass | 0.57–0.76 / 0.55–0.70, fail | 0.64–0.84 / 0.50–0.72, fail |
| Subcritical | 0.80, pass | 0.65, pass | 0.60–0.65, pass |
| Structure rule points the right way (cos ≥ 0.9) | 0.99, pass | 0.88–0.90, fail | 0.94–0.95, pass |
| Task credit matches the exact score (cos ≥ 0.9) | 0.96–0.98, pass | 0.64–0.80, fail | 0.67–0.81, fail |
| Settles within a 20 ms step | 1–6 ms, pass | 82–118 ms from rest, fail | 4–16 ms, pass |

**Why each check fails, in plain words.**

- **Flow.** A deterministic LIF neuron does nothing until its input crosses a hard threshold, so its firing is not proportional to its input: weakly driven neurons stay silent. Noise rounds off the threshold and restores proportionality. Even then each spike causes only about 0.64 follow-on spikes instead of 0.8, because charge is lost to the leak and the reset, so the network acts like a walk that restarts more often than Eq. (1) assumes.
- **Pairings.** The rule counts "pre fires, then post fires 1 ms later" as one walk step. In LIF that count is distorted by one factor per neuron on each side. Fast-firing targets respond inside the 1 ms window more often than slow ones. Clock-like presynaptic neurons produce fewer coincidences per spike than irregular ones. These two factors explain 98–99.5% of the distortion, but they make the count non-proportional, so the rule's assumption fails.
- **Structure rule vs task credit.** The map term only needs to know whether a step left a module, a coarse yes/no signal, so moderate distortion barely changes its direction (cosine 0.94–0.95 with noise). Task credit asks how much each individual neuron's spikes matter downstream, and the per-neuron distortion hands that credit to the wrong neurons (off by 20–35%).
- **Settling.** Started from rest, every membrane must charge up before it fires, one stage after another, which takes 80–120 ms, far longer than a 20 ms environment step. Once running, membranes sit at scattered points between rest and threshold, so a new stimulus passes through within 1–2 ms. Noise shortens the cold start to 4–16 ms.

**Why TD-STDP does well on LIF anyway.** TD-STDP's network is feed-forward: inputs project straight to the actor and critic, with no recurrence and no walk. Its noise comes entirely from the random input spikes, and its rule never assumes pairings equal walk steps. Map-STDP needs recurrence and a walk, which is exactly where deterministic LIF struggles.

**Limitations and recommendation.**

- **Do not use the deterministic LIF as is** for Map-STDP.
- **Add input noise** of about 2 mV/√ms. Then the flow, the subcritical condition, the structure rule and settling all work, and 20 ms persistent frames are fine.
- **Compensate the weaker cascade**, either by using the general walk with measured gains (K3) or by raising the recurrent gain to about 13 mV so each spike again causes about 0.8 follow-ons.
- **Treat the task term as empirical.** No tested LIF setting makes the pairing count proportional, so the task term's credit is approximate. A per-neuron correction of the coincidence window (a row gain) might fix it; untested.

---

## 6. QIF: the quadratic integrate-and-fire neuron

**What it is.** A neuron whose voltage grows quadratically once it nears threshold, so spikes are sharp, self-generated events, like real cortical neurons near firing. Its population behaviour has an exact mean-field description (Montbrió, Pazó and Roxin 2015), which is why it was the target for a more realistic Map-STDP. Rates follow the QIF transfer curve, and each neuron's gain $G_i$ comes from it.

**Why it might match the walk.** As for LIF, the general theory says offspring = gain × weight, and the derivation's Proposal (§5, Lemma K and Theorem K) restates the rule for this case: the structural rule becomes descent scaled by each neuron's offspring mean, and the exact task credit uses a *cascade value* $\Phi_i$, the total downstream effect of one spike of $i$.

**What was measured** (iteration 013 phase B, at the tested operating point `qif.MPR_POINT`):

- **Supercritical.** The excitatory offspring matrix has largest eigenvalue 1.1–1.7: one extra spike causes more than one extra spike. The network is held stable only by per-module inhibition (an *inhibition-stabilised network*, a real cortical regime). A supercritical cascade is not a walk.
- **Very uneven gains.** Neuron gains vary by a coefficient of variation of 1.3–2.2, so some neurons carry far more of the walk than their weights suggest.
- **Pairings do not follow the walk.** Spikes' effects spread over time, and long chains and shared input add coincidences unrelated to the synapse. The structural rule's average has cosine only 0.19–0.61 with its target per synapse. Aggregated to modules it is better (0.86–0.95), but the update acts on synapses.
- **The map term makes it worse.** Sealing modules increases within-module recurrence, which pushes the excitatory network further into supercriticality.
- **Holding the flow fixed loses more.** The EM-split direction keeps cosine 0.25–0.44 with the full gradient on QIF networks, against 0.55–0.92 for flow.
- **Closed loop.** With the corrected score, CartPole reached 191 ± 101 (n = 18), against 488 for flow at the same settings.

**In plain words.** A QIF network at this operating point behaves less like walkers hopping along synapses and more like a crowd that amplifies itself and is held in check by inhibition. Counting "pre then post" coincidences then measures the crowd's mood more than the individual synapse, so the rule cannot see which synapse to change.

**Limitations and what tuning must achieve.** The QIF port is deferred until an operating point exists where:

- each neuron's offspring mean is below 1 (subcritical without relying on inhibition);
- the pairing window is long compared with a spike's response, so offspring are caught evenly;
- the response is close to linear (input kicks small against the spread of excitabilities);
- firing is Poisson-like.

The LIF findings add a fifth: noise has to be enough to linearise the threshold. These are the same conditions the LIF diagnostic tested, so the LIF results are a useful rehearsal for QIF tuning.

---

## 7. Side by side

| | Flow | Branching | LIF (TD-STDP's) | QIF |
| --- | --- | --- | --- | --- |
| Spikes | none (expectations) | stochastic, one step = one hop | deterministic, 1 ms steps, 100 ms leak | self-generated, sharp |
| Flow matches the walk | exact | yes | no (yes with noise, partly) | not tested directly |
| Subcritical | by construction (0.8) | yes (0.8) | yes (0.60–0.65) | **no** (1.1–1.7) |
| Pairings track link flow | exact | yes on initial weights; biased on sealed trained weights | **no** (slope 0.5–0.75) | **no** |
| Structure rule direction | exact | 0.99 (initial), 0.44–0.89 (trained) | 0.88–0.90 (0.94–0.95 with noise) | 0.19–0.61 per synapse |
| Task credit | 0.98–0.998 (one-hop surrogate) | 0.96–0.98 (initial) | 0.64–0.81 | approximate |
| Derivation status | implemented | implemented special case | Proposal, untested before 013 | Proposal; conditions unmet |
| Biological realism | lowest | Poisson-like noise, memoryless | realistic membrane, unrealistic lack of noise | most realistic |

The pattern: **the more realistic the neuron, the weaker the link between pairings and walk steps.** Flow and branching meet the assumption by design. LIF meets part of it once noise is added. QIF at the tested point meets none of the key conditions.

---

## 8. What this means for the experiments

- **Comparing against TD-STDP.** The fair comparison runs Map-STDP and TD-STDP on the same engine. For flow and branching the Map-STDP rule is on solid ground. For LIF, Map-STDP needs input noise and should be reported as "rule applied outside its derivation for the task term", or the task term must be corrected first.
- **Branching on trained weights** needs a closed-loop check of the new bias. One option to test: restrict the map term to within-module synapses, where its direction stays at cosine 0.87–0.93.
- **QIF** stays deferred until tuning reaches an operating point that passes the same checks the LIF diagnostic used.

---

## Glossary

- **Walk / walker:** an imaginary token that hops between neurons following spikes; the map equation describes its path.
- **Flow ($\pi$):** the fraction of the walk's time spent on each neuron, or the normalised firing rate.
- **Link flow ($J$):** the walk's rate of hops along one synapse.
- **Pairing ($\kappa$):** a pre spike followed by a post spike within the rule's window; the rule's stand-in for one hop.
- **Offspring ($\Lambda$, $\nu$):** extra spikes caused by one extra spike; $\nu_j$ is the total for neuron $j$.
- **Subcritical / supercritical:** each spike causes fewer / more than one extra spike on average.
- **Gain ($G$):** how strongly a neuron's firing rate responds to a small change in input.
- **Map equation:** the description length of a walk given a partition into modules; low when walkers rarely leave their module.
- **Modulator ($g$, $M^{\ast}$):** the broadcast signal that tells synapses how costly a module's leakage is.
- **EM split:** taking gradients with the flow held fixed, ignoring how weight changes move the flow.
- **Cascade value ($\Phi$):** the total downstream effect on the policy of one spike of a neuron.
