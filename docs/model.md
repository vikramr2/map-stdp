# Map-STDP model: current form

**Version:** v1 (2026-10-05). It is the model after iteration 001 (M1), and the configuration M2 starts from. The specification is in [`derivation.md`](derivation.md) and [`SPEC.md`](SPEC.md). Iterations are logged in [`iterations/`](iterations/), and every change to this file is listed in the version history at the end.

Status tags:

- **impl**: implemented and tested.
- **spec**: decided, not yet implemented.
- **Proposal**: an undecided option, behind a flag.
- **abstraction**: a biological-plausibility note from the `neuroscientist` review.

## 1. Network

| Item | Current form | Status |
| --- | --- | --- |
| Neurons | Modules C, L, L, A, A, all excitatory. M1 used 16 neurons per module (80 total). For CartPole the controller has 36 neurons (§1, Stimulus). | impl (M1) |
| Weights | $W_{ij} \ge 0$, pre $j$ → post $i$, bounded in $[10^{-3}, 1]$. $T = W/d$. In SuperNeuroMAT, `W_snm = (1-α)·(W/d).T`, so the output gain is $1/d_j$. | impl (M1) |
| Initialisation | SBM: $p_{same} = 0.5$, controller recurrence $p_{cc} = 0.1$, controller→latent/action routing $p_{route} = 0.3$, other cross links $p_{cross} = 0.05$. Initial $d_j = 1$. | impl (M1) |
| Stimulus | **Conjunctive** Gaussian receptive fields. For CartPole: a 6×6 grid over $(\theta/0.21, \dot{\theta}/3.5)$, clipped to $[-1, 1]$, $\sigma = 0.4$, $v$ normalised. Per-dimension fields cap the policy at about 84 steps. | spec (M2) |
| Excitation/inhibition | Excitatory only. **No plausibility claim** until per-module inhibitory (PV-like) pools with inhibitory STDP replace the software normalisation (derivation §7). | abstraction |

## 2. Per-frame flow (one walk per frame)

| Item | Current form | Status |
| --- | --- | --- |
| Flow | $\pi_f = (1-\alpha) T \pi_f + \alpha v_f$, by $n = 20$ hops of power iteration (Eq. 1a), with $\alpha = 0.2$. Within the bound on both initial and trained $W$. | impl (M1) |
| Carry-over | Eq. 1b, $\rho = 0$. | Proposal |
| Spiking mode | **Branching:** `leak=inf`, uniform threshold noise $U(-\theta, \theta)$ with $\theta = 0.5$, so $P(\text{spike}) = u/(2\theta)$ is linear. Teleportation is Bernoulli input at rate $1.0 \cdot v_i$. Mean rate ≈0.057 per neuron per step. Abstraction: one step ≈ 3–10 ms, valid at ≤0.05 spikes per neuron per step. The branching ratio $1-\alpha = 0.8$ is much more subcritical than cortex (≈0.98). | impl (M1) |
| Frame window | $k = 100$ steps. Module error ≈ $1.3 / \sqrt{S_{mod}}$, with $S_{mod}$ the spikes per module per frame; about 170 spikes per module give 0.1. Finite leak has a 0.09–0.15 error floor and is not used. | impl (M1) |
| Pairings $\kappa$ | **Covariance count** $C_{ij} - c_i c_j (k-1)/k^2$: lag-1 causal pairings minus the product of spike counts. It correlates 0.91 with $J$, and its average update has cosine 0.992 with $-W \odot \nabla D$. Causal-only is an ablation; balanced is dropped (cosine 0.847). Biological variant (Proposal): subtract slow pre/post traces online. | impl (M1) |

## 3. Readout and state model

| Item | Current form | Status |
| --- | --- | --- |
| Module chain | $T^K_{ba}(\mathbf{o}) = J_{ba} / p_a$ (Eq. 1c). | impl (M1) |
| Policy | $P(a \mid \mathbf{o}) = p_a / \sum_{a'} p_{a'}$ (Eq. 1d). In spiking, a race readout, whose KL to Eq. 1d must be measured at M2. | impl / spec |

## 4. Plasticity

The full rule is $\Delta W_{ij} = \eta \delta_f e_{ij} - \lambda (g_{ij} - \bar{g}_j + \epsilon (d_j - 1)) \kappa_{ij}$.

| Item | Current form | Status |
| --- | --- | --- |
| Structural modulator | Map equation, $g_{ba} = M^{\ast}_a \mathbb{I}[b \ne a]$, with $M^{\ast}$ clipped at 20. $\lambda = 0.05$, constant from frame 0. In spiking, $\lambda$ is divided by the spikes per frame, keeping the per-count $\lambda \lesssim 5 \times 10^{-4}$. | impl (M1); λ spec |
| Routing protection | **Exempt** (M2 default): $g = 0$ on controller→latent/action pairs, so the controller acts as an input layer. Ablations: none (collapses learning in the closed loop) and dual (Eq. 4a, $q^{\ast} = 0.75 J_{route}(W_0)$, $\eta_\mu = 2$). | impl (M1); default spec (M2) |
| $d_j$ homeostasis | $+\epsilon (d_j - 1)$ in the per-neuron baseline, $\epsilon = 1$. Its expectation is a uniform column rescaling, so $T$ and $D$ are unchanged. It fixes the $w_{min}$-floor ratchet: max $d_j$ goes from 2.04 to 1.12. | spec (M2) |
| Task eligibility | $e_{ij} \leftarrow (1 - 1/\tau_e) e_{ij} + \kappa_{ij} (h_{m(i)} - \bar{h}_j)$ (Eq. 4b), $\tau_e = 3$ frames. | spec (M2) |
| Task modulator | TD error $\delta_f = r_f + \gamma V_{f+1} - V_f$, $\gamma = 0.99$. Critic $V = u \cdot \pi_{\mathcal{C}} / p_{\mathcal{C}}$, a readout of controller flow shares, trained by NLMS with $\eta_V = 0.1$. Reward −1 on termination, 0 otherwise, bootstrapping on truncation. $\eta = 10$. | spec (M2) |

## 5. Parameters

| Symbol | Meaning | Value |
| --- | --- | --- |
| $\alpha$, $n$, $\rho$ | teleportation share, hops per frame, carry-over | 0.2, 20, 0 |
| $\lambda$, $\epsilon$ | structural rate, $d_j$ homeostasis | 0.05, 1 |
| $\eta$, $\tau_e$ | task rate, eligibility time constant (frames) | 10, 3 |
| $\gamma$, $\eta_V$ | discount, critic rate (NLMS) | 0.99, 0.1 |
| $\theta$, input rate, $k$ | spiking noise, input rate, frame window | 0.5, 1.0, 100 |
| $w_{min}$, $w_{max}$, $M^{\ast}$ clip | bounds | $10^{-3}$, 1, 20 |
| seeds, episodes | M2 runs | 0–4, 1500 |

## Version history

| Version | Date | Change | Iteration |
| --- | --- | --- | --- |
| v0 | 2026-10-05 | Initial form from derivation and SPEC; nothing implemented. | — |
| v1 | 2026-10-05 | M1 implemented. Changes: covariance $\kappa$; output gain $1/d_j$ and $d_j$ homeostasis; branching spiking mode with $k = 100$; conjunctive encoding; routing exempt by default; constant $\lambda$; task-term parameters from flow-level CartPole tests. | [001](iterations/001-m1.md) |
