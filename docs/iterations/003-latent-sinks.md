# Iteration 003: latent modules as flow sinks

**Date:** 2026-10-06.

- **Model:** v3 ([`model.md`](../model.md)), plus the latent-output barrier (Eq. 4e) as a Proposal flag.
- **Agents:**
  - experiments and the derivation check: `snn-expert`;
  - plausibility: `neuroscientist`.
- **Compute:** local workstation, no SLURM.
  - The prototype runs are in the session scratchpad (`sink/variants.py`).
  - The confirmation runs used the repo CLI and wrote `experiments/results/m2_*_lb{0.15,0.3}_v3_s*.json`.

All numbers are preliminary: 5 seeds × 1500 episodes, return as the mean of the last 100 episodes, mean ± sd over seeds.

## Problem

In the trained v3 agent, the latent modules absorb flow from the controller and pass almost none of it on. On the spiking seed-3 weights, flow-level module chain over 50 held-out states:

| | Initial | Trained |
| --- | --- | --- |
| Share of a latent module's flow that stays in it | 0.59–0.65 | 0.983 |
| Latent → action share | 0.05–0.09 | 0.002–0.004 |
| Latent visit share $p_{\mathcal{L}}$ | 0.34 | 0.41 |

## Diagnosis (`snn-expert`)

Flow level, metrics over 100 held-out states. $T_{\mathcal{L}\mathcal{L}}$ is the share a latent keeps, $T_{\mathcal{A}\mathcal{L}}$ its share into both action modules.

| Run | $T_{\mathcal{L}\mathcal{L}}$ | $T_{\mathcal{A}\mathcal{L}}$ | Latent exit to L′ / A / C | $p_{\mathcal{L}}$ | Return |
| --- | --- | --- | --- | --- | --- |
| Initial | 0.657 | 0.105 | 0.061 / 0.105 / 0.175 | 0.34 | – |
| $\lambda = 0$ | 0.719 | 0.002 | 0.090 / 0.002 / 0.188 | 0.55 | 187 ± 15 |
| $\lambda = 0.05$ (v3) | 0.995 | 0.002 | 0.001 / 0.002 / 0.002 | 0.53 | 287 ± 21 |

1. **The task term cuts latent→action flow, with or without the structural term.**
   - At $\lambda = 0$ the latent→action share reaches 0.002 by episode 50.
   - Latent flow carries no stimulus information, so it only dilutes Eq. 1d, and the policy gradient prunes it.
2. **The map term seals latents against other latents and the controller, and that is its benefit.**
   - At $\lambda = 0$, latents send 0.19 of their flow back into the controller and blur its stimulus code.
   - Applying the map term to action modules only gives 184 ± 11, the $\lambda = 0$ level.
3. **The latents' visit share (≈ 0.5) does not depend on $\lambda$.** The map term changes where latent flow goes, not how much there is.

## Fixes tested (flow; baseline 287 ± 21)

| Variant | Return | $T_{\mathcal{A}\mathcal{L}}$ | Outcome |
| --- | --- | --- | --- |
| Exit floor (dual $\nu_a$), $q_{min}$ 0.1–0.2 | 126–196 | 0.005–0.13 | The dual enters a limit cycle. The restored exit goes to C and L′. |
| Dual on $J_{\mathcal{A}\ell}$, target 0.1–0.2 | 161–192 | 0.20 | Limit cycle |
| Exempt L→A | 306 ± 19 ($p = 0.20$) | 0.002 | No effect on sinks; cannot regrow |
| Exempt all inputs to A (postsynaptic) | 290 ± 31 | 0.002 | No effect on sinks |
| Exempt L→A and L→L′ | 294 ± 30 | 0.002 | Unseals only L↔L′ |
| Log-barrier on latent total exit, strength 0.5 / 1.5 | 285 ± 22 / 170 ± 17 | 0.002 | Exits go to C |
| Occupancy homeostasis + exempt A | 108–110 | 0.002 | Caps $p_{\mathcal{L}}$ at 0.20 but loses return |
| 25% "output" neurons per latent | 254 ± 33 | 0.002 | Exits go to C |
| $M^{\ast}$ clipped at 4 on latents | 296 ± 26 | 0.002 | No effect |
| **Latent-output barrier (Eq. 4e), $\beta$ 0.15** | **274 ± 24** ($p = 0.45$) | 0.012 | |
| **Latent-output barrier (Eq. 4e), $\beta$ 0.3** | **279 ± 22** ($p = 0.62$) | 0.029 | |
| Latent-output barrier, $\beta$ 0.5 / 1.0 | 246 ± 16 / 199 ± 13 | 0.057 / 0.144 | |

**Starting from already-sealed weights** (the spiking-trained `viz_m2_W_s1–4`, then 500 more flow episodes):

| Variant | $T_{\mathcal{A}\mathcal{L}}$ after 500 episodes | Return |
| --- | --- | --- |
| Base | 0.002 | 321 ± 37 |
| Map term on actions only | 0.002 | 287 ± 55 |
| Barrier, $\beta$ 0.3 | 0.029 | 290 ± 28 |
| Barrier, $\beta$ 1 | 0.146 | 186 ± 23 |

Only the barrier regrows latent→action synapses. Its modulator grows like $1/J_{\mathcal{A}\ell}$ and cancels the vanishing pairing rate.

## Confirmation through the repo CLI

`python -m experiments.m2 --backend {flow,spiking} --la-beta {0.15,0.3} --seed 0..4`. All 20 runs reproduced the prototype episode for episode.

| Backend | $\beta_{LA}$ | Return | Persist L/A | $p_{\mathcal{A}}$ |
| --- | --- | --- | --- | --- |
| flow | 0 (v3) | 287 ± 21 | 0.996 | 0.23 |
| flow | 0.15 | 274 ± 24 | 0.990 | 0.32 |
| flow | 0.3 | 279 ± 22 | 0.980 | 0.41 |
| spiking | 0 (v3) | 184 ± 13 | 0.981 | 0.39 |
| spiking | 0.15 | 171 ± 9 ($p = 0.16$) | 0.977 | 0.41 |
| spiking | 0.3 | 155 ± 9 ($p = 0.009$) | 0.970 | 0.44 |

## Activity and cross-module traffic

These numbers answer the user's question of whether restoring latent output costs cross-module communication.

| Variant | Cross-module flow (off-diagonal $J^K$) | Of which L→A | Return per unit cross-module flow |
| --- | --- | --- | --- |
| v3 | 0.195 | 0.001 | 1468 |
| $\lambda = 0$ | 0.474 | 0.001 | 394 |
| Barrier, $\beta$ 0.3 | 0.208 | 0.011 | 1340 |
| Exit floor, $q_{min}$ 0.2 | 0.376 | 0.002 | 522 |
| L→A dual, target 0.2 | 0.231 | 0.032 | 830 |

- **Spiking:** 477 spikes per frame in every variant, so the total is fixed by the branching drive. The barrier changes only where the spikes go.
- **Cross-module pairings per frame:** 287 for v3, 290 at $\beta$ 0.15 and 293 at $\beta$ 0.3.
- **Restoring latent output with the barrier barely raises cross-module traffic.**

## Partial-observability check

CartPole with $\dot{\theta}$ hidden from the encoder, flow level, carry-over (Eq. 1b):

| Variant | Return |
| --- | --- |
| v3, $\rho = 0$ | 49.6 ± 2.7 |
| v3, $\rho = 0.5$ | 51.6 ± 2.9 |
| Barrier, $\beta$ 0.3, $\rho = 0.5$ | 47.3 ± 1.4 |
| Map term on actions only, $\rho = 0.5$ | 43.5 ± 2.5 |

All flat. Unsealing the latents is necessary for M4 but not sufficient, for two reasons:

- the fixed-flow score (Eq. 4b) gives no credit to controller→latent or latent→latent synapses;
- linear carry-over forgets.

## Derivation checks

Done with `fdcheck.py` (selftest pass). Recorded in derivation Appendix A:

- **Eq. 4e:** Eq. 5 matches finite differences to ≤ 9e-10. The Monte Carlo correlation with $-W \odot \nabla D_\beta$ is 0.9999, against 0.92 with the plain map gradient. The full-resolvent cosine is 0.934 at $\beta$ 0.3.
- **The rejected duals and barriers:** exact by the same tests.
- **The routing exemption, as implemented, is not an exact gradient:** cosine 0.985. It is exact with $M^{\ast}$ computed from exit rates that exclude the exempt pairs.

## Reviews

**`neuroscientist`:**

- Strong recurrence is biological, as in reverberating assemblies, but a sink that nothing reads out is not.
- Persistence of 0.98 is far beyond anatomy: about 80% of an area's inputs are intrinsic [Markov et al. 2011]. That suggests a latent exit fraction of 0.1–0.3.
- Fixes ranked by plausibility:
  1. a postsynaptic-indexed exemption into actions;
  2. occupancy homeostasis;
  3. the latent→action dual;
  4. the exit floor, which is presynaptic-indexed and undirected.
- It advised against exempting latent→latent.

Its plausibility ranking and the measured results disagree. The two most plausible fixes had no effect or cost return. The barrier works but is not on its list; it is a module-level broadcast like the map modulator.

## Decisions

- **Adopted as a Proposal:** the latent-output barrier, Eq. 4e, behind `--la-beta` (default 0). v1–v3 are unchanged, and `tests/golden.json` is bit-identical. It is not a new default: on CartPole it costs spiking return because latents carry no task information there. Candidates for M4 are 0.15 and 0.3.
- **Rejected:** the exit floor, which replaces the earlier Proposal, plus the latent→action dual, the exemptions, occupancy homeostasis, the output-neuron split and the $M^{\ast}$ clip.
- **Carried to M4:**
  - credit through the latents (an upstream score for C→L and L→L);
  - nonlinear carry-over;
  - making the controller exemption exact by computing $M^{\ast}$ from the remaining exits.
