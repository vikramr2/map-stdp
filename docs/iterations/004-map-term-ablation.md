# Iteration 004: does the map term help? (three-factor STDP alone)

**Date:** 2026-10-12.

- **Model:** v1–v3 ([`model.md`](../model.md)), map term on ($\lambda = 0.05$) against off ($\lambda = 0$).
- **Compute:** no new runs. This iteration reads the committed `experiments/results/m2_*.json` through
  the repo CLI (`python -m experiments.m2 --summary`); the $\lambda = 0$ sweeps were already run
  (`m2_{flow,spiking}_exempt_lam0_{v1,v2,v3}_s{0..4}.json`).

All numbers are preliminary, reproduced through the repo CLI: 5 seeds × 1500 episodes, return as the
mean of the last 100 episodes, mean ± sd over seeds. Random-policy return is 22.3 ± 11.5.

## Problem

Is the map term (the structural, cost-modulated update $\Delta W_{ij} = -\lambda(g_{ij} - \bar{g}_j)
\kappa_{ij}$, `core.structural_update`, gated by `--lam`) doing work, or can the three-factor task rule
alone — TD-modulated eligibility (Eqs. 4b, 4c) plus $d_j$ homeostasis (Eq. 4d), everything except the
map term — solve CartPole and produce the state-space abstraction? Setting $\lambda = 0$ isolates it.

## Results (`--summary`, exempt routing)

| Model | Backend | $\lambda$ | Return (last 100) | $D$ (bits) | $p_{\mathcal A}$ | persist $\mathcal L$ |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | flow | 0 | 186.7 ± 15.1 | 6.59 → 6.27 | 0.34 → 0.09 | 0.63 → 0.61 |
| v3 | flow | 0.05 | **286.9 ± 20.8** | 6.59 → **4.51** | 0.34 → 0.23 | 0.63 → **0.996** |
| v3 | spiking | 0 | 77.2 ± 4.0 | 6.59 → 6.43 | 0.34 → 0.18 | 0.63 → 0.59 |
| v3 | spiking | 0.05 | **183.6 ± 13.1** | 6.59 → **4.60** | 0.34 → 0.39 | 0.63 → **0.981** |
| v2 | flow | 0 | 144.9 ± 13.7 | 6.59 → 6.25 | — | 0.62 |
| v2 | flow | 0.05 | 260.9 ± 21.1 | 6.59 → 4.52 | — | 0.995 |
| v2 | spiking | 0 | 61.4 ± 3.3 | 6.59 → 6.51 | — | 0.59 |
| v2 | spiking | 0.05 | 120.3 ± 9.5 | 6.59 → 4.66 | — | 0.973 |
| v1 | flow | 0 | 209.5 ± 42.1 | 6.59 → 6.14 | — | 0.66 |
| v1 | flow | 0.05 | 258.4 ± 21.0 | 6.59 → 4.51 | — | 0.996 |
| v1 | spiking | 0 | 114.0 ± 17.9 | 6.59 → 6.17 | — | 0.70 |
| v1 | spiking | 0.05 | 140.0 ± 5.9 | 6.59 → 4.68 | — | 0.968 |

For reference, the best configuration keeps the map term and restores routing with the dual (Eq. 4a):
v3 flow `dual` $\lambda = 0.05$ reaches 327.0 ± 46.1.

## Diagnosis

1. **The map term helps, in every model and both backends.** It adds +23% (v1 flow) to +138% (v3
   spiking) of return. Three-factor STDP alone is not useless — it clears the random baseline (22) and
   on flow it reaches 145–210 — but it plateaus well below the full rule, and on the spiking backend it
   is weak (61–114).
2. **The decisive, model-independent signature is description length.** At $\lambda = 0$, $D$ barely
   moves from its initial 6.587 (to 6.14–6.51); with the map term on it always drops to 4.5–4.7. **The
   task term does not compress the walk.** The map term is what makes the fixed partition the
   minimum-description-length one — which is the project's research question, not an incidental metric.
3. **Communities only sharpen with the map term.** The share of a latent's flow that stays in it
   (persist $\mathcal L$) ends at 0.59–0.70 without the map term and 0.97–0.996 with it. Without it the
   given partition is not a partition the dynamics respect.
4. **Mechanism (cross-reference [003](003-latent-sinks.md)).** The task term cuts latent→action flow
   with or without the map term (so part of CartPole is learnable at $\lambda = 0$), but only the map
   term seals latents against the controller so they stop blurring the stimulus code. Applying the map
   term to the action modules only gives 184 — the $\lambda = 0$ level — confirming the benefit is the
   sealing, not the action-module shaping.
5. **The map term needs routing handled.** With `none` routing (no exemption) and $\lambda > 0$ the
   agent collapses (39–63): the cross-module map term starves the controller→action/latent routing
   links (derivation §4.5). So the honest statement is "the map term helps **given** the routing
   exemption (default) or the dual", not "any map term helps".

## Decision

- **The map term is retained.** Three-factor STDP alone can partially solve CartPole but leaves the
  walk uncompressed ($D \approx 6.3$ vs 4.5) and the communities blurry (persist $\approx 0.6$ vs
  0.99), so it does not deliver the state-space abstraction the project is after. The map term is the
  component responsible for that abstraction and for ~+40–140% of return.
- **No model change.** This iteration only documents the existing $\lambda = 0$ ablation; `model.md` and
  `tests/golden.json` are untouched.
