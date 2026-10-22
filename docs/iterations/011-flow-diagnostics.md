# Iteration 011: why flow v4 misses the MountainCar and Acrobot thresholds (diagnostics)

**Date:** 2026-10-21.

- **Model:** v4 ([`model.md`](../model.md)); unchanged. No repo code changed.
- **Compute:** local Linux workstation (72 cores), no SLURM. Everything ran in scratch code in the session scratchpad `011/`, so **all training numbers here are preliminary** until rerun through the repo CLI (iteration 012). The v4 retrains on seeds 0–39 for both tasks reproduced the committed `returns` lists exactly.
- **Agent:** `snn-expert`.
- **Status:** done. A binding limit was found for each task.

Return is the mean of the last 100 episodes, mean ± sd over seeds (ddof = 1). Thresholds: MountainCar −110, Acrobot −100. Ceilings are level-1 CEM (`experiments/ceiling.py` method), here with **10 CEM runs per cell**, each the mean of 100 fresh episodes.

## 1. Goal

After iterations 007–010, v4 sits at MountainCar −131.0 (0 of 40 seeds at threshold) and Acrobot −107.8 (17 of 40). Find out what limits each task before changing the rule. Five diagnostics: D1 readout sharpness on frozen weights; D2 why adaptive $\beta$ stops climbing; D3 Acrobot's good against bad seeds; D4 the MountainCar encoder ceiling; D5 the learned policy against the CEM policy.

## 2. Results

### D1. A sharper readout does not help

Frozen v4 weights, 100 fresh episodes per seed with common random numbers, 40 seeds:

| Task | $\beta = 1$ | learned $\beta$ | $\beta = 16$ | argmax |
| --- | --- | --- | --- | --- |
| MountainCar | −186.6 | −133.4 (0/40) | −132.9 | −133.6 (0/40) |
| Acrobot | −145.8 | −107.1 (18/40) | −106.0 (19/40) | −107.6 (18/40) |

The gap is in the weights, not in sampling noise. On MountainCar every seed converges to the same policy (between-seed sd 1.3), a fixed point of the rule rather than seed noise.

### D2. Adaptive $\beta$ stops because return stops improving

- Eq. 1g integrates the clipped return trend: $\log \beta_{final} = \kappa_\beta \sum z$ exactly on MountainCar (error $10^{-14}$). The clip never fires.
- On MountainCar, the mean $z$ falls from +0.61 (episodes 20–100) to −0.002 (episodes 700–1500); 97% of the final $\log \beta$ is reached by episode 500. So $\beta$ ends at about $\exp(0.8 \Delta R / s_R)$, the total improvement in units of the return's sd: about 2 sd, hence $\beta \approx 6.6$.
- This is harmless (D1). In runs that keep improving (D5), $\beta$ reaches the cap by itself (median 15.8).

### D3. Acrobot: bad seeds entrench the zero-torque action early

All 40 seeds, 17 good (≥ −100) and 23 bad:

| Measure | good | bad | p |
| --- | --- | --- | --- |
| zero-torque share, last 100 episodes | 0.06 | 0.18 | 7e-5 |
| zero-torque share, episodes 100–200 | 0.09 | 0.18 | .005 |
| best 50-episode average | −84.8 | −104.9 | 6e-8 |
| last 100 minus best 50-episode average | −6.9 | −14.9 | 2e-4 |
| first episode with $R \ge -100$ | 52 | 77 | .06 |
| $\beta$ at episodes 100 / 200 / 300 | 2.3 / 4.4 / 6.3 | 2.4 / 4.4 / 6.2 | n.s. |

- **Early convergence to a worse policy, not slow learning.** Bad seeds reach the goal as fast, then plateau by episodes 200–300. Which seeds will be bad is predictable early: the AUC of mean return is 0.79 over episodes 100–200 and 0.89 over 200–300.
- **Intervention on frozen weights:** masking the zero-torque action lifts bad seeds from −118.6 to −102.7 (paired p = 1e-7; 2 → 11 of 23 at threshold), and all seeds from −107.1 to −97.4 (18 → 26 of 40). Bad seeds have about twice the latent→zero-torque weight, and their latent→action weights are not differentiated across actions.

### D4. The MountainCar encoder caps the ceiling

| Encoder (controller RFs) | $\beta = 8$ | $\beta = 16$ | argmax |
| --- | --- | --- | --- |
| 6 × 6, $\sigma$ 0.4 (current) | −119.9 ± 8.1 | −114.2 ± 4.3 | −115.2 ± 8.5 |
| 8 × 8, $\sigma$ 0.3 | −112.0 ± 4.4 | −110.5 ± 6.3 | −108.0 ± 3.7 |
| 10 × 10, $\sigma$ 0.2 | −109.0 ± 4.2 | −106.1 ± 5.7 | −106.7 ± 8.0 |
| **12 × 12, $\sigma$ 0.15** | −104.5 ± 9.7 | **−101.1 ± 3.4** | −102.8 ± 6.1 |
| two scales (6 × 6 $\sigma$ 0.4 + 12 × 12 $\sigma$ 0.15) | −104.8 ± 4.6 | −104.6 ± 6.2 | −102.3 ± 3.5 |
| 144 random place cells, $\sigma$ 0.15 | −104.5 ± 5.1 | −104.4 ± 7.3 | −101.8 ± 6.2 |

- **Correction.** With 10 CEM runs the current encoder does not exceed −110 under any readout (one-sided p ≥ .96). The 3-run values of iterations 005 and 010 (−110.3 at $\beta = 16$, −109 at argmax) were optimistic. Even optimal weights on the 6 × 6 grid sit near −114.
- Every finer or place-cell encoder lifts the ceiling past −110 beyond CEM noise (12 × 12 at $\beta = 16$: p = 7e-7 against 6 × 6; one-sided p = 9e-6 against −110). At 16 × 16, $\sigma$ 0.1, CEM is unstable. The value-iteration optimum's greedy policy scores −97.7.

### D5. What the rule fails to learn on MountainCar

- **The learned policy is energy pumping** (push along the velocity): argmax agrees with it on 96% of visited states, and the pump heuristic alone scores −131.1. CEM departs from pumping in 5–7% of frames, all at $\lvert v \rvert < 0.018$: it turns before the left wall and picks the first swing.
- **The missing improvements are below the critic's resolution.** Exact $Q$ of the frozen argmax policies: one policy-improvement step would take them from about −132 to about −119; the median single-state gain is 3 steps. With $r_f = -0.1$ and $\gamma = 0.99$ at $T \approx 120$, 3 steps are worth about 0.1 in TD-error units, well below the critic's RMSE of about 0.8 (iteration 009).
- **Training checks** (v4 flags, scratch subclasses; p against v4's 40 seeds):

| Variant | Return | At −110 |
| --- | --- | --- |
| v4 | −131.0 ± 3.1 | 0/40 |
| exact critic $V^{\ast}$ (value iteration), 6 × 6 | −115.1 ± 5.4 (p 4e-12) | 3/20 |
| 12 × 12 encoder, critic on its controller shares | −125.3 ± 9.0 (bimodal) | 2/20 |
| 6 × 6 + separate critic, 12 × 12 / 24 × 24 / 48 × 48 RF layer | −126.6 / −119.8 / −133.1 | 0/10 each |
| 8 × 8 + 24 × 24 critic | −114.4 ± 9.4 | 4/10 |
| 10 × 10 + 24 × 24 critic | −111.3 ± 9.7 | 5/10 |
| **12 × 12 + 24 × 24 critic** | **−101.9 ± 6.0** (p 1e-16) | **18/20** |
| 12 × 12 + exact critic | −102.6 ± 8.9 | 16/20 |

"Separate critic" means the critic is NLMS (Eq. 4c) on its own fixed Gaussian RF layer instead of on the controller's module flow shares. An exact critic takes the 6 × 6 encoder to its own ceiling (−115 against −114); a good critic on a fine encoder reaches the optimum. Under 12 × 12 + 24 × 24 critic, $\beta$ reaches the cap (median 15.8), and the median time to a 50-episode average of −110 is about 250 episodes (four seeds take 800–1300).

## 3. Conclusion

- **MountainCar** has two limits that stack: the encoder's resolution (D4) and the critic's advantage resolution (D5). Neither alone reaches −110; together they reach −101.9, 18 of 20 seeds. $\beta$ and the readout are not limits (D1, D2).
- **Acrobot** is limited by its weights: about half the seeds entrench the zero-torque action by episodes 150–300 (D3).
- This corrects iteration 009, which judged the critic not to be the MountainCar bottleneck from its value correlation (0.99): the critic tracks the value but cannot resolve the small advantages that remain.

## 4. Not run or failed

- A richer critic on Acrobot (`critic='rich'`, 256 pooled RFs) gives −170.2 ± 87.1; on MountainCar it gives −200 on all 10 seeds (not diagnosed; the pooled encoder is probably fitted on valley states only).
- Larger Acrobot controllers (pool 128 / 256, untuned rates): −221.6 / −182.7.
- Two of ten 16 × 16 CEM runs crashed: $p^{16}$ underflows to NaN in `ceiling.act`.
- No exact critic for Acrobot; no spiking runs; no CartPole or LunarLander checks.

## 5. Next

**Iteration 012** (user decision), each component behind its own default-off flag, filtered by the `neuroscientist` first:

1. **MountainCar:** a 12 × 12 controller grid and a separate critic on a 24 × 24 RF layer.
2. **CartPole, Acrobot, LunarLander:** a Fourier-basis encoder [Chung & Kozma 2020], checked against the current encoders on the CEM ceiling first, and the separate critic population.
3. **All tasks:** an entropy bonus on the actor score, aimed at Acrobot's early entrenchment and LunarLander's collapses.

Pre-registered against v4 on seeds 0–19, then fresh seeds and CartPole/LunarLander non-inferiority.

## Files

Scratch only, in the session scratchpad `011/`: `train.py` (traced v4 retrain), `d1.py`, `d1_table.py`, `d2.py`, `d3.py`, `d3b.py`, `enc_cem.py`, `d4_table.py`, `d5.py`, `d5b.py`, `d5c.py`, `vstar.py` (value iteration), `train2.py`, `t2_table.py`; results in `W/`, `d1/`, `enc/` (310 CEM files) and `t2/`. Nothing was written to `experiments/results/`.
