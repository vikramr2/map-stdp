# Iteration 012: finer encoders, a separate critic, and entropy terms (flow)

**Date:** 2026-10-21.

- **Model:** v4 ([`model.md`](../model.md)); no new version. The per-task configurations below are the best results so far, kept as Proposals while iteration 013 compares the model with a TD-STDP backbone.
- **Compute:** local Linux workstation (72 cores), no SLURM. All runs went through the repo CLI in parallel (`xargs`, `OMP_NUM_THREADS=1`). About 680 new `m3_*` results and 230 new `ceiling_*` results.
- **Agents:** `neuroscientist` (plausibility filter, before any code); `snn-expert` (implementation, derivation checks, runs).
- **Status:** done. MountainCar, Acrobot and CartPole improved and confirmed on fresh seeds; LunarLander did not.

Return is the mean of the last 100 episodes, mean ± sd over seeds (ddof = 1); p is Welch's two-sided t-test against v4 on the same seeds. Thresholds: CartPole 500, MountainCar −110, Acrobot −100, LunarLander 200. "Episodes to solve" is the first episode whose trailing-100 mean reaches the threshold [Chung & Kozma 2020].

## 1. Goal

Act on iteration 011 (MountainCar is limited by encoder and critic resolution; half of Acrobot's seeds entrench the zero-torque action) and on Chung & Kozma 2020, who solve CartPole and LunarLander with a Fourier-basis encoder, a separate critic population and an entropy regulariser. User decision: MountainCar keeps grid encoders.

## 2. Plausibility filter (`neuroscientist`, before any code)

| Component | Verdict | Conditions |
| --- | --- | --- |
| (A) Fourier encoder | Accept with conditions | Nonnegative ON/OFF rates $[\cos]_+$, $[-\cos]_+$; conjunction order ≤ 2; one feature per controller neuron; report density. Parallels cosine tuning [Georgopoulos et al. 1982] and grid-cell codes for circular and spatial variables only; for velocities it is an engineering basis [Konidaris et al. 2011]. |
| (B) Separate critic | Accept | Fixed nonnegative RF layer, not trained through the actor; signed $u$ as an E/I or differential pair. A ventral-striatal critic on hippocampal input [O'Doherty et al. 2004; Lansink et al. 2009; Foster et al. 2000]. Control: a critic layer at the controller's own resolution. "Communities as states" then covers the policy only. |
| (C) Entropy term | Accept with conditions; least plausible | **Not inside the $\delta$-gated score (4b)**, where it would average away: a separate ungated per-frame update, as in Chung & Kozma's Eq. 26. Entropy of the $\beta = 1$ shares, so $\beta$ stays the only temperature. Framed as usage homeostasis of action channels through shared inhibition, not a neuromodulator. A slow usage variant (C2) as well. Calibrate to 0.1–0.3 of the task term's RMS. |

## 3. What was built

All default off; golden 9/9 bit-identical; `ruff` clean; `mapstdp.core` self-check extended (finite-difference checks of C1 and C2, Fourier sanity checks). The v4 path through the new code reproduced the committed Acrobot seed-0 returns, and the MountainCar 12 × 12 + 24 × 24-critic runs reproduced iteration 011's scratch runs bit for bit.

- `mapstdp/core.py`: `fourier_coeffs`, `fourier_features`, `fourier_stimulus`, `action_inflow`, `action_entropy`, `entropy_score` (C1, Eq. 4h), `usage_score` (C2, Eq. 4i).
- `experiments/tasks.py`: encoder specs `g<n>[s<σ>]` (an $n \times n$ Gaussian grid) and `f<order>c<conj>[h][b]` (Fourier basis, ON/OFF by default; `h` = $(1 + \cos)/2$; `b` = a bias unit); `FOURIER_BOUNDS` (hand-set observation bounds), `with_encoder`, `critic_layer`.
- `experiments/m2.py`: `Agent(encoder, critic_layer, c_ent, c_use, use_tau, ent_target)`. The regulariser is applied once per frame, ungated, on the pre-update $W$, scaled by the annealed task rate. Diagnostics: RMS ratios to the task update; density; $R^2$ of $V$ from module and neuron shares.
- `experiments/m3.py`: `--encoder`, `--critic-layer`, `--c-ent`, `--ent-target`, `--c-use`, `--use-tau`; tokens `enc<spec>`, `crit<spec>`, `ce<c>[H<f>]`, `cu<c>[t<T>]`, and `ep<E>` for a non-default `--episodes`.
- `experiments/ceiling.py`: `--encoder` (token `_enc<spec>`), `--no-train` (level 1 only; reproduces the committed Acrobot $\beta = 16$ seed 0 bit for bit), and an underflow guard for $p^\beta$.

Derivation checks (Appendix A): with $\pi$ fixed, (4h) and (4i) match finite differences to $2 \times 10^{-10}$; Monte Carlo correlation of the three-factor average ≥ 0.9998; cosine with the full-resolvent gradient 0.996–0.999 on small networks and 0.92–0.93 on the repo's network.

## 4. Results

### 4.1 Stage 1: encoder ceilings (level-1 CEM, 10 runs per cell)

| Task | Encoder ($N_C$) | $\beta = 8$ | $\beta = 16$ | argmax | Gate |
| --- | --- | --- | --- | --- | --- |
| CartPole | 6 × 6 (36) / `f2c2` (64) | 499.9 / 500 | 500 / 500 | 500 / 500 | pass (saturated) |
| Acrobot | pool (64) | −82.7 ± 3.2 | −82.6 ± 7.4 | −81.8 ± 9.9 | |
| Acrobot | `f1c2` (42) | −82.6 (p .99) | −83.9 (p .66) | −84.3 (p .47) | pass |
| Acrobot | `f2c2` (144) | −87.6 (p .017) | −88.2 (p .13) | −86.5 (p .21) | fail |
| LunarLander | pool (64) | 77.6 ± 42.6 | 115.1 ± 63.9 | 106.2 ± 59.4 | |
| LunarLander | `f1c1b` / `f2c1b` / `f1c2b` | −29.1 / −81.1 / – | −37.4 / −65.1 / −55.9 | −37.8 / −75.5 / −50.1 | fail (p ≤ 1e-4) |

LunarLander keeps the pool encoder. No CEM run on it reaches 200 (best 199), so its encoder or readout may itself bind. Density (controller neurons above 10% of peak teleport, held-out states): Fourier 0.37–0.48, the 6 × 6 grids 0.39–0.42, MountainCar 12 × 12 0.07, LunarLander pool 0.18.

### 4.2 Stage 2: pilot (seeds 0–4, exploratory)

- **C1 and C2 at the prescribed 0.1–0.3 RMS calibration are destructive:** Acrobot C1 at $c_e$ 0.1/0.3/1 gives −497/−498/−499 (uniform policy, all 15 seeds); CartPole C1 0.1 gives 200; MountainCar C1 0.3 gives −200; CartPole C2 0.3 gives 70. The task increment is mostly zero-mean noise, while the entropy term is pure drift. And at $\beta = 1$ the v4 action shares are already near uniform ($H(\tilde{P})$ = 0.95 of 1.10 nats on Acrobot), so a $\beta = 1$ entropy term flattens the contrast that $\beta$ amplifies.
- The pre-registered settings, C1 = `--c-ent 0.01 --ent-target 0.8` and C2 = `--c-use 0.01`, have realised RMS ratios of 0.002–0.019, **a deviation from the neuroscientist's calibration**, not yet signed off.
- Critic layers: Acrobot Fourier critic `f2c2b` on the pool encoder −91.3 ± 2.4. CartPole: a Fourier critic on the old $(\theta, \dot{\theta})$ encoder hurt (312 ± 152; it sees cart position, which the actor cannot), with the Fourier encoder it gave 500. LunarLander: separate critics `f1c1b`, `f2c1b`, `f1c2b` and the pooled critic gave 6, 16, 67 and 58, against 82.

### 4.3 Stage 3: pre-registered, seeds 0–19

The pre-registration (scratch `012/PREREG_012.txt`) was written at 20:39, before the first seed 5–19 run (20:56). Seeds 0–4 are reused from the pilot (the runs are deterministic), so stage 3 carries some selection bias; stage 4 removes it. Bonferroni × 14.

| Arm | Return | At threshold | p (× 14) | Median episodes to solve |
| --- | --- | --- | --- | --- |
| MC v4 | −131.9 ± 3.1 | 0/20 | | |
| M1 `--encoder g12s0.15 --critic-layer g24` | −101.9 ± 6.0 | 18/20 | 5e-17 | 806 |
| M2 `--encoder g12s0.15 --critic-layer g12` (control) | −119.6 ± 12.4 | 5/20 | .004 | 423 |
| M3 M1 + C1 | −100.8 ± 5.7 | 19/20 | 3e-18 | 829 |
| M4 M1 + C2 | −104.5 ± 9.8 | 16/20 | 4e-10 | 680 |
| CP v4 | 487.6 ± 51.7 | 17/20 | | 530 |
| P1 `--encoder f2c2 --critic-layer f2c2b` | 500 ± 0 | 20/20 | 1 | 317 (Mann–Whitney p 2e-7) |
| P2 P1 + C1 / P3 P1 + C2 | 500 / 500 | 20/20 | 1 | 295 / 326 |
| AC v4 | −107.6 ± 17.5 | 9/20 | | 406 |
| A1 `--critic-layer f2c2b` | −92.9 ± 5.9 | 18/20 | .023 | 210 |
| A2 `--encoder f1c2 --critic-layer f2c2b` | −93.9 ± 11.0 | 19/20 | .08 | 306 |
| A3 A1 + C1 / A4 A1 + C2 | −93.7 / −98.8 | 15/20 / 13/20 | .047 / .67 | 291 / 244 |
| LL v4 | 77.5 ± 99.6 | 1/20 | | |
| L1 `--critic-layer f1c2b` / L2 v4 + C1 / L3 v4 + C2 | 80.6 / 66.3 / 92.6 | 1 / 0 / 1 of 20 | 1 | |

### 4.4 Stage 4: fresh seeds 20–39

| Arm | Return | At threshold | p (× 6) | Non-inferiority p (margin) |
| --- | --- | --- | --- | --- |
| MC v4 | −130.1 ± 2.9 | 0/20 | | |
| M1 | −108.7 ± 10.1 | 12/20 | 4e-8 | 6e-11 (5) |
| **M3** | **−100.6 ± 4.3** | **19/20** | 2e-22 | 1e-25 |
| CP v4 | 485.9 ± 63.3 | 19/20 | | |
| **P1** | **500 ± 0** | **20/20** | 1 | .006 (25) |
| AC v4 | −108.0 ± 18.1 | 8/20 | | |
| **A1** | **−92.6 ± 6.6** | **17/20** | .010 | 4e-5 (5) |
| LL v4 | 85.0 ± 58.8 | 0/20 | | |
| L1 / L3 | 96.0 / 62.2 | 3/20 / 2/20 | 1 / 1 | .11 / .54 (20): not shown |

- **"Task encoder + separate critic"** ({M1, P1, A1, L1}) is non-inferior on three tasks, not on LunarLander, so no single configuration covers all four.
- **M1 weakened on fresh seeds** (−108.7); M3 beat it head to head (p = .003, exploratory). Under M3, $c_e$ rises to a median 0.021 around episode 200 (RMS ratio up to 0.086) and decays to $4 \times 10^{-4}$: a transient early-exploration bonus.

**All 40 seeds, Chung & Kozma metrics:**

| Arm | Return | Ever solved | Episodes to solve, mean / median | At threshold at end |
| --- | --- | --- | --- | --- |
| MC v4 / M3 | −131.0 / **−100.7 ± 5.0** | 0 / 38 of 40 | – / 848, 790 | 0 / 38 |
| CP v4 / P1 | 486.7 / **500 ± 0** | 39 / 40 | 646, 542 / **316, 296** | 36 / 40 |
| AC v4 / A1 | −107.8 / **−92.8 ± 6.2** | 25 / 40 | 347, 292 / 287, 230 | 17 / 35 |
| LL v4 / L1 | 81.3 / 88.3 | 2 / 8 | | 1 / 4 |

Chung & Kozma report CartPole solved at a mean of 170 episodes, and LunarLander 10 of 10 solved at a mean of 2575 (with Adam, batches of 16 and at least 3000 episodes).

**LunarLander, 3000 episodes, seeds 0–9:** v4 149.0 ± 35.6 (0/10 at the end, 4/10 ever solved); L1 88.7 ± 71.4 (p .032, worse; 1/10, 8/10); L3 126.4 ± 72.5. The separate critic reaches 200 transiently more often but does not hold it.

### 4.5 Value from module shares

Median cross-validated $R^2$ of $V$ decoded from the latent and action module shares: MountainCar 0.61–0.66, CartPole 0.22–0.36, Acrobot 0.27–0.33 (0.70 with the Fourier encoder), LunarLander 0.41–0.43; from neuron-level shares 0.82–0.99. The separate critic's value is only partly represented at module resolution.

## 5. Conclusion

- **Best results so far** (all 40 seeds): MountainCar −100.7 (38/40 at threshold) with M3; Acrobot −92.8 (35/40) with A1; CartPole 500 (40/40, solved in a median 296 episodes) with P1. LunarLander stays at v4.
- The critic's resolution was the main MountainCar limit (M1 against the M2 control). The entropy term helps only as a small, decaying exploration bonus, far below the prescribed calibration.
- No new model version: each task needs its own encoder and critic, and the next iteration may change the backbone.

## 6. Next

**Iteration 013** (on a new branch, after a cleanup and refactor): a TD-STDP backbone [Chung & Kozma 2020] reproduced in this harness, then a hybrid with latent communities and the map term between the input and action columns, with ablations of the latent layer and the map term. The readout should stay bioplausible: action columns with mutual inhibition (e.g. Eq. 1h) rather than an explicit softmax. Open from this iteration: neuroscientist sign-off on the C1 calibration; an entropy term on $P_\beta$; LunarLander's late-training instability.

## Commands

```bash
V4="--td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5"
python -m experiments.ceiling --task T --level 1 --readout R --encoder E --no-train --seed S   # stage 1; S 0-9
python -m experiments.m3 --task MountainCar $V4 --encoder g12s0.15 --critic-layer g24 --c-ent 0.01 --ent-target 0.8 --seed S   # M3
python -m experiments.m3 --task CartPole $V4 --encoder f2c2 --critic-layer f2c2b --seed S      # P1
python -m experiments.m3 --task Acrobot $V4 --critic-layer f2c2b --seed S                      # A1
python -m experiments.m3 --task LunarLander $V4 --critic-layer f1c2b --episodes 3000 --seed S  # L1, long run
```

Scratch in the session scratchpad `012/`: `PREREG_012.txt`, `dcheck.py`, `dcheck_sbm.py`, `ceil_table.py`, `table.py`, `stage.py`, `ck.py`, `ll3000.py`, `r2_v4.py`, job lists and `logs/`.
