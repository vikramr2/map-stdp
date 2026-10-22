# Iteration 007: making the adaptive β robust (model v5)

**Date:** 2026-10-21.

- **Model:** v4 ([006](006-m3-reward.md)), leading to v5 ([`model.md`](../model.md)): Eq. 1g plus the experience-dependent set point of Eq. 1g′ (Proposal), `--beta-eps 0.005 --beta-decay 1000`.
- **Compute:** local Linux workstation (72 cores), no SLURM. Runs went through the repo CLI in parallel (`xargs -P 40–60`, `OMP_NUM_THREADS=1`). Results are in `experiments/results/m3_*.json`: 360 runs by `snn-expert` and 80 for the leaky set point (plus one reproduction rerun).
- **Agents:** `snn-expert` (diagnosis, prototype, implementation, runs); `neuroscientist` (review of the set point).
- **Status:** done for the flow backend. It is a robustness fix; return does not change.

All numbers are flow level. Return is the mean of the last 100 episodes, mean ± sd over seeds (ddof = 1); p is Welch's t-test. Numbers stay preliminary until reproduced on a second machine.

## 1. Goal

Item 1 of iteration 006's next steps, three known problems of the adaptive $\beta$ (Eq. 1g):

- **(A)** After a collapse, $\beta$ falls and then stays low: on a new plateau $z_e \approx 0$.
- **(B)** $\beta$ and return reinforce each other, which may explain LunarLander's bimodal outcomes.
- **(C)** 1g needs episode boundaries.

Constraints: one global scalar, causal, no knowledge of the run length, one setting for all tasks; flags whose default path draws nothing extra, so v4 and `tests/golden.json` reproduce.

## 2. The reference is noisier than v4 suggested

Changing $\kappa_\beta$ by 1% (0.01 → 0.0099) changes nothing in the model, only the trajectory, yet:

| Variant | n | CartPole | MountainCar | Acrobot | LunarLander |
| --- | --- | --- | --- | --- | --- |
| v4 | 10 | 500.0 ± 0.1 | −131.8 ± 2.8 | −100.1 ± 11.7 | 102.5 ± 94.1 |
| v4, $\kappa_\beta$ 0.0099 | 10 | 432.0 ± 143.0 (p = .17) | −133.1 ± 3.1 | −113.7 ± 22.0 (p = .11) | 98.6 ± 95.1 |
| **v4 family (pooled)** | 20 | 466.0 ± 104.4 | −132.4 ± 2.9 | −106.9 ± 18.5 | 100.6 ± 92.1 |

From now on, variants are compared against the pooled family, each run at both $\kappa_\beta$ values.

## 3. Diagnoses

- **Collapses live in the weights.** Checkpoints frozen at collapses and replayed at several $\beta$ without learning (scratch `ckpt.py`, `frozen.py`):
  - CartPole seed 1 ($\beta_{max}$ 8), episode 850: about 130 even at $\beta = 64$.
  - Same run, episode 1000: 500 at $\beta \ge 8$, while the running $\beta$ was 2.1 and training return 203. Re-raising $\beta$ would have recovered here.
  - Same run, episode 1200: about 190 at any $\beta$.
  - LunarLander seed 7 (v4), episode 750: −90 to −170 at every $\beta$.

  A $\beta$ rule can help only while the weights still hold a good policy. The collapses happen at high $\beta$, where the effective task step $\beta \eta_e$ is largest.
- **The $\beta$–return correlation is mostly by design.** On LunarLander, corr(late $\beta$, return) is +0.68 under v4 and +0.70 with $\beta$ divided out of the task step (`--score-unit`), so it is not evidence that the step coupling feeds back.
- **Critic-based per-frame signals fail.** The critic starts optimistic ($u = 0$) and $\bar{V}$ falls for about 500 Acrobot episodes. Correlation with the episode $z_e$: mean $\delta / \mathrm{RMS}(\delta)$ −0.59 to −0.95 on Acrobot; the trend of $V$ −0.8 to −0.9 there; the trend of $\delta$ unstable in sign.
- **A critic-free trace works as a signal.** $G_f = r_f + \gamma (1 - \mathrm{term}_{f-1}) G_{f-1}$, reset only at terminal frames (not at truncation). Its fast-minus-slow trend correlates +0.72 to +0.92 with $z_e$ offline, and is zero-mean on stationary synthetic data (derivation Appendix A).

## 4. Results

**Candidates against the family** (Welch p against v4 at $\kappa_\beta$ 0.01, n = 10, unless pooled):

| Variant | n | CartPole | MountainCar | Acrobot | LunarLander |
| --- | --- | --- | --- | --- | --- |
| C1: 1g′, strict maximum, $\epsilon_\beta$ 0.005 | 10 | 500.0 ± 0.0 | −131.9 ± 3.2 | −98.9 ± 10.2 | 102.4 ± 104.4 |
| C1, pooled with $\kappa_\beta$ 0.0099 | 20 | 473.6 ± 74.7 (p = .79 vs family) | −131.1 ± 3.1 (.15) | −106.0 ± 16.9 (.88) | 101.1 ± 79.0 (.99) |
| **v5: 1g′, leaky ($\tau_D$ 1000)** | 10 | 500.0 ± 0.0 | −129.8 ± 3.5 | −99.5 ± 10.5 | 91.2 ± 100.6 |
| **v5, pooled with $\kappa_\beta$ 0.0099** | 20 | 489.8 ± 45.1 (.36) | −130.4 ± 3.6 (.05) | −106.7 ± 19.3 (.98) | 92.3 ± 77.8 (.76) |
| C2: fixed set point $\beta_0$ 8, $\epsilon_\beta$ 0.001 | 10 | 464.6 ± 111.3 (.34) | −132.1 ± 3.4 | −112.9 ± 17.2 (.07) | 121.1 ± 62.8 (.61) |
| C3: per-frame trace, $\kappa$ 5e-5, recovery 7e-5 | 10 | 342.3 ± 166.5 (.015) | −130.4 ± 3.1 | −106.9 ± 10.8 | 91.5 ± 88.4 |
| C3b: same, $\kappa$ 1e-4 | 10 | 452.3 ± 119.1 | −129.6 ± 2.8 | −120.5 ± 24.3 (.03) | 15.3 ± 100.3 (.06) |
| C4: per-frame trace, no recovery | 10 | 260.8 ± 56.4 (3e-7) | −130.9 ± 4.1 | −107.0 ± 13.6 | 44.3 ± 36.2 (.09) |

**$\beta$ drops and recoveries** (drop: $\beta < 0.6 \times$ its running peak, peak > 3; recovery: back to $\ge 0.9 \times$ that peak; $\beta$ read every 50 episodes, scratch `traj.py`; 20 runs per group):

| Group | CartPole | LunarLander |
| --- | --- | --- |
| v4 family | 3 drops, 1 recovered | 2, 0 |
| C1 (strict) | 4, 4 | 2, 1 |
| v5 (leaky) | 5, 5 | 1, 0 |

MountainCar and Acrobot have no drops in any group. LunarLander seed 7 collapses at episode 800 under every variant (its weights are bad, §3).

**Screened and rejected** (prototype, n = 5):

- **Two-sided reversion toward a slow average of $\log \beta$.** It drags $\beta$ down while it climbs: CartPole 221, LunarLander −8.
- **A two-sided ratchet** was worse on CartPole and LunarLander.
- **A floor of $\beta \ge 2$.** Acrobot −130, LunarLander 17.
- **Fixed set points 4 or 8 at $\epsilon_\beta$ 0.002** were neutral to worse.
- **A leaky mean of $\delta / \mathrm{RMS}(\delta)$** (the 006 candidate). $\beta$ stays at 1 on MountainCar (−171) and Acrobot (−116).
- **The trends of $\delta$ and of $V$** gave CartPole 276 and 335.

**Iteration 006 rows extended to n = 10** (p against v4):

| Row | CartPole | MountainCar | Acrobot | LunarLander |
| --- | --- | --- | --- | --- |
| $\beta_{max}$ 32 | 443.8 ± 129.1 (.20) | −131.8 ± 2.8 | −100.1 ± 11.7 | 90.7 ± 80.0 (.77) |
| v4 + score unit | 415.0 ± 102.2 (.027) | −129.3 ± 4.2 | −110.2 ± 22.0 | 64.0 ± 95.6 (.38) |
| Oracle + score unit | 231.9 ± 79.5 | −131.0 ± 2.2 | −114.7 ± 13.8 | 4.4 ± 46.7 |
| $p_{cc}$ 0.1 + adaptive $\beta$ | 466.4 ± 65.8 (.14) | −129.9 ± 1.9 (.09) | −109.2 ± 15.1 (.15) | 26.6 ± 56.2 (.045) |

- **$\beta_{max}$ 32.** MountainCar and Acrobot are bit-identical to v4: $\beta$ never reaches 16 there.
- **Score unit.** The 006 claim that dividing out $\beta$ removes most of LunarLander's gain does not hold at n = 10. The 006 record carries the correction.

**By problem:**

- **(A)** Solved by 1g′: $\beta$ recovers. Return is unchanged.
- **(B)** Not solved by a $\beta$ rule. Clamping $\beta$ (C2, or $\beta_{max}$ 8) narrows LunarLander's spread without raising its mean, and costs CartPole.
- **(C)** Open. The per-frame trace is a usable signal, but no single per-frame rate fits both CartPole and LunarLander, because the per-episode rate of change scales with episode length.

**Correction (iteration 008).** The $\kappa_\beta$ 0.0099 control and seeds 0–9 understated the seed noise. At $\kappa_\beta$ 0.01 on seeds 0–19, v5 scores CartPole 449.7 ± 121.2 (seeds 10–19 alone: 399.4), MountainCar −129.5, Acrobot −105.5 and LunarLander 60.3 ± 103.5, and v4 scores 487.6, −131.9, −107.6 and 77.5: v5 is not better than v4 on any task (MountainCar 2.4 points, p = .017). On those runs v5's CartPole drops do not recover (0 of 3, two within 200 episodes of the end), while v4 recovers 2 of 3 by itself. So the recovery evidence above rests on the $\kappa_\beta$ 0.0099 runs; the mechanism holds (Appendix A of the derivation), its effect on return is not shown. See [008](008-beta-step-gain.md).

## 5. Plausibility (`neuroscientist`)

- **A strict running maximum: rejected.** No tonic LC-NE or dopamine set point is known to be non-decreasing. A strict maximum assumes a stationary task: after a change in contingency it keeps pulling $\beta$ toward exploitation, against the adaptive gain theory, and it would be the model's only permanent quantity (fragile for M4 carry-over). The fix, adopted, is a slowly leaking maximum ($\tau_D \gg \tau_S$).
- **The leaky set point: an acceptable abstraction** of a history-dependent homeostatic set point (metaplasticity [Abraham & Bear 1996]; tonic dopamine as a slow average of reward rate [Niv et al. 2007]; behaviourally, learned industriousness [Eisenberger 1992]). It is not a known LC-NE mechanism. The word "crystallising" was dropped.
- **The fixed set point (C2)** is a more generic mechanism but less consistent with the theory: it pulls $\beta$ down while performance is good, and $\beta_0$ is a hand-set prior. Its variance reduction is a performance argument, not a plausibility one.
- **Trace reset at terminal outcomes: acceptable** (outcomes evoke phasic LC responses; network reset [Bouret & Sara 2005]). It still needs an outcome signal, so it should be called outcome-reset, not episode-free. Truncation must not reset it; the $(1 - \mathrm{term})$ factor ensures that.

## 6. Code and commands

- **`experiments/m2.py`.**
  - New `Agent`/`run` keyword arguments: `beta_eps` (ε), `beta_set` (fixed set point; 0 = experience-dependent), `beta_decay` (τ_D in episodes; 0 = strict maximum), `beta_signal` (`episode` or `trace`).
  - New methods `beta_move()` and `beta_frame()`.
  - New constants `BETA_TF`, `BETA_TS`, `BETA_TH` = 4000, 20000 and 300 frames.
  - The result JSON records the new keys only when they are non-default.
- **`experiments/m3.py`.**
  - Flags `--beta-eps`, `--beta-set`, `--beta-decay`, `--beta-signal`.
  - File-name tokens `re<ε>[s<β₀>][d<τ_D>]` and `tr`.
  - `nondefault()` updated.
- **Checks.**
  - `pytest -q tests`: 9 passed, golden bit-identical. `ruff check .`: clean.
  - v4 reproduces exactly without the new flags.
  - `--beta-decay 0` reproduces the strict-maximum runs bit for bit (MountainCar seed 0 rerun).
  - `/derivation-check` numerics are in derivation Appendix A.

```bash
V4="--td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5"
python -m experiments.m3 --task T $V4 --beta-eps 0.005 --beta-decay 1000 --seed S     # v5 (also with --beta-kappa 0.0099)
python -m experiments.m3 --task T $V4 --beta-eps 0.005 --seed S                       # C1, strict maximum
python -m experiments.m3 --task T $V4 --beta-eps 0.001 --beta-set 8 --seed S          # C2
python -m experiments.m3 --task T --td-norm 0.04 --p-cc 0 --beta-max 16 --eta 5 --beta-signal trace --beta-kappa 5e-5 [--beta-eps 7e-5] --seed S   # C4 [C3]
```

T ∈ {CartPole, MountainCar, Acrobot, LunarLander}; S = 0–9.

## 7. Next

- **The step gain at high $\beta$.** This is the lever for (B) that no rule acting on $\beta$ alone has. Cap how much $\beta$ multiplies the task step, somewhere between the score unit (none) and the matched score (all). Not tested.
- **Per-frame $\beta$ without per-task rates.** Scale the per-frame step by the running mean interval between terminal events, so that the rate per episode is equal across tasks. This uses outcome events, not resets.
- **A matched sharpened spiking readout,** then the spiking heads sweep from iteration 005.
