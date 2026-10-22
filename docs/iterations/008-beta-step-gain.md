# Iteration 008: β's gain on the task step

**Date:** 2026-10-21.

- **Model:** v5 ([007](007-robust-beta.md)).
- **Compute:** local Linux workstation (72 cores), no SLURM. Runs went through the repo CLI in parallel (`xargs`, `OMP_NUM_THREADS=1`). Results are in `experiments/results/m3_*.json`, all at $\kappa_\beta = 0.01$.
- **Agents:** `snn-expert` (implementation, runs, diagnostics); `neuroscientist` (review of the step gain).
- **Status:** done. No step shape is confirmed; v5's step is kept. At n = 40, v5 is not better than v4 on any task (§6).

All numbers are flow level. Return is the mean of the last 100 episodes, mean ± sd over seeds (ddof = 1); p is Welch's t-test against v5 unless stated. Numbers stay preliminary until reproduced on a second machine.

## 1. Goal

$\beta$ has two roles. It sharpens the policy (Eq. 1e), and through the matched score in Eq. 4b it multiplies the task increment. CartPole needs the sharpness ($\beta \ge 8$), and LunarLander's collapses happen at high $\beta$ and damage the weights (iteration 007). The hypothesis: limiting $\beta$'s effect on the step, with the policy keeping the full $\beta$, keeps CartPole and tightens LunarLander.

Also, at the user's request, seed noise is now measured with more seeds at $\kappa_\beta = 0.01$, not with the $\kappa_\beta$ 0.0099 control of iteration 007.

## 2. What was built

The task increment uses $\beta_\eta(\beta)$ in place of $\beta$, i.e. the Eq. 4b score is multiplied by $\beta_\eta / \beta$:

| Shape | $\beta_\eta(\beta)$ | Flag |
| --- | --- | --- |
| min | $\min(\beta, c)$ | `--beta-step-cap c` |
| sat | $c \beta / (c + \beta)$ | `--beta-step-cap c --beta-step-shape sat` |
| invu | $\beta e^{1 - \beta / c}$ | `--beta-step-cap c --beta-step-shape invu` |
| hyb | $\beta$ for $\beta \le c$, $c (1 + \log(\beta / c))$ above | `--beta-step-cap c --beta-step-shape hyb` |

- `m2.Agent(beta_step_cap, beta_step_shape)`, exclusive with `score_unit` (which is $\beta_\eta \equiv 1$). File tokens `sc<c>`, `sc<c>sat`, `sc<c>iu`, `sc<c>hy`; the JSON records them only when non-default.
- **Direction unchanged.** $\beta_\eta / \beta$ is one positive global scalar, so the increment keeps the matched score's direction exactly: cosine 1.000000000000 between capped and uncapped increments for every shape, norm ratio $\beta_\eta / \beta$ to six digits (scratch `dircheck008.py`, LunarLander network, $\beta \in \lbrace 1.5, 6, 16 \rbrace$). It is a $\beta$-dependent step size on the same policy gradient, not a new objective.
- **Checks.** `pytest -q tests` 9/9 with `tests/golden.json` unchanged; `ruff check .` clean; v5 reproduces its committed files, and `--beta-step-cap 16` (the identity, since $\beta \le 16$) reproduces v5 exactly.

```bash
V5="--td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --beta-eps 0.005 --beta-decay 1000 --eta 5"
python -m experiments.m3 --task T $V5 --beta-step-cap C [--beta-step-shape sat|invu|hyb] --seed S
```

## 3. Results, seeds 0–19

| Group | CartPole | MountainCar | Acrobot | LunarLander | CartPole seeds ≥ 489 | LunarLander seeds < 0 |
| --- | --- | --- | --- | --- | --- | --- |
| v4 | 487.6 ± 51.7 (.21) | −131.9 ± 3.1 (.017) | −107.6 ± 17.5 (.68) | 77.5 ± 99.6 (.59) | 18 | 3 |
| v5 | 449.7 ± 121.2 | −129.5 ± 3.1 | −105.5 ± 15.5 | 60.3 ± 103.5 | 16 | 5 |
| min $c = 2$ | 451.7 ± 102.6 (.96) | −130.3 ± 3.4 (.42) | −105.7 ± 14.6 (.96) | 71.8 ± 88.0 (.71) | 16 | 3 |
| min $c = 4$ | **499.4 ± 2.4** (.08) | −129.9 ± 3.2 (.69) | −114.8 ± 17.5 (.08) | 56.5 ± 92.0 (.90) | 20 | 5 |
| min $c = 8$ | 475.3 ± 84.5 (.44) | −129.4 ± 3.0 (.97) | −113.7 ± 15.3 (.10) | 53.1 ± 91.2 (.82) | 17 | 5 |
| sat $c = 8$ | 491.9 ± 36.2 (.15) | −131.1 ± 3.8 (.16) | −107.2 ± 13.9 (.72) | 52.9 ± 73.5 (.80) | 19 | 4 |
| sat $c = 16$ | 463.2 ± 112.7 (.72) | −131.5 ± 3.4 (.06) | −104.1 ± 12.5 (.75) | 73.5 ± **56.5** (.62) | 18 | **0** |
| invu $c = 6$ | 484.3 ± 70.3 (.28) | −138.3 ± 21.5 (.08) | **−119.6 ± 18.4** (.012) | 90.6 ± **47.9** (.24) | 19 | **0** |

- **No variant beats v5 on mean return** on any task.
- **LunarLander's spread** (Levene, median-centred, against v5): invu $c = 6$ p = .001, sat $c = 16$ p = .009, sat $c = 8$ p = .08, every min cap p > 0.2. Only the smooth shapes narrow it, cutting both tails.
- **CartPole:** min $c = 4$ has sd 2.4 against 121 (Levene p = .07); on seeds 10–19 alone it is 498.9 ± 3.4.
- **invu is rejected.** Its LunarLander is the best, but on MountainCar 2 of 20 seeds never learn ($\beta \approx 1$ throughout; invu amplifies the low-$\beta$ step, $\approx 2.3 \beta$ at $\beta = 1$), and Acrobot is worse.
- **The shapes differ below $c$.** sat shrinks the step at low $\beta$ too ($0.89 \beta$ at $\beta = 1$, $c = 8$) and invu enlarges it, while min leaves it unchanged; so sat and invu are not tests of a high-$\beta$ cap alone. The hybrid (hyb) isolates it.

**Seed-set noise.** v5 on seeds 0–9 against 10–19: CartPole 500.0 ± 0.0 (10 of 10 at 500) against 399.4 ± 159.3 (6 of 10); Acrobot −99.5 against −111.5; LunarLander 91.2 against 29.4; MountainCar −129.8 against −129.1. v4 shows the same: CartPole 500.0 against 475.1, LunarLander 102.5 against 52.5. **At n = 20, v5 is not better than v4 on any task**, and iteration 007's conclusions drawn at 10 seeds are corrected there.

**$\beta$ drops and recoveries** (drops / recovered over 20 seeds, scratch `traj.py` definitions):

| Group | CartPole | LunarLander |
| --- | --- | --- |
| v4 | 3 / 2 | 3 / 0 |
| v5 | 3 / 0 | 3 / 0 |
| min $c = 2$ | 8 / 5 | 1 / 0 |
| min $c = 4$ | 6 / 6 | 1 / 0 |
| min $c = 8$ | 2 / 0 | 2 / 0 |
| sat $c = 8$ | 4 / 3 | 1 / 0 |
| sat $c = 16$ | 6 / 4 | 0 / 0 |
| invu $c = 6$ | 6 / 5 | 0 / 0 |

MountainCar and Acrobot have no drops in any group. Two of v5's three CartPole drops come within 200 episodes of the end of the run (episodes 1300 and 1500), too late to recover at $\epsilon_\beta = 0.005$. v5 seed 12 collapses at $\beta = 16$ (500 → 193 at episode 1300); under min $c = 4$ the same seed ends at 499.6.

## 4. Diagnostics

- **The realised step does not grow with $\beta$.** On LunarLander seed 7, the mean $\lvert \eta_t d_{task} e \rvert$ per synapse under v5 is $0.12$, $0.10$ and $0.09 \times 10^{-4}$ at $\beta < 4$, 4–8 and $\ge 8$: the factor $\mathbb{I} - \tilde{P}_\beta$ in the matched score shrinks as the policy sharpens, and $\eta$ is annealed. So "the effective step is $\beta \eta$" holds for the gain, not for the realised magnitude. min $c = 4$ halves the step at high $\beta$ ($0.04$).
- **LunarLander seed 7.** v5 collapses (220 → −168 around episode 700–750; last-100 −96.9). min $c = 4$ does not (168.8, $\beta$ holds at 12–13); min $c = 8$ still collapses (−87.8); min $c = 2$ ends at 33.3. The frozen episode-750 checkpoint of the min $c = 4$ run is also bad at $\beta \ge 8$ (−185 to −233), so the cap did not prevent the transient weight damage; the run recovered by episode 1000 (frozen 205–238), with $\beta$ never falling. That is one seed.
- **High-$\beta$ fragility is not only the step.** invu CartPole seed 2 still collapses at $\beta = 16$ (500 → 281), where its step gain is 0.8. A plausible reason, not tested: $d \log P = \beta \cdot d \log J^{in}$, so a sharp policy amplifies any weight change by about $\beta$, which no step cap removes.

## 5. Plausibility (`neuroscientist`)

- **A plausible but untested abstraction.** Neuromodulator effects on response gain and on plasticity have dissociable, non-monotone dose-responses: noradrenaline acts through receptor classes of different affinity in prefrontal cortex [Arnsten 2009]; dopamine D1 effects on prefrontal tuning follow an inverted U [Vijayraghavan et al. 2007]; noradrenaline sets the sign and window of cortical STDP through opposing α1- and β-adrenergic receptors [Salgado et al. 2012; Seol et al. 2007]. The specific form, plasticity saturating while gain does not, is not a known receptor mechanism; "receptor saturation" must not be cited for it (the plasticity pathway needs more noradrenaline, not less).
- **Smooth saturation is favoured** over a hard minimum: occupancy and second-messenger saturation are hyperbolic, and $c \beta / (c + \beta)$ is exactly Michaelis–Menten. Both are crossbar-native (one global scalar).
- **Framing.** Not "not a gradient": the step keeps the matched score's direction and changes only its length. A $\beta$-dependent step size, not a new objective.
- An inverted U is more faithful but its classic evidence concerns performance and tuning, not LTP magnitude directly.

## 6. Confirmation on seeds 20–39 (pre-registered)

Fixed before any seed 20–39 run: v5, min $c = 4$, sat $c = 16$ and hyb $c = 4$ on all four tasks, seeds 20–39, $\kappa_\beta = 0.01$ (v4 as a secondary reference if time allows). Primary tests on seeds 20–39 alone, Bonferroni over four:

- **H1.** min $c = 4$, CartPole: mean (Welch against v5) and seeds ≥ 489 (Fisher exact).
- **H2.** min $c = 4$, Acrobot: is the −9 cost real (Welch, two-sided)?
- **H3.** sat $c = 16$, LunarLander: spread (Levene, median-centred) and seeds below 0 (Fisher).
- **H4.** hyb $c = 4$, CartPole mean (Welch); and does it avoid the Acrobot cost?

Secondary: all rows on all four tasks, pooled over seeds 0–39; $\beta$ drops and recoveries.

400 runs: the four arms plus v4 at seeds 20–39 (hyb at $c = 4$ gives $\beta_\eta(16) = 9.55$, a milder cap than min 4 or sat 16; direction check cosine 1.000000000000, norm ratio $\beta_\eta / \beta$). `pytest` 9/9, golden unchanged, ruff clean.

**Fresh seeds 20–39** (n = 20, p against v5):

| Group | CartPole | MountainCar | Acrobot | LunarLander |
| --- | --- | --- | --- | --- |
| v4 | 485.9 ± 63.3 (.35) | −130.1 ± 2.9 (.26) | −108.0 ± 18.1 (.91) | 85.0 ± 58.8 (.063) |
| v5 | 499.4 ± 2.5 | −131.3 ± 3.9 | −108.7 ± 18.4 | 44.5 ± 73.8 |
| min $c = 4$ | 445.6 ± 111.8 (.044) | −130.8 ± 3.3 (.69) | −109.5 ± 15.6 (.89) | 87.7 ± 89.5 (.10) |
| sat $c = 16$ | 435.9 ± 119.9 (.028) | −132.5 ± 3.5 (.33) | −106.1 ± 13.3 (.62) | 68.5 ± 83.2 (.34) |
| hyb $c = 4$ | 484.4 ± 50.9 (.20) | −132.8 ± 3.1 (.17) | −114.4 ± 28.7 (.46) | 67.2 ± 68.6 (.32) |

**Primary tests** (raw p, Bonferroni × 4):

| Test | Result | Raw p | Corrected |
| --- | --- | --- | --- |
| H1: min 4, CartPole mean | 445.6 against 499.4, **opposite to the hypothesis**; seeds ≥ 489 16 against 19 (Fisher .34) | .044 | .18 |
| H2: min 4, Acrobot | −0.8; the −9 of seeds 0–19 does not replicate | .89 | 1 |
| H3: sat 16, LunarLander spread | sd 83.2 against 73.8; seeds < 0 2 against 6 (Fisher .24) | .90 | 1 |
| H4: hyb 4, CartPole mean | 484.4 against 499.4; seeds ≥ 489 18 (Fisher 1); no Acrobot cost (.46) | .20 | .81 |

**No hypothesis is confirmed.**

**Pooled seeds 0–39** (n = 40; hyb only seeds 20–39):

| Group | CartPole | MountainCar | Acrobot | LunarLander | LunarLander seeds < 0 |
| --- | --- | --- | --- | --- | --- |
| v4 | 486.7 ± 57.0 (.47) | −131.0 ± 3.1 (.42) | −107.8 ± 17.6 (.85) | 81.3 ± 80.8 (.13) | 4 / 40 |
| v5 | 474.6 ± 88.3 | −130.4 ± 3.6 | −107.1 ± 16.8 | 52.4 ± 89.1 | 11 / 40 |
| min $c = 4$ | 472.5 ± 82.7 (.91) | −130.4 ± 3.2 (.96) | −112.1 ± 16.6 (.18) | 72.1 ± 91.0 (.33) | — |
| sat $c = 16$ | 449.5 ± 115.7 (.28) | −132.0 ± 3.4 (.048) | −105.1 ± 12.8 (.55) | 71.0 ± 70.2 (.30) | 2 / 40 |

- sat 16's fewer LunarLander failures (2 of 40 against 11 for v5, Fisher p = .013) is post hoc and only an edge over v5: v4 has 4 of 40 (Fisher p = .68 against sat 16). It is not a lead over v4.
- **v5 against v4 at n = 40:** no task differs significantly; v5 is nominally worse on LunarLander (52.4 against 81.3, p = .13; 11 against 4 seeds below 0).
- **Seed-set noise dominates.** v5's CartPole mean over seeds 0–9, 10–19 and 20–39: 500.0, 399.4, 499.4. min 4's 499.4 ± 2.4 on seeds 0–19 became 445.6 ± 111.8 on 20–39.

**$\beta$ drops / recoveries, pooled 0–39** (CartPole; LunarLander): v4 8 / 6, 5 / 1; v5 5 / 2, 4 / 1; min 4 15 / 13, 1 / 0; sat 16 15 / 9, 0 / 0 (hyb 4, seeds 20–39: 6 / 5, 2 / 1). MountainCar and Acrobot have none. The caps roughly triple CartPole's $\beta$ drops and lower LunarLander's. A possible reason, not tested: a smaller step at high $\beta$ slows relearning after a dip, so dips last long enough to trigger a drop.

Scratch: `prereg008.py`, `drops008c.py`.

## 7. Next

- **Reporting unit.** CartPole's mean swings by about 100 between 10-seed sets. Report at least 20 seeds per arm, with the comparison fixed before the runs.
- **v4 reinstated** (the user's decision): at n = 40 the 1g′ set point gives no measurable benefit and LunarLander is nominally worse.
- **The step is not the lever.** None of four step-gain shapes helps, the realised step does not grow with $\beta$ (§4), and high-$\beta$ collapses also happen with a small step. If fragility at high $\beta$ is in the policy's sensitivity to $W$, a check of policy change per unit $\lVert \Delta W \rVert$ against $\beta$ would show it.
- **A matched sharpened spiking readout,** then the spiking heads sweep from iteration 005.
