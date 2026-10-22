# Iteration 006: raising M3 reward (feed-forward controller, adaptive β)

**Date:** 2026-10-20 to 2026-10-21.

- **Model:** v3 + $\beta_{LA} = 0.3$ + `--td-norm 0.04` (the iteration 005 baseline), leading to v4 ([`model.md`](../model.md)): $p_{cc} = 0$ and the adaptive $\beta$ (Eq. 1g, Proposal) at $\eta = 5$.
- **Compute:** local Linux workstation (72 cores), no SLURM. Runs went through the repo CLI in parallel (`xargs -P 40–70`, `OMP_NUM_THREADS=1`). Results are in `experiments/results/m3_*.json`.
- **Agents:** `snn-expert` (diagnosis, prototype, implementation, runs); `neuroscientist` (two plausibility reviews).
- **Status:** done for the flow backend. Spiking has no matched sharpened readout yet (§6).

All numbers are flow level. Return is the mean of the last 100 episodes, mean ± sd over seeds (ddof = 1); p is Welch's t-test against the baseline of the same task. Numbers stay preliminary until reproduced on a second machine.

## 1. Goal

The user paused the memory work (M4, branch `m4-tmaze`) to raise reward on the four fully observed M3 tasks: CartPole, MountainCar, Acrobot, LunarLander. Iteration 005 had found that Map-STDP converges well below the policy-class ceiling, with the policy's sharpness pinned.

## 2. Diagnosis: an ideal learner in the same class (scratchpad prototype)

An actor-critic on $R = \exp \theta$ (actions × controller), started from the same policy, with the exact Eq. 1d score and otherwise the same critic, trace and rate schedule. It is a yardstick, not a candidate: its gradient is non-local.

| Task | Map-STDP | Ideal, Eq. 1d | Ideal, sharpened |
| --- | --- | --- | --- |
| CartPole | 266 | 229 | 321 ± 44 (oscillating) |
| MountainCar | −182 | −164 | −128 |
| Acrobot | −124 | −99 to −111 | −99 |
| LunarLander | −29 | −3 | 114 |

- **On MountainCar, Acrobot and LunarLander, Map-STDP's update was part of the gap.** On CartPole the TD actor-critic recipe is the limit: the ideal learner did worse.
- **The pinned sharpness is mostly the Eq. 1d readout,** not Map-STDP: the ideal learner at $\beta = 1$ is only slightly sharper.
- **The defect is controller recurrence.** The cosine of the expected Eq. 4b increment with the exact score falls from 0.98 (untrained) to 0.55–0.85 on trained weights. The missing part sits in controller→controller synapses, whose credit is per neuron, so no $K$-vector can carry it. Without them the cosine is 0.83–0.99 (derivation §2.4 and Appendix A; re-run for this record, finite differences to $4.6 \times 10^{-11}$).
- **Rejected in the prototype:** the Eq. 4f resolvent credit from `m4-tmaze` (the controller seals itself, CartPole 82–100); an entropy-gradient term (worse on every task); higher fixed $\beta$.

## 3. What was built

| Piece | Where | Notes |
| --- | --- | --- |
| Controller recurrence | `m2.Agent(p_cc=0.1)`, `m3 --p-cc` | Passed to `core.build_network`, which draws the same random arrays for any $p_{cc}$, so the default is bit-identical. |
| Adaptive $\beta$ (Proposal) | `m2.Agent(beta_kappa=0.0, beta_max=4.0)`, `m3 --beta-kappa --beta-max` | Eq. 1g, after each episode. Flow only (asserted). |
| Oracle ramp (reference) | `m2.Agent(beta_end, beta_ramp)`, `m3 --beta-end --beta-ramp` | Linear $\beta$ ramp over a fixed number of episodes. Labelled in code and help text as an oracle reference ablation, **not part of the model**: it needs the run length in advance (the user's decision, after the `neuroscientist` review). |
| Score-unit ablation | `m2.Agent(score_unit=True)`, `m3 --score-unit` | Divides the Eq. 4b increment by $\beta$: the score direction without its $\beta$ step-size gain. |
| Network critic | `m3 --critic net` | Critic features: flow shares over controller and latent neurons (a ventral-striatum-like readout). |
| Task rate, $\lambda$ | `m3 --eta`, `m3 --lam` | `--lam 0` is the map-term ablation (§5). |
| `CartPoleFull` task | `tasks.py` | CartPole with a 64-RF pool encoder over all four observations. Negative result (§4). |
| Summary | `m3.summary` | `nondefault()` keeps every iteration-006 variant out of the baseline groups; file-name tokens `lam`, `pcc`, `b…-…r…`, `bk…m…`, `eta`, `su`. |

`pytest -q tests` passes (9/9, golden fingerprints bit-identical on this machine) and `ruff check .` is clean.

## 4. Results

```bash
python -m experiments.m3 --task T --td-norm 0.04 --seed S                                   # baseline
python -m experiments.m3 --task T --td-norm 0.04 --p-cc 0 --seed S
python -m experiments.m3 --task T --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5 --seed S   # v4
python -m experiments.m3 --task T --td-norm 0.04 --p-cc 0 --beta-end 4 --eta 5 --seed S     # oracle reference
```

**Goes forward** (n = 10, seeds 0–9):

| Variant | CartPole | MountainCar | Acrobot | LunarLander |
| --- | --- | --- | --- | --- |
| Baseline ($p_{cc}$ 0.1, $\beta$ 1) | 272.5 ± 28.0 | −184.2 ± 5.9 | −128.8 ± 7.3 | −37.8 ± 7.5 |
| $p_{cc}$ 0 | 308.3 ± 21.0 (p = .005) | −174.4 ± 2.2 (p = 4e-4) | −123.2 ± 6.3 (p = .08) | −29.1 ± 6.7 (p = .013) |
| **v4: $p_{cc}$ 0 + adaptive $\beta$, $\beta_{max}$ 16, $\eta$ 5** | **500.0 ± 0.1** (p = 1e-9) | **−131.8 ± 2.8** (p = 2e-12) | **−100.1 ± 11.7** (p = 9e-6) | **102.5 ± 94.1** (p = .001) |
| same, $\beta_{max}$ 8 | 385.9 ± 132.5 | −131.8 ± 2.8 | −108.1 ± 17.6 | 106.7 ± 44.2 |
| same, $\beta_{max}$ 4 | 262.9 ± 78.6 | −131.5 ± 4.8 | −104.2 ± 11.7 | 46.4 ± 23.0 |

- Final $\beta$ under v4: CartPole 16 on every seed; MountainCar 5.9–7.5; Acrobot 6.8–15.3; LunarLander 3.2–16.
- **LunarLander is bimodal:** seed 7 collapsed to −101. $\beta_{max} = 8$ is tighter there (106.7 ± 44.2) but costs CartPole (385.9).
- With adaptive $\beta$, $p_{cc} = 0$ matters on LunarLander (102.5 against 29.6 at $p_{cc}$ 0.1, n = 5, p = .04; n = 10 in iteration 007: 26.6 ± 56.2, p = .045) and not significantly on the other three.
- **v4's own numbers are partly a lucky draw (iteration 007).** Changing $\kappa_\beta$ by 1% (0.0099), which changes only the trajectory, gives CartPole 432.0 ± 143.0 and Acrobot −113.7 ± 22.0. Compare against the pooled v4 reference (n = 20): CartPole 466.0 ± 104.4, MountainCar −132.4 ± 2.9, Acrobot −106.9 ± 18.5, LunarLander 100.6 ± 92.1.
- $\beta_{max}$ 32 (n = 5): CartPole 421 ± 176, no better elsewhere (n = 10 in iteration 007: 443.8 ± 129.1; MountainCar and Acrobot bit-identical to v4). $\kappa_\beta$ 0.005 and 0.02 and $\eta = 3$ are worse.

**Reference and ablations:**

| Variant | n | CartPole | MountainCar | Acrobot | LunarLander |
| --- | --- | --- | --- | --- | --- |
| Oracle ramp $\beta$ 1→4, $p_{cc}$ 0, $\eta$ 5 | 10 | 272.9 ± 128.1 | −131.9 ± 2.1 | −106.6 ± 19.1 | 60.6 ± 17.0 |
| Oracle + score unit | 5 | 219.6 ± 89.7 | −130.0 ± 1.6 | −116.7 ± 12.1 | 26.9 ± 21.5 |
| v4 + score unit | 5 | 405.9 ± 129.1 | −131.4 ± 3.7 | −114.2 ± 28.4 | 7.0 ± 38.8 |

- The adaptive rule beats the oracle on CartPole (p = 3e-4) and is not significantly different on the other three tasks.
- **$\beta$ is two things at once.** Dividing it out of the score keeps all of MountainCar's gain (it is sharpness) and removes most of LunarLander's under v4 (it is the larger effective step). Acrobot and CartPole lie between.
- **Correction (iteration 007, n = 10):** v4 + score unit gives CartPole 415.0 ± 102.2, MountainCar −129.3 ± 4.2, Acrobot −110.2 ± 22.0, LunarLander 64.0 ± 95.6 (p = .38 against v4); oracle + score unit gives 231.9, −131.0, −114.7, 4.4. So the LunarLander claim above does not hold at n = 10: with the adaptive rule, dividing $\beta$ out of the score costs LunarLander return, but not significantly. It costs CartPole (p = .027). The step-size part of $\beta$ is clearest under the oracle ramp (LunarLander 60.6 → 4.4, p = .011).

**CartPole: the oscillation is cart drift.** Under a sharp policy (oracle ramp, seed 0, episodes 300–600) 295 of 300 episodes ended with $\lvert x \rvert > 2.4$, 1 with the pole falling and 4 truncated; at $\beta = 1$, $p_{cc} = 0$: 211, 58 and 31. The encoder sees only $(\theta, \dot{\theta})$. Giving the critic $x$ (`--critic rich`, an abstraction: 222.9 at $\beta = 1$, 127.3 with the oracle) or the controller all four observations (`CartPoleFull`: 233.7 baseline, 240.2 at $p_{cc}$ 0) made it worse; the network critic (`--critic net`) changed nothing (303.4 at $p_{cc}$ 0). What fixed it was $\beta$ past about 8: the near-deterministic policy holds 500, and the adaptive rule finds that by itself.

## 5. Map-term ablation on v4

The user's question: does v4 still need the map term? `--lam 0` turns off the structural term (Eq. 4) and with it the latent-output barrier (Eq. 4e), which lives inside the modulator. Homeostasis (Eq. 4d) stays.

```bash
python -m experiments.m3 --task T --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5 --lam 0 --seed S
```

n = 10, seeds 0–9; p against v4:

| Task | v4 | v4, $\lambda = 0$ | p | Baseline (005) |
| --- | --- | --- | --- | --- |
| CartPole | 500.0 ± 0.1 | 396.4 ± 144.8 | .05 | 272.5 |
| MountainCar | −131.8 ± 2.8 | −187.1 ± 27.6 | 1e-4 | −184.2 |
| Acrobot | −100.1 ± 11.7 | −98.3 ± 14.8 | .76 | −128.8 |
| LunarLander | 102.5 ± 94.1 | 11.0 ± 59.2 | .02 | −37.8 |

- **The map term still helps on three of four tasks.** Without it, CartPole loses its reliability (4 of 10 seeds at 500 against 10 of 10; three seeds end at 143–220 with $\beta$ 2.4–4), MountainCar falls back to the baseline, and LunarLander loses most of its gain.
- **MountainCar fails by never reaching the goal.** At $\lambda = 0$, 8 of 10 seeds stay at −200 with $\beta = 1$: the adaptive $\beta$ sees no improving trend, so nothing sharpens. The two seeds that reach the goal (−127, −144) climb to $\beta$ 6–8 as in v4.
- **Acrobot does not need it** under v4 (−98.3 against −100.1).
- Without the map term the run still beats the iteration 005 baseline on CartPole, Acrobot and LunarLander (p = .03, 6e-5, .03), so $p_{cc} = 0$ and the adaptive $\beta$ carry part of the gain on their own.
- In iteration 004 (v3, CartPole) the map term was also worth return; this repeats that on the M3 tasks. Why it helps here (sealing of latent and action modules, iteration 003) is not yet measured on v4.

## 6. Plausibility (`neuroscientist`)

- **$p_{cc} = 0$: accepted.** A feed-forward input layer has good precedent (thalamic relay, cerebellar granule layer; the plastic controller→action projection matches cortico-striatal plasticity). The controller is now an encoder, not a community and not a workspace: SPEC's controller entry is reworded. The alternative of suppressing controller→controller synapses through the cost term was dropped: the map term's diagonal is zero, so it would potentiate them, and a penalty there is not a description-length gradient.
- **The time-based ramp: not plausible.** It needs the run length in advance and was tuned per task. Kept only as an oracle reference.
- **Adaptive $\beta$: an acceptable abstraction** of a utility-driven global gain (the adaptive gain theory of LC-NE [Aston-Jones & Cohen 2005]; normalising by the spread of returns parallels dopamine range adaptation [Tobler et al. 2005]). It is not driven by $\lvert \delta \rvert$, which would sharpen exactly when the agent never succeeds.
- **Open issues it flagged:**
  - The rule needs episode boundaries. A plain per-frame reward rate would not replace it: it is constant in MountainCar and Acrobot. A leaky mean of $\delta / \mathrm{RMS}(\delta)$ is the candidate.
  - After a collapse $\beta$ stays stuck. Slow mean reversion of $\log \beta$ toward a set point is preferred; a term that raises $\beta$ on low utility would be engineering, not the adaptive gain theory.
  - Positive feedback between $\beta$ and return (amplified by the step-size coupling) plausibly explains LunarLander's bimodality.
- **Encoder alternatives** (asked by the user): keep the fixed feed-forward encoder through M3. The free-$R$ ceilings show the encoder is not the current limit. Candidates for later, in front of the stimulus rather than as controller recurrence: random sparse expansion or an on-policy tiling (test with `ceiling.py` first); competitive STDP or sparse coding for pixels; predictive coding as an M4 world-model objective.

## 7. Spiking: deferred

The race readout has no matched sharpened form. A race to $k$ spikes after burn-in (scratch `race_k.py`, simulated with self-excitation) is consistent with Eq. 1e (KL at the sampling floor) but reaches only $\beta_{eff} \approx 1.1$–1.55 for $k = 1$–16 on trained weights, far below the 8–16 that v4 uses. It needs much larger $k$, many heads, or online calibration.

## 8. M4 check

`m4-tmaze` carries only latent activity, so $p_{cc} = 0$ should not affect it. A read-only copy at $D = 4$, $\rho = 0.8$, $\gamma = 2$, `up` credit, 3000 episodes, 5 seeds: eval success 0.976 ± 0.007 at $p_{cc}$ 0 against 0.969 ± 0.02 at 0.1.

## 9. Next

- Fix $\beta$'s stuck-after-collapse failure (mean reversion) and LunarLander's bimodality.
- A per-frame utility signal (leaky mean of normalised $\delta$) so the rule needs no episode boundaries.
- A matched sharpened spiking readout; then the spiking heads sweep from iteration 005.
- Re-check the $n = 5$ rows at $n = 10$.
