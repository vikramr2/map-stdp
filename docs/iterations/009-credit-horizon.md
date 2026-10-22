# Iteration 009: the credit horizon on MountainCar and Acrobot (negative)

**Date:** 2026-10-21.

- **Model:** v4 ([`model.md`](../model.md)); unchanged by this iteration.
- **Compute:** local Linux workstation (72 cores), no SLURM. Runs went through the repo CLI in parallel (`xargs`, `OMP_NUM_THREADS=1`). 280 new `experiments/results/m3_*.json` files, all at $\kappa_\beta = 0.01$.
- **Agents:** `neuroscientist` (filtered the proposals before any code); `snn-expert` (implementation, runs, diagnostics).
- **Status:** done. No arm confirmed; v4 kept.

All numbers are flow level. Return is the mean of the last 100 episodes, mean ± sd over seeds (ddof = 1); p is Welch's two-sided t-test against v4 on the same seeds.

## 1. Goal

Raise MountainCar (v4 −131; Gym threshold −110) and Acrobot (v4 −108; threshold −100). v4 is close to the ideal learner of iteration 006 (MountainCar −128, Acrobot −99), so the limit looked like the shared actor-critic recipe: a TD(0) critic, a 3-frame task trace, and $\gamma = 0.99$, whose 100-frame horizon is shorter than a MountainCar episode.

## 2. Plausibility filter (`neuroscientist`, before any code)

| Candidate | Verdict | Conditions |
| --- | --- | --- |
| Critic trace, TD($\lambda$) | Accept | $\lambda$ 0.9 (0.7 secondary): 0.3–2.5 s at 100–250 ms per frame, inside dopamine-gated eligibility windows [Yagishita et al. 2014; Shindou et al. 2019; He et al. 2015]; TD($\lambda$) traces are argued for in the dopamine reward-learning network [Pan et al. 2005]. Reset at termination; normalise the step at update time, not the trace input; $z$ is periphery state. |
| Task trace $\tau_e = 10$ | Accept | 1–2.5 s. $\tau_e = 30$ (3–7.5 s) exceeds the dopamine-trace evidence; behavioural-timescale plasticity [Bittner et al. 2017] is only analogous. |
| $\gamma = 0.995$ | Accept (its own addition) | A parameter change; no plausibility cost. |
| Sparse-expansion encoder | Lowest priority | Excitatory, fixed or unsupervised, low fan-in [Litwin-Kumar et al. 2017], fixed controller size; check the ceiling first. |
| Novelty bonus, successor-representation critic, replay | Deferred | Low expected gain, recurrence, or non-local in time. |
| Average reward, dopamine ramps | Rejected as arms | Average reward removes the signal under a constant −1 per step; ramps are kept as a diagnostic. |

## 3. What was built

- `m2.Agent(critic_lambda, gamma)`, `m3 --critic-lambda`, `--gamma`; tokens `cl<λ>`, `g<γ>`. The critic always runs through the trace: $z \leftarrow \gamma \lambda z + x$, $u \mathrel{+}= \eta_V \delta z / \lVert x \rVert^2$ (derivation Eq. 4c'), with $z = 0$ at each episode start. At $\lambda = 0$ it is the old critic bit for bit (golden 9/9, and full v4 reruns of MountainCar and Acrobot seed 0 identical).
- $\tau_e = 10$ uses the existing ablation path (`--tau-e`, token `tau_e10`); `nondefault()` now catches it. The task trace resets at episode start, as before.
- $\gamma$ is used only by the critic and $\delta$ (and by the reward trace of `beta_signal='trace'`, off in v4, left at 0.99).

## 4. Results

**Stage 1, seeds 0–19, MountainCar and Acrobot** (pre-registered: arms A, B, C, D against v4, Bonferroni × 8):

| Arm | MountainCar | p (× 8) | Acrobot | p (× 8) |
| --- | --- | --- | --- | --- |
| v4 | −131.9 ± 3.1 | | −107.6 ± 17.5 | |
| A: TD($\lambda$ 0.9) | −129.4 ± 3.5 | .023 (.19) | **−240.2 ± 161.2** | .002 (.013), worse |
| B: $\tau_e = 10$ | −129.9 ± 2.9 | .043 (.34) | −105.0 ± 13.4 | .60 (1) |
| C: A + B | −130.1 ± 5.2 | .20 (1) | **−388.6 ± 172.7** | 6e-7 (5e-6), worse |
| D: $\gamma = 0.995$ | −129.5 ± 3.0 | .017 (.13) | −109.9 ± 15.4 | .67 (1) |
| A($\lambda$ 0.7), secondary | −129.3 ± 3.4 | .014 | −155.1 ± 90.7 | .032, worse |

- **The critic trace diverges on Acrobot.** $V$ reached about $10^{21}$ and the TD RMS up to $10^{151}$ on 20 of 20 seeds at $\lambda = 0.9$ (onset episodes 50–150), 19 of 20 at 0.7. `--td-norm` kept the actor learning, which hides it in the return. MountainCar is unaffected: $\lVert x \rVert^2$ is nearly constant there (coefficient of variation 0.002–0.01) and varies about twelvefold on Acrobot. The likely mechanism: the update-time step $1 / \lVert x_f \rVert^2$ is applied to a trace that holds earlier, larger inputs, which NLMS's stability argument does not bound (not proven; the spectral-radius check is open). So A and C on Acrobot measure divergence, not TD($\lambda$).
- **Exploratory, not pre-registered:** normalising the trace input instead ($z \leftarrow \gamma \lambda z + x / \lVert x \rVert^2$, $\lambda = 0.9$) is stable (TD RMS ≤ 0.64): MountainCar −129.5 ± 2.4 (p = .010 uncorrected), Acrobot −105.5 ± 14.0 (p = .68). The same size as B's stage-1 gain, which did not replicate.

**Stage 2.** B had the largest summed standardised gain (+0.79 v4 sd; D +0.65). On fresh seeds 20–39: MountainCar −130.5 ± 3.2 against −130.1 ± 2.9 (p = .66), Acrobot −107.3 ± 14.4 against −108.0 ± 18.1 (p = .89). **Not confirmed.**

**Non-inferiority of B, seeds 0–19:** CartPole **295.6 ± 124.4** against 487.6 ± 51.7 (p = 1e-6; 1 of 20 seeds at 500 against 18), LunarLander 33.8 ± 48.3 against 77.5 ± 99.6 (p = .09). A 10-frame trace also has about $10/3$ the steady-state gain of a 3-frame one, so B enlarges the actor's step; CartPole is step-sensitive. A step-matched control ($\tau_e = 10$, $\eta = 1.5$) was not run.

**Ideal learner with the same change** (5 seeds, scratch `run009.py`; baselines reproduce iteration 006): $\tau_e = 10$ at the tuned rate collapses 3 of 5 seeds on both tasks; step-matched ($\eta \times 0.3$) it gives MountainCar −139 ± 28 (one collapse) and Acrobot −98.0 ± 3.8 against −102.8 ± 13.8. No recipe gain.

## 5. Diagnostics

- **The critic is not the MountainCar bottleneck.** Along successful episodes (5 seeds, last 100 episodes), $V$ ramps toward the goal and tracks the true discounted value (correlation 0.99 under v4, 0.98 under B; RMSE 0.8–0.9): 100 frames out −5.8 against −6.3, 10 frames −1.1 against −1.0. $\lvert \delta \rvert \le 0.05$. **Corrected in iteration 011:** tracking the value is not enough. The improvements left on MountainCar are worth about 0.1 in TD-error units, below the critic's RMSE, and a separate, finer critic (with a finer encoder) reaches −101.9.
- **TD($\lambda$) speeds early learning only.** Median first goal (MountainCar, seeds 0–19): v4 episode 25 (frame 5.2k), A episode 14 (3.1k, p = .001), B episode 53 (10.8k, slower), D 19. The final return is unchanged.
- **Encoder gate** (level-1 CEM over a free $R$, one run per grid, scratch). Current 6 × 6 grid, $\sigma = 0.4$: Eq. 1d −143, $\beta = 4$ −115 (the committed iteration 005 values are −150 and −122; the gap is CEM noise). Finer grids (6 × 6 at $\sigma$ 0.25, 8 × 8, 10 × 10, 12 × 12) give Eq. 1d −131 to −200 and $\beta = 4$ −102 to −127. No finer encoder clearly raises the Eq. 1d ceiling; only sharper readouts benefit.

## 6. Conclusion

- **v4 is kept.** No arm replicates; the critic trace as specified is unstable; $\tau_e = 10$ costs CartPole.
- **Where the gap is.** v4 samples from Eq. 1e with adaptive $\beta$ (6–8 on MountainCar), whose ceilings on this encoder are −115 to −122 at $\beta = 4$ and −109 at argmax; Acrobot −85 ($\beta = 4$) and −81. So v4 (−131, −108) sits 10–25 below the ceiling of its own readout: the remaining flow gap is the learned weights, not the policy class. (An earlier draft of this record compared v4 with the Eq. 1d ceiling of about −140, which v4 does not use.) Reaching −110 on MountainCar needs both a near-deterministic readout and weights near the CEM optimum. **Corrected in iteration 011:** with 10 CEM runs instead of 3, the 6 × 6 encoder's ceiling is about −114 under any readout (−109 was optimistic), so on MountainCar the encoder and the critic, not only the learned weights, are limits; on Acrobot the weights are (early entrenchment of zero torque).
- Over iterations 007–009 every learning-rule change was within seed noise at 20–40 seeds.

## 7. Next

- **Iteration 010: an action-selection readout** (a race on spike-count differences to a bound), designed with the `neuroscientist` and measured with `ceiling.py` before any training. Its main payoff is the missing sharpened spiking readout; in flow it can help only if its three-action form beats Eq. 1e's ceiling.
- Open from this iteration: the spectral-radius check of (4c'); a step-matched $\tau_e = 10$; whether trace-time normalisation is plausible enough to adopt.

## Commands

```bash
V4="--td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5"
python -m experiments.m3 --task T $V4 --critic-lambda 0.9 --seed S          # A (0.7: secondary)
python -m experiments.m3 --task T $V4 --tau-e 10 --seed S                   # B
python -m experiments.m3 --task T $V4 --tau-e 10 --critic-lambda 0.9 --seed S   # C
python -m experiments.m3 --task T $V4 --gamma 0.995 --seed S                # D
```

T ∈ {MountainCar, Acrobot} (B also CartPole and LunarLander); S = 0–19 (B also 20–39). Scratch analysis in the session scratchpad `m3/009/` (`table.py`, `check_tdl.py`, `diag.py`, `first.py`, `enc_gate.py`, `expl/`) and `run009.py`.
