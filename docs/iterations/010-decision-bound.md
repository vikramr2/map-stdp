# Iteration 010: a decision-bound action readout (spiking Proposal)

**Date:** 2026-10-21.

- **Model:** v4 ([`model.md`](../model.md)); unchanged. The new readout is a spiking Proposal, default off.
- **Compute:** local Linux workstation (72 cores), no SLURM. Runs went through the repo CLI in parallel (`xargs`, `OMP_NUM_THREADS=1`).
- **Agents:** `neuroscientist` (design, before any code); `snn-expert` (implementation, ceilings, sharpness measurements, pilot).
- **Status:** done. Flow: no gain, so it was stopped at the gate. Spiking: the MountainCar pilot reaches flow parity, but CartPole regresses; the readout is not yet adopted.

Return is the mean of the last 100 episodes, mean ± sd over seeds (ddof = 1); p is Welch's two-sided t-test. Ceilings are level-1 CEM (`experiments/ceiling.py`), each the mean of 100 fresh episodes, mean ± sd over 3 CEM seeds.

## 1. Goal

Iteration 009 found that v4 sits 10–25 below the ceiling of its own readout, Eq. 1e with adaptive $\beta$ (6–8 on MountainCar). The spiking backend has no matched sharpened readout at all: a race to $k$ spikes reaches only $\beta_{eff} \approx 1.5$. This iteration asks for a bioplausible readout whose sharpness tracks $\beta$, and checks whether its three-action form beats Eq. 1e in flow.

## 2. Design (`neuroscientist`, before any code)

- **A race on spike-count differences to a bound $\theta$** (derivation Eq. 1h). For two actions the lead is a ±1 walk absorbed at ±$\theta$, so by gambler's ruin it is Eq. 1e with $\beta = \theta$ exactly. Sharpness is linear in $\theta$; a race to $k$ spikes grows only as about $\sqrt{k}$. The cost is about $\theta^2$ spikes near a tie, plus a deadline.
- **Biology.** A discrete drift-diffusion with mutual inhibition through a shared inhibitory pool (Dale's law holds). A uniform gain on the action modules does not sharpen a race, so $\beta$ maps onto a decision threshold set by cortico-striatal and subthalamic circuits more directly than onto noradrenergic gain; locus coeruleus is only a candidate.
- **$K > 2$.** The basal-ganglia MSPRT [Bogacz & Gurney 2007] is the principled form, tested beside (1h). A Wang-type attractor was kept as a fallback only.
- **Gates.** Step 1 (flow ceilings) gates flow training: proceed only if (1h) or MSPRT at $K = 3$ beats Eq. 1e at matched $\beta$ by more than the CEM noise. Step 2 (spiking sharpness on trained weights) gates a spiking pilot: proceed if $\beta_{eff} \ge 6$.

## 3. What was built

All default off; golden 9/9 bit-identical, `ruff` clean, `mapstdp.spiking` self-check extended (deterministic raster, per-head cap, a $K = 2$ gambler's-ruin Monte Carlo).

- `mapstdp/spiking.py`: `diffbound(S, groups, theta, rng, burn, cap, heads)`. Optionally counts at most `cap` spikes per module, per head and per step.
- `experiments/ceiling.py`: readouts `beta<B>` (generalised), `db<T>` and `msprt<T>c<C>`, and `--nmax` (deadline in action spikes). Bound readouts draw only on their own path, so the existing `1d`, `argmax` and `beta4` draws are unchanged. Results gain `spikes_mean`, `deadline_frac` and `R`.
- `experiments/m2.py`: `Agent(readout='diffbound')`, spiking only. $\theta$ comes from $\beta$ through `BOUND_CAL`, the measured $\beta_{eff}(\theta)$ table at 16 heads (log-log interpolation, a fractional $\theta$ randomised between floor and ceiling; one extra uniform per frame, only on this path). Adaptive $\beta$ (Eq. 1g) is allowed with it.
- `experiments/m3.py`: `--readout {race,diffbound}`, file token `dbc`; `nondefault()` excludes it from the summary.

## 4. Results

### 4.1 Derivation checks

For $K = 2$, the exact absorbing chain gives (1e) with $\beta = \theta$ to $2 \times 10^{-15}$, and the expected count $\theta \tanh(\theta x) / \tanh x$ ($\theta^2$ at a tie) to $10^{-12}$. For $K = 3$, the KL from (1e) at $\beta = \theta$ is at most 0.0065, and the fitted $\beta_{eff}$ on trained v4 states is slightly above $\theta$ (8.08–8.18 at $\theta = 8$). MSPRT is never closer to (1e) than (1h). Details are in derivation Appendix A.

### 4.2 Step 1: flow ceilings (gate fails)

| | MountainCar | Acrobot |
| --- | --- | --- |
| Eq. 1e $\beta = 4$ | −120.8 ± 1.2 | −87.4 ± 2.1 |
| (1h) $\theta = 4$ / MSPRT | −119.0 ± 5.4 / −119.4 ± 11.3 | −90.6 ± 4.2 / −87.8 ± 3.6 |
| Eq. 1e $\beta = 8$ | −113.1 ± 3.7 | −84.0 ± 1.4 |
| (1h) $\theta = 8$ / MSPRT | −111.4 ± 1.6 / −113.8 ± 4.4 | −84.1 ± 1.8 / −84.1 ± 7.3 |
| Eq. 1e $\beta = 16$ | −110.3 ± 2.9 | −80.4 ± 7.3 |
| (1h) $\theta = 16$ / MSPRT | −110.3 ± 2.1 / −110.1 ± 2.2 | −79.8 ± 2.6 / −80.8 ± 5.6 |

All Welch p ≥ 0.5 against Eq. 1e at matched $\beta$ (at $\beta = 4$ computed by the agent with the seed-0 rerun; with the committed seed 0 the differences stay within 3.5). MSPRT uses $c = 1$. Action spikes per decision for (1h) at $\theta$ = 2/4/8/16 are 3/8.6/23/65 (MountainCar) and 3/7.6/21/43 (Acrobot); the deadline at 1024 spikes is hit in at most 0.07% of decisions. **Flow training was not run.**

The $\beta = 4$ row uses seeds 1–2 rerun here and the committed seed 0 of iteration 005. A rerun of seed 0 on this machine does not reproduce the committed value (MountainCar −115.2 against −121.6). Its training return differs too, and training never calls the readout, so the drift is at machine or BLAS level and predates this change. The committed files were kept; the reruns are in scratch `010/rerun/`.

### 4.3 Step 2: spiking sharpness (gate passes)

Measured on trained v4 weights (seeds 0–2, retrained; all 12 reproduce the committed last-100 exactly), 20 held-out stimuli × 200 frames, $\beta_{eff}$ fitted by maximum likelihood against flow Eq. 1e.

- **Why the old race could not sharpen.** Action modules are nearly closed (persistence 0.992–0.996), so each entry is a cascade of about 5 spikes with a heavy tail. Per-frame action counts are overdispersed by $F$ = 14–15 at one head and 17–21 at 4 or more heads, and $\beta_{eff} \approx \theta / F$. The mean spike shares match the flow (L1 0.003–0.04; argmax agrees on 90–100% of stimuli), so only the readout was wrong.
- **The fix.** Counting at most one spike per module, per head and per step (a saturating readout synapse) lowers $F$ by 1.5–2 times. Counting entries only is the most spike-efficient, but its count saturates within a frame and hits the deadline.
- **Range with the cap, 16 heads, 100 steps** (mean of 4 tasks; `BOUND_CAL`):

| $\theta$ | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\beta_{eff}$ | 3.1 | 3.5 | 4.0 | 5.0 | 6.5 | 9.2 | 11.5 |
| deadline fraction | 0 | 0 | 0 | 0 | 0 | ≤ 0.04 | 0.10–0.24 |

- At 4 heads the top is about 6.4 at 100 steps (Acrobot deadline 34%), or 7.5–10 at 250 steps. At one head it is 2.6–3.1 (100 steps) and 4.6–5.4 (250 steps).
- **The floor.** At 16 heads, $\theta = 1$ already takes a majority over the first step's pooled spikes, so the readout cannot go below $\beta \approx 3.1$.
- **Latency** (16 heads, cap). The median is 4–8 steps at $\theta = 16$ (p90 10–23) and 9–16 at $\theta = 32$ (p90 17–57), roughly 5–140 ms at 1–2.5 ms per step.
- **Cost.** At 16 heads a frame takes 6.4–8.1 ms against 0.8–1.2 ms for flow (27–30 min per run against 4–8), with 2300–3200 action spikes per frame. On hardware it means 16 crossbar copies or 16-fold time-multiplexing.

### 4.4 Step 3: spiking pilot

16 heads, seeds 0–4, 1500 episodes, flags `--td-norm 0.04 --p-cc 0 --eta 5`. There was no iteration-005 spiking baseline at these flags on disk, so a race baseline was run under the same flags.

| Task | Race | (1h) + adaptive $\beta$ | p | Flow v4 (seeds 0–4) |
| --- | --- | --- | --- | --- |
| MountainCar | −177.0 ± 1.8 | **−132.2 ± 2.8** | 1.4e-8 | −132.2 |
| CartPole | 298.3 ± 12.7 | 153.9 ± 51.2 | .0025, worse | 499.9 |

- **MountainCar** reaches flow parity, the first task on which spiking does. Mean $\theta$ 15.4, final $\beta$ 7.1, deadline fraction ≤ 0.004.
- **CartPole** regresses. Mean $\theta$ 5.3, final $\beta$ 3.6; $\beta$ wanders between 2 and 10 and never climbs. At fixed $\beta$: 4 gives 186.7 ± 43.3; 8 gives 127 ± 162, with 3 of 5 seeds locked at return 9.3. CartPole needs $\beta$ to reach about 16, and intermediate $\beta$ hurts even in flow (an adaptive $\beta$ capped at 4 gives 263, against 308 at $\beta = 1$). Here the readout spans only 3.1–11.5, and below 3.1 it is sharper than the score (4b) assumes.
- **Matched-$\beta$ cross-check.** The spiking race matches flow at $\beta = 1$ on both tasks (298 against 308; −177 against −174). So the spiking learning signals were already at flow parity, and the readout was the missing piece.

## 5. Conclusion

- **Flow:** (1h) and MSPRT have the same ceilings as Eq. 1e; v4 is unchanged.
- **Spiking:** (1h) with a per-head cap is the first matched sharpened spiking readout. It is kept as a Proposal, not adopted, until CartPole is fixed and a 20-seed test confirms MountainCar.
- **Dropped for now:** MSPRT (no gain at $K = 3$) and the attractor fallback (the conditions for it did not arise).

## 6. Next

- **Spiking.** Lower the floor with fewer readout heads at small $\beta$ (readout heads $H_r = f(\beta)$), and raise the top toward 16 with 250 steps or onset-style (short-term-depression) counting; test them separately to tell whether the floor or the top breaks CartPole. Then pre-register all four tasks over seeds 0–19 against the race at 16 heads. Open: does `BOUND_CAL` drift with module persistence during training?
- **Flow (iteration 011).** Diagnose why v4 misses the Gym thresholds before changing the rule: v4 weights under a sharper readout, why adaptive $\beta$ stalls (MountainCar median 6.6, never at the cap), Acrobot's good against bad seeds, the MountainCar encoder ceiling with 3 CEM seeds per grid, and the trained policy against the CEM policy.

## Commands

```bash
python -m experiments.ceiling --task T --level 1 --readout R --seed S     # R in beta4 beta8 beta16 db2..db16 msprt2c1..msprt16c1; T MountainCar, Acrobot; S 0-2
B="python -m experiments.m3 --task T --backend spiking --heads 16 --td-norm 0.04 --p-cc 0 --eta 5 --seed S"
$B                                                                        # race baseline
$B --readout diffbound --beta-kappa 0.01 --beta-max 16                    # (1h) + adaptive beta
$B --readout diffbound --beta 4                                           # fixed beta (CartPole diagnostic; also 8)
```

T ∈ {CartPole, MountainCar}, S = 0–4 for the pilot. Scratch in the session scratchpad `010/` (derivation checks `exact.py`, `check_k2.py`, `check_k3.py`, `k3kl.py`, `burst.py`; sharpness `sharp.py`, `sharp_table.py`, `steps.py`; tables `gate.py`, `pilot_table.py`; `W/` trained weights; `rerun/`) and an independent check `010mine/k2.py`.
