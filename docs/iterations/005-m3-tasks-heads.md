# Iteration 005: M3 tasks, spiking heads, and what limits the learner

**Date:** 2026-10-20.

- **Model:** v3 ([`model.md`](../model.md)) with the opened latents: the latent-output barrier (Eq. 4e) at $\beta_{LA} = 0.3$, the user's choice over 0.15. New Proposals behind flags: parallel spiking heads, TD-error normalisation, and the M3 task encoders.
- **Compute:** local Mac workstation (16 cores), no SLURM. Runs were launched with `xargs -P 14` over the repo CLI. Logs are in `logs/` (gitignored); results are in `experiments/results/m3_*.json` and `ceiling_*.json`.
- **Status:** in progress. The spiking heads sweep has not been run; it is on hold until the flow reference is worth approaching (§5).

All numbers are preliminary: flow level, 5 seeds, return as the mean of the last 100 episodes, mean ± sd over seeds, unless marked seed 0.

## 1. Goal

The user's two questions:

1. How does the current model (v3 + opened latents) do on CartPole, MountainCar, Acrobot and LunarLander (SPEC §4 task ladder)?
2. Spiking underperforms flow, plausibly because flow is the mean field of spiking. Can an **ensemble of parallel spiking heads** that infer together approach the flow agent?

## 2. What was built

| Piece | Where | Notes |
| --- | --- | --- |
| Task registry | `experiments/tasks.py` | Per task: module sizes (controller, 2 latents of 16, one action module of 16 per action), encoder, reward fed to Eq. 4c, default episodes. |
| Encoders (Proposal) | `tasks.py`, `core.rbf_stimulus` | CartPole: the M2 6×6 grid, bit-identical. MountainCar: 6×6 grid over (position, velocity) normalised by the bounds, $\sigma = 0.4$. Acrobot, LunarLander: 64 RF centers drawn from 20000 random-policy states, observation z-scored by that pool, $\sigma$ the median nearest-center distance. Fixed encoder seed for all agent seeds. |
| Rewards | `tasks.py` | CartPole −1 on termination (M2). MountainCar and Acrobot $0.1 r$, LunarLander $0.01 r$, so failing episodes give TD errors of order 1. |
| Spiking heads (Proposal) | `spiking.tile_heads`, `head_groups`, `pooled`; `m2.Agent(heads=H)` | H independent replicas on the shared W (own noise and teleportation draws). Spikes are pooled: $\hat{\pi}$ from summed counts, $\kappa$ the sum of per-head covariance counts divided by total spikes, one race over all heads' action groups. One plasticity update per frame; every head gets the new W. H = 1 is the M2 agent. |
| Batched engine | `spiking.run_heads`, `m2.Agent(engine="numpy")` | In branching mode the SuperNeuroMAT LIF step is $z_t = [\mathrm{inp}_t + z_{t-1} W_{snm} > \theta]$; all heads share one $N \times N$ product per step. Same rng draws as `run_frame`; the `mapstdp.spiking` self-check requires the identical raster. Closed-loop runs are identical on both engines (CartPole and LunarLander, returns, W and KL). M3 default; M2 stays on SuperNeuroMAT. |
| SuperNeuroMAT input fast path | `spiking.set_inputs` | Writes the input array directly instead of SuperNeuroMAT's pure-Python dict loop (half the frame time). `snn.sparse = False` in `build`: 'auto' goes sparse above about 150 neurons and the jit backend is dense-only. |
| TD normalisation (Proposal) | `m2.Agent(td_norm=s)`, `--td-norm` | Task term uses $\delta s / \mathrm{RMS}(\delta)$, running mean square over 1000 frames; the critic learns from the raw $\delta$. One scalar gain on the broadcast modulator. |
| Diagnostics | `m2.Agent.diagnostics()` | Logged with every eval: $\lvert \delta \rvert$, RMS, task-term $\lvert \delta \rvert$, critic V range, share of synapses at $w_{min}$ / $w_{max}$, held-out Eq. 1d policy mean, entropy and mean largest probability. Never touches rng or W. |
| Ablation flags | `m2.Agent(ablate=...)`; `m3 --eta-tau --wmin --tau-e --critic` | `critic rich`: 256 RFs on the full observation, outside the network (diagnostic only). |
| Policy-class ceiling | `experiments/ceiling.py` | §6. |
| Fixes | `m2.run` | Frame count was `sum(returns)`, wrong for negative-reward tasks; now counted. CartPole unchanged (373066 = 373066). |

The behaviour lock: `tests/golden.json` fails on this Mac **even at a clean HEAD** (all 5 closed-loop cases differ in the last bit; the file was generated on the cluster with a different BLAS). It was not regenerated. Instead, fingerprints of a clean worktree of HEAD and of the working tree were generated on this machine and compared: bit-identical after every change in this iteration. Long Mac runs drift from committed cluster runs after 57–553 episodes for the same reason (M3 CartPole 271 ± 21 against M2 `--la-beta 0.3` 279 ± 22).

Commands:

```bash
python -m experiments.m3 --task Acrobot --backend flow --seed 0 --td-norm 0.04      # one run
python -m experiments.m3 --task Acrobot --backend spiking --heads 8 --seed 0        # spiking, numpy engine
python -m experiments.m3 --fidelity        # pooled pi-hat and kappa error against flow, per task and H
python -m experiments.m3 --summary         # per task: random, flow, spiking per H
python -m experiments.m3 --ablations       # learning-rule ablations
python -m experiments.ceiling --task CartPole --level {learned,2,1} [--readout {1d,beta4,argmax}]
python -m experiments.ceiling --summary
```

## 3. Spiking heads: signal fidelity

The race already samples Eq. 1d exactly (derivation §2.4), and a pooled race over H heads is still Eq. 1d, so heads do not change the policy. What they change is the noise in the learning signals: $\hat{\pi}$ feeds the critic and the eligibility score, $\kappa$ feeds the eligibility and structural terms. `m3 --fidelity`, initial W, 10 held-out stimuli × 20 frames (`m3_fidelity.json`):

| H | CartPole L1($\hat{\pi}$, $\pi$) | 1 − corr($\kappa$, J) | Acrobot L1 | 1 − corr | LunarLander L1 | 1 − corr |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.351 | 0.713 | 0.437 | 0.797 | 0.448 | 0.784 |
| 4 | 0.175 | 0.487 | 0.214 | 0.607 | 0.222 | 0.599 |
| 16 | 0.089 | 0.239 | 0.107 | 0.352 | 0.110 | 0.339 |

The $\hat{\pi}$ error falls exactly as $1/\sqrt{H}$, so the ensemble converges to the mean field as intended. Whether that turns into return is the sweep (§8).

Wall-clock per frame, H = 16: SuperNeuroMAT block-diagonal 54–109 ms (dense $(HN)^2$ product, 75% of the time in `lif_jit`); numpy engine 7.6–8.7 ms. H = 1: 3.3–4 ms against 1.2–1.6 ms.

The readout question is open: the pooled race is the mean-field-matched readout; a majority vote over per-head races converges to argmax $p_a$, a different, greedier policy, and would be a labelled ablation.

## 4. Flow reference on the four tasks

| Task | Random | Flow, v3 + $\beta_{LA}$ 0.3 | Flow, + `--td-norm 0.04` |
| --- | --- | --- | --- |
| CartPole-v1 | 22.8 ± 12.9 | 271.3 ± 20.9 | 274.6 ± 26.0 |
| MountainCar-v0 | −200 | −186.1 ± 7.1 | −181.6 ± 2.0 |
| Acrobot-v1 | −499 ± 13 | **−373.6 ± 155.7** (−500, −158, −500, −210, −500) | **−128.4 ± 6.7** |
| LunarLander-v3 | −184 ± 118 | −35.6 ± 7.7 | −40.6 ± 6.8 |

Episodes: 1500 (CartPole, MountainCar), 1000 (Acrobot, LunarLander). Flow costs 0.65–1.26 ms per frame.

**Acrobot collapse (diagnostics, seed 0).** The applied TD error was about 0.15 per frame against CartPole's 0.014: the reward is dense (−0.1 every frame), while $\eta = 10$ was calibrated on CartPole's one −1 per episode. Homeostasis lost ($d_{max}$ up to 3, against $\le 1.04$ in M2 v3). Around episodes 250–300 the agent fell into a policy that never reaches the goal (still mixed, 1.22 bits, not a one-action collapse). Then every frame gives −0.1, the critic settles at $V \approx -0.1 / (1 - \gamma) = -10$, $\delta \to 0$ and learning stops. With `--td-norm 0.04` (CartPole's measured RMS $\delta$ is 0.043) the task $\lvert \delta \rvert$ is 0.016, $d_{max}$ stays at 1.00 and no seed collapses.

**MountainCar** is the same trap from the start: the goal is almost never reached, so $\delta$ carries no information. Normalisation cannot fix that; it is exploration and credit.

**Every policy is very stochastic** at the end of training: mean largest action probability 0.76 (CartPole), 0.56–0.63 (3- and 4-action tasks), 1.1–1.4 bits.

## 5. Are the latents the problem?

The user's hunch: the latents should store memory about the task. Two observations against memory being the bottleneck here:

- All four tasks are fully observed (velocities are in the observation), so reactive policies solve them. Frames cold-start ($\rho = 0$), so latents cannot carry memory across frames anyway; that is M4.
- At the flow level the policy family is fixed, whatever the latents do. The flow is linear in the stimulus, so the action shares are $p_{\mathcal{A}} = R v_{\mathcal{C}}(\mathbf o)$ with $R$ a nonnegative actions × controller matrix, and

$$
P(a \mid \mathbf o) = \frac{(R v_{\mathcal{C}}(\mathbf o))_a}{\sum_{a'} (R v_{\mathcal{C}}(\mathbf o))_{a'}}
$$

Latents only change which $R$ the weights produce; they add no nonlinearity. For communities to act as states, activity would have to commit to one community (for example winner-take-all between latents through per-module inhibition, derivation §7). Recorded as a direction, not tested.

## 6. Policy-class ceiling (seed 0)

`experiments/ceiling.py`. The Map-STDP flow agent (v3, $\beta_{LA}$ 0.3, `--td-norm 0.04`) is trained, then:

- **learned:** its $R$ frozen, scored on 100 fresh episodes;
- **level 2:** cross-entropy method (CEM: population 64, 8 elites, 80 iterations, 4 episodes per candidate, common random numbers) over the controller→action synapses of the learned W, within $[w_{min}, w_{max}]$, Eq. 1d;
- **level 1:** CEM over any $R = \exp(\theta)$, readouts Eq. 1d, Eq. 1e ($\beta = 4$) and argmax.

The $R$ of level 2 is computed from W with the same 20-hop operator as `core.flow`; the script asserts it reproduces the agent's Eq. 1d policy.

| Policy | CartPole | MountainCar | Acrobot | LunarLander |
| --- | --- | --- | --- | --- |
| Map-STDP learned, frozen | 205 | −183 | −133 | −26 |
| Level 2, Eq. 1d | 215 | −183 | −132 | −21 |
| **Level 1, Eq. 1d** | **465** | **−150** | **−90** | −38 |
| Level 1, $\beta = 4$ | 500 | −122 | −85 | 61 |
| Level 1, argmax | 500 | −109 | −81 | 43 |
| Learned $R$, argmax readout | 264 | −127 | −109 | −108 |

The last row and the checks below come from a session scratch script, not the repo CLI.

- **The family has headroom.** With the model's own readout, a good $R$ gives 2.3× the CartPole return, +33 on MountainCar, +43 on Acrobot. LunarLander's learned $R$ already matches the free Eq. 1d optimum.
- **The network can express sharp votes.** With controller→action synapses at the bounds, each controller cell's preferred-action share is 0.98 (learned: 0.77 CartPole, 0.82 MountainCar). Turning off latent→action, cross-action and controller recurrence changes that by at most about 1%; flow into action modules comes 18% from the controller, 2–3% from latents and 80% from action-module recurrence. Level 2 stayed flat because CEM over hundreds of noisy per-synapse logits barely moves $R$: a search failure, not a cap. Not proven: that the network can produce exactly level 1's $R$.
- **Binarising the learned votes hurts** (CartPole 71): the right policy is a graded mix across cells, not more contrast.
- **Conclusion:** Map-STDP converges to a poor $R$; Eq. 1d's stochasticity is a second, separate cap (largest on LunarLander, +80–100).

## 7. Learning-rule ablations (Option A)

Flow, `--td-norm 0.04` baseline, 5 seeds (`m3 --ablations`):

| Variant | CartPole | Acrobot | Mean largest P (CartPole / Acrobot) |
| --- | --- | --- | --- |
| Baseline | 274.6 ± 26.0 | −128.4 ± 6.7 | 0.76 / 0.61 |
| `--eta-tau 0` (constant $\eta$) | 230.3 ± 19.9 | −133.7 ± 11.8 | 0.77 / 0.60 |
| `--eta-tau 1500` | 254.1 ± 10.3 | −142.3 ± 22.7 | 0.77 / 0.60 |
| `--tau-e 10` | 289.0 ± 15.1 | −128.2 ± 4.5 | 0.76 / 0.59 |
| `--tau-e 30` | 280.7 ± 16.3 | −137.0 ± 9.7 | 0.76 / 0.57 |
| `--wmin 0.003` | 230.4 ± 9.2 | −138.7 ± 8.0 | 0.75 / 0.59 |
| `--wmin 0.01` | 114.5 ± 10.9 | −177.7 ± 10.0 | 0.69 / 0.52 |
| `--critic rich` | 222.9 ± 27.5 | −162.9 ± 25.3 | 0.76 / 0.59 |

- **None closes the gap** to the level-1 ceiling (465, −90).
- **The weight floor helps.** Raising it lowers return in proportion and raises entropy: the learner sharpens by pruning wrong-action synapses to the floor. The multiplicative-floor hypothesis had the sign backwards.
- **Policy sharpness is pinned** at about 0.76 (CartPole) and 0.60 (Acrobot) in every variant with the default floor: annealing, credit window and critic change return, not sharpness.
- **A richer critic does worse**, plausibly because 256 features learn more slowly at the same NLMS rate.
- `--tau-e 10` is a small CartPole gain, not significant over 5 seeds.

## 8. Decisions and next steps

- **Adopted as Proposals, defaults unchanged:** heads (`--heads`), the numpy engine (`--engine`, M3 default), `--td-norm` (candidate M3 default at 0.04: it fixes Acrobot and is neutral elsewhere), the M3 encoders and rewards.
- **Rejected as fixes:** constant or slower annealing, a higher weight floor, longer eligibility, a richer critic.
- **Next, agreed in principle, not yet built: an ideal learner in the same class.** Actor-critic directly on $R = \exp(\theta)$ with Eq. 1d, the same critic, reward, trace and rate, but the exact log-policy gradient. If it reaches about 465 on CartPole, Map-STDP's update (fixed-flow Eq. 4b, per-synapse multiplicative scaling, structural and homeostasis interference) is the gap; if it also stalls near 275 with the same pinned sharpness, the TD actor-critic recipe is the limit and the fix is at the algorithm level (advantage normalisation, sharpened readout, exploration).
- **Then:** the spiking heads sweep, 4 tasks × H ∈ {1, 2, 4, 8, 16} × 5 seeds on the numpy engine, about 3–4 h locally; the readout choice (pooled race, and optionally majority vote as an ablation) is for the user.
- **Environment:** `box2d-py` and `pygame` added to the `map-stdp` env and `environment.yml` (LunarLander).
