# Iteration 013: TD-STDP backbone on the branching engine (step 1)

**Date:** 2026-10-23.

- **Model:** none. This is a baseline, not Map-STDP; `docs/model.md` and `docs/derivation.md` are unchanged.
- **Plan:** `jiggly-chasing-sparkle` (user-approved), step 1 of 4: reproduce the feed-forward TD-STDP actor-critic of Chung & Kozma 2020 [CK] (arXiv:2008.13044) on our engine. The later steps are removing Adam, action modules with a race readout, and QIF.
- **Compute:** local Linux workstation (72 cores, at most 40 workers, `OMP_NUM_THREADS=1`), no SLURM. Every number below comes from `python -m experiments.tdstdp --summary`.
- **Agents:** `snn-expert` (implementation, runs). The neuroscientist filter is deferred to step 3, because this step deliberately reproduces off-hardware parts: softmax, Adam, batch, signed weights and the per-synapse trace pair.
- **Status:** E1 failed its gate. The true-LIF control fails the same way. E2 (Adam) learns. E3 and E4 (LunarLander) are still running; see §5.6.

## 1. Goal

Find out whether CK's backbone learns on the SuperNeuroMAT branching engine before building modules on it. The pre-registered gate (E1) is CK's CartPole configuration without Adam, reaching at least 18/20 solved runs and a median $T_s$ of at most 340 (2× CK).

## 2. What was built

- `mapstdp/tdstdp.py` (new):
  - `Layout`: the post populations, ordered critic E, critic I, then actor E and I per action.
  - `simulate`: the `spiking.run_heads` branching step without presynaptic normalisation, $P(\text{spike}) = \mathrm{clip}(W x(t-1))$ with $2\theta = 1$.
  - `simulate_snm`: the same step on SuperNeuroMAT.
  - `simulate_lif`: the true-LIF control. CK Eq. 1 with an exponential leak, or SuperNeuroMAT's subtractive leak.
  - `pre_trace`, `rates`, `softmax`, `entropy_coef` (CK Eq. 25).
  - `zsum`/`block_learn`: the z and q traces (CK Eqs. 72, 76–77) and the update terms (Eqs. 73, 78, 26, 31), summed in closed form over a block of L ms with W frozen. L = 1 is CK's per-ms rule exactly.
  - `Adam`: block form with per-ms-matched moments, see §3.
  - The `__main__` self-check, registered in `tests/test_regression.py`.
- `experiments/tdstdp.py` (new):
  - `Config` and `PRESETS`: CK Table 1 and §4 in CK units.
  - `Runner`: H env copies on shared W, run in 20 ms ticks.
  - `calibrate`: the E4 rule.
  - `solve_times`, `run`, the CLI and `--summary`.
  - Results go to `experiments/results/tdstdp_<task>[_<tokens>]_s<seed>.json`.
  - Features come from `tasks.critic_layer`: CartPole `f2c4hb` (81 features, one input neuron each) and LunarLander `f1c1b` (17 features, 16 input neurons each).
- Golden is bit-identical. `pytest -q tests` passes (8 tests) and `ruff` is clean.

## 3. Decisions and deviations from CK

These were fixed before E1 unless marked otherwise.

1. **Weight units.** Our W is a spike probability per input spike. CK's Eq. 1 and Eq. 44 move the membrane by $Rw/\tau$ per spike, against a gap of 13 mV and τ = 100 ms. A high-rate CK neuron therefore fires once per 1300 units of input, and we divide η, the Adam step and $1/c_w$ by `MV` = 1300. A first debug reading used 13, the BindsNET convention of an input adding w in mV. With it, one terminal TD error moved the actor weights by about 25 times their size and destroyed the network within one episode. That run (seed 0, 60 episodes or fewer, both readings) was seen before the choice was fixed. The `lif` control below uses Table 1 verbatim in CK units, so it checks this conversion.
2. **Weight init** (pre-registered): $W \sim U(0, w_0)$ with every post at a 0.2 per ms spike probability for the mean input. The mean input is taken over 5000 random-policy states (seed 777). Weights are signed and unclipped, as in CK.
3. **Batch 16** (pre-registered as 16 env copies with averaged updates): each copy runs its own episodes asynchronously. The update is averaged over the copies learning in that block. Episode e's return is the mean of the copies' e-th returns, so 3000 episodes means 3000 per copy. $T_f$ is the first e in which some copy scores at least 200. $T_s$ is computed on the copy-mean series.
4. **Timing.** Each env step is a 20 ms tick. The action is sampled at the start of the tick from the current softmax (every 1 or 2 steps), the env is stepped, and the pre-step observation is presented for 20 ms with the step's scaled reward spread evenly. The TD error $\delta_t$ (CK Eq. 69) uses $V_{t+1}$, so the increment of ms t is applied one ms later, paired with the traces at t. Target $V := 0$ in the last 2 ms of an episode, on termination or truncation, as CK do. There are 100 ms of warm-up per episode (5 ticks) with no learning.
5. **Block form.** CartPole runs at B = 1, which is CK's rule exactly. LunarLander runs at B = 20, freezing W for one env step and summing the 20 per-ms increments exactly; at B = 1 it would cost about 10× more (0.5 s per tick).
6. **Block Adam** (changed after E2, before any E3 result): the first form fed the block sum G to Adam with $\beta^{20}$ and a 20× step. On CartPole at B = 20 it collapsed 3/10 seeds against 0/10 for per-ms Adam (MW p 0.064), because for noise-dominated per-ms updates it makes $m/\sqrt{v}$ about $\sqrt{20}$ times larger than per-ms Adam does. The adopted form counts t in ms, feeds $G/L$ to m and $G^2/L$ to v with weights $1-\beta^L$, and steps $L \cdot lr \cdot \hat{m}/\sqrt{\hat{v}}$. It is identical to Adam at L = 1, and E2 reproduces bit for bit. The acceptance gate, set before the run, was CartPole seeds 0–9 at B = 20 against E2 with MW p > 0.05 and no extra collapses. It passed (§5.4). The first E3 launch used the old form and was stopped at episode 50 with random-level returns. Its results are in the session scratchpad (`a1_block_adam/`), not in the repo.
7. **E4 rule (Adam-equivalent step at init):** per population (critic, actor), η = Adam step / RMS of the per-ms update averaged over copies. The RMS is measured per ms (B = 1) with the initial W frozen, over 200 ticks on its own rng stream. An RMS-sized per-ms update then moves W by the step that per-ms Adam takes at init. The first launch used the block-update RMS times B, which is $\sqrt{B}$ larger; it was corrected together with item 6, again before any E4 result.
8. **Not specified by CK and chosen here:** LunarLander-v3 (CK used v2); the Fourier observation bounds (`tasks.FOURIER_BOUNDS`); the rate filter $\rho \leftarrow e^{-1/\tau_n}\rho + X/\tau_n$; posts driven by the input spikes of the previous ms (delay 1); and $T_s$ as the last episode of the first qualifying 100-episode window (consistent with CK's Fig. 2).
9. **True-LIF control.** SuperNeuroMAT's leak is subtractive (a fixed amount toward reset per step), not CK's exponential leak. The control therefore has two variants: `lif` (CK Eq. 1, numpy) and `lif-snm` (SuperNeuroMAT's leak, verified spike for spike against SuperNeuroMAT). Both have no noise and no refractory period, and use W in CK units.

## 4. Self-check (`python -m mapstdp.tdstdp`)

| Check | Result |
| --- | --- |
| (a) numpy branching step = SuperNeuroMAT raster (7 inputs, 11 posts, 300 ms) | identical, rate 0.394 |
| (a′) `lif-snm` = SuperNeuroMAT with a finite leak (2000 ms) | identical, rate 0.127 |
| (b) block closed form = per-ms loop, L = 1 and L = 20 | max relative error $4 \times 10^{-16}$ |
| (c) critic STE: Monte Carlo $E[z]$ against finite-difference $\partial E[V]/\partial W$ | cosine 0.9975 global, 1.0000 per post row |
| (d) actor rule on a 2-armed bandit: $E[r(A) q]$ against finite-difference $\partial E[r]/\partial W$ | cosine 0.9685 global, 1.0000 per row |
| closed loop, `--engine snm` against numpy (CartPole, 3 episodes) | identical returns and W |

In branching mode the STE (CK Theorem 1) is exact up to a per-postsynaptic gain $p_j$: $E[z_{ij}] \propto p_j f_i$, while the exact gradient is proportional to $f_i$. Rows are therefore exact, and the global cosine drops only as far as the post rates differ. The plan's claim that the STE becomes exact holds per row, not globally.

## 5. Results

The columns are as follows; all of them come from `--summary`.

- **Solved:** CK's criterion, 100 consecutive episodes at 500.
- **$T_f$:** the first episode at 500, mean ± sd over the runs that reach it [count].
- **$T_s$:** the last episode of the first solved window.
- **$T_{475}$:** secondary and not pre-registered. It is the first episode whose trailing-100 mean reaches 475 (Gymnasium's threshold), included because CK's $T_s$ censors almost every run.
- **Return:** the mean of the last 100 episodes, mean ± sd over seeds.
- **ms/frame:** wall time per env step per copy (20 ms simulated) with about 30–40 jobs on the machine.

CK, 10 runs: $T_f$ 57.0 ± 24.0, $T_s$ 169.5 ± 23.5, all solved. Model v4 on CartPole: flow 486.7 ± 56.3 (n = 40); spiking `h16 dbc` 153.9 ± 45.8 (n = 5).

### 5.1 E1 (gate: CK's CartPole, no Adam) and the true-LIF control

| Run | Seeds | Solved | $T_f$ | $T_s$ | Return | ms/frame |
| --- | --- | --- | --- | --- | --- | --- |
| E1, branching | 0–19 | 0/20 | 214.0 ± 121.0 [6] | – | 15.4 ± 16.4 | 6.95 |
| true LIF (`lif`, CK Eq. 1, Table 1 verbatim) | 0–4 | 0/5 | 229.0 ± 137.0 [2] | – | 18.4 ± 18.0 | 6.14 |
| true LIF (`lif-snm`, SuperNeuroMAT leak) | 0–4 | 0/5 | – [0] | – | 19.0 ± 14.0 | 6.47 |

**E1 fails the gate** (0/20 solved, median $T_s$ infinite), and CK's own neuron fails the same way, so the failure is not specific to branching mode. The failure looks like this:
- Six seeds reach 500 once, and most end on a constant action (return 9.4).
- The policy entropy falls to 0.00–0.05 nats out of 0.69.
- At an episode end, $\delta \approx -V \approx -0.2$ for 2 ms and $q \approx \tau_q \lvert A_k - s_k \rvert z \approx 800$. With Table 1's actor η, one terminal step moves the actor weights by 25–35%.
- The policy therefore flips between the two actions from episode to episode until the softmax saturates. Once saturated, $A_k - s_k \approx 0$ for both actions and learning stops: the absorbing state.

### 5.2 E2 (Adam, per ms) and the block-form gate

| Run | Seeds | Solved | $T_f$ | $T_s$ | $T_{475}$ | Return | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2, branching + Adam (B = 1) | 0–19 | 1/20 | 123.0 ± 30.2 [20] | 322 [1] | 258.6 ± 33.4 [13] | 377.8 ± 83.2 | 8.36 |
| branching + Adam, B = 20 (block-Adam gate) | 0–9 | 0/10 | 134.5 ± 38.3 [10] | – | 248.6 ± 33.4 [9] | 435.5 ± 42.9 | 0.76 |

- **E2 against E1:**
  - On $T_s$ (unsolved counted as 401) there is no difference, MW p 0.34, because both groups are almost all censored.
  - On return, MW p $6.7 \times 10^{-8}$.
  - All 20 E2 seeds reach 500, and none collapses.
  - E2 is about 2× slower than CK on $T_f$ (123 against 57), and only 1/20 meets CK's strict $T_s$.
- **Block-Adam gate:** B = 20 against E2 on seeds 0–9 (369.7 ± 83.9) gives MW p 0.14 with no collapses, so it passes. The rejected first form scored 236.1 on the same seeds with 3/10 collapsed (MW p 0.064, session scratchpad, §3 item 6).

### 5.3 Post hoc (not pre-registered)

| Run | Seeds | Solved | $T_f$ | $T_s$ | $T_{475}$ | Return | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true LIF + Adam (`--adam --engine lif`) | 0–9 | 2/10 | 131.7 ± 25.8 [10] | 316.5 ± 19.5 [2] | 295.4 ± 45.8 [8] | 464.8 ± 31.4 | 7.59 |
| SGD, η from the E4 rule (`--sgd`) | 0–9 | 0/10 | 144.0 ± 48.3 [10] | – | – [0] | 88.7 ± 130.8 | 5.72 |
| SGD, B = 20 | 0–9 | 0/10 | 137.3 ± 32.6 [10] | – | – [0] | 59.5 ± 87.2 | 0.77 |
| E1 with init rate 0.1, V(init) ≈ 0 (`--rate0 0.1`) | 0–4 | 0/5 | 288.2 ± 60.2 [4] | – | – | 17.6 ± 11.9 | 7.04 |
| same, true LIF | 0–4 | 0/5 | 231.0 [1] | – | – | 20.9 ± 14.2 | 7.14 |

- **Branching against CK's neuron under Adam (seeds 0–9):** the learning onset is the same ($T_f$ 126.0 against 131.7, MW p 0.62). The final return is lower on the branching engine (369.7 ± 83.9 against 464.8 ± 31.4, MW p 0.006). Branching mode carries independent escape noise on every post neuron, while the LIF is deterministic given its inputs.
- **SGD with the Adam-equivalent η:**
  - The calibrated η is 3.1e-3 for the critic (Table 1: 2.5e-3) and 1.1e-3 for the actor (Table 1: 1e-2, so about 9× smaller).
  - With it every seed reaches 500 once, so the converted Table 1 actor step accounts for much of the E1 failure.
  - Without Adam, 6/10 seeds at B = 1 and 7/10 at B = 20 still end in the absorbing state.
  - B = 1 against B = 20 gives MW p 0.62: the 20 ms weight freeze does not change the SGD outcome detectably.
- **Init rate 0.1:** this matches CK's Fig. 3 rates and removes the initial terminal TD error, but it does not rescue E1.

### 5.4 Cost

- 7–8 ms per frame for per-ms learning (B = 1) and 0.76 ms at B = 20, single-threaded, on CartPole.
- LunarLander (16 copies, 272 inputs, 512 posts, B = 20): 2.6 ms per copy-frame on an idle machine and 6.0 ms under the current load.
- SuperNeuroMAT (`--engine snm`) costs 98 ms per frame against 10 ms for numpy, with the same raster.

### 5.5 E3 and E4 (LunarLander), running

- Started 2026-10-23 at 11:13 EDT, local, 10 seeds each.
- Commands: `--task LunarLander --seed S` (E3, the CK preset: Adam with block moments, batch 16, entropy, decay) and `--task LunarLander --seed S --sgd` (E4).
- E4's calibrated η, in CK units, is 1.1–1.3e-4 for the critic and 0.92–1.02e-5 for the actor. The Adam steps are 1.25e-4 and 6.25e-5.
- At episode 100 per copy, both groups are at random level (−202.5 last-50 mean) with near-uniform policies ($H$ 1.36 of 1.39 nats).
- Progress is written to `experiments/results/tdstdp_LunarLander[_sgd]_s<seed>.partial` every 50 episodes. Final results are written as `.json` and enter `--summary`.
- At current episode lengths, 3000 episodes per copy takes at least 8.5 h, and longer if the landers learn to hover up to the 1000-step cap.
- Pass criteria (pre-registered): E3 solves at least 6/10, with $T_s$ reported against 2575; E4 is reported against E3.

## 6. Reading

- **The CK backbone learns on the branching engine**, but only with Adam. With per-ms Adam it reaches 377.8 ± 83.2 (all seeds reach 500; no collapse), about 2× slower than CK on $T_f$. Its final return is below CK's LIF on matched seeds (p 0.006).
- **CK's no-Adam CartPole configuration does not reproduce on either neuron model** under our reading of the units. One terminal TD error moves the actor by 25–35%, and the softmax saturates into an absorbing state.
- **Adam's per-synapse normalisation is load-bearing.** It is what step 2 (removing Adam) has to replace. An SGD step sized to Adam at init learns at first but still collapses in most seeds. A crossbar-native replacement needs a bounded or normalised step on the terminal TD error, or a policy that cannot saturate; this is a Proposal, untested.
- **Open:**
  - LunarLander (E3, E4).
  - Whether CK's weights really carry the 1/τ input scaling. The LIF control with Table 1 verbatim fails like the converted branching run, so the conversion is not the only problem.

## 7. Commands

```bash
python -m mapstdp.tdstdp                                                # self-check
python -m experiments.tdstdp --task CartPole --seed S                   # E1, S = 0..19
python -m experiments.tdstdp --task CartPole --seed S --engine lif      # true-LIF control (also lif-snm), S = 0..4
python -m experiments.tdstdp --task CartPole --seed S --adam            # E2, S = 0..19
python -m experiments.tdstdp --task CartPole --seed S --adam --block 20 # block-Adam gate, S = 0..9
python -m experiments.tdstdp --task CartPole --seed S --sgd [--block 20]        # post hoc, S = 0..9
python -m experiments.tdstdp --task CartPole --seed S --rate0 0.1 [--engine lif] # post hoc, S = 0..4
python -m experiments.tdstdp --task CartPole --seed S --adam --engine lif       # post hoc, S = 0..9
python -m experiments.tdstdp --task LunarLander --seed S                # E3, S = 0..9
python -m experiments.tdstdp --task LunarLander --seed S --sgd          # E4, S = 0..9
python -m experiments.tdstdp --summary
```
