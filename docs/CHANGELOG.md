# Changelog

## 2026-10-27

### Changed (study `emergent-v4`, CartPole only; not launched)

- `experiments/report.py`, `STUDIES["emergent-v4"]`:
  - CartPole only (1500 episodes, seeds 2000–2019);
  - arms on / off / sealed (`--lam 0 --p-cross 0`) on four engines, 12 groups, 240 runs;
  - 8 heads.
- `M3_ENGINE_FLAG` no longer fixes `--heads`; each study passes it.
- Pairs take an optional family, `(a, b, family)`, with Holm applied within a family. On vs off is `primary`; on vs sealed is `sealed`.
- The summary's pair table now has Group / Against / Family columns, and `experiments/structure.py` accepts 3-tuples and labels its pair table generically. The test is updated.
- **Added:** `m2.BOUND_CAL_QIF[8]` (0.67–4.26), from `experiments.bound_cal --engine qif --heads 8` on the pilot014 flow CartPole models. The same models at 16 heads give 1.01–5.11, within 11% of `BOUND_CAL_QIF[16]`. Without it, QIF could not run diffbound at 8 heads.
- `reports/emergent-v4/README.md` was rewritten as a draft pre-registration. It covers:
  - P1 (AUC) and P2 (L_obs), on vs off, Holm over 4 engines;
  - S, on vs sealed, its own Holm family;
  - frozen-W returns and structure as secondary results;
  - 32 workers, run from a frozen worktree.

### Results (iteration 014, second pilot; development)

- `docs/iterations/014-emergent-v4.md` §10: CartPole, 3 seeds × 500 episodes.
  - The map term raises AUC on every spiking engine (branching 94 → 204, QIF 79 → 182, LIF 174 → 235) and lowers L_obs on all four. Flow is unchanged in AUC.
  - Sealed (λ = 0, `--p-cross 0`) matches on for AUC, but on reaches lower L_obs than sealed on the spiking engines.
- **Decided:**
  - 8 heads adopted (same returns at half the cost);
  - LIF 20 ms window harmless, not adopted;
  - QIF homeostasis ×2 dropped.
- **Reversal (user decision):** the LunarLander part of the pilot was cancelled; 7 runs were stopped with checkpoints. The `emergent-v4` study becomes CartPole-only, and LunarLander moves to the TD-STDP backbone study (iteration 015).

### Changed (iteration 014 wording)

- `docs/iterations/014-emergent-v4.md` §7: the sealed control is reframed as learned versus prescribed structure (sealing is the minimum-description structure for the given partition), not structure versus sealing. The readings now cover sealed ≈ on, on > sealed (with candidate mechanisms), sealed > on, and what a sealed fix of the collapse implies, given that routing pairs are exempt in both arms. No change to the pilot or study design.

### Changed (canonical derivation: the emergent walk)

- `docs/derivation.md` is rewritten from the user's emergent-walk drafts (`docs/derivation-guide-emergent.md`, `docs/derivation-details-emergent.md`, PDF). The walk is observed from spikes: pair counts κ set the walk and the baseline ē^obs, a signed counter c sets each synapse's step (Lemma E, Eq. 9; Theorem E, Eq. 10), and homeostasis is −ε_h (d_j − 1) π_j W_ij / d_j (Eq. 4d). Section §5 is now the four engines (flow, branching, LIF, QIF) with their counters (11a–11c), and §7 the E–I stability condition (12). The K-series Proposal (K1–K9) is dropped. The flow engine is the special case a = X = 0 and its rule is unchanged.
- Equation map: 1–7, 4a–4h keep their meaning, rewritten for the observed walk (4b now uses the signed counter; 4d as above); new 8–12.
- Corrections to the drafts: the column-sum stability bound with inhibition is max(χ_E, χ_I) G_max < 1, not the sum (D10 Step 4); homeostasis pulls d_j toward 1 rather than conserving it; the branching counter is bounded by n_j |b_ij| / 2θ rather than proportional to W_ij, so Dale's law holds for small steps, with the w_min clip in code.
- **Open:** the mean counter (11b) and Eq. 4d make updates proportional to W_ij, which conflicts with the old crossbar rule against weight-proportional decay (§6, marked Open).
- Neuroscientist review (accepted with changes, folded in): c is presynaptic arrival × postsynaptic sensitivity (voltage-based plasticity), Siegert gain and homeostasis are tagged abstractions, the module-activity signal is own-module plus global activity (not a neuromodulator), chance caps M* and pulls ē^obs toward 1 − p_m, running estimates of ē^obs and M* over 10–50 frames are allowed.
- Checks (`/derivation-check`, new `fdcheck.py --emergent`, Appendix A F1–F4): Lemma E vs finite differences 3.8 × 10⁻⁷ (excitatory) and 3.7 × 10⁻⁷ (4 of 16 inhibitory); Theorem E cosine 1.0000000; reduction to the old Lemma 1.0 × 10⁻¹⁷; Monte Carlo of the sampled rule 0.999999 / 0.999998.
### Added (code: the emergent-walk rule, behind `rule="emergent"`; default `"v4"` keeps `tests/golden.json` bit-identical)

- `mapstdp/core.py`: `observed_walk` (Eq. 8, module rows only, silent columns fall back to T), `structural_update_emergent` (Eq. 4 with ḡ^obs), `homeostasis_update_emergent` (4d), `eligibility_step_emergent` (4b, one-hop credit from the observed walk), `module_offspring` (Eq. 12 monitor), `mstar_ceiling`, `lemma_e_grad` (Eq. 9); self-check adds Lemma E vs finite differences (4.4 × 10⁻⁷) and Theorem E.
- `mapstdp/spiking.py`: `run_heads(..., return_drive=True)` (no extra draws, raster unchanged) and `branching_counters` (exact counter, Eq. 11a; firing probability clip(u/2θ, 0, 1), forced external spikes contribute 0). Monte Carlo on frozen branching W: cosine 0.999995 with −E_j W ⊙ ∇L (κ as the step: 0.907).
- `experiments/m2.py`: Config `rule`, `lam_scale`, `eta_scale`, `obs_tau` (running observed-walk statistics), ablations `ebar` (`obs` | `module`) and `step_stat` (`c` | `kappa`); per-episode monitors (L_obs, M* and its chance ceiling, ē^obs, ρ(Λ̂^K), causal share, clip rate, silent frames, block weight shares); held-out L_obs in `evaluate`; `match_scales` (step sizes RMS-matched to flow at initial W); checkpoints every 50 episodes with resume (`resumes` in the result); `save_model` (W at episode 0, every 250 and final, float32 npz). Flow under `rule="emergent"` is bit-identical to v4.
- `experiments/m3.py`: `--rule`, `--obs-tau`, `--ebar`, `--step-stat`, `--lam-scale`, `--eta-scale`, `--match-scales`, `--out`, `--save-model`.
- `experiments/report.py`: studies take `runner` (`tdstdp` | `m3`); the m3 runner saves models under `reports/<study>/models/`, `summarize_m3` writes summary.csv, pairs.csv (Mann–Whitney with Holm per metric over declared pairs) and summary.md. tdstdp summaries are byte-identical (tested).
- `tests/test_emergent_rule.py` (10 tests).
- `mapstdp/lif.py` (new): noisy LIF engine, normalised units (θ = 1), τ_m 20 ms, dt 0.5 ms, OU background σ_V 0.2, 100 ms frames, window 60 ms with pairs straddling frames credited once, warm-up of 60 ms at episode start; counters `siegert` (default, Eq. 11b with 11c plus a discrete-time threshold correction), `psi`, `regression`; `siegert_gain` (also used by `tests/test_emergent.py`, whose import of the absent `test_eligibility` it replaces). **Offspring χG = 0.5, not 0.8:** the convex excitatory-only LIF ignites a high-rate state near 0.8. Checks: Siegert rate and gain within 1%; counter vs weight-scaled derivative corr 0.945, ratio 1.04; vs finite removal 0.94, 1.54; pair counts are about 99% chance; 17 / 29 ms per frame (CartPole / LunarLander, 16 heads).
- `mapstdp/qif.py`: emergent counter (Eq. 11b) with raw pair counts in the existing window (pairs straddling frames counted), E neurons only; gain from a running regression of rate on the injected noise current (total input gives the wrong sign, because the fixed pools respond to the neuron's own spikes), shrunk to module and mean-field gains. Checks: counter vs removal corr 0.89, ratio 1.04; gain vs finite differences corr 0.93, ratio 1.01. `--match-scales` should use ≥ 50 frames on QIF.
- `experiments/m2.py`, `m3.py`: `engine="lif"` (`--engine lif`, `--lif-<field>` overrides; requires `rule="emergent"`), burn-in 40 steps (20 ms), per-episode LIF state with any carried gain checkpointed; evaluation streams never touch training state. `match_scales` defaults to 50 frames on spiking engines (CartPole: LIF λ × 1.48, η × 1.09; QIF 0.98, 0.86; branching 1.26, 0.91).
- `experiments/bound_cal.py` (new): the iteration-010 diffbound calibration as a CLI (trained flow models, 20 held-out stimuli × 200 warm frames, β_eff by maximum likelihood to Eq. 1e). It reproduces `BOUND_CAL_QIF` within 6%. New `BOUND_CAL_LIF` (H = 16): β_eff 0.47, 0.61, 0.78, 1.06, 1.63, 2.20, 2.35 at θ = 1–64. **The LIF readout saturates at β ≈ 2.35** (background spikes reach every bound), and at θ = 64 69–84% of frames hit the deadline, so v4's `--beta-max 16` is unreachable on LIF.
- `docs/neuron-variants.md` rewritten for the observed walk: chance as teleportation, why two statistics, one section per engine with measured numbers, LIF and QIF limitations in one place, and a side-by-side table. `CLAUDE.md`: the derivation description, the `rule="emergent"` code map and commands.
- `experiments/structure.py` (new): from saved models, per snapshot the block weight shares, the K × K module weight table and L_W (map equation of the trained W's own walk under uniform teleportation, engine-free); on-vs-off Mann–Whitney per declared pair; heatmaps of the mean final module table per arm and on − off (`structure_<on>.png`). Smoke (flow CartPole, seed 0, 300 episodes, dev): L_W 4.37 bits with the map term against 6.12 without; within-module weight 63% against 46%; latent → action weight about 0.5% with the map term (the neuroscientist's predicted disconnection). Models are about 22 KB each, so they are committed with a study.
- `docs/derivation.md` §5.3–§5.4 and Appendix A: the counter estimates the weight-scaled derivative (finite removal is about 0.65 of it on the v4 LIF network); the ψ counter is exact only without reset; the LIF offspring and chance share; the QIF noise-instrument gain.
- `ruff.toml`: per-file ignores for the user's standalone `tests/test_emergent.py`. Its `test_lif_inhibitory_kick` fails until `siegert_gain` exists (it imported the absent `test_eligibility.py`).

### Results (iteration 014 pilot, development: 56 runs, 2 seeds, 300 episodes, `experiments/results/pilot014/`)

- `docs/iterations/014-emergent-v4.md` (new): pilot design, results, decisions.
- The map term lowers description length on all four engines and both tasks: held-out D 6.5 → 4.0–4.5 bits (CartPole) and 7.4 → 3.8–4.5 (LunarLander), against 5.8–6.5 and 6.6–7.6 without it. L_obs falls too, by only about 0.3 bits on LIF, whose pair counts are about 99% chance. Non-routing cross-module weight falls from 15–37% of the total to under 1.3%.
- Performance is mixed at this length. The map term is ahead in AUC for branching, QIF and LIF on CartPole and for QIF and LIF on LunarLander, and behind or mixed for flow and for branching on LunarLander.
- QIF with the map term runs at ρ(Λ̂^K) ≈ 1.0 (5% silent-module frames on LunarLander).
- Decisions:
  - LIF uses diffbound with β ≤ 1.6. With β ≤ 16, 36–61% of frames hit the deadline, past BOUND_CAL_LIF's 2.35 saturation. Race scores about 30 on CartPole.
  - The rule ablations `--ebar module` and `--step-stat kappa` do not replace the defaults.

### Found (iteration 014 §7: the action collapse)

- In the map-term-off spiking arms, the weights swing for 20–60 episodes at a time to one action. Controller→A1 weight reached 2.7–9× controller→A2, and the flow policy on the same W also collapses, so the readout is not the cause.
- The map term prevents this by sealing the modules (every action module's within-module offspring is 0.80). Part of its task benefit on spiking engines is therefore sealing, not a better abstraction.
- Training returns can hide a collapsed W: branching off seed 0 had a training return of 295 over its last 30 episodes and 10 with frozen weights.

### Added (iteration 014 §7–§8: a sealed control and runtime)

- `m2.Config.p_cross` and `m3 --p-cross`: the SBM's cross-module connection probability outside the routing pairs. The default 0.05 is unchanged. 0 builds the modules sealed: the λ = 0 control for "sealed modules help". The draws are the same; only the mask differs. Results record `p_cross` only when it is not the default.
- `core.build_network`: casts `p_cross` to float. An integer 0 had made the whole probability table integer, so every connection probability truncated to 0.
- `mapstdp/lif.py`:
  - numba loop `_kernel_nb`. `LIFParams.kernel`: `nbf` (default) draws variance-matched Irwin–Hall(4) noise from a xorshift64* generator, `nb` Gaussian noise, `np` the numpy reference.
  - External pulses are drawn as one Poisson count per head and step, split by v. One seed per frame comes from the frame's rng.
  - numba `pair_counts` (identical integer counts).
  - A LIF frame now takes 2.6 ms instead of 19 ms. LIF runs are not bit-identical to the pilot's: the stream differs, the distributions do not (module rates within 0.4%; the self-check passes).
- `mapstdp/spiking.py`: numba `branching_counters` over the synapses j → i only. Equal to the numpy path up to summation order. Per frame 8.4 → 1.3 ms (CartPole) and 12.2 → 2.0 ms (LunarLander).
- `experiments/report.py`: runs set `NUMBA_NUM_THREADS=1`. QIF's `parallel=True` kernel had started up to 72 threads in each of 40 processes.
- `tests/test_emergent_rule.py`: numba paths against numpy, the three LIF kernels' rates, and the sealed build.
- Worker benchmark (iteration 014 §8): throughput peaks at about 60 concurrent runs (3726 frames/s) and is 91% of that at 48. Under load, branching costs 12 ms/frame, LIF 10 and QIF 36, against 53–112 in the pilot.
- `m2.frozen_returns`, called at the end of every `rule="emergent"` run (`frozen_returns` in the result): 20 episodes with W frozen, on a deep copy of the agent, from the stream (seed, `FROZEN_SEED` = 7331). `report.m3_stats` and the pairs gain `frozen` (summary column "Frozen").
- `m2.BOUND_CAL[8]`, `BOUND_CAL_LIF[8]`: diffbound tables for 8 heads, from the pilot014 flow CartPole models (iteration 014 §9).
- Second pilot launched (`experiments/results/pilot014b/`, 116 runs): sealed control, LIF 20 ms window, QIF homeostasis ×2, 8 heads, LunarLander at 3000 episodes.

### Added (study `emergent-v4`, declared, not launched)

- `experiments/report.py`: `STUDIES["emergent-v4"]`: {LunarLander 1000, CartPole 1500 episodes} × {flow, branching, QIF, LIF} × λ {0.05, 0} × seeds 2000–2019 (320 runs, LunarLander queued first), with 8 on/off pairs.
- `reports/emergent-v4/README.md`: pre-registration draft, awaiting the user's approval. P1 is AUC and P2 is end-of-run held-out L_obs, each on vs off by two-sided Mann–Whitney with Holm over the 8 cells.

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-10-24

### Changed (no-Adam arm renamed: `sgd` → `noadam`, `--sgd` → `--eta-cal`)

- "SGD" read as backprop's stochastic gradient descent, which is non-local. The arm is the local three-factor rule without Adam, with a fixed η per population calibrated once at initialisation (E4′ rule).
- `experiments/tdstdp.py`: the flag is `--eta-cal`; `--sgd` stays as a hidden alias, because manifests and the 013 record contain it. The result-file token stays `sgd`, so existing intermediates keep their names.
- `experiments/report.py`: a study's `labels` map run keys to display names. `summary.*` show `LunarLander_noadam_<engine>`, with a "Display names" section. Run directories and manifests are unchanged.
- Both study READMEs carry an amendment. Both summaries were rebuilt from the same `runs/`, and no number changed.

### Results (`reports/tdstdp-baseline-flow/`, 60 runs, 0 failures)

- CartPole, no Adam: 17/20 solved at B = 20 and 18/20 at B = 1 (Fisher p = 1.0), so B = 20 stands.
- LunarLander: 3/10 solved with Adam; 0/10 without Adam. The no-Adam return peaks at 67–103 and then falls to −24.
- Record 013 §14.

### Added (flow-engine arms of the TD-STDP baseline, `reports/tdstdp-baseline-flow/`)

- New companion study `tdstdp-baseline-flow`, pre-registered in its README: TD-STDP on the flow engine (the branching step's noise-free limit) for CartPole without Adam, and LunarLander with Adam and with SGD, on the baseline's seeds. It is a separate study because the baseline ran at `675e49e` and a study uses one code state.
- A CartPole block-1 group is the control for the block sum, which was gated on branching only. On dev seeds 60–79, flow at B = 20 collapsed 4/20 CartPole runs to a one-action policy, against 0/20 at block 1.
- `experiments/report.py`: `ENGINE_FLAG` gains `flow`; a study's `reference` may name another study's group as `"<study>/<group>"` (`load_runs`). New test `test_summarize_cross_study_reference`.

### Added (explainer `docs/neuron-variants.md` and the LIF walk diagnostic)

- **`docs/neuron-variants.md` (new):** a plain-language guide to Map-STDP on the four engines (flow, branching, LIF, QIF). It covers what the rule assumes (pairings in proportion to link flow, flow matching the walk, subcriticality), how each engine meets those assumptions, the limitations of LIF and QIF, a side-by-side table, and a glossary. It explains the derivation and does not replace it.
- **LIF diagnostic** (frozen weights, `docs/iterations/013-tdstdp-backbone.md` §13; pre-registration, scripts and text outputs archived in `docs/iterations/013-lifdiag/`):
  - Model v4's recurrent network on the TD-STDP LIF fails the walk checks as it is. With input noise of 2–4 mV/√ms it passes subcriticality, the structural-rule direction (cos 0.94–0.95) and settling, but no operating point makes the pairing count proportional (slope 0.5–0.75), so task credit is 20–35% off.
  - On trained, nearly sealed weights the structural rule loses direction even on the branching engine (cos 0.44–0.89), from module-level co-fluctuation in κ on cross-module synapses. This needs a closed-loop check.
- `ruff.toml` excludes `docs/iterations` (archived diagnostic scripts).
- `CLAUDE.md` lists `docs/neuron-variants.md` under Docs.

### Changed (`reports/tdstdp-baseline/`: CartPole Adam groups dropped)

- **Decision (user):** `CartPole_adam_lif` and `CartPole_adam_branching` are dropped from the reported baseline. Both solved 0/20, because the authors' Adam step (the per-ms SGD η) is 40–160× LunarLander's for CartPole and collapses the policy at once. CK report no CartPole Adam configuration, and CartPole solves without Adam (20/20 on `lif`).
- `experiments/report.py`: studies take a `dropped` map (group → reason). `summarize` leaves those groups out of `summary.csv` and the tables, and lists them under "Dropped groups". Their run files stay in `runs/`.
- The study README has a dated amendment, and `summary.csv` and `summary.md` were regenerated. No other group, gate or statistic changed. 9 tests pass and ruff is clean.

## 2026-10-23

### Added (final-results layout and the reported TD-STDP baseline, `reports/tdstdp-baseline/`)

- **Layout (decided):**
  - `experiments/results/` is now, by documentation, the intermediates store: every dev, tuning and diagnostic run, and the default output of every CLI. No file was moved or renamed.
  - New top-level `reports/<study>/` holds only final reported outputs:
    - `README.md`, the pre-registration;
    - `manifest.json`: git HEAD, the sha256 of the uncommitted code in `mapstdp/` and `experiments/` (untracked files included, results excluded), with `code.patch`; versions; per-run commands;
    - `runs/<group>/s<seed>.json`;
    - `summary.csv` and `summary.md`.
  - Checkpoints and partials there are gitignored.
- **Code:**
  - New `experiments/report.py`: `run <study> [--workers N]` (launch, skip finished runs, resume checkpointed ones, refuse on a code-state change) and `summarize <study>` (from `runs/` only). Studies are declared in `STUDIES`.
  - `experiments/tdstdp.py`:
    - `--out PATH` (default unchanged);
    - `group_stats`, factored out of `summary()`. The `--summary` output is byte-identical before and after.
  - New `tests/test_report.py`: `summarize` on a tiny fake study gives the expected csv.
  - Golden is bit-identical, 9 tests pass and ruff is clean.
- **Docs:**
  - `CLAUDE.md`: the results layout, the `report` commands, and the trap "final numbers come only from `reports/`; never hand-copy results there".
  - Record 013: new §12, and a status line. The E1′, E3′, E4′ and S1–S3 numbers stay as development results.
  - Fixed three `\times` in record 013 §5.2 that had been corrupted to a tab followed by `imes`.
- **Study `tdstdp-baseline`** (pre-registered in its README):
  - Settings: the authors'-code path, B = 20, `--fast`, fresh seeds.
  - CartPole: {no Adam, Adam} × {lif, branching}, seeds 1000–1019.
  - LunarLander: {Adam, `--sgd`} × {lif, branching}, seeds 1000–1009.
  - 120 runs, 7.3 h, 0 resumes.
  - **Gates:**
    - CartPole no-Adam lif: 20/20, pass;
    - LunarLander Adam lif: 8/10, pass.
  - Reported:
    - CartPole branching: 5/20.
    - CartPole Adam: 0/20 on both engines. The authors' Adam step equals the SGD η, which collapses CartPole; this is not a code fault, because B = 1 behaves the same.
    - LunarLander Adam branching: 6/10.
    - LunarLander SGD: 0/10 on both engines.
  - The numbers are in `reports/tdstdp-baseline/summary.md`.

### Added (iteration 013: TD-STDP speed-ups and the flow limit, record §10–§11)

- **Code:**
  - `tdstdp.block_learn_ck_fast` with `--fast`: the update is summed over copies in one product. It is equal to the default up to summation order.
  - `tdstdp.flow_step` with `--engine flow`: the branching step's noise-free limit, rates instead of spikes, plus the input-autocorrelation term of z.
  - The defaults are unchanged and bit-identical. This was checked against the committed E1′ (both engines) and the legacy E1.
  - New self-checks:
    - (b″) the fast form equals `block_learn_ck` to $2 \times 10^{-16}$;
    - (e) the flow update against the Monte Carlo mean of the sampled rule: cosine 0.99997.
  - Golden is bit-identical, 8 tests pass and ruff is clean. `CLAUDE.md`'s Commands line lists `--block 20`, `--engine flow` and `--fast`.
- **S1, block sum without Adam** (CartPole, seeds 40–59, B = 20 against B = 1): **passes its gate on both engines.**
  - LIF: solved 20 against 19, Mann–Whitney p 0.27 on $T_s$ and 0.54 on return.
  - Branching: solved 6 against 7, p 0.99 and 0.39.
  - 6.7–7.1× faster.
- **S2, `--fast`** (LunarLander, seeds 100–104, 400 pooled episodes): the return sequences are identical to the default's. It is 1.8× faster per copy-frame (8.26 against 4.48 ms under load). Env stepping is about 1% of a tick, because the copies already share one batched network step.
- **S3, flow limit** (CartPole, seeds 60–79): 20/20 solved, $T_s$ 212.7 ± 51.7 and return 467.7 ± 107.8. Against E1′ branching (4/20), Mann–Whitney p $7.4 \times 10^{-8}$ on $T_s$. Escape noise is what fails branching's strict $T_s$.

### Changed (iteration 013: TD-STDP follows the authors' code; E1′, E3′ and E4′)

- **Code** (`mapstdp/tdstdp.py`, `experiments/tdstdp.py`):
  - `python -m experiments.tdstdp` now defaults to the authors' code (`reference/td_stdp`, BindsNET 0.2.7) wherever it disagrees with the paper. Results carry the token `ck`.
  - `PRESETS` are their `config_cp.ini` and `config_ll.ini` (η in mV). Weights convert to branching units by `MV` = 13, not 1300.
  - New `tdstdp.block_learn_ck`: their per-ms order with the $1/\tau_z$ and $1/\tau_q$ trace scale.
  - New `tdstdp.simulate_bindsnet` (`--engine lif`): their deterministic LIF with W in mV.
  - New `CKRunner`: their loop. The action is sampled from the response to the current state; p is fixed at the sampling probability; $V' := 0$ only on true termination; each episode has 6 presentations of $s_0$; the update is the mean over all copies; episodes are pooled over copies, with their min/max episode stopping rule.
  - Also new: their Fourier encoder and bounds; Adam ε mapped to the unit-trace gradient; `calibrate_ck` (the E4′ η rule); and checkpoints for long runs (`*.ckpt.npz`, now in `.gitignore`).
  - `--legacy` keeps the pre-audit path. It reproduces the committed E1/E2 results bit for bit under their original names.
  - `--summary` separates the two families and counts the authors'-code episodes pooled. It now reports Mann–Whitney p on $T_f$, $T_s$ and return, plus the number of runs with a last-100 return of at least 475.
- **Self-check:**
  - (b′) `block_learn_ck`'s closed form equals a literal per-ms transcription of `FBTDSTDP` to $6 \times 10^{-16}$.
  - (d′) The actor rule in the authors' order on a bandit has cosine 0.984 with the exact gradient, and 1.0000 per row.
  - Golden is bit-identical, 8 tests pass and ruff is clean.
- **Stopped:** the first E3 and E4 (LunarLander) were stopped at the wrong settings: the Adam step was 100× too small, episodes were counted per copy, entropy was 20× too weak, and decay was on. Their partials were moved out of `experiments/results` to the session scratchpad.
- **E1′** (pre-registered in record §9; CartPole, the authors' config, no Adam, B = 1, seeds 20–39):
  - **BindsNET LIF passes the gate:** 20/20 solved, $T_f$ 64.0 ± 8.4, $T_s$ 174.8 ± 18.5 (median 168), against CK's 57.0 and 169.5.
  - **Branching fails it:** 4/20 solved and last-100 306.9 ± 214.7. It has the same $T_f$ (62.6, MW p 0.46), but 7/20 collapse into a constant action after a period of near-perfect play (last-100 against LIF: MW p $6.8 \times 10^{-4}$).
- **E3′** (LunarLander, the authors' config, Adam, branching, B = 20, seeds 0–9): **passes its gate.**
  - 7/10 solved; $T_f$ 390.4 ± 99.8 against CK's 383.8.
  - $T_s$ 6774 ± 1896 pooled episodes against CK's 2575, so 2.6× slower.
- **E4′** (the same without Adam, η by the pre-registered Adam-equivalent rule): 0/10 solved and last-100 −0.9 ± 45.8. Against E3′, Mann–Whitney p ≤ 0.002 on $T_f$, $T_s$ and return. Adam is load-bearing on LunarLander.
- **LIF arm** (reported only): seeds 0–3 solved 3/4. Seed 4 was stopped at 5600 of 10000 pooled episodes; it can be resumed from its checkpoint.
- **Interruption:** all 25 LunarLander runs were killed at about 15:15 by a coordinating agent and resumed from their checkpoints. W, the Adam state, the returns and the rng were restored exactly; fewer than 50 episodes per run and the in-flight episodes were lost (record §5.2).
- **Docs:**
  - `docs/iterations/013-tdstdp-backbone.md` is rewritten to the current state. §3 states the corrected units and timing, lists the paper-against-code disagreements, and keeps the pre-audit choices. The pre-fix E1/E2 results are kept and labelled as such. §6's earlier reading that Adam is load-bearing is withdrawn for CartPole; E4′ shows that it holds for LunarLander.
  - `CLAUDE.md`: the tdstdp Code and Commands lines (`MV` = 13, `--legacy`).
  - `docs/derivation.md`, `docs/model.md` and `docs/SPEC.md` are unchanged.

### Found (iteration 013: audit against the authors' code, post hoc)

- `docs/iterations/013-tdstdp-backbone.md` §8 (new) compares iteration 013 with `reference/td_stdp` and BindsNET 0.2.7. No repo code changed; the diagnostic runs live in the session scratchpad.
- **The E1 failure is a reproduction error.** The cause is the η scale:
  - the authors' traces carry $1/\tau_z$ (critic) and $1/(\tau_z\tau_q)$ (actor), which the paper's Eqs. 72 and 77 omit;
  - BindsNET adds w mV per spike, so weights convert by ÷13, not ÷1300;
  - the code's critic η is 0.005.
  - Our critic step was therefore 10× too small and our actor step 8× too large.
- **Two loop details caused the collapses:** our action is sampled from the previous observation's response, and $V' := 0$ is also applied on truncation, while the authors bootstrap there.
- **Post hoc test, 5 seeds:** with all fixes and their deterministic LIF, the run reproduces CK: 5/5 solved, $T_f$ 65.0 ± 7.4, $T_s$ median 185. On the branching engine the same fixes reach a return of 490.4 ± 9.8 with 1/5 solved.
- **E3 is affected too.** The authors' Adam step is 100× ours, and their episode counts pool the 16 copies.

### Added (reference code of Chung & Kozma 2020)

- `reference/td_stdp/`: a shallow clone of https://github.com/stephen-chung-mh/td_stdp (commit `a5cb7f4`), the paper authors' BindsNET code, used to compare against the iteration 013 implementation. `reference/` is gitignored; reference only, not imported by the project.

### Added (iteration 013 step 1: TD-STDP backbone of Chung & Kozma 2020 on the branching engine)

- **Code:**
  - `mapstdp/tdstdp.py` (new): the feed-forward TD-STDP actor-critic of Chung & Kozma 2020, a baseline and not Map-STDP. It holds the branching step without presynaptic normalisation and a SuperNeuroMAT twin, a true-LIF control (CK Eq. 1, or SuperNeuroMAT's subtractive leak), a closed-form block sum of the z/q traces and updates (block 1 = the per-ms rule), a block Adam with per-ms-matched moments, and a self-check.
  - `experiments/tdstdp.py` (new): closed loop, CLI and `--summary`. Results go to `tdstdp_*.json`.
  - `tests/test_regression.py` now also runs the `mapstdp.tdstdp` self-check.
  - Golden is bit-identical, 8 tests pass and ruff is clean.
- **Self-check:**
  - The numpy step equals SuperNeuroMAT spike for spike, both in branching mode and with a finite leak, and the closed loop on the `snm` engine reproduces the numpy engine's returns and W.
  - The block closed form matches the per-ms loop to $4 \times 10^{-16}$.
  - The STE in branching mode is exact per postsynaptic row (cosine 1.0000) but carries the post rate as a row gain (global cosine 0.9975 critic, 0.9685 actor).
- **Decisions and deviations** (`docs/iterations/013-tdstdp-backbone.md` §3):
  - Weight units: η is divided by 1300, from CK Eq. 1 and Eq. 44. The BindsNET reading of 13 destroyed the network in a debug run.
  - Batch 16 runs as 16 asynchronous env copies with 3000 episodes per copy.
  - LunarLander runs at B = 20.
  - Block Adam was changed to per-ms-matched moments after the first form collapsed 3/10 CartPole seeds. It passed a gate set in advance (MW p 0.14 against per-ms Adam).
  - The E4 rule uses the per-ms update RMS.
  - E3 and E4 were stopped at episode 50 and relaunched under these two fixes.
- **Results** (`python -m experiments.tdstdp --summary`):
  - E1 (CK CartPole, no Adam, 20 seeds) **fails its gate**: 0/20 solved, last-100 15.4 ± 16.4. The true-LIF control with Table 1 verbatim fails the same way (0/10).
  - E2 (+ per-ms Adam) learns: 377.8 ± 83.2, $T_f$ 123.0 ± 30.2 against CK's 57.0, and 1/20 meets CK's strict $T_s$.
  - Post hoc: LIF + Adam gives 464.8 ± 31.4 (branching 369.7 on matched seeds, MW p 0.006). SGD with the Adam-equivalent η learns and then collapses in 6/10 seeds. An init at rate 0.1 does not rescue E1.
  - E3 and E4 (LunarLander) are running locally, started 11:13 EDT, at least 8.5 h.
- **Not changed:** `docs/derivation.md` and `docs/model.md` (TD-STDP is a baseline, not the model). `CLAUDE.md` gains the Code and Commands lines for the new module and script.

### Decided (iteration 013: TD-STDP on the branching engine first; QIF deferred)

- **Result (preliminary, scratch only; no repo code):** the derivation §5 rule on the `mpr` rate backend (`mprK`: pairing count $J^\Lambda$, eligibility $\kappa_{ij} \Phi^K_{m(i)}$ with no presynaptic baseline, pairing-weighted structural baseline, kernel clipped at 0 for $\kappa$ and $T^{eff}$, eligibility scale 0.75 fixed before training) against `mpr` and flow v4, v4 flags, last-100 mean ± sd:

  | Task (seeds) | flow v4 | mpr | mprK | mprK vs mpr (Welch / MWU p) |
  | --- | --- | --- | --- | --- |
  | CartPole (0–19) | 487.6 ± 51.7 | 231.1 ± 103.2 | 190.9 ± 100.8 (n = 18) | 0.23 / 0.63 |
  | Acrobot (0–19) | −107.6 ± 17.5 | −123.9 ± 23.5 | −114.5 ± 27.2 | 0.25 / 0.05 |
  | MountainCar (0–9) | −131.8 ± 2.8 | −166.6 ± 35.1 | −159.0 ± 34.8 | 0.63 / 0.57 |
  | LunarLander (0–9) | 102.5 ± 94.1 | −6.0 ± 57.6 | 4.5 ± 34.4 | 0.63 / 0.62 |

  On frozen weights the new score's cosine with the exact policy gradient is 0.83–0.97 after removing the column rescaling (the old 4b on `mpr`: 0.07–0.42), but in closed loop it closes none of the CartPole gap and is not significantly better than `mpr` on any task. Confounds: the structural and homeostasis terms ran about 1.7× stronger, and `mprK` drives $W$ toward regimes where the rate solver needs fallbacks (5–25× more). CartPole seeds 15 and 17 were stopped at the user's direction (8.6 h, episodes 950 and 600) and are excluded. Scripts and results: session scratchpad `qif/kscore/`.
- **Decision (user):** the QIF migration postpones the hard part (recurrent modules) rather than removing it. Build the TD-STDP backbone [Chung & Kozma 2020] on the SuperNeuroMAT branching engine first, then tune QIF. Derivation §5 stays a Proposal.

### Changed (derivation rewritten: current state only, map equation only, linear-response base)

- **Docs:** `docs/derivation.md` rewritten from 1037 lines (about 19k words) to about 480 (about 9k words), at the user's direction. History lives here and in `docs/iterations/`, not in the derivation.
- **Cut:**
  - Appendix B (corrections) and every dated or "previously/corrected" note.
  - Iteration-by-iteration narrative and result tables; iterations are now cited in one clause where a design choice needs evidence.
  - §5's alternative description lengths (entropy rate, entropy production, cost-weighted map equation, Markov stability, SBM, predictive dissipation, information flow, latent-chain entropy rate). The structural term is the map equation only.
  - The lateral-inhibition score (old Eq. 8, §7).
  - The rejected latent-output alternatives and the withdrawn or removed variants: Eq. 1g′ (β set point), Eq. 4c′ (critic TD(λ)) and Eq. 4i (usage term).
  - The dual routing term (4a) is reduced to a short m1-only ablation, and the routing exemption is the stated choice.
- **Restructured:**
  - §2.1 now derives the walk as the network's linear response (Hawkes kernel $K = \mathrm{diag}(G) B$, $r = Kr + \mu$, walk $T^{eff} = K/m$, escape $1 - m_j$), with branching mode ($K = (1-\alpha)T$, $G = 1$) as the implemented special case that gives Eq. 1.
  - The new §5 (**Proposal**, not implemented) states Lemma K (K6), Theorem K3 (K7: pairing-weighted baseline, preconditioner $m_j$) and the policy score (K8–K9: cascade value $\Phi$, no presynaptic baseline), with their open conditions.
  - §7 is now Inhibition: code the E→E walk, with inhibition entering only through $G$ and $\pi$; the score keeps the pools.
  - Appendix A is compressed to short proofs and one check table. Its EM-split cosines are flow 0.85/0.92/0.55/0.92 and QIF 0.25–0.44, replacing the 0.977 small-network figure.
  - Lemma K and (K8)–(K9) were re-checked independently on a 12-neuron softplus network: FD error $9 \times 10^{-10}$ and $7 \times 10^{-9}$ relative.
- **Equation numbers unchanged** for every surviving equation (1, 1a–1e, 1g, 1h, 2, 3, 3a, 4, 4a–4e, 4c″, 4h, 5–7). The retired numbers (1g′, 4c′, 4i, 8) are not reused. New equations are labelled K1–K9.
- **References:** dropped works no longer cited. Added Hawkes 1971, Hawkes & Oakes 1974, Lambiotte & Rosvall 2012, Montbrió et al. 2015, Pernice et al. 2011, Trousdale et al. 2012, Tsodyks et al. 1997 and Ozeki et al. 2009 (the last two verified this session).
- **Other docs:**
  - CLAUDE.md: dropped the Appendix B sentence and the "seven alternatives" framing.
  - SPEC (user decision: map equation only): dropped the §6 candidate table and its scoring criteria, the "estimators" and "composite modulators" open items, the per-candidate metric and ablation, and milestone M5 (modulator comparison; M6 keeps its ID). The energy-proxy criterion moved into the energy-model open item. Citations of derivation §3.4 now point to §3; the branching mode now cites §2.1.
  - model.md: rows citing Eqs. 1g′, 4c′ and 4i now cite their iterations.
  - `.claude/skills/doc-edit/SKILL.md`: the Appendix B rule is replaced by "no history in the derivation".
  - No code changed.
- **Review fixes (snn-expert and neuroscientist reviews, same day):**
  - Notation: offspring matrix $K \to \Lambda$ (so $J^\Lambda$, $\Lambda^{pop}$), offspring mean $m_j \to \nu_j$, Theorem K3 → Theorem K, latent-output barrier weight $\beta \to \beta_{LA}$ ($D_{LA}$), entropy-target gain and fraction $k_H$, $\phi_H$ (logs in nats). $K$ is now only the number of modules. Equation labels unchanged.
  - Corrections: the exact score is $\kappa_{ij} \Phi_i$ (neuron level); the module table $\Phi^K_{m(i)}$ is an approximation (cosine 0.83–0.97). Appendix A: baseline-as-renormalisation is $0.025 \lambda$ (was $2.5 \lambda$), (4h) MC correlation $\ge 0.9997$ (was 0.9998); §4.6 untrained (4b) cosine 0.98–0.998. "$m_j$ is a continuation probability" is replaced by "$\nu_j$ offspring on average". The Markov-stability remark, the Theorem 2 proof (base-2 logs) and the §6 hardware caveat are reworded; the non sequitur in §9 item 1 is dropped.
  - Added: one line on TD normalisation (model v4, `--td-norm`) after (4c); a note that the (4c'') label is kept.
  - Structure: new §2.5 "Action readout" ((1d), race, (1e), (1g), (1h), policy limits); the world model stays in §2.4. §4.5's second paragraph is split into three.
  - Biology (neuroscientist): the hypothesis that crossing costs more is labelled as a hypothesis; (K1) is the per-neuron QIF transfer "as in" the MPR mean field; the race burn-in cites Frank 2006 as a "hold your horses" STN signal; the per-neuron baselines $\bar{g}_j$, $\bar{h}_j$, $\bar{g}^{eff}_j$ are marked as abstractions (Royer & Paré is input-side only); the action broadcast is "plausible but untested"; $\Phi^K$ is not a plausible neuromodulator payload, so it is treated as a peripheral computation; §6 restores the backward (dendritically normalised) walk sentence and adds astrocyte domains [Bushong et al. 2002] and a presynaptic-locus factor; §7's E→E coding is a **Proposal**, with feed-forward PV inhibition [Gabernet et al. 2005] as the caveat. Added Gabernet 2005 and Bushong 2002 to the references.
  - Pointers: SPEC (controller recurrence) and model.md (Initialisation) now cite §2.5; SPEC's Locality convention cites §6 (Biological locality). Comment and docstring only: `spiking.set_weights` cites "§2.2, constant presynaptic output", `core.modulator` says exempt is the agent's default, and `m3` says "former Eq. 4i". `spiking.pooled` and `spiking.race` cite §2.5 for the race; SPEC §1 cites §2.4–2.5 for the readout.

## 2026-10-22

### Added (iteration 013 prep: QIF engine, Proposal; gate failed)

- **Code:** `mapstdp/qif.py` (new): quadratic integrate-and-fire engine in θ-form (τ 10 ms, dt 0.1 ms, Lorentzian η per head, exponential synapse τ_s 2 ms, delay 1 ms, fixed per-module PV pools, Bernoulli input pulses, warm start, 1 ms raster bins), numba kernel with a numpy twin, Montbrió–Pazó–Roxin mean field (`mpr`), Brian2 reference (`brian_frame`, lazy import) and self-check. `experiments/m2.py` `Config.engine="qif"`, `Config.qif`, `BOUND_CAL_QIF`; `experiments/m3.py` `--engine qif`, `--q-<field>` flags, file token `qif[...]`, fidelity columns `1-corr_bias`, `1-corr_row`, `1-corr_Jhat`, `S_mod`. Default path unchanged: golden bit-identical, 7 tests pass, ruff clean.
- **Env:** `environment-brian2.yml` (env `map-stdp-brian2`, numpy < 2.4, brian2 2.9) for the reference check only. Brian2 2.9.0 does not import under numpy 2.4 (`ndarray.ptp`), and pinning `map-stdp` to numpy 2.3 changes the spiking golden case, so the training env is unchanged.
- **Design** filtered by the neuroscientist; changes made during calibration: Δ 0.25 → 0.05, η truncated at 10, A_ext 3 → 5, fresh noise σ 1, PV pools on by default.
- **Results (preliminary, scratch only):** numba = numpy = Brian2 spike for spike. Fidelity on initial CartPole networks (16 heads): L1(π̂, π) 0.22 and 1 − corr(κ, J) 0.46, against 0.09 and 0.24 for branching mode. v4 smoke (CartPole, 5 seeds × 300 episodes): QIF 20.7–26.0 (random ≈ 22), branching 216–271. With flow π and J, the QIF race learns (63–99); with QIF π̂ and κ, nothing learns, because the structural rule dissolves the modules. Neuron-level MPR predicts the module shares better than the linear flow (L1 0.10 against 0.20).
- **Phase A, tune before rebuild (preliminary, scratch only):** qif-path scale fields `QIFParams.lam_scale`, `eta_scale`, `eps_h_scale` (default 1; flags `--q-lam-scale`, `--q-eta-scale`, `--q-eps-h-scale`, `--q-tau`). Matching update RMS to branching does not help (CartPole 20–25). A fidelity sweep (600 configs) finds no point with L1 ≤ 0.10 and corr(E[κ], J) ≥ 0.9; the best, at 0.19 and 0.80, needs strong fresh noise (σ 3.4, τ 3.5 ms), far from the Montbrió–Pazó–Roxin-exact regime. There, CartPole reaches 44–56 (branching 216–271) and Acrobot −291 to −340 (branching −137 to −162). Decision: QIF needs the rebuild (Phase B: π from rates, κ against G·T·r̂, the 4b score on r̂).
- **Phase B steps 4–5 (preliminary, scratch only):** `qif.rates` (neuron-level MPR fixed point: damped Newton, continuation fallback, returns per-neuron gain G), `qif.MPR_POINT` (σ 0, η̄ −2, Δ 0.5, gain 20, w_ie 15, rate 1, 32 I neurons per pool, τ_s 20), and `m2.Config.backend = "mpr"` (π = r*/Σr*, κ = T·diag(r*), Eq. 1d/1e on module shares). Solver: 6–12 ms per frame, 0 failures, fallbacks below 0.1% of frames. v4 flags, 5 seeds, against flow v4: CartPole 261 vs 500, MountainCar −159.6 vs −132.2, Acrobot −118.2 vs −96.3, LunarLander −24.0 vs 82.4. Gate (within 10%) failed on every task. The cosine of the 4b score with the true policy gradient drops on trained networks from 0.63–0.96 (flow) to 0.07–0.25 (MPR), so the module score, not κ, is the binding problem.
- **Phase B step 4 gates (passed):** L1 of module shares, rates against QIF spikes, is 0.027 / 0.064 / 0.028 / 0.041 on CartPole / MountainCar / Acrobot / LunarLander (per-neuron corr 0.92–0.95). From 10 starts × 20 random W per task, plus trained W, every solve converges to the same fixed point. Self-checks: residual below 1e-9, and G matches finite differences. A gain-aware credit (a K × K module resolvent with module-mean gains, times each neuron's own gain G_i) reaches cosine 0.94–0.97 with the exact gradient, against 0.07–0.25 for 4b (scratch `psi.py`).

### Removed / Changed (iteration 013 prep: prune unadopted variants, `m2.Config`)

- **Removed (not adopted; code, CLI flags and golden cases):** models v1 and v2 and the dual routing term (Eq. 4a) in `experiments/m2.py` (`--model`, `--routing dual`, `Agent.mu`/`q_star`, `EPS`, `ETA_SPK`, `ETA_MU`, `Q_STAR_FRAC`); the β ramp (`--beta-end`, `--beta-ramp`), mean reversion (Eq. 1g′: `--beta-eps`, `--beta-set`, `--beta-decay`), trace signal (`--beta-signal`), step cap (`--beta-step-cap`, `--beta-step-shape`) and score-unit ablation (`--score-unit`); the TD(λ) critic (Eq. 4c′, `--critic-lambda`); the `rich` and `net` critics (`--critic`); the usage term (Eq. 4i: `--c-use`, `--use-tau`, `core.usage_score` and its self-check); the MSPRT readout in `experiments/ceiling.py`. `mapstdp/core.py` keeps `modulator(routing='dual')` for `experiments/m1.py`. Golden cases `flow-dual-0.05-v1`, `flow-none-0.05-v2`, `spiking-exempt-0.0-v1` deleted from `tests/golden.json`; nothing regenerated.
- **Changed:** `m2.Agent(cfg)` takes one frozen `m2.Config` dataclass (shared by the m2 and m3 CLIs, `m3.fidelity`, `ceiling`, `m2_viz`); `m2.run(cfg, episodes, name=None)`. `Agent.episode` split into `task_step` and `track`; the KL readout check moved to `Agent.race_fidelity`; `regulariser` became `entropy_term` (Eq. 4h only).
- **Result files:** names unchanged for every kept configuration. New files no longer carry the pruned keys `mu`, `q_star`, `beta_end`, `beta_ramp`, `score_unit`, `params.eta_mu`, or `use_rms_ratio`/`ubar` in evals. `m2 --summary`, `m3 --summary`, `m3 --ablations` and `ceiling --summary` print byte-identical output over the committed results (v1/v2, dual, MSPRT and pruned-variant files included).
- **Behaviour:** remaining golden cases (`flow-exempt-0.05-v3`, `spiking-exempt-0.05-v3`, `spiking-readout`) bit-identical; v4 LunarLander, iteration-012 MountainCar, spiking 2-head and diffbound runs reproduce the original code's returns exactly. ruff clean; pyright 42 → 33 errors.
- **Docs:** CLAUDE.md `m2.py` bullet updated for `m2.Config` and the pruned model versions; dual routing marked as `m1.py`-only. Decision (user): no no-latent control or new golden fingerprints before the prune; the no-latent ablation is deferred to the latent fixes and sweeps of iteration 013, and code still needed as reference (flow training, ceilings, ablations) is kept until superseded.

## 2026-10-21

### Added (iteration 012: encoders, separate critic, entropy terms)

- **Code** (all default off; golden 9/9 bit-identical; ruff clean): `mapstdp/core.py` Fourier basis (`fourier_coeffs`, `fourier_features`, `fourier_stimulus`), `action_inflow`, `action_entropy`, `entropy_score` (Eq. 4h), `usage_score` (Eq. 4i), with self-checks. `experiments/tasks.py` encoder specs `g<n>[s<σ>]`, `f<order>c<conj>[h][b]`, `FOURIER_BOUNDS`, `with_encoder`, `critic_layer`. `experiments/m2.py` `Agent(encoder, critic_layer, c_ent, c_use, use_tau, ent_target)` and diagnostics. `experiments/m3.py` `--encoder`, `--critic-layer`, `--c-ent`, `--ent-target`, `--c-use`, `--use-tau`; tokens `enc`, `crit`, `ce`, `cu`, `ep<E>`. `experiments/ceiling.py` `--encoder`, `--no-train`, an underflow guard for $p^\beta$.
- **Results:** about 680 `m3_*` runs (pilot seeds 0–4, pre-registered seeds 0–19, fresh seeds 20–39, LunarLander 3000-episode runs) and about 230 `ceiling_*` runs (10 CEM runs per cell).
- **Docs:** `docs/iterations/012-encoders-critic-entropy.md`; derivation §4.6 (Eq. 4c'' separate critic, Eqs. 4h and 4i, Proposals), Appendix A (numerical checks for 4h and 4i: finite differences to $2 \times 10^{-10}$, Monte Carlo correlation ≥ 0.9998, full-resolvent cosine 0.92–0.999), Appendix B item 17; references Chung & Kozma 2020, Foster et al. 2000, Haarnoja et al. 2018, Lansink et al. 2009, O'Doherty et al. 2004, Tervo et al. 2014; model.md rows, parameter and history; SPEC M3 row; CLAUDE.md iteration range and the best MountainCar command.
- **Findings (40 seeds):** MountainCar −100.7, 38/40 at −110 (12 × 12 encoder, 24 × 24 separate critic, entropy target); Acrobot −92.8, 35/40 (separate Fourier critic); CartPole 500, 40/40, median 296 episodes to solve (Fourier encoder and critic); LunarLander unchanged. Confirmed on fresh seeds 20–39.
- **Decision:** no model version bump; the per-task configurations stay Proposals. The entropy term's setting deviates from the neuroscientist's calibration (destructive at 0.1–0.3 of the task RMS); sign-off pending. Next: iteration 013 on a new branch, a TD-STDP backbone with communities.

### Added (iteration 011: flow diagnostics)

- **Docs:** `docs/iterations/011-flow-diagnostics.md` (scratch code only; no repo code or results changed). model.md history row; SPEC M3 row; CLAUDE.md iteration range.
- **Findings (preliminary):** a sharper readout does not help on frozen v4 weights (argmax: MountainCar −133.6, Acrobot −107.6). MountainCar is limited by the encoder and the critic: 12 × 12 controller grid + separate critic on a 24 × 24 RF layer gives −101.9 ± 6.0, 18/20 at −110 (v4 −131.0, 0/40). Acrobot's bad seeds entrench the zero-torque action early; masking it on frozen weights gives −97.4 (26/40 at −100).
- **Corrections:** the 6 × 6 MountainCar encoder's CEM ceiling is about −114 at 10 CEM runs, not −110 (iterations 005, 009, 010 used 3 runs); iteration 009's "the critic is not the MountainCar bottleneck" is withdrawn (009 record, derivation §4.6).
- **Decision (user):** iteration 012 tests a 12 × 12 encoder + separate 24 × 24 critic on MountainCar, a Fourier-basis encoder [Chung & Kozma 2020] and a separate critic on the other tasks, and an entropy bonus on all tasks.

### Added (iteration 010: decision-bound readout)

- **`mapstdp/spiking.py`:** `diffbound()`, a race on count differences to a bound (Eq. 1h), optionally capped at one spike per module, head and step; self-checks (deterministic raster, cap, $K = 2$ gambler's-ruin Monte Carlo). **`experiments/ceiling.py`:** readouts `beta<B>`, `db<T>`, `msprt<T>c<C>`, `--nmax`; existing readouts draw unchanged. **`experiments/m2.py`:** `Agent(readout='diffbound')`, spiking only, $\theta$ from $\beta$ through `BOUND_CAL`. **`experiments/m3.py`:** `--readout`, token `dbc`. All default off; golden 9/9 bit-identical, ruff clean.
- **Results:** level-1 ceilings on MountainCar and Acrobot for Eq. 1e at $\beta$ 4/8/16, (1h) and MSPRT at $\theta$ 2–16 (3 CEM seeds); spiking pilot at 16 heads on CartPole and MountainCar (race, (1h) with adaptive $\beta$, (1h) at fixed $\beta$ 4 and 8; seeds 0–4).
- **Docs:** `docs/iterations/010-decision-bound.md`; derivation §2.4 (Eq. 1h, Proposal, spiking only; $\beta$ read as a decision threshold rather than noradrenergic gain; burst overdispersion $\beta_{eff} \approx \theta / F$), §9 item 14, Appendix A (numerical checks for 1h: $K = 2$ identity to $2 \times 10^{-15}$, expected count to $10^{-12}$, $K = 3$ KL ≤ 0.0065, burst formula), Appendix B item 16; references Bogacz et al. 2006, Bogacz & Gurney 2007, Bogacz et al. 2010, Cavanagh et al. 2011, Feller 1968, Lo & Wang 2006, Usher & McClelland 2001, Wald 1947; model.md row and history; SPEC M3 row; CLAUDE.md iteration range.
- **Decision:** flow unchanged (the gate failed: (1h) and MSPRT ceilings equal Eq. 1e's at matched $\beta$). Spiking (1h) kept as a Proposal, not adopted: MountainCar −132.2 against −177.0 for the race (flow parity), CartPole 153.9 against 298.3. MSPRT and the attractor fallback dropped.
- **Note:** the committed iteration-005 $\beta = 4$ ceiling at seed 0 does not reproduce on this machine (MountainCar −115.2 against −121.6); training return differs too, so the drift is machine level. The committed files are kept.

### Added (iteration 009: credit horizon, negative)

- **`experiments/m2.py`:** `Agent(critic_lambda, gamma)`: the critic runs through a presynaptic trace $z$ (Eq. 4c'), identical to Eq. 4c at $\lambda = 0$ (golden 9/9, full v4 reruns identical). **`experiments/m3.py`:** `--critic-lambda`, `--gamma`; tokens `cl<λ>`, `g<γ>`; `nondefault()` catches `tau_e`.
- **Results:** 280 new `experiments/results/m3_*` runs (MountainCar and Acrobot arms, seeds 0–19; $\tau_e = 10$ also on fresh seeds 20–39 and on CartPole and LunarLander).
- **Docs:** `docs/iterations/009-credit-horizon.md`; derivation §4.6 (Eq. 4c', tested, not adopted), Appendix A (numerical checks for 4c': $\lambda = 0$ bit-identical; forward-view equivalences to $10^{-15}$; update-time normalisation off by up to 12% on Acrobot), Appendix B item 15; references Pan et al. 2005, Sutton 1988; model.md rows; SPEC M3 row.

### Changed (decisions, iteration 009)

- **v4 kept.** The `neuroscientist` filtered the proposals first (critic TD($\lambda$), $\tau_e = 10$, $\gamma = 0.995$ accepted; $\tau_e = 30$ rejected as beyond the dopamine-trace evidence; encoder expansion lowest priority). No arm survived the pre-registered Bonferroni tests, $\tau_e = 10$ did not replicate on fresh seeds and cost CartPole (295.6 against 487.6), and the critic trace with update-time normalisation diverged on Acrobot (20 of 20 seeds).
- **The critic is not the bottleneck** (V correlates 0.99 with the true value on MountainCar). v4 sits 10–25 below the ceiling of its own Eq. 1e readout, so the flow gap is the learned weights. Next, iteration 010: a decision-bound readout, mainly for a sharpened spiking readout.

### Added (iteration 008: β's gain on the task step)

- **`experiments/m2.py`:** `Agent(beta_step_cap, beta_step_shape)`: the Eq. 4b increment uses $\beta_\eta(\beta)$ in place of $\beta$, with shapes `min`, `sat` ($c\beta/(c+\beta)$), `invu` ($\beta e^{1-\beta/c}$) and `hyb` (identity to $c$, logarithmic above); the direction is unchanged (cosine 1). Exclusive with `score_unit`. Defaults off; `tests/golden.json` bit-identical (9/9).
- **`experiments/m3.py`:** `--beta-step-cap`, `--beta-step-shape`; tokens `sc<c>[sat|iu|hy]`.
- **Results:** 960 new `experiments/results/m3_*` runs at $\kappa_\beta = 0.01$: v4 and v5 at seeds 10–39, six step shapes at seeds 0–19, and the pre-registered confirmation (v5, min 4, sat 16, hyb 4) at seeds 20–39.
- **Docs:** `docs/iterations/008-beta-step-gain.md`; derivation §2.4 note; model.md and the 007 record corrected; SPEC M3 row.

### Changed (decisions, iteration 008)

- **No step shape adopted.** None of the four pre-registered tests on fresh seeds 20–39 survives Bonferroni; min 4's CartPole effect from seeds 0–19 reversed (445.6 against 499.4). `invu` rejected (MountainCar seeds that never learn, worse Acrobot).
- **v5 withdrawn; the documented model is v4 again** (the user's decision). At $\kappa_\beta = 0.01$ and n = 40, v5 is not better than v4 on any task (LunarLander nominally worse, 52.4 against 81.3, p = .13); the 1g′ recovery evidence rested on the $\kappa_\beta$ 0.0099 runs. Eq. 1g′ stays in the derivation as a tested, unadopted Proposal; its flags remain.
- **Seed noise:** the $\kappa_\beta$ 0.0099 control is replaced by more seeds at $\kappa_\beta = 0.01$ (the user's decision). v5's CartPole mean is 500.0, 399.4 and 499.4 over seeds 0–9, 10–19 and 20–39; report at least 20 seeds per arm.
- **Neuroscientist review:** a plasticity gain that saturates while response gain does not is a plausible, untested abstraction (dissociable, non-monotone neuromodulator dose-responses: Arnsten 2009, Vijayraghavan et al. 2007, Salgado et al. 2012, Seol et al. 2007); smooth saturation preferred to a hard cap; "not a gradient" corrected to "a $\beta$-dependent step size on the same gradient".

### Added (iteration 007: robust adaptive β)

- **Model v5** (`docs/model.md`): v4 plus `--beta-eps 0.005 --beta-decay 1000`, the experience-dependent set point of Eq. 1g′ (Proposal). $\beta$ now recovers after a collapse (CartPole, 20 runs: 5 of 5 drops recover, against 1 of 3 in the v4 family). Return is unchanged: pooled over $\kappa_\beta$ 0.01 and 0.0099 (n = 20), CartPole 489.8, MountainCar −130.4, Acrobot −106.7, LunarLander 92.3, against 466.0, −132.4, −106.9 and 100.6 (all p ≥ .05). Code defaults unchanged; `tests/golden.json` bit-identical (9/9).
- **`experiments/m2.py`:** `Agent(beta_eps, beta_set, beta_decay, beta_signal)`, `beta_move()`, `beta_frame()` (per-frame trace signal), constants `BETA_TF`, `BETA_TS`, `BETA_TH`; the result JSON records the new keys only when non-default.
- **`experiments/m3.py`:** `--beta-eps`, `--beta-set`, `--beta-decay`, `--beta-signal`; tokens `re<ε>[s<β₀>][d<τ_D>]` and `tr`; `nondefault()` updated.
- **Results:** 440 new `experiments/results/m3_*` runs: candidates C1–C4 and v5, the $\kappa_\beta$ 0.0099 noise control, and the iteration 006 rows extended to 10 seeds.
- **Docs:** `docs/iterations/007-robust-beta.md`; derivation §2.4 (Proposal Eq. 1g′, open issues updated), §9 item 14, Appendix A (numerical checks for 1g′ from `/derivation-check`: reduction to 1g bit for bit on 33 119 updates, fixed point and recovery rate exact, trace signal zero-mean when stationary), Appendix B item 14; three references (Abraham & Bear 1996, Eisenberger 1992, Niv et al. 2007); SPEC M3 row; CLAUDE.md v5 command.

### Changed (decisions, iteration 007)

- **v4's numbers are partly a lucky draw.** Changing $\kappa_\beta$ by 1% gives CartPole 432 and Acrobot −114. Variants are now compared against the pooled v4 family (n = 20).
- **Correction to iteration 006:** at n = 10, dividing $\beta$ out of the task step gives LunarLander 64.0 (p = .38 against v4), not 7.0; "most of LunarLander's gain is the step size" no longer holds. The 006 record is annotated.
- **A strict running maximum as the set point is rejected** (`neuroscientist`): it never decreases, so it would block disengagement after a change in task. A maximum leaking over $\tau_D = 1000$ episodes replaces it, with the same recovery.
- **Rejected:** a leaky mean of $\delta / \mathrm{RMS}(\delta)$ as the episode-free signal ($\beta$ stays at 1 on MountainCar and Acrobot); two-sided reversion toward a slow average; a $\beta$ floor of 2.
- **Not adopted:** a fixed two-sided set point ($\beta_0 = 8$; narrows LunarLander's spread, no gain in mean, hand-set prior); the per-frame outcome-reset trace (the signal works, but no single per-frame rate fits all tasks).
- **Collapses live in the weights:** frozen-checkpoint tests show re-raising $\beta$ helps only while the weights still hold a good policy. The next lever is the $\beta$ step gain at high $\beta$, not the $\beta$ rule.

### Added (iteration 006: M3 reward)

- **Model v4** (`docs/model.md`): the M3 configuration `--p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5` (with `--td-norm 0.04`, `--la-beta 0.3`). Flow return over 10 seeds: CartPole 272.5 → 500.0, MountainCar −184.2 → −131.8, Acrobot −128.8 → −100.1, LunarLander −37.8 → 102.5 (± 94, one seed collapsed). Code defaults unchanged, so v1–v3 and `tests/golden.json` reproduce bit for bit (9/9).
- **`experiments/m2.py`:** `Agent(p_cc, beta_end, beta_ramp, beta_kappa, beta_max, score_unit)` and `ablate={"critic": "net"}`; the result JSON records them.
- **`experiments/m3.py`:** `--lam`, `--p-cc`, `--beta`, `--beta-end`, `--beta-ramp`, `--eta`, `--beta-kappa`, `--beta-max`, `--score-unit`, `--critic net`; file-name tokens for each; `summary()` keeps the variants out of the baseline groups.
- **`experiments/tasks.py`:** `CartPoleFull` (all four observations; a negative result).
- **Results:** about 515 `experiments/results/m3_*` runs, including the map-term ablation of v4 (`lam0`).
- **Docs:** `docs/iterations/006-m3-reward.md`; derivation §2.4 (credit limit, Proposal Eq. 1g), §4.6, §9 items 13–14, Appendix A (numerical checks for the credit limit, from `/derivation-check`: exact score to $4.6 \times 10^{-11}$, cosines and block decomposition) and Appendix B item 13; three references; SPEC controller entry and M3 row.

### Changed (decisions)

- **Controller built without recurrence ($p_{cc} = 0$).** Map-STDP's module-level task term cannot credit controller→controller synapses (cosine of Eq. 4b with the exact score 0.55–0.85 on trained weights, 0.83–0.99 without them). The controller is now described as a feed-forward encoder, not cortex or a workspace (`neuroscientist`).
- **Correction:** the $0.997$ cosine of Eq. 4b with the true score holds only on untrained weights.
- **Proposal: adaptive $\beta$ (Eq. 1g)**, a causal utility-trend gain, one setting for all tasks. **The time-based $\beta$ ramp is rejected as a model component** (the user's decision): it needs the run length in advance. It stays only as a labelled oracle reference flag.
- **Rejected:** a $(\mathcal{C}, \mathcal{C})$ cost penalty (not a description-length gradient), the Eq. 4f resolvent credit on reactive tasks, an entropy-gradient term, the rich and network critics, `CartPoleFull`.
- **Deferred:** a matched sharpened spiking readout (a race to $k$ spikes reaches only $\beta_{eff} \approx 1.5$).
- **Encoder:** kept fixed through M3 (`neuroscientist` review of visual-cortex models).
- **The map term still earns its place in v4.** With `--lam 0` (10 seeds): CartPole 500.0 → 396.4 (p = .05), MountainCar −131.8 → −187.1 (p = 1e-4; 8 of 10 seeds never reach the goal, so $\beta$ never rises), LunarLander 102.5 → 11.0 (p = .02); Acrobot unchanged (−100.1 against −98.3).

## 2026-10-20

### Changed (CLAUDE.md)

- **CLAUDE.md** gains three things a fresh session would otherwise learn by hitting them:
  - **A "Traps" list:** RNG-call order and the golden tests, no `zip(strict=True)`, the `m2_*` result-file namespace, and silent spiking modules.
  - **The unmerged `m4-tmaze` branch:** a T-maze M4 prototype whose iteration number 004 collides with `main`'s.
  - **The symptoms of launching outside the `map-stdp` env:** the KaTeX hook errors and CodeGraph does not connect.

### Added (iteration 005: M3 tasks, spiking heads, learner limits)

- **M3 tasks:** `experiments/tasks.py` registry (CartPole, MountainCar, Acrobot, LunarLander): module sizes, encoder, reward, episodes. `core.rbf_stimulus` (stable conjunctive RFs at arbitrary centers). CartPole encoder and reward are bit-identical to M2.
- **`experiments/m3.py`:** per-task runs on either backend, `--summary`, `--fidelity`, `--ablations`; flags `--heads`, `--engine`, `--td-norm`, `--eta-tau`, `--wmin`, `--tau-e`, `--critic`.
- **Proposal: parallel spiking heads** (`spiking.tile_heads`, `head_groups`, `pooled`; `m2.Agent(heads=H)`): H replicas on shared W, spikes pooled for $\hat{\pi}$, $\kappa$ and one race. $\hat{\pi}$ error against flow falls as $1/\sqrt{H}$ (0.35 → 0.089 at H = 16, CartPole). The return sweep has not been run.
- **Batched spiking engine** `spiking.run_heads`: raster-identical to SuperNeuroMAT (new self-check), 7–13× faster at H = 16. M3 default; M2 stays on SuperNeuroMAT. `spiking.set_inputs` writes SuperNeuroMAT's input array directly; `build` pins `sparse = False` (the jit backend is dense-only).
- **Proposal: TD normalisation** `--td-norm s`. At $s = 0.04$ it fixes the Acrobot collapse (−374 ± 156 → −128 ± 7); neutral on CartPole, MountainCar, LunarLander.
- **`Agent.diagnostics()`** logged with every eval (TD error, critic range, weights at bounds, policy entropy and sharpness). `Agent(ablate=...)` for learning-rule ablations.
- **`experiments/ceiling.py`:** policy-class ceiling by CEM. Free $R$ with Eq. 1d reaches CartPole 465, MountainCar −150, Acrobot −90 against learned 205, −183, −133 (seed 0); sharp readouts add more.
- **Results:** `experiments/results/m3_*.json` (flow, all tasks, ± `--td-norm`, CartPole and Acrobot ablations), `ceiling_*.json`, `m3_fidelity.json`.
- **Environment:** `box2d-py`, `pygame` in `environment.yml` and the `map-stdp` env.
- **Docs:** `docs/iterations/005-m3-tasks-heads.md`; `docs/model.md` rows (M3 stimulus, heads, engine, M3 reward, TD normalisation) and history; CLAUDE.md code and commands; SPEC M3 row.

### Fixed

- **Frame count in `m2.run`** was `sum(returns)`, wrong for negative-reward tasks; now counted per frame. CartPole results unchanged.

### Changed (decisions)

- **Diagnosis:** Map-STDP converges to a poor $R$ (the actions × controller matrix of the flow policy); the network can express much sharper votes, and Eq. 1d's stochasticity is a second cap. Latent memory is not the bottleneck on these fully observed tasks.
- **Rejected as fixes** (CartPole, Acrobot, 5 seeds): constant or slower task-rate annealing, a higher weight floor (it hurts: pruning to the floor is how the policy sharpens), longer eligibility, a richer critic.
- **Next:** an ideal actor-critic on $R$ with the same signals, then the spiking heads sweep.
- **Known issue:** `tests/golden.json` fails on the Mac workstation even at a clean HEAD (last-bit BLAS differences against the cluster); not regenerated. This iteration was checked against fingerprints of HEAD generated on the same machine: bit-identical.

## 2026-10-06

### Added (iteration 003: latent sinks)

- **Finding:** in the trained v3 agent, latent modules are flow sinks. They keep 0.98 of their flow, pass 0.002 to the actions and hold about half the activity.
  - **The task term causes the latent→action cut,** which also happens at $\lambda = 0$.
  - **The map term seals latents against the controller and each other,** which is where its CartPole benefit comes from (184 with the map term on actions only, against 287).
- **Proposal: latent-output barrier, Eq. 4e** (`mapstdp/core.py` `latent_output_barrier`; `experiments/m2.py --la-beta`, default 0).
  - **Derivation check:** finite differences to 9e-10, Monte Carlo 0.9999. A new self-check covers it.
  - **Cost on CartPole:** flow 274 ± 24 at $\beta$ 0.15 and 279 ± 22 at 0.3; spiking 171 ± 9 and 155 ± 9, against 287 and 184.
  - **Effect:** the latent→action share goes from 0.002 to 0.012–0.029. It regrows from sealed weights, and cross-module traffic barely rises.
  - **Confirmation:** the 20 runs through the repo CLI reproduce the prototype exactly. `pytest`: 9/9, v1–v3 bit-identical.
- **`docs/derivation.md`:**
  - **§4.5:** the diagnosis, (4e), the rejected alternatives, and why the (4a) dual diverged (its target was infeasible once modules seal).
  - **Appendix A:** the checks.
  - **Appendix B:** item 12.
- **`docs/model.md`:** a "Latent output" row and the $\beta_{LA}$ parameter.
- **`docs/SPEC.md`:** the ablations list.
- **`docs/iterations/003-latent-sinks.md`:** the record.

### Fixed (replay page)

- **Walk hops were overdrawn.** The page drew a hop for every synapse whose pre spiked at $t - 1$ and whose post spiked at $t$. That gave 2.0 hops per spike, and 57% of spikes got several parents. Stimulus input spikes got spurious parents too. The cross-module share came out at 0.31, against a true value of about 0.22.
  - **Fix:** `sp.run_frame(..., return_ext=True)` now returns which spikes were stimulus input. `m2_viz.walk_hops` gives every other spike exactly one parent, sampled in proportion to $T_{ij}$ among the spikes one step earlier. That is the exact cause posterior in branching mode.
  - **Check:** input share 0.212 against $\alpha = 0.2$, 0.788 hops per spike, every spike accounted for, and a cross-module share of 0.220 against 0.215 at the flow level.
- **The page now:**
  - marks input spikes with a ring;
  - draws every cross-module synapse with opacity proportional to $\sqrt{T}$, where before it showed only those with $T > 0.04$, which hid the weak latent→action links;
  - counts each frame's hops: total, between modules, latent→action, and input.
- **The replay uses the latent-output barrier** (`m2_viz --la-beta 0.3`; seed 4 is the best of 4, with last-100 training return 180). Its 366-frame episode has 3.7% of latent hops ending in an action module. Republished to the same artifact link.

### Changed

- **Decision reversal: the exit-floor Proposal is rejected.** Its dual enters a limit cycle, and its restored exit goes to the controller, not the actions (return 126–196).
- **Also rejected:**
  - the latent→action dual (limit cycle);
  - exemptions into the action modules (no effect, cannot regrow);
  - occupancy homeostasis (return 108);
  - a fixed set of output neurons per latent (exits to the controller);
  - an $M^{\ast}$ clip on latents.
- **Carried to M4:** credit through the latents and nonlinear carry-over. A $\dot{\theta}$-hidden CartPole test was flat for every variant (44–52).

## 2026-10-05

### Changed (expert review of the cleanup)

- **Review:** `/expert-review` of the cleanup diff by `snn-expert`, `superneuro-expert` and `neuroscientist`. All three found the cleanup numerically safe; `superneuro-expert` ran 20 spiking episodes on the old and new code and got identical frames, identical weights and the same 3.3 ms per frame. Applied, all names, docstrings and comments only; `pytest` 9/9 bit-identical and `ruff` clean afterwards:
  - **Renames in `experiments/m2.py`:** `local_plasticity` → `pre_td_updates` (all three reviewers: it is neither more local than the task term nor purely plasticity, and it mutates W and μ); `td_step` → `critic_step` (the task weight update stays in `episode`).
  - **`experiments/m2.py` docstrings:**
    - `pre_td_updates`: its update order, which the golden file locks;
    - `episode`: the task update lands one walk late;
    - `frame`: what κ and the action are in each backend;
    - `Agent`: the dual and the annealing;
    - `evaluate`: in spiking it consumes the training RNG.
  - **`mapstdp/spiking.py`:**
    - `pi_hat`: cited as the batch form of Eq. 2, replacing the bare §2.2 citation;
    - `run_frame`: what `reset()` clears;
    - `build` and `set_weights`: synapses are fixed at build time;
    - `pairings`: now says `lag`.
  - **`mapstdp/core.py`:**
    - `homeostasis_update`: marked as presynaptic conservation, an abstraction;
    - `flow`: a comment links `pi` to the iterates $x_k$.
  - **Not applied:** a separate evaluation RNG. `evaluate()` in spiking draws from the training RNG, so the logging interval changes trajectories. Fixing that would change `tests/golden.json`, so it is documented and deferred.

### Changed (docs aligned with the code; found by the codebase-cleaner)

- **`docs/derivation.md`:**
  - **Eq. 4c** now states the critic the code uses since iteration 001: $V = u \cdot \pi_{\mathcal{C}} / p_{\mathcal{C}}$ over controller neurons, with NLMS, reward −1 on termination and bootstrapping on truncation. The module-level critic is recorded as the weaker alternative (160 against 195, preliminary, iteration 001).
  - **§6 table** and **§9 item 13** are updated to match.
  - **§8 step 2:** the M2 frame is cold-started, $k = 100$ (was "about $4 \tau / \alpha$, warm-started").
  - **§8 step 5:** reordered to match the code. The pairings are the covariance count (was "causal minus acausal"), and the TD error and the annealed $\eta_e$ step come after the next frame's walk.
  - **§9 item 2:** the mean-field assumption needs the covariance count, not the balanced count. The note that it is untested in spiking is replaced by the M1 cosine of 0.992.
- **`docs/SPEC.md`:** the C2 critic line is updated to match Eq. 4c.

### Changed (code cleanup)

- **Behaviour-preserving cleanup** by the `codebase-cleaner`, with no change to numerics, RNG-call order, CLI flags, result-file names or JSON keys. `pytest -q tests` passes before and after (9/9, `golden.json` untouched). `m2.summary()` output and a short `m2.run` (flow and spiking, results written to a scratch directory) are byte-identical to the baseline.
  - **Lint:** `ruff check .` goes from 21 findings to 0. Long lines are wrapped (`m1.py` keeps the same AST), imports are sorted (`m2_viz.py`, `fdcheck.py`), `pairings` uses `lag` instead of `l`, the `core` self-check drops its semicolons and binds `beta` in `logP`, and `fdcheck.map_L` uses a `def` instead of a lambda.
  - **Pyright:** 18 errors go down to 16, after an annotation on `m2.Agent.evaluate`'s output dict. The rest are inference limits on `sum()`/`0` accumulators and optional arguments, and are left alone.
  - **Names:** `core.flow` names its iterate `pi` instead of `x`. `m2.Agent.C` becomes `Agent.ctrl`, since `C` means causal counts in `spiking`.
  - **Structure:** `m2.Agent.episode` is split along derivation §8 into `pre_td_updates` (module statistics, Eq. 4b trace, Eq. 4, Eq. 4d, Eq. 4a dual) and `critic_step` (TD error and NLMS critic). These names come from the expert review below; the cleaner first called them `local_plasticity` and `td_step`. Operation order is unchanged. The dual-target fraction 0.75 is now `m2.Q_STAR_FRAC`. Nested conditional expressions in `Agent.__init__` and `summary()` are now `if` statements.
  - **Docstrings:** added to functions that had none, with citations (`spiking.build`, `tau`, `pi_hat`, `run_frame`; `core._plogp`; `m2.Agent` and its methods, `random_return`, `run`, `summary`; `m1` helpers; `m2_viz.first_race_step`). The `m2.py` header now reads "models v1-v3".

### Added (code hygiene)

- **`.claude/agents/codebase-cleaner.md`:** adapted from another project for Map-STDP. Changes:
  - **Scope and layout:** the scope, the forbidden paths, and this repo's layout and dependency direction.
  - **Checks:** ruff, pytest and pyright in place of the original's tools.
  - **Hard limits for this code:** numerics, RNG-call order, the $W$/`W_snm` orientation, the crossbar-native rules, CLI and result-file formats, and no `strict=` on zips.
  - **Review:** it points to `/expert-review` instead of `design-guardian`, which doesn't exist here.
- **`tests/test_regression.py` and `tests/golden.json`:** a behaviour lock. Checked:
  - Fixed-seed runs are bit-identical across processes, spiking included.
  - The suite covers five closed-loop configurations, the module self-checks, pairing counts with the race readout, and `m2.summary()`.
  - Changing the weight floor by 0.01% fails all five closed-loop cases.
- **`ruff.toml`:**
  - **Rules pinned:** E, F, W, I and B, because ruff 0.16's defaults are much wider.
  - **B905 off:** `m2.evaluate` relies on zip truncation.
  - **No `ruff format`.**
  - **Baseline:** 21 findings, left for the cleaner.
- **`environment.yml`:** adds ruff and pytest; both are installed in the local env.

### Fixed

- **`m2.py --summary` crashed** on the replay recording, because its `m2_*.json` glob matched `m2_viz_episode_s3.json`. The viz outputs are renamed `viz_m2_*`.
- **`pyrightconfig.json`** no longer hard-codes the cluster env path. Pyright and the IDE now resolve imports from the active interpreter on any machine.

### Added (visualisation)

- **`experiments/m2_viz.py` and `experiments/m2_viz_template.html`:** a replay page of a trained spiking v3 agent. The left pane shows spikes and walk hops on the network, with the controller drawn as its 6×6 (θ, θ̇) grid and the race winner marked; the right pane shows CartPole. Commands:
  - `train` saves the weights (`viz_m2_W_s<seed>.npz`, renamed from `m2_viz_*` so `m2.py --summary` no longer picks them up; seeds 1–4 have last-100 returns of 193, 143, 207 and 192);
  - `record` keeps the best of 20 frozen-weight episodes (seed 3: 302 frames);
  - `page` embeds the recorded episode into the template.

### Changed (environment)

- **Local workstation env:** Miniforge in `~/miniforge3`, with the `map-stdp` env created from `environment.yml` plus katex, pyright and CodeGraph 1.6.2 (telemetry off). The math hook, the self-checks and the CodeGraph index (148 nodes) work there. The M2 runs used a scratchpad venv with the same packages before this env existed.
- **`CLAUDE.md` and `environment.yml`:** the "environment.yml is stale" note is removed; the file already lists the current stack. The setup comments now cover any machine as well as the cluster.

### Added (iteration 002: M2 done)

- **M2 results.** 110 runs, 5 seeds × 1500 episodes each, run locally on a 72-core workstation with no SLURM; results are in `experiments/results/m2_*_{v1,v2,v3}_s*.json`. Model v3, mean of the last 100 episodes, against a random policy of 22:
  - **Flow:** exempt 287 ± 21, dual 327 ± 46, none 39 ± 11, $\lambda = 0$ 187 ± 15.
  - **Spiking:** exempt 184 ± 13, $\lambda = 0$ 77 ± 4.
  - **Race vs. Eq. 1d:** KL 0.010, against a sampling floor of 0.010.

  All M2 criteria are met; SPEC §11 marks M2 done. Full record: `docs/iterations/002-m2.md` (v1 baseline, reviews, v2, annealing test, v3, outcome).
- **Reviews** by `snn-expert` (spiking diagnostics) and `neuroscientist` (plausibility), merged in the iteration record.
- **Code:**
  - `mapstdp/core.py`:
    - `homeostasis_update` (Eq. 4d);
    - `beta` in `policy`, `action_score` and `eligibility_step`, for the sharpened policy (Eq. 1e). The self-check now verifies the $\beta = 3$ score against finite differences.
  - `mapstdp/spiking.py`: `race(..., burn)`, with a self-check.
  - `experiments/m2.py`:
    - `--model {v1,v2,v3}` (default v3), `--beta`, `--eta`, `--eta-tau`;
    - the variant goes into the result filename;
    - `--summary` groups by model and variant.

  The v1 result files were renamed with a `_v1` suffix.
- **`docs/derivation.md`:**
  - **§2.2:** homeostasis as its own term (4d); weak-synapse rectification does not depend on the floor value; soft bounds (**Proposal**).
  - **§2.4:** the race equals (1d) only after relaxation, so a burn-in is adopted; sharpened policy (1e) (**Proposal**).
  - **§4.5:** M2 numbers; sealing caveat (persistence 0.996 against about 80% intrinsic input in cortex); exit-floor dual (**Proposal**, M4); the dual's $\mu$ diverges.
  - **§4.6:** $\bar{h}_j$ is presynaptic; the $\tau_e$ timescale.
  - **§8:** burn-in, (4d) and annealing in the training loop.
  - **§9, item 14:** updated.
  - **Appendix A:** iteration-002 checks.
  - **Appendix B:** item 11.
  - **References** (each checked by the `neuroscientist`): Chistiakova et al. 2014, Frank 2006, Gold & Shadlen 2007, Gurney et al. 2015, Markov et al. 2011, Royer & Paré 2003, Turrigiano et al. 1998, van Rossum et al. 2000, Zenke & Gerstner 2017.

### Changed (model v1 → v3, `docs/model.md`)

- **Decision: $d_j$ homeostasis is its own term on all plasticity** (Eq. 4d, $\epsilon_h = 5$, independent of $\lambda$), not part of the structural baseline. Inside $\lambda$ it was about 100× too weak, and it vanished at $\lambda = 0$. The task term ratcheted $d_{max}$ to 13 (flow) and 19 (spiking). Now $d_{max} \le 1.04$. Its expectation is still a column rescaling: spiking Monte Carlo cosine 0.992 (preliminary).
- **Decision: race burn-in of 10 steps.** Without it, the race read the cold-start transient (KL 0.022 against a floor of 0.010). With it, KL is at the floor.
- **Decision: anneal the task rate,** $\eta = 10$ decayed as $\eta / (1 + e/300)$ over episodes, in both backends (v3). v2 used a constant rate with spiking $\eta = 5$, and its return fell: spiking 140 → 120, flow at $\lambda = 0$ 210 → 145. The v1 ratchet had acted as hidden annealing. Constant $\eta = 10$ gives spiking 88, annealed gives 184.
- **Decision: routing exemption stays the default.** The dual is within about one sd ($t = 1.6$, $p = 0.15$), its order relative to the exemption flipped between v2 and v3, and its $\mu$ never converges.
- **Rejected for now: the sharpened policy (Eq. 1e).** At the flow level it collapsed on 3 of 5 seeds at $\beta = 4$ and on 1 of 5 at $\beta = 2$. It stays a **Proposal**, needing an entropy floor and a matched spiking readout.
- **Deferred: the exit floor on module sealing.** Sealing costs no return on CartPole (`snn-expert`), but it is implausible as anatomy (`neuroscientist`). Revisit at M4.
- **`docs/SPEC.md`:** status line, M2 marked done, the M2 routing results, and ablations (homeostasis placement, task-rate schedule, temperature, exit floor, race burn-in).

### Added (implementation, iteration 001: M1)

- **Code:**
  - `mapstdp/core.py`: numpy flow-level reference;
  - `mapstdp/spiking.py`: SuperNeuroMAT backend;
  - `experiments/m1.py`: the six M1 checks, with results in `experiments/results/m1.json`.

  Both modules have assert-based self-checks. Built by `superneuro-expert`. `pyrightconfig.json` points pyright at the `map-stdp` env.
- **`docs/model.md`:** the current model, versioned. v0 is the spec; v1 follows M1.
- **`docs/iterations/001-m1.md`:** the M1 record, with build, results, reviews (`snn-expert`, `neuroscientist`) and decisions.

### Added (iteration 002: M2, in progress)

- **Code** (built by `superneuro-expert`; self-checks pass):
  - `mapstdp/core.py`: the CartPole conjunctive encoder, $d_j$ homeostasis, and the action-gated eligibility (Eq. 4b);
  - `mapstdp/spiking.py`: the race readout;
  - `experiments/m2.py`: the closed-loop CLI and `--summary`.
- **`docs/iterations/002-m2.md`:** status, run plan, preliminary numbers, and open problems:
  - a $d_j$ ratchet from the task term;
  - the race readout reading the early transient.
- **No M2 results yet.** The SLURM jobs on `secondary` sat pending on priority for over 4 h and were cancelled on the user's instruction. M2 continues on a GPU machine that needs no SLURM.

### Changed (portability)

- **Paths:** the hooks, `.mcp.json` and the skills no longer hard-code `/scratch/vikramr2`.
  - The hooks use the active env (`CONDA_PREFIX`).
  - `math_check.js` finds katex next to the running `node`.
  - CodeGraph and the skills resolve from `PATH`.

  Launch Claude from `conda activate map-stdp` on any machine.

### Changed (iteration 001 decisions)

- **Decision: pairing count $\kappa$ is the covariance count** (derivation §2.2), replacing the balanced count. Cosine with $-W \odot \nabla D$: 0.992 covariance, 0.979 causal, 0.847 balanced, reproduced at 5k frames. References added: Sejnowski 1977 and Kempter et al. 1999.
- **Decision: presynaptic normalisation** as an output gain $1/d_j$, plus $d_j$ homeostasis $\epsilon (d_j - 1)$ in the baseline (derivation §2.2).
- **Decision: spiking uses branching mode,** with $k = 100$. Frame windows are stated in spikes per module (error ≈ $1.3 / \sqrt{S_{mod}}$). The $4 \tau / \alpha$ estimate in §2.3 was optimistic.
- **Decision: conjunctive encoding** (SPEC §4, derivation §2.4). Per-dimension fields cap the CartPole policy at about 84 steps.
- **Decision: routing exemption is the M2 default** (SPEC §6, derivation §4.5). In the flow-level closed loop, the map term without protection collapsed learning (≈53), while exempt routing at $\lambda = 0.05$ reached ≈343 against ≈195 for $\lambda = 0$. These numbers are preliminary. The dual term stays a Proposal and becomes an ablation. A constant $\lambda$ replaces the ramp, which becomes an ablation.
- **SPEC:**
  - M1 marked done;
  - status line updated;
  - ablations updated (pairing count, normalisation, encoding, schedule);
  - "Controller in $D$" provisionally answered (input layer).

### Added

- **`map-stdp` conda env** (`~/.conda/envs/map-stdp`): Python 3.11 and Node.js 26, so the tooling has Node without a system-wide install.
- **ponytail plugin** (v4.11.0, [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)), installed at project scope through `.claude/settings.json`. It is a least-code skill. Its hooks were reviewed before install: they inject instructions and write small mode files under `~/.claude` and `~/.config/ponytail`, with no network calls and no subprocesses. It needs `node` on PATH, so activate the env before launching `claude`.
- **CodeGraph MCP server** (v1.6.2, [colbymchenry/codegraph](https://github.com/colbymchenry/codegraph)), installed in the env and registered at project scope in `.mcp.json`.
  - Telemetry is off, both by `codegraph telemetry off` and by `CODEGRAPH_TELEMETRY=0`.
  - The local index is in `.codegraph/`, which ignores itself.
  - The repo has no code yet, so the index is empty until implementation starts.
- **Project hooks** (`.claude/hooks/project_hooks.py` and `math_check.js`, registered in `.claude/settings.json`), each tested with sample events:
  - **Math check.** After any edit to `docs/*.md` or `CLAUDE.md`, KaTeX renders every math span in strict mode and banned macros (`\,` `\;` `\tag` `\{` `\}` `\|`) are flagged. Failures are fed back so they get fixed immediately.
  - **Changelog guard.** Blocks `git commit` when changes are staged without `docs/CHANGELOG.md`. Override with "skip changelog" in the message.
  - **No global installs.** Blocks pip, npm `-g` and conda installs that don't target the `map-stdp` env, a venv, or `--target`.
  - **Environment check.** At session start, warns if the `map-stdp` env or `node` is missing.
- **Project skills** (`.claude/skills/`):
  - **`/derivation-check`** (`scripts/fdcheck.py`): the finite-difference and Monte Carlo checks CLAUDE.md requires. Its selftest reproduces a map-gradient error of 7e-10 and a three-factor correlation of 0.99994.
  - **`/doc-edit`**: house rules for the docs.
  - **`/expert-review`**: runs the expert agents in parallel and merges their findings.
  - **`/changelog`**: drafts an entry in this file's style.
- **Official plugins at project scope:**
  - **math-proof**, for proofs of the derivation's theorems;
  - **pyright-lsp**, with pyright 1.1.414 installed in the env rather than globally;
  - **commit-commands**;
  - **hookify**.
- **`map-stdp` env:** gained katex, pyright, numpy and scipy.
- **SLURM template** (`slurm/job.sbatch`): the user's standard header (partition `zgdrasil`, 8 CPUs, 64 GB, 2 days). It activates `map-stdp` and runs the given command. `logs/` is gitignored.
  - CLAUDE.md gains a "Compute (SLURM)" section: long runs go to `sbatch`, never the login node.
  - The `snn-expert` and `superneuro-expert` agents point to it.
  - **Decision:** jobs go to the `secondary` partition (4 h, preemptible), on the user's instruction. Installs run on the login node.
- **Env moved to `/scratch/vikramr2/conda/envs/map-stdp`.** The `/u` home hit its 500k-file quota at 503k files (`~/.conda/pkgs` held ~224k).
  - `conda clean --all` freed ~107k files.
  - The env was rebuilt on scratch, with its package cache there too, and registered through `envs_dirs`. It now also has numba, gymnasium, matplotlib, networkx and superneuromat 3.5.0.
  - Hooks, skills and `.mcp.json` point to the new path.
  - `environment.yml` is rewritten to match; it had listed the old PyTorch stack.
  - The old home env is removed once Claude is relaunched from the new env.

## 2026-10-04

### Changed (simulator)

- **Decision reversal: SuperNeuroMAT replaces Brian2** as the spiking simulator (SPEC §3). Reasons:
  - its discrete-time matrix update makes one step one hop, and one crossbar read;
  - the per-frame closed loop is plain Python;
  - the plasticity code can be shared with the numpy reference;
  - it has CPU, JIT and GPU backends;
  - it can export to the NeuroCoreX FPGA platform (inference only).

  Brian2 stays as a fallback for continuous-time checks. SuperNeuroABM is an option for heterogeneous neurons.
- **Smoke test** (SuperNeuroMAT 3.5.0, installed only in the scratchpad):
  - `weight_mat()` is `[pre, post]` (the transpose of $W$);
  - built-in `aneg` is a per-step decay on non-coincident synapses, not acausal STDP, so Map-STDP is applied per frame outside `simulate`;
  - `add_spike` times are relative to the current step;
  - cost per frame is about 1 ms at ~100 neurons, 40–50 ms at ~500 and ~230 ms at ~1000, dominated by weight get/set in Python at larger sizes.
- **`docs/derivation.md` §2.3:** a discrete-time note. One step with delay 1 is one hop. A branching-process mode (infinite leak, stochastic input) is a Proposal for M1.
- **`CLAUDE.md`:** simulator, submodule, transpose convention, and no global installs.
- **`.claude/agents/snn-expert.md`:** tooling is no longer Brian2-specific.

### Added (simulator)

- **`.claude/agents/superneuro-expert.md`**: a subagent expert in coding SuperNeuroMAT and SuperNeuroABM. Its API facts were checked against the source, and it follows the project constraints.

### Changed (review revisions)

Applied the findings of the neuroscientist and SNN-expert reviews of `docs/derivation.md`.

- **Decision: new task term.** TD-modulated STDP with an action-gated eligibility trace (derivation Eq. 4b, 4c, §4.6) replaces $(R - \bar{R}) e_{ij}$. That term gets no action credit when the action is sampled from flow shares, and stalls under sparse reward. The new eligibility has the same crossbar form as the structural rule.
- **Proposal: dual routing term** (Eq. 4a, §4.5). The map term alone starves controller→action and controller→latent routing, because that routing is cross-module. The dual term holds routing at a target $q^{\ast}$, and the simplest variant exempts the routing pairs. It stays a Proposal until simulated.
- **Spiking timing.** Spiking frames take about $4\tau/\alpha$ (100–250 ms), not 20 ms. The RL ladder runs on a numpy flow-level reference first, and Brian2 is used for fidelity at M1 and M2.
- **Mean field.** The doc now states its three requirements: constant $d_j$, global normalisation, and causal-minus-acausal pairing counts.
- **Locality (§6).** Restated: the postsynaptic module is the local one. The doc also notes what neuromodulators can carry, and that the per-neuron baseline equals presynaptic renormalisation to first order.
- **Readout and memory.**
  - A race readout matches Eq. 1d, and the limits of the policy family are stated.
  - Linear carry-over forgets geometrically; a module-level nonlinearity is proposed.
- **Lateral inhibition (§7)** is routed through interneurons with inhibitory STDP, for Dale's law.
- **Training (§8)** gains stability measures and a $\lambda$ ramp. **Assumptions (§9)** gain items 12–15.
- **Claims softened:** controller ↔ cortex (now a functional abstraction with basal-ganglia, OFC and hippocampal analogues), the characterisation of Buesing et al., the metabolic reading of the map equation, Friston, and Lynn et al. (an fMRI measure).
- **References.** 14 added, each checked by title, venue and year during the review.
- **New numerical checks (Appendix A):**
  - action-gated eligibility: cosine 0.997 with the true $\nabla \log P$, and the Monte Carlo correlation is 0.997;
  - baseline as renormalisation: first order, relative difference $2.5\lambda$;
  - dual-term gradient: exact;
  - carry-over memory decays 0.29–0.39 per frame.

  The preliminary spiking and bandit numbers come from the review's own simulations and still need reproducing.
- **`docs/SPEC.md`:**
  - architecture analogies softened;
  - spiking timing corrected;
  - task-term decision recorded;
  - routing conflict and dual Proposal added;
  - new open questions (controller in $D$, $q^{\ast}$, nonlinear carry-over, device multiplicativity, Dale's law, efference copy);
  - new ablations;
  - M1 and M2 re-scoped.
- **`CLAUDE.md`:** task term and routing-conflict notes.

### Added

- **`.claude/agents/snn-expert.md`**: a subagent expert in SNNs, local and three-factor learning, Brian2 and closed-loop RL, focused on improving performance within the crossbar-native constraints. It can run and edit code.
- **`.claude/agents/neuroscientist.md`**: a read-only subagent that reviews biological plausibility (locality, timescales, neuromodulator specificity, Dale's law, strength of analogies) with verified citations.

### Changed

- **Reformulation: communities as states.** The primary goal is now a state-space abstraction for spiking networks in which communities are the states. Map-STDP is framed as the **inverse** of community detection: the partition is given, and the rule learns weights that make it the minimum-description-length partition of the walk. The thermodynamic surrogate hypothesis becomes a secondary side tie, and the energy proxy a secondary metric.
- **`docs/derivation.md`:**
  - New title, goal and motivation. The controller community is motivated by the cerebral cortex, with Global Workspace Theory kept as an analogy.
  - Notation adds module types $\mathcal C$ (controller), $\mathcal A$ (actions) and $\mathcal L$ (latent), frame index $f$, hops $n$, and module-pair flows $J_{ba}$, $T^K_{ba}$.
  - New §2.3, one walk per frame:
    - per-frame flow $\pi_f$, computed implicitly by $n$ hops with error $\le 2(1-\alpha)^n$ (Eq. 1a);
    - software, crossbar and spiking realisations;
    - timescale ordering (hops, then frames, then plasticity);
    - a carry-over proposal, Eq. 1b, so latent communities persist across frames.
  - New §2.4, the extracted Markov model:
    - the module chain $T^K(\mathbf o)$ (Eq. 1c);
    - why a central controller makes it stimulus-conditioned;
    - policy readout from action communities (Eq. 1d);
    - a proposed across-frame latent chain as a world model.
  - New §3.4: the inverse problem $\min_W \mathbb{E}_f[L(M; \pi_f)]$ (Eq. 3a). Theorems 1–3 hold frame by frame, and re-detecting latent modules makes training an alternating minimisation.
  - §5: candidates table gains a "state abstraction" column. New §5.9 proposes the entropy rate of the latent chain as a candidate; its modulator is not yet derived.
  - §6: per-frame walk row in the crossbar table.
  - §8: training loop restructured by frame. Only latent modules may be re-detected.
  - §9: assumptions 9–11 added (convergence within a frame, carry-over, number of latent modules).
  - Appendix A and Appendix B item 8 added.

  New results, each checked numerically on a 15-node, 5-module network:
  - the per-frame and frame-averaged map-equation gradients match finite differences ($\le 10^{-9}$);
  - the three-factor rule averaged over frames matches $-W \odot \nabla D$ (correlation 0.99996, or 0.99993 with a pooled modulator);
  - modular networks contract at nearly the $1-\alpha$ bound (0.78 vs. 0.8);
  - with carry-over held fixed, the gradient has cosine 0.98 with the full gradient;
  - the controller column of $T^K$ carries most of the stimulus dependence.
- **`docs/SPEC.md`:**
  - §1 has the new goal and hypotheses.
  - §2 has the new architecture: controller, latent and action communities, with carry-over.
  - §4 is RL-only. The ladder is CartPole → Acrobot/MountainCar → LunarLander → a partially observable task. **Vision is deferred.**
  - §5 merges stimulus and timescale. Implicit walk per frame and carry-over are **Proposals**.
  - §6: the candidates table gains a state-abstraction column and the $h_{\mathcal L}$ proposal. Decision criteria now include extracted-model quality, and energy is secondary.
  - §8: pinned vs. free latent partitions; $\lvert \mathcal L \rvert$; the world-model open question.
  - Metrics and ablations are split into §9 and §10 and extended (policy KL, implicit vs. spiking walk, carry-over, latent communities).
  - Milestones M1–M6 are re-scoped.
- **`CLAUDE.md`:** research question rewritten; conventions for per-frame flow and module types.

## 2026-10-01

### Added

- **`docs/review/search.py`** and **`docs/review/data/`**: the Phase 2 search script and its output. It queries OpenAlex, PubMed and arXiv and returns 1,870 records, which come to 1,276 after deduplication.
- **`docs/review/screening/`**: screening rules (clarifications of protocol §B.2), the 50-record calibration set, a blank author sheet, and the AI's calibration decisions, committed before the author screens.
- **`docs/review/protocol.md`**, a draft PRISMA-P protocol (frozen at v1.2 after author approval; not registered, git commit serves as the timestamp) for a systematic review of how neural signalling dissipates energy and what evidence links information-theoretic measures to metabolic or thermodynamic cost. It covers three sub-questions (mechanism, information–cost linkage, network level) and separates cost classes M/T/W/P with no pooling across classes. It also adds a gap check for description-length measures, including the map equation, and records Devil's Advocate Checkpoint 1. The protocol is awaiting author approval, and no search has been run beyond pilot counts.

### Changed

- **Review protocol v1.3:** the search strings were revised after the seed recall check missed 3 of 10 eligible seeds. The revised strings retrieve all 10. Details are in the protocol's §B.13.

## 2026-09-27

### Removed

- **The original LaTeX derivation** and its template files, superseded by `docs/derivation.md`. Its training-loop figure was dropped because it depicted the old per-sample E-step/M-step loop.
- **The PyTorch prototype in `models/`.** It had:
  - an SBM starting-topology generator;
  - a shared LIF simulation with an E-step/M-step training loop;
  - 4-bit suppression STDP (Gautam & Kohno 2023);
  - two Map-STDP variants trained on N-MNIST: fixed-community ($G_{sim}$) and lateral-inhibition ($G_{LI}$).

  The project restarts from ideation. The code remains in git history (last at `fcedb0a`).

### Added

- **`docs/derivation.md`**, a corrected markdown rewrite of the original LaTeX derivation. The rewrite:
  - fixes a sign error in the expansion of $L(M)$. The gradient's log factor becomes $\log\frac{q_\curvearrowright(p_m+q_m)}{q_m^2} \ge 0$, verified by finite differences.
  - adopts a single direction convention: $W_{ij}$ is pre $j$ → post $i$, and the walk follows spikes forward. As a result, $G_{sim}(i,j) = \mathbb I[i\notin m(j)] - \bar e_j$ is indexed by the presynaptic neuron.
  - adds the stimulus as a teleportation term, $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$.
  - turns the identification of the firing marginal with walk flow into an explicit mean-field assumption.
  - replaces the fixed-point "convergence" proof with a contraction proof.
  - restates the lateral-inhibition variant with a soft indicator $\chi_{ij}$, and flags the problems with its anti-Hebbian rule.
- **`docs/SPEC.md`**, the project spec: goal and hypothesis, architecture, and five considerations with proposals (Brian2; the RL and vision task ladder, with classification as a contextual bandit; stimulus as teleportation; candidate description lengths vs. thermodynamic entropy; 3-factor variants). It also lists metrics, ablations and milestones M0–M6.
- **`docs/CHANGELOG.md`**, this file.

### Changed

- **`CLAUDE.md`**, rewritten for the docs-first phase. It now also lists math-rendering conventions.
- **`docs/derivation.md`**, consolidated to put interpretation first:
  - an "At a glance" section with the rule, a sign/magnitude table and a worked example;
  - a notation table saying where each quantity lives;
  - an intuition for the map equation as a two-level code, and the fact that leakage always costs bits;
  - a single assumptions list;
  - the corrections to the earlier version moved to Appendix B.

  New result: $\sum_i W_{ij} G(i,j) = 0$, so a multiplicative map term conserves each neuron's outgoing weight (checked numerically). All math now renders in KaTeX, GitHub and Markdown Preview Enhanced: no `\tag`, `\,`, `\{` or `\|`, and every span is validated with KaTeX.
- **`docs/SPEC.md`**, cross-references updated to the new section and equation numbers. $G_{sim}$ is renamed to $G$, and math spacing is fixed.
- **Decision: structure is a neuromodulated 3-factor rule.** The structural term is now cost-modulated STDP: $\Delta W_{ij} = -\lambda (g_{ij} - \bar{g}_j) \kappa_{ij}$. The neuromodulator broadcasts the *marginal* description cost $g_{ij} = \partial D / \partial J_{ij}$, a $K \times K$ module-pair table.
  - Chosen over a factorized-gradient hardware rule because of the new memristive-crossbar target.
  - The exact gradient remains the analysis reference and simulation baseline.
  - Supersedes SPEC 3-factor variants A/B/C.
- **`docs/derivation.md`**, restructured around the rule:
  - new §4: the rule, and Theorem 3, which shows the expected update is $-\lambda W_{ij} \partial D / \partial W_{ij}$; marginal vs. pointwise cost;
  - new §5: eight candidate description lengths as modulators (map equation, entropy rate, entropy production, cost-weighted map equation, Markov stability, SBM description length, predictive dissipation, cross-module information flow);
  - new §6: crossbar mapping;
  - neuromodulator/energy motivation with verified citations.

  New results, each checked numerically:
  - $M^{\ast}_m$, previously absorbed into the learning rate, is the map equation's per-module neuromodulator, firing only on module exits.
  - $G$ is exactly the lag-1 Markov-stability gradient.
  - Broadcasting pointwise codeword length is biased: correlation 0.976 with the true descent direction, against 0.9999 for the marginal cost.
  - Coarse-grained $\sigma_K$ lower-bounds $\sigma$, but $h_K$ has no general ordering with $h$.
- **`docs/SPEC.md`**, restructured:
  - §6 merges C4 and C5 into "Description length as a neuromodulator", with a candidate table and decision criteria;
  - new §7 (C6) sets memristive crossbars as the hardware target, with its constraints;
  - ablations, M1 and M5 updated.
- **`CLAUDE.md`** adds the crossbar-native constraint.
