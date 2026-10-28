# Iteration 016: when do columns help? Acrobot and Taxi

**Status:** pilots done (§7–§9). Study `hybrid-columns-2` (Acrobot only) is drafted, not launched. Pilots and the study start after the `hybrid-columns` study (iteration 015) finishes, so that they do not share its CPUs. The plan is `~/.claude/plans/jiggly-chasing-sparkle.md`.

**Question (user, 2026-10-27):** across tasks, when does each regime of modularity help?
- **implicit:** the task alone shapes the columns (`cols_lam0`);
- **learned:** the map term shapes them (`cols_on`);
- **enforced:** the columns are sealed (`cols_sealed`).

Two tasks (CartPole, LunarLander) cannot show a pattern, so this iteration adds two tasks chosen by **how their actions relate**, not by difficulty.

## 1. Working hypothesis (from iteration 015, interim)

- **CartPole:** 2 opposite actions, a purely reactive policy. All regimes keep performance.
- **LunarLander:** 4 actions that are combined and sequenced. At about 4600 episodes, task-formed columns are ahead of base (last-100 about 34 vs 23); map-term and sealed columns are level with base. In pilot C the task term built a hub into the main-engine column.
- **Hypothesis:** sealed columns cost nothing when actions are independent alternatives. Cross-column links help when actions must work together.

The final 015 numbers replace these before pre-registration.

## 2. Tasks (user decision, 2026-10-27)

| Task | Actions | How the actions relate |
| --- | --- | --- |
| Acrobot-v1 | 3: torque −1, 0, +1 | exclusive, but must be **sequenced** (energy pumping) |
| Taxi-v4 | 6: 4 moves, pickup, dropoff | **hierarchical**: sub-goals (reach the passenger, pick up, reach the destination, drop off); two functional groups of actions |

- **Acrobot** separates sequencing from combining.
- **Taxi** is the classic hierarchical RL task (Dietterich's MAXQ). It also adds a different domain: a grid world with a discrete state, where the policy is effectively a lookup table.
- **Rejected:**
  - a discretised Pendulum (with CartPole and Acrobot, a third pendulum);
  - MountainCar (sparse reward; a base arm at floor would make every arm level);
  - MinAtar (needs a new package; the actor's linear policy risks a floor).

## 3. Predictions (draft; fixed in the study README)

- **P-seq (Acrobot):** if sequencing alone makes cross-column links useful, `cols_lam0` beats `cols_sealed`. If sealed matches, combining actions is what mattered on LunarLander.
- **P-hier (Taxi):** `cols_lam0` and `cols_on` beat `cols_sealed`.
- **P-group (Taxi structure, `rec.RK`):** cross-column weight is concentrated within the movement group, not between the movement and passenger groups.
- **Headline contrasts:** as in 015, each regime vs base, with Holm within families. The margin Δ is restated per task at pre-registration: the AUC scales are about −500 to −60 (Acrobot) and −770 to +8 (Taxi).

## 4. Settings: pre-set rules, never tuned on return (`experiments/tdstdp.py`, `PRESETS`)

No authors' configs exist for these tasks.

- **Populations, α_a, τ_d, freeze, init:** CartPole's (`n_ce` 40, `n_ae` 20 per action, no I posts, one copy, α_a 50, τ_d 1000 ms, 20 ms, U(0, 0.1)).
- **Step:** `--eta-cal`, with LunarLander's Adam steps (critic 1.25e-4, actor 6.25e-5 mV) as the step to match. These are the authors' only Adam steps.
- **Low-rate scaling k:** `r_ck_hz` = 140 Hz, CartPole's BindsNET rate (015 study, 20 seeds: 138–142), so k = 3.5 and the logits keep CartPole's range. A task's own BindsNET rate would tie the scaling to an engine these tasks were never tuned on.
  - Measured BindsNET rates: Acrobot 204 Hz, Taxi 55 Hz.
- **Encoders:**
  - **Acrobot:** LunarLander's (per-dimension order 1, ON/OFF split, constant unit, `in_rep` 16), giving 208 inputs. CartPole's order-2 cross terms would give 3^6 = 729 features. Bounds are the M3 `FOURIER_BOUNDS`.
  - **Taxi:** a one-hot of the state index (`--onehot`, 500 inputs). `in_rep` 1 is the smallest replication, and it is feasible: `calibrate_s_in` reaches 40 Hz with s_in 11.8. `in_rep` 16 also works (s_in 0.72) but makes 8000 inputs.
- **Reward scale:** `r_scale` = 0.02 / RMS of the per-step reward under a random policy (20–50k steps). CartPole's RMS is 1.
  - Acrobot: RMS 1.00, `r_scale` 0.02.
  - Taxi: RMS 5.74, `r_scale` 0.0035.
- **Critic floor:** β_c = min(−0.2, 1.2·V_rand), where V_rand = r_scale·E[r]·τ_d / 20 ms is the value of a random policy (user decision, 2026-10-28, after pilot v1, §7).
  - The critic reads V = α_c·rate + β_c ≥ β_c, so β_c must lie below the values the task needs.
  - Resulting β_c: Acrobot −1.2 (V_rand −1.00), Taxi −0.82 (V_rand −0.68).
  - It is consistent with the authors' configs: LunarLander β_c −2.0 vs V_rand −2.31; CartPole −0.2 with V_rand +1.0.
- **Exploration:** c_e = 1e-4, LunarLander's entropy bonus (Eq. 26). It is the authors' only exploration setting beyond CartPole's config (user decision, 2026-10-28, after pilot v2, §8).
- **Episodes (fixed, no early stop):** Acrobot 500; Taxi 2000, from the tabular-RL convergence range.
- **Solve (secondary):** trailing-100 mean ≥ gymnasium's `reward_threshold` (Acrobot −100, Taxi 8).
- **Primary metric:** AUC (`report.TD_PAIR_METRICS`: AUC on every task except LunarLander).

## 5. Study `hybrid-columns-2` (declared in `experiments/report.py`, not pre-registered)

- **Arms:** `base`, `cols_lam0`, `cols_on`, `cols_sealed`. `lat_on` is added only if 015 gives a reason, decided before launch.
- **Flags:** `--rate-target 40 --block 20 --fast --eta-cal` (+ arm flags).
- **Seeds:** 4000–4019, fresh. 2 tasks × 4 arms × 20 = 160 runs.
- **Pairs:**
  - headline: each regime vs base;
  - secondary: `cols_on` vs `cols_lam0`, `cols_on` vs `cols_sealed`, `cols_lam0` vs `cols_sealed`.

## 6. Pilot (after 015; dev seeds 0–2, `experiments/results/pilot016/`)

Gates are on stability and structure only:
- rates hit 40 Hz;
- max Re λ ≤ 0.5, τ_loop ≤ 40 ms;
- no NaN;
- base is not at floor by the episode cap.

If base is at floor, report to the user before changing anything.

## 7. Pilot v1 (2026-10-28; `experiments/results/pilot016/v1/`, superseded)

24 runs: seeds 0–2, 2 tasks × 4 arms.

- **Taxi:** all 12 crashed when writing the result (`n_in` was a numpy int, not JSON-serialisable). Fixed in `ck_encoder`.
- **Acrobot:** at floor in every arm. The return was −500 (the goal was never reached) in 12 of 12 runs.
  - **Cause:** with CartPole's β_c = −0.2, the critic cannot represent Acrobot's negative values (V_rand −1.0). The critic posts went silent (rate 0.000 from episode 50), V sat at β_c and the TD error stayed negative. The policy collapsed (entropy 0.001).
  - **Stability gates passed:** rates 41 Hz, max Re λ ≤ 0.28, no NaN.
  - **Structure (final within-column share):** `cols_lam0` 0.83–0.86, `cols_on` 0.998, sealed 1.
- **Decision (user):** the β_c rule of §4, then pilot v2 with the same seeds.

## 8. Pilot v2 (2026-10-28; β_c rule; `experiments/results/pilot016/v2/`, superseded)

**Stability gates pass:** rates 40–41 Hz, max Re λ ≤ 0.28, no NaN.

**The policy collapses before it has explored:**
- **Acrobot:** 3 of 12 runs learn (base 1 of 3).
  - In base seed 2, entropy was 0.32 at episode 50 and the return reached −120 by episode 150.
  - In seeds 0 and 1, entropy fell to 0.09 at episode 50 and 0.001 later: a single fixed torque, return −500.
- **Taxi:** every run learns to avoid illegal actions (−350 to −200) but no run delivers a passenger (+8 needed). Entropy is 0.02 by mid-run.
- **Structure (final within-column share):**
  - Taxi: `cols_lam0` 0.46–0.53, `cols_on` 0.996, sealed 1;
  - Acrobot: `cols_lam0` 0.81–0.82.

**Decision (user):** add LunarLander's entropy bonus (§4) and run pilot v3 on the same seeds. If base still collapses (leaves the floor in fewer than 2 of 3 seeds), report to the user rather than trying further fixes.

## 9. Pilot v3 (2026-10-28; entropy bonus; `experiments/results/pilot016/`)

**Acrobot passes:**
- Base learned in 3 of 3 seeds: last-100 −110 to −145, AUC −163 to −166.
- Column arms learned in 7 of 9 seeds. Two collapsed to a fixed torque at −500: `cols_lam0` s1 and `cols_sealed` s0.
- Gates: rates 41 Hz, max Re λ ≤ 0.28, no NaN.
- Final within-column share: `cols_lam0` 0.65–0.89, `cols_on` 0.996–0.998, sealed 1.

**Taxi fails the floor gate:**
- Base delivered 0 passengers in 6000 episodes; 3 deliveries across all 12 runs.
- Every arm learns only to avoid illegal pickups and dropoffs, reaching the −200 time-limit plateau, so the hierarchy is never exercised.
- **Likely cause:** the one-hot input learns each of the 500 states separately, from rare visits, at a small no-Adam step.

**Decision (user, 2026-10-28):**
- Run the study on Acrobot alone (`hybrid-columns-2`, 4 arms × 20 seeds).
- No more Taxi fixes inside this study. A replacement for the hierarchical slot, or a Taxi variant, gets its own pilot later.
- The Taxi presets stay in the code, tested, for that later work.

## 10. Collapse in the column arms, and per-arm rate matching (2026-10-28)

**The problem:** in pilot v3, 2 of 9 Acrobot column runs collapsed to a fixed torque (−500); no base run did. The user asked for the cause and a fix.

**Diagnosis:**
- **The column arms start hotter.** `calibrate_s_in` probes without recurrence, so all arms share s_in, and the recurrent kicks add drive on top.
  - Actor rate at episode 50: about 0.047 per ms in every column run vs 0.039 in base.
  - At a shared s_in, the initial post rate with R on is 46 Hz on Acrobot and 43–44 Hz on LunarLander.
- **A sharper policy follows.** The policy is softmax(α_a × column rate), and the within-column loop (max Re λ ≈ 0.27) amplifies rate differences between columns.
  - Entropy at episode 50: median 0.29 with columns vs 0.58 in base.
  - Both collapsed runs were below 0.16.
- **The deterministic corner absorbs.** (A − p) → 0 when p → 1, and so does the entropy gradient (Eq. 25). The actor's update vanishes, the critic goes silent and the policy never recovers (action repeat 1.00).
- **Implication for iteration 015:** its column arms had the same higher starting rate. The early LunarLander speed-up (AUC +9 to +11 over the first 3000 episodes) was as large for sealed columns, which fits a gain effect as well as a structural one.

**Fix (user decision): `--rate-match-rec`.** `probe_rate` runs with the arm's R at init, so `calibrate_s_in` brings every arm to the same initial post rate.
- It is a pre-set rule, the same for every arm (a no-op for base).
- Check at seed 0 (initial rate with R on):

  | Arm | Shared s_in | Matched s_in |
  | --- | --- | --- |
  | Acrobot cols | 46.0 Hz | 42.4 Hz |
  | Acrobot sealed | 46.1 Hz | 44.8 Hz |
  | LunarLander cols | 43.7 Hz | 39.9 Hz |
  | LunarLander sealed | 42.8 Hz | 40.1 Hz |

  On Acrobot the closed-loop rate is steep in s_in, so the match is looser.

**Pilots** (`experiments/results/pilot016/rm/` and `ll_rm/`):
- **Acrobot:** 4 arms × dev seeds 0–9, all rate-matched. Measures the collapse count per arm and the entropy at episode 50.
- **LunarLander:** base, `cols_lam0` and `cols_sealed`, each unmatched and matched, over dev seeds 10–15 and 3000 episodes. Asks whether the early speed-up survives rate matching.

**Results** (development, small n; analysis script in the session scratchpad):

- **Acrobot, rate-matched** (seeds 0–9; AUC = mean return over 500 episodes):

  | Arm | Collapsed (last-100 = −500) | AUC median [range] | AUC < −300 | Actor rate, ep 50 | H(s), ep 50 (median, min) |
  | --- | --- | --- | --- | --- | --- |
  | base | 0/10 | −163 [−271, −134] | 0 | 0.038 | 0.52, 0.28 |
  | `cols_lam0` | 1/10 | −234 [−495, −158] | 4 | 0.040 | 0.35, 0.14 |
  | `cols_on` | 1/10 | −225 [−495, −162] | 2 | 0.039 | 0.33, 0.11 |
  | `cols_sealed` | 1/10 | −181 [−499, −155] | 2 | 0.041 | 0.40, 0.12 |

  - Rate matching removed the rate gap (0.040 vs 0.038 per ms at episode 50), but not the entropy gap or the slow and collapsed runs.
  - HL shift vs base: `cols_lam0` −73 (MW p 0.011), `cols_on` −61 (p 0.009), `cols_sealed` −22 (p 0.27).
  - **So the starting rate is not the cause.** The policy is sharper at the same mean rate, so the likelier driver is the within-column loop amplifying rate *differences* between columns (a gain on the contrast, not on the mean). This is a hypothesis, not tested here.
- **LunarLander** (seeds 10–15, AUC over 3000 episodes; base −58.2):

  | Arm | AUC mean | HL shift vs base | MW p | Last-100 mean |
  | --- | --- | --- | --- | --- |
  | base | −58.2 | – | – | 14.6 |
  | `cols_lam0` | −45.7 | +11.0 | 0.026 | 50.8 |
  | `cols_lam0` matched | −49.7 | +5.9 | 0.31 | 37.4 |
  | `cols_sealed` | −46.4 | +12.9 | 0.026 | 28.0 |
  | `cols_sealed` matched | −54.0 | +4.9 | 0.13 | 25.2 |

  - The unmatched arms reproduce 015's early speed-up (+11 to +13).
  - Rate matching roughly halves it (paired by seed: −4.0 for `cols_lam0`, −7.6 for `cols_sealed`). With n = 6 per arm the remainder is not significant.
  - **So part of 015's early LunarLander gain is a gain (starting-rate) effect.** Whether a structural part remains needs more seeds.

## 11. Per-arm entropy matching (2026-10-28)

**Check: policy entropy at init** (W and R frozen, rule off, after each arm's own calibration; CAL stream; scratch probe):

| Task | Arm | H₀ (nats; median) | Logit spread (SD across actions) |
| --- | --- | --- | --- |
| Acrobot (max ln 3 = 1.10) | base | 0.977 | 0.49 |
| | cols, shared s_in | 0.928 | 0.59 |
| | cols, rate-matched | 0.923 | 0.61 |
| | sealed, rate-matched | 0.898 | 0.67 |
| LunarLander (max ln 4 = 1.39) | base | 1.274 | 0.46 |
| | cols, rate-matched | 1.251 | 0.52 |
| | sealed, rate-matched | 1.234 | 0.55 |

- **Confirmed:** at the same mean rate the column policies start sharper. The within-column loop amplifies the logit spread (+25 to +37% on Acrobot), and rate matching does not change that.
- **Qualified later (§13):** at init about 93% of the logit variance is rate noise, so what the loop amplifies is mostly correlated noise within a column, not a signal difference between columns; the noise-only logit SNR is about 10% lower with columns. Entropy matching therefore calibrates noise, and stays fixed while the loop gain changes over training.
- This acts as a larger α_a, the softmax gain.

**Fix (user decision): `--ent-match-rec`.**
- `calibrate_low` multiplies α_a by f, found by `match_entropy` (bisection on log f) so that `probe_entropy` with the arm's R equals the base run's at the same s_in.
- It is a pre-set rule, a no-op for base, and recorded in `low_rate` (`ent_scale`, `h_base`, `h_got`).
- With `--eta-cal`, η is normalised by the RMS of the update at the arm's α_a, so the logit step per update is also matched.
- Seed checks (with `--rate-match-rec`): Acrobot f 0.81 (cols), 0.70 (sealed); LunarLander f 0.89 (cols), 0.82 (sealed). The matched entropy is within 0.012 nats of base.

**Pilot** (`experiments/results/pilot016/em/` and `ll_em/`; both flags on; base from `rm/` and `ll_rm/`, since neither flag changes base):
- Acrobot: the three column arms × dev seeds 0–9.
- LunarLander: `cols_lam0` and `cols_sealed` × dev seeds 10–15, 3000 episodes.

**Results** (development, small n; HL shift and MW p vs the same base runs):

- **Acrobot** (seeds 0–9; AUC over 500 episodes; base median −163):

  | Arm | Matching | Collapsed | AUC < −300 | AUC median | HL vs base | MW p | H(s), ep 50 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | base | – | 0 | 0 | −163 | – | – | 0.52 |
  | `cols_lam0` | rate | 1 | 4 | −234 | −73 | 0.011 | 0.35 |
  | `cols_lam0` | rate + entropy | 0 | 2 | −200 | −32 | 0.045 | 0.43 |
  | `cols_on` | rate | 1 | 2 | −225 | −61 | 0.009 | 0.33 |
  | `cols_on` | rate + entropy | 0 | 2 | −216 | −47 | 0.021 | 0.47 |
  | `cols_sealed` | rate | 1 | 2 | −181 | −22 | 0.27 | 0.40 |
  | `cols_sealed` | rate + entropy | 0 | 1 | −217 | −51 | 0.031 | 0.54 |

  - Entropy matching removed the collapses (0/30 vs 3/30) and restored the entropy at episode 50.
  - The column arms are still below base (HL −32 to −51).
  - It is not perseveration: action repeat is below chance in every column arm (0.60–0.63 vs 0.63–0.66).
  - The cause of the remaining deficit is open.
- **LunarLander** (seeds 10–15; AUC over 3000 episodes; base −58.2):

  | Arm | Unmatched | Rate | Rate + entropy |
  | --- | --- | --- | --- |
  | `cols_lam0` | +11.0 (p 0.026) | +5.9 (p 0.31) | +3.2 (p 0.39) |
  | `cols_sealed` | +12.9 (p 0.026) | +4.9 (p 0.13) | +5.5 (p 0.24) |

  - With both matchings, the early gain is +3 to +6 and not significant at n = 6. **Most of 015's early LunarLander speed-up was a gain (rate and temperature) effect.**

## 12. Study `hybrid-columns-2` launched (2026-10-28)

- Pre-registered 15:25 EDT (README approved by the user, Δ = 25 confirmed); code at commit `8d0a9e4`.
- Worktree `../map-stdp-hc2`, branch `hybrid-columns-2-study`.
- Command: `pytest -q tests && python -m experiments.report run hybrid-columns-2 --workers 64` (local workstation, threads 1), log `logs/hybrid-columns-2.log` in the worktree.

**Results** (official, `../map-stdp-hc2/reports/hybrid-columns-2/`; 20 seeds per arm, finished 15:36, 0 failed). Primary: AUC over the 500 episodes; HL shift vs base [95% CI], Holm-adjusted p:

| Contrast | Shift [CI] | p (Holm) | Pre-registered call |
| --- | --- | --- | --- |
| `cols_lam0` vs base | −35.2 [−59.8, −12.3] | 0.014 | **worse** |
| `cols_on` vs base | −16.6 [−43.1, +4.6] | 0.13 | inconclusive (CI below −Δ = −25) |
| `cols_sealed` vs base | −31.0 [−56.9, −9.8] | 0.014 | **worse** |
| `cols_on` vs `cols_lam0` | +19.7 [−10.4, +48.3] | 0.47 | null |
| `cols_on` vs `cols_sealed` | +15.1 [−12.3, +42.9] | 0.47 | null |
| `cols_lam0` vs `cols_sealed` (P-seq) | −3.1 [−31.3, +26.8] | 0.80 | null: level |

- **Secondary:**
  - Last-100 return: base −141.5 ± 17.1; `cols_lam0` −171 ± 78, `cols_on` −159 ± 32, `cols_sealed` −166 ± 81.
  - Solved (trailing-100 ≥ −100): 0/20 in every arm.
  - Collapsed runs: 1 (`cols_lam0`), 0, 0. Runs with AUC < −300: 0 base; 2, 1, 3 in the column arms.
- **Matching held:**
  - α_a factor f median 0.79 (cols), 0.73 (sealed);
  - initial entropy within 0.02 nats of base;
  - initial rate 40.0–43.1 Hz.
- **Reading (softened after the expert review, §13):**
  - On Acrobot, implicit and enforced columns cost performance even at matched starts. The study stands as pre-registered.
  - But the contrast is confounded: the Acrobot preset paired an E-only actor with the entropy bonus, which drives a silencing drift in every arm, and the loop roughly doubles its speed. The cost shows up *through* the loop; whether the loop costs anything without the drift is untested (§13).
  - P-seq is level, so cross-column links neither help nor hurt here; but all arms share the drift, and `cols_on` converges to sealed, so P-seq is not a clean test of sequencing.
  - The map term does not significantly beat either `cols_lam0` or `cols_sealed`. In the sealed arm, and in `cols_on` once sealed, R gets no task credit and almost no map signal (015 §8): these arms test fixed random recurrence, not learned column structure.

## 13. Expert review and the entropy-bonus confound (2026-10-28)

The user asked (after §12): is it *how* columns and the map term are used, rather than whether? A divergent-then-convergent review by the neuroscientist and snn-expert agents (about 29 and 27 strategies, then a shortlist). Reports in the session; the snn-expert's scratch analyses are development results.

**Finding (snn-expert, preliminary): the Acrobot actor-rate decline is the entropy bonus, not the columns.**
- CK's entropy update on an E-only actor always lowers the total actor drive: with g the entropy coefficient (Eq. 25), Σ_k g_k r_k = −α Var(r) ≤ 0 (checked on 1e5 random draws). The bonus buys entropy by silencing the actor; the logits α_a × rate shrink and the policy flattens late (§12 logs).
- The pre-set rule (§4) paired CartPole's E-only actor (`n_ai` 0) with LunarLander's c_e. The authors never ran that pairing: `config_ll.ini` pairs c_e with opponent I posts (`n_ai` = `n_ae`).
- Evidence: pilot v2 (c_e 0) kept the actor rate (0.036–0.059 per ms) through 500 episodes in all 12 runs; every c_e run declines four- to five-fold. Instrumented replays of study seeds (bit-identical to the study) split the update exactly: the entropy part drives the drift in every arm, columns add about 1.3–1.5× at matched rate, times the loop gain of about 1.3, giving the 2× faster decline seen in the study logs. Base with `n_ai` 20 holds its E rate within 8% to episode 300 (−72% without); LunarLander, which has I posts, stays flat over 3000 episodes.

**Convergent shortlist (both reviewers, ranked):**
1. **Gain control of the actor in every arm** (opponent I posts, the authors' pairing; or normalised / log-rate logits; or inhibitory STDP to 40 Hz). Levels the field; makes columns non-inferior, not better. Linear E-only columns read by a linear softmax can at best match base.
2. **Columns as the readout:** competition through inhibition (a fixed shared inhibitory pool, or per-column I posts whose cross-column inhibition the map term learns) with a race readout replacing the softmax. Gives the map term a function: keeping cross-column excitation low so the columns compete. Expected win: energy at matched return (neuroscientist), 0 to +10% AUC speculative (snn-expert). Risk: re-sharpening; keep 1 and the entropy bonus. Precedent: Frémaux, Sprekeler & Gerstner 2013.
3. **Latent state modules with latent→action exempt from the map term** (routing, as C→A is): the research question proper. 015's only positive structural hint: `lat_on` vs `cols_on` on CartPole +28 AUC (Holm 0.23). First measure the policy-class ceiling of the linear actor on Acrobot.
4. Conditions where modularity is known to pay off (observation noise, non-stationarity); and a loop-lag diagnostic (τ_loop 23–28 ms vs the 20 ms resample).

**Claims softened:** §11 and §12 readings above; 015 §8 (early speed-up; sealed R is a fixed amplifier); v4's benefit was a different mechanism (sealing fixed inter-module leakage that corrupted the readout; the CK softmax has no such leakage). "Cortical columns" overstates 20-neuron E-only pools; "action assemblies" is nearer.

**User decisions (2026-10-28):** fix the preset pairing (opponent I posts); pilot strategies 1, 2 and 3; apply the doc edits.

**Preset fix:** `_NEW` (Acrobot, Taxi) now has `n_ai` 20 = `n_ae`, the authors' pairing with c_e. This changes a pre-set rule after the study, for a mechanistic reason independent of return; recorded here, and the `hybrid-columns-2` study is reported as run. Its rerun under the fixed preset is a new study.

**Pilot (`experiments/results/pilot016/ai/`, logs `logs/pilot016/ai/`):** the 4 arms of `hybrid-columns-2` under the fixed preset, dev seeds 0–9, both matchings. Gates (stability only): actor E rate at episodes 250 and 500 within 20–60 Hz in ≥ 9/10 runs per arm; H(s) at episode 50 ≥ 0.1; max Re λ ≤ 0.5; no NaN; base not at floor. Falsifier of the drift explanation: the E rate holds within ±25% and the column arms still lose more than Δ = 25 AUC to base.

## 14. Design draft: columns as the readout, and latent state modules (2026-10-28; Proposal, revised after the neuroscientist's review)

Both follow the derivation's own table for the hybrid (§8.1): decision units read column spikes and send nothing back; inhibitory synapses across modules strengthen under the map term (lateral inhibition), within a module they weaken or are exempt (§7); routing is exempt. New features go behind flags whose default path draws nothing extra.

### 14.1 Strategy 2: competitive columns with a race readout

Staged, so that each stage has one new mechanism:

**Stage A, a fixed shared inhibitory pool (`--pool`).** As the QIF pools of §5.4: outside W, outside R and outside the walk.
- n_I pool posts, driven by every actor E column uniformly (E→I kick χ_EI / n_E per spike), and inhibiting every actor E post uniformly (I→E kick −χ_IE / n_I). No plasticity on the pool. Noisy LIF like the columns. n_I = `n_ae` and χ_EI = χ_rec are conventions, not principles, fixed before any run.
- The pool is **subtractive** common-mode cancellation (divisive only through rectification). A single shared pool is the precedented first model (pooled inhibition as the normaliser: Carandini & Heeger 2012; striatal fast-spiking interneurons: Koós & Tepper 1999); per-column pools are less precedented, so they come later if at all.
- **Pre-set rule for χ_IE:** bisection so that `probe_rate` with R and the pool equals the base run's rate at the base run's s_in. The pool then cancels the common-mode gain of the loop (1 / (1 − λ_E + λ_I) ≈ 1), and only the difference modes keep the loop's gain 1 / (1 − λ_E). This replaces `--rate-match-rec`, which lowered s_in and so the signal with it. χ_EI is fixed at χ_rec.
- `--ent-match-rec` then corrects only the difference-mode gain. The calibration holds at init only: R's loop gain grows while the pool is fixed, so the common-mode gain is logged per frame.
- **Readout unchanged at this stage** (softmax over column rates), so Stage A isolates one question: does cancelling the common mode with inhibition, rather than with a lower s_in, remove the column deficit? Stage A tests normalisation, not Map-STDP.
- **Structure monitors:** the pool anticorrelates the columns, which the pair counts record as below-chance cross-column transitions (§2.1 of the derivation), so M* and `within` rise with the pool even at fixed R. Compare arms at matched pool, never pool vs no-pool on structure.
- **Where the map term fits:** it keeps cross-column E low so that difference modes exist; a hub (015 §8, LunarLander) folds them into the common mode. Prediction: with the pool on, `cols_lam0` < `cols_on` ≈ `cols_sealed` on LunarLander.
- **Monitor:** the full signed offspring table (2K × 2K with the pool; derivation (12)), max Re λ ≤ 0.5; no column silent (> 1 Hz) in ≥ 95% of frames.

**Stage B, the race readout (`--readout race`).** The decision units of §8.1: the action is the column whose spike count first reaches a bound θ in the decision window (the count-difference race of Eq. 1h, as `experiments/m2.py` uses). No softmax; the policy is implicit.
- **What the race counts:** each column's net count, E spikes minus its own opponent I spikes, the same quantity the softmax's logits read (`Layout.Ma`). Fixed before piloting.
- **Pre-set rules, all fixed before any return is seen:** the decision window is the freeze window (20 ms on CartPole and Acrobot, 40 ms on LunarLander); if no column reaches θ, the leader at the window's end wins; ties break uniformly (one rng draw, behind the flag); θ by bisection so that the initial action entropy equals base's (the `match_entropy` probe on the realised action distribution). `BOUND_CAL` is not used: it was measured on M2 action modules, not on CK columns at 40 Hz.
- **Learning:** the exact local score is each actor neuron's spike minus its hazard over the decision window, times δ (Frémaux, Sprekeler & Gerstner 2013; Urbanczik & Senn 2009, who also show that feeding the winner back speeds learning). The population surrogate is the derivation's matched score for (1h): A_k − P̃_θ(k), with P̃_θ the θ-sharpened rate shares (calibrated at init), used for both the actor eligibility and the entropy bonus (Eq. 25 needs p too). The per-neuron score is run as a check on one seed.
- **Co-primary metrics:** return non-inferior (Δ as before) **and** synaptic operations per decision at least 30% below base: spikes of the columns, the pool and the I posts, each weighted by its fan-out (Attwell & Laughlin 2001; Lennie 2003). Losing columns are suppressed, not silent.
- **Risk:** competition re-sharpens the policy and brings back the absorbing corner. Keep the entropy bonus and the opponent I posts.

**Stage C, learned lateral inhibition** (per-column I pools in R; map term on cross-column I→E only; within-column I→E and cross-column E→I exempt). Cross-column I→E is an abstraction: in cortex and striatum lateral inhibition runs through long-range excitation onto *local* interneurons (Adesnik et al. 2012; Packer & Yuste 2011), and the map term would prune that pathway as cross-module excitation. Only if Stage A shows inhibition matters (criterion: the Stage A pool removes at least half of the column deficit). Needs plastic inhibition in `HybridRec` and the 2K × 2K monitor each frame. The design is in the subcritical regime (χG ≤ 0.5); the inhibition-stabilised conditions of derivation §7 apply only if χ_rec is raised above the excitation-only bound, and would be checked by the paradoxical test (pool drive up, pool rate down).

### 14.2 Strategy 3: latent state modules with exempt routing (`--rec lat --lat-exempt`)

- Latent modules as in 015 (`rec lat`): n_lat_mod modules of n_ae posts between the input and the action columns, input→L frozen at init, L→A and L↔L in R.
- **Change:** latent→action is routing. Set g = 0 on the (A ← L) entries of the modulator table in `HybridRec.frame`, the same exemption as C→A (derivation §4.5), in place of the latent-output barrier `la_beta`, which did not hold (015 §6: L→A pruned 0.25 → 0.004). The map term then only seals the latent modules from each other.
- **Arms:** base; and L→A learned by TD × {map term on L↔L, off}, with λ = 0 and L→A exempt to attribute any residual pruning of L→A to the task term (§4.5 says the task term prunes it too on fully observed tasks). The random-expansion arm keeps L→A learning (otherwise it has no readout).
- **What to expect on CartPole, LunarLander and Acrobot:** learned L ≈ random L is *expected* on a fully observed task. "Communities as states" predicts something beyond a random expansion only under noise or partial observability (slow, metastable module dynamics: Litwin-Kumar & Doiron 2012; Mazzucato et al. 2015). So the test is (i) structural: the module chain T^K(o) of learned L predicts the next observation better than random L's; (ii) functional, on a partially observed or noisy-observation arm (M4's T-maze, or Gaussian observation noise fixed in advance): learned L > random L.
- **Gates:** `lat_out` ≥ 0.1 at the end, latent rates > 1 Hz, max Re λ ≤ 0.5.
- **Before the pilot:** the policy-class ceiling of the linear CK actor on Acrobot (an analogue of `experiments/ceiling.py`), since 0/20 runs solved it. If the linear actor is at its ceiling, latents are the only arm that can raise it.

### 14.3 Order and cost

1. §13 pilot (preset fix): running.
2. Stage A on Acrobot (4 arms × dev seeds 0–9, both with and without `--pool`): about 10 minutes.
3. Stage B on CartPole and Acrobot (base-race, cols-race): about 15 minutes.
4. Strategy 3: the Acrobot ceiling first; then the structural test on LunarLander (dev seeds 10–15, 3000 episodes, about 1 hour); the functional test needs a partially observed arm, outside the three-task scope.

**Precedent for and against (neuroscientist):** action competition in vertebrates is implemented by convergent, mostly fixed inhibition (basal-ganglia focused selection: Mink 1996; Redgrave, Prescott & Gurney 1999; Gurney, Prescott & Redgrave 2001; Bogacz & Gurney 2007), so Stages A and B have strong precedent, and Stage C's learned long-range I→E is the weak link. "Columns" remains an overstatement for 20-neuron E pools; "action channels" is nearer. Proposed one-sentence caveat under the derivation's §8.1 table (not yet applied): the map term's optimum for inhibition, lateral I→E, is an abstraction of lateral inhibition that in cortex runs through long-range excitation onto local interneurons.

**§13 pilot result (fixed preset, `pilot016/ai/`, dev seeds 0–9, both matchings; development):**

| Arm | Collapsed | AUC < −300 | AUC median | Actor rate ep 50 / 250 / 500 | H(s) ep 50 / 250 / 500 |
| --- | --- | --- | --- | --- | --- |
| base, E-only (§11 pilot) | 0 | 0 | −163 | 0.038 / 0.022 / 0.010 | 0.52 / 0.42 / 0.61 |
| base, I posts | 3 | 4 | −264 | 0.037 / 0.030 / 0.029 | 0.32 / 0.08 / 0.13 |
| `cols_lam0`, I posts | 3 | 5 | −369 | 0.037 / 0.028 / 0.025 | 0.23 / 0.08 / 0.24 |
| `cols_on`, I posts | 7 | 8 | −499 | 0.037 / 0.031 / 0.031 | 0.15 / 0.00 / 0.00 |
| `cols_sealed`, I posts | 4 | 5 | −365 | 0.037 / 0.026 / 0.024 | 0.24 / 0.10 / 0.12 |

- **The drift is gone:** the actor rate holds within 25% of its start in every arm (the predicted effect of the I posts).
- **But base is now at floor** (3/10 collapsed, median AUC −264), and the column arms worse. The policy sharpens early instead: entropy 0.32 at episode 50 and 0.08 by 250 in base, against 0.52 and 0.42 without I posts. This is the absorbing corner of §10 again, reached through the gain rather than through the loop.
- **Gates:** rate gate passed; the H(s) ≥ 0.1 gate at episode 50 passed in base and `cols_lam0`/`cols_sealed` medians, failed for `cols_on`; max Re λ 0.28; no NaN; **base at floor: failed.** Per the pre-set rule, reported to the user before any change.
- **Likely cause (untested):** the opponent pairing doubles the logit's learning gain. The logit is α_a (r_E − r_I), and the task update moves the E and I rows in opposite directions, so each update changes the logit about twice as much as on an E-only actor, at the same α_a. The authors' I-post configs use α_a 15 (LunarLander), not CartPole's 50 that the Acrobot preset inherits. So `n_ai` 20 with α_a 50 is another combination the authors never ran.
- **Init check (seed 0, base, after calibration):** with I posts the initial policy entropy is 0.874 vs 0.969 E-only, at the same α_a (175 after k). The E − I difference of two independent noisy populations has about √2 the spread of E alone, so the opponent readout is sharper from the start, and its logit learning gain is about doubled. The I posts remove the drift and add a gain. Reported to the user; no change made.

**User decision (2026-10-28): gain control by divisive normalisation of the action scores** (`--norm-logits`; Proposal, neuroscientist check before implementation). In every arm, base included:

- **Scores:** score_k = α_a · r_k · r₀ / max(r̄, ε), with r_k column k's rate as now (`Layout.Ma`), r̄ the pooled rate over the actor posts (the pilot's presets are E-only, so r̄ is the mean actor E rate; with I posts the normaliser is the unsigned pooled rate and the direct term stays signed), r₀ = `rate_target` (the pre-set 40 Hz). No new free parameter. At init r̄ ≈ r₀, so the policy is the backbone's; a common change of all rates leaves it unchanged, so the entropy bonus can no longer flatten it by silencing the actor, and the loop's common-mode gain no longer matters for the policy (it still matters for stability, the critic and the pair counts; and the normalisation removes the common-mode confound, not §10's contrast-gain hypothesis, for which `--ent-match-rec` remains the control).
- **Floor (neuroscientist's rule, no free number):** ε = α_a r₀ / (τ_n n_ae), the pooled rate at which one spike (which raises a column's filtered rate by 1 / (τ_n n_ae)) moves a score by one nat; with the pre-set α_a (175 after k), r₀, τ_n 20 ms and n_ae 20 this is 0.0175 per ms (17.5 Hz). Above the floor the readout is exactly scale-free; below it the normaliser is the constant ε (gain capped at r₀ / ε ≈ 2.3, correction term u = 0). Score noise scales as r̄^(−1/2), so without a floor a 4× rate drop would double the policy noise.
- **Learning, the exact gradient through the normalisation:** ∂ log p_a / ∂ r_j = (α_a r₀ / r̄) · [(A_j − p_j) − (r_a − Σ_k p_k r_k) / (K r̄)]. Column j's eligibility factor is therefore (r₀ / r̄) · [(A_j − p_j) − u], with u = (r_a − Σ_k p_k r_k) / (K r̄) one scalar shared by all columns. Both factors are broadcast scalars (crossbar-native). The entropy bonus (Eq. 25) chains through the same factor: κ_j ∝ (r₀ / r̄) · [g_j − (1 / (K r̄)) Σ_k g_k r_k]; the second term is what cancels the silencing drift of §13 (Σ_k g_k r_k ≤ 0 is no longer a drift on the total drive, because p is invariant to the common mode).
- **The factor r₀ / r̄** keeps the step on the score scale matched as rates change (the gain compensation that `--ent-match-rec` could only fix at init).
- **Precedent:** divisive normalisation by pooled activity as a canonical cortical computation (Carandini & Heeger 2012); the pooled rate is the quantity a shared inhibitory pool would carry (Stage A then implements it in neurons).
- **Alternative considered:** the Luce readout p_k ∝ r_k^β (per-column factor 1/r_k on the eligibility). Also local and drift-free, but it changes the init policy and needs β calibrated. Rejected for the extra parameter.
- **Pilot:** Acrobot, the 4 arms, dev seeds 0–9, E-only presets (`n_ai` back to 0: the opponent pairing is withdrawn, §13 pilot), both matchings, `--norm-logits`. Gates as §13; the falsifier as before.
- **Forking paths, stated:** this is the third gain-control choice (E-only drift → opponent I posts → normalisation) made after seeing Acrobot returns on dev seeds 0–9. The pilot is a stability pilot only; its AUCs are not evidence on columns. The study under `--norm-logits` uses fresh seeds, its README records the three-step path as a post-hoc readout change, `base --norm-logits` is its reference, and `hybrid-columns-2`'s base remains the CK-faithful reference.
- **Neuroscientist check (2026-10-28):** gradient and entropy chain confirmed (log p_a and H are homogeneous of degree 0 in r, so the Euler identity gives the zero drift); locality holds as two broadcast scalars (an abstraction: a shared pool carries r̄, Stage A; u is a decision-circuit quantity); pooled-over-options normalisation of action value is observed in LIP (Louie, Grattan & Glimcher 2011; Louie, Khaw & Glimcher 2013); the critic's rate as normaliser and the Luce ratio were rejected. The only new risk: the backbone's accidental late re-flattening (E-only base H 0.42 → 0.61 was the drift) is gone, so base may sharpen monotonically; the entropy bonus then acts on contrast only, the intended regime.
- **Numerical check (scratch, K = 4, 200 random rate vectors):** the gradient above matches central finite differences of log p_a and of H to a relative error of 1e-9; Σ_j r_j ∂H/∂r_j = 0 exactly with the normalisation (−0.35 for the unnormalised scores at the same rates: the drift).

**`--norm-logits` pilot result (`pilot016/nl/`, Acrobot, dev seeds 0–9, both matchings, E-only presets; development, stability pilot):**

| Arm | Collapsed | AUC < −300 | AUC median | HL vs its base | MW p | Actor rate ep 50 / 250 / 500 | H(s) ep 50 / 250 / 500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base, E-only, unnormalised (§11) | 0 | 0 | −163 | – | – | 0.038 / 0.022 / 0.010 | 0.52 / 0.42 / 0.61 |
| `cols_lam0`, unnormalised (§11) | 0 | 2 | −200 | −32 | 0.045 | 0.038 / 0.014 / 0.007 | 0.43 / 0.53 / 0.75 |
| base, normalised | 1 | 1 | −186 | – | – | 0.048 / 0.047 / 0.041 | 0.71 / 0.34 / 0.27 |
| `cols_lam0`, normalised | 0 | 0 | −176 | +13 | 0.39 | 0.051 / 0.050 / 0.048 | 0.61 / 0.32 / 0.27 |
| `cols_on`, normalised | 0 | 0 | −180 | +4 | 0.97 | 0.052 / 0.052 / 0.049 | 0.61 / 0.24 / 0.26 |
| `cols_sealed`, normalised | 0 | 0 | −199 | −2 | 0.97 | 0.052 / 0.047 / 0.044 | 0.69 / 0.31 / 0.25 |

- **Gates passed:** actor rate within 20–60 Hz at episodes 250 and 500 in 10/10 base runs (and every column run); H(s) ≥ 0.1 at episode 50 in all; max Re λ 0.28 (columns), no NaN; base not at floor (median −186, 1/10 collapsed).
- **The falsifier did not fire:** the rate holds and the column arms are level with base (HL +13, +4, −2; all p ≥ 0.39), with no collapsed or slow column run (0/30). The Acrobot deficit of §12 was the silencing drift acting through the loop, not a cost of the loop itself.
- **Base is lower than the unnormalised E-only base** (−186 vs −163, HL −22): as the neuroscientist predicted, the backbone's accidental late re-flattening (H 0.42 → 0.61, which was the drift) is gone, so the policy sharpens monotonically (H 0.27 at episode 500 in every arm). The entropy bonus now acts on contrast only. The comparison to make is columns vs base under the same readout; `hybrid-columns-2`'s base remains the CK-faithful reference.
- `--ent-match-rec` factors are now 0.95–1.05: with the normalisation the arms' initial policies already match.
- **Actor rates rise slightly above the 40 Hz target** (0.048–0.052 per ms by episode 50) and stay there: the task update's common mode is no longer corrected downward by the bonus. Within the rate gate.

## 15. Sequence and study `hybrid-columns-3` (2026-10-28)

**User decisions:** structure has to matter before the map term can be used strategically. Run sequentially: (1) the three-task study under the normalised scores, the backbone result set; (2) Stage B, the race readout (§14.1), where cross-column links act at the decision; (3) a stimulus-conditioned map term, informed by the study's results.

**Why the conditioned map term waits (chat, 2026-10-28):** a K×K modulator indexed by a stimulus class (the active action column, or a latent state module) is local, crossbar-native and cheap (S × K × K); the implementation's `obs_tau` averaging currently washes the stimulus out of the walk statistics. But every backbone result so far shows the softmax readout cannot see R's structure (hub ≈ modular ≈ sealed at matched gain; P-seq null): a link between columns acts within one loop time (about 25 ms), far below the task's sequencing timescale, and the window average removes it. So a conditioned map term now would shape structure no one reads. Candidates that make structure matter: the race readout, slow (NMDA-like) recurrent synapses, latent state modules; then the conditioned map term is a learned, state-dependent chunking of actions. Two further strategies noted: a map term that defers to the task term (a K×K exemption table from the task credit), and energy as the metric where sealing already pays.

**Study `hybrid-columns-3`** (declared in `experiments/report.py`; README draft `reports/hybrid-columns-3/README.md`): CartPole, LunarLander, Acrobot × base / `cols_lam0` / `cols_on` / `cols_sealed`, seeds 5000–5019, flags `--rate-target 40 --block 20 --fast --eta-cal --rate-match-rec --ent-match-rec --norm-logits`. Primary: LunarLander last-100, AUC elsewhere; Δ = 25; Holm per task. `--eta-cal` on every task (015 used CK's preset η on CartPole). **Stability pilots** before pre-registration (`pilot016/nl_cp/`: CartPole 4 arms × dev seeds 0–4; `pilot016/nl_ll/`: LunarLander 4 arms × dev seeds 10–12, 1000 episodes). Gates as §13.

**Stability pilots (development):**
- **LunarLander** (`pilot016/nl_ll/`, dev seeds 10–12, 1000 episodes): rates hold (0.040–0.045 per ms at episodes 50, 500 and 1000 in every arm), entropy 1.26 → 0.95 (base) and the same in the column arms, max Re λ ≤ 0.26, no NaN; AUC over the 1000 episodes level with base (`cols_lam0` −1, `cols_on` −17, `cols_sealed` −7; n = 3). Gates passed.
- **CartPole, first attempt** (`--eta-cal` on every task): base at the random-policy level (AUC 11.7, 0/5 solved). Not the normalisation: CartPole's preset η (0.005, 0.01) are CK's SGD learning rates, not Adam steps, so `--eta-cal` divides them by the update RMS and gives η_a 8.7 mV (LunarLander's 0.07); |W| grew to 7 (015: 0.06) and the actor fell silent. That is why 015 kept CK's preset η on CartPole. The policy at init under the normalisation is the backbone's (traced on seed 0). **Fix:** CartPole without `--eta-cal`, as in `hybrid-columns`; declaration, test and README updated; pilot rerun.
- **CartPole, second attempt** (CK's preset η): base solved 5/5 (T_s 221–269; AUC 354; rates 0.041 → 0.004 by episode 400 as the solved policy settles, as in 015). The column arms were dead (AUC 36, entropy ln 2 throughout): `match_entropy` had driven α_a to its floor (f = 0.125). **Cause, a bug in `probe_entropy`:** it averaged the policy entropy over the live copies per tick; CartPole's single copy spends many calibration ticks in reset (random-policy episodes last about 20 ticks), the empty mean is NaN, every NaN comparison in the bisection reads as "entropy too high", and f walks to the floor. Acrobot and LunarLander episodes are long enough never to hit it (their factors were 0.7–1.05). **Fix:** the probe pools the entropy over the live copies of all ticks (identical for one copy without resets; a count-weighted mean otherwise). Test added. Pilot rerun.
- **CartPole, third attempt** (preset η, fixed probe; `pilot016/nl_cp/`, dev seeds 0–4): every arm solved 5/5; f 0.95–0.99 with the entropies matched (0.59 = 0.59); rates 0.041–0.051 per ms; max Re λ 0.28; no NaN. AUC vs base: `cols_lam0` −16 (p 0.42), `cols_on` −1 (p 0.84), `cols_sealed` −43 (p 0.03; T_s 256–356 vs base 221–269). Gates passed. README expectations filled; the study awaits the user's approval, Δ confirmation and commit.

**Monitor added before launch (user decision, 2026-10-28): the stimulus-conditioned walk.** `HybridRec.window` now takes the copies' executed action as the class and splits the window's pair counts by it (`kc`, K × nR × nR; per head the counts are independent integers, so the split sums to the single call exactly, and `a["kappa"]` is unchanged). At each `SNAP_EVERY` snapshot and at the end, `class_table()` reduces it to the module level (K × Kr × Kr shares per class and the class's pair count) and resets; the result carries it as `rec.RKc` next to `RK`, and checkpoints carry `rec_kc` and `snaps_kc`. It changes no update and draws nothing: a measurement of whether the cross-column walk depends on the state, which the conditioned map term would need. Test `test_class_pair_counts`.
