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

- **Confirmed:** at the same mean rate the column policies start sharper. The within-column loop amplifies rate *differences* between columns (logit spread +25 to +37% on Acrobot), and rate matching does not change that.
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
