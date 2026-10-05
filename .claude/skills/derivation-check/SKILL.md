---
name: derivation-check
description: Numerically verify a math change to docs/derivation.md before writing it. Finite-difference checks of gradient claims and Monte Carlo checks that a three-factor rule's average matches -W ⊙ ∇D, as CLAUDE.md requires. Use whenever adding or changing a gradient, modulator, objective, eligibility rule or theorem in the derivation.
---

# Derivation check

CLAUDE.md requires this before any change to a derivation:

1. **Gradient claims:** compare against finite differences on a small random network.
2. **New modulators:** check by Monte Carlo that the three-factor average matches $-W \odot \nabla D$.

## Tool

`scripts/fdcheck.py` (numpy). Run it with the project env's python (`conda activate map-stdp`). First confirm that the reference still reproduces:

```bash
python .claude/skills/derivation-check/scripts/fdcheck.py --selftest
```

You should get a map gradient error of at most `1e-8` and a Monte Carlo correlation of at least `0.999`.

Building blocks, all using the derivation's conventions (`W[i, j]` is pre `j` → post `i`):

| Function | Use |
| --- | --- |
| `make_net(sizes, alpha, seed)` | Random modular network. Module 0 is the controller. |
| `stimulus(net)`, `flow(W, v, alpha)` | Per-frame flow $\pi_f$, exact resolvent. |
| `map_L`, `map_modulator` | Map equation, and its marginal cost $g$ (Eq. 6). |
| `lemma_grad(W, pi, g)` | Eq. 5: $\frac{\pi_j}{d_j}(g_{ij} - \bar{g}_j)$ for *any* modulator $g$. |
| `fd(f, W)`, `check_gradient(D, grad, W)` | Finite differences. Returns the maximum absolute error. |
| `check_three_factor(W, pis, g_fn, target)` | Monte Carlo of the rule, with pairings sampled at rate $J$. Returns the correlation with `target`. |

## Procedure

1. **Write a scratch script** in the session scratchpad, not the repo. It imports `fdcheck`, defines the new quantity $D$ (as a function of `W` with `pi` fixed, per the EM split) and its claimed gradient or modulator.
2. **Check the gradient:** `check_gradient(D, grad, W)` for several stimuli. Exact claims should give at most 1e-8.
3. **Check the three-factor average.** For a new modulator, run `check_three_factor` against `-W * grad` of the frame-averaged $D$. Expect a correlation of at least 0.999 for exact rules. Report the actual value for surrogates.
4. **Check against the true quantity.** If the claim involves the full flow, for example through the resolvent, also compare with finite differences that recompute `flow`, and report the cosine.
5. **Report the numbers.** Put them in the derivation's **Appendix A** under a bold "Numerical checks for …" heading: network size, $\alpha$, the error or correlation, and the number of frames. Then mention the result in the CHANGELOG entry.

If a check fails, do not edit the derivation. Report the discrepancy instead.
