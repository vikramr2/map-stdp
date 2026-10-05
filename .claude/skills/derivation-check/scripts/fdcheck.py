#!/usr/bin/env python3
"""Numerical checks required by CLAUDE.md before changing docs/derivation.md.

Import it (``sys.path.insert(0, <this dir>)``; ``import fdcheck as fc``) or run
``python fdcheck.py --selftest`` to reproduce the reference numbers.

Conventions follow the derivation: W[i, j] is the synapse pre j -> post i,
T = W / d with d_j = sum_i W[i, j], J = pi_j T[i, j], modules m[i] in 0..K-1.
"""
import argparse
import numpy as np


def make_net(sizes=(4, 3, 3, 3), alpha=0.2, seed=0, within=0.6):
    """Random modular network. Module 0 is the controller; the rest are latent/action."""
    rng = np.random.default_rng(seed)
    m = np.repeat(np.arange(len(sizes)), sizes)
    N = len(m)
    W = rng.uniform(0.05, 0.4, (N, N))
    W[m[:, None] == m[None, :]] += within
    np.fill_diagonal(W, 0)
    return dict(W=W, m=m, K=len(sizes), N=N, alpha=alpha, rng=rng)


def T_of(W):
    return W / W.sum(0, keepdims=True)


def flow(W, v, alpha):
    """Exact per-frame flow pi = alpha (I - (1 - alpha) T)^-1 v."""
    return alpha * np.linalg.solve(np.eye(len(v)) - (1 - alpha) * T_of(W), v)


def stimulus(net, module=0, conc=0.5):
    """Random teleportation vector concentrated on one module (the controller by default)."""
    v = np.zeros(net["N"])
    idx = net["m"] == module
    v[idx] = net["rng"].dirichlet(np.ones(idx.sum()) * conc)
    return v


def module_sums(x, m, K):
    return np.array([x[m == k].sum() for k in range(K)])


def map_L(W, pi, m, K):
    """Map equation L(M) in nats, with pi held fixed (EM split)."""
    T = T_of(W)
    ebar = (T * (m[:, None] != m[None, :])).sum(0)
    q, p = module_sums(pi * ebar, m, K), module_sums(pi, m, K)
    xlx = lambda x: x * np.log(x)
    return xlx(q.sum()) - 2 * xlx(q).sum() - (pi * np.log(pi)).sum() + xlx(p + q).sum()


def map_modulator(W, pi, m, K):
    """g_ij = M*_{m(j)} I[i not in m(j)]: the map equation's marginal cost (Eq. 6)."""
    T = T_of(W)
    cross = (m[:, None] != m[None, :]).astype(float)
    q = module_sums(pi * (T * cross).sum(0), m, K)
    p = module_sums(pi, m, K)
    Ms = np.log(q.sum() * (p + q) / q**2)
    return Ms[m][None, :] * cross


def lemma_grad(W, pi, g):
    """Eq. 5: dD/dW_ij = (pi_j / d_j) (g_ij - gbar_j), gbar_j = sum_k T_kj g_kj."""
    T, d = T_of(W), W.sum(0)
    return (pi / d)[None, :] * (g - (T * g).sum(0)[None, :])


def fd(f, W, h=1e-6):
    """Central finite differences of scalar f(W) over off-diagonal entries."""
    g = np.zeros_like(W)
    for i, j in zip(*np.nonzero(~np.eye(len(W), dtype=bool))):
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += h
        Wm[i, j] -= h
        g[i, j] = (f(Wp) - f(Wm)) / (2 * h)
    return g


def off(W):
    return ~np.eye(len(W), dtype=bool)


def check_gradient(D, grad, W):
    """Max |analytic - finite-difference| over off-diagonal entries. D: W -> float."""
    return float(np.abs(grad - fd(D, W))[off(W)].max())


def check_three_factor(W, pis, g_fn, target, frames=4000, steps=200, rng=None):
    """Monte Carlo of the three-factor rule averaged over frames.

    pis: per-frame flows; g_fn(W, pi) -> modulator matrix g_ij; target: matrix to compare
    with (usually -W * grad of the frame-averaged D). Pairings are sampled at rate J_ij.
    Returns the correlation between the average update and the target.
    """
    rng = rng or np.random.default_rng(1)
    acc = np.zeros_like(W)
    T = T_of(W)
    for _ in range(frames):
        pi = pis[rng.integers(len(pis))]
        J = pi[None, :] * T
        g = g_fn(W, pi)
        cnt = rng.multinomial(steps, (J / J.sum()).ravel()).reshape(W.shape) * (J.sum() / steps)
        acc += -(g - (T * g).sum(0)[None, :]) * cnt
    acc /= frames
    o = off(W)
    return float(np.corrcoef(acc[o], target[o])[0, 1])


def selftest():
    net = make_net()
    W, m, K, a = net["W"], net["m"], net["K"], net["alpha"]
    pis = [flow(W, stimulus(net), a) for _ in range(6)]
    errs = [check_gradient(lambda X, pi=pi: map_L(X, pi, m, K), lemma_grad(W, pi, map_modulator(W, pi, m, K)), W)
            for pi in pis]
    gbar = np.mean([lemma_grad(W, pi, map_modulator(W, pi, m, K)) for pi in pis], 0)
    corr = check_three_factor(W, pis, lambda X, pi: map_modulator(X, pi, m, K), -W * gbar)
    print(f"map-equation gradient vs FD, max error over 6 frames: {max(errs):.1e}   (expect <= 1e-8)")
    print(f"three-factor Monte Carlo vs -W*grad D (frame-averaged): corr {corr:.5f}   (expect >= 0.999)")
    ok = max(errs) <= 1e-8 and corr >= 0.999
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(0 if selftest() else 1)
    ap.print_help()
