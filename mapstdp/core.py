"""Map-STDP numpy flow-level reference.

Conventions (docs/derivation.md): W[i, j] is the synapse pre j -> post i, T = W / d with
d_j = sum_i W[i, j], J[i, j] = pi_j T[i, j], module tables are indexed [b, a] = (post module, pre module).
Logs are base 2 (bits). Task term: action-gated eligibility (Eq. 4b) here; TD critic in experiments/m2.py.
"""
import numpy as np


def build_network(sizes=(16, 16, 16, 16, 16), roles="CLLAA", p_same=0.5, p_cc=0.1,
                  p_route=0.3, p_cross=0.05, seed=0):
    """SBM: dense within modules, sparse controller recurrence, controller->L/A routing links.
    Returns W (columns normalised to d_j = 1), module labels m, role per module, synapse mask."""
    rng = np.random.default_rng(seed)
    m = np.repeat(np.arange(len(sizes)), sizes)
    roles = np.array(list(roles))
    P = np.full((len(sizes),) * 2, p_cross)
    np.fill_diagonal(P, p_same)
    P[route_pairs(roles)] = p_route
    c = int(np.flatnonzero(roles == "C")[0])
    P[c, c] = p_cc
    mask = rng.random((len(m),) * 2) < P[m][:, m]
    np.fill_diagonal(mask, False)
    assert mask.any(0).all(), "neuron without outgoing synapse"
    W = mask * rng.uniform(0.5, 1.5, mask.shape)
    return W / W.sum(0), m, roles, mask


def route_pairs(roles):
    """K x K bool table [b, a]: a is the controller and b is a latent or action module (Eq. 4a)."""
    return (roles == "C")[None, :] & np.isin(roles, ["A", "L"])[:, None]


def stimulus(o, m, roles, width=0.1):
    """v(o): Gaussian receptive fields on a ring over the controller neurons, o in [0, 1)."""
    ctrl = np.flatnonzero(roles[m] == "C")
    dist = np.abs((np.arange(len(ctrl)) / len(ctrl) - o + 0.5) % 1 - 0.5)
    v = np.zeros(len(m))
    v[ctrl] = np.exp(-dist ** 2 / (2 * width ** 2))
    return v / v.sum()


def cartpole_stimulus(o, m, roles, n=6, sigma=0.4, scale=(0.21, 3.5)):
    """Conjunctive Gaussian RFs on an n x n grid over (theta / 0.21, theta_dot / 3.5) clipped to [-1, 1];
    one controller neuron per cell (needs n^2 controller neurons), v sums to 1."""
    x = np.clip(np.asarray(o, float)[2:] / scale, -1, 1)
    g = np.linspace(-1, 1, n)
    c = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    b = np.exp(-((c - x) ** 2).sum(1) / (2 * sigma ** 2))
    v = np.zeros(len(m))
    v[roles[m] == "C"] = b / b.sum()
    return v


def flow(W, v, alpha=0.2, n=None):
    """Per-frame flow (Eq. 1). n=None: exact solve; else n hops of power iteration from v (Eq. 1a)."""
    T = W / W.sum(0)
    if n is None:
        return alpha * np.linalg.solve(np.eye(len(v)) - (1 - alpha) * T, v)
    pi = v.copy()  # the iterates x_k of derivation §2.3 (Eq. 1a), converging to pi
    for _ in range(n):
        pi = (1 - alpha) * T @ pi + alpha * v
    return pi


def module_stats(W, pi, m, K):
    """p (visit), q (exit), JK[b, a] (module flow), TK = JK / p (Eq. 1c)."""
    H = np.eye(K)[m]
    J = W / W.sum(0) * pi
    JK = H.T @ J @ H
    p = H.T @ pi
    q = JK.sum(0) - np.diag(JK)
    with np.errstate(divide="ignore", invalid="ignore"):
        TK = JK / p
    return p, q, JK, TK


def policy(p, roles, beta=1.0):
    """Eq. 1d: action-module flow shares; beta != 1 is the sharpened policy p_a^beta / sum p^beta (Proposal, Eq. 1e)."""
    pa = p[roles == "A"] ** beta
    return pa / pa.sum()


def _plogp(x):
    """x log2 x elementwise, with 0 log 0 = 0."""
    x = np.asarray(x, float)
    return np.where(x > 0, x * np.log2(np.where(x > 0, x, 1)), 0.0)


def map_equation(p, q, pi):
    """L(M), Eq. 3, in bits."""
    return float(_plogp(q.sum()) - 2 * _plogp(q).sum() - _plogp(pi).sum() + _plogp(p + q).sum())


def mstar(p, q, clip=20.0):
    """M*_m = log2(q_tot (p_m + q_m) / q_m^2) >= 0, clipped to [0, clip] (diverges as q_m -> 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.log2(q.sum() * (p + q) / q ** 2)
    return np.clip(np.nan_to_num(M, nan=clip, posinf=clip), 0, clip)


def modulator(p, q, roles, clip=20.0, routing="none", mu=0.0):
    """K x K table g[b, a] = M*_a I[b != a] (Eq. 6). routing: 'none' | 'exempt' (g = 0 on routing
    pairs) | 'dual' (subtract mu on routing pairs, Eq. 4a). Both routing options are Proposals."""
    g = mstar(p, q, clip)[None, :] * (1 - np.eye(len(p)))
    if routing == "exempt":
        g[route_pairs(roles)] = 0.0
    elif routing == "dual":
        g[route_pairs(roles)] -= mu
    return g


def j_route(JK, roles):
    """Controller routing flow J_route = sum over b in A, L of J_{bC} (Eq. 4a)."""
    return float(JK[route_pairs(roles)].sum())


def centered(tab, W, m):
    """tab[m(i), m(j)] - baseline_j, baseline_j = sum_k T[k, j] tab[m(k), m(j)] (Eqs. 4, 6).
    tab = g gives the modulator minus its per-neuron baseline; tab = 1 - I gives G(i, j) (Eq. 7)."""
    T = W / W.sum(0)
    G = tab[m][:, m]
    return G - (T * G).sum(0)


def structural_update(W, mask, g, kappa, m, lam, eps=0.0, wmin=1e-3, wmax=1.0):
    """Eq. 4: W += -lam (g_ij - gbar_j + eps (d_j - 1)) kappa_ij, then sign mask and bounds on existing synapses.
    eps: d_j homeostasis in the baseline (derivation §2.2). kappa: expected (c * J), sampled (sample_pairings)
    or spiking (causal, balanced or covariance) pairings."""
    return clip_weights(W - lam * (centered(g, W, m) + eps * (W.sum(0) - 1)) * kappa, mask, wmin, wmax)


def action_score(W, pi, m, roles, a, beta=1.0):
    """h_b of Eq. 4b (score of the surrogate policy (J^in_a)^beta / sum_A (J^in)^beta) for chosen action module a:
    h_b = beta (I[b = a] - P_beta(b)) / J^in_b on action modules, 0 elsewhere; beta = 1 is Eq. 4b. Returned as a
    K x K table tab[b, a'] = h_b, ready for centered()."""
    K, A = len(roles), roles == "A"
    Jin = np.bincount(m, (W / W.sum(0) * pi).sum(1), K)
    P = np.zeros(K)
    P[A] = Jin[A] ** beta / (Jin[A] ** beta).sum()
    h = np.where(A, -beta * P / np.where(A, Jin, 1.0), 0.0)
    h[a] += beta / Jin[a]
    return np.repeat(h[:, None], K, 1)


def eligibility_step(e, W, kappa, pi, m, roles, a, tau_e=3.0, beta=1.0):
    """Eq. 4b: e <- (1 - 1/tau_e) e + kappa_ij (h_m(i) - hbar_j)."""
    return (1 - 1 / tau_e) * e + kappa * centered(action_score(W, pi, m, roles, a, beta), W, m)


def homeostasis_update(W, mask, kappa, eps_h, wmin=1e-3, wmax=1.0):
    """d_j homeostasis as its own term on all plasticity (model v2): W -= eps_h (d_j - 1) kappa_ij. Its expectation
    -eps_h (d_j - 1) pi_j W_ij / d_j rescales column j uniformly, so T, pi and D are unchanged.
    Abstraction: presynaptic (outgoing) conservation, not postsynaptic synaptic scaling (docs/model.md)."""
    return clip_weights(W - eps_h * (W.sum(0) - 1) * kappa, mask, wmin, wmax)


def additive_G_update(W, mask, m, roles, eta, routing="none"):
    """Exact-gradient baseline (Eq. 7 with positive factors dropped): W += -eta G(i, j) on synapses."""
    c = 1 - np.eye(len(roles))
    if routing == "exempt":
        c[route_pairs(roles)] = 0.0
    return clip_weights(W - eta * centered(c, W, m) * mask, mask)


def clip_weights(W, mask, wmin=1e-3, wmax=1.0):
    """Explicit bounds; wmin > 0 is the additive floor that keeps multiplicative updates from freezing."""
    return np.where(mask, np.clip(W, wmin, wmax), 0.0)


def grad_L(W, pi, m, K):
    """Exact dL/dW with pi fixed (Eq. 7): M*_{m(j)} pi_j / d_j G(i, j); M* unclipped."""
    p, q, _, _ = module_stats(W, pi, m, K)
    g = mstar(p, q, np.inf)[None, :] * (1 - np.eye(K))
    return centered(g, W, m) * pi / W.sum(0)


def sample_pairings(J, n, rng):
    """n causal pairings drawn with probability proportional to J_ij (mean-field assumption)."""
    return rng.multinomial(n, (J / J.sum()).ravel()).reshape(J.shape).astype(float)


if __name__ == "__main__":  # self-check: Eq. 1a bound, Eq. 7 vs finite differences, Theorem 3 by Monte Carlo
    W, m, roles, mask = build_network((4, 3, 3, 3), "CLAA", p_same=0.8, p_cross=0.2, seed=1)
    K, alpha, rng = len(roles), 0.2, np.random.default_rng(0)
    v = stimulus(0.3, m, roles)
    pi = flow(W, v, alpha)
    for n in (5, 10, 20):
        assert np.abs(flow(W, v, alpha, n) - pi).sum() <= 2 * (1 - alpha) ** n
    L = lambda W_: map_equation(*module_stats(W_, pi, m, K)[:2], pi)  # noqa: E731 (pi held fixed)
    gr, h = grad_L(W, pi, m, K), 1e-6
    for i, j in zip(*np.nonzero(mask)):
        E = np.zeros_like(W)
        E[i, j] = h
        assert abs((L(W + E) - L(W - E)) / (2 * h) - gr[i, j]) < 1e-6
    p, q, JK, _ = module_stats(W, pi, m, K)
    g = modulator(p, q, roles, clip=np.inf)
    assert np.allclose((W * centered(g, W, m)).sum(0), 0)  # conserves d_j in expectation
    J = W / W.sum(0) * pi
    mean_dW = sum(-centered(g, W, m) * sample_pairings(J, 500, rng) for _ in range(4000)) / 4000
    r = np.corrcoef(mean_dW[mask], (-W * gr * 500)[mask])[0, 1]
    assert r > 0.99, r
    a = int(np.flatnonzero(roles == "A")[0])  # Eq. 4b: expected increment (kappa = J) = W * dlog P~/dW, pi fixed
    for beta in (1.0, 3.0):  # beta = 3: the sharpened score (Eq. 1e, Proposal)
        def logP(W_, beta=beta):
            Jin = np.bincount(m, (W_ / W_.sum(0) * pi).sum(1), K)[roles == "A"] ** beta
            return np.log(Jin[0] / Jin.sum())
        inc = eligibility_step(0, W, J, pi, m, roles, a, tau_e=1.0, beta=beta)
        for i, j in zip(*np.nonzero(mask)):
            E = np.zeros_like(W)
            E[i, j] = h
            assert abs(W[i, j] * (logP(W + E) - logP(W - E)) / (2 * h) - inc[i, j]) < 1e-6
    W2 = structural_update(W * 1.5, mask, 0 * g, np.ones_like(W), m, 0.1, eps=1.0)  # homeostasis pulls d_j to 1
    assert (np.abs(W2.sum(0) - 1) < np.abs(1.5 * W.sum(0) - 1)).all()
    W3 = homeostasis_update(W * 1.5, mask, np.ones_like(W), 0.1)
    assert (np.abs(W3.sum(0) - 1) < np.abs(1.5 * W.sum(0) - 1)).all()
    Wc, mc, rc, _ = build_network((36, 4, 4), "CLA")
    vc = cartpole_stimulus([0, 0, 0.21, -3.5], mc, rc)  # theta at +max, theta_dot at -max: grid cell (5, 0)
    assert np.isclose(vc.sum(), 1) and vc.argmax() == 30
    print(f"core self-check ok (Monte Carlo corr with -W*grad = {r:.4f}; Eq. 4b score, homeostasis, encoder)")
