"""Feed-forward TD-STDP actor-critic of Chung & Kozma 2020 [CK] (arXiv:2008.13044) on the branching engine
(iteration 013, step 1: a baseline, not Map-STDP). [CK Eq. n] cites that paper's equations.

Network: n_in Bernoulli input neurons drive the posts (critic E, critic I, then per action k actor E, actor I);
feed-forward, no recurrence. W[i, j] is the synapse from input j to post i (repo convention; CK write w_ij for
input i -> neuron j), and every trace below is indexed [post, pre] like W.
Neuron: the spiking.run_heads branching step without presynaptic normalisation, X_i(t) = [U(-theta, theta) +
(W x(t-1))_i > theta], so P(spike) = clip((W x(t-1))_i, 0, 1) with 2 theta = 1 (memoryless, 1 step = 1 ms).
Rates rho (Eq. 3, discrete: rho <- e^(-1/tau_n) rho + X / tau_n); value V = alpha (mean rho_E - mean rho_I) + beta
over the critic (Eq. 4); policy s = softmax(alpha_a (rho_E,k - rho_I,k)) (Eqs. 10-13); TD error
delta_t = e^(-1/tau) V_(t+1) + e^(-1/2tau) r_t - V_t (Eq. 69, Delta t = 1 ms; V_(t+1) := 0 in the last 2 ms of an
episode). With A+ = 1, A- = 0 (Table 1) the post trace P_j is unused: pre trace P_i (Eq. 71/75),
z_ij <- e^(-1/tau_z) z_ij + P_i X_j (Eq. 72/76), actor q_ij <- e^(-1/tau_q) q_ij + (A_k - s_k) z_ij (Eq. 77);
w += sign eta delta_t z(t) for the critic (Eq. 73) and sign eta delta_t q(t) for the actor (Eq. 78), sign +1 on
excitatory and -1 on inhibitory posts; entropy term sign c_e g_k z (Eq. 26), weight decay -c_w w / 2 (Eq. 27),
target rate -c_t (rho_e - rho_hat) z on excitatory actor posts (Eq. 31). delta_t needs V_(t+1), so the increment of
ms t is applied one ms later.
Block form: weights are frozen for blocks of L ms and the block's per-ms increments are summed in closed form
(zsum, block_learn); L = 1 is the per-ms rule exactly. The self-check verifies the closed form against the per-ms
loop.
Authors' code (reference/td_stdp, BindsNET 0.2.7; iteration 013 §8; the default since the audit): block_learn_ck
follows the code's per-ms order and trace scale, which the paper's equations omit: z <- e^(-1/tau_z) z + P X / tau_z,
q <- e^(-1/tau_q) q + (A - p) z(t-1) / tau_q, the update at ms t is delta_t z(t-1) (critic) and delta_t q(t) +
kap_t z(t-1) (actor), and p is the probability the action was sampled with (fixed over the 20 ms window).
simulate_bindsnet is BindsNET's LIFNodes (a spike through w adds w mV, threshold 13 mV above rest, no refractory
period); in branching units w_branching = w_mV / 13.
Not crossbar-native as a whole (iteration 013 reproduces CK first): softmax readout, signed weights, the per-synapse
trace pair (z, q) and Adam (per-synapse m, v) are off-hardware; the plan removes them in later steps."""
import numpy as np

THETA = 0.5
TAU_M, THETA_L = 100.0, 13.0  # CK §2.2: tau = 100 ms, theta - E_res = -52 - (-65) mV (true-LIF control)


class Layout:
    """Post populations, in order: critic E (n_ce), critic I (n_ci), then for each action k: actor E (n_ae), actor I
    (n_ai). sign: +1 excitatory, -1 inhibitory. Mv: V = alpha_c rho @ Mv + beta (Eq. 4). Ma (K x n_post): action
    logits / alpha_a = rho @ Ma.T (Eqs. 10-12). act: action of each actor post. ae: mask of excitatory actor posts."""

    def __init__(self, n_in, n_ce, n_ci, n_ae, n_ai, K):
        self.n_in, self.K, self.n_crit = n_in, K, n_ce + n_ci
        self.n_post = self.n_crit + K * (n_ae + n_ai)
        blk = np.r_[np.ones(n_ae), -np.ones(n_ai)]
        self.sign = np.r_[np.ones(n_ce), -np.ones(n_ci), np.tile(blk, K)]
        self.Mv = np.r_[np.ones(n_ce) / max(n_ce, 1), -np.ones(n_ci) / max(n_ci, 1),
                        np.zeros(self.n_post - self.n_crit)]
        self.act = np.repeat(np.arange(K), n_ae + n_ai)
        self.Ma = np.zeros((K, self.n_post))
        for k in range(K):
            o = self.n_crit + k * (n_ae + n_ai)
            self.Ma[k, o:o + n_ae] = 1 / n_ae
            self.Ma[k, o + n_ae:o + n_ae + n_ai] = -1 / max(n_ai, 1)
        self.ae = np.zeros(self.n_post - self.n_crit, bool)
        self.ae[np.flatnonzero(blk > 0)[None] + (n_ae + n_ai) * np.arange(K)[:, None]] = True


def init_weights(lay, fbar, rate, rng, unit=1.0):
    """W ~ U(0, w0), w0 = 2 rate unit / sum(fbar): every post's expected initial spike probability per ms is `rate` at
    the mean input activity fbar (pre-registered init, iteration 013; CK do not report theirs). unit: input per spike
    per ms of rate, 1 in branching mode, THETA_L TAU_M for the true-LIF control (high-rate limit)."""
    return rng.uniform(0, 2 * rate * unit / fbar.sum(), (lay.n_post, lay.n_in))


def draw(f, L, n_post, rng):
    """One block's random draws: input spikes x ~ Bernoulli(f) (L x H x n_in) and escape noise U(-theta, theta)
    (L x H x n_post). Shared by both engines, so they see identical draws."""
    x = (rng.random((L, *f.shape)) < f).astype(float)
    return x, rng.uniform(-THETA, THETA, (L, len(f), n_post))


def simulate(W, x_prev, x, noise):
    """Branching step of L ms (numpy engine): X(t) = [noise(t) + W x(t-1) > theta], x(-1) = x_prev (H x n_in)."""
    X = np.empty(noise.shape, bool)
    xp = x_prev
    for t in range(len(x)):
        X[t] = noise[t] + xp @ W.T > THETA
        xp = x[t]
    return X


def simulate_snm(W, x_prev, x, noise, snn=None):
    """simulate() on SuperNeuroMAT (leak = inf, delay-1 synapses, input neurons forced by 2 theta inputs), one head.
    SNM step s = t + 1: input neurons spike x(s - 1) (x_prev at s = 0), posts get noise(s - 1); the step-0 post row
    is discarded. Returns (X, snn); pass snn back to reuse the network (weights are rewritten every call)."""
    from mapstdp import spiking as sp
    n_post, n_in = W.shape
    Wf = np.zeros((n_in + n_post,) * 2)
    Wf[n_in:, :n_in] = W
    if snn is None:
        snn = sp.build(np.pad(np.ones_like(W), ((n_in, 0), (0, n_post))), gain=1.0, theta=THETA)
    sp.set_weights(snn, Wf, gain=1.0, presyn_norm=False)
    L = len(x)
    inp = np.zeros((L + 1, n_in + n_post))
    inp[:, :n_in] = 2 * THETA * np.vstack([x_prev[0], x[:, 0]])
    inp[0, n_in:] = -THETA
    inp[1:, n_in:] = noise[:, 0]
    snn.reset()
    sp.set_inputs(snn, inp)
    snn.simulate(L + 1, use="jit")
    return np.asarray(snn.ispikes)[1:, n_in:][:, None].astype(bool), snn


def simulate_lif(W, x_prev, x, v, leak="exp"):
    """True-LIF control (iteration 013): CK Eq. 1 with rest 0, threshold THETA_L = 13 mV above rest, reset to rest,
    no refractory period, no noise; W in CK units, so a presynaptic spike moves the membrane by W / TAU_M (R = 1,
    CK Eq. 44). leak 'exp': v <- e^(-1/TAU_M) v (CK Eq. 1); 'snm': SuperNeuroMAT's subtractive leak, v moves
    THETA_L / TAU_M towards rest per step. Then v += W x(t-1) / TAU_M, spike if v > THETA_L, reset.
    v: H x n_post membrane state, carried across blocks. Returns (X, v)."""
    X, v, xp, a, lk = np.empty((len(x), *v.shape), bool), v.copy(), x_prev, np.exp(-1 / TAU_M), THETA_L / TAU_M
    Wt = W.T / TAU_M
    for t in range(len(x)):
        v = a * v if leak == "exp" else np.where(v > 0, np.maximum(v - lk, 0), np.minimum(v + lk, 0))
        v = v + xp @ Wt
        X[t] = v > THETA_L
        v[X[t]] = 0
        xp = x[t]
    return X, v


def simulate_bindsnet(W, x_prev, x, v):
    """BindsNET 0.2.7 LIFNodes as the authors configure them (reference/td_stdp main.py: refrac 0, tc_decay 100),
    rest = reset = 0, threshold THETA_L = 13 mV: v <- e^(-1/TAU_M) v + W x(t-1) (W in mV, no 1 / tau), spike if
    v >= THETA_L, reset to 0. Deterministic. v: H x n_post membrane state, carried across blocks. Returns (X, v)."""
    X, v, xp, a = np.empty((len(x), *v.shape), bool), v.copy(), x_prev, np.exp(-1 / TAU_M)
    for t in range(len(x)):
        v = a * v + xp @ W.T
        X[t] = v >= THETA_L
        v[X[t]] = 0
        xp = x[t]
    return X, v


def pre_trace(P0, x, tau_p):
    """Pre traces P(t) for t = 0..L-1 (Eq. 71/75): P <- e^(-1/tau_p) P + x(t)."""
    a, P, out = np.exp(-1 / tau_p), P0, np.empty_like(x)
    for t in range(len(x)):
        P = a * P + x[t]
        out[t] = P
    return out


def rates(rho0, X, tau_n):
    """Filtered rates rho(t), t = 0..L-1 (Eq. 3 discretised)."""
    a, rho, out = np.exp(-1 / tau_n), rho0, np.empty(X.shape)
    for t in range(len(X)):
        rho = a * rho + X[t] / tau_n
        out[t] = rho
    return out


def softmax(logits):
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def entropy_coef(s):
    """g_k = -sum_m s_m (log s_m + 1)(I[m = k] - s_k) = -s_k (log s_k + H(s)) (Eq. 25)."""
    ls = np.log(np.maximum(s, 1e-300))
    return -s * (ls - (s * ls).sum(-1, keepdims=True))


def zsum(z0, P, X, u, a):
    """sum_(t=-1..L-1) u_t * z(t) for the trace z(t) = a z(t-1) + X(t) P(t)^T (outer product per head; Eq. 72/76),
    z(-1) = z0. z0: H x n x n_in, P: L x H x n_in, X: L x H x n, u: M x (L+1) x H x n (u[:, 0] is t = -1, broadcast
    along presynaptic inputs). Returns M x H x n x n_in. Closed form:
    z(t) = a^(t+1) z0 + sum_(s<=t) a^(t-s) X(s) P(s)^T, so the sum is coef0 z0 + sum_s (X(s) U_s) P(s)^T with
    U_s = sum_(t>=s) a^(t-s) u_t: one batched matmul."""
    M, L1, H, n = u.shape
    L = L1 - 1
    U = np.empty((M, L, H, n))
    acc = np.zeros((M, H, n))
    for s in range(L - 1, -1, -1):
        acc = u[:, s + 1] + a * acc
        U[:, s] = acc
    coef0 = u[:, 0] + a * U[:, 0] if L else u[:, 0]
    A = (X[None] * U).transpose(2, 0, 3, 1).reshape(H, M * n, L)
    R = np.matmul(A, P.transpose(1, 0, 2)).reshape(H, M, n, P.shape[-1]).transpose(1, 0, 2, 3)
    return z0[None] * coef0[..., None] + R


def block_learn(z0, q0, P, X, d, c, kap, lay, tau_z, tau_q):
    """One block's summed update direction and end traces (frozen W). d: (L+1) x H TD errors delta_t for
    t = -1..L-1 (0 where unknown or not learning); c: L x H x n_act feedback A_k - s_k(t) (Eq. 77); kap: L x H x n_act
    per-post z coefficients applied without lag (entropy, Eq. 26; target rate, Eq. 31).
    Critic rows: sign sum_t delta_t z(t) (Eq. 73). Actor rows: sign sum_t delta_t q(t) + sum_t kap_t z(t) (Eqs. 78,
    26, 31), with q(t) = b^(t+1) q0 + sum_(u<=t) b^(t-u) c_u z(u), so sum_t d_t q(t) = D_(-1) q0 + sum_u D_u c_u z(u),
    D_u = sum_(t>=u) d_t b^(t-u). Returns (G, z_end, q_end): G is n_post x n_in per head (H x n_post x n_in)."""
    a, b, nc = np.exp(-1 / tau_z), np.exp(-1 / tau_q), lay.n_crit
    L, H = len(P), d.shape[1]
    sc, sa = lay.sign[:nc], lay.sign[nc:]
    e_end = np.zeros((L + 1, H, 1))
    e_end[L] = 1
    uc = np.stack([d[..., None] * sc, np.broadcast_to(e_end, (L + 1, H, nc))])
    Gc, zc = zsum(z0[:, :nc], P, X[..., :nc], uc, a)
    D = np.empty_like(d)
    D[L] = d[L]
    for i in range(L - 1, -1, -1):
        D[i] = d[i] + b * D[i + 1]
    na = lay.n_post - nc
    ug = np.zeros((L + 1, H, na))
    ug[1:] = D[1:, :, None] * c * sa + kap
    uq = np.zeros((L + 1, H, na))
    uq[1:] = b ** np.arange(L - 1, -1, -1)[:, None, None] * c
    Ga, qa, za = zsum(z0[:, nc:], P, X[..., nc:], np.stack([ug, uq, np.broadcast_to(e_end, (L + 1, H, na))]), a)
    Ga = Ga + D[0][:, None, None] * sa[:, None] * q0
    qa = qa + b ** L * q0
    return np.concatenate([Gc, Ga], 1), np.concatenate([zc, za], 1), qa


def block_learn_ck(z0, q0, P, X, d, c, kap, lay, tau_z, tau_q):
    """One block's summed update direction and end traces in the authors' per-ms order (bindsnet_add.py
    FBTDSTDP._connection_update; W frozen). Per ms t = 0..L-1, with z = z(t-1) on entry:
      q <- b q + c_t z / tau_q;  G += sign d_t z (critic rows), sign d_t q + kap_t z (actor rows);
      z <- a z + X(t) P(t)^T / tau_z   (a = e^(-1/tau_z), b = e^(-1/tau_q)).
    The 1 / tau_z and 1 / tau_q trace scale is the code's (CK Eqs. 72, 77 omit it). d: L x H TD errors delta_t
    (delta_t = e^(-1/tau) V_t + e^(-1/2tau) r - V_(t-1), all known inside the block, so there is no lag across blocks);
    c: L x H x n_act feedback A_k - p_k (p = the sampling probability); kap: L x H x n_act per-post z coefficients
    (entropy, sign included). z0, q0 and the returned traces are the normalised ones. Closed form with zsum, using
    sum_t d_t q(t) = b D_0 q0 + sum_s D_s c_s z(s-1) / tau_q, D_s = sum_(t>=s) d_t b^(t-s).
    Returns (G, z_end, q_end)."""
    a, b, nc = np.exp(-1 / tau_z), np.exp(-1 / tau_q), lay.n_crit
    L, H = d.shape
    Pn, sc, sa, na = P / tau_z, lay.sign[:nc], lay.sign[nc:], lay.n_post - nc
    e_end = np.zeros((L + 1, H, 1))
    e_end[L] = 1  # zsum index i <-> z(i - 1): u[i] multiplies z(i - 1), u[L] the end trace z(L - 1)
    uc = np.zeros((L + 1, H, nc))
    uc[:L] = d[..., None] * sc
    Gc, zc = zsum(z0[:, :nc], Pn, X[..., :nc], np.stack([uc, np.broadcast_to(e_end, (L + 1, H, nc))]), a)
    D = np.empty_like(d)
    D[L - 1] = d[L - 1]
    for i in range(L - 2, -1, -1):
        D[i] = d[i] + b * D[i + 1]
    ug, uq = np.zeros((L + 1, H, na)), np.zeros((L + 1, H, na))
    ug[:L] = D[..., None] * sa * c / tau_q + kap
    uq[:L] = b ** np.arange(L - 1, -1, -1)[:, None, None] * c / tau_q
    Ga, qa, za = zsum(z0[:, nc:], Pn, X[..., nc:], np.stack([ug, uq, np.broadcast_to(e_end, (L + 1, H, na))]), a)
    Ga = Ga + b * D[0][:, None, None] * sa[:, None] * q0
    qa = qa + b ** L * q0
    return np.concatenate([Gc, Ga], 1), np.concatenate([zc, za], 1), qa


def _ucoef(u, a):
    """zsum's coefficients: U_s = sum_(t>=s) a^(t-s) u_(t+1) (M x L x H x n) and coef0 = u_0 + a U_0 (M x H x n)."""
    M, L1, H, n = u.shape
    U, acc = np.empty((M, L1 - 1, H, n)), np.zeros((M, H, n))
    for s in range(L1 - 2, -1, -1):
        acc = u[:, s + 1] + a * acc
        U[:, s] = acc
    return U, (u[:, 0] + a * U[:, 0] if L1 > 1 else u[:, 0])


def block_learn_ck_fast(z0, q0, P, X, d, c, kap, lay, tau_z, tau_q, K=None):
    """block_learn_ck with the update summed over copies inside one matrix product (iteration 013 §10, S2): returns
    (G summed over the H copies, n_post x n_in; z_end; q_end). Equal to block_learn_ck(...)[0].sum(0) and its end
    traces up to floating-point summation order (self-check (b'')). K (H x n_post x n_in, optional): an extra z
    increment per ms, z <- a z + (X P^T + K) / tau_z, constant over the block (the flow engine's input
    autocorrelation term, S3)."""
    a, b, nc = np.exp(-1 / tau_z), np.exp(-1 / tau_q), lay.n_crit
    L, H = d.shape
    Pn, sc, sa, n_in = P / tau_z, lay.sign[:nc], lay.sign[nc:], P.shape[-1]
    u = np.zeros((3, L + 1, H, lay.n_post))  # critic + actor policy/entropy rows, actor q rows (padded), end
    u[0, :L, :, :nc] = d[..., None] * sc
    D = np.empty_like(d)
    D[L - 1] = d[L - 1]
    for i in range(L - 2, -1, -1):
        D[i] = d[i] + b * D[i + 1]
    u[0, :L, :, nc:] = D[..., None] * sa * c / tau_q + kap
    u[1, :L, :, nc:] = b ** np.arange(L - 1, -1, -1)[:, None, None] * c / tau_q
    u[2, L] = 1
    U, c0 = _ucoef(u, a)
    XU = X[None] * U  # 3 x L x H x n_post
    G = (XU[0].reshape(L * H, -1).T @ Pn.reshape(L * H, n_in)) + np.einsum("hn,hni->ni", c0[0], z0)
    G[nc:] += sa[:, None] * np.einsum("h,hni->ni", b * D[0], q0)
    E = np.concatenate([XU[2], XU[1][..., nc:]], -1).transpose(1, 2, 0)  # H x (n_post + n_act) x L
    R = np.matmul(E, Pn.transpose(1, 0, 2))
    z_end = a ** L * z0 + R[:, :lay.n_post]
    q_end = b ** L * q0 + c0[1][:, nc:, None] * z0[:, nc:] + R[:, lay.n_post:]
    if K is not None:
        Kn, Us = K / tau_z, U.sum(1)  # sum over s of U_s: the coefficient of a per-ms constant increment
        G += np.einsum("hn,hni->ni", Us[0], Kn)
        z_end += Us[2][..., None] * Kn
        q_end += Us[1][:, nc:, None] * Kn[:, nc:]
    return G, z_end, q_end


def flow_step(W, f_prev, f, L):
    """The branching step's noise-free limit (iteration 013 §10, S3): over a block of L ms with input rates f (H x n_in;
    f_prev for the first ms), X(t) = clip(W f(t-1), 0, 1) (L x H x n_post, a spike probability) and the input
    autocorrelation term of E[X_j(t) P_i(t)] - E[X_j] E[P_i] = W_ji a_p f_i (1 - f_i) without the factor a_p (the
    caller multiplies by e^(-1/tau_p)), masked to posts in the linear regime. Returns (x = f broadcast over the block,
    X, K0 = W_ji f_i (1 - f_i) for the current f, H x n_post x n_in)."""
    x = np.broadcast_to(f, (L, *f.shape)).copy()  # writable: the runner resets its last row on episode resets
    p_prev, p = np.clip(f_prev @ W.T, 0, 1), f @ W.T
    X = np.concatenate([p_prev[None], np.broadcast_to(np.clip(p, 0, 1), (L - 1, *p.shape))])
    lin = (p > 0) & (p < 1)
    return x, X, lin[..., None] * W[None] * (f * (1 - f))[:, None, :]


class Adam:
    """Adam on the ascent direction g (CK §3.2 comment 2: the per-ms update fed to Adam as its gradient), b1, b2 and
    the step lr per ms. step(G, lr, L) takes the sum G of L per-ms updates (block form) and matches per-ms Adam's
    moments: t advances by L ms, m <- b1^L m + (1 - b1^L) G / L, v <- b2^L v + (1 - b2^L) G^2 / L, and W moves by
    L lr m_hat / sqrt(v_hat). For noise-dominated per-ms updates E[G^2 / L] = E[g^2], so m and v have per-ms Adam's
    statistics; feeding G itself would make m / sqrt(v) sqrt(L) times larger (iteration 013: that variant collapsed
    3/10 CartPole seeds at L = 20). L = 1 is standard Adam. Per-synapse moments m, v: off-hardware."""

    def __init__(self, shape, b1, b2, eps=1e-8):
        self.m, self.v, self.t, self.b1, self.b2, self.eps = np.zeros(shape), np.zeros(shape), 0, b1, b2, eps

    def step(self, g, lr, L=1):
        self.t += L
        a1, a2 = self.b1 ** L, self.b2 ** L
        self.m = a1 * self.m + (1 - a1) * (g / L)
        self.v = a2 * self.v + (1 - a2) * (g * g / L)
        mh, vh = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        return lr * L * mh / (np.sqrt(vh) + self.eps)


def _per_ms_reference(z0, q0, P, X, d, c, kap, lay, tau_z, tau_q):
    """Per-ms loop of Eqs. 76-78 (and 72-73) for the block_learn self-check."""
    a, b, nc = np.exp(-1 / tau_z), np.exp(-1 / tau_q), lay.n_crit
    z, q = z0.copy(), q0.copy()
    G = np.zeros_like(z0)
    G[:, :nc] += d[0][:, None, None] * lay.sign[:nc, None] * z[:, :nc]
    G[:, nc:] += d[0][:, None, None] * lay.sign[nc:, None] * q
    for t in range(len(P)):
        z = a * z + X[t][..., None] * P[t][:, None, :]
        q = b * q + c[t][..., None] * z[:, nc:]
        G[:, :nc] += d[t + 1][:, None, None] * lay.sign[:nc, None] * z[:, :nc]
        G[:, nc:] += d[t + 1][:, None, None] * lay.sign[nc:, None] * q + kap[t][..., None] * z[:, nc:]
    return G, z, q


def _per_ms_reference_ck(z0, q0, P, X, d, c, kap, lay, tau_z, tau_q, K=0.0):
    """Literal per-ms transcription of FBTDSTDP._connection_update (bindsnet_add.py:180-224) for the block_learn_ck
    self-check: the action trace from the old eligibility trace, the update from the old z and the new q, then z."""
    a, b, nc = np.exp(-1 / tau_z), np.exp(-1 / tau_q), lay.n_crit
    z, q = z0.copy(), q0.copy()
    G = np.zeros_like(z0)
    for t in range(len(P)):
        q = b * q + c[t][..., None] * z[:, nc:] / tau_q
        G[:, :nc] += d[t][:, None, None] * lay.sign[:nc, None] * z[:, :nc]
        G[:, nc:] += d[t][:, None, None] * lay.sign[nc:, None] * q + kap[t][..., None] * z[:, nc:]
        z = a * z + (X[t][..., None] * P[t][:, None, :] + K) / tau_z
    return G, z, q


def _cos(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _row_cos(a, b):
    return float(np.mean([_cos(x, y) for x, y in zip(a, b)]))


if __name__ == "__main__":
    from mapstdp import spiking as sp
    rng = np.random.default_rng(0)
    # (a) numpy engine == SuperNeuroMAT raster on the same draws
    lay = Layout(7, 3, 2, 2, 1, 2)
    W = rng.uniform(-0.05, 0.2, (lay.n_post, lay.n_in))
    f = rng.uniform(0, 1, (1, lay.n_in))
    x_prev = (rng.random((1, lay.n_in)) < 0.5).astype(float)
    x, noise = draw(f, 300, lay.n_post, rng)
    X1 = simulate(W, x_prev, x, noise)
    X2, _ = simulate_snm(W, x_prev, x, noise)
    assert X1.shape == X2.shape and (X1 == X2).all() and 0.05 < X1.mean() < 0.95, X1.mean()
    print(f"(a) numpy step == SuperNeuroMAT raster: {X1.size} spike slots, rate {X1.mean():.3f}")
    # (a') true-LIF control, SuperNeuroMAT leak: simulate_lif(leak='snm') == SuperNeuroMAT (finite leak) raster;
    # W / TAU_M on a 1/64 grid so both engines sum exactly
    import superneuromat as snm
    Wl = rng.integers(-8, 64, (lay.n_post, lay.n_in)) / 64 * TAU_M
    net = snm.SNN()
    net.stdp, net.sparse = False, False
    for i in range(lay.n_in + lay.n_post):
        inp_n = i < lay.n_in
        net.create_neuron(threshold=THETA if inp_n else THETA_L, leak=np.inf if inp_n else THETA_L / TAU_M,
                          reset_state=0.0, refractory_period=0)
    for i, j in np.ndindex(Wl.shape):
        net.create_synapse(j, lay.n_in + i, weight=Wl[i, j] / TAU_M, delay=1, stdp_enabled=False)
    x = (rng.random((2000, 1, lay.n_in)) < f).astype(float)
    inp = np.zeros((2001, lay.n_in + lay.n_post))
    inp[1:, :lay.n_in] = 2 * THETA * x[:, 0]
    sp.set_inputs(net, inp)
    net.simulate(2001, use="jit")
    S1 = np.asarray(net.ispikes)[2:, lay.n_in:]
    S2 = simulate_lif(Wl, x[0], x[1:], np.zeros((1, lay.n_post)), leak="snm")[0][:, 0]
    assert S1.shape == S2.shape and (S1 == S2).all() and 0.05 < S2.mean() < 0.9, S2.mean()
    print(f"(a') LIF with SuperNeuroMAT's leak == SuperNeuroMAT raster: {S2.size} spike slots, rate {S2.mean():.3f}")
    # (b) block closed form == per-ms loop (frozen W), L = 1 and L = 20, H = 3
    for L in (1, 20):
        H, na = 3, lay.n_post - lay.n_crit
        z0, q0 = rng.random((H, lay.n_post, lay.n_in)), rng.normal(size=(H, na, lay.n_in))
        P, X = rng.random((L, H, lay.n_in)), rng.random((L, H, lay.n_post)) < 0.3
        d, c, kap = rng.normal(size=(L + 1, H)), rng.normal(size=(L, H, na)), rng.normal(size=(L, H, na))
        args = (z0, q0, P, X, d, c, kap, lay, 20, 40)
        got, ref = block_learn(*args), _per_ms_reference(*args)
        err = max(np.abs(g - r).max() / np.abs(r).max() for g, r in zip(got, ref))
        assert err < 1e-12, err
        print(f"(b) block closed form == per-ms loop, L={L:2d}: max rel err {err:.1e}")
        got, ref = block_learn_ck(*args[:4], d[1:], *args[5:]), _per_ms_reference_ck(*args[:4], d[1:], *args[5:])
        err = max(np.abs(g - r).max() / np.abs(r).max() for g, r in zip(got, ref))
        assert err < 1e-12, err
        print(f"(b') authors' order and trace scale: closed form == per-ms loop, L={L:2d}: max rel err {err:.1e}")
        K = np.random.default_rng(1).random(z0.shape)  # own stream: keeps (c)-(d) draws unchanged
        args_ck = (*args[:4], d[1:], *args[5:])
        Gs, zf, qf = block_learn_ck_fast(*args_ck, K=K)
        Gr, zr, qr = _per_ms_reference_ck(*args_ck, K=K)
        err = max(np.abs(g - r).max() / np.abs(r).max() for g, r in ((Gs, Gr.sum(0)), (zf, zr), (qf, qr)))
        Gs, zf, qf = block_learn_ck_fast(*args_ck)
        Gd, zd, qd = block_learn_ck(*args_ck)
        err2 = max(np.abs(g - r).max() / np.abs(r).max() for g, r in ((Gs, Gd.sum(0)), (zf, zd), (qf, qd)))
        assert max(err, err2) < 1e-12, (err, err2)
        print(f"(b'') copy-summed fast form == block_learn_ck (rel err {err2:.1e}) and, with a constant z increment K, "
              f"== the per-ms loop (rel err {err:.1e}), L={L:2d}")
    # (c) STE in branching mode, critic (Theorem 1): E[z] vs the finite-difference gradient of E[V] = alpha mean(W f)
    # (exact in the linear regime, no clipping); delta is a constant here, so E[delta z] is delta E[z].
    lay = Layout(8, 4, 0, 1, 0, 1)
    f = rng.uniform(0.2, 0.9, (1, lay.n_in))
    W = rng.uniform(0.01, 0.06, (lay.n_post, lay.n_in))
    p = (W @ f[0])[:lay.n_crit]
    assert (0 < p).all() and (p < 1).all()
    T, L, nc = 200000, 1000, lay.n_crit
    zacc, z = np.zeros((nc, lay.n_in)), np.zeros((1, lay.n_post, lay.n_in))
    xp, Pt = np.zeros((1, lay.n_in)), np.zeros((1, lay.n_in))
    u = np.zeros((2, L + 1, 1, lay.n_post))
    u[0, 1:], u[1, L] = 1, 1  # sum_(t=0..L-1) z(t), and z(L-1)
    for _ in range(T // L):
        x, noise = draw(f, L, lay.n_post, rng)
        X = simulate(W, xp, x, noise)
        P = pre_trace(Pt, x, 20)
        Zs = zsum(z, P, X, u, np.exp(-1 / 20))
        zacc += Zs[0, 0, :nc]
        z, xp, Pt = Zs[1], x[-1], P[-1]
    Ez = zacc / T
    alpha, h = 2.0, 1e-6
    Ev = lambda Wc: alpha * np.mean(Wc @ f[0])  # noqa: E731
    fd = np.zeros((nc, lay.n_in))
    for i in range(nc):
        for j in range(lay.n_in):
            Wp = W[:nc].copy()
            Wp[i, j] += h
            fd[i, j] = (Ev(Wp) - Ev(W[:nc])) / h
    cg, cr = _cos(Ez, fd), _row_cos(Ez, fd)
    gain = Ez.sum(1) / fd.sum(1)
    print(f"(c) critic STE: cos(E[z], dE[V]/dW) global {cg:.4f}, per post row {cr:.5f}; row gain / p_j "
          f"{np.round(gain / p / gain.mean() * p.mean(), 3)} (the STE carries the post rate p_j as a row gain)")
    assert cr > 0.99 and cg > 0.9
    # (d) actor rule (Eqs. 74-78) on a 2-armed bandit: E[r(A) q] vs the finite-difference gradient of
    # E[r] = sum_k s_k(alpha_a W f) r_k (mean-field rates), action resampled every 20 ms
    lay = Layout(6, 0, 0, 3, 0, 2)
    f = rng.uniform(0.2, 0.9, (1, lay.n_in))
    W = rng.uniform(0.01, 0.06, (lay.n_post, lay.n_in))
    r_arm, aa, tq = np.array([1.0, 0.0]), 25.0, 40
    z, q = np.zeros((1, lay.n_post, lay.n_in)), np.zeros((1, lay.n_post, lay.n_in))
    xp, Pt, rho = np.zeros((1, lay.n_in)), np.zeros((1, lay.n_in)), np.tile(W @ f[0], (1, 1)) * 1.025
    G = np.zeros((lay.n_post, lay.n_in))
    dprev = np.zeros(1)
    T = 0
    for blk in range(20000):
        s0 = softmax(aa * rho @ lay.Ma.T)[0]
        A = rng.choice(2, p=s0)
        x, noise = draw(f, 20, lay.n_post, rng)
        X = simulate(W, xp, x, noise)
        P = pre_trace(Pt, x, 20)
        R = rates(rho, X, 20)
        s = softmax(aa * R @ lay.Ma.T)
        c = (np.eye(2)[A][lay.act] - s[..., lay.act])
        d = np.r_[dprev, np.full(19, r_arm[A]), 0.0][:, None]
        Gb, z, q = block_learn(z, q, P, X, d, c, np.zeros_like(c), lay, 20, tq)
        dprev = np.full(1, r_arm[A])
        if blk >= 100:
            G += Gb[0]
            T += 20
        xp, Pt, rho = x[-1], P[-1], R[-1]
    Er = lambda Wa: (softmax(aa * (Wa @ f[0]) @ lay.Ma.T) * r_arm).sum()  # noqa: E731
    fd = np.zeros_like(W)
    for i in range(lay.n_post):
        for j in range(lay.n_in):
            Wp = W.copy()
            Wp[i, j] += h
            fd[i, j] = (Er(Wp) - Er(W)) / h
    ca, cra = _cos(G, fd), _row_cos(G, fd)
    print(f"(d) actor rule on a bandit: cos(E[r(A) q], dE[r]/dW) global {ca:.4f}, per post row {cra:.4f}")
    assert ca > 0.9 and cra > 0.95
    # (d') the same bandit in the authors' order (block_learn_ck): window k presents the stimulus with (A_(k-1),
    # p_(k-1), r(A_(k-1))), p fixed at the sampling probability; then A_k ~ p_k = softmax at the window's end
    z, q = np.zeros((1, lay.n_post, lay.n_in)), np.zeros((1, lay.n_post, lay.n_in))
    xp, Pt, rho = np.zeros((1, lay.n_in)), np.zeros((1, lay.n_in)), np.tile(W @ f[0], (1, 1)) * 1.025
    G[:] = 0
    p0 = softmax(aa * rho @ lay.Ma.T)[0]
    A = rng.choice(2, p=p0)
    for blk in range(20000):
        x, noise = draw(f, 20, lay.n_post, rng)
        X = simulate(W, xp, x, noise)
        P = pre_trace(Pt, x, 20)
        R = rates(rho, X, 20)
        c = np.broadcast_to(np.eye(2)[A][lay.act] - p0[lay.act], (20, 1, lay.n_post))
        Gb, z, q = block_learn_ck(z, q, P, X, np.full((20, 1), r_arm[A]), c, np.zeros(c.shape), lay, 20, tq)
        if blk >= 100:
            G += Gb[0]
        xp, Pt, rho = x[-1], P[-1], R[-1]
        p0 = softmax(aa * rho @ lay.Ma.T)[0]
        A = rng.choice(2, p=p0)
    ca, cra = _cos(G, fd), _row_cos(G, fd)
    print(f"(d') actor rule, authors' order and trace scale: cos(E[r(A) q], dE[r]/dW) global {ca:.4f}, "
          f"per post row {cra:.4f}")
    assert ca > 0.9 and cra > 0.95
    # (e) flow limit (S3): with W, f, delta and the feedback fixed, the Monte Carlo average of the sampled rule's update
    # (branching step, block_learn_ck) == the flow update (flow_step, block_learn_ck_fast with K)
    lay = Layout(9, 4, 2, 3, 1, 2)
    f = rng.uniform(0.1, 0.9, (1, lay.n_in))
    W = rng.uniform(-0.01, 0.05, (lay.n_post, lay.n_in))
    assert ((W @ f[0] > 0) & (W @ f[0] < 1)).all()
    na, L, nb = lay.n_post - lay.n_crit, 20, 5000
    d, cc = rng.normal(size=(L, 1)), rng.normal(size=(L, 1, na))
    kap = 0.3 * rng.normal(size=(L, 1, na))
    zs, qs, Gs = np.zeros((1, lay.n_post, lay.n_in)), np.zeros((1, na, lay.n_in)), np.zeros((lay.n_post, lay.n_in))
    zf, qf, Gf = zs.copy(), qs.copy(), Gs.copy()
    xp, Pt, Pf = np.zeros((1, lay.n_in)), np.zeros((1, lay.n_in)), np.zeros((1, lay.n_in))
    for blk in range(nb):
        x, noise = draw(f, L, lay.n_post, rng)
        X = simulate(W, xp, x, noise)
        P = pre_trace(Pt, x, 20)
        G, zs, qs = block_learn_ck(zs, qs, P, X, d, cc, kap, lay, 20, 40)
        xq, Xf, K0 = flow_step(W, f, f, L)
        Pq = pre_trace(Pf, xq, 20)
        G2, zf, qf = block_learn_ck_fast(zf, qf, Pq, Xf, d, cc, kap, lay, 20, 40, K=np.exp(-1 / 20) * K0)
        if blk >= 50:
            Gs += G[0]
            Gf += G2
        xp, Pt, Pf = x[-1], P[-1], Pq[-1]
    cf = _cos(Gs, Gf)
    print(f"(e) flow limit: cos(Monte Carlo mean of the sampled update, flow update) {cf:.5f}, "
          f"norm ratio {np.linalg.norm(Gs) / np.linalg.norm(Gf):.4f}")
    assert cf > 0.99
    print("tdstdp self-check ok")
