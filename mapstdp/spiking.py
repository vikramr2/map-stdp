"""SuperNeuroMAT backend for Map-STDP (built-in STDP off; plasticity is applied per frame from ispikes).

Neuron: threshold theta, reset 0, refractory 0, delay-1 synapses with W_snm = gain * W.T.
Noise: every neuron gets i.i.d. U(-theta, theta) input each step. With leak=inf (branching mode) the
state is input + W_snm.T @ z, so P(spike) = clip(u / (2 theta), 0, 1): linear escape noise, and one
step is one hop of a branching process. With 2 theta = 1 and gain = 1 - alpha, each spike has on
average 1 - alpha offspring spread by T (when d_j = 1). Teleportation: Bernoulli input spikes of
amplitude 2 theta to neuron i with probability rate * v_i per step.
Time constant: tau = 1 + theta / leak steps (synaptic delay plus the time the subtractive leak takes
to drain a threshold-sized charge); tau = 1 in branching mode.
"""
import numpy as np
import superneuromat as snm


def build(W, gain=0.8, leak=np.inf, theta=0.5):
    """One neuron per row of W, delay-1 synapse j -> i of weight gain * W[i, j] (W_snm = gain * W.T), STDP off.
    Synapses only where W != 0 (sparsity fixed thereafter); no presyn normalisation here, so call set_weights
    before running."""
    snn = snm.SNN()
    snn.stdp = False
    snn.sparse = False  # 'auto' turns sparse above ~150 neurons, and the jit backend is dense-only
    for _ in range(len(W)):
        snn.create_neuron(threshold=theta, leak=leak, reset_state=0.0, refractory_period=0)
    for i, j in zip(*np.nonzero(W)):
        snn.create_synapse(int(j), int(i), weight=gain * W[i, j], delay=1, stdp_enabled=False)
    return snn


def set_weights(snn, W, gain=0.8, presyn_norm=True):
    """W_snm = gain * W.T. presyn_norm divides each presynaptic neuron's output by d_j (a per-neuron
    output gain, one masked crossbar read), so the network implements T and its rates do not drift with d_j
    (derivation §2.2, mean-field condition 1). T, hence L and pi, is invariant to this column scaling.
    Only synapses created by build (the support of W at build time) are written; new nonzeros are silently ignored."""
    snn.set_weights_from_mat(gain * (W / np.maximum(W.sum(0), 1e-12) if presyn_norm else W).T)


def tau(leak, theta=0.5):
    """Effective time constant in steps, 1 + theta / leak (module docstring)."""
    return 1 + theta / leak


def set_inputs(snn, inp):
    """Queue a k x N input array for the next simulate(k): the array SuperNeuroMAT 3.5.0's setup_input_spikes would
    build from an input_spikes dict with every neuron listed at every step (that pure-Python loop was half the frame
    time). Overrides setup_input_spikes on this instance only; the queued dict stays empty."""
    snn.input_spikes = {}
    arr = np.vstack([inp, np.zeros((1, inp.shape[1]))]).astype(snn.dd)  # row k: nothing queued after the frame
    snn.setup_input_spikes = lambda time_steps: setattr(snn, "_input_spikes", arr)


def tile_heads(W, H):
    """Block-diagonal W for H heads: independent spiking replicas of one network on shared weights (Proposal). Head h
    is neurons h N .. (h + 1) N - 1; the heads' noise and teleportation draws are independent."""
    return W if H == 1 else np.kron(np.eye(H), W)


def head_groups(groups, H, N):
    """Neuron groups pooled over H heads, so race() runs one race over all heads' spikes (still Eq. 1d, §2.4)."""
    return [np.concatenate([np.asarray(g) + h * N for h in range(H)]) for g in groups]


def pooled(S, H):
    """Frame readouts pooled over H heads of a k x (H N) raster: (pi_hat of the summed counts, the sum of per-head
    covariance counts, total spikes). Cross-head coincidences are excluded: heads share weights, not synapses."""
    Sh = S.reshape(len(S), H, -1)
    return pi_hat(Sh.sum(1)), sum(pairings(Sh[:, h])[2] for h in range(H)), int(S.sum())


def run_frame(snn, v, k, rate, rng, theta=0.5, use="jit", return_ext=False):
    """Cold-start frame of k steps (one walk per frame, derivation §2.3 and §8 step 2): snn.reset() zeroes states and
    refractory counters and clears the spike train and queued inputs, then each neuron gets U(-theta, theta) escape
    noise plus amplitude-2 theta teleportation spikes with probability rate * v_i per step. Returns ispikes (exactly
    this frame's k x N bool raster) and the number of external spikes, or with return_ext the k x N bool raster of
    external spikes (each one forces its neuron to spike: state > theta whatever the noise)."""
    N = snn.num_neurons
    ext = rng.random((k, N)) < rate * v
    inp = rng.uniform(-theta, theta, (k, N)) + 2 * theta * ext
    snn.reset()
    set_inputs(snn, inp)
    snn.simulate(k, use=use)
    return snn.ispikes, (ext if return_ext else int(ext.sum()))


def run_heads(W, v, H, k, rate, rng, theta=0.5, gain=0.8):
    """run_frame on build(tile_heads(W, H)) + set_weights, without SuperNeuroMAT: in branching mode (leak = inf,
    refractory 0) its LIF step is z_t = [inp_t + z_{t-1} W_snm > theta], so all heads share one N x N product per step
    instead of a dense (H N)^2 one. Same rng draws as run_frame, and the self-check requires the same raster."""
    N = len(W)
    Wsnm = gain * (W / np.maximum(W.sum(0), 1e-12)).T  # [pre, post], as set_weights writes it
    ext = rng.random((k, H * N)) < rate * np.tile(v, H)
    inp = (rng.uniform(-theta, theta, (k, H * N)) + 2 * theta * ext).reshape(k, H, N)
    S, z = np.zeros((k, H, N), bool), np.zeros((H, N))
    for t in range(k):
        S[t] = (inp[t] + z @ Wsnm) > theta
        z = S[t].astype(float)
    return S.reshape(k, H * N)


def pi_hat(S):
    """Flow estimate: the frame's normalised spike counts c_j / sum c (batch form of Eq. 2 under the mean-field
    assumption, derivation §2.2; error ~1.3 / sqrt(S_mod), §2.3)."""
    c = S.sum(0).astype(float)
    return c / max(c.sum(), 1.0)


def race(S, groups, rng, burn=0):
    """Race readout (derivation §2.4): index of the group (list of neuron-index arrays) that spikes first in the
    frame at or after step `burn` (skips the cold-start transient, so the race matches Eq. 1d); ties within a step
    split in proportion to that step's spike counts. None if no group spikes."""
    cnt = np.stack([S[burn:, g].sum(1) for g in groups], 1)
    t = np.flatnonzero(cnt.sum(1))
    if not len(t):
        return None
    c = cnt[t[0]]
    return int(rng.choice(len(c), p=c / c.sum()))


def diffbound(S, groups, theta, rng, burn=0, cap=0, heads=1):
    """Count-difference race to a bound (iteration 010, Proposal): from step `burn` on, integrate each group's spike
    count (per step, at most `cap` spikes per group and head if cap > 0; groups pooled over `heads` heads as
    head_groups builds them) and stop at the first step at which one group leads every other by theta or more.
    For two groups whose spikes form an i.i.d. sequence of labels this is Eq. 1e with beta = theta (gambler's ruin).
    theta = 1 differs from race() only in same-step ties (no stop, keep counting).
    Returns (group index, deadline hit, action spikes counted up to the decision), or None if no group spikes.
    Deadline (end of the frame): the leader, ties uniform; a silent group simply never leads."""
    if cap > 0:
        cnt = np.stack([np.minimum(S[burn:, g].reshape(len(S) - burn, heads, -1).sum(2), cap).sum(1)
                        for g in groups], 1)
    else:
        cnt = np.stack([S[burn:, g].sum(1) for g in groups], 1)
    n = np.cumsum(cnt, 0)
    if not len(n) or not n[-1].any():
        return None
    top2 = np.sort(n, 1)[:, -2:]
    hit = np.flatnonzero(top2[:, 1] - top2[:, 0] >= theta)
    if len(hit):
        return int(np.argmax(n[hit[0]])), False, int(n[hit[0]].sum())
    lead = np.flatnonzero(n[-1] == n[-1].max())
    return int(lead[0] if len(lead) == 1 else rng.choice(lead)), True, int(n[-1].sum())


def pairings(S, lags=1):
    """Causal counts C[i, j] = #(pre j at t, post i at t + lag), lag = 1..lags; acausal is C.T.
    Returns (causal, causal - acausal, causal - chance), chance = c_i c_j sum_l (k - l) / k^2 from the
    frame's spike counts c (covariance count: a rank-1 pulse-count product, crossbar-native)."""
    S = S.astype(float)
    k, c = len(S), S.sum(0)
    C = sum(S[lag:].T @ S[:-lag] for lag in range(1, lags + 1))
    return C, C - C.T, C - np.outer(c, c) * sum(k - lag for lag in range(1, lags + 1)) / k ** 2


if __name__ == "__main__":  # self-check: W_snm orientation and lag-1 causal counting on a 2-neuron chain
    W = np.array([[0.0, 0.0], [1.0, 0.0]])  # 0 -> 1
    snn = build(W, gain=1.0)
    assert snn.weight_mat()[0, 1] == 1.0
    set_weights(snn, 2 * W, 1.0)
    assert snn.weight_mat()[0, 1] == 1.0  # presynaptic normalisation
    S, _ = run_frame(snn, np.array([1.0, 0.0]), 10, 1.0, np.random.default_rng(0))
    assert S[:, 0].all() and S[1:, 1].all() and not S[0, 1]
    C, B, V = pairings(S)
    assert C[1, 0] == 9 and B[1, 0] == 9 - C[0, 1] and np.isclose(V[1, 0], 9 - 10 * 9 * 9 / 100)
    r = race(np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0]], bool), [[0, 1], [2]], np.random.default_rng(0))
    assert r == 1 and race(np.zeros((3, 3), bool), [[0], [1]], None) is None
    assert race(np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], bool), [[0, 1], [2]], np.random.default_rng(0), burn=2) == 0
    D, G = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], bool), [[0], [1, 2]]  # bound, cap, deadline
    assert diffbound(D, G, 2, None) == (1, False, 2) and diffbound(D, G, 2, None, cap=1) == (1, False, 4)
    assert diffbound(D, G, 9, None) == (1, True, 6) and diffbound(D[1:2], G, 1, None) == (0, False, 1)
    assert diffbound(np.zeros((4, 2), bool), [[0], [1]], 2, None) is None
    D2 = np.hstack([D, D])  # two heads: the cap is per head, so the capped count doubles and so does the bound
    assert diffbound(D2, head_groups(G, 2, 3), 4, None, cap=1, heads=2) == (1, False, 8)
    rs = np.random.default_rng(0)  # K = 2 gambler's ruin: one labelled spike per step w.p. p -> P(0) = p^th / sum p^th
    for th in (2, 4):
        lab = rs.choice(3, p=[0.3, 0.2, 0.5], size=(4000, 300))  # label 2 = no spike this step
        wins = [diffbound(np.stack([x == 0, x == 1], 1), [[0], [1]], th, rs)[0] == 0 for x in lab]
        assert abs(np.mean(wins) - 0.6 ** th / (0.6 ** th + 0.4 ** th)) < 0.025, (th, np.mean(wins))
    from mapstdp import core  # heads: the numpy engine reproduces SuperNeuroMAT's raster spike for spike
    Wc, mc, rc, _ = core.build_network((12, 8, 8, 8), "CLAA", seed=1)
    vc, H = core.stimulus(0.4, mc, rc), 3
    snn = build(tile_heads(Wc, H))
    for s in range(5):
        set_weights(snn, tile_heads(Wc, H))
        S1 = run_frame(snn, np.tile(vc, H), 100, 1.0, np.random.default_rng(s))[0]
        S2 = run_heads(Wc, vc, H, 100, 1.0, np.random.default_rng(s))
        assert S1.shape == S2.shape and (S1 == S2).all() and S1.sum() > 0
    print("spiking self-check ok (incl. run_heads == SuperNeuroMAT raster)")
