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


def run_frame(snn, v, k, rate, rng, theta=0.5, use="jit"):
    """Cold-start frame of k steps (one walk per frame, derivation §2.3 and §8 step 2): snn.reset() zeroes states and
    refractory counters and clears the spike train and queued inputs, then each neuron gets U(-theta, theta) escape
    noise plus amplitude-2 theta teleportation spikes with probability rate * v_i per step. Returns ispikes (exactly
    this frame's k x N bool raster) and the number of external spikes."""
    N = snn.num_neurons
    ext = rng.random((k, N)) < rate * v
    inp = rng.uniform(-theta, theta, (k, N)) + 2 * theta * ext
    snn.reset()
    nids = list(range(N))
    snn.input_spikes = {t: {"nids": nids, "values": inp[t].tolist()} for t in range(k)}
    snn.simulate(k, use=use)
    return snn.ispikes, int(ext.sum())


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
    print("spiking self-check ok")
