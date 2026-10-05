"""M1 checks (docs/SPEC.md §11): flow-level reference vs SuperNeuroMAT on a small C/L/A network, no task.
Run from the repo root: python -m experiments.m1  -> experiments/results/m1.json + printed summary."""
import json
import pathlib
import time

import numpy as np

from mapstdp import core
from mapstdp import spiking as sp

ALPHA, N_HOPS, SEED = 0.2, 20, 0
W0, m, roles, MASK = core.build_network(seed=SEED)  # 80 neurons: C, L, L, A, A x 16
K, C = len(roles), int(np.flatnonzero(roles == "C")[0])
TEST = [core.stimulus(o, m, roles) for o in np.arange(16) / 16]
RATE = 1.0  # teleport spikes per step in total -> ~RATE / alpha = 5 spikes per step network-wide
P = 200  # pairings per frame for expected / sampled kappa (~ balanced spiking count at k = 50)
LAM, ETA_G, ETA_MU, K_LEARN, FRAMES = 0.002, 2e-4, 2.0, 50, 3000
MODES = {"branching": np.inf, "leak0.25": 0.25, "leak0.125": 0.125}
WINDOWS = (5, 10, 20, 40, 80, 160, 320, 640, 1280)
rng = np.random.default_rng(SEED)


def walk_check(W):
    out = {}
    for n in (5, 10, 20, 40):
        errs = [np.abs(core.flow(W, v, ALPHA, n) - core.flow(W, v, ALPHA)).sum()
                for v in (core.stimulus(o, m, roles) for o in rng.random(32))]
        out[n] = {"max_L1": max(errs), "bound": 2 * (1 - ALPHA) ** n, "ok": max(errs) <= 2 * (1 - ALPHA) ** n}
    return out


def p_exact(W, v):
    return core.module_stats(W, core.flow(W, v, ALPHA), m, K)[0]


def calibrate(snn, W, leak):
    """Bisect the gain on W_snm so the measured external share of spikes alpha_hat ~ alpha (alpha_hat falls
    with gain). Also returns the spontaneous rate (no input) at that gain."""
    def a_hat(gain):
        sp.set_weights(snn, W, gain)
        runs = [sp.run_frame(snn, core.stimulus(rng.random(), m, roles), 400, RATE, rng) for _ in range(20)]
        return sum(e for _, e in runs) / sum(S.sum() for S, _ in runs)
    lo, hi = 0.05, 1.5
    for _ in range(10):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if a_hat(mid) > ALPHA else (lo, mid)
    gain = (lo + hi) / 2
    a = a_hat(gain)
    spont = sp.run_frame(snn, np.zeros(len(m)), 1000, RATE, rng)[0].mean()
    return gain, a, float(spont)


def spiking_error(snn, W, leak):
    rows = []
    for k in WINDOWS:
        e_mod, e_neu, t0 = [], [], time.perf_counter()
        for _ in range(40):
            v = core.stimulus(rng.random(), m, roles)
            S, _ = sp.run_frame(snn, v, k, RATE, rng)
            ph = sp.pi_hat(S)
            e_mod.append(np.abs(np.bincount(m, ph, K) - p_exact(W, v)).sum())
            e_neu.append(np.abs(ph - core.flow(W, v, ALPHA)).sum())
        rows.append({"k": k, "k_over_tau_alpha": k * ALPHA / sp.tau(leak), "module_L1": np.mean(e_mod),
                     "module_L1_sd": np.std(e_mod), "neuron_L1": np.mean(e_neu),
                     "ms_per_frame": (time.perf_counter() - t0) / 40 * 1e3})
    return rows


def ols_r2(y, *xs):
    X = np.column_stack([np.ones_like(y), *xs])
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))


def pairing_regression(snn, W, frames=300, k=100, lags=1):
    """Accumulate lag-1 pairing counts and predictors over frames with random stimuli; regress over synapses."""
    Csum, Vsum, Jsum, RR, counts, act_silent = 0, 0, 0, 0, 0, 0
    for _ in range(frames):
        v = core.stimulus(rng.random(), m, roles)
        S, _ = sp.run_frame(snn, v, k, RATE, rng)
        c = S.sum(0).astype(float)
        caus, _, cov = sp.pairings(S, lags)
        Csum, Vsum = Csum + caus, Vsum + cov
        Jsum = Jsum + W / W.sum(0) * core.flow(W, v, ALPHA) * c.sum()  # expected causal ~ J_ij x spikes
        RR = RR + np.outer(c, c) / k  # chance coincidences of independent trains
        counts = counts + c
        act_silent += S[:, roles[m] == "A"].sum() == 0
    s = MASK
    caus, bal, cov, J, JT, rr = Csum[s], (Csum - Csum.T)[s], Vsum[s], Jsum[s], Jsum.T[s], RR[s]
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    rate = counts / (frames * k)
    return {
        "lags": lags,
        "causal": {"r_J": corr(caus, J), "r_rr": corr(caus, rr), "R2_J+rr": ols_r2(caus, J, rr)},
        "balanced": {"r_J": corr(bal, J), "r_rr": corr(bal, rr), "r_J-JT": corr(bal, J - JT),
                     "R2_J+rr": ols_r2(bal, J, rr), "R2_J+JT": ols_r2(bal, J, JT)},
        "covariance": {"r_J": corr(cov, J), "r_rr": corr(cov, rr), "R2_J+rr": ols_r2(cov, J, rr),
                       "sum_ratio_to_J": float(cov.sum() / J.sum())},
        "acausal_over_causal": float(Csum.T[s].sum() / caus.sum()),
        "reciprocal_synapse_frac": float((MASK & MASK.T)[s].mean()),
        "rates": {"mean": float(rate.mean()), "max": float(rate.max()), "silent_neuron_frac": float((counts == 0).mean()),
                  "frames_no_action_spike_frac": act_silent / frames},
    }


def evaluate(W):
    rows = []
    for v in TEST:
        pi = core.flow(W, v, ALPHA)
        p, q, JK, _ = core.module_stats(W, pi, m, K)
        rows.append([core.map_equation(p, q, pi), q[C], p[roles == "A"].sum(), core.j_route(JK, roles), core.mstar(p, q)[C]])
    return dict(zip(("D", "q_C", "p_A", "J_route", "Mstar_C"), np.mean(rows, 0).tolist()))


def learn(kind, routing, W, mask, snn=None, gain=None, q_star=None, count="covariance", presyn_norm=True):
    """Structural term only, random controller stimuli each frame. kind: expected | sampled | spiking | additiveG.
    Spiking: count is causal | balanced | covariance pairings; modulator from spike-estimated flow."""
    mu, hist, rates, t0 = 0.0, [], [], time.perf_counter()
    for f in range(FRAMES + 1):
        if f % 100 == 0:
            hist.append({"frame": f, **evaluate(W), "mu": mu})
        if f == FRAMES:
            break
        v = core.stimulus(rng.random(), m, roles)
        if kind == "additiveG":
            W = core.additive_G_update(W, mask, m, roles, ETA_G, routing)
            continue
        if kind == "spiking":
            sp.set_weights(snn, W, gain, presyn_norm)
            S, _ = sp.run_frame(snn, v, K_LEARN, RATE, rng)
            pi, kappa = sp.pi_hat(S), sp.pairings(S)[("causal", "balanced", "covariance").index(count)]
            rates.append(S.mean(0))
        else:
            pi = core.flow(W, v, ALPHA, N_HOPS)
            J = W / W.sum(0) * pi
            kappa = P * J if kind == "expected" else core.sample_pairings(J, P, rng)
        p, q, JK, _ = core.module_stats(W, pi, m, K)
        W = core.structural_update(W, mask, core.modulator(p, q, roles, routing=routing, mu=mu), kappa, m, LAM)
        if routing == "dual":
            mu = max(0.0, mu + ETA_MU * (q_star - core.j_route(JK, roles)))
    d = W.sum(0)
    if kind == "spiking":
        kind = f"spiking_{count}" + ("" if presyn_norm else "_rawW")
    out = {"kind": kind, "routing": routing, "start": hist[0], "end": hist[-1], "curve": hist,
           "d_min": float(d.min()), "d_max": float(d.max()), "ms_per_frame": (time.perf_counter() - t0) / FRAMES * 1e3}
    if rates:
        r = np.array(rates)
        out["rates"] = {"mean": float(r.mean()), "max_neuron": float(r.mean(0).max()),
                        "last500_mean": float(r[-500:].mean()), "silent_neuron_frac": float((r.sum(0) == 0).mean())}
    return W, out


def main():
    res = {"params": {"N": len(m), "K": K, "roles": "".join(roles), "synapses": int(MASK.sum()), "alpha": ALPHA,
                      "n": N_HOPS, "rate": RATE, "P": P, "lam": LAM, "eta_G": ETA_G, "eta_mu": ETA_MU,
                      "k_learn": K_LEARN, "frames": FRAMES, "seed": SEED}}
    res["1_walk"] = walk_check(W0)

    res["2_spiking_error"], res["4_pairings"], gains = {}, {}, {}
    for name, leak in MODES.items():
        snn = sp.build(W0, 1 - ALPHA, leak)
        sp.run_frame(snn, TEST[0], 5, RATE, rng)  # jit warm-up
        gain, a_hat, spont = calibrate(snn, W0, leak)
        gains[name] = gain
        res["2_spiking_error"][name] = {"tau": sp.tau(leak), "gain": gain, "alpha_hat": a_hat, "spont_rate": spont,
                                        "rows": spiking_error(snn, W0, leak)}
        res["4_pairings"][name] = pairing_regression(snn, W0)  # lag 1; lags = tau was worse (more chance pairs)

    res["3_learning"] = []
    q_star = 0.75 * evaluate(W0)["J_route"]
    for kind, routing in [("expected", "none"), ("expected", "exempt"), ("expected", "dual"), ("sampled", "none"),
                          ("sampled", "exempt"), ("additiveG", "none"), ("additiveG", "exempt")]:
        W, out = learn(kind, routing, W0.copy(), MASK, q_star=q_star)
        res["3_learning"].append(out)
        if (kind, routing) == ("expected", "none"):
            res["1_walk_trained"] = walk_check(W)
    snn = sp.build(W0, gains["branching"])
    for count, routing, norm in [("covariance", "none", True), ("covariance", "exempt", True), ("causal", "none", True),
                                 ("causal", "exempt", True), ("balanced", "none", True), ("covariance", "none", False)]:
        res["3_learning"].append(learn("spiking", routing, W0.copy(), MASK, snn, gains["branching"], count=count,
                                       presyn_norm=norm)[1])
    Wd, _, _, mask_d = core.build_network(p_cc=0.5, seed=SEED)  # dense controller recurrence: can it seal?
    for routing in ("none", "exempt"):
        out = learn("expected", routing, Wd, mask_d)[1]
        out["kind"] = "expected_dense_ctrl"
        res["3_learning"].append(out)

    snn = sp.build(W0, gains["branching"])
    res["6_timing_ms_per_frame"] = {}
    for use in ("jit", "cpu"):
        for k in (50, 200):
            t0 = time.perf_counter()
            for _ in range(100):
                sp.run_frame(snn, TEST[0], k, RATE, rng, use=use)
            res["6_timing_ms_per_frame"][f"{use}_k{k}"] = (time.perf_counter() - t0) * 10

    out = pathlib.Path(__file__).parent / "results" / "m1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    summary(res)


def summary(res):
    print("1. walk L1 error vs bound 2(1-a)^n (init / trained):")
    for n, r in res["1_walk"].items():
        print(f"   n={n:>2}: {r['max_L1']:.2e} / {res['1_walk_trained'][n]['max_L1']:.2e}  bound {r['bound']:.2e}")
    print("2. spiking module-level L1 vs window k (k in units of tau/alpha):")
    for name, r in res["2_spiking_error"].items():
        print(f"   {name}: tau={r['tau']:.0f} gain={r['gain']:.3f} alpha_hat={r['alpha_hat']:.3f} spont={r['spont_rate']:.4f}")
        print("     " + "  ".join(f"k{x['k']}({x['k_over_tau_alpha']:.1f}):{x['module_L1']:.3f}" for x in r["rows"]))
    print("3. learning (D bits | q_C | p_A | J_route | M*_C), start -> end:")
    for r in res["3_learning"]:
        s, e = r["start"], r["end"]
        print(f"   {r['kind']:>26} {r['routing']:>6}: D {s['D']:.3f}->{e['D']:.3f} q_C {s['q_C']:.3f}->{e['q_C']:.3f} "
              f"p_A {s['p_A']:.3f}->{e['p_A']:.3f} J_route {e['J_route']:.3f} M*_C {e['Mstar_C']:.2f} mu {e['mu']:.2f} "
              f"d [{r['d_min']:.2f},{r['d_max']:.2f}] {r['ms_per_frame']:.2f} ms/f" + (f" rate {r['rates']}" if "rates" in r else ""))
    print("4/5. pairing regression over synapses, rates:")
    for name, r in res["4_pairings"].items():
        print(f"   {name} (lags {r['lags']}): causal {r['causal']}\n      balanced {r['balanced']}\n      covariance {r['covariance']}\n      acausal/causal "
              f"{r['acausal_over_causal']:.3f} reciprocal {r['reciprocal_synapse_frac']:.2f} rates {r['rates']}")
    print("6. ms per frame:", {k: round(v, 2) for k, v in res["6_timing_ms_per_frame"].items()})


if __name__ == "__main__":
    main()
