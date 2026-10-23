"""Iteration 013 step 1: the feed-forward TD-STDP actor-critic of Chung & Kozma 2020 [CK] (arXiv:2008.13044,
mapstdp/tdstdp.py) in closed loop on CartPole and LunarLander, on the branching engine. Run from the repo root:
  python -m experiments.tdstdp --task {CartPole,LunarLander} --seed 0 [--adam | --no-adam | --sgd] [--heads H]
    [--block B] [--episodes E] [--rate0 p] [--engine {snm,lif,lif-snm}]
    -> experiments/results/tdstdp_<task>[_<tokens>]_s<seed>.json   (tokens: options off the task's preset, in
       order adam | noadam | sgd, B<B>, h<H>, ep<E>, r<rate0>, engine snm | lif | lifsnm)
  python -m experiments.tdstdp --summary   -> per group: T_f, T_s, solved fraction, last-100 return, ms per frame,
       next to CK's Table 2 and the model v4 (flow, spiking) returns
PRESETS hold CK Table 1 and §4 in CK units (CK Eq. 1: a spike through w moves the membrane by R w / tau, Eq. 44).
Our weights are spike probabilities per input spike, so eta, the Adam step and 1 / c_w are divided by
MV = (theta - E_res) tau / R = 13 mV x 100 ms / 1 Ohm = 1300: a high-rate CK neuron (leak << input) fires once per
1300 units of summed input, so d rate / dw = f / 1300 there against f here (iteration 013, decided before E1).
Episode loop per head: warm-up of 100 ms (5 env-step ticks; no learning, no action, initial observation), then per
env step (20 ms tick): resample the action every `resample` steps from the current softmax, step the env, present the
pre-step observation for 20 ms with the scaled reward spread evenly over its ms; V := 0 as the target in the last
2 ms of the episode (termination or truncation, as CK). Heads (batch, CK §4.2): H env copies on shared W, each runs
its own episodes; the update is averaged over learning heads; episode e's return is the mean of the copies' e-th
returns (a copy stops after `episodes`). Block B: W frozen for B ms, the per-ms increments summed (tdstdp.block_learn);
Adam then steps once per block with per-ms-matched moments (tdstdp.Adam).
--sgd (E4): no Adam; eta per population = Adam's per-ms step / RMS of the per-ms update, measured per ms with frozen
initial W over CAL_TICKS ticks of a calibration run on its own rng stream (calibrate)."""
import argparse
import dataclasses
import glob
import json
import pathlib
import re
import time

import gymnasium as gym
import numpy as np

from experiments.tasks import TASKS, critic_layer, random_states
from mapstdp import tdstdp as td

OUT = pathlib.Path(__file__).parent / "results"
TDSTDP_SEED, CAL_SEED, CAL_TICKS, FBAR_STATES, MV, MS, WARM_TICKS = 1302, 1303, 200, 5000, 1300.0, 20, 5
PAPER = {"CartPole": dict(T_f=(57.0, 23.97), T_s=(169.5, 23.47)),
         "LunarLander": dict(T_f=(383.8, 71.49), T_s=(2575.2, 666.51))}  # CK Table 2 (10 runs)
SOLVE = {"CartPole": (500.0, "all"), "LunarLander": (200.0, "mean")}  # perfect score; solved = last 100 all / mean
M_THR = {"CartPole": 475.0, "LunarLander": 200.0}  # secondary: trailing-100 mean threshold (summary only)


@dataclasses.dataclass(frozen=True)
class Config:
    task: str = "CartPole"
    seed: int = 0
    episodes: int = 400
    adam: bool = False
    sgd: bool = False  # E4: SGD with eta from the Adam-equivalent step at init
    heads: int = 1
    block: int = 1
    engine: str = "numpy"  # numpy | snm (SuperNeuroMAT, slow; smoke runs only) | lif | lif-snm (true-LIF control)
    spec: str = "f2c4hb"  # tasks.critic_layer Fourier spec of the input features
    in_rep: int = 1  # input neurons per feature
    n_ce: int = 40
    n_ci: int = 0
    n_ae: int = 20
    n_ai: int = 0
    eta_c: float = 2.5e-3  # CK units (mV); with adam: the Adam step
    eta_a: float = 1e-2
    alpha_c: float = 2.0
    beta_c: float = -0.2
    alpha_a: float = 25.0
    tau_n: float = 20.0
    tau_p: float = 20.0
    tau_z: float = 20.0
    tau_q: float = 40.0
    c_e: float = 0.0
    c_w: float = 0.0  # CK units (1 / mV)
    c_t: float = 0.0
    rho_hat: float = 0.0
    tau_d: float = 1000.0  # discount time constant (ms)
    r_scale: float = 0.02
    resample: int = 1  # env steps per action sample
    adam_b1: float = 0.995
    adam_b2: float = 0.99995
    rate0: float = 0.2  # initial spike probability per ms (pre-registered weight init)


PRESETS = {
    "CartPole": {},
    "LunarLander": dict(episodes=3000, adam=True, heads=16, block=20, spec="f1c1b", in_rep=16, n_ce=128, n_ci=128,
                        n_ae=32, n_ai=32, eta_c=1.25e-4, eta_a=6.25e-5, alpha_c=4.0, beta_c=-2.0, alpha_a=15.0,
                        tau_q=20.0, c_e=1e-4, c_w=1e-8, tau_d=2000.0, r_scale=0.012, resample=2),
}


def preset(task, **kw):
    return dataclasses.replace(Config(task=task, **PRESETS[task]), **kw)


def tokens(cfg):
    p = preset(cfg.task)
    t = ["sgd"] if cfg.sgd else ["adam"] if cfg.adam and not p.adam else ["noadam"] if p.adam and not cfg.adam else []
    t += [f"B{cfg.block}"] if cfg.block != p.block else []
    t += [f"h{cfg.heads}"] if cfg.heads != p.heads else []
    t += [f"ep{cfg.episodes}"] if cfg.episodes != p.episodes else []
    t += [f"r{cfg.rate0:g}"] if cfg.rate0 != p.rate0 else []
    return t + ([cfg.engine.replace("-", "")] if cfg.engine != "numpy" else [])


def name(cfg):
    return "_".join(["tdstdp", cfg.task, *tokens(cfg)]) + f"_s{cfg.seed}"


_FBAR = {}


def features(cfg):
    """(phi, fbar): input rates phi(o) (each feature repeated in_rep times) and their mean over FBAR_STATES states a
    random policy visits (encoder seed of tasks.py), used by the weight init only."""
    task = TASKS[cfg.task]
    base = critic_layer(task, cfg.spec)
    phi = lambda o: np.repeat(base(o), cfg.in_rep)  # noqa: E731
    key = (cfg.task, cfg.spec, cfg.in_rep)
    if key not in _FBAR:
        _FBAR[key] = np.mean([phi(o) for o in random_states(task.env_id, task.n_actions, FBAR_STATES, 777)], 0)
    return phi, _FBAR[key]


class Runner:
    """H env copies on shared W (Layout of mapstdp.tdstdp), stepped in 20 ms ticks."""

    def __init__(self, cfg, rng_seed=TDSTDP_SEED, frozen=False):
        self.cfg, self.frozen = cfg, frozen
        task = TASKS[cfg.task]
        self.K, self.H = task.n_actions, cfg.heads
        self.phi, fbar = features(cfg)
        self.lay = lay = td.Layout(len(fbar), cfg.n_ce, cfg.n_ci, cfg.n_ae, cfg.n_ai, self.K)
        self.rng = np.random.default_rng((cfg.seed, rng_seed))
        self.lif = cfg.engine.startswith("lif")  # true-LIF control: W in CK units, Table 1 values verbatim
        self.mv = 1.0 if self.lif else MV
        self.W = td.init_weights(lay, fbar, cfg.rate0, self.rng, td.THETA_L * td.TAU_M if self.lif else 1.0)
        nc, H = lay.n_crit, self.H
        self.eta = np.r_[np.full(nc, cfg.eta_c), np.full(lay.n_post - nc, cfg.eta_a)][:, None] / self.mv
        self.c_w = cfg.c_w * self.mv
        self.v = np.zeros((H, lay.n_post))  # LIF membranes
        self.opt = td.Adam(self.W.shape, cfg.adam_b1, cfg.adam_b2) if cfg.adam else None
        self.envs = [gym.make(task.env_id) for _ in range(H)]
        self.obs = [None] * H
        self.ret, self.steps, self.warm, self.action = np.zeros(H), np.zeros(H, int), np.zeros(H, int), np.zeros(H, int)
        self.returns, self.done = [[] for _ in range(H)], np.zeros(H, bool)
        self.z, self.q = np.zeros((H, lay.n_post, lay.n_in)), np.zeros((H, lay.n_post - nc, lay.n_in))
        self.P, self.xprev, self.rho = np.zeros((H, lay.n_in)), np.zeros((H, lay.n_in)), np.zeros((H, lay.n_post))
        self.Vprev, self.rprev = np.zeros(H), np.zeros(H)
        self.zprev, self.cact = np.zeros(H, bool), np.zeros(H, bool)
        self.s_last = np.full((H, self.K), 1 / self.K)
        self.frames, self.snn = 0, [None] * H
        self.gsq, self.nupd = np.zeros(2), 0  # calibration: summed squared block update per population
        self.stats = np.zeros(5)  # sums of: critic rate, actor rate, policy entropy, V, ticks
        for h in range(H):
            self.reset(h)

    def reset(self, h):
        self.obs[h] = self.envs[h].reset(seed=int(self.rng.integers(1 << 31)))[0]
        self.ret[h], self.steps[h], self.warm[h] = 0.0, 0, WARM_TICKS
        for a in (self.z, self.q, self.P, self.xprev, self.rho, self.v):
            a[h] = 0
        self.cact[h] = False

    def tick(self):
        cfg, lay, H, K = self.cfg, self.lay, self.H, self.K
        f, r_ms = np.zeros((H, lay.n_in)), np.zeros(H)
        final, learn, A = np.zeros(H, bool), np.zeros(H, bool), np.zeros((H, K))
        for h in range(H):
            if self.done[h]:
                continue
            f[h] = self.phi(self.obs[h])
            if self.warm[h]:
                self.warm[h] -= 1
                continue
            if self.steps[h] % cfg.resample == 0:
                self.action[h] = self.rng.choice(K, p=self.s_last[h])
            o2, r, term, trunc, _ = self.envs[h].step(int(self.action[h]))
            r_ms[h], final[h], learn[h], A[h, self.action[h]] = cfg.r_scale * r / MS, term or trunc, True, 1
            self.ret[h] += r
            self.steps[h] += 1
            self.obs[h] = o2
        self.frames += int(learn.sum())
        for b in range(MS // cfg.block):
            self.learn_block(f, r_ms, final, learn, A, np.arange(b * cfg.block, (b + 1) * cfg.block))
        for h in np.flatnonzero(final):
            self.returns[h].append(float(self.ret[h]))
            if len(self.returns[h]) >= cfg.episodes:
                self.done[h] = True
            else:
                self.reset(h)

    def learn_block(self, f, r_ms, final, learn, A, ms):
        cfg, lay, L, nc = self.cfg, self.lay, len(ms), self.lay.n_crit
        x, noise = td.draw(f, L, lay.n_post, self.rng)
        if self.lif:
            X, self.v = td.simulate_lif(self.W, self.xprev, x, self.v, "exp" if cfg.engine == "lif" else "snm")
        elif cfg.engine == "snm":
            X = np.concatenate([self._snm(h, x[:, h:h + 1], noise[:, h:h + 1]) for h in range(self.H)], 1)
        else:
            X = td.simulate(self.W, self.xprev, x, noise)
        P = td.pre_trace(self.P, x, cfg.tau_p)
        R = td.rates(self.rho, X, cfg.tau_n)
        V = cfg.alpha_c * R @ lay.Mv + cfg.beta_c  # Eq. 4
        s = td.softmax(cfg.alpha_a * R @ lay.Ma.T)  # Eq. 12
        zero = final[None] & (ms >= MS - 2)[:, None]  # target V := 0 in the last 2 ms
        eV, e2 = np.exp(-1 / cfg.tau_d), np.exp(-1 / (2 * cfg.tau_d))
        d = np.zeros((L + 1, self.H))  # Eq. 69 for t = -1..L-1
        d[0] = self.cact * (eV * V[0] * ~self.zprev + e2 * self.rprev - self.Vprev)
        d[1:L] = learn * (eV * V[1:] * ~zero[:-1] + e2 * r_ms - V[:-1])
        d[L] = (learn & zero[-1]) * (e2 * r_ms - V[-1])
        act, sa = lay.act, lay.sign[nc:]
        c = learn[:, None] * (A[:, act] - s[..., act])  # Eq. 77 feedback A_k - s_k
        kap = learn[:, None] * (cfg.c_e * sa * td.entropy_coef(s)[..., act]
                                - cfg.c_t * (R[..., nc:][..., lay.ae].mean(-1, keepdims=True) - cfg.rho_hat) * lay.ae)
        G, self.z, self.q = td.block_learn(self.z, self.q, P, X, d, c, kap, lay, cfg.tau_z, cfg.tau_q)
        n = int((learn | self.cact).sum())
        if n:
            g = G.sum(0) / n - 0.5 * self.c_w * L * self.W  # Eqs. 73, 78, 26, 27, 31
            if self.frozen:
                self.gsq += [(g[:nc] ** 2).mean(), (g[nc:] ** 2).mean()]
                self.nupd += 1
            elif self.opt is not None:
                self.W += self.opt.step(g, self.eta, L)
            else:
                self.W += self.eta * g
        self.xprev, self.P, self.rho, self.s_last = x[-1], P[-1], R[-1], s[-1]
        self.Vprev, self.rprev, self.zprev, self.cact = V[-1], r_ms, zero[-1], learn & ~zero[-1]
        m = learn | self.cact
        if m.any():
            self.stats += [R[..., :nc][:, m].mean(), R[..., nc:][:, m].mean(),
                           -(s[:, m] * np.log(np.maximum(s[:, m], 1e-300))).sum(-1).mean(), V[:, m].mean(), 1]

    def _snm(self, h, x, noise):
        X, self.snn[h] = td.simulate_snm(self.W, self.xprev[h:h + 1], x, noise, self.snn[h])
        return X


def calibrate(cfg):
    """E4 rule (Adam-equivalent step at init): eta per population (critic, actor) = Adam step per ms / RMS of the
    per-ms update (averaged over heads), measured per ms (block 1) with the initial W frozen over CAL_TICKS ticks of a
    run on the (seed, CAL_SEED) stream: an RMS-sized per-ms update moves W by the step per-ms Adam takes at init
    (|m_hat| / sqrt(v_hat) = 1). Returns (eta, rms) in CK units."""
    r = Runner(dataclasses.replace(cfg, adam=False, sgd=False, block=1), rng_seed=CAL_SEED, frozen=True)
    for _ in range(CAL_TICKS):
        r.tick()
    rms = np.sqrt(r.gsq / r.nupd)
    return np.array([cfg.eta_c, cfg.eta_a]) / rms, rms


def solve_times(task, returns_heads):
    """(T_f, T_s, per-episode return): returns_heads is a list of per-copy return lists. T_f: first episode in which
    some copy reaches the perfect score. T_s: first episode e >= 100 at which the last 100 (copy-mean) returns are all
    perfect (CartPole) or average >= 200 (LunarLander). 1-indexed; None if never."""
    perf, rule = SOLVE[task]
    E = min(len(r) for r in returns_heads)
    M = np.array([r[:E] for r in returns_heads])
    ret = M.mean(0)
    hit = np.flatnonzero((M >= perf).any(0))
    Tf = int(hit[0]) + 1 if len(hit) else None
    ok = M.min(0) >= perf if rule == "all" else None
    Ts = None
    for e in range(100, E + 1):
        if (ok[e - 100:e].all() if rule == "all" else ret[e - 100:e].mean() >= perf):
            Ts = e
            break
    return Tf, Ts, ret.tolist()


def run(cfg, log_every=50):
    eta_sgd = None
    if cfg.sgd:
        eta_sgd, rms = calibrate(cfg)
        cfg = dataclasses.replace(cfg, eta_c=float(eta_sgd[0]), eta_a=float(eta_sgd[1]))
        print(f"sgd calibration: per-ms update RMS {rms}, eta (CK units) {eta_sgd}", flush=True)
    r, t0, nxt = Runner(cfg), time.perf_counter(), log_every
    OUT.mkdir(exist_ok=True)
    fn = OUT / f"{name(cfg)}.json"
    while not r.done.all():
        r.tick()
        e = min(len(x) for x in r.returns)
        if e >= nxt:
            nxt += log_every
            st = r.stats / max(r.stats[4], 1)
            r.stats[:] = 0
            last = np.mean([np.mean(x[-log_every:]) for x in r.returns])
            print(f"ep {e} last{log_every} {last:.1f} rate_c {st[0]:.3f} rate_a {st[1]:.3f} H(s) {st[2]:.3f} "
                  f"V {st[3]:.3f} |W| {np.abs(r.W).mean():.4f} {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            res = result(cfg, r, t0, eta_sgd)
            fn.with_suffix(".partial").write_text(json.dumps(res))
    res = result(cfg, r, t0, eta_sgd)
    fn.write_text(json.dumps(res))
    fn.with_suffix(".partial").unlink(missing_ok=True)
    print(f"T_f {res['T_f']} T_s {res['T_s']} last100 {np.mean(res['returns'][-100:]):.1f} "
          f"ms/frame {res['ms_per_frame']:.2f}")
    return res


def result(cfg, r, t0, eta_sgd):
    Tf, Ts, ret = solve_times(cfg.task, r.returns)
    wall = time.perf_counter() - t0
    out = {"config": dataclasses.asdict(cfg), "returns": ret, "T_f": Tf, "T_s": Ts, "frames": r.frames,
           "ms_per_frame": wall / max(r.frames, 1) * 1e3, "wall_s": wall, "W_abs_mean": float(np.abs(r.W).mean()),
           "W_neg_frac": float((r.W < 0).mean()), "n_in": r.lay.n_in, "mv": r.mv}
    if cfg.heads > 1:
        out["returns_heads"] = r.returns
    if eta_sgd is not None:
        out["eta_sgd_ck_units"] = eta_sgd.tolist()
    return out


def v4_returns(task):
    """Last-100 returns over seeds of the model v4 runs (flow; spiking per heads and variant), from experiments.m3."""
    pat = re.compile(rf"m3_{task}_(flow|spiking_h\d+)_lb0\.3_tn0\.04_pcc0_bk0\.01m16(_\w+?)?_eta5_s\d+")
    out = {}
    for f in sorted(glob.glob(str(OUT / f"m3_{task}_*_eta5_s*.json"))):
        m = pat.fullmatch(pathlib.Path(f).stem)
        if m and (m[1] != "flow" or not m[2]):
            k = "v4 " + m[1].replace("_", " ") + (" " + m[2][1:] if m[2] else "")
            out.setdefault(k, []).append(np.mean(json.loads(pathlib.Path(f).read_text())["returns"][-100:]))
    return out


def trailing_mean_time(ret, thr):
    """First episode e >= 100 whose trailing-100 mean return reaches thr (1-indexed), or None."""
    c = np.cumsum(np.r_[0.0, ret])
    hit = np.flatnonzero((c[100:] - c[:-100]) / 100 >= thr)
    return int(hit[0]) + 100 if len(hit) else None


def summary():
    from scipy.stats import mannwhitneyu
    print("T_f / T_s (CK definitions, pre-registered): mean +- sd over runs that reach it [n]; median T_s counts "
          "unsolved as infinite. Secondary: T_m = first episode whose trailing-100 mean reaches M_THR (CartPole 475, "
          "Gymnasium's threshold; LunarLander 200 = T_s). ret: last-100 return, mean +- sd over seeds. MW: "
          "Mann-Whitney p against the task preset, on T_s (unsolved = episodes + 1) / on ret.")
    groups = {}
    for f in sorted(glob.glob(str(OUT / "tdstdp_*.json"))):
        groups.setdefault(pathlib.Path(f).stem.rsplit("_s", 1)[0], []).append(json.loads(pathlib.Path(f).read_text()))
    for task in PAPER:
        gs = {g: rs for g, rs in groups.items() if g.split("_")[1] == task}
        if not gs:
            continue
        p = PAPER[task]
        print(f"\n{task}: CK 2020 T_f {p['T_f'][0]:.1f} +- {p['T_f'][1]:.1f}  "
              f"T_s {p['T_s'][0]:.1f} +- {p['T_s'][1]:.1f}")
        for k, R in v4_returns(task).items():
            print(f"  {k:28} n={len(R):2d} ret {np.mean(R):7.1f} +- {np.std(R):5.1f}")
        base = f"tdstdp_{task}"
        ms = lambda x: f"{np.mean(x):6.1f} +- {np.std(x):5.1f}" if x else "   nan          "  # noqa: E731
        for g, rs in sorted(gs.items(), key=lambda kv: (kv[0] != base, kv[0])):
            Tf = [r["T_f"] for r in rs if r["T_f"] is not None]
            Ts = [r["T_s"] for r in rs if r["T_s"] is not None]
            Tm = [t for t in (trailing_mean_time(r["returns"], M_THR[task]) for r in rs) if t is not None]
            med = np.median([r["T_s"] if r["T_s"] is not None else np.inf for r in rs])
            R = [np.mean(r["returns"][-100:]) for r in rs]
            line = (f"  {g:28} n={len(rs):2d} solved {len(Ts):2d}/{len(rs):2d}  T_f {ms(Tf)} [{len(Tf):2d}]  "
                    f"T_s {ms(Ts)} median {med:5.0f}  T_m {ms(Tm)} [{len(Tm):2d}]  ret {np.mean(R):6.1f} +- "
                    f"{np.std(R):5.1f}  ms/fr {np.mean([r['ms_per_frame'] for r in rs]):5.2f}")
            if g != base and base in gs:
                cens = lambda rr: [r["T_s"] or r["config"]["episodes"] + 1 for r in rr]  # noqa: E731
                R0 = [np.mean(r["returns"][-100:]) for r in gs[base]]
                line += (f"  MW {mannwhitneyu(cens(rs), cens(gs[base])).pvalue:.2g} / "
                         f"{mannwhitneyu(R, R0).pvalue:.2g}")
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(PRESETS), default="CartPole")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--adam", action="store_true", default=None)
    g.add_argument("--no-adam", dest="adam", action="store_false")
    g.add_argument("--sgd", action="store_true", help="E4: no Adam, eta from the Adam-equivalent step at init")
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--rate0", type=float, default=None, help="initial spike probability per ms (post-hoc diagnostic)")
    ap.add_argument("--engine", choices=("numpy", "snm", "lif", "lif-snm"), default="numpy",
                    help="lif: CK Eq. 1 neuron (true-LIF control); lif-snm: with SuperNeuroMAT's subtractive leak")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    kw = {k: v for k, v in (("episodes", a.episodes), ("heads", a.heads), ("block", a.block), ("rate0", a.rate0))
          if v is not None}
    if a.sgd:
        kw.update(adam=False, sgd=True)
    elif a.adam is not None:
        kw["adam"] = a.adam
    run(preset(a.task, seed=a.seed, engine=a.engine, **kw))


if __name__ == "__main__":
    main()
