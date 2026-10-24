"""Iteration 013 step 1: the feed-forward TD-STDP actor-critic of Chung & Kozma 2020 [CK] (arXiv:2008.13044,
mapstdp/tdstdp.py) in closed loop on CartPole and LunarLander. Run from the repo root:
  python -m experiments.tdstdp --task {CartPole,LunarLander} --seed 0 [--adam | --no-adam | --sgd] [--heads H]
    [--block B] [--episodes E] [--engine {numpy,snm,lif}] [--legacy [--rate0 p] [--engine lif-snm]]
    -> experiments/results/tdstdp_<task>[_ck][_<tokens>]_s<seed>.json   (tokens: options off the preset, in order
       adam | noadam | sgd, B<B>, h<H>, ep<E>, r<rate0> (legacy), engine snm | lif | lifsnm)
  python -m experiments.tdstdp --summary   -> per group: T_f, T_s, solved fraction, last-100 return, ms per frame,
       next to CK's Table 2 and the model v4 (flow, spiking) returns
Default (token ck): the authors' code, reference/td_stdp (commit a5cb7f4, BindsNET 0.2.7), where it disagrees with
the paper (iteration 013 §8). PRESETS are config/config_cp.ini and config_ll.ini (eta in mV); weights convert to
branching units by MV = 13 (a BindsNET spike through w adds w mV against a 13 mV gap), engine lif is BindsNET's
LIF with W in mV. CKRunner transcribes main.py, bindsnet_add.Snn_actor / Reward_fn and util.batch_envs: the action
is sampled from the response to the current observation, p is fixed at its sampling probability, V' := 0 only on true
termination, 6 presentations of s_0 per episode, the update is the mean over all copies, and episodes are pooled
over copies (min_eps = episodes, max_eps). Block B: W frozen for B ms (tdstdp.block_learn_ck), Adam then steps once
per block with per-ms-matched moments (tdstdp.Adam); the authors step per ms (B = 1).
--fast (iteration 013 §10, S2): the update summed over copies in one matrix product (tdstdp.block_learn_ck_fast);
same rng order, equal up to floating-point summation order (not bit-identical). --engine flow (S3): the branching
step's noise-free limit (tdstdp.flow_step): rates instead of spikes, plus the input autocorrelation term of z."""
import argparse
import dataclasses
import glob
import itertools
import json
import pathlib
import re
import time

import gymnasium as gym
import numpy as np

from experiments.tasks import TASKS, critic_layer, random_states
from mapstdp import tdstdp as td

OUT = pathlib.Path(__file__).parent / "results"
TDSTDP_SEED, CAL_SEED, CAL_TICKS, FBAR_STATES, MV_LEGACY, MS, WARM_TICKS = 1302, 1303, 200, 5000, 1300.0, 20, 5
MV, WARM_N = td.THETA_L, 5  # authors' code: mV per unit of branching weight; batch_envs warm_n (windows)
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
    rate0: float = 0.2  # initial spike probability per ms (pre-registered weight init; legacy)
    # authors' code (ck = True); the defaults above are the legacy CartPole configuration
    ck: bool = False
    max_eps: int = 0  # pooled episodes cap (main.py max_eps); episodes is min_eps
    init_scale: float = 0.1  # W ~ U(0, init_scale) mV
    freeze: int = 20  # ms between action samples (freeze_action)
    f_order: int = 2  # Fourier encoder (util.State_to_spike_Fourier): order, cross terms, ON/OFF split, constant unit
    f_cross: bool = True
    f_double: bool = False
    f_basis: bool = False
    adam_eps: float = 1e-8
    fast: bool = False  # authors' code: update summed over copies in one product (block_learn_ck_fast; §10 S2)


PRESETS = {  # the authors' configs, reference/td_stdp/config/config_{cp,ll}.ini (eta in mV, before trace scaling)
    "CartPole": dict(ck=True, episodes=400, max_eps=400, eta_c=0.005, eta_a=0.01, alpha_a=50.0, init_scale=0.1,
                     adam_eps=1e-9),
    "LunarLander": dict(ck=True, episodes=3000, max_eps=10000, adam=True, heads=16, block=20, in_rep=16, f_order=1,
                        f_cross=False, f_double=True, f_basis=True, n_ce=128, n_ci=128, n_ae=32, n_ai=32,
                        eta_c=1.25e-4, eta_a=6.25e-5, init_scale=0.125, alpha_c=4.0, beta_c=-2.0, alpha_a=15.0,
                        tau_q=20.0, c_e=1e-4, tau_d=2000.0, r_scale=0.012, freeze=40, adam_eps=1e-9),
}  # LunarLander: weight_reg = 1e-8 is never passed to the rule (main.py), so c_w = 0; block 20 is ours (B = 1 theirs)
CK_BOUNDS = {"CartPole": ([-2.4, -3.2, -np.pi / 12, -3.2], [2.4, 3.2, np.pi / 12, 3.2]),  # main.py state_lim
             "LunarLander": ([-1, -0.2, -1, -1, -1, -1, 0, 0], [1, 2, 1, 1, 1, 1, 1, 1])}
LEGACY = {  # pre-audit (E1, E2, first E3/E4): CK Table 1 and §4 in CK units, divided by MV_LEGACY
    "CartPole": {},
    "LunarLander": dict(episodes=3000, adam=True, heads=16, block=20, spec="f1c1b", in_rep=16, n_ce=128, n_ci=128,
                        n_ae=32, n_ai=32, eta_c=1.25e-4, eta_a=6.25e-5, alpha_c=4.0, beta_c=-2.0, alpha_a=15.0,
                        tau_q=20.0, c_e=1e-4, c_w=1e-8, tau_d=2000.0, r_scale=0.012, resample=2),
}


def preset(task, legacy=False, **kw):
    return dataclasses.replace(Config(task=task, **(LEGACY if legacy else PRESETS)[task]), **kw)


def tokens(cfg):
    p = preset(cfg.task, legacy=not cfg.ck)
    t = ["ck"] if cfg.ck else []
    t += ["sgd"] if cfg.sgd else ["adam"] if cfg.adam and not p.adam else ["noadam"] if p.adam and not cfg.adam else []
    t += [f"B{cfg.block}"] if cfg.block != p.block else []
    t += [f"h{cfg.heads}"] if cfg.heads != p.heads else []
    t += [f"ep{cfg.episodes}"] if cfg.episodes != p.episodes else []
    t += [f"mx{cfg.max_eps}"] if cfg.max_eps != p.max_eps else []
    t += [f"r{cfg.rate0:g}"] if cfg.rate0 != p.rate0 else []
    t += ["fast"] if cfg.fast else []
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
    """Legacy (pre-audit) loop: H env copies on shared W (Layout of mapstdp.tdstdp), stepped in 20 ms ticks."""

    def __init__(self, cfg, rng_seed=TDSTDP_SEED, frozen=False):
        self.cfg, self.frozen = cfg, frozen
        task = TASKS[cfg.task]
        self.K, self.H = task.n_actions, cfg.heads
        self.phi, fbar = features(cfg)
        self.lay = lay = td.Layout(len(fbar), cfg.n_ce, cfg.n_ci, cfg.n_ae, cfg.n_ai, self.K)
        self.rng = np.random.default_rng((cfg.seed, rng_seed))
        self.lif = cfg.engine.startswith("lif")  # true-LIF control: W in CK units, Table 1 values verbatim
        self.mv = 1.0 if self.lif else MV_LEGACY
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
    """Legacy E4 rule (Adam-equivalent step at init): eta per population (critic, actor) = Adam step per ms / RMS of the
    per-ms update (averaged over heads), measured per ms (block 1) with the initial W frozen over CAL_TICKS ticks of a
    run on the (seed, CAL_SEED) stream: an RMS-sized per-ms update moves W by the step per-ms Adam takes at init
    (|m_hat| / sqrt(v_hat) = 1). Returns (eta, rms) in CK units."""
    r = Runner(dataclasses.replace(cfg, adam=False, sgd=False, block=1), rng_seed=CAL_SEED, frozen=True)
    for _ in range(CAL_TICKS):
        r.tick()
    rms = np.sqrt(r.gsq / r.nupd)
    return np.array([cfg.eta_c, cfg.eta_a]) / rms, rms


def ck_encoder(cfg):
    """util.State_to_spike_Fourier: the observation scaled to [0, 1] by CK_BOUNDS (clipped), coefficient vectors
    {0..order}^d (cross terms, the constant included) or order 1..f_order per dimension; (1 + cos) / 2 or the ON/OFF
    split ([cos]+, [-cos]+); a constant unit if f_basis; the whole vector repeated in_rep times. Returns
    (phi: H x d observations -> H x n_in input rates, n_in)."""
    lo, hi = (np.array(v, float) for v in CK_BOUNDS[cfg.task])
    n, k = len(lo), cfg.f_order
    if cfg.f_cross:
        C = np.array(list(itertools.product(range(k + 1), repeat=n)), float)[:, ::-1].T  # column i: digits of i
    else:
        C = np.zeros((n, n * k))
        for j, m in itertools.product(range(n), range(k)):
            C[j, j * k + m] = m + 1

    def phi(o):
        c = np.cos(np.pi * np.clip((np.asarray(o, float) - lo) / (hi - lo), 0, 1) @ C)
        f = np.concatenate([np.maximum(c, 0), np.maximum(-c, 0)], -1) if cfg.f_double else (1 + c) / 2
        if cfg.f_basis:
            f = np.concatenate([f, np.ones((*f.shape[:-1], 1))], -1)
        return np.tile(f, cfg.in_rep)
    return phi, C.shape[1] * (2 if cfg.f_double else 1) * cfg.in_rep + cfg.f_basis * cfg.in_rep


class CKRunner:
    """The authors' closed loop (reference/td_stdp: main.py, bindsnet_add.Snn_actor / Reward_fn, util.batch_envs):
    H env copies on shared W in lockstep, one 20 ms window per tick. Per window: copies whose env was just reset
    (code 2) clear traces, rates and membranes (every copy's previous input spikes too, as BindsNET's Input layer
    reset ignores the index); every copy presents its current observation with (A_(k-1), p_(k-1), r_(k-1)); delta = 0
    in reset (2) and warm-up (3) windows; V' := 0 in the last 2 ms only after a true termination (bootstrapped at
    truncation); the entropy term applies in every window; the update is the mean over all H copies (BindsNET's batch
    reduction). Then p = softmax at the window's end, the action is resampled when the copy's timer reaches `freeze`
    ms, and the env copies step: an ended copy resets, then WARM_N warm-up windows, so s_0 is presented 6 times
    (5 + 1 on the first episode, whose first window is live as in main.py). Episodes are pooled over copies in the
    order they end; the run stops when solved with >= `episodes` pooled episodes, or at `max_eps`."""

    def __init__(self, cfg, rng_seed=TDSTDP_SEED, frozen=False):
        assert cfg.ck and cfg.engine in ("numpy", "snm", "lif", "flow") and cfg.c_t == 0 and MS % cfg.block == 0
        self.cfg, self.frozen = cfg, frozen
        task = TASKS[cfg.task]
        self.K, self.H = K, H = task.n_actions, cfg.heads
        self.phi, n_in = ck_encoder(cfg)
        self.lay = lay = td.Layout(n_in, cfg.n_ce, cfg.n_ci, cfg.n_ae, cfg.n_ai, K)
        self.rng = np.random.default_rng((cfg.seed, rng_seed))
        self.lif = cfg.engine == "lif"  # BindsNET LIF, W in mV
        self.mv = 1.0 if self.lif else MV
        self.W = self.rng.uniform(0, cfg.init_scale / self.mv, (lay.n_post, lay.n_in))
        nc, na = lay.n_crit, lay.n_post - lay.n_crit
        self.nu = np.r_[np.full(nc, cfg.eta_c), np.full(na, cfg.eta_a)][:, None]  # mV
        # the code folds nu into the pre trace, so its Adam sees nu g: eps on our unit-trace gradient g is eps / nu
        self.opt = td.Adam(self.W.shape, cfg.adam_b1, cfg.adam_b2, cfg.adam_eps / self.nu) if cfg.adam else None
        self.envs = [gym.make(task.env_id) for _ in range(H)]
        self.obs = np.array([e.reset(seed=int(self.rng.integers(1 << 31)))[0] for e in self.envs], float)
        self.reward, self.is_end, self.trunc = np.zeros(H), np.zeros(H, bool), np.zeros(H, bool)
        self.code, self.warm, self.rest = np.zeros(H, int), np.zeros(H, int), np.zeros(H, int)
        self.p, self.action, self.timer = np.full((H, K), 1 / K), self.rng.choice(K, size=H), np.zeros(H)
        self.z, self.q = np.zeros((H, lay.n_post, lay.n_in)), np.zeros((H, na, lay.n_in))
        self.P, self.xprev = np.zeros((H, lay.n_in)), np.zeros((H, lay.n_in))
        self.rho, self.v, self.Vprev = np.zeros((H, lay.n_post)), np.zeros((H, lay.n_post)), np.zeros(H)
        self.c_ret, self.returns, self.returns_heads = np.zeros(H), [], [[] for _ in range(H)]
        self.done, self.solved, self.frames, self.snn = False, False, 0, [None] * H
        self.gsq, self.nupd = np.zeros(2), 0  # calibration: summed squared per-ms update per population
        self.stats = np.zeros(5)  # sums of: critic rate, actor rate, policy entropy, V, ticks

    def tick(self):
        cfg, lay, K = self.cfg, self.lay, self.K
        rs = self.code == 2
        if rs.any():
            for a in (self.z, self.q, self.P, self.rho, self.v):
                a[rs] = 0
            self.xprev[:] = 0
        act, sa = lay.act, lay.sign[lay.n_crit:]
        c = np.eye(K)[self.action][:, act] - self.p[:, act]  # (A - p) with p fixed at the sampling probability
        kap = cfg.c_e * sa * td.entropy_coef(self.p)[:, act]  # Eq. 26, from the window's p
        f, r_ms = self.phi(self.obs), cfg.r_scale * self.reward / MS
        end, learn = self.is_end & ~self.trunc, self.code < 2
        for b in range(MS // cfg.block):
            self.learn_block(f, r_ms, end, learn, c, kap, np.arange(b * cfg.block, (b + 1) * cfg.block))
        self.p = td.softmax(cfg.alpha_a * self.rho @ lay.Ma.T)
        self.timer += MS
        nb = self.timer >= cfg.freeze
        for h in np.flatnonzero(nb):
            self.action[h] = self.rng.choice(K, p=self.p[h])
        self.timer[nb] = 0
        was_end = self.is_end.copy()  # batch_envs.step (rest_n = 0)
        self.rest[self.is_end] += 1
        self.warm += 1
        for h in np.flatnonzero(self.rest > 0):
            self.obs[h] = self.envs[h].reset(seed=int(self.rng.integers(1 << 31)))[0]
            self.reward[h], self.rest[h], self.warm[h], self.trunc[h], self.is_end[h] = 0, 0, 0, False, False
        live = (self.warm > WARM_N) & ~self.is_end
        self.reward[~live] = 0
        for h in np.flatnonzero(live):
            o, r, term, trunc, _ = self.envs[h].step(int(self.action[h]))
            self.obs[h], self.reward[h], self.is_end[h], self.trunc[h] = o, r, term or trunc, trunc and not term
        self.frames += int(live.sum())
        self.code[live] = 0
        self.code[self.rest >= 1] = 1
        self.code[self.warm <= WARM_N] = 3
        self.code[self.warm == 0] = 2
        self.c_ret += self.reward  # main.py: pooled episode returns
        new = ~was_end & self.is_end
        for h in np.flatnonzero(new):
            self.returns.append(float(self.c_ret[h]))
            self.returns_heads[h].append(float(self.c_ret[h]))
        self.c_ret[new] = 0
        if new.any():
            n = len(self.returns)
            if not self.solved and n > 100:
                self.solved = trailing_mean_time(self.returns, SOLVE[cfg.task][0]) is not None
            self.done = (self.solved and n >= cfg.episodes) or n >= cfg.max_eps

    def learn_block(self, f, r_ms, end, learn, c, kap, ms):
        cfg, lay, L, nc, H = self.cfg, self.lay, len(ms), self.lay.n_crit, self.H
        K = None
        if cfg.engine == "flow":  # noise-free limit: no spike draws; z gets the input autocorrelation term
            x, X, K0 = td.flow_step(self.W, self.xprev, f, L)
            K = np.exp(-1 / cfg.tau_p) * K0
        else:
            x, noise = td.draw(f, L, lay.n_post, self.rng)
            if self.lif:
                X, self.v = td.simulate_bindsnet(self.W, self.xprev, x, self.v)
            elif cfg.engine == "snm":
                X = np.concatenate([Runner._snm(self, h, x[:, h:h + 1], noise[:, h:h + 1]) for h in range(H)], 1)
            else:
                X = td.simulate(self.W, self.xprev, x, noise)
        P = td.pre_trace(self.P, x, cfg.tau_p)
        R = td.rates(self.rho, X, cfg.tau_n)
        V = cfg.alpha_c * R @ lay.Mv + cfg.beta_c  # Eq. 4
        live = (ms < MS - 2)[:, None] | ~end  # Reward_fn.isLive
        Vp = np.concatenate([self.Vprev[None], V[:-1]])
        d = learn * (np.exp(-1 / (2 * cfg.tau_d)) * r_ms + np.exp(-1 / cfg.tau_d) * V * live - Vp)  # Eq. 69
        bc = lambda a: np.broadcast_to(a, (L, *a.shape))  # noqa: E731
        if cfg.fast or K is not None:
            G, self.z, self.q = td.block_learn_ck_fast(self.z, self.q, P, X, d, bc(c), bc(kap), lay, cfg.tau_z,
                                                       cfg.tau_q, K)
            g = G / H
        else:
            G, self.z, self.q = td.block_learn_ck(self.z, self.q, P, X, d, bc(c), bc(kap), lay, cfg.tau_z, cfg.tau_q)
            g = G.mean(0)
        if self.frozen:
            self.gsq += [(g[:nc] ** 2).mean(), (g[nc:] ** 2).mean()]
            self.nupd += 1
        elif self.opt is not None:
            self.W += self.opt.step(g, self.nu / self.mv, L)
        else:
            self.W += self.nu / self.mv * g
        self.xprev, self.P, self.rho, self.Vprev = x[-1], P[-1], R[-1], V[-1]
        if learn.any():
            s = td.softmax(cfg.alpha_a * R[:, learn] @ lay.Ma.T)
            self.stats += [R[..., :nc][:, learn].mean(), R[..., nc:][:, learn].mean(),
                           -(s * np.log(np.maximum(s, 1e-300))).sum(-1).mean(), V[:, learn].mean(), 1]

    def save(self, fn, wall, resumes):
        o = self.opt
        np.savez(fn, W=self.W, m=o.m if o else 0, v=o.v if o else 0, t=o.t if o else 0, frames=self.frames,
                 wall=wall, resumes=resumes, returns=json.dumps([self.returns, self.returns_heads]),
                 rng=json.dumps(self.rng.bit_generator.state))

    def load(self, fn):
        """Resume from a checkpoint: W, Adam moments, pooled returns, rng; every copy starts a fresh episode with
        cleared traces (not bit-identical to an uninterrupted run). Returns (wall, resumes)."""
        c = np.load(fn)
        self.W = c["W"]
        if self.opt is not None:
            self.opt.m, self.opt.v, self.opt.t = c["m"], c["v"], int(c["t"])
        self.returns, self.returns_heads = json.loads(str(c["returns"]))
        self.rng.bit_generator.state = json.loads(str(c["rng"]))
        self.frames = int(c["frames"])
        return float(c["wall"]), int(c["resumes"]) + 1


def calibrate_ck(cfg):
    """E4' rule (the E4 rule on the authors' loop): eta per population (critic, actor) = Adam step per ms / RMS of the
    per-ms update g (mean over all copies, the authors' units), measured per ms (block 1) with the initial W frozen
    over CAL_TICKS ticks after the first WARM_N + 1 (no env step yet) of a run on the (seed, CAL_SEED) stream: an
    RMS-sized per-ms update moves W by the step per-ms Adam takes at init. Returns (eta, rms), eta in mV."""
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=1), rng_seed=CAL_SEED, frozen=True)
    for _ in range(WARM_N + 1):
        r.tick()
    r.gsq[:], r.nupd = 0, 0
    for _ in range(CAL_TICKS):
        r.tick()
    rms = np.sqrt(r.gsq / r.nupd)
    return np.array([cfg.eta_c, cfg.eta_a]) / rms, rms


def run_ck(cfg, log_every=50):
    eta_sgd = None
    if cfg.sgd:
        eta_sgd, rms = calibrate_ck(cfg)
        cfg = dataclasses.replace(cfg, eta_c=float(eta_sgd[0]), eta_a=float(eta_sgd[1]))
        print(f"sgd calibration: per-ms update RMS {rms}, eta (mV) {eta_sgd}", flush=True)
    r, wall0, resumes = CKRunner(cfg), 0.0, 0
    OUT.mkdir(exist_ok=True)
    fn = OUT / f"{name(cfg)}.json"
    ck = fn.with_suffix(".ckpt.npz")
    if ck.exists():
        wall0, resumes = r.load(ck)
        print(f"resumed from {ck.name} at {len(r.returns)} episodes", flush=True)
    t0, nxt = time.perf_counter() - wall0, (len(r.returns) // log_every + 1) * log_every
    while not r.done:
        r.tick()
        e = len(r.returns)
        if e >= nxt:
            nxt = (e // log_every + 1) * log_every
            st = r.stats / max(r.stats[4], 1)
            r.stats[:] = 0
            print(f"ep {e} last{log_every} {np.mean(r.returns[-log_every:]):.1f} rate_c {st[0]:.3f} rate_a "
                  f"{st[1]:.3f} H(s) {st[2]:.3f} V {st[3]:.3f} |W| {np.abs(r.W).mean():.4f} "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            fn.with_suffix(".partial").write_text(json.dumps(result_ck(cfg, r, t0, eta_sgd, resumes)))
            r.save(ck, time.perf_counter() - t0, resumes)
    res = result_ck(cfg, r, t0, eta_sgd, resumes)
    fn.write_text(json.dumps(res))
    fn.with_suffix(".partial").unlink(missing_ok=True)
    ck.unlink(missing_ok=True)
    print(f"T_f {res['T_f']} T_s {res['T_s']} episodes {len(res['returns'])} last100 "
          f"{np.mean(res['returns'][-100:]):.1f} ms/frame {res['ms_per_frame']:.2f}")
    return res


def result_ck(cfg, r, t0, eta_sgd, resumes):
    Tf, Ts = pooled_times(cfg.task, r.returns)
    wall = time.perf_counter() - t0
    out = {"config": dataclasses.asdict(cfg), "returns": r.returns, "T_f": Tf, "T_s": Ts, "frames": r.frames,
           "ms_per_frame": wall / max(r.frames, 1) * 1e3, "wall_s": wall, "W_abs_mean": float(np.abs(r.W).mean()),
           "W_neg_frac": float((r.W < 0).mean()), "n_in": r.lay.n_in, "mv": r.mv, "resumes": resumes}
    if cfg.heads > 1:
        out["returns_heads"] = r.returns_heads
    if eta_sgd is not None:
        out["eta_sgd_mV"] = eta_sgd.tolist()
    return out


def pooled_times(task, ret):
    """main.py's t_f and t_s on the pooled episode sequence (1-indexed; None if never): T_f = first episode with return
    >= the perfect score, T_s = last episode of the first 100-episode window whose mean reaches it."""
    perf, ret = SOLVE[task][0], np.asarray(ret, float)
    hit = np.flatnonzero(ret >= perf)
    return (int(hit[0]) + 1 if len(hit) else None), trailing_mean_time(ret, perf)


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
    if cfg.ck:
        return run_ck(cfg, log_every)
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
          "unsolved as infinite. Authors'-code groups (ck) count episodes pooled over copies (main.py); legacy "
          "LunarLander groups count per copy (copy-mean returns). Secondary: T_m = first episode whose trailing-100 "
          "mean reaches M_THR (CartPole 475, Gymnasium's threshold; LunarLander 200 = T_s). ret: last-100 return, "
          "mean +- sd over seeds; >=M: runs with ret >= M_THR. MW: Mann-Whitney p against the family's base group "
          "(the task preset), on T_f / T_s (never reached = episode cap + 1) / ret.")
    groups = {}
    for f in sorted(glob.glob(str(OUT / "tdstdp_*.json"))):
        groups.setdefault(pathlib.Path(f).stem.rsplit("_s", 1)[0], []).append(json.loads(pathlib.Path(f).read_text()))
    ms = lambda x: f"{np.mean(x):6.1f} +- {np.std(x):5.1f}" if x else "   nan          "  # noqa: E731
    cap = lambda r: (r["config"].get("max_eps") or r["config"]["episodes"]) + 1  # noqa: E731
    for task in PAPER:
        if not any(g.split("_")[1] == task for g in groups):
            continue
        p = PAPER[task]
        print(f"\n{task}: CK 2020 T_f {p['T_f'][0]:.1f} +- {p['T_f'][1]:.1f}  "
              f"T_s {p['T_s'][0]:.1f} +- {p['T_s'][1]:.1f}")
        for k, R in v4_returns(task).items():
            print(f"  {k:28} n={len(R):2d} ret {np.mean(R):7.1f} +- {np.std(R):5.1f}")
        for ck, base in ((False, f"tdstdp_{task}"), (True, f"tdstdp_{task}_ck")):
            gs = {g: rs for g, rs in groups.items() if g.split("_")[1] == task and (g.split("_")[2:3] == ["ck"]) == ck}
            if not gs:
                continue
            print("  authors' code (default):" if ck else "  pre-audit code (--legacy):")
            for g, rs in sorted(gs.items(), key=lambda kv: (kv[0] != base, kv[0])):
                Tf = [r["T_f"] for r in rs if r["T_f"] is not None]
                Ts = [r["T_s"] for r in rs if r["T_s"] is not None]
                Tm = [t for t in (trailing_mean_time(r["returns"], M_THR[task]) for r in rs) if t is not None]
                med = np.median([r["T_s"] if r["T_s"] is not None else np.inf for r in rs])
                R = [np.mean(r["returns"][-100:]) for r in rs]
                line = (f"  {g:28} n={len(rs):2d} solved {len(Ts):2d}/{len(rs):2d}  T_f {ms(Tf)} [{len(Tf):2d}]  "
                        f"T_s {ms(Ts)} median {med:5.0f}  T_m {ms(Tm)} [{len(Tm):2d}]  ret {np.mean(R):6.1f} +- "
                        f"{np.std(R):5.1f} >=M {sum(x >= M_THR[task] for x in R):2d}  "
                        f"ms/fr {np.mean([r['ms_per_frame'] for r in rs]):5.2f}")
                if g != base and base in gs:
                    rb = gs[base]
                    cens = lambda rr, k: [r[k] or cap(r) for r in rr]  # noqa: E731
                    R0 = [np.mean(r["returns"][-100:]) for r in rb]
                    line += (f"  MW {mannwhitneyu(cens(rs, 'T_f'), cens(rb, 'T_f')).pvalue:.2g} / "
                             f"{mannwhitneyu(cens(rs, 'T_s'), cens(rb, 'T_s')).pvalue:.2g} / "
                             f"{mannwhitneyu(R, R0).pvalue:.2g}")
                print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(PRESETS), default="CartPole")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="authors' code: min_eps = max_eps (pooled)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--adam", action="store_true", default=None)
    g.add_argument("--no-adam", dest="adam", action="store_false")
    g.add_argument("--sgd", action="store_true", help="E4: no Adam, eta from the Adam-equivalent step at init")
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--legacy", action="store_true", help="pre-audit implementation (E1, E2, first E3/E4)")
    ap.add_argument("--rate0", type=float, default=None, help="legacy: initial spike probability per ms (post hoc)")
    ap.add_argument("--engine", choices=("numpy", "snm", "lif", "lif-snm", "flow"), default="numpy",
                    help="lif: BindsNET's LIF (legacy: CK Eq. 1); lif-snm (legacy): SuperNeuroMAT's subtractive leak; "
                         "flow: the branching step's noise-free limit")
    ap.add_argument("--fast", action="store_true", help="update summed over copies in one product (not bit-identical)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    kw = {k: v for k, v in (("episodes", a.episodes), ("heads", a.heads), ("block", a.block), ("rate0", a.rate0))
          if v is not None}
    if a.fast:
        kw["fast"] = True
    if a.episodes is not None and not a.legacy:
        kw["max_eps"] = a.episodes
    if a.sgd:
        kw.update(adam=False, sgd=True)
    elif a.adam is not None:
        kw["adam"] = a.adam
    run(preset(a.task, legacy=a.legacy, seed=a.seed, engine=a.engine, **kw))


if __name__ == "__main__":
    main()
