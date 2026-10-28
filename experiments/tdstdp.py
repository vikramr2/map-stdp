"""Iteration 013 step 1: the feed-forward TD-STDP actor-critic of Chung & Kozma 2020 [CK] (arXiv:2008.13044,
mapstdp/tdstdp.py) in closed loop on CartPole and LunarLander. Run from the repo root:
  python -m experiments.tdstdp --task {CartPole,LunarLander} --seed 0 [--adam | --no-adam | --eta-cal] [--heads H]
    [--block B] [--episodes E] [--engine {numpy,snm,lif,flow,nlif}] [--s-in S] [--legacy [--rate0 p]
    [--engine lif-snm]] [--rec {cols,lat} [--chi-rec X] [--lam L] [--lam-ratio r] [--save-model PATH] ...]
    -> experiments/results/tdstdp_<task>[_ck][_<tokens>]_s<seed>.json   (tokens: options off the preset, in order
       adam | noadam | sgd (= --eta-cal), B<B>, h<H>, ep<E>, r<rate0> (legacy), fast, si<S>, engine snm | lif |
       lifsnm | flow | nlif)
    [--out PATH]   (result path; the default above; experiments.report uses it to write reports/<study>/runs/)
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
step's noise-free limit (tdstdp.flow_step): rates instead of spikes, plus the input autocorrelation term of z.
--engine nlif (iteration 015): the noisy LIF of mapstdp.lif on the 1 ms grid (tdstdp.simulate_nlif) on every post, the
critic included; W, init and eta in mV as engine lif. The input scale s_in is set before the run, not tuned on return
(calibrate_s_in: the mean initial post rate equals engine lif's on the same stream); --s-in fixes it instead. Rates rho,
value, softmax and the rule are unchanged; the membranes are carried across blocks and set to mu_bg (no draw) on an
episode reset. The neuron noise has its own stream, so the main rng (inputs, actions, env seeds) is consumed as
engine lif's.
--rate-target R (iteration 015 step 3, engine nlif; token rt<R>): the low-rate regime, tau_ref 2 ms (lif.DT1R), s_in
set so the mean initial post rate is R Hz, the softmax and critic gains alpha_a, alpha_c times k = r_CK / R (r_CK:
BindsNET's initial rate on the same probe) and, without --eta-cal or Adam, eta_c, eta_a matched to the RMS of
BindsNET's initial per-ms update (calibrate_low; the result's 'low_rate' records it). tau_n stays CK's 20 ms.
--rec {cols,lat} (iteration 015 step 2, engine nlif, --block 20, no Adam; derivation §8.1, Proposal): recurrent E->E
modules in the actor (tdstdp.HybridRec): R among the actor E posts, one module per action ('cols'), plus --n-lat-mod
latent modules of n_ae posts ('lat'; input->latent W frozen at init, latent->action R learned, latent-output barrier
--la-beta). Kicks chi_rec R / d_j (--chi-rec). R learns by the one-hop task credit (eta_rec) and the map term (lam_rec,
off with --lam 0) with d_j homeostasis; the counter's gain is the kernel's own transfer (lif.gain_markov). eta_rec and
lam_rec are pre-set by calibrate_rec unless given (--eta-rec, --lam-rec); s_in and --eta-cal are calibrated without
recurrence, so every arm of a seed shares them. --rec-seal drops R's cross-module synapses from the same draws
(enforced modularity). Tokens after the engine: cols | lat, xr<chi_rec>, seal, lam0 | lam<l>,
lr<lam_ratio>, lx<lam_rec>, er<eta_rec>, sw<stat_windows>, pw<pair_ms>, nl<n_lat_mod>, lb<la_beta> (off-default
only, except the first two). The result's 'rec' holds the calibration and the monitors (HybridRec.log, one row per
pooled-episode event); --save-model PATH writes R snapshots (episode 0, every SNAP_EVERY, end) with m, roles, mask."""
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
from mapstdp import lif
from mapstdp import tdstdp as td

OUT = pathlib.Path(__file__).parent / "results"
TDSTDP_SEED, CAL_SEED, CAL_TICKS, FBAR_STATES, MV_LEGACY, MS, WARM_TICKS = 1302, 1303, 200, 5000, 1300.0, 20, 5
MV, WARM_N = td.THETA_L, 5  # authors' code: mV per unit of branching weight; batch_envs warm_n (windows)
PAPER = {"CartPole": dict(T_f=(57.0, 23.97), T_s=(169.5, 23.47)),
         "LunarLander": dict(T_f=(383.8, 71.49), T_s=(2575.2, 666.51))}  # CK Table 2 (10 runs)
SOLVE = {"CartPole": (500.0, "all"), "LunarLander": (200.0, "mean")}  # perfect score; solved = last 100 all / mean
M_THR = {"CartPole": 475.0, "LunarLander": 200.0}  # secondary: trailing-100 mean threshold (summary only)
SNAP_EVERY = {"CartPole": 100, "LunarLander": 500}  # rec: R snapshots every this many pooled episodes


@dataclasses.dataclass(frozen=True)
class Config:
    task: str = "CartPole"
    seed: int = 0
    episodes: int = 400
    adam: bool = False
    sgd: bool = False  # E4 (--eta-cal): no Adam; the local rule with eta from the Adam-equivalent step at init
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
    s_in: float = 0.0  # engine nlif: feed-forward input scale (0 = calibrate_s_in before the run)
    rate_target: float = 0.0  # engine nlif, low-rate regime (Hz; 0 = CK-matched): calibrate_low before the run
    # iteration 015 (engine nlif only; tdstdp.HybridRec): recurrent modules in the actor
    rec: str = "none"  # 'none' | 'cols' (E->E matrix R on the actor E posts, a module per action) | 'lat' (+ latents)
    chi_rec: float = 0.3  # total kick of one spike over its targets (threshold units)
    lam: float = 1.0  # map term: lambda = lam lam_rec (0 = off)
    lam_ratio: float = 1.0  # rec_cal: lam_rec = lam_ratio RMS(task step on R) / RMS(structural step), per frame
    lam_rec: float = 0.0  # structural rate (0 = calibrate_rec before the run)
    eta_rec: float = 0.0  # task rate on R (0 = calibrate_rec: the actor W's relative step at init)
    stat_windows: int = 5  # windows (20 ms) per statistics frame
    pair_ms: int = 20  # pair-count window (ms)
    n_lat_mod: int = 2  # rec 'lat': latent modules of n_ae posts each (input->latent W frozen at init)
    la_beta: float = 0.3  # rec 'lat': latent-output barrier (core.latent_output_barrier)
    rec_seal: bool = False  # enforced modularity: R without cross-module synapses (same draws; iteration 015)


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
    t += [f"si{cfg.s_in:g}"] if cfg.s_in else []
    t += [cfg.engine.replace("-", "")] if cfg.engine != "numpy" else []
    t += [f"rt{cfg.rate_target:g}"] if cfg.rate_target else []
    if cfg.rec != "none":
        d = Config()
        t += [cfg.rec, f"xr{cfg.chi_rec:g}"] + (["seal"] if cfg.rec_seal else [])
        t += ["lam0"] if cfg.lam == 0 else [f"lam{cfg.lam:g}"] if cfg.lam != 1 else []
        t += [f"lr{cfg.lam_ratio:g}"] if cfg.lam_ratio != d.lam_ratio else []
        t += [f"lx{cfg.lam_rec:g}"] if cfg.lam_rec else []
        t += [f"er{cfg.eta_rec:g}"] if cfg.eta_rec else []
        t += [f"sw{cfg.stat_windows}"] if cfg.stat_windows != d.stat_windows else []
        t += [f"pw{cfg.pair_ms}"] if cfg.pair_ms != d.pair_ms else []
        if cfg.rec == "lat":
            t += [f"nl{cfg.n_lat_mod}"] if cfg.n_lat_mod != d.n_lat_mod else []
            t += [f"lb{cfg.la_beta:g}"] if cfg.la_beta != d.la_beta else []
    return t


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
        assert cfg.engine != "nlif", "engine nlif: authors' code only"
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
    order they end; the run stops when solved with >= `episodes` pooled episodes, or at `max_eps`.
    Engine nlif: the membranes (lif state lst: V, ref, z) are reset to mu_bg on code 2 like BindsNET's to rest; the
    neuron noise draws from nrng. probe = True (calibrate_s_in) skips the rule and counts spikes in spk."""

    def __init__(self, cfg, rng_seed=TDSTDP_SEED, frozen=False):
        assert cfg.ck and cfg.engine in ("numpy", "snm", "lif", "flow", "nlif") and cfg.c_t == 0 and MS % cfg.block == 0
        assert cfg.engine != "nlif" or cfg.s_in > 0, "engine nlif needs s_in (run_ck calibrates it)"
        assert not cfg.rate_target or cfg.engine == "nlif", "rate_target: engine nlif"
        assert cfg.rec in ("none", "cols", "lat")
        assert cfg.rec == "none" or (cfg.engine == "nlif" and cfg.block == MS and not cfg.adam), \
            "rec: engine nlif, block 20, no Adam"
        self.cfg, self.frozen = cfg, frozen
        task = TASKS[cfg.task]
        self.K, self.H = K, H = task.n_actions, cfg.heads
        self.phi, n_in = ck_encoder(cfg)
        n_lat = cfg.n_lat_mod * cfg.n_ae if cfg.rec == "lat" else 0
        self.lay = lay = td.Layout(n_in, cfg.n_ce, cfg.n_ci, cfg.n_ae, cfg.n_ai, K, n_lat)
        self.rng = np.random.default_rng((cfg.seed, rng_seed))
        self.lif = cfg.engine == "lif"  # BindsNET LIF, W in mV
        self.nlif = cfg.engine == "nlif"  # noisy LIF (mapstdp.lif, dt 1 ms), W in mV
        self.mv = 1.0 if self.lif or self.nlif else MV
        self.probe, self.spk = False, np.zeros(2)  # s_in calibration: summed post spikes, spike slots
        if self.nlif:
            self.lp = dataclasses.replace(lif.LIFParams(), **(lif.DT1R if cfg.rate_target else lif.DT1))
            self.lst = lif.init_state(lay.n_post, H, self.lp)
            self.lst["V"] = np.full((H, lay.n_post), self.lp.mu_bg)
            self.nrng = np.random.default_rng((cfg.seed, rng_seed, 1))
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
        self.lowrate = None  # rate_target: calibrate_low's record (run_ck sets it; saved in checkpoints)
        self.hr = None  # rec: tdstdp.HybridRec (own stream (seed, rng_seed, 2)); W rows of latent posts frozen
        if cfg.rec != "none":
            self.hr = td.HybridRec(lay, K, cfg.n_ae, cfg.n_lat_mod if n_lat else 0, cfg.chi_rec, self.lp, H,
                                   (cfg.seed, rng_seed, 2), cfg.stat_windows, cfg.pair_ms, cfg.lam * cfg.lam_rec,
                                   cfg.la_beta if n_lat else 0.0, seal=cfg.rec_seal)
            self.hs = np.zeros((H, self.hr.Kr))  # module scores of the window's credit
            self.snaps = {0: self.hr.R.copy()}
            self.rsq = np.zeros(5)  # rec_cal sums: actor g^2, actor W^2 (per entry), task dir^2, frame task^2, dS^2
            self.rcnt = np.zeros(3)  # counts: windows, frames (task), frames (structural)
            self.ftask = 0.0  # rec_cal: task direction summed over the current frame

    def tick(self):
        cfg, lay, K = self.cfg, self.lay, self.K
        rs = self.code == 2
        if rs.any():
            for a in (self.z, self.q, self.P, self.rho, self.v):
                a[rs] = 0
            self.xprev[:] = 0
            if self.nlif:
                self.lst["V"][rs], self.lst["ref"][rs], self.lst["z"][rs] = self.lp.mu_bg, 0, False
            if self.hr is not None:
                self.hr.reset_heads(rs)
        act, sa = lay.act, lay.sign[lay.n_crit:]
        c = np.eye(K)[self.action][:, act] - self.p[:, act]  # (A - p) with p fixed at the sampling probability
        kap = cfg.c_e * sa * td.entropy_coef(self.p)[:, act]  # Eq. 26, from the window's p
        if self.hr is not None:  # CK's action score per module (latents 0; their W rows get no update)
            self.hs[:, :K] = cfg.alpha_a * (np.eye(K)[self.action] - self.p) / cfg.n_ae
            if lay.n_lat:
                c[:, lay.lat], kap[:, lay.lat] = 0.0, 0.0
        f, r_ms = self.phi(self.obs), cfg.r_scale * self.reward / MS
        end, learn = self.is_end & ~self.trunc, self.code < 2
        for b in range(MS // cfg.block):
            self.learn_block(f, r_ms, end, learn, c, kap, np.arange(b * cfg.block, (b + 1) * cfg.block))
        self.p = td.softmax(cfg.alpha_a * self.rho @ lay.Ma.T)
        self.timer += MS
        nb = self.timer >= cfg.freeze
        prev = self.action.copy()
        for h in np.flatnonzero(nb):
            self.action[h] = self.rng.choice(K, p=self.p[h])
        self.timer[nb] = 0
        if self.hr is not None and nb.any():  # perseveration monitor (copies in an episode)
            ok = nb & (self.code < 2)
            if ok.any():
                self.hr.log_actions(float((self.action[ok] == prev[ok]).mean()), float((self.p[ok] ** 2).sum(1).mean()))
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
            if self.hr is not None:
                self.hr.flush(n)
                if n // SNAP_EVERY[cfg.task] > (n - int(new.sum())) // SNAP_EVERY[cfg.task]:
                    self.snaps[n // SNAP_EVERY[cfg.task] * SNAP_EVERY[cfg.task]] = self.hr.R.copy()
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
            elif self.nlif and self.hr is not None:
                X, xs, xq = td.simulate_nlif(self.W, self.xprev, x, self.lst, self.nrng, self.lp, cfg.s_in,
                                             self.hr.bt(), self.hr.ridx)
            elif self.nlif:
                X = td.simulate_nlif(self.W, self.xprev, x, self.lst, self.nrng, self.lp, cfg.s_in)[0]
            elif cfg.engine == "snm":
                X = np.concatenate([Runner._snm(self, h, x[:, h:h + 1], noise[:, h:h + 1]) for h in range(H)], 1)
            else:
                X = td.simulate(self.W, self.xprev, x, noise)
        P = td.pre_trace(self.P, x, cfg.tau_p)
        R = td.rates(self.rho, X, cfg.tau_n)
        V = cfg.alpha_c * R @ lay.Mv + cfg.beta_c  # Eq. 4
        if self.probe:  # calibrate_s_in: rates only, W frozen
            self.spk += [X.sum(), X.size]
            self.xprev, self.P, self.rho, self.Vprev = x[-1], P[-1], R[-1], V[-1]
            return
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
        if lay.n_lat:
            g[nc:][lay.lat] = 0.0  # input -> latent W frozen at init
        if self.frozen:
            self.gsq += [(g[:nc] ** 2).mean(), (g[nc:] ** 2).mean()]
            self.nupd += 1
        elif self.opt is not None:
            self.W += self.opt.step(g, self.nu / self.mv, L)
        else:
            self.W += self.nu / self.mv * g
        if self.hr is not None:
            self.rec_block(X, xs, xq, d, g)
        self.xprev, self.P, self.rho, self.Vprev = x[-1], P[-1], R[-1], V[-1]
        if learn.any():
            s = td.softmax(cfg.alpha_a * R[:, learn] @ lay.Ma.T)
            self.stats += [R[..., :nc][:, learn].mean(), R[..., nc:][:, learn].mean(),
                           -(s * np.log(np.maximum(s, 1e-300))).sum(-1).mean(), V[:, learn].mean(), 1]

    def rec_block(self, X, xs, xq, d, g):
        """rec (iteration 015, tdstdp.HybridRec), once per 20 ms window: the window's statistics, the one-hop credit
        of the previous window's counter with this window's (A - p) (the action sampled from that window's rates), the
        task step eta_rec mean_h (sum_t delta_ht) e_h on R, and at the end of a statistics frame the structural update
        and homeostasis. Frozen (calibrate_rec): R is not changed; the RMS sums of the actor's step, the task direction
        and the structural increment are accumulated instead."""
        hr, cfg, nc, lay = self.hr, self.cfg, self.lay.n_crit, self.lay
        ri = hr.ridx
        G, n = hr.window(X[..., ri], xs[:, ri], xq[:, ri])
        hr.credit(self.hs, G, n, np.exp(-MS / cfg.tau_q))
        u = hr.task_dir(d.sum(0))
        if self.frozen:
            act = nc + np.flatnonzero(~lay.lat)
            self.rsq[:3] += [(g[act] ** 2).mean(), (self.W[act] ** 2).mean(), (u[hr.mask] ** 2).mean()]
            self.rcnt[0] += 1
            self.ftask = self.ftask + u
        else:
            hr.apply_task(cfg.eta_rec * u)
        dS = hr.frame()
        if dS is not None and self.frozen:
            self.rsq[3:] += [(self.ftask[hr.mask] ** 2).mean() if np.ndim(self.ftask) else 0.0,
                             (dS[hr.mask] ** 2).mean()]
            self.rcnt[1:] += 1
            self.ftask = 0.0

    def save(self, fn, wall, resumes):
        o = self.opt
        nl = dict(V_lif=self.lst["V"], ref_lif=self.lst["ref"], z_lif=self.lst["z"], s_in=self.cfg.s_in,
                  nrng=json.dumps(self.nrng.bit_generator.state)) if self.nlif else {}
        if self.lowrate is not None:
            nl["lowrate"] = json.dumps(self.lowrate)
        if self.hr is not None:
            nl |= self.hr.state() | dict(eta_rec=self.cfg.eta_rec, lam_rec=self.cfg.lam_rec,
                                        snaps=json.dumps({k: v.tolist() for k, v in self.snaps.items()}))
        np.savez(fn, W=self.W, m=o.m if o else 0, v=o.v if o else 0, t=o.t if o else 0, frames=self.frames,
                 wall=wall, resumes=resumes, returns=json.dumps([self.returns, self.returns_heads]),
                 rng=json.dumps(self.rng.bit_generator.state), **nl)

    def load(self, fn):
        """Resume from a checkpoint: W, Adam moments, pooled returns, rng; every copy starts a fresh episode with
        cleared traces (not bit-identical to an uninterrupted run); engine nlif also restores the membranes and the
        neuron-noise stream. Returns (wall, resumes)."""
        c = np.load(fn)
        self.W = c["W"]
        if self.nlif:
            assert float(c["s_in"]) == self.cfg.s_in, (float(c["s_in"]), self.cfg.s_in)
            self.lst["V"], self.lst["ref"], self.lst["z"] = c["V_lif"].copy(), c["ref_lif"].copy(), c["z_lif"].copy()
            self.nrng.bit_generator.state = json.loads(str(c["nrng"]))
        if self.hr is not None:
            assert (float(c["eta_rec"]), float(c["lam_rec"])) == (self.cfg.eta_rec, self.cfg.lam_rec)
            self.hr.load_state(c)
            self.snaps = {int(k): np.array(v) for k, v in json.loads(str(c["snaps"])).items()}
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
    RMS-sized per-ms update moves W by the step per-ms Adam takes at init. Without recurrence (rec 'none'), so every rec
    arm gets the base run's eta. Returns (eta, rms), eta in mV."""
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=1, rec="none"), rng_seed=CAL_SEED, frozen=True)
    for _ in range(WARM_N + 1):
        r.tick()
    r.gsq[:], r.nupd = 0, 0
    for _ in range(CAL_TICKS):
        r.tick()
    rms = np.sqrt(r.gsq / r.nupd)
    return np.array([cfg.eta_c, cfg.eta_a]) / rms, rms


def probe_rate(cfg):
    """Mean post spike probability per ms (all posts and copies) over CAL_TICKS ticks after the first WARM_N + 1, W
    frozen at its init, on the (seed, CAL_SEED) stream; rule off (CKRunner.probe). Without recurrence (rec 'none'), so
    every rec arm gets the base run's s_in."""
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, block=MS, rec="none"), rng_seed=CAL_SEED, frozen=True)
    r.probe = True
    for _ in range(WARM_N + 1):
        r.tick()
    r.spk[:] = 0
    for _ in range(CAL_TICKS):
        r.tick()
    return r.spk[0] / r.spk[1]


def calibrate_s_in(cfg, rtol=2e-3):
    """Pre-set input scale of engine nlif (not tuned on return): s_in such that probe_rate equals engine lif's
    (BindsNET, deterministic) on the same stream and initial W, or with rate_target that rate (Hz), by bisection (the
    noise stream is common to every probe; the closed loop makes the rate only approximately monotone in s_in).
    Returns (s_in, target, rate, BindsNET's rate), rates per ms."""
    target = r_ck = probe_rate(dataclasses.replace(cfg, engine="lif", s_in=0.0, rate_target=0.0))
    if cfg.rate_target:  # low-rate regime: the target is rate_target, not BindsNET's rate
        target = cfg.rate_target * 1e-3
    lo, hi = 0.0, 1.0
    while probe_rate(dataclasses.replace(cfg, s_in=hi)) < target:
        lo, hi = hi, 2 * hi
        assert hi < 1e3, "nlif cannot reach the target rate"
    while hi - lo > rtol * hi:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if probe_rate(dataclasses.replace(cfg, s_in=mid)) < target else (lo, mid)
    return hi, float(target), float(probe_rate(dataclasses.replace(cfg, s_in=hi))), float(r_ck)


def calibrate_low(cfg):
    """Low-rate nlif regime (--rate-target R Hz; iteration 015 step 3), pre-set, not tuned on return, all without
    recurrence (so every rec arm of a seed shares it): tau_ref 2 ms (lif.DT1R); s_in by calibrate_s_in so that the mean
    initial post rate is R (unless --s-in is given); k = r_CK / R with r_CK BindsNET's initial rate on the same probe;
    alpha_a and alpha_c times k (policy logits and V keep CK's range; beta_c kept); without --eta-cal or Adam, eta_c
    and eta_a times RMS(g_CK) / RMS(g_low), the per-ms update RMS at init with W frozen on the CAL stream
    (calibrate_ck's measurement) of BindsNET's engine at CK's alphas and of the low-rate engine at the scaled ones
    (with --eta-cal, calibrate_ck normalises eta by RMS(g) itself). Returns (cfg with s_in, alphas and etas replaced,
    record)."""
    if cfg.s_in:
        s_in, target, r_ck = cfg.s_in, cfg.rate_target * 1e-3, probe_rate(dataclasses.replace(
            cfg, engine="lif", s_in=0.0, rate_target=0.0))
        got = probe_rate(cfg)
    else:
        s_in, target, got, r_ck = calibrate_s_in(cfg)
    k = r_ck / target
    low = dataclasses.replace(cfg, s_in=s_in, alpha_a=k * cfg.alpha_a, alpha_c=k * cfg.alpha_c)
    es = np.ones(2)
    if not (cfg.sgd or cfg.adam):
        rms_ck = calibrate_ck(dataclasses.replace(cfg, engine="lif", rate_target=0.0, s_in=0.0))[1]
        rms_low = calibrate_ck(low)[1]
        es = rms_ck / rms_low
        low = dataclasses.replace(low, eta_c=float(cfg.eta_c * es[0]), eta_a=float(cfg.eta_a * es[1]))
    rep = {f: getattr(low, f) for f in ("s_in", "alpha_a", "alpha_c", "eta_c", "eta_a")}
    return low, dict(k=float(k), r_ck_hz=1e3 * float(r_ck), r_target_hz=cfg.rate_target, r_got_hz=1e3 * float(got),
                     s_in=float(s_in), alpha_scale=float(k), eta_scale=[float(x) for x in es], cfg=rep)


def calibrate_rec(cfg):
    """Pre-set rates of rec (iteration 015; not tuned on return), on a run with W and R frozen at init on the
    (seed, CAL_SEED) stream, over CAL_TICKS windows after the first WARM_N + 1: eta_rec gives R the actor W's relative
    step per window, eta_a RMS(g_actor) / RMS(W_actor) = eta_rec RMS(u) / RMS(R) (u = mean_h (sum_t delta) e_h, the
    task direction of a window; RMS over existing synapses); lam_rec = lam_ratio RMS(eta_rec x u summed over a frame) /
    RMS(structural increment of a frame) (lam = 1). Measured on the unsealed draws (rec_seal off), so every arm of a
    seed shares the rates: with sealed R every target of j is in m(j), the one-hop credit h_m(i) - hbar_j is exactly 0,
    and RMS(u) is rounding noise. Returns (eta_rec, lam_rec, dict of the RMS values)."""
    r = CKRunner(dataclasses.replace(cfg, adam=False, sgd=False, eta_rec=1.0, lam_rec=0.0, rec_seal=False),
                 rng_seed=CAL_SEED, frozen=True)
    for _ in range(WARM_N + 1):
        r.tick()
    r.rsq[:], r.rcnt[:], r.ftask, r.hr.acc = 0, 0, 0.0, r.hr._fresh()
    for _ in range(CAL_TICKS):
        r.tick()
    hr, q = r.hr, np.sqrt(r.rsq / r.rcnt[[0, 0, 0, 1, 2]])
    rel_w = cfg.eta_a / r.mv * q[0] / q[1]
    eta = cfg.eta_rec or rel_w * float(np.sqrt((hr.R[hr.mask] ** 2).mean())) / q[2]
    lam = cfg.lam_rec or cfg.lam_ratio * eta * q[3] / q[4]
    return float(eta), float(lam), dict(rel_step_W=float(rel_w), rms_g_actor=q[0], rms_W_actor=q[1], rms_u=q[2],
                                        rms_u_frame=q[3], rms_dS=q[4])


def run_ck(cfg, log_every=50, out=None, save_model=None):
    eta_sgd, s_cal, rec_cal = None, None, None
    fn = pathlib.Path(out) if out else OUT / f"{name(cfg)}.json"
    fn.parent.mkdir(parents=True, exist_ok=True)
    ck = fn.with_suffix(".ckpt.npz")
    low = None
    if cfg.engine == "nlif" and cfg.rate_target:
        if ck.exists():
            low = json.loads(str(np.load(ck)["lowrate"]))
            cfg = dataclasses.replace(cfg, **low["cfg"])
        else:
            cfg, low = calibrate_low(cfg)
            print("low-rate calibration: " + ", ".join(f"{k} {v}" for k, v in low.items()), flush=True)
    elif cfg.engine == "nlif" and not cfg.s_in:
        if ck.exists():
            cfg = dataclasses.replace(cfg, s_in=float(np.load(ck)["s_in"]))
        else:
            s_in, target, got, _ = calibrate_s_in(cfg)
            cfg, s_cal = dataclasses.replace(cfg, s_in=s_in), [target, got]
            print(f"s_in calibration: initial post rate (spikes / ms) lif {target:.4f}, nlif {got:.4f} at s_in "
                  f"{s_in:.4f}", flush=True)
    if cfg.sgd:
        eta_sgd, rms = calibrate_ck(cfg)
        cfg = dataclasses.replace(cfg, eta_c=float(eta_sgd[0]), eta_a=float(eta_sgd[1]))
        print(f"eta calibration: per-ms update RMS {rms}, eta (mV) {eta_sgd}", flush=True)
    if cfg.rec != "none" and not (cfg.eta_rec and cfg.lam_rec):
        if ck.exists():
            c = np.load(ck)
            cfg = dataclasses.replace(cfg, eta_rec=float(c["eta_rec"]), lam_rec=float(c["lam_rec"]))
        else:
            eta_rec, lam_rec, rec_cal = calibrate_rec(cfg)
            cfg = dataclasses.replace(cfg, eta_rec=eta_rec, lam_rec=lam_rec)
            print(f"rec calibration: eta_rec {eta_rec:.4g}, lam_rec {lam_rec:.4g} (lam {cfg.lam:g}); "
                  + ", ".join(f"{k} {v:.3g}" for k, v in rec_cal.items()), flush=True)
    r, wall0, resumes = CKRunner(cfg), 0.0, 0
    r.lowrate = low
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
                  f"{(time.perf_counter() - t0) / 60:.1f} min" + (rec_line(r.hr) if r.hr else ""), flush=True)
            fn.with_suffix(".partial").write_text(json.dumps(result_ck(cfg, r, t0, eta_sgd, resumes, s_cal,
                                                                       rec_cal)))
            r.save(ck, time.perf_counter() - t0, resumes)
    if r.hr is not None:
        r.snaps[len(r.returns)] = r.hr.R.copy()
        if save_model:
            np.savez(save_model, m=r.hr.m, roles=r.hr.roles, mask=r.hr.mask, ridx=r.hr.ridx,
                     episodes=np.array(sorted(r.snaps)), **{f"R_ep{k}": v for k, v in r.snaps.items()})
    res = result_ck(cfg, r, t0, eta_sgd, resumes, s_cal, rec_cal)
    fn.write_text(json.dumps(res))
    fn.with_suffix(".partial").unlink(missing_ok=True)
    ck.unlink(missing_ok=True)
    print(f"T_f {res['T_f']} T_s {res['T_s']} episodes {len(res['returns'])} last100 "
          f"{np.mean(res['returns'][-100:]):.1f} ms/frame {res['ms_per_frame']:.2f}")
    return res


def rec_line(hr):
    """Log-line monitors of rec over the last logged bins (HybridRec.log)."""
    lg = hr.log
    if not lg["episode"]:
        return ""
    k = slice(-10, None)
    mx = lambda key: float(np.mean(lg[key][k]))  # noqa: E731
    return (f" | R: maxRe {mx('maxre'):.2f} (max {max(lg['maxre_max'][k]):.2f}) chiG {mx('chiG'):.2f} causal "
            f"{mx('causal'):.4f} within {mx('within'):.3f} L_obs {mx('L_obs'):.3f} rate {mx('rate'):.0f} Hz repeat "
            f"{mx('repeat'):.2f}/{mx('repeat_chance'):.2f}" + (f" lat {mx('rate_lat0'):.0f}/{mx('rate_lat1'):.0f} Hz "
                                                               f"lat_out {mx('lat_out'):.3f}" if hr.Kr > hr.K else ""))


def result_ck(cfg, r, t0, eta_sgd, resumes, s_cal=None, rec_cal=None):
    Tf, Ts = pooled_times(cfg.task, r.returns)
    wall = time.perf_counter() - t0
    out = {"config": dataclasses.asdict(cfg), "returns": r.returns, "T_f": Tf, "T_s": Ts, "frames": r.frames,
           "ms_per_frame": wall / max(r.frames, 1) * 1e3, "wall_s": wall, "W_abs_mean": float(np.abs(r.W).mean()),
           "W_neg_frac": float((r.W < 0).mean()), "n_in": r.lay.n_in, "mv": r.mv, "resumes": resumes}
    if cfg.heads > 1:
        out["returns_heads"] = r.returns_heads
    if eta_sgd is not None:
        out["eta_sgd_mV"] = eta_sgd.tolist()
    if cfg.engine == "nlif":
        out["s_in"] = cfg.s_in  # also in config; s_in_cal: [lif, nlif] initial post rate (absent after a resume)
        if s_cal is not None:
            out["s_in_cal"] = s_cal
    if r.lowrate is not None:
        out["low_rate"] = r.lowrate
    if r.hr is not None:  # iteration 015: monitors per flush (HybridRec.log), calibration (absent after a resume)
        hr = r.hr
        out["rec"] = {"eta_rec": cfg.eta_rec, "lam": cfg.lam * cfg.lam_rec, "log": hr.log, "n_R": hr.nR,
                      "n_syn": int(hr.mask.sum()),
                      "within_final": float((hr.R * (hr.m[:, None] == hr.m[None, :])).sum() / hr.R.sum())}
        H = np.eye(hr.m.max() + 1)[hr.m]  # module weight table [post, pre] / total R, per snapshot (SNAP_EVERY)
        out["rec"]["RK"] = {str(k): (H.T @ v @ H / v.sum()).tolist() for k, v in sorted(r.snaps.items())}
        if rec_cal is not None:
            out["rec"]["cal"] = rec_cal
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


def run(cfg, log_every=50, out=None, save_model=None):
    """Run cfg; the result goes to `out` (a .json path; .partial and .ckpt.npz alongside) or, by default, to
    experiments/results/<name(cfg)>.json. save_model (rec only): R snapshots."""
    if cfg.ck:
        return run_ck(cfg, log_every, out, save_model)
    eta_sgd = None
    if cfg.sgd:
        eta_sgd, rms = calibrate(cfg)
        cfg = dataclasses.replace(cfg, eta_c=float(eta_sgd[0]), eta_a=float(eta_sgd[1]))
        print(f"eta calibration: per-ms update RMS {rms}, eta (CK units) {eta_sgd}", flush=True)
    r, t0, nxt = Runner(cfg), time.perf_counter(), log_every
    fn = pathlib.Path(out) if out else OUT / f"{name(cfg)}.json"
    fn.parent.mkdir(parents=True, exist_ok=True)
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


def group_stats(task, rs, ref=None):
    """Summary statistics of one group of runs (result dicts) on `task`, shared by --summary and experiments.report:
    T_f / T_s (CK definitions) mean, sd (ddof 0) and count over runs that reach them; median T_s with unsolved =
    infinite; T_m = first episode whose trailing-100 mean reaches M_THR; last-100 return mean, sd and the count >=
    M_THR; mean ms per frame. With ref (another group's runs): two-sided Mann-Whitney p on T_f, T_s (never reached =
    episode cap + 1) and the last-100 return."""
    from scipy.stats import mannwhitneyu
    Tf = [r["T_f"] for r in rs if r["T_f"] is not None]
    Ts = [r["T_s"] for r in rs if r["T_s"] is not None]
    Tm = [t for t in (trailing_mean_time(r["returns"], M_THR[task]) for r in rs) if t is not None]
    R = [float(np.mean(r["returns"][-100:])) for r in rs]
    ms = lambda x: (float(np.mean(x)), float(np.std(x))) if x else (float("nan"),) * 2  # noqa: E731
    out = {"n": len(rs), "solved": len(Ts), "T_f_mean": ms(Tf)[0], "T_f_sd": ms(Tf)[1], "T_f_n": len(Tf),
           "T_s_mean": ms(Ts)[0], "T_s_sd": ms(Ts)[1],
           "T_s_median": float(np.median([r["T_s"] if r["T_s"] is not None else np.inf for r in rs])),
           "T_m_mean": ms(Tm)[0], "T_m_sd": ms(Tm)[1], "T_m_n": len(Tm), "ret_mean": float(np.mean(R)),
           "ret_sd": float(np.std(R)), "ret_ge_M": sum(x >= M_THR[task] for x in R),
           "ms_per_frame": float(np.mean([r["ms_per_frame"] for r in rs]))}
    if ref is not None:
        cap = lambda r: (r["config"].get("max_eps") or r["config"]["episodes"]) + 1  # noqa: E731
        cens = lambda rr, k: [r[k] or cap(r) for r in rr]  # noqa: E731
        R0 = [np.mean(r["returns"][-100:]) for r in ref]
        out |= {"mw_T_f": float(mannwhitneyu(cens(rs, "T_f"), cens(ref, "T_f")).pvalue),
                "mw_T_s": float(mannwhitneyu(cens(rs, "T_s"), cens(ref, "T_s")).pvalue),
                "mw_ret": float(mannwhitneyu(R, R0).pvalue)}
    return out


def summary():
    print("T_f / T_s (CK definitions, pre-registered): mean +- sd over runs that reach it [n]; median T_s counts "
          "unsolved as infinite. Authors'-code groups (ck) count episodes pooled over copies (main.py); legacy "
          "LunarLander groups count per copy (copy-mean returns). Secondary: T_m = first episode whose trailing-100 "
          "mean reaches M_THR (CartPole 475, Gymnasium's threshold; LunarLander 200 = T_s). ret: last-100 return, "
          "mean +- sd over seeds; >=M: runs with ret >= M_THR. MW: Mann-Whitney p against the family's base group "
          "(the task preset), on T_f / T_s (never reached = episode cap + 1) / ret.")
    groups = {}
    for f in sorted(glob.glob(str(OUT / "tdstdp_*.json"))):
        groups.setdefault(pathlib.Path(f).stem.rsplit("_s", 1)[0], []).append(json.loads(pathlib.Path(f).read_text()))
    ms = lambda m, s: f"{m:6.1f} +- {s:5.1f}" if m == m else "   nan          "  # noqa: E731
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
                st = group_stats(task, rs, gs[base] if g != base and base in gs else None)
                line = (f"  {g:28} n={st['n']:2d} solved {st['solved']:2d}/{st['n']:2d}  T_f "
                        f"{ms(st['T_f_mean'], st['T_f_sd'])} [{st['T_f_n']:2d}]  T_s {ms(st['T_s_mean'], st['T_s_sd'])}"
                        f" median {st['T_s_median']:5.0f}  T_m {ms(st['T_m_mean'], st['T_m_sd'])} [{st['T_m_n']:2d}]  "
                        f"ret {st['ret_mean']:6.1f} +- {st['ret_sd']:5.1f} >=M {st['ret_ge_M']:2d}  "
                        f"ms/fr {st['ms_per_frame']:5.2f}")
                if "mw_T_f" in st:
                    line += f"  MW {st['mw_T_f']:.2g} / {st['mw_T_s']:.2g} / {st['mw_ret']:.2g}"
                print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(PRESETS), default="CartPole")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="authors' code: min_eps = max_eps (pooled)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--adam", action="store_true", default=None)
    g.add_argument("--no-adam", dest="adam", action="store_false")
    g.add_argument("--eta-cal", dest="sgd", action="store_true",
                   help="E4: no Adam; the local rule with eta calibrated to Adam's step at init")
    g.add_argument("--sgd", dest="sgd", action="store_true", help=argparse.SUPPRESS)  # old name, in recorded commands
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--legacy", action="store_true", help="pre-audit implementation (E1, E2, first E3/E4)")
    ap.add_argument("--rate0", type=float, default=None, help="legacy: initial spike probability per ms (post hoc)")
    ap.add_argument("--engine", choices=("numpy", "snm", "lif", "lif-snm", "flow", "nlif"), default="numpy",
                    help="lif: BindsNET's LIF (legacy: CK Eq. 1); lif-snm (legacy): SuperNeuroMAT's subtractive leak; "
                         "flow: the branching step's noise-free limit; nlif: the noisy LIF of mapstdp.lif (dt 1 ms)")
    ap.add_argument("--s-in", type=float, default=None, help="nlif: input scale (default: calibrate_s_in)")
    ap.add_argument("--rate-target", type=float, default=None,
                    help="nlif: low-rate regime: initial post rate (Hz; tau_ref 2 ms; calibrate_low)")
    ap.add_argument("--fast", action="store_true", help="update summed over copies in one product (not bit-identical)")
    ap.add_argument("--rec", choices=("none", "cols", "lat"), default="none",
                    help="nlif: recurrent E->E actor modules (iteration 015): cols, one per action; lat, + latents")
    for f, t, h in (("chi-rec", float, "rec: total kick per spike (threshold units)"),
                    ("lam", float, "rec: map term multiplier (0 = off)"),
                    ("lam-ratio", float, "rec: lam_rec = ratio x RMS(task step) / RMS(structural step)"),
                    ("lam-rec", float, "rec: structural rate (default: calibrate_rec)"),
                    ("eta-rec", float, "rec: task rate on R (default: calibrate_rec)"),
                    ("stat-windows", int, "rec: windows per statistics frame"),
                    ("pair-ms", int, "rec: pair-count window (ms)"),
                    ("n-lat-mod", int, "rec lat: latent modules"),
                    ("la-beta", float, "rec lat: latent-output barrier")):
        ap.add_argument(f"--{f}", type=t, default=None, help=h)
    ap.add_argument("--rec-seal", action="store_true", help="rec: no cross-module synapses in R (enforced modularity)")
    ap.add_argument("--save-model", default=None, help="rec: R snapshots (.npz) for structure analysis")
    ap.add_argument("--out", default=None, help="result .json path (default experiments/results/<name>.json)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    kw = {k: v for k, v in (("episodes", a.episodes), ("heads", a.heads), ("block", a.block), ("rate0", a.rate0),
                            ("s_in", a.s_in)) if v is not None}
    if a.fast:
        kw["fast"] = True
    kw.update({k: v for k, v in (("rate_target", a.rate_target), ("chi_rec", a.chi_rec), ("lam", a.lam),
                                 ("lam_ratio", a.lam_ratio),
                                 ("lam_rec", a.lam_rec), ("eta_rec", a.eta_rec), ("stat_windows", a.stat_windows),
                                 ("pair_ms", a.pair_ms), ("n_lat_mod", a.n_lat_mod), ("la_beta", a.la_beta))
               if v is not None})
    if a.rec != "none":
        kw["rec"] = a.rec
    if a.rec_seal:
        kw["rec_seal"] = True
    if a.episodes is not None and not a.legacy:
        kw["max_eps"] = a.episodes
    if a.sgd:
        kw.update(adam=False, sgd=True)
    elif a.adam is not None:
        kw["adam"] = a.adam
    run(preset(a.task, legacy=a.legacy, seed=a.seed, engine=a.engine, **kw), out=a.out, save_model=a.save_model)


if __name__ == "__main__":
    main()
