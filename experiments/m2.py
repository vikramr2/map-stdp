"""M2 (docs/SPEC.md §11): CartPole in closed loop with Map-STDP, model v3 (docs/model.md). Run from the repo root:
  python -m experiments.m2 --backend {flow,spiking} --routing {exempt,none} --lam 0.05 --seed 0 --episodes 1500
    [--beta B] [--eta E] [--eta-tau T] [--la-beta B]
    -> experiments/results/m2_<backend>_<routing>_lam<lam>[_b<B>][_eta<E>][_at<T>][_lb<B>]_v3_s<seed>.json
  python -m experiments.m2 --summary   -> results table over seeds (plus the random-policy return)
Model v3: d_j homeostasis as its own term (Eq. 4d, EPS_H, every frame), race burn-in BURN steps, task rate eta
annealed as eta / (1 + episode / ETA_TAU). Models v1 and v2 and the dual routing term (Eq. 4a) were removed in
iteration 013; their committed results still load in --summary (git history has the code).
Config (one per run, shared by the m2 and m3 CLIs) holds every option; its defaults reproduce v3 on CartPole:
  --eta-tau T: task rate annealed as eta / (1 + episode / T) (ablation; 0 = constant).
  --beta B: sharpened policy P ~ p_a^beta with the matched score (Eq. 1e, Proposal); flow, or spiking with diffbound.
  --la-beta B: latent-output barrier (Eq. 4e, Proposal) subtracted from g on latent->action pairs; 0 is off.
  p_cc: controller->controller connection probability of the SBM (core.build_network; 0.1 = M2). p_cc = 0 removes
    the C->C recurrence, whose neuron-level credit the module-level Eq. 4b cannot represent (iteration 006).
  p_cross: cross-module connection probability of the SBM outside the routing pairs (0.05 = M2). p_cross = 0 builds
    the modules sealed, the structure the map term converges to (iteration 014): the control for "the map term helps"
    versus "sealed modules help". The SBM draws are the same, so only the mask changes.
  beta_kappa k, beta_max B: causal adaptive gain (Eq. 1g, Proposal), one global scalar: after each episode
    log beta += k z, z = (R_fast - R_slow) / sd(R) clipped to [-2, 2] (fast / slow running returns over BETA_TAU_F /
    BETA_TAU_S episodes), beta kept in [beta, B]. It rises while return improves and falls when it drops.
  gamma (iteration 009): discount of the critic and of delta (Eq. 4c); default GAMMA = 0.99.
  readout='diffbound' (Proposal, iteration 010; spiking only): the action is the count-difference race to a bound
    (Eq. 1h, sp.diffbound, at most one spike per action module and head per step) instead of the race. beta (fixed,
    or adaptive with beta_kappa) is the target sharpness: theta = BOUND_CAL^-1(beta), log-log interpolation of the
    measured beta_eff(theta) table for the agent's heads, clipped to the table, randomised between floor and ceil for
    a fractional theta. The Eq. 4b score uses beta (matched where the table reaches beta; below its first entry the
    readout is sharper than the score). Draws one extra uniform per frame, only on this path.
  Iteration 012 (Proposals, flow; all off by default, none draws from rng):
  encoder SPEC: the controller encoder replaced by tasks.with_encoder(task, SPEC) (grid g<n>s<sigma> or Fourier
    f<order>c<conj>[h][b]); the controller size becomes the feature count.
  critic_layer SPEC: separate critic, V = u . phi(o) on its own fixed RF layer (tasks.critic_layer), NLMS (Eq. 4c'');
    the actor's flow is not used by the critic.
  c_ent c[, ent_target f]: entropy term (Eq. 4h), a separate per-frame ungated update W += eta_e c kappa
    (g_b / J^in_b - gbar_j) (core.entropy_score; average W * grad H(P~), P~ the beta = 1 action shares, pi fixed);
    eta_e the annealed task rate. ent_target adapts c SAC-style, log c += ENT_K (f log|A| - H(P~)) per frame (nats).
  backend='mpr' (Proposal, iteration 013, Phase B): no spikes; pi is the normalised neuron-level MPR fixed point of the
    QIF network (qif.rates, at qif.MPR_POINT with Config.qif overrides, warm-started from the previous frame), kappa the
    expected pairings T * pi at those rates, the action from Eq. 1d / 1e as in flow; evaluate() uses the same pi.
  engine='qif' (Proposal, iteration 013; spiking): mapstdp.qif, QIF neurons with fixed Lorentzian excitabilities drawn
    at init from their own rng stream (seed, QIF_SEED), Config.qif overriding qif.QIFParams fields. The state is carried
    across frames (warm start, reset at episode start; QIFParams.warm = False: cold start every frame), the race
    burn-in is QIFParams.burn bins and kappa uses the QIF pairing window (qif.pairings). The teleportation pulses draw
    from self.rng as the other engines do; evaluate() races from a reset state per held-out stimulus, after one
    warm-up frame.
Both backends share one frame update; they differ only in where pi, kappa and the action come from:
  flow:    pi = n-hop flow, kappa = J (expected pairings), action ~ Eq. 1d;
  spiking: pi = spike shares, kappa = covariance count / spikes in the frame, action = race readout.
Dividing kappa by the frame's spike count is the 1/(spikes per frame) scaling of lambda and eta (model.md §4).
rule='emergent' (derivation Eqs. 8-12): the walk is observed from spikes (pair counts kappa_raw, Eq. 8) and sets the
  baselines and M*; a signed counter c (Eq. 11a for branching, engine='numpy') sets each synapse's step: structural
  -lam (g - gbar^obs) c (Eq. 4), eligibility c (h - hbar^obs) (Eq. 4b), homeostasis -eps_h (d_j - 1) W pi_j / d_j
  (Eq. 4d). Flow (kappa_raw = c = J) runs the v4 code, bit for bit. Options obs_tau, ebar, step_stat; per-episode
  monitors in the result's 'emergent'; evaluate() adds L_obs and block_shares. Engines plug in at
  Agent.counter_frame (its docstring is the contract); qif: Eq. 11b with a regression gain (Agent.qif_counter_frame);
  lif: mapstdp.lif (Agent.lif_counter_frame; Config.lif overriding LIFParams; rule 'emergent' only, race or
  diffbound with BOUND_CAL_LIF, burn-in LIF_BURN steps).
run(): checkpoints every LOG_EVERY episodes (<out>.ckpt.npz, resumed, counted in 'resumes'), optional --save-model;
match_scales(): the pre-set step-size match of lambda and eta to flow (Config.lam_scale, eta_scale)."""
import argparse
import copy
import dataclasses
import glob
import json
import pathlib
import time

import gymnasium as gym
import numpy as np

from experiments.tasks import TASKS, random_states, with_encoder
from experiments.tasks import critic_layer as critic_layer_fn
from mapstdp import core, lif, qif
from mapstdp import spiking as sp

ALPHA, N_HOPS, K_SPK, RATE, THETA = 0.2, 20, 100, 1.0, 0.5
ETA, TAU_E, GAMMA, ETA_V = 10.0, 3.0, 0.99, 0.1
EPS_H, BURN = 5.0, 10  # d_j homeostasis rate (Eq. 4d, own term), race burn-in (steps)
ETA_TAU = 300.0  # task-rate annealing time constant (episodes)
WMIN = 1e-3  # weight floor (ablate wmin overrides)
ENT_K = 1e-3  # iteration 012: entropy-target gain
REC_EPIS = 20  # iteration 012: episodes recorded at the end of a run for diag012
BETA_TAU_F, BETA_TAU_S = 20.0, 100.0  # beta_kappa (Eq. 1g): fast and slow running episode return (episodes)
TD_TAU = 1000.0  # --td-norm (Proposal): time constant (frames) of the running mean square of delta
QIF_SEED = 7451  # engine='qif': second rng stream (Lorentzian excitabilities and reset states)
BOUND_CAL = {16: ((1, 3.1), (2, 3.5), (4, 4.0), (8, 5.0), (16, 6.5), (32, 9.2), (64, 11.5)),  # heads: (theta, beta_eff)
             8: ((1, 2.35), (2, 2.76), (4, 3.24), (8, 4.11), (16, 5.61), (32, 7.78), (64, 8.47))}
# H = 8 (iteration 014): experiments.bound_cal on the pilot014 flow CartPole models (seeds 0, 1; their H = 16 table
# reproduces the one above within 7%); the LunarLander models' policies are near-deterministic, so beta_eff diverges
# engine='qif' at the default QIFParams (iteration 013): same method on flow-v4 W (seed 0; mean of Acrobot, MountainCar,
# CartPole; warm frames after one warm-up frame); deadline fraction <= 0.03 at 16, 0.02-0.41 at 32, 0.27-0.88 at 64
BOUND_CAL_QIF = {16: ((1, 1.13), (2, 1.29), (4, 1.49), (8, 1.89), (16, 2.85), (32, 4.70), (64, 5.49)),
                 8: ((1, 0.67), (2, 0.81), (4, 1.07), (8, 1.62), (16, 2.95), (32, 4.24), (64, 4.26))}
# H = 8 (iteration 014): as BOUND_CAL[8], on the pilot014 flow CartPole models (their H = 16 table: 1.01-5.11, within
# 11% of the one above); deadline fraction <= 0.06 at 16, 0.22-0.38 at 32, 0.56-0.73 at 64
# engine='lif' at the default LIFParams: same method, python -m experiments.bound_cal --engine lif --heads 16 on flow-v4
# W (m3 --save-model, seed 0) of Acrobot, MountainCar, CartPole (the same command with --engine qif reproduces
# BOUND_CAL_QIF within 6%); deadline fraction <= 0.015 at 16, 0.22-0.31 at 32, 0.69-0.84 at 64. The background
# spikes cap the sharpness near 2.4 and make small bounds flatter than Eq. 1d (beta_eff < 1 below theta 8)
BOUND_CAL_LIF = {16: ((1, 0.47), (2, 0.61), (4, 0.78), (8, 1.06), (16, 1.63), (32, 2.20), (64, 2.35)),
                 8: ((1, 0.28), (2, 0.40), (4, 0.52), (8, 0.81), (16, 1.48), (32, 2.14), (64, 2.19))}  # as BOUND_CAL[8]
LIF_BURN = 40  # engine='lif': race / diffbound burn-in, steps of LIFParams.dt (20 ms, as QIF's burn)
MODEL = "v3"  # the model tag of m2 result files
# rule 'emergent': rng streams (seed, EVAL_SEED, episode) of evaluate()'s frames and (seed, MATCH_SEED) of
# match_scales(); neither touches the training rng
EVAL_SEED, MATCH_SEED = 9137, 5521
FROZEN_EPIS, FROZEN_SEED = 20, 7331  # rule 'emergent': end-of-run episodes with W frozen, stream (seed, FROZEN_SEED)
SAVE_EVERY = 250  # --save-model: W snapshot every SAVE_EVERY episodes (plus episode 0 and the end)
COUNTER_ENGINES = ("numpy", "qif", "lif")  # rule 'emergent': engines with a counter hook (Agent.counter_frame)
EP_LOG = ("L_obs", "Mstar", "Mstar_ceiling", "e_obs", "rho", "causal_share", "clip_bind", "silent_module")
ROUTINGS = ("exempt", "none")  # routing pairs exempt from the map term (default, derivation §4.5), or not
LOG_EVERY = 50  # module sizes are per task (tasks.Task.sizes; CartPole 36, 16, 16, 16, 16)
OUT = pathlib.Path(__file__).parent / "results"


@dataclasses.dataclass(frozen=True)
class Config:
    """One closed-loop run's options (module doc); the defaults are model v3 on CartPole (M2)."""
    backend: str = "flow"  # 'flow' | 'spiking'
    routing: str = "exempt"  # ROUTINGS
    lam: float = 0.05  # structural (map) term rate (Eq. 4); 0 = map-term ablation
    seed: int = 0
    beta: float = 1.0  # sharpened policy exponent (Eq. 1e); the lower bound of the adaptive beta
    eta: float | None = None  # task rate (None: ETA)
    eta_tau: float | None = None  # task-rate annealing time constant (None: ETA_TAU; 0 = constant)
    la_beta: float = 0.0  # latent-output barrier (Eq. 4e)
    task: str = "CartPole"  # a tasks.TASKS key (M3)
    heads: int = 1  # parallel spiking replicas on shared W, pooled (Proposal, spiking only)
    engine: str = "snm"  # 'snm' | 'numpy' (sp.run_heads, the same raster) | 'qif' (mapstdp.qif) | 'lif' (mapstdp.lif)
    td_norm: float = 0.0  # TD-error normalisation to this RMS (Proposal; 0 = off)
    ablate: dict = dataclasses.field(default_factory=dict)  # learning-rule ablations (iteration 005): wmin, tau_e
    p_cc: float = 0.1  # controller->controller connection probability
    p_cross: float = 0.05  # cross-module (non-routing) connection probability; 0 = modules sealed from the start
    beta_kappa: float = 0.0  # adaptive beta gain (Eq. 1g; 0 = off)
    beta_max: float = 4.0  # upper bound of the adaptive beta
    gamma: float = GAMMA  # discount (Eq. 4c)
    readout: str = "race"  # spiking action readout: 'race' (Eq. 1d) | 'diffbound' (Eq. 1h)
    encoder: str | None = None  # controller encoder spec (iteration 012)
    critic_layer: str | None = None  # separate critic RF layer spec (Eq. 4c'')
    c_ent: float = 0.0  # entropy term coefficient (Eq. 4h; 0 = off)
    ent_target: float | None = None  # adapt c_ent toward H(P~) = ent_target log|A|
    qif: dict = dataclasses.field(default_factory=dict)  # engine='qif': qif.QIFParams overrides
    lif: dict = dataclasses.field(default_factory=dict)  # engine='lif': lif.LIFParams overrides (rule 'emergent' only)
    rule: str = "v4"  # 'v4' | 'emergent' (Eqs. 8-12, module doc); 'v4' is the rule of models v3 and v4
    lam_scale: float = 1.0  # lambda x this, every engine (step-size matching, match_scales)
    eta_scale: float = 1.0  # task rate x this, every engine
    obs_tau: float = 0.0  # emergent, spiking: running averages of the walk statistics over obs_tau frames (0 = frame)
    ebar: str = "obs"  # emergent ablation: 'obs' (e^obs_j, Eq. 4) | 'module' (the chance walk: baseline weights p_b)
    step_stat: str = "c"  # emergent ablation: 'c' (signed counter, Eq. 11a) | 'kappa' (pair count, presynaptic sign)


def held_out_obs(n=100, seed=12345, task=TASKS["CartPole"]):
    """Fixed evaluation observations: states visited by a random policy."""
    return random_states(task.env_id, task.n_actions, n, seed)


def random_return(episodes=1000, seed=0, task=TASKS["CartPole"]):
    """Mean and sd of the uniform-random-policy return (the --summary baseline)."""
    env, rng, R = gym.make(task.env_id), np.random.default_rng(seed), []
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(1 << 30)))
        n, done = 0, False
        while not done:
            _, r, term, trunc, _ = env.step(int(rng.integers(task.n_actions)))
            n, done = n + r, term or trunc
        R.append(n)
    return float(np.mean(R)), float(np.std(R))


def td_stats():
    """Fresh running sums of the training diagnostics (Agent.diagnostics)."""
    return dict(n=0, td_abs=0.0, td_sq=0.0, td_task_abs=0.0, V_min=np.inf, V_max=-np.inf)


class Obs(tuple):
    """Rule 'emergent', spiking: a frame's walk and counter statistics divided by its total spikes n (kraw, c), plus
    the per-neuron spike counts ns, n and the head-steps (for the chance floor of the monitors)."""
    kraw = property(lambda s: s[0])
    c = property(lambda s: s[1])
    ns = property(lambda s: s[2])
    n = property(lambda s: s[3])
    steps = property(lambda s: s[4])


def obs_frame(S, kraw, c, H):
    """(pi, Obs) of a frame from the engine's raster and raw counters: pi = spike shares of the counts summed over
    heads (sp.pi_hat), kraw and c divided by the frame's total spikes (as kappa in rule 'v4')."""
    ns = S.reshape(len(S), H, -1).sum((0, 1)).astype(float)
    n = max(ns.sum(), 1.0)
    return ns / n, Obs((kraw / n, c / n, ns, n, len(S) * H))


class Agent:
    """Closed-loop Map-STDP agent: structural (Eq. 4), d_j homeostasis (Eq. 4d) and TD task terms (Eqs. 4b, 4c; task
    rate annealed), configured by a Config (module doc)."""

    def __init__(self, cfg):
        self.cfg, self.task, self.heads, self.engine = cfg, TASKS[cfg.task], cfg.heads, cfg.engine
        backend, bound = cfg.backend, cfg.readout == "diffbound"
        if cfg.encoder:  # iteration 012: controller encoder spec (tasks.with_encoder)
            self.task = with_encoder(self.task, cfg.encoder)
        assert not (cfg.encoder or cfg.critic_layer or cfg.c_ent) or backend == "flow", "iteration 012: flow"
        assert cfg.ent_target is None or cfg.c_ent > 0, "ent_target adapts c_ent multiplicatively: start it above 0"
        assert cfg.routing in ROUTINGS, cfg.routing
        assert cfg.heads == 1 or backend == "spiking", "heads are spiking replicas"
        assert cfg.engine in ("snm", "numpy", "qif", "lif"), "numpy: sp.run_heads, the batched branching step"
        assert cfg.engine != "lif" or (cfg.rule == "emergent" and backend == "spiking"), \
            "engine='lif': rule='emergent', spiking (no v4 covariance path)"
        cal = {"qif": BOUND_CAL_QIF, "lif": BOUND_CAL_LIF}.get(cfg.engine, BOUND_CAL)
        self.bound_cal = cal
        assert cfg.readout == "race" or (bound and backend == "spiking" and cal.get(cfg.heads)), "diffbound: cal H"
        assert cfg.beta == 1.0 or backend in ("flow", "mpr") or bound, "the race has no matched sharpened policy"
        assert not cfg.beta_kappa or backend in ("flow", "mpr") or bound, "adaptive beta: flow, mpr or diffbound"
        self.c_ent, self.ent_target, self.reg_sq = cfg.c_ent, cfg.ent_target, np.zeros(2)  # task, entropy sq sums
        self.crit = critic_layer_fn(self.task, cfg.critic_layer) if cfg.critic_layer else None
        self.wmin, self.tau_e = cfg.ablate.get("wmin", WMIN), cfg.ablate.get("tau_e", TAU_E)
        self.td_norm, self.td_ms, self.frames = cfg.td_norm, cfg.td_norm ** 2, 0  # td_ms starts at the reference
        self.stats = td_stats()
        self.W, self.m, self.roles, self.mask = core.build_network(self.task.sizes, self.task.roles, p_cc=cfg.p_cc,
                                                                   p_cross=cfg.p_cross,
                                                                   seed=cfg.seed)
        self.K, self.backend, self.routing, self.lam = len(self.roles), backend, cfg.routing, cfg.lam
        self.eta, self.eps_h = cfg.eta or ETA, EPS_H
        self.eta_tau = ETA_TAU if cfg.eta_tau is None else cfg.eta_tau
        self.burn, self.ep, self.la_beta, self.gamma = BURN, 0, cfg.la_beta, cfg.gamma
        self.readout, self.bound_n, self.bound_dl, self.bound_th = cfg.readout, 0, 0, 0.0
        self.beta, self.beta0, self.beta_kappa, self.beta_max = cfg.beta, cfg.beta, cfg.beta_kappa, cfg.beta_max
        self.r_fast = None
        self.A = np.flatnonzero(self.roles == "A")
        self.ctrl = self.roles[self.m] == "C"
        self.groups = [np.flatnonzero(self.m == a) for a in self.A]
        self.race_groups = sp.head_groups(self.groups, cfg.heads, len(self.m))
        self.rng = np.random.default_rng(cfg.seed)
        n_u = self.ctrl.sum() if self.crit is None else len(self.crit(held_out_obs(1, task=self.task)[0]))
        self.u, self.silent = np.zeros(n_u), 0
        self.held = [self.stim(o) for o in held_out_obs(task=self.task)]
        self.rec = None  # list: record observations (iteration 012 end-of-run diagnostics; no rng)
        self.snn, self.live = None, backend == "spiking"  # live: evaluate() runs races, so it draws from rng
        if self.live and cfg.engine == "snm":
            self.snn = sp.build(sp.tile_heads(self.W, cfg.heads), 1 - ALPHA, theta=THETA)
        self.qp, self.qstate, self.r_warm = None, None, None
        if backend == "mpr":  # rates backend (Phase B): qif.MPR_POINT, Config.qif overriding
            assert cfg.heads == 1 and cfg.readout == "race", "mpr: no spikes"
            self.qp = qif.QIFParams(**{**qif.MPR_POINT, **cfg.qif})
            self.mpr_stats = dict(n=0, it=0, fallback=0, failed=0)
        if cfg.engine == "qif":  # own rng stream: the default path draws nothing extra
            self.qp, self.qrng = qif.QIFParams(**cfg.qif), np.random.default_rng((cfg.seed, QIF_SEED))
            M = qif.n_neurons(len(self.m), self.m, self.roles, self.qp)
            self.eta_q, self.burn = qif.lorentzian(self.qrng, (cfg.heads, M), self.qp), self.qp.burn
            self.lam, self.eta = self.lam * self.qp.lam_scale, self.eta * self.qp.eta_scale  # RMS match (Phase A)
            self.eps_h = self.eps_h * self.qp.eps_h_scale
            self.qgain = qif.GainEstimator(cfg.heads, len(self.m), self.m, self.K, self.qp)  # emergent; draws nothing
            self.qtail = None  # emergent: the previous frame's pairing tail (reset with qstate)
        self.lp, self.lstate, self.lcarry = None, None, None
        if cfg.engine == "lif":  # no rng of its own: every draw is from the frame's rng
            self.lp, self.burn = lif.LIFParams(**cfg.lif), LIF_BURN
            self.lam, self.eta = self.lam * self.lp.lam_scale, self.eta * self.lp.eta_scale
            self.eps_h = self.eps_h * self.lp.eps_h_scale
        if cfg.lam_scale != 1.0:  # Config-level step-size match (any engine; match_scales)
            self.lam = self.lam * cfg.lam_scale
        if cfg.eta_scale != 1.0:
            self.eta = self.eta * cfg.eta_scale
        self.rule, self.obs_tau, self.ebar, self.step_stat = cfg.rule, cfg.obs_tau, cfg.ebar, cfg.step_stat
        assert cfg.rule in ("v4", "emergent"), cfg.rule
        assert cfg.ebar in ("obs", "module") and cfg.step_stat in ("c", "kappa"), (cfg.ebar, cfg.step_stat)
        assert cfg.rule == "emergent" or (cfg.obs_tau == 0 and cfg.ebar == "obs" and cfg.step_stat == "c"), \
            "obs_tau, ebar and step_stat are options of rule='emergent'"
        assert cfg.rule == "v4" or backend in ("flow", "spiking"), "emergent: flow or spiking"
        assert cfg.rule == "v4" or backend == "flow" or cfg.engine in COUNTER_ENGINES, \
            f"emergent: spiking engines with a counter {tuple(COUNTER_ENGINES)} (snm: use engine='numpy', same raster)"
        self.run_obs, self.ep_log = None, {k: [] for k in EP_LOG}  # emergent: running walk statistics, episode logs
        self.elog = dict.fromkeys(EP_LOG, 0.0) | {"n": 0}

    def walk(self, v, warm=False):
        """The frame's flow pi: the n-hop flow (Eq. 1a; flow and spiking backends), or for backend='mpr' the normalised
        neuron-level MPR fixed point r* / sum r* over the E neurons (qif.rates; warm: start from and keep the training
        walk's previous solution). Counts the solver's iterations, fallbacks and failures in self.mpr_stats."""
        if self.backend != "mpr":
            return core.flow(self.W, v, ALPHA, N_HOPS)
        r, _, info = qif.rates(self.W, v, self.qp, self.m, self.roles, r0=self.r_warm if warm else None)
        st = self.mpr_stats
        st["n"], st["it"] = st["n"] + 1, st["it"] + info["it"]
        st["fallback"] += bool(info["fallback"])
        st["failed"] += not info["converged"]
        if warm:
            self.r_warm = r
        rE = r[:len(self.m)]
        return rE / rE.sum()

    def stim(self, o):
        """v(o): the task's encoder over the controller (tasks.Task.stim; CartPole: core.cartpole_stimulus)."""
        return self.task.stim(o, self.m, self.roles)

    def frame(self, v):
        """Walk and act for one frame (derivation §8 steps 2-3; spiking is cold-started, pi from spike shares).
        Returns pi, kappa (flow: expected pairings J = T * pi; spiking: covariance pairings / total spikes) and the
        action index into A (flow: sampled from Eq. 1d / 1e; spiking: race after burn-in, uniform if silent)."""
        W = self.W
        if self.backend in ("flow", "mpr"):
            pi = self.walk(v, warm=True)
            p = np.bincount(self.m, pi, self.K)
            return pi, W / W.sum(0) * pi, int(self.rng.choice(len(self.A), p=core.policy(p, self.roles, self.beta)))
        if self.rule == "emergent":
            S, kraw, c = self.counter_frame(v, self.rng)
        else:
            S = self.spike_frame(v)
        a = sp.race(S, self.race_groups, self.rng, self.burn) if self.readout == "race" else self.bound(S)
        if a is None:
            self.silent, a = self.silent + 1, int(self.rng.integers(len(self.A)))
        if self.rule == "emergent":
            return (*obs_frame(S, kraw, c, self.heads), a)
        pi, kappa, n = sp.pooled(S, self.heads) if self.qp is None else qif.pooled(S, self.heads, self.qp)
        return pi, kappa / max(n, 1), a

    def counter_frame(self, v, rng):
        """Rule 'emergent', spiking: one frame on the current W with the engine's walk and counter statistics.
        Engine contract (one branch per engine; add LIF here, mapstdp/lif.py): return (S, kappa_raw, c) with
          S          k x (heads N) bool raster, head h in columns h N .. (h + 1) N - 1 (the race reads it);
          kappa_raw  N x N [post, pre] pair counts in the engine's window w: spikes of i in (t, t + w] after each spike
                     of j at t (the walk, Eq. 8), summed over heads, cross-head pairs excluded;
          c          N x N signed counter, unbiased for the windowed direct effect, E[c_ij] = n_j D_ij (Eq. 11a, exact,
                     for branching; Eq. 11b, c = tau G_i b_ij n_j with a gain estimate, for LIF and QIF), summed over
                     heads.
        Both raw: the agent divides them by the frame's total spikes (obs_frame). Draw only from `rng` (evaluate() and
        match_scales() pass their own streams). An engine that carries state across frames should count the pairs that
        straddle the frame boundary (keep the previous frame's last window of spikes in its state); the branching engine
        is cold-started every frame (y_0 = 0), so every frame is independent and no pair is dropped."""
        if self.engine == "numpy":
            S, Y, X = sp.run_heads(self.W, v, self.heads, K_SPK, RATE, rng, THETA, 1 - ALPHA, return_drive=True)
            return (S, *sp.branching_counters(S, Y, X, self.W, self.heads, THETA, 1 - ALPHA))
        if self.engine == "qif":
            return self.qif_counter_frame(v, rng)
        if self.engine == "lif":
            return self.lif_counter_frame(v, rng)
        raise NotImplementedError(f"rule='emergent': no counter for engine {self.engine!r} (LIF: mapstdp/lif.py)")

    def qif_counter_frame(self, v, rng):
        """counter_frame for engine='qif': raw pair counts in the QIF pairing window (qif.window_pairs, E neurons only:
        the inhibitory pools are fixed, non-plastic and outside W, so they are not in the walk) and the mean counter
        c = tau G_i b_ij n_j (Eq. 11b, qif.mean_counter) with G from qif.GainEstimator (regression of rate on the fresh
        noise current across frames). Training (rng is self.rng): the warm state of spike_frame (same draws, same
        raster as rule v4), pairs straddling the previous frame counted (self.qtail), the gain estimate updated after
        the frame (the counter uses the estimate from earlier frames). Any other rng (evaluate, frozen W): a fresh
        state drawn from rng, one warm-up frame, then the counted frame; the training state and gain sums untouched."""
        p, H = self.qp, self.heads
        args = (self.W, v, H, rng, self.eta_q)
        if rng is self.rng:
            if self.qstate is None or not p.warm:
                self.qstate, self.qtail = qif.init_state(self.eta_q, self.qrng, p), None
            S, inp = qif.run_qif(*args, self.qstate, p, self.m, self.roles, inputs=True)
            kraw, self.qtail = qif.window_pairs(S, H, p, self.qtail if p.warm else None)
            G = self.qgain.gains(qif.prior_gain(inp, p))
            self.qgain.update(inp["xi"], inp["n"] / inp["T"])
        else:
            st = qif.init_state(self.eta_q, rng, p)
            tail = qif.window_pairs(qif.run_qif(*args, st, p, self.m, self.roles), H, p)[1] if p.warm else None
            if not p.warm:
                st = qif.init_state(self.eta_q, rng, p)
            S, inp = qif.run_qif(*args, st, p, self.m, self.roles, inputs=True)
            kraw = qif.window_pairs(S, H, p, tail)[0]
            G = self.qgain.gains(qif.prior_gain(inp, p))
        return S, kraw, qif.mean_counter(S, self.W, H, G, p)

    def lif_counter_frame(self, v, rng):
        """counter_frame for engine='lif' (mapstdp.lif.counter_frame: pair counts in the window LIFParams.window,
        straddling pairs included; c by LIFParams.counter). Training (rng is self.rng): the warm per-episode state
        self.lstate (lif.init_state at episode start, its gain EMA / regression moments carried over from the previous
        episode in self.lcarry, so they persist like a neuron's own running estimates). Any other rng (evaluate,
        frozen W): a fresh state with a copy of the carried gain, warmed up inside the frame from rng; the training
        state is untouched."""
        N, p = len(self.m), self.lp
        if rng is self.rng:
            if self.lstate is None:
                self.lstate = lif.init_state(N, self.heads, p)
                if self.lcarry is not None:
                    self.lstate["G"], self.lstate["reg"] = self.lcarry
            out = lif.counter_frame(self.W, v, self.heads, rng, self.lstate, p)
            self.lcarry = (self.lstate["G"], self.lstate["reg"])
            return out
        st = lif.init_state(N, self.heads, p)
        if self.lcarry is not None:
            st["G"], st["reg"] = copy.deepcopy(self.lcarry)
        return lif.counter_frame(self.W, v, self.heads, rng, st, p)

    def bound(self, S):
        """readout='diffbound' (Eq. 1h, module doc): theta from beta through BOUND_CAL, then sp.diffbound; None if
        silent."""
        th_cal, b_cal = np.log(np.array(self.bound_cal[self.heads], float)).T
        th = float(np.exp(np.interp(np.log(self.beta), b_cal, th_cal)))
        th = np.floor(th) + (self.rng.random() < th - np.floor(th))
        r = sp.diffbound(S, self.race_groups, th, self.rng, self.burn, cap=1, heads=self.heads)
        self.bound_n, self.bound_th = self.bound_n + 1, self.bound_th + th
        if r is None:
            return None
        self.bound_dl += r[1]
        return r[0]

    def spike_frame(self, v):
        """Spike raster of one frame on the current weights, k x (heads N), head h in columns h N .. (h + 1) N - 1."""
        if self.engine == "lif":  # the race of evaluate(); no gain statistics (return_stats=False)
            if self.lstate is None:
                self.lstate = lif.init_state(len(self.m), self.heads, self.lp)
            return lif.run_lif(self.W, v, self.heads, self.rng, self.lstate, self.lp, return_stats=False)
        if self.engine == "qif":
            if self.qstate is None or not self.qp.warm:
                self.qstate = qif.init_state(self.eta_q, self.qrng, self.qp)
            return qif.run_qif(self.W, v, self.heads, self.rng, self.eta_q, self.qstate, self.qp, self.m, self.roles)
        if self.engine == "numpy":
            return sp.run_heads(self.W, v, self.heads, K_SPK, RATE, self.rng, THETA, 1 - ALPHA)
        sp.set_weights(self.snn, sp.tile_heads(self.W, self.heads), 1 - ALPHA)  # output gain (1 - alpha) / d_j
        return sp.run_frame(self.snn, np.tile(v, self.heads), K_SPK, RATE, self.rng, THETA)[0]

    def feats(self, pi, o):
        """Critic features (Eq. 4c): x = pi_C / p_C (controller flow shares), so V = u . x; with critic_layer the
        separate RF layer y = phi(o) (Eq. 4c'')."""
        if self.crit is not None:
            return self.crit(o)
        return pi[self.ctrl] / max(pi[self.ctrl].sum(), 1e-12)

    def pre_td_updates(self, pi, kappa, a, e):
        """Per-frame updates that need no TD error (derivation §8 steps 4-5), in this order, which tests/golden.json
        locks: module statistics on the current W; entropy term (Eq. 4h, pre-update W); eligibility increment
        (Eq. 4b, its h-bar uses the pre-update W); structural term (Eq. 4); d_j homeostasis (Eq. 4d); entropy term
        applied. Mutates self.W; returns the updated trace e. rule='emergent': pre_td_emergent."""
        if self.rule == "emergent":
            return self.pre_td_emergent(pi, kappa, a, e)
        return self.pre_td_v4(pi, kappa, a, e)

    def pre_td_v4(self, pi, kappa, a, e):
        """pre_td_updates of rule 'v4' (Eqs. 4, 4b, 4d, 4e, 4h; the order tests/golden.json locks)."""
        p, q, JK, _ = core.module_stats(self.W, pi, self.m, self.K)
        reg = self.entropy_term(pi, kappa) if self.c_ent else None
        e = core.eligibility_step(e, self.W, kappa, pi, self.m, self.roles, self.A[a], self.tau_e, self.beta)
        if self.lam > 0:
            g = core.modulator(p, q, self.roles, routing=self.routing)
            if self.la_beta > 0:
                g = g - core.latent_output_barrier(p, JK, self.roles, self.la_beta)
            self.W = core.structural_update(self.W, self.mask, g, kappa, self.m, self.lam, wmin=self.wmin)
        self.W = core.homeostasis_update(self.W, self.mask, kappa, self.eps_h, self.wmin)
        if reg is not None:
            self.W = core.clip_weights(self.W + reg, self.mask, self.wmin)
        return e

    def walk_stats(self, pi, kraw):
        """The observed walk of a frame (Eq. 8) at module resolution: (p, q, JK, tK, e_obs) per frame, and the
        statistics the structural term uses (ps, qs, JKs, tKs): the frame's, or with obs_tau > 0 (spiking) those of
        exponential running averages over ~obs_tau frames of R[b, j] = sum_{i in b} kappa_ij / spikes and of pi
        (per-neuron and module counts only, no per-synapse state). Silent columns fall back to the module shares of
        T."""
        m, K = self.m, self.K
        fb = core.module_rows(self.W / self.W.sum(0), m, K)
        R = core.module_rows(kraw, m, K)
        fr = core.observed_walk_modules(R, pi, m, K, fb)
        if not self.obs_tau or self.backend == "flow":
            return fr, fr[:4]
        if self.run_obs is None:
            self.run_obs = [R.copy(), pi.copy()]
        else:
            for x, y in zip(self.run_obs, (R, pi)):
                x += (y - x) / self.obs_tau
        return fr, core.observed_walk_modules(*self.run_obs, m, K, fb)[:4]

    def increments(self, pi, kappa, a, task=True):
        """Rule 'emergent' per-frame increments on the current W, without rates: the structural increment
        -(g - gbar^obs) step (Eq. 4; step = c, or kappa_raw with step_stat='kappa'; baseline weights tK, or the chance
        walk p_b with ebar='module'), the eligibility increment c (h - hbar^obs) (Eq. 4b) and the walk statistics.
        Flow: kappa_raw = c = J (derivation: the flow engine is the a = X = 0 case of the observed walk). task=False:
        de = None (not computed)."""
        kraw, c = (kappa, kappa) if self.backend == "flow" else (kappa.kraw, kappa.c)
        fr, (ps, qs, JKs, tKs) = self.walk_stats(pi, kraw)
        tKb = tKs if self.ebar == "obs" else np.repeat(ps[:, None] / max(ps.sum(), 1e-300), len(self.m), 1)
        g = core.modulator(ps, qs, self.roles, routing=self.routing)
        if self.la_beta > 0:
            g = g - core.latent_output_barrier(ps, JKs, self.roles, self.la_beta)
        dS = core.structural_increment_emergent(g, tKb, c if self.step_stat == "c" else kraw, self.m)
        de = core.eligibility_step_emergent(0.0, c, fr[2], fr[3], self.m, self.roles, self.A[a], 1.0, self.beta) \
            if task else None
        return dS, de, fr, (ps, qs), kraw, c

    def pre_td_emergent(self, pi, kappa, a, e):
        """Rule 'emergent' (Eqs. 8-11): per frame, from the observed walk of the frame's spikes, the eligibility
        e <- (1 - 1/tau_e) e + c (h - hbar^obs) (Eq. 4b), the structural term W += -lam (g - gbar^obs) c (Eq. 4; g the
        map table with the routing exemption and the latent-output barrier on the observed module flow) and the d_j
        homeostasis W -= eps_h (d_j - 1) W pi_j / d_j (Eq. 4d), in pre_td_v4's order; logs the per-frame monitors.
        Flow with ebar='obs' runs pre_td_v4 itself, so emergent == v4 bit for bit there (the general formulas agree to
        1e-12: tests/test_emergent_rule.py)."""
        v4 = self.backend == "flow" and self.ebar == "obs"
        dS, de, fr, (ps, qs), kraw, c = self.increments(pi, kappa, a, task=not v4)
        dS = self.lam * dS if self.lam > 0 else 0.0
        self.monitor(pi, kappa, kraw, c, fr, ps, qs, dS)
        if v4:
            return self.pre_td_v4(pi, kappa, a, e)
        e = (1 - 1 / self.tau_e) * e + de
        self.W = core.clip_weights(self.W + dS, self.mask, self.wmin)
        self.W = core.homeostasis_update_emergent(self.W, self.mask, pi, self.eps_h, self.wmin)
        return e

    def monitor(self, pi, kappa, kraw, c, fr, ps, qs, dS):
        """Per-frame monitors of rule 'emergent', summed into self.elog (episode means in self.ep_log; no rng):
        L_obs (map equation of the frame's observed walk, bits); mean M* of the structural term and its chance ceiling
        (spiking: q at the chance floor of the window, mstar_ceiling); mean e^obs over active columns; the spectral
        radius of the module offspring Lambda^K (Eq. 12; flow: (1 - alpha) J); the median causal share
        sum_i |c_ij| / sum_i kappa_ij; the share of synapses whose structural or homeostatic update hit wmin or wmax;
        and silent-module frames."""
        p, q, JK, tK, e_obs = fr
        live = kraw.sum(0) > 0
        el, flow = self.elog, self.backend == "flow"
        el["n"] += 1
        el["L_obs"] += core.map_equation(p, q, pi)
        el["Mstar"] += float(core.mstar(ps, qs).mean())
        el["e_obs"] += float(e_obs[live].mean()) if live.any() else 0.0
        el["rho"] += core.module_offspring((1 - ALPHA) * c if flow else c, p, self.m, self.K)[1]
        el["causal_share"] += float(np.median(np.abs(c[:, live]).sum(0) / kraw[:, live].sum(0))) if live.any() else 0.0
        el["silent_module"] += float((p <= 0).any())
        if not flow:  # chance floor of each module's exit flow (derivation D2 step 5); a_i: spikes per step and head
            a_i = kappa.ns / kappa.steps
            Ej = np.where(kappa.ns > 0, kraw.sum(0) * kappa.n / np.maximum(kappa.ns, 1), 0.0)  # E_j, raw units
            out = (self.m[:, None] != self.m[None, :]) * a_i[:, None]
            qc = np.bincount(self.m, np.where(Ej > 0, pi * out.sum(0) / np.where(Ej > 0, Ej, 1.0), 0.0), self.K)
            el["Mstar_ceiling"] += float(core.mstar_ceiling(ps, qs, qc).mean())
        W1 = self.W + dS
        hit = (W1 < self.wmin) | (W1 > 1.0)
        W1 = core.clip_weights(W1, self.mask, self.wmin)
        W2 = W1 - self.eps_h * (W1.sum(0) - 1) * (W1 / W1.sum(0) * pi)
        el["clip_bind"] += float((hit | (W2 < self.wmin) | (W2 > 1.0))[self.mask].mean())

    def flush_elog(self):
        """Episode means of the rule-'emergent' monitors into self.ep_log (silent_module: frame count)."""
        n = max(self.elog["n"], 1)
        for k in EP_LOG:
            v = self.elog[k] if k == "silent_module" else self.elog[k] / n
            self.ep_log[k].append(float("nan") if (k == "Mstar_ceiling" and self.backend == "flow") else float(v))
        self.elog = dict.fromkeys(EP_LOG, 0.0) | {"n": 0}

    def task_rate(self):
        """Annealed task rate eta_e = eta / (1 + episode / eta_tau) (model v3); eta_tau = 0: constant."""
        return self.eta / (1 + self.ep / self.eta_tau) if self.eta_tau else self.eta

    def entropy_term(self, pi, kappa):
        """Entropy term (Eq. 4h, module doc) on the pre-update W (as Eq. 4b): the ungated per-frame increment
        eta_e c_ent kappa (g^H - gbar^H); W is clipped by the caller. With ent_target, c_ent adapts first.
        Accumulates its squared norm for the RMS ratio against the task term."""
        eta, W, m = self.task_rate(), self.W, self.m
        if self.ent_target is not None:
            H = core.action_entropy(W, pi, m, self.roles)
            H0 = self.ent_target * np.log(len(self.A))
            self.c_ent = float(np.clip(self.c_ent * np.exp(ENT_K * (H0 - H)), 1e-6, 10.0))
        dH = eta * self.c_ent * core.centered(core.entropy_score(W, pi, m, self.roles), W, m) * kappa
        self.reg_sq[1] += (dH[self.mask] ** 2).sum()
        return dH

    def critic_step(self, x, x2, term, r=0.0):
        """TD error delta_f (Eq. 4c: reward r_f = task.reward(r, term), for CartPole -1 on termination, 0 otherwise;
        bootstraps on truncation; V = u . x; discount self.gamma), computed before the NLMS critic update of u.
        Returns delta; the caller applies the task update eta_e delta e."""
        delta = self.task.reward(r, term) + (0.0 if term else self.gamma * self.u @ x2) - self.u @ x
        self.u += ETA_V * delta * x / (x @ x)
        return delta

    def task_step(self, e, delta):
        """Task term: W += eta_e delta' e (Eq. 4b trace e times the broadcast TD error delta' = td_scale(delta)),
        then the weight bounds. Returns delta'."""
        d_task, eta = self.td_scale(delta), self.task_rate()
        if self.c_ent:  # RMS of the task update (entropy-term calibration)
            self.reg_sq[0] += ((eta * d_task * e)[self.mask] ** 2).sum() if np.ndim(e) else 0.0
        self.W = core.clip_weights(self.W + eta * d_task * e, self.mask, self.wmin)
        return d_task

    def track(self, delta, d_task, V):
        """Accumulate the training diagnostics (diagnostics())."""
        st = self.stats
        st["n"], st["td_abs"], st["td_sq"] = st["n"] + 1, st["td_abs"] + abs(delta), st["td_sq"] + delta ** 2
        st["td_task_abs"] += abs(d_task)
        st["V_min"], st["V_max"] = min(st["V_min"], V), max(st["V_max"], V)

    def episode(self, env):
        """One episode, frame by frame (derivation §8): walk and act, pre-TD updates, env step, next frame's walk,
        critic, then frame f's task term (so the task update lands one walk late; spike_frame reads W at the start of
        each walk); with beta_kappa, the adaptive beta (Eq. 1g) after the episode."""
        o, _ = env.reset(seed=int(self.rng.integers(1 << 30)))
        self.qstate = self.lstate = None  # engine='qif' / 'lif': reset at episode start
        pi, kappa, a = self.frame(self.stim(o))
        x, e, R, done = self.feats(pi, o), 0.0, 0, False
        while not done:
            e = self.pre_td_updates(pi, kappa, a, e)
            o, r, term, trunc, _ = env.step(a)
            R, done = R + r, term or trunc
            pi, kappa, a = self.frame(self.stim(o))
            x2 = self.feats(pi, o)
            V = self.u @ x
            delta = self.critic_step(x, x2, term, r)
            if self.rec is not None:
                self.rec.append(o)
            d_task = self.task_step(e, delta)
            x = x2
            self.frames += 1
            self.track(delta, d_task, V)
        if self.beta_kappa:
            self.beta_step(R)
        if self.rule == "emergent":
            self.flush_elog()
        self.ep += 1
        return R

    def beta_step(self, R):
        """Causal adaptive gain on beta (Eq. 1g, Proposal): log beta integrates the normalised return trend."""
        if self.r_fast is None:
            self.r_fast, self.r_slow, self.r_var = R, R, 0.0
            return
        self.r_fast += (R - self.r_fast) / BETA_TAU_F
        self.r_var += ((R - self.r_slow) ** 2 - self.r_var) / BETA_TAU_S
        self.r_slow += (R - self.r_slow) / BETA_TAU_S
        if self.ep >= BETA_TAU_F:
            z = np.clip((self.r_fast - self.r_slow) / np.sqrt(self.r_var + 1e-12), -2, 2)
            self.beta = float(np.clip(self.beta * np.exp(self.beta_kappa * z), self.beta0, self.beta_max))

    def td_scale(self, delta):
        """Task-term TD error. --td-norm s (Proposal): delta * s / RMS(delta), the RMS a running mean square over
        TD_TAU frames, so the task update has RMS s on every task (s = CartPole's RMS keeps eta's M2 calibration).
        One scalar gain on the broadcast modulator; the critic still learns from the raw delta. 0 = off (M2)."""
        if not self.td_norm:
            return delta
        self.td_ms += (delta ** 2 - self.td_ms) / TD_TAU
        return delta * self.td_norm / np.sqrt(self.td_ms)

    def diagnostics(self):
        """Training and policy diagnostics since the last call (never touches rng or W): TD error |delta| and RMS, the
        task-term |delta|, critic V range, share of synapses at the weight bounds, and on the held-out stimuli the
        mean Eq. 1d policy, its mean entropy (bits) and the mean largest action probability."""
        st, n = self.stats, max(self.stats["n"], 1)
        P = np.array([core.policy(np.bincount(self.m, self.walk(v), self.K), self.roles,
                                  self.beta) for v in self.held])
        w = self.W[self.mask]
        out = dict(td_abs=st["td_abs"] / n, td_rms=float(np.sqrt(st["td_sq"] / n)), td_task_abs=st["td_task_abs"] / n,
                   V_min=float(st["V_min"]), V_max=float(st["V_max"]), frac_wmin=float((w <= WMIN + 1e-12).mean()),
                   frac_wmax=float((w >= 1 - 1e-12).mean()), P_mean=P.mean(0).tolist(),
                   P_entropy=float(-(P * np.log2(P)).sum(1).mean()), P_max=float(P.max(1).mean()))
        if self.c_ent:  # iteration 012: RMS ratio of the entropy term to the task update
            out.update(ent_rms_ratio=float(np.sqrt(self.reg_sq[1] / max(self.reg_sq[0], 1e-300))), c_ent=self.c_ent)
            self.reg_sq = np.zeros(2)
        self.stats = td_stats()
        return out

    def evaluate(self, kl_obs=20, kl_reps=50):
        """Flow-level metrics on held-out stimuli; spiking adds KL(Eq. 1d || race) on kl_obs held-out stimuli.
        Spiking: consumes self.rng, so LOG_EVERY, kl_obs and kl_reps change the training trajectory."""
        rows, W, m = [], self.W, self.m
        same = m[:, None] == m[None, :]
        persist = np.bincount(m, (W * same).sum(0) / W.sum(0), self.K) / np.bincount(m, minlength=self.K)
        P1d = []
        for v in self.held:
            pi = self.walk(v)
            p, q, JK, _ = core.module_stats(W, pi, m, self.K)
            rows.append([core.map_equation(p, q, pi), q[self.roles == "C"][0], p[self.A].sum(),
                         core.j_route(JK, self.roles)])
            P1d.append(core.policy(p, self.roles, self.beta))
        out: dict = dict(zip(("D", "q_C", "p_A", "J_route"), np.mean(rows, 0).tolist()))
        out.update(persist_C=float(persist[self.roles == "C"].mean()),
                   persist_LA=float(persist[self.roles != "C"].mean()), persist=persist.tolist(),
                   d_max=float(W.sum(0).max()), P1d_mean_right=float(np.mean(P1d, 0)[1]))
        if self.rule == "emergent":
            out.update(self.evaluate_emergent(out["D"]))
        if self.live:
            out.update(self.race_fidelity(P1d, kl_obs, kl_reps))
        return out

    def evaluate_emergent(self, D):
        """Rule 'emergent' evaluation on the held-out stimuli with W frozen: L_obs, the map equation of the engine's own
        observed walk (Eq. 8; flow: equal to D), averaged over the stimuli, one frame each from the (seed, EVAL_SEED,
        episode) stream (never the training rng), with the mean M*, e^obs, offspring spectral radius (Eq. 12) and spikes
        per head and frame; and the shares of the total weight by module block (block_shares)."""
        out = {"blocks": block_shares(self.W, self.m, self.roles)}
        if self.backend == "flow":
            return out | {"L_obs": D}
        m, K, rng, rows = self.m, self.K, np.random.default_rng((self.cfg.seed, EVAL_SEED, self.ep)), []
        fb = core.module_rows(self.W / self.W.sum(0), m, K)
        for v in self.held:
            pi, ob = obs_frame(*self.counter_frame(v, rng), self.heads)
            p, q, _, _, e_obs = core.observed_walk_modules(core.module_rows(ob.kraw, m, K), pi, m, K, fb)
            live = ob.kraw.sum(0) > 0
            rows.append([core.map_equation(p, q, pi), core.mstar(p, q).mean(),
                         e_obs[live].mean() if live.any() else np.nan, core.module_offspring(ob.c, p, m, K)[1],
                         ob.n / self.heads])
        keys = ("L_obs", "Mstar_eval", "e_obs_eval", "rho_eval", "spikes_per_head")
        return out | dict(zip(keys, (float(x) for x in np.nanmean(rows, 0))))

    STATE = ("beta", "r_fast", "r_slow", "r_var", "td_ms", "frames", "ep", "silent", "c_ent", "stats", "bound_n",
             "bound_dl", "bound_th", "elog", "ep_log", "reg_sq", "mpr_stats")

    def state(self):
        """Checkpoint state at an episode boundary (the trace e is 0 there and qif's state is reset at episode start):
        (arrays, meta JSON) with W, u, the running walk statistics, the MPR warm start, and every scalar, counter and
        rng state (training rng, which also seeds every env.reset; qif's stream)."""
        arr = {"W": self.W, "u": self.u}
        if self.run_obs is not None:
            arr.update(run_R=self.run_obs[0], run_pi=self.run_obs[1])
        if self.r_warm is not None:
            arr["r_warm"] = self.r_warm
        if self.lcarry is not None:  # engine='lif': the gain EMA / regression moments carried across episodes
            G, reg = self.lcarry
            arr.update({} if G is None else {"lcarry_G": G})
            arr.update({} if reg is None else {f"lcarry_reg_{k}": np.asarray(x) for k, x in reg.items()})
        if hasattr(self, "qgain"):  # engine='qif': the gain regression's running sums
            arr.update({f"qgain_{k}": getattr(self.qgain, k) for k in qif.GainEstimator.STATE})
        meta = {k: getattr(self, k) for k in self.STATE if hasattr(self, k)}
        meta["rng"] = self.rng.bit_generator.state
        if hasattr(self, "qrng"):
            meta["qrng"] = self.qrng.bit_generator.state
        return arr, json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

    def load_state(self, arr, meta):
        """Inverse of state() on a fresh Agent of the same Config."""
        meta = json.loads(meta)
        self.W, self.u = np.array(arr["W"]), np.array(arr["u"])
        if "run_R" in arr:
            self.run_obs = [np.array(arr["run_R"]), np.array(arr["run_pi"])]
        if "r_warm" in arr:
            self.r_warm = np.array(arr["r_warm"])
        if self.lp is not None and any(k.startswith("lcarry_") for k in arr):
            reg = {k[11:]: np.array(arr[k]) for k in arr if k.startswith("lcarry_reg_")}
            self.lcarry = (np.array(arr["lcarry_G"]) if "lcarry_G" in arr else None,
                           ({k: (int(x) if k == "n" else x) for k, x in reg.items()} if reg else None))
        for k in qif.GainEstimator.STATE if "qgain_sxr" in arr else ():
            setattr(self.qgain, k, np.array(arr[f"qgain_{k}"]) if k != "n" else int(arr["qgain_n"]))
        self.rng.bit_generator.state = meta.pop("rng")
        if "qrng" in meta:
            self.qrng.bit_generator.state = meta.pop("qrng")
        for k, v in meta.items():
            setattr(self, k, np.array(v) if k == "reg_sq" else v)

    def race_fidelity(self, P1d, kl_obs, kl_reps):
        """KL and L1 of the race policy (kl_reps races per held-out stimulus, add-1/2 estimate) against Eq. 1d, and
        the KL floor the race would show if it were exactly Eq. 1d. Draws from self.rng."""
        kl, l1, floor, qstate, lstate = [], [], [], self.qstate, self.lstate
        smooth = lambda n: (n + 0.5) / (kl_reps + 1)  # noqa: E731
        for v, P in zip(self.held[:kl_obs], P1d):
            if self.qp is not None:  # qif: reset state, one warm-up frame
                self.qstate = None
                if self.qp.warm:
                    self.spike_frame(v)
            self.lstate = None  # lif: reset state (the first frame warms up inside)
            races = [sp.race(self.spike_frame(v), self.race_groups, self.rng, self.burn) for _ in range(kl_reps)]
            Q = smooth(np.bincount([r for r in races if r is not None], minlength=len(self.A)))
            kl.append(float((P * np.log(P / Q)).sum()))
            l1.append(float(np.abs(P - Q).sum()))
            Qs = smooth(self.rng.multinomial(kl_reps, P, 200))
            floor.append(float((P * np.log(P / Qs)).sum(1).mean()))
        self.qstate, self.lstate = qstate, lstate
        return dict(KL_1d_race=float(np.mean(kl)), KL_floor=float(np.mean(floor)), L1_1d_race=float(np.mean(l1)))


def block_shares(W, m, roles):
    """Shares of the total weight sum(W) by module block (post b, pre a): within a module, and across modules C->A,
    C->L, A<->A, A<->L (both directions), L<->L, and into the controller (L, A -> C)."""
    K = len(roles)
    H = np.eye(K)[m]
    WK = H.T @ W @ H / W.sum()
    rb, ra = roles[:, None], roles[None, :]
    cross = ~np.eye(K, dtype=bool)
    cls = {"within": ~cross, "C->A": (ra == "C") & (rb == "A"), "C->L": (ra == "C") & (rb == "L"),
           "A<->A": cross & (ra == "A") & (rb == "A"),
           "A<->L": ((ra == "A") & (rb == "L")) | ((ra == "L") & (rb == "A")),
           "L<->L": cross & (ra == "L") & (rb == "L"), "->C": cross & (rb == "C")}
    return {k: float(WK[c].sum()) for k, c in cls.items()}


def match_scales(cfg, n_frames=None):
    """Step-size matching (a pre-set rule, no tuning): with W at its initial value (frozen), the RMS over synapses of
    the structural increment (g - gbar) kappa (rule v4) or (g - gbar^obs) c (rule emergent) and of the Eq. 4b
    eligibility increment, over the first n_frames held-out stimuli (action module f mod |A| for frame f), for cfg's
    engine and for the flow engine on the same stimuli. Returns lam_scale = RMS_struct(flow) / RMS_struct(engine),
    eta_scale = RMS_task(flow) / RMS_task(engine) and the RMS values; the engine's frames draw from the (seed,
    MATCH_SEED) stream. lam_scale and eta_scale are Config fields; QIFParams' / LIFParams' own scales should then
    stay 1. n_frames default: 50 for spiking engines (the QIF / LIF gain estimates need them to settle; the QIF match
    moved by up to 10% between 20 and 50 frames), 20 for flow."""
    n_frames = n_frames or (50 if cfg.backend == "spiking" else 20)
    flow_cfg = dataclasses.replace(cfg, backend="flow", heads=1, engine="snm", readout="race", lam_scale=1.0,
                                   eta_scale=1.0, obs_tau=0.0, qif={}, lif={})
    ref, eng = Agent(flow_cfg), Agent(dataclasses.replace(cfg, lam_scale=1.0, eta_scale=1.0))
    assert (ref.W == eng.W).all()
    eng.rng = np.random.default_rng((cfg.seed, MATCH_SEED))
    ss, mask = np.zeros(4), ref.mask
    for f, v in enumerate(ref.held[:n_frames]):
        a = f % len(ref.A)
        pi, J = ref.walk(v), ref.W / ref.W.sum(0) * ref.walk(v)
        dS, de = ref.increments(pi, J, a)[:2]
        if eng.backend == "flow":
            pe, ke = pi, J
        else:
            pe, ke, _ = eng.frame(v)
        if eng.rule == "emergent":
            dSe, dee = eng.increments(pe, ke, a)[:2]
        else:
            p, q, JK, _ = core.module_stats(eng.W, pe, eng.m, eng.K)
            g = core.modulator(p, q, eng.roles, routing=eng.routing)
            if eng.la_beta > 0:
                g = g - core.latent_output_barrier(p, JK, eng.roles, eng.la_beta)
            dSe = -core.centered(g, eng.W, eng.m) * ke
            dee = core.eligibility_step(0.0, eng.W, ke, pe, eng.m, eng.roles, eng.A[a], 1.0, eng.beta)
        ss += [(x[mask] ** 2).sum() for x in (dS, dSe, de, dee)]
    rms = np.sqrt(ss / (n_frames * mask.sum()))
    return dict(lam_scale=float(rms[0] / rms[1]), eta_scale=float(rms[2] / rms[3]), rms_struct_flow=float(rms[0]),
                rms_struct=float(rms[1]), rms_task_flow=float(rms[2]), rms_task=float(rms[3]), n_frames=n_frames)


def default_name(cfg):
    """The m2 result name of cfg (m2_<backend>_<routing>_lam<lam>[...]_<MODEL>_s<seed>)."""
    b = (f"_b{cfg.beta:g}" if cfg.beta != 1 else "") + (f"_eta{cfg.eta:g}" if cfg.eta else "")
    b += f"_at{cfg.eta_tau:g}" if cfg.eta_tau is not None else ""
    b += f"_lb{cfg.la_beta:g}" if cfg.la_beta else ""
    return f"m2_{cfg.backend}_{cfg.routing}_lam{cfg.lam:g}{b}_{MODEL}_s{cfg.seed}"


def save_checkpoint(ck, agent, returns, evals, snaps, resumes, wall):
    """Write <out>.ckpt.npz atomically: the agent's state(), the returns and evaluations so far, the model snapshots,
    the resume count and the wall time."""
    arr, meta = agent.state()
    tmp = ck.with_name(ck.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, meta=meta, run=json.dumps(dict(returns=returns, evals=evals, resumes=resumes, wall=wall)),
                 **arr, **{f"snap_{k}": x for k, x in snaps.items()})
    tmp.replace(ck)


def load_checkpoint(ck, agent):
    """Restore save_checkpoint's file into a fresh agent; returns (returns, evals, snaps, resumes, wall)."""
    with np.load(ck) as z:
        arr = {k: z[k] for k in z.files if k not in ("meta", "run") and not k.startswith("snap_")}
        agent.load_state(arr, str(z["meta"]))
        r = json.loads(str(z["run"]))
        snaps = {k[5:]: z[k] for k in z.files if k.startswith("snap_")}
    return r["returns"], r["evals"], snaps, r["resumes"], r["wall"]


def frozen_returns(agent, n=FROZEN_EPIS):
    """Returns of n episodes with the final W frozen (rule 'emergent'; a training return can hide a W that has just
    collapsed onto one action, iteration 014 §7): a deep copy of the agent acts through frame() as in episode(), with
    no learning (W, critic and beta fixed), its frames and environment seeds from (seed, FROZEN_SEED); the agent itself
    is untouched."""
    a = copy.deepcopy(agent)
    a.rng = np.random.default_rng((a.cfg.seed, FROZEN_SEED))
    env, out = gym.make(a.task.env_id), []
    for _ in range(n):
        o, _ = env.reset(seed=int(a.rng.integers(1 << 30)))
        a.qstate = a.lstate = None
        R, done = 0.0, False
        while not done:
            o, r, term, trunc, _ = env.step(a.frame(a.stim(o))[2])
            R, done = R + r, term or trunc
        out.append(float(R))
    return out


def save_model(path, agent, snaps, cfg):
    """--save-model: compressed float32 .npz with W_ep<k> (episode 0, every SAVE_EVERY and the end), mask, m, roles,
    the Config as JSON and the final critic u."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: x.astype(np.float32) for k, x in snaps.items()}, mask=agent.mask, m=agent.m,
                        roles=agent.roles, config=json.dumps(dataclasses.asdict(cfg)), u=agent.u.astype(np.float32))


def run(cfg, episodes, name=None, out=None, save_model_path=None, scales=None):
    """Train for `episodes`, evaluating every LOG_EVERY (evaluate() plus diagnostics()), and write the result file
    (`out`, a .json path, or OUT/<name>.json, default name m2_*.json). Iteration-012 runs add end-of-run diagnostics
    (diag012) over the last REC_EPIS episodes. Checkpoints: every LOG_EVERY episodes <out>.ckpt.npz (save_checkpoint);
    a run that finds one resumes from it (not bit-identical in general; counted in the result's `resumes`) and deletes
    it on completion. save_model_path: save_model at the end. scales: match_scales' record, stored in the result."""
    agent = Agent(cfg)
    new012 = bool(cfg.encoder or cfg.critic_layer or cfg.c_ent or cfg.ent_target)
    env = gym.make(agent.task.env_id)
    fn = pathlib.Path(out) if out else OUT / f"{name or default_name(cfg)}.json"
    fn.parent.mkdir(parents=True, exist_ok=True)
    ck = fn.with_suffix(".ckpt.npz")
    if ck.exists():
        returns, evals, snaps, resumes, wall = load_checkpoint(ck, agent)
        resumes += 1
        print(f"resumed from {ck.name} at episode {len(returns)}", flush=True)
    else:
        returns, evals, resumes, wall = [], [{"episode": 0, **agent.evaluate(), **agent.diagnostics()}], 0, 0.0
        snaps = {"W_ep0": agent.W.astype(np.float32)} if save_model_path else {}
    t0 = time.perf_counter() - wall
    for ep in range(len(returns) + 1, episodes + 1):
        if new012 and agent.rec is None and ep >= episodes - REC_EPIS + 1:
            agent.rec = []
        returns.append(agent.episode(env))
        if save_model_path and (ep % SAVE_EVERY == 0 or ep == episodes):
            snaps[f"W_ep{ep}"] = agent.W.astype(np.float32)
        if ep % LOG_EVERY == 0:
            evals.append({"episode": ep, **agent.evaluate(), **agent.diagnostics()})
            if cfg.beta_kappa:
                evals[-1]["beta"] = agent.beta
            ev = evals[-1]
            kl = f"KL {ev['KL_1d_race']:.3f} " if cfg.backend == "spiking" else ""
            print(f"ep {ep} last50 {np.mean(returns[-50:]):.1f} D {ev['D']:.3f} p_A {ev['p_A']:.3f} "
                  f"persist_LA {ev['persist_LA']:.3f} H(P) {ev['P_entropy']:.2f} |td| {ev['td_abs']:.3f} "
                  f"d_max {ev['d_max']:.2f} " + kl + f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            if ep < episodes:
                save_checkpoint(ck, agent, returns, evals, snaps, resumes, time.perf_counter() - t0)
    frames = agent.frames
    res = {"backend": cfg.backend, "routing": cfg.routing, "lam": cfg.lam, "seed": cfg.seed, "episodes": episodes,
           "returns": returns, "model": MODEL, "beta": cfg.beta, "eta_tau": agent.eta_tau, "eta_task": agent.eta,
           "la_beta": cfg.la_beta, "td_norm": cfg.td_norm, "ablate": cfg.ablate, "p_cc": cfg.p_cc,
           **({"p_cross": cfg.p_cross} if cfg.p_cross != 0.05 else {}),
           "beta_kappa": cfg.beta_kappa, "beta_max": cfg.beta_max,
           **({"gamma": cfg.gamma} if cfg.gamma != GAMMA else {}),
           **({"readout": cfg.readout, "bound_deadline_frac": agent.bound_dl / max(agent.bound_n, 1),
               "bound_theta_mean": agent.bound_th / max(agent.bound_n, 1)} if cfg.readout != "race" else {}),
           "evals": evals, "frames": frames, "ms_per_frame": (time.perf_counter() - t0) / frames * 1e3,
           "silent_race_frames": agent.silent, "resumes": resumes,
           "params": dict(alpha=ALPHA, n=N_HOPS, k=K_SPK, rate=RATE, theta=THETA, eta=ETA, tau_e=TAU_E,
                          gamma=cfg.gamma, eta_v=ETA_V, eps=0.0, eps_h=agent.eps_h, burn=agent.burn, eta_task=agent.eta,
                          beta=cfg.beta, eta_tau=agent.eta_tau, sizes=agent.task.sizes)}
    if (cfg.task, cfg.heads) != ("CartPole", 1):
        res.update(task=cfg.task, heads=cfg.heads, engine=cfg.engine, encoder=agent.task.encoder,
                   n_ctrl=agent.task.n_ctrl)
    if agent.qp is not None:
        res.update(qif=dataclasses.asdict(agent.qp))
    if agent.lp is not None:
        res.update(lif=dataclasses.asdict(agent.lp))
    if cfg.backend == "mpr":
        res.update(mpr_stats=agent.mpr_stats)
    if cfg.rule != "v4":
        res.update(rule=cfg.rule, obs_tau=cfg.obs_tau, ebar=cfg.ebar, step_stat=cfg.step_stat, emergent=agent.ep_log,
                   frozen_returns=frozen_returns(agent))
    if cfg.lam_scale != 1.0 or cfg.eta_scale != 1.0 or scales:
        res.update(lam_scale=cfg.lam_scale, eta_scale=cfg.eta_scale, **({"scales": scales} if scales else {}))
    if new012:
        res.update({k: v for k, v in (("encoder_spec", cfg.encoder), ("critic_layer", cfg.critic_layer),
                                      ("c_ent", cfg.c_ent), ("ent_target", cfg.ent_target)) if v},
                   n_ctrl=agent.task.n_ctrl, diag012=diag012(agent))
    fn.write_text(json.dumps(res))
    if save_model_path:
        save_model(save_model_path, agent, snaps, cfg)
    ck.unlink(missing_ok=True)
    return res


def r2_cv(X, y, ridge=1e-3):
    """Two-fold cross-validated R^2 of a linear (ridge on z-scored X, with intercept) fit of y on X (fixed split)."""
    idx = np.random.default_rng(0).permutation(len(y))
    X1 = np.c_[np.ones(len(y)), (X - X.mean(0)) / (X.std(0) + 1e-12)]
    pred = np.zeros(len(y))
    for tr, te in ((idx[::2], idx[1::2]), (idx[1::2], idx[::2])):
        A = X1[tr].T @ X1[tr] + ridge * len(tr) * np.eye(X1.shape[1])
        pred[te] = X1[te] @ np.linalg.solve(A, X1[tr].T @ y[tr])
    return float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300))


def diag012(agent, n=2000):
    """Iteration-012 end-of-run diagnostics on up to n observations from the last REC_EPIS episodes (never touches
    agent.rng or W): controller density (share of controller neurons with v_i > 10% of max v), the critic's V on those
    states, and how well V decodes linearly from module flow shares: p_b over latent and action modules (R2_LA), the
    controller-share features of the default critic (R2_ctrl, the V of a pi_C critic) and neuron-level L/A shares
    (R2_LA_neurons, ridge)."""
    obs = agent.rec or []
    obs = [obs[i] for i in np.linspace(0, len(obs) - 1, min(n, len(obs))).astype(int)] if obs else []
    agent.rec = None
    if len(obs) < 20:
        return {}
    LA, nLA = np.isin(agent.roles, ["L", "A"]), np.isin(agent.roles[agent.m], ["L", "A"])
    dens, P, PN, PC, V = [], [], [], [], []
    for o in obs:
        v = agent.stim(o)
        vc = v[agent.ctrl]
        dens.append(float((vc > 0.1 * vc.max()).mean()))
        pi = core.flow(agent.W, v, ALPHA, N_HOPS)
        P.append(np.bincount(agent.m, pi, agent.K)[LA])
        PN.append(pi[nLA])
        PC.append(pi[agent.ctrl] / pi[agent.ctrl].sum())
        V.append(float(agent.u @ agent.feats(pi, o)))
    V = np.array(V)
    held = [float((v[agent.ctrl] > 0.1 * v[agent.ctrl].max()).mean()) for v in agent.held]
    return dict(n=len(obs), density=float(np.mean(dens)), density_held=float(np.mean(held)), V_sd=float(V.std()),
                R2_LA=r2_cv(np.array(P), V), R2_ctrl=r2_cv(np.array(PC), V), R2_LA_neurons=r2_cv(np.array(PN), V, 1e-2))


def summary():
    """Results table over seeds of every m2_*.json in OUT (v1 and v2 files included), grouped by model, backend,
    routing, lam and variant."""
    rand = random_return()
    print(f"random policy: {rand[0]:.1f} +- {rand[1]:.1f} (1000 episodes)")
    groups = {}
    for f in sorted(glob.glob(str(OUT / "m2_*.json"))):
        r = json.loads(pathlib.Path(f).read_text())
        var = f"b{r.get('beta', 1.0):g}" + (f",lb{r['la_beta']:g}" if r.get("la_beta") else "")
        if r.get("eta_tau") is not None and r["model"] != "v3":
            var += f",eta{r['eta_task']:g}" + (f",at{r['eta_tau']:g}" if r["eta_tau"] else "")
        groups.setdefault((r.get("model", "v1"), r["backend"], r["routing"], r["lam"], var), []).append(r)
    cols = ("D", "J_route", "p_A", "persist_LA", "persist_C", "d_max", "KL_1d_race", "KL_floor")
    print("return: mean +- sd over seeds of per-seed mean return; metrics: seed-mean at episode 0 > final eval (sd)")
    print(f"{'model':5} {'backend':8} {'routing':7} {'lam':>5} {'variant':>14} {'n':>2} {'epis':>5} "
          f"{'ret first100':>13} {'ret last100':>13} " + " ".join(f"{c:>18}" for c in cols))
    ms = lambda x: f"{np.mean(x):6.1f}+-{np.std(x):5.1f}"  # noqa: E731
    for (mo, b, ro, lam, be), rs in sorted(groups.items()):
        line = f"{mo:5} {b:8} {ro:7} {lam:5g} {be:>14} {len(rs):2} {min(r['episodes'] for r in rs):5} " \
               f"{ms([np.mean(r['returns'][:100]) for r in rs])} {ms([np.mean(r['returns'][-100:]) for r in rs])}"
        for c in cols:
            if c in rs[0]["evals"][-1]:
                s, e = [r["evals"][0][c] for r in rs], [r["evals"][-1][c] for r in rs]
                line += f" {np.mean(s):5.3f}>{np.mean(e):5.3f}({np.std(e):.3f})"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("flow", "spiking"), default="flow")
    ap.add_argument("--routing", choices=ROUTINGS, default="exempt")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=None, help="override the task rate")
    ap.add_argument("--eta-tau", type=float, default=None, help="override the annealing time constant (0 = constant)")
    ap.add_argument("--la-beta", type=float, default=0.0, help="latent-output barrier (Eq. 4e, Proposal)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    cfg = Config(a.backend, a.routing, a.lam, a.seed, beta=a.beta, eta=a.eta, eta_tau=a.eta_tau, la_beta=a.la_beta)
    r = run(cfg, a.episodes)
    print(f"done: last100 {np.mean(r['returns'][-100:]):.1f}, {r['frames']} frames, {r['ms_per_frame']:.2f} ms/frame")


if __name__ == "__main__":
    main()
