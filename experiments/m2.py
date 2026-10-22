"""M2 (docs/SPEC.md §11): CartPole in closed loop with Map-STDP, models v1-v3 (docs/model.md).
Run from the repo root:
  python -m experiments.m2 --backend {flow,spiking} --routing {exempt,none,dual} --lam 0.05 --seed 0 --episodes 1500
    [--model {v1,v2,v3}] [--beta B]   -> experiments/results/m2_<backend>_<routing>_lam<lam>[_b<B>]_<model>_s<seed>.json
Model v1 (iteration 002 baseline): eps inside lambda, race from step 0, eta = 10.
Model v2: d_j homeostasis as its own term (EPS_H, every frame), race burn-in BURN steps, spiking eta = 5.
Model v3 (default): v2 with eta = 10 in both backends, annealed as eta / (1 + episode / ETA_TAU).
--eta-tau T: task rate annealed as eta / (1 + episode / T) (ablation; 0 = constant).
--beta: sharpened policy P ~ p_a^beta with the matched score (Eq. 1e, Proposal); flow backend only.
--la-beta B: latent-output barrier (Eq. 4e, Proposal) subtracted from g on latent->action pairs; 0 (default) is off.
Agent(p_cc=...): controller->controller connection probability of the SBM (core.build_network; 0.1 = M2). p_cc = 0
  removes the C->C recurrence, whose neuron-level credit the module-level Eq. 4b cannot represent (iteration 006).
Agent(beta_end=B, beta_ramp=R): beta ramped linearly from beta to B over the first R episodes (default half the
  task's episodes), then held (flow only). ORACLE reference ablation, not part of the model: it needs the run length.
Agent(beta_kappa=k, beta_max=B): causal adaptive gain (Proposal, flow only), one global scalar: after each episode
  log beta += k z, z = (R_fast - R_slow) / sd(R) clipped to [-2, 2] (fast / slow running returns over BETA_TAU_F /
  BETA_TAU_S episodes), beta kept in [beta, B]. It rises while return improves and falls when it drops.
Agent(beta_eps=e[, beta_set=B0]) (Proposal, iteration 007): mean reversion, log beta += e (log s - log beta) per update.
  beta_set B0 > 0: a fixed set point s = B0, both ways (homeostatic tone). beta_set 0: an experience-dependent set point
  (Eq. 1g'), pulling up only: s rises at once to the slow average of log beta (BETA_TAU_S episodes, BETA_TS frames) and,
  with beta_decay D > 0, leaks back toward it over D episodes (0: a strict running maximum). beta returns toward the
  gain it had sustained after a fall, and the update is unchanged while beta is at or above s.
Agent(beta_signal='trace') (Proposal, iteration 007): beta updated every frame, no episode boundaries. Utility is the
  reset-at-termination discounted reward trace G_f = r_f + gamma G_{f-1} (G_{f-1} := 0 after a terminal frame); z_f is
  the trend of its bias-corrected running means over BETA_TF and BETA_TS frames, over the spread of its BETA_TH-frame
  mean, clipped to [-2, 2]; log beta += beta_kappa z_f per frame (beta_kappa is then per frame) after BETA_TF frames.
Agent(score_unit=True): ablation, the Eq. 4b increment divided by beta (matched direction, beta-free magnitude).
Agent(beta_step_cap=c[, beta_step_shape=S]) (Proposal, iteration 008): the Eq. 4b increment scaled by beta_eta / beta,
  a beta-dependent step size on the same matched-score direction, while the policy (Eq. 1e) keeps the full beta.
  beta_eta: 'min' (default) min(beta, c), a hard cap; 'sat' c beta / (c + beta), smooth saturation (half at beta = c);
  'invu' beta exp(1 - beta / c), an inverted U peaking at beta = c; 'hyb' beta up to c, then c (1 + log(beta / c))
  (continuous, slope 1 at c). c = 0 (default) is off. Excludes score_unit.
ablate critic='net': critic features from network activity, the flow shares of controller and latent neurons.
Agent(critic_lambda=l) (Proposal, iteration 009): TD(lambda) critic, Eq. 4c with the trace z <- gamma l z + x and the
  normalised step u += eta_V delta z / (x . x); z is periphery state, reset each episode; l = 0 is Eq. 4c bit for bit.
Agent(gamma=g) (Proposal, iteration 009): discount of the critic and of delta (Eq. 4c); default GAMMA = 0.99. The
  beta_signal='trace' reward trace keeps GAMMA (a separate quantity); the task trace e uses tau_e, not gamma.
Agent(readout='diffbound') (Proposal, iteration 010; spiking only): the action is the count-difference race to a bound
  (sp.diffbound, at most one spike per action module and head per step) instead of the race. beta (fixed, or adaptive
  with beta_kappa) is the target sharpness: theta = BOUND_CAL^-1(beta), log-log interpolation of the measured
  beta_eff(theta) table for the agent's heads, clipped to the table, randomised between floor and ceil for a fractional
  theta. The Eq. 4b score uses beta (matched where the table reaches beta; below its first entry the readout is sharper
  than the score). Draws one extra uniform per frame, only on this path.
  python -m experiments.m2 --summary   -> results table over seeds (plus the random-policy return)
Both backends share one frame update; they differ only in where pi, kappa and the action come from:
  flow:    pi = n-hop flow, kappa = J (expected pairings), action ~ Eq. 1d;
  spiking: pi = spike shares, kappa = covariance count / spikes in the frame, action = race readout.
Dividing kappa by the frame's spike count is the 1/(spikes per frame) scaling of lambda and eta (model.md §4).
Iteration 012 (Proposals, flow; all off by default, none draws from rng):
Agent(encoder=SPEC): the controller encoder replaced by tasks.with_encoder(task, SPEC) (grid g<n>s<sigma> or Fourier
  f<order>c<conj>[h][b]); the controller size becomes the feature count.
Agent(critic_layer=SPEC): separate critic, V = u . phi(o) on its own fixed RF layer (tasks.critic_layer), NLMS as
  Eq. 4c; the actor's flow is not used by the critic.
Agent(c_ent=c): C1, entropy term, a separate per-frame ungated update W += eta_e c kappa (g_b / J^in_b - gbar_j)
  (core.entropy_score; average W * grad H(P~), P~ the beta = 1 action shares, pi fixed); eta_e the annealed task rate.
Agent(c_use=c[, use_tau=T]): C2, usage homeostasis, W -= eta_e c kappa ((ubar_b - 1/|A|) / J^in_b - baseline_j)
  (core.usage_score), ubar the running average of P~ over T frames.
Agent(ent_target=f[, ent_k=k]): c_ent adapted SAC-style, log c_ent += k (f log|A| - H(P~)) per frame (nats)."""
import argparse
import dataclasses
import glob
import json
import pathlib
import time

import gymnasium as gym
import numpy as np

from experiments.tasks import TASKS, random_states, with_encoder
from experiments.tasks import critic_layer as critic_layer_fn
from mapstdp import core
from mapstdp import spiking as sp

ALPHA, N_HOPS, K_SPK, RATE, THETA = 0.2, 20, 100, 1.0, 0.5
ETA, TAU_E, GAMMA, ETA_V, EPS, ETA_MU = 10.0, 3.0, 0.99, 0.1, 1.0, 2.0
ETA_SPK, EPS_H, BURN = 5.0, 5.0, 10  # v2: spiking task rate, d_j homeostasis rate (own term), race burn-in (steps)
ETA_TAU = 300.0  # v3: task-rate annealing time constant (episodes)
USE_TAU, ENT_K = 300.0, 1e-3  # iteration 012: usage-average time constant (frames), entropy-target gain
REC_EPIS = 20  # iteration 012: episodes recorded at the end of a run for diag012
BETA_TAU_F, BETA_TAU_S = 20.0, 100.0  # beta_kappa (Proposal): fast and slow running episode return (episodes)
BETA_TF, BETA_TS, BETA_TH = 4000.0, 20000.0, 300.0  # beta_signal='trace': fast, slow and spread time constants (frames)
TD_TAU = 1000.0  # --td-norm (Proposal): time constant (frames) of the running mean square of delta
RICH_N = 256  # ablation critic='rich': RFs on the full observation (tasks 'pool' encoder)
RICH_MR = (np.zeros(RICH_N, int), np.array(["C"]))  # module labels / roles for a stand-alone RF layer
Q_STAR_FRAC = 0.75  # dual routing target q* as a fraction of the held-out mean J_route at W_0 (Eq. 4a)
BOUND_CAL = {16: ((1, 3.1), (2, 3.5), (4, 4.0), (8, 5.0), (16, 6.5), (32, 9.2), (64, 11.5))}  # heads: (theta, beta_eff)
LOG_EVERY = 50  # module sizes are per task (tasks.Task.sizes; CartPole 36, 16, 16, 16, 16)
OUT = pathlib.Path(__file__).parent / "results"


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


def rich_critic(task):
    """A 'pool' encoder with RICH_N centers over the task's full observation (critic='rich' ablation)."""
    return dataclasses.replace(task, n_ctrl=RICH_N, encoder="pool", enc={})


class Agent:
    """Closed-loop Map-STDP agent: structural (Eq. 4) with optional routing dual (Eq. 4a), d_j homeostasis (Eq. 4d)
    and TD task terms (Eqs. 4b, 4c; task rate annealed in v3). task: a tasks.TASKS key (M3); heads: H parallel
    spiking replicas on shared W whose spikes are pooled for pi, kappa and the race (Proposal, spiking only)."""

    def __init__(self, backend, routing, lam, seed, model="v3", beta=1.0, eta_tau=None, la_beta=0.0,
                 task="CartPole", heads=1, engine="snm", td_norm=0.0, ablate=None, p_cc=0.1, beta_end=None,
                 beta_ramp=None, beta_kappa=0.0, beta_max=4.0, score_unit=False, beta_eps=0.0, beta_set=0.0,
                 beta_signal="episode", beta_decay=0.0, beta_step_cap=0.0, beta_step_shape="min", critic_lambda=0.0,
                 gamma=GAMMA, readout="race", encoder=None, critic_layer=None, c_ent=0.0, c_use=0.0,
                 use_tau=USE_TAU, ent_target=None, ent_k=ENT_K):
        self.task, self.heads, self.engine = TASKS[task], heads, engine
        if encoder:  # iteration 012: controller encoder spec (tasks.with_encoder)
            self.task = with_encoder(self.task, encoder)
        assert not (encoder or critic_layer or c_ent or c_use or ent_target) or backend == "flow", "iteration 012: flow"
        self.encoder, self.critic_layer, self.c_ent, self.c_use, self.use_tau = encoder, critic_layer, c_ent, c_use, \
            use_tau
        self.ent_target, self.ent_k, self.reg_sq = ent_target, ent_k, np.zeros(3)  # task, entropy, usage sq sums
        self.crit = critic_layer_fn(self.task, critic_layer) if critic_layer else None
        assert ent_target is None or c_ent > 0, "ent_target adapts c_ent multiplicatively: start it above 0"
        ab = dict(wmin=1e-3, tau_e=TAU_E, critic="ctrl") | (ablate or {})  # M3 learning-rule ablations (iteration 005)
        self.ablate, self.wmin, self.tau_e = ab, ab["wmin"], ab["tau_e"]
        self.rich = rich_critic(self.task) if ab["critic"] == "rich" else None
        self.td_norm, self.td_ms, self.frames = td_norm, td_norm ** 2, 0  # td_ms starts at the reference: no rescale
        self.stats = dict(n=0, td_abs=0.0, td_sq=0.0, td_task_abs=0.0, V_min=np.inf, V_max=-np.inf)
        assert heads == 1 or backend == "spiking", "heads are spiking replicas"
        assert engine in ("snm", "numpy"), "numpy: sp.run_heads, the batched branching step (same raster as snm)"
        self.W, self.m, self.roles, self.mask = core.build_network(self.task.sizes, self.task.roles, p_cc=p_cc,
                                                                   seed=seed)
        self.K, self.backend, self.routing, self.lam = len(self.roles), backend, routing, lam
        v2 = model in ("v2", "v3")
        if eta_tau is None:
            eta_tau = ETA_TAU if model == "v3" else 0.0
        self.eps, self.eps_h, self.burn = (0.0, EPS_H, BURN) if v2 else (EPS, 0.0, 0)
        self.eta = ETA_SPK if model == "v2" and backend == "spiking" else ETA
        self.beta, self.eta_tau, self.ep, self.la_beta = beta, eta_tau, 0, la_beta
        bound = readout == "diffbound"
        assert readout == "race" or (bound and backend == "spiking" and heads in BOUND_CAL), "diffbound: spiking, cal H"
        assert beta == 1.0 or backend == "flow" or bound, "the race readout has no matched sharpened policy"
        self.readout, self.bound_n, self.bound_dl, self.bound_th = readout, 0, 0, 0.0
        self.p_cc, self.beta0, self.beta_end = p_cc, beta, beta_end
        self.beta_ramp = (beta_ramp or self.task.episodes / 2) if beta_end is not None else None
        assert beta_end is None or backend == "flow", "the beta ramp sharpens Eq. 1e: flow only"
        assert not beta_kappa or ((backend == "flow" or bound) and beta_end is None), "adaptive beta: flow, no ramp"
        self.beta_kappa, self.beta_max, self.score_unit, self.r_fast = beta_kappa, beta_max, score_unit, None
        assert beta_signal in ("episode", "trace") and (beta_kappa or not (beta_eps or beta_signal == "trace"))
        self.beta_eps, self.beta_set, self.beta_signal, self.beta_decay = beta_eps, beta_set, beta_signal, beta_decay
        assert not (beta_step_cap and score_unit), "beta_step_cap and score_unit both rescale the Eq. 4b increment"
        assert beta_step_shape in ("min", "sat", "invu", "hyb") and (beta_step_cap or beta_step_shape == "min")
        self.beta_step_cap, self.beta_step_shape = beta_step_cap, beta_step_shape
        self.log_set = np.log(beta_set) if beta_set else np.log(beta)  # set point s (log)
        self.log_avg, self.trace, self.tr_ema = np.log(beta), 0.0, np.zeros((4, 2))  # trace EMAs f, s, h, var
        self.A = np.flatnonzero(self.roles == "A")
        self.ctrl = self.roles[self.m] == "C"
        self.groups = [np.flatnonzero(self.m == a) for a in self.A]
        self.race_groups = sp.head_groups(self.groups, heads, len(self.m))
        self.rng = np.random.default_rng(seed)
        self.net = np.isin(self.roles[self.m], ["C", "L"]) if ab["critic"] == "net" else None
        n_u = RICH_N if self.rich is not None else (self.ctrl if self.net is None else self.net).sum()
        if self.crit is not None:
            n_u = len(self.crit(held_out_obs(1, task=self.task)[0]))
        self.ubar = np.full(len(self.A), 1 / len(self.A))  # C2: running average of the beta = 1 action shares
        self.u, self.mu, self.silent = np.zeros(n_u), 0.0, 0
        self.critic_lambda, self.gamma, self.z = critic_lambda, gamma, 0.0  # z: Eq. 4c trace, periphery
        self.held = [self.stim(o) for o in held_out_obs(task=self.task)]
        self.rec = None  # list: record observations (iteration 012 end-of-run diagnostics; no rng)
        self.snn, self.live = None, False  # live after q_star, so this evaluate() runs no races, draws nothing from rng
        self.q_star = Q_STAR_FRAC * self.evaluate()["J_route"]  # ponytail: fixed, learn it if dual matters
        if backend == "spiking":
            self.live = True
            if engine == "snm":
                self.snn = sp.build(sp.tile_heads(self.W, heads), 1 - ALPHA, theta=THETA)

    def stim(self, o):
        """v(o): the task's encoder over the controller (tasks.Task.stim; CartPole: core.cartpole_stimulus)."""
        return self.task.stim(o, self.m, self.roles)

    def frame(self, v):
        """Walk and act for one frame (derivation §8 steps 2-3; spiking is cold-started, pi from spike shares).
        Returns pi, kappa (flow: expected pairings J = T * pi; spiking: covariance pairings / total spikes) and the
        action index into A (flow: sampled from Eq. 1d / 1e; spiking: race after burn-in, uniform if silent)."""
        W = self.W
        if self.backend == "flow":
            pi = core.flow(W, v, ALPHA, N_HOPS)
            p = np.bincount(self.m, pi, self.K)
            return pi, W / W.sum(0) * pi, int(self.rng.choice(len(self.A), p=core.policy(p, self.roles, self.beta)))
        S = self.spike_frame(v)
        a = sp.race(S, self.race_groups, self.rng, self.burn) if self.readout == "race" else self.bound(S)
        if a is None:
            self.silent, a = self.silent + 1, int(self.rng.integers(len(self.A)))
        pi, kappa, n = sp.pooled(S, self.heads)
        return pi, kappa / max(n, 1), a

    def bound(self, S):
        """readout='diffbound' (module doc): theta from beta through BOUND_CAL, then sp.diffbound; None if silent."""
        th_cal, b_cal = np.log(np.array(BOUND_CAL[self.heads], float)).T
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
        if self.engine == "numpy":
            return sp.run_heads(self.W, v, self.heads, K_SPK, RATE, self.rng, THETA, 1 - ALPHA)
        sp.set_weights(self.snn, sp.tile_heads(self.W, self.heads), 1 - ALPHA)  # output gain (1 - alpha) / d_j
        return sp.run_frame(self.snn, np.tile(v, self.heads), K_SPK, RATE, self.rng, THETA)[0]

    def feats(self, pi, o=None):
        """Critic features x = pi_C / p_C (controller flow shares), so V = u . x. Ablation critic='rich': RICH_N
        Gaussian RFs on the full observation, outside the network (diagnostic only: is the critic the bottleneck?)."""
        if self.crit is not None:
            return self.crit(o)
        if self.rich is not None:
            return self.rich.stim(o, *RICH_MR)[:RICH_N]
        if self.net is not None:
            return pi[self.net] / max(pi[self.net].sum(), 1e-12)
        return pi[self.ctrl] / max(pi[self.ctrl].sum(), 1e-12)

    def pre_td_updates(self, pi, kappa, a, e):
        """Per-frame updates that need no TD error (derivation §8 steps 4-5), in this order, which tests/golden.json
        locks: module statistics on the current W; eligibility increment (Eq. 4b, its h-bar uses the pre-update W);
        structural term (Eq. 4); d_j homeostasis (Eq. 4d); routing dual mu (Eq. 4a, from the pre-update JK).
        Mutates self.W and self.mu; returns the updated trace e."""
        p, q, JK, _ = core.module_stats(self.W, pi, self.m, self.K)
        reg = self.regulariser(pi, kappa) if (self.c_ent or self.c_use or self.ent_target) else None
        kap = kappa / self.beta if self.score_unit else kappa  # Eq. 4b is linear in kappa: increment / beta
        if self.beta_step_cap:  # step gain beta_eta(beta) instead of beta; the score keeps the full beta
            kap = kappa * (self.beta_eta() / self.beta)
        e = core.eligibility_step(e, self.W, kap, pi, self.m, self.roles, self.A[a], self.tau_e, self.beta)
        if self.lam > 0:
            g = core.modulator(p, q, self.roles, routing=self.routing, mu=self.mu)
            if self.la_beta > 0:
                g = g - core.latent_output_barrier(p, JK, self.roles, self.la_beta)
            self.W = core.structural_update(self.W, self.mask, g, kappa, self.m, self.lam, self.eps, self.wmin)
        if self.eps_h > 0:
            self.W = core.homeostasis_update(self.W, self.mask, kappa, self.eps_h, self.wmin)
        if self.routing == "dual":
            self.mu = max(0.0, self.mu + ETA_MU * (self.q_star - core.j_route(JK, self.roles)))
        if reg is not None:
            self.W = core.clip_weights(self.W + reg, self.mask, self.wmin)
        return e

    def task_rate(self):
        return self.eta / (1 + self.ep / self.eta_tau) if self.eta_tau else self.eta

    def regulariser(self, pi, kappa):
        """Iteration 012 C1 + C2 (module doc), on the pre-update W (as Eq. 4b): the ungated per-frame increment
        eta_e kappa (c_ent (g^H - gbar^H) - c_use (g^U - gbar^U)); W is clipped by the caller. Updates ubar and,
        with ent_target, c_ent. Accumulates the squared norms for the RMS ratio against the task term."""
        eta, W, m, inc = self.task_rate(), self.W, self.m, 0.0
        if self.c_ent or self.ent_target:
            if self.ent_target is not None:
                H = core.action_entropy(W, pi, m, self.roles)
                H0 = self.ent_target * np.log(len(self.A))
                self.c_ent = float(np.clip(self.c_ent * np.exp(self.ent_k * (H0 - H)), 1e-6, 10.0))
            dH = eta * self.c_ent * core.centered(core.entropy_score(W, pi, m, self.roles), W, m) * kappa
            self.reg_sq[1] += (dH[self.mask] ** 2).sum()
            inc = inc + dH
        if self.c_use:
            Jin = core.action_inflow(W, pi, m, self.K)[self.A]
            self.ubar += (Jin / Jin.sum() - self.ubar) / self.use_tau
            dU = -eta * self.c_use * core.centered(core.usage_score(W, pi, m, self.roles, self.ubar), W, m) * kappa
            self.reg_sq[2] += (dU[self.mask] ** 2).sum()
            inc = inc + dU
        return inc

    def beta_eta(self):
        """Plasticity gain of the Eq. 4b increment under beta_step_cap c (module doc): min, sat, invu or hyb of beta."""
        b, c = self.beta, self.beta_step_cap
        if self.beta_step_shape == "sat":
            return c * b / (c + b)
        if self.beta_step_shape == "hyb":
            return b if b <= c else c * (1 + np.log(b / c))
        return b * np.exp(1 - b / c) if self.beta_step_shape == "invu" else min(b, c)

    def critic_step(self, x, x2, term, r=0.0):
        """TD error delta_f (Eq. 4c: reward r_f = task.reward(r, term), for CartPole -1 on termination, 0 otherwise;
        bootstraps on truncation; V = u . x; discount self.gamma, GAMMA unless --gamma), computed before the NLMS
        critic update of u. Returns delta; the caller applies the task update eta_e delta e.
        critic_lambda (Proposal, iteration 009): TD(lambda) critic, Eq. 4c with an accumulating trace,
        z <- gamma lambda z + x, u += eta_V delta z / (x . x): a raw presynaptic trace on the critic's inputs, the
        normalisation a global scalar applied at update time. z is periphery state (the critic's feature synapses),
        not crossbar state; it is reset at each episode start (after a terminal frame, and after a truncation, which
        bootstraps first). lambda = 0 gives z = x exactly, so the update is bit for bit the NLMS step of Eq. 4c."""
        delta = self.task.reward(r, term) + (0.0 if term else self.gamma * self.u @ x2) - self.u @ x
        self.z = self.gamma * self.critic_lambda * self.z + x
        self.u += ETA_V * delta * self.z / (x @ x)  # NLMS (lambda = 0) / normalised TD(lambda)
        return delta

    def episode(self, env):
        """One episode, frame by frame (derivation §8): walk and act, pre-TD updates, env step, next frame's walk,
        critic, then frame f's task term (so the task update lands one walk late; spike_frame reads W at the start of
        each walk). With beta_end set, beta is first moved along its ramp (no rng draw)."""
        if self.beta_end is not None:
            self.beta = self.beta0 + (self.beta_end - self.beta0) * min(1.0, self.ep / self.beta_ramp)
        o, _ = env.reset(seed=int(self.rng.integers(1 << 30)))
        self.z = 0.0  # critic trace reset at episode boundaries (the task trace e restarts at 0 below, as before)
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
            if self.beta_signal == "trace":
                self.beta_frame(self.task.reward(r, term), term)
            d_task = self.td_scale(delta)
            eta = self.eta / (1 + self.ep / self.eta_tau) if self.eta_tau else self.eta
            if self.c_ent or self.c_use or self.ent_target:  # RMS of the task update (C1/C2 calibration)
                self.reg_sq[0] += ((eta * d_task * e)[self.mask] ** 2).sum() if np.ndim(e) else 0.0
            if self.rec is not None:
                self.rec.append(o)
            self.W = core.clip_weights(self.W + eta * d_task * e, self.mask, self.wmin)
            x = x2
            self.frames += 1
            st = self.stats
            st["n"], st["td_abs"], st["td_sq"] = st["n"] + 1, st["td_abs"] + abs(delta), st["td_sq"] + delta ** 2
            st["td_task_abs"] += abs(d_task)
            st["V_min"], st["V_max"] = min(st["V_min"], V), max(st["V_max"], V)
        if self.beta_kappa and self.beta_signal == "episode":
            self.beta_step(R)
        self.ep += 1
        return R

    def beta_step(self, R):
        """Causal adaptive gain on beta (Proposal): log beta integrates the normalised return trend (module doc)."""
        if self.r_fast is None:
            self.r_fast, self.r_slow, self.r_var = R, R, 0.0
            return
        self.r_fast += (R - self.r_fast) / BETA_TAU_F
        self.r_var += ((R - self.r_slow) ** 2 - self.r_var) / BETA_TAU_S
        self.r_slow += (R - self.r_slow) / BETA_TAU_S
        if self.ep >= BETA_TAU_F:
            z = np.clip((self.r_fast - self.r_slow) / np.sqrt(self.r_var + 1e-12), -2, 2)
            if self.beta_eps:
                return self.beta_move(z, 1 / BETA_TAU_S)
            self.beta = float(np.clip(self.beta * np.exp(self.beta_kappa * z), self.beta0, self.beta_max))

    def beta_move(self, z, rate):
        """Eq. 1g': log beta += kappa z + eps (log s - log beta), s fixed (beta_set) or experience-dependent:
        the leaky (beta_decay) or strict running maximum of the slow average of log beta (rate per update), toward
        which the pull is upward only. The leak scales with rate: beta_decay is in episodes (trace: BETA_TS / 100)."""
        lb = np.log(self.beta)
        if self.beta_set:
            pull = self.log_set - lb
        else:
            self.log_avg += (lb - self.log_avg) * rate
            leak = rate * BETA_TAU_S / self.beta_decay if self.beta_decay else 0.0  # 1 / D per episode
            self.log_set = max(self.log_avg, self.log_set - (self.log_set - self.log_avg) * leak)
            pull = max(self.log_set - lb, 0.0)
        self.beta = float(np.clip(self.beta * np.exp(self.beta_kappa * z + self.beta_eps * pull), self.beta0,
                                  self.beta_max))

    def beta_frame(self, r, term):
        """beta_signal='trace': per-frame update from the reset-at-termination discounted reward trace (module doc)."""
        self.trace = r + GAMMA * self.trace
        g, e = self.trace, self.tr_ema
        if term:
            self.trace = 0.0
        for i, (x, tau) in enumerate(((g, BETA_TF), (g, BETA_TS), (g, BETA_TH))):
            e[i] += (np.array([x, 1.0]) - e[i]) / tau  # bias-corrected running mean: e[i, 0] / e[i, 1]
        gf, gs, h = e[:3, 0] / e[:3, 1]
        e[3] += (np.array([(h - gs) ** 2, 1.0]) - e[3]) / BETA_TS
        if self.frames >= BETA_TF:  # self.frames counts the frames before this one
            z = float(np.clip((gf - gs) / (np.sqrt(e[3, 0] / e[3, 1]) + 1e-12), -2, 2))
            if self.beta_eps:
                return self.beta_move(z, 1 / BETA_TS)
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
        P = np.array([core.policy(np.bincount(self.m, core.flow(self.W, v, ALPHA, N_HOPS), self.K), self.roles,
                                  self.beta) for v in self.held])
        w = self.W[self.mask]
        out = dict(td_abs=st["td_abs"] / n, td_rms=float(np.sqrt(st["td_sq"] / n)), td_task_abs=st["td_task_abs"] / n,
                   V_min=float(st["V_min"]), V_max=float(st["V_max"]), frac_wmin=float((w <= 1e-3 + 1e-12).mean()),
                   frac_wmax=float((w >= 1 - 1e-12).mean()), P_mean=P.mean(0).tolist(),
                   P_entropy=float(-(P * np.log2(P)).sum(1).mean()), P_max=float(P.max(1).mean()))
        if self.c_ent or self.c_use or self.ent_target:  # iteration 012: RMS ratios of C1 / C2 to the task update
            t = max(self.reg_sq[0], 1e-300)
            out.update(ent_rms_ratio=float(np.sqrt(self.reg_sq[1] / t)),
                       use_rms_ratio=float(np.sqrt(self.reg_sq[2] / t)),
                       c_ent=self.c_ent, ubar=self.ubar.tolist())
            self.reg_sq = np.zeros(3)
        self.stats = dict(n=0, td_abs=0.0, td_sq=0.0, td_task_abs=0.0, V_min=np.inf, V_max=-np.inf)
        return out

    def evaluate(self, kl_obs=20, kl_reps=50):
        """Flow-level metrics on held-out stimuli; spiking adds KL(Eq. 1d || race) on kl_obs held-out stimuli.
        Spiking: consumes self.rng, so LOG_EVERY, kl_obs and kl_reps change the training trajectory."""
        rows, W, m = [], self.W, self.m
        same = m[:, None] == m[None, :]
        persist = np.bincount(m, (W * same).sum(0) / W.sum(0), self.K) / np.bincount(m, minlength=self.K)
        P1d = []
        for v in self.held:
            pi = core.flow(W, v, ALPHA, N_HOPS)
            p, q, JK, _ = core.module_stats(W, pi, m, self.K)
            rows.append([core.map_equation(p, q, pi), q[self.roles == "C"][0], p[self.A].sum(),
                         core.j_route(JK, self.roles)])
            P1d.append(core.policy(p, self.roles, self.beta))
        out: dict = dict(zip(("D", "q_C", "p_A", "J_route"), np.mean(rows, 0).tolist()))
        out.update(persist_C=float(persist[self.roles == "C"].mean()),
                   persist_LA=float(persist[self.roles != "C"].mean()), persist=persist.tolist(),
                   d_max=float(W.sum(0).max()), P1d_mean_right=float(np.mean(P1d, 0)[1]))
        if self.live:
            kl, l1, floor = [], [], []
            smooth = lambda n: (n + 0.5) / (kl_reps + 1)  # noqa: E731 (add-1/2 estimate of the race policy)
            for v, P in zip(self.held[:kl_obs], P1d):
                races = [sp.race(self.spike_frame(v), self.race_groups, self.rng, self.burn) for _ in range(kl_reps)]
                Q = smooth(np.bincount([r for r in races if r is not None], minlength=len(self.A)))
                kl.append(float((P * np.log(P / Q)).sum()))
                l1.append(float(np.abs(P - Q).sum()))
                Qs = smooth(self.rng.multinomial(kl_reps, P, 200))  # KL if the race were exactly Eq. 1d
                floor.append(float((P * np.log(P / Qs)).sum(1).mean()))
            out.update(KL_1d_race=float(np.mean(kl)), KL_floor=float(np.mean(floor)), L1_1d_race=float(np.mean(l1)))
        return out


def run(backend, routing, lam, seed, episodes, model="v3", beta=1.0, eta_tau=None, eta=None, la_beta=0.0,
        task="CartPole", heads=1, name=None, engine="snm", td_norm=0.0, ablate=None, p_cc=0.1, beta_end=None,
        beta_ramp=None, beta_kappa=0.0, beta_max=4.0, score_unit=False, beta_eps=0.0, beta_set=0.0,
        beta_signal="episode", beta_decay=0.0, beta_step_cap=0.0, beta_step_shape="min", critic_lambda=0.0,
        gamma=GAMMA, readout="race", encoder=None, critic_layer=None, c_ent=0.0, c_use=0.0, use_tau=USE_TAU,
        ent_target=None):
    """Train for `episodes`, evaluating every LOG_EVERY (evaluate() plus diagnostics()), and write the result file
    (default name m2_*.json). Iteration-012 runs add end-of-run diagnostics (diag012) over the last REC_EPIS
    episodes."""
    agent = Agent(backend, routing, lam, seed, model, beta, eta_tau, la_beta, task, heads, engine, td_norm, ablate,
                  p_cc, beta_end, beta_ramp, beta_kappa, beta_max, score_unit, beta_eps, beta_set, beta_signal,
                  beta_decay, beta_step_cap, beta_step_shape, critic_lambda, gamma, readout, encoder, critic_layer,
                  c_ent, c_use, use_tau, ent_target)
    new012 = bool(encoder or critic_layer or c_ent or c_use or ent_target)
    env = gym.make(agent.task.env_id)
    agent.eta = eta or agent.eta
    returns, evals, t0 = [], [{"episode": 0, **agent.evaluate(), **agent.diagnostics()}], time.perf_counter()
    for ep in range(1, episodes + 1):
        if new012 and ep == episodes - REC_EPIS + 1:
            agent.rec = []
        returns.append(agent.episode(env))
        if ep % LOG_EVERY == 0:
            evals.append({"episode": ep, **agent.evaluate(), **agent.diagnostics()})
            if agent.beta_kappa or agent.beta_end is not None:
                evals[-1]["beta"] = agent.beta
            ev = evals[-1]
            kl = f"KL {ev['KL_1d_race']:.3f} " if backend == "spiking" else ""
            print(f"ep {ep} last50 {np.mean(returns[-50:]):.1f} D {ev['D']:.3f} p_A {ev['p_A']:.3f} "
                  f"persist_LA {ev['persist_LA']:.3f} H(P) {ev['P_entropy']:.2f} |td| {ev['td_abs']:.3f} "
                  f"d_max {ev['d_max']:.2f} " + kl + f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    frames = agent.frames
    res = {"backend": backend, "routing": routing, "lam": lam, "seed": seed, "episodes": episodes, "returns": returns,
           "model": model, "beta": beta, "eta_tau": agent.eta_tau, "eta_task": agent.eta, "la_beta": la_beta,
           "td_norm": td_norm, "ablate": ablate or {}, "p_cc": p_cc, "beta_end": beta_end,
           "beta_ramp": agent.beta_ramp, "beta_kappa": beta_kappa, "beta_max": beta_max, "score_unit": score_unit,
           **({"beta_eps": beta_eps, "beta_set": beta_set} if beta_eps else {}),
           **({"beta_decay": beta_decay} if beta_decay else {}),
           **({"beta_step_cap": beta_step_cap} if beta_step_cap else {}),
           **({"beta_step_shape": beta_step_shape} if beta_step_shape != "min" else {}),
           **({"beta_signal": beta_signal} if beta_signal != "episode" else {}),
           **({"critic_lambda": critic_lambda} if critic_lambda else {}),
           **({"gamma": gamma} if gamma != GAMMA else {}),
           **({"readout": readout, "bound_deadline_frac": agent.bound_dl / max(agent.bound_n, 1),
               "bound_theta_mean": agent.bound_th / max(agent.bound_n, 1)} if readout != "race" else {}),
           "evals": evals, "frames": frames, "ms_per_frame": (time.perf_counter() - t0) / frames * 1e3,
           "silent_race_frames": agent.silent, "mu": agent.mu, "q_star": agent.q_star,
           "params": dict(alpha=ALPHA, n=N_HOPS, k=K_SPK, rate=RATE, theta=THETA, eta=ETA, tau_e=TAU_E, gamma=gamma,
                          eta_v=ETA_V, eps=agent.eps, eps_h=agent.eps_h, burn=agent.burn, eta_task=agent.eta,
                          beta=beta, eta_tau=agent.eta_tau, eta_mu=ETA_MU, sizes=agent.task.sizes)}
    if (task, heads) != ("CartPole", 1):
        res.update(task=task, heads=heads, engine=engine, encoder=agent.task.encoder, n_ctrl=agent.task.n_ctrl)
    if new012:
        res.update({k: v for k, v in (("encoder_spec", encoder), ("critic_layer", critic_layer), ("c_ent", c_ent),
                                      ("c_use", c_use), ("ent_target", ent_target)) if v},
                   **({"use_tau": use_tau} if c_use else {}), n_ctrl=agent.task.n_ctrl, diag012=diag012(agent))
    OUT.mkdir(exist_ok=True)
    if name is None:
        b = (f"_b{beta:g}" if beta != 1 else "") + (f"_eta{eta:g}" if eta else "")
        b += f"_at{eta_tau:g}" if eta_tau is not None else ""
        b += f"_lb{la_beta:g}" if la_beta else ""
        name = f"m2_{backend}_{routing}_lam{lam:g}{b}_{model}_s{seed}"
    (OUT / f"{name}.json").write_text(json.dumps(res))
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
    """Results table over seeds of every m2_*.json in OUT, grouped by model, backend, routing, lam and variant."""
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
    ap.add_argument("--routing", choices=("exempt", "none", "dual"), default="exempt")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--model", choices=("v1", "v2", "v3"), default="v3")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=None, help="override the task rate")
    ap.add_argument("--eta-tau", type=float, default=None, help="override the annealing time constant (0 = constant)")
    ap.add_argument("--la-beta", type=float, default=0.0, help="latent-output barrier (Eq. 4e, Proposal)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    r = run(a.backend, a.routing, a.lam, a.seed, a.episodes, a.model, a.beta, a.eta_tau, a.eta, a.la_beta)
    print(f"done: last100 {np.mean(r['returns'][-100:]):.1f}, {r['frames']} frames, {r['ms_per_frame']:.2f} ms/frame")


if __name__ == "__main__":
    main()
