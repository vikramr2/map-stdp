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
  python -m experiments.m2 --summary   -> results table over seeds (plus the random-policy return)
Both backends share one frame update; they differ only in where pi, kappa and the action come from:
  flow:    pi = n-hop flow, kappa = J (expected pairings), action ~ Eq. 1d;
  spiking: pi = spike shares, kappa = covariance count / spikes in the frame, action = race readout.
Dividing kappa by the frame's spike count is the 1/(spikes per frame) scaling of lambda and eta (model.md §4)."""
import argparse
import glob
import json
import pathlib
import time

import gymnasium as gym
import numpy as np

from mapstdp import core
from mapstdp import spiking as sp

ALPHA, N_HOPS, K_SPK, RATE, THETA = 0.2, 20, 100, 1.0, 0.5
ETA, TAU_E, GAMMA, ETA_V, EPS, ETA_MU = 10.0, 3.0, 0.99, 0.1, 1.0, 2.0
ETA_SPK, EPS_H, BURN = 5.0, 5.0, 10  # v2: spiking task rate, d_j homeostasis rate (own term), race burn-in (steps)
ETA_TAU = 300.0  # v3: task-rate annealing time constant (episodes)
Q_STAR_FRAC = 0.75  # dual routing target q* as a fraction of the held-out mean J_route at W_0 (Eq. 4a)
SIZES, LOG_EVERY = (36, 16, 16, 16, 16), 50
OUT = pathlib.Path(__file__).parent / "results"


def held_out_obs(n=100, seed=12345):
    """Fixed evaluation observations: states visited by a random policy."""
    env, rng, obs = gym.make("CartPole-v1"), np.random.default_rng(seed), []
    while len(obs) < n:
        o, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            obs.append(o)
            o, _, term, trunc, _ = env.step(int(rng.integers(2)))
            done = term or trunc
    return np.array(obs[:n])


def random_return(episodes=1000, seed=0):
    """Mean and sd of the uniform-random-policy return (the --summary baseline)."""
    env, rng, R = gym.make("CartPole-v1"), np.random.default_rng(seed), []
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(1 << 30)))
        n, done = 0, False
        while not done:
            _, r, term, trunc, _ = env.step(int(rng.integers(2)))
            n, done = n + r, term or trunc
        R.append(n)
    return float(np.mean(R)), float(np.std(R))


class Agent:
    """Closed-loop Map-STDP agent: structural (Eq. 4) with optional routing dual (Eq. 4a), d_j homeostasis (Eq. 4d)
    and TD task terms (Eqs. 4b, 4c; task rate annealed in v3)."""

    def __init__(self, backend, routing, lam, seed, model="v3", beta=1.0, eta_tau=None, la_beta=0.0):
        self.W, self.m, self.roles, self.mask = core.build_network(SIZES, seed=seed)
        self.K, self.backend, self.routing, self.lam = len(self.roles), backend, routing, lam
        v2 = model in ("v2", "v3")
        if eta_tau is None:
            eta_tau = ETA_TAU if model == "v3" else 0.0
        self.eps, self.eps_h, self.burn = (0.0, EPS_H, BURN) if v2 else (EPS, 0.0, 0)
        self.eta = ETA_SPK if model == "v2" and backend == "spiking" else ETA
        self.beta, self.eta_tau, self.ep, self.la_beta = beta, eta_tau, 0, la_beta
        assert beta == 1.0 or backend == "flow", "the race readout has no matched sharpened policy"
        self.A = np.flatnonzero(self.roles == "A")
        self.ctrl = self.roles[self.m] == "C"
        self.groups = [np.flatnonzero(self.m == a) for a in self.A]
        self.rng = np.random.default_rng(seed)
        self.u, self.mu, self.silent = np.zeros(self.ctrl.sum()), 0.0, 0
        self.held = [self.stim(o) for o in held_out_obs()]
        self.snn = None  # built after q_star, so this evaluate() runs no races and draws nothing from rng
        self.q_star = Q_STAR_FRAC * self.evaluate()["J_route"]  # ponytail: fixed, learn it if dual matters
        self.snn = sp.build(self.W, 1 - ALPHA, theta=THETA) if backend == "spiking" else None

    def stim(self, o):
        """v(o): CartPole encoder over the controller (core.cartpole_stimulus)."""
        return core.cartpole_stimulus(o, self.m, self.roles)

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
        a = sp.race(S, self.groups, self.rng, self.burn)
        if a is None:
            self.silent, a = self.silent + 1, int(self.rng.integers(len(self.A)))
        return sp.pi_hat(S), sp.pairings(S)[2] / max(S.sum(), 1), a

    def spike_frame(self, v):
        """Spike raster of one frame on the current weights."""
        sp.set_weights(self.snn, self.W, 1 - ALPHA)  # output gain (1 - alpha) / d_j
        return sp.run_frame(self.snn, v, K_SPK, RATE, self.rng, THETA)[0]

    def feats(self, pi):
        """Critic features x = pi_C / p_C (controller flow shares), so V = u . x."""
        return pi[self.ctrl] / max(pi[self.ctrl].sum(), 1e-12)

    def pre_td_updates(self, pi, kappa, a, e):
        """Per-frame updates that need no TD error (derivation §8 steps 4-5), in this order, which tests/golden.json
        locks: module statistics on the current W; eligibility increment (Eq. 4b, its h-bar uses the pre-update W);
        structural term (Eq. 4); d_j homeostasis (Eq. 4d); routing dual mu (Eq. 4a, from the pre-update JK).
        Mutates self.W and self.mu; returns the updated trace e."""
        p, q, JK, _ = core.module_stats(self.W, pi, self.m, self.K)
        e = core.eligibility_step(e, self.W, kappa, pi, self.m, self.roles, self.A[a], TAU_E, self.beta)
        if self.lam > 0:
            g = core.modulator(p, q, self.roles, routing=self.routing, mu=self.mu)
            if self.la_beta > 0:
                g = g - core.latent_output_barrier(p, JK, self.roles, self.la_beta)
            self.W = core.structural_update(self.W, self.mask, g, kappa, self.m, self.lam, self.eps)
        if self.eps_h > 0:
            self.W = core.homeostasis_update(self.W, self.mask, kappa, self.eps_h)
        if self.routing == "dual":
            self.mu = max(0.0, self.mu + ETA_MU * (self.q_star - core.j_route(JK, self.roles)))
        return e

    def critic_step(self, x, x2, term):
        """TD error delta_f (Eq. 4c: reward -1 on termination, 0 otherwise; bootstraps on truncation; V = u . x),
        computed before the NLMS critic update of u. Returns delta; the caller applies the task update eta_e delta e."""
        delta = (-1.0 if term else 0.0) + (0.0 if term else GAMMA * self.u @ x2) - self.u @ x
        self.u += ETA_V * delta * x / (x @ x)  # NLMS
        return delta

    def episode(self, env):
        """One episode, frame by frame (derivation §8): walk and act, pre-TD updates, env step, next frame's walk,
        critic, then frame f's task term (so the task update lands one walk late; spike_frame reads W at the start of
        each walk)."""
        o, _ = env.reset(seed=int(self.rng.integers(1 << 30)))
        pi, kappa, a = self.frame(self.stim(o))
        x, e, R, done = self.feats(pi), 0.0, 0, False
        while not done:
            e = self.pre_td_updates(pi, kappa, a, e)
            o, r, term, trunc, _ = env.step(a)
            R, done = R + r, term or trunc
            pi, kappa, a = self.frame(self.stim(o))
            x2 = self.feats(pi)
            delta = self.critic_step(x, x2, term)
            eta = self.eta / (1 + self.ep / self.eta_tau) if self.eta_tau else self.eta
            self.W = core.clip_weights(self.W + eta * delta * e, self.mask)
            x = x2
        self.ep += 1
        return R

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
        if self.snn is not None:
            kl, l1, floor = [], [], []
            smooth = lambda n: (n + 0.5) / (kl_reps + 1)  # noqa: E731 (add-1/2 estimate of the race policy)
            for v, P in zip(self.held[:kl_obs], P1d):
                races = [sp.race(self.spike_frame(v), self.groups, self.rng, self.burn) for _ in range(kl_reps)]
                Q = smooth(np.bincount([r for r in races if r is not None], minlength=len(self.A)))
                kl.append(float((P * np.log(P / Q)).sum()))
                l1.append(float(np.abs(P - Q).sum()))
                Qs = smooth(self.rng.multinomial(kl_reps, P, 200))  # KL if the race were exactly Eq. 1d
                floor.append(float((P * np.log(P / Qs)).sum(1).mean()))
            out.update(KL_1d_race=float(np.mean(kl)), KL_floor=float(np.mean(floor)), L1_1d_race=float(np.mean(l1)))
        return out


def run(backend, routing, lam, seed, episodes, model="v3", beta=1.0, eta_tau=None, eta=None, la_beta=0.0):
    """Train for `episodes`, evaluating every LOG_EVERY, and write the m2_*.json result file."""
    agent, env = Agent(backend, routing, lam, seed, model, beta, eta_tau, la_beta), gym.make("CartPole-v1")
    agent.eta = eta or agent.eta
    returns, evals, t0 = [], [{"episode": 0, **agent.evaluate()}], time.perf_counter()
    for ep in range(1, episodes + 1):
        returns.append(agent.episode(env))
        if ep % LOG_EVERY == 0:
            evals.append({"episode": ep, **agent.evaluate()})
            ev = evals[-1]
            kl = f"KL {ev['KL_1d_race']:.3f} " if backend == "spiking" else ""
            print(f"ep {ep} last50 {np.mean(returns[-50:]):.1f} D {ev['D']:.3f} p_A {ev['p_A']:.3f} "
                  f"persist_LA {ev['persist_LA']:.3f} " + kl + f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    frames = sum(returns)
    res = {"backend": backend, "routing": routing, "lam": lam, "seed": seed, "episodes": episodes, "returns": returns,
           "model": model, "beta": beta, "eta_tau": agent.eta_tau, "eta_task": agent.eta, "la_beta": la_beta,
           "evals": evals, "frames": frames, "ms_per_frame": (time.perf_counter() - t0) / frames * 1e3,
           "silent_race_frames": agent.silent, "mu": agent.mu, "q_star": agent.q_star,
           "params": dict(alpha=ALPHA, n=N_HOPS, k=K_SPK, rate=RATE, theta=THETA, eta=ETA, tau_e=TAU_E, gamma=GAMMA,
                          eta_v=ETA_V, eps=agent.eps, eps_h=agent.eps_h, burn=agent.burn, eta_task=agent.eta,
                          beta=beta, eta_tau=agent.eta_tau, eta_mu=ETA_MU, sizes=SIZES)}
    OUT.mkdir(exist_ok=True)
    b = (f"_b{beta:g}" if beta != 1 else "") + (f"_eta{eta:g}" if eta else "")
    b += f"_at{eta_tau:g}" if eta_tau is not None else ""
    b += f"_lb{la_beta:g}" if la_beta else ""
    (OUT / f"m2_{backend}_{routing}_lam{lam:g}{b}_{model}_s{seed}.json").write_text(json.dumps(res))
    return res


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
