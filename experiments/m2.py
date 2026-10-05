"""M2 (docs/SPEC.md §11): CartPole in closed loop with Map-STDP, model v1 (docs/model.md).
Run from the repo root:
  python -m experiments.m2 --backend {flow,spiking} --routing {exempt,none,dual} --lam 0.05 --seed 0 --episodes 1500
    -> experiments/results/m2_<backend>_<routing>_lam<lam>_s<seed>.json
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
    def __init__(self, backend, routing, lam, seed):
        self.W, self.m, self.roles, self.mask = core.build_network(SIZES, seed=seed)
        self.K, self.backend, self.routing, self.lam = len(self.roles), backend, routing, lam
        self.A = np.flatnonzero(self.roles == "A")
        self.C = self.roles[self.m] == "C"
        self.groups = [np.flatnonzero(self.m == a) for a in self.A]
        self.rng = np.random.default_rng(seed)
        self.u, self.mu, self.silent = np.zeros(self.C.sum()), 0.0, 0
        self.held = [self.stim(o) for o in held_out_obs()]
        self.snn = None
        self.q_star = 0.75 * self.evaluate()["J_route"]  # Eq. 4a target; ponytail: fixed, learn it if dual matters
        self.snn = sp.build(self.W, 1 - ALPHA, theta=THETA) if backend == "spiking" else None

    def stim(self, o):
        return core.cartpole_stimulus(o, self.m, self.roles)

    def frame(self, v):
        """pi, kappa and action (index into A) for one frame."""
        W = self.W
        if self.backend == "flow":
            pi = core.flow(W, v, ALPHA, N_HOPS)
            p = np.bincount(self.m, pi, self.K)
            return pi, W / W.sum(0) * pi, int(self.rng.choice(len(self.A), p=core.policy(p, self.roles)))
        S = self.spike_frame(v)
        a = sp.race(S, self.groups, self.rng)
        if a is None:
            self.silent, a = self.silent + 1, int(self.rng.integers(len(self.A)))
        return sp.pi_hat(S), sp.pairings(S)[2] / max(S.sum(), 1), a

    def spike_frame(self, v):
        sp.set_weights(self.snn, self.W, 1 - ALPHA)  # output gain (1 - alpha) / d_j
        return sp.run_frame(self.snn, v, K_SPK, RATE, self.rng, THETA)[0]

    def feats(self, pi):
        return pi[self.C] / max(pi[self.C].sum(), 1e-12)  # critic V = u . pi_C / p_C

    def episode(self, env):
        o, _ = env.reset(seed=int(self.rng.integers(1 << 30)))
        pi, kappa, a = self.frame(self.stim(o))
        x, e, R, done = self.feats(pi), 0.0, 0, False
        while not done:
            p, q, JK, _ = core.module_stats(self.W, pi, self.m, self.K)
            e = core.eligibility_step(e, self.W, kappa, pi, self.m, self.roles, self.A[a], TAU_E)
            if self.lam > 0:
                g = core.modulator(p, q, self.roles, routing=self.routing, mu=self.mu)
                self.W = core.structural_update(self.W, self.mask, g, kappa, self.m, self.lam, EPS)
            if self.routing == "dual":
                self.mu = max(0.0, self.mu + ETA_MU * (self.q_star - core.j_route(JK, self.roles)))
            o, r, term, trunc, _ = env.step(a)
            R, done = R + r, term or trunc
            pi, kappa, a = self.frame(self.stim(o))
            x2 = self.feats(pi)
            delta = (-1.0 if term else 0.0) + (0.0 if term else GAMMA * self.u @ x2) - self.u @ x
            self.u += ETA_V * delta * x / (x @ x)  # NLMS
            self.W = core.clip_weights(self.W + ETA * delta * e, self.mask)
            x = x2
        return R

    def evaluate(self, kl_obs=20, kl_reps=50):
        """Flow-level metrics on held-out stimuli; spiking adds KL(Eq. 1d || race) on kl_obs held-out stimuli."""
        rows, W, m = [], self.W, self.m
        same = m[:, None] == m[None, :]
        persist = np.bincount(m, (W * same).sum(0) / W.sum(0), self.K) / np.bincount(m, minlength=self.K)
        P1d = []
        for v in self.held:
            pi = core.flow(W, v, ALPHA, N_HOPS)
            p, q, JK, _ = core.module_stats(W, pi, m, self.K)
            rows.append([core.map_equation(p, q, pi), q[self.roles == "C"][0], p[self.A].sum(), core.j_route(JK, self.roles)])
            P1d.append(core.policy(p, self.roles))
        out = dict(zip(("D", "q_C", "p_A", "J_route"), np.mean(rows, 0).tolist()))
        out.update(persist_C=float(persist[self.roles == "C"].mean()), persist_LA=float(persist[self.roles != "C"].mean()),
                   persist=persist.tolist(), d_max=float(W.sum(0).max()), P1d_mean_right=float(np.mean(P1d, 0)[1]))
        if self.snn is not None:
            kl, l1, floor = [], [], []
            smooth = lambda n: (n + 0.5) / (kl_reps + 1)  # noqa: E731 (add-1/2 estimate of the race policy)
            for v, P in zip(self.held[:kl_obs], P1d):
                races = [sp.race(self.spike_frame(v), self.groups, self.rng) for _ in range(kl_reps)]
                Q = smooth(np.bincount([r for r in races if r is not None], minlength=len(self.A)))
                kl.append(float((P * np.log(P / Q)).sum()))
                l1.append(float(np.abs(P - Q).sum()))
                Qs = smooth(self.rng.multinomial(kl_reps, P, 200))  # KL if the race were exactly Eq. 1d
                floor.append(float((P * np.log(P / Qs)).sum(1).mean()))
            out.update(KL_1d_race=float(np.mean(kl)), KL_floor=float(np.mean(floor)), L1_1d_race=float(np.mean(l1)))
        return out


def run(backend, routing, lam, seed, episodes):
    agent, env = Agent(backend, routing, lam, seed), gym.make("CartPole-v1")
    returns, evals, t0 = [], [{"episode": 0, **agent.evaluate()}], time.perf_counter()
    for ep in range(1, episodes + 1):
        returns.append(agent.episode(env))
        if ep % LOG_EVERY == 0:
            evals.append({"episode": ep, **agent.evaluate()})
            print(f"ep {ep} last50 {np.mean(returns[-50:]):.1f} D {evals[-1]['D']:.3f} p_A {evals[-1]['p_A']:.3f} "
                  f"persist_LA {evals[-1]['persist_LA']:.3f} " + (f"KL {evals[-1]['KL_1d_race']:.3f} " if backend == "spiking" else "")
                  + f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    frames = sum(returns)
    res = {"backend": backend, "routing": routing, "lam": lam, "seed": seed, "episodes": episodes, "returns": returns,
           "evals": evals, "frames": frames, "ms_per_frame": (time.perf_counter() - t0) / frames * 1e3,
           "silent_race_frames": agent.silent, "mu": agent.mu, "q_star": agent.q_star,
           "params": dict(alpha=ALPHA, n=N_HOPS, k=K_SPK, rate=RATE, theta=THETA, eta=ETA, tau_e=TAU_E, gamma=GAMMA,
                          eta_v=ETA_V, eps=EPS, eta_mu=ETA_MU, sizes=SIZES)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"m2_{backend}_{routing}_lam{lam:g}_s{seed}.json").write_text(json.dumps(res))
    return res


def summary():
    rand = random_return()
    print(f"random policy: {rand[0]:.1f} +- {rand[1]:.1f} (1000 episodes)")
    groups = {}
    for f in sorted(glob.glob(str(OUT / "m2_*.json"))):
        r = json.loads(pathlib.Path(f).read_text())
        groups.setdefault((r["backend"], r["routing"], r["lam"]), []).append(r)
    cols = ("D", "J_route", "p_A", "persist_LA", "persist_C", "d_max", "KL_1d_race", "KL_floor")
    print("return: mean +- sd over seeds of per-seed mean return; metrics: seed-mean at episode 0 > final eval (sd)")
    print(f"{'backend':8} {'routing':7} {'lam':>5} {'n':>2} {'epis':>5} {'ret first100':>13} {'ret last100':>13} "
          + " ".join(f"{c:>18}" for c in cols))
    ms = lambda x: f"{np.mean(x):6.1f}+-{np.std(x):5.1f}"  # noqa: E731
    for (b, ro, lam), rs in sorted(groups.items()):
        line = f"{b:8} {ro:7} {lam:5g} {len(rs):2} {min(r['episodes'] for r in rs):5} " \
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
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    r = run(a.backend, a.routing, a.lam, a.seed, a.episodes)
    print(f"done: last100 {np.mean(r['returns'][-100:]):.1f}, {r['frames']} frames, {r['ms_per_frame']:.2f} ms/frame")


if __name__ == "__main__":
    main()
