"""Policy-class ceiling (M3 diagnostic, iteration 005): the best return the flow policy family reaches when its
parameters are searched directly by the cross-entropy method (CEM) instead of learned by Map-STDP.
At the flow level pi(o) = M v(o) with M the n-hop operator of Eq. 1a, so the action shares are p = R v_C(o) with
R = H_A^T M[:, C] (actions x controller, nonnegative) and Eq. 1d is P(a | o) = (R v_C)_a / sum (R v_C).
  learned:  R of the W Map-STDP learned (flow, v3, beta_LA 0.3, --td-norm 0.04), frozen;
  level 2:  CEM over the controller -> action synapses of that W (bounds [w_min, w_max], mask, alpha, the latents and
            everything else as learned), Eq. 1d readout; the realistic ceiling;
  level 1:  CEM over any R = exp(theta), readouts Eq. 1d, Eq. 1e (beta = 4) and argmax; the generous upper bound.
Run from the repo root:
  python -m experiments.ceiling --task Acrobot --level {learned,2,1} [--readout {1d,beta4,argmax}] [--seed 0]
    -> experiments/results/ceiling_<task>_<level>[_<readout>]_s<seed>.json
  python -m experiments.ceiling --summary"""
import argparse
import glob
import json
import pathlib
import time

import gymnasium as gym
import numpy as np

from experiments import m2
from experiments.tasks import TASKS
from mapstdp import core

POP, ELITE, ITERS, EPIS, FINAL = 64, 8, 80, 4, 100  # CEM population, elites, iterations, episodes/candidate, final
WMIN, WMAX, TD_NORM, LA_BETA = 1e-3, 1.0, 0.04, 0.3


def hop_operator(W, alpha=m2.ALPHA, n=m2.N_HOPS):
    """M with pi_n = M v for core.flow(W, v, alpha, n): M_0 = I, M_{k+1} = (1 - alpha) T M_k + alpha I (Eq. 1a)."""
    B, M = (1 - alpha) * W / W.sum(0), np.eye(len(W))
    for _ in range(n):
        M = B @ M + alpha * np.eye(len(W))
    return M


def action_matrix(W, agent):
    """R = H_A^T M[:, C]: action-module shares p_A = R v_C for any stimulus."""
    H = (agent.m[:, None] == agent.A[None, :]).astype(float)
    return H.T @ hop_operator(W)[:, agent.ctrl]


def act(p, readout, rng):
    if readout == "argmax":
        return int(np.argmax(p))
    q = p ** 4 if readout == "beta4" else p
    return int(rng.choice(len(q), p=q / q.sum()))


def rollouts(R, agent, env, readout, seeds):
    """Mean return of the frozen policy over episodes with the given env seeds (policy rng seeded likewise)."""
    out = []
    for s in seeds:
        o, _ = env.reset(seed=int(s))
        rng, total, done = np.random.default_rng(s), 0.0, False
        while not done:
            o, r, term, trunc, _ = env.step(act(R @ agent.stim(o)[agent.ctrl], readout, rng))
            total, done = total + r, term or trunc
        out.append(total)
    return float(np.mean(out))


def trained_agent(task, seed):
    """The Map-STDP flow agent after the task's training episodes (the M3 flow reference with --td-norm)."""
    agent = m2.Agent("flow", "exempt", 0.05, seed, la_beta=LA_BETA, task=task, td_norm=TD_NORM)
    env, R = gym.make(agent.task.env_id), []
    for _ in range(agent.task.episodes):
        R.append(agent.episode(env))
    return agent, float(np.mean(R[-100:]))


def cem(score, mu, sd, rng, log):
    """Diagonal-Gaussian CEM maximising score(theta, seeds); common random numbers (one seed set) per iteration."""
    best = (-np.inf, mu)
    for it in range(ITERS):
        seeds = rng.integers(1 << 30, size=EPIS)
        th = mu + sd * rng.standard_normal((POP, len(mu)))
        f = np.array([score(t, seeds) for t in th])
        el = th[np.argsort(f)[-ELITE:]]
        mu, sd = el.mean(0), el.std(0) + 0.02  # noise floor keeps the search from freezing
        if f.max() > best[0]:
            best = (float(f.max()), th[np.argmax(f)])
        log(f"it {it} best {f.max():.1f} mean {f.mean():.1f} elite {np.sort(f)[-ELITE:].mean():.1f} sd {sd.mean():.3f}")
    return mu


def run(task, level, readout, seed):
    t0 = time.perf_counter()
    agent, train_ret = trained_agent(task, seed)
    env, rng, W0 = gym.make(agent.task.env_id), np.random.default_rng(seed + 1000), agent.W.copy()
    R0 = action_matrix(W0, agent)
    v = agent.held[0]  # R reproduces the agent's own Eq. 1d policy
    assert np.allclose(R0 @ v[agent.ctrl] / (R0 @ v[agent.ctrl]).sum(),
                       core.policy(np.bincount(agent.m, core.flow(W0, v, m2.ALPHA, m2.N_HOPS), agent.K), agent.roles))
    log = lambda s: print(f"{task} L{level} {readout}: {s} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)  # noqa: E731
    log(f"trained, last100 training return {train_ret:.1f}")
    if level == "learned":
        R = R0
    elif level == "2":
        sel = agent.mask & (agent.roles[agent.m] == "A")[:, None] & agent.ctrl[None, :]  # controller -> action
        def to_R(th):
            W = W0.copy()
            W[sel] = WMIN + (WMAX - WMIN) / (1 + np.exp(-th))
            return action_matrix(W, agent)
        x = np.clip((W0[sel] - WMIN) / (WMAX - WMIN), 1e-4, 1 - 1e-4)
        R = to_R(cem(lambda th, s: rollouts(to_R(th), agent, env, readout, s), np.log(x / (1 - x)),
                     np.ones(sel.sum()), rng, log))
    else:
        R = np.exp(cem(lambda th, s: rollouts(np.exp(th.reshape(R0.shape)), agent, env, readout, s),
                       np.zeros(R0.size), 2 * np.ones(R0.size), rng, log).reshape(R0.shape))
    final = [rollouts(R, agent, env, readout, [s]) for s in np.random.default_rng(99).integers(1 << 30, size=FINAL)]
    res = dict(task=task, level=level, readout=readout, seed=seed, train_last100=train_ret,
               final_mean=float(np.mean(final)), final_sd=float(np.std(final)), final=final,
               minutes=(time.perf_counter() - t0) / 60,
               params=dict(pop=POP, elite=ELITE, iters=ITERS, epis=EPIS, td_norm=TD_NORM, la_beta=LA_BETA))
    suffix = "" if level == "learned" else f"_{readout}"
    (m2.OUT / f"ceiling_{task}_L{level}{suffix}_s{seed}.json").write_text(json.dumps(res))
    log(f"final {res['final_mean']:.1f} +- {res['final_sd']:.1f} over {FINAL} fresh episodes")


def summary():
    rows = {}
    for f in sorted(glob.glob(str(m2.OUT / "ceiling_*.json"))):
        r = json.loads(pathlib.Path(f).read_text())
        rows.setdefault(r["task"], []).append(r)
    for task, rs in rows.items():
        print(f"\n{task}: Map-STDP training last100 {rs[0]['train_last100']:.1f}")
        for r in sorted(rs, key=lambda r: (r["level"] != "learned", r["level"], r["readout"])):
            print(f"  level {r['level']:7} {r['readout']:6}  {r['final_mean']:8.1f} +- {r['final_sd']:6.1f}  "
                  f"({r['minutes']:.0f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS))
    ap.add_argument("--level", choices=("learned", "2", "1"), default="2")
    ap.add_argument("--readout", choices=("1d", "beta4", "argmax"), default="1d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    assert a.level == "1" or a.readout == "1d", "levels learned and 2 use the model's own readout, Eq. 1d"
    run(a.task, a.level, a.readout, a.seed)


if __name__ == "__main__":
    main()
