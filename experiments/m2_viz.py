"""Record one CartPole episode of a trained spiking Map-STDP agent (model v3) for the replay page.
Run from the repo root:
  python -m experiments.m2_viz train  --seed 3 --episodes 1500   -> experiments/results/m2_viz_W_s<seed>.npz
  python -m experiments.m2_viz record --seed 3                   -> experiments/results/m2_viz_episode_s<seed>.json
  python -m experiments.m2_viz page --seed 3 --out replay.html   -> episode embedded in m2_viz_template.html
Recording freezes the weights and the critic; every frame keeps its k x N spike raster, the race winner and the
cart state, so the page can replay the per-frame walk next to the pole."""
import argparse
import json
import pathlib

import gymnasium as gym
import numpy as np

from mapstdp import core
from mapstdp import spiking as sp
from experiments import m2


def path(kind, seed):
    return m2.OUT / (f"m2_viz_W_s{seed}.npz" if kind == "W" else f"m2_viz_episode_s{seed}.json")


def train(seed, episodes):
    agent, env, R = m2.Agent("spiking", "exempt", 0.05, seed, "v3"), gym.make("CartPole-v1"), []
    for ep in range(1, episodes + 1):
        R.append(agent.episode(env))
        if ep % 50 == 0:
            print(f"ep {ep} last50 {np.mean(R[-50:]):.1f}", flush=True)
    np.savez(path("W", seed), W=agent.W, mask=agent.mask, m=agent.m, roles=agent.roles, returns=R)


def first_race_step(S, groups, burn):
    cnt = np.stack([S[burn:, g].sum(1) for g in groups], 1).sum(1)
    t = np.flatnonzero(cnt)
    return int(t[0]) + burn if len(t) else -1


def record(seed, tries=20):
    """Best of `tries` frozen-weight episodes (the policy is stochastic), so the page shows a long balance."""
    z = np.load(path("W", seed))
    agent = m2.Agent("spiking", "exempt", 0.05, seed, "v3")
    agent.W = z["W"]
    env, best = gym.make("CartPole-v1"), []
    for k in range(tries):
        o, _ = env.reset(seed=1000 + k)
        frames, done = [], False
        while not done:
            v = agent.stim(o)
            S = agent.spike_frame(v)
            a = sp.race(S, agent.groups, agent.rng, agent.burn)
            a = int(agent.rng.integers(2)) if a is None else a
            p = core.module_stats(agent.W, core.flow(agent.W, v, m2.ALPHA, m2.N_HOPS), agent.m, agent.K)[0]
            frames.append({"obs": [round(float(x), 4) for x in o], "a": a, "race_t": first_race_step(S, agent.groups, agent.burn),
                           "P": [round(float(x), 3) for x in core.policy(p, agent.roles)],
                           "S": [np.flatnonzero(s).tolist() for s in S]})
            o, _, term, trunc, _ = env.step(a)
            done = term or trunc
        print(f"try {k}: {len(frames)} frames", flush=True)
        if len(frames) > len(best):
            best = frames
        if len(best) >= 500:
            break
    W = agent.W / agent.W.sum(0)
    i, j = np.nonzero(z["mask"])
    out = {"seed": seed, "k": m2.K_SPK, "burn": agent.burn, "alpha": m2.ALPHA, "m": agent.m.tolist(),
           "roles": agent.roles.tolist(), "train_returns": z["returns"].tolist(),
           "edges": [[int(a), int(b), round(float(W[a, b]), 4)] for a, b in zip(i, j)],  # [post, pre, T]
           "frames": best}
    path("E", seed).write_text(json.dumps(out, separators=(",", ":")))
    print(f"recorded {len(best)} frames -> {path('E', seed)}")


def page(seed, out):
    tpl = (pathlib.Path(__file__).parent / "m2_viz_template.html").read_text()
    pathlib.Path(out).write_text(tpl.replace("/*DATA*/", path("E", seed).read_text()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "record", "page"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--out", default="replay.html")
    a = ap.parse_args()
    {"train": lambda: train(a.seed, a.episodes), "record": lambda: record(a.seed), "page": lambda: page(a.seed, a.out)}[a.cmd]()
