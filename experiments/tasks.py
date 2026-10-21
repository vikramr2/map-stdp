"""Task registry for M2 and M3 (docs/SPEC.md §4, docs/model.md §1): per Gymnasium task, the module sizes, the
stimulus encoder v(o) over the controller and the reward r_f fed to the TD error (Eq. 4c).
CartPole keeps its M2 encoder (6 x 6 grid over theta, theta_dot) and reward (-1 on termination), bit for bit.
The M3 encoders (Proposal) are conjunctive Gaussian RFs over the whole normalised observation:
  grid:  a 6 x 6 grid over a 2-D observation normalised by its bounds (MountainCar);
  pool:  RF centers drawn from states a random policy visits, observation z-scored by that pool, sigma the median
         nearest-center distance (Acrobot, LunarLander: the 'random sparse conjunctive fields' of model.md §1).
Rewards are the environment's, scaled so that a failing episode's TD errors are O(1) like CartPole's."""
import dataclasses
from collections.abc import Callable

import gymnasium as gym
import numpy as np

from mapstdp import core

LATENTS, ACTION_SIZE, ENC_SEED, POOL = (16, 16), 16, 777, 20000


@dataclasses.dataclass
class Task:
    env_id: str
    n_actions: int
    n_ctrl: int
    episodes: int  # default training length (M3 reporting unit: 5 seeds x episodes)
    reward: Callable  # (r, term) -> r_f of Eq. 4c
    encoder: str  # 'cartpole' | 'grid' | 'pool'
    enc: dict = dataclasses.field(default_factory=dict)  # fitted encoder state (centers, sigma, normalisation)

    @property
    def sizes(self):
        return (self.n_ctrl, *LATENTS, *(ACTION_SIZE,) * self.n_actions)

    @property
    def roles(self):
        return "C" + "L" * len(LATENTS) + "A" * self.n_actions

    def stim(self, o, m, roles):
        """v(o) over the controller neurons."""
        if self.encoder == "cartpole":
            return core.cartpole_stimulus(o, m, roles)
        if not self.enc:
            self.enc = fit_encoder(self)
        e = self.enc
        return core.rbf_stimulus(np.clip((np.asarray(o, float) - e["shift"]) / e["scale"], *e["clip"]),
                                 e["centers"], e["sigma"], m, roles)


def random_states(env_id, n_actions, n, seed):
    """n states visited by a uniform-random policy, episodes from consecutive resets."""
    env, rng, obs = gym.make(env_id), np.random.default_rng(seed), []
    while len(obs) < n:
        o, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            obs.append(o)
            o, _, term, trunc, _ = env.step(int(rng.integers(n_actions)))
            done = term or trunc
    return np.array(obs[:n])


def fit_encoder(task):
    """Encoder state for the 'grid' and 'pool' encoders (fixed seed: a sensory front-end shared by every agent seed)."""
    if task.encoder == "grid":
        sp = gym.make(task.env_id).observation_space
        g = np.linspace(-1, 1, 6)
        return dict(shift=(sp.high + sp.low) / 2, scale=(sp.high - sp.low) / 2, clip=(-1, 1), sigma=0.4,
                    centers=np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2))
    X = random_states(task.env_id, task.n_actions, POOL, ENC_SEED)
    shift, scale = X.mean(0), np.maximum(X.std(0), 1e-6)
    Z = (X - shift) / scale
    C = Z[np.random.default_rng(ENC_SEED).choice(len(Z), task.n_ctrl, replace=False)]
    D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1)) + np.diag(np.full(len(C), np.inf))
    return dict(shift=shift, scale=scale, clip=(-4, 4), centers=C, sigma=float(np.median(D.min(1))))


TASKS = {t.env_id.split("-")[0]: t for t in (
    Task("CartPole-v1", 2, 36, 1500, lambda r, term: -1.0 if term else 0.0, "cartpole"),
    Task("MountainCar-v0", 3, 36, 1500, lambda r, term: 0.1 * r, "grid"),
    Task("Acrobot-v1", 3, 64, 1000, lambda r, term: 0.1 * r, "pool"),
    Task("LunarLander-v3", 4, 64, 1000, lambda r, term: 0.01 * r, "pool"),
)}
# CartPole with the 'pool' encoder over all four observations (iteration 006): the M2 encoder sees only theta and
# theta_dot, so neither the policy nor a network-activity critic can see the cart position that ends sharp-policy runs.
TASKS["CartPoleFull"] = Task("CartPole-v1", 2, 64, 1500, TASKS["CartPole"].reward, "pool")
