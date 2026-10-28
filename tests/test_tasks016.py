"""Iteration 016: the TD-STDP backbone on Acrobot and Taxi (experiments.tdstdp PRESETS by pre-set rules): the one-hot
encoder, the presets' sizes, a short closed loop with recurrent columns, k from CartPole's BindsNET rate, the r_scale
rule and the hybrid-columns-2 study declaration."""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest

from experiments import report
from experiments import tdstdp as ex


def test_onehot_encoder():
    cfg = ex.preset("Taxi", seed=0, in_rep=2)
    phi, n = ex.ck_encoder(cfg)
    f = phi(np.array([3.0, 499.0]))
    assert n == 1000 and f.shape == (2, 1000) and f.sum() == 4
    assert f[0, 3] == f[0, 503] == f[1, 499] == f[1, 999] == 1


@pytest.mark.parametrize("task, n_in, K", [("Acrobot", 208, 3), ("Taxi", 500, 6)])
def test_preset_closed_loop(task, n_in, K):
    """rec 'cols' on the new tasks: sizes, a short run, the module table of R."""
    cfg = ex.preset(task, seed=0, engine="nlif", s_in=1.0, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols", eta_rec=1.0, lam_rec=1.0)
    r = ex.CKRunner(cfg)
    assert r.lay.n_in == n_in and r.K == K and r.hr.Kr == K and r.lay.n_post == cfg.n_ce + K * cfg.n_ae
    for _ in range(300):
        r.tick()
    assert r.frames > 0 and np.isfinite(r.W).all() and np.isfinite(r.hr.R).all()
    RK = ex.result_ck(cfg, r, 0.0, None, 0)["rec"]["RK"]["0"]
    assert np.shape(RK) == (K, K) and np.isclose(np.sum(RK), 1)


def test_k_from_cartpole_rate():
    """r_ck_hz: k = CartPole's BindsNET rate / rate_target, not this task's."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.18, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False)
    low, rec = ex.calibrate_low(cfg)
    assert rec["k"] == pytest.approx(ex.R_CK_CARTPOLE / 40) and low.alpha_a == pytest.approx(rec["k"] * 50.0)


@pytest.mark.parametrize("task", ["CartPole", "Acrobot", "Taxi"])
def test_reward_rules(task):
    """r_scale = 0.02 / RMS of the per-step reward under a random policy (CartPole: RMS 1, the preset 0.02); beta_c =
    min(-0.2, 1.2 V_rand), V_rand = r_scale E[r] tau_d / MS (CartPole: -0.2)."""
    env, rng, r = gym.make(ex.TD_TASKS[task].env_id), np.random.default_rng(0), []
    env.reset(seed=0)
    while len(r) < 20000:
        _, x, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        r.append(x)
        if term or trunc:
            env.reset(seed=int(rng.integers(1 << 31)))
    p = ex.preset(task)
    assert p.r_scale == pytest.approx(0.02 / np.sqrt(np.mean(np.square(r))), rel=0.03)
    assert p.beta_c == pytest.approx(min(-0.2, 1.2 * p.r_scale * np.mean(r) * p.tau_d / ex.MS), rel=0.05)


def test_hybrid_columns_2_declared():
    s = report.STUDIES["hybrid-columns-2"]
    assert len(s["groups"]) == 4 and all(list(g["seeds"]) == list(range(4000, 4020)) for g in s["groups"].values())
    assert all({"--rate-match-rec", "--ent-match-rec"} <= set(g["flags"]) for g in s["groups"].values())
    heads = {p[0] for p in s["pairs"] if len(p) == 2}
    assert heads == {f"Acrobot_{a}" for a in ("cols_lam0", "cols_on", "cols_sealed")}
    r = dict(config=dataclasses.asdict(ex.preset("Taxi")), returns=list(range(2000)))
    assert report.TD_PAIR_METRICS["primary"](r) == report._auc_min(r)


def test_rate_match_rec():
    """--rate-match-rec: probe_rate runs with the arm's recurrence (the kicks raise the rate at a shared s_in), and only
    on engine nlif (BindsNET's probe stays without recurrence); default off, token 'rm'."""
    cfg = ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols")
    on = dataclasses.replace(cfg, rate_match_rec=True)
    assert ex.probe_rate(on) > ex.probe_rate(cfg) and not ex.Config().rate_match_rec
    assert ex.probe_rate(dataclasses.replace(on, engine="lif", s_in=0.0, rate_target=0.0)) > 0
    assert ex.tokens(on)[-1] != "rm" and "rm" in ex.tokens(on) and "rm" not in ex.tokens(cfg)


def test_ent_match_rec():
    """--ent-match-rec: at a shared s_in the columns' loop sharpens the initial policy (probe_entropy lower than the
    base run's), which match_entropy undoes with f < 1 on alpha_a; default off, token 'em'."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols", alpha_a=175.0)
    assert ex.probe_entropy(cfg) < ex.probe_entropy(dataclasses.replace(cfg, rec="none"))
    on = dataclasses.replace(cfg, ent_match_rec=True)
    assert not ex.Config().ent_match_rec and "em" in ex.tokens(on) and "em" not in ex.tokens(cfg)
