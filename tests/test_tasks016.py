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
    assert r.lay.n_in == n_in and r.K == K and r.hr.Kr == K
    assert r.lay.n_post == cfg.n_ce + K * cfg.n_ae and cfg.n_ai == 0
    for _ in range(300):
        r.tick()
    assert r.frames > 0 and np.isfinite(r.W).all() and np.isfinite(r.hr.R).all()
    RK = ex.result_ck(cfg, r, 0.0, None, 0)["rec"]["RK"]["0"]
    assert np.shape(RK) == (K, K) and np.isclose(np.sum(RK), 1)


def test_k_from_cartpole_rate():
    """r_ck_hz: k = CartPole's BindsNET rate / rate_target, not this task's."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.18, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, norm_logits=True)  # norm_logits: the BindsNET reference probes run without it
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
    cp = ex.preset("CartPole", seed=0, engine="nlif", s_in=0.28, rate_target=40.0, block=20, fast=True, adam=False,
                   alpha_a=175.0)  # one copy, short episodes: ticks with no live copy must not make the probe NaN
    assert 0 < ex.probe_entropy(cp) <= np.log(2) + 1e-9


def test_norm_logits():
    """--norm-logits (016 §13): the policy is invariant to a common scaling of the rates above the floor, the scores
    are CK's at rbar = r0, the eligibility factors are the exact gradient of log p and H through the normalisation
    (finite differences), the floor rule, default off, token 'nl'."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, norm_logits=True)
    r = ex.CKRunner(cfg)
    K, H, rng = r.K, r.H, np.random.default_rng(0)
    assert r.eps_n == pytest.approx(cfg.alpha_a * 0.04 / (cfg.tau_n * cfg.n_ae))
    R = rng.uniform(0.02, 0.08, (H, r.lay.n_post))
    sc, rk, rbar = r.scores(R)
    assert np.allclose(ex.td.softmax(sc), ex.td.softmax(r.scores(0.5 * R)[0]))  # scale-free above the floor
    R0 = R * 0.04 / rbar[:, None]  # pooled rate = r0: CK's scores
    assert np.allclose(r.scores(R0)[0], cfg.alpha_a * R0 @ r.lay.Ma.T)
    assert np.allclose(r.scores(R * 1e-3)[0], cfg.alpha_a * (0.04 / r.eps_n) * (R * 1e-3) @ r.lay.Ma.T)  # floored
    # the exact gradient: d log p_a / d r_j and dH / d r_j vs finite differences on the column rates
    r.prk, r.prbar, r.p, r.action = rk, rbar, ex.td.softmax(sc), np.zeros(H, int)
    ge = ex.td.entropy_coef(r.p)
    fac, u, ug = r.norm_factors(ge)
    claim = fac[:, None] * (np.eye(K)[r.action] - r.p - u[:, None]) * cfg.alpha_a
    claimH = fac[:, None] * (ge - ug[:, None]) * cfg.alpha_a

    def f(rk_, h):
        s = cfg.alpha_a * 0.04 / max(rk_.mean(), r.eps_n) * rk_
        p = ex.td.softmax(s)
        return np.log(p[0]), -(p * np.log(p)).sum()
    for h in range(H):
        for j in range(K):
            d = 1e-6 * np.eye(K)[j]
            lp1, H1 = f(rk[h] + d, h)
            lp0, H0 = f(rk[h] - d, h)
            assert claim[h, j] == pytest.approx((lp1 - lp0) / 2e-6, rel=1e-4, abs=1e-6)
            assert claimH[h, j] == pytest.approx((H1 - H0) / 2e-6, rel=1e-4, abs=1e-6)
    for _ in range(200):
        r.tick()
    assert np.isfinite(r.W).all() and np.isfinite(r.p).all()
    assert not ex.Config().norm_logits and "nl" in ex.tokens(cfg) and "nl" not in ex.tokens(ex.preset("Acrobot"))


def test_hybrid_columns_3_declared():
    s = report.STUDIES["hybrid-columns-3"]
    assert len(s["groups"]) == 12 and all(list(g["seeds"]) == list(range(5000, 5020)) for g in s["groups"].values())
    flags = {"--rate-match-rec", "--ent-match-rec", "--norm-logits"}
    assert all(flags <= set(g["flags"]) for g in s["groups"].values())
    assert all(("--eta-cal" in g["flags"]) == (g["task"] != "CartPole") for g in s["groups"].values())
    assert {g["task"] for g in s["groups"].values()} == {"CartPole", "LunarLander", "Acrobot"}
    assert len([p for p in s["pairs"] if len(p) == 2]) == 9 and len(s["pairs"]) == 18


def test_class_pair_counts():
    """The stimulus-conditioned walk (016 §15): the per-class split of the pair counts sums to the one call exactly,
    class_table gives per-class shares that sum to 1 (0 for an empty class) and resets, and the result and checkpoint
    carry it (rec.RKc)."""
    cfg = ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols", eta_rec=1.0, lam_rec=1.0, heads=4)
    r = ex.CKRunner(cfg)
    hr, rng = r.hr, np.random.default_rng(1)
    XR = rng.random((20, 4, hr.nR)) < 0.05
    xs, xq = rng.random((4, hr.nR)), rng.random((4, hr.nR)) + 1
    cls = np.array([0, 2, 2, 3])
    hr.window(XR, xs, xq, cls)
    split = hr.acc["kappa"].copy()
    hr.acc, hr.kc[:] = hr._fresh(), 0
    hr.tail[:] = False
    hr.window(XR, xs, xq)
    assert np.array_equal(split, hr.acc["kappa"])
    hr.tail[:] = False
    hr.window(XR, xs, xq, cls)
    t = hr.class_table()
    assert t["share"].shape == (r.K, hr.Kr, hr.Kr) and t["n"][1] == 0 and (t["n"][[0, 2, 3]] > 0).all()
    assert np.allclose(t["share"].sum((1, 2)), [1, 0, 1, 1]) and not hr.kc.any()
    for _ in range(60):
        r.tick()
    out = ex.result_ck(cfg, r, 0.0, None, 0)["rec"]
    k = max(int(x) for x in out["RK"])
    assert str(k) in out["RK"] and set(out["RKc"]) <= set(out["RK"]) or out["RKc"] == {}
    assert "rec_kc" in hr.state()
