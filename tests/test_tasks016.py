"""Iteration 016: the TD-STDP backbone on Acrobot and Taxi (experiments.tdstdp PRESETS by pre-set rules): the one-hot
encoder, the presets' sizes, a short closed loop with recurrent columns, k from CartPole's BindsNET rate, the r_scale
rule and the hybrid-columns-2 study declaration; the race readout of §14.1 Stage B (the decision, the theta rule, the
surrogate's gradient, closed loops, calibrate_race, tokens)."""
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
    csplit = hr.cc.sum(0)
    t = hr.class_table()
    assert t["share"].shape == t["cshare"].shape == (r.K, hr.Kr, hr.Kr)
    assert t["n"][1] == 0 and (t["n"][[0, 2, 3]] > 0).all()
    assert np.allclose(t["share"].sum((1, 2)), [1, 0, 1, 1]) and np.allclose(t["cshare"].sum((1, 2)), [1, 0, 1, 1])
    assert not hr.kc.any() and not hr.cc.any() and np.allclose(csplit, hr.acc["c"])  # the counter's split sums too
    for _ in range(60):
        r.tick()
    out = ex.result_ck(cfg, r, 0.0, None, 0)["rec"]
    k = max(int(x) for x in out["RK"])
    assert str(k) in out["RK"] and set(out["RKc"]) <= set(out["RK"]) or out["RKc"] == {}
    assert "rec_kc" in hr.state()


def _race_runner(**kw):
    return ex.CKRunner(ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True,
                                 sgd=True, adam=False, readout="race", race_beta=6.0, **kw))


def test_race_decision():
    """The race (Eq. 1h, CKRunner.race_block) on a synthetic raster: the first column whose E count reaches theta
    wins at that ms (an earlier lead does not), a copy with no column at theta takes the leader at the deadline, an
    exact tie costs exactly one uniform draw (none otherwise), the actor spikes up to the bound ms (or the window at
    the deadline) are tallied as the energy to decide, and I posts are not counted."""
    r = _race_runner()
    L, n, nc, n_ae = 20, r.lay.n_post, r.lay.n_crit, r.cfg.n_ae
    assert r.theta == 16 and r.pend.tolist() == [-1] and not r.cnt.any()
    X = np.zeros((L, 1, n), bool)
    X[:, 0, nc + n_ae:nc + 2 * n_ae] = True  # column 1: 20 spikes per ms, at theta in the first ms
    X[:, 0, nc] = True  # column 0: 1 spike per ms
    r.race_block(X)
    assert r.pend[0] == 1 and r.cnt[0].tolist() == [20, 400, 0] and r.rdec[0] == 1
    assert r.ebnd.tolist() == [21, 1] and r.wspk[0] == 420  # 21 actor spikes in the bound ms; 420 in the block
    r.cnt[:], r.pend[:], r.wspk[:] = 0, -1, 0
    X = np.zeros((L, 1, n), bool)
    X[:5, 0, nc] = True  # column 0 leads early (5 spikes), column 2 reaches theta at ms 7 (2 spikes per ms)
    X[:, 0, nc + 2 * n_ae] = True
    X[:16, 0, nc + 2 * n_ae + 1] = True
    r.race_block(X)
    assert r.pend[0] == 2 and r.cnt[0].tolist() == [5, 0, 36] and r.ebnd.tolist() == [42, 2]  # 5 + 16 by ms 7
    # the deadline: no column at theta by the end of the decision window (theta out of reach), the leader wins
    r = _race_runner(race_theta=1e9)
    r.cnt[0], r.pend[0], r.timer[:] = [0.0, 1000.0, 0.0], -1, r.cfg.freeze - ex.MS
    r.wspk[0] = 7.0
    st = r.rng.bit_generator.state
    r.tick()
    assert r.action[0] == 1 and r.pend[0] == -1 and not r.cnt.any() and r.rdec[1] == 1 and r.rdec[2] == 0
    assert r.ebnd[1] == 1 and r.ebnd[0] >= 7 and r.wspk[0] == 0  # the window's spikes (7 carried in plus the tick's)
    # I posts (LunarLander: n_ai = n_ae) are not tallied: a raster of I spikes only moves no count
    q = ex.CKRunner(ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True,
                              sgd=True, adam=False, heads=1, readout="race", race_beta=4.0))
    ipost = np.flatnonzero((q.lay.Ma < 0).any(0))  # the columns' I posts
    assert len(ipost) == q.K * q.cfg.n_ae and np.isin(ipost, q.apost).all()
    Xi = np.zeros((L, 1, q.lay.n_post), bool)
    Xi[:, 0, ipost] = True  # every I post spikes every ms
    q.race_block(Xi)
    assert not q.cnt.any() and q.pend[0] == -1 and q.wspk[0] == L * len(ipost)
    # ties: one draw among the tied columns; no draw without a tie
    r = _race_runner()
    st = r.rng.bit_generator.state
    assert r._leader(np.array([3.0, 7.0, 7.0])) in (1, 2) and r.rdec[2] == 1
    ref = np.random.default_rng(0)
    ref.bit_generator.state = st
    ref.choice(np.array([1, 2]))
    assert r.rng.bit_generator.state == ref.bit_generator.state
    st = r.rng.bit_generator.state
    assert r._leader(np.array([3.0, 7.0, 2.0])) == 1 and r.rng.bit_generator.state == st


def test_race_theta_rule():
    """theta = round(n_ae r0 freeze): Acrobot 20 x 0.04 x 20 = 16, LunarLander 32 x 0.04 x 40 = 51; --race-theta
    overrides."""
    assert ex.race_theta(ex.preset("Acrobot", rate_target=40.0)) == 16
    assert ex.race_theta(ex.preset("LunarLander", rate_target=40.0)) == 51
    assert ex.race_theta(ex.preset("CartPole", rate_target=40.0, race_theta=9)) == 9


def test_race_gradient():
    """The per-column factor (race_factor) times (A - p~) and times g(p~) is the exact gradient of log p~_a and of H
    with respect to the column rates (central differences; alpha_a / n_ae absorbed into eta as CK's rule), and the
    common mode sum_j r_j dH / dr_j is zero (H is invariant to a common scaling of the rates above the floor)."""
    r = _race_runner()
    K, cfg, rng = r.K, r.cfg, np.random.default_rng(3)
    assert r.eps_r == pytest.approx(1 / (cfg.tau_n * cfg.n_ae))
    for _ in range(50):
        rk = rng.uniform(0.005, 0.1, K)
        p, a = r.surrogate(rk), int(rng.integers(K))
        fr = r.race_factor(rk[None])[0]
        claim = fr * (np.eye(K)[a] - p) * cfg.alpha_a
        claimH = fr * ex.td.entropy_coef(p) * cfg.alpha_a

        def f(x, a=a):
            q = r.surrogate(x)
            return np.log(q[a]), -(q * np.log(q)).sum()
        for j in range(K):
            d = 1e-6 * np.eye(K)[j]
            (lp1, H1), (lp0, H0) = f(rk + d), f(rk - d)
            assert claim[j] == pytest.approx((lp1 - lp0) / 2e-6, rel=1e-4, abs=1e-6)
            assert claimH[j] == pytest.approx((H1 - H0) / 2e-6, rel=1e-4, abs=1e-6)
        assert abs((rk * claimH).sum()) < 1e-9
    assert np.allclose(r.surrogate(np.full(K, 1e-9)), 1 / K) and not r.race_factor(np.full((1, K), 1e-9)).any()


def test_race_closed_loop():
    """readout race in closed loop: Acrobot (one copy) and LunarLander (4 copies, rec cols) give finite W, R and p; the
    result carries the energy monitor and the decision counts; the softmax path still reports the energy."""
    r = _race_runner()
    for _ in range(100):
        r.tick()
    assert r.frames > 0 and np.isfinite(r.W).all() and np.isfinite(r.p).all() and r.rdec[:2].sum() > 0
    out = ex.result_ck(r.cfg, r, 0.0, None, 0)
    assert out["actor_spikes_per_decision"] > 0 and out["race_decisions"]["theta"] == 16
    assert out["race_decisions"]["at_bound"] + out["race_decisions"]["at_deadline"] == r.rdec[:2].sum()
    cfg = ex.preset("LunarLander", seed=0, engine="nlif", s_in=0.1, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, rec="cols", eta_rec=1.0, lam_rec=1.0, heads=4, readout="race", race_beta=4.0)
    r = ex.CKRunner(cfg)
    assert r.theta == 51
    for _ in range(60):
        r.tick()
    assert np.isfinite(r.W).all() and np.isfinite(r.hr.R).all() and np.isfinite(r.p).all()
    s = ex.CKRunner(dataclasses.replace(cfg, readout="softmax", race_beta=0.0, rec="none"))
    for _ in range(10):
        s.tick()
    assert "race_decisions" not in ex.result_ck(s.cfg, s, 0.0, None, 0) and s.espk[0] > 0 and s.stats[6] > 0


def test_calibrate_race():
    """calibrate_race on Acrobot (the contrast probe): a finite beta strictly inside the bracket, the surrogate at beta
    reproducing column 0's realised win rate within 0.1 at every contrast, the record's diagnostics."""
    cfg = ex.preset("Acrobot", seed=0, engine="nlif", s_in=0.16, rate_target=40.0, block=20, fast=True, sgd=True,
                    adam=False, readout="race")
    beta, rec = ex.calibrate_race(cfg)
    assert np.isfinite(beta) and 2 ** -3 * 1.05 < beta < 2 ** 6 / 1.05 and rec["beta"] == beta
    assert rec["deltas"] == list(ex.RACE_DELTAS) and rec["n_decisions"] == 8 * len(ex.RACE_DELTAS) * 25
    w, ws = np.array(rec["win0"]), np.array(rec["win0_surrogate"])
    assert np.abs(w - ws).max() < 0.1 and w[-1] > w[0]  # the contrast moves column 0's win rate
    assert rec["loglik"] < 0 and 0 < rec["H_race"] < np.log(3) + 0.2 and 0 < rec["H_surrogate"] <= np.log(3) + 1e-9
    assert rec["beta_poisson"] == pytest.approx(1.6 * np.sqrt(8))


def test_race_tokens_and_defaults():
    d = ex.Config()
    assert d.readout == "softmax" and d.race_theta == 0.0 and d.race_beta == 0.0
    cfg = ex.preset("Acrobot", engine="nlif", rate_target=40.0, readout="race")
    assert ex.tokens(cfg)[-1] == "race" and "race" not in ex.tokens(ex.preset("Acrobot", engine="nlif"))
    assert ex.tokens(dataclasses.replace(cfg, race_theta=12.0, race_beta=2.5))[-3:] == ["race", "th12", "rb2.5"]
    with pytest.raises(AssertionError):
        ex.CKRunner(dataclasses.replace(cfg, s_in=0.16, norm_logits=True))
    with pytest.raises(AssertionError):
        ex.CKRunner(dataclasses.replace(cfg, s_in=0.16, rate_target=0.0))
