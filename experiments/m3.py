"""M3 (docs/SPEC.md §11): model v3 with the opened latents (latent-output barrier Eq. 4e, beta_LA = 0.3) on CartPole,
MountainCar, Acrobot and LunarLander, flow against spiking with H parallel heads (Proposal). Run from the repo root:
  python -m experiments.m3 --task {CartPole,MountainCar,Acrobot,LunarLander} --backend {flow,spiking,mpr} [--heads H]
    --seed 0 [--episodes E] [--td-norm s] [variant options]
    -> experiments/results/m3_<task>_<backend>[_h<H>]_lb<beta_LA>[_tn<s>][_<variant>]_s<seed>.json
  Every option maps onto an m2.Config field (m2 module doc). Variant tokens, in file-name order:
    ablations (iteration 005) wmin<w>, tau_e<T>, at<T> (--wmin, --tau-e, --eta-tau);
    lam<l>, pcc<P> (--lam, --p-cc); b<B> (--beta, Eq. 1e); bk<k>m<B> (--beta-kappa, --beta-max: adaptive beta,
    Eq. 1g); g<g> (--gamma, iteration 009); dbc (--readout diffbound, Eq. 1h, spiking, iteration 010);
    enc<SPEC>, crit<SPEC>, ce<c>[H<f>] (--encoder, --critic-layer, --c-ent [--ent-target]: iteration 012, flow);
    eta<E> (--eta); qif[-<field><value>...] (--engine qif and its QIFParams overrides, iteration 013, e.g. qif-tau_s0
    -inhma-warm0-lags3); lif[-<field><value>...] (--engine lif, rule emergent only, and its LIFParams overrides
    --lif-<field>); emergent[-ot<T>][-ebmod][-stk] (--rule emergent, Eqs. 8-12, and its options --obs-tau,
    --ebar module, --step-stat kappa); ls<x>, es<x> (--lam-scale, --eta-scale), ms (--match-scales); ep<E> (--episodes
    off the task's default).
  Run plumbing: --out PATH (result file; its .ckpt.npz checkpoint alongside, resumed if present), --save-model PATH
    (compressed float32 .npz of W at episode 0, every 250 episodes and the end, with mask, m, roles, config, u),
    --match-scales (m2.match_scales: lam and eta scales from the RMS of the structural and task increments at the
    initial W against flow's; printed, applied and recorded in the result).
  Model v4 is the flag set --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5 (docs/model.md).
  The beta ramp, the beta mean reversion / trace signal / step cap, the score-unit ablation, the TD(lambda) critic,
  the rich and net critics and the usage term (former Eq. 4i) were removed in iteration 013 (not adopted); their result
  files still load in --summary and --ablations.
  python -m experiments.m3 --summary    -> per task: random, flow and spiking per H (last-100 return over seeds)
  python -m experiments.m3 --ablations  -> flow learning-rule ablations against the --td-norm 0.04 baseline
  python -m experiments.m3 --fidelity [--engine qif ...]  -> per task and H: error of the pooled pi-hat and kappa
    against the flow (qif: also the per-postsynaptic-row 1 - corr and the bias 1 - corr(E[kappa], J))
Heads: H independent spiking replicas of the network on the same W (tasks share one block-diagonal SuperNeuroMAT
network). Their spikes are pooled: pi-hat from summed counts, kappa the sum of per-head covariance counts divided by
the total spikes, one race over all heads (still Eq. 1d). H = 1 is the M2 spiking agent."""
import argparse
import dataclasses
import glob
import json
import pathlib

import numpy as np

from experiments import m2
from experiments.tasks import TASKS
from mapstdp import core, lif, qif
from mapstdp import spiking as sp

LA_BETA, HEADS = 0.3, (1, 2, 4, 8, 16)


def name(task, backend, heads, la_beta, seed, td_norm=0.0, variant=""):
    return f"m3_{task}_{backend}" + (f"_h{heads}" if backend == "spiking" else "") + f"_lb{la_beta:g}" + \
        (f"_tn{td_norm:g}" if td_norm else "") + (f"_{variant}" if variant else "") + f"_s{seed}"


def row_corr(K, J, mask):
    """Mean over postsynaptic rows (>= 3 synapses, non-constant) of corr(K[i], J[i]) on the mask: insensitive to a
    per-postsynaptic gain."""
    c = [np.corrcoef(K[i, mask[i]], J[i, mask[i]])[0, 1] for i in range(len(K)) if mask[i].sum() >= 3]
    return float(np.nanmean(c))


def fidelity(task, Hs=HEADS, reps=20, n_obs=10, seed=0, engine="numpy", qp=None):
    """Learning-signal error of H pooled heads against the flow on the initial W, over held-out stimuli: L1 of pi-hat
    to the n-hop flow, and 1 - corr of kappa with J (the flow's expected pairings), mean over reps frames.
    engine='qif' (qp: qif.QIFParams): warm frames after one warm-up frame per stimulus (cold if not qp.warm), eta from
    the (seed, m2.QIF_SEED) stream as the agent draws it; adds the mean per-row 1 - corr (1-corr_row), the
    spikes per module per frame (S_mod) and 1 - corr(mean kappa, T * mean pi-hat) (1-corr_Jhat: kappa against the
    pairings the network's own rates predict). All engines: the bias 1 - corr(mean kappa over reps, J) (1-corr_bias)."""
    agent = m2.Agent(m2.Config("flow", seed=seed, la_beta=LA_BETA, task=task))
    rows, N = [], len(agent.W)
    for H in Hs:
        rng, l1, rk, rr, rb, rh, smod = np.random.default_rng(seed), [], [], [], [], [], []
        if engine == "qif":
            qrng = np.random.default_rng((seed, m2.QIF_SEED))
            eta = qif.lorentzian(qrng, (H, qif.n_neurons(N, agent.m, agent.roles, qp)), qp)
        for v in agent.held[:n_obs]:
            pi = core.flow(agent.W, v, m2.ALPHA, m2.N_HOPS)
            J = agent.W / agent.W.sum(0) * pi
            if engine == "qif":
                st = qif.init_state(eta, qrng, qp)
                if qp.warm:
                    qif.run_qif(agent.W, v, H, rng, eta, st, qp, agent.m, agent.roles)
            ks, phs = [], []
            for _ in range(reps):
                if engine == "qif":
                    if not qp.warm:
                        st = qif.init_state(eta, qrng, qp)
                    S = qif.run_qif(agent.W, v, H, rng, eta, st, qp, agent.m, agent.roles)
                    ph, kap, _ = qif.pooled(S, H, qp)
                    smod.append(np.bincount(agent.m, S.reshape(len(S), H, N).sum((0, 1)), agent.K))
                    rr.append(1 - row_corr(kap, J, agent.mask))
                else:
                    S = sp.run_heads(agent.W, v, H, m2.K_SPK, m2.RATE, rng, m2.THETA, 1 - m2.ALPHA)
                    ph, kap, _ = sp.pooled(S, H)
                l1.append(np.abs(ph - pi).sum())
                rk.append(1 - np.corrcoef(kap[agent.mask], J[agent.mask])[0, 1])
                ks.append(kap)
                phs.append(ph)
            rb.append(1 - np.corrcoef(np.mean(ks, 0)[agent.mask], J[agent.mask])[0, 1])
            Jh = agent.W / agent.W.sum(0) * np.mean(phs, 0)
            rh.append(1 - np.corrcoef(np.mean(ks, 0)[agent.mask], Jh[agent.mask])[0, 1])
        rows.append({"H": H, "L1_pi": float(np.mean(l1)), "1-corr_kappa_J": float(np.mean(rk)),
                     "1-corr_bias": float(np.mean(rb))})
        if engine == "qif":
            rows[-1].update({"1-corr_row": float(np.mean(rr)), "1-corr_Jhat": float(np.mean(rh)),
                             "S_mod": np.mean(smod, 0).tolist()})
        print(f"{task:12} H={H:2d}  " + "  ".join(f"{k} {x:.4f}" for k, x in rows[-1].items()
                                                if k not in ("H", "S_mod")) +
              (f"  S_mod {np.round(rows[-1]['S_mod']).astype(int).tolist()}" if engine == "qif" else ""), flush=True)
    return rows


def nondefault(r):
    """True for a variant run (any option off its M3 default). Also reads the keys of the variants removed in
    iteration 013 (beta_end, score_unit, beta_eps, beta_signal, beta_step_cap, critic_lambda, c_use), so their
    committed files stay out of the baseline groups."""
    eta0 = m2.ETA if r["backend"] == "flow" else r["params"]["eta_task"]
    return r.get("lam", 0.05) != 0.05 or r.get("p_cc", 0.1) != 0.1 or r.get("p_cross", 0.05) != 0.05 or \
        r.get("beta", 1.0) != 1.0 or \
        r.get("beta_end") is not None or \
        r.get("eta_task", eta0) != eta0 or bool(r.get("beta_kappa")) or bool(r.get("score_unit")) or \
        bool(r.get("beta_eps")) or r.get("beta_signal", "episode") != "episode" or bool(r.get("beta_step_cap")) or \
        bool(r.get("critic_lambda")) or r.get("gamma", m2.GAMMA) != m2.GAMMA or \
        r.get("ablate", {}).get("tau_e", m2.TAU_E) != m2.TAU_E or r.get("readout", "race") != "race" or \
        bool(r.get("encoder_spec") or r.get("critic_layer") or r.get("c_ent") or r.get("c_use")) or \
        r.get("engine") in ("qif", "lif") or r.get("rule", "v4") != "v4" or r.get("lam_scale", 1.0) != 1.0 or \
        r.get("eta_scale", 1.0) != 1.0 or \
        r["episodes"] != TASKS[r.get("task", "CartPole")].episodes


def variant_tokens(a):
    """File-name tokens of the variants (module doc), in file-name order; empty on the defaults."""
    t = ([f"lam{a.lam:g}"] if a.lam != 0.05 else []) + ([f"pcc{a.p_cc:g}"] if a.p_cc != 0.1 else [])
    t += [f"px{a.p_cross:g}"] if a.p_cross != 0.05 else []
    t += [f"b{a.beta:g}"] if a.beta != 1.0 else []
    t += [f"bk{a.beta_kappa:g}m{a.beta_max:g}"] if a.beta_kappa else []
    t += [f"g{a.gamma:g}"] if a.gamma != m2.GAMMA else []
    t += ["dbc"] if a.readout == "diffbound" else []
    t += ([f"enc{a.encoder}"] if a.encoder else []) + ([f"crit{a.critic_layer}"] if a.critic_layer else [])
    t += [f"ce{a.c_ent:g}" + (f"H{a.ent_target:g}" if a.ent_target is not None else "")] if a.c_ent else []
    t += [f"eta{a.eta:g}"] if a.eta is not None else []
    t += [qif_token(a)] if a.engine == "qif" and a.backend == "spiking" else []
    t += [lif_token(a)] if a.engine == "lif" and a.backend == "spiking" else []
    t += [qif_token(a).replace("qif", "q", 1)] if a.backend == "mpr" and qif_overrides(a) else []
    if a.rule != "v4":
        t += ["-".join([a.rule] + ([f"ot{a.obs_tau:g}"] if a.obs_tau else [])
                       + (["ebmod"] if a.ebar == "module" else []) + (["stk"] if a.step_stat == "kappa" else []))]
    t += ([f"ls{a.lam_scale:g}"] if a.lam_scale != 1.0 else []) + ([f"es{a.eta_scale:g}"] if a.eta_scale != 1.0 else [])
    t += ["ms"] if a.match_scales else []
    return t + ([f"ep{a.episodes}"] if a.episodes and a.episodes != TASKS[a.task].episodes else [])


QIF_FLAGS = {"tau_s": float, "inh": str, "warm": int, "lags": int, "tau_plus": float, "gain": float, "rate": float,
             "a_ext": float, "delta": float, "eta_bar": float, "eta_max": float, "sigma": float, "k": int,
             "burn": int, "w_ei": float, "w_ie": float, "n_inh": int, "tau": float, "lam_scale": float,
             "eta_scale": float, "eps_h_scale": float}  # --engine qif: QIFParams fields (--q-<field>)


def qif_overrides(a):
    """QIFParams overrides given on the command line (--q-<field>), in QIF_FLAGS order."""
    return {k: getattr(a, f"q_{k}") for k in QIF_FLAGS if getattr(a, f"q_{k}") is not None}


def qif_token(a):
    """File-name token of --engine qif: qif, then -<field><value> per override."""
    o = qif_overrides(a)
    return "-".join(["qif", *(f"{k}{v if isinstance(v, str) else format(v, 'g')}" for k, v in o.items())])


LIF_FLAGS = {"tau_m": float, "dt": float, "tau_ref": float, "mu_bg": float, "sigma_V": float, "delta_V": float,
             "rho0": float, "reset": int, "chi": float, "a_ext": float, "ext_rate": float, "k": int, "window": float,
             "counter": str, "omega": float, "gain_tau": float, "reg_tau": float, "reg_min": int, "warm": int,
             "warmup": int, "lam_scale": float, "eta_scale": float,
             "eps_h_scale": float}  # --engine lif: LIFParams fields (--lif-<field>)


def lif_overrides(a):
    """LIFParams overrides given on the command line (--lif-<field>), in LIF_FLAGS order (int flags of bool fields
    become bools)."""
    o = {k: getattr(a, f"lif_{k}") for k in LIF_FLAGS if getattr(a, f"lif_{k}") is not None}
    return {k: bool(v) if isinstance(getattr(lif.LIFParams(), k), bool) else v for k, v in o.items()}


def lif_token(a):
    """File-name token of --engine lif: lif, then -<field><value> per override."""
    o = {k: getattr(a, f"lif_{k}") for k in LIF_FLAGS if getattr(a, f"lif_{k}") is not None}
    return "-".join(["lif", *(f"{k}{v if isinstance(v, str) else format(v, 'g')}" for k, v in o.items())])


def summary():
    """Per task: random-policy return, then flow and spiking per H as mean +- sd over seeds of the last-100 return,
    and the share of the flow-over-spiking(H = 1) gap that H heads close."""
    print("return: mean +- sd over seeds of the per-seed mean of the last 100 episodes; "
          "closed = (spiking_H - spiking_1) / (flow - spiking_1)")
    for task in TASKS:
        runs = {}
        for f in sorted(glob.glob(str(m2.OUT / f"m3_{task}_*.json"))):
            r = json.loads(pathlib.Path(f).read_text())
            if r.get("ablate") or r.get("eta_tau", 300.0) != 300.0 or nondefault(r):
                continue  # learning-rule ablations and variants: see ablation_summary()
            runs.setdefault((r["backend"], r.get("heads", 1), r["la_beta"], r.get("td_norm", 0.0)), []).append(r)
        if not runs:
            continue
        rand = m2.random_return(200, task=TASKS[task])
        print(f"\n{task}: random {rand[0]:.1f} +- {rand[1]:.1f}")
        last = {k: [np.mean(r["returns"][-100:]) for r in rs] for k, rs in runs.items()}
        for (b, H, lb, tn), rs in sorted(runs.items(), key=lambda kv: (kv[0][0] != "flow", kv[0][3], *kv[0][1:3])):
            R = last[(b, H, lb, tn)]
            line = f"  {b:8} H={H:2d} lb={lb:g} tn={tn:g} n={len(rs)} epis={min(r['episodes'] for r in rs):5} " \
                   f"ret {np.mean(R):8.1f} +- {np.std(R):6.1f}  ms/fr {np.mean([r['ms_per_frame'] for r in rs]):5.1f}"
            fl, s1 = last.get(("flow", 1, lb, tn)), last.get(("spiking", 1, lb, tn))
            if b == "spiking" and fl and s1 and abs(np.mean(fl) - np.mean(s1)) > 1e-9:
                line += f"  closed {(np.mean(R) - np.mean(s1)) / (np.mean(fl) - np.mean(s1)):+.2f}"
            print(line)


def ablation_summary():
    """Flow learning-rule ablations (iteration 005) against the --td-norm 0.04 baseline: last-100 return over seeds,
    final policy entropy and share of synapses at the weight floor."""
    groups = {}
    for f in sorted(glob.glob(str(m2.OUT / "m3_*_flow_lb0.3_tn0.04*_s*.json"))):
        r = json.loads(pathlib.Path(f).read_text())
        var = pathlib.Path(f).stem.split("_tn0.04")[1].rsplit("_s", 1)[0].lstrip("_") or "baseline"
        groups.setdefault((r.get("task", "CartPole"), var), []).append(r)
    for (task, var), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "baseline", kv[0][1])):
        R, ev = [np.mean(r["returns"][-100:]) for r in rs], [r["evals"][-1] for r in rs]
        print(f"{task:11} {var:16} n={len(rs)} ret {np.mean(R):7.1f} +- {np.std(R):5.1f}  "
              f"[{' '.join(f'{x:.0f}' for x in R)}]  H(P) {np.mean([e['P_entropy'] for e in ev]):.2f}  "
              f"P_max {np.mean([e['P_max'] for e in ev]):.2f}  at_floor {np.mean([e['frac_wmin'] for e in ev]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS), default=None, help="required to train; --fidelity: default all")
    ap.add_argument("--backend", choices=("flow", "spiking", "mpr"), default="flow",
                    help="mpr: rates of the QIF network's MPR fixed point, no spikes (Phase B); --q-<field> overrides")
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="default: the task's (tasks.TASKS)")
    ap.add_argument("--engine", choices=("numpy", "snm", "qif", "lif"), default="numpy",
                    help="spiking engine: numpy = sp.run_heads (SuperNeuroMAT's raster, batched over heads); "
                         "qif = mapstdp.qif (Proposal, iteration 013); lif = mapstdp.lif (noisy LIF, --rule emergent)")
    for k, typ in LIF_FLAGS.items():
        ap.add_argument(f"--lif-{k.replace('_', '-')}", dest=f"lif_{k}", type=typ, default=None,
                        help=f"--engine lif: LIFParams.{k} (default {getattr(lif.LIFParams(), k)!r})")
    for k, typ in QIF_FLAGS.items():
        ap.add_argument(f"--q-{k.replace('_', '-')}", dest=f"q_{k}", type=typ, default=None,
                        help=f"--engine qif: QIFParams.{k} (default {getattr(qif.QIFParams(), k)!r})")
    ap.add_argument("--la-beta", type=float, default=LA_BETA, help="latent-output barrier (Eq. 4e, Proposal)")
    ap.add_argument("--td-norm", type=float, default=0.0, help="TD-error normalisation to this RMS (Proposal; 0 = off)")
    ap.add_argument("--eta-tau", type=float, default=None, help="ablation: task-rate annealing (0 = constant)")
    ap.add_argument("--wmin", type=float, default=None, help="ablation: weight floor (default 1e-3)")
    ap.add_argument("--tau-e", type=float, default=None, help="ablation: eligibility time constant (default 3)")
    ap.add_argument("--lam", type=float, default=0.05, help="structural (map) term rate; 0 = map-term ablation")
    ap.add_argument("--p-cc", type=float, default=0.1, help="controller->controller connection probability")
    ap.add_argument("--p-cross", type=float, default=0.05,
                    help="cross-module (non-routing) connection probability; 0 = sealed modules (iteration 014)")
    ap.add_argument("--beta", type=float, default=1.0, help="sharpened policy exponent (Eq. 1e, Proposal; flow)")
    ap.add_argument("--eta", type=float, default=None, help="task rate (default: model v3's)")
    ap.add_argument("--beta-kappa", type=float, default=0.0, help="causal adaptive beta gain (Eq. 1g; 0 = off)")
    ap.add_argument("--beta-max", type=float, default=4.0, help="upper bound of the adaptive beta")
    ap.add_argument("--gamma", type=float, default=m2.GAMMA, help="discount of the critic and delta (Eq. 4c)")
    ap.add_argument("--readout", choices=("race", "diffbound"), default="race",
                    help="spiking action readout: race (Eq. 1d) or count-difference bound (Eq. 1h, iteration 010)")
    ap.add_argument("--encoder", default=None, help="controller encoder spec (iteration 012): g<n>s<sigma> | "
                    "f<order>c<conj>[h][b]")
    ap.add_argument("--critic-layer", default=None, help="separate critic RF layer spec (Eq. 4c'', iteration 012)")
    ap.add_argument("--c-ent", type=float, default=0.0, help="entropy term coefficient (Eq. 4h; 0 = off)")
    ap.add_argument("--ent-target", type=float, default=None, help="adapt c_ent toward H(P~) = this x log|A|")
    ap.add_argument("--rule", choices=("v4", "emergent"), default="v4",
                    help="plasticity rule: v4 (models v3/v4) or emergent (observed walk + signed counter, Eqs. 8-12)")
    ap.add_argument("--obs-tau", type=float, default=0.0,
                    help="emergent, spiking: running averages of the walk statistics over this many frames (0 = frame)")
    ap.add_argument("--ebar", choices=("obs", "module"), default="obs", help="emergent ablation: structural baseline")
    ap.add_argument("--step-stat", choices=("c", "kappa"), default="c",
                    help="emergent ablation: structural step factor")
    ap.add_argument("--lam-scale", type=float, default=1.0, help="structural rate x this (any engine)")
    ap.add_argument("--eta-scale", type=float, default=1.0, help="task rate x this (any engine)")
    ap.add_argument("--match-scales", action="store_true",
                    help="set --lam-scale/--eta-scale by m2.match_scales (RMS match to flow at the initial W)")
    ap.add_argument("--out", default=None, help="result path (default experiments/results/<name>.json)")
    ap.add_argument("--save-model", default=None, help="write the trained model (.npz, m2.save_model) here")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--ablations", action="store_true", help="summary of the flow learning-rule ablations")
    ap.add_argument("--fidelity", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if a.ablations:
        return ablation_summary()
    if a.fidelity:
        qp = qif.QIFParams(**qif_overrides(a)) if a.engine == "qif" else None
        Hs = (a.heads,) if a.heads != 1 else HEADS
        out = {t: fidelity(t, Hs, engine=a.engine, qp=qp) for t in ([a.task] if a.task else TASKS)}
        tag = f"_{qif_token(a)}" if a.engine == "qif" else ""
        (m2.OUT / f"m3_fidelity{tag}.json").write_text(json.dumps(out, indent=1))
        return
    if a.task is None:
        ap.error("--task is required to train")
    ablate = {k: v for k, v in (("wmin", a.wmin), ("tau_e", a.tau_e)) if v is not None}
    variant = [f"{k}{v:g}" for k, v in ablate.items()] + ([f"at{a.eta_tau:g}"] if a.eta_tau is not None else [])
    variant = "_".join(variant + variant_tokens(a))
    cfg = m2.Config(a.backend, "exempt", a.lam, a.seed, beta=a.beta, eta=a.eta, eta_tau=a.eta_tau, la_beta=a.la_beta,
                    task=a.task, heads=a.heads, engine=a.engine, td_norm=a.td_norm, ablate=ablate, p_cc=a.p_cc,
                    p_cross=a.p_cross, beta_kappa=a.beta_kappa, beta_max=a.beta_max, gamma=a.gamma, readout=a.readout,
                    encoder=a.encoder, critic_layer=a.critic_layer, c_ent=a.c_ent, ent_target=a.ent_target,
                    qif=qif_overrides(a) if a.engine == "qif" or a.backend == "mpr" else {}, rule=a.rule,
                    lif=lif_overrides(a) if a.engine == "lif" else {},
                    lam_scale=a.lam_scale, eta_scale=a.eta_scale, obs_tau=a.obs_tau, ebar=a.ebar,
                    step_stat=a.step_stat)
    scales = None
    if a.match_scales:
        assert (a.q_lam_scale or 1.0) == 1.0 and (a.q_eta_scale or 1.0) == 1.0, "--match-scales: QIFParams scales 1"
        assert (a.lif_lam_scale or 1.0) == 1.0 and (a.lif_eta_scale or 1.0) == 1.0, "--match-scales: LIFParams scales 1"
        scales = m2.match_scales(cfg)
        cfg = dataclasses.replace(cfg, lam_scale=cfg.lam_scale * scales["lam_scale"],
                                  eta_scale=cfg.eta_scale * scales["eta_scale"])
        print("match_scales: " + " ".join(f"{k} {v:.4g}" for k, v in scales.items()), flush=True)
    r = m2.run(cfg, a.episodes or TASKS[a.task].episodes,
               name=name(a.task, a.backend, a.heads, a.la_beta, a.seed, a.td_norm, variant), out=a.out,
               save_model_path=a.save_model, scales=scales)
    print(f"done: last100 {np.mean(r['returns'][-100:]):.1f}, {r['frames']} frames, {r['ms_per_frame']:.2f} ms/frame")


if __name__ == "__main__":
    main()
