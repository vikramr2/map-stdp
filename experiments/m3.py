"""M3 (docs/SPEC.md §11): model v3 with the opened latents (latent-output barrier Eq. 4e, beta_LA = 0.3) on CartPole,
MountainCar, Acrobot and LunarLander, flow against spiking with H parallel heads (Proposal). Run from the repo root:
  python -m experiments.m3 --task {CartPole,MountainCar,Acrobot,LunarLander} --backend {flow,spiking} [--heads H]
    --seed 0 [--episodes E]   -> experiments/results/m3_<task>_<backend>[_h<H>]_lb<beta_LA>_s<seed>.json
  Variants (iteration 006, flow): --p-cc P (controller recurrence, default 0.1), --beta B [--beta-end B2 --beta-ramp R]
    (sharpened policy Eq. 1e; --beta-end/--beta-ramp ramp B -> B2 over R episodes: an ORACLE reference ablation that
    needs the run length, never a default or a recommended setting), --eta E (task rate); file-name tokens
    pcc<P>, b<B>[-<B2>r<R>], eta<E> after the ablation tokens; --beta-kappa k [--beta-max B] (causal adaptive beta,
    token bk<k>m<B>), --score-unit (Eq. 4b increment / beta, token su), --critic net (network-activity critic).
  Iteration 007 (adaptive beta, Proposal): --beta-eps e [--beta-set B0] (mean reversion of log beta toward a fixed set
    point B0 or, without --beta-set, upward toward an experience-dependent one, leaking over --beta-decay D episodes
    (0: strict running maximum); tokens re<e>[s<B0>][d<D>] after bk..m..),
    --beta-signal trace (per-frame update from the reset-at-termination reward trace, no episode boundaries;
    --beta-kappa is then per frame; token tr).
  Iteration 008 (Proposal): --beta-step-cap c [--beta-step-shape {min,sat,invu,hyb}] (Eq. 4b increment scaled by
    beta_eta / beta, beta_eta = min(beta, c), c beta / (c + beta), beta exp(1 - beta / c) or beta up to c and
    c (1 + log(beta / c)) above; policy keeps the full beta; token sc<c>[sat|iu|hy] after the re.. token; excludes
    --score-unit).
  Iteration 009 (Proposal): --critic-lambda l (TD(lambda) critic, Eq. 4c with a trace; token cl<l>), --gamma g (critic
    and delta discount; token g<g>), both after the sc.. token; --tau-e T (task trace, ablation token tau_e<T>).
  Iteration 010 (Proposal, spiking): --readout diffbound (count-difference race to a bound, theta from beta through
    m2.BOUND_CAL; combine with --beta B or --beta-kappa k --beta-max B; token dbc after the cl../g.. tokens).
  Iteration 012 (Proposal, flow): --encoder SPEC (controller encoder, tasks.with_encoder: g<n>s<sigma> grid or
    f<order>c<conj>[h][b] Fourier; token enc<SPEC>), --critic-layer SPEC (separate critic on its own fixed RF layer,
    tasks.critic_layer; token crit<SPEC>), --c-ent c [--ent-target H0] (C1 entropy term; token ce<c>[H<H0>]),
    --c-use c [--use-tau T] (C2 usage homeostasis; token cu<c>[t<T>]); all after the dbc token, before eta.
    --episodes E off the task's default adds the token ep<E> (after su).
  python -m experiments.m3 --summary    -> per task: random, flow and spiking per H (last-100 return over seeds)
  python -m experiments.m3 --fidelity   -> per task and H: error of the pooled pi-hat and kappa against the flow
Heads: H independent spiking replicas of the network on the same W (tasks share one block-diagonal SuperNeuroMAT
network). Their spikes are pooled: pi-hat from summed counts, kappa the sum of per-head covariance counts divided by
the total spikes, one race over all heads (still Eq. 1d). H = 1 is the M2 spiking agent."""
import argparse
import glob
import json
import pathlib

import numpy as np

from experiments import m2
from experiments.tasks import TASKS
from mapstdp import core
from mapstdp import spiking as sp

LA_BETA, HEADS = 0.3, (1, 2, 4, 8, 16)


def name(task, backend, heads, la_beta, seed, td_norm=0.0, variant=""):
    return f"m3_{task}_{backend}" + (f"_h{heads}" if backend == "spiking" else "") + f"_lb{la_beta:g}" + \
        (f"_tn{td_norm:g}" if td_norm else "") + (f"_{variant}" if variant else "") + f"_s{seed}"


def fidelity(task, Hs=HEADS, reps=20, n_obs=10, seed=0):
    """Learning-signal error of H pooled heads against the flow on the initial W, over held-out stimuli: L1 of pi-hat
    to the n-hop flow, and 1 - corr of kappa with J (the flow's expected pairings), mean over reps frames."""
    agent = m2.Agent("flow", "exempt", 0.05, seed, la_beta=LA_BETA, task=task)
    rows = []
    for H in Hs:
        rng, l1, rk = np.random.default_rng(seed), [], []
        for v in agent.held[:n_obs]:
            pi = core.flow(agent.W, v, m2.ALPHA, m2.N_HOPS)
            J = agent.W / agent.W.sum(0) * pi
            for _ in range(reps):
                S = sp.run_heads(agent.W, v, H, m2.K_SPK, m2.RATE, rng, m2.THETA, 1 - m2.ALPHA)
                ph, kap, _ = sp.pooled(S, H)
                l1.append(np.abs(ph - pi).sum())
                rk.append(1 - np.corrcoef(kap[agent.mask], J[agent.mask])[0, 1])
        rows.append({"H": H, "L1_pi": float(np.mean(l1)), "1-corr_kappa_J": float(np.mean(rk))})
        print(f"{task:12} H={H:2d}  L1(pi_hat, pi) {rows[-1]['L1_pi']:.4f}  1-corr(kappa, J) "
              f"{rows[-1]['1-corr_kappa_J']:.4f}", flush=True)
    return rows


def nondefault(r):
    """True for an iteration-006 variant (lam, p_cc, beta or its ramp, or the task rate off their M3 defaults)."""
    eta0 = m2.ETA if r["backend"] == "flow" else r["params"]["eta_task"]
    return r.get("lam", 0.05) != 0.05 or r.get("p_cc", 0.1) != 0.1 or r.get("beta", 1.0) != 1.0 or \
        r.get("beta_end") is not None or \
        r.get("eta_task", eta0) != eta0 or bool(r.get("beta_kappa")) or bool(r.get("score_unit")) or \
        bool(r.get("beta_eps")) or r.get("beta_signal", "episode") != "episode" or bool(r.get("beta_step_cap")) or \
        bool(r.get("critic_lambda")) or r.get("gamma", m2.GAMMA) != m2.GAMMA or \
        r.get("ablate", {}).get("tau_e", m2.TAU_E) != m2.TAU_E or r.get("readout", "race") != "race" or \
        bool(r.get("encoder_spec") or r.get("critic_layer") or r.get("c_ent") or r.get("c_use")) or \
        r["episodes"] != TASKS[r.get("task", "CartPole")].episodes


def variant_tokens(a):
    """File-name tokens of the iteration-006 variants, empty on the defaults."""
    t = ([f"lam{a.lam:g}"] if a.lam != 0.05 else []) + ([f"pcc{a.p_cc:g}"] if a.p_cc != 0.1 else [])
    if a.beta != 1.0 or a.beta_end is not None:
        t.append(f"b{a.beta:g}" + (f"-{a.beta_end:g}r{a.beta_ramp:g}" if a.beta_end is not None else ""))
    t += [f"bk{a.beta_kappa:g}m{a.beta_max:g}"] if a.beta_kappa else []
    t += [f"re{a.beta_eps:g}" + (f"s{a.beta_set:g}" if a.beta_set else "") + (f"d{a.beta_decay:g}" if a.beta_decay
                                                                              else "")] if a.beta_eps else []
    t += [f"sc{a.beta_step_cap:g}" + {"min": "", "sat": "sat", "invu": "iu", "hyb": "hy"}[a.beta_step_shape]] \
        if a.beta_step_cap else []
    t += ["tr"] if a.beta_signal == "trace" else []
    t += ([f"cl{a.critic_lambda:g}"] if a.critic_lambda else []) + ([f"g{a.gamma:g}"] if a.gamma != m2.GAMMA else [])
    t += ["dbc"] if a.readout == "diffbound" else []
    t += ([f"enc{a.encoder}"] if a.encoder else []) + ([f"crit{a.critic_layer}"] if a.critic_layer else [])
    t += [f"ce{a.c_ent:g}" + (f"H{a.ent_target:g}" if a.ent_target is not None else "")] if a.c_ent else []
    t += [f"cu{a.c_use:g}" + (f"t{a.use_tau:g}" if a.use_tau != m2.USE_TAU else "")] if a.c_use else []
    t += ([f"eta{a.eta:g}"] if a.eta is not None else []) + (["su"] if a.score_unit else [])
    return t + ([f"ep{a.episodes}"] if a.episodes and a.episodes != TASKS[a.task].episodes else [])


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
    ap.add_argument("--backend", choices=("flow", "spiking"), default="flow")
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="default: the task's (tasks.TASKS)")
    ap.add_argument("--engine", choices=("numpy", "snm"), default="numpy",
                    help="spiking engine: numpy = sp.run_heads (SuperNeuroMAT's raster, batched over heads)")
    ap.add_argument("--la-beta", type=float, default=LA_BETA, help="latent-output barrier (Eq. 4e, Proposal)")
    ap.add_argument("--td-norm", type=float, default=0.0, help="TD-error normalisation to this RMS (Proposal; 0 = off)")
    ap.add_argument("--eta-tau", type=float, default=None, help="ablation: task-rate annealing (0 = constant)")
    ap.add_argument("--wmin", type=float, default=None, help="ablation: weight floor (default 1e-3)")
    ap.add_argument("--tau-e", type=float, default=None, help="ablation: eligibility time constant (default 3)")
    ap.add_argument("--critic", choices=("ctrl", "rich", "net"), default=None,
                    help="ablation: rich = RFs on the full obs (abstraction); net = controller + latent flow shares")
    ap.add_argument("--lam", type=float, default=0.05, help="structural (map) term rate; 0 = map-term ablation")
    ap.add_argument("--p-cc", type=float, default=0.1, help="controller->controller connection probability")
    ap.add_argument("--beta", type=float, default=1.0, help="sharpened policy exponent (Eq. 1e, Proposal; flow)")
    ap.add_argument("--beta-end", type=float, default=None,
                    help="ORACLE reference ablation, not part of the model: ramp beta linearly to this value (flow)")
    ap.add_argument("--beta-ramp", type=float, default=None,
                    help="ORACLE reference ablation: ramp length in episodes (default: half the run)")
    ap.add_argument("--eta", type=float, default=None, help="task rate (default: model v3's)")
    ap.add_argument("--beta-kappa", type=float, default=0.0, help="causal adaptive beta gain (0 = off)")
    ap.add_argument("--beta-max", type=float, default=4.0, help="upper bound of the adaptive beta")
    ap.add_argument("--score-unit", action="store_true", help="ablation: Eq. 4b increment divided by beta")
    ap.add_argument("--beta-step-cap", type=float, default=0.0,
                    help="cap c on beta's gain of the Eq. 4b increment (min(beta, c)); policy keeps beta; 0 = off")
    ap.add_argument("--beta-step-shape", choices=("min", "sat", "invu", "hyb"), default="min",
                    help="shape of the plasticity gain under --beta-step-cap: hard cap, saturating, inverted U, "
                         "identity then logarithmic")
    ap.add_argument("--beta-eps", type=float, default=0.0, help="adaptive beta: mean-reversion rate of log beta")
    ap.add_argument("--beta-set", type=float, default=0.0,
                    help="adaptive beta: fixed set point (two-sided); 0 = experience-dependent set point, upward only")
    ap.add_argument("--beta-decay", type=float, default=0.0,
                    help="adaptive beta: leak time (episodes) of the experience-dependent set point; 0 = strict max")
    ap.add_argument("--beta-signal", choices=("episode", "trace"), default="episode",
                    help="adaptive beta: per-episode return trend (Eq. 1g) or per-frame reward trace (no boundaries)")
    ap.add_argument("--critic-lambda", type=float, default=0.0,
                    help="TD(lambda) critic trace (Eq. 4c with z <- gamma lambda z + x; Proposal); 0 = Eq. 4c")
    ap.add_argument("--gamma", type=float, default=m2.GAMMA, help="discount of the critic and delta (Eq. 4c)")
    ap.add_argument("--readout", choices=("race", "diffbound"), default="race",
                    help="spiking action readout: race (Eq. 1d) or count-difference bound (Proposal, iteration 010)")
    ap.add_argument("--encoder", default=None, help="controller encoder spec (iteration 012): g<n>s<sigma> | "
                    "f<order>c<conj>[h][b]")
    ap.add_argument("--critic-layer", default=None, help="separate critic RF layer spec (iteration 012)")
    ap.add_argument("--c-ent", type=float, default=0.0, help="C1 entropy term coefficient (0 = off)")
    ap.add_argument("--ent-target", type=float, default=None, help="C1: adapt c_ent toward H(P~) = this x log|A|")
    ap.add_argument("--c-use", type=float, default=0.0, help="C2 usage homeostasis coefficient (0 = off)")
    ap.add_argument("--use-tau", type=float, default=m2.USE_TAU, help="C2 usage average time constant (frames)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--ablations", action="store_true", help="summary of the flow learning-rule ablations")
    ap.add_argument("--fidelity", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if a.ablations:
        return ablation_summary()
    if a.fidelity:
        out = {t: fidelity(t) for t in ([a.task] if a.task else TASKS)}
        (m2.OUT / "m3_fidelity.json").write_text(json.dumps(out, indent=1))
        return
    if a.task is None:
        ap.error("--task is required to train")
    episodes = a.episodes or TASKS[a.task].episodes
    ablate = {k: v for k, v in (("wmin", a.wmin), ("tau_e", a.tau_e), ("critic", a.critic)) if v is not None}
    variant = "_".join(f"{k}{v:g}" if isinstance(v, float) else f"{k}-{v}" for k, v in ablate.items())
    variant += (("_" if variant else "") + f"at{a.eta_tau:g}") if a.eta_tau is not None else ""
    if a.beta_end is not None and a.beta_ramp is None:
        a.beta_ramp = episodes / 2
    variant = "_".join(([variant] if variant else []) + variant_tokens(a))
    r = m2.run(a.backend, "exempt", a.lam, a.seed, episodes, beta=a.beta, eta_tau=a.eta_tau, eta=a.eta,
               la_beta=a.la_beta, task=a.task, heads=a.heads,
               name=name(a.task, a.backend, a.heads, a.la_beta, a.seed, a.td_norm, variant), engine=a.engine,
               td_norm=a.td_norm, ablate=ablate, p_cc=a.p_cc, beta_end=a.beta_end, beta_ramp=a.beta_ramp,
               beta_kappa=a.beta_kappa, beta_max=a.beta_max, score_unit=a.score_unit, beta_eps=a.beta_eps,
               beta_set=a.beta_set, beta_signal=a.beta_signal, beta_decay=a.beta_decay,
               beta_step_cap=a.beta_step_cap, beta_step_shape=a.beta_step_shape, critic_lambda=a.critic_lambda,
               gamma=a.gamma, readout=a.readout, encoder=a.encoder, critic_layer=a.critic_layer, c_ent=a.c_ent,
               c_use=a.c_use, use_tau=a.use_tau, ent_target=a.ent_target)
    print(f"done: last100 {np.mean(r['returns'][-100:]):.1f}, {r['frames']} frames, {r['ms_per_frame']:.2f} ms/frame")


if __name__ == "__main__":
    main()
