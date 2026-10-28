"""Final reported studies: reports/<study>/ holds only what this runner writes (experiments/results/ is the
intermediates store: dev, tuning and diagnostic runs, and the default output of every CLI). Run from the repo root:
  python -m experiments.report run <study> [--workers N]
    -> reports/<study>/runs/<group>/s<seed>.json (+ manifest.json, code.patch if the tree is dirty); skips finished
       runs, resumes checkpointed ones (*.ckpt.npz, gitignored); refuses to launch if mapstdp/ or experiments/ differ
       from the code state recorded in the manifest; summarizes at the end
  python -m experiments.report summarize <study>   -> reports/<study>/summary.csv, summary.md, from runs/ only
Studies are declared in STUDIES: groups -> (task, engine, flags, seeds), a reference group per task (Mann-Whitney) and
gates (minimum solved count). A reference "<study>/<group>" names a group of another study (a companion study added
under a later code state compares against it without rerunning it). Statistics are experiments.tdstdp.group_stats,
the same as tdstdp --summary.
A study's `runner` is "tdstdp" (default: experiments.tdstdp, the statistics above) or "m3" (experiments.m3 with
M3_ENGINE_FLAG[engine], --out and --save-model reports/<study>/models/<group>/s<seed>.npz; statistics m3_stats; a
study's `pairs` lists (on, off[, family]) groups compared by two-sided Mann-Whitney, Holm-adjusted per metric within
a family)."""
import argparse
import concurrent.futures as cf
import csv
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

from experiments import m2, tdstdp

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs" / "report"
ENGINE_FLAG = {"lif": ["--engine", "lif"], "branching": [],  # branching = the numpy engine (tdstdp default)
               "flow": ["--engine", "flow"], "nlif": ["--engine", "nlif"]}  # nlif: noisy LIF (iteration 015)
M3_ENGINE_FLAG = {"flow": ["--backend", "flow"],  # runner 'm3': engine -> experiments.m3 flags (heads per study)
                  "branching": ["--backend", "spiking", "--engine", "numpy"],
                  "qif": ["--backend", "spiking", "--engine", "qif"],
                  "lif": ["--backend", "spiking", "--engine", "lif"]}
M3_THR = {"CartPole": 475.0, "LunarLander": 200.0}  # trailing-100 mean thresholds of m3_stats

_CP, _LL = range(1000, 1020), range(1000, 1010)
# Display names (2026-10-24): "sgd" read as backprop SGD; the arm is the local rule without Adam (--sgd = --eta-cal).
# The keys stay the run directories and the recorded commands.
_NOADAM_LABELS = {f"LunarLander_sgd_{e}": f"LunarLander_noadam_{e}" for e in ("lif", "branching", "flow")}
STUDIES = {
    "tdstdp-baseline": dict(
        title="TD-STDP baseline (Chung & Kozma 2020, authors' code path), block sum B = 20, --fast",
        groups={
            "CartPole_noadam_lif": dict(task="CartPole", engine="lif", flags=["--block", "20", "--fast"], seeds=_CP),
            "CartPole_noadam_branching": dict(task="CartPole", engine="branching", flags=["--block", "20", "--fast"],
                                              seeds=_CP),
            "CartPole_adam_lif": dict(task="CartPole", engine="lif", flags=["--adam", "--block", "20", "--fast"],
                                      seeds=_CP),
            "CartPole_adam_branching": dict(task="CartPole", engine="branching",
                                            flags=["--adam", "--block", "20", "--fast"], seeds=_CP),
            "LunarLander_adam_lif": dict(task="LunarLander", engine="lif", flags=["--block", "20", "--fast"],
                                         seeds=_LL),
            "LunarLander_adam_branching": dict(task="LunarLander", engine="branching",
                                               flags=["--block", "20", "--fast"], seeds=_LL),
            "LunarLander_sgd_lif": dict(task="LunarLander", engine="lif", flags=["--sgd", "--block", "20", "--fast"],
                                        seeds=_LL),
            "LunarLander_sgd_branching": dict(task="LunarLander", engine="branching",
                                              flags=["--sgd", "--block", "20", "--fast"], seeds=_LL),
        },
        reference={"CartPole": "CartPole_noadam_lif", "LunarLander": "LunarLander_adam_lif"},
        labels=_NOADAM_LABELS,
        gates={"CartPole_noadam_lif": 18, "LunarLander_adam_lif": 6},
        # Declared and run, then dropped (2026-10-24): kept in runs/, left out of summary.csv and the tables.
        dropped={k: "the authors' Adam step is the per-ms eta, 40-160x LunarLander's for CartPole, and collapses "
                    "the policy from the first episodes; CK report no CartPole Adam config and CartPole needs no Adam"
                 for k in ("CartPole_adam_lif", "CartPole_adam_branching")},
    ),
    "tdstdp-baseline-flow": dict(
        title="TD-STDP baseline, flow engine (the branching step's noise-free limit), block sum B = 20, --fast",
        groups={
            "CartPole_noadam_flow": dict(task="CartPole", engine="flow", flags=["--block", "20", "--fast"], seeds=_CP),
            # block-sum control: B = 20 was gated on branching only (iteration 013 §11, S1)
            "CartPole_noadam_flow_b1": dict(task="CartPole", engine="flow", flags=["--block", "1", "--fast"],
                                            seeds=_CP),
            "LunarLander_adam_flow": dict(task="LunarLander", engine="flow", flags=["--block", "20", "--fast"],
                                          seeds=_LL),
            "LunarLander_sgd_flow": dict(task="LunarLander", engine="flow", flags=["--sgd", "--block", "20", "--fast"],
                                         seeds=_LL),
        },
        reference={"CartPole": "tdstdp-baseline/CartPole_noadam_lif",
                   "LunarLander": "tdstdp-baseline/LunarLander_adam_lif"},
        gates={},
        labels=_NOADAM_LABELS,
    ),
}


def _emergent_v4():
    """emergent-v4 (iteration 014, CartPole only after the second pilot, §10): engine x map term on (lam 0.05) / off
    (lam 0) / sealed (lam 0 on a network built without cross-module synapses, --p-cross 0), v4 flags, rule emergent.
    Spiking engines: 8 heads, diffbound readout (BOUND_CAL[8]), step sizes matched to flow; LIF caps beta at 1.6
    (pilot: v4's 16 put 36-61% of LIF frames on the deadline, beyond its calibration range, BOUND_CAL_LIF)."""
    v4 = ["--rule", "emergent", "--td-norm", "0.04", "--p-cc", "0", "--beta-kappa", "0.01", "--eta", "5"]
    sp = ["--heads", "8", "--readout", "diffbound", "--match-scales"]
    eng = {"flow": ["--beta-max", "16"], "branching": ["--beta-max", "16", *sp], "qif": ["--beta-max", "16", *sp],
           "lif": ["--beta-max", "1.6", *sp]}
    arms = {"on": ["--lam", "0.05"], "off": ["--lam", "0"], "sealed": ["--lam", "0", "--p-cross", "0"]}
    groups, pairs = {}, []
    for e, f in eng.items():
        for arm, a in arms.items():
            groups[f"CartPole_{e}_{arm}"] = dict(task="CartPole", engine=e, seeds=range(2000, 2020),
                                                 flags=[*v4, *f, *a, "--episodes", "1500"])
        pairs += [(f"CartPole_{e}_on", f"CartPole_{e}_off"), (f"CartPole_{e}_on", f"CartPole_{e}_sealed", "sealed")]
    return dict(title="Emergent-walk Map-STDP on v4, CartPole: map term on vs off vs sealed, four engines",
                runner="m3", groups=groups, pairs=pairs, reference={}, gates={})


STUDIES["emergent-v4"] = _emergent_v4()


def _hybrid_columns():
    """hybrid-columns (iteration 015): Map-STDP action columns in the TD-STDP backbone, no Adam, low-rate noisy LIF
    (--rate-target 40). base = no recurrence; three regimes of modularity on recurrent columns: cols_lam0 (implicit:
    task term only), cols_on (learned: plus the map term, r = 1), cols_sealed (enforced: no cross-column synapses,
    task term only); lat_on = cols_on plus 2 latent modules. chi_rec 0.3 and r 1 frozen in pilots B and C (§5, §6).
    Step rule as the baseline: preset eta on CartPole, --eta-cal on LunarLander. Headline: each regime vs base
    (improvement, and non-inferiority at margin 25 in the README); secondary: between regimes and lat_on."""
    rec = ["--chi-rec", "0.3", "--rec"]
    arms = {"base": [], "cols_lam0": [*rec, "cols", "--lam", "0"], "cols_on": [*rec, "cols", "--lam", "1"],
            "cols_sealed": [*rec, "cols", "--rec-seal", "--lam", "0"], "lat_on": [*rec, "lat", "--lam", "1"]}
    task = {"CartPole": [], "LunarLander": ["--eta-cal"]}
    groups = {f"{t}_{a}": dict(task=t, engine="nlif", seeds=range(3000, 3020),
                               flags=["--rate-target", "40", "--block", "20", "--fast", *tf, *af])
              for t, tf in task.items() for a, af in arms.items()}
    pairs = [(f"{t}_{a}", f"{t}_{b}") for t in task for a, b in
             (("cols_lam0", "base"), ("cols_on", "base"), ("cols_sealed", "base"))]
    pairs += [(f"{t}_{a}", f"{t}_{b}", "secondary") for t in task for a, b in
              (("cols_on", "cols_lam0"), ("cols_on", "cols_sealed"), ("lat_on", "cols_on"))]
    return dict(title="Map-STDP columns in the TD-STDP backbone, no Adam, noisy LIF at 40 Hz",
                groups=groups, pairs=pairs, reference={t: f"{t}_base" for t in task}, gates={})


STUDIES["hybrid-columns"] = _hybrid_columns()


def _hybrid_columns_2():
    """hybrid-columns-2 (iteration 016): the regimes of hybrid-columns on Acrobot (exclusive torques that must be
    sequenced), preset by the pre-set rules of tdstdp.PRESETS (--eta-cal). Taxi failed the pilot's floor gate (no
    deliveries; iteration 016 §9) and is not in the study. Every arm is rate- and entropy-matched at init
    (--rate-match-rec --ent-match-rec; no-ops for base; iteration 016 §10-11). Primary: AUC. Pairs as hybrid-columns
    without lat_on."""
    rec = ["--chi-rec", "0.3", "--rec"]
    arms = {"base": [], "cols_lam0": [*rec, "cols", "--lam", "0"], "cols_on": [*rec, "cols", "--lam", "1"],
            "cols_sealed": [*rec, "cols", "--rec-seal", "--lam", "0"]}
    tasks = ("Acrobot",)
    groups = {f"{t}_{a}": dict(task=t, engine="nlif", seeds=range(4000, 4020),
                               flags=["--rate-target", "40", "--block", "20", "--fast", "--eta-cal",
                                      "--rate-match-rec", "--ent-match-rec", *af])
              for t in tasks for a, af in arms.items()}
    pairs = [(f"{t}_{a}", f"{t}_{b}") for t in tasks for a, b in
             (("cols_lam0", "base"), ("cols_on", "base"), ("cols_sealed", "base"))]
    pairs += [(f"{t}_{a}", f"{t}_{b}", "secondary") for t in tasks for a, b in
              (("cols_on", "cols_lam0"), ("cols_on", "cols_sealed"), ("cols_lam0", "cols_sealed"))]
    return dict(title="Map-STDP columns in the TD-STDP backbone on Acrobot, no Adam, noisy LIF at 40 Hz",
                groups=groups, pairs=pairs, reference={t: f"{t}_base" for t in tasks}, gates={})


STUDIES["hybrid-columns-2"] = _hybrid_columns_2()
COLUMNS = ["group", "task", "engine", "flags", "n_declared", "n", "solved", "T_f_mean", "T_f_sd", "T_f_n", "T_s_mean",
           "T_s_sd", "T_s_median", "T_m_mean", "T_m_sd", "T_m_n", "ret_mean", "ret_sd", "ret_ge_M", "ms_per_frame",
           "resumes", "mw_ref", "mw_T_f", "mw_T_s", "mw_ret", "gate_min_solved", "gate_pass"]


def command(g, seed, out, runner="tdstdp", model=None):
    """The run's command line: experiments.tdstdp, or for runner 'm3' experiments.m3 (model: --save-model path)."""
    if runner == "m3":
        return [sys.executable, "-m", "experiments.m3", "--task", g["task"], "--seed", str(seed),
                *M3_ENGINE_FLAG[g["engine"]], *g["flags"], "--out", str(out), "--save-model", str(model)]
    return [sys.executable, "-m", "experiments.tdstdp", "--task", g["task"], "--seed", str(seed),
            *ENGINE_FLAG[g["engine"]], *g["flags"], "--out", str(out)]


def run_path(study, group, seed, root=REPORTS):
    return root / study / "runs" / group / f"s{seed}.json"


def model_path(study, group, seed, root=REPORTS):
    return root / study / "models" / group / f"s{seed}.npz"


def study_command(study, k, sd):
    """command() of group k, seed sd of a declared study, with repo-relative paths."""
    s = STUDIES[study]
    return command(s["groups"][k], sd, run_path(study, k, sd).relative_to(ROOT), s.get("runner", "tdstdp"),
                   model_path(study, k, sd).relative_to(ROOT))


def code_state():
    """(patch, sha256): `git diff HEAD` of mapstdp/ and experiments/ (results excluded) plus every untracked file there,
    as a patch; sha256 of the patch bytes ('' if the tree is clean)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa: E731
    spec = ["--", "mapstdp", "experiments", ":(exclude)experiments/results"]
    patch = git("diff", "HEAD", *spec)
    for f in git("ls-files", "--others", "--exclude-standard", *spec).split():
        patch += git("diff", "--no-index", "--", "/dev/null", f)
    return patch, hashlib.sha256(patch.encode()).hexdigest()


def versions():
    import gymnasium
    import numpy
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "gymnasium": gymnasium.__version__}


def manifest(study, workers):
    s, d = STUDIES[study], REPORTS / study
    patch, sha = code_state()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    groups = {k: {"task": g["task"], "engine": g["engine"], "flags": g["flags"], "seeds": list(g["seeds"]),
                  "commands": [" ".join(["python", *study_command(study, k, sd)[1:]]) for sd in g["seeds"]]}
              for k, g in s["groups"].items()}
    return {"study": study, "title": s["title"], "created": datetime.datetime.now().astimezone().isoformat(),
            "git_head": head, "dirty": bool(patch), "code_sha256": sha, "code_patch": "code.patch" if patch else None,
            "versions": versions(), "env": {"OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}, "workers": workers,
            "reference": s["reference"], "gates": s["gates"], "groups": groups, "launches": []}, patch, d


def run(study, workers):
    m, patch, d = manifest(study, workers)
    mf = d / "manifest.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        if old["code_sha256"] != m["code_sha256"] or old["git_head"] != m["git_head"]:
            sys.exit(f"code state differs from {mf} (git {old['git_head'][:7]}, sha {old['code_sha256'][:12]}): "
                     f"a study must use one code state; delete reports/{study}/runs and manifest.json to relaunch")
        m = old
    else:
        d.mkdir(parents=True, exist_ok=True)
        if patch:
            (d / "code.patch").write_text(patch)
    todo = [(k, sd) for k, g in STUDIES[study]["groups"].items() for sd in g["seeds"]
            if not run_path(study, k, sd).exists()]
    m["launches"].append({"date": datetime.datetime.now().astimezone().isoformat(), "workers": workers,
                          "pending": len(todo), "resuming": sum(run_path(study, k, sd).with_suffix(".ckpt.npz").exists()
                                                                for k, sd in todo)})
    mf.write_text(json.dumps(m, indent=1))
    (LOGS / study).mkdir(parents=True, exist_ok=True)
    env = os.environ | {f"{k}_NUM_THREADS": "1" for k in ("OMP", "MKL", "OPENBLAS", "NUMBA")}

    def one(k, sd):
        out = run_path(study, k, sd)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(LOGS / study / f"{k}_s{sd}.log", "a") as log:
            rc = subprocess.run(study_command(study, k, sd), cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        print(f"{time.strftime('%H:%M:%S')} {k} s{sd} rc {rc}", flush=True)
        return rc

    print(f"{study}: {len(todo)} runs pending, {workers} workers", flush=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(lambda a: one(*a), todo))
    bad = sum(rc != 0 for rc in rcs)
    print(f"{study}: done, {bad} failed", flush=True)
    return bad


def fmt(x, nd=1):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return "inf" if x == float("inf") else "" if x != x else f"{x:.{nd}f}"
    return str(x)


def pm(r, a):
    """'mean ± sd [count]' of statistic a in a summary row, or '–' if no run reaches it."""
    return f"{fmt(r[a + '_mean'])} ± {fmt(r[a + '_sd'])} [{r[a + '_n']}]" if r[a + "_n"] else "–"


def load_runs(d, k):
    """Finished runs (s<seed>.json) of group k in study directory d."""
    return [json.loads(p.read_text()) for p in sorted((d / "runs" / k).glob("s*.json"))]


_auc_min = lambda r: float(np.mean(r["returns"][:r["config"]["episodes"]]))  # noqa: E731
TD_PAIR_METRICS = {  # tdstdp-runner pairs (iteration 015); primary: LunarLander last-100 return, other tasks AUC
    "primary": lambda r: float(np.mean(r["returns"][-100:])) if r["config"]["task"] == "LunarLander" else _auc_min(r),
    "auc": _auc_min, "ret": lambda r: float(np.mean(r["returns"][-100:]))}


def summarize(study, root=REPORTS, studies=None):
    """summary.csv and summary.md of reports/<study>/ from runs/ only (finished runs, s<seed>.json)."""
    s, d = (studies or STUDIES)[study], root / study
    if s.get("runner", "tdstdp") == "m3":
        return summarize_m3(study, s, d)
    runs = {k: load_runs(d, k) for k in s["groups"]}
    for ref in s["reference"].values():  # "<study>/<group>": a group of another study
        if "/" in ref:
            runs[ref] = load_runs(root / ref.split("/")[0], ref.split("/")[1])
    rows = []
    dropped = s.get("dropped", {})
    label = lambda k: s.get("labels", {}).get(k, k)  # noqa: E731
    for k, g in s["groups"].items():
        if k in dropped:
            continue
        ref = s["reference"].get(g["task"])
        row = {"group": label(k), "task": g["task"], "engine": g["engine"], "flags": " ".join(g["flags"]),
               "n_declared": len(g["seeds"]), "n": len(runs[k])}
        if runs[k]:
            st = tdstdp.group_stats(g["task"], runs[k], runs[ref] if k != ref and runs.get(ref) else None)
            row |= st | {"resumes": sum(r.get("resumes", 0) for r in runs[k])}
            row["mw_ref"] = label(ref) if "mw_T_f" in st else ""
        if k in s["gates"]:
            row["gate_min_solved"] = s["gates"][k]
            done = row["n"] == row["n_declared"]
            row["gate_pass"] = row.get("solved", 0) >= s["gates"][k] if done else "incomplete"
        rows.append(row)
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows([{c: (repr(r[c]) if isinstance(r.get(c), float) else r.get(c, "")) for c in COLUMNS} for r in rows])
    mf = d / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    md = [f"# {study}: summary", "",
          f"Generated by `python -m experiments.report summarize {study}` from `runs/` only. Code: git "
          f"`{man.get('git_head', '?')[:7]}`, patch sha256 `{man.get('code_sha256', '?')[:12]}`.", "",
          "Columns: solved = CK's criterion; $T_f$, $T_s$, $T_m$ = mean ± sd (ddof 0) over the runs that reach them "
          "[count]; median $T_s$ counts unsolved runs as infinite; return = last-100 mean, mean ± sd over seeds; "
          "MW = two-sided Mann–Whitney p against the task's reference group on $T_f$ / $T_s$ (never reached = episode "
          "cap + 1) / return.", ""]
    for task in dict.fromkeys(g["task"] for g in s["groups"].values()):
        p = tdstdp.PAPER.get(task)
        md += [f"## {task}", "", (f"CK 2020 (10 runs, all solved): $T_f$ {p['T_f'][0]:.1f} ± {p['T_f'][1]:.1f}, "
               f"$T_s$ {p['T_s'][0]:.1f} ± {p['T_s'][1]:.1f}. " if p else "No CK 2020 result. ") +
               f"Reference group: `{label(s['reference'].get(task, '-'))}`. "
               f"$T_m$ threshold: {tdstdp.M_THR[task]:g}.", "",
               "| Group | n | Solved | $T_f$ | $T_s$ (median) | $T_m$ | Return | ms/frame "
               "| MW $T_f$ / $T_s$ / return | Gate |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in (r for r in rows if r["task"] == task):
            if not r["n"]:
                md.append(f"| `{r['group']}` | 0/{r['n_declared']} | | | | | | | | |")
                continue
            ts = (f"{fmt(r['T_s_mean'])} ± {fmt(r['T_s_sd'])} ({fmt(r['T_s_median'], 0)})" if r["solved"]
                  else f"– ({fmt(r['T_s_median'], 0)})")
            mw = " / ".join(f"{r[c]:.2g}" for c in ("mw_T_f", "mw_T_s", "mw_ret")) if r.get("mw_ref") else "ref"
            verdict = {True: "**pass**", False: "**fail**"}.get(r.get("gate_pass"), "incomplete")
            gate = f"≥ {r['gate_min_solved']}: {verdict}" if "gate_min_solved" in r else ""
            md.append(f"| `{r['group']}` | {r['n']}/{r['n_declared']} | {r['solved']}/{r['n']} | {pm(r, 'T_f')} | "
                      f"{ts} | {pm(r, 'T_m')} | {fmt(r['ret_mean'])} ± {fmt(r['ret_sd'])} | {r['ms_per_frame']:.2f} | "
                      f"{mw} | {gate} |")
        md.append("")
    pairs = m3_pairs(s, runs, TD_PAIR_METRICS, hl=True) if s.get("pairs") else []
    if pairs:
        with open(d / "pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(pairs[0]))
            w.writeheader()
            w.writerows([{c: (repr(v) if isinstance(v, float) else v) for c, v in r.items()} for r in pairs])
        tasks = {g["task"] for g in s["groups"].values()}
        prim = ("CartPole AUC, LunarLander last-100 return" if tasks <= {"CartPole", "LunarLander"} else
                "LunarLander last-100 return, AUC on the other tasks")
        md += ["## Pairs", "", f"Primary = {prim}. AUC = mean return over the first "
               "`episodes` (minimum) episodes. Shift = Hodges–Lehmann first − second [95% CI]; p = two-sided "
               "Mann–Whitney, Holm-adjusted per metric over the pairs of one family.", "",
               "| Group | Against | Family | n | Primary shift [CI] (p, Holm) | AUC shift [CI] (p, Holm) "
               "| Return shift [CI] (p, Holm) |", "| --- | --- | --- | --- | --- | --- | --- |"]
        cell = lambda r, k: "–" if r[f"{k}_p"] is None else (  # noqa: E731
            f"{r[f'{k}_hl']:+.1f} [{r[f'{k}_hl_lo']:+.1f}, {r[f'{k}_hl_hi']:+.1f}] "
            f"({r[f'{k}_p']:.2g}, {r[f'{k}_p_holm']:.2g})")
        for r in pairs:
            md.append(f"| `{label(r['on'])}` | `{label(r['off'])}` | {r['family']} | {r['n_on']}/{r['n_off']} | "
                      + " | ".join(cell(r, k) for k in TD_PAIR_METRICS) + " |")
        md.append("")
    if s.get("labels"):
        md += ["## Display names", "", "Groups shown under a later name; `runs/<key>/` and the manifest keep the key.",
               ""] + [f"- `{v}` = `{k}`" for k, v in s["labels"].items() if k in s["groups"]] + [""]
    if dropped:
        md += ["## Dropped groups", "",
               "Declared and run, then dropped after the run; their files stay in `runs/`.", ""]
        md += [f"- `{k}`: {why}." for k, why in dropped.items()] + [""]
    (d / "summary.md").write_text("\n".join(md))
    return rows


def m3_stats(task, rs):
    """Summary statistics of one group of experiments.m3 runs: AUC = mean return over all episodes (training
    efficiency), last-100 return, the frozen-W return (mean of each run's frozen_returns), the count of runs whose
    trailing-100 mean reaches M3_THR[task] (and the first such episode, T_m), the held-out L_obs and D at the first
    and last evaluation (mean, and sd at the end), ms per frame and resumes. Means and sds over runs (ddof 0)."""
    ms = lambda x: (float(np.mean(x)), float(np.std(x))) if len(x) else (float("nan"),) * 2  # noqa: E731
    auc = [float(np.mean(r["returns"])) for r in rs]
    R = [float(np.mean(r["returns"][-100:])) for r in rs]
    thr = M3_THR.get(task)
    Tm = [t for t in (tdstdp.trailing_mean_time(r["returns"], thr) for r in rs) if t is not None] if thr else []
    ev = lambda i, k: [r["evals"][i][k] for r in rs if k in r["evals"][i]]  # noqa: E731
    F = [float(np.mean(r["frozen_returns"])) for r in rs if r.get("frozen_returns")]
    out = {"n": len(rs), "auc_mean": ms(auc)[0], "auc_sd": ms(auc)[1], "ret_mean": ms(R)[0], "ret_sd": ms(R)[1],
           "frozen_mean": ms(F)[0], "frozen_sd": ms(F)[1],
           "thr": thr, "thr_n": len(Tm), "T_m_mean": ms(Tm)[0], "T_m_sd": ms(Tm)[1]}
    for k in ("L_obs", "D"):
        out |= {f"{k}_first": ms(ev(0, k))[0], f"{k}_last_mean": ms(ev(-1, k))[0], f"{k}_last_sd": ms(ev(-1, k))[1]}
    return out | {"ms_per_frame": float(np.mean([r["ms_per_frame"] for r in rs])),
                  "resumes": sum(r.get("resumes", 0) for r in rs)}


M3_COLUMNS = ["group", "task", "engine", "flags", "n_declared", "n", "auc_mean", "auc_sd", "ret_mean", "ret_sd",
              "frozen_mean", "frozen_sd", "thr",
              "thr_n", "T_m_mean", "T_m_sd", "L_obs_first", "L_obs_last_mean", "L_obs_last_sd", "D_first",
              "D_last_mean", "D_last_sd", "ms_per_frame", "resumes"]
M3_PAIR_METRICS = {"auc": lambda r: float(np.mean(r["returns"])), "ret": lambda r: float(np.mean(r["returns"][-100:])),
                   "frozen": lambda r: float(np.mean(r["frozen_returns"])) if r.get("frozen_returns") else None,
                   "L_obs_last": lambda r: r["evals"][-1].get("L_obs"), "D_last": lambda r: r["evals"][-1].get("D")}


def holm(ps):
    """Holm step-down adjusted p-values (None stays None)."""
    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    out, run = list(ps), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - r) * ps[i]))
        out[i] = run
    return out


def hodges_lehmann(a, b, conf=0.95):
    """Hodges-Lehmann shift a - b (median of all pairwise differences) and its distribution-free CI (Moses; normal
    approximation of the Mann-Whitney quantile)."""
    from scipy.stats import norm
    d = np.sort(np.subtract.outer(a, b).ravel())
    n, m = len(a), len(b)
    k = int(np.floor(n * m / 2 - norm.ppf(0.5 + conf / 2) * np.sqrt(n * m * (n + m + 1) / 12)))
    k = min(max(k, 1), len(d))
    return float(np.median(d)), float(d[k - 1]), float(d[len(d) - k])


def m3_pairs(s, runs, metrics=M3_PAIR_METRICS, hl=False):
    """Per declared pair (on, off[, family]) and metric: means, difference on - off, two-sided Mann-Whitney p, and the
    Holm adjustment of p over the pairs of the same family (per metric; family 'primary' if not given). hl: also the
    Hodges-Lehmann shift on - off with its 95% CI."""
    from scipy.stats import mannwhitneyu
    rows = []
    for on, off, *fam in s.get("pairs", []):
        row = {"on": on, "off": off, "family": fam[0] if fam else "primary", "n_on": len(runs.get(on, [])),
               "n_off": len(runs.get(off, []))}
        for k, f in metrics.items():
            a = [x for x in map(f, runs.get(on, [])) if x is not None]
            b = [x for x in map(f, runs.get(off, [])) if x is not None]
            row |= {f"{k}_on": float(np.mean(a)) if a else None, f"{k}_off": float(np.mean(b)) if b else None,
                    f"{k}_p": float(mannwhitneyu(a, b).pvalue) if a and b else None}
            row[f"{k}_diff"] = row[f"{k}_on"] - row[f"{k}_off"] if a and b else None
            if hl:
                row |= dict(zip((f"{k}_hl", f"{k}_hl_lo", f"{k}_hl_hi"),
                                hodges_lehmann(a, b) if a and b else (None,) * 3))
        rows.append(row)
    for k in metrics:
        for fam in dict.fromkeys(r["family"] for r in rows):
            fr = [r for r in rows if r["family"] == fam]
            for row, p in zip(fr, holm([r[f"{k}_p"] for r in fr])):
                row[f"{k}_p_holm"] = p
    return rows


def summarize_m3(study, s, d):
    """summary.csv, pairs.csv and summary.md of an m3-runner study from runs/ only."""
    runs = {k: load_runs(d, k) for k in s["groups"]}
    rows = []
    for k, g in s["groups"].items():
        row = {"group": k, "task": g["task"], "engine": g["engine"], "flags": " ".join(g["flags"]),
               "n_declared": len(g["seeds"]), "n": len(runs[k])}
        rows.append(row | (m3_stats(g["task"], runs[k]) if runs[k] else {}))
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, M3_COLUMNS)
        w.writeheader()
        w.writerows([{c: (repr(r[c]) if isinstance(r.get(c), float) else r.get(c, "")) for c in M3_COLUMNS}
                     for r in rows])
    pairs = m3_pairs(s, runs)
    if pairs:
        with open(d / "pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(pairs[0]))
            w.writeheader()
            w.writerows([{c: (repr(v) if isinstance(v, float) else v) for c, v in r.items()} for r in pairs])
    mf = d / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    md = [f"# {study}: summary", "",
          f"Generated by `python -m experiments.report summarize {study}` from `runs/` only. Code: git "
          f"`{man.get('git_head', '?')[:7]}`, patch sha256 `{man.get('code_sha256', '?')[:12]}`.", "",
          "Columns: AUC = mean return over all episodes; return = last-100 mean; frozen = mean return of "
          f"{m2.FROZEN_EPIS} end-of-run episodes with W frozen; each mean ± sd over seeds (ddof 0); "
          "thr = runs whose trailing-100 mean reaches the threshold, $T_m$ the first such episode; $L_{obs}$ and $D$ = "
          "held-out description length (bits) of the engine's observed walk and of the flow walk, first eval → last "
          "eval (sd).", ""]
    for task in dict.fromkeys(g["task"] for g in s["groups"].values()):
        md += [f"## {task}", "", "| Group | n | AUC | Return | Frozen | thr | $T_m$ | $L_{obs}$ | $D$ | ms/frame |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in (r for r in rows if r["task"] == task):
            if not r["n"]:
                md.append(f"| `{r['group']}` | 0/{r['n_declared']} | | | | | | | | |")
                continue
            lo, D = (f"{fmt(r[k + '_first'], 3)} → {fmt(r[k + '_last_mean'], 3)} ({fmt(r[k + '_last_sd'], 3)})"
                     for k in ("L_obs", "D"))
            tm = f"{fmt(r['T_m_mean'])} ± {fmt(r['T_m_sd'])}" if r["thr_n"] else "–"
            md.append(f"| `{r['group']}` | {r['n']}/{r['n_declared']} | {fmt(r['auc_mean'])} ± {fmt(r['auc_sd'])} | "
                      f"{fmt(r['ret_mean'])} ± {fmt(r['ret_sd'])} | {fmt(r['frozen_mean'])} ± {fmt(r['frozen_sd'])} | "
                      f"{r['thr_n']}/{r['n']} (≥ {fmt(r['thr'], 0)}) | "
                      f"{tm} | {lo} | {D} | {r['ms_per_frame']:.2f} |")
        md.append("")
    if pairs:
        md += ["## Pairs", "", "Two-sided Mann–Whitney p, Holm-adjusted per metric over the pairs of one family; "
               "diff = first group − second.", "",
               "| Group | Against | Family | n | AUC diff (p, Holm) | Return diff (p, Holm) | Frozen diff (p, Holm) "
               "| $L_{obs}$ diff (p, Holm) | $D$ diff (p, Holm) |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        cell = lambda r, k: "–" if r[f"{k}_p"] is None else \
            f"{r[f'{k}_diff']:+.3g} ({r[f'{k}_p']:.2g}, {r[f'{k}_p_holm']:.2g})"  # noqa: E731
        for r in pairs:
            md.append(f"| `{r['on']}` | `{r['off']}` | {r['family']} | {r['n_on']}/{r['n_off']} | "
                      + " | ".join(cell(r, k) for k in M3_PAIR_METRICS) + " |")
        md.append("")
    (d / "summary.md").write_text("\n".join(md))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize"))
    ap.add_argument("study", choices=tuple(STUDIES))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "run":
        bad = run(a.study, a.workers)
        summarize(a.study)
        sys.exit(1 if bad else 0)
    summarize(a.study)


if __name__ == "__main__":
    main()
