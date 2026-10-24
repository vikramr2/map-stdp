"""Final reported studies: reports/<study>/ holds only what this runner writes (experiments/results/ is the
intermediates store: dev, tuning and diagnostic runs, and the default output of every CLI). Run from the repo root:
  python -m experiments.report run <study> [--workers N]
    -> reports/<study>/runs/<group>/s<seed>.json (+ manifest.json, code.patch if the tree is dirty); skips finished
       runs, resumes checkpointed ones (*.ckpt.npz, gitignored); refuses to launch if mapstdp/ or experiments/ differ
       from the code state recorded in the manifest; summarizes at the end
  python -m experiments.report summarize <study>   -> reports/<study>/summary.csv, summary.md, from runs/ only
Studies are declared in STUDIES: groups -> (task, engine, flags, seeds), a reference group per task (Mann-Whitney) and
gates (minimum solved count). Statistics are experiments.tdstdp.group_stats, the same as tdstdp --summary."""
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

from experiments import tdstdp

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs" / "report"
ENGINE_FLAG = {"lif": ["--engine", "lif"], "branching": []}  # branching = the numpy engine (tdstdp default)

_CP, _LL = range(1000, 1020), range(1000, 1010)
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
        gates={"CartPole_noadam_lif": 18, "LunarLander_adam_lif": 6},
        # Declared and run, then dropped (2026-10-24): kept in runs/, left out of summary.csv and the tables.
        dropped={k: "the authors' Adam step is the per-ms SGD eta, 40-160x LunarLander's for CartPole, and collapses "
                    "the policy from the first episodes; CK report no CartPole Adam config and CartPole needs no Adam"
                 for k in ("CartPole_adam_lif", "CartPole_adam_branching")},
    ),
}
COLUMNS = ["group", "task", "engine", "flags", "n_declared", "n", "solved", "T_f_mean", "T_f_sd", "T_f_n", "T_s_mean",
           "T_s_sd", "T_s_median", "T_m_mean", "T_m_sd", "T_m_n", "ret_mean", "ret_sd", "ret_ge_M", "ms_per_frame",
           "resumes", "mw_ref", "mw_T_f", "mw_T_s", "mw_ret", "gate_min_solved", "gate_pass"]


def command(g, seed, out):
    return [sys.executable, "-m", "experiments.tdstdp", "--task", g["task"], "--seed", str(seed),
            *ENGINE_FLAG[g["engine"]], *g["flags"], "--out", str(out)]


def run_path(study, group, seed, root=REPORTS):
    return root / study / "runs" / group / f"s{seed}.json"


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
                  "commands": [" ".join(["python", *command(g, sd, run_path(study, k, sd).relative_to(ROOT))[1:]])
                               for sd in g["seeds"]]} for k, g in s["groups"].items()}
    return {"study": study, "title": s["title"], "created": datetime.datetime.now().astimezone().isoformat(),
            "git_head": head, "dirty": bool(patch), "code_sha256": sha, "code_patch": "code.patch" if patch else None,
            "versions": versions(), "env": {"OMP_NUM_THREADS": "1"}, "workers": workers,
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
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

    def one(k, sd):
        g, out = STUDIES[study]["groups"][k], run_path(study, k, sd)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(LOGS / study / f"{k}_s{sd}.log", "a") as log:
            rc = subprocess.run(command(g, sd, out.relative_to(ROOT)), cwd=ROOT, env=env, stdout=log,
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


def summarize(study, root=REPORTS, studies=None):
    """summary.csv and summary.md of reports/<study>/ from runs/ only (finished runs, s<seed>.json)."""
    s, d = (studies or STUDIES)[study], root / study
    runs = {k: [json.loads(p.read_text()) for p in sorted((d / "runs" / k).glob("s*.json"))] for k in s["groups"]}
    rows = []
    dropped = s.get("dropped", {})
    for k, g in s["groups"].items():
        if k in dropped:
            continue
        ref = s["reference"].get(g["task"])
        row = {"group": k, "task": g["task"], "engine": g["engine"], "flags": " ".join(g["flags"]),
               "n_declared": len(g["seeds"]), "n": len(runs[k])}
        if runs[k]:
            st = tdstdp.group_stats(g["task"], runs[k], runs[ref] if k != ref and runs.get(ref) else None)
            row |= st | {"resumes": sum(r.get("resumes", 0) for r in runs[k])}
            row["mw_ref"] = ref if "mw_T_f" in st else ""
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
        p = tdstdp.PAPER[task]
        md += [f"## {task}", "", f"CK 2020 (10 runs, all solved): $T_f$ {p['T_f'][0]:.1f} ± {p['T_f'][1]:.1f}, "
               f"$T_s$ {p['T_s'][0]:.1f} ± {p['T_s'][1]:.1f}. Reference group: `{s['reference'].get(task, '-')}`. "
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
    if dropped:
        md += ["## Dropped groups", "",
               "Declared and run, then dropped after the run; their files stay in `runs/`.", ""]
        md += [f"- `{k}`: {why}." for k, why in dropped.items()] + [""]
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
