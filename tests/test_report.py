"""experiments.report summarize on a tiny fake study: summary.csv from runs/ only."""
import csv
import json

from scipy.stats import mannwhitneyu

from experiments import report

STUDY = {"fake": dict(title="fake", reference={"CartPole": "ref"}, gates={"ref": 2},
                      groups={"ref": dict(task="CartPole", engine="lif", flags=["--block", "20"], seeds=[0, 1]),
                              "alt": dict(task="CartPole", engine="branching", flags=[], seeds=[0, 1, 2])})}


def fake(Tf, Ts, ret, ms):
    return {"config": {"episodes": 400, "max_eps": 400}, "returns": ret, "T_f": Tf, "T_s": Ts, "ms_per_frame": ms}


def test_summarize_fake_study(tmp_path):
    runs = {"ref": [fake(1, 100, [500.0] * 150, 1.0), fake(3, 102, [10.0, 10.0] + [500.0] * 148, 3.0)],
            "alt": [fake(None, None, [10.0] * 150, 2.0), fake(None, None, [20.0] * 150, 2.0)]}
    for g, rs in runs.items():
        (tmp_path / "fake" / "runs" / g).mkdir(parents=True)
        for i, r in enumerate(rs):
            (tmp_path / "fake" / "runs" / g / f"s{i}.json").write_text(json.dumps(r))
    (tmp_path / "fake" / "runs" / "alt" / "s2.partial").write_text("{}")  # unfinished: ignored
    report.summarize("fake", root=tmp_path, studies=STUDY)
    rows = {r["group"]: r for r in csv.DictReader(open(tmp_path / "fake" / "summary.csv"))}
    ref, alt = rows["ref"], rows["alt"]
    assert (ref["n"], ref["solved"], ref["T_f_n"], ref["T_m_n"], ref["gate_pass"]) == ("2", "2", "2", "2", "True")
    assert float(ref["T_f_mean"]) == 2.0 and float(ref["T_f_sd"]) == 1.0 and float(ref["T_s_median"]) == 101.0
    assert float(ref["T_m_mean"]) == 100.0 and float(ref["ret_mean"]) == 500.0 and float(ref["ms_per_frame"]) == 2.0
    assert ref["mw_ref"] == "" and ref["mw_T_f"] == ""
    assert (alt["n_declared"], alt["n"], alt["solved"], alt["T_f_n"], alt["mw_ref"]) == ("3", "2", "0", "0", "ref")
    assert float(alt["T_s_median"]) == float("inf") and float(alt["ret_mean"]) == 15.0 and float(alt["ret_sd"]) == 5.0
    assert float(alt["mw_T_s"]) == mannwhitneyu([401, 401], [100, 102]).pvalue  # never reached = cap + 1
    assert float(alt["mw_ret"]) == mannwhitneyu([10.0, 20.0], [500.0, 500.0]).pvalue
    assert alt["gate_pass"] == "" and "| `ref` | 2/2 | 2/2 |" in (tmp_path / "fake" / "summary.md").read_text()


def test_summarize_cross_study_reference(tmp_path):
    studies = STUDY | {"comp": dict(title="comp", reference={"CartPole": "fake/ref"}, gates={},
                                    groups={"new": dict(task="CartPole", engine="flow", flags=[], seeds=[0])})}
    for st, g, r in (("fake", "ref", fake(1, 100, [500.0] * 150, 1.0)),
                     ("comp", "new", fake(None, None, [10.0] * 150, 1.0))):
        (tmp_path / st / "runs" / g).mkdir(parents=True)
        (tmp_path / st / "runs" / g / "s0.json").write_text(json.dumps(r))
    report.summarize("comp", root=tmp_path, studies=studies)
    row = next(csv.DictReader(open(tmp_path / "comp" / "summary.csv")))
    assert row["group"] == "new" and row["mw_ref"] == "fake/ref"
    assert float(row["mw_ret"]) == mannwhitneyu([10.0], [500.0]).pvalue
