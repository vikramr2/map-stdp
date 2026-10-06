"""Behaviour lock for refactors (used by the codebase-cleaner agent): fixed-seed runs must reproduce golden.json
bit for bit. Regenerate the golden file only for an intended behaviour change, and say so in the CHANGELOG:
  python -m tests.test_regression --update"""
import json
import pathlib
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

from experiments import m2
from mapstdp import spiking as sp

GOLDEN = pathlib.Path(__file__).parent / "golden.json"
CASES = [("flow", "exempt", 0.05, "v3"), ("flow", "dual", 0.05, "v1"), ("flow", "none", 0.05, "v2"),
         ("spiking", "exempt", 0.05, "v3"), ("spiking", "exempt", 0.0, "v1")]


def fingerprint(backend, routing, lam, model, episodes=3):
    agent, env = m2.Agent(backend, routing, lam, 0, model), gym.make("CartPole-v1")
    returns = [agent.episode(env) for _ in range(episodes)]
    ev = agent.evaluate(kl_obs=3, kl_reps=5)
    return {"returns": returns, "W_sum": float(agent.W.sum()), "W_sq": float((agent.W ** 2).sum()),
            "u_sum": float(agent.u.sum()), "mu": agent.mu, "q_star": agent.q_star, "silent": agent.silent, "eval": ev}


def case_id(case):
    return "-".join(map(str, case))


@pytest.mark.parametrize("module", ["mapstdp.core", "mapstdp.spiking"])
def test_self_checks(module):
    assert subprocess.run([sys.executable, "-m", module], capture_output=True).returncode == 0


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_m2_reproduces_golden(case):
    assert fingerprint(*case) == json.loads(GOLDEN.read_text())[case_id(case)]


def test_summary_reads_all_results(capsys):
    m2.summary()  # every m2_*.json in experiments/results must parse as an m2 run
    assert "random policy" in capsys.readouterr().out


def readout_fingerprint():
    S = np.random.default_rng(0).random((50, 12)) < 0.2
    C, B, V = sp.pairings(S)
    races = [sp.race(S, [[0, 1, 2], [3, 4, 5]], np.random.default_rng(s), burn=10) for s in range(5)]
    return {"C": C.tolist(), "B": B.tolist(), "V": V.round(12).tolist(), "races": races}


def test_spiking_readout_and_pairings():
    assert readout_fingerprint() == json.loads(GOLDEN.read_text())["spiking-readout"]


if __name__ == "__main__" and "--update" in sys.argv:
    golden = {case_id(c): fingerprint(*c) for c in CASES} | {"spiking-readout": readout_fingerprint()}
    GOLDEN.write_text(json.dumps(golden, indent=1))
    print(f"wrote {GOLDEN}")
