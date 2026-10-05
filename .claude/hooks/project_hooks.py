#!/usr/bin/env python3
"""Project hooks for map-stdp. Usage: project_hooks.py {math|changelog|installs|env}.

Reads the hook event JSON on stdin. Exit 2 with a message on stderr blocks the
tool call (PreToolUse) or feeds the problem back to Claude (PostToolUse).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))
ENV = Path.home() / ".conda" / "envs" / "map-stdp"

# Optional prefixes before the real command word: env assignments, sudo, a path.
HEAD = r"^(?:\w+=\S*\s+)*(?:sudo\s+)?(?:\S*/)?"


def segments(cmd):
    """Split a shell command into simple commands. Quotes are not parsed, but every
    pattern is anchored at the start of a segment, so text inside strings rarely matches."""
    return [seg.strip() for seg in re.split(r"&&|\|\||;|\||\n", cmd)]


def block(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


def math(event):
    """H1: after editing docs markdown or CLAUDE.md, check every math span."""
    f = Path(event.get("tool_input", {}).get("file_path", ""))
    try:
        rel = f.resolve().relative_to(ROOT)
    except ValueError:
        return
    if not (rel.suffix == ".md" and (rel.parts[0] == "docs" or rel == Path("CLAUDE.md"))):
        return
    if rel.parts[:2] == ("docs", "superneuro"):
        return
    node = ENV / "bin" / "node"
    node = str(node) if node.exists() else shutil.which("node")
    if not node:
        print("math check skipped: node not found (activate the map-stdp env)", file=sys.stderr)
        return
    r = subprocess.run([node, str(ROOT / ".claude/hooks/math_check.js"), str(f)],
                       capture_output=True, text=True)
    if r.returncode:
        block(f"Math rendering check failed for {rel}:\n{r.stderr.strip()}")


def changelog(event):
    """H2: block commits with staged changes but no docs/CHANGELOG.md entry."""
    cmd = event.get("tool_input", {}).get("command", "")
    commits = [s for s in segments(cmd) if re.match(HEAD + r"git(\s+-C\s+\S+)?\s+commit\b", s)]
    if not commits or "skip changelog" in cmd:
        return
    git = ["git", "-C", str(ROOT)]
    staged = subprocess.run(git + ["diff", "--cached", "--name-only"],
                            capture_output=True, text=True).stdout.split()
    if any(re.search(r"\s(-a|--all|-[b-zA-Z]*a[b-zA-Z]*)(\s|$)", c) for c in commits):
        staged += subprocess.run(git + ["diff", "--name-only"],
                                 capture_output=True, text=True).stdout.split()
    if staged and "docs/CHANGELOG.md" not in staged:
        block("Commit blocked: changes are staged but docs/CHANGELOG.md is not. "
              "CLAUDE.md requires a dated changelog entry for every change "
              "(add one, or put 'skip changelog' in the commit message for trivial commits).")


def installs(event):
    """H3: block package installs that could land outside the project env."""
    cmd = event.get("tool_input", {}).get("command", "")
    env_active = os.environ.get("CONDA_DEFAULT_ENV") == "map-stdp"
    for seg in segments(cmd):
        # Accept the env path in any spelling (absolute, ~, $HOME); variables are not expanded here.
        targets_env = "envs/map-stdp/" in seg or "-n map-stdp" in seg or "--name map-stdp" in seg
        if re.match(HEAD + r"(pip3?|python3?(\.\d+)?\s+-m\s+pip)\s+install\b", seg):
            venv = re.search(r"(\.venv|venv)/bin/(pip|python)", seg)
            if "--user" in seg:
                block("Blocked: `pip install --user` installs outside the project env.")
            if not (targets_env or venv or "--target" in seg or re.search(r"\s-t\s", seg) or env_active):
                block("Blocked: pip install outside the map-stdp env. CLAUDE.md: never install globally "
                      "without asking. Use the env's pip, a venv, or `--target <scratchpad>`.")
        if re.match(HEAD + r"npm\s+(install|i|add)\b.*(\s-g\b|--global)", seg):
            if not (targets_env or env_active):
                block("Blocked: global npm install outside the map-stdp env.")
        if re.match(HEAD + r"(conda|mamba)\s+install\b", seg):
            if not (targets_env or (env_active and " -n " not in seg and "--name" not in seg)):
                block("Blocked: conda install not targeting the map-stdp env (use `-n map-stdp`).")


def env(event):
    """H4: at session start, warn if the project env (and so node) is missing."""
    problems = []
    if os.environ.get("CONDA_DEFAULT_ENV") != "map-stdp":
        problems.append("the map-stdp conda env is not active")
    if not shutil.which("node"):
        problems.append("`node` is not on PATH, so the ponytail plugin's hooks will fail")
    if problems:
        msg = ("Environment warning: " + "; ".join(problems) +
               ". Tell the user to quit and relaunch with `conda activate map-stdp && claude --continue`.")
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart",
                                                 "additionalContext": msg}}))


if __name__ == "__main__":
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        event = {}
    {"math": math, "changelog": changelog, "installs": installs, "env": env}[sys.argv[1]](event)
