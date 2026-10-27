#!/usr/bin/env python3
"""Promote a candidate: judge check first, submit ONLY if it beats our best board entry for that problem.

  promote.py <problem> <file> --predict X [--range lo,hi] --hyp "..." [--card C] [--margin 0.0]

Law (enforced here, not by doctrine):
- refuses measurement-only files (okc_run also refuses them on submit);
- a red-team file must exist for the candidate's sha (design/redteam/<problem>-<name>-<sha16>.md) — under Dex's
  standing override it may be a one-line override note; promote.py writes that note itself only if the env var
  OKC_DEX_OVERRIDE=1 is set by the director;
- submits only if: check passed AND check speedup > best submitted speedup for the problem (+ margin) AND the check's
  lower CI bound > best submitted speedup's lower CI bound (no promotion inside the noise).
Every step goes through tools/okc_run.py (hypothesis + prediction + log), so experiments/log.jsonl stays complete.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments" / "log.jsonl"
PY = str(ROOT / ".venv" / "bin" / "python")


def records(problem):
    if not LOG.exists():
        return []
    out = []
    for ln in LOG.read_text().splitlines():
        r = json.loads(ln)
        if r.get("problem") == problem:
            out.append(r)
    return out


def best_submit(problem):
    subs = [r for r in records(problem) if r.get("mode") == "submit" and r.get("passed") and r.get("speedup")]
    return max(subs, key=lambda r: r["speedup"]) if subs else None


def run(mode, a, redteam=None):
    cmd = [PY, str(ROOT / "tools" / "okc_run.py"), mode, a.problem, a.file, "--hyp", a.hyp, "--predict", str(a.predict)]
    if a.range:
        cmd += ["--range", a.range]
    if a.card:
        cmd += ["--card", a.card]
    if redteam:
        cmd += ["--redteam", redteam]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip()[-500:])
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("file")
    ap.add_argument("--predict", type=float, required=True)
    ap.add_argument("--range")
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--card")
    ap.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()
    src = (ROOT / a.file) if not os.path.isabs(a.file) else Path(a.file)
    text = src.read_text()
    if "MEASUREMENT-ONLY" in text or "def _measure" in text or src.name.startswith(("trace_", "measure_")) or "torch.profiler" in text:
        sys.exit("promote: measurement-only file — check it with okc_run.py, never promote it")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    rt = ROOT / "design" / "redteam" / f"{a.problem}-{src.stem}-{sha}.md"
    if not rt.exists():
        if os.environ.get("OKC_DEX_OVERRIDE") == "1":
            rt.write_text(f"# Red team: {a.file} (sha256 {sha}…)\nMinister override (Dex 2026-10-24, standing for "
                          f"non-attention lanes): promote without review if the check beats the board entry.\n")
        else:
            sys.exit(f"promote: no red-team file {rt.relative_to(ROOT)} (get >= 2 ACK or set OKC_DEX_OVERRIDE=1)")
    if run("check", a) != 0:
        sys.exit("promote: check failed to run")
    chk = [r for r in records(a.problem) if r.get("mode") == "check" and r.get("sha256") == sha[:16]]
    chk = chk[-1] if chk else None
    best = best_submit(a.problem)
    if not chk or not chk.get("passed"):
        sys.exit("promote: NOT SUBMITTED — check failed tests")
    b_s = best["speedup"] if best else 1.0
    b_lo = (best.get("ci") or [b_s, b_s])[0] if best else 1.0
    c_lo = (chk.get("ci") or [chk["speedup"], chk["speedup"]])[0]
    if chk["speedup"] > b_s + a.margin and c_lo > b_lo:
        run("submit", a, redteam=str(rt.relative_to(ROOT)))
        print(f"promote: SUBMITTED ({chk['speedup']:.3f}x > board {b_s:.3f}x)")
    else:
        print(f"promote: NOT SUBMITTED ({chk['speedup']:.3f}x [lo {c_lo:.3f}] vs board {b_s:.3f}x [lo {b_lo:.3f}])")


if __name__ == "__main__":
    main()
