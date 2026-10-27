#!/usr/bin/env python3
"""The only way this project calls the judge. Director-only (it loads the okc token).

  okc_run.py check  <problem> <file> --hyp "..." --predict 1.25 [--range 1.15,1.35] [--card C4]
  okc_run.py submit <problem> <file> --hyp "..." --predict 1.25 --redteam <ack-ref> [--card C4]
  okc_run.py log [n]                      last n records, one line each

Every run appends one record to experiments/log.jsonl and copies the profile to experiments/profiles/.
A run without a hypothesis and a prediction is refused; a submit without a red-team ack is refused.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments" / "log.jsonl"
PROFILES = ROOT / "experiments" / "profiles"
OKC = Path.home() / ".okc" / "bin" / "okc"
ENV = Path.home() / ".config" / "fbt" / "okc.env"
PROBLEMS = {"muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"}


def _env():
    env = dict(os.environ)
    for line in ENV.read_text().splitlines():
        if line.startswith("OKC_TOKEN="):
            env["OKC_TOKEN"] = line.split("=", 1)[1].strip()
    if "OKC_TOKEN" not in env:
        sys.exit("okc_run: no OKC_TOKEN (director only)")
    return env


def _git_head():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    return r.stdout.strip() + ("+dirty" if dirty.strip() else "")


def _verdict(res, predict, lo, hi):
    if not res.get("passed"):
        return "failed-test" if res.get("status") == "ok" else res.get("status") or "error"
    s, ci = res.get("speedup"), res.get("speedup_ci") or [None, None]
    if s is None:
        return "error"
    if ci[0] is not None and ci[0] <= 1.0 <= ci[1] and not (lo <= 1.0 <= hi):
        return "noise"
    return "confirmed" if lo <= s <= hi else ("better" if s > hi else "refuted")


def run(a):
    if a.problem not in PROBLEMS:
        sys.exit(f"okc_run: unknown problem {a.problem}")
    src = Path(a.file).resolve()
    first = src.read_text().splitlines()[0] if src.exists() else ""
    if first.strip() != f"# okc: problem={a.problem}" and "baseline" not in src.name:
        sys.exit(f"okc_run: first line of {src} must be '# okc: problem={a.problem}'")
    try:  # never spend a judge run on a file that does not even compile
        compile(src.read_text(), str(src), "exec")
    except SyntaxError as e:
        sys.exit(f"okc_run: {src} does not compile: {e}")
    if not a.hyp or a.predict is None:
        sys.exit("okc_run: --hyp and --predict are required")
    if a.mode == "submit" and ("MEASUREMENT-ONLY" in src.read_text() or "def _measure" in src.read_text() or src.name.startswith(("trace_", "measure_")) or "torch.profiler" in src.read_text()):
        sys.exit("okc_run: refusing to submit a measurement-only file (fable-06 D1)")
    if a.mode == "submit" and not a.redteam:
        sys.exit("okc_run: submit needs --redteam <ack ref> (Grok Bot review)")
    lo, hi = (float(x) for x in a.range.split(",")) if a.range else (a.predict * 0.93, a.predict * 1.07)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    started = time.time()
    cmd = [str(OKC), a.mode, a.problem, str(src), "--json"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=_env(), stdin=subprocess.DEVNULL, cwd=src.parent)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        out = {"result": {"status": "okc-error", "error": (p.stdout + p.stderr)[-2000:]}}
    res = out.get("result") or {}
    job = out.get("id")
    PROFILES.mkdir(parents=True, exist_ok=True)
    prof = None
    if job:
        for f in (src.parent / "profiles").glob(f"*.{job}.*"):
            dst = PROFILES / f"{a.problem}.{job}{f.suffix}"
            shutil.copy(f, dst)
            if f.suffix == ".txt":
                prof = dst
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "wall_s": round(time.time() - started),
        "track": "kernel", "mode": a.mode, "problem": a.problem, "card": a.card,
        "file": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src), "sha256": sha[:16],
        "git": _git_head(), "hypothesis": a.hyp, "predicted": a.predict, "range": [round(lo, 3), round(hi, 3)],
        "redteam": a.redteam, "job": job, "status": res.get("status"), "passed": res.get("passed"),
        "speedup": res.get("speedup"), "ci": res.get("speedup_ci"), "score_us": res.get("score_us"),
        "error": (res.get("error") or None) and str(res.get("error"))[:500],
        "flags": res.get("flags"), "profile": prof and str(prof.relative_to(ROOT)),
        "limits": out.get("limits"),
    }
    rec["verdict"] = _verdict(res, a.predict, lo, hi)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(_line(rec))


def _line(r):
    s = f"{r['speedup']:.3f}x" if r.get("speedup") else "-"
    ci = r.get("ci") or []
    ci = f"[{ci[0]:.3f},{ci[1]:.3f}]" if len(ci) == 2 else ""
    return f"{r['ts']} {r['mode']:6} {r['problem']:22} {s:>7} {ci:15} pred {r['predicted']} {r['verdict']:11} job {r['job']} {r['file']}"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for m in ("check", "submit"):
        s = sub.add_parser(m)
        s.add_argument("problem")
        s.add_argument("file")
        s.add_argument("--hyp", required=True)
        s.add_argument("--predict", type=float, required=True)
        s.add_argument("--range")
        s.add_argument("--card")
        s.add_argument("--redteam")
    lg = sub.add_parser("log")
    lg.add_argument("n", nargs="?", type=int, default=20)
    a = ap.parse_args()
    if a.mode == "log":
        lines = LOG.read_text().splitlines() if LOG.exists() else []
        for ln in lines[-a.n:]:
            print(_line(json.loads(ln)))
        return
    run(a)


if __name__ == "__main__":
    main()
