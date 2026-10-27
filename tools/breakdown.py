#!/usr/bin/env python3
"""Full indexed time breakdown from judge profile JSONs (no judge runs needed).

  breakdown.py [--jobs problem=job ...] [--out design/breakdown]      default: best submitted job per problem

Per problem → per shape (weight, sglang / ours / sol µs, run spread) → per kernel (index, GPU µs, share, calls, µs/call)
plus the rows the judge report does not show as kernels:
  hidden  = report total − Σ listed kernels (the "... N more" tail)
  idle    = our median wall − report total (launch gaps / bubbles / graph overhead; negative = stream overlap)
  floor   = sol µs (reference speed-of-light bound); above-floor = ours − sol
Score lens: the board is a weighted geo-mean, so a shape's loss is weight × ln(ours / sol) ("log-µs headroom").
Writes <out>/<problem>.md, <out>/kernels.csv (one row per shape × kernel), <out>/INDEX.md (ranked bottlenecks).
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLOOR_US = 2.5   # practical single-kernel floor (Grok red team I4): judge sol below one launch is unreachable
PROF = ROOT / "experiments" / "profiles"
ROW = re.compile(r"^(.*?)\s{2,}([\d.,]+) µs\s+([\d.]+)%\s+(\d+)\s*$")
MORE = re.compile(r"^\.\.\. (\d+) more")
TOTAL = re.compile(r"^total\s+([\d.,]+) µs")


def best_jobs():
    best = {}
    for r in map(json.loads, (ROOT / "experiments" / "log.jsonl").read_text().splitlines()):
        if r.get("mode") == "submit" and r.get("passed") and r.get("speedup"):
            p = r["problem"]
            if p not in best or r["speedup"] > best[p]["speedup"]:
                best[p] = r
    return {p: r["job"] for p, r in best.items()}


def family(name):
    n = name.lower()
    for key, fam in (("marlin", "moe_gemm(marlin)"), ("smallm_kernel", "moe_gemm(cuda smallm)"),
                     ("largem_kernel", "moe_gemm(cuda largem)"), ("smallm_route", "routing"), ("largem_route", "routing"),
                     ("count_and_sort", "routing"), ("local_ids", "routing"), ("moe_align", "routing"),
                     ("routing", "routing"), ("topk", "routing"),
                     ("nvjet", "dense_gemm(cublas)"), ("cublaslt::splitkreduce", "dense_gemm(cublas)"),
                     ("gemm", "dense_gemm"), ("gemv", "dense_gemm"),
                     ("_full_cat_kernel", "attention"), ("_full_kernel", "attention"), ("attn", "attention"),
                     ("_reduce_kernel", "attention_reduce"), ("layer_norm", "norm/residual"), ("vectorized_layer_", "norm/residual"),
                     ("norm_residual", "norm/residual"), ("post_expert", "moe_epilogue"), ("epilogue", "moe_epilogue"),
                     ("act_and_mul", "activation"), ("silu", "activation"), ("memcpy", "copy/cast"), ("copy", "copy/cast"),
                     ("fillfunc", "fill/memset"), ("fill", "fill/memset"), ("memset", "fill/memset"),
                     ("elementwise", "elementwise")):
        if key in n:
            return fam
    return "other"


def parse_report(rep):
    ks, more, total = [], 0, None
    for ln in rep.splitlines():
        m = ROW.match(ln)
        if m:
            ks.append(dict(name=m.group(1).strip(), us=float(m.group(2).replace(",", "")), share=float(m.group(3)),
                           calls=int(m.group(4))))
        elif MORE.match(ln):
            more = int(MORE.match(ln).group(1))
        elif TOTAL.match(ln):
            total = float(TOTAL.match(ln).group(1).replace(",", ""))
    return ks, more, total


def case_label(c):
    keys = [k for k in c if k not in ("seed", "repeats", "weight")]
    return " ".join(f"{k}={c[k]}" for k in keys)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def analyse(problem, job):
    d = json.loads((PROF / f"{problem}.{job}.json").read_text())
    profs = {p["index"]: p for p in d.get("profiles", [])}
    shapes = []
    for b in d["benchmarks"]:
        i = b["index"]
        ours, sg, sol = b["median_ns"] / 1e3, b["baseline_ns"] / 1e3, max(b["speed_of_light_ns"] / 1e3, FLOOR_US)
        ovh = (b.get("replay_overhead") or {}).get("overhead_ns", 0) / 1e3
        runs = [x / 1e3 - ovh for x in b.get("runs_ns", [])]
        w = float(b["case"].get("weight", 1))
        ks, more, total = parse_report(profs[i]["report"]) if i in profs else ([], 0, None)
        listed = sum(k["us"] for k in ks)
        shapes.append(dict(
            index=i, label=case_label(b["case"]), weight=w, sglang=sg, ours=ours, sol=sol, speedup=b["speedup"],
            p10=pct(runs, .1) if runs else None, p90=pct(runs, .9) if runs else None,
            std=statistics.pstdev(runs) if runs else None,
            replay_overhead=(b.get("replay_overhead") or {}).get("overhead_ns", 0) / 1e3,
            kernels=ks, more=more, total=total, hidden=(total - listed) if total else None,
            idle=(ours - total) if total else None,
            loss=w * math.log(ours / sol), headroom_vs_sglang=w * math.log(sg / sol)))
    return d["summary"], shapes


def write_problem(out, problem, job, summary, shapes, csvw):
    L = [f"# {problem} — judge job {job} (speedup {summary['speedup']:.3f}x, score_us {summary['score_us']:.0f})", "",
         "Units µs (judge median, replay overhead already subtracted; p10–p90 overhead-corrected; sol floored at 2.5 µs). loss = weight × ln(ours/sol): what the board still",
         "pays on this shape vs the floor; Σloss over shapes is the problem's total headroom in log units.", "",
         "| # | shape | w | sglang | ours | sol | speedup | ours/sol | p10–p90 | kernels Σ | hidden | idle | loss |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot_loss = sum(s["loss"] for s in shapes) or 1
    for s in shapes:
        L.append(f"| {s['index']} | {s['label']} | {s['weight']:g} | {s['sglang']:.1f} | {s['ours']:.1f} | {s['sol']:.1f} | "
                 f"{s['speedup']:.3f} | {s['ours']/s['sol']:.2f} | {s['p10']:.1f}–{s['p90']:.1f} | "
                 f"{(s['total'] or 0):.1f} | {(s['hidden'] or 0):.1f} ({s['more']}) | {(s['idle'] or 0):.1f} | "
                 f"{s['loss']:.2f} ({100*s['loss']/tot_loss:.0f}%) |")
    fam_tot = {}
    for s in shapes:
        L += ["", f"## shape {s['index']}: {s['label']}  (w {s['weight']:g}, ours {s['ours']:.1f} µs, sol {s['sol']:.1f})", "",
              "| k# | kernel | family | GPU µs | share of wall | calls | µs/call |", "|---|---|---|---|---|---|---|"]
        for j, k in enumerate(s["kernels"]):
            fam = family(k["name"])
            L.append(f"| {s['index']}.{j} | `{k['name'][:70]}` | {fam} | {k['us']:.1f} | {100*k['us']/s['ours']:.1f}% | "
                     f"{k['calls']} | {k['us']/k['calls']:.1f} |")
            fam_tot[fam] = fam_tot.get(fam, 0) + k["us"] / s["ours"] * s["loss"]
            csvw.writerow([problem, job, s["index"], s["label"], s["weight"], j, k["name"], fam, k["us"], k["calls"],
                           round(k["us"] / s["ours"], 4), round(s["ours"], 2), round(s["sol"], 2)])
        for nm, v, note in (("(hidden tail)", s["hidden"], f"{s['more']} kernels not listed by the judge report"),
                            ("(idle / gaps)", s["idle"], "wall − Σkernels: launch gaps, bubbles, graph overhead")):
            if v is not None:
                L.append(f"| {s['index']}.– | {nm} | {note} | {v:.1f} | {100*v/s['ours']:.1f}% | | |")
                csvw.writerow([problem, job, s["index"], s["label"], s["weight"], "-", nm, nm, round(v, 2), "",
                               round(v / s["ours"], 4), round(s["ours"], 2), round(s["sol"], 2)])
    L += ["", "## families, weighted by each shape's loss share (where the board's headroom sits)", "",
          "| family | weighted loss share |", "|---|---|"]
    tf = sum(fam_tot.values()) or 1
    for fam, v in sorted(fam_tot.items(), key=lambda x: -x[1]):
        L.append(f"| {fam} | {100*v/tf:.1f}% |")
    (out / f"{problem}.md").write_text("\n".join(L) + "\n")
    return fam_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", default=[])
    ap.add_argument("--out", default="design/breakdown")
    a = ap.parse_args()
    jobs = best_jobs()
    jobs.update(dict(j.split("=", 1) for j in a.jobs))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    idx = ["# Bottleneck index (from judge profiles; regenerate: tools/breakdown.py)", "",
           "| problem | job | speedup | worst shapes by loss | top families by weighted loss |", "|---|---|---|---|---|"]
    with (out / "kernels.csv").open("w", newline="") as fh:
        csvw = csv.writer(fh)
        csvw.writerow(["problem", "job", "shape", "label", "weight", "k", "kernel", "family", "gpu_us", "calls",
                       "share_of_wall", "ours_us", "sol_us"])
        for problem, job in sorted(jobs.items()):
            if not (PROF / f"{problem}.{job}.json").exists():
                idx.append(f"| {problem} | {job} | – | profile JSON missing | |")
                continue
            summary, shapes = analyse(problem, job)
            fam = write_problem(out, problem, job, summary, shapes, csvw)
            worst = sorted(shapes, key=lambda s: -s["loss"])[:3]
            tf = sum(fam.values()) or 1
            idx.append(f"| [{problem}]({problem}.md) | {job} | {summary['speedup']:.3f} | " +
                       "; ".join(f"#{s['index']} {s['ours']/s['sol']:.1f}× sol" for s in worst) + " | " +
                       ", ".join(f"{f} {100*v/tf:.0f}%" for f, v in sorted(fam.items(), key=lambda x: -x[1])[:3]) + " |")
    (out / "INDEX.md").write_text("\n".join(idx) + "\n")
    print((out / "INDEX.md").read_text())


if __name__ == "__main__":
    main()
