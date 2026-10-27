#!/usr/bin/env python3
"""Charts from the level-1 traces + judge JSONs (Dex 2026-10-25: "see the shapes, kernel speeds and transitions").

  visuals.py [--out design/breakdown/visuals]      uses the latest trace check per problem in experiments/log.jsonl

Writes per problem: <p>-timeline.png (every shape: kernels in launch order, colour = family, gaps visible, lane per stream),
<p>-speed.png (SGLang vs ours vs floor per shape, speedup labels), <p>-mix.png (where the time goes per shape, % of span);
plus overview.png (speedup and rank vs leader per problem) and VISUALS.md that embeds them (GitHub renders it).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from breakdown import family  # noqa: E402

PROBLEMS = ["muse_moe_w4a16", "muse_decode_layer", "muse_attention_decode", "muse_bf16_proj"]
COLORS = {
    "moe_gemm(cuda largem)": "#d62728", "moe_gemm(cuda smallm)": "#ff7f0e", "moe_gemm(marlin)": "#8c564b",
    "dense_gemm(cublas)": "#1f77b4", "dense_gemm": "#17becf", "attention": "#2ca02c", "attention_reduce": "#98df8a",
    "norm/residual": "#9467bd", "routing": "#e377c2", "moe_epilogue": "#bcbd22", "activation": "#f7b6d2",
    "fill/memset": "#c7c7c7", "copy/cast": "#7f7f7f", "elementwise": "#aec7e8", "other": "#393b79",
}
SHORT = [("w4_largem_kernel<true", "experts GEMM1 (ours)"), ("w4_largem_kernel<false", "experts GEMM2 (ours)"),
         ("w4_smallm_kernel<true", "experts GEMM1 (ours)"), ("w4_smallm_kernel<false", "experts GEMM2 (ours)"),
         ("marlin", "Marlin experts"), ("_full_cat_kernel", "attention full"), ("_full_kernel", "attention verify-full"),
         ("_attn_kernel", "attention"), ("_reduce_kernel", "attn combine"), ("splitkreduce", "cuBLAS reduce"),
         ("nvjet", "cuBLAS GEMM"), ("_gemv_splitk", "GEMV (ours)"), ("norm_residual", "norm+residual"),
         ("layer_norm", "layer norm"), ("routing_kernel", "routing"), ("route_kernel", "route"), ("epilogue", "epilogue"),
         ("post_expert", "epilogue"), ("act_and_mul", "SiLU*mul"), ("moe_align", "align"), ("count_and_sort", "sort"),
         ("local_ids", "local ids"), ("fillfunc", "fill"), ("memcpy", "copy")]


def short(name):
    n = name.lower()
    for k, v in SHORT:
        if k.lower() in n:
            return v
    return name[:18]


def latest_trace_jobs():
    jobs = {}
    for r in map(json.loads, (ROOT / "experiments" / "log.jsonl").read_text().splitlines()):
        if r.get("mode") == "check" and "/trace_" in str(r.get("file", "")) and r.get("job"):
            jobs[r["problem"]] = r["job"]
    return jobs


def load(problem, job):
    names, shapes = {}, {}
    for ln in (ROOT / "experiments" / "profiles" / f"{problem}.{job}.log").read_text().splitlines():
        t = ln.split()
        if not t:
            continue
        if t[0] == "TN":
            names[t[1]] = " ".join(t[2:])
        elif t[0] == "TS":
            kv = dict(x.split("=", 1) for x in t[2:] if "=" in x)
            shapes[int(t[1])] = dict(w=float(kv["w"]), wall=float(kv["wall"]), span=float(kv["span"]), label=kv.get("label", ""), k=[])
        elif t[0] == "TK":
            si = int(t[1].split(".")[0])
            kv = dict(x.split("=", 1) for x in t[3:] if "=" in x)
            nm = names.get(t[2], "?")
            shapes[si]["k"].append(dict(name=nm, fam=family(nm), s=float(kv["s"]), d=float(kv["d"]), g=float(kv.get("g", 0)),
                                        st=int(float(kv.get("st", 0)))))
    js = json.loads((ROOT / "experiments" / "profiles" / f"{problem}.{job}.json").read_text())
    for b in js["benchmarks"]:
        if b["index"] in shapes:
            shapes[b["index"]].update(sglang=b["baseline_ns"] / 1e3, ours=b["median_ns"] / 1e3, sol=b["speed_of_light_ns"] / 1e3,
                                      speedup=b["speedup"], case=b["case"])
    return shapes, js["summary"]


def label(sh):
    c = sh.get("case", {})
    skip = {"seed", "repeats", "weight", "page", "ctx", "dist", "heads", "sharding"}
    return " ".join(f"{v}" if k in ("layer", "load") else f"{k}={v}" for k, v in c.items() if k not in skip)


def timeline(problem, shapes, out):
    idx = sorted(shapes)
    fig, axes = plt.subplots(len(idx), 1, figsize=(15, 0.95 * len(idx) + 1.2), squeeze=False)
    used = set()
    for ax, i in zip(axes[:, 0], idx):
        sh = shapes[i]
        lanes = sorted({k["st"] for k in sh["k"]})
        lane_end = {}
        for k in sh["k"]:
            y = lanes.index(k["st"])
            hgt = 0.8
            if k["s"] < lane_end.get(k["st"], -1) - 0.05:      # starts before the previous kernel on this stream ends (PDL early launch)
                y, hgt = y - 0.2, 0.4
            lane_end[k["st"]] = max(lane_end.get(k["st"], 0), k["s"] + k["d"])
            col = COLORS.get(k["fam"], "#393b79")
            used.add(k["fam"])
            ax.barh(y, k["d"], left=k["s"], height=hgt, color=col, edgecolor="black" if hgt < 0.8 else "white", linewidth=0.6)
            if k["d"] > 0.07 * sh["span"]:
                ax.text(k["s"] + k["d"] / 2, y, f"{short(k['name'])}\n{k['d']:.1f} µs" + (" (early start)" if hgt < 0.8 else ""),
                        ha="center", va="center", fontsize=6.5,
                        color="white" if k["fam"] not in ("fill/memset", "attention_reduce", "moe_epilogue", "elementwise") else "black")
        ax.set_xlim(0, sh["span"] * 1.005)
        ax.set_ylim(-0.6, len(lanes) - 0.4)
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels([f"stream {s}" for s in lanes], fontsize=6)
        ax.tick_params(axis="x", labelsize=6)
        gaps = sum(k["g"] for k in sh["k"])
        ax.set_title(f"#{i}  {label(sh)}   |   {sh['span']:.1f} µs scored, {len(sh['k'])} kernels, gaps {gaps:.1f} µs   |   "
                     f"SGLang {sh.get('sglang', 0):.1f} µs → {sh.get('speedup', 0):.2f}×", fontsize=8, loc="left")
    axes[-1, 0].set_xlabel("µs from the first kernel (each row has its own scale)", fontsize=8)
    fig.legend(handles=[Patch(color=COLORS.get(f, "#393b79"), label=f) for f in sorted(used)], loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=8, fontsize=7,
               frameon=False)
    fig.suptitle(f"{problem}: kernel timeline per shape (judge H200, CUDA-graph replay, cold L2)", fontsize=11, y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def speed(problem, shapes, summary, out):
    idx = sorted(shapes)
    labels = [f"#{i} {label(shapes[i])}" for i in idx]
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(idx) + 1.5))
    y = range(len(idx))
    h = 0.27
    ax.barh([v + h for v in y], [shapes[i]["sglang"] for i in idx], h, color="#bbbbbb", label="SGLang")
    ax.barh(list(y), [shapes[i]["ours"] for i in idx], h, color="#d62728", label="ours")
    ax.barh([v - h for v in y], [shapes[i]["sol"] for i in idx], h, color="#2ca02c", label="floor (speed of light)")
    for v, i in zip(y, idx):
        ax.text(shapes[i]["sglang"] * 1.02, v + h, f"{shapes[i]['speedup']:.2f}×", va="center", fontsize=7, weight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("µs per call (log scale) — shorter is better", fontsize=8)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_title(f"{problem}: SGLang vs ours vs floor per shape (this run {summary['speedup']:.3f}×)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def mix(problem, shapes, out):
    idx = sorted(shapes)
    fams = sorted({k["fam"] for i in idx for k in shapes[i]["k"]})
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(idx) + 1.5))
    left = [0.0] * len(idx)
    for f in fams + ["gaps"]:
        vals = []
        for i in idx:
            sh = shapes[i]
            excl, gap, end = {}, 0.0, 0.0           # exclusive main-stream time: overlap goes to the kernel that started first
            for k in sorted((k for k in sh["k"] if k["st"] == 0), key=lambda k: k["s"]):
                if k["s"] > end:
                    gap += k["s"] - end
                excl[k["fam"]] = excl.get(k["fam"], 0) + max(0.0, k["s"] + k["d"] - max(k["s"], end))
                end = max(end, k["s"] + k["d"])
            gap += max(0.0, sh["span"] - end)       # side-stream tail beyond the main stream
            v = gap if f == "gaps" else excl.get(f, 0.0)
            vals.append(100 * v / sh["span"])
        ax.barh(range(len(idx)), vals, left=left, color=COLORS.get(f, "#ffffff" if f == "gaps" else "#393b79"),
                edgecolor="black" if f == "gaps" else "white", hatch="//" if f == "gaps" else None, label=("idle / waiting on side stream" if f == "gaps" else f), linewidth=0.4)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([f"#{i} {label(shapes[i])} ({shapes[i]['span']:.0f} µs)" for i in idx], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% of scored time (main stream, exclusive; gaps = main stream idle; side stream overlaps and is not counted)", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=False)
    ax.set_title(f"{problem}: where the time goes per shape", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def overview(rows, out):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = [r["p"].replace("muse_", "") for r in rows]
    ax.bar(names, [r["lead"] for r in rows], color="#dddddd", label="leader (≈)")
    ax.bar(names, [r["ours"] for r in rows], color="#d62728", width=0.5, label="ours")
    for i, r in enumerate(rows):
        ax.text(i, r["ours"] / 2, f"{r['ours']:.2f}×\nrank {r['rank']}/{r['of']}", ha="center", va="center", color="white", fontsize=8,
                weight="bold")
        ax.text(i, r["lead"] + 0.03, f"{r['lead']:.2f}×", ha="center", fontsize=8)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_ylabel("speedup vs SGLang")
    ax.legend(fontsize=8)
    ax.set_title("Board now: our speedup and rank vs the leader per problem", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="design/breakdown/visuals")
    ap.add_argument("--board", default="", help="json {problem: [speedup, rank, of, to_first]} for overview.png")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = latest_trace_jobs()
    md = ["# okc visuals (judge H200 traces; regenerate: `tools/visuals.py`)", ""]
    if a.board:
        b = json.loads(a.board)
        rows = [dict(p=p, ours=v[0], rank=v[1], of=v[2], lead=v[0] * v[3]) for p, v in b.items()]
        overview(rows, out / "overview.png")
        md += ["## Board", "", "![overview](overview.png)", ""]
    for p in PROBLEMS:
        if p not in jobs:
            continue
        shapes, summary = load(p, jobs[p])
        timeline(p, shapes, out / f"{p}-timeline.png")
        speed(p, shapes, summary, out / f"{p}-speed.png")
        mix(p, shapes, out / f"{p}-mix.png")
        md += [f"## {p} (trace job {jobs[p]})", "", f"![speed]({p}-speed.png)", "", f"![mix]({p}-mix.png)", "",
               f"![timeline]({p}-timeline.png)", ""]
    (out / "VISUALS.md").write_text("\n".join(md) + "\n")
    print("\n".join(sorted(str(x.relative_to(ROOT)) for x in out.iterdir())))


if __name__ == "__main__":
    main()
