#!/usr/bin/env python3
import argparse
import csv
import math
import os
import xml.sax.saxutils as saxutils
from collections import defaultdict
from pathlib import Path
import sys

def family(name):
    n = name.lower()
    for key, fam in (("marlin", "moe_gemm(marlin)"), ("smallm", "moe_gemm(cuda smallm)"), ("nvjet", "dense_gemm(cublas)"),
                     ("cublaslt::splitkreduce", "dense_gemm(cublas)"), ("gemm", "dense_gemm"), ("gemv", "dense_gemm"),
                     ("attn", "attention"), ("reduce", "attention_reduce"), ("norm", "norm/residual"),
                     ("layer_norm", "norm/residual"), ("routing", "routing"), ("topk", "routing"), ("align", "routing"),
                     ("act_and_mul", "activation"), ("silu", "activation"), ("epilogue", "moe_epilogue"),
                     ("post_expert", "moe_epilogue"), ("copy", "copy/cast"), ("elementwise", "elementwise"),
                     ("fill", "fill/memset"), ("memset", "fill/memset")):
        if key in n:
            return fam
    return "other"

FAM_COLORS = {
    "moe_gemm(marlin)": "#ffb3ba",
    "moe_gemm(cuda smallm)": "#ffdfba",
    "dense_gemm(cublas)": "#ffffba",
    "dense_gemm": "#baffc9",
    "attention": "#bae1ff",
    "attention_reduce": "#e1baff",
    "norm/residual": "#ffb3f7",
    "routing": "#b3ffd9",
    "activation": "#d9b3ff",
    "moe_epilogue": "#ffd9b3",
    "copy/cast": "#cccccc",
    "elementwise": "#e6e6e6",
    "fill/memset": "#f2f2f2",
    "other": "#dddddd"
}

def get_color(fam):
    return FAM_COLORS.get(fam, "#dddddd")

def parse_kv(parts):
    d = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d

def gen_html(problem, data, out_dir):
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: sans-serif; margin: 20px; background: #fafafa; }",
        "h1, h2, h3 { color: #333; }",
        ".shape { margin-bottom: 40px; background: white; padding: 10px; border: 1px solid #ccc; }",
        ".err { color: red; font-weight: bold; }",
        ".lane { font-size: 12px; fill: #666; }",
        "svg { border: 1px solid #eee; background: #fff; }",
        "</style>",
        f"<title>Timeline: {problem}</title>",
        "</head><body>",
        f"<h1>Timeline: {problem}</h1>"
    ]

    for s in data["sorted_shapes"]:
        html.append(f"<div class='shape'>")
        title = f"Shape {s['idx']} (weight: {s['weight']}, wall: {s['wall']} &mu;s, span: {s['span']} &mu;s"
        if s['sol']:
            title += f", sol: {s['sol']} &mu;s"
        title += f")<br><i>{saxutils.escape(s['label'])}</i>"
        html.append(f"<h3>{title}</h3>")
        if s['te']:
            html.append(f"<p class='err'>Error: {saxutils.escape(s['te'])}</p>")

        kernels = s["kernels"]
        if not kernels:
            html.append("</div>")
            continue

        streams = set(k["stream"] for k in kernels)
        stream_list = sorted(list(streams))
        stream_map = {st: i for i, st in enumerate(stream_list)}

        lane_h = 30
        margin_y = 20
        svg_h = len(stream_list) * lane_h + 2 * margin_y

        max_t = 0
        for k in kernels:
            max_t = max(max_t, k["start"] + k["dur"])

        if max_t == 0:
            max_t = 1.0

        width = 1200
        scale = width / max_t

        html.append(f"<svg width='{width+100}' height='{svg_h}'>")

        for i, st in enumerate(stream_list):
            y = margin_y + i * lane_h
            html.append(f"<text x='5' y='{y + 20}' class='lane'>Stream {st}</text>")
            html.append(f"<line x1='80' y1='{y + lane_h}' x2='{width+80}' y2='{y + lane_h}' stroke='#eee'/>")

        for i, k in enumerate(kernels):
            lane_idx = stream_map[k["stream"]]
            y = margin_y + lane_idx * lane_h + 5
            x = 80 + k["start"] * scale
            w = max(k["dur"] * scale, 1.0)

            name = data["names"].get(k["name_id"], k["name_id"])
            fam = family(name)
            color = get_color(fam)

            if k["gap"] >= 1.0:
                gap_w = k["gap"] * scale
                gap_x = x - gap_w
                if gap_x >= 80:
                    html.append(f"<rect x='{gap_x}' y='{y}' width='{gap_w}' height='{lane_h - 10}' fill='url(#hatched)' stroke='#999' stroke-width='0.5'/>")
                    html.append(f"<text x='{gap_x + 2}' y='{y - 2}' font-size='10' fill='#999'>{k['gap']:.1f}</text>")

            tt = f"[{k['pos']}] {name}\nStart: {k['start']:.2f} \u03bcs\nDur: {k['dur']:.2f} \u03bcs\nGap before: {k['gap']:.2f} \u03bcs"
            html.append(f"<rect x='{x}' y='{y}' width='{w}' height='{lane_h - 10}' fill='{color}' stroke='#333' stroke-width='1'>")
            html.append(f"<title>{saxutils.escape(tt)}</title>")
            html.append("</rect>")

        html.append("<defs><pattern id='hatched' width='4' height='4' patternUnits='userSpaceOnUse'>")
        html.append("<path d='M-1,1 l2,-2 M0,4 l4,-4 M3,5 l2,-2' stroke='#ccc' stroke-width='1'/></pattern></defs>")
        html.append("</svg>")
        html.append("</div>")

    html.append("</body></html>")

    with open(out_dir / f"timeline_{problem}.html", "w") as f:
        f.write("\n".join(html))

def gen_md_and_csv(problems, bytes_model, out_dir):
    md = []
    csv_rows = []

    cross_items = []

    for prob, data in sorted(problems.items()):
        md.append(f"# {prob}")
        md.append("")
        md.append("## Shapes")
        md.append("| index | label | wall | &Sigma;dur | busy | &Sigma;gaps | head | tail | kernels |")
        md.append("|---|---|---|---|---|---|---|---|---|")

        prob_items = []
        fam_stats = defaultdict(lambda: {"w_us": 0.0, "bytes": 0.0, "flops": 0.0})

        for s in data["sorted_shapes"]:
            w = s["weight"]
            sum_dur = sum(k["dur"] for k in s["kernels"])
            tg = s["tg"] or {"busy": 0.0, "gaps": 0.0, "head": 0.0, "tail": 0.0}

            md.append(f"| {s['idx']} | {s['label']} | {s['wall']:.2f} | {sum_dur:.2f} | {tg['busy']:.2f} | "
                      f"{tg['gaps']:.2f} | {tg['head']:.2f} | {tg['tail']:.2f} | {len(s['kernels'])} |")

            for k in s["kernels"]:
                name = data["names"].get(k["name_id"], k["name_id"])
                fam = family(name)

                item = {
                    "problem": prob, "shape": s["idx"], "label": s["label"], "weight": w,
                    "pos": k["pos"], "kind": "kernel", "name": name, "family": fam,
                    "start_us": k["start"], "dur_us": k["dur"], "stream": k["stream"],
                    "w_us": w * k["dur"]
                }
                prob_items.append(item)
                cross_items.append(item)

                fam_stats[fam]["w_us"] += item["w_us"]

                csv_rows.append([prob, s["idx"], s["label"], w, k["pos"], "kernel", name, fam, k["start"], k["dur"], k["stream"]])

                if k["gap"] > 0:
                    gap_item = {
                        "problem": prob, "shape": s["idx"], "label": s["label"], "weight": w,
                        "pos": k["pos"], "kind": "gap", "name": "(gap)", "family": "other",
                        "start_us": k["start"] - k["gap"], "dur_us": k["gap"], "stream": k["stream"],
                        "w_us": w * k["gap"]
                    }
                    prob_items.append(gap_item)
                    cross_items.append(gap_item)
                    csv_rows.append([prob, s["idx"], s["label"], w, k["pos"], "gap", "(gap)", "other", k["start"] - k["gap"], k["gap"], k["stream"]])

            if tg["head"] > 0:
                head_item = {
                    "problem": prob, "shape": s["idx"], "label": s["label"], "weight": w,
                    "pos": "-", "kind": "head", "name": "(head)", "family": "other",
                    "start_us": 0.0, "dur_us": tg["head"], "stream": "-",
                    "w_us": w * tg["head"]
                }
                prob_items.append(head_item)
                cross_items.append(head_item)
                csv_rows.append([prob, s["idx"], s["label"], w, "-", "head", "(head)", "other", 0.0, tg["head"], "-"])

            if tg["tail"] > 0:
                last_end = max([k["start"] + k["dur"] for k in s["kernels"]] + [0.0])
                tail_item = {
                    "problem": prob, "shape": s["idx"], "label": s["label"], "weight": w,
                    "pos": "-", "kind": "tail", "name": "(tail)", "family": "other",
                    "start_us": last_end, "dur_us": tg["tail"], "stream": "-",
                    "w_us": w * tg["tail"]
                }
                prob_items.append(tail_item)
                cross_items.append(tail_item)
                csv_rows.append([prob, s["idx"], s["label"], w, "-", "tail", "(tail)", "other", last_end, tg["tail"], "-"])

            for fam in fam_stats.keys():
                bm = bytes_model.get((prob, str(s["idx"]), fam))
                if bm:
                    fam_stats[fam]["bytes"] += bm["bytes"] * w
                    fam_stats[fam]["flops"] += bm["flops"] * w

        md.append("")
        md.append("## Top 15 items")
        md.append("| rank | kind | shape | name | stream | start &mu;s | dur &mu;s | weight &times; &mu;s |")
        md.append("|---|---|---|---|---|---|---|---|")

        prob_items.sort(key=lambda x: x["w_us"], reverse=True)
        for i, item in enumerate(prob_items[:15]):
            md.append(f"| {i+1} | {item['kind']} | {item['shape']} | {item['name'][:60]} | {item['stream']} | "
                      f"{item['start_us']:.2f} | {item['dur_us']:.2f} | {item['w_us']:.2f} |")

        md.append("")

        total_w_us = sum(f["w_us"] for f in fam_stats.values()) or 1.0

        # Family stats
        has_bytes = any(f["bytes"] > 0 for f in fam_stats.values())
        md.append("## Families")
        if has_bytes:
            md.append("| family | w &mu;s | share | floor &mu;s | efficiency |")
            md.append("|---|---|---|---|---|")
        else:
            md.append("| family | w &mu;s | share |")
            md.append("|---|---|---|")

        for fam, stats in sorted(fam_stats.items(), key=lambda x: x[1]["w_us"], reverse=True):
            share = stats["w_us"] / total_w_us * 100
            if has_bytes:
                floor_us = max(stats["bytes"] / 4.0e12, stats["flops"] / 989e12) * 1e6
                eff = (floor_us / stats["w_us"]) * 100 if stats["w_us"] > 0 else 0.0
                md.append(f"| {fam} | {stats['w_us']:.2f} | {share:.1f}% | {floor_us:.2f} | {eff:.1f}% |")
            else:
                md.append(f"| {fam} | {stats['w_us']:.2f} | {share:.1f}% |")

        md.append("")

    md.append("# Top 20 items (all problems)")
    md.append("| rank | prob | kind | shape | name | dur &mu;s | weight &times; &mu;s |")
    md.append("|---|---|---|---|---|---|---|")

    cross_items.sort(key=lambda x: x["w_us"], reverse=True)
    for i, item in enumerate(cross_items[:20]):
        md.append(f"| {i+1} | {item['problem']} | {item['kind']} | {item['shape']} | {item['name'][:60]} | "
                  f"{item['dur_us']:.2f} | {item['w_us']:.2f} |")

    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md))

    with open(out_dir / "items.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "shape", "label", "weight", "pos", "kind", "name", "family", "start_us", "dur_us", "stream"])
        writer.writerows(csv_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Trace log files")
    parser.add_argument("--bytes", type=str, help="Bytes model CSV")
    parser.add_argument("--out", type=str, default="design/breakdown/report")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sol_map = {}
    kernels_csv = Path("design/breakdown/kernels.csv")
    if kernels_csv.exists():
        with open(kernels_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sol_map[(row["problem"], str(row["shape"]))] = float(row["sol_us"])
                except ValueError:
                    pass

    bytes_model = {}
    if args.bytes and Path(args.bytes).exists():
        with open(args.bytes, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prob = row["problem"]
                shape = str(row["shape"])
                fam = row["kernel_family"]
                b = float(row["bytes"])
                fl = float(row["flops"])
                bytes_model[(prob, shape, fam)] = {"bytes": b, "flops": fl}

    problems = {}

    for log_file in args.logs:
        with open(log_file, "r", encoding="utf-8") as f:
            problem = None
            names = {}
            shapes = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0]
                if cmd not in ("TV", "TN", "TS", "TK", "TG", "TE", "TZ"):
                    continue

                try:
                    if cmd == "TV" and len(parts) >= 3:
                        problem = parts[2]
                        if problem not in problems:
                            problems[problem] = {"names": {}, "shapes": {}}
                        names = problems[problem]["names"]
                        shapes = problems[problem]["shapes"]
                    elif cmd == "TN" and len(parts) >= 3:
                        names[parts[1]] = " ".join(parts[2:])
                    elif cmd == "TS" and len(parts) >= 3:
                        shape_idx = parts[1]
                        kv = parse_kv(parts[2:])
                        shapes[shape_idx] = {
                            "idx": shape_idx,
                            "weight": float(kv.get("w", 1.0)),
                            "wall": float(kv.get("wall", 0.0)),
                            "span": float(kv.get("span", 0.0)),
                            "n": int(kv.get("n", 0)),
                            "label": str(kv.get("label", "")),
                            "kernels": [],
                            "tg": None,
                            "te": None
                        }
                    elif cmd == "TK" and len(parts) >= 4:
                        idx_pos = parts[1]
                        if "." in idx_pos:
                            shape_idx = idx_pos.split(".")[0]
                        else:
                            shape_idx = idx_pos
                        if shape_idx in shapes:
                            name_id = parts[2]
                            kv = parse_kv(parts[3:])
                            shapes[shape_idx]["kernels"].append({
                                "pos": idx_pos,
                                "name_id": name_id,
                                "start": float(kv.get("s", 0.0)),
                                "dur": float(kv.get("d", 0.0)),
                                "gap": float(kv.get("g", 0.0)),
                                "stream": int(float(kv.get("st", 0))),
                                "overlap": float(kv.get("o", 0.0))
                            })
                    elif cmd == "TG" and len(parts) >= 2:
                        shape_idx = parts[1]
                        if shape_idx in shapes:
                            kv = parse_kv(parts[2:])
                            shapes[shape_idx]["tg"] = {
                                "gaps": float(kv.get("gaps", 0.0)),
                                "head": float(kv.get("head", 0.0)),
                                "tail": float(kv.get("tail", 0.0)),
                                "busy": float(kv.get("busy", 0.0))
                            }
                    elif cmd == "TE" and len(parts) >= 3:
                        shape_idx = parts[1]
                        if shape_idx in shapes:
                            shapes[shape_idx]["te"] = " ".join(parts[2:])
                except Exception:
                    pass

    for prob, data in problems.items():
        shape_list = list(data["shapes"].values())
        for s in shape_list:
            sol = sol_map.get((prob, str(s["idx"])), None)
            s["sol"] = sol
            if sol and sol > 0:
                s["sort_val"] = s["weight"] * math.log(max(s["wall"], 1e-9) / sol)
            else:
                s["sort_val"] = -float('inf')

        def sort_key(s):
            if s["sol"] and s["sol"] > 0:
                return (0, -(s["weight"] * math.log(max(s["wall"], 1e-9) / s["sol"])))
            return (1, float(s["idx"]) if s["idx"].isdigit() else 0)

        shape_list.sort(key=sort_key)
        data["sorted_shapes"] = shape_list

        gen_html(prob, data, out_dir)

    gen_md_and_csv(problems, bytes_model, out_dir)

if __name__ == "__main__":
    main()
