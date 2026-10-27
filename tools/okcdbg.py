#!/usr/bin/env python3
import argparse
import csv
import sys
import re
from pathlib import Path

# Add parent directory to path to import breakdown
sys.path.insert(0, str(Path(__file__).resolve().parent))
from breakdown import family as _family

FLOOR_US = 2.5  # practical single-kernel floor (same as tools/breakdown.py)


def nfam(f):
    """Normalise families so trace kernels join the bytes model (largem/smallm/marlin are all 'moe_gemm')."""
    return "moe_gemm" if f.startswith("moe_gemm") else f


def family(name):
    return nfam(_family(name))
from parse_trace import parse as parse_trace

def load_bytes_model(csv_path):
    bytes_model = {}
    if not csv_path or not Path(csv_path).exists():
        return bytes_model
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob = row['problem']
            shape = row['shape']
            fam = nfam(row['kernel_family'])
            bytes_amt = float(row['bytes'])
            flops = float(row['flops'])
            if prob not in bytes_model:
                bytes_model[prob] = {}
            if shape not in bytes_model[prob]:
                bytes_model[prob][shape] = {}
            prev = bytes_model[prob][shape].get(fam, {'bytes': 0.0, 'flops': 0.0})
            bytes_model[prob][shape][fam] = {'bytes': prev['bytes'] + bytes_amt, 'flops': prev['flops'] + flops}
    return bytes_model

def truncate_to_budget(lines, budget):
    text = "\n".join(lines)
    if len(text) <= budget:
        return text
    # Need to truncate, drop lowest-weight shapes first (or just omit)
    # The requirement says: drop lowest-weight shapes first, print `...+N omitted`
    # Let's handle shape dropping explicitly in the report generator
    return None

def report(trace_text, bytes_csv, bw_ach=4100.0, tf_ach=600.0, budget=14000):
    tr = parse_trace(trace_text)
    bm = load_bytes_model(bytes_csv)

    # Check trace parse result
    if not tr["tv"]:
        return "Error: no TV line"

    problem = tr["tv"][0]["problem"]

    # Build shape info list
    shapes = []
    for idx, s in tr["shapes"].items():
        weight = s.get("w", 1.0)
        shapes.append((weight, float(idx), idx, s))

    # Sort shapes by weight descending, then index ascending for deterministic dropping
    shapes.sort(key=lambda x: (-x[0], x[1]))

    all_lines = []

    def generate_report(active_shapes):
        lines = ["#OKC"]

        # We need `#K`, `#D3`, `#D4`, `#S` per active shape
        for _, _, idx, s in sorted(active_shapes, key=lambda x: x[1]):
            ks = s["kernels"]
            tg = s.get("tg", {})
            wall = s.get("wall", 0.0)

            # `#K` lines
            shape_floor_us = 0.0
            shape_gpu_us = sum(k["d"] for k in ks)

            fam_d = {}
            for k in ks:
                f_ = family(tr["names"].get(k["name_id"], ""))
                fam_d[f_] = fam_d.get(f_, 0.0) + k["d"]

            def kbytes(k, fam, fam_model):
                # a family row (e.g. both MoE GEMMs) is shared across its kernels in proportion to their duration
                share = k["d"] / fam_d[fam] if fam_d.get(fam) else 1.0
                return fam_model['bytes'] * share, share

            for k in ks:
                fam = family(tr["names"].get(k["name_id"], ""))

                # floor calculation
                fam_model = bm.get(problem, {}).get(idx, {}).get(fam, None)
                floor_val = "-"
                passes_t_str = "-"
                gb_s_str = "-"
                ach_str = "-"
                x_fl_str = "-"

                if fam_model:
                    kb, share = kbytes(k, fam, fam_model)
                    floor_us = max((kb / (bw_ach * 1e9)) * 1e6, FLOOR_US * share)
                    shape_floor_us += floor_us
                    floor_val = f"{floor_us:.2f}"

                    if k["d"] > 0:
                        x_fl = k["d"] / floor_us if floor_us > 0 else 0
                        x_fl_str = f"{x_fl:.2f}"
                        # GB/s
                        gb_s = (kb / 1e9) / (k["d"] / 1e6)
                        gb_s_str = f"{gb_s:.1f}"
                        # %ach
                        ach = (gb_s / bw_ach) * 100
                        ach_str = f"{ach:.1f}%"
                        # passes_t = us * bw_ach / unique_bytes
                        passes_t = (k["d"] / 1e6) * (bw_ach * 1e9) / kb if kb > 0 else 0
                        passes_t_str = f"{passes_t:.2f}"

                pct_tot = (k["d"] / wall * 100) if wall > 0 else 0
                name = tr["names"].get(k["name_id"], "?")[:50]

                lines.append(f"#K {idx} {name} {k['d']:.2f} {pct_tot:.1f}% {gb_s_str} {ach_str} {floor_val} {x_fl_str} {passes_t_str}")

            # `#D3` lines (time-implied passes vs declared traffic/unique where the bytes model has a row)
            # D3 requires traffic/unique which we might not have in the simple CSV,
            # wait, the spec says "where the bytes model has a row".
            # We output passes_t which we computed above. Let's just output D3 lines for families in the model.
            for k in ks:
                fam = family(tr["names"].get(k["name_id"], ""))
                fam_model = bm.get(problem, {}).get(idx, {}).get(fam, None)
                if fam_model and fam_model['bytes'] > 0 and fam_model['bytes'] / (bw_ach * 1e3) >= 0.5:
                    kb, _ = kbytes(k, fam, fam_model)
                    passes_t = (k["d"] / 1e6) * (bw_ach * 1e9) / kb
                    name = tr["names"].get(k["name_id"], "?")[:50]
                    # We only have `passes_t`, spec says `#D3 lines (time-implied passes vs declared traffic/unique)`
                    lines.append(f"#D3 {idx} {name} passes_t={passes_t:.2f}")

            # `#D4` lines
            gaps = [k["g"] for k in ks if k["g"] > 0]
            gap_sum = sum(gaps)
            gap_max = max(gaps) if gaps else 0
            gap_count_3us = sum(1 for g in gaps if g > 3.0)

            copy_kernels = []
            for k in ks:
                name = tr["names"].get(k["name_id"], "")
                if re.search(r'copy_|elementwise_kernel|FillFunctor|Memcpy|Memset', name, re.IGNORECASE):
                    copy_kernels.append(name[:30])

            overlap_us = sum(k.get("o", 0.0) for k in ks)
            overlap_note = f" overlap={overlap_us:.2f}" if overlap_us > 0 else ""

            lines.append(f"#D4 {idx} gaps: sum={gap_sum:.2f} max={gap_max:.2f} >3us={gap_count_3us} copy/fill:{len(copy_kernels)}{overlap_note}")

            # `#S` summary per shape
            gaps_us = tg.get("gaps", gap_sum)
            recoverable_us = shape_gpu_us - shape_floor_us
            span = shape_gpu_us + gaps_us
            # speedup if every kernel hit its floor and gaps vanished (a ceiling, never a prediction)
            bound = span / shape_floor_us if shape_floor_us > 0 else 0.0
            x_floor = shape_gpu_us / shape_floor_us if shape_floor_us > 0 else 0

            lines.append(f"#S {idx} kernels={len(ks)} gpu_us={shape_gpu_us:.2f} gaps_us={gaps_us:.2f} floor_us={shape_floor_us:.2f} x_floor={x_floor:.2f} recoverable_us={recoverable_us:.2f} bound={bound:.2f}")

        return "\n".join(lines)

    # Try full report
    text = generate_report(shapes)
    if len(text) <= budget:
        return text

    # Keep dropping lowest weight shapes until it fits
    # shapes is sorted by (-weight, index)
    dropped = 0
    active_shapes = shapes.copy()
    while active_shapes and len(text) > budget:
        active_shapes.pop() # remove last (lowest weight)
        dropped += 1
        text = generate_report(active_shapes)
        text += f"\n...+{dropped} omitted"
        if len(text) <= budget:
            break

    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--bytes", default="design/breakdown/bytes_model.csv")
    parser.add_argument("--bw", type=float, default=4100.0)
    parser.add_argument("--tf", type=float, default=600.0)
    parser.add_argument("--budget", type=int, default=14000)
    args = parser.parse_args()

    try:
        trace_text = Path(args.trace).read_text(errors="replace")
    except Exception as e:
        print(f"Error reading {args.trace}: {e}")
        return

    out = report(trace_text, args.bytes, bw_ach=args.bw, tf_ach=args.tf, budget=args.budget)
    print(out)

if __name__ == "__main__":
    main()
