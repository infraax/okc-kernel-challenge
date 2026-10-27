#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import breakdown
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_trace import parse as parse_trace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check"])
    parser.add_argument("trace")
    parser.add_argument("--golden", required=True)
    parser.add_argument("--ci", type=float, default=0.015)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    try:
        trace_text = Path(args.trace).read_text(errors="replace")
    except Exception as e:
        print(f"Error reading {args.trace}: {e}")
        sys.exit(1)

    tr = parse_trace(trace_text)
    if not tr["tv"]:
        print("Error: no TV line")
        sys.exit(1)

    problem = tr["tv"][0]["problem"]

    golden_path = Path(args.golden)
    golden = {}
    if golden_path.exists():
        with open(golden_path) as f:
            golden = json.load(f)

    if problem not in golden:
        golden[problem] = {}

    fail = False
    messages = []

    for idx, s in tr["shapes"].items():
        ks = s["kernels"]
        actual_n = len(ks)
        actual_span = s["span"]

        idx_str = str(idx)
        if idx_str not in golden[problem]:
            if args.update:
                golden[problem][idx_str] = {"kernels": actual_n, "span_us": actual_span}
                messages.append(f"Added golden for {problem} {idx_str}: kernels={actual_n}, span={actual_span:.2f}")
            continue

        g = golden[problem][idx_str]
        golden_n = g["kernels"]
        golden_span = g["span_us"]

        # Check kernel count
        if actual_n > golden_n:
            fail = True
            messages.append(f"FAIL: {problem} {idx_str} kernel count increased: {actual_n} > golden {golden_n}")

        # Check span vs limit
        limit_span = golden_span * (1 + args.ci)
        if actual_span > limit_span:
            fail = True
            messages.append(f"FAIL: {problem} {idx_str} span {actual_span:.2f} > limit {limit_span:.2f} (golden {golden_span:.2f} * {1 + args.ci})")

        # Check if "too good" (> 20% better) without --update
        too_good_threshold = golden_span * 0.8
        if actual_span < too_good_threshold:
            if args.update:
                golden[problem][idx_str] = {"kernels": actual_n, "span_us": actual_span}
                messages.append(f"UPDATED: {problem} {idx_str} span {actual_span:.2f} was >20% better than golden {golden_span:.2f}")
            else:
                fail = True
                messages.append(f"FAIL (too good): {problem} {idx_str} span {actual_span:.2f} is >20% better than golden {golden_span:.2f}. Run with --update.")
        else:
            if args.update and (actual_n < golden_n or actual_span < golden_span):
                golden[problem][idx_str] = {"kernels": min(actual_n, golden_n), "span_us": min(actual_span, golden_span)}
                messages.append(f"UPDATED: {problem} {idx_str} to kernels={min(actual_n, golden_n)}, span={min(actual_span, golden_span):.2f}")

    if args.update:
        with open(golden_path, "w") as f:
            json.dump(golden, f, indent=2)

    for msg in messages:
        print(msg)

    if fail:
        sys.exit(1)

if __name__ == "__main__":
    main()
