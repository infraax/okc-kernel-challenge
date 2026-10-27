import argparse
import json
import os
import re
import sys

def parse_txt_scored_shapes(txt_path):
    scored_shapes = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        in_table = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('case') and 'sglang' in line:
                in_table = True
                continue
            if in_table:
                if line.startswith('profile'):
                    break
                parts = line.rsplit(maxsplit=4)
                if len(parts) == 5:
                    case = parts[0].strip()
                    sglang = float(parts[1])
                    yours = float(parts[2])
                    sol = float(parts[4])
                    scored_shapes[case] = {'yours': yours, 'sol': sol}
    return scored_shapes

def case_to_str(case_dict):
    parts = []
    # Matching the exact case keys and formatting
    order = ['batch', 'ctx', 'dist', 'layer', 'load', 'page', 'q_len', 'spread']
    for k in order:
        if k in case_dict:
            parts.append(f"{k}={case_dict[k]}")
    # Catch any other keys that are not test runner internals just in case
    for k, v in sorted(case_dict.items()):
        if k not in order and k not in ('seed', 'repeats', 'weight'):
            parts.append(f"{k}={v}")
    return " ".join(parts)

def classify_origin(kernel_name):
    if kernel_name.startswith('sglang::') or 'sglang::' in kernel_name or kernel_name.startswith('nvjet') or 'cublas' in kernel_name:
        return 'library'
    if kernel_name.startswith('_') or 'w4_' in kernel_name or 'moe' in kernel_name:
        return 'ours'
    return 'library'

def parse_time(time_str):
    return float(time_str.replace(',', ''))

def analyze_report(report_text):
    rows = []
    total = 0.0
    hidden_tail = 0.0
    
    row_re = re.compile(r"^(.*?)\s+([\d,.]+)\s+(µs|ms|ns|s)\s+([\d.]+%)\s+(\d+)$")
    more_re = re.compile(r"^\.\.\.\s+(\d+)\s+more$")
    total_re = re.compile(r"^total\s+([\d,.]+)\s+(µs|ms|ns|s)$")
    
    more_count = 0
    listed_us = 0.0
    
    lines = report_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('kernel'):
            continue
            
        match_row = row_re.match(line)
        if match_row:
            kernel, t_val_str, unit, share, calls = match_row.groups()
            t_val = parse_time(t_val_str)
            if unit == 'ms':
                t_val *= 1000.0
            elif unit == 'ns':
                t_val /= 1000.0
            elif unit == 's':
                t_val *= 1000000.0
                
            origin = classify_origin(kernel)
            ramp = origin == 'ours' and t_val < 8.0
            
            rows.append({
                'kernel': kernel,
                'us': t_val,
                'share': share,
                'calls': int(calls),
                'origin': origin,
                'ramp': ramp
            })
            listed_us += t_val
            continue
            
        match_more = more_re.match(line)
        if match_more:
            more_count = int(match_more.group(1))
            continue
            
        match_total = total_re.match(line)
        if match_total:
            t_val_str, unit = match_total.groups()
            t_val = parse_time(t_val_str)
            if unit == 'ms':
                t_val *= 1000.0
            elif unit == 'ns':
                t_val /= 1000.0
            elif unit == 's':
                t_val *= 1000000.0
            total = t_val
            continue
            
    hidden_tail = max(0.0, total - listed_us)
    
    sum_ours_us = sum(r['us'] for r in rows if r['origin'] == 'ours')
    sum_lib_us = sum(r['us'] for r in rows if r['origin'] == 'library')
    ramp_count = sum(1 for r in rows if r['ramp'])
    ramp_budget = 1.5 * ramp_count
    
    return {
        'rows': rows,
        'more_count': more_count,
        'total': total,
        'listed_us': listed_us,
        'hidden_tail': hidden_tail,
        'sum_ours_us': sum_ours_us,
        'sum_lib_us': sum_lib_us,
        'ramp_count': ramp_count,
        'ramp_budget': ramp_budget
    }

def main():
    parser = argparse.ArgumentParser(description="kernel-chain analyzer for judge profiles")
    parser.add_argument("json", help="path to JSON profile")
    parser.add_argument("--shape", type=int, help="index of shape to process")
    parser.add_argument("--json", action="store_true", help="output as JSON", dest="out_json")
    args = parser.parse_args()
    
    json_path = args.json
    if not json_path.endswith('.json'):
        print("Expected a .json file", file=sys.stderr)
        sys.exit(1)
        
    txt_path = json_path[:-5] + '.txt'
    if not os.path.exists(txt_path):
        print(f"Text profile not found: {txt_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}", file=sys.stderr)
        sys.exit(1)
        
    scored_shapes = parse_txt_scored_shapes(txt_path)
    
    profiles = data.get('profiles', [])
    indices = range(len(profiles))
    if args.shape is not None:
        if 0 <= args.shape < len(profiles):
            indices = [args.shape]
        else:
            print(f"Shape index {args.shape} out of range (0-{len(profiles)-1})", file=sys.stderr)
            sys.exit(1)
            
    out_results = []
    
    for idx in indices:
        prof = profiles[idx]
        case_dict = prof.get('case', {})
        case_str = case_to_str(case_dict)
        
        scores = scored_shapes.get(case_str, {'yours': 0.0, 'sol': 0.0})
        ours_us = scores['yours']
        sol_us = scores['sol']
        ours_sol = ours_us / sol_us if sol_us > 0 else 0.0
        
        report_text = prof.get('report', '')
        analysis = analyze_report(report_text)
        
        result = {
            'index': idx,
            'case': case_str,
            'ours_us': ours_us,
            'sol_us': sol_us,
            'ours_sol': ours_sol,
            'analysis': analysis
        }
        out_results.append(result)
        
    if args.out_json:
        print(json.dumps(out_results, indent=2))
    else:
        for r in out_results:
            print(f"case {r['case']}")
            print(f"ours {r['ours_us']:.2f} µs")
            print(f"sol {r['sol_us']:.2f} µs")
            print(f"ours/sol {r['ours_sol']:.3f}")
            for row in r['analysis']['rows']:
                ramp_str = " ramp?" if row['ramp'] else ""
                print(f"{row['us']:.2f} µs, {row['share']}, {row['calls']}, origin={row['origin']}{ramp_str}: {row['kernel']}")
            print(f"Σ ours µs: {r['analysis']['sum_ours_us']:.2f}")
            print(f"Σ library µs: {r['analysis']['sum_lib_us']:.2f}")
            print(f"ramp? kernels: {r['analysis']['ramp_count']}")
            print(f"hidden tail: {r['analysis']['hidden_tail']:.2f} µs")
            print(f"ramp budget: {r['analysis']['ramp_budget']:.2f} µs (est.)")
            print()

if __name__ == '__main__':
    main()
