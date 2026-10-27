import argparse
import sys
import os
import json
import re
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Predict candidate score.")
    parser.add_argument('input', help="Profile file (e.g., experiments/profiles/*.txt) or JSON map {case: our_us}.")
    parser.add_argument('--what-if', type=str, help="Replace selected shapes. Format: case=us,case2=us2")
    parser.add_argument('--floor', action='store_true', help="Print speedup if every shape hit its sol.")
    parser.add_argument('--replay-us', type=float, default=0.0,
                        help="Graph replay cost subtracted from both sides per shape. Default 0: the profile's times "
                             "are taken as already net of it (unverified; TASK.md says the judge subtracts ~4 us).")
    return parser.parse_args()

def parse_task_md(problem_name):
    task_path = os.path.join("problems", problem_name, "TASK.md")
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Could not find TASK.md for problem '{problem_name}' at {task_path}")

    weights = {}
    with open(task_path, 'r', encoding='utf-8') as f:
        in_table = False
        for line in f:
            if line.startswith('| load |'):
                in_table = True
                continue
            if in_table and line.startswith('|---'):
                continue
            if in_table and line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 3:
                    load = parts[0]
                    shape = parts[1]
                    weight_str = parts[2]

                    try:
                        weight = float(weight_str)
                    except ValueError:
                        continue

                    # Instead of parsing out the layer and m from the string "m 1, layer qkv_gate",
                    # we can map the exact load and shape to the case string found in profiles.
                    # Wait, the profile has case strings like "layer=qkv_gate load=decode 1 m=1"
                    # But if we just read JSON map input, it's easier to just assume the user will
                    # pass cases that match the profile exactly.
                    # Let's write a small heuristic to match them up.

                    weights[f"{load}::{shape}"] = {'weight': weight, 'load': load, 'shape': shape}
            elif in_table and not line.strip():
                in_table = False

    return weights

def parse_profile(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        in_table = False
        for line in f:
            if line.startswith('case ') and 'sglang' in line:
                in_table = True
                continue
            if in_table:
                if not line.strip():
                    in_table = False
                    break
                parts = line.strip().split()
                if len(parts) >= 5:
                    case = " ".join(parts[:-4])
                    try:
                        sglang = float(parts[-4])
                        yours = float(parts[-3])
                        sol = float(parts[-1])
                        data[case] = {'sglang': sglang, 'yours': yours, 'sol': sol}
                    except ValueError:
                        continue
    return data

def get_case_match(case_str, tasks):
    """
    Given a profile case_str e.g. "layer=qkv_gate load=decode 1 m=1",
    and tasks map containing { "decode 1::m 1, layer qkv_gate": info },
    try to match them robustly.
    """
    case_lower = case_str.lower()

    # Extract load from case_str:
    # Look for "load="
    load = None
    m_val = None
    layer_val = None

    match_load = re.search(r'load=([\w\s]+?)(?:\s+(?:m|layer)=|$)', case_str)
    if match_load:
        load = match_load.group(1).strip()

    match_m = re.search(r'm=(\d+)', case_str)
    if match_m:
        m_val = match_m.group(1)

    match_layer = re.search(r'layer=(\w+)', case_str)
    if match_layer:
        layer_val = match_layer.group(1)

    for task_key, info in tasks.items():
        # Check if load matches
        if load and load != info['load']:
            continue

        shape = info['shape']
        # Check m_val
        shape_m = None
        match_shape_m = re.search(r'm (\d+)', shape)
        if match_shape_m:
            shape_m = match_shape_m.group(1)

        if m_val and shape_m and m_val != shape_m:
            continue

        # Check layer
        shape_layer = None
        match_shape_layer = re.search(r'layer (\w+)', shape)
        if match_shape_layer:
            shape_layer = match_shape_layer.group(1)

        if layer_val and shape_layer and layer_val != shape_layer:
            continue

        return task_key

    return None

def main():
    args = parse_args()

    # Try to infer problem_name from the input path
    problem_name = "muse_bf16_proj" # fallback
    if "muse_bf16_proj" in args.input:
        problem_name = "muse_bf16_proj"
    else:
        basename = os.path.basename(args.input)
        if '.' in basename:
            problem_name = basename.split('.')[0]

    weights = parse_task_md(problem_name)

    # Load all profiles to gather a robust SGLang baseline for JSON input
    profile_dir = "experiments/profiles"
    baseline_sglang = {}

    if os.path.exists(profile_dir):
        for p in os.listdir(profile_dir):
            if p.endswith('.txt') and p.startswith(problem_name):
                try:
                    p_data = parse_profile(os.path.join(profile_dir, p))
                    for k, v in p_data.items():
                        baseline_sglang[k] = v['sglang']
                except:
                    pass

    data = {}
    if args.input.endswith('.json'):
        with open(args.input, 'r', encoding='utf-8') as f:
            json_map = json.load(f)

        # Determine mapping of JSON cases to SGLang baselines
        for case, our_us in json_map.items():
            if case in baseline_sglang:
                data[case] = {'sglang': baseline_sglang[case], 'yours': float(our_us), 'sol': float(our_us)}
            else:
                data[case] = {'sglang': 0.0, 'yours': float(our_us), 'sol': float(our_us)}
    else:
        data = parse_profile(args.input)
        # Update baseline_sglang to include the data we just parsed, in case it wasn't in the directory
        for k, v in data.items():
            baseline_sglang[k] = v['sglang']

    if args.what_if:
        replacements = args.what_if.split(',')
        for replacement in replacements:
            if '=' in replacement:
                # case might contain '=' itself, e.g., layer=qkv_gate load=decode 1 m=1=14.47
                case, us = replacement.rsplit('=', 1)

                # Check if case exists in data, else initialize it
                if case not in data:
                    data[case] = {'sglang': baseline_sglang.get(case, 0.0), 'yours': float(us), 'sol': float(us)}
                else:
                    data[case]['yours'] = float(us)

    if args.floor:
        for case in data:
            data[case]['yours'] = data[case]['sol']

    loads_sglang = {}
    loads_yours = {}

    # The judge subtracts ~4 us graph replay cost from both sides; whether the profile's times are already net of it
    # is unverified, so it is a flag (default 0).
    GRAPH_REPLAY_COST = args.replay_us

    # Match each case in data to a shape in TASK.md
    for case, info in data.items():
        task_key = get_case_match(case, weights)
        if not task_key:
            continue

        load = weights[task_key]['load']
        weight = weights[task_key]['weight']

        if load not in loads_sglang:
            loads_sglang[load] = 0
            loads_yours[load] = 0

        sglang_time = info['sglang']
        yours_time = info['yours']

        # If the json input was provided and sglang time is missing (0.0), we might get a negative time if we subtract 4.0
        # By using max(1e-9, ...), we prevent negative times. But really a 0 sglang time means we have no baseline.
        if sglang_time == 0.0:
            sglang_time = yours_time # fallback to 1.0x speedup for missing baseline shapes

        loads_sglang[load] += max(1e-9, sglang_time - GRAPH_REPLAY_COST) * weight
        loads_yours[load] += max(1e-9, yours_time - GRAPH_REPLAY_COST) * weight

    if not loads_sglang:
        print("No matching cases found to compute speedup.")
        sys.exit(1)

    ratios = []
    for load in loads_sglang:
        if loads_yours[load] > 0:
            ratios.append(loads_sglang[load] / loads_yours[load])

    speedup = math.prod(ratios) ** (1.0 / len(ratios))
    print(f"speedup {speedup:.4f}x")

if __name__ == "__main__":
    main()
