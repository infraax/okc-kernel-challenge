import json
import os
import subprocess
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.chain import analyze_report, classify_origin, parse_time, case_to_str

def test_inline_fixture_parsing():
    report = """kernel                                                           GPU time  share  calls
_gemv_splitk                                                      28.4 µs  24.2%      2
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      24.7 µs  21.1%      2
_epilogue_kernel                                                   5.8 µs   4.9%      1
comma_kernel_test                                              1,242.3 µs   1.2%      1
... 4 more
total                                                            1,385.9 µs"""
    
    result = analyze_report(report)
    assert len(result['rows']) == 4
    
    # Check parsing with comma
    assert result['rows'][3]['kernel'] == 'comma_kernel_test'
    assert result['rows'][3]['us'] == 1242.3
    
    assert result['more_count'] == 4
    assert result['total'] == 1385.9
    
    # Sum of listed = 28.4 + 24.7 + 5.8 + 1242.3 = 1301.2
    assert abs(result['listed_us'] - 1301.2) < 1e-5
    
    # total - listed = 1385.9 - 1301.2 = 84.7
    assert abs(result['hidden_tail'] - 84.7) < 1e-5

def test_origin_classification():
    assert classify_origin('_gemv_splitk') == 'ours'
    assert classify_origin('_epilogue_kernel') == 'ours'
    assert classify_origin('w4_kernel_something') == 'ours'
    assert classify_origin('something_moe_something') == 'ours'
    
    assert classify_origin('nvjet_sm90_tst') == 'library'
    assert classify_origin('void sglang::device::something') == 'library'
    assert classify_origin('void cublasLt::splitKreduce') == 'library'
    assert classify_origin('unknown_kernel') == 'library'

def test_ramp_flag_threshold():
    report = """kernel                                                           GPU time  share  calls
_fast_ours                                                         7.9 µs   1.0%      1
_slow_ours                                                         8.0 µs   2.0%      1
_slower_ours                                                      10.5 µs   3.0%      1
fast_library                                                       7.5 µs   1.0%      1
total                                                             33.9 µs"""
    
    result = analyze_report(report)
    rows = result['rows']
    assert len(rows) == 4
    
    assert rows[0]['kernel'] == '_fast_ours'
    assert rows[0]['ramp'] is True
    
    assert rows[1]['kernel'] == '_slow_ours'
    assert rows[1]['ramp'] is False
    
    assert rows[2]['kernel'] == '_slower_ours'
    assert rows[2]['ramp'] is False
    
    assert rows[3]['kernel'] == 'fast_library'
    assert rows[3]['ramp'] is False
    
    assert result['ramp_count'] == 1
    assert result['ramp_budget'] == 1.5

def test_summary_arithmetic():
    report = """kernel                                                           GPU time  share  calls
_gemv_splitk                                                      20.0 µs  20.0%      1
nvjet_kernel                                                      30.0 µs  30.0%      1
_fast_kernel                                                       5.0 µs   5.0%      1
total                                                            100.0 µs"""
    
    result = analyze_report(report)
    assert result['sum_ours_us'] == 25.0  # 20.0 + 5.0
    assert result['sum_lib_us'] == 30.0
    assert result['hidden_tail'] == 45.0  # 100.0 - (20+30+5)
    assert result['ramp_count'] == 1      # _fast_kernel (5.0 < 8.0)
    assert result['ramp_budget'] == 1.5

def test_case_to_str():
    case = {
        "batch": 1,
        "ctx": 80000,
        "dist": "soft",
        "layer": "sliding",
        "load": "decode 1",
        "page": 1,
        "repeats": 60,
        "seed": 3945212514520402400,
        "weight": 46
    }
    s = case_to_str(case)
    assert s == "batch=1 ctx=80000 dist=soft layer=sliding load=decode 1 page=1"
    
    case_spread = {
        "batch": 16,
        "ctx": 80000,
        "dist": "soft",
        "layer": "sliding",
        "load": "decode 16",
        "page": 1,
        "spread": "agentic"
    }
    s = case_to_str(case_spread)
    assert s == "batch=16 ctx=80000 dist=soft layer=sliding load=decode 16 page=1 spread=agentic"

def test_cli_end_to_end(tmp_path):
    json_path = tmp_path / "dummy.json"
    txt_path = tmp_path / "dummy.txt"
    
    dummy_json = {
        "benchmarks": [],
        "profiles": [
            {
                "case": {
                    "batch": 1,
                    "ctx": 80000,
                    "dist": "soft",
                    "layer": "sliding",
                    "load": "decode 1",
                    "page": 1
                },
                "event": "profile",
                "index": 0,
                "report": "kernel                                                           GPU time  share  calls\n_gemv_splitk                                                      28.4 µs  24.2%      2\nnvjet_kernel                                                      10.0 µs  10.0%      1\n... 2 more\ntotal                                                            100.0 µs"
            }
        ]
    }
    
    dummy_txt = """okc run 09e8f297c372409c on muse_decode_layer
submission  dummy.py

scored shapes (µs; speed of light is the reference's bound)
case                                                         sglang      yours  speedup        sol  
batch=1 ctx=80000 dist=soft layer=sliding load=decode 1 page=1     173.20     129.19   1.341x      40.02  
"""
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dummy_json, f)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(dummy_txt)
        
    cmd = [sys.executable, os.path.join("tools", "chain.py"), str(json_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    out = result.stdout
    assert "case batch=1 ctx=80000 dist=soft layer=sliding load=decode 1 page=1" in out
    assert "ours 129.19 µs" in out
    assert "sol 40.02 µs" in out
    assert "origin=ours: _gemv_splitk" in out
    assert "origin=library: nvjet_kernel" in out
    assert "Σ ours µs: 28.40" in out
    assert "Σ library µs: 10.00" in out
    assert "hidden tail: 61.60 µs" in out
