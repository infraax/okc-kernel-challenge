import pytest
from pathlib import Path
import subprocess
import json

def test_ratchet_golden(tmp_path):
    trace_text = """TV 1 muse_test dummy torch=x cuda=x gpu=x cupti=graph replays=1
TN 0 copy_kernel
TS 0 w=10 wall=100.0 span=90.0 n=1 label=x
TK 0.0 0 s=0.0 d=10.0 g=0.0 st=0 o=0.0
TG 0 gaps=0.0 head=0.0 tail=0.0 busy=10.0
TZ done 1.0
"""
    trace_file = tmp_path / "trace.log"
    trace_file.write_text(trace_text)

    golden = {
        "muse_test": {
            "0": {"kernels": 1, "span_us": 85.0}
        }
    }
    golden_file = tmp_path / "golden.json"
    golden_file.write_text(json.dumps(golden))

    # 90.0 vs 85.0 limit=85*1.015=86.275. It should fail
    res = subprocess.run(["python3", "tools/ratchet.py", "check", str(trace_file), "--golden", str(golden_file)], capture_output=True, text=True)
    assert res.returncode == 1
    assert "FAIL" in res.stdout
    assert "90.00 > limit" in res.stdout

    # Now try update mode where it was better
    trace_text2 = """TV 1 muse_test dummy torch=x cuda=x gpu=x cupti=graph replays=1
TN 0 copy_kernel
TS 0 w=10 wall=100.0 span=60.0 n=1 label=x
TK 0.0 0 s=0.0 d=10.0 g=0.0 st=0 o=0.0
TG 0 gaps=0.0 head=0.0 tail=0.0 busy=10.0
TZ done 1.0
"""
    trace_file2 = tmp_path / "trace2.log"
    trace_file2.write_text(trace_text2)

    # Check fail because it's too good without --update
    res = subprocess.run(["python3", "tools/ratchet.py", "check", str(trace_file2), "--golden", str(golden_file)], capture_output=True, text=True)
    assert res.returncode == 1
    assert "FAIL (too good)" in res.stdout

    # Update it
    res = subprocess.run(["python3", "tools/ratchet.py", "check", str(trace_file2), "--golden", str(golden_file), "--update"], capture_output=True, text=True)
    assert res.returncode == 0

    # Verify updated
    with open(golden_file) as f:
        new_g = json.load(f)
    assert new_g["muse_test"]["0"]["span_us"] == 60.0

    # Missing from golden => adds it if update
    golden_file2 = tmp_path / "golden2.json"
    golden_file2.write_text("{}")
    res = subprocess.run(["python3", "tools/ratchet.py", "check", str(trace_file2), "--golden", str(golden_file2), "--update"], capture_output=True, text=True)
    assert res.returncode == 0
    with open(golden_file2) as f:
        new_g2 = json.load(f)
    assert new_g2["muse_test"]["0"]["span_us"] == 60.0
