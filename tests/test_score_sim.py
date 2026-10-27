import subprocess
import os
import json

def test_score_sim_default():
    cmd = ["python", "tools/score_sim.py", "experiments/profiles/muse_bf16_proj.728c9344dc3b4559.txt"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "speedup 1.0022x" in result.stdout  # judge: 1.00217 (replay cost 0: profile times are net)

def test_score_sim_what_if():
    cmd = ["python", "tools/score_sim.py", "experiments/profiles/muse_bf16_proj.728c9344dc3b4559.txt", "--what-if", "layer=qkv_gate load=decode 1 m=1=5.0"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "speedup" in result.stdout

def test_score_sim_floor():
    cmd = ["python", "tools/score_sim.py", "experiments/profiles/muse_bf16_proj.728c9344dc3b4559.txt", "--floor"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "speedup" in result.stdout

def test_score_sim_json(tmp_path):
    json_path = tmp_path / "muse_bf16_proj.json" # Ensure problem_name is parsed correctly
    with open(json_path, 'w') as f:
        json.dump({"layer=qkv_gate load=decode 1 m=1": 10.0}, f)

    cmd = ["python", "tools/score_sim.py", str(json_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "speedup" in result.stdout
