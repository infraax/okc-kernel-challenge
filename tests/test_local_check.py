"""tools/local_check.py: case parsing for all five problems, the starter submissions pass on CPU
(bf16_proj, attention_decode: fast; the layer / MoE starters take minutes and are run by hand, see
design/cards/C2-local-check.md), and deliberately wrong kernels FAIL.

Run through tools/heavy.sh: .venv/bin/python -m pytest -q tests/test_local_check.py"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import local_check as lc  # noqa: E402

EXPECTED_TESTS = {"muse_bf16_proj": 9, "muse_attention_decode": 16, "muse_moe_w4a16": 5,
                  "muse_decode_layer": 13, "muse_prefill_layer": 5}


def test_parse_case():
    assert lc.parse_case("batch 4, ctx 6000, heads tp8, layer full, q_len 8, kv_scale 0.015625") == {
        "batch": 4, "ctx": 6000, "heads": "tp8", "layer": "full", "q_len": 8, "kv_scale": 0.015625}
    with pytest.raises(ValueError):
        lc.parse_case("batch")


@pytest.mark.parametrize("problem,n", sorted(EXPECTED_TESTS.items()))
def test_cases_parse_for_every_problem(problem, n):
    assert len(lc.select_cases(problem, "tests")) == n
    assert len(lc.select_cases(problem, "small")) == n
    assert len(lc.select_cases(problem, "scored")) >= 5
    assert len(lc.select_cases(problem, "all")) == n + len(lc.select_cases(problem, "scored"))


def test_scored_attention_table():
    scored = lc.select_cases("muse_attention_decode", "scored")
    assert len(scored) == 12  # 3 batch sizes x sliding/full x decode/verify
    assert {c["ctx"] for c in scored} == {80000} and {c.get("q_len", 1) for c in scored} == {1, 8}
    assert all(c["page"] == 1 for c in scored)


def test_scale_down_keeps_what_the_test_is_about():
    big = {"batch": 2, "ctx": 1048576, "heads": "tp8", "layer": "full", "q_len": 8}
    small = lc.scale_down(big)
    assert 2048 < small["ctx"] <= 4096 and small["q_len"] == 8 and small["layer"] == "full"
    assert lc.scale_down({"batch": 33, "ctx": 20000})["batch"] == 4
    assert lc.scale_down({"batch": 3, "ctx": 10}) == {"batch": 3, "ctx": 10}
    assert lc.scale_down({"m": 1000, "layer": "unpooler"})["m"] <= 46
    assert lc.scale_down({"layer": "sliding", "ctx": 0, "q_len": 16384})["q_len"] <= 124


def _run(problem, path, **kw):
    lines = []
    passed, total = lc.run(problem, path, "cpu", "small", out=lines.append, **kw)
    return passed, total, lines


def test_bf16_proj_starter_passes():
    p = ROOT / "problems/muse_bf16_proj/submission.py"
    passed, total, lines = _run("muse_bf16_proj", p)
    assert passed == total == 9, lines
    assert lines[-1] == "local_check: muse_bf16_proj 9/9 passed"


def test_bf16_proj_wrong_kernel_fails(tmp_path):
    f = tmp_path / "wrong.py"
    f.write_text(
        "# okc: problem=muse_bf16_proj\n"
        "import torch\n"
        "def prepare(w):\n    return w\n"
        "def kernel_fn(data, weights):\n"
        "    w = weights[data['config']['layer']][: data['config']['n']]\n"
        "    out = data['x'].float() @ w.float().T\n"
        "    out = out * 1.05\n"
        "    return out if w.dtype == torch.float32 else out.to(torch.bfloat16)\n")
    passed, total, lines = _run("muse_bf16_proj", f)
    assert passed == 0 and total == 9
    assert all(line.startswith("FAIL") for line in lines[:-1])
    assert lines[-1] == "local_check: muse_bf16_proj 0/9 passed"


def test_crash_and_wrong_signature_are_reported(tmp_path):
    crash = tmp_path / "crash.py"
    crash.write_text("# okc: problem=muse_bf16_proj\ndef prepare(w):\n    return w\n"
                     "def kernel_fn(data, weights):\n    raise RuntimeError('boom')\n")
    passed, total, lines = _run("muse_bf16_proj", crash)
    assert passed == 0 and "RuntimeError: boom" in lines[0]
    noweights = tmp_path / "noweights.py"  # bf16_proj has weights, kernel_fn must take them
    noweights.write_text("# okc: problem=muse_bf16_proj\ndef kernel_fn(data):\n    return data['x']\n")
    with pytest.raises(ValueError):
        _run("muse_bf16_proj", noweights)


def test_attention_starter_passes_and_wrong_kernel_fails(tmp_path):
    starter = ROOT / "problems/muse_attention_decode/submission.py"
    passed, total, lines = _run("muse_attention_decode", starter)
    assert passed == total == 16, lines
    src = starter.read_text()
    marker = "return out.reshape(batch, length, hq * d).to(torch.bfloat16)"
    assert marker in src
    f = tmp_path / "wrong_attention.py"
    f.write_text(src.replace(marker, "return (0.5 * out.reshape(batch, length, hq * d)).to(torch.bfloat16)"))
    passed, total, lines = _run("muse_attention_decode", f)
    assert passed == 0 and total == 16, lines
    assert "relative L2 error" in lines[0]


def test_main_exit_codes(tmp_path, capsys):
    starter = str(ROOT / "problems/muse_attention_decode/submission.py")
    assert lc.main(["muse_attention_decode", starter, "--device", "cpu", "--cases", "small"]) == 0
    assert lc.main(["muse_attention_decode", starter, "--device", "cpu", "--graph"]) == 2  # graph needs cuda
