"""TRACE kit (card C17): tools/make_trace.py, tools/trace_runtime.py (the appended block), tools/parse_trace.py.

No GPU needed for the file / format / parser tests. The generator-identity and end-to-end tests need a CUDA device and are written
for the sm_75 dev card (small cases; MoE with 32 experts instead of 256, decode_layer / MoE run their torch fallback because the
dev box has no SGLang). Run:
    set -o pipefail; tools/heavy.sh .venv/bin/python -m pytest -q tests/test_trace_kit.py
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_trace  # noqa: E402
import parse_trace  # noqa: E402

KERNELS = {
    "muse_bf16_proj": "director_v1b",
    "muse_attention_decode": "sonnet_v3",
    "muse_decode_layer": "sonnet_v3_s1",
    "muse_moe_w4a16": "sonnet_v3b",
}
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
BANNED = ("subprocess", "urllib", "requests", "socket", "os.system", ".item()", "inspect", "environ", "import os", "import sys",
          "_getframe", "__import__", "importlib")

# small cases that exercise every branch of each problem's generate_input
GEN_CASES = {
    "muse_bf16_proj": [dict(m=5, layer="unpooler", seed=3), dict(m=40, layer="router", seed=4, dist="outliers"),
                       dict(m=1, layer="qkv_gate", seed=2**62 + 5, weight=62, load="decode 1")],
    "muse_attention_decode": [
        dict(batch=3, ctx=200, heads="tp8", layer="sliding", seed=5, dist="soft", page=1, spread="agentic", q_len=8),
        dict(batch=2, ctx=2600, heads="tp8", layer="full", seed=6, dist="coherent", page=1),
        dict(batch=2, ctx=300, heads="tp8", layer="sliding", seed=7, dist="edges", page=64, kv_scale=0.015625)],
    "muse_decode_layer": [
        dict(batch=2, ctx=100, layer="sliding", seed=7, dist="soft", page=1, repeats=60, weight=46, load="decode 1"),
        dict(batch=2, ctx=120, layer="full", seed=8, dist="outliers", page=1, q_len=8, spread="agentic", local_start=32)],
    "muse_moe_w4a16": [dict(bs=4, sharding="ep8", seed=8, weight=60, load="decode 1"),
                       dict(bs=6, sharding="ep8", seed=9, dist="skewed"), dict(bs=3, sharding="full", seed=10)],
}
# one small case per problem for the end-to-end run
E2E_CASES = {
    "muse_bf16_proj": [dict(m=1, layer="qkv_gate", seed=5, weight=62, load="decode 1"),
                       dict(m=16, layer="o", seed=6, weight=62, load="decode 16"),
                       dict(m=8, layer="router", seed=7, weight=62, load="verify 1")],
    "muse_attention_decode": [dict(batch=2, ctx=300, dist="soft", heads="tp8", layer="sliding", load="decode 1", page=1, seed=11, weight=46),
                              dict(batch=3, ctx=500, dist="coherent", heads="tp8", layer="full", load="verify 1", page=1, q_len=8,
                                   spread="agentic", seed=12, weight=16)],
    "muse_decode_layer": [dict(batch=1, ctx=64, dist="soft", layer="sliding", load="decode 1", page=1, seed=11, weight=46, repeats=60)],
    "muse_moe_w4a16": [dict(bs=1, load="decode 1", sharding="ep8", seed=11, weight=60)],
}


# ------------------------------------------------------------------------------------------------------------ helpers
def build(problem, **kw):
    return make_trace.build(problem, ROOT / "kernels" / problem / f"{KERNELS[problem]}.py", **kw)


def load_trace(tmp_path, problem, cases, **overrides):
    src = build(problem, cases=cases, replays=6, budget_s=120)
    path = tmp_path / f"trace_{problem}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"trace_{problem}", path)
    mod = importlib.util.module_from_spec(spec)
    mod._TR_NO_AUTORUN = True          # attention has no prepare(): the block would otherwise run at import
    spec.loader.exec_module(mod)
    for k, v in overrides.items():
        setattr(mod, k, v)
    return mod


def load_reference(problem):
    pdir = ROOT / "problems" / problem
    for name in ("checks", "nvfp4"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(pdir))
    try:
        spec = importlib.util.spec_from_file_location(f"ref_{problem}", pdir / "reference.py")
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
    finally:
        sys.path.remove(str(pdir))
    return ref


def same(a, b, path=""):
    assert type(a) is type(b), f"{path}: {type(a)} vs {type(b)}"
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and a.device == b.device, path
        assert torch.equal(a.view(torch.uint8) if a.dtype.itemsize == 1 else a, b.view(torch.uint8) if b.dtype.itemsize == 1 else b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a} vs {b}"


def moe_small_weights(ref):
    """32 experts instead of 256 (the 2060 cannot hold 7 GB)."""
    orig = ref.random_weights

    def rw(rows, k, *, generator, device, leading=(), wide_scales=False):
        if tuple(leading) == (256,):
            leading = (32,)
        return orig(rows, k, generator=generator, device=device, leading=leading, wide_scales=wide_scales)

    ref.random_weights = rw


# ------------------------------------------------------------------------------------------------------------ files (no GPU)
@pytest.mark.parametrize("problem", list(KERNELS))
def test_trace_file_is_measurement_only_and_kernel_verbatim(problem):
    kernel = ROOT / "kernels" / problem / f"{KERNELS[problem]}.py"
    src = build(problem, replays=20, budget_s=200)          # default job = best submitted job
    compile(src, "trace", "exec")
    kernel_lines = kernel.read_text().splitlines(keepends=True)
    lines = src.splitlines(keepends=True)
    assert lines[0] == kernel_lines[0] and lines[0].startswith(f"# okc: problem={problem}")
    assert "MEASUREMENT-ONLY" in lines[1] and "NEVER submit" in lines[1]
    assert "MEASUREMENT-ONLY" in "".join(lines[:5])
    body = "".join(lines[:1] + lines[2:])
    kernel_text = kernel.read_text()
    assert body.startswith(kernel_text if kernel_text.endswith("\n") else kernel_text + "\n")     # kernel file verbatim
    appended = body[len(kernel_text):]
    for banned in BANNED:
        assert banned not in appended, banned
    import json
    job = make_trace.default_job(problem)
    cases = [b["case"] for b in json.loads((ROOT / "experiments" / "profiles" / f"{problem}.{job}.json").read_text())["benchmarks"]]
    ns = {}
    start = appended.index("_TR_CASES = [")
    exec(appended[start:appended.index("\n]", start) + 2], ns)
    assert ns["_TR_CASES"] == cases and len(cases) >= 6
    assert all(isinstance(c["seed"], int) and "weight" in c for c in cases)
    # okc_run.py refuses to submit it
    assert "MEASUREMENT-ONLY" in src
    pre = ast_prepare(kernel_text)
    assert f"_TR_HAS_PREPARE = {pre}" in appended


def ast_prepare(text):
    import ast
    return any(isinstance(n, ast.FunctionDef) and n.name == "prepare" for n in ast.parse(text).body)


def test_attention_has_no_prepare_and_runs_at_import():
    src = build("muse_attention_decode")
    assert "_TR_HAS_PREPARE = False" in src and "_tr_run(None)" in src


# ------------------------------------------------------------------------------------------------------------ generator identity (GPU)
@needs_cuda
@pytest.mark.parametrize("problem", list(KERNELS))
def test_inlined_generator_is_bit_identical_to_reference(tmp_path, problem):
    ref = load_reference(problem)
    mod = load_trace(tmp_path, problem, GEN_CASES[problem])
    for case in GEN_CASES[problem]:
        want = ref.generate_input(**case)
        got = mod._tr_generate_input(**case)
        same(want, got, problem)
        assert "_tr_" in mod._tr_generate_input.__name__
    # no inlined helper leaks a name without the prefix
    names = {n for n in vars(mod) if n.startswith("_tr_")}
    assert "_tr_generate_input" in names


# ------------------------------------------------------------------------------------------------------------ runtime pieces (CPU)
def runtime_ns():
    ns = dict(_TR_PROBLEM="p", _TR_FILE="f.py", _TR_CASES=[], _TR_REPLAYS=4, _TR_BUDGET_S=100.0, _TR_MAX_BYTES=14000,
              _TR_HAS_PREPARE=False, _TR_KFN_ARGS=1, _TR_LS_FROM_IDS=False, _TR_NO_AUTORUN=True)
    exec(compile(make_trace.RUNTIME.read_text(), "trace_runtime.py", "exec"), ns)
    return ns


def test_analyze_gaps_overlap_streams():
    ns = runtime_ns()
    # ns: A 0-100 s0 | B 120-300 s0 | C 150-250 s1 (inside B) | D 400-450 s0     markers: before 0 ends at -50, after starts 480
    rows = [(1000, 1100, 7, "A"), (1120, 1300, 7, "B"), (1150, 1250, 9, "C"), (1400, 1450, 7, "D")]
    r = ns["_tr_analyze"](rows, 950, 1480)
    ks = r["kernels"]
    assert [k["name"] for k in ks] == ["A", "B", "C", "D"]
    assert [k["s"] for k in ks] == [0, 120, 150, 400]
    assert [k["g"] for k in ks] == [0, 20, 0, 100]
    assert [k["st"] for k in ks] == [0, 0, 1, 0]
    assert [k["o"] for k in ks] == [0, 100, 100, 0]
    assert r["gaps"] == 120 and r["busy"] == 100 + 180 + 50 and r["span"] == 450 and r["head"] == 50 and r["tail"] == 30
    assert r["span"] == r["busy"] + r["gaps"]


def test_aggregate_modal_sequence_and_medians():
    ns = runtime_ns()
    mk = lambda names, d: ns["_tr_analyze"]([(100 * i, 100 * i + d, 1, n) for i, n in enumerate(names)], -10, 100 * len(names) + 10)  # noqa: E731
    reps = [mk("ab", 40), mk("ab", 50), mk("ba", 10), mk("ab", 60)]
    agg = ns["_tr_aggregate"](reps)
    assert [k["name"] for k in agg["kernels"]] == ["a", "b"] and agg["rep_ok"] == 3 and agg["rep_all"] == 4
    assert agg["kernels"][0]["d"] == 50


def fake_results(n_shapes, n_kernels, weights=None, name_len=70):
    out = []
    for i in range(n_shapes):
        ks = [dict(name=f"kernel_{(i * 7 + p) % 30}_" + "x" * (name_len - 12), s=1234.56 + 17.0 * p, d=12.34 + p, g=0.52,
                   o=0.0 if p % 4 else 3.21, st=0 if p % 5 else 1) for p in range(n_kernels)]
        agg = dict(kernels=ks, busy=1000.0, gaps=100.0, span=1100.0, head=3.0, tail=1.0, rep_ok=20, rep_all=20)
        w = (weights or {}).get(i, 46)
        case = dict(batch=32, ctx=80000, dist="soft", layer="sliding", load="decode 32", page=1, spread="agentic", weight=w, seed=1)
        out.append((i, case, dict(wall=1200.0, agg=dict(agg, span=1100.0 * 1000.0, busy=1e6, gaps=1e5, head=3e3, tail=1e3,
                                                        kernels=[dict(k, s=k["s"] * 1000, d=k["d"] * 1000, g=k["g"] * 1000, o=k["o"] * 1000)
                                                                 for k in ks]))))
    return out


@pytest.mark.parametrize("problem,shapes,kernels", [("muse_decode_layer", 12, 22), ("muse_attention_decode", 12, 3),
                                                    ("muse_moe_w4a16", 6, 8), ("muse_bf16_proj", 30, 3)])
def test_output_fits_16kb_tail_for_realistic_worst_case(problem, shapes, kernels):
    """Pessimistic fake (every name 70 chars, 30 distinct names, 4-digit start times, st/o on 1 in 4 lines). The real maxima:
    22 launches per decode_layer shape, 3 per attention, 8 per MoE, 3 per bf16_proj (judge profiles, calls summed)."""
    ns = runtime_ns()
    weights = {i: (46 if i % 2 == 0 else 16) for i in range(shapes)}
    full = ns["_tr_render"]("TV 1 p f.py", fake_results(shapes, kernels, weights), [], 190.0)
    text = ns["_tr_emit"](fake_results(shapes, kernels, weights), [], "TV 1 p f.py", 190.0)
    print(f"[size] {problem}: {shapes} shapes x {kernels} kernels -> {len(full)} bytes untruncated, {len(text)} emitted",
          file=sys.stderr)
    assert len(text) <= 14000
    dropped = [int(ln.split()[1]) for ln in text.splitlines() if ln.endswith("dropped:size")]
    assert all(weights[d] == 16 for d in dropped), dropped          # only the light shapes ever go
    if problem != "muse_decode_layer":
        assert not dropped and len(full) == len(text)
    assert len([ln for ln in text.splitlines() if ln.startswith("TS ")]) == shapes - len(dropped)


def test_oversize_drops_lowest_weight_last_first():
    ns = runtime_ns()
    weights = {i: (46 if i % 2 == 0 else 16) for i in range(12)}
    text = ns["_tr_emit"](fake_results(12, 60, weights), [], "TV 1 p f.py", 190.0)
    assert len(text) <= 14000
    dropped = [int(ln.split()[1]) for ln in text.splitlines() if ln.startswith("TE ") and ln.endswith("dropped:size")]
    kept = [int(ln.split()[1]) for ln in text.splitlines() if ln.startswith("TS ")]
    assert dropped and sorted(dropped + kept) == list(range(12))
    # every weight-16 shape (odd index) goes before any weight-46 shape, later index first
    assert all(d % 2 == 1 for d in dropped[: min(len(dropped), 6)])
    assert dropped[:3] == sorted(dropped[:3], reverse=True)
    assert not (set(dropped) & set(kept))
    # names are interned: each TN once, every TK id has a TN
    tn = [ln.split()[1] for ln in text.splitlines() if ln.startswith("TN ")]
    assert len(tn) == len(set(tn))
    tr = parse_trace.parse(text)       # (the fake timings are not self-consistent: only the structure is checked here)
    assert not [e for errs in parse_trace.check(tr).values() for e in errs if "TN" in e or "unparsable" in e or "positions" in e]


def test_single_huge_shape_is_truncated_not_lost():
    ns = runtime_ns()
    text = ns["_tr_emit"](fake_results(1, 600), [], "TV 1 p f.py", 190.0)
    assert len(text) <= 14000 and "truncated:size n=600" in text and text.count("\nTK ") > 50


# ------------------------------------------------------------------------------------------------------------ parser
GOOD = """junk line
TV 1 p f.py torch=2 cuda=13 gpu=X cupti=graph replays=20
TN 0 kernel_a
TN 1 kernel_b
TS 0 w=46 wall=110.00 span=104.00 n=3 label=a=1,b=2
TK 0.0 0 s=0.00 d=40.00 g=0.00
TK 0.1 1 s=45.00 d=50.00 g=5.00 st=1 o=10.00
TK 0.2 0 s=90.00 d=14.00 g=0.00 o=10.00
TG 0 gaps=5.00 head=3.00 tail=3.00 busy=99.00
TZ done 12
"""


def test_parser_accepts_good_and_flags_each_violation():
    tr = parse_trace.parse(GOOD)
    assert not parse_trace.check(tr)
    md = parse_trace.report(tr, {})
    assert "all checks pass" in md and "kernel_a" in md and "Largest gaps" in md
    bad = {
        "TS 0 w=46 wall=90.00": GOOD.replace("wall=110.00", "wall=90.00"),            # wall < span
        "n=3": GOOD.replace("n=3", "n=4"),
        "positions": GOOD.replace("TK 0.2", "TK 0.5"),
        "busy": GOOD.replace("busy=99.00", "busy=30.00"),
        "gaps": GOOD.replace("gaps=5.00", "gaps=40.00"),
        "TN": GOOD.replace("TN 1 kernel_b\n", ""),
        "TZ": GOOD.replace("TZ done 12\n", ""),
    }
    for what, text in bad.items():
        assert parse_trace.check(parse_trace.parse(text)), what


def test_parser_cli_exit_code(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text(GOOD)
    assert parse_trace.main([str(f), "--md", str(tmp_path / "o.md")]) == 0
    f.write_text(GOOD.replace("busy=99.00", "busy=1.00"))
    assert parse_trace.main([str(f), "--md", str(tmp_path / "o.md")]) == 1


# ------------------------------------------------------------------------------------------------------------ end to end (GPU)
def run_trace(mod, problem, capsys):
    mod._TR_MAX_BYTES = 10**7        # dev-box torch fallbacks launch ~1000 kernels per call: do not truncate
    ref = load_reference(problem)
    if problem == "muse_moe_w4a16":
        moe_small_weights(ref)
    if mod._TR_HAS_PREPARE:
        mod.prepare(ref.generate_weights(ref.WEIGHTS_SEED))
    else:
        mod._tr_run(None)
    return capsys.readouterr().out


@needs_cuda
@pytest.mark.parametrize("problem", list(KERNELS))
def test_end_to_end_dev_gpu_and_parse(tmp_path, capsys, problem):
    cases = E2E_CASES[problem]
    mod = load_trace(tmp_path, problem, cases)
    out = run_trace(mod, problem, capsys)
    log = tmp_path / "log.txt"
    log.write_text(out)
    tv = [ln for ln in out.splitlines() if ln.startswith("TV ")]        # (the kernel files print their own okc-diag lines too)
    assert tv and tv[0].startswith("TV 1 " + problem) and "cupti=graph" in tv[0], out[:300]
    tr = parse_trace.parse(out)
    assert not tr["errors"], tr["errors"]
    assert sorted(tr["shapes"], key=int) == [str(i) for i in range(len(cases))]
    for s in tr["shapes"].values():
        assert s["n"] == len(s["kernels"]) >= 1 and s["span"] > 0 and s["tg"] is not None
    assert parse_trace.check(tr) == {}, parse_trace.check(tr)
    assert parse_trace.main([str(log), "--md", str(tmp_path / "r.md")]) == 0
    assert (tmp_path / "r.md").read_text().count("## shape") == len(cases)
    print(f"[sample {problem}]\n" + "\n".join([ln for ln in out.splitlines() if ln[:1] == "T"][:14]), file=sys.stderr)
    # Jules' tools/trace_report.py reads the same lines
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "trace_report.py"), str(log), "--out", str(tmp_path / "rep")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    summary = (tmp_path / "rep" / "SUMMARY.md").read_text()
    assert f"# {problem}" in summary and (tmp_path / "rep" / f"timeline_{problem}.html").exists()
    for i in range(len(cases)):
        assert f"| {i} | " in summary


@needs_cuda
@pytest.mark.parametrize("flag", ["_TR_FORCE_CHROME", "_TR_FORCE_EAGER"])
def test_profile_fallbacks_dev_gpu(tmp_path, capsys, flag):
    problem = "muse_bf16_proj"
    mod = load_trace(tmp_path, problem, E2E_CASES[problem][:2], **{flag: True})
    out = run_trace(mod, problem, capsys)
    tr = parse_trace.parse(out)
    assert parse_trace.check(tr) == {} and len(tr["shapes"]) == 2, out[:500]
    assert ("cupti=eager" in out) == (flag == "_TR_FORCE_EAGER")


@needs_cuda
def test_failing_case_reports_te_and_continues(tmp_path, capsys):
    problem = "muse_bf16_proj"
    cases = [dict(m=1, layer="no_such_layer", seed=1, weight=62), E2E_CASES[problem][0]]
    mod = load_trace(tmp_path, problem, cases)
    out = run_trace(mod, problem, capsys)
    tr = parse_trace.parse(out)
    assert "0" in tr["errors"] and "1" in tr["shapes"] and parse_trace.check(tr) == {}


@needs_cuda
def test_budget_exhaustion_drops_remaining_cases(tmp_path, capsys):
    problem = "muse_bf16_proj"
    mod = load_trace(tmp_path, problem, E2E_CASES[problem] * 2, _TR_BUDGET_S=9.0)   # 8 s are reserved for parsing
    out = run_trace(mod, problem, capsys)
    assert out.count("dropped:budget") == 6 and "TZ done" in out


@needs_cuda
def test_cupti_failure_still_reports_walls(tmp_path, capsys):
    problem = "muse_bf16_proj"
    mod = load_trace(tmp_path, problem, E2E_CASES[problem][:2])
    mod._tr_selftest = lambda env: "fail:no_kernel_records"           # what a CUPTI-blind sandbox would answer
    out = run_trace(mod, problem, capsys)
    tr = parse_trace.parse(out)
    assert "cupti=fail:no_kernel_records" in out and len(tr["shapes"]) == 2
    assert all(s["wall"] > 0 and s["n"] == 0 and not s["kernels"] for s in tr["shapes"].values()), out
