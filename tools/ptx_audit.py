#!/usr/bin/env python3
"""C29: PTX / resource audit of every Triton kernel a candidate file launches (and of its NVRTC CUDA source, when it has one).

  .venv/bin/python tools/ptx_audit.py <kernel_file> [--configs auto] [--cases scored|tests|all] [--filter REGEX]
                                      [--md design/breakdown/ptx-audit.md] [--no-nvrtc]

How: the file's own host code runs on the dev GPU (any CUDA card: only the Triton driver is needed) with
  * torch.cuda.get_device_capability patched to (9, 0), so the file takes its Hopper branches,
  * JITFunction.run patched to RECORD the launch instead of running it: the kernel, its argument types and
    16-byte-alignment / ==1 specialisation (Triton's own binder), its constexprs, num_warps / num_stages, the grid,
  * inputs shaped like the judged shapes (problems/<p>/TASK.md scored table; caches are torch.empty, contents are irrelevant).
Every distinct (kernel, constexprs, argument types, options) is then AOT-compiled for GPUTarget("cuda", 90, 32) (sm_90a),
so the PTX is what the H200 runs; registers and spills come from the bundled ptxas -v on that PTX.

One line per (kernel, config); all counts are STATIC instruction counts in the PTX (an unrolled loop counts every copy):
  #C <kernel> <config> regs=<n> spills=<spill store+load bytes> smem=<bytes> ld.global=<n> ld.global.nc=<n> cp.async=<n>
     cp.async.bulk=<n> cp.async.bulk.tensor=<n> mbarrier=<n> ldmatrix=<n> mma.sync=<n> wgmma=<n> cvt=<n> ex2=<n> bar.sync=<n> atom=<n> st.global=<n>
ld.global counts every global load including ld.global.nc (a subset); cp.async = cp.async.ca/cg (the 4/8/16-byte async copies);
atom = atom.* + red.* (a Triton atomic_add whose result is unused lowers to red); smem = dynamic shared bytes Triton reserves
(NVRTC kernels: static shared bytes only). `#F` lines flag num_stages > 1 with no cp.async (and a dynamic loop to pipeline): pipelining silently inert.

For files with a CUDA source compiled by NVRTC (_SMALLM_DEVICE_SRC [+ _LARGEM_EXTRA_SRC]) the device source is compiled for sm_90
and the same counters are printed per template instantiation (config "nvrtc").
"""
import argparse
import ast
import collections
import contextlib
import importlib
import inspect
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

FIELDS = ("regs", "spills", "smem", "ld.global", "ld.global.nc", "cp.async", "cp.async.bulk", "cp.async.bulk.tensor", "mbarrier",
          "ldmatrix", "mma.sync", "wgmma", "cvt", "ex2", "bar.sync", "atom", "st.global")
PTX_PATTERNS = {
    "ld.global": r"\bld\.global\b",
    "ld.global.nc": r"\bld\.global\.nc\b",
    "cp.async": r"\bcp\.async\.(?:ca|cg)\b",
    "cp.async.bulk": r"\bcp\.async\.bulk\b(?!\.tensor)",
    "cp.async.bulk.tensor": r"\bcp\.async\.bulk\.tensor\b",
    "mbarrier": r"\bmbarrier\.",
    "ldmatrix": r"\bldmatrix\.",
    "mma.sync": r"\bmma\.sync\b",
    "wgmma": r"\bwgmma\.mma_async\b",
    "cvt": r"\bcvt\.",
    "ex2": r"\bex2\.",
    "bar.sync": r"\bbar(?:rier)?\.(?:cta\.)?sync\b",
    "atom": r"\b(?:atom|red)\.",
    "st.global": r"\bst\.global\b",
}
PTXAS = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"


# ------------------------------------------------------------------------------------------------ static discovery
def discover(path):
    """AST pass over the file: ({kernel name: line}, {kernel name: [launch lines]}) for every @triton.jit function and every
    `name[grid](...)` launch site."""
    tree = ast.parse(Path(path).read_text())
    kernels, launches = {}, collections.defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                dn = ast.unparse(d.func if isinstance(d, ast.Call) else d)
                if dn in ("triton.jit", "jit"):
                    kernels[node.name] = node.lineno
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            v = node.func.value
            name = v.id if isinstance(v, ast.Name) else (v.attr if isinstance(v, ast.Attribute) else None)
            if name in kernels:
                launches[name].append(node.lineno)
    return kernels, dict(launches)


def has_dynamic_loop(jf, mod, _seen=None):
    """True when the kernel (or a @triton.jit helper it calls) has a `range(...)` / `tl.range(...)` loop (Triton pipelines those; static_range is unrolled)."""
    _seen = _seen if _seen is not None else set()
    if jf in _seen:
        return False
    _seen.add(jf)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(jf.fn)))
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            f = node.iter.func
            nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if nm == "range":
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            callee = getattr(mod, node.func.id, None) or jf.fn.__globals__.get(node.func.id)
            if callee is not None and callee.__class__.__name__ == "JITFunction" and has_dynamic_loop(callee, mod, _seen):
                return True
    return False


# ------------------------------------------------------------------------------------------------ launch capture
class Capture:
    """Context manager: patches JITFunction.run (record, do not launch) and torch.cuda.get_device_capability ((9, 0))."""

    def __init__(self, capability=(9, 0)):
        self.records = []          # dicts, see _run
        self.errors = []           # non-fatal host errors after the recorded launches (e.g. cuBLAS fallbacks on the dev card)
        self.note = ""
        self.case = ""
        self.capability = capability

    def __enter__(self):
        import torch
        from triton.runtime import jit as tj
        self._tj, self._torch = tj, torch
        self._orig_run = tj.JITFunction.run
        self._orig_cap = torch.cuda.get_device_capability
        cap = self.capability
        torch.cuda.get_device_capability = lambda device=None: cap
        cap_self = self

        def run(self, *args, grid, warmup, **kwargs):
            return cap_self._run(self, args, grid, kwargs)

        tj.JITFunction.run = run
        return self

    def __exit__(self, *exc):
        self._tj.JITFunction.run = self._orig_run
        self._torch.cuda.get_device_capability = self._orig_cap
        return False

    def _run(self, jf, args, grid, kwargs):
        from triton import knobs
        from triton.runtime import driver
        kwargs = dict(kwargs)
        kwargs["debug"] = kwargs.get("debug", jf.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
        device = driver.active.get_current_device()
        _, _, _target, backend, binder = jf.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)
        options, signature, constexprs, attrs = jf._pack_args(backend, kwargs, bound_args, specialization, options)
        sigkeys = [p.name for p in jf.params]
        opts = {k: v for k, v in kwargs.items() if k not in sigkeys and k not in ("debug", "instrumentation_mode")}
        names = {}
        for path, val in constexprs.items():
            names[sigkeys[path[0]] if len(path) == 1 else str(path)] = val
        g = grid(bound_args) if callable(grid) else grid
        self.records.append(dict(jf=jf, name=jf.fn.__name__, signature=signature, constexprs=constexprs, cnames=names, attrs=attrs,
                                 options=opts, grid=tuple(g), case=self.case, note=self.note))
        return None


def _hashable(v):
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


def record_key(r):
    return (r["name"], tuple(sorted((k, _hashable(v)) for k, v in r["cnames"].items())), tuple(r["signature"].values()),
            repr(sorted(r["attrs"].items(), key=repr)), tuple(sorted((k, _hashable(v)) for k, v in r["options"].items())))


# ------------------------------------------------------------------------------------------------ compile + count
def count_ptx(ptx):
    body = re.sub(r"//[^\n]*", "", ptx)
    counts = {k: len(re.findall(p, body)) for k, p in PTX_PATTERNS.items()}
    wg = collections.Counter(int(n) for n in re.findall(r"\bcp\.async\.wait_group\s+(\d+)", body))
    counts["wait_groups"] = ",".join(f"{n}x{c}" for n, c in sorted(wg.items())) or None   # "N x count": wait_group N leaves N copy groups in flight
    return counts


def ptxas_usage(ptx, arch="sm_90a"):
    """(regs, spill bytes, static smem) from `ptxas -v` on the PTX, or (None, None, None)."""
    if not PTXAS.exists():
        return None, None, None
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.ptx"
        p.write_text(ptx)
        r = subprocess.run([str(PTXAS), "-v", f"--gpu-name={arch}", "-o", os.devnull, str(p)], capture_output=True, text=True)
    return parse_ptxas_log(r.stderr + r.stdout)


def parse_ptxas_log(log):
    m = re.search(r"Used (\d+) registers", log)
    regs = int(m.group(1)) if m else None
    spills = None
    m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    if m:
        spills = int(m.group(1)) + int(m.group(2))
    m = re.search(r"(\d+) bytes smem", log)
    return regs, spills, int(m.group(1)) if m else 0


def compile_record(r, align="real"):
    """align "real": the 16-byte divisibility attributes Triton derives from the actual arguments (what the judge runs).
    align "none": no attributes at all (what tools/ptx_probe_gemv.py compiled): every pointer / stride unknown, no vectorisation, no cp.async."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    attrs = r["attrs"] if align == "real" else {k: [] for k in r["attrs"]}
    src = ASTSource(r["jf"], r["signature"], r["constexprs"], attrs)
    c = triton.compile(src, target=GPUTarget("cuda", 90, 32), options=r["options"])
    ptx = c.asm["ptx"]
    counts = count_ptx(ptx)
    regs, spills, _ = ptxas_usage(ptx)
    counts.update(regs=regs, spills=spills, smem=int(c.metadata.shared))
    counts["warps"] = int(c.metadata.num_warps)
    counts["stages"] = int(r["options"].get("num_stages", 3))
    return counts


def fmt_val(v):
    s = str(v)
    return s.replace("triton.language.", "").replace("tl.", "").replace(" ", "")


def line(kernel, cfg, m):
    def f(k):
        return "n/a" if m.get(k) is None else m[k]
    return f"#C {kernel} {cfg} " + " ".join(f"{k}={f(k)}" for k in FIELDS)


# ------------------------------------------------------------------------------------------------ drivers (judged shapes)
def _patch_reference(ref):
    """Cache contents never matter to the audit: torch.empty instead of the reference's randn fill."""
    import torch
    fp8 = torch.float8_e4m3fn

    def fill(gen, shape, device, *a, **k):
        return torch.empty(shape, dtype=fp8, device=device), torch.empty(shape, dtype=fp8, device=device)

    if hasattr(ref, "_fill_cache"):
        ref._fill_cache = fill
    if hasattr(ref, "_coherent_tail"):
        ref._coherent_tail = lambda *a, **k: None


def _cases(problem, which, flt):
    from tools import local_check as lc
    cases = lc.select_cases(problem, which)
    if flt:
        cases = [c for c in cases if re.search(flt, lc.format_case(c))]
    return cases


def drive_generic(L, cap, cases, device="cuda"):
    """attention_decode / bf16_proj: ref.generate_input -> (prepare) -> kernel_fn, with the launches recorded."""
    import torch
    from tools import local_check as lc
    ref, cand = L.ref, L.cand
    _patch_reference(ref)
    weights = state = None
    if L.has_weights:
        weights = ref.generate_weights(ref.WEIGHTS_SEED, device=device)
        state = cand.prepare(weights) if hasattr(cand, "prepare") else weights
    for kw in cases:
        cap.case = lc.format_case(kw)
        data = ref.generate_input(**{**kw, "seed": 0, "device": device})
        try:
            (cand.kernel_fn(data, state) if L.has_weights else cand.kernel_fn(data))
        except Exception as e:  # recorded launches stay; the non-Triton remainder may not run on the dev card
            cap.errors.append(f"{cap.case}: {type(e).__name__}: {str(e)[:120]}")
        del data
        torch.cuda.empty_cache()


def drive_moe(L, cap, cases, device="cuda"):
    """muse_moe_w4a16: the Triton parts of the judge path. M <= 256 run the small-M / large-M CUDA kernels (NVRTC audit); their
    epilogue is Triton below 32 tokens (SGLang's fused kernel above); the local-id kernel is launched on the Marlin fallback path."""
    import torch
    cand = L.cand
    for kw in cases:
        m = kw["bs"]
        cap.case = f"bs {m}"
        conf = cand.config_for(m)
        ids = torch.randint(0, 256, (m, 8), dtype=torch.int32, device=device)
        scores = torch.rand((m, 8), dtype=torch.float32, device=device)
        y = torch.empty((m, 8, 4096), dtype=torch.bfloat16, device=device)
        norm_w = torch.zeros(4096, dtype=torch.bfloat16, device=device)
        cap.note = "Marlin fallback path (judged shapes <= 256 tokens take the CUDA small-M / large-M path)"
        local = cand.local_ids_triton(ids, 0, 32)
        cap.note = ""
        if conf["epilogue"] != "sg":
            cand.epilogue_triton(y, scores, local, norm_w, block_h=conf["block_h"] or cand.EPILOGUE_BLOCK_H, warps=conf["warps"] or cand.EPILOGUE_WARPS)


def drive_decode_layer(L, cap, cases, device="cuda"):
    """muse_decode_layer: the Triton stages of _kernel_sglang for the judged shapes (router / pooler GEMV, attention + reduce,
    local ids, epilogue); the cuBLAS / Marlin / FA3 / NVRTC parts are not Triton. S2 (fused routing) is audited as dormant."""
    import torch
    ref, mod = L.ref, L.cand
    _patch_reference(ref)
    hid = ref.HIDDEN
    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    state = {"router_hi": torch.empty((256, hid), dtype=bf, device=device), "router_lo": torch.empty((256, hid), dtype=bf, device=device),
             "router_bias": torch.zeros(256, dtype=f32, device=device), "dense": {"pooler": torch.empty((4096, hid), dtype=bf, device=device)},
             "post_expert_w": torch.zeros(hid, dtype=bf, device=device), "f2_done": torch.zeros(64, dtype=i32, device=device)}
    done_t = set()
    from tools import local_check as lc
    for kw in cases:
        cap.case = lc.format_case(kw)
        data = ref.generate_input(**{**kw, "seed": 0, "device": device})
        cfg = data["config"]
        t = cfg["tokens"]
        # attention (+ reduce): the Triton path whenever the geometry is covered
        if mod._attention_path(cfg, True) == "triton":
            qkv = torch.empty((t, ref.QKV_SIZE), dtype=bf, device=device)
            mod._attention_decode({**data, "qkv": qkv})
        del data
        torch.cuda.empty_cache()
        if t in done_t:
            continue
        done_t.add(t)
        cap.case = f"tokens {t}"
        triton_proj = True
        rpath, ppath = mod._router_path(t, triton_proj), mod._pooler_path(t, triton_proj)
        x = torch.empty((t, hid), dtype=bf, device=device)
        rin = torch.empty((t, hid), dtype=f32, device=device) if t <= mod.ROUTER_FP32_MAX_TOKENS else None
        mod._pooler(x, state, ppath)
        mod._router_logits(x, rin, state, rpath)
        ids = torch.randint(0, 256, (t, 8), dtype=i32, device=device)
        scores = torch.rand((t, 8), dtype=f32, device=device)
        y = torch.empty((t, 8, hid), dtype=bf, device=device)
        start = torch.zeros(1, dtype=i32, device=device)
        if t > mod.SM_MAX_M:        # Marlin path computes the local ids with our kernel; the small-M route kernel (CUDA) does it below
            local = mod.local_ids_triton(ids, start)
        else:
            local = torch.randint(-1, 32, (t, 8), dtype=i32, device=device)
        if mod._epilogue_path(t) == "triton":
            block_h = mod._marlin_knobs(t)[2]
            mod.epilogue_triton(y, scores, local, state["post_expert_w"], block_h=block_h)
        if hasattr(mod, "pre_norm_triton"):      # C25 G2: the Triton pre-expert norm
            mod.pre_norm_triton(torch.empty((t, 4096), dtype=bf, device=device), state["post_expert_w"][:4096])
        if hasattr(mod, "unpool_sliced") and t <= mod.SLICED_UNPOOL_MAX_TOKENS:      # C25 G3: the sliced unpooler GEMV
            ust = {"dense": {"unpooler": torch.empty((8192, 4096), dtype=bf, device=device)},
                   "unpool_part": torch.empty((mod.UNPOOL_SPLIT_MAX, 16, 1024), dtype=f32, device=device),
                   "unpool_done": torch.zeros(64, dtype=i32, device=device)}
            mod.unpool_sliced(torch.empty((t, 4096), dtype=bf, device=device), ust)
        # dormant: S2 fused routing (S2_FUSED_ROUTING = False in this file)
        if 1 <= t <= mod.F2_MAX_TOKENS and not mod.S2_FUSED_ROUTING:
            cap.note = f"dormant: S2_FUSED_ROUTING={mod.S2_FUSED_ROUTING}"
            old = mod.S2_FUSED_ROUTING
            try:
                mod.fused_router_f2(rin if rin is not None else x, state["router_hi"], state["router_lo"], state["router_bias"], start, state["f2_done"])
            finally:
                mod.S2_FUSED_ROUTING = old
                cap.note = ""


DRIVERS = {"muse_attention_decode": drive_generic, "muse_bf16_proj": drive_generic, "muse_moe_w4a16": drive_moe,
           "muse_decode_layer": drive_decode_layer}


def moe_cases(which, flt):
    from tools import local_check as lc
    cases = lc.select_cases("muse_moe_w4a16", which)
    return [c for c in cases if not flt or re.search(flt, lc.format_case(c))]


# ------------------------------------------------------------------------------------------------ NVRTC (CUDA device source)
def nvrtc_audit(mod, out=print):
    """Compile mod._SMALLM_DEVICE_SRC (+ _LARGEM_EXTRA_SRC) for sm_90: PTX counts + ptxas -v (via NVRTC's log) per template instantiation."""
    rows = []
    if not hasattr(mod, "_SMALLM_DEVICE_SRC"):
        return rows
    try:
        import nvrtc_smallm as nv
    except Exception as e:
        out(f"#N nvrtc unavailable: {e}")
        return rows
    if not nv.HAVE_NVRTC:
        out("#N cuda-python / NVRTC not available")
        return rows
    jobs = [(mod._SMALLM_DEVICE_SRC, nv.KERNEL_NAMES)]
    if hasattr(mod, "_LARGEM_EXTRA_SRC"):
        jobs.append((mod._SMALLM_DEVICE_SRC + mod._LARGEM_EXTRA_SRC, nv.LARGE_NAMES))
    seen = set()
    for src, names in jobs:
        names = [n for n in names if n not in seen]
        try:
            ptx, lowered = nv.compile_device(src, "compute_90", want="ptx", names=names)
            usage = _cubin_usage(nv, src, names)
        except Exception as e:
            out(f"#N nvrtc compile failed: {str(e)[:300]}")
            continue
        ptx = ptx.decode(errors="replace") if isinstance(ptx, bytes) else ptx
        blocks = re.split(r"(?m)(?=^\.(?:visible|weak) \.entry )", ptx)
        by_entry = {re.match(r"\.(?:visible|weak) \.entry (\S+?)\(", b).group(1): b for b in blocks if re.match(r"\.(?:visible|weak) \.entry ", b)}
        for nm in names:
            seen.add(nm)
            low = lowered[nm].rstrip("\x00")
            blk = by_entry.get(low)
            if blk is None:
                continue
            counts = count_ptx(blk)
            regs, spills, smem = usage.get(low, (None, None, None))
            counts.update(regs=regs, spills=spills, smem=smem, warps=None, stages=None)
            rows.append(dict(kernel=nm, cfg="nvrtc", counts=counts, note="CUDA (NVRTC, sm_90)"))
    return rows


CUOBJDUMP = PTXAS.parent / "cuobjdump"


def _cubin_usage(nv, src, names):
    """{lowered name: (regs, local bytes, static smem)} from cuobjdump --dump-resource-usage on the NVRTC sm_90 cubin
    (local = spills + local arrays; ptxas -v is not reachable through NVRTC's log)."""
    cubin, _ = nv.compile_device(src, "sm_90", want="cubin", names=names)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.cubin"
        p.write_bytes(cubin)
        r = subprocess.run([str(CUOBJDUMP), "--dump-resource-usage", str(p)], capture_output=True, text=True)
    usage = {}
    for m in re.finditer(r"Function (\S+?):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", r.stdout):
        usage[m.group(1)] = (int(m.group(2)), int(m.group(5)), int(m.group(4)))
    return usage


# ------------------------------------------------------------------------------------------------ report
def label_configs(entries):
    """Config label per entry: w<warps>s<stages> + the constexprs / options that differ between this kernel's configs."""
    by_kernel = collections.defaultdict(list)
    for e in entries:
        by_kernel[e["kernel"]].append(e)
    for k, es in by_kernel.items():
        keys = sorted({c for e in es for c in e["cvals"]})
        varying = [c for c in keys if len({repr(e["cvals"].get(c)) for e in es}) > 1]
        for e in es:
            extra = ",".join(f"{c}={'rt' if e['cvals'].get(c) is None else fmt_val(e['cvals'][c])}" for c in varying)   # rt: a runtime int, not specialised to a constexpr
            e["cfg"] = f"w{e['counts']['warps']}s{e['counts']['stages']}" + (f",{extra}" if extra else "")
        # same label, different compile: the arguments differ in Triton's 16-divisibility specialisation (e.g. M = 8 vs 16): name the undivisible ones
        sig = collections.defaultdict(set)
        for e in es:
            sig[e["cfg"]].add(tuple(e["counts"].get(f) for f in FIELDS))
        base = {id(e): e["cfg"] for e in es}
        for e in es:
            if len(sig[base[id(e)]]) > 1 and e.get("nd"):      # only when the variants really compile differently
                e["cfg"] += ",nodiv16=" + "+".join(e["nd"])


def merge(entries):
    """Collapse entries with identical kernel, label and metrics (they differ only in runtime-int specialisation)."""
    out = {}
    for e in entries:
        k = (e["kernel"], e["cfg"], tuple(e["counts"].get(f) for f in FIELDS))
        if k in out:
            out[k]["cases"].update(e["cases"])
            out[k]["notes"].update(e["notes"])
            out[k]["nvariants"] += e.get("nvariants", 1)
        else:
            out[k] = dict(e, cases=set(e["cases"]), notes=set(e["notes"]), nvariants=e.get("nvariants", 1))
    return list(out.values())


def audit(kernel_file, cases="scored", flt=None, nvrtc=True, out=print, case_list=None, align="real"):
    """Returns dict(rows=[...], kernels=..., launches=..., missing=[...], errors=[...])."""
    import torch
    from tools import local_check as lc
    path = Path(kernel_file).resolve()
    first = path.read_text().splitlines()[0]
    m = re.match(r"#\s*okc:\s*problem=(\w+)", first)
    if not m:
        raise ValueError(f"{path}: first line must be '# okc: problem=<problem>'")
    problem = m.group(1)
    if problem not in DRIVERS:
        raise ValueError(f"no audit driver for problem {problem}")
    if not torch.cuda.is_available():
        raise RuntimeError("ptx_audit needs a CUDA device for Triton's driver (any card; nothing is launched)")
    kernels, launches = discover(path)
    L = lc.Loaded(problem, path)
    if case_list is None:
        case_list = moe_cases(cases, flt) if problem == "muse_moe_w4a16" else _cases(problem, cases, flt)
    errors = []
    with torch.no_grad(), Capture() as cap:
        try:
            DRIVERS[problem](L, cap, case_list)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
    errors += ["host (non-fatal): " + e for e in cap.errors[:3]]
    # compile each distinct key once
    groups = collections.OrderedDict()
    for r in cap.records:
        g = groups.setdefault(record_key(r), dict(rec=r, cases=[], notes=set(), n=0))
        g["cases"].append(r["case"])
        if r["note"]:
            g["notes"].add(r["note"])
        g["n"] += 1
    entries = []
    for g in groups.values():
        r = g["rec"]
        try:
            counts = compile_record(r, align)
        except Exception as e:
            errors.append(f"compile {r['name']}: {type(e).__name__}: {str(e)[:300]}")
            continue
        cvals = {k: fmt_val(v) for k, v in r["cnames"].items()}
        cvals.update({k: fmt_val(v) for k, v in r["options"].items() if k not in ("num_warps", "num_stages")})
        nd = [n for i, (n, ty) in enumerate(r["signature"].items())
              if ty != "constexpr" and (ty.startswith("i") or ty.startswith("*")) and not r["attrs"].get((i,))]
        entries.append(dict(kernel=r["name"], nd=nd, counts=counts, cvals=cvals, cases=set(g["cases"]), notes=g["notes"], jf=r["jf"],
                            grids=sorted({r["grid"]}), nvariants=1, allc=cvals))
    label_configs(entries)
    rows = merge(entries)
    for e in rows:
        e["loop"] = has_dynamic_loop(e["jf"], L.cand)
        e["dormant"] = bool(e["notes"]) and all(n.startswith("dormant") for n in e["notes"])
    if nvrtc:
        for nr in nvrtc_audit(L.cand, out=out):
            rows.append(dict(kernel=nr["kernel"], cfg=nr["cfg"], counts=nr["counts"], cvals={}, cases={"-"}, notes={nr["note"]}, jf=None,
                             loop=None, dormant=False, nvariants=1, grids=[]))
    reached = {e["kernel"] for e in rows if e["jf"] is not None}
    missing = [k for k in kernels if k not in reached and k in launches]
    return dict(rows=rows, kernels=kernels, launches=launches, missing=missing, errors=errors, problem=problem, path=path, ncases=len(case_list))


def print_report(res, out=print):
    for e in sorted(res["rows"], key=lambda e: (e["jf"] is None, list(res["kernels"]).index(e["kernel"]) if e["kernel"] in res["kernels"] else 0)):
        out(line(e["kernel"], e["cfg"], e["counts"]))
    for e in res["rows"]:
        c = e["counts"]
        if e["jf"] is not None and c["stages"] > 1 and c["cp.async"] == 0 and c["cp.async.bulk"] == 0 and c["cp.async.bulk.tensor"] == 0 and e["loop"]:
            out(f"#F {e['kernel']} {e['cfg']} num_stages={c['stages']} but cp.async=0: pipelining silently inert (ld.global={c['ld.global']})")
    for e in res["rows"]:
        c = e["counts"]
        if e["jf"] is not None and c["stages"] > 1 and c["cp.async"] > 0 and e["loop"] and not re.search(r"(?:^|,)[1-9]\d*x", c.get("wait_groups") or ""):
            out(f"#F {e['kernel']} {e['cfg']} num_stages={c['stages']}, cp.async={c['cp.async']} but every cp.async.wait_group is 0 ({c.get('wait_groups')}): no copy stays in flight across iterations")
    for k in res["missing"]:
        out(f"#M {k} has a launch site (lines {res['launches'][k]}) but no judged shape reached it")
    for k in res["kernels"]:
        if k not in res["launches"] and not any(e["kernel"] == k for e in res["rows"]):
            out(f"#H {k}: @triton.jit helper (called from other kernels, never launched)")
    for e in res["errors"]:
        out("#E " + e.splitlines()[0])


def md_section(res, rel):
    rows = sorted(res["rows"], key=lambda e: (e["jf"] is None, list(res["kernels"]).index(e["kernel"]) if e["kernel"] in res["kernels"] else 0, e["cfg"]))
    L = [f"<!-- audit:begin {rel} -->", f"### `{rel}` ({res['problem']}, {res['ncases']} judged shapes)", ""]
    L.append("| kernel | config | w | st | loop | regs | spl B | smem B | ld.global | .nc | cp.async | bulk | bulk.tensor | mbar | ldmatrix | mma.sync | wgmma | cvt | ex2 | bar.sync | atom | st.global | notes |")
    L.append("|" + "---|" * 23)
    for e in rows:
        c = e["counts"]
        note = "; ".join(sorted(e["notes"]))
        if e["jf"] is not None:
            shapes = sorted(e["cases"])
            short = re.sub(r",? ?(ctx 80000|heads tp8|page 1|dist \w+|spread agentic)", "", shapes[0])
            note = (note + "; " if note else "") + f"{len(shapes)} shape(s), e.g. {short}"
            if c["stages"] > 1 and not e["loop"]:
                note = "no dynamic loop: num_stages irrelevant; " + note
        if c.get("wait_groups"):
            note = (note + "; " if note else "") + f"cp.async.wait_group (N x count): {c['wait_groups']}"
        flag = ""
        if e["jf"] is not None and c["stages"] > 1 and c["cp.async"] == 0 and c["cp.async.bulk"] == 0 and e["loop"]:
            flag = " **INERT**"
        vals = [c.get(f) if c.get(f) is not None else "n/a" for f in FIELDS]
        L.append(f"| `{e['kernel']}` | `{e['cfg']}`{flag} | {c['warps'] if c['warps'] else '-'} | {c['stages'] if c['stages'] else '-'} | "
                 f"{'y' if e['loop'] else ('n' if e['loop'] is False else '-')} | " + " | ".join(str(v) for v in vals) + f" | {note} |")
    L.append("")
    if res["missing"]:
        L.append("Launch sites no judged shape reaches: " + ", ".join(f"`{k}`" for k in res["missing"]) + ".")
    if res["errors"]:
        L.append("Errors: " + " / ".join(e.splitlines()[0] for e in res["errors"]))
    L.append("<!-- audit:end -->")
    return "\n".join(L) + "\n"


def write_md(md_path, rel, section):
    p = Path(md_path)
    text = p.read_text() if p.exists() else ""
    pat = re.compile(rf"<!-- audit:begin {re.escape(rel)} -->.*?<!-- audit:end -->\n?", re.S)
    if pat.search(text):
        text = pat.sub(lambda _m: section, text)
    else:
        text = text.rstrip("\n") + "\n\n" + section if text else section
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kernel_file")
    ap.add_argument("--configs", default="auto", choices=["auto"], help="auto: launch configs recorded from the file's own host code on the judged shapes")
    ap.add_argument("--cases", default="scored", choices=["scored", "tests", "all"])
    ap.add_argument("--filter", default=None, help="regex over the case label (e.g. 'layer router')")
    ap.add_argument("--md", default=None, help="markdown file to update (a per-file section between audit:begin/end markers)")
    ap.add_argument("--no-nvrtc", action="store_true")
    ap.add_argument("--align", default="real", choices=["real", "none"],
                    help="real: Triton's divisibility attributes of the actual arguments (default, = the judge); none: no attributes (the old ptx_probe_gemv.py compile)")
    a = ap.parse_args(argv)
    res = audit(a.kernel_file, a.cases, a.filter, nvrtc=not a.no_nvrtc, align=a.align)
    print_report(res)
    if a.md:
        rel = str(Path(a.kernel_file).resolve().relative_to(ROOT))
        write_md(a.md, rel, md_section(res, rel))
    return 0 if res["rows"] and not res["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
