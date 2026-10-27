#!/usr/bin/env python3
"""Regenerate kernels/muse_decode_layer/sonnet_v5h.py (C40) = sonnet_v5f.py + F1 (the FFN boundary folded into the unpooler kernel, T <= 8, S10_FFN_FUSE) + the C33 TOOLS_LIGHT rider
(S11_TOOLS_LIGHT: PDL + <= 64 KB per CTA L2 prefetch before gdc_wait on our weight-streaming Triton kernels). Everything is added to v5f's text by exact-match substitutions that fail
loudly when v5f changes; the twin kernels come from tools/gen_measure_corridor_v1.py's twins() (renamed _mc_ -> _tl_).
Run after any change to an input:   .venv/bin/python tools/gen_sonnet_v5h.py"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
SRC = ROOT / "kernels" / "muse_decode_layer" / "sonnet_v5f.py"
OUT = ROOT / "kernels" / "muse_decode_layer" / "sonnet_v5h.py"

HEADER = '''# okc: problem=muse_decode_layer
# sonnet_v5h (C40) -- sonnet_v5f (board entry 1.433x) plus three switchable changes (module-level switches, shapes / hardware only; each off = sonnet_v5f's code path, bit for bit): sonnet_v5g with F1 rebuilt WITHOUT
# the grid-wide barrier. Judge on v5g (check 4910822d6ee544e9): #0 107.6 -> 101.4 us, #1 113.9 -> 112.5, v1 sliding 164.5 -> 161.5, v1 full unchanged; in profile 0 the fused kernel ran 10.0 us = v5f's unpool (6.2)
# + the boundary (3.4-4.4): the launch was saved, the wait of 64 tile CTAs on each other ate the rest.
#   F1 (S10_FFN_FUSE)  T <= SLICED_UNPOOL_MAX_TOKENS (8): the FFN boundary (SGLang's muse_spark_norm_residual: post-FFN norm + residual gate + the next layer's pre-attention norm) behind the unpooler kernel. At T <= 8
#       f is zero beyond column 1023 (TASK.md), so rmsnorm(f * (1 + w)) = f_live (1 + w) g with g = rsqrt(sum(f_live^2 (1 + w)^2) / 8192 + 1e-8); the residual update s' = alpha s + beta g f (1 + w) and the next norm
#       h = rmsnorm(s') (1 + w_next) need only sums: sum(s'^2) = S1 + 2 g S2 + g^2 S3 with S1 = sum((alpha s)^2) over all 8192 columns, S2 = sum(alpha beta s f w), S3 = sum((beta f w)^2) over the live columns.
#       NO CTA WAITS FOR ANOTHER. _unpool_ffn_kernel (the GEMV of the unpooler, same grid): before its gdc_wait / K loop each CTA loads what does not come from the previous grid (s, alpha, beta, the gain, its 16 live + 112
#       pad columns of s) and computes S1 of its columns; the last CTA of a column tile keeps its FP32 f columns in a scratch and publishes C / S1 / S2 / S3 of its columns per token; the LAST tile-last CTA (a global ticket on a
#       counter zeroed inside the call by a memset on the side stream) sums the 64 partials in tile order and writes two scalars per token (g, 1 / rms). _ffn_apply_kernel (grid 16 x tokens, launched with PDL while the tails
#       run, its s / alpha / beta / gain / w_next loads issued BEFORE its gdc_wait) then applies them: s' (FP32) and h (BF16) of 512 columns. Chosen over a single final CTA doing all 8192 x T columns: that CTA would move
#       ~100 KB x T through one SM (1.5 us at T = 1, 5 us at T = 8 > the 3.9 us it replaces), the apply kernel is parallel over columns and tokens and costs ~1 us after the unpooler. f is never rounded to BF16.
#       Numerics: the FP32 chain of reference.py (ref_back) with ONE BF16 rounding (h). tests print the max rel error vs the FP32 reference chain and vs the board's BF16-f path.
#   S11_TOOLS_LIGHT  the C33 TOOLS_LIGHT variant as a rider (judge job fcbc85a47127453a: +2.0 us at T = 1, TOOLS_FULL -3.9 us = corridor congestion): PDL on for every Triton kernel of the layer (S3's gdc_wait /
#       launch_dependents) and, for _gemv_splitk, _router_fix_kernel, _unpool_kernel (and F1's), _pre_norm_kernel and _epilogue_kernel, the _tl_ twins: prefetch.global.L2 of the first <= 64 KB of the CTA's weight slice
#       BEFORE gdc_wait. Math unchanged (bitwise v5f). Inert before sm_90.
#   S12_ROUTE_SIDE  (C37 ROUTE_SIDE, judge job f31c983903fd4930: +1.1 / +1.2 us at T = 1, +0.6 at T = 16): the small-M route kernel runs on the SIDE stream right after the routing, behind the pooler. Bitwise v5f.
# PREDICTION: relative to v5g's own judge check (the two measured shapes) and v5f's per-shape table (job 62a4b559a7f14ddc, whose load formula reproduces 1.4300): the fused kernel was 10.0 us at T = 1; v5h's unpooler
#     kernel ~6.9 us (v5f's 6.2 + ~0.7 for the partials and the ticket tail) + the apply kernel ~1.2 us mostly hidden behind the PDL launch: ~8 us, i.e. -2 us (range -1 .. -3) on the four T <= 8 shapes against v5g
#     (v5g's barrier version gave -0.0 .. -2 against v5f): #0 101.4 -> ~99.6   #1 112.5 -> ~110.7   #6 161.5 -> ~159.7   #7 ~177 -> ~175 (v5g: unchanged); all other shapes unchanged.
#   PROBLEM: v5g check 1.439 -> ~1.446 (range 1.443 - 1.449); against the 1.433 board ~ +0.013.
#   Kill: F1 does not meet the tolerance (vs the FP32 reference chain hidden 1.7e-3 / stream 5e-7, vs the board's BF16-f chain stream 9e-4 / hidden 1.9e-3; the layer bound is >= 3 %), or the check shows the T <= 8 shapes
#   slower than v5g; switch order: F1 (S10_FFN_FUSE) first, then S12_ROUTE_SIDE, then S11_TOOLS_LIGHT.
#   Risks: the apply kernel's launch after the unpooler is a second launch (~1 us if PDL does not overlap it; the board's boundary was 3.9 us incl. its gap); the three gains were measured separately and may not add up;
#   the Triton compile of the fused kernels on the judge is covered by prepare()'s self-check G5 (reference chain + two identical runs + the counter: it switches F1 off on any failure).
# --- the header below is sonnet_v5f's ---
'''

SWITCHES = '''S11_EPI_GRID = True       # G4 (H9): T <= EPI_GRID_MAX_TOKENS: the MoE epilogue on grid (tokens, EPI_PARTS), one 1024-column quarter per CTA (bitwise v5b's epilogue)
S10_FFN_FUSE = True       # C38 F1: T <= SLICED_UNPOOL_MAX_TOKENS: the FFN boundary (post-FFN norm + residual + next pre-attention norm) folded into the unpooler kernel; off = v5f's SGLang boundary
S12_ROUTE_SIDE = True      # C38 rider (C37): smallm_route on the side stream right after the routing (T <= 16, small-M experts live); off = v5f (route on the main stream, in the experts)
S11_TOOLS_LIGHT = True    # C38 rider (C33 TOOLS_LIGHT): PDL on our Triton kernels + a <= 64 KB per CTA L2 prefetch of their weights before gdc_wait; off = v5f (S3_PDL decides PDL)
'''

TOOLS_LIGHT_BYTES = 65536

SHIM = '''
# =============================================================================================== C38 rider: the _tl_ twins behind the original names
_TL_LIGHT_BYTES = 65536   # TOOLS_LIGHT: the first <= 64 KB per CTA (C33: the whole slice, TOOLS_FULL, was 3.9 us SLOWER at T = 1: corridor congestion)


@functools.lru_cache(maxsize=None)
def _tools_hw(index):
    """Hopper-or-newer with Triton's PDL API (griddepcontrol is sm_90+); hardware only."""
    return bool(HAVE_PDL_API and torch.cuda.is_available() and torch.cuda.get_device_capability(index)[0] >= 9)


def _tools_live():
    return bool(S11_TOOLS_LIGHT and _tools_hw(torch.cuda.current_device() if torch.cuda.is_available() else 0))


def _tl_pfl(name, a, kw):
    """Prefetch lines (128 bytes, a power of two) per weight row of one launch: the first <= _TL_LIGHT_BYTES of the CTA's slice."""
    def cdiv(x, y):
        return -(-x // y)
    if name == "_gemv_splitk":
        bn, bk, split, k = kw["BN"], kw["BK"], kw["SPLIT"], a[6]
        rows, rowl, nmat = bn, cdiv(cdiv(k, split * bk) * bk * 2, 128), 2 if kw["HAS_LO"] else 1
    elif name == "_router_fix_kernel":
        rows, rowl, nmat = kw["BN"], cdiv(kw["K"] // kw["SPLIT"] * 2, 128), 2
    elif name == "_unpool_kernel":
        rows, rowl, nmat = kw["BN"], cdiv(kw["K"] // kw["SPLIT"] * 2, 128), 1
    elif name == "_pre_norm_kernel":
        rows, rowl, nmat = 1, cdiv(kw["N"] * 2, 128), 1
    else:   # _epilogue_kernel: the gain vector
        rows, rowl, nmat = 1, cdiv(kw["H"] * 2, 128), 1
    return triton.next_power_of_2(max(1, min(rowl, max(1, _TL_LIGHT_BYTES // (rows * 128 * nmat)))))


class _ToolsShim:
    """Stands in for one of our Triton kernels: with S11_TOOLS_LIGHT live (Hopper) the launch goes to its _tl_ twin with PDL and the prefetch depth, else to the original unchanged."""

    def __init__(self, name, orig, twin):
        self.name, self.orig, self.twin = name, orig, twin

    def __getitem__(self, grid):
        orig, twin, name = self.orig[grid], self.twin[grid], self.name

        def launch(*a, **kw):
            if not _tools_live():
                return orig(*a, **kw)
            kw = dict(kw)
            kw.update(PDL=True, launch_pdl=True, PFL=_tl_pfl(name, a, kw))
            return twin(*a, **kw)
        return launch


_gemv_splitk = _ToolsShim("_gemv_splitk", _gemv_splitk, _tl_gemv_splitk)
_router_fix_kernel = _ToolsShim("_router_fix_kernel", _router_fix_kernel, _tl_router_fix_kernel)
_unpool_kernel = _ToolsShim("_unpool_kernel", _unpool_kernel, _tl_unpool_kernel)
_pre_norm_kernel = _ToolsShim("_pre_norm_kernel", _pre_norm_kernel, _tl_pre_norm_kernel)
_epilogue_kernel = _ToolsShim("_epilogue_kernel", _epilogue_kernel, _tl_epilogue_kernel)
'''

F1 = (ROOT / "tools" / "sonnet_v5h_f1.txt").read_text()

SELFCHECK = '''    try:
        if S10_FFN_FUSE and _unpool_live(8, dev) and "f1" in state:   # G5 (F1): the fused boundary against the FP32 reference chain (reference.py ref_back) and against itself (deterministic)
            y = (0.5 * torch.randn((8, D_EXPERT), generator=g, device=dev)).to(torch.bfloat16)
            s = torch.randn((8, HIDDEN), generator=g, device=dev)
            f1 = state["f1"]
            f1["ctr"].zero_()
            h1, s1 = unpool_ffn(y, s, state)
            f1["ctr"].zero_()
            h2, s2 = unpool_ffn(y, s, state)
            f = F.pad((y.float() @ state["dense"]["unpooler"][:UNPOOL_SLICE].float().T), (0, HIDDEN - UNPOOL_SLICE)) * f1["wf"]
            sn = f1["alpha"] * s + f1["beta"] * _rms(f, POST_NORM_EPS)
            hn = _rms(sn) * f1["wn"]
            torch.cuda.synchronize()
            e_s = float((s1 - sn).norm() / sn.norm())
            e_h = float((h1.float() - hn).norm() / hn.norm())
            ok = e_s < 1e-3 and e_h < 1e-2 and bool(torch.equal(h1, h2)) and bool(torch.equal(s1, s2)) and int(f1["ctr"]) == UNPOOL_SLICE // _unpool_cfg(state)[0] and not int(state["unpool_done"].abs().sum())
            print(f"okc-diag G5 FFN fusion rel L2: stream {e_s:.2e} hidden {e_h:.2e} deterministic={bool(torch.equal(h1, h2))} ok={ok}")
            S10_FFN_FUSE = ok
    except Exception as ex:  # noqa: BLE001
        print(f"okc-diag G5 selfcheck raised {type(ex).__name__}: {str(ex)[:200]}")
        S10_FFN_FUSE = False
    try:
        if S12_ROUTE_SIDE and state.get("smallm") is not None:   # G6 (ROUTE_SIDE): the experts with the route done up front are bitwise the experts that route themselves
            x, w, ids = _smallm_selfcheck_inputs(8, 3008, dev)
            start = torch.zeros(1, dtype=torch.int32, device=dev)
            a = _experts_smallm(x, w, ids, start, state)
            b = _experts_smallm(x, w, ids, start, state, pre=smallm_route_side(ids, start, state))
            torch.cuda.synchronize()
            same = bool(torch.equal(a, b))
            print(f"okc-diag G6 route-side bitwise={same}")
            S12_ROUTE_SIDE = same
    except Exception as ex:  # noqa: BLE001
        print(f"okc-diag G6 selfcheck raised {type(ex).__name__}: {str(ex)[:200]}")
        S12_ROUTE_SIDE = False
    print(f"okc-diag glue switches: G2={S9_PRE_NORM} G3={S10_UNPOOL_GEMV} G4={S11_EPI_GRID} G5={S10_FFN_FUSE} G6={S12_ROUTE_SIDE} rider={S11_TOOLS_LIGHT}")
'''


def _sub(text, old, new, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"gen_sonnet_v5h: expected {count} x {old[:80]!r} in sonnet_v5f.py, found {n}: v5f changed, update the substitutions")
    return text.replace(old, new)


def twins_block():
    import gen_measure_corridor_v1 as g
    t = g.twins()
    assert "_mc_" in t
    t = t.replace("_mc_", "_tl_")
    assert "_mc" not in t
    return "\n\n# =============================================================================================== C38 rider: twins of our weight-streaming Triton kernels (prefetch before gdc_wait)\n" + t + "\n"


def build():
    s = SRC.read_text()
    assert s.startswith("# okc: problem=muse_decode_layer\n")
    s = HEADER + s[len("# okc: problem=muse_decode_layer\n"):]
    # switches + PDL gate
    s = _sub(s, "S11_EPI_GRID = True       # G4 (H9): T <= EPI_GRID_MAX_TOKENS: the MoE epilogue on grid (tokens, EPI_PARTS), one 1024-column quarter per CTA (bitwise v5b's epilogue)\n", SWITCHES)
    s = _sub(s, "    return bool(S3_PDL and HAVE_PDL_API and device.type == \"cuda\" and torch.cuda.get_device_capability(device)[0] >= 9)",
             "    return bool((S3_PDL or S11_TOOLS_LIGHT) and HAVE_PDL_API and device.type == \"cuda\" and torch.cuda.get_device_capability(device)[0] >= 9)")
    # state: the F1 vectors and scratch
    s = _sub(s, "    del freqs\n    torch.cuda.synchronize()\n    return state\n",
             "    state[\"f1\"] = _f1_state(weights, device)   # C38 F1\n    del freqs\n    torch.cuda.synchronize()\n    return state\n")
    # self-check
    s = _sub(s, "    global S9_PRE_NORM, S10_UNPOOL_GEMV, S11_EPI_GRID\n", "    global S9_PRE_NORM, S10_UNPOOL_GEMV, S11_EPI_GRID, S10_FFN_FUSE, S12_ROUTE_SIDE\n")
    s = _sub(s, "    print(f\"okc-diag glue switches: G2={S9_PRE_NORM} G3={S10_UNPOOL_GEMV} G4={S11_EPI_GRID}\")\n", SELFCHECK)
    # twins + shim + F1 kernel, before the entry points
    marker = "def prepare(weights):\n"
    i = s.index(marker)
    assert s.count(marker) == 1
    s = s[:i] + twins_block() + SHIM + F1 + "\n\n" + s[i:]
    # kernel_fn: the zeroed arrival counter on the side stream, the fused boundary
    s = _sub(s, "    with torch.cuda.stream(side_stream):\n        if f2 is not None:",
             "    fuse = _ffn_fuse_live(tokens, device, state, stream)\n    with torch.cuda.stream(side_stream):\n        if fuse:\n            state[\"f1\"][\"ctr\"].zero_()  # F1's arrival counter, zeroed inside the timed call (side stream: hidden behind the pooler)\n        if f2 is not None:")
    old_tail = ("    if _unpool_live(tokens, device) and \"unpool_part\" in state:  # G3: slice GEMV + zero pad in one launch\n"
                "        ffn_output = unpool_sliced(expert_output, state)  # all-gather: identity\n")
    new_tail = ("    if fuse:  # C38 F1: slice GEMV + the whole FFN boundary in one launch (T <= 8)\n"
                "        next_attn_input, stream = unpool_ffn(expert_output, stream, state)  # all-gather: identity\n"
                "        return next_attn_input, stream, topk_ids\n"
                "    if _unpool_live(tokens, device) and \"unpool_part\" in state:  # G3: slice GEMV + zero pad in one launch\n"
                "        ffn_output = unpool_sliced(expert_output, state)  # all-gather: identity\n")
    s = _sub(s, old_tail, new_tail)
    # ---- S12 ROUTE_SIDE (C37): smallm_route on the side stream right after the routing
    s = _sub(s, "def smallm_experts(x, topk_ids, sm, start, count, cfg):", "def smallm_experts(x, topk_ids, sm, start, count, cfg, pre=None):")
    s = _sub(s, "    ids = topk_ids.to(torch.int32).contiguous()\n    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)\n    ext.smallm_route(ids, local_ids, sm[\"tab\"], start, count)\n",
             "    if pre is None:\n        ids = topk_ids.to(torch.int32).contiguous()\n        local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)\n"
             "        ext.smallm_route(ids, local_ids, sm[\"tab\"], start, count)\n    else:  # C38 S12: the route table and the local ids were built on the side stream\n        local_ids = pre\n")
    s = _sub(s, "def _experts_smallm(x, topk_weights, topk_ids, local_start, state):", "def _experts_smallm(x, topk_weights, topk_ids, local_start, state, pre=None):")
    s = _sub(s, "    y, local_ids = smallm_experts(x.contiguous(), topk_ids, state[\"smallm\"], local_start, LOCAL_EXPERTS, smallm_cfg(m))\n",
             "    y, local_ids = smallm_experts(x.contiguous(), topk_ids, state[\"smallm\"], local_start, LOCAL_EXPERTS, smallm_cfg(m), pre=pre)\n")
    s = _sub(s, "def _experts(x, topk_weights, topk_ids, local_start, state, local_ids=None):", "def _experts(x, topk_weights, topk_ids, local_start, state, local_ids=None, pre=None):")
    s = _sub(s, "        return _experts_smallm(x, topk_weights, topk_ids, local_start, state)\n    if local_ids is None and _largem_live",
             "        return _experts_smallm(x, topk_weights, topk_ids, local_start, state, pre=pre)\n    if local_ids is None and _largem_live")
    s = _sub(s, "    local_ids = None\n    fuse = _ffn_fuse_live(", "    local_ids = None\n    pre = None   # C38 S12: smallm_route's local ids, built on the side stream\n    fuse = _ffn_fuse_live(")
    s = _sub(s, "            topk_weights, topk_ids = state[\"routing\"](None, logits, TOP_K, True)\n",
             "            topk_weights, topk_ids = state[\"routing\"](None, logits, TOP_K, True)\n"
             "        if S12_ROUTE_SIDE and local_ids is None and _smallm_live(tokens, state):  # C38 S12: smallm_route needs only topk_ids + local_start: behind the pooler, off the main stream\n"
             "            pre = smallm_route_side(topk_ids, data[\"local_start\"], state)\n")
    s = _sub(s, "        for tensor in (topk_weights, topk_ids) + ((local_ids,) if local_ids is not None else ()):\n",
             "        for tensor in (topk_weights, topk_ids) + ((local_ids,) if local_ids is not None else ()) + ((pre,) if pre is not None else ()):\n")
    s = _sub(s, "    expert_output = _experts(expert_input if normed else _pre_expert(expert_input, state), topk_weights, topk_ids, data[\"local_start\"], state,\n                             local_ids=local_ids)\n",
             "    expert_output = _experts(expert_input if normed else _pre_expert(expert_input, state), topk_weights, topk_ids, data[\"local_start\"], state,\n                             local_ids=local_ids, pre=pre)\n")
    # smallm_route_side (placed with the F1 block)
    s = _sub(s, "\n# =============================================================================================== C40 F1: the FFN boundary behind the unpooler kernel, no CTA waits for another\n",
             "\n# =============================================================================================== C38 S12: smallm_route on the side stream\ndef smallm_route_side(topk_ids, start, state):\n"
             "    \"\"\"The small-M route kernel on its own (it needs only topk_ids and local_start): local ids [T, 8] int32; the route table lands in the small-M state's tab.\"\"\"\n"
             "    sm = state[\"smallm\"]\n    ids = topk_ids.to(torch.int32).contiguous()\n    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)\n"
             "    sm[\"ext\"].smallm_route(ids, local_ids, sm[\"tab\"], start, LOCAL_EXPERTS)\n    return local_ids\n"
             "\n\n# =============================================================================================== C40 F1: the FFN boundary behind the unpooler kernel, no CTA waits for another\n")
    return s


if __name__ == "__main__":
    text = build()
    compile(text, str(OUT), "exec")
    OUT.write_text(text)
    print("wrote", OUT, len(text), "bytes")
