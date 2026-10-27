"""Test helper (development card, no nvcc): compile the small-M CUDA device source of kernels/muse_moe_w4a16/sonnet_v3.py with NVRTC (cuda-python)
and expose the same three entry points as the load_inline extension (smallm_gemm, smallm_route, smallm_init), launched through the driver API on
torch's current stream. The sm_75 card has no BF16 tensor cores: the source then takes its shuffle-emulation branch (same fragment layouts).
Also `sass_check`: compile for sm_90 (NVRTC runs ptxas), which validates every inline-asm string and instruction for the judge's GPU."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import torch

try:
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    HAVE_NVRTC = True
except Exception:  # pragma: no cover
    HAVE_NVRTC = False

INCLUDE = None
KERNEL_NAMES = [f"w4_smallm_kernel<{f},{w}>" for f in ("true", "false") for w in (2, 4, 8)] + ["smallm_route_kernel"]
LARGE_NAMES = [f"w4_largem_kernel<{f},{w},{r}>" for f in ("true", "false") for (w, r) in ((4, 1), (8, 1), (4, 2), (8, 2))] + ["largem_route_kernel"]


def _chk(res):
    err = res[0]
    ok = (err == nvrtc.nvrtcResult.NVRTC_SUCCESS) if isinstance(err, nvrtc.nvrtcResult) else (err == cu.CUresult.CUDA_SUCCESS)
    if not ok:
        raise RuntimeError(f"cuda-python error {err}")
    return res[1:] if len(res) > 2 else (res[1] if len(res) == 2 else None)


def compile_device(src, arch, want="cubin", extra=(), names=None):
    names = KERNEL_NAMES if names is None else names
    """Returns (binary bytes, {name: lowered name}). arch like 'sm_75' (cubin) or 'compute_90' (PTX)."""
    prog = _chk(nvrtc.nvrtcCreateProgram(src.encode(), b"smallm.cu", 0, [], []))
    for n in names:
        _chk(nvrtc.nvrtcAddNameExpression(prog, n.encode()))
    opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17", b"-default-device"] + [e.encode() for e in extra]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    n = _chk(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * n
    _chk(nvrtc.nvrtcGetProgramLog(prog, log))
    if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError("NVRTC compile failed:\n" + log.decode(errors="replace"))
    lowered = {nm: _chk(nvrtc.nvrtcGetLoweredName(prog, nm.encode())).decode() for nm in names}
    if want == "cubin":
        size = _chk(nvrtc.nvrtcGetCUBINSize(prog))
        buf = b" " * size
        _chk(nvrtc.nvrtcGetCUBIN(prog, buf))
    else:
        size = _chk(nvrtc.nvrtcGetPTXSize(prog))
        buf = b" " * size
        _chk(nvrtc.nvrtcGetPTX(prog, buf))
    return buf, lowered


def sass_check(src, sm="sm_90", extra=(), names=None):
    """NVRTC (which runs its own ptxas) for `sm`, e.g. the judge's sm_90: raises if any inline-asm string or instruction is rejected for that GPU.
    Returns (cubin bytes, lowered names)."""
    return compile_device(src, sm, want="cubin", extra=extra, names=names)


class NvrtcExt:
    """Same call signatures as the compiled extension of sonnet_v3 (torch tensors in, kernels launched on the current stream)."""

    def __init__(self, device_src, cc=None):
        torch.zeros(1, device="cuda")  # make sure torch's primary context is current
        major, minor = torch.cuda.get_device_capability() if cc is None else cc
        self.arch = f"sm_{major}{minor}"
        cubin, self.lowered = compile_device(device_src, self.arch)
        self.module = _chk(cu.cuModuleLoadData(cubin))
        self.fn = {n: _chk(cu.cuModuleGetFunction(self.module, low.encode())) for n, low in self.lowered.items()}

    @staticmethod
    def _launch(fn, grid, block, smem, args):
        stream = torch.cuda.current_stream().cuda_stream
        holders, ptrs = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                h = ctypes.c_void_p(a.data_ptr())
            elif isinstance(a, bool):
                h = ctypes.c_int(int(a))
            else:
                h = ctypes.c_int(int(a))
            holders.append(h)
            ptrs.append(ctypes.addressof(h))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        _chk(cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem, stream, ctypes.addressof(arr), 0))

    def smallm_init(self):
        for n in KERNEL_NAMES[:-1]:
            _chk(cu.cuFuncSetAttribute(self.fn[n], cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 49152))

    def smallm_route(self, ids, local_ids, tab, lo, nloc):
        self._launch(self.fn["smallm_route_kernel"], (1, 1, 1), (128, 1, 1), 0, [ids, local_ids, tab, ids.numel(), lo, nloc])

    def smallm_gemm(self, wq, sq, gs, lut, act, tab, part, cnt, out, G, NIT, K, KS, nz, out_stride, act_shift, nw, fuse):
        kc = K // KS
        smem = 1024 + 16 * (kc + 8) * 2
        fn = self.fn[f"w4_smallm_kernel<{'true' if fuse else 'false'},{nw}>"]
        if smem > 48 * 1024:
            _chk(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
        self._launch(fn, (G // nw, KS, nz), (nw * 32, 1, 1), smem,
                     [wq, sq, gs, lut, act, tab, part, cnt, out, G, NIT, K, KS, out_stride, act_shift])


class NvrtcLargeExt(NvrtcExt):
    """Same call signatures as the compiled large-M extension of sonnet_v4 (largem_gemm, largem_route, largem_init)."""

    def __init__(self, device_src, cc=None):
        torch.zeros(1, device="cuda")
        major, minor = torch.cuda.get_device_capability() if cc is None else cc
        self.arch = f"sm_{major}{minor}"
        cubin, self.lowered = compile_device(device_src, self.arch, names=LARGE_NAMES)
        self.module = _chk(cu.cuModuleLoadData(cubin))
        self.fn = {n: _chk(cu.cuModuleGetFunction(self.module, low.encode())) for n, low in self.lowered.items()}

    def largem_init(self):
        pass

    def largem_route(self, ids, local_ids, ctab, slotl, lo, nloc):
        self._launch(self.fn["largem_route_kernel"], (1, 1, 1), (1024, 1, 1), 0, [ids, local_ids, ctab, slotl, ids.numel(), lo, nloc])

    def largem_gemm(self, wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, G, NIT, K, KS, nz, out_stride, act_shift, nw, mr, fuse):
        kc = K // KS
        smem = 1024 + 32 * (kc + 8) * 2
        fn = self.fn[f"w4_largem_kernel<{'true' if fuse else 'false'},{nw},{mr}>"]
        if smem > 48 * 1024:
            _chk(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
        self._launch(fn, (G // (nw * mr), KS, nz), (nw * 32, 1, 1), smem,
                     [wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, G, NIT, K, KS, out_stride, act_shift])


class NvrtcMoe5Ext(NvrtcExt):
    """Same call signatures as the compiled v5 extension of sonnet_v5 (moe5_gemm, moe5_route, moe5_init). `instances` = the (fuse, nw, mr, lb, dbg)
    list to instantiate (sonnet_v5.MOE5_INSTANCES or the measurement file's); `names` as sonnet_v5.moe5_instance_names."""

    def __init__(self, device_src, names, cc=None):
        torch.zeros(1, device="cuda")
        major, minor = torch.cuda.get_device_capability() if cc is None else cc
        self.arch = f"sm_{major}{minor}"
        self.names = list(names) + ["moe5_route_small_kernel", "largem_route_kernel"]
        cubin, self.lowered = compile_device(device_src, self.arch, names=self.names)
        self.module = _chk(cu.cuModuleLoadData(cubin))
        self.fn = {n: _chk(cu.cuModuleGetFunction(self.module, low.encode())) for n, low in self.lowered.items()}

    def moe5_init(self):
        pass

    def moe5_route(self, ids, local_ids, ctab, slotl, lo, nloc, small):
        if small:
            self._launch(self.fn["moe5_route_small_kernel"], (1, 1, 1), (128, 1, 1), 0, [ids, local_ids, ctab, slotl, ids.numel(), lo, nloc])
        else:
            self._launch(self.fn["largem_route_kernel"], (1, 1, 1), (1024, 1, 1), 0, [ids, local_ids, ctab, slotl, ids.numel(), lo, nloc])

    def moe5_gemm(self, wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, dbg, G, NIT, K, KS, nz, NB, rows, out_stride, act_shift,
                  nw, mr, lb, dbg_flags, fuse, pdl):
        kc = K // KS
        smem = 1024 + rows * (kc + 8) * 2
        fn = self.fn[f"w4_moe5_kernel<{'true' if fuse else 'false'},{nw},{mr},{lb},{dbg_flags}>"]
        if smem > 48 * 1024:
            _chk(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
        assert G % (nw * mr * NB) == 0, (G, nw, mr, NB)
        self._launch(fn, (G // (nw * mr) // NB, KS, nz), (nw * 32, 1, 1), smem,
                     [wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, dbg, G, NIT, K, KS, NB, out_stride, act_shift])


class NvrtcMoe6Ext(NvrtcMoe5Ext):
    """Same call signatures as the compiled v6 extension of sonnet_v6 (moe6_gemm, moe6_route, moe6_init); names as sonnet_v6.moe6_instance_names."""

    def moe6_init(self):
        pass

    def moe6_route(self, ids, local_ids, ctab, slotl, lo, nloc, small):
        self.moe5_route(ids, local_ids, ctab, slotl, lo, nloc, small)

    def moe6_gemm(self, wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, dbg, G, NIT, K, KS, nz, NB, rows, out_stride, act_shift,
                  nw, mr, d, rb, rs, lb, dbg_flags, fuse, pdl):
        kc = K // KS
        smem = 1024 + (nw * rs * mr * d * 576 if rs else 0) + rows * (kc + 8) * 2
        fn = self.fn[f"w4_moe6_kernel<{'true' if fuse else 'false'},{nw},{mr},{d},{rb},{rs},{lb},{dbg_flags}>"]
        if smem > 48 * 1024:
            _chk(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
        assert G % (nw * mr * NB) == 0, (G, nw, mr, NB)
        self._launch(fn, (G // (nw * mr) // NB, KS, nz), (nw * 32, 1, 1), smem,
                     [wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, dbg, G, NIT, K, KS, NB, out_stride, act_shift])


class NvrtcMeasExt(NvrtcMoe5Ext):
    """measure_altA_v1's extension: the v5 kernels + meas_wstream / meas_cal (device source = measure_altA_v1.meas_device_src(), names = its MEAS_NAMES)."""

    def meas_wstream(self, wq, sq, ctab, dbg, G, NIT, KS, nz, nw, mr):
        fn = self.fn[f"meas_wstream_kernel<{nw},{mr}>"]
        self._launch(fn, (G // (nw * mr), KS, nz), (nw * 32, 1, 1), 0, [wq, sq, ctab, dbg, G, NIT, KS])

    def meas_cal(self, buf, dbg, grid):
        self._launch(self.fn["meas_cal_kernel"], (grid, 1, 1), (256, 1, 1), 0, [buf, dbg, buf.numel() * buf.element_size() // 16])


class NvrtcMoe7Ext(NvrtcExt):
    """Same call signatures as the compiled v7 extension of sonnet_v7 (moe7_gemm, moe7_route, moe7_init); `names` as sonnet_v7.moe7_instance_names (the template
    instantiations + the two route kernels). On the sm_75 dev card the kernel takes its portable fallback (smem counters, lane-loop copies, bounded spins), or, with
    extra=("-DM7_EMU_MBAR=1",), the software-mbarrier build that runs the sm_90 protocol (phase parity, expect_tx byte accounting) on synchronous copies."""

    def __init__(self, device_src, names, cc=None, extra=()):
        torch.zeros(1, device="cuda")
        major, minor = torch.cuda.get_device_capability() if cc is None else cc
        self.arch = f"sm_{major}{minor}"
        self.names = list(names)
        cubin, self.lowered = compile_device(device_src, self.arch, names=self.names, extra=extra)
        self.module = _chk(cu.cuModuleLoadData(cubin))
        self.fn = {n: _chk(cu.cuModuleGetFunction(self.module, low.encode())) for n, low in self.lowered.items()}

    def _opt_in(self):
        dev = _chk(cu.cuCtxGetDevice())
        return _chk(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev))

    def moe7_init(self):
        """The host code's moe7_init: per instantiation maxDynamic = min(227 KB, optin - static) (NOT optin: static + dynamic must fit), diagnostics line."""
        optin = self._opt_in()
        out = ""
        for n, fn in self.fn.items():
            if not n.startswith("w4_moe7_kernel"):
                continue
            static = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn))
            regs = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn))
            thr = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn))
            dyn = min(232448, optin - static)
            _chk(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dyn))
            out += f"{n} regs={regs} maxThreads={thr} static={static} maxDynamic={dyn} optin={optin}\n"
        return out

    def moe7_occupancy(self, d, dbg_t, fuse, smem):
        fn = self.fn[f"w4_moe7_kernel<{'true' if fuse else 'false'},{d},{dbg_t}>"]
        res = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, 416, smem)
        return int(res[1]) if res[0] == cu.CUresult.CUDA_SUCCESS else -1

    def moe7_route(self, ids, local_ids, ctab, slotl, sloc, ctr, lo, nloc, small):
        if small:
            self._launch(self.fn["m7_route_small_kernel"], (1, 1, 1), (128, 1, 1), 0, [ids, local_ids, ctab, slotl, sloc, ctr, ids.numel(), lo, nloc])
        else:
            self._launch(self.fn["m7_route_large_kernel"], (1, 1, 1), (1024, 1, 1), 0, [ids, local_ids, ctab, slotl, sloc, ctr, ids.numel(), lo, nloc])

    def moe7_gather(self, x, ctab, slotl, dst, zb, K, act_shift, R, pdl):
        self._launch(self.fn["m7_gather_kernel"], (zb, (K // 64 + 7) // 8, 1), (256, 1, 1), 0, [x, ctab, slotl, dst, K, act_shift, R])

    def moe7_gemm(self, wb, gs, lut, act, ctab, slotl, sloc, part, cnt, out, ctr, dbg, G, NIT, K, KS, S, slot_bytes, R, out_stride, nctas, dbg_mode,
                  spin_max, d, dbg_t, fuse, pdl):
        smem = 1024 + S * slot_bytes
        fn = self.fn[f"w4_moe7_kernel<{'true' if fuse else 'false'},{d},{dbg_t}>"]
        # (the attribute was set by moe7_init, as in the host code; a launch with more dynamic smem than that is the driver's INVALID_VALUE)
        self._launch(fn, (nctas, 1, 1), (416, 1, 1), smem,
                     [wb, gs, lut, act, ctab, slotl, sloc, part, cnt, out, ctr, dbg, G, NIT, K, KS, S, slot_bytes, R, out_stride, dbg_mode, spin_max])


class NvrtcMeasWsExt(NvrtcMoe7Ext):
    """measure_ws_v1's extension: the v7 kernels + meas_wstream / meas_cal (device source = measure_ws_v1.meas_device_src(), names = its MEAS_NAMES)."""

    def meas_wstream(self, wq, sq, ctab, dbg, G, NIT, KS, nz):
        fn = self.fn["meas_wstream_kernel<4,2>"]
        self._launch(fn, (G // 8, KS, nz), (128, 1, 1), 0, [wq, sq, ctab, dbg, G, NIT, KS])

    def meas_cal(self, buf, dbg, grid):
        self._launch(self.fn["meas_cal_kernel"], (grid, 1, 1), (256, 1, 1), 0, [buf, dbg, buf.numel() * buf.element_size() // 16])


class NvrtcGemvWsExt(NvrtcExt):
    """measure_gemv_ws_v1's extension on the dev card: same call signatures as the load_inline module (gw_gemv, gw_init, gw_occupancy). `names` = the template
    instantiations of w16_gemv (measure_gemv_ws_v1._gw_inst_names()); the source is its _GW_DEVICE_SRC (sm_75: the shuffle-emulated mma)."""

    def __init__(self, device_src, names, inst, cc=None, extra=()):
        torch.zeros(1, device="cuda")
        major, minor = torch.cuda.get_device_capability() if cc is None else cc
        self.arch = f"sm_{major}{minor}"
        self.names = list(names)
        self.inst = [tuple(i) for i in inst]
        cubin, self.lowered = compile_device(device_src, self.arch, names=self.names, extra=extra)
        self.module = _chk(cu.cuModuleLoadData(cubin))
        self.fn = {n: _chk(cu.cuModuleGetFunction(self.module, low.encode())) for n, low in self.lowered.items()}

    def _opt_in(self):
        dev = _chk(cu.cuCtxGetDevice())
        return _chk(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev))

    def gw_init(self):
        """The host code's gw_init: per instantiation maxDynamic = optin - static (static + dynamic must fit the opt-in limit), one diagnostic line each."""
        optin = self._opt_in()
        out = ""
        for n, fn in self.fn.items():
            static = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn))
            regs = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn))
            local = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn))
            dyn = optin - static
            _chk(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dyn))
            out += f"{n} regs={regs} local={local} static={static} maxDynamic={dyn}\n"
        return out

    def gw_occupancy(self, inst, nw, smem):
        fn = self.fn[self.names[inst]]
        res = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, int(nw) * 32, int(smem))
        return int(res[1]) if res[0] == cu.CUresult.CUDA_SUCCESS else -1

    def gw_gemv(self, x, w, wlo, out, ws, done, M, N, K, SPLIT, nw, inst, ldx, ldo):
        mt, rpw, u, xs, lo = self.inst[inst]
        kc = K // SPLIT
        wn = u // (rpw * (1 + lo))
        assert K % SPLIT == 0 and kc % (32 * wn) == 0 and kc % 64 == 0 and N % (8 * rpw * nw) == 0 and 16 * (mt - 1) < M <= 16 * mt, "gw: bad shape"
        gx = N // (8 * rpw * nw)
        assert SPLIT == 1 or (gx <= done.numel() and gx * SPLIT * nw * rpw * mt * 32 <= ws.shape[0]), "gw: workspace"
        smem = M * (2 * kc + 64) if xs else 0
        self._launch(self.fn[self.names[inst]], (gx, SPLIT, 1), (nw * 32, 1, 1), smem, [x, w, wlo if lo else w, out, ws, done, M, N, K, SPLIT, ldx, ldo])
