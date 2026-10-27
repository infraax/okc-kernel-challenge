# moe_smallm_cuda.py
# okc: problem=muse_moe_w4a16
"""Small-M (M <= 16) streaming split-K W4A16 MoE GEMM1 (fused SwiGLU) + GEMM2 in CUDA."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

MAX_M = 16
TOP_K = 8
POST_NORM_EPS = 1e-8
EPILOGUE_BLOCK_H = 1024
EPILOGUE_WARPS = 8

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
    uint32_t s = (b & 0x80u) << 24;
    uint32_t e = (b >> 3) & 0xFu;
    uint32_t m = b & 0x7u;
    if (e == 0) return (b & 0x80u ? -1.0f : 1.0f) * (float(m) * (1.0f / 512.0f));
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float x) {
#if __CUDA_ARCH__ >= 800
    uint16_t out;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(x));
    return out;
#else
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffU) > 0x7f800000U) return 0x7fc0U;
    return uint16_t((u + (0x7fffU + ((u >> 16) & 1U))) >> 16);
#endif
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
#if __CUDA_ARCH__ >= 800
    uint32_t out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
    return out;
#else
    return uint32_t(f32_to_bf16_bits(lo)) | (uint32_t(f32_to_bf16_bits(hi)) << 16);
#endif
}

__device__ __forceinline__ float bf16_lo(uint32_t p) { return __uint_as_float((p & 0xFFFFu) << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t p) { return __uint_as_float(p & 0xFFFF0000u); }

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    int lane = threadIdx.x & 31, gid = lane >> 2, t_grp = lane & 3;
    float s0 = c[0], s1 = c[1], s2 = c[2], s3 = c[3];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int sa = (gid << 2) | t, sb0 = ((2 * t_grp) << 2) | t, sb1 = ((2 * t_grp + 1) << 2) | t;
        uint32_t a0 = __shfl_sync(0xffffffff, a[0], sa), a1 = __shfl_sync(0xffffffff, a[1], sa);
        uint32_t a2 = __shfl_sync(0xffffffff, a[2], sa), a3 = __shfl_sync(0xffffffff, a[3], sa);
        uint32_t b00 = __shfl_sync(0xffffffff, b[0], sb0), b10 = __shfl_sync(0xffffffff, b[1], sb0);
        uint32_t b01 = __shfl_sync(0xffffffff, b[0], sb1), b11 = __shfl_sync(0xffffffff, b[1], sb1);
        s0 += bf16_lo(a0) * bf16_lo(b00) + bf16_hi(a0) * bf16_hi(b00) + bf16_lo(a2) * bf16_lo(b10) + bf16_hi(a2) * bf16_hi(b10);
        s1 += bf16_lo(a0) * bf16_lo(b01) + bf16_hi(a0) * bf16_hi(b01) + bf16_lo(a2) * bf16_lo(b11) + bf16_hi(a2) * bf16_hi(b11);
        s2 += bf16_lo(a1) * bf16_lo(b00) + bf16_hi(a1) * bf16_hi(b00) + bf16_lo(a3) * bf16_lo(b10) + bf16_hi(a3) * bf16_hi(b10);
        s3 += bf16_lo(a1) * bf16_lo(b01) + bf16_hi(a1) * bf16_hi(b01) + bf16_lo(a3) * bf16_lo(b11) + bf16_hi(a3) * bf16_hi(b11);
    }
    d[0] = s0; d[1] = s1; d[2] = s2; d[3] = s3;
#endif
}

template <bool FUSE_SWIGLU, bool X_IS_SLOTTED>
__global__ __launch_bounds__(128) void w4a16_smallm_splitk_kernel(
    const uint32_t* __restrict__ x_u32,
    const uint8_t* __restrict__ w_ptr,
    const uint8_t* __restrict__ sf_ptr,
    const float* __restrict__ gs_ptr,
    const int32_t* __restrict__ local_ids,
    float* __restrict__ partials,
    int32_t* __restrict__ counters,
    uint32_t* __restrict__ out_u32,
    int M, int top_k, int N_out, int K, int k_per_slice) {

    __shared__ uint32_t smem_mask[4];
    __shared__ int smem_routes[16];
    __shared__ int smem_m_e;
    __shared__ int smem_is_last;
    __shared__ float smem_lut[16];
    __shared__ union {
        uint32_t a[2][16][17];
        float up[16][65];
    } smem;

    int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    int gid = lane >> 2, t = lane & 3;
    int n_tile = blockIdx.x, k_slice = blockIdx.y, e = blockIdx.z;
    int n_tiles = gridDim.x, k_slices = gridDim.y;
    int total_slots = M * top_k;

    bool match = (tid < total_slots) && (local_ids[tid] == e);
    uint32_t mask = __ballot_sync(0xffffffff, match);
    if (lane == 0) smem_mask[warp_id] = mask;
    if (tid < 16) {
        const float lut[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                              -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
        smem_lut[tid] = lut[tid];
    }
    __syncthreads();
    if (tid == 0) {
        smem_m_e = __popc(smem_mask[0]) + __popc(smem_mask[1]) +
                   __popc(smem_mask[2]) + __popc(smem_mask[3]);
    }
    if (match) {
        int prefix = 0;
        #pragma unroll
        for (int w = 0; w < 4; ++w) if (w < warp_id) prefix += __popc(smem_mask[w]);
        int rank = prefix + __popc(mask & ((1u << lane) - 1u));
        if (rank < 16) smem_routes[rank] = tid;
    }
    __syncthreads();
    int m_e = smem_m_e;
    if (m_e == 0) return;
    if (m_e > 16) m_e = 16;

    int N_w = FUSE_SWIGLU ? (2 * N_out) : N_out;
    int n_row = FUSE_SWIGLU
        ? ((warp_id < 2) ? (n_tile * 64 + warp_id * 32 + lane) : (N_out + n_tile * 64 + (warp_id - 2) * 32 + lane))
        : (n_tile * 128 + warp_id * 32 + lane);
    const uint8_t* w_row = w_ptr + (size_t(e) * N_w + n_row) * (K / 2);
    const uint8_t* sf_row = sf_ptr + (size_t(e) * N_w + n_row) * (K / 16);

    int a_row = tid >> 3, a_col_vec = tid & 7;
    int in_row = (a_row < m_e) ? (X_IS_SLOTTED ? smem_routes[a_row] : (smem_routes[a_row] / top_k)) : 0;
    const uint2* x_ptr = reinterpret_cast<const uint2*>(x_u32 + size_t(in_row) * (K / 2)) + a_col_vec;

    float acc[4][4] = {0.f};
    int k_start = k_slice * k_per_slice;
    int k_end = min(K, k_start + k_per_slice);

    for (int k = k_start, stage = 0; k < k_end; k += 32, stage ^= 1) {
        uint4 w_v = *reinterpret_cast<const uint4*>(w_row + k / 2);
        uint16_t sf_v = *reinterpret_cast<const uint16_t*>(sf_row + k / 16);
        uint2 x_v = (a_row < m_e) ? x_ptr[k / 4] : make_uint2(0u, 0u);
        smem.a[stage][a_row][2 * a_col_vec] = x_v.x;
        smem.a[stage][a_row][2 * a_col_vec + 1] = x_v.y;
        __syncthreads();

        float s0 = e4m3_to_float(uint8_t(sf_v & 0xFFu));
        float s1 = e4m3_to_float(uint8_t(sf_v >> 8));
        uint32_t a0[4] = {smem.a[stage][gid][t], smem.a[stage][gid + 8][t],
                          smem.a[stage][gid][t + 4], smem.a[stage][gid + 8][t + 4]};
        uint32_t a1[4] = {smem.a[stage][gid][t + 8], smem.a[stage][gid + 8][t + 8],
                          smem.a[stage][gid][t + 12], smem.a[stage][gid + 8][t + 12]};

        #pragma unroll
        for (int sn = 0; sn < 4; ++sn) {
            int src = sn * 8 + gid;
            uint32_t wl0 = __shfl_sync(0xffffffff, w_v.x, src), wh0 = __shfl_sync(0xffffffff, w_v.y, src);
            float sc0 = __shfl_sync(0xffffffff, s0, src);
            uint32_t blo0 = (wl0 >> (8 * t)) & 0xFFu, bhi0 = (wh0 >> (8 * t)) & 0xFFu;
            uint32_t b0[2] = {pack_bf16x2(smem_lut[blo0 & 0xFu] * sc0, smem_lut[blo0 >> 4] * sc0),
                              pack_bf16x2(smem_lut[bhi0 & 0xFu] * sc0, smem_lut[bhi0 >> 4] * sc0)};
            mma_m16n8k16_bf16(acc[sn], a0, b0, acc[sn]);

            uint32_t wl1 = __shfl_sync(0xffffffff, w_v.z, src), wh1 = __shfl_sync(0xffffffff, w_v.w, src);
            float sc1 = __shfl_sync(0xffffffff, s1, src);
            uint32_t blo1 = (wl1 >> (8 * t)) & 0xFFu, bhi1 = (wh1 >> (8 * t)) & 0xFFu;
            uint32_t b1[2] = {pack_bf16x2(smem_lut[blo1 & 0xFu] * sc1, smem_lut[blo1 >> 4] * sc1),
                              pack_bf16x2(smem_lut[bhi1 & 0xFu] * sc1, smem_lut[bhi1 >> 4] * sc1)};
            mma_m16n8k16_bf16(acc[sn], a1, b1, acc[sn]);
        }
    }

    int c_base = warp_id * 32 + 2 * t;
    if (k_slices > 1) {
        #pragma unroll
        for (int sn = 0; sn < 4; ++sn) {
            int c0 = c_base + sn * 8;
            if (gid < m_e) {
                size_t off = ((size_t(smem_routes[gid]) * n_tiles + n_tile) * k_slices + k_slice) * 128 + c0;
                *reinterpret_cast<float2*>(partials + off) = make_float2(acc[sn][0], acc[sn][1]);
            }
            if (gid + 8 < m_e) {
                size_t off = ((size_t(smem_routes[gid + 8]) * n_tiles + n_tile) * k_slices + k_slice) * 128 + c0;
                *reinterpret_cast<float2*>(partials + off) = make_float2(acc[sn][2], acc[sn][3]);
            }
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            int prev = atomicAdd(&counters[e * n_tiles + n_tile], 1);
            smem_is_last = (prev == k_slices - 1);
            if (smem_is_last) counters[e * n_tiles + n_tile] = 0;
        }
        __syncthreads();
        if (!smem_is_last) return;
        __threadfence();

        #pragma unroll
        for (int sn = 0; sn < 4; ++sn) {
            acc[sn][0] = acc[sn][1] = acc[sn][2] = acc[sn][3] = 0.f;
            int c0 = c_base + sn * 8;
            for (int s = 0; s < k_slices; ++s) {
                if (gid < m_e) {
                    size_t off = ((size_t(smem_routes[gid]) * n_tiles + n_tile) * k_slices + s) * 128 + c0;
                    float2 v = *reinterpret_cast<const volatile float2*>(partials + off);
                    acc[sn][0] += v.x; acc[sn][1] += v.y;
                }
                if (gid + 8 < m_e) {
                    size_t off = ((size_t(smem_routes[gid + 8]) * n_tiles + n_tile) * k_slices + s) * 128 + c0;
                    float2 v = *reinterpret_cast<const volatile float2*>(partials + off);
                    acc[sn][2] += v.x; acc[sn][3] += v.y;
                }
            }
        }
    }

    float gs = gs_ptr[e];
    if constexpr (FUSE_SWIGLU) {
        __syncthreads();
        if (warp_id >= 2) {
            int c_up = (warp_id - 2) * 32 + 2 * t;
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                smem.up[gid][c_up + sn * 8] = acc[sn][0] * gs;
                smem.up[gid][c_up + sn * 8 + 1] = acc[sn][1] * gs;
                smem.up[gid + 8][c_up + sn * 8] = acc[sn][2] * gs;
                smem.up[gid + 8][c_up + sn * 8 + 1] = acc[sn][3] * gs;
            }
        }
        __syncthreads();
        if (warp_id < 2) {
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                int c_g = c_base + sn * 8;
                if (gid < m_e) {
                    float g0 = acc[sn][0] * gs, g1 = acc[sn][1] * gs;
                    float v0 = (g0 / (1.0f + expf(-g0))) * smem.up[gid][c_g];
                    float v1 = (g1 / (1.0f + expf(-g1))) * smem.up[gid][c_g + 1];
                    out_u32[size_t(smem_routes[gid]) * (N_out / 2) + (n_tile * 64 + c_g) / 2] = pack_bf16x2(v0, v1);
                }
                if (gid + 8 < m_e) {
                    float g2 = acc[sn][2] * gs, g3 = acc[sn][3] * gs;
                    float v2 = (g2 / (1.0f + expf(-g2))) * smem.up[gid + 8][c_g];
                    float v3 = (g3 / (1.0f + expf(-g3))) * smem.up[gid + 8][c_g + 1];
                    out_u32[size_t(smem_routes[gid + 8]) * (N_out / 2) + (n_tile * 64 + c_g) / 2] = pack_bf16x2(v2, v3);
                }
            }
        }
    } else {
        #pragma unroll
        for (int sn = 0; sn < 4; ++sn) {
            int c0 = c_base + sn * 8;
            if (gid < m_e) {
                out_u32[size_t(smem_routes[gid]) * (N_out / 2) + (n_tile * 128 + c0) / 2] =
                    pack_bf16x2(acc[sn][0] * gs, acc[sn][1] * gs);
            }
            if (gid + 8 < m_e) {
                out_u32[size_t(smem_routes[gid + 8]) * (N_out / 2) + (n_tile * 128 + c0) / 2] =
                    pack_bf16x2(acc[sn][2] * gs, acc[sn][3] * gs);
            }
        }
    }
}

void launch_w4a16_smallm(
    torch::Tensor x, torch::Tensor w, torch::Tensor sf, torch::Tensor gs,
    torch::Tensor local_ids, torch::Tensor partials, torch::Tensor counters,
    torch::Tensor out, int k_slices, bool fuse_swiglu, bool x_is_slotted) {
    c10::cuda::CUDAGuard guard(x.device());
    int M = local_ids.size(0), top_k = local_ids.size(1);
    int E = w.size(0), K = w.size(2) * 2;
    int N_out = fuse_swiglu ? (w.size(1) / 2) : w.size(1);
    int n_tiles = fuse_swiglu ? (N_out / 64) : (N_out / 128);
    int k_per_slice = ((K / k_slices + 31) / 32) * 32;
    dim3 grid(n_tiles, k_slices, E);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto fn = fuse_swiglu
        ? (x_is_slotted ? w4a16_smallm_splitk_kernel<true, true> : w4a16_smallm_splitk_kernel<true, false>)
        : (x_is_slotted ? w4a16_smallm_splitk_kernel<false, true> : w4a16_smallm_splitk_kernel<false, false>);
    fn<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(x.data_ptr()),
        reinterpret_cast<const uint8_t*>(w.data_ptr()),
        reinterpret_cast<const uint8_t*>(sf.data_ptr()),
        gs.data_ptr<float>(),
        local_ids.data_ptr<int32_t>(),
        partials.data_ptr<float>(),
        counters.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(out.data_ptr()),
        M, top_k, N_out, K, k_per_slice);
}
"""

_CPP_SRC = "void launch_w4a16_smallm(torch::Tensor x, torch::Tensor w, torch::Tensor sf, torch::Tensor gs, torch::Tensor local_ids, torch::Tensor partials, torch::Tensor counters, torch::Tensor out, int k_slices, bool fuse_swiglu, bool x_is_slotted);"
_MOD = None


def _get_mod():
    global _MOD
    if _MOD is None:
        _MOD = load_inline(
            name="moe_smallm_cuda_ext",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["launch_w4a16_smallm"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _MOD


def make_smallm_workspace(device: torch.device | str, max_m: int = MAX_M, top_k: int = TOP_K,
                          n_local: int = 32, max_n_tiles: int = 64, max_k_slices: int = 8) -> dict[str, torch.Tensor]:
    """Allocate split-K partials (FP32) and deterministic fixup counters (INT32, zeroed once in prepare)."""
    partials = torch.empty((max_m * top_k, max_n_tiles, max_k_slices, 128), dtype=torch.float32, device=device)
    counters = torch.zeros((2, n_local * max_n_tiles), dtype=torch.int32, device=device)
    return {"partials": partials, "counters": counters}


def _pick_k_slices(M: int, N_tiles: int, K: int, k_slices: int | None) -> int:
    if k_slices is not None:
        return k_slices
    target = 4 if (N_tiles >= 64 or M >= 16) else 8
    while target > 1 and (K // target < 32 or (K // target) % 32 != 0):
        target //= 2
    return max(1, target)


def gemm1_smallm(
    x: torch.Tensor, w13: torch.Tensor, w13_sf: torch.Tensor, w13_gs: torch.Tensor,
    local_ids: torch.Tensor, partials: torch.Tensor, counters: torch.Tensor,
    out: torch.Tensor | None = None, k_slices: int | None = None,
) -> torch.Tensor:
    """Fused W4A16 GEMM1 + SwiGLU for M <= 16.
    x: bfloat16 [M, H]; w13: uint8 [E, 2I, H/2]; w13_sf: float8_e4m3fn [E, 2I, H/16]; w13_gs: float32 [E];
    local_ids: int32 [M, K_top] in [-1, E); returns inter: bfloat16 [M, K_top, I].
    """
    M, top_k = local_ids.shape
    I = w13.shape[1] // 2
    K = w13.shape[2] * 2
    if out is None:
        out = torch.empty((M, top_k, I), dtype=torch.bfloat16, device=x.device)
    ks = _pick_k_slices(M, I // 64, K, k_slices)
    _get_mod().launch_w4a16_smallm(x, w13, w13_sf, w13_gs, local_ids, partials, counters[0], out, ks, True, False)
    return out


def gemm2_smallm(
    inter: torch.Tensor, w2: torch.Tensor, w2_sf: torch.Tensor, w2_gs: torch.Tensor,
    local_ids: torch.Tensor, partials: torch.Tensor, counters: torch.Tensor,
    out: torch.Tensor | None = None, k_slices: int | None = None,
) -> torch.Tensor:
    """W4A16 GEMM2 for M <= 16.
    inter: bfloat16 [M, K_top, I]; w2: uint8 [E, H, I/2]; w2_sf: float8_e4m3fn [E, H, I/16]; w2_gs: float32 [E];
    local_ids: int32 [M, K_top] in [-1, E); returns y: bfloat16 [M, K_top, H].
    """
    M, top_k = local_ids.shape
    H = w2.shape[1]
    K = w2.shape[2] * 2
    if out is None:
        out = torch.empty((M, top_k, H), dtype=torch.bfloat16, device=inter.device)
    ks = _pick_k_slices(M, H // 128, K, k_slices)
    _get_mod().launch_w4a16_smallm(inter, w2, w2_sf, w2_gs, local_ids, partials, counters[1], out, ks, False, True)
    return out


# ===================================================================== sonnet_v1 local-id + epilogue
@triton.jit
def _local_ids_kernel(ids_ptr, out_ptr, n, lo, hi, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    e = tl.load(ids_ptr + offs, mask=mask, other=-1)
    tl.store(out_ptr + offs, tl.where((e >= lo) & (e < hi), e - lo, -1), mask=mask)


@triton.jit
def _epilogue_kernel(y_ptr, w_ptr, s_ptr, ids_ptr, out_ptr, eps, TOPK: tl.constexpr, KP: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr):
    tok = tl.program_id(0)
    NCHUNK: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    slots = tl.arange(0, KP)
    in_k = slots < TOPK
    live = in_k & (tl.load(ids_ptr + tok * TOPK + slots, mask=in_k, other=-1) >= 0)
    s = tl.load(s_ptr + tok * TOPK + slots, mask=live, other=0.0).to(tl.float32)
    row = (tok * TOPK + slots)[:, None] * H
    offs0 = tl.arange(0, BLOCK_H)
    sumsq = tl.zeros([KP], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        offs = c * BLOCK_H + offs0
        hm = offs < H
        w = tl.load(w_ptr + offs, mask=hm, other=0.0).to(tl.float32)
        x = tl.load(y_ptr + row + offs[None, :], mask=live[:, None] & hm[None, :], other=0.0).to(tl.float32)
        yv = x * w[None, :]
        sumsq += tl.sum(yv * yv, axis=1)
    coef = tl.where(live, s * tl.rsqrt(sumsq / H + eps), 0.0)
    for c in tl.static_range(NCHUNK):
        offs = c * BLOCK_H + offs0
        hm = offs < H
        w = tl.load(w_ptr + offs, mask=hm, other=0.0).to(tl.float32)
        x = tl.load(y_ptr + row + offs[None, :], mask=live[:, None] & hm[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + tok * H + offs, (tl.sum(x * coef[:, None], axis=0) * w).to(out_ptr.dtype.element_ty), mask=hm)


def local_ids_triton(topk_ids: torch.Tensor, start: int, count: int) -> torch.Tensor:
    ids = topk_ids.contiguous()
    out = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    _local_ids_kernel[(triton.cdiv(ids.numel(), 1024),)](ids, out, ids.numel(), start, start + count, BLOCK=1024, num_warps=4)
    return out


def epilogue_triton(expert_out: torch.Tensor, topk_scores: torch.Tensor, slot_ids: torch.Tensor, norm_w: torch.Tensor, eps: float = POST_NORM_EPS) -> torch.Tensor:
    y = expert_out.contiguous()
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    if tokens == 0:
        return out
    block_h = min(EPILOGUE_BLOCK_H, triton.next_power_of_2(hidden))
    _epilogue_kernel[(tokens,)](y, norm_w, topk_scores.contiguous(), slot_ids.contiguous(), out, eps,
                                TOPK=topk, KP=triton.next_power_of_2(topk), H=hidden, BLOCK_H=block_h, num_warps=EPILOGUE_WARPS)
    return out
