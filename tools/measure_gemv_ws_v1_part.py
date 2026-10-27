# ---------------------------------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY block (tools/gen_measure_gemv_ws_v1.py appends it after the unchanged director_v2 code and the inlined generate_input). Nothing here feeds kernel_fn:
# prepare() runs director_v2's prepare(), builds the CUDA GEMV family of this study (one torch load_inline build, 10 instantiations, `-w`), then times every
# (layer, m) of the judged shapes on generate_input-recipe inputs (3 seeds, read-only L2 flush, CUDA-graph replays, interleaved rounds with the board kernel as pivot) and prints
# the S / B / D lines.
# ---------------------------------------------------------------------------------------------------------------------
import time as _gw_time

_GW_T0 = _gw_time.perf_counter()
_GW_BUDGET_S = 262.0        # prepare() ends by here (the judge's limit is 300 s); candidates that do not fit are dropped, not run late
_GW_SEEDS = (11, 12, 13)
_GW_REPS = 5                # timed replays per (variant, seed); the median of those is that seed's value
_GW_FLUSH_MB = 128          # read-only L2 flush buffer (> the 50 MB L2)
_GW_MAX_BYTES = 13800
_GW_TOL = {"router": 1e-4}  # relative L2 bound; every other layer 1e-2 (problems/muse_bf16_proj/reference.py)
_GW_ROW_FACTOR, _GW_ROW_FLOOR, _GW_PROBE_TOL = 5.0, 0.1, 1e-3
_GW_MS = (1, 8, 16, 32)
_GW_LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
_GW_NW = (1, 2, 4, 8)       # warps per CTA of the candidates
_GW_SPLITS = (1, 2, 4)      # split-K of the candidates
_GW_MAX_CAND = 36           # candidates per cell (the ones whose CTA count is closest to 4 CTAs / SM)
_GW_WIN_PCT = 5.0           # D line: the best WS variant beats the board by at least this many percent
_GW_SMEM_CAP = 200 * 1024
_GW_WS_FLOAT4 = 8 * 1024 * 1024 // 16     # split-K partial workspace (fragment layout)
_GW_DONE = 4096
_GW_NOTES = []

# the instantiations (MT, RPW, U, XS, LO): MT = 16-token M tiles (1: m <= 16, 2: m <= 32), RPW = 8-row W tiles per warp, U = 16-byte W loads in flight per lane (hi + lo of a router
# tile count together), XS = x staged in smem (1) or read through L1 (0), LO = router (BF16 hi + lo weights, FP32 output) -- 10 instantiations
_GW_INST = ((1, 1, 8, 1, 0), (1, 1, 16, 1, 0), (1, 2, 16, 1, 0), (1, 1, 16, 0, 0), (2, 1, 16, 1, 0), (2, 1, 16, 0, 0), (2, 2, 16, 0, 0),
            (1, 1, 16, 1, 1), (1, 2, 16, 1, 1), (2, 1, 16, 0, 1))


def _gw_elapsed():
    return _gw_time.perf_counter() - _GW_T0


def _gw_note(text):
    _GW_NOTES.append(" ".join(str(text).split())[:200])


def _gw_where(ex):
    return f"{type(ex).__name__}: {str(ex)[:90]}".replace("\n", " ")


def _gw_med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _gw_nk(layer, m):
    n, k = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}[layer]
    if layer == "unpooler" and m <= 8:
        n = 1024   # this GPU's slice (problems/muse_bf16_proj/reference.py UNPOOL_SLICE); the layer's decode path computes exactly this
    return n, k


# ================================================================================================================ the CUDA source
_GW_DEVICE_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;
#define FULLMASK 0xffffffffu

__device__ __forceinline__ uint4 ldg_stream4(const void* p) {   // the weights: streamed once, no L1 allocation
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 ldg_x4(const void* p) {        // x: read-only for the whole kernel, small, L1 / L2 resident
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 ldcg4(const void* p) {          // split-K partials of other CTAs: L2 only
    uint4 r;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p) : "memory");
    return r;
}
__device__ __forceinline__ u32 f2bf2(float lo, float hi) {       // two FP32 -> one BF16x2 word (round to nearest even)
#if __CUDA_ARCH__ >= 800
    u32 d;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(lo), "f"(hi));
    return d;
#else
    u32 a = __float_as_uint(lo), b = __float_as_uint(hi);
    a = (a + 0x7fffu + ((a >> 16) & 1u)) >> 16;
    b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
    return a | b;
#endif
}

#if __CUDA_ARCH__ < 800
// sm_75 emulation of the mma fragment layout with shuffles (the development card has no BF16 tensor cores); not inlined.
__device__ __forceinline__ float bflo(u32 p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float bfhi(u32 p) { return __uint_as_float(p & 0xffff0000u); }
__device__ __noinline__ void mma_emul(float* c, const u32* a, u32 b0, u32 b1) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, t = lane & 3;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int sa = gid * 4 + p, sb0 = (2 * t) * 4 + p, sb1 = (2 * t + 1) * 4 + p;
        u32 a0 = __shfl_sync(FULLMASK, a[0], sa), a1 = __shfl_sync(FULLMASK, a[1], sa);
        u32 a2 = __shfl_sync(FULLMASK, a[2], sa), a3 = __shfl_sync(FULLMASK, a[3], sa);
        u32 x00 = __shfl_sync(FULLMASK, b0, sb0), x01 = __shfl_sync(FULLMASK, b1, sb0);
        u32 x10 = __shfl_sync(FULLMASK, b0, sb1), x11 = __shfl_sync(FULLMASK, b1, sb1);
        s0 += bflo(a0) * bflo(x00) + bfhi(a0) * bfhi(x00) + bflo(a2) * bflo(x01) + bfhi(a2) * bfhi(x01);
        s1 += bflo(a0) * bflo(x10) + bfhi(a0) * bfhi(x10) + bflo(a2) * bflo(x11) + bfhi(a2) * bfhi(x11);
        s2 += bflo(a1) * bflo(x00) + bfhi(a1) * bfhi(x00) + bflo(a3) * bflo(x01) + bfhi(a3) * bfhi(x01);
        s3 += bflo(a1) * bflo(x10) + bfhi(a1) * bfhi(x10) + bflo(a3) * bflo(x11) + bfhi(a3) * bfhi(x11);
    }
    c[0] += s0; c[1] += s1; c[2] += s2; c[3] += s3;
}
#endif

// D[16 x 8] += A[16 x 16] * B[16 x 8], BF16 operands, FP32 accumulate (PTX m16n8k16 fragment layout).
__device__ __forceinline__ void mma16816(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    const u32 a[4] = {a0, a1, a2, a3};
    mma_emul(c, a, b0, b1);
#endif
}

// y[m, N] = x[m, K] . W[N, K]^T (+ x . Wlo^T for the router), FP32 accumulation on the tensor cores with the WEIGHTS as the B operand (an n8 tile = 8 weight rows) and the tokens as
// A (16-token M tiles). One 16-byte load of a weight row per lane = the lane's whole B fragment of two k16 steps: the k order inside a k32 window is permuted (lane t of a quad holds
// k = 8 t .. 8 t + 7 of the window), and x is read with the SAME permutation (16 bytes of a token row per lane), so every product is formed exactly once.
//   grid (N / (8 RPW nw), SPLIT), block nw warps; CTA = nw x RPW weight tiles of 8 rows over one K slice Kc = K / SPLIT; each lane keeps U independent 16-byte loads in flight
//   (one round = U loads, then the mma chain), x (m rows of the slice) staged in smem once per CTA (XS) or read through L1; split-K: every CTA stores its FP32 partials in the
//   fragment layout, the last arriving CTA of an n-block adds them in slice order 0..SPLIT-1 (deterministic), stores the output (BF16, or FP32 for the router) and resets the ticket.
template <int MT, int RPW, int U, int XS, int LO>
__global__ void __launch_bounds__(256)
w16_gemv(const u16* __restrict__ X, const u16* __restrict__ W, const u16* __restrict__ Wlo, void* __restrict__ Out, float4* __restrict__ ws, int* __restrict__ done,
         int M, int N, int K, int SPLIT, int ldx, int ldo) {
    constexpr int NL = 1 + LO;
    constexpr int WN = U / (RPW * NL);          // k32 windows per round
    static_assert(WN >= 1 && U % (RPW * NL) == 0, "bad U");
    extern __shared__ __align__(16) u8 smem_raw[];
    __shared__ int s_flag;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t = lane & 3, nw = blockDim.x >> 5;
    const int Kc = K / SPLIT, k0 = blockIdx.y * Kc;
    const int nrow0 = (blockIdx.x * nw + warp) * RPW * 8;     // first weight row of this warp
    const u16* wp[RPW];
    const u16* wl[RPW];
#pragma unroll
    for (int rt = 0; rt < RPW; ++rt) {
        wp[rt] = W + (u64)(nrow0 + rt * 8 + gid) * K + k0 + 8 * t;
        wl[rt] = LO ? Wlo + (u64)(nrow0 + rt * 8 + gid) * K + k0 + 8 * t : wp[rt];
    }
    uint4 w[U];
#define ISSUE(KK)                                                                                  \
    _Pragma("unroll") for (int rt = 0; rt < RPW; ++rt)                                              \
        _Pragma("unroll") for (int wi = 0; wi < WN; ++wi) {                                         \
            w[rt * WN + wi] = ldg_stream4(wp[rt] + (KK) + 32 * wi);                                 \
            if constexpr (LO) w[RPW * WN + rt * WN + wi] = ldg_stream4(wl[rt] + (KK) + 32 * wi);      \
        }
    ISSUE(0)                                       // the first round's weights are in flight while x is staged
    const int pitch = Kc + 32;                      // elements: a row pitch of 64 bytes mod 128 keeps the quarter-warp 16-byte reads conflict-free
    if (XS) {
        u16* xs = reinterpret_cast<u16*>(smem_raw);
        const int cpr = Kc >> 3;
        for (int idx = tid; idx < M * cpr; idx += blockDim.x) {
            const int r = idx / cpr, c = idx - r * cpr;
            *reinterpret_cast<uint4*>(xs + (u64)r * pitch + c * 8) = ldg_x4(X + (u64)r * ldx + k0 + c * 8);
        }
        __syncthreads();
    }
    float acc[RPW][MT][4];
#pragma unroll
    for (int rt = 0; rt < RPW; ++rt)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) acc[rt][mt][0] = acc[rt][mt][1] = acc[rt][mt][2] = acc[rt][mt][3] = 0.f;
    // token rows of this lane's A fragment: gid and gid + 8 of each M tile (rows beyond M repeat row M - 1: valid memory, results discarded)
    int ra[MT], rb[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        ra[mt] = min(mt * 16 + gid, M - 1);
        rb[mt] = min(mt * 16 + gid + 8, M - 1);
    }
    for (int kk = 0; kk < Kc; kk += 32 * WN) {
        if (kk > 0) {
            ISSUE(kk)
        }
#pragma unroll
        for (int wi = 0; wi < WN; ++wi) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                uint4 xa, xb;
                if (XS) {
                    const u16* xs = reinterpret_cast<const u16*>(smem_raw);
                    xa = *reinterpret_cast<const uint4*>(xs + (u64)ra[mt] * pitch + kk + 32 * wi + 8 * t);
                    xb = *reinterpret_cast<const uint4*>(xs + (u64)rb[mt] * pitch + kk + 32 * wi + 8 * t);
                } else {
                    xa = ldg_x4(X + (u64)ra[mt] * ldx + k0 + kk + 32 * wi + 8 * t);
                    xb = ldg_x4(X + (u64)rb[mt] * ldx + k0 + kk + 32 * wi + 8 * t);
                }
#pragma unroll
                for (int rt = 0; rt < RPW; ++rt) {
#pragma unroll
                    for (int l = 0; l < NL; ++l) {
                        const uint4 wv = w[l * RPW * WN + rt * WN + wi];
                        mma16816(acc[rt][mt], xa.x, xb.x, xa.y, xb.y, wv.x, wv.y);     // k elements 0..3 of the lane's 8
                        mma16816(acc[rt][mt], xa.z, xb.z, xa.w, xb.w, wv.z, wv.w);     // k elements 4..7
                    }
                }
            }
        }
    }
#undef ISSUE
    // ---- output: fragment d0 d1 = (token gid, rows 2t, 2t + 1), d2 d3 = (token gid + 8, same rows)
    if (SPLIT > 1) {
        const u64 pb = ((((u64)blockIdx.x * SPLIT + blockIdx.y) * nw + warp) * RPW * MT) * 32 + lane;
#pragma unroll
        for (int rt = 0; rt < RPW; ++rt)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) ws[pb + (rt * MT + mt) * 32] = make_float4(acc[rt][mt][0], acc[rt][mt][1], acc[rt][mt][2], acc[rt][mt][3]);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int prev = atomicAdd(&done[blockIdx.x], 1);
            const int last = (prev == SPLIT - 1);
            s_flag = last;
            if (last) done[blockIdx.x] = 0;
        }
        __syncthreads();
        if (!s_flag) return;
        __threadfence();
#pragma unroll
        for (int rt = 0; rt < RPW; ++rt)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                for (int s = 0; s < SPLIT; ++s) {
                    const uint4 v = ldcg4(ws + ((((u64)blockIdx.x * SPLIT + s) * nw + warp) * RPW * MT + rt * MT + mt) * 32 + lane);
                    a0 += __uint_as_float(v.x); a1 += __uint_as_float(v.y); a2 += __uint_as_float(v.z); a3 += __uint_as_float(v.w);
                }
                acc[rt][mt][0] = a0; acc[rt][mt][1] = a1; acc[rt][mt][2] = a2; acc[rt][mt][3] = a3;
            }
    }
#pragma unroll
    for (int rt = 0; rt < RPW; ++rt) {
        const int row = nrow0 + rt * 8 + 2 * t;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int tk0 = mt * 16 + gid, tk1 = tk0 + 8;
            if (LO) {
                float* o = reinterpret_cast<float*>(Out);
                if (tk0 < M) *reinterpret_cast<float2*>(o + (u64)tk0 * ldo + row) = make_float2(acc[rt][mt][0], acc[rt][mt][1]);
                if (tk1 < M) *reinterpret_cast<float2*>(o + (u64)tk1 * ldo + row) = make_float2(acc[rt][mt][2], acc[rt][mt][3]);
            } else {
                u16* o = reinterpret_cast<u16*>(Out);
                if (tk0 < M) *reinterpret_cast<u32*>(o + (u64)tk0 * ldo + row) = f2bf2(acc[rt][mt][0], acc[rt][mt][1]);
                if (tk1 < M) *reinterpret_cast<u32*>(o + (u64)tk1 * ldo + row) = f2bf2(acc[rt][mt][2], acc[rt][mt][3]);
            }
        }
    }
}
"""


def _gw_inst_names():
    return [f"w16_gemv<{mt},{rpw},{u},{xs},{lo}>" for mt, rpw, u, xs, lo in _GW_INST]


_GW_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <string>

#define GW_CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { cudaGetLastError(); \
        TORCH_CHECK(false, "gw: ", #call, " failed: ", cudaGetErrorName(e_), " (", cudaGetErrorString(e_), ")"); } } while (0)

template <int MT, int RPW, int U, int XS, int LO>
static void gw_launch(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, torch::Tensor ws, torch::Tensor done, int64_t M, int64_t N, int64_t K,
                      int64_t SPLIT, int64_t nw, int64_t ldx, int64_t ldo) {
    constexpr int NL = 1 + LO;
    constexpr int WN = U / (RPW * NL);
    const int64_t Kc = K / SPLIT;
    const size_t smem = XS ? (size_t)M * (size_t)(2 * Kc + 64) : 0;
    TORCH_CHECK(K % SPLIT == 0 && Kc % (32 * WN) == 0 && Kc % 64 == 0 && N % (8 * RPW * nw) == 0 && nw >= 1 && nw <= 8 && M >= 1 && M <= 16 * MT && M > 16 * (MT - 1),
                "gw: bad shape M=", M, " N=", N, " K=", K, " SPLIT=", SPLIT, " nw=", nw);
    TORCH_CHECK(out.scalar_type() == (LO ? torch::kFloat32 : torch::kBFloat16), "gw: output dtype");
    const unsigned gx = (unsigned)(N / (8 * RPW * nw));
    TORCH_CHECK(SPLIT == 1 || (gx <= (unsigned)done.numel() && (size_t)gx * SPLIT * nw * RPW * MT * 32 <= (size_t)ws.size(0)), "gw: workspace");
    auto fn = w16_gemv<MT, RPW, U, XS, LO>;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(gx, (unsigned)SPLIT, 1);
    cfg.blockDim = dim3((unsigned)(nw * 32), 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    const u16* a_x = reinterpret_cast<const u16*>(x.data_ptr());
    const u16* a_w = reinterpret_cast<const u16*>(w.data_ptr());
    const u16* a_wl = LO ? reinterpret_cast<const u16*>(wlo.data_ptr()) : a_w;
    void* a_out = out.data_ptr();
    float4* a_ws = reinterpret_cast<float4*>(ws.data_ptr());
    int* a_done = done.data_ptr<int>();
    int i_M = (int)M, i_N = (int)N, i_K = (int)K, i_S = (int)SPLIT, i_lx = (int)ldx, i_lo = (int)ldo;
    void* args[] = {&a_x, &a_w, &a_wl, &a_out, &a_ws, &a_done, &i_M, &i_N, &i_K, &i_S, &i_lx, &i_lo};
    const cudaError_t e = cudaLaunchKernelExC(&cfg, (const void*)fn, args);
    if (e != cudaSuccess) {
        cudaGetLastError();
        TORCH_CHECK(false, "gw: launch failed: ", cudaGetErrorName(e), " (", cudaGetErrorString(e), ") inst=", MT, RPW, U, XS, LO, " grid=", gx, "x", SPLIT, " nw=", nw, " smem=", smem);
    }
}

void gw_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, torch::Tensor ws, torch::Tensor done, int64_t M, int64_t N, int64_t K,
             int64_t SPLIT, int64_t nw, int64_t inst, int64_t ldx, int64_t ldo) {
    c10::cuda::CUDAGuard guard(x.device());
    int64_t i = 0;
#define GW_CASE(MT, RPW, U, XS, LO) if (inst == i++) { gw_launch<MT, RPW, U, XS, LO>(x, w, wlo, out, ws, done, M, N, K, SPLIT, nw, ldx, ldo); return; }
    GW_CASE(1, 1, 8, 1, 0) GW_CASE(1, 1, 16, 1, 0) GW_CASE(1, 2, 16, 1, 0) GW_CASE(1, 1, 16, 0, 0) GW_CASE(2, 1, 16, 1, 0) GW_CASE(2, 1, 16, 0, 0) GW_CASE(2, 2, 16, 0, 0)
    GW_CASE(1, 1, 16, 1, 1) GW_CASE(1, 2, 16, 1, 1) GW_CASE(2, 1, 16, 0, 1)
#undef GW_CASE
    TORCH_CHECK(false, "gw: no instantiation ", inst);
}

template <int MT, int RPW, int U, int XS, int LO>
static void gw_init_one(const char* name, int optin, std::string& out) {
    const void* fn = (const void*)w16_gemv<MT, RPW, U, XS, LO>;
    cudaFuncAttributes a;
    GW_CK(cudaFuncGetAttributes(&a, fn));
    const int dyn = optin - (int)a.sharedSizeBytes;     // static + dynamic must fit the opt-in limit
    GW_CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
    GW_CK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
    out += std::string(name) + " regs=" + std::to_string(a.numRegs) + " local=" + std::to_string(a.localSizeBytes) + " static=" + std::to_string(a.sharedSizeBytes) + " maxDynamic=" + std::to_string(dyn) + "\n";
}

// every instantiation: attributes (checked), one diagnostic line each
std::string gw_init() {
    std::string out;
    int dev = 0, optin = 0;
    GW_CK(cudaGetDevice(&dev));
    GW_CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    gw_init_one<1, 1, 8, 1, 0>("w16_gemv<1,1,8,1,0>", optin, out);
    gw_init_one<1, 1, 16, 1, 0>("w16_gemv<1,1,16,1,0>", optin, out);
    gw_init_one<1, 2, 16, 1, 0>("w16_gemv<1,2,16,1,0>", optin, out);
    gw_init_one<1, 1, 16, 0, 0>("w16_gemv<1,1,16,0,0>", optin, out);
    gw_init_one<2, 1, 16, 1, 0>("w16_gemv<2,1,16,1,0>", optin, out);
    gw_init_one<2, 1, 16, 0, 0>("w16_gemv<2,1,16,0,0>", optin, out);
    gw_init_one<2, 2, 16, 0, 0>("w16_gemv<2,2,16,0,0>", optin, out);
    gw_init_one<1, 1, 16, 1, 1>("w16_gemv<1,1,16,1,1>", optin, out);
    gw_init_one<1, 2, 16, 1, 1>("w16_gemv<1,2,16,1,1>", optin, out);
    gw_init_one<2, 1, 16, 0, 1>("w16_gemv<2,1,16,0,1>", optin, out);
    return out;
}

// pre-flight: CTAs per SM of an instantiation with `nw` warps and `smem` bytes of dynamic shared memory (<= 0: the launch would fail)
int64_t gw_occupancy(int64_t inst, int64_t nw, int64_t smem) {
    int64_t i = 0;
    int n = 0;
    cudaError_t e = cudaSuccess;
#define GW_OCC(MT, RPW, U, XS, LO) if (inst == i++) { e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void*)w16_gemv<MT, RPW, U, XS, LO>, (int)(nw * 32), (size_t)smem); \
        if (e != cudaSuccess) { cudaGetLastError(); return -1; } return n; }
    GW_OCC(1, 1, 8, 1, 0) GW_OCC(1, 1, 16, 1, 0) GW_OCC(1, 2, 16, 1, 0) GW_OCC(1, 1, 16, 0, 0) GW_OCC(2, 1, 16, 1, 0) GW_OCC(2, 1, 16, 0, 0) GW_OCC(2, 2, 16, 0, 0)
    GW_OCC(1, 1, 16, 1, 1) GW_OCC(1, 2, 16, 1, 1) GW_OCC(2, 1, 16, 0, 1)
#undef GW_OCC
    return -2;
}
"""
_GW_CPP_SRC = ("void gw_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, torch::Tensor ws, torch::Tensor done, int64_t M, int64_t N, int64_t K, "
               "int64_t SPLIT, int64_t nw, int64_t inst, int64_t ldx, int64_t ldo);\n"
               "std::string gw_init();\n"
               "int64_t gw_occupancy(int64_t inst, int64_t nw, int64_t smem);\n")
_GW_EXT = {"ext": None, "err": None, "secs": None, "make": None, "info": ""}      # make: tests inject an NVRTC extension factory


def _gw_build_ext():
    """Builds the extension once (10 instantiations, `-w`, ptxas -v in the log); returns it or None (the study then reports the cuBLAS / board lines only)."""
    if _GW_EXT["ext"] is not None or _GW_EXT["err"] is not None:
        return _GW_EXT["ext"]
    t0 = _gw_time.perf_counter()
    try:
        if _GW_EXT["make"] is not None:
            _GW_EXT["ext"] = _GW_EXT["make"](_GW_DEVICE_SRC)
        else:
            import hashlib

            from torch.utils.cpp_extension import load_inline

            src = _GW_DEVICE_SRC + _GW_HOST_SRC
            tag = hashlib.sha1(src.encode()).hexdigest()[:12]
            _GW_EXT["ext"] = load_inline(name=f"okc_gw_{tag}", cpp_sources=_GW_CPP_SRC, cuda_sources=src, functions=["gw_gemv", "gw_init", "gw_occupancy"],
                                         extra_cuda_cflags=["-O3", "-Xptxas", "-v", "-w"], verbose=True)
        info = _GW_EXT["ext"].gw_init()
        _GW_EXT["info"] = info if isinstance(info, str) else ""
        if _GW_EXT["info"]:
            print("MEAS-GW attributes:\n" + _GW_EXT["info"].rstrip("\n"), flush=True)
    except Exception as ex:  # noqa: BLE001
        _GW_EXT["ext"] = None
        _GW_EXT["err"] = _gw_where(ex)
        _gw_note(f"extension unavailable ({_GW_EXT['err']}): only the cuBLAS / board lines")
    _GW_EXT["secs"] = _gw_time.perf_counter() - t0
    return _GW_EXT["ext"]


# ================================================================================================================ candidates
def _gw_wn(inst):
    mt, rpw, u, xs, lo = _GW_INST[inst]
    return u // (rpw * (1 + lo))


def _gw_smem(inst, m, k, split):
    return m * (2 * (k // split) + 64) if _GW_INST[inst][3] else 0


def _gw_fits(inst, layer, m, n, k, nw, split):
    """Shape-only legality of a candidate (what the host launcher asserts)."""
    mt, rpw, u, xs, lo = _GW_INST[inst]
    if (lo == 1) != (layer == "router") or mt != (1 if m <= 16 else 2):
        return False
    wn = _gw_wn(inst)
    kc = k // split
    if k % split or kc % (32 * wn) or kc % 64 or n % (8 * rpw * nw):
        return False
    if _gw_smem(inst, m, k, split) > _GW_SMEM_CAP:
        return False
    gx = n // (8 * rpw * nw)
    return gx * split * nw * rpw * mt * 32 <= _GW_WS_FLOAT4 and gx <= _GW_DONE


def _gw_candidates(layer, m, n, k):
    """[(name, inst, nw, split)] legal for the cell, the _GW_MAX_CAND whose CTA count is closest to 4 CTAs / SM (528 on 132 SMs)."""
    import math
    out = []
    for inst in range(len(_GW_INST)):
        mt, rpw, u, xs, lo = _GW_INST[inst]
        for nw in _GW_NW:
            for split in _GW_SPLITS:
                if _gw_fits(inst, layer, m, n, k, nw, split):
                    ctas = n // (8 * rpw * nw) * split
                    out.append((abs(math.log(ctas / 528.0)), f"ws.{mt}{rpw}{u}{xs}.w{nw}.s{split}", inst, nw, split))
    out.sort()
    return [(name, inst, nw, split) for _d, name, inst, nw, split in out[:_GW_MAX_CAND]]


def _gw_run(ext, state, layer, m, n, inst, nw, split):
    """-> fn(x) running the candidate (the output is allocated by the call; the workspace / counters are made once in prepare)."""
    lo = layer == "router"
    w = state["router_hi"] if lo else state[layer]
    wlo = state["router_lo"] if lo else w

    def fn(x):
        out = torch.empty((m, n), device=x.device, dtype=torch.float32 if lo else torch.bfloat16)
        ext.gw_gemv(x, w, wlo, out, state["gw_ws"], state["gw_done"], m, n, w.shape[1], split, nw, inst, x.stride(0), out.stride(0))
        return out
    return fn


# ================================================================================================================ timing and checks
class _GwEnv:
    def __init__(self, dev):
        self.dev = dev
        self.flush = torch.ones(_GW_FLUSH_MB * 1024 * 1024 // 4, device=dev, dtype=torch.float32)
        self.sink = torch.zeros((), device=dev, dtype=torch.float32)

    def replay_us(self, g):
        torch.sum(self.flush, dim=0, out=self.sink)   # read-only: 128 MB through L2, like the judge's flush
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) * 1000.0

    def capture(self, fn, x):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(x)
            fn(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fn(x)
        return g, y

    def overhead(self, reps=21):
        """The fixed cost of a graph replay (one tiny kernel), which the judge subtracts (about 4 us). -> raw median us."""
        buf = torch.zeros(4, device=self.dev, dtype=torch.int32)
        g, _ = self.capture(lambda x: (buf.add_(1), buf)[1], buf)
        return _gw_med([self.replay_us(g) for _ in range(reps)])


def _gw_check(y, ref, x, layer):
    """The organisers' bound (checks.compare + probe rows), re-implemented. -> (ok, share of the tightest bound used, why, relative L2)."""
    tol = _GW_TOL.get(layer, 1e-2)
    if y.shape != ref.shape or y.dtype != ref.dtype:
        return False, 99.0, f"shape/dtype {tuple(y.shape)} {y.dtype}", 1.0
    out, r = y.float(), ref.float()
    if not bool(torch.isfinite(out).all()):
        return False, 99.0, "nan/inf", 1.0
    diff = out - r
    err = float(diff.norm() / r.norm().clamp(min=1e-12))
    rows_d, rows_r = diff.norm(dim=-1), r.norm(dim=-1)
    floor = (_GW_ROW_FLOOR * rows_r.mean()).clamp(min=1e-12)
    worst = float((rows_d / torch.maximum(rows_r, floor)).max())
    probes = ((x != 0).sum(-1) == 1).nonzero().flatten()
    perr = 0.0
    if len(probes):
        pd = (y[probes].double() - ref[probes].double()).norm(dim=-1) / ref[probes].double().norm(dim=-1).clamp(min=1e-30)
        perr = float(pd.max())
    use = max(err / tol, worst / (_GW_ROW_FACTOR * tol), perr / _GW_PROBE_TOL)
    return use <= 1.0, use, f"err={err:.2e} row={worst:.2e} probe={perr:.2e}", err


def _gw_ref(w, layer, n, x):
    ref = x.double() @ w[:n].double().T
    return ref.to(torch.float32 if layer == "router" else torch.bfloat16)


def _gw_unique_bytes(layer, m, n, k):
    wb = 4 if layer == "router" else 2
    return n * k * wb + m * k * 2 + m * n * wb


def _gw_cell_variants(layer, m, n, state, ext):
    """-> [(name, kind, fn(x))]; kind board | ref | ws. The board kernel first (the pivot)."""
    cfg = {"m": m, "n": n, "k": _gw_nk(layer, m)[1], "layer": layer}
    out = [("board", "board", lambda x: kernel_fn({"x": x, "config": cfg}, state))]
    w = state[layer]
    if layer == "router":
        out.append(("cublas", "ref", lambda x: _baseline(x, state, layer, m, n)))
    else:
        out.append(("cublas", "ref", lambda x: torch.matmul(x, w[:n].t())))
    if ext is not None:
        for name, inst, nw, split in _gw_candidates(layer, m, n, cfg["k"]):
            smem = _gw_smem(inst, m, cfg["k"], split)
            try:
                occ = int(ext.gw_occupancy(inst, nw, smem))
            except Exception:  # noqa: BLE001
                occ = 1
            if occ < 1:
                continue                                       # pre-flight: never launch what cannot run (a failed launch poisons the context)
            out.append((name, "ws", _gw_run(ext, state, layer, m, n, inst, nw, split)))
    return out


def _gw_cell(env, layer, m, state, ext, ovh, deadline):
    """-> dict name -> {kind, us (net, median over seeds), xs (per-seed board/variant), ok, use, why, err, status}."""
    n, k = _gw_nk(layer, m)
    seeds = [_tr_generate_input(m, layer, s, "normal", "cuda") for s in _GW_SEEDS]
    xs = [d["x"].contiguous() for d in seeds]
    refs = [_gw_ref(state[layer], layer, n, x) for x in xs]
    x = xs[0].clone()
    res, graphs = {}, []
    for name, kind, fn in _gw_cell_variants(layer, m, n, state, ext):
        if name != "board" and _gw_elapsed() > deadline:
            res[name] = dict(kind=kind, status="skip")
            continue
        try:
            g, y = env.capture(fn, x)
            graphs.append((name, kind, g, y))
        except Exception as ex:  # noqa: BLE001
            res[name] = dict(kind=kind, status="fail " + _gw_where(ex))
            if name == "board":
                raise
    per_seed = {name: [] for name, _, _, _ in graphs}
    okv = {name: True for name, _, _, _ in graphs}
    bad = {name: (0.0, "", 0.0) for name, _, _, _ in graphs}
    for si in range(len(_GW_SEEDS)):
        x.copy_(xs[si])
        samples = {name: [] for name, _, _, _ in graphs}
        for _ in range(_GW_REPS):
            for name, _, g, _ in graphs:             # board first in every round: the pivot
                samples[name].append(env.replay_us(g))
        for name, kind, g, y in graphs:
            per_seed[name].append(_gw_med(samples[name]))
            ok, use, why, err = _gw_check(y, refs[si], x, layer)
            okv[name] = okv[name] and ok
            if use >= bad[name][0]:
                bad[name] = (use, why, err)
    board = per_seed["board"]
    for name, kind, g, y in graphs:
        net = [max(0.5, v - ovh) for v in per_seed[name]]
        bnet = [max(0.5, v - ovh) for v in board]
        res[name] = dict(kind=kind, us=_gw_med(net), xs=[b / v for b, v in zip(bnet, net)], ok=okv[name], use=bad[name][0], why=bad[name][1], err=bad[name][2], status="ok")
    del graphs
    return res


# ================================================================================================================ report
def _gw_best(r):
    c = [(v["us"], name) for name, v in r.items() if v.get("kind") == "ws" and v.get("status") == "ok" and v["ok"]]
    return min(c) if c else None


def _gw_render(results, ovh, dev_name, t_total, n_build):
    lines = [f"S gemv ws v1: {dev_name} torch={torch.__version__} inst={len(_GW_INST)} build={'-' if n_build is None else '%.0fs' % n_build} replay_ovh={ovh:.2f}us seeds={len(_GW_SEEDS)}x{_GW_REPS} "
             f"flush={_GW_FLUSH_MB}MB t={t_total:.0f}s"]
    lines.append("S us = graph replay minus replay_ovh (median of seeds of medians); cu = torch.matmul (router: the baseline path); board = director_v2's _PLAN cell; ws = the best correct WS variant "
                 "(name ws.<MT><RPW><U><XS>.w<warps>.s<split>); TB/s on unique bytes (W + x + out); err = relative L2 vs FP64 (tol 1e-2, router 1e-4)")
    wins, nbad = [], 0
    for layer in _GW_LAYERS:
        for m in _GW_MS:
            r = results.get((layer, m))
            if not r:
                lines.append(f"S {layer} m={m}: not measured")
                continue
            n, k = _gw_nk(layer, m)
            b, cu = r.get("board", {}), r.get("cublas", {})
            bn = _gw_best(r)
            nbad += sum(1 for v in r.values() if v.get("kind") == "ws" and v.get("status") == "ok" and not v["ok"])
            nb = sum(1 for v in r.values() if v.get("kind") == "ws" and v.get("status") == "ok" and not v["ok"])
            head = f"S {layer:8s} m={m:<2d} cu={cu.get('us', 0):.2f} board={b.get('us', 0):.2f}"
            if bn is None:
                lines.append(head + " ws=none (no correct WS variant)")
                continue
            v = r[bn[1]]
            ub = _gw_unique_bytes(layer, m, n, k)
            tbs = "/".join(f"{ub / (u * 1e-6) / 1e12:.2f}" for u in (cu.get("us", 0) or 1e9, b.get("us", 0) or 1e9, v["us"]))
            x = _gw_med(v["xs"])
            lines.append(head + f" ws={v['us']:.2f} {bn[1]} x_board={x:.2f} x_cu={cu.get('us', 0) / v['us']:.2f} TB/s(cu/board/ws)={tbs} err={v['err']:.1e} OK bad={nb}")
            if x >= 1.0 + _GW_WIN_PCT / 100.0:
                wins.append(f"{layer}.{m}({x:.2f})")
    lines.append(f"D gemv ws: beats the board by >= {_GW_WIN_PCT:.0f}% on {len(wins)} of {len(_GW_LAYERS) * len(_GW_MS)} cells: " + (", ".join(wins) if wins else "none")
                 + f" | {nbad} WS variants outside the correctness bound (BAD in the B lines)")
    for note in _GW_NOTES:
        lines.append("S note: " + note)
    body = []
    for layer in _GW_LAYERS:
        for m in _GW_MS:
            r = results.get((layer, m))
            if not r:
                continue
            ws = sorted((nm for nm, v in r.items() if v.get("kind") == "ws" and v.get("status") == "ok"), key=lambda nm: r[nm]["us"])
            for name in ["board", "cublas"] + ws[:4]:     # the board, cuBLAS and the best four WS variants of the cell
                v = r.get(name)
                if not v:
                    continue
                if v.get("status") != "ok":
                    body.append(f"B gemv {layer} m={m} | {name} {v['status'][:50]}")
                    continue
                s = f"B gemv {layer} m={m} | {name} us={v['us']:.2f} x={_gw_med(v['xs']):.2f}"
                if not v["ok"]:
                    s += f" BAD({v['use']:.1f})"
                body.append(s)
    out = "\n".join(lines + body)
    if len(out.encode()) > _GW_MAX_BYTES:
        out = out[:_GW_MAX_BYTES - 40] + "\n...(report cut at budget)"
    return out


# ================================================================================================================ main
def _gw_main(state, env=None):
    if not torch.cuda.is_available():
        print("MEAS-GW skip: no CUDA")
        return None
    dev = state["router"].device
    env = env or _GwEnv(dev)
    ext = _gw_build_ext()
    state = dict(state)
    state["gw_ws"] = torch.empty((_GW_WS_FLOAT4, 4), device=dev, dtype=torch.float32)
    state["gw_done"] = torch.zeros(_GW_DONE, device=dev, dtype=torch.int32)
    ovh = env.overhead()
    if not (1.0 < ovh < 8.0):
        _gw_note(f"replay overhead {ovh:.2f} us outside 1..8: x values unreliable")
    print(f"MEAS-GW start t={_gw_elapsed():.0f}s replay_ovh={ovh:.2f}us", flush=True)
    results = {}
    order = [(layer, m) for m in _GW_MS for layer in _GW_LAYERS]
    try:
        for layer, m in order:
            if _gw_elapsed() > _GW_BUDGET_S - 6:
                _gw_note(f"{layer} m={m} dropped:budget")
                continue
            try:
                results[(layer, m)] = _gw_cell(env, layer, m, state, ext, ovh, _GW_BUDGET_S - 10)
            except Exception as ex:  # noqa: BLE001
                _gw_note(f"{layer} m={m} failed: {_gw_where(ex)}")
            print(f"MEAS-GW p {layer}.{m} {_gw_elapsed():.0f}s", flush=True)
    finally:
        print(_gw_render(results, ovh, "_".join(torch.cuda.get_device_name().split()), _gw_elapsed(), _GW_EXT["secs"]), flush=True)
    return results


_gw_orig_prepare = prepare


def prepare(weights):  # noqa: F811
    global _GW_T0
    _GW_T0 = _gw_time.perf_counter()
    state = _gw_orig_prepare(weights)
    try:
        _gw_main(state)
    except Exception as ex:  # noqa: BLE001
        print(f"MEAS-GW failed: {_gw_where(ex)}", flush=True)
    return state
