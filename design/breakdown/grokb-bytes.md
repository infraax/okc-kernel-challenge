I'll read the problem specs, breakdown tables, and board kernels, then build the bytes/FLOPS model from those numbers only.The breakdown tables name the families. Next I’ll pull the stored shapes and dtypes from the board kernels so the byte and FLOP counts stay tied to those dims.Prefill has no judge table in `design/breakdown/`, so it has no rows. Every scored dim below is taken from `TASK.md` or from the `Weights` / `Config` classes those files point at (and the same comments in the four board kernels). Nothing numeric was filled in by guesswork.

## Constants

| item | value |
|---|---|
| BF16 / FP32 / FP8 | 2 / 4 / 1 byte |
| Layer hidden H | 8192 |
| MoE width H = I | 4096 |
| Heads | Hq=16, Hkv=2, d=64 (TP8) |
| QKV width | (2·16+2·2)·64 = 2304 |
| Attended KV | sliding **2048** tokens; full **ctx** |
| Agentic full ctx, minimum | 80000·(1−0.075) = **74000** (judge `_shortest_share`; the draw is uniform in ±7.5%, so the mean is 80000) |
| KV bytes / token | K and V, FP8: 2·2·64 = **256** |
| One local expert, NVFP4 | w13 packed 8192·2048 + w13 scales FP8 8192·256 + w2 packed 4096·2048 + w2 scales 4096·256 + 2·FP32 global scales = **28,311,560** |
| Projections | qkv `[2304,8192]` BF16, o `[8192,1024]` BF16, pooler `[4096,8192]` BF16, unpooler `[8192,4096]` BF16 (N=**1024** when tokens ≤ 8), router `[256,8192]` FP32 |

GEMM bytes = N·K·wbytes + m·K·2 + m·N·wbytes. GEMM FLOPs = 2·m·N·K (one matmul; router hi+lo is the same 4 bytes/weight as FP32). Attention FLOPs = 4·B·S·L·Hq·d (QK and AV). MoE FLOPs = 2·routes·3·4096·4096 = routes·100,663,296. Split-K partials, SiLU workspace, and the layernorm the judge truncates to `vectorized_layer_...` (so `family()` returns `other`) are not compulsory HBM.

## Distinct local experts

Scored MoE shapes do not set `dist=skewed`, so the generator default is `normal`: uniform scores plus a Student-t (2 dof) popularity of scale 0.03, top-8 of 256, then rotate so the busiest group of 32 is local. `reference.speed_of_light_ns` sets `touched = 0.75 · router_distinct` and the published floors are bandwidth-bound (they flatten from bs 128 → 256, and sliding vs full floors differ only by the KV term). Inverting `floor_µs · 4.8e12` through that formula:

| problem | shape | T or bs | touched = 0.75·E | **E[distinct]** |
|---|---|---|---|---|
| moe | 0 | 1 | 1.93 | **2.58** |
| moe | 3 | 8 | 6.32 | **8.43** |
| moe | 1 | 16 | 8.89 | **11.86** |
| moe | 2 | 32 | 11.99 | **15.98** |
| moe | 4 | 128 | 17.10 | **22.80** |
| moe | 5 | 256 | 18.32 | **24.42** |
| decode | 0–1 | 1 | 1.87 | **2.50** |
| decode | 6–7 | 8 | 5.62 | **7.50** |
| decode | 2–3 | 16 | 5.6–5.7 | **7.5** |
| decode | 4–5 | 32 | 6.7–6.8 | **9.0** |
| decode | 8–9 | 128 | 13.8–13.9 | **18.4** |
| decode | 10–11 | 256 | 15.9–16.0 | **21.3** |

Bytes below use **E**, not the judge’s 0.75 haircut. A 3-trial CPU sample in `experiments/routing_hint_t1.txt` lands in the same band (decode T=1 → 2.3–3.0, T=256 → ~17.5).

**Routes (for FLOPs).** At 1 token, routes = distinct. For `muse_moe_w4a16`, one token drops Hypergeometric(N=256, K=32, n=8) hits on a shard (mean 1, variance 0.851); the busiest of 8 shards is about `M + 1.2·√(0.851·M)`: **2.58, 11.13, 20.43, 38.26, 140.5, 273.7** at bs 1, 8, 16, 32, 128, 256. Popularity pulls distinct below a pure-uniform occupancy (24 not 32 at bs 256), so the hot shard’s route count can sit a bit above this. For `muse_decode_layer` the router is mean-aligned (`ROUTER_ALIGN=0.07`); per-token local hits are anchored at the T=1 value 2.50 and held flat: **routes = 2.50·T**.

## Per family

**dense_gemm / dense_gemm(cublas)** — one projection, weights once. On the layer, cuBLAS is qkv, o, unpooler, and pooler when T>32; Triton GEMV is the router, plus the pooler when T≤32. Layer qkv also reads the BF16 attn input `T·8192·2`. Router elementwise (the 0.7 µs kernel) adds no compulsory bytes.

**attention / other** — `other` is only the full-attention kernel in `muse_attention_decode` (`_full_cat_kernel` / `_full_kernel` do not contain the substring `attn`). Bytes = attended K+V plus the QKV input `(B·L·2304·2)` plus the new FP8 KV write on full layers (`B·L·256`; on sliding layers those tokens already sit inside the 2048). **attention_reduce** writes the BF16 output `B·L·1024·2` and does none of the matmul. Inside the layer, QKV and the attention output are intermediates, so attention keeps only the KV term.

**moe_gemm(marlin) / moe_gemm(cuda smallm)** — `E · 28,311,560` plus, on the MoE problem only, the BF16 input `M·4096·2`. **moe_epilogue** (small M) or **norm/residual** (the post-expert kernel; `family()` matches `norm` before `post_expert`) writes `M·4096·2` and reads the BF16 post-norm (8192) and the FP32 top-k weights. **activation**, **routing**, **other** on the big-M path are index/SiLU traffic: routing `M·64` bytes, other `M·64`, activation 0 (SiLU is fusable). SiLU FLOPs are `4·routes·4096`.

**norm/residual** on the layer — compulsory IO that is not the qkv input: FP32 stream in and out plus BF16 hidden out = `T·8192·10`, plus the 8192-byte post-norm vector. FLOPs `T·8192·16`.

```csv
problem,shape,kernel_family,bytes,flops,note
muse_attention_decode,0,attention,528896,8388608,"B1 L1 sliding S=2048 FP8 KV 256B/tok + qkv BF16 2304*2"
muse_attention_decode,0,attention_reduce,2048,0,"out BF16 [1,1,1024]; split-K partials not compulsory"
muse_attention_decode,1,other,20484864,327680000,"full fixed S=80000 KV + qkv + new KV write 256B"
muse_attention_decode,1,attention_reduce,2048,0,"out BF16"
muse_attention_decode,2,attention,8462336,134217728,"B16 L1 sliding S=2048"
muse_attention_decode,2,attention_reduce,32768,0,"out BF16"
muse_attention_decode,3,other,303181824,4849664000,"full agentic minimum S=74000 = 0.925*80000; mean would be 80000"
muse_attention_decode,3,attention_reduce,32768,0,"out BF16"
muse_attention_decode,4,attention,16924672,268435456,"B32 L1 sliding S=2048"
muse_attention_decode,4,attention_reduce,65536,0,"out BF16"
muse_attention_decode,5,other,606363648,9699328000,"B32 full agentic S=74000 + new KV"
muse_attention_decode,5,attention_reduce,65536,0,"out BF16"
muse_attention_decode,6,attention,561152,67108864,"B1 L8 sliding; union of the window is 2048+7, counted as 2048"
muse_attention_decode,6,attention_reduce,16384,0,"out BF16"
muse_attention_decode,7,other,20518912,2621440000,"B1 L8 full fixed S=80000 + new KV 8*256"
muse_attention_decode,7,attention_reduce,16384,0,"out BF16"
muse_attention_decode,8,attention,8978432,1073741824,"B16 L8 sliding"
muse_attention_decode,8,attention_reduce,262144,0,"out BF16"
muse_attention_decode,9,other,303726592,38797312000,"B16 L8 full agentic S=74000"
muse_attention_decode,9,attention_reduce,262144,0,"out BF16"
muse_attention_decode,10,attention,17956864,2147483648,"B32 L8 sliding"
muse_attention_decode,10,attention_reduce,524288,0,"out BF16"
muse_attention_decode,11,other,607453184,77594624000,"B32 L8 full agentic S=74000 + new KV 65536"
muse_attention_decode,11,attention_reduce,524288,0,"out BF16"
muse_bf16_proj,0,dense_gemm(cublas),37769728,37748736,"qkv m=1 N=2304 K=8192 BF16 W+X+Y"
muse_bf16_proj,1,dense_gemm,16795648,16777216,"o m=1 N=8192 K=1024"
muse_bf16_proj,2,dense_gemm,67133440,67108864,"pooler m=1 N=4096 K=8192"
muse_bf16_proj,3,dense_gemm(cublas),8398848,8388608,"unpooler m=1 slice N=1024 K=4096"
muse_bf16_proj,4,dense_gemm,8406016,4194304,"router m=1 N=256 K=8192 FP32 W and Y; X BF16"
muse_bf16_proj,4,elementwise,0,0,"split-K epilogue; no bytes beyond the GEMM"
muse_bf16_proj,5,dense_gemm(cublas),38084608,603979776,"qkv m=16"
muse_bf16_proj,6,dense_gemm,17072128,268435456,"o m=16"
muse_bf16_proj,7,dense_gemm,67502080,1073741824,"pooler m=16"
muse_bf16_proj,8,dense_gemm(cublas),67502080,1073741824,"unpooler m=16 full N=8192"
muse_bf16_proj,9,dense_gemm,8667136,67108864,"router m=16"
muse_bf16_proj,9,elementwise,0,0,"no extra compulsory traffic"
muse_bf16_proj,10,dense_gemm(cublas),38420480,1207959552,"qkv m=32"
muse_bf16_proj,11,dense_gemm,17367040,536870912,"o m=32"
muse_bf16_proj,12,dense_gemm,67895296,2147483648,"pooler m=32"
muse_bf16_proj,13,dense_gemm(cublas),67895296,2147483648,"unpooler m=32 full"
muse_bf16_proj,14,dense_gemm,8945664,134217728,"router m=32"
muse_bf16_proj,14,elementwise,0,0,"no extra compulsory traffic"
muse_bf16_proj,15,dense_gemm(cublas),37916672,301989888,"qkv m=8"
muse_bf16_proj,16,dense_gemm,16924672,134217728,"o m=8"
muse_bf16_proj,17,dense_gemm,67305472,536870912,"pooler m=8"
muse_bf16_proj,18,dense_gemm(cublas),8470528,67108864,"unpooler m=8 slice N=1024"
muse_bf16_proj,19,dense_gemm,8527872,33554432,"router m=8"
muse_bf16_proj,19,elementwise,0,0,"no extra compulsory traffic"
muse_bf16_proj,20,dense_gemm(cublas),40435712,4831838208,"qkv m=128"
muse_bf16_proj,21,dense_gemm(cublas),19136512,2147483648,"o m=128"
muse_bf16_proj,22,dense_gemm(cublas),70254592,8589934592,"pooler m=128"
muse_bf16_proj,23,dense_gemm(cublas),70254592,8589934592,"unpooler m=128 full"
muse_bf16_proj,24,dense_gemm,10616832,536870912,"router m=128"
muse_bf16_proj,24,elementwise,0,0,"no extra compulsory traffic"
muse_bf16_proj,25,dense_gemm(cublas),43122688,9663676416,"qkv m=256; FLOP-bound vs 989 TFLOP/s"
muse_bf16_proj,26,dense_gemm(cublas),21495808,4294967296,"o m=256"
muse_bf16_proj,27,dense_gemm(cublas),73400320,17179869184,"pooler m=256"
muse_bf16_proj,28,dense_gemm(cublas),73400320,17179869184,"unpooler m=256 full"
muse_bf16_proj,29,dense_gemm,12845056,1073741824,"router m=256"
muse_bf16_proj,29,elementwise,0,0,"no extra compulsory traffic"
muse_moe_w4a16,0,moe_gemm(cuda smallm),73052017,259711304,"E=2.58 experts * 28311560 + x BF16 [1,4096]; routes=2.58"
muse_moe_w4a16,0,moe_epilogue,16448,32768,"y BF16 [1,4096] + post_norm 8192 + topk fp32/ids"
muse_moe_w4a16,1,moe_gemm(cuda smallm),335906174,2056551137,"E=11.86; routes=20.43; x [16,4096]"
muse_moe_w4a16,1,moe_epilogue,140288,524288,"y + post_norm + topk"
muse_moe_w4a16,2,moe_gemm(marlin),452680873,3851377705,"E=15.98; routes=38.26; x [32,4096]"
muse_moe_w4a16,2,norm/residual,270336,1048576,"post_expert norm is norm/residual via family(); y+norm+topk"
muse_moe_w4a16,2,activation,0,627016,"SiLU fusable; 4*routes*4096"
muse_moe_w4a16,2,routing,1024,65536,"moe_align ids int32 [32,8]"
muse_moe_w4a16,2,other,2048,0,"local_ids + count/sort index traffic"
muse_moe_w4a16,3,moe_gemm(cuda smallm),238732107,1120382484,"E=8.43; routes=11.13; x [8,4096]"
muse_moe_w4a16,3,moe_epilogue,74240,262144,"y + post_norm + topk"
muse_moe_w4a16,4,moe_gemm(marlin),646552144,14145206354,"E=22.80; routes=140.5; x [128,4096]"
muse_moe_w4a16,4,norm/residual,1056768,4194304,"y + post_norm + topk"
muse_moe_w4a16,4,activation,0,2301952,"4*routes*4096"
muse_moe_w4a16,4,routing,4096,262144,"ids [128,8]"
muse_moe_w4a16,4,other,8192,0,"index kernels"
muse_moe_w4a16,5,moe_gemm(marlin),693465447,27552550748,"E=24.42; routes=273.7; x [256,4096]"
muse_moe_w4a16,5,norm/residual,2105344,8388608,"y + post_norm + topk"
muse_moe_w4a16,5,activation,0,4483891,"4*routes*4096"
muse_moe_w4a16,5,routing,8192,524288,"ids [256,8]"
muse_moe_w4a16,5,other,16384,0,"index kernels"
muse_decode_layer,0,dense_gemm(cublas),62930944,62914560,"T=1 qkv+o+unpooler[:1024] + attn_in BF16"
muse_decode_layer,0,dense_gemm,75497472,71303168,"pooler + router FP32; both Triton GEMV at T<=32"
muse_decode_layer,0,moe_gemm(marlin),70778900,251658240,"E=2.50 * 28311560; routes=2.50"
muse_decode_layer,0,attention,524288,8388608,"sliding KV 2048*256; qkv is an intermediate"
muse_decode_layer,0,norm/residual,90112,163840,"stream fp32 in+out + hidden bf16 out + post_norm"
muse_decode_layer,0,moe_epilogue,8224,32768,"post_norm + topk weights; y fused onward"
muse_decode_layer,0,routing,64,2048,"expert ids int32 [1,8] + topk weights"
muse_decode_layer,0,other,0,0,"truncated layernorm; its tensor is in norm/residual"
muse_decode_layer,1,dense_gemm(cublas),62930944,62914560,"same projections as shape 0"
muse_decode_layer,1,dense_gemm,75497472,71303168,"pooler + router"
muse_decode_layer,1,moe_gemm(marlin),70778900,251658240,"E=2.50; routes=2.50"
muse_decode_layer,1,attention,20480256,327680000,"full fixed KV 80000*256 + new KV 256"
muse_decode_layer,1,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,1,norm/residual,90112,163840,"stream + hidden out"
muse_decode_layer,1,moe_epilogue,8224,32768,"post_norm + topk"
muse_decode_layer,1,other,0,0,"truncated layernorm"
muse_decode_layer,2,dense_gemm(cublas),121896960,1946157056,"T=16 qkv+o+full unpooler + attn_in"
muse_decode_layer,2,dense_gemm,75497472,1140850688,"pooler + router"
muse_decode_layer,2,moe_gemm(marlin),214884740,4026531840,"E=7.59; routes=40"
muse_decode_layer,2,attention,8388608,134217728,"B16 sliding 2048*256"
muse_decode_layer,2,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,2,norm/residual,1318912,2621440,"stream + hidden out"
muse_decode_layer,2,moe_epilogue,8704,524288,"post_norm + topk"
muse_decode_layer,2,routing,1024,32768,"ids [16,8]"
muse_decode_layer,2,other,0,0,"truncated layernorm + index kernel bytes counted in routing"
muse_decode_layer,3,dense_gemm(cublas),121896960,1946157056,"T=16 projections"
muse_decode_layer,3,dense_gemm,75497472,1140850688,"pooler + router"
muse_decode_layer,3,moe_gemm(marlin),210921122,4026531840,"E=7.45; routes=40"
muse_decode_layer,3,attention,303108096,4849664000,"B16 full agentic S=74000 + new KV"
muse_decode_layer,3,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,3,norm/residual,1318912,2621440,"stream + hidden out"
muse_decode_layer,3,moe_epilogue,8704,524288,"post_norm + topk"
muse_decode_layer,3,routing,1024,32768,"ids [16,8]"
muse_decode_layer,3,other,0,0,"truncated layernorm"
muse_decode_layer,4,dense_gemm(cublas),122159104,3892314112,"T=32 qkv+o+full unpooler + attn_in"
muse_decode_layer,4,dense_gemm,75497472,2281701376,"pooler + router"
muse_decode_layer,4,moe_gemm(marlin),251123537,8053063680,"E=8.87; routes=80"
muse_decode_layer,4,attention,16777216,268435456,"B32 sliding"
muse_decode_layer,4,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,4,norm/residual,2629632,5242880,"includes post_expert kernel (family norm)"
muse_decode_layer,4,routing,2048,65536,"ids [32,8]"
muse_decode_layer,4,other,0,0,"truncated layernorm"
muse_decode_layer,5,dense_gemm(cublas),122159104,3892314112,"T=32 projections"
muse_decode_layer,5,dense_gemm,75497472,2281701376,"pooler + router"
muse_decode_layer,5,moe_gemm(marlin),256502734,8053063680,"E=9.06; routes=80"
muse_decode_layer,5,attention,606216192,9699328000,"B32 full agentic S=74000 + new KV"
muse_decode_layer,5,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,5,norm/residual,2629632,5242880,"stream + hidden + post_expert"
muse_decode_layer,5,routing,2048,65536,"ids [32,8]"
muse_decode_layer,5,elementwise,0,0,"cast; no extra compulsory bytes"
muse_decode_layer,5,other,0,0,"truncated layernorm"
muse_decode_layer,6,dense_gemm(cublas),63045632,503316480,"T=8 sliced unpooler N=1024"
muse_decode_layer,6,dense_gemm,75497472,570425344,"pooler + router"
muse_decode_layer,6,moe_gemm(marlin),212336700,2013265920,"E=7.50; routes=20"
muse_decode_layer,6,attention,524288,67108864,"B1 L8 sliding"
muse_decode_layer,6,norm/residual,663552,1310720,"stream + hidden out"
muse_decode_layer,6,moe_epilogue,8448,262144,"post_norm + topk"
muse_decode_layer,6,routing,512,16384,"ids [8,8]"
muse_decode_layer,6,other,0,0,"truncated layernorm"
muse_decode_layer,7,dense_gemm(cublas),63045632,503316480,"T=8 sliced unpooler"
muse_decode_layer,7,dense_gemm,75497472,570425344,"pooler + router"
muse_decode_layer,7,moe_gemm(marlin),212053584,2013265920,"E=7.49; routes=20"
muse_decode_layer,7,attention,20482048,2621440000,"full fixed S=80000 + new KV 2048"
muse_decode_layer,7,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,7,norm/residual,663552,1310720,"stream + hidden out"
muse_decode_layer,7,moe_epilogue,8448,262144,"post_norm + topk"
muse_decode_layer,7,other,0,0,"truncated layernorm"
muse_decode_layer,8,dense_gemm(cublas),190840832,24159191040,"T=128 qkv+o+pooler+full unpooler + attn_in"
muse_decode_layer,8,dense_gemm,8388608,536870912,"router only; pooler is cuBLAS above T=32"
muse_decode_layer,8,moe_gemm(marlin),519517126,32212254720,"E=18.35; routes=320"
muse_decode_layer,8,attention,8388608,1073741824,"B16 L8 sliding"
muse_decode_layer,8,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,8,norm/residual,10493952,20971520,"stream + hidden + post_expert"
muse_decode_layer,8,activation,0,5242880,"4*routes*4096; fusable"
muse_decode_layer,8,other,0,0,"truncated layernorm"
muse_decode_layer,9,dense_gemm(cublas),190840832,24159191040,"T=128 projections"
muse_decode_layer,9,dense_gemm,8388608,536870912,"router GEMV"
muse_decode_layer,9,moe_gemm(marlin),524046976,32212254720,"E=18.51; routes=320"
muse_decode_layer,9,attention,303136768,38797312000,"B16 L8 full agentic S=74000 + new KV"
muse_decode_layer,9,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,9,norm/residual,10493952,20971520,"stream + hidden + post_expert"
muse_decode_layer,9,activation,0,5242880,"4*320*4096"
muse_decode_layer,9,other,0,0,"truncated layernorm"
muse_decode_layer,10,dense_gemm(cublas),192937984,48318382080,"T=256 projections + attn_in"
muse_decode_layer,10,dense_gemm,8388608,1073741824,"router GEMV"
muse_decode_layer,10,moe_gemm(marlin),601620650,64424509440,"E=21.25; routes=640"
muse_decode_layer,10,attention,16777216,2147483648,"B32 L8 sliding"
muse_decode_layer,10,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,10,norm/residual,20979712,41943040,"stream + hidden + post_expert"
muse_decode_layer,10,activation,0,10485760,"4*640*4096"
muse_decode_layer,10,other,0,0,"truncated layernorm"
muse_decode_layer,11,dense_gemm(cublas),192937984,48318382080,"T=256 projections + attn_in"
muse_decode_layer,11,dense_gemm,8388608,1073741824,"router GEMV; table shows 42.7 us on this shape"
muse_decode_layer,11,moe_gemm(marlin),605301153,64424509440,"E=21.38; routes=640"
muse_decode_layer,11,attention,606273536,77594624000,"B32 L8 full agentic S=74000 + new KV"
muse_decode_layer,11,attention_reduce,0,0,"partials not compulsory"
muse_decode_layer,11,norm/residual,20979712,41943040,"stream + hidden + post_expert"
muse_decode_layer,11,activation,0,10485760,"4*640*4096"
muse_decode_layer,11,other,0,0,"truncated layernorm"
```

`muse_prefill_layer` is not in `design/breakdown/`. No judged shape index, so no rows. Its TASK shapes are the same layer at prefill `(batch, ctx, q_len)`.

## Furthest above bytes / 4.0 TB/s

Score = shape weight × (family GPU µs − bytes/4.0e12). Measured µs is the sum of that family’s rows in the breakdown table. This is an absolute gap, not a ratio: sliding attention is ~30× its byte floor but only ~5 µs, so it does not place.

| rank | pair | measured µs | byte floor µs | weight × gap |
|---|---|---|---|---|
| 1 | muse_moe_w4a16 #5 moe_gemm(marlin) | 634.7 | 173.4 | 27680 |
| 2 | muse_decode_layer #10 moe_gemm(marlin) | 591.9 | 150.4 | 20309 |
| 3 | muse_moe_w4a16 #4 moe_gemm(marlin) | 403.6 | 161.6 | 14518 |
| 4 | muse_decode_layer #8 moe_gemm(marlin) | 403.8 | 129.9 | 12600 |
| 5 | muse_decode_layer #11 moe_gemm(marlin) | 635.8 | 151.3 | 7752 |
| 6 | muse_moe_w4a16 #2 moe_gemm(marlin) | 211.6 | 113.2 | 5906 |
| 7 | muse_decode_layer #9 moe_gemm(marlin) | 351.3 | 131.0 | 3525 |
| 8 | muse_decode_layer #4 moe_gemm(marlin) | 135.6 | 62.8 | 3350 |
| 9 | muse_attention_decode #11 other | 358.3 | 151.9 | 3303 |
| 10 | muse_decode_layer #11 attention | 342.7 | 151.6 | 3058 |

Marlin on the large verify shapes is the gap: tens of experts of NVFP4 (~15–24 × 28.3 MB) at a measured rate well under 4 TB/s. Next would be decode #2 Marlin (2851) and moe #1 small-M (2797). The worst **ratios** are the sliding decode kernels (attention #0: 4.7 µs against a 0.13 µs floor), which this weight × gap ranking does not select.