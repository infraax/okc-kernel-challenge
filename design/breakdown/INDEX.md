# Bottleneck index (from judge profiles; regenerate: tools/breakdown.py)

| problem | job | speedup | worst shapes by loss | top families by weighted loss |
|---|---|---|---|---|
| [muse_attention_decode](muse_attention_decode.md) | 4b047c1bf5354162 | 1.689 | #8 6.9× sol; #10 6.8× sol; #2 5.5× sol | attention 75%, attention_reduce 25% |
| [muse_bf16_proj](muse_bf16_proj.md) | c3b7c70b5edd43c5 | 1.156 | #29 4.6× sol; #24 3.9× sol; #9 3.0× sol | dense_gemm 50%, dense_gemm(cublas) 47%, elementwise 4% |
| [muse_decode_layer](muse_decode_layer.md) | 0108f70b516d4ea7 | 1.226 | #10 4.7× sol; #8 4.0× sol; #0 3.1× sol | moe_gemm(marlin) 36%, moe_gemm(cuda smallm) 21%, dense_gemm(cublas) 15% |
| [muse_moe_w4a16](muse_moe_w4a16.md) | 76d9f4196733442d | 1.436 | #5 4.0× sol; #0 3.6× sol; #4 3.1× sol | moe_gemm(cuda largem) 55%, moe_gemm(cuda smallm) 39%, moe_epilogue 4% |
