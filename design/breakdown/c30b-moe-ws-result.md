# C30b result: v7 warp-specialised MoE on the H200 (job c0b3b88b859d4d62, 2026-10-26 10:18Z)
Launch fix works (preflight ok on all buckets; v7 bitwise vs v5 on every bucket). Speed: SLOWER than v5 everywhere.
| bs | v5 g1 | d1s8 g1 | d2s5 g1 | nocomp (D1) | nocomp2 (D2) | consonly / 2 | wstream |
|---|---|---|---|---|---|---|---|
| 256 | 235 | 695–754 | 397–431 | 597–649 | 319–346 | 717–779 / 411–445 | 114–127 |
| 128 | 180–211 | 568–628 | 326–361 | 491–542 | 263–292 | | 103–116 |
| 32 | 87–111 | 274–340 | 159–196 | | 129–159 | | 53–69 |
| 16 | 67–85 | 211–275 | 123–158 | | 100–126 | | 45–58 |
Stamps: consumers wait on `full` 42–64 % of the time, producer waits on `empty` 3–5 % → the PRODUCER binds.
Reading (fable-10 D3 rule iv: NOCOMPUTE ≪ 3.2 TB/s → fix the producer, no verdict on the design):
- Producer-only (nocomp) is as slow as the full kernel, and halving the number of copies (D2: 1-KB copies instead of 512 B) makes it
  ~2× faster → time ∝ number of bulk-copy instructions. Each stage issues 24 word + 24 scale + up to 32 act-row copies from lane-varying
  addresses; SASS turns each into an ELECT/R2UR/UBLKCP loop (~10 instr, serialised per lane) → ~80 serialized copies per stage.
- CONSONLY (fixed L2-hot source) is no faster → not DRAM, the copy-issue path itself.
Fix (C30c): ≤ 3 bulk copies per stage: (1) repack weights+scales in prepare() so one stage of an item (24 groups × D k64) is ONE contiguous
block; (2) gather the chunk's activation rows inside the timed call (route/pre-pass) into a k-tiled contiguous buffer so one stage's act
slice is ONE contiguous block (or one cp.async.bulk.tensor.2d). Alternative per D3: 4 cp.async producer warps.
